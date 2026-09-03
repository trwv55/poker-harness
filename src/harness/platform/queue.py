"""Очередь `jobs` (спека §8): захват без гонок, сериализация по игроку, эскалация.

`JobsQueue` получает не сессию, а `session_factory` — тот же `async_sessionmaker`,
что и фикстура `db_factory` (задача 14). Это меняет то, кому принадлежит граница
транзакции: у репозиториев (`memory/repos.py`) коммit — дело вызывающего, потому
что снаружи уже есть открытая сессия с известной границей. Здесь снаружи только
фабрика — открыть и закрыть транзакцию больше некому, кроме самой очереди.
Поэтому каждый публичный метод — одна транзакция целиком: открыть сессию,
сделать дело, закоммитить, вернуть результат. Для `claim()` это не стиль, а
необходимость: `FOR UPDATE SKIP LOCKED` держит лок над строкой-кандидатом ровно
на время своей транзакции, и та обязана закрыться коммитом внутри метода — иначе
второй воркер не увидит эту блокировку и получит ту же строку. Гарантия здесь
именно такая, точечная: она не мешает двум воркерам одновременно взять РАЗНЫЕ
строки — сериализация по игроку (см. следующий абзац) устроена отдельно.

**Сериализация по игроку — в двух слоях, и оба обязательны (задача 15, fix round
1).** `NOT EXISTS` внутри `_CLAIM_SQL` — быстрый путь без лишних блокировок в
общем случае, но сам по себе недостаточен: под READ COMMITTED он видит только
закоммиченное, а `FOR UPDATE SKIP LOCKED` блокирует только саму строку-кандидата
`j`, не строки, которые проверяет подзапрос `r`. Из-за этого два конкурентных
`claim()` могли одновременно выбрать РАЗНЫЕ `queued`-задачи одного игрока — ни
один не видел решения другого, потому что решение другого ещё не закоммичено
(воспроизведено на реальном Postgres 16, 300/300). Партиционный уникальный индекс
`uq_jobs_player_id_running` (миграция `0002`, `on jobs(player_id) WHERE
status='running'`) — вторая линия защиты и настоящая гарантия: он делает вторую
одновременную `running`-строку физически невозможной при любом уровне изоляции,
а `NOT EXISTS` остаётся оптимизацией, которая в большинстве случаев не даёт делу
дойти до конфликта на индексе.

Приоритеты типов задач (спека §8.1: `screenshot_analyze`/`deep_dive` — 100,
`hh_scan` — 200 не срочен, `eval_run` — 900 самый терпеливый) `enqueue`
подставляет сама по `type`, если вызывающий явно не передал `priority` —
CHECK-констрейнт `jobs.type_allowed` и так ограничивает `type` этими четырьмя
значениями, так что таблица соответствий здесь исчерпывающая, а не эвристика.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from asyncpg.exceptions import UniqueViolationError
from sqlalchemy import select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from harness.memory.models import JOBS_RUNNING_UNIQUE_INDEX, Job


class JobPreconditionFailed(Exception):
    """Задача найдена, но её текущее состояние не совпадает с тем, что ожидал
    вызывающий: `resume()` вызван не на `awaiting_user`, либо `complete()`/`fail()`
    переданы с `worker_id`, который уже не владелец (`locked_by`) — например,
    `reap()` успел отдать задачу другому воркеру, пока первый зомби-воркер ещё
    жив и пытается закрыть то, что ему больше не принадлежит (задача 15, fix
    round 1, Finding 2). Отдельно от `LookupError` (задачи вовсе нет в БД) —
    вызывающему важно различать эти два случая, а не получать в обоих один и
    тот же молчаливый провал.
    """

# Порядок и значения — спека §8.1 дословно.
_DEFAULT_PRIORITY_BY_TYPE: dict[str, int] = {
    "screenshot_analyze": 100,
    "deep_dive": 100,
    "hh_scan": 200,
    "eval_run": 900,
}

# Единственный SQL, которому доверена сама атомарность захвата: WHERE-подзапрос
# и UPDATE — один стейтмент, поэтому SELECT ... FOR UPDATE SKIP LOCKED и запись
# результата происходят в одном обращении к Postgres, без окна между «выбрали»
# и «заняли», в которое мог бы влезть второй воркер. NOT EXISTS — сериализация
# по игроку (спека §8.1): `running` считается активной, `awaiting_user` — нет,
# ровно потому что в подзапросе он не упомянут в списке статусов.
_CLAIM_SQL = text(
    """
    UPDATE jobs SET status='running', locked_by=:worker_id, locked_at=now(), attempts=attempts+1
    WHERE id = (
      SELECT j.id FROM jobs j
      WHERE j.status = 'queued'
        AND NOT EXISTS (SELECT 1 FROM jobs r
                        WHERE r.player_id = j.player_id AND r.status = 'running')
      ORDER BY j.priority, j.created_at
      FOR UPDATE SKIP LOCKED LIMIT 1
    ) RETURNING *
    """
)

# Сообщение игроку/логу для задачи, которую reaper закрыл после последней
# попытки — честная формулировка вместо тихого исчезновения задачи.
_REAP_ERROR = "воркер завис и не ответил — исчерпаны все попытки (reaper)"


def _is_running_serialization_conflict(exc: IntegrityError) -> bool:
    """`claim()` вправе проглотить РОВНО ОДИН конфликт — проигрыш гонки за
    `JOBS_RUNNING_UNIQUE_INDEX` — и ничего больше (fix round 2, Item 2): широкий
    `except IntegrityError` ловил вообще любое нарушение целостности, и любой
    будущий констрейнт на `jobs`, случайно задетый этим же `UPDATE`, молча
    превратился бы в «пустая очередь» вместо честного падения.

    SQLAlchemy заворачивает исключение asyncpg в свой DBAPI-совместимый класс
    (`exc.orig`), но не теряет оригинал: он доступен через `exc.orig.__cause__`
    (`raise ... from ...` внутри диалекта asyncpg) — только там есть
    `constraint_name`, а не только SQLSTATE класса unique_violation (`23505`),
    которым могло бы совпасть и нарушение СОВСЕМ ДРУГОГО уникального индекса,
    если такой когда-нибудь появится на `jobs`. Узость проверена на настоящем
    Postgres — `test_claim_propagates_unrelated_integrity_error`.
    """
    cause = exc.orig.__cause__ if exc.orig is not None else None
    if not isinstance(cause, UniqueViolationError):
        return False
    # asyncpg выставляет constraint_name динамически (из ErrorResponse Postgres),
    # его нет в статических тайп-стабах — getattr вместо `cause.constraint_name`,
    # чтобы pyright не спотыкался о реально существующий рантайм-атрибут.
    return getattr(cause, "constraint_name", None) == JOBS_RUNNING_UNIQUE_INDEX


class JobsQueue:
    """Очередь + журнал `jobs`: захват, статусы, эскалация, страховочный reaper."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def enqueue(
        self,
        *,
        type: str,
        player_id: int,
        session_id: int,
        payload: dict,
        priority: int | None = None,
    ) -> int:
        """Поставить задачу в очередь. `priority=None` — дефолт по `type` (см.
        модульный докстринг); явный `priority` переопределяет его — нужен эскалации
        и ручным сценариям, где приоритет диктует не тип задачи, а обстоятельство.
        """
        resolved_priority = (
            _DEFAULT_PRIORITY_BY_TYPE.get(type, 100) if priority is None else priority
        )
        async with self._session_factory() as session:
            job = Job(
                type=type,
                player_id=player_id,
                session_id=session_id,
                payload=payload,
                priority=resolved_priority,
            )
            session.add(job)
            await session.commit()
            return job.id

    async def claim(self, worker_id: str) -> Job | None:
        """Атомарно забрать одну задачу — `_CLAIM_SQL`, см. докстринг модуля.

        Проигравший гонку за партиционный уникальный индекс (`JOBS_RUNNING_
        UNIQUE_INDEX`, миграция `0002`) получает `IntegrityError` прямо на
        `execute()` — Postgres не откладывает проверку уникального индекса.
        Транзакция после этого испорчена (Postgres требует `ROLLBACK`, повторный
        `commit()` кинул бы `PendingRollbackError`), а не просто "ничего не
        нашли". Внутри этого же вызова заново не пытаемся: `worker_id` внутри
        `_CLAIM_SQL` уже выбрал конкретную строку-кандидата и проиграл её
        конкретному конкуренту — заново выбирать пришлось бы с нуля, отдельным
        запросом, а вызывающий (воркер в цикле опроса) и так трактует `None` как
        «сейчас нечего взять» и попробует снова на следующей итерации. Раз
        `None` — легитимный, ожидаемый исход именно ЭТОГО конфликта (см.
        `_is_running_serialization_conflict`, fix round 2, Item 2) — а не
        признак сбоя, наружу он не протекает никогда. Любой ДРУГОЙ
        `IntegrityError` — это не «нечего взять», а настоящая поломка, и
        перевыбрасывается как есть.
        """
        async with self._session_factory() as session:
            try:
                result = await session.execute(_CLAIM_SQL, {"worker_id": worker_id})
                row = result.mappings().first()
                await session.commit()
            except IntegrityError as exc:
                if not _is_running_serialization_conflict(exc):
                    raise
                await session.rollback()
                return None
            if row is None:
                return None
            # Явные именованные поля, а не `Job(**dict(row))`: конструктор
            # ORM-модели от произвольного маппинга строки типами не проверяется,
            # explicit — единообразно с остальным repos.py (Player(...), Hand(...)).
            return Job(
                id=row["id"],
                type=row["type"],
                status=row["status"],
                payload=row["payload"],
                session_id=row["session_id"],
                player_id=row["player_id"],
                hand_id=row["hand_id"],
                attempts=row["attempts"],
                max_attempts=row["max_attempts"],
                priority=row["priority"],
                locked_by=row["locked_by"],
                locked_at=row["locked_at"],
                created_at=row["created_at"],
                finished_at=row["finished_at"],
                error=row["error"],
            )

    async def complete(self, job_id: int, worker_id: str | None = None) -> None:
        """Закрыть задачу успехом. `worker_id`, если передан, обязан совпасть с
        `locked_by` — без этой проверки зомби-воркер, чья задача уже вернулась
        в очередь через `reap()` и подхвачена другим воркером, мог бы молча
        затереть чужой прогресс (Finding 2, fix round 1). Параметр остаётся
        опциональным, чтобы не ломать вызовы без него (задача 18 будет передавать
        его всегда); `status == 'running'` — обязательная предпосылка независимо
        от `worker_id`: закрывать успехом можно только то, что реально было в
        работе.
        """
        async with self._session_factory() as session:
            await self._apply_transition(
                session,
                job_id=job_id,
                expected_status="running",
                worker_id=worker_id,
                values={"status": "done", "finished_at": datetime.now(UTC)},
            )

    async def fail(self, job_id: int, error: str, worker_id: str | None = None) -> None:
        """Закрыть задачу неудачей — те же гарантии владения и предпосылки, что
        у `complete()` (см. её докстринг).
        """
        async with self._session_factory() as session:
            await self._apply_transition(
                session,
                job_id=job_id,
                expected_status="running",
                worker_id=worker_id,
                values={"status": "failed", "error": error, "finished_at": datetime.now(UTC)},
            )

    async def await_user(
        self, job_id: int, resume_payload: dict, worker_id: str | None = None
    ) -> None:
        """Точка возврата эскалации (спека §8.3): задача перестаёт быть активной
        для сериализации по игроку (§8.1), но не начинается заново — `payload`
        сливается с накопленным, а не заменяет его, чтобы не потерять чекпоинты
        станций, уже записанные до вопроса игроку (§8.2, например message_id
        прогресс-сообщения).

        Та же дыра, что была у `complete()`/`fail()` до fix round 1, и контроллер
        назвал её отдельно (fix round 2, Item 1): `reap()` возвращает зависшую
        задачу в очередь, другой воркер её подхватывает, а «ожившему» зомби-
        воркеру больше нельзя молча ставить `awaiting_user` — иначе живая задача
        нового владельца зависла бы в ожидании ответа на вопрос, которого никто
        не задавал. `worker_id`/предпосылка `status == 'running'` — тот же
        `_apply_transition`, что у `complete()`/`fail()`, специально не другой
        приём: контроллер прямо просил ту же форму, а не её вариацию.

        Слияние `payload` читается ОТДЕЛЬНЫМ запросом до атомарного перехода —
        не через `session.get()` (тот пометил бы объект в identity map и испортил
        диагностику `_apply_transition` в ветке провала, см. её докстринг), а
        через точечный `select(Job.payload)`, не трогающий ORM-состояние сессии.
        Если между этим чтением и переходом владение сменится, атомарный `WHERE`
        всё равно отклонит запись — вычисленное слияние на основе устаревшего
        чтения просто не попадёт в БД, а не попадёт молча поверх чужого прогресса.
        """
        async with self._session_factory() as session:
            current_payload = await session.scalar(select(Job.payload).where(Job.id == job_id))
            merged_payload = {**(current_payload or {}), **resume_payload}
            await self._apply_transition(
                session,
                job_id=job_id,
                expected_status="running",
                worker_id=worker_id,
                values={"status": "awaiting_user", "payload": merged_payload},
            )

    async def resume(self, job_id: int) -> None:
        """Ответ игрока получен — `awaiting_user` → `queued`, задачу возьмёт
        любой свободный воркер (спека §8.3, шаг 3). Предпосылка `status ==
        'awaiting_user'` обязательна (Minor, fix round 1): без неё повторный или
        запоздавший вызов молча переоткрыл бы уже `done`/`failed`/`running` задачу.
        """
        async with self._session_factory() as session:
            await self._apply_transition(
                session,
                job_id=job_id,
                expected_status="awaiting_user",
                worker_id=None,
                values={"status": "queued"},
            )

    async def reap(self, older_than_minutes: int = 10) -> int:
        """Страховка по возрасту лока (спека §8.1): если воркер умер, не дойдя
        до `complete()`/`fail()`, его задача осталась `running` с устаревшим
        `locked_at` навсегда — транзакция `claim()` уже закоммичена и своей
        блокировки строки давно не держит, отследить зависание больше нечем,
        кроме как по времени. `attempts` уже увеличен захватом —
        `reap()` его не трогает, а лишь читает: не исчерпан лимит — назад в
        `queued` на следующую попытку; исчерпан (`attempts >= max_attempts`) —
        `failed` с честной причиной, а не тихий бесконечный повтор.
        Возвращает число обработанных задач (обе ветки вместе).
        """
        threshold = datetime.now(UTC) - timedelta(minutes=older_than_minutes)
        async with self._session_factory() as session:
            failed = await session.execute(
                update(Job)
                .where(
                    Job.status == "running",
                    Job.locked_at < threshold,
                    Job.attempts >= Job.max_attempts,
                )
                .values(
                    status="failed",
                    error=_REAP_ERROR,
                    finished_at=datetime.now(UTC),
                    locked_by=None,
                    locked_at=None,
                )
                .execution_options(synchronize_session=False)
                .returning(Job.id)
            )
            requeued = await session.execute(
                update(Job)
                .where(
                    Job.status == "running",
                    Job.locked_at < threshold,
                    Job.attempts < Job.max_attempts,
                )
                .values(status="queued", locked_by=None, locked_at=None)
                .execution_options(synchronize_session=False)
                .returning(Job.id)
            )
            # `.returning(Job.id)` вместо `.rowcount`: у последнего нет надёжной
            # типизации на общем `Result[Any]` (pyright ругается), а посчитать
            # вернувшиеся id — тот же результат и без приведения типов.
            reaped = len(failed.all()) + len(requeued.all())
            await session.commit()
            return reaped

    async def _apply_transition(
        self,
        session: AsyncSession,
        *,
        job_id: int,
        expected_status: str,
        worker_id: str | None,
        values: dict,
    ) -> None:
        """Атомарный переход состояния: один `UPDATE ... WHERE id=... AND
        status=expected_status [AND locked_by=worker_id] RETURNING id`, а не
        `session.get()` + мутация + `commit()`. Между чтением и записью в схеме
        "прочитали — поменяли в Python — закоммитили" есть окно, где статус
        задачи мог смениться другой транзакцией (например `reap()`) — этот метод
        существует именно для того, чтобы такого окна не было (Finding 2, fix
        round 1: тот же принцип, что и партиционный индекс для `claim()`,
        "гарантия в БД, а не в порядке вызовов Python").

        0 обновлённых строк — это ещё не факт "их запись". Различаем: если
        задачи вовсе нет — `LookupError` (второй, диагностический запрос — но он
        случается только на уже исключительном пути, не в общем случае успеха);
        если есть, но `status`/`locked_by` не совпали с ожиданием —
        `JobPreconditionFailed`.
        """
        conditions = [Job.id == job_id, Job.status == expected_status]
        if worker_id is not None:
            conditions.append(Job.locked_by == worker_id)
        result = await session.execute(
            update(Job)
            .where(*conditions)
            .values(**values)
            .execution_options(synchronize_session=False)
            .returning(Job.id)
        )
        matched = result.first() is not None
        if matched:
            await session.commit()
            return
        await session.rollback()
        job = await session.get(Job, job_id)
        if job is None:
            raise LookupError(f"задача {job_id} не найдена")
        raise JobPreconditionFailed(
            f"задача {job_id}: ожидали status={expected_status!r}"
            + (f", locked_by={worker_id!r}" if worker_id is not None else "")
            + f", а сейчас status={job.status!r}, locked_by={job.locked_by!r}"
        )
