"""Очередь `jobs` (спека §8): захват без гонок, сериализация по игроку, эскалация.

`JobsQueue` получает не сессию, а `session_factory` — тот же `async_sessionmaker`,
что и фикстура `db_factory` (задача 14). Это меняет то, кому принадлежит граница
транзакции: у репозиториев (`memory/repos.py`) коммit — дело вызывающего, потому
что снаружи уже есть открытая сессия с известной границей. Здесь снаружи только
фабрика — открыть и закрыть транзакцию больше некому, кроме самой очереди.
Поэтому каждый публичный метод — одна транзакция целиком: открыть сессию,
сделать дело, закоммитить, вернуть результат. Для `claim()` это не стиль, а
необходимость: `FOR UPDATE SKIP LOCKED` держит лок ровно на время своей
транзакции, и она обязана закрыться коммитом внутри метода — иначе конкурентные
воркеры не увидят блокировки друг друга (проверено `test_claim_atomic_two_workers`
на настоящем Postgres: `db` из задачи 14, откатываемая транзакция, для этого
принципиально не годится).

Приоритеты типов задач (спека §8.1: `screenshot_analyze`/`deep_dive` — 100,
`hh_scan` — 200 не срочен, `eval_run` — 900 самый терпеливый) `enqueue`
подставляет сама по `type`, если вызывающий явно не передал `priority` —
CHECK-констрейнт `jobs.type_allowed` и так ограничивает `type` этими четырьмя
значениями, так что таблица соответствий здесь исчерпывающая, а не эвристика.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from harness.memory.models import Job

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
        """Атомарно забрать одну задачу — `_CLAIM_SQL`, см. докстринг модуля."""
        async with self._session_factory() as session:
            result = await session.execute(_CLAIM_SQL, {"worker_id": worker_id})
            row = result.mappings().first()
            await session.commit()
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

    async def complete(self, job_id: int) -> None:
        async with self._session_factory() as session:
            job = await self._get_job(session, job_id)
            job.status = "done"
            job.finished_at = datetime.now(UTC)
            await session.commit()

    async def fail(self, job_id: int, error: str) -> None:
        async with self._session_factory() as session:
            job = await self._get_job(session, job_id)
            job.status = "failed"
            job.error = error
            job.finished_at = datetime.now(UTC)
            await session.commit()

    async def await_user(self, job_id: int, resume_payload: dict) -> None:
        """Точка возврата эскалации (спека §8.3): задача перестаёт быть активной
        для сериализации по игроку (§8.1), но не начинается заново — `payload`
        сливается с накопленным, а не заменяет его, чтобы не потерять чекпоинты
        станций, уже записанные до вопроса игроку (§8.2, например message_id
        прогресс-сообщения).
        """
        async with self._session_factory() as session:
            job = await self._get_job(session, job_id)
            job.payload = {**job.payload, **resume_payload}
            job.status = "awaiting_user"
            await session.commit()

    async def resume(self, job_id: int) -> None:
        """Ответ игрока получен — `awaiting_user` → `queued`, задачу возьмёт
        любой свободный воркер (спека §8.3, шаг 3).
        """
        async with self._session_factory() as session:
            job = await self._get_job(session, job_id)
            job.status = "queued"
            await session.commit()

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

    async def _get_job(self, session: AsyncSession, job_id: int) -> Job:
        job = await session.get(Job, job_id)
        if job is None:
            raise LookupError(f"задача {job_id} не найдена")
        return job
