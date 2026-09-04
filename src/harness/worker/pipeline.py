"""Оркестрация одной попытки задачи: `run_job(job, deps)` — станции по типу `job.type`,
раскладка руки чекпоинтами (`hands.raw` → `.canonical` → `.enriched`), идемпотентная
отправка результата.

**Чекпоинты — суть задачи, не деталь.** `hands.raw`/`.canonical`/`.enriched` nullable
именно затем, чтобы повторная попытка ПРОДОЛЖАЛА, а не начинала заново (спека §8.2):
`_run_hh_scan` при входе смотрит, что уже сохранено (`jobs.payload["tournament_id"]`,
`jobs.payload["hands_saved"]`, число строк `hands` турнира), и не трогает то, что уже
есть. `test_resume_skips_done_stations` — не формальность, а спецификация: он ломает
`parse_file` и требует, чтобы задача всё равно дошла до конца.

**Идемпотентная отправка — вторая половина того же принципа.** `message_id` прогресса
и результата живут в `jobs.payload` (не в памяти воркера): повторный `run_job` редактирует
уже отправленное сообщение (`Sender.edit`), а не шлёт дубль. `_send_idempotent` — одна
реализация для обоих (прогресс редактируется станцией, результат — при повторном прогоне).

**Дедлайн задачи (контроллерский рулинг задачи 18, п.3).** `JobsQueue.reap()` считает
задачу зависшей и возвращает её в очередь через 10 минут молчания (`locked_at` не
обновляется по ходу работы) — воркер, который сам не бросит слишком долгую попытку,
рискует быть переигранным reaper'ом: вторую попытку возьмёт другой воркер, а первая
всё ещё будет работать и в конце попробует закрыть задачу, которая ему уже не
принадлежит (отсюда и фенсинг `worker_id`, п.2 ниже). `_JOB_DEADLINE_S` — бюджет по
типу задачи, с запасом меньше 10 минут: `asyncio.wait_for` вокруг работы станций (не
вокруг трейса/`complete`/`fail` — тем всегда дают дожить до конца). Задача 16 сознательно
не ограничила длительность самого вызова модели (только ожидание лимитера) и оставила
общий бюджет здесь — сейчас `llm` станциями v1-HH не вызывается вовсе (интерфейсы этой
задачи, дословно), но дедлайн станции уже покрывает и будущий вызов модели внутри неё
транзитивно, не дожидаясь отдельной задачи на таймаут конкретно LLM-запроса.

**Фенсинг (контроллерский рулинг задачи 18, п.2).** `job.locked_by`, который вернул
`claim()`, передаётся В КАЖДЫЙ вызов `complete()`/`fail()`/`await_user()` как `worker_id`
— задача 15 сделала параметр опциональным ровно чтобы задача 18 начала его передавать;
пропустить его здесь значило бы вновь открыть дыру, которую фенсинг закрывает: зомби-
воркер, чья задача уже подхвачена другим после `reap()`, молча затирает чужой прогресс.

**Общий кэш расчётов (контроллерский рулинг задачи 18, п.1).** Эквити-кэш `preflop.py`
живёт в памяти процесса и на диске воркера — при масштабировании (`--scale worker=N`,
задача 20) каждый воркер грел бы его заново, и холодные 235.8с скана (задача 13) возвращались
бы на каждом новом контейнере. `_run_hh_scan` перед сканом читает `calc_cache` (общий на всех
воркеров и все турниры, задача 13: ключ уровня класса руки и глубины, не раздачи), сеет
прочитанное во внутрипроцессный кэш подпроцесса пула аргументом `run_in_executor`
(`_scan_tournament_with_cache`), а после скана записывает обратно то, что подпроцесс досчитал
— `ON CONFLICT DO NOTHING`: значения детерминированы сидом сэмплера, переписывать нечем.
Дисковый файл (`equity_mc_cache.json`) при этом не убран — он остаётся тёплым внутри ОДНОГО
процесса между задачами, `calc_cache` закрывает то, что диск не может: общее хранилище МЕЖДУ
процессами/воркерами. Эскалация `await_user`/`resume` здесь не используется: `validate()`
для `Provenance.HAND_HISTORY` — только `pass`/`reject` (см. `engine/validation.py`, докстринг
модуля дословно), `escalate` — путь скриншота, которого у v1-HH нет.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import Executor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

import structlog
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from harness.analysis import analyze_hand
from harness.analysis.preflop import (
    equity_cache_export,
    equity_cache_fingerprint,
    equity_cache_seed,
)
from harness.analysis.scan import ScanSummary, scan_tournament
from harness.contracts import AnalysisResult, EnrichedHand, ValidationStatus, Zone
from harness.engine import enrich
from harness.memory.models import Job as JobModel
from harness.memory.models import Player
from harness.memory.repos import AnalysesRepo, CalcCacheRepo, HandsRepo, TournamentsRepo
from harness.normalizer import normalize
from harness.parsers import hh_parser
from harness.platform.llm import LLM
from harness.platform.queue import JobPreconditionFailed, JobsQueue
from harness.platform.trace import Clock, Trace
from harness.presentation import Msg, deep_dive_msg, failed_msg, progress_text, scan_summary_msg

__all__ = ["Deps", "Sender", "run_job"]

_log = structlog.get_logger(__name__)

# Станции v1-HH, показываемые игроку прогрессом (спека — `progress_text` знает четыре
# ярлыка, "explain" сюда не входит: текст вердикта LLM формулирует задача 21, здесь его
# ещё не пишем, см. модульный докстринг presentation/messages.py).
_Station = Literal["parse", "validate", "analyze"]

# Бюджет попытки по типу задачи — с запасом меньше десятиминутного окна reap()
# (см. модульный докстринг). `hh_scan` может обрабатывать сотни рук и считать эквити
# по файлу целиком — щедрее; `deep_dive` — одна рука, дёшево даже с запасом.
_JOB_DEADLINE_S: dict[str, float] = {"hh_scan": 480.0, "deep_dive": 120.0}
_DEFAULT_JOB_DEADLINE_S = 300.0

# Квота по умолчанию (спека §9, пример дословно: "разборов 17/50 за 24ч") — используется,
# когда у игрока нет персонального `quota_daily` (players.quota_daily IS NULL). Полное
# решение о допуске (`allowed`/`hours_to_free`) — задача 19 (bot, §9): здесь только числа
# для честной подписи сообщения, воркер сам не отказывает в разборе (отказ — до постановки
# задачи в очередь, не после).
_QUOTA_TOTAL_DEFAULT = 50
_INTERACTIVE_JOB_TYPES = ("deep_dive", "screenshot_analyze")

# Закрытый набор причин отказа, которые честно показать игроку (fix round 1,
# Important 1). `str(exc)` бывает `FileNotFoundError: [Errno 2] ... '/data/hh/
# {hash}.txt'` (путь на диске), номер руки, текст ошибки SQLAlchemy — ровно то,
# что публикационная политика репозитория (fixtures/hh, номера рук как приватные
# данные игрока) запрещает публиковать, а правило единого голоса (presentation
# формулирует то, что видит игрок, а не необработанное исключение) запрещает
# показывать буквально. Внутренняя причина никуда не девается — `jobs.error`
# (queue.fail) и лог (`_log.error` в `run_job`) остаются полными и точными,
# это ops-данные, не то, что уходит в Sender.
_PUBLIC_FAILURE_REASON_DEFAULT = "внутренняя ошибка сервиса"
_PUBLIC_FAILURE_REASONS: tuple[tuple[type[BaseException], str], ...] = (
    (FileNotFoundError, "файл раздач недоступен"),
    (LookupError, "не нашли нужные данные по этой раздаче"),
    (TimeoutError, "расчёт не уложился в отведённое время"),
)


def _public_failure_reason(exc: BaseException) -> str:
    """Классифицировать исключение в одну из заранее названных, безопасных для
    игрока причин — никогда не `str(exc)` напрямую (см. константы выше)."""
    for exc_type, reason in _PUBLIC_FAILURE_REASONS:
        if isinstance(exc, exc_type):
            return reason
    return _PUBLIC_FAILURE_REASON_DEFAULT


class Sender(Protocol):
    """То, чем `run_job` доставляет сообщения игроку — реализация (Телеграм или
    тестовый двойник) ему не известна, только эти два метода.
    """

    async def send(self, chat_id: int, msg: Msg) -> int: ...

    async def edit(self, chat_id: int, message_id: int, msg: Msg) -> None: ...


@dataclass(frozen=True, slots=True)
class Deps:
    """Зависимости одного воркер-процесса — общие на все задачи подряд, не на одну
    попытку (`Trace`, в отличие от этого, заводится внутри `run_job` заново каждый раз,
    см. `platform/trace.py`).

    `process_pool` — не часть пятёрки `(db_factory, queue, sender, llm, clock)` из
    брифа задачи буквально, но без него нечем выполнить его же явное требование
    "скан-расчёты — через `run_in_executor(process_pool, ...)`" (спека §2: CPU не
    блокирует луп). `None` — тестовый дефолт (упрощённые тесты, которым процессный
    пул не нужен: `run_in_executor(None, ...)` использует ThreadPoolExecutor по
    умолчанию); прод (`worker/main.py`) всегда передаёт настоящий `ProcessPoolExecutor`.
    """

    db_factory: async_sessionmaker[AsyncSession]
    queue: JobsQueue
    sender: Sender
    llm: LLM
    clock: Clock = time.monotonic
    process_pool: Executor | None = None


def _cache_delta(cache_seed: dict[str, float]) -> dict[str, float]:
    """Только НОВОЕ поверх засеянного — не весь процессный кэш целиком (fix
    round 1, Minor: `upsert_many` без этого пересылал бы в `calc_cache` весь
    накопленный кэш подпроцесса на каждый скан/разбор, включая то, что мы сами
    только что туда положили строками выше — O(глобального кэша) на запись
    вместо O(того, что реально досчитала эта задача), на кэше, спроектированном
    как общий на всех пользователей). Всё, что осталось в кэше и НЕ было частью
    засеянного, стоит отправить в БД независимо от источника (досчитано этой
    задачей ИЛИ уже лежало в дисковом файле подпроцесса, которого в `calc_cache`
    ещё не было) — оба случая законно "то, чего БД пока не знает".
    """
    exported = equity_cache_export()
    return {key: value for key, value in exported.items() if key not in cache_seed}


def _scan_tournament_with_cache(
    enriched: list[EnrichedHand], cache_seed: dict[str, float]
) -> tuple[ScanSummary, dict[str, float]]:
    """Обёртка процессного пула: переносит общий эквити-кэш через границу процесса
    аргументом/результатом (модуль-level состояние `preflop.py` подпроцессы не делят
    автоматически — только явной сериализацией). Должна остаться функцией верхнего
    уровня модуля: `ProcessPoolExecutor` пиклит вызываемое по имени, замыкание или
    метод объекта он бы не принял.
    """
    equity_cache_seed(cache_seed)
    summary = scan_tournament(enriched)
    return summary, _cache_delta(cache_seed)


def _analyze_hand_with_cache(
    enriched: EnrichedHand, cache_seed: dict[str, float]
) -> tuple[AnalysisResult, dict[str, float]]:
    """Сиблинг `_scan_tournament_with_cache` для `deep_dive` (fix round 1,
    Important 2, дважды). Во-первых: `analyze_hand` зовёт тот же `verdict_for`,
    что и скан, и на холодном кэше эквити может стоить секунды на одну руку —
    синхронный вызов внутри корутины блокировал бы весь процесс (`reap_loop`,
    прогресс ДРУГИХ задач) на всё это время; спека §2 требует CPU-работу через
    пул, а буква брифа называет "скан-расчёты" только потому, что не различала
    два вызова одного и того же расчётного ядра. Во-вторых: без сидирования/
    экспорта здесь `deep_dive` не участвовал бы в общем `calc_cache` вовсе —
    первый разбор свежего контейнера считал бы заново и выбрасывал результат,
    несмотря на то, что скан того же турнира уже мог посчитать тот же спот
    (контроллерский рулинг задачи 18, п.1, распространяется на оба пути, не
    только на скан).
    """
    equity_cache_seed(cache_seed)
    result = analyze_hand(enriched)
    return result, _cache_delta(cache_seed)


async def _fenced_update(
    session: AsyncSession, job_id: int, worker_id: str | None, **values: Any
) -> None:
    """Обновить `jobs` только если `locked_by` всё ещё совпадает с `worker_id` —
    тот же приём, что `JobsQueue._apply_transition` (задача 15), перенесённый
    в этот модуль (fix round 1, Important 4). Ноль обновлённых строк — не
    "записали и ладно", а "эта попытка больше не владеет задачей" (`reap()`
    успел отдать её другому воркеру, пока эта ещё дорабатывала): молча
    продолжать значило бы затирать состояние нового владельца тем, что
    досчитал зомби. Коммитит сама на успехе — каждый вызов самостоятельный
    чекпоинт, обязанный пережить крах сразу после записи (см. `_sync_payload`).
    """
    result = await session.execute(
        update(JobModel)
        .where(JobModel.id == job_id, JobModel.locked_by == worker_id)
        .values(**values)
        .returning(JobModel.id)
    )
    if result.first() is None:
        await session.rollback()
        raise JobPreconditionFailed(
            f"задача {job_id}: воркер {worker_id!r} больше не владелец — запись отклонена"
        )
    await session.commit()


async def _sync_payload(
    session: AsyncSession, job_id: int, worker_id: str | None, payload: dict[str, Any]
) -> None:
    """Записать текущий чекпоинт `jobs.payload` — единственное место в этом модуле,
    где обновление таблицы `jobs` не идёт через `JobsQueue` (тот не даёт менять
    `payload` без смены статуса, а резюме нужно делать это посреди `running`, не
    трогая статус). Фенсинг — через `_fenced_update` (см. её докстринг).
    """
    await _fenced_update(session, job_id, worker_id, payload=payload)


async def _send_idempotent(
    deps: Deps,
    session: AsyncSession,
    job_id: int,
    worker_id: str | None,
    key: Literal["progress_message_id", "result_message_id"],
    chat_id: int,
    msg: Msg,
) -> dict[str, Any]:
    """Отправить один раз, дальше — редактировать. Читает `payload` ЗАНОВО из БД
    (не полагается на снимок, который вызывающий мог сделать в начале попытки —
    сама попытка может идти минуты, задача 13) и сверяет владельца ПЕРЕД
    отправкой: зомби, чья задача уже переиграна `reap()`, обязан узнать об этом
    ДО вызова `Sender.send()`, а не после — иначе игрок уже получил дубль, и
    фенсинг самой ЗАПИСИ (`_fenced_update`) только не даёт зомби затереть
    `payload` нового владельца, но не спасает от уже отправленного дубля (fix
    round 1, Important 4; `test_stale_worker_cannot_complete_reclaimed_job`
    проверяет именно это, не только состояние `jobs`).

    Возвращает актуальный `payload` — вызывающий обязан подхватить его
    (`payload = await _send_idempotent(...)`), а не продолжать со своей
    локальной копией: та могла устареть даже за время ЭТОГО вызова.
    """
    current_payload, current_owner = (
        await session.execute(
            select(JobModel.payload, JobModel.locked_by).where(JobModel.id == job_id)
        )
    ).one()
    if current_owner != worker_id:
        raise JobPreconditionFailed(
            f"задача {job_id}: воркер {worker_id!r} больше не владелец — отправка отменена"
        )
    payload = dict(current_payload or {})
    message_id = payload.get(key)
    if message_id is None:
        message_id = await deps.sender.send(chat_id, msg)
        payload[key] = message_id
        await _fenced_update(session, job_id, worker_id, payload=payload)
    else:
        await deps.sender.edit(chat_id, message_id, msg)
    return payload


async def _ensure_progress(
    deps: Deps,
    session: AsyncSession,
    job_id: int,
    worker_id: str | None,
    chat_id: int,
    station: _Station,
) -> dict[str, Any]:
    return await _send_idempotent(
        deps,
        session,
        job_id,
        worker_id,
        "progress_message_id",
        chat_id,
        Msg(text=progress_text(station)),
    )


async def _chat_id(session: AsyncSession, player_id: int) -> int:
    """Телеграм для приватного чата с ботом — `chat_id == tg_user_id` (свойство
    личных чатов Bot API, не отдельная колонка схемы: заводить её ради одного
    равенства было бы дублированием источника истины).
    """
    player = await session.get(Player, player_id)
    if player is None:
        raise LookupError(f"игрок {player_id} не найден")
    return player.tg_user_id


async def _quota_numbers(session: AsyncSession, player_id: int) -> tuple[int, int]:
    """Числа для строки «разборов X/Y за 24ч» (спека §9: скользящее окно, SQL-счётчик
    интерактивных задач). Полная проверка допуска (`allowed`, `hours_to_free`) — задача
    19 (бот, §9 дословно): бот отказывает ДО постановки в очередь, воркер уже взял
    задачу в работу и не вправе отказывать — ему нужны только эти два числа для честной
    подписи сообщения, а не решение "пускать или нет".
    """
    player = await session.get(Player, player_id)
    total = (
        player.quota_daily
        if player is not None and player.quota_daily is not None
        else _QUOTA_TOTAL_DEFAULT
    )
    since = datetime.now(UTC) - timedelta(hours=24)
    used = await session.scalar(
        select(func.count())
        .select_from(JobModel)
        .where(
            JobModel.player_id == player_id,
            JobModel.type.in_(_INTERACTIVE_JOB_TYPES),
            JobModel.created_at > since,
        )
    )
    left = max(total - int(used or 0), 0)
    return left, total


async def _run_hh_scan(job: JobModel, deps: Deps, trace: Trace) -> None:
    worker_id = job.locked_by
    async with deps.db_factory() as session:
        payload: dict[str, Any] = dict(job.payload)
        chat_id = await _chat_id(session, job.player_id)
        hands_repo = HandsRepo(session)
        tournaments_repo = TournamentsRepo(session)

        if payload.get("hands_saved"):
            tournament_id: int = payload["tournament_id"]
        else:
            async with trace.span("parse"):
                payload = await _ensure_progress(deps, session, job.id, worker_id, chat_id, "parse")
                source_file = payload["source_file"]
                raw_text = Path(source_file).read_text(encoding="utf-8")
                # Импорт модулем, а не `from ... import parse_file` (задача 18,
                # falsификация `test_resume_skips_done_stations`): тест ломает
                # ИМЕННО `harness.parsers.hh_parser.parse_file` через monkeypatch,
                # а `from`-импорт связал бы имя здесь один раз при загрузке модуля
                # — патч исходника тогда не долетел бы до уже готовой ссылки (тот
                # же урок, что `_sleep = asyncio.sleep` в `platform/llm.py`).
                raw_hands = hh_parser.parse_file(raw_text, source_ref=source_file)

                if "tournament_id" in payload:
                    tournament_id = payload["tournament_id"]
                else:
                    tournament_id = await tournaments_repo.create(
                        session_id=job.session_id, source_file=source_file
                    )
                    payload["tournament_id"] = tournament_id
                    await _sync_payload(session, job.id, worker_id, payload)

                existing_raw = await hands_repo.count_by_tournament(tournament_id)
                # Известный остаточный риск (fix round 1, вне объёма Important 4):
                # эти построчные записи `hands` не фенсятся по `worker_id`, в
                # отличие от `jobs.payload`/`hand_id`. Зомби, потерявший владение
                # ПОСЕРЕДИНЕ этого цикла (а не на границе `_ensure_progress`,
                # которая фенсит вход в каждый проход), теоретически может
                # писать `hands` одновременно с новым владельцем. Узкое окно —
                # ограничено одним проходом одной станции, не всей задачей — и
                # исходом было бы самое большее несколько лишних строк `hands`
                # в рамках турнира, не дубль игроку и не порча `jobs`.
                for idx, raw in enumerate(raw_hands):
                    if idx < existing_raw:
                        continue  # чекпоинт: уже сохранена в прошлой попытке
                    await hands_repo.save_raw(
                        session_id=job.session_id, tournament_id=tournament_id, raw=raw
                    )
                    await session.commit()

            async with trace.span("validate"):
                payload = await _ensure_progress(
                    deps, session, job.id, worker_id, chat_id, "validate"
                )
                for idx, record in enumerate(await hands_repo.list_by_tournament(tournament_id)):
                    if record.canonical is not None and record.enriched is not None:
                        continue  # чекпоинт: обе станции уже пройдены этой рукой
                    canonical = record.canonical
                    if canonical is None:
                        canonical = normalize(record.raw).model_copy(update={"hand_index": idx})
                        await hands_repo.save_canonical(record.id, canonical)
                    if record.enriched is None:
                        enriched = enrich(canonical)
                        if enriched.verdict.status == ValidationStatus.REJECT:
                            # HH — факт рума: расхождение значит баг парсера, а не игрока
                            # (engine/validation.py, докстринг модуля) — лог разработчику,
                            # не отказ игроку и не подмена данных.
                            _log.warning(
                                "hh_hand_rejected",
                                hand_no=canonical.hand_no,
                                reasons=enriched.verdict.reasons,
                            )
                        await hands_repo.save_enriched(record.id, enriched)
                    await session.commit()

            payload["hands_saved"] = True
            await _sync_payload(session, job.id, worker_id, payload)

        async with trace.span("analyze"):
            payload = await _ensure_progress(deps, session, job.id, worker_id, chat_id, "analyze")
            hand_records = await hands_repo.list_by_tournament(tournament_id)
            enriched_hands = [r.enriched for r in hand_records if r.enriched is not None]

            cache_repo = CalcCacheRepo(session)
            prefix = f"equity_mc:{equity_cache_fingerprint()}:"
            seed = await cache_repo.get_all(prefix)
            # Не держим транзакцию через долгий исполнитель (fix round 1,
            # Important 3): `get_all` выше уже открыл транзакцию на чтение —
            # закрыть её здесь, ДО того как `run_in_executor` займёт до
            # `_JOB_DEADLINE_S["hh_scan"]` секунд снаружи event loop. Без этого
            # соединение простаивает "idle in transaction" всё это время,
            # умноженное на `WORKER_CONCURRENCY` и число реплик, блокирует
            # autovacuum, а под любым `idle_in_transaction_session_timeout`
            # соединение обрывается — и уже ГОТОВЫЙ результат скана теряется на
            # `upsert_many` ниже, а не только на самом расчёте.
            await session.commit()

            loop = asyncio.get_running_loop()
            summary, exported = await loop.run_in_executor(
                deps.process_pool, _scan_tournament_with_cache, enriched_hands, seed
            )
            await cache_repo.upsert_many(prefix, exported)
            await session.commit()

            await tournaments_repo.save_scan_summary(tournament_id, summary)
            await session.commit()

        quota_left, quota_total = await _quota_numbers(session, job.player_id)
        msg = scan_summary_msg(summary, quota_left, quota_total)
        await _send_idempotent(deps, session, job.id, worker_id, "result_message_id", chat_id, msg)
        await session.commit()


async def _run_deep_dive(job: JobModel, deps: Deps, trace: Trace, started_at: float) -> int | None:
    """Возвращает `hand_id` разобранной руки — `run_job` кладёт его в `traces.hand_id`
    (не в `_run_hh_scan`: там рук много, ни одна не выделена)."""
    worker_id = job.locked_by
    async with deps.db_factory() as session:
        payload: dict[str, Any] = dict(job.payload)
        chat_id = await _chat_id(session, job.player_id)
        hand_no = payload["hand_no"]
        hands_repo = HandsRepo(session)
        analyses_repo = AnalysesRepo(session)

        async with trace.span("analyze"):
            payload = await _ensure_progress(deps, session, job.id, worker_id, chat_id, "analyze")
            hand = await hands_repo.find_by_hand_no(job.session_id, hand_no)
            if hand is None:
                raise LookupError(f"рука {hand_no!r} не найдена в сессии {job.session_id}")
            if hand.enriched is None:
                raise LookupError(f"рука {hand_no!r} ещё не прошла чекпоинт enriched")

            await _fenced_update(session, job.id, worker_id, hand_id=hand.id)

            existing = await analyses_repo.get_by_hand(hand.id)
            result: AnalysisResult
            if existing is not None:
                result = existing.result  # чекпоинт: разбор уже посчитан прошлой попыткой
            else:
                cache_repo = CalcCacheRepo(session)
                prefix = f"equity_mc:{equity_cache_fingerprint()}:"
                seed = await cache_repo.get_all(prefix)
                await session.commit()  # тот же приём, что и в hh_scan — см. Important 3

                loop = asyncio.get_running_loop()
                result, exported = await loop.run_in_executor(
                    deps.process_pool, _analyze_hand_with_cache, hand.enriched, seed
                )
                await cache_repo.upsert_many(prefix, exported)
                await session.commit()

                await analyses_repo.save(hand_id=hand.id, result=result)
                await session.commit()

        elapsed_s = round(deps.clock() - started_at)
        zone = result.points[result.ranked[0]].zone if result.ranked else Zone.STRICT
        quota_left, quota_total = await _quota_numbers(session, job.player_id)
        msg = deep_dive_msg(result, elapsed_s, zone, quota_left, quota_total)
        await _send_idempotent(deps, session, job.id, worker_id, "result_message_id", chat_id, msg)
        await session.commit()

        hand_id = hand.id

    return hand_id


async def _dispatch(job: JobModel, deps: Deps, trace: Trace, started_at: float) -> int | None:
    if job.type == "hh_scan":
        await _run_hh_scan(job, deps, trace)
        return None
    if job.type == "deep_dive":
        return await _run_deep_dive(job, deps, trace, started_at)
    # `screenshot_analyze`/`eval_run` существуют в CHECK-констрейнте `jobs.type_allowed`
    # (задача 15, под будущие задачи 19+/22), но станций для них этот воркер ещё не
    # знает — явный отказ вместо молчаливого "ничего не произошло".
    raise NotImplementedError(f"воркер не умеет станцию для типа задачи {job.type!r}")


async def _finish_fenced(
    deps: Deps, job: JobModel, *, outcome: Literal["complete", "fail"], error: str | None = None
) -> bool:
    """Закрыть задачу в очереди — устойчиво к фенсингу (контроллерский рулинг
    задачи 18, п.2). `job.locked_by` — значение, которое `claim()` вернул ИМЕННО
    этому воркеру в момент захвата; если к моменту завершения станции текущий
    владелец в БД уже другой (`reap()` успел отдать задачу другому воркеру, пока
    этот ещё дорабатывал), `complete()`/`fail()` отклонят переход как
    `JobPreconditionFailed` (см. её докстринг в `platform/queue.py`) — это не
    баг и не повод падать, а именно то состояние, ради которого фенсинг
    существует: чужой прогресс важнее, чем упрямое доведение до конца попытки,
    которая уже никому не принадлежит. `run_job` эту ситуацию проглатывает
    молча (с логом) сознательно — `test_finish_fenced_does_not_complete_job_
    with_stale_worker_id` проверяет это утверждение напрямую, юнитом на этой
    функции (не через `run_job`/`_send_idempotent`: их собственный пред-чек
    владения, fix round 1, останавливает зомби раньше и до этой функции в
    типичном сценарии не доходит — см. докстринг `test_stale_worker_cannot_
    complete_reclaimed_job`, fix round 2, Item 1).

    Возвращает `True`, если переход в БД действительно произошёл — вызывающий
    (`run_job`) шлёт `failed_msg` игроку только в этом случае: задача, которую
    фенсинг отклонил, больше не наша, и извещать о её судьбе — не наше дело.
    """
    try:
        if outcome == "complete":
            await deps.queue.complete(job.id, worker_id=job.locked_by)
        else:
            await deps.queue.fail(job.id, error or "", worker_id=job.locked_by)
        return True
    except JobPreconditionFailed:
        _log.warning("job_fencing_lost_ownership", job_id=job.id, worker_id=job.locked_by)
        return False


async def _notify_failure(deps: Deps, job: JobModel, reason: str) -> None:
    """Лучшее из возможного уведомление игроку о провале — не должно само уронить
    обработку провала: если отправка тоже не удалась (нет сети, чат заблокирован,
    фенсинг), задача всё равно обязана остаться `failed` в БД, а не повиснуть.
    `reason` уже обязан быть публично безопасным — см. `_public_failure_reason`,
    эта функция сама текст не выбирает и не трогает.
    """
    try:
        async with deps.db_factory() as session:
            chat_id = await _chat_id(session, job.player_id)
            # `_send_idempotent` сама читает `payload` заново из БД и сверяет
            # владельца — вручную предчитывать его здесь больше не нужно
            # (fix round 1, Important 4: было единственным местом с этой
            # дисциплиной, теперь она в одном месте на все три вызывающих).
            await _send_idempotent(
                deps,
                session,
                job.id,
                job.locked_by,
                "result_message_id",
                chat_id,
                failed_msg(reason),
            )
    except Exception:  # noqa: BLE001 — намеренно: сбой уведомления не должен
        # уронить обработку провала задачи (задача уже честно `failed` в БД,
        # доложить о ней — только best-effort поверх этого).
        _log.exception("job_failure_notify_failed", job_id=job.id)


async def run_job(job: JobModel, deps: Deps) -> None:
    """Одна попытка одной задачи: станция(и) по `job.type`, трейс, финальный переход
    очереди. Никогда не бросает исключение наружу — вызывающий (`worker/main.py`,
    цикл `claim → run_job`) не обязан оборачивать каждый вызов в свой `try`, а
    `attempts < max_attempts` — легитимный "ещё не готово", не сбой цикла.

    **Политика ретрая — намеренно НЕ немедленный реквеуинг.** У `JobsQueue` нет
    метода "вернуть в queued прямо сейчас" (не в файлах этой задачи — трогать
    `platform/queue.py` не входит в её объём); единственный путь `running` → `queued`
    после ошибки — `reap()` по устаревшему `locked_at` (страховка на смерть воркера,
    не быстрый ретрай). Раз так, попытка, которой ещё есть куда расти (`attempts <
    max_attempts`), просто НЕ завершается ни `complete()`, ни `fail()` — воркер
    оставляет её как есть и ждёт `reap()`; последняя попытка (`attempts >=
    max_attempts`) закрывается `fail()` и игрок получает `failed_msg`
    (`test_failure_marks_failed_and_notifies`: "статус failed (после max_attempts)"
    — дословно то же самое требование).

    **Потеря владения — не провал попытки, а отдельный исход.** `JobPreconditionFailed`
    из `_dispatch` (через `_fenced_update`/`_send_idempotent`, fix round 1,
    Important 4) означает, что `reap()` уже отдал задачу другому воркеру, пока
    эта станция ещё работала — считать это "ещё одной попыткой" (наравне с
    обычным исключением) исказило бы и лог, и `attempts`: задача не провалилась,
    она просто больше не наша. Ни `fail()`, ни `complete()`, ни уведомление
    здесь не идут — задачей уже занимается (или уже закончил) новый владелец.
    """
    structlog.contextvars.bind_contextvars(job_id=job.id, job_type=job.type)
    trace = Trace(deps.db_factory, clock=deps.clock)
    hand_id: int | None = None
    try:
        deadline = _JOB_DEADLINE_S.get(job.type, _DEFAULT_JOB_DEADLINE_S)
        started_at = deps.clock()
        try:
            hand_id = await asyncio.wait_for(
                _dispatch(job, deps, trace, started_at), timeout=deadline
            )
        except JobPreconditionFailed:
            _log.warning(
                "job_fencing_lost_ownership_mid_station", job_id=job.id, worker_id=job.locked_by
            )
            return
        except Exception as exc:  # noqa: BLE001 — намеренно: станция может упасть
            # чем угодно (парсер, БД, дедлайн-таймаут) — политика ретрая/отказа
            # ниже одна и та же для любой причины, различать типы здесь незачем.
            if job.attempts < job.max_attempts:
                _log.warning("job_attempt_failed_will_retry", error=repr(exc))
                return
            _log.error("job_failed", error=repr(exc))
            if await _finish_fenced(deps, job, outcome="fail", error=str(exc)):
                await _notify_failure(deps, job, _public_failure_reason(exc))
        else:
            await _finish_fenced(deps, job, outcome="complete")
    finally:
        await trace.flush(job.id, hand_id=hand_id)
        structlog.contextvars.unbind_contextvars("job_id", "job_type")
