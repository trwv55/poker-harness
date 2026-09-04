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
    return summary, equity_cache_export()


async def _sync_payload(session: AsyncSession, job_id: int, payload: dict[str, Any]) -> None:
    """Записать текущий чекпоинт `jobs.payload` и сразу закоммитить — единственное
    место в этом модуле, где обновление таблицы `jobs` не идёт через `JobsQueue`
    (тот не даёт менять `payload` без смены статуса, а резюме нужно делать это
    посреди `running`, не трогая статус). Коммитит сама, в отличие от репозиториев
    `memory/repos.py` (тем есть чей коммит ждать — вызывающий внутри той же сессии;
    здесь каждый вызов — самостоятельный чекпоинт, который обязан пережить крах
    сразу после записи, а не только в конце всей станции).
    """
    await session.execute(update(JobModel).where(JobModel.id == job_id).values(payload=payload))
    await session.commit()


async def _send_idempotent(
    deps: Deps,
    session: AsyncSession,
    job_id: int,
    payload: dict[str, Any],
    key: Literal["progress_message_id", "result_message_id"],
    chat_id: int,
    msg: Msg,
) -> None:
    """Отправить один раз, дальше — редактировать: `message_id` под `key` в `payload`
    делает повтор `run_job` (после `reap()` или ручного `resume()`) идемпотентным —
    `test_send_idempotent` доказывает это для результата, тот же путь у прогресса.
    """
    message_id = payload.get(key)
    if message_id is None:
        message_id = await deps.sender.send(chat_id, msg)
        payload[key] = message_id
        await _sync_payload(session, job_id, payload)
    else:
        await deps.sender.edit(chat_id, message_id, msg)


async def _ensure_progress(
    deps: Deps,
    session: AsyncSession,
    job_id: int,
    payload: dict[str, Any],
    chat_id: int,
    station: _Station,
) -> None:
    await _send_idempotent(
        deps,
        session,
        job_id,
        payload,
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
    async with deps.db_factory() as session:
        payload: dict[str, Any] = dict(job.payload)
        chat_id = await _chat_id(session, job.player_id)
        hands_repo = HandsRepo(session)
        tournaments_repo = TournamentsRepo(session)

        if payload.get("hands_saved"):
            tournament_id: int = payload["tournament_id"]
        else:
            async with trace.span("parse"):
                await _ensure_progress(deps, session, job.id, payload, chat_id, "parse")
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
                    await _sync_payload(session, job.id, payload)

                existing_raw = await hands_repo.count_by_tournament(tournament_id)
                for idx, raw in enumerate(raw_hands):
                    if idx < existing_raw:
                        continue  # чекпоинт: уже сохранена в прошлой попытке
                    await hands_repo.save_raw(
                        session_id=job.session_id, tournament_id=tournament_id, raw=raw
                    )
                    await session.commit()

            async with trace.span("validate"):
                await _ensure_progress(deps, session, job.id, payload, chat_id, "validate")
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
            await _sync_payload(session, job.id, payload)

        async with trace.span("analyze"):
            await _ensure_progress(deps, session, job.id, payload, chat_id, "analyze")
            hand_records = await hands_repo.list_by_tournament(tournament_id)
            enriched_hands = [r.enriched for r in hand_records if r.enriched is not None]

            cache_repo = CalcCacheRepo(session)
            prefix = f"equity_mc:{equity_cache_fingerprint()}:"
            seed = await cache_repo.get_all(prefix)

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
        await _send_idempotent(deps, session, job.id, payload, "result_message_id", chat_id, msg)
        await session.commit()


async def _run_deep_dive(job: JobModel, deps: Deps, trace: Trace, started_at: float) -> int | None:
    """Возвращает `hand_id` разобранной руки — `run_job` кладёт его в `traces.hand_id`
    (не в `_run_hh_scan`: там рук много, ни одна не выделена)."""
    async with deps.db_factory() as session:
        payload: dict[str, Any] = dict(job.payload)
        chat_id = await _chat_id(session, job.player_id)
        hand_no = payload["hand_no"]
        hands_repo = HandsRepo(session)
        analyses_repo = AnalysesRepo(session)

        async with trace.span("analyze"):
            await _ensure_progress(deps, session, job.id, payload, chat_id, "analyze")
            hand = await hands_repo.find_by_hand_no(job.session_id, hand_no)
            if hand is None:
                raise LookupError(f"рука {hand_no!r} не найдена в сессии {job.session_id}")
            if hand.enriched is None:
                raise LookupError(f"рука {hand_no!r} ещё не прошла чекпоинт enriched")

            await session.execute(
                update(JobModel).where(JobModel.id == job.id).values(hand_id=hand.id)
            )
            await session.commit()

            existing = await analyses_repo.get_by_hand(hand.id)
            result: AnalysisResult
            if existing is not None:
                result = existing.result  # чекпоинт: разбор уже посчитан прошлой попыткой
            else:
                result = analyze_hand(hand.enriched)
                await analyses_repo.save(hand_id=hand.id, result=result)
                await session.commit()

        elapsed_s = round(deps.clock() - started_at)
        zone = result.points[result.ranked[0]].zone if result.ranked else Zone.STRICT
        quota_left, quota_total = await _quota_numbers(session, job.player_id)
        msg = deep_dive_msg(result, elapsed_s, zone, quota_left, quota_total)
        await _send_idempotent(deps, session, job.id, payload, "result_message_id", chat_id, msg)
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
    молча (с логом) сознательно — `test_stale_worker_cannot_complete_reclaimed_job`
    доказывает, что переигранная задача при этом не переписывается зомби-попыткой.

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
    обработку провала: если отправка тоже не удалась (нет сети, чат заблокирован),
    задача всё равно обязана остаться `failed` в БД, а не повиснуть.
    """
    try:
        async with deps.db_factory() as session:
            current = await session.scalar(select(JobModel.payload).where(JobModel.id == job.id))
            payload: dict[str, Any] = dict(current or {})
            chat_id = await _chat_id(session, job.player_id)
            await _send_idempotent(
                deps, session, job.id, payload, "result_message_id", chat_id, failed_msg(reason)
            )
            await session.commit()
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
        except Exception as exc:  # noqa: BLE001 — намеренно: станция может упасть
            # чем угодно (парсер, БД, дедлайн-таймаут) — политика ретрая/отказа
            # ниже одна и та же для любой причины, различать типы здесь незачем.
            if job.attempts < job.max_attempts:
                _log.warning("job_attempt_failed_will_retry", error=repr(exc))
                return
            _log.error("job_failed", error=repr(exc))
            if await _finish_fenced(deps, job, outcome="fail", error=str(exc)):
                await _notify_failure(deps, job, str(exc))
        else:
            await _finish_fenced(deps, job, outcome="complete")
    finally:
        await trace.flush(job.id, hand_id=hand_id)
        structlog.contextvars.unbind_contextvars("job_id", "job_type")
