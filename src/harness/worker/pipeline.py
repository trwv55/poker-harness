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
общий бюджет здесь. С задачи 21 модель вызывается на станции `explain` (текст вердикта
и рассказ по турниру), и покрывает её тот же дедлайн станции — отдельного таймаута на
LLM-запрос по-прежнему нет.

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
from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from harness.analysis import analyze_hand
from harness.analysis.player_stats import player_stats_by_label
from harness.analysis.preflop import (
    equity_cache_export,
    equity_cache_fingerprint,
    equity_cache_seed,
)
from harness.analysis.scan import scan_tournament
from harness.analysis.tournament import tournament_report
from harness.calcs.routing import answer_question
from harness.contracts import (
    AnalysisResult,
    CanonicalHand,
    EnrichedHand,
    PlayerStats,
    RawHand,
    ScanSummary,
    TournamentReport,
    TournamentTextOut,
    ValidationStatus,
    VerdictTextOut,
    VisionCheck,
    Zone,
    is_judged,
)
from harness.engine import enrich
from harness.explanation import (
    UnfaithfulText,
    hand_replay,
    render_range_png,
    tournament_text,
    verdict_text,
)
from harness.memory.models import Job as JobModel
from harness.memory.models import Player
from harness.memory.repos import (
    AnalysesRepo,
    CalcCacheRepo,
    HandsRepo,
    QuotaRepo,
    TournamentsRepo,
)
from harness.normalizer import normalize
from harness.parsers import hh_parser
from harness.parsers.vision_adapter import (
    VisionOutcome,
    VisionReadFailed,
    can_apply_vision_answer,
    vision_extract,
)
from harness.platform.llm import LLM, LLMProviderError, LLMSchemaError
from harness.platform.queue import JobPreconditionFailed, JobsQueue
from harness.platform.trace import Clock, Trace
from harness.presentation import (
    Msg,
    Photo,
    deep_dive_msg,
    escalation_msg,
    failed_msg,
    hand_in_progress_msg,
    not_a_hand_msg,
    note_nicks_for_hand,
    progress_text,
    question_msg,
    question_refusal_msg,
    range_image_title,
    range_photos,
    scan_summary_msg,
    send_as_file_msg,
    tournament_report_msg,
    tournament_story_msg,
    vision_gave_up_msg,
)

__all__ = ["Deps", "Sender", "run_job"]

_log = structlog.get_logger(__name__)

# Станции, показываемые игроку прогрессом. "explain" появилась в задаче 21: с неё
# начинается второй (и последний) вызов модели в системе — текст вердикта.
_Station = Literal["ask", "read", "parse", "validate", "analyze", "explain"]

# Бюджет попытки по типу задачи — с запасом меньше десятиминутного окна reap()
# (см. модульный докстринг). `hh_scan` может обрабатывать сотни рук и считать эквити
# по файлу целиком — щедрее; `deep_dive` — одна рука, дёшево даже с запасом.
_JOB_DEADLINE_S: dict[str, float] = {
    "hh_scan": 480.0,
    # Вопрос — один вызов фасада с инструментами; обращений к провайдеру внутри
    # него больше одного (`test_a_tool_call_round_trip_logs_one_row`), а работы
    # с данными — выборка по готовым колонкам, без пересчёта эквити.
    "question": 120.0,
    "deep_dive": 120.0,
    # Скрин — два вызова модели в худшем случае (каскад) плюс расчёт одной руки.
    "screenshot_analyze": 240.0,
}
_DEFAULT_JOB_DEADLINE_S = 300.0


class SourceFileUnavailable(OSError):
    """Файл раздач, на который ссылается задача, не прочитался (round 5, Item I).

    Своё исключение, а не голый `FileNotFoundError`: причина отказа игроку
    выбирается по ТИПУ, и тип обязан значить ровно то, что мы утверждаем в
    тексте. Поднимается ровно в одном месте — там, где мы сами открываем
    `payload["source_file"]`, — и потому «файл раздач недоступен» под ним верно
    всегда, а не «обычно».
    """


class ScreenshotUnreadable(LookupError):
    """Картинки нет на диске или игрок не назвал свой ник в руме.

    Путь к картинке приходит в `jobs.payload` от бота (он же её туда и положил),
    поэтому `data_dir` воркера этой станции не нужен: том общий, а путь абсолютный.

    Свой тип по той же причине, что `SourceFileUnavailable`: причина отказа
    игроку выбирается по ТИПУ, и «скриншот не прочитать» под ним верно всегда.
    """


class HandDataMissing(LookupError):
    """Нужной руки (или её чекпоинта) нет в базе — round 5, Item I.

    Отдельный тип по той же причине, что и `SourceFileUnavailable`. До этой
    правки на его месте стоял базовый `LookupError`, а он — предок `KeyError` и
    `IndexError`: любой промах по словарю (`payload["hand_no"]`, `_STATION_TEXT
    [station]`, что угодно внутри парсера) выдавал игроку уверенное и конкретное
    «не нашли нужные данные по этой раздаче» про совсем другую поломку.
    """


# Закрытый набор причин отказа, которые честно показать игроку (fix round 1,
# Important 1). `str(exc)` бывает `FileNotFoundError: [Errno 2] ... '/data/hh/
# {hash}.txt'` (путь на диске), номер руки, текст ошибки SQLAlchemy — ровно то,
# что публикационная политика репозитория (fixtures/hh, номера рук как приватные
# данные игрока) запрещает публиковать, а правило единого голоса (presentation
# формулирует то, что видит игрок, а не необработанное исключение) запрещает
# показывать буквально. Внутренняя причина никуда не девается: `jobs.error`
# (queue.fail) хранит `str(exc)`, а лог (`_log.error`/`_log.warning` в `run_job`)
# — и `repr(exc)`, и трейсбек через `exc_info` (round 5, Item K.1: до этой правки
# трейсбека не было, и «полный и точный» было верно только про `jobs.error`).
# Это ops-данные, не то, что уходит в Sender.
#
# Ключи — доменные исключения ЭТОГО модуля, не встроенные типы (round 5, Item I):
# закрытый набор причин верен, а вот опознавать их по `FileNotFoundError`/
# `LookupError` было нельзя — второй накрывает `KeyError` и `IndexError`, то есть
# почти любой баг в коде. Исключение — `TimeoutError`: он называет ИСХОД («не
# уложились в срок»), а не место поломки, и остаётся верным независимо от того,
# кто его поднял.
_PUBLIC_FAILURE_REASON_DEFAULT = "внутренняя ошибка сервиса"
_PUBLIC_FAILURE_REASONS: tuple[tuple[type[BaseException], str], ...] = (
    (SourceFileUnavailable, "файл раздач недоступен"),
    (ScreenshotUnreadable, "скриншот не прочитать"),
    (VisionReadFailed, "не удалось прочитать скриншот"),
    (HandDataMissing, "не нашли нужные данные по этой раздаче"),
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
    тестовый двойник) ему не известна, только эти три метода.

    `send_photo` — отдельным методом, а не полем `Msg`: картинка уходит своим
    запросом Bot API (`sendPhoto`, multipart), и идемпотентность у неё своя —
    отредактировать уже отправленную картинку, как текст, нельзя.
    """

    async def send(self, chat_id: int, msg: Msg) -> int: ...

    async def edit(self, chat_id: int, message_id: int, msg: Msg) -> None: ...

    async def send_photo(self, chat_id: int, photo: Photo) -> int: ...


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
    data_dir: Path | None = None
    # Настроена ли дорогая ступень каскада зрения (`LLM_VISION_FALLBACK_MODEL`).
    # Флаг, а не имя модели: имя знает только `Config`, а станции нужно ровно
    # одно решение — звать вторую ступень или сразу спрашивать игрока.
    vision_fallback: bool = False


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
    """Записать текущий чекпоинт `jobs.payload` в обход `JobsQueue`: тот не даёт
    менять `payload` без смены статуса, а резюме нужно делать это посреди
    `running`, не трогая статус. Фенсинг — через `_fenced_update` (см. её
    докстринг).

    «Единственное такое место в модуле» здесь стояло неверно (round 5, Item K.6)
    — и, что показательно, было дописано раундом, который сам же и добавил
    остальные: `jobs` мимо очереди пишут ещё `_send_idempotent` (сохраняет
    `message_id` только что отправленного сообщения) и `_run_deep_dive`
    (`hand_id` разобранной руки). Общего у всех трёх ровно одно, и оно
    существенно: каждое идёт через `_fenced_update`, то есть ни одно не может
    записать в задачу, которой этот воркер уже не владеет.
    """
    await _fenced_update(session, job_id, worker_id, payload=payload)


async def _send_idempotent(
    deps: Deps,
    session: AsyncSession,
    job_id: int,
    worker_id: str | None,
    key: Literal[
        "progress_message_id",
        "result_message_id",
        "report_message_id",
        "story_message_id",
        "escalation_message_id",
    ],
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


# Сколько картинок диапазонов уже ушло игроку по этой задаче. Число, а не флаг:
# отправка идёт по одной, и повтор попытки обязан продолжить с той, на которой
# оборвалось, а не прислать заново всё (`_send_range_photos`).
_RANGE_PHOTOS_SENT = "range_photos_sent"


async def _saved_range_images(analyses_repo: AnalysesRepo, hand_id: int) -> list[str]:
    """Пути картинок из `analyses.range_images` — источник один и он в БД.

    Читается заново, а не берётся из переменной станции `explain`: у повторной
    попытки, которая нашла готовый разбор чекпоинтом, этой переменной нет вовсе,
    а картинки уже нарисованы.
    """
    record = await analyses_repo.get_by_hand(hand_id)
    return [] if record is None else (record.range_images or [])


async def _send_range_photos(
    deps: Deps,
    session: AsyncSession,
    job_id: int,
    worker_id: str | None,
    chat_id: int,
    photos: list[Photo],
) -> None:
    """Отправить матрицы диапазонов вслед за вердиктом — по одной, с отметкой.

    Матрица 13×13 — визуальное доказательство того, что числа настоящие
    (ARCHITECTURE, «Ценностное ядро»). До этой задачи она рисовалась
    (`_render_ranges`) и оставалась на диске: `Sender` не умел отправлять
    картинки вовсе.

    **Идемпотентность — счётчиком, а не флагом.** Отправленную картинку нельзя
    отредактировать, как текст (`_send_idempotent`), поэтому повтор попытки
    обязан знать, сколько уже ушло: `payload[_RANGE_PHOTOS_SENT]` растёт после
    КАЖДОЙ картинки, и обрыв на третьей из пяти стоит игроку двух недошедших,
    а не пяти дублей.

    **Пропавший файл не роняет разбор.** Картинка — дополнение к числам, а не
    они сами (то же правило, что у `_render_ranges`); файл мог исчезнуть вместе
    с томом, и падать из-за него после уже отправленного вердикта незачем.
    """
    if not photos:
        return
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
    already = int(payload.get(_RANGE_PHOTOS_SENT, 0))
    for index, photo in enumerate(photos[already:], start=already):
        if not Path(photo.path).exists():
            _log.warning("range_image_missing", job_id=job_id)
            continue
        await deps.sender.send_photo(chat_id, photo)
        payload[_RANGE_PHOTOS_SENT] = index + 1
        await _fenced_update(session, job_id, worker_id, payload=payload)


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
    интерактивных задач).

    Считает `QuotaRepo` (задача 19) — ТА ЖЕ реализация окна, по которой бот решает,
    пускать ли в разбор. До задачи 19 здесь стоял собственный счётчик-заглушка; два
    счётчика с одинаковым смыслом разошлись бы молча, и игрок увидел бы «осталось 3»
    ровно там, где ему отказали. Решение о допуске (`allowed`) воркер не читает
    намеренно: он уже взял задачу в работу и отказывать не вправе — отказ случается
    до постановки в очередь, не после.
    """
    quota = await QuotaRepo(session).check(player_id)
    return quota.left, quota.total


async def _tournament_stats(
    session: AsyncSession, tournament_id: int | None
) -> dict[str, PlayerStats] | None:
    """Частоты соседей по столу — или `None`, если считать их не по чему.

    Провенанс решает (спека §5.6): метка участника сквозная внутри турнира,
    поэтому на HH-входе частоты набираются по рукам турнира, а у скриншота
    `tournament_id` нет — и частот нет. `None`, не пустой словарь: «не считали»
    и «посчитали, вышло пусто» — разные вещи.
    """
    if tournament_id is None:
        return None
    hands = await HandsRepo(session).canonical_by_tournament(tournament_id)
    return player_stats_by_label(hands) if hands else None


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
                try:
                    raw_text = Path(source_file).read_text(encoding="utf-8")
                except OSError as exc:
                    # Единственное место, где мы сами открываем файл игрока, —
                    # значит единственное, про которое мы вправе сказать игроку
                    # «файл раздач недоступен» (round 5, Item I). Ловится `OSError`
                    # целиком, не только `FileNotFoundError`: права, битый том,
                    # оборванный NFS — для игрока это одно и то же «не читается».
                    raise SourceFileUnavailable(f"файл раздач не прочитан: {exc}") from exc
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

        # Отчёт по турниру (задача 23) — считается по уже готовым артефактам:
        # руки этого турнира и его сводка на руках, история игрока — два запроса
        # в `memory`. Ни одного расчёта эквити здесь нет, поэтому станция стоит
        # вне процессного пула и вне спана `analyze`.
        #
        # Пустой файл (`enriched_hands == []`) отчёта не получает: `tournament_
        # report` на нуле раздач отказывает, и правильно — describe там нечего.
        # Молчания при этом не возникает: сводка ниже уходит всегда и говорит
        # «Скан завершён: 0 рук» прямым текстом.
        if enriched_hands:
            report = tournament_report(
                enriched_hands,
                summary,
                player_tournaments=await hands_repo.player_hands_by_tournament(job.player_id),
                past_summaries=await tournaments_repo.player_scan_summaries(
                    job.player_id, exclude=tournament_id
                ),
            )
            # Рассказ словами — перед отчётом с числами, отчёт — перед сводкой:
            # сводка несёт кнопки «разобрать», и им место под последним
            # сообщением, а не отлистанными вверх. Рассказа может не быть вовсе
            # (модель недоступна либо её текст не прошёл проверку) — тогда игрок
            # получает те же два сообщения с числами, и это полноценный ответ.
            async with trace.span("explain"):
                payload = await _ensure_progress(
                    deps, session, job.id, worker_id, chat_id, "explain"
                )
                # Чекпоинт рассказа (ревью, раздел G). `story_message_id` в
                # payload означает, что рассказ УЖЕ отправлен прошлой попыткой:
                # звать модель снова значило бы заплатить второй раз и
                # переписать игроку уже прочитанное сообщение другим текстом —
                # модель не детерминирована, и это был бы не «тот же результат»,
                # как у остальных станций, а другой
                # (`test_a_repeat_scan_does_not_pay_for_the_story_twice`).
                story = (
                    None
                    if payload.get("story_message_id") is not None
                    else await _tournament_story(deps, trace, report)
                )
            if story is not None:
                await _send_idempotent(
                    deps,
                    session,
                    job.id,
                    worker_id,
                    "story_message_id",
                    chat_id,
                    tournament_story_msg(story),
                )
            await _send_idempotent(
                deps,
                session,
                job.id,
                worker_id,
                "report_message_id",
                chat_id,
                tournament_report_msg(report),
            )

        quota_left, quota_total = await _quota_numbers(session, job.player_id)
        msg = scan_summary_msg(summary, quota_left, quota_total)
        await _send_idempotent(deps, session, job.id, worker_id, "result_message_id", chat_id, msg)
        await session.commit()


# --- станция explain: слова поверх посчитанного (задача 21) -------------------------

# Отказы, после которых разбор ВСЁ РАВНО уходит игроку — без прозы, но с числами.
# Ни один из них не означает, что расчёт неверен: модель недоступна, ответила не по
# схеме или сказала то, чего расчёт не говорил. Числа, вердикты и зоны посчитаны
# кодом и от модели не зависят — прятать их из-за её ответа было бы хуже, чем
# показать разбор молча. Причина при этом не теряется: она в логе.
_EXPLANATION_FAILURES = (UnfaithfulText, LLMSchemaError, LLMProviderError)


async def _verdict_prose(deps: Deps, trace: Trace, result: AnalysisResult) -> VerdictTextOut | None:
    """Текст модели к разбору или `None`, если его не удалось получить честно."""
    try:
        return await verdict_text(deps.llm, result, trace_id=trace.trace_id)
    except _EXPLANATION_FAILURES as exc:
        _log.warning("verdict_text_unavailable", hand_no=result.hand_no, error=repr(exc))
        return None
    except Exception:  # noqa: BLE001 — см. `_EXPLANATION_FAILURES`: слова
        # необязательны, числа обязательны. Любой сбой слоя изложения (сюда
        # попадает и неверная конфигурация провайдера — `UserError` PydanticAI,
        # который не наследует наши типы) не имеет права отменить разбор, уже
        # посчитанный кодом. Причина уходит в лог целиком, с трейсбеком.
        _log.exception("verdict_text_crashed", hand_no=result.hand_no)
        return None


async def _tournament_story(
    deps: Deps, trace: Trace, report: TournamentReport
) -> TournamentTextOut | None:
    """Рассказ по турниру или `None` — по тем же правилам, что и текст разбора."""
    try:
        return await tournament_text(deps.llm, report, trace_id=trace.trace_id)
    except _EXPLANATION_FAILURES as exc:
        _log.warning("tournament_text_unavailable", error=repr(exc))
        return None
    except Exception:  # noqa: BLE001 — та же граница, что у `_verdict_prose`.
        _log.exception("tournament_text_crashed")
        return None


def _render_ranges(data_dir: Path | None, hand_id: int, result: AnalysisResult) -> list[str]:
    """Картинки диапазонов на диск; возвращает пути для `analyses.range_images`.

    Рисуются ТОЛЬКО допущения (`PointVerdict.assumption`) — то есть ровно те
    диапазоны, на которые опирается вывод в зоне «предполагая». У строгой точки
    показывать нечего: её вывод не зависит от догадки о поле, и картинка
    подразумевала бы обратное.

    Сбой записи не роняет разбор: картинка — дополнение к числам, а не они сами.
    """
    if data_dir is None:
        return []
    paths: list[str] = []
    directory = data_dir / "ranges"
    for index in result.ranked:
        point = result.points[index]
        if point.assumption is None:
            continue
        path = directory / f"{hand_id}-{point.dp_index}.png"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path.write_bytes(render_range_png(point.assumption.range, range_image_title(point)))
        except OSError as exc:
            _log.warning("range_image_failed", hand_id=hand_id, error=repr(exc))
            continue
        paths.append(str(path))
    return paths


def _hand_zone(result: AnalysisResult, not_checked: Sequence[str] = ()) -> Zone | None:
    """Зона доверия ВСЕЙ руки — из всех точек с названной линией, консервативно.

    Два правила, оба из CLAUDE.md («`strict` — только когда вывод не опирается на
    угаданный диапазон»):

    * вывода нет вовсе — зоны нет, `None`. Раньше здесь стоял `Zone.STRICT`, и
      игрок получал самую уверенную подпись продукта под сообщением «по этой
      раздаче точек с вердиктом нет»: строгость там, где не было вывода;
    * есть хоть одна точка `assuming` — вся рука `assuming`. Раньше зона бралась
      у ПЕРВОЙ точки `ranked`, поэтому шапка «зона: строго» могла стоять над
      строками, каждая из которых помечена «(по модели диапазонов)». Слабейшее
      звено определяет, чему можно верить, — не самое дорогое.

    Считаются точки, чей вывод игрок ВИДИТ: судимые из `ranked` плюс те, что
    цены не несут, но называют лучшую линию, — риверная точка с доказанным
    фолдом (`test_a_proven_river_fold_puts_its_zone_in_the_status_line`).
    Судимая точка вне `ranked` в расчёт не идёт: показывается ровно `ranked`.
    """
    shown = [result.points[idx] for idx in result.ranked]
    shown += [point for point in result.points if point.best_action and not is_judged(point)]
    zones = {point.zone for point in shown}
    if not zones:
        return None
    if not_checked:
        # Вход, часть которого проверить было нечем, не бывает «строгим»: под
        # этой подписью продукт обещает точный расчёт, а расчёт здесь опирается
        # на непроверенное чтение (реестр D1, ревью раунда 1, C).
        return Zone.ASSUMING
    return Zone.STRICT if zones == {Zone.STRICT} else Zone.ASSUMING


# --- станция скриншота (задача 22) -------------------------------------------

# Сколько раз подряд мы вправе спросить игрока по одной руке. Второй вопрос
# законен: первый мог уточнить банк, а следом не сойтись рассадка. Третий уже
# означает, что чтение не спасти уточнениями, и продолжать значило бы строить
# уверенный вывод поверх спорных чисел.
_MAX_ESCALATIONS = 2

# Проверки, чей провал лечится пересылкой того же скрина файлом: обе про карты.
# Сжатие Телеграма безопасно для чисел и опасно для мелких значков мастей
# (реестр, «Фото или файл»), поэтому просьба уместна ровно здесь.
_FILE_HINT_FIELDS = frozenset({"cards", "equity"})


def _unresolved_checks(raw: RawHand) -> list[VisionCheck]:
    """Контрольные суммы, которые всё ещё не сошлись, — включая после ответа игрока.

    Ответ, который код УМЕЛ подставить, закрывает свою проверку
    (`vision_adapter.apply_vision_answer`); ответ, который подставить не вышло,
    не закрывает ничего. Разница видна только здесь, и она решающая: без неё
    возобновлённая задача пропускает чтение, идёт в валидатор — а тот про
    неверную масть ничего не знает, дубля нет, деньги сходятся — и игрок
    получает вердикт «зона: строго» по спорному чтению (ревью раунда 1, R1).
    """
    return [check for check in (raw.vision.checks if raw.vision else []) if not check.passed]


def _escalation_question(field: str, checks: list[VisionCheck]) -> tuple[str, list[str], str]:
    """Вопрос, два варианта ответа и субъект — из самой непройденной проверки.

    Варианты не выдумываются: каждая контрольная сумма сравнивает ДВА
    независимых прочтения одного экрана, и оба и есть кнопки. Выбрать из двух
    прочитанных чисел игроку проще, чем набрать своё, — а «ввести вручную»
    остаётся третьей кнопкой (спека §8.3).

    Субъект нужен там, где спор про конкретного игрока: два варианта карт без
    ответа на «чьих» подставить некуда.
    """
    check = next((c for c in checks if c.name == field and not c.passed), None)
    if check is None:
        return (_VISION_QUESTIONS.get(field, f"Поле «{field}» распознано верно?"), [], "")
    return (_VISION_QUESTIONS.get(field, check.detail), check.options, check.subject)


# Имя поля кнопки в вердикте валидатора (`engine.validation`). Продублировано
# строкой, а не импортом приватного имени чужого модуля: разойдутся — увидит
# тест `test_the_validator_asks_about_the_button_with_the_nicknames_it_read`.
_VALIDATOR_FIELD_BUTTON = "button"

_VISION_QUESTIONS: dict[str, str] = {
    "pot": "Банк на этом скрине распознан верно?",
    "button": "Фишка дилера стоит на том игроке?",
    "cards": "Карты распознаны верно?",
    "equity": "Карты распознаны верно?",
    "hero": "Кто из них вы?",
    "stacks": "Стеки распознаны верно?",
}


async def _ask_player(
    deps: Deps,
    session: AsyncSession,
    job: JobModel,
    payload: dict[str, Any],
    chat_id: int,
    *,
    field: str,
    question: str,
    options: list[str],
    subject: str = "",
) -> None:
    """Задать вопрос и отпустить воркера — точка возврата уже зафиксирована.

    Спека §8.3 дословно: сообщение с кнопками, `await_user`, освобождение. Ответ
    ловит бот, он же пишет ground truth в `eval_cases`, патчит `hands.raw`,
    сбрасывает чекпоинты ниже и возвращает задачу в очередь.
    """
    worker_id = job.locked_by
    payload = await _send_idempotent(
        deps,
        session,
        job.id,
        worker_id,
        "escalation_message_id",
        chat_id,
        escalation_msg(job.id, field, question, options),
    )
    if field in _FILE_HINT_FIELDS:
        await deps.sender.send(chat_id, send_as_file_msg())
    payload["escalation_field"] = field
    payload["escalation_options"] = options
    payload["escalation_subject"] = subject
    payload["escalations"] = payload.get("escalations", 0) + 1
    payload.pop("manual_entry", None)
    await session.commit()
    # `escalation_message_id` СНИМАЕТСЯ вместе с переходом в `awaiting_user`, а
    # не раньше: между отправкой и `await_user` попытка может умереть, и тогда
    # повтор обязан увидеть уже отправленный вопрос, а не задать его второй раз
    # (ревью раунда 1, F). Ключ нужен ровно до этой границы и не дальше — иначе
    # следующий вопрос отредактировал бы предыдущий вместо нового сообщения.
    payload.pop("escalation_message_id", None)
    await deps.queue.await_user(job.id, payload, worker_id=worker_id)


def _validator_options(field: str, raw: RawHand) -> list[str]:
    """Варианты ответа на вопрос ВАЛИДАТОРА — их у него, в отличие от сверок, нет.

    Контрольная сумма адаптера сравнивает два прочтения и обоими и отвечает;
    валидатор сравнивает прочтение с правилами покера, и второго прочтения у
    него не бывает. Кнопки поэтому берутся из того, что уже прочитано с экрана:
    для кнопки дилера это список ников — игроку остаётся показать, у кого она
    стояла на самом деле.
    """
    if field != _VALIDATOR_FIELD_BUTTON or raw.vision is None:
        return []
    return list(raw.vision.hero_candidates)


async def _give_up(
    deps: Deps,
    session: AsyncSession,
    job: JobModel,
    chat_id: int,
    fields: Sequence[str],
) -> None:
    """Прекратить разбор с честным текстом — и назвать путь дальше, если он есть.

    Молча разобрать руку со спорными числами нельзя: вывод поверх неизвестного
    хуже отсутствия вывода (CLAUDE.md). Если спор про карты, у игрока есть
    рабочий следующий шаг — прислать тот же экран файлом, без сжатия.
    """
    await _send_idempotent(
        deps,
        session,
        job.id,
        job.locked_by,
        "result_message_id",
        chat_id,
        vision_gave_up_msg(),
    )
    if any(field in _FILE_HINT_FIELDS for field in fields):
        await deps.sender.send(chat_id, send_as_file_msg())
    await session.commit()


async def _read_screen(
    job: JobModel, deps: Deps, trace: Trace, session: AsyncSession, payload: dict[str, Any]
) -> VisionOutcome:
    """Один вызов адаптера плюс перенос ступеней каскада в трейс.

    Ступени возвращаются значением, а не пишутся адаптером: конвейерные пакеты
    про трейс не знают (правило зависимостей CLAUDE.md). Перенести их обязана
    станция — иначе в трейсе осталось бы «зрение отработало», а на каком
    переходе появилось расхождение, видно бы не было.
    """
    player = await session.get(Player, job.player_id)
    if player is None or not player.gg_nickname:
        raise ScreenshotUnreadable("игрок не назвал свой ник в руме")
    path = Path(payload["image_file"])
    try:
        image = path.read_bytes()
    except OSError as exc:
        raise ScreenshotUnreadable("файл скриншота недоступен") from exc

    outcome = await vision_extract(
        deps.llm,
        image,
        gg_nickname=player.gg_nickname,
        trace_id=trace.trace_id,
        source_ref=payload.get("image_hash", ""),
        image_hash=payload.get("image_hash"),
        fallback_available=deps.vision_fallback,
    )
    for hop in outcome.hops:
        trace.record(
            f"vision:{hop.role}",
            model=hop.model,
            failed_checks=hop.failed_checks,
            error=hop.error,
        )
    return outcome


async def _run_screenshot(job: JobModel, deps: Deps, trace: Trace, started_at: float) -> int | None:
    """Скрин -> разбор: чтение моделью, проверки, эскалация, ядро, слова.

    **Чекпоинт зрения — `hands.raw`**, и он же гасит петлю эскалаций: после
    ответа игрока задача возвращается сюда, видит уже сохранённую руку и модель
    больше не зовёт. Иначе каждый ответ игрока оплачивался бы новым чтением,
    контрольные суммы падали бы на том же месте, и вопрос повторялся бы вечно.
    """
    worker_id = job.locked_by
    async with deps.db_factory() as session:
        payload: dict[str, Any] = dict(job.payload)
        chat_id = await _chat_id(session, job.player_id)
        hands_repo = HandsRepo(session)
        analyses_repo = AnalysesRepo(session)
        hand_id: int | None = payload.get("hand_id")

        async with trace.span("read"):
            payload = await _ensure_progress(deps, session, job.id, worker_id, chat_id, "read")
            if hand_id is None:
                outcome = await _read_screen(job, deps, trace, session, payload)
                if outcome.hand_in_progress:
                    # Рука ещё идёт — станция чтения и есть та станция, где такой
                    # экран останавливается (решение владельца 2026-09-09). Ни
                    # `hands.raw`, ни вопроса игроку: разбирать нечего, а всё
                    # ниже по конвейеру считает по сыгранным действиям, которых
                    # на этом экране ещё нет.
                    await _send_idempotent(
                        deps,
                        session,
                        job.id,
                        worker_id,
                        "result_message_id",
                        chat_id,
                        hand_in_progress_msg(),
                    )
                    await session.commit()
                    return None
                if outcome.raw is None:
                    # Отказ модели — это ответ, а не сбой: экран не был раздачей.
                    await _send_idempotent(
                        deps,
                        session,
                        job.id,
                        worker_id,
                        "result_message_id",
                        chat_id,
                        not_a_hand_msg(outcome.refusal or "не похоже на раздачу"),
                    )
                    await session.commit()
                    return None
                hand_id = await hands_repo.save_raw(session_id=job.session_id, raw=outcome.raw)
                await session.commit()
                payload["hand_id"] = hand_id
                await _fenced_update(session, job.id, worker_id, hand_id=hand_id, payload=payload)
                if outcome.escalate:
                    # Спрашивать можно только о том, что код умеет подставить.
                    # Провалиться может и `stacks` (поправка на обрезку), и
                    # `equity`, и обе сверки рассадки — вопрос по ним стоил бы
                    # игроку внимания и всё равно кончился бы отказом после
                    # ответа (ревью раунда 2, F2).
                    failed = [check.name for check in outcome.failed]
                    field = next(
                        (name for name in failed if can_apply_vision_answer(name)), None
                    )
                    if field is None:
                        await _give_up(deps, session, job, chat_id, failed)
                        return hand_id
                    question, options, subject = _escalation_question(field, outcome.checks)
                    await _ask_player(
                        deps,
                        session,
                        job,
                        payload,
                        chat_id,
                        field=field,
                        question=question,
                        options=options,
                        subject=subject,
                    )
                    return hand_id

        record = await hands_repo.get(hand_id)
        unresolved = _unresolved_checks(record.raw)
        if unresolved:
            # Сюда попадает только возобновлённая задача: на первом проходе
            # непройденная проверка уходит вопросом игроку и возвращается выше.
            # Значит, ответ расхождение не закрыл — и разбирать эту руку нельзя.
            await _give_up(deps, session, job, chat_id, [c.name for c in unresolved])
            return hand_id

        async with trace.span("validate"):
            await _ensure_progress(deps, session, job.id, worker_id, chat_id, "validate")
            canonical: CanonicalHand = record.canonical or normalize(record.raw)
            enriched = record.enriched or enrich(canonical)
            await hands_repo.save_canonical(hand_id, canonical)
            await hands_repo.save_enriched(hand_id, enriched)
            await session.commit()
            if enriched.verdict.status is ValidationStatus.ESCALATE:
                answerable = [
                    (field, question)
                    for field, question in zip(
                        enriched.verdict.fields, enriched.verdict.questions, strict=True
                    )
                    if can_apply_vision_answer(field)
                ]
                # Вопрос по полю, ответ на которое подставить некуда, — мёртвый:
                # игрок отвечает, ответ ложится в eval-датасет, рука остаётся
                # прежней, и следующий проход упирается в то же расхождение
                # (ревью раунда 1, R3). Такие поля не спрашиваем вовсе.
                if not answerable or payload.get("escalations", 0) >= _MAX_ESCALATIONS:
                    await _give_up(deps, session, job, chat_id, enriched.verdict.fields)
                    return hand_id
                field, question = answerable[0]
                await _ask_player(
                    deps,
                    session,
                    job,
                    payload,
                    chat_id,
                    field=field,
                    question=question,
                    options=_validator_options(field, record.raw),
                )
                return hand_id

        async with trace.span("analyze"):
            await _ensure_progress(deps, session, job.id, worker_id, chat_id, "analyze")
            existing = await analyses_repo.get_by_hand(hand_id)
            if existing is not None:
                result = existing.result
            else:
                loop = asyncio.get_running_loop()
                result, _exported = await loop.run_in_executor(
                    deps.process_pool, _analyze_hand_with_cache, enriched, {}
                )
                await analyses_repo.save(
                    hand_id=hand_id,
                    result=result,
                    decision_points=enriched.report.decision_points,
                )
                await session.commit()

        async with trace.span("explain"):
            await _ensure_progress(deps, session, job.id, worker_id, chat_id, "explain")
            saved = existing.verdict_text if existing is not None else None
            if saved is not None:
                verdict = VerdictTextOut.model_validate_json(saved)
            else:
                verdict = await _verdict_prose(deps, trace, result)
                images = _render_ranges(deps.data_dir, hand_id, result)
                await analyses_repo.set_explanation(
                    hand_id=hand_id,
                    verdict_text=None if verdict is None else verdict.model_dump_json(),
                    range_images=images,
                )
                await session.commit()

        quota_left, quota_total = await _quota_numbers(session, job.player_id)
        replay = hand_replay(
            enriched,
            # Цена решения печатается, только если её кто-то вынес: у пустого
            # `ranked` `total_ev_loss_bb` — умолчание 0.0, а не измеренный ноль.
            ev_loss_bb=result.total_ev_loss_bb if result.ranked else None,
            stats=None,  # скрин: одна рука, знаменателя нет
        )
        msg = deep_dive_msg(
            result,
            round(deps.clock() - started_at),
            _hand_zone(result, enriched.verdict.not_checked),
            quota_left,
            quota_total,
            verdict=verdict,
            replay=replay,
            not_checked=enriched.verdict.not_checked,
            note_nicks=note_nicks_for_hand(enriched.hand),
        )
        await _send_idempotent(deps, session, job.id, worker_id, "result_message_id", chat_id, msg)
        await session.commit()
        await _send_range_photos(
            deps,
            session,
            job.id,
            worker_id,
            chat_id,
            range_photos(await _saved_range_images(analyses_repo, hand_id), result),
        )
        await session.commit()
    return hand_id


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
                raise HandDataMissing(f"рука {hand_no!r} не найдена в сессии {job.session_id}")
            if hand.enriched is None:
                raise HandDataMissing(f"рука {hand_no!r} ещё не прошла чекпоинт enriched")

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

                await analyses_repo.save(
                    hand_id=hand.id,
                    result=result,
                    decision_points=hand.enriched.report.decision_points,
                )
                await session.commit()

        async with trace.span("explain"):
            await _ensure_progress(deps, session, job.id, worker_id, chat_id, "explain")
            saved = existing.verdict_text if existing is not None else None
            if saved is not None:
                # Чекпоинт станции: слова уже сказаны прошлой попыткой — второй раз
                # за них не платим (спека §8.2, тот же принцип, что у `hands.*`).
                verdict = VerdictTextOut.model_validate_json(saved)
            else:
                verdict = await _verdict_prose(deps, trace, result)
                # Картинки диапазонов — выход КОДА, и от того, ответила ли
                # модель, они не зависят: сохраняются всегда (ревью, раздел G;
                # прежде отказ модели выбрасывал уже нарисованные файлы).
                images = _render_ranges(deps.data_dir, hand.id, result)
                await analyses_repo.set_explanation(
                    hand_id=hand.id,
                    verdict_text=None if verdict is None else verdict.model_dump_json(),
                    range_images=images,
                )
                await session.commit()

        elapsed_s = round(deps.clock() - started_at)
        zone = _hand_zone(result, hand.enriched.verdict.not_checked)
        quota_left, quota_total = await _quota_numbers(session, job.player_id)
        replay = hand_replay(
            hand.enriched,
            ev_loss_bb=result.total_ev_loss_bb if result.ranked else None,
            stats=await _tournament_stats(session, hand.tournament_id),
        )
        msg = deep_dive_msg(
            result,
            elapsed_s,
            zone,
            quota_left,
            quota_total,
            verdict=verdict,
            replay=replay,
            not_checked=hand.enriched.verdict.not_checked,
            note_nicks=note_nicks_for_hand(hand.enriched.hand),
        )
        await _send_idempotent(deps, session, job.id, worker_id, "result_message_id", chat_id, msg)
        await session.commit()
        await _send_range_photos(
            deps,
            session,
            job.id,
            worker_id,
            chat_id,
            range_photos(await _saved_range_images(analyses_repo, hand.id), result),
        )
        await session.commit()

        hand_id = hand.id

    return hand_id


async def _run_question(job: JobModel, deps: Deps, trace: Trace) -> None:
    """Вопрос игрока: модель выбирает расчёт, код считает, игрок получает подпись.

    Чекпоинта у этой станции нет: повторная попытка зовёт модель заново. Копить
    было бы нечего — весь результат станции это одно сообщение, а дубля его не
    будет и так (`_send_idempotent`).

    Отказ модели (провайдер, схема) сюда не перехватывается, в отличие от
    `_verdict_prose`: там числа посчитаны и без слов, здесь без вызова модели
    нет ни расчёта, ни ответа — задача честно падает, и игрок получает
    `failed_msg`.
    """
    worker_id = job.locked_by
    async with deps.db_factory() as session:
        payload: dict[str, Any] = dict(job.payload)
        chat_id = await _chat_id(session, job.player_id)
        async with trace.span("ask"):
            await _ensure_progress(deps, session, job.id, worker_id, chat_id, "ask")
            outcome = await answer_question(
                session,
                deps.llm,
                player_id=job.player_id,
                session_id=job.session_id,
                question=payload["question"],
                trace_id=trace.trace_id,
            )
        msg = (
            question_refusal_msg()
            if outcome.result is None
            else question_msg(outcome.result, outcome.prose)
        )
        await _send_idempotent(deps, session, job.id, worker_id, "result_message_id", chat_id, msg)
        await session.commit()


async def _dispatch(job: JobModel, deps: Deps, trace: Trace, started_at: float) -> int | None:
    if job.type == "question":
        await _run_question(job, deps, trace)
        return None
    if job.type == "hh_scan":
        await _run_hh_scan(job, deps, trace)
        return None
    if job.type == "deep_dive":
        return await _run_deep_dive(job, deps, trace, started_at)
    if job.type == "screenshot_analyze":
        return await _run_screenshot(job, deps, trace, started_at)
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
    очереди.

    **Наружу не летит ни одно `Exception` — и теперь это обеспечено, а не заявлено
    (round 5, Item E).** Обещание стояло здесь и раньше, но было ложным: `finally:
    await trace.flush(...)` делал ввод-вывод в БД вообще без `try`, а
    `_finish_fenced` ловит только `JobPreconditionFailed` — `LookupError` из
    `complete()`/`fail()` или любая ошибка SQLAlchemy улетали в цикл воркера.
    Теперь всё тело, включая `flush`, обёрнуто внешним `except Exception`, который
    логирует крах с трейсбеком; задача остаётся `running` и достаётся `reap()` —
    честная деградация вместо остановленной корутины воркера.

    `asyncio.CancelledError` — намеренное и единственное исключение из этого
    правила: она `BaseException`, а не `Exception`, и проглотить её значило бы
    сломать останов воркера по SIGTERM (`worker/main.py`), где отмена и есть
    штатный способ свернуть работу. Наружу она обязана уйти.

    Обещание не отменяет `except` в самом цикле воркера: цикл, чьё выживание
    держится на докстринге соседнего модуля, — это не гарантия. Оба слоя нужны.
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
        # Строка `traces` — ДО станций (round 5, Item D): на неё ссылается
        # `llm_calls.trace_id` (FK NOT NULL), а `LLM` пишет свою строку до
        # обращения к модели. См. модульный докстринг `platform/trace.py`.
        await trace.open(job.id)
        try:
            deadline = _JOB_DEADLINE_S.get(job.type, _DEFAULT_JOB_DEADLINE_S)
            started_at = deps.clock()
            try:
                hand_id = await asyncio.wait_for(
                    _dispatch(job, deps, trace, started_at), timeout=deadline
                )
            except JobPreconditionFailed:
                _log.warning(
                    "job_fencing_lost_ownership_mid_station",
                    job_id=job.id,
                    worker_id=job.locked_by,
                )
                return
            except Exception as exc:  # noqa: BLE001 — намеренно: станция может упасть
                # чем угодно (парсер, БД, дедлайн-таймаут) — политика ретрая/отказа
                # ниже одна и та же для любой причины, различать типы здесь незачем.
                # `exc_info=exc` обязателен (round 5, Item K.1): без него в лог
                # уходил один `repr(exc)` без места падения — то есть ровно тот
                # симптом, который чинила задача 19 и который её собственный
                # докстринг (`platform/logs.py`) числил среди уже исправленных.
                if job.attempts < job.max_attempts:
                    _log.warning("job_attempt_failed_will_retry", error=repr(exc), exc_info=exc)
                    return
                _log.error("job_failed", error=repr(exc), exc_info=exc)
                if await _finish_fenced(deps, job, outcome="fail", error=str(exc)):
                    await _notify_failure(deps, job, _public_failure_reason(exc))
            else:
                await _finish_fenced(deps, job, outcome="complete")
        finally:
            await trace.flush(hand_id=hand_id)
    except Exception:  # noqa: BLE001 — см. докстринг: это и есть та граница,
        # на которой обещание «наружу ничего не летит» становится правдой.
        _log.exception("job_run_crashed", job_id=job.id)
    finally:
        structlog.contextvars.unbind_contextvars("job_id", "job_type")
