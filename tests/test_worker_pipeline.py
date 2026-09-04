"""Воркер (задача 18): станции по типу задачи, чекпоинты резюме, идемпотентная
отправка, фенсинг, отказ после исчерпания попыток — на настоящем Postgres и
настоящем конвейере (`db_factory`, задача 14), с `FakeSender` вместо сети.

**Почему настоящий конвейер, а не моки.** `_run_hh_scan`/`_run_deep_dive` не
пересказывают решения `parse_file`/`normalize`/`enrich`/`analyze_hand`/
`scan_tournament` — они их вызывают. Подменить их моками значило бы проверить,
что воркер вызывает функции с ожидаемыми именами, а не что чекпоинты и резюме
действительно работают на реальных данных (тот же принцип, что у
`test_regression_grid.py`).

**`test_hh_scan_end_to_end` помечен `slow` (задача 13, тот же рулинг, что и
`test_scan.py`).** Он гонит `scan_tournament` по всем 146 рукам реальной
фикстуры — на холодном кэше эквити это минуты, не секунды (235.8с, задача 13).
Остальные тесты этого модуля используют небольшой СРЕЗ той же настоящей
фикстуры (первые несколько рук), а не синтетику и не полный файл: гарантии
резюме/идемпотентности/фенсинга/отказа не зависят от размера файла, а платить
за них временем полного скана незачем.

**Falsификация каждой гарантии — часть приёмки этой задачи (не только этого
файла).** Для `test_resume_skips_done_stations`, `test_send_idempotent`,
`test_failure_marks_failed_and_notifies`, `test_failure_before_max_attempts_
retries_silently`, `test_stale_worker_cannot_complete_reclaimed_job`,
`test_job_deadline_bounds_a_stuck_station` и `test_hh_scan_end_to_end`
(проверка `calc_cache`) соответствующее поведение `worker/pipeline.py` было
временно сломано и тест действительно покраснел — см. отчёт задачи 18
(`.superpowers/sdd/2026-08-29-poker-harness-v1/task-18-report.md`), раздел
"Falsификация".
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
from sqlalchemy import text

from harness.engine import enrich
from harness.memory.models import Job
from harness.memory.repos import HandsRepo, PlayersRepo, SessionsRepo, TournamentsRepo
from harness.normalizer import normalize
from harness.parsers.hh_parser import parse_file
from harness.platform.config import Config
from harness.platform.llm import LLM
from harness.platform.queue import JobsQueue
from harness.presentation import Msg
from harness.worker import pipeline as pipeline_module
from harness.worker.main import configure_logging
from harness.worker.pipeline import Deps, run_job
from tests.conftest import FIXTURE_DAILY, requires_fixtures

# Тестовый `Config` — те же плейсхолдеры, что в `test_llm_facade.py`: `LLM` внутри
# `Deps` собирается по-настоящему (тип `Deps.llm` — конкретный класс, не протокол),
# но станции v1-HH его не вызывают ни разу (интерфейсы задачи 18, дословно), так
# что модель никогда не резолвится и в сеть эти тесты не стучатся.
_TEST_CFG = Config(
    llm_vision_model="anthropic:claude-sonnet-test",
    llm_verdict_model="anthropic:claude-haiku-test",
    llm_max_concurrency=4,
    llm_max_per_minute=1000,
    database_url="unused-in-tests",
    telegram_token="unused-in-tests",
)


def _boom(*args: object, **kwargs: object) -> None:
    raise RuntimeError("эта функция не должна была вызываться в этом тесте")


class FakeSender:
    """Тестовый `Sender` (протокол задачи 18): список отправленного и
    отредактированного, без единого сетевого вызова.
    """

    def __init__(self) -> None:
        self.sent: list[Msg] = []
        self.edits: list[tuple[int, Msg]] = []
        self._next_id = 1000

    async def send(self, chat_id: int, msg: Msg) -> int:
        self.sent.append(msg)
        self._next_id += 1
        return self._next_id

    async def edit(self, chat_id: int, message_id: int, msg: Msg) -> None:
        self.edits.append((message_id, msg))


@pytest.fixture
def fake_sender() -> FakeSender:
    return FakeSender()


@pytest.fixture(scope="session")
def process_pool():
    """Один процессный пул на весь прогон — переиспользуется между тестами
    (спека §2: скан-расчёты обязаны идти через `run_in_executor(process_pool,
    ...)`, не через `None`/ThreadPoolExecutor по умолчанию). Пересоздавать его
    на каждый тест значило бы платить стоимость запуска подпроцесса (импорт
    всего пакета `harness` заново) многократно без всякой пользы для теста.
    """
    with ProcessPoolExecutor(max_workers=2) as pool:
        yield pool


@pytest.fixture
def queue(db_factory) -> JobsQueue:
    return JobsQueue(db_factory)


@pytest.fixture
def deps(db_factory, queue, fake_sender, process_pool) -> Deps:
    llm = LLM(_TEST_CFG, db_factory)
    return Deps(
        db_factory=db_factory,
        queue=queue,
        sender=fake_sender,
        llm=llm,
        process_pool=process_pool,
    )


async def _make_scope(db_factory, tg_user_id: int = 1) -> tuple[int, int]:
    """Валидные `player_id`/`session_id` — тот же приём, что в `test_queue.py`
    (задача 15): реальные закоммиченные строки через репозитории, а не голые
    числа, которые FK `jobs.player_id`/`jobs.session_id` (NOT NULL) не пропустят.
    `db_factory` truncate'ит таблицы с `RESTART IDENTITY` между тестами (см.
    `conftest.py`), поэтому первая пара в СВЕЖЕМ тесте всегда `(1, 1)` — то,
    что буквально называет бриф задачи (`enqueue_hh_scan(FIXTURE_DAILY,
    player_id=1, session_id=1)`), не совпадение и не хардкод здесь.
    """
    async with db_factory() as session:
        player = await PlayersRepo(session).get_or_create(tg_user_id=tg_user_id)
        session_row = await SessionsRepo(session).active_or_create(player.id)
        await session.commit()
        return player.id, session_row.id


async def enqueue_hh_scan(
    queue: JobsQueue, source_file: Path, *, player_id: int, session_id: int
) -> int:
    return await queue.enqueue(
        type="hh_scan",
        player_id=player_id,
        session_id=session_id,
        payload={"source_file": str(source_file)},
    )


async def enqueue_deep_dive(
    queue: JobsQueue, hand_no: str, *, player_id: int, session_id: int
) -> int:
    return await queue.enqueue(
        type="deep_dive",
        player_id=player_id,
        session_id=session_id,
        payload={"hand_no": hand_no},
    )


async def job_status(db_factory, job_id: int) -> str:
    async with db_factory() as session:
        job = await session.get(Job, job_id)
        assert job is not None, f"задача {job_id} не найдена"
        return job.status


async def count(db_factory, table: str) -> int:
    async with db_factory() as session:
        result = await session.execute(text(f"SELECT count(*) FROM {table}"))
        return result.scalar_one()


async def _seed_checkpointed_hands(
    db_factory, *, session_id: int, source_file: Path, n: int
) -> tuple[int, list]:
    """Прогоняет первые `n` рук `source_file` через настоящие `parse_file` →
    `normalize` → `enrich` и сохраняет их чекпоинтами `hands.raw/canonical/
    enriched` — то самое состояние, которое `_run_hh_scan` должен УМЕТЬ найти
    и не пересчитывать заново. Возвращает `(tournament_id, raw_hands)`.
    """
    raw_hands = parse_file(source_file.read_text("utf-8"), str(source_file))[:n]
    async with db_factory() as session:
        tournaments_repo = TournamentsRepo(session)
        hands_repo = HandsRepo(session)
        tournament_id = await tournaments_repo.create(
            session_id=session_id, source_file=str(source_file)
        )
        for idx, raw in enumerate(raw_hands):
            hand_id = await hands_repo.save_raw(
                session_id=session_id, tournament_id=tournament_id, raw=raw
            )
            canonical = normalize(raw).model_copy(update={"hand_index": idx})
            await hands_repo.save_canonical(hand_id, canonical)
            await hands_repo.save_enriched(hand_id, enrich(canonical))
        await session.commit()
    return tournament_id, raw_hands


# --- общий кэш расчётов: перенос через границу процесса ---------------------------


def test_equity_cache_seed_and_export_roundtrip():
    """Юнит-проверка самого механизма (без Postgres, без реального скана):
    `_scan_tournament_with_cache` (задача 18) полагается на то, что домешанное
    `equity_cache_seed()` значение ДЕЙСТВИТЕЛЬНО попадает в тот же кэш, который
    читает `_model_equity`, а не просто сохраняется где-то рядом с ним.
    Комплексная проверка (запись в саму `calc_cache`) — часть `test_hh_scan_
    end_to_end` выше (`calc_cache` растёт после реального скана).
    """
    from harness.analysis.preflop import equity_cache_export, equity_cache_seed

    equity_cache_seed({"__task18_test_key__": 0.4242})
    assert equity_cache_export()["__task18_test_key__"] == 0.4242


# --- hh_scan: сквозной прогон на реальной фикстуре (146 рук) ----------------------


@requires_fixtures
@pytest.mark.slow  # реальный скан 146 рук — минуты на холодном кэше (задача 13)
async def test_hh_scan_end_to_end(db_factory, fake_sender, queue, deps):
    player_id, session_id = await _make_scope(db_factory)
    jid = await enqueue_hh_scan(queue, FIXTURE_DAILY, player_id=player_id, session_id=session_id)
    job = await queue.claim("w1")
    assert job is not None
    await run_job(job, deps)

    assert (await job_status(db_factory, jid)) == "done"
    assert fake_sender.edits  # прогресс редактировался
    final = fake_sender.sent[-1]
    assert "bb" in final.text and any(
        b.callback_data.startswith("deep:") for row in final.buttons for b in row
    )
    assert await count(db_factory, "hands") == 146  # артефакты записаны
    assert await count(db_factory, "traces") == 1
    # Контроллерский рулинг задачи 18, п.1: эквити-кэш скана обязан осесть в
    # `calc_cache` (Postgres), а не только на диске этого процесса — иначе
    # следующий воркер (или следующий масштабированный контейнер) греет его
    # заново, и холодные 235.8с скана (задача 13) возвращаются на каждом old.
    assert await count(db_factory, "calc_cache") > 0


# --- резюме: чекпоинты пропускают уже сделанное ------------------------------------


@requires_fixtures
async def test_resume_skips_done_stations(db_factory, fake_sender, queue, deps, monkeypatch):
    """Спека §8.2: `hands.raw/canonical/enriched` предзаполнены (руки уже прошли
    все три станции в "прошлой попытке"), `jobs.payload["hands_saved"]` это
    подтверждает — задача обязана дойти до конца, ни разу не позвав `parse_file`.
    """
    player_id, session_id = await _make_scope(db_factory)
    tournament_id, raw_hands = await _seed_checkpointed_hands(
        db_factory, session_id=session_id, source_file=FIXTURE_DAILY, n=3
    )

    jid = await queue.enqueue(
        type="hh_scan",
        player_id=player_id,
        session_id=session_id,
        payload={
            "source_file": str(FIXTURE_DAILY),
            "tournament_id": tournament_id,
            "hands_saved": True,
        },
    )

    monkeypatch.setattr("harness.parsers.hh_parser.parse_file", _boom)

    job = await queue.claim("w1")
    assert job is not None
    await run_job(job, deps)  # должен пройти без парсера

    assert (await job_status(db_factory, jid)) == "done"  # чекпоинты работают — §8.2
    assert await count(db_factory, "hands") == len(raw_hands)  # ни одной новой строки


# --- отправка: повторный прогон редактирует, а не дублирует -----------------------


@requires_fixtures
async def test_send_idempotent(db_factory, fake_sender, queue, deps):
    """`payload` уже содержит `result_message_id` — воркер обязан отредактировать
    это сообщение (`Sender.edit`), а не отправить второе (§8.2).
    """
    player_id, session_id = await _make_scope(db_factory)
    _tournament_id, raw_hands = await _seed_checkpointed_hands(
        db_factory, session_id=session_id, source_file=FIXTURE_DAILY, n=1
    )
    hand_no = raw_hands[0].hand_no

    jid = await queue.enqueue(
        type="deep_dive",
        player_id=player_id,
        session_id=session_id,
        payload={"hand_no": hand_no, "result_message_id": 555},
    )

    job = await queue.claim("w1")
    assert job is not None
    await run_job(job, deps)

    assert (await job_status(db_factory, jid)) == "done"
    assert any(message_id == 555 for message_id, _ in fake_sender.edits)  # результат отредактирован
    assert len(fake_sender.sent) == 1  # только прогресс — результат дубля не породил


# --- отказ: провал после исчерпания попыток извещает игрока -----------------------


async def _seed_one_hand_deep_dive_job(
    db_factory, queue: JobsQueue, *, player_id: int, session_id: int
) -> tuple[int, Job]:
    _tournament_id, raw_hands = await _seed_checkpointed_hands(
        db_factory, session_id=session_id, source_file=FIXTURE_DAILY, n=1
    )
    jid = await enqueue_deep_dive(
        queue, raw_hands[0].hand_no, player_id=player_id, session_id=session_id
    )
    return jid, raw_hands[0]


@requires_fixtures
async def test_failure_marks_failed_and_notifies(db_factory, fake_sender, queue, deps, monkeypatch):
    player_id, session_id = await _make_scope(db_factory)
    jid, _raw = await _seed_one_hand_deep_dive_job(
        db_factory, queue, player_id=player_id, session_id=session_id
    )
    async with db_factory() as session:
        await session.execute(text("UPDATE jobs SET max_attempts = 1 WHERE id = :id"), {"id": jid})
        await session.commit()

    monkeypatch.setattr("harness.worker.pipeline.analyze_hand", _boom)

    job = await queue.claim("w1")
    assert job is not None
    assert job.attempts >= job.max_attempts  # последняя попытка — иначе тест не о том
    await run_job(job, deps)

    assert (await job_status(db_factory, jid)) == "failed"  # после max_attempts
    assert fake_sender.sent  # игроку отправлено уведомление
    assert "Не получилось разобрать" in fake_sender.sent[-1].text


@requires_fixtures
async def test_failure_before_max_attempts_retries_silently(
    db_factory, fake_sender, queue, deps, monkeypatch
):
    """Дополняет тест выше падением ДО последней попытки (`max_attempts=3` по
    умолчанию, `attempts=1` после первого `claim()`): статус обязан остаться
    `running` (задача ждёт `reap()`, см. докстринг `run_job`), а игрок — не
    получить уведомление о провале, которого по факту ещё нет.
    """
    player_id, session_id = await _make_scope(db_factory)
    jid, _raw = await _seed_one_hand_deep_dive_job(
        db_factory, queue, player_id=player_id, session_id=session_id
    )

    monkeypatch.setattr("harness.worker.pipeline.analyze_hand", _boom)

    job = await queue.claim("w1")
    assert job is not None
    assert job.attempts < job.max_attempts  # НЕ последняя попытка — иначе тест не о том
    await run_job(job, deps)

    assert (await job_status(db_factory, jid)) == "running"  # не failed — рано
    # Прогресс ("Считаю эквити…") отправляется ДО того, как `analyze_hand`
    # падает — это не уведомление о провале. Настоящего извещения (`failed_msg`)
    # быть не должно: провал ещё не окончательный.
    assert not any("Не получилось" in msg.text for msg in fake_sender.sent)


# --- фенсинг: переигранная попытка не переписывает нового владельца ---------------


@requires_fixtures
async def test_stale_worker_cannot_complete_reclaimed_job(db_factory, fake_sender, queue, deps):
    """Контроллерский рулинг задачи 18, п.2: `worker_id` из `claim()` обязан
    доходить до `complete()`/`fail()`. Сценарий — зомби-воркер "ожил" уже после
    того, как `reap()` вернул его задачу в очередь и её подхватил кто-то другой:
    результат зомби-попытки не должен затереть состояние нового владельца.
    """
    player_id, session_id = await _make_scope(db_factory)
    jid, _raw = await _seed_one_hand_deep_dive_job(
        db_factory, queue, player_id=player_id, session_id=session_id
    )

    stale_job = await queue.claim("w-stale")
    assert stale_job is not None

    async with db_factory() as session:
        await session.execute(
            text("UPDATE jobs SET locked_at = now() - interval '20 minutes' WHERE id = :id"),
            {"id": jid},
        )
        await session.commit()
    assert await queue.reap(older_than_minutes=10) == 1

    new_owner = await queue.claim("w-new")
    assert new_owner is not None
    assert new_owner.locked_by == "w-new"

    # "Ожившая" зомби-попытка старого воркера доигрывает всю станцию (найдёт руку,
    # посчитает разбор, даже пошлёт сообщение) — но не имеет права закрыть чужую
    # задачу. `run_job` не бросает исключение (фенсинг проглочен внутри неё).
    await run_job(stale_job, deps)

    async with db_factory() as session:
        row = await session.get(Job, jid)
        assert row is not None
        assert row.status == "running"  # не done — зомби не победил фенсинг
        assert row.locked_by == "w-new"  # владение нового воркера не тронуто


# --- дедлайн задачи: зависшая станция не висит вечно -------------------------------


@requires_fixtures
async def test_job_deadline_bounds_a_stuck_station(
    db_factory, fake_sender, queue, deps, monkeypatch
):
    """Контроллерский рулинг задачи 18, п.3: станция, которая не укладывается в
    `_JOB_DEADLINE_S`, не виснет до конца жизни воркера. `_dispatch` подменён на
    функцию, спящую 5 секунд, дедлайн урезан до 0.05с — `run_job` обязан
    вернуться на порядок быстрее (единицы, не секунды), пройдя ту же
    attempts-политику, что и обычное исключение (`asyncio.TimeoutError` ловится
    тем же `except`, что и любой другой сбой станции).
    """
    player_id, session_id = await _make_scope(db_factory)
    jid, _raw = await _seed_one_hand_deep_dive_job(
        db_factory, queue, player_id=player_id, session_id=session_id
    )
    async with db_factory() as session:
        await session.execute(text("UPDATE jobs SET max_attempts = 1 WHERE id = :id"), {"id": jid})
        await session.commit()

    monkeypatch.setitem(pipeline_module._JOB_DEADLINE_S, "deep_dive", 0.05)

    async def _hangs(*args: object, **kwargs: object) -> None:
        await asyncio.sleep(5)

    monkeypatch.setattr(pipeline_module, "_dispatch", _hangs)

    job = await queue.claim("w1")
    assert job is not None

    started = time.monotonic()
    await run_job(job, deps)
    elapsed = time.monotonic() - started

    assert elapsed < 2.0  # дедлайн реально оборвал 5-секундное ожидание
    assert (await job_status(db_factory, jid)) == "failed"
    assert "Не получилось разобрать" in fake_sender.sent[-1].text


# --- логирование: одна настройка на процесс маршрутизирует и лимитер --------------


def test_configure_logging_routes_limiter_warnings_through_structlog():
    """Контроллерский рулинг задачи 18, п.4: `platform/limiter.py` (задача 16)
    зовёт stdlib `logging.getLogger(__name__).warning(...)` — `configure_logging()`
    обязана сделать так, чтобы эта запись ДЕЙСТВИТЕЛЬНО прошла через тот же
    форматтер, что и структлог-события воркера, а не просто "не упасть при
    вызове". Без реальной проверки текста на выходе `logging.lastResort`
    (дефолтный обработчик stdlib logging, который печатает WARNING+ голым
    текстом в stderr, если рут не настроен) прошёл бы незамеченным — здесь
    вывод перехвачен в `io.StringIO()` (не в реальный `stderr`), поэтому если
    маршрутизация не настроена, `buf` останется пустым и тест покраснеет.
    """
    buf = io.StringIO()
    configure_logging(stream=buf)
    try:
        logging.getLogger("harness.platform.limiter").warning("лимитер: слот занят дольше обычного")
    finally:
        # Не оставляем рут сконфигурированным в JSON-на-StringIO для остальных
        # тестов сессии — тот же принцип, что и явный teardown фикстур.
        logging.getLogger().handlers = []

    output = buf.getvalue()
    assert "лимитер: слот занят дольше обычного" in output
    assert "warning" in output.lower()  # уровень добавлен структлог-процессором
