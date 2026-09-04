"""`LLM` + `PgLimiter` (спека §7) — на настоящем Postgres (`db_factory`, задача 14):
центральное утверждение лимитера — сериализация по СОЕДИНЕНИЮ (advisory-локи
Postgres — сессионные, не транзакционные), а значит и по процессу; внутри одной
отменяемой транзакции (`db`) это невыразимо в принципе, ровно как у `test_queue.py`
(задача 15). Ключ на моделях провайдера — `TestModel`/`FunctionModel` из PydanticAI:
в этом окружении нет ключа провайдера, тесты не имеют права стучаться в сеть
(ограничение задачи 16), а тестовые двойники дают именно то, что нужно — управляемый
результат и управляемый провал без единого HTTP-запроса.

`trace_id` — обязательный параметр `LLM.__call__` (см. докстринг `llm.py`), а
`llm_calls.trace_id` — FK NOT NULL на `traces.id`, у которой в свою очередь NOT NULL
`job_id`. Бриф задачи 16 показывает вызовы фасада без этого параметра — тот же
разрыв между буквой брифа и тем, что реально требует схема БД, что был у
`player_id=1, session_id=1` в брифе задачи 15 (см. `_make_scope` в `test_queue.py`):
там разрешился настоящими `player`/`session` через репозитории, здесь — тем же
приёмом, но до `job`/`trace` (`_make_trace_scope` ниже; репозитория на `jobs`/
`traces` вне ORM пока нет — как и `test_queue.py`, довольствуемся моделями напрямую).
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel
from pydantic_ai import BinaryContent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from harness.memory.models import Job, LlmCall, Trace
from harness.memory.repos import PlayersRepo, SessionsRepo
from harness.platform.config import Config, MissingEnvVar
from harness.platform.limiter import (
    _ADVISORY_LOCK_CLASS,
    PgLimiter,
    PgLimiterTimeout,
    PgLimiterUnlockFailed,
)
from harness.platform.llm import LLM, LLMProviderError, LLMSchemaError, _sniff_image_media_type

# Строки моделей ниже никогда не резолвятся в реального провайдера — во всех тестах
# `model_override` подменяет модель тестовым двойником PydanticAI ДО того, как
# `LLM` попытался бы построить `Agent` с ней; значения существуют только чтобы
# `Config` был валиден по форме (спека §7: `"<provider>:<model>"`).
cfg = Config(
    llm_vision_model="anthropic:claude-sonnet-test",
    llm_verdict_model="anthropic:claude-haiku-test",
    llm_max_concurrency=4,
    llm_max_per_minute=1000,
    database_url="unused-in-tests",
    telegram_token="unused-in-tests",
)


async def _make_trace_scope(session_factory, tg_user_id: int = 1) -> int:
    """Валидный `trace_id` для FK `llm_calls.trace_id` — цепочка player -> session
    -> job -> trace, все настоящие закоммиченные строки (лимитер и логирование
    должны быть видны с других соединений, см. модульный докстринг). `Job`/`Trace`
    собраны напрямую через ORM, а не через `JobsQueue`/будущий `platform/trace.py`
    (задача 18, ещё не существует) — здесь это голая scaffolding-цепочка FK, тот же
    приём, что `session.get(Job, ...)` в `test_queue.py`.
    """
    async with session_factory() as session:
        player = await PlayersRepo(session).get_or_create(tg_user_id=tg_user_id)
        session_row = await SessionsRepo(session).active_or_create(player.id)
        job = Job(type="deep_dive", player_id=player.id, session_id=session_row.id, payload={})
        session.add(job)
        await session.flush()
        trace = Trace(job_id=job.id)
        session.add(trace)
        await session.commit()
        return trace.id


async def fetch_all(session_factory, sql: str) -> list[tuple]:
    async with session_factory() as session:
        result = await session.execute(text(sql))
        return [tuple(row) for row in result.all()]


class Out(BaseModel):
    text: str


async def test_limiter_serializes_across_connections(db_factory):
    """Advisory-локи — кросс-СОЕДИНЕНИЕ, значит и кросс-процессно (бриф задачи 16
    дословно): два конкурентных `slot()` на одном `db_factory` (каждый вызов —
    своё соединение из пула) обязаны выполняться по очереди, не перекрываясь.
    """
    lim = PgLimiter(db_factory, max_concurrency=1, max_per_minute=1000)
    order: list[str] = []

    async def hold(tag: str) -> None:
        async with lim.slot():
            order.append(f"{tag}-in")
            await asyncio.sleep(0.2)
            order.append(f"{tag}-out")

    await asyncio.gather(hold("a"), hold("b"))
    assert order in (["a-in", "a-out", "b-in", "b-out"], ["b-in", "b-out", "a-in", "a-out"])


async def test_rate_window_blocks(db_factory, monkeypatch):
    """`max_per_minute` строк уже в окне -> `slot()` не проходит немедленно, а ждёт
    (спека §7: "если ≥ лимита — sleep с джиттером до входа в окно"). Чтобы не ждать
    реальную минуту, подменяем `_sleep` — модульную косвенность `limiter.py` над
    `asyncio.sleep` (fix round 1, small item 4: патч ИМЕННО этого атрибута, а не
    `asyncio.sleep` глобально — см. докстринг `_sleep` в `limiter.py`). Подмена не
    просто ускоряет паузу, а на первом вызове САМА состаривает одну из засеянных
    строк за пределы 60-секундного окна — тест доказывает и что лимитер
    ДЕЙСТВИТЕЛЬНО уходит в ожидание при полном окне (иначе подмена никогда не
    вызовется и `freed` останется `False`), и что он корректно возобновляется,
    когда окно освобождается.
    """
    trace_id = await _make_trace_scope(db_factory, tg_user_id=2)
    max_per_minute = 2
    seed_ids: list[int] = []
    async with db_factory() as session:
        for _ in range(max_per_minute):
            row = LlmCall(trace_id=trace_id, provider="anthropic", model="test", purpose="verdict_text")
            session.add(row)
            await session.flush()
            seed_ids.append(row.id)
        await session.commit()

    freed = False
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        nonlocal freed
        if not freed:
            async with db_factory() as session:
                await session.execute(
                    text("UPDATE llm_calls SET started_at = now() - interval '2 minutes' WHERE id = :id"),
                    {"id": seed_ids[0]},
                )
                await session.commit()
            freed = True
        await real_sleep(0)  # отдать управление event loop, не тратя реальный джиттер

    monkeypatch.setattr("harness.platform.limiter._sleep", fake_sleep)

    lim = PgLimiter(db_factory, max_concurrency=1, max_per_minute=max_per_minute)
    async with lim.slot():
        pass

    assert freed  # доказательство, что лимитер реально упёрся в полное окно и ждал


async def test_facade_validates_and_logs(db_factory):
    trace_id = await _make_trace_scope(db_factory, tg_user_id=3)
    llm = LLM(cfg, db_factory, model_override=TestModel())

    out, meta = await llm("verdict_text", Out, prompt="скажи привет", trace_id=trace_id)

    assert isinstance(out, Out)
    assert meta.tokens_in > 0 and meta.tokens_out > 0
    assert meta.latency_ms >= 0

    rows = await fetch_all(db_factory, "select purpose, status from llm_calls")
    assert rows == [("verdict_text", "ok")]


async def test_retry_on_schema_error_then_fail(db_factory):
    """`FunctionModel`, неизменно отдающая мусор (не проходит `output_type=Out`
    ни разу): 1 ретрай, затем `LLMSchemaError`; в `llm_calls` — ДВЕ строки со
    `status='schema_error'`, ни одной 'ok' (спека §7: "1 повтор, потом ошибка задачи").
    """
    trace_id = await _make_trace_scope(db_factory, tg_user_id=4)

    def garbage(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="это не JSON и не вызов инструмента")])

    llm = LLM(cfg, db_factory, model_override=FunctionModel(garbage))

    with pytest.raises(LLMSchemaError):
        await llm("verdict_text", Out, prompt="скажи привет", trace_id=trace_id)

    rows = await fetch_all(db_factory, "select status from llm_calls order by id")
    assert rows == [("schema_error",), ("schema_error",)]


async def test_backoff_retries_on_http_error_then_succeeds(db_factory, monkeypatch):
    """Бэкофф на 429/5xx (спека §7) — отдельный от схема-ретрая механизм: провайдер
    падает дважды с `ModelHTTPError(503)`, третья попытка проходит. Подменяем
    `_sleep` — модульную косвенность `llm.py` над `asyncio.sleep` (fix round 1,
    small item 4: патч `harness.platform.llm._sleep`, а не `asyncio.sleep`
    глобально), чтобы не ждать реальные 2^n секунд — подмена лишь ускоряет паузу,
    самого вызова не пропускает (`real_sleep(0)` всё равно отдаёт управление event
    loop), поэтому падение без бэкоффа (см. падение теста ниже при
    `_MAX_HTTP_ATTEMPTS=1` в самопроверке отчёта) по-прежнему ловится.
    """
    trace_id = await _make_trace_scope(db_factory, tg_user_id=5)
    attempts = {"n": 0}

    def flaky(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ModelHTTPError(status_code=503, model_name="test", body="upstream busy")
        return ModelResponse(parts=[ToolCallPart(tool_name="final_result", args={"text": "ok"})])

    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr("harness.platform.llm._sleep", fake_sleep)

    llm = LLM(cfg, db_factory, model_override=FunctionModel(flaky))
    out, _meta = await llm("verdict_text", Out, prompt="скажи привет", trace_id=trace_id)

    assert isinstance(out, Out) and out.text == "ok"
    assert attempts["n"] == 3
    assert len(sleeps) == 2  # пауза между 1->2 и 2->3 попытками, не перед первой

    rows = await fetch_all(db_factory, "select status from llm_calls order by id")
    assert rows == [("error",), ("error",), ("ok",)]


async def test_backoff_exhausted_raises_provider_error(db_factory, monkeypatch):
    """Симметричный случай: провайдер 429 на ВСЕХ `_MAX_HTTP_ATTEMPTS` попытках ->
    `LLMProviderError`, а не тихий провал и не бесконечный ретрай.
    """
    trace_id = await _make_trace_scope(db_factory, tg_user_id=6)

    def always_429(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=429, model_name="test", body="rate limited")

    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr("harness.platform.llm._sleep", fake_sleep)

    llm = LLM(cfg, db_factory, model_override=FunctionModel(always_429))
    with pytest.raises(LLMProviderError):
        await llm("verdict_text", Out, prompt="скажи привет", trace_id=trace_id)

    rows = await fetch_all(db_factory, "select status from llm_calls order by id")
    assert rows == [("error",), ("error",), ("error",)]


async def test_pinned_connection_survives_autocommit_statement_boundaries(db_factory):
    """Fix round 1, Important 3 — прямое доказательство механизма пиннинга, а не
    вывод из тайминга гонки: если `AUTOCOMMIT` (`engine.connect()` +
    `execution_options(isolation_level="AUTOCOMMIT")`) когда-нибудь перестанет
    держать физическое соединение закреплённым за одним `AsyncConnection` через
    несколько операторов подряд, мы молча возвращаемся к самому багу, который был
    найден и исправлен (см. докстринг `PgLimiter.slot()`, "Пиннинг соединения") —
    контроллер прямо потребовал эту проверку, потому что этот же класс ошибки уже
    случился один раз незамеченным.

    `pg_backend_pid()` — физическая личность соединения; операторы между первым и
    последним замером — включая успешный захват advisory-лока — каждый коммитится
    сам по себе под `AUTOCOMMIT` (в отличие от `AsyncSession.commit()`, который
    вернул бы соединение в пул). Отдельная проверка со ВТОРОГО, независимого
    соединения — что лок и правда ещё держится, а не просто pid не поменялся.
    """
    engine = db_factory.kw["bind"]
    conn = await engine.connect()
    conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
    try:
        pid_before = await conn.scalar(text("SELECT pg_backend_pid()"))
        acquired = await conn.scalar(
            text("SELECT pg_try_advisory_lock(:cls, :idx)"),
            {"cls": _ADVISORY_LOCK_CLASS, "idx": 0},
        )
        assert acquired
        await conn.execute(text("SELECT 1"))  # ещё один auto-commit-оператор во время удержания
        pid_after = await conn.scalar(text("SELECT pg_backend_pid()"))
        assert pid_before == pid_after

        async with db_factory() as probe:
            still_locked = not await probe.scalar(
                text("SELECT pg_try_advisory_lock(:cls, :idx)"),
                {"cls": _ADVISORY_LOCK_CLASS, "idx": 0},
            )
            assert still_locked
    finally:
        await conn.execute(
            text("SELECT pg_advisory_unlock(:cls, :idx)"), {"cls": _ADVISORY_LOCK_CLASS, "idx": 0}
        )
        await conn.close()


async def test_leaked_lock_self_heals_on_next_checkout(db_factory):
    """Fix round 1, Important 1: воркер падает посреди удержания слота (`finally`
    не успевает выполниться) — соединение возвращается в пул ЖИВЫМ, а advisory-лок
    остаётся висеть на нём (реентерабелен на уровне сессии). Без самолечения
    (`pg_advisory_unlock_all()` первым делом в `slot()`) следующий держатель ЭТОГО
    ЖЕ физического соединения получил бы `pg_try_advisory_lock` `True` не потому
    что слот свободен, а по наследству от мертвеца — то есть два логических
    держателя на одном индексе одновременно, ровно тот класс бага, что уже был
    найден и исправлен в этой задаче.

    Пул с `pool_size=1, max_overflow=0` — не тесту "может повезёт", а гарантия:
    следующий чекаут ОБЯЗАН получить то же самое (единственное) физическое
    соединение.

    Проверка "не зависло" одна НЕДОСТАТОЧНА: без самолечения `slot()` ТОЖЕ не
    виснет — реентерабельность advisory-лока на уровне сессии сама даёт `True`
    унаследованному держателю (тот же баг, другой симптом, не hang). Наблюдаемое
    отличие — не в том, зависает ли ЭТОТ `slot()`, а в том, что после его
    завершения остаётся: без самолечения счётчик реентерабельности так и
    остаётся на 1 (зомби так и не разблокировал, а "исцелившийся" держатель снял
    только СВОЙ уровень, 2->1, но не тот, что от зомби) — НЕЗАВИСИМОЕ третье
    соединение видит лок как всё ещё занятый. С самолечением счётчик обнулён ДО
    захвата, и после освобождения независимое соединение видит лок свободным.
    """
    dsn = db_factory.kw["bind"].url
    tight_engine = create_async_engine(dsn, pool_size=1, max_overflow=0)
    tight_factory = async_sessionmaker(tight_engine, expire_on_commit=False)
    try:
        zombie = await tight_engine.connect()
        zombie = await zombie.execution_options(isolation_level="AUTOCOMMIT")
        acquired = await zombie.scalar(
            text("SELECT pg_try_advisory_lock(:cls, :idx)"),
            {"cls": _ADVISORY_LOCK_CLASS, "idx": 0},
        )
        assert acquired
        await zombie.close()  # "падение" воркера — без unlock

        lim = PgLimiter(tight_factory, max_concurrency=1, max_per_minute=1000)
        async with asyncio.timeout(5):  # не повиснуть на все 120с _ACQUIRE_TIMEOUT_S, если баг вернулся
            async with lim.slot():
                pass

        # Независимое соединение (свой собственный движок, не единственное из
        # tight_engine) — единственный способ увидеть настоящее состояние лока
        # снаружи, не унаследовав реентерабельность той же сессии.
        async with db_factory() as outsider:
            freed_for_real = await outsider.scalar(
                text("SELECT pg_try_advisory_lock(:cls, :idx)"),
                {"cls": _ADVISORY_LOCK_CLASS, "idx": 0},
            )
            assert freed_for_real
            await outsider.execute(
                text("SELECT pg_advisory_unlock(:cls, :idx)"),
                {"cls": _ADVISORY_LOCK_CLASS, "idx": 0},
            )
    finally:
        await tight_engine.dispose()


async def test_acquire_gives_up_with_diagnostics_when_slot_never_frees(db_factory, monkeypatch):
    """Fix round 1, Important 2: без ограничения `_acquire_advisory_lock` ждал бы
    вечно и молча — утечка из Important 1 (или любая другая причина затора)
    вырождалась бы в неотличимый от нормальной работы вечный стопор. Держим
    единственный слот (K=1) занятым С ДРУГОГО соединения, которое НИКОГДА не
    отпускает лок за время теста, и проверяем, что `slot()` сдаётся
    `PgLimiterTimeout`, а не виснет. Оба порога укорочены monkeypatch — иначе тест
    честно ждал бы 120 реальных секунд.
    """
    monkeypatch.setattr("harness.platform.limiter._ACQUIRE_TIMEOUT_S", 0.3)
    monkeypatch.setattr("harness.platform.limiter._STALL_LOG_INTERVAL_S", 0.05)

    holder = await db_factory.kw["bind"].connect()
    holder = await holder.execution_options(isolation_level="AUTOCOMMIT")
    try:
        acquired = await holder.scalar(
            text("SELECT pg_try_advisory_lock(:cls, :idx)"),
            {"cls": _ADVISORY_LOCK_CLASS, "idx": 0},
        )
        assert acquired

        lim = PgLimiter(db_factory, max_concurrency=1, max_per_minute=1000)
        with pytest.raises(PgLimiterTimeout):
            async with lim.slot():
                pass
    finally:
        await holder.execute(
            text("SELECT pg_advisory_unlock(:cls, :idx)"), {"cls": _ADVISORY_LOCK_CLASS, "idx": 0}
        )
        await holder.close()


async def test_unlock_failure_raises_instead_of_swallowing(db_factory, monkeypatch):
    """Small item: `pg_advisory_unlock` возвращающий `false` — сигнал, что
    инвариант слота нарушен, не то, что можно молча проигнорировать (house-style
    `queue.py`: раскрывать нарушенный инвариант, не проглатывать булев результат).
    Подменяем сам SQL юнлока на заведомо возвращающий `false` — реальный сценарий
    рассинхрона (лок снят кем-то ещё до `finally`) труднодостижим детерминированно,
    а эта подмена бьёт точно в проверяемую ветку кода.
    """
    monkeypatch.setattr("harness.platform.limiter._UNLOCK_SQL", text("SELECT false"))
    lim = PgLimiter(db_factory, max_concurrency=1, max_per_minute=1000)
    with pytest.raises(PgLimiterUnlockFailed):
        async with lim.slot():
            pass


def test_sniff_image_media_type_png_and_jpeg():
    """Small item: захардкоженный `image/png` был неверен для реального входа —
    Telegram переотдаёт сжатые фото как JPEG (владелец продукта, fix round 1).
    Магические байты, не расширение файла — вызывающий передаёт голые `bytes`.
    """
    assert _sniff_image_media_type(b"\x89PNG\r\n\x1a\n" + b"...") == "image/png"
    assert _sniff_image_media_type(b"\xff\xd8\xff" + b"...") == "image/jpeg"
    with pytest.raises(ValueError):
        _sniff_image_media_type(b"not-an-image-at-all")


async def test_call_sniffs_jpeg_media_type_for_images(db_factory):
    """То же самое, но через публичный `LLM.__call__` end-to-end: JPEG-байты в
    `images=` обязаны долететь до модели как `BinaryContent(media_type='image/
    jpeg')`, не `'image/png'` — иначе задача 22 первым же реальным скрином из
    Telegram получает 400 от провайдера на vision-пути.
    """
    trace_id = await _make_trace_scope(db_factory, tg_user_id=7)
    seen_media_types: list[str] = []

    def inspect_images(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last = messages[-1]
        for part in last.parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, list):
                seen_media_types.extend(
                    c.media_type for c in part.content if isinstance(c, BinaryContent)
                )
        return ModelResponse(parts=[ToolCallPart(tool_name="final_result", args={"text": "ok"})])

    llm = LLM(cfg, db_factory, model_override=FunctionModel(inspect_images))
    jpeg_bytes = b"\xff\xd8\xff" + b"\x00" * 16
    await llm("vision_extract", Out, prompt="что на фото?", images=[jpeg_bytes], trace_id=trace_id)

    assert seen_media_types == ["image/jpeg"]


async def test_unexpected_exception_finalizes_row_as_error(db_factory):
    """Discretionary item: без третьего `except` в `_attempt` строка `llm_calls`
    осталась бы `status='started'` навсегда для сбоя, не опознанного ни как 429/5xx,
    ни как невалидная схема (обрыв соединения, таймаут чтения, что угодно ещё) —
    а у `llm_calls`, в отличие от `jobs`, нет reaper'а, который бы её когда-нибудь
    закрыл.
    """
    trace_id = await _make_trace_scope(db_factory, tg_user_id=8)

    def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ConnectionError("оборвалось на середине")

    llm = LLM(cfg, db_factory, model_override=FunctionModel(boom))
    with pytest.raises(ConnectionError):
        await llm("verdict_text", Out, prompt="скажи привет", trace_id=trace_id)

    rows = await fetch_all(db_factory, "select status from llm_calls")
    assert rows == [("error",)]


def test_config_from_env_reads_all_six_vars(monkeypatch):
    monkeypatch.setenv("LLM_VISION_MODEL", "anthropic:claude-sonnet-x")
    monkeypatch.setenv("LLM_VERDICT_MODEL", "anthropic:claude-haiku-x")
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "4")
    monkeypatch.setenv("LLM_MAX_PER_MINUTE", "60")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h/db")
    monkeypatch.setenv("TELEGRAM_TOKEN", "123:abc")

    loaded = Config.from_env()

    assert loaded.llm_vision_model == "anthropic:claude-sonnet-x"
    assert loaded.llm_verdict_model == "anthropic:claude-haiku-x"
    assert loaded.llm_max_concurrency == 4
    assert loaded.llm_max_per_minute == 60
    assert loaded.database_url == "postgresql+asyncpg://u:p@h/db"
    assert loaded.telegram_token == "123:abc"


def test_config_from_env_missing_var_fails_loudly(monkeypatch):
    """Отсутствующая переменная — явный `MissingEnvVar`, а не `KeyError` без
    контекста и не тихая подстановка дефолта (спека §7: смена лимита — дело
    окружения, а не источник для угадывания)."""
    monkeypatch.delenv("LLM_VISION_MODEL", raising=False)
    monkeypatch.setenv("LLM_VERDICT_MODEL", "anthropic:claude-haiku-x")
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "4")
    monkeypatch.setenv("LLM_MAX_PER_MINUTE", "60")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h/db")
    monkeypatch.setenv("TELEGRAM_TOKEN", "123:abc")

    with pytest.raises(MissingEnvVar, match="LLM_VISION_MODEL"):
        Config.from_env()
