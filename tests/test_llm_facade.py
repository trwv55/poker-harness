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
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from sqlalchemy import text

from harness.memory.models import Job, LlmCall, Trace
from harness.memory.repos import PlayersRepo, SessionsRepo
from harness.platform.config import Config, MissingEnvVar
from harness.platform.limiter import PgLimiter
from harness.platform.llm import LLM, LLMProviderError, LLMSchemaError

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
    реальную минуту, подменяем `asyncio.sleep` внутри `limiter.py`: подмена не просто
    ускоряет паузу, а на первом вызове САМА состаривает одну из засеянных строк за
    пределы 60-секундного окна — тест доказывает и что лимитер ДЕЙСТВИТЕЛЬНО уходит
    в ожидание при полном окне (иначе подмена никогда не вызовется и `freed` останется
    `False`), и что он корректно возобновляется, когда окно освобождается.
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

    monkeypatch.setattr("harness.platform.limiter.asyncio.sleep", fake_sleep)

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
    `asyncio.sleep` в `llm.py`, чтобы не ждать реальные 2^n секунд — подмена лишь
    ускоряет паузу, самого вызова не пропускает (`real_sleep(0)` всё равно отдаёт
    управление event loop), поэтому падение без бэкоффа (см. падение теста ниже
    при `_MAX_HTTP_ATTEMPTS=1` в самопроверке отчёта) по-прежнему ловится.
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

    monkeypatch.setattr("harness.platform.llm.asyncio.sleep", fake_sleep)

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

    monkeypatch.setattr("harness.platform.llm.asyncio.sleep", fake_sleep)

    llm = LLM(cfg, db_factory, model_override=FunctionModel(always_429))
    with pytest.raises(LLMProviderError):
        await llm("verdict_text", Out, prompt="скажи привет", trace_id=trace_id)

    rows = await fetch_all(db_factory, "select status from llm_calls order by id")
    assert rows == [("error",), ("error",), ("error",)]


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
