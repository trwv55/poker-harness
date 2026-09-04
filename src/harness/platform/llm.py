"""`LLM` — единственная дверь к моделям во всей системе (спека §7): `await llm(purpose,
schema, prompt=..., images=...) -> (schema_instance, meta)`. Всё, что делает этот
фасад отдельным модулем, а не тонкой обёрткой над `pydantic_ai.Agent`, — три вещи,
которые спека прямо запрещает откладывать («заложить сразу» из SCALING.md):

1. **Лимиты — кросс-процессные, через `PgLimiter`** (`platform/limiter.py`, задача 16):
   одновременность и темп проверяются в Postgres на каждую попытку, а не в памяти
   этого процесса. Внутрипроцессный `asyncio.Semaphore` (`self._local_semaphore`)
   здесь тоже есть, но ровно как дешёвый предфильтр — не даёт этому процессу самому
   забить себе очередь на лимитер, когда он уже выбрал свою локальную долю слотов;
   источником истины не является (спека §7 дословно).
2. **Каждая попытка обращения к модели — своя строка `llm_calls`**, вставленная ДО
   вызова (`status='started'`) и обновлённая по завершении. "Попытка" здесь — именно
   одно HTTP-обращение к провайдеру: и бэкофф-ретрай на 429/5xx, и ретрай на
   невалидную схему — это РАЗНЫЕ попытки, каждая логируется отдельно, иначе окно
   темпа (`PgLimiter`, считает по `llm_calls`) занижает нагрузку ровно там, где она
   реальна.
3. **Два независимых механизма ретрая**, оба из спеки §7 дословно:
   - бэкофф с джиттером на 429/5xx (`ModelHTTPError`) — до `_MAX_HTTP_ATTEMPTS`
     попыток ОДНОЙ и той же попытки понять модель, пауза растёт как 2^n;
   - один повтор на ответ, не прошедший pydantic-валидацию `schema`
     (`UnexpectedModelBehavior` — при `retries=0` в `Agent` это ровно она, см.
     `_attempt`), затем `LLMSchemaError`, а не тихий провал.
   Они не переиспользуют бюджет друг друга: 429 на второй схема-попытке не тратит
   схема-ретрай, а невалидная схема не тратит HTTP-бюджет — это разные причины
   неудачи, и вызывающему (воркеру, задача 18) важно различать их по типу
   исключения, а не только по факту провала.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.models import Model
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from harness.memory.models import LlmCall
from harness.platform.config import Config
from harness.platform.limiter import PgLimiter

# Модульная косвенность вместо прямых вызовов `asyncio.sleep` (fix round 1, small
# item 4, тот же приём и то же обоснование, что в `limiter.py`): патч `harness.
# platform.llm._sleep` не задевает глобальный `asyncio.sleep`, а патч `harness.
# platform.llm.asyncio.sleep` задевал бы его ГЛОБАЛЬНО, для всего процесса, что
# и произошло в тестах до этого фикса (докстринги утверждали обратное — ложно).
_sleep = asyncio.sleep

T = TypeVar("T", bound=BaseModel)

# Спека §7 дословно: "ретрай при ответе, не прошедшем pydantic-валидацию (1 повтор,
# потом ошибка задачи)" — итого 2 попытки; "экспоненциальный бэкофф ... на 429/5xx"
# — бриф задачи 16 называет число попыток явно: 3.
_MAX_SCHEMA_ATTEMPTS = 2
_MAX_HTTP_ATTEMPTS = 3
_HTTP_JITTER_MIN_S = 0.05
_HTTP_JITTER_MAX_S = 0.15

# Магические байты — единственный надёжный способ узнать формат вложения (fix
# round 1: захардкоженный `image/png` был ПРОСТО НЕВЕРЕН для реального входа —
# Telegram переотдаёт сжатые фото как JPEG, не PNG; подписанный PNG на самом деле
# JPEG — это 400 от провайдера на первом же боевом скрине, задача 22). Сигнатуры
# по первым байтам, не по расширению файла — вызывающий передаёт голые `bytes`,
# без имени файла. GIF и WebP добавлены round 2, Item 3: пересланный стикер или
# нетипичный документ прежде поднимал бы `ValueError` до вызова модели вместо
# того, чтобы быть прочитанным — оба формата провайдер принимает.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
_GIF_MAGICS = (b"GIF87a", b"GIF89a")
# WebP — контейнер RIFF: 4 байта "RIFF", 4 байта длины (не проверяем — не часть
# сигнатуры формата), 4 байта "WEBP".
_RIFF_MAGIC = b"RIFF"
_WEBP_MAGIC = b"WEBP"


def _sniff_image_media_type(data: bytes) -> str:
    if data.startswith(_PNG_MAGIC):
        return "image/png"
    if data.startswith(_JPEG_MAGIC):
        return "image/jpeg"
    if data.startswith(_GIF_MAGICS):
        return "image/gif"
    if data[:4] == _RIFF_MAGIC and data[8:12] == _WEBP_MAGIC:
        return "image/webp"
    raise ValueError(
        "неизвестный формат изображения — ни PNG, ни JPEG, ни GIF, ни WebP "
        f"(первые байты: {data[:12]!r})"
    )


@dataclass(frozen=True, slots=True)
class CallMeta:
    """Метаданные одной УСПЕШНОЙ попытки — то, что возвращается вызывающему вместе
    с распарсенным результатом. Не путать со строкой `llm_calls`: та живёт для ВСЕХ
    попыток, включая провалившиеся, эта — только для той, что дошла до успеха.

    `cost_usd` — ОЦЕНКА (PydanticAI/genai-prices по прайс-листу поставщика, best-
    effort), не выставленный провайдером счёт: CLAUDE.md запрещает выдумывать
    денежные числа, и непомеченная оценка в колонке `llm_calls.cost` была бы ровно
    такой выдумкой. Годится для приблизительного мониторинга, не для биллинга.
    """

    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float | None
    latency_ms: int


class LLMSchemaError(Exception):
    """Ответ модели не прошёл pydantic-валидацию `schema` `_MAX_SCHEMA_ATTEMPTS`
    попыток подряд (спека §7). Отдельно от `LLMProviderError` — вызывающий (воркер,
    задача 18) должен различать "модель систематически не может выполнить схему"
    (возможно, стоит эскалировать игроку) и "провайдер недоступен" (стоит повторить
    задачу позже) — не одна и та же причина отказа.
    """


class LLMProviderError(Exception):
    """Провайдер вернул 429/5xx на всех `_MAX_HTTP_ATTEMPTS` попытках экспоненциального
    бэкоффа — исчерпание ретраев, а не первая неудача (см. `LLMSchemaError`)."""


def _describe_model(model: Model | str) -> tuple[str, str]:
    """`(provider, model_name)` для колонок `llm_calls.provider`/`.model` — из
    строки конфига формата `"<provider>:<model>"` (спека §7) либо, для тестовых
    двойников PydanticAI (`TestModel`/`FunctionModel`, задача 16), из атрибутов
    самого объекта `Model` (`.system`/`.model_name`) — у них нет строкового вида.
    """
    if isinstance(model, str):
        provider, sep, name = model.partition(":")
        return (provider, name) if sep else (provider, provider)
    return model.system, model.model_name


def _is_retryable_http_error(exc: ModelHTTPError) -> bool:
    return exc.status_code == 429 or 500 <= exc.status_code < 600


def _backoff_delay(attempt: int) -> float:
    """2^attempt секунд плюс джиттер (спека §7) — растущая пауза вместо мгновенного
    повторного удара по провайдеру, который уже сказал "подожди"."""
    return (2**attempt) + random.uniform(_HTTP_JITTER_MIN_S, _HTTP_JITTER_MAX_S)


class LLM:
    """Единственная дверь к моделям (см. модульный докстринг). `session_factory` —
    тот же `async_sessionmaker`, что у `JobsQueue`/`PgLimiter` (задачи 14 и 16):
    каждая запись в `llm_calls` — своя короткая транзакция, коммитящаяся сразу, а
    не часть транзакции лимитера или вызывающего кода.

    `model_override` — тестовый крюк (задача 16, ограничение окружения: в этом
    репозитории нет ключа провайдера, тесты обязаны обходиться `TestModel`/
    `FunctionModel` из PydanticAI и никогда не обращаться к сети). В проде не
    передаётся — модель выбирается из `cfg` по `purpose`.
    """

    def __init__(
        self,
        cfg: Config,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        model_override: Model | str | None = None,
    ) -> None:
        self._cfg = cfg
        self._session_factory = session_factory
        self._model_override = model_override
        self._limiter = PgLimiter(
            session_factory,
            max_concurrency=cfg.llm_max_concurrency,
            max_per_minute=cfg.llm_max_per_minute,
        )
        # Предфильтр, не источник истины — см. модульный докстринг, пункт 1.
        self._local_semaphore = asyncio.Semaphore(cfg.llm_max_concurrency)

    async def __call__(
        self,
        purpose: Literal["vision_extract", "verdict_text"],
        schema: type[T],
        *,
        prompt: str,
        images: Sequence[bytes] = (),
        trace_id: int,
    ) -> tuple[T, CallMeta]:
        """`trace_id` — обязательный параметр каждого вызова, а не конструктора:
        один `LLM` живёт на весь процесс воркера и обслуживает МНОГО задач/трейсов
        подряд (задача 18, `Deps.llm` — общий на все `run_job`), а `llm_calls.trace_id`
        обязан указывать на трейс ИМЕННО этого вызова, не на трейс, с которым фасад
        был бы создан когда-то давно.
        """
        model: Model | str = (
            self._model_override if self._model_override is not None else self._resolve_model(purpose)
        )
        provider, model_name = _describe_model(model)
        agent = Agent(model, output_type=schema, retries=0)
        user_prompt: list[str | BinaryContent] = [
            prompt,
            *(
                BinaryContent(data=image, media_type=_sniff_image_media_type(image))
                for image in images
            ),
        ]

        async with self._local_semaphore:
            last_schema_error: UnexpectedModelBehavior | None = None
            for _schema_attempt in range(_MAX_SCHEMA_ATTEMPTS):
                try:
                    return await self._attempt(
                        agent,
                        user_prompt,
                        trace_id=trace_id,
                        purpose=purpose,
                        provider=provider,
                        model_name=model_name,
                    )
                except UnexpectedModelBehavior as exc:
                    last_schema_error = exc
            raise LLMSchemaError(
                f"{purpose}: ответ модели не прошёл валидацию {schema.__name__} "
                f"после {_MAX_SCHEMA_ATTEMPTS} попыток"
            ) from last_schema_error

    def _resolve_model(self, purpose: str) -> str:
        return (
            self._cfg.llm_vision_model
            if purpose == "vision_extract"
            else self._cfg.llm_verdict_model
        )

    async def _attempt(
        self,
        agent: Agent[None, T],
        user_prompt: list[str | BinaryContent],
        *,
        trace_id: int,
        purpose: str,
        provider: str,
        model_name: str,
    ) -> tuple[T, CallMeta]:
        """Один "логический" вызов модели с точки зрения `__call__` — но внутри до
        `_MAX_HTTP_ATTEMPTS` попыток на 429/5xx, каждая со своей строкой `llm_calls`
        (см. пункт 2 модульного докстринга). `UnexpectedModelBehavior` (невалидная
        схема) наружу не оборачивается — `__call__` ловит её напрямую и решает,
        тратить ли схема-ретрай; эта функция лишь успевает залогировать строку перед
        тем, как исключение уйдёт выше.

        Третий `except` (fix round 1, discretionary item) ловит ЛЮБОЙ иной сбой
        (обрыв соединения, таймаут чтения, что угодно не опознанное как 429/5xx или
        невалидная схема) и закрывает строку `llm_calls` статусом `error` перед тем,
        как перевыбросить как есть — без него такая строка осталась бы `started`
        навсегда: у `llm_calls`, в отличие от `jobs`, нет reaper'а, который бы её
        когда-нибудь закрыл.

        Отмена — ОТДЕЛЬНЫЙ `except` перед ним, не покрывается третьим (fix round
        2, Item 2): `asyncio.CancelledError` наследуется от `BaseException`, не
        от `Exception`, и «любой иной сбой» её не ловит — а в асинхронном
        воркере с таймаутами отмена и есть самый частый вид «сбоя», не редкое
        исключение из него. Строка `llm_calls` закрывается тем же `status=
        'error'`, и `CancelledError` ОБЯЗАНА быть перевыброшена — проглоченная
        отмена ломает распространение отмены по всему дереву задач, а не только
        эту попытку.
        """
        last_http_error: ModelHTTPError | None = None
        for http_attempt in range(_MAX_HTTP_ATTEMPTS):
            async with self._limiter.slot():
                call_id = await self._log_started(
                    trace_id=trace_id, provider=provider, model=model_name, purpose=purpose
                )
                started = time.monotonic()
                try:
                    result = await agent.run(user_prompt)
                except asyncio.CancelledError:
                    await self._log_finished(call_id, status="error")
                    raise
                except UnexpectedModelBehavior:
                    await self._log_finished(call_id, status="schema_error")
                    raise
                except ModelHTTPError as exc:
                    await self._log_finished(call_id, status="error")
                    last_http_error = exc
                    if http_attempt + 1 >= _MAX_HTTP_ATTEMPTS or not _is_retryable_http_error(exc):
                        raise LLMProviderError(
                            f"{purpose}: провайдер вернул {exc.status_code} после "
                            f"{http_attempt + 1} попыт(ок)"
                        ) from exc
                    await _sleep(_backoff_delay(http_attempt))
                    continue
                except Exception:
                    await self._log_finished(call_id, status="error")
                    raise

                latency_ms = round((time.monotonic() - started) * 1000)
                usage = result.usage
                cost_usd = float(usage.cost) if usage.cost is not None else None
                await self._log_finished(
                    call_id,
                    status="ok",
                    tokens_in=usage.input_tokens,
                    tokens_out=usage.output_tokens,
                    cost=usage.cost,
                    latency_ms=latency_ms,
                )
                return result.output, CallMeta(
                    model=result.response.model_name or model_name,
                    tokens_in=usage.input_tokens or 0,
                    tokens_out=usage.output_tokens or 0,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                )
        # Недостижимо по факту цикла (последняя итерация всегда либо возвращает,
        # либо поднимает `LLMProviderError` выше) — только затем, чтобы pyright не
        # спотыкался о "функция не всегда возвращает значение": `range(N)` для него
        # не доказуемо непустой.
        raise LLMProviderError(
            f"{purpose}: провайдер недоступен после {_MAX_HTTP_ATTEMPTS} попыт(ок)"
        ) from last_http_error

    async def _log_started(self, *, trace_id: int, provider: str, model: str, purpose: str) -> int:
        """Строка вставляется ДО вызова модели, `status='started'` — in-flight
        вызовы обязаны входить в окно темпа `PgLimiter` (спека §7, см. также
        докстринг `models.LlmCall`)."""
        async with self._session_factory() as session:
            row = LlmCall(trace_id=trace_id, provider=provider, model=model, purpose=purpose)
            session.add(row)
            await session.commit()
            return row.id

    async def _log_finished(
        self,
        call_id: int,
        *,
        status: Literal["ok", "error", "schema_error"],
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        cost: Decimal | None = None,
        latency_ms: int | None = None,
    ) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(LlmCall)
                .where(LlmCall.id == call_id)
                .values(
                    status=status,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost=cost,
                    latency_ms=latency_ms,
                )
            )
            await session.commit()
