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
   одно HTTP-обращение к провайдеру, включая каждый запрос внутри прогона с
   инструментами: и бэкофф-ретрай на 429/5xx, и ретрай на невалидную схему — это
   РАЗНЫЕ попытки, каждая логируется отдельно, иначе окно темпа (`PgLimiter`,
   считает по `llm_calls`) занижает нагрузку ровно там, где она реальна.
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

import anyio
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent
from pydantic_ai._genai_prices import fill_response_cost
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.profiles.anthropic import AnthropicModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import Tool
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from harness.memory.models import LlmCall
from harness.platform import limiter as _limiter_module
from harness.platform.config import Config
from harness.platform.limiter import PgLimiter

# Модульная косвенность вместо прямых вызовов `asyncio.sleep` (fix round 1, small
# item 4, тот же приём и то же обоснование, что в `limiter.py`): патч `harness.
# platform.llm._sleep` не задевает глобальный `asyncio.sleep`, а патч `harness.
# platform.llm.asyncio.sleep` задевал бы его ГЛОБАЛЬНО, для всего процесса, что
# и произошло в тестах до этого фикса (докстринги утверждали обратное — ложно).
_sleep = asyncio.sleep

# `_limiter_module._ACQUIRE_TIMEOUT_S` — обращение через модуль, не через `from
# ... import _ACQUIRE_TIMEOUT_S` (fix round 3, Item 1: найдено на живом коде —
# первая версия этого файла именно так и импортировала константу, и тест на
# разделяемый дедлайн, `test_limiter_budget_bounds_full_retry_chain_not_
# multiplied`, завис на реальные ~120с вместо монкипатченных долей секунды).
# `from module import NAME` копирует ЗНАЧЕНИЕ на момент импорта — `monkeypatch.
# setattr("harness.platform.limiter._ACQUIRE_TIMEOUT_S", ...)` меняет атрибут
# ИСХОДНОГО модуля, а уже скопированное в `llm.py` число остаётся прежним. Тот
# же класс ошибки, что и с `asyncio.sleep` выше, только в обратную сторону:
# там прямая ссылка на модуль патчилась ШИРЕ, чем нужно, здесь — импорт
# значения не патчился бы ВООБЩЕ.


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


# Предел на одну картинку у провайдера — 10 МБ, и считает он его по BASE64-
# представлению, а не по сырым байтам. Картинка едет к модели как
# `BinaryContent`, то есть в base64, поэтому в сырых байтах принимать можно
# только меньше: base64 добавляет треть. Число ниже взято с запасом и годится
# при обоих прочтениях «мегабайта» — 10 000 000 здесь именно строгое из двух.
# Соотношение держит `test_the_image_cap_leaves_room_for_base64_inflation`.
PROVIDER_BASE64_IMAGE_LIMIT = 10_000_000
MAX_IMAGE_BYTES = 7 * 1024 * 1024
MAX_IMAGE_MB = MAX_IMAGE_BYTES // (1024 * 1024)


@dataclass(frozen=True, slots=True)
class CallMeta:
    """Метаданные одной УСПЕШНОЙ попытки — то, что возвращается вызывающему вместе
    с распарсенным результатом. Не путать со строкой `llm_calls`: та живёт для ВСЕХ
    попыток, включая провалившиеся, эта — только для той, что дошла до успеха.

    `cost_usd` — ОЦЕНКА (PydanticAI/genai-prices по прайс-листу поставщика, best-
    effort), не выставленный провайдером счёт: CLAUDE.md запрещает выдумывать
    денежные числа, и непомеченная оценка в колонке `llm_calls.cost` была бы ровно
    такой выдумкой. Годится для приблизительного мониторинга, не для биллинга.

    `latency_ms` — длительность прогона целиком, со всеми запросами к провайдеру и
    вызовами инструментов внутри; длительность каждого запроса — в его строке
    `llm_calls`.

    `messages` — `result.all_messages()` удачного прогона: переданная история плюс
    сообщения этого хода
    (`test_the_history_reaches_the_model_and_comes_back_with_the_new_turn`).
    """

    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float | None
    latency_ms: int
    messages: tuple[ModelMessage, ...] = ()


class LLMSchemaError(Exception):
    """Ответ модели не прошёл pydantic-валидацию `schema` `_MAX_SCHEMA_ATTEMPTS`
    попыток подряд (спека §7). Отдельно от `LLMProviderError` — вызывающий (воркер,
    задача 18) должен различать "модель систематически не может выполнить схему"
    (возможно, стоит эскалировать игроку) и "провайдер недоступен" (стоит повторить
    задачу позже) — не одна и та же причина отказа.
    """


class LLMNotConfigured(Exception):
    """Назначение вызова требует модели, которой в конфиге нет.

    Отдельно от `LLMProviderError`: провайдер тут ни при чём, дело в переменной
    окружения, и оператору надо править `.env`, а не ждать, пока провайдер
    оживёт.
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


# Модели Anthropic, которые отвечают 400 `tool_choice: type "tool" and "any" are not
# supported for this model.` на принудительный выбор инструмента. Структурный вывод
# PydanticAI его и шлёт, а собственный список библиотеки
# (`profiles/anthropic.py`, `anthropic_supports_forced_tool_choice`) этих моделей не
# знает. С флагом `False` библиотека сама переходит на `tool_choice: auto`.
_REJECTS_FORCED_TOOL_CHOICE = ("claude-sonnet-5-5",)


def _build_model(model: str) -> Model | str:
    """Строка конфига — как есть, кроме моделей из `_REJECTS_FORCED_TOOL_CHOICE`:
    им `AnthropicModel` с профилем без принудительного `tool_choice`
    (`test_the_agent_model_does_not_force_tool_choice`,
    `test_other_anthropic_models_keep_the_library_profile`)."""
    provider, sep, name = model.partition(":")
    if sep and provider == "anthropic" and name.startswith(_REJECTS_FORCED_TOOL_CHOICE):
        return AnthropicModel(
            name, profile=AnthropicModelProfile(anthropic_supports_forced_tool_choice=False)
        )
    return model


def _served_by(response: object) -> str | None:
    """Хостер, фактически обслуживший вызов, — из метаданных ответа шлюза.

    Между нами и моделью может стоять шлюз (OpenRouter), который маршрутизирует
    запрос к одному из нескольких хостеров одной и той же модели. Хостеры
    различаются тем, что нам критично: `seed` поддерживают не все, а
    `tool_choice: required`, без которого нет структурированного вывода, — не все
    из оставшихся. Имя приходит в `provider_details.downstream_provider`.

    `None` при прямом вызове вендора: он хостера не называет, и это факт, а не
    пропуск — выдумывать здесь нечего (`llm_calls.served_by` nullable).
    """
    details = getattr(response, "provider_details", None)
    if not isinstance(details, dict):
        return None
    name = details.get("downstream_provider")
    return name[:64] if isinstance(name, str) and name else None


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
        purpose: Literal[
            "vision_extract",
            "vision_extract_fallback",
            "verdict_text",
            "tournament_text",
            "question_answer",
            "agent",
        ],
        schema: type[T],
        *,
        prompt: str,
        images: Sequence[bytes] = (),
        tools: Sequence[Tool[None]] = (),
        message_history: Sequence[ModelMessage] = (),
        instructions: str | None = None,
        trace_id: int,
    ) -> tuple[T, CallMeta]:
        """`trace_id` — обязательный параметр каждого вызова, а не конструктора:
        один `LLM` живёт на весь процесс воркера и обслуживает МНОГО задач/трейсов
        подряд (задача 18, `Deps.llm` — общий на все `run_job`), а `llm_calls.trace_id`
        обязан указывать на трейс ИМЕННО этого вызова, не на трейс, с которым фасад
        был бы создан когда-то давно.

        Дедлайн лимитера (fix round 3, Item 1) считается ЗДЕСЬ, один раз на всю
        цепочку ретраев (схема- и HTTP-), и передаётся в каждый `self._limiter.
        slot()` неизменным — не заново на каждой попытке. Иначе полная цепочка
        (до `_MAX_HTTP_ATTEMPTS × _MAX_SCHEMA_ATTEMPTS` разных `slot()`) могла
        бы суммарно ждать лимитер в разы дольше окна `JobsQueue.reap()`, ради
        которого `_ACQUIRE_TIMEOUT_S` вообще считался (см. `limiter.py`,
        докстринг константы) — задача выглядела бы зависшей, reap() отдал бы
        её другому воркеру, и провайдеру заплатили бы за неё дважды.

        Точка отсчёта — не вход в эту функцию, а момент, когда получен
        внутрипроцессный `self._local_semaphore` (round 5, Item K.5: три
        докстринга говорили «от входа в `LLM.__call__`» — неверно). Разница
        видна под нагрузкой: вызов, вставший в очередь за локальным
        семафором, начинает свои 120с только когда семафор его пропустил, и
        стояние в этой очереди из бюджета лимитера не вычитается. Так и
        задумано — предфильтр очереди не должен обвинять лимитер, — но
        означает, что дедлайн НЕ ограничивает полную длительность `__call__`
        сверху, только её часть.

        Внутри же этой точки бюджет АБСОЛЮТНЫЙ, а не накопитель одного лишь
        времени ожидания лимитера (round 4, Item 1): время самого ответа
        модели (каждый запрос внутри `agent.run()`) и бэкофф-паузы между
        HTTP-попытками утекают из тех же 120с, потому что часы между попытками
        не останавливаются — здоровый, но медленный провайдер оставляет
        следующей попытке МЕНЬШЕ бюджета `slot()`, не отдельный полный. Это
        осознанный компромисс контроллера, не недосмотр: подробное обоснование
        — в докстринге `limiter._ACQUIRE_TIMEOUT_S`.

        `tools` — инструменты, которые модель вправе позвать внутри одного
        `agent.run()`. Вызов с инструментами обращается к провайдеру БОЛЬШЕ
        одного раза (ответ с вызовом инструмента, затем ответ по его
        результату), и на каждый запрос своя строка `llm_calls`
        (`test_a_turn_with_two_tool_calls_logs_three_requests`,
        `test_a_tool_call_round_trip_logs_a_row_per_request`).

        `message_history` уходит модели перед `prompt`, `instructions` — в
        инструкции этого прогона (`test_instructions_reach_the_model`).
        """
        model: Model | str = (
            self._model_override if self._model_override is not None else self._resolve_model(purpose)
        )
        provider, model_name = _describe_model(model)
        if isinstance(model, str):
            model = _build_model(model)
        user_prompt: list[str | BinaryContent] = [
            prompt,
            *(
                BinaryContent(data=image, media_type=_sniff_image_media_type(image))
                for image in images
            ),
        ]

        async with self._local_semaphore:
            deadline = time.monotonic() + _limiter_module._ACQUIRE_TIMEOUT_S
            metered = _MeteredModel(
                model,
                llm=self,
                trace_id=trace_id,
                purpose=purpose,
                provider=provider,
                model_name=model_name,
                deadline=deadline,
            )
            agent: Agent[None, T] = Agent(
                metered,
                output_type=schema,
                retries=0,
                tools=tools,
                deps_type=type(None),
                instructions=instructions,
            )
            last_schema_error: UnexpectedModelBehavior | None = None
            for _schema_attempt in range(_MAX_SCHEMA_ATTEMPTS):
                try:
                    return await self._attempt(
                        agent, metered, user_prompt, message_history, model_name
                    )
                except UnexpectedModelBehavior as exc:
                    last_schema_error = exc
            raise LLMSchemaError(
                f"{purpose}: ответ модели не прошёл валидацию {schema.__name__} "
                f"после {_MAX_SCHEMA_ATTEMPTS} попыток"
            ) from last_schema_error

    def _resolve_model(self, purpose: str) -> str:
        """Модель по назначению вызова — строка конфига, а не ветка кода.

        Обе переменные зрения необязательны (`platform/config.py`), и вызвать их
        назначение с пустой строкой значит попросить `Agent("")` — отказ
        провайдера тремя уровнями глубже вместо названной причины. Поэтому
        проверка здесь, у обеих одинаковая: решение «звать ли модель» принимает
        вызывающий, а этот отказ ловит того, кто решение обошёл.

        Остальные назначения берут `LLM_VERDICT_MODEL`, и она обязательна:
        на неё падает `question_answer` — единственный вызов модели, который
        конвейер сегодня делает, кроме агента, у которого своя модель
        (`LLM_AGENT_MODEL`, `test_the_agent_purpose_takes_its_own_model`).
        """
        if purpose == "vision_extract":
            if not self._cfg.llm_vision_model:
                raise LLMNotConfigured("LLM_VISION_MODEL не задана — скрин-вход закрыт")
            return self._cfg.llm_vision_model
        if purpose == "vision_extract_fallback":
            if not self._cfg.llm_vision_fallback_model:
                raise LLMNotConfigured(
                    "LLM_VISION_FALLBACK_MODEL не задана — второй ступени каскада нет"
                )
            return self._cfg.llm_vision_fallback_model
        if purpose == "agent":
            return self._cfg.llm_agent_model
        return self._cfg.llm_verdict_model

    async def _attempt(
        self,
        agent: Agent[None, T],
        metered: _MeteredModel,
        user_prompt: list[str | BinaryContent],
        message_history: Sequence[ModelMessage],
        model_name: str,
    ) -> tuple[T, CallMeta]:
        """Один прогон агента. Слоты, строки `llm_calls`, таймаут и HTTP-ретраи —
        внутри `_MeteredModel.request`, на каждый запрос. Здесь — только схема:
        ответ, не прошедший `schema`, переписывает статус строки ПОСЛЕДНЕГО
        запроса на `schema_error` (запрос удался, ответ — нет;
        `test_retry_on_schema_error_then_fail`) и уходит в `__call__`, который
        решает про схема-ретрай."""
        started = time.monotonic()
        try:
            result = await agent.run(user_prompt, message_history=list(message_history) or None)
        except UnexpectedModelBehavior:
            if metered.last_call_id is not None:
                await self._log_status(metered.last_call_id, "schema_error")
            raise
        usage = result.usage
        return result.output, CallMeta(
            model=result.response.model_name or model_name,
            tokens_in=usage.input_tokens or 0,
            tokens_out=usage.output_tokens or 0,
            cost_usd=float(usage.cost) if usage.cost is not None else None,
            latency_ms=round((time.monotonic() - started) * 1000),
            messages=tuple(result.all_messages()),
        )

    async def _log_status(self, call_id: int, status: Literal["schema_error"]) -> None:
        """Переписать только статус — токены и цена удачного запроса остаются."""
        async with self._session_factory() as session:
            await session.execute(
                update(LlmCall).where(LlmCall.id == call_id).values(status=status)
            )
            await session.commit()

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
        served_by: str | None = None,
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
                    served_by=served_by,
                )
            )
            await session.commit()


class _MeteredModel(WrapperModel):
    """Модель, у которой КАЖДЫЙ запрос к провайдеру — свой слот лимитера, своя
    строка `llm_calls`, свой таймаут и свои HTTP-ретраи.

    Прогон агента с инструментами — несколько запросов; считать его одним значило
    занижать окно темпа `PgLimiter` ровно там, где нагрузка растёт
    (`test_a_turn_with_two_tool_calls_logs_three_requests`). Таймаут
    (`Config.llm_request_timeout_s`) обёрнут вокруг одного запроса, а не прогона:
    зависший запрос снимается сам, строка закрывается `error`, наружу —
    `LLMProviderError` (`test_a_hung_provider_request_is_cut_by_the_request_timeout`).

    `deadline` — общий бюджет ожидания лимитера на весь вызов `LLM.__call__`
    (считается там один раз) и передаётся в каждый `slot()` неизменным —
    на каждом запросе прогона, каждой HTTP-попытке и каждом схема-ретрае
    (`test_call_shares_one_deadline_across_retry_attempts`,
    `test_call_shares_one_deadline_across_schema_retry_attempts`). Часы между
    попытками не останавливаются: время ответов модели и бэкофф-пауз съедает
    бюджет следующего `slot()` (подробно — докстринг
    `limiter._ACQUIRE_TIMEOUT_S`).

    Бэкофф-пауза спит СНАРУЖИ `slot()`: спящая корутина провайдера не нагружает
    и слот держать не должна (`test_backoff_sleeps_outside_the_concurrency_slot`).

    Отмена закрывает строку `error` под `asyncio.shield` и перевыбрасывается
    (`test_cancellation_finalizes_row_and_repropagates`); любой иной сбой закрывает
    строку `error` и уходит как есть (`test_unexpected_exception_finalizes_row_as_error`)
    — у `llm_calls` нет reaper'а, `started` осталась бы навсегда.

    Цена запроса в строке — оценка библиотеки (`fill_response_cost`, та же функция,
    что граф агента зовёт после запроса; уже заполненную она не перезаписывает).
    Без неё колонка `llm_calls.cost` пустела бы: граф ставит цену уже после
    возврата из `request` (`test_each_request_row_carries_its_own_cost_estimate`).

    `last_call_id` — строка последнего запроса: `_attempt` переписывает ей статус
    на `schema_error`, если ответ этого запроса не прошёл схему уже после того,
    как сам запрос удался.
    """

    def __init__(
        self,
        wrapped: Model | str,
        *,
        llm: LLM,
        trace_id: int,
        purpose: str,
        provider: str,
        model_name: str,
        deadline: float,
    ) -> None:
        # `KnownModelName` — подсказка типа, а не ограничение: `infer_model`
        # принимает любую строку `"<provider>:<model>"` из конфига.
        super().__init__(wrapped)  # pyright: ignore[reportArgumentType]
        # Префикс `_meter_` — не стиль: у `Model` есть свой `_provider`, и
        # одноимённый атрибут обёртки подменил бы его.
        self._meter_llm = llm
        self._meter_trace_id = trace_id
        self._meter_purpose = purpose
        self._meter_provider = provider
        self._meter_model_name = model_name
        self._meter_deadline = deadline
        self.last_call_id: int | None = None

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Учёт запроса идёт в СВОЕЙ задаче asyncio, а не в задаче графа.

        Граф агента зовёт модель внутри `anyio.CancelScope`, а отмена anyio
        уровневая: пока область отменена, она снова и снова отменяет задачу графа
        и всё, чего та ждёт, — включая уборку: закрытие строки `llm_calls` и выход
        из `slot()`. Отдельная задача в эту область не входит; ждём её через
        `asyncio.shield`, чтобы повторные отмены до неё не доходили, и отменяем её
        сами ровно один раз — уборка в `_metered_request` доходит до конца.
        Ожидание этой уборки — под щитом anyio
        (`test_cancellation_finalizes_row_and_repropagates`).
        """
        inner = asyncio.create_task(
            self._metered_request(messages, model_settings, model_request_parameters)
        )
        try:
            return await asyncio.shield(inner)
        except asyncio.CancelledError:
            inner.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.wait({inner})
            raise

    async def _metered_request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        llm = self._meter_llm
        timeout_s = llm._cfg.llm_request_timeout_s
        last_http_error: ModelHTTPError | None = None
        for http_attempt in range(_MAX_HTTP_ATTEMPTS):
            async with llm._limiter.slot(deadline=self._meter_deadline):
                call_id = await llm._log_started(
                    trace_id=self._meter_trace_id,
                    provider=self._meter_provider,
                    model=self._meter_model_name,
                    purpose=self._meter_purpose,
                )
                self.last_call_id = call_id
                started = time.monotonic()
                try:
                    async with asyncio.timeout(timeout_s):
                        response = await self.wrapped.request(
                            messages, model_settings, model_request_parameters
                        )
                except asyncio.CancelledError:
                    # `shield`: вторая отмена, подоспевшая во время записи, не
                    # оборвёт уже запущенный `_log_finished`.
                    await asyncio.shield(llm._log_finished(call_id, status="error"))
                    raise
                except TimeoutError as exc:
                    # До общего `except Exception`: `asyncio.timeout` поднимает
                    # именно `TimeoutError`, и он должен стать отказом провайдера.
                    await llm._log_finished(call_id, status="error")
                    raise LLMProviderError(
                        f"{self._meter_purpose}: запрос к провайдеру не уложился в {timeout_s} с"
                    ) from exc
                except ModelHTTPError as exc:
                    await llm._log_finished(call_id, status="error")
                    last_http_error = exc
                    if http_attempt + 1 >= _MAX_HTTP_ATTEMPTS or not _is_retryable_http_error(exc):
                        raise LLMProviderError(
                            f"{self._meter_purpose}: провайдер вернул {exc.status_code} после "
                            f"{http_attempt + 1} попыт(ок)"
                        ) from exc
                except Exception:
                    await llm._log_finished(call_id, status="error")
                    raise
                else:
                    fill_response_cost(response)
                    usage = response.usage
                    await llm._log_finished(
                        call_id,
                        status="ok",
                        tokens_in=usage.input_tokens,
                        tokens_out=usage.output_tokens,
                        cost=usage.cost,
                        latency_ms=round((time.monotonic() - started) * 1000),
                        served_by=_served_by(response),
                    )
                    return response
            # СНАРУЖИ `slot()`: сюда доводит только ретраибельная `ModelHTTPError`.
            await _sleep(_backoff_delay(http_attempt))
        # Недостижимо по факту цикла — только для pyright (`range(N)` для него не
        # доказуемо непустой).
        raise LLMProviderError(
            f"{self._meter_purpose}: провайдер недоступен после {_MAX_HTTP_ATTEMPTS} попыт(ок)"
        ) from last_http_error
