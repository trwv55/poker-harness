"""Маршрутизация вопроса к расчёту: инструменты, одно ограничение на каждое правило.

**В сеть тесты не ходят** (ограничение задачи 16): модель — `FunctionModel`
PydanticAI, сценарий вызовов задаёт сам тест. Настоящий Postgres нужен по той
же причине, что и словарю расчётов: инструмент делает выборку SQL, и подделка
репозитория проверяла бы подделку.

**Что здесь НЕ проверяется.** Правильность выбора расчёта. Механической
проверки у неё нет: маршрут меряется размеченным набором вопросов (EVALS,
этаж 3), который копится из настоящих вопросов игроков. Проверено здесь то,
что ограничивает ущерб от неверного выбора: числа берутся только из выхода
вызванного инструмента, расчёт называет себя сам, второго расчёта не будет, а
ответ без расчёта игроку не показывается.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy import text as sql_text

from harness.calcs.routing import CalcRun, answer_question, tools_for
from harness.contracts import CalcName, FrequencyResult
from harness.explanation.question import QuestionDraft
from harness.memory.models import Job, Trace
from harness.memory.repos import PlayersRepo, SessionsRepo
from harness.platform.config import Config
from harness.platform.llm import LLM
from tests.conftest import requires_prompts
from tests.test_calcs import _open_from, _store

_TEST_CFG = Config(
    llm_vision_model="anthropic:vision-test",
    llm_vision_fallback_model="",
    llm_verdict_model="anthropic:verdict-test",
    llm_max_concurrency=4,
    llm_max_per_minute=1000,
    database_url="unused-in-tests",
    telegram_token="unused-in-tests",
)


class _Script:
    """Сценарий модели: сначала названные вызовы инструментов, потом ответ.

    Возвраты инструментов копятся в `returns` — тест читает их, чтобы увидеть,
    что инструмент ответил модели (отказ второго расчёта, жалоба на параметр).
    """

    def __init__(self, calls: list[tuple[str, dict[str, Any]]], answer: str) -> None:
        self.calls = calls
        self.answer = answer
        self.returns: list[str] = []
        self.requests = 0

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.requests += 1
        self.returns = [
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and isinstance(part.content, str)
        ]
        step = sum(isinstance(message, ModelResponse) for message in messages)
        if step < len(self.calls):
            name, args = self.calls[step]
            return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args)])
        assert info.output_tools
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"answer": self.answer})]
        )


async def _scope(session, tg_user_id: int) -> tuple[int, int, int]:
    """Игрок, его сессия и валидный `trace_id` для FK `llm_calls.trace_id`."""
    player = await PlayersRepo(session).get_or_create(tg_user_id=tg_user_id)
    session_row = await SessionsRepo(session).active_or_create(player.id)
    job = Job(type="question", player_id=player.id, session_id=session_row.id, payload={})
    session.add(job)
    await session.flush()
    trace = Trace(job_id=job.id)
    session.add(trace)
    await session.commit()
    return player.id, session_row.id, trace.id


async def _seeded(session, tg_user_id: int) -> tuple[int, int, int]:
    """Две раздачи героя в базе — чтобы у частоты был ненулевой знаменатель."""
    player_id, session_id, trace_id = await _scope(session, tg_user_id)
    for index in (1, 2):
        await _store(
            session,
            session_id=session_id,
            raw=_open_from("CO", tournament_id="T1", hand_no=f"Q{index}"),
        )
    await session.commit()
    return player_id, session_id, trace_id


async def _ask(db_factory, session, script: _Script, *, scope, question="вопрос"):
    player_id, session_id, trace_id = scope
    llm = LLM(_TEST_CFG, db_factory, model_override=FunctionModel(script))
    return await answer_question(
        session,
        llm,
        player_id=player_id,
        session_id=session_id,
        question=question,
        trace_id=trace_id,
    )


def test_the_tool_set_covers_every_calculation_name_exactly_once():
    """Набор инструментов и набор имён словаря — один список, а не две копии.

    Разойтись они могли бы молча: лишний инструмент считал бы то, чего в
    словаре нет, а забытый делал бы имя недостижимым для вопроса.
    """
    tools = tools_for(None, 1, 1, CalcRun())  # type: ignore[arg-type] — база здесь не трогается
    assert sorted(tool.name for tool in tools) == sorted(name.value for name in CalcName)


@requires_prompts
async def test_a_number_from_the_tool_reaches_the_player(db_factory):
    """Число, которое инструмент посчитал, доходит до игрока вместе с расчётом."""
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=7001)
        script = _Script(
            [("hero_frequency", {"stat": "vpip"})],
            "Вы входите в банк в 100.0% раздач (2 из 2).",
        )
        outcome = await _ask(db_factory, session, script, scope=scope)

    assert outcome.result is not None
    assert outcome.result.calc is CalcName.HERO_FREQUENCY
    assert outcome.prose == "Вы входите в банк в 100.0% раздач (2 из 2)."


@requires_prompts
async def test_a_number_the_tool_did_not_return_costs_the_whole_prose(db_factory):
    """Выдуманное число не правится и не показывается: игрок получает числа расчёта.

    Слова при этом теряются целиком, а не по предложению: правка текста модели
    была бы тем самым «подправить, чтобы сошлось», которое запрещено.
    """
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=7002)
        script = _Script(
            [("hero_frequency", {"stat": "vpip"})],
            "Вы входите в банк в 43.0% раздач.",
        )
        outcome = await _ask(db_factory, session, script, scope=scope)

    assert outcome.result is not None
    assert outcome.prose is None


@requires_prompts
async def test_the_second_calculation_is_refused(db_factory):
    """Один вопрос — один расчёт: второй инструмент не выполняется вовсе.

    Проверяется и то, что модели сказали об этом текстом, и то, что наружу
    уехал ПЕРВЫЙ расчёт: перебор срезов находит отклонение по случайности,
    поэтому запрет механический, а не просьба в промпте.
    """
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=7003)
        script = _Script(
            [
                ("hero_frequency", {"stat": "vpip"}),
                ("leaks", {}),
            ],
            "Сказать нечего.",
        )
        outcome = await _ask(db_factory, session, script, scope=scope)

    assert outcome.result is not None
    assert outcome.result.calc is CalcName.HERO_FREQUENCY
    assert any("один вопрос — один расчёт" in text.lower() for text in script.returns)


@requires_prompts
async def test_a_question_without_a_calculation_is_refused(db_factory):
    """Ответ, не опирающийся ни на один инструмент, игроку не показывается.

    Уверенный текст из общих знаний о покере модель напишет всегда; отличить
    его по виду нельзя, поэтому отличаем по факту вызова.
    """
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=7004)
        script = _Script([], "На BB вы минусовой, это нормально для позиции.")
        outcome = await _ask(db_factory, session, script, scope=scope)

    assert outcome.result is None
    assert outcome.prose is None


@requires_prompts
async def test_a_bad_parameter_does_not_spend_the_single_calculation(db_factory):
    """Жалоба на параметр — не расчёт: после неё инструмент вправе быть вызван.

    Иначе опечатка модели в позиции стоила бы игроку всего ответа.
    """
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=7005)
        script = _Script(
            [
                ("hero_frequency", {"stat": "vpip", "position": "BUTTON"}),
                ("hero_frequency", {"stat": "vpip", "position": "CO"}),
            ],
            "Вы входите в банк в 100.0% раздач (2 из 2).",
        )
        outcome = await _ask(db_factory, session, script, scope=scope)

    assert isinstance(outcome.result, FrequencyResult)
    assert outcome.result.position == "CO"
    assert outcome.prose is not None
    assert any("BUTTON" in text for text in script.returns)


@requires_prompts
async def test_an_unnamed_opponent_is_not_guessed(db_factory):
    """Ник, которого игрок не называл, не превращается в чужую выборку.

    Привязку оппонента к его местам утверждает только владелец (`/alias`);
    догадка здесь дала бы статистику другого человека под его ником.
    """
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=7006)
        script = _Script(
            [("opponent_frequency", {"stat": "vpip", "opponent": "кто-то"})],
            "Сказать нечего.",
        )
        outcome = await _ask(db_factory, session, script, scope=scope)

    assert outcome.result is None
    assert any("не называл" in text for text in script.returns)


@requires_prompts
async def test_the_question_reaches_the_prompt(db_factory):
    """Вопрос игрока едет в промпт целиком — иначе модель выбирает вслепую."""
    seen: list[str] = []

    class _Peeking(_Script):
        def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for message in messages:
                for part in message.parts:
                    if not isinstance(part, UserPromptPart):
                        continue
                    content = part.content
                    seen.extend([content] if isinstance(content, str) else
                                [item for item in content if isinstance(item, str)])
            return super().__call__(messages, info)

    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=7007)
        script = _Peeking(
            [("hero_frequency", {"stat": "vpip"})],
            "Вы входите в банк в 100.0% раздач (2 из 2).",
        )
        await _ask(db_factory, session, script, scope=scope, question="есть ли у меня третий баррель")

    assert any("есть ли у меня третий баррель" in text for text in seen)


async def test_a_tool_call_round_trip_logs_one_row(db_factory):
    """Вызов с инструментами — одна строка `llm_calls`, а не по строке на обращение.

    Внутри такого вызова провайдеру уходит два запроса (выбор инструмента и
    ответ по его выходу), а окно темпа `PgLimiter` считает его за один: цена
    вопроса в этом счёте занижена, и знать об этом надо до, а не после
    телеметрии.
    """
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=7008)
        script = _Script(
            [("hero_frequency", {"stat": "vpip"})],
            "Вы входите в банк в 100.0% раздач (2 из 2).",
        )
        llm = LLM(_TEST_CFG, db_factory, model_override=FunctionModel(script))
        run_state = CalcRun()
        tools = tools_for(session, scope[0], scope[1], run_state)
        await llm(
            "question_answer",
            QuestionDraft,
            prompt="вопрос",
            tools=tools,
            trace_id=scope[2],
        )

    async with db_factory() as session:
        rows = (
            await session.execute(sql_text("select purpose, status from llm_calls order by id"))
        ).all()
    assert script.requests == 2, "сценарий не сделал двух обращений — проверять нечего"
    assert [tuple(row) for row in rows] == [("question_answer", "ok")]
    assert run_state.result is not None


@pytest.mark.parametrize("name", [name.value for name in CalcName])
def test_every_tool_describes_its_parameters(name: str):
    """У каждого инструмента есть описание и схема параметров.

    Без описания модель выбирает по одному имени; схема при этом собирается из
    сигнатуры, и пустая означала бы инструмент, который нечем параметризовать.
    """
    tools = {tool.name: tool for tool in tools_for(None, 1, 1, CalcRun())}  # type: ignore[arg-type]
    assert tools[name].description


@requires_prompts
async def test_a_wrong_argument_value_is_handed_back_to_the_model(db_factory):
    """Значение параметра, которого нет в перечислении, не роняет задачу.

    Проверка аргумента не пускает такой вызов до расчёта вовсе, поэтому повтор
    ничего не пересчитывает — а без него опечатка модели стоила бы игроку
    всего ответа.
    """
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=7009)
        script = _Script(
            [
                ("hero_frequency", {"stat": "как часто вхожу"}),
                ("hero_frequency", {"stat": "vpip"}),
            ],
            "Вы входите в банк в 100.0% раздач (2 из 2).",
        )
        outcome = await _ask(db_factory, session, script, scope=scope)

    assert outcome.result is not None
    assert outcome.prose is not None
