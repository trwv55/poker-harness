"""Ход агента: инструменты, потолок, проверки, попытка переписать, история.

В сеть не ходим: модель — `FunctionModel`, сценарий по прогонам задаёт тест.
База настоящая — инструменты делают выборку SQL (довод `test_question_routing.py`).
Выбор расчёта здесь не проверяется: это eval маршрутизации (спека №2).
"""

from __future__ import annotations

import re
from collections.abc import Collection
from itertools import pairwise
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy import text as sql_text

import harness.calcs as calcs_module
from harness.agent.history import (
    HAND_KEY,
    NEW_HAND_MARK,
    NO_HAND_MARK,
    allowed_from_history,
    assumed_numbers_in,
    calcs_in_history,
    chart_numbers_in,
    dump_history,
    estimate_tokens,
    frequency_numbers,
    hand_player_texts,
    load_history,
    narrowings_in,
    player_numbers,
    trim_history,
    turns,
)
from harness.agent.prompt import instructions_for, load_system_prompt
from harness.agent.run import MAX_CALCS_PER_TURN, REWRITE_HEAD, AgentDraft, run_agent_turn
from harness.analysis.charts import lookup, parse_range
from harness.analysis.frequency import defense_from_bet
from harness.analysis.opponent_range import RangeRefused, narrow
from harness.calcs.routing import CalcRun, tools_for
from harness.contracts import (
    AGENT_ONLY_CALCS,
    CalcName,
    DefenseResult,
    EquityVsRangeResult,
    OpponentRangeResult,
    RangeNarrowing,
    Street,
)
from harness.explanation.agent_checks import agent_context
from harness.memory.repos import SessionsRepo
from harness.platform.llm import LLM, LLMSchemaError
from harness.presentation.messages import agent_turn_msg
from tests.conftest import requires_agent_prompt
from tests.test_calcs import stored_analysis
from tests.test_hand_replay import _postflop_hand
from tests.test_opponent_range import BTN_OPENS_BB_CALLS, defend_file
from tests.test_postflop_line import T, _bet, _call, _check
from tests.test_question_routing import _TEST_CFG, _seeded
from tests.test_verdict_text import _point, _result

_SYSTEM = "СИСТЕМНЫЙ-СИНТ"
_THINKING = "РАССУЖДЕНИЕ-СИНТ"


class _TurnScript:
    """Сценарий по прогонам: в каждом — вызовы инструментов, потом ответ.

    Шаг считается от ПОСЛЕДНЕГО сообщения игрока, а не от начала истории:
    иначе прошлые ходы вечера сдвигали бы сценарий (у `_Script` из
    `test_question_routing.py` истории нет, у агента — есть).
    """

    def __init__(
        self,
        runs: list[tuple[list[tuple[str, dict[str, Any]]], str]],
        *,
        chatter: bool = False,
        as_text: Collection[int] = (),
        assumptions: dict[int, str] | None = None,
        think: bool = False,
    ) -> None:
        """`chatter` — модель пишет текст рядом с вызовом вывода, как живые
        провайдеры («Итак: …»); `FunctionModel` сам этого не делает.
        `as_text` — номера прогонов, в которых модель отвечает простым текстом,
        а не вызовом вывода (живая модель с `tool_choice: auto` делает и так,
        и так); такой ответ идёт с рассуждением рядом. `think` — рассуждение
        в каждом ответе модели, и в вызовах тоже, как у живой модели с thinking.
        `thinking_seen` — модели пришло рассуждение из истории."""
        self.runs = runs
        self.chatter = chatter
        self.as_text = frozenset(as_text)
        # Номер прогона → блок «Оценка модели» вторым полем вывода.
        self.assumptions = assumptions or {}
        self.run = -1
        self.requests = 0
        self.instructions: list[str] = []
        self.returns: list[str] = []
        self.think = think
        self.thinking_seen = False
        # Что модель получила на каждом запросе — сообщения целиком.
        self.seen: list[list[ModelMessage]] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.seen.append(list(messages))
        self.thinking_seen |= any(
            isinstance(part, ThinkingPart) for message in messages for part in message.parts
        )
        response = self._respond(messages, info)
        if self.think and not any(isinstance(p, ThinkingPart) for p in response.parts):
            response = ModelResponse(parts=[ThinkingPart(_THINKING), *response.parts])
        return response

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.requests += 1
        self.instructions.append(info.instructions or "")
        self.returns = [
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and isinstance(part.content, str)
        ]
        starts = [
            index
            for index, message in enumerate(messages)
            if isinstance(message, ModelRequest)
            and any(isinstance(part, UserPromptPart) for part in message.parts)
        ]
        step = sum(isinstance(message, ModelResponse) for message in messages[starts[-1]:])
        if step == 0:
            self.run += 1
        calls, answer = self.runs[self.run]
        if step < len(calls):
            name, args = calls[step]
            return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args)])
        if self.run in self.as_text:
            assert info.allow_text_output
            return ModelResponse(parts=[ThinkingPart(_THINKING), TextPart(answer)])
        assert info.output_tools
        args: dict[str, Any] = {"answer": answer}
        if self.run in self.assumptions:
            args["assumption"] = self.assumptions[self.run]
        output = ToolCallPart(info.output_tools[0].name, args)
        if self.chatter:
            return ModelResponse(parts=[TextPart(f"Итак: {answer}"), output])
        return ModelResponse(parts=[output])


_VPIP = ("hero_frequency", {"stat": "vpip"})
_GOOD = "Вы входите в банк в 100.0% раздач (2 из 2)."


async def _turn(
    db_factory, session, script, *, scope, text="вопрос", history=(), context=None,
    system=_SYSTEM, analysis_id=None,
):
    player_id, session_id, trace_id = scope
    llm = LLM(_TEST_CFG, db_factory, model_override=FunctionModel(script))
    return await run_agent_turn(
        session, llm, player_id=player_id, session_id=session_id, text=text,
        trace_id=trace_id, history=history, context=context, system_prompt=system,
        analysis_id=analysis_id,
    )


async def test_a_number_from_the_tool_reaches_the_player_with_its_calculation(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8101)
        turn = await _turn(db_factory, session, _TurnScript([([_VPIP], _GOOD)]), scope=scope)
    assert turn.state == "prose" and turn.prose == _GOOD
    assert [r.calc for r in turn.results] == [CalcName.HERO_FREQUENCY]


async def test_several_calculations_in_one_turn_are_all_listed(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8102)
        script = _TurnScript([([_VPIP, ("leaks", {})], _GOOD)])
        turn = await _turn(db_factory, session, script, scope=scope)
    assert [r.calc for r in turn.results] == [CalcName.HERO_FREQUENCY, CalcName.LEAKS]


async def test_the_ninth_calculation_is_refused(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8103)
        script = _TurnScript([([_VPIP] * (MAX_CALCS_PER_TURN + 1), _GOOD)])
        turn = await _turn(db_factory, session, script, scope=scope)
    assert len(turn.results) == MAX_CALCS_PER_TURN
    assert any("8" in text and "расчёт" in text.lower() for text in script.returns)
    assert [c.get("refused", False) for c in turn.calls] == [False] * MAX_CALCS_PER_TURN + [True]


async def test_an_answer_from_the_context_alone_passes_without_a_call(db_factory):
    """Вопрос про руку часто отвечается сводкой: числа из неё разрешены."""
    ctx = agent_context(_result([_point(dp_index=0, ev_diff_bb=-1.4)]), _postflop_hand())
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8104)
        script = _TurnScript([([], "Банк на флопе был 6.6, вы сдались на ставку 3.0.")])
        turn = await _turn(db_factory, session, script, scope=scope, context=ctx)
    assert turn.state == "prose" and turn.results == () and turn.from_context
    assert _SYSTEM in script.instructions[0] and "6.6" in script.instructions[0]


async def test_an_answer_without_a_call_and_without_a_context_is_refused(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8105)
        script = _TurnScript([([], "На BB вы минусовой, это нормально.")])
        turn = await _turn(db_factory, session, script, scope=scope)
    assert turn.state == "refusal" and turn.prose is None


async def test_one_rewrite_then_numbers_without_prose(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8106)
        script = _TurnScript([
            ([_VPIP], "Вы входите в 43.0% раздач."),
            ([], "Вы входите в 44.0% раздач."),
        ])
        turn = await _turn(db_factory, session, script, scope=scope)
    assert turn.state == "numbers" and turn.prose is None
    assert len(turn.violations) == 2 and script.run == 1, "переписать — ровно один раз"


async def test_a_rewrite_that_passes_shows_the_prose(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8107)
        script = _TurnScript([([_VPIP], "Вы входите в 43.0% раздач."), ([], _GOOD)])
        turn = await _turn(db_factory, session, script, scope=scope)
    assert turn.state == "prose" and turn.prose == _GOOD


async def test_a_number_from_an_earlier_turn_of_the_evening_stays_allowed(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8108)
        first = await _turn(db_factory, session, _TurnScript([([_VPIP], _GOOD)]), scope=scope)
        second = await _turn(
            db_factory, session, _TurnScript([([], "Те же 100.0% (2 из 2).")]),
            scope=scope, history=first.messages,
        )
    assert second.state == "prose"


async def test_the_word_leak_without_the_leaks_calculation_is_rewritten(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8109)
        script = _TurnScript([([_VPIP], _GOOD + " Это ваш лик."), ([], _GOOD)])
        turn = await _turn(db_factory, session, script, scope=scope)
    assert turn.state == "prose" and any("лик" in v for v in turn.violations[0])


async def test_a_turn_with_three_model_requests_logs_three_agent_rows(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8110)
        await _turn(db_factory, session, _TurnScript([([_VPIP, ("leaks", {})], _GOOD)]), scope=scope)
    async with db_factory() as session:
        rows = (await session.execute(sql_text("select purpose, status from llm_calls order by id"))).all()
    assert [tuple(r) for r in rows] == [("agent", "ok")] * 3


async def test_history_survives_serialization(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8111)
        turn = await _turn(db_factory, session, _TurnScript([([_VPIP], _GOOD)]), scope=scope)
    restored = load_history(dump_history(turn.messages))
    assert calcs_in_history(restored) == {"hero_frequency"}
    assert allowed_from_history(restored) >= {100.0, 2.0}


async def test_trimming_drops_whole_turns_and_their_numbers(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8112)
        first = await _turn(db_factory, session, _TurnScript([([_VPIP], _GOOD)]), scope=scope)
        second = await _turn(
            db_factory, session, _TurnScript([([("leaks", {})], "Расхождений нет.")]),
            scope=scope, history=first.messages,
        )
    assert 100.0 in allowed_from_history(second.messages)
    last_turn_only = trim_history(second.messages, max_tokens=_tokens_of_last_turn(second.messages))
    assert calcs_in_history(last_turn_only) == {"leaks"}
    assert "hero_frequency" not in calcs_in_history(last_turn_only)
    first_numbers = allowed_from_history(first.messages) - allowed_from_history(last_turn_only)
    assert first_numbers, "выброшенный ход не унёс ни одного числа — проверять нечего"


async def test_trimming_leaves_no_orphan_tool_return(db_factory):
    """Регрессия: обрезка по ходам не отрывает вызов от ответа. Ответ на
    последний вызов прошлого хода (`final_result`) остаётся в своём ходе, и
    оставшаяся история начинается репликой игрока, а не ответом на вызов,
    которого нет, — такую историю провайдер отвергает. Падает, если `turns()`
    начнёт резать ход на любом запросе."""
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8113)
        first = await _turn(db_factory, session, _TurnScript([([_VPIP], _GOOD)]), scope=scope)
        second = await _turn(
            db_factory, session, _TurnScript([([], "Те же 100.0% (2 из 2).")]),
            scope=scope, history=first.messages,
        )
    trimmed = trim_history(second.messages, max_tokens=_tokens_of_last_turn(second.messages))
    head = trimmed[0]
    assert isinstance(head, ModelRequest)
    assert all(isinstance(part, UserPromptPart) for part in head.parts)


# --- числа игрока (решение владельца R1) и метки частот ------------------------------


async def test_a_number_the_player_typed_is_allowed(db_factory):
    """Игрок сам назвал «12 BB» — повторить его число не выдумка."""
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8114)
        script = _TurnScript([([_VPIP], _GOOD + " При ваших 12 BB это важно.")])
        turn = await _turn(db_factory, session, script, scope=scope, text="у меня 12 BB")
    assert turn.state == "prose" and len(turn.violations) == 1


async def test_a_number_nobody_said_is_still_rejected(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8115)
        bad = _GOOD + " При ваших 13 BB это важно."
        script = _TurnScript([([_VPIP], bad), ([], bad)])
        turn = await _turn(db_factory, session, script, scope=scope, text="у меня 12 BB")
    assert turn.state == "numbers"
    assert all(any("13" in v for v in found) for found in turn.violations)


async def test_a_number_the_player_typed_earlier_in_the_evening_stays_allowed(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8116)
        first = await _turn(
            db_factory, session, _TurnScript([([_VPIP], _GOOD)]), scope=scope, text="у меня 12 BB"
        )
        second = await _turn(
            db_factory, session, _TurnScript([([], "При 12 BB — те же 100.0%.")]),
            scope=scope, history=first.messages,
        )
    assert second.state == "prose"


async def test_the_rewrite_request_does_not_allow_the_numbers_it_lists(db_factory):
    """Список нарушений уходит модели репликой, и выдуманное число в нём есть;
    повтор того же числа после переписывания всё равно не проходит."""
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8117)
        bad = "Вы входите в 43.0% раздач."
        script = _TurnScript([([_VPIP], bad), ([], bad)])
        turn = await _turn(db_factory, session, script, scope=scope)
    assert turn.state == "numbers"
    assert any("43" in v for v in turn.violations[1])


async def test_a_rewrite_stays_in_the_turn_of_its_question(db_factory):
    """В сохранённой истории ход с попыткой переписать — один ход: обрезка не
    отрывает ответ от вопроса и его расчётов."""
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8118)
        script = _TurnScript([([_VPIP], "Вы входите в 43.0% раздач."), ([], _GOOD)])
        turn = await _turn(db_factory, session, script, scope=scope)
    assert len(turns(turn.saved_messages(None))) == 1


async def test_a_frequency_term_is_allowed_by_its_calculation(db_factory):
    """Метка частоты из выхода расчёта — с пространством имён `stat:`."""
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8119)
        script = _TurnScript([
            ([("hero_frequency", {"stat": "cbet_flop"})], "По c-бету на флопе наблюдений нет."),
        ])
        turn = await _turn(db_factory, session, script, scope=scope)
    assert turn.state == "prose", turn.violations


async def test_the_journal_names_calls_checks_and_the_rewrite(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8120)
        script = _TurnScript([([_VPIP], "Вы входите в 43.0% раздач."), ([], _GOOD)])
        turn = await _turn(db_factory, session, script, scope=scope)
    journal = turn.journal()
    assert journal["state"] == "prose" and journal["answer"] == _GOOD
    assert journal["rewrite"] is True and journal["checks"][1] == []
    assert [c["calc"] for c in journal["calls"]] == ["hero_frequency"]
    call = journal["calls"][0]
    assert call["zone"] == "strict" and call["params"]["stat"] == "vpip"
    assert "100%" in call["output"] and call["duration_s"] >= 0


def test_instructions_carry_the_system_prompt_and_the_context():
    ctx = agent_context(_result([_point(dp_index=0, ev_diff_bb=-1.4)]), _postflop_hand())
    assert instructions_for(_SYSTEM, ctx).startswith(_SYSTEM)
    assert ctx.text in instructions_for(_SYSTEM, ctx)
    assert ctx.text not in instructions_for(_SYSTEM, None)


@requires_agent_prompt
def test_the_system_prompt_loads():
    assert load_system_prompt().strip()


def _tokens_of_last_turn(messages: list[ModelMessage] | tuple[ModelMessage, ...]) -> int:
    from harness.agent.history import estimate_tokens, turns

    return estimate_tokens(turns(list(messages))[-1])


# --- правки по ревью: потолок истории, сохраняемая история, черновики в журнале -------


async def test_a_long_system_prompt_does_not_eat_the_history_budget(db_factory):
    """Инструкции каждого запроса — промпт и сводка — в историю не попадают и
    потолок не тратят: четыре хода с расчётом при промпте в 7000 знаков влезают в
    30 000, хотя вместе с инструкциями весили бы больше."""
    system = "Синтетический системный промпт. " * 220
    assert len(system) >= 7000
    history: tuple[ModelMessage, ...] = ()
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8121)
        for _ in range(4):
            turn = await _turn(
                db_factory, session, _TurnScript([([_VPIP], _GOOD)]),
                scope=scope, history=history, system=system,
            )
            history = turn.messages
    with_instructions = len(ModelMessagesTypeAdapter.dump_json(list(history))) // 3
    assert with_instructions > 30_000, "сценарий не нагружает потолок — проверять нечего"
    kept = trim_history(history, max_tokens=30_000)
    assert len(turns(kept)) == 4
    assert all(m.instructions is None for m in kept if isinstance(m, ModelRequest))


def _stored_text(messages) -> str:
    """Тексты частей сохранённых сообщений — реплики, вызовы с аргументами, выходы
    расчётов, рассуждения. Без меток времени: в JSON всего сообщения «43.0» находилось
    в секундах метки, и проверки «числа нет» падали через раз."""
    chunks: list[str] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                chunks.append(part.args_as_json_str())
            elif isinstance(part, ToolReturnPart):
                chunks.append(part.model_response_str())
            elif isinstance(part, UserPromptPart):
                content = part.content
                chunks.append(content if isinstance(content, str) else " ".join(map(str, content)))
            elif isinstance(part, TextPart | ThinkingPart):
                chunks.append(part.content)
            else:
                chunks.append(str(getattr(part, "content", "")))
    return "\n".join(chunks)


def test_the_stored_text_ignores_timestamps():
    """Флейк из финального ревью: метка времени с секундами 43.0… не число ответа."""
    from datetime import UTC, datetime

    stamp = datetime(2026, 1, 1, 0, 0, 43, 5, tzinfo=UTC)
    message = ModelResponse(parts=[TextPart(_GOOD)], timestamp=stamp)
    assert "43.0" in ModelMessagesTypeAdapter.dump_json([message]).decode()
    assert "43.0" not in _stored_text([message]) and _GOOD in _stored_text([message])


async def test_the_saved_history_keeps_no_rejected_draft_and_no_rewrite_request(db_factory):
    """Решение контроллера (вопрос владельца 2): в истории — то, что игрок
    видел, и вызовы расчётов. Следующий ход принимает такую историю."""
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8122)
        script = _TurnScript([
            ([_VPIP], "Вы входите в 43.0% раздач."),
            ([], "Вы входите в 44.0% раздач."),
        ])
        first = await _turn(db_factory, session, script, scope=scope)
        assert first.state == "numbers"
        saved = load_history(dump_history(first.saved_messages("ПОКАЗАНО-СИНТ")))
        stored = _stored_text(saved)
        assert "43.0" not in stored and "44.0" not in stored and REWRITE_HEAD not in stored
        assert "ПОКАЗАНО-СИНТ" in stored and calcs_in_history(saved) == {"hero_frequency"}
        second = await _turn(
            db_factory, session, _TurnScript([([], "Те же 100.0% (2 из 2).")]),
            scope=scope, history=saved,
        )
    assert second.state == "prose"


async def test_a_refused_answer_is_saved_as_the_refusal_the_player_saw(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8123)
        script = _TurnScript([([], "На BB вы минусовой, это нормально.")])
        turn = await _turn(db_factory, session, script, scope=scope)
    stored = _stored_text(turn.saved_messages("ОТКАЗ-СИНТ"))
    assert "минусовой" not in stored and "ОТКАЗ-СИНТ" in stored


async def test_a_passed_prose_is_saved_as_it_is(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8124)
        script = _TurnScript([([_VPIP], "Вы входите в 43.0% раздач."), ([], _GOOD)])
        turn = await _turn(db_factory, session, script, scope=scope)
    stored = _stored_text(turn.saved_messages(None))
    assert _GOOD in stored and "43.0" not in stored and REWRITE_HEAD not in stored


async def test_the_journal_keeps_the_rejected_drafts(db_factory):
    """Решение контроллера (вопрос владельца 3): черновики, которых игрок не
    видел, — в журнале, для evals и «не то посчитано»."""
    bad, worse = "Вы входите в 43.0% раздач.", "Вы входите в 44.0% раздач."
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8125)
        numbers = await _turn(db_factory, session, _TurnScript([([_VPIP], bad), ([], worse)]), scope=scope)
        rewritten = await _turn(db_factory, session, _TurnScript([([_VPIP], bad), ([], _GOOD)]), scope=scope)
        clean = await _turn(db_factory, session, _TurnScript([([_VPIP], _GOOD)]), scope=scope)
        refused = await _turn(db_factory, session, _TurnScript([([], "Без расчёта.")]), scope=scope)
    assert numbers.journal()["drafts"] == [bad, worse]
    assert rewritten.journal()["drafts"] == [bad]
    assert clean.journal()["drafts"] == []
    assert refused.journal()["drafts"] == ["Без расчёта."]


# --- правки по второму ревью: текст рядом с вызовами, просьба переписать по месту ----


def _alternates(messages) -> bool:
    """Ответы модели не стоят подряд: между ними всегда запрос."""
    kinds = [type(m) for m in messages]
    return all(not (a is ModelResponse and b is ModelResponse) for a, b in pairwise(kinds))


async def test_text_next_to_a_rejected_draft_is_not_saved(db_factory):
    bad = "Вы входите в 43.0% раздач."
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8126)
        script = _TurnScript([([_VPIP], bad), ([], _GOOD)], chatter=True)
        turn = await _turn(db_factory, session, script, scope=scope)
        assert turn.state == "prose"
        saved = load_history(dump_history(turn.saved_messages(None)))
        stored = _stored_text(saved)
        assert "43.0" not in stored and "Итак" not in stored and REWRITE_HEAD not in stored
        assert _GOOD in stored and _alternates(saved)
        assert not any(isinstance(p, TextPart) for m in saved for p in m.parts)
        nxt = await _turn(
            db_factory, session, _TurnScript([([], "Те же 100.0% (2 из 2).")]),
            scope=scope, history=saved,
        )
    assert nxt.state == "prose"


async def test_text_next_to_the_final_answer_is_not_saved(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8127)
        bad, worse = "Вы входите в 43.0% раздач.", "Вы входите в 44.0% раздач."
        script = _TurnScript([([_VPIP], bad), ([], worse)], chatter=True)
        turn = await _turn(db_factory, session, script, scope=scope)
    assert turn.state == "numbers"
    saved = turn.saved_messages("ПОКАЗАНО-СИНТ")
    stored = _stored_text(saved)
    assert "Итак" not in stored and "44.0" not in stored and "43.0" not in stored
    assert "ПОКАЗАНО-СИНТ" in stored and _alternates(saved)


async def test_a_player_message_that_looks_like_a_rewrite_request_is_kept(db_factory):
    """Просьба переписать опознаётся по месту в ходе, а не по тексту: реплика
    игрока с той же фразой — его реплика, её числа разрешены, а ответ прошлого
    хода не считается отвергнутым черновиком."""
    player_text = REWRITE_HEAD + " у меня 12 BB"
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8128)
        first = await _turn(db_factory, session, _TurnScript([([_VPIP], _GOOD)]), scope=scope)
        history = first.saved_messages(None)
        second = await _turn(
            db_factory, session,
            _TurnScript([([_VPIP], _GOOD + " При ваших 12 BB это важно.")]),
            scope=scope, text=player_text, history=history,
        )
    assert second.state == "prose" and len(second.violations) == 1
    saved = second.saved_messages(None)
    stored = _stored_text(saved)
    assert player_text in stored and stored.count(_GOOD) == 2
    assert len(turns(saved)) == 2


# --- правки финального ревью: карты игрока, размеры от модели, честная подпись -------


async def test_a_card_rank_the_player_typed_is_not_an_allowed_number(db_factory):
    """«K9o» — карта, а не число: выдуманные «9 BB» в ответе не проходят (I1)."""
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8129)
        bad = _GOOD + " Вы вкладываете 9 BB."
        script = _TurnScript([([_VPIP], bad), ([], bad)])
        turn = await _turn(db_factory, session, script, scope=scope, text="у меня K9o, почему фолд?")
    assert turn.state == "numbers"
    assert all(any("9" in v for v in found) for found in turn.violations)


async def test_a_card_rank_the_player_typed_earlier_is_not_an_allowed_number(db_factory):
    """То же по истории вечера: ранг из прошлой реплики игрока числом не становится,
    а число игрока рядом с картой остаётся разрешённым."""
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8130)
        first = await _turn(
            db_factory, session, _TurnScript([([_VPIP], _GOOD)]),
            scope=scope, text="у меня K9o и 12 BB",
        )
        bad = "При 12 BB вы вкладываете 9 BB."
        second = await _turn(
            db_factory, session, _TurnScript([([], bad), ([], "При 12 BB — те же 100.0%.")]),
            scope=scope, history=first.messages,
        )
    assert second.state == "prose" and second.prose == "При 12 BB — те же 100.0%."
    assert second.violations[0] and all("12" not in v for v in second.violations[0])


def _ctx():
    return agent_context(_result([_point(dp_index=0, ev_diff_bb=-1.4)]), _postflop_hand())


# Банк 6.6 и ставка CO 3.0 — как их печатает сводка `_ctx()` («Флоп: … (банк 6.6):
# … CO бет 3.0»); 7.0 в сводке нет.
_FROM_HAND = {"pot_before_bb": 6.6, "bet_bb": 3.0}
_INVENTED = {"pot_before_bb": 7.0, "bet_bb": 3.0}


async def test_an_invented_pot_is_refused_by_the_tool_and_allows_nothing(db_factory):
    """Решение контроллера (I2, вариант b): банк и ставку, которых нет ни в
    сводке, ни у игрока, ни в выходе расчётов вечера, инструмент не считает."""
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8131)
        script = _TurnScript([
            ([("defense_frequency", _INVENTED)], "Сводка разбора есть."),
        ])
        turn = await _turn(db_factory, session, script, scope=scope, context=_ctx())
    assert turn.results == ()
    assert any("сводк" in r and "игрок" in r for r in script.returns)
    assert not allowed_from_history(turn.messages) & {7.0, 3.0}
    assert calcs_in_history(turn.messages) == frozenset()
    assert turn.calls[0]["calc"] == "defense_frequency" and "error" in turn.calls[0]


async def test_an_invented_bet_size_threshold_is_refused(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8132)
        args = {"stat": "fold_to_cbet", "threshold_source": "bet_size", **_INVENTED}
        script = _TurnScript([([("frequency_vs_threshold", args)], "Сводка разбора есть.")])
        turn = await _turn(db_factory, session, script, scope=scope, context=_ctx())
    assert turn.results == ()
    assert not allowed_from_history(turn.messages) & {7.0, 3.0}


async def test_a_pot_from_the_hand_summary_is_computed_and_shown_in_bb(db_factory):
    """Банк 6.6 и ставка 3.0 — числа сводки в BB, как их печатает разбор: расчёт
    идёт, и модель и игрок видят те же BB, а не фишки масштаба (решение
    контроллера по единицам)."""
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8133)
        script = _TurnScript([([("defense_frequency", _FROM_HAND)], "Сводка разбора есть.")])
        turn = await _turn(db_factory, session, script, scope=scope, context=_ctx())
    assert [r.calc for r in turn.results] == [CalcName.DEFENSE_FREQUENCY]
    result = turn.results[0]
    assert isinstance(result, DefenseResult)
    assert (result.pot_before, result.bet) == (660, 300)  # сотые доли BB, не фишки руки
    echoed = script.returns[-1]
    assert "6.6 BB" in echoed and "3.0 BB" in echoed and "660" not in echoed
    shown = agent_turn_msg("numbers", None, turn.results, from_context=False, job_id=1).text
    assert "6.6BB" in shown and "3.0BB" in shown
    assert "660" not in shown and "300" not in shown
    assert turn.calls[0]["params"]["sizes_bb"] == {"pot_before_bb": 6.6, "bet_bb": 3.0}


async def test_sizes_in_bb_give_the_same_result_as_chips_of_the_same_ratio(db_factory):
    """Выход защиты зависит только от отношения ставки к банку: 7.7/3.3 BB у
    агента и 7700/3300 фишек у `/ask` — одни и те же частоты и эквити."""
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8137)
        script = _TurnScript([
            ([("defense_frequency", {"pot_before_bb": 7.7, "bet_bb": 3.3})], "Посчитано."),
        ])
        turn = await _turn(
            db_factory, session, script, scope=scope, text="банк 7.7 BB, ставка 3.3 BB"
        )
    agent = turn.results[0]
    chips = defense_from_bet(7700, 3300)
    assert isinstance(agent, DefenseResult)
    assert (agent.defend_frequency, agent.fold_frequency, agent.required_equity) == (
        chips.defend_frequency, chips.fold_frequency, chips.required_equity,
    )


async def test_a_non_positive_size_is_refused_by_the_tool(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8138)
        script = _TurnScript([
            ([("defense_frequency", {"pot_before_bb": 6.6, "bet_bb": 0})], "Сводка разбора есть."),
        ])
        turn = await _turn(db_factory, session, script, scope=scope, context=_ctx())
    assert turn.results == ()
    assert any("положительн" in r for r in script.returns)
    assert "error" in turn.calls[0]


def test_agent_tools_take_sizes_in_bb_and_ask_tools_in_chips():
    """У агента размеры в BB, как в разборе; `/ask` как был — в фишках."""
    from harness.calcs.routing import CalcRun, tools_for

    def schemas(run_state):
        tools = {t.name: t for t in tools_for(None, 1, 1, run_state)}  # type: ignore[arg-type]
        return {
            name: tools[name].function_schema.json_schema["properties"]
            for name in ("defense_frequency", "frequency_vs_threshold")
        }

    agent = schemas(CalcRun(max_calcs=8, sizes_from=frozenset()))
    ask = schemas(CalcRun())
    for props in agent.values():
        assert "pot_before" not in props and "bet" not in props
        for key in ("pot_before_bb", "bet_bb"):
            assert "в BB, как в разборе" in props[key]["description"]
    assert agent["defense_frequency"]["pot_before_bb"]["type"] == "number"
    for props in ask.values():
        assert "pot_before" in props and "bet" in props and "pot_before_bb" not in props
    assert ask["defense_frequency"]["pot_before"]["type"] == "integer"


async def test_a_pot_the_player_typed_is_computed(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8134)
        script = _TurnScript([
            ([("defense_frequency", {"pot_before_bb": 7.7, "bet_bb": 3.3})], "Посчитано."),
        ])
        turn = await _turn(
            db_factory, session, script, scope=scope,
            text="банк 7.7 BB, ставка 3.3 BB — сколько защищать?",
        )
    assert [r.calc for r in turn.results] == [CalcName.DEFENSE_FREQUENCY]


async def test_the_signature_flags_name_the_sources_of_a_turn_without_calculations(db_factory):
    """Нет расчётов в ходе: `from_context` — есть сводка разбора, `from_history` —
    в истории вечера остались расчёты (M4)."""
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8135)
        first = await _turn(db_factory, session, _TurnScript([([_VPIP], _GOOD)]), scope=scope)
        both = await _turn(
            db_factory, session, _TurnScript([([], "Те же 100.0% (2 из 2).")]),
            scope=scope, history=first.messages, context=_ctx(),
        )
        history_only = await _turn(
            db_factory, session, _TurnScript([([], "Те же 100.0% (2 из 2).")]),
            scope=scope, history=first.messages,
        )
        context_only = await _turn(
            db_factory, session, _TurnScript([([], "Банк на флопе был 6.6.")]),
            scope=scope, context=_ctx(),
        )
        fresh = await _turn(
            db_factory, session, _TurnScript([([_VPIP], _GOOD)]),
            scope=scope, history=first.messages, context=_ctx(),
        )
    assert (both.from_context, both.from_history) == (True, True)
    assert (history_only.from_context, history_only.from_history) == (False, True)
    assert (context_only.from_context, context_only.from_history) == (True, False)
    assert (fresh.from_context, fresh.from_history) == (False, False)


async def test_a_size_from_an_earlier_calculation_of_the_same_turn_is_computed(db_factory):
    """Выход расчёта этого хода — тоже код: его число годится в размер."""
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8136)
        script = _TurnScript([
            ([_VPIP, ("defense_frequency", {"pot_before_bb": 100, "bet_bb": 2})], _GOOD),
        ])
        turn = await _turn(db_factory, session, script, scope=scope)
    assert [r.calc for r in turn.results] == [CalcName.HERO_FREQUENCY, CalcName.DEFENSE_FREQUENCY]


def test_ask_does_not_check_the_sizes():
    """`/ask` (`CalcRun()` без `sizes_from`): размеры из вопроса игрока не сверяются."""
    from harness.calcs.routing import CalcRun

    assert CalcRun().unsourced_sizes(CalcName.DEFENSE_FREQUENCY, 777, 333) is None


async def test_the_agent_sees_its_own_output(db_factory):
    """Агенту — выход без имени расчёта в скобках, с уровнем данных и выборкой
    словами. Что у `/ask` выход прежний, держит
    `test_agent_outputs.test_the_ask_digest_is_unchanged_by_the_agent_one`."""
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8141)
        script = _TurnScript([([_VPIP], _GOOD)])
        await _turn(db_factory, session, script, scope=scope)
    echoed = script.returns[-1]
    assert "(hero_frequency)" not in echoed and "Оценки у этой величины нет" not in echoed
    assert "Уровень данных: вы, все позиции." in echoed
    assert "Выборка мала (меньше 20 возможностей)." in echoed


# Порог от ставки 1 BB в банк 100 BB: сбрасывать достаточно ≈1%, а вход в банк
# героя — 2 из 2, так что вывод расчёта — выше порога.
_DECIDED = (
    "frequency_vs_threshold",
    {"stat": "vpip", "threshold_source": "bet_size", "pot_before_bb": 100, "bet_bb": 1},
)


async def test_a_decided_threshold_in_the_conversation_allows_a_profit_conclusion(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8142)
        without = await _turn(
            db_factory, session, _TurnScript([([_VPIP], "Ставка в плюсе."), ([], "Ставка в плюсе.")]),
            scope=scope,
        )
        decided = await _turn(
            db_factory, session, _TurnScript([([_DECIDED], "Ставка в плюсе.")]),
            scope=scope, text="банк 100 BB, ставка 1 BB",
        )
        later = await _turn(
            db_factory, session, _TurnScript([([], "Ставка в плюсе.")]),
            scope=scope, history=decided.messages,
        )
    assert without.state == "numbers"
    assert any("frequency_vs_threshold" in v for v in without.violations[0])
    assert decided.results[0].outcome.value == "above"  # type: ignore[union-attr]
    assert decided.state == "prose", decided.violations
    assert later.state == "prose", later.violations


# Порог от ставки 1 BB в банк 1 BB: сбрасывать достаточно 50%, а 2 из 2 у героя
# интервал не отделяет от 50% — вывод «не определено».
_UNDECIDED = (
    "frequency_vs_threshold",
    {"stat": "vpip", "threshold_source": "bet_size", "pot_before_bb": 1, "bet_bb": 1},
)


async def test_an_undecided_threshold_does_not_allow_a_profit_conclusion(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8143)
        turn = await _turn(
            db_factory, session,
            _TurnScript([([_UNDECIDED], "Ставка в плюсе."), ([], "Ставка в плюсе.")]),
            scope=scope, text="банк 1 BB, ставка 1 BB",
        )
    assert turn.results[0].outcome.value == "undecided"  # type: ignore[union-attr]
    assert turn.state == "numbers"
    assert any("frequency_vs_threshold" in v for v in turn.violations[0])


async def test_the_threshold_of_frequency_vs_threshold_reaches_the_check(db_factory):
    """Порог выхода (50% от ставки 1 BB в банк 1 BB) лежит в метаданных отдельно и
    доходит до проверки: «Порог фолдов у BB 50%» — имя порога, а не утверждение."""
    from harness.agent.history import threshold_numbers_in

    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8146)
        turn = await _turn(
            db_factory, session, _TurnScript([([_UNDECIDED], "Порог фолдов у BB 50%.")]),
            scope=scope, text="банк 1 BB, ставка 1 BB",
        )
    assert threshold_numbers_in(turn.messages) == frozenset({50.0})
    assert turn.state == "prose", turn.violations


async def test_the_measured_frequency_is_kept_apart_from_other_calc_numbers(db_factory):
    """Метаданные ответа инструмента несут измеренную частоту отдельно: у защиты её
    нет, у частоты героя — доля, числитель и знаменатель."""
    from harness.agent.history import frequency_numbers

    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8144)
        script = _TurnScript([
            ([_VPIP, ("defense_frequency", {"pot_before_bb": 7.7, "bet_bb": 3.3})], _GOOD),
        ])
        turn = await _turn(db_factory, session, script, scope=scope, text="банк 7.7 BB, ставка 3.3 BB")
    assert frequency_numbers(turn.messages) == frozenset({2.0, 100.0})
    assert 30.0 in allowed_from_history(turn.messages)  # доля защиты — расчёт, не частота


# --- ответ простым текстом: модель с `tool_choice: auto` не всегда зовёт вывод ------


async def test_a_final_answer_in_plain_text_is_shown_and_saved(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8140)
        script = _TurnScript([([_VPIP], f"  {_GOOD}\n")], as_text={0})
        turn = await _turn(db_factory, session, script, scope=scope)
        assert turn.state == "prose" and turn.prose == _GOOD
        assert [r.calc for r in turn.results] == [CalcName.HERO_FREQUENCY]
        saved = load_history(dump_history(turn.saved_messages(None)))
        stored = _stored_text(saved)
        assert _GOOD in stored and _THINKING not in stored and _alternates(saved)
        assert calcs_in_history(saved) == {"hero_frequency"}
        nxt = await _turn(
            db_factory, session, _TurnScript([([], "Те же 100.0% (2 из 2).")], as_text={0}),
            scope=scope, history=saved,
        )
    assert nxt.state == "prose"
    assert len(turns(nxt.saved_messages(None))) == 2


async def test_a_plain_text_answer_goes_through_the_checks(db_factory):
    """Текст проверяется так же, как вывод: выдуманное число — переписать."""
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8141)
        script = _TurnScript(
            [([_VPIP], "Вы входите в 43.0% раздач."), ([], "Вы входите в 44.0% раздач.")],
            as_text={0, 1},
        )
        turn = await _turn(db_factory, session, script, scope=scope)
    assert turn.state == "numbers" and len(turn.violations) == 2
    assert turn.journal()["drafts"] == ["Вы входите в 43.0% раздач.", "Вы входите в 44.0% раздач."]
    saved = turn.saved_messages("ПОКАЗАНО-СИНТ")
    stored = _stored_text(saved)
    assert "43.0" not in stored and "44.0" not in stored and REWRITE_HEAD not in stored
    assert "ПОКАЗАНО-СИНТ" in stored and _alternates(saved)


async def test_a_rejected_text_draft_is_not_saved_and_the_text_rewrite_is(db_factory):
    bad = "Вы входите в 43.0% раздач."
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8142)
        script = _TurnScript([([_VPIP], bad), ([], _GOOD)], as_text={0, 1})
        turn = await _turn(db_factory, session, script, scope=scope)
        assert turn.state == "prose" and turn.prose == _GOOD
        saved = load_history(dump_history(turn.saved_messages(None)))
        stored = _stored_text(saved)
        assert "43.0" not in stored and REWRITE_HEAD not in stored and _THINKING not in stored
        assert stored.count(_GOOD) == 1 and _alternates(saved)
        nxt = await _turn(
            db_factory, session, _TurnScript([([], "Те же 100.0% (2 из 2).")]),
            scope=scope, history=saved,
        )
    assert nxt.state == "prose"


async def test_a_text_draft_and_a_tool_rewrite_mix(db_factory):
    """Черновик текстом отвергнут, переписано вызовом вывода — и наоборот."""
    bad = "Вы входите в 43.0% раздач."
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8143)
        for as_text in ({0}, {1}):
            script = _TurnScript([([_VPIP], bad), ([], _GOOD)], as_text=as_text)
            turn = await _turn(db_factory, session, script, scope=scope)
            assert turn.state == "prose" and turn.prose == _GOOD, as_text
            saved = load_history(dump_history(turn.saved_messages(None)))
            stored = _stored_text(saved)
            assert "43.0" not in stored and REWRITE_HEAD not in stored, as_text
            assert stored.count(_GOOD) == 1 and _alternates(saved), as_text
            nxt = await _turn(
                db_factory, session, _TurnScript([([], "Те же 100.0% (2 из 2).")]),
                scope=scope, history=saved,
            )
            assert nxt.state == "prose", as_text


async def test_a_refused_text_answer_is_saved_as_the_refusal_the_player_saw(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8144)
        script = _TurnScript([([], "На BB вы минусовой, это нормально.")], as_text={0})
        turn = await _turn(db_factory, session, script, scope=scope)
    assert turn.state == "refusal"
    stored = _stored_text(turn.saved_messages("ОТКАЗ-СИНТ"))
    assert "минусовой" not in stored and "ОТКАЗ-СИНТ" in stored


async def test_a_blank_text_answer_is_a_schema_error(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8145)
        script = _TurnScript([([_VPIP], " \n "), ([_VPIP], " \n ")], as_text={0, 1})
        with pytest.raises(LLMSchemaError):
            await _turn(db_factory, session, script, scope=scope)


# --- диапазон соперника по чарту и эквити против суженного (только агент) ----------

_RANGE_CALL = ("opponent_range", {"opponent_position": "BB", "street": "flop"})
_EQUITY_CALL = (
    "equity_vs_range", {"opponent_position": "BB", "street": "flop", "hands": "QQ, 77, 22, AQs"},
)


async def _evening_with_hand(session, tg_user_id: int):
    scope = await _seeded(session, tg_user_id=tg_user_id)
    analysis_id = await stored_analysis(session, session_id=scope[1])
    await SessionsRepo(session).set_last_analysis(scope[1], analysis_id)
    await session.commit()
    return scope, analysis_id


def test_the_agent_tool_set_covers_every_calculation_name_exactly_once():
    tools = tools_for(None, 1, 1, CalcRun(max_calcs=8, sizes_from=frozenset()))  # type: ignore[arg-type]
    assert sorted(tool.name for tool in tools) == sorted(name.value for name in CalcName)


def test_ask_does_not_get_the_agent_only_calculations():
    tools = tools_for(None, 1, 1, CalcRun())  # type: ignore[arg-type]
    assert not {tool.name for tool in tools} & {name.value for name in AGENT_ONLY_CALCS}


async def test_the_agent_reads_the_range_by_the_chart_and_counts_equity_against_its_narrowing(
    db_factory, tmp_path, monkeypatch
):
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    monkeypatch.setattr(calcs_module, "AGENT_EQUITY_ITERATIONS", 2_000)
    async with db_factory() as session:
        scope, analysis_id = await _evening_with_hand(session, tg_user_id=8301)
        script = _TurnScript([([_RANGE_CALL, _EQUITY_CALL], "Ответ без чисел (допущение).")])
        turn = await _turn(db_factory, session, script, scope=scope, analysis_id=analysis_id)
    assert [r.calc for r in turn.results] == [CalcName.OPPONENT_RANGE, CalcName.EQUITY_VS_RANGE]
    assert [c["zone"] for c in turn.calls] == ["strict", "assuming"]
    range_result, equity_result = turn.results
    assert isinstance(range_result, OpponentRangeResult)
    assert isinstance(equity_result, EquityVsRangeResult)
    assert chart_numbers_in(turn.messages) == frozenset(
        {round(100.0 * range_result.hands_share, 1), float(round(range_result.live_combos))}
    )
    assert assumed_numbers_in(turn.messages) == frozenset({round(100.0 * equity_result.equity, 1)})


async def test_equity_numbers_are_calculation_numbers_and_never_a_measured_frequency(
    db_factory, tmp_path, monkeypatch
):
    """Эквити и доля от стартового — числа расчёта (`allowed`), но не измеренная
    частота: частота — опора действию соперника, а сужение агента ею не бывает."""
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    monkeypatch.setattr(calcs_module, "AGENT_EQUITY_ITERATIONS", 2_000)
    async with db_factory() as session:
        scope, analysis_id = await _evening_with_hand(session, tg_user_id=8304)
        script = _TurnScript([([_RANGE_CALL, _EQUITY_CALL], "Ответ без чисел (допущение).")])
        turn = await _turn(db_factory, session, script, scope=scope, analysis_id=analysis_id)
    assumed = assumed_numbers_in(turn.messages)
    assert assumed and assumed <= allowed_from_history(turn.messages)
    assert chart_numbers_in(turn.messages) <= allowed_from_history(turn.messages)
    assert frequency_numbers(turn.messages) == frozenset()


async def test_the_fold_hint_numbers_are_allowed_but_never_assumed(
    db_factory, tmp_path, monkeypatch
):
    """Ориентир сдачи (доля оставленного, сдача на с-бет с выборкой) — числа выхода:
    они в `allowed`, но не в `assumed` — там только эквити (2f3d92a, 5394010)."""
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    monkeypatch.setattr(calcs_module, "AGENT_EQUITY_ITERATIONS", 500)
    async with db_factory() as session:
        scope, analysis_id = await _evening_with_hand(session, tg_user_id=8305)
        script = _TurnScript([([_EQUITY_CALL], "Ответ без чисел (допущение).")])
        turn = await _turn(db_factory, session, script, scope=scope, analysis_id=analysis_id)
    (result,) = turn.results
    assert isinstance(result, EquityVsRangeResult) and result.fold_hint is not None
    assert result.kept_combos is not None and result.source_combos is not None
    kept = round(100.0 * result.kept_combos / result.source_combos, 1)
    hint = {kept, round(100.0 - kept, 1), 0.0, 1.0}
    output = turn.calls[0]["output"]
    assert "сдаёт" in output and "Для сверки с долей" in output
    assert hint <= allowed_from_history(turn.messages)
    assert assumed_numbers_in(turn.messages) == frozenset({round(100.0 * result.equity, 1)})


async def test_without_a_hand_analysis_the_range_tool_says_so(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8302)
        script = _TurnScript([([_RANGE_CALL], _GOOD)])
        turn = await _turn(db_factory, session, script, scope=scope)
    assert turn.results == ()
    assert any("Разбора руки в этом вечере нет" in text for text in script.returns)


async def test_a_refused_narrowing_reaches_the_model_and_spends_no_calculation(
    db_factory, tmp_path, monkeypatch
):
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    async with db_factory() as session:
        scope, analysis_id = await _evening_with_hand(session, tg_user_id=8303)
        bad = ("equity_vs_range", {"opponent_position": "BB", "street": "flop", "hands": "AKo"})
        script = _TurnScript([([bad], "Ответ без чисел (допущение).")])
        turn = await _turn(db_factory, session, script, scope=scope, analysis_id=analysis_id)
    assert turn.results == ()
    assert "нет в стартовом диапазоне" in turn.calls[0]["error"]


# --- Task 12: блок «Оценка модели» (решение владельца 2026-10-15) -----------------

_BLOCK = "BB держит пары и сбросит слабое на тёрне; против такого диапазона у вас 100.0% раздач."
_BAD_BLOCK = "BB держит пары, эквити против них 55%."


def test_text_with_the_separator_splits_into_answer_and_block():
    draft = AgentDraft.from_text(f"  {_GOOD}\n\nоценка МОДЕЛИ:\n{_BLOCK}\n")
    assert draft.answer == _GOOD and draft.assumption == _BLOCK
    same_line = AgentDraft.from_text(f"{_GOOD}\nОценка модели: {_BLOCK}")
    assert same_line.answer == _GOOD and same_line.assumption == _BLOCK


@pytest.mark.parametrize(
    "separator", ["**Оценка модели:**", "**Оценка модели**:", "> Оценка модели:", "## Оценка модели:"]
)
def test_a_markdown_separator_also_splits_the_text(separator: str):
    draft = AgentDraft.from_text(f"{_GOOD}\n{separator}\n{_BLOCK}")
    assert draft.answer == _GOOD and draft.assumption == _BLOCK


def test_text_without_the_separator_at_a_line_start_is_all_answer():
    inline = f"{_GOOD} Оценка модели: дальше."
    assert AgentDraft.from_text(inline) == AgentDraft(answer=inline)
    assert AgentDraft.from_text(_GOOD).assumption is None
    empty = AgentDraft.from_text(f"{_GOOD}\nОценка модели:\n  ")
    assert empty.answer == _GOOD and empty.assumption is None


async def test_a_block_with_claims_about_the_opponent_passes_by_the_output_call(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8160)
        script = _TurnScript([([_VPIP], _GOOD)], assumptions={0: _BLOCK})
        turn = await _turn(db_factory, session, script, scope=scope)
    assert turn.state == "prose" and turn.prose == _GOOD and turn.assumption == _BLOCK
    assert turn.violations == ((),)
    journal = turn.journal()
    assert journal["assumption"] == _BLOCK and journal["assumption_dropped"] is False
    stored = _stored_text(load_history(dump_history(turn.saved_messages(None))))
    assert _GOOD in stored and _BLOCK in stored


async def test_a_block_by_text_with_the_separator_passes_and_is_saved(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8161)
        script = _TurnScript([([_VPIP], f"{_GOOD}\nОценка модели:\n{_BLOCK}")], as_text={0})
        turn = await _turn(db_factory, session, script, scope=scope)
    assert turn.state == "prose" and turn.prose == _GOOD and turn.assumption == _BLOCK
    stored = _stored_text(load_history(dump_history(turn.saved_messages(None))))
    assert _BLOCK in stored and _THINKING not in stored


@pytest.mark.parametrize("as_text", [frozenset(), frozenset({0, 1})])
async def test_a_block_that_fails_twice_is_dropped_and_the_answer_is_shown(db_factory, as_text):
    """Не прошёл только блок — игрок получает прозу без блока; журнал говорит, что
    блок выброшен, история хранит показанное, без блока."""
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8162 + len(as_text))
        if as_text:
            runs: list[tuple[list[tuple[str, dict[str, Any]]], str]] = [
                ([_VPIP], f"{_GOOD}\nОценка модели:\n{_BAD_BLOCK}")
            ] * 2
            script = _TurnScript(runs, as_text=as_text)
        else:
            script = _TurnScript(
                [([_VPIP], _GOOD), ([], _GOOD)], assumptions={0: _BAD_BLOCK, 1: _BAD_BLOCK}
            )
        turn = await _turn(db_factory, session, script, scope=scope)
        assert turn.state == "prose" and turn.prose == _GOOD and turn.assumption is None
        assert len(turn.violations) == 2
        assert all(v.startswith("блок «Оценка модели»") for v in turn.violations[0])
        journal = turn.journal()
        assert journal["assumption"] is None and journal["assumption_dropped"] is True
        assert len(journal["drafts"]) == 2 and all(_BAD_BLOCK in d for d in journal["drafts"])
        saved = load_history(dump_history(turn.saved_messages(None)))
        stored = _stored_text(saved)
        assert _GOOD in stored and "эквити против них" not in stored
        assert "Оценка модели" not in stored
        assert REWRITE_HEAD not in stored and _alternates(saved)
        nxt = await _turn(
            db_factory, session, _TurnScript([([], "Те же 100.0% (2 из 2).")]),
            scope=scope, history=saved,
        )
    assert nxt.state == "prose"


async def test_a_block_fixed_by_the_rewrite_is_shown(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8165)
        script = _TurnScript(
            [([_VPIP], _GOOD), ([], _GOOD)], assumptions={0: _BAD_BLOCK, 1: _BLOCK}
        )
        turn = await _turn(db_factory, session, script, scope=scope)
    assert turn.state == "prose" and turn.assumption == _BLOCK
    assert turn.journal()["assumption_dropped"] is False


@pytest.mark.parametrize("second_block", [None, _BLOCK], ids=["no-block", "good-block"])
async def test_a_failed_rewrite_keeps_the_first_answer_without_the_block(db_factory, second_block):
    """Решение контроллера по финальному ревью: в первом черновике прошёл ответ, не
    прошёл только блок, а переписанный ответ не прошёл — игрок получает ответ первого
    черновика без блока, а не числа без прозы; история — показанное."""
    bad = "Вы входите в 43.0% раздач."
    assumptions = {0: _BAD_BLOCK} | ({1: second_block} if second_block else {})
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8167 + (second_block is not None))
        script = _TurnScript([([_VPIP], _GOOD), ([], bad)], assumptions=assumptions)
        turn = await _turn(db_factory, session, script, scope=scope)
        assert turn.state == "prose" and turn.prose == _GOOD and turn.assumption is None
        assert len(turn.violations) == 2 and turn.violations[1]
        journal = turn.journal()
        assert journal["answer"] == _GOOD
        assert journal["assumption"] is None and journal["assumption_dropped"] is True
        assert len(journal["drafts"]) == 2
        saved = load_history(dump_history(turn.saved_messages(None)))
        stored = _stored_text(saved)
        assert stored.count(_GOOD) == 1 and "43.0" not in stored
        assert "эквити против них" not in stored and "Оценка модели" not in stored
        assert "сбросит слабое" not in stored
        assert REWRITE_HEAD not in stored and _alternates(saved)
        nxt = await _turn(
            db_factory, session, _TurnScript([([], "Те же 100.0% (2 из 2).")]),
            scope=scope, history=saved,
        )
    assert nxt.state == "prose"


_OTHER_GOOD = "Раздач, где вы входите в банк, 100.0% (2 из 2)."


async def test_a_rewrite_that_passes_fully_replaces_the_first_answer(db_factory):
    """Переписанный черновик прошёл целиком — показывается он, с блоком."""
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8169)
        script = _TurnScript(
            [([_VPIP], _GOOD), ([], _OTHER_GOOD)], assumptions={0: _BAD_BLOCK, 1: _BLOCK}
        )
        turn = await _turn(db_factory, session, script, scope=scope)
    assert turn.state == "prose" and turn.prose == _OTHER_GOOD and turn.assumption == _BLOCK
    assert turn.journal()["assumption_dropped"] is False
    stored = _stored_text(load_history(dump_history(turn.saved_messages(None))))
    assert _OTHER_GOOD in stored and _BLOCK in stored and _GOOD not in stored


async def test_a_failed_answer_with_a_good_block_is_numbers_without_prose(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8166)
        bad = "Вы входите в 43.0% раздач."
        script = _TurnScript([([_VPIP], bad), ([], bad)], assumptions={0: _BLOCK, 1: _BLOCK})
        turn = await _turn(db_factory, session, script, scope=scope)
    assert turn.state == "numbers" and turn.prose is None and turn.assumption is None
    assert turn.journal()["assumption"] is None


# --- рассуждения в истории: подпись привязана к системному промпту и инструментам -----


def _has_thinking(messages) -> bool:
    return any(isinstance(part, ThinkingPart) for message in messages for part in message.parts)


async def test_no_thinking_is_saved_on_any_path_and_the_next_turn_gets_none(db_factory):
    """Рассуждение в каждом ответе: у вызова расчёта, у отвергнутого черновика текстом,
    у переписанного вывода — и в истории вечера, пришедшей со старой строки."""
    stale = [
        ModelRequest(parts=[UserPromptPart("прошлый вопрос")]),
        ModelResponse(parts=[ThinkingPart(_THINKING), TextPart("Прошлый ответ.")]),
    ]
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8180)
        script = _TurnScript(
            [([_VPIP], "Вы входите в 43.0% раздач."), ([], _GOOD)], as_text={0}, think=True
        )
        turn = await _turn(db_factory, session, script, scope=scope, history=stale)
        assert turn.state == "prose"
        saved = load_history(dump_history(turn.saved_messages(None)))
        assert not _has_thinking(turn.saved_messages(None)) and _alternates(saved)
        assert _GOOD in _stored_text(saved) and "Прошлый ответ." in _stored_text(saved)
        nxt_script = _TurnScript([([], "Те же 100.0% (2 из 2).")], as_text={0}, think=True)
        nxt = await _turn(db_factory, session, nxt_script, scope=scope, history=saved)
    assert nxt.state == "prose" and not nxt_script.thinking_seen
    assert not _has_thinking(nxt.saved_messages(None))


async def test_a_stored_row_with_thinking_loads_without_it(db_factory):
    """Строка, сохранённая старым кодом: рассуждение рядом с вызовом и с текстом и
    ответ из одного рассуждения — без рассуждений, пустой ответ уходит целиком."""
    call = ToolCallPart("hero_frequency", {"stat": "vpip"}, tool_call_id="c1")
    old = [
        ModelRequest(parts=[UserPromptPart("первый вопрос")]),
        ModelResponse(parts=[ThinkingPart(_THINKING, signature="подпись-синт"), call]),
        ModelRequest(parts=[ToolReturnPart(
            "hero_frequency", "выход-синт", tool_call_id="c1",
            metadata={"calc": "hero_frequency", "allowed": [100.0, 2.0]},
        )]),
        ModelResponse(parts=[ThinkingPart(_THINKING), TextPart("Первый ответ.")]),
        ModelRequest(parts=[UserPromptPart("второй вопрос")]),
        ModelResponse(parts=[ThinkingPart(_THINKING)]),
        ModelRequest(parts=[UserPromptPart("третий вопрос")]),
        ModelResponse(parts=[ThinkingPart(_THINKING), TextPart("Третий ответ.")]),
    ]
    raw = ModelMessagesTypeAdapter.dump_python(old, mode="json")
    loaded = load_history(raw)
    assert not _has_thinking(loaded) and _alternates(loaded)
    assert all(message.parts for message in loaded)
    assert [p.tool_call_id for m in loaded for p in m.parts if isinstance(p, ToolCallPart)] == ["c1"]
    assert [p.tool_call_id for m in loaded for p in m.parts if isinstance(p, ToolReturnPart)] == ["c1"]
    assert sum(isinstance(m, ModelResponse) for m in loaded) == 3
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8181)
        script = _TurnScript([([], "Те же 100.0% (2 из 2).")], as_text={0})
        turn = await _turn(db_factory, session, script, scope=scope, history=loaded)
    assert turn.state == "prose" and not script.thinking_seen


def _strictly_alternates(messages) -> bool:
    """Запрос, ответ, запрос, … — двух запросов подряд нет: внутри такой истории
    библиотеке склеивать нечего, и её длина — та, с которой начинается ход."""
    return all(
        isinstance(m, ModelRequest if i % 2 == 0 else ModelResponse) for i, m in enumerate(messages)
    )


async def test_dropped_thinking_only_answers_do_not_shift_the_start_of_the_turn(db_factory):
    """Два ответа из одного рассуждения посреди строки и один в конце: ходы, оставшиеся
    без ответа, ушли целиком (иначе библиотека склеила бы их реплики со следующими) —
    длина истории та же, что у библиотеки, начало хода не съезжает, и отвергнутый
    черновик хода не сохраняется."""
    old = [
        ModelRequest(parts=[UserPromptPart("первый вопрос")]),
        ModelResponse(parts=[ToolCallPart("hero_frequency", {"stat": "vpip"}, tool_call_id="c1")]),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "hero_frequency",
                    "выход-синт",
                    tool_call_id="c1",
                    metadata={"calc": "hero_frequency", "allowed": [100.0, 2.0]},
                )
            ]
        ),
        ModelResponse(parts=[ThinkingPart(_THINKING), TextPart("Первый ответ.")]),
        ModelRequest(parts=[UserPromptPart("второй вопрос")]),
        ModelResponse(parts=[ThinkingPart(_THINKING)]),
        ModelRequest(parts=[UserPromptPart("третий вопрос")]),
        ModelResponse(parts=[ThinkingPart(_THINKING)]),
        ModelRequest(parts=[UserPromptPart("четвёртый вопрос")]),
        ModelResponse(parts=[ThinkingPart(_THINKING), TextPart("Четвёртый ответ.")]),
        ModelRequest(parts=[UserPromptPart("пятый вопрос")]),
        ModelResponse(parts=[ThinkingPart(_THINKING)]),
    ]
    loaded = load_history(ModelMessagesTypeAdapter.dump_python(old, mode="json"))
    assert not _has_thinking(loaded) and _strictly_alternates(loaded) and len(loaded) == 6
    assert "пятый вопрос" not in _stored_text(loaded)
    bad = "Вы входите в 43.0% раздач."
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8182)
        script = _TurnScript([([], bad), ([], "Те же 100.0% (2 из 2).")], as_text={0, 1})
        turn = await _turn(db_factory, session, script, scope=scope, history=loaded)
        assert turn.state == "prose" and turn.history_len == len(loaded)
        start = turn.messages[turn.history_len]
        assert isinstance(start, ModelRequest) and _stored_text([start]) == "вопрос"
        saved = turn.saved_messages(None)
    stored = _stored_text(saved)
    assert "43.0" not in stored and REWRITE_HEAD not in stored and "Те же 100.0%" in stored
    assert _strictly_alternates(saved)


async def test_a_turn_ending_in_a_thinking_only_answer_leaves_no_orphan_tool_return(db_factory):
    """Ход с вызовом расчёта, кончившийся ответом из одного рассуждения, игроку ничего
    не дал и уходит целиком: выход его расчёта не склеивается с репликой следующего
    хода, и ни одна обрезка не оставляет выход расчёта в начале истории."""
    old = [
        ModelRequest(parts=[UserPromptPart("первый вопрос")]),
        ModelResponse(parts=[ToolCallPart("hero_frequency", {"stat": "vpip"}, tool_call_id="c1")]),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "hero_frequency",
                    "выход-синт",
                    tool_call_id="c1",
                    metadata={"calc": "hero_frequency", "allowed": [100.0, 2.0]},
                )
            ]
        ),
        ModelResponse(parts=[ThinkingPart(_THINKING)]),
        ModelRequest(parts=[UserPromptPart("второй вопрос")]),
        ModelResponse(parts=[TextPart("Второй ответ.")]),
    ]
    loaded = load_history(ModelMessagesTypeAdapter.dump_python(old, mode="json"))
    assert _strictly_alternates(loaded)
    assert not any(isinstance(p, ToolReturnPart | ToolCallPart) for m in loaded for p in m.parts)
    assert [len(t) for t in turns(loaded)] == [2]
    for budget in range(0, 2000, 25):
        trimmed = trim_history(loaded, budget)
        if trimmed:
            head = trimmed[0]
            assert isinstance(head, ModelRequest)
            assert not any(isinstance(p, ToolReturnPart) for p in head.parts)
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8183)
        script = _TurnScript([([], "Те же 100.0% (2 из 2).")], as_text={0})
        turn = await _turn(db_factory, session, script, scope=scope, history=loaded)
    # Расчёт ушёл вместе с ходом: опоры нет, ход — отказ, но начало хода не съезжает.
    assert turn.state == "refusal" and turn.history_len == len(loaded)
    assert _stored_text([turn.messages[turn.history_len]]) == "вопрос"


def _output_turn(question: str, answer: str, call_id: str) -> list[ModelMessage]:
    """Ход, отвеченный вызовом вывода, в форме прежнего кода: выход вывода —
    отдельным запросом после вызова."""
    return [
        ModelRequest(parts=[UserPromptPart(question)]),
        ModelResponse(parts=[ToolCallPart("final_result", {"answer": answer}, tool_call_id=call_id)]),
        ModelRequest(parts=[ToolReturnPart("final_result", "Final result processed.",
                                           tool_call_id=call_id)]),
    ]


async def test_after_an_output_answer_a_dropped_thinking_only_turn_keeps_the_output_return(
    db_factory,
):
    """Ход ответил вызовом вывода, следующий кончился одним рассуждением и уходит
    целиком; выход вывода, склеенный с его репликой, остаётся — без него вызов
    вывода висел бы без ответа, и библиотека отвергала бы каждый следующий ход."""
    first = _output_turn("первый вопрос", "Первый ответ.", "f1")
    output_return = first[2]
    assert isinstance(output_return, ModelRequest)
    row = [
        *first[:2],
        ModelRequest(parts=[*output_return.parts, UserPromptPart("второй вопрос")]),
        ModelResponse(parts=[ThinkingPart(_THINKING)]),
    ]
    loaded = load_history(ModelMessagesTypeAdapter.dump_python(row, mode="json"))
    assert "второй вопрос" not in _stored_text(loaded)
    assert [p.tool_call_id for m in loaded for p in m.parts if isinstance(p, ToolReturnPart)] == ["f1"]
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8190)
        script = _TurnScript([([_VPIP], _GOOD)])
        turn = await _turn(db_factory, session, script, scope=scope, history=loaded)
    assert turn.state == "prose" and turn.history_len == len(loaded)
    saved = load_history(dump_history(turn.saved_messages(None)))
    assert _strictly_alternates(saved) and _GOOD in _stored_text(saved)


async def test_an_old_row_with_separate_output_returns_loads_merged(db_factory):
    """Строка прежней формы: выход вывода — отдельным запросом перед репликой
    следующего хода. Загружается склеенной, как склеит библиотека, и следующий ход
    начинается там, где кончается загруженная история."""
    row = [
        *_output_turn("первый вопрос", "Первый ответ.", "f1"),
        *_output_turn("второй вопрос", "Второй ответ.", "f2"),
    ]
    loaded = load_history(ModelMessagesTypeAdapter.dump_python(row, mode="json"))
    assert _strictly_alternates(loaded[:-1]) and len(loaded) == 5
    assert len(turns(loaded)) == 2
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8191)
        script = _TurnScript([([_VPIP], _GOOD)])
        turn = await _turn(db_factory, session, script, scope=scope, history=loaded)
    assert turn.state == "prose" and turn.history_len == len(loaded)
    assert _stored_text([turn.messages[turn.history_len]]) == "вопрос"
    saved = load_history(dump_history(turn.saved_messages(None)))
    assert len(turns(saved)) == 3 and _GOOD in _stored_text(saved)


async def test_the_start_of_the_turn_holds_from_the_third_turn_with_output_answers(db_factory):
    """Ответ вызовом вывода оставляет в конце хода запрос с его выходом; в истории
    из двух таких ходов он стоит перед репликой следующего, и библиотека склеила
    бы их — история стала бы короче, и начало хода съехало бы. Сохранённая история
    уже склеена, перед попыткой переписать — тоже, поэтому начало хода не съезжает ни
    на третьем ходе, ни при попытке переписать, отвергнутый черновик не
    сохраняется, а обрезка не оставляет выход вывода в начале истории."""
    bad = "Вы входите в 43.0% раздач."
    history: list[ModelMessage] = []
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8189)
        for index in range(3):
            script = _TurnScript([([_VPIP], _GOOD)])
            if index == 2:
                script = _TurnScript([([_VPIP], bad), ([], _GOOD)])
            turn = await _turn(
                db_factory, session, script, scope=scope, history=history,
                text=f"вопрос {index + 1}",
            )
            assert turn.state == "prose"
            assert _stored_text([turn.messages[turn.history_len]]).endswith(f"вопрос {index + 1}")
            history = load_history(dump_history(turn.saved_messages(None)))
            assert _strictly_alternates(history) and len(turns(history)) == index + 1
    stored = _stored_text(history)
    assert "43.0" not in stored and REWRITE_HEAD not in stored
    for budget in range(0, 3000, 50):
        trimmed = trim_history(history, budget)
        if trimmed:
            assert not any(isinstance(p, ToolReturnPart) for p in trimmed[0].parts)


# --- граница руки в истории вечера (решение владельца 2026-10-15) ---------------------


def _marks(message: ModelMessage) -> list[str]:
    """Тексты отметок-границ в реплике: части `TextContent`, не строки игрока."""
    return [
        item.content
        for part in message.parts
        if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
        for item in part.content
        if isinstance(item, TextContent)
    ]


async def _evening(db_factory, session, scope, hands, history=()):
    """Ходы вечера подряд, история — через сохранение и загрузку, как у воркера.
    Возвращает отметки в реплике, с которой модель начала каждый ход, и историю."""
    history = list(history)
    marks: list[list[str]] = []
    for index, hand in enumerate(hands):
        script = _TurnScript([([_VPIP], _GOOD)])
        turn = await _turn(
            db_factory, session, script, scope=scope, history=history, analysis_id=hand,
            text=f"вопрос {index + 1}",
        )
        assert turn.state == "prose"
        question = script.seen[0][-1]
        assert _stored_text([question]).endswith(f"вопрос {index + 1}")
        marks.append(_marks(question))
        history = load_history(dump_history(turn.saved_messages(None)))
    return marks, history


async def test_a_new_hand_gets_a_boundary_before_the_question_and_the_same_hand_does_not(
    db_factory,
):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8184)
        marks, history = await _evening(db_factory, session, scope, [1, 1, 2, 2])
    assert marks == [[], [], [NEW_HAND_MARK], []]
    # Отметка — перед вопросом, в той же реплике, и в сохранённой истории она остаётся.
    third = turns(history)[2][0]
    assert isinstance(third, ModelRequest)
    [content] = [p.content for p in third.parts if isinstance(p, UserPromptPart)]
    assert isinstance(content, list) and content[1:] == ["вопрос 3"]
    assert _marks(third) == [NEW_HAND_MARK]
    assert len(turns(history)) == 4


async def test_a_turn_without_a_hand_and_back_gets_boundaries(db_factory):
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8185)
        marks, _ = await _evening(db_factory, session, scope, [None, 1, None, None, 1])
    assert marks == [[], [NEW_HAND_MARK], [NO_HAND_MARK], [], [NEW_HAND_MARK]]


def test_the_boundary_has_no_numbers_and_is_not_the_players_text():
    assert not re.search(r"\d", NEW_HAND_MARK) and not re.search(r"\d", NO_HAND_MARK)
    request = ModelRequest(
        parts=[UserPromptPart([TextContent("— 77 —"), "до 12"])]
    )
    assert player_numbers([request]) == frozenset({12.0})


async def test_trimming_keeps_the_boundary_with_its_turn(db_factory):
    """Отметка — часть реплики игрока: не ход и не висит в начале истории без
    вопроса; обрезка, срезавшая руку A, оставляет её при вопросе руки B, и
    следующий ход о руке B отметки не получает."""
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8186)
        _, history = await _evening(db_factory, session, scope, [1, 2])
        assert len(turns(history)) == 2
        last_turn = turns(history)[1]
        trimmed = trim_history(history, estimate_tokens(last_turn))
        assert trimmed[1:] == last_turn[1:] and _marks(trimmed[0]) == [NEW_HAND_MARK]
        assert _stored_text(trimmed[:1]).endswith("вопрос 2")
        marks, _ = await _evening(db_factory, session, scope, [2], history=trimmed)
    assert marks == [[]]


async def test_a_rewrite_on_a_new_hand_keeps_the_boundary_and_drops_the_request(db_factory):
    bad = "Вы входите в 43.0% раздач."
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8187)
        _, history = await _evening(db_factory, session, scope, [1])
        script = _TurnScript([([], bad), ([], "Те же 100.0% (2 из 2).")], as_text={0, 1})
        turn = await _turn(
            db_factory, session, script, scope=scope, history=history, analysis_id=2
        )
    assert turn.state == "prose" and turn.rewrite_at is not None
    assert _stored_text([turn.messages[turn.history_len]]).endswith("вопрос")
    assert _stored_text([turn.messages[turn.rewrite_at]]).startswith(REWRITE_HEAD)
    # Второй прогон видит отметку там же, перед вопросом, и не получает второй.
    marked = [m for m in script.seen[-1] if _marks(m)]
    assert len(marked) == 1 and _stored_text(marked).endswith("вопрос")
    assert _marks(turn.messages[turn.history_len]) == [NEW_HAND_MARK]
    saved = load_history(dump_history(turn.saved_messages(None)))
    assert REWRITE_HEAD not in _stored_text(saved) and "43.0" not in _stored_text(saved)
    assert [_marks(t[0]) for t in turns(saved)] == [[], [NEW_HAND_MARK]]


async def test_an_old_row_without_hand_tags_loads_and_gets_no_boundary(db_factory):
    """Строка, сохранённая до отметок: о какой руке был ход, неизвестно — отметки нет;
    сохранённый ход уже несёт свою руку, и следующая смена руки отметку даёт."""
    old = [
        ModelRequest(parts=[UserPromptPart("прошлый вопрос")]),
        ModelResponse(parts=[TextPart("Прошлый ответ.")]),
    ]
    loaded = load_history(ModelMessagesTypeAdapter.dump_python(old, mode="json"))
    async with db_factory() as session:
        scope = await _seeded(session, tg_user_id=8188)
        marks, _ = await _evening(db_factory, session, scope, [3, 3, None, 4], history=loaded)
    assert marks == [[], [], [NO_HAND_MARK], [NEW_HAND_MARK]]


def test_the_summary_numbers_reach_the_check_of_the_equity_share():
    """Шанс дро из сводки, совпавший с числом `equity_vs_range` из разговора, метки не
    требует; доля только из выхода эквити — требует
    (`test_a_share_of_the_summary_that_equals_an_equity_number_needs_no_mark`)."""
    from harness.agent.run import _draft_violations
    from harness.contracts import POSTFLOP_LINE_DETAIL, Draw, DrawKind, PostflopLineDetail
    from tests.test_agent_checks import _TOP_PAIR, _with_line

    line = PostflopLineDetail.model_validate(_with_line(_TOP_PAIR)[POSTFLOP_LINE_DETAIL])
    draw = Draw(
        kinds=[DrawKind.OPEN_ENDED], out_ranks=["Q", "7"], outs=["Qc"] * 8, unseen=46,
        hit_next=0.174, hit_by_river=None,
    )
    detail: dict[str, object] = {
        POSTFLOP_LINE_DETAIL: line.model_copy(update={"draw": draw}).model_dump(mode="json")
    }
    context = agent_context(
        _result([_point(dp_index=0, ev_diff_bb=-1.4, detail=detail)]), _postflop_hand()
    )
    assert 17.4 in context.allowed
    messages: list[ModelMessage] = [ModelRequest(parts=[ToolReturnPart(
        "equity_vs_range", "эквити",
        metadata={"calc": "equity_vs_range", "allowed": [17.4, 30.1], "assumed": [17.4, 30.1]},
    )])]
    said: frozenset[float] = frozenset()
    summary = AgentDraft(answer="Шанс собрать стрит на следующей улице 17.4%.")
    assert _draft_violations(summary, messages, context, said)[0] == []
    equity = AgentDraft(answer="Против его колла у вас 30.1%.")
    assert any("эквити" in v for v in _draft_violations(equity, messages, context, said)[0])


# --- план 2026-10-17, пп. 6 и 9: сужение по улицам и классы по просьбе игрока ----------


def _narrowing(street: str, classes: list[str], *, analysis_id: int = 1, position: str = "BB"):
    return RangeNarrowing(
        analysis_id=analysis_id, opponent_position=position, street=Street(street),
        classes=classes,
    )


def test_the_prior_is_the_last_narrowing_of_the_previous_street():
    """Ключ — разбор, позиция соперника, улица; берётся последнее сужение строго
    предыдущей улицы: у ривера при одном флопе — `None`, флоп другой руки или другого
    соперника не засчитывается."""
    run_state = CalcRun(max_calcs=8, sizes_from=frozenset(), narrowings=[
        _narrowing("flop", ["AA", "KK"]),
        _narrowing("flop", ["AA"]),
        _narrowing("flop", ["QQ"], analysis_id=2),
        _narrowing("flop", ["JJ"], position="SB"),
    ])
    assert run_state.prior_for(1, "BB", Street.TURN) == _narrowing("flop", ["AA"])
    assert run_state.prior_for(1, "BB", Street.RIVER) is None
    assert run_state.prior_for(1, "BB", Street.FLOP) is None
    assert run_state.prior_for(1, "BB", Street.PREFLOP) is None
    assert run_state.prior_for(3, "BB", Street.TURN) is None
    assert run_state.prior_for(2, "BB", Street.TURN) == _narrowing("flop", ["QQ"], analysis_id=2)
    assert run_state.prior_for(1, "SB", Street.TURN) == _narrowing("flop", ["JJ"], position="SB")
    assert run_state.prior_for(1, "CO", Street.TURN) is None
    run_state.narrowings.append(_narrowing("turn", ["AA"]))
    assert run_state.prior_for(1, "BB", Street.RIVER) == _narrowing("turn", ["AA"])


_THROUGH_TURN_ACTIONS = [
    *BTN_OPENS_BB_CALLS, _check(T, "P6"), _bet(T, "Hero", 400), _call(T, "P6", 400),
]


def _equity(street: str, hands: str, **extra: str) -> tuple[str, dict[str, Any]]:
    return ("equity_vs_range", {"opponent_position": "BB", "street": street, "hands": hands,
                                **extra})


async def _evening_through_the_turn(session, tg_user_id: int):
    scope = await _seeded(session, tg_user_id=tg_user_id)
    analysis_id = await stored_analysis(
        session, session_id=scope[1], actions=_THROUGH_TURN_ACTIONS
    )
    await SessionsRepo(session).set_last_analysis(scope[1], analysis_id)
    await session.commit()
    return scope, analysis_id


async def test_a_class_removed_on_the_flop_of_the_same_turn_is_refused_on_the_turn(
    db_factory, tmp_path, monkeypatch
):
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    monkeypatch.setattr(calcs_module, "AGENT_EQUITY_ITERATIONS", 500)
    async with db_factory() as session:
        scope, analysis_id = await _evening_through_the_turn(session, tg_user_id=8401)
        script = _TurnScript([([
            _equity("flop", "QQ, 77, A5s"),
            _equity("flop", "QQ, 77"),
            _equity("turn", "QQ, 77, A5s"),
            _equity("turn", "QQ"),
        ], "Ответ без чисел (допущение).")])
        turn = await _turn(db_factory, session, script, scope=scope, analysis_id=analysis_id)
    assert [c.get("error") for c in turn.calls] == [
        None, None,
        "на флопе эти классы вы уже убрали: A5s — сужение тёрна берётся из сужения флопа",
        None,
    ]
    assert len(turn.results) == 3
    assert any("Расчёт не выполнен: на флопе эти классы" in text for text in script.returns)


async def test_the_flop_and_the_turn_narrowed_in_one_answer_chain(
    db_factory, tmp_path, monkeypatch
):
    """Оба вызова в одном ответе модели: эквити — барьер, тёрн видит сужение флопа."""
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    monkeypatch.setattr(calcs_module, "AGENT_EQUITY_ITERATIONS", 500)

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[
                ToolCallPart(tool_name=name, args=args)
                for name, args in (_equity("flop", "QQ, 77"), _equity("turn", "QQ, A5s"))
            ])
        return ModelResponse(parts=[
            ToolCallPart(info.output_tools[0].name, {"answer": "Ответ без чисел (допущение)."})
        ])

    async with db_factory() as session:
        scope, analysis_id = await _evening_through_the_turn(session, tg_user_id=8402)
        player_id, session_id, trace_id = scope
        llm = LLM(_TEST_CFG, db_factory, model_override=FunctionModel(model))
        turn = await run_agent_turn(
            session, llm, player_id=player_id, session_id=session_id, text="вопрос",
            trace_id=trace_id, history=(), context=None, system_prompt=_SYSTEM,
            analysis_id=analysis_id,
        )
    assert turn.calls[1]["error"] == (
        "на флопе эти классы вы уже убрали: A5s — сужение тёрна берётся из сужения флопа"
    )


@pytest.mark.parametrize(
    ("flop", "street", "error"),
    [
        (None, "turn", (
            "сначала сузьте флоп: вызовите equity_vs_range для флопа этого соперника, тёрн "
            "сужается из него"
        )),
        (_equity("flop", "QQ"), "river", (
            "сначала сузьте тёрн: вызовите equity_vs_range для тёрна этого соперника, ривер "
            "сужается из него"
        )),
    ],
)
async def test_a_later_street_without_the_previous_narrowing_of_this_opponent_is_refused(
    db_factory, tmp_path, monkeypatch, flop, street: str, error: str
):
    """Тёрн без сужения флопа этого соперника и ривер без сужения тёрна (при сужении
    флопа в том же ходе) — отказ модели; ключ соперника и руки — в
    `test_the_prior_is_the_last_narrowing_of_the_previous_street`."""
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    monkeypatch.setattr(calcs_module, "AGENT_EQUITY_ITERATIONS", 500)
    async with db_factory() as session:
        scope, analysis_id = await _evening_through_the_turn(session, tg_user_id=8404)
        calls = [*([flop] if flop else []), _equity(street, "QQ")]
        turn = await _turn(
            db_factory, session, _TurnScript([(calls, "Ответ без чисел (допущение).")]),
            scope=scope, analysis_id=analysis_id,
        )
    assert turn.calls[-1]["error"] == error


async def test_a_turn_in_the_next_turn_is_bound_by_the_flop_of_the_history(
    db_factory, tmp_path, monkeypatch
):
    """Сужение флопа прошлого хода лежит в метаданных истории и ограничивает тёрн."""
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    monkeypatch.setattr(calcs_module, "AGENT_EQUITY_ITERATIONS", 500)
    async with db_factory() as session:
        scope, analysis_id = await _evening_through_the_turn(session, tg_user_id=8403)
        first = await _turn(
            db_factory, session,
            _TurnScript([([_equity("flop", "QQ, 77")], "Ответ без чисел (допущение).")]),
            scope=scope, analysis_id=analysis_id,
        )
        saved = load_history(dump_history(first.saved_messages(None)))
        assert narrowings_in(saved) == [
            RangeNarrowing(analysis_id=analysis_id, opponent_position="BB",
                           street=Street.FLOP, classes=["QQ", "77"]),
        ]
        second = await _turn(
            db_factory, session,
            _TurnScript([([_equity("turn", "QQ, AQs"), _equity("turn", "QQ")],
                          "Ответ без чисел (допущение).")]),
            scope=scope, history=saved, analysis_id=analysis_id,
        )
    assert "на флопе эти классы вы уже убрали: AQs" in second.calls[0]["error"]
    assert "error" not in second.calls[1]


async def test_a_class_the_player_named_is_added_and_said_in_its_own_line(
    db_factory, tmp_path, monkeypatch
):
    """Игрок назвал «22–77» — агент добавляет их параметром `added`: зона по-прежнему
    assuming, в выходе отдельная строка, доля добавленных — полные комбо."""
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path, call="88+, AQs"))
    monkeypatch.setattr(calcs_module, "AGENT_EQUITY_ITERATIONS", 500)
    async with db_factory() as session:
        scope, analysis_id = await _evening_through_the_turn(session, tg_user_id=8404)
        script = _TurnScript([([_equity("flop", "QQ, AQs", added="22-77")],
                               "Ответ без чисел (допущение).")])
        turn = await _turn(
            db_factory, session, script, scope=scope, analysis_id=analysis_id,
            text="а если у него бывают пары 22–77?",
        )
    assert [c["zone"] for c in turn.calls] == ["assuming"]
    (result,) = turn.results
    assert isinstance(result, EquityVsRangeResult)
    assert result.added == ["77", "66", "55", "44", "33", "22"]
    assert ("Добавлено по вашей просьбе: пара семёрок, пара шестёрок, пара пятёрок, пара "
            "четвёрок, пара троек, пара двоек — по чарту их здесь нет.") in turn.calls[0]["output"]
    assert turn.calls[0]["params"]["player_classes"] == sorted(["22", "33", "44", "55", "66", "77"])
    assert "стартового" not in turn.calls[0]["output"]
    assert assumed_numbers_in(turn.messages) == frozenset({round(100.0 * result.equity, 1)})


async def test_classes_the_agent_adds_with_a_reason_pass_and_carry_to_the_turn(
    db_factory, tmp_path, monkeypatch
):
    """Классы вне чарта с причиной модели (`added_reason`) проходят без записи игрока;
    сужение флопа помнит их причину, и тёрн, вызванный в том же ответе, печатает их
    строкой агента (спека 2026-10-24, §7)."""
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path, call="88+, AQs"))
    monkeypatch.setattr(calcs_module, "AGENT_EQUITY_ITERATIONS", 500)
    reason = "на этих лимитах BB так коллирует"
    async with db_factory() as session:
        scope, analysis_id = await _evening_through_the_turn(session, tg_user_id=8417)
        script = _TurnScript([(
            [_equity("flop", "QQ, AQs", added="22-33", added_reason=reason),
             _equity("turn", "QQ, 22, 33")],
            "Ответ без чисел (допущение).",
        )])
        turn = await _turn(
            db_factory, session, script, scope=scope, analysis_id=analysis_id,
            text="что у него на флопе?",
        )
    assert ["error" in call for call in turn.calls] == [False, False]
    line = f"Добавлено агентом (допущение): пара троек, пара двоек — {reason}."
    assert line in turn.calls[0]["output"] and line in turn.calls[1]["output"]
    assert "по вашей просьбе" not in turn.calls[1]["output"]
    assert narrowings_in(turn.messages)[0].agent_added == {"33": reason, "22": reason}


async def test_an_agent_class_passed_again_in_added_without_a_reason_stays_the_agents(
    db_factory, tmp_path, monkeypatch
):
    """Класс агента из последнего сужения, переданный снова в `added` без причины
    (`kept_agent`), остаётся «добавлено агентом» с прежней причиной, а не «по вашей
    просьбе»."""
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path, call="88+, AQs"))
    monkeypatch.setattr(calcs_module, "AGENT_EQUITY_ITERATIONS", 500)
    reason = "на этих лимитах BB так коллирует"
    async with db_factory() as session:
        scope, analysis_id = await _evening_through_the_turn(session, tg_user_id=8419)
        script = _TurnScript([(
            [_equity("flop", "QQ, AQs", added="22", added_reason=reason),
             _equity("flop", "QQ", added="22")],
            "Ответ без чисел (допущение).",
        )])
        turn = await _turn(
            db_factory, session, script, scope=scope, analysis_id=analysis_id,
            text="что у него на флопе?",
        )
    assert ["error" in call for call in turn.calls] == [False, False]
    second = turn.calls[1]["output"]
    assert f"Добавлено агентом (допущение): пара двоек — {reason}." in second
    assert "по вашей просьбе" not in second
    assert turn.calls[1]["params"]["kept_agent"] == {"22": reason}


async def test_the_category_summary_combos_are_not_assumed_numbers(
    db_factory, tmp_path, monkeypatch
):
    """Комбо сводки по категориям — числа выхода (`allowed`), но не `assumed` и не
    измеренная частота: метки они не требуют."""
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    monkeypatch.setattr(calcs_module, "AGENT_EQUITY_ITERATIONS", 500)
    async with db_factory() as session:
        scope, analysis_id = await _evening_through_the_turn(session, tg_user_id=8418)
        script = _TurnScript([([_equity("flop", "оверпара, AQs")], "Ответ без чисел (допущение).")])
        turn = await _turn(db_factory, session, script, scope=scope, analysis_id=analysis_id)
    (result,) = turn.results
    assert isinstance(result, EquityVsRangeResult)
    assert result.kept_combos is not None and result.source_combos is not None
    counts = {float(round(result.kept_combos)), float(round(result.source_combos))}
    assert counts <= allowed_from_history(turn.messages)
    assert assumed_numbers_in(turn.messages) == frozenset({round(100.0 * result.equity, 1)})
    assert frequency_numbers(turn.messages) == frozenset()


async def test_classes_the_player_did_not_name_are_not_added(db_factory, tmp_path, monkeypatch):
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path, call="88+, AQs"))
    async with db_factory() as session:
        scope, analysis_id = await _evening_through_the_turn(session, tg_user_id=8405)
        script = _TurnScript([([_equity("flop", "QQ", added="22-77")],
                               "Ответ без чисел (допущение).")])
        turn = await _turn(
            db_factory, session, script, scope=scope, analysis_id=analysis_id,
            text="а если у него бывают мелкие пары?",
        )
    assert turn.results == ()
    assert "игрок не назвал эти классы записью" in turn.calls[0]["error"]


async def test_classes_named_earlier_in_the_same_hand_are_added_and_after_a_new_hand_not(
    db_factory, tmp_path, monkeypatch
):
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path, call="88+, AQs"))
    monkeypatch.setattr(calcs_module, "AGENT_EQUITY_ITERATIONS", 500)
    answer = "Ответ без чисел (допущение)."
    async with db_factory() as session:
        scope, analysis_id = await _evening_through_the_turn(session, tg_user_id=8406)
        said = await _turn(
            db_factory, session, _TurnScript([([], "Понял.")]), scope=scope,
            analysis_id=analysis_id, text="допустим, у него бывают 22-55",
        )
        history = load_history(dump_history(said.saved_messages(None)))
        same = await _turn(
            db_factory, session, _TurnScript([([_equity("flop", "QQ", added="22-44")], answer)]),
            scope=scope, history=history, analysis_id=analysis_id, text="посчитай",
        )
        assert "error" not in same.calls[0]
        history = load_history(dump_history(same.saved_messages(None)))
        other_id = await stored_analysis(
            session, session_id=scope[1], actions=_THROUGH_TURN_ACTIONS
        )
        between = await _turn(
            db_factory, session, _TurnScript([([], "Понял.")]), scope=scope,
            history=history, analysis_id=other_id, text="другая рука",
        )
        history = load_history(dump_history(between.saved_messages(None)))
        back = await _turn(
            db_factory, session, _TurnScript([([
                _equity("flop", "QQ", added="22-44"), _equity("flop", "QQ", added="55"),
            ], answer)]),
            scope=scope, history=history, analysis_id=analysis_id, text="посчитай",
        )
    # После возврата к руке: добавленные в её последнем сужении — без новой просьбы;
    # названные до смены руки, но не добавленные, — уже нет.
    assert "error" not in back.calls[0]
    assert "игрок не назвал эти классы записью: 55" in back.calls[1]["error"]


@pytest.mark.parametrize(("analysis_id", "position"), [(2, "BB"), (1, "SB")])
def test_an_added_class_of_one_hand_and_opponent_opens_no_other(analysis_id: int, position: str):
    """Добавленное в руке 1 против BB не открывает `added` ни в руке 2 против BB, ни
    против SB в руке 1: ключ `last_for` — и рука, и позиция соперника."""
    run_state = CalcRun(max_calcs=8, sizes_from=frozenset(), narrowings=[
        _narrowing("flop", ["AA", "22"]),
    ])
    same = run_state.last_for(1, "BB")
    assert same is not None
    assert narrow(parse_range("88+"), "AA", added="22", kept=same.classes).weight("22") == 1.0
    other = run_state.last_for(analysis_id, position)
    with pytest.raises(RangeRefused, match="игрок не назвал эти классы записью: 22"):
        narrow(parse_range("88+"), "AA", added="22", kept=other.classes if other else [])


def test_a_narrowing_trimmed_from_the_history_refuses_the_turn():
    """Ход с сужением флопа ушёл обрезкой — его сужения в истории больше нет, и тёрн
    этого соперника получает отказ «сначала сузьте флоп» (решение владельца
    2026-10-23: раньше тёрн сужался от стартового и выходил шире фактического флопа)."""
    narrowed = _narrowing("flop", ["AA"]).model_dump(mode="json")
    old = [
        ModelRequest(parts=[UserPromptPart("старый вопрос")], metadata={HAND_KEY: 1}),
        ModelResponse(parts=[ToolCallPart("equity_vs_range", {}, tool_call_id="c1")]),
        ModelRequest(parts=[ToolReturnPart("equity_vs_range", "выход", tool_call_id="c1",
                                           metadata={"calc": "equity_vs_range",
                                                     "narrowed": narrowed})]),
        ModelResponse(parts=[TextPart("ответ")]),
    ]
    new = [ModelRequest(parts=[UserPromptPart("новый")], metadata={HAND_KEY: 1}),
           ModelResponse(parts=[TextPart("ответ")])]
    assert narrowings_in([*old, *new]) == [_narrowing("flop", ["AA"])]
    trimmed = trim_history([*old, *new], estimate_tokens(new))
    assert len(trimmed) == len(new)
    assert narrowings_in(trimmed) == []
    kept = CalcRun(max_calcs=8, sizes_from=frozenset(), narrowings=narrowings_in([*old, *new]))
    assert narrow(parse_range("AA, KK"), "AA", street=Street.TURN,
                  prior=kept.prior_for(1, "BB", Street.TURN)).weights == {"AA": 1.0}
    after_trim = CalcRun(max_calcs=8, sizes_from=frozenset(), narrowings=narrowings_in(trimmed))
    with pytest.raises(RangeRefused, match="^сначала сузьте флоп"):
        narrow(parse_range("AA, KK"), "AA", street=Street.TURN,
               prior=after_trim.prior_for(1, "BB", Street.TURN))


def test_the_player_texts_of_the_hand_stop_at_the_last_boundary():
    def said(text: str, analysis_id: int | None) -> ModelRequest:
        return ModelRequest(parts=[UserPromptPart(text)], metadata={HAND_KEY: analysis_id})

    history = [said("о первой: 22-77", 1), said("о второй", 2), said("снова первая", 1),
               ModelResponse(parts=[TextPart("ответ")]), said("и ещё A5s", 1)]
    assert hand_player_texts(history, 1) == ["и ещё A5s", "снова первая"]
    assert hand_player_texts(history, 2) == []
    assert hand_player_texts(history, None) == []
    assert hand_player_texts([ModelRequest(parts=[UserPromptPart("без пометки")])], 1) == []
