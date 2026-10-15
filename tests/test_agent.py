"""Ход агента: инструменты, потолок, проверки, попытка переписать, история.

В сеть не ходим: модель — `FunctionModel`, сценарий по прогонам задаёт тест.
База настоящая — инструменты делают выборку SQL (довод `test_question_routing.py`).
Выбор расчёта здесь не проверяется: это eval маршрутизации (спека №2).
"""

from __future__ import annotations

from collections.abc import Collection
from itertools import pairwise
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
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
    allowed_from_history,
    assumed_numbers_in,
    calcs_in_history,
    chart_numbers_in,
    dump_history,
    frequency_numbers,
    load_history,
    trim_history,
    turns,
)
from harness.agent.prompt import instructions_for, load_system_prompt
from harness.agent.run import MAX_CALCS_PER_TURN, REWRITE_HEAD, AgentDraft, run_agent_turn
from harness.analysis.charts import lookup
from harness.analysis.frequency import defense_from_bet
from harness.calcs.routing import CalcRun, tools_for
from harness.contracts import (
    AGENT_ONLY_CALCS,
    CalcName,
    DefenseResult,
    EquityVsRangeResult,
    OpponentRangeResult,
)
from harness.explanation.agent_checks import agent_context
from harness.memory.repos import SessionsRepo
from harness.platform.llm import LLM, LLMSchemaError
from harness.presentation.messages import agent_turn_msg
from tests.conftest import requires_agent_prompt
from tests.test_calcs import stored_analysis
from tests.test_hand_replay import _postflop_hand
from tests.test_opponent_range import defend_file
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
    ) -> None:
        """`chatter` — модель пишет текст рядом с вызовом вывода, как живые
        провайдеры («Итак: …»); `FunctionModel` сам этого не делает.
        `as_text` — номера прогонов, в которых модель отвечает простым текстом,
        а не вызовом вывода (живая модель с `tool_choice: auto` делает и так,
        и так); такой ответ идёт с рассуждением рядом."""
        self.runs = runs
        self.chatter = chatter
        self.as_text = frozenset(as_text)
        # Номер прогона → блок «Оценка модели» вторым полем вывода.
        self.assumptions = assumptions or {}
        self.run = -1
        self.requests = 0
        self.instructions: list[str] = []
        self.returns: list[str] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
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
    assumed = frozenset(
        {round(100.0 * equity_result.equity, 1), round(100.0 * equity_result.share_of_start, 1)}
    )
    assert assumed_numbers_in(turn.messages) == assumed


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
