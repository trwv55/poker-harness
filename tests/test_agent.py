"""Ход агента: инструменты, потолок, проверки, попытка переписать, история.

В сеть не ходим: модель — `FunctionModel`, сценарий по прогонам задаёт тест.
База настоящая — инструменты делают выборку SQL (довод `test_question_routing.py`).
Выбор расчёта здесь не проверяется: это eval маршрутизации (спека №2).
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy import text as sql_text

from harness.agent.history import (
    allowed_from_history,
    calcs_in_history,
    dump_history,
    load_history,
    trim_history,
    turns,
)
from harness.agent.prompt import instructions_for, load_system_prompt
from harness.agent.run import MAX_CALCS_PER_TURN, REWRITE_HEAD, run_agent_turn
from harness.contracts import CalcName
from harness.explanation.agent_checks import agent_context
from harness.platform.llm import LLM
from tests.conftest import requires_agent_prompt
from tests.test_hand_replay import _postflop_hand
from tests.test_question_routing import _TEST_CFG, _seeded
from tests.test_verdict_text import _point, _result

_SYSTEM = "СИСТЕМНЫЙ-СИНТ"


class _TurnScript:
    """Сценарий по прогонам: в каждом — вызовы инструментов, потом ответ.

    Шаг считается от ПОСЛЕДНЕГО сообщения игрока, а не от начала истории:
    иначе прошлые ходы вечера сдвигали бы сценарий (у `_Script` из
    `test_question_routing.py` истории нет, у агента — есть).
    """

    def __init__(
        self, runs: list[tuple[list[tuple[str, dict[str, Any]]], str]], *, chatter: bool = False
    ) -> None:
        """`chatter` — модель пишет текст рядом с вызовом вывода, как живые
        провайдеры («Итак: …»); `FunctionModel` сам этого не делает."""
        self.runs = runs
        self.chatter = chatter
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
        assert info.output_tools
        output = ToolCallPart(info.output_tools[0].name, {"answer": answer})
        if self.chatter:
            return ModelResponse(parts=[TextPart(f"Итак: {answer}"), output])
        return ModelResponse(parts=[output])


_VPIP = ("hero_frequency", {"stat": "vpip"})
_GOOD = "Вы входите в банк в 100.0% раздач (2 из 2)."


async def _turn(
    db_factory, session, script, *, scope, text="вопрос", history=(), context=None, system=_SYSTEM
):
    player_id, session_id, trace_id = scope
    llm = LLM(_TEST_CFG, db_factory, model_override=FunctionModel(script))
    return await run_agent_turn(
        session, llm, player_id=player_id, session_id=session_id, text=text,
        trace_id=trace_id, history=history, context=context, system_prompt=system,
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
    assert "100.0" in call["output"] and call["duration_s"] >= 0


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
    return ModelMessagesTypeAdapter.dump_json(list(messages)).decode()


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
