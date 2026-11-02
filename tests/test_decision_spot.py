"""A text description stops at the hero's decision and cannot become a played hand."""

from typing import Any, cast

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from sqlalchemy import text

from harness.agent.decision_spot import MAX_SPOT_SCENARIOS_PER_TURN, spot_tools
from harness.agent.history import seen_history
from harness.agent.run import _SPOT_ADVICE_RE
from harness.analysis.decision_spot import spot_call_scenario
from harness.contracts.decision_spot import (
    RawDecisionSpot,
    RawSpotAction,
    RawSpotPlayer,
    SpotExtraction,
    SpotFact,
)
from harness.engine.decision_spot import InvalidDecisionSpot, replay_until_decision
from harness.memory.repos import DecisionSpotsRepo, PlayersRepo, SessionsRepo
from harness.normalizer.decision_spot import _position_in_quote, normalize_decision_spot
from harness.parsers.decision_spot import extract_decision_spot, looks_like_decision_spot
from harness.presentation.messages import spot_scenario_msg
from harness.worker.pipeline import _non_spot_history, _only_context, _other_contexts, _spot_history


def fact(value: str, quote: str) -> SpotFact:
    return SpotFact(value=value, quote=quote, status="known")


def sample() -> RawDecisionSpot:
    text = (
        "8 players, blinds 500/1000, ante 100 each. "
        "Hero BB 9h9d stack 30000 before ante. "
        "UTG folds, UTG+1 folds, LJ folds, HJ folds. "
        "CO opens to 2500, stack 50000 before ante. "
        "BTN all-in to 28000, stack 28100 before ante. SB folds. Hero to act."
    )
    return RawDecisionSpot(
        source_text=text,
        table_size=fact("8", "8 players"),
        sb=fact("500", "blinds 500/1000"),
        bb=fact("1000", "blinds 500/1000"),
        ante=fact("100", "ante 100 each"),
        ante_type="per_player",
        hero_position=fact("BB", "Hero BB"),
        hero_cards=fact("9h9d", "Hero BB 9h9d"),
        players=[
            RawSpotPlayer(
                position="BB",
                stack=fact("30000", "stack 30000 before ante"),
                stack_moment="before_ante",
            ),
            RawSpotPlayer(
                position="CO",
                stack=fact("50000", "stack 50000 before ante"),
                stack_moment="before_ante",
            ),
            RawSpotPlayer(
                position="BTN",
                stack=fact("28100", "stack 28100 before ante"),
                stack_moment="before_ante",
            ),
        ],
        actions=[
            RawSpotAction(position=p, kind="fold", quote=f"{p} folds")
            for p in ("UTG", "UTG+1", "LJ", "HJ")
        ]
        + [
            RawSpotAction(
                position="CO",
                kind="raise",
                amount=fact("2500", "CO opens to 2500"),
                amount_kind="total",
                quote="CO opens to 2500",
            ),
            RawSpotAction(
                position="BTN",
                kind="all_in",
                amount=fact("28000", "BTN all-in to 28000"),
                amount_kind="total",
                quote="BTN all-in to 28000",
            ),
            RawSpotAction(position="SB", kind="fold", quote="SB folds"),
        ],
    )


def test_prefix_stops_before_hero_and_keeps_co_to_act() -> None:
    spot, questions = normalize_decision_spot(sample())
    assert not questions and spot is not None
    snapshot = replay_until_decision(spot)
    assert snapshot.pot_before == 32800
    assert snapshot.to_call == 27000
    assert snapshot.positions_still_to_act == ["CO"]
    assert snapshot.committed["BB"] == 1000
    assert len(spot.actions) == 7
    assert spot.hero_to_act
    scenario = spot_call_scenario(7, spot, snapshot, fold_after_hero=("CO",))
    assert scenario.equity_threshold == 27000 / (32800 + 27000)
    assert scenario.equity is None
    assert scenario.condition == "если CO сбросят"


def test_unknown_ante_and_ambiguous_all_in_request_facts() -> None:
    raw = sample()
    raw.ante_type = "unknown"
    assert normalize_decision_spot(raw)[0] is None
    raw = sample()
    raw.actions[5].amount_kind = "unspecified"
    spot, questions = normalize_decision_spot(raw)
    assert spot is None and "доплата" in questions[0]


def test_hallucinated_size_and_missing_action_cannot_replay() -> None:
    raw = sample()
    raw.actions[4].amount.value = "3500"
    assert normalize_decision_spot(raw)[0] is None
    raw = sample()
    raw.actions.pop(0)
    spot, _ = normalize_decision_spot(raw)
    assert spot is not None
    try:
        replay_until_decision(spot)
    except InvalidDecisionSpot as exc:
        assert "UTG" in str(exc)
    else:
        raise AssertionError("missing action must be rejected")


def test_simple_question_does_not_spend_an_extraction_call() -> None:
    assert not looks_like_decision_spot("Каков мой VPIP за неделю?")
    assert looks_like_decision_spot("Я на BB с 99, CO открыл, BTN пушит. Разбери руку.")
    assert looks_like_decision_spot("BB 99, CO открыл, BTN пуш. Что делать?")
    assert looks_like_decision_spot("на большом блайнде 99, CO открыл, BTN пуш. Что делать?")


def test_a_bet_in_bb_is_not_evidence_of_the_bb_position() -> None:
    assert not _position_in_quote("BB", "CO opens to 2.5 bb")
    assert _position_in_quote("BB", "BB calls 2500")


def test_implicit_hero_position_can_be_read_from_a_short_question() -> None:
    for phrase, position_quote in (
        ("BB 9h9d, что делать?", "BB"),
        ("на большом блайнде 9h9d, что делать?", "большом блайнде"),
    ):
        raw = sample()
        raw.source_text = raw.source_text.replace("Hero BB 9h9d", phrase)
        raw.hero_position = fact("BB", position_quote)
        raw.hero_cards = fact("9h9d", "9h9d")
        spot, questions = normalize_decision_spot(raw)
        assert not questions and spot is not None
        assert spot.hero_position == "BB"


@pytest.mark.asyncio
async def test_spot_versions_are_owned_by_the_session_and_separate_from_hands(db) -> None:
    owner = await PlayersRepo(db).get_or_create(tg_user_id=916001)
    stranger = await PlayersRepo(db).get_or_create(tg_user_id=916002)
    session = await SessionsRepo(db).active_or_create(owner.id)
    raw = sample()
    spot, questions = normalize_decision_spot(raw)
    assert not questions and spot is not None
    snapshot = replay_until_decision(spot)
    repo = DecisionSpotsRepo(db)

    draft = await repo.save(
        player_id=owner.id, session_id=session.id, source_text=raw.source_text,
        raw=raw.model_dump(mode="json"), canonical=None, snapshot=None,
    )
    assert draft.version == 1 and draft.status == "draft"
    verified = await repo.save(
        player_id=owner.id, session_id=session.id, source_text=raw.source_text,
        raw=raw.model_dump(mode="json"), canonical=spot.model_dump(mode="json"),
        snapshot=snapshot.model_dump(mode="json"), prior_id=draft.id,
    )
    assert verified.version == 2 and verified.status == "verified"
    assert session.active_spot_id == verified.id
    assert await repo.get(verified.id, stranger.id, session.id) is None
    assert await db.scalar(text("select count(*) from hands")) == 0

    await SessionsRepo(db).set_last_analysis(session.id, None)
    await db.refresh(session)
    assert session.active_spot_id is None


def test_semantics_must_match_cited_words() -> None:
    raw = sample()
    raw.actions[4].kind = "fold"
    assert normalize_decision_spot(raw)[0] is None
    raw = sample()
    raw.ante_type = "big_blind"
    assert normalize_decision_spot(raw)[0] is None
    raw = sample()
    raw.actions[4].amount_kind = "additional"
    assert normalize_decision_spot(raw)[0] is None
    raw = sample()
    raw.players[0].stack_moment = "behind"
    assert normalize_decision_spot(raw)[0] is None


def test_bet_and_blind_numbers_must_match_their_roles_in_the_quote() -> None:
    raw = sample()
    raw.sb = fact("100", "blinds 500/1000, ante 100 each")
    assert normalize_decision_spot(raw)[0] is None

    raw = sample()
    raw.actions[4].amount = fact("50000", "CO opens to 2500, stack 50000 before ante")
    raw.actions[4].quote = "CO opens to 2500, stack 50000 before ante"
    assert normalize_decision_spot(raw)[0] is None


@pytest.mark.asyncio
async def test_long_clarification_reaches_extractor_and_keeps_full_evidence() -> None:
    source = "initial quote " + "x" * 4200 + " latest answer stack 28000 before ante"

    class FakeLLM:
        prompt = ""

        async def __call__(self, purpose, response_type, **kwargs):
            self.prompt = kwargs["prompt"]
            return SpotExtraction(is_spot=True, spot=sample()), None

    llm = FakeLLM()
    raw = await extract_decision_spot(llm, source, 1)  # type: ignore[arg-type]
    assert "latest answer stack 28000 before ante" in llm.prompt
    assert len(llm.prompt) <= 4010
    assert raw is not None and raw.source_text == source


def test_heads_up_fold_ends_hand_before_hero_decision() -> None:
    from harness.contracts.decision_spot import CanonicalDecisionSpot, SpotAction, SpotPlayer

    spot = CanonicalDecisionSpot(
        table_size=2, sb=500, bb=1000, ante=0, ante_type="none",
        hero_position="BB", hero_class="99",
        players=[SpotPlayer(position="BTN"), SpotPlayer(position="BB")],
        actions=[SpotAction(position="BTN", kind="fold", quote="BTN folds")],
    )
    with pytest.raises(InvalidDecisionSpot, match="Все соперники"):
        replay_until_decision(spot)


def test_class_suffix_and_card_suits_are_not_invented() -> None:
    raw = sample()
    raw.hero_cards = fact("AKs", "Hero BB 9h9d")
    assert normalize_decision_spot(raw)[0] is None
    raw.source_text = raw.source_text.replace("9h9d", "AKs")
    raw.hero_cards = fact("AKs", "Hero BB AKs")
    spot, questions = normalize_decision_spot(raw)
    assert not questions and spot is not None and spot.hero_class == "AKs"
    assert spot.hero_cards is None
    raw.hero_cards = fact("AhKh", "Hero BB AKs")
    assert normalize_decision_spot(raw)[0] is None


def test_short_all_in_raise_is_explicitly_refused() -> None:
    raw = sample()
    raw.source_text = raw.source_text.replace("BTN all-in to 28000", "BTN all-in to 3500")
    raw.actions[5].amount = fact("3500", "BTN all-in to 3500")
    raw.actions[5].quote = "BTN all-in to 3500"
    raw.players[2].stack = fact("3600", "stack 3600 before ante")
    raw.source_text = raw.source_text.replace("stack 28100 before ante", "stack 3600 before ante")
    spot, questions = normalize_decision_spot(raw)
    assert not questions and spot is not None
    try:
        replay_until_decision(spot)
    except InvalidDecisionSpot as exc:
        assert "Неполный рейз" in str(exc)
    else:
        raise AssertionError("short all-in must not be treated as a full raise")


def test_missing_folded_stacks_do_not_require_fake_values() -> None:
    spot, _ = normalize_decision_spot(sample())
    assert spot is not None
    assert next(p for p in spot.players if p.position == "SB").stack is None
    snapshot = replay_until_decision(spot)
    assert snapshot.pot_before == 32800
    assert "CO" in snapshot.positions_still_to_act


def test_explicit_no_ante_is_zero_without_inventing_a_number() -> None:
    raw = sample()
    raw.source_text = raw.source_text.replace("ante 100 each", "No ante")
    raw.ante = fact("0", "No ante")
    raw.ante_type = "none"
    raw.source_text = raw.source_text.replace("stack 28100 before ante", "stack 28000 before ante")
    raw.players[2].stack = fact("28000", "stack 28000 before ante")
    spot, questions = normalize_decision_spot(raw)
    assert not questions and spot is not None and spot.ante == 0
    assert replay_until_decision(spot).pot_before == 32000


def test_unknown_folded_stacks_require_full_ante_evidence() -> None:
    raw = sample()
    raw.source_text = raw.source_text.replace("ante 100 each", "ante 100")
    raw.ante.quote = "ante 100"
    spot, questions = normalize_decision_spot(raw)
    assert spot is None and "Анте платил" in questions[0]


def test_spot_tool_survives_saved_history_and_contexts_are_isolated() -> None:
    messages = [
        ModelRequest(parts=[UserPromptPart("если CO сбросит?")]),
        ModelResponse(parts=[ToolCallPart("spot_call_scenario", {}, tool_call_id="c1")]),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "spot_call_scenario",
                    "порог 45%",
                    tool_call_id="c1",
                    metadata={"calc": "spot_call_scenario", "allowed": [45.0]},
                )
            ]
        ),
        ModelResponse(parts=[TextPart("Порог 45% при условии фолда CO.")]),
    ]
    saved = seen_history(messages, start=0, rewrite_at=None, final_answer=None, analysis_id=-7)
    assert any(
        isinstance(part, ToolCallPart) and part.tool_name == "spot_call_scenario"
        for message in saved
        for part in message.parts
    )
    assert any(
        isinstance(part, ToolReturnPart) and part.tool_name == "spot_call_scenario"
        for message in saved
        for part in message.parts
    )
    hh = [
        ModelRequest(parts=[UserPromptPart("вопрос о HH")], metadata={"analysis_id": 12}),
        ModelResponse(parts=[TextPart("ответ о HH")]),
    ]
    combined = [*hh, *saved]
    assert _only_context(combined, -7) == saved
    assert _non_spot_history(combined, 12) == hh
    assert _spot_history(combined) == saved


def test_hh_history_selects_only_current_analysis_and_preserves_other_turns() -> None:
    first = [
        ModelRequest(parts=[UserPromptPart("первая HH")], metadata={"analysis_id": 12}),
        ModelResponse(parts=[TextPart("цифры первой HH")]),
    ]
    second = [
        ModelRequest(parts=[UserPromptPart("вторая HH")], metadata={"analysis_id": 13}),
        ModelResponse(parts=[TextPart("цифры второй HH")]),
    ]
    assert _non_spot_history([*first, *second], 13) == second
    assert _only_context([*first, *second], 13) == second
    assert [*_other_contexts([*first, *second], 13), *second] == [*first, *second]


@pytest.mark.asyncio
async def test_spot_scenario_tool_has_per_turn_budget() -> None:
    spot, questions = normalize_decision_spot(sample())
    assert not questions and spot is not None
    snapshot = replay_until_decision(spot)
    calls: list[dict] = []
    tool = spot_tools(7, spot, snapshot, calls)[0]
    invoke = cast(Any, tool.function)
    for _ in range(MAX_SPOT_SCENARIOS_PER_TURN):
        await invoke(["CO"])
    refused = await invoke(["CO"], opponent_range="99+", range_source="user_range")
    assert "Лимит расчётов" in refused
    assert calls[-1]["refused"]


def test_spot_response_guard_and_code_fallback() -> None:
    assert _SPOT_ADVICE_RE.search("Здесь колл есть")
    assert _SPOT_ADVICE_RE.search("Я бы коллировал")
    assert not _SPOT_ADVICE_RE.search("Порог для колла зависит от диапазона")
    msg = spot_scenario_msg("если CO сбросит", 32800, 27000, 27000 / 59800,
                            None, "Без диапазона решения нет.")
    assert "45%" in msg.text and "Без диапазона решения нет" in msg.text
