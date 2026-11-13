"""Synthetic examples of defaulted text scenarios and prefix replay."""

import json
from decimal import Decimal
from typing import cast

import pytest

from harness.agent.natural_hand import NaturalOpinion, natural_opinion
from harness.analysis.natural_hand import conditional_equity
from harness.contracts.natural_hand import NaturalAction, NaturalHand, NaturalPlayer, NaturalRead
from harness.engine.natural_hand import InvalidScenario, replay_natural_hand
from harness.normalizer.natural_hand import normalize_natural_hand
from harness.parsers.natural_hand import (
    apply_natural_correction,
    may_describe_hand,
    read_natural_hand,
)
from harness.platform.llm import LLM
from harness.presentation.natural_hand import natural_scenario_msg


def test_bb_call_spot_keeps_co_live_and_counts_only_one_small_blind() -> None:
    raw = NaturalHand(
        table_size=8, sb_chips="400", bb_chips="800", ante="100", ante_unit="chips",
        hero_position="BB", hero_cards="9♠9♣",
        players=[
            NaturalPlayer(position="BB", stack_bb="59,49", quote="за спиной 59,49 ББ"),
            NaturalPlayer(position="CO", stack_bb="20,80", stack_moment="after_action",
                          quote="у него осталось 20,80 ББ"),
        ],
        actions=[
            NaturalAction(position="CO", kind="raise", amount="2", quote="CO открыл 2 ББ"),
            NaturalAction(position="BTN", kind="all_in", amount="28,92",
                          quote="BTN пошёл олл-ин 28,92 ББ"),
            NaturalAction(position="SB", kind="fold", quote="SB сфолдил"),
        ],
    )
    spot = normalize_natural_hand(raw)
    snapshot = replay_natural_hand(spot)
    assert snapshot.pot_before_bb == Decimal("33.42")
    assert snapshot.to_call_bb == Decimal("27.92")
    assert snapshot.positions_after_hero == ["CO"]
    assert spot.stacks_behind_bb["CO"].value == Decimal("22.80")
    assert snapshot.remaining_bb["CO"] == Decimal("20.80")
    msg = natural_scenario_msg(spot, snapshot)
    assert "33,42 BB" in msg.text
    assert "27,92 BB" in msg.text
    assert "45,52%" in msg.text
    assert "Если CO сбросят" in msg.text


def test_defaulted_three_bet_reaches_flop_before_sb_decision() -> None:
    raw = NaturalHand(
        hero_position="SB", hero_cards="AKs", flop="K72",
        actions=[
            NaturalAction(position="BTN", kind="raise", quote="BTN открыл"),
            NaturalAction(position="SB", kind="raise", quote="я 3-бет с SB"),
            NaturalAction(position="BTN", kind="call", quote="BTN колл"),
        ],
    )
    spot = normalize_natural_hand(raw)
    snapshot = replay_natural_hand(spot)
    assert spot.actions[0].committed_after == Decimal(2)
    assert spot.actions[1].committed_after == Decimal(6)
    assert snapshot.street == "flop"
    assert snapshot.pot_before_bb == Decimal(14)
    assert snapshot.to_call_bb == 0
    assert snapshot.live_positions == ["SB", "BTN"]
    assert len(set(spot.board + list(spot.hero_cards or ()))) == 5
    assert {card[1] for card in spot.board} == {"s", "h", "d"}
    opinion = "На сухом борде можно поставить для ценности."
    text = natural_scenario_msg(spot, snapshot, opinion=opinion).text
    assert "Банк перед вашим ходом: 14 BB." in text
    assert "Оценка модели: " + opinion in text
    assert "Подставил: " in text
    assert "Неназванные пасы: " in text
    assert "Вы можете чекнуть без доплаты." not in text
    assert "Если решите поставить 33% банка" not in text
    assert "Выбор линии зависит от диапазона соперника." not in text


def test_default_postflop_bet_is_33_percent_of_current_pot() -> None:
    raw = NaturalHand(
        hero_position="SB", hero_cards="AKs", flop="K72",
        actions=[
            NaturalAction(position="BTN", kind="raise", quote="BTN открыл"),
            NaturalAction(position="SB", kind="raise", quote="SB 3-бет"),
            NaturalAction(position="BTN", kind="call", quote="BTN колл"),
            NaturalAction(street="flop", position="SB", kind="check", quote="я чекнул"),
            NaturalAction(street="flop", position="BTN", kind="bet", quote="BTN поставил"),
        ],
    )
    spot = normalize_natural_hand(raw)
    snapshot = replay_natural_hand(spot)
    assert spot.actions[-1].committed_after == Decimal("4.62")
    assert snapshot.pot_before_bb == Decimal("18.62")
    assert snapshot.to_call_bb == Decimal("4.62")
    assert "BTN ставка 33% банка" in spot.assumptions
    text = natural_scenario_msg(spot, snapshot).text
    assert "Для колла нужно добавить 4,62 BB." in text
    assert "для колла нужно не меньше" in text


def test_stack_left_after_flop_bet_includes_both_streets() -> None:
    raw = NaturalHand(
        hero_position="SB", hero_cards="AKs", flop="K72",
        players=[NaturalPlayer(position="BTN", stack_bb="10", stack_moment="after_action",
                               quote="после ставки осталось 10 BB")],
        actions=[
            NaturalAction(position="BTN", kind="raise", quote="BTN открыл"),
            NaturalAction(position="SB", kind="raise", quote="SB 3-бет"),
            NaturalAction(position="BTN", kind="call", quote="BTN колл"),
            NaturalAction(street="flop", position="SB", kind="check", quote="SB чек"),
            NaturalAction(street="flop", position="BTN", kind="bet", quote="BTN поставил"),
        ],
    )
    hand = normalize_natural_hand(raw)
    snapshot = replay_natural_hand(hand)
    assert hand.stacks_behind_bb["BTN"].value == Decimal("20.62")
    assert snapshot.remaining_bb["BTN"] == Decimal(10)


def test_cannot_invent_a_missing_hero_action_to_get_to_flop() -> None:
    raw = NaturalHand(
        hero_position="BB", hero_cards="99", flop="K72",
        actions=[NaturalAction(position="BTN", kind="raise", quote="BTN открыл")],
    )
    with pytest.raises(InvalidScenario, match="не закончены действия"):
        replay_natural_hand(normalize_natural_hand(raw))


def test_short_stack_correction_replaces_only_btn_stack() -> None:
    original = NaturalHand(
        hero_position="SB", hero_cards="AKs", flop="K72",
        actions=[
            NaturalAction(position="BTN", kind="raise", quote="BTN открыл"),
            NaturalAction(position="SB", kind="raise", quote="SB 3-бет"),
            NaturalAction(position="BTN", kind="call", quote="BTN колл"),
        ],
        source_text="BTN открыл, SB 3-бет, BTN колл. Флоп K72.",
    )
    patch = NaturalHand(
        players=[NaturalPlayer(position="BTN", stack_bb="18", quote="у BTN было 18 BB")],
        source_text="у BTN было 18 BB",
    )
    updated = apply_natural_correction(original, patch)
    spot = normalize_natural_hand(updated)
    snapshot = replay_natural_hand(spot)
    assert spot.stacks_behind_bb["BTN"].value == Decimal(18)
    assert spot.stacks_behind_bb["BTN"].source == "player"
    assert spot.stacks_behind_bb["SB"].value == Decimal(40)
    assert snapshot.street == "flop"
    assert "у BTN было 18 BB" in updated.source_text


def test_free_wording_routes_to_natural_extraction() -> None:
    text = (
        "Ты — BB с 9♠9♣, за спиной 59,49 ББ. CO открыл 2 ББ, "
        "BTN пошёл олл-ин 28,92 ББ. Что думаешь, есть колл?"
    )
    assert may_describe_hand(text)
    assert may_describe_hand("У BTN было 18 BB", has_active=True)
    assert not may_describe_hand("Каков мой VPIP за неделю?")
    assert not may_describe_hand("какой диапазон у BB")
    assert not may_describe_hand("какова частота колла у поля?")


@pytest.mark.asyncio
async def test_model_range_is_only_an_assumption_and_equity_is_computed_by_code() -> None:
    hand = normalize_natural_hand(NaturalHand(
        table_size=8, hero_position="BB", hero_cards="9s9c", question="Есть колл?",
        actions=[
            NaturalAction(position="CO", kind="raise", amount="2", quote="CO открыл"),
            NaturalAction(position="BTN", kind="all_in", amount="28.92", quote="BTN пуш"),
            NaturalAction(position="SB", kind="fold", quote="SB пас"),
        ],
    ))
    snapshot = replay_natural_hand(hand)

    class FakeLLM:
        async def __call__(self, *args: object, **kwargs: object):
            return NaturalOpinion(
                explanation="Против узкого диапазона колл выглядит сомнительным.",
                opponent_range="TT+,AQs+,AKo",
            ), None

    opinion = await natural_opinion(cast(LLM, FakeLLM()), hand, snapshot, 1)
    assert opinion is not None and opinion.opponent_range is not None
    equity = conditional_equity(hand, snapshot, opinion.opponent_range)
    assert equity is not None and equity < 0.4552
    message = natural_scenario_msg(
        hand, snapshot, opinion=opinion.explanation,
        opponent_range=opinion.opponent_range, equity=equity,
    )
    assert "предполагаемого диапазона" in message.text
    assert "колл не проходит порог" in message.text


@pytest.mark.asyncio
async def test_model_numeric_claim_is_not_shown() -> None:
    hand = normalize_natural_hand(NaturalHand(
        hero_position="SB", hero_cards="AKs", flop="K72",
        actions=[
            NaturalAction(position="BTN", kind="raise", quote="BTN открыл"),
            NaturalAction(position="SB", kind="raise", quote="SB 3-бет"),
            NaturalAction(position="BTN", kind="call", quote="BTN колл"),
        ],
    ))
    snapshot = replay_natural_hand(hand)

    class FakeLLM:
        async def __call__(self, *args: object, **kwargs: object):
            return NaturalOpinion(explanation="Ставьте 70% банка."), None

    assert await natural_opinion(cast(LLM, FakeLLM()), hand, snapshot, 1) is None


@pytest.mark.asyncio
async def test_long_qualitative_opinion_keeps_first_three_sentences() -> None:
    hand = normalize_natural_hand(NaturalHand(
        hero_position="SB", hero_cards="AKs", flop="K72",
        actions=[
            NaturalAction(position="BTN", kind="raise", quote="BTN открыл"),
            NaturalAction(position="SB", kind="raise", quote="SB трибет"),
            NaturalAction(position="BTN", kind="call", quote="BTN колл"),
        ],
    ))
    snapshot = replay_natural_hand(hand)

    class FakeLLM:
        async def __call__(self, *args: object, **kwargs: object):
            sentence = "Можно начать с чека и оценить ответ соперника. "
            return NaturalOpinion(explanation=sentence * 20), None

    opinion = await natural_opinion(cast(LLM, FakeLLM()), hand, snapshot, 1)
    assert opinion is not None
    assert opinion.explanation.count("соперника") == 3


@pytest.mark.asyncio
async def test_stringified_model_json_recovers_blinds_and_hero_stack() -> None:
    source = (
        "Блайнды 400/800, анте 100, за столом 8 игроков.\n"
        "Ты — BB с 9♠9♣, за спиной 59,49 ББ.\n"
        "CO открыл 2 ББ, у него осталось 20,80 ББ.\n"
        "BTN пошёл олл-ин 28,92 ББ.\nSB поставил 0,5 ББ и сфолдил.\n"
        "Что думаешь, есть колл?"
    )
    extracted = {
        "table_size": 8, "ante": "100", "ante_unit": "chips",
        "hero_position": "BB", "hero_cards": "9s9c", "hero_quote": "Ты — BB с 9♠9♣",
        "players": [
            {"position": "CO", "stack_bb": "20.80", "stack_moment": "after_action",
             "quote": "у него осталось 20,80 ББ"},
            {"position": "BTN"},
        ],
        "actions": [
            {"position": "CO", "kind": "raise", "amount": "2", "quote": "CO открыл 2 ББ"},
            {"position": "BTN", "kind": "all_in", "amount": "28.92",
             "quote": "BTN пошёл олл-ин 28,92 ББ"},
            {"position": "SB", "kind": "fold", "quote": "SB поставил 0,5 ББ и сфолдил"},
        ],
    }
    model_output = NaturalRead.model_validate(
        {"intent": "new_hand", "hand": json.dumps(extracted, ensure_ascii=False)}
    )

    class FakeLLM:
        async def __call__(self, *args: object, **kwargs: object):
            return model_output, None

    read = await read_natural_hand(cast(LLM, FakeLLM()), source, 1)
    assert read.hand is not None
    assert read.hand.bb_chips == "800"
    assert any(p.position == "BB" and p.stack_bb == "59.49" for p in read.hand.players)
    spot = normalize_natural_hand(read.hand)
    snapshot = replay_natural_hand(spot)
    assert snapshot.pot_before_bb == Decimal("33.42")
    assert snapshot.to_call_bb == Decimal("27.92")
