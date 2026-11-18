"""Synthetic table only; real hands and model accuracy live in private evals."""
from copy import deepcopy

import pytest

from harness.contracts.pokerdom_vision import PokerdomEvent, PokerdomPlayer, PokerdomReading
from harness.contracts.raw import Street
from harness.parsers.pokerdom_vision import process_pokerdom
from harness.presentation.pokerdom_vision import pokerdom_report


def table():
    events = [("A", "Ante", 10), ("B", "Ante", 10), ("A", "SB", 100),
              ("B", "BB", 200), ("A", "R", 500), ("B", "Fold", None),
              ("A", "UB", 400)]
    return PokerdomReading(is_hand=True, complete=True, cropped=False, unit="chips",
        stack_semantics="before_forced",
        players=[PokerdomPlayer(row_id="A", nickname="synthetic-alice", stack_shown=5000,
                      button_shown=True, cards=["Ah", "Kd"]),
                 PokerdomPlayer(row_id="B", nickname="synthetic-bob", stack_shown=5000,
                      button_shown=False, cards=[])],
        events=[PokerdomEvent(number=i, player=p, street=Street.PREFLOP, token=t, amount=a,
                     amount_text=str(a) if a is not None else None)
                for i, (p, t, a) in enumerate(events, 1)],
        board={}, pot_after_street={Street.PREFLOP: 420})


def process(reading, expected=None, nickname="synthetic-alice"):
    return process_pokerdom(reading, nickname=nickname, expected=expected)


def test_raise_addition_ante_and_return_once_terminal_pot():
    result = process(table(), table())
    assert result.replay_status == "passed", result
    assert result.truth_status == "matched"
    assert result.raw is not None and result.enriched is not None
    assert result.raw.actions[0].to_amount == 600
    assert result.raw.ante == 10
    assert result.enriched.report.final_pot == 420
    assert result.enriched.report.stacks_end == {"Hero": 5210, "B": 4790}
    assert result.checks[0].status == "passed"
    assert {c.name for c in result.checks if c.status == "not_checked"} == {
        "room_payout", "room_end_stacks"}
    assert not result.raw.collected and result.raw.summary is None


def test_order_is_numbers_not_row_or_array_order():
    reading = table()
    reading.events.reverse()
    result = process(reading, table())
    assert result.replay_status == "passed"
    assert result.truth_status == "matched"


@pytest.mark.parametrize("mutation,fragment", [
    (lambda r: setattr(r.events[2], "number", 2), "numbers"),
    (lambda r: setattr(r.events[2], "number", 20), "numbers"),
    (lambda r: setattr(r.events[2], "player", "absent"), "unknown player"),
    (lambda r: setattr(r.events[4], "token", "strange"), "unknown token"),
    (lambda r: setattr(r.events[4], "amount", None), "missing"),
    (lambda r: setattr(r.events[4], "amount_text", "500.1"), "integer"),
    (lambda r: setattr(r.players[0], "stack_shown", None), "stack"),
    (lambda r: setattr(r.players[0], "cards", ["Ah", "Ah"]), "collision"),
    (lambda r: setattr(r.players[0], "cards", ["Ax", "Kd"]), "invalid card"),
    (lambda r: setattr(r.events[6], "amount", 399), "integer"),
    (lambda r: setattr(r, "cropped", True), "incomplete"),
    (lambda r: setattr(r, "unit", "unknown"), "units"),
    (lambda r: setattr(r, "stack_semantics", "unknown"), "semantics"),
    (lambda r: setattr(r.players[1], "button_shown", True), "button"),
    (lambda r: setattr(r.events[4], "street", "flop"), "contradicts"),
])
def test_incomplete_does_not_reach_raw_or_lose_transcript(mutation, fragment):
    reading = table()
    mutation(reading)
    result = process(reading)
    assert result.read_status == "incomplete"
    assert result.replay_status == "not_run"
    assert result.raw is None
    assert fragment in result.errors[0]
    assert "#7" in pokerdom_report(result)


@pytest.mark.parametrize("nickname", [None, "", "other"])
def test_hero_is_explicit_and_missing_is_neutral(nickname):
    result = process(table(), nickname=nickname)
    assert result.raw is None and result.replay_status == "not_run"
    assert "Hero" not in pokerdom_report(result)


def test_ambiguous_hero_and_not_hand():
    reading = table()
    reading.players[1].nickname = "synthetic-alice"
    assert process(reading).raw is None
    reading.is_hand = False
    assert process(reading).read_status == "not_hand"


@pytest.mark.parametrize("actual,status", [(450, "within_tolerance"), (550, "within_tolerance"),
    (449, "mismatched"), (551, "mismatched"), (500, "matched"), (None, "mismatched")])
def test_inclusive_event_tolerance_uses_printed_raise_addition(actual, status):
    expected = table()
    reading = deepcopy(expected)
    reading.events[4].amount = actual
    reading.events[4].amount_text = str(actual) if actual is not None else None
    result = process(reading, expected)
    assert result.truth_status == status
    comparison = next(c for c in result.amount_comparisons if c.number == 5)
    assert comparison.expected == 500 and comparison.actual == actual
    assert expected.events[4].amount == 500
    if actual != 500:
        assert result.replay_status != "passed"  # tolerance cannot repair UB or legality


def test_zero_missing_and_no_compensation():
    expected = table()
    expected.events[0].amount = 0
    expected.events[0].amount_text = "0"
    reading = deepcopy(expected)
    assert process(reading, expected).amount_comparisons[0].within_tolerance
    for value in (1, None):
        reading.events[0].amount = value
        reading.events[0].amount_text = str(value) if value is not None else None
        assert process(reading, expected).truth_status == "mismatched"
    expected = table()
    reading = table()
    reading.events[0].amount = 12
    reading.events[1].amount = 8
    assert process(reading, expected).truth_status == "mismatched"


@pytest.mark.parametrize("field", ["cards", "stack", "pot"])
def test_non_event_fields_have_no_tolerance(field):
    reading = table()
    if field == "cards":
        reading.players[0].cards[0] = "Ac"
    elif field == "stack":
        reading.players[0].stack_shown = 5001
    else:
        reading.pot_after_street[Street.PREFLOP] = 421
    result = process(reading, table())
    assert result.truth_status == "mismatched"
    if field != "pot":
        assert result.replay_status == "passed"  # legal wrong suit evades money checks
    else:
        assert result.replay_status == "failed"


def test_return_wrong_size_or_duplicate_is_rejected():
    reading = table()
    reading.events[-1].amount = 399
    reading.events[-1].amount_text = "399"
    assert "UB" in process(reading).errors[0]
    reading = table()
    event = reading.events[-1].model_copy(update={"number": 8})
    reading.events.append(event)
    assert process(reading).raw is None


def test_report_lists_all_exact_events_and_checks_without_analysis():
    result = process(table())
    text = pokerdom_report(result)
    assert all(f"#{i} " in text for i in range(1, 8))
    assert "рейз до 600 (доплата 500)" in text
    assert "UB 400 (возврат)" in text
    assert "Банк после улицы, после возврата (прочитано): 420" in text
    assert "карты неизвестны" in text
    assert "room_payout: not_checked" in text
    assert "Рассчитанные конечные стеки" in text
    assert "BB" not in text.split("старт до взносов")[0]
    assert "эквити" not in text and "стратег" not in text


def showdown_table(explicit=True, known=True):
    reading = table()
    reading.events = reading.events[:4]
    sequence = [("A", "C", 100, Street.PREFLOP), ("B", "Check", None, Street.PREFLOP)]
    for street in (Street.FLOP, Street.TURN, Street.RIVER):
        sequence.extend([("B", "Check", None, street), ("A", "Check", None, street)])
    reading.events.extend(PokerdomEvent(number=i, player=p, token=t, amount=a,
        amount_text=str(a) if a is not None else None, street=street)
        for i, (p, t, a, street) in enumerate(sequence, 5))
    reading.board = {Street.FLOP: ["2c", "3d", "7s"], Street.TURN: ["8c"], Street.RIVER: ["9d"]}
    reading.pot_after_street = dict.fromkeys(Street, 420)
    reading.players[0].showdown_shown = explicit
    reading.players[1].showdown_shown = explicit
    reading.players[1].cards = ["Qh", "Jd"] if known else []
    return reading


@pytest.mark.parametrize("explicit,known", [(False, True), (True, False)])
def test_visible_cards_or_unknown_showdown_cannot_confirm_outcome(explicit, known):
    result = process(showdown_table(explicit, known))
    assert not result.outcome_confirmed
    assert not known or result.outcome_basis == "visible_cards"
    assert "Рассчитанные конечные стеки" not in pokerdom_report(result)
    assert next(c for c in result.checks if c.name == "outcome").status == "not_checked"


def test_full_board_and_explicit_showdown_compute_outcome():
    result = process(showdown_table())
    assert result.replay_status == "passed"
    assert result.outcome_confirmed
    assert result.calculated_payouts == {"Hero": 420}
    assert len([c for c in result.checks if c.name.startswith("pot_after")]) == 4


def test_incomplete_board_and_missing_pot():
    reading = showdown_table()
    reading.board[Street.FLOP].pop()
    assert "incomplete board" in process(reading).errors[0]
    reading = table()
    reading.pot_after_street = {}
    result = process(reading)
    assert result.replay_status == "passed"
    assert next(c for c in result.checks if c.name == "pot_after.preflop").status == "not_checked"


def test_card_ten_normalizes_in_code_and_scoring():
    reading = table()
    reading.players[0].cards = ["10h", "Kd"]
    expected = table()
    expected.players[0].cards = ["Th", "Kd"]
    result = process(reading, expected)
    assert result.truth_status == "matched"
    assert result.raw is not None and result.raw.dealt["Hero"] == ["Th", "Kd"]


def test_bad_action_inside_tolerance_still_fails_replay():
    expected = showdown_table()
    reading = deepcopy(expected)
    reading.events[4].amount = 110
    reading.events[4].amount_text = "110"
    result = process(reading, expected)
    assert result.truth_status == "within_tolerance"
    assert result.replay_status == "failed"


@pytest.mark.parametrize("value", [1.5, True, "1"])
def test_chips_are_strict_integers(value):
    from pydantic import ValidationError
    data = table().model_dump()
    data["events"][0]["amount"] = value
    with pytest.raises(ValidationError):
        PokerdomReading.model_validate(data)


def test_local_row_keys_compare_by_row_relation_without_rewriting_reading():
    reading = table()
    reading.players[0].row_id = "row-one"
    reading.players[1].row_id = "row-two"
    for event in reading.events:
        event.player = {"A": "row-one", "B": "row-two"}[event.player]
    result = process(reading, table())
    assert result.truth_status == "matched"
    assert reading.players[0].row_id == "row-one"
    # Changing the link to the other row must still fail.
    reading.events[4].player = "row-two"
    assert process(reading, table()).truth_status == "mismatched"


def test_absent_and_explicit_empty_unplayed_street_are_equivalent_but_zero_is_not():
    reading = table()
    reading.board[Street.PREFLOP] = []
    reading.board[Street.RIVER] = []
    reading.pot_after_street[Street.RIVER] = None
    assert process(reading, table()).truth_status == "matched"
    reading.pot_after_street[Street.RIVER] = 0
    assert process(reading, table()).truth_status == "mismatched"
