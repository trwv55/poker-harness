"""Synthetic v2 actor binding and source completeness regressions; no model calls."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from harness.contracts.pokerdom_vision import (
    PokerdomReading,
    PokerdomReadingV2,
    PokerdomResult,
    decode_pokerdom_reading,
)
from harness.contracts.raw import Street
from harness.parsers.pokerdom_vision import process_pokerdom
from harness.presentation.pokerdom_vision import pokerdom_report
from tests.test_pokerdom_vision import showdown_table, table


def v2(legacy=None):
    legacy = table() if legacy is None else legacy
    names = {p.row_id: p.nickname for p in legacy.players}
    return PokerdomReadingV2.model_validate({
        **legacy.model_dump(exclude={"schema_version", "complete", "issues", "players", "events"}),
        "schema_version": 2, "action_log_complete": legacy.complete,
        "players": [{**p.model_dump(exclude={"row_id"}), "nickname_complete": True}
                    for p in legacy.players],
        "events": [{**e.model_dump(exclude={"player"}), "player_nickname": names[e.player]}
                   for e in legacy.events],
        "uncertainties": [], "notes": [],
    })


def process(reading, expected=None):
    return process_pokerdom(reading, nickname="synthetic-alice", expected=expected)


def test_new_wire_has_nicknames_only_code_assigns_rows_and_preserves_source():
    reading = v2()
    before = reading.model_dump()
    result = process(reading, table())
    assert result.replay_status == "passed" and result.truth_status == "matched"
    assert result.raw is not None and result.raw.vision is not None
    assert result.raw.vision.nicknames == {"Hero": "synthetic-alice", "P2": "synthetic-bob"}
    assert result.raw.actions[0].to_amount == 600
    assert result.enriched is not None and result.enriched.report.stacks_end == {"Hero": 5210, "P2": 4790}
    assert reading.model_dump() == before
    assert "row_id" not in reading.model_dump_json() and '"player":' not in reading.model_dump_json()
    assert process(table(), reading).truth_status == "matched"


@pytest.mark.parametrize("nickname", [None, "A", "Hero", "synthetic-alic", "Synthetic-alice", "synthetic-alice "])
def test_no_fuzzy_or_id_or_position_actor_fallback(nickname):
    reading = v2()
    reading.events[4].player_nickname = nickname
    result = process(reading, table())
    assert result.read_status == "incomplete" and result.raw is None
    assert result.replay_status == "not_run" and result.truth_status == "mismatched"
    assert "nickname" in result.errors[0]
    assert "#7" in pokerdom_report(result)


@pytest.mark.parametrize("mode", ["duplicate", "truncated", "unknown"])
def test_ambiguous_player_row_is_diagnostic_not_merged(mode):
    reading = v2()
    if mode == "duplicate":
        reading.players[1].nickname = reading.players[0].nickname
    elif mode == "truncated":
        reading.players[0].nickname_complete = False
    else:
        reading.players[0].nickname = None
    result = process(reading)
    assert result.raw is None and result.replay_status == "not_run"
    assert "unknown, duplicate or truncated" in result.errors[0]


def test_different_hands_do_not_share_a_persistent_nickname_identity():
    first = v2()
    second = v2()
    second.players[1].stack_shown = 6000
    a, b = process(first), process(second)
    assert a.raw is not None and b.raw is not None
    assert a.raw.seats[1].stack == 5000 and b.raw.seats[1].stack == 6000
    assert a.raw.seats[1].label == b.raw.seats[1].label == "P2"  # hand-local only


def test_informational_absences_do_not_block_fold_out():
    reading = v2()
    reading.notes = ["No payout/end-stack row", "No explicit showdown", "River unplayed"]
    result = process(reading)
    assert result.replay_status == "passed" and result.outcome_basis == "folds"
    assert result.raw is not None and result.raw.collected == [] and result.raw.summary is None
    checks = {c.name: c.status for c in result.checks}
    assert checks["room_payout"] == checks["room_end_stacks"] == "not_checked"
    text = pokerdom_report(result)
    assert "Примечание: No payout" in text and "UB 400" in text


def test_complete_river_without_explicit_showdown_stays_conditional():
    reading = v2(showdown_table(explicit=False))
    reading.notes = ["No winner or payout", "Visible cards, no explicit showdown indication"]
    result = process(reading)
    assert result.replay_status == "passed"
    assert result.outcome_basis == "visible_cards" and not result.outcome_confirmed
    assert result.raw is not None and result.raw.showdowns == []
    assert next(c for c in result.checks if c.name == "outcome").status == "not_checked"
    assert "Условный расчёт" in pokerdom_report(result)


def test_uncertainty_is_blocking_even_when_values_are_present():
    data = v2().model_dump()
    data["uncertainties"] = [{"field": "events[4].amount", "reason": "digits unreadable"}]
    reading = PokerdomReadingV2.model_validate(data)
    result = process(reading)
    assert result.raw is None and result.replay_status == "not_run"
    assert "events[4].amount" in result.errors[0]
    assert "Неопределённость" in pokerdom_report(result)


@pytest.mark.parametrize("mode", ["amount", "stack", "board", "button", "showdown_cards", "crop", "incomplete"])
def test_notes_cannot_hide_missing_required_facts(mode):
    reading = v2(showdown_table())
    reading.notes = ["informational, not a license to fill facts"]
    if mode == "amount":
        reading.events[4].amount = None
    elif mode == "stack":
        reading.players[0].stack_shown = None
    elif mode == "board":
        reading.board[Street.FLOP].pop()
    elif mode == "button":
        reading.players[0].button_shown = False
    elif mode == "showdown_cards":
        reading.players[1].cards = []
    elif mode == "crop":
        reading.cropped = True
    else:
        reading.action_log_complete = False
    result = process(reading)
    assert result.raw is None and result.replay_status == "not_run"
    assert result.read_status == "incomplete"


@pytest.mark.parametrize("mode", ["preflop_only", "river_missing_check"])
def test_model_complete_claim_is_not_terminal_evidence(mode):
    reading = v2(showdown_table(explicit=False))
    if mode == "preflop_only":
        reading.events = reading.events[:6]
        reading.board = {}
        reading.pot_after_street = {Street.PREFLOP: 420}
    else:
        reading.events.pop()  # contiguous numbering, but only one river check remains
    assert reading.action_log_complete
    result = process(reading)
    assert result.raw is None and result.replay_status == "not_run"
    assert "log" in result.errors[0]


def test_legacy_decode_keeps_bad_actor_format_and_legacy_issues_blocking():
    data = table().model_dump(mode="json")
    data.pop("schema_version")
    old = decode_pokerdom_reading(data)
    assert isinstance(old, PokerdomReading)
    old.events[4].player = "synthetic-alice"  # nickname was not a v1 row key
    result = process(old, table())
    assert result.replay_status == "not_run" and result.truth_status == "mismatched"
    old = table()
    old.issues = ["No payout row"]
    assert process(old).replay_status == "not_run"  # no text heuristics rewriting historical v1
    assert isinstance(decode_pokerdom_reading(v2().model_dump()), PokerdomReadingV2)
    with pytest.raises(ValueError, match="unsupported"):
        decode_pokerdom_reading({"schema_version": 3})


def test_saved_results_roundtrip_both_versions_and_reject_mixed_wire_fields():
    for reading in (table(), v2()):
        result = process(reading)
        assert PokerdomResult.model_validate_json(result.model_dump_json()) == result
    data = v2().model_dump()
    data["players"][0]["row_id"] = "invented"
    with pytest.raises(ValidationError):
        PokerdomReadingV2.model_validate(data)


def test_v2_comparison_remains_exact_for_actors_and_only_tolerates_event_money():
    original = v2()
    actual = deepcopy(original)
    actual.events[4].amount = 550
    actual.events[4].amount_text = "550"
    result = process(actual, original)
    assert result.truth_status == "within_tolerance" and result.replay_status != "passed"
    actual = deepcopy(original)
    actual.players[0].stack_shown = 5001
    assert process(actual, original).truth_status == "mismatched"


def test_literal_printed_hero_nickname_is_not_replaced_by_an_identity_guess():
    reading = v2()
    old_name = reading.players[0].nickname
    reading.players[0].nickname = "Hero"
    for event in reading.events:
        if event.player_nickname == old_name:
            event.player_nickname = "Hero"
    result = process_pokerdom(reading, nickname="Hero")
    assert result.replay_status == "passed" and result.raw is not None
    assert result.raw.vision is not None and result.raw.vision.nicknames["Hero"] == "Hero"
