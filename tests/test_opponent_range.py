"""Стартовый диапазон соперника по чарту и его сужение агентом — ядро без базы и сети.

Чарт защиты — синтетика во временном файле, а не файл владельца: ожидания не зависят
от его чисел; опен-чарт — файл репозитория, ожидания сверяются с `open_strategy`, а
не с числами чарта.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from harness.analysis.charts import open_strategy, parse_range
from harness.analysis.opponent_range import (
    RangeRefused,
    board_at,
    equity_spot,
    live_combos,
    narrow,
    range_spot,
    start_range,
)
from harness.contracts import Post, PostKind, Provenance, Range, RawHand, SeatInfo, Street
from tests.test_postflop_line import F, P, R, T, _bet, _call, _check, _fold, _played_out, _raise

_SB, _BB = 50, 100
# Кнопка — Hero: P5 — SB, P6 — BB, P1 — UTG, P2 — HJ, P3 — CO.
_SIX = ("P1", "P2", "P3", "Hero", "P5", "P6")
_BOARD = {F: ["Qc", "7d", "2s"], T: ["5h"], R: ["9s"]}
_DEFEND_CALL = "22+, A2s-AQs, KTs+, QTs+, JTs, ATo-AQo, KJo+"


def _raw(
    actions,
    *,
    labels: tuple[str, ...] = _SIX,
    button: str = "Hero",
    stacks: dict[str, int] | None = None,
    hero_cards: tuple[str, str] = ("Kd", "Tc"),
    boards: dict[Street, list[str]] | None = None,
) -> RawHand:
    stacks = stacks or {}
    at = labels.index(button)
    small, big = labels[(at + 1) % len(labels)], labels[(at + 2) % len(labels)]
    return RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="synthetic",
        hand_no="SYN-RANGE-1",
        tournament_id="TSYN",
        tournament_name="synthetic",
        level=1,
        sb=_SB,
        bb=_BB,
        ante=0,
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        table_name="syn",
        max_seats=len(labels),
        button_seat=at + 1,
        seats=[SeatInfo(seat=n, label=label, stack=stacks.get(label, 10_000))
               for n, label in enumerate(labels, 1)],
        posts=[
            Post(label=small, kind=PostKind.SMALL_BLIND, amount=_SB),
            Post(label=big, kind=PostKind.BIG_BLIND, amount=_BB),
        ],
        dealt={"Hero": list(hero_cards)},
        actions=actions,
        boards=boards or _BOARD,
    )


def _hand(actions, **kw):
    return _played_out(_raw(actions, **kw)).hand


BTN_OPENS_BB_CALLS = [
    _fold(P, "P1"), _fold(P, "P2"), _fold(P, "P3"), _raise(P, "Hero", 250), _fold(P, "P5"),
    _call(P, "P6", 150), _check(F, "P6"), _bet(F, "Hero", 160), _call(F, "P6", 160),
]


def defend_file(tmp_path: Path, **over: object) -> Path:
    entry: dict[str, object] = {
        "seats": 8, "position": "BB", "vs_position": "BTN", "depth_bb": 25,
        "ante_type": "per_player", "source": "синтетика", "revised_at": "2026-10-14",
        "call": _DEFEND_CALL, "raise": "AKs, AKo:0.5",
    }
    entry.update(over)
    path = tmp_path / "defend.json"
    path.write_text(json.dumps({"schema_version": 3, "entries": [entry]}, ensure_ascii=False),
                    encoding="utf-8")
    return path


# --- линия соперника и стартовый диапазон ---------------------------------------


def test_the_bb_call_against_a_button_open_takes_the_defence_call_range(tmp_path: Path):
    spot = range_spot(_hand(BTN_OPENS_BB_CALLS), "BB")
    assert (spot.label, spot.line, spot.opener_position, spot.seats) == ("P6", "call", "BTN", 6)
    assert spot.depth_bb == 100.0
    start = start_range(spot, defend_path=defend_file(tmp_path))
    assert start.range == parse_range(_DEFEND_CALL)
    assert (start.chart, start.chart_seats, start.chart_position, start.vs_chart_position) == (
        "defend", 8, "BB", "BTN",
    )
    assert start.chart_depth_bb == 25.0


def test_a_bb_three_bet_takes_the_raise_range(tmp_path: Path):
    actions = [
        _fold(P, "P1"), _fold(P, "P2"), _fold(P, "P3"), _raise(P, "Hero", 250), _fold(P, "P5"),
        _raise(P, "P6", 800), _call(P, "Hero", 550),
    ]
    spot = range_spot(_hand(actions), "BB")
    assert spot.line == "three_bet"
    assert start_range(spot, defend_path=defend_file(tmp_path)).range == parse_range("AKs, AKo:0.5")


def test_the_opener_range_is_the_raise_range_of_the_open_chart():
    """Герой на BB защищается против опена BTN: диапазон BTN — рейз опен-чарта,
    стол на 6 мест — по 8-max BTN."""
    actions = [
        _fold(P, "P5"), _fold(P, "P6"), _fold(P, "P1"), _raise(P, "P2", 250), _fold(P, "P3"),
        _call(P, "Hero", 150),
    ]
    spot = range_spot(_hand(actions, button="P2"), "BTN")
    assert (spot.label, spot.line, spot.depth_bb) == ("P2", "open_raise", 100.0)
    start = start_range(spot)
    assert start.range == open_strategy(8, "BTN", 100.0, "per_player").raise_range
    assert (start.chart, start.chart_seats, start.chart_position) == ("open", 8, "BTN")


def test_the_opener_depth_is_capped_by_the_deepest_player_left_behind():
    """Сбросившие до опена глубину открывшего не задают: позади только SB и BB."""
    actions = [
        _fold(P, "P5"), _fold(P, "P6"), _fold(P, "P1"), _raise(P, "P2", 250), _fold(P, "P3"),
        _call(P, "Hero", 150),
    ]
    hand = _hand(actions, button="P2", stacks={"P3": 2_500, "Hero": 3_000})
    assert range_spot(hand, "BTN").depth_bb == 30.0


def test_a_short_table_defence_reads_the_eight_max_chart_by_position_from_the_end(tmp_path: Path):
    """6 мест: опен UTG — пятеро позади, как у LJ за 8-max (§5)."""
    actions = [
        _raise(P, "P1", 250), _fold(P, "P2"), _fold(P, "P3"), _fold(P, "Hero"), _fold(P, "P5"),
        _call(P, "P6", 150),
    ]
    start = start_range(
        range_spot(_hand(actions), "BB"), defend_path=defend_file(tmp_path, vs_position="LJ")
    )
    assert (start.chart_seats, start.vs_chart_position, start.spot.seats) == (8, "LJ", 6)


def test_a_three_handed_defence_has_no_chart(tmp_path: Path):
    actions = [_raise(P, "Hero", 250), _fold(P, "P2"), _call(P, "P3", 150)]
    spot = range_spot(_hand(actions, labels=("Hero", "P2", "P3")), "BB")
    with pytest.raises(RangeRefused, match="чарта для этого спота нет"):
        start_range(spot, defend_path=defend_file(tmp_path))


def test_without_the_defence_file_the_spot_has_no_chart(tmp_path: Path):
    with pytest.raises(RangeRefused, match="чарта для этого спота нет"):
        start_range(range_spot(_hand(BTN_OPENS_BB_CALLS), "BB"), defend_path=tmp_path / "нет.json")


def test_a_short_stack_is_push_fold_not_a_chart(tmp_path: Path):
    spot = range_spot(_hand(BTN_OPENS_BB_CALLS, stacks={"P6": 1_000}), "BB")
    assert spot.depth_bb == 10.0
    with pytest.raises(RangeRefused, match="пуш-фолд"):
        start_range(spot, defend_path=defend_file(tmp_path))


_LIMPED = [_call(P, "P1", 100), _fold(P, "P2"), _fold(P, "P3"), _fold(P, "Hero"),
           _call(P, "P5", 50), _check(P, "P6")]
_LIMP_THEN_RAISE = [_call(P, "P1", 100), _fold(P, "P2"), _fold(P, "P3"), _raise(P, "Hero", 400),
                    _fold(P, "P5"), _call(P, "P6", 300), _call(P, "P1", 300)]
_COLD_CALL = [_fold(P, "P1"), _fold(P, "P2"), _fold(P, "P3"), _raise(P, "Hero", 250),
              _call(P, "P5", 200), _call(P, "P6", 150)]
_FOUR_BET = [_fold(P, "P1"), _fold(P, "P2"), _fold(P, "P3"), _raise(P, "Hero", 250),
             _fold(P, "P5"), _raise(P, "P6", 800), _raise(P, "Hero", 2_000), _call(P, "P6", 1_200)]
_SB_CALLS = [_fold(P, "P1"), _fold(P, "P2"), _fold(P, "P3"), _raise(P, "Hero", 250),
             _call(P, "P5", 200), _fold(P, "P6")]
_HERO_BB_THREE_BETS = [_fold(P, "P5"), _fold(P, "P6"), _fold(P, "P1"), _raise(P, "P2", 250),
                       _fold(P, "P3"), _raise(P, "Hero", 800), _call(P, "P2", 550)]


@pytest.mark.parametrize(
    ("actions", "button", "position", "reason"),
    [
        pytest.param(_LIMPED, "Hero", "UTG", "никто не повышал", id="лимп-банк"),
        pytest.param(_LIMP_THEN_RAISE, "Hero", "BB", "лимп", id="лимп до опена"),
        pytest.param(_COLD_CALL, "Hero", "BB", "между опеном и BB", id="колд-колл перед BB"),
        pytest.param(_FOUR_BET, "Hero", "BB", "не колл и не 3-бет", id="4-бет"),
        pytest.param(_SB_CALLS, "Hero", "SB", "только для BB", id="колл не из BB"),
        pytest.param(_HERO_BB_THREE_BETS, "P2", "BTN", "3-бет", id="открывший после 3-бета"),
        pytest.param(BTN_OPENS_BB_CALLS, "Hero", "BTN", "героя", id="позиция героя"),
        pytest.param(BTN_OPENS_BB_CALLS, "Hero", "UTG+1", "нет позиции", id="позиции нет"),
    ],
)
def test_lines_outside_the_charts_are_refused(actions, button, position, reason):
    with pytest.raises(RangeRefused, match=reason):
        range_spot(_hand(actions, button=button), position)


# --- сужение и мёртвые карты -------------------------------------------------------

_START = Range(weights={"AA": 1.0, "KK": 1.0, "QQ": 0.0, "AKs": 0.5, "72o": 0.0})


def test_narrowing_keeps_the_chart_weights_of_the_kept_classes():
    assert narrow(_START, "AA, AKs") == Range(weights={"AA": 1.0, "AKs": 0.5})


def test_a_class_with_zero_weight_in_the_chart_is_outside_the_start():
    with pytest.raises(RangeRefused, match="72o"):
        narrow(_START, "AA, 72o")


def test_a_class_outside_the_start_is_refused_by_name():
    """«QQ+» — QQ, KK, AA: QQ в чарте с долей 0, сужение рук не добавляет."""
    with pytest.raises(RangeRefused, match="нет в стартовом диапазоне: QQ"):
        narrow(_START, "QQ+")


def test_weights_from_the_agent_are_refused():
    with pytest.raises(RangeRefused, match="веса"):
        narrow(_START, "AKs:1")


def test_dead_cards_leave_the_combos():
    rng = parse_range("AA, AKs")
    assert live_combos(rng, ()) == 10.0
    assert live_combos(rng, ("As",)) == 6.0  # AA: 3 без пиковой, AKs: 3
    assert live_combos(Range(weights={"AKs": 0.5}), ()) == 2.0


def test_the_board_is_the_board_of_the_street():
    hand = _hand(BTN_OPENS_BB_CALLS)
    assert board_at(hand, Street.PREFLOP) == []
    assert board_at(hand, Street.TURN) == ["Qc", "7d", "2s", "5h"]


def test_the_equity_spot_takes_the_street_board_and_the_share_of_the_start(tmp_path: Path):
    spot = equity_spot(
        _hand(BTN_OPENS_BB_CALLS), "BB", Street.FLOP, "QQ, 77, 22, AQs",
        defend_path=defend_file(tmp_path),
    )
    assert spot.hero == ("Kd", "Tc") and spot.board == ["Qc", "7d", "2s"]
    assert spot.live_combos == 12.0  # по 3 у каждой пары с картой на борде, AQs без Qc — 3
    dead = ("Kd", "Tc", "Qc", "7d", "2s")
    assert spot.share_of_start == pytest.approx(12.0 / live_combos(parse_range(_DEFEND_CALL), dead))


def test_a_fully_blocked_range_is_refused_not_zero(tmp_path: Path):
    hand = _hand(
        BTN_OPENS_BB_CALLS, hero_cards=("Kd", "Kc"),
        boards={F: ["Kh", "7d", "2s"], T: ["5h"], R: ["9s"]},
    )
    with pytest.raises(RangeRefused, match="заблокированы"):
        equity_spot(hand, "BB", Street.FLOP, "KK", defend_path=defend_file(tmp_path))


def test_multiway_equity_is_refused():
    """Три живых на флопе: мультивей — вне среза (§3)."""
    actions = [
        _fold(P, "P5"), _fold(P, "P6"), _fold(P, "P1"), _raise(P, "P2", 250), _call(P, "P3", 200),
        _call(P, "Hero", 150),
    ]
    with pytest.raises(RangeRefused, match="больше двух"):
        equity_spot(_hand(actions, button="P2"), "BTN", Street.FLOP, "AA")


# --- лимп SB и чек BB (решение владельца 2026-10-14, §6.1) ---------------------------

# Кнопка — P2: P3 — SB, Hero — BB, P5 — UTG, P6 — HJ, P1 — CO.
_SB_LIMPS_HERO_CHECKS = [
    _fold(P, "P5"), _fold(P, "P6"), _fold(P, "P1"), _fold(P, "P2"), _call(P, "P3", 50),
    _check(P, "Hero"),
]
# Кнопка — P3: Hero — SB, P5 — BB.
_HERO_LIMPS_BB_CHECKS = [
    _fold(P, "P6"), _fold(P, "P1"), _fold(P, "P2"), _fold(P, "P3"), _call(P, "Hero", 50),
    _check(P, "P5"),
]
# Кнопка — Hero: P5 — SB, P6 — BB.
_SB_LIMPS_BB_ISOLATES = [
    _fold(P, "P1"), _fold(P, "P2"), _fold(P, "P3"), _fold(P, "Hero"), _call(P, "P5", 50),
    _raise(P, "P6", 400), _call(P, "P5", 300),
]
_LIMP_CHECK = "22-99, A2s-A9s, K2s-K9s, Q5s+, J7s+, T7s+, 97s+, 87s"


def limp_file(tmp_path: Path) -> Path:
    return defend_file(
        tmp_path, vs_position="SB", vs_action="limp", call="", check=_LIMP_CHECK,
        **{"raise": "TT+, AJs+, AQo+"},
    )


def test_the_sb_limp_takes_the_limp_range_of_the_open_chart():
    spot = range_spot(_hand(_SB_LIMPS_HERO_CHECKS, button="P2"), "SB")
    assert (spot.label, spot.line, spot.opener_position, spot.depth_bb) == ("P3", "limp", "SB", 100.0)
    start = start_range(spot)
    assert start.range == open_strategy(8, "SB", 100.0, "per_player").limp_range
    assert (start.chart, start.chart_position, start.vs_chart_position) == ("open", "SB", None)


def test_the_bb_check_after_a_sb_limp_takes_the_check_range_against_the_limp(tmp_path: Path):
    spot = range_spot(_hand(_HERO_LIMPS_BB_CHECKS, button="P3"), "BB")
    assert (spot.label, spot.line, spot.opener_position) == ("P5", "check", "SB")
    start = start_range(spot, defend_path=limp_file(tmp_path))
    assert start.range == parse_range(_LIMP_CHECK)
    assert (start.chart, start.chart_position, start.vs_chart_position) == ("defend", "BB", "SB")


def test_the_check_range_is_not_taken_from_the_chart_against_a_raise(tmp_path: Path):
    """Файл без записи против лимпа: запись против рейза SB не подставляется."""
    spot = range_spot(_hand(_HERO_LIMPS_BB_CHECKS, button="P3"), "BB")
    with pytest.raises(RangeRefused, match="чарта для этого спота нет"):
        start_range(spot, defend_path=defend_file(tmp_path, vs_position="SB"))


@pytest.mark.parametrize("position", ["SB", "BB"])
def test_an_isolation_after_the_sb_limp_is_refused(position: str):
    hand = _hand(_SB_LIMPS_BB_ISOLATES)
    with pytest.raises(RangeRefused, match="изоляц"):
        range_spot(hand, position)


# --- решения ядра там, где бриф молчит ------------------------------------------------


def test_a_call_against_an_all_in_open_is_not_the_call_against_a_raise():
    """Чарт защиты снят против опена рейзом: колл олл-ина им не описывается."""
    actions = [
        _fold(P, "P1"), _fold(P, "P2"), _fold(P, "P3"),
        _raise(P, "Hero", 2_000).model_copy(update={"is_all_in": True}), _fold(P, "P5"),
        _call(P, "P6", 1_900),
    ]
    with pytest.raises(RangeRefused, match="олл-ин"):
        range_spot(_hand(actions, stacks={"Hero": 2_000}), "BB")


def test_an_empty_chart_range_is_refused_not_an_empty_start(tmp_path: Path):
    """BB сыграл 3-бет, а в чарте 3-бета нет ни одной руки: диапазона нет, а не ноль."""
    actions = [
        _fold(P, "P1"), _fold(P, "P2"), _fold(P, "P3"), _raise(P, "Hero", 250), _fold(P, "P5"),
        _raise(P, "P6", 800), _call(P, "Hero", 550),
    ]
    spot = range_spot(_hand(actions), "BB")
    with pytest.raises(RangeRefused, match="ни одной руки"):
        start_range(spot, defend_path=defend_file(tmp_path, **{"raise": ""}))


def test_a_player_who_folded_preflop_has_no_range():
    with pytest.raises(RangeRefused, match="сбросил"):
        range_spot(_hand(BTN_OPENS_BB_CALLS), "SB")
