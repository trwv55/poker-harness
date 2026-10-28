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
    absent_groups,
    board_at,
    classes_named,
    equity_spot,
    live_at,
    live_combos,
    narrow,
    range_spot,
    start_range,
)
from harness.analysis.player_stats import hero_bet_called
from harness.contracts import (
    FrequencyStat,
    Post,
    PostKind,
    Provenance,
    Range,
    RangeNarrowing,
    RawHand,
    SeatInfo,
    Street,
)
from tests.test_postflop_line import F, P, R, T, _bet, _call, _check, _fold, _played_out, _raise

_SB, _BB = 50, 100
# 30bb у всех по умолчанию: ближайший чарт защиты (25bb) не дальше допуска глубины.
_STACK = 3_000
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
        seats=[SeatInfo(seat=n, label=label, stack=stacks.get(label, _STACK))
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
    assert spot.depth_bb == 30.0
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
    assert (spot.label, spot.line, spot.depth_bb) == ("P2", "open_raise", 30.0)
    start = start_range(spot)
    assert start.range == open_strategy(8, "BTN", 30.0, "per_player").raise_range
    assert (start.chart, start.chart_seats, start.chart_position) == ("open", 8, "BTN")


def test_the_opener_depth_is_capped_by_the_deepest_player_left_behind():
    """Сбросившие до опена глубину открывшего не задают: позади только SB и BB."""
    actions = [
        _fold(P, "P5"), _fold(P, "P6"), _fold(P, "P1"), _raise(P, "P2", 250), _fold(P, "P3"),
        _call(P, "Hero", 150),
    ]
    hand = _hand(actions, button="P2", stacks={
        "P5": 10_000, "P6": 10_000, "P1": 10_000, "P2": 10_000, "P3": 2_500, "Hero": 3_000,
    })
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



def test_preflop_equity_counts_the_players_at_the_hero_decision(tmp_path: Path):
    """BTN открывает, BB 3-бетит, SB сбросил раньше: на ответе героя банк хедз-ап."""
    actions = [
        _fold(P, "P1"), _fold(P, "P2"), _fold(P, "P3"), _raise(P, "Hero", 250), _fold(P, "P5"),
        _raise(P, "P6", 800), _call(P, "Hero", 550),
    ]
    spot = equity_spot(
        _hand(actions), "BB", Street.PREFLOP, "AKs", defend_path=defend_file(tmp_path)
    )
    assert spot.board == []
    assert spot.live_combos == 3.0  # AKs без Kd


def test_a_third_player_folding_before_the_hero_decision_leaves_a_heads_up_flop():
    """CO открывает, герой на BTN и SB коллируют; на флопе SB сбросил на ставку героя
    до того, как герой отвечает на рейз CO."""
    actions = [
        _fold(P, "P1"), _fold(P, "P2"), _raise(P, "P3", 250), _call(P, "Hero", 250),
        _call(P, "P5", 200), _fold(P, "P6"),
        _check(F, "P5"), _check(F, "P3"), _bet(F, "Hero", 300), _fold(F, "P5"),
        _raise(F, "P3", 900), _call(F, "Hero", 600),
    ]
    spot = equity_spot(_hand(actions), "CO", Street.FLOP, "AA")
    assert spot.start.spot.line == "open_raise"


def test_no_hero_turn_on_the_street_is_no_decision_point():
    """Герой пошёл олл-ин на префлопе: на флопе хода у него нет — точки решения нет,
    отказ словами, а не места с префлопа."""
    actions = [
        _fold(P, "P1"), _fold(P, "P2"), _fold(P, "P3"), _raise(P, "Hero", _STACK),
        _fold(P, "P5"), _call(P, "P6", _STACK - 100),
    ]
    hand = _hand(actions)
    assert "Hero" in live_at(hand, Street.PREFLOP)
    with pytest.raises(RangeRefused, match="точки решения нет"):
        live_at(hand, Street.FLOP)


def test_three_players_at_the_hero_preflop_decision_are_multiway():
    actions = [
        _fold(P, "P5"), _fold(P, "P6"), _fold(P, "P1"), _raise(P, "P2", 250), _call(P, "P3", 200),
        _call(P, "Hero", 150),
    ]
    with pytest.raises(RangeRefused, match="мультивей"):
        equity_spot(_hand(actions, button="P2"), "BTN", Street.PREFLOP, "AA")


# --- допуск глубины чарта защиты ----------------------------------------------------


@pytest.mark.parametrize(
    ("bb_stack", "refused"),
    [
        pytest.param(3_490, False, id="34.9bb — чарт 25bb"),
        pytest.param(3_500, False, id="35bb — ровно 10bb, ещё чарт"),
        pytest.param(3_540, True, id="35.4bb — дальше 10bb"),
    ],
)
def test_a_defence_chart_further_than_ten_bb_away_is_refused(
    tmp_path: Path, bb_stack: int, refused: bool
):
    hand = _hand(BTN_OPENS_BB_CALLS, stacks={"P6": bb_stack, "Hero": 10_000})
    spot = range_spot(hand, "BB")
    if refused:
        with pytest.raises(RangeRefused, match="чарта защиты для такой глубины нет: ближайший 25 BB"):
            start_range(spot, defend_path=defend_file(tmp_path))
    else:
        assert start_range(spot, defend_path=defend_file(tmp_path)).chart_depth_bb == 25.0


def test_the_open_chart_keeps_the_nearest_depth_without_a_limit():
    """Опен-чарт допуска не получил: 100bb судится самым глубоким чартом."""
    actions = [
        _fold(P, "P5"), _fold(P, "P6"), _fold(P, "P1"), _raise(P, "P2", 250), _fold(P, "P3"),
        _call(P, "Hero", 150),
    ]
    hand = _hand(actions, button="P2", stacks=dict.fromkeys(_SIX, 10_000))
    start = start_range(range_spot(hand, "BTN"))
    assert start.spot.depth_bb == 100.0
    assert start.chart_depth_bb == 60.0

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
    assert (spot.label, spot.line, spot.opener_position, spot.depth_bb) == ("P3", "limp", "SB", 30.0)
    start = start_range(spot)
    assert start.range == open_strategy(8, "SB", 30.0, "per_player").limp_range
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


# --- план 2026-10-17, п. 6: сужение улицы — подмножество прошлой ---------------------

_THROUGH_RIVER = [
    *BTN_OPENS_BB_CALLS,
    _check(T, "P6"), _bet(T, "Hero", 400), _call(T, "P6", 400),
    _check(R, "P6"), _bet(R, "Hero", 900), _call(R, "P6", 900),
]


def _prior(street: Street, classes: str) -> RangeNarrowing:
    return RangeNarrowing(
        analysis_id=1, opponent_position="BB", street=street,
        classes=list(parse_range(classes).weights),
    )


def test_a_class_removed_on_the_flop_is_refused_on_the_turn_by_name(tmp_path: Path):
    """Класс стартового, убранный на флопе, тёрн не возвращает: отказ называет классы,
    улицу, где их убрали, и откуда берётся сужение."""
    with pytest.raises(RangeRefused) as refused:
        equity_spot(
            _hand(_THROUGH_RIVER), "BB", Street.TURN, "QQ, 77, A5s, KJo",
            prior=_prior(Street.FLOP, "QQ, 77"), defend_path=defend_file(tmp_path),
        )
    assert str(refused.value) == (
        "на флопе эти классы вы уже убрали: A5s, KJo — сужение тёрна берётся из сужения флопа"
    )


_FIRST_THE_FLOP = (
    "сначала сузьте флоп: вызовите equity_vs_range для флопа этого соперника, тёрн "
    "сужается из него"
)
_FIRST_THE_TURN = (
    "сначала сузьте тёрн: вызовите equity_vs_range для тёрна этого соперника, ривер "
    "сужается из него"
)


@pytest.mark.parametrize(
    ("street", "prior", "refusal"),
    [
        (Street.TURN, None, _FIRST_THE_FLOP),
        (Street.RIVER, None, _FIRST_THE_TURN),
        # Ривер с сужением флопа, но без тёрна: граница — строго предыдущая улица.
        (Street.RIVER, Street.FLOP, _FIRST_THE_TURN),
        (Street.TURN, Street.TURN, _FIRST_THE_FLOP),
        (Street.TURN, Street.PREFLOP, _FIRST_THE_FLOP),
    ],
)
def test_a_later_street_without_the_previous_narrowing_is_refused(
    tmp_path: Path, street: Street, prior: Street | None, refusal: str
):
    """Тёрн без сужения флопа и ривер без сужения тёрна — отказ словами модели, даже
    если классы — из стартового (решение владельца 2026-10-23)."""
    with pytest.raises(RangeRefused) as refused:
        equity_spot(
            _hand(_THROUGH_RIVER), "BB", street, "QQ",
            prior=_prior(prior, "QQ, A5s") if prior is not None else None,
            defend_path=defend_file(tmp_path),
        )
    assert str(refused.value) == refusal


def test_the_flop_without_a_prior_is_narrowed_from_the_start(tmp_path: Path):
    spot = equity_spot(
        _hand(_THROUGH_RIVER), "BB", Street.FLOP, "QQ, A5s", defend_path=defend_file(tmp_path)
    )
    assert spot.narrowed == Range(weights={"QQ": 1.0, "A5s": 1.0})


def test_the_turn_after_the_flop_and_the_river_after_the_turn_pass(tmp_path: Path):
    turn = equity_spot(
        _hand(_THROUGH_RIVER), "BB", Street.TURN, "QQ",
        prior=_prior(Street.FLOP, "QQ, A5s"), defend_path=defend_file(tmp_path),
    )
    river = equity_spot(
        _hand(_THROUGH_RIVER), "BB", Street.RIVER, "QQ",
        prior=_prior(Street.TURN, "QQ"), defend_path=defend_file(tmp_path),
    )
    assert turn.narrowed == river.narrowed == Range(weights={"QQ": 1.0})


def test_a_subset_of_the_earlier_narrowing_keeps_the_chart_weights():
    start = parse_range("QQ, AKs:0.5, A5s")
    narrowed = narrow(start, "AKs", street=Street.TURN, prior=_prior(Street.FLOP, "QQ, AKs"))
    assert narrowed == Range(weights={"AKs": 0.5})


def test_a_class_outside_the_start_is_still_refused_as_outside_with_a_prior():
    """Класса нет ни в стартовом, ни в прошлом сужении — это «нет в стартовом», а не
    «уже убрали»."""
    with pytest.raises(RangeRefused, match="нет в стартовом диапазоне: 72o"):
        narrow(_START, "AA, 72o", street=Street.TURN, prior=_prior(Street.FLOP, "AA"))


def test_the_river_is_bound_by_the_turn_and_the_message_names_the_streets():
    with pytest.raises(RangeRefused, match="на тёрне эти классы вы уже убрали: KK — сужение "
                       "ривера берётся из сужения тёрна"):
        narrow(_START, "AA, KK", street=Street.RIVER, prior=_prior(Street.TURN, "AA"))


# --- план 2026-10-17, п. 9: классы вне чарта по просьбе игрока ------------------------


def test_a_class_the_player_named_is_added_with_full_combos(tmp_path: Path):
    """«22–77» игрока: классы вне чарта входят с долей 1.0 и названы в `added`."""
    start = parse_range("88+, AKs:0.5")
    narrowed = narrow(start, "AA, AKs", added="22-77", named=classes_named("а если пары 22–77?"))
    assert narrowed.weights == {
        "AA": 1.0, "AKs": 0.5, **dict.fromkeys(["77", "66", "55", "44", "33", "22"], 1.0),
    }


def test_the_equity_spot_names_the_added_classes(tmp_path: Path):
    spot = equity_spot(
        _hand(BTN_OPENS_BB_CALLS), "BB", Street.FLOP, "QQ", added="KJs",
        named=frozenset({"KJs"}), defend_path=defend_file(tmp_path, call="QQ+, A2s-AQs"),
    )
    assert spot.added == ("KJs",)
    assert spot.narrowed.weight("KJs") == 1.0


def test_a_class_added_with_the_model_reason_needs_no_player_record():
    """С причиной модели (`reason`) класс вне чарта входит без записи игрока, с долей
    1.0 (спека 2026-10-24, §7); класс чарта в `added` — прежний отказ."""
    start = parse_range("88+")
    narrowed = narrow(start, "AA", added="22-33", reason="на этих лимитах коллирует пары")
    assert narrowed.weights == {"AA": 1.0, "33": 1.0, "22": 1.0}
    with pytest.raises(RangeRefused, match="передайте их в hands"):
        narrow(start, "AA", added="99", reason="причина")


def test_an_agent_class_is_inherited_by_the_next_street_with_its_reason(tmp_path: Path):
    """Класс агента с причиной — `agent_added`; на тёрне, переданный в `hands` из сужения
    флопа, он остаётся классом агента с той же причиной; класс игрока — игрока."""
    defend = defend_file(tmp_path, call="QQ+, A2s-AQs")
    flop = equity_spot(
        _hand(_THROUGH_RIVER), "BB", Street.FLOP, "QQ", added="KJs, 22",
        reason="  на этих лимитах так коллируют  ", defend_path=defend,
    )
    assert flop.added == ("22", "KJs")
    assert flop.agent_added == {"22": "на этих лимитах так коллируют",
                                "KJs": "на этих лимитах так коллируют"}
    prior = _prior(Street.FLOP, "QQ, KJs, 22, 33").model_copy(
        update={"agent_added": {"KJs": "причина флопа"}}
    )
    turn = equity_spot(
        _hand(_THROUGH_RIVER), "BB", Street.TURN, "QQ, KJs, 33", prior=prior,
        defend_path=defend,
    )
    assert turn.added == ("33", "KJs")
    assert turn.agent_added == {"KJs": "причина флопа"}


def test_a_class_the_player_did_not_name_is_refused():
    with pytest.raises(RangeRefused, match="игрок не назвал эти классы записью: 33 .*уточните"):
        narrow(parse_range("88+"), "AA", added="22-33", named=frozenset({"22"}))


def test_a_charted_class_cannot_be_added():
    """Класс из чарта — в `hands`, а не в `added`, даже если игрок его назвал."""
    with pytest.raises(RangeRefused, match="есть в стартовом диапазоне по чарту: AA"):
        narrow(parse_range("88+"), "KK", added="AA", named=frozenset({"AA"}))


def test_weights_in_the_added_classes_are_refused():
    with pytest.raises(RangeRefused, match="веса"):
        narrow(parse_range("88+"), "AA", added="22:0.5", named=frozenset({"22"}))


def test_an_added_class_kept_from_the_last_narrowing_needs_no_new_request():
    """Класс вне чарта, уже добавленный в этой руке, принимается в `added` без новой
    записи игрока; новый — нет."""
    start = parse_range("88+")
    assert narrow(start, "AA", added="22", kept=["AA", "22"]).weight("22") == 1.0
    with pytest.raises(RangeRefused, match="игрок не назвал эти классы записью: 33"):
        narrow(start, "AA", added="22, 33", kept=["AA", "22"])


def test_an_added_class_is_inherited_by_the_next_street():
    """Добавленный на флопе класс вне чарта на тёрне — обычный класс прошлого сужения,
    с долей 1.0 и без повторной просьбы игрока."""
    start = parse_range("88+")
    narrowed = narrow(start, "AA, 22", street=Street.TURN, prior=_prior(Street.FLOP, "AA, 22"))
    assert narrowed == Range(weights={"AA": 1.0, "22": 1.0})


@pytest.mark.parametrize(
    ("text", "classes"),
    [
        ("а если у него бывают пары 22–77?", "22-77"),
        ("пары 22-77", "22-77"),
        ("пары 22 — 44", "22-44"),
        ("A2s-A5s и KQo", "A2s-A5s, KQo"),
        ("а 77+?", "77+"),
        ("AJs+ тоже", "AJs+"),
    ],
)
def test_the_player_names_classes_by_notation(text: str, classes: str):
    assert classes_named(text) == frozenset(parse_range(classes).weights)


@pytest.mark.parametrize(
    "text",
    [
        "мелкие пары", "все пары", "AK", "стек 22 BB", "22 бб", "в 33% случаев", "25-30",
        "стек 33-44 BB", "55 % эквити", "блайнды 55/110", "уровень 22", "на уровне 22",
        "88 игроков", "в 22:30", "№66", "77 фишек", "33 раздачи",
    ],
)
def test_words_name_no_classes(text: str):
    """Слова, непара без суффикса и число из цифр в контексте счёта — не запись класса:
    перед единицей (BB, %, игроки, фишки, раздачи…), «/» или «:», после «№» или
    «уровень»; диапазон цифр — так же."""
    assert classes_named(text) == frozenset()


# --- план 2026-10-17, п. 8: чего нет в колле защиты -----------------------------------


def test_the_absent_groups_are_the_classes_the_chart_raises_or_shoves():
    """Пары целиком — рейз и олл-ин; тузы — по составу; бродвей без туза — олл-ин.
    Класс, который чарт сбрасывает (нет ни в одном действии), не отсутствующий."""
    groups = absent_groups(
        parse_range("A2s-A9s, KJs+, KQo"),
        parse_range("TT+, AKs, ATs-AQs"),
        parse_range("22-99, A9o+, QJs"),
    )
    assert [g.model_dump() for g in groups] == [
        {"group": "pairs", "classes": list(parse_range("22+").weights)[::-1], "whole": True,
         "raised": True, "all_in": True},
        {"group": "suited_aces", "classes": ["AKs", "AQs", "AJs", "ATs"], "whole": False,
         "raised": True, "all_in": False},
        {"group": "offsuit_aces", "classes": ["AKo", "AQo", "AJo", "ATo", "A9o"],
         "whole": False, "raised": False, "all_in": True},
        {"group": "broadways", "classes": ["QJs"], "whole": False, "raised": False,
         "all_in": True},
    ]


def test_a_class_partly_called_is_not_absent():
    """Доля колла 0.2 — класс в колле есть, хоть чарт и 3-бетит его чаще."""
    assert absent_groups(parse_range("AA:0.2"), parse_range("AA:0.8"), Range(weights={})) == ()


def test_the_bb_call_range_carries_its_absent_groups_and_a_three_bet_does_not(tmp_path: Path):
    path = defend_file(tmp_path)
    call = start_range(range_spot(_hand(BTN_OPENS_BB_CALLS), "BB"), defend_path=path)
    assert [(g.group, g.classes) for g in call.absent] == [
        ("suited_aces", ["AKs"]), ("offsuit_aces", ["AKo"]),
    ]
    three_bet = [
        _fold(P, "P1"), _fold(P, "P2"), _fold(P, "P3"), _raise(P, "Hero", 250), _fold(P, "P5"),
        _raise(P, "P6", 800), _call(P, "Hero", 550),
    ]
    assert start_range(range_spot(_hand(three_bet), "BB"), defend_path=path).absent == ()


def test_the_bb_check_after_a_limp_carries_the_groups_it_raises(tmp_path: Path):
    spot = range_spot(_hand(_HERO_LIMPS_BB_CHECKS, button="P3"), "BB")
    start = start_range(spot, defend_path=limp_file(tmp_path))
    assert [g.group for g in start.absent] == ["pairs", "suited_aces", "offsuit_aces"]
    assert start.absent[0].classes == ["AA", "KK", "QQ", "JJ", "TT"]


def test_the_opener_range_has_no_absent_groups():
    """Строка о группах — только у колла и чека защиты, у опена её нет."""
    actions = [
        _fold(P, "P5"), _fold(P, "P6"), _fold(P, "P1"), _raise(P, "P2", 250), _fold(P, "P3"),
        _call(P, "Hero", 150),
    ]
    assert start_range(range_spot(_hand(actions, button="P2"), "BTN")).absent == ()


def test_the_share_of_the_start_counts_the_added_combos_over_the_start(tmp_path: Path):
    """Добавленные комбо считаются сверх стартового: доля может быть больше единицы."""
    spot = equity_spot(
        _hand(BTN_OPENS_BB_CALLS), "BB", Street.FLOP, "AA", added="KJo, K9o",
        named=frozenset({"KJo", "K9o"}), defend_path=defend_file(tmp_path, call="AA"),
    )
    assert spot.share_of_start > 1.0


# --- ставка героя, которую соперник уравнял (ориентир сдачи, спека 2026-10-24, §13) ---


_COLD_CALL_FLOP_BOTH_CALL = [
    *_COLD_CALL, _check(F, "P5"), _check(F, "P6"), _bet(F, "Hero", 300), _call(F, "P5", 300),
    _call(F, "P6", 300),
]
_BB_RAISES_THE_CBET = [
    *BTN_OPENS_BB_CALLS[:-1], _raise(F, "P6", 480), _call(F, "Hero", 320),
]
_BB_DONKS = [*BTN_OPENS_BB_CALLS[:-3], _bet(F, "P6", 160), _call(F, "Hero", 160)]


@pytest.mark.parametrize(
    ("actions", "label", "street", "stat"),
    [
        (BTN_OPENS_BB_CALLS, "P6", F, FrequencyStat.FOLD_TO_CBET),
        (_THROUGH_RIVER, "P6", T, FrequencyStat.FOLD_TO_TURN_BARREL),
        (_THROUGH_RIVER, "P6", R, None),
        (_BB_RAISES_THE_CBET, "P6", F, None),
        (_BB_DONKS, "P6", F, None),
        (_COLD_CALL_FLOP_BOTH_CALL, "P6", F, None),
        (_COLD_CALL_FLOP_BOTH_CALL, "P5", F, None),
    ],
    ids=["cbet-called", "barrel-called", "river", "cbet-raised", "donk", "multiway-last",
         "multiway-first"],
)
def test_the_hero_bet_the_opponent_called_names_the_fold_stat(actions, label, street, stat):
    """С-бет героя, уравненный соперником, — `fold_to_cbet`; второй баррель —
    `fold_to_turn_barrel`; рейз, донк, ривер и больше двух в банке на ответе — `None`."""
    assert hero_bet_called(_hand(actions), label, street) is stat
