"""Дро, бэкдоры, оверкарты и ценность на вскрытии (`analysis.tools.draws`).

Все руки синтетические; рука §2 спеки — та же форма: у героя 5c 3c, флоп Qh 4d 2d,
тёрн 7s, ривер Qd. Спека — `docs/superpowers/specs/2026-10-09-postflop-line-design.md`.
"""

from fractions import Fraction
from itertools import combinations_with_replacement
from math import comb

import pytest

from harness.analysis.tools.draws import (
    BACKDOOR_FLUSH_APPROX,
    BACKDOOR_STRAIGHT_APPROX,
    MIN_DRAW_OUTS,
    OVERCARDS_APPROX_BY_RIVER,
    OVERCARDS_APPROX_NEXT,
    backdoors,
    draw_missed,
    draw_of,
    overcards,
    showdown_value,
)
from harness.contracts import RANKS, Backdoor, Draw, DrawKind

_HERO = ("5c", "3c")
_FLOP = ["Qh", "4d", "2d"]
_TURN = [*_FLOP, "7s"]
_RIVER = [*_TURN, "Qd"]


def _draw(hero: str, board: str) -> Draw | None:
    return draw_of(tuple(hero.split()), board.split())


def _exact(value: float, fraction: Fraction) -> bool:
    return abs(Fraction(value) - fraction) < Fraction(1, 10**12)


# --- рука §2 ---------------------------------------------------------------------------


def test_the_example_flop_is_an_open_ender_to_the_ace_and_the_six():
    draw = draw_of(_HERO, _FLOP)
    assert draw is not None
    assert draw.kinds == [DrawKind.OPEN_ENDED]
    assert draw.out_ranks == ["A", "6"]
    assert draw.outs == ["As", "Ah", "Ad", "Ac", "6s", "6h", "6d", "6c"]
    assert draw.unseen == 52 - 2 - 3 == 47
    assert draw.hit_next == 8 / 47
    # 1 − C(39, 2) / C(47, 2) = 1 − 741 / 1081.
    assert _exact(draw.hit_by_river or 0.0, Fraction(340, 1081))
    assert _exact(draw.hit_by_river or 0.0, 1 - Fraction(comb(39, 2), comb(47, 2)))
    assert round(draw.hit_by_river or 0.0, 4) == 0.3145


def test_the_example_turn_keeps_the_same_eight_outs_to_one_card():
    draw = draw_of(_HERO, _TURN)
    assert draw is not None
    assert draw.kinds == [DrawKind.OPEN_ENDED]
    assert draw.out_ranks == ["A", "6"]
    assert len(draw.outs) == 8
    assert draw.unseen == 46
    assert draw.hit_next == 8 / 46
    assert draw.hit_by_river is None


def test_there_is_no_draw_on_the_river():
    assert draw_of(_HERO, _RIVER) is None


def test_the_example_river_misses_the_draw():
    assert draw_missed(_HERO, _RIVER) is True


def test_a_river_out_closes_the_draw():
    assert draw_missed(_HERO, [*_TURN, "6h"]) is False


def test_no_turn_draw_means_nothing_missed():
    assert draw_missed(("Kc", "Kd"), ["Qh", "4d", "2d", "7s", "Qd"]) is False


def test_the_missed_draw_is_a_river_label_only():
    assert draw_missed(_HERO, _TURN) is False


def test_showdown_value_of_the_example_river_is_zero():
    value = showdown_value(_HERO, _RIVER)
    assert value is not None
    assert (value.wins, value.ties, value.losses) == (0, 8, 982)
    assert value.wins + value.ties + value.losses == comb(45, 2) == 990
    # Делят банк только 5-3: шесть разномастных и две одномастные (5d3d — флеш).
    assert value.ties_with == ["53o", "53s"]


def test_showdown_value_of_the_nuts_has_no_losses():
    value = showdown_value(("Ts", "9s"), ["As", "Ks", "Qs", "Js", "2h"])
    assert value is not None
    assert (value.wins, value.ties, value.losses) == (990, 0, 0)
    assert value.ties_with == []


def test_showdown_value_is_a_river_label_only():
    assert showdown_value(_HERO, _TURN) is None


# --- виды стрит-дро ----------------------------------------------------------------------


def test_double_gutshot():
    # 5 7 8 9 J: шестёрка даёт 5-9, десятка — 7-J; цепочки из четырёх подряд нет.
    draw = _draw("9h 7d", "Jc 5s 8h 2c")
    assert draw is not None
    assert draw.kinds == [DrawKind.DOUBLE_GUTSHOT]
    assert draw.out_ranks == ["T", "6"]
    assert len(draw.outs) == 8


def test_gutshot():
    draw = _draw("9h 7d", "Jc 8s 2h")
    assert draw is not None
    assert draw.kinds == [DrawKind.GUTSHOT]
    assert draw.out_ranks == ["T"]
    assert draw.hit_next == 4 / 47


def test_wheel_gutshot_counts_the_ace_low():
    draw = _draw("Ah 5d", "3c 4s Kh")
    assert draw is not None
    assert draw.kinds == [DrawKind.GUTSHOT]
    assert draw.out_ranks == ["2"]


def test_ace_two_three_four_is_a_one_sided_gutshot():
    draw = _draw("Ah 2d", "3c 4s Kh")
    assert draw is not None
    assert draw.kinds == [DrawKind.GUTSHOT]
    assert draw.out_ranks == ["5"]


def test_jack_queen_king_ace_is_a_one_sided_gutshot():
    draw = _draw("Ah Kd", "Qc Js 3h")
    assert draw is not None
    assert draw.kinds == [DrawKind.GUTSHOT]
    assert draw.out_ranks == ["T"]


def test_a_straight_completed_on_the_board_alone_is_not_an_out():
    # 5 6 7 8 на борде, у героя T: девятка даёт ему 6-T, четвёрка — стрит 4-8
    # из одних карт борда.
    draw = _draw("Th 2d", "5c 6s 7h 8d")
    assert draw is not None
    assert draw.kinds == [DrawKind.GUTSHOT]
    assert draw.out_ranks == ["9"]


# --- флеш и комбо ------------------------------------------------------------------------


def test_flush_draw_counts_nine_outs():
    draw = _draw("Ah 7h", "Kh 9h 2c")
    assert draw is not None
    assert draw.kinds == [DrawKind.FLUSH]
    assert draw.out_ranks == []
    assert len(draw.outs) == 9
    assert all(card[1] == "h" for card in draw.outs)


def test_four_suited_on_the_board_without_a_hero_card_is_not_a_draw():
    assert _draw("Ac Qd", "Kh 9h 5h 2h") is None


def test_combo_draw_counts_each_out_once():
    # Девять червей и восемь карт на стрит; Th и 5h — в обоих счетах.
    draw = _draw("9h 8h", "7h 6c 2h")
    assert draw is not None
    assert draw.kinds == [DrawKind.FLUSH, DrawKind.OPEN_ENDED]
    assert draw.out_ranks == ["T", "5"]
    assert len(draw.outs) == len(set(draw.outs)) == 9 + 8 - 2 == 15
    assert draw.hit_next == 15 / 47


def test_fewer_outs_than_the_threshold_is_no_draw():
    # У героя готовый флеш: десятки дают стрит, но лучшую руку улучшает только
    # Th (роял-флеш) — один аут.
    assert MIN_DRAW_OUTS == 4
    assert _draw("Ah Kh", "Qh Jh 2h") is None


# --- бэкдоры -----------------------------------------------------------------------------


def test_backdoor_flush():
    assert backdoors(("Ah", "7h"), ["Kh", "2c", "2d"]) == [_flush_backdoor()]


def test_backdoor_flush_needs_a_hero_card():
    assert backdoors(("Ac", "7d"), ["Kh", "9h", "2h"]) == []


def test_no_backdoor_flush_with_a_flush_draw():
    assert backdoors(("Ah", "7h"), ["Kh", "9h", "2c"]) == []


@pytest.mark.parametrize(
    ("hero", "board", "variants"),
    [
        # 7-8-9: 5-6, 6-T, T-J.
        (("7c", "8d"), ["9s", "2h", "Kd"], 3),
        # 7-8-T: 6-9, 9-J.
        (("7c", "8d"), ["Ts", "2h", "Kd"], 2),
        # Q-K-A: J-T.
        (("Qc", "Kd"), ["As", "2h", "7d"], 1),
    ],
)
def test_backdoor_straight_variants(hero, board, variants):
    assert backdoors(hero, board) == [_straight_backdoor(variants)]


def test_three_ranks_all_on_the_board_are_not_a_backdoor():
    # 7 8 9 — целиком на борде: стрит из них собрался бы без карт героя. Остаётся
    # только 8-9-Q с дамой героя: T-J.
    assert backdoors(("Kc", "Qd"), ["7s", "8h", "9d"]) == [_straight_backdoor(1)]


def test_no_backdoor_straight_with_a_straight_draw():
    # Гатшот на десятку; окно 5-9 с тремя рангами в бэкдор не идёт.
    assert backdoors(("9h", "7d"), ["Jc", "8s", "2h"]) == []


def test_no_backdoor_straight_with_a_made_straight():
    assert backdoors(("9h", "8d"), ["7c", "6s", "5d"]) == []


def test_both_backdoors_at_once():
    assert backdoors(("7h", "8h"), ["9c", "2h", "Kd"]) == [
        _flush_backdoor(),
        _straight_backdoor(3),
    ]


def test_backdoors_are_a_flop_label_only():
    assert backdoors(("7h", "8h"), ["9c", "2h", "Kd", "3s"]) == []
    assert backdoors(("7h", "8h"), ["9c", "2h", "Kd", "3s", "3d"]) == []


def test_backdoor_constants_round_the_exact_odds():
    assert abs(BACKDOOR_FLUSH_APPROX - 10 / 47 * 9 / 46) < 0.005
    for variants, approx in BACKDOOR_STRAIGHT_APPROX.items():
        assert abs(approx - variants * 2 * 4 / 47 * 4 / 46) < 0.001


def _flush_backdoor() -> Backdoor:
    return Backdoor(kind="flush", variants=None, approx=0.04)


def _straight_backdoor(variants: int) -> Backdoor:
    approx = {1: 0.015, 2: 0.03, 3: 0.045}[variants]
    return Backdoor(kind="straight", variants=variants, approx=approx)


# --- оверкарты ---------------------------------------------------------------------------


def test_two_overcards_on_the_flop():
    result = overcards(("Kd", "Ah"), ["Qc", "7h", "2d"])
    assert result is not None
    assert result.cards == ["Ah", "Kd"]
    assert (result.approx_by_river, result.approx_next) == (0.12, 0.065)


def test_one_overcard_on_the_flop():
    result = overcards(("Ah", "5d"), ["Qc", "7h", "2d"])
    assert result is not None
    assert result.cards == ["Ah"]
    assert (result.approx_by_river, result.approx_next) == (0.065, 0.03)


def test_overcards_on_the_turn_keep_the_one_card_figure():
    two = overcards(("Ah", "Kd"), ["Qc", "7h", "2d", "3s"])
    one = overcards(("Ah", "5d"), ["Qc", "7h", "2d", "3s"])
    assert two is not None and one is not None
    assert (two.approx_by_river, two.approx_next) == (None, 0.065)
    assert (one.approx_by_river, one.approx_next) == (None, 0.03)


def test_overcards_are_owner_constants():
    assert OVERCARDS_APPROX_BY_RIVER == {2: 0.12, 1: 0.065}
    assert OVERCARDS_APPROX_NEXT == {2: 0.065, 1: 0.03}


def test_no_overcards_on_the_river():
    assert overcards(("Ah", "Kd"), ["Qc", "7h", "2d", "3s", "4h"]) is None


def test_no_overcards_with_a_hero_pair():
    assert overcards(("Ah", "Qd"), ["Qc", "7h", "2d"]) is None
    assert overcards(("Ah", "Ad"), ["Qc", "7h", "2d"]) is None


def test_overcards_over_a_pair_on_the_board():
    result = overcards(("Ah", "Kd"), ["Qc", "Qh", "2d"])
    assert result is not None
    assert result.cards == ["Ah", "Kd"]


def test_no_overcards_with_a_made_straight():
    assert overcards(("Ah", "Kd"), ["Qc", "Jh", "Td"]) is None


def test_no_overcards_below_the_top_board_card():
    assert overcards(("5c", "3c"), _FLOP) is None


def test_backdoors_and_overcards_are_not_outs():
    # Бэкдор флеша, бэкдор стрита (Q-K-A) и две оверкарты, а дро нет.
    hero, board = ("Ah", "Kh"), ["Qh", "7c", "2d"]
    assert draw_of(hero, board) is None
    assert len(backdoors(hero, board)) == 2
    assert overcards(hero, board) is not None


def test_the_example_flop_has_no_backdoors_and_no_overcards():
    # §2.1: две трефы у героя при флопе без треф, бубны на флопе без бубны у героя;
    # стрит-дро уже есть; карт старше дамы у героя нет.
    assert backdoors(_HERO, _FLOP) == []
    assert overcards(_HERO, _FLOP) is None


# --- вход --------------------------------------------------------------------------------


@pytest.mark.parametrize("call", [draw_of, draw_missed, backdoors, overcards, showdown_value])
def test_bad_input_is_refused(call):
    with pytest.raises(ValueError):
        call(("5c", "5c"), _FLOP)
    with pytest.raises(ValueError):
        call(_HERO, ["Qh", "4d"])


def test_backdoor_straight_variants_never_leave_the_owner_table():
    # Таблица владельца знает n = 1, 2, 3. Масти бэкдору стрита не важны, поэтому
    # перебираются все сочетания рангов флопа и карт героя без пяти одинаковых.
    seen: set[int] = set()
    for hero_ranks in combinations_with_replacement(RANKS, 2):
        for board_ranks in combinations_with_replacement(RANKS, 3):
            ranks = [*hero_ranks, *board_ranks]
            if max(ranks.count(rank) for rank in ranks) > 4:
                continue
            cards = _distinct_cards(ranks)
            for backdoor in backdoors(tuple(cards[:2]), cards[2:]):
                if backdoor.kind == "straight":
                    seen.add(backdoor.variants or 0)
    assert seen == set(BACKDOOR_STRAIGHT_APPROX) == {1, 2, 3}


def _distinct_cards(ranks: list[str]) -> list[str]:
    """Карты этих рангов без повторов: масти по кругу, отдельно для каждого ранга."""
    used: dict[str, int] = {}
    cards = []
    for rank in ranks:
        cards.append(rank + "shdc"[used.get(rank, 0)])
        used[rank] = used.get(rank, 0) + 1
    return cards
