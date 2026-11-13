"""Эквити героя против класса и категории — точный перебор (спека 2026-10-24, §14.3).

Эталон считается руками: у соперника KK против AA героя на сухом борде без пар и без
стритовых и флешевых дро выигрывает только король (и не выигрывает при тузе вместе с
ним). Борды и руки синтетические.
"""

from __future__ import annotations

import pytest

from harness.analysis.range_categories import layout
from harness.analysis.tools.exact_equity import class_equities, group_equity
from harness.contracts import CategoryClass, CategoryGroup, Range, RangeCategory

_AA = ("Ah", "Ad")
_DRY_FLOP = ["2c", "7s", "Jd"]


@pytest.mark.parametrize(
    ("board", "expected"),
    [
        # 45 карт, 990 пар доборов. KK выигрывает при хотя бы одном короле (из двух) и
        # без туза (из двух): K+K — 1, K+другая (41) — 2·41; AA — остальные 907.
        (_DRY_FLOP, 907 / 990),
        # 44 карты ривера, KK выигрывает только королём (два).
        ([*_DRY_FLOP, "3h"], 42 / 44),
        ([*_DRY_FLOP, "3h", "9c"], 1.0),
        ([*_DRY_FLOP, "3h", "Kc"], 0.0),
        (["Ts", "9h", "8c", "7d", "6s"], 0.5),  # стрит на борде у обоих
    ],
    ids=["flop", "turn", "river-win", "river-loss", "river-tie"],
)
def test_class_equity_is_exact_on_every_street(board: list[str], expected: float):
    """Перебор совпадает с эталоном до последнего знака; повтор — то же число, сида нет."""
    first = class_equities(_AA, board)
    assert first["KK"] == pytest.approx(expected, abs=1e-12)
    assert class_equities(_AA, board) == first


def test_a_class_without_live_combos_has_no_equity():
    """AA героя и туз на борде: у класса AA живых комбо нет — эквити против него нет."""
    found = class_equities(("Ah", "Ad"), ["As", "7s", "2d"])
    assert "AA" not in found
    assert "KK" in found


def test_a_category_equity_is_weighted_by_the_live_combos_with_the_chart_share():
    """Эквити категории — среднее эквити классов, взвешенное живыми комбо с долей: доля
    чарта меньше единицы меняет вес и число."""
    board = _DRY_FLOP
    by_class = class_equities(_AA, board)
    assert by_class["KJs"] != pytest.approx(by_class["JTs"])
    full = Range(weights={"KJs": 1.0, "JTs": 1.0})
    half = Range(weights={"KJs": 1.0, "JTs": 0.25})
    dead = [*_AA, *board]
    top = RangeCategory.TOP_PAIR_STRONG_KICKER
    (group_full,) = [g for g in layout(full, board, dead) if g.category is top]
    (group_half,) = [g for g in layout(half, board, dead) if g.category is top]
    # У обоих классов по 3 живых комбо (Jd на борде); у JTs с долей 0.25 — 0.75.
    assert group_equity(group_full, by_class) == pytest.approx(
        (3 * by_class["KJs"] + 3 * by_class["JTs"]) / 6
    )
    assert group_equity(group_half, by_class) == pytest.approx(
        (3 * by_class["KJs"] + 0.75 * by_class["JTs"]) / 3.75
    )
    assert group_equity(group_half, by_class) != pytest.approx(group_equity(group_full, by_class))


def test_a_category_equity_takes_the_combos_of_each_class_as_given():
    """Вес — `CategoryClass.combos` как есть, а не число классов."""
    group = CategoryGroup(category=RangeCategory.SET, classes=[
        CategoryClass(hand_class="77", weight=1.0, combos=3.0),
        CategoryClass(hand_class="22", weight=0.5, combos=1.0),
    ])
    assert group_equity(group, {"77": 0.2, "22": 0.6}) == pytest.approx((0.6 + 0.6) / 4)
