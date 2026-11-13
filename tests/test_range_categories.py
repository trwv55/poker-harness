"""Раскладка диапазона по категориям на борде (спека 2026-10-24, §4). Борды и диапазоны
синтетические."""

from __future__ import annotations

import pytest

from harness.analysis.charts import parse_range
from harness.analysis.opponent_range import live_combos
from harness.analysis.range_categories import (
    CategoryError,
    category_by_name,
    combo_category,
    expand_hands,
    kept_by_category,
    layout,
)
from harness.contracts import Range, RangeCategory, all_classes

_PAIRED = ["Jh", "8h", "8c"]
_HERO = ["Kd", "Tc"]


@pytest.mark.parametrize(
    "board", [
        _PAIRED, ["Th", "9h", "2c"], ["As", "Ks", "Qs"], ["Th", "9h", "2c", "5d"],
        ["Th", "9h", "2c", "5d", "3h"],
    ],
    ids=["paired", "two-tone", "mono", "turn", "river"],
)
def test_every_class_is_in_exactly_one_category_and_the_combos_add_up(board: list[str]):
    """Каждый класс с живыми комбо — ровно в одной категории; сумма комбо категорий (с
    долями) — комбо диапазона без карт героя и борда."""
    rng = Range(weights={cls: (0.5 if cls.endswith("o") else 1.0) for cls in all_classes()})
    dead = [*_HERO, *board]
    groups = layout(rng, board, dead)
    seen = [item.hand_class for group in groups for item in group.classes]
    assert len(seen) == len(set(seen))
    assert set(seen) == {
        cls for cls in all_classes() if live_combos(Range(weights={cls: 1.0}), dead) > 0
    }
    assert sum(group.combos for group in groups) == pytest.approx(live_combos(rng, dead))
    assert [group.category for group in groups] == sorted(
        (group.category for group in groups), key=list(RangeCategory).index
    )
    assert all(group.classes for group in groups)


def test_the_categories_follow_the_hand_class_of_the_core():
    """Трипс, оверпара, топ-пара по кикеру, слабая пара, гатшот, бэкдоры и оверкарты,
    ничего — по `hand_strength` и `draws` на спаренном флопе."""
    rng = parse_range("A8s, AA, AJo, J7o, 22, QTo, AKo, 43s")
    by_class = {
        item.hand_class: group.category
        for group in layout(rng, _PAIRED, [*_HERO, *_PAIRED])
        for item in group.classes
    }
    assert by_class == {
        "A8s": RangeCategory.TRIPS,
        "AA": RangeCategory.OVERPAIR,
        "AJo": RangeCategory.TOP_PAIR_STRONG_KICKER,
        "J7o": RangeCategory.TOP_PAIR_WEAK_KICKER,
        "22": RangeCategory.WEAK_PAIR,
        "QTo": RangeCategory.GUTSHOT,
        "AKo": RangeCategory.BACKDOORS_OVERCARDS,
        "43s": RangeCategory.NOTHING,
    }


def test_a_pair_with_a_draw_stays_a_pair_with_a_mark():
    """Готовая рука с дро — в категории руки, дро — пометкой: на T♥9♥7♣ у T8s топ-пара
    со слабым кикером и двусторонний у всех живых комбо."""
    board = ["Th", "9h", "7c"]
    assert combo_category(["Ts", "8s"], board) == (
        RangeCategory.TOP_PAIR_WEAK_KICKER, RangeCategory.OPEN_ENDED,
    )
    (group,) = layout(parse_range("T8s"), board, board)
    (item,) = group.classes
    assert group.category is RangeCategory.TOP_PAIR_WEAK_KICKER
    assert [(m.category, m.combos, m.of) for m in item.marks] == [
        (RangeCategory.OPEN_ENDED, 3, 3)
    ]


def test_a_suited_class_with_a_flush_draw_in_one_combo_is_marked_partly():
    """Класс целый: у A5s на J♥8♥8♣ флеш-дро только у A♥5♥ — класс в категории
    большинства комбо, флеш-дро — пометкой «частью комбо»; комбо класса — все живые."""
    (group,) = layout(parse_range("A5s"), _PAIRED, _PAIRED)
    (item,) = group.classes
    assert combo_category(["Ah", "5h"], _PAIRED)[0] is RangeCategory.FLUSH_DRAW
    assert group.category is RangeCategory.BACKDOORS_OVERCARDS
    assert item.combos == 4.0
    assert [(m.category, m.combos, m.of) for m in item.marks] == [
        (RangeCategory.FLUSH_DRAW, 1, 4)
    ]


def test_an_even_split_of_combos_takes_the_weaker_category():
    """Поровну живых комбо в двух категориях — класс в более слабой, и блокер героя на
    ничьей его не поднимает: у 65s на 7♥4♥K♣ без блокеров комбо-дро одно из четырёх,
    с мёртвыми 6♠ и 6♦ — одно из двух, и оба раза класс — «двусторонний»."""
    board = ["7h", "4h", "Kc"]
    (free,) = layout(parse_range("65s"), board, board)
    (tied,) = layout(parse_range("65s"), board, [*board, "6s", "6d"])
    assert free.category is tied.category is RangeCategory.OPEN_ENDED
    assert [(m.category, m.combos, m.of) for m in tied.classes[0].marks] == [
        (RangeCategory.COMBO_DRAW, 1, 2)
    ]


def test_a_monotone_flop_and_a_three_suited_turn_keep_offsuit_aces_out_of_the_flush_draw():
    """Q♠7♠2♠: у AKo флеш-дро у 6 из 12 комбо (с тузом или королём пик) — ничья, класс в
    «бэкдорах и старших картах» с пометкой; у AKs готовый флеш только у A♠K♠ — комбо в
    «стрит или флеш», класс — по большинству. На тёрне Q♠7♠2♥5♠ — так же у AKo."""
    flop = ["Qs", "7s", "2s"]
    by_class = {
        item.hand_class: (group.category, [(m.category, m.combos, m.of) for m in item.marks])
        for group in layout(parse_range("AKo, AKs"), flop, flop)
        for item in group.classes
    }
    assert by_class["AKo"] == (
        RangeCategory.BACKDOORS_OVERCARDS, [(RangeCategory.FLUSH_DRAW, 6, 12)]
    )
    assert combo_category(["As", "Ks"], flop)[0] is RangeCategory.STRAIGHT_OR_FLUSH
    assert by_class["AKs"] == (
        RangeCategory.BACKDOORS_OVERCARDS, [(RangeCategory.STRAIGHT_OR_FLUSH, 1, 4)]
    )
    turn = ["Qs", "7s", "2h", "5s"]
    (group,) = layout(parse_range("AKo"), turn, turn)
    assert group.category is RangeCategory.BACKDOORS_OVERCARDS
    assert [(m.category, m.combos, m.of) for m in group.classes[0].marks] == [
        (RangeCategory.FLUSH_DRAW, 6, 12)
    ]


# --- сужение категориями (§5) -----------------------------------------------------


def _groups() -> list:
    return layout(parse_range("A8s, K8s, QTo, Q9o, AKo"), _PAIRED, [*_HERO, *_PAIRED])


def test_a_category_name_expands_to_its_classes_minus_the_excluded():
    """Имя категории — её классы на этом борде, вперемешку с классами; «-<класс>»
    вычитает: и при категории, и отдельным элементом."""
    assert expand_hands("трипс, AKo", _groups()) == "AKo, A8s, K8s"
    assert expand_hands("гатшот -QTo, трипс", _groups()) == "A8s, K8s, Q9o"
    assert expand_hands("трипс, гатшот, -K8s", _groups()) == "A8s, QTo, Q9o"
    assert expand_hands("AKo, QTo", _groups()) == "AKo, QTo"


@pytest.mark.parametrize(
    "name", ["Трипс", "  ТРИПС ", "Бэкдоры  и старшие карты", "Топ-пара с сильным кикером",
             "топ – пара с сильным кикером", "комбо—дро"],
)
def test_a_category_name_is_read_regardless_of_case_spaces_and_dashes(name: str):
    assert category_by_name(name) is not None


@pytest.mark.parametrize(
    ("hands", "groups", "match"),
    [
        ("оверпара", "board", "нет ни одного класса"),
        ("трипс -AKo", "board", "среди названного нет: AKo"),
        ("трипс -A8s -K8s", "board", "не осталось ни одной руки"),
        ("трипс", None, "только после флопа"),
        ("сильные руки", "board", "ни класс, ни категория"),
    ],
)
def test_a_category_narrowing_that_cannot_be_read_is_refused(hands, groups, match):
    with pytest.raises(CategoryError, match=match):
        expand_hands(hands, _groups() if groups == "board" else None)


def test_the_equity_output_sums_up_what_is_kept():
    """Сводка: от категории — целиком, частью (с убранными классами) или ничего;
    комбо оставленного из исходного без добавленных."""
    groups = _groups()
    kept = kept_by_category(groups, parse_range("A8s, K8s, QTo"))
    assert [(k.category, k.kept_classes, k.classes, k.removed) for k in kept] == [
        (RangeCategory.TRIPS, 2, 2, []),
        (RangeCategory.GUTSHOT, 1, 2, ["Q9o"]),
        (RangeCategory.BACKDOORS_OVERCARDS, 0, 1, ["AKo"]),
    ]
    assert sum(k.combos for k in kept) == pytest.approx(sum(g.combos for g in groups))


def test_the_river_layout_has_no_draw_categories():
    """Ривер (спека 2026-10-24, §14.2): дро, бэкдоров и оверкарт нет — категории только
    готовые руки и «ничего»; пометок дро у готовых рук тоже нет."""
    board = ["Th", "9h", "2c", "5d", "3h"]
    rng = Range(weights=dict.fromkeys(all_classes(), 1.0))
    groups = layout(rng, board, [*_HERO, *board])
    drawing = {
        RangeCategory.COMBO_DRAW, RangeCategory.FLUSH_DRAW, RangeCategory.OPEN_ENDED,
        RangeCategory.DOUBLE_GUTSHOT, RangeCategory.GUTSHOT, RangeCategory.BACKDOORS_OVERCARDS,
    }
    found = {group.category for group in groups}
    assert RangeCategory.NOTHING in found and not found & drawing
    assert not any(
        mark.category in drawing for group in groups for item in group.classes for mark in item.marks
    )
