"""Раскладка диапазона по категориям на борде и сужение категориями (спека 2026-10-24, §4, §5).

**Категория комбо** (`combo_category`) — из того, что уже считает ядро: готовая рука —
`hand_strength` (`HandCategory`), дро — `draw_of`, когда готовой руки нет или она слабее
пары (рука «на борде» — тоже нет), бэкдоры и оверкарты — `backdoors`, `overcards`, иначе
«ничего». У готовой руки дро — пометка, а не категория
(`test_a_pair_with_a_draw_stays_a_pair_with_a_mark`).

**Категория класса** (`layout`) — та, в которую попадает большинство его живых комбо;
при равенстве — более слабая, так что блокер героя на ничьей класс выше не поднимает
(`test_an_even_split_of_combos_takes_the_weaker_category`). Класс целый: одномастный
класс, у части комбо которого флеш-дро, не делится, а несёт пометку с числом комбо
(`test_a_suited_class_with_a_flush_draw_in_one_combo_is_marked_partly`,
`test_a_monotone_flop_and_a_three_suited_turn_keep_offsuit_aces_out_of_the_flush_draw`). Каждый класс
с живыми комбо — ровно в одной категории, и сумма комбо по категориям равна комбо
диапазона (`test_every_class_is_in_exactly_one_category_and_the_combos_add_up`).

**Сужение категориями** (`expand_hands`): имя категории разворачивается в её классы на
этом борде, «-<класс>» вычитает
(`test_a_category_name_expands_to_its_classes_minus_the_excluded`).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Sequence
from functools import lru_cache

from harness.analysis.charts import NotationError, parse_range
from harness.analysis.tools.draws import backdoors, draw_of, overcards
from harness.analysis.tools.equity import combos_of_class
from harness.analysis.tools.hand_class import hand_strength
from harness.contracts import (
    RANGE_CATEGORY_NAMES,
    CategoryClass,
    CategoryGroup,
    CategoryKept,
    CategoryMark,
    DrawKind,
    HandCategory,
    Range,
    RangeCategory,
    all_classes,
)

__all__ = [
    "CategoryError",
    "category_by_name",
    "combo_category",
    "expand_hands",
    "kept_by_category",
    "layout",
]

_MADE: dict[HandCategory, RangeCategory] = {
    HandCategory.STRAIGHT_FLUSH: RangeCategory.FULL_HOUSE_PLUS,
    HandCategory.QUADS: RangeCategory.FULL_HOUSE_PLUS,
    HandCategory.FULL_HOUSE: RangeCategory.FULL_HOUSE_PLUS,
    HandCategory.FLUSH: RangeCategory.STRAIGHT_OR_FLUSH,
    HandCategory.STRAIGHT: RangeCategory.STRAIGHT_OR_FLUSH,
    HandCategory.WEAK_FLUSH: RangeCategory.STRAIGHT_OR_FLUSH,
    HandCategory.WEAK_STRAIGHT: RangeCategory.STRAIGHT_OR_FLUSH,
    HandCategory.SET: RangeCategory.SET,
    HandCategory.TRIPS: RangeCategory.TRIPS,
    HandCategory.TWO_PAIR: RangeCategory.TWO_PAIR,
    HandCategory.OVERPAIR: RangeCategory.OVERPAIR,
    HandCategory.TOP_PAIR_STRONG_KICKER: RangeCategory.TOP_PAIR_STRONG_KICKER,
    HandCategory.TOP_PAIR_WEAK_KICKER: RangeCategory.TOP_PAIR_WEAK_KICKER,
    HandCategory.MIDDLE_PAIR: RangeCategory.MIDDLE_PAIR,
    HandCategory.WEAK_PAIR: RangeCategory.WEAK_PAIR,
}
_ORDER: dict[RangeCategory, int] = {category: i for i, category in enumerate(RangeCategory)}
_STRAIGHT_DRAWS: tuple[tuple[DrawKind, RangeCategory], ...] = (
    (DrawKind.OPEN_ENDED, RangeCategory.OPEN_ENDED),
    (DrawKind.DOUBLE_GUTSHOT, RangeCategory.DOUBLE_GUTSHOT),
    (DrawKind.GUTSHOT, RangeCategory.GUTSHOT),
)


class CategoryError(ValueError):
    """Сужение категориями не разобрано — причина словами для модели."""


@lru_cache(maxsize=1 << 17)
def _combo_category(
    combo: tuple[str, str], board: tuple[str, ...]
) -> tuple[RangeCategory, RangeCategory | None]:
    made = _MADE.get(hand_strength(combo, board).category)
    found = draw_of(combo, board)
    drawing: RangeCategory | None = None
    if found is not None:
        flush = DrawKind.FLUSH in found.kinds
        straight = next((cat for kind, cat in _STRAIGHT_DRAWS if kind in found.kinds), None)
        if flush and straight is not None:
            drawing = RangeCategory.COMBO_DRAW
        elif flush:
            drawing = RangeCategory.FLUSH_DRAW
        else:
            drawing = straight
    if made is not None:
        return made, drawing
    if drawing is not None:
        return drawing, None
    if backdoors(combo, board) or overcards(combo, board) is not None:
        return RangeCategory.BACKDOORS_OVERCARDS, None
    return RangeCategory.NOTHING, None


def combo_category(
    combo: Sequence[str], board: Sequence[str]
) -> tuple[RangeCategory, RangeCategory | None]:
    """Категория комбо на борде (флоп или тёрн) и дро готовой руки (пометка); у
    категории без готовой руки второе — `None`."""
    first, second = combo
    return _combo_category((first, second), tuple(board))


def layout(rng: Range, board: Sequence[str], dead: Collection[str]) -> list[CategoryGroup]:
    """Классы диапазона с ненулевой долей по категориям на борде, в порядке
    `RangeCategory`, классы — в порядке `all_classes()`. Комбо класса — его живые комбо
    (без `dead` — карт героя и борда) с долей диапазона; класс без живых комбо в
    раскладку не входит. Пустых категорий нет."""
    blocked = set(dead) | set(board)
    groups: dict[RangeCategory, list[CategoryClass]] = {}
    for cls in all_classes():
        weight = rng.weight(cls)
        if weight <= 0.0:
            continue
        live = [c for c in combos_of_class(cls) if c[0] not in blocked and c[1] not in blocked]
        if not live:
            continue
        found = [combo_category(combo, board) for combo in live]
        counts = Counter(category for category, _ in found)
        main = min(counts, key=lambda category: (-counts[category], -_ORDER[category]))
        marked: Counter[RangeCategory] = Counter()
        for category, drawing in found:
            extra = {category, drawing} - {main, None}
            marked.update(c for c in extra if c is not None)
        marks = [
            CategoryMark(category=category, combos=marked[category], of=len(live))
            for category in sorted(marked, key=_ORDER.__getitem__)
        ]
        groups.setdefault(main, []).append(
            CategoryClass(hand_class=cls, weight=weight, combos=weight * len(live), marks=marks)
        )
    return [
        CategoryGroup(category=category, classes=groups[category])
        for category in RangeCategory
        if category in groups
    ]


def kept_by_category(
    groups: Sequence[CategoryGroup], narrowed: Range
) -> list[CategoryKept]:
    """Сколько от каждой категории исходной раскладки оставлено в сужении: классы и комбо
    (с долями раскладки), убранные классы (`test_the_equity_output_sums_up_what_is_kept`)."""
    kept: list[CategoryKept] = []
    for group in groups:
        left = [item for item in group.classes if narrowed.weight(item.hand_class) > 0.0]
        kept.append(CategoryKept(
            category=group.category,
            classes=len(group.classes),
            kept_classes=len(left),
            combos=group.combos,
            kept_combos=sum(item.combos for item in left),
            removed=[
                item.hand_class for item in group.classes
                if narrowed.weight(item.hand_class) <= 0.0
            ],
        ))
    return kept


def _normal(text: str) -> str:
    """Имя без различий в регистре, дефисах и тире, пробелах; «ё» читается как «е»."""
    text = text.casefold().replace("ё", "е")
    text = re.sub(r"\s*[-‐‑‒–—]\s*", "-", text)
    return re.sub(r"\s+", " ", text).strip()


_BY_NAME: dict[str, RangeCategory] = {
    _normal(name): category for category, name in RANGE_CATEGORY_NAMES.items()
}


def category_by_name(text: str) -> RangeCategory | None:
    """Категория по имени из выхода (`RANGE_CATEGORY_NAMES`) — без различий в регистре,
    виде дефиса и тире и пробелах; иначе `None`
    (`test_a_category_name_is_read_regardless_of_case_spaces_and_dashes`)."""
    return _BY_NAME.get(_normal(text))


# Вычитание класса: «гатшот -T7o», отдельный элемент «-T7o». Минус — перед записью
# класса и после пробела или в начале элемента: дефис внутри «A2s-A5s» — диапазон.
_EXCLUDE_RE = re.compile(r"(?:^|\s)-\s*([2-9TJQKA][^\s,]*)")


def _classes_of(token: str) -> list[str]:
    try:
        return list(parse_range(token).weights)
    except (NotationError, ValueError) as exc:
        names = "; ".join(RANGE_CATEGORY_NAMES.values())
        raise CategoryError(
            f"запись рук не разобрана: «{token}» — ни класс, ни категория ({names})"
        ) from exc


def expand_hands(hands: str, groups: Sequence[CategoryGroup] | None) -> str:
    """`hands` с именами категорий — классами через запятую. Категория разворачивается в
    свои классы из `groups` (раскладка исходного диапазона на этом борде), «-<классы>»
    вычитает их из всего названного. Вычтенный класс, которого среди названного нет, —
    отказ, как и пустой итог. `groups is None` — категорий на этой улице нет (до флопа и
    на ривере): имя категории — отказ. Без категорий и вычитаний строка уходит как есть."""
    if ":" in hands:
        return hands  # веса — отказ `narrow`, как без категорий
    by_category = {group.category: group for group in groups or ()}
    named: list[str] = []
    excluded: list[str] = []
    used_categories = False
    for raw in hands.split(","):
        token = raw.strip()
        if not token:
            continue
        for match in _EXCLUDE_RE.finditer(token):
            excluded += _classes_of(match.group(1))
        head = _EXCLUDE_RE.sub("", token).strip()
        if not head:
            continue
        category = category_by_name(head)
        if category is None:
            named += _classes_of(head)
            continue
        used_categories = True
        if groups is None:
            raise CategoryError(
                "категории — только на флопе и тёрне; на этой улице назовите классы"
            )
        if category not in by_category:
            raise CategoryError(
                f"в категории «{RANGE_CATEGORY_NAMES[category]}» на этом борде нет ни "
                "одного класса диапазона"
            )
        named += [item.hand_class for item in by_category[category].classes]
    if not used_categories and not excluded:
        return hands
    missing = [cls for cls in all_classes() if cls in excluded and cls not in named]
    if missing:
        raise CategoryError(
            "вычтены классы, которых среди названного нет: " + ", ".join(missing)
        )
    kept = [cls for cls in all_classes() if cls in named and cls not in excluded]
    if not kept:
        raise CategoryError("после вычитания не осталось ни одной руки")
    return ", ".join(kept)
