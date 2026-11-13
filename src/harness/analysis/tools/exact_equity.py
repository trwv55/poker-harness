"""Точная эквити руки героя против каждого комбо и каждого класса — перебором доборов.

**Перебор, а не Монте-Карло** (спека 2026-10-24, §14.3). Против конкретного комбо
соперника на флопе — все пары доборов, на тёрне — все карты ривера, на ривере
добора нет и исход — 1, 0 или ½ (`equity_by_combo`;
`test_the_turn_enumerates_every_river_card`,
`test_the_flop_enumerates_every_pair_of_run_outs`). Сида нет, число одно на вход
(`test_class_equity_is_exact_on_every_street`). Ничья — половина, как в
`turn_flop_call`.

**Эквити класса** (`class_equities`) — среднее эквити его живых комбо (без карт
героя и борда); класс без живых комбо в ответ не входит. **Эквити категории**
(`group_equity`) — среднее эквити её классов, взвешенное живыми комбо с долей
диапазона (`CategoryClass.combos`;
`test_a_category_equity_is_weighted_by_the_live_combos_with_the_chart_share`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from math import comb

import eval7

from harness.analysis.tools.equity import _FULL_DECK, combos_of_class
from harness.contracts import CategoryGroup, all_classes

__all__ = ["class_equities", "equity_by_combo", "group_equity"]

_FULL_BOARD = 5


def equity_by_combo(
    hero_cards: list[eval7.Card],
    board_cards: list[eval7.Card],
    cards: list[eval7.Card],
    to_come: int,
) -> dict[tuple[int, int], float]:
    """Доля героя во вскрытии против каждого комбо — перебором всех доборов.

    Ключ — пара индексов в `cards` (i < j). Ничья считается половиной.

    Знаменатель один на все комбо: доборы, не задевающие двух карт комбо, — это
    сочетания из оставшихся `size - 2` карт, и их число от самого комбо не
    зависит (`test_every_combo_is_dealt_the_same_number_of_run_outs`).

    Цикл идёт по доборам, а не по комбо: рука героя на добранном борде считается
    тогда один раз на добор, а не заново на каждое комбо соперника.
    """
    size = len(cards)
    won = [[0.0] * size for _ in range(size)]
    evaluate = eval7.evaluate
    for run_out in combinations(range(size), to_come):
        taken = set(run_out)
        full_board = board_cards + [cards[x] for x in run_out]
        hero_score = evaluate(hero_cards + full_board)
        for i in range(size):
            if i in taken:
                continue
            row = won[i]
            with_first = full_board + [cards[i]]
            for j in range(i + 1, size):
                if j in taken:
                    continue
                score = evaluate(with_first + [cards[j]])
                if score < hero_score:
                    row[j] += 1.0
                elif score == hero_score:
                    row[j] += 0.5
    run_outs = comb(size - 2, to_come)
    return {
        (i, j): won[i][j] / run_outs for i in range(size) for j in range(i + 1, size)
    }


def class_equities(hero: Sequence[str], board: Sequence[str]) -> dict[str, float]:
    """Эквити героя против каждого класса с живыми комбо — среднее по ним.

    `board` — три, четыре или пять карт; карты героя и борда — из колоды в записи
    `equity` ("Ah", "Td") и не пересекаются. Иначе `ValueError`."""
    if len(board) not in (3, 4, _FULL_BOARD):
        raise ValueError(f"борд — 3, 4 или 5 карт, получено {len(board)}")
    known = [*hero, *board]
    unknown = [card for card in known if card not in _FULL_DECK]
    if unknown:
        raise ValueError(f"не карты колоды: {unknown}")
    if len(set(known)) != len(known):
        raise ValueError(f"карта встречается дважды среди карт героя и борда: {known}")
    rest = [card for card in _FULL_DECK if card not in set(known)]
    index = {card: i for i, card in enumerate(rest)}
    by_combo = equity_by_combo(
        [eval7.Card(card) for card in hero],
        [eval7.Card(card) for card in board],
        [eval7.Card(card) for card in rest],
        _FULL_BOARD - len(board),
    )
    found: dict[str, float] = {}
    for cls in all_classes():
        live: list[float] = []
        for a, b in combos_of_class(cls):
            if a in index and b in index:
                i, j = sorted((index[a], index[b]))
                live.append(by_combo[(i, j)])
        if live:
            found[cls] = sum(live) / len(live)
    return found


def group_equity(group: CategoryGroup, by_class: Mapping[str, float]) -> float:
    """Эквити категории — среднее эквити её классов, взвешенное `CategoryClass.combos`
    (живые комбо с долей диапазона). Класса без эквити в `by_class` быть не должно —
    `KeyError`."""
    total = sum(item.combos for item in group.classes)
    return sum(by_class[item.hand_class] * item.combos for item in group.classes) / total
