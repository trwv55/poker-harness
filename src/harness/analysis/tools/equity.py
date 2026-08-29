"""Эквити против диапазона: собственный Монте-Карло поверх `eval7.evaluate`.

`eval7` не даёт готового детерминированного расчёта эквити — проверено на живой
библиотеке (0.1.11, Python 3.12), а не по памяти:
- `py_hand_vs_range_monte_carlo` недетерминирован между вызовами (два одинаковых
  вызова дали 0.5598 и 0.5628) — тест на воспроизводимость с ним невозможен;
- `py_hand_vs_range_exact` вернул 0.0 и 1.0 и на пустой доске, и на флопе —
  ведёт себя не так, как можно предположить по имени.

Поэтому единственный используемый примитив библиотеки — `eval7.evaluate`
(эвалуатор 7 карт, сравнение результатов корректно), а сэмплирование — свой
цикл с явным `random.Random(seed)` для детерминизма.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from itertools import accumulate

import eval7

from harness.contracts import Range, all_classes

_RANKS = "AKQJT98765432"
_SUITS = "shdc"
_FULL_DECK: tuple[str, ...] = tuple(rank + suit for rank in _RANKS for suit in _SUITS)

_DEFAULT_ITERATIONS_HEADS_UP = 200_000
_DEFAULT_ITERATIONS_MULTIWAY = 100_000
_DEFAULT_SEED = 42

# Кап на число попыток раздать оппонентам непересекающиеся руки на одной итерации MC.
# Без кепа `while True` может не завершиться никогда, если пространство комбо у оппонентов
# исчерпано настолько, что коллизия гарантирована на каждой попытке (например, у обоих
# оппонентов после раздачи в диапазоне остаётся ровно одно и то же комбо). 10 000 — с большим
# запасом: даже при 5% шансе успеха на попытку случайный отказ практически невозможен, а
# по-настоящему исчерпанное пространство падает за сотые доли секунды.
_MAX_COLLISION_ATTEMPTS = 10_000

_ComboSource = tuple[list[tuple[str, str]], list[float]]  # (комбо, кумулятивные веса)


def combos_of_class(cls: str) -> list[tuple[str, str]]:
    """Все конкретные комбо карт для класса ("AKs" -> 4 комбо и т. п.), без учёта мёртвых карт."""
    if len(cls) == 2:
        rank = cls[0]
        return [
            (rank + s1, rank + s2) for i, s1 in enumerate(_SUITS) for s2 in _SUITS[i + 1 :]
        ]
    hi, lo, kind = cls[0], cls[1], cls[2]
    if kind == "s":
        return [(hi + s, lo + s) for s in _SUITS]
    return [(hi + s1, lo + s2) for s1 in _SUITS for s2 in _SUITS if s1 != s2]


def _expand_range(rng: Range, dead: set[str]) -> _ComboSource:
    """Разворачивает `Range` в конкретные комбо и их кумулятивные веса.

    Исключает комбо, пересекающиеся с `dead` (карты героя и уже открытого борда).
    Комбо внутри класса равновероятны; классы взвешены весом `Range`. Пустой
    результат (диапазон пуст или полностью заблокирован) — ошибка вызова.
    """
    combos: list[tuple[str, str]] = []
    weights: list[float] = []
    for cls in all_classes():
        weight = rng.weight(cls)
        if weight <= 0.0:
            continue
        for c1, c2 in combos_of_class(cls):
            if c1 in dead or c2 in dead:
                continue
            combos.append((c1, c2))
            weights.append(weight)
    if not combos:
        raise ValueError(
            "диапазон пуст или полностью заблокирован картами героя/борда — считать эквити не от чего"
        )
    return combos, list(accumulate(weights))


def _validate_no_overlap(*groups: Sequence[str]) -> None:
    seen: set[str] = set()
    for group in groups:
        for card in group:
            if card in seen:
                raise ValueError(f"карта {card!r} встречается дважды среди героя/оппонентов/борда")
            seen.add(card)


def _mc_equity(
    hero: tuple[str, str],
    opponent_sources: Sequence[_ComboSource],
    board: list[str],
    *,
    iterations: int,
    seed: int,
) -> float:
    """Общий движок: доля банка героя за `iterations` случайных розыгрышей.

    На каждой итерации сэмплируется по одному комбо на каждый источник (при
    коллизии карт между оппонентами — вся выборка итерации отбрасывается и
    берётся заново, не более `_MAX_COLLISION_ATTEMPTS` раз — иначе `ValueError`,
    а не зависание), дораздаётся борд до 5 карт, вскрытие сравнивается через
    `eval7.evaluate`. Ничья между несколькими игроками делится поровну.
    """
    if len(board) > 5:
        raise ValueError(f"на борде не может быть больше 5 карт, получено {len(board)}")

    dead = set(hero) | set(board)
    rand = random.Random(seed)
    hero_cards = [eval7.Card(c) for c in hero]
    board_fixed = [eval7.Card(c) for c in board]
    n_to_deal = 5 - len(board)

    wins = 0.0
    for _ in range(iterations):
        for _attempt in range(_MAX_COLLISION_ATTEMPTS):
            used = set(dead)
            opponent_hands: list[tuple[str, str]] = []
            collided = False
            for combos, cum_weights in opponent_sources:
                combo = rand.choices(combos, cum_weights=cum_weights, k=1)[0]
                if combo[0] in used or combo[1] in used:
                    collided = True
                    break
                used.update(combo)
                opponent_hands.append(combo)
            if not collided:
                break
        else:
            raise ValueError(
                f"не удалось раздать оппонентам непересекающиеся руки за "
                f"{_MAX_COLLISION_ATTEMPTS} попыток — диапазоны оппонентов не допускают "
                f"коллизионно-свободной раздачи (слишком мало живых комбо относительно "
                f"числа оппонентов)"
            )

        remaining = [c for c in _FULL_DECK if c not in used]
        extra_board = [eval7.Card(c) for c in rand.sample(remaining, n_to_deal)]
        full_board = board_fixed + extra_board

        hero_score = eval7.evaluate(hero_cards + full_board)
        opponent_scores = [
            eval7.evaluate([eval7.Card(c) for c in hand] + full_board) for hand in opponent_hands
        ]
        best = max(hero_score, *opponent_scores)
        winners = 1 + sum(1 for score in opponent_scores if score == best)
        if hero_score == best:
            wins += 1.0 / winners

    return wins / iterations


def equity_vs_range(
    hero: tuple[str, str],
    rng: Range,
    board: list[str] | None = None,
    *,
    iterations: int = 200_000,
    seed: int = 42,
) -> float:
    """Доля банка героя при вскрытии против одного взвешенного диапазона."""
    return equity_vs_ranges(hero, [rng], board, iterations=iterations, seed=seed)


def equity_vs_ranges(
    hero: tuple[str, str],
    ranges: Sequence[Range],
    board: list[str] | None = None,
    *,
    iterations: int = 100_000,
    seed: int = 42,
) -> float:
    """Доля банка героя при вскрытии против нескольких диапазонов сразу (мультивей).

    Коллизии карт между сэмплами разных оппонентов отбрасываются и пересэмплируются
    (см. `_mc_equity`). Пустой диапазон или диапазон, полностью заблокированный
    картами героя/борда, — ошибка вызова (`ValueError`), а не эквити 0.
    """
    if not ranges:
        raise ValueError("нужен хотя бы один диапазон оппонента")
    board_cards = list(board or [])
    _validate_no_overlap(hero, board_cards)

    dead = set(hero) | set(board_cards)
    opponent_sources = [_expand_range(r, dead) for r in ranges]

    return _mc_equity(hero, opponent_sources, board_cards, iterations=iterations, seed=seed)


def equity_hand_vs_hand(
    hero: tuple[str, str],
    villain: tuple[str, str],
    board: list[str] | None = None,
) -> float:
    """Доля банка героя против конкретной (не диапазонной) руки оппонента."""
    board_cards = list(board or [])
    _validate_no_overlap(hero, villain, board_cards)

    villain_source: _ComboSource = ([villain], [1.0])
    return _mc_equity(
        hero,
        [villain_source],
        board_cards,
        iterations=_DEFAULT_ITERATIONS_HEADS_UP,
        seed=_DEFAULT_SEED,
    )
