"""Дро, бэкдоры, оверкарты и ценность на вскрытии (спека постфлоп-линии, §4.2, §4.2.1, §4.8).

Всё — точный счёт и полный перебор над картами героя и борда, без Монте-Карло и без
единой карты соперника: карты соперников неизвестны и из колоды не снимаются
(стандартный счёт аутов).

**Ауты** (`draw_of`). Аут — невидимая карта, которая на следующей улице даёт герою
стрит или флеш с участием его карты, которого у него ещё нет, и при этом улучшает его
лучшую руку (`eval7.evaluate` растёт). Участие — то же правило, что в
`hand_class`: стрит засчитывается, если его ранги не собираются из одних рангов
борда (`test_a_straight_completed_on_the_board_alone_is_not_an_out`); флеш — если
карта героя среди пяти старших карт масти
(`test_four_suited_on_the_board_without_a_hero_card_is_not_a_draw`). Улучшать лучшую
руку обязан каждый аут: у готового флеша десятки на стрит — не ауты
(`test_fewer_outs_than_the_threshold_is_no_draw`). Карта, которая
даёт и стрит, и флеш, — один аут (`test_combo_draw_counts_each_out_once`). Дро
есть при `MIN_DRAW_OUTS` аутах и больше; на ривере его нет.

**Вид стрит-дро — по рангам аутов на стрит.** Один ранг — гатшот (сюда же
односторонние цепочки A-2-3-4 и J-Q-K-A). Два ранга, продолжающие с обеих сторон
одну цепочку из четырёх рангов подряд (туз — и младший), — двусторонний; участие
карты героя в этой цепочке уже стоит в определении аута. Иначе — двойной
гатшот: ауты из разных цепочек.

**Шанс собрать** — точная дробь: на следующей карте `outs / unseen`; на флопе ещё
и к риверу, `1 − C(unseen − outs, 2) / C(unseen, 2)`.

**Бэкдоры и оверкарты** — не ауты: их цифры — константы владельца из таблиц ниже,
а не расчёт; в шанс собрать они не входят.

**Ценность на вскрытии** (`showdown_value`) — перебор всех пар из невидимых карт на
ривере (990 комбо), как в `river_call`.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from math import comb

import eval7

from harness.analysis.tools.equity import _FULL_DECK
from harness.analysis.tools.hand_class import (
    RANK_VALUE,
    check_cards,
    hand_strength,
)
from harness.contracts import (
    Backdoor,
    Draw,
    DrawKind,
    HandCategory,
    Overcards,
    ShowdownValue,
    class_of,
)

# Дро печатается от этого числа аутов (спека, §4.2).
MIN_DRAW_OUTS = 4

# Бэкдоры (спека, §4.2.1): константы владельца, печатаются с ≈. Флеш — точная
# вероятность двух карт масти подряд 10/47 · 9/46 = 4.16%, округлённая; стрит — по
# числу вариантов пары недостающих рангов n (точно n · 2 · 4/47 · 4/46).
BACKDOOR_FLUSH_APPROX = 0.04
BACKDOOR_STRAIGHT_APPROX = {1: 0.015, 2: 0.03, 3: 0.045}

# Оверкарты (спека, §4.2.1): константы владельца с дисконтом за то, что пара может
# оказаться не лучшей; код их не выводит. Ключ — число оверкарт.
OVERCARDS_APPROX_BY_RIVER = {2: 0.12, 1: 0.065}
OVERCARDS_APPROX_NEXT = {2: 0.065, 1: 0.03}

_FLOP, _TURN, _RIVER = 3, 4, 5
_STREETS = (_FLOP, _TURN, _RIVER)
_STRAIGHT_LEN = 5
_CHAIN_LEN = 4
_BACKDOOR_PRESENT = 3
_FLUSH_LEN = 5
_DECK_SIZE = 52
_ACE_LOW = 1
_ANY_SUIT = "x"  # масть-заглушка: стрит смотрит только на ранги

# Окна стритов как множества значений ранга: колесо (туз = 1) … бродвей (туз = 14).
_WINDOWS = tuple(frozenset(range(low, low + _STRAIGHT_LEN)) for low in range(1, 11))


def draw_of(hero: Sequence[str], board: Sequence[str]) -> Draw | None:
    """Дро героя на флопе или тёрне; `None` — меньше `MIN_DRAW_OUTS` аутов или ривер.

    `hero` — две карты героя, `board` — три, четыре или пять карт борда; иначе
    `ValueError`.
    """
    check_cards(hero, board, _STREETS)
    if len(board) == _RIVER:
        return None
    unseen = _unseen(hero, board)
    before = _score(list(hero) + list(board))
    had_flush = _flush_with_hero(hero, board)
    had_straight = _straight_with_hero(hero, board)

    flush_outs: list[str] = []
    straight_outs: list[str] = []
    for card in unseen:
        after = [*board, card]
        if _score(list(hero) + after) <= before:
            continue
        if not had_flush and _flush_with_hero(hero, after):
            flush_outs.append(card)
        if not had_straight and _straight_with_hero(hero, after):
            straight_outs.append(card)

    outs = [card for card in unseen if card in flush_outs or card in straight_outs]
    if len(outs) < MIN_DRAW_OUTS:
        return None
    kinds: list[DrawKind] = []
    if flush_outs:
        kinds.append(DrawKind.FLUSH)
    out_ranks = sorted({card[0] for card in straight_outs}, key=RANK_VALUE.__getitem__)
    out_ranks.reverse()
    if out_ranks:
        kinds.append(_straight_kind(hero, board, out_ranks))
    n_unseen = len(unseen)
    return Draw(
        kinds=kinds,
        out_ranks=out_ranks,
        outs=outs,
        unseen=n_unseen,
        hit_next=len(outs) / n_unseen,
        hit_by_river=_hit_by_river(len(outs), n_unseen) if len(board) == _FLOP else None,
    )


def draw_missed(hero: Sequence[str], board: Sequence[str]) -> bool:
    """«Дро не закрылось»: на ривере, если на тёрне было дро, а ривер не аут.

    До ривера — `False`: печатается только на ривере (спека, §4.2).
    """
    check_cards(hero, board, _STREETS)
    if len(board) != _RIVER:
        return False
    turn_draw = draw_of(hero, board[:_TURN])
    return turn_draw is not None and board[_TURN] not in turn_draw.outs


def backdoors(hero: Sequence[str], board: Sequence[str]) -> list[Backdoor]:
    """Бэкдоры флеша и стрита на флопе; на тёрне и ривере — пусто (спека, §4.2.1).

    Флеш: у героя и на борде ровно три карты одной масти, хотя бы одна у героя (при
    флеш-дро их четыре). Стрит: окна стрита, в которых есть ровно три ранга, и эти
    три не собираются из одних рангов борда; `variants` — число различных пар
    недостающих рангов. Бэкдора стрита нет при стрит-дро и при готовом стрите героя.
    """
    check_cards(hero, board, _STREETS)
    if len(board) != _FLOP:
        return []
    found: list[Backdoor] = []
    cards = [*hero, *board]
    for suit in {card[1] for card in hero}:
        if sum(card[1] == suit for card in cards) == _BACKDOOR_PRESENT:
            found.append(Backdoor(kind="flush", variants=None, approx=BACKDOOR_FLUSH_APPROX))

    if not _straight_ranks_to_come(hero, board):
        variants = _backdoor_straight_variants(hero, board)
        if variants:
            found.append(
                Backdoor(
                    kind="straight",
                    variants=variants,
                    approx=BACKDOOR_STRAIGHT_APPROX[variants],
                )
            )
    return found


def overcards(hero: Sequence[str], board: Sequence[str]) -> Overcards | None:
    """Карты героя старше старшей карты борда, когда его карта не входит ни в одну комбинацию.

    Только флоп и тёрн; на ривере и при руке героя (пара и сильнее с участием его
    карты) — `None` (спека, §4.2.1). Цифры — константы владельца.
    """
    check_cards(hero, board, _STREETS)
    if len(board) == _RIVER:
        return None
    category = hand_strength(hero, board).category
    if category not in (HandCategory.NO_PAIR, HandCategory.ON_BOARD):
        return None
    top = max(RANK_VALUE[card[0]] for card in board)
    over = sorted(
        (card for card in hero if RANK_VALUE[card[0]] > top),
        key=lambda card: RANK_VALUE[card[0]],
        reverse=True,
    )
    if not over:
        return None
    return Overcards(
        cards=over,
        approx_by_river=OVERCARDS_APPROX_BY_RIVER[len(over)] if len(board) == _FLOP else None,
        approx_next=OVERCARDS_APPROX_NEXT[len(over)],
    )


def showdown_value(hero: Sequence[str], board: Sequence[str]) -> ShowdownValue | None:
    """Перебор всех комбо соперника из невидимых карт на ривере; до ривера — `None`."""
    check_cards(hero, board, _STREETS)
    if len(board) != _RIVER:
        return None
    parsed = {card: eval7.Card(card) for card in _FULL_DECK}
    board_cards = [parsed[card] for card in board]
    hero_score = eval7.evaluate([parsed[card] for card in hero] + board_cards)
    wins = ties = losses = 0
    tied_classes: set[str] = set()
    for card1, card2 in combinations(_unseen(hero, board), 2):
        score = eval7.evaluate([parsed[card1], parsed[card2], *board_cards])
        if score < hero_score:
            wins += 1
        elif score > hero_score:
            losses += 1
        else:
            ties += 1
            tied_classes.add(class_of(card1, card2))
    return ShowdownValue(wins=wins, ties=ties, losses=losses, ties_with=sorted(tied_classes))


def _unseen(hero: Sequence[str], board: Sequence[str]) -> list[str]:
    known = {*hero, *board}
    unseen = [card for card in _FULL_DECK if card not in known]
    assert len(unseen) == _DECK_SIZE - len(known)
    return unseen


def _score(cards: Sequence[str]) -> int:
    return eval7.evaluate([eval7.Card(card) for card in cards])


def _hit_by_river(outs: int, unseen: int) -> float:
    return 1 - comb(unseen - outs, 2) / comb(unseen, 2)


def _values(ranks: set[str]) -> set[int]:
    """Значения рангов для окон стрита: туз — и 14, и 1."""
    values = {RANK_VALUE[rank] for rank in ranks}
    if RANK_VALUE["A"] in values:
        values.add(_ACE_LOW)
    return values


def _best_window(ranks: set[str]) -> frozenset[int] | None:
    """Старший стрит из этих рангов как окно значений; `None` — стрита нет."""
    values = _values(ranks)
    for window in reversed(_WINDOWS):
        if window <= values:
            return window
    return None


def _straight_with_hero(hero: Sequence[str], board: Sequence[str]) -> bool:
    """Старший стрит из карт героя и борда не собирается из одних рангов борда."""
    window = _best_window({card[0] for card in [*hero, *board]})
    return window is not None and not window <= _values({card[0] for card in board})


def _flush_with_hero(hero: Sequence[str], board: Sequence[str]) -> bool:
    """Есть масть из пяти карт и больше, и карта героя среди пяти старших её карт."""
    cards = [*hero, *board]
    for suit in {card[1] for card in hero}:
        suited = sorted(
            (card for card in cards if card[1] == suit),
            key=lambda card: RANK_VALUE[card[0]],
            reverse=True,
        )
        if len(suited) >= _FLUSH_LEN and any(card in hero for card in suited[:_FLUSH_LEN]):
            return True
    return False


def _straight_ranks_to_come(hero: Sequence[str], board: Sequence[str]) -> set[str]:
    """Ранги, с картой которых на следующей улице у героя есть стрит, — по одним рангам.

    Тот же счёт, что у аутов на стрит, но без условия «лучшая рука растёт»: бэкдор
    стрита не печатается и тогда, когда стрит-дро есть, а лучшую руку героя
    стрит не улучшит. При готовом стрите героя сюда попадает любой ранг.
    """
    return {
        rank
        for rank in RANK_VALUE
        if _straight_with_hero(hero, [*board, rank + _ANY_SUIT])
    }


def _straight_kind(hero: Sequence[str], board: Sequence[str], out_ranks: list[str]) -> DrawKind:
    """Вид стрит-дро по рангам аутов на стрит (правило — докстринг модуля)."""
    if len(out_ranks) == 1:
        return DrawKind.GUTSHOT
    if len(out_ranks) == 2 and _open_ended(hero, board, set(out_ranks)):
        return DrawKind.OPEN_ENDED
    return DrawKind.DOUBLE_GUTSHOT


def _open_ended(hero: Sequence[str], board: Sequence[str], out_ranks: set[str]) -> bool:
    """Оба ранга аутов продолжают с обеих сторон одну цепочку из четырёх рангов подряд."""
    known = _values({card[0] for card in [*hero, *board]})
    outs = _values(out_ranks)
    for low in range(_ACE_LOW, RANK_VALUE["A"] - _CHAIN_LEN + 2):
        chain = set(range(low, low + _CHAIN_LEN))
        ends = {low - 1, low + _CHAIN_LEN}
        if chain <= known and ends <= outs:
            return True
    return False


def _backdoor_straight_variants(hero: Sequence[str], board: Sequence[str]) -> int:
    """Число различных пар недостающих рангов, которые двумя картами дают стрит героя."""
    known = _values({card[0] for card in [*hero, *board]})
    on_board = _values({card[0] for card in board})
    pairs: set[frozenset[int]] = set()
    for window in _WINDOWS:
        present = window & known
        if len(present) == _BACKDOOR_PRESENT and not present <= on_board:
            pairs.add(frozenset(window - present))
    return len(pairs)

