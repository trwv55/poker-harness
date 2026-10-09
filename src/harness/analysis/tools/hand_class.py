"""Сила готовой руки героя на флопе, тёрне и ривере (спека постфлоп-линии, §4.1).

**Комбинация — по `eval7`, участие — по рангам.** Класс лучшей пятикарточной
комбинации из карт героя и борда называет `eval7.handtype`; сами пять карт —
подмножество с наибольшим `eval7.evaluate`. Комбинация засчитывается герою, только
если его карта входит в саму комбинацию (в пару, две пары, сет, стрит, флеш), а не в
кикер. Проверка — по мультимножеству рангов: ядро комбинации (пары, тройка, каре;
у стрита и фулл-хауса — все пять рангов) не собирается из одних рангов борда.
Флеш и стрит-флеш проверяются по самим картам: карта героя входит в пять старших
карт масти (`test_a_flush_card_below_five_board_flush_cards_does_not_play`).

Иначе рука «на борде» (`HandCategory.ON_BOARD`), и называется, чем играет герой:
кикером (старший ранг, который герой добавил к пяти картам сверх борда) или
бордом (`test_pair_on_the_board_plays_the_heros_kicker`,
`test_board_straight_plays_the_board`).

**Таблица §4.1 сверху вниз.** Ранги борда берутся различные, по убыванию: на
Q Q 7 2 старшая — Q, вторая — 7, младшая — 2. Пара героя меряется своим рангом
среди них, а не силой всей комбинации: на борде Q Q 4 у героя 4 — две пары, но
меряется пара четвёрок, второй различный ранг борда, — средняя пара
(`test_two_pair_with_the_board_pair_is_judged_by_the_heros_pair`). Пара героя,
которая не вошла в лучшие пять карт (на борде Q Q 7 7 4 у героя 4), — уже рука на
борде (`test_a_counterfeited_pair_is_on_the_board`).

Трипс и две пары целиком на борде при старшем из возможных кикеров у героя — класс
«средняя», а не «слабая» (решение владельца 2026-10-09). Старший из возможных —
старший ранг, которого нет на борде (`test_trips_on_the_board_with_the_top_kicker_is_medium`,
`test_with_an_ace_on_the_board_the_top_possible_kicker_is_the_king`).

Карты соперника функция не читает: на входе только карты героя и борд.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import combinations

import eval7

from harness.analysis.tools.equity import _FULL_DECK
from harness.contracts import (
    RANKS,
    Combination,
    HandCategory,
    HandStrength,
    StrengthClass,
)

# Порог сильного кикера у топ-пары (спека, §4.1): вторая карта героя этого ранга
# или старше. Предложение спеки, одной константой.
TOP_PAIR_STRONG_KICKER_MIN = "T"

_BOARD_SIZES = (3, 4, 5)
_HAND_SIZE = 5
_FOUR = 4

# Сила ранга: A — 14, 2 — 2. Колесо (A-2-3-4-5) считается отдельно: в нём туз младший.
RANK_VALUE = {rank: 14 - index for index, rank in enumerate(RANKS)}

# Имена классов `eval7.handtype` — те же строки, что в `river_call._HAND_TYPES`;
# незнакомая строка роняет расчёт `KeyError`, а не съезжает в неверный класс.
_COMBINATION = {
    "High Card": Combination.HIGH_CARD,
    "Pair": Combination.PAIR,
    "Two Pair": Combination.TWO_PAIR,
    "Trips": Combination.TRIPS,
    "Straight": Combination.STRAIGHT,
    "Flush": Combination.FLUSH,
    "Full House": Combination.FULL_HOUSE,
    "Quads": Combination.QUADS,
    "Straight Flush": Combination.STRAIGHT_FLUSH,
}

# Комбинации, у которых ядро — все пять карт; у остальных ядро — ранги, встреченные
# в пятёрке больше одного раза.
_WHOLE_FIVE = (
    Combination.STRAIGHT,
    Combination.FLUSH,
    Combination.FULL_HOUSE,
    Combination.STRAIGHT_FLUSH,
)
_SUITED = (Combination.FLUSH, Combination.STRAIGHT_FLUSH)
_STRONG_MADE = {
    Combination.STRAIGHT_FLUSH: HandCategory.STRAIGHT_FLUSH,
    Combination.QUADS: HandCategory.QUADS,
    Combination.FULL_HOUSE: HandCategory.FULL_HOUSE,
}
_MEDIUM_ON_BOARD = (Combination.TRIPS, Combination.TWO_PAIR)


def check_cards(hero: Sequence[str], board: Sequence[str], board_sizes: Sequence[int]) -> None:
    """Две карты героя, борд допустимой длины, все карты из колоды и без повторов.

    Иначе `ValueError`. Общая проверка входа для `hand_class` и `draws`.
    """
    if len(hero) != 2:
        raise ValueError(f"у героя две карты, получено {len(hero)}")
    if len(board) not in board_sizes:
        raise ValueError(f"борд из {tuple(board_sizes)} карт, получено {len(board)}")
    known = list(hero) + list(board)
    unknown = [card for card in known if card not in _FULL_DECK]
    if unknown:
        raise ValueError(f"не карты колоды: {unknown}")
    if len(set(known)) != len(known):
        raise ValueError(f"карта встречается дважды среди карт героя и борда: {known}")


def hand_strength(hero: Sequence[str], board: Sequence[str]) -> HandStrength:
    """Категория и класс силы готовой руки героя по таблице §4.1.

    `hero` — две карты героя, `board` — три, четыре или пять карт борда в записи
    колоды `equity` ("Ah", "Td"). Иначе `ValueError`.
    """
    check_cards(hero, board, _BOARD_SIZES)
    five, combination = best_five(list(hero) + list(board))
    five_ranks = Counter(card[0] for card in five)
    board_ranks = Counter(card[0] for card in board)
    core = _core(five_ranks, combination)

    if combination is Combination.HIGH_CARD:
        return HandStrength(
            category=HandCategory.NO_PAIR,
            strength=StrengthClass.WEAK,
            combination=combination,
            ranks=[_highest(card[0] for card in hero)],
            plays="hand",
            kicker=None,
        )
    if combination in _SUITED:
        mine = any(card in hero for card in five)
    else:
        mine = bool(core - board_ranks)
    if not mine:
        return _on_board(five_ranks, board_ranks, core, combination, five)
    return _heros(hero, board, five, core, combination)


def best_five(cards: Sequence[str]) -> tuple[tuple[str, ...], Combination]:
    """Пять карт с наибольшим `eval7.evaluate` и их класс комбинации.

    Из нескольких пятёрок с равным счётом берётся первая; участие героя дальше
    проверяется по рангам, а не по этой пятёрке, поэтому выбор не важен
    (`test_board_straight_plays_the_board`: пятёрка героя и пятёрка борда).
    """
    parsed = {card: eval7.Card(card) for card in cards}
    best_score = -1
    best: tuple[str, ...] = ()
    for five in combinations(cards, _HAND_SIZE):
        score = eval7.evaluate([parsed[card] for card in five])
        if score > best_score:
            best_score, best = score, five
    return best, _COMBINATION[eval7.handtype(best_score)]


def _core(five_ranks: Counter[str], combination: Combination) -> Counter[str]:
    """Ранги самой комбинации (без кикеров) как мультимножество."""
    if combination in _WHOLE_FIVE:
        return Counter(five_ranks)
    return Counter({rank: count for rank, count in five_ranks.items() if count > 1})


def _heros(
    hero: Sequence[str],
    board: Sequence[str],
    five: Sequence[str],
    core: Counter[str],
    combination: Combination,
) -> HandStrength:
    """Комбинация с участием карты героя: строки §4.1 от стрит-флеша до слабой пары."""
    if combination in _STRONG_MADE:
        return _made(_STRONG_MADE[combination], StrengthClass.STRONG, combination, core, five)
    if combination is Combination.FLUSH:
        suit = five[0][1]
        weak = sum(card[1] == suit for card in board) >= _FOUR
        category = HandCategory.WEAK_FLUSH if weak else HandCategory.FLUSH
        strength = StrengthClass.MEDIUM if weak else StrengthClass.STRONG
        return _made(category, strength, combination, core, five)
    if combination is Combination.STRAIGHT:
        weak = len(set(core) & {card[0] for card in board}) >= _FOUR
        category = HandCategory.WEAK_STRAIGHT if weak else HandCategory.STRAIGHT
        strength = StrengthClass.MEDIUM if weak else StrengthClass.STRONG
        return _made(category, strength, combination, core, five)

    hero_ranks = [card[0] for card in hero]
    if combination is Combination.TRIPS:
        (rank,) = core
        pocket = hero_ranks.count(rank) == 2
        category = HandCategory.SET if pocket else HandCategory.TRIPS
        return _made(category, StrengthClass.STRONG, combination, core, five)

    # Пара или две пары: сила — у пары героя, а не у всей комбинации.
    board_distinct = sorted({card[0] for card in board}, key=RANK_VALUE.__getitem__, reverse=True)
    paired = sorted({rank for rank in hero_ranks if rank in core}, key=RANK_VALUE.__getitem__)
    if hero_ranks[0] == hero_ranks[1]:
        above = sum(RANK_VALUE[rank] > RANK_VALUE[hero_ranks[0]] for rank in board_distinct)
        category, strength = _POCKET[min(above, 2)]
        return _pair(category, strength, combination, [hero_ranks[0]])
    if len(paired) == 2:
        return _pair(
            HandCategory.TWO_PAIR, StrengthClass.STRONG, combination, list(reversed(paired))
        )
    (rank,) = paired
    place = board_distinct.index(rank)
    if place == 0:
        other = hero_ranks[1] if hero_ranks[0] == rank else hero_ranks[0]
        if RANK_VALUE[other] >= RANK_VALUE[TOP_PAIR_STRONG_KICKER_MIN]:
            return _pair(
                HandCategory.TOP_PAIR_STRONG_KICKER, StrengthClass.STRONG, combination, [rank]
            )
        return _pair(HandCategory.TOP_PAIR_WEAK_KICKER, StrengthClass.MEDIUM, combination, [rank])
    if place == 1:
        return _pair(HandCategory.MIDDLE_PAIR, StrengthClass.MEDIUM, combination, [rank])
    return _pair(HandCategory.WEAK_PAIR, StrengthClass.WEAK, combination, [rank])


# Карманная пара по числу различных рангов борда выше неё: ни одного — оверпара,
# один — средняя пара, два и больше — слабая.
_POCKET = (
    (HandCategory.OVERPAIR, StrengthClass.STRONG),
    (HandCategory.MIDDLE_PAIR, StrengthClass.MEDIUM),
    (HandCategory.WEAK_PAIR, StrengthClass.WEAK),
)


def _on_board(
    five_ranks: Counter[str],
    board_ranks: Counter[str],
    core: Counter[str],
    combination: Combination,
    five: Sequence[str],
) -> HandStrength:
    """Рука на борде: чем играет герой и класс «слабая» или «средняя» (§4.1)."""
    added = five_ranks - board_ranks
    kicker = _highest(added) if added else None
    top_possible = _highest(rank for rank in RANKS if rank not in board_ranks)
    medium = combination in _MEDIUM_ON_BOARD and kicker == top_possible
    return HandStrength(
        category=HandCategory.ON_BOARD,
        strength=StrengthClass.MEDIUM if medium else StrengthClass.WEAK,
        combination=combination,
        ranks=_named_ranks(core, combination, five),
        plays="kicker" if kicker is not None else "board",
        kicker=kicker,
    )


def _made(
    category: HandCategory,
    strength: StrengthClass,
    combination: Combination,
    core: Counter[str],
    five: Sequence[str],
) -> HandStrength:
    return HandStrength(
        category=category,
        strength=strength,
        combination=combination,
        ranks=_named_ranks(core, combination, five),
        plays="hand",
        kicker=None,
    )


def _pair(
    category: HandCategory, strength: StrengthClass, combination: Combination, ranks: list[str]
) -> HandStrength:
    return HandStrength(
        category=category,
        strength=strength,
        combination=combination,
        ranks=ranks,
        plays="hand",
        kicker=None,
    )


def _named_ranks(core: Counter[str], combination: Combination, five: Sequence[str]) -> list[str]:
    """Что называть у комбинации: старшую карту стрита и флеша, иначе ранги ядра.

    Ранги ядра — по убыванию числа карт, затем по старшинству: у фулл-хауса тройка
    идёт первой, у двух пар — старшая пара.
    """
    if combination in (Combination.STRAIGHT, Combination.STRAIGHT_FLUSH):
        return [straight_top({card[0] for card in five})]
    if combination is Combination.FLUSH:
        return [_highest(card[0] for card in five)]
    return sorted(core, key=lambda rank: (core[rank], RANK_VALUE[rank]), reverse=True)


def straight_top(ranks: set[str]) -> str:
    """Старшая карта стрита из ровно этих пяти рангов; у колеса — пятёрка."""
    if ranks == {"A", "2", "3", "4", "5"}:
        return "5"
    return _highest(ranks)


def _highest(ranks: Iterable[str]) -> str:
    """Старший ранг из перечисленных."""
    return max(ranks, key=RANK_VALUE.__getitem__)
