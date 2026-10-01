"""EV решения пуш-фолда симуляцией полной раздачи: карты сданы всем.

Аналитический EV (`shove_ev_bb`, `call_shove_ev_bb`) снимает из колоды только
карты героя: оппоненты коллируют независимо друг от друга, как будто колода
перед каждым полна. На деле сбросившие уносят преимущественно слабые руки, и у
того, кто коллирует, чаще сильная — замер против GTO-солвера (спека
2026-10-01-pushfold-overcalls-dead-cards, §1) показал, что это половина
переоценки пар. Здесь каждая итерация сдаёт карты ВСЕМ местам и разыгрывает
действие по диапазонам равновесия (`multiway.py`): снятие карт, мультивей-
вскрытия и сайд-поты получаются по построению, а не приближением.

**Розыгрыш.** Рука героя — случайное комбо его класса (вердикт судит класс, как
и равновесие). Места отвечают по порядку хода: пока коллеров нет — холодным
колл-диапазоном, когда коллер уже есть — оверколл-диапазоном (при двух и более
коллерах — им же: решатель третьей роли не знает, а оверколл — самый тесный из
имеющихся ответов). Дробный вес класса — вероятность колла.

**Против шова** (`call_ev_full_deal`) рука шовера сдаётся из его диапазона
шова, а сбросившие между шовером и героем — условием: раздача, где кто-то из
них заколлировал бы, отбрасывается. Так их сброс снимает из колоды ровно те
карты, которые сброс и означает.

**Деньги.** Каждый участник — (поставлено, вклад) в bb, «поставлено» входит во
вклад. Вскрытие делится слоями: каждый слой между соседними уровнями вкладов
разыгрывают те, чей вклад до него дотянулся; непокрытый остаток возвращается.
Деньги, не принадлежащие ни одному участнику (анте, посты сбросивших), лежат в
нижнем слое. EV — относительно паса героя: его собственные посты в базлайне
«я пасую» уже потеряны.

**Детерминизм.** Сид фиксирован вызывающим; одни и те же входы дают одно и то
же число — на этом стоит дисковый кэш вердикта.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from itertools import accumulate

import eval7

from harness.analysis.tools.equity import combos_of_class
from harness.contracts import Range, class_of

_RANKS = "AKQJT98765432"
_SUITS = "shdc"
_DECK: tuple[str, ...] = tuple(rank + suit for rank in _RANKS for suit in _SUITS)
_CARD: dict[str, eval7.Card] = {card: eval7.Card(card) for card in _DECK}

# Сколько раз подряд можно отбросить раздачу, прежде чем признать условие
# невыполнимым: сбросившие перед героем держат колл-диапазоны, которые почти
# всегда допускают пас, и на реальных точках доля отброса — единицы процентов.
_MAX_REJECTIONS_IN_A_ROW = 10_000


@dataclass(frozen=True)
class Responder:
    """Место, которое отвечает на шов: холодный колл и оверколл."""

    posted_bb: float
    total_bb: float
    cold: Range
    over: Range


@dataclass(frozen=True)
class Shover:
    """Чужой шов, на который отвечает герой: диапазон и деньги шовера."""

    posted_bb: float
    total_bb: float
    push: Range


@cache
def _class_by_pair() -> dict[tuple[str, str], str]:
    return {(a, b): class_of(a, b) for a in _DECK for b in _DECK if a != b}


def _weights(rng: Range) -> dict[str, float]:
    return {cls: w for cls, w in rng.weights.items() if w > 0.0}


def _settle(
    hero_rank: int,
    hero_commit: float,
    others: Sequence[tuple[int, float]],
    dead_bb: float,
) -> float:
    """Сколько выигрывает герой при вскрытии слоями. `others` — (сила, вклад)."""
    entries = [(hero_rank, hero_commit), *others]
    levels = sorted({commit for _, commit in entries})
    won = 0.0
    previous = 0.0
    for level_index, level in enumerate(levels):
        eligible = [rank for rank, commit in entries if commit >= level]
        layer = (level - previous) * len(eligible) + (dead_bb if level_index == 0 else 0.0)
        previous = level
        if hero_commit < level:
            break
        best = max(eligible)
        if hero_rank == best:
            won += layer / sum(1 for rank in eligible if rank == best)
    return won


def _commits(aggressor_total: float, caller_totals: Sequence[float]) -> tuple[float, list[float]]:
    """Вклады при олл-ине: коллер ставит не больше ставки агрессора, агрессор —
    не больше, чем покрыл самый глубокий коллер (остаток возвращается)."""
    callers = [min(total, aggressor_total) for total in caller_totals]
    aggressor = min(aggressor_total, max(callers)) if callers else aggressor_total
    return aggressor, callers


def _deal_hero(hero_cls: str, rand: random.Random) -> tuple[str, str]:
    combos = combos_of_class(hero_cls)
    return combos[rand.randrange(len(combos))]


def _respond(
    hands: Sequence[tuple[str, str]],
    callers_already: int,
    rand: random.Random,
    cold_weights: Sequence[dict[str, float]],
    over_weights: Sequence[dict[str, float]],
) -> list[int]:
    """Индексы ответивших коллом мест — по порядку хода.

    Жребий тянется для каждого места всегда, даже при нулевом весе: тогда при
    тех же сиде и картах разные диапазоны получают одни и те же жребии, и EV
    соседних ширин сетки отличается диапазонами, а не шумом.
    """
    by_pair = _class_by_pair()
    called: list[int] = []
    for index, hand in enumerate(hands):
        table = cold_weights[index] if callers_already + len(called) == 0 else over_weights[index]
        draw = rand.random()
        if draw < table.get(by_pair[hand], 0.0):
            called.append(index)
    return called


def _rank(hand: tuple[str, str], board: Sequence[eval7.Card]) -> int:
    return eval7.evaluate([_CARD[hand[0]], _CARD[hand[1]], *board])


def shove_ev_full_deal(
    hero_cls: str,
    hero_posted_bb: float,
    hero_total_bb: float,
    responders: Sequence[Responder],
    pot_dead_bb: float,
    *,
    iterations: int,
    seed: int,
) -> float:
    """EV шова героя в неоткрытый банк относительно паса, в bb.

    `pot_dead_bb` — весь банк на решении, включая посты героя и всех
    `responders` (как у `shove_ev_bb`).
    """
    if not responders:
        raise ValueError("позади героя нет ни одного игрока — разыгрывать нечего")
    dead_bb = pot_dead_bb - hero_posted_bb - sum(r.posted_bb for r in responders)
    if dead_bb < -1e-9:
        raise ValueError("сумма постов больше банка: банк обязан включать посты всех мест")
    cold_weights = [_weights(r.cold) for r in responders]
    over_weights = [_weights(r.over) for r in responders]
    rand = random.Random(seed)
    seats = len(responders)
    total = 0.0
    for _ in range(iterations):
        hero = _deal_hero(hero_cls, rand)
        deck = [card for card in _DECK if card not in hero]
        rand.shuffle(deck)
        hands = [(deck[2 * i], deck[2 * i + 1]) for i in range(seats)]
        called = _respond(hands, 0, rand, cold_weights, over_weights)
        if not called:
            total += pot_dead_bb
            continue
        board = [_CARD[card] for card in deck[2 * seats : 2 * seats + 5]]
        hero_commit, caller_commits = _commits(
            hero_total_bb, [responders[i].total_bb for i in called]
        )
        folded_posts = sum(r.posted_bb for i, r in enumerate(responders) if i not in called)
        won = _settle(
            _rank(hero, board),
            hero_commit,
            [
                (_rank(hands[i], board), commit)
                for i, commit in zip(called, caller_commits, strict=True)
            ],
            dead_bb + folded_posts,
        )
        total += won - (hero_commit - hero_posted_bb)
    return total / iterations


def call_ev_full_deal(
    hero_cls: str,
    hero_posted_bb: float,
    hero_total_bb: float,
    shover: Shover,
    folded_between: Sequence[Range],
    responders: Sequence[Responder],
    pot_dead_bb: float,
    *,
    iterations: int,
    seed: int,
) -> float:
    """EV колла чужого шова относительно паса, в bb.

    `folded_between` — холодные колл-диапазоны тех, кто между шовером и героем
    сбросил: раздача, где кто-то из них заколлировал бы, отбрасывается.
    `responders` — живые позади героя; герой уже коллер, поэтому они отвечают
    оверколл-диапазонами. `shover.total_bb` — уровень ставки шовера: больше
    него не ставит ни один коллер. `pot_dead_bb` — банк на решении, включая посты (и
    ставку) шовера, героя и всех `responders`, но не включая доплату героя.
    """
    dead_bb = (
        pot_dead_bb - hero_posted_bb - shover.posted_bb - sum(r.posted_bb for r in responders)
    )
    if dead_bb < -1e-9:
        raise ValueError("сумма постов больше банка: банк обязан включать посты всех мест")
    by_pair = _class_by_pair()
    shove_combos = [
        combo
        for cls, weight in sorted(_weights(shover.push).items())
        for combo in combos_of_class(cls)
    ]
    shove_cumulative = list(
        accumulate(shover.push.weights[by_pair[combo]] for combo in shove_combos)
    )
    folded_weights = [_weights(rng) for rng in folded_between]
    cold_weights = [_weights(r.cold) for r in responders]
    over_weights = [_weights(r.over) for r in responders]
    rand = random.Random(seed)
    others = len(folded_between) + len(responders)
    total = 0.0
    for _ in range(iterations):
        # Отбрасывается раздача целиком (и рука героя тоже): так условное
        # распределение «шовер в своём диапазоне, сбросившие сбросили» точное.
        rejected = 0
        while True:
            hero = _deal_hero(hero_cls, rand)
            shove_hand = rand.choices(shove_combos, cum_weights=shove_cumulative)[0]
            if shove_hand[0] not in hero and shove_hand[1] not in hero:
                deck = [card for card in _DECK if card not in hero and card not in shove_hand]
                rand.shuffle(deck)
                hands = [(deck[2 * i], deck[2 * i + 1]) for i in range(others)]
                between = hands[: len(folded_between)]
                draws = [rand.random() for _ in between]
                if all(
                    draw >= weights.get(by_pair[hand], 0.0)
                    for draw, hand, weights in zip(draws, between, folded_weights, strict=True)
                ):
                    break
            rejected += 1
            if rejected >= _MAX_REJECTIONS_IN_A_ROW:
                raise ValueError(
                    "сбросившие между шовером и героем почти никогда не сбрасывают при "
                    "этих диапазонах — условие раздачи невыполнимо"
                )
        behind_hands = hands[len(folded_between) :]
        called = _respond(behind_hands, 1, rand, cold_weights, over_weights)
        board = [_CARD[card] for card in deck[2 * others : 2 * others + 5]]
        shover_commit, caller_commits = _commits(
            shover.total_bb, [hero_total_bb, *(responders[i].total_bb for i in called)]
        )
        folded_posts = sum(r.posted_bb for i, r in enumerate(responders) if i not in called)
        won = _settle(
            _rank(hero, board),
            caller_commits[0],
            [
                (_rank(shove_hand, board), shover_commit),
                *(
                    (_rank(behind_hands[i], board), commit)
                    for i, commit in zip(called, caller_commits[1:], strict=True)
                ),
            ],
            dead_bb + folded_posts,
        )
        total += won - (caller_commits[0] - hero_posted_bb)
    return total / iterations
