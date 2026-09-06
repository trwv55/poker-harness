"""Статистика игрока по списку рук — чистые функции над действиями (задача 23).

Ни одного числа отсюда не приходит от модели: каждая величина — счётчик
раздач или ситуаций, а формула счётчика записана в докстринге его функции и
пришпилена тестом на синтетической руке (`tests/test_player_stats.py`).

**Почему счётчики, а не проценты.** Наружу отдаётся `PlayerStats` со
числителями и знаменателями (доли — свойства контракта): среднее игрока по
нескольким турнирам считается сложением счётчиков, а не усреднением процентов,
иначе турнир из 12 раздач весил бы столько же, сколько турнир из 300. Поэтому
и функция одна: `player_stats(hands)` над ЛЮБЫМ списком рук — одного турнира
или всех сразу.

**Вход — `CanonicalHand`, не `EnrichedHand`.** Всё, что нужно этим формулам, —
действия, разложенные по улицам, и метка героя; отчёт движка не нужен ни одной
из них. Это не экономия строки импорта: среднее по всем турнирам игрока читает
из базы только колонку `hands.canonical` (`memory.repos`), а `enriched` там
весит кратно больше и не пригодился бы.

**Блайнды и анте — не действия.** Нормализатор кладёт посты в
`CanonicalHand.posts` отдельно от `actions` (`normalizer/normalize.py`),
поэтому вынужденная ставка не может попасть в VPIP по построению формулы:
считаются только действия героя. Чек большого блайнда — `ActionKind.CHECK`, и
в VPIP он тоже не входит (`test_a_posted_blind_is_not_a_voluntary_investment`).
"""

from __future__ import annotations

from collections.abc import Iterable

from harness.contracts import ActionKind, CanonicalAction, CanonicalHand, PlayerStats, Street

__all__ = ["player_stats"]

# Действия, которыми игрок добровольно кладёт фишки в банк. Пас и чек не кладут
# ничего, посты сюда не попадают вовсе (см. докстринг модуля).
_VOLUNTARY = frozenset({ActionKind.CALL, ActionKind.BET, ActionKind.RAISE})
# Действия, повышающие ставку. `BET` — префлоп-повышение в неоткрытом банке
# (источник может записать шов и так, и рейзом), `RAISE` — повышение поверх
# чужой ставки.
_AGGRESSIVE = frozenset({ActionKind.BET, ActionKind.RAISE})


def _hero_seated(hand: CanonicalHand) -> bool:
    return any(p.label == hand.hero_label for p in hand.players)


def _street(hand: CanonicalHand, street: Street) -> list[CanonicalAction]:
    return [a for a in hand.actions if a.street is street]


def _is_vpip(hand: CanonicalHand) -> bool:
    """VPIP: есть ли у героя на префлопе хоть одно действие вида колл/бет/рейз.

    Формула: `any(a.kind in {call, bet, raise} for a in preflop-действия героя)`.
    Знаменатель — раздача (одна рука даёт не больше единицы, сколько бы раз
    герой ни доложил).
    """
    return any(
        a.label == hand.hero_label and a.kind in _VOLUNTARY
        for a in _street(hand, Street.PREFLOP)
    )


def _is_pfr(hand: CanonicalHand) -> bool:
    """PFR: есть ли у героя на префлопе повышение (`bet` или `raise`).

    Формула: `any(a.kind in {bet, raise} for a in preflop-действия героя)`.
    Колл в PFR не входит (`test_a_call_is_vpip_but_not_pfr`).
    """
    return any(
        a.label == hand.hero_label and a.kind in _AGGRESSIVE
        for a in _street(hand, Street.PREFLOP)
    )


def _reraise(hand: CanonicalHand) -> tuple[bool, bool]:
    """Ре-рейз: (была ли возможность, воспользовался ли герой) — по префлопу.

    Возможность: перед ПЕРВЫМ действием героя на префлопе кто-то другой уже
    повысил (`bet`/`raise`). То есть герой отвечает на повышение, а не открывает
    банк (`test_an_open_raise_is_not_a_reraise_chance`).

    Засчитано: это первое действие героя — само повышение
    (`test_raising_over_an_open_is_a_reraise`).

    Считается «3-бет и выше»: повышение поверх 3-бета (4-бет) попадает сюда же —
    формула смотрит на то, что герой отвечает повышением на повышение, а не на
    номер круга торговли.
    """
    raised_before = False
    for action in _street(hand, Street.PREFLOP):
        if action.label == hand.hero_label:
            return raised_before, raised_before and action.kind in _AGGRESSIVE
        if action.kind in _AGGRESSIVE:
            raised_before = True
    return False, False


def _fold_to_cbet(hand: CanonicalHand) -> tuple[bool, bool]:
    """Сдача на продолженную ставку: (была ли она против героя, сдался ли герой).

    Продолженная ставка определена так: первая ставка (`bet`) на флопе, сделанная
    ПОСЛЕДНИМ игроком, повысившим на префлопе (префлоп-агрессором). Ставка от
    коллера, когда агрессор чекнул, продолженной не считается
    (`test_a_flop_bet_from_a_caller_is_not_a_continuation_bet`).

    Против героя она есть, если агрессор — не сам герой
    (`test_hero_own_cbet_is_not_a_chance`) и у героя есть действие на флопе после
    этой ставки. Сдачей считается, если это действие — пас.
    """
    aggressor: str | None = None
    for action in _street(hand, Street.PREFLOP):
        if action.kind in _AGGRESSIVE:
            aggressor = action.label
    if aggressor is None or aggressor == hand.hero_label:
        return False, False

    flop = _street(hand, Street.FLOP)
    bets = [i for i, a in enumerate(flop) if a.kind is ActionKind.BET]
    if not bets or flop[bets[0]].label != aggressor:
        return False, False

    for action in flop[bets[0] + 1 :]:
        if action.label == hand.hero_label:
            return True, action.kind is ActionKind.FOLD
    return False, False


def player_stats(hands: Iterable[CanonicalHand]) -> PlayerStats:
    """Счётчики игрока по списку рук: VPIP, PFR, ре-рейз, сдача на c-bet.

    Раздачи, где героя нет за столом, не считаются вовсе — ни в числителе, ни в
    знаменателе: `hero_label` в такой руке не указывает ни на одно место, и
    любая из формул выше говорила бы о несуществующем игроке.

    Список может быть и всей историей игрока сразу: счётчики складываются по
    рукам, поэтому турнир из 300 раздач весит в среднем ровно в 25 раз больше
    турнира из 12 (`test_hands_from_several_tournaments_are_weighed_by_hands`).
    Усреднить готовые проценты по турнирам дало бы другое число.
    """
    stats = PlayerStats()
    for hand in hands:
        if not _hero_seated(hand):
            continue
        stats.hands += 1
        stats.vpip += _is_vpip(hand)
        stats.pfr += _is_pfr(hand)
        chance, taken = _reraise(hand)
        stats.reraise_chances += chance
        stats.reraise += taken
        faced, folded = _fold_to_cbet(hand)
        stats.cbet_faced += faced
        stats.fold_to_cbet += folded
    return stats
