"""Постфлоп-линия героя: что сыграно, с какой рукой и сколько это требует (спека, §4.3–§4.6).

Спека — `docs/superpowers/specs/2026-10-09-postflop-line-design.md`. Описание
сыгранного, а не вердикт: зоны, цены и лучшего действия здесь нет, и точка,
получившая линию, судимой не становится (`test_the_line_changes_neither_the_sum_nor_the_ranking`).

**Вход — только последовательность действий, карты героя и борд.** Карты
соперника и вскрытие не читаются ни одной функцией
(`test_opponent_cards_and_the_showdown_do_not_move_the_line`); ярлыки «по факту»
(флоат, повторная ставка) читают более поздние действия раздачи — действия, не
карты.

**Агрессор улицы** — последний, кто ставил или повышал на ней; **префлоп-агрессор**
— последний повышавший префлоп, при одних лимпах его нет. **В позиции** — герой
ходит последним на улице среди тех, кто ещё может ходить: не сбросил карты и не
в олл-ине. Порядок хода на постфлопе — `POSITIONS_BY_COUNT` от SB; в хедз-апе
первым ходит BB.

**Ярлык ставки** — первая подошедшая строка таблицы §4.3: c-bet, баррель (§4.4),
повторная ставка, отложенный c-bet, проба, донк, ставка после чека, ставка.
«Ставка после чека» требует, чтобы до героя на улице кто-то чекнул: при ставке
до героя других ходов, кроме чеков, не бывает, и без этого условия строка 8 была
бы недостижима. Ответы: чек-рейз, чек-колл, чек-фолд — герой уже чекнул на этой
улице; рейз, 3-бет, ре-рейз — первое, второе, третье и дальше повышение улицы;
флоат — колл в позиции, после которого на следующей улице все до героя
чекнули, а герой поставил.

**Баррель** — ставка героя на тёрне или ривере, когда на каждой улице от флопа
до предыдущей герой ставил (именно ставка, не рейз), его ставку заколлировали и
рейза на улице не было; тёрн — второй, ривер — третий.

**Размер.** Ставка — `B / P`, рейз — `(повышение сверх колла) / (P + колл)`, где
`P` — `DecisionPoint.pot_before`. Тег — по целым фишкам, без деления.

**Назначение** — по классу силы руки и дро (§4.5). Бэкдоры и оверкарты в
назначение и пороги не входят.

**Пороги** (§4.6) — только у блефа и полублефа. Чистый блеф — `B / (P + B)`, где
`B` — сколько герой добавляет, урезанное стеком соперника: в `B` идёт ставка не
выше стека на улице самого глубокого живого соперника с фишками за спиной (не
`DecisionPoint.eff_stack`: в мультивее за агрессором может сидеть глубже) за
вычетом поставленного героем на улице до неё
(`test_the_thresholds_cut_the_bet_by_the_opponents_stack`,
`test_a_multiway_raise_is_cut_by_the_deepest_opponent_who_can_call`,
`test_a_multiway_raise_is_not_cut_by_a_deeper_opponent_than_the_hero`,
`test_the_cut_takes_the_hero_stack_and_not_the_bet_left_after_his_bet`); размер в
«сыграно» считается от полной ставки. Полублеф — только один на один:
`E = q·(P + B + C) − B`, где `C` — колл соперника (`B` у ставки, `B − колл` у
рейза, оба после урезания), `q` — шанс собрать к концу раздачи, точной дробью;
порог `−E / (P − E)`, при `E ≥ 0` (включая ровно ноль,
`test_a_semibluff_with_zero_expectation_pays_without_folds`) — ноль и признак
«окупается без фолдов». Колл с дро — `C / (P + C)` против шанса на
следующей карте, а если после колла у героя или у соперника не остаётся фишек —
против шанса к риверу; иначе `X = C / q − (P + C)` точной дробью, вверх до
фишки, и сравнение `X` с меньшим из стеков после колла,
`DecisionPoint.eff_stack − (поставлено героем на улице после колла)`: «добрать
столько нельзя» — только при `X` строго больше.

**Альтернативы** (решение владельца 2026-10-10, спека §12) — только у последнего
решения героя в раздаче, если это чек или бет не в олл-ин без ставки перед ним:
порог ставки 50% банка `B / (P + B)` при `B = P / 2` (ровно 1/3; нет, если
эффективный стек не больше половины банка) и олл-ина `S / (P + S)`, где `S` —
`DecisionPoint.eff_stack`.

**Порог опена первым** (`open_threshold`, там же) — префлоп: рейз в неоткрытый
банк окупается сразу при доле фолдов от `R / (P + R)`, где `P` —
`DecisionPoint.pot_before`, `R` — сколько рейз добавляет к блайнду героя (анте
уже в банке), урезанное тем же потолком, что у блефа.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from math import ceil, comb

from harness.analysis.classifier import action_index
from harness.analysis.tools.draws import (
    backdoors,
    draw_missed,
    draw_of,
    overcards,
    showdown_value,
)
from harness.analysis.tools.hand_class import hand_strength
from harness.analysis.tools.pot_odds import required_equity
from harness.contracts import (
    ActionKind,
    ActionTag,
    Backdoor,
    BetAlternatives,
    CanonicalAction,
    CanonicalHand,
    DecisionPoint,
    Draw,
    DrawCall,
    EnrichedHand,
    FoldThreshold,
    HandStrength,
    Line,
    OpenThreshold,
    Overcards,
    PostflopLineDetail,
    Purpose,
    ShowdownValue,
    SizeTag,
    Street,
    StrengthClass,
)
from harness.engine.validation import forced_blind
from harness.normalizer import POSITIONS_BY_COUNT

__all__ = ["open_threshold", "postflop_line"]

_POSTFLOP = (Street.FLOP, Street.TURN, Street.RIVER)
_BOARD_SIZE = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}
_BARREL_NUMBER = {Street.TURN: 2, Street.RIVER: 3}
_HERO_CARDS = 2
_HEADS_UP = 2
_AGGRESSIVE = (ActionKind.BET, ActionKind.RAISE)

# Тег размера: нижняя граница интервала в процентах банка, по убыванию (§4.3).
_SIZE_TAGS = ((100, SizeTag.OVERBET), (75, SizeTag.BIG), (33, SizeTag.STANDARD))

_AGGRESSIVE_PURPOSE = {
    StrengthClass.STRONG: Purpose.VALUE,
    StrengthClass.MEDIUM: Purpose.MEDIUM_HAND,
}
_CALL_PURPOSE = {
    StrengthClass.STRONG: Purpose.CALL_STRONG,
    StrengthClass.MEDIUM: Purpose.BLUFF_CATCH,
}
_THRESHOLD_PURPOSES = (Purpose.BLUFF, Purpose.SEMIBLUFF)


def postflop_line(
    dp: DecisionPoint, en: EnrichedHand, *, fold_proven: bool = False
) -> PostflopLineDetail:
    """Рука, дро, линия и пороги героя в постфлоп-точке.

    `fold_proven` — фолд доказан на ривере (`analysis.river`): колл тогда —
    «ловля блефа», а не «колл с сильной рукой». Точка префлопа — `ValueError`.
    """
    if dp.street not in _POSTFLOP:
        raise ValueError(f"постфлоп-линия строится на флопе, тёрне и ривере, получено {dp.street}")
    hand = en.hand
    index = action_index(hand, dp)
    cards = _Cards.of(hand, dp.street)
    line = _line(dp, en, index, cards, fold_proven=fold_proven)
    purpose = line.purpose if line is not None else None
    return PostflopLineDetail(
        hand=cards.strength,
        draw=cards.draw,
        draw_missed=cards.missed,
        backdoors=cards.backdoors,
        overcards=cards.overcards,
        line=line,
        fold_threshold=(
            _fold_threshold(dp, en, index, cards.draw) if purpose in _THRESHOLD_PURPOSES else None
        ),
        draw_call=(
            _draw_call(dp, cards.draw)
            if purpose is Purpose.CALL_WITH_DRAW and cards.draw is not None
            else None
        ),
        showdown=cards.showdown,
        alternatives=_alternatives(dp, hand, index),
    )


def open_threshold(dp: DecisionPoint, en: EnrichedHand) -> OpenThreshold:
    """Порог опена первым: доля фолдов, при которой рейз окупается сразу.

    `risk` — рейз до `committed_after` за вычетом блайнда героя, урезанный стеком
    самого глубокого соперника, который может уравнять (`_call_cap`); анте в риск
    не входит. Звать для рейза героя в неоткрытый банк — проверку делает
    вызывающий (`preflop.verdict_for`).
    """
    hand = en.hand
    index = action_index(hand, dp)
    taken = hand.actions[index]
    hero = next(player for player in hand.players if player.label == taken.label)
    posted = forced_blind(hand, hero, min(hand.ante, hero.stack))
    cap = _call_cap(hand, index, _live(en, index), posted + dp.to_call)
    risk = min(taken.committed_after, cap) - posted
    pot = dp.pot_before
    return OpenThreshold(risk=risk, pot=pot, fold_share=risk / (pot + risk))


def _alternatives(dp: DecisionPoint, hand: CanonicalHand, index: int) -> BetAlternatives | None:
    """Ставка 50% банка и олл-ин вместо последнего чека или бета героя (§12).

    Только последнее решение героя в раздаче, только чек или бет не в олл-ин и
    без ставки перед ним; ставки 50% нет, если эффективный стек не больше
    половины банка — такая ставка и есть олл-ин.
    """
    taken = hand.actions[index]
    if taken.kind not in (ActionKind.CHECK, ActionKind.BET) or taken.is_all_in or dp.to_call:
        return None
    if any(action.label == taken.label for action in hand.actions[index + 1 :]):
        return None
    pot, stack = dp.pot_before, dp.eff_stack
    if pot <= 0 or stack <= 0:
        return None
    half = Fraction(pot, 2)
    return BetAlternatives(
        pot=pot,
        half_pot=float(half / (pot + half)) if 2 * stack > pot else None,
        all_in_chips=stack,
        all_in=stack / (pot + stack),
    )


class _Cards:
    """Всё, что считается по картам героя и борду улицы; пусто, если карт нет."""

    def __init__(
        self,
        strength: HandStrength | None = None,
        draw: Draw | None = None,
        missed: bool = False,
        backdoors: list[Backdoor] | None = None,
        overcards: Overcards | None = None,
        showdown: ShowdownValue | None = None,
    ) -> None:
        self.strength = strength
        self.draw = draw
        self.missed = missed
        self.backdoors = backdoors or []
        self.overcards = overcards
        self.showdown = showdown

    @classmethod
    def of(cls, hand: CanonicalHand, street: Street) -> _Cards:
        hero = hand.dealt.get(hand.hero_label, [])
        board = [
            card
            for board_street in _POSTFLOP[: _POSTFLOP.index(street) + 1]
            for card in hand.boards.get(board_street, [])
        ]
        if len(hero) != _HERO_CARDS or len(board) != _BOARD_SIZE[street]:
            return cls()
        try:
            return cls(
                strength=hand_strength(hero, board),
                draw=draw_of(hero, board),
                missed=draw_missed(hero, board),
                backdoors=backdoors(hero, board),
                overcards=overcards(hero, board),
                showdown=showdown_value(hero, board),
            )
        except ValueError:
            # Карты не из колоды или повтор карты: считать по ним нечего, а
            # подставить другие значило бы описать чужую руку.
            return cls()


# --- линия ------------------------------------------------------------------------------


def _line(
    dp: DecisionPoint, en: EnrichedHand, index: int, cards: _Cards, *, fold_proven: bool
) -> Line | None:
    hand = en.hand
    taken = hand.actions[index]
    if taken.kind is ActionKind.CHECK:
        return None
    tag, barrel = _action_tag(en, index)
    size_pct: float | None = None
    size_tag: SizeTag | None = None
    if taken.kind in _AGGRESSIVE:
        added = taken.committed_after - _committed_on_street(hand, index, taken.label)
        over_call = added - dp.to_call
        base = dp.pot_before + dp.to_call
        size_pct = over_call / base
        size_tag = _size_tag(over_call, base)
    return Line(
        action=tag,
        barrel=barrel,
        size_pct=size_pct,
        size_tag=size_tag,
        purpose=_purpose(taken.kind, cards, fold_proven=fold_proven),
    )


def _size_tag(chips: int, base: int) -> SizeTag:
    """Тег по полуоткрытым интервалам, сравнением целых фишек: `100·chips` против `pct·base`."""
    for lower_pct, tag in _SIZE_TAGS:
        if 100 * chips >= lower_pct * base:
            return tag
    return SizeTag.BLOCK


def _action_tag(en: EnrichedHand, index: int) -> tuple[ActionTag, int | None]:
    hand = en.hand
    taken = hand.actions[index]
    hero = taken.label
    street = taken.street
    before = [action for action in hand.actions[:index] if action.street is street]
    hero_checked = any(a.label == hero and a.kind is ActionKind.CHECK for a in before)

    if taken.kind is ActionKind.BET:
        return _bet_tag(en, index, before)
    if taken.kind is ActionKind.RAISE:
        if hero_checked:
            return ActionTag.CHECK_RAISE, None
        raises = sum(a.kind is ActionKind.RAISE for a in before)
        return (ActionTag.RAISE, ActionTag.THREE_BET, ActionTag.RERAISE)[min(raises, 2)], None
    if taken.kind is ActionKind.CALL:
        if hero_checked:
            return ActionTag.CHECK_CALL, None
        if _in_position(en, index) and _bet_into_checks_next_street(hand, hero, street):
            return ActionTag.FLOAT, None
        return ActionTag.CALL, None
    return (ActionTag.CHECK_FOLD if hero_checked else ActionTag.FOLD), None


def _bet_tag(
    en: EnrichedHand, index: int, before: Sequence[CanonicalAction]
) -> tuple[ActionTag, int | None]:
    """Строки таблицы §4.3 сверху вниз; первая подошедшая побеждает."""
    hand = en.hand
    hero = hand.actions[index].label
    street = hand.actions[index].street
    previous = _POSTFLOP[_POSTFLOP.index(street) - 1] if street is not Street.FLOP else None
    preflop_aggressor = _last_aggressor(hand, Street.PREFLOP)

    if street is Street.FLOP and preflop_aggressor == hero:
        return ActionTag.CBET, None
    if previous is not None and all(
        _bet_called_unraised(hand, hero, chain_street)
        for chain_street in _POSTFLOP[: _POSTFLOP.index(street)]
    ):
        return ActionTag.BARREL, _BARREL_NUMBER[street]
    if street is Street.RIVER and _bet_called_unraised(hand, hero, Street.TURN):
        return ActionTag.REPEAT_BET, None
    if street is Street.TURN and preflop_aggressor == hero and _checked_through(hand, Street.FLOP):
        return ActionTag.DELAYED_CBET, None
    if (
        previous is not None
        and preflop_aggressor != hero
        and not _in_position(en, index)
        and _checked_through(hand, previous)
    ):
        return ActionTag.PROBE, None
    aggressor = preflop_aggressor if previous is None else _last_aggressor(hand, previous)
    if (
        aggressor is not None
        and aggressor != hero
        and aggressor in _live(en, index)
        and not any(action.label == aggressor for action in before)
    ):
        return ActionTag.DONK, None
    if any(action.kind is ActionKind.CHECK for action in before):
        return ActionTag.BET_AFTER_CHECK, None
    return ActionTag.BET, None


def _purpose(kind: ActionKind, cards: _Cards, *, fold_proven: bool) -> Purpose | None:
    """Назначение по таблице §4.5; `None` у фолда и без карт героя."""
    strength = cards.strength
    if strength is None or kind is ActionKind.FOLD:
        return None
    has_draw = cards.draw is not None
    if kind in _AGGRESSIVE:
        if strength.strength is StrengthClass.WEAK:
            return Purpose.SEMIBLUFF if has_draw else Purpose.BLUFF
        return _AGGRESSIVE_PURPOSE[strength.strength]
    if strength.strength is StrengthClass.WEAK:
        return Purpose.CALL_WITH_DRAW if has_draw else Purpose.BLUFF_CATCH
    if fold_proven:
        return Purpose.BLUFF_CATCH
    return _CALL_PURPOSE[strength.strength]


# --- пороги -----------------------------------------------------------------------------


def _fold_threshold(
    dp: DecisionPoint, en: EnrichedHand, index: int, draw: Draw | None
) -> FoldThreshold:
    hand = en.hand
    taken = hand.actions[index]
    pot = dp.pot_before
    before = _committed_on_street(hand, index, taken.label)
    # Ставка сверх стека соперника вернётся герою: в пороги идёт только та её
    # часть, которую соперник может уравнять. Размер в «сыграно» описывает
    # сыгранное и считается от полной ставки (`_line`).
    bet = min(taken.committed_after, _call_cap(hand, index, _live(en, index), before + dp.to_call))
    bet -= before
    semibluff: float | None = None
    free = False
    if draw is not None and len(_live(en, index)) == _HEADS_UP:
        hit = _exact_hit(draw, to_the_end=True)
        their_call = bet - dp.to_call
        ev = hit * (pot + bet + their_call) - bet
        free = ev >= 0
        semibluff = 0.0 if free else float(-ev / (pot - ev))
    return FoldThreshold(bluff=bet / (pot + bet), semibluff=semibluff, semibluff_free=free)


def _draw_call(dp: DecisionPoint, draw: Draw) -> DrawCall:
    call, pot = dp.to_call, dp.pot_before
    need = required_equity(call, pot)
    # Меньший из стеков героя и соперника после колла: глубина решения против
    # ставящего за вычетом того, что герой поставил на улице вместе с коллом.
    left_after = dp.eff_stack - dp.action.committed_after
    hit = _hit_to_the_end(draw) if left_after <= 0 else draw.hit_next
    if hit >= need:
        return DrawCall(
            required_equity=need,
            hit=hit,
            by_pot_odds=True,
            implied_needed_chips=None,
            beyond_stack=False,
        )
    exact = _exact_hit(draw, to_the_end=left_after <= 0)
    implied = ceil(call / exact - (pot + call))
    return DrawCall(
        required_equity=need,
        hit=hit,
        by_pot_odds=False,
        implied_needed_chips=implied,
        beyond_stack=implied > left_after,
    )


def _exact_hit(draw: Draw, *, to_the_end: bool) -> Fraction:
    """Тот же шанс собрать точной дробью — по формулам `Draw`, из аутов и невидимых.

    Нужен сумме в фишках: `X` округляется вверх, и двоичный хвост дроби мог бы
    поднять его на фишку.
    """
    outs, unseen = len(draw.outs), draw.unseen
    if to_the_end and draw.hit_by_river is not None:
        return 1 - Fraction(comb(unseen - outs, 2), comb(unseen, 2))
    return Fraction(outs, unseen)


def _hit_to_the_end(draw: Draw) -> float:
    """Шанс собрать к риверу: на флопе — двумя картами, на тёрне — одной."""
    return draw.hit_by_river if draw.hit_by_river is not None else draw.hit_next


# --- последовательность действий --------------------------------------------------------


def _street_actions(hand: CanonicalHand, street: Street) -> list[CanonicalAction]:
    return [action for action in hand.actions if action.street is street]


def _last_aggressor(hand: CanonicalHand, street: Street) -> str | None:
    """Последний, кто ставил или повышал на улице; `None` — таких не было."""
    aggressor: str | None = None
    for action in _street_actions(hand, street):
        if action.kind in _AGGRESSIVE:
            aggressor = action.label
    return aggressor


def _checked_through(hand: CanonicalHand, street: Street) -> bool:
    """Улица прошла чеками у всех: действия на ней были, и все — чеки."""
    actions = _street_actions(hand, street)
    return bool(actions) and all(action.kind is ActionKind.CHECK for action in actions)


def _bet_called_unraised(hand: CanonicalHand, label: str, street: Street) -> bool:
    """На улице игрок ставил, его ставку заколлировали, рейза не было."""
    actions = _street_actions(hand, street)
    return (
        any(action.label == label and action.kind is ActionKind.BET for action in actions)
        and any(action.kind is ActionKind.CALL for action in actions)
        and not any(action.kind is ActionKind.RAISE for action in actions)
    )


def _bet_into_checks_next_street(hand: CanonicalHand, hero: str, street: Street) -> bool:
    """На следующей улице все до героя чекнули, и первым действием герой поставил."""
    if street is Street.RIVER:
        return False
    following = _street_actions(hand, _POSTFLOP[_POSTFLOP.index(street) + 1])
    first = next((i for i, action in enumerate(following) if action.label == hero), None)
    if first is None or following[first].kind is not ActionKind.BET:
        return False
    return first > 0 and all(action.kind is ActionKind.CHECK for action in following[:first])


def _committed_on_street(hand: CanonicalHand, index: int, label: str) -> int:
    """Поставлено игроком на улице действия `index` до этого действия."""
    street = hand.actions[index].street
    committed = 0
    for action in hand.actions[:index]:
        if action.street is street and action.label == label:
            committed = action.committed_after
    return committed


def _live(en: EnrichedHand, index: int) -> set[str]:
    """Места в руке перед действием `index`: не сбросили карты и не вычеркнуты движком."""
    hand = en.hand
    folded = {action.label for action in hand.actions[:index] if action.kind is ActionKind.FOLD}
    return {
        player.label
        for player in hand.players
        if player.label not in folded and player.label not in en.report.forfeits
    }


def _invested(hand: CanonicalHand, index: int, label: str) -> tuple[int, int, dict[Street, int]]:
    """Стек игрока, его анте и поставленное по улицам до действия `index`."""
    player = next(p for p in hand.players if p.label == label)
    ante = min(hand.ante, player.stack)
    by_street = {Street.PREFLOP: forced_blind(hand, player, ante)}
    for action in hand.actions[:index]:
        if action.label == label:
            by_street[action.street] = action.committed_after
    return player.stack, ante, by_street


def _all_in_before(hand: CanonicalHand, index: int, label: str) -> bool:
    """Игрок вложил в руку весь стек до действия `index`: анте, блайнд, ставки улиц."""
    stack, ante, by_street = _invested(hand, index, label)
    return ante + sum(by_street.values()) >= stack


def _street_depth(hand: CanonicalHand, index: int, label: str) -> int:
    """Сколько игрок может вложить на улице действия `index` всего: остаток плюс
    уже поставленное на ней — стек на входе в улицу."""
    street = hand.actions[index].street
    stack, ante, by_street = _invested(hand, index, label)
    earlier = sum(chips for s, chips in by_street.items() if s is not street)
    return stack - ante - earlier


def _call_cap(hand: CanonicalHand, index: int, live: set[str], level: int) -> int:
    """Потолок ставки героя, которую соперники могут уравнять (§4.6).

    Самый глубокий живой соперник, у которого перед ходом есть фишки за спиной:
    его остаток плюс поставленное на улице. Блеф окупается, только когда
    сбрасывают все, а проигрыш при колле считается против того, кто может
    заплатить больше всех.
    """
    hero = hand.actions[index].label
    depths = [
        _street_depth(hand, index, label)
        for label in live
        if label != hero and not _all_in_before(hand, index, label)
    ]
    # `level` — поставленное героем до хода плюс колл, уже уравненная часть.
    # Движок не принимает ставку и рейз, когда уравнять их некому, так что на
    # принятой руке потолок выше; `level` лишь не даёт пустому списку упасть.
    return max([level, *depths])


def _in_position(en: EnrichedHand, index: int) -> bool:
    """Герой ходит последним на улице среди тех, кто ещё может ходить."""
    hand = en.hand
    hero = hand.actions[index].label
    order = _postflop_order(len(hand.players))
    position = {player.label: player.position for player in hand.players}
    acting = [
        label
        for label in _live(en, index)
        if label == hero or not _all_in_before(hand, index, label)
    ]
    return max(acting, key=lambda label: order.index(position[label])) == hero


def _postflop_order(players: int) -> list[str]:
    """Порядок хода на постфлопе: от SB; в хедз-апе первым ходит BB."""
    order = POSITIONS_BY_COUNT.get(players)
    if order is None:
        raise ValueError(f"стол на {players} мест: порядка позиций для такого нет")
    return list(reversed(order)) if players == _HEADS_UP else list(order)
