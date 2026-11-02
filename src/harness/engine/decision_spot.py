"""Validate a known preflop prefix and stop before the hero acts."""

from __future__ import annotations

from harness.contracts.decision_spot import CanonicalDecisionSpot, DecisionSnapshot
from harness.normalizer.normalize import POSITIONS_BY_COUNT


class InvalidDecisionSpot(ValueError):
    """A text prefix is inconsistent or leaves an unobserved action."""


def replay_until_decision(spot: CanonicalDecisionSpot) -> DecisionSnapshot:
    positions = POSITIONS_BY_COUNT[spot.table_size]
    if [p.position for p in spot.players] != positions:
        raise InvalidDecisionSpot("Позиции не соответствуют числу мест.")
    if spot.hero_position not in positions or not spot.hero_to_act:
        raise InvalidDecisionSpot("Герой не стоит в названной точке решения.")
    if spot.hero_cards is not None and (spot.hero_cards[0] == spot.hero_cards[1]):
        raise InvalidDecisionSpot("Карта героя повторяется.")
    players = {p.position: p for p in spot.players}
    small = "BTN" if spot.table_size == 2 else "SB"
    committed = dict.fromkeys(positions, 0)
    committed[small] = spot.sb
    committed["BB"] = spot.bb
    ante_paid = {
        p: spot.ante
        if spot.ante_type == "per_player" or (spot.ante_type == "big_blind" and p == "BB")
        else 0
        for p in positions
    }
    for pos in positions:
        stack = players[pos].stack
        if stack is not None and stack < committed[pos] + ante_paid[pos]:
            raise InvalidDecisionSpot(f"Стек {pos} меньше обязательного взноса.")

    # First preflop actor is left of BB; in heads-up BTN/SB acts first.
    queue = positions[2:] + positions[:2] if spot.table_size > 2 else positions.copy()
    highest = spot.bb
    min_raise = spot.bb
    folded: set[str] = set()
    all_in: set[str] = set()
    for action in spot.actions:
        if not queue or action.position != queue[0]:
            expected = queue[0] if queue else "никто"
            raise InvalidDecisionSpot(f"Перед {action.position} должен действовать {expected}.")
        pos = queue.pop(0)
        if pos == spot.hero_position:
            raise InvalidDecisionSpot("Описание уже содержит действие героя.")
        prior = committed[pos]
        total = action.committed_after
        kind = action.kind
        if kind == "fold":
            folded.add(pos)
        elif kind == "check":
            if prior != highest:
                raise InvalidDecisionSpot(f"{pos} не может чекнуть против ставки.")
        elif kind == "call":
            if total is None or total != highest or total <= prior:
                raise InvalidDecisionSpot(f"Колл {pos} не равен требуемой сумме.")
            committed[pos] = total
        else:
            if total is None or total <= highest:
                raise InvalidDecisionSpot(f"Ставка {pos} не повышает текущую.")
            increment = total - highest
            if increment < min_raise:
                raise InvalidDecisionSpot(
                    f"Неполный рейз {pos} требует отдельной проверки переоткрытия торговли."
                    if kind == "all_in" else f"Рейз {pos} ниже минимального размера."
                )
            min_raise = max(increment, min_raise)
            committed[pos] = total
            highest = total
            # Every live, non-all-in player behind the raiser must have a response.
            start = positions.index(pos)
            queue = [
                p
                for p in (positions[start + 1 :] + positions[: start + 1])
                if p != pos and p not in folded and p not in all_in
            ]
        if kind == "all_in":
            all_in.add(pos)
            queue = [p for p in queue if p != pos]
        stack = players[pos].stack
        if stack is not None and committed[pos] + ante_paid[pos] > stack:
            raise InvalidDecisionSpot(f"Ставка {pos} превышает известный стек.")
        if kind == "all_in" and stack is not None and committed[pos] + ante_paid[pos] != stack:
            raise InvalidDecisionSpot(f"Олл-ин {pos} не совпадает с известным стеком.")

    if not queue or queue[0] != spot.hero_position:
        expected = queue[0] if queue else "никто"
        raise InvalidDecisionSpot(f"Перед героем ещё должен действовать {expected}.")
    if all(p == spot.hero_position or p in folded for p in positions):
        raise InvalidDecisionSpot("Все соперники уже сбросили; решения героя нет.")
    to_call = highest - committed[spot.hero_position]
    if to_call < 0:
        raise InvalidDecisionSpot("Вклад героя больше текущей ставки.")
    hero_stack = players[spot.hero_position].stack
    active_stacks = [
        p.stack
        for p in spot.players
        if p.position != spot.hero_position and p.position not in folded
    ]
    known_stacks = [stack for stack in active_stacks if stack is not None]
    effective = min(hero_stack, *known_stacks) if hero_stack is not None and (
        known_stacks and len(known_stacks) == len(active_stacks)
    ) else None
    not_checked = spot.not_checked.copy()
    if effective is None:
        not_checked.append("эффективный стек: не все стеки подтверждены")
    if len(queue) > 1:
        not_checked.append("после героя ещё действуют: " + ", ".join(queue[1:]))
    possible_side_pots = bool(all_in) and (
        len([p for p in spot.players if p.position not in folded]) > 2 or effective is None
    )
    if possible_side_pots:
        not_checked.append("побочные банки и право на них")
    return DecisionSnapshot(
        hero_position=spot.hero_position,
        positions_still_to_act=queue[1:],
        pot_before=sum(committed.values()) + sum(ante_paid.values()),
        to_call=to_call,
        committed=committed,
        effective_stack=effective,
        possible_side_pots=possible_side_pots,
        not_checked=not_checked,
    )
