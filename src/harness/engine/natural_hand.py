"""Replay a described NLHE prefix and stop before the hero's next action."""

from __future__ import annotations

from decimal import Decimal

from harness.contracts.natural_hand import ScenarioHand, ScenarioSnapshot
from harness.normalizer.normalize import POSITIONS_BY_COUNT

_STREETS = ("preflop", "flop", "turn", "river")


class InvalidScenario(ValueError):
    """An explicit action contradicts the table state."""


def replay_natural_hand(hand: ScenarioHand) -> ScenarioSnapshot:
    """Compute the pot and legal actor from explicit and defaulted contributions."""
    positions = POSITIONS_BY_COUNT[int(hand.table_size.value)]
    if hand.hero_position not in positions:
        raise InvalidScenario("Герой не сидит за названным столом.")
    small = "BTN" if len(positions) == 2 else "SB"
    forced = dict.fromkeys(positions, Decimal(0))
    forced[small] = Decimal("0.5")
    forced["BB"] = Decimal(1)
    committed = forced.copy()
    remaining = {p: hand.stacks_behind_bb[p].value for p in positions}
    pot_prior = hand.ante_bb.value * len(positions)
    highest = Decimal(1)
    min_raise = Decimal(1)
    street = "preflop"
    folded: set[str] = set()
    all_in: set[str] = set()
    assumed_folds: list[str] = []
    queue = positions[2:] + positions[:2] if len(positions) > 2 else positions.copy()

    def postflop_queue() -> list[str]:
        return [p for p in positions if p not in folded and p not in all_in]

    def advance(target: str) -> None:
        nonlocal street, committed, pot_prior, highest, min_raise, queue
        while street != target:
            if queue:
                raise InvalidScenario(f"Перед {target} не закончены действия на {street}.")
            pot_prior += sum(committed.values())
            committed = dict.fromkeys(positions, Decimal(0))
            highest = Decimal(0)
            min_raise = Decimal(1)
            street = _STREETS[_STREETS.index(street) + 1]
            queue = postflop_queue()

    for action in hand.actions:
        if _STREETS.index(action.street) < _STREETS.index(street):
            raise InvalidScenario("Улицы идут не по порядку.")
        advance(action.street)
        while queue and queue[0] != action.position and queue[0] != hand.hero_position:
            skipped = queue.pop(0)
            folded.add(skipped)
            assumed_folds.append(skipped)
        if not queue or queue[0] != action.position:
            raise InvalidScenario(f"Перед {action.position} должен действовать "
                                  f"{queue[0] if queue else 'другой игрок'}.")
        queue.pop(0)
        pos = action.position
        before = committed[pos]
        total = action.committed_after
        kind = action.kind
        if kind == "fold":
            if pos == hand.hero_position and len(hand.actions) == 1:
                raise InvalidScenario("После паса героя нет его следующего решения.")
            folded.add(pos)
        elif kind == "check":
            if before != highest:
                raise InvalidScenario(f"{pos} не может чекнуть против ставки.")
        elif kind == "call":
            if total != highest or total <= before:
                raise InvalidScenario(f"Колл {pos} не равен требуемой сумме.")
        else:
            if total <= highest:
                raise InvalidScenario(f"Ставка {pos} не повышает текущую.")
            increment = total - highest
            if kind != "all_in" and increment < min_raise:
                raise InvalidScenario(f"Рейз {pos} меньше минимального.")
            min_raise = max(min_raise, increment)
            highest = total
            start = positions.index(pos)
            queue = [p for p in positions[start + 1:] + positions[:start + 1]
                     if p != pos and p not in folded and p not in all_in]
        delta = total - before
        if delta < 0 or delta > remaining[pos]:
            raise InvalidScenario(f"У {pos} не хватает стека на действие.")
        remaining[pos] -= delta
        committed[pos] = total
        if kind == "all_in":
            if remaining[pos] != 0:
                raise InvalidScenario(f"Олл-ин {pos} не совпал со стеком.")
            all_in.add(pos)
            queue = [p for p in queue if p != pos]
        if kind == "fold":
            queue = [p for p in queue if p != pos]

    target_street = ("flop" if len(hand.board) == 3 else
                     "turn" if len(hand.board) == 4 else
                     "river" if len(hand.board) == 5 else "preflop")
    if _STREETS.index(street) > _STREETS.index(target_street):
        raise InvalidScenario("Действие указано после последней карты борда.")
    advance(target_street)
    while queue and queue[0] != hand.hero_position:
        skipped = queue.pop(0)
        if street != "preflop":
            raise InvalidScenario(f"До решения героя на {street} должен действовать {skipped}.")
        folded.add(skipped)
        assumed_folds.append(skipped)
    if not queue or hand.hero_position in folded or hand.hero_position in all_in:
        raise InvalidScenario("В описанной точке герой не может действовать.")
    live = [p for p in positions if p not in folded]
    if len(live) < 2:
        raise InvalidScenario("Соперники уже сбросили; решения героя нет.")
    return ScenarioSnapshot(
        street=street,
        pot_before_bb=pot_prior + sum(committed.values()),
        to_call_bb=highest - committed[hand.hero_position],
        committed_bb=committed,
        remaining_bb=remaining,
        live_positions=live,
        positions_after_hero=queue[1:],
        assumed_folds=assumed_folds,
        possible_side_pots=bool(all_in) and len(live) > 2,
    )
