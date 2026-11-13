"""Apply explicit product defaults to a described hand without changing HH data."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from harness.contracts.natural_hand import (
    NaturalHand,
    ScenarioAction,
    ScenarioHand,
    ScenarioValue,
)
from harness.normalizer.normalize import POSITIONS_BY_COUNT

_SUITS = str.maketrans({"♠": "s", "♥": "h", "♦": "d", "♣": "c"})
_RANKS = "23456789TJQKA"
_SUIT_ORDER = "shdc"
_CARD = re.compile(r"[2-9TJQKA][shdc]", re.IGNORECASE)
_CLASS = re.compile(r"([2-9TJQKA])([2-9TJQKA])([so]?)", re.IGNORECASE)
_STREETS = ("preflop", "flop", "turn", "river")


class InvalidNaturalHand(ValueError):
    """The player's explicit facts cannot form the named decision."""


def _number(value: str | None, *, name: str) -> Decimal:
    if value is None:
        raise InvalidNaturalHand(f"Не указан {name}.")
    try:
        result = Decimal(value.replace(",", ".").strip())
    except InvalidOperation as exc:
        raise InvalidNaturalHand(f"Не понял {name}.") from exc
    if not result.is_finite() or result < 0:
        raise InvalidNaturalHand(f"Неверный {name}.")
    return result


def _value(number: Decimal | int, source: str, note: str = "") -> ScenarioValue:
    return ScenarioValue(value=Decimal(number), source=source, note=note)  # type: ignore[arg-type]


def _cards(raw: NaturalHand) -> tuple[tuple[str, str] | None, str | None, list[str], bool]:
    board_parts = [raw.flop, raw.turn, raw.river]
    provided = [part.translate(_SUITS).replace(" ", "") if part else None for part in board_parts]
    exact_board: list[str] = []
    for part in provided:
        if part:
            exact_board.extend(card[0].upper() + card[1].lower() for card in _CARD.findall(part))
    if len(exact_board) != len(set(exact_board)):
        raise InvalidNaturalHand("На борде одна карта названа дважды.")
    used = set(exact_board)
    hero: tuple[str, str] | None = None
    hero_class: str | None = None
    if raw.hero_cards:
        token = raw.hero_cards.translate(_SUITS).replace(" ", "")
        exact = _CARD.findall(token)
        if len(exact) == 2 and len("".join(exact)) == len(token):
            hero = (exact[0][0].upper() + exact[0][1].lower(),
                    exact[1][0].upper() + exact[1][1].lower())
            if hero[0] == hero[1] or any(card in used for card in hero):
                raise InvalidNaturalHand("Карты героя пересекаются с бордом или повторяются.")
        elif match := _CLASS.fullmatch(token):
            first, second, shape = match.groups()
            first, second, shape = first.upper(), second.upper(), shape.lower()
            if first == second and shape:
                raise InvalidNaturalHand("У карманной пары не бывает suited/offsuit.")
            hero_class = first + second + shape
            choices = [
                (first + s1, second + s2)
                for s1 in _SUIT_ORDER for s2 in _SUIT_ORDER
                if (first != second or s1 != s2)
                and (shape != "s" or s1 == s2)
                and (shape != "o" or s1 != s2)
                and (shape != "" or s1 != s2)
                and first + s1 not in used and second + s2 not in used
            ]
            if not choices:
                raise InvalidNaturalHand("Для карт героя не осталось допустимых мастей.")
            hero = choices[0]
        else:
            raise InvalidNaturalHand("Не понял карты героя.")
        used.update(hero)
    board: list[str] = []
    has_assumed_suits = False
    for part_index, part in enumerate(provided):
        if part is None:
            if any(later is not None for later in provided[part_index + 1:]):
                raise InvalidNaturalHand("Назван тёрн или ривер без предыдущего борда.")
            break
        expected = 3 if part_index == 0 else 1
        tokens = re.findall(r"[2-9TJQKA](?:[shdc])?", part, re.IGNORECASE)
        if len(tokens) != expected or "".join(tokens).lower() != part.lower():
            raise InvalidNaturalHand("Нужны ранги трёх карт флопа и затем тёрн/ривер.")
        used_suits_this_street: set[str] = set()
        for token in tokens:
            rank = token[0].upper()
            if rank not in _RANKS:
                raise InvalidNaturalHand("Неизвестный ранг карты борда.")
            if len(token) == 2:
                card = rank + token[1].lower()
            else:
                has_assumed_suits = True
                options = [rank + suit for suit in _SUIT_ORDER
                           if rank + suit not in used and (
                               part_index != 0 or suit not in used_suits_this_street
                           )]
                if not options:
                    raise InvalidNaturalHand("Нельзя назначить разные карты борда.")
                card = options[0]
            if card in board or (card in used and card not in exact_board):
                raise InvalidNaturalHand("Карта борда повторяется или совпала с картой героя.")
            board.append(card)
            used.add(card)
            used_suits_this_street.add(card[1])
    if len(board) not in (0, 3, 4, 5):
        raise InvalidNaturalHand("Неполный борд.")
    return hero, hero_class, board, has_assumed_suits


def normalize_natural_hand(raw: NaturalHand) -> ScenarioHand:
    """Resolve stated amounts, then defaults, keeping their origin in the result."""
    if not raw.hero_position:
        raise InvalidNaturalHand("На какой позиции вы были?")
    hero = raw.hero_position.upper()
    mentioned = {hero, *(p.position.upper() for p in raw.players),
                 *(a.position.upper() for a in raw.actions)}
    size = raw.table_size or (9 if "UTG+2" in mentioned else 8)
    if size not in POSITIONS_BY_COUNT or not mentioned <= set(POSITIONS_BY_COUNT[size]):
        raise InvalidNaturalHand("Уточните число игроков и позиции за столом.")
    positions = POSITIONS_BY_COUNT[size]
    if raw.bb_chips:
        big_blind = _number(raw.bb_chips, name="большой блайнд")
        if big_blind <= 0:
            raise InvalidNaturalHand("Большой блайнд должен быть положительным.")
        if raw.sb_chips and _number(raw.sb_chips, name="малый блайнд") * 2 != big_blind:
            raise InvalidNaturalHand("Уточните соотношение малого и большого блайндов.")
    else:
        big_blind = Decimal(1)
    if raw.ante is not None and raw.ante_unit == "chips" and raw.bb_chips is None:
        raise InvalidNaturalHand("Анте указано в фишках; нужны блайнды в фишках.")
    ante = (_number(raw.ante, name="анте") / big_blind if raw.ante_unit == "chips"
            else _number(raw.ante, name="анте")) if raw.ante is not None else Decimal("0.125")
    hero_cards, hero_class, board, assumed_suits = _cards(raw)
    assumptions = []
    if raw.table_size is None:
        assumptions.append(f"стол {size} игроков")
    if raw.ante is None:
        assumptions.append("анте 0,125 BB с каждого")
    if hero_class:
        assumptions.append("масти карт героя назначены для сценария")
    if assumed_suits:
        assumptions.append("масти борда назначены для сценария")

    small = "BTN" if size == 2 else "SB"
    committed = dict.fromkeys(positions, Decimal(0))
    committed[small] = Decimal("0.5")
    committed["BB"] = Decimal(1)
    prior_pot = ante * size
    highest = Decimal(1)
    street = "preflop"
    raises_this_street = 0
    actions: list[ScenarioAction] = []
    default_stacks = {p: _value(Decimal(40), "default", "40 BB") for p in positions}
    spent = dict.fromkeys(positions, Decimal(0))
    raw_players = {p.position.upper(): p for p in raw.players}
    for pos, entry in raw_players.items():
        if entry.stack_bb is not None and entry.stack_moment == "before_action":
            default_stacks[pos] = _value(_number(entry.stack_bb, name=f"стек {pos}"), "player")

    for entry in raw.actions:
        pos = entry.position.upper()
        if pos not in committed:
            raise InvalidNaturalHand(f"Позиция {pos} не соответствует столу.")
        target_street = _STREETS.index(entry.street)
        current_street = _STREETS.index(street)
        if target_street < current_street or target_street > current_street + 1:
            raise InvalidNaturalHand("Улицы в описании идут не по порядку.")
        if target_street > current_street:
            prior_pot += sum(committed.values())
            committed = dict.fromkeys(positions, Decimal(0))
            highest = Decimal(0)
            street = entry.street
            raises_this_street = 0
        before = committed[pos]
        amount = None
        provenance = "player"
        if entry.amount is not None:
            amount = _number(entry.amount, name=f"ставка {pos}")
            if entry.amount_unit == "chips":
                amount /= big_blind
            elif entry.amount_unit == "pot_percent":
                amount = (prior_pot + sum(committed.values())) * amount / 100
            elif entry.amount_unit == "times":
                amount *= highest if highest else Decimal(1)
        elif entry.kind == "raise":
            if street != "preflop":
                raise InvalidNaturalHand("Укажите размер постфлоп рейза.")
            if raises_this_street >= 2:
                raise InvalidNaturalHand("Укажите размер 4-бета или следующего рейза.")
            amount = Decimal(2) if highest <= 1 else Decimal(6)
            provenance = "default"
            assumptions.append(f"{pos} {('опен до 2 BB' if highest <= 1 else '3-бет до 6 BB')}")
        elif entry.kind == "bet":
            if street == "preflop":
                raise InvalidNaturalHand("Уточните размер префлоп ставки.")
            amount = (prior_pot + sum(committed.values())) * Decimal("0.33")
            provenance = "default"
            assumptions.append(f"{pos} ставка 33% банка")
        elif entry.kind == "call":
            amount = highest
            provenance = "derived"
        elif entry.kind == "all_in":
            amount = before + default_stacks[pos].value - spent[pos]
            provenance = default_stacks[pos].source
            if provenance == "default":
                assumptions.append(f"{pos} олл-ин по стеку 40 BB")
        if amount is not None and entry.amount_mode == "additional" and entry.amount is not None:
            amount += before
        if entry.kind in ("fold", "check"):
            amount = before
        if amount is None or amount < before:
            raise InvalidNaturalHand(f"Неверный размер действия {pos}.")
        committed[pos] = amount
        spent[pos] += amount - before
        if (entry.kind == "all_in" and entry.amount is not None
                and (pos not in raw_players or raw_players[pos].stack_bb is None)):
            default_stacks[pos] = _value(spent[pos], "derived", "олл-ин")
        highest = max(highest, amount)
        if entry.kind in ("raise", "all_in"):
            raises_this_street += 1
        actions.append(ScenarioAction(street=entry.street, position=pos, kind=entry.kind,
                                      committed_after=amount, all_in=entry.kind == "all_in",
                                      source=provenance, note=entry.quote))
    for pos, entry in raw_players.items():
        if entry.stack_bb is not None and entry.stack_moment == "after_action":
            if not any(a.position == pos for a in actions):
                raise InvalidNaturalHand(f"Не нашёл действие, после которого остался стек {pos}.")
            # spent counts each voluntary contribution once across all streets.
            # Forced blinds are excluded because stack_bb is behind after blinds.
            default_stacks[pos] = _value(_number(entry.stack_bb, name=f"остаток {pos}")
                                         + spent[pos], "derived", "остаток после действия")
    for pos in sorted(mentioned):
        if default_stacks[pos].source == "default":
            assumptions.append(f"{pos} стек 40 BB")
    return ScenarioHand(
        table_size=_value(size, "player" if raw.table_size is not None else "default"),
        ante_bb=_value(ante, "player" if raw.ante is not None else "default"),
        hero_position=hero, hero_cards=hero_cards, hero_class=hero_class,
        board=board, board_source="default" if assumed_suits else "player",
        stacks_behind_bb=default_stacks, actions=actions,
        question=raw.question, context=raw.context,
        assumptions=list(dict.fromkeys(assumptions)),
    )
