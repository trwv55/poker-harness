"""Normalize only evidenced facts of a text decision; keep unknowns unknown."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from harness.contracts.decision_spot import (
    CanonicalDecisionSpot,
    RawDecisionSpot,
    RawSpotPlayer,
    SpotAction,
    SpotFact,
    SpotPlayer,
)
from harness.normalizer.normalize import POSITIONS_BY_COUNT

_CARD = re.compile(r"[2-9TJQKA][shdc]", re.IGNORECASE)
_SUITS = str.maketrans({"♠": "s", "♥": "h", "♦": "d", "♣": "c"})
_CLASS = re.compile(r"(?:[2-9TJQKA]){2}[so]?", re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_BLIND_PAIR = re.compile(
    r"(?:blinds?|блайнд\w*)\s+(\d+(?:[.,]\d+)?)\s*[/:-]\s*(\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)
_ANTE_NONE = re.compile(r"без анте|no ante", re.IGNORECASE)
_ACTION_WORDS = {
    "fold": re.compile(r"fold|фолд|пас|сброс|скин", re.IGNORECASE),
    "check": re.compile(r"check|чек", re.IGNORECASE),
    "call": re.compile(r"call|колл|уравн|достав", re.IGNORECASE),
    "raise": re.compile(r"raise|рейз|откр|opens|повыс|3.?бет|постав|ставит", re.IGNORECASE),
    "all_in": re.compile(r"all.?in|олл.?ин|пуш|shove|jam", re.IGNORECASE),
}
_TOTAL = re.compile(r"\bto\b|\btotal\b|до\s+\d|итого|всего|общая ставка", re.IGNORECASE)
_ADDITIONAL = re.compile(r"доплат|добав|ещ[её]|\bmore\b|\badditional\b", re.IGNORECASE)
_BEHIND = re.compile(r"за спиной|behind|остал", re.IGNORECASE)
_POSITION_WORDS = {
    "BTN": ("btn", "баттон", "кнопк"),
    "SB": ("sb", "малый блайнд", "мб"),
    "BB": ("bb", "большой блайнд", "бб"),
    "CO": ("co", "катофф", "cutoff"),
}
_RUSSIAN_POSITION_RE = {
    "BTN": r"баттон\w*|кнопк\w*",
    "SB": r"мал\w*\s+блайнд\w*",
    "BB": r"больш\w*\s+блайнд\w*",
    "CO": r"катофф\w*",
}


def _position_in_quote(position: str, quote: str) -> bool:
    if (pattern := _RUSSIAN_POSITION_RE.get(position)) and re.search(
        pattern, quote, re.IGNORECASE
    ):
        return True
    for word in _POSITION_WORDS.get(position, (position.lower(),)):
        for match in re.finditer(r"(?<!\w)" + re.escape(word) + r"(?!\w)", quote, re.IGNORECASE):
            # In "CO opens 2.5 bb", bb is a unit, not evidence of a BB action.
            if word in {"bb", "бб"} and re.search(
                r"\d+(?:[.,]\d+)?\s*$", quote[: match.start()]
            ):
                continue
            return True
    return False


def _cards_after_position(position: str, cards: str, source: str) -> bool:
    cards = cards.translate(_SUITS)
    source = source.translate(_SUITS)
    aliases = [
        r"(?<!\w)" + re.escape(word) + r"(?!\w)"
        for word in _POSITION_WORDS.get(position, (position.lower(),))
    ]
    if russian := _RUSSIAN_POSITION_RE.get(position):
        aliases.append(r"(?:" + russian + r")")
    return any(
        re.search(alias + r".{0,24}" + re.escape(cards) + r"(?!\w)", source, re.IGNORECASE)
        for alias in aliases
    )


def _action_kind_in_quote(kind: str, quote: str) -> bool:
    present = {name for name, pattern in _ACTION_WORDS.items() if pattern.search(quote)}
    return present == {kind}


def _amount_kind_in_quote(kind: str, quote: str) -> bool:
    observed = {
        "total": bool(_TOTAL.search(quote)),
        "additional": bool(_ADDITIONAL.search(quote)),
        "behind": bool(_BEHIND.search(quote)),
    }
    if kind == "total" and not any(observed.values()):
        return True  # A bare raise size is the total on this street.
    return observed.get(kind, False) and sum(observed.values()) == 1


def _blinds_match_quotes(raw: RawDecisionSpot, sb: int, bb: int) -> bool:
    quotes = (raw.sb.quote, raw.bb.quote)
    pairs = [match for quote in quotes if (match := _BLIND_PAIR.search(quote))]
    if pairs:
        return all(
            Decimal(match.group(1).replace(",", ".")) == sb
            and Decimal(match.group(2).replace(",", ".")) == bb
            for match in pairs
        )
    return bool(
        re.search(
            r"(?:\bSB\b|мал\w*\s+блайнд\w*)\D*" + str(sb) + r"\b",
            raw.sb.quote,
            re.IGNORECASE,
        )
        and re.search(
            r"(?:\bBB\b|больш\w*\s+блайнд\w*)\D*" + str(bb) + r"\b",
            raw.bb.quote,
            re.IGNORECASE,
        )
    )


def _blinds_from_source(source: str) -> tuple[int, int] | None:
    """Recover one literal blind pair when extraction splits it incorrectly."""
    pairs = {
        (Decimal(match.group(1).replace(",", ".")), Decimal(match.group(2).replace(",", ".")))
        for match in _BLIND_PAIR.finditer(source)
    }
    if len(pairs) != 1:
        return None
    small, big = pairs.pop()
    if small != int(small) or big != int(big):
        return None
    return int(small), int(big)


def _action_amount_matches_quote(amount: SpotFact, kind: str, quote: str) -> bool:
    token = _NUMBER.search(amount.value or "")
    if token is None:
        return False
    number = token.group().replace(",", ".")
    lead = (
        r"(?:\bto\b|\btotal\b|до|итого|всего)"
        if kind == "total"
        else r"(?:доплат\w*|добав\w*|ещ[её]|\bmore\b|\badditional\b)"
    )
    pattern = re.compile(lead + r"\s*(\d+(?:[.,]\d+)?)", re.IGNORECASE)
    if any(match.group(1).replace(",", ".") == number for match in pattern.finditer(quote)):
        return True
    numbers = [match.group().replace(",", ".") for match in _NUMBER.finditer(quote)]
    return kind == "total" and numbers == [number]


def _evidenced(fact: SpotFact, text: str) -> bool:
    if fact.status != "known" or not fact.value or not fact.quote:
        return False
    return fact.quote.casefold() in text.casefold()


def _chips(fact: SpotFact, text: str, bb: int | None = None) -> int | None:
    if not _evidenced(fact, text):
        return None
    token = _NUMBER.search(fact.value or "")
    if token is None:
        return None
    number = token.group().replace(",", ".")
    # Numeric values must be present in the quoted source; the model may not invent sizes.
    if number not in [match.group().replace(",", ".") for match in _NUMBER.finditer(fact.quote)]:
        return None
    try:
        amount = Decimal(number)
    except InvalidOperation:
        return None
    unit = (fact.value or "").lower()
    bb_unit = r"(?<![a-z])bb\b|бб|блайнд"
    if re.search(bb_unit, unit):
        if not re.search(bb_unit, fact.quote.lower()):
            return None
        if bb is None:
            return None
        amount *= bb
    if amount != amount.to_integral_value() or amount < 0:
        return None
    return int(amount)


def _cards(raw: RawDecisionSpot) -> tuple[tuple[str, str] | None, str | None]:
    fact = raw.hero_cards
    if not _evidenced(fact, raw.source_text):
        return None, None
    value = (fact.value or "").translate(_SUITS).replace(" ", "")
    quote = fact.quote.translate(_SUITS)
    cards = _CARD.findall(value)
    if len(cards) == 2 and "".join(cards).lower() == value.lower():
        if any(card.lower() not in quote.lower() for card in cards):
            return None, None
        return (
            cards[0][0].upper() + cards[0][1].lower(),
            cards[1][0].upper() + cards[1][1].lower(),
        ), None
    hand_class = _CLASS.fullmatch(value)
    if hand_class:
        value = hand_class.group()
        if value.lower() not in quote.lower():
            return None, None
        return None, value[:2].upper() + value[2:].lower()
    return None, None


def normalize_decision_spot(raw: RawDecisionSpot) -> tuple[CanonicalDecisionSpot | None, list[str]]:
    """Return a verified prefix or concrete questions. No guessed ante, sizes or stacks."""
    questions: list[str] = []
    source = raw.source_text
    size = _chips(raw.table_size, source)
    if size not in POSITIONS_BY_COUNT:
        questions.append("Сколько игроков было за столом (от 2 до 9)?")
        return None, questions
    positions = POSITIONS_BY_COUNT[size]
    sb = _chips(raw.sb, source)
    bb = _chips(raw.bb, source)
    pair = _blinds_from_source(source)
    if pair is not None:
        if sb is None:
            sb = pair[0]
        if bb is None or "/" in (raw.bb.value or ""):
            bb = pair[1]
    if sb is None or bb is None or sb <= 0 or bb <= sb:
        questions.append("Укажите малый и большой блайнды в фишках.")
    hero = (raw.hero_position.value or "").upper() if _evidenced(raw.hero_position, source) else ""
    explicit_hero = re.search(
        r"геро|у меня|\bя\b|\bты\b|hero", raw.hero_position.quote, re.IGNORECASE
    )
    implicit_hero = (
        re.search(r"что делать|как сыграть|какое решение", source, re.IGNORECASE)
        and raw.hero_cards.status == "known"
        and bool(raw.hero_cards.value)
        and _cards_after_position(hero, raw.hero_cards.value or "", source)
    )
    if (hero not in positions or not _position_in_quote(hero, raw.hero_position.quote)
            or not (explicit_hero or implicit_hero)):
        questions.append("На какой позиции герой?")
    cards, hand_class = _cards(raw)
    if cards is None and hand_class is None:
        questions.append("Какие две карты у героя? Можно указать класс руки, например 99.")
    if not _evidenced(raw.ante, source):
        questions.append("Укажите размер анте или напишите «без анте».")
    if questions:
        return None, questions[:2]
    assert sb is not None and bb is not None
    assert hero in positions
    if not (pair == (sb, bb) or _blinds_match_quotes(raw, sb, bb)):
        return None, ["Уточните, какая сумма относится к малому и большому блайнду."]
    ante_quote = raw.ante.quote
    if _ANTE_NONE.search(ante_quote):
        ante = 0
    else:
        ante = _chips(raw.ante, source, bb)
    if ante is None:
        return None, ["Укажите размер анте в фишках или напишите «без анте»."]
    ante_type = "per_player" if ante > 0 else "none"
    if cards is not None and (
        cards[0] == cards[1]
        or any(c[0] not in "23456789TJQKA" or c[1] not in "shdc" for c in cards)
    ):
        return None, ["Уточните карты героя: две разные карты с мастями."]

    players = [SpotPlayer(position=p) for p in positions]
    by_pos = {p.position: p for p in players}
    entries = [entry.model_copy(deep=True) for entry in raw.players]
    if _evidenced(raw.hero_stack, source):
        hero_entry = next((entry for entry in entries if entry.position.upper() == hero), None)
        if hero_entry is None:
            entries.append(RawSpotPlayer(position=hero, stack=raw.hero_stack))
        elif not _evidenced(hero_entry.stack, source):
            hero_entry.stack = raw.hero_stack
    for entry in entries:
        pos = entry.position.upper()
        if pos not in by_pos:
            return None, [f"Позиция {pos} не соответствует столу на {size} игроков."]
        if entry.stack.status != "known":
            continue
        stack = _chips(entry.stack, source, bb)
        if stack is None:
            return None, [f"Уточните стек {pos} и единицы измерения."]
        # Described stacks are chips behind after forced posts, before voluntary
        # actions. A "behind" quote for someone who already acted is ambiguous.
        if entry.stack_moment == "behind" and any(
            a.position.upper() == pos for a in raw.actions
        ):
            continue
        if re.search(r"~|≈|около|примерно|приблизительно|\babout\b", entry.stack.quote, re.IGNORECASE):
            by_pos[pos].approx_stack_behind = stack
            continue
        blind = sb if pos == ("BTN" if size == 2 else "SB") else bb if pos == "BB" else 0
        by_pos[pos].stack = stack + ante + blind

    committed = {p: 0 for p in positions}
    committed["BTN" if size == 2 else "SB"] = sb
    committed["BB"] = bb
    actions: list[SpotAction] = []
    for action in raw.actions:
        pos = action.position.upper()
        if (
            pos not in by_pos
            or not action.quote
            or action.quote.casefold() not in source.casefold()
            or not _position_in_quote(pos, action.quote)
        ):
            return None, ["Уточните позицию и порядок действий перед героем."]
        if not _action_kind_in_quote(action.kind, action.quote):
            return None, [f"Что именно сделал {pos}: пас, колл, рейз или олл-ин?"]
        amount: int | None = None
        if action.kind not in ("fold", "check"):
            amount = _chips(action.amount, source, bb)
            if amount is None:
                return None, [f"Уточните размер действия {pos} в фишках."]
            amount_kind = "total" if action.amount_kind == "unspecified" else action.amount_kind
            if action.kind in ("raise", "all_in") and amount_kind == "behind":
                return None, [f"Укажите итоговую ставку {pos} на улице."]
            if not _amount_kind_in_quote(amount_kind, action.quote):
                return None, [f"Сумма действия {pos} указана как доплата или как итог на улице?"]
            if not _action_amount_matches_quote(action.amount, amount_kind, action.quote):
                return None, [f"Уточните сумму действия {pos}: число не связано с действием."]
            if amount_kind == "additional":
                amount += committed[pos]
            if amount < committed[pos]:
                return None, [f"Сумма действия {pos} меньше уже внесённого."]
            committed[pos] = amount
        actions.append(
            SpotAction(
                position=pos,
                kind=action.kind,
                committed_after=amount,
                all_in=action.kind == "all_in",
                quote=action.quote,
            )
        )
    supplied_stacks = {
        entry.position.upper() for entry in entries
        if _evidenced(entry.stack, source)
    }
    required_stacks = {hero} | {
        action.position.upper() for action in raw.actions if action.kind not in {"fold", "check"}
    }
    missing_stacks = sorted(required_stacks - supplied_stacks)
    if missing_stacks:
        return None, ["Укажите примерные стеки до действий: " + ", ".join(missing_stacks) + "."]
    unknown_action = next(
        (line for line in raw.unknown_lines if re.search(
            r"лимп|бет|ставк|рейз|колл|пас|фолд|пуш|олл.?ин|raise|call|fold|shove|check",
            line, re.IGNORECASE,
        )),
        None,
    )
    if unknown_action:
        return None, ["Не понял действие в описании: " + unknown_action[:100]]
    return CanonicalDecisionSpot(
        table_size=size,
        sb=sb,
        bb=bb,
        ante=ante,
        ante_type=ante_type,
        hero_position=hero,
        hero_cards=cards,
        hero_class=hand_class,
        players=players,
        actions=actions,
        tournament_context=raw.tournament_context.quote if _evidenced(raw.tournament_context, source) else None,
        icm_context=raw.icm_context.quote if _evidenced(raw.icm_context, source) else None,
        ko_context=raw.ko_context.quote if _evidenced(raw.ko_context, source) else None,
    ), []
