"""Normalize only evidenced facts of a text decision; keep unknowns unknown."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from harness.contracts.decision_spot import (
    CanonicalDecisionSpot,
    RawDecisionSpot,
    SpotAction,
    SpotFact,
    SpotPlayer,
)
from harness.normalizer.normalize import POSITIONS_BY_COUNT

_CARD = re.compile(r"[2-9TJQKA][shdc]", re.IGNORECASE)
_CLASS = re.compile(r"(?:[2-9TJQKA]){2}[so]?", re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_BLIND_PAIR = re.compile(
    r"(?:blinds?|блайнд\w*)\s+(\d+(?:[.,]\d+)?)\s*[/:-]\s*(\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)
_ANTE_EACH = re.compile(r"кажд|все|each|per.?player|all players|everyone", re.IGNORECASE)
_ANTE_BB = re.compile(r"\bBB\s+ante\b|big.?blind\s+ante|анте\s+(?:с\s+)?(?:большого\s+блайнда|бб)|бб\s+анте", re.IGNORECASE)
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
_STACK_MOMENTS = {
    "before_ante": re.compile(r"до анте|before ante|до начала раздачи|starting stack", re.IGNORECASE),
    "after_ante": re.compile(r"после анте|after ante", re.IGNORECASE),
    "after_post": re.compile(r"после (?:поста|блайнд)|after (?:posting|post|blind)", re.IGNORECASE),
    "behind": _BEHIND,
}
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
    return any(match.group(1).replace(",", ".") == number for match in pattern.finditer(quote))


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
    if re.search(r"\bbb\b|бб|блайнд", unit):
        if not re.search(r"\bbb\b|бб|блайнд", fact.quote.lower()):
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
    value = (fact.value or "").replace(" ", "")
    cards = _CARD.findall(value)
    if len(cards) == 2 and "".join(cards).lower() == value.lower():
        if any(card.lower() not in fact.quote.lower() for card in cards):
            return None, None
        return (
            cards[0][0].upper() + cards[0][1].lower(),
            cards[1][0].upper() + cards[1][1].lower(),
        ), None
    hand_class = _CLASS.fullmatch(value)
    if hand_class:
        value = hand_class.group()
        if value.lower() not in fact.quote.lower():
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
    if sb is None or bb is None or sb <= 0 or bb <= sb:
        questions.append("Укажите малый и большой блайнды в фишках.")
    hero = (raw.hero_position.value or "").upper() if _evidenced(raw.hero_position, source) else ""
    explicit_hero = re.search(
        r"геро|у меня|\bя\b|hero", raw.hero_position.quote, re.IGNORECASE
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
    if raw.ante_type == "unknown" or not _evidenced(raw.ante, source):
        questions.append("Было ли анте? Если да, укажите размер и кто его ставил.")
    if questions:
        return None, questions[:2]
    assert sb is not None and bb is not None
    assert hero in positions
    assert raw.ante_type != "unknown"
    if not _blinds_match_quotes(raw, sb, bb):
        return None, ["Уточните, какая сумма относится к малому и большому блайнду."]
    ante_quote = raw.ante.quote
    if raw.ante_type == "none" and _ANTE_NONE.search(ante_quote):
        ante = 0
    else:
        ante = _chips(raw.ante, source, bb)
    if ante is None or (raw.ante_type == "none" and ante != 0):
        return None, ["Уточните точный размер и тип анте в фишках."]
    if (
        raw.ante_type == "per_player"
        and ante > 0
        and (not _ANTE_EACH.search(ante_quote) or _ANTE_BB.search(ante_quote))
        or raw.ante_type == "big_blind"
        and ante > 0
        and (not _ANTE_BB.search(ante_quote) or _ANTE_EACH.search(ante_quote))
        or raw.ante_type == "none"
        and not _ANTE_NONE.search(ante_quote)
    ):
        return None, ["Анте платил каждый игрок, только большой блайнд или анте не было?"]
    known_stacks = {p.position.upper() for p in raw.players if p.stack.status == "known"}
    if ante > 0 and raw.ante_type == "per_player" and any(
        position not in known_stacks for position in positions
    ) and not _ANTE_EACH.search(ante_quote):
        return None, ["Все игроки внесли полное анте, без короткого олл-ин анте?"]
    if cards is not None and (
        cards[0] == cards[1]
        or any(c[0] not in "23456789TJQKA" or c[1] not in "shdc" for c in cards)
    ):
        return None, ["Уточните карты героя: две разные карты с мастями."]

    players = [SpotPlayer(position=p) for p in positions]
    by_pos = {p.position: p for p in players}
    for entry in raw.players:
        pos = entry.position.upper()
        if pos not in by_pos:
            return None, [f"Позиция {pos} не соответствует столу на {size} игроков."]
        if entry.stack.status != "known":
            continue
        stack = _chips(entry.stack, source, bb)
        if stack is None:
            return None, [f"Уточните стек {pos} и единицы измерения."]
        moment = _STACK_MOMENTS.get(entry.stack_moment)
        observed_moments = {
            name for name, pattern in _STACK_MOMENTS.items() if pattern.search(entry.stack.quote)
        }
        if moment is None or observed_moments != {entry.stack_moment}:
            return None, [f"Когда измерен стек {pos}: до анте или за спиной?"]
        if entry.stack_moment == "before_ante":
            initial = stack
        elif entry.stack_moment in ("after_ante", "after_post", "behind"):
            # This conversion is safe only before voluntary action.
            if any(a.position.upper() == pos for a in raw.actions):
                return None, [f"Стек {pos} указан после действия; нужен стек до раздачи."]
            initial = stack + (ante if raw.ante_type == "per_player" else 0)
            if raw.ante_type == "big_blind" and pos == "BB":
                initial += ante
            if entry.stack_moment in ("after_post", "behind"):
                initial += (
                    sb if pos in ("BTN" if size == 2 else "SB",) else bb if pos == "BB" else 0
                )
        else:
            return None, [f"Когда измерен стек {pos}: до анте или за спиной?"]
        by_pos[pos].stack = initial

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
            if action.amount_kind == "unspecified" or (
                action.kind in ("raise", "all_in") and action.amount_kind == "behind"
            ):
                return None, [f"Сумма действия {pos} — доплата или итоговая ставка?"]
            if not _amount_kind_in_quote(action.amount_kind, action.quote):
                return None, [f"Сумма действия {pos} указана как доплата или как итог на улице?"]
            if not _action_amount_matches_quote(action.amount, action.amount_kind, action.quote):
                return None, [f"Уточните сумму действия {pos}: число не связано с действием."]
            if action.amount_kind == "additional":
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
    if raw.unknown_lines:
        return None, ["Не понял часть описания: " + raw.unknown_lines[0][:100]]
    return CanonicalDecisionSpot(
        table_size=size,
        sb=sb,
        bb=bb,
        ante=ante,
        ante_type=raw.ante_type,
        hero_position=hero,
        hero_cards=cards,
        hero_class=hand_class,
        players=players,
        actions=actions,
    ), []
