"""Read one natural-language hand or one correction with a typed model output."""

from __future__ import annotations

import json
import re
from pathlib import Path

from harness.contracts.natural_hand import NaturalHand, NaturalPlayer, NaturalRead
from harness.platform.llm import LLM

MAX_NATURAL_TEXT = 4000
_HAND_CUE = re.compile(
    r"флоп|т[её]рн|ривер|раздач|рук[уае]\b|пуш|олл.?ин|3.?бет|"
    r"колл|рейз|откр|сфолд|поставил|чекн|есть колл|"
    r"\b(?:flop|turn|river|opens?|shove|calls?|raises?|folds?)\b",
    re.IGNORECASE,
)
_POSITION = re.compile(r"\b(?:UTG(?:\+[12])?|LJ|HJ|CO|BTN|SB|BB)\b", re.IGNORECASE)
_HOLE_CARDS = re.compile(r"\b[2-9TJQKA]{2}[so]?\b|[2-9TJQKA][♠♥♦♣][2-9TJQKA][♠♥♦♣]", re.IGNORECASE)
_STATS_QUESTION = re.compile(r"частот|статист|\bVPIP\b|\bPFR\b|диапазон|\brange\b", re.IGNORECASE)
_STRONG_HAND_CUE = re.compile(
    r"раздач|разбер\w*\s+рук|флоп|т[её]рн|ривер|"
    r"\b(?:flop|turn|river|opens?|shove|raises?|folds?)\b|откр|пуш|олл.?ин|3.?бет",
    re.IGNORECASE,
)
_BLINDS = re.compile(
    r"(?:блайнд\w*|blinds?)\s+(\d+(?:[.,]\d+)?)\s*[/:-]\s*(\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)
_STACK = re.compile(r"(?:за спиной|стек(?:ом)?|stack)\s*(\d+(?:[.,]\d+)?)\s*(?:ББ|BB)", re.IGNORECASE)
_TABLE_SIZE = re.compile(r"\b([2-9])\s*(?:игрок\w*|мест\w*|players?|handed)\b", re.IGNORECASE)
_ANTE = re.compile(
    r"(?:анте|\bante\b)\s*(\d+(?:[.,]\d+)?)\s*(ББ|BB)?",
    re.IGNORECASE,
)
_POSITION_STACK = re.compile(
    r"\b(UTG(?:\+[12])?|LJ|HJ|CO|BTN|SB|BB)\b\s*(\d+(?:[.,]\d+)?)\s*(?:ББ|BB)\b",
    re.IGNORECASE,
)
_HERO_STACK_AFTER_CARDS = re.compile(
    r"\b(?:with|с)\s*[2-9TJQKA]{2}[so]?\s+(?:and|и)\s*(\d+(?:[.,]\d+)?)\s*(?:ББ|BB)\b",
    re.IGNORECASE,
)
_SUITS = str.maketrans({"♠": "s", "♥": "h", "♦": "d", "♣": "c"})
_ACTION_WORDS = {
    "fold": re.compile(r"фолд|пас|сброс|\bfold", re.IGNORECASE),
    "check": re.compile(r"чек|\bcheck", re.IGNORECASE),
    "call": re.compile(r"колл|уравн|достав|\bcall", re.IGNORECASE),
    "bet": re.compile(r"став|бет|\bbet", re.IGNORECASE),
    "raise": re.compile(r"откр|рейз|повыс|3.?бет|4.?бет|\braise|\bopen", re.IGNORECASE),
    "all_in": re.compile(r"олл.?ин|пуш|\ball.?in|\bshove|\bjam", re.IGNORECASE),
}
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


class InvalidNaturalReading(ValueError):
    """A cited model fact contradicts the quote in the user's message."""


def may_describe_hand(text: str, *, has_active: bool = False) -> bool:
    """Broad cost guard: model decides intent; this guard is not a spot verdict."""
    if not text.strip() or len(text) > MAX_NATURAL_TEXT:
        return False
    if (_STATS_QUESTION.search(text) and not _STRONG_HAND_CUE.search(text)
            and not (_POSITION.search(text) and _HOLE_CARDS.search(text))
            and not has_active):
        return False
    return bool(_HAND_CUE.search(text) or (_POSITION.search(text) and _HOLE_CARDS.search(text))
                or (has_active and (
        _POSITION.search(text) or re.search(r"\d|маст|стек|борд|на самом деле", text, re.IGNORECASE)
    )))


_PROMPT_PATH = Path(__file__).parent / "prompts" / "natural_hand.md"


def _supported_quote(quote: str, source: str) -> bool:
    return bool(quote and quote.casefold() in source.casefold())


def _literal_or_quote(value: str | None, quote: str, source: str) -> bool:
    if not value:
        return False
    normalized = source.translate(_SUITS).casefold()
    return value.casefold() in normalized and (
        not quote or _supported_quote(quote, source)
    )


async def read_natural_hand(
    llm: LLM, text: str, trace_id: int, *, active: NaturalHand | None = None,
) -> NaturalRead:
    """One extraction call; cited facts that do not occur in input are discarded."""
    context = ""
    if active is not None:
        context = ("Активный сценарий (для определения, это поправка или новая рука):\n"
                   + json.dumps(active.model_dump(exclude={"source_text"}), ensure_ascii=False)
                   + "\n\nНовое сообщение игрока:\n")
    result, _ = await llm(
        "spot_extract", NaturalRead, prompt=context + text,
        instructions=_PROMPT_PATH.read_text(encoding="utf-8"), trace_id=trace_id,
    )
    if result.intent == "other" or result.hand is None:
        return NaturalRead(intent="other")
    hand = result.hand.model_copy(deep=True)
    # Only the new message may evidence a changed field. Old facts are merged
    # later from the stored scenario, so a correction cannot steal old quotes.
    if (hand.hero_position or hand.hero_cards) and not (
        _supported_quote(hand.hero_quote, text) or
        (_literal_or_quote(hand.hero_position, "", text) and
         (hand.hero_cards is None or _literal_or_quote(hand.hero_cards, "", text)))
    ):
        hand.hero_position = None
        hand.hero_cards = None
    if hand.hero_cards and not _literal_or_quote(hand.hero_cards, "", text):
        hand.hero_cards = None
    for name in ("flop", "turn", "river"):
        if getattr(hand, name) is not None and not _literal_or_quote(
            getattr(hand, name), getattr(hand, name + "_quote"), text
        ):
            setattr(hand, name, None)
    checked_players = []
    for player in hand.players:
        if player.stack_bb is None:
            checked_players.append(player)
            continue
        if not _supported_quote(player.quote, text):
            continue
        amount = player.stack_bb.replace(",", ".")
        quoted = {m.group().replace(",", ".") for m in _NUMBER.finditer(player.quote)}
        if amount not in quoted:
            raise InvalidNaturalReading("Уточните стек: " + player.quote[:80])
        checked_players.append(player)
    hand.players = checked_players
    checked_actions = []
    for action in hand.actions:
        if not _supported_quote(action.quote, text):
            continue
        if not _ACTION_WORDS[action.kind].search(action.quote):
            raise InvalidNaturalReading("Не понял действие: " + action.quote[:80])
        if action.amount is not None:
            amount = action.amount.replace(",", ".")
            quoted = {m.group().replace(",", ".") for m in _NUMBER.finditer(action.quote)}
            if amount not in quoted:
                raise InvalidNaturalReading("Уточните размер действия: " + action.quote[:80])
        checked_actions.append(action)
    hand.actions = checked_actions
    sizes = {int(match.group(1)) for match in _TABLE_SIZE.finditer(text)}
    hand.table_size = sizes.pop() if len(sizes) == 1 else None
    if ante := _ANTE.search(text):
        hand.ante = ante.group(1).replace(",", ".")
        hand.ante_unit = "bb" if ante.group(2) else "chips"
    elif re.search(r"без анте|no ante", text, re.IGNORECASE):
        hand.ante = "0"
        hand.ante_unit = "bb"
    else:
        hand.ante = None
    if blinds := _BLINDS.search(text):
        hand.sb_chips = blinds.group(1).replace(",", ".")
        hand.bb_chips = blinds.group(2).replace(",", ".")
    elif not re.search(r"\bSB\s*\d.*\bBB\s*\d", text, re.IGNORECASE):
        hand.sb_chips = None
        hand.bb_chips = None
    for match in _POSITION_STACK.finditer(text):
        pos = match.group(1).upper()
        if not any(p.position.upper() == pos and p.stack_bb is not None for p in hand.players):
            hand.players.append(NaturalPlayer(
                position=pos, stack_bb=match.group(2).replace(",", "."), quote=match.group(),
            ))
    if hand.hero_position and not any(
        p.position.upper() == hand.hero_position.upper() and p.stack_bb is not None
        for p in hand.players
    ):
        for line in text.splitlines():
            if not (_POSITION.search(line) and hand.hero_position.upper() in line.upper()):
                continue
            if stack := _STACK.search(line):
                hand.players.append(NaturalPlayer(
                    position=hand.hero_position, stack_bb=stack.group(1).replace(",", "."),
                    quote=stack.group(),
                ))
                break
        else:
            if stack := _HERO_STACK_AFTER_CARDS.search(text):
                hand.players.append(NaturalPlayer(
                    position=hand.hero_position, stack_bb=stack.group(1).replace(",", "."),
                    quote=stack.group(),
                ))
    hand.source_text = text
    return NaturalRead(intent=result.intent, hand=hand)


def apply_natural_correction(active: NaturalHand, patch: NaturalHand) -> NaturalHand:
    """Replace only the facts named in a short follow-up, then replay from scratch."""
    updated = active.model_copy(deep=True)
    for field in (
        "table_size", "sb_chips", "bb_chips", "ante", "hero_position", "hero_cards",
        "flop", "turn", "river", "question", "context",
    ):
        value = getattr(patch, field)
        if value is not None:
            setattr(updated, field, value)
            quote_field = field + "_quote"
            if hasattr(updated, quote_field):
                setattr(updated, quote_field, getattr(patch, quote_field))
    if patch.ante is not None:
        updated.ante_unit = patch.ante_unit
    for player in patch.players:
        updated.players = [p for p in updated.players if p.position.upper() != player.position.upper()]
        updated.players.append(player)
    for action in patch.actions:
        matches = [i for i, old in enumerate(updated.actions)
                   if old.street == action.street and old.position.upper() == action.position.upper()
                   and old.kind == action.kind]
        if matches:
            updated.actions[matches[-1]] = action
        else:
            updated.actions.append(action)
    updated.source_text = active.source_text + "\n" + patch.source_text
    return updated
