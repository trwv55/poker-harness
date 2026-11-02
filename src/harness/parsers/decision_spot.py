"""Cheap routing and evidenced extraction of a described preflop decision."""

from __future__ import annotations

import re

from harness.contracts.decision_spot import RawDecisionSpot, SpotExtraction
from harness.platform.llm import LLM

MAX_SPOT_TEXT = 4000
_POSITION = re.compile(
    r"\b(?:UTG\+?[12]?|LJ|HJ|CO|BTN|SB|BB)\b|\b(?:баттон|блайнд\w*|катофф)\b", re.IGNORECASE
)
_CARDS = re.compile(r"\b(?:[2-9TJQKA]){2}[so]?\b|\b(?:[2-9TJQKA][shdc]){2}\b", re.IGNORECASE)
_ACTION = re.compile(
    r"олл.?ин|пуш|рейз|откр|колл|пас|фолд|сброс|став|call|raise|fold|shove", re.IGNORECASE
)
_HERO = re.compile(r"геро|у меня|моя рука|я (?:на|с|в)|hero", re.IGNORECASE)
_QUESTION = re.compile(r"что делать|как сыграть|разбер|какое решение|что выбрать", re.IGNORECASE)


def looks_like_decision_spot(text: str) -> bool:
    """Conservative prefilter: no extra LLM call for a generic agent question."""
    if len(text) > MAX_SPOT_TEXT or not text.strip():
        return False
    return bool(
        _ACTION.search(text)
        and (_HERO.search(text) or _QUESTION.search(text))
        and _POSITION.search(text)
        and _CARDS.search(text)
    )


_INSTRUCTIONS = """Извлеки факты ОДНОГО описанного ретроспективного префлоп-спота NLHE MTT.
Верни is_spot=false, если это не описание раздачи до решения героя. Не придумывай факты.
Каждое известное поле получит status=known и короткую ДОСЛОВНУЮ quote из сообщения;
отсутствующее — status=missing, неоднозначное — status=ambiguous.
Суммы сохраняй как в тексте: '28 bb', '1000'; для действия различай total,
additional, behind, unspecified. Если неизвестно, ставь unspecified.
Для стека сохраняй момент измерения: before_ante, after_ante, after_post,
behind, unknown. Для анте различай per_player, big_blind, none, unknown.
Сохраняй все действия до героя в хронологическом порядке, не добавляй его действие.
Неразобранные важные фразы верни в unknown_lines. Не считай банк, шансы, EV или диапазон.
source_text оставь пустым: исходное сообщение подставит код.
"""


async def extract_decision_spot(llm: LLM, text: str, trace_id: int) -> RawDecisionSpot | None:
    """Run one extraction call on candidate text; its output is untrusted."""
    result, _ = await llm(
        "spot_extract",
        SpotExtraction,
        prompt=(
            text if len(text) <= MAX_SPOT_TEXT else
            text[: MAX_SPOT_TEXT // 2] + "\n[…]\n" + text[-MAX_SPOT_TEXT // 2 :]
        ),
        instructions=_INSTRUCTIONS,
        trace_id=trace_id,
    )
    if not result.is_spot or result.spot is None:
        return None
    spot = result.spot.model_copy(update={"source_text": text})
    facts = [
        spot.table_size, spot.sb, spot.bb, spot.ante, spot.hero_position, spot.hero_cards,
        *(player.stack for player in spot.players),
        *(action.amount for action in spot.actions),
    ]
    if any(
        fact.status == "known" and (
            not fact.value or not fact.quote or fact.quote.casefold() not in text.casefold()
        )
        for fact in facts
    ):
        return None
    return spot
