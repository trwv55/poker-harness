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
_HAND_REQUEST = re.compile(r"разбер\w*\s+(?:эту\s+|мою\s+)?(?:руку|раздачу)", re.IGNORECASE)


def looks_like_decision_spot(text: str) -> bool:
    """Conservative prefilter: no extra LLM call for a generic agent question."""
    if len(text) > MAX_SPOT_TEXT or not text.strip():
        return False
    return bool(_HAND_REQUEST.search(text) or (
        _ACTION.search(text)
        and (_HERO.search(text) or _QUESTION.search(text))
        and (_POSITION.search(text) or _CARDS.search(text))
    ))


_INSTRUCTIONS = """Извлеки факты ОДНОГО описанного ретроспективного префлоп-спота NLHE MTT.
Верни is_spot=false, если это не описание раздачи до решения героя. Не придумывай факты.
Каждое известное поле получит status=known и короткую ДОСЛОВНУЮ quote из сообщения;
отсутствующее — status=missing, неоднозначное — status=ambiguous.
Суммы сохраняй как в тексте: '28 bb', '1000'; для действия различай total,
additional, behind, unspecified. Если неизвестно, ставь unspecified.
Названный стек обычно означает остаток после анте и блайндов, до добровольных
действий. Укажи stack_moment=behind только если текст явно говорит «за спиной»;
иначе unknown. Тип плательщика анте не извлекай: если анте положительное,
его платят все по правилу продукта. Если явно написано «без анте», поставь
ante.value='0', quote с этими словами и status=known.
Сумму рейза без слов «доплатил» считай итогом на улице (amount_kind=total).
Стек героя обязательно положи в hero_stack, даже если не добавляешь героя в players.
В sb и bb запиши два отдельных числа: при «блайнды 400/800» sb=400, bb=800.
Турнирную стадию, слова игрока об ICM и формат KO/PKO сохрани отдельно в
tournament_context, icm_context, ko_context с дословными quote. Если игрок
не знает, оставь его «не знаю»; не придумывай ICM-расчёт или баунти.
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
    spot = result.spot.model_copy(deep=True, update={"source_text": text})
    facts = [
        spot.table_size, spot.sb, spot.bb, spot.ante, spot.hero_position, spot.hero_cards,
        spot.hero_stack,
        spot.tournament_context, spot.icm_context, spot.ko_context,
        *(player.stack for player in spot.players),
        *(action.amount for action in spot.actions),
    ]
    for fact in facts:
        if fact.status == "known" and (
            not fact.value or not fact.quote or fact.quote.casefold() not in text.casefold()
        ):
            # Keep the usable parts of a partial extraction; the normalizer will
            # ask for this fact instead of discarding the entire hand description.
            fact.status = "missing"
            fact.value = None
            fact.quote = ""
    return spot
