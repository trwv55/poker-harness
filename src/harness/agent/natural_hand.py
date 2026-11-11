"""Bound one model opinion to a replayed, clearly defaulted text scenario."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from harness.contracts.natural_hand import ScenarioHand, ScenarioSnapshot
from harness.platform.llm import LLM


class NaturalOpinion(BaseModel):
    explanation: str = Field(description="Short qualitative reasoning without numbers")
    opponent_range: str | None = Field(
        default=None,
        description="Hypothetical opponent preflop range in class notation, never a chart",
    )


_PROMPT_PATH = Path(__file__).parent / "prompts" / "natural_hand.md"
_NUMERIC = re.compile(r"\d|%")


async def natural_opinion(
    llm: LLM, hand: ScenarioHand, snapshot: ScenarioSnapshot, trace_id: int,
) -> NaturalOpinion | None:
    """Request qualitative judgment; numeric text is discarded, not shown."""
    hero = "".join(hand.hero_cards) if hand.hero_cards else hand.hero_class or "неизвестны"
    actions = "; ".join(
        f"{a.street} {a.position} {a.kind} до {a.committed_after} BB"
        for a in hand.actions
    )
    prompt = (
        f"Вопрос игрока: {hand.question or 'Как оценить решение?'}\n"
        f"Герой {hand.hero_position}, карты {hero}; борд {''.join(hand.board) or 'нет'}.\n"
        f"Действия: {actions}.\n"
        f"Банк {snapshot.pot_before_bb} BB, доплата {snapshot.to_call_bb} BB. "
        f"После героя действуют {', '.join(snapshot.positions_after_hero) or 'никто'}.\n"
        f"Подставлено: {'; '.join(hand.assumptions) or 'ничего'}."
    )
    result, _ = await llm(
        "agent", NaturalOpinion, prompt=prompt, instructions=_PROMPT_PATH.read_text(encoding="utf-8"),
        trace_id=trace_id,
    )
    explanation = result.explanation.strip()
    if _NUMERIC.search(explanation):
        explanation = ""
    elif len(explanation) > 600:
        sentences = re.split(r"(?<=[.!?])\s+", explanation)
        explanation = " ".join(sentences[:3])[:600].strip()
    opponent_range = result.opponent_range
    if opponent_range and (len(opponent_range) > 120 or not re.fullmatch(
        r"[2-9TJQKAso+,:;\s-]+", opponent_range, re.IGNORECASE
    )):
        opponent_range = None
    if not explanation and not opponent_range:
        return None
    return NaturalOpinion(explanation=explanation, opponent_range=opponent_range)
