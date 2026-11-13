"""Conditional equity for a replayed, defaulted text scenario."""

from __future__ import annotations

from harness.analysis.charts.notation import parse_range
from harness.analysis.tools.equity import equity_vs_range
from harness.contracts.natural_hand import ScenarioHand, ScenarioSnapshot


def conditional_equity(
    hand: ScenarioHand, snapshot: ScenarioSnapshot, range_text: str,
) -> float | None:
    """One heads-up branch: everyone after hero folds; range remains an assumption."""
    if hand.hero_cards is None or snapshot.to_call_bb <= 0:
        return None
    others = [p for p in snapshot.live_positions if p != hand.hero_position]
    later = [p for p in snapshot.positions_after_hero if p in others]
    if len(others) - len(later) != 1 or snapshot.possible_side_pots and not later:
        return None
    if snapshot.to_call_bb > snapshot.remaining_bb[hand.hero_position]:
        return None
    try:
        rng = parse_range(range_text)
        return equity_vs_range(hand.hero_cards, rng, hand.board, iterations=20_000, seed=42)
    except (ValueError, KeyError):
        return None
