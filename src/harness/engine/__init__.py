"""Движок руки: механика (`replay`) и политика (`validate`) — в одном прогоне.

`enrich` проигрывает руку ровно один раз и складывает результат в
`EnrichedHand`; все последующие сервисы конвейера читают готовое, а не
переигрывают руку заново (спека §3 арх.).
"""

from __future__ import annotations

from harness.contracts import CanonicalHand, EnrichedHand
from harness.engine.replay import replay
from harness.engine.validation import validate

__all__ = ["enrich", "replay", "validate"]


def enrich(hand: CanonicalHand) -> EnrichedHand:
    """Проиграть руку движком и вынести вердикт — единственный прогон на руку."""
    report = replay(hand)
    return EnrichedHand(hand=hand, report=report, verdict=validate(hand, report))
