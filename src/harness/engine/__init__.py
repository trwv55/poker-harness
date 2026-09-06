"""Движок руки: механика (`replay`/`state_report`) и политика (`validate`) — в одном прогоне.

`enrich` проигрывает руку ровно один раз и складывает результат в
`EnrichedHand`; все последующие сервисы конвейера читают готовое, а не
переигрывают руку заново (спека §3 арх.).

**Механик две, и выбор между ними — свойство входа, а не настройка.** Полную
руку проигрывает `replay`; состояние в точке решения проиграть нельзя по
построению (последовательности действий на экране не напечатано), и его берёт
`state.state_report` — см. докстринг `harness.engine.state`. Развилку держит
`CanonicalHand.completeness`, выведенный из прочитанного, а не тип экрана
(решение владельца C1).
"""

from __future__ import annotations

from harness.contracts import CanonicalHand, Completeness, EnrichedHand
from harness.engine.replay import replay
from harness.engine.state import StateNotReadable, state_report
from harness.engine.validation import validate

__all__ = ["StateNotReadable", "enrich", "replay", "state_report", "validate"]


def enrich(hand: CanonicalHand) -> EnrichedHand:
    """Проиграть руку движком и вынести вердикт — единственный прогон на руку.

    Состояние в точке решения (`Completeness.STATE`) идёт мимо реплея: там нечего
    проигрывать. Вердикт выносит тот же `validate` — он сам знает, какие из его
    проверок на состоянии выполнимы, а какие названы неисполненными.
    """
    report = state_report(hand) if hand.completeness is Completeness.STATE else replay(hand)
    return EnrichedHand(hand=hand, report=report, verdict=validate(hand, report))
