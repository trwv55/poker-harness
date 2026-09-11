"""Аналитическое ядро: превращает `EnrichedHand` в вердикт по каждой точке решения.

Здесь конвейер впервые говорит игроку «так играть было верно» или «здесь потеряно
столько-то». Всё, что для этого нужно, считает код: LLM в этом пакете нет.

Границы v1 названы явно и в результате видны: ЦЕНУ получают только
префлоп-точки пуш-фолд-зоны, и та — в chip-EV (bb) без поправки на ICM
(призовой структуры в hand history нет). Точки, чью цену ядро посчитать не
умеет, возвращаются с нулевым `ev_diff_bb` и не ранжируются: неизвестная цена не
выдаётся за ноль.

Разбор при этом шире цены. Постфлоп-точка несёт требование к ставящему диапазону
соперника: на ривере (`analysis.river`) — а когда борд исчерпан, и лучшее
действие; на тёрне и флопе (`analysis.turn_flop`) — только требование, потому
что за коллом там идут улицы, которых перебор не считает. Цены у такой точки нет
ни в одном из случаев, и в сумму потерь она не входит.
"""

from __future__ import annotations

from harness.analysis.classifier import classify
from harness.analysis.error_cost import rank_points, total_ev_loss_bb
from harness.analysis.preflop import cheap_fold_verdict, verdict_for, zone_for
from harness.contracts import AnalysisResult, EnrichedHand

__all__ = [
    "analyze_hand",
    "cheap_fold_verdict",
    "classify",
    "rank_points",
    "total_ev_loss_bb",
    "verdict_for",
    "zone_for",
]


def analyze_hand(en: EnrichedHand) -> AnalysisResult:
    """Разобрать все точки решения героя в одной руке."""
    points = [verdict_for(dp, en) for dp in en.report.decision_points]
    return AnalysisResult(
        hand_no=en.hand.hand_no,
        points=points,
        ranked=rank_points(points),
        total_ev_loss_bb=total_ev_loss_bb(points),
    )
