"""Conditional call arithmetic for a verified text spot."""

from __future__ import annotations

from harness.analysis.charts.notation import parse_range
from harness.analysis.tools.equity import combos_of_class, equity_vs_range
from harness.analysis.tools.pot_odds import required_equity
from harness.contracts.decision_spot import (
    CanonicalDecisionSpot,
    DecisionSnapshot,
    SpotCallScenario,
)


def spot_call_scenario(
    spot_id: int,
    spot: CanonicalDecisionSpot,
    snapshot: DecisionSnapshot,
    *,
    fold_after_hero: tuple[str, ...],
    opponent_range: str | None = None,
    range_source: str | None = None,
) -> SpotCallScenario:
    """Only a named heads-up outcome may receive a pot-odds threshold."""
    waiting = set(snapshot.positions_still_to_act)
    if set(fold_after_hero) != waiting or len(fold_after_hero) != len(waiting):
        raise ValueError("Укажите, что именно сделают все игроки после героя.")
    folded = {a.position for a in spot.actions if a.kind == "fold"} | waiting
    live = [
        p.position
        for p in spot.players
        if p.position not in folded and p.position != spot.hero_position
    ]
    condition = "если " + (
        ", ".join(fold_after_hero) + " сбросят"
        if fold_after_hero
        else "после героя никто не действует"
    )
    if snapshot.to_call <= 0:
        raise ValueError("Это не точка колла: доплата равна нулю.")
    if len(live) != 1:
        return SpotCallScenario(
            spot_id=spot_id,
            condition=condition,
            pot_before=snapshot.pot_before,
            to_call=snapshot.to_call,
            eligible_pot=None,
            equity_threshold=None,
            explanation="После условия остаётся не один оппонент; нужен отдельный мультивей-сценарий.",
        )
    hero_stack = next(p.stack for p in spot.players if p.position == spot.hero_position)
    if hero_stack is None or snapshot.committed[spot.hero_position] + snapshot.to_call > hero_stack:
        return SpotCallScenario(
            spot_id=spot_id,
            condition=condition,
            pot_before=snapshot.pot_before,
            to_call=snapshot.to_call,
            eligible_pot=None,
            equity_threshold=None,
            explanation="Нужен подтверждённый стек героя и возможный сайд-пот.",
        )
    cap = min(
        snapshot.committed[spot.hero_position] + snapshot.to_call, snapshot.committed[live[0]]
    )
    if any(snapshot.committed[p] > cap for p in folded):
        return SpotCallScenario(
            spot_id=spot_id,
            condition=condition,
            pot_before=snapshot.pot_before,
            to_call=snapshot.to_call,
            eligible_pot=None,
            equity_threshold=None,
            explanation="Возможен побочный банк; простой порог колла здесь неверен.",
        )
    threshold = required_equity(snapshot.to_call, snapshot.pot_before)
    scenario = SpotCallScenario(
        spot_id=spot_id,
        condition=condition,
        pot_before=snapshot.pot_before,
        to_call=snapshot.to_call,
        eligible_pot=snapshot.pot_before + snapshot.to_call,
        equity_threshold=threshold,
        explanation="Порог условный; без диапазона оппонента вывода о колле нет.",
    )
    if opponent_range is None:
        return scenario
    if range_source not in ("user_range", "model_assumption"):
        raise ValueError("Укажите источник диапазона: игрок или явная оценка модели.")
    scenario.opponent_range = opponent_range
    scenario.range_source = range_source  # type: ignore[assignment]
    if spot.hero_cards is None:
        scenario.explanation = "Для эквити против диапазона нужны точные масти карт героя."
        return scenario
    rng = parse_range(opponent_range)
    dead = set(spot.hero_cards)
    scenario.excluded_blocked_combos = sum(
        sum(c1 in dead or c2 in dead for c1, c2 in combos_of_class(cls))
        for cls, weight in rng.weights.items()
        if weight > 0
    )
    scenario.equity = equity_vs_range(spot.hero_cards, rng, iterations=20_000, seed=42)
    scenario.explanation = "Эквити относится только к явно указанному диапазону и условию."
    return scenario
