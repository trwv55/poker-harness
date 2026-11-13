"""Bound a model turn to one verified text decision and its conditional tool."""

from __future__ import annotations

from typing import Any

from pydantic_ai import ToolReturn
from pydantic_ai.tools import Tool

from harness.analysis.decision_spot import spot_call_scenario
from harness.contracts.decision_spot import CanonicalDecisionSpot, DecisionSnapshot
from harness.contracts.ranges import class_of
from harness.explanation.agent_checks import AgentContext

MAX_SPOT_SCENARIOS_PER_TURN = 3


def spot_agent_context(spot: CanonicalDecisionSpot, snapshot: DecisionSnapshot) -> AgentContext:
    cards = "".join(spot.hero_cards) if spot.hero_cards else spot.hero_class
    lines = [
        "Текстовый спот: слова игрока, не сыгранная HH-раздача. Префлоп, до действия героя.",
        f"Герой {spot.hero_position}, карты {cards}.",
        f"Блайнды {spot.sb}/{spot.bb}, анте {spot.ante} ({spot.ante_type}).",
        "Контекст турнира: " + (spot.tournament_context or "неизвестен"),
        "ICM: " + (spot.icm_context or "неизвестно"),
        "KO/PKO: " + (spot.ko_context or "неизвестно"),
        "Примерные стеки после обязательных взносов: " + (
            "; ".join(
                f"{p.position} ~{p.approx_stack_behind / spot.bb:g}BB"
                for p in spot.players if p.approx_stack_behind is not None
            ) or "нет"
        ),
        f"Подтверждённый банк до героя {snapshot.pot_before} фишек; доплата {snapshot.to_call} фишек.",
        "Действия до героя: "
        + "; ".join(
            f"{a.position} {a.kind} {a.committed_after if a.committed_after is not None else ''}"
            for a in spot.actions
        ),
        "После героя ещё действуют: " + (", ".join(snapshot.positions_still_to_act) or "никто"),
        "Не проверено: " + ("; ".join(snapshot.not_checked) or "нет"),
        (
            "Не называй вердикт о сыгранном: действие героя неизвестно. Не используй HH-чарты "
            "для линии, которую они не покрывают. Для порога колла сначала вызови "
            "spot_call_scenario с явным условием о всех игроках после героя. "
            "Без диапазона оппонента не советуй колл или фолд. "
            "Без выплат и баунти не называй численный ICM/KO EV."
        ),
    ]
    numbers = {
        float(spot.sb),
        float(spot.bb),
        float(spot.ante),
        float(snapshot.pot_before),
        float(snapshot.to_call),
        *(float(v) for v in snapshot.committed.values()),
        *(float(p.approx_stack_behind / spot.bb) for p in spot.players if p.approx_stack_behind is not None),
    }
    return AgentContext(
        text="\n".join(lines),
        allowed=frozenset(numbers),
        labels=frozenset(),
        hero_class=class_of(*spot.hero_cards) if spot.hero_cards else spot.hero_class,
        board_ranks=(),
    )


def spot_tools(
    spot_id: int,
    spot: CanonicalDecisionSpot,
    snapshot: DecisionSnapshot,
    calls: list[dict[str, Any]],
) -> list[Tool[None]]:
    async def spot_call_scenario_tool(
        fold_after_hero: list[str],
        opponent_range: str | None = None,
        range_source: str | None = None,
    ) -> str | ToolReturn:
        """Calculate a conditional preflop call, with explicitly named later folds.

        Args:
            fold_after_hero: positions that will fold after hero; name all of them.
            opponent_range: optional explicitly named opponent range in class notation.
            range_source: user_range or model_assumption when a range is supplied.
        """
        if len(calls) >= MAX_SPOT_SCENARIOS_PER_TURN:
            reason = "Лимит расчётов сценария за один ответ исчерпан."
            calls.append({"calc": "spot_call_scenario", "refused": reason})
            return f"Расчёт не выполнен: {reason}"
        try:
            result = spot_call_scenario(
                spot_id,
                spot,
                snapshot,
                fold_after_hero=tuple(fold_after_hero),
                opponent_range=opponent_range,
                range_source=range_source,
            )
        except ValueError as exc:
            calls.append({"calc": "spot_call_scenario", "refused": str(exc)})
            return f"Расчёт не выполнен: {exc}"
        threshold = (
            round(result.equity_threshold * 100, 1) if result.equity_threshold is not None else None
        )
        equity = round(result.equity * 100, 1) if result.equity is not None else None
        text = (
            f"Условие: {result.condition}. Банк до хода {result.pot_before} фишек; "
            f"доплата {result.to_call} фишек. "
            + (f"Порог эквити {threshold}%. " if threshold is not None else "")
            + (f"Эквити против диапазона {equity}%. " if equity is not None else "")
            + result.explanation
        )
        calls.append(
            {
                "calc": "spot_call_scenario",
                "condition": result.condition,
                "output": result.model_dump(mode="json"),
            }
        )
        allowed = [float(result.pot_before), float(result.to_call)]
        if result.eligible_pot is not None:
            allowed.append(float(result.eligible_pot))
        if threshold is not None:
            allowed.append(threshold)
        if equity is not None:
            allowed.append(equity)
        return ToolReturn(
            return_value=text,
            metadata={
                "calc": "spot_call_scenario",
                "allowed": allowed,
                "threshold": [threshold] if threshold is not None else [],
                "equity": [equity] if equity is not None else [],
            },
        )

    return [Tool(spot_call_scenario_tool, name="spot_call_scenario", sequential=True)]
