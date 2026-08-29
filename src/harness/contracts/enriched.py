"""Результат движка (пот/стеки/решения) поверх канонической руки плюс вердикт валидации.

`EnrichedHand` — вход для аналитического ядра: помимо самой руки несёт отчёт
движка (`EngineReport`) с точками решений (`DecisionPoint`) и вердикт
валидатора (`Verdict`), решающий, годна ли рука для дальнейшего анализа.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from harness.contracts.canonical import CanonicalAction, CanonicalHand
from harness.contracts.raw import Street


class SidePot(BaseModel):
    amount: int
    eligible: list[str]


class DecisionPoint(BaseModel):
    index: int
    street: Street
    label: str
    position: str
    to_call: int
    pot_before: int
    eff_stack: int
    eff_stack_bb: float
    spr: float | None = None
    action: CanonicalAction
    live_total: int = 0  # игроков ещё в руке на момент решения, включая Hero
    live_behind: int = 0  # из них ещё не действовавших после Hero — вход правила зоны


class ValidationStatus(StrEnum):
    PASS = "pass"
    ESCALATE = "escalate"
    REJECT = "reject"


class Verdict(BaseModel):
    status: ValidationStatus
    fields: list[str] = []
    questions: list[str] = []
    reasons: list[str] = []


class EngineReport(BaseModel):
    pot_by_street: dict[Street, int]
    final_pot: int
    side_pots: list[SidePot] = []
    stacks_end: dict[str, int]
    decision_points: list[DecisionPoint]
    illegal_actions: list[str] = []


class EnrichedHand(BaseModel):
    schema_version: int = 1
    hand: CanonicalHand
    report: EngineReport
    verdict: Verdict
