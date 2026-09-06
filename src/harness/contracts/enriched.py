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
    # `None` — решение ещё НЕ ПРИНЯТО: на живом столе экран застаёт героя до
    # хода, и действия, которое можно было бы судить, не существует. Придумать
    # его нельзя (это было бы утверждение о том, чего игрок не делал), поэтому
    # поле необязательное, а ядро на такой точке отказывается судить с названной
    # причиной (`harness.analysis.preflop.verdict_for`).
    action: CanonicalAction | None = None
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
    # Проверки, которые на этом входе выполнить НЕ ИЗ ЧЕГО, названные поимённо.
    # Пустой список у hand history: там источник даёт три независимых факта
    # (`Total pot`, строки `collected`, порядок хода). У состояния в точке
    # решения нет ни одного, и молчаливый `pass` был бы неотличим от
    # проверенного входа — отсюда список, а не тишина (реестр D1, E1).
    not_checked: list[str] = []


class EngineReport(BaseModel):
    pot_by_street: dict[Street, int]
    final_pot: int
    side_pots: list[SidePot] = []
    stacks_end: dict[str, int]
    decision_points: list[DecisionPoint]
    illegal_actions: list[str] = []
    # Игроки, которым источник записал пас при нулевом стеке (олл-ин с
    # вынужденной ставки): движок исполняет такой пас, хотя правила NLHE
    # оставляют игрока в руке. Список делает эту поправку видимой в трассе.
    forfeits: list[str] = []


class EnrichedHand(BaseModel):
    schema_version: int = 1
    hand: CanonicalHand
    report: EngineReport
    verdict: Verdict
