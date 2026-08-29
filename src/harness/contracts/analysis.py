"""Результат аналитического ядра: по одному вердикту на точку решения.

`Zone` различает точки, где применяется точное правило (`strict`, зона
пуш/фолд, шов и т.п.), от точек, где вывод строится на допущении о диапазоне
оппонента (`assuming`) — см. `Assumption`. `AnalysisResult` собирает вердикты
по всем точкам решения одной руки и суммарные потери в bb.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from harness.contracts.ranges import Range
from harness.contracts.raw import Street


class Zone(StrEnum):
    STRICT = "strict"
    ASSUMING = "assuming"


class SpotKind(StrEnum):
    PUSHFOLD_UNOPENED = "pushfold_unopened"
    PUSHFOLD_FACING_SHOVE = "pushfold_facing_shove"
    PREFLOP_OTHER = "preflop_other"
    POSTFLOP = "postflop"


class Assumption(BaseModel):
    range: Range
    source: str
    note: str = ""


class PointVerdict(BaseModel):
    dp_index: int
    street: Street
    spot: SpotKind
    zone: Zone
    action_taken: str
    best_action: str
    ev_diff_bb: float  # <0 = потеря
    assumption: Assumption | None = None
    tools: list[str] = []
    detail: dict[str, Any] = {}


class AnalysisResult(BaseModel):
    schema_version: int = 1
    hand_no: str
    points: list[PointVerdict]
    ranked: list[int] = []
    total_ev_loss_bb: float = 0.0
