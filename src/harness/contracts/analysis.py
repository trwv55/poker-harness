"""Результат аналитического ядра: по одному вердикту на точку решения.

`Zone` различает точки, где применяется точное правило (`strict`, зона
пуш/фолд, шов и т.п.), от точек, где вывод строится на допущении о диапазоне
оппонента (`assuming`) — см. `Assumption`. `AnalysisResult` собирает вердикты
по всем точкам решения одной руки и суммарные потери в bb.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, model_validator

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

    @model_validator(mode="after")
    def _assumption_matches_zone(self) -> PointVerdict:
        """Допущение заполнено тогда и только тогда, когда зона `assuming`.

        Правило держится в контракте, а не в договорённости: вывод в зоне
        «предполагая» обязан нести показанное игроку допущение, а вывод, помеченный
        «строго», не имеет права его нести — иначе продукт либо скрывает догадку,
        либо выдаёт догадку за точный расчёт. Инвариант касается всех, кто
        конструирует вердикты (ядро, скан, изложение), поэтому проверяет его сам
        тип, а не тест конкретного модуля.
        """
        if (self.assumption is not None) != (self.zone is Zone.ASSUMING):
            raise ValueError(
                f"допущение должно быть заполнено ровно при зоне assuming: "
                f"зона {self.zone}, допущение {'есть' if self.assumption else 'нет'}"
            )
        return self


class AnalysisResult(BaseModel):
    schema_version: int = 1
    hand_no: str
    points: list[PointVerdict]
    ranked: list[int] = []
    total_ev_loss_bb: float = 0.0
