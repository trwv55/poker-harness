"""Контракты данных: модели, которыми обмениваются сервисы конвейера.

Публичный API пакета — реэкспорт из всех подмодулей (`raw`, `canonical`,
`enriched`, `ranges`, `analysis`), чтобы последующие задачи импортировали
из `harness.contracts`, а не из отдельных модулей.
"""

from __future__ import annotations

from harness.contracts.analysis import (
    AnalysisResult,
    Assumption,
    PointVerdict,
    SpotKind,
    Zone,
)
from harness.contracts.canonical import (
    CanonicalAction,
    CanonicalHand,
    Identity,
    PlayerState,
)
from harness.contracts.enriched import (
    DecisionPoint,
    EngineReport,
    EnrichedHand,
    SidePot,
    ValidationStatus,
    Verdict,
)
from harness.contracts.ranges import RANKS, Range, all_classes, class_of
from harness.contracts.raw import (
    ActionKind,
    Collected,
    Post,
    PostKind,
    Provenance,
    RawAction,
    RawHand,
    SeatInfo,
    ShowdownEntry,
    Street,
    SummaryInfo,
    Uncalled,
    VisionMeta,
)

__all__ = [
    "RANKS",
    "ActionKind",
    "AnalysisResult",
    "Assumption",
    "CanonicalAction",
    "CanonicalHand",
    "Collected",
    "DecisionPoint",
    "EngineReport",
    "EnrichedHand",
    "Identity",
    "PlayerState",
    "PointVerdict",
    "Post",
    "PostKind",
    "Provenance",
    "Range",
    "RawAction",
    "RawHand",
    "SeatInfo",
    "ShowdownEntry",
    "SidePot",
    "SpotKind",
    "Street",
    "SummaryInfo",
    "Uncalled",
    "ValidationStatus",
    "Verdict",
    "VisionMeta",
    "Zone",
    "all_classes",
    "class_of",
]
