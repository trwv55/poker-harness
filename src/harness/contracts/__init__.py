"""Контракты данных: модели, которыми обмениваются сервисы конвейера.

Публичный API пакета — реэкспорт из всех подмодулей (`raw`, `canonical`,
`enriched`, `ranges`, `analysis`, `explanation`), чтобы последующие задачи импортировали
из `harness.contracts`, а не из отдельных модулей.
"""

from __future__ import annotations

from harness.contracts.analysis import (
    AllInEvent,
    AnalysisResult,
    Assumption,
    ChipMove,
    EvInterval,
    EvSplit,
    Finding,
    LevelLine,
    PlayerStats,
    PointVerdict,
    ScanItem,
    ScanSummary,
    SpotKind,
    StackTrajectory,
    TournamentReport,
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
from harness.contracts.explanation import (
    PointText,
    TournamentTextOut,
    VerdictLabel,
    VerdictTextOut,
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
    "AllInEvent",
    "AnalysisResult",
    "Assumption",
    "CanonicalAction",
    "CanonicalHand",
    "ChipMove",
    "Collected",
    "DecisionPoint",
    "EngineReport",
    "EnrichedHand",
    "EvInterval",
    "EvSplit",
    "Finding",
    "Identity",
    "LevelLine",
    "PlayerState",
    "PlayerStats",
    "PointText",
    "PointVerdict",
    "Post",
    "PostKind",
    "Provenance",
    "Range",
    "RawAction",
    "RawHand",
    "ScanItem",
    "ScanSummary",
    "SeatInfo",
    "ShowdownEntry",
    "SidePot",
    "SpotKind",
    "StackTrajectory",
    "Street",
    "SummaryInfo",
    "TournamentReport",
    "TournamentTextOut",
    "Uncalled",
    "ValidationStatus",
    "Verdict",
    "VerdictLabel",
    "VerdictTextOut",
    "VisionMeta",
    "Zone",
    "all_classes",
    "class_of",
]
