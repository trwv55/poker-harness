"""Нормализованное представление руки — вход для движка и аналитического ядра.

В отличие от `RawHand`, здесь суммы действий — это накопленный итог,
поставленный игроком на текущей улице (`committed_after`), а не доплата;
позиции игроков вычислены явно (`PlayerState.position`); личность каждого
игрока классифицирована (`Identity`).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel

from harness.contracts.raw import (
    ActionKind,
    Collected,
    Provenance,
    ShowdownEntry,
    Street,
    SummaryInfo,
    Uncalled,
    VisionMeta,
)


class Identity(StrEnum):
    HERO = "hero"
    NICK = "nick"
    ANON = "anon"


class PlayerState(BaseModel):
    seat: int
    label: str
    identity: Identity
    position: str  # "BTN"/"SB"/"BB"/"UTG"/...
    stack: int
    stack_bb: float


class CanonicalAction(BaseModel):
    street: Street
    label: str
    kind: ActionKind
    committed_after: int  # ИТОГО поставлено игроком на этой улице после действия
    is_all_in: bool = False
    raw_line: str


class CanonicalHand(BaseModel):
    schema_version: int = 1
    provenance: Provenance
    tournament_id: str
    hand_no: str
    hand_index: int | None = None
    level: int
    sb: int
    bb: int
    ante: int
    ante_type: str = "per_player"
    timestamp: datetime
    button_seat: int
    hero_label: str = "Hero"
    players: list[PlayerState]
    dealt: dict[str, list[str]] = {}
    actions: list[CanonicalAction] = []
    boards: dict[Street, list[str]] = {}
    uncalled: list[Uncalled] = []
    showdowns: list[ShowdownEntry] = []
    collected: list[Collected] = []
    summary: SummaryInfo | None = None
    bounties: dict[str, int] | None = None
    bounty_source: str | None = None
    vision: VisionMeta | None = None
