"""Сырые данные, извлечённые парсером из источника (hand history / скриншот).

`RawHand` — это буквальное представление того, что написано в источнике: суммы
действий записаны как доплаты (как в GG hand history), улицы разложены по
секциям, а всё, что парсер не смог распознать, попадает в `unknown_lines`.
Нормализация в `CanonicalHand` (см. `canonical.py`) происходит позже.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class Provenance(StrEnum):
    HAND_HISTORY = "hand_history"
    SCREENSHOT = "screenshot"


class Street(StrEnum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class ActionKind(StrEnum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


class PostKind(StrEnum):
    ANTE = "ante"
    SMALL_BLIND = "small_blind"
    BIG_BLIND = "big_blind"


class SeatInfo(BaseModel):
    seat: int
    label: str
    stack: int


class Post(BaseModel):
    label: str
    kind: PostKind
    amount: int


class RawAction(BaseModel):
    street: Street
    label: str
    kind: ActionKind
    amount: int | None = None  # calls/bets N — ДОПЛАТА, как в источнике
    to_amount: int | None = None  # raises X to Y -> Y
    is_all_in: bool = False
    raw_line: str


class Uncalled(BaseModel):
    label: str
    amount: int


class ShowdownEntry(BaseModel):
    label: str
    cards: list[str]


class Collected(BaseModel):
    label: str
    amount: int  # мейн/сайд не подписаны — как в GG


class SummaryInfo(BaseModel):
    total_pot: int
    rake: int
    jackpot: int
    bingo: int
    fortune: int
    tax: int
    board: list[str] = []
    seat_lines: list[str] = []


class VisionMeta(BaseModel):  # только для скринов
    confidence: dict[str, float] = {}
    needs_review: list[str] = []
    image_hash: str | None = None
    nicknames: dict[str, str] = {}
    bounties: dict[str, int] = {}
    displayed_pot: int | None = None


class RawHand(BaseModel):
    schema_version: int = 1
    provenance: Provenance
    source_ref: str
    hand_no: str
    tournament_id: str
    tournament_name: str
    level: int
    sb: int
    bb: int
    ante: int
    ante_type: str = "per_player"
    timestamp: datetime
    table_name: str
    max_seats: int
    button_seat: int
    seats: list[SeatInfo]
    posts: list[Post]
    dealt: dict[str, list[str]] = {}  # пустой список = Dealt to без карт
    actions: list[RawAction] = []
    boards: dict[Street, list[str]] = {}
    uncalled: list[Uncalled] = []
    showdowns: list[ShowdownEntry] = []
    collected: list[Collected] = []
    summary: SummaryInfo | None = None
    vision: VisionMeta | None = None
    unknown_lines: list[str] = []  # всё, что парсер не распознал
