"""Literal Pokerdom table observations; deliberately separate from GG vision JSON."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, create_model

from harness.contracts.enriched import EnrichedHand
from harness.contracts.raw import RawHand, Street


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PokerdomPlayer(Observation):
    row_id: str
    nickname: str
    stack_shown: StrictInt | None
    button_shown: bool
    cards: list[str] = Field(default_factory=list)
    showdown_shown: bool = False


class PokerdomEvent(Observation):
    number: StrictInt
    player: str
    street: Street
    token: str
    amount_text: str | None
    amount: StrictInt | None
    cell: str = ""


class PokerdomReading(Observation):
    schema_version: Literal[1] = 1
    is_hand: bool
    complete: bool
    cropped: bool
    unit: Literal["chips", "bb", "unknown"]
    stack_semantics: Literal["before_forced", "unknown"]
    players: list[PokerdomPlayer]
    events: list[PokerdomEvent]
    board: dict[Street, list[str]]
    pot_after_street: dict[Street, StrictInt | None]
    metadata: dict[str, str] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)


class PokerdomPlayerV2(Observation):
    nickname: str | None = Field(description="Nickname printed on this player row; null if unreadable.")
    nickname_complete: bool = Field(description="False for truncated or unreadable nickname; never guess missing characters.")
    stack_shown: StrictInt | None
    button_shown: bool
    cards: list[str] = Field(default_factory=list)
    showdown_shown: bool = False


class PokerdomEventV2(Observation):
    number: StrictInt
    player_nickname: str | None = Field(description="Exact printed nickname of the row owning this event; must match exactly one players.nickname. Not a row ID, seat, position or Hero.")
    street: Street
    token: str
    amount_text: str | None
    amount: StrictInt | None
    cell: str = ""


class PokerdomUncertainty(Observation):
    field: str = Field(description="Concrete unreadable/missing required field or incomplete action-log location.")
    reason: str


class TableFormatObservation(Observation):
    kind: Literal["numbered_hh_table", "other_hh", "live_table", "not_hand", "uncertain"]
    player_rows: bool
    street_columns: bool
    numbered_actions: bool
    known_tokens: bool
    board_area: bool
    end_street_pots: bool
    evidence: list[str]


class PokerdomReadingV2(Observation):
    schema_version: Literal[2]
    format_observation: TableFormatObservation | None = None
    is_hand: bool
    action_log_complete: bool = Field(description="All numbered events through the end of this hand are visible/read. Does not require every street, payout rows, or an explicit showdown marker.")
    cropped: bool
    unit: Literal["chips", "bb", "unknown"]
    stack_semantics: Literal["before_forced", "unknown"]
    players: list[PokerdomPlayerV2]
    events: list[PokerdomEventV2]
    board: dict[Street, list[str]]
    pot_after_street: dict[Street, StrictInt | None]
    metadata: dict[str, str] = Field(default_factory=dict)
    uncertainties: list[PokerdomUncertainty] = Field(description="Blocking missing/unreadable required facts. Card backs, unplayed streets and absent optional payout/showdown rows are not uncertainties.")
    notes: list[str] = Field(description="Informational observations, including absent optional payout/showdown or unplayed streets; never invented source facts.")


# Pydantic's dynamic model deliberately replaces the optional saved-data field
# with a required, non-null field on the TG wire, without changing old v2 JSON.
PokerdomTelegramReading = create_model(
    "PokerdomTelegramReading", __base__=PokerdomReadingV2,
    format_observation=(TableFormatObservation, ...),
)


type PokerdomObservation = PokerdomReading | PokerdomReadingV2


def decode_pokerdom_reading(data: dict[str, Any]) -> PokerdomObservation:
    """Missing version means legacy v1; never reinterpret its player keys as nicknames."""
    version = data.get("schema_version", 1)
    if type(version) is not int:
        raise ValueError("schema_version must be an integer")
    if version == 1:
        return PokerdomReading.model_validate(data)
    if version == 2:
        return PokerdomReadingV2.model_validate(data)
    raise ValueError(f"unsupported Pokerdom schema_version: {version}")


class PokerdomCheck(BaseModel):
    name: str
    status: Literal["passed", "failed", "not_checked"]
    observed: int | str | None = None
    calculated: int | str | None = None
    reason: str = ""


class AmountComparison(BaseModel):
    number: int
    expected: int | None
    actual: int | None
    matched: bool
    within_tolerance: bool
    absolute_error: int | None
    relative_error: Decimal | None


class PokerdomResult(BaseModel):
    reading: PokerdomObservation
    read_status: Literal["ok", "incomplete", "not_hand", "error"]
    replay_status: Literal["passed", "failed", "not_run"] = "not_run"
    truth_status: Literal["not_scored", "matched", "within_tolerance", "mismatched"] = "not_scored"
    errors: list[str] = Field(default_factory=list)
    checks: list[PokerdomCheck] = []
    amount_comparisons: list[AmountComparison] = []
    truth_errors: list[str] = Field(default_factory=list)
    field_matches: dict[str, bool] = Field(default_factory=dict)
    missing_events: list[int] = []
    added_events: list[int] = []
    calculated_payouts: dict[str, int] = Field(default_factory=dict)
    raw: RawHand | None = None
    enriched: EnrichedHand | None = None
    outcome_confirmed: bool = False
    outcome_basis: Literal["folds", "explicit_showdown", "visible_cards", "unknown"] = "unknown"
