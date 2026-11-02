"""A described decision is a partial preflop prefix, never a played hand."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SpotStatus = Literal["known", "missing", "ambiguous"]
SpotActionKind = Literal["fold", "check", "call", "raise", "all_in"]
AmountKind = Literal["total", "additional", "behind", "unspecified"]


class SpotFact(BaseModel):
    value: str | None = None
    quote: str = ""
    status: SpotStatus = "missing"


class RawSpotPlayer(BaseModel):
    position: str
    stack: SpotFact = Field(default_factory=SpotFact)
    stack_moment: Literal["before_ante", "after_ante", "after_post", "behind", "unknown"] = (
        "unknown"
    )
    name: str | None = None


class RawSpotAction(BaseModel):
    position: str
    kind: SpotActionKind
    amount: SpotFact = Field(default_factory=SpotFact)
    amount_kind: AmountKind = "unspecified"
    quote: str


class RawDecisionSpot(BaseModel):
    source_kind: Literal["telegram_text"] = "telegram_text"
    source_text: str = ""
    table_size: SpotFact = Field(default_factory=SpotFact)
    sb: SpotFact = Field(default_factory=SpotFact)
    bb: SpotFact = Field(default_factory=SpotFact)
    ante: SpotFact = Field(default_factory=SpotFact)
    ante_type: Literal["per_player", "big_blind", "none", "unknown"] = "unknown"
    hero_position: SpotFact = Field(default_factory=SpotFact)
    hero_cards: SpotFact = Field(default_factory=SpotFact)
    players: list[RawSpotPlayer] = []
    actions: list[RawSpotAction] = []
    unknown_lines: list[str] = Field(default_factory=list)
    question: str | None = None


class SpotExtraction(BaseModel):
    is_spot: bool
    spot: RawDecisionSpot | None = None


class SpotPlayer(BaseModel):
    position: str
    stack: int | None = None  # initial stack, not chips behind
    folded: bool = False


class SpotAction(BaseModel):
    position: str
    kind: SpotActionKind
    committed_after: int | None = None
    all_in: bool = False
    quote: str


class CanonicalDecisionSpot(BaseModel):
    source_kind: Literal["telegram_text"] = "telegram_text"
    table_size: int
    sb: int
    bb: int
    ante: int
    ante_type: Literal["per_player", "big_blind", "none"]
    hero_position: str
    hero_cards: tuple[str, str] | None = None
    hero_class: str | None = None
    players: list[SpotPlayer]
    actions: list[SpotAction]
    hero_to_act: bool = True
    not_checked: list[str] = Field(default_factory=list)


class DecisionSnapshot(BaseModel):
    hero_position: str
    positions_still_to_act: list[str]
    pot_before: int
    to_call: int
    committed: dict[str, int]
    effective_stack: int | None
    possible_side_pots: bool
    not_checked: list[str]


class SpotCallScenario(BaseModel):
    spot_id: int
    condition: str
    pot_before: int
    to_call: int
    eligible_pot: int | None
    equity_threshold: float | None
    opponent_range: str | None = None
    range_source: Literal["user_range", "model_assumption"] | None = None
    equity: float | None = None
    excluded_blocked_combos: int | None = None
    explanation: str
