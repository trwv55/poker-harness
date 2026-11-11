"""A described hand is a versioned scenario, never a played hand history."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, field_validator

StreetName = Literal["preflop", "flop", "turn", "river"]
ActionName = Literal["fold", "check", "call", "bet", "raise", "all_in"]


class NaturalPlayer(BaseModel):
    position: str
    stack_bb: str | None = None
    stack_moment: Literal["before_action", "after_action"] = "before_action"
    quote: str = ""


class NaturalAction(BaseModel):
    street: StreetName = "preflop"
    position: str
    kind: ActionName
    amount: str | None = None
    amount_unit: Literal["bb", "chips", "pot_percent", "times"] = "bb"
    amount_mode: Literal["total", "additional"] = "total"
    quote: str


class NaturalHand(BaseModel):
    schema_version: int = 2
    table_size: int | None = None
    sb_chips: str | None = None
    bb_chips: str | None = None
    ante: str | None = None
    ante_unit: Literal["bb", "chips"] = "bb"
    hero_position: str | None = None
    hero_cards: str | None = None
    hero_quote: str = ""
    players: list[NaturalPlayer] = []
    actions: list[NaturalAction] = []
    flop: str | None = None
    turn: str | None = None
    river: str | None = None
    flop_quote: str = ""
    turn_quote: str = ""
    river_quote: str = ""
    question: str | None = None
    context: str | None = None
    # Provenance is attached to each extracted item, and defaults are recorded by
    # the normalizer. This text stays in the session DB, never in HH tables.
    source_text: str = ""


class NaturalRead(BaseModel):
    intent: Literal["new_hand", "correction", "other"]
    hand: NaturalHand | None = None

    @field_validator("hand", mode="before")
    @classmethod
    def parse_stringified_hand(cls, value: object) -> object:
        # The current OpenRouter Qwen endpoint stringifies nested tool JSON.
        if isinstance(value, str) and len(value) <= 20_000:
            return json.loads(value)
        return value


class ScenarioValue(BaseModel):
    value: Decimal
    source: Literal["player", "default", "derived"]
    note: str = ""


class ScenarioAction(BaseModel):
    street: StreetName
    position: str
    kind: ActionName
    committed_after: Decimal
    all_in: bool = False
    source: Literal["player", "default", "derived"] = "player"
    note: str = ""


class ScenarioHand(BaseModel):
    source_kind: Literal["telegram_text"] = "telegram_text"
    table_size: ScenarioValue
    ante_bb: ScenarioValue
    hero_position: str
    hero_cards: tuple[str, str] | None = None
    hero_class: str | None = None
    board: list[str] = Field(default_factory=list)
    board_source: Literal["player", "default"] = "player"
    stacks_behind_bb: dict[str, ScenarioValue]
    actions: list[ScenarioAction]
    question: str | None = None
    context: str | None = None
    assumptions: list[str] = Field(default_factory=list)


class ScenarioSnapshot(BaseModel):
    street: StreetName
    pot_before_bb: Decimal
    to_call_bb: Decimal
    committed_bb: dict[str, Decimal]
    remaining_bb: dict[str, Decimal]
    live_positions: list[str]
    positions_after_hero: list[str]
    assumed_folds: list[str] = Field(default_factory=list)
    possible_side_pots: bool = False
