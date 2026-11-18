"""Strict observation adapter and checks. No infrastructure imports or repairs."""
from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from harness.contracts.pokerdom_vision import (
    AmountComparison,
    PokerdomCheck,
    PokerdomObservation,
    PokerdomReading,
    PokerdomReadingV2,
    PokerdomResult,
    PokerdomTelegramReading,
)
from harness.contracts.raw import (
    ActionKind,
    Post,
    PostKind,
    RawAction,
    RawHand,
    SeatInfo,
    ShowdownEntry,
    Street,
    Uncalled,
    VisionMeta,
)
from harness.engine import enrich
from harness.normalizer import normalize

STREETS = list(Street)
POSTS = {"Ante": PostKind.ANTE, "SB": PostKind.SMALL_BLIND, "BB": PostKind.BIG_BLIND}
ACTIONS = {"Fold": ActionKind.FOLD, "Check": ActionKind.CHECK, "C": ActionKind.CALL,
           "B": ActionKind.BET, "R": ActionKind.RAISE}
MONEY = {*POSTS, "C", "B", "R", "UB"}
PROMPT_PATH = Path(__file__).with_name("prompts") / "pokerdom-v2.md"
TG_PROMPT_PATH = Path(__file__).with_name("prompts") / "pokerdom-tg-v2.md"


async def read_pokerdom(llm: Any, image: bytes, *, trace_id: int,
                        provider: str, observer: Any, detect_format: bool = False) -> tuple[PokerdomReadingV2, Any]:
    """Only image and format instructions reach the injected facade; no case truth/hero."""
    schema = PokerdomTelegramReading if detect_format else PokerdomReadingV2
    prompt_path = TG_PROMPT_PATH if detect_format else PROMPT_PATH
    return await llm("vision_extract", schema, prompt=prompt_path.read_text(),
                     images=[image], trace_id=trace_id, experiment_provider=provider,
                     experiment_observer=observer)


def reading_rows(reading: PokerdomObservation, *, strict: bool = False) -> PokerdomReading:
    """Versioned internal row view. Source reading stays unchanged; v1 remains strict."""
    if isinstance(reading, PokerdomReading):
        return reading
    nicknames = [p.nickname for p in reading.players]
    unique = {
        p.nickname: f"P{i}" for i, p in enumerate(reading.players, 1)
        if p.nickname and p.nickname_complete and nicknames.count(p.nickname) == 1
    }
    if strict:
        if len(unique) != len(reading.players):
            raise ValueError("unknown, duplicate or truncated player nickname")
        if any(e.player_nickname not in unique for e in reading.events):
            raise ValueError("event nickname unknown or ambiguous; exact match required")
        if any(p.showdown_shown and len(p.cards) != 2 for p in reading.players):
            raise ValueError("explicit showdown cards missing")
    return PokerdomReading.model_validate({
        "schema_version": 1, "is_hand": reading.is_hand,
        "complete": reading.action_log_complete, "cropped": reading.cropped,
        "unit": reading.unit, "stack_semantics": reading.stack_semantics,
        "players": [{**p.model_dump(exclude={"nickname_complete"}),
                     "nickname": p.nickname or "", "row_id": f"P{i}"}
                    for i, p in enumerate(reading.players, 1)],
        "events": [{**e.model_dump(exclude={"player_nickname"}),
                    "player": unique.get(e.player_nickname or "", f"unresolved-nickname:{e.player_nickname}")}
                   for e in reading.events],
        "board": reading.board, "pot_after_street": reading.pot_after_street,
        "metadata": reading.metadata,
        "issues": [f"{u.field}: {u.reason}" for u in reading.uncertainties],
    })


def _terminal_log(reading: PokerdomReading) -> None:
    """Conservative completed-log gate for v2; the shared engine still checks legality.

    Supported terminal evidence: explicit fold-out, or a closed river round.
    Unknown all-in/runout semantics are diagnostic, never generated continuations.
    """
    events = sorted(reading.events, key=lambda e: e.number)
    actions = [e for e in events if e.token in ACTIONS]
    folded = {e.player for e in actions if e.token == "Fold"}
    alive = {p.row_id for p in reading.players} - folded
    if len(alive) == 1 and actions and actions[-1].token == "Fold":
        return
    if len(alive) < 2 or not actions or actions[-1].street != Street.RIVER:
        raise ValueError("action log has no supported terminal evidence (fold-out or closed river)")
    river = [e for e in actions if e.street == Street.RIVER]
    aggression = max((e.number for e in river if e.token in {"B", "R"}), default=0)
    aggressor = next((e.player for e in river if e.number == aggression), None)
    responders = {e.player for e in river if e.number > aggression}
    required = alive - ({aggressor} if aggressor else set())
    commits = {p: sum(e.amount or 0 for e in river if e.player == p and e.token in {"R", "B", "C"})
               for p in alive}
    if not required.issubset(responders) or len(set(commits.values())) != 1:
        raise ValueError("river action log is not closed; missing response or unsupported all-in")
    # No extra outcome/payout observations are manufactured here.


def _card(card: str) -> str:
    value = card.replace("10", "T")
    if not re.fullmatch(r"[2-9TJQKA][cdhs]", value):
        raise ValueError(f"invalid card: {card}")
    return value


def _prepare(reading: PokerdomReading, nickname: str | None) -> RawHand:
    if not reading.complete or reading.cropped or reading.issues:
        raise ValueError("incomplete table: " + "; ".join(reading.issues))
    if reading.unit != "chips" or reading.stack_semantics != "before_forced":
        raise ValueError("unsupported units or unknown stack semantics")
    players = reading.players
    ids = [p.row_id for p in players]
    if len(ids) != len(set(ids)) or not 2 <= len(ids) <= 9:
        raise ValueError("duplicate rows or unsupported table size")
    if any(not p.nickname or p.stack_shown is None or p.stack_shown <= 0 for p in players):
        raise ValueError("unknown nickname/starting stack")
    heroes = [p.row_id for p in players if nickname and p.nickname == nickname]
    if len(heroes) != 1:
        raise ValueError("hero nickname absent or ambiguous; pass room-specific nickname")
    buttons = [i + 1 for i, p in enumerate(players) if p.button_shown]
    if len(buttons) != 1:
        raise ValueError("button absent or ambiguous")
    events = sorted(reading.events, key=lambda e: e.number)
    if not events or [e.number for e in events] != list(range(1, len(events) + 1)):
        raise ValueError("event numbers: duplicate or gap (must start at 1)")
    if any(e.player not in ids for e in events):
        raise ValueError("event references unknown player")
    if [STREETS.index(e.street) for e in events] != sorted(STREETS.index(e.street) for e in events):
        raise ValueError("event number contradicts street")
    labels = {pid: "Hero" if pid == heroes[0] else pid for pid in ids}
    raw = RawHand(max_seats=len(ids), button_seat=buttons[0],
                  vision=VisionMeta(nicknames={labels[p.row_id]: p.nickname for p in players}))
    raw.seats = [SeatInfo(seat=i+1, label=labels[p.row_id], stack=p.stack_shown)
                 for i, p in enumerate(players) if p.stack_shown is not None]
    all_cards: list[str] = []
    for p in players:
        cards = [_card(c) for c in p.cards]
        if len(cards) not in (0, 2):
            raise ValueError(f"incomplete cards: {p.row_id}")
        all_cards.extend(cards)
        if cards:
            raw.dealt[labels[p.row_id]] = cards
        if p.showdown_shown:
            raw.showdowns.append(ShowdownEntry(label=labels[p.row_id], cards=cards))
    terminal = events[-1].street
    for street in STREETS[1:STREETS.index(terminal)+1]:
        cards = [_card(c) for c in reading.board.get(street, [])]
        if len(cards) != (3 if street == Street.FLOP else 1):
            raise ValueError(f"incomplete board: {street}")
        raw.boards[street] = cards
        all_cards.extend(cards)
    if any(cards for street, cards in reading.board.items() if STREETS.index(street) > STREETS.index(terminal)):
        raise ValueError("board beyond terminal event street")
    if len(all_cards) != len(set(all_cards)):
        raise ValueError("card collision")
    committed: dict[tuple[Street, str], int] = {}
    seen_action = False
    for e in events:
        label = labels[e.player]
        if e.token not in {*POSTS, *ACTIONS, "UB"}:
            raise ValueError(f"event {e.number}: unknown token {e.token}")
        if e.token in MONEY:
            if e.amount is None or e.amount < 0 or e.amount_text is None:
                raise ValueError(f"event {e.number}: missing/negative amount")
            printed = e.amount_text.replace(" ", "").replace("\u00a0", "")
            if not re.fullmatch(r"\d+", printed) or int(printed) != e.amount:
                raise ValueError(f"event {e.number}: amount is not exact integer chips")
        elif e.amount is not None or e.amount_text not in (None, ""):
            raise ValueError(f"event {e.number}: unexpected amount")
        amount = e.amount or 0
        key = (e.street, label)
        if e.token in POSTS:
            if seen_action or e.street != Street.PREFLOP:
                raise ValueError(f"event {e.number}: forced post after action")
            raw.posts.append(Post(label=label, kind=POSTS[e.token], amount=amount))
            if e.token != "Ante":
                committed[key] = committed.get(key, 0) + amount
        elif e.token == "UB":
            other = max((value for (street, player), value in committed.items()
                         if street == e.street and player != label), default=0)
            owed = max(committed.get(key, 0) - other, 0)
            if amount <= 0 or amount != owed or e.number != events[-1].number:
                raise ValueError(f"event {e.number}: UB does not match uncalled contribution")
            raw.uncalled.append(Uncalled(label=label, amount=amount))
            committed[key] = committed.get(key, 0) - amount
        else:
            seen_action = True
            kind = ACTIONS[e.token]
            action = RawAction(street=e.street, label=label, kind=kind,
                               raw_line=f"#{e.number} {e.token} {e.amount_text or ''}")
            if e.token in {"C", "B", "R"}:
                committed[key] = committed.get(key, 0) + amount
                if kind == ActionKind.RAISE:
                    action.to_amount = committed[key]
                else:
                    action.amount = amount
            raw.actions.append(action)
    for token, field in (("SB", "sb"), ("BB", "bb")):
        posts = [e for e in events if e.token == token]
        if len(posts) != 1 or not posts[0].amount:
            raise ValueError(f"missing/duplicate {token}")
        setattr(raw, field, posts[0].amount)
    antes = [p.amount for p in raw.posts if p.kind == PostKind.ANTE]
    if (len(antes) != len(players) or len(set(antes)) != 1 or
            {p.label for p in raw.posts if p.kind == PostKind.ANTE} != set(labels.values())):
        raise ValueError("per-player ante missing or inconsistent")
    raw.ante = antes[0]
    return raw


def process_pokerdom(reading: PokerdomObservation, *, nickname: str | None = None,
                     expected: PokerdomObservation | None = None) -> PokerdomResult:
    result = PokerdomResult(reading=reading, read_status="ok" if reading.is_hand else "not_hand")
    if reading.is_hand:
        try:
            rows = reading_rows(reading, strict=True)
            candidate = _prepare(rows, nickname)
            if isinstance(reading, PokerdomReadingV2):
                _terminal_log(rows)
            result.raw = candidate
        except ValueError as exc:
            result.read_status = "incomplete"
            result.errors.append(str(exc))
        else:
            reading = rows
            phase = "normalize"
            try:
                canonical = normalize(result.raw)
                phase = "replay"
                result.enriched = enrich(canonical)
            except Exception as exc:  # noqa: BLE001 — diagnostic result preserves transcription
                result.replay_status = "failed"
                result.errors.append(f"{phase}: {type(exc).__name__}: {exc}")
            else:
                en = result.enriched
                result.replay_status = "passed" if en.verdict.status.value == "pass" else "failed"
                result.errors.extend(en.verdict.reasons)
                terminal = max(reading.events, key=lambda e: e.number).street
                for street, observed in reading.pot_after_street.items():
                    calculated = (en.report.final_pot if street == terminal
                                  else en.report.pot_by_street.get(street))
                    status = "not_checked" if observed is None or calculated is None else (
                        "passed" if observed == calculated else "failed")
                    result.checks.append(PokerdomCheck(name=f"pot_after.{street}", status=status,
                                                      observed=observed, calculated=calculated))
                for street in dict.fromkeys(e.street for e in reading.events):
                    if reading.pot_after_street.get(street) is None:
                        result.checks.append(PokerdomCheck(name=f"pot_after.{street}",
                            status="not_checked", reason="no observed end-of-street pot"))
                folded = {e.player for e in reading.events if e.token == "Fold"}
                alive = [p for p in reading.players if p.row_id not in folded]
                result.outcome_confirmed = len(alive) == 1 or (
                    len(alive) > 1 and all(p.showdown_shown and len(p.cards) == 2 for p in alive))
                result.outcome_basis = ("folds" if len(alive) == 1 else
                    "explicit_showdown" if result.outcome_confirmed else
                    "visible_cards" if terminal == Street.RIVER and all(len(p.cards) == 2 for p in alive)
                    else "unknown")
                result.checks.extend([
                    PokerdomCheck(name="room_payout", status="not_checked", reason="no payout row"),
                    PokerdomCheck(name="room_end_stacks", status="not_checked", reason="no end-stack row"),
                    PokerdomCheck(name="outcome", status="passed" if result.outcome_confirmed else "not_checked",
                                  reason="computed from folds/explicit showdown" if result.outcome_confirmed
                                  else "showdown not independently observed"),
                ])
                for reason in en.verdict.not_checked:
                    result.checks.append(PokerdomCheck(name=reason, status="not_checked"))
                if result.outcome_basis != "unknown" and result.replay_status == "passed":
                    contributions = dict.fromkeys((p.label for p in en.hand.players), 0)
                    labels = {p.row_id: raw_seat.label for p, raw_seat in
                              zip(reading.players, result.raw.seats, strict=True)}
                    for event in reading.events:
                        if event.token in MONEY:
                            contributions[labels[event.player]] += (event.amount or 0) * (
                                -1 if event.token == "UB" else 1)
                    result.calculated_payouts = {
                        p.label: en.report.stacks_end[p.label] - p.stack + contributions[p.label]
                        for p in en.hand.players
                        if en.report.stacks_end[p.label] - p.stack + contributions[p.label] > 0}
                if any(c.status == "failed" for c in result.checks):
                    result.replay_status = "failed"
    if expected is not None:
        score_pokerdom(result, expected)
    return result


def score_pokerdom(result: PokerdomResult, expected: PokerdomObservation) -> None:
    """Exact visible fields; only printed event amounts receive the inclusive 10% rule."""
    # Compare observation semantics across versions; never repair a malformed v1 player key.
    actual = reading_rows(result.reading).model_dump(mode="json")
    wanted = reading_rows(expected).model_dump(mode="json")
    result.field_matches.clear()
    # Incidental diagnostic wording and cell coordinates are not truth fields.
    for data in (actual, wanted):
        data.pop("issues")
        # Row identifiers are local keys, not text printed by the room.
        # Compare the row relation, while retaining every original key in the reading.
        row_keys = {p["row_id"]: f"row.{index}" for index, p in enumerate(data["players"], 1)}
        for event in data["events"]:
            event["player"] = row_keys.get(event["player"], f"unresolved:{event['player']}")
        for index, player in enumerate(data["players"], 1):
            player["row_id"] = f"row.{index}"
        for player in data["players"]:
            player["cards"] = [c.replace("10", "T") for c in player["cards"]]
        data["board"] = {street: [c.replace("10", "T") for c in cards]
                         for street, cards in data["board"].items() if cards}
        data["pot_after_street"] = {street: value for street, value in
                                    data["pot_after_street"].items() if value is not None}
        data["events"] = sorted(data["events"], key=lambda e: e["number"])
        for event in data["events"]:
            event.pop("cell")
            event.pop("amount_text")
    aevents, wevents = actual.pop("events"), wanted.pop("events")
    errors = []
    for field in wanted:
        result.field_matches[field] = actual[field] == wanted[field]
        if actual[field] != wanted[field]:
            errors.append(field)
    result.missing_events = sorted({e["number"] for e in wevents} - {e["number"] for e in aevents})
    result.added_events = sorted({e["number"] for e in aevents} - {e["number"] for e in wevents})
    for index, player in enumerate(wanted["players"]):
        actual_player = actual["players"][index] if index < len(actual["players"]) else {}
        for field in ("nickname", "stack_shown", "cards", "button_shown", "showdown_shown"):
            result.field_matches[f"row.{index+1}.{field}"] = actual_player.get(field) == player[field]
    for street, cards in wanted["board"].items():
        result.field_matches[f"board.{street}"] = actual["board"].get(street) == cards
    for street, pot in wanted["pot_after_street"].items():
        result.field_matches[f"pot_after.{street}"] = actual["pot_after_street"].get(street) == pot
    comparisons = []
    for index in range(max(len(aevents), len(wevents))):
        if index >= len(aevents) or index >= len(wevents):
            errors.append(f"events[{index}]: missing/added")
            continue
        a, w = aevents[index].copy(), wevents[index].copy()
        av, wv = a.pop("amount"), w.pop("amount")
        for field in ("number", "player", "street", "token"):
            result.field_matches[f"event.{w['number']}.{field}"] = a[field] == w[field]
        if a != w:
            errors.append(f"events[{index}]: number/player/street/token")
        money = w["token"] in MONEY
        matched = av == wv
        within = (av is not None and wv is not None and wv >= 0
                  and abs(av-wv)*10 <= wv)
        if money:
            result.field_matches[f"event.{w['number']}.amount"] = matched and wv is not None
            comparisons.append(AmountComparison(number=w["number"], expected=wv, actual=av,
                matched=matched and wv is not None, within_tolerance=within,
                absolute_error=abs(av-wv) if av is not None and wv is not None else None,
                relative_error=Decimal(abs(av-wv))/Decimal(wv) if av is not None and wv else None))
            if not within:
                errors.append(f"event {w['number']}: amount outside tolerance or missing")
        elif not matched:
            errors.append(f"event {w['number']}: unexpected amount")
    result.amount_comparisons = comparisons
    result.truth_errors = errors
    result.truth_status = "mismatched" if errors else (
        "matched" if all(c.matched for c in comparisons) else "within_tolerance")
