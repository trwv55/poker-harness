"""Нормализация `RawHand` в `CanonicalHand` — единственное место канонизации.

Приводит сырьё от любого источника (hand history, скрин) к единому канону:
позиции выводятся из кнопки (а не из блайндов), суммы ставок унифицируются в
«итого поставлено на улице после действия», личности игроков классифицируются
(hero/nick/anon). Про форматы источников этот модуль не знает ничего — он
работает только со структурированным `RawHand`.
"""

from __future__ import annotations

from harness.contracts import (
    ActionKind,
    CanonicalAction,
    CanonicalHand,
    Identity,
    PlayerState,
    PostKind,
    RawAction,
    RawHand,
    Street,
)

# Позиции в порядке мест от SB. Для HU (2 игрока) кнопка сама ставит малый
# блайнд, поэтому порядок мест начинается с кнопки, а не со следующего места.
POSITIONS_BY_COUNT: dict[int, list[str]] = {
    2: ["BTN", "BB"],  # HU: BTN = SB
    3: ["SB", "BB", "BTN"],
    4: ["SB", "BB", "UTG", "BTN"],
    5: ["SB", "BB", "UTG", "CO", "BTN"],
    6: ["SB", "BB", "UTG", "HJ", "CO", "BTN"],
    7: ["SB", "BB", "UTG", "UTG+1", "HJ", "CO", "BTN"],
    8: ["SB", "BB", "UTG", "UTG+1", "LJ", "HJ", "CO", "BTN"],
    9: ["SB", "BB", "UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN"],
}


def _positions_by_label(raw: RawHand) -> dict[str, str]:
    """Метки позиций по занятым местам, идущим по кругу от кнопки.

    Порядок начинается со следующего после кнопки места; для HU (2 игрока) —
    с самой кнопки, т.к. в хедз-апе кнопка ставит малый блайнд.
    """
    occupied = sorted(seat.seat for seat in raw.seats)
    n = len(occupied)
    labels = POSITIONS_BY_COUNT[n]
    btn_idx = occupied.index(raw.button_seat)
    start = btn_idx if n == 2 else (btn_idx + 1) % n
    order = [occupied[(start + i) % n] for i in range(n)]
    seat_to_label = {seat.seat: seat.label for seat in raw.seats}
    return {seat_to_label[seat_no]: pos for seat_no, pos in zip(order, labels, strict=True)}


def _identity(label: str, raw: RawHand) -> Identity:
    if label == "Hero":
        return Identity.HERO
    if raw.vision is not None and label in raw.vision.nicknames:
        return Identity.NICK
    return Identity.ANON


def _canonical_actions(raw: RawHand) -> list[CanonicalAction]:
    """Экшены `RawHand` -> `CanonicalAction` с унифицированным `committed_after`.

    Блайнды входят в префлоп-коммит (анте — нет, оно в банке отдельно).
    `calls N`/`bets N` — доплата (прибавка к уже поставленному на улице);
    `raises X to Y` — установка итога в Y.
    """
    committed: dict[tuple[Street, str], int] = {}
    for post in raw.posts:
        if post.kind in (PostKind.SMALL_BLIND, PostKind.BIG_BLIND):
            key = (Street.PREFLOP, post.label)
            committed[key] = committed.get(key, 0) + post.amount

    return [
        CanonicalAction(
            street=action.street,
            label=action.label,
            kind=action.kind,
            committed_after=_update_committed(committed, action),
            is_all_in=action.is_all_in,
            raw_line=action.raw_line,
        )
        for action in raw.actions
    ]


def _update_committed(committed: dict[tuple[Street, str], int], action: RawAction) -> int:
    """Обновить накопленный коммит по `(street, label)` действием и вернуть новое значение.

    Если источник не дал нужного поля (у `raise` нет `to_amount`, у `call`/`bet` нет
    `amount` — схема это допускает, и vision-вход задачи 22 даст именно такое усечение),
    аккумулятор остаётся на последнем известном значении: нормалайзер не придумывает
    сумму. Получившаяся несходимость — сигнал для движка и валидатора (задача 6),
    которые эскалируют её игроку; падать здесь нельзя — краш обходит машину эскалаций,
    ради которой на vision-пути всё и затевалось.
    """
    key = (action.street, action.label)
    if action.kind == ActionKind.RAISE and action.to_amount is not None:
        committed[key] = action.to_amount
    elif action.kind in (ActionKind.CALL, ActionKind.BET) and action.amount is not None:
        committed[key] = committed.get(key, 0) + action.amount
    # FOLD/CHECK, либо raise/call/bet с отсутствующим полем — коммит не меняем,
    # читаем уже накопленное (0, если ещё не ставил).
    return committed.get(key, 0)


def normalize(raw: RawHand) -> CanonicalHand:
    """Привести `RawHand` к канону: позиции, унифицированные ставки, identity."""
    positions = _positions_by_label(raw)
    players = [
        PlayerState(
            seat=seat.seat,
            label=seat.label,
            identity=_identity(seat.label, raw),
            position=positions[seat.label],
            stack=seat.stack,
            stack_bb=seat.stack / raw.bb,
        )
        for seat in raw.seats
    ]

    bounties: dict[str, int] | None = None
    bounty_source: str | None = None
    if raw.vision is not None and raw.vision.bounties:
        bounties = dict(raw.vision.bounties)
        bounty_source = "vision"

    return CanonicalHand(
        provenance=raw.provenance,
        tournament_id=raw.tournament_id,
        hand_no=raw.hand_no,
        hand_index=None,  # проставляет воркер при сохранении (задача 18)
        level=raw.level,
        sb=raw.sb,
        bb=raw.bb,
        ante=raw.ante,
        ante_type=raw.ante_type,
        timestamp=raw.timestamp,
        button_seat=raw.button_seat,
        players=players,
        dealt=raw.dealt,
        posts=raw.posts,
        actions=_canonical_actions(raw),
        boards=raw.boards,
        uncalled=raw.uncalled,
        showdowns=raw.showdowns,
        collected=raw.collected,
        summary=raw.summary,
        bounties=bounties,
        bounty_source=bounty_source,
        vision=raw.vision,
    )
