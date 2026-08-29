from datetime import datetime

from harness.contracts import (
    ActionKind,
    Post,
    PostKind,
    Provenance,
    RawAction,
    RawHand,
    SeatInfo,
    Street,
)
from harness.normalizer import normalize
from harness.parsers.hh_parser import parse_hand
from tests.test_hh_parser import SAMPLE


def test_positions_and_bb():
    h = normalize(parse_hand(SAMPLE, source_ref="x"))
    pos = {p.label: p.position for p in h.players}
    assert pos["Hero"] == "SB" and pos["fcc9bf19"] == "BB"
    assert pos["c3986130"] == "BTN" and pos["5553a2cd"] == "UTG"
    hero = next(p for p in h.players if p.label == "Hero")
    assert hero.identity == "hero" and abs(hero.stack_bb - 3891 / 6000) < 1e-9
    anon = next(p for p in h.players if p.label == "c30a7c9e")
    assert anon.identity == "anon"


def test_committed_after_unified():
    h = normalize(parse_hand(SAMPLE, source_ref="x"))
    acts = {(a.label, a.kind): a for a in h.actions}   # ключ типизирован ActionKind —
    assert acts[("5553a2cd", ActionKind.RAISE)].committed_after == 69000   # строковый литерал
    # Hero: блайнд 3000 + доплата 141 = 3141 (источник пишет доплату, канон — итог)
    assert acts[("Hero", ActionKind.CALL)].committed_after == 3141
    assert acts[("Hero", ActionKind.CALL)].is_all_in


def test_heads_up_button_is_sb():
    # синтетика: 2 игрока, кнопка = SB
    raw = parse_hand(SAMPLE, source_ref="x").model_copy(deep=True)
    raw.seats = raw.seats[:2]  # места 1 и 2
    raw.button_seat = raw.seats[0].seat
    raw.posts = [p for p in raw.posts if p.label in {raw.seats[0].label, raw.seats[1].label}]
    raw.actions, raw.dealt, raw.showdowns, raw.collected = [], {}, [], []
    h = normalize(raw)
    assert [p.position for p in h.players] == ["BTN", "BB"]  # HU: кнопка ставит SB


def test_raise_missing_to_amount_does_not_crash():
    # vision-вход (задача 22) может не распознать итоговую сумму рейза. Нормалайзер не
    # должен падать и не должен придумывать число — коммит остаётся на последнем
    # известном значении (здесь — блайнд), расхождение эскалирует движок/валидатор
    # задачи 6.
    raw = RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="x",
        hand_no="T1",
        tournament_id="1",
        tournament_name="T",
        level=1,
        sb=3000,
        bb=6000,
        ante=0,
        timestamp=datetime(2026, 8, 20, 22, 22, 36),  # noqa: DTZ001
        table_name="1",
        max_seats=2,
        button_seat=1,
        seats=[SeatInfo(seat=1, label="A", stack=100_000), SeatInfo(seat=2, label="B", stack=100_000)],
        posts=[
            Post(label="A", kind=PostKind.SMALL_BLIND, amount=3000),
            Post(label="B", kind=PostKind.BIG_BLIND, amount=6000),
        ],
        actions=[
            RawAction(
                street=Street.PREFLOP,
                label="A",
                kind=ActionKind.RAISE,
                to_amount=None,  # vision не распознал итог
                raw_line="A: raises ? to ?",
            )
        ],
    )

    h = normalize(raw)  # не должно бросать исключение

    act = h.actions[0]
    assert act.kind == ActionKind.RAISE
    assert act.committed_after == 3000  # последнее известное значение (блайнд), не выдумка
