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
    acts = {(a.label, a.kind): a for a in h.actions}
    # ActionKind — StrEnum; сравнение по значению верно в рантайме, но
    # pyright не считает str-литерал подтипом ActionKind при subscript.
    assert acts[("5553a2cd", "raise")].committed_after == 69000  # type: ignore[reportArgumentType]
    # Hero: SB 3000 + call 141 = 3144? нет: блайнд 3000 + доплата 141 = 3141
    assert acts[("Hero", "call")].committed_after == 3141  # type: ignore[reportArgumentType]
    assert acts[("Hero", "call")].is_all_in  # type: ignore[reportArgumentType]


def test_heads_up_button_is_sb():
    # синтетика: 2 игрока, кнопка = SB
    raw = parse_hand(SAMPLE, source_ref="x").model_copy(deep=True)
    raw.seats = raw.seats[:2]  # места 1 и 2
    raw.button_seat = raw.seats[0].seat
    raw.posts = [p for p in raw.posts if p.label in {raw.seats[0].label, raw.seats[1].label}]
    raw.actions, raw.dealt, raw.showdowns, raw.collected = [], {}, [], []
    h = normalize(raw)
    assert [p.position for p in h.players] == ["BTN", "BB"]  # HU: кнопка ставит SB
