import pytest

from harness.contracts import Range, RawHand, all_classes, class_of


def make_min_raw(**over):
    base = {
        "provenance": "hand_history",
        "source_ref": "f.txt",
        "hand_no": "TM1",
        "tournament_id": "306148954",
        "tournament_name": "Daily Classic $4",
        "level": 23,
        "sb": 3000,
        "bb": 6000,
        "ante": 750,
        "timestamp": "2026-08-20T22:22:36",
        "table_name": "8",
        "max_seats": 8,
        "button_seat": 3,
        "seats": [{"seat": 4, "label": "Hero", "stack": 3891}],
        "posts": [],
    }
    base.update(over)
    return base


def test_raw_roundtrip():
    h = RawHand.model_validate(make_min_raw())
    assert RawHand.model_validate_json(h.model_dump_json()) == h


def test_old_json_readable_by_new_model():
    d = make_min_raw()
    d.pop("ante_type", None)  # «старый» документ без нового поля (его тут и не было)
    assert RawHand.model_validate(d).ante_type == "per_player"


def test_class_of():
    assert class_of("Kc", "3c") == "K3s"
    assert class_of("3c", "Kc") == "K3s"  # порядок карт не важен
    assert class_of("Ah", "Kd") == "AKo"
    assert class_of("Qs", "Qh") == "QQ"


def test_169_classes():
    cs = all_classes()
    assert len(cs) == 169 and len(set(cs)) == 169
    assert {"AA", "AKs", "AKo", "32o"} <= set(cs)


def test_range_validates():
    r = Range(weights={"AA": 1.0, "AKs": 0.5})
    assert r.weight("AA") == 1.0 and r.weight("72o") == 0.0
    with pytest.raises(Exception):  # noqa: B017 - brief specifies generic Exception
        Range(weights={"XX": 1.0})
    with pytest.raises(Exception):  # noqa: B017 - brief specifies generic Exception
        Range(weights={"AA": 1.5})


def test_fraction_of_hands():
    assert abs(Range(weights={c: 1.0 for c in all_classes()}).fraction_of_hands() - 1.0) < 1e-9
    assert abs(Range(weights={"AA": 1.0}).fraction_of_hands() - 6 / 1326) < 1e-9
