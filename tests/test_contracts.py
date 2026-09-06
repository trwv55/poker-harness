import pytest
from pydantic import ValidationError

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
    with pytest.raises(ValidationError):  # pytest.raises(Exception) прошёл бы и на сломанном коде
        Range(weights={"XX": 1.0})  # класса нет среди 169
    with pytest.raises(ValidationError):
        Range(weights={"AA": 1.5})  # вес вне [0, 1]


def test_fraction_of_hands():
    assert abs(Range(weights={c: 1.0 for c in all_classes()}).fraction_of_hands() - 1.0) < 1e-9
    assert abs(Range(weights={"AA": 1.0}).fraction_of_hands() - 6 / 1326) < 1e-9


def test_ev_interval_ceiling_bounds_the_error_in_both_directions():
    """Потолок цены — модуль худшего конца, и он покрывает обе стороны ошибки.

    Пас при верхнем конце +0.8 стоит не больше 0.8; вход при нижнем −0.3 стоит
    не больше 0.3. Потолок обязан быть не меньше каждой из этих величин, иначе
    строка «дороже столько-то не проиграть» была бы обещанием, которого расчёт
    не даёт.
    """
    from harness.contracts import EvInterval

    interval = EvInterval(point_bb=0.1, low_bb=-0.3, high_bb=0.8, near_zero=True)
    assert interval.cost_ceiling_bb == 0.8
    assert EvInterval(point_bb=-3.0, low_bb=-5.03, high_bb=-1.11).cost_ceiling_bb == 5.03


def test_ev_interval_rejects_a_point_outside_its_own_interval():
    """Точка вне собственного интервала — признак, что числа посчитаны врозь."""
    from harness.contracts import EvInterval

    with pytest.raises(ValidationError):
        EvInterval(point_bb=1.5, low_bb=-0.3, high_bb=0.8)
    with pytest.raises(ValidationError):
        EvInterval(point_bb=0.1, low_bb=0.8, high_bb=-0.3)  # концы перепутаны местами


# --- задача 22: полнота входа и схема наблюдений vision -----------------------


def test_a_hand_written_before_task_22_reads_as_a_whole_hand():
    """Уже записанный jsonb без поля полноты обязан читаться как рука целиком.

    Эволюция контрактов — только необязательными полями (Global Constraints
    плана): весь HH-путь писал `hands.raw` до появления `completeness`, и другой
    полноты у него не бывает.
    """
    from harness.contracts import Completeness

    stored = make_min_raw()
    assert "completeness" not in stored
    assert RawHand.model_validate(stored).completeness is Completeness.HAND


def test_a_state_hand_is_marked_as_such_and_survives_a_roundtrip():
    from harness.contracts import Completeness

    hand = RawHand.model_validate(make_min_raw(completeness="state"))
    assert hand.completeness is Completeness.STATE
    assert RawHand.model_validate_json(hand.model_dump_json()).completeness is Completeness.STATE


def test_a_decision_point_may_have_no_action_taken_yet():
    """Точка решения без действия — живой стол до хода героя.

    Судить там нечего, и `action` обязан быть необязательным: иначе состояние в
    точке решения нельзя выразить, не придумав действие, которого игрок не делал.
    """
    from harness.contracts import DecisionPoint, Street

    dp = DecisionPoint(
        index=0,
        street=Street.PREFLOP,
        label="Hero",
        position="BB",
        to_call=9000,
        pot_before=14700,
        eff_stack=36700,
        eff_stack_bb=36.7,
    )
    assert dp.action is None


def test_a_verdict_names_the_checks_it_could_not_run():
    """`not_checked` — то, чего проверить НЕ ИЗ ЧЕГО, названное поимённо."""
    from harness.contracts import ValidationStatus, Verdict

    assert Verdict(status=ValidationStatus.PASS).not_checked == []
    named = Verdict(status=ValidationStatus.PASS, not_checked=["payouts"])
    assert named.not_checked == ["payouts"]


def test_the_reading_keeps_the_two_card_renderings_apart():
    """Карты у места и карты в логе — два независимых наблюдения, а не одно.

    Измерено (реестр, «Карты отрисованы дважды»): спрошенная один раз модель
    схлопывает избыточность экрана и подставляет одно чтение в оба места.
    """
    from harness.contracts import SeenPlayer

    player = SeenPlayer(seat=1, cards_at_seat=["As", "5c"], cards_in_log=["As", "5s"])
    assert player.cards_at_seat != player.cards_in_log


def test_the_reading_keeps_the_ante_pool_apart_from_the_per_player_ante():
    """Пул анте и подушевое анте — разные поля: делит код, не модель (реестр B2)."""
    from harness.contracts import Unit, VisionReading

    reading = VisionReading(ante_pool_shown=6800.0, ante_unit=Unit.CHIPS)
    assert reading.ante_per_player_shown is None


def test_the_reading_can_refuse_a_screen_that_is_not_a_hand():
    """Честный отказ дешевле выдуманной из лобби руки (реестр C3)."""
    from harness.contracts import VisionReading

    reading = VisionReading(not_a_hand=True, refusal_reason="это лобби турнира")
    assert reading.players == []
