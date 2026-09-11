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


def test_a_decision_point_cannot_be_built_without_the_action_it_judges():
    """Точка решения без сыгранного действия невыразима — так держит тип.

    Вердикт сравнивает сыгранное с лучшим, и точка без действия была бы точкой,
    про которую разбору нечего сказать.
    """
    from harness.contracts import DecisionPoint, Street

    with pytest.raises(ValidationError):
        DecisionPoint(
            index=0,
            street=Street.PREFLOP,
            label="Hero",
            position="BB",
            to_call=9000,
            pot_before=14700,
            eff_stack=36700,
            eff_stack_bb=36.7,
        )  # pyright: ignore[reportCallIssue]


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


# --- таксономия ликов (задача 23) ----------------------------------------------------


def _leak_point(*, spot, action_taken, best_action, ev_diff_bb=-1.0, **over):
    """Точка решения ровно с той тройкой, по которой опознаётся тип лика."""
    from harness.contracts import PointVerdict, Street, Zone

    return PointVerdict(
        dp_index=0,
        street=Street.PREFLOP,
        spot=spot,
        zone=Zone.STRICT,
        action_taken=action_taken,
        best_action=best_action,
        ev_diff_bb=ev_diff_bb,
        **over,
    )


def test_every_leak_rule_recognises_its_own_point():
    """Таблица правил и опознание по ней — одно и то же, а не два списка.

    Правило заводится строкой, и строка обязана быть достаточной: точка,
    собранная ИЗ правила, обязана этим же правилом и опознаться.
    """
    from harness.contracts import LEAK_RULES, leak_rule_of_point

    for rule in LEAK_RULES:
        point = _leak_point(
            spot=rule.spot, action_taken=rule.action_taken, best_action=rule.best_action
        )
        assert leak_rule_of_point(point) is rule


def test_leak_rule_keys_and_titles_are_unique():
    """Два типа с одним ключом молча слились бы в один столбец агрегата."""
    from harness.contracts import LEAK_RULES

    assert len({rule.key for rule in LEAK_RULES}) == len(LEAK_RULES)
    assert len({rule.title for rule in LEAK_RULES}) == len(LEAK_RULES)


def test_a_near_zero_point_matches_no_leak_rule():
    """«Около нуля» — не лик: упрекать не за что, и цена решения ноль.

    Точка такой формы несёт в `best_action` русскую фразу ядра, а не токен
    действия, поэтому ни одна тройка таблицы с ней не совпадает.
    """
    from harness.contracts import EvInterval, SpotKind, leak_rule_of_point

    point = _leak_point(
        spot=SpotKind.PUSHFOLD_UNOPENED,
        action_taken="fold",
        best_action="около нуля, оба варианта допустимы",
        ev_diff_bb=0.0,
        interval=EvInterval(point_bb=0.1, low_bb=-0.3, high_bb=0.4, near_zero=True),
    )
    assert leak_rule_of_point(point) is None


def test_an_unjudged_point_matches_no_leak_rule():
    """Точка без вердикта несёт пустой `best_action` — это «не посчитано», не лик."""
    from harness.contracts import SpotKind, leak_rule_of_point

    point = _leak_point(
        spot=SpotKind.POSTFLOP, action_taken="fold", best_action="", ev_diff_bb=0.0
    )
    assert leak_rule_of_point(point) is None


def test_the_reserved_open_raise_leak_matches_nothing_the_core_judges_today():
    """«Открывает слишком широко» зарезервирован под чарты и сегодня пуст.

    Держится не намерением, а тем, что судимых спотов ровно два
    (`JUDGED_SPOTS`), и спот этого правила в них не входит: вердикта с таким
    спотом ядро не выносит, значит и совпасть правилу не с чем.
    """
    from harness.contracts import JUDGED_SPOTS, LEAK_RULES

    reserved = next(rule for rule in LEAK_RULES if rule.key == "open_too_wide")
    assert reserved.spot not in JUDGED_SPOTS


def test_the_core_judges_a_point_by_the_predicate_of_the_contracts():
    """У правила «точка судима» одна формулировка, а не одна на каждого читателя.

    Ядро ранжирует точки, память пишет колонку `decision_points.judged` —
    и оба зовут ОДНУ функцию, а не две одинаковых. Проверяется тождеством
    объектов: копия, сделанная «по образцу», этот тест не пройдёт.
    """
    from harness.analysis import error_cost
    from harness.contracts import is_judged

    assert error_cost.is_judged is is_judged


def test_a_leak_rule_is_found_by_the_raw_strings_of_jsonb():
    """Память группирует точки в SQL и приносит спот строкой, а не `SpotKind`.

    Правило обязано опознаваться и так: иначе агрегат из БД не совпал бы ни с
    одним типом, и экран «Мои лики» был бы пуст при полной базе разборов.
    """
    from harness.contracts import LEAK_RULES, leak_rule_for

    rule = LEAK_RULES[0]
    assert leak_rule_for(str(rule.spot), rule.action_taken, rule.best_action) is rule


# --- риверная точка в `detail` (задача «постфлоп перестаёт молчать») ------------------


def test_a_river_point_without_the_key_has_no_river_numbers():
    """Ключа нет — ответ `None`, и изложение печатает по этой точке ничего."""
    from harness.contracts import SpotKind, river_call_detail

    point = _leak_point(
        spot=SpotKind.POSTFLOP, action_taken="call", best_action="", ev_diff_bb=0.0
    )
    assert river_call_detail(point) is None


def test_a_foreign_shape_under_the_river_key_is_not_swallowed():
    """Ключ с чужим содержимым — поломка, а не повод тихо промолчать.

    Тихий `None` вернул бы игроку то самое молчание, ради которого разбор
    ривера и заведён, и сделал бы это незаметно.
    """
    import pytest
    from pydantic import ValidationError

    from harness.contracts import RIVER_CALL_DETAIL, SpotKind, river_call_detail

    point = _leak_point(
        spot=SpotKind.POSTFLOP,
        action_taken="call",
        best_action="",
        ev_diff_bb=0.0,
        detail={RIVER_CALL_DETAIL: {"pot_before": 398_000}},
    )
    with pytest.raises(ValidationError):
        river_call_detail(point)


# --- точка тёрна и флопа в `detail` --------------------------------------------------


def test_a_turn_flop_point_without_the_key_has_no_numbers():
    """Ключа нет — ответ `None`, и изложение печатает по этой точке ничего."""
    from harness.contracts import SpotKind, turn_flop_call_detail

    point = _leak_point(
        spot=SpotKind.POSTFLOP, action_taken="call", best_action="", ev_diff_bb=0.0
    )
    assert turn_flop_call_detail(point) is None


def test_a_foreign_shape_under_the_turn_flop_key_is_not_swallowed():
    """Ключ с чужим содержимым — поломка, а не повод тихо промолчать."""
    import pytest
    from pydantic import ValidationError

    from harness.contracts import TURN_FLOP_CALL_DETAIL, SpotKind, turn_flop_call_detail

    point = _leak_point(
        spot=SpotKind.POSTFLOP,
        action_taken="call",
        best_action="",
        ev_diff_bb=0.0,
        detail={TURN_FLOP_CALL_DETAIL: {"pot_before": 160_000}},
    )
    with pytest.raises(ValidationError):
        turn_flop_call_detail(point)


def test_the_share_of_bluffs_and_their_count_are_filled_together():
    """Доля без числа (или число без доли) — половина ответа, выданная за целый."""
    import pytest
    from pydantic import ValidationError

    from harness.contracts import TurnFlopCallDetail

    whole = {
        "pot_before": 160_000,
        "to_call": 69_000,
        "required_equity": 0.3013,
        "min_value_combos": 55,
        "bluffs_needed_min_value": 18,
        "bluff_share": 18 / 73,
    }
    assert TurnFlopCallDetail.model_validate(whole).bluffs_needed_min_value == 18
    assert TurnFlopCallDetail.model_validate(
        {**whole, "bluffs_needed_min_value": None, "bluff_share": None}
    ).bluff_share is None
    for half in ({"bluffs_needed_min_value": None}, {"bluff_share": None}):
        with pytest.raises(ValidationError):
            TurnFlopCallDetail.model_validate({**whole, **half})


# --- вывод модели: конверт вокруг полезной нагрузки --------------------------
#
# Измеренный класс сбоя, а не гипотеза: Sonnet кладёт заполненную схему внутрь
# одного контейнерного ключа вместо раскладки по корню. Имя ключа плавает
# (`params`, `$PARAMETER_NAME`), поэтому сравнивать с конкретной строкой нельзя.
# Схемы вывода состоят из необязательных полей — без разворота такой ответ
# валиден, пуст и от честного «ничего не вижу» неотличим ничем.


def _full_reading_payload() -> dict:
    return {
        "hand_no": "TM1",
        "board": ["6s", "4h", "Jc"],
        "players": [{"nickname": "N1", "seat": 1}],
    }


@pytest.mark.parametrize("envelope", ["params", "$PARAMETER_NAME", "properties"])
def test_an_output_wrapped_in_one_container_key_is_unwrapped(envelope: str):
    from harness.contracts import VisionReading

    reading = VisionReading.model_validate({envelope: _full_reading_payload()})

    assert reading.hand_no == "TM1"
    assert len(reading.players) == 1
    assert reading.board == ["6s", "4h", "Jc"]


def test_an_unknown_key_is_an_error_and_not_a_silently_empty_output():
    """Тихая потеря становится громкой: неизвестное поле — отказ валидации.

    Без этого структурно неверный ответ неотличим от пустого чтения, а повтор
    на той же модели даёт тот же результат и ту же цену (`llm_calls`: два
    вызова по 1400 выходных токенов, оба выброшены).
    """
    from harness.contracts import VisionReading

    with pytest.raises(ValidationError):
        VisionReading.model_validate({**_full_reading_payload(), "лишнее": 1})


@pytest.mark.parametrize(
    ("schema_path", "payload"),
    [
        ("harness.contracts:VisionReading", {"hand_no": "TM1"}),
        ("harness.contracts:TournamentTextOut", {"paragraphs": ["а", "б"]}),
        ("harness.explanation.verdict_text:VerdictDraft", {"points": [], "summary": "с"}),
        ("harness.explanation.question:QuestionDraft", {"answer": "о"}),
    ],
)
def test_every_schema_the_model_fills_survives_the_container_key(schema_path: str, payload: dict):
    """Разворот конверта — на всех четырёх местах LLM, а не только на зрении.

    Конверт — свойство транспорта, а не одной схемы: он приходит от того, КАК
    модель заполняет вызов инструмента. У трёх схем поля обязательные, и конверт
    там не теряется молча, а падает валидацией — но падает он оплаченным
    вызовом, и схема-ретрай фасада платит второй раз за то же самое.
    """
    import importlib

    module_name, class_name = schema_path.split(":")
    schema = getattr(importlib.import_module(module_name), class_name)

    assert schema.model_validate({"params": payload}) == schema.model_validate(payload)
