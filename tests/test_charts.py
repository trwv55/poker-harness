"""Тесты каркаса справочника опен-чартов: запись диапазона, лукап, отказы.

Тут проверяется ФОРМА, а не покер: ни один тест не утверждает, что какой-то
диапазон верен для позиции — чарты владельца, и код их не судит. Проверяется
ровно то, что механика не подменяет эталон: точный ключ или отказ, образец
формата не отдаётся, испорченный файл не загружается.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.analysis import open_chart
from harness.analysis.charts import (
    DEFAULT_CHART_PATH,
    DEFAULT_DEFEND_PATH,
    ChartFileError,
    ChartKey,
    ChartMissing,
    ChartPlaceholder,
    DefendKey,
    DepthNotCharted,
    NotationError,
    chart_keys,
    defend_strategy,
    load_chart_book,
    load_defend_book,
    open_range,
    open_strategy,
    parse_range,
    to_notation,
)
from harness.contracts import RawHand, all_classes

# --- запись диапазона -----------------------------------------------------------


def test_owner_example_string_expands_to_exactly_these_classes():
    # Строка из спецификации владельца — единственный пример, который он назвал.
    rng = parse_range("66+, ATs+, KQs, AJo+")
    assert set(rng.weights) == {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66",
        "ATs", "AJs", "AQs", "AKs",
        "KQs",
        "AJo", "AQo", "AKo",
    }
    assert set(rng.weights.values()) == {1.0}


def test_plus_on_a_pair_walks_up_to_aces():
    assert set(parse_range("QQ+").weights) == {"QQ", "KK", "AA"}
    assert len(parse_range("22+").weights) == 13


def test_plus_on_a_kicker_stops_below_the_high_card():
    # Кикер растёт до старшей карты и не превращается в пару: AKs есть, AA нет.
    assert set(parse_range("ATs+").weights) == {"ATs", "AJs", "AQs", "AKs"}
    assert set(parse_range("AKs+").weights) == {"AKs"}
    assert set(parse_range("KTo+").weights) == {"KTo", "KJo", "KQo"}


def test_span_is_inclusive_and_endpoint_order_does_not_matter():
    assert set(parse_range("TT-77").weights) == {"77", "88", "99", "TT"}
    assert parse_range("77-TT").weights == parse_range("TT-77").weights
    assert set(parse_range("A5s-A2s").weights) == {"A2s", "A3s", "A4s", "A5s"}


def test_weight_suffix_carries_a_mixed_frequency():
    rng = parse_range("AA, AJo:0.25, KQs:0")
    assert rng.weights == {"AA": 1.0, "AJo": 0.25, "KQs": 0.0}
    assert rng.weight("AJo") == 0.25
    assert rng.weight("22") == 0.0  # не названный класс — ноль, это контракт Range


def test_separators_may_be_commas_or_spaces_and_case_is_free():
    assert parse_range("aks, 66+").weights == parse_range("AKs 66+").weights


def test_notation_round_trip_is_bit_exact():
    rng = parse_range("66+, ATs+, KQs, AJo+, T9s:0.5, 32o:0.3333333333333333")
    assert parse_range(to_notation(rng)).weights == rng.weights


def test_to_notation_does_not_invent_compaction():
    # Разложенное обратно в "66+" не сворачивается — это было бы угадыванием.
    assert to_notation(parse_range("QQ+")) == "AA, KK, QQ"


@pytest.mark.parametrize(
    "text",
    [
        "",                # пустая запись
        "KAs",             # старший ранг не первым
        "AK",              # непара без суффикса
        "AAs",             # суффикс у пары
        "AXs",             # неизвестный ранг
        "AKx",             # неизвестный суффикс
        "AKss",            # не запись руки
        "AJs-KQs",         # отрезок между разными старшими картами
        "A5s-A2o",         # отрезок между разными мастностями
        "TT-A5s",          # концы отрезка разного вида
        "AA:1.5",          # вес вне [0,1]
        "AA:-0.1",         # вес вне [0,1]
        "AA:много",        # вес не число
        "AA:0.5:0.5",      # два двоеточия
        "AA, AA",          # класс назван дважды
        "66+, 77",         # тот же класс через раскрытие
    ],
)
def test_broken_notation_is_refused(text: str):
    with pytest.raises(NotationError):
        parse_range(text)


def test_all_expanded_classes_are_among_the_169():
    known = set(all_classes())
    assert set(parse_range("22+, A2s+, A2o+, K2s+").weights) <= known


# --- файл справочника, который лежит в репозитории -------------------------------


def test_shipped_file_serves_the_owner_charts_and_no_example():
    """Файл в репозитории читается целиком, и каждая его запись — настоящий чарт.

    Образцов формата в нём больше нет: с 2026-10-02 там чарты владельца на 15, 20,
    25, 30, 35, 40, 50 и 60bb, все позиции RFI на каждой глубине. Загрузка уже прогнала
    все проверки формы, включая сверку `open_pct`, — здесь закрепляется, что
    отдаётся каждая запись.
    """
    book = load_chart_book()
    keys = book.all_keys()
    rfi_positions = {"UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB"}
    for depth in (15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0):
        assert {k.position for k in keys if k.depth_bb == depth} == rfi_positions
    for key in keys:
        assert book.get(key).opening.weights, key


def test_shipped_examples_use_the_ante_type_vocabulary_of_the_pipeline():
    # Ключ сверяется точным сравнением, поэтому словарь ante_type в чарте обязан
    # совпадать с тем, что кладёт в руку конвейер, иначе лукап промахнётся всегда.
    pipeline_default = RawHand.model_fields["ante_type"].default
    assert {key.ante_type for key in chart_keys()} == {pipeline_default}


def test_shipped_file_documents_itself():
    payload = json.loads(DEFAULT_CHART_PATH.read_text(encoding="utf-8"))
    readme = "\n".join(payload["readme"])
    assert "source" in readme and "revised_at" in readme
    for entry in payload["entries"]:
        assert entry["source"] and entry["revised_at"]


# --- выбор глубины ----------------------------------------------------------------


def _depths_file(tmp_path: Path) -> Path:
    return _write(
        tmp_path,
        [_entry(depth_bb=d, **{"raise": r}) for d, r in ((15, "AA"), (20, "KK"), (30, "QQ"))],
    )


@pytest.mark.parametrize(
    ("eff_bb", "depth"),
    [
        (15.0, 15.0),
        (17.4, 15.0),
        (17.5, 15.0),  # ровно посередине — меньшая глубина
        (17.6, 20.0),
        (24.9, 20.0),
        (25.0, 20.0),
        (25.1, 30.0),
        (300.0, 30.0),  # глубже самого глубокого чарта — самый глубокий
    ],
)
def test_the_nearest_charted_depth_is_taken(tmp_path: Path, eff_bb: float, depth: float):
    book = load_chart_book(_depths_file(tmp_path))
    assert book.nearest(8, "CO", eff_bb, "per_player").depth_bb == depth


def test_the_stack_is_judged_by_the_chart_of_that_depth(tmp_path: Path):
    path = _depths_file(tmp_path)
    assert set(open_range(8, "CO", 19.0, "per_player", path=path).weights) == {"KK"}


def test_a_stack_between_13_and_15bb_is_judged_by_the_shallowest_chart(tmp_path: Path):
    # Решение владельца 2026-10-03: открытие на 13–15bb судит чарт 15bb.
    book = load_chart_book(_depths_file(tmp_path))
    assert book.nearest(8, "CO", 13.0, "per_player").depth_bb == 15.0


@pytest.mark.parametrize("eff_bb", [12.99, 5.0, 0.0, -1.0, float("nan")])
def test_a_stack_below_the_charts_is_refused_not_clamped(tmp_path: Path, eff_bb: float):
    # Ниже 13bb эталон считается равновесием пуш-фолда; вернуть самый мелкий чарт
    # значило бы выдать чарт там, где справочник не применим.
    with pytest.raises(DepthNotCharted):
        load_chart_book(_depths_file(tmp_path)).nearest(8, "CO", eff_bb, "per_player")


# --- лукап по файлу ---------------------------------------------------------------


def _write(tmp_path: Path, entries: list[dict[str, object]], **file_fields: object) -> Path:
    payload: dict[str, object] = {"schema_version": 3, "entries": entries}
    payload.update(file_fields)
    path = tmp_path / "charts.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _entry(**over: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "seats": 8,
        "position": "CO",
        "depth_bb": 20,
        "ante_type": "per_player",
        "source": "тестовый источник",
        "revised_at": "2026-09-06",
        "status": "chart",
        "raise": "66+, ATs+, KQs, AJo+",
    }
    entry.update(over)
    return entry


def test_exact_key_returns_the_range_of_that_entry(tmp_path: Path):
    path = _write(
        tmp_path,
        [_entry(), _entry(depth_bb=30, **{"raise": "AA"})],
    )
    assert open_range(8, "CO", 20.0, "per_player", path=path).weights == parse_range(
        "66+, ATs+, KQs, AJo+"
    ).weights
    assert set(open_range(8, "CO", 30.0, "per_player", path=path).weights) == {"AA"}


def test_a_hand_split_between_actions_keeps_each_share_and_folds_the_rest(tmp_path: Path):
    """AJo наполовину рейзом, на треть олл-ином: открытие 5/6, фолд 1/6."""
    path = _write(
        tmp_path,
        [
            _entry(
                **{"raise": "AA, AJo:0.5"},
                allin="KK, AJo:0.33",
                limp="QQ:0.25",
            )
        ],
    )
    strategy = open_strategy(8, "CO", 20.0, "per_player", path=path)
    assert strategy.raise_range.weight("AJo") == 0.5
    assert strategy.allin_range.weight("AJo") == 0.33
    assert strategy.opening.weights == {"AA": 1.0, "KK": 1.0, "AJo": 0.83}
    assert strategy.fold_weight("AJo") == pytest.approx(0.17)
    assert strategy.fold_weight("QQ") == pytest.approx(0.75)
    assert strategy.fold_weight("72o") == 1.0


def test_a_declared_open_share_that_matches_the_ranges_is_accepted(tmp_path: Path):
    # 66+ (54) + ATs+ (16) + KQs (4) + AJo+ (36) = 110 комбо = 8.30%.
    path = _write(tmp_path, [_entry(open_pct="8.3%")])
    assert open_range(8, "CO", 20.0, "per_player", path=path).weights


def test_entry_returns_the_provenance_and_refuses_the_same_way(tmp_path: Path):
    # Провенанс нужен трейсу подключения: вердикт обязан называть, откуда чарт.
    path = _write(
        tmp_path,
        [_entry(source="солвер X, настройки Y"), _entry(position="BTN", status="example")],
    )
    entry = load_chart_book(path).entry(ChartKey(8, "CO", 20.0, "per_player"))
    assert entry.source == "солвер X, настройки Y"
    assert entry.revised_at.isoformat() == "2026-09-06"
    with pytest.raises(ChartPlaceholder):
        load_chart_book(path).entry(ChartKey(8, "BTN", 20.0, "per_player"))
    with pytest.raises(ChartMissing):
        load_chart_book(path).entry(ChartKey(8, "HJ", 20.0, "per_player"))


def test_missing_key_raises_and_no_neighbour_is_substituted(tmp_path: Path):
    # Глубина берётся ближайшая, но соседняя позиция, раскладка и тип анте — никогда.
    path = _write(tmp_path, [_entry(), _entry(position="BTN")])
    with pytest.raises(ChartMissing):
        open_range(8, "HJ", 20.0, "per_player", path=path)
    with pytest.raises(ChartMissing):
        open_range(9, "CO", 20.0, "per_player", path=path)
    with pytest.raises(ChartMissing):
        open_range(8, "CO", 20.0, "bb_ante", path=path)
    with pytest.raises(ChartMissing) as missing:
        load_chart_book(path).get(ChartKey(8, "CO", 40.0, "per_player"))
    assert "20.0" in str(missing.value)


def test_placeholder_entry_is_refused_even_on_an_exact_key(tmp_path: Path):
    path = _write(tmp_path, [_entry(status="example")])
    assert ChartKey(8, "CO", 20.0, "per_player") in load_chart_book(path).all_keys()
    with pytest.raises(ChartPlaceholder):
        open_range(8, "CO", 20.0, "per_player", path=path)


def test_rewritten_file_is_reread_not_served_from_cache(tmp_path: Path):
    path = _write(tmp_path, [_entry(**{"raise": "AA"})])
    assert set(open_range(8, "CO", 20.0, "per_player", path=path).weights) == {"AA"}
    _write(tmp_path, [_entry(**{"raise": "KK"})])
    assert set(open_range(8, "CO", 20.0, "per_player", path=path).weights) == {"KK"}


def test_absent_file_is_a_clear_error(tmp_path: Path):
    with pytest.raises(ChartFileError):
        load_chart_book(tmp_path / "нет-такого.json")


@pytest.mark.parametrize(
    ("entries", "file_fields"),
    [
        pytest.param([_entry(seats=3, position="CO")], {}, id="позиции нет за таким столом"),
        pytest.param([_entry(seats=12)], {}, id="нет раскладки для такого стола"),
        pytest.param([_entry(position="MP")], {}, id="неизвестная позиция"),
        pytest.param([_entry(depth_bb=10)], {}, id="глубина чарта ниже 15bb"),
        pytest.param([_entry(depth_bucket="20-30")], {}, id="поле схемы 2"),
        pytest.param([_entry(**{"raise": "Zs"})], {}, id="битая запись диапазона"),
        pytest.param([_entry(allin="AA:1.4")], {}, id="вес вне [0,1]"),
        pytest.param([_entry(**{"raise": ""})], {}, id="открытия нет: пусты raise и allin"),
        pytest.param(
            [_entry(**{"raise": "AA, AJo:0.6"}, allin="AJo:0.5")], {}, id="доли руки больше 1"
        ),
        pytest.param(
            [_entry(**{"raise": "AJo:0.5"}, allin="AJo:0.3", limp="AJo:0.22")],
            {},
            id="доли руки больше 1 вместе с лимпом",
        ),
        pytest.param([_entry(open_pct="9.5%")], {}, id="open_pct расходится с диапазонами"),
        pytest.param([_entry(open_pct="много")], {}, id="open_pct без процента"),
        pytest.param([_entry(hands="AA")], {}, id="поле схемы 1"),
        pytest.param([_entry(source="")], {}, id="пустой source"),
        pytest.param([_entry(ante_type=" ")], {}, id="пустой ante_type"),
        pytest.param([_entry(revised_at="вчера")], {}, id="дата не ISO"),
        pytest.param([_entry(status="draft")], {}, id="неизвестный status"),
        pytest.param([_entry(comment="лишнее поле")], {}, id="опечатка в имени поля"),
        pytest.param([_entry(), _entry()], {}, id="ключ дважды"),
        pytest.param([], {}, id="ни одной записи"),
        pytest.param([_entry()], {"schema_version": 2}, id="чужая версия схемы"),
        pytest.param([_entry()], {"unexpected": 1}, id="лишнее поле файла"),
    ],
)
def test_malformed_file_is_rejected(
    tmp_path: Path, entries: list[dict[str, object]], file_fields: dict[str, object]
):
    path = _write(tmp_path, entries, **file_fields)
    with pytest.raises(ChartFileError):
        load_chart_book(path)


@pytest.mark.parametrize(
    ("shares", "loads"),
    [(("AJo:0.5", "AJo:0.3", "AJo:0.21"), True), (("AJo:0.5", "AJo:0.3", "AJo:0.22"), False)],
)
def test_shares_rounded_to_hundredths_may_sum_to_one_point_zero_one(
    tmp_path: Path, shares: tuple[str, str, str], loads: bool
):
    """Солвер округляет каждую долю до сотых: сумма руки 1.01 — округление, 1.02 —
    ошибка. Доли не перенормируются: каждое действие держит свою долю как записана."""
    raise_, allin, limp = shares
    path = _write(tmp_path, [_entry(**{"raise": raise_}, allin=allin, limp=limp)])
    if not loads:
        with pytest.raises(ChartFileError):
            load_chart_book(path)
        return
    strategy = open_strategy(8, "CO", 20.0, "per_player", path=path)
    assert (
        strategy.raise_range.weight("AJo"),
        strategy.allin_range.weight("AJo"),
        strategy.limp_range.weight("AJo"),
    ) == (0.5, 0.3, 0.21)


def test_not_json_at_all_is_rejected(tmp_path: Path):
    path = tmp_path / "charts.json"
    path.write_text("{это не json", encoding="utf-8")
    with pytest.raises(ChartFileError):
        load_chart_book(path)


# --- чарты защиты ---------------------------------------------------------------
#
# Формат — владельца (спека 2026-10-14, §4): ключ несёт `vs_position` и `vs_action`
# (`raise` по умолчанию, `limp` — только против SB); против рейза действия `call`,
# `raise` (3-бет), `allin`, против лимпа — `check`, `raise` (изоляция), `allin`.
# Сверки `*_pct` — у каждого действия и `fold_pct`.
# Синтетика: колл 22+ (78) + A2s-AQs (44) + K9o+ (48) = 170 комбо = 12.82%, AKs
# наполовину 3-бетом (2) = 0.15%, фолд — остальные 1154 = 87.03%.


def _defend(**over: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "seats": 8,
        "position": "BB",
        "vs_position": "BTN",
        "depth_bb": 25,
        "ante_type": "per_player",
        "source": "тестовый источник",
        "revised_at": "2026-10-14",
        "call": "22+, A2s-AQs, K9o+",
        "raise": "AKs:0.5",
    }
    entry.update(over)
    return {key: value for key, value in entry.items() if value is not None}


def _limp(**over: object) -> dict[str, object]:
    """Синтетика против лимпа SB: чек 22+ (78) = 5.88%, изоляция AKs (4) = 0.30%;
    рук вне действий сколько угодно — форма их не судит."""
    entry: dict[str, object] = {
        "vs_position": "SB",
        "vs_action": "limp",
        "call": None,
        "check": "22+",
        "raise": "AKs",
    }
    entry.update(over)
    return _defend(**entry)


def _write_defend(tmp_path: Path, entries: list[dict[str, object]], **file_fields: object) -> Path:
    payload: dict[str, object] = {"schema_version": 3, "entries": entries}
    payload.update(file_fields)
    path = tmp_path / "defend.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_a_defence_entry_gives_three_actions_and_folds_the_rest(tmp_path: Path):
    path = _write_defend(
        tmp_path,
        [_defend(call_pct="12.8%", raise_pct="0.2%", allin_pct="0%", fold_pct="87.0%")],
    )
    strategy = defend_strategy(8, "BB", "BTN", 25.0, "per_player", path=path)
    assert strategy.call_range.weight("A5s") == 1.0
    assert strategy.raise_range.weight("AKs") == 0.5
    assert strategy.allin_range.weights == {}
    assert strategy.check_range.weights == {}
    assert strategy.fold_weight("AKs") == pytest.approx(0.5)
    assert strategy.fold_weight("72o") == 1.0


def test_the_defence_takes_the_nearest_depth_like_the_open(tmp_path: Path):
    """Одно правило на опен и защиту: ближайшая глубина, при равенстве — меньшая."""
    path = _write_defend(tmp_path, [_defend(depth_bb=20), _defend(depth_bb=30, call="22+")])
    book = load_defend_book(path)
    assert book.nearest(8, "BB", "BTN", 25.0, "per_player").depth_bb == 20.0
    assert book.nearest(8, "BB", "BTN", 26.0, "per_player").depth_bb == 30.0
    with pytest.raises(DepthNotCharted):
        book.nearest(8, "BB", "BTN", 12.0, "per_player")


def test_a_missing_defence_file_means_no_charts(tmp_path: Path):
    """Без файла лукап отказывает, а не падает."""
    absent = tmp_path / "нет-такого.json"
    assert load_defend_book(absent).all_keys() == []
    with pytest.raises(ChartMissing):
        defend_strategy(8, "BB", "BTN", 25.0, "per_player", path=absent)


def test_no_neighbour_defence_is_substituted(tmp_path: Path):
    path = _write_defend(tmp_path, [_defend(), _defend(vs_position="SB")])
    with pytest.raises(ChartMissing):
        defend_strategy(8, "BB", "CO", 25.0, "per_player", path=path)
    with pytest.raises(ChartMissing):
        defend_strategy(8, "SB", "BTN", 25.0, "per_player", path=path)
    with pytest.raises(ChartMissing):
        defend_strategy(8, "BB", "BTN", 25.0, "bb_ante", path=path)
    with pytest.raises(ChartMissing):
        defend_strategy(8, "BB", "SB", 25.0, "per_player", vs_action="limp", path=path)


def test_a_defence_placeholder_is_refused(tmp_path: Path):
    path = _write_defend(tmp_path, [_defend(status="example")])
    assert DefendKey(8, "BB", "BTN", "raise", 25.0, "per_player") in load_defend_book(path).all_keys()
    with pytest.raises(ChartPlaceholder):
        defend_strategy(8, "BB", "BTN", 25.0, "per_player", path=path)


def test_the_owner_spelling_utg1_is_read_as_utg_plus_one(tmp_path: Path):
    path = _write_defend(tmp_path, [_defend(vs_position="UTG1")])
    assert load_defend_book(path).all_keys() == [
        DefendKey(8, "BB", "UTG+1", "raise", 25.0, "per_player")
    ]
    strategy = defend_strategy(8, "BB", "UTG+1", 25.0, "per_player", path=path)
    assert strategy.call_range.weight("A5s") == 1.0


def test_vs_action_is_raise_unless_named(tmp_path: Path):
    path = _write_defend(tmp_path, [_defend(vs_position="SB")])
    assert load_defend_book(path).all_keys() == [
        DefendKey(8, "BB", "SB", "raise", 25.0, "per_player")
    ]


def test_the_raise_and_the_limp_of_the_sb_are_two_keys(tmp_path: Path):
    path = _write_defend(
        tmp_path,
        [
            _defend(vs_position="SB", vs_action="raise"),
            _limp(check_pct="5.9%", raise_pct="0.3%", allin_pct="0%", raise_size_bb=3),
        ],
    )
    against_raise = defend_strategy(8, "BB", "SB", 25.0, "per_player", path=path)
    against_limp = defend_strategy(8, "BB", "SB", 25.0, "per_player", vs_action="limp", path=path)
    assert against_raise.call_range.weight("A5s") == 1.0
    assert against_raise.check_range.weights == {}
    assert against_limp.check_range.weight("22") == 1.0
    assert against_limp.call_range.weights == {}
    assert against_limp.raise_range.weight("AKs") == 1.0
    assert against_limp.fold_weight("22") == 0.0
    book = load_defend_book(path)
    limp_key = book.nearest(8, "BB", "SB", 25.0, "per_player", vs_action="limp")
    assert book.entry(limp_key).raise_size_bb == 3


@pytest.mark.parametrize(
    "entries",
    [
        pytest.param([_defend(seats=4, vs_position="LJ")], id="открывшего нет за таким столом"),
        pytest.param([_defend(vs_position="BB")], id="защищающийся и открывший — одно место"),
        pytest.param([_defend(call="", **{"raise": ""})], id="защиты нет"),
        pytest.param([_defend(call="AKs", **{"raise": "AKs:0.5"})], id="доли руки больше 1"),
        pytest.param([_defend(call_pct="13.0%")], id="call_pct расходится с диапазоном"),
        pytest.param([_defend(raise_pct="0.3%")], id="raise_pct расходится с диапазоном"),
        pytest.param([_defend(allin_pct="0.2%")], id="allin_pct расходится с диапазоном"),
        pytest.param([_defend(fold_pct="86.9%")], id="fold_pct расходится с диапазонами"),
        pytest.param([_limp(check_pct="6.0%")], id="check_pct расходится с диапазоном"),
        pytest.param([_defend(raise_pct="без процента")], id="сверка без процента"),
        pytest.param([_defend(check="AA")], id="чек против рейза"),
        pytest.param([_defend(check_pct="0%")], id="сверка чека против рейза"),
        pytest.param([_limp(call="AA")], id="колл против лимпа"),
        pytest.param([_limp(call_pct="0%")], id="сверка колла против лимпа"),
        pytest.param([_limp(fold_pct="5%")], id="фолд против лимпа"),
        pytest.param([_limp(vs_position="BTN")], id="лимп не от SB"),
        pytest.param([_defend(vs_action="call")], id="неизвестное vs_action"),
        pytest.param([_defend(defend_pct="13.0%")], id="поле defend_pct из схемы Task 1"),
        pytest.param([_defend(limp="AA")], id="поле опен-чарта в чарте защиты"),
        pytest.param([_defend(depth_bb=10)], id="глубина ниже 15bb"),
        pytest.param([_defend(source=" ")], id="пустой source"),
        pytest.param([_defend(revised_at=None)], id="нет revised_at"),
        pytest.param([_defend(), _defend()], id="ключ дважды"),
        pytest.param([_defend(vs_position="UTG1"), _defend(vs_position="UTG+1")], id="UTG1 и UTG+1 — один ключ"),
        pytest.param([], id="ни одной записи"),
    ],
)
def test_a_malformed_defence_file_is_rejected(tmp_path: Path, entries: list[dict[str, object]]):
    with pytest.raises(ChartFileError):
        load_defend_book(_write_defend(tmp_path, entries))


@pytest.mark.parametrize(("raise_share", "loads"), [("0.51", True), ("0.52", False)])
def test_defence_shares_follow_the_same_rounding_rule(
    tmp_path: Path, raise_share: str, loads: bool
):
    """Тот же допуск, что у опенов: K9o 0.5 чеком и 0.51 изоляцией (1.01) грузится
    как записано, 0.52 (1.02) — отказ."""
    entry = _limp(check="22+, K9o:0.5", **{"raise": f"AKs, K9o:{raise_share}"})
    path = _write_defend(tmp_path, [entry])
    if not loads:
        with pytest.raises(ChartFileError):
            load_defend_book(path)
        return
    strategy = defend_strategy(8, "BB", "SB", 25.0, "per_player", vs_action="limp", path=path)
    assert (strategy.check_range.weight("K9o"), strategy.raise_range.weight("K9o")) == (0.5, 0.51)
    assert strategy.fold_weight("K9o") == 0.0


def test_a_limp_entry_may_declare_zero_fold(tmp_path: Path):
    """Против лимпа фолда нет: `fold_pct` отсутствует или «0%» — и «0%» сверяется
    с диапазонами, как любая сверка. Синтетика: всё, кроме AKs, — чек (1322 комбо)."""
    every_class_but_aks = ", ".join(cls for cls in all_classes() if cls != "AKs")
    full = _limp(check=every_class_but_aks, check_pct="99.7%", raise_pct="0.3%")
    load_defend_book(_write_defend(tmp_path, [{**full, "fold_pct": "0%"}]))
    with pytest.raises(ChartFileError):
        load_defend_book(_write_defend(tmp_path, [_limp(fold_pct="0%")]))


# --- чарт защиты владельца в репозитории ----------------------------------------

_OWNER_OPENERS = ("UTG", "UTG+1", "LJ", "HJ", "CO", "BTN")


def test_the_defence_file_in_the_repo_loads_with_every_check():
    """Восемь записей владельца: BB против шести опенов рейзом и против SB дважды;
    загрузка — значит, все сверки `*_pct` и сумма долей прошли."""
    book = load_defend_book(DEFAULT_DEFEND_PATH)
    keys = {(k.vs_position, k.vs_action) for k in book.all_keys()}
    assert keys == {*((opener, "raise") for opener in _OWNER_OPENERS), ("SB", "raise"), ("SB", "limp")}
    assert {(k.seats, k.position, k.depth_bb, k.ante_type) for k in book.all_keys()} == {
        (8, "BB", 25.0, "per_player")
    }


def test_the_repo_defence_against_the_button():
    strategy = defend_strategy(8, "BB", "BTN", 25.0, "per_player")
    assert strategy.call_range.weight("A9s") == 1.0
    assert strategy.raise_range.weight("AA") == 1.0
    assert strategy.allin_range.weight("TT") == 1.0
    assert strategy.check_range.weights == {}


def test_the_repo_defence_against_the_sb_differs_by_its_action():
    against_raise = defend_strategy(8, "BB", "SB", 25.0, "per_player")
    against_limp = defend_strategy(8, "BB", "SB", 25.0, "per_player", vs_action="limp")
    assert against_raise.call_range.weight("AA") == 1.0
    assert against_raise.check_range.weights == {}
    assert against_limp.check_range.weight("K8s") == 1.0
    assert against_limp.raise_range.weight("AA") == 1.0
    assert against_limp.call_range.weights == {}


def test_a_seven_handed_defence_reads_the_eight_max_entry_from_the_repo():
    seats, position, vs_position = open_chart.defend_table(7, "BB", "BTN")
    assert defend_strategy(seats, position, vs_position, 25.0, "per_player") == defend_strategy(
        8, "BB", "BTN", 25.0, "per_player"
    )
