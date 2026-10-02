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

from harness.analysis.charts import (
    DEFAULT_CHART_PATH,
    ChartFileError,
    ChartKey,
    ChartMissing,
    ChartPlaceholder,
    DepthNotCharted,
    NotationError,
    chart_keys,
    load_chart_book,
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
    30, 40, 50 и 60bb, все позиции RFI на каждой глубине. Загрузка уже прогнала
    все проверки формы, включая сверку `open_pct`, — здесь закрепляется, что
    отдаётся каждая запись.
    """
    book = load_chart_book()
    keys = book.all_keys()
    rfi_positions = {"UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB"}
    for depth in (15.0, 20.0, 30.0, 40.0, 50.0, 60.0):
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


@pytest.mark.parametrize("eff_bb", [14.99, 5.0, 0.0, -1.0, float("nan")])
def test_a_stack_below_the_charts_is_refused_not_clamped(tmp_path: Path, eff_bb: float):
    # Ниже 15bb эталон считается равновесием; вернуть самый мелкий чарт значило бы
    # выдать чарт там, где справочник не применим.
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
            [_entry(**{"raise": "AJo:0.5"}, allin="AJo:0.3", limp="AJo:0.21")],
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


def test_not_json_at_all_is_rejected(tmp_path: Path):
    path = tmp_path / "charts.json"
    path.write_text("{это не json", encoding="utf-8")
    with pytest.raises(ChartFileError):
        load_chart_book(path)
