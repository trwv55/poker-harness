"""Вердикт по чарту: открытие первым при стеке от 13bb (`analysis.open_chart`).

Раздачи — синтетика 8-max, чтобы в справочнике владельца был чарт для стола;
столы на 4–7 мест судятся 8-max чартом по позиции от конца (решение владельца
2026-10-10).
Где частоты руки нужны точными, они берутся из чарта в репозитории (CO 35bb:
AJo — рейз 100%, 22 — рейз 42% и фолд 58%); где нужен порог 30% — из
временного файла с подобранными частотами.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from harness.analysis import analyze_hand, open_chart
from harness.analysis.charts import load_chart_book
from harness.analysis.classifier import classify, table_state
from harness.contracts import (
    ActionKind,
    Post,
    PostKind,
    Provenance,
    RawAction,
    RawHand,
    SeatInfo,
    SpotKind,
    Street,
    leak_rule_of_point,
)
from harness.engine import enrich
from harness.normalizer import normalize

_SB, _BB, _ANTE = 50, 100, 12
_EIGHT_MAX = ("SB", "BB", "UTG", "UTG+1", "LJ", "HJ", "CO", "BTN")
_SEVEN_MAX = ("SB", "BB", "UTG", "UTG+1", "HJ", "CO", "BTN")
_SIX_MAX = ("SB", "BB", "UTG", "HJ", "CO", "BTN")
_FOUR_MAX = ("SB", "BB", "UTG", "BTN")
_THREE_MAX = ("SB", "BB", "BTN")
_NINE_MAX = ("SB", "BB", "UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN")


def _act(
    label: str,
    kind: ActionKind,
    *,
    amount: int | None = None,
    to_amount: int | None = None,
    is_all_in: bool = False,
) -> RawAction:
    return RawAction(
        street=Street.PREFLOP,
        label=label,
        kind=kind,
        amount=amount,
        to_amount=to_amount,
        is_all_in=is_all_in,
        raw_line=label,
    )


def _leak_key(point) -> str | None:
    rule = leak_rule_of_point(point)
    return None if rule is None else rule.key


def _hand(
    hero: str,
    cards: tuple[str, str],
    eff_bb: float,
    action: str,
    *,
    opener: str = "",
    table: tuple[str, ...] = _EIGHT_MAX,
):
    """Hero действует первым в неоткрытом банке (или после рейза `opener`).

    `action`: fold, raise (до 2.5bb), shove, limp (только SB). `table` — раскладка
    позиций стола от SB до BTN (`POSITIONS_BY_COUNT`), кнопка на последнем месте.
    """
    stack = round(eff_bb * _BB)
    labels = {pos: ("Hero" if pos == hero else pos) for pos in table}
    seats = [SeatInfo(seat=i + 1, label=labels[p], stack=stack) for i, p in enumerate(table)]
    posts = [Post(label=s.label, kind=PostKind.ANTE, amount=_ANTE) for s in seats] + [
        Post(label=labels["SB"], kind=PostKind.SMALL_BLIND, amount=_SB),
        Post(label=labels["BB"], kind=PostKind.BIG_BLIND, amount=_BB),
    ]
    already = {"SB": _SB, "BB": _BB}.get(hero, 0)
    behind_stack = stack - _ANTE
    actions: list[RawAction] = []
    for pos in (*table[2:], *table[:2]):
        if pos == hero:
            break
        if pos == opener:
            actions.append(_act(pos, ActionKind.RAISE, amount=250, to_amount=250))
        else:
            actions.append(_act(labels[pos], ActionKind.FOLD))
    if action == "fold":
        actions.append(_act("Hero", ActionKind.FOLD))
    elif action == "raise":
        actions.append(_act("Hero", ActionKind.RAISE, amount=250 - already, to_amount=250))
    elif action == "shove":
        actions.append(
            _act(
                "Hero",
                ActionKind.RAISE,
                amount=behind_stack - already,
                to_amount=behind_stack,
                is_all_in=True,
            )
        )
    elif action == "limp":
        actions.append(_act("Hero", ActionKind.CALL, amount=_BB - _SB))
    raw = RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="synthetic",
        hand_no="SYN8",
        tournament_id="T1",
        tournament_name="synthetic",
        level=1,
        sb=_SB,
        bb=_BB,
        ante=_ANTE,
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        table_name="syn",
        max_seats=len(table),
        button_seat=len(table),
        seats=seats,
        posts=posts,
        dealt={"Hero": list(cards)},
        actions=actions,
        boards={},
        showdowns=[],
    )
    return enrich(normalize(raw))


def _point(en):
    return analyze_hand(en).points[0]


# --- классификация --------------------------------------------------------------


@pytest.mark.parametrize(
    ("eff_bb", "action", "spot"),
    [
        (35.0, "fold", SpotKind.OPEN_CHART),
        (35.0, "raise", SpotKind.OPEN_CHART),
        (13.0, "shove", SpotKind.OPEN_CHART),
        (12.9, "shove", SpotKind.PUSHFOLD_UNOPENED),
        (12.9, "raise", SpotKind.PREFLOP_OTHER),
    ],
)
def test_the_open_chart_starts_at_13bb(eff_bb, action, spot):
    en = _hand("CO", ("Ah", "Jd"), eff_bb, action)
    assert classify(en.report.decision_points[0], en) == spot


def test_a_shove_at_exactly_13bb_gets_a_verdict_by_the_15bb_chart():
    """Граница спота и граница справочника — одно число: точка на 13bb судима."""
    point = _point(_hand("CO", ("Ah", "Jd"), 13.0, "shove"))
    assert point.best_action != "" and point.mismatch is not None
    assert point.detail["chart_depth_bb"] == 15.0


def test_an_answer_to_an_open_is_not_an_open():
    en = _hand("CO", ("Ah", "Jd"), 35.0, "fold", opener="UTG")
    assert classify(en.report.decision_points[0], en) == SpotKind.PREFLOP_OTHER


# --- вердикт по настоящему чарту ------------------------------------------------


def test_folding_a_hand_the_chart_always_raises_is_a_mismatch():
    point = _point(_hand("CO", ("Ah", "Jd"), 35.0, "fold"))
    assert point.spot == SpotKind.OPEN_CHART and point.zone == "strict"
    assert point.action_taken == "fold" and point.best_action == "raise"
    assert point.mismatch is True and point.ev_diff_bb == 0.0
    assert point.detail["chart_depth_bb"] == 35.0
    assert point.detail["taken_frequency"] == 0.0
    assert _leak_key(point) == "open_not_opened_raise"


def test_raising_it_is_within_the_chart():
    point = _point(_hand("CO", ("Ah", "Jd"), 35.0, "raise"))
    assert point.mismatch is False and point.detail["taken_frequency"] == 1.0
    assert leak_rule_of_point(point) is None


def test_a_mixed_hand_is_within_the_chart_whichever_frequent_action_is_played():
    """22 у CO на 35bb: рейз 42%, фолд 58% — оба действия не реже порога 30%."""
    for action, frequency in (("fold", 0.58), ("raise", 0.42)):
        point = _point(_hand("CO", ("2h", "2d"), 35.0, action))
        assert point.mismatch is False
        assert point.detail["taken_frequency"] == pytest.approx(frequency)
        assert point.detail["chart_frequencies"]["raise"] == pytest.approx(0.42)


def test_shoving_a_hand_the_chart_raises_is_its_own_leak():
    point = _point(_hand("CO", ("Ah", "Jd"), 35.0, "shove"))
    assert point.action_taken == "shove" and point.mismatch is True
    assert _leak_key(point) == "open_shove_instead_of_raise"


def test_a_small_blind_complete_is_a_limp():
    """K4s у SB на 35bb чарт лимпует всегда."""
    point = _point(_hand("SB", ("Kh", "4h"), 35.0, "limp"))
    assert point.action_taken == "limp" and point.mismatch is False


def test_an_open_between_13_and_15bb_is_judged_by_the_15bb_chart():
    point = _point(_hand("CO", ("Ah", "Jd"), 13.5, "raise"))
    assert point.mismatch is not None and point.detail["chart_depth_bb"] == 15.0


def test_the_nearest_charted_depth_is_named():
    assert _point(_hand("CO", ("Ah", "Jd"), 38.0, "raise")).detail["chart_depth_bb"] == 40.0


def test_the_result_of_the_hand_does_not_enter_the_verdict():
    """Судим против стратегии, а не против вскрытых карт: исход раздачи не читается."""
    en = _hand("CO", ("Ah", "Jd"), 35.0, "fold")
    other = en.model_copy(deep=True)
    other.hand.dealt = {"Hero": ["Ah", "Jd"], "BTN": ["As", "Ad"]}
    assert _point(en).model_dump() == _point(other).model_dump()


# --- порог 30% и «самое частое» на временном чарте -------------------------------


def _chart_file(tmp_path: Path, raise_: str, allin: str = "", limp: str = "") -> Path:
    entry: dict[str, object] = {
        "seats": 8,
        "position": "CO",
        "depth_bb": 35,
        "ante_type": "per_player",
        "source": "тестовый чарт",
        "revised_at": "2026-10-03",
        "raise": raise_,
    }
    if allin:
        entry["allin"] = allin
    if limp:
        entry["limp"] = limp
    path = tmp_path / "charts.json"
    path.write_text(
        json.dumps({"schema_version": 3, "entries": [entry]}, ensure_ascii=False), encoding="utf-8"
    )
    return path


@pytest.mark.parametrize(("weight", "mismatch"), [("0.29", True), ("0.3", False)])
def test_the_threshold_is_thirty_percent(tmp_path, monkeypatch, weight, mismatch):
    path = _chart_file(tmp_path, f"AA, AJo:{weight}")
    monkeypatch.setattr(open_chart, "load_chart_book", lambda: load_chart_book(path))
    point = _point(_hand("CO", ("Ah", "Jd"), 35.0, "raise"))
    assert point.best_action == "fold" and point.mismatch is mismatch


def test_the_most_frequent_action_is_never_a_mismatch(tmp_path, monkeypatch):
    """Рейз 28%, шов 27%, лимп 24%, фолд 21%: рейз реже порога, но он самый частый."""
    path = _chart_file(tmp_path, "AA, AJo:0.28", allin="AJo:0.27", limp="AJo:0.24")
    monkeypatch.setattr(open_chart, "load_chart_book", lambda: load_chart_book(path))
    point = _point(_hand("CO", ("Ah", "Jd"), 35.0, "raise"))
    assert point.best_action == "raise" and point.mismatch is False


def test_a_tie_prefers_the_active_action(tmp_path, monkeypatch):
    path = _chart_file(tmp_path, "AA, AJo:0.5")
    monkeypatch.setattr(open_chart, "load_chart_book", lambda: load_chart_book(path))
    assert _point(_hand("CO", ("Ah", "Jd"), 35.0, "fold")).best_action == "raise"


# --- цена, ранжирование, сводка, изложение --------------------------------------


def _priced(ev: float, index: int = 0):
    from harness.contracts import PointVerdict, Zone

    return PointVerdict(
        dp_index=index,
        street=Street.PREFLOP,
        spot=SpotKind.PUSHFOLD_UNOPENED,
        zone=Zone.STRICT,
        action_taken="fold",
        best_action="shove",
        ev_diff_bb=ev,
        detail={"hero_class": "88"},
    )


def _chart(taken_frequency: float, *, mismatch: bool, index: int = 0):
    from harness.contracts import PointVerdict, Zone

    return PointVerdict(
        dp_index=index,
        street=Street.PREFLOP,
        spot=SpotKind.OPEN_CHART,
        zone=Zone.STRICT,
        action_taken="fold",
        best_action="raise",
        ev_diff_bb=0.0,
        mismatch=mismatch,
        detail={
            "hero_class": "AJo",
            "chart_depth_bb": 35.0,
            "chart_frequencies": {
                "raise": 1.0 - taken_frequency,
                "shove": 0.0,
                "limp": 0.0,
                "fold": taken_frequency,
            },
            "taken_frequency": taken_frequency,
        },
    )


def test_a_chart_point_does_not_enter_the_price_and_ranks_after_the_priced():
    from harness.analysis.error_cost import rank_points, total_ev_loss_bb

    points = [
        _chart(0.6, mismatch=False, index=0),
        _priced(-1.2, index=1),
        _chart(0.2, mismatch=True, index=2),
        _chart(0.0, mismatch=True, index=3),
    ]
    assert total_ev_loss_bb(points) == -1.2
    # Ценовые → расхождения по чарту (реже у солвера выше) → в пределах чарта.
    assert rank_points(points) == [1, 3, 2, 0]


def test_the_scan_lists_chart_mismatches_after_the_priced_rarest_first():
    from harness.analysis import scan

    priced_item = scan._item(_hand("CO", ("Ah", "Jd"), 35.0, "fold"), _priced(-0.5))
    rare = scan._item(_hand("CO", ("Ah", "Jd"), 35.0, "fold"), _chart(0.0, mismatch=True))
    less_rare = scan._item(_hand("CO", ("Ah", "Jd"), 35.0, "fold"), _chart(0.2, mismatch=True))
    assert priced_item.taken_frequency is None and rare.taken_frequency == 0.0
    assert sorted([less_rare, rare, priced_item], key=scan._item_order) == [
        priced_item,
        rare,
        less_rare,
    ]


def test_the_chart_verdict_line_shows_frequencies_and_no_price():
    from harness.presentation.messages import _verdict_lines

    (line,) = _verdict_lines(_chart(0.6, mismatch=False))
    assert "открытие по чарту 35BB" in line and "зона строго" in line
    assert "по чарту: рейз 40.0%, фолд 60.0%" in line and line.endswith("в пределах чарта")
    assert "цена" not in line and "лучше" not in line
    (line,) = _verdict_lines(_chart(0.0, mismatch=True))
    assert line.endswith("расхождение") and "фолд 0" not in line


def test_the_scan_summary_names_how_many_chart_mismatches_did_not_fit():
    from harness.contracts import ScanItem, ScanSummary, Zone
    from harness.presentation import scan_summary_msg

    def item(n: int, freq: float | None) -> ScanItem:
        return ScanItem(
            hand_no=str(n),
            hand_index=n,
            hero_class="AJo",
            spot=SpotKind.OPEN_CHART if freq is not None else SpotKind.PUSHFOLD_UNOPENED,
            action_taken="fold",
            best_action="raise" if freq is not None else "shove",
            ev_diff_bb=0.0 if freq is not None else -1.0,
            zone=Zone.STRICT,
            taken_frequency=freq,
        )

    items = [item(n, None) for n in range(19)] + [item(100 + n, 0.0) for n in range(4)]
    text = scan_summary_msg(
        ScanSummary(hands_total=30, hands_with_decision=30, items=items, total_loss_bb=-19.0), 5, 10
    ).text
    assert "№100 · AJo · открытие по чарту: фолд — у чарта 0.0%, чаще всего рейз" in text
    assert "…и ещё 3 расхождения по чарту." in text


# --- отказы справочника и запасной путь -----------------------------------------


@pytest.mark.parametrize(
    ("contents", "words"),
    [
        ("{это не json", "справочник чартов не прочитался"),
        (
            json.dumps(
                {
                    "schema_version": 3,
                    "entries": [
                        {
                            "seats": 8,
                            "position": "CO",
                            "depth_bb": 35,
                            "ante_type": "per_player",
                            "source": "образец",
                            "revised_at": "2026-10-03",
                            "status": "example",
                            "raise": "AA",
                        }
                    ],
                }
            ),
            "помечен образцом формата",
        ),
    ],
)
def test_a_broken_chart_book_is_named_in_the_words_of_the_player(
    tmp_path, monkeypatch, contents, words
):
    """Ни путь к файлу, ни имя модуля до игрока не доходят — только своя причина."""
    path = tmp_path / "charts.json"
    path.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(open_chart, "load_chart_book", lambda: load_chart_book(path))
    en = _hand("CO", ("Ah", "Jd"), 35.0, "fold")
    dp = en.report.decision_points[0]
    point = open_chart.open_chart_verdict(dp, en, table_state(dp, en))
    reason = point.detail["unjudged"]
    assert words in reason and str(tmp_path) not in reason and "analysis." not in reason
    assert point.action_taken == "fold"


def test_a_table_without_a_chart_keeps_the_push_fold_verdict_up_to_15bb(tmp_path, monkeypatch):
    """Чарта для стола нет — шов на 14bb судит равновесие, как до справочника."""
    book = _other_table_book(tmp_path)
    monkeypatch.setattr(open_chart, "load_chart_book", lambda: load_chart_book(book))
    point = _point(_hand("CO", ("Ah", "Ad"), 14.0, "shove"))
    assert point.spot == SpotKind.PUSHFOLD_UNOPENED and point.best_action == "shove"


def _other_table_book(tmp_path: Path) -> Path:
    path = tmp_path / "charts.json"
    entry = {
        "seats": 9,
        "position": "CO",
        "depth_bb": 35,
        "ante_type": "per_player",
        "source": "чарт другого стола",
        "revised_at": "2026-10-03",
        "raise": "AA",
    }
    path.write_text(json.dumps({"schema_version": 3, "entries": [entry]}), encoding="utf-8")
    return path


def test_every_detail_key_of_a_chart_point_has_a_label():
    from harness.presentation.messages import _DETAIL_LABELS

    point = _point(_hand("CO", ("Ah", "Jd"), 35.0, "fold"))
    assert set(point.detail) - {"chart_frequencies"} <= set(_DETAIL_LABELS)


def test_a_chart_leak_is_shown_without_a_price():
    from harness.contracts import LEAK_RULES, LeakStat
    from harness.presentation.messages import _leak_line

    rule = next(r for r in LEAK_RULES if r.key == "open_too_wide")
    line = _leak_line(LeakStat(rule=rule, count=3, loss_bb=0.0))
    assert line.endswith("по чарту, без цены") and "bb" not in line and "BB" not in line


# --- стол меньше 8 мест: 8-max чарт по позиции от конца (решение владельца 2026-10-10) ---


@pytest.mark.parametrize(
    ("seats", "position", "chart"),
    [
        (8, "UTG", (8, "UTG")),
        (8, "LJ", (8, "LJ")),
        (7, "BTN", (8, "BTN")),
        (7, "CO", (8, "CO")),
        (7, "HJ", (8, "HJ")),
        (7, "UTG+1", (8, "LJ")),
        (7, "UTG", (8, "UTG+1")),
        (7, "SB", (8, "SB")),
        (6, "UTG", (8, "LJ")),
        (6, "HJ", (8, "HJ")),
        (6, "BTN", (8, "BTN")),
        (4, "UTG", (8, "CO")),
        (4, "BTN", (8, "BTN")),
        (4, "SB", (8, "SB")),
        (3, "BTN", (3, "BTN")),
        (3, "SB", (3, "SB")),
        (2, "BTN", (2, "BTN")),
        (9, "UTG+2", (9, "UTG+2")),
    ],
)
def test_the_chart_table_counts_the_players_behind(seats, position, chart):
    """На 4–7 местах позиция 8-max — та, у которой столько же игроков позади до BB;
    8, 9, 3 места и хедз-ап — как есть."""
    assert open_chart.chart_table(seats, position) == chart


def _same_hand_on_eight_max(hero8: str, cards: tuple[str, str], eff_bb: float, action: str):
    return _point(_hand(hero8, cards, eff_bb, action))


def test_a_seven_handed_button_is_judged_by_the_eight_max_button():
    point = _point(_hand("BTN", ("Kh", "9h"), 35.0, "raise", table=_SEVEN_MAX))
    eight = _same_hand_on_eight_max("BTN", ("Kh", "9h"), 35.0, "raise")
    assert point.spot == SpotKind.OPEN_CHART and point.zone == "strict"
    assert (point.detail["chart_seats"], point.detail["chart_position"]) == (8, "BTN")
    assert point.detail["chart_frequencies"] == eight.detail["chart_frequencies"]
    assert point.mismatch is eight.mismatch is False


def test_a_seven_handed_utg_is_judged_by_the_eight_max_utg1():
    point = _point(_hand("UTG", ("Ah", "Jd"), 35.0, "fold", table=_SEVEN_MAX))
    eight = _same_hand_on_eight_max("UTG+1", ("Ah", "Jd"), 35.0, "fold")
    assert (point.detail["chart_seats"], point.detail["chart_position"]) == (8, "UTG+1")
    assert point.detail["chart_frequencies"] == eight.detail["chart_frequencies"]
    assert point.mismatch is eight.mismatch


def test_a_six_handed_table_is_judged_by_position_from_the_end():
    """6 мест: UTG — пятеро позади, как у LJ за 8-max."""
    point = _point(_hand("UTG", ("Ah", "Jd"), 35.0, "raise", table=_SIX_MAX))
    eight = _same_hand_on_eight_max("LJ", ("Ah", "Jd"), 35.0, "raise")
    assert (point.detail["chart_seats"], point.detail["chart_position"]) == (8, "LJ")
    assert point.detail["chart_frequencies"] == eight.detail["chart_frequencies"]


def test_a_four_handed_table_is_judged_by_position_from_the_end():
    """4 места: UTG — трое позади до BB, как у CO за 8-max."""
    point = _point(_hand("UTG", ("Ah", "Jd"), 35.0, "raise", table=_FOUR_MAX))
    eight = _same_hand_on_eight_max("CO", ("Ah", "Jd"), 35.0, "raise")
    assert point.spot == SpotKind.OPEN_CHART and point.zone == "strict"
    assert (point.detail["chart_seats"], point.detail["chart_position"]) == (8, "CO")
    assert point.detail["chart_frequencies"] == eight.detail["chart_frequencies"]


def test_three_handed_and_heads_up_tables_get_no_chart_verdict():
    """3 места и хедз-ап 8-max чартом не судятся: причина словами игрока."""
    from tests.test_postflop_line import P, _fold, _raise
    from tests.test_postflop_line import _hand as _heads_up_hand

    three = _point(_hand("BTN", ("Ah", "Jd"), 35.0, "raise", table=_THREE_MAX))
    assert three.spot == SpotKind.OPEN_CHART and three.best_action == ""
    assert three.detail["unjudged"] == "чарта открытия для стола на 3 места в справочнике нет"
    assert "chart_seats" not in three.detail

    heads_up = _point(_heads_up_hand([_raise(P, "Hero", 300), _fold(P, "V")], button="Hero"))
    assert heads_up.spot == SpotKind.OPEN_CHART and heads_up.best_action == ""
    assert heads_up.detail["unjudged"] == "чарта открытия для хедз-апа в справочнике нет"


def test_an_eight_max_point_carries_no_mark_of_another_table():
    point = _point(_hand("CO", ("Ah", "Jd"), 35.0, "raise"))
    assert "chart_seats" not in point.detail and "chart_position" not in point.detail


def test_the_verdict_line_names_the_eight_max_chart_of_a_smaller_table():
    from harness.presentation.messages import _detail_lines, _verdict_lines

    point = _point(_hand("UTG", ("Ah", "Jd"), 35.0, "raise", table=_SEVEN_MAX))
    (line,) = _verdict_lines(point)
    assert line.startswith("    вердикт: открытие по чарту 35BB · чарт 8-max UTG+1 · зона строго")
    assert not any("chart_seats" in row or "chart_position" in row for row in _detail_lines(point))
    (line,) = _verdict_lines(_point(_hand("CO", ("Ah", "Jd"), 35.0, "raise")))
    assert "чарт 8-max" not in line


def test_a_missing_chart_names_the_real_reason():
    """9 мест: чартов для такого стола нет — и об анте, которое есть, ни слова."""
    point = _point(_hand("CO", ("Ah", "Jd"), 35.0, "raise", table=_NINE_MAX))
    assert point.detail["unjudged"] == "чарта открытия для стола на 9 мест в справочнике нет"
    en = _hand("CO", ("Ah", "Jd"), 35.0, "raise", table=_SEVEN_MAX)
    en.hand.ante_type = "bb_ante"
    dp = en.report.decision_points[0]
    reason = open_chart.open_chart_verdict(dp, en, table_state(dp, en)).detail["unjudged"]
    assert reason == "чарта открытия для анте «bb_ante» в справочнике нет"


def test_a_table_without_its_own_chart_keeps_the_push_fold_verdict_up_to_15bb():
    """7 мест на 14bb: шов судит равновесие этого стола, рейз — 8-max чарт от конца."""
    shove = _point(_hand("CO", ("Ah", "Ad"), 14.0, "shove", table=_SEVEN_MAX))
    assert shove.spot == SpotKind.PUSHFOLD_UNOPENED and shove.best_action == "shove"
    raised = _point(_hand("CO", ("Ah", "Ad"), 14.0, "raise", table=_SEVEN_MAX))
    assert raised.spot == SpotKind.OPEN_CHART and raised.detail["chart_seats"] == 8
