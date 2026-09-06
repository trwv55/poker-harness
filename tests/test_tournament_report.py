"""Отчёт по турниру: факты, траектория, находки, честный счёт EV (задача 23).

Синтетика собирается как `RawHand` и проходит настоящий конвейер
(`normalize` → `enrich`) — исход раздачи считает движок по картам, а не тест:
подставлять «герой потерял столько-то» руками значило бы проверять отчёт на
входе, которого конвейер никогда не произведёт.

Сводка скана (`ScanSummary`) приходит отчёту аргументом и здесь собирается
прямо: это ВХОД, и гонять ради него настоящий скан (минуты на реальном файле)
незачем — сам скан проверен в `test_scan.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from harness.analysis.tournament import tournament_report
from harness.contracts import (
    ActionKind,
    EnrichedHand,
    Post,
    PostKind,
    Provenance,
    RawAction,
    RawHand,
    ScanItem,
    ScanSummary,
    SeatInfo,
    ShowdownEntry,
    SpotKind,
    Street,
    ValidationStatus,
    Zone,
)
from harness.engine import enrich
from harness.normalizer import normalize

_START = datetime(2026, 8, 20, 20, 0, 0, tzinfo=UTC)
_BOARD = {Street.FLOP: ["Kd", "8h", "3s"], Street.TURN: ["4c"], Street.RIVER: ["9d"]}
_HERO_WINS = (("Ac", "Ad"), ("7c", "2s"))
_HERO_LOSES = (("7c", "2s"), ("Ac", "Ad"))


# --- Синтетика: хедз-ап, герой на малом блайнде ------------------------------------


def _raw(
    *,
    hand_no: str,
    level: int,
    bb: int,
    stack: int,
    minute: int,
    actions: list[RawAction],
    dealt: dict[str, list[str]],
    boards: dict[Street, list[str]] | None = None,
    showdowns: list[ShowdownEntry] | None = None,
    villain_stack: int | None = None,
) -> RawHand:
    return RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="synthetic",
        hand_no=hand_no,
        tournament_id="T1",
        tournament_name="synthetic",
        level=level,
        sb=bb // 2,
        bb=bb,
        ante=0,
        timestamp=_START + timedelta(minutes=minute),
        table_name="syn",
        max_seats=2,
        button_seat=1,
        seats=[
            SeatInfo(seat=1, label="Hero", stack=stack),
            SeatInfo(seat=2, label="V", stack=villain_stack if villain_stack else stack),
        ],
        posts=[
            Post(label="Hero", kind=PostKind.SMALL_BLIND, amount=bb // 2),
            Post(label="V", kind=PostKind.BIG_BLIND, amount=bb),
        ],
        dealt=dealt,
        actions=actions,
        boards=boards or {},
        showdowns=showdowns or [],
    )


def _enriched(raw: RawHand, index: int) -> EnrichedHand:
    en = enrich(normalize(raw).model_copy(update={"hand_index": index}))
    assert en.verdict.status is not ValidationStatus.REJECT, en.verdict.reasons
    return en


def _fold_hand(*, hand_no: str, level: int, bb: int, stack: int, minute: int, index: int):
    """Герой пасует с малого блайнда — теряет ровно малый блайнд."""
    raw = _raw(
        hand_no=hand_no,
        level=level,
        bb=bb,
        stack=stack,
        minute=minute,
        actions=[
            RawAction(
                street=Street.PREFLOP,
                label="Hero",
                kind=ActionKind.FOLD,
                raw_line="Hero: folds",
            )
        ],
        dealt={"Hero": ["3c", "2d"]},
    )
    return _enriched(raw, index)


def _all_in_hand(
    *,
    hand_no: str,
    level: int,
    bb: int,
    stack: int,
    minute: int,
    index: int,
    hero_wins: bool,
    villain_stack: int | None = None,
):
    """Герой шовит с малого блайнда, оппонент уравнивает; исход решают карты.

    Кто выиграл, определяет движок по вскрытым картам и доске, а не тест: так
    `stacks_end` — результат реплея, а не подставленное число.
    """
    hero_cards, villain_cards = _HERO_WINS if hero_wins else _HERO_LOSES
    raw = _raw(
        hand_no=hand_no,
        level=level,
        bb=bb,
        stack=stack,
        minute=minute,
        villain_stack=villain_stack,
        actions=[
            RawAction(
                street=Street.PREFLOP,
                label="Hero",
                kind=ActionKind.RAISE,
                amount=stack - bb // 2,
                to_amount=stack,
                is_all_in=True,
                raw_line=f"Hero: raises to {stack} and is all-in",
            ),
            RawAction(
                street=Street.PREFLOP,
                label="V",
                kind=ActionKind.CALL,
                amount=min(stack, villain_stack or stack) - bb,
                is_all_in=True,
                raw_line="V: calls and is all-in",
            ),
        ],
        dealt={"Hero": list(hero_cards), "V": list(villain_cards)},
        boards=_BOARD,
        showdowns=[
            ShowdownEntry(label="Hero", cards=list(hero_cards)),
            ShowdownEntry(label="V", cards=list(villain_cards)),
        ],
    )
    return _enriched(raw, index)


def _summary(
    hands_total: int,
    *,
    items: list[ScanItem] | None = None,
    total_loss_bb: float = 0.0,
    points_total: int = 0,
    points_judged: int = 0,
) -> ScanSummary:
    return ScanSummary(
        hands_total=hands_total,
        hands_with_decision=hands_total,
        items=items or [],
        total_loss_bb=total_loss_bb,
        points_total=points_total,
        points_judged=points_judged,
    )


def _item(
    hand_no: str,
    *,
    ev_diff_bb: float = -1.0,
    action_taken: str = "call",
    best_action: str = "fold",
    spot: SpotKind = SpotKind.PUSHFOLD_FACING_SHOVE,
    zone: Zone = Zone.STRICT,
) -> ScanItem:
    return ScanItem(
        hand_no=hand_no,
        hand_index=None,
        hero_class="K3s",
        spot=spot,
        action_taken=action_taken,
        best_action=best_action,
        ev_diff_bb=ev_diff_bb,
        zone=zone,
    )


# --- Что случилось за турнир -------------------------------------------------------


def test_hands_levels_and_duration_come_from_the_hands_themselves():
    hands = [
        _fold_hand(hand_no="H1", level=5, bb=100, stack=5000, minute=0, index=0),
        _fold_hand(hand_no="H2", level=5, bb=100, stack=4950, minute=7, index=1),
        _fold_hand(hand_no="H3", level=6, bb=200, stack=4900, minute=20, index=2),
    ]
    report = tournament_report(hands, _summary(3))
    assert report.hands_total == 3
    assert (report.first_level, report.last_level, report.levels_played) == (5, 6, 2)
    assert report.duration_minutes == 20


def test_levels_carry_their_own_hands_and_stack_edges():
    """Стек уровня — в bb ЭТОГО уровня: те же фишки на следующем уровне мельче."""
    hands = [
        _fold_hand(hand_no="H1", level=5, bb=100, stack=5000, minute=0, index=0),
        _fold_hand(hand_no="H2", level=5, bb=100, stack=4950, minute=1, index=1),
        _fold_hand(hand_no="H3", level=6, bb=200, stack=4900, minute=2, index=2),
    ]
    report = tournament_report(hands, _summary(3))
    first, second = report.trajectory.levels
    assert (first.level, first.hands) == (5, 2)
    assert (first.start_bb, first.end_bb) == (50.0, 49.0)
    assert (second.level, second.hands) == (6, 1)
    assert (second.start_bb, second.end_bb) == (24.5, 24.0)


def test_the_turning_point_is_the_hand_with_the_biggest_stack():
    """Перелом — раздача с максимальным стеком на входе; дальше идёт спуск."""
    hands = [
        _fold_hand(hand_no="H1", level=5, bb=100, stack=3000, minute=0, index=0),
        _all_in_hand(
            hand_no="H2", level=5, bb=100, stack=2950, minute=1, index=1, hero_wins=True
        ),
        _fold_hand(hand_no="H3", level=6, bb=200, stack=8000, minute=2, index=2),
        _fold_hand(hand_no="H4", level=6, bb=200, stack=7900, minute=3, index=3),
    ]
    report = tournament_report(hands, _summary(4))
    assert report.trajectory.peak_hand_no == "H3"
    assert report.trajectory.peak_level == 6
    assert report.trajectory.peak_bb == 40.0
    assert report.trajectory.hands_after_peak == 1


def test_the_turning_point_is_measured_in_blinds_not_in_chips():
    """3000 фишек на уровне 5 и 6000 на уровне 6 — одна глубина, 30 bb.

    Блайнды растут, и максимум по фишкам показал бы переломом раздачу, где
    глубина не менялась вовсе. При равной глубине перелом — самая ранняя
    раздача: спуск начинается с первого достижения максимума.
    """
    hands = [
        _fold_hand(hand_no="H1", level=5, bb=100, stack=3000, minute=0, index=0),
        _fold_hand(hand_no="H2", level=6, bb=200, stack=6000, minute=1, index=1),
    ]
    report = tournament_report(hands, _summary(2))
    assert report.trajectory.peak_hand_no == "H1"
    assert report.trajectory.peak_bb == 30.0


def test_an_all_in_is_listed_with_its_outcome():
    """Олл-ины — с исходом в bb; знак `delta_bb` и есть исход."""
    hands = [
        _all_in_hand(
            hand_no="WIN", level=5, bb=100, stack=1000, minute=0, index=0, hero_wins=True
        ),
        _all_in_hand(
            hand_no="LOSS", level=5, bb=100, stack=2000, minute=1, index=1, hero_wins=False
        ),
    ]
    report = tournament_report(hands, _summary(2))
    # Порядок — по стеку, с которым герой вошёл в раздачу: крупнейший первым.
    assert [a.hand_no for a in report.all_ins] == ["LOSS", "WIN"]
    by_no = {a.hand_no: a for a in report.all_ins}
    assert by_no["WIN"].delta_bb == 10.0  # выиграл стек оппонента
    assert by_no["LOSS"].delta_bb == -20.0  # отдал свой
    assert by_no["LOSS"].stack_before_bb == 20.0
    assert all(a.showdown for a in report.all_ins)


def test_a_fold_without_an_all_in_is_not_an_all_in_event():
    hands = [_fold_hand(hand_no="H1", level=5, bb=100, stack=1000, minute=0, index=0)]
    report = tournament_report(hands, _summary(1))
    assert report.all_ins == []


# --- Где ушли фишки ----------------------------------------------------------------


def test_chip_moves_are_only_losses_and_the_dearest_comes_first():
    hands = [
        _fold_hand(hand_no="CHEAP", level=5, bb=100, stack=1000, minute=0, index=0),
        _all_in_hand(
            hand_no="DEAR", level=5, bb=100, stack=1000, minute=1, index=1, hero_wins=False
        ),
        _all_in_hand(
            hand_no="WON", level=5, bb=100, stack=1000, minute=2, index=2, hero_wins=True
        ),
    ]
    report = tournament_report(hands, _summary(3))
    assert [m.hand_no for m in report.chip_moves] == ["DEAR", "CHEAP"]
    assert report.chip_moves[0].cost_bb == 10.0
    assert report.chip_moves[0].all_in is True
    assert report.chip_moves[1].cost_bb == 0.5  # малый блайнд
    assert report.chip_moves[1].last_street is Street.PREFLOP


# --- Честный счёт EV ---------------------------------------------------------------


def test_a_lost_all_in_without_a_gap_is_variance_not_error():
    """Проигранный олл-ин, к которому у расчёта нет претензий, — дисперсия.

    Ни одной фишки такой раздачи не оказывается в судимом столбце: расчёт
    решение не оспорил, и записать его цену в ошибку значило бы судить против
    вскрытой карты.
    """
    hands = [
        _all_in_hand(
            hand_no="COOLER", level=5, bb=100, stack=1000, minute=0, index=0, hero_wins=False
        )
    ]
    report = tournament_report(hands, _summary(1, points_total=1, points_judged=1))
    assert report.ev.chips_in_lost_allins_bb == 10.0
    assert report.ev.chips_in_gap_hands_bb == 0.0
    assert report.ev.judged_loss_bb == 0.0


def test_chips_of_a_hand_with_a_gap_are_not_counted_as_variance():
    """Тот же проигранный олл-ин, но расхождение по нему найдено — столбец другой."""
    hands = [
        _all_in_hand(
            hand_no="GAP", level=5, bb=100, stack=1000, minute=0, index=0, hero_wins=False
        )
    ]
    summary = _summary(
        1, items=[_item("GAP", ev_diff_bb=-2.5)], total_loss_bb=-2.5, points_total=1, points_judged=1
    )
    report = tournament_report(hands, summary)
    assert report.ev.chips_in_gap_hands_bb == 10.0
    assert report.ev.chips_in_lost_allins_bb == 0.0
    # Цена расхождения — своя величина, и с потерей фишек она не совпадает.
    assert report.ev.judged_loss_bb == 2.5


def test_judged_loss_is_the_scan_total_by_magnitude():
    hands = [_fold_hand(hand_no="H1", level=5, bb=100, stack=1000, minute=0, index=0)]
    summary = _summary(1, total_loss_bb=-3.75, points_total=4, points_judged=2)
    report = tournament_report(hands, summary)
    assert report.ev.judged_loss_bb == 3.75
    assert (report.ev.points_judged, report.ev.points_total) == (2, 4)


def test_chip_buckets_are_a_partition_of_every_lost_chip():
    """Три фишечных столбца в сумме дают все потерянные фишки — ни больше, ни меньше."""
    hands = [
        _fold_hand(hand_no="BLIND", level=5, bb=100, stack=1000, minute=0, index=0),
        _all_in_hand(
            hand_no="GAP", level=5, bb=100, stack=1000, minute=1, index=1, hero_wins=False
        ),
        _all_in_hand(
            hand_no="COOLER", level=5, bb=100, stack=1000, minute=2, index=2, hero_wins=False
        ),
        _all_in_hand(
            hand_no="WON", level=5, bb=100, stack=1000, minute=3, index=3, hero_wins=True
        ),
    ]
    summary = _summary(4, items=[_item("GAP")], total_loss_bb=-1.0)
    report = tournament_report(hands, summary)
    lost = sum(m.cost_bb for m in report.chip_moves)
    assert lost == pytest.approx(
        report.ev.chips_in_gap_hands_bb
        + report.ev.chips_in_lost_allins_bb
        + report.ev.chips_elsewhere_bb
    )
    assert report.ev.chips_elsewhere_bb == 0.5  # блайнд из сброшенной раздачи


# --- Находки -----------------------------------------------------------------------


def test_a_single_point_is_not_a_repeating_pattern():
    hands = [_fold_hand(hand_no="H1", level=5, bb=100, stack=1000, minute=0, index=0)]
    report = tournament_report(hands, _summary(1, items=[_item("H1")], total_loss_bb=-1.0))
    assert report.findings == []


def test_a_repeated_pattern_becomes_a_finding_with_its_price():
    hands = [
        _fold_hand(hand_no=f"H{i}", level=5, bb=100, stack=1000, minute=i, index=i)
        for i in range(3)
    ]
    items = [_item("H0", ev_diff_bb=-1.0), _item("H1", ev_diff_bb=-2.0)]
    report = tournament_report(hands, _summary(3, items=items, total_loss_bb=-3.0))
    (finding,) = report.findings
    assert (finding.count, finding.total_cost_bb) == (2, 3.0)
    assert finding.hand_nos == ["H0", "H1"]
    assert finding.action_taken == "call" and finding.best_action == "fold"
    assert finding.seen_before == 0


def test_a_pattern_from_a_past_tournament_is_a_finding_even_once():
    """Одна точка, но та же развилка была в прошлом турнире — это находка.

    Ровно та ссылка на его собственную историю, ради которой прошлые сводки и
    приходят аргументом.
    """
    hands = [_fold_hand(hand_no="H1", level=5, bb=100, stack=1000, minute=0, index=0)]
    past = _summary(9, items=[_item("OLD1"), _item("OLD2")], total_loss_bb=-2.0)
    report = tournament_report(
        hands,
        _summary(1, items=[_item("H1")], total_loss_bb=-1.0),
        past_summaries=[past],
    )
    (finding,) = report.findings
    assert finding.count == 1
    assert (finding.seen_before, finding.seen_before_tournaments) == (2, 1)


def test_a_finding_inherits_the_weakest_zone_of_its_points():
    """Хоть одна точка «предполагая» — вся находка «предполагая»."""
    hands = [
        _fold_hand(hand_no=f"H{i}", level=5, bb=100, stack=1000, minute=i, index=i)
        for i in range(2)
    ]
    items = [_item("H0", zone=Zone.STRICT), _item("H1", zone=Zone.ASSUMING)]
    report = tournament_report(hands, _summary(2, items=items, total_loss_bb=-2.0))
    assert report.findings[0].zone is Zone.ASSUMING


# --- Среднее по всем турнирам игрока -----------------------------------------------


def test_baseline_is_absent_when_the_player_has_a_single_tournament():
    """Один турнир — среднее совпало бы с ним самим, сравнивать не с чем."""
    hands = [_fold_hand(hand_no="H1", level=5, bb=100, stack=1000, minute=0, index=0)]
    report = tournament_report(
        hands, _summary(1), player_tournaments=[[h.hand for h in hands]]
    )
    assert report.baseline is None
    assert report.baseline_tournaments == 1


def test_baseline_covers_every_tournament_of_the_player():
    hands = [
        _all_in_hand(
            hand_no="H1", level=5, bb=100, stack=1000, minute=0, index=0, hero_wins=True
        )
    ]
    other = [_fold_hand(hand_no="P1", level=1, bb=10, stack=1000, minute=0, index=0).hand]
    report = tournament_report(
        hands,
        _summary(1),
        player_tournaments=[[h.hand for h in hands], other],
    )
    assert report.baseline_tournaments == 2
    assert report.baseline is not None
    assert report.baseline.hands == 2  # обе руки, а не только этого турнира
    assert report.stats.hands == 1
    assert report.stats.vpip_pct == 100.0
    assert report.baseline.vpip_pct == 50.0


# --- Отказ вместо тихого расхождения -----------------------------------------------


def test_report_refuses_a_summary_of_another_tournament():
    """Сводка не по этим рукам — отказ, а не отчёт, собранный из двух источников."""
    hands = [_fold_hand(hand_no="H1", level=5, bb=100, stack=1000, minute=0, index=0)]
    with pytest.raises(ValueError, match="раздач"):
        tournament_report(hands, _summary(7))


def test_report_refuses_an_empty_tournament():
    with pytest.raises(ValueError, match="раздач"):
        tournament_report([], _summary(0))
