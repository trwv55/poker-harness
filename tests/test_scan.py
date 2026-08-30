"""Префлоп-скан турнира: сводка расхождений по всем рукам файла (задача 13).

Синтетические руки собираются как `RawHand` и прогоняются через настоящий
конвейер (`normalize` → `enrich`) — тот же принцип, что и в
`test_preflop_analysis.py`: собирать `CanonicalHand`/`EnrichedHand` вручную
значило бы проверять скан на входе, которого конвейер никогда не произведёт.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from harness.analysis.preflop import cheap_fold_verdict
from harness.analysis.scan import ScanItem, ScanSummary, scan_tournament
from harness.contracts import (
    ActionKind,
    Assumption,
    Post,
    PostKind,
    Provenance,
    RawAction,
    RawHand,
    SeatInfo,
    Street,
    Zone,
)
from harness.engine import enrich
from harness.normalizer import normalize
from harness.parsers.hh_parser import parse_file
from tests.conftest import FIXTURE_DAILY, FIXTURE_PKO, requires_fixtures

_BB = 2
_SB = 1
_SIX_MAX_SEATS: tuple[str, ...] = ("SB", "BB", "UTG", "HJ", "CO", "BTN")


# --- Синтетика: минимальный набор строительных блоков (см. test_preflop_analysis.py) ----


def _raw(
    *,
    seats: list[SeatInfo],
    button_seat: int,
    posts: list[Post],
    actions: list[RawAction],
    dealt: dict[str, list[str]],
    ante: int = 0,
) -> RawHand:
    return RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="synthetic",
        hand_no="SYN",
        tournament_id="T1",
        tournament_name="synthetic",
        level=1,
        sb=_SB,
        bb=_BB,
        ante=ante,
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        table_name="syn",
        max_seats=len(seats),
        button_seat=button_seat,
        seats=seats,
        posts=posts,
        dealt=dealt,
        actions=actions,
        boards={},
        showdowns=[],
    )


def _fold(label: str) -> RawAction:
    return RawAction(
        street=Street.PREFLOP, label=label, kind=ActionKind.FOLD, raw_line=f"{label}: folds"
    )


def _shove(label: str, to_amount: int, already: int = 0) -> RawAction:
    return RawAction(
        street=Street.PREFLOP,
        label=label,
        kind=ActionKind.RAISE,
        amount=to_amount - already,
        to_amount=to_amount,
        is_all_in=True,
        raw_line=f"{label}: raises {to_amount - already} to {to_amount} and is all-in",
    )


def _make_hu_fold_hand(hero_cards: tuple[str, str], eff_bb: float):
    """Хедз-ап, Hero на кнопке (= малый блайнд) фолдит в неоткрытый банк.

    Раздача заканчивается сразу — второй игрок не действует вовсе: фолд героя
    первым же действием обрывает раздачу, коллировать нечего.
    """
    stack = round(eff_bb * _BB)
    seats = [SeatInfo(seat=1, label="Hero", stack=stack), SeatInfo(seat=2, label="V", stack=stack)]
    posts = [
        Post(label="Hero", kind=PostKind.SMALL_BLIND, amount=_SB),
        Post(label="V", kind=PostKind.BIG_BLIND, amount=_BB),
    ]
    raw = _raw(
        seats=seats,
        button_seat=1,
        posts=posts,
        actions=[_fold("Hero")],
        dealt={"Hero": list(hero_cards)},
    )
    return enrich(normalize(raw))


def _six_max(stacks: dict[str, int], hero_position: str, ante: int = 0):
    labels = {pos: ("Hero" if pos == hero_position else pos) for pos in _SIX_MAX_SEATS}
    seats = [
        SeatInfo(seat=i + 1, label=labels[pos], stack=stacks[pos])
        for i, pos in enumerate(_SIX_MAX_SEATS)
    ]
    posts = [
        Post(label=labels["SB"], kind=PostKind.SMALL_BLIND, amount=_SB),
        Post(label=labels["BB"], kind=PostKind.BIG_BLIND, amount=_BB),
    ]
    if ante:
        posts = [Post(label=s.label, kind=PostKind.ANTE, amount=ante) for s in seats] + posts
    return labels, seats, posts


def _make_multiway_fold_hand(hero_cards: tuple[str, str], eff_bb: float, players_behind: int):
    """Hero фолдит в неоткрытый банк, позади него `players_behind` живых игроков.

    Зеркало `_make_multiway_shove_hand` из test_preflop_analysis.py, но с
    фолдом вместо шова — нужна раздача, где префильтр обязан НЕ сработать
    (сильная рука, сброшенная вопреки чарту), чтобы полный расчёт остался
    единственным источником вердикта для настоящей ошибки.
    """
    order = list(_SIX_MAX_SEATS[2:]) + ["SB", "BB"]  # порядок хода на префлопе
    hero_index = len(order) - players_behind - 1
    hero_position = order[hero_index]
    stack = round(eff_bb * _BB)
    labels, seats, posts = _six_max(dict.fromkeys(_SIX_MAX_SEATS, stack), hero_position)
    actions = [_fold(labels[pos]) for pos in order[:hero_index]]
    actions.append(_fold("Hero"))
    # Все, кто позади героя, тоже фолдят, кроме самого последнего — тот
    # выигрывает автоматом, когда все остальные сбросили, и явного действия
    # не подаёт. Без этого раздача осталась бы «незавершённой» для валидатора
    # (позиции после героя так и не походили) — на вердикт точки решения это
    # не влияет (табличное состояние строится ДО этих действий), но раздача
    # обязана быть валидной сама по себе, а не просто достаточной для ассерта.
    actions += [_fold(labels[pos]) for pos in order[hero_index + 1 : -1]]
    raw = _raw(
        seats=seats,
        button_seat=6,
        posts=posts,
        actions=actions,
        dealt={"Hero": list(hero_cards)},
    )
    return enrich(normalize(raw))


# --- Тесты плана (брифа) ---------------------------------------------------------


@requires_fixtures
def test_scan_daily_classic_runs():
    ens = [enrich(normalize(r)) for r in parse_file(FIXTURE_DAILY.read_text("utf-8"), "d")]
    s = scan_tournament(ens)
    assert s.hands_total == 146
    assert 0 < s.hands_with_decision <= 146  # префильтр отсёк тривиальные фолды
    assert all(s.items[i].ev_diff_bb <= s.items[i + 1].ev_diff_bb for i in range(len(s.items) - 1))
    assert all(it.ev_diff_bb < -0.1 for it in s.items)


def test_prefilter_cheap(monkeypatch):
    import harness.analysis.tools.equity as eq

    calls = {"n": 0}
    real = eq.equity_vs_range
    monkeypatch.setattr(
        eq,
        "equity_vs_range",
        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or real(*a, **k),
    )
    # синтетика: HU 10bb, Hero на BTN фолдит 32o без добровольных вложений; Нэш согласен (фолд)
    en = _make_hu_fold_hand(hero_cards=("3c", "2d"), eff_bb=10.0)
    s = scan_tournament([en])
    assert s.items == []  # расхождения нет
    assert calls["n"] == 0  # EV не считался — сработал префильтр (SCALING §3)


@requires_fixtures
def test_scan_pko_bounty_runs():
    """Вторая реальная фикстура — те же инварианты, другой формат (PKO)."""
    ens = [enrich(normalize(r)) for r in parse_file(FIXTURE_PKO.read_text("utf-8"), "pko")]
    s = scan_tournament(ens)
    assert s.hands_total == 172
    assert 0 < s.hands_with_decision <= 172
    assert all(s.items[i].ev_diff_bb <= s.items[i + 1].ev_diff_bb for i in range(len(s.items) - 1))
    assert all(it.ev_diff_bb < -0.1 for it in s.items)


# --- Честность гейта: префильтр не имеет права спрятать настоящую ошибку --------


def test_prefilter_does_not_fire_on_a_real_blunder():
    """AA сброшено в неоткрытый банк хедз-ап — чарт говорит «шов», префильтр молчит.

    Если бы префильтр закрывал ЛЮБОЙ фолд без расчёта, эта раздача выпала бы из
    скана как «расхождения нет» — то есть пробел выдавался бы за верную игру,
    ровно то, что запрещает `error_cost.py`. Полный расчёт обязан отработать и
    найти дорогую потерю.

    Нарочно хедз-ап (1 живой позади), а не мультивей: полный расчёт против
    ОДНОГО оппонента идёт по таблице эквити (быстро), и тест проверяет именно
    инвариант «префильтр не маскирует ошибку», а не скорость полного расчёта на
    мультивее — та отдельно замерена на реальных фикстурах (отчёт задачи 13).
    """
    en = _make_multiway_fold_hand(hero_cards=("Ac", "As"), eff_bb=12.0, players_behind=1)
    s = scan_tournament([en])
    assert len(s.items) == 1
    assert s.items[0].hero_class == "AA"
    assert s.items[0].ev_diff_bb < -0.1


def test_cheap_fold_verdict_returns_none_for_a_real_blunder():
    """Прямая (белоящичная) проверка: лукап не даёт вердикта для AA — вес шова велик."""
    en = _make_multiway_fold_hand(hero_cards=("Ac", "As"), eff_bb=12.0, players_behind=3)
    dp = en.report.decision_points[0]
    assert cheap_fold_verdict(dp, en) is None


def test_cheap_fold_verdict_hu_is_strict_no_assumption():
    """Хедз-ап фолд, закрытый префильтром, обязан быть `strict` без допущения.

    Тот же инвариант, что и у полного расчёта (задача 12): точный расчёт есть
    там, где колл-диапазон не угадан — хедз-ап SB против BB. Дешёвый лукап не
    имеет права понижать уверенность там, где расчёт её и не требует.
    """
    en = _make_hu_fold_hand(hero_cards=("3c", "2d"), eff_bb=10.0)
    dp = en.report.decision_points[0]
    point = cheap_fold_verdict(dp, en)
    assert point is not None
    assert point.zone is Zone.STRICT
    assert point.assumption is None
    assert point.best_action == "fold"
    assert point.action_taken == "fold"
    assert point.ev_diff_bb == 0.0


def test_cheap_fold_verdict_multiway_is_assuming_with_shown_range():
    """Мультивей-фолд, закрытый префильтром, обязан нести допущение (правило зоны).

    Мультивей-равновесия у нас нет (задача 12), и дешёвый лукап здесь опирается
    на равновесный чарт ОДНОГО оппонента на самой короткой глубине — то есть
    сам является допущением и обязан быть показан игроку, а не выдан за точный
    расчёт (контракт `PointVerdict._assumption_matches_zone`).
    """
    en = _make_multiway_fold_hand(hero_cards=("3c", "2d"), eff_bb=8.0, players_behind=3)
    dp = en.report.decision_points[0]
    point = cheap_fold_verdict(dp, en)
    assert point is not None
    assert point.zone is Zone.ASSUMING
    assert isinstance(point.assumption, Assumption)


def _make_fold_with_short_allin_bb(hero_cards: tuple[str, str], eff_bb: float):
    """UTG (Hero) фолдит неоткрытый банк за столом, где BB — вынужденный олл-ин.

    BB посажен со стеком РОВНО в блайнд: пост забирает стек целиком, и BB живой,
    но `behind == 0` — тот самый случай, который `_unopened_verdict` учитывает
    отдельно (`all_in_behind`) и который заставляет `zone_for` вернуть
    `assuming` через `unmodelled`, а не через сетку ширин. Полный перебор
    подмножеств коллеров этого игрока просто не видит; префильтр не имеет права
    быть увереннее него.
    """
    hero_stack = round(eff_bb * _BB)
    labels = {"SB": "SB", "BB": "BB", "UTG": "Hero", "BTN": "BTN"}
    seats = [
        SeatInfo(seat=1, label=labels["SB"], stack=hero_stack * 4),
        SeatInfo(seat=2, label=labels["BB"], stack=_BB),  # весь стек уходит на пост
        SeatInfo(seat=3, label=labels["UTG"], stack=hero_stack),
        SeatInfo(seat=4, label=labels["BTN"], stack=hero_stack * 4),
    ]
    posts = [
        Post(label=labels["SB"], kind=PostKind.SMALL_BLIND, amount=_SB),
        Post(label=labels["BB"], kind=PostKind.BIG_BLIND, amount=_BB),
    ]
    # Ход после героя (UTG): BTN, SB, затем BB — но BB уже без фишек (весь
    # стек ушёл на пост) и решения не принимает. BTN и SB фолдят явно, чтобы
    # раздача была валидна целиком; вердикт точки героя это не трогает —
    # табличное состояние строится ДО этих действий.
    actions = [_fold("Hero"), _fold(labels["BTN"]), _fold(labels["SB"])]
    raw = _raw(
        seats=seats,
        button_seat=4,
        posts=posts,
        actions=actions,
        dealt={"Hero": list(hero_cards)},
    )
    return enrich(normalize(raw))


def test_cheap_fold_verdict_defers_when_a_live_player_behind_is_already_all_in():
    """Живой олл-ин от блайнда позади — вне модели, префильтр обязан отступить.

    Даже для заведомо мусорной руки (72o) лукап не имеет права закрыть точку:
    `_unopened_verdict` в этой же ситуации ставит `assuming` через `unmodelled`
    (см. докстринг `zone_for`), а не через сетку ширин, — префильтр не обязан
    и не должен пытаться воспроизвести это решение дёшево, он обязан просто
    уступить дорогу полному расчёту.
    """
    en = _make_fold_with_short_allin_bb(hero_cards=("7c", "2d"), eff_bb=10.0)
    dp = en.report.decision_points[0]
    assert cheap_fold_verdict(dp, en) is None


def _make_wide_field_fold_hand(hero_cards: tuple[str, str], eff_bb: float):
    """9-max, Hero на UTG фолдит неоткрытый банк — 8 живых позади героя.

    Больше `_MAX_MODELLED_CALLERS` (7) в `_unopened_verdict`: полный перебор
    подмножеств коллеров на таком поле сам не считается и возвращает точку без
    вердикта. Префильтр обязан отступить по той же границе, а не выдать
    вердикт там, где полный расчёт прямо отказывается судить.
    """
    order = ("UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB")
    labels = {pos: ("Hero" if pos == "UTG" else pos) for pos in order}
    stack = round(eff_bb * _BB)
    seats = [SeatInfo(seat=i + 1, label=labels[pos], stack=stack) for i, pos in enumerate(order)]
    posts = [
        Post(label=labels["SB"], kind=PostKind.SMALL_BLIND, amount=_SB),
        Post(label=labels["BB"], kind=PostKind.BIG_BLIND, amount=_BB),
    ]
    # Все семеро между героем и BB фолдят явно; BB выигрывает автоматом и
    # действия не подаёт — та же логика, что и в `_make_multiway_fold_hand`.
    actions = [_fold("Hero")] + [_fold(labels[pos]) for pos in order[1:-1]]
    raw = _raw(
        seats=seats,
        button_seat=7,  # BTN — седьмое место в order (индекс 6, seat 7)
        posts=posts,
        actions=actions,
        dealt={"Hero": list(hero_cards)},
    )
    return enrich(normalize(raw))


def test_cheap_fold_verdict_defers_when_too_many_callers_behind():
    """8 живых позади (> `_MAX_MODELLED_CALLERS`) — префильтр не увереннее полного расчёта."""
    en = _make_wide_field_fold_hand(hero_cards=("7c", "2d"), eff_bb=10.0)
    dp = en.report.decision_points[0]
    assert cheap_fold_verdict(dp, en) is None
    # И полный путь тоже не выносит вердикта — это не «расхождение», а пробел.
    assert scan_tournament([en]).items == []


# --- Сборка сводки: сортировка, порог, зоны, отказ по руке ----------------------


def test_scan_summary_shape_and_threshold():
    trivial = _make_hu_fold_hand(hero_cards=("3c", "2d"), eff_bb=10.0)
    blunder = _make_multiway_fold_hand(hero_cards=("Ac", "As"), eff_bb=12.0, players_behind=3)
    s = scan_tournament([trivial, blunder])
    assert isinstance(s, ScanSummary)
    assert s.hands_total == 2
    assert s.hands_with_decision == 2
    assert len(s.items) == 1
    item = s.items[0]
    assert isinstance(item, ScanItem)
    assert item.hand_index is None or isinstance(item.hand_index, int)
    assert item.hand_no == blunder.hand.hand_no
    # total_loss_bb суммирует ВСЕ отрицательные расхождения по вердикту (не только
    # показанные в items — там ещё и порог 0.1bb), поэтому здесь он равен цене
    # единственной судимой потери: у тривиального фолда ev_diff_bb == 0.
    assert s.total_loss_bb == pytest.approx(item.ev_diff_bb)


def test_scan_hands_with_no_hero_decision_are_not_counted():
    """Рука без единой точки решения героя не считается «рукой с решением»."""
    en = _make_hu_fold_hand(hero_cards=("3c", "2d"), eff_bb=10.0)
    en = en.model_copy(update={"report": en.report.model_copy(update={"decision_points": []})})
    s = scan_tournament([en])
    assert s.hands_total == 1
    assert s.hands_with_decision == 0
    assert s.items == []


def test_scan_skips_a_hand_that_fails_reconciliation_and_counts_it(monkeypatch):
    """Политика отказа: одна нестандартная рука не должна ронять весь скан.

    `verdict_for`/`cheap_fold_verdict` падают `ValueError`, когда восстановленный
    анализом стол расходится с движком (`classifier._cross_check`) — сигнал
    внутреннего рассогласования, а не форма руки, которую видит пользователь.
    Разбор одной руки обязан падать (там ждут ответ по конкретной раздаче), а
    скан по файлу — нет: пропуск виден через `hands_failed`, остальные руки
    файла разбираются как обычно.
    """
    import harness.analysis.scan as scan_mod

    good = _make_hu_fold_hand(hero_cards=("3c", "2d"), eff_bb=10.0)
    bad = _make_multiway_fold_hand(hero_cards=("Ac", "As"), eff_bb=12.0, players_behind=3)

    real_verdict_for = scan_mod.verdict_for

    def flaky(dp, en):
        if en is bad:
            raise ValueError("восстановленный стол разошёлся с движком (синтетика теста)")
        return real_verdict_for(dp, en)

    monkeypatch.setattr(scan_mod, "verdict_for", flaky)
    monkeypatch.setattr(scan_mod, "cheap_fold_verdict", lambda dp, en: None)

    s = scan_tournament([good, bad])
    assert s.hands_total == 2
    assert s.hands_failed == 1
    assert s.hands_with_decision == 1  # только «good» дошла до вердикта
    assert s.items == []  # good — верный фолд, bad пропущена целиком
