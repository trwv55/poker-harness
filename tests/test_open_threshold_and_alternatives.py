"""Порог опена первым и альтернативы последнему чеку или бету героя.

Решения владельца 2026-10-10 (спека постфлоп-линии, §12). Руки синтетические:
метки мест свои, каждая проходит настоящий конвейер (`normalize` → `enrich`).
Считает ядро (`analysis.postflop_line`), `presentation` только печатает.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from harness.analysis import analyze_hand
from harness.analysis.postflop_line import PLAYED_HALF_POT_PCT
from harness.contracts import (
    OPEN_THRESHOLD_DETAIL,
    POSTFLOP_LINE_DETAIL,
    ActionKind,
    PointVerdict,
    RawAction,
    open_threshold_detail,
    postflop_line_detail,
)
from tests.test_open_chart import _hand as _table_hand
from tests.test_postflop_line import (
    _HERO_OPENS,
    F,
    P,
    R,
    T,
    _bet,
    _call,
    _check,
    _fold,
    _hand,
    _ip,
    _raise,
)
from tests.test_postflop_line_text import _text

# --- порог опена первым ------------------------------------------------------------------

# Стол `tests.test_open_chart`: 8 мест, блайнды 50/100, анте 12 с места, рейз до 250.
_POT_BEFORE_OPEN = 8 * 12 + 50 + 100


def _preflop_point(en) -> PointVerdict:
    return analyze_hand(en).points[0]


def test_an_open_from_the_button_risks_the_whole_raise():
    threshold = open_threshold_detail(_preflop_point(_table_hand("BTN", ("Kh", "9h"), 35.0, "raise")))
    assert threshold is not None
    assert (threshold.risk, threshold.pot) == (250, _POT_BEFORE_OPEN)
    assert threshold.fold_share == 250 / (_POT_BEFORE_OPEN + 250)


def test_an_open_from_the_small_blind_risks_the_raise_without_its_blind():
    """На SB 0.5BB уже в банке: в риск идёт рейз сверх блайнда, анте не входит."""
    threshold = open_threshold_detail(_preflop_point(_table_hand("SB", ("Kh", "9h"), 35.0, "raise")))
    assert threshold is not None
    assert (threshold.risk, threshold.pot) == (200, _POT_BEFORE_OPEN)
    assert threshold.fold_share == 200 / (_POT_BEFORE_OPEN + 200)


def test_a_limp_a_fold_and_an_answer_to_an_open_carry_no_open_threshold():
    for en in (
        _table_hand("SB", ("Kh", "4h"), 35.0, "limp"),
        _table_hand("CO", ("Ah", "Jd"), 35.0, "fold"),
        _table_hand("CO", ("Ah", "Jd"), 35.0, "fold", opener="UTG"),
    ):
        assert OPEN_THRESHOLD_DETAIL not in analyze_hand(en).points[-1].detail


def test_an_open_is_cut_by_the_deepest_stack_that_can_call():
    """Рейз до 2.5BB против стека 2BB: при колле теряется только то, что соперник уравняет."""
    en = _hand([_raise(P, "Hero", 250), _fold(P, "V")], seats=(("Hero", 10_000), ("V", 200)))
    threshold = open_threshold_detail(_preflop_point(en))
    assert threshold is not None
    assert (threshold.risk, threshold.pot) == (200 - 50, 150)
    assert threshold.fold_share == 150 / 300


def test_an_all_in_open_carries_no_open_threshold():
    """Шов первым: там вердикт равновесия — ни данных порога, ни строки опена."""
    shove = RawAction(
        street=P,
        label="Hero",
        kind=ActionKind.RAISE,
        to_amount=10_000,
        is_all_in=True,
        raw_line="Hero: raise",
    )
    en = _hand([shove], seats=(("Hero", 10_000), ("V", 2_000)), button="Hero")
    assert OPEN_THRESHOLD_DETAIL not in _preflop_point(en).detail
    assert "опен " not in _text(en)
    table_shove = _table_hand("CO", ("Ah", "Ad"), 35.0, "shove")
    assert OPEN_THRESHOLD_DETAIL not in _preflop_point(table_shove).detail
    assert "опен " not in _text(table_shove)


def test_an_open_prints_the_fold_threshold_instead_of_the_pot_odds():
    text = _text(_table_hand("BTN", ("Kh", "9h"), 35.0, "raise"))
    block = text[text.index("1. Префлоп") :].split("\n\n")[0]
    assert "    опен 2.5BB в банк 2.5BB окупается сразу при фолдах от 50%" in block.splitlines()
    assert "доставить" not in block and "шансы банка" not in block
    assert "open_threshold" not in text


def test_a_small_blind_open_prints_the_risk_without_the_blind():
    text = _text(_table_hand("SB", ("Kh", "9h"), 35.0, "raise"))
    assert "    опен 2.0BB в банк 2.5BB окупается сразу при фолдах от 45%" in text.splitlines()


def test_a_limp_keeps_the_pot_odds_line():
    text = _text(_table_hand("SB", ("Kh", "4h"), 35.0, "limp"))
    assert "шансы банка: колл окупается от" in text and "опен " not in text


# --- альтернативы последнему чеку или бету -----------------------------------------------


def _checked_down(**kwargs):
    """Герой на кнопке открывает, BB коллирует, дальше чеки до конца."""
    return _ip(
        [
            *_HERO_OPENS,
            _check(F, "V"),
            _check(F, "Hero"),
            _check(T, "V"),
            _check(T, "Hero"),
            _check(R, "V"),
            _check(R, "Hero"),
        ],
        **kwargs,
    )


def _lines(en):
    return [postflop_line_detail(p) for p in analyze_hand(en).points if p.street is not P]


def test_the_last_check_gets_a_half_pot_bet_and_an_all_in():
    flop, turn, river = _lines(_checked_down())
    assert flop is not None and turn is not None and river is not None
    assert flop.alternatives is None and turn.alternatives is None, "не последнее решение"
    alt = river.alternatives
    assert alt is not None
    assert (alt.pot, alt.all_in_chips) == (600, 9_700)
    assert alt.half_pot == float(Fraction(1, 3))
    assert alt.all_in == 9_700 / (600 + 9_700)


def test_the_last_check_prints_both_thresholds():
    text = _text(_checked_down())
    assert (
        "    альтернатива: ставка 50% — нужно 33% фолдов; олл-ин 97.0BB в 6.0BB — нужно 94% фолдов"
        in text.splitlines()
    )
    assert text.count("альтернатива:") == 1


def test_no_half_pot_bet_when_the_stack_is_not_deeper_than_half_the_pot():
    """Эфф. стек 2BB при банке 6BB: ставка 50% и есть олл-ин — её строки нет."""
    en = _checked_down(seats=(("Hero", 10_000), ("V", 500)))
    *_, river = _lines(en)
    assert river is not None and river.alternatives is not None
    assert river.alternatives.half_pot is None
    assert river.alternatives.all_in == 200 / 800
    assert "    альтернатива: олл-ин 2.0BB в 6.0BB — нужно 25% фолдов" in _text(en).splitlines()


def test_the_last_bet_keeps_its_payoff_and_gets_the_alternatives():
    """Бет 75% банка: 450 / (600 + 450) = 42.9% → «43%»."""
    en = _ip([*_HERO_OPENS, _check(F, "V"), _bet(F, "Hero", 450), _fold(F, "V")])
    (flop,) = _lines(en)
    assert flop is not None and flop.fold_threshold is not None and flop.alternatives is not None
    lines = _text(en).splitlines()
    assert "    окупается: от 43% фолдов" in lines
    assert any(line.startswith("    альтернатива: ставка 50%") for line in lines)


def test_a_call_a_fold_an_all_in_and_an_earlier_decision_get_no_alternatives():
    calls = _ip([*_HERO_OPENS, _bet(F, "V", 300), _call(F, "Hero", 300)])
    folds = _ip([*_HERO_OPENS, _check(F, "V"), _check(F, "Hero"), _bet(T, "V", 300), _fold(T, "Hero")])
    jam = RawAction(
        street=F, label="Hero", kind=ActionKind.BET, amount=9_700, is_all_in=True, raw_line="jam"
    )
    all_in = _ip([*_HERO_OPENS, _check(F, "V"), jam])
    # Колл на флопе (дальше рука доиграна чеками — альтернатива у последнего из них);
    # чек и фолд тёрна; шов флопа.
    call_point = _lines(calls)[0]
    assert call_point is not None and call_point.alternatives is None
    for en in (folds, all_in):
        assert all(line is None or line.alternatives is None for line in _lines(en))
        assert "альтернатива:" not in _text(en)


def test_a_hand_that_ends_preflop_gets_no_alternatives():
    """Рука кончилась на префлопе: постфлоп-линии у точки нет, альтернатив тоже (§12)."""
    en = _table_hand("BTN", ("Kh", "9h"), 35.0, "raise")
    assert POSTFLOP_LINE_DETAIL not in _preflop_point(en).detail
    assert "альтернатива:" not in _text(en)


# Банк 20BB к флопу: рейз до 10BB и колл; 1% банка — 20 фишек, 0.1% — 2.
_BIG_POT_OPEN = [_raise(P, "Hero", 1_000), _call(P, "V", 900)]


@pytest.mark.parametrize(
    ("bet", "half_pot_printed"),
    [(898, True), (900, False), (1_100, False), (1_102, True)],
    ids=["44.9%", "45%", "55%", "55.1%"],
)
def test_a_played_bet_near_half_pot_is_not_offered_again(bet, half_pot_printed):
    """Бет от 45% до 55% банка включительно — и есть ставка 50%: остаётся только олл-ин."""
    assert PLAYED_HALF_POT_PCT == (45, 55)
    en = _ip([*_BIG_POT_OPEN, _check(F, "V"), _bet(F, "Hero", bet), _fold(F, "V")])
    (flop,) = _lines(en)
    assert flop is not None and flop.alternatives is not None
    assert (flop.alternatives.half_pot is not None) is half_pot_printed
    (line,) = [row for row in _text(en).splitlines() if row.startswith("    альтернатива:")]
    assert ("ставка 50%" in line) is half_pot_printed
    assert "олл-ин 90.0BB в 20.0BB" in line
