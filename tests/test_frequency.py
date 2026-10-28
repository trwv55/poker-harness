"""Арифметика словаря расчётов: интервал, вердикт, недостающая выборка, выбор мест.

Ни базы, ни модели — всё, что здесь проверяется, считается из пары чисел.
Синтетические руки берутся у tests/test_player_stats.py: тот же конвейер
(`normalize` → `enrich`), и второго построителя рук в тестах нет.
"""

from __future__ import annotations

import importlib.util

import pytest

import harness.calcs as calcs_module
from harness.analysis.frequency import (
    compare_to_reference,
    compare_to_threshold,
    defense_from_bet,
    observations_needed,
    observations_needed_against,
    threshold_from_bet,
    wilson_interval,
)
from harness.analysis.player_stats import (
    measurement_of,
    player_stats,
    player_stats_of_seats,
    seat_position,
)
from harness.analysis.tools.pot_odds import required_equity
from harness.calcs import REGISTRY, seats_of
from harness.contracts import (
    POSITIONS,
    CalcName,
    FrequencyStat,
    Measurement,
    PlayerStats,
    RangeNarrowing,
    Subject,
    ThresholdOutcome,
    ThresholdSide,
)
from harness.normalizer import POSITIONS_BY_COUNT
from tests.test_player_stats import _fold, _hand, _raise_to

# Порог, вокруг которого собраны примеры на вердикт: одно и то же число против
# разных выборок, чтобы видно было, что решает не их размер.
_THRESHOLD = 0.61


def _interval(numerator: float, denominator: float) -> tuple[float, float]:
    """Интервал при заведомо ненулевом знаменателе — без него распаковка пары."""
    bounds = wilson_interval(numerator, denominator)
    assert bounds is not None
    return bounds


# --- интервал ----------------------------------------------------------------------


def test_a_wide_gap_on_a_small_sample_decides():
    """Двадцать девять наблюдений с большим отрывом дают вердикт.

    Фиксированный порог на выборку (`n >= 30`) этот вывод запретил бы, хотя
    интервал лежит целиком выше порога. Пара к
    `test_a_narrow_gap_on_a_large_sample_stays_undecided`: вместе они
    показывают, что размер выборки решением не является ни в одну сторону.
    """
    low, high = _interval(23, 29)
    assert (round(low, 4), round(high, 4)) == (0.6161, 0.9015)
    assert compare_to_threshold(Measurement(numerator=23, denominator=29), _THRESHOLD) == (
        ThresholdOutcome.ABOVE,
        None,
    )


def test_a_narrow_gap_on_a_large_sample_stays_undecided():
    """Сто наблюдений с малым отрывом вердикта не дают: интервал накрывает порог."""
    low, high = _interval(63, 100)
    assert (round(low, 4), round(high, 4)) == (0.5322, 0.7182)
    outcome, needed = compare_to_threshold(
        Measurement(numerator=63, denominator=100), _THRESHOLD
    )
    assert outcome is ThresholdOutcome.UNDECIDED
    assert needed is not None and needed > 100


def test_the_interval_never_leaves_the_unit_segment():
    """Доли 0 и 1 остаются с интервалом ненулевой ширины внутри [0, 1].

    Нормальное приближение `p ± z·√(p(1−p)/n)` даёт здесь ширину ноль и объявило
    бы вердикт по пяти наблюдениям при любом пороге.
    """
    for numerator in (0, 5):
        low, high = _interval(numerator, 5)
        assert 0.0 <= low < high <= 1.0


def test_no_observations_leave_the_interval_undefined():
    """При нулевом знаменателе интервала нет, и вердикта тоже нет."""
    assert wilson_interval(0, 0) is None
    assert compare_to_threshold(Measurement(numerator=0, denominator=0), _THRESHOLD) == (
        ThresholdOutcome.UNDECIDED,
        None,
    )


def test_a_verdict_below_the_threshold_is_reported_as_below():
    """Интервал целиком ниже порога — тоже утверждение, а не «данных мало»."""
    outcome, needed = compare_to_threshold(
        Measurement(numerator=2, denominator=40), _THRESHOLD
    )
    assert (outcome, needed) == (ThresholdOutcome.BELOW, None)


# --- недостающая выборка -----------------------------------------------------------


def test_the_needed_sample_is_the_first_that_decides():
    """Найденное число решает, а на единицу меньшее — ещё нет.

    Это и есть определение «сколько наблюдений не хватает»: наименьшее, при
    котором знак определяется, а не круглое число рядом.
    """
    share = 9 / 12
    needed = observations_needed(share, _THRESHOLD)
    assert needed is not None
    low, high = _interval(share * needed, needed)
    assert low > _THRESHOLD or high < _THRESHOLD
    one_less_low, one_less_high = _interval(share * (needed - 1), needed - 1)
    assert one_less_low <= _THRESHOLD <= one_less_high


def test_a_share_equal_to_the_threshold_never_decides():
    """Доля, совпавшая с порогом, не расходится с ним ни при каком числе наблюдений."""
    assert observations_needed(_THRESHOLD, _THRESHOLD) is None


def test_a_share_next_to_the_threshold_needs_more_than_the_ceiling():
    """Отрыв в тысячные требует выборки выше потолка поиска — ответом идёт `None`.

    Это не отказ считать, а честный ответ «столько не собрать»: число,
    напечатанное здесь, было бы больше всех рук, сыгранных за жизнь.
    """
    assert observations_needed(_THRESHOLD + 0.0005, _THRESHOLD) is None


def test_nothing_observed_yields_no_needed_sample():
    """Недостающая выборка не считается от доли, которой ещё не наблюдали."""
    assert (
        compare_to_threshold(Measurement(numerator=0, denominator=0), _THRESHOLD)[1] is None
    )


# --- измеренный порог --------------------------------------------------------------


def test_a_measured_threshold_is_harder_to_beat_than_the_same_number_as_a_point():
    """То же число порогом-точкой даёт вердикт, а порогом-измерением — нет.

    У измеренного порога своя неопределённость, и сравнение с ним как с точкой
    объявило бы расхождение там, где его не видно.
    """
    measured = Measurement(numerator=23, denominator=29)
    reference = Measurement(numerator=61, denominator=100)
    assert compare_to_threshold(measured, 0.61)[0] is ThresholdOutcome.ABOVE
    assert compare_to_reference(measured, reference)[0] is ThresholdOutcome.UNDECIDED


def test_a_wide_enough_gap_beats_a_measured_threshold_too():
    """Против измеренного эталона вердикт всё же выдаётся — когда отрыв велик."""
    outcome, needed = compare_to_reference(
        Measurement(numerator=180, denominator=200), Measurement(numerator=61, denominator=200)
    )
    assert (outcome, needed) == (ThresholdOutcome.ABOVE, None)


def test_a_thin_reference_can_leave_the_difference_undecided_forever():
    """Узкий эталон не сужается от чужих наблюдений — знака может не быть никогда."""
    assert observations_needed_against(0.75, Measurement(numerator=1, denominator=2)) is None


def test_the_needed_sample_against_a_reference_grows_only_the_measured_side():
    """Считается выборка ИЗМЕРЯЕМОГО: эталон остаётся тем, что есть."""
    reference = Measurement(numerator=61, denominator=100)
    needed = observations_needed_against(23 / 29, reference)
    assert needed is not None and needed > 29
    assert (
        compare_to_reference(
            Measurement(numerator=round(23 / 29 * needed), denominator=needed), reference
        )[0]
        is ThresholdOutcome.ABOVE
    )


# --- требуемая частота защиты ------------------------------------------------------


def test_the_defence_frequency_is_the_one_that_zeroes_a_bet():
    """Подстановка возвращённой частоты обращает выигрыш ставки в ноль.

    Ставка `bet` в банк `pot_before` при защите с частотой `f` приносит
    `(1 - f) * pot_before - f * bet`; проверяется само это выражение, а не
    формула, которой оно решено.
    """
    for pot_before, bet in ((100, 50), (100, 100), (30, 7), (100, 250)):
        f = defense_from_bet(pot_before, bet).defend_frequency
        assert (1.0 - f) * pot_before - f * bet == pytest.approx(0.0)


def test_the_two_sides_of_the_threshold_add_up_to_one():
    """Защита и сдача — дополнения друг друга, и сторона выбирается параметром."""
    requirement = defense_from_bet(100, 75)
    assert requirement.defend_frequency + requirement.fold_frequency == pytest.approx(1.0)
    assert threshold_from_bet(100, 75, ThresholdSide.DEFEND) == requirement.defend_frequency
    assert threshold_from_bet(100, 75, ThresholdSide.FOLD) == requirement.fold_frequency


def test_the_required_equity_is_the_pot_odds_of_the_same_bet():
    """Эквити колла берётся у `pot_odds`, а не считается здесь второй формулой."""
    assert defense_from_bet(100, 50).required_equity == required_equity(50, 150)


@pytest.mark.parametrize(("pot_before", "bet"), [(0, 50), (100, 0), (-1, 10)])
def test_a_bet_or_a_pot_that_is_not_chips_is_refused(pot_before: int, bet: int):
    """Нулевой банк или нулевая ставка — отказ, а не частота защиты, равная единице."""
    with pytest.raises(ValueError):
        defense_from_bet(pot_before, bet)


# --- имена, позиции, знаменатели ---------------------------------------------------


def test_positions_are_the_ones_the_normalizer_writes():
    """Список позиций словаря совпадает с тем, которым нормализатор подписывает места.

    Копия, а не импорт (`contracts` не тянут конвейер), поэтому равенство
    держится этим тестом: новая позиция в нормализаторе краснит его.
    """
    assert POSITIONS == {pos for row in POSITIONS_BY_COUNT.values() for pos in row}


def test_every_named_frequency_matches_the_share_the_contract_computes():
    """Пара «числитель, знаменатель» каждой частоты сходится с долей самого контракта.

    Это и есть защита от расхождения подписи со знаменателем: числитель,
    приставленный к чужому знаменателю, разошёлся бы с процентом, который
    `PlayerStats` считает по своим полям.
    """
    stats = PlayerStats(
        hands=40,
        vpip=11,
        pfr=7,
        reraise=3,
        reraise_chances=9,
        fold_to_cbet=5,
        cbet_faced=13,
        cbet_flop=8,
        cbet_flop_chances=15,
        barrel_turn=4,
        barrel_turn_chances=8,
        barrel_river=2,
        barrel_river_chances=4,
        fold_to_turn_barrel=3,
        turn_barrel_faced=7,
        raise_cbet=2,
        raise_turn_barrel=1,
        showdowns=6,
        flops_seen=17,
    )
    shares = {
        FrequencyStat.VPIP: stats.vpip_pct,
        FrequencyStat.PFR: stats.pfr_pct,
        FrequencyStat.RERAISE: stats.reraise_pct,
        FrequencyStat.FOLD_TO_CBET: stats.fold_to_cbet_pct,
        FrequencyStat.CBET_FLOP: stats.cbet_flop_pct,
        FrequencyStat.BARREL_TURN: stats.barrel_turn_pct,
        FrequencyStat.BARREL_RIVER: stats.barrel_river_pct,
        FrequencyStat.FOLD_TO_TURN_BARREL: stats.fold_to_turn_barrel_pct,
        FrequencyStat.RAISE_CBET: stats.raise_cbet_pct,
        FrequencyStat.RAISE_TURN_BARREL: stats.raise_turn_barrel_pct,
        FrequencyStat.SHOWDOWN: stats.showdown_pct,
    }
    assert set(shares) == set(FrequencyStat)
    for stat, pct in shares.items():
        measurement = measurement_of(stats, stat)
        assert measurement.share is not None and pct is not None
        assert measurement.share * 100.0 == pytest.approx(pct)


def test_the_dictionary_covers_every_name_exactly_once():
    """У каждого имени набора один тип параметров и один исполнитель.

    Имя, забытое в реестре, вызвать нечем: диспетчер берёт исполнителя отсюда
    же, второго списка нет.
    """
    assert set(REGISTRY) == set(CalcName)
    for name, (params_type, runner) in REGISTRY.items():
        assert params_type.model_fields["calc"].default is name
        assert callable(runner)


def test_the_line_range_left_no_trace():
    """Колл-диапазон по линии снят (спека 2026-10-24, §8): ни имени в наборе, ни ядра,
    ни кэша порядка рук; частоты рейза на с-бет и на второй баррель остаются."""
    assert "line_range" not in {name.value for name in CalcName}
    assert importlib.util.find_spec("harness.analysis.line_range") is None
    assert not hasattr(calcs_module, "line_order_cache_prefix")
    assert {FrequencyStat.RAISE_CBET, FrequencyStat.RAISE_TURN_BARREL} <= set(FrequencyStat)
    old = {"analysis_id": 1, "opponent_position": "BB", "street": "flop", "classes": ["AA"],
           "by": "line_range"}
    assert RangeNarrowing.model_validate(old).classes == ["AA"]  # старая история читается


def test_the_fold_hint_and_the_single_equity_left_no_trace():
    """Ориентир сдачи (§13) и одно общее эквити сняты (спека 2026-10-24, §14.4): ни
    подсказки, ни линии ставки героя, ни `assumed` с его исключениями, ни Монте-Карло
    агента; старый выход эквити в истории читается."""
    import inspect

    from harness import contracts
    from harness.agent import history
    from harness.analysis import player_stats
    from harness.calcs import routing
    from harness.contracts import EquityVsRangeResult
    from harness.explanation import agent_checks
    from harness.memory.repos import OpponentsRepo

    assert not hasattr(contracts, "FoldHint")
    assert not {"fold_hint", "equity", "share_of_start"} & set(EquityVsRangeResult.model_fields)
    for name in ("_fold_hint", "AGENT_EQUITY_ITERATIONS", "AGENT_EQUITY_SEED",
                 "equity_cache_prefix"):
        assert not hasattr(calcs_module, name), name
    assert not hasattr(player_stats, "hero_bet_called")
    assert not hasattr(player_stats, "bet_reply")
    assert not hasattr(OpponentsRepo, "holder")
    assert not hasattr(history, "assumed_numbers_in")
    for name in ("_assumed_share_in", "_frequency_share", "_threshold_share",
                 "_hero_chances_share"):
        assert not hasattr(agent_checks, name), name
    parameters = inspect.signature(agent_checks.check_agent_answer).parameters
    assert not {"assumed_numbers", "summary_numbers"} & set(parameters)
    assert '"assumed"' not in inspect.getsource(routing._perform)
    old = {"calc": "equity_vs_range", "opponent_position": "BB", "street": "flop",
           "range_text": "AA", "live_combos": 6.0, "share_of_start": 0.1, "equity": 0.3,
           "fold_hint": None}
    assert EquityVsRangeResult.model_validate(old).layout == []


# --- выбор мест ---------------------------------------------------------------------


def _hero_opens_from_co():
    return _hand(
        hero_position="CO",
        actions=[
            _fold("UTG"),
            _fold("HJ"),
            _raise_to("Hero", 6, already=0),
            *[_fold(pos) for pos in ("BTN", "SB", "BB")],
        ],
    )


def test_the_field_counts_every_seat_but_the_hero():
    """Поле — все места, кроме героя, сложенные в одно число."""
    hand = _hero_opens_from_co()
    assert sorted(seats_of(Subject.FIELD, None)(hand)) == sorted(
        p.label for p in hand.players if p.label != hand.hero_label
    )


def test_two_seats_in_one_hand_count_twice():
    """Знаменатель поля — место-раздача: одна рука прибавляет столько, сколько мест."""
    hand = _hero_opens_from_co()
    stats = player_stats_of_seats([hand], seats_of(Subject.FIELD, None))
    assert stats.hands == len(hand.players) - 1


def test_a_position_filter_keeps_only_the_seat_that_sat_there():
    """Фильтр по позиции отсекает места, сидевшие в этой раздаче не там."""
    hand = _hero_opens_from_co()
    assert seats_of(Subject.FIELD, "BTN")(hand) == ["BTN"]
    assert seats_of(Subject.HERO, "CO")(hand) == [hand.hero_label]
    assert seats_of(Subject.HERO, "BTN")(hand) == []
    assert seat_position(hand, hand.hero_label) == "CO"


def test_an_opponent_is_counted_only_where_the_binding_names_him():
    """Турнир без сшивки не даёт ни одного места: про него не сказано, кто в нём он."""
    hand = _hero_opens_from_co()
    assert seats_of(Subject.OPPONENT, None, {hand.tournament_id: "BTN"})(hand) == ["BTN"]
    assert seats_of(Subject.OPPONENT, None, {})(hand) == []


def test_every_entry_point_shares_one_accumulator():
    """`player_stats` — это `player_stats_of_seats` с местом героя, а не вторая формула."""
    hands = [_hero_opens_from_co()]
    assert player_stats(hands) == player_stats_of_seats(
        hands, seats_of(Subject.HERO, None)
    )
