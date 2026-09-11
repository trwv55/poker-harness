"""Выжимка выхода расчёта и проверка ответа модели — без базы и без сети.

Здесь проверяется КОД вокруг модели: что уходит в промпт и что принимается
обратно. Выбор расчёта этими тестами не проверяется вовсе и проверен быть не
может — маршрут меряется размеченным набором вопросов (EVALS, этаж 3), которого
в проекте пока нет.
"""

from __future__ import annotations

import pytest

from harness.contracts import (
    CalcName,
    CalcResult,
    CoverageResult,
    DefenseResult,
    FrequencyResult,
    FrequencyStat,
    LeakRule,
    LeaksResult,
    LeakStat,
    Measurement,
    PointFilter,
    SpotKind,
    Street,
    Subject,
    ThresholdOutcome,
    ThresholdResult,
    Window,
)
from harness.explanation.faithfulness import numbers_in
from harness.explanation.question import check_answer, question_digest
from harness.explanation.verdict_text import UnfaithfulText


def _frequency() -> FrequencyResult:
    return FrequencyResult(
        calc=CalcName.HERO_FREQUENCY,
        subject=Subject.HERO,
        stat=FrequencyStat.CBET_FLOP,
        position="BTN",
        window=Window(),
        measurement=Measurement(numerator=5, denominator=7),
    )


def _opponent_frequency() -> FrequencyResult:
    return FrequencyResult(
        calc=CalcName.OPPONENT_FREQUENCY,
        subject=Subject.FIELD,
        stat=FrequencyStat.FOLD_TO_CBET,
        window=Window(session_id=3),
        measurement=Measurement(numerator=0, denominator=0),
    )


def _coverage() -> CoverageResult:
    return CoverageResult(
        calc=CalcName.COVERAGE,
        filter=PointFilter(street=Street.PREFLOP, spot=SpotKind.PUSHFOLD_UNOPENED),
        judged=Measurement(numerator=38, denominator=330),
        priced=Measurement(numerator=21, denominator=38),
        loss_bb=12.4,
    )


def _leaks() -> LeaksResult:
    rule = LeakRule(
        key="no_shove",
        title="Не шовит, где надо",
        spot=SpotKind.PUSHFOLD_UNOPENED,
        action_taken="fold",
        best_action="shove",
    )
    return LeaksResult(
        calc=CalcName.LEAKS,
        window=Window(),
        judged=Measurement(numerator=38, denominator=330),
        leaks=[LeakStat(rule=rule, count=6, loss_bb=9.3)],
    )


def _defense() -> DefenseResult:
    return DefenseResult(
        calc=CalcName.DEFENSE_FREQUENCY,
        pot_before=100,
        bet=50,
        defend_frequency=2 / 3,
        fold_frequency=1 / 3,
        required_equity=0.25,
    )


def _threshold(outcome: ThresholdOutcome, needed: int | None) -> ThresholdResult:
    return ThresholdResult(
        calc=CalcName.FREQUENCY_VS_THRESHOLD,
        subject=Subject.HERO,
        stat=FrequencyStat.FOLD_TO_CBET,
        window=Window(),
        measurement=Measurement(numerator=23, denominator=29),
        threshold=0.61,
        threshold_source="field",
        reference=Measurement(numerator=61, denominator=100),
        outcome=outcome,
        observations_needed=needed,
    )


ALL_RESULTS: list[CalcResult] = [
    _frequency(),
    _opponent_frequency(),
    _coverage(),
    _leaks(),
    _defense(),
    _threshold(ThresholdOutcome.ABOVE, None),
    _threshold(ThresholdOutcome.UNDECIDED, 42),
]


@pytest.mark.parametrize("result", ALL_RESULTS, ids=lambda r: r.calc.value)
def test_the_digest_registers_every_number_it_prints(result: CalcResult):
    """Ни одно число не печатается мимо `NumberBook`.

    Дисциплина автора выжимки, которую реестр гарантировать не может: обычная
    f-строка ему не подотчётна. Вычитаем разрешённые из чисел готового текста и
    требуем пустоты — тот же тест, что у выжимок разбора и турнира, на всех
    шести расчётах разом.
    """
    digest = question_digest(result)
    assert set(numbers_in(digest.text)) - digest.allowed == set()


@pytest.mark.parametrize("result", ALL_RESULTS, ids=lambda r: r.calc.value)
def test_the_digest_names_the_calculation_it_came_from(result: CalcResult):
    """Выжимка называет расчёт: модель не должна гадать, чей это выход."""
    assert result.calc.value in question_digest(result).text


def test_every_calculation_name_has_a_digest():
    """У каждого имени словаря есть выжимка — непокрытого имени в наборе нет."""
    assert {result.calc for result in ALL_RESULTS} == set(CalcName)


def test_a_measurement_without_observations_gets_no_share():
    """«0 из 0» — не ноль процентов: доли у пустого знаменателя не печатается."""
    text = question_digest(_opponent_frequency()).text
    assert "наблюдений нет" in text
    assert "%" not in text


def test_a_number_from_the_calculation_passes():
    digest = question_digest(_frequency())
    check_answer("Вы ставите продолженную ставку в 71.4% случаев (5 из 7).", digest.allowed)


def test_an_invented_number_is_refused():
    """Число, которого в выходе инструмента нет, отбраковывает ответ целиком."""
    digest = question_digest(_frequency())
    with pytest.raises(UnfaithfulText):
        check_answer("Вы ставите продолженную ставку в 62.0% случаев.", digest.allowed)


def test_calling_a_decision_a_mistake_is_refused():
    """«Ошибка» — вопрос верности, а не тона: расчёт судит решение против
    диапазона, а измеренная частота не судит его вовсе."""
    digest = question_digest(_frequency())
    with pytest.raises(UnfaithfulText):
        check_answer("Это ошибка: 5 из 7 — слишком часто.", digest.allowed)


def test_an_empty_answer_is_a_refusal_not_a_text():
    with pytest.raises(UnfaithfulText):
        check_answer("   ", question_digest(_frequency()).allowed)
