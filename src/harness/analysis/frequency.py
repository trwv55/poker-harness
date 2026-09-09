"""Частота против порога: интервал, вердикт и число недостающих наблюдений.

Чистая арифметика над парами «числитель, знаменатель». Ни базы, ни модели: на
вход приходят `Measurement`, наружу уходит вердикт.

**Интервал — Уилсон.** Причины, проверяемые не покидая файла:

* формула замкнутая и считается стандартной библиотекой — квантиль берётся у
  `statistics.NormalDist`, новой зависимости не нужно;
* концы не выходят за отрезок [0, 1] по построению (`_clamp` правит только
  ошибку округления), а ширина у доли 0 или 1 остаётся ненулевой — в отличие от
  нормального приближения `p ± z·√(p(1−p)/n)`, у которого она там ровно ноль
  (`test_the_interval_never_leaves_the_unit_segment`);
* вырождается только при нулевом знаменателе, где доли нет вовсе.

Тот же интервал служит и двухвыборочному сравнению: разность двух долей
собирается из двух интервалов Уилсона (`compare_to_reference`), поэтому метод в
модуле один, а не два.

**Вердикт — по интервалу, а не по числу наблюдений.** Утверждение выдаётся,
когда интервал целиком по одну сторону порога; фиксированный порог на выборку
ошибается в обе стороны (`test_a_wide_gap_on_a_small_sample_decides`,
`test_a_narrow_gap_on_a_large_sample_stays_undecided`).

**ИНТЕРВАЛ НАРУЖУ НЕ ВЫВОДИТСЯ.** `wilson_interval` вызывается тестами и
соседними функциями этого модуля; ни один тип результата словаря его не несёт
(`contracts/calcs.py`, докстринг модуля).

**«Мало данных» — число, а не стена.** Когда знак не определён, считается,
сколько наблюдений при ТОЙ ЖЕ доле хватит, чтобы он определился
(`observations_needed`).
"""

from __future__ import annotations

from collections.abc import Callable
from math import sqrt
from statistics import NormalDist

from harness.analysis.tools.pot_odds import required_equity
from harness.contracts import (
    CalcName,
    DefenseResult,
    Measurement,
    ThresholdOutcome,
    ThresholdSide,
)

__all__ = [
    "CONFIDENCE",
    "MAX_OBSERVATIONS",
    "compare_to_reference",
    "compare_to_threshold",
    "defense_from_bet",
    "observations_needed",
    "observations_needed_against",
    "threshold_from_bet",
    "wilson_interval",
]

# Уровень доверия интервала. Двусторонний, отсюда квантиль на `1 - (1-CONFIDENCE)/2`.
CONFIDENCE = 0.95
_Z = NormalDist().inv_cdf(1.0 - (1.0 - CONFIDENCE) / 2.0)

# Потолок поиска недостающей выборки. Выше него ответ не число наблюдений, а
# «столько не собрать»: при доле, отстоящей от порога на доли процента, требуемая
# выборка растёт быстрее любого правдоподобного числа рук
# (`test_a_share_next_to_the_threshold_needs_more_than_the_ceiling`).
MAX_OBSERVATIONS = 100_000


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def wilson_interval(numerator: float, denominator: float) -> tuple[float, float] | None:
    """Интервал Уилсона для доли `numerator / denominator`; `None` при нулевом знаменателе.

    Числитель вещественный, а не целый, нарочно: поиск недостающей выборки
    (`observations_needed`) держит долю неизменной и растит знаменатель, отчего
    числитель перестаёт быть целым. Формула от этого не меняется — она
    определена для любого `0 <= numerator <= denominator`.
    """
    if denominator <= 0.0:
        return None
    z2 = _Z * _Z
    spread = denominator + z2
    center = (numerator + z2 / 2.0) / spread
    half = (
        _Z
        / spread
        * sqrt(numerator * (denominator - numerator) / denominator + z2 / 4.0)
    )
    return _clamp(center - half), _clamp(center + half)


def _outcome_vs_threshold(numerator: float, denominator: float, threshold: float) -> ThresholdOutcome:
    """Где лежит интервал доли относительно порога — строго выше, строго ниже, или накрывает."""
    interval = wilson_interval(numerator, denominator)
    if interval is None:
        return ThresholdOutcome.UNDECIDED
    low, high = interval
    if low > threshold:
        return ThresholdOutcome.ABOVE
    if high < threshold:
        return ThresholdOutcome.BELOW
    return ThresholdOutcome.UNDECIDED


def _smallest_deciding(decides: Callable[[int], bool]) -> int | None:
    """Наименьшее `n <= MAX_OBSERVATIONS`, на котором знак определился, или `None`.

    Двоичный поиск опирается на монотонность предиката по `n` при неизменной
    доле: с ростом знаменателя интервал Уилсона сужается вокруг доли и, однажды
    разойдясь с порогом, обратно его не накрывает. Что возвращённое число —
    первое такое, а на единицу меньшее ещё нет, закреплено
    `test_the_needed_sample_is_the_first_that_decides`.
    """
    if not decides(MAX_OBSERVATIONS):
        return None
    low, high = 1, MAX_OBSERVATIONS
    while low < high:
        middle = (low + high) // 2
        if decides(middle):
            high = middle
        else:
            low = middle + 1
    return low


def observations_needed(share: float, threshold: float) -> int | None:
    """Сколько наблюдений при доле `share` нужно, чтобы знак против порога определился.

    Доля держится неизменной, знаменатель растёт: числитель на пробном `n` —
    это `share * n`, а не округление до целого. Округление сделало бы ответ
    рваным (соседние `n` дают то одну долю, то другую), и найденное число
    зависело бы от того, куда легло округление, а не от данных.

    `None`, когда порога не хватит никакому `n` до `MAX_OBSERVATIONS`: доля,
    равная порогу, не расходится с ним никогда.
    """
    return _smallest_deciding(
        lambda n: _outcome_vs_threshold(share * n, float(n), threshold)
        is not ThresholdOutcome.UNDECIDED
    )


def compare_to_threshold(
    measurement: Measurement, threshold: float
) -> tuple[ThresholdOutcome, int | None]:
    """Измерение против ТОЧНОГО порога: вердикт и, если его нет, недостающая выборка.

    Точного порога — то есть такого, у которого своей неопределённости нет:
    арифметика от размера ставки данных не требует и интервалом не приходит.

    Недостающая выборка не считается при нулевом знаменателе: доли, которую
    можно было бы удержать, ещё не наблюдали
    (`test_nothing_observed_yields_no_needed_sample`).
    """
    outcome = _outcome_vs_threshold(
        float(measurement.numerator), float(measurement.denominator), threshold
    )
    if outcome is not ThresholdOutcome.UNDECIDED:
        return outcome, None
    share = measurement.share
    if share is None:
        return outcome, None
    return outcome, observations_needed(share, threshold)


def _difference_interval(
    numerator: float, denominator: float, reference: Measurement
) -> tuple[float, float] | None:
    """Интервал разности двух долей, собранный из двух интервалов Уилсона.

    Гибрид Ньюкомба: если у измеряемого интервал `(l1, u1)` вокруг доли `p1`, а у
    эталона `(l2, u2)` вокруг `p2`, то разность `p1 - p2` лежит в

        [ (p1-p2) - √((p1-l1)² + (u2-p2)²),  (p1-p2) + √((u1-p1)² + (p2-l2)²) ]

    Второй метод здесь не заводится: концы берутся у той же `wilson_interval`,
    что и одновыборочный вердикт.
    """
    own = wilson_interval(numerator, denominator)
    other = wilson_interval(float(reference.numerator), float(reference.denominator))
    if own is None or other is None:
        return None
    p1 = numerator / denominator
    p2 = reference.numerator / reference.denominator
    low1, high1 = own
    low2, high2 = other
    delta = p1 - p2
    return (
        delta - sqrt((p1 - low1) ** 2 + (high2 - p2) ** 2),
        delta + sqrt((high1 - p1) ** 2 + (p2 - low2) ** 2),
    )


def _outcome_vs_reference(
    numerator: float, denominator: float, reference: Measurement
) -> ThresholdOutcome:
    interval = _difference_interval(numerator, denominator, reference)
    if interval is None:
        return ThresholdOutcome.UNDECIDED
    low, high = interval
    if low > 0.0:
        return ThresholdOutcome.ABOVE
    if high < 0.0:
        return ThresholdOutcome.BELOW
    return ThresholdOutcome.UNDECIDED


def observations_needed_against(share: float, reference: Measurement) -> int | None:
    """Сколько наблюдений у измеряемого нужно, чтобы он разошёлся с ИЗМЕРЕННЫМ эталоном.

    Растёт только выборка измеряемого; эталон остаётся тем, что есть. Это и есть
    вопрос, который задают: «сколько ещё рук против него нужно», а не «сколько
    ещё поля». Из этого следует, что ответа может не быть и при доле, далёкой от
    эталонной: узкий эталон не сужается, и разность может не оторваться от нуля
    никаким числом наблюдений с той стороны
    (`test_a_thin_reference_can_leave_the_difference_undecided_forever`).
    """
    return _smallest_deciding(
        lambda n: _outcome_vs_reference(share * n, float(n), reference)
        is not ThresholdOutcome.UNDECIDED
    )


def compare_to_reference(
    measurement: Measurement, reference: Measurement
) -> tuple[ThresholdOutcome, int | None]:
    """Измерение против ИЗМЕРЕННОГО эталона: неопределённостей две, сравнивается разность.

    Порог, посчитанный по выборке поля, — не точка, и сравнение с ним как с
    точкой объявило бы расхождение там, где его нет
    (`test_a_measured_threshold_is_harder_to_beat_than_the_same_number_as_a_point`).
    """
    outcome = _outcome_vs_reference(
        float(measurement.numerator), float(measurement.denominator), reference
    )
    if outcome is not ThresholdOutcome.UNDECIDED:
        return outcome, None
    share = measurement.share
    if share is None:
        return outcome, None
    return outcome, observations_needed_against(share, reference)


def defense_from_bet(pot_before: int, bet: int) -> DefenseResult:
    """Требуемая частота защиты и её дополнение — от банка ДО ставки и самой ставки.

    Ставка `bet` в банк `pot_before` рискует `bet`, чтобы выиграть `pot_before`.
    При частоте защиты `f` она приносит `(1 - f) * pot_before - f * bet`; это
    выражение обращается в ноль при

        f = pot_before / (pot_before + bet)

    — величина `defend_frequency`. Равенство нулю проверяется подстановкой
    возвращённого числа (`test_the_defence_frequency_is_the_one_that_zeroes_a_bet`),
    а не утверждается докстрингом.

    `required_equity` — та же пара чисел, приведённая к виду
    `pot_odds.required_equity`: банк на решении там уже со ставкой внутри.

    Обе величины строго положительны: банк и ставка — фишки, и нулевая ставка
    делает частоту защиты единицей, то есть утверждением ни о чём.
    """
    if pot_before <= 0:
        raise ValueError(f"банк до ставки должен быть положительным, получено {pot_before}")
    if bet <= 0:
        raise ValueError(f"ставка должна быть положительной, получено {bet}")
    defend = pot_before / (pot_before + bet)
    return DefenseResult(
        calc=CalcName.DEFENSE_FREQUENCY,
        pot_before=pot_before,
        bet=bet,
        defend_frequency=defend,
        fold_frequency=1.0 - defend,
        required_equity=required_equity(bet, pot_before + bet),
    )


def threshold_from_bet(pot_before: int, bet: int, side: ThresholdSide) -> float:
    """Названная сторона арифметического порога — числом.

    Сторона приходит параметром, потому что две частоты дают в сумму единицу, и
    сравнение частоты сдачи с частотой защиты сравнивало бы её с дополнением к
    нужному числу (`contracts.calcs.ThresholdSide`).
    """
    requirement = defense_from_bet(pot_before, bet)
    return (
        requirement.defend_frequency
        if side is ThresholdSide.DEFEND
        else requirement.fold_frequency
    )
