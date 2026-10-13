"""Выходы расчётов, которые видит модель агента (`agent_digest`), — без базы и сети.

У `/ask` выходы прежние (`question_digest`, `tests/test_question_text.py`): агентский
текст собран отдельно. Здесь — что агент видит иначе: сторона ставки в защите,
выборка словами, уровень данных, проценты без «.0» и ни одной служебной строки.
"""

from __future__ import annotations

import pytest

from harness.contracts import (
    CalcName,
    CalcResult,
    FrequencyStat,
    Measurement,
    Subject,
    ThresholdOutcome,
)
from harness.explanation.faithfulness import numbers_in, unsupported_numbers
from harness.explanation.question import MIN_STAT_OPPORTUNITIES, agent_digest, question_digest
from tests.test_question_text import ALL_RESULTS, _defense, _frequency, _threshold


def _agent_defense():
    """Синтетика: банк 4.0 BB, ставка 1.0 BB — сбрасывать 20%, защищаться 80%."""
    return _defense().model_copy(update={
        "pot_before": 400, "bet": 100, "pot_before_bb": 4.0, "bet_bb": 1.0,
        "defend_frequency": 0.8, "fold_frequency": 0.2, "required_equity": 1 / 6,
    })


@pytest.mark.parametrize("result", ALL_RESULTS, ids=lambda r: r.calc.value)
def test_the_agent_digest_registers_every_number_it_prints(result: CalcResult):
    digest = agent_digest(result)
    assert set(numbers_in(digest.text)) - digest.allowed == set()


@pytest.mark.parametrize("result", ALL_RESULTS, ids=lambda r: r.calc.value)
def test_the_agent_digest_has_no_service_lines(result: CalcResult):
    """Имя расчёта в скобках и служебные оговорки модели не нужны: она пересказала
    их игроку («данных игрока здесь нет вовсе»)."""
    text = agent_digest(result).text
    assert f"({result.calc.value})" not in text
    assert "Данных игрока здесь нет" not in text
    assert "Оценки у этой величины нет" not in text


def test_the_ask_digest_is_unchanged_by_the_agent_one():
    assert "(defense_frequency)" in question_digest(_defense()).text
    assert "Данных игрока здесь нет вовсе" in question_digest(_defense()).text


def test_the_defense_output_names_the_side_of_each_number():
    text = agent_digest(_agent_defense()).text
    assert (
        "Для ставящего: ставка окупается на одних фолдах, если защищающийся сбрасывает "
        "чаще 20%." in text
    )
    assert "Для защищающегося: защищаться 80%, эквити колла 16.7%." in text
    assert "Банк до ставки: 4.0 BB; ставка: 1.0 BB." in text


def test_whole_percents_go_without_a_trailing_zero():
    """24, а не 24.0; проверка чисел сравнивает значения, и «20%» в ответе
    проходит так же, как «20.0%»."""
    digest = agent_digest(_agent_defense())
    assert "20.0" not in digest.text and "80.0" not in digest.text
    assert unsupported_numbers("Соперник должен сбрасывать чаще 20%.", digest.allowed) == []
    assert unsupported_numbers("Соперник должен сбрасывать чаще 20.0%.", digest.allowed) == []


def test_a_small_sample_is_said_in_words_with_the_threshold_registered():
    """5 из 7: выборка мала. Порог печатается в строке и разрешён — модели не нужно
    брать «20» из промпта."""
    digest = agent_digest(_frequency())
    assert MIN_STAT_OPPORTUNITIES == 20
    assert "Выборка мала (меньше 20 возможностей)." in digest.text
    assert 20.0 in digest.allowed
    assert unsupported_numbers("Выборка меньше 20 возможностей.", digest.allowed) == []


def test_a_sample_at_the_threshold_is_enough():
    result = _frequency().model_copy(
        update={"measurement": Measurement(numerator=12, denominator=MIN_STAT_OPPORTUNITIES)}
    )
    text = agent_digest(result).text
    assert "Выборка достаточная." in text and "Выборка мала" not in text


def test_no_observations_get_no_sample_verdict():
    result = _frequency().model_copy(update={"measurement": Measurement(numerator=0, denominator=0)})
    text = agent_digest(result).text
    assert "наблюдений нет" in text and "Выборка" not in text


def test_the_threshold_output_names_the_sample_too():
    text = agent_digest(_threshold(ThresholdOutcome.ABOVE, None)).text
    assert "Выборка достаточная." in text  # 23 из 29
    assert "Порог: 61% (измеренная частота поля)." in text


@pytest.mark.parametrize(
    ("subject", "position", "nick", "level"),
    [
        (Subject.FIELD, "BB", None, "Уровень данных: поле в этой позиции (BB)."),
        (Subject.FIELD, None, None, "Уровень данных: поле в целом, все позиции."),
        (Subject.OPPONENT, None, "SynthNick", "Уровень данных: этот соперник (SynthNick), все позиции."),
        (Subject.OPPONENT, "CO", "SynthNick", "Уровень данных: этот соперник (SynthNick), позиция CO."),
        (Subject.HERO, "BTN", None, "Уровень данных: вы, позиция BTN."),
        (Subject.HERO, None, None, "Уровень данных: вы, все позиции."),
    ],
)
def test_every_statistic_names_its_level(
    subject: Subject, position: str | None, nick: str | None, level: str
):
    calc = CalcName.HERO_FREQUENCY if subject is Subject.HERO else CalcName.OPPONENT_FREQUENCY
    frequency = _frequency().model_copy(
        update={"calc": calc, "subject": subject, "position": position,
                "opponent_id": 5 if nick else None}
    )
    threshold = _threshold(ThresholdOutcome.ABOVE, None).model_copy(
        update={"subject": subject, "position": position, "opponent_id": 5 if nick else None}
    )
    for result in (frequency, threshold):
        assert level in agent_digest(result, opponent=nick).text


def test_digits_in_a_nick_are_allowed_numbers():
    result = _frequency().model_copy(
        update={"calc": CalcName.OPPONENT_FREQUENCY, "subject": Subject.OPPONENT,
                "opponent_id": 5, "stat": FrequencyStat.FOLD_TO_CBET}
    )
    digest = agent_digest(result, opponent="Synth77")
    assert 77.0 in digest.allowed
