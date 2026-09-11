"""Выжимка выхода расчёта для модели и проверка её ответа — третий вход в модель.

Те же два механизма, что у `verdict_text`, поставленные на другой вход: числа
идут в промпт через `NumberBook` и ровно ими же ограничен ответ, а не прошедший
проверку текст игроку не уходит. Разница одна, и она в том, ГДЕ берётся
множество разрешённых чисел: у разбора это выжимка `AnalysisResult`, здесь —
выход одного вызванного инструмента (`contracts.calcs.CalcResult`). Всё, что
модель посчитала сама или вспомнила, разрешённым не становится.

**Расчёт называет себя сам.** Подпись под ответом собирает `presentation` по
полю `CalcResult.calc`, а не по словам модели, поэтому назвать не тот
инструмент, которым посчитано, она не может.

**Глоссарий ниже — не словарь единого голоса.** Он описывает величины модели,
которая пишет прозу сама; слова игроку собирает `presentation` своим словарём.
Та же развилка и то же решение, что у `_SPOT_BRIEF` в `verdict_text`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from harness.contracts import (
    CalcName,
    CalcResult,
    CoverageResult,
    DefenseResult,
    FrequencyResult,
    FrequencyStat,
    LeaksResult,
    Measurement,
    ModelOutput,
    Subject,
    ThresholdOutcome,
    ThresholdResult,
    Window,
)
from harness.explanation.digest import NumberBook
from harness.explanation.faithfulness import error_words_in, unsupported_numbers
from harness.explanation.verdict_text import (
    Digest,
    UnfaithfulText,
    read_prompt,
)

__all__ = [
    "QuestionDraft",
    "check_answer",
    "question_digest",
    "question_prompt",
]

_PROMPT_PATH = Path(__file__).parent / "prompts" / "question.md"

_CALC_BRIEF: dict[CalcName, str] = {
    CalcName.HERO_FREQUENCY: "частота героя",
    CalcName.OPPONENT_FREQUENCY: "частота оппонента",
    CalcName.COVERAGE: "покрытие разбора и цена расхождений",
    CalcName.LEAKS: "типы расхождений с ценой",
    CalcName.DEFENSE_FREQUENCY: "требуемая частота защиты от размера ставки",
    CalcName.FREQUENCY_VS_THRESHOLD: "частота против порога",
}

# Подписи величин повторяют то, что считает `analysis/player_stats.py`: у
# каждой свой знаменатель, и он назван в подписи словами («из скольких»).
_STAT_BRIEF: dict[FrequencyStat, str] = {
    FrequencyStat.VPIP: "добровольный вход в банк (из раздач за столом)",
    FrequencyStat.PFR: "повышение до флопа (из раздач за столом)",
    FrequencyStat.RERAISE: "ре-рейз до флопа (из случаев, когда до него повысили)",
    FrequencyStat.FOLD_TO_CBET: "сдача на продолженную ставку (из случаев, когда она сделана)",
    FrequencyStat.CBET_FLOP: "продолженная ставка на флопе (из флопов в роли агрессора)",
    FrequencyStat.BARREL_TURN: "ставка на тёрне после своей ставки на флопе",
    FrequencyStat.BARREL_RIVER: "ставка на ривере после своей ставки на тёрне",
    FrequencyStat.SHOWDOWN: "доход до вскрытия (из сыгранных флопов)",
}

_SUBJECT_BRIEF: dict[Subject, str] = {
    Subject.HERO: "герой",
    Subject.OPPONENT: "названный оппонент",
    Subject.FIELD: "поле (все оппоненты героя в его раздачах)",
}

_OUTCOME_BRIEF: dict[ThresholdOutcome, str] = {
    ThresholdOutcome.ABOVE: "величина выше порога",
    ThresholdOutcome.BELOW: "величина ниже порога",
    ThresholdOutcome.UNDECIDED: "знак против порога не определён",
}


class QuestionDraft(ModelOutput, BaseModel):
    """Что модели разрешено вернуть: слова, и ничего кроме слов.

    Ни имени расчёта, ни чисел отдельным полем: подпись собирает код по выходу
    инструмента, а числа проверяются по нему же.
    """

    answer: str


def _window_line(window: Window, book: NumberBook) -> str:
    """Окно расчёта словами. Идентификатор сессии числом не печатается: он не
    величина, и регистрация сделала бы его разрешённым в ответе."""
    if window.session_id is not None:
        return "Окно: один вечер (текущая сессия)."
    if window.since is not None:
        return f"Окно: с {book.token(window.since.date().isoformat())}."
    return "Окно: вся история разборов этого игрока."


def _measurement_phrase(measurement: Measurement, book: NumberBook) -> str:
    """Величина со своим знаменателем; доля — только когда знаменатель ненулевой.

    «0 из 0» — не ноль процентов, а отсутствие наблюдений (`Measurement.share`),
    и печатать при нём долю значило бы назвать число, которого нет.
    """
    counted = (
        f"{book.count(measurement.numerator)} из {book.count(measurement.denominator)}"
    )
    share = measurement.share
    if share is None:
        return f"{counted} (наблюдений нет)"
    return f"{counted} ({book.pct(100.0 * share)}%)"


def _frequency_lines(result: FrequencyResult, book: NumberBook) -> list[str]:
    lines = [
        f"Подлежащее: {_SUBJECT_BRIEF[result.subject]}.",
        f"Величина: {_STAT_BRIEF[result.stat]} — {_measurement_phrase(result.measurement, book)}.",
    ]
    if result.position is not None:
        lines.append(f"Позиция: {result.position}.")
    lines.append(_window_line(result.window, book))
    lines.append("Оценки у этой величины нет: эталона для неё в расчёте не предусмотрено.")
    return lines


def _coverage_lines(result: CoverageResult, book: NumberBook) -> list[str]:
    filters = result.filter
    named = [
        ("улица", filters.street.value if filters.street is not None else None),
        ("спот", filters.spot.value if filters.spot is not None else None),
        ("позиция", filters.position),
    ]
    lines = [
        f"Судимых точек решения: {_measurement_phrase(result.judged, book)} попавших под фильтр.",
        f"Из судимых с посчитанной ценой: {_measurement_phrase(result.priced, book)}.",
        f"Сумма расхождений по точкам с ценой: {book.bb(result.loss_bb)} bb.",
        (
            "Сумма и покрытие — РАЗНЫЕ множества: складывать их и выдавать за цену "
            "всей игры нельзя."
        ),
    ]
    chosen = [f"{title}: {value}" for title, value in named if value is not None]
    lines.append("Фильтр: " + ("; ".join(chosen) if chosen else "без ограничений") + ".")
    lines.append(_window_line(filters.window, book))
    return lines


def _leaks_lines(result: LeaksResult, book: NumberBook) -> list[str]:
    lines = [
        f"Судимых точек решения: {_measurement_phrase(result.judged, book)}.",
        _window_line(result.window, book),
        "Список ранжирован по цене, поэтому в него попадают только точки с ценой.",
    ]
    if not result.leaks:
        lines.append("Повторяющихся расхождений среди судимых точек не нашлось.")
        return lines
    lines.append("Типы расхождений:")
    lines += [
        f"  {stat.rule.title}: {book.count(stat.count)} раз, {book.bb(stat.loss_bb)} bb"
        for stat in result.leaks
    ]
    return lines


def _defense_lines(result: DefenseResult, book: NumberBook) -> list[str]:
    return [
        f"Банк до ставки: {book.count(result.pot_before)}; ставка: {book.count(result.bet)}.",
        (
            f"Требуемая частота защиты: {book.pct(100.0 * result.defend_frequency)}%; "
            f"сдаваться допустимо в {book.pct(100.0 * result.fold_frequency)}% случаев."
        ),
        f"Требуемое эквити колла: {book.pct(100.0 * result.required_equity)}%.",
        "Данных игрока здесь нет вовсе: это арифметика от размера ставки.",
    ]


def _threshold_lines(result: ThresholdResult, book: NumberBook) -> list[str]:
    lines = [
        f"Подлежащее: {_SUBJECT_BRIEF[result.subject]}.",
        f"Величина: {_STAT_BRIEF[result.stat]} — {_measurement_phrase(result.measurement, book)}.",
    ]
    if result.position is not None:
        lines.append(f"Позиция: {result.position}.")
    source = (
        "арифметика от размера ставки"
        if result.threshold_source == "bet_size"
        else "измеренная частота поля"
    )
    lines.append(f"Порог: {book.pct(100.0 * result.threshold)}% ({source}).")
    if result.reference is not None:
        lines.append(
            f"Выборка порога: {_measurement_phrase(result.reference, book)} — порог "
            f"тоже измерен, а не известен точно."
        )
    lines.append(f"Вывод расчёта: {_OUTCOME_BRIEF[result.outcome]}.")
    if result.outcome is ThresholdOutcome.UNDECIDED:
        needed = result.observations_needed
        lines.append(
            "Утверждать нельзя: наблюдений для вывода нужно около "
            f"{book.count(needed)}."
            if needed is not None
            else "Утверждать нельзя: при такой доле не хватит никакого числа наблюдений."
        )
    lines.append(_window_line(result.window, book))
    return lines


def question_digest(result: CalcResult) -> Digest:
    """Выход одного расчёта для промпта: текст и числа, которые он содержит.

    Числа печатаются только через `NumberBook`, и они же становятся
    разрешёнными в ответе — одной записью, как в `verdict_digest`. Что мимо
    книги не напечатано ни одно число ни у одного из шести расчётов, держит
    `test_the_digest_registers_every_number_it_prints`.
    """
    book = NumberBook()
    lines = [f"Расчёт: {_CALC_BRIEF[result.calc]} ({result.calc.value})."]
    if isinstance(result, FrequencyResult):
        lines += _frequency_lines(result, book)
    elif isinstance(result, CoverageResult):
        lines += _coverage_lines(result, book)
    elif isinstance(result, LeaksResult):
        lines += _leaks_lines(result, book)
    elif isinstance(result, DefenseResult):
        lines += _defense_lines(result, book)
    else:
        lines += _threshold_lines(result, book)
    return Digest(text="\n".join(lines), allowed=book.allowed)


def question_prompt(question: str) -> str:
    """Промпт маршрутизации с подставленным вопросом игрока.

    Файл промпта закрыт политикой публикации, и его отсутствие — громкий отказ
    (`read_prompt`), а не пустое задание модели.
    """
    return read_prompt(_PROMPT_PATH).replace("{question}", question)


def check_answer(answer: str, allowed: frozenset[float]) -> None:
    """Проверить ответ модели; непрошедший — `UnfaithfulText`, и игроку он не уходит.

    Две проверки, обе кодовые:

    * каждое число ответа есть в выходе вызванного инструмента;
    * решения не названы ошибкой — расчёт судит решение против диапазона, а
      вопрос о частоте не судит его вовсе.
    """
    stripped = answer.strip()
    if not stripped:
        raise UnfaithfulText("модель вернула пустой ответ")
    invented = unsupported_numbers(stripped, allowed)
    if invented:
        raise UnfaithfulText(f"числа не из расчёта — {invented}")
    errors = error_words_in(stripped)
    if errors:
        raise UnfaithfulText(f"решение названо ошибкой — {errors}")
