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
    EquityVsRangeResult,
    FrequencyResult,
    FrequencyStat,
    LeaksResult,
    Measurement,
    ModelOutput,
    OpponentRangeResult,
    Street,
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
    "MIN_STAT_OPPORTUNITIES",
    "QuestionDraft",
    "agent_digest",
    "check_answer",
    "question_digest",
    "question_prompt",
    "size_bb",
]

_PROMPT_PATH = Path(__file__).parent / "prompts" / "question.md"

_CALC_BRIEF: dict[CalcName, str] = {
    CalcName.HERO_FREQUENCY: "частота героя",
    CalcName.OPPONENT_FREQUENCY: "частота оппонента",
    CalcName.COVERAGE: "покрытие разбора и цена расхождений",
    CalcName.LEAKS: "типы расхождений с ценой",
    CalcName.DEFENSE_FREQUENCY: "требуемая частота защиты от размера ставки",
    CalcName.FREQUENCY_VS_THRESHOLD: "частота против порога",
    CalcName.OPPONENT_RANGE: "стартовый диапазон соперника по чарту",
    CalcName.EQUITY_VS_RANGE: "эквити против диапазона, суженного агентом",
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
    FrequencyStat.FOLD_TO_TURN_BARREL: (
        "сдача на второй баррель (из случаев, когда после колла продолженной ставки "
        "агрессор поставил и на тёрне)"
    ),
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


def _measurement_phrase(
    measurement: Measurement, book: NumberBook, *, trim: bool = False
) -> str:
    """Величина со своим знаменателем; доля — только когда знаменатель ненулевой.

    «0 из 0» — не ноль процентов, а отсутствие наблюдений (`Measurement.share`),
    и печатать при нём долю значило бы назвать число, которого нет. `trim` —
    доля без «.0» (выходы агента).
    """
    counted = (
        f"{book.count(measurement.numerator)} из {book.count(measurement.denominator)}"
    )
    share = measurement.share
    if share is None:
        return f"{counted} (наблюдений нет)"
    return f"{counted} ({book.pct(100.0 * share, trim=trim)}%)"


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


def _coverage_lines(
    result: CoverageResult, book: NumberBook, *, trim: bool = False
) -> list[str]:
    filters = result.filter
    named = [
        ("улица", filters.street.value if filters.street is not None else None),
        ("спот", filters.spot.value if filters.spot is not None else None),
        ("позиция", filters.position),
    ]
    lines = [
        (
            f"Судимых точек решения: {_measurement_phrase(result.judged, book, trim=trim)} "
            "попавших под фильтр."
        ),
        (
            "Из судимых с посчитанной ценой: "
            f"{_measurement_phrase(result.priced, book, trim=trim)}."
        ),
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


def _leaks_lines(result: LeaksResult, book: NumberBook, *, trim: bool = False) -> list[str]:
    lines = [
        f"Судимых точек решения: {_measurement_phrase(result.judged, book, trim=trim)}.",
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


def size_bb(value: float) -> str:
    """Размер в BB, как его передал агент: до сотых — его масштаб в `calcs.routing`,
    — и хотя бы один знак после точки (`6.6`, `3.0`)."""
    return f"{value:.2f}".removesuffix("0")


def _defense_sizes(result: DefenseResult, book: NumberBook) -> str:
    if result.pot_before_bb is not None and result.bet_bb is not None:
        return (
            f"Банк до ставки: {book.prose(size_bb(result.pot_before_bb))} BB; "
            f"ставка: {book.prose(size_bb(result.bet_bb))} BB."
        )
    return f"Банк до ставки: {book.count(result.pot_before)}; ставка: {book.count(result.bet)}."


def _defense_lines(result: DefenseResult, book: NumberBook) -> list[str]:
    return [
        _defense_sizes(result, book),
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
    книги не напечатано ни одно число ни у одного из расчётов `/ask`, держит
    `test_the_digest_registers_every_number_it_prints`. У расчётов агента
    выжимки `/ask` нет — отказ (`test_ask_has_no_digest_for_agent_only_calculations`).
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
    elif isinstance(result, (OpponentRangeResult, EquityVsRangeResult)):
        # Ошибка в значении имени расчёта, а не в типе вызова: `ValueError`.
        raise ValueError(f"{result.calc.value} — расчёт агента, у /ask его нет")  # noqa: TRY004
    else:
        lines += _threshold_lines(result, book)
    return Digest(text="\n".join(lines), allowed=book.allowed)


# --- выходы для агента ------------------------------------------------------------

# Порог выборки агента: меньше стольких возможностей стата — выборка мала (правило
# промпта агента; SESSIONS_UX, «Частоты»). Одна константа на весь агент: выход
# расчёта печатает его в строке о выборке и регистрирует, чтобы модели не нужно
# было число из промпта (`test_a_small_sample_is_said_in_words_with_the_threshold_registered`).
MIN_STAT_OPPORTUNITIES = 20


def _sample_line(measurement: Measurement, book: NumberBook) -> list[str]:
    """Выборка словами; без наблюдений строки нет — «наблюдений нет» уже сказано."""
    if measurement.denominator == 0:
        return []
    if measurement.denominator < MIN_STAT_OPPORTUNITIES:
        return [f"Выборка мала (меньше {book.count(MIN_STAT_OPPORTUNITIES)} возможностей)."]
    return ["Выборка достаточная."]


def _level_line(
    subject: Subject, position: str | None, opponent: str | None, book: NumberBook
) -> str:
    """Уровень данных — по фильтрам, реально применённым к выборке: подлежащее и
    позиция (`test_every_statistic_names_its_level`). Цифры ника разрешены: модель
    вправе назвать соперника (`test_digits_in_a_nick_are_allowed_numbers`)."""
    where = f"позиция {position}" if position is not None else "все позиции"
    if subject is Subject.FIELD:
        level = f"поле в этой позиции ({position})" if position is not None else (
            "поле в целом, все позиции"
        )
    elif subject is Subject.OPPONENT:
        named = f" ({book.token(opponent)})" if opponent is not None else ""
        level = f"этот соперник{named}, {where}"
    else:
        level = f"вы, {where}"
    return f"Уровень данных: {level}."


def _agent_frequency_lines(
    result: FrequencyResult | ThresholdResult, opponent: str | None, book: NumberBook
) -> list[str]:
    phrase = _measurement_phrase(result.measurement, book, trim=True)
    return [
        _level_line(result.subject, result.position, opponent, book),
        f"Величина: {_STAT_BRIEF[result.stat]} — {phrase}.",
        *_sample_line(result.measurement, book),
    ]


def _agent_threshold_lines(
    result: ThresholdResult, opponent: str | None, book: NumberBook
) -> list[str]:
    lines = _agent_frequency_lines(result, opponent, book)
    source = (
        "арифметика от размера ставки"
        if result.threshold_source == "bet_size"
        else "измеренная частота поля"
    )
    lines.append(f"Порог: {book.pct(100.0 * result.threshold, trim=True)}% ({source}).")
    if result.reference is not None:
        lines.append(
            f"Выборка порога: {_measurement_phrase(result.reference, book, trim=True)} — "
            "порог тоже измерен, а не известен точно."
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
    return lines


def _agent_defense_lines(result: DefenseResult, book: NumberBook) -> list[str]:
    """Величины по сторонам ставки (`test_the_defense_output_names_the_side_of_each_number`).
    Кто ставил — герой или соперник — расчёт не знает, поэтому стороны названы
    ролью, а не именем."""
    sizes = _defense_sizes(result, book)
    fold = book.pct(100.0 * result.fold_frequency, trim=True)
    defend = book.pct(100.0 * result.defend_frequency, trim=True)
    equity = book.pct(100.0 * result.required_equity, trim=True)
    return [
        sizes,
        (
            "Для ставящего: ставка окупается на одних фолдах, если защищающийся сбрасывает "
            f"чаще {fold}%."
        ),
        f"Для защищающегося: защищаться {defend}%, эквити колла {equity}%.",
    ]


_STREET_WORD: dict[Street, str] = {
    Street.PREFLOP: "префлоп", Street.FLOP: "флоп", Street.TURN: "тёрн", Street.RIVER: "ривер",
}

_LINE_BRIEF: dict[str, str] = {
    "open_raise": "опен рейзом",
    "open_allin": "опен олл-ином",
    "limp": "лимп",
    "call": "колл против опена {opener}",
    "check": "чек после лимпа {opener}",
    "three_bet": "3-бет против опена {opener}",
    "allin": "олл-ин против опена {opener}",
}

# Карманная пара словами: «99» в тексте сверка чисел считала бы числом, а модель,
# пересказавшая «99», — выдумкой (решение контроллера R2, 2026-10-15).
_PAIR_WORD: dict[str, str] = {
    "A": "тузов", "K": "королей", "Q": "дам", "J": "валетов", "T": "десяток",
    "9": "девяток", "8": "восьмёрок", "7": "семёрок", "6": "шестёрок", "5": "пятёрок",
    "4": "четвёрок", "3": "троек", "2": "двоек",
}


def _depth_text(value: float, book: NumberBook) -> str:
    return book.count(int(value)) if float(value).is_integer() else book.bb(value)


def _range_text(text: str, book: NumberBook) -> str:
    """Диапазон по классам (`to_notation`): пара — словами («пара девяток»), класс с
    буквой — как есть; доля чарта — процентом в скобках при классе, и только она
    регистрируется. Цифры классов с буквой сверка вырезает (`strip_card_tokens`)
    (`test_pocket_pairs_are_printed_in_words_and_are_not_numbers`,
    `test_the_range_outputs_register_every_number_that_stays_a_number`)."""
    parts: list[str] = []
    for token in text.split(", "):
        cls, _, weight = token.partition(":")
        word = f"пара {_PAIR_WORD[cls[0]]}" if len(cls) == 2 and cls[0] == cls[1] else cls
        if weight:
            word += f" ({book.pct(100.0 * float(weight), trim=True)}%)"
        parts.append(word)
    return ", ".join(parts)


def _agent_range_lines(result: OpponentRangeResult, book: NumberBook) -> list[str]:
    chart = (
        f"чарт опена {result.chart_position}"
        if result.chart == "open"
        else f"чарт защиты {result.chart_position} против {result.vs_chart_position}"
    )
    source = (
        f"{chart}, {book.count(result.chart_seats)}-max, "
        f"{_depth_text(result.chart_depth_bb, book)} BB"
    )
    if result.chart_seats != result.table_seats:
        source += (
            f"; за столом мест — {book.count(result.table_seats)}: чарт "
            f"{book.count(result.chart_seats)}-max по позиции от конца (правило, не допущение)"
        )
    line = _LINE_BRIEF[result.line].format(opener=result.opener_position)
    lines = [
        f"Соперник: {result.opponent_position}, линия до флопа — {line}.",
        f"Источник: {source}.",
        f"Стартовый диапазон: {_range_text(result.range_text, book)}.",
        (
            f"Доля рук: {book.pct(100.0 * result.hands_share, trim=True)}%; комбо без ваших "
            f"карт и борда ({_STREET_WORD[result.street]}): "
            f"{book.count(round(result.live_combos))}."
        ),
    ]
    if result.absent:
        lines.append(_absent_line(result, book))
    return lines


_ABSENT_GROUP: dict[str, str] = {
    "pairs": "карманных пар",
    "suited_aces": "одномастных тузов",
    "offsuit_aces": "разномастных тузов",
    "broadways": "бродвеев без туза",
}


def _absent_line(result: OpponentRangeResult, book: NumberBook) -> str:
    """Группы, которых нет в колле (чеке) защиты, и чем их играет чарт — факт чарта
    (план 2026-10-17, п. 8). Чисел в строке нет: классы — запись (`_range_text`, пары
    словами), действие — словом (`test_the_absent_groups_line_names_the_chart_actions`)."""
    checked = result.line == "check"
    raised = "рейзит" if checked else "3-бетит"
    parts: list[str] = []
    for group in result.absent:
        verb = " или ".join(
            word for word, used in ((raised, group.raised), ("идёт олл-ин", group.all_in)) if used
        )
        name = _ABSENT_GROUP[group.group]
        if group.whole:
            parts.append(f"{name} — ни одной, ими {verb}")
        else:
            parts.append(f"{name}: {_range_text(', '.join(group.classes), book)} — ими {verb}")
    played = "чек" if checked else "колл"
    return (
        f"Чего нет в {played}-диапазоне по чарту {result.chart_position} против "
        f"{result.vs_chart_position} (факт чарта, не допущение): " + "; ".join(parts) + "."
    )


def _agent_equity_lines(result: EquityVsRangeResult, book: NumberBook) -> list[str]:
    """С добавленными по просьбе игрока классами доли от стартового нет — она может
    быть больше 100% (`test_the_added_classes_are_a_separate_line_without_the_share`)."""
    named = _range_text(", ".join(result.added), book)
    added = [f"Добавлено по вашей просьбе: {named} — по чарту их здесь нет."] if result.added else []
    return [
        f"Соперник: {result.opponent_position}, улица: {_STREET_WORD[result.street]}.",
        (
            f"Диапазон, суженный агентом: {_range_text(result.range_text, book)}. "
            "Диапазон сужен агентом — это допущение."
        ),
        (
            f"Комбо: {book.count(round(result.live_combos))}."
            if result.added
            else f"Комбо: {book.count(round(result.live_combos))} — "
            f"{book.pct(100.0 * result.share_of_start, trim=True)}% стартового диапазона."
        ),
        *added,
        f"Эквити вашей руки при вскрытии: {book.pct(100.0 * result.equity, trim=True)}%.",
    ]


def agent_digest(result: CalcResult, *, opponent: str | None = None) -> Digest:
    """Выход расчёта, который видит модель агента; у `/ask` — `question_digest`.

    Те же числа через ту же книгу (`test_the_agent_digest_registers_every_number_it_prints`),
    но без служебных строк — имени расчёта в скобках и оговорок «данных игрока
    нет», «оценки нет» (`test_the_agent_digest_has_no_service_lines`): модель
    пересказывала их игроку. Проценты — без «.0». У статистики — уровень данных
    и выборка словами; `opponent` — ник, по которому модель позвала инструмент.
    """
    book = NumberBook()
    lines = [f"Расчёт: {_CALC_BRIEF[result.calc]}."]
    if isinstance(result, FrequencyResult):
        lines += _agent_frequency_lines(result, opponent, book)
        lines.append(_window_line(result.window, book))
    elif isinstance(result, CoverageResult):
        lines += _coverage_lines(result, book, trim=True)
    elif isinstance(result, LeaksResult):
        lines += _leaks_lines(result, book, trim=True)
    elif isinstance(result, DefenseResult):
        lines += _agent_defense_lines(result, book)
    elif isinstance(result, OpponentRangeResult):
        lines += _agent_range_lines(result, book)
    elif isinstance(result, EquityVsRangeResult):
        lines += _agent_equity_lines(result, book)
    else:
        lines += _agent_threshold_lines(result, opponent, book)
        lines.append(_window_line(result.window, book))
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
