"""Проверки ответа агента и сводка разбора — без базы и без сети.

Проверяется КОД вокруг модели: что разрешено и что отвергается. Механическая
проверка допущений, терминов и лучшей линии частичная по построению (спека
агента №1, §6): явные обороты ловятся здесь, остальное — судья (спека №2).
"""

from __future__ import annotations

import pytest

from harness.contracts import (
    POSTFLOP_LINE_DETAIL,
    ActionTag,
    Backdoor,
    Combination,
    Draw,
    DrawKind,
    FoldThreshold,
    HandCategory,
    HandStrength,
    Line,
    Overcards,
    PostflopLineDetail,
    Purpose,
    SizeTag,
    Street,
    StrengthClass,
)
from harness.explanation.agent_checks import (
    ASSUMPTION_MARK,
    STAT_LABEL,
    agent_context,
    check_agent_answer,
    strip_card_tokens,
)
from harness.explanation.digest import NumberBook
from harness.explanation.faithfulness import numbers_in, unsupported_numbers
from tests.test_hand_replay import _postflop_hand
from tests.test_verdict_text import _point, _result

_NONE: frozenset[float] = frozenset()


def _check(answer: str, **kw) -> list[str]:
    kw.setdefault("allowed", _NONE)
    kw.setdefault("labels", frozenset())
    kw.setdefault("leaks_called", False)
    kw.setdefault("calc_numbers", _NONE)
    kw.setdefault("frequency_numbers", _NONE)
    return check_agent_answer(answer, **kw)


# --- числа и обозначения карт ---------------------------------------------------


def test_a_number_from_the_allowed_set_passes_and_an_invented_one_does_not():
    assert (
        _check("Вы входите в банк в 100.0% раздач (2 из 2).", allowed=frozenset({100.0, 2.0})) == []
    )
    assert any(
        "числа" in v for v in _check("Вы входите в 43.0% раздач.", allowed=frozenset({100.0}))
    )


@pytest.mark.parametrize("text", ["A5s", "T9o", "KQ и AA", "J♥️9♥️", "10♠", "борд K-7-2"])
def test_card_tokens_are_not_numbers(text: str):
    assert numbers_in(strip_card_tokens(text)) == []


@pytest.mark.parametrize(
    "text", ["9 BB", "22 раздачи", "9.5", "99%", "стек 22.5", "стек 99.5", "банк 1.99"]
)
def test_real_numbers_stay_numbers(text: str):
    assert numbers_in(strip_card_tokens(text, hero_class="99", board_ranks=("Q", "J", "6"))) != []


@pytest.mark.parametrize("text", ["BB против UTG+1", "чарт опена UTG+2", "позиция UTG1"])
def test_a_position_name_with_a_digit_is_not_a_number(text: str):
    """Позиция — имя, а не число (решение контроллера R12): «UTG+1» проходит сверку
    при пустом наборе разрешённых."""
    assert numbers_in(strip_card_tokens(text)) == []
    assert _check(f"Чарт защиты: {text}.") == []


@pytest.mark.parametrize("text", ["Он поставил 1 BB.", "Ставка +1 BB.", "Это 1 из 5 раз."])
def test_a_digit_outside_a_position_name_stays_a_number(text: str):
    assert any("числа" in v for v in _check(text))


def test_a_bare_digit_pair_is_always_a_number():
    """Чисто цифровую пару (`22`…`99`) от суммы шаблоном не отличить (тех-спека
    §5.7). Решение: она число всегда, даже если герой держит эту пару — иначе
    выдуманная сумма, равная паре героя, прошла бы без проверки."""
    assert numbers_in(strip_card_tokens("у вас 99", hero_class="99")) == [99.0]
    assert numbers_in(strip_card_tokens("у вас 99")) == [99.0]


@pytest.mark.parametrize(
    "text",
    [
        "Стек 22.",
        "У вас 22 больших блайнда.",
        "22 блайнда в стеке.",
        "Ставка 22 бига.",
        "Пуш на 22.",
        "22 в стеке.",
    ],
)
def test_an_amount_equal_to_the_hero_pair_is_checked(text: str):
    assert any("числа" in v for v in _check(text, hero_class="22"))
    assert _check(text, hero_class="22", allowed=frozenset({22.0})) == []


@pytest.mark.parametrize(
    ("text", "board"),
    [
        ("на борде 8-8-J", ("8", "8", "J")),
        ("на борде 2-5-8", ("2", "5", "8")),
        ("на борде K-7-2", ()),
        ("у вас 98s", ()),
        ("у вас A5s", ()),
        ("у вас KQ", ()),
    ],
)
def test_unambiguous_card_tokens_are_still_stripped(text: str, board: tuple[str, ...]):
    assert numbers_in(strip_card_tokens(text, hero_class="22", board_ranks=board)) == []


@pytest.mark.parametrize("text", ["Стек 258 BB", "банк 852", "2580 фишек"])
def test_digits_without_separators_are_not_a_board(text: str):
    """Цифровой борд 2-5-8 не съедает сумму «258»: без разделителя цифры слипаются
    в борд, только если рядом буква."""
    assert numbers_in(strip_card_tokens(text, board_ranks=("2", "5", "8"))) != []


def test_a_board_with_a_letter_may_go_without_separators():
    assert numbers_in(strip_card_tokens("борд KQ7", board_ranks=("K", "Q", "7"))) == []


def test_a_ten_written_as_10_is_the_board_ten():
    assert numbers_in(strip_card_tokens("борд 10-7-2", board_ranks=("T", "7", "2"))) == []


def test_board_ranks_are_cards_only_when_they_are_the_board():
    """Без разбора — ранги борда с буквой (`K-7-2`); с разбором — только то,
    что лежит на борде: «7-2» на борде Q-J-6 — числа, и они не разрешены."""
    assert numbers_in(strip_card_tokens("на борде Q-J-6", board_ranks=("Q", "J", "6"))) == []
    assert numbers_in(strip_card_tokens("на борде K-7-2", board_ranks=("Q", "J", "6"))) == [
        7.0,
        2.0,
    ]
    assert numbers_in(strip_card_tokens("на борде K-7-2")) == []


# --- допущение ------------------------------------------------------------------


def test_a_range_claim_without_the_mark_is_refused_and_with_it_passes():
    assert any(ASSUMPTION_MARK in v for v in _check("Его диапазон тут широкий."))
    assert _check(f"Его диапазон тут широкий {ASSUMPTION_MARK}.") == []


def test_a_range_sentence_with_allowed_numbers_still_needs_the_mark():
    """В v1 диапазоны не считает ни один расчёт (спека №3), и разрешённое число
    метку не снимает: иначе её снимала бы приписка любого факта руки."""
    assert any(
        ASSUMPTION_MARK in v
        for v in _check("Его диапазон очень широкий, банк 6.6.", allowed=frozenset({6.6}))
    )


def test_a_decimal_point_does_not_split_a_sentence():
    assert _check(
        f"Его диапазон широкий при банке 6.6 {ASSUMPTION_MARK}.", allowed=frozenset({6.6})
    ) == []


# Двадцать формулировок ревью части A (2026-10-13): ни одна не прошла новую узкую
# проверку, четыре из них ловила и прежняя («любое предложение со словом диапазон»).
REVIEW_CLAIMS = [
    "Его диапазон в основном пары.",
    "В его диапазоне много Jx.",
    "Его диапазон — это в основном бродвеи.",
    "Его рендж тут капнут.",
    "BB здесь часто держит Jx.",
    "Соперник защитит широко.",
    "Такие игроки редко сдаются.",
    "Он почти всегда коллирует с любой парой.",
    "Соперник часто сбрасывает здесь.",
    "У BB тут много слабых пар.",
    "Он будет защищаться широко.",
    "Он часто фолдит на такие ставки.",
    "Скорее всего у оппонента тут Jx.",
    "Оппонент коллирует слишком часто.",
    "Противник сбросит почти всё.",
    "Он сбросит, если вы поставите.",
    "Если поставите, он сбросит почти всё.",
    "У него широкий диапазон, если смотреть на префлоп.",
    "У него 2 пары.",
    "Допущение о диапазоне оппонента — 1.5% всех рук.",
]

OTHER_CLAIMS = [
    "У него широкий диапазон.",
    "У него тут старшая пара.",
    "Он держит много слабых пар.",
    "Он заколлирует любую пару.",
    "Он сфолдит без пары.",
    "Его диапазон очень сильный.",
    "Его рендж состоит из бродвеев.",
    "Диапазон BB содержит много пар.",
    "Диапазон соперника полон слабых рук.",
    "Поле здесь редко защищается.",
    "Регуляры тут почти не блефуют.",
    "У него AK.",
    "Если у него широкий диапазон, ставка работает чаще.",
    "Если он заколлирует, на тёрне банк больше.",
    "Он фолдит 25%, и у него широкий диапазон.",
]


@pytest.mark.parametrize("text", REVIEW_CLAIMS + OTHER_CLAIMS)
def test_a_claim_about_what_the_opponent_holds_or_does_needs_the_mark(text: str):
    """Предложение о сопернике и о том, что он держит или сделает, — с меткой, даже
    с «если» и с разрешёнными числами (правка 1 и 2 контроллера)."""
    allowed = frozenset({1.5, 2.0, 25.0})
    assert any(ASSUMPTION_MARK in v for v in _check(text, allowed=allowed))
    marked = text.removesuffix(".") + f" {ASSUMPTION_MARK}."
    assert _check(marked, allowed=allowed) == []


@pytest.mark.parametrize(
    "text",
    [
        # Арифметика порога — порог из расчёта в условии или с «нужно» и инфинитивом.
        "Сопернику достаточно сбрасывать 25% диапазона, а защищаться ему нужно 75%.",
        "Ставка окупается на одних фолдах, если защищающийся сбрасывает чаще 25%.",
        "Ставка окупается, если он сбрасывает чаще 25%.",
        "Ставка окупается при 25% фолдов.",
        "Колл окупается от 25% эквити.",
        # Измерение с знаменателем.
        "Поле в BB сбрасывало на с-бет 30 из 50 (60%) — выше порога.",
        "Этот соперник сбрасывает на продолженную ставку 30 из 50.",
        # Метаречь и слово без соперника.
        "Это поле целиком, так что про его диапазон это мало говорит.",
        "Расчёта диапазонов здесь нет.",
        "Это поле, а не этот соперник.",
        "Ставка 2.5 BB в банк 5.0 BB.",
    ],
)
def test_calc_backed_statements_and_meta_speech_need_no_mark(text: str):
    allowed = frozenset({2.5, 5.0, 25.0, 30.0, 50.0, 60.0, 75.0})
    measured = frozenset({30.0, 50.0, 60.0})
    assert not any(
        ASSUMPTION_MARK in v
        for v in _check(text, allowed=allowed, calc_numbers=allowed, frequency_numbers=measured)
    )


@pytest.mark.parametrize(
    "text",
    [
        "Сопернику достаточно сбрасывать 25% диапазона, а защищаться ему нужно 75%.",
        "Этот соперник сбрасывает на продолженную ставку 30 из 50.",
    ],
)
def test_a_number_not_from_a_calculation_backs_nothing(text: str):
    """Опора — только число из выхода расчёта в разговоре: то же число из сводки
    или из слов игрока (разрешено, но не `calc_numbers`) действие не освобождает."""
    allowed = frozenset({25.0, 30.0, 50.0, 75.0})
    assert any(ASSUMPTION_MARK in v for v in _check(text, allowed=allowed))


@pytest.mark.parametrize(
    "text",
    [
        "Он сбросит на ставку 33.3% банка.",
        "Он сбросит почти всё, если вы поставите 25%.",
        "Соперник сфолдит без пары при 33.3% банка.",
        "Он будет коллировать с любой парой в 60% раз.",
    ],
)
def test_a_bet_size_or_a_hand_in_the_window_backs_nothing(text: str):
    """Обходы повторного ревью: доля банка, «поставите N%» и рука между действием и
    долей опорой не служат, даже если число выдал расчёт."""
    numbers = frozenset({25.0, 33.3, 60.0})
    assert any(
        ASSUMPTION_MARK in v for v in _check(text, allowed=numbers, calc_numbers=numbers)
    )


@pytest.mark.parametrize(
    "text", ["В диапазоне много Jx.", "Диапазон тут широкий.", "Он держит слабые руки."]
)
def test_a_range_word_needs_the_mark_by_default(text: str):
    """«Диапазон», «рендж», «держ…» — метка по умолчанию, и без слова о сопернике
    (правка 1 раунда 2)."""
    assert any(ASSUMPTION_MARK in v for v in _check(text))
    assert _check(text.removesuffix(".") + f" {ASSUMPTION_MARK}.") == []


# --- «в плюсе» / «в минусе» ------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Ставка на флопе в плюсе.",
        "Этот колл в минусе.",
        "Ставка окупается.",
        "Ставка в плюсе, если верить расчёту.",
    ],
)
def test_a_profit_conclusion_needs_a_decided_threshold(text: str):
    """Голый вывод «в плюсе» — только по вызову `frequency_vs_threshold` с выводом
    выше или ниже порога в разговоре; без него это мнение модели."""
    assert any("порог" in v for v in _check(text))
    assert _check(text, threshold_decided=True) == []


@pytest.mark.parametrize(
    "text",
    [
        "Ставка окупается на одних фолдах, если защищающийся сбрасывает чаще 25%.",
        "Ставка окупается на одних фолдах, когда защищающийся сбрасывает чаще 25%.",
        "Ставка окупается при 25% фолдов.",
        "Колл окупается от 19.4% эквити.",
        "Защищающийся должен сбрасывать чаще 25%, чтобы ставка окупалась.",
    ],
)
def test_profit_arithmetic_with_a_calc_share_is_allowed(text: str):
    """Арифметика с долей из выхода расчёта в условии — не вывод (правка 4, раунд 1);
    та же доля не из расчёта — уже вывод (раунд 2)."""
    numbers = frozenset({25.0, 19.4})
    assert not any(
        "порог" in v for v in _check(text, allowed=numbers, calc_numbers=numbers)
    )
    assert any("порог" in v for v in _check(text, allowed=numbers))


@pytest.mark.parametrize(
    "text",
    [
        "Ставка в плюсе от 25% и выше, тут так и есть.",
        "Колл в минусе при любом раскладе, у вас 25% эквити.",
        "Ставка в плюсе, если он сбрасывает почти всё, а это 25% рук.",
    ],
)
def test_a_bare_profit_conclusion_with_some_share_still_needs_a_threshold(text: str):
    """Обходы повторных ревью: доля вне условного придаточного или свободное
    придаточное-вывод («…, тут так и есть») — нарушение и при доле из расчёта."""
    numbers = frozenset({25.0})
    assert any("порог" in v for v in _check(text, allowed=numbers))
    assert any("порог" in v for v in _check(text, allowed=numbers, calc_numbers=numbers))


def test_a_threshold_in_the_main_clause_is_not_a_condition():
    """«Чаще/реже» — не условие (раунд 3): порог защиты в главном придаточном не
    снимает правило «в плюсе» и не служит опорой прогнозу."""
    defense = frozenset({33.3, 66.7, 25.0, 5.0, 2.5})
    kw = {"allowed": defense, "calc_numbers": defense}
    violations = _check("Ставка в плюсе: он сбрасывает чаще 33.3%.", **kw)
    assert any("порог" in v for v in violations)
    assert any(ASSUMPTION_MARK in v for v in violations)


@pytest.mark.parametrize(
    "text",
    ["Он точно сбросит чаще 33.3%, так что ставка хорошая.", "Он будет защищаться 76% раз."],
)
def test_only_a_measured_frequency_backs_an_opponent_action(text: str):
    """Порог и доля защиты из расчёта прогноз не освобождают (раунд 3)."""
    defense = frozenset({33.3, 66.7, 76.0, 24.0})
    assert any(
        ASSUMPTION_MARK in v for v in _check(text, allowed=defense, calc_numbers=defense)
    )


def test_a_measured_frequency_of_the_field_backs_its_action():
    numbers = frozenset({38.9, 21.0, 54.0})
    text = "Поле сбрасывает на с-бет 38.9% (21 из 54)."
    assert _check(
        text, allowed=numbers, calc_numbers=numbers, frequency_numbers=numbers,
        labels=frozenset({STAT_LABEL + "fold_to_cbet"}),
    ) == []


NARRATION = [
    "На флопе он чекнул, вы поставили 2.5 BB.",
    "У вас K-high без дро.",
    "Он поставил 2 BB, вы сбросили.",
    "BB заколлировал префлоп.",
    "Вы открылись с баттона, BB заколлировал.",
    "На тёрне соперник поставил 5 BB, и вы сфолдили.",
    "Вы заколлировали его рейз на флопе.",
    "Он сделал рейз до 8 BB, вы сбросили.",
    "Префлоп вы рейзнули, он ответил коллом.",
    "На ривере он пошёл олл-ин, вы заколлировали и проиграли.",
    "SB сбросил, BB защитился.",
    "Вы держите топ-пару на флопе.",
    "У вас пара дам, у соперника оказался сет.",
    "Он сбросил на ваш c-бет.",
    "Вы поставили c-бет, он сфолдил.",
]


@pytest.mark.parametrize("text", NARRATION)
def test_hand_narration_is_not_an_opponent_claim(text: str):
    """Пересказ руки — факт, не прогноз (раунд 3, решение (а)): прошедшее время и
    придаточное с «вы» метки не требуют; «вы держите» — о своей руке."""
    sizes = frozenset({2.0, 2.5, 5.0, 8.0})
    assert not any(ASSUMPTION_MARK in v for v in _check(text, allowed=sizes))


# Обходы освобождений пересказа (повторное ревью 3, решения контроллера раунда 4).
SUBJUNCTIVE = [
    "Он бы сбросил на такую ставку.",
    "Он явно сбросил бы на такую ставку.",
    "Он сделал бы рейз на тёрне.",
    "Скорее всего, он сфолдил бы любую слабую руку.",
    "Он сбросил бы, если бы вы поставили 5 BB.",
    "Он бы защитился почти всем.",
    "Он мог бы сбросить на такую ставку.",
    "Он не стал бы защищаться.",
]

HABITUAL = [
    "Он всегда коллировал такие ставки.",
    "Раньше этот соперник часто фолдил на с-бет.",
    "BB обычно защищался против такой ставки.",
    "Такие игроки на BB обычно сбрасывали на с-бет.",
    "Поле всегда коллировало такие размеры.",
]

HERO_AND_OPPONENT = [
    "Он сбросит и вы заберёте банк.",
    "Если вы поставите он почти наверняка сбросит.",
    "Он сбросит, и вы заберёте банк.",
    "Если вы поставите, он почти наверняка сбросит.",
    "Вы видели, что он держит пару.",
    "Против вас он будет блефовать.",
]


@pytest.mark.parametrize("text", SUBJUNCTIVE)
def test_the_subjunctive_is_a_prediction_not_a_fact_of_the_hand(text: str):
    """«Бы» в предложении снимает освобождение по прошедшему времени: сослагательное
    наклонение — прогноз (раунд 4, решение 1)."""
    assert any(ASSUMPTION_MARK in v for v in _check(text, allowed=frozenset({5.0})))
    marked = text.removesuffix(".") + f" {ASSUMPTION_MARK}."
    assert _check(marked, allowed=frozenset({5.0})) == []


@pytest.mark.parametrize("text", HABITUAL)
def test_a_habit_in_the_past_is_a_claim_not_a_fact_of_the_hand(text: str):
    """Наречие повторяемости («всегда», «обычно», «часто», «раньше»…) снимает
    освобождение по прошедшему времени: это обобщение, а не ход руки (раунд 4,
    решение 2)."""
    assert any(ASSUMPTION_MARK in v for v in _check(text))


@pytest.mark.parametrize(
    "marker",
    ["всегда", "обычно", "часто", "редко", "как правило", "раньше", "постоянно", "регулярно"],
)
def test_every_habit_marker_cancels_the_narration_exemption(marker: str):
    assert any(ASSUMPTION_MARK in v for v in _check(f"Он {marker} сбрасывал на с-бет."))
    assert not any(ASSUMPTION_MARK in v for v in _check("Он сбрасывал на с-бет."))


def test_never_in_the_past_is_a_habit_too():
    assert any(ASSUMPTION_MARK in v for v in _check("Он никогда не блефовал на ривере."))


def test_a_purpose_clause_is_not_the_subjunctive():
    """«Чтобы» содержит «бы», но это не частица: пересказ с «чтобы» остаётся
    пересказом."""
    text = "Он поставил 2 BB, чтобы вы сбросили."
    assert not any(ASSUMPTION_MARK in v for v in _check(text, allowed=frozenset({2.0})))


def test_any_measured_frequency_backs_an_opponent_action_whatever_its_subject():
    """Принятая граница (раунд 4): частота не привязана к подлежащему — измеренная
    частота поля служит опорой и фразе «он сбрасывает N%». Несовпадение ловит
    судья (спека №2)."""
    numbers = frozenset({38.9, 21.0, 54.0})
    assert not any(
        ASSUMPTION_MARK in v
        for v in _check(
            "Он сбрасывает 38.9%.", allowed=numbers, calc_numbers=numbers,
            frequency_numbers=numbers,
        )
    )


def test_a_measured_frequency_still_backs_a_habit_in_the_past():
    """Обобщение, за которым стоит измеренная частота из расчёта, — данные, а не
    допущение: опора вырезается раньше, чем смотрится время глагола."""
    numbers = frozenset({30.0, 50.0})
    text = "Поле обычно сбрасывало на продолженную ставку 30 из 50."
    assert not any(
        ASSUMPTION_MARK in v
        for v in _check(text, allowed=numbers, calc_numbers=numbers, frequency_numbers=numbers)
    )


@pytest.mark.parametrize("text", HERO_AND_OPPONENT)
def test_a_clause_with_an_opponent_subject_is_checked_despite_you(text: str):
    """«Вы» освобождает только придаточное о герое: придаточное, где подлежащее —
    соперник, проверяется, даже если в нём или рядом есть «вы», с запятой или без
    (раунд 4, решение 3)."""
    assert any(ASSUMPTION_MARK in v for v in _check(text))


def test_a_hero_clause_joined_by_and_with_a_past_opponent_action_is_narration():
    assert not any(ASSUMPTION_MARK in v for v in _check("Вы поставили и он сбросил."))


def test_a_noun_action_after_and_is_not_covered_by_a_past_verb_before_it():
    """Глагол прошедшего времени освобождает существительное («рейз») только в своей
    части: «и он сделает рейз» — прогноз."""
    assert any(ASSUMPTION_MARK in v for v in _check("Вы поставили и он сделает рейз."))


@pytest.mark.parametrize("text", ["При такой ставке он сбросит 33.3%.", "Он сбросит 33.3% при такой ставке."])
def test_a_bare_preposition_pri_is_not_a_condition(text: str):
    """«При» — условие арифметики порога только перед долей со словом порога («при
    24% фолдов»); «при такой ставке» порог опорой прогнозу не делает, а измеренная
    частота делает (раунд 4, решение 4)."""
    numbers = frozenset({33.3})
    assert any(ASSUMPTION_MARK in v for v in _check(text, allowed=numbers, calc_numbers=numbers))
    assert not any(
        ASSUMPTION_MARK in v
        for v in _check(text, allowed=numbers, calc_numbers=numbers, frequency_numbers=numbers)
    )


@pytest.mark.parametrize("text", ["Ставка окупается при 24% фолдов.", "Колл в плюсе при 19.4% эквити."])
def test_pri_before_a_share_with_a_threshold_word_is_still_a_condition(text: str):
    numbers = frozenset({24.0, 19.4})
    assert not any("порог" in v for v in _check(text, allowed=numbers, calc_numbers=numbers))


def test_a_profit_conclusion_with_a_bare_pri_needs_a_threshold():
    """«При такой доске» — не условие, даже если доля из расчёта стоит в том же
    придаточном."""
    numbers = frozenset({25.0})
    text = "Колл в плюсе при такой доске с 25% эквити."
    assert any("порог" in v for v in _check(text, allowed=numbers, calc_numbers=numbers))


# Живой ход на стенде (раунд 5): обе попытки отклонены за допущение, хотя это
# арифметика порога. «Он» — c-бет, «него» — порог; 24 — порог из расчёта защиты.
LIVE_THRESHOLD_MATH = [
    "Он окупается на одних фолдах, если защищающийся сбрасывает чаще 24%.",
    "Порог по фолдам у этой ставки 24%: выше него она окупается на одних фолдах.",
]


@pytest.mark.parametrize("text", LIVE_THRESHOLD_MATH)
def test_live_threshold_math_about_the_bet_passes(text: str):
    """Предложение о прибыльности ставки («окупается») — не утверждение о
    сопернике, а «фолдах» — не действие: при решённом пороге в разговоре и пороге
    24 из расчёта оно проходит целиком."""
    calc = frozenset({24.0, 76.0, 19.4})
    assert _check(text, allowed=calc, calc_numbers=calc, threshold_decided=True) == []


@pytest.mark.parametrize(
    "text",
    [
        "Порог по фолдам у него низкий.",
        "На фолдах он теряет немного.",
        "На одних фолдах он отдаёт мало.",
        "Без защиты у него банк уходит.",
    ],
)
def test_a_fold_or_defence_noun_after_a_preposition_is_not_an_opponent_action(text: str):
    """«Фолд», «сброс», «защита» существительным после предлога (на/по/от/при/без/
    для, между ними — не больше одного прилагательного) — не действие соперника
    (раунды 5–6). Что такие фразы говорят о сопернике другими словами, ловит судья
    (спека №2)."""
    assert not any(ASSUMPTION_MARK in v for v in _check(text))


@pytest.mark.parametrize(
    "text",
    [
        "Он сделал бы фолд на тёрне.",
        "Вы поставили и он сделает фолд.",
        "Соперник обычно делает фолд.",
        "Он сделает фолд на ривере.",
        "У него будет фолд на ривере.",
        "Он выбирает фолд почти всегда.",
        "Его защита тут слабая.",
        "Сбросов у него хватает.",
        "Ставка стоит на фолдах, а у него фолдов много.",
        "Он на ривере делает фолд.",
    ],
)
def test_a_fold_or_defence_noun_outside_a_preposition_is_an_action(text: str):
    """Без предлога перед ним (и вне части о прибыльности) существительное —
    действие, как «рейз» (повторное ревью 5, решение контроллера раунда 6)."""
    assert any(ASSUMPTION_MARK in v for v in _check(text))


@pytest.mark.parametrize(
    "text",
    [
        "Он будет на фолдах терять фишки.",
        "Обычно на фолдах он отдаёт банк.",
        "Он бы на фолдах отдал банк.",
    ],
)
def test_the_subjunctive_future_or_habit_cancels_the_noun_exemption(text: str):
    """«Бы», будущее («будет», «сделает») или наречие повторяемости в предложении —
    существительное не освобождается и после предлога (раунд 6)."""
    assert any(ASSUMPTION_MARK in v for v in _check(text))


def test_a_fold_noun_in_a_profit_clause_is_not_an_action():
    """В части о прибыльности ставки существительное освобождается и без предлога:
    «него» — порог, «фолдами» — источник прибыли ставки (раунд 6)."""
    assert _check("Выше него ставка окупается фолдами.", threshold_decided=True) == []


@pytest.mark.parametrize(
    "text", ["Он фолдит здесь.", "Соперник сфолдит.", "Он защищается.", "BB сбросит."]
)
def test_a_fold_or_defence_verb_is_still_an_opponent_action(text: str):
    assert any(ASSUMPTION_MARK in v for v in _check(text))


def test_a_hand_word_in_a_profit_clause_is_about_the_bet():
    """«Он» в придаточном о прибыльности — ставка: рука рядом не утверждение о
    сопернике, придаточное судит правило «в плюсе» (раунд 5)."""
    text = "Он окупается против любой пары."
    assert _check(text, threshold_decided=True) == []
    assert any("порог" in v for v in _check(text))


def test_an_opponent_verb_in_a_profit_clause_is_still_a_prediction():
    """Глагольная форма действия соперника — прогноз и в придаточном о прибыльности,
    без запятой тоже."""
    assert any(
        ASSUMPTION_MARK in v
        for v in _check("Ставка в плюсе он сбросит почти всё.", threshold_decided=True)
    )


def test_a_prediction_joined_to_a_profit_clause_by_and_is_still_checked():
    """Освобождается только часть о прибыльности ставки: «и он сбросит почти всё»
    в том же придаточном — прогноз."""
    assert any(
        ASSUMPTION_MARK in v
        for v in _check("Ставка окупается и он сбросит почти всё.", threshold_decided=True)
    )


@pytest.mark.parametrize(
    "text",
    [
        "Ставка окупается, если, по расчёту, соперник сбрасывает чаще 24%.",
        "Ставка окупается на одних фолдах, когда соперник, судя по выходу, сбрасывает чаще 24%.",
    ],
)
def test_threshold_math_after_a_profit_word_is_exempt_across_commas(text: str):
    """«X окупается…, если <подлежащее> сбрасывает чаще N%» с порогом из расчёта —
    арифметика порога, как бы ни делили его запятые (раунд 5)."""
    calc = frozenset({24.0, 76.0})
    assert not any(
        ASSUMPTION_MARK in v
        for v in _check(text, allowed=calc, calc_numbers=calc, threshold_decided=True)
    )


def test_across_commas_only_the_more_or_less_than_threshold_form_is_exempt():
    """Через запятые освобождается только форма «сбрасывает чаще/реже N%»:
    «…, если вы поставите, он сбросит 24%» — прогноз с порогом."""
    calc = frozenset({24.0})
    text = "Ставка в плюсе, если вы поставите, он сбросит 24%."
    assert any(
        ASSUMPTION_MARK in v
        for v in _check(text, allowed=calc, calc_numbers=calc, threshold_decided=True)
    )


@pytest.mark.parametrize(
    "text",
    [
        "Ставка в плюсе, если вы поставите, он сбросит чаще 24%.",
        "Если вы поставите, он сбросит чаще 24%, и ставка окупится.",
        "Ставка в плюсе, если вы поставите он сбросит чаще 24%.",
        "Ставка окупается, если флоп сухой, а соперник сбрасывает чаще 24%.",
        # Каждое ограничение отдельно: подлежащее-герой без глагола, глагол без героя,
        # не вставка между запятыми, два придаточных без союза.
        "Ставка окупается, если у вас на баттоне он сбрасывает чаще 24%.",
        "Ставка окупается, если флоп выйдет сухим и соперник сбрасывает чаще 24%.",
        "Ставка окупается, если, флоп сухой, соперник сбрасывает чаще 24%.",
        "Ставка окупается, если флоп сухой, соперник сбрасывает чаще 24%.",
    ],
)
def test_a_hero_subject_or_another_clause_breaks_the_threshold_condition(text: str):
    """Условие держит действие защищающегося, только если оно стоит в самом
    придаточном с «если»: подлежащее-герой или глагол другого придаточного между
    ними разрывают опору; через запятые пропускаются только вставки («по расчёту»,
    «судя по выходу»; повторное ревью 5, раунд 6)."""
    calc = frozenset({24.0})
    assert any(
        ASSUMPTION_MARK in v
        for v in _check(text, allowed=calc, calc_numbers=calc, threshold_decided=True)
    )


def test_a_condition_without_a_profit_word_does_not_reach_across_commas():
    """Без слова прибыльности условие из другого придаточного опорой не служит:
    «Если вы поставите 2.5 BB, он сбросит 24%.» — прогноз с порогом."""
    calc = frozenset({24.0, 2.5})
    assert any(
        ASSUMPTION_MARK in v
        for v in _check(
            "Если вы поставите 2.5 BB, он сбросит 24%.", allowed=calc, calc_numbers=calc
        )
    )


# --- лучшая линия на флопе и тёрне ---------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "На флопе лучше было чекнуть.",
        "На терне правильно было бы сдаться.",
        "На тёрне правильно было сдаться.",
        "Лучшая линия на флопе — чек.",
        "На тёрне правильная линия — пуш.",
        "По GTO на флопе здесь чек.",
    ],
)
def test_the_best_line_on_flop_or_turn_is_refused(text: str):
    assert any("лучшая линия" in v for v in _check(text))


@pytest.mark.parametrize(
    "text",
    [
        "Это ничего не стоило на флопе.",
        "На ривере лучше было коллировать.",
        "Стоило ли ставить на флопе, расчёт не судит.",
        "На тёрне вам стоило поставить.",
    ],
)
def test_phrases_outside_the_owner_best_line_list_are_not_checked(text: str):
    """Проверка держит список владельца (2026-10-13): «лучшая линия», «GTO»,
    «правильно было», «лучше было», «правильная линия» — по флопу и тёрну.
    «Стоило» в этот список не входит, поэтому механикой не ловится; это не
    утверждение, что вердикт «вам стоило поставить» допустим."""
    assert not any("лучшая линия" in v for v in _check(text))


# --- термины --------------------------------------------------------------------


def test_a_term_passes_only_with_its_classifier_label():
    assert any("c-бет" in v for v in _check("Ваш c-бет на флопе."))
    assert _check("Ваш c-бет на флопе.", labels=frozenset({"action:cbet"})) == []
    assert _check("Ваш с-бет на флопе.", labels=frozenset({STAT_LABEL + "cbet_flop"})) == []


def test_a_bare_label_without_its_kind_does_not_count():
    """Метки с пространством имён: голое `cbet` — не метка классификатора."""
    assert any("c-бет" in v for v in _check("Ваш c-бет на флопе.", labels=frozenset({"cbet"})))


def test_a_word_that_merely_contains_a_term_is_not_the_term():
    assert _check("Сетка вечера и пробег.") == []


@pytest.mark.parametrize(
    "text", ["Без пары и без флеш-дро.", "Нет флеш-дро.",
             "Это не флеш-дро.", "Ни пары, ни флеш-дро."]
)
def test_a_negated_term_is_not_the_term(text: str):
    assert not any("флеш-дро" in v for v in _check(text))


def test_a_term_negated_once_and_said_once_is_still_the_term():
    assert any("флеш-дро" in v for v in _check("Не флеш-дро, а флеш-дро на тёрне."))


# --- «лик» и «ошибка» -----------------------------------------------------------


def test_the_word_leak_needs_the_leaks_calculation():
    assert any("лик" in v for v in _check("Это ваш лик."))
    assert _check("Это ваш лик.", leaks_called=True) == []
    assert _check("Клик и ликвидность.") == []


def test_calling_a_decision_a_mistake_is_refused():
    assert any("ошибкой" in v for v in _check("Здесь вы ошиблись."))


def test_an_empty_answer_is_a_violation():
    assert _check("   ") == ["пустой ответ"]


# --- сводка разбора -------------------------------------------------------------


_TOP_PAIR = HandStrength(
    category=HandCategory.TOP_PAIR_WEAK_KICKER,
    strength=StrengthClass.MEDIUM,
    combination=Combination.PAIR,
    ranks=["J"],
    plays="hand",
    kicker=None,
)
_MADE_FLUSH = HandStrength(
    category=HandCategory.FLUSH,
    strength=StrengthClass.STRONG,
    combination=Combination.FLUSH,
    ranks=["A"],
    plays="hand",
    kicker=None,
)
_FLUSH_DRAW = Draw(
    kinds=[DrawKind.FLUSH],
    out_ranks=[],
    outs=["2d", "3d", "4d", "5d", "7d", "8d", "9d", "Td", "Kd"],
    unseen=47,
    hit_next=9 / 47,
    hit_by_river=0.35,
)


def _with_line(
    hand: HandStrength | None = _TOP_PAIR, draw: Draw | None = None
) -> dict[str, object]:
    detail = PostflopLineDetail(
        hand=hand,
        draw=draw,
        draw_missed=False,
        backdoors=[],
        overcards=None,
        line=Line(
            action=ActionTag.CHECK_FOLD, barrel=None, size_pct=None, size_tag=None, purpose=None
        ),
        fold_threshold=None,
        draw_call=None,
        showdown=None,
    )
    return {POSTFLOP_LINE_DETAIL: detail.model_dump(mode="json")}


def test_the_context_registers_every_number_it_prints():
    """Тот же инвариант, что у выжимок (`digest.py`): число сводки, которое
    останется числом в сверке ответа, — разрешено.

    Исключения два. Обозначения карт (`J♥9♥`, борд) сверка вырезает, и их ранги
    сводка не разрешает сознательно (`test_a_card_rank_of_the_hand_is_not_an_allowed_number`),
    поэтому сравнивается текст после `strip_card_tokens`. Заголовок точки
    («Точка 1 (dp_index 0)») несёт идентификаторы, а не величины, — то же
    исключение, что в `test_verdict_text.test_the_digest_registers_every_number_it_prints`."""
    ctx = agent_context(_result([_point(dp_index=0, ev_diff_bb=-1.4)]), _postflop_hand())
    body = "\n".join(line for line in ctx.text.splitlines() if not line.startswith("Точка 1 ("))
    cleaned = strip_card_tokens(body, hero_class=ctx.hero_class, board_ranks=ctx.board_ranks)
    assert unsupported_numbers(cleaned, ctx.allowed) == []


def test_a_card_rank_of_the_hand_is_not_an_allowed_number():
    """Ранг карты героя (J♥9♥) и борда — карта, а не число руки: «9 BB» и
    «8%» выдуманы. Борд с восьмёркой — копия фикстуры с другим флопом (в ней
    «6» — честное число руки: шесть игроков)."""
    hand = _postflop_hand()
    ctx = agent_context(_result([_point(dp_index=0, ev_diff_bb=-1.4)]), hand)
    assert 9.0 not in ctx.allowed
    kw = {"allowed": ctx.allowed, "hero_class": ctx.hero_class, "board_ranks": ctx.board_ranks}
    assert any("числа" in v for v in _check("У соперника 9 BB.", **kw))

    eight = hand.model_copy(
        update={"hand": hand.hand.model_copy(update={"boards": {Street.FLOP: ["8s", "Jd", "Qd"]}})}
    )
    ctx = agent_context(_result([_point(dp_index=0, ev_diff_bb=-1.4)]), eight)
    assert "8♠" in ctx.text and 8.0 not in ctx.allowed
    kw = {"allowed": ctx.allowed, "hero_class": ctx.hero_class, "board_ranks": ctx.board_ranks}
    assert any("числа" in v for v in _check("Соперник фолдит в 8% случаев.", **kw))


def test_the_context_carries_the_hand_board_and_classifier_labels():
    ctx = agent_context(
        _result([_point(dp_index=0, ev_diff_bb=-1.4, detail=_with_line())]), _postflop_hand()
    )
    assert ctx.hero_class == "J9s"
    assert ctx.board_ranks == ("6", "J", "Q")
    assert {"hand:top_pair_weak_kicker", "action:check_fold"} <= ctx.labels
    assert "Метки классификатора, префлоп:" in ctx.text
    assert "preflop" not in ctx.text


def test_a_flush_draw_term_needs_the_draw_and_not_a_made_flush():
    made = agent_context(
        _result([_point(dp_index=0, ev_diff_bb=-1.4, detail=_with_line(_MADE_FLUSH))]),
        _postflop_hand(),
    )
    drawing = agent_context(
        _result([_point(dp_index=0, ev_diff_bb=-1.4, detail=_with_line(_TOP_PAIR, _FLUSH_DRAW))]),
        _postflop_hand(),
    )
    answer = "У вас было флеш-дро."
    assert any("флеш-дро" in v for v in _check(answer, labels=made.labels))
    assert _check(answer, labels=drawing.labels) == []


def test_prose_registers_every_number_of_a_code_built_text():
    book = NumberBook()
    book.prose("Флоп: 6♠ J♦ Q♦ (банк 6.6): вы чек → CO бет 3.0, итог −2.6BB")
    assert {6.6, 3.0, -2.6} <= book.allowed


def test_an_unjudged_hand_prints_no_service_count():
    """«Судимых точек 0» модель пересказала игроку; без судимых точек сводка
    говорит это словами и без числа."""
    ctx = agent_context(
        _result([_point(dp_index=0, ev_diff_bb=0.0, detail=_with_line())], ranked=[]),
        _postflop_hand(),
    )
    assert "Судимых точек" not in ctx.text and "суммарная цена" not in ctx.text
    assert "Решений с вердиктом в этой раздаче нет." in ctx.text
    assert "Раздача TM99." in ctx.text


_BACKDOOR_FLUSH = Backdoor(kind="flush", variants=None, approx=0.04)
_OVERCARDS = Overcards(cards=["Ah", "Ks"], approx_by_river=0.12, approx_next=0.06)


def test_the_context_carries_the_hero_draws_backdoors_and_overcards():
    """Эквити героя из уже посчитанного разбора (`postflop_line`): дро с аутами и
    шансом, бэкдоры, оверкарты — числа зарегистрированы."""
    detail = _with_line(_TOP_PAIR, _FLUSH_DRAW)
    line = PostflopLineDetail.model_validate(detail[POSTFLOP_LINE_DETAIL])
    line = line.model_copy(update={"backdoors": [_BACKDOOR_FLUSH], "overcards": _OVERCARDS})
    ctx = agent_context(
        _result([_point(
            dp_index=0, ev_diff_bb=-1.4,
            detail={POSTFLOP_LINE_DETAIL: line.model_dump(mode="json")},
        )]),
        _postflop_hand(),
    )
    assert (
        "Рука героя, префлоп: дро — флеш, 9 аутов, собрать на следующей улице 19.1%, "
        "к риверу 35%; бэкдор — флеш ≈4%; оверкарты — A♥K♠, ≈12% к риверу, "
        "если пара будет лучшей." in ctx.text
    )
    kw = {"allowed": ctx.allowed, "hero_class": ctx.hero_class, "board_ranks": ctx.board_ranks}
    assert _check("У вас 9 аутов: 19.1% на тёрне, 35% к риверу, бэкдор ≈4%, оверкарты ≈12%.", **kw) == []


def test_without_draws_the_context_prints_no_hero_equity_line():
    ctx = agent_context(
        _result([_point(dp_index=0, ev_diff_bb=-1.4, detail=_with_line())]), _postflop_hand()
    )
    assert "Рука героя" not in ctx.text


def _bet_line(threshold: FoldThreshold | None, purpose: Purpose) -> dict[str, object]:
    detail = PostflopLineDetail(
        hand=None, draw=None, draw_missed=False, backdoors=[], overcards=None,
        line=Line(
            action=ActionTag.CBET, barrel=None, size_pct=0.32, size_tag=SizeTag.BLOCK,
            purpose=purpose,
        ),
        fold_threshold=threshold, draw_call=None, showdown=None,
    )
    return {POSTFLOP_LINE_DETAIL: detail.model_dump(mode="json")}


def _ctx(detail: dict[str, object]):
    return agent_context(
        _result([_point(dp_index=0, ev_diff_bb=-1.4, detail=detail)]), _postflop_hand()
    )


def test_the_context_names_the_purpose_and_the_fold_threshold_of_a_hero_bet():
    ctx = _ctx(_bet_line(FoldThreshold(bluff=0.24, semibluff=None, semibluff_free=False), Purpose.BLUFF))
    assert "Назначение, префлоп: блеф; ставка окупается, если соперник сбрасывает чаще 24%." in ctx.text
    assert 24.0 in ctx.allowed and ctx.threshold_numbers == frozenset({24.0})


def test_a_semibluff_names_both_thresholds_or_that_it_pays_without_folds():
    both = _ctx(_bet_line(FoldThreshold(bluff=0.24, semibluff=0.091, semibluff_free=False), Purpose.SEMIBLUFF))
    assert (
        "Назначение, префлоп: полублеф; порог фолдов как у чистого блефа 24%, "
        "с учётом аутов 9.1%." in both.text
    )
    assert both.threshold_numbers == frozenset({24.0, 9.1})
    free = _ctx(_bet_line(FoldThreshold(bluff=0.24, semibluff=0.0, semibluff_free=True), Purpose.SEMIBLUFF))
    assert "с учётом аутов ставка окупается и без фолдов." in free.text


def test_a_value_bet_names_its_purpose_without_a_threshold():
    ctx = _ctx(_bet_line(None, Purpose.VALUE))
    assert "Назначение, префлоп: вэлью." in ctx.text
    assert ctx.threshold_numbers == frozenset()


def test_the_purpose_words_are_the_words_of_the_hand_breakdown():
    """Копия словаря разбора (`explanation` не импортирует `presentation`)."""
    from harness.explanation.agent_checks import _PURPOSE_WORD as SUMMARY
    from harness.presentation.messages import _PURPOSE_WORD as BREAKDOWN

    assert SUMMARY == BREAKDOWN


def test_a_threshold_from_the_summary_backs_profit_arithmetic():
    """Порог в сводке — из ядра, не догадка: «окупается, если сбрасывает чаще 24%»
    с ним — арифметика порога (спека 2026-10-14, §7). Без `threshold_numbers` та
    же фраза — голый вывод без расчёта."""
    from harness.agent.run import _violations

    ctx = _ctx(_bet_line(FoldThreshold(bluff=0.24, semibluff=None, semibluff_free=False), Purpose.BLUFF))
    answer = "Ставка окупается, если соперник сбрасывает чаще 24%."
    assert _violations(answer, [], ctx, frozenset()) == []
    assert any(
        "в плюсе" in v
        for v in _check(answer, allowed=ctx.allowed, calc_numbers=frozenset(), labels=ctx.labels)
    )


_FREE = FoldThreshold(bluff=0.24, semibluff=0.0, semibluff_free=True)


def test_a_semibluff_that_pays_without_folds_may_be_retold_from_the_summary():
    """Признак ядра `semibluff_free` — факт сводки: «окупается без фолдов» модель
    может пересказать, если он в контексте; без него — голый вывод."""
    from harness.agent.run import _violations

    free = _ctx(_bet_line(_FREE, Purpose.SEMIBLUFF))
    assert free.profit_without_folds
    for answer in (
        "С учётом аутов ставка окупается и без фолдов.",
        "Ставка окупается без фолдов.",
        "Полублеф в плюсе и без фолдов.",
    ):
        assert _violations(answer, [], free, frozenset()) == [], answer
        assert _check(answer, profit_without_folds=True) == [], answer

    paid = _ctx(_bet_line(
        FoldThreshold(bluff=0.24, semibluff=0.091, semibluff_free=False), Purpose.SEMIBLUFF
    ))
    assert not paid.profit_without_folds
    answer = "С учётом аутов ставка окупается и без фолдов."
    assert any("в плюсе" in v for v in _violations(answer, [], paid, frozenset()))
    assert any("в плюсе" in v for v in _check(answer))


def test_the_without_folds_flag_frees_only_that_conclusion():
    """Флаг освобождает только «окупается / в плюсе (и) без фолдов»: другой вывод,
    отрицание и «в минусе» в том же предложении остаются выводами."""
    for answer in (
        "Ставка окупается.",
        "Ставка не окупается без фолдов.",
        "Ставка в минусе без фолдов.",
        "Ставка окупается и без фолдов, а на ривере в минусе.",
    ):
        assert any("в плюсе" in v for v in _check(answer, profit_without_folds=True)), answer


# --- регрессия живого хода (синтетика той же формы) ------------------------------


def _turn_shaped_allowed() -> frozenset[float]:
    """Выходы трёх расчётов, как в ходе-образце: защита от ставки 2.5 в банк 5.0,
    сдача поля в BB против порога от этой ставки, свой c-бет на баттоне. Размеры
    2.5 и 5.0 выход защиты печатает сам."""
    from harness.contracts import (
        CalcName,
        FrequencyResult,
        FrequencyStat,
        Measurement,
        Subject,
        ThresholdOutcome,
        Window,
    )
    from harness.explanation.question import agent_digest
    from tests.test_question_text import _defense, _threshold

    defense = _defense().model_copy(update={
        "pot_before": 500, "bet": 250, "pot_before_bb": 5.0, "bet_bb": 2.5,
        "defend_frequency": 2 / 3, "fold_frequency": 1 / 3, "required_equity": 0.25,
    })
    threshold = _threshold(ThresholdOutcome.ABOVE, None).model_copy(update={
        "subject": Subject.FIELD, "position": "BB", "threshold": 1 / 3,
        "threshold_source": "bet_size", "reference": None,
        "measurement": Measurement(numerator=30, denominator=50),
    })
    hero = FrequencyResult(
        calc=CalcName.HERO_FREQUENCY, subject=Subject.HERO, stat=FrequencyStat.CBET_FLOP,
        position="BTN", window=Window(), measurement=Measurement(numerator=5, denominator=7),
    )
    calc: frozenset[float] = frozenset()
    for result in (defense, threshold, hero):
        calc |= agent_digest(result).allowed
    return calc


# Измеренные частоты тех же расчётов: сдача поля 30 из 50 (60%), свой c-бет 5 из 7.
_TURN_MEASURED = frozenset({30.0, 50.0, 60.0, 5.0, 7.0, 71.4})


_TURN_KW = {
    "labels": frozenset({STAT_LABEL + "cbet_flop", STAT_LABEL + "fold_to_cbet"}),
    "board_ranks": ("7", "7", "Q"),
}

# Отклонённый черновик хода-образца, синтетика: другие карты, борд и размеры.
_REJECTED_DRAFT = (
    "Ставка 2.5 в банк 5.0: чтобы она не приносила прибыль на одних фолдах, сопернику "
    "достаточно сбрасывать 33.3% диапазона, а защищаться ему нужно 66.7%. Поле в BB "
    "сбрасывало на с-бет 30 из 50 (60%) — это выше порога 33.3%, и выборка больше 20. "
    "Но это поле целиком, а не конкретный BB (ника у нас нет), так что про его диапазон "
    "это мало говорит. На борде 7-7-Q у вас A♠5♠ — без пары в руке и без флеш-дро; "
    "с-бет здесь скорее ставка за фолд-эквити, а не за готовую руку (допущение). Свои "
    "с-беты на баттоне вы делали 5 из 7 (71.4%) — выборка маленькая, выводов из неё не делаю."
)

_FIXED_ANSWER = (
    "Для вашей ставки 2.5 в банк 5.0 она окупается на одних фолдах, если соперник "
    "сбрасывает чаще 33.3%. Поле в BB сбрасывало на с-бет 30 из 50 (60%) — выше порога, "
    "выборка достаточная, так что ставка в плюсе. Это поле, а не этот соперник. Свои "
    "с-беты на баттоне — 5 из 7 (71.4%), выборка мала (меньше 20 возможностей)."
)


def test_the_rejected_draft_of_the_live_turn_now_passes():
    """Все четыре отказа живого хода были ложными: «20» (теперь в выходе
    расчёта), две фразы со словом «диапазон» без утверждения о составе и
    «без флеш-дро»."""
    calc = _turn_shaped_allowed()
    assert _check(_REJECTED_DRAFT, allowed=calc, calc_numbers=calc, frequency_numbers=_TURN_MEASURED, **_TURN_KW) == []


def test_the_fixed_answer_passes_with_a_decided_threshold_only():
    calc = _turn_shaped_allowed()
    kw = {"allowed": calc, "calc_numbers": calc, "frequency_numbers": _TURN_MEASURED, **_TURN_KW}
    assert _check(_FIXED_ANSWER, threshold_decided=True, **kw) == []
    assert any("frequency_vs_threshold" in v for v in _check(_FIXED_ANSWER, **kw))


def test_a_real_violation_in_the_live_turn_shape_is_still_caught():
    draft = _REJECTED_DRAFT + " У него широкий диапазон, и выборка больше 30.5."
    calc = _turn_shaped_allowed()
    violations = _check(draft, allowed=calc, calc_numbers=calc, frequency_numbers=_TURN_MEASURED, **_TURN_KW)
    assert len(violations) == 2
    assert any("30.5" in v for v in violations)
    assert any(ASSUMPTION_MARK in v for v in violations)


def test_on_the_turn_the_hero_equity_line_has_no_river_odds():
    """Тёрн: шанс одной карты — «на следующей улице», без «к риверу»."""
    draw = _FLUSH_DRAW.model_copy(update={"hit_by_river": None})
    over = _OVERCARDS.model_copy(update={"approx_by_river": None})
    line = PostflopLineDetail.model_validate(_with_line(_TOP_PAIR, draw)[POSTFLOP_LINE_DETAIL])
    line = line.model_copy(update={"overcards": over})
    point = _point(
        dp_index=0, ev_diff_bb=-1.4, detail={POSTFLOP_LINE_DETAIL: line.model_dump(mode="json")}
    ).model_copy(update={"street": Street.TURN})
    ctx = agent_context(_result([point]), _postflop_hand())
    assert (
        "Рука героя, тёрн: дро — флеш, 9 аутов, собрать на следующей улице 19.1%; "
        "оверкарты — A♥K♠, ≈6% на следующей улице, если пара будет лучшей." in ctx.text
    )


# --- диапазон по чарту и эквити против суженного (спека среза диапазона, §6, §8) ---

_CHART = frozenset({38.2, 410.0})


def test_a_chart_range_sentence_needs_no_mark():
    """Стартовый диапазон по чарту — зона `strict`: пересказ со словом «чарт» и
    числом выхода `opponent_range` метки не требует; без «чарт» или без числа его
    выхода слово диапазона требует её, как раньше."""
    kw = {"allowed": _CHART, "calc_numbers": _CHART, "chart_numbers": _CHART}
    assert _check("Стартовый диапазон BB по чарту — 38.2% рук, 410 комбо.", **kw) == []
    assert any(ASSUMPTION_MARK in v for v in _check("Стартовый диапазон BB — 38.2% рук.", **kw))
    plain = {"allowed": _CHART, "calc_numbers": _CHART}
    assert any(
        ASSUMPTION_MARK in v
        for v in _check("Стартовый диапазон BB по чарту — 38.2% рук, 410 комбо.", **plain)
    )


def test_a_chart_fact_does_not_cover_a_prediction_in_the_same_sentence():
    kw = {"allowed": _CHART, "calc_numbers": _CHART, "chart_numbers": _CHART}
    assert any(
        ASSUMPTION_MARK in v
        for v in _check("По чарту у BB 38.2% рук, и он сбросит всё без пары.", **kw)
    )


def test_a_chart_word_with_a_number_not_from_the_chart_needs_the_mark():
    """Слово «чарт» и число не из выхода `opponent_range` — не пересказ чарта."""
    other = _CHART | {12.5}
    kw = {"allowed": other, "calc_numbers": other, "chart_numbers": _CHART}
    assert any(
        ASSUMPTION_MARK in v for v in _check("Диапазон BB по чарту — 12.5% рук.", **kw)
    )
    assert any(
        ASSUMPTION_MARK in v for v in _check("Диапазон BB по чарту 12.5% рук.", **kw)
    )


@pytest.mark.parametrize(
    "sentence",
    [
        "Его реальный диапазон уже чарта — 38.2% рук.",
        "Его диапазон уже чарта в 38.2% рук.",
        "Диапазон BB шире чарта — 38.2% рук.",
        "Диапазон BB по чарту сужен до 38.2% рук.",
        "Диапазон BB урезан против чарта — 38.2% рук.",
        "Его диапазон меньше чарта в 38.2% рук.",
        "Реальный диапазон BB против чарта — 38.2% рук.",
    ],
)
def test_a_narrowing_word_in_the_chart_clause_needs_the_mark(sentence):
    """Сравнение с чартом или сужение — утверждение о сужении, а не пересказ чарта."""
    kw = {"allowed": _CHART, "calc_numbers": _CHART, "chart_numbers": _CHART}
    assert any(ASSUMPTION_MARK in v for v in _check(sentence, **kw))


@pytest.mark.parametrize(
    "sentence",
    [
        "По чарту у BB 38.2% рук, он держит только сильные руки.",
        "По чарту у BB 38.2% рук, но здесь его диапазон намного уже.",
        "По чарту у BB 38.2% рук, а реальный рендж сужен до топа.",
        "По чарту у BB 38.2% рук — его диапазон намного уже.",
        "По чарту у BB 38.2% рук и он держит только сильные руки.",
        "По чарту всё иначе, но его диапазон здесь 38.2% рук.",
    ],
)
def test_the_chart_exemption_covers_only_the_clause_that_retells_the_chart(sentence):
    """Исключение «чарт» — только для придаточного-пересказа; «держит» и слово
    диапазона в другом придаточном требуют метки."""
    kw = {"allowed": _CHART, "calc_numbers": _CHART, "chart_numbers": _CHART}
    assert any(ASSUMPTION_MARK in v for v in _check(sentence, **kw))


def test_an_equity_against_a_narrowed_range_needs_the_mark():
    """Эквити против суженного — допущение даже без слова «диапазон»."""
    eq = frozenset({31.4, 31.8})
    kw = {"allowed": eq, "calc_numbers": eq, "assumed_numbers": eq}
    assert any("суженного" in v for v in _check("Против этих рук у вас 31.4%.", **kw))
    assert _check("Против этих рук у вас 31.4% (допущение).", **kw) == []
    assert _check(
        "Если BB держит только пары и дамы (допущение), эквити 31.4%.", **kw
    ) == []


def test_the_prompt_range_paragraph_passes_with_its_numbers_and_marks():
    """Третий абзац ответа о прибыльности (спека §8): стартовый диапазон по чарту —
    факт, сужение и эквити против суженного — с меткой; эквити без метки — нарушение."""
    chart = frozenset({38.2, 410.0})
    eq = frozenset({22.0})
    kw = {
        "allowed": chart | eq,
        "calc_numbers": chart | eq,
        "chart_numbers": chart,
        "assumed_numbers": eq,
        "board_ranks": ("8", "8", "J"),
    }
    paragraph = (
        "По чарту защиты BB против BTN стартовый диапазон — 38.2% рук, 410 комбо. "
        "После колла на 8-8-J у BB остаются восьмёрки, валеты, карманные пары и "
        "стрит-дро (допущение). Против этого диапазона у вас 22% (допущение)."
    )
    assert _check(paragraph, **kw) == []
    violations = _check("Против этого диапазона у вас 22%.", **kw)
    assert any("суженного" in v for v in violations)
    assert any(
        ASSUMPTION_MARK in v
        for v in _check("По чарту защиты у BB 38.2% рук, и он сбросит всё без пары.", **kw)
    )


def test_the_turn_checks_read_chart_and_assumed_numbers_from_history():
    from pydantic_ai.messages import ModelRequest, ToolReturnPart

    from harness.agent.run import _violations

    messages = [ModelRequest(parts=[
        ToolReturnPart(tool_name="opponent_range", content="…", tool_call_id="c1", metadata={
            "calc": "opponent_range", "allowed": [38.2, 410.0], "chart": [38.2, 410.0],
        }),
        ToolReturnPart(tool_name="equity_vs_range", content="…", tool_call_id="c2", metadata={
            "calc": "equity_vs_range", "allowed": [31.4], "assumed": [31.4],
        }),
    ])]
    assert _violations("Стартовый диапазон BB по чарту — 38.2% рук.", messages, None, frozenset()) == []
    assert _violations("У вас 31.4%.", messages, None, frozenset()) != []
