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
    HandCategory,
    HandStrength,
    Line,
    Overcards,
    PostflopLineDetail,
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
        # Арифметика порога — действие, привязанное к доле из расчёта.
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
    assert not any(
        ASSUMPTION_MARK in v for v in _check(text, allowed=allowed, calc_numbers=allowed)
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
    """Обходы повторного ревью: доля не из расчёта или вне условного придаточного."""
    numbers = frozenset({25.0})
    assert any("порог" in v for v in _check(text, allowed=numbers))
    if "в минусе при любом" in text or "почти всё" in text:
        assert any(
            "порог" in v for v in _check(text, allowed=numbers, calc_numbers=numbers)
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
    assert _check(_REJECTED_DRAFT, allowed=calc, calc_numbers=calc, **_TURN_KW) == []


def test_the_fixed_answer_passes_with_a_decided_threshold_only():
    calc = _turn_shaped_allowed()
    kw = {"allowed": calc, "calc_numbers": calc, **_TURN_KW}
    assert _check(_FIXED_ANSWER, threshold_decided=True, **kw) == []
    assert any("frequency_vs_threshold" in v for v in _check(_FIXED_ANSWER, **kw))


def test_a_real_violation_in_the_live_turn_shape_is_still_caught():
    draft = _REJECTED_DRAFT + " У него широкий диапазон, и выборка больше 30.5."
    calc = _turn_shaped_allowed()
    violations = _check(draft, allowed=calc, calc_numbers=calc, **_TURN_KW)
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
