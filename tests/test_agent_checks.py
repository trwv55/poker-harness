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
    Combination,
    Draw,
    DrawKind,
    HandCategory,
    HandStrength,
    Line,
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


def test_a_pocket_pair_is_a_card_only_when_the_hero_holds_it():
    """Чисто цифровую пару (`22`…`99`) от суммы шаблоном не отличить (тех-спека
    §5.7). Решение: пара — карта, только если это класс руки героя в разборе."""
    assert numbers_in(strip_card_tokens("у вас 99", hero_class="99")) == []
    assert numbers_in(strip_card_tokens("у вас 99")) == [99.0]
    assert numbers_in(strip_card_tokens("у вас 99", hero_class="J9s")) == [99.0]


@pytest.mark.parametrize("text", ["99 BB", "у вас 99 bb", "99%", "99 раздач", "99 фишек"])
def test_a_hero_pair_followed_by_a_unit_stays_a_number(text: str):
    """Пара героя — карта, но та же пара с единицей суммы — сумма: «99 BB»."""
    assert numbers_in(strip_card_tokens(text, hero_class="99")) == [99.0]


def test_a_hero_pair_amount_with_a_unit_is_checked_as_a_number():
    assert any("числа" in v for v in _check("Стек 22 BB.", hero_class="22"))
    assert _check("У вас 22.", hero_class="22") == []


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
        for v in _check("В допущении о диапазоне 1.5% всех рук.", allowed=frozenset({1.5}))
    )
    assert any(
        ASSUMPTION_MARK in v
        for v in _check("Его диапазон очень широкий, банк 6.6.", allowed=frozenset({6.6}))
    )


def test_a_decimal_point_does_not_split_a_sentence():
    assert _check(f"Диапазон на 6.6 банка {ASSUMPTION_MARK}.", allowed=frozenset({6.6})) == []


# --- лучшая линия на флопе и тёрне ---------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "На флопе лучше было чекнуть.",
        "На тёрне вам стоило поставить.",
        "На терне правильно было бы сдаться.",
    ],
)
def test_the_best_line_on_flop_or_turn_is_refused(text: str):
    assert any("лучшая линия" in v for v in _check(text))


@pytest.mark.parametrize(
    "text", ["Это ничего не стоило на флопе.", "На ривере лучше было коллировать."]
)
def test_what_is_not_a_best_line_verdict_on_flop_or_turn_passes(text: str):
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
    """Тот же инвариант, что у выжимок (`digest.py`): число в сводке — разрешено.

    Кроме заголовка точки («Точка 1 (dp_index 0)»): там идентификаторы, а не
    величины, и в реестр они не идут сознательно — то же исключение, что в
    `test_verdict_text.test_the_digest_registers_every_number_it_prints`."""
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
