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
    HandCategory,
    HandStrength,
    Line,
    PostflopLineDetail,
    StrengthClass,
)
from harness.explanation.agent_checks import (
    ASSUMPTION_MARK,
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


@pytest.mark.parametrize("text", ["9 BB", "22 раздачи", "9.5", "99%", "стек 22.5"])
def test_real_numbers_stay_numbers(text: str):
    assert numbers_in(strip_card_tokens(text, hero_class="99", board_ranks=("Q", "J", "6"))) != []


def test_a_pocket_pair_is_a_card_only_when_the_hero_holds_it():
    """Чисто цифровую пару (`22`…`99`) от суммы шаблоном не отличить (тех-спека
    §5.7). Решение: пара — карта, только если это класс руки героя в разборе."""
    assert numbers_in(strip_card_tokens("у вас 99", hero_class="99")) == []
    assert numbers_in(strip_card_tokens("у вас 99")) == [99.0]
    assert numbers_in(strip_card_tokens("у вас 99", hero_class="J9s")) == [99.0]


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


def test_a_range_sentence_citing_only_allowed_numbers_needs_no_mark():
    """Фраза о диапазоне, все числа которой из сводки (зона «предполагая»
    разбора), — цитата расчёта, а не допущение агента."""
    assert _check("В допущении о диапазоне 1.5% всех рук.", allowed=frozenset({1.5})) == []


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
    assert _check("Ваш c-бет на флопе.", labels=frozenset({"cbet"})) == []
    assert _check("Ваш с-бет на флопе.", labels=frozenset({"cbet_flop"})) == []


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


def _with_line() -> dict[str, object]:
    detail = PostflopLineDetail(
        hand=HandStrength(
            category=HandCategory.TOP_PAIR_WEAK_KICKER,
            strength=StrengthClass.MEDIUM,
            combination=Combination.PAIR,
            ranks=["J"],
            plays="hand",
            kicker=None,
        ),
        draw=None,
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
    body = [line for line in ctx.text.splitlines() if not line.startswith("Точка 1 (")]
    assert unsupported_numbers("\n".join(body), ctx.allowed) == []


def test_the_context_carries_the_hand_board_and_classifier_labels():
    ctx = agent_context(
        _result([_point(dp_index=0, ev_diff_bb=-1.4, detail=_with_line())]), _postflop_hand()
    )
    assert ctx.hero_class == "J9s"
    assert ctx.board_ranks == ("6", "J", "Q")
    assert {"top_pair_weak_kicker", "check_fold"} <= ctx.labels


def test_prose_registers_every_number_of_a_code_built_text():
    book = NumberBook()
    book.prose("Флоп: 6♠ J♦ Q♦ (банк 6.6): вы чек → CO бет 3.0, итог −2.6BB")
    assert {6.6, 3.0, -2.6} <= book.allowed
