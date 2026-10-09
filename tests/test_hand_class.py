"""Сила готовой руки (`analysis.tools.hand_class`): по тесту на строку таблицы §4.1.

Все руки синтетические. Спека — `docs/superpowers/specs/2026-10-09-postflop-line-design.md`.
"""

import pytest
from pydantic import ValidationError

from harness.analysis.tools.hand_class import TOP_PAIR_STRONG_KICKER_MIN, hand_strength
from harness.contracts import Combination, HandCategory, HandStrength, StrengthClass

_STRONG = StrengthClass.STRONG
_MEDIUM = StrengthClass.MEDIUM
_WEAK = StrengthClass.WEAK


def _hand(hero: str, board: str) -> HandStrength:
    return hand_strength(tuple(hero.split()), board.split())


# --- строки «сильная»: комбинация с картой героя ------------------------------------


def test_straight_flush_with_a_hero_card_is_strong():
    result = _hand("9h 8h", "7h 6h 5h")
    assert (result.category, result.strength) == (HandCategory.STRAIGHT_FLUSH, _STRONG)
    assert result.ranks == ["9"]
    assert (result.plays, result.kicker) == ("hand", None)


def test_quads_with_a_hero_card_are_strong():
    result = _hand("7h 7d", "7s 7c 2d")
    assert (result.category, result.strength) == (HandCategory.QUADS, _STRONG)
    assert result.ranks == ["7"]


def test_full_house_with_a_hero_card_is_strong():
    result = _hand("Kh 2c", "Kc Kd 7s 7d")
    assert (result.category, result.strength) == (HandCategory.FULL_HOUSE, _STRONG)
    # Тройка первой, пара второй.
    assert result.ranks == ["K", "7"]


def test_flush_with_three_suited_on_the_board_is_strong():
    result = _hand("Ah 2h", "Kh Th 6h")
    assert (result.category, result.strength) == (HandCategory.FLUSH, _STRONG)
    assert result.ranks == ["A"]


def test_straight_with_three_of_its_ranks_on_the_board_is_strong():
    result = _hand("9h 8h", "Ts Js Qs 2c 3d")
    assert (result.category, result.strength) == (HandCategory.STRAIGHT, _STRONG)
    assert result.ranks == ["Q"]


def test_the_wheel_is_named_by_its_five():
    result = _hand("Ah 2d", "3c 4s 5h")
    assert result.category is HandCategory.STRAIGHT
    assert result.ranks == ["5"]


# --- флеш и стрит слабые: четыре карты на борде ----------------------------------------


def test_flush_with_four_suited_on_the_board_is_medium():
    result = _hand("Ah 2d", "Kh Th 8h 6h 4c")
    assert (result.category, result.strength) == (HandCategory.WEAK_FLUSH, _MEDIUM)
    assert result.plays == "hand"


def test_straight_with_four_of_its_ranks_on_the_board_is_medium():
    result = _hand("Th 2d", "9c 8s 7d 6c Kd")
    assert (result.category, result.strength) == (HandCategory.WEAK_STRAIGHT, _MEDIUM)
    assert result.ranks == ["T"]


# --- сет, трипс, две пары -------------------------------------------------------------


def test_pocket_pair_with_a_board_card_is_a_set():
    result = _hand("Kh Ks", "Kc 7d 2s")
    assert (result.category, result.strength) == (HandCategory.SET, _STRONG)
    assert result.ranks == ["K"]


def test_one_hero_card_with_a_board_pair_is_trips():
    result = _hand("Kh 7h", "Kc Kd 2s")
    assert (result.category, result.strength) == (HandCategory.TRIPS, _STRONG)
    assert result.ranks == ["K"]


def test_both_hero_cards_paired_with_different_board_ranks_are_two_pair():
    result = _hand("Kh 7h", "Kc 7d 2s")
    assert (result.category, result.strength) == (HandCategory.TWO_PAIR, _STRONG)
    assert result.ranks == ["K", "7"]


# --- пары ------------------------------------------------------------------------------


def test_pocket_pair_above_the_top_board_rank_is_an_overpair():
    result = _hand("Ah Ad", "Kc 7d 2s")
    assert (result.category, result.strength) == (HandCategory.OVERPAIR, _STRONG)
    assert result.ranks == ["A"]


def test_overpair_on_a_paired_board():
    result = _hand("9h 9d", "8h 8s 7d")
    assert result.combination is Combination.TWO_PAIR
    assert (result.category, result.strength) == (HandCategory.OVERPAIR, _STRONG)
    assert result.ranks == ["9"]


def test_top_pair_with_a_kicker_at_the_threshold_is_strong():
    assert TOP_PAIR_STRONG_KICKER_MIN == "T"
    result = _hand("Kh Td", "Kc 7d 2s")
    assert (result.category, result.strength) == (HandCategory.TOP_PAIR_STRONG_KICKER, _STRONG)
    assert result.ranks == ["K"]


def test_top_pair_with_a_kicker_below_the_threshold_is_medium():
    result = _hand("Kh 9d", "Kc 7d 2s")
    assert (result.category, result.strength) == (HandCategory.TOP_PAIR_WEAK_KICKER, _MEDIUM)


def test_pair_with_the_second_board_rank_is_a_middle_pair():
    result = _hand("7h 9d", "Kc 7d 2s")
    assert (result.category, result.strength) == (HandCategory.MIDDLE_PAIR, _MEDIUM)
    assert result.ranks == ["7"]


def test_pocket_pair_between_the_top_and_second_rank_is_a_middle_pair():
    result = _hand("9h 9d", "Kc 7d 2s")
    assert (result.category, result.strength) == (HandCategory.MIDDLE_PAIR, _MEDIUM)


def test_pair_with_the_third_board_rank_is_a_weak_pair():
    result = _hand("2h 9d", "Kc 7d 2s")
    assert (result.category, result.strength) == (HandCategory.WEAK_PAIR, _WEAK)
    assert result.ranks == ["2"]


def test_pocket_pair_below_the_second_rank_is_a_weak_pair():
    result = _hand("5h 5d", "Kc 7d 2s")
    assert (result.category, result.strength) == (HandCategory.WEAK_PAIR, _WEAK)


def test_a_paired_board_counts_distinct_ranks():
    # Q Q 7 2: старшая — Q, вторая — 7, младшая (третья) — 2, хотя это четвёртая карта.
    assert _hand("7h 3d", "Qc Qd 7s 2h").category is HandCategory.MIDDLE_PAIR
    assert _hand("2c 3d", "Qc Qd 7s 2h").category is HandCategory.WEAK_PAIR


def test_two_pair_with_the_board_pair_is_judged_by_the_heros_pair():
    # Q Q 4 и четвёрка у героя: в комбинации две пары, но сила — у пары героя.
    # Различные ранги борда — Q и 4, и 4 здесь второй ранг: средняя пара по строке
    # таблицы §4.1 (заметка под таблицей называет этот случай «слабой парой» — см.
    # отчёт подзадачи).
    result = _hand("4h 3d", "Qh Qs 4d")
    assert result.combination is Combination.TWO_PAIR
    assert (result.category, result.strength) == (HandCategory.MIDDLE_PAIR, _MEDIUM)
    assert result.plays == "hand"


# --- на борде --------------------------------------------------------------------------


def test_pair_on_the_board_plays_the_heros_kicker():
    # Рука §2 на ривере: Q Q 7 5 4, пятёрка — карта героя.
    result = _hand("5c 3c", "Qh 4d 2d 7s Qd")
    assert result.combination is Combination.PAIR
    assert (result.category, result.strength) == (HandCategory.ON_BOARD, _WEAK)
    assert result.ranks == ["Q"]
    assert (result.plays, result.kicker) == ("kicker", "5")


def test_pair_on_the_board_with_an_ace_kicker_stays_weak():
    # Правило старшего кикера — только для трипса и двух пар на борде.
    result = _hand("Ah 3c", "Qh Qd 7s 5c 2d")
    assert (result.category, result.strength) == (HandCategory.ON_BOARD, _WEAK)
    assert result.kicker == "A"


def test_board_straight_plays_the_board():
    # Пятёрка героя повторяет пятёрку борда: стрит собран из одних карт борда.
    result = _hand("5h 2c", "9c 8s 7d 6c 5d")
    assert result.combination is Combination.STRAIGHT
    assert result.category is HandCategory.ON_BOARD
    assert (result.plays, result.kicker) == ("board", None)
    assert result.ranks == ["9"]


def test_a_flush_card_below_five_board_flush_cards_does_not_play():
    result = _hand("2h 3d", "Kh Th 8h 6h 4h")
    assert result.combination is Combination.FLUSH
    assert (result.category, result.strength) == (HandCategory.ON_BOARD, _WEAK)
    assert result.plays == "board"


def test_full_house_on_the_board_is_weak():
    result = _hand("7h 2c", "Kc Kd Ks 7s 7d")
    assert result.combination is Combination.FULL_HOUSE
    assert (result.category, result.strength) == (HandCategory.ON_BOARD, _WEAK)
    assert result.ranks == ["K", "7"]


def test_a_counterfeited_pair_is_on_the_board():
    # Четвёрка героя спарилась с бордом, но в лучшие пять карт пара не вошла:
    # Q Q 7 7 и кикер K.
    result = _hand("4h Kd", "Qh Qs 7d 7c 4c")
    assert result.combination is Combination.TWO_PAIR
    assert (result.category, result.strength) == (HandCategory.ON_BOARD, _WEAK)
    assert result.ranks == ["Q", "7"]
    assert result.kicker == "K"


def test_two_pair_on_the_board_with_the_top_kicker_is_medium():
    result = _hand("As 5d", "Qh Qs 7d 7c 2c")
    assert (result.category, result.strength) == (HandCategory.ON_BOARD, _MEDIUM)
    assert result.kicker == "A"


def test_two_pair_on_the_turn_board_with_the_top_kicker_is_medium():
    result = _hand("As 3d", "Qh Qs 7d 7c")
    assert (result.category, result.strength) == (HandCategory.ON_BOARD, _MEDIUM)


def test_two_pair_on_the_board_with_a_lower_kicker_is_weak():
    result = _hand("Ks 3d", "Qh Qs 7d 7c 2c")
    assert (result.category, result.strength) == (HandCategory.ON_BOARD, _WEAK)
    assert result.kicker == "K"


def test_trips_on_the_board_with_the_top_kicker_is_medium():
    result = _hand("Ah Kd", "Qh Qs Qd 2c")
    assert result.combination is Combination.TRIPS
    assert (result.category, result.strength) == (HandCategory.ON_BOARD, _MEDIUM)
    assert result.kicker == "A"


def test_with_an_ace_on_the_board_the_top_possible_kicker_is_the_king():
    result = _hand("Kh 3d", "Qh Qs Qd Ac 2c")
    assert (result.category, result.strength) == (HandCategory.ON_BOARD, _MEDIUM)
    assert result.kicker == "K"


def test_trips_on_the_board_when_the_hero_card_does_not_play():
    # Кикеры A и K — на борде: карта героя не играет.
    result = _hand("Jh 3d", "Qh Qs Qd Ac Kc")
    assert (result.category, result.strength) == (HandCategory.ON_BOARD, _WEAK)
    assert (result.plays, result.kicker) == ("board", None)


# --- без пары ----------------------------------------------------------------------------


def test_no_pair_is_named_by_the_heros_top_card():
    # Рука §2 на флопе: «старшая 5, без пары».
    result = _hand("5c 3c", "Qh 4d 2d")
    assert result.combination is Combination.HIGH_CARD
    assert (result.category, result.strength) == (HandCategory.NO_PAIR, _WEAK)
    assert result.ranks == ["5"]
    assert (result.plays, result.kicker) == ("hand", None)


# --- вход и контракт ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hero", "board"),
    [
        (("Ah",), ["Kc", "7d", "2s"]),
        (("Ah", "Kd"), ["Kc", "7d"]),
        (("Ah", "Kd"), ["Kc", "7d", "2s", "3s", "4s", "5s"]),
        (("Ah", "Kd"), ["Kc", "7d", "Ah"]),
        (("Ah", "Kd"), ["Kc", "7d", "1s"]),
    ],
)
def test_bad_input_is_refused(hero, board):
    with pytest.raises(ValueError):
        hand_strength(hero, board)


def test_only_a_hand_on_the_board_plays_a_kicker_or_the_board():
    base = {
        "strength": _WEAK,
        "combination": Combination.PAIR,
        "ranks": ["Q"],
    }
    with pytest.raises(ValidationError):
        HandStrength(category=HandCategory.ON_BOARD, plays="hand", kicker=None, **base)
    with pytest.raises(ValidationError):
        HandStrength(category=HandCategory.WEAK_PAIR, plays="board", kicker=None, **base)
    with pytest.raises(ValidationError):
        HandStrength(category=HandCategory.ON_BOARD, plays="kicker", kicker=None, **base)
    with pytest.raises(ValidationError):
        HandStrength(category=HandCategory.ON_BOARD, plays="board", kicker="5", **base)
