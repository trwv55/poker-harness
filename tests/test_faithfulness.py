"""Верность изложения (EVALS этаж 3) — проверками кода, а не доверием.

Здесь тестируется КОД проверок, а не модель: сами проверки применяются и в
проде (`explanation.verdict_text` отказывается отдать игроку текст с выдуманным
числом), и в eval-прогоне по настоящей модели (`evals/verdict/checks.py`).
Модель в этом файле не участвует ни разу — это этаж 1, а не этаж 3.

Ловушка, ради которой тесты написаны парами: проверка, которая всегда говорит
«всё хорошо», проходит любой односторонний тест. Поэтому у каждой — пример
верного текста И пример неверного.
"""

from __future__ import annotations

from harness.explanation.faithfulness import (
    has_assumption_words,
    numbers_in,
    unsupported_numbers,
    verdict_label_for,
)

# --- числа в тексте ------------------------------------------------------------------


def test_numbers_are_pulled_out_with_sign_and_fraction():
    assert numbers_in("шов −1.2 bb против фолда 0.0") == [-1.2, 0.0]
    assert numbers_in("цена 1,5 bb") == [1.5]


def test_thousands_are_read_as_one_number_not_two():
    """«31 250» — одно число, а не 31 и 250: иначе проверка пропускает выдумку
    ровно в тех суммах, где она дороже всего."""
    assert numbers_in("стек 31 250 фишек") == [31250.0]
    assert numbers_in("стек 31 250 фишек") == numbers_in("стек 31250 фишек")


def test_a_hyphen_inside_an_identifier_is_not_a_minus_sign():
    """«SYN-3» — номер раздачи, а не число −3. Найдено первым живым прогоном
    evals: проверка объявляла выдумкой номер, который мы сами и дали модели."""
    assert numbers_in("В раздаче SYN-3 расхождение") == [3.0]
    assert numbers_in("цена −3 bb") == [-3.0]
    assert numbers_in("от -1.9 до -0.2") == [-1.9, -0.2]


def test_a_number_from_the_calculation_passes_and_an_invented_one_does_not():
    allowed = {-1.2, 3.4}
    assert unsupported_numbers("потеря −1.2 bb при банке 3.4 bb", allowed) == []
    assert unsupported_numbers("потеря 2.7 bb", allowed) == [2.7]


def test_the_sign_may_be_carried_by_the_word_instead_of_the_minus():
    """«теряет 1.2 bb» — то же число, что «−1.2 bb»: знак несёт слово.

    Поэтому разрешённым считается и модуль числа. Обратное (текст поставил плюс
    там, где расчёт дал минус) проверкой чисел не ловится и не должно: за это
    отвечает структурная метка вердикта, которую модель не выбирает.
    """
    assert unsupported_numbers("теряет 1.2 bb", {-1.2}) == []


def test_rounding_to_a_tenth_is_the_same_on_both_sides():
    """Сверка идёт с округлением до 0.1 — как печатает изложение (план дословно)."""
    assert unsupported_numbers("−1.2 bb", {-1.23}) == []
    assert unsupported_numbers("−1.3 bb", {-1.23}) == [-1.3]


# --- метка вердикта ------------------------------------------------------------------


def test_verdict_label_thresholds_are_the_core_ones():
    """Пороги из плана: ok при ev_diff ≥ −0.1, mistake при < −0.5, иначе marginal."""
    assert verdict_label_for(0.0) == "ok"
    assert verdict_label_for(-0.1) == "ok"
    assert verdict_label_for(-0.2) == "marginal"
    assert verdict_label_for(-0.5) == "marginal"
    assert verdict_label_for(-0.51) == "mistake"


# --- слова допущения -----------------------------------------------------------------


def test_assumption_words_are_found_in_any_of_their_spellings():
    assert has_assumption_words("если оппонент коллирует шире")
    assert has_assumption_words("Предполагая, что BB отвечает 20%")
    assert has_assumption_words("вывод опирается на допущение о диапазоне")
    assert has_assumption_words("по модели диапазонов это дороже")


def test_a_text_without_assumption_words_is_detected():
    assert not has_assumption_words("шов здесь дороже фолда на 1.2 bb")
