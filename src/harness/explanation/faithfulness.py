"""Верность изложения: три проверки, которыми код держит текст модели.

EVALS, этаж 3, дословно: проверяется не «хорош ли текст», а верность —
(1) все числа текста существуют в выходе ядра, (2) вердикт текста не
противоречит вердикту ядра, (3) допущение зоны «предполагая» названо словами.
Архитектура делает это почти кодовой задачей: числа считает код, модель их
только излагает.

**Одни и те же функции работают в двух местах.** В проде их зовёт
`explanation.verdict_text` и отказывается отдать игроку текст, не прошедший
проверку; в eval-прогоне (`evals/verdict/checks.py`) — те же, по настоящей
модели. Вторая реализация «для evals» разошлась бы с первой на первой же
правке, и расходились бы ровно там, где проверка перестала бы что-то значить.

**Второй пункт кодом не проверяется, а обеспечивается.** Метку вердикта ставит
`verdict_label_for` по `ev_diff_bb` ядра, модель её не выбирает
(`contracts.explanation`); сравнивать текст с меткой по смыслу было бы работой
для второй модели. Eval сверяет метку с ядром механически — это ловит
рассинхрон порогов, а не «похвалу вместо упрёка».
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from harness.contracts import VerdictLabel

__all__ = [
    "ASSUMPTION_WORDS",
    "has_assumption_words",
    "numbers_in",
    "unsupported_numbers",
    "verdict_label_for",
]

# Паттерн числа — из плана дословно, плюс типографский минус (U+2212): изложение
# печатает именно его, и модель, копируя число из промпта, копирует и его.
_NUMBER_RE = re.compile(r"[-−]?\d+(?:[.,]\d+)?")

# Разряды, разделённые пробелом (обычным или неразрывным), — ОДНО число.
# Без этой склейки «31 250» читается как 31 и 250: оба «числа» находятся в
# любом разборе, и проверка пропускает выдуманную сумму именно там, где цена
# ошибки наибольшая (`test_thousands_are_read_as_one_number_not_two`).
_THOUSANDS_RE = re.compile(r"(?<=\d)[   ](?=\d{3}\b)")

# Точность сверки — 0.1 bb: та же, с которой изложение печатает числа
# (`presentation._bb_number`). Сверять точнее значило бы требовать от модели
# цифр, которых она не видела.
_PRECISION = 1

# Слова допущения (план дословно: «если », «предполагая», «допущени») плюс
# «по модели» — формулировка, уже утверждённая владельцем в изложении
# (`presentation._ASSUMING_MARKER`, «по модели диапазонов»).
ASSUMPTION_WORDS: tuple[str, ...] = ("если ", "предполаг", "допущени", "по модели")

# Пороги метки — из плана дословно. Живут здесь, а не в `analysis`: ядро считает
# цену, а раскладка цены на три слова — дело изложения.
_OK_FROM_BB = -0.1
_MISTAKE_BELOW_BB = -0.5


def numbers_in(text: str) -> list[float]:
    """Все числа текста в порядке появления, с учётом разрядов и запятой-дроби."""
    joined = _THOUSANDS_RE.sub("", text)
    return [
        float(match.group().replace(",", ".").replace("−", "-"))
        for match in _NUMBER_RE.finditer(joined)
    ]


def unsupported_numbers(text: str, allowed: Iterable[float]) -> list[float]:
    """Числа текста, которых нет среди посчитанных, — с округлением до 0.1.

    Разрешён и модуль каждого числа: знак в русской фразе часто несёт слово
    («теряет 1.2 bb»), и требовать минус означало бы штрафовать за грамотность.
    Обратное направление — плюс там, где расчёт дал минус, — этой проверкой не
    ловится и не должно (см. модульный докстринг).
    """
    permitted = {round(value, _PRECISION) for value in allowed}
    permitted |= {abs(value) for value in permitted}
    return [
        number
        for number in numbers_in(text)
        if round(number, _PRECISION) not in permitted
    ]


def verdict_label_for(ev_diff_bb: float) -> VerdictLabel:
    """Метка точки по цене расхождения — единственный источник метки в системе."""
    if ev_diff_bb >= _OK_FROM_BB:
        return "ok"
    if ev_diff_bb < _MISTAKE_BELOW_BB:
        return "mistake"
    return "marginal"


def has_assumption_words(text: str) -> bool:
    """Названо ли допущение словами. Регистр не важен — фраза бывает в начале."""
    lowered = text.lower()
    return any(word in lowered for word in ASSUMPTION_WORDS)
