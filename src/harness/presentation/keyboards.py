"""Клавиатуры под сообщениями игроку: тип кнопки и правило `callback_data`.

`messages.py` отвечает за текст, этот модуль — за то, что нажимается и что
приходит боту в ответ. Разделение по брифу задачи 17: одна ответственность на
файл, и `callback_data` каждой кнопки собирается в одном месте, а не
россыпью по конструкторам сообщений — иначе бот и клавиатура разойдутся в
формате префикса (`deep:`, `ranges:`, `escalate:`), и это всплывёт только на
хендлере, который его парсит.

Модуль знает про разметку Телеграма (кнопки, callback_data), но не про
Телеграм-API, не про БД и не про LLM — это по-прежнему чистые функции
(«правило единого голоса», спека §4): результат возвращается значением,
отправляет его позже `bot`.
"""

from __future__ import annotations

from pydantic import BaseModel


class Btn(BaseModel):
    """Одна инлайн-кнопка: подпись и то, что вернётся боту при нажатии."""

    text: str
    callback_data: str


def deep_dive_button(hand_no: str) -> Btn:
    """Кнопка под строкой скана — запускает полный разбор конкретной раздачи."""
    return Btn(text="разобрать", callback_data=f"deep:{hand_no}")


def verdict_buttons(hand_no: str) -> list[Btn]:
    """Три кнопки под вердиктом (SESSIONS_UX): диапазоны, полный разбор, возражение.

    «Не согласен» — не декорация: нажатие уходит в eval-датасет (`verdict_dispute`,
    задача 21), поэтому `callback_data` несёт `hand_no` уже здесь, а не
    достраивается хендлером бота.
    """
    return [
        Btn(text="🎯 Диапазоны", callback_data=f"ranges:{hand_no}"),
        Btn(text="🔍 Подробнее", callback_data=f"detail:{hand_no}"),
        Btn(text="✋ Не согласен", callback_data=f"disagree:{hand_no}"),
    ]


def escalation_buttons(field: str, options: list[str]) -> list[Btn]:
    """Варианты эскалации плюс постоянная кнопка ручного ввода — одним рядом.

    `field` в `callback_data` — так хендлер бота узнаёт, какое поле дозаполняет
    ответ игрока, не заглядывая в текст вопроса.
    """
    buttons = [Btn(text=option, callback_data=f"escalate:{field}:{option}") for option in options]
    buttons.append(Btn(text="ввести вручную", callback_data=f"escalate:{field}:manual"))
    return buttons


__all__ = ["Btn", "deep_dive_button", "escalation_buttons", "verdict_buttons"]
