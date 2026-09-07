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

from collections.abc import Sequence

from pydantic import BaseModel


class Btn(BaseModel):
    """Одна инлайн-кнопка: подпись и то, что вернётся боту при нажатии."""

    text: str
    callback_data: str


# Подписи кнопок нижнего меню (SESSIONS_UX, раскладка дословно). Постоянная
# клавиатура Телеграма присылает боту сам ТЕКСТ нажатой кнопки, поэтому подпись
# — ещё и ключ разбора входящего сообщения (`bot/menus.py`). Держится здесь
# затем, чтобы клавиатура и разбор нажатия читали одну и ту же строку.
MENU_SESSIONS = "🎬 Сессии"
MENU_TOURNAMENT = "📁 Турнир (HH)"
MENU_LEAKS = "📊 Мои лики"
MENU_NOTES = "🃏 Заметки"
MENU_SETTINGS = "⚙️ Настройки"
MENU_HELP = "❓ Help"

MAIN_MENU: list[list[str]] = [
    [MENU_SESSIONS, MENU_TOURNAMENT],
    [MENU_LEAKS, MENU_NOTES],
    [MENU_SETTINGS, MENU_HELP],
]

# Префиксы `callback_data` всех инлайн-кнопок задачи 23. Константами, а не
# литералами по месту: строку собирает этот модуль, а разбирает `bot/handlers.py`,
# и разойтись они могут только молча — нажатие просто перестанет узнаваться.
SESSION_PREFIX = "session:"
NEW_SESSION_DATA = "newsession"
NOTE_ADD_PREFIX = "note:"
NOTE_EDIT_PREFIX = "noteedit:"
NOTE_COLOR_PREFIX = "notecolor:"
NOTE_COLOR_SET_PREFIX = "notecolorset:"
NOTE_DELETE_PREFIX = "notedel:"
SET_NICKNAME_DATA = "setnick"
RANGES_PREFIX = "ranges:"
DETAIL_PREFIX = "detail:"
DISAGREE_PREFIX = "disagree:"


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
        Btn(text="🎯 Диапазоны", callback_data=f"{RANGES_PREFIX}{hand_no}"),
        Btn(text="🔍 Подробнее", callback_data=f"{DETAIL_PREFIX}{hand_no}"),
        Btn(text="✋ Не согласен", callback_data=f"{DISAGREE_PREFIX}{hand_no}"),
    ]


# Сколько кнопок «заметка на оппонента» вешается под разбором. Ограничение не от
# Телеграма, а от смысла: заметка ценна скоростью записи В МОМЕНТ наблюдения
# (решение владельца 2026-09-04), и стол на девять ников превратил бы её в ещё
# один экран выбора. Больше пяти — уже список, а не один тап.
_MAX_NOTE_BUTTONS = 5


def note_buttons_for_hand(nicks: Sequence[str]) -> list[list[Btn]]:
    """Ряды кнопок «заметка на оппонента» под разбором раздачи.

    Путь «добавить заметку» начинается ИЗ РАЗБОРА с подставленным оппонентом
    (решение владельца 2026-09-04), поэтому ник едет прямо в `callback_data`, а
    не выбирается на отдельном экране. Ники приходят только со скрина: в HH они
    анонимизированы (спека §5.2), и вызывающий передаёт пустой список.

    По две кнопки в ряд — ник длинный, и в один ряд их влезает немного.
    """
    buttons = [
        Btn(text=f"🃏 {nick}", callback_data=f"{NOTE_ADD_PREFIX}{nick}")
        for nick in list(nicks)[:_MAX_NOTE_BUTTONS]
    ]
    return [buttons[index : index + 2] for index in range(0, len(buttons), 2)]


def session_buttons(sessions: Sequence[tuple[int, str]]) -> list[list[Btn]]:
    """Кнопка на каждую сессию плюс «Начать новую» последним рядом.

    Сводка вечера считается по запросу (решение владельца 2026-09-07) — нажатие
    на сессию и есть этот запрос. «Начать новую» — та же логика, что `/new`
    (SESSIONS_UX: «обработчик один»).
    """
    rows = [
        [Btn(text=title, callback_data=f"{SESSION_PREFIX}{session_id}")]
        for session_id, title in sessions
    ]
    rows.append([Btn(text="➕ Начать новую", callback_data=NEW_SESSION_DATA)])
    return rows


def note_row(note_id: int) -> list[Btn]:
    """Три действия над одной заметкой: правка текста, цвет, удаление."""
    return [
        Btn(text="✏️ Изменить", callback_data=f"{NOTE_EDIT_PREFIX}{note_id}"),
        Btn(text="🎨 Цвет", callback_data=f"{NOTE_COLOR_PREFIX}{note_id}"),
        Btn(text="🗑 Удалить", callback_data=f"{NOTE_DELETE_PREFIX}{note_id}"),
    ]


def note_color_buttons(note_id: int, colors: Sequence[tuple[str, str]]) -> list[list[Btn]]:
    """Кнопка на каждый цветовой архетип — по одной в ряд, чтобы подпись влезла."""
    return [
        [Btn(text=label, callback_data=f"{NOTE_COLOR_SET_PREFIX}{note_id}:{key}")]
        for key, label in colors
    ]


def set_nickname_button(known: bool) -> Btn:
    """Кнопка ввода ника в руме — единственный вход этого ввода (задача 23).

    До этой задачи ник записывался первым же текстовым сообщением игрока, и
    случайная реплика молча становилась ником. Теперь ввод открывается только
    отсюда либо командой `/nick`.
    """
    return Btn(
        text="✏️ Изменить ник" if known else "✏️ Указать ник",
        callback_data=SET_NICKNAME_DATA,
    )


def escalation_buttons(job_id: int, field: str, options: list[str]) -> list[Btn]:
    """Варианты эскалации плюс постоянная кнопка ручного ввода — одним рядом.

    **`job_id` в `callback_data` обязателен.** У игрока может ждать ответа
    больше одной задачи разом (спека §8.1: `awaiting_user` не считается
    активной и следующий скрин разбирается независимо), и ответ без номера
    задачи применялся бы к свежайшей — то есть к чужой руке, да ещё и писался бы
    ground truth с чужим `hand_id` (ревью раунда 1, R2).

    **Значение — ИНДЕКС варианта, а не сам вариант.** `callback_data` Телеграма
    ограничен 64 байтами, а вариантом бывает ник игрока; сам список лежит в
    `jobs.payload["escalation_options"]`, откуда бот его и берёт.
    """
    buttons = [
        Btn(text=option, callback_data=f"escalate:{job_id}:{field}:{index}")
        for index, option in enumerate(options)
    ]
    buttons.append(Btn(text="ввести вручную", callback_data=f"escalate:{job_id}:{field}:manual"))
    return buttons


__all__ = [
    "DETAIL_PREFIX",
    "DISAGREE_PREFIX",
    "MAIN_MENU",
    "MENU_HELP",
    "MENU_LEAKS",
    "MENU_NOTES",
    "MENU_SESSIONS",
    "MENU_SETTINGS",
    "MENU_TOURNAMENT",
    "NEW_SESSION_DATA",
    "NOTE_ADD_PREFIX",
    "NOTE_COLOR_PREFIX",
    "NOTE_COLOR_SET_PREFIX",
    "NOTE_DELETE_PREFIX",
    "NOTE_EDIT_PREFIX",
    "RANGES_PREFIX",
    "SESSION_PREFIX",
    "SET_NICKNAME_DATA",
    "Btn",
    "deep_dive_button",
    "escalation_buttons",
    "note_buttons_for_hand",
    "note_color_buttons",
    "note_row",
    "session_buttons",
    "set_nickname_button",
    "verdict_buttons",
]
