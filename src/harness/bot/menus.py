"""Экраны нижнего меню: подпись нажатой кнопки → готовое сообщение игроку.

Постоянная клавиатура Телеграма присылает боту не `callback_data`, а САМ ТЕКСТ
нажатой кнопки — обычным текстовым сообщением. Поэтому «меню» здесь это
таблица «подпись → экран» (`_SCREENS`), а не набор обработчиков нажатий: подписи
берутся из `presentation.keyboards`, откуда их берёт и сама клавиатура, и
разойтись они не могут.

**Что этот модуль делает и чего не делает.** Делает: читает память
(`memory/repos.py`) и зовёт конструктор текста (`presentation`). Не делает:
ничего не решает про транзакции — `AsyncSession` приходит открытой снаружи, как
и во всех репозиториях, а границу транзакции держит вызывающий
(`bot/handlers.py`, см. его модульный докстринг). Ни одной строки текста здесь
тоже нет: экран собирает `presentation`, правило единого голоса (спека §4).

Отдельным модулем, а не частью `handlers.py`, — так задан план задачи 23
(«Create: `bot/menus.py`»), и разделение полезное: `handlers.py` отвечает за то,
ЧТО значит входящее сообщение, а этот файл — за то, что показать на экране,
который уже опознан.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from harness.memory.models import Player
from harness.memory.repos import (
    LeaksRepo,
    NotesRepo,
    QuotaRepo,
    SessionsRepo,
)
from harness.presentation import (
    MENU_HELP,
    MENU_LEAKS,
    MENU_NOTES,
    MENU_SESSIONS,
    MENU_SETTINGS,
    MENU_TOURNAMENT,
    Msg,
    help_msg,
    hh_prompt_msg,
    leaks_msg,
    notes_msg,
    session_summary_msg,
    sessions_msg,
    settings_msg,
)

__all__ = ["is_menu_label", "render_menu_screen", "session_summary_screen"]

# Сколько вечеров показывает экран «Сессии». Не предел Телеграма, а предел
# смысла: история нужна, чтобы вернуться во вчерашний вечер, а не листать месяц.
_MAX_SESSIONS_LISTED = 10


async def _sessions_screen(db: AsyncSession, player: Player) -> Msg:
    return sessions_msg(
        await SessionsRepo(db).list_for_player(player.id, limit=_MAX_SESSIONS_LISTED)
    )


async def _tournament_screen(_db: AsyncSession, _player: Player) -> Msg:
    return hh_prompt_msg()


async def _leaks_screen(db: AsyncSession, player: Player) -> Msg:
    return leaks_msg(await LeaksRepo(db).overview(player.id))


async def _notes_screen(db: AsyncSession, player: Player) -> Msg:
    return notes_msg(await NotesRepo(db).list_for_player(player.id))


async def _settings_screen(db: AsyncSession, player: Player) -> Msg:
    """Настройки: ник в руме, остаток дневного лимита, о боте.

    Остаток берётся у `QuotaRepo` — той же реализации окна, по которой бот
    отказывает в разборе, а воркер подписывает вердикт. Второго счётчика в
    продукте нет (спека §9).
    """
    quota = await QuotaRepo(db).check(player.id)
    return settings_msg(player.gg_nickname, quota.left, quota.total, is_dev=player.is_dev)


async def _help_screen(_db: AsyncSession, _player: Player) -> Msg:
    return help_msg()


_SCREENS: dict[str, Callable[[AsyncSession, Player], Awaitable[Msg]]] = {
    MENU_SESSIONS: _sessions_screen,
    MENU_TOURNAMENT: _tournament_screen,
    MENU_LEAKS: _leaks_screen,
    MENU_NOTES: _notes_screen,
    MENU_SETTINGS: _settings_screen,
    MENU_HELP: _help_screen,
}


def is_menu_label(text: str) -> bool:
    """Это нажатие кнопки нижнего меню, а не обычный текст игрока."""
    return text in _SCREENS


async def render_menu_screen(db: AsyncSession, player: Player, text: str) -> Msg | None:
    """Экран по подписи нажатой кнопки; `None` — это не кнопка меню."""
    screen = _SCREENS.get(text)
    if screen is None:
        return None
    return await screen(db, player)


async def session_summary_screen(
    db: AsyncSession, player: Player, session_id: int
) -> Msg | None:
    """Сводка вечера по нажатию на сессию; `None` — сессия чужая или её нет.

    Считается по запросу, а не показывается сама при `/new` (решение владельца
    2026-09-07). Владельца сверяет `SessionsRepo.summary`: номер приезжает в
    `callback_data`, то есть из внешнего мира.
    """
    summary = await SessionsRepo(db).summary(session_id, player.id)
    return None if summary is None else session_summary_msg(summary)
