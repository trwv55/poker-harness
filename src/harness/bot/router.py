"""Обвязка aiogram: обновление Телеграма → вызов обработчика → отправка `Msg`.

Здесь намеренно нет ни одного решения. Всё, что можно решить неправильно —
какая сессия активна, тратит ли действие квоту, что именно сказать игроку —
живёт в `handlers.py` и `presentation/`, и проверяется тестами без токена бота.
Этот модуль умеет ровно три вещи: достать `tg_user_id` из обновления, скачать
файл и превратить `Msg` в вызов Bot API. Так и должно остаться: строчка логики,
попавшая сюда, окажется непроверяемой без живого Телеграма.

Ответ на callback (`deep:`) — `Msg | None`, и `None` здесь значимо: успешная
постановка задачи молчит, потому что дальше говорит воркер (одно редактируемое
сообщение прогресса, SESSIONS_UX). Отвечает бот только отказом по квоте.

**Сбой обработчика тоже говорит игроку** (fix round 1). Исключение, вылетевшее из
хендлера, aiogram логирует и глотает — для игрока это неотличимо от того, что бот
просто не заметил его файл. `on_error` (наблюдатель `router.errors`) закрывает
это одним местом на все входы сразу: причина уходит в лог целиком, игрок получает
`bot_failure_msg()` из `presentation`. Ровно тот же раздел ответственности, что у
воркера между `jobs.error` и `failed_msg`.
"""

from __future__ import annotations

import structlog
from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    ErrorEvent,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from harness.bot.handlers import (
    BotDeps,
    handle_deep_dive_callback,
    handle_document,
    handle_new_session,
    handle_start,
)
from harness.presentation import Msg, bot_failure_msg, photo_soon_msg

__all__ = ["DEEP_DIVE_PREFIX", "build_router"]

_log = structlog.get_logger(__name__)

# Префикс `callback_data` кнопки «разобрать» (`presentation/keyboards.py`,
# `deep_dive_button`). Держится строкой в одном месте, чтобы фильтр роутера и
# разбор `callback_data` ниже не могли разойтись между собой.
DEEP_DIVE_PREFIX = "deep:"


def _markup(msg: Msg) -> InlineKeyboardMarkup | None:
    if not msg.buttons:
        return None
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=b.text, callback_data=b.callback_data) for b in row]
            for row in msg.buttons
        ]
    )


async def _download(bot: Bot, file_id: str) -> bytes:
    """Скачать документ целиком в память.

    Оба `None` ниже — состояния Bot API, которых при вызове `download_file` без
    `destination` не бывает: они есть только в типах. Бросать здесь можно именно
    потому, что ниже есть `on_error`: он превращает любое исключение обработчика
    в лог плюс `bot_failure_msg()` игроку. Без него исход был бы тот же, что у
    тихого `return` — aiogram записал бы трейсбек и замолчал, а игрок остался бы
    ни с чем (первая версия этого докстринга утверждала обратное — fix round 1).
    """
    file = await bot.get_file(file_id)
    if file.file_path is None:
        raise RuntimeError(f"Telegram не вернул путь файла: file_id={file_id}")
    buffer = await bot.download_file(file.file_path)
    if buffer is None:
        raise RuntimeError(f"Telegram не отдал содержимое файла: file_id={file_id}")
    return buffer.read()


def build_router(deps: BotDeps) -> Router:
    """Роутер со всеми входами игрока. `deps` замыкается здесь, а не приходит
    через middleware aiogram: обработчики ниже — тонкие адаптеры, и лишний слой
    внедрения зависимостей только увеличил бы непроверяемую тестами часть.
    """
    router = Router(name="harness")

    @router.message(CommandStart())
    async def on_start(message: Message) -> None:
        if message.from_user is None:
            return
        msg = await handle_start(deps, message.from_user.id)
        await message.answer(msg.text, reply_markup=_markup(msg))

    @router.message(Command("new"))
    async def on_new(message: Message) -> None:
        if message.from_user is None:
            return
        msg = await handle_new_session(deps, message.from_user.id)
        await message.answer(msg.text, reply_markup=_markup(msg))

    @router.message(F.document)
    async def on_document(message: Message, bot: Bot) -> None:
        if message.from_user is None or message.document is None:
            return
        file_bytes = await _download(bot, message.document.file_id)
        msg = await handle_document(
            deps,
            tg_user_id=message.from_user.id,
            file_bytes=file_bytes,
            filename=message.document.file_name or "",
        )
        await message.answer(msg.text, reply_markup=_markup(msg))

    @router.message(F.photo)
    async def on_photo(message: Message) -> None:
        # Vision — задача 22. До неё честная заглушка, а не молчание в ответ на
        # главное действие продукта («кинул скрин»).
        msg = photo_soon_msg()
        await message.answer(msg.text, reply_markup=_markup(msg))

    @router.callback_query(F.data.startswith(DEEP_DIVE_PREFIX))
    async def on_deep_dive(callback: CallbackQuery, bot: Bot) -> None:
        # `answer()` первым делом — Телеграм гасит «часики» на кнопке, даже если
        # ниже случится отказ по квоте или исключение.
        await callback.answer()
        if callback.data is None:
            return
        hand_no = callback.data.removeprefix(DEEP_DIVE_PREFIX)
        msg = await handle_deep_dive_callback(deps, callback.from_user.id, hand_no)
        if msg is None:
            return
        # `chat_id == tg_user_id` для приватного чата (то же равенство, что в
        # `worker/pipeline.py::_chat_id`) — не полагаемся на `callback.message`,
        # которого у старого сообщения может уже не быть.
        await bot.send_message(callback.from_user.id, msg.text, reply_markup=_markup(msg))

    @router.errors()
    async def on_error(event: ErrorEvent, bot: Bot) -> None:
        """Единственное место, где сбой обработчика превращается в слова игроку.

        Текст — из `presentation` и без единой подробности: причина целиком
        уходит в лог (`exc_info`), как `jobs.error` у воркера. Ответ шлём в
        приватный чат по `tg_user_id` того, кто прислал обновление; если и это
        не проходит (`TelegramAPIError` — Телеграм недоступен, чат заблокирован),
        молчим уже осознанно и с записью в лог, а не потому, что не подумали.
        """
        _log.error("bot_handler_failed", exc_info=event.exception)
        source = event.update.message or event.update.callback_query
        if source is None or source.from_user is None:
            return
        msg = bot_failure_msg()
        try:
            await bot.send_message(source.from_user.id, msg.text)
        except TelegramAPIError:
            _log.exception("bot_failure_notice_undelivered")

    return router
