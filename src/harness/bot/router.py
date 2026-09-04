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
"""

from __future__ import annotations

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from harness.bot.handlers import (
    BotDeps,
    handle_deep_dive_callback,
    handle_document,
    handle_new_session,
    handle_start,
)
from harness.presentation import Msg, photo_soon_msg

__all__ = ["DEEP_DIVE_PREFIX", "build_router"]

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
    `destination` не бывает: они есть только в типах. Поэтому здесь громкое
    исключение (его увидит лог), а не тихий `return` — молчание в ответ на
    присланный файл выглядело бы для игрока как «бот сломался и не признаётся».
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

    return router
