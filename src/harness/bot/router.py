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
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (
    CallbackQuery,
    ErrorEvent,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)

from harness.bot.handlers import (
    ESCALATION_PREFIX,
    UI_CALLBACK_PREFIXES,
    BotDeps,
    handle_deep_dive_callback,
    handle_document,
    handle_escalation_callback,
    handle_invite_command,
    handle_new_session,
    handle_nickname_command,
    handle_photo,
    handle_start,
    handle_text,
    handle_ui_callback,
)
from harness.presentation import (
    Msg,
    bot_failure_msg,
    unknown_button_msg,
)

__all__ = ["DEEP_DIVE_PREFIX", "build_router"]

_log = structlog.get_logger(__name__)

# Префикс `callback_data` кнопки «разобрать» (`presentation/keyboards.py`,
# `deep_dive_button`). Держится строкой в одном месте, чтобы фильтр роутера и
# разбор `callback_data` ниже не могли разойтись между собой.
DEEP_DIVE_PREFIX = "deep:"


def _markup(msg: Msg) -> InlineKeyboardMarkup | ReplyKeyboardMarkup | None:
    """Клавиатура сообщения: инлайн-кнопки ИЛИ нижнее меню — их не бывает двух.

    Взаимное исключение обеспечивает сам `Msg` (у Bot API одно поле
    `reply_markup`), здесь только выбор ветки. `resize_keyboard` — чтобы шесть
    кнопок меню не занимали пол-экрана телефона.
    """
    if msg.menu is not None:
        return ReplyKeyboardMarkup(
            keyboard=[[KeyboardButton(text=label) for label in row] for row in msg.menu],
            resize_keyboard=True,
        )
    if not msg.buttons:
        return None
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=b.text, callback_data=b.callback_data) for b in row]
            for row in msg.buttons
        ]
    )


async def _deliver(bot: Bot, chat_id: int, msg: Msg) -> None:
    """Отправить сообщение целиком: текст, потом картинки к нему.

    Картинки идут отдельными сообщениями и после текста — так они читаются как
    приложение к сказанному. Файл берётся с диска (`FSInputFile`): матрицы
    диапазонов рисует воркер и кладёт в общий том (`docker-compose.yml`), а не
    передаёт байтами через БД.
    """
    if msg.text:
        await bot.send_message(
            chat_id, msg.text, reply_markup=_markup(msg), parse_mode=msg.parse_mode
        )
    for photo in msg.photos:
        await bot.send_photo(chat_id, FSInputFile(photo.path), caption=photo.caption)


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

    @router.message(CommandStart(deep_link=True))
    async def on_start_with_code(message: Message, bot: Bot, command: CommandObject) -> None:
        """`/start КОД` — вход по инвайту (deep-link `t.me/бот?start=КОД`)."""
        if message.from_user is None:
            return
        msg = await handle_start(deps, message.from_user.id, command.args or "")
        await _deliver(bot, message.chat.id, msg)

    @router.message(CommandStart())
    async def on_start(message: Message, bot: Bot) -> None:
        if message.from_user is None:
            return
        msg = await handle_start(deps, message.from_user.id)
        await _deliver(bot, message.chat.id, msg)

    @router.message(Command("nick"))
    async def on_nick(message: Message, bot: Bot) -> None:
        if message.from_user is None:
            return
        msg = await handle_nickname_command(deps, message.from_user.id)
        if msg is None:
            return
        await _deliver(bot, message.chat.id, msg)

    @router.message(Command("invite"))
    async def on_invite(message: Message, bot: Bot) -> None:
        """`/invite` — команда владельца. Чужому нажатию не отвечает ничего."""
        if message.from_user is None:
            return
        msg = await handle_invite_command(deps, message.from_user.id)
        if msg is None:
            return
        await _deliver(bot, message.chat.id, msg)

    @router.message(Command("new"))
    async def on_new(message: Message, bot: Bot) -> None:
        if message.from_user is None:
            return
        msg = await handle_new_session(deps, message.from_user.id)
        await _deliver(bot, message.chat.id, msg)

    @router.message(F.document)
    async def on_document(message: Message, bot: Bot) -> None:
        """Файл: раздачи `.txt` либо скрин без сжатия — какой именно, решает не здесь.

        `mime_type` уходит обработчику как есть, вместе с именем файла: обе
        приметы приходят от Телеграма, и выбирать между ними — решение, а
        решений в этом модуле нет. `None` в ответе значит то же, что у кнопки
        «разобрать»: задача поставлена, дальше говорит воркер.
        """
        if message.from_user is None or message.document is None:
            return
        file_bytes = await _download(bot, message.document.file_id)
        msg = await handle_document(
            deps,
            tg_user_id=message.from_user.id,
            file_bytes=file_bytes,
            filename=message.document.file_name or "",
            mime_type=message.document.mime_type,
        )
        if msg is None:
            return
        await _deliver(bot, message.chat.id, msg)

    @router.message(F.photo)
    async def on_photo(message: Message, bot: Bot) -> None:
        """Фото — главный вход продукта. Берём САМЫЙ КРУПНЫЙ из присланных размеров.

        Телеграм отдаёт одно фото несколькими превью, от самого мелкого к самому
        крупному; читать надо последнее. Мелкое превью «прочиталось бы» тоже — и
        выдало бы уверенно неверные числа, потому что цифры на нём не различимы.
        """
        if message.from_user is None or not message.photo:
            return
        file_bytes = await _download(bot, message.photo[-1].file_id)
        msg = await handle_photo(deps, message.from_user.id, file_bytes)
        if msg is None:
            return
        await _deliver(bot, message.chat.id, msg)

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
        await _deliver(bot, callback.from_user.id, msg)

    @router.callback_query(F.data.startswith(UI_CALLBACK_PREFIXES))
    async def on_ui(callback: CallbackQuery, bot: Bot) -> None:
        """Кнопки экранов задачи 23 — один вход на все префиксы.

        Фильтр перечисляет ровно те префиксы, которые разбирает
        `handle_ui_callback`: кнопка с любым другим по-прежнему доходит до
        общего ответа «ещё не работает» ниже, а не пропадает молча.
        """
        await callback.answer()
        if callback.data is None:
            return
        msg = await handle_ui_callback(deps, callback.from_user.id, callback.data)
        if msg is None:
            return
        await _deliver(bot, callback.from_user.id, msg)

    @router.callback_query(F.data.startswith(ESCALATION_PREFIX))
    async def on_escalation(callback: CallbackQuery, bot: Bot) -> None:
        # `answer()` первым делом — «часики» на кнопке гасятся до любой работы.
        await callback.answer()
        if callback.data is None:
            return
        msg = await handle_escalation_callback(deps, callback.from_user.id, callback.data)
        if msg is None:
            return
        await _deliver(bot, callback.from_user.id, msg)

    @router.message(F.text & ~F.text.startswith("/"))
    async def on_text(message: Message, bot: Bot) -> None:
        """Обычный текст: ник в руме либо число, введённое вручную по эскалации.

        Команды сюда не попадают — их обработчики зарегистрированы выше, а
        фильтр отсекает всё, что начинается со слэша: иначе неизвестная команда
        сохранялась бы как ник в руме.
        """
        if message.from_user is None or message.text is None:
            return
        msg = await handle_text(deps, message.from_user.id, message.text)
        if msg is None:
            return
        await _deliver(bot, message.chat.id, msg)

    @router.callback_query()
    async def on_unhandled_callback(callback: CallbackQuery) -> None:
        """Ответ на любую кнопку, у которой обработчика ещё нет (round 5, Item G).

        Регистрируется ПОСЛЕ остальных и без фильтра: aiogram отдаёт обновление
        первому подошедшему обработчику, поэтому кнопки с известными префиксами
        уходят наверх, а всё остальное — сюда
        (`test_a_button_without_a_handler_still_gets_an_answer_not_a_spinner`).
        Без этого обработчика Телеграм крутил бы «часики» на кнопке, пока не
        сдался с ошибкой.

        В лог идёт только ПРЕФИКС `callback_data`: за ним следует номер руки, а
        это приватные данные игрока (docs/publishing-policy.md).
        """
        prefix = (callback.data or "").split(":", 1)[0]
        _log.info("callback_without_handler", prefix=prefix)
        await callback.answer(unknown_button_msg().text)

    @router.errors()
    async def on_error(event: ErrorEvent, bot: Bot) -> None:
        """Единственное место, где сбой обработчика превращается в слова игроку.

        Текст — из `presentation` и без единой подробности: причина уходит в лог
        трейсбеком целиком — тот же раздел ответственности, что у воркера между
        `jobs.error` и `failed_msg`, но НЕ тот же объём: `jobs.error` хранит
        `str(exc)`, одну строку без кадров стека, а здесь в лог идёт полный
        трейсбек (round 5, Item K.2 — прежняя формулировка сравнивала эти два
        места как равные и тем самым занижала одно и завышала другое). «Целиком»
        — не фигура речи и заслуга не этой строки: до fix round 2 `exc_info`
        рендерился в JSON как repr исключения без единого кадра стека, и это
        утверждение было ложным; трейсбек появляется `format_exc_info` в
        `platform/logs.py`, и держит его там регрессионный тест, а не намерение.
        Ответ шлём в
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
