"""Temporary image intake: explicit hero per screenshot, no profile/history writes."""
from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from harness.memory.repos import PlayersRepo, QuotaRepo
from harness.presentation import Msg, invite_required_msg, quota_exceeded_msg

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from harness.bot.handlers import BotDeps
    from harness.memory.models import Player

MAX_IMAGE_BYTES = 10 * 1024 * 1024
INTAKE_TTL_S = 3600
INPUT_KIND = 'pokerdom_image_nickname'


def image_help() -> Msg:
    return Msg(text=('Поддерживается полная таблица истории завершённой раздачи: игроки, '
                     'ники/стеки, номера действий по улицам, борд и Total Pot. '
                     'Пришлите JPEG, PNG или WebP до 10 МБ, лучше файлом без сжатия. '
                     'Live-столы и другие форматы не поддерживаются. '
                     'При первом чтении спрошу ваш ник и запомню для следующих скринов; '
                     'рука не записывается в игровую историю.'))


async def preflight(deps: BotDeps, user_id: int, caption: str,
                    size: int | None) -> Msg | None:
    """Gate before Telegram download; caption never determines format or hero."""
    async with deps.db_factory() as db:
        player = await PlayersRepo(db).find(user_id)
        if player is None:
            return invite_required_msg()
        if player.pending_input:
            return Msg(text='Сначала завершите текущий ввод. Для отмены вопроса о нике напишите «отмена». Затем пришлите следующий скрин.')
        if size is not None and (size <= 0 or size > MAX_IMAGE_BYTES):
            return Msg(text='Изображение должно быть не больше 10 МБ. Пришлите таблицу меньшим файлом.')
        quota = await QuotaRepo(db).check(player.id)
        if not quota.allowed:
            return quota_exceeded_msg(quota.hours_to_free)
    return None


async def enqueue_image(deps: BotDeps, user_id: int, image: bytes, caption: str,
                        *, original: bool) -> Msg | None:
    """Hold validated bytes; reuse this player's explicit room setting, or ask once."""
    refusal = await preflight(deps, user_id, caption, len(image))
    if refusal:
        return refusal
    if not (image.startswith((b'\x89PNG\r\n\x1a\n', b'\xff\xd8\xff'))
            or (image.startswith(b'RIFF') and image[8:12] == b'WEBP')):
        return Msg(text='Не удалось прочитать JPEG, PNG или WebP. Пришлите изображение таблицы файлом.')
    async with deps.db_factory() as db:
        player = await PlayersRepo(db).find(user_id)
        assert player is not None
        # Lock/recheck FSM: concurrent photos must not replace a held image.
        from sqlalchemy import select

        from harness.memory.models import Player
        player = (await db.execute(select(Player).where(Player.id == player.id).with_for_update().execution_options(populate_existing=True))).scalar_one()
        if player.pending_input:
            return Msg(text='Сначала ответьте на текущий вопрос о нике или напишите «отмена».')
        quota = await QuotaRepo(db).check(player.id)
        if not quota.allowed:
            return quota_exceeded_msg(quota.hours_to_free)
        directory = deps.data_dir / 'vision' / str(player.id)
        directory.mkdir(parents=True, exist_ok=True)
        path = (directory / f'{uuid.uuid4().hex}.image').resolve()
        path.write_bytes(image)
        if player.pokerdom_nickname:
            from harness.memory.models import Job
            db.add(Job(type='pokerdom_read', session_id=None, player_id=player.id,
                       max_attempts=1, priority=100, payload={
                           'source_file': str(path), 'image_sha256': hashlib.sha256(image).hexdigest(),
                           'nickname': player.pokerdom_nickname, 'adapter_format': 'numbered_hh_table',
                           'original_document': original, 'save_requested': False,
                       }))
            await db.commit()
            return None
        await PlayersRepo(db).set_pending_input(player.id, {
            'kind': INPUT_KIND, 'owner_id': player.id, 'source_file': str(path),
            'image_sha256': hashlib.sha256(image).hexdigest(),
            'original_document': original, 'created_at': time.time(),
        })
        await db.commit()
    return Msg(text='Какой ник ваш на этом скрине? Напишите его точно как в таблице. '
                    'Запомню его для следующих скринов этой таблицы, в том числе в новой сессии. '
                    'Для отмены напишите «отмена». Изображение ждёт ответ один час.')


async def answer_nickname(deps: BotDeps, db: AsyncSession, player: Player,
                          pending: dict, answer: str) -> Msg:
    path = Path(pending.get('source_file', ''))
    root = (deps.data_dir / 'vision' / str(player.id)).resolve()
    owned = pending.get('owner_id') == player.id and path.resolve().parent == root
    if not owned:
        await PlayersRepo(db).set_pending_input(player.id, None)
        return Msg(text='Изображение недоступно. Пришлите его заново.')
    expired = time.time() - float(pending.get('created_at', 0)) > INTAKE_TTL_S
    if expired or answer.casefold() in {'отмена', '/cancel'}:
        await PlayersRepo(db).set_pending_input(player.id, None)
        path.unlink(missing_ok=True)
        return Msg(text=('Время ожидания истекло.' if expired else 'Чтение отменено.') + ' Пришлите изображение заново, когда будете готовы.')
    if not answer or len(answer) > 64 or any(ord(c) < 32 for c in answer):
        return Msg(text='Нужен ваш точный ник на этом скрине, от 1 до 64 знаков, одной строкой. Или напишите «отмена».')
    quota = await QuotaRepo(db).check(player.id)
    if not quota.allowed:
        return quota_exceeded_msg(quota.hours_to_free)
    if not path.is_file():
        await PlayersRepo(db).set_pending_input(player.id, None)
        return Msg(text='Изображение недоступно. Пришлите его заново.')
    # Queue insert and intake completion share this transaction: no lost image/duplicate
    # job between commit and enqueue. Same defaults as the queue for this type.
    from harness.memory.models import Job
    db.add(Job(type='pokerdom_read', session_id=None, player_id=player.id,
               max_attempts=1, priority=100, payload={
                   'source_file': str(path), 'image_sha256': pending['image_sha256'],
                   'nickname': answer, 'adapter_format': 'numbered_hh_table',
                   'original_document': pending['original_document'], 'save_requested': False,
               }))
    await PlayersRepo(db).set_pokerdom_nickname(player.id, answer)
    await PlayersRepo(db).set_pending_input(player.id, None)
    return Msg(text='Запомнил ник для скринов Покердома (/pdnick — изменить). Принял ник. Читаю таблицу и проверяю действия; рука не записывается в игровую историю.')


async def change_nickname(deps: BotDeps, user_id: int, nickname: str) -> Msg:
    async with deps.db_factory() as db:
        player = await PlayersRepo(db).find(user_id)
        if player is None:
            return invite_required_msg()
        if not nickname.strip():
            return Msg(text="/pdnick ВАШ_НИК — изменить ник для следующих скринов таблицы Покердома. Это отдельная настройка от ника GG.")
        try:
            await PlayersRepo(db).set_pokerdom_nickname(player.id, nickname)
        except ValueError:
            return Msg(text="Нужен точный ник от 1 до 64 знаков, одной строкой.")
        await db.commit()
    return Msg(text="Ник для скринов Покердома изменён. Пришлите изображение заново; история рук не сохраняется.")


async def cancel_image(deps: BotDeps, user_id: int) -> Msg:
    """Slash route cancels only this intake, never another input or an agent turn."""
    from sqlalchemy import select

    from harness.memory.models import Player

    async with deps.db_factory() as db:
        player = (await db.execute(select(Player).where(Player.tg_user_id == user_id)
                                  .with_for_update())).scalar_one_or_none()
        if player is None:
            return invite_required_msg()
        pending = dict(player.pending_input or {})
        if pending.get('kind') != INPUT_KIND:
            return Msg(text='Нет ожидающего изображения. Пришлите таблицу, когда будете готовы.')
        msg = await answer_nickname(deps, db, player, pending, '/cancel')
        await db.commit()
        return msg
