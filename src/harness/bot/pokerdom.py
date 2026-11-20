"""Owned image intake; validated hands are automatically saved by the worker."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING

from harness.memory.repos import PlayersRepo, QuotaRepo, SessionsRepo
from harness.platform.source_storage import LocalSourceStorage, SourceRef, image_mime
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
                     'проверенная полная рука автоматически сохраняется в историю.'))


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
    created: list[tuple[int, SourceRef]] = []
    try:
        return await _enqueue_image(deps, user_id, image, caption, original=original, created=created)
    except BaseException:
        # The inner session has rolled back before compensation queries references.
        # Hard process death skips this block; adapter intent + grace recovery handles it.
        from harness.platform.source_lifecycle import release_source
        store = deps.source_storage or LocalSourceStorage(deps.data_dir)
        for owner_id, ref in created:
            async with deps.db_factory() as db:
                await release_source(db, store, owner_id, ref)
                await db.commit()
        raise


async def _enqueue_image(deps: BotDeps, user_id: int, image: bytes, caption: str,
                        *, original: bool, created: list) -> Msg | None:
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
        store = deps.source_storage or LocalSourceStorage(deps.data_dir)
        from harness.memory.vision import image_lock, owned_image
        from harness.platform.source_lifecycle import recover_sources
        await SessionsRepo(db).lock_player(player.id)
        await recover_sources(db, store, player.id)
        sha = hashlib.sha256(image).hexdigest()
        await image_lock(db, player.id, sha)
        saved = await owned_image(db, player.id, sha)
        if saved is not None:
            ref = SourceRef.model_validate_json(saved.raw['source_ref'])
        else:
            ref = store.put(player.id, image, image_mime(image), 'image:' + sha)
            created.append((player.id, ref))
        session = await SessionsRepo(db).active_or_create(player.id)
        revision = await SessionsRepo(db).select_context(session.id, player.id)
        if player.pokerdom_nickname:
            from harness.memory.models import Job
            db.add(Job(type='pokerdom_read', session_id=session.id, player_id=player.id,
                       max_attempts=1, priority=100, payload={
                           'source_ref': ref.model_dump(), 'image_sha256': ref.sha256,
                           'context_revision': revision,
                           'nickname': player.pokerdom_nickname, 'adapter_format': 'numbered_hh_table',
                           'original_document': original, 'save_requested': False,
                       }))
            await db.commit()
            return None
        await PlayersRepo(db).set_pending_input(player.id, {
            'kind': INPUT_KIND, 'owner_id': player.id, 'source_ref': ref.model_dump(),
            'image_sha256': ref.sha256, 'session_id': session.id, 'context_revision': revision,
            'original_document': original, 'created_at': time.time(),
        })
        await db.commit()
    return Msg(text='Какой ник ваш на этом скрине? Напишите его точно как в таблице. '
                    'Запомню его для следующих скринов этой таблицы, в том числе в новой сессии. '
                    'Для отмены напишите «отмена». Изображение ждёт ответ один час.')


async def answer_nickname(deps: BotDeps, db: AsyncSession, player: Player,
                          pending: dict, answer: str) -> Msg:
    store = deps.source_storage or LocalSourceStorage(deps.data_dir)
    try:
        if pending.get('owner_id') != player.id:
            raise ValueError('intake owner mismatch')
        if 'source_ref' in pending:
            ref = SourceRef.model_validate(pending['source_ref'])
        else:
            ref = LocalSourceStorage(deps.data_dir).legacy(player.id, Path(pending['source_file']), pending['image_sha256'])
        store.read(player.id, ref, MAX_IMAGE_BYTES)
    except (ValueError, OSError):
        await PlayersRepo(db).set_pending_input(player.id, None)
        return Msg(text='Изображение недоступно. Пришлите его заново.')
    expired = time.time() - float(pending.get('created_at', 0)) > INTAKE_TTL_S
    if expired or answer.casefold() in {'отмена', '/cancel'}:
        await PlayersRepo(db).set_pending_input(player.id, None)
        from harness.memory.vision import image_lock, source_is_referenced
        await db.flush()
        await image_lock(db, player.id, ref.sha256)
        if not await source_is_referenced(db, ref.encode(), ref.model_dump()):
            store.delete(player.id, ref, ref.sha256)
        return Msg(text=('Время ожидания истекло.' if expired else 'Чтение отменено.') + ' Пришлите изображение заново, когда будете готовы.')
    if not answer or len(answer) > 64 or any(ord(c) < 32 for c in answer):
        return Msg(text='Нужен ваш точный ник на этом скрине, от 1 до 64 знаков, одной строкой. Или напишите «отмена».')
    quota = await QuotaRepo(db).check(player.id)
    if not quota.allowed:
        return quota_exceeded_msg(quota.hours_to_free)
    session_id = pending.get('session_id')
    revision = pending.get('context_revision')
    if session_id is None:
        session = await SessionsRepo(db).active_or_create(player.id)
        session_id = session.id
        revision = await SessionsRepo(db).select_context(session_id, player.id)
    from harness.memory.models import Session as SessionRow
    session = await db.get(SessionRow, session_id)
    if session is None or session.closed_at is not None or session.player_id != player.id:
        await PlayersRepo(db).set_pending_input(player.id, None)
        return Msg(text='Сессия изменилась. Пришлите изображение заново.')
    # Queue insert and intake completion share this transaction: no lost image/duplicate
    # job between commit and enqueue. Same defaults as the queue for this type.
    from harness.memory.models import Job
    db.add(Job(type='pokerdom_read', session_id=session_id, player_id=player.id,
               max_attempts=1, priority=100, payload={
                   'source_ref': ref.model_dump(), 'image_sha256': ref.sha256,
                   'context_revision': revision,
                   'nickname': answer, 'adapter_format': 'numbered_hh_table',
                   'original_document': pending['original_document'], 'save_requested': False,
               }))
    await PlayersRepo(db).set_pokerdom_nickname(player.id, answer)
    await PlayersRepo(db).set_pending_input(player.id, None)
    return Msg(text='Запомнил ник для скринов Покердома (/pdnick — изменить). Принял ник. Читаю таблицу и проверяю действия; проверенная полная рука автоматически сохраняется в историю.')


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
    return Msg(text="Ник для скринов Покердома изменён. Пришлите изображение заново; проверенные полные руки сохраняются в историю.")


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
