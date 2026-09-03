"""Очередь `jobs`: захват без гонок, сериализация по игроку, reaper — на настоящем
Postgres (`db_factory`, задача 14). `db` (откатываемая транзакция) здесь не годится
принципиально: `test_claim_atomic_two_workers` и `test_per_player_serialization`
нужны два воркера на РАЗНЫХ соединениях, видящих закоммиченные строки друг друга —
внутри одной отменяемой транзакции блокировки `FOR UPDATE SKIP LOCKED` конкурентам
попросту не от чего конфликтовать.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from harness.memory.models import Job
from harness.memory.repos import PlayersRepo, SessionsRepo
from harness.platform.queue import JobsQueue


async def _make_scope(session_factory) -> tuple[int, int]:
    """Валидные `player_id`/`session_id` для FK `jobs.player_id`/`jobs.session_id`
    (обе NOT NULL) — через настоящий коммит `db_factory`, а не `db`: очередь
    работает с закоммиченными строками, значит и подготовка сцены должна быть
    закоммичена, иначе воркер с другого соединения их просто не увидит.
    """
    async with session_factory() as session:
        player = await PlayersRepo(session).get_or_create(tg_user_id=1)
        session_row = await SessionsRepo(session).active_or_create(player.id)
        await session.commit()
        return player.id, session_row.id


async def test_claim_atomic_two_workers(db_factory):
    q = JobsQueue(db_factory)
    player_id, session_id = await _make_scope(db_factory)
    await q.enqueue(type="hh_scan", player_id=player_id, session_id=session_id, payload={})
    a, b = await asyncio.gather(q.claim("w1"), q.claim("w2"))
    assert sorted([a is not None, b is not None]) == [False, True]  # взял ровно один


async def test_per_player_serialization(db_factory):
    q = JobsQueue(db_factory)
    player_id, session_id = await _make_scope(db_factory)
    await q.enqueue(
        type="screenshot_analyze", player_id=player_id, session_id=session_id, payload={}
    )
    await q.enqueue(
        type="screenshot_analyze", player_id=player_id, session_id=session_id, payload={}
    )
    j1 = await q.claim("w1")
    assert j1 is not None
    assert await q.claim("w2") is None  # у игрока уже running
    await q.complete(j1.id)
    assert (await q.claim("w2")) is not None  # теперь можно


async def test_awaiting_user_not_active(db_factory):
    q = JobsQueue(db_factory)
    player_id, session_id = await _make_scope(db_factory)
    await q.enqueue(
        type="screenshot_analyze", player_id=player_id, session_id=session_id, payload={}
    )
    j1 = await q.claim("w1")  # первая задача игрока -> running
    assert j1 is not None
    await q.await_user(j1.id, resume_payload={"station": "engine"})
    await q.enqueue(
        type="screenshot_analyze", player_id=player_id, session_id=session_id, payload={}
    )
    assert (await q.claim("w2")) is not None  # awaiting_user не блокирует — спека §8.1


async def test_reap_returns_stuck_running(db_factory):
    q = JobsQueue(db_factory)
    player_id, session_id = await _make_scope(db_factory)
    jid = await q.enqueue(type="hh_scan", player_id=player_id, session_id=session_id, payload={})
    claimed = await q.claim("w1")
    assert claimed is not None and claimed.id == jid

    # Руками увести locked_at в прошлое — воркер, который взял задачу и умер, не
    # дойдя до complete()/fail(). reap() обязан найти её сам по возрасту лока.
    async with db_factory() as session:
        job = await session.get(Job, jid)
        assert job is not None
        job.locked_at = datetime.now(UTC) - timedelta(minutes=20)
        await session.commit()

    assert await q.reap(older_than_minutes=10) == 1

    async with db_factory() as session:
        job = await session.get(Job, jid)
        assert job is not None
        assert job.status == "queued"
        assert job.attempts == 1  # инкремент случился при claim(), reap() его не трогает
        assert job.locked_by is None
        assert job.locked_at is None


async def test_fail_after_max_attempts(db_factory):
    q = JobsQueue(db_factory)
    player_id, session_id = await _make_scope(db_factory)
    jid = await q.enqueue(type="hh_scan", player_id=player_id, session_id=session_id, payload={})
    assert await q.claim("w1") is not None

    # attempts достиг max_attempts (лимит уже израсходован) и задача всё ещё
    # зависла — reap() не отправляет её на четвёртую попытку, а закрывает failed.
    async with db_factory() as session:
        job = await session.get(Job, jid)
        assert job is not None
        job.attempts = job.max_attempts
        job.locked_at = datetime.now(UTC) - timedelta(minutes=20)
        await session.commit()

    assert await q.reap(older_than_minutes=10) == 1

    async with db_factory() as session:
        job = await session.get(Job, jid)
        assert job is not None
        assert job.status == "failed"
        assert job.error is not None
