"""Очередь `jobs`: захват без гонок, сериализация по игроку, reaper — на настоящем
Postgres (`db_factory`, задача 14). `db` (откатываемая транзакция) здесь не годится
принципиально: `test_claim_atomic_two_workers` и `test_per_player_serialization`
нужны два воркера на РАЗНЫХ соединениях, видящих закоммиченные строки друг друга —
внутри одной отменяемой транзакции блокировки `FOR UPDATE SKIP LOCKED` конкурентам
попросту не от чего конфликтовать.

Fix round 1 (ревью нашло Critical в `_CLAIM_SQL` — гонка по сериализации при ДВУХ
queued-задачах одного игрока, Finding 1) добавил ещё три теста: гонку на реальной
конкурентности (единственный способ её увидеть — последовательные вызовы её не
ловят, см. `test_per_player_serialization`), и по одному на каждое поведенческое
изменение из Finding 2/Minor — `complete()` больше не даёт зомби-воркеру затереть
задачу, подхваченную заново, `resume()` больше не переоткрывает задачу не в том
статусе молча.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from harness.memory.models import Job, async_session_factory
from harness.memory.repos import PlayersRepo, SessionsRepo
from harness.platform.queue import JobPreconditionFailed, JobsQueue


async def _make_scope(session_factory, tg_user_id: int = 1) -> tuple[int, int]:
    """Валидные `player_id`/`session_id` для FK `jobs.player_id`/`jobs.session_id`
    (обе NOT NULL) — через настоящий коммит `db_factory`, а не `db`: очередь
    работает с закоммиченными строками, значит и подготовка сцены должна быть
    закоммичена, иначе воркер с другого соединения их просто не увидит.
    `tg_user_id` настраивается — гоночному тесту нужен новый игрок на каждую
    итерацию, иначе `running`-задача из предыдущей итерации помешает следующей
    (сериализация по игроку сработает и там — по назначению, но не даст тесту
    завести новую пару задач того же игрока).
    """
    async with session_factory() as session:
        player = await PlayersRepo(session).get_or_create(tg_user_id=tg_user_id)
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


async def test_claim_serializes_concurrent_two_queued_jobs_one_player(pg, db_factory):
    """Fix round 1, Finding 1 (Critical): гонка, которую `test_per_player_
    serialization` не может поймать по конструкции — там `claim("w1")`
    полностью завершается (включая коммит) до начала `claim("w2")`, поэтому
    второй вызов честно видит уже закоммиченный `running` первого. Гонка живёт
    именно в конкурентности при ДВУХ queued-задачах одного игрока: `NOT EXISTS`
    в `_CLAIM_SQL` под READ COMMITTED не видит решения другого воркера, пока
    оно не закоммичено, а `FOR UPDATE SKIP LOCKED` блокирует только саму
    строку-кандидата — без партиционного уникального индекса
    (`uq_jobs_player_id_running`, миграция `0002`) оба воркера могли забрать
    РАЗНЫЕ задачи одного игрока одновременно.

    Первая версия этого теста гоняла оба `claim()` через один и тот же (уже
    прогретый предыдущими вызовами) `db_factory` — и не ловила гонку НИ РАЗУ
    (0/5): на прогретом пуле соединение уже установлено, `execute()`
    отрабатывает настолько быстро, что asyncio не успевает переключиться на
    вторую корутину до `commit()` первой, и `NOT EXISTS` в одиночку (без
    индекса) уже случайно всё сериализует. Тест был бы пройден и с багом, и без
    — то есть ничего не проверял (см. task-15-report.md, "Fix round 1", раздел
    про "drop the index" — там же доказательство от противного). Настоящая
    гонка нуждается в двух воркерах на СВОИХ, ещё не прогретых соединениях —
    так, как это и было бы у двух разных процессов воркера в проде — и даже
    тогда она вероятностная (эмпирически ~65-70% попаданий за прогон, не 100%):
    15 независимых попыток на разных игроках доводят шанс ни разу не поймать
    настоящий баг до исчезающе малого (0.3^15 ≈ 10⁻⁸), сохраняя тест честным
    прогоном через публичный `claim()`, а не подглядыванием во внутренний SQL.
    """
    dsn = pg.get_connection_url(driver="asyncpg")
    shared_q = JobsQueue(db_factory)

    for i in range(15):
        player_id, session_id = await _make_scope(db_factory, tg_user_id=90_000 + i)
        await shared_q.enqueue(
            type="screenshot_analyze", player_id=player_id, session_id=session_id, payload={}
        )
        await shared_q.enqueue(
            type="screenshot_analyze", player_id=player_id, session_id=session_id, payload={}
        )

        factory_a, factory_b = async_session_factory(dsn), async_session_factory(dsn)
        try:
            a, b = await asyncio.gather(
                JobsQueue(factory_a).claim(f"w1-{i}"), JobsQueue(factory_b).claim(f"w2-{i}")
            )
        finally:
            await factory_a.kw["bind"].dispose()
            await factory_b.kw["bind"].dispose()

        claimed = [job for job in (a, b) if job is not None]
        assert len(claimed) == 1, f"итерация {i}: оба воркера получили running одного игрока"


async def test_complete_rejects_zombie_worker_after_reclaim(db_factory):
    """Fix round 1, Finding 2 (Important): w1 берёт задачу и зависает —
    `locked_at` состарен вручную, как если бы воркер упал, не дойдя до
    `complete()`. `reap()` возвращает задачу в очередь, w2 забирает её заново.
    Если в этот момент "оживший" w1 всё-таки вызовет `complete()` со своим
    `worker_id`, до фикса это молча затёрло бы статус задачи, которую теперь
    активно ведёт w2 — без ошибки, без лога. С проверкой `locked_by` — честный
    `JobPreconditionFailed`, а закрыть задачу успехом всё ещё может её реальный
    текущий владелец (w2).
    """
    q = JobsQueue(db_factory)
    player_id, session_id = await _make_scope(db_factory)
    jid = await q.enqueue(type="hh_scan", player_id=player_id, session_id=session_id, payload={})

    j1 = await q.claim("w1")
    assert j1 is not None and j1.id == jid

    async with db_factory() as session:
        job = await session.get(Job, jid)
        assert job is not None
        job.locked_at = datetime.now(UTC) - timedelta(minutes=20)
        await session.commit()
    assert await q.reap(older_than_minutes=10) == 1

    j2 = await q.claim("w2")
    assert j2 is not None and j2.id == jid  # ту же задачу теперь ведёт w2

    with pytest.raises(JobPreconditionFailed):
        await q.complete(jid, worker_id="w1")  # w1 больше не владелец — не молча

    await q.complete(jid, worker_id="w2")  # настоящий владелец всё ещё может закрыть
    async with db_factory() as session:
        job = await session.get(Job, jid)
        assert job is not None
        assert job.status == "done"


async def test_resume_happy_path_returns_job_to_queue(db_factory):
    q = JobsQueue(db_factory)
    player_id, session_id = await _make_scope(db_factory)
    await q.enqueue(type="hh_scan", player_id=player_id, session_id=session_id, payload={})
    j1 = await q.claim("w1")
    assert j1 is not None
    await q.await_user(j1.id, resume_payload={"station": "engine"})

    await q.resume(j1.id)

    async with db_factory() as session:
        job = await session.get(Job, j1.id)
        assert job is not None
        assert job.status == "queued"


async def test_resume_requires_awaiting_user(db_factory):
    """Fix round 1, Minor: без предпосылки `resume()` молча переоткрыл бы
    задачу в любом статусе — например, повторный вызов на уже `queued`.
    """
    q = JobsQueue(db_factory)
    player_id, session_id = await _make_scope(db_factory)
    jid = await q.enqueue(type="hh_scan", player_id=player_id, session_id=session_id, payload={})
    # только что созданная задача — queued, не awaiting_user
    with pytest.raises(JobPreconditionFailed):
        await q.resume(jid)
