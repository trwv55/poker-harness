"""Synthetic Telegram intake → temporary queue → vision → report, no paid calls."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from harness.bot.handlers import BotDeps, handle_text
from harness.bot.pokerdom import MAX_IMAGE_BYTES, enqueue_image, preflight
from harness.bot.router import build_router
from harness.contracts.pokerdom_vision import PokerdomReadingV2, PokerdomUncertainty
from harness.contracts.raw import Street
from harness.memory.repos import PlayersRepo, QuotaRepo
from harness.parsers.pokerdom_vision import process_pokerdom
from harness.platform.queue import JobsQueue
from harness.presentation.pokerdom_vision import pokerdom_messages
from harness.worker.pipeline import Deps, run_job
from tests.test_pokerdom_nickname import v2
from tests.test_pokerdom_vision import showdown_table
from tests.test_worker_pipeline import FakeSender

IMAGE = b'\x89PNG\r\n\x1a\n' + b'synthetic image'
USER = 76543


def observed():
    data = v2().model_dump()
    data['format_observation'] = {'kind': 'numbered_hh_table', 'player_rows': True,
        'street_columns': True, 'numbered_actions': True, 'known_tokens': True,
        'board_area': True, 'end_street_pots': True, 'evidence': ['Player rows, numbered action cells and Total Pot labels']}
    return PokerdomReadingV2.model_validate(data)


@pytest.fixture
async def intake(db_factory, tmp_path):
    deps = BotDeps(db_factory=db_factory, queue=JobsQueue(db_factory), data_dir=tmp_path)
    async with db_factory() as db:
        p = await PlayersRepo(db).get_or_create(USER)
        p.gg_nickname = 'unrelated-gg-name'
        await db.commit()
    return deps


async def scalar(factory, sql):
    async with factory() as db:
        return await db.scalar(text(sql))


async def start(deps, *, original=True):
    msg = await enqueue_image(deps, USER, IMAGE, '/pokerdom not-a-shortcut', original=original)
    assert msg is not None and 'Какой ник ваш' in msg.text
    assert await scalar(deps.db_factory, 'select count(*) from jobs') == 0
    await handle_text(deps, USER, 'synthetic-alice')
    job = await deps.queue.claim('test-pd-worker')
    assert job is not None and job.type == 'pokerdom_read'
    assert job.session_id is None and job.max_attempts == 1
    return job


@pytest.mark.parametrize('media', ['photo', 'document'])
async def test_actual_router_to_queue_worker_report_no_gaming_writes(intake, monkeypatch, media):
    deps = intake
    router = build_router(deps)
    handler = next(h.callback for h in router.message.handlers if h.callback.__name__ == 'on_'+media)
    bot = SimpleNamespace(get_file=AsyncMock(return_value=SimpleNamespace(file_path='private.image', file_size=len(IMAGE))),
                          send_message=AsyncMock())
    async def download(path, destination):
        destination.write(IMAGE)
    bot.download_file = download
    message = SimpleNamespace(from_user=SimpleNamespace(id=USER), chat=SimpleNamespace(id=USER,type='private'),
        caption=None, photo=[SimpleNamespace(file_id='photo',file_size=len(IMAGE))],
        document=SimpleNamespace(file_id='doc',file_size=len(IMAGE),file_name='screen.png',mime_type='image/png'))
    await handler(message, bot)
    assert 'Какой ник ваш' in bot.send_message.call_args.args[1]
    await handle_text(deps, USER, 'synthetic-alice')
    job = await deps.queue.claim('test-pd-worker')
    calls=[]
    async def read(llm, image, **kwargs):
        assert image == IMAGE and kwargs['detect_format'] is True
        assert 'nickname' not in kwargs
        calls.append(kwargs)
        kwargs['observer']({'served_by':'synthetic-host','provider_reported_cost_usd':'.01','response':{'synthetic':True}})
        return observed(), None
    monkeypatch.setattr('harness.worker.pokerdom.read_pokerdom', read)
    sender=FakeSender()
    worker=Deps(db_factory=deps.db_factory,queue=deps.queue,sender=sender,llm=MagicMock(),pokerdom_llm=MagicMock())
    await run_job(job, worker)
    assert len(calls)==1
    assert await scalar(deps.db_factory, 'select status from jobs') == 'done'
    assert await scalar(deps.db_factory, 'select count(*) from traces') == 1
    for table in ('hands','analyses','tournaments','sessions','opponents','opponent_links','decision_points'):
        assert await scalar(deps.db_factory, f'select count(*) from {table}') == 0
    assert await scalar(deps.db_factory,'select gg_nickname from players') == 'unrelated-gg-name'
    assert await scalar(deps.db_factory,'select pending_input from players') is None
    report='\n'.join(m.text for m in sender.sent)
    assert 'тебе возвращается 400' in report and 'повышаешь до 600' in report
    assert '5\u2009000 фишек (25,00 ББ)' in report and 'не записана' not in report
    image_path=job.payload['source_file']
    from pathlib import Path
    assert Path(image_path).read_bytes()==IMAGE
    assert job.payload['original_document']==(media=='document')
    # Simulate resumed processing with durable result + reading; no second model call.
    await run_job(job, worker)
    assert len(calls)==1


@pytest.mark.parametrize('kind',['other_hh','live_table','not_hand','uncertain'])
async def test_unsupported_format_refuses_without_replay(intake,monkeypatch,kind):
    job=await start(intake)
    reading=observed(); assert reading.format_observation is not None
    reading.format_observation.kind=kind
    monkeypatch.setattr('harness.worker.pokerdom.read_pokerdom', AsyncMock(return_value=(reading,None)))
    sender=FakeSender()
    await run_job(job,Deps(db_factory=intake.db_factory,queue=intake.queue,sender=sender,llm=MagicMock(),pokerdom_llm=MagicMock()))
    report='\n'.join(m.text for m in sender.sent)
    assert 'Не удалось распознать полную таблицу' in report
    assert 'Расчёт кодом' not in report and kind not in report


@pytest.mark.parametrize('marker',['player_rows','street_columns','numbered_actions','known_tokens','board_area','end_street_pots','evidence','legacy'])
async def test_missing_format_marker_never_uses_poker_legality_as_format_detector(intake,monkeypatch,marker):
    job=await start(intake); reading=observed()
    if marker=='legacy': reading.format_observation=None
    else: setattr(reading.format_observation,marker,[] if marker=='evidence' else False)
    monkeypatch.setattr('harness.worker.pokerdom.read_pokerdom',AsyncMock(return_value=(reading,None)))
    sender=FakeSender()
    await run_job(job,Deps(db_factory=intake.db_factory,queue=intake.queue,sender=sender,llm=MagicMock(),pokerdom_llm=MagicMock()))
    assert any('Не удалось распознать полную таблицу' in m.text for m in sender.sent)


@pytest.mark.parametrize('mode',['cancel','expired','wrong-owner','blank','long','quota'])
async def test_nickname_intake_is_owned_expiring_cancellable_and_quota_gated(intake,mode):
    await enqueue_image(intake,USER,IMAGE,'',original=True)
    async with intake.db_factory() as db:
        p=await PlayersRepo(db).find(USER)
        assert p is not None
        pending: dict = dict(p.pending_input or {})
        if mode=='expired':pending['created_at']=0
        if mode=='wrong-owner':pending['owner_id']=p.id+1
        if mode=='quota':p.quota_daily=0
        p.pending_input=pending
        await db.commit()
    answer={'cancel':'отмена','blank':' ','long':'x'*65}.get(mode,'synthetic-alice')
    msg=await handle_text(intake,USER,answer)
    assert msg is not None
    assert await scalar(intake.db_factory,'select count(*) from jobs')==0
    assert await scalar(intake.db_factory,'select count(*) from sessions')==0


async def test_before_download_access_size_quota_and_pending_input(intake):
    assert await preflight(intake,USER+1,'',10) is not None
    assert await preflight(intake,USER,'',MAX_IMAGE_BYTES+1) is not None
    assert not list(intake.data_dir.rglob('*.image'))
    await enqueue_image(intake,USER,IMAGE,'',original=True)
    before=list(intake.data_dir.rglob('*.image'))
    assert await preflight(intake,USER,'',10) is not None
    await enqueue_image(intake,USER,IMAGE,'',original=True)
    assert list(intake.data_dir.rglob('*.image'))==before


@pytest.mark.parametrize('job_type',['agent','hh_scan','deep_dive','question','screenshot_analyze','eval_run'])
async def test_only_temporary_job_can_lack_session(intake,job_type):
    async with intake.db_factory() as db:
        player=await PlayersRepo(db).find(USER)
        assert player is not None
        with pytest.raises(IntegrityError):
            await intake.queue.enqueue(type=job_type,player_id=player.id,session_id=None,payload={})


async def test_temporary_job_spends_quota_but_no_game_session(intake):
    job=await start(intake)
    async with intake.db_factory() as db:
        quota=await QuotaRepo(db).check(job.player_id)
        assert quota.left==quota.total-1


@pytest.mark.parametrize('mode',['read-error','uncertain','incomplete','wrong-nickname','missing-card','missing-pot','no-events'])
async def test_read_failures_are_diagnostic_no_hidden_retry(intake,monkeypatch,mode):
    job=await start(intake); reading=observed()
    if mode=='uncertain':reading.uncertainties=[PokerdomUncertainty(field='event',reason='unreadable')]
    if mode=='incomplete':reading.action_log_complete=False
    if mode in {'missing-card','missing-pot'}:
        format_observation=reading.format_observation
        reading=v2(showdown_table(explicit=False));reading.format_observation=format_observation
    if mode=='no-events':reading.events=[]
    if mode=='missing-card':reading.board[Street.FLOP]=[]
    if mode=='missing-pot':reading.pot_after_street[Street.FLOP]=None
    if mode=='wrong-nickname':job.payload['nickname']='unknown-exact-name'
    stub=AsyncMock(side_effect=RuntimeError('synthetic provider error')) if mode=='read-error' else AsyncMock(return_value=(reading,None))
    monkeypatch.setattr('harness.worker.pokerdom.read_pokerdom',stub)
    sender=FakeSender()
    await run_job(job,Deps(db_factory=intake.db_factory,queue=intake.queue,sender=sender,llm=MagicMock(),pokerdom_llm=MagicMock()))
    assert stub.await_count==1
    report='\n'.join(m.text for m in sender.sent)
    if mode=='missing-pot':assert 'Не удалось прочитать банк' in report
    elif mode=='no-events':assert 'последовательность действий' in report
    elif mode=='read-error':assert await scalar(intake.db_factory,'select status from jobs')=='failed'
    else:assert 'Не удалось' in report
    assert await scalar(intake.db_factory,'select count(*) from hands')==0


def test_human_report_conditional_cards_unknown_pot_and_unicode_length():
    reading=v2(showdown_table(explicit=False));reading.notes=['<b>not markup</b>😀'*800]
    result=process_pokerdom(reading,nickname='synthetic-alice')
    messages=pokerdom_messages(result)
    assert len(messages)==1
    assert all(m.parse_mode is None and len(m.text.encode('utf-16-le'))//2<=3500 for m in messages)
    full=''.join(m.text for m in messages)
    assert '<b>not markup</b>' not in full and 'По видимым картам' in full and 'вскрытие не подтверждено' in full


async def test_nickname_saved_once_reused_across_sessions_and_changeable(intake):
    from harness.bot.handlers import handle_new_session
    from harness.bot.pokerdom import change_nickname
    await start(intake)
    assert await scalar(intake.db_factory, 'select pokerdom_nickname from players') == 'synthetic-alice'
    await intake.queue.complete(1, worker_id='test-pd-worker')
    assert await enqueue_image(intake,USER,IMAGE,'ignored caption',original=True) is None
    assert await scalar(intake.db_factory,'select count(*) from jobs')==2
    assert await scalar(intake.db_factory,'select count(*) from sessions')==0
    await handle_new_session(intake,USER)
    assert await scalar(intake.db_factory,'select pokerdom_nickname from players')=='synthetic-alice'
    assert await enqueue_image(intake,USER,IMAGE,'',original=True) is None
    await change_nickname(intake,USER,'synthetic-new-nick')
    assert await enqueue_image(intake,USER,IMAGE,'',original=True) is None
    assert await scalar(intake.db_factory,"select payload->>'nickname' from jobs order by id desc limit 1")=='synthetic-new-nick'
    assert await scalar(intake.db_factory,'select gg_nickname from players')=='unrelated-gg-name'


async def test_second_user_does_not_inherit_owner_or_gg_nickname(intake):
    await start(intake)
    async with intake.db_factory() as db:
        p=await PlayersRepo(db).get_or_create(USER+1)
        p.gg_nickname='synthetic-alice'
        await db.commit()
    msg=await enqueue_image(intake,USER+1,IMAGE,'',original=True)
    assert msg is not None and 'Какой ник ваш' in msg.text
    assert await scalar(intake.db_factory,'select count(*) from jobs')==1


async def test_concurrent_nickname_answers_queue_held_image_only_once(intake):
    import asyncio
    await enqueue_image(intake,USER,IMAGE,'',original=True)
    await asyncio.gather(handle_text(intake,USER,'synthetic-alice'),handle_text(intake,USER,'synthetic-alice'))
    assert await scalar(intake.db_factory,'select count(*) from jobs')==1
    assert await scalar(intake.db_factory,'select count(*) from sessions')==0


@pytest.mark.parametrize('media',['photo','document'])
@pytest.mark.parametrize('mode',['stranger','oversize','group','pending','quota'])
async def test_router_gates_before_getfile_and_download(intake,media,mode):
    if mode=='pending':await enqueue_image(intake,USER,IMAGE,'',original=True)
    if mode=='quota':
        async with intake.db_factory() as db:
            p=await PlayersRepo(db).find(USER); assert p is not None
            p.quota_daily=0;await db.commit()
    router=build_router(intake)
    handler=next(h.callback for h in router.message.handlers if h.callback.__name__=='on_'+media)
    size=MAX_IMAGE_BYTES+1 if mode=='oversize' else len(IMAGE)
    message=SimpleNamespace(from_user=SimpleNamespace(id=USER+1 if mode=='stranger' else USER),
        chat=SimpleNamespace(id=USER,type='group' if mode=='group' else 'private'),caption=None,
        photo=[SimpleNamespace(file_id='image',file_size=size)],
        document=SimpleNamespace(file_id='image',file_size=size,file_name='image.jpg',mime_type='image/jpeg'))
    bot=SimpleNamespace(get_file=AsyncMock(),download_file=AsyncMock(),send_message=AsyncMock())
    await handler(message,bot)
    bot.get_file.assert_not_awaited();bot.download_file.assert_not_awaited()


def test_tg_schema_requires_format_but_old_v2_result_remains_readable():
    from pydantic import ValidationError

    from harness.contracts.pokerdom_vision import PokerdomTelegramReading
    data=v2().model_dump();data.pop('format_observation')
    assert PokerdomReadingV2.model_validate(data).format_observation is None
    with pytest.raises(ValidationError):PokerdomTelegramReading.model_validate(data)
    data['format_observation']=None
    with pytest.raises(ValidationError):PokerdomTelegramReading.model_validate(data)


def test_migration_0017_up_down_preserves_old_jobs_and_separates_room_setting():
    from alembic import command
    from sqlalchemy import create_engine
    from testcontainers.community.postgres import PostgresContainer

    from tests.conftest import alembic_config
    with PostgresContainer('postgres:16-alpine') as container:
        dsn=container.get_connection_url(driver='psycopg')
        cfg=alembic_config(dsn)
        command.upgrade(cfg,'0016')
        engine=create_engine(dsn)
        try:
            with engine.begin() as db:
                player=db.execute(text('insert into players(tg_user_id,gg_nickname) values (1222,\'gg-only\') returning id')).scalar_one()
                session=db.execute(text("insert into sessions(player_id,started_at,title) values (:p,now(),'synthetic') returning id"),{'p':player}).scalar_one()
                db.execute(text("insert into jobs(type,player_id,session_id,payload) values ('agent',:p,:s,'{}')"),{'p':player,'s':session})
            command.upgrade(cfg,'head')
            with engine.begin() as db:
                assert db.execute(text('select pokerdom_nickname from players')).scalar_one() is None
                db.execute(text("update players set pokerdom_nickname='pd-only'"))
                job=db.execute(text("insert into jobs(type,player_id,session_id,payload) values ('pokerdom_read',:p,NULL,'{}') returning id"),{'p':player}).scalar_one()
                trace=db.execute(text('insert into traces(job_id) values (:j) returning id'),{'j':job}).scalar_one()
                db.execute(text("insert into llm_calls(trace_id,provider,model,purpose) values (:t,'synthetic','m','vision_extract')"),{'t':trace})
            command.downgrade(cfg,'0016')
            with engine.begin() as db:
                assert db.execute(text('select type from jobs')).scalars().all()==['agent']
                assert db.execute(text('select gg_nickname from players')).scalar_one()=='gg-only'
                assert db.execute(text('select count(*) from traces')).scalar_one()==0
            command.upgrade(cfg,'head')
            with engine.begin() as db:
                assert db.execute(text('select pokerdom_nickname from players')).scalar_one() is None
        finally:engine.dispose()


async def test_reading_checkpoint_survives_adaptation_crash_without_second_model_call(intake,monkeypatch):
    from pathlib import Path
    job=await start(intake)
    stub=AsyncMock(return_value=(observed(),None))
    monkeypatch.setattr('harness.worker.pokerdom.read_pokerdom',stub)
    monkeypatch.setattr('harness.worker.pokerdom.process_pokerdom',MagicMock(side_effect=RuntimeError('after read checkpoint')))
    deps=Deps(db_factory=intake.db_factory,queue=intake.queue,sender=FakeSender(),llm=MagicMock(),pokerdom_llm=MagicMock())
    await run_job(job,deps)
    directory=Path(job.payload['source_file']).parent/f'job-{job.id}'
    assert (directory/'reading.json').exists() and not (directory/'result.json').exists()
    assert await scalar(intake.db_factory,'select status from jobs')=='failed'
    # Explicit test-only recovery, not an automatic product retry policy.
    async with intake.db_factory() as db:
        await db.execute(text("update jobs set status='queued',max_attempts=2 where id=:j"),{'j':job.id})
        await db.commit()
    resumed=await intake.queue.claim('resumed-worker');assert resumed is not None
    monkeypatch.setattr('harness.worker.pokerdom.process_pokerdom',process_pokerdom)
    await run_job(resumed,deps)
    assert stub.await_count==1
    assert await scalar(intake.db_factory,'select status from jobs')=='done'
    assert (directory/'result.json').exists()


async def test_slash_cancel_real_dispatch_cleans_held_intake_without_queue_or_llm(intake,monkeypatch):
    from datetime import UTC, datetime

    from aiogram import Bot, Dispatcher
    from aiogram.types import Chat, Message, MessageEntity, Update, User
    await enqueue_image(intake,USER,IMAGE,'',original=True)
    held=list(intake.data_dir.rglob('*.image'))
    assert len(held)==1 and held[0].exists()
    model=AsyncMock()
    monkeypatch.setattr('harness.worker.pokerdom.read_pokerdom',model)
    bot=Bot(token='123456:synthetic-token')
    sent=AsyncMock()
    monkeypatch.setattr(bot,'send_message',sent)
    dispatcher=Dispatcher()
    dispatcher.include_router(build_router(intake))
    update=Update(update_id=1,message=Message(
        message_id=1,date=datetime.now(UTC),chat=Chat(id=USER,type='private'),
        from_user=User(id=USER,is_bot=False,first_name='Synthetic'),text='/cancel',
        entities=[MessageEntity(type='bot_command',offset=0,length=7)],
    ))
    try:
        await dispatcher.feed_update(bot,update)
    finally:
        await bot.session.close()
    assert sent.await_count==1 and 'отменено' in sent.call_args.args[1]
    assert await scalar(intake.db_factory,'select pending_input from players') is None
    assert not held[0].exists()
    assert await scalar(intake.db_factory,'select count(*) from jobs')==0
    assert await scalar(intake.db_factory,'select count(*) from sessions')==0
    model.assert_not_awaited()
