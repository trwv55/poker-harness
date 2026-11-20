"""Actual PG intake/worker/storage/analysis integration; all poker data synthetic."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select, text

from harness.bot.handlers import handle_new_session, handle_text
from harness.bot.pokerdom import MAX_IMAGE_BYTES, cancel_image, enqueue_image
from harness.calcs import _analysed_hand, _measure
from harness.contracts import FrequencyStat, Street, Subject, Window
from harness.memory.models import Hand
from harness.memory.repos import (
    AnalysesRepo,
    DecisionSpotsRepo,
    HandsRepo,
    PlayersRepo,
    SessionsRepo,
)
from harness.memory.vision import save_reading
from harness.parsers.pokerdom_vision import process_pokerdom
from harness.platform.source_storage import LocalSourceStorage, SourceRef
from harness.worker.pipeline import Deps, _agent_context, run_job
from tests.test_pokerdom_telegram import IMAGE, USER, intake, observed, scalar, start
from tests.test_source_storage import FakeStorage
from tests.test_worker_pipeline import FakeSender

# Imported fixture is intentionally shared: exercises the same real Telegram intake.
__all__ = ['intake']


def worker(deps):
    return Deps(db_factory=deps.db_factory,queue=deps.queue,sender=FakeSender(),llm=MagicMock(),
        pokerdom_llm=MagicMock(),data_dir=deps.data_dir,source_storage=deps.source_storage)


async def read_once(deps, monkeypatch):
    stub=AsyncMock(return_value=(observed(),None))
    monkeypatch.setattr('harness.worker.pokerdom.read_pokerdom',stub)
    job=await start(deps)
    await run_job(job,worker(deps))
    assert await scalar(deps.db_factory,'select status from jobs where id=1')=='done'
    assert await scalar(deps.db_factory,'select count(*) from analyses')==1
    return job,stub


async def test_full_persist_discussion_and_duplicate_after_new(intake,monkeypatch):
    job,stub=await read_once(intake,monkeypatch)
    async with intake.db_factory() as db:
        hand=await db.scalar(select(Hand))
        assert hand is not None
        analysis=await AnalysesRepo(db).get_by_hand(hand.id)
        assert analysis is not None
        assert hand.tournament_id is None and hand.raw['hand_no']==''
        assert hand.raw['timestamp'].startswith('0001-') and hand.raw['collected']==[]
        assert hand.provenance=='screenshot' and hand.image_hash==job.payload['image_sha256']
        assert SourceRef.model_validate_json(hand.raw['source_ref']).key
        first_id, first_analysis, origin=hand.id,analysis.id,hand.session_id
        points=await scalar(intake.db_factory,'select count(*) from decision_points')
        context=await _agent_context(db,analysis.id,job.player_id,origin)
        assert context is not None and 'гипотеза' in context.text
        assert (await _analysed_hand(db,job.player_id,analysis.id)).bb==200
        await db.commit()
    await handle_new_session(intake,USER)
    await enqueue_image(intake,USER,IMAGE,'',original=True)
    duplicate=await intake.queue.claim('duplicate');assert duplicate is not None
    assert duplicate.session_id!=origin
    await run_job(duplicate,worker(intake))
    assert stub.await_count==1
    assert await scalar(intake.db_factory,'select count(*) from hands')==1
    assert await scalar(intake.db_factory,'select count(*) from analyses')==1
    assert await scalar(intake.db_factory,'select count(*) from decision_points')==points
    async with intake.db_factory() as db:
        assert (await HandsRepo(db).get(first_id)).session_id==origin
        state=await SessionsRepo(db).agent_state(duplicate.session_id)
        assert state.last_analysis_id==first_analysis
        context=await _agent_context(db,first_analysis,job.player_id,duplicate.session_id)
        assert context is not None and 'прежнем вечере' in context.text
        summary=await SessionsRepo(db).summary(duplicate.session_id,job.player_id)
        assert summary is not None and summary.hands==0
        summary=await SessionsRepo(db).summary(origin,job.player_id)
        assert summary is not None and summary.hands==1
        assert await HandsRepo(db).find_by_hand_no(origin,'') is None
        assert await HandsRepo(db).find_session_by_hand_no(job.player_id,'') is None


async def test_different_owner_cannot_resolve_activate_or_dedup(intake,monkeypatch):
    job,_=await read_once(intake,monkeypatch)
    async with intake.db_factory() as db:
        analysis=await db.scalar(text('select id from analyses'))
        other=await PlayersRepo(db).get_or_create(USER+1)
        session=await SessionsRepo(db).active_or_create(other.id)
        revision=await SessionsRepo(db).select_context(session.id,other.id)
        with pytest.raises(ValueError):
            await SessionsRepo(db).publish_analysis(session.id,other.id,revision,analysis)
        assert await _agent_context(db,analysis,other.id,session.id) is None
        with pytest.raises(ValueError):
            await _analysed_hand(db,other.id,analysis)
        await db.commit()
    # Same bytes uploaded by another owner must actually be read for that owner.
    await enqueue_image(intake,USER+1,IMAGE,'',original=True)
    await handle_text(intake,USER+1,'synthetic-alice')
    second=await intake.queue.claim('other');assert second is not None
    await run_job(second,worker(intake))
    assert await scalar(intake.db_factory,'select count(*) from hands')==2
    assert await scalar(intake.db_factory,'select count(*) from analyses')==2
    store=LocalSourceStorage(intake.data_dir)
    with pytest.raises(ValueError):
        store.read(second.player_id,SourceRef.model_validate(job.payload['source_ref']),MAX_IMAGE_BYTES)


async def test_analysis_failure_resume_uses_same_enriched_without_ocr_or_replay(intake,monkeypatch):
    from harness.worker import pipeline
    original=pipeline._analyze_hand_with_cache
    monkeypatch.setattr(pipeline,'_analyze_hand_with_cache',MagicMock(side_effect=RuntimeError('synthetic analysis failure')))
    stub=AsyncMock(return_value=(observed(),None));monkeypatch.setattr('harness.worker.pokerdom.read_pokerdom',stub)
    job=await start(intake);deps=worker(intake)
    await run_job(job,deps)
    assert await scalar(intake.db_factory,'select count(*) from hands')==1
    assert await scalar(intake.db_factory,'select count(*) from analyses')==0
    assert await scalar(intake.db_factory,'select last_analysis_id from sessions') is None
    assert isinstance(deps.sender,FakeSender)
    assert any('временно недоступен' in m.text for m in deps.sender.sent)
    monkeypatch.setattr(pipeline,'_analyze_hand_with_cache',original)
    monkeypatch.setattr('harness.worker.pokerdom.process_pokerdom',MagicMock(side_effect=AssertionError('second replay forbidden')))
    await enqueue_image(intake,USER,IMAGE,'',original=True)
    resumed=await intake.queue.claim('resume');assert resumed is not None
    await run_job(resumed,worker(intake))
    assert stub.await_count==1
    assert await scalar(intake.db_factory,'select count(*) from analyses')==1
    assert await scalar(intake.db_factory,'select count(*) from hands')==1


@pytest.mark.parametrize('selection',['new-image','new-session','spot'])
async def test_late_finish_does_not_publish_over_new_selection(intake,monkeypatch,selection):
    job=await start(intake)
    async def read(*args,**kwargs):
        if selection=='new-image':
            await enqueue_image(intake,USER,IMAGE+b'different','',original=True)
        elif selection=='new-session':
            await handle_new_session(intake,USER)
        else:
            async with intake.db_factory() as db:
                await DecisionSpotsRepo(db).save(player_id=job.player_id,session_id=job.session_id,
                    source_text='synthetic draft',raw={},canonical=None,snapshot=None)
                await db.commit()
        return observed(),None
    monkeypatch.setattr('harness.worker.pokerdom.read_pokerdom',read)
    await run_job(job,worker(intake))
    assert await scalar(intake.db_factory,'select count(*) from hands')==1
    assert await scalar(intake.db_factory,'select last_analysis_id from sessions order by id desc limit 1') is None
    if selection=='spot':
        assert await scalar(intake.db_factory,'select active_spot_id from sessions') is not None


async def test_source_port_worker_uses_fake_without_local_source_paths(intake,monkeypatch):
    deps=replace(intake,source_storage=FakeStorage())
    await read_once(deps,monkeypatch)
    assert not (intake.data_dir/'sources').exists()
    ref=await scalar(intake.db_factory,"select raw->>'source_ref' from hands")
    assert SourceRef.model_validate_json(ref).backend=='fake'


async def test_concurrent_persistence_is_one_owned_hand(intake):
    job=await start(intake)
    result=process_pokerdom(observed(),nickname='synthetic-alice')
    assert result.raw is not None and result.raw.vision is not None
    result.raw.vision.image_hash=job.payload['image_sha256']
    result.raw.source_ref=SourceRef.model_validate(job.payload['source_ref']).encode()
    async def persist():
        async with intake.db_factory() as db:
            hand_id=await save_reading(db,job,result)
            await asyncio.sleep(.01)
            await db.commit()
            return hand_id
    assert len(set(await asyncio.gather(persist(),persist())))==1
    assert await scalar(intake.db_factory,'select count(*) from hands')==1


async def test_cancel_does_not_delete_referenced_history_source(intake,monkeypatch):
    job,_=await read_once(intake,monkeypatch)
    ref=SourceRef.model_validate(job.payload['source_ref'])
    async with intake.db_factory() as db:
        player=await PlayersRepo(db).find(USER);assert player is not None
        await PlayersRepo(db).set_pending_input(player.id,{'kind':'pokerdom_image_nickname',
            'owner_id':player.id,'source_ref':ref.model_dump(),'image_sha256':ref.sha256,'created_at':0})
        await db.commit()
    await cancel_image(intake,USER)
    assert LocalSourceStorage(intake.data_dir).read(job.player_id,ref,MAX_IMAGE_BYTES)==IMAGE


async def test_unknown_showdown_excluded_only_from_outcome_sample(intake,monkeypatch):
    from tests.test_pokerdom_nickname import v2
    from tests.test_pokerdom_vision import showdown_table
    reading=v2(showdown_table(explicit=False))
    reading.format_observation=observed().format_observation
    monkeypatch.setattr('harness.worker.pokerdom.read_pokerdom',AsyncMock(return_value=(reading,None)))
    job=await start(intake);await run_job(job,worker(intake))
    async with intake.db_factory() as db:
        actions=await _measure(db,job.player_id,stat=FrequencyStat.VPIP,subject=Subject.HERO,opponent_id=None,position=None,window=Window())
        outcome=await _measure(db,job.player_id,stat=FrequencyStat.SHOWDOWN,subject=Subject.HERO,opponent_id=None,position=None,window=Window())
        assert actions.denominator==1 and outcome.denominator==0
        hand=await db.scalar(select(Hand));assert hand is not None
        assert hand.raw['collected']==[] and hand.raw['showdowns']==[]
        assert len(await HandsRepo(db).player_canonical(job.player_id,Window()))==1
        analysis=await AnalysesRepo(db).get_by_hand(hand.id);assert analysis is not None
        context=await _agent_context(db,analysis.id,job.player_id,job.session_id)
        assert context is not None and 'условный исход' in context.text


async def test_source_failure_never_creates_history(intake,monkeypatch):
    job=await start(intake)
    ref=SourceRef.model_validate(job.payload['source_ref'])
    (intake.data_dir/'sources'/ref.key).write_bytes(b'corrupt')
    stub=AsyncMock();monkeypatch.setattr('harness.worker.pokerdom.read_pokerdom',stub)
    await run_job(job,worker(intake))
    stub.assert_not_awaited()
    assert await scalar(intake.db_factory,'select count(*) from hands')==0
    assert await scalar(intake.db_factory,'select status from jobs')=='failed'


async def test_tool_checks_current_discussion_before_and_after_new(intake,monkeypatch):
    from harness.calcs.routing import CalcRun, tools_for
    job,_=await read_once(intake,monkeypatch)
    async with intake.db_factory() as db:
        analysis_id=await db.scalar(text('select id from analyses'))
        state=CalcRun(max_calcs=8,analysis_id=analysis_id,sizes_from=frozenset())
        tools={tool.name:tool for tool in tools_for(db,job.player_id,job.session_id,state)}
        # HU has no supported chart: the real tool returns a capability refusal,
        # not a fabricated range. Current context still resolves this hand.
        range_tool: Any = tools['opponent_range'].function
        output=await range_tool(opponent_position='BB',street=Street.PREFLOP)
        assert output is not None
        await db.commit()
        await handle_new_session(intake,USER)
        output=await range_tool(opponent_position='BB',street=Street.PREFLOP)
        assert 'Текущая рука изменилась' in output


async def test_legacy_job_source_is_guarded_bound_once_and_checkpointed(intake,monkeypatch):
    import hashlib

    from harness.memory.models import Job
    async with intake.db_factory() as db:
        player=await PlayersRepo(db).find(USER);assert player is not None
        path=intake.data_dir/'vision'/str(player.id)/('b'*32+'.image')
        path.parent.mkdir(parents=True);path.write_bytes(IMAGE)
        db.add(Job(type='pokerdom_read',player_id=player.id,session_id=None,max_attempts=1,
            payload={'source_file':str(path),'image_sha256':hashlib.sha256(IMAGE).hexdigest(),
                     'nickname':'synthetic-alice','original_document':True}))
        await db.commit()
    stub=AsyncMock(return_value=(observed(),None));monkeypatch.setattr('harness.worker.pokerdom.read_pokerdom',stub)
    job=await intake.queue.claim('legacy');assert job is not None
    await run_job(job,worker(intake))
    assert await scalar(intake.db_factory,'select count(*) from hands')==1
    assert await scalar(intake.db_factory,'select session_id from jobs') is not None
    assert await scalar(intake.db_factory,"select payload->'source_ref'->>'backend' from jobs")=='local'
    assert path.read_bytes()==IMAGE


def test_migration_0018_up_down_and_pg_backup_restore():
    from alembic import command
    from sqlalchemy import create_engine
    from testcontainers.community.postgres import PostgresContainer

    from tests.conftest import alembic_config
    with PostgresContainer('postgres:16-alpine') as container:
        dsn=container.get_connection_url(driver='psycopg')
        cfg=alembic_config(dsn);command.upgrade(cfg,'0017')
        engine=create_engine(dsn)
        try:
            with engine.begin() as db:
                p=db.execute(text('insert into players(tg_user_id) values (6789) returning id')).scalar_one()
                db.execute(text("insert into sessions(player_id,started_at,title) values (:p,now(),'synthetic')"),{'p':p})
            command.upgrade(cfg,'head')
            with engine.begin() as db:
                assert db.execute(text('select context_revision from sessions')).scalar_one()==0
                assert db.execute(text("select indexname from pg_indexes where indexname='ix_hands_image_hash'")).scalar_one()
                db.execute(text('update sessions set context_revision=3'))
            # Snapshot + restore in a DIFFERENT disposable database, never live DB.
            code,_=container.exec(['sh','-c','pg_dump -U test -d test > /tmp/vision-snapshot.sql && createdb -U test restored && psql -v ON_ERROR_STOP=1 -U test -d restored -f /tmp/vision-snapshot.sql >/dev/null'])
            assert code==0
            code,output=container.exec(['psql','-U','test','-d','restored','-tAc','select context_revision from sessions'])
            assert code==0 and isinstance(output,bytes) and output.strip()==b'3'
            command.downgrade(cfg,'0017')
            with engine.begin() as db:
                assert db.execute(text('select count(*) from sessions')).scalar_one()==1
                assert db.execute(text("select count(*) from information_schema.columns where table_name='sessions' and column_name='context_revision'")).scalar_one()==0
            command.upgrade(cfg,'head')
        finally:
            engine.dispose()


async def test_duplicate_missing_saved_source_stops_without_second_ocr_or_hand(intake,monkeypatch):
    job,stub=await read_once(intake,monkeypatch)
    ref=SourceRef.model_validate(job.payload['source_ref'])
    (intake.data_dir/'sources'/ref.key).unlink()
    await enqueue_image(intake,USER,IMAGE,'',original=True)
    duplicate=await intake.queue.claim('missing-source');assert duplicate is not None
    await run_job(duplicate,worker(intake))
    assert stub.await_count==1
    assert await scalar(intake.db_factory,'select count(*) from hands')==1
    assert await scalar(intake.db_factory,'select status from jobs order by id desc limit 1')=='failed'


@pytest.mark.parametrize('shown',[False,True])
async def test_multiway_showdown_observability_keeps_known_folded_seat(intake,monkeypatch,shown):
    from tests.test_pokerdom_friendly import six_players
    reading=six_players(river=True)
    reading.format_observation=observed().format_observation
    if shown:
        reading.players[1].showdown_shown=True
        reading.players[2].showdown_shown=True
    monkeypatch.setattr('harness.worker.pokerdom.read_pokerdom',AsyncMock(return_value=(reading,None)))
    job=await start(intake);await run_job(job,worker(intake))
    async with intake.db_factory() as db:
        async def measured(subject,stat,position=None):
            return await _measure(db,job.player_id,subject=subject,stat=stat,
                opponent_id=None,position=position,window=Window())
        field=await measured(Subject.FIELD,FrequencyStat.SHOWDOWN)
        hero=await measured(Subject.HERO,FrequencyStat.SHOWDOWN)
        folded=await measured(Subject.FIELD,FrequencyStat.SHOWDOWN,'HJ')
        assert (folded.numerator,folded.denominator)==(0,1)
        assert (field.numerator,field.denominator)==((1,2) if shown else (0,1))
        assert (hero.numerator,hero.denominator)==((1,1) if shown else (0,0))
        actions=await measured(Subject.FIELD,FrequencyStat.VPIP)
        assert actions.denominator==5


async def test_intake_and_new_use_one_player_then_session_lock_order(intake,monkeypatch):
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import AsyncEngine
    async with intake.db_factory() as db:
        player=await PlayersRepo(db).find(USER);assert player is not None
        player.pokerdom_nickname='synthetic-alice'
        await SessionsRepo(db).active_or_create(player.id)
        await db.commit()
    engine=intake.db_factory.kw['bind'];assert isinstance(engine,AsyncEngine)
    intake_holds_lock=asyncio.Event();new_attempts_lock=asyncio.Event()
    original=SessionsRepo.active_or_create
    async def barrier(repo,player_id):
        result=await original(repo,player_id)
        task=asyncio.current_task()
        if task is not None and task.get_name()=='racing-intake':
            intake_holds_lock.set()
            await asyncio.wait_for(new_attempts_lock.wait(),5)
        return result
    def before_query(conn,cursor,statement,parameters,context,executemany):
        task=asyncio.current_task()
        if task is not None and task.get_name()=='racing-new' and 'pg_advisory_xact_lock' in statement:
            new_attempts_lock.set()
    monkeypatch.setattr(SessionsRepo,'active_or_create',barrier)
    event.listen(engine.sync_engine,'before_cursor_execute',before_query)
    try:
        upload=asyncio.create_task(enqueue_image(intake,USER,IMAGE,'',original=True),name='racing-intake')
        await asyncio.wait_for(intake_holds_lock.wait(),5)
        new=asyncio.create_task(handle_new_session(intake,USER),name='racing-new')
        results=await asyncio.wait_for(asyncio.gather(upload,new,return_exceptions=True),10)
        assert not any(isinstance(result,BaseException) for result in results),repr(results)
    finally:
        event.remove(engine.sync_engine,'before_cursor_execute',before_query)
    assert await scalar(intake.db_factory,'select count(*) from jobs')==1
    assert await scalar(intake.db_factory,'select count(*) from sessions where closed_at is null')==1


async def test_same_owner_exact_image_source_is_stable_before_worker(intake):
    job=await start(intake)
    await intake.queue.complete(job.id,worker_id='test-pd-worker')
    await enqueue_image(intake,USER,IMAGE,'',original=True)
    async with intake.db_factory() as db:
        refs=(await db.execute(text("select payload->'source_ref' from jobs order by id"))).scalars().all()
    assert refs[0]==refs[1]


async def test_intake_db_rollback_compensates_unreferenced_source(intake,monkeypatch):
    async def fail(*args):
        raise RuntimeError('synthetic DB transaction failure after put')
    monkeypatch.setattr(SessionsRepo,'select_context',fail)
    with pytest.raises(RuntimeError,match='transaction failure'):
        await enqueue_image(intake,USER,IMAGE,'',original=True)
    store=LocalSourceStorage(intake.data_dir)
    async with intake.db_factory() as db:
        player=await PlayersRepo(db).find(USER);assert player is not None
        assert store.orphan_candidates(player.id,float('inf'))==[]
    assert await scalar(intake.db_factory,'select count(*) from jobs')==0
    assert await scalar(intake.db_factory,'select pending_input from players') is None


async def test_crash_intent_recovery_is_owned_grace_and_ref_aware(intake,monkeypatch):
    import time

    from harness.platform.source_lifecycle import recover_sources
    job,_=await read_once(intake,monkeypatch)
    store=LocalSourceStorage(intake.data_dir)
    abandoned=store.put(job.player_id,b'crashed input','image/png','crashed-operation')
    foreign=store.put(job.player_id+100,b'other owner','image/png','other-operation')
    async with intake.db_factory() as db:
        assert await recover_sources(db,store,job.player_id,before=0)==0
        assert await recover_sources(db,store,job.player_id,before=time.time()+1)==1
        await db.commit()
        hand=await db.scalar(select(Hand));assert hand is not None
        image_ref=SourceRef.model_validate_json(hand.raw['source_ref'])
        reading_ref=SourceRef.model_validate_json(hand.raw['vision']['reading_ref'])
    assert store.read(job.player_id,image_ref,MAX_IMAGE_BYTES)==IMAGE
    assert store.read(job.player_id,reading_ref,20*1024*1024)
    assert store.read(job.player_id+100,foreign,100)==b'other owner'
    with pytest.raises(FileNotFoundError):store.read(job.player_id,abandoned,100)


@pytest.mark.parametrize('status,attempts,maximum,protected',[
    ('queued',0,1,True),('running',1,1,True),('awaiting_user',1,1,True),
    ('failed',1,2,True),('failed',1,1,False),('done',1,1,False)])
async def test_source_recovery_protects_active_and_permitted_retry_only(intake,status,attempts,maximum,protected):
    from harness.memory.models import Job
    from harness.platform.source_lifecycle import recover_sources
    store=LocalSourceStorage(intake.data_dir)
    async with intake.db_factory() as db:
        player=await PlayersRepo(db).find(USER);assert player is not None
        ref=store.put(player.id,b'job input','image/png','job-source')
        db.add(Job(type='pokerdom_read',player_id=player.id,session_id=None,status=status,
            attempts=attempts,max_attempts=maximum,payload={'source_ref':ref.model_dump()}))
        await db.commit()
        count=await recover_sources(db,store,player.id,before=float('inf'))
        await db.commit()
    assert count==int(not protected)
    if protected:
        assert store.read(player.id,ref,100)==b'job input'
    else:
        assert await scalar(intake.db_factory,"select payload->>'source_released' from jobs")=='true'
        assert await scalar(intake.db_factory,"select payload->'source_ref' from jobs") is None


async def test_legacy_duplicate_job_repoints_to_history_then_releases_extra(intake,monkeypatch):
    job,stub=await read_once(intake,monkeypatch)
    store=LocalSourceStorage(intake.data_dir)
    extra=store.put(job.player_id,IMAGE,'image/png','legacy-random-intake')
    await intake.queue.enqueue(type='pokerdom_read',player_id=job.player_id,session_id=job.session_id,
        payload={'source_ref':extra.model_dump(),'image_sha256':extra.sha256,
                 'nickname':'synthetic-alice','original_document':True})
    duplicate=await intake.queue.claim('legacy-extra');assert duplicate is not None
    await run_job(duplicate,worker(intake))
    assert stub.await_count==1
    assert await scalar(intake.db_factory,'select count(*) from hands')==1
    async with intake.db_factory() as db:
        refs=(await db.execute(text("select payload->'source_ref' from jobs order by id"))).scalars().all()
    assert refs[0]==refs[1]
    assert store.read(job.player_id,SourceRef.model_validate(refs[0]),MAX_IMAGE_BYTES)==IMAGE
    with pytest.raises(FileNotFoundError):store.read(job.player_id,extra,MAX_IMAGE_BYTES)
