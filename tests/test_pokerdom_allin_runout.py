"""Synthetic all-in completion matrix; source amounts stay immutable, no API."""
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select, text

from harness.contracts.pokerdom_vision import PokerdomEvent, PokerdomPlayer, PokerdomReading
from harness.contracts.raw import Street
from harness.memory.models import Hand
from harness.memory.repos import AnalysesRepo, SessionsRepo
from harness.parsers.pokerdom_vision import process_pokerdom
from harness.presentation.pokerdom_vision import pokerdom_messages
from harness.worker.pipeline import _agent_context, run_job
from tests.test_pokerdom_nickname import v2
from tests.test_pokerdom_telegram import intake, observed, start
from tests.test_vision_autosave import worker

__all__ = ['intake']


def allin(stage=Street.PREFLOP, multiway=False):
    players=[PokerdomPlayer(row_id='A',nickname='synthetic-alice',stack_shown=1010,button_shown=True,cards=['Kh','Kd']),
             PokerdomPlayer(row_id='B',nickname='synthetic-bob',stack_shown=810,button_shown=False,cards=['Qh','Qd'])]
    sequence=[('A','Ante',10,Street.PREFLOP),('B','Ante',10,Street.PREFLOP)]
    if multiway:
        players.append(PokerdomPlayer(row_id='C',nickname='synthetic-carol',stack_shown=610,button_shown=False,cards=['Jh','Jd']))
        sequence += [('C','Ante',10,Street.PREFLOP),('B','SB',100,Street.PREFLOP),('C','BB',200,Street.PREFLOP),
                     ('A','R',1000,Street.PREFLOP),('B','C',700,Street.PREFLOP),('C','C',400,Street.PREFLOP),('A','UB',200,Street.PREFLOP)]
    else:
        sequence += [('A','SB',100,Street.PREFLOP),('B','BB',200,Street.PREFLOP)]
        if stage == Street.PREFLOP:
            sequence += [('A','R',900,stage),('B','C',600,stage),('A','UB',200,stage)]
        else:
            sequence += [('A','C',100,Street.PREFLOP),('B','Check',None,Street.PREFLOP)]
            if stage == Street.TURN:
                sequence += [('B','Check',None,Street.FLOP),('A','Check',None,Street.FLOP)]
            sequence += [('B','B',600,stage),('A','C',600,stage)]
    pot=2230 if multiway else 1620
    pots: dict[Street, int | None]={s:(420 if not multiway and list(Street).index(s)<list(Street).index(stage) else pot) for s in Street}
    return PokerdomReading(is_hand=True,complete=True,cropped=False,unit='chips',stack_semantics='before_forced',players=players,
        events=[PokerdomEvent(number=i,player=p,token=t,amount=a,amount_text=str(a) if a is not None else None,street=s)
                for i,(p,t,a,s) in enumerate(sequence,1)],
        board={Street.FLOP:['2c','3d','7s'],Street.TURN:['8c'],Street.RIVER:['9h']},pot_after_street=pots)


def process(reading):
    return process_pokerdom(reading,nickname='synthetic-alice')


@pytest.mark.parametrize('stage',[Street.PREFLOP,Street.FLOP,Street.TURN])
@pytest.mark.parametrize('version',[1,2])
@pytest.mark.parametrize('postflop_return',[False,True])
def test_complete_runout_no_invented_actions_one_replay(stage,version,postflop_return,monkeypatch):
    reading=allin(stage)
    if postflop_return and stage != Street.PREFLOP:
        first=len(reading.events)-1
        reading.events=reading.events[:-2]+[
            PokerdomEvent(number=i,player=p,token=t,amount=a,amount_text=str(a) if a is not None else None,street=stage)
            for i,(p,t,a) in enumerate([('B','Check',None),('A','B',800),('B','C',600),('A','UB',200)],first)]
    reading=reading if version==1 else v2(reading)
    before=reading.model_dump();calls=[]
    from harness.parsers import pokerdom_vision as adapter
    original=adapter.enrich
    def counted(hand):
        calls.append(hand);return original(hand)
    monkeypatch.setattr(adapter,'enrich',counted)
    result=process(reading)
    assert result.replay_status=='passed',result.errors
    assert len(calls)==1 and reading.model_dump()==before
    assert result.raw is not None and result.enriched is not None
    assert len(result.raw.actions)==len([e for e in reading.events if e.token in {'R','B','C','Check','Fold'}])
    assert all(c.status=='passed' for c in result.checks if c.name.startswith('pot_after.'))
    assert result.enriched.report.final_pot==1620
    assert result.enriched.report.stacks_end=={'Hero':1820,'P2' if version==2 else 'B':0}
    assert result.outcome_basis=='visible_cards' and not result.outcome_confirmed
    assert not result.raw.showdowns and not result.raw.collected
    story='\n'.join(m.text for m in pokerdom_messages(result))
    assert 'Ривер 9♥' in story and 'Ривер: не сыгран' not in story
    assert 'условный результат' in story


def test_multiway_side_pots_and_uncalled_return_once():
    result=process(v2(allin(multiway=True)))
    assert result.replay_status=='passed',result.errors
    assert result.enriched is not None and result.raw is not None
    assert sorted(p.amount for p in result.enriched.report.side_pots)==[400,1830]
    assert result.enriched.report.final_pot==2230
    assert result.enriched.report.stacks_end=={'Hero':2430,'P2':0,'P3':0}
    assert len(result.raw.uncalled)==1 and result.raw.uncalled[0].amount==200


@pytest.mark.parametrize('kind',['missing_flop','missing_turn','missing_river','collision','overspend','bad_ub','unmatched','wrong_order','two_stacks','foldout_ghost','partial_side_betting'])
def test_neighbouring_incomplete_or_illegal_inputs_remain_refused(kind):
    reading=allin()
    if kind.startswith('missing_'):
        reading.board.pop(Street(kind.removeprefix('missing_')))
    elif kind=='collision':reading.board[Street.RIVER]=['Kh']
    elif kind=='overspend':reading.events[4].amount=901;reading.events[4].amount_text='901'
    elif kind=='bad_ub':reading.events[-1].amount=199;reading.events[-1].amount_text='199'
    elif kind=='unmatched':reading.events=reading.events[:5]
    elif kind=='wrong_order':
        reading.events[4].player,reading.events[5].player=reading.events[5].player,reading.events[4].player
    elif kind=='two_stacks':reading.players[0].stack_shown=2010;reading.players[1].stack_shown=1810
    elif kind=='foldout_ghost':
        from tests.test_pokerdom_vision import table
        board=reading.board;reading=table();reading.board=board
    else:
        reading=allin(multiway=True);reading.players[0].stack_shown=2010;reading.players[1].stack_shown=1810
    result=process(v2(reading))
    assert result.replay_status!='passed',kind
    assert result.errors,kind
    assert result.calculated_payouts=={}


def test_forced_blind_allin_with_return_and_no_voluntary_action():
    reading=allin();reading.players[0].stack_shown=110;reading.players[1].stack_shown=210
    reading.events=reading.events[:4]+[PokerdomEvent(number=5,player='B',token='UB',amount=100,amount_text='100',street=Street.PREFLOP)]
    reading.pot_after_street=dict.fromkeys(Street,220)
    result=process(v2(reading))
    assert result.replay_status=='passed',result.errors
    assert result.enriched is not None and result.enriched.report.final_pot==220


async def test_intake_worker_real_persist_analysis_current_context(intake,monkeypatch):
    reading=v2(allin(multiway=True));reading.format_observation=observed().format_observation
    stub=AsyncMock(return_value=(reading,None));monkeypatch.setattr('harness.worker.pokerdom.read_pokerdom',stub)
    job=await start(intake);deps=worker(intake);await run_job(job,deps)
    async with intake.db_factory() as db:
        assert await db.scalar(text('select count(*) from hands'))==1
        assert await db.scalar(text('select count(*) from analyses'))==1
        hand=await db.scalar(select(Hand));assert hand is not None
        analysis=await AnalysesRepo(db).get_by_hand(hand.id);assert analysis is not None
        state=await SessionsRepo(db).agent_state(job.session_id);assert state.last_analysis_id==analysis.id
        context=await _agent_context(db,analysis.id,job.player_id,job.session_id)
        assert context is not None and 'гипотеза' in context.text
        assert hand.raw['collected']==[] and hand.raw['showdowns']==[]
    assert stub.await_count==1
