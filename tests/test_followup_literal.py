"""Synthetic explicit-range and exact-call context regressions. No model API."""
from typing import Any

import pytest
from pydantic import ValidationError

from harness import calcs
from harness.agent.literal_range import canonical_literal, user_literal_ranges
from harness.analysis import analyze_hand
from harness.analysis.charts import parse_range
from harness.analysis.opponent_range import hero_cards, live_combos
from harness.analysis.tools.pot_odds import required_equity
from harness.calcs.routing import CalcRun, tools_for
from harness.contracts import ActionKind, EquityVsRangeParams, EquityVsRangeResult, Street
from harness.explanation.agent_checks import agent_context
from harness.explanation.question import agent_digest
from harness.memory.repos import AnalysesRepo, SessionsRepo
from tests.test_calcs import _player_with_session, stored_analysis
from tests.test_hand_replay import _postflop_hand
from tests.test_pokerdom_telegram import intake

__all__ = ["intake"]


@pytest.mark.parametrize('text',['Против диапазона SB QQ+, AK','диапазон qq+ ,  ak','диапазон QQ+,AKs,AKo'])
def test_literal_quote_normalization(text):
    assert user_literal_ranges([text])==frozenset([canonical_literal('QQ+,AK')])
    assert all(weight==1 for weight in parse_range(canonical_literal('QQ+, AK')).weights.values())
    assert user_literal_ranges(['покажи диапазон SB по чарту QQ+, AK'])==frozenset()
    assert user_literal_ranges(['какое эквити против Васи'])==frozenset()


@pytest.mark.parametrize('patch',[{'mode':'literal'}, {'mode':'chart','literal_source':'QQ+'},
 {'mode':'literal','literal_source':'QQ+','street':Street.FLOP},
 {'mode':'literal','literal_source':'QQ+','added':'AKs'},
 {'mode':'literal','literal_source':'QQ+','hands':''},
 {'mode':'literal','literal_source':'QQ+','analysis_id':0}])
def test_source_modes_reject_ambiguous_parameters(patch):
    data={'analysis_id':1,'opponent_position':'BB','street':Street.PREFLOP,'hands':'QQ+'}
    with pytest.raises(ValidationError):EquityVsRangeParams(**(data|patch))


async def test_literal_uses_all_weights_no_chart_and_exact_dead_cards(db,monkeypatch):
    player,sid=await _player_with_session(db,tg_user_id=45001)
    aid=await stored_analysis(db,session_id=sid)
    hand=await AnalysesRepo(db).enriched_hand(aid,player);assert hand is not None
    def no_chart(*args,**kwargs):raise AssertionError('literal must not select a chart')
    monkeypatch.setattr(calcs,'equity_spot',no_chart)
    seen=[]
    original=calcs.equity_vs_range
    def count(hero,rng,board,**kwargs):
        seen.append(rng)
        return original(hero,rng,board,**kwargs)
    monkeypatch.setattr(calcs,'equity_vs_range',count)
    notation=canonical_literal('QQ+,AK')
    result=await calcs.run(db,player,EquityVsRangeParams(analysis_id=aid,opponent_position='BB',
        street=Street.PREFLOP,hands=notation,mode='literal',literal_source=notation))
    assert seen and set(seen[0].weights.values())=={1.0}
    assert result.range_source=='user_literal'
    assert result.live_combos==live_combos(parse_range(notation),hero_cards(hand.hand))
    digest=agent_digest(result)
    assert 'Явный диапазон пользователя (допущение)' in digest.text
    assert 'Веса взяты из записи пользователя' in digest.text
    assert 'Диапазон, суженный агентом' not in digest.text
    assert 'сужение диапазона агентом' not in digest.text
    with pytest.raises(ValueError):await calcs.run(db,player+1,EquityVsRangeParams(
        analysis_id=aid,opponent_position='BB',street=Street.PREFLOP,hands=notation,mode='literal',literal_source=notation))


async def test_tool_literal_current_owner_quote_guard_and_no_chart_fallback(db):
    player,sid=await _player_with_session(db,tg_user_id=45002)
    aid=await stored_analysis(db,session_id=sid)
    revision=await SessionsRepo(db).select_context(sid,player)
    await SessionsRepo(db).publish_analysis(sid,player,revision,aid)
    state=CalcRun(max_calcs=8,sizes_from=frozenset(),analysis_id=aid)
    state.user_literal_ranges=user_literal_ranges(['диапазон QQ+, AK'])
    state.literal_required=True
    tool: Any=next(t.function for t in tools_for(db,player,sid,state) if t.name=='equity_vs_range')
    assert 'mode=literal' in await tool(opponent_position='BB',street=Street.PREFLOP,hands='QQ+,AK')
    assert 'явно написанная' in await tool(opponent_position='BB',street=Street.PREFLOP,hands='22',mode='literal')
    assert 'только до флопа' in await tool(opponent_position='BB',street=Street.FLOP,hands='QQ+,AK',mode='literal')
    output=await tool(opponent_position='BB',street=Street.PREFLOP,hands='qq+, ak',mode='literal')
    assert not isinstance(output,str)
    assert state.narrowings==[]
    await SessionsRepo(db).select_context(sid,player)
    assert 'Текущая рука изменилась' in await tool(opponent_position='BB',street=Street.PREFLOP,hands='QQ+,AK',mode='literal')


def test_exact_call_context_labels_pre_action_bank_and_registers_numbers():
    en=_postflop_hand()
    analysis=analyze_hand(en)
    # Independent unequal money values make pre-bet vs pre-call visible.
    point=en.report.decision_points[0].model_copy(deep=True)
    point.to_call=317;point.pot_before=1729
    point.action.kind=ActionKind.CALL
    en.report.decision_points=[point]
    context=agent_context(analysis,en)
    line=next(line for line in context.text.splitlines() if line.startswith('Точный колл'))
    assert 'банк перед действием героя (ставка соперника уже включена) 1729 фишек' in line
    assert 'доплата героя 317 фишек' in line and 'банк после колла 2046 фишек' in line
    assert '1412 фишек' not in line  # pre-bet bank must not be labelled pre-call
    assert {1729,317,2046,round(100*required_equity(317,1729),8)} <= context.allowed


async def test_persisted_vision_two_followups_use_exact_context_and_literal_tool(intake,monkeypatch):
    from pydantic_ai.models.function import FunctionModel
    from sqlalchemy import text

    from harness.agent.history import dump_history, load_history
    from harness.agent.run import run_agent_turn
    from harness.contracts.pokerdom_vision import PokerdomEvent
    from harness.platform.llm import LLM
    from harness.worker.pipeline import _agent_context, run_job
    from tests.test_agent import _TurnScript
    from tests.test_pokerdom_nickname import v2
    from tests.test_pokerdom_telegram import observed, start
    from tests.test_pokerdom_vision import showdown_table
    from tests.test_question_routing import _TEST_CFG
    from tests.test_vision_autosave import worker
    source=showdown_table(explicit=False)
    source.events=source.events[:-2]
    source.events += [PokerdomEvent(number=len(source.events)+i+1,player=actor,
        street=Street.RIVER,token=token,amount=amount,amount_text=str(amount) if amount else None)
        for i,(actor,token,amount) in enumerate([('B','B',200),('A','C',200)])]
    source.pot_after_street[Street.RIVER]=820
    reading=v2(source);reading.format_observation=observed().format_observation
    async def saved_ocr(*args,**kwargs):return reading,None
    monkeypatch.setattr('harness.worker.pokerdom.read_pokerdom',saved_ocr)
    job=await start(intake);await run_job(job,worker(intake))
    async with intake.db_factory() as db:
        state=await SessionsRepo(db).agent_state(job.session_id)
        assert state.last_analysis_id is not None
        context=await _agent_context(db,state.last_analysis_id,job.player_id,job.session_id)
        assert context is not None and 'Точный колл, ривер' in context.text
        trace_id=await db.scalar(text('select id from traces limit 1'))
        await db.commit()
        # Model double chooses the production tool by its real schema; no API.
        script=_TurnScript([([], 'Точный банк перед вашим действием уже включает ставку соперника.'),
            ([('equity_vs_range',{'opponent_position':'BB','street':'preflop','hands':'QQ+, AK','mode':'literal'})],
             'Расчёт для диапазона пользователя готов (допущение).')])
        llm=LLM(_TEST_CFG,intake.db_factory,model_override=FunctionModel(script))
        first=await run_agent_turn(db,llm,player_id=job.player_id,session_id=job.session_id,
            trace_id=trace_id,text='Шансы банка перед моим коллом на ривере?',history=[],context=context,
            analysis_id=state.last_analysis_id,system_prompt='Синтетический системный промпт')
        assert first.state=='prose' and first.from_context
        second=await run_agent_turn(db,llm,player_id=job.player_id,session_id=job.session_id,
            trace_id=trace_id,text='Эквити до флопа против диапазона BB QQ+, AK',
            history=load_history(dump_history(first.saved_messages(None))),context=context,
            analysis_id=state.last_analysis_id,system_prompt='Синтетический системный промпт')
        assert second.state=='prose'
        assert isinstance(second.results[0],EquityVsRangeResult)
        assert second.results[0].range_source=='user_literal'
        assert second.calls[0]['params']['analysis_id']==state.last_analysis_id
        assert second.calls[0]['params']['mode']=='literal'
        assert await db.scalar(text('select count(*) from hands'))==1
        assert await db.scalar(text('select count(*) from analyses'))==1
