"""CALL threshold evidence is not profit, equity or opponent-frequency evidence."""
from dataclasses import replace

import pytest

from harness.agent.run import AgentDraft, _draft_violations
from harness.analysis import analyze_hand
from harness.contracts import Street
from harness.explanation.agent_checks import CallThresholdEvidence, agent_context
from tests.test_agent_checks import _check
from tests.test_hand_replay import _postflop_hand

PROOF=(CallThresholdEvidence(7,Street.TURN,16.875),)
PHRASE='Порог эквити для окупаемости колла — 17%.'


@pytest.mark.parametrize('text',[
    PHRASE,'Порог эквити для окупаемости колла: 16.9%.',
    'Для окупаемости колла нужна эквити 17%.',
    'Порог эквити для окупаемости колла — 17%; банк уже включает ставку.',
])
def test_only_typed_call_threshold_supports_description(text):
    assert _check(text,allowed=frozenset({17,16.9}),call_thresholds=PROOF)==[]
    assert _check(text,allowed=frozenset({17,16.9}))


@pytest.mark.parametrize('text',[
    'Порог эквити для окупаемости колла — 18%.',
    'Колл окупается, у него 17% фолдов.',
    'Колл в плюсе при 17%.',
    'Колл выгоден при пороге 17%.',
    'Колл даёт EV при 17%.',
    PHRASE+' Колл в плюсе.',
    PHRASE+' Колл выгоден.',
    PHRASE+' Колл даёт EV.',
    'Порог эквити для окупаемости колла — 17%, и колл в плюсе.',
    'У вашей руки 17% эквити, колл окупается.',
    'У вашей руки 17% эквити.',
    'Эквити вашей руки — 17% (допущение).',
    'Порог эквити для окупаемости колла — 17%, соперник фолдит в 17%.',
    'Диапазон соперника слабый, поэтому колл окупается при 17%.',
])
def test_same_number_does_not_authorize_neighbor_predictions_or_profit(text):
    assert _check(text,allowed=frozenset({17,18}),call_thresholds=PROOF)


def test_full_draft_check_receives_call_proof_separately_from_allowed_numbers():
    en=_postflop_hand();context=agent_context(analyze_hand(en),en)
    assert context.call_thresholds
    context=replace(context,allowed=context.allowed|{17},call_thresholds=PROOF)
    assert _draft_violations(AgentDraft(answer=PHRASE),[],context,frozenset(),question="Перед коллом на тёрне")==([],[])
    assert _draft_violations(AgentDraft(answer=PHRASE),[],replace(context,call_thresholds=()),frozenset())[0]
    # Generic or frequency thresholds never become CALL evidence.
    assert _check(PHRASE,allowed=frozenset({17}),calc_numbers=frozenset({17}),
                  threshold_numbers=frozenset({17}),frequency_numbers=frozenset({17}))


MULTI=(CallThresholdEvidence(1,Street.PREFLOP,32.125),
       CallThresholdEvidence(4,Street.FLOP,22.125),
       CallThresholdEvidence(9,Street.RIVER,16.875))
RIVER_Q='Посчитай шансы банка перед моим коллом на ривере этой руки'


def multi_context():
    en=_postflop_hand()
    context=agent_context(analyze_hand(en),en)
    return replace(context,allowed=context.allowed|{17,22,32},call_thresholds=MULTI)


@pytest.mark.parametrize('pct,passes',[(17,True),(32,False),(22,False)])
def test_latest_river_question_never_uses_another_call_street(pct,passes):
    draft=AgentDraft(answer=f'Порог эквити для окупаемости колла — {pct}%.')
    found,_=_draft_violations(draft,[],multi_context(),frozenset(),question=RIVER_Q)
    assert bool(found) is not passes


@pytest.mark.parametrize('question,evidence',[
    (None,MULTI),('Какой порог колла?',MULTI),('Что произошло на ривере?',MULTI),
    ('Порог колла до флопа?',MULTI),('Порог колла на флопе и ривере?',MULTI),
    (RIVER_Q,MULTI[:-1]),
    (RIVER_Q,(*MULTI,CallThresholdEvidence(10,Street.RIVER,16.875))),
    (RIVER_Q,()),
])
def test_missing_unknown_or_ambiguous_call_target_has_no_permission(question,evidence):
    context=replace(multi_context(),call_thresholds=evidence)
    assert _draft_violations(AgentDraft(answer=PHRASE),[],context,frozenset(),question=question)[0]


@pytest.mark.parametrize('suffix',[' Колл в плюсе.',' Колл выгоден.',' Соперник фолдит в 17%.'])
def test_target_selection_does_not_whitelist_neighbor_statements(suffix):
    assert _draft_violations(AgentDraft(answer=PHRASE+suffix),[],multi_context(),frozenset(),question=RIVER_Q)[0]


def test_low_level_does_not_accept_an_unselected_bag_of_call_proofs():
    assert _check(PHRASE,allowed=frozenset({17}),call_thresholds=MULTI)
