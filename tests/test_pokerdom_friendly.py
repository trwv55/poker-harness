"""Synthetic user-facing narratives; never model calls or real hand fixtures."""
import pytest

from harness.contracts.pokerdom_vision import PokerdomEvent, PokerdomPlayer
from harness.contracts.raw import Street
from harness.parsers.pokerdom_vision import process_pokerdom
from harness.presentation.pokerdom_vision import pokerdom_messages, pokerdom_report
from tests.test_pokerdom_nickname import v2
from tests.test_pokerdom_vision import table


def six_players(river=False):
    source=table()
    names=['synthetic-button','synthetic-bob','synthetic-alice','synthetic-early','synthetic-third','synthetic-late']
    source.players=[PokerdomPlayer(row_id=f'P{i}',nickname=name,stack_shown=7000,button_shown=i==0,
        cards=['Ac','Qc'] if i==2 else ['Qh','Jd'] if i==1 and river else []) for i,name in enumerate(names)]
    events=[]
    def add(player,token,amount=None,street=Street.PREFLOP):
        events.append(PokerdomEvent(number=len(events)+1,player=f'P{player}',token=token,amount=amount,
            amount_text=str(amount) if amount is not None else None,street=street))
    for i in (2,3,4,5,0,1):add(i,'Ante',20)
    add(1,'SB',100);add(2,'BB',200)
    if river:
        add(3,'Fold');add(4,'C',200);add(5,'Fold');add(0,'Fold');add(1,'C',100);add(2,'Check')
        for street in (Street.FLOP,Street.TURN):
            for player in (1,2,4):add(player,'Check',street=street)
        add(1,'Check',street=Street.RIVER);add(2,'B',300,Street.RIVER);add(4,'Fold',street=Street.RIVER);add(1,'C',300,Street.RIVER)
        source.board={Street.FLOP:['2c','3d','7s'],Street.TURN:['8c'],Street.RIVER:['9d']}
        source.pot_after_street={Street.PREFLOP:720,Street.FLOP:720,Street.TURN:720,Street.RIVER:1320}
    else:
        # Hero UTG in this independent toy hand (row3); BB continues, others fold.
        source.players[2].cards=[];source.players[3].cards=['Ac','Qc']
        source.players[2].nickname='synthetic-blind';source.players[3].nickname='synthetic-alice'
        add(3,'R',440);add(4,'Fold');add(5,'Fold');add(0,'Fold');add(1,'Fold');add(2,'C',240)
        add(2,'Check',street=Street.FLOP);add(3,'B',300,Street.FLOP);add(2,'C',300,Street.FLOP)
        add(2,'Check',street=Street.TURN);add(3,'B',600,Street.TURN);add(2,'Fold',street=Street.TURN);add(3,'UB',600,Street.TURN)
        source.board={Street.FLOP:['4h','6s','Td'],Street.TURN:['Qh']}
        source.pot_after_street={Street.PREFLOP:1100,Street.FLOP:1700,Street.TURN:1700}
    source.events=events
    return v2(source)


def render(reading):
    result=process_pokerdom(reading,nickname='synthetic-alice')
    return result,''.join(m.text for m in pokerdom_messages(result))


def test_preflop_foldout_uses_raise_total_refund_once_and_hero_only_end_stack():
    result,text=render(v2(table()))
    assert result.replay_status=='passed'
    assert text.startswith('Рука с A♥K♦.\nТы на SB с A♥K♦ против BB с неизвестными картами.')
    assert 'у тебя 5\u2009000 фишек (25,00 ББ), у BB 5\u2009000 фишек (25,00 ББ)' in text
    assert 'Префлоп: ты повышаешь до 600 фишек (3,00 ББ), BB пасует, тебе возвращается 400 фишек (2,00 ББ).\nБанк 420 фишек (2,10 ББ).' in text
    assert 'твой чистый результат +210 фишек (+1,05 ББ), конечный стек 5\u2009210 фишек (26,05 ББ)' in text
    assert text.count('возвращается')==1 and '4\u2009790' not in text
    # Full diagnostic transcript remains unchanged and independently available.
    assert '#7' in pokerdom_report(result) and 'room_payout' in pokerdom_report(result)


def test_six_player_bet_fold_story_has_only_continuing_opponent_and_no_technical_clutter():
    result,text=render(six_players())
    assert result.replay_status=='passed'
    header=text.split('\n\n')[0]
    assert 'Ты на UTG с A♣Q♣ против BB с неизвестными картами' in header
    assert 'у BB 7\u2009000 фишек (35,00 ББ)' in header and 'у SB' not in header and 'у BTN' not in header
    assert '\n\nФлоп 4♥6♠T♦: BB чек, ты ставишь 300 фишек (1,50 ББ), BB доплачивает 300 фишек (1,50 ББ).\nБанк 1\u2009700 фишек (8,50 ББ).' in text
    assert 'Тёрн Q♥: BB чек, ты ставишь 600 фишек (3,00 ББ), BB пасует, тебе возвращается 600 фишек (3,00 ББ).' in text
    assert '\n\nРивер: не сыгран.' in text
    assert 'Результат по расчёту: банк 1\u2009700 фишек (8,50 ББ), твой чистый результат +940 фишек (+4,70 ББ), конечный стек 7\u2009940 фишек (39,70 ББ).' in text
    for unwanted in ('|','Чтение:','проверка чтения','Проверка пройдена','not_checked','room_','#1','Примечание','Dmarker','не записана','стратег','P1','SB 6'):
        assert unwanted not in text


def test_multiway_header_keeps_hj_that_folds_on_river_and_conditional_outcome():
    result,text=render(six_players(river=True))
    assert result.replay_status=='passed' and not result.outcome_confirmed
    assert 'против SB с Q♥J♦ и HJ с неизвестными картами' in text
    assert 'у HJ 7\u2009000 фишек (35,00 ББ)' in text
    assert 'Ривер 9♦: SB чек, ты ставишь 300 фишек (1,50 ББ), HJ пасует, SB доплачивает 300 фишек (1,50 ББ).' in text
    assert 'По видимым картам, условный результат' in text and 'Явное вскрытие не подтверждено.' in text
    assert 'Результат по расчёту:' not in text


@pytest.mark.parametrize('mode',['unknown','explicit','river-fold'])
def test_river_outcome_never_promotes_unknown_or_unobserved_cards(mode):
    reading=six_players(river=True)
    if mode=='unknown':reading.players[1].cards=[]
    if mode=='explicit':
        reading.players[1].showdown_shown=True;reading.players[2].showdown_shown=True
    if mode=='river-fold':
        last=reading.events[-1];last.token='Fold';last.amount=None;last.amount_text=None
        reading.events.append(last.model_copy(update={'number':last.number+1,'player_nickname':'synthetic-alice','token':'UB','amount':300,'amount_text':'300'}))
        reading.pot_after_street[Street.RIVER]=720
    result,text=render(reading)
    assert result.replay_status=='passed'
    if mode=='unknown':
        assert result.outcome_basis=='unknown'
        assert 'Победитель и конечный стек не определены' in text and 'твой чистый результат' not in text
    else:
        assert result.outcome_confirmed and 'Результат по расчёту:' in text
        assert 'По видимым картам' not in text


@pytest.mark.parametrize('mode',['bank','cards','nickname','incomplete','not-hand'])
def test_bad_read_is_short_concrete_diagnostic_and_next_step_not_success(mode):
    reading=six_players()
    if mode=='bank':
        value=reading.pot_after_street[Street.FLOP];assert value is not None
        reading.pot_after_street[Street.FLOP]=value+1
    if mode=='cards':reading.board[Street.FLOP]=[]
    if mode=='nickname':reading.events[8].player_nickname='not-a-player'
    if mode=='incomplete':reading.action_log_complete=False
    if mode=='not-hand':reading.is_hand=False
    result,text=render(reading)
    assert result.replay_status!='passed'
    assert len(text)<350 and 'Рука с' not in text and 'результат +' not in text
    assert 'Пришли' in text or 'пришли' in text
    if mode=='bank':assert 'Флоп: банк' in text and 'не сошёлся' in text
    if mode=='nickname':assert '/pdnick' in text
    assert 'room_' not in text and 'not_checked' not in text and '#1' not in text


def test_notes_and_markup_nicknames_do_not_leak_into_story_and_long_actions_are_not_lost():
    reading=table()
    reading.players[0].nickname='<b>synthetic</b>😀'
    reading.players[0].stack_shown=reading.players[1].stack_shown=1000000
    events=reading.events[:4]
    commits={'A':100,'B':200};target=200
    for i in range(201):
        player='A' if i%2==0 else 'B';target+=400
        amount=target-commits[player];commits[player]=target
        events.append(PokerdomEvent(number=len(events)+1,player=player,street=Street.PREFLOP,token='R',amount=amount,amount_text=str(amount)))
    events.append(PokerdomEvent(number=len(events)+1,player='B',street=Street.PREFLOP,token='Fold',amount=None,amount_text=None))
    events.append(PokerdomEvent(number=len(events)+1,player='A',street=Street.PREFLOP,token='UB',amount=400,amount_text='400'))
    reading.events=events;reading.pot_after_street[Street.PREFLOP]=20+commits['B']*2
    result=process_pokerdom(v2(reading),nickname='<b>synthetic</b>😀')
    assert result.replay_status=='passed'
    messages=pokerdom_messages(result);text=''.join(m.text for m in messages)
    assert len(messages)>1 and all(m.parse_mode is None and len(m.text.encode('utf-16-le'))//2<=3500 for m in messages)
    assert text.count('повышаешь до')==101 and text.count('повышает до')==100
    import re
    money_pattern = r"[+−]?[0-9][0-9\u2009]* фиш(?:ка|ки|ек) \([^)]*ББ\)"
    assert re.findall(money_pattern,text)==[pair for message in messages for pair in re.findall(money_pattern,message.text)]
    assert '<b>' not in text and text.endswith('конечный стек 1\u2009080\u2009210 фишек (5401,05 ББ).')


@pytest.mark.parametrize('amount,bb,expected',[
    (1,8,'1 фишка (0,13 ББ)'),(2,8,'2 фишки (0,25 ББ)'),
    (11,8,'11 фишек (1,38 ББ)'),(21,8,'21 фишка (2,63 ББ)'),
    (22,8,'22 фишки (2,75 ББ)'),(0,8,'0 фишек (0,00 ББ)'),
    (0,100,'0 фишек (0,00 ББ)'),(105,100,'+105 фишек (+1,05 ББ)'),(-105,100,'−105 фишек (−1,05 ББ)'),
    (1,1000,'+1 фишка (+0,00 ББ)'),(-1,1000,'−1 фишка (−0,00 ББ)'),
])
def test_money_display_exact_chips_plural_signs_and_half_up(amount,bb,expected):
    from harness.presentation.pokerdom_vision import _amount
    assert _amount(amount,bb,signed=bb>=100)==expected


@pytest.mark.parametrize('bb',[0,-1])
def test_unknown_or_invalid_blind_is_not_replaced_by_one(bb):
    from harness.presentation.pokerdom_vision import _amount
    with pytest.raises(ValueError):_amount(100,bb)
    result=process_pokerdom(v2(table()),nickname='synthetic-alice')
    assert result.raw is not None
    result.raw.bb=bb
    text=''.join(m.text for m in pokerdom_messages(result))
    assert 'Не удалось определить большой блайнд' in text and 'Рука с' not in text


def test_losing_hero_uses_same_negative_sign_for_chips_and_bb():
    result=process_pokerdom(v2(table()),nickname='synthetic-bob')
    assert result.replay_status=='passed' and result.outcome_confirmed
    text=''.join(m.text for m in pokerdom_messages(result))
    assert 'чистый результат −210 фишек (−1,05 ББ)' in text
    assert 'конечный стек 4\u2009790 фишек (23,95 ББ)' in text
