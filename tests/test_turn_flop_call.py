"""Требование к ставящему диапазону на незавершённом борде: перебор и его границы.

Числа живой руки владельца — борд `Jc 6d As 2h`, у героя `Jh Ts`, банк 160 000,
доплата 69 000 — проверяются здесь против ИНСТРУМЕНТА; что они доезжают до
вердикта и до игрока неискажёнными, проверяет `test_turn_flop_analysis`.

Тяжёлый перебор проверяется вторым, независимо написанным счётом (`_naive_equity`):
он идёт по комбо снаружи и по доборам внутри — в порядке, обратном рабочему, — и
знаменатель берёт длиной собственного списка доборов, а не формулой. Совпадение
двух счётов до последнего знака и есть гейт на полноту перебора.
"""

from itertools import combinations
from math import comb

import eval7
import pytest

from harness.analysis.tools.equity import _FULL_DECK
from harness.analysis.tools.pot_odds import required_equity
from harness.analysis.tools.river_call import _HAND_TYPES
from harness.analysis.tools.turn_flop_call import (
    _equity_by_combo,
    turn_flop_call_requirement,
)

# Живая рука владельца на тёрне: у героя вторая пара, борд тузовый и несдвоенный.
_LIVE_TURN = ["Jc", "6d", "As", "2h"]
_LIVE_FLOP = ["Jc", "6d", "As"]
_SECOND_PAIR = ("Jh", "Ts")
_POT_BEFORE = 160_000
_TO_CALL = 69_000

_TURN_COMBOS = comb(46, 2)  # 52 карты минус две героя и четыре борда
_FLOP_COMBOS = comb(47, 2)


def _cards(names):
    return [eval7.Card(name) for name in names]


def _naive_equity(hero, board):
    """Эквити героя против каждого комбо — своим перебором на каждое комбо.

    Второй счёт того же числа: внешний цикл по комбо соперника, внутренний по
    доборам, знаменатель — длина списка доборов. Рабочий перебор устроен
    наоборот, и совпадение двух ответов не может быть совпадением одной ошибки.
    """
    dead = set(hero) | set(board)
    rest = [name for name in _FULL_DECK if name not in dead]
    card = {name: eval7.Card(name) for name in _FULL_DECK}
    hero_cards = [card[name] for name in hero]
    board_cards = [card[name] for name in board]
    to_come = 5 - len(board)
    out = {}
    for i, j in combinations(range(len(rest)), 2):
        opponent = [card[rest[i]], card[rest[j]]]
        run_outs = [card[name] for name in rest if name not in (rest[i], rest[j])]
        won = 0.0
        deals = 0
        for run_out in combinations(run_outs, to_come):
            full_board = board_cards + list(run_out)
            hero_score = eval7.evaluate(hero_cards + full_board)
            opponent_score = eval7.evaluate(opponent + full_board)
            deals += 1
            if hero_score > opponent_score:
                won += 1.0
            elif hero_score == opponent_score:
                won += 0.5
        out[(i, j)] = won / deals
    return out


def _split_by_hand_class(hero, board, equity):
    """Эквити героя против минимального вэлью и против всего остального.

    Правило минимального вэлью написано здесь заново — бьёт героя И старшим
    классом, — чтобы проверять расчёт, а не пересказывать его.
    """
    dead = set(hero) | set(board)
    rest = [name for name in _FULL_DECK if name not in dead]
    board_cards = _cards(board)
    hero_score = eval7.evaluate(_cards(hero) + board_cards)
    hero_type = _HAND_TYPES.index(eval7.handtype(hero_score))
    value, pool = [], []
    for i, j in combinations(range(len(rest)), 2):
        score = eval7.evaluate(_cards([rest[i], rest[j]]) + board_cards)
        beats = score > hero_score
        stronger_class = _HAND_TYPES.index(eval7.handtype(score)) > hero_type
        (value if beats and stronger_class else pool).append(equity[(i, j)])
    return value, pool


def test_required_equity_comes_from_the_pot_odds_tool():
    result = turn_flop_call_requirement(_SECOND_PAIR, _LIVE_TURN, _POT_BEFORE, _TO_CALL)
    assert result.required_equity == required_equity(_TO_CALL, _POT_BEFORE)
    assert abs(result.required_equity - 0.3013) < 5e-4


def test_dead_cards_are_excluded_from_the_enumeration():
    turn = turn_flop_call_requirement(_SECOND_PAIR, _LIVE_TURN, _POT_BEFORE, _TO_CALL)
    flop = turn_flop_call_requirement(_SECOND_PAIR, _LIVE_FLOP, _POT_BEFORE, _TO_CALL)
    assert turn.combos_total == _TURN_COMBOS == 1035
    assert flop.combos_total == _FLOP_COMBOS == 1081


def test_the_live_turn_of_the_owners_hand():
    """Числа живой руки целиком — расхождение в любом из них станет красным."""
    result = turn_flop_call_requirement(_SECOND_PAIR, _LIVE_TURN, _POT_BEFORE, _TO_CALL)
    assert result.combos_ahead == 188
    assert result.min_value_combos == 55
    assert abs(result.min_value_equity - 0.0781) < 5e-4
    assert result.bluffs_needed_min_value == 18


def test_min_value_on_the_turn_is_two_pair_and_better():
    """Минимальное вэлью пересчитано по картам борда, а не взято у инструмента.

    Борд `Jc 6d As 2h`: старше пары валетов класс начинается с двух пар. Живых
    тузов 3, валетов 2 (`Jc` на борде, `Jh` у героя), шестёрок 3, двоек 3.
    """
    two_pair = 3 * 2 + 3 * 3 + 3 * 3 + 2 * 3 + 2 * 3 + 3 * 3  # AJ A6 A2 J6 J2 62
    trips = comb(3, 2) + comb(2, 2) + comb(3, 2) + comb(3, 2)  # AA JJ 66 22
    result = turn_flop_call_requirement(_SECOND_PAIR, _LIVE_TURN, _POT_BEFORE, _TO_CALL)
    assert result.min_value_combos == two_pair + trips == 55
    # Пара тузов бьёт героя тем же классом и в минимум не входит — как на ривере.
    assert result.combos_ahead - result.min_value_combos == 133


def test_min_value_never_exceeds_the_combos_ahead():
    for hero, board in (
        (_SECOND_PAIR, _LIVE_TURN),
        (_SECOND_PAIR, _LIVE_FLOP),
        (("3c", "2d"), ["Ac", "Kd", "Qh", "9s"]),
        (("Ah", "Ad"), ["Ac", "Kd", "Qh"]),
    ):
        result = turn_flop_call_requirement(hero, board, _POT_BEFORE, _TO_CALL)
        assert result.min_value_combos <= result.combos_ahead


def test_the_turn_enumerates_every_river_card():
    """Эквити каждого комбо совпадает со вторым, независимым счётом."""
    naive = _naive_equity(_SECOND_PAIR, _LIVE_TURN)
    dead = set(_SECOND_PAIR) | set(_LIVE_TURN)
    rest = [name for name in _FULL_DECK if name not in dead]
    working = _equity_by_combo(_cards(_SECOND_PAIR), _cards(_LIVE_TURN), _cards(rest), 1)
    assert len(working) == _TURN_COMBOS
    assert working == naive


@pytest.mark.slow  # 990 доборов на каждое из 1081 комбо, и всё это дважды
def test_the_flop_enumerates_every_pair_of_run_outs():
    """То же на флопе: перебираются пары доборов, а не выборка из них."""
    naive = _naive_equity(_SECOND_PAIR, _LIVE_FLOP)
    dead = set(_SECOND_PAIR) | set(_LIVE_FLOP)
    rest = [name for name in _FULL_DECK if name not in dead]
    working = _equity_by_combo(_cards(_SECOND_PAIR), _cards(_LIVE_FLOP), _cards(rest), 2)
    assert len(working) == _FLOP_COMBOS
    assert working == naive


def test_every_combo_is_dealt_the_same_number_of_run_outs():
    """Знаменатель у всех комбо один — на этом стоит формула вместо счётчика."""
    for board, to_come, expected in ((_LIVE_TURN, 1, 44), (_LIVE_FLOP, 2, 990)):
        dead = set(_SECOND_PAIR) | set(board)
        rest = [name for name in _FULL_DECK if name not in dead]
        seen = {
            len([run_out for run_out in combinations(rest, to_come) if not set(run_out) & set(pair)])
            for pair in combinations(rest, 2)
        }
        assert seen == {expected} == {comb(len(rest) - 2, to_come)}


def test_the_bluff_count_cannot_be_lowered_by_another_choice():
    """Найденное число блефов достигает требования, а на единицу меньшее — нет.

    Обе средние считаются в тесте по своему разбиению и своим эквити, поэтому
    занижение или завышение счёта в инструменте разойдётся с ними.
    """
    result = turn_flop_call_requirement(_SECOND_PAIR, _LIVE_TURN, _POT_BEFORE, _TO_CALL)
    value, pool = _split_by_hand_class(_SECOND_PAIR, _LIVE_TURN, _naive_equity(_SECOND_PAIR, _LIVE_TURN))
    best = sorted(pool, reverse=True)
    needed = result.bluffs_needed_min_value
    assert needed is not None and needed > 0

    def mean(count):
        return (sum(value) + sum(best[:count])) / (len(value) + count)

    assert mean(needed) >= result.required_equity
    assert mean(needed - 1) < result.required_equity
    # Тот же размер, но другой выбор комбо: требование не достигается — значит
    # число отвечает именно на «сколько минимум», а не на «сколько-нибудь».
    worst = sorted(pool)[:needed]
    assert (sum(value) + sum(worst)) / (len(value) + needed) < result.required_equity


def test_a_hopeless_hand_has_no_number_of_bluffs_that_helps():
    """Требуемой эквити не даёт ни один диапазон — числа блефов не существует."""
    result = turn_flop_call_requirement(("7c", "2d"), ["Ac", "Ah", "As", "Kd"], 200_000, 800_000)
    assert result.required_equity == 0.8
    assert result.min_value_combos > 0
    assert result.bluffs_needed_min_value is None


def test_a_hand_ahead_of_the_value_needs_no_bluffs():
    """Эквити против самого вэлью уже покрывает цену колла — блефов нужно ноль."""
    result = turn_flop_call_requirement(_SECOND_PAIR, _LIVE_TURN, 1_000_000, 1)
    assert result.min_value_combos == 55
    assert result.min_value_equity > result.required_equity
    assert result.bluffs_needed_min_value == 0


def test_no_value_on_the_board_means_no_requirement():
    """Комбо старшего класса на борде нет вовсе — требования к диапазону тоже."""
    result = turn_flop_call_requirement(("As", "Ks"), ["Qs", "Js", "Ts", "2h"], 100_000, 30_000)
    assert (result.combos_ahead, result.min_value_combos) == (0, 0)
    assert result.min_value_equity == 0.0
    assert result.bluffs_needed_min_value == 0


@pytest.mark.parametrize(
    ("hero", "board", "pot_before", "to_call", "message"),
    [
        (_SECOND_PAIR, ["Jc", "6d", "As", "2h", "Ac"], 160_000, 69_000, "3 или 4"),
        (_SECOND_PAIR, ["Jc", "6d"], 160_000, 69_000, "3 или 4"),
        (_SECOND_PAIR, _LIVE_TURN, 0, 69_000, "банк"),
        (_SECOND_PAIR, _LIVE_TURN, 160_000, 0, "цена колла"),
        (("Jh", "Zz"), _LIVE_TURN, 160_000, 69_000, "не карты колоды"),
        (("Jc", "Ts"), _LIVE_TURN, 160_000, 69_000, "дважды"),
    ],
)
def test_the_shape_of_the_call_is_checked_before_the_enumeration(
    hero, board, pot_before, to_call, message
):
    """Каждая граница формы снимает расчёт с названной причиной, а не считает как-нибудь."""
    with pytest.raises(ValueError, match=message):
        turn_flop_call_requirement(hero, board, pot_before, to_call)
