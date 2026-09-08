from math import comb

import eval7
import pytest

from harness.analysis.tools.pot_odds import required_equity
from harness.analysis.tools.river_call import (
    _HAND_TYPES,
    river_call_requirement,
)

# Рука с ривера: борд спарен тузами, у героя вторая пара с кикером.
_PAIRED_ACE_BOARD = ["Jc", "6d", "As", "2h", "Ac"]
_SECOND_PAIR = ("Jh", "Ts")
_POT_BEFORE = 398_000
_TO_CALL = 169_000


def test_required_equity_comes_from_the_pot_odds_tool():
    result = river_call_requirement(_SECOND_PAIR, _PAIRED_ACE_BOARD, _POT_BEFORE, _TO_CALL)
    assert result.required_equity == required_equity(_TO_CALL, _POT_BEFORE)
    assert abs(result.required_equity - 0.298) < 5e-4


def test_dead_cards_are_excluded_from_the_enumeration():
    result = river_call_requirement(_SECOND_PAIR, _PAIRED_ACE_BOARD, _POT_BEFORE, _TO_CALL)
    # 47 карт без борда, минус комбо, задевающие две карты героя (2*46 - 1 = 91).
    assert result.combos_total == comb(47, 2) - 91 == comb(45, 2) == 990
    assert result.combos_ahead + result.combos_behind + result.combos_tied == 990

    # Тот же борд, но у героя оба оставшихся туза: если бы карты героя не были
    # мёртвыми, комбо AhAd делило бы с ним каре и попало в ничьи.
    quads = river_call_requirement(("Ah", "Ad"), _PAIRED_ACE_BOARD, _POT_BEFORE, _TO_CALL)
    assert quads.combos_total == 990
    assert quads.combos_ahead == 0
    assert quads.combos_tied == 0


def test_live_hand_second_pair_on_paired_ace_board():
    result = river_call_requirement(_SECOND_PAIR, _PAIRED_ACE_BOARD, _POT_BEFORE, _TO_CALL)
    assert (result.combos_ahead, result.combos_behind, result.combos_tied) == (122, 862, 6)
    # Ничьи — это ровно другие JT: два живых валета на четыре живые десятки.
    assert result.combos_tied == 2 * 3
    assert abs(result.bluffs_needed - 50.08) < 5e-3
    assert abs(result.bluffs_needed_min_value - 38.19) < 5e-3
    # 38 блефов при 862 проигрывающих комбо на борде — фолд ничем не доказан.
    assert result.fold_proven is False


def test_min_value_excludes_better_hands_of_the_same_class():
    result = river_call_requirement(_SECOND_PAIR, _PAIRED_ACE_BOARD, _POT_BEFORE, _TO_CALL)
    # Минимальное вэлью — старший класс комбинации: 86 комбо с одним тузом
    # (трипс), AhAd (каре) и 7 карманных пар к картам борда (фулл-хаус).
    assert result.min_value_combos == 86 + 1 + 7 == 94
    # Оставшиеся 28 бьют героя ВНУТРИ двух пар: KK и QQ (по 6 комбо) плюс
    # JQ и JK (по 8) — их соперник может и не поставить, в минимум они не входят.
    assert result.combos_ahead - result.min_value_combos == 6 + 6 + 8 + 8 == 28


def test_min_value_is_a_subset_of_combos_ahead():
    for hero, board in (
        (_SECOND_PAIR, _PAIRED_ACE_BOARD),
        (("3c", "2d"), ["Ac", "Kd", "Qh", "Js", "9s"]),
        (("Ah", "Ad"), _PAIRED_ACE_BOARD),
        (("Kc", "Qd"), ["7s", "7h", "7d", "2c", "2s"]),
    ):
        result = river_call_requirement(hero, board, _POT_BEFORE, _TO_CALL)
        assert result.min_value_combos <= result.combos_ahead


def test_nuts_require_no_bluffs_at_all():
    # Роял-флеш: комбо, которые бьют героя или делят с ним банк, не существует.
    result = river_call_requirement(("Ts", "9s"), ["As", "Ks", "Qs", "Js", "2h"], 398_000, 169_000)
    assert (result.combos_ahead, result.combos_tied, result.combos_behind) == (0, 0, 990)
    assert result.bluffs_needed == 0.0
    assert result.bluffs_needed_min_value == 0.0
    assert result.fold_proven is False


def test_worst_possible_hand_proves_the_fold():
    # Герой разыгрывает борд с худшими картами: проиграть ему нечем, любая
    # улучшившаяся рука бьёт его старшим классом комбинации.
    result = river_call_requirement(("3c", "2d"), ["Ac", "Kd", "Qh", "Js", "9s"], 398_000, 169_000)
    assert result.combos_behind == 0
    assert result.min_value_combos == result.combos_ahead == 701
    assert result.bluffs_needed_min_value > result.combos_behind
    assert result.fold_proven is True


def test_fold_proof_stands_on_the_minimal_value_only():
    # Спаренный королями борд, герой разыгрывает борд с десяткой: проигрывающих
    # комбо мало (220), и полного вэлью хватило бы, чтобы объявить фолд
    # доказанным (286 > 220). Минимальное вэлью требует 186 блефов — они на
    # борде есть, и доказательства нет.
    result = river_call_requirement(("Td", "5d"), ["Ks", "Kc", "6s", "8d", "Qd"], 398_000, 169_000)
    assert (result.combos_ahead, result.combos_behind, result.combos_tied) == (713, 220, 57)
    assert result.min_value_combos == 477
    assert abs(result.bluffs_needed - 286.36) < 5e-3
    assert abs(result.bluffs_needed_min_value - 186.15) < 5e-3
    assert result.bluffs_needed > result.combos_behind >= result.bluffs_needed_min_value
    assert result.fold_proven is False


def test_board_that_plays_itself_leaves_only_ties():
    # Роял-флеш на борде: все 990 комбо делят банк, эквити героя ровно 1/2.
    result = river_call_requirement(("2c", "3d"), ["As", "Ks", "Qs", "Js", "Ts"], 398_000, 169_000)
    assert (result.combos_ahead, result.combos_behind, result.combos_tied) == (0, 0, 990)
    assert result.bluffs_needed == 0.0
    assert result.fold_proven is False


def test_ties_are_counted_the_way_that_favours_calling():
    cheap = river_call_requirement(_SECOND_PAIR, _PAIRED_ACE_BOARD, 398_000, 169_000)
    # req < 1/2: ничья лучше требуемой доли, поэтому все 6 ничьих идут в
    # ставящий диапазон и снижают требование.
    req = cheap.required_equity
    assert abs(cheap.bluffs_needed - (req * 122 + (req - 0.5) * 6) / (1 - req)) < 1e-9

    expensive = river_call_requirement(_SECOND_PAIR, _PAIRED_ACE_BOARD, 100_000, 200_000)
    # req = 2/3 > 1/2: ничья хуже требуемой доли, минимум требования достигается
    # на пустом множестве ничьих — 244, а не 246.
    assert expensive.required_equity == 2 / 3
    assert abs(expensive.bluffs_needed - 2 * 122) < 1e-9
    assert abs(expensive.bluffs_needed_min_value - 2 * 94) < 1e-9


def test_hand_type_names_match_eval7():
    examples = [
        ["2c", "7d", "9s", "Jh", "4c", "Qd", "Kc"],
        ["2c", "2d", "9s", "Jh", "4c", "Qd", "Kc"],
        ["2c", "2d", "9s", "9h", "4c", "Qd", "Kc"],
        ["2c", "2d", "2s", "9h", "4c", "Qd", "Kc"],
        ["2c", "3d", "4s", "5h", "6c", "Qd", "Kc"],
        ["2c", "7c", "9c", "Jc", "4c", "Qd", "Kh"],
        ["2c", "2d", "2s", "9h", "9c", "Qd", "Kc"],
        ["2c", "2d", "2s", "2h", "9c", "Qd", "Kc"],
        ["2c", "3c", "4c", "5c", "6c", "Qd", "Kh"],
    ]
    scores = [eval7.evaluate([eval7.Card(card) for card in hand]) for hand in examples]
    assert [eval7.handtype(score) for score in scores] == list(_HAND_TYPES)
    # Порядок в `_HAND_TYPES` — порядок по силе: на нём стоит правило минимального вэлью.
    assert scores == sorted(scores)


def test_rejects_impossible_input():
    with pytest.raises(ValueError, match="борд"):
        river_call_requirement(_SECOND_PAIR, _PAIRED_ACE_BOARD[:4], _POT_BEFORE, _TO_CALL)
    with pytest.raises(ValueError, match="дважды"):
        river_call_requirement(("Jc", "Ts"), _PAIRED_ACE_BOARD, _POT_BEFORE, _TO_CALL)
    with pytest.raises(ValueError, match="не карты колоды"):
        river_call_requirement(("Jh", "T"), _PAIRED_ACE_BOARD, _POT_BEFORE, _TO_CALL)
    with pytest.raises(ValueError, match="банк"):
        river_call_requirement(_SECOND_PAIR, _PAIRED_ACE_BOARD, 0, _TO_CALL)
    with pytest.raises(ValueError, match="колла"):
        river_call_requirement(_SECOND_PAIR, _PAIRED_ACE_BOARD, _POT_BEFORE, 0)
