"""Требование к ставящему диапазону соперника на ривере: сколько нужно блефов.

**Перебор, а не оценка.** Борд подаётся полным (ровно пять карт), поэтому каждое
комбо соперника сравнивается с рукой героя одним вызовом `eval7.evaluate` на семи
картах и попадает ровно в один из трёх исходов. Инструмент проходит ВСЕ комбо,
оставшиеся в колоде после мёртвых карт, и предъявляет требование к ставящему
диапазону; модели диапазона здесь нет, эквити против угаданного не объявляется.

**Что считается.** Пусть ставящий диапазон соперника состоит из `V` комбо,
которые героя бьют, `T` комбо, которые делят банк, и `B` комбо, которые герою
проигрывают. Эквити героя при вскрытии — `(B + T/2) / (V + T + B)`, колл
безубыточен при равенстве этой доли `required_equity` (`pot_odds`), откуда

    B = (req·V + (req − ½)·T) / (1 − req)

**Ничьи.** Сколько именно ничейных комбо лежит в ставящем диапазоне, перебор
знать не может — это уже была бы модель диапазона. Поэтому берётся не
предположение, а минимум по всем возможным подмножествам: `B(T)` линейна по `T`,
значит на отрезке `T ∈ [0, combos_tied]` минимум достигается на конце — при
`req < ½` на `T = combos_tied`, при `req > ½` на `T = 0`. Возвращается этот
минимум, то есть требование, которое нельзя занизить никаким выбором доли ничьих
(`test_ties_are_counted_the_way_that_favours_calling`).

**Правило минимального вэлью.** Комбо входит в минимальное вэлью, если оно бьёт
героя И составляет комбинацию СТАРШЕГО КЛАССА, чем комбинация героя (класс — то,
что `eval7.handtype` называет "Two Pair", "Trips", "Full House" и т. д.). Руки,
которые бьют героя внутри того же класса — старшая вторая пара, тот же класс с
лучшим кикером, — в минимальное вэлью не входят: они и есть та часть вэлью,
которую соперник может не ставить
(`test_min_value_excludes_better_hands_of_the_same_class`). Минимальное вэлью —
подмножество `combos_ahead` (`test_min_value_is_a_subset_of_combos_ahead`).

**Доказательство фолда.** `bluffs_needed` растёт по `V`, поэтому требование,
посчитанное на минимальном вэлью, — самое слабое из всех: любой ставящий
диапазон, который содержит минимальное вэлью целиком, требует блефов не меньше.
Если и это требование превышает ЧИСЛО ВСЕХ проигрывающих комбо на борде, столько
блефов на этом борде просто не существует — и фолд верен при единственном
допущении «сильнейшие руки соперник ставит» (`test_worst_possible_hand_proves_the_fold`).
Требование от полного вэлью для этого признака не годится и не используется
(`test_fold_proof_stands_on_the_minimal_value_only`); оба числа возвращаются
рядом, чтобы утверждение проверялось глазами.

Инструмент отвечает на один вопрос одной точки решения — «коллировать ли на
ривере». Частоту защиты (MDF) он не считает, поэтому и не назван «защитой».
"""

from __future__ import annotations

import eval7
from pydantic import BaseModel

# Колода и разворачивание класса в комбо берутся у `equity`, а не заводятся
# заново: перебор обязан ходить по тем же строкам карт, что и остальные
# инструменты, иначе мёртвые карты отсекались бы сравнением разных алфавитов.
from harness.analysis.tools.equity import _FULL_DECK, combos_of_class
from harness.analysis.tools.pot_odds import required_equity
from harness.contracts import all_classes

_BOARD_SIZE = 5
_TIE_SHARE = 0.5

# Классы комбинаций в порядке возрастания силы — ровно те строки, которые
# возвращает `eval7.handtype` (`test_hand_type_names_match_eval7`). Сравниваем по
# имени, а не по внутреннему представлению счёта: незнакомая строка обязана
# уронить расчёт, а не тихо съехать в неверный класс.
_HAND_TYPES = (
    "High Card",
    "Pair",
    "Two Pair",
    "Trips",
    "Straight",
    "Flush",
    "Full House",
    "Quads",
    "Straight Flush",
)


class RiverCallRequirement(BaseModel):
    """Разложение борда и требование к ставящему диапазону соперника.

    `combos_ahead` + `combos_behind` + `combos_tied` = `combos_total` — все
    комбо, которые остались в колоде после мёртвых карт (карты героя и борд).

    `bluffs_needed` — требование против ставящего диапазона, в котором вэлью —
    ВСЕ бьющие героя комбо; `bluffs_needed_min_value` — против минимального
    вэлью (правило в докстринге модуля). Оба — минимум по подмножествам ничьих.

    `fold_proven` — `bluffs_needed_min_value` больше, чем `combos_behind`.
    """

    required_equity: float
    combos_total: int
    combos_ahead: int
    combos_behind: int
    combos_tied: int
    min_value_combos: int
    bluffs_needed: float
    bluffs_needed_min_value: float
    fold_proven: bool


def river_call_requirement(
    hero: tuple[str, str],
    board: list[str],
    pot_before: int,
    to_call: int,
) -> RiverCallRequirement:
    """Перебирает весь состав борда и возвращает требование к диапазону соперника.

    `pot_before` — банк на момент решения, уже со ставкой соперника внутри (как
    в `pot_odds.required_equity`); `to_call` — цена колла. Обе величины в фишках
    и обе строго положительны: при пустом банке требуемая эквити равна единице,
    и требование к диапазону не определено.

    Борд — ровно пять карт, карты героя и борда не пересекаются, все карты — из
    колоды в записи `equity` ("Ah", "Td"). Иначе `ValueError`.
    """
    if len(board) != _BOARD_SIZE:
        raise ValueError(f"на ривере борд из {_BOARD_SIZE} карт, получено {len(board)}")
    if pot_before <= 0:
        raise ValueError(f"банк на решении должен быть положительным, получено {pot_before}")
    if to_call <= 0:
        raise ValueError(f"цена колла должна быть положительной, получено {to_call}")

    known = list(hero) + list(board)
    unknown = [card for card in known if card not in _FULL_DECK]
    if unknown:
        raise ValueError(f"не карты колоды: {unknown}")
    dead = set(known)
    if len(dead) != len(known):
        raise ValueError(f"карта встречается дважды среди карт героя и борда: {known}")

    req = required_equity(to_call, pot_before)
    board_cards = [eval7.Card(card) for card in board]
    hero_score = eval7.evaluate([eval7.Card(card) for card in hero] + board_cards)
    hero_type = _type_index(hero_score)

    ahead = behind = tied = min_value = 0
    cards: dict[str, eval7.Card] = {}
    for hand_class in all_classes():
        for card1, card2 in combos_of_class(hand_class):
            if card1 in dead or card2 in dead:
                continue
            for card in (card1, card2):
                if card not in cards:
                    cards[card] = eval7.Card(card)
            score = eval7.evaluate([cards[card1], cards[card2]] + board_cards)
            if score > hero_score:
                ahead += 1
                if _type_index(score) > hero_type:
                    min_value += 1
            elif score < hero_score:
                behind += 1
            else:
                tied += 1

    bluffs_needed = _bluffs_needed(ahead, tied, req)
    bluffs_needed_min_value = _bluffs_needed(min_value, tied, req)
    return RiverCallRequirement(
        required_equity=req,
        combos_total=ahead + behind + tied,
        combos_ahead=ahead,
        combos_behind=behind,
        combos_tied=tied,
        min_value_combos=min_value,
        bluffs_needed=bluffs_needed,
        bluffs_needed_min_value=bluffs_needed_min_value,
        fold_proven=bluffs_needed_min_value > behind,
    )


def _type_index(score: int) -> int:
    """Место класса комбинации в `_HAND_TYPES`; незнакомый класс — `ValueError`."""
    return _HAND_TYPES.index(eval7.handtype(score))


def _bluffs_needed(value_combos: int, tie_combos: int, req: float) -> float:
    """Минимум блефов на `value_combos` вэлью — по всем подмножествам ничьих.

    Ноль, а не `None`, когда требование выродилось (вэлью нет вовсе, или ничьи
    при `req < ½` уже дают героя достаточно эквити): «блефов нужно не меньше
    нуля» — верное утверждение о любом диапазоне, а не отсутствие ответа
    (`test_nuts_require_no_bluffs_at_all`).
    """
    ties_in_range = tie_combos if req < _TIE_SHARE else 0
    needed = (req * value_combos + (req - _TIE_SHARE) * ties_in_range) / (1.0 - req)
    return max(0.0, needed)
