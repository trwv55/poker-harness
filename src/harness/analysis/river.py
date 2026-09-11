"""Разбор риверной точки: цена решения и требование к ставящему диапазону.

**Что здесь считается.** Улица последняя, борд полный, поэтому эквити ни против
чего угадывать не надо: `analysis.tools.river_call` перебирает ВСЕ комбо,
оставшиеся в колоде, и говорит, сколько блефов должно быть в ставящем диапазоне
соперника, чтобы колл вышел в ноль. Модели диапазона здесь нет — есть перебор
борда, — и потому зона у точки `strict`, а не `assuming`: допущение о том, с
какими руками соперник ставит, в расчёт не входит.

**Вердикт есть не всегда, и это не отказ.** Требование к диапазону посчитано на
каждой точке нашей формы, но лучшее действие названо только тогда, когда борд
исчерпан: `fold_proven` означает, что даже минимальное вэлью требует блефов
больше, чем проигрывающих героя комбо на борде вообще существует
(правило — докстринг `analysis.tools.river_call`). Тогда точка несёт
`best_action = "fold"` при единственном допущении «сильнейшие руки он ставит»;
его называет словами `presentation`, потому что диапазоном (`Assumption`) оно
не является — диапазон мы не берём ниоткуда.

Во всех остальных случаях `best_action` остаётся пустым: расчёт не знает, что
лучше, и сказать «колл верен» ему нечем. Числа при этом остаются в
`detail[RIVER_CALL_DETAIL]` и показываются игроку — точка без вердикта
перестаёт быть пустой, но ценой не обзаводится
(`test_a_river_point_without_the_proof_still_carries_its_numbers`).

**Цены у риверной точки нет ни в одном из двух случаев.** `SpotKind.POSTFLOP`
не входит в `JUDGED_SPOTS`, поэтому такая точка не попадает ни в ранжирование,
ни в сумму потерь руки, ни в колонку `decision_points.judged`
(`test_a_proven_river_fold_is_not_priced_and_never_enters_the_sum`). Требование
к диапазону — не EV решения: сколько стоил колл, известно только вместе с
диапазоном, а его мы не угадываем.

**Границы формы.** Инструмент отвечает на вопрос «коллировать ли», поэтому
точка разбирается, только когда перед героем ставка (`to_call > 0`), живых в
руке ровно двое, карты героя известны и борд из пяти карт. Каждая граница
снимает разбор целиком с названной причиной — поправлять в переборе нечего.
"""

from __future__ import annotations

from harness.analysis.classifier import action_name, unjudged_point
from harness.analysis.tools.river_call import RiverCallRequirement, river_call_requirement
from harness.contracts import (
    RIVER_CALL_DETAIL,
    DecisionPoint,
    EnrichedHand,
    PointVerdict,
    RiverCallDetail,
    SpotKind,
    Street,
    Zone,
)

__all__ = ["river_verdict"]

_BOARD_SIZE = 5
_HERO_CARDS = 2
_HEADS_UP = 2

# Форма борда в порядке сдачи: перебор комбо ждёт пять карт одним списком, а
# каноническая рука хранит их по улицам.
_BOARD_STREETS = (Street.FLOP, Street.TURN, Street.RIVER)

# Причина, по которой у посчитанной точки нет лучшего действия. Строку видит
# только разбор: игроку показываются числа, а не то, чего доказать не удалось
# (SESSIONS_UX — не рассказывать о том, чего не умеем).
_NOT_PROVEN = "фолд не доказан: требование к ставящему диапазону посчитано, лучшего действия нет"

_TOOL = "river_call"


def river_verdict(dp: DecisionPoint, en: EnrichedHand) -> PointVerdict | None:
    """Вердикт по риверной точке — или `None`, если точка не риверная."""
    if dp.street is not Street.RIVER:
        return None

    hero = en.hand.dealt.get(en.hand.hero_label, [])
    board = [card for street in _BOARD_STREETS for card in en.hand.boards.get(street, [])]
    if dp.to_call <= 0:
        return unjudged_point(dp, SpotKind.POSTFLOP, "перед героем нет ставки: колл не оценивается")
    if dp.live_total != _HEADS_UP:
        return unjudged_point(
            dp,
            SpotKind.POSTFLOP,
            f"живых в руке {dp.live_total}: перебор считает требование к одному диапазону",
        )
    if len(hero) != _HERO_CARDS:
        return unjudged_point(dp, SpotKind.POSTFLOP, "карты героя неизвестны")
    if len(board) != _BOARD_SIZE:
        return unjudged_point(
            dp, SpotKind.POSTFLOP, f"борд из {len(board)} карт, а перебор ждёт {_BOARD_SIZE}"
        )

    try:
        requirement = river_call_requirement((hero[0], hero[1]), board, dp.pot_before, dp.to_call)
    except ValueError as failure:
        # Карты, которых нет в колоде, или разъехавшийся банк: считать по ним
        # нельзя, а подправить их значило бы соврать про деньги (CLAUDE.md).
        return unjudged_point(dp, SpotKind.POSTFLOP, str(failure))

    detail = {RIVER_CALL_DETAIL: _detail(requirement, dp).model_dump(mode="json")}
    if not requirement.fold_proven:
        return unjudged_point(dp, SpotKind.POSTFLOP, _NOT_PROVEN, detail, tools=[_TOOL])
    return PointVerdict(
        dp_index=dp.index,
        street=dp.street,
        spot=SpotKind.POSTFLOP,
        zone=Zone.STRICT,
        action_taken=action_name(dp),
        best_action="fold",
        ev_diff_bb=0.0,
        assumption=None,
        tools=[_TOOL],
        detail=detail,
    )


def _detail(requirement: RiverCallRequirement, dp: DecisionPoint) -> RiverCallDetail:
    """Что из перебора уезжает в вердикт: цена решения и требование к диапазону.

    Разложение борда по исходам (сколько комбо бьёт героя, сколько проигрывает,
    сколько делит банк) сюда не едет: игроку показывается ровно то, что лежит
    здесь, а перебор возможного — не диапазон соперника.
    """
    value = requirement.min_value_combos
    bluffs = requirement.bluffs_needed_min_value
    return RiverCallDetail(
        pot_before=dp.pot_before,
        to_call=dp.to_call,
        required_equity=requirement.required_equity,
        min_value_combos=value,
        bluffs_needed_min_value=bluffs,
        # Ноль при нулевом знаменателе — вэлью в минимуме нет вовсе, требования
        # к диапазону тоже, и доли в нём не существует. Строку про блефы
        # изложение на такой точке не печатает.
        bluff_share=bluffs / (value + bluffs) if value + bluffs > 0.0 else 0.0,
        fold_proven=requirement.fold_proven,
    )
