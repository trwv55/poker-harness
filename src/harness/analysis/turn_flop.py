"""Разбор точки тёрна или флопа: цена решения и требование к ставящему диапазону.

**Что здесь считается.** То же, что на ривере (`analysis.river`), и тем же
способом: `analysis.tools.turn_flop_call` перебирает ВСЕ комбо, оставшиеся в
колоде, и говорит, сколько комбо помимо несомненного вэлью обязано быть в
ставящем диапазоне соперника, чтобы эквити героя против такого диапазона дошла
до цены колла. Модели диапазона здесь нет — есть перебор борда и перебор
доборов, — поэтому и на этих улицах зона `strict`.

**Лучшего действия здесь не называется никогда, и это не пробел.** На ривере
`fold_proven` доказывает фолд, потому что колл заканчивает раздачу: после него
идёт только вскрытие, и эквити к вскрытию — это весь остаток руки. На тёрне и
флопе за коллом следуют улицы, на которых в банк пойдут ещё ставки; перебор их
не считает, а значит требование к диапазону — не ответ на вопрос «коллировать
ли». Числа при этом остаются в `detail[TURN_FLOP_CALL_DETAIL]` и показываются
игроку (`test_a_turn_point_carries_its_numbers_without_a_best_action`).

Отсюда же и зона всей руки: точка без названной линии в неё не входит
(`worker.pipeline._hand_zone`), то есть тёрн и флоп не подписывают руку ни
«строго», ни «предполагая» — подписывать нечего.

**Цены у такой точки нет.** `SpotKind.POSTFLOP` не входит в `JUDGED_SPOTS`,
поэтому точка не попадает ни в ранжирование, ни в сумму потерь руки, ни в
колонку `decision_points.judged`
(`test_a_turn_point_is_not_priced_and_never_enters_the_sum`).

**Границы формы** — те же, что у ривера: перед героем ставка (`to_call > 0`),
живых в руке ровно двое, карты героя известны, борд той длины, которая улице
положена. Борд берётся ПО УЛИЦУ РЕШЕНИЯ, а не целиком: на флопе в переборе три
карты, на тёрне четыре, и карты, которые в момент решения ещё не сданы, в расчёт
не входят (`test_the_flop_numbers_do_not_move_when_later_cards_change`).
"""

from __future__ import annotations

from harness.analysis.classifier import unjudged_point
from harness.analysis.tools.turn_flop_call import (
    TurnFlopCallRequirement,
    turn_flop_call_requirement,
)
from harness.contracts import (
    TURN_FLOP_CALL_DETAIL,
    DecisionPoint,
    EnrichedHand,
    PointVerdict,
    SpotKind,
    Street,
    TurnFlopCallDetail,
)

__all__ = ["turn_flop_verdict"]

_HERO_CARDS = 2
_HEADS_UP = 2

# Карты, лежащие на столе к моменту решения на каждой из двух улиц, и сколько их
# там. Борд собирается по этому списку, а не по всему `boards`, где лежат и
# карты, которые сдадут позже; длина проверяется отдельно, потому что борд из
# трёх карт — законный вход перебора, и точка тёрна с недосданным бордом иначе
# посчиталась бы как флоп и была бы подписана «Тёрн»
# (`test_a_turn_without_its_card_is_not_counted_as_a_flop`).
_VISIBLE_BOARD: dict[Street, tuple[tuple[Street, ...], int]] = {
    Street.FLOP: ((Street.FLOP,), 3),
    Street.TURN: ((Street.FLOP, Street.TURN), 4),
}

# Причина, по которой у посчитанной точки нет лучшего действия. Строку видит
# только разбор: игроку показываются числа, а не то, чего расчёт не решает
# (SESSIONS_UX — не рассказывать о том, чего не умеем).
_NO_BEST_ACTION = "требование к ставящему диапазону посчитано, лучшего действия нет"

_TOOL = "turn_flop_call"


def turn_flop_verdict(dp: DecisionPoint, en: EnrichedHand) -> PointVerdict | None:
    """Вердикт по точке тёрна или флопа — или `None`, если точка не с этих улиц."""
    visible = _VISIBLE_BOARD.get(dp.street)
    if visible is None:
        return None
    streets, board_size = visible

    hero = en.hand.dealt.get(en.hand.hero_label, [])
    board = [card for street in streets for card in en.hand.boards.get(street, [])]
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
    if len(board) != board_size:
        return unjudged_point(
            dp, SpotKind.POSTFLOP, f"борд из {len(board)} карт, а на этой улице их {board_size}"
        )

    try:
        requirement = turn_flop_call_requirement(
            (hero[0], hero[1]), board, dp.pot_before, dp.to_call
        )
    except ValueError as failure:
        # Карты, которых нет в колоде, борд не той длины или разъехавшийся банк:
        # считать по ним нельзя, а подправить их значило бы соврать про деньги
        # (CLAUDE.md).
        return unjudged_point(dp, SpotKind.POSTFLOP, str(failure))

    detail = {TURN_FLOP_CALL_DETAIL: _detail(requirement, dp).model_dump(mode="json")}
    return unjudged_point(dp, SpotKind.POSTFLOP, _NO_BEST_ACTION, detail, tools=[_TOOL])


def _detail(requirement: TurnFlopCallRequirement, dp: DecisionPoint) -> TurnFlopCallDetail:
    """Что из перебора уезжает в вердикт: цена решения и требование к диапазону.

    Разложение борда (сколько комбо бьёт героя, какова эквити против вэлью)
    сюда не едет: игроку показывается ровно то, что лежит здесь, а перебор
    возможного — не диапазон соперника.
    """
    value = requirement.min_value_combos
    bluffs = requirement.bluffs_needed_min_value
    return TurnFlopCallDetail(
        pot_before=dp.pot_before,
        to_call=dp.to_call,
        required_equity=requirement.required_equity,
        min_value_combos=value,
        bluffs_needed_min_value=bluffs,
        # Ноль при нулевом знаменателе — вэлью в минимуме нет вовсе, требования
        # к диапазону тоже, и доли в нём не существует. Строку про блефы
        # изложение на такой точке не печатает.
        bluff_share=(
            None if bluffs is None else bluffs / (value + bluffs) if value + bluffs > 0 else 0.0
        ),
    )
