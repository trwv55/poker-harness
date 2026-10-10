"""Вердикт по чарту: открытие первым в неоткрытый банк при стеке от 13bb.

Эталон — стратегия солвера из справочника владельца (`analysis.charts`): для
руки героя у чарта есть частоты рейза, шова, лимпа и фолда. Вердикт сверяет с
ними сыгранное действие и **цены не называет**: из частот её не вывести
(спека `docs/superpowers/specs/2026-10-03-open-chart-verdict.md`).

**Расхождение** — сыгранное действие встречается в чарте реже
`MISMATCH_BELOW` (30%, решение владельца) и при этом не самое частое у чарта:
смешанная рука, сыгранная любым из своих частых действий, расхождением не
является. Частоты печатаются в каждой такой точке — и при расхождении, и без.

**Зона `strict`** (решение владельца 2026-10-03): чарт — GTO, вердикт говорит об
отклонении от GTO, а не об угаданном диапазоне соперника; поправки под
конкретных соперников — комментарии игрока.

Правило «судить против диапазона, а не против вскрытой карты» выполняется по
построению: эталон — стратегия, результат раздачи в вердикт не входит.

**Стол меньше 8 мест судится 8-max чартом по позиции от конца** (решение владельца
2026-10-10, `chart_position`): позиция берётся та, у которой в 8-max столько же
игроков позади до BB (на 7 местах BTN — BTN, UTG — UTG+1). Зона та же, `strict`;
в `detail` точка несёт `chart_seats` и `chart_position` — каким чартом судили.
Стол на 8 мест судится своим чартом, как прежде; на 9 местах чарта нет.
"""

from __future__ import annotations

from harness.analysis.charts import (
    ChartFileError,
    ChartMissing,
    ChartPlaceholder,
    DepthNotCharted,
    OpenStrategy,
    load_chart_book,
)
from harness.analysis.classifier import TableState, open_depth_bb, unjudged_point
from harness.contracts import (
    ActionKind,
    CanonicalHand,
    DecisionPoint,
    EnrichedHand,
    PointVerdict,
    SpotKind,
    Zone,
    class_of,
)
from harness.normalizer import POSITIONS_BY_COUNT

MISMATCH_BELOW = 0.30

# Раскладка, для которой в справочнике есть чарты; стол меньше судится ими по
# позиции от конца (`chart_position`).
CHART_SEATS = 8

# Порядок при равных частотах «лучшего» действия: активное раньше пассивного —
# то же предпочтение, что у пуш-фолда при EV ровно ноль (`preflop._best_of`).
_ACTIONS: tuple[str, ...] = ("raise", "shove", "limp", "fold")


def taken_token(dp: DecisionPoint, state: TableState) -> str:
    """Сыгранное действие в словаре чарта: fold, raise, shove или limp.

    Размер рейза не важен. Шов — любое действие, после которого у героя не
    осталось фишек (рейз в олл-ин или колл блайнда на весь стек); колл без
    олл-ина в неоткрытом банке — лимп (бывает только у SB).
    """
    kind = dp.action.kind
    if kind is ActionKind.FOLD:
        return "fold"
    if state.hero_all_in_after:
        return "shove"
    if kind in (ActionKind.BET, ActionKind.RAISE):
        return "raise"
    return "limp"


def frequencies(strategy: OpenStrategy, hero_cls: str) -> dict[str, float]:
    """Доли четырёх действий у класса руки по чарту, в сумме 1."""
    return {
        "raise": strategy.raise_range.weight(hero_cls),
        "shove": strategy.allin_range.weight(hero_cls),
        "limp": strategy.limp_range.weight(hero_cls),
        "fold": strategy.fold_weight(hero_cls),
    }


def most_frequent(freqs: dict[str, float]) -> str:
    """Самое частое действие у чарта; при равенстве — активное (`_ACTIONS`)."""
    return max(_ACTIONS, key=lambda action: (freqs[action], -_ACTIONS.index(action)))


def chart_exists(state: TableState, ante_type: str) -> bool:
    """Есть ли в справочнике СВОЙ чарт стола — той же раскладки, позиции героя и анте.

    Нужен диспетчеру (`preflop.verdict_for`): на 13–15bb шов и фолд за столом без
    своего чарта судит равновесие пуш-фолда, посчитанное для этого стола, как до
    появления справочника; 8-max чарт по позиции от конца (`chart_table`) его не
    заменяет (`test_a_table_without_its_own_chart_keeps_the_push_fold_verdict_up_to_15bb`).
    Нечитаемый справочник — тоже «чарта нет»: причину назовёт сам вердикт, если
    точка до него дойдёт.
    """
    try:
        book = load_chart_book()
        book.nearest(len(state.seats), state.hero.position, open_depth_bb(state), ante_type)
    except (ChartMissing, DepthNotCharted, ChartFileError):
        return False
    return True


def chart_table(seats: int, position: str) -> tuple[int, str]:
    """Раскладка и позиция чарта, которым судится открытие за столом на `seats` мест.

    Стол меньше `CHART_SEATS` — 8-max и позиция с тем же числом игроков позади до
    BB (`test_a_seven_handed_button_is_judged_by_the_eight_max_button`,
    `test_a_seven_handed_utg_is_judged_by_the_eight_max_utg1`,
    `test_a_six_handed_table_is_judged_by_position_from_the_end`); остальные — как
    есть.
    """
    if seats >= CHART_SEATS or seats not in POSITIONS_BY_COUNT:
        return seats, position
    order = _preflop_order(seats)
    if position not in order:
        return seats, position
    behind = len(order) - 1 - order.index(position)
    chart_order = _preflop_order(CHART_SEATS)
    return CHART_SEATS, chart_order[len(chart_order) - 1 - behind]


def _preflop_order(seats: int) -> list[str]:
    """Порядок хода на префлопе: от первого после BB до BB; в хедз-апе первой
    ходит кнопка (она же SB)."""
    order = POSITIONS_BY_COUNT[seats]
    return list(order) if seats == 2 else [*order[2:], *order[:2]]


def open_chart_verdict(dp: DecisionPoint, en: EnrichedHand, state: TableState) -> PointVerdict:
    """Вердикт по чарту для точки открытия первым; без чарта — точка без вердикта."""
    spot = SpotKind.OPEN_CHART
    taken = taken_token(dp, state)

    def refused(reason: str) -> PointVerdict:
        # Действие — в словаре чарта и у точки без вердикта: у одного спота в
        # `decision_points.action_taken` один словарь (лимп SB — `limp`, не `call`).
        return unjudged_point(dp, spot, reason).model_copy(update={"action_taken": taken})

    hero_cls = _hero_class(en.hand)
    if hero_cls is None:
        return refused("карты героя неизвестны")
    seats = len(state.seats)
    position = state.hero.position
    chart_seats, chart_pos = chart_table(seats, position)
    depth = open_depth_bb(state)
    try:
        book = load_chart_book()
        key = book.nearest(chart_seats, chart_pos, depth, en.hand.ante_type)
        strategy = book.get(key)
        entry = book.entry(key)
    except ChartMissing:
        return refused(_missing_reason(seats, chart_seats, chart_pos, en.hand.ante_type))
    except ChartPlaceholder:
        return refused(
            f"чарт для позиции {position} на этой глубине помечен образцом формата — "
            f"эталоном он не служит"
        )
    except DepthNotCharted:
        return refused(f"стек {depth:.1f}BB мельче, чем судит справочник чартов")
    except ChartFileError:
        return refused("справочник чартов не прочитался — сверить открытие не с чем")

    freqs = frequencies(strategy, hero_cls)
    best = most_frequent(freqs)
    taken_frequency = freqs[taken]
    mismatch = taken != best and taken_frequency < MISMATCH_BELOW
    return PointVerdict(
        dp_index=dp.index,
        street=dp.street,
        spot=spot,
        zone=Zone.STRICT,
        action_taken=taken,
        best_action=best,
        # Цены у точки по чарту нет; число — заглушка контракта, расхождение
        # названо в `mismatch` (`contracts.is_mismatch`).
        ev_diff_bb=0.0,
        mismatch=mismatch,
        tools=["open_chart"],
        detail={
            "method": "open_chart",
            "hero_class": hero_cls,
            "open_depth_bb": round(depth, 2),
            "chart_depth_bb": key.depth_bb,
            "chart_frequencies": {action: round(freqs[action], 4) for action in _ACTIONS},
            "taken_frequency": round(taken_frequency, 4),
            "chart_source": entry.source,
            "chart_revised_at": entry.revised_at.isoformat(),
            **(
                {"chart_seats": chart_seats, "chart_position": chart_pos}
                if chart_seats != seats
                else {}
            ),
        },
    )


def _missing_reason(seats: int, chart_seats: int, chart_pos: str, ante_type: str) -> str:
    """Чего именно нет в справочнике: раскладки, позиции или типа анте.

    Называется первое недостающее, а не все три сразу: «нет чарта для 7 мест,
    позиции BTN и анте per_player» обвиняла бы анте, которое в справочнике есть
    (`test_a_missing_chart_names_the_real_reason`).
    """
    keys = load_chart_book().all_keys()
    if not any(key.seats == chart_seats for key in keys):
        return f"чарта открытия для стола на {seats} мест в справочнике нет"
    if not any(key.seats == chart_seats and key.position == chart_pos for key in keys):
        return f"чарта открытия для позиции {chart_pos} стола на {chart_seats} мест в справочнике нет"
    return f"чарта открытия для анте «{ante_type}» в справочнике нет"


def _hero_class(hand: CanonicalHand) -> str | None:
    cards = hand.dealt.get(hand.hero_label, [])
    if len(cards) != 2:
        return None
    return class_of(*cards)
