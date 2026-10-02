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

MISMATCH_BELOW = 0.30

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
    """Есть ли в справочнике чарт для такого стола, позиции героя и типа анте.

    Нужен диспетчеру (`preflop.verdict_for`): на 13–15bb точку без чарта судит
    равновесие пуш-фолда, как до появления справочника, а не оставляет без
    вердикта. Нечитаемый справочник — тоже «чарта нет»: причину назовёт сам
    вердикт, если точка до него дойдёт.
    """
    try:
        book = load_chart_book()
        book.nearest(len(state.seats), state.hero.position, open_depth_bb(state), ante_type)
    except (ChartMissing, DepthNotCharted, ChartFileError):
        return False
    return True


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
    depth = open_depth_bb(state)
    try:
        book = load_chart_book()
        key = book.nearest(seats, position, depth, en.hand.ante_type)
        strategy = book.get(key)
        entry = book.entry(key)
    except ChartMissing:
        return refused(
            f"чарта открытия для стола на {seats} мест, позиции {position} и анте "
            f"«{en.hand.ante_type}» в справочнике нет"
        )
    except ChartPlaceholder:
        return refused(
            f"чарт для позиции {position} на этой глубине помечен образцом формата — "
            f"эталоном он не служит"
        )
    except DepthNotCharted:
        return refused(f"стек {depth:.1f}bb мельче, чем судит справочник чартов")
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
        },
    )


def _hero_class(hand: CanonicalHand) -> str | None:
    cards = hand.dealt.get(hand.hero_label, [])
    if len(cards) != 2:
        return None
    return class_of(*cards)
