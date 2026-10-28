"""Стартовый диапазон соперника по чарту и его сужение агентом (спека 2026-10-14, §6).

**Стартовый — только из чарта.** Линию соперника до флопа, глубину, раскладку и
анте код читает из руки сам; модель называет только позицию соперника и улицу.
Опен-чарт даёт диапазон открывшего рейзом или олл-ином и лимп SB, чарт защиты —
колл или 3-бет BB против одного опена рейзом и чек BB после лимпа SB (решение
владельца 2026-10-14). Стол на 4–7 мест — 8-max по позиции от конца
(`open_chart.chart_table`, `defend_table`): это правило, а не допущение. Чего в
чартах нет — `RangeRefused` словами, и своего диапазона никто не подставляет
(`test_lines_outside_the_charts_are_refused`,
`test_an_isolation_after_the_sb_limp_is_refused`,
`test_a_call_against_an_all_in_open_is_not_the_call_against_a_raise`); действие,
которым чарт не играет ни одной руки, — тоже отказ, а не пустой диапазон
(`test_an_empty_chart_range_is_refused_not_an_empty_start`).

**Глубина** — в мерах чарта, стек до анте: у открывшего (и у лимпившего SB) — его
стек, но не больше самого глубокого из не сбросивших до его хода; у BB — меньший из
стеков BB и открывшего. Чарт — ближайшей снятой глубины; у чарта защиты — не дальше
`DEFEND_DEPTH_TOLERANCE_BB`, иначе отказ.

**Хедз-ап — на точке решения героя** (`live_at`): мультивей — отказ, префлоп считается
наравне с улицами после флопа.

**Сужение — подмножество, целыми классами** (`narrow`): класс вне стартового или с
долей 0 — отказ; оставленные классы сохраняют долю чарта, модель весов не задаёт.
Граница улицы — сужение строго предыдущей улицы того же соперника в той же руке
(`prior`): сужение улицы обязано быть его подмножеством, иначе отказ с убранными
классами (`test_a_class_removed_on_the_flop_is_refused_on_the_turn_by_name`). Тёрн без
сужения флопа и ривер без сужения тёрна — отказ «сначала сузьте …»
(`test_a_later_street_without_the_previous_narrowing_is_refused`); флоп без него — от
стартового (решение владельца 2026-10-23). Тёрн, прошедший чек-чек (ставок на нём не
было), ривер сужает и из сужения флопа (`previous_streets`, решение владельца
2026-10-28; `test_a_checked_through_turn_lets_the_river_narrow_from_the_flop`).

**Добавление по просьбе игрока** (`added`, план 2026-10-17, п. 9) — единственный путь
класса вне чарта в диапазон: только класс, которого в стартовом нет, и только если игрок
назвал его записью в репликах этой руки (`classes_named`); доля — 1.0, полные комбо
(`test_a_class_the_player_named_is_added_with_full_combos`). Добавленный класс входит в
сужение своей улицы и на следующей наследуется как обычный
(`test_an_added_class_is_inherited_by_the_next_street`).

**Чего нет в колле защиты** (`absent_groups`, п. 8): у колла и чека BB — классы
заметных групп, которых в этом действии нет вовсе, а чарт той же записи играет их
рейзом или олл-ином (`test_the_absent_groups_are_the_classes_the_chart_raises_or_shoves`).

**Мёртвые карты** — карты героя и борд улицы — убираются из комбо (`live_combos`);
полностью заблокированный диапазон — отказ, а не эквити 0.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from harness.analysis.charts import (
    ChartFileError,
    ChartMissing,
    ChartPlaceholder,
    DepthNotCharted,
    NotationError,
    load_chart_book,
    load_defend_book,
    parse_range,
)
from harness.analysis.open_chart import chart_table, defend_table
from harness.analysis.range_categories import (
    CategoryError,
    expand_hands,
    kept_by_category,
    layout,
)
from harness.analysis.tools.equity import combos_of_class
from harness.contracts import (
    AbsentGroup,
    AbsentGroupName,
    ActionKind,
    CanonicalAction,
    CanonicalHand,
    CategoryGroup,
    CategoryKept,
    Range,
    RangeLine,
    RangeNarrowing,
    Street,
    all_classes,
)

__all__ = [
    "DEFEND_DEPTH_TOLERANCE_BB",
    "EquitySpot",
    "LayoutOf",
    "RangeRefused",
    "RangeSpot",
    "StartRange",
    "absent_groups",
    "board_at",
    "classes_named",
    "equity_spot",
    "hero_cards",
    "layout_source",
    "live_at",
    "live_combos",
    "narrow",
    "previous_streets",
    "prior_range",
    "range_spot",
    "require_previous",
    "start_range",
    "turn_checked_through",
]

_STREETS: list[Street] = list(Street)
_HEADS_UP = 2

# Чарт защиты берётся ближайшей глубины, но не дальше этого от глубины соперника
# (включительно): снят пока один, 25bb, и BB на 100bb не должен молча получать
# диапазон 25bb. Решение оркестратора 2026-10-15, владельцу на подтверждение; у
# опен-чарта допуска нет — его глубины сняты от 15 до 60bb
# (`test_a_defence_chart_further_than_ten_bb_away_is_refused`,
# `test_the_open_chart_keeps_the_nearest_depth_without_a_limit`).
DEFEND_DEPTH_TOLERANCE_BB = 10.0
_OPEN_LINES: tuple[RangeLine, ...] = ("open_raise", "open_allin", "limp")


class RangeRefused(ValueError):
    """Диапазона по чарту нет — причина словами для модели. `ValueError`, чтобы
    `calcs.routing._perform` вернул её модели как «Расчёт не выполнен: …»."""


@dataclass(frozen=True)
class RangeSpot:
    label: str
    position: str
    opener_position: str
    line: RangeLine
    seats: int
    depth_bb: float
    ante_type: str


@dataclass(frozen=True)
class StartRange:
    spot: RangeSpot
    range: Range
    chart: Literal["open", "defend"]
    chart_seats: int
    chart_position: str
    vs_chart_position: str | None
    chart_depth_bb: float
    absent: tuple[AbsentGroup, ...] = ()


@dataclass(frozen=True)
class EquitySpot:
    hero: tuple[str, str]
    board: list[str]
    start: StartRange
    narrowed: Range
    live_combos: float
    added: tuple[str, ...] = ()
    agent_added: dict[str, str] | None = None
    categories_of: Literal["start", "flop", "turn", "flop_turn_checked"] | None = None
    categories: tuple[CategoryKept, ...] = ()
    narrowed_layout: tuple[CategoryGroup, ...] = ()


def range_spot(hand: CanonicalHand, position: str) -> RangeSpot:
    """Линия соперника до флопа и ключ чарта — из руки; вне чартов — `RangeRefused`."""
    by_position = {p.position: p for p in hand.players}
    opponent = by_position.get(position)
    if opponent is None:
        raise RangeRefused(f"за этим столом нет позиции {position}")
    if opponent.label == hand.hero_label:
        raise RangeRefused(f"{position} — позиция героя, а не соперника")
    stack = {p.label: p.stack for p in hand.players}
    position_of = {p.label: p.position for p in hand.players}
    preflop = [a for a in hand.actions if a.street is Street.PREFLOP]
    if any(a.label == opponent.label and a.kind is ActionKind.FOLD for a in preflop):
        raise RangeRefused(f"{position} сбросил карты до флопа")
    seats = len(hand.players)

    def spot(line: RangeLine, opener: str, depth_chips: int) -> RangeSpot:
        return RangeSpot(
            label=opponent.label, position=position, opener_position=position_of[opener],
            line=line, seats=seats, depth_bb=depth_chips / hand.bb, ante_type=hand.ante_type,
        )

    def first_in_depth(at: int) -> int:
        """Стек вошедшего первым, но не больше самого глубокого из не сбросивших до него."""
        who = preflop[at].label
        folded = {a.label for a in preflop[:at] if a.kind is ActionKind.FOLD}
        others = [s for label, s in stack.items() if label != who and label not in folded]
        return min(stack[who], max(others)) if others else stack[who]

    raise_at = [i for i, a in enumerate(preflop) if a.kind is ActionKind.RAISE]
    entered_at = [i for i, a in enumerate(preflop) if a.kind is not ActionKind.FOLD]
    if entered_at and _is_sb_limp(preflop[entered_at[0]], position_of):
        limper = preflop[entered_at[0]].label
        if raise_at:
            raise RangeRefused(
                "после лимпа SB был рейз или олл-ин — изоляция BB и ответ SB на неё в этом "
                "срезе не поддерживаются, поддержана только линия «лимп SB, чек BB»"
            )
        if opponent.label == limper:
            return spot("limp", limper, first_in_depth(entered_at[0]))
        if position != "BB":
            raise RangeRefused("после лимпа SB в чартах есть только линии SB и BB")
        return spot("check", limper, min(stack[opponent.label], stack[limper]))
    if not raise_at:
        raise RangeRefused("до флопа никто не повышал — для лимп-банка чартов нет")
    opened_at = raise_at[0]
    opening = preflop[opened_at]
    if any(a.kind is ActionKind.CALL for a in preflop[:opened_at]):
        raise RangeRefused("до опена был лимп — такой линии в чартах нет")
    opener = opening.label

    if opponent.label == opener:
        if len(raise_at) > 1:
            raise RangeRefused("после опена был 3-бет — линии с 3-бетом для открывшего в чартах нет")
        return spot(
            "open_allin" if opening.is_all_in else "open_raise", opener, first_in_depth(opened_at)
        )
    if position != "BB":
        raise RangeRefused("чарты колла и 3-бета есть только для BB")
    if opening.is_all_in:
        raise RangeRefused(
            "опен был олл-ином — колл и 3-бет против опена олл-ином не поддерживаются: "
            "чарт защиты снят против опена рейзом"
        )
    own_at = [i for i, a in enumerate(preflop) if a.label == opponent.label]
    if not own_at:
        raise RangeRefused("BB до флопа не действовал — линии нет")
    if any(a.kind is not ActionKind.FOLD for a in preflop[opened_at + 1 : own_at[0]]):
        raise RangeRefused(
            "между опеном и BB кто-то вошёл — защиты против нескольких игроков в чартах нет"
        )
    first = preflop[own_at[0]]
    if first.kind is ActionKind.CALL and len(raise_at) == 1:
        line: RangeLine = "call"
    elif first.kind is ActionKind.RAISE and len(raise_at) == 2 and raise_at[1] == own_at[0]:
        line = "allin" if first.is_all_in else "three_bet"
    else:
        raise RangeRefused(
            "линия BB до флопа — не колл и не 3-бет против одного опена: в чартах её нет"
        )
    return spot(line, opener, min(stack[opponent.label], stack[opener]))


def _is_sb_limp(action: CanonicalAction, position_of: dict[str, str]) -> bool:
    return action.kind is ActionKind.CALL and position_of[action.label] == "SB"


def start_range(
    spot: RangeSpot, *, open_path: Path | None = None, defend_path: Path | None = None
) -> StartRange:
    """Стартовый диапазон по чарту ближайшей глубины. Отказы словами: зона пуш-фолда,
    чарта нет, образец формата, справочник не прочитался, действие без единой руки."""
    try:
        start = _charted(spot, open_path, defend_path)
    except DepthNotCharted as exc:
        raise RangeRefused("стек в зоне пуш-фолда — чарта для этого спота нет") from exc
    except (ChartMissing, ChartPlaceholder) as exc:
        raise RangeRefused("чарта для этого спота нет") from exc
    except ChartFileError as exc:
        raise RangeRefused("справочник чартов не прочитался — чарта для этого спота нет") from exc
    if not any(weight > 0.0 for weight in start.range.weights.values()):
        raise RangeRefused(
            "этим действием чарт на этой глубине не играет ни одной руки — диапазона нет"
        )
    return start


def _charted(spot: RangeSpot, open_path: Path | None, defend_path: Path | None) -> StartRange:
    if spot.line in _OPEN_LINES:
        chart_seats, chart_position = chart_table(spot.seats, spot.position)
        book = load_chart_book(open_path)
        key = book.nearest(chart_seats, chart_position, spot.depth_bb, spot.ante_type)
        opened = book.get(key)
        rng = {
            "open_raise": opened.raise_range,
            "open_allin": opened.allin_range,
            "limp": opened.limp_range,
        }[spot.line]
        return StartRange(spot, rng, "open", chart_seats, chart_position, None, key.depth_bb)
    chart_seats, chart_position, chart_vs = defend_table(
        spot.seats, spot.position, spot.opener_position
    )
    defence_book = load_defend_book(defend_path)
    defence_key = defence_book.nearest(
        chart_seats, chart_position, chart_vs, spot.depth_bb, spot.ante_type,
        "limp" if spot.line == "check" else "raise",
    )
    if abs(defence_key.depth_bb - spot.depth_bb) > DEFEND_DEPTH_TOLERANCE_BB:
        raise RangeRefused(
            f"чарта защиты для такой глубины нет: ближайший {defence_key.depth_bb:g} BB, "
            f"а у соперника {spot.depth_bb:.1f} BB"
        )
    defence = defence_book.get(defence_key)
    rng = {
        "call": defence.call_range,
        "check": defence.check_range,
        "three_bet": defence.raise_range,
        "allin": defence.allin_range,
    }[spot.line]
    absent = (
        absent_groups(rng, defence.raise_range, defence.allin_range)
        if spot.line in ("call", "check")
        else ()
    )
    return StartRange(
        spot, rng, "defend", chart_seats, chart_position, chart_vs, defence_key.depth_bb, absent
    )


def _group_of(cls: str) -> AbsentGroupName | None:
    if len(cls) == 2:
        return "pairs"
    if cls[0] == "A":
        return "suited_aces" if cls[2] == "s" else "offsuit_aces"
    if cls[0] in _BROADWAY and cls[1] in _BROADWAY:
        return "broadways"
    return None


_BROADWAY = frozenset("KQJT")
_GROUPS: tuple[AbsentGroupName, ...] = ("pairs", "suited_aces", "offsuit_aces", "broadways")


def absent_groups(
    played: Range, raised: Range, all_in: Range
) -> tuple[AbsentGroup, ...]:
    """Заметные группы (пары, одномастные и разномастные тузы, бродвеи без туза), в
    которых есть класс без доли в `played` (колл или чек), но с долей в `raised` или
    `all_in` той же записи; класс с любой ненулевой долей в `played` отсутствующим не
    считается (`test_a_class_partly_called_is_not_absent`). Группы без таких классов не
    возвращаются."""
    found: dict[AbsentGroupName, list[str]] = {group: [] for group in _GROUPS}
    members: dict[AbsentGroupName, int] = dict.fromkeys(_GROUPS, 0)
    for cls in all_classes():
        group = _group_of(cls)
        if group is None:
            continue
        members[group] += 1
        if played.weight(cls) <= 0.0 and (raised.weight(cls) > 0.0 or all_in.weight(cls) > 0.0):
            found[group].append(cls)
    return tuple(
        AbsentGroup(
            group=group,
            classes=classes,
            whole=len(classes) == members[group],
            raised=any(raised.weight(cls) > 0.0 for cls in classes),
            all_in=any(all_in.weight(cls) > 0.0 for cls in classes),
        )
        for group, classes in found.items()
        if classes
    )


def hero_cards(hand: CanonicalHand) -> list[str]:
    return list(hand.dealt.get(hand.hero_label, []))


def board_at(hand: CanonicalHand, street: Street) -> list[str]:
    """Борд, открытый к улице точки решения; улицы, до которой раздача не дошла, —
    отказ."""
    if street is not Street.PREFLOP and not hand.boards.get(street):
        raise RangeRefused("до этой улицы раздача не дошла")
    upto = _STREETS[1 : _STREETS.index(street) + 1]
    return [card for s in upto for card in hand.boards.get(s, [])]


def live_at(hand: CanonicalHand, street: Street) -> list[str]:
    """Места в раздаче на точке решения героя — его последнем ходе на улице: сбросившие
    раньше, в том числе на этой же улице, выбыли
    (`test_preflop_equity_counts_the_players_at_the_hero_decision`,
    `test_a_third_player_folding_before_the_hero_decision_leaves_a_heads_up_flop`).
    Хода у героя на улице нет — отказ
    (`test_no_hero_turn_on_the_street_is_no_decision_point`)."""
    turns = [
        i for i, a in enumerate(hand.actions)
        if a.street is street and a.label == hand.hero_label
    ]
    if not turns:
        raise RangeRefused("у героя на этой улице нет хода — точки решения нет")
    folded = {a.label for a in hand.actions[: turns[-1]] if a.kind is ActionKind.FOLD}
    return [p.label for p in hand.players if p.label not in folded]


def live_combos(rng: Range, dead: Collection[str]) -> float:
    """Взвешенные комбо диапазона без мёртвых карт."""
    blocked = set(dead)
    return float(sum(
        weight * sum(1 for c1, c2 in combos_of_class(cls) if c1 not in blocked and c2 not in blocked)
        for cls, weight in rng.weights.items()
    ))


_STREET_AT: dict[Street, str] = {
    Street.PREFLOP: "до флопа", Street.FLOP: "на флопе", Street.TURN: "на тёрне",
    Street.RIVER: "на ривере",
}
_STREET_OF: dict[Street, str] = {
    Street.PREFLOP: "префлопа", Street.FLOP: "флопа", Street.TURN: "тёрна",
    Street.RIVER: "ривера",
}
# Улица, сужение которой обязательно до сужения этой, и отказ без него (решение
# владельца 2026-10-23: прежняя граница «ближайшая более ранняя, иначе стартовый»
# пускала тёрн шире фактического флопа, срезанного из истории).
_PREVIOUS_REQUIRED: dict[Street, tuple[Street, str]] = {
    Street.TURN: (
        Street.FLOP,
        (
            "сначала сузьте флоп: вызовите equity_vs_range для флопа этого соперника, тёрн "
            "сужается из него"
        ),
    ),
    Street.RIVER: (
        Street.TURN,
        (
            "сначала сузьте тёрн: вызовите equity_vs_range для тёрна этого соперника, ривер "
            "сужается из него"
        ),
    ),
}


# Ривер после чек-чек тёрна сужается из сужения флопа (решение владельца 2026-10-28):
# без него — отказ с этой причиной.
_FIRST_THE_FLOP_TURN_CHECKED = (
    "сначала сузьте флоп: вызовите equity_vs_range для флопа этого соперника — на тёрне "
    "ставок не было, ривер сужается из сужения флопа"
)


def turn_checked_through(hand: CanonicalHand) -> bool:
    """Тёрн роздан, и ставок на нём не было ни у кого — ни бета, ни рейза; после олл-ина
    на флопе ходов на тёрне нет — ставок тоже не было
    (`test_a_checked_through_turn_lets_the_river_narrow_from_the_flop`,
    `test_a_turn_with_a_raise_or_not_dealt_is_not_checked_through`)."""
    return bool(hand.boards.get(Street.TURN)) and not any(
        a.street is Street.TURN and a.kind in (ActionKind.BET, ActionKind.RAISE)
        for a in hand.actions
    )


def previous_streets(hand: CanonicalHand, street: Street) -> tuple[Street, ...]:
    """Улицы, сужение которых годится исходным для сужения этой: тёрн — флоп; ривер —
    тёрн, а после чек-чек тёрна — и флоп; флоп и префлоп — никакая (флоп — от
    стартового)."""
    if street is Street.TURN:
        return (Street.FLOP,)
    if street is Street.RIVER:
        return (Street.TURN, Street.FLOP) if turn_checked_through(hand) else (Street.TURN,)
    return ()


def require_previous(
    street: Street,
    prior: RangeNarrowing | None,
    previous: Collection[Street],
    reset: Collection[Street] = (),
) -> None:
    """Отказ словами, если исходного сужения нужной улицы нет (сужают по очереди по
    улицам, `test_a_later_street_without_the_previous_narrowing_is_refused`); сужение,
    сброшенное пересужением более ранней улицы (`reset`), — «флоп (префлоп) сужен заново»
    (`test_a_turn_narrowing_reset_by_a_new_flop_is_refused_by_name`,
    `test_a_flop_narrowing_reset_by_a_new_preflop_is_refused_by_name`); причина — по
    самой ранней сброшенной улице
    (`test_a_river_after_a_new_preflop_names_the_preflop_reset`)."""
    if not previous or (prior is not None and prior.street in previous):
        return
    required = _PREVIOUS_REQUIRED[street][0]
    chain = set(_STREETS[_STREETS.index(Street.FLOP) : _STREETS.index(required) + 1])
    dropped = [st for st in _STREETS if st in reset and st in chain and st in _RESET_REFUSAL]
    if dropped:
        # Причина — по самой ранней сброшенной улице: «префлоп сужен заново» важнее тёрна.
        raise RangeRefused(_RESET_REFUSAL[dropped[0]])
    if street is Street.RIVER and Street.FLOP in previous:
        raise RangeRefused(_FIRST_THE_FLOP_TURN_CHECKED)
    raise RangeRefused(_PREVIOUS_REQUIRED[street][1])


# Сужение улицы сброшено пересужением более ранней (решение владельца 2026-10-28).
_RESET_REFUSAL: dict[Street, str] = {
    Street.TURN: (
        "сначала сузьте тёрн: флоп сужен заново — вызовите equity_vs_range для тёрна этого "
        "соперника, ривер сужается из него"
    ),
    Street.FLOP: (
        "сначала сузьте флоп: префлоп сужен заново — вызовите equity_vs_range для флопа "
        "этого соперника, тёрн сужается из него"
    ),
}


def prior_range(start: Range, prior: RangeNarrowing) -> Range:
    """Диапазон прошлого сужения с долями: класс стартового — с долей чарта, добавленный
    вне чарта — 1.0, как его записывает `narrow`."""
    return Range(weights={
        cls: start.weight(cls) if start.weight(cls) > 0.0 else 1.0 for cls in prior.classes
    })


def _classes(text: str) -> Range:
    if ":" in text:
        raise RangeRefused("веса не задаются: перечислите классы рук, доли берутся из чарта")
    try:
        return parse_range(text)
    except (NotationError, ValueError) as exc:
        raise RangeRefused(f"запись рук не разобрана: {exc}") from exc


def narrow(
    start: Range,
    hands: str,
    *,
    street: Street | None = None,
    prior: RangeNarrowing | None = None,
    added: str | None = None,
    named: Collection[str] = (),
    kept: Collection[str] = (),
    reason: str | None = None,
    previous: Collection[Street] | None = None,
) -> Range:
    """Сужение агентом: классы без весов, только из стартового диапазона с ненулевой
    долей; доли — чарта. С `prior` (сужение предыдущей улицы `street` того же
    соперника) — только из его классов; добавленные по просьбе игрока классы `prior`
    (вне чарта) сохраняют долю 1.0. Тёрн и ривер без `prior` предыдущей улицы — отказ
    (`_PREVIOUS_REQUIRED`). `added` — классы вне чарта, только из `named` или
    `kept` — классов последнего сужения этого соперника в этой руке
    (`test_an_added_class_kept_from_the_last_narrowing_needs_no_new_request`);
    `previous` — улицы, чьё сужение годится `prior` (`previous_streets`; по умолчанию —
    строго предыдущая); с
    `reason` (причина модели) — любые классы вне чарта
    (`test_a_class_added_with_the_model_reason_needs_no_player_record`)."""
    if street is not None and street in _PREVIOUS_REQUIRED:
        require_previous(
            street, prior, previous if previous is not None else (_PREVIOUS_REQUIRED[street][0],)
        )
    asked = _classes(hands)
    if not asked.weights:
        raise RangeRefused("не названо ни одной руки")
    extra = _classes(added) if added and added.strip() else Range(weights={})
    charted = [cls for cls in all_classes() if cls in extra.weights and start.weight(cls) > 0.0]
    if charted:
        raise RangeRefused(
            "эти классы есть в стартовом диапазоне по чарту: " + ", ".join(charted)
            + " — передайте их в hands, а в added — только классы вне чарта"
        )
    allowed_extra = {*named, *kept}
    unnamed = [
        cls for cls in all_classes()
        if cls in extra.weights and cls not in allowed_extra and not reason
    ]
    if unnamed:
        raise RangeRefused(
            "игрок не назвал эти классы записью: " + ", ".join(unnamed)
            + " — вне чарта добавляются классы, которые игрок назвал сам («22-77», «A5s»), "
            "или ваши с причиной в added_reason; уточните у него"
        )
    earlier = set(prior.classes) if prior is not None else set()
    outside = [
        cls for cls in all_classes()
        if cls in asked.weights and start.weight(cls) <= 0.0 and cls not in earlier
    ]
    if outside:
        raise RangeRefused(
            "этих рук нет в стартовом диапазоне: " + ", ".join(outside)
            + " — сужение рук не добавляет; вне чарта — только параметром added и только "
            "классы, которые игрок назвал сам"
        )
    if prior is not None:
        removed = [cls for cls in all_classes() if cls in asked.weights and cls not in earlier]
        if removed:
            later = _STREET_OF[street] if street is not None else "этой улицы"
            raise RangeRefused(
                f"{_STREET_AT[prior.street]} эти классы вы уже убрали: " + ", ".join(removed)
                + f" — сужение {later} берётся из сужения {_STREET_OF[prior.street]}"
            )
    weights = {
        cls: start.weight(cls) if start.weight(cls) > 0.0 else 1.0 for cls in asked.weights
    }
    weights.update(dict.fromkeys(extra.weights, 1.0))
    return Range(weights=weights)


# Запись класса или диапазона классов в тексте игрока: «A5s», «KQo», «77», «77+»,
# «A2s-A5s», «22–77» (дефис, короткое или длинное тире, пробел вокруг допустим). Ранги —
# заглавными, суффикс — строчной; непара без суффикса («AK») — не запись (грамматика
# `parse_range`). Голая пара цифрами перед «BB», «бб» или «%» — стек или доля, не класс.
_HAND = r"[2-9TJQKA]{2}[so]?"
_NAMED_RE = re.compile(
    r"(?<![\w.,+])(" + _HAND + r")(?:\s?[-–—]\s?(" + _HAND + r")|(\+))?(?![\w+%]|[.,]\d)"
)
# Число, а не класс: за записью из одних цифр — единица, «%», «/», «:» или
# существительное счёта; перед ней — «№» или «уровень».
_UNIT_AFTER_RE = re.compile(
    r"\s*(?:%|/|:|(?:bb|бб)(?!\w)|(?:игрок|фиш|уровн|раздач|рук|мест|турнир|мин|сек|час))",
    re.IGNORECASE,
)
_MARK_BEFORE_RE = re.compile(r"(?:№|уровень|уровне|уровня)\s*$", re.IGNORECASE)


def classes_named(text: str) -> frozenset[str]:
    """Классы, которые игрок назвал записью (`_NAMED_RE`); слова («мелкие пары»)
    классов не дают (`test_words_name_no_classes`). Запись, которую `parse_range` не
    разбирает, пропускается."""
    named: set[str] = set()
    for match in _NAMED_RE.finditer(text):
        left, right, plus = match.groups()
        digits = left.isdigit() and (right is None or right.isdigit())
        if digits and (
            _UNIT_AFTER_RE.match(text, match.end())
            or _MARK_BEFORE_RE.search(text, 0, match.start())
        ):
            continue
        token = left + (f"-{right}" if right else "") + (plus or "")
        try:
            named.update(parse_range(token).weights)
        except (NotationError, ValueError):
            continue
    return frozenset(named)


def equity_spot(
    hand: CanonicalHand,
    position: str,
    street: Street,
    hands: str,
    *,
    added: str | None = None,
    prior: RangeNarrowing | None = None,
    named: Collection[str] = (),
    kept: Collection[str] = (),
    reason: str | None = None,
    kept_agent: Mapping[str, str] | None = None,
    reset: Collection[Street] = (),
    open_path: Path | None = None,
    defend_path: Path | None = None,
) -> EquitySpot:
    """Всё, что нужно расчёту эквити, кроме самого перебора: рука героя, борд улицы,
    стартовый и суженный диапазоны, их живые комбо. `added`, `prior`, `named`, `kept`,
    `reason` — как у `narrow`. Класс вне чарта — агента (`agent_added`, класс →
    причина), если добавлен в этом вызове с `reason` или был агента в `prior` или
    `kept_agent` (последнем сужении), иначе — игрока
    (`test_an_agent_class_is_inherited_by_the_next_street_with_its_reason`). После флопа
    `hands` может называть категории раскладки исходного диапазона (на флопе —
    стартового, на тёрне и ривере — сужения прошлой улицы, на ривере после чек-чек
    тёрна — и флопа) — `expand_hands`; сводка по категориям — в `categories`, раскладка
    суженного — в `narrowed_layout`
    (`test_a_river_narrowing_takes_categories_of_the_turn_narrowing`). Тёрн и ривер без
    исходного сужения — отказ раньше разбора категорий."""
    cards = hero_cards(hand)
    if len(cards) != 2:
        raise RangeRefused("карты героя неизвестны — эквити считать не от чего")
    start = start_range(range_spot(hand, position), open_path=open_path, defend_path=defend_path)
    board = board_at(hand, street)
    live = live_at(hand, street)
    if start.spot.label not in live:
        raise RangeRefused("соперник сбросил карты до точки решения героя")
    if len(live) > _HEADS_UP:
        raise RangeRefused(
            "на точке решения героя в банке больше двух игроков — мультивей в этом срезе "
            "не считается"
        )
    dead = [*cards, *board]
    previous = previous_streets(hand, street)
    categories_of, groups = layout_source(
        start.range, street, prior, board, dead, previous, reset
    )
    try:
        expanded = expand_hands(hands, groups)
    except CategoryError as exc:
        raise RangeRefused(str(exc)) from exc
    reason = reason.strip() if reason and reason.strip() else None
    narrowed = narrow(start.range, expanded, street=street, prior=prior, added=added,
                      named=named, kept=kept, reason=reason, previous=previous)
    combos = live_combos(narrowed, dead)
    if combos == 0.0:
        raise RangeRefused("все комбо этих рук заблокированы вашими картами и бордом")
    low, high = sorted(cards)
    outside = tuple(
        cls for cls in all_classes() if cls in narrowed.weights and start.range.weight(cls) <= 0.0
    )
    this_call = set(_classes(added).weights) if added and added.strip() else set()
    inherited = {**(kept_agent or {}), **(prior.agent_added if prior is not None else {})}
    agent_added: dict[str, str] = {}
    for cls in outside:
        if reason is not None and cls in this_call:
            agent_added[cls] = reason
        elif cls in inherited:
            agent_added[cls] = inherited[cls]
    return EquitySpot(
        hero=(low, high), board=board, start=start, narrowed=narrowed,
        live_combos=combos,
        added=outside,
        agent_added=agent_added,
        categories_of=categories_of,
        categories=tuple(kept_by_category(groups, narrowed)) if groups is not None else (),
        narrowed_layout=tuple(layout(narrowed, board, dead)) if groups is not None else (),
    )


LayoutOf = Literal["start", "flop", "turn", "flop_turn_checked"]


def layout_source(
    start: Range,
    street: Street,
    prior: RangeNarrowing | None,
    board: list[str],
    dead: Collection[str],
    previous: Collection[Street] = (),
    reset: Collection[Street] = (),
) -> tuple[LayoutOf | None, list[CategoryGroup] | None]:
    """Чья раскладка по категориям на этой улице и она сама: до флопа — нет; на флопе —
    стартового; на тёрне и ривере — исходного сужения (`prior` с долями — `prior_range`;
    годные улицы — `previous`), без него — `RangeRefused` «сначала сузьте …»: от
    стартового тёрн и ривер не раскладываются
    (`test_the_turn_layout_is_of_the_flop_narrowing_and_without_it_is_refused`)."""
    if street is Street.PREFLOP:
        return None, None
    if street is Street.FLOP:
        return "start", layout(start, board, dead)
    require_previous(street, prior, previous, reset)
    if prior is None:  # без `previous` исходного нет — не раскладывается
        raise RangeRefused(_PREVIOUS_REQUIRED[street][1])
    source: LayoutOf = "turn" if prior.street is Street.TURN else "flop"
    if street is Street.RIVER and prior.street is Street.FLOP:
        source = "flop_turn_checked"
    return source, layout(prior_range(start, prior), board, dead)
