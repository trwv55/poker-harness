"""Отчёт по турниру: факты и статистика, посчитанные кодом (задача 23).

Две трети того, что игрок хочет видеть после турнира, — ФАКТЫ: сколько раздач
и уровней сыграно, как ходил стек, где ушли фишки, чем кончились олл-ины. Для
них модель вердикта не нужна вовсе, они считаются из истории рук напрямую —
и потому доходят до игрока даже там, где покрытие разбора мало. Вердиктная
часть (`findings`) ограничена тем, что разбор УЖЕ судит, и всегда идёт с
покрытием рядом.

**Ни одного слова здесь нет.** Модуль возвращает `TournamentReport` —
структуру чисел; прозу по ней пишет LLM (задача 21), а строки игроку собирает
`presentation.messages`. Граница ровно та же, что у всего конвейера: точное
считает код.

**Дисперсия отделена от ошибки, потому что это разные измерители.** Цена
расхождения считается в EV на момент решения и против диапазона; потеря стека
— по факту раздачи. `EvSplit` держит их порознь и не складывает: проигранный
олл-ин, к которому у расчёта нет претензий, попадает в фишечный столбец
дисперсии и ни одной фишкой не попадает в судимый (CLAUDE.md: правильный вход,
проигравший по случайности, ошибкой не считается).

**Вход — аргументы, не база.** Руки и сводка скана приходят списком, история
игрока (все его турниры, прошлые сводки) — тоже аргументами; запросы живут в
`memory.repos`, склейка — в `worker.pipeline`. Из счётного пакета здесь не
используется ничего: модуль импортирует только контракты и `player_stats`, и
потому дёшев — ни `pokerkit`, ни `eval7`.
"""

from __future__ import annotations

from collections.abc import Sequence

from harness.analysis.player_stats import player_stats
from harness.contracts import (
    AllInEvent,
    CanonicalAction,
    CanonicalHand,
    ChipMove,
    EnrichedHand,
    EvSplit,
    Finding,
    LevelLine,
    PlayerState,
    ScanItem,
    ScanSummary,
    StackTrajectory,
    Street,
    TournamentReport,
    Zone,
    class_of,
)

__all__ = ["tournament_report"]

# Точность, до которой округляются bb в отчёте. Та же, что у `scan_tournament`
# (`round(total_loss_bb, 6)`): числа сюда приходят делением фишек на bb, и
# двоичный хвост деления не должен оседать в контракте.
_BB_PRECISION = 6


def _hero(hand: CanonicalHand) -> PlayerState:
    """Место героя за столом. Его отсутствие — не пробел данных, а чужая раздача."""
    for player in hand.players:
        if player.label == hand.hero_label:
            return player
    raise ValueError(f"в раздаче {hand.hand_no} нет места героя ({hand.hero_label})")


def _bb(value: int, hand: CanonicalHand) -> float:
    return round(value / hand.bb, _BB_PRECISION)


def _delta_bb(en: EnrichedHand) -> float:
    """Изменение стека героя за раздачу, в bb её уровня.

    Считается по стекам движка (`stacks_end`), а не по строкам выплат источника:
    движок проигрывает руку сам и не верит записанным суммам (ARCHITECTURE.md).
    """
    hand = en.hand
    return _bb(en.report.stacks_end[hand.hero_label] - _hero(hand).stack, hand)


def _hero_class(hand: CanonicalHand) -> str:
    cards = hand.dealt.get(hand.hero_label, [])
    return class_of(*cards) if len(cards) == 2 else ""


def _hero_actions(hand: CanonicalHand) -> list[CanonicalAction]:
    return [a for a in hand.actions if a.label == hand.hero_label]


def _went_all_in(hand: CanonicalHand) -> bool:
    """Герой отправил фишки в банк целиком — по пометке источника на его действии.

    Олл-ин с вынужденной ставки (блайнд забрал последние фишки, действия у
    героя нет вовсе) сюда не попадает: такой раздачи нет в списке олл-инов, и
    её потеря считается обычной потерей блайнда.
    """
    return any(a.is_all_in for a in _hero_actions(hand))


def _last_street(hand: CanonicalHand) -> Street:
    """Улица последнего действия героя; префлоп — если герой не действовал вовсе."""
    actions = _hero_actions(hand)
    return actions[-1].street if actions else Street.PREFLOP


def _showdown(hand: CanonicalHand) -> bool:
    return any(entry.label == hand.hero_label for entry in hand.showdowns)


def _trajectory(enriched: Sequence[EnrichedHand]) -> StackTrajectory:
    """Стек по уровням и переломная точка — раздача с максимальным стеком на входе.

    Уровни идут в порядке игры и группируются подряд: внутри турнира раздачи
    одного уровня идут одна за другой, и порядок раздач здесь — тот, в котором
    они сыграны.
    """
    levels: list[LevelLine] = []
    for en in enriched:
        hand = en.hand
        start_bb = _bb(_hero(hand).stack, hand)
        end_bb = _bb(en.report.stacks_end[hand.hero_label], hand)
        if levels and levels[-1].level == hand.level:
            levels[-1].hands += 1
            levels[-1].end_bb = end_bb
        else:
            levels.append(
                LevelLine(level=hand.level, hands=1, start_bb=start_bb, end_bb=end_bb)
            )

    peak_index = max(
        range(len(enriched)),
        key=lambda i: (_bb(_hero(enriched[i].hand).stack, enriched[i].hand), -i),
    )
    peak = enriched[peak_index].hand
    last = enriched[-1]
    return StackTrajectory(
        levels=levels,
        start_bb=_bb(_hero(enriched[0].hand).stack, enriched[0].hand),
        final_bb=_bb(last.report.stacks_end[last.hand.hero_label], last.hand),
        peak_level=peak.level,
        peak_bb=_bb(_hero(peak).stack, peak),
        peak_hand_no=peak.hand_no,
        hands_after_peak=len(enriched) - peak_index - 1,
    )


def _all_in_events(enriched: Sequence[EnrichedHand]) -> list[AllInEvent]:
    """Олл-ины героя, крупнейший первый — по стеку, с которым он в раздачу вошёл."""
    events = [
        AllInEvent(
            hand_no=en.hand.hand_no,
            hand_index=en.hand.hand_index,
            level=en.hand.level,
            hero_class=_hero_class(en.hand),
            stack_before_bb=_bb(_hero(en.hand).stack, en.hand),
            delta_bb=_delta_bb(en),
            showdown=_showdown(en.hand),
        )
        for en in enriched
        if _went_all_in(en.hand)
    ]
    events.sort(key=lambda e: (-e.stack_before_bb, e.hand_no))
    return events


def _chip_moves(enriched: Sequence[EnrichedHand]) -> list[ChipMove]:
    """Раздачи, в которых стек уменьшился, — дороже первой.

    Выигранные раздачи в список не идут: он отвечает на вопрос «где ушли
    фишки», а не «что происходило».
    """
    moves = [
        ChipMove(
            hand_no=en.hand.hand_no,
            hand_index=en.hand.hand_index,
            level=en.hand.level,
            hero_class=_hero_class(en.hand),
            last_street=_last_street(en.hand),
            all_in=_went_all_in(en.hand),
            showdown=_showdown(en.hand),
            cost_bb=-_delta_bb(en),
        )
        for en in enriched
        if _delta_bb(en) < 0
    ]
    moves.sort(key=lambda m: (-m.cost_bb, m.hand_no))
    return moves


def _pattern_key(item: ScanItem) -> tuple[str, str, str]:
    return (item.spot.value, item.action_taken, item.best_action)


def _findings(summary: ScanSummary, past_summaries: Sequence[ScanSummary]) -> list[Finding]:
    """Повторяющиеся паттерны среди оценённых точек — с ценой, зоной и историей.

    Паттерн попадает в находки, если он либо повторился В ЭТОМ турнире, либо уже
    встречался в прошлых турнирах игрока: одна точка сама по себе — событие, а
    не паттерн, но та же развилка, сыгранная так же месяц назад, — уже он
    (`test_a_pattern_from_a_past_tournament_is_a_finding_even_once`).

    Зона находки — слабейшая из зон её точек: хоть одна «предполагая» — вся
    находка «предполагая». Правило то же, что у зоны всей руки
    (`worker.pipeline._hand_zone`): чему можно верить, определяет слабейшее
    звено, а не самое дорогое.
    """
    groups: dict[tuple[str, str, str], list[ScanItem]] = {}
    for item in summary.items:
        groups.setdefault(_pattern_key(item), []).append(item)

    seen_before: dict[tuple[str, str, str], int] = {}
    seen_tournaments: dict[tuple[str, str, str], int] = {}
    for past in past_summaries:
        keys = [_pattern_key(item) for item in past.items]
        for key in keys:
            seen_before[key] = seen_before.get(key, 0) + 1
        for key in set(keys):
            seen_tournaments[key] = seen_tournaments.get(key, 0) + 1

    findings: list[Finding] = []
    for key, items in groups.items():
        before = seen_before.get(key, 0)
        if len(items) < 2 and before == 0:
            continue
        zones = {item.zone for item in items}
        findings.append(
            Finding(
                spot=items[0].spot,
                action_taken=items[0].action_taken,
                best_action=items[0].best_action,
                zone=Zone.STRICT if zones == {Zone.STRICT} else Zone.ASSUMING,
                count=len(items),
                total_cost_bb=round(-sum(item.ev_diff_bb for item in items), _BB_PRECISION),
                hand_nos=[item.hand_no for item in items],
                seen_before=before,
                seen_before_tournaments=seen_tournaments.get(key, 0),
            )
        )
    findings.sort(key=lambda f: (-f.total_cost_bb, f.hand_nos[0]))
    return findings


def _ev_split(
    enriched: Sequence[EnrichedHand], summary: ScanSummary, moves: Sequence[ChipMove]
) -> EvSplit:
    """Разложить потерянные фишки по трём столбцам и поставить рядом цену расхождений.

    Раздача попадает ровно в один фишечный столбец, поэтому три столбца в сумме
    дают все потерянные фишки: сперва раздачи, где найдено расхождение, затем из
    остальных — проигранные олл-ины (дисперсия), затем всё прочее.
    """
    gap_hands = {item.hand_no for item in summary.items}
    all_in_hands = {en.hand.hand_no for en in enriched if _went_all_in(en.hand)}

    in_gap = 0.0
    in_lost_all_ins = 0.0
    elsewhere = 0.0
    for move in moves:
        if move.hand_no in gap_hands:
            in_gap += move.cost_bb
        elif move.hand_no in all_in_hands:
            in_lost_all_ins += move.cost_bb
        else:
            elsewhere += move.cost_bb

    return EvSplit(
        judged_loss_bb=round(abs(summary.total_loss_bb), _BB_PRECISION),
        points_judged=summary.points_judged,
        points_total=summary.points_total,
        chips_in_gap_hands_bb=round(in_gap, _BB_PRECISION),
        chips_in_lost_allins_bb=round(in_lost_all_ins, _BB_PRECISION),
        chips_elsewhere_bb=round(elsewhere, _BB_PRECISION),
    )


def tournament_report(
    enriched: Sequence[EnrichedHand],
    summary: ScanSummary,
    *,
    player_tournaments: Sequence[Sequence[CanonicalHand]] = (),
    past_summaries: Sequence[ScanSummary] = (),
) -> TournamentReport:
    """Отчёт по одному турниру из его рук и сводки его скана.

    `player_tournaments` — руки ВСЕХ турниров этого игрока, по списку на турнир
    (включая этот). Среднее считается по ним всем сразу, сложением счётчиков; при
    единственном турнире среднее совпало бы с самим турниром, и `baseline`
    остаётся пуст. `past_summaries` — сводки его ПРОШЛЫХ турниров, источник
    «этот паттерн уже был». Оба аргумента необязательны: без истории отчёт
    беднее ровно на сравнение, а не сломан.

    Сводка обязана быть сводкой ЭТИХ раздач — иначе покрытие и цена расхождений
    относились бы к другому файлу, и отчёт собрался бы из двух источников молча
    (`test_report_refuses_a_summary_of_another_tournament`). Проверяется тем
    единственным, что их связывает, — числом раздач.
    """
    if not enriched:
        raise ValueError("отчёт по турниру без раздач не строится")
    if summary.hands_total != len(enriched):
        raise ValueError(
            f"сводка скана считает {summary.hands_total} раздач, а рук передано "
            f"{len(enriched)} — это сводка другого турнира"
        )

    hands = [en.hand for en in enriched]
    moves = _chip_moves(enriched)
    baseline = (
        player_stats([h for tournament in player_tournaments for h in tournament])
        if len(player_tournaments) > 1
        else None
    )
    return TournamentReport(
        hands_total=len(enriched),
        hands_failed=summary.hands_failed,
        levels_played=len({hand.level for hand in hands}),
        first_level=hands[0].level,
        last_level=hands[-1].level,
        duration_minutes=int(
            (hands[-1].timestamp - hands[0].timestamp).total_seconds() // 60
        ),
        stats=player_stats(hands),
        baseline=baseline,
        baseline_tournaments=len(player_tournaments),
        trajectory=_trajectory(enriched),
        all_ins=_all_in_events(enriched),
        chip_moves=moves,
        findings=_findings(summary, past_summaries),
        ev=_ev_split(enriched, summary, moves),
    )
