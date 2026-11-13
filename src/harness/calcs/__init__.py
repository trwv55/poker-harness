"""Словарь параметризованных расчётов: выборка из базы плюс счёт, склеенные здесь.

**Почему отдельный пакет.** Расчёту нужны и база (точки решения, руки, сшивки
оппонентов), и арифметика (`analysis`), а конвейерные пакеты про базу не знают
(CLAUDE.md, правило зависимостей). Положить склейку в `memory` нельзя по
причине, уже закреплённой тестом: образ бота импортирует `memory`, и импорт
расчётного стека оттуда затянул бы в процесс бота `pokerkit` и `eval7`
(`test_bot_image_does_not_import_calculation_stack`). Поэтому склейка живёт
здесь — пакет знает про инфраструктуру, как `bot`, `worker` и `platform`, — а
то, что бот его не импортирует, закреплено
`test_the_bot_does_not_load_the_dictionary`.

**Что где считается.** Выборка — `memory` (обычные условия по колонкам
`decision_points` и `hands`), арифметика — `analysis` (`player_stats`,
`frequency`), формы — `contracts.calcs`. В этом модуле нет ни одной формулы:
он выбирает расчёт по имени, зовёт выборку, передаёт её счёту и подписывает
результат именем расчёта.

**Набор закрыт.** `REGISTRY` перечисляет ровно `CalcName`, у каждого имени один
тип параметров и один исполнитель
(`test_the_dictionary_covers_every_name_exactly_once`). Маршрутизация модели
(следующая задача) выбирает из этого набора и подставляет параметры; считает —
код.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from typing import Any, cast, overload

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from harness.analysis.charts import to_notation
from harness.analysis.frequency import (
    compare_to_reference,
    compare_to_threshold,
    defense_from_bet,
    threshold_from_bet,
)
from harness.analysis.opponent_range import (
    EquitySpot,
    RangeRefused,
    board_at,
    equity_spot,
    hero_cards,
    layout_source,
    live_combos,
    previous_streets,
    range_spot,
    start_range,
)
from harness.analysis.player_stats import (
    measurement_of,
    player_stats_of_seats,
    seat_position,
)
from harness.analysis.range_categories import layout
from harness.analysis.solver_guide import SolverFileError, hand_bet, load_library, nearest
from harness.analysis.tools.equity import equity_vs_range
from harness.analysis.tools.exact_equity import class_equities, group_equity
from harness.contracts import (
    POSITIONS,
    CalcName,
    CalcParams,
    CalcResult,
    CanonicalHand,
    CategoryGroup,
    CoverageParams,
    CoverageResult,
    DefenseParams,
    DefenseResult,
    EquityVsRangeParams,
    EquityVsRangeResult,
    FrequencyResult,
    FrequencyStat,
    HeroFrequencyParams,
    LeaksParams,
    LeaksResult,
    Measurement,
    OpponentFrequencyParams,
    OpponentRangeParams,
    OpponentRangeResult,
    PointFilter,
    SolverAction,
    SolverGuide,
    Street,
    Subject,
    ThresholdParams,
    ThresholdResult,
    Window,
)
from harness.memory.repos import (
    AnalysesRepo,
    CalcCacheRepo,
    HandsRepo,
    LeaksRepo,
    OpponentsRepo,
)

__all__ = [
    "CLASS_EQUITY_PREFIX",
    "PREFLOP_EQUITY_ITERATIONS",
    "PREFLOP_EQUITY_PREFIX",
    "PREFLOP_EQUITY_SEED",
    "REGISTRY",
    "class_equity_key",
    "run",
    "seats_of",
]

_Runner = Callable[[AsyncSession, int, Any], Awaitable[CalcResult]]


def seats_of(
    subject: Subject, position: str | None, labels: Mapping[str, str] | None = None
) -> Callable[[CanonicalHand], list[str]]:
    """Какие места считать в каждой раздаче — вход `player_stats_of_seats`.

    Три подлежащих различаются только этим выбором, а не формулой частоты:

    * герой — его место в раздаче;
    * оппонент — место, которое сшивка (`opponent_links`) закрепила за ним в
      ЭТОМ турнире; в турнире без сшивки не считается ни одно место, потому что
      про такой турнир не сказано, кто в нём он;
    * поле — все места, кроме героя, сложенные в одно число
      (`test_the_field_counts_every_seat_but_the_hero`).

    Позиция, если названа, отсекает места, сидевшие в этой раздаче не там
    (`analysis.player_stats.seat_position`).
    """

    def pick(hand: CanonicalHand) -> list[str]:
        if subject is Subject.HERO:
            chosen = [hand.hero_label]
        elif subject is Subject.OPPONENT:
            label = (labels or {}).get(hand.tournament_id)
            chosen = [] if label is None else [label]
        else:
            chosen = [p.label for p in hand.players if p.label != hand.hero_label]
        if position is None:
            return chosen
        return [label for label in chosen if seat_position(hand, label) == position]

    return pick


def _check_position(position: str | None) -> None:
    """Незнакомая позиция — отказ, а не пустая выборка.

    Фильтр по позиции, которой не бывает, дал бы «0 из 0»: ответ, неотличимый от
    «такого не случалось», хотя случилась опечатка
    (`test_an_unknown_position_is_refused_instead_of_answering_zero_of_zero`).
    """
    if position is not None and position not in POSITIONS:
        raise ValueError(f"неизвестная позиция {position!r}; известны: {sorted(POSITIONS)}")


async def _measure(
    db: AsyncSession,
    player_id: int,
    *,
    subject: Subject,
    stat: FrequencyStat,
    opponent_id: int | None,
    position: str | None,
    window: Window,
) -> Measurement:
    """Одна измеренная частота: выборка рук окна, выбор мест, счётчики, пара чисел."""
    labels: Mapping[str, str] | None = None
    if subject is Subject.OPPONENT:
        labels = await OpponentsRepo(db).links(cast(int, opponent_id), player_id)
    hands = await HandsRepo(db).player_canonical(player_id, window)
    stats = player_stats_of_seats(hands, seats_of(subject, position, labels))
    return measurement_of(stats, stat)


def _subject_of(opponent_id: int | None) -> Subject:
    """Оппонент, если названа сшивка; иначе поле — источник эталона по умолчанию."""
    return Subject.OPPONENT if opponent_id is not None else Subject.FIELD


async def _hero_frequency(
    db: AsyncSession, player_id: int, params: HeroFrequencyParams
) -> FrequencyResult:
    _check_position(params.position)
    return FrequencyResult(
        calc=params.calc,
        subject=Subject.HERO,
        stat=params.stat,
        position=params.position,
        window=params.window,
        measurement=await _measure(
            db,
            player_id,
            subject=Subject.HERO,
            stat=params.stat,
            opponent_id=None,
            position=params.position,
            window=params.window,
        ),
    )


async def _opponent_frequency(
    db: AsyncSession, player_id: int, params: OpponentFrequencyParams
) -> FrequencyResult:
    _check_position(params.position)
    subject = _subject_of(params.opponent_id)
    return FrequencyResult(
        calc=params.calc,
        subject=subject,
        stat=params.stat,
        opponent_id=params.opponent_id,
        position=params.position,
        window=params.window,
        measurement=await _measure(
            db,
            player_id,
            subject=subject,
            stat=params.stat,
            opponent_id=params.opponent_id,
            position=params.position,
            window=params.window,
        ),
    )


async def _coverage(
    db: AsyncSession, player_id: int, params: CoverageParams
) -> CoverageResult:
    _check_position(params.filter.position)
    return await LeaksRepo(db).coverage_and_cost(player_id, params.filter)


async def _leaks(db: AsyncSession, player_id: int, params: LeaksParams) -> LeaksResult:
    """Лики по типам с ценой плюс знаменатель, из которого они посчитаны.

    Фильтр здесь только окно, без улицы и спота: тип лика сам содержит спот
    (`LeakRule`), и второе условие по той же колонке означало бы либо повтор,
    либо пустой ответ.
    """
    filters = PointFilter(window=params.window)
    repo = LeaksRepo(db)
    judged, total = await repo.coverage(player_id, filters)
    return LeaksResult(
        calc=params.calc,
        window=params.window,
        judged=Measurement(numerator=judged, denominator=total),
        leaks=await repo.by_type(player_id, filters),
    )


async def _defense(
    db: AsyncSession, player_id: int, params: DefenseParams
) -> DefenseResult:
    """Требуемая частота защиты — единственный расчёт набора, которому данные не нужны.

    База и игрок в подписи есть, потому что подпись одна на весь реестр; ни то,
    ни другое здесь не читается
    (`test_the_defence_calculation_touches_no_data`).
    """
    return defense_from_bet(params.pot_before, params.bet)


async def _threshold(
    db: AsyncSession, player_id: int, params: ThresholdParams
) -> ThresholdResult:
    """Частота против порога: вердикт по интервалу либо недостающая выборка.

    Источников порога два, и математика у них разная. Арифметика от размера
    ставки — точка, у неё своей неопределённости нет. Частота поля — измерение
    со своим знаменателем, и сравнение идёт интервалом разности
    (`analysis.frequency.compare_to_reference`), иначе измеренный порог выдавался
    бы за точно известное число.
    """
    _check_position(params.position)
    measurement = await _measure(
        db,
        player_id,
        subject=params.subject,
        stat=params.stat,
        opponent_id=params.opponent_id,
        position=params.position,
        window=params.window,
    )
    reference: Measurement | None = None
    if params.threshold.source == "bet_size":
        threshold = threshold_from_bet(
            params.threshold.pot_before, params.threshold.bet, params.threshold.side
        )
        outcome, needed = compare_to_threshold(measurement, threshold)
    else:
        reference = await _measure(
            db,
            player_id,
            subject=Subject.FIELD,
            stat=params.stat,
            opponent_id=None,
            position=params.position,
            window=params.window,
        )
        share = reference.share
        if share is None:
            raise ValueError("поле не наблюдалось ни разу: порога из него не выходит")
        threshold = share
        outcome, needed = compare_to_reference(measurement, reference)
    return ThresholdResult(
        calc=params.calc,
        subject=params.subject,
        stat=params.stat,
        opponent_id=params.opponent_id,
        position=params.position,
        window=params.window,
        measurement=measurement,
        threshold=threshold,
        threshold_source=params.threshold.source,
        reference=reference,
        outcome=outcome,
        observations_needed=needed,
    )


# Пространство ключей `calc_cache` эквити героя против класса (спека 2026-10-24, §14):
# метод и версия в префиксе — число другого метода (Монте-Карло `eqr:`) не отдаётся.
CLASS_EQUITY_PREFIX = "eqc:v1:exact:"

# До флопа категорий нет, и эквити против суженного — одно число (§14.11). Точный
# перебор там — 1.7 млн бордов на комбо, поэтому Монте-Карло, как до §14: потолок
# итераций — по замеру на стенде (план 2026-10-14, Task 5), сид — умолчание
# `equity_vs_range`; оба в префиксе ключа — смена любого не отдаёт старое число.
PREFLOP_EQUITY_ITERATIONS = 50_000
PREFLOP_EQUITY_SEED = 42
PREFLOP_EQUITY_PREFIX = f"eqr:v1:{PREFLOP_EQUITY_ITERATIONS}:{PREFLOP_EQUITY_SEED}:"


def class_equity_key(hero: Sequence[str], board: Sequence[str], hand_class: str) -> str:
    """Ключ строки `calc_cache` эквити класса: карты героя, борд, класс."""
    return f"{''.join(hero)}|{''.join(board)}|{hand_class}"


async def _class_equities(
    db: AsyncSession, hero: Sequence[str], board: Sequence[str], classes: Collection[str]
) -> dict[str, float]:
    """Эквити героя против каждого из `classes` (точный перебор, `class_equities`): все
    есть в `calc_cache` — без пересчёта
    (`test_a_cached_class_equity_is_returned_without_counting_again`); иначе перебор
    в потоке, и в кэш уходят все классы с живыми комбо на этом борде — любое другое
    сужение той же руки на той же улице собирается из них."""
    cache = CalcCacheRepo(db)
    keys = {cls: class_equity_key(hero, board, cls) for cls in classes}
    found = await cache.get_many(CLASS_EQUITY_PREFIX, list(keys.values()))
    if all(key in found for key in keys.values()):
        return {cls: found[key] for cls, key in keys.items()}
    counted = await asyncio.to_thread(class_equities, list(hero), list(board))
    await cache.upsert_many(CLASS_EQUITY_PREFIX, {
        class_equity_key(hero, board, cls): value for cls, value in counted.items()
    })
    return {cls: counted[cls] for cls in classes}


async def _with_equity(
    db: AsyncSession, hero: Sequence[str], board: Sequence[str], groups: Sequence[CategoryGroup]
) -> list[CategoryGroup]:
    """Раскладка с эквити против каждой категории — средним эквити её классов по живым
    комбо с долей (`group_equity`)."""
    by_class = await _class_equities(
        db, hero, board, [item.hand_class for group in groups for item in group.classes]
    )
    return [
        group.model_copy(update={"equity": group_equity(group, by_class)}) for group in groups
    ]


async def _analysed_hand(db: AsyncSession, player_id: int, analysis_id: int) -> CanonicalHand:
    enriched = await AnalysesRepo(db).enriched_hand(analysis_id, player_id)
    if enriched is None:
        raise ValueError("разбора этой руки нет — диапазон соперника считать не по чему")
    return enriched.hand


def _solver(
    hand: CanonicalHand, pot_before_flop: int, label: str, depth_bb: float
) -> tuple[SolverGuide | None, str | None]:
    """Ориентир флопа из библиотеки выгрузок: ставка руки, ближайшая выгрузка, её действия
    по категориям на борде выгрузки (`test_the_flop_output_carries_the_nearest_solver_guide`).
    Нечем сравнить — `(None, None)`; библиотека не прочиталась — причина вторым."""
    bet = hand_bet(hand, label, pot_before_flop, depth_bb)
    if bet is None:
        return None, None
    try:
        library = load_library()
    except SolverFileError as exc:
        return None, f"библиотека ориентиров не прочиталась ({exc})"
    dump = nearest(library, bet.key, bet.board, bet.bet_share)
    if dump is None:
        return None, None
    actions = [
        SolverAction(action=action, layout=layout(rng, dump.board, dump.board))
        for action, rng in dump.actions.items()
    ]
    return SolverGuide(
        board=list(dump.board), bet_share=dump.bet_share, hand_board=list(bet.board),
        hand_bet_share=bet.bet_share, actions=actions,
    ), None


async def _opponent_range(
    db: AsyncSession, player_id: int, params: OpponentRangeParams
) -> OpponentRangeResult:
    """Стартовый диапазон по чарту в руке разбора; отказы ядра — `ValueError` словами.
    После флопа — и раскладка по категориям (спека 2026-10-24, §4) с эквити руки героя
    против каждой (§14): на тёрне и ривере — исходного сужения этой руки (прошлой улицы,
    после чек-чек тёрна на ривере — и флопа); без него раскладки нет, а в
    `layout_refusal` — «сначала сузьте …» (§14.11)
    (`test_the_turn_layout_is_of_the_flop_narrowing_and_without_it_is_refused`,
    `test_every_category_of_the_range_layout_carries_its_own_equity`)."""
    _check_position(params.opponent_position)
    enriched = await AnalysesRepo(db).enriched_hand(params.analysis_id, player_id)
    if enriched is None:
        raise ValueError("разбора этой руки нет — диапазон соперника считать не по чему")
    hand = enriched.hand
    start = start_range(range_spot(hand, params.opponent_position))
    board = board_at(hand, params.street)
    hero = sorted(hero_cards(hand))
    dead = [*hero, *board]
    layout_refusal: str | None = None
    try:
        layout_of, found = await asyncio.to_thread(
            layout_source, start.range, params.street, params.prior, board, dead,
            previous_streets(hand, params.street), params.reset,
        )
    except RangeRefused as exc:
        layout_of, found, layout_refusal = None, None, str(exc)
    groups = list(found or [])
    if groups and len(hero) == 2:
        groups = await _with_equity(db, hero, board, groups)
    solver: SolverGuide | None = None
    solver_error: str | None = None
    if params.street is Street.FLOP:
        solver, solver_error = await asyncio.to_thread(
            _solver, hand, enriched.report.pot_by_street.get(Street.PREFLOP, 0),
            start.spot.label, start.chart_depth_bb,
        )
    return OpponentRangeResult(
        calc=params.calc,
        opponent_position=params.opponent_position,
        street=params.street,
        line=start.spot.line,
        opener_position=start.spot.opener_position,
        chart=start.chart,
        chart_seats=start.chart_seats,
        table_seats=start.spot.seats,
        chart_position=start.chart_position,
        vs_chart_position=start.vs_chart_position,
        chart_depth_bb=start.chart_depth_bb,
        range_text=to_notation(start.range),
        hands_share=start.range.fraction_of_hands(),
        live_combos=live_combos(start.range, dead),
        absent=list(start.absent),
        layout_of=layout_of,
        layout=groups,
        layout_refusal=layout_refusal,
        solver=solver,
        solver_error=solver_error,
    )


async def _preflop_equity(db: AsyncSession, spot: EquitySpot) -> float:
    """Одно эквити против суженного до флопа — Монте-Карло с фиксированным сидом и
    `calc_cache` (`test_the_preflop_equity_is_one_number_deterministic_and_cached`)."""
    cache = CalcCacheRepo(db)
    key = f"{''.join(spot.hero)}|{''.join(spot.board)}|{to_notation(spot.narrowed)}"
    equity = await cache.get(PREFLOP_EQUITY_PREFIX, key)
    if equity is None:
        equity = await asyncio.to_thread(
            equity_vs_range, spot.hero, spot.narrowed, spot.board,
            iterations=PREFLOP_EQUITY_ITERATIONS, seed=PREFLOP_EQUITY_SEED,
        )
        await cache.upsert_many(PREFLOP_EQUITY_PREFIX, {key: equity})
    return equity


async def _equity_vs_range(
    db: AsyncSession, player_id: int, params: EquityVsRangeParams
) -> EquityVsRangeResult:
    """Сужение и эквити героя против каждой категории суженного — точным перебором,
    без сида, с `calc_cache` по классам (спека 2026-10-24, §14); одного общего эквити
    после флопа нет (`test_every_category_of_the_narrowed_range_carries_its_own_equity`).
    До флопа категорий нет — одно эквити, Монте-Карло (`_preflop_equity`, §14.11).
    Счёт — в потоке, чтобы не держать цикл событий воркера."""
    _check_position(params.opponent_position)
    hand = await _analysed_hand(db, player_id, params.analysis_id)
    spot = await asyncio.to_thread(
        equity_spot, hand, params.opponent_position, params.street, params.hands,
        added=params.added, prior=params.prior, named=params.player_classes,
        kept=params.kept_classes, reason=params.added_reason, kept_agent=params.kept_agent,
        reset=params.reset,
    )
    preflop = params.street is Street.PREFLOP
    groups = [] if preflop else await _with_equity(db, spot.hero, spot.board, spot.narrowed_layout)
    return EquityVsRangeResult(
        calc=params.calc,
        opponent_position=params.opponent_position,
        street=params.street,
        range_text=to_notation(spot.narrowed),
        live_combos=spot.live_combos,
        added=list(spot.added),
        agent_added=dict(spot.agent_added or {}),
        categories_of=spot.categories_of,
        categories=list(spot.categories),
        source_combos=sum(c.combos for c in spot.categories) if spot.categories else None,
        kept_combos=sum(c.kept_combos for c in spot.categories) if spot.categories else None,
        layout=groups,
        preflop_equity=await _preflop_equity(db, spot) if preflop else None,
    )


# Реестр словаря: имя → (тип параметров, исполнитель). Одно место, где набор
# перечислен: диспетчер берёт исполнителя отсюда же, поэтому имя, забытое в
# реестре, не вызывается вовсе, а не вызывается вторым списком.
REGISTRY: dict[CalcName, tuple[type[BaseModel], _Runner]] = {
    CalcName.HERO_FREQUENCY: (HeroFrequencyParams, _hero_frequency),
    CalcName.OPPONENT_FREQUENCY: (OpponentFrequencyParams, _opponent_frequency),
    CalcName.COVERAGE: (CoverageParams, _coverage),
    CalcName.LEAKS: (LeaksParams, _leaks),
    CalcName.DEFENSE_FREQUENCY: (DefenseParams, _defense),
    CalcName.FREQUENCY_VS_THRESHOLD: (ThresholdParams, _threshold),
    CalcName.OPPONENT_RANGE: (OpponentRangeParams, _opponent_range),
    CalcName.EQUITY_VS_RANGE: (EquityVsRangeParams, _equity_vs_range),
}


@overload
async def run(
    db: AsyncSession, player_id: int, params: HeroFrequencyParams
) -> FrequencyResult: ...
@overload
async def run(
    db: AsyncSession, player_id: int, params: OpponentFrequencyParams
) -> FrequencyResult: ...
@overload
async def run(
    db: AsyncSession, player_id: int, params: CoverageParams
) -> CoverageResult: ...
@overload
async def run(db: AsyncSession, player_id: int, params: LeaksParams) -> LeaksResult: ...
@overload
async def run(db: AsyncSession, player_id: int, params: DefenseParams) -> DefenseResult: ...
@overload
async def run(
    db: AsyncSession, player_id: int, params: ThresholdParams
) -> ThresholdResult: ...
@overload
async def run(
    db: AsyncSession, player_id: int, params: OpponentRangeParams
) -> OpponentRangeResult: ...
@overload
async def run(
    db: AsyncSession, player_id: int, params: EquityVsRangeParams
) -> EquityVsRangeResult: ...


async def run(db: AsyncSession, player_id: int, params: CalcParams) -> CalcResult:
    """Выполнить названный расчёт над данными этого игрока.

    Перегрузки выше — то же соответствие «имя → форма результата», что в
    `REGISTRY`, на языке типов: вызывающий, попросивший покрытие, получает
    `CoverageResult`, а не объединение всех восьми.

    Расчёт выбирается полем `params.calc`, а не типом аргумента: тот же ключ
    подписывает результат, поэтому ответ на не тот вопрос виден по подписи
    (`test_every_result_names_the_calculation_that_produced_it`).

    Область — всегда `player_id`: каждая выборка внутри ограничена его сессиями,
    и чужого расчёт не видит.
    """
    _, runner = REGISTRY[params.calc]
    return await runner(db, player_id, params)
