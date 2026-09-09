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

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast, overload

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from harness.analysis.frequency import (
    compare_to_reference,
    compare_to_threshold,
    defense_from_bet,
    threshold_from_bet,
)
from harness.analysis.player_stats import (
    measurement_of,
    player_stats_of_seats,
    seat_position,
)
from harness.contracts import (
    POSITIONS,
    CalcName,
    CalcParams,
    CalcResult,
    CanonicalHand,
    CoverageParams,
    CoverageResult,
    DefenseParams,
    DefenseResult,
    FrequencyResult,
    FrequencyStat,
    HeroFrequencyParams,
    LeaksParams,
    LeaksResult,
    Measurement,
    OpponentFrequencyParams,
    PointFilter,
    Subject,
    ThresholdParams,
    ThresholdResult,
    Window,
)
from harness.memory.repos import HandsRepo, LeaksRepo, OpponentsRepo

__all__ = ["REGISTRY", "run", "seats_of"]

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


async def run(db: AsyncSession, player_id: int, params: CalcParams) -> CalcResult:
    """Выполнить названный расчёт над данными этого игрока.

    Перегрузки выше — то же соответствие «имя → форма результата», что в
    `REGISTRY`, на языке типов: вызывающий, попросивший покрытие, получает
    `CoverageResult`, а не объединение всех шести.

    Расчёт выбирается полем `params.calc`, а не типом аргумента: тот же ключ
    подписывает результат, поэтому ответ на не тот вопрос виден по подписи
    (`test_every_result_names_the_calculation_that_produced_it`).

    Область — всегда `player_id`: каждая выборка внутри ограничена его сессиями,
    и чужого расчёт не видит.
    """
    _, runner = REGISTRY[params.calc]
    return await runner(db, player_id, params)
