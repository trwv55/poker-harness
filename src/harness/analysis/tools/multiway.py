"""Равновесие пуш-фолда в неоткрытый банк, когда позади героя несколько игроков.

**Игра, которая здесь решается.** Место 0 — герой, места 1..N — живые игроки
позади в порядке хода. Банк `pot_dead_bb` уже содержит анте всего стола и
блайнды. У каждого места есть `posted_bb` (уже лежит в банке) и `behind_bb` (чем
он ещё может заплатить); вклад места — сумма этих двух. Герой шовит на весь свой
`behind_bb` либо пасует; игроки позади по очереди коллируют или пасуют. **Первый
колл закрывает торговлю:** заколлировавший идёт на вскрытие с героем один на
один, остальные пасуют. Все величины в bb, все EV — относительно паса (пас = 0).

Деньги ветки «заколлировал ровно игрок j»:

    at_risk   = min(вклад героя, вклад j)
    contested = pot_dead_bb − posted_герой − posted_j + 2·at_risk
    hero_risk = at_risk − posted_герой
    resp_risk = at_risk − posted_j

Непокрытый остаток шова возвращается, поэтому цена ветки герою — `hero_risk`, а
не весь стек за спиной. Ветка «все спасовали» даёт герою `pot_dead_bb` целиком:
его собственные посты в базлайне «я пасую» уже потеряны. Это та же арифметика,
что в `shove_ev_bb`, сведённая к одной ветке вскрытия.

**Чего в этой игре нет.** Оверколлов: дерево обрывается на первом колле, и
игрок, за которым ещё есть места, отвечает так, будто перекрыть его некому.
Сайд-потов, ICM, блокеров между двумя РАЗНЫМИ оппонентами (снимаются только
карты того, чью руку считаем) — как и в `nash_hu`.

**Как решается.** Тем же fictitious play, что и `nash_hu`: лучший ответ на
среднюю стратегию оппонентов, усреднение 1/t, те же четыре критерия остановки.
Связь в системе одна — диапазон шова героя зависит от всех колл-диапазонов, а
игроки позади между собой не связаны: вероятность «до меня все спасовали»
входит в EV колла положительным множителем и знака не меняет.

При N = 1 денежные константы совпадают с константами хедз-ап игры
`_solve_nash_hu` тождественно, и решения совпадают поклассно — это закреплено
`test_one_player_behind_reproduces_nash_hu` (обе стороны, шесть пар «глубина,
мёртвые деньги», допуск по весу 0).

**Равновесие здесь не заявляется, а измеряется.** `MultiwaySolution` несёт
достигнутую эксплуатируемость и максимальный пер-хендовый регрет — в тех же bb,
которыми меряет `nash_hu_regret_bb`; вызывающая сторона решает, что с этими
числами делать. Недостижение порогов — это `DidNotConverge`, а не молча
возвращённое последнее среднее: правдоподобный результат вместо отказа здесь
опаснее падения.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# Таблица эквити 169x169, условное распределение руки оппонента и априорные
# вероятности классов — те же самые объекты, на которых стоит `nash_hu`. Своей
# копии здесь нет намеренно: два решателя обязаны считать по одной таблице,
# иначе опора N = 1 сравнивала бы разные игры.
from harness.analysis.tools.pushfold import (
    _class_priors,
    _conditional_class_probs,
    _eq169,
)
from harness.contracts import Range, all_classes

# Столько же, сколько перебирает `shove_ev_bb` (`_MAX_CALLERS`): больше семи
# игроков позади за столом 9-max не бывает.
_MAX_SEATS_BEHIND = 7

# Критерии остановки — те же четыре, что у `_solve_nash_hu`, и в тех же
# величинах. Агрегатная эксплуатируемость суммируется по N + 1 игроку, а не по
# двум, то есть при том же пороге требует от каждого меньшего регрета.
_FP_EXPLOITABILITY_BB = 1e-3
_FP_MAX_HAND_REGRET_BB = 1e-2
_FP_VALUE_TOLERANCE_BB = 1e-3
_FP_MIN_AVERAGING_STEPS = 200

# Потолок итераций на порядок выше, чем у `_solve_nash_hu` (20 000). Это не запас
# «на всякий случай»: число шагов усреднения растёт с числом игроков, и на
# реальных точках сетки оно доходит до шести значащих цифр — величина замерена и
# записана в спеке (docs/superpowers/specs/2026-09-06-multiway-call-ranges.md, §4).
_FP_MAX_ITERATIONS = 200_000

# Стартовое убеждение: в среднее не входит — на первом шаге оно целиком
# заменяется первым лучшим ответом.
_INITIAL_WEIGHT = 0.5

# Веса округляются до того же знака, что и у `nash_hu`: диапазоны из двух
# решателей сравниваются поклассно в опоре N = 1, и разная точность сделала бы
# сравнение неопределённым.
_WEIGHT_PRECISION = 6

# Решение зависит от конфигурации стеков всех мест, поэтому дискового кэша (как у
# `nash_hu`) здесь нет: на реальных фикстурах ключи не повторяются. Память нужна
# ровно для того, чтобы одна и та же точка решения, посчитанная дважды за один
# разбор, стоила одного решения.
_SOLUTION_CACHE_SIZE = 256


class DidNotConverge(RuntimeError):
    """Пороги остановки не достигнуты за `_FP_MAX_ITERATIONS` итераций."""


@dataclass(frozen=True)
class Seat:
    """Место за столом: сколько уже в банке и чем ещё может заплатить, в bb."""

    posted_bb: float
    behind_bb: float

    @property
    def total_bb(self) -> float:
        """Вклад места — тем и меряется, на сколько его можно заколлировать."""
        return self.posted_bb + self.behind_bb


@dataclass(frozen=True)
class MultiwaySolution:
    """Профиль стратегий и то, насколько он эксплуатируем.

    `calls` идёт в том же порядке, что и переданные места позади. `push` — тот
    диапазон шова, ПРОТИВ КОТОРОГО эти коллы являются наилучшим ответом; пара
    неразрывна, и брать колл-сторону в паре с другим шовом значит отвечать не на
    ту игру.
    """

    push: Range
    calls: tuple[Range, ...]
    averaging_steps: int
    exploitability_bb: float
    hand_regret_bb: float


def unopened_shove_equilibrium(
    hero: Seat, behind: Sequence[Seat], pot_dead_bb: float
) -> MultiwaySolution:
    """Решает игру из докстринга модуля и возвращает профиль вместе с его ценой.

    `behind` — живые игроки позади в порядке хода, от одного до
    `_MAX_SEATS_BEHIND`; у каждого должно остаться что ставить.

    Одинаковый вход даёт одинаковый выход (fictitious play детерминирован) и
    считается один раз: результат запоминается на `_SOLUTION_CACHE_SIZE`
    последних конфигураций.
    """
    if not behind:
        raise ValueError("позади героя нет ни одного игрока — решать нечего")
    if len(behind) > _MAX_SEATS_BEHIND:
        raise ValueError(f"игроков позади {len(behind)}, решается не больше {_MAX_SEATS_BEHIND}")
    if hero.behind_bb <= 0.0:
        raise ValueError(f"герою нечем шовить: за спиной {hero.behind_bb} bb")
    if hero.posted_bb < 0.0 or any(seat.posted_bb < 0.0 for seat in behind):
        raise ValueError("поставленное в банк не может быть отрицательным")
    if any(seat.behind_bb <= 0.0 for seat in behind):
        raise ValueError("у игрока позади не осталось фишек за спиной — коллировать ему нечем")
    if pot_dead_bb <= 0.0:
        raise ValueError(f"банк на решении должен быть положительным, получено {pot_dead_bb}")

    return _solve_cached(
        hero.posted_bb,
        hero.behind_bb,
        tuple((seat.posted_bb, seat.behind_bb) for seat in behind),
        pot_dead_bb,
    )


@lru_cache(maxsize=_SOLUTION_CACHE_SIZE)
def _solve_cached(
    hero_posted_bb: float,
    hero_behind_bb: float,
    behind: tuple[tuple[float, float], ...],
    pot_dead_bb: float,
) -> MultiwaySolution:
    classes = all_classes()
    equity = np.asarray(_eq169()[1], dtype=np.float64)
    conditional = np.asarray(_conditional_class_probs(), dtype=np.float64)
    priors = np.asarray(_class_priors(), dtype=np.float64)
    size = len(classes)
    seats = len(behind)
    hero_total = hero_posted_bb + hero_behind_bb

    contested = np.empty(seats)
    hero_risk = np.empty(seats)
    resp_risk = np.empty(seats)
    for j, (posted, seat_behind) in enumerate(behind):
        at_risk = min(hero_total, posted + seat_behind)
        contested[j] = pot_dead_bb - hero_posted_bb - posted + 2.0 * at_risk
        hero_risk[j] = at_risk - hero_posted_bb
        resp_risk[j] = at_risk - posted

    # push_term[j][h][c] — вклад «герой с h, коллер j с c» в EV шова, уже с P(c|h).
    push_term = np.stack(
        [conditional * (equity * contested[j] - hero_risk[j]) for j in range(seats)]
    )
    # call_term[j][b][h] — «колл минус пас» для игрока j с b против шова руки h.
    call_term = np.stack(
        [conditional * (equity * contested[j] - resp_risk[j]) for j in range(seats)]
    )

    avg_push = np.full(size, _INITIAL_WEIGHT)
    avg_call = np.full((seats, size), _INITIAL_WEIGHT)

    steps = 0
    previous_value: float | None = None
    exploitability = float("inf")
    hand_regret = float("inf")
    converged = False

    for _iteration in range(_FP_MAX_ITERATIONS):
        # call_prob[h][j] — вероятность, что игрок j коллирует шов руки h.
        call_prob = conditional @ avg_call.T
        # branch[h][j] — ценность ветки «заколлировал именно j», без веса
        # «до него все спасовали».
        branch = np.einsum("jhc,jc->hj", push_term, avg_call)

        survive = np.ones(size)
        shove_ev = np.zeros(size)
        for j in range(seats):
            shove_ev += survive * branch[:, j]
            survive = survive * (1.0 - call_prob[:, j])
        shove_ev += survive * pot_dead_bb

        # call_minus_fold[j][b] — цена колла игрока j рукой b против среднего шова.
        call_minus_fold = np.einsum("jbh,h->jb", call_term, avg_push)

        value = float(priors @ (avg_push * shove_ev))
        regret_hero = np.maximum(shove_ev, 0.0) - avg_push * shove_ev
        regret_behind = np.maximum(call_minus_fold, 0.0) - avg_call * call_minus_fold
        exploitability = float(priors @ regret_hero + (regret_behind @ priors).sum())
        hand_regret = float(max(regret_hero.max(), regret_behind.max()))

        if (
            steps >= _FP_MIN_AVERAGING_STEPS
            and exploitability <= _FP_EXPLOITABILITY_BB
            and hand_regret <= _FP_MAX_HAND_REGRET_BB
            and previous_value is not None
            and abs(value - previous_value) < _FP_VALUE_TOLERANCE_BB
        ):
            converged = True
            break
        previous_value = value

        best_push = (shove_ev > 0.0).astype(np.float64)
        best_call = (call_minus_fold > 0.0).astype(np.float64)
        steps += 1
        if steps == 1:
            avg_push, avg_call = best_push, best_call
        else:
            rate = 1.0 / steps
            avg_push = avg_push + (best_push - avg_push) * rate
            avg_call = avg_call + (best_call - avg_call) * rate

    if not converged:
        raise DidNotConverge(
            f"fictitious play не сошёлся за {_FP_MAX_ITERATIONS} итераций при "
            f"{seats} игроках позади (эксплуатируемость {exploitability:.6f}, "
            f"пер-хендовый регрет {hand_regret:.6f}) — возвращать последнее "
            f"среднее как равновесие нельзя"
        )

    return MultiwaySolution(
        push=_range_of(classes, avg_push),
        calls=tuple(_range_of(classes, avg_call[j]) for j in range(seats)),
        averaging_steps=steps,
        exploitability_bb=exploitability,
        hand_regret_bb=hand_regret,
    )


def _range_of(classes: Sequence[str], weights: np.ndarray) -> Range:
    return Range(
        weights={
            cls: round(float(weight), _WEIGHT_PRECISION)
            for cls, weight in zip(classes, weights, strict=True)
            if round(float(weight), _WEIGHT_PRECISION) > 0.0
        }
    )
