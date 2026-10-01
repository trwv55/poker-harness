"""Равновесие пуш-фолда в неоткрытый банк, когда позади героя несколько игроков.

**Игра, которая здесь решается.** Место 0 — герой, места 1..N — живые игроки
позади в порядке хода. Банк `pot_dead_bb` уже содержит анте всего стола и
блайнды. У каждого места есть `posted_bb` (уже лежит в банке) и `behind_bb` (чем
он ещё может заплатить); вклад места — сумма этих двух. Герой шовит на весь свой
`behind_bb` либо пасует; игроки позади по очереди коллируют или пасуют. У
каждого места позади ДВЕ стратегии: **холодный колл** (до него никто не
коллировал) и **оверколл** (до него ровно один коллер). **Второй колл закрывает
торговлю:** после двух коллеров остальные пасуют, вскрытие втроём. Все величины
в bb, все EV — относительно паса (пас = 0).

Деньги ветки «заколлировал ровно игрок j»:

    at_risk   = min(вклад героя, вклад j)
    contested = pot_dead_bb − posted_герой − posted_j + 2·at_risk
    hero_risk = at_risk − posted_герой
    resp_risk = at_risk − posted_j

Ветка «заколлировали j и k» считается слоями банка (`_three_way_money`): общий
банк на троих до наименьшего вклада, сайд-пот двух старших вкладов до среднего,
непокрытый остаток возвращается. Ветка «все спасовали» даёт герою `pot_dead_bb`
целиком: его собственные посты в базлайне «я пасую» уже потеряны. Ветки
«все спасовали» и «один коллер» — та же арифметика, что в `shove_ev_bb`; ветку
втроём `shove_ev_bb` считает без сайд-потов, здесь они есть.

**Зачем оверколлы.** Без них холодный коллер отвечает так, будто перекрыть его
некому, и места с одинаковыми постами коллируют одинаково — UTG+1 с шестью
игроками позади так же широко, как BTN с двумя. Замер против GTO-солвера
(спека 2026-10-01-pushfold-overcalls-dead-cards, §1) показал, что это половина
переоценки пар в EV шова: ранние места в GTO коллируют теснее и тяжелее по
старшим парам. Риск оверколла входит в EV холодного колла — и делает ранние
коллы теснее поздних.

**Вскрытие втроём** считается не Монте-Карло, а из двух парных эквити по модели
Брэдли–Терри с поклассной поправкой (`three_way_equity`, данные
`eq3_correction.json` из `scripts/build_eq3_correction.py`). Остаточная ошибка
на парах диапазонов вне калибровки замерена до 1.85 п.п.; тест
`test_three_way_equity_matches_monte_carlo_on_held_out_ranges` держит 2.5 п.п. —
с запасом на шум Монте-Карло. Сайд-пот разыгрывается по парному эквити против
того, с кем он делится.

**Чего в этой игре нет.** Третьего коллера. Различения, КТО именно заколлировал
первым, в информационном множестве оверколлера: он отвечает на смесь холодных
колл-диапазонов мест до него, взвешенную вероятностью «первым был именно он».
ICM. Блокеров между двумя РАЗНЫМИ оппонентами (снимаются только карты того, чью
руку считаем) — как и в `nash_hu`. Цену решения на реальной руке, где всё это
важно, считает симуляция полной раздачи (`full_deal.py`) против диапазонов
отсюда; решатель отвечает за сами диапазоны.

**Как решается.** Тем же fictitious play, что и `nash_hu`: лучший ответ на
среднюю стратегию оппонентов, усреднение 1/t, те же четыре критерия остановки.
Вероятность «до меня все спасовали» входит в EV холодного колла положительным
множителем и знака не меняет, поэтому в нём не учитывается.

При N = 1 оверколлировать некому, денежные константы совпадают с константами
хедз-ап игры `_solve_nash_hu` тождественно, и решения совпадают поклассно — это
закреплено `test_one_player_behind_reproduces_nash_hu` (обе стороны, шесть пар
«глубина, мёртвые деньги», допуск по весу 0).

**Регрет оверколлера разбавлен.** EV оверколла считается ex ante — с
множителем «герой шовит и ровно один холодный коллер до меня», который у
ранних мест порядка 0.01–0.05. Поэтому тот же порог `_FP_MAX_HAND_REGRET_BB`
допускает у оверколлера реальный регрет на порядок больше, чем у холодного
коллера. На вердикт это влияет слабо — ветки втроём весят мало, — но точность
оверколл-диапазонов этим порогом не гарантирована и отдельно не замерена.

**Равновесие здесь не заявляется, а измеряется.** `MultiwaySolution` несёт
достигнутую эксплуатируемость и максимальный пер-хендовый регрет — в тех же bb,
которыми меряет `nash_hu_regret_bb`; вызывающая сторона решает, что с этими
числами делать. Недостижение порогов — это `DidNotConverge`, а не молча
возвращённое последнее среднее: правдоподобный результат вместо отказа здесь
опаснее падения.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

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
# величинах. Агрегатная эксплуатируемость суммируется по всем стратегиям всех
# игроков, то есть при том же пороге требует от каждой меньшего регрета.
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

# Ниже этого парного эквити модель Брэдли–Терри делит на ноль; доля банка
# такой руки втроём ничтожна при любом значении из [0, 1e-9].
_MIN_PAIRWISE_EQUITY = 1e-9

# Эквити руки против пустого диапазона не определено; вес такой ветки — ноль
# (вероятность колла пустым диапазоном), и подставленное значение на результат
# не влияет.
_EQUITY_VS_EMPTY = 0.5

_EQ3_CORRECTION_PATH = Path(__file__).parent / "data" / "eq3_correction.json"


def bradley_terry_three_way(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Доля банка игрока во вскрытии втроём по его парным эквити x и y.

    Модель силы: эквити один на один — `σ_я / (σ_я + σ_он)`, отсюда
    `σ_он / σ_я = (1 − x) / x`, и доля втроём `σ_я / (σ_я + σ_a + σ_b)`.
    Сама по себе ошибается до 4.5 п.п. — решатель берёт её только вместе с
    поклассной поправкой (`three_way_equity`). Парное эквити 0 даёт 0.
    """
    x_safe = np.maximum(x, _MIN_PAIRWISE_EQUITY)
    y_safe = np.maximum(y, _MIN_PAIRWISE_EQUITY)
    return 1.0 / (1.0 + (1.0 - x_safe) / x_safe + (1.0 - y_safe) / y_safe)


@cache
def _eq3_correction() -> np.ndarray:
    """Поклассная поправка к `bradley_terry_three_way`, порядок — `all_classes()`."""
    if not _EQ3_CORRECTION_PATH.exists():
        raise FileNotFoundError(
            f"нет поправки трёхстороннего эквити {_EQ3_CORRECTION_PATH} — она генерируется "
            f"один раз командой `uv run python scripts/build_eq3_correction.py` и "
            f"коммитится как данные"
        )
    payload = json.loads(_EQ3_CORRECTION_PATH.read_text(encoding="utf-8"))
    if payload["classes"] != all_classes():
        raise ValueError("порядок классов в eq3_correction.json разошёлся с all_classes()")
    return np.asarray(payload["correction"], dtype=np.float64)


def three_way_equity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Доля банка во вскрытии втроём для каждого из 169 классов игрока.

    `x[c]`, `y[c]` — эквити класса c против каждого из двух диапазонов (с его
    блокерами). Модель Брэдли–Терри плюс поправка класса, обрезанные в [0, 1].
    """
    return np.clip(bradley_terry_three_way(x, y) + _eq3_correction(), 0.0, 1.0)


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

    `calls` и `overcalls` идут в том же порядке, что и переданные места позади:
    `calls[j]` — холодный колл места j (до него никто не коллировал),
    `overcalls[j]` — его ответ, когда коллер уже есть. У первого места позади
    оверколла не бывает, его диапазон пуст. `push` — тот диапазон шова, ПРОТИВ
    КОТОРОГО эти коллы являются наилучшим ответом; тройка неразрывна, и брать
    колл-сторону в паре с другим шовом значит отвечать не на ту игру.
    """

    push: Range
    calls: tuple[Range, ...]
    overcalls: tuple[Range, ...]
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


@dataclass(frozen=True)
class _ThreeWayMoney:
    """Деньги вскрытия втроём с точки зрения одного участника («я»).

    `main` — общий банк на троих, `side` — сайд-пот двух старших вкладов,
    `side_vs` — с кем «я» делю сайд-пот (0 — с первым из двух других, 1 — со
    вторым, None — «я» в нём не участвую), `risk` — сколько «я» докладываю сверх
    уже поставленного.
    """

    main: float
    side: float
    side_vs: int | None
    risk: float


@dataclass(frozen=True)
class _MoneyColumns:
    """`_ThreeWayMoney` всех пар столбцами формы (пары, 1) — под строки классов."""

    main: np.ndarray
    risk: np.ndarray
    side_vs_first: np.ndarray
    side_vs_second: np.ndarray

    @classmethod
    def of(cls, money: Sequence[_ThreeWayMoney]) -> _MoneyColumns:
        def column(values: Sequence[float]) -> np.ndarray:
            return np.asarray(values, dtype=np.float64).reshape(-1, 1)

        return cls(
            main=column([m.main for m in money]),
            risk=column([m.risk for m in money]),
            side_vs_first=column([m.side if m.side_vs == 0 else 0.0 for m in money]),
            side_vs_second=column([m.side if m.side_vs == 1 else 0.0 for m in money]),
        )

    def value(self, share: np.ndarray, vs_first: np.ndarray, vs_second: np.ndarray) -> np.ndarray:
        """EV ветки по доле общего банка и парным эквити против двух других."""
        return (
            share * self.main
            - self.risk
            + vs_first * self.side_vs_first
            + vs_second * self.side_vs_second
        )


def _three_way_money(
    me: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
    pot_dead_bb: float,
) -> _ThreeWayMoney:
    """Слои банка при вскрытии втроём; каждое место — пара (поставлено, вклад)."""
    (my_posted, my_total), (first_posted, first_total), (second_posted, second_total) = (
        me,
        first,
        second,
    )
    lowest, middle, _ = sorted((my_total, first_total, second_total))
    others = pot_dead_bb - my_posted - first_posted - second_posted
    side = 2.0 * (middle - lowest)
    side_vs: int | None = None
    if side > 0.0 and my_total > lowest:
        side_vs = 0 if first_total > lowest else 1
    return _ThreeWayMoney(
        main=others + 3.0 * lowest,
        side=side,
        side_vs=side_vs,
        risk=min(my_total, middle) - my_posted,
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
    weighted_equity = conditional * equity

    contested = np.empty(seats)
    hero_risk = np.empty(seats)
    resp_risk = np.empty(seats)
    for j, (posted, seat_behind) in enumerate(behind):
        at_risk = min(hero_total, posted + seat_behind)
        contested[j] = pot_dead_bb - hero_posted_bb - posted + 2.0 * at_risk
        hero_risk[j] = at_risk - hero_posted_bb
        resp_risk[j] = at_risk - posted

    # Вскрытия втроём «холодный j + оверколл k», все пары разом: индекс p
    # пробегает пары, `cold_of[p]` = j, `over_of[p]` = k. Деньги — с точки
    # зрения каждого из трёх участников. Места: (поставлено, вклад).
    # Коллер ставит не больше шова героя: сайд-пота между двумя коллерами,
    # глубже героя, не бывает — дальше шова никто не торгуется.
    hero_seat = (hero_posted_bb, hero_total)
    seat_of = [(posted, min(posted + seat_behind, hero_total)) for posted, seat_behind in behind]
    pairs = [(j, k) for j in range(seats) for k in range(j + 1, seats)]
    cold_of = np.array([j for j, _ in pairs], dtype=np.intp)
    over_of = np.array([k for _, k in pairs], dtype=np.intp)
    # Суммирование по парам в место: by_cold[j][p] = 1, если холодный в паре p — j.
    by_cold = np.zeros((seats, len(pairs)))
    by_over = np.zeros((seats, len(pairs)))
    by_cold[cold_of, np.arange(len(pairs))] = 1.0
    by_over[over_of, np.arange(len(pairs))] = 1.0
    money_hero = _MoneyColumns.of(
        [_three_way_money(hero_seat, seat_of[j], seat_of[k], pot_dead_bb) for j, k in pairs]
    )
    money_cold = _MoneyColumns.of(
        [_three_way_money(seat_of[j], hero_seat, seat_of[k], pot_dead_bb) for j, k in pairs]
    )
    money_over = _MoneyColumns.of(
        [_three_way_money(seat_of[k], hero_seat, seat_of[j], pot_dead_bb) for j, k in pairs]
    )

    # push_term[j][h][c] — вклад «герой с h, коллер j с c» в EV шова, уже с P(c|h).
    push_term = np.stack(
        [conditional * (equity * contested[j] - hero_risk[j]) for j in range(seats)]
    )
    # call_term[j][b][h] — «колл минус пас» для игрока j с b против шова руки h.
    call_term = np.stack(
        [conditional * (equity * contested[j] - resp_risk[j]) for j in range(seats)]
    )

    def versus(strategy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Вероятность, что диапазон «ответит», и эквити против него — для каждого класса."""
        mass = conditional @ strategy
        share = np.divide(
            weighted_equity @ strategy,
            mass,
            out=np.full(size, _EQUITY_VS_EMPTY),
            where=mass > 0.0,
        )
        return mass, share

    avg_push = np.full(size, _INITIAL_WEIGHT)
    avg_call = np.full((seats, size), _INITIAL_WEIGHT)
    # У первого места позади оверколла не бывает: узла нет, стратегия нулевая.
    avg_over = np.full((seats, size), _INITIAL_WEIGHT)
    avg_over[0] = 0.0

    steps = 0
    previous_value: float | None = None
    exploitability = float("inf")
    hand_regret = float("inf")
    converged = False

    for _iteration in range(_FP_MAX_ITERATIONS):
        # call_prob[h][j] — вероятность, что игрок j коллирует шов руки h.
        call_prob = conditional @ avg_call.T
        # branch[h][j] — ценность ветки «заколлировал именно j», без веса
        # «до него все спасовали» и без оверколлов позади него.
        branch = np.einsum("jhc,jc->hj", push_term, avg_call)

        # «колл минус пас» без оверколлов: игрок j с b против среднего шова.
        call_minus_fold = np.einsum("jbh,h->jb", call_term, avg_push)

        # Строки по местам: вероятность «ответит» и эквити против диапазона —
        # для каждого класса «я» (с его блокерами).
        push_mass, push_share = versus(avg_push)
        cold_mass = avg_call @ conditional.T
        cold_share = np.divide(
            avg_call @ weighted_equity.T,
            cold_mass,
            out=np.full((seats, size), _EQUITY_VS_EMPTY),
            where=cold_mass > 0.0,
        )
        over_prob = avg_over @ conditional.T
        over_share = np.divide(
            avg_over @ weighted_equity.T,
            over_prob,
            out=np.full((seats, size), _EQUITY_VS_EMPTY),
            where=over_prob > 0.0,
        )

        # quiet[p] — никто из мест между холодным и оверколлером пары p не
        # оверколлировал; tail[j] — после места j не оверколлировал никто;
        # none_before[i] — до места i никто не коллировал холодно.
        quiet = np.empty((len(pairs), size))
        tail = np.empty((seats, size))
        none_before = np.empty((seats, size))
        index = 0
        cold_run = np.ones(size)
        for j in range(seats):
            none_before[j] = cold_run
            cold_run = cold_run * (1.0 - cold_mass[j])
            run = np.ones(size)
            for k in range(j + 1, seats):
                quiet[index] = run
                run = run * (1.0 - over_prob[k])
                index += 1
            tail[j] = run

        reach = quiet * over_prob[over_of]
        hero_value = money_hero.value(
            three_way_equity(cold_share[cold_of], over_share[over_of]),
            cold_share[cold_of],
            over_share[over_of],
        )
        hero_three = by_cold @ (reach * cold_mass[cold_of] * hero_value)
        cold_value = money_cold.value(
            three_way_equity(push_share, over_share[over_of]), push_share, over_share[over_of]
        )
        # Три-вей ветки холодного коллера взвешены вероятностью «герой шовит»
        # так же, как два-вей в `call_minus_fold`.
        cold_ev = tail * call_minus_fold + push_mass * (by_cold @ (reach * cold_value))

        # over_ev[k][b] — оверколл места k с рукой b: шов героя и ровно один
        # холодный коллер i < k, взвешенный вероятностью того, что первым был i.
        over_value = money_over.value(
            three_way_equity(push_share, cold_share[cold_of]), push_share, cold_share[cold_of]
        )
        over_ev = push_mass * (
            by_over @ (none_before[cold_of] * cold_mass[cold_of] * quiet * over_value)
        )

        shove_ev = np.zeros(size)
        survive = np.ones(size)
        for j in range(seats):
            shove_ev += survive * (tail[j] * branch[:, j] + hero_three[j])
            survive = survive * (1.0 - call_prob[:, j])
        shove_ev += survive * pot_dead_bb

        value = float(priors @ (avg_push * shove_ev))
        regret_hero = np.maximum(shove_ev, 0.0) - avg_push * shove_ev
        regret_cold = np.maximum(cold_ev, 0.0) - avg_call * cold_ev
        regret_over = np.maximum(over_ev, 0.0) - avg_over * over_ev
        exploitability = float(
            priors @ regret_hero + (regret_cold @ priors).sum() + (regret_over @ priors).sum()
        )
        hand_regret = float(max(regret_hero.max(), regret_cold.max(), regret_over.max()))

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
        best_call = (cold_ev > 0.0).astype(np.float64)
        best_over = (over_ev > 0.0).astype(np.float64)
        steps += 1
        if steps == 1:
            avg_push, avg_call, avg_over = best_push, best_call, best_over
        else:
            rate = 1.0 / steps
            avg_push = avg_push + (best_push - avg_push) * rate
            avg_call = avg_call + (best_call - avg_call) * rate
            avg_over = avg_over + (best_over - avg_over) * rate

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
        overcalls=tuple(_range_of(classes, avg_over[j]) for j in range(seats)),
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
