import pytest

from harness.analysis.tools.multiway import (
    DidNotConverge,
    MultiwaySolution,
    Seat,
    unopened_shove_equilibrium,
)
from harness.analysis.tools.pushfold import (
    CallerModel,
    default_call_prob,
    nash_hu,
    nash_hu_regret_bb,
    shove_ev_bb,
)
from harness.contracts import all_classes

# Шесть пар «глубина, мёртвые деньги», на которых проверяется вырождение в
# `nash_hu`: две глубины фикстуры с её анте, игра без анте, мелкий стек, глубокий
# стек и 2bb (где колл раскрывается в «любые две карты»).
_ANCHOR_POINTS = [(12.0, 1.2), (11.75, 1.2), (10.0, 0.0), (5.0, 0.9), (20.0, 1.2), (2.0, 0.0)]


def _heads_up(eff_bb: float, dead_bb: float, hero_posted_bb: float = 0.5) -> MultiwaySolution:
    """Та же игра, что решает `nash_hu`: герой на SB, единственный игрок позади — BB.

    Посты 0.5 и 1.0, банк 1.5 + мёртвые деньги; вклад обоих равен `eff_bb`.
    `hero_posted_bb` вынесен в параметр только ради теста-фальсификации ниже.
    """
    return unopened_shove_equilibrium(
        Seat(posted_bb=hero_posted_bb, behind_bb=eff_bb - hero_posted_bb),
        [Seat(posted_bb=1.0, behind_bb=eff_bb - 1.0)],
        1.5 + dead_bb,
    )


def _max_weight_gap(left, right) -> float:
    return max(abs(left.weight(cls) - right.weight(cls)) for cls in all_classes())


# --- Опора: вырождение в `nash_hu` при одном игроке позади -----------------------


@pytest.mark.slow  # шесть равновесий fictitious play плюс шесть хедз-ап
@pytest.mark.parametrize("eff_bb,dead_bb", _ANCHOR_POINTS)
def test_one_player_behind_reproduces_nash_hu(eff_bb, dead_bb):
    """Самая сильная опора решателя: при N = 1 он обязан повторить `nash_hu` точно.

    `nash_hu` — единственное место в системе, где ответ подтверждён независимо
    (якорные тесты против опубликованных чартов). Денежные константы этой игры
    при одном игроке позади совпадают с константами хедз-ап игры тождественно,
    поэтому допуск здесь ноль по весу КАЖДОГО из 169 классов, а не «близко»:
    любое расхождение означает, что решается другая игра.
    """
    solution = _heads_up(eff_bb, dead_bb)
    reference_push, reference_call = nash_hu(eff_bb, dead_extra_bb=dead_bb)

    assert _max_weight_gap(solution.push, reference_push) == 0.0
    assert _max_weight_gap(solution.calls[0], reference_call) == 0.0


@pytest.mark.slow
def test_the_anchor_notices_a_wrong_seat_price():
    """Фальсификация опоры: сдвинутая цена места обязана её сломать.

    Если бы опора проходила при любых денежных константах, она не проверяла бы
    ничего. Здесь герою приписан пост 0.4 вместо 0.5 при том же вкладе и том же
    банке — то есть испорчена ровно та константа, из которой считаются
    `contested` и `hero_risk`, — и совпадение с `nash_hu` пропадает.
    """
    spoiled = _heads_up(10.0, 0.0, hero_posted_bb=0.4)
    reference_push, reference_call = nash_hu(10.0)

    assert _max_weight_gap(spoiled.push, reference_push) > 0.0
    assert _max_weight_gap(spoiled.calls[0], reference_call) > 0.0


# --- Референсный спот TM6292955427 ----------------------------------------------

# 8-макс, анте 1200 с каждого из восьми мест, блайнды 4000/8000, UTG спасовал.
# Герой на UTG+1: посты 0.15bb, за спиной 12.1025bb. Позади шестеро.
_REFERENCE_POT_BB = (8 * 1200.0 + 4000.0 + 8000.0) / 8000.0
_REFERENCE_STACKS = {
    "Hero": 98_020.0,
    "LJ": 94_639.0,
    "HJ": 106_210.0,
    "CO": 182_918.0,
    "BTN": 120_540.0,
    "SB": 338_739.0,
    "BB": 112_420.0,
}
_REFERENCE_BEHIND = ("LJ", "HJ", "CO", "BTN", "SB", "BB")


def _reference_seat(name: str) -> Seat:
    posted = 1200.0 + {"SB": 4000.0, "BB": 8000.0}.get(name, 0.0)
    return Seat(posted_bb=posted / 8000.0, behind_bb=(_REFERENCE_STACKS[name] - posted) / 8000.0)


@pytest.fixture(scope="module")
def reference_solution() -> MultiwaySolution:
    return unopened_shove_equilibrium(
        _reference_seat("Hero"),
        [_reference_seat(name) for name in _REFERENCE_BEHIND],
        _REFERENCE_POT_BB,
    )


@pytest.mark.slow
def test_reference_spot_call_widths(reference_solution):
    """Числа референсного спота закреплены: шов героя и шесть колл-диапазонов.

    Порядок мест — порядок хода, и он же порядок `calls`. Ширины различаются по
    местам потому, что различаются посты: у SB в банке 0.65bb, у BB 1.15bb, у
    остальных только анте, и колл им обходится дороже.
    """
    widths = [round(rng.fraction_of_hands() * 100, 2) for rng in reference_solution.calls]
    assert widths == [11.50, 10.54, 10.54, 10.54, 12.30, 13.86]
    assert round(reference_solution.push.fraction_of_hands() * 100, 2) == 18.82


@pytest.mark.slow
def test_reference_spot_is_less_exploitable_than_the_heads_up_equilibrium(reference_solution):
    """Эксплуатируемость профиля названа числом, а не заявлена словом.

    Мера та же, которой уже меряет `nash_hu_regret_bb`: максимальный по классам
    выигрыш от перехода на лучшее чистое действие, в bb на сдачу. Сравнение с
    хедз-ап равновесием той же глубины показывает, что порог остановки здесь
    достигается не слабее принятого в проекте.
    """
    assert reference_solution.hand_regret_bb <= nash_hu_regret_bb(12.0, dead_extra_bb=1.2)
    assert reference_solution.hand_regret_bb == pytest.approx(0.004210, abs=5e-6)


@pytest.mark.slow
def test_reference_spot_table_model(reference_solution):
    """Вероятность общего паса и ожидаемое число коллеров — по тем же `CallerModel`.

    Считается тем же `default_call_prob`, которым считает саму EV `shove_ev_bb`:
    новых определений вероятности колла здесь не вводится.
    """
    seats = [_reference_seat(name) for name in _REFERENCE_BEHIND]
    callers = [
        CallerModel(call_range=rng, behind_bb=seat.behind_bb, posted_bb=seat.posted_bb)
        for seat, rng in zip(seats, reference_solution.calls, strict=True)
    ]
    probs = [default_call_prob(caller, "A5s") for caller in callers]

    p_all_fold = 1.0
    for probability in probs:
        p_all_fold *= 1.0 - probability
    assert round(p_all_fold * 100, 2) == 51.44
    assert round(sum(probs), 3) == 0.629


# --- Монотонность по числу игроков позади ---------------------------------------


@pytest.mark.slow
def test_shove_range_narrows_with_every_player_behind():
    """Чем больше мест позади, тем уже равновесный шов — при том же банке и стеке.

    Опора слабая по силе (грубо неверную модель она отвергает, две правдоподобные
    не различает) и дешёвая: считается тем же решателем на том же входе, в
    котором меняется одно число.
    """
    hero = Seat(posted_bb=0.15, behind_bb=12.1025)
    tail = [Seat(posted_bb=0.15, behind_bb=12.1025)] * 4 + [
        Seat(posted_bb=0.65, behind_bb=11.6025),
        Seat(posted_bb=1.15, behind_bb=11.1025),
    ]

    widths = [
        unopened_shove_equilibrium(hero, tail[-count:], 2.70).push.fraction_of_hands()
        for count in (1, 2, 3, 4, 5, 6)
    ]
    assert widths == sorted(widths, reverse=True)
    assert len(set(widths)) == len(widths)


# --- Отказ вместо правдоподобного ответа ----------------------------------------


def test_refuses_instead_of_returning_an_unconverged_average(monkeypatch):
    """Молчаливый возврат последнего среднего опаснее падения: его примут за равновесие."""
    from harness.analysis.tools import multiway

    monkeypatch.setattr(multiway, "_FP_MAX_ITERATIONS", 5)
    multiway._solve_cached.cache_clear()
    try:
        with pytest.raises(DidNotConverge, match="не сошёлся"):
            unopened_shove_equilibrium(
                Seat(posted_bb=0.15, behind_bb=12.0),
                [Seat(posted_bb=0.15, behind_bb=12.0)] * 3,
                2.70,
            )
    finally:
        multiway._solve_cached.cache_clear()


@pytest.mark.parametrize(
    "hero,behind,pot,message",
    [
        (Seat(0.15, 10.0), [], 2.7, "нет ни одного игрока"),
        (Seat(0.15, 10.0), [Seat(0.15, 10.0)] * 8, 2.7, "не больше 7"),
        (Seat(0.15, 0.0), [Seat(0.15, 10.0)], 2.7, "нечем шовить"),
        (Seat(-0.15, 10.0), [Seat(0.15, 10.0)], 2.7, "не может быть отрицательным"),
        (Seat(0.15, 10.0), [Seat(0.15, 0.0)], 2.7, "не осталось фишек"),
        (Seat(0.15, 10.0), [Seat(0.15, 10.0)], 0.0, "должен быть положительным"),
    ],
)
def test_rejects_a_table_it_cannot_solve(hero, behind, pot, message):
    with pytest.raises(ValueError, match=message):
        unopened_shove_equilibrium(hero, behind, pot)


def test_same_table_gives_the_same_solution():
    """Решатель детерминирован, и повторный вызов обязан дать тот же объект-значение."""
    hero = Seat(posted_bb=0.5, behind_bb=4.5)
    behind = [Seat(posted_bb=1.0, behind_bb=4.0)]
    assert unopened_shove_equilibrium(hero, behind, 1.5) == unopened_shove_equilibrium(
        hero, behind, 1.5
    )


def test_the_call_range_answers_the_shove_range_of_the_same_solution():
    """Пара «шов / колл» неразрывна: колл — наилучший ответ именно на этот шов.

    Проверяется тем, что переданный отдельно колл-диапазон закрывающего места
    вместе с диапазоном шова той же пары даёт положительную цену шова только на
    руках самого диапазона шова: иначе пара была бы взята из разных решений.
    """
    solution = _heads_up(10.0, 0.0)
    caller = CallerModel(call_range=solution.calls[0], behind_bb=9.0, posted_bb=1.0)
    inside = shove_ev_bb("AA", 9.5, 1.5, [caller], hero_posted_bb=0.5)
    outside = shove_ev_bb("32o", 9.5, 1.5, [caller], hero_posted_bb=0.5)

    assert solution.push.weight("AA") == 1.0 and inside > 0.0
    assert solution.push.weight("32o") == 0.0 and outside < 0.0
