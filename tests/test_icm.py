import pytest

from harness.analysis.tools.icm import icm_equities


def test_equal_stacks_equal_equity():
    eq = icm_equities([5000, 5000, 5000], [0.5, 0.3, 0.2])
    assert all(abs(e - 1 / 3) < 1e-9 for e in eq)

def test_two_players_known():
    eq = icm_equities([3000, 1000], [0.6, 0.4])
    # P(1st) = 0.75/0.25 -> 0.75*0.6+0.25*0.4 = 0.55
    assert abs(eq[0] - 0.55) < 1e-9 and abs(eq[1] - 0.45) < 1e-9

def test_sums_to_total():
    eq = icm_equities([7000, 2000, 1000], [0.5, 0.3, 0.2])
    assert abs(sum(eq) - 1.0) < 1e-9
    assert eq[0] > eq[1] > eq[2]
    # значения посчитаны независимым перебором всех порядков финиша
    for got, want in zip(eq, [0.435278, 0.308889, 0.255833]):
        assert abs(got - want) < 1e-5

def test_chip_lead_worth_far_less_than_chip_share():
    # суть ICM одним числом: 90% фишек стоят 47.9% призовых, а не 90%
    eq = icm_equities([9000, 500, 500], [0.5, 0.3, 0.2])
    assert abs(eq[0] - 0.479474) < 1e-5
    assert abs(eq[1] - 0.260263) < 1e-5 and abs(eq[2] - 0.260263) < 1e-5
    assert eq[0] < 0.9 - 0.4          # доля фишек 0.9, доля призовых заметно ниже

def test_matches_independent_brute_force():
    # Оракул: прямой перебор всех порядков финиша — другой алгоритм, тот же ответ.
    # Рекурсия Малмута-Харвилла должна совпадать с ним до 1e-9 на малых n.
    from itertools import permutations

    def oracle(stacks, payouts):
        n, S = len(stacks), sum(stacks)
        eq = [0.0] * n
        for order in permutations(range(n)):
            p, rem = 1.0, S
            for who in order:
                p *= stacks[who] / rem
                rem -= stacks[who]
            for place, who in enumerate(order):
                if place < len(payouts):
                    eq[who] += p * payouts[place]
        return eq

    for stacks in ([5000, 3000, 2000], [12000, 800, 600, 400], [1, 1, 98]):
        pay = [0.5, 0.3, 0.2]
        for got, want in zip(icm_equities(stacks, pay), oracle(stacks, pay)):
            assert abs(got - want) < 1e-9, (stacks, got, want)


def test_too_many_paid_places_raises():
    # len(payouts) <= 4 — жёсткий предел брифа (Step 3): для MTT передают
    # хвост релевантной призовой структуры, а не всю сетку на сотни мест.
    with pytest.raises(ValueError, match="4 платных мест"):
        icm_equities([100] * 5, [0.2, 0.2, 0.2, 0.2, 0.2])

def test_empty_stacks_raises():
    with pytest.raises(ValueError, match="пустым"):
        icm_equities([], [0.5, 0.3, 0.2])

def test_zero_total_stack_raises():
    with pytest.raises(ValueError, match="положительными"):
        icm_equities([0, 0, 0], [0.5, 0.3, 0.2])

def test_zero_stack_player_among_positive_raises():
    # Не только некорректно по смыслу (выбывший игрок), но и математически:
    # если бы это прошло, распределение мест на подмножестве из одних нулевых
    # стеков после того, как игрок с фишками уже занял более высокое место,
    # упёрлось бы в 0/0 внутри first_place_probs.
    with pytest.raises(ValueError, match="положительными"):
        icm_equities([100, 0], [0.6, 0.4])

def test_more_paid_places_than_players_raises():
    with pytest.raises(ValueError, match="больше числа игроков"):
        icm_equities([100, 200], [0.5, 0.3, 0.2])
