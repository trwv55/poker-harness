"""Симуляция полной раздачи (`tools/full_deal.py`) и якорь против GTO-солвера.

Якорь — расчёт владельца, снятый 2026-10-01 коммерческим солвером: 8-max, стеки
10bb ПОСЛЕ анте (анте 1bb целиком платит BB сверх своих 10bb, у него 9bb за
спиной после блайнда), банк до шова 2.5bb, chipEV без ICM, в дереве есть мин-рейз
(спека 2026-10-01-pushfold-overcalls-dead-cards, §2 и §9).

Мин-рейз делает несравнимыми ДИАПАЗОНЫ: солвер разыгрывает часть премиумов
(AA, KK, QQ целиком) мин-рейзом, его шов UTG — 12.6% без них, и коллы солвера
отвечают на этот шов, а не на наш. Поэтому якорем служат только EV:
EV шова UTG (наши диапазоны, сверка всей связки) и EV колла против шова UTG,
посчитанный против ДИАПАЗОНА ШОВА СОЛВЕРА (сверка расчёта колла как такового).
"""

from __future__ import annotations

import pytest

from harness.analysis.tools.full_deal import (
    Responder,
    Shover,
    _commits,
    _settle,
    call_ev_full_deal,
    shove_ev_full_deal,
)
from harness.analysis.tools.multiway import Seat, unopened_shove_equilibrium
from harness.analysis.tools.pushfold import CallerModel, shove_ev_bb
from harness.contracts import Range

# --- Деньги вскрытия --------------------------------------------------------------


def test_a_caller_never_puts_in_more_than_the_shove():
    """Двое глубоких коллеров против короткого шова ставят ровно шов — сайд-пота нет."""
    aggressor, callers = _commits(10.0, [30.0, 25.0])
    assert aggressor == 10.0 and callers == [10.0, 10.0]


def test_the_uncovered_part_of_a_shove_goes_back():
    """Шов глубже всех коллеров покрыт только самым глубоким из них."""
    aggressor, callers = _commits(20.0, [5.0, 12.0])
    assert aggressor == 12.0 and callers == [5.0, 12.0]


def test_side_pot_goes_to_the_best_hand_that_reached_it():
    """Слои: общий банк на троих до 5, сайд-пот двоих от 5 до 10.

    Мёртвые деньги (2.5) лежат в нижнем слое. Лучшая рука — у коротыша: он
    забирает общий банк (3·5 + 2.5), герою со второй рукой достаётся сайд-пот
    (2·5) — ровно его доля, а не ноль и не всё.
    """
    hero_rank, short_rank, deep_rank = 200, 300, 100
    won = _settle(hero_rank, 10.0, [(short_rank, 5.0), (deep_rank, 10.0)], 2.5)
    assert won == pytest.approx(10.0)
    won_best = _settle(400, 10.0, [(short_rank, 5.0), (deep_rank, 10.0)], 2.5)
    assert won_best == pytest.approx(3 * 5.0 + 2.5 + 2 * 5.0)


def test_a_tie_splits_the_layer():
    won = _settle(100, 10.0, [(100, 10.0)], 1.0)
    assert won == pytest.approx((2 * 10.0 + 1.0) / 2)


# --- Детерминизм и согласие с аналитикой -----------------------------------------


def _heads_up_responder(call: Range) -> Responder:
    return Responder(posted_bb=1.0, total_bb=10.0, cold=call, over=Range(weights={}))


def test_same_inputs_give_the_same_number():
    call = Range(weights={"AA": 1.0, "KK": 1.0, "AKs": 1.0})
    args = ("TT", 0.5, 10.0, [_heads_up_responder(call)], 1.5)
    assert shove_ev_full_deal(*args, iterations=2_000, seed=7) == shove_ev_full_deal(
        *args, iterations=2_000, seed=7
    )


@pytest.mark.slow  # 200 000 раздач
def test_with_one_caller_the_simulation_agrees_with_the_analytic_ev():
    """Один коллер: снимать из колоды нечего, кроме карт героя, и вскрытие хедз-ап.

    Тогда симуляция и аналитика `shove_ev_bb` считают одну и ту же игру, и
    расходиться они могут только на шум симуляции (≈ 0.02bb) и на усреднение по
    комбо класса героя против одного представителя у аналитики.
    """
    call = Range(weights={"AA": 1.0, "KK": 1.0, "QQ": 1.0, "JJ": 1.0, "AKs": 1.0, "AKo": 1.0})
    simulated = shove_ev_full_deal(
        "88", 0.5, 10.0, [_heads_up_responder(call)], 1.5, iterations=200_000, seed=3
    )
    analytic = shove_ev_bb(
        "88", 9.5, 1.5, [CallerModel(call_range=call, behind_bb=9.0, posted_bb=1.0)],
        hero_posted_bb=0.5,
    )
    assert simulated == pytest.approx(analytic, abs=0.06)


def test_a_folded_player_who_never_folds_is_refused():
    """Условие «сбросившие сбросили» невыполнимо — отказ, а не бесконечный цикл."""
    everything = Range(weights={c: 1.0 for c in _all_classes()})
    with pytest.raises(ValueError, match="невыполнимо"):
        call_ev_full_deal(
            "AA",
            1.0,
            10.0,
            Shover(posted_bb=10.0, total_bb=10.0, push=everything),
            [everything],
            [],
            11.5,
            iterations=1,
            seed=1,
        )


def _all_classes() -> list[str]:
    from harness.contracts import all_classes

    return all_classes()


# --- Якорь: UTG шовит 10bb, восемь мест, BB-анте ---------------------------------

_SEATS = ("UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB")


def _anchor_seat(name: str) -> Seat:
    """Геометрия солвера: 10bb после анте; анте 1bb целиком с BB сверх стека."""
    if name == "SB":
        return Seat(posted_bb=0.5, behind_bb=9.5)
    if name == "BB":
        return Seat(posted_bb=1.0, behind_bb=9.0)
    return Seat(posted_bb=0.0, behind_bb=10.0)


@pytest.fixture(scope="module")
def anchor_solution():
    return unopened_shove_equilibrium(
        _anchor_seat("UTG"), [_anchor_seat(n) for n in _SEATS[1:]], 2.5
    )


@pytest.mark.slow
def test_call_narrows_with_every_player_behind_the_caller(anchor_solution):
    """Без оверколлов UTG+1…BTN коллировали одинаково (10.18%); теперь — по позиции."""
    widths = dict(zip(_SEATS[1:], anchor_solution.calls, strict=True))
    early_to_late = [widths[n].fraction_of_hands() for n in ("UTG+1", "LJ", "HJ", "CO", "BTN")]
    assert early_to_late == sorted(early_to_late)


# EV шова UTG по солверу, bb. Восстановлено из сверки владельца 2026-10-01:
# EV харнесса той сессии минус названная владельцем разница «харнесс − GTO».
_GTO_SHOVE_EV = {
    "QQ": 3.620,
    "TT": 2.196,
    "88": 1.187,
    "66": 0.468,
    "AKo": 2.365,
    "AQo": 1.610,
    "AJo": 0.893,
    "KQs": 0.676,
    "A5s": 0.093,
}

_QQ_EV_OPEN = pytest.mark.xfail(
    strict=True,
    reason=(
        "EV шова QQ ниже солвера на ~0.1bb устойчиво; причина не установлена. "
        "Кандидат: коллы солвера отвечают на его шов без премиумов (мин-рейз), "
        "наши — на шов со всеми премиумами, и против QQ они другие"
    ),
)


@pytest.fixture(scope="module")
def anchor_responders(anchor_solution):
    behind = [_anchor_seat(n) for n in _SEATS[1:]]
    return [
        Responder(posted_bb=s.posted_bb, total_bb=s.total_bb, cold=c, over=o)
        for s, c, o in zip(behind, anchor_solution.calls, anchor_solution.overcalls, strict=True)
    ]


@pytest.mark.slow  # 400 000 раздач на руку
@pytest.mark.parametrize(
    "hand",
    [pytest.param(h, marks=_QQ_EV_OPEN) if h == "QQ" else h for h in _GTO_SHOVE_EV],
)
def test_shove_ev_of_utg_follows_the_solver(anchor_responders, hand):
    """EV шова в пределах 0.08bb от солвера (спека §2.1) — и у пар, завышенных прежде на 0.3–0.4.

    Сверяется вся связка: диапазоны решателя и симуляция. 400 000 раздач, а не
    100 000 продукта: стандартная ошибка на 100 000 — 0.020–0.027bb, и на
    допуске 0.08 исход решал бы шум.
    """
    ev = shove_ev_full_deal(hand, 0.0, 10.0, anchor_responders, 2.5, iterations=400_000, seed=11)
    assert ev == pytest.approx(_GTO_SHOVE_EV[hand], abs=0.08)


# Шов UTG солвера (только олл-ин, без мин-рейза), снят владельцем 2026-10-01.
_GTO_UTG_SHOVE = (
    "TT, 99, 77, 66, AQs, AJs, ATs, A9s, A5s, A4s, KQs, KJs, KTs, K9s, QJs, QTs, JTs, "
    "T9s, AKo, AQo, AJo, KQo, JJ:0.8, 88:0.95, 44:0.95, 55:0.6, AKs:0.35, A8s:0.4, "
    "A7s:0.85, Q9s:0.45, J9s:0.8, ATo:0.65"
)

# EV колла BB против шова UTG по солверу, bb (владелец, 2026-10-01). Якорь только
# у BB: позади него никого, и всё, что входит в расчёт, — шов солвера и наши
# сбросившие. У UTG+1 позади шестеро, их оверколлы — наши, подобранные против
# нашего шова, а не солверного; расхождения там до 0.1bb (KK +0.09, AJs −0.10)
# записаны в спеке §9 как замер, якорем они не служат.
_GTO_BB_CALL_EV = {
    "AA": 9.19, "KK": 7.27, "QQ": 6.49, "JJ": 5.53, "TT": 4.43, "99": 3.23, "88": 2.51,
    "77": 1.92, "66": 1.36, "55": 0.88, "44": 0.34, "33": 0.06, "AKs": 4.64, "AKo": 4.24,
    "AQs": 3.75, "AQo": 3.30, "AJs": 2.64, "AJo": 2.12, "ATs": 1.79, "ATo": 1.21,
    "A9s": 0.97, "A9o": 0.31, "A8s": 0.62, "A5s": 0.22, "A4s": 0.04, "KQs": 1.25,
    "KQo": 0.65, "KJs": 0.67, "KJo": 0.01, "KTs": 0.37, "QJs": 0.36, "QTs": 0.05,
}


@pytest.mark.slow  # 200 000 раздач на руку, 32 руки
@pytest.mark.parametrize("hand", list(_GTO_BB_CALL_EV))
def test_bb_call_ev_against_the_solver_shove_follows_the_solver(anchor_solution, hand):
    """EV колла BB в пределах 0.08bb от солвера — против ЕГО шова UTG.

    Спот «колл шова» сверяется отдельно от решателя: диапазон шова — солвера, так
    что проверяется сам расчёт колла (деньги, вскрытие, карты шестерых
    сбросивших между шовером и BB). До подстановки шова солвера премиумы
    расходились на −0.8…−1.1bb: разница деревьев (мин-рейз), а не расчёта.
    """
    from harness.analysis.charts.notation import parse_range

    bb = _anchor_seat("BB")
    ev = call_ev_full_deal(
        hand,
        bb.posted_bb,
        bb.total_bb,
        Shover(posted_bb=10.0, total_bb=10.0, push=parse_range(_GTO_UTG_SHOVE)),
        list(anchor_solution.calls[:-1]),
        [],
        12.5,
        iterations=200_000,
        seed=7,
    )
    assert ev == pytest.approx(_GTO_BB_CALL_EV[hand], abs=0.08)
