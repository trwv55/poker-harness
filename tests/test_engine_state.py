"""Путь состояния в точке решения: что движок и валидатор умеют на неполном входе.

Фикстура — живой стол из записки спайка (`СВЕРКА-2`, экран A): Bounty Hunters,
уровень 27, блайнды 10 000/20 000, пул анте 24 000 на восьмерых, показанный банк
14.7 BB. Ники заменены на нейтральные метки: реальные ники — приватные данные
игрока (docs/publishing-policy.md), а проверяемые здесь величины от них не
зависят.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from harness.analysis import analyze_hand
from harness.contracts import (
    ActionKind,
    Completeness,
    PostKind,
    Provenance,
    RawAction,
    RawHand,
    SeatInfo,
    Street,
    ValidationStatus,
)
from harness.contracts.raw import Post, VisionMeta
from harness.engine import enrich
from harness.engine.state import STATE_NOT_CHECKED, StateNotReadable, state_report
from harness.normalizer import normalize

BB = 20_000
SB = 10_000
ANTE = 3_000

# Место -> (метка, показанный на экране остаток, фишки перед игроком, в руке ли).
# Порядок мест выбран так, чтобы кнопка стояла на месте 5: тогда место 6 — малый
# блайнд, место 7 — большой (герой), и раскладка позиций выводится из кнопки,
# как её и выводит нормалайзер.
_SEATS: list[tuple[int, str, int, int, bool]] = [
    (1, "P1", 460_000, 0, False),
    (2, "P2", 146_000, 0, False),
    (3, "P3", 0, 200_000, True),  # олл-ин на 10 BB
    (4, "P4", 96_000, 0, False),
    (5, "P5", 508_000, 0, False),  # кнопка
    (6, "P6", 392_000, SB, True),
    (7, "Hero", 734_000, BB, True),
    (8, "P8", 504_000, 40_000, True),  # открыл на 2 BB
]


def state_hand(**over) -> RawHand:
    """`RawHand` состояния так, как его собирает адаптер: стек = остаток + ставка + анте."""
    base = {
        "provenance": Provenance.SCREENSHOT,
        "completeness": Completeness.STATE,
        "source_ref": "screenshot",
        "hand_no": "",
        "tournament_id": "",
        "tournament_name": "Bounty Hunters Deepstack Turbo",
        "level": 27,
        "sb": SB,
        "bb": BB,
        "ante": ANTE,
        "timestamp": datetime(2026, 8, 15, 0, 35, tzinfo=UTC),
        "table_name": "172",
        "max_seats": 8,
        "button_seat": 5,
        "seats": [
            SeatInfo(seat=seat, label=label, stack=shown + bet + ANTE)
            for seat, label, shown, bet, _live in _SEATS
        ],
        "visible_bets": {label: bet for _s, label, _shown, bet, _live in _SEATS if bet},
        "posts": [
            Post(label="P6", kind=PostKind.SMALL_BLIND, amount=SB),
            Post(label="Hero", kind=PostKind.BIG_BLIND, amount=BB),
        ],
        "dealt": {"Hero": ["Ah", "2h"]},
        # Пас — перевод наблюдения «перед игроком нет карт» (см. `engine.state`).
        "actions": [
            RawAction(
                street=Street.PREFLOP,
                label=label,
                kind=ActionKind.FOLD,
                raw_line=f"нет карт у места {seat}",
            )
            for seat, label, _shown, _bet, live in _SEATS
            if not live
        ],
        "vision": VisionMeta(displayed_pot=294_000),
    }
    base.update(over)
    return RawHand.model_validate(base)


def test_a_live_table_gives_exactly_one_decision_point_and_it_is_the_hero_s():
    """Состояние судит точку героя и только её: чужих решений экран не показывает."""
    report = state_report(normalize(state_hand()))
    assert [dp.label for dp in report.decision_points] == ["Hero"]
    assert report.decision_points[0].position == "BB"


def test_the_pot_of_a_state_is_the_visible_contributions_plus_the_ante_pool():
    """14.7 BB на экране = 10 + 2 + 1 + 0.5 ставок плюс 1.2 пула анте (реестр D2)."""
    dp = state_report(normalize(state_hand())).decision_points[0]
    assert dp.pot_before == 200_000 + 40_000 + BB + SB + 8 * ANTE
    assert dp.pot_before == 294_000


def test_the_call_amount_is_the_biggest_bet_minus_what_the_hero_already_put_in():
    """Кнопка «Колл 9 BB» на экране = 10 BB чужого олл-ина минус свой большой блайнд."""
    dp = state_report(normalize(state_hand())).decision_points[0]
    assert dp.to_call == 200_000 - BB


def test_the_depth_of_a_state_is_capped_by_the_deepest_live_opponent():
    """Глубина — потолок руки: больше, чем есть у самого глубокого живого, не проиграть.

    Та же ветка, что у реплея при отсутствии агрессора (`replay._effective_stack`):
    агрессора на состоянии выделить не из чего, экран не говорит, кто ставил, а кто
    отвечал.
    """
    dp = state_report(normalize(state_hand())).decision_points[0]
    assert dp.eff_stack == 504_000 + 40_000  # самый глубокий живой — открывший на 2 BB
    assert dp.eff_stack_bb == pytest.approx(27.2)


def test_a_state_has_no_taken_action_and_no_end_of_hand():
    """Ни сыгранного действия, ни конца стеков: рука не сыграна и не кончилась."""
    report = state_report(normalize(state_hand()))
    assert report.decision_points[0].action is None
    assert report.stacks_end == {}


def test_the_validator_does_not_call_an_incomplete_input_broken():
    """Неполный вход — не битый: денежных сверок на нём нет по построению."""
    en = enrich(normalize(state_hand()))
    assert en.verdict.status is ValidationStatus.PASS
    assert en.verdict.reasons == []


def test_the_validator_names_every_check_it_could_not_run_on_a_state():
    """Тишина вместо проверки неотличима от пройденной проверки — отсюда список."""
    en = enrich(normalize(state_hand()))
    assert set(STATE_NOT_CHECKED) <= set(en.verdict.not_checked)
    assert "сохранение фишек" in en.verdict.not_checked


def test_a_full_hand_never_carries_the_state_refusals():
    """У полной руки проверять есть чем, и пустой `not_checked` это утверждает."""
    from harness.parsers.hh_parser import parse_hand
    from tests.test_hh_parser import SAMPLE

    en = enrich(normalize(parse_hand(SAMPLE, source_ref="x")))
    assert en.verdict.not_checked == []


def test_the_button_check_still_runs_on_a_state_and_catches_a_wrong_button():
    """Единственная денежная улика состояния — две прочитанные врозь рассадки.

    Кнопка читается отдельно от фишек перед игроками; сместив кнопку на одно
    место, получаем малый и большой блайнд не у тех, и валидатор обязан это
    назвать (реестр D3: на скрине эта сверка единственная).
    """
    en = enrich(normalize(state_hand(button_seat=4)))
    assert en.verdict.status is ValidationStatus.ESCALATE
    assert "button" in en.verdict.fields


def test_the_core_refuses_to_judge_a_decision_that_has_not_been_taken():
    """Отказ с названной причиной, а не «ошибок нет»: герой ещё не сходил."""
    result = analyze_hand(enrich(normalize(state_hand())))
    assert len(result.points) == 1
    point = result.points[0]
    assert point.best_action == ""
    assert result.ranked == []
    assert "решение ещё не принято" in point.detail["unjudged"]


def test_the_state_path_refuses_a_complete_hand():
    """Полноту решает вход, и путь состояния не берётся за руку целиком."""
    with pytest.raises(StateNotReadable):
        state_report(normalize(state_hand(completeness=Completeness.HAND)))


def test_the_state_path_refuses_a_board_that_cannot_be_a_street():
    with pytest.raises(StateNotReadable):
        state_report(normalize(state_hand(boards={"flop": ["Ah", "Kd"]})))


def test_the_player_is_told_why_a_state_has_no_verdict(monkeypatch):
    """Названная причина доходит до сообщения, а не остаётся в `detail` (ревью, C).

    «Кинул скрин за столом» — главный сценарий продукта, и общая строка «точек с
    вердиктом нет» была бы на нём ответом ни о чём.
    """
    from harness.presentation import deep_dive_msg

    result = analyze_hand(enrich(normalize(state_hand())))
    msg = deep_dive_msg(result, 5, None, 17, 50)
    assert "Решение по этой раздаче ещё не принято" in msg.text
    assert "точек с вердиктом нет" not in msg.text


def test_the_unchecked_list_of_a_state_reaches_the_player():
    """Проверок на состоянии нет по построению — и это обязано быть видно."""
    from harness.presentation import deep_dive_msg

    en = enrich(normalize(state_hand()))
    result = analyze_hand(en)
    msg = deep_dive_msg(result, 5, None, 17, 50, not_checked=en.verdict.not_checked)
    assert "Проверить на этом экране было нечем" in msg.text
    assert "сохранение фишек" in msg.text
