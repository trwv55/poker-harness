"""Статистика места: VPIP, PFR, ре-рейз, продолженная ставка, баррели, вскрытие.

Синтетические руки собираются как `RawHand` и прогоняются через настоящий
конвейер (`normalize` → `enrich`) — тот же принцип, что в `test_scan.py`:
считать статистику по руке, которую движок не принял бы, значило бы проверять
её на входе, которого конвейер никогда не произведёт. Из результата берётся
`CanonicalHand` — вход `player_stats`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from harness.analysis.player_stats import (
    player_stats,
    player_stats_across_tournaments,
    player_stats_by_label,
)
from harness.contracts import (
    ActionKind,
    CanonicalHand,
    Post,
    PostKind,
    Provenance,
    RawAction,
    RawHand,
    SeatInfo,
    ShowdownEntry,
    Street,
    ValidationStatus,
)
from harness.engine import enrich
from harness.normalizer import normalize
from tests.conftest import FIXTURE_DAILY, FIXTURE_PKO, requires_fixtures

_SB = 1
_BB = 2
_STACK = 200
_SIX_MAX: tuple[str, ...] = ("SB", "BB", "UTG", "HJ", "CO", "BTN")
_FLOP = ["Qd", "8c", "3s"]
_TURN = ["2h"]
_RIVER = ["7d"]
_BOARD = {Street.FLOP: _FLOP, Street.TURN: _TURN, Street.RIVER: _RIVER}


# --- Синтетика ---------------------------------------------------------------------


def _act(
    label: str,
    kind: ActionKind,
    *,
    street: Street = Street.PREFLOP,
    amount: int | None = None,
    to_amount: int | None = None,
    all_in: bool = False,
) -> RawAction:
    return RawAction(
        street=street,
        label=label,
        kind=kind,
        amount=amount,
        to_amount=to_amount,
        is_all_in=all_in,
        raw_line=f"{label}: {kind.value} {amount or to_amount or ''}".strip(),
    )


def _fold(label: str, street: Street = Street.PREFLOP) -> RawAction:
    return _act(label, ActionKind.FOLD, street=street)


def _check(label: str, street: Street = Street.PREFLOP) -> RawAction:
    return _act(label, ActionKind.CHECK, street=street)


def _call(label: str, amount: int, street: Street = Street.PREFLOP, **kw) -> RawAction:
    return _act(label, ActionKind.CALL, street=street, amount=amount, **kw)


def _bet(label: str, amount: int, street: Street = Street.FLOP) -> RawAction:
    return _act(label, ActionKind.BET, street=street, amount=amount)


def _raise_to(label: str, to: int, already: int, street: Street = Street.PREFLOP, **kw):
    return _act(
        label, ActionKind.RAISE, street=street, amount=to - already, to_amount=to, **kw
    )


def _hand(
    *,
    hero_position: str,
    actions: list[RawAction],
    boards: dict[Street, list[str]] | None = None,
    hero_cards: tuple[str, str] = ("Ah", "Kd"),
    showdowns: list[ShowdownEntry] | None = None,
    tournament_id: str = "T1",
) -> CanonicalHand:
    """Рука 6-max, где место героя занимает метка `Hero`, а остальные названы позицией.

    Метка = имя позиции для всех, кроме героя, поэтому в тестах действия
    оппонентов пишутся прямо позицией («UTG»), а героя — «Hero».
    """
    labels = {pos: ("Hero" if pos == hero_position else pos) for pos in _SIX_MAX}
    seats = [
        SeatInfo(seat=i + 1, label=labels[pos], stack=_STACK) for i, pos in enumerate(_SIX_MAX)
    ]
    posts = [
        Post(label=labels["SB"], kind=PostKind.SMALL_BLIND, amount=_SB),
        Post(label=labels["BB"], kind=PostKind.BIG_BLIND, amount=_BB),
    ]
    raw = RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="synthetic",
        hand_no="SYN",
        tournament_id=tournament_id,
        tournament_name="synthetic",
        level=1,
        sb=_SB,
        bb=_BB,
        ante=0,
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        table_name="syn",
        max_seats=6,
        button_seat=6,
        seats=seats,
        posts=posts,
        dealt={"Hero": list(hero_cards)},
        actions=actions,
        boards=boards or {},
        showdowns=showdowns or [],
    )
    en = enrich(normalize(raw))
    assert en.verdict.status is not ValidationStatus.REJECT, en.verdict.reasons
    return en.hand


def _folded_to_hero_bb() -> CanonicalHand:
    """Все сбросили герою в большой блайнд — герой не действует вовсе."""
    return _hand(
        hero_position="BB",
        actions=[_fold(pos) for pos in ("UTG", "HJ", "CO", "BTN", "SB")],
    )


def _hero_completes_and_folds_to_a_shove() -> CanonicalHand:
    """Герой на малом блайнде уравнивает, большой блайнд шовит, герой пасует."""
    return _hand(
        hero_position="SB",
        actions=[
            *[_fold(pos) for pos in ("UTG", "HJ", "CO", "BTN")],
            _call("Hero", _BB - _SB),
            _raise_to("BB", _STACK, already=_BB, all_in=True),
            _fold("Hero"),
        ],
    )


def _hero_opens_and_takes_it() -> CanonicalHand:
    """Герой открывает рейзом с CO, все пасуют."""
    return _hand(
        hero_position="CO",
        actions=[
            _fold("UTG"),
            _fold("HJ"),
            _raise_to("Hero", 6, already=0),
            *[_fold(pos) for pos in ("BTN", "SB", "BB")],
        ],
    )


def _hero_faces_an_open(hero_reraises: bool) -> CanonicalHand:
    """UTG открывает рейзом, доходит до героя в большом блайнде."""
    tail = (
        [_raise_to("Hero", 18, already=_BB), _fold("UTG")]
        if hero_reraises
        else [_fold("Hero")]
    )
    return _hand(
        hero_position="BB",
        actions=[
            _raise_to("UTG", 6, already=0),
            *[_fold(pos) for pos in ("HJ", "CO", "BTN", "SB")],
            *tail,
        ],
    )


def _hero_faces_a_cbet(hero_folds: bool) -> CanonicalHand:
    """CO открывает, герой в большом блайнде уравнивает, на флопе CO ставит."""
    tail = (
        [_fold("Hero", Street.FLOP)]
        if hero_folds
        # 194 = стек минус 6, уже вложенных на префлопе: больше остатка герой
        # поставить не может, и движок бы такую руку отверг.
        else [
            _raise_to("Hero", _STACK - 6, already=0, street=Street.FLOP, all_in=True),
            _fold("CO", Street.FLOP),
        ]
    )
    return _hand(
        hero_position="BB",
        actions=[
            _fold("UTG"),
            _fold("HJ"),
            _raise_to("CO", 6, already=0),
            _fold("BTN"),
            _fold("SB"),
            _call("Hero", 4),
            _check("Hero", Street.FLOP),
            _bet("CO", 6),
            *tail,
        ],
        boards={Street.FLOP: _FLOP},
    )


def _hero_is_the_cbettor() -> CanonicalHand:
    """Герой открывает с CO, ставит на флопе сам и пасует на чек-рейз.

    Пас героя ПОСЛЕ первой ставки на флопе стоит здесь намеренно: без него
    формула сдачи на c-bet возвращала бы «возможности не было» просто потому,
    что герой больше не действовал, и проверка «своя ставка — не возможность»
    была бы пустой (найдено фальсификацией: снятие условия про агрессора тест
    не роняло).
    """
    return _hand(
        hero_position="CO",
        actions=[
            _fold("UTG"),
            _fold("HJ"),
            _raise_to("Hero", 6, already=0),
            _fold("BTN"),
            _fold("SB"),
            _call("BB", 4),
            _check("BB", Street.FLOP),
            _bet("Hero", 6),
            _raise_to("BB", 20, already=6, street=Street.FLOP),
            _fold("Hero", Street.FLOP),
        ],
        boards={Street.FLOP: _FLOP},
    )


def _flop_bet_comes_from_a_caller() -> CanonicalHand:
    """Первую ставку на флопе делает не префлоп-агрессор, а его коллер."""
    return _hand(
        hero_position="BB",
        actions=[
            _fold("UTG"),
            _fold("HJ"),
            _raise_to("CO", 6, already=0),
            _call("BTN", 6),
            _fold("SB"),
            _call("Hero", 4),
            _check("Hero", Street.FLOP),
            _check("CO", Street.FLOP),
            _bet("BTN", 9),
            _fold("Hero", Street.FLOP),
            _fold("CO", Street.FLOP),
        ],
        boards={Street.FLOP: _FLOP},
    )


def _barrel_line(
    *,
    hero_bets_turn: bool = True,
    bb_donks_turn: bool = False,
    bb_folds_river: bool = False,
    bb_shows: bool = False,
) -> CanonicalHand:
    """Герой открывает с CO, BB уравнивает, и рука идёт до ривера.

    Флоп герой ставит всегда — это точка отсчёта для баррелей; дальше поведение
    тёрна и ривера задаётся флагами.
    """
    if bb_donks_turn:
        turn = [
            _bet("BB", 20, street=Street.TURN),
            _raise_to("Hero", 60, already=0, street=Street.TURN),
            _call("BB", 40, street=Street.TURN),
        ]
    elif hero_bets_turn:
        turn = [
            _check("BB", Street.TURN),
            _bet("Hero", 20, street=Street.TURN),
            _call("BB", 20, street=Street.TURN),
        ]
    else:
        turn = [_check("BB", Street.TURN), _check("Hero", Street.TURN)]
    river_tail = (
        [_fold("BB", Street.RIVER)]
        if bb_folds_river
        else [_call("BB", 30, street=Street.RIVER)]
    )
    return _hand(
        hero_position="CO",
        actions=[
            _fold("UTG"),
            _fold("HJ"),
            _raise_to("Hero", 6, already=0),
            _fold("BTN"),
            _fold("SB"),
            _call("BB", 4),
            _check("BB", Street.FLOP),
            _bet("Hero", 10),
            _call("BB", 10, street=Street.FLOP),
            *turn,
            _check("BB", Street.RIVER),
            _bet("Hero", 30, street=Street.RIVER),
            *river_tail,
        ],
        boards=_BOARD,
        showdowns=[ShowdownEntry(label="BB", cards=["Js", "Td"])] if bb_shows else None,
    )


def _with_other_labels(hand: CanonicalHand) -> CanonicalHand:
    """Та же раздача с другими метками оппонентов — как другой стол того же турнира.

    Нужна затем, чтобы «у каждого места свой знаменатель» вообще можно было
    проверить: у всех синтетических раздач выше состав меток один и тот же, и
    подсчёт, приписывающий каждую метку каждой раздаче, на них неотличим от
    верного.
    """
    data = hand.model_dump()
    for group in ("players", "actions", "posts", "showdowns"):
        for row in data[group]:
            if row["label"] != hand.hero_label:
                row["label"] = row["label"] + "*"
    return CanonicalHand.model_validate(data)


def _hero_shoves_and_is_called() -> CanonicalHand:
    """Герой шовит с CO, BB уравнивает олл-ин — доска доезжает без единого действия."""
    return _hand(
        hero_position="CO",
        actions=[
            _fold("UTG"),
            _fold("HJ"),
            _raise_to("Hero", _STACK, already=0, all_in=True),
            _fold("BTN"),
            _fold("SB"),
            _call("BB", _STACK - _BB, all_in=True),
        ],
        boards=_BOARD,
    )


# --- VPIP / PFR --------------------------------------------------------------------


def test_a_posted_blind_is_not_a_voluntary_investment():
    """Большой блайнд, которому все сбросили, — 0 VPIP при 1 раздаче.

    Блайнд лежит в `CanonicalHand.posts`, а не в действиях, и попасть в VPIP не
    может по построению формулы: у героя в этой руке действий нет ни одного.
    """
    stats = player_stats([_folded_to_hero_bb()])
    assert stats.hands == 1
    assert stats.vpip == 0 and stats.pfr == 0
    assert stats.vpip_pct == 0.0


def test_a_call_is_vpip_but_not_pfr():
    stats = player_stats([_hero_completes_and_folds_to_a_shove()])
    assert (stats.hands, stats.vpip, stats.pfr) == (1, 1, 0)


def test_an_open_raise_is_both_vpip_and_pfr():
    stats = player_stats([_hero_opens_and_takes_it()])
    assert (stats.hands, stats.vpip, stats.pfr) == (1, 1, 1)


def test_shares_are_counted_against_every_dealt_hand():
    """Знаменатель VPIP/PFR — все раздачи, включая те, где герой не действовал."""
    hands = [_folded_to_hero_bb(), _folded_to_hero_bb(), _hero_opens_and_takes_it()]
    stats = player_stats(hands)
    assert stats.hands == 3
    assert stats.vpip_pct is not None and round(stats.vpip_pct, 1) == 33.3
    assert stats.pfr_pct is not None and round(stats.pfr_pct, 1) == 33.3


# --- Ре-рейз (3-бет и выше) --------------------------------------------------------


def test_an_open_raise_is_not_a_reraise_chance():
    """Открывая банк, герой ни на что не отвечает — возможности ре-рейза нет."""
    stats = player_stats([_hero_opens_and_takes_it()])
    assert (stats.reraise_chances, stats.reraise) == (0, 0)
    assert stats.reraise_pct is None


def test_folding_to_an_open_is_a_chance_taken_no():
    stats = player_stats([_hero_faces_an_open(hero_reraises=False)])
    assert (stats.reraise_chances, stats.reraise) == (1, 0)
    assert stats.reraise_pct == 0.0


def test_raising_over_an_open_is_a_reraise():
    stats = player_stats([_hero_faces_an_open(hero_reraises=True)])
    assert (stats.reraise_chances, stats.reraise) == (1, 1)
    assert stats.reraise_pct == 100.0


# --- Сдача на продолженную ставку --------------------------------------------------


def test_folding_to_the_aggressors_flop_bet_is_counted():
    stats = player_stats([_hero_faces_a_cbet(hero_folds=True)])
    assert (stats.cbet_faced, stats.fold_to_cbet) == (1, 1)
    assert stats.fold_to_cbet_pct == 100.0


def test_facing_a_cbet_without_folding_is_a_chance_taken_no():
    stats = player_stats([_hero_faces_a_cbet(hero_folds=False)])
    assert (stats.cbet_faced, stats.fold_to_cbet) == (1, 0)
    assert stats.fold_to_cbet_pct == 0.0


def test_hero_own_cbet_is_not_a_chance():
    """Ставку делает сам герой — сдаваться ему не на что."""
    stats = player_stats([_hero_is_the_cbettor()])
    assert (stats.cbet_faced, stats.fold_to_cbet) == (0, 0)
    assert stats.fold_to_cbet_pct is None


def test_a_flop_bet_from_a_caller_is_not_a_continuation_bet():
    """Первая ставка на флопе от коллера — не продолженная: агрессор чекнул.

    Без этого различия статистика считала бы любую ставку на флопе
    продолженной, и «сдача на c-bet» перестала бы значить то, как её читает
    игрок.
    """
    stats = player_stats([_flop_bet_comes_from_a_caller()])
    assert (stats.cbet_faced, stats.fold_to_cbet) == (0, 0)


# --- Среднее по нескольким турнирам ------------------------------------------------


def test_hands_from_several_tournaments_are_weighed_by_hands():
    """Счётчики складываются по рукам, а не усредняются по турнирам.

    Три раздачи без добровольных вложений и одна с ними дают 25%, а не 50% —
    столько вышло бы, усредни мы проценты двух турниров (0% и 100%).
    """
    tournament_a = [_folded_to_hero_bb() for _ in range(3)]
    tournament_b = [_hero_opens_and_takes_it()]
    stats = player_stats([*tournament_a, *tournament_b])
    assert stats.hands == 4
    assert stats.vpip_pct == 25.0


# --- Пустой знаменатель ------------------------------------------------------------


def test_no_hands_means_no_shares_at_all():
    """Ноль раздач — доли отсутствуют, а не равны нулю: «0 из 0» не ноль процентов."""
    stats = player_stats([])
    assert stats.hands == 0
    assert stats.vpip_pct is None and stats.pfr_pct is None
    assert stats.reraise_pct is None and stats.fold_to_cbet_pct is None


# --- Любое место, не только герой --------------------------------------------------


def test_stats_are_counted_for_a_seat_that_is_not_hero():
    """Открывший банк оппонент получает свои VPIP и PFR, а не статистику героя."""
    hand = _hero_faces_an_open(hero_reraises=True)
    opener = player_stats([hand], "UTG")
    assert (opener.hands, opener.vpip, opener.pfr) == (1, 1, 1)
    assert (opener.reraise_chances, opener.reraise) == (0, 0)
    hero = player_stats([hand])
    assert (hero.reraise_chances, hero.reraise) == (1, 1)


def test_hands_without_that_label_are_not_counted():
    """Метки нет за столом — раздача не идёт ни в числитель, ни в знаменатель."""
    stats = player_stats([_hero_opens_and_takes_it()], "нет-такого-места")
    assert stats.hands == 0
    assert stats.vpip_pct is None


def test_by_label_matches_the_single_label_call():
    """Словарь по всем местам и запрос одного места дают одно и то же."""
    hands = [_barrel_line(), _hero_faces_an_open(hero_reraises=True)]
    by_label = player_stats_by_label(hands)
    for label in ("Hero", "UTG", "BB"):
        assert by_label[label] == player_stats(hands, label)


def test_by_label_counts_each_label_in_its_own_hands():
    """У каждого места свой знаменатель: считаются только те раздачи, где оно за столом."""
    first = _hero_opens_and_takes_it()
    by_label = player_stats_by_label([first, _with_other_labels(first)])
    assert by_label["Hero"].hands == 2
    assert by_label["BB"].hands == 1 and by_label["BB*"].hands == 1
    assert set(by_label) == {"Hero"} | {
        f"{pos}{star}" for pos in ("SB", "BB", "UTG", "HJ", "BTN") for star in ("", "*")
    }


# --- Продолженная ставка и баррели -------------------------------------------------


def test_the_preflop_aggressor_betting_the_flop_is_a_cbet():
    stats = player_stats([_barrel_line()])
    assert (stats.cbet_flop_chances, stats.cbet_flop) == (1, 1)
    assert stats.cbet_flop_pct == 100.0


def test_a_check_by_the_aggressor_is_a_chance_not_taken():
    """Агрессор чекнул флоп — возможность была, ставки не было."""
    stats = player_stats([_flop_bet_comes_from_a_caller()], "CO")
    assert (stats.cbet_flop_chances, stats.cbet_flop) == (1, 0)
    assert stats.cbet_flop_pct == 0.0


def test_only_the_preflop_aggressor_gets_a_cbet_chance():
    """Коллер на флопе тоже действует, но продолжать ему нечего — открывал не он."""
    stats = player_stats([_barrel_line()], "BB")
    assert (stats.cbet_flop_chances, stats.cbet_flop) == (0, 0)
    assert stats.cbet_flop_pct is None


def test_an_all_in_aggressor_has_no_flop_bet_chance():
    """Агрессор вошёл во флоп олл-ином — действий у него там нет, ставить нечем."""
    stats = player_stats([_hero_shoves_and_is_called()])
    assert (stats.cbet_flop_chances, stats.cbet_flop) == (0, 0)
    assert stats.cbet_flop_pct is None


def test_a_bet_on_each_street_counts_as_a_second_and_third_barrel():
    stats = player_stats([_barrel_line()])
    assert (stats.barrel_turn_chances, stats.barrel_turn) == (1, 1)
    assert (stats.barrel_river_chances, stats.barrel_river) == (1, 1)


def test_a_barrel_needs_a_bet_on_the_street_before():
    """Чек на тёрне: возможность второго барреля была, третьего — не было вовсе."""
    stats = player_stats([_barrel_line(hero_bets_turn=False)])
    assert (stats.barrel_turn_chances, stats.barrel_turn) == (1, 0)
    assert (stats.barrel_river_chances, stats.barrel_river) == (0, 0)
    assert stats.barrel_river_pct is None


def test_raising_someone_elses_turn_bet_is_not_a_barrel():
    """Первым на тёрне поставил оппонент — повышение поверх баррелем не считается."""
    stats = player_stats([_barrel_line(bb_donks_turn=True)])
    assert (stats.barrel_turn_chances, stats.barrel_turn) == (1, 0)
    assert (stats.barrel_river_chances, stats.barrel_river) == (0, 0)


# --- Доход до вскрытия -------------------------------------------------------------


def test_reaching_the_river_leaves_both_players_at_showdown():
    by_label = player_stats_by_label([_barrel_line()])
    for label in ("Hero", "BB"):
        assert (by_label[label].flops_seen, by_label[label].showdowns) == (1, 1)
        assert by_label[label].showdown_pct == 100.0


def test_folding_preflop_is_not_a_flop_seen():
    """Спасовавший до флопа флопа не видел — он не в знаменателе."""
    stats = player_stats([_barrel_line()], "UTG")
    assert (stats.flops_seen, stats.showdowns) == (0, 0)
    assert stats.showdown_pct is None


def test_folding_on_the_flop_still_counts_as_seeing_it():
    """Знаменатель считает дошедших до улицы, а не доигравших её."""
    stats = player_stats([_hero_faces_a_cbet(hero_folds=True)])
    assert (stats.flops_seen, stats.showdowns) == (1, 0)
    assert stats.showdown_pct == 0.0


def test_a_river_fold_leaves_no_showdown_for_anyone():
    """Последнюю ставку не уравняли — карт не открыл никто, включая победителя."""
    by_label = player_stats_by_label([_barrel_line(bb_folds_river=True)])
    for label in ("Hero", "BB"):
        assert (by_label[label].flops_seen, by_label[label].showdowns) == (1, 0)


def test_cards_shown_after_a_fold_are_not_a_showdown():
    """Источник записал показ карт за спасовавшим — вскрытием это не является."""
    stats = player_stats([_barrel_line(bb_folds_river=True, bb_shows=True)], "BB")
    assert (stats.flops_seen, stats.showdowns) == (1, 0)


# --- Малая выборка -----------------------------------------------------------------


def test_a_single_observation_is_returned_with_its_denominator():
    """Порога на малую выборку нет: доля от одного наблюдения отдаётся вместе с ним.

    Скрыть её значило бы скрыть и знаменатель, а именно он и говорит читателю,
    чего эта доля стоит.
    """
    stats = player_stats([_hero_faces_a_cbet(hero_folds=True)])
    assert stats.fold_to_cbet_pct == 100.0
    assert (stats.fold_to_cbet, stats.cbet_faced) == (1, 1)


# --- Настоящие файлы ---------------------------------------------------------------


@requires_fixtures
@pytest.mark.parametrize("path", [FIXTURE_DAILY, FIXTURE_PKO])
def test_showdown_counter_matches_the_cards_the_source_recorded(path):
    """Доход до вскрытия, посчитанный по пасам и доске, сходится с показом карт в файле.

    Числитель считается двумя независимыми путями и по разным полям: у
    `player_stats_by_label` — по пасам и доске, здесь — по строкам показа карт
    самого источника. Из показов вычитаются двое: спасовавшие
    (`test_cards_shown_after_a_fold_are_not_a_showdown`) и одиночные — показ
    карт единственным непасовавшим вскрытием не является, он добровольный
    (`test_a_river_fold_leaves_no_showdown_for_anyone`).
    """
    from harness.parsers.hh_parser import parse_file

    hands = [
        normalize(raw)
        for raw in parse_file(path.read_text(encoding="utf-8"), source_ref=path.name)
    ]
    by_label = player_stats_by_label(hands)
    from_source = 0
    for hand in hands:
        folded = {a.label for a in hand.actions if a.kind is ActionKind.FOLD}
        shown = {sd.label for sd in hand.showdowns} - folded
        from_source += len(shown) if len(shown) >= 2 else 0
    assert sum(stats.showdowns for stats in by_label.values()) == from_source


@requires_fixtures
@pytest.mark.parametrize("path", [FIXTURE_DAILY, FIXTURE_PKO])
def test_every_numerator_stays_within_its_own_denominator(path):
    """Знаменатели вложены: третий баррель ⊆ второго ⊆ сибета, вскрытие ⊆ флопов."""
    from harness.parsers.hh_parser import parse_file

    hands = [
        normalize(raw)
        for raw in parse_file(path.read_text(encoding="utf-8"), source_ref=path.name)
    ]
    for stats in player_stats_by_label(hands).values():
        assert stats.vpip <= stats.hands and stats.pfr <= stats.vpip
        assert stats.reraise <= stats.reraise_chances
        assert stats.fold_to_cbet <= stats.cbet_faced
        assert stats.cbet_flop <= stats.cbet_flop_chances
        assert stats.barrel_turn_chances <= stats.cbet_flop
        assert stats.barrel_turn <= stats.barrel_turn_chances
        assert stats.barrel_river_chances <= stats.barrel_turn
        assert stats.barrel_river <= stats.barrel_river_chances
        assert stats.showdowns <= stats.flops_seen <= stats.hands


# --- один человек под разными метками в разных турнирах -----------------------------


def _opens_from(position: str, tournament_id: str) -> CanonicalHand:
    """Раздача, где место `position` открывает рейзом и забирает банк.

    Герой сидит в большом блайнде и пасует, поэтому от турнира к турниру
    меняется только то, за каким местом сидит интересующий нас человек — ровно
    то, ради чего эти руки и собраны.
    """
    # Порядок хода на префлопе, а не порядок мест: раздачу, где место говорит
    # не в свой черёд, движок отвергает.
    order = ("UTG", "HJ", "CO", "BTN", "SB")
    opener = order.index(position)
    return _hand(
        hero_position="BB",
        tournament_id=tournament_id,
        actions=[
            *[_fold(pos) for pos in order[:opener]],
            _raise_to(position, 6, already=0),
            *[_fold(pos) for pos in order[opener + 1 :]],
            _fold("Hero"),
        ],
    )


def _counters(stats) -> dict[str, int]:
    return {name: getattr(stats, name) for name in type(stats).model_fields}


def test_stats_across_tournaments_add_up_what_each_tournament_counted():
    """Человек, у которого в каждом турнире своя метка, считается одной статистикой:
    числители и знаменатели турниров складываются — все до одного.
    """
    first = [_opens_from("UTG", "T1")]
    second = [_opens_from("CO", "T2")]

    across = player_stats_across_tournaments(first + second, {"T1": "UTG", "T2": "CO"})

    apart = _counters(player_stats(first, "UTG"))
    other = _counters(player_stats(second, "CO"))
    assert _counters(across) == {name: apart[name] + other[name] for name in apart}
    assert (across.hands, across.vpip, across.pfr) == (2, 2, 2)


def test_a_tournament_nobody_bound_is_not_counted():
    """Про турнир, которого нет в соответствии, не сказано, кто в нём наш, — и он
    не попадает ни в числитель, ни в знаменатель.
    """
    hands = [_opens_from("UTG", "T1"), _opens_from("UTG", "T3")]

    only_bound = player_stats_across_tournaments(hands, {"T1": "UTG"})

    assert _counters(only_bound) == _counters(player_stats([hands[0]], "UTG"))
    assert only_bound.hands == 1


def test_a_label_that_never_sat_at_that_table_is_not_counted():
    """Метка, которой за столом нет, не указывает ни на кого: раздача пропускается
    целиком, а не увеличивает знаменатель на пустом месте.
    """
    hands = [_opens_from("UTG", "T1")]

    assert _counters(player_stats_across_tournaments(hands, {"T1": "нет-такого"})) == _counters(
        player_stats([], "UTG")
    )


def test_the_same_person_under_two_labels_is_not_the_sum_of_two_seats():
    """Проверка того, что метка берётся ПОСВОЕМУ турниру, а не одна на все:
    у метки `UTG` в T2 сидит другой человек, и его раздачи в счёт не идут.
    """
    hands = [_opens_from("UTG", "T1"), _opens_from("CO", "T2")]

    by_tournament = player_stats_across_tournaments(hands, {"T1": "UTG", "T2": "CO"})
    one_label_everywhere = player_stats(hands, "UTG")

    assert (by_tournament.pfr, one_label_everywhere.pfr) == (2, 1)
