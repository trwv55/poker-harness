"""Статистика игрока: VPIP, PFR, ре-рейз, сдача на продолженную ставку (задача 23).

Синтетические руки собираются как `RawHand` и прогоняются через настоящий
конвейер (`normalize` → `enrich`) — тот же принцип, что в `test_scan.py`:
считать статистику по руке, которую движок не принял бы, значило бы проверять
её на входе, которого конвейер никогда не произведёт. Из результата берётся
`CanonicalHand` — вход `player_stats`.
"""

from __future__ import annotations

from datetime import UTC, datetime

from harness.analysis.player_stats import player_stats
from harness.contracts import (
    ActionKind,
    CanonicalHand,
    Post,
    PostKind,
    Provenance,
    RawAction,
    RawHand,
    SeatInfo,
    Street,
    ValidationStatus,
)
from harness.engine import enrich
from harness.normalizer import normalize

_SB = 1
_BB = 2
_STACK = 200
_SIX_MAX: tuple[str, ...] = ("SB", "BB", "UTG", "HJ", "CO", "BTN")
_FLOP = ["Qd", "8c", "3s"]


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
        tournament_id="T1",
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
