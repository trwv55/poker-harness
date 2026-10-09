"""Постфлоп-линия героя: тип действия, баррель, размер, назначение, пороги, причины.

Спека — `docs/superpowers/specs/2026-10-09-postflop-line-design.md`, гейты §9
(1 в части ядра, 4–8, 13). Руки синтетические, метки мест свои, но каждая
собирается как `RawHand` и проходит настоящий конвейер (`normalize` → `enrich`):
линия проверяется на входе, который конвейер действительно производит.
"""

from __future__ import annotations

from datetime import UTC, datetime
from fractions import Fraction
from math import ceil, comb

import pytest

from harness.analysis import analyze_hand, rank_points, total_ev_loss_bb, verdict_for
from harness.analysis.classifier import (
    POSTFLOP_CHECK_REASON,
    POSTFLOP_FOLD_FREQUENCY_REASON,
)
from harness.analysis.postflop_line import postflop_line
from harness.analysis.river import river_verdict
from harness.analysis.turn_flop import turn_flop_verdict
from harness.contracts import (
    POSTFLOP_LINE_DETAIL,
    ActionKind,
    ActionTag,
    Collected,
    DecisionPoint,
    DrawKind,
    EnrichedHand,
    HandCategory,
    Post,
    PostflopLineDetail,
    PostKind,
    Provenance,
    Purpose,
    RawAction,
    RawHand,
    SeatInfo,
    ShowdownEntry,
    SizeTag,
    Street,
    StrengthClass,
    ValidationStatus,
    is_judged,
    postflop_line_detail,
)
from harness.engine import enrich
from harness.normalizer import normalize

P, F, T, R = Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER
_SB, _BB = 50, 100
_STACK = 10_000

# Борд по умолчанию: у героя с 5♣3♣ на нём ни пары, ни дро, ни оверкарт.
_BOARD = {F: ["Kh", "9d", "7s"], T: ["2c"], R: ["Jd"]}
_AIR = ("5c", "3c")


def _act(
    street: Street,
    label: str,
    kind: ActionKind,
    amount: int | None = None,
    to: int | None = None,
) -> RawAction:
    """`amount` — доплата колла и ставки, `to` — итог рейза на улице (как в источнике)."""
    return RawAction(
        street=street,
        label=label,
        kind=kind,
        amount=amount,
        to_amount=to,
        raw_line=f"{label}: {kind}",
    )


def _check(street: Street, label: str) -> RawAction:
    return _act(street, label, ActionKind.CHECK)


def _bet(street: Street, label: str, amount: int) -> RawAction:
    return _act(street, label, ActionKind.BET, amount)


def _call(street: Street, label: str, amount: int) -> RawAction:
    return _act(street, label, ActionKind.CALL, amount)


def _raise(street: Street, label: str, to: int) -> RawAction:
    return _act(street, label, ActionKind.RAISE, to=to)


def _fold(street: Street, label: str) -> RawAction:
    return _act(street, label, ActionKind.FOLD)


def _hand(
    actions: list[RawAction],
    *,
    seats: tuple[tuple[str, int], ...] = (("Hero", _STACK), ("V", _STACK)),
    button: str = "Hero",
    hero_cards: tuple[str, str] = _AIR,
    boards: dict[Street, list[str]] | None = None,
) -> EnrichedHand:
    """Стол без анте; блайнды ставят места после кнопки (в хедз-апе SB — кнопка)."""
    seat_infos = [
        SeatInfo(seat=n, label=label, stack=stack) for n, (label, stack) in enumerate(seats, 1)
    ]
    labels = [label for label, _ in seats]
    at = labels.index(button)
    if len(labels) == 2:
        small, big = labels[at], labels[1 - at]
    else:
        small, big = labels[(at + 1) % len(labels)], labels[(at + 2) % len(labels)]
    raw = RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="synthetic",
        hand_no="SYN-PL-1",
        tournament_id="TSYN",
        tournament_name="synthetic",
        level=1,
        sb=_SB,
        bb=_BB,
        ante=0,
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        table_name="syn",
        max_seats=len(seats),
        button_seat=at + 1,
        seats=seat_infos,
        posts=[
            Post(label=small, kind=PostKind.SMALL_BLIND, amount=_SB),
            Post(label=big, kind=PostKind.BIG_BLIND, amount=_BB),
        ],
        dealt={"Hero": list(hero_cards)},
        actions=actions,
        boards=boards or _BOARD,
    )
    en = _played_out(raw)
    assert en.verdict.status is not ValidationStatus.REJECT, en.verdict.reasons
    return en


_STREETS = (P, F, T, R)
_OUT_OF_ACTIONS = "действия источника кончились, а ход за "


def _played_out(raw: RawHand) -> EnrichedHand:
    """Доиграть руку до конца: тесту нужна только её начало.

    Пока движок говорит «ход за X», а действий нет, X получает первое легальное
    из: пас на улице последнего действия, чек на ней, чек на следующих улицах.
    """
    for _ in range(20):
        en = enrich(normalize(raw))
        pending = [r for r in en.report.illegal_actions if r.startswith(_OUT_OF_ACTIONS)]
        if not pending:
            return en
        label = pending[0].removeprefix(_OUT_OF_ACTIONS)
        last = raw.actions[-1].street
        candidates = [_fold(last, label)] + [
            _check(street, label) for street in _STREETS[_STREETS.index(last) :]
        ]
        for candidate in candidates:
            trial = raw.model_copy(update={"actions": [*raw.actions, candidate]})
            illegal = enrich(normalize(trial)).report.illegal_actions
            if all(r.startswith(_OUT_OF_ACTIONS) for r in illegal):
                raw = trial
                break
        else:
            raise AssertionError(f"руку не доиграть: {pending[0]}")
    raise AssertionError("руку не доиграть за 20 ходов")


def _points(en: EnrichedHand, street: Street) -> list[DecisionPoint]:
    return [dp for dp in en.report.decision_points if dp.street is street]


def _detail(en: EnrichedHand, street: Street, nth: int = -1) -> PostflopLineDetail:
    """Линия героя в `nth`-й его точке на улице (по умолчанию — последней)."""
    return postflop_line(_points(en, street)[nth], en)


def _tag(en: EnrichedHand, street: Street, nth: int = -1) -> tuple[ActionTag, int | None]:
    line = _detail(en, street, nth).line
    assert line is not None
    return line.action, line.barrel


# Префлоп: герой на кнопке открывает и получает колл; соперник на кнопке
# открывает, герой на BB коллирует; лимп кнопки-героя; лимп соперника.
_HERO_OPENS = [_raise(P, "Hero", 300), _call(P, "V", 200)]
_V_OPENS = [_raise(P, "V", 300), _call(P, "Hero", 200)]
_HERO_LIMPS = [_call(P, "Hero", 50), _check(P, "V")]
_V_LIMPS = [_call(P, "V", 50), _check(P, "Hero")]


def _ip(actions: list[RawAction], **kwargs) -> EnrichedHand:
    """Герой на кнопке, в позиции после флопа."""
    return _hand(actions, button="Hero", **kwargs)


def _oop(actions: list[RawAction], **kwargs) -> EnrichedHand:
    """Герой на BB, ходит первым после флопа."""
    return _hand(actions, button="V", **kwargs)


# --- гейт 1: эталонная рука §2 -----------------------------------------------------------


def _reference_hand(*, opponent_cards: tuple[str, str] = ("Tc", "Qc")) -> EnrichedHand:
    """Рука §2 спеки в синтетике: BB защищает 5♣3♣ против опена UTG+1.

    Флоп Q♥4♦2♦ чек-чек, тёрн 7♠ ставка 750 — колл, ривер Q♦ ставка 2127 — колл.
    Банк до тёрна 1335, до ривера 2835, итог 7089. `opponent_cards` — карты
    открывшего в раздаче и на вскрытии: линия их не читает (гейт 8).
    """
    opener = "P4"
    seats = [
        SeatInfo(seat=1, label="P1", stack=20_000),
        SeatInfo(seat=2, label="Hero", stack=9_275),
        SeatInfo(seat=3, label="P3", stack=20_000),
        SeatInfo(seat=4, label=opener, stack=25_000),
        *[SeatInfo(seat=n, label=f"P{n}", stack=20_000) for n in (5, 6, 7)],
    ]
    raw = RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="synthetic",
        hand_no="SYN-LINE-2",
        tournament_id="TSYN",
        tournament_name="synthetic",
        level=5,
        sb=125,
        bb=250,
        ante=30,
        timestamp=datetime(2026, 8, 20, 21, 30, tzinfo=UTC),
        table_name="syn",
        max_seats=7,
        button_seat=7,
        seats=seats,
        posts=[Post(label=s.label, kind=PostKind.ANTE, amount=30) for s in seats]
        + [
            Post(label="P1", kind=PostKind.SMALL_BLIND, amount=125),
            Post(label="Hero", kind=PostKind.BIG_BLIND, amount=250),
        ],
        dealt={"Hero": ["5c", "3c"], opener: list(opponent_cards)},
        actions=[
            _fold(P, "P3"),
            _raise(P, opener, 500),
            *[_fold(P, label) for label in ("P5", "P6", "P7", "P1")],
            _call(P, "Hero", 250),
            _check(F, "Hero"),
            _check(F, opener),
            _bet(T, "Hero", 750),
            _call(T, opener, 750),
            _bet(R, "Hero", 2_127),
            _call(R, opener, 2_127),
        ],
        boards={F: ["Qh", "4d", "2d"], T: ["7s"], R: ["Qd"]},
        showdowns=[
            ShowdownEntry(label="Hero", cards=["5c", "3c"]),
            ShowdownEntry(label=opener, cards=list(opponent_cards)),
        ],
        collected=[Collected(label=opener, amount=7_089)],
    )
    en = enrich(normalize(raw))
    assert en.verdict.status is ValidationStatus.PASS, en.verdict.reasons
    return en


def test_the_reference_hand_carries_the_data_the_owners_layout_prints():
    """Гейт 1 (ядро): каждая постфлоп-точка §2 несёт ровно данные макета.

    Дроби — из §9: 750/1335, 2127/2835, 8/47, 1 − 39·38/(47·46), 8/46, 750/2085,
    2127/4962, полублеф 0.1614, вскрытие 0/8/982 из 990.
    """
    points = analyze_hand(_reference_hand()).points
    preflop, flop, turn, river = points
    assert [p.street for p in points] == [P, F, T, R]

    assert POSTFLOP_LINE_DETAIL not in preflop.detail
    assert preflop.detail["unjudged"] == "нет чарта защиты BB против опена UTG+1 на 37 ББ"

    # Флоп: чек, рука и дро, строки «линия» нет.
    line = postflop_line_detail(flop)
    assert line is not None
    assert line.hand is not None
    assert (line.hand.category, line.hand.strength, line.hand.ranks) == (
        HandCategory.NO_PAIR,
        StrengthClass.WEAK,
        ["5"],
    )
    assert line.draw is not None
    assert line.draw.kinds == [DrawKind.OPEN_ENDED]
    assert line.draw.out_ranks == ["A", "6"]
    assert (len(line.draw.outs), line.draw.unseen) == (8, 47)
    assert line.draw.hit_next == 8 / 47
    assert line.draw.hit_by_river == pytest.approx(1 - 39 * 38 / (47 * 46), abs=1e-12)
    assert (line.backdoors, line.overcards, line.showdown) == ([], None, None)
    assert line.draw_missed is False
    assert (line.line, line.fold_threshold, line.draw_call) == (None, None, None)
    assert flop.detail["unjudged"] == POSTFLOP_CHECK_REASON == ""

    # Тёрн: проба · полублеф, 56% банка, пороги 750/2085 и 0.1614.
    line = postflop_line_detail(turn)
    assert line is not None and line.draw is not None and line.line is not None
    assert line.hand is not None and line.hand.category is HandCategory.NO_PAIR
    assert (len(line.draw.outs), line.draw.unseen, line.draw.hit_by_river) == (8, 46, None)
    assert line.draw.hit_next == 8 / 46
    assert line.line.action is ActionTag.PROBE
    assert line.line.barrel is None
    assert line.line.size_pct == 750 / 1335
    assert line.line.size_tag is SizeTag.STANDARD
    assert line.line.purpose is Purpose.SEMIBLUFF
    assert line.fold_threshold is not None
    assert line.fold_threshold.bluff == 750 / 2085
    q = Fraction(8, 46)
    ev = q * (1335 + 2 * 750) - 750
    assert line.fold_threshold.semibluff == pytest.approx(float(-ev / (1335 - ev)), abs=1e-12)
    assert line.fold_threshold.semibluff == pytest.approx(0.1614, abs=5e-5)
    assert line.fold_threshold.semibluff_free is False
    assert line.draw_call is None
    assert turn.detail["unjudged"] == POSTFLOP_FOLD_FREQUENCY_REASON

    # Ривер: пара дам на борде с кикером 5, дро не закрылось, вскрытие 0/8/982.
    line = postflop_line_detail(river)
    assert line is not None and line.hand is not None and line.line is not None
    assert (line.hand.category, line.hand.strength, line.hand.ranks) == (
        HandCategory.ON_BOARD,
        StrengthClass.WEAK,
        ["Q"],
    )
    assert (line.hand.plays, line.hand.kicker) == ("kicker", "5")
    assert line.draw is None
    assert line.draw_missed is True
    assert line.showdown is not None
    assert (line.showdown.wins, line.showdown.ties, line.showdown.losses) == (0, 8, 982)
    assert line.showdown.ties_with == ["53o", "53s"]
    assert line.line.action is ActionTag.REPEAT_BET
    assert line.line.size_pct == 2127 / 2835
    assert line.line.size_tag is SizeTag.BIG
    assert line.line.purpose is Purpose.BLUFF
    assert line.fold_threshold is not None
    assert line.fold_threshold.bluff == 2127 / 4962
    assert line.fold_threshold.semibluff is None
    assert river.detail["unjudged"] == POSTFLOP_FOLD_FREQUENCY_REASON


# --- гейт 4: тип действия, строки таблицы §4.3 -------------------------------------------


def test_a_flop_bet_by_the_preflop_aggressor_is_a_cbet():
    en = _ip([*_HERO_OPENS, _check(F, "V"), _bet(F, "Hero", 300), _call(F, "V", 300)])
    assert _tag(en, F) == (ActionTag.CBET, None)


def test_a_bet_in_a_chain_called_without_a_raise_is_a_barrel():
    en = _ip(
        [
            *_HERO_OPENS,
            _check(F, "V"),
            _bet(F, "Hero", 300),
            _call(F, "V", 300),
            _check(T, "V"),
            _bet(T, "Hero", 600),
            _call(T, "V", 600),
            _check(R, "V"),
            _bet(R, "Hero", 1_200),
        ]
    )
    assert _tag(en, F) == (ActionTag.CBET, None), "префлоп цепочку не начинает"
    assert _tag(en, T) == (ActionTag.BARREL, 2)
    assert _tag(en, R) == (ActionTag.BARREL, 3)


def test_a_river_bet_after_a_chain_started_on_the_turn_is_a_repeat_bet():
    en = _ip(
        [
            *_HERO_OPENS,
            _check(F, "V"),
            _check(F, "Hero"),
            _check(T, "V"),
            _bet(T, "Hero", 300),
            _call(T, "V", 300),
            _check(R, "V"),
            _bet(R, "Hero", 900),
        ]
    )
    assert _tag(en, R) == (ActionTag.REPEAT_BET, None)


def test_a_turn_bet_by_the_preflop_aggressor_after_flop_checks_is_a_delayed_cbet():
    en = _ip(
        [*_HERO_OPENS, _check(F, "V"), _check(F, "Hero"), _check(T, "V"), _bet(T, "Hero", 300)]
    )
    assert _tag(en, T) == (ActionTag.DELAYED_CBET, None)


def test_a_turn_probe_after_flop_checks_does_not_become_a_donk():
    """Префлоп-агрессор жив и на тёрне не ходил, но флоп прошёл чеками: проба."""
    en = _oop([*_V_OPENS, _check(F, "Hero"), _check(F, "V"), _bet(T, "Hero", 300)])
    assert _tag(en, T) == (ActionTag.PROBE, None)


def test_a_river_probe_after_turn_checks():
    en = _oop(
        [
            *_V_OPENS,
            _check(F, "Hero"),
            _bet(F, "V", 300),
            _call(F, "Hero", 300),
            _check(T, "Hero"),
            _check(T, "V"),
            _bet(R, "Hero", 600),
        ]
    )
    assert _tag(en, R) == (ActionTag.PROBE, None)


def test_a_flop_lead_into_the_preflop_aggressor_is_a_donk():
    en = _oop([*_V_OPENS, _bet(F, "Hero", 300)])
    assert _tag(en, F) == (ActionTag.DONK, None)


def test_a_turn_lead_into_the_flop_aggressor_is_a_donk():
    en = _oop(
        [
            *_V_OPENS,
            _check(F, "Hero"),
            _bet(F, "V", 300),
            _call(F, "Hero", 300),
            _bet(T, "Hero", 600),
        ]
    )
    assert _tag(en, T) == (ActionTag.DONK, None)


def test_a_bet_after_the_heros_check_raise_on_the_previous_street_is_not_a_donk():
    """Агрессор флопа — сам герой: донка нет, и рейз цепочку барреля не начинает."""
    en = _oop(
        [
            *_V_OPENS,
            _check(F, "Hero"),
            _bet(F, "V", 300),
            _raise(F, "Hero", 1_000),
            _call(F, "V", 700),
            _bet(T, "Hero", 1_500),
        ]
    )
    assert _tag(en, F) == (ActionTag.CHECK_RAISE, None)
    assert _tag(en, T) == (ActionTag.BET, None)


def test_a_folded_aggressor_does_not_make_a_donk():
    """Сбросивший агрессор донка не порождает: ставку проверяют строки 7–8 (§4.3).

    В легальной раздаче последний агрессор улицы сбросить может только ходом на
    следующей улице — а сходивший уже не «ещё не ходил». Чтобы проверить само
    условие «он жив», пас открывшего вписан в каноническую руку поверх прогона
    движка: в конец префлопа, до ставки героя на флопе.
    """
    seats = (("Hero", _STACK), ("V2", _STACK), ("V1", _STACK))
    en = _hand(
        [_raise(P, "V1", 300), _call(P, "Hero", 250), _call(P, "V2", 200), _bet(F, "Hero", 300)],
        seats=seats,
        button="V1",
    )
    assert _tag(en, F) == (ActionTag.DONK, None), "без паса — донк в открывшего"
    at = next(i for i, a in enumerate(en.hand.actions) if a.street is F)
    folded = en.hand.actions[at - 1].model_copy(
        update={"label": "V1", "kind": ActionKind.FOLD, "committed_after": 300}
    )
    actions = [*en.hand.actions[:at], folded, *en.hand.actions[at:]]
    en = en.model_copy(update={"hand": en.hand.model_copy(update={"actions": actions})})
    assert _tag(en, F) == (ActionTag.BET, None)


def test_a_multiway_lead_after_another_players_check_into_the_aggressor_behind_is_a_donk():
    """A чекнул, герой ставит, открывший позади ещё не ходил: донк, а не
    «ставка после чека» — строка 6 таблицы §4.3 раньше строки 7."""
    seats = (("A", _STACK), ("Hero", _STACK), ("V", _STACK))
    en = _hand(
        [
            _raise(P, "V", 300),
            _call(P, "A", 250),
            _call(P, "Hero", 200),
            _check(F, "A"),
            _bet(F, "Hero", 300),
        ],
        seats=seats,
        button="V",
    )
    assert _tag(en, F) == (ActionTag.DONK, None)


def test_a_turn_bet_in_position_after_flop_checks_is_not_a_probe():
    """Проба — только вне позиции: в позиции та же ставка — «ставка после чека»."""
    en = _ip(
        [
            _call(P, "Hero", 50),
            _raise(P, "V", 300),
            _call(P, "Hero", 200),
            _check(F, "V"),
            _check(F, "Hero"),
            _check(T, "V"),
            _bet(T, "Hero", 300),
        ]
    )
    assert _tag(en, T) == (ActionTag.BET_AFTER_CHECK, None)


def test_a_bet_after_checks_of_the_players_before_the_hero():
    en = _ip([*_HERO_LIMPS, _check(F, "V"), _bet(F, "Hero", 100)])
    assert _tag(en, F) == (ActionTag.BET_AFTER_CHECK, None)


def test_a_lead_with_no_aggressor_anywhere_is_a_plain_bet():
    en = _oop([*_V_LIMPS, _bet(F, "Hero", 100)])
    assert _tag(en, F) == (ActionTag.BET, None)


# --- гейт 4: ответы и повышения ------------------------------------------------------------


def test_answers_after_a_check_are_check_call_check_fold_check_raise():
    called = _oop([*_V_OPENS, _check(F, "Hero"), _bet(F, "V", 300), _call(F, "Hero", 300)])
    folded = _oop([*_V_OPENS, _check(F, "Hero"), _bet(F, "V", 300), _fold(F, "Hero")])
    assert _tag(called, F) == (ActionTag.CHECK_CALL, None)
    assert _tag(folded, F) == (ActionTag.CHECK_FOLD, None)


def test_raises_are_counted_within_the_street():
    """Первое повышение — рейз, второе — 3-бет, третье и дальше — ре-рейз."""
    three_bet = _oop(
        [*_V_OPENS, _bet(F, "Hero", 300), _raise(F, "V", 900), _raise(F, "Hero", 2_400)]
    )
    assert _tag(three_bet, F) == (ActionTag.THREE_BET, None)
    reraise = _ip(
        [
            *_HERO_LIMPS,
            _bet(F, "V", 100),
            _raise(F, "Hero", 300),
            _raise(F, "V", 900),
            _raise(F, "Hero", 2_400),
        ]
    )
    assert _tag(reraise, F, nth=0) == (ActionTag.RAISE, None)
    assert _tag(reraise, F, nth=1) == (ActionTag.RERAISE, None)


def test_a_call_after_the_heros_own_bet_is_a_call():
    en = _oop([*_V_OPENS, _bet(F, "Hero", 300), _raise(F, "V", 900), _call(F, "Hero", 600)])
    assert _tag(en, F) == (ActionTag.CALL, None)
    line = _detail(en, F).line
    assert line is not None and (line.size_pct, line.size_tag) == (None, None)


def test_a_call_in_position_followed_by_a_bet_into_a_check_is_a_float():
    floated = _ip(
        [
            *_HERO_OPENS,
            _bet(F, "V", 300),
            _call(F, "Hero", 300),
            _check(T, "V"),
            _bet(T, "Hero", 600),
        ]
    )
    assert _tag(floated, F) == (ActionTag.FLOAT, None)
    plain = _ip(
        [*_HERO_OPENS, _bet(F, "V", 300), _call(F, "Hero", 300), _check(T, "V"), _check(T, "Hero")]
    )
    assert _tag(plain, F) == (ActionTag.CALL, None)


def test_a_call_out_of_position_is_never_a_float():
    en = _oop(
        [
            *_V_OPENS,
            _bet(F, "Hero", 300),
            _raise(F, "V", 900),
            _call(F, "Hero", 600),
            _check(T, "Hero"),
            _check(T, "V"),
        ]
    )
    assert _tag(en, F) == (ActionTag.CALL, None)


def test_an_all_in_player_behind_does_not_take_the_position_away():
    """Позиция — среди тех, кто ещё может ходить: олл-ин позади героя не ходит.

    Порядок хода после флопа: A (SB), Hero (BB), B (кнопка). B в олл-ине с
    префлопа — колл героя в позиции и флоат; B с фишками — просто колл.
    """

    def hand(b_stack: int) -> EnrichedHand:
        b_moves = [] if b_stack == 1_000 else [_call(F, "B", 1_000)]
        return _hand(
            [
                _raise(P, "B", 1_000),
                _call(P, "A", 950),
                _call(P, "Hero", 900),
                _bet(F, "A", 1_000),
                _call(F, "Hero", 1_000),
                *b_moves,
                _check(T, "A"),
                _bet(T, "Hero", 2_000),
            ],
            seats=(("A", _STACK), ("Hero", _STACK), ("B", b_stack)),
            button="B",
        )

    assert _tag(hand(1_000), F) == (ActionTag.FLOAT, None)
    assert _tag(hand(_STACK), F) == (ActionTag.CALL, None)


def test_a_check_has_no_line_and_a_fold_has_no_purpose():
    en = _oop([*_V_OPENS, _check(F, "Hero"), _bet(F, "V", 300), _fold(F, "Hero")])
    check, fold = (_detail(en, F, nth) for nth in (0, 1))
    assert check.line is None
    assert fold.line is not None
    assert (fold.line.purpose, fold.line.size_pct, fold.line.size_tag) == (None, None, None)
    assert (fold.fold_threshold, fold.draw_call) == (None, None)


# --- гейт 5: баррель §4.4 ------------------------------------------------------------------


def test_a_donk_call_and_a_bet_is_a_second_barrel():
    en = _oop([*_V_OPENS, _bet(F, "Hero", 300), _call(F, "V", 300), _bet(T, "Hero", 600)])
    assert _tag(en, F) == (ActionTag.DONK, None)
    assert _tag(en, T) == (ActionTag.BARREL, 2)


def test_a_checked_through_street_breaks_the_chain():
    en = _ip(
        [
            *_HERO_OPENS,
            _check(F, "V"),
            _bet(F, "Hero", 300),
            _call(F, "V", 300),
            _check(T, "V"),
            _check(T, "Hero"),
            _check(R, "V"),
            _bet(R, "Hero", 600),
        ]
    )
    assert _tag(en, R) == (ActionTag.BET_AFTER_CHECK, None)


def test_a_raise_on_the_previous_street_breaks_the_chain():
    en = _oop(
        [
            *_V_LIMPS[:1],
            _raise(P, "Hero", 300),
            _call(P, "V", 200),
            _bet(F, "Hero", 300),
            _raise(F, "V", 900),
            _call(F, "Hero", 600),
            _bet(T, "Hero", 1_000),
        ]
    )
    assert _tag(en, F, nth=0) == (ActionTag.CBET, None)
    assert _tag(en, T) == (ActionTag.DONK, None), "рейз флопа: ставка в агрессора, не баррель"


def test_a_chain_of_another_player_is_not_the_heros_barrel():
    en = _oop(
        [
            *_V_OPENS,
            _check(F, "Hero"),
            _bet(F, "V", 300),
            _call(F, "Hero", 300),
            _bet(T, "Hero", 600),
        ]
    )
    assert _tag(en, T)[0] is not ActionTag.BARREL


def test_a_chain_started_on_the_turn_is_not_a_barrel_on_the_river():
    en = _oop(
        [
            *_V_OPENS,
            _check(F, "Hero"),
            _check(F, "V"),
            _bet(T, "Hero", 300),
            _call(T, "V", 300),
            _bet(R, "Hero", 900),
        ]
    )
    assert _tag(en, R) == (ActionTag.REPEAT_BET, None)


# --- гейт 6: размер ----------------------------------------------------------------------

# Банк перед флопом — 10 000: рейз до 5 000 и колл.
_DEEP = 100_000
_BIG_POT_PREFLOP = [_raise(P, "V", 5_000), _call(P, "Hero", 4_900)]


@pytest.mark.parametrize(
    ("bet", "tag"),
    [
        (3_299, SizeTag.BLOCK),
        (3_300, SizeTag.STANDARD),
        (7_499, SizeTag.STANDARD),
        (7_500, SizeTag.BIG),
        (9_999, SizeTag.BIG),
        (10_000, SizeTag.OVERBET),
    ],
)
def test_the_size_tag_is_cut_by_chips_on_half_open_intervals(bet, tag):
    """Границы 33/100, 75/100, 100/100 — по фишкам: 32.99% — ещё блок."""
    en = _oop(
        [*_BIG_POT_PREFLOP, _bet(F, "Hero", bet)],
        seats=(("Hero", _DEEP), ("V", _DEEP)),
    )
    line = _detail(en, F).line
    assert line is not None
    assert (line.size_pct, line.size_tag) == (bet / 10_000, tag)


def test_the_raise_size_is_the_raise_over_the_call_to_the_pot_with_the_call():
    """Рейз до 6 000 против ставки 2 000 в банк 10 000: 4 000 / (12 000 + 2 000)."""
    en = _oop(
        [*_BIG_POT_PREFLOP, _check(F, "Hero"), _bet(F, "V", 2_000), _raise(F, "Hero", 6_000)],
        seats=(("Hero", _DEEP), ("V", _DEEP)),
    )
    dp = _points(en, F)[-1]
    assert (dp.pot_before, dp.to_call) == (12_000, 2_000)
    line = _detail(en, F).line
    assert line is not None
    assert line.size_pct == 4_000 / 14_000
    assert line.size_tag is SizeTag.BLOCK


# --- гейт 7: назначение и пороги ----------------------------------------------------------


def test_value_and_medium_hands_carry_no_fold_threshold():
    value = _oop([*_V_OPENS, _bet(F, "Hero", 300)], hero_cards=("Kc", "Ks"))
    medium = _oop([*_V_OPENS, _bet(F, "Hero", 300)], hero_cards=("Kc", "4s"))
    for en, purpose in ((value, Purpose.VALUE), (medium, Purpose.MEDIUM_HAND)):
        detail = _detail(en, F)
        assert detail.line is not None and detail.line.purpose is purpose
        assert detail.fold_threshold is None


def test_a_raise_threshold_takes_what_the_hero_adds_against_the_pot_with_the_bet():
    """Блеф-рейз до 6 000 против 2 000 в банк 10 000: B = 6 000, P = 12 000."""
    en = _oop(
        [*_BIG_POT_PREFLOP, _check(F, "Hero"), _bet(F, "V", 2_000), _raise(F, "Hero", 6_000)],
        seats=(("Hero", _DEEP), ("V", _DEEP)),
    )
    detail = _detail(en, F)
    assert detail.line is not None and detail.line.purpose is Purpose.BLUFF
    assert detail.fold_threshold is not None
    assert detail.fold_threshold.bluff == 6_000 / 18_000
    assert detail.fold_threshold.semibluff is None


def test_a_semibluff_raise_counts_the_opponents_call_over_his_bet():
    """Полублеф-рейз до 12 000 против 2 000 в банк 10 000:
    E = q·(P + B + (B − колл)) − B, q — к риверу на флопе."""
    board = {F: ["Kh", "9d", "8s"], T: ["2c"], R: ["2d"]}
    en = _oop(
        [*_BIG_POT_PREFLOP, _check(F, "Hero"), _bet(F, "V", 2_000), _raise(F, "Hero", 12_000)],
        seats=(("Hero", _DEEP), ("V", _DEEP)),
        hero_cards=("Tc", "7c"),
        boards=board,
    )
    detail = _detail(en, F)
    assert detail.draw is not None and len(detail.draw.outs) == 8
    assert detail.line is not None and detail.line.purpose is Purpose.SEMIBLUFF
    q = 1 - Fraction(comb(39, 2), comb(47, 2))
    pot, bet = 12_000, 12_000
    ev = q * (pot + bet + (bet - 2_000)) - bet
    assert detail.fold_threshold is not None
    assert detail.fold_threshold.semibluff == pytest.approx(float(-ev / (pot - ev)), abs=1e-12)
    assert detail.fold_threshold.semibluff_free is False
    assert detail.fold_threshold.bluff == 12_000 / 24_000


def test_a_semibluff_that_pays_without_folds_is_flagged_and_needs_none():
    """Флеш-дро с двусторонним (15 аутов) и ставка 1 000 в 10 000: E ≥ 0."""
    board = {F: ["Kh", "9c", "8c"], T: ["2d"], R: ["2s"]}
    en = _oop(
        [*_BIG_POT_PREFLOP, _bet(F, "Hero", 1_000)],
        seats=(("Hero", _DEEP), ("V", _DEEP)),
        hero_cards=("Tc", "7c"),
        boards=board,
    )
    detail = _detail(en, F)
    assert detail.draw is not None and len(detail.draw.outs) == 15
    assert detail.fold_threshold is not None
    assert (detail.fold_threshold.semibluff, detail.fold_threshold.semibluff_free) == (0.0, True)
    assert detail.fold_threshold.bluff == 1_000 / 11_000


# Флеш-дро с двусторонним, 15 аутов: T♣7♣ на K♥9♣8♣, тёрн 2♦.
_COMBO_BOARD = {F: ["Kh", "9c", "8c"], T: ["2d"], R: ["2s"]}


def test_the_thresholds_cut_the_bet_by_the_opponents_stack():
    """Шов 49 700 в банк 600 против стека 4 700: соперник уравняет только 4 700.

    Пороги считаются от B = 4 700: блеф 4 700 / 5 300 = 88.7%, полублеф с 15
    аутами на тёрне 70.6%. От полной ставки вышло бы 98.8% и 96.6%. Размер в
    «сыграно» описывает сыгранное и остаётся от полной ставки.
    """
    en = _oop(
        [
            *_V_OPENS,
            _check(F, "Hero"),
            _check(F, "V"),
            _bet(T, "Hero", 49_700),
        ],
        seats=(("Hero", 50_000), ("V", 5_000)),
        hero_cards=("Tc", "7c"),
        boards=_COMBO_BOARD,
    )
    dp = _points(en, T)[-1]
    assert (dp.pot_before, dp.to_call, dp.eff_stack) == (600, 0, 4_700)
    detail = _detail(en, T)
    assert detail.draw is not None and len(detail.draw.outs) == 15
    assert detail.line is not None and detail.line.purpose is Purpose.SEMIBLUFF
    assert detail.line.size_pct == 49_700 / 600
    threshold = detail.fold_threshold
    assert threshold is not None
    assert threshold.bluff == 4_700 / 5_300
    assert round(threshold.bluff, 3) == 0.887
    ev = Fraction(15, 46) * (600 + 4_700 + 4_700) - 4_700
    assert threshold.semibluff == pytest.approx(float(-ev / (600 - ev)), abs=1e-12)
    assert threshold.semibluff is not None and round(threshold.semibluff, 3) == 0.706


def test_a_raise_threshold_cuts_the_raise_and_the_call_by_the_opponents_stack():
    """Рейз в олл-ин 49 700 на тёрне против ставки 1 000 при стеке соперника
    4 700: B = 4 700, C = 4 700 − 1 000, P = 1 600 (банк со ставкой)."""
    en = _oop(
        [
            *_V_OPENS,
            _check(F, "Hero"),
            _check(F, "V"),
            _check(T, "Hero"),
            _bet(T, "V", 1_000),
            _raise(T, "Hero", 49_700),
        ],
        seats=(("Hero", 50_000), ("V", 5_000)),
        hero_cards=("Tc", "7c"),
        boards=_COMBO_BOARD,
    )
    dp = _points(en, T)[-1]
    assert (dp.pot_before, dp.to_call, dp.eff_stack) == (1_600, 1_000, 4_700)
    detail = _detail(en, T)
    assert detail.line is not None and detail.line.purpose is Purpose.SEMIBLUFF
    assert detail.line.size_pct == (49_700 - 1_000) / (1_600 + 1_000)
    threshold = detail.fold_threshold
    assert threshold is not None
    assert threshold.bluff == 4_700 / 6_300
    ev = Fraction(15, 46) * (1_600 + 4_700 + 3_700) - 4_700
    assert threshold.semibluff == pytest.approx(float(-ev / (1_600 - ev)), abs=1e-12)


def test_a_semibluff_with_zero_expectation_pays_without_folds():
    """Ровно E = 0: 15/46 · (1 600 + 2 · 1 500) − 1 500 = 0 — «окупается без фолдов»."""
    en = _oop(
        [
            _raise(P, "V", 800),
            _call(P, "Hero", 700),
            _check(F, "Hero"),
            _check(F, "V"),
            _bet(T, "Hero", 1_500),
        ],
        hero_cards=("Tc", "7c"),
        boards=_COMBO_BOARD,
    )
    assert _points(en, T)[-1].pot_before == 1_600
    detail = _detail(en, T)
    assert detail.line is not None and detail.line.purpose is Purpose.SEMIBLUFF
    assert Fraction(15, 46) * (1_600 + 2 * 1_500) - 1_500 == 0
    assert detail.fold_threshold is not None
    assert (detail.fold_threshold.semibluff, detail.fold_threshold.semibluff_free) == (0.0, True)


def test_the_semibluff_threshold_is_heads_up_only():
    board = {F: ["Kh", "9d", "8s"], T: ["2c"], R: ["2d"]}
    seats = (("Hero", _STACK), ("V2", _STACK), ("V1", _STACK))
    en = _hand(
        [_raise(P, "V1", 300), _call(P, "Hero", 250), _call(P, "V2", 200), _bet(F, "Hero", 300)],
        seats=seats,
        button="V1",
        hero_cards=("Tc", "7c"),
        boards=board,
    )
    detail = _detail(en, F)
    assert detail.line is not None and detail.line.purpose is Purpose.SEMIBLUFF
    assert detail.fold_threshold is not None
    assert detail.fold_threshold.bluff == 300 / 1_200
    assert detail.fold_threshold.semibluff is None


def test_backdoors_and_overcards_change_neither_the_purpose_nor_the_threshold():
    """A♣K♣ на 9♣7♦2♠: две оверкарты и бэкдор флеша, но дро нет — блеф."""
    board = {F: ["9c", "7d", "2s"], T: ["3h"], R: ["4h"]}
    en = _oop([*_V_OPENS, _bet(F, "Hero", 300)], hero_cards=("Ac", "Kc"), boards=board)
    detail = _detail(en, F)
    assert detail.backdoors and detail.overcards is not None
    assert detail.draw is None
    assert detail.line is not None and detail.line.purpose is Purpose.BLUFF
    assert detail.fold_threshold is not None
    assert (detail.fold_threshold.bluff, detail.fold_threshold.semibluff) == (300 / 900, None)


def test_a_draw_call_by_pot_odds_compares_the_next_card():
    """Колл 300 в банк 900 (нужно 25%) с 15 аутами на тёрне: 15/46 ≥ 25%."""
    board = {F: ["Kh", "9c", "8c"], T: ["2d"], R: ["2s"]}
    en = _oop(
        [
            *_V_OPENS,
            _check(F, "Hero"),
            _check(F, "V"),
            _check(T, "Hero"),
            _bet(T, "V", 300),
            _call(T, "Hero", 300),
        ],
        hero_cards=("Tc", "7c"),
        boards=board,
    )
    detail = _detail(en, T)
    assert detail.line is not None and detail.line.purpose is Purpose.CALL_WITH_DRAW
    assert detail.draw_call is not None
    assert detail.draw_call.required_equity == 300 / 1_200
    assert detail.draw_call.hit == 15 / 46
    assert detail.draw_call.by_pot_odds is True
    assert (detail.draw_call.implied_needed_chips, detail.draw_call.beyond_stack) == (None, False)


def _gutshot_call(stack: int) -> EnrichedHand:
    """Гатшот (4 аута на семёрку) у 6♣5♣ на K♥9♦8♠; колл ставки в банк на флопе."""
    return _oop(
        [*_BIG_POT_PREFLOP, _check(F, "Hero"), _bet(F, "V", 10_000), _call(F, "Hero", 10_000)],
        seats=(("Hero", stack), ("V", stack)),
        hero_cards=("6c", "5c"),
        boards={F: ["Kh", "9d", "8s"], T: ["2c"], R: ["2d"]},
    )


def test_a_draw_call_short_of_pot_odds_names_the_chips_to_win_later():
    """X = C / q − (P + C), вверх до фишки; глубже X — добрать можно."""
    detail = _detail(_gutshot_call(500_000), F)
    assert detail.draw is not None and detail.draw.kinds == [DrawKind.GUTSHOT]
    assert detail.draw_call is not None
    q = Fraction(4, 47)
    assert detail.draw_call.hit == 4 / 47
    assert detail.draw_call.by_pot_odds is False
    assert detail.draw_call.implied_needed_chips == ceil(10_000 / q - (20_000 + 10_000))
    assert detail.draw_call.beyond_stack is False


def test_an_implied_amount_beyond_the_stacks_is_a_line_and_the_point_stays_unjudged():
    """X ≈ 87 500 больше 85 000, что останется у обоих после колла."""
    en = _gutshot_call(_DEEP)
    dp = _points(en, F)[-1]
    detail = postflop_line(dp, en)
    assert detail.draw_call is not None
    assert detail.draw_call.implied_needed_chips is not None
    assert detail.draw_call.implied_needed_chips > _DEEP - 5_000 - 10_000
    assert detail.draw_call.beyond_stack is True
    point = verdict_for(dp, en)
    assert point.best_action == "" and not is_judged(point)


@pytest.mark.parametrize(
    ("hero_stack", "villain_stack"), [(20_000, _DEEP), (_DEEP, 20_000)], ids=["hero", "villain"]
)
def test_a_call_that_ends_the_betting_on_the_flop_takes_the_chance_to_the_river(
    hero_stack, villain_stack
):
    """Колл всем стеком героя или против олл-ина соперника: торговли дальше нет,
    шанс — двумя картами, а добирать позже не из чего."""
    en = _oop(
        [*_BIG_POT_PREFLOP, _check(F, "Hero"), _bet(F, "V", 15_000), _call(F, "Hero", 15_000)],
        seats=(("Hero", hero_stack), ("V", villain_stack)),
        hero_cards=("6c", "5c"),
        boards={F: ["Kh", "9d", "8s"], T: ["2c"], R: ["2d"]},
    )
    detail = _detail(en, F)
    assert detail.draw is not None and detail.draw.hit_by_river is not None
    assert detail.draw_call is not None
    assert detail.draw_call.hit == detail.draw.hit_by_river
    assert detail.draw_call.beyond_stack is True


def _gutshot_check_call(bet: int, stack: int) -> EnrichedHand:
    """Гатшот 6♣5♣ на K♥9♦8♠: чек-колл ставки `bet` в банк 600."""
    return _oop(
        [*_V_OPENS, _check(F, "Hero"), _bet(F, "V", bet), _call(F, "Hero", bet)],
        seats=(("Hero", stack), ("V", stack)),
        hero_cards=("6c", "5c"),
        boards={F: ["Kh", "9d", "8s"], T: ["2c"], R: ["2d"]},
    )


def test_a_fractional_implied_amount_is_rounded_up_to_a_chip():
    """X = 303 · 47/4 − (903 + 303) = 2 354.25 — печатается 2 355, не 2 354."""
    detail = _detail(_gutshot_check_call(303, _DEEP), F)
    assert detail.draw_call is not None
    assert Fraction(303 * 47, 4) - (903 + 303) == Fraction(9_417, 4)
    assert detail.draw_call.implied_needed_chips == 2_355


def test_the_implied_amount_is_an_exact_fraction_and_not_a_float():
    """Флеш-дро с гатшотом, 12 аутов: X = 372 · 47/12 − 1 344 = 113 ровно.

    Деление на шанс в двоичной дроби даёт 113.000…01, и округление вверх
    подняло бы сумму на фишку.
    """
    en = _oop(
        [*_V_OPENS, _check(F, "Hero"), _bet(F, "V", 372), _call(F, "Hero", 372)],
        seats=(("Hero", _DEEP), ("V", _DEEP)),
        hero_cards=("Jc", "Tc"),
        boards={F: ["8c", "7d", "2c"], T: ["3h"], R: ["4h"]},
    )
    detail = _detail(en, F)
    assert detail.draw is not None and len(detail.draw.outs) == 12
    assert detail.draw_call is not None
    assert Fraction(372) / Fraction(12, 47) - 1_344 == 113
    assert ceil(372 / (12 / 47) - 1_344) == 114, "двоичная дробь ушла бы на фишку вверх"
    assert detail.draw_call.implied_needed_chips == 113


def test_an_implied_amount_equal_to_the_stack_left_can_still_be_won():
    """X = 400 · 47/4 − 1 400 = 3 300 — ровно столько у обоих после колла."""
    en = _gutshot_check_call(400, 4_000)
    dp = _points(en, F)[-1]
    assert dp.eff_stack - dp.action.committed_after == 3_300
    detail = postflop_line(dp, en)
    assert detail.draw_call is not None
    assert detail.draw_call.implied_needed_chips == 3_300
    assert detail.draw_call.beyond_stack is False


def test_the_stack_left_after_a_call_takes_the_heros_earlier_chips_on_the_street():
    """Донк 300, рейз до 800, колл 500 при 4 300 у обоих на флопе: после колла
    остаётся 4 300 − 800 = 3 500, а не 4 300 − 500 = 3 800. X = 3 675 — уже
    больше остатка."""
    en = _oop(
        [*_V_OPENS, _bet(F, "Hero", 300), _raise(F, "V", 800), _call(F, "Hero", 500)],
        seats=(("Hero", 4_600), ("V", 4_600)),
        hero_cards=("6c", "5c"),
        boards={F: ["Kh", "9d", "8s"], T: ["2c"], R: ["2d"]},
    )
    dp = _points(en, F)[-1]
    assert (dp.eff_stack, dp.pot_before, dp.to_call) == (4_300, 1_700, 500)
    detail = postflop_line(dp, en)
    assert detail.draw_call is not None
    assert detail.draw_call.implied_needed_chips == 3_675
    assert detail.draw_call.beyond_stack is True


def test_a_call_against_an_overbet_is_priced_by_what_the_hero_can_call():
    """Ставка 40 000 против остатка героя 15 000: банк до хода урезан движком до
    25 000, доплата — 15 000; колл в олл-ин берёт шанс к риверу."""
    en = _oop(
        [*_BIG_POT_PREFLOP, _check(F, "Hero"), _bet(F, "V", 40_000), _call(F, "Hero", 15_000)],
        seats=(("Hero", 20_000), ("V", _DEEP)),
        hero_cards=("6c", "5c"),
        boards={F: ["Kh", "9d", "8s"], T: ["2c"], R: ["2d"]},
    )
    dp = _points(en, F)[-1]
    assert (dp.pot_before, dp.to_call, dp.eff_stack) == (25_000, 15_000, 15_000)
    detail = postflop_line(dp, en)
    assert detail.draw is not None and detail.draw_call is not None
    assert detail.draw_call.required_equity == 15_000 / 40_000
    assert detail.draw_call.hit == detail.draw.hit_by_river
    q = 1 - Fraction(comb(43, 2), comb(47, 2))
    assert detail.draw_call.implied_needed_chips == ceil(15_000 / q - 40_000)
    assert detail.draw_call.beyond_stack is True


def test_a_proven_river_fold_turns_a_strong_call_into_a_bluff_catch():
    en = _oop(
        [
            *_V_OPENS,
            _check(F, "Hero"),
            _check(F, "V"),
            _check(T, "Hero"),
            _check(T, "V"),
            _check(R, "Hero"),
            _bet(R, "V", 300),
            _call(R, "Hero", 300),
        ],
        hero_cards=("Kc", "Ks"),
    )
    dp = _points(en, R)[-1]
    strong, caught = (postflop_line(dp, en, fold_proven=flag).line for flag in (False, True))
    assert strong is not None and strong.purpose is Purpose.CALL_STRONG
    assert caught is not None and caught.purpose is Purpose.BLUFF_CATCH


def test_unknown_hero_cards_leave_the_cards_empty_and_the_line_without_purpose():
    en = _oop([*_V_OPENS, _bet(F, "Hero", 300)])
    blind = en.model_copy(update={"hand": en.hand.model_copy(update={"dealt": {}})})
    detail = _detail(blind, F)
    assert (detail.hand, detail.draw, detail.overcards, detail.showdown) == (None, None, None, None)
    assert (detail.backdoors, detail.draw_missed) == ([], False)
    assert detail.line is not None
    assert (detail.line.action, detail.line.purpose) == (ActionTag.DONK, None)
    assert detail.fold_threshold is None


def test_a_preflop_point_has_no_postflop_line():
    en = _oop([*_V_OPENS, _bet(F, "Hero", 300)])
    with pytest.raises(ValueError, match="постфлоп-линия"):
        postflop_line(_points(en, P)[0], en)


# --- гейт 8: граница — карты соперника и вскрытие ----------------------------------------


def test_opponent_cards_and_the_showdown_do_not_move_the_line():
    def lines(en: EnrichedHand) -> list[object]:
        return [p.detail.get(POSTFLOP_LINE_DETAIL) for p in analyze_hand(en).points]

    assert lines(_reference_hand()) == lines(_reference_hand(opponent_cards=("Ac", "Kc")))


# --- гейт 13 и §4.9: судимость не тронута, причины ---------------------------------------


def _hands_for_the_sum() -> list[EnrichedHand]:
    return [
        _reference_hand(),
        _gutshot_call(_DEEP),
        _oop(
            [
                *_V_OPENS,
                _check(F, "Hero"),
                _check(F, "V"),
                _check(T, "Hero"),
                _check(T, "V"),
                _check(R, "Hero"),
                _bet(R, "V", 300),
                _call(R, "Hero", 300),
            ],
            hero_cards=("Kc", "Ks"),
        ),
    ]


def test_the_line_changes_neither_the_sum_nor_the_ranking():
    """Ключ линии — единственное, чем постфлоп-точка `verdict_for` отличается
    от разбора улицы; сумма потерь и ранжирование без него те же."""
    for en in _hands_for_the_sum():
        result = analyze_hand(en)
        stripped = [
            p.model_copy(
                update={"detail": {k: v for k, v in p.detail.items() if k != POSTFLOP_LINE_DETAIL}}
            )
            for p in result.points
        ]
        assert rank_points(stripped) == result.ranked
        assert total_ev_loss_bb(stripped) == result.total_ev_loss_bb
        postflop = [dp for dp in en.report.decision_points if dp.street is not P]
        for dp in postflop:
            street_point = river_verdict(dp, en) or turn_flop_verdict(dp, en)
            assert street_point is not None
            point = verdict_for(dp, en)
            assert POSTFLOP_LINE_DETAIL in point.detail
            without = {k: v for k, v in point.detail.items() if k != POSTFLOP_LINE_DETAIL}
            assert point.model_copy(update={"detail": without}) == street_point
            assert is_judged(point) == is_judged(street_point)


def test_postflop_reasons_name_what_a_verdict_lacks():
    """§4.9: ставка и рейз — частота фолдов, чек — без причины, колл — как был."""
    en = _oop(
        [
            *_V_OPENS,
            _check(F, "Hero"),
            _bet(F, "V", 300),
            _raise(F, "Hero", 1_000),
            _call(F, "V", 700),
            _bet(T, "Hero", 1_000),
            _raise(T, "V", 3_000),
            _call(T, "Hero", 2_000),
        ]
    )
    reasons = [p.detail["unjudged"] for p in analyze_hand(en).points if p.street in (F, T)]
    assert reasons == [
        POSTFLOP_CHECK_REASON,
        POSTFLOP_FOLD_FREQUENCY_REASON,
        POSTFLOP_FOLD_FREQUENCY_REASON,
        "требование к ставящему диапазону посчитано, лучшего действия нет",
    ]


def test_a_deep_answer_to_an_open_names_the_missing_defence_chart():
    """Ответ на опен глубже пуш-фолда: чарта защиты нет; глубина — целым числом."""
    point = analyze_hand(_oop([*_V_OPENS, _check(F, "Hero"), _check(F, "V")])).points[0]
    assert point.street is P
    assert point.detail["unjudged"] == "нет чарта защиты BB против опена BTN на 100 ББ"


@pytest.mark.parametrize(("stack", "depth"), [(3_650, 37), (3_649, 36), (3_750, 38)])
def test_the_open_depth_in_the_reason_rounds_a_half_up(stack, depth):
    """36.5 ББ — 37, а не банковское 36."""
    en = _oop([*_V_OPENS, _check(F, "Hero"), _check(F, "V")], seats=(("Hero", stack), ("V", _STACK)))
    point = analyze_hand(en).points[0]
    assert point.detail["unjudged"] == f"нет чарта защиты BB против опена BTN на {depth} ББ"


def test_a_deep_answer_off_the_blinds_names_a_missing_response_chart():
    """«Защита» — у блайндов; кнопка на опен UTG отвечает."""
    seats = (("A", _STACK), ("B", _STACK), ("V", _STACK), ("Hero", _STACK))
    en = _hand(
        [
            _raise(P, "V", 300),
            _call(P, "Hero", 300),
            _fold(P, "A"),
            _fold(P, "B"),
            _check(F, "V"),
            _check(F, "Hero"),
        ],
        seats=seats,
        button="Hero",
    )
    point = analyze_hand(en).points[0]
    assert point.street is P
    assert point.detail["unjudged"] == "нет чарта ответа BTN на опен UTG на 100 ББ"


def _raises_on_every_street() -> EnrichedHand:
    """Герой чек-рейзит флоп, тёрн и ривер; соперник коллирует."""
    return _oop(
        [
            *_V_OPENS,
            _check(F, "Hero"),
            _bet(F, "V", 300),
            _raise(F, "Hero", 1_000),
            _call(F, "V", 700),
            _check(T, "Hero"),
            _bet(T, "V", 500),
            _raise(T, "Hero", 1_500),
            _call(T, "V", 1_000),
            _check(R, "Hero"),
            _bet(R, "V", 500),
            _raise(R, "Hero", 1_500),
            _call(R, "V", 1_000),
        ]
    )


@pytest.mark.parametrize(
    ("dealt", "refusal"),
    [({}, "карты героя неизвестны"), ({"Hero": ["Kh", "3c"]}, None)],
    ids=["unknown", "tool-failure"],
)
def test_a_raise_keeps_the_fold_frequency_reason_where_the_call_tool_refuses(dealt, refusal):
    """Инструмент колла к рейзу героя неприменим: его отказ по данным — карт нет
    или перебор не принял карты (K♥ героя уже на борде) — остаётся у колла, а у
    рейза причина §4.9."""
    en = _raises_on_every_street()
    broken = en.model_copy(update={"hand": en.hand.model_copy(update={"dealt": dealt})})
    points = [p for p in analyze_hand(broken).points if p.street is not P]
    raises = [p for p in points if p.action_taken == "raise"]
    assert [p.street for p in raises] == [F, T, R]
    for point in raises:
        assert point.detail["unjudged"] == POSTFLOP_FOLD_FREQUENCY_REASON
    for dp in (_points(broken, F)[-1], _points(broken, R)[-1]):
        verdict = turn_flop_verdict(dp, broken) or river_verdict(dp, broken)
        assert verdict is not None
        assert verdict.detail["unjudged"] == POSTFLOP_FOLD_FREQUENCY_REASON
    # Та же граница у колла остаётся причиной о данных.
    called = _oop(
        [*_V_OPENS, _check(F, "Hero"), _bet(F, "V", 300), _call(F, "Hero", 300)]
    )
    called = called.model_copy(
        update={"hand": called.hand.model_copy(update={"dealt": dealt})}
    )
    call_point = turn_flop_verdict(_points(called, F)[-1], called)
    assert call_point is not None
    reason = call_point.detail["unjudged"]
    assert reason != POSTFLOP_FOLD_FREQUENCY_REASON
    if refusal is not None:
        assert reason == refusal


def test_a_raise_on_a_short_board_keeps_the_fold_frequency_reason():
    """Борд без карт тёрна и ривера — граница инструмента колла, но не рейза."""
    en = _raises_on_every_street()
    short = en.model_copy(
        update={"hand": en.hand.model_copy(update={"boards": {F: en.hand.boards[F]}})}
    )
    for street in (T, R):
        dp = _points(short, street)[-1]
        assert dp.action.kind is ActionKind.RAISE
        verdict = turn_flop_verdict(dp, short) or river_verdict(dp, short)
        assert verdict is not None
        assert verdict.detail["unjudged"] == POSTFLOP_FOLD_FREQUENCY_REASON


def test_a_deep_limped_pot_keeps_its_old_reason():
    """Причину без опена §4.9 не трогает."""
    point = analyze_hand(_oop([*_V_LIMPS, _check(F, "Hero"), _check(F, "V")])).points[0]
    assert point.street is P
    assert point.detail["unjudged"].startswith("глубже пуш-фолд-зоны")


def test_a_foreign_shape_under_the_postflop_line_key_is_not_swallowed():
    point = analyze_hand(_reference_hand()).points[2]
    broken = point.model_copy(update={"detail": {**point.detail, POSTFLOP_LINE_DETAIL: {"x": 1}}})
    with pytest.raises(ValueError):
        postflop_line_detail(broken)
