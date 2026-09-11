"""Точка тёрна и флопа в разборе: числа перебора, границы формы, отсутствие цены.

Рука синтетическая, но собирается как `RawHand` и прогоняется через настоящий
конвейер (`normalize` → `enrich` → `analyze_hand`) — та же дисциплина, что в
`test_river_analysis`: анализ проверяется на входе, который конвейер
действительно производит.

Числа тёрна живой руки (борд `Jc 6d As 2h`, у героя `Jh Ts`, банк 160 000,
доплата 69 000) — те же, что в `test_turn_flop_call`: там они проверены против
инструмента, здесь — что до вердикта и до игрока они доезжают неискажёнными.
"""

from __future__ import annotations

from datetime import UTC, datetime

from harness.analysis import analyze_hand
from harness.analysis.tools.turn_flop_call import turn_flop_call_requirement
from harness.analysis.turn_flop import turn_flop_verdict
from harness.contracts import (
    TURN_FLOP_CALL_DETAIL,
    ActionKind,
    EnrichedHand,
    Post,
    PostKind,
    Provenance,
    RawAction,
    RawHand,
    SeatInfo,
    Street,
    Zone,
    is_judged,
    leak_rule_of_point,
    turn_flop_call_detail,
)
from harness.engine import enrich
from harness.normalizer import normalize

_SB, _BB = 5_000, 10_000
_STACK = 400_000

# Вклады улиц подобраны так, чтобы банк перед решением на тёрне сошёлся с живой
# рукой: 20 500 + 25 000 на каждого — это 91 000, плюс ставка 69 000. Открытие
# префлопа не меньше двух блайндов — иначе движок отвергает рейз как ниже
# минимального, и точек решения в раздаче не остаётся вовсе.
_PREFLOP_TO = 20_500
_FLOP_BET = 25_000
_TURN_BET = 69_000
_FLOP_POT = 2 * _PREFLOP_TO + _FLOP_BET
_TURN_POT = 2 * (_PREFLOP_TO + _FLOP_BET) + _TURN_BET

_LIVE_BOARD = {Street.FLOP: ["Jc", "6d", "As"], Street.TURN: ["2h"], Street.RIVER: ["Ac"]}
_LIVE_HERO = ("Jh", "Ts")
_LIVE_FLOP = ["Jc", "6d", "As"]
_LIVE_TURN = ["Jc", "6d", "As", "2h"]


def _act(street, label, kind, amount=None, to_amount=None) -> RawAction:
    return RawAction(
        street=street,
        label=label,
        kind=kind,
        amount=amount,
        to_amount=to_amount,
        raw_line=f"{label}: {kind} {amount or ''}".strip(),
    )


def _hand(
    *,
    hero_cards: tuple[str, str] = _LIVE_HERO,
    boards: dict[Street, list[str]] | None = None,
    hero_faces_a_bet: bool = True,
) -> EnrichedHand:
    """Хедз-ап: соперник ставит на флопе и на тёрне, герой отвечает коллом.

    `hero_faces_a_bet=False` обрывает торговлю тёрна чеком обоих: точка решения
    у героя есть, ставки перед ней нет.
    """
    seats = [
        SeatInfo(seat=1, label="V", stack=_STACK),
        SeatInfo(seat=2, label="Hero", stack=_STACK),
    ]
    posts = [
        Post(label="V", kind=PostKind.SMALL_BLIND, amount=_SB),
        Post(label="Hero", kind=PostKind.BIG_BLIND, amount=_BB),
    ]
    actions = [
        _act(Street.PREFLOP, "V", ActionKind.RAISE, _PREFLOP_TO - _SB, _PREFLOP_TO),
        _act(Street.PREFLOP, "Hero", ActionKind.CALL, _PREFLOP_TO - _BB),
        _act(Street.FLOP, "Hero", ActionKind.CHECK),
        _act(Street.FLOP, "V", ActionKind.BET, _FLOP_BET, _FLOP_BET),
        _act(Street.FLOP, "Hero", ActionKind.CALL, _FLOP_BET),
        _act(Street.TURN, "Hero", ActionKind.CHECK),
    ]
    if hero_faces_a_bet:
        actions += [
            _act(Street.TURN, "V", ActionKind.BET, _TURN_BET, _TURN_BET),
            _act(Street.TURN, "Hero", ActionKind.CALL, _TURN_BET),
        ]
    else:
        actions.append(_act(Street.TURN, "V", ActionKind.CHECK))
    actions += [
        _act(Street.RIVER, "Hero", ActionKind.CHECK),
        _act(Street.RIVER, "V", ActionKind.CHECK),
    ]
    raw = RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="synthetic",
        hand_no="SYN-TURN",
        tournament_id="T1",
        tournament_name="synthetic",
        level=1,
        sb=_SB,
        bb=_BB,
        ante=0,
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        table_name="syn",
        max_seats=len(seats),
        button_seat=1,
        seats=seats,
        posts=posts,
        dealt={"Hero": list(hero_cards)},
        actions=actions,
        boards=boards or _LIVE_BOARD,
    )
    return enrich(normalize(raw))


def _last_point_on(res, street: Street):
    """Последняя точка героя на улице: та, в которой перед ним стоит ставка.

    Точек на улице две — чек героя в начале торговли и ответ на ставку
    соперника; разбирается вторая, и брать её надо с конца, а не с начала.
    """
    return [point for point in res.points if point.street is street][-1]


def _last_decision_on(en: EnrichedHand, street: Street):
    """То же на уровне движка: последняя точка решения героя на улице."""
    return [dp for dp in en.report.decision_points if dp.street is street][-1]


def test_the_hand_reaches_the_turn_with_the_numbers_of_the_live_hand():
    """Гейт на саму фикстуру: банк и доплата в точке — те, о которых идёт речь."""
    en = _hand()
    turn = _last_decision_on(en, Street.TURN)
    assert (turn.pot_before, turn.to_call, turn.live_total) == (_TURN_POT, _TURN_BET, 2)
    assert _TURN_POT == 160_000 and _TURN_BET == 69_000
    flop = _last_decision_on(en, Street.FLOP)
    assert (flop.pot_before, flop.to_call) == (_FLOP_POT, _FLOP_BET)


def test_a_turn_point_carries_its_numbers_without_a_best_action():
    """Требование посчитано, лучшего действия нет — и точка при этом не пуста."""
    point = _last_point_on(analyze_hand(_hand()), Street.TURN)
    detail = turn_flop_call_detail(point)
    assert point.best_action == ""
    assert point.tools == ["turn_flop_call"]
    assert detail is not None
    assert (detail.pot_before, detail.to_call) == (_TURN_POT, _TURN_BET)
    assert detail.min_value_combos == 55
    assert detail.bluffs_needed_min_value == 18
    assert abs(detail.required_equity - 0.3013) < 5e-4


def test_the_numbers_of_the_point_are_the_numbers_of_the_tool():
    """Перебор зовётся с банком и доплатой ИЗ ДВИЖКА, а не из пересчёта анализа."""
    detail = turn_flop_call_detail(_last_point_on(analyze_hand(_hand()), Street.TURN))
    tool = turn_flop_call_requirement(_LIVE_HERO, _LIVE_TURN, _TURN_POT, _TURN_BET)
    assert detail is not None
    assert detail.required_equity == tool.required_equity
    assert detail.min_value_combos == tool.min_value_combos
    assert detail.bluffs_needed_min_value == tool.bluffs_needed_min_value


def test_the_bluff_share_is_computed_from_the_counted_numbers():
    """Доля берётся от чисел расчёта, а не собирается изложением заново."""
    detail = turn_flop_call_detail(_last_point_on(analyze_hand(_hand()), Street.TURN))
    assert detail is not None and detail.bluffs_needed_min_value is not None
    assert detail.bluff_share == 18 / (55 + 18)


def test_a_turn_point_is_not_priced_and_never_enters_the_sum():
    """Требование к диапазону — не EV решения: цены у точки нет ни в каком виде."""
    res = analyze_hand(_hand())
    point = _last_point_on(res, Street.TURN)
    assert point.ev_diff_bb == 0.0
    assert point.zone is Zone.STRICT and point.assumption is None
    assert is_judged(point) is False
    assert res.ranked == []
    assert res.total_ev_loss_bb == 0.0
    assert leak_rule_of_point(point) is None


def test_the_flop_point_is_counted_on_the_three_cards_of_its_street():
    """На флопе в перебор идут три карты — те, что лежат на столе в тот момент."""
    detail = turn_flop_call_detail(_last_point_on(analyze_hand(_hand()), Street.FLOP))
    on_three = turn_flop_call_requirement(_LIVE_HERO, _LIVE_FLOP, _FLOP_POT, _FLOP_BET)
    on_four = turn_flop_call_requirement(_LIVE_HERO, _LIVE_TURN, _FLOP_POT, _FLOP_BET)
    assert detail is not None
    assert detail.min_value_combos == on_three.min_value_combos
    assert on_three.min_value_combos != on_four.min_value_combos


def test_the_flop_numbers_do_not_move_when_later_cards_change():
    """Карты, которых в момент решения на столе ещё нет, в расчёт не входят."""
    other = dict(_LIVE_BOARD) | {Street.TURN: ["Kd"], Street.RIVER: ["7c"]}
    first = turn_flop_call_detail(_last_point_on(analyze_hand(_hand()), Street.FLOP))
    second = turn_flop_call_detail(_last_point_on(analyze_hand(_hand(boards=other)), Street.FLOP))
    assert first is not None
    assert first == second
    # А точка тёрна на другой карте — уже другая: подмена доехала до расчёта.
    changed = turn_flop_call_detail(_last_point_on(analyze_hand(_hand(boards=other)), Street.TURN))
    assert changed is not None
    assert changed.min_value_combos != 55


def test_a_check_on_the_turn_is_not_a_call_decision():
    """Перед героем нет ставки — коллировать нечего, и числа не считаются."""
    point = _last_point_on(analyze_hand(_hand(hero_faces_a_bet=False)), Street.TURN)
    assert turn_flop_call_detail(point) is None
    assert "нет ставки" in point.detail["unjudged"]


def test_hero_cards_unknown_leave_the_turn_without_numbers():
    """Перебор сравнивает комбо С РУКОЙ ГЕРОЯ: без неё считать нечего."""
    en = _hand()
    blind = en.model_copy(update={"hand": en.hand.model_copy(update={"dealt": {}})})
    dp = _last_decision_on(blind, Street.TURN)
    point = turn_flop_verdict(dp, blind)
    assert point is not None
    assert turn_flop_call_detail(point) is None
    assert point.detail["unjudged"] == "карты героя неизвестны"


def test_a_river_point_is_none_for_this_module():
    """Ривер этот модуль не разбирает и отказ за него не формулирует."""
    en = _hand()
    river = _last_decision_on(en, Street.RIVER)
    assert turn_flop_verdict(river, en) is None


def test_the_turn_numbers_never_reach_the_model():
    """Модель не видит точку тёрна — ни саму её, ни её числа.

    Две независимые причины, и проверяются обе: выжимка берётся из `ranked`
    (постфлоп-точки там нет), а величины `detail` попадают в промпт по белому
    списку (ключа тёрна в нём нет). Число, не попавшее в выжимку, для ответа
    модели запрещено (`faithfulness.unsupported_numbers`).
    """
    from harness.explanation.verdict_text import verdict_digest

    digest = verdict_digest(analyze_hand(_hand()))
    assert "тёрн" not in digest.text.lower()
    assert TURN_FLOP_CALL_DETAIL not in digest.text
    assert "55" not in digest.text
    assert 55.0 not in digest.allowed
    assert 18.0 not in digest.allowed


def test_a_multiway_turn_is_out_of_the_shape():
    """Живых больше двух — требование считалось бы к одному диапазону из нескольких."""
    en = _hand()
    dp = _last_decision_on(en, Street.TURN).model_copy(update={"live_total": 3})
    point = turn_flop_verdict(dp, en)
    assert point is not None
    assert turn_flop_call_detail(point) is None
    assert "живых в руке 3" in point.detail["unjudged"]


def test_a_turn_without_its_card_is_not_counted_as_a_flop():
    """Борд из трёх карт — законный вход перебора, и на тёрне это не повод считать.

    Иначе точка тёрна с недосданным бордом получила бы флоповые числа под
    подписью «Тёрн» — расчёт не по той улице, показанный игроку как по той.
    """
    en = _hand()
    short = en.model_copy(
        update={"hand": en.hand.model_copy(update={"boards": {Street.FLOP: _LIVE_FLOP}})}
    )
    point = turn_flop_verdict(_last_decision_on(short, Street.TURN), short)
    assert point is not None
    assert turn_flop_call_detail(point) is None
    assert point.detail["unjudged"] == "борд из 3 карт, а на этой улице их 4"
