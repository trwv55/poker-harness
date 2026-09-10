"""Риверная точка в разборе: числа перебора, границы формы, отсутствие цены.

Руки синтетические, но собираются как `RawHand` и прогоняются через настоящий
конвейер (`normalize` → `enrich` → `analyze_hand`) — та же дисциплина, что в
`test_preflop_analysis`: анализ проверяется на входе, который конвейер
действительно производит.

Числа живой руки (борд `Jc 6d As 2h Ac`, у героя `Jh Ts`, банк 398 000,
доплата 169 000) — те же, что в `test_river_call`: там они проверены против
инструмента, здесь — что до вердикта и до игрока они доезжают неискажёнными.
"""

from __future__ import annotations

from datetime import UTC, datetime

from harness.analysis import analyze_hand
from harness.analysis.river import river_verdict
from harness.analysis.tools.river_call import river_call_requirement
from harness.contracts import (
    RIVER_CALL_DETAIL,
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
    river_call_detail,
)
from harness.engine import enrich
from harness.normalizer import normalize

_SB, _BB = 5_000, 10_000
_STACK = 400_000

# Вклады улиц подобраны так, чтобы банк перед риверным решением сошёлся с живой
# рукой: 24 500 + 30 000 + 60 000 на каждого — это 229 000, плюс ставка 169 000.
_PREFLOP_TO = 24_500
_FLOP_BET = 30_000
_TURN_BET = 60_000
_RIVER_BET = 169_000
_POT_BEFORE = 398_000

_LIVE_BOARD = {Street.FLOP: ["Jc", "6d", "As"], Street.TURN: ["2h"], Street.RIVER: ["Ac"]}
_LIVE_HERO = ("Jh", "Ts")

# Герой разыгрывает борд худшими картами: проигрывающих ему комбо на борде нет
# вовсе, поэтому блефов требуется больше, чем их существует.
_PROVEN_BOARD = {Street.FLOP: ["Ac", "Kd", "Qh"], Street.TURN: ["Js"], Street.RIVER: ["9s"]}
_PROVEN_HERO = ("3c", "2d")


def _act(street, label, kind, amount=None, to_amount=None) -> RawAction:
    return RawAction(
        street=street,
        label=label,
        kind=kind,
        amount=amount,
        to_amount=to_amount,
        raw_line=f"{label}: {kind} {amount or ''}".strip(),
    )


def _river_hand(
    *,
    hero_cards: tuple[str, str] = _LIVE_HERO,
    boards: dict[Street, list[str]] | None = None,
    hero_answer: ActionKind = ActionKind.CALL,
    hero_faces_a_bet: bool = True,
) -> EnrichedHand:
    """Хедз-ап до ривера: соперник ставит, герой отвечает.

    `hero_faces_a_bet=False` обрывает торговлю ривера чеком обоих: точка решения
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
    ]
    for street, bet in ((Street.FLOP, _FLOP_BET), (Street.TURN, _TURN_BET)):
        actions += [
            _act(street, "Hero", ActionKind.CHECK),
            _act(street, "V", ActionKind.BET, bet, bet),
            _act(street, "Hero", ActionKind.CALL, bet),
        ]
    actions.append(_act(Street.RIVER, "Hero", ActionKind.CHECK))
    if hero_faces_a_bet:
        actions.append(_act(Street.RIVER, "V", ActionKind.BET, _RIVER_BET, _RIVER_BET))
        actions.append(
            _act(Street.RIVER, "Hero", ActionKind.FOLD)
            if hero_answer is ActionKind.FOLD
            else _act(Street.RIVER, "Hero", ActionKind.CALL, _RIVER_BET)
        )
    else:
        actions.append(_act(Street.RIVER, "V", ActionKind.CHECK))
    raw = RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="synthetic",
        hand_no="SYN-RIVER",
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


def _facing_the_bet(en: EnrichedHand):
    """Точка ривера, в которой перед героем стоит ставка, — последняя в раздаче."""
    return en.report.decision_points[-1]


def test_the_hand_reaches_the_river_with_the_numbers_of_the_live_hand():
    """Гейт на саму фикстуру: банк и доплата в точке — те, о которых идёт речь."""
    dp = _facing_the_bet(_river_hand())
    assert (dp.street, dp.pot_before, dp.to_call) == (Street.RIVER, _POT_BEFORE, _RIVER_BET)
    assert dp.live_total == 2


def test_a_river_point_without_the_proof_still_carries_its_numbers():
    """Фолд не доказан: лучшего действия нет, но точка не пуста."""
    en = _river_hand()
    point = analyze_hand(en).points[-1]
    detail = river_call_detail(point)
    assert point.best_action == ""
    assert detail is not None
    assert (detail.pot_before, detail.to_call) == (_POT_BEFORE, _RIVER_BET)
    assert detail.fold_proven is False
    assert detail.min_value_combos == 94
    assert abs(detail.bluffs_needed_min_value - 38.19) < 5e-3
    assert abs(detail.required_equity - 0.298) < 5e-4


def test_the_numbers_of_the_point_are_the_numbers_of_the_tool():
    """Перебор зовётся с банком и доплатой ИЗ ДВИЖКА, а не из пересчёта анализа."""
    point = analyze_hand(_river_hand()).points[-1]
    detail = river_call_detail(point)
    assert detail is not None
    tool = river_call_requirement(
        _LIVE_HERO, ["Jc", "6d", "As", "2h", "Ac"], _POT_BEFORE, _RIVER_BET
    )
    assert detail.required_equity == tool.required_equity
    assert detail.min_value_combos == tool.min_value_combos
    assert detail.bluffs_needed_min_value == tool.bluffs_needed_min_value
    assert detail.fold_proven == tool.fold_proven


def test_the_bluff_share_is_computed_from_the_unrounded_numbers():
    """Доля берётся от посчитанных чисел, а не от округлённых до штук."""
    detail = river_call_detail(analyze_hand(_river_hand()).points[-1])
    assert detail is not None
    expected = detail.bluffs_needed_min_value / (
        detail.min_value_combos + detail.bluffs_needed_min_value
    )
    assert detail.bluff_share == expected
    # На округлённых числах доля вышла бы другой уже в первом знаке процента.
    assert abs(detail.bluff_share - 38 / (94 + 38)) > 1e-4


def test_a_proven_river_fold_names_the_line_and_stays_strict():
    """Борд исчерпан — лучшее действие названо, зона «строго», диапазона нет."""
    en = _river_hand(hero_cards=_PROVEN_HERO, boards=_PROVEN_BOARD, hero_answer=ActionKind.FOLD)
    point = analyze_hand(en).points[-1]
    detail = river_call_detail(point)
    assert detail is not None and detail.fold_proven is True
    assert point.best_action == "fold"
    assert point.action_taken == "fold"
    assert point.zone is Zone.STRICT
    assert point.assumption is None
    assert point.tools == ["river_call"]


def test_a_proven_river_fold_is_not_priced_and_never_enters_the_sum():
    """Требование к диапазону — не EV решения: цены у точки нет ни в каком виде."""
    en = _river_hand(hero_cards=_PROVEN_HERO, boards=_PROVEN_BOARD)
    res = analyze_hand(en)
    point = res.points[-1]
    assert point.best_action == "fold"
    assert point.ev_diff_bb == 0.0
    assert is_judged(point) is False
    assert res.ranked == []
    assert res.total_ev_loss_bb == 0.0


def test_a_river_point_matches_no_leak_rule():
    """Таксономия ликов постфлоп не описывает — ни с вердиктом, ни без него."""
    proven = analyze_hand(
        _river_hand(hero_cards=_PROVEN_HERO, boards=_PROVEN_BOARD)
    ).points[-1]
    unproven = analyze_hand(_river_hand()).points[-1]
    assert leak_rule_of_point(proven) is None
    assert leak_rule_of_point(unproven) is None


def test_a_check_on_the_river_is_not_a_call_decision():
    """Перед героем нет ставки — коллировать нечего, и числа не считаются."""
    en = _river_hand(hero_faces_a_bet=False)
    point = analyze_hand(en).points[-1]
    assert point.street is Street.RIVER
    assert river_call_detail(point) is None
    assert "нет ставки" in point.detail["unjudged"]


def _three_handed_river_hand() -> EnrichedHand:
    """Трое доходят до ривера: кнопка коллирует префлоп, дальше все чекают.

    Числа живой руки здесь не нужны — нужна только форма: в момент решения
    героя живых трое, и один из них ещё не отвечал на ставку.
    """
    seats = [
        SeatInfo(seat=1, label="V", stack=_STACK),
        SeatInfo(seat=2, label="Hero", stack=_STACK),
        SeatInfo(seat=3, label="X", stack=_STACK),
    ]
    posts = [
        Post(label="V", kind=PostKind.SMALL_BLIND, amount=_SB),
        Post(label="Hero", kind=PostKind.BIG_BLIND, amount=_BB),
    ]
    actions = [
        _act(Street.PREFLOP, "X", ActionKind.CALL, _BB),
        _act(Street.PREFLOP, "V", ActionKind.CALL, _BB - _SB),
        _act(Street.PREFLOP, "Hero", ActionKind.CHECK),
    ]
    for street in (Street.FLOP, Street.TURN):
        actions += [
            _act(street, "V", ActionKind.CHECK),
            _act(street, "Hero", ActionKind.CHECK),
            _act(street, "X", ActionKind.CHECK),
        ]
    actions += [
        _act(Street.RIVER, "V", ActionKind.BET, _RIVER_BET, _RIVER_BET),
        _act(Street.RIVER, "Hero", ActionKind.FOLD),
        _act(Street.RIVER, "X", ActionKind.FOLD),
    ]
    raw = RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="synthetic",
        hand_no="SYN-RIVER-3",
        tournament_id="T1",
        tournament_name="synthetic",
        level=1,
        sb=_SB,
        bb=_BB,
        ante=0,
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        table_name="syn",
        max_seats=3,
        button_seat=3,
        seats=seats,
        posts=posts,
        dealt={"Hero": list(_LIVE_HERO)},
        actions=actions,
        boards=_LIVE_BOARD,
    )
    return enrich(normalize(raw))


def test_a_three_handed_river_is_out_of_the_shape():
    """Живых трое — требование считалось бы к одному диапазону из двух."""
    en = _three_handed_river_hand()
    dp = _facing_the_bet(en)
    assert dp.street is Street.RIVER and dp.to_call == _RIVER_BET
    assert dp.live_total == 3
    point = analyze_hand(en).points[-1]
    assert river_call_detail(point) is None
    assert "живых в руке 3" in point.detail["unjudged"]


def test_the_turn_and_the_flop_stay_without_numbers():
    """Этот кусок — только ривер: на тёрне и флопе не появилось ни одного числа."""
    res = analyze_hand(_river_hand())
    earlier = [p for p in res.points if p.street in (Street.FLOP, Street.TURN)]
    assert earlier
    for point in earlier:
        assert river_call_detail(point) is None
        assert RIVER_CALL_DETAIL not in point.detail
        assert point.best_action == ""


def test_hero_cards_unknown_leave_the_river_without_numbers():
    """Перебор сравнивает комбо С РУКОЙ ГЕРОЯ: без неё считать нечего."""
    en = _river_hand()
    blind = en.model_copy(update={"hand": en.hand.model_copy(update={"dealt": {}})})
    point = river_verdict(_facing_the_bet(blind), blind)
    assert point is not None
    assert river_call_detail(point) is None
    assert point.detail["unjudged"] == "карты героя неизвестны"


def test_a_point_before_the_river_is_none_for_this_module():
    """Тёрн и флоп этот модуль не разбирает и отказ за них не формулирует."""
    en = _river_hand()
    turn = next(dp for dp in en.report.decision_points if dp.street is Street.TURN)
    assert river_verdict(turn, en) is None


def test_the_river_numbers_never_reach_the_model():
    """Модель не видит риверную точку — ни саму её, ни её числа.

    Две независимые причины, и проверяются обе: выжимка берётся из `ranked`
    (риверной точки там нет), а величины `detail` попадают в промпт по белому
    списку (риверных ключей в нём нет). Число, не попавшее в выжимку, для
    ответа модели запрещено (`faithfulness.unsupported_numbers`), то есть
    сказать его она не вправе, даже угадав.
    """
    from harness.explanation.verdict_text import verdict_digest

    res = analyze_hand(_river_hand())
    digest = verdict_digest(res)
    assert "ривер" not in digest.text.lower()
    assert "94" not in digest.text
    assert 94.0 not in digest.allowed
    assert 38.2 not in digest.allowed
