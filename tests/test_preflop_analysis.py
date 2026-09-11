"""Префлоп-анализ: классификация спота, вердикт, зона доверия, цена расхождения.

Синтетические руки собираются как `RawHand` и прогоняются через настоящий
конвейер (`normalize` → `enrich`): собирать `CanonicalHand` вручную значило бы
проверять анализ на входе, которого конвейер никогда не произведёт.
"""

from datetime import UTC, datetime

import pytest

from harness.analysis import analyze_hand
from harness.analysis import preflop as preflop_module
from harness.analysis.classifier import (
    PUSHFOLD_MAX_EFF_BB,
    classify,
    in_action_order_after,
    spot_for,
    table_state,
)
from harness.analysis.error_cost import rank_points, total_ev_loss_bb
from harness.analysis.preflop import (
    _depth_key,
    _model_equity,
    _posted_before_shove,
    _rivals_when_shoved,
    _shover,
    _shover_equilibrium,
    _table_dead_bb,
    cheap_fold_verdict,
    verdict_for,
    zone_for,
)
from harness.analysis.tools.multiway import Seat as MultiwaySeat
from harness.analysis.tools.multiway import unopened_shove_equilibrium
from harness.analysis.tools.pushfold import call_shove_ev_bb, nash_hu
from harness.contracts import (
    ActionKind,
    Assumption,
    Post,
    PostKind,
    Provenance,
    RawAction,
    RawHand,
    SeatInfo,
    Street,
    all_classes,
)
from harness.engine import enrich
from harness.normalizer import normalize
from harness.parsers.hh_parser import parse_hand
from tests.conftest import FIXTURE_DAILY, FIXTURE_PKO, requires_fixtures
from tests.test_hh_parser import SAMPLE

_BB = 2
_SB = 1
_BOARD = {Street.FLOP: ["Kd", "8h", "3s"], Street.TURN: ["4c"], Street.RIVER: ["9d"]}


def _max_weight_gap(left, right) -> float:
    """Худшее расхождение двух диапазонов по весу класса — мера опоры N = 1."""
    return max(abs(left.weight(cls) - right.weight(cls)) for cls in all_classes())


def _raw(
    *,
    seats: list[SeatInfo],
    button_seat: int,
    posts: list[Post],
    actions: list[RawAction],
    dealt: dict[str, list[str]],
    ante: int = 0,
    sb: int = _SB,
    bb: int = _BB,
    boards: dict[Street, list[str]] | None = None,
    showdowns: list | None = None,
) -> RawHand:
    """Синтетическая рука без `collected`/`summary`.

    Без строк выплат валидатор не сверяет выплаты (их в источнике нет), и рука
    проходит на одной сохранности фишек — этого достаточно: проверяется анализ,
    а не парсер.
    """
    return RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="synthetic",
        hand_no="SYN",
        tournament_id="T1",
        tournament_name="synthetic",
        level=1,
        sb=sb,
        bb=bb,
        ante=ante,
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        table_name="syn",
        max_seats=len(seats),
        button_seat=button_seat,
        seats=seats,
        posts=posts,
        dealt=dealt,
        actions=actions,
        boards=boards or {},
        showdowns=showdowns or [],
    )


def _fold(label: str) -> RawAction:
    return RawAction(
        street=Street.PREFLOP, label=label, kind=ActionKind.FOLD, raw_line=f"{label}: folds"
    )


def _shove(label: str, to_amount: int, already: int = 0) -> RawAction:
    return RawAction(
        street=Street.PREFLOP,
        label=label,
        kind=ActionKind.RAISE,
        amount=to_amount - already,
        to_amount=to_amount,
        is_all_in=True,
        raw_line=f"{label}: raises {to_amount - already} to {to_amount} and is all-in",
    )


def _call(label: str, amount: int, *, all_in: bool = False) -> RawAction:
    return RawAction(
        street=Street.PREFLOP,
        label=label,
        kind=ActionKind.CALL,
        amount=amount,
        is_all_in=all_in,
        raw_line=f"{label}: calls {amount}",
    )


def _showdown(label: str, cards: list[str]) -> dict[str, object]:
    return {"label": label, "cards": cards}


def _make_hu_shove_hand(hero_cards: tuple[str, str], eff_bb: float, *, called: bool = True):
    """Хедз-ап, Hero на кнопке (= малый блайнд) шовит, оппонент коллирует.

    Вскрытие в руке есть намеренно: без него тест «вердикт не зависит от
    вскрытых карт» был бы пустым.
    """
    stack = round(eff_bb * _BB)
    seats = [SeatInfo(seat=1, label="Hero", stack=stack), SeatInfo(seat=2, label="V", stack=stack)]
    posts = [
        Post(label="Hero", kind=PostKind.SMALL_BLIND, amount=_SB),
        Post(label="V", kind=PostKind.BIG_BLIND, amount=_BB),
    ]
    actions: list[RawAction] = [_shove("Hero", stack, already=_SB)]
    villain_cards = ["7c", "2s"]
    if called:
        actions.append(_call("V", stack - _BB, all_in=True))
        boards, showdowns = _BOARD, [
            _showdown("Hero", list(hero_cards)),
            _showdown("V", villain_cards),
        ]
    else:
        actions.append(_fold("V"))
        boards, showdowns = {}, []
    raw = _raw(
        seats=seats,
        button_seat=1,
        posts=posts,
        actions=actions,
        dealt={"Hero": list(hero_cards), "V": villain_cards if called else []},
        boards=boards,
        showdowns=showdowns,
    )
    return enrich(normalize(raw))


def _make_hu_facing_shove_hand(stack: int, ante: int = 0):
    """Хедз-ап: оппонент на кнопке (= малый блайнд) шовит, Hero в большом блайнде.

    Зеркало `_make_hu_shove_hand`. Позади шовера ровно одно место — герой, — и
    это та единственная конфигурация, в которой равновесие его стола обязано
    совпасть с `nash_hu`. Стек задаётся в фишках: глубина равновесия считается
    уже за вычетом анте, и дробить её здесь было бы лишним звеном.
    """
    seats = [
        SeatInfo(seat=1, label="V", stack=stack),
        SeatInfo(seat=2, label="Hero", stack=stack),
    ]
    posts = (
        [Post(label=label, kind=PostKind.ANTE, amount=ante) for label in ("V", "Hero")]
        if ante
        else []
    )
    posts += [
        Post(label="V", kind=PostKind.SMALL_BLIND, amount=_SB),
        Post(label="Hero", kind=PostKind.BIG_BLIND, amount=_BB),
    ]
    actions = [
        _shove("V", stack - ante, already=_SB),
        _call("Hero", stack - ante - _BB, all_in=True),
    ]
    raw = _raw(
        seats=seats,
        button_seat=1,
        posts=posts,
        actions=actions,
        dealt={"Hero": ["Ah", "Ad"]},
        ante=ante,
    )
    return enrich(normalize(raw))


_SIX_MAX_SEATS: tuple[str, ...] = ("SB", "BB", "UTG", "HJ", "CO", "BTN")


def _six_max(stacks: dict[str, int], hero_position: str, ante: int = 0):
    """Места 6-max с кнопкой на месте 6: позиции идут SB, BB, UTG, HJ, CO, BTN."""
    labels = {pos: ("Hero" if pos == hero_position else pos) for pos in _SIX_MAX_SEATS}
    seats = [
        SeatInfo(seat=i + 1, label=labels[pos], stack=stacks[pos])
        for i, pos in enumerate(_SIX_MAX_SEATS)
    ]
    posts = [
        Post(label=labels["SB"], kind=PostKind.SMALL_BLIND, amount=_SB),
        Post(label=labels["BB"], kind=PostKind.BIG_BLIND, amount=_BB),
    ]
    if ante:
        posts = [Post(label=s.label, kind=PostKind.ANTE, amount=ante) for s in seats] + posts
    return labels, seats, posts


def _make_multiway_shove_hand(
    hero_cards: tuple[str, str], eff_bb: float, players_behind: int, *, ante: int = 0
):
    """Hero шовит в неоткрытый банк, позади него `players_behind` живых игроков.

    Позиция героя выбирается так, чтобы позади осталось ровно нужное число мест
    (BTN + блайнды = 3), все до него пасуют.
    """
    order = list(_SIX_MAX_SEATS[2:]) + ["SB", "BB"]  # порядок хода на префлопе
    hero_index = len(order) - players_behind - 1
    hero_position = order[hero_index]
    stack = round(eff_bb * _BB)
    labels, seats, posts = _six_max(dict.fromkeys(_SIX_MAX_SEATS, stack), hero_position, ante)
    already = {"SB": _SB, "BB": _BB}.get(hero_position, 0)
    actions = [_fold(labels[pos]) for pos in order[:hero_index]]
    actions.append(_shove("Hero", stack, already=already))
    actions += [_fold(labels[pos]) for pos in order[hero_index + 1 :]]
    raw = _raw(
        seats=seats,
        button_seat=6,
        posts=posts,
        actions=actions,
        dealt={"Hero": list(hero_cards)},
        ante=ante,
    )
    return enrich(normalize(raw))


def _make_facing_shove_hand(hero_cards: tuple[str, str], eff_bb: float, shover_bb: float):
    """UTG шовит, все пасуют до Hero в большом блайнде.

    Это НЕ равновесная раздача «SB против BB»: шовит игрок ранней позиции, а
    блайнд малого — мёртвые деньги. Зону такой точке обязан ставить bracket-тест.
    """
    hero_stack = round(eff_bb * _BB)
    shover_stack = round(shover_bb * _BB)
    stacks = dict.fromkeys(_SIX_MAX_SEATS, max(hero_stack, shover_stack) * 4)
    stacks["BB"] = hero_stack
    stacks["UTG"] = shover_stack
    labels, seats, posts = _six_max(stacks, "BB")
    actions = [_shove(labels["UTG"], shover_stack)]
    actions += [_fold(labels[pos]) for pos in ("HJ", "CO", "BTN", "SB")]
    actions.append(_call("Hero", min(shover_stack, hero_stack) - _BB, all_in=True))
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": list(hero_cards)}
    )
    return enrich(normalize(raw))


def _make_shove_with_deep_player_behind(shover_bb: float, hero_bb: float, deep_bb: float):
    """UTG шовит коротким стеком, Hero в SB пасует, а глубокий BB ещё не ходил.

    Здесь ровно та форма, на которой ломалась глубина: живой игрок, которого
    решение Hero не касается, глубже шовера. Все, кто между ними, пасуют — они
    из живых выбывают и на прежнюю формулу не влияли бы.
    """
    deep = round(deep_bb * _BB)
    stacks = dict.fromkeys(_SIX_MAX_SEATS, deep)
    stacks["UTG"] = round(shover_bb * _BB)
    stacks["SB"] = round(hero_bb * _BB)
    stacks["BB"] = deep
    labels, seats, posts = _six_max(stacks, "SB")
    actions = [_shove(labels["UTG"], stacks["UTG"])]
    actions += [_fold(labels[pos]) for pos in ("HJ", "CO", "BTN")]
    actions += [_fold("Hero"), _fold(labels["BB"])]
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["Kd", "Qs"]}
    )
    return enrich(normalize(raw))


# --- Тесты плана ----------------------------------------------------------------


def test_short_shove_stays_in_pushfold_zone_with_deep_player_behind():
    """Шов короткого стека судится как пуш-фолд, даже когда позади живой глубокий игрок.

    Гейт глубины обязан спрашивать «на какую глубину играется ЭТО решение», а
    ответ на него даёт шовер: больше его стека в решении не разыграть. Прежняя
    формула брала максимум остатков живых оппонентов и в этой раздаче выбирала
    BB — игрока, чьей ставку Hero не отвечает, — после чего спот вылетал из
    разбора как «глубже пуш-фолд-зоны».

    Тест разводит две причины, по которым точка могла бы попасть в зону: он
    проверяет не только класс спота (у классификатора есть и другие гейты,
    любой из которых мог бы дать тот же класс), но и само число глубины и то,
    что альтернативный кандидат в оппоненты заведомо за границей зоны.
    """
    en = _make_shove_with_deep_player_behind(shover_bb=3.0, hero_bb=40.0, deep_bb=40.0)
    dp = next(d for d in en.report.decision_points if d.label == "Hero")
    state = table_state(dp, en)

    shover = state.aggressor
    assert shover is not None and shover.behind == 0  # перед Hero олл-ин
    deep = next(s for s in state.seats if s.position == "BB")
    assert deep.live and not deep.acted  # он ещё в руке и ходит после Hero
    # Кандидат, которого выбирала прежняя формула, — заведомо за границей зоны:
    # без правильного выбора оппонента этот тест не может пройти случайно.
    assert deep.behind / _BB > PUSHFOLD_MAX_EFF_BB

    assert dp.eff_stack == shover.stack_after_ante == 6
    assert dp.eff_stack_bb == 3.0
    assert spot_for(dp, state) == "pushfold_facing_shove"


def test_eff_stack_is_the_depth_the_model_indexes_by():
    """Глубина движка и `stack_after_ante` анализа — одна величина двумя путями.

    Движок считает её реплеем (`stacks + bets` в PokerKit), анализ — по строкам
    руки. Расхождение означало бы, что анте попало в глубину: оно мёртвое, и
    складывать его со стеком значило бы посчитать его дважды. Рука настоящая и с
    анте 750.

    Тест закрепляет ровно одно равенство — с `stack_after_ante` героя и
    агрессора. Про другие понятия глубины, которые считает анализ, он не
    высказывается.
    """
    en = enrich(normalize(parse_hand(SAMPLE, source_ref="x")))
    dp = next(d for d in en.report.decision_points if d.label == "Hero")
    state = table_state(dp, en)
    assert state.aggressor is not None
    assert dp.eff_stack == min(state.hero.stack_after_ante, state.aggressor.stack_after_ante)
    assert dp.eff_stack == 3891 - 750  # стек Hero без анте


def test_fixture_hand_correct_call_not_flagged():
    """Hero 0.65bb в SB с K3s коллит 141 в банк 14532 — верно против любого диапазона."""
    res = analyze_hand(enrich(normalize(parse_hand(SAMPLE, source_ref="x"))))
    hero_points = [p for p in res.points if p.spot == "pushfold_facing_shove"]
    assert len(hero_points) == 1
    p = hero_points[0]
    # strict здесь не по улице, а по правилу зоны: колл 141 в банк 14532 верен
    # против любого диапазона (порог эквити 0.96%) и при любом поведении живого
    # BB за героем — вилка устойчива по обеим осям, допущение нагрузки не несёт.
    assert p.zone == "strict" and p.assumption is None
    assert p.ev_diff_bb >= -0.05
    assert p.best_action == "call" and p.action_taken == "call"
    assert p.detail["bracket"] == "stable"
    assert p.detail["required_equity"] < 0.01
    assert p.detail["live_others"] == 1  # игрок позади есть, но вердикт не двигает


def test_synthetic_bad_open_shove_flagged():
    """HU 10bb, Hero открывает олином 32o — по равновесию фолд, шов стоит денег."""
    en = _make_hu_shove_hand(hero_cards=("3c", "2d"), eff_bb=10.0)
    res = analyze_hand(en)
    p = res.points[0]
    assert p.spot == "pushfold_unopened" and p.ev_diff_bb < -0.3
    assert p.best_action == "fold" and p.action_taken == "shove"
    assert p.zone == "strict" and p.assumption is None
    assert res.ranked[0] == 0 and res.total_ev_loss_bb <= p.ev_diff_bb


def test_zone_rule_direct():
    assert zone_for("shove", "shove", live_total=2)[0] == "strict"  # HU: равновесие
    assert zone_for("shove", "shove", live_total=5)[0] == "strict"  # bracket стабилен
    z, why = zone_for("shove", "fold", live_total=5)
    assert z == "assuming" and why  # вердикт зависит от модели
    # HU без равновесной формы судится вилкой, как мультивей
    assert zone_for("shove", "fold", live_total=2, equilibrium=False)[0] == "assuming"


def test_multiway_shove_zone_invariant():
    en = _make_multiway_shove_hand(hero_cards=("Ac", "Ts"), eff_bb=12.0, players_behind=3)
    points = analyze_hand(en).points
    assert points
    for p in points:
        assert (p.assumption is not None) == (p.zone == "assuming")
        if p.assumption is not None:
            assert "мультивей" in p.assumption.note or "модел" in p.assumption.note


def test_range_independent_call_is_strict():
    """Колл шова с AA верен против любого диапазона -> bracket стабилен -> strict."""
    en = _make_facing_shove_hand(hero_cards=("Ah", "Ad"), eff_bb=12.0, shover_bb=12.0)
    p = analyze_hand(en).points[0]
    assert p.spot == "pushfold_facing_shove"
    assert p.best_action == "call" and p.zone == "strict"
    assert p.assumption is None
    assert p.detail["bracket"] == "stable"


def test_no_llm_and_no_result_bias():
    """Вскрытые карты соперника на вердикт не влияют — судим против диапазона."""
    en1 = _make_hu_shove_hand(hero_cards=("Ah", "Ad"), eff_bb=10.0)
    assert en1.hand.showdowns, "рука без вскрытия сделала бы этот тест пустым"
    en2 = en1.model_copy(deep=True)
    en2.hand.showdowns = []
    en2.hand.dealt = {"Hero": en1.hand.dealt["Hero"]}  # карты соперника убраны и отсюда
    assert analyze_hand(en1).points[0].ev_diff_bb == analyze_hand(en2).points[0].ev_diff_bb


# --- Классификация --------------------------------------------------------------


def test_classify_deep_stack_is_preflop_other():
    """40bb — не пуш-фолд-зона: вердикт не выносится, цена нулевая."""
    en = _make_multiway_shove_hand(hero_cards=("Ac", "Ts"), eff_bb=40.0, players_behind=3)
    dp = en.report.decision_points[0]
    assert classify(dp, en) == "preflop_other"
    p = analyze_hand(en).points[0]
    assert p.best_action == "" and p.ev_diff_bb == 0.0 and p.assumption is None


def test_classify_limp_in_pushfold_zone_is_not_priced():
    """Лимп на 10bb моделью пуш-фолда не оценивается — «не размечен», а не «фолд»."""
    stack = 20
    labels, seats, posts = _six_max(dict.fromkeys(_SIX_MAX_SEATS, stack), "CO")
    actions = [_fold(labels["UTG"]), _fold(labels["HJ"]), _call("Hero", _BB)]
    actions += [_fold(labels[pos]) for pos in ("BTN", "SB")]
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["Ac", "Ts"]}
    )
    en = enrich(normalize(raw))
    dp = en.report.decision_points[0]
    assert classify(dp, en) == "preflop_other"
    p = analyze_hand(en).points[0]
    assert p.action_taken == "call" and p.best_action == "" and p.ev_diff_bb == 0.0


def test_classify_postflop_is_skipped():
    """Постфлоп в v1 пропускается: спот размечен, вердикта нет, в ранжирование не идёт."""
    stack = 200
    labels, seats, posts = _six_max(dict.fromkeys(_SIX_MAX_SEATS, stack), "BB")
    actions = [_fold(labels[pos]) for pos in ("UTG", "HJ", "CO", "BTN")]
    actions.append(_call(labels["SB"], _SB))
    actions.append(
        RawAction(
            street=Street.PREFLOP,
            label="Hero",
            kind=ActionKind.CHECK,
            raw_line="Hero: checks",
        )
    )
    actions += [
        RawAction(
            street=Street.FLOP, label=labels["SB"], kind=ActionKind.CHECK, raw_line="SB: checks"
        ),
        RawAction(street=Street.FLOP, label="Hero", kind=ActionKind.CHECK, raw_line="Hero: checks"),
    ]
    raw = _raw(
        seats=seats,
        button_seat=6,
        posts=posts,
        actions=actions,
        dealt={"Hero": ["Ac", "Ts"]},
        boards={Street.FLOP: ["Kd", "8h", "3s"]},
    )
    en = enrich(normalize(raw))
    res = analyze_hand(en)
    postflop = [p for p in res.points if p.spot == "postflop"]
    assert len(postflop) == 1
    assert postflop[0].ev_diff_bb == 0.0 and postflop[0].best_action == ""
    assert res.ranked == []


# --- Восстановление банка и постов ----------------------------------------------


@requires_fixtures
@pytest.mark.parametrize("path", [FIXTURE_DAILY, FIXTURE_PKO])
def test_reconstructed_pot_matches_engine(path):
    """Посты и вклады, восстановленные анализом, сходятся с движком на всех руках.

    Это гейт на формулу банка: `shove_ev_bb` и `call_shove_ev_bb` кормятся
    восстановленными постами, и молчаливое расхождение с движком дало бы
    правдоподобно неверную цену решения.
    """
    from harness.analysis.classifier import table_state
    from harness.parsers.hh_parser import parse_file

    raws = parse_file(path.read_text(encoding="utf-8"), source_ref=path.name)
    checked = 0
    for raw in raws:
        en = enrich(normalize(raw))
        for dp in en.report.decision_points:
            if dp.street != Street.PREFLOP:
                continue
            state = table_state(dp, en)
            assert state.pot_before == dp.pot_before, (en.hand.hand_no, dp.index)
            assert state.to_call == dp.to_call, (en.hand.hand_no, dp.index)
            checked += 1
    assert checked > 100


# --- Референсный спот мультивей-равновесия ---------------------------------------


@requires_fixtures
@pytest.mark.slow  # решение равновесия на шестерых позади плюс мультивей-Монте-Карло
def test_the_reference_multiway_spot_reproduces_the_solved_table():
    """`TM6292955427` целиком через конвейер: колл-диапазоны, стол, цена шова.

    8-макс, анте 1200 с восьми мест, блайнды 4000/8000, UTG спасовал, герой
    UTG+1 с A5s и шестью игроками позади. Числа зафиксированы здесь потому, что
    это единственная точка, где связка «решатель — восстановление стола —
    `shove_ev_bb`» проверяется на настоящей раздаче, а не на синтетике: ошибка в
    переносе постов или банка сдвинет их все сразу.

    Разные ширины колла у мест — следствие разных постов: SB и BB платят за колл
    меньше на свой блайнд, и коллируют шире. Порядок `call_range_fractions` —
    порядок хода.
    """
    from harness.parsers.hh_parser import parse_file

    raws = parse_file(FIXTURE_PKO.read_text(encoding="utf-8"), source_ref="pko")
    target = next(raw for raw in raws if raw.hand_no == "TM6292955427")
    point = analyze_hand(enrich(normalize(target))).points[0]

    assert point.spot == "pushfold_unopened"
    assert point.detail["hero_class"] == "A5s"
    assert [round(w * 100, 2) for w in point.detail["call_range_fractions"]] == [
        11.50,
        10.54,
        10.54,
        10.54,
        12.30,
        13.86,
    ]
    assert round(point.detail["shove_range_fraction"] * 100, 2) == 18.82
    assert round(point.detail["p_all_fold"] * 100, 2) == 51.44
    assert point.detail["expected_callers"] == pytest.approx(0.629, abs=5e-4)
    assert point.detail["ev_shove_bb"] == pytest.approx(0.0628, abs=5e-5)
    # Эксплуатируемость профиля названа числом и уезжает в `detail` наружу —
    # заявлять «строго» «потому что равновесие» здесь оснований нет.
    assert point.detail["equilibrium_hand_regret_bb"] == pytest.approx(0.00421, abs=5e-6)


# --- Фолд-эквити ----------------------------------------------------------------


def test_fold_equity_gate_recorded_for_shove():
    """Гейт фолд-эквити виден в вердикте: шов без возможности фолда так и помечен."""
    en = _make_multiway_shove_hand(hero_cards=("Ac", "Ts"), eff_bb=12.0, players_behind=3)
    p = analyze_hand(en).points[0]
    assert p.spot == "pushfold_unopened"
    assert isinstance(p.detail["fold_equity_ok"], bool)
    assert p.detail["method"] == "subset_enumeration"
    assert p.detail["branches"] == 2**3


# --- Оценщик и ранжирование -----------------------------------------------------


def test_rank_points_orders_by_loss_and_skips_unjudged():
    from harness.contracts import PointVerdict, SpotKind, Zone

    def pv(index: int, spot: SpotKind, ev: float) -> PointVerdict:
        return PointVerdict(
            dp_index=index,
            street=Street.PREFLOP,
            spot=spot,
            zone=Zone.STRICT,
            action_taken="fold",
            best_action="shove",
            ev_diff_bb=ev,
        )

    points = [
        pv(0, SpotKind.PUSHFOLD_UNOPENED, -0.4),
        pv(1, SpotKind.PREFLOP_OTHER, 0.0),
        pv(2, SpotKind.PUSHFOLD_FACING_SHOVE, -2.5),
        pv(3, SpotKind.PUSHFOLD_UNOPENED, 0.0),
        pv(4, SpotKind.POSTFLOP, 0.0),
    ]
    assert rank_points(points) == [2, 0, 3]
    assert total_ev_loss_bb(points) == pytest.approx(-2.9)


# --- Зона: когда допущение несёт нагрузку ---------------------------------------


def test_two_live_without_equilibrium_shape_is_judged_by_bracket():
    """Шов ранней позиции, до которого спасовали все, кроме BB, — не игра `nash_hu`.

    Живых двое, но малый блайнд мёртвый, а шовер не малый блайнд: выдавать это за
    равновесие нельзя, зону обязан ставить bracket-тест.
    """
    en = _make_facing_shove_hand(hero_cards=("Ah", "Ad"), eff_bb=12.0, shover_bb=12.0)
    dp = en.report.decision_points[0]
    assert dp.live_total == 2
    p = analyze_hand(en).points[0]
    # причина зоны — устойчивость вилки по ширинам, а не равновесие
    assert "ширине" in p.detail["zone_reason"]
    assert "равновеси" not in p.detail["zone_reason"]


def test_hu_equilibrium_shape_is_strict_even_when_bracket_unstable():
    """Хедз-ап SB против BB: колл-диапазон известен из равновесия, вилка не нужна."""
    en = _make_hu_shove_hand(hero_cards=("3c", "2d"), eff_bb=10.0)
    p = analyze_hand(en).points[0]
    assert p.detail["bracket"] == "unstable"  # вилка сама по себе вердикт не удержала
    assert p.zone == "strict" and p.assumption is None
    assert "равновеси" in p.detail["zone_reason"]


def test_shove_without_fold_equity_is_marked():
    """На 2bb равновесный колл — любые две карты: фолд-эквити структурно нет.

    Пометка стоит у СУДИМОЙ точки: `p_all_fold` здесь ноль по построению, и
    вердикт всё равно выносится — «на этот шов всегда отвечают» есть свойство
    раздачи на 2bb, а не повод не называть её цену.
    """
    en = _make_multiway_shove_hand(hero_cards=("Ac", "Ts"), eff_bb=2.0, players_behind=2)
    p = analyze_hand(en).points[0]
    assert p.spot == "pushfold_unopened"
    assert p.detail["fold_equity_ok"] is False
    assert p.detail["p_all_fold"] == 0.0
    assert "unjudged" not in p.detail and p.best_action == "shove"


# --- Равновесие как эталон и неустойчивая вилка ----------------------------------


def test_a_shove_the_field_would_rarely_let_through_is_still_judged():
    """Широкий равновесный ответ вердикта не снимает: равновесие — эталон, а не гипотеза.

    На 4bb с пятерыми позади равновесные колл-диапазоны широки: шов проходит без
    ответа в 16% случаев, а отвечают на него в среднем полтора игрока из пяти
    (`p_all_fold` 0.1596, `expected_callers` 1.4777). Это не признак сломанной
    модели, а верный ответ на 4bb, и вердикт здесь есть.

    Больше того, вердикт от ширины колла вообще не зависит: интервал по всей
    калиброванной полосе лежит выше нуля, зона `strict`. То, насколько охотно
    отвечает настоящее поле, — замер, и живёт он в полосе ширин
    (`_CALL_WIDTH_BAND`), а не в гейте на существование вердикта.
    """
    en = _make_multiway_shove_hand(hero_cards=("Ad", "5d"), eff_bb=4.0, players_behind=5)
    p = analyze_hand(en).points[0]

    assert p.spot == "pushfold_unopened"
    assert "unjudged" not in p.detail
    assert p.best_action == "shove"
    assert p.zone == "strict" and p.assumption is None
    # Ровно те числа, по которым точка прежде снималась целиком.
    assert p.detail["p_all_fold"] == pytest.approx(0.1596, abs=5e-5)
    assert p.detail["expected_callers"] == pytest.approx(1.4777, abs=5e-5)
    # Вердикт не опирается на угаданную ширину: он одинаков на всей полосе.
    assert p.interval is not None and p.interval.low_bb > 0.0
    assert min(p.detail["ev_shove_by_width_bb"].values()) > 0.0


def test_the_table_equilibrium_never_hands_the_heads_up_range_to_everyone():
    """Хедз-ап колл-диапазон, розданный всем позади, `_table_equilibrium` не производит.

    Это тот самый дефект, ради которого когда-то считалась контрольная сумма
    модели: одна и та же ширина у каждого места независимо от того, сколько их.
    Решатель подыгры так не умеет — он решает места совместно, и на 12bb с
    пятерыми позади даёт им 6.6...7.9% комбо против 33.1% хедз-ап на той же
    глубине. Сводные числа модели расходятся соответственно: равновесие —
    `p_all_fold` 0.7156 при 0.3237 отвечающих, хедз-ап-раздача — 0.1554 при
    1.5544.
    """
    from harness.analysis.preflop import (
        _call_model_detail,
        _depth_key,
        _table_dead_bb,
        _table_equilibrium,
    )
    from harness.analysis.tools.pushfold import CallerModel, nash_hu

    en = _make_multiway_shove_hand(hero_cards=("Ad", "5d"), eff_bb=12.0, players_behind=5)
    dp = en.report.decision_points[0]
    state = table_state(dp, en)
    behind = [s for s in state.behind_hero if s.behind > 0]
    dead_bb = _table_dead_bb(state)
    depths = [min(state.hero.stack_after_ante, s.stack_after_ante) / en.hand.bb for s in behind]

    solved = list(_table_equilibrium(state, behind, en.hand.bb).calls)
    heads_up = [nash_hu(_depth_key(d), dead_extra_bb=dead_bb)[1] for d in depths]

    solved_widths = [rng.fraction_of_hands() for rng in solved]
    hu_widths = [rng.fraction_of_hands() for rng in heads_up]
    # Хедз-ап-раздача даёт всем одно число; решённое равновесие — разные, и все
    # они уже потому вне подозрения, что вдвое с лишним теснее.
    assert len({round(w, 4) for w in hu_widths}) == 1
    assert len({round(w, 4) for w in solved_widths}) > 1
    assert all(s < h / 2 for s, h in zip(solved_widths, hu_widths, strict=True))

    def callers(ranges):
        return [
            CallerModel(call_range=rng, behind_bb=s.behind / en.hand.bb, posted_bb=0.0)
            for s, rng in zip(behind, ranges, strict=True)
        ]

    assert _call_model_detail(callers(solved), "A5s") == {
        "p_all_fold": pytest.approx(0.7156, abs=5e-5),
        "expected_callers": pytest.approx(0.3237, abs=5e-5),
    }
    assert _call_model_detail(callers(heads_up), "A5s") == {
        "p_all_fold": pytest.approx(0.1554, abs=5e-5),
        "expected_callers": pytest.approx(1.5544, abs=5e-5),
    }


def test_an_interval_across_zero_gets_the_near_zero_verdict_with_a_ceiling():
    """Q2o на 10bb в двоих позади: интервал EV лежит по обе стороны нуля.

    Вердикт здесь есть, но не точечный: против самого тесного поля полосы шов
    мусором прибылен за счёт фолд-эквити, против равновесного — убыточен.
    Игроку называются знак и
    порядок величины (`point_bb`), сам интервал и потолок цены; упрёка нет,
    потому что оба варианта допустимы.
    """
    en = _make_multiway_shove_hand(hero_cards=("Qd", "2c"), eff_bb=10.0, players_behind=2)
    p = analyze_hand(en).points[0]

    assert p.spot == "pushfold_unopened"
    assert "unjudged" not in p.detail  # вердикт есть, отказа нет
    assert p.best_action == "около нуля, оба варианта допустимы"
    assert p.ev_diff_bb == 0.0
    assert p.detail["bracket"] == "unstable"

    interval = p.interval
    assert interval is not None and interval.near_zero is True
    assert interval.low_bb < 0.0 < interval.high_bb
    assert interval.low_bb <= interval.point_bb <= interval.high_bb
    # Потолок ограничивает цену ошибки в ЛЮБУЮ сторону: пас стоит не больше
    # верхнего конца, вход — не больше модуля нижнего.
    assert interval.cost_ceiling_bb == max(abs(interval.low_bb), interval.high_bb)
    values = list(p.detail["ev_shove_by_width_bb"].values())
    assert min(values) < 0.0 < max(values)


def test_an_interval_across_zero_facing_a_shove_gets_the_near_zero_verdict():
    """То же правило на другом споте: колл против шова, модели диапазона шовера.

    Глубина здесь 5bb, а не 12: на калиброванной полосе диапазона шовера KTs
    против шова на 12bb стал обычным вердиктом «фолд» — интервал перестал
    пересекать ноль. Форма «около нуля» на этом споте никуда не делась, она
    просто переехала туда, где решение и правда пограничное.
    """
    en = _make_facing_shove_hand(hero_cards=("Kd", "Ts"), eff_bb=5.0, shover_bb=5.0)
    p = analyze_hand(en).points[0]

    assert p.spot == "pushfold_facing_shove"
    assert "unjudged" not in p.detail
    assert p.best_action == "около нуля, оба варианта допустимы"
    assert p.ev_diff_bb == 0.0
    assert p.interval is not None and p.interval.near_zero is True
    assert p.interval.low_bb < 0.0 < p.interval.high_bb


def test_the_near_zero_form_is_refused_to_an_interval_that_is_too_wide():
    """Широкий интервал через ноль — не «около нуля», а отсутствие ответа.

    Форма «около нуля» обещает игроку, что выбор дёшев («не больше столько-то»).
    88 против шова на 12bb даёт интервал шире 3bb: знак не установлен, порядок
    величины не установлен, и то же обещание здесь было бы неверным. Такая точка
    возвращается без вердикта — числа остаются в `detail`, игроку не показывается
    ничего. Порог взят из разрыва в замеренном распределении ширин
    (`_NEAR_ZERO_MAX_WIDTH_BB`), а узкий интервал того же спота форму сохраняет
    (тест выше).
    """
    from harness.analysis.preflop import _NEAR_ZERO_MAX_WIDTH_BB

    en = _make_facing_shove_hand(hero_cards=("8d", "8c"), eff_bb=12.0, shover_bb=12.0)
    p = analyze_hand(en).points[0]

    assert p.best_action == ""  # вердикта нет
    assert p.ev_diff_bb == 0.0
    reason = str(p.detail["unjudged"])
    assert "расчёт не говорит ничего" in reason
    # Причина называет саму величину расхождения моделей, а не только факт.
    # Размах берётся по тем же числам, что и интервал: сетка ширин плюс сама
    # модель (`_interval_of`) — у снятой точки `interval` уже нет.
    values = [*p.detail["ev_call_by_width_bb"].values(), p.detail["ev_call_bb"]]
    width = max(values) - min(values)
    assert width > _NEAR_ZERO_MAX_WIDTH_BB
    assert f"{width:.1f} bb" in reason


def test_a_wide_interval_on_one_side_of_zero_keeps_its_point_verdict():
    """АА в неоткрытый банк: интервал широкий, но целиком плюсовой — вердикт обычный.

    Правило показа стоит по ЗНАКУ интервала, а не по его ширине: пока обе
    границы по одну сторону нуля, разброс по моделям колла вердикта не трогает
    и формы «около нуля» не включает.
    """
    en = _make_multiway_shove_hand(hero_cards=("Ac", "As"), eff_bb=12.0, players_behind=3)
    p = analyze_hand(en).points[0]

    assert p.best_action == "shove"
    interval = p.interval
    assert interval is not None and interval.near_zero is False
    assert interval.low_bb > 0.0
    # Ширина интервала здесь больше bb — и всё равно ни на что не влияет.
    assert interval.high_bb - interval.low_bb > 1.0


# --- Квантование глубины --------------------------------------------------------


@pytest.mark.slow
def test_depth_grid_does_not_move_ev_past_reporting_threshold():
    """Полшага сетки глубин двигает EV меньше, чем порог показа расхождения (0.1bb).

    Сетка нужна ради кэша равновесий, но платить за неё вердиктом нельзя: если
    квантование двигает EV на величину порядка порога, рука попадала бы в сводку
    или выпадала из неё в зависимости от округления.
    """
    from harness.analysis.preflop import _DEPTH_STEP_BB, _model_equity
    from harness.analysis.tools.pushfold import CallerModel, nash_hu, shove_ev_bb

    half = _DEPTH_STEP_BB / 2
    worst = 0.0
    for eff in (3.0, 8.0, 14.0):
        for cls in ("32o", "K9o", "QTo"):

            def ev(depth: float, eff: float = eff, cls: str = cls) -> float:
                caller = CallerModel(
                    call_range=nash_hu(depth)[1], behind_bb=eff - 1.0, posted_bb=1.0
                )
                return shove_ev_bb(
                    cls, eff - 0.5, 1.5, [caller], hero_posted_bb=0.5, equity_fn=_model_equity
                )

            base = ev(eff)
            worst = max(worst, abs(ev(eff + half) - base), abs(ev(eff - half) - base))
    assert worst < 0.05, worst


# --- Вилка обязана накрывать саму модель ----------------------------------------


def test_zone_is_assuming_when_model_disagrees_with_the_whole_interval():
    """Вердикт, которого не подтверждает ни одна ширина вилки, `strict` быть не может.

    Найдено прогоном по реальной фикстуре: вилка была собрана как модель
    КОЛЛ-диапазона (top-40% на широком конце), а против шова моделью служит
    пуш-сторона равновесия — с анте это около 70% комбо, то есть за краем вилки.
    Концы совпадали между собой, противореча самому выданному вердикту.

    Сетка ширин теперь доходит до 100% и модель заведомо накрывает, поэтому такой
    случай на реальных руках больше не возникает — но правило остаётся как
    защита: сетку могут однажды сузить.
    """
    z, why = zone_for("fold", "fold", live_total=5, best_model="call", best_interior=("fold",))
    assert z == "assuming" and "модели" in why


def test_zone_for_requires_model_to_agree_with_bracket():
    assert zone_for("shove", "shove", live_total=5, best_model="shove")[0] == "strict"
    z, why = zone_for("fold", "fold", live_total=5, best_model="call")
    assert z == "assuming" and "call" in why


def test_hero_already_committed_is_not_a_pushfold_spot():
    """Герой заколлировал опен, затем шов и ре-шов — это не пуш-фолд-точка.

    Найдено прогоном по реальной фикстуре (TM6292927496): точка оценивалась
    моделью «колл против диапазона открытого шова», хотя ре-шов после опена и
    колла — диапазон совсем другой ширины. Цена расхождения выходила -11bb, и
    это было бы самой громкой цифрой всего разбора.
    """
    stacks = {**dict.fromkeys(_SIX_MAX_SEATS, 200), "UTG": 42, "HJ": 40, "BB": 11}
    labels, seats, posts = _six_max(stacks, "HJ")
    open_raise = RawAction(
        street=Street.PREFLOP,
        label=labels["UTG"],
        kind=ActionKind.RAISE,
        amount=4,
        to_amount=4,
        raw_line="UTG: raises 2 to 4",
    )
    actions = [
        open_raise,
        _call("Hero", 4),
        _fold(labels["CO"]),
        _fold(labels["BTN"]),
        _fold(labels["SB"]),
        _shove(labels["BB"], 11, already=_BB),
        _shove(labels["UTG"], 42, already=4),
        _fold("Hero"),
    ]
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["9h", "9d"]}
    )
    en = enrich(normalize(raw))
    points = analyze_hand(en).points
    assert len(points) == 2
    # вторая точка: доплата съедает весь стек героя, но он уже вложился на этой улице
    assert points[1].spot == "preflop_other"
    assert points[1].best_action == "" and points[1].ev_diff_bb == 0.0


# --- Границы применимости модели «колл против диапазона шова» -------------------


def test_reshove_over_an_open_is_not_priced():
    """Шов ПОВЕРХ чужого опена — не открытый шов, и его диапазон нам неизвестен.

    Найдено прогоном по фикстуре (TM6292927967): 926ffe96 открывает рейзом 2bb,
    малый блайнд ре-шовит 12bb, Hero в BB пасует ATo. Модель брала пуш-сторону
    равновесия на 12bb (около 53% комбо) — диапазон ОТКРЫТОГО шова. Реальный
    ре-шов поверх опена вчетверо уже, и точка выходила «пас стоил 5.15bb».
    """
    stacks = {**dict.fromkeys(_SIX_MAX_SEATS, 400), "SB": 24, "BB": 30}
    labels, seats, posts = _six_max(stacks, "BB")
    open_raise = RawAction(
        street=Street.PREFLOP,
        label=labels["UTG"],
        kind=ActionKind.RAISE,
        amount=4,
        to_amount=4,
        raw_line="UTG: raises 2 to 4",
    )
    actions = [
        open_raise,
        _fold(labels["HJ"]),
        _fold(labels["CO"]),
        _fold(labels["BTN"]),
        _shove(labels["SB"], 24, already=_SB),
        _fold("Hero"),
    ]
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["Tc", "Ad"]}
    )
    p = analyze_hand(enrich(normalize(raw))).points[0]
    assert p.spot == "preflop_other" and p.best_action == ""
    assert "шов" in p.detail["unjudged"] or "олл-ин" in p.detail["unjudged"]


def test_two_all_ins_before_hero_are_not_priced():
    """Два олл-ина перед героем — сайд-поты и вскрытие на троих, а модель считает пару.

    `call_shove_ev_bb` меряет эквити против ОДНОГО диапазона. При двух уже
    вложившихся всё эквити героя завышено, и завышение идёт в сторону колла —
    самое опасное направление. Такую точку v1 не оценивает.
    """
    stacks = {**dict.fromkeys(_SIX_MAX_SEATS, 400), "UTG": 24, "HJ": 10, "BB": 26}
    labels, seats, posts = _six_max(stacks, "BB")
    actions = [
        _shove(labels["UTG"], 24),  # открытый шов — агрессор он же и открыл банк
        _call(labels["HJ"], 10, all_in=True),  # второй олл-ин, короче первого
        _fold(labels["CO"]),
        _fold(labels["BTN"]),
        _fold(labels["SB"]),
        _fold("Hero"),
    ]
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["Tc", "Ad"]}
    )
    p = analyze_hand(enrich(normalize(raw))).points[0]
    assert p.spot == "preflop_other" and p.best_action == ""


def test_a_player_who_already_called_the_shove_is_not_priced():
    """Шов, ответ на него, и только потом Hero: на вскрытии два диапазона, а не один.

    Заколлировавший — не развилка «войдёт или нет»: он уже вложился, и модельный
    диапазон КОЛЛА ему приписывать нечему. Прежние гейты этот спот пропускали, и
    тест это показывает: олл-ин в руке ровно один (заколлировавший покрыл шов и
    остался с фишками), банк открыт тем же, кто и поставил.
    """
    stacks = {**dict.fromkeys(_SIX_MAX_SEATS, 60), "UTG": 20, "BB": 26}
    labels, seats, posts = _six_max(stacks, "BB")
    actions = [
        _shove(labels["UTG"], 20),
        _call(labels["HJ"], 20),  # покрыл шов и остался с фишками — не олл-ин
        _fold(labels["CO"]),
        _fold(labels["BTN"]),
        _fold(labels["SB"]),
        _fold("Hero"),
    ]
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["Tc", "Ad"]}
    )
    en = enrich(normalize(raw))
    dp = en.report.decision_points[0]
    state = table_state(dp, en)
    # Оба прежних гейта пропускают эту точку — ловит её именно новый.
    assert len(state.all_in_besides_hero) == 1
    assert state.opened_by_aggressor
    assert {s.label for s in state.callers_before_hero} == {labels["HJ"]}

    p = analyze_hand(en).points[0]
    assert p.spot == "preflop_other" and p.best_action == "" and p.ev_diff_bb == 0.0
    # Подстрока выбрана так, чтобы её не давала соседняя причина про два олл-ина.
    assert "ответ на него" in p.detail["unjudged"]


def test_a_blind_all_in_behind_hero_is_not_priced():
    """Живой без фишек за спиной позади героя — вердикта нет, а не ярлык.

    BB со стеком в один блайнд уходит в олл-ин самим постом: он жив, фишек за
    спиной нет, действия в круге у него не было. Коллером он быть не может —
    коллировать нечем; но и посчитать точку не из чего: `call_shove_ev_bb` берёт
    эквити против одного диапазона и весь банк записывает герою.

    Открывший рейз НЕ в олл-ин выбран намеренно: только так олл-ин в руке
    остаётся один и точка доходит до `_facing_shove_verdict`, а не отсекается
    гейтом класса (это проверяет соседний тест).
    """
    stacks = {**dict.fromkeys(_SIX_MAX_SEATS, 40), "BB": _BB, "SB": 5}
    labels, seats, posts = _six_max(stacks, "SB")
    open_raise = RawAction(
        street=Street.PREFLOP,
        label=labels["UTG"],
        kind=ActionKind.RAISE,
        amount=6,
        to_amount=6,
        raw_line="UTG: raises 4 to 6",
    )
    actions = [
        open_raise,
        _fold(labels["HJ"]),
        _fold(labels["CO"]),
        _fold(labels["BTN"]),
        _fold("Hero"),
    ]
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["Tc", "Ad"]}
    )
    en = enrich(normalize(raw))
    # Рум пас за него не писал — это не форфейт, а настоящий олл-ин с блайнда.
    assert en.report.forfeits == []
    dp = en.report.decision_points[0]
    state = table_state(dp, en)
    blind = next(s for s in state.seats if s.position == "BB")
    assert blind.live and not blind.acted and blind.behind == 0

    # (1) он считается олл-ином помимо героя;
    assert {s.label for s in state.all_in_besides_hero} == {blind.label}
    # (2) класс спота при этом остаётся пуш-фолдным — отсекает не гейт класса,
    #     а сам расчёт, и причина называется его словами;
    assert state.aggressor is not None and not state.aggressor_all_in
    assert spot_for(dp, state) == "pushfold_facing_shove"
    # (3) вердикта нет, и он не подменён ни ярлыком зоны, ни числом.
    p = analyze_hand(en).points[0]
    assert p.spot == "pushfold_facing_shove"
    assert p.best_action == "" and p.ev_diff_bb == 0.0 and p.assumption is None
    assert "без фишек за спиной" in p.detail["unjudged"]
    assert "live_others" not in p.detail


def test_a_blind_all_in_behind_the_hero_shove_is_not_priced():
    """Живой без фишек за спиной позади шова героя — вердикта нет, а не ярлык зоны.

    Банк неоткрыт, Hero на UTG, большой блайнд отдал посту весь стек: он жив,
    фишек за спиной нет, действия в круге у него не было. Тест проверяет, что
    вердикта по такой точке нет вовсе — ни действия, ни цены, ни допущения:
    та же граница, что у колла шова
    (`test_a_blind_all_in_behind_hero_is_not_priced`).
    """
    stacks = {**dict.fromkeys(_SIX_MAX_SEATS, 20), "BB": _BB}
    labels, seats, posts = _six_max(stacks, "UTG")
    actions = [
        _fold("Hero"),
        _fold(labels["HJ"]),
        _fold(labels["CO"]),
        _fold(labels["BTN"]),
        _fold(labels["SB"]),
    ]
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["7c", "2d"]}
    )
    en = enrich(normalize(raw))
    # Пас за него рум не писал — это настоящий олл-ин с блайнда, не форфейт.
    assert en.report.forfeits == []
    dp = en.report.decision_points[0]
    state = table_state(dp, en)
    blind = next(s for s in state.behind_hero if s.position == "BB")
    assert blind.live and not blind.acted and blind.behind == 0
    assert spot_for(dp, state) == "pushfold_unopened"

    p = analyze_hand(en).points[0]
    assert p.spot == "pushfold_unopened"
    assert p.best_action == "" and p.ev_diff_bb == 0.0 and p.assumption is None
    assert "без фишек за спиной" in p.detail["unjudged"]
    assert "all_in_behind_ignored" not in p.detail


def test_shove_plus_blind_all_in_is_not_priced():
    """Шов и олл-ин с блайнда — два олл-ина помимо героя: гейт класса обязан отсечь.

    Ровно тот спот, который гейт `len(all_in_besides_hero) <= 1` создан
    исключать; он проходил его, пока свойство требовало состоявшегося действия,
    а у олл-ина с блайнда действия нет.
    """
    stacks = {**dict.fromkeys(_SIX_MAX_SEATS, 60), "UTG": 20, "SB": _SB, "BB": 26}
    labels, seats, posts = _six_max(stacks, "BB")
    actions = [
        _shove(labels["UTG"], 20),
        _fold(labels["HJ"]),
        _fold(labels["CO"]),
        _fold(labels["BTN"]),
        _fold("Hero"),
    ]
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["Tc", "Ad"]}
    )
    en = enrich(normalize(raw))
    assert en.report.forfeits == []  # пас за малый блайнд рум не писал
    dp = en.report.decision_points[0]
    state = table_state(dp, en)
    assert {s.position for s in state.all_in_besides_hero} == {"UTG", "SB"}

    p = analyze_hand(en).points[0]
    assert p.spot == "preflop_other" and p.best_action == ""
    assert "больше одного олл-ина помимо героя (всего 2)" in p.detail["unjudged"]


# --- Форфейт: место, которое движок вычёркивает из руки ---------------------------


def _make_forfeit_hand(hero_position: str):
    """Шов UTG; место с нулевым стеком в блайнде, которому рум записал `folds`.

    Что это за форма и почему движок снимает такое место с руки — в докстрингах
    `replay._is_forfeit` и `replay._forfeit`. Здесь важно только, что `enrich`
    её распознаёт: это проверяет ассерта в конце функции.

    Позиция героя выбирает, до или после его решения стоит строка `folds`
    форфейта: при Hero в BB форфейтит SB и его пас попадает в действия ДО
    решения героя, при Hero в HJ форфейтит BB и его пас стоит после.
    """
    blind = "SB" if hero_position == "BB" else "BB"
    amount = _SB if blind == "SB" else _BB
    stacks = {**dict.fromkeys(_SIX_MAX_SEATS, 60), "UTG": 20, blind: amount}
    stacks[hero_position] = 26
    labels, seats, posts = _six_max(stacks, hero_position)
    order = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
    actions = [_shove(labels["UTG"], 20)]
    actions += [_fold(labels[pos]) for pos in order[1:] if pos != "UTG"]
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["Tc", "Ad"]}
    )
    en = enrich(normalize(raw))
    assert en.report.forfeits == [labels[blind]]
    return en, labels, blind


def test_a_forfeited_seat_behind_hero_is_not_an_all_in():
    """Форфейт позади героя: движок снял его с руки, значит и стол его не держит.

    Строка `folds` форфейтного BB стоит ПОСЛЕ решения героя, поэтому проход по
    действиям до точки решения её не видит — место остаётся живым, если не
    прочитать `report.forfeits`. Тогда оно идёт и в олл-ины помимо героя (спот
    снимается с оценки с причиной, которой в руке нет), и в моделируемые
    коллеры (равновесный диапазон колла игроку, который в руке не участвует).
    """
    en, labels, blind = _make_forfeit_hand("HJ")
    dp = next(d for d in en.report.decision_points if d.label == "Hero")
    state = table_state(dp, en)
    seat = next(s for s in state.seats if s.label == labels[blind])
    assert not seat.live and seat.behind == 0

    assert {s.label for s in state.all_in_besides_hero} == {labels["UTG"]}
    p = analyze_hand(en).points[0]
    # Точка дошла до модели колла шова: если бы форфейтное место считалось живым
    # олл-ином, спот снялся бы с оценки раньше и `method` бы не появился.
    assert p.spot == "pushfold_facing_shove" and p.detail["method"] == "call_ev"
    # CO, BTN и SB — все живые за героем помимо шовера; форфейтного BB среди них нет.
    assert p.detail["live_others"] == 3


def test_a_forfeited_seat_before_hero_is_not_an_all_in():
    """Тот же форфейт, но его пас стоит ДО решения героя — держит уже фильтр `live`.

    Здесь `table_state` выводит место из живых обычным проходом по действиям, и
    закрепляется именно `live` в `all_in_besides_hero`: без него место с нулевым
    остатком считалось бы вторым олл-ином и спот ушёл бы без вердикта.
    """
    en, labels, blind = _make_forfeit_hand("BB")
    dp = next(d for d in en.report.decision_points if d.label == "Hero")
    state = table_state(dp, en)
    seat = next(s for s in state.seats if s.label == labels[blind])
    assert seat.acted and not seat.live and seat.behind == 0

    assert {s.label for s in state.all_in_besides_hero} == {labels["UTG"]}
    assert spot_for(dp, state) == "pushfold_facing_shove"


def test_live_total_of_the_table_leaves_out_a_forfeited_seat():
    """Состав стола считается один раз и без форфейта; движковое число его держит.

    `dp.live_total` снят реплеем в момент решения героя, а строка `folds`
    форфейтного места стоит позже, поэтому движок в этот момент считает его
    живым. `table_state` читает `report.forfeits` и не считает — обе величины
    тест сравнивает на одной точке.
    """
    en, labels, blind = _make_forfeit_hand("HJ")
    dp = next(d for d in en.report.decision_points if d.label == "Hero")
    state = table_state(dp, en)
    assert dp.live_total == 6
    assert state.live_total == 5
    assert labels[blind] in state.forfeits


def test_a_forfeited_small_blind_does_not_open_the_heads_up_equilibrium():
    """Двое живых после вычета форфейта — ещё не та игра, для которой есть равновесие.

    Hero на BTN, малый блайнд отдал стек посту и получил от рума `folds` (его
    строка стоит после решения героя), большой блайнд жив. Живых помимо
    форфейта двое, и один из них в BB — форма, на которой гейт равновесия
    открывался бы, — но малый блайнд вложил в банк больше своего анте. Зона
    обязана определяться вилкой, а не равновесием.
    """
    stacks = {**dict.fromkeys(_SIX_MAX_SEATS, 20), "SB": _SB}
    labels, seats, posts = _six_max(stacks, "BTN")
    actions = [
        _fold(labels["UTG"]),
        _fold(labels["HJ"]),
        _fold(labels["CO"]),
        _fold("Hero"),
        _fold(labels["SB"]),
    ]
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["7c", "2d"]}
    )
    en = enrich(normalize(raw))
    assert en.report.forfeits == [labels["SB"]]
    dp = en.report.decision_points[0]
    state = table_state(dp, en)
    # Форма, на которой гейт равновесия и открывался бы: живых двое, позади один
    # и он в большом блайнде.
    assert dp.live_total == 3 and state.live_total == 2
    assert [s.position for s in state.behind_hero] == ["BB"]

    p = analyze_hand(en).points[0]
    assert p.spot == "pushfold_unopened"
    assert "равновеси" not in p.detail["zone_reason"]
    # Дешёвый лукап закрывает только равновесную форму; здесь гейт закрыт, и он
    # отдаёт точку полному расчёту вместо того, чтобы назвать её равновесием.
    assert cheap_fold_verdict(dp, en) is None


def test_an_ante_only_forfeit_does_not_open_the_heads_up_equilibrium():
    """Тот же гейт, но мёртвых денег сверх анте в банке нет — держит проверка героя.

    Малому блайнду хватило стека ровно на анте, поэтому блайнд с него не взяли
    (`forced_blind` = 0) и его вклад равен его анте: условие «кроме героя и
    названного места никто не вложил больше своего анте» выполняется. Живых
    помимо форфейта двое, позади один и он в BB — гейт открылся бы, хотя Hero
    сидит на кнопке и вынужденной ставки не делал вовсе.

    Это единственная известная форма, где `dp.live_total` и `state.live_total`
    расходятся при открытом гейте: тем же тестом закреплена и замена одного
    счётчика на другой.
    """
    stacks = {**dict.fromkeys(_SIX_MAX_SEATS, 24), "SB": 1}
    labels, seats, posts = _six_max(stacks, "BTN", ante=1)
    actions = [
        _fold(labels["UTG"]),
        _fold(labels["HJ"]),
        _fold(labels["CO"]),
        _shove("Hero", 23),
        _fold(labels["SB"]),
        _fold(labels["BB"]),
    ]
    raw = _raw(
        seats=seats,
        button_seat=6,
        posts=posts,
        actions=actions,
        dealt={"Hero": ["7c", "2d"]},
        ante=1,
    )
    en = enrich(normalize(raw))
    assert en.verdict.status == "pass" and en.report.forfeits == [labels["SB"]]
    dp = en.report.decision_points[0]
    state = table_state(dp, en)
    small = next(s for s in state.seats if s.position == "SB")
    assert small.contributed == small.ante and not small.live
    assert state.hero.contributed == state.hero.ante
    assert dp.live_total == 3 and state.live_total == 2
    assert [s.position for s in state.behind_hero] == ["BB"]

    p = analyze_hand(en).points[0]
    assert p.spot == "pushfold_unopened"
    assert "равновеси" not in p.detail["zone_reason"]
    # Гейт закрыт — значит интервал вердикт не держит, и точечной оценки у точки
    # нет, а есть форма «около нуля». Было бы открыто равновесие — вердикт бы остался.
    assert p.detail["bracket"] == "unstable"
    assert p.best_action == "около нуля, оба варианта допустимы"


def test_a_dead_small_blind_is_not_the_heads_up_equilibrium_against_a_shove():
    """Шов кнопки после паса малого блайнда — не та игра, что считает равновесие.

    Живых двое, герой в большом блайнде, ставку поставила кнопка — гейт
    равновесия по форме открыт. Но блайнд спасовавшего лежит в банке, а
    `_table_dead_bb` складывает только анте, то есть мёртвые деньги переданы не
    те. Зона обязана определяться вилкой.
    """
    stacks = {**dict.fromkeys(_SIX_MAX_SEATS, 60), "BTN": 20, "BB": 20}
    labels, seats, posts = _six_max(stacks, "BB")
    actions = [
        _fold(labels["UTG"]),
        _fold(labels["HJ"]),
        _fold(labels["CO"]),
        _shove(labels["BTN"], 20),
        _fold(labels["SB"]),
        _fold("Hero"),
    ]
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["Tc", "9d"]}
    )
    en = enrich(normalize(raw))
    dp = en.report.decision_points[0]
    state = table_state(dp, en)
    small = next(s for s in state.seats if s.position == "SB")
    assert small.contributed > small.ante and not small.live
    assert state.live_total == 2 and state.hero.position == "BB"

    p = analyze_hand(en).points[0]
    assert p.spot == "pushfold_facing_shove"
    # Проверяемое здесь — ЧЕМ поставлена зона, а не какая она вышла: гейт
    # равновесия закрыт, и причина обязана прийти от вилки ширин.
    assert "равновеси" not in p.detail["zone_reason"]
    assert "ни на одной ширине" in p.detail["zone_reason"]
    # На калиброванной полосе диапазона шовера вердикт этой точки перестал
    # зависеть от ширины: прежде вилка была неустойчива (полоса доходила до
    # диапазона «любые две карты»), теперь «фолд» стоит на всех пяти точках.
    assert p.detail["bracket"] == "stable"
    assert p.best_action == "fold"


def test_a_forfeited_seat_could_not_have_answered_the_shove():
    """Состав на момент шова тоже без форфейта: движок вычеркнул место из руки.

    Фильтр `_rivals_when_shoved` знает только пасы и олл-ины, записанные ДО
    шова, а `folds` форфейта стоит после решения героя — без чтения форфейтов
    место остаётся в наборе тех, кто мог ответить на шов.
    """
    en, labels, blind = _make_forfeit_hand("HJ")
    dp = next(d for d in en.report.decision_points if d.label == "Hero")
    state = table_state(dp, en)
    shover = _shover(state)
    assert shover is not None and shover.label == labels["UTG"]
    rivals = {s.label for s in _rivals_when_shoved(en.hand, dp, state, shover)}
    assert labels[blind] not in rivals
    assert rivals == {"Hero", labels["CO"], labels["BTN"], labels["SB"]}


def test_a_limped_pot_is_not_a_shove():
    """Банк открыт лимпом: повышавшего нет, и диапазон шова приписывать некому.

    Доплата съедает весь остаток героя, поэтому `call_is_all_in` истинно и точка
    выглядела пуш-фолдной, хотя `state.aggressor` здесь `None` — обе величины
    тест проверяет до вердикта.
    """
    stacks = {**dict.fromkeys(_SIX_MAX_SEATS, 60), "SB": _BB}
    labels, seats, posts = _six_max(stacks, "SB")
    actions = [
        _call(labels["UTG"], _BB),  # лимп
        _fold(labels["HJ"]),
        _fold(labels["CO"]),
        _fold(labels["BTN"]),
        _fold("Hero"),
    ]
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["Tc", "Ad"]}
    )
    en = enrich(normalize(raw))
    dp = en.report.decision_points[0]
    state = table_state(dp, en)
    assert state.aggressor is None
    assert state.opened_voluntarily and state.call_is_all_in

    p = analyze_hand(en).points[0]
    assert p.spot == "preflop_other" and p.best_action == ""
    # Подстрока целиком: слово «лимп» есть и в причине про НЕоткрытый банк.
    assert "банк открыт лимпом" in p.detail["unjudged"]


def test_a_limped_pot_names_the_limp_when_the_call_is_not_all_in():
    """Лимп, колл героя, фишки за спиной остались: причина называет лимп.

    `aggressor` здесь `None`, а `call_is_all_in` — False; обе величины тест
    проверяет до вердикта. В `unpriced_reason` лимповая ветка стоит выше ветки
    «банк открыт рейзом не в олл-ин», и порядок закрепляет именно эта ассерта:
    при обратном порядке точку забрала бы вторая.
    """
    stacks = dict.fromkeys(_SIX_MAX_SEATS, 20)
    labels, seats, posts = _six_max(stacks, "SB")

    def _check(label: str, street: Street) -> RawAction:
        return RawAction(
            street=street, label=label, kind=ActionKind.CHECK, raw_line=f"{label}: checks"
        )

    actions = [
        _call(labels["UTG"], _BB),  # лимп
        _fold(labels["HJ"]),
        _fold(labels["CO"]),
        _fold(labels["BTN"]),
        _call("Hero", _SB),  # доплата 1 при остатке 19 — олл-ином колл не был
        _check(labels["BB"], Street.PREFLOP),
    ]
    # Рука доигрывается чеками до вскрытия: лимпленный банк всегда видит флоп,
    # а оборванная раздача отвергается валидатором.
    for street in (Street.FLOP, Street.TURN, Street.RIVER):
        actions += [_check("Hero", street), _check(labels["BB"], street), _check(labels["UTG"], street)]
    raw = _raw(
        seats=seats,
        button_seat=6,
        posts=posts,
        actions=actions,
        dealt={"Hero": ["Tc", "Ad"]},
        boards=_BOARD,
        showdowns=[
            _showdown("Hero", ["Tc", "Ad"]),
            _showdown(labels["BB"], ["7c", "2s"]),
            _showdown(labels["UTG"], ["9s", "9h"]),
        ],
    )
    en = enrich(normalize(raw))
    assert en.verdict.status == "pass"
    dp = en.report.decision_points[0]
    state = table_state(dp, en)
    assert state.aggressor is None
    assert state.opened_voluntarily and not state.call_is_all_in

    p = analyze_hand(en).points[0]
    assert p.spot == "preflop_other" and p.best_action == ""
    assert "банк открыт лимпом" in p.detail["unjudged"]


# --- Глубина шовера: только те, кто мог ему ответить ------------------------------


def _make_deep_folder_before_the_shove():
    """UTG (60bb) пасует, ЗАТЕМ HJ (12bb) шовит; все живые после него мельче него.

    Форма подобрана так, чтобы состав оппонентов вообще влиял на число: `min`
    не упирается в стек шовера (он глубже всех, кто остался жив), а сброшенный
    ДО шова UTG глубже самого шовера. Состав «все места, кроме шовера» выбрал бы
    UTG и дал бы 12.0bb — стек самого шовера; состав «живые на момент шова» даёт
    9.0bb по самому глубокому из тех, кто ещё мог заколлировать.
    """
    stacks = {"UTG": 120, "HJ": 24, "CO": 18, "BTN": 18, "SB": 18, "BB": 16}
    labels, seats, posts = _six_max(stacks, "BB")
    actions = [
        _fold(labels["UTG"]),
        _shove(labels["HJ"], 24),
        _fold(labels["CO"]),
        _fold(labels["BTN"]),
        _fold(labels["SB"]),
        _fold("Hero"),
    ]
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["Tc", "Ad"]}
    )
    return enrich(normalize(raw))


def test_shover_depth_counts_only_those_live_when_he_shoved():
    """Стек сбросившегося ДО шова в глубину шова не входит.

    Заколлировать шов он не мог, и на диапазон, с которым шовер входил, его
    стек не влиял. Проверяется само число глубины, а не только класс спота:
    класс дают и другие гейты, а разъезжается здесь именно глубина — 9.0bb
    против 12.0bb.
    """
    p = analyze_hand(_make_deep_folder_before_the_shove()).points[0]
    assert p.spot == "pushfold_facing_shove"
    assert p.detail["shover_depth_bb"] == 9.0


def test_shover_depth_ignores_a_player_already_all_in_when_the_shove_landed():
    """Уже стоявший в олл-ине жив, но выбора «коллировать или пас» у него нет.

    Глубина здесь берётся ровно ради диапазона шова, то есть ради того, против
    чьего выбора шовер ставил, — и такого игрока в составе быть не должно.

    Проверяется состав, а не итоговое число: на числе исключение здесь ничего не
    меняет, в составе же ошибка видна, а в максимуме — нет.

    Спот при этом остаётся без вердикта, и причин на то сразу две: перед героем
    два олл-ина, а шов сыгран поверх чужого олл-ина, а не открыт. Поэтому через
    `analyze_hand` состав не проверить вовсе.
    """
    from harness.analysis.preflop import _rivals_when_shoved, _shover

    stacks = {"UTG": 22, "HJ": 24, "CO": 40, "BTN": 10, "SB": 10, "BB": 10}
    labels, seats, posts = _six_max(stacks, "BB")
    actions = [
        _shove(labels["UTG"], 22),  # олл-ин ДО шова
        _shove(labels["HJ"], 24),  # шов, на который отвечает Hero
        _fold(labels["CO"]),
        _fold(labels["BTN"]),
        _fold(labels["SB"]),
        _fold("Hero"),
    ]
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["Tc", "Ad"]}
    )
    en = enrich(normalize(raw))
    # Реплей не отверг ре-шов: сама форма руки законна, а на нехватку карт борда
    # (синтетика их не даёт, а два олл-ина ведут к вскрытию) префлоп не опирается.
    assert not [x for x in en.report.illegal_actions if "raises" in x]
    dp = en.report.decision_points[0]
    state = table_state(dp, en)
    shover = _shover(state)
    assert shover is not None and shover.label == labels["HJ"]

    rivals = _rivals_when_shoved(en.hand, dp, state, shover)
    assert {seat.label for seat in rivals} == {labels["CO"], labels["BTN"], labels["SB"], "Hero"}
    assert labels["UTG"] not in {seat.label for seat in rivals}

    assert analyze_hand(en).points[0].spot == "preflop_other"


# --- Диапазон шовера: равновесие ЕГО стола, а не хедз-ап -------------------------


def _shover_solution(en):
    """Решение подыгры шовера по руке — тем же путём, которым его берёт вердикт."""
    dp = en.report.decision_points[0]
    state = table_state(dp, en)
    shover = _shover(state)
    assert shover is not None
    rivals = _rivals_when_shoved(en.hand, dp, state, shover)
    posted = _posted_before_shove(en.hand, dp, shover)
    return state, shover, rivals, _shover_equilibrium(state, shover, posted, rivals, en.hand.bb)


@pytest.mark.parametrize("stack,ante", [(24, 0), (20, 0), (10, 0), (24, 1), (30, 1)])
def test_the_shover_range_with_one_player_behind_is_the_nash_push_range(stack, ante):
    """Опора: при ОДНОМ игроке позади шовера обе стороны обязаны совпасть с `nash_hu`.

    `nash_hu` — единственное место в системе, где ответ подтверждён независимо
    (якорные тесты против опубликованных чартов). Хедз-ап SB против BB — ровно
    та игра, в которую играл шовер, когда позади него сидел один герой, поэтому
    допуск здесь ноль по весу КАЖДОГО из 169 классов, а не «близко».
    """
    en = _make_hu_facing_shove_hand(stack, ante)
    state, shover, rivals, solution = _shover_solution(en)
    assert [seat.label for seat in rivals] == ["Hero"]

    eff_bb = min(shover.stack_after_ante, rivals[0].stack_after_ante) / en.hand.bb
    reference_push, reference_call = nash_hu(eff_bb, dead_extra_bb=_table_dead_bb(state))

    assert _max_weight_gap(solution.push, reference_push) == 0.0
    assert _max_weight_gap(solution.calls[0], reference_call) == 0.0


def test_the_shover_subgame_starts_before_his_own_shove():
    """Деньги подыгры берутся на ходе ШОВЕРА: без его шова ни в банке, ни в его посте.

    На решении героя шов уже лежит в банке, и взять `pot_before` с постами из
    того же снимка значило бы решать игру, в которой шовер платит второй раз.
    """
    en = _make_hu_facing_shove_hand(24)
    dp = en.report.decision_points[0]
    state = table_state(dp, en)
    shover = _shover(state)
    assert shover is not None

    posted = _posted_before_shove(en.hand, dp, shover)
    assert posted == _SB  # только малый блайнд: анте нет, до шова он не ходил
    assert shover.contributed == 24  # а в снимке героя он уже весь в банке
    assert state.pot_before == 24 + _BB  # банк на решении героя содержит шов


def test_the_shover_anchor_notices_the_pot_taken_after_the_shove():
    """Фальсификация опоры: с банком, взятым на решении героя, совпадения нет.

    Опора обязана ловить именно восстановление момента шова. Если бы она
    проходила и с банком, который шов уже содержит, она не проверяла бы ничего.
    """
    en = _make_hu_facing_shove_hand(24)
    state, shover, rivals, solution = _shover_solution(en)
    reference_push, _ = nash_hu(12.0)
    assert _max_weight_gap(solution.push, reference_push) == 0.0

    bb = en.hand.bb
    posted = _posted_before_shove(en.hand, en.report.decision_points[0], shover)
    spoiled = unopened_shove_equilibrium(
        MultiwaySeat(posted_bb=posted / bb, behind_bb=(shover.stack - posted) / bb),
        [
            MultiwaySeat(
                posted_bb=rivals[0].contributed / bb, behind_bb=rivals[0].behind / bb
            )
        ],
        state.pot_before / bb,
    )
    assert _max_weight_gap(spoiled.push, reference_push) > 0.0


def test_the_table_equilibrium_turns_a_heads_up_call_into_a_fold():
    """Направление сдвига: приписанный шоверу диапазон уже, и колл дешевеет.

    UTG шовит 12bb, позади него пятеро. Хедз-ап пуш-диапазон этой глубины —
    53.3% комбо, равновесие его стола — 13.4%; KQo против первого коллируется,
    против второго сбрасывается. Направление названо замером, а не рассуждением:
    обе цены считает один и тот же `call_shove_ev_bb`, меняется только диапазон.
    """
    stacks = {**dict.fromkeys(_SIX_MAX_SEATS, 96), "SB": 24, "UTG": 24}
    labels, seats, posts = _six_max(stacks, "SB")
    actions = [_shove(labels["UTG"], 24)]
    actions += [_fold(labels[pos]) for pos in ("HJ", "CO", "BTN")]
    actions += [_fold("Hero"), _fold(labels["BB"])]
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["Kd", "Qc"]}
    )
    en = enrich(normalize(raw))
    state, shover, rivals, solution = _shover_solution(en)
    assert len(rivals) == 5

    bb = en.hand.bb
    heads_up_push = nash_hu(
        _depth_key(min(shover.stack_after_ante, max(s.stack_after_ante for s in rivals)) / bb),
        dead_extra_bb=_table_dead_bb(state),
    )[0]

    def ev(rng):
        return call_shove_ev_bb(
            "KQo",
            state.hero.behind / bb,
            rng,
            state.pot_before / bb,
            state.to_call / bb,
            equity_fn=_model_equity,
        )

    assert round(heads_up_push.fraction_of_hands(), 4) == 0.5332
    assert round(solution.push.fraction_of_hands(), 4) == 0.1337
    assert ev(heads_up_push) > 0.0 > ev(solution.push)

    point = analyze_hand(en).points[0]
    assert point.detail["best_vs_one"] == "fold"
    assert point.detail["shove_range_fraction"] == round(solution.push.fraction_of_hands(), 6)
    assert point.detail["equilibrium_hand_regret_bb"] == round(solution.hand_regret_bb, 6)


def test_players_behind_hero_answer_the_same_shove_as_hero():
    """Колл-диапазоны живых за героем берутся из ТОГО ЖЕ решения, что и диапазон шова.

    Разрывать пару нельзя: колл-сторона является наилучшим ответом на шов того
    же решения. Проверяется поимённо — каждое место позади героя есть среди тех,
    кто отвечал на шов, и его ширина в `detail` совпадает с решением.
    """
    stacks = {**dict.fromkeys(_SIX_MAX_SEATS, 96), "SB": 24, "UTG": 24}
    labels, seats, posts = _six_max(stacks, "SB")
    actions = [_shove(labels["UTG"], 24)]
    actions += [_fold(labels[pos]) for pos in ("HJ", "CO", "BTN")]
    actions += [_call("Hero", 23, all_in=True), _fold(labels["BB"])]
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["Ah", "Ad"]}
    )
    en = enrich(normalize(raw))
    state, _, rivals, solution = _shover_solution(en)

    behind = [s for s in state.seats if s.live and s.label not in ("Hero", labels["UTG"])]
    assert [s.label for s in behind] == [labels["BB"]]
    rival_at = {seat.label: index for index, seat in enumerate(rivals)}
    assert set(rival_at) >= {seat.label for seat in behind}

    point = analyze_hand(en).points[0]
    assert point.detail["call_range_fractions"] == [
        round(solution.calls[rival_at[labels["BB"]]].fraction_of_hands(), 6)
    ]
    assert point.detail["rivals_when_shoved"] == len(rivals)


# --- Порядок мест на входе решателя ---------------------------------------------


def _order_test_table():
    """Стол, на котором порядок мест и порядок хода РАЗНЫЕ.

    Кнопка на месте 6, то есть места идут SB, BB, UTG, HJ, CO, BTN, а ход на
    префлопе — UTG, HJ, CO, BTN, SB, BB. Герой в CO шовит в неоткрытый банк,
    поэтому позади него BTN, SB и BB: по кругу это хвост, заворачивающийся через
    последнее место за стол, — по номеру места те же трое идут SB, BB, BTN.

    Остатки за спиной у всех троих разные (11.0, 9.5 и 12.0 bb) — по ним и
    видно, в каком порядке места ушли в решатель.
    """
    stacks = {"SB": 20, "BB": 26, "UTG": 40, "HJ": 40, "CO": 24, "BTN": 22}
    labels, seats, posts = _six_max(stacks, "CO")  # labels нужен для строк пасов
    actions = [_fold(labels["UTG"]), _fold(labels["HJ"]), _shove("Hero", stacks["CO"])]
    actions += [_fold(labels[pos]) for pos in ("BTN", "SB", "BB")]
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["Ah", "Kh"]}
    )
    return labels, enrich(normalize(raw))


@pytest.mark.slow  # настоящее решение равновесия на троих позади
def test_the_solver_gets_the_seats_behind_hero_in_action_order(monkeypatch):
    """Места позади героя уходят решателю в порядке ХОДА, а не в порядке мест.

    Решатель (`multiway.unopened_shove_equilibrium`) документирует места 1..N как
    «живых игроков позади в порядке хода» и суммирует EV шова так, что вес ветки
    «заколлировал именно j» зависит от того, кто ходит до него. Порядок мест из
    hand history — другой (проверяется здесь же), и подать его значило бы решать
    не ту игру.

    Фальсификация: вернуть `behind_hero` к порядку `self.seats` — записанные
    остатки станут (9.5, 12.0, 11.0), и тест покраснеет.
    """
    _, en = _order_test_table()
    dp = en.report.decision_points[0]
    state = table_state(dp, en)

    # порядок мест за столом — из строк Seat N, и он не порядок хода
    assert [seat.position for seat in state.seats] == list(_SIX_MAX_SEATS)
    assert [seat.position for seat in state.seats if seat.live and not seat.acted] != [
        "BTN",
        "SB",
        "BB",
    ]

    recorded: list[list[tuple[float, float]]] = []
    solve = preflop_module.unopened_shove_equilibrium

    def spy(hero, behind, pot_dead_bb):
        recorded.append([(seat.posted_bb, seat.behind_bb) for seat in behind])
        return solve(hero, behind, pot_dead_bb)

    monkeypatch.setattr(preflop_module, "unopened_shove_equilibrium", spy)
    verdict_for(dp, en)

    assert recorded, "решатель равновесия не вызывался — проверять нечего"
    # BTN (ничего не поставил, 11bb), SB (0.5bb поста, 9.5bb), BB (1bb поста, 12bb)
    assert recorded[0] == [(0.0, 11.0), (0.5, 9.5), (1.0, 12.0)]
    assert [seat.position for seat in state.behind_hero] == ["BTN", "SB", "BB"]


def test_the_rivals_of_the_shover_are_in_action_order_after_him():
    """Состав, отвечавший на чужой шов, тоже идёт в порядке хода — от шовера.

    Круг тот же самый, но крутится он от места шовера, а не от места героя:
    поэтому порядок задаёт общая `in_action_order_after`, а не `behind_hero`.
    Здесь шовит UTG, и позади него по ходу — HJ, CO, BTN, SB, BB, тогда как по
    номеру места те же пятеро идут SB, BB, HJ, CO, BTN.
    """
    stacks = {**dict.fromkeys(_SIX_MAX_SEATS, 96), "UTG": 24, "BB": 24}
    labels, seats, posts = _six_max(stacks, "BB")
    actions = [_shove(labels["UTG"], 24)]
    actions += [_fold(labels[pos]) for pos in ("HJ", "CO", "BTN", "SB")]
    actions.append(_call("Hero", 22, all_in=True))
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["Ah", "Ad"]}
    )
    en = enrich(normalize(raw))
    dp = en.report.decision_points[0]
    state = table_state(dp, en)
    shover = _shover(state)
    assert shover is not None

    rivals = _rivals_when_shoved(en.hand, dp, state, shover)
    assert [seat.position for seat in rivals] == ["HJ", "CO", "BTN", "SB", "BB"]
    assert [seat.position for seat in state.seats if seat.label != shover.label] == [
        "SB",
        "BB",
        "HJ",
        "CO",
        "BTN",
    ]


def test_the_action_order_helper_wraps_around_the_table():
    """`in_action_order_after` заворачивает круг и само место не возвращает.

    Проверяется на том же столе тремя разными опорами: от UTG (первый ход
    префлопа), от BB (последний — круг заворачивается целиком) и от героя.
    """
    _, en = _order_test_table()
    state = table_state(en.report.decision_points[0], en)
    by_position = {seat.position: seat for seat in state.seats}

    def order_after(position: str) -> list[str]:
        return [
            seat.position
            for seat in in_action_order_after(state.seats, by_position[position].label)
        ]

    assert order_after("UTG") == ["HJ", "CO", "BTN", "SB", "BB"]
    assert order_after("BB") == ["UTG", "HJ", "CO", "BTN", "SB"]
    assert order_after("CO") == ["BTN", "SB", "BB", "UTG", "HJ"]
    with pytest.raises(ValueError, match="места"):
        in_action_order_after(state.seats, "нет такого места")


# --- Анте стола входит в равновесие ---------------------------------------------


_POSITIONS_8 = ("SB", "BB", "UTG", "UTG+1", "LJ", "HJ", "CO", "BTN")


def _ante_table_shove(
    ante_chips: int,
    hero_cards: tuple[str, str] = ("Kc", "9d"),
    *,
    behind: int = 3,
    depth_bb: float = 10.0,
    seats_count: int = 6,
):
    """Стол с анте, все пасуют до Hero, Hero шовит. Блайнды 20/40.

    `behind` — сколько живых остаётся позади героя, `depth_bb` — эффективный стек
    ПОСЛЕ анте (та же величина, что уходит в равновесие).
    """
    bb_chips, sb_chips = 40, 20
    stack = round(depth_bb * bb_chips) + ante_chips
    order = _SIX_MAX_SEATS if seats_count == 6 else _POSITIONS_8
    act_order = list(order[2:]) + ["SB", "BB"]  # порядок хода на префлопе
    hero_index = len(act_order) - behind - 1
    hero_position = act_order[hero_index]
    labels = {pos: ("Hero" if pos == hero_position else pos) for pos in order}
    seats = [
        SeatInfo(seat=i + 1, label=labels[pos], stack=stack) for i, pos in enumerate(order)
    ]
    posts = [Post(label=s.label, kind=PostKind.ANTE, amount=ante_chips) for s in seats if ante_chips]
    posts += [
        Post(label=labels["SB"], kind=PostKind.SMALL_BLIND, amount=sb_chips),
        Post(label=labels["BB"], kind=PostKind.BIG_BLIND, amount=bb_chips),
    ]
    already = {"SB": sb_chips, "BB": bb_chips}.get(hero_position, 0)
    actions = [_fold(labels[pos]) for pos in act_order[:hero_index]]
    actions.append(
        RawAction(
            street=Street.PREFLOP,
            label="Hero",
            kind=ActionKind.RAISE,
            amount=stack - already,
            to_amount=stack,
            is_all_in=True,
            raw_line=f"Hero: raises {stack - already} to {stack} and is all-in",
        )
    )
    actions += [_fold(labels[pos]) for pos in act_order[hero_index + 1 :]]
    raw = _raw(
        seats=seats,
        button_seat=len(order),
        posts=posts,
        actions=actions,
        dealt={"Hero": list(hero_cards)},
        ante=ante_chips,
        sb=sb_chips,
        bb=bb_chips,
    )
    return enrich(normalize(raw))


def test_equilibrium_uses_table_ante_not_the_ante_free_game():
    """Анте стола входит в решаемую игру, а не выбрасывается.

    Продукт заявлен для MTT с анте. Равновесие без анте ТЕСНЕЕ разыгрываемого
    (на 10bb пуш 58.3% против 70.7%), поэтому по нему верные шовы помечались бы
    ошибкой — для тренажёра ложное обвинение хуже пропущенной ошибки.
    """
    dry = analyze_hand(_ante_table_shove(0, hero_cards=("9c", "9d"))).points[0]
    # 5/40 = 0.125bb с игрока
    ante = analyze_hand(_ante_table_shove(5, hero_cards=("9c", "9d"))).points[0]
    assert dry.detail["dead_extra_bb"] == 0.0
    assert ante.detail["dead_extra_bb"] == pytest.approx(6 * 0.125, abs=0.03)
    # больше мёртвых денег в банке -> шов прибыльнее, и это не округление
    assert ante.detail["ev_shove_bb"] > dry.detail["ev_shove_bb"] + 0.5


def test_ante_can_flip_the_verdict_from_mistake_to_correct():
    """Ровно тот отказ, ради которого правилась игра: верный шов помечался ошибкой.

    A4o, шов 10bb при трёх игроках позади. Без анте модель насчитывает минус
    («вы ошиблись»), с анте стола — плюс. Ложное обвинение учит пасовать там,
    где надо входить, и рушит доверие при первой же сверке с солвером.

    Знак читается из посчитанной EV шова, а не из `best_action`: у безантевой
    точки вердикта нет вовсе (вилка рвётся), и сравнивать надо ту величину,
    которая существует в обоих случаях.
    """
    dry = analyze_hand(_ante_table_shove(0, hero_cards=("Ac", "4d"))).points[0]
    ante = analyze_hand(_ante_table_shove(5, hero_cards=("Ac", "4d"))).points[0]
    assert dry.detail["ev_shove_bb"] < 0.0 < ante.detail["ev_shove_bb"]


# --- Живые игроки за героем при колле шова --------------------------------------


def test_live_players_behind_do_not_force_assuming_when_verdict_is_unmoved():
    """Живые за героем сами по себе точность не отменяют — отменяет их влияние.

    AA против шова верны и один на один, и когда малый блайнд тоже заколлирует.
    Вторая ось вилки это проверяет, а не постулирует: категорическое «живые
    позади -> assuming» недо-заявляло бы там, где заявлять есть что.
    """
    stacks = {**dict.fromkeys(_SIX_MAX_SEATS, 96), "SB": 24, "UTG": 24}
    labels, seats, posts = _six_max(stacks, "SB")
    actions = [_shove(labels["UTG"], 24)]
    actions += [_fold(labels[pos]) for pos in ("HJ", "CO", "BTN")]
    actions.append(_call("Hero", 23, all_in=True))
    actions.append(_fold(labels["BB"]))
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["Ah", "Ad"]}
    )
    p = analyze_hand(enrich(normalize(raw))).points[0]
    assert p.detail["live_others"] == 1
    assert p.detail["bracket"] == "stable"
    assert p.zone == "strict" and p.assumption is None


def test_players_behind_axis_is_computed_and_can_disagree():
    """Вторая ось считается и умеет расходиться с первой.

    AQo против шова 12bb: один на один колл плюсовой (+1.55bb), а если малый
    блайнд тоже войдёт — минусовой (−1.19bb). Ось помечена `unstable`.

    Изолированного случая, где вторая ось двигает вердикт, а первая нет, найти
    не удалось — ни на прежней вилке, ни на калиброванной полосе (повторный
    перебор по глубинам шова 8–20bb, стекам героя, глубине стола и 21 классу рук
    — ноль попаданий): узкий конец полосы диапазона шовера настолько тесен, что
    везде срабатывает раньше. Поэтому саму развилку проверяет модульный тест на
    `zone_for`, а здесь — что ось действительно считается по руке.
    """
    stacks = {**dict.fromkeys(_SIX_MAX_SEATS, 96), "SB": 24, "UTG": 24}
    labels, seats, posts = _six_max(stacks, "SB")
    actions = [_shove(labels["UTG"], 24)]
    actions += [_fold(labels[pos]) for pos in ("HJ", "CO", "BTN")]
    actions.append(_call("Hero", 23, all_in=True))
    actions.append(_fold(labels["BB"]))
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["Ah", "Qc"]}
    )
    p = analyze_hand(enrich(normalize(raw))).points[0]
    assert p.detail["ev_call_bb"] > 0.0 > p.detail["ev_call_all_behind_bb"]
    assert p.detail["behind_axis"] == "unstable"
    # Интервал по моделям шова на этой руке лежит по обе стороны нуля И шире
    # порога, поэтому вердикта у точки нет вовсе. Ось при этом посчитана и видна
    # в `detail` — тем и проверяется, что она считается независимо от того, чем
    # кончилась первая.
    assert p.best_action == ""
    assert p.detail["bracket"] == "unstable"


def test_zone_for_takes_the_players_behind_axis():
    assert zone_for("call", "call", live_total=4, best_model="call", best_behind=("call",))[0] == (
        "strict"
    )
    z, why = zone_for("call", "call", live_total=4, best_model="call", best_behind=("fold",))
    assert z == "assuming" and "позади" in why


# --- Вилка опрашивает интервал, а не два его конца -------------------------------


def test_the_call_width_grid_is_built_from_the_measured_band():
    """Сетка ширин выводится из полосы, а не набирается руками.

    Полоса — единственное место, где живёт замер (докстринг `_CALL_WIDTH_BAND`);
    сетка обязана начинаться на её нижнем конце, заканчиваться на верхнем и быть
    равномерной по логарифму множителя. Набранная руками, она разошлась бы с
    полосой молча — и докстринг ссылался бы на замер, которого сетка не
    воспроизводит.
    """
    from harness.analysis.preflop import _CALL_WIDTH_BAND
    from harness.analysis.preflop import _SHOVE_CALL_WIDTH_MULTIPLIERS as grid

    assert len(grid) == 5
    assert grid[0] == pytest.approx(_CALL_WIDTH_BAND[0])
    assert grid[-1] == pytest.approx(_CALL_WIDTH_BAND[1])
    ratios = [grid[i + 1] / grid[i] for i in range(len(grid) - 1)]
    assert all(r == pytest.approx(ratios[0], rel=1e-3) for r in ratios)


def test_the_shover_width_grid_is_built_from_the_measured_band():
    """То же требование к сетке ширин диапазона ШОВЕРА: она выводится из полосы.

    Прежде здесь стояли абсолютные доли комбо, ни к какому замеру не привязанные;
    теперь замер живёт в `_SHOVE_WIDTH_BAND`, и сетка обязана его воспроизводить,
    а не соседствовать с ним.
    """
    from harness.analysis.preflop import _SHOVE_WIDTH_BAND
    from harness.analysis.preflop import _SHOVER_WIDTH_MULTIPLIERS as grid

    assert len(grid) == 5
    assert grid[0] == pytest.approx(_SHOVE_WIDTH_BAND[0])
    assert grid[-1] == pytest.approx(_SHOVE_WIDTH_BAND[1])
    ratios = [grid[i + 1] / grid[i] for i in range(len(grid) - 1)]
    assert all(r == pytest.approx(ratios[0], rel=1e-3) for r in ratios)


def test_the_shover_width_family_reproduces_the_equilibrium_at_one():
    """На множителе x1 семейство ширин обязано быть самим равновесным шовом.

    Множитель осмыслен только как отклонение ОТ МОДЕЛИ: если на единице
    построенный диапазон — уже не модель, то и «x0.14», и «x1.46» отсчитываются
    не от неё, и полоса перестаёт значить то, что замерено.

    Порог 0.97 отделяет нынешнее построение от заменённого: на этой подыгре
    порядок из самого решения даёт 0.993, а ранжирование по эквити против
    равновесного шова той же глубины — то, что делал прежний `range_of_width`, —
    0.891. Замер по 29 точкам «колл шова» обеих фикстур (в докстринге
    `_shover_range_models`) даёт те же две величины медианами 0.995 и 0.906.
    """
    from harness.analysis.preflop import _WIDTH_KEY, _shover_range_models
    from harness.analysis.tools.equity import combos_of_class
    from harness.analysis.tools.multiway import Seat, unopened_shove_equilibrium

    solution = unopened_shove_equilibrium(
        Seat(posted_bb=0.0, behind_bb=11.0),
        [Seat(posted_bb=0.5, behind_bb=10.5), Seat(posted_bb=1.0, behind_bb=10.0)],
        2.375,
    )
    push = solution.push
    assert 0.0 < push.fraction_of_hands() < 1.0  # иначе на x1 сравнивать нечего

    at_one = _shover_range_models(solution, multipliers=(1.0,))[_WIDTH_KEY(1.0)]
    weight = {cls: push.weight(cls) * len(combos_of_class(cls)) for cls in push.weights}
    inside = sum(w for cls, w in weight.items() if cls in at_one.weights)
    assert inside / sum(weight.values()) >= 0.97


def test_zone_for_notices_a_reversal_inside_the_interval():
    """Концы сетки согласны, а внутри вердикт обратный — `strict` заявлять нельзя.

    Опрос двух концов объявил бы такой вывод устойчивым на интервале, внутри
    которого он меняет знак: зона `strict` стояла бы на выводе, который сам себя
    опровергает. Это единственное место, где правило проверяется прямо: на
    калиброванной полосе разворота внутри интервала не встретилось ни на одной
    из 58 точек сетки обеих фикстур (замер — в докстринге
    `_SHOVE_CALL_WIDTH_MULTIPLIERS`), то есть на реальных руках правило сейчас
    не срабатывает, а страховкой быть не перестаёт.
    """
    zone, why = zone_for(
        "shove", "shove", live_total=5, best_model="shove", best_interior=("fold", "shove")
    )
    assert zone == "assuming"
    assert "внутри" in why


def test_the_calibrated_band_removed_the_interior_reversal_of_this_spot():
    """Контрпример координатора J8o — на калиброванной полосе разворота больше нет.

    Под прежней полосой x0.5..x2.0 этот спот (J8o, шов 8bb, анте стола
    8 x 0.125bb, двое позади) давал +0.87 / +0.31 / -0.07 / -0.20 / +0.06 bb:
    оба конца «шов», внутри дважды минус. Под полосой, откалиброванной по
    наблюдаемому поведению поля, EV по сетке монотонна — концы и правда
    ограничивают интервал, — а сам интервал по-прежнему пересекает ноль, и
    точка остаётся в форме «около нуля». Разворот при этом не «исправлен»:
    полоса просто больше не заходит в ту область ширин, где он происходил.
    """
    en = _ante_table_shove(5, hero_cards=("Jc", "8d"), behind=2, depth_bb=8.0, seats_count=8)
    p = analyze_hand(en).points[0]
    values = list(p.detail["ev_shove_by_width_bb"].values())

    assert values == sorted(values, reverse=True)  # монотонно: разворота внутри нет
    assert values[0] > 0.0 > values[-1]  # но знак между концами меняется
    assert p.detail["dead_extra_bb"] == pytest.approx(1.0, abs=0.03)
    assert p.best_action == "около нуля, оба варианта допустимы"
    assert p.interval is not None and p.interval.near_zero is True


def test_a_tight_end_still_never_objects_to_a_junk_shove_and_that_is_honest():
    """Против вдвое более тесного поля шов мусором остаётся плюсовым.

    72o на 10.25bb, трое позади: против половины равновесной ширины колла шов
    даёт плюс, против равновесной и шире — крупный минус. Фолд-эквити тем
    больше, чем реже отвечают, и это факт об игре, а не дефект вилки.

    Отсюда следствие: вердикт «пас был верен» здесь и правда держится на том,
    что за столом отвечают не реже равновесия, — поэтому точечной оценки у
    точки нет вовсе, а не выдаётся со `strict`.
    """
    en = _ante_table_shove(5, hero_cards=("7c", "2d"), behind=3, depth_bb=10.25, seats_count=8)
    p = analyze_hand(en).points[0]
    values = list(p.detail["ev_shove_by_width_bb"].values())

    assert values[0] > 0.0  # вдвое более тесное поле — шов прибылен
    assert values[-1] < 0.0  # вдвое более широкое — крупный минус
    assert p.detail["ev_shove_bb"] < 0.0  # и по самой модели шов минусовой
    assert p.best_action == "около нуля, оба варианта допустимы"


# --- Цена не должна опираться на то, что вторая ось уже опровергла ---------------


def test_price_does_not_charge_for_a_reproach_the_second_axis_refutes():
    """AQo: колл плюсовой один на один и минусовой, если малый блайнд тоже войдёт.

    Упрекать игрока за пас на 1.55bb, когда мы сами посчитали, что при входе
    игрока позади колл теряет 1.19bb, нельзя: это число ведёт и ранжирование, и
    сумму потерь руки.
    """
    stacks = {**dict.fromkeys(_SIX_MAX_SEATS, 96), "SB": 24, "UTG": 24}
    labels, seats, posts = _six_max(stacks, "SB")
    actions = [_shove(labels["UTG"], 24)]
    actions += [_fold(labels[pos]) for pos in ("HJ", "CO", "BTN")]
    actions.append(_fold("Hero"))
    actions.append(_fold(labels["BB"]))
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["Ah", "Qc"]}
    )
    res = analyze_hand(enrich(normalize(raw)))
    p = res.points[0]
    assert p.detail["ev_call_bb"] > 0.0 > p.detail["ev_call_all_behind_bb"]
    assert p.ev_diff_bb == 0.0  # упрёк снят, хотя модель одна на один советует колл
    assert res.total_ev_loss_bb == 0.0


def test_a_verdict_that_flips_with_the_players_behind_names_the_fork():
    """Две точки модели дают разный оптимум — вместо одного действия названа развилка.

    Тот же AQo против шова, что и в тесте выше: `ev_call_bb` плюсовой,
    `ev_call_all_behind_bb` минусовой. Цена по правилу самого мягкого упрёка
    равна 0.0, и одно названное действие рядом с нулём выглядело бы бесплатным
    расхождением. Оба вердикта при этом остаются в `detail` по отдельности.

    Вторая половина теста — та же рука с AA, где обе точки модели согласны:
    там называется действие, а не развилка, иначе правило срабатывало бы всегда
    и ничего не различало.
    """
    stacks = {**dict.fromkeys(_SIX_MAX_SEATS, 96), "SB": 24, "UTG": 24}
    labels, seats, posts = _six_max(stacks, "SB")

    def hand(hero_cards: list[str], hero_calls: bool):
        actions = [_shove(labels["UTG"], 24)]
        actions += [_fold(labels[pos]) for pos in ("HJ", "CO", "BTN")]
        actions.append(_call("Hero", 23, all_in=True) if hero_calls else _fold("Hero"))
        actions.append(_fold(labels["BB"]))
        raw = _raw(
            seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": hero_cards}
        )
        return analyze_hand(enrich(normalize(raw))).points[0]

    split = hand(["Ah", "Qc"], hero_calls=False)
    assert split.detail["ev_call_bb"] > 0.0 > split.detail["ev_call_all_behind_bb"]
    assert split.detail["best_vs_one"] == "call"
    assert split.detail["best_all_behind"] == "fold"
    # Интервал по моделям шова на этой руке лежит по обе стороны нуля и шире
    # порога — вердикта у точки нет, и развилку игрок здесь не увидит. Оба
    # вердикта по отдельности при этом остаются в `detail`.
    assert split.best_action == ""
    assert split.ev_diff_bb == 0.0

    agreed = hand(["Ah", "Ad"], hero_calls=True)
    assert agreed.detail["best_vs_one"] == agreed.detail["best_all_behind"] == "call"
    assert agreed.best_action == "call"


# --- Инвариант зоны закреплён в контракте ---------------------------------------


def test_point_verdict_rejects_a_broken_zone_invariant():
    from pydantic import ValidationError

    from harness.contracts import PointVerdict, Range, SpotKind, Zone

    def build(zone: Zone, assumption: Assumption | None) -> PointVerdict:
        return PointVerdict(
            dp_index=0,
            street=Street.PREFLOP,
            spot=SpotKind.PUSHFOLD_UNOPENED,
            zone=zone,
            action_taken="shove",
            best_action="shove",
            ev_diff_bb=0.0,
            assumption=assumption,
        )

    shown = Assumption(range=Range(weights={"AA": 1.0}), source="model:test", note="модель")
    with pytest.raises(ValidationError, match="assuming"):
        build(Zone.STRICT, shown)
    with pytest.raises(ValidationError, match="assuming"):
        build(Zone.ASSUMING, None)
    assert build(Zone.ASSUMING, shown).zone == "assuming"
