"""Префлоп-анализ: классификация спота, вердикт, зона доверия, цена расхождения.

Синтетические руки собираются как `RawHand` и прогоняются через настоящий
конвейер (`normalize` → `enrich`): собирать `CanonicalHand` вручную значило бы
проверять анализ на входе, которого конвейер никогда не произведёт.
"""

from datetime import UTC, datetime

import pytest

from harness.analysis import analyze_hand
from harness.analysis.classifier import classify
from harness.analysis.error_cost import rank_points, total_ev_loss_bb
from harness.analysis.preflop import zone_for
from harness.contracts import (
    ActionKind,
    Post,
    PostKind,
    Provenance,
    RawAction,
    RawHand,
    SeatInfo,
    Street,
)
from harness.engine import enrich
from harness.normalizer import normalize
from harness.parsers.hh_parser import parse_hand
from tests.conftest import FIXTURE_DAILY, FIXTURE_PKO, requires_fixtures
from tests.test_hh_parser import SAMPLE

_BB = 2
_SB = 1
_BOARD = {Street.FLOP: ["Kd", "8h", "3s"], Street.TURN: ["4c"], Street.RIVER: ["9d"]}


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


# --- Тесты плана ----------------------------------------------------------------


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


def test_marginal_multiway_shove_is_assuming_with_shown_range():
    """Q9o на 10bb в двоих позади: против узкой модели шов плюсовой, против широкой — нет.

    Вердикт держится на догадке о том, как широко коллируют, — значит он обязан
    быть помечен `assuming`, а сама догадка показана игроку диапазоном.
    """
    en = _make_multiway_shove_hand(hero_cards=("Qc", "9d"), eff_bb=10.0, players_behind=2)
    p = analyze_hand(en).points[0]
    assert p.zone == "assuming" and p.detail["bracket"] == "unstable"
    assert p.detail["ev_shove_tight_bb"] > 0.0 > p.detail["ev_shove_wide_bb"]
    assert p.best_action == "fold" and p.action_taken == "shove" and p.ev_diff_bb < 0.0
    assert p.assumption is not None
    assert p.assumption.source == "model:nash_hu_call"
    assert 0.0 < p.assumption.range.fraction_of_hands() < 1.0


def test_two_live_without_equilibrium_shape_is_judged_by_bracket():
    """Шов ранней позиции, до которого спасовали все, кроме BB, — не игра `nash_hu`.

    Живых двое, но малый блайнд мёртвый, а шовер не малый блайнд: выдавать это за
    равновесие нельзя, зону обязан ставить bracket-тест.
    """
    en = _make_facing_shove_hand(hero_cards=("Ah", "Ad"), eff_bb=12.0, shover_bb=12.0)
    dp = en.report.decision_points[0]
    assert dp.live_total == 2
    p = analyze_hand(en).points[0]
    # причина зоны — устойчивость вилки, а не равновесие
    assert "узкой" in p.detail["zone_reason"] and "широкой" in p.detail["zone_reason"]
    assert "равновеси" not in p.detail["zone_reason"]


def test_hu_equilibrium_shape_is_strict_even_when_bracket_unstable():
    """Хедз-ап SB против BB: колл-диапазон известен из равновесия, вилка не нужна."""
    en = _make_hu_shove_hand(hero_cards=("3c", "2d"), eff_bb=10.0)
    p = analyze_hand(en).points[0]
    assert p.detail["bracket"] == "unstable"  # вилка сама по себе вердикт не удержала
    assert p.zone == "strict" and p.assumption is None
    assert "равновеси" in p.detail["zone_reason"]


def test_shove_without_fold_equity_is_marked():
    """На 2bb равновесный колл — любые две карты: фолд-эквити структурно нет."""
    en = _make_multiway_shove_hand(hero_cards=("Ac", "Ts"), eff_bb=2.0, players_behind=2)
    p = analyze_hand(en).points[0]
    assert p.spot == "pushfold_unopened"
    assert p.detail["fold_equity_ok"] is False


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


def test_zone_is_assuming_when_model_disagrees_with_both_bracket_ends():
    """Вердикт, который не подтверждает ни один конец вилки, `strict` быть не может.

    Найдено прогоном по реальной фикстуре: модель шова (пуш-сторона равновесия на
    13.5bb — около 50% комбо) ШИРЕ широкого конца вилки (top-40%), поэтому вилка
    её не накрывает. Обе её точки говорили «фолд», модель — «колл», и точка
    объявлялась `strict` по совпадению концов вилки между собой. Ровно та
    переоценка уверенности, ради предотвращения которой правило зоны и заведено.
    """
    en = _make_facing_shove_hand(hero_cards=("Jd", "Th"), eff_bb=9.0, shover_bb=13.5)
    p = analyze_hand(en).points[0]
    assert p.detail["ev_call_bb"] > 0.0  # модель: коллировать
    assert p.detail["ev_call_tight_bb"] < 0.0 and p.detail["ev_call_wide_bb"] < 0.0
    assert p.best_action == "call"
    assert p.zone == "assuming" and p.assumption is not None


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


# --- Анте стола входит в равновесие ---------------------------------------------


def _ante_table_shove(ante_chips: int, hero_cards: tuple[str, str] = ("Kc", "9d")):
    """6-max, все пасуют до Hero на CO, Hero шовит 10bb. Блайнды 20/40."""
    bb_chips, sb_chips = 40, 20
    stack = 10 * bb_chips
    labels = {pos: ("Hero" if pos == "CO" else pos) for pos in _SIX_MAX_SEATS}
    seats = [
        SeatInfo(seat=i + 1, label=labels[pos], stack=stack)
        for i, pos in enumerate(_SIX_MAX_SEATS)
    ]
    posts = [Post(label=s.label, kind=PostKind.ANTE, amount=ante_chips) for s in seats if ante_chips]
    posts += [
        Post(label=labels["SB"], kind=PostKind.SMALL_BLIND, amount=sb_chips),
        Post(label=labels["BB"], kind=PostKind.BIG_BLIND, amount=bb_chips),
    ]
    actions = [_fold(labels["UTG"]), _fold(labels["HJ"])]
    actions.append(
        RawAction(
            street=Street.PREFLOP,
            label="Hero",
            kind=ActionKind.RAISE,
            amount=stack,
            to_amount=stack,
            is_all_in=True,
            raw_line=f"Hero: raises {stack} to {stack} and is all-in",
        )
    )
    actions += [_fold(labels[pos]) for pos in ("BTN", "SB", "BB")]
    raw = _raw(
        seats=seats,
        button_seat=6,
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
    dry = analyze_hand(_ante_table_shove(0)).points[0]
    ante = analyze_hand(_ante_table_shove(5)).points[0]  # 5/40 = 0.125bb с игрока
    assert dry.detail["dead_extra_bb"] == 0.0
    assert ante.detail["dead_extra_bb"] == pytest.approx(6 * 0.125, abs=0.03)
    # больше мёртвых денег в банке -> шов прибыльнее, и это не округление
    assert ante.detail["ev_shove_bb"] > dry.detail["ev_shove_bb"] + 0.3


def test_ante_can_flip_the_verdict_from_mistake_to_correct():
    """Ровно тот отказ, ради которого правилась игра: верный шов помечался ошибкой.

    K9o, шов 10bb с CO при трёх игроках позади. Без анте модель насчитывает
    -0.29bb («вы ошиблись»), с анте стола — +0.52bb («сыграно верно»). Ложное
    обвинение учит пасовать там, где надо входить, и рушит доверие при первой же
    сверке с солвером.
    """
    dry = analyze_hand(_ante_table_shove(0, hero_cards=("Kc", "9d"))).points[0]
    ante = analyze_hand(_ante_table_shove(5, hero_cards=("Kc", "9d"))).points[0]
    assert dry.best_action == "fold" and dry.ev_diff_bb < -0.2
    assert ante.best_action == "shove" and ante.ev_diff_bb == 0.0


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

    A2o против шова 12bb: один на один колл плюсовой (+0.72bb), а если малый
    блайнд тоже войдёт — минусовой (−2.00bb). Ось помечена `unstable`, зона
    `assuming`.

    Изолированного случая, где вторая ось двигает вердикт, а первая нет, найти
    не удалось (перебор по глубинам шовера 5–20bb, стекам героя, анте и 21 классу
    рук — ноль попаданий): узкий конец `BRACKET_TIGHT` настолько тесен, что везде
    срабатывает раньше. Поэтому саму развилку проверяет модульный тест на
    `zone_for`, а здесь — что ось действительно считается по руке.
    """
    stacks = {**dict.fromkeys(_SIX_MAX_SEATS, 96), "SB": 24, "UTG": 24}
    labels, seats, posts = _six_max(stacks, "SB")
    actions = [_shove(labels["UTG"], 24)]
    actions += [_fold(labels[pos]) for pos in ("HJ", "CO", "BTN")]
    actions.append(_call("Hero", 23, all_in=True))
    actions.append(_fold(labels["BB"]))
    raw = _raw(
        seats=seats, button_seat=6, posts=posts, actions=actions, dealt={"Hero": ["Ah", "2c"]}
    )
    p = analyze_hand(enrich(normalize(raw))).points[0]
    assert p.detail["ev_call_bb"] > 0.0 > p.detail["ev_call_all_behind_bb"]
    assert p.detail["behind_axis"] == "unstable"
    assert p.zone == "assuming" and p.assumption is not None
    assert "позади" in p.assumption.note


def test_zone_for_takes_the_players_behind_axis():
    assert zone_for("call", "call", live_total=4, best_model="call", best_behind=("call",))[0] == (
        "strict"
    )
    z, why = zone_for("call", "call", live_total=4, best_model="call", best_behind=("fold",))
    assert z == "assuming" and "позади" in why
