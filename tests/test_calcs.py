"""Словарь расчётов целиком: выборка из базы, счёт, подпись результата.

Настоящий Postgres (`db` из `conftest`) — по той же причине, что у остальных
тестов памяти: фильтры словаря это условия SQL, и проверять их подделкой
репозитория значило бы проверять подделку.

Руки собираются построителем из tests/test_player_stats.py и кладутся в базу в обеих
формах — исходной и канонической, — потому что построитель отдаёт первую, а
конвейер делает из неё вторую. Второй руки, собранной отдельно, здесь нет.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any, cast

import pytest

import harness.calcs as calcs_module
from harness.analysis import analyze_hand, solver_guide
from harness.analysis.charts import lookup, parse_range, to_notation
from harness.analysis.frequency import wilson_interval
from harness.analysis.opponent_range import live_combos
from harness.analysis.tools.equity import equity_vs_range
from harness.analysis.tools.exact_equity import class_equities
from harness.calcs import run
from harness.contracts import (
    ActionKind,
    AnalysisResult,
    BetSizeThreshold,
    CalcName,
    CanonicalAction,
    CoverageParams,
    DecisionPoint,
    DefenseParams,
    EquityVsRangeParams,
    FieldThreshold,
    FrequencyStat,
    HeroFrequencyParams,
    LeaksParams,
    OpponentFrequencyParams,
    OpponentRangeParams,
    PointFilter,
    PointVerdict,
    RangeCategory,
    RangeNarrowing,
    SpotKind,
    Street,
    Subject,
    ThresholdOutcome,
    ThresholdParams,
    ThresholdSide,
    Window,
    Zone,
)
from harness.memory.repos import (
    AnalysesRepo,
    CalcCacheRepo,
    HandsRepo,
    OpponentsRepo,
    PlayersRepo,
    SessionsRepo,
)
from harness.normalizer import normalize
from tests.test_opponent_range import (
    _BB_THREE_BETS,
    _DEFEND_CALL,
    _THROUGH_RIVER,
    _TURN_CHECKED,
    BTN_OPENS_BB_CALLS,
    _raw,
    defend_file,
)
from tests.test_player_stats import _call, _fold, _raise_to, _raw_hand
from tests.test_postflop_line import _played_out
from tests.test_solver_guide import write_dump


async def _player_with_session(db, tg_user_id: int) -> tuple[int, int]:
    player = await PlayersRepo(db).get_or_create(tg_user_id=tg_user_id)
    session_row = await SessionsRepo(db).active_or_create(player.id)
    return player.id, session_row.id


def _open_from(
    hero_position: str, *, tournament_id: str, hand_no: str, called_by: str | None = None
):
    """Герой открывает рейзом; названное место уравнивает, остальные пасуют.

    Уравнявшее место нужно, чтобы частота поля не была тождественным нулём: на
    нуле проверка «порог взят у поля» прошла бы и с чужим знаменателем.
    """
    others = [pos for pos in ("UTG", "HJ", "CO", "BTN", "SB", "BB") if pos != hero_position]
    actions = [_raise_to("Hero", 6, already=0)]
    for pos in others:
        already = {"SB": 1, "BB": 2}.get(pos, 0)
        actions.append(_call(pos, 6 - already) if pos == called_by else _fold(pos))
    return _raw_hand(
        hero_position=hero_position,
        actions=actions,
        tournament_id=tournament_id,
        hand_no=hand_no,
    )


async def _store(db, *, session_id: int, raw) -> int:
    hand_id = await HandsRepo(db).save_raw(session_id=session_id, raw=raw)
    await HandsRepo(db).save_canonical(hand_id, normalize(raw))
    return hand_id


def _point(
    *,
    index: int,
    street: Street,
    spot: SpotKind,
    position: str,
    best: str,
    ev_diff_bb: float,
) -> tuple[PointVerdict, DecisionPoint]:
    """Вердикт и обстановка одной точки: репозиторий сшивает их по `dp_index`."""
    return (
        PointVerdict(
            dp_index=index,
            street=street,
            spot=spot,
            zone=Zone.STRICT,
            action_taken="fold",
            best_action=best,
            ev_diff_bb=ev_diff_bb,
        ),
        DecisionPoint(
            index=index,
            street=street,
            label="Hero",
            position=position,
            to_call=2,
            pot_before=3,
            eff_stack=200,
            eff_stack_bb=100.0,
            action=CanonicalAction(
                street=street,
                label="Hero",
                kind=ActionKind.FOLD,
                committed_after=0,
                raw_line="Hero: folds",
            ),
        ),
    )


async def _save_points(db, *, session_id: int, hand_no: str, points) -> None:
    raw = _open_from("CO", tournament_id="TP", hand_no=hand_no)
    hand_id = await _store(db, session_id=session_id, raw=raw)
    await AnalysesRepo(db).save(
        hand_id=hand_id,
        result=AnalysisResult(hand_no=hand_no, points=[verdict for verdict, _ in points]),
        decision_points=[context for _, context in points],
    )


# --- подпись результата -------------------------------------------------------------


async def test_every_result_names_the_calculation_that_produced_it(db):
    """Каждый расчёт набора возвращает результат, подписанный своим именем.

    Неверная маршрутизация опасна тем, что невидима: ответ выглядит нормально,
    но отвечает не на тот вопрос. Подпись делает ошибку видимой, поэтому она
    проверяется у всех шести, а не у одного.
    """
    player_id, session_id = await _player_with_session(db, tg_user_id=9101)
    await _store(db, session_id=session_id, raw=_open_from("CO", tournament_id="T1", hand_no="H1"))
    every = [
        HeroFrequencyParams(stat=FrequencyStat.VPIP),
        OpponentFrequencyParams(stat=FrequencyStat.VPIP),
        CoverageParams(),
        LeaksParams(),
        DefenseParams(pot_before=100, bet=50),
        ThresholdParams(
            subject=Subject.HERO,
            stat=FrequencyStat.VPIP,
            threshold=BetSizeThreshold(pot_before=100, bet=50, side=ThresholdSide.DEFEND),
        ),
    ]
    for params in every:
        result = await run(db, player_id, params)
        assert result.calc is params.calc


async def test_the_defence_calculation_touches_no_data(db):
    """Требуемая частота защиты считается без базы вовсе — она чистая арифметика.

    Проверяется подстановкой `None` вместо сессии: любое обращение к базе
    внутри уронило бы вызов.
    """
    result = await run(cast(Any, None), 0, DefenseParams(pot_before=100, bet=50))
    assert result.calc.value == "defense_frequency"
    assert result.defend_frequency == pytest.approx(2 / 3)


# --- частоты ------------------------------------------------------------------------


async def test_the_hero_and_the_field_are_measured_from_the_same_hands(db):
    """Знаменатель героя — раздачи, знаменатель поля — места в тех же раздачах.

    Две величины из одной выборки, и знаменатели у них РАЗНЫЕ по определению
    поля: сложены все оппоненты, а не раздачи, где хоть кто-то из них сыграл.
    """
    player_id, session_id = await _player_with_session(db, tg_user_id=9102)
    for i in (1, 2):
        await _store(
            db,
            session_id=session_id,
            raw=_open_from("CO", tournament_id="T1", hand_no=f"F{i}"),
        )

    hero = await run(db, player_id, HeroFrequencyParams(stat=FrequencyStat.VPIP))
    field = await run(db, player_id, OpponentFrequencyParams(stat=FrequencyStat.VPIP))

    assert (hero.subject, hero.measurement.numerator, hero.measurement.denominator) == (
        Subject.HERO,
        2,
        2,
    )
    assert (field.subject, field.measurement.numerator, field.measurement.denominator) == (
        Subject.FIELD,
        0,
        10,
    )


async def test_a_position_narrows_the_hero_denominator(db):
    """Фильтр по позиции считает только раздачи, где герой сидел именно там."""
    player_id, session_id = await _player_with_session(db, tg_user_id=9103)
    await _store(db, session_id=session_id, raw=_open_from("CO", tournament_id="T1", hand_no="P1"))
    await _store(db, session_id=session_id, raw=_open_from("BTN", tournament_id="T1", hand_no="P2"))

    on_button = await run(
        db, player_id, HeroFrequencyParams(stat=FrequencyStat.VPIP, position="BTN")
    )
    assert on_button.measurement.denominator == 1


async def test_an_unknown_position_is_refused_instead_of_answering_zero_of_zero(db):
    """Опечатка в позиции — отказ, а не «0 из 0», неотличимое от «не случалось»."""
    player_id, _ = await _player_with_session(db, tg_user_id=9104)
    with pytest.raises(ValueError):
        await run(db, player_id, HeroFrequencyParams(stat=FrequencyStat.VPIP, position="BUTTON"))


async def test_the_opponent_is_measured_only_where_the_binding_names_him(db):
    """Оппонент считается в турнирах со сшивкой; турнир без неё в знаменатель не входит.

    Метка участника сквозная только внутри турнира, и связать метки разных
    турниров может лишь владелец. Раздачи турнира, про который он ничего не
    сказал, не считаются вовсе — ни в числителе, ни в знаменателе.
    """
    player_id, session_id = await _player_with_session(db, tg_user_id=9105)
    await _store(db, session_id=session_id, raw=_open_from("CO", tournament_id="TA", hand_no="O1"))
    await _store(db, session_id=session_id, raw=_open_from("CO", tournament_id="TB", hand_no="O2"))
    opponents = OpponentsRepo(db)
    opponent_id = await opponents.get_or_create(owner_player_id=player_id, nick="villain")
    await opponents.link(
        owner_player_id=player_id,
        opponent_id=opponent_id,
        room_tournament_id="TA",
        participant_label="BTN",
    )

    result = await run(
        db,
        player_id,
        OpponentFrequencyParams(stat=FrequencyStat.VPIP, opponent_id=opponent_id),
    )

    assert result.subject is Subject.OPPONENT
    assert result.measurement.denominator == 1


async def test_the_window_narrows_a_calculation_to_one_evening(db):
    """Окно вечера отсекает раздачи других вечеров того же игрока."""
    player_id, first = await _player_with_session(db, tg_user_id=9106)
    await _store(db, session_id=first, raw=_open_from("CO", tournament_id="T1", hand_no="W1"))
    await SessionsRepo(db).close_active(player_id)
    second = (await SessionsRepo(db).active_or_create(player_id)).id
    await _store(db, session_id=second, raw=_open_from("CO", tournament_id="T1", hand_no="W2"))

    whole = await run(db, player_id, HeroFrequencyParams(stat=FrequencyStat.VPIP))
    evening = await run(
        db,
        player_id,
        HeroFrequencyParams(stat=FrequencyStat.VPIP, window=Window(session_id=second)),
    )

    assert whole.measurement.denominator == 2
    assert evening.measurement.denominator == 1
    assert evening.window.session_id == second


async def test_another_players_hands_never_enter_a_calculation(db):
    """Область расчёта — сессии названного игрока, и чужие раздачи в неё не входят."""
    mine, _my_session = await _player_with_session(db, tg_user_id=9107)
    _theirs, their_session = await _player_with_session(db, tg_user_id=9108)
    await _store(
        db, session_id=their_session, raw=_open_from("CO", tournament_id="T1", hand_no="X1")
    )

    result = await run(db, mine, HeroFrequencyParams(stat=FrequencyStat.VPIP))

    assert result.measurement.denominator == 0


# --- покрытие и цена ----------------------------------------------------------------


async def test_points_without_a_price_do_not_enter_the_sum(db):
    """Покрытие считает все точки, сумма — только те, у которых цена посчитана.

    Три величины и три знаменателя: судимых из всех, с ценой из судимых, сумма
    по последним. У точки без вердикта в этой руке стоит отрицательное число, и
    в сумму оно не входит: судимость решает колонка `judged`, а не знак поля,
    иначе не посчитанное выдавалось бы за посчитанное.
    """
    player_id, session_id = await _player_with_session(db, tg_user_id=9109)
    await _save_points(
        db,
        session_id=session_id,
        hand_no="CP",
        points=[
            _point(
                index=0,
                street=Street.PREFLOP,
                spot=SpotKind.PUSHFOLD_UNOPENED,
                position="BTN",
                best="shove",
                ev_diff_bb=-2.0,
            ),
            _point(
                index=1,
                street=Street.PREFLOP,
                spot=SpotKind.PUSHFOLD_UNOPENED,
                position="BTN",
                best="около нуля, оба варианта допустимы",
                ev_diff_bb=0.0,
            ),
            _point(
                index=2,
                street=Street.FLOP,
                spot=SpotKind.POSTFLOP,
                position="BTN",
                best="",
                ev_diff_bb=-7.0,
            ),
        ],
    )

    result = await run(db, player_id, CoverageParams())

    assert (result.judged.numerator, result.judged.denominator) == (2, 3)
    assert (result.priced.numerator, result.priced.denominator) == (1, 2)
    assert result.loss_bb == pytest.approx(2.0)


async def test_every_filter_narrows_the_same_query(db):
    """Улица, спот и позиция — обычные условия по колонкам, и складываются как угодно.

    Ради этого точки и переехали в таблицу: отдельного запроса на каждое
    сочетание фильтров нет.
    """
    player_id, session_id = await _player_with_session(db, tg_user_id=9110)
    await _save_points(
        db,
        session_id=session_id,
        hand_no="CF",
        points=[
            _point(
                index=0,
                street=Street.PREFLOP,
                spot=SpotKind.PUSHFOLD_UNOPENED,
                position="BTN",
                best="shove",
                ev_diff_bb=-2.0,
            ),
            _point(
                index=1,
                street=Street.PREFLOP,
                spot=SpotKind.PUSHFOLD_UNOPENED,
                position="SB",
                best="shove",
                ev_diff_bb=-1.0,
            ),
            _point(
                index=2,
                street=Street.FLOP,
                spot=SpotKind.POSTFLOP,
                position="BTN",
                best="",
                ev_diff_bb=0.0,
            ),
        ],
    )

    async def total(**kwargs) -> int:
        result = await run(db, player_id, CoverageParams(filter=PointFilter(**kwargs)))
        return result.judged.denominator

    assert await total() == 3
    assert await total(street=Street.PREFLOP) == 2
    assert await total(position="BTN") == 2
    assert await total(spot=SpotKind.POSTFLOP) == 1
    assert await total(street=Street.PREFLOP, position="BTN") == 1
    assert await total(street=Street.FLOP, position="SB") == 0


async def test_the_leak_report_carries_the_denominator_it_answers_from(db):
    """Список ликов едет вместе с покрытием окна, из которого он посчитан.

    Лики ранжируются по цене, значит в список попадают только точки с ценой;
    без своего знаменателя список читался бы как полная картина игры.
    """
    player_id, session_id = await _player_with_session(db, tg_user_id=9111)
    await _save_points(
        db,
        session_id=session_id,
        hand_no="LK",
        points=[
            _point(
                index=0,
                street=Street.PREFLOP,
                spot=SpotKind.PUSHFOLD_UNOPENED,
                position="BTN",
                best="shove",
                ev_diff_bb=-2.0,
            ),
            _point(
                index=1,
                street=Street.FLOP,
                spot=SpotKind.POSTFLOP,
                position="BTN",
                best="",
                ev_diff_bb=0.0,
            ),
        ],
    )

    result = await run(db, player_id, LeaksParams())

    assert (result.judged.numerator, result.judged.denominator) == (1, 2)
    assert [stat.rule.key for stat in result.leaks] == ["no_shove"]


# --- вердикт по порогу --------------------------------------------------------------


async def _hero_with(
    db, tg_user_id: int, positions: list[str], called_by: str | None = None
) -> tuple[int, int]:
    player_id, session_id = await _player_with_session(db, tg_user_id=tg_user_id)
    for index, position in enumerate(positions):
        await _store(
            db,
            session_id=session_id,
            raw=_open_from(
                position, tournament_id="T1", hand_no=f"V{index}", called_by=called_by
            ),
        )
    return player_id, session_id


async def test_observations_needed_is_filled_exactly_when_undecided(db):
    """Число недостающих наблюдений стоит при неопределённом знаке и только при нём.

    Утверждение и недостающая выборка — взаимоисключающие состояния: заполнить
    оба значило бы сказать «вывод есть, но нужно ещё столько-то».
    """
    player_id, _ = await _hero_with(db, 9112, ["CO", "BTN", "SB"])

    decided = await run(
        db,
        player_id,
        ThresholdParams(
            subject=Subject.HERO,
            stat=FrequencyStat.VPIP,
            threshold=BetSizeThreshold(pot_before=100, bet=900, side=ThresholdSide.DEFEND),
        ),
    )
    undecided = await run(
        db,
        player_id,
        ThresholdParams(
            subject=Subject.HERO,
            stat=FrequencyStat.VPIP,
            threshold=BetSizeThreshold(pot_before=100, bet=10, side=ThresholdSide.DEFEND),
        ),
    )

    assert decided.outcome is ThresholdOutcome.ABOVE
    assert decided.observations_needed is None
    assert undecided.outcome is ThresholdOutcome.UNDECIDED
    assert undecided.observations_needed is not None


async def test_a_measured_threshold_comes_with_its_own_denominator(db):
    """Порог, взятый у поля, — измерение: рядом с ним едет его собственная выборка.

    Иначе «порог 61%» выглядел бы величиной, которую кто-то знает точно, хотя
    её посчитали по тем же рукам, что и саму частоту.
    """
    player_id, _ = await _hero_with(db, 9113, ["CO", "BTN"], called_by="BB")

    result = await run(
        db,
        player_id,
        ThresholdParams(
            subject=Subject.HERO, stat=FrequencyStat.VPIP, threshold=FieldThreshold()
        ),
    )

    assert result.threshold_source == "field"
    assert result.reference is not None
    assert (result.reference.numerator, result.reference.denominator) == (2, 10)
    assert result.threshold == pytest.approx(0.2)


async def test_no_result_carries_an_interval_bound(db):
    """Ни один конец доверительного интервала в результат не попадает.

    Интервал считается и решает, утверждать или нет, но наружу не выводится
    (решение владельца). Держится это формой типа: числа, которого нет в выходе
    инструмента, не может быть и в тексте.
    """
    player_id, _ = await _hero_with(db, 9114, ["CO", "BTN", "SB"])

    result = await run(
        db,
        player_id,
        ThresholdParams(
            subject=Subject.HERO,
            stat=FrequencyStat.VPIP,
            threshold=BetSizeThreshold(pot_before=100, bet=10, side=ThresholdSide.DEFEND),
        ),
    )

    measurement = result.measurement
    bounds = wilson_interval(measurement.numerator, measurement.denominator)
    assert bounds is not None
    assert not _numbers_in(result.model_dump()) & {round(bound, 6) for bound in bounds}


def _numbers_in(value: Any) -> set[float]:
    """Все числа документа, на любой глубине вложенности.

    Обходом, а не перечислением полей верхнего уровня: конец интервала,
    спрятанный внутрь вложенной модели, проверка по верхнему уровню пропустила
    бы.
    """
    if isinstance(value, bool):
        return set()
    if isinstance(value, (int, float)):
        return {round(float(value), 6)}
    if isinstance(value, dict):
        return set().union(*(_numbers_in(item) for item in value.values()), set())
    if isinstance(value, (list, tuple)):
        return set().union(*(_numbers_in(item) for item in value), set())
    return set()


# --- граница пакетов ----------------------------------------------------------------


def test_the_bot_does_not_load_the_dictionary():
    """Образ бота не тянет словарь, а значит и расчётный стек за ним.

    Это и есть причина, по которой склейка живёт отдельным пакетом, а не в
    `memory`: `memory` бот импортирует, и словарь оттуда затащил бы в его
    процесс `pokerkit` и `eval7`.
    """
    code = (
        "import sys, harness.bot.main;"
        "print(sorted({m for m in sys.modules} & {'harness.calcs', 'harness.agent', 'pokerkit', 'eval7'}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]", f"бот загрузил словарь: {result.stdout}"


# --- диапазон соперника и эквити против суженного (только агент) -------------------


async def stored_analysis(db, *, session_id: int, actions=None) -> int:
    """Синтетическая рука вечера с разбором: BTN открывает, BB коллирует, флоп."""
    raw = _raw(actions or BTN_OPENS_BB_CALLS)
    en = _played_out(raw)
    hand_id = await HandsRepo(db).save_raw(session_id=session_id, raw=raw)
    await HandsRepo(db).save_canonical(hand_id, en.hand)
    await HandsRepo(db).save_enriched(hand_id, en)
    return await AnalysesRepo(db).save(
        hand_id=hand_id, result=analyze_hand(en), decision_points=en.report.decision_points
    )


async def test_the_opponent_range_comes_from_the_chart_of_the_analysed_hand(
    db, tmp_path, monkeypatch
):
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    player_id, session_id = await _player_with_session(db, tg_user_id=9201)
    analysis_id = await stored_analysis(db, session_id=session_id)
    result = await run(db, player_id, OpponentRangeParams(
        analysis_id=analysis_id, opponent_position="BB", street=Street.FLOP,
    ))
    assert (result.calc, result.line, result.chart, result.vs_chart_position) == (
        CalcName.OPPONENT_RANGE, "call", "defend", "BTN",
    )
    assert (result.chart_seats, result.table_seats) == (8, 6)


async def test_another_players_analysis_is_not_read(db, tmp_path, monkeypatch):
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    _owner, owner_session = await _player_with_session(db, tg_user_id=9202)
    stranger, _ = await _player_with_session(db, tg_user_id=9203)
    analysis_id = await stored_analysis(db, session_id=owner_session)
    with pytest.raises(ValueError, match="разбора этой руки нет"):
        await run(db, stranger, OpponentRangeParams(
            analysis_id=analysis_id, opponent_position="BB", street=Street.FLOP,
        ))


async def test_equity_is_deterministic_and_cached(db, tmp_path, monkeypatch):
    """Эквити по категориям — точный перебор: число категории равно взвешенному среднему
    эквити классов `class_equities`; эквити каждого класса с живыми комбо уходит в
    `calc_cache` (`eqc:v1:exact:`), повтор — то же число."""
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    player_id, session_id = await _player_with_session(db, tg_user_id=9204)
    analysis_id = await stored_analysis(db, session_id=session_id)
    params = EquityVsRangeParams(
        analysis_id=analysis_id, opponent_position="BB", street=Street.FLOP,
        hands="QQ, 77, 22, AQs",
    )
    first = await run(db, player_id, params)
    by_class = class_equities(("Kd", "Tc"), ["Qc", "7d", "2s"])
    prefix = calcs_module.CLASS_EQUITY_PREFIX
    cached = await CalcCacheRepo(db).get_all(prefix)
    assert prefix == "eqc:v1:exact:"
    assert cached == {f"KdTc|Qc7d2s|{cls}": value for cls, value in by_class.items()}
    (sets,) = [g for g in first.layout if g.category is RangeCategory.SET]
    assert sets.equity == pytest.approx(
        (3 * by_class["QQ"] + 3 * by_class["77"] + 3 * by_class["22"]) / 9
    )
    assert (await run(db, player_id, params)).layout == first.layout
    assert first.live_combos == 12.0


async def test_a_cached_class_equity_is_returned_without_counting_again(
    db, tmp_path, monkeypatch
):
    """Строки кэша всех классов сужения отдаются как есть: перебор не зовётся."""
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    player_id, session_id = await _player_with_session(db, tg_user_id=9206)
    analysis_id = await stored_analysis(db, session_id=session_id)
    await CalcCacheRepo(db).upsert_many(calcs_module.CLASS_EQUITY_PREFIX, {
        calcs_module.class_equity_key(("Kd", "Tc"), ["Qc", "7d", "2s"], "QQ"): 0.25,
        calcs_module.class_equity_key(("Kd", "Tc"), ["Qc", "7d", "2s"], "AQs"): 0.75,
    })

    def must_not_run(*_args: object, **_kw: object) -> dict[str, float]:
        raise AssertionError("эквити посчитано заново при строках в кэше")

    monkeypatch.setattr(calcs_module, "class_equities", must_not_run)
    result = await run(db, player_id, EquityVsRangeParams(
        analysis_id=analysis_id, opponent_position="BB", street=Street.FLOP, hands="QQ, AQs",
    ))
    assert [(g.category, g.equity) for g in result.layout] == [
        (RangeCategory.SET, 0.25), (RangeCategory.TOP_PAIR_STRONG_KICKER, 0.75),
    ]


async def test_a_class_missing_from_the_cache_counts_the_whole_board_again(
    db, tmp_path, monkeypatch
):
    """Нет в кэше хоть одного класса — перебор заново, и число кэша другого метода
    (Монте-Карло, `eqr:`) не отдаётся."""
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    player_id, session_id = await _player_with_session(db, tg_user_id=9207)
    analysis_id = await stored_analysis(db, session_id=session_id)
    key = calcs_module.class_equity_key(("Kd", "Tc"), ["Qc", "7d", "2s"], "QQ")
    await CalcCacheRepo(db).upsert_many(calcs_module.CLASS_EQUITY_PREFIX, {key: 0.25})
    await CalcCacheRepo(db).upsert_many("eqr:v1:50000:42:", {key: 0.5})
    monkeypatch.setattr(
        calcs_module, "class_equities", lambda *_a, **_kw: {"QQ": 0.1, "AQs": 0.9}
    )
    result = await run(db, player_id, EquityVsRangeParams(
        analysis_id=analysis_id, opponent_position="BB", street=Street.FLOP, hands="QQ, AQs",
    ))
    assert [g.equity for g in result.layout] == pytest.approx([0.1, 0.9])


async def test_every_category_of_the_narrowed_range_carries_its_own_equity(
    db, tmp_path, monkeypatch
):
    """Выход `equity_vs_range` (спека 2026-10-24, §14): раскладка суженного — у каждой
    категории своё эквити, общего числа нет; на ривере — из сужения тёрна."""
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    player_id, session_id = await _player_with_session(db, tg_user_id=9213)
    analysis_id = await stored_analysis(db, session_id=session_id, actions=_THROUGH_RIVER)
    flop = await run(db, player_id, EquityVsRangeParams(
        analysis_id=analysis_id, opponent_position="BB", street=Street.FLOP,
        hands="сет, AQs, AJo",
    ))
    assert "equity" not in flop.model_dump() and "share_of_start" not in flop.model_dump()
    assert [g.category for g in flop.layout] == [
        RangeCategory.SET, RangeCategory.TOP_PAIR_STRONG_KICKER,
        RangeCategory.BACKDOORS_OVERCARDS,
    ]
    assert all(g.equity is not None and 0.0 <= g.equity <= 1.0 for g in flop.layout)
    turn_narrowing = RangeNarrowing(
        analysis_id=analysis_id, opponent_position="BB", street=Street.TURN,
        classes=["QQ", "77", "JTs"],
    )
    river = await run(db, player_id, EquityVsRangeParams(
        analysis_id=analysis_id, opponent_position="BB", street=Street.RIVER,
        hands="сет, ничего", prior=turn_narrowing,
    ))
    assert river.categories_of == "turn"
    # У героя K-T на Q-7-2-5-9: сеты его бьют, J-T без пары — нет.
    assert [(g.category, g.equity) for g in river.layout] == [
        (RangeCategory.SET, 0.0), (RangeCategory.NOTHING, 1.0),
    ]
    with pytest.raises(ValueError, match="сначала сузьте тёрн"):
        await run(db, player_id, EquityVsRangeParams(
            analysis_id=analysis_id, opponent_position="BB", street=Street.RIVER, hands="сет",
        ))


async def test_the_preflop_equity_is_one_number_deterministic_and_cached(
    db, tmp_path, monkeypatch
):
    """До флопа категорий нет (§14.11): одно эквити против суженного — Монте-Карло с
    фиксированным сидом, равное прямому расчёту; строка `calc_cache` — своё пространство
    ключей; раскладки нет."""
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    player_id, session_id = await _player_with_session(db, tg_user_id=9214)
    analysis_id = await stored_analysis(db, session_id=session_id, actions=_BB_THREE_BETS)
    params = EquityVsRangeParams(
        analysis_id=analysis_id, opponent_position="BB", street=Street.PREFLOP, hands="AKs",
    )
    first = await run(db, player_id, params)
    expected = equity_vs_range(
        ("Kd", "Tc"), parse_range("AKs"), [],
        iterations=calcs_module.PREFLOP_EQUITY_ITERATIONS, seed=calcs_module.PREFLOP_EQUITY_SEED,
    )
    assert first.preflop_equity == expected
    assert first.layout == [] and first.categories_of is None
    assert calcs_module.PREFLOP_EQUITY_PREFIX == "eqr:v1:50000:42:"
    key = "KdTc||" + to_notation(parse_range("AKs"))
    assert await CalcCacheRepo(db).get(calcs_module.PREFLOP_EQUITY_PREFIX, key) == expected
    monkeypatch.setattr(calcs_module, "equity_vs_range", lambda *_a, **_kw: 0.0)
    assert (await run(db, player_id, params)).preflop_equity == expected


async def test_a_narrowing_outside_the_start_is_refused_by_the_calculation(
    db, tmp_path, monkeypatch
):
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    player_id, session_id = await _player_with_session(db, tg_user_id=9205)
    analysis_id = await stored_analysis(db, session_id=session_id)
    with pytest.raises(ValueError, match="нет в стартовом диапазоне: AKo"):
        await run(db, player_id, EquityVsRangeParams(
            analysis_id=analysis_id, opponent_position="BB", street=Street.FLOP, hands="AKo",
        ))


async def test_the_turn_layout_is_of_the_flop_narrowing_and_without_it_is_refused(
    db, tmp_path, monkeypatch
):
    """Раскладка по категориям (спека 2026-10-24, §4, §14.11): флоп — стартового; тёрн —
    сужения флопа этой руки, ривер — сужения тёрна; без них раскладки нет, вместо неё —
    «сначала сузьте …», и от стартового тёрн и ривер не раскладываются (ставка на тёрне
    была); стартовый по чарту — как прежде. После чек-чек тёрна ривер — из сужения
    флопа."""
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    player_id, session_id = await _player_with_session(db, tg_user_id=9208)
    analysis_id = await stored_analysis(db, session_id=session_id, actions=_THROUGH_RIVER)

    def params(street: Street, prior: RangeNarrowing | None = None) -> OpponentRangeParams:
        return OpponentRangeParams(
            analysis_id=analysis_id, opponent_position="BB", street=street, prior=prior,
        )

    def classes(result) -> set[str]:
        return {item.hand_class for group in result.layout for item in group.classes}

    preflop = await run(db, player_id, params(Street.PREFLOP))
    assert (preflop.layout_of, preflop.layout) == (None, [])
    flop = await run(db, player_id, params(Street.FLOP))
    assert flop.layout_of == "start"
    assert classes(flop) == set(parse_range(_DEFEND_CALL).weights)
    assert sum(group.combos for group in flop.layout) == pytest.approx(flop.live_combos)
    bare_turn = await run(db, player_id, params(Street.TURN))
    assert (bare_turn.layout_of, bare_turn.layout) == (None, [])
    assert (bare_turn.layout_refusal or "").startswith("сначала сузьте флоп")
    assert bare_turn.range_text == flop.range_text
    narrowing = RangeNarrowing(
        analysis_id=analysis_id, opponent_position="BB", street=Street.FLOP,
        classes=["QQ", "77", "AQs", "KQo"],
    )
    turn = await run(db, player_id, params(Street.TURN, narrowing))
    assert turn.layout_of == "flop"
    assert classes(turn) == {"QQ", "77", "AQs", "KQo"}
    bare_river = await run(db, player_id, params(Street.RIVER))
    assert (bare_river.layout_of, bare_river.layout) == (None, [])
    assert (bare_river.layout_refusal or "").startswith("сначала сузьте тёрн")
    from_flop = await run(db, player_id, params(Street.RIVER, narrowing))
    assert (from_flop.layout_of, from_flop.layout) == (None, [])
    assert (from_flop.layout_refusal or "").startswith("сначала сузьте тёрн")
    turn_narrowing = narrowing.model_copy(update={"street": Street.TURN, "classes": ["QQ", "KQo"]})
    river = await run(db, player_id, params(Street.RIVER, turn_narrowing))
    assert river.layout_of == "turn"
    assert classes(river) == {"QQ", "KQo"}
    assert river.layout_refusal is None
    checked_id = await stored_analysis(db, session_id=session_id, actions=_TURN_CHECKED)
    checked = await run(db, player_id, OpponentRangeParams(
        analysis_id=checked_id, opponent_position="BB", street=Street.RIVER,
        prior=narrowing.model_copy(update={"analysis_id": checked_id}),
    ))
    assert checked.layout_of == "flop_turn_checked"
    assert classes(checked) == {"QQ", "77", "AQs", "KQo"}


async def test_every_category_of_the_range_layout_carries_its_own_equity(
    db, tmp_path, monkeypatch
):
    """Выход `opponent_range` на флопе, тёрне и ривере (спека 2026-10-24, §14): у каждой
    категории раскладки своё эквити руки героя — взвешенное среднее эквити её классов;
    общего числа нет; у категорий ориентира солвера эквити нет."""
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    library = tmp_path / "solver"
    monkeypatch.setattr(solver_guide, "DEFAULT_SOLVER_DIR", library)
    hand = _played_out(_raw(_THROUGH_RIVER)).hand
    write_dump(library, "dry.json", ante_type=hand.ante_type, call="QQ, 77, AKo")
    player_id, session_id = await _player_with_session(db, tg_user_id=9215)
    analysis_id = await stored_analysis(db, session_id=session_id, actions=_THROUGH_RIVER)
    boards = {
        Street.FLOP: ["Qc", "7d", "2s"], Street.TURN: ["Qc", "7d", "2s", "5h"],
        Street.RIVER: ["Qc", "7d", "2s", "5h", "9s"],
    }
    priors = {
        Street.TURN: RangeNarrowing(analysis_id=analysis_id, opponent_position="BB",
                                    street=Street.FLOP, classes=["QQ", "77", "AQs", "KQo"]),
        Street.RIVER: RangeNarrowing(analysis_id=analysis_id, opponent_position="BB",
                                     street=Street.TURN, classes=["QQ", "KQo"]),
    }
    for street, board in boards.items():
        result = await run(db, player_id, OpponentRangeParams(
            analysis_id=analysis_id, opponent_position="BB", street=street,
            prior=priors.get(street),
        ))
        assert "equity" not in result.model_dump(exclude={"layout", "solver"})
        by_class = class_equities(("Kd", "Tc"), board)
        assert result.layout
        for group in result.layout:
            expected = sum(by_class[c.hand_class] * c.combos for c in group.classes) / group.combos
            assert group.equity == pytest.approx(expected)
        if street is Street.FLOP:
            assert result.solver is not None
            assert all(g.equity is None for a in result.solver.actions for g in a.layout)


async def test_a_narrowing_by_category_counts_what_is_kept(db, tmp_path, monkeypatch):
    """`hands` с категорией раскладки (спека 2026-10-24, §5): категория разворачивается в
    классы стартового на борде, «-класс» вычитает; сводка — сколько оставлено из
    стартового. Тёрн по категории — из сужения флопа, правило подмножества прежнее."""
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    player_id, session_id = await _player_with_session(db, tg_user_id=9209)
    analysis_id = await stored_analysis(db, session_id=session_id, actions=_THROUGH_RIVER)
    flop = await run(db, player_id, EquityVsRangeParams(
        analysis_id=analysis_id, opponent_position="BB", street=Street.FLOP,
        hands="оверпара -AA, AQs",
    ))
    assert flop.range_text == "KK, AQs"
    assert flop.categories_of == "start"
    assert flop.source_combos == pytest.approx(
        live_combos(parse_range(_DEFEND_CALL), ["Kd", "Tc", "Qc", "7d", "2s"])
    )
    assert flop.kept_combos == pytest.approx(flop.live_combos)
    narrowing = RangeNarrowing(
        analysis_id=analysis_id, opponent_position="BB", street=Street.FLOP,
        classes=["KK", "AQs"],
    )
    turn = await run(db, player_id, EquityVsRangeParams(
        analysis_id=analysis_id, opponent_position="BB", street=Street.TURN,
        hands="оверпара", prior=narrowing,
    ))
    assert (turn.range_text, turn.categories_of) == ("KK", "flop")
    with pytest.raises(ValueError, match="в категории «трипс» на этом борде нет"):
        await run(db, player_id, EquityVsRangeParams(
            analysis_id=analysis_id, opponent_position="BB", street=Street.TURN,
            hands="трипс", prior=narrowing,
        ))


async def test_the_flop_output_carries_the_nearest_solver_guide(db, tmp_path, monkeypatch):
    """Флоп: выгрузка библиотеки того же ключа, текстуры и корзины — ориентир, действия
    разложены по категориям на борде выгрузки; тёрн ориентира не несёт; пустая
    библиотека — ориентира нет; нечитаемая — причина в `solver_error`."""
    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    library = tmp_path / "solver"
    monkeypatch.setattr(solver_guide, "DEFAULT_SOLVER_DIR", library)
    player_id, session_id = await _player_with_session(db, tg_user_id=9210)
    analysis_id = await stored_analysis(db, session_id=session_id, actions=_THROUGH_RIVER)

    def params(street: Street) -> OpponentRangeParams:
        return OpponentRangeParams(analysis_id=analysis_id, opponent_position="BB", street=street)

    empty = await run(db, player_id, params(Street.FLOP))
    assert (empty.solver, empty.solver_error) == (None, None)
    hand = _played_out(_raw(_THROUGH_RIVER)).hand
    write_dump(library, "dry.json", ante_type=hand.ante_type, call="QQ, 77, AKo")
    flop = await run(db, player_id, params(Street.FLOP))
    assert flop.solver is not None
    assert (flop.solver.board, flop.solver.bet_share) == (["Qh", "7c", "2d"], 0.2)
    assert flop.solver.hand_board == ["Qc", "7d", "2s"]
    call = next(a for a in flop.solver.actions if a.action == "call")
    assert [(g.category, [c.hand_class for c in g.classes]) for g in call.layout] == [
        (RangeCategory.SET, ["QQ", "77"]), (RangeCategory.BACKDOORS_OVERCARDS, ["AKo"]),
    ]
    assert (await run(db, player_id, params(Street.TURN))).solver is None
    (library / "broken.json").write_text("{", encoding="utf-8")
    broken = await run(db, player_id, params(Street.FLOP))
    assert broken.solver is None and "broken.json" in (broken.solver_error or "")


async def test_the_tool_passes_the_reset_streets_to_the_calculation(db, tmp_path, monkeypatch):
    """Сквозь `tools_for`: F1 → T1 → F2 в истории, на тёрне была ставка — ривер получает
    сброшенный тёрн (`reset`) и отказывает «флоп сужен заново»; без сброса (F1 → T1) —
    считается от T1."""
    from harness.calcs.routing import CalcRun, tools_for

    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    player_id, session_id = await _player_with_session(db, tg_user_id=9216)
    analysis_id = await stored_analysis(db, session_id=session_id, actions=_THROUGH_RIVER)

    def narrowing(street: Street, classes: list[str]) -> RangeNarrowing:
        return RangeNarrowing(
            analysis_id=analysis_id, opponent_position="BB", street=street, classes=classes,
        )

    async def river(*history: RangeNarrowing) -> str:
        run_state = CalcRun(
            max_calcs=8, sizes_from=frozenset(), analysis_id=analysis_id,
            narrowings=list(history),
        )
        tool = next(t for t in tools_for(db, player_id, session_id, run_state)
                    if t.name == "equity_vs_range")
        function = cast(Any, tool.function)
        out = await function(opponent_position="BB", street=Street.RIVER, hands="QQ")
        return out if isinstance(out, str) else str(out.return_value)

    f1, t1 = narrowing(Street.FLOP, ["QQ", "77"]), narrowing(Street.TURN, ["QQ"])
    f2 = narrowing(Street.FLOP, ["QQ"])
    refused = await river(f1, t1, f2)
    assert "сначала сузьте тёрн: флоп сужен заново" in refused
    counted = await river(f1, t1)
    assert "Расчёт не выполнен" not in counted and "Эквити вашей руки" in counted


async def test_a_lower_case_position_is_the_same_opponent(db, tmp_path, monkeypatch):
    """Позиция приводится к верхнему регистру на границе инструмента: «bb» считается, а
    не отвергается, и сужение пишется на того же BB — тёрн «bb» берёт его исходным."""
    from harness.calcs.routing import CalcRun, tools_for

    monkeypatch.setattr(lookup, "DEFAULT_DEFEND_PATH", defend_file(tmp_path))
    player_id, session_id = await _player_with_session(db, tg_user_id=9217)
    analysis_id = await stored_analysis(db, session_id=session_id, actions=_THROUGH_RIVER)
    run_state = CalcRun(max_calcs=8, sizes_from=frozenset(), analysis_id=analysis_id)
    tools = {
        t.name: cast(Any, t.function) for t in tools_for(db, player_id, session_id, run_state)
    }
    flop = await tools["equity_vs_range"](
        opponent_position="bb", street=Street.FLOP, hands="QQ, 77"
    )
    assert not isinstance(flop, str), flop
    assert run_state.narrowings[-1].opponent_position == "BB"
    assert run_state.prior_for(analysis_id, "BB", Street.TURN) == run_state.narrowings[-1]
    turn = await tools["opponent_range"](opponent_position="bb", street=Street.TURN)
    assert not isinstance(turn, str), turn
    assert "Раскладка по категориям (тёрн; ваше сужение флопа" in str(turn.return_value)
