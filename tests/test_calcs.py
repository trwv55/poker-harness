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

from harness.analysis.frequency import wilson_interval
from harness.calcs import run
from harness.contracts import (
    AnalysisResult,
    BetSizeThreshold,
    CoverageParams,
    DecisionPoint,
    DefenseParams,
    FieldThreshold,
    FrequencyStat,
    HeroFrequencyParams,
    LeaksParams,
    OpponentFrequencyParams,
    PointFilter,
    PointVerdict,
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
    HandsRepo,
    OpponentsRepo,
    PlayersRepo,
    SessionsRepo,
)
from harness.normalizer import normalize
from tests.test_player_stats import _call, _fold, _raise_to, _raw_hand


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
        "print(sorted({m for m in sys.modules} & {'harness.calcs', 'pokerkit', 'eval7'}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]", f"бот загрузил словарь: {result.stdout}"
