"""Выходы расчётов, которые видит модель агента (`agent_digest`), — без базы и сети.

У `/ask` выходы прежние (`question_digest`, `tests/test_question_text.py`): агентский
текст собран отдельно. Здесь — что агент видит иначе: сторона ставки в защите,
выборка словами, уровень данных, проценты без «.0» и ни одной служебной строки.
"""

from __future__ import annotations

from typing import get_args

import pytest

from harness.analysis.charts import parse_range
from harness.analysis.range_categories import kept_by_category, layout
from harness.contracts import (
    AbsentGroup,
    CalcName,
    CalcResult,
    CategoryGroup,
    EquityVsRangeResult,
    FrequencyStat,
    Measurement,
    OpponentRangeResult,
    RangeLine,
    SolverAction,
    SolverGuide,
    Street,
    Subject,
    ThresholdOutcome,
)
from harness.explanation.agent_checks import (
    check_agent_assumption,
    strip_card_tokens,
)
from harness.explanation.faithfulness import numbers_in, unsupported_numbers
from harness.explanation.question import MIN_STAT_OPPORTUNITIES, agent_digest, question_digest
from tests.test_question_text import ALL_RESULTS, _defense, _frequency, _threshold


def _agent_defense():
    """Синтетика: банк 4.0 BB, ставка 1.0 BB — сбрасывать 20%, защищаться 80%."""
    return _defense().model_copy(update={
        "pot_before": 400, "bet": 100, "pot_before_bb": 4.0, "bet_bb": 1.0,
        "defend_frequency": 0.8, "fold_frequency": 0.2, "required_equity": 1 / 6,
    })


@pytest.mark.parametrize("result", ALL_RESULTS, ids=lambda r: r.calc.value)
def test_the_agent_digest_registers_every_number_it_prints(result: CalcResult):
    digest = agent_digest(result)
    assert set(numbers_in(digest.text)) - digest.allowed == set()


@pytest.mark.parametrize("result", ALL_RESULTS, ids=lambda r: r.calc.value)
def test_the_agent_digest_has_no_service_lines(result: CalcResult):
    """Имя расчёта в скобках и служебные оговорки модели не нужны: она пересказала
    их игроку («данных игрока здесь нет вовсе»)."""
    text = agent_digest(result).text
    assert f"({result.calc.value})" not in text
    assert "Данных игрока здесь нет" not in text
    assert "Оценки у этой величины нет" not in text


def test_the_ask_digest_is_unchanged_by_the_agent_one():
    assert "(defense_frequency)" in question_digest(_defense()).text
    assert "Данных игрока здесь нет вовсе" in question_digest(_defense()).text


def test_the_defense_output_names_the_side_of_each_number():
    text = agent_digest(_agent_defense()).text
    assert (
        "Для ставящего: ставка окупается на одних фолдах, если защищающийся сбрасывает "
        "чаще 20%." in text
    )
    assert "Для защищающегося: защищаться 80%, эквити колла 16.7%." in text
    assert "Банк до ставки: 4.0 BB; ставка: 1.0 BB." in text


def test_whole_percents_go_without_a_trailing_zero():
    """24, а не 24.0; проверка чисел сравнивает значения, и «20%» в ответе
    проходит так же, как «20.0%»."""
    digest = agent_digest(_agent_defense())
    assert "20.0" not in digest.text and "80.0" not in digest.text
    assert unsupported_numbers("Соперник должен сбрасывать чаще 20%.", digest.allowed) == []
    assert unsupported_numbers("Соперник должен сбрасывать чаще 20.0%.", digest.allowed) == []


def test_a_small_sample_is_said_in_words_with_the_threshold_registered():
    """5 из 7: выборка мала. Порог печатается в строке и разрешён — модели не нужно
    брать «20» из промпта."""
    digest = agent_digest(_frequency())
    assert MIN_STAT_OPPORTUNITIES == 20
    assert "Выборка мала (меньше 20 возможностей)." in digest.text
    assert 20.0 in digest.allowed
    assert unsupported_numbers("Выборка меньше 20 возможностей.", digest.allowed) == []


def test_a_sample_at_the_threshold_is_enough():
    result = _frequency().model_copy(
        update={"measurement": Measurement(numerator=12, denominator=MIN_STAT_OPPORTUNITIES)}
    )
    text = agent_digest(result).text
    assert "Выборка достаточная." in text and "Выборка мала" not in text


def test_no_observations_get_no_sample_verdict():
    result = _frequency().model_copy(update={"measurement": Measurement(numerator=0, denominator=0)})
    text = agent_digest(result).text
    assert "наблюдений нет" in text and "Выборка" not in text


def test_the_threshold_output_names_the_sample_too():
    text = agent_digest(_threshold(ThresholdOutcome.ABOVE, None)).text
    assert "Выборка достаточная." in text  # 23 из 29
    assert "Порог: 61% (измеренная частота поля)." in text


@pytest.mark.parametrize(
    ("subject", "position", "nick", "level"),
    [
        (Subject.FIELD, "BB", None, "Уровень данных: поле в этой позиции (BB)."),
        (Subject.FIELD, None, None, "Уровень данных: поле в целом, все позиции."),
        (Subject.OPPONENT, None, "SynthNick", "Уровень данных: этот соперник (SynthNick), все позиции."),
        (Subject.OPPONENT, "CO", "SynthNick", "Уровень данных: этот соперник (SynthNick), позиция CO."),
        (Subject.HERO, "BTN", None, "Уровень данных: вы, позиция BTN."),
        (Subject.HERO, None, None, "Уровень данных: вы, все позиции."),
    ],
)
def test_every_statistic_names_its_level(
    subject: Subject, position: str | None, nick: str | None, level: str
):
    calc = CalcName.HERO_FREQUENCY if subject is Subject.HERO else CalcName.OPPONENT_FREQUENCY
    frequency = _frequency().model_copy(
        update={"calc": calc, "subject": subject, "position": position,
                "opponent_id": 5 if nick else None}
    )
    threshold = _threshold(ThresholdOutcome.ABOVE, None).model_copy(
        update={"subject": subject, "position": position, "opponent_id": 5 if nick else None}
    )
    for result in (frequency, threshold):
        assert level in agent_digest(result, opponent=nick).text


def test_digits_in_a_nick_are_allowed_numbers():
    result = _frequency().model_copy(
        update={"calc": CalcName.OPPONENT_FREQUENCY, "subject": Subject.OPPONENT,
                "opponent_id": 5, "stat": FrequencyStat.FOLD_TO_CBET}
    )
    digest = agent_digest(result, opponent="Synth77")
    assert 77.0 in digest.allowed


# --- диапазон соперника и эквити против суженного ---------------------------------


def range_result(**over: object) -> OpponentRangeResult:
    """Синтетика: 99 (6) + 66 наполовину (3) + A5s (4) + KQo (12) = 25 комбо из 1326."""
    base: dict[str, object] = {
        "calc": CalcName.OPPONENT_RANGE, "opponent_position": "BB", "street": Street.FLOP, "line": "call",
        "opener_position": "BTN", "chart": "defend", "chart_seats": 8, "table_seats": 6,
        "chart_position": "BB", "vs_chart_position": "BTN", "chart_depth_bb": 25.0,
        "range_text": "99, 66:0.5, A5s, KQo", "hands_share": 25 / 1326, "live_combos": 22.0,
    }
    base.update(over)
    return OpponentRangeResult.model_validate(base)


_LAYOUT_BOARD = ["Qc", "7d", "2s"]


def _with_equities(groups: list[CategoryGroup], equities: list[float]) -> list[CategoryGroup]:
    return [
        group.model_copy(update={"equity": equity}) for group, equity in zip(groups, equities, strict=True)
    ]


def equity_result(**over: object) -> EquityVsRangeResult:
    """Синтетика: суженный «99, A5s» на Q-7-2 — средняя пара (6 комбо, эквити 31.4%) и
    бэкдоры (4 комбо, 61.7%)."""
    narrowed = layout(parse_range("99, A5s"), _LAYOUT_BOARD, ["Kd", "Tc", *_LAYOUT_BOARD])
    base: dict[str, object] = {
        "calc": CalcName.EQUITY_VS_RANGE, "opponent_position": "BB", "street": Street.FLOP,
        "range_text": "99, A5s", "live_combos": 7.0,
        "layout": _with_equities(narrowed, [0.314, 0.617]),
    }
    base.update(over)
    return EquityVsRangeResult.model_validate(base)


def layout_result() -> OpponentRangeResult:
    """`range_result` с раскладкой ядра на синтетическом флопе Q-7-2 (герой — K♦T♦) и
    эквити против каждой категории."""
    groups = layout(
        parse_range("99, 66:0.5, A5s, KQo"), _LAYOUT_BOARD, ["Kd", "Tc", *_LAYOUT_BOARD]
    )
    return range_result(
        layout_of="start", layout=_with_equities(groups, [0.22, 0.385, 0.556, 0.61])
    )


def test_the_layout_lines_name_every_class_with_its_category():
    """Строка категории: имя, число классов и комбо, эквити руки героя против неё, все
    классы — пары словами, доля меньше целой процентом; числа зарегистрированы, общего
    эквити нет."""
    digest = agent_digest(layout_result())
    lines = digest.text.splitlines()
    head = lines.index(
        "Раскладка по категориям (флоп; стартовый диапазон; категорию руки на борде "
        "определяет код; эквити — против каждой категории отдельно, а не всего диапазона):"
    )
    assert lines[head + 1 :] == [
        "топ-пара с сильным кикером — 1 класс, 7 комбо, эквити вашей руки против неё — 22%: KQo.",
        "средняя пара — 1 класс, 6 комбо, эквити вашей руки против неё — 38.5%: пара девяток.",
        (
            "слабая пара — 1 класс, 3 комбо, эквити вашей руки против неё — 55.6%: пара "
            "шестёрок (50%)."
        ),
        "бэкдоры и старшие карты — 1 класс, 4 комбо, эквити вашей руки против неё — 61%: A5s.",
    ]
    assert {22.0, 38.5, 55.6, 61.0} <= digest.allowed
    assert unsupported_numbers(strip_card_tokens(digest.text), digest.allowed) == []


def test_a_mark_on_part_of_the_combos_prints_its_count_as_output_numbers():
    """Пометка части комбо — «у N из M комбо»: оба числа — числа выхода (книга)."""
    board = ["Jh", "8h", "8c"]
    groups = layout(parse_range("A5s"), board, board)
    digest = agent_digest(range_result(layout_of="start", layout=groups))
    assert (
        "бэкдоры и старшие карты — 1 класс, 4 комбо: A5s (+ флеш-дро у 1 из 4 комбо)."
        in digest.text.splitlines()
    )
    assert {1.0, 4.0} <= digest.allowed
    assert unsupported_numbers(strip_card_tokens(digest.text), digest.allowed) == []


def test_the_solver_guide_line_names_the_nearest_line_and_its_categories():
    """Ориентир: борд и ставка выгрузки против руки, по действию — категории на борде
    выгрузки; числа зарегистрированы, карты борда числами не считаются. Библиотека не
    прочиталась — строка с причиной, без ориентира."""
    dump_board = ["Qh", "7c", "2d"]
    guide = SolverGuide(
        board=dump_board, bet_share=0.2, hand_board=_LAYOUT_BOARD, hand_bet_share=0.29,
        actions=[
            SolverAction(action="call", layout=layout(parse_range("QQ, AQs:0.5"), dump_board,
                                                      dump_board)),
            SolverAction(action="fold", layout=layout(parse_range("43o"), dump_board,
                                                      dump_board)),
        ],
    )
    digest = agent_digest(layout_result().model_copy(update={"solver": guide}))
    lines = digest.text.splitlines()
    head = next(i for i, line in enumerate(lines) if line.startswith("Ориентир солвера"))
    assert lines[head:] == [
        (
            "Ориентир солвера (ближайшая линия: Q♥7♣2♦, ставка 20%; в руке Q♣7♦2♠, 29%; "
            "данные владельца, категории — на борде выгрузки):"
        ),
        (
            "колл — сет — 1 класс, 3 комбо: пара дам; топ-пара с сильным кикером — 1 класс, "
            "2 комбо: AQs (50%)."
        ),
        "фолд — бэкдоры и старшие карты — 1 класс, 12 комбо: 43o.",
    ]
    assert not any("эквити" in line for line in lines[head:])
    assert unsupported_numbers(strip_card_tokens(digest.text), digest.allowed) == []
    broken = agent_digest(layout_result().model_copy(update={"solver_error": "нет файла"}))
    assert "Ориентир солвера: нет файла — ориентира нет." in broken.text.splitlines()


def test_the_equity_digest_prints_the_kept_summary():
    """Сводка сужения по категориям: целиком, частью с убранными классами (пары
    словами), убрана целиком; итог «Оставлено N комбо из M» — без доли оставленного и
    сданного (спека 2026-10-24, §14.4); числа зарегистрированы."""
    groups = layout_result().layout
    kept = kept_by_category(groups, parse_range("99, KQo"))
    result = equity_result(
        range_text="99, KQo", categories_of="start", categories=kept,
        source_combos=sum(k.combos for k in kept), kept_combos=sum(k.kept_combos for k in kept),
    )
    digest = agent_digest(result)
    assert (
        "По категориям (исходный — стартовый диапазон): топ-пара с сильным кикером — "
        "целиком, 7 комбо; средняя пара — целиком, 6 комбо; слабая пара — убрана целиком "
        "(3 комбо); бэкдоры и старшие карты — убрана целиком (4 комбо)."
    ) in digest.text.splitlines()
    assert "Оставлено 13 комбо из 20." in digest.text.splitlines()
    assert "сдаёт" not in digest.text and "на начало улицы" not in digest.text
    assert 65.0 not in digest.allowed
    assert unsupported_numbers(strip_card_tokens(digest.text), digest.allowed) == []


def test_the_equity_digest_has_an_equity_per_category_and_no_total():
    """Эквити — по категориям суженного, у каждой комбо и своё число; строки одного
    общего эквити нет (спека 2026-10-24, §14.6)."""
    digest = agent_digest(equity_result())
    lines = digest.text.splitlines()
    assert lines[-1] == (
        "Эквити вашей руки по категориям суженного (каждое — против этой категории, а не "
        "всего диапазона): средняя пара — 6 комбо, 31.4%; бэкдоры и старшие карты — 4 комбо, "
        "61.7%."
    )
    assert not any("при вскрытии" in line for line in lines)
    assert {31.4, 61.7} <= digest.allowed
    assert unsupported_numbers(strip_card_tokens(digest.text), digest.allowed) == []


def test_the_turn_and_the_river_name_the_narrowing_they_come_from():
    """Сводка на тёрне — из сужения флопа, на ривере — из сужения тёрна."""
    kept = kept_by_category(layout_result().layout, parse_range("99, KQo"))
    river = agent_digest(equity_result(street=Street.RIVER, categories_of="turn", categories=kept))
    assert "По категориям (исходный — ваше сужение тёрна):" in river.text
    turn = agent_digest(range_result(street=Street.RIVER, layout_of="turn",
                                     layout=layout_result().layout))
    assert "Раскладка по категориям (ривер; ваше сужение тёрна этого соперника (допущение);" in (
        turn.text
    )


@pytest.mark.parametrize("result", [range_result(), equity_result()], ids=["range", "equity"])
def test_the_range_outputs_register_every_number_that_stays_a_number(result: CalcResult):
    """Классы с буквой сверка вырезает (`strip_card_tokens`) — их цифры не
    регистрируются; пары печатаются словами, доли чарта — процентами, и они разрешены."""
    digest = agent_digest(result)
    assert unsupported_numbers(strip_card_tokens(digest.text), digest.allowed) == []


def test_the_range_output_names_the_chart_and_the_short_table_rule():
    text = agent_digest(range_result()).text
    assert "Соперник: BB, линия до флопа — колл против опена BTN." in text
    assert (
        "Источник: чарт защиты BB против BTN, 8-max, 25 BB; за столом мест — 6: "
        "чарт 8-max по позиции от конца (правило, не допущение)." in text
    )
    assert "Доля рук: 1.9%; комбо без ваших карт и борда (флоп): 22." in text


def test_pocket_pairs_are_printed_in_words_and_are_not_numbers():
    """Пара словами, доля чарта — процентом при классе; «99» и «66» числами не
    становятся: модель, сказавшая «99», не прошла бы сверку и не должна её видеть."""
    digest = agent_digest(range_result())
    assert "Стартовый диапазон: пара девяток, пара шестёрок (50%), A5s, KQo." in digest.text
    assert "99" not in digest.text and "66" not in digest.text
    assert 50.0 in digest.allowed
    assert not {99.0, 66.0, 0.5} & digest.allowed


def test_a_weighted_lettered_class_keeps_its_letters_and_prints_the_share():
    text = agent_digest(range_result(range_text="AKo:0.25, TT")).text
    assert "Стартовый диапазон: AKo (25%), пара десяток." in text


@pytest.mark.parametrize("line", get_args(RangeLine))
def test_every_range_line_is_named_in_words(line: str):
    """Каждая линия `RangeLine` названа: лимп SB и чек BB — тоже, а не `KeyError`."""
    text = agent_digest(range_result(line=line, opener_position="SB")).text
    assert line not in text


def test_the_sb_limp_check_line_names_the_limper():
    text = agent_digest(range_result(line="check", opener_position="SB")).text
    assert "Соперник: BB, линия до флопа — чек после лимпа SB." in text


def test_a_position_with_a_digit_passes_the_number_check():
    """Открывший UTG+1, соперник UTG+2: цифры имён позиций не регистрируются и
    сверкой не считаются (R12)."""
    result = range_result(
        opponent_position="UTG+2", chart="open", chart_position="UTG+2",
        vs_chart_position=None, opener_position="UTG+1", line="open_raise",
    )
    digest = agent_digest(result)
    assert "UTG+2" in digest.text
    assert not {1.0, 2.0} & digest.allowed
    assert unsupported_numbers(strip_card_tokens(digest.text), digest.allowed) == []
    defend = agent_digest(range_result(vs_chart_position="UTG+1", opener_position="UTG+1"))
    assert "чарт защиты BB против UTG+1" in defend.text
    assert unsupported_numbers(strip_card_tokens(defend.text), defend.allowed) == []


def test_an_eight_max_range_says_nothing_about_another_table():
    assert "за столом мест" not in agent_digest(range_result(table_seats=8)).text


def test_the_equity_output_says_the_narrowing_is_an_assumption():
    text = agent_digest(equity_result()).text
    assert "Диапазон, суженный агентом: пара девяток, A5s." in text
    assert "Диапазон сужен агентом — это допущение." in text
    assert "Комбо: 7." in text and "стартового" not in text


def test_ask_has_no_digest_for_agent_only_calculations():
    with pytest.raises(ValueError, match="расчёт агента"):
        question_digest(range_result())


# --- план 2026-10-17, пп. 8 и 9: группы вне колла и классы по просьбе игрока -----------

_ABSENT = [
    AbsentGroup(group="pairs", classes=["AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66",
                                        "55", "44", "33", "22"],
                whole=True, raised=True, all_in=True),
    AbsentGroup(group="suited_aces", classes=["AKs", "AQs"], whole=False, raised=True,
                all_in=False),
    AbsentGroup(group="offsuit_aces", classes=["AKo", "A9o"], whole=False, raised=False,
                all_in=True),
    AbsentGroup(group="broadways", classes=["KQo"], whole=False, raised=False, all_in=True),
]


def test_the_absent_groups_line_names_the_chart_actions():
    text = agent_digest(range_result(absent=_ABSENT)).text
    assert (
        "Чего нет в колл-диапазоне по чарту BB против BTN (факт чарта, не допущение): "
        "карманных пар — ни одной, ими 3-бетит или идёт олл-ин; одномастных тузов: AKs, AQs "
        "— ими 3-бетит; разномастных тузов: AKo, A9o — ими идёт олл-ин; бродвеев без туза: "
        "KQo — ими идёт олл-ин."
    ) in text


def test_partly_absent_pairs_are_named_in_words():
    pairs = AbsentGroup(group="pairs", classes=["AA", "99"], whole=False, raised=True,
                        all_in=False)
    text = agent_digest(range_result(absent=[pairs])).text
    assert "карманных пар: пара тузов, пара девяток — ими 3-бетит." in text


def test_the_check_after_a_limp_says_check_and_raise():
    """Против лимпа SB у BB чек, а не колл, и рейз (изоляция), а не 3-бет."""
    text = agent_digest(range_result(
        line="check", opener_position="SB", vs_chart_position="SB", absent=_ABSENT[:2],
    )).text
    assert "Чего нет в чек-диапазоне по чарту BB против SB" in text
    assert "ими рейзит или идёт олл-ин" in text and "3-бет" not in text.split("Чего нет")[1]


def test_without_absent_groups_there_is_no_line():
    assert "Чего нет" not in agent_digest(range_result()).text


def test_the_absent_groups_line_adds_no_number_and_passes_the_block_check():
    """Строка о группах не регистрирует чисел и не даёт модели числа, которое сверка
    отвергла бы: классы с буквой вырезаются, пары — словами, «3-бетит» — термин."""
    digest = agent_digest(range_result(absent=_ABSENT))
    assert digest.allowed == agent_digest(range_result()).allowed
    absent_line = digest.text.splitlines()[-1]
    assert numbers_in(strip_card_tokens(absent_line)) == []
    assert check_agent_assumption(absent_line, allowed=frozenset(), leaks_called=False) == []


def test_the_agent_added_classes_are_said_with_the_reason():
    """Классы агента — строкой «добавлено агентом (допущение): … — <причина>», по строке
    на причину; классы игрока — прежней строкой; доли от стартового нет."""
    digest = agent_digest(equity_result(
        range_text="99, A5s, 22, 33, KJs", added=["33", "22", "KJs"],
        agent_added={"33": "пары коллируют чаще", "22": "пары коллируют чаще"},
    ))
    lines = digest.text.splitlines()
    assert "Добавлено по вашей просьбе: KJs — по чарту их здесь нет." in lines
    assert "Добавлено агентом (допущение): пара троек, пара двоек — пары коллируют чаще." in lines
    assert "стартового" not in digest.text


def test_the_added_classes_are_a_separate_line_without_the_share():
    """С добавленными классами — отдельная строка, доли от стартового нет; комбо и эквити
    по категориям — как без них."""
    plain = agent_digest(equity_result())
    digest = agent_digest(equity_result(range_text="99, A5s, 22, 33", added=["33", "22"]))
    assert (
        "Добавлено по вашей просьбе: пара троек, пара двоек — по чарту их здесь нет." in digest.text
    )
    assert "Диапазон сужен агентом — это допущение." in digest.text
    assert "Комбо: 7." in digest.text and "стартового" not in digest.text
    assert digest.allowed == plain.allowed
    assert "Добавлено" not in plain.text


def test_the_layout_head_promises_no_equity_without_it():
    """Карты героя неизвестны — раскладка без эквити: заголовок его не обещает, строк
    эквити нет."""
    groups = layout(parse_range("99, A5s"), _LAYOUT_BOARD, _LAYOUT_BOARD)
    text = agent_digest(range_result(layout_of="start", layout=groups)).text
    assert (
        "Раскладка по категориям (флоп; стартовый диапазон; категорию руки на борде "
        "определяет код):"
    ) in text.splitlines()
    assert "эквити" not in text


def test_without_the_previous_narrowing_the_layout_is_a_refusal_line():
    """Тёрн без сужения флопа (ривер без тёрна): вместо раскладки — «сначала сузьте …»,
    стартовый по чарту печатается как прежде (§14.11)."""
    refusal = "сначала сузьте флоп: вызовите equity_vs_range для флопа этого соперника"
    text = agent_digest(range_result(street=Street.TURN, layout_refusal=refusal)).text
    assert f"Раскладки по категориям нет: {refusal}." in text.splitlines()
    assert "Стартовый диапазон: пара девяток, пара шестёрок (50%), A5s, KQo." in text
    assert "Раскладка по категориям (" not in text


def test_the_preflop_equity_digest_prints_one_number():
    """До флопа категорий нет — одно эквити; число зарегистрировано."""
    digest = agent_digest(equity_result(street=Street.PREFLOP, layout=[], preflop_equity=0.314))
    assert "Эквити вашей руки при вскрытии (до флопа категорий нет): 31.4%." in (
        digest.text.splitlines()
    )
    assert 31.4 in digest.allowed
    assert "по категориям суженного" not in digest.text


def test_the_river_after_a_checked_turn_names_the_flop_narrowing():
    text = agent_digest(range_result(street=Street.RIVER, layout_of="flop_turn_checked",
                                     layout=layout_result().layout)).text
    assert (
        "Раскладка по категориям (ривер; ваше сужение флопа этого соперника; на тёрне ставок "
        "не было (допущение);"
    ) in text
