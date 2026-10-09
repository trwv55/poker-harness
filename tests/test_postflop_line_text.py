"""Постфлоп-линия в тексте разбора: заголовок точки, строки руки, дро, линии, «окупается».

Спека — `docs/superpowers/specs/2026-10-09-postflop-line-design.md`, §2 (макет), §4.2.1,
§4.5–§4.9, §7; гейты §9: 1 (в части текста), 9, 10, 11. Руки синтетические (метки
P1…P7, номера свои); расчёт идёт настоящим `analyze_hand`, а печатает его
`presentation` — проверяется то, что видит игрок, а не промежуточные структуры.
"""

from __future__ import annotations

import re

import pytest

from harness.analysis import analyze_hand
from harness.analysis.tools.hand_class import hand_strength
from harness.contracts import (
    ActionTag,
    Backdoor,
    Combination,
    Draw,
    DrawKind,
    EnrichedHand,
    FoldThreshold,
    HandCategory,
    Line,
    Overcards,
    PostflopLineDetail,
    Purpose,
    ShowdownValue,
    SizeTag,
    postflop_line_detail,
)
from harness.contracts import DrawCall as DrawCallDetail
from harness.explanation import hand_replay
from harness.presentation import Msg, hand_analysis_msgs
from harness.presentation.messages import (
    _ACTION_TAG_WORD,
    _DETAIL_LABELS,
    _DRAW_KIND_WORD,
    _PURPOSE_WORD,
    _UNPRINTED_TAGS,
    _half_up_percent,
    _hand_words,
    _postflop_line_lines,
)
from tests.test_postflop_line import (
    _DEEP,
    _HERO_OPENS,
    _V_OPENS,
    F,
    P,
    R,
    T,
    _bet,
    _call,
    _check,
    _gutshot_call,
    _hand,
    _oop,
    _raise,
    _reference_hand,
)

# --- эталон: макет §2 спеки, строка в строку ------------------------------------------

# Пронумерованная часть сообщения от «1. Префлоп» до «Сверка денег…». Отступ у
# строк под заголовком — четыре пробела, как у остального разбора (вердикт, требование
# к диапазону): макет спеки набран с тремя.
_REFERENCE_POINTS = """\
1. Префлоп · BB · 5♣3♣ · сыграно: колл 1.0 ББ
    банк до хода 4.3 ББ · доставить 1.0 ББ · эфф. 37.0 ББ · живых 2 (после вас 0)
    шансы банка: колл окупается от 18.7% эквити
    вердикта нет: нет чарта защиты BB против опена UTG+1 на 37 ББ

2. Флоп Q♥4♦2♦ · BB · сыграно: чек
    банк 5.3 ББ · эфф. 35.0 ББ · SPR 6.6 · живых 2 (после вас 1)
    рука: старшая 5, без пары
    дро: двусторонний стрит (A, 6) · 8 аутов · шанс собрать: на тёрне 17.0%, тёрн + ривер 31.5%

3. Тёрн 7♠ · BB · сыграно: бет 3.0 ББ (56% банка)
    банк 5.3 ББ · эфф. 35.0 ББ · SPR 6.6 · живых 2 (после вас 1)
    рука: старшая 5, без пары
    дро: двусторонний стрит (A, 6) · 8 аутов · шанс собрать на ривере 17.4%
    линия: проба · полублеф
    окупается: как чистый блеф от 36% фолдов; с учётом аутов от 16% фолдов
    вердикта нет: частота фолдов оппонента зависит от его диапазона

4. Ривер Q♦ · BB · сыграно: бет 8.5 ББ (75% банка)
    банк 11.3 ББ · эфф. 32.0 ББ · SPR 2.8 · живых 2 (после вас 1)
    рука: пара дам на борде, играет кикер 5 · дро не закрылось
    ценность на вскрытии: нулевая (не выигрывает ни у одной руки, делит банк только с 5-3)
    линия: повторная ставка · блеф
    окупается: от 43% фолдов
    вердикта нет: частота фолдов оппонента зависит от его диапазона"""


def _messages(en: EnrichedHand, **kw) -> list[Msg]:
    kw.setdefault("replay", hand_replay(en))
    return hand_analysis_msgs(analyze_hand(en), en, 12, None, 17, 50, **kw)


def _text(en: EnrichedHand) -> str:
    (msg,) = _messages(en)
    return msg.text


def _points_part(text: str) -> str:
    """Пронумерованная часть: от «1. » до пустой строки перед сверкой денег."""
    start = text.index("\n\n1. ") + 2
    return text[start : text.index("\n\nСверка денег")]


def _blocks(text: str) -> list[str]:
    """Блоки точек по пустой строке между ними."""
    return _points_part(text).split("\n\n")


def test_the_reference_hand_prints_the_owners_layout_line_by_line():
    """Гейт 1: рука §2 печатает ровно точки макета (отступ — см. `_REFERENCE_POINTS`)."""
    text = _text(_reference_hand())
    assert _points_part(text) == _REFERENCE_POINTS
    assert text.split("\n\n")[-2].startswith("Сверка денег с источником: сошлась.")


def test_the_effective_stack_is_signed_short_and_the_old_signature_is_gone():
    """Гейт 9: «эфф. 37.0 / 35.0 / 35.0 / 32.0 ББ», подписи «эфф. стек» в тексте нет."""
    text = _text(_reference_hand())
    assert re.findall(r"эфф\. (\d+\.\d) ББ", text) == ["37.0", "35.0", "35.0", "32.0"]
    assert "эфф. стек" not in text
    assert "стек" not in _points_part(text)


def test_the_point_title_writes_cards_like_the_replay_block():
    """Масти — символом, как в блоке «Что было:»: те же символы, без букв и без U+FE0F."""
    en = _reference_hand()
    text = _text(en)
    titles = [block.splitlines()[0] for block in _blocks(text)]
    assert [t.split(" · ")[0] for t in titles] == [
        "1. Префлоп",
        "2. Флоп Q♥4♦2♦",
        "3. Тёрн 7♠",
        "4. Ривер Q♦",
    ]
    replay = hand_replay(en).plain
    for card in ("5♣3♣", "Q♥", "4♦", "2♦", "7♠", "Q♦"):
        assert card in titles[0] + titles[1] + titles[2] + titles[3]
    assert "Q♥ 4♦ 2♦" in replay, "борд в реплее — через пробел, флоп в заголовке — подряд"
    for title in titles:
        assert "️" not in title
        assert not re.search(r"\d[shdc]\b", title), f"масть буквой: {title}"


def test_the_preflop_title_has_no_cards_when_the_hero_cards_are_unknown():
    en = _reference_hand()
    blind = en.model_copy(update={"hand": en.hand.model_copy(update={"dealt": {}})})
    assert _blocks(_text(blind))[0].splitlines()[0] == "1. Префлоп · BB · сыграно: колл 1.0 ББ"


# --- заголовок: «сыграно» --------------------------------------------------------------


@pytest.mark.parametrize(
    ("share", "percent"),
    [(0.5618, 56), (0.7503, 75), (0.285, 29), (0.284999, 28), (0.145, 15), (0.005, 1), (0.0049, 0)],
)
def test_a_percent_of_the_pot_rounds_half_up(share, percent):
    assert _half_up_percent(share) == percent


def test_a_bet_and_a_raise_print_the_percent_of_the_pot_and_not_the_size_tag():
    """Слово «до» — только у рейза; тег размера хранится, но не печатается."""
    en = _oop([*_V_OPENS, _check(F, "Hero"), _bet(F, "V", 300), _raise(F, "Hero", 900)])
    detail = postflop_line_detail(analyze_hand(en).points[-1])
    assert detail is not None and detail.line is not None
    assert detail.line.size_tag is SizeTag.STANDARD
    text = _text(en)
    assert "сыграно: рейз до 9.0 ББ (50% банка)" in text
    for tag in ("блок", "стандарт", "крупная", "овербет"):
        assert tag not in _points_part(text)

    bet = _text(_oop([*_V_OPENS, _bet(F, "Hero", 300)]))
    assert "сыграно: бет 3.0 ББ (50% банка)" in bet
    assert "бет до" not in bet


def test_a_preflop_raise_prints_no_percent_of_the_pot():
    first = _blocks(_text(_hand(_HERO_OPENS, button="Hero")))[0].splitlines()[0]
    assert first.endswith("сыграно: рейз до 3.0 ББ")
    assert "%" not in first


def test_an_all_in_bet_keeps_the_percent_and_then_names_the_all_in():
    shove = _bet(F, "Hero", 600).model_copy(update={"is_all_in": True})
    en = _oop([*_V_OPENS, shove], seats=(("Hero", 900), ("V", 5_000)))
    first = next(
        line for line in _text(en).splitlines() if line.startswith("2. ") and "сыграно: бет" in line
    )
    assert first.endswith("сыграно: бет 6.0 ББ (100% банка), олл-ин")


# --- строка чисел ----------------------------------------------------------------------


def test_the_numbers_line_prints_only_what_there_is_to_say():
    """Гейт 10: «доставить 0.0» и «SPR» на префлопе не печатаются; «доставить» — только
    при доплате; на префлопе «банк до хода», на постфлопе «банк»."""
    blocks = _blocks(_text(_reference_hand()))
    preflop, flop = blocks[0].splitlines(), blocks[1].splitlines()
    assert "SPR" not in blocks[0]
    assert "доставить" in preflop[1]
    assert preflop[1].lstrip().startswith("банк до хода ")
    assert "доставить" not in blocks[1] and "0.0 ББ" not in flop[1]
    assert flop[1].lstrip().startswith("банк 5.3 ББ")
    assert "банк до хода" not in "\n".join(blocks[1:])
    for block in blocks:
        assert "доставить 0.0" not in block


def test_a_postflop_call_prints_the_pot_odds_without_repeating_the_pot_and_the_price():
    en = _gutshot_call(500_000)
    call = next(b for b in _blocks(_text(en)) if "сыграно: колл 100.0 ББ" in b and "Флоп" in b)
    lines = call.splitlines()
    assert lines[1].startswith("    банк 200.0 ББ · доставить 100.0 ББ · эфф. ")
    assert lines[2] == "    шансы банка: колл окупается от 33.3% эквити"


# --- сила руки словами ------------------------------------------------------------------

_HANDS = [
    (("5c", "3c"), ["Qh", "4d", "2d"], HandCategory.NO_PAIR, "старшая 5, без пары"),
    (("Ah", "Ad"), ["Kc", "7d", "2s"], HandCategory.OVERPAIR, "оверпара тузов"),
    (("Ks", "Kd"), ["Kc", "7d", "2h"], HandCategory.SET, "сет королей"),
    (("Kc", "9d"), ["Kh", "Kd", "2s"], HandCategory.TRIPS, "трипс королей"),
    (("Ah", "Qd"), ["Ac", "Qs", "2d"], HandCategory.TWO_PAIR, "две пары (тузов и дам)"),
    (("Ah", "Kd"), ["Kc", "7d", "2s"], HandCategory.TOP_PAIR_STRONG_KICKER,
     "топ-пара королей, сильный кикер"),
    (("Kh", "3d"), ["Kc", "7d", "2s"], HandCategory.TOP_PAIR_WEAK_KICKER,
     "топ-пара королей, слабый кикер"),
    (("7h", "3d"), ["Kc", "7d", "2s"], HandCategory.MIDDLE_PAIR, "средняя пара семёрок"),
    (("2h", "3d"), ["Kc", "7d", "2s"], HandCategory.WEAK_PAIR, "слабая пара двоек"),
    (("9h", "8d"), ["7c", "6d", "5s"], HandCategory.STRAIGHT, "стрит (старшая 9)"),
    (("Ah", "2h"), ["Kh", "7h", "3h"], HandCategory.FLUSH, "флеш (старшая A)"),
    (("Kh", "Kd"), ["Kc", "7d", "7s"], HandCategory.FULL_HOUSE,
     "фулл-хаус (тройка королей, пара семёрок)"),
    (("Kh", "Kd"), ["Kc", "Ks", "2d"], HandCategory.QUADS, "каре королей"),
    (("9h", "8h"), ["7h", "6h", "5h"], HandCategory.STRAIGHT_FLUSH, "стрит-флеш (старшая 9)"),
    (("2h", "Ad"), ["Kh", "7h", "3h", "9h"], HandCategory.WEAK_FLUSH, "слабый флеш (старшая K)"),
    (("9c", "2d"), ["5c", "6d", "7h", "8s"], HandCategory.WEAK_STRAIGHT, "слабый стрит (старшая 9)"),
    (("2c", "3d"), ["Ah", "Ad", "Kc", "Kd", "Qs"], HandCategory.ON_BOARD,
     "две пары (тузов и королей) на борде, играет борд"),
    (("5c", "3c"), ["Qh", "4d", "2d", "7s", "Qd"], HandCategory.ON_BOARD,
     "пара дам на борде, играет кикер 5"),
    (("2c", "3d"), ["Ah", "Ad", "Ac", "Kd", "Qs"], HandCategory.ON_BOARD,
     "трипс тузов на борде, играет борд"),
    (("2c", "3d"), ["9h", "8d", "7c", "6d", "5s"], HandCategory.ON_BOARD,
     "стрит (старшая 9) на борде, играет борд"),
]


@pytest.mark.parametrize(("hero", "board", "category", "words"), _HANDS)
def test_every_category_of_the_hand_is_named_in_words(hero, board, category, words):
    strength = hand_strength(hero, board)
    assert strength.category is category, "тест перестал держать свою категорию"
    assert _hand_words(strength) == words


def test_the_hand_words_cover_every_category_the_table_names():
    assert {category for *_, category, _ in _HANDS} == set(HandCategory)


def test_a_rank_is_named_in_the_genitive_plural():
    for rank, word in (("A", "тузов"), ("K", "королей"), ("Q", "дам"), ("5", "пятёрок")):
        strength = hand_strength(("Ah", "Kd"), ["2c", "7d", "9s"])
        assert strength.ranks == ["A"]
        pair = strength.model_copy(
            update={
                "category": HandCategory.WEAK_PAIR,
                "combination": Combination.PAIR,
                "ranks": [rank],
            }
        )
        assert _hand_words(pair) == f"слабая пара {word}"


# --- детали-конструкторы для строк без расчёта ------------------------------------------


def _detail(**fields) -> PostflopLineDetail:
    base = {
        "hand": None,
        "draw": None,
        "draw_missed": False,
        "backdoors": [],
        "overcards": None,
        "line": None,
        "fold_threshold": None,
        "draw_call": None,
        "showdown": None,
    }
    return PostflopLineDetail(**{**base, **fields})


def _printed(**fields) -> list[str]:
    return _postflop_line_lines(_detail(**fields), 100)


def _line(action: ActionTag, purpose: Purpose | None, barrel: int | None = None) -> Line:
    return Line(
        action=action, barrel=barrel, size_pct=0.5, size_tag=SizeTag.STANDARD, purpose=purpose
    )


def test_an_empty_field_prints_no_line():
    """Гейт 10: пустое поле не печатается — ни пустой подписи, ни прочерка."""
    assert _printed() == []


def test_every_new_line_stands_in_the_order_of_the_layout():
    weak = hand_strength(("As", "Kd"), ["9c", "7d", "2s"])
    draw = Draw(
        kinds=[DrawKind.GUTSHOT],
        out_ranks=["T"],
        outs=["Th", "Td", "Tc", "Ts"],
        unseen=47,
        hit_next=4 / 47,
        hit_by_river=0.16,
    )
    lines = _printed(
        hand=weak,
        showdown=ShowdownValue(wins=0, ties=0, losses=9, ties_with=[]),
        draw=draw,
        backdoors=[Backdoor(kind="flush", variants=None, approx=0.04)],
        overcards=Overcards(cards=["As", "Kd"], approx_by_river=0.12, approx_next=0.065),
        line=_line(ActionTag.DONK, Purpose.SEMIBLUFF),
        fold_threshold=FoldThreshold(bluff=0.4, semibluff=0.3, semibluff_free=False),
    )
    assert [line.split(":")[0].strip() for line in lines] == [
        "рука",
        "ценность на вскрытии",
        "дро",
        "бэкдор",
        "оверкарты",
        "линия",
        "окупается",
    ]


# --- рука на ривере, дро не закрылось, ценность на вскрытии -----------------------------


def test_a_missed_draw_is_a_suffix_of_the_hand_line_only_when_it_missed():
    strength = hand_strength(("5c", "3c"), ["Qh", "4d", "2d", "7s", "Qd"])
    assert _printed(hand=strength, draw_missed=True) == [
        "    рука: пара дам на борде, играет кикер 5 · дро не закрылось"
    ]
    assert _printed(hand=strength) == ["    рука: пара дам на борде, играет кикер 5"]


def test_the_showdown_value_names_only_the_extremes():
    zero = ShowdownValue(wins=0, ties=8, losses=982, ties_with=["53o", "53s"])
    assert _printed(showdown=zero) == [
        (
            "    ценность на вскрытии: нулевая (не выигрывает ни у одной руки, "
            "делит банк только с 5-3)"
        )
    ]
    nuts = ShowdownValue(wins=900, ties=2, losses=0, ties_with=["AKo"])
    assert _printed(showdown=nuts) == [
        "    ценность на вскрытии: натс (не проигрывает ни одной руке)"
    ]
    middle = ShowdownValue(wins=300, ties=0, losses=690, ties_with=[])
    assert _printed(showdown=middle) == []


def test_a_showdown_with_no_ties_names_no_ties_and_a_pair_class_is_named_once():
    alone = ShowdownValue(wins=0, ties=0, losses=990, ties_with=[])
    assert _printed(showdown=alone) == [
        "    ценность на вскрытии: нулевая (не выигрывает ни у одной руки)"
    ]
    several = ShowdownValue(wins=0, ties=9, losses=981, ties_with=["55", "64o", "64s"])
    (line,) = _printed(showdown=several)
    assert line.endswith("делит банк только с 55, 6-4)")


def test_the_nuts_are_nuts_even_when_nothing_is_won():
    """Рука, которой нечего проигрывать, не названа «нулевой»."""
    board_nuts = ShowdownValue(wins=0, ties=990, losses=0, ties_with=["AKo"])
    assert _printed(showdown=board_nuts) == [
        "    ценность на вскрытии: натс (не проигрывает ни одной руке)"
    ]


# --- дро --------------------------------------------------------------------------------


def _draw(kinds, out_ranks, outs: int, *, by_river: float | None = None) -> Draw:
    return Draw(
        kinds=kinds,
        out_ranks=out_ranks,
        outs=[f"{'A23456789TJQK'[i % 13]}{'shdc'[i // 13]}" for i in range(outs)],
        unseen=47 if by_river is not None else 46,
        hit_next=outs / (47 if by_river is not None else 46),
        hit_by_river=by_river,
    )


def test_the_draw_line_names_the_kind_the_outs_and_the_chance_to_hit():
    flop = _draw([DrawKind.OPEN_ENDED], ["A", "6"], 8, by_river=1 - 39 * 38 / (47 * 46))
    assert _printed(draw=flop) == [
        (
            "    дро: двусторонний стрит (A, 6) · 8 аутов · "
            "шанс собрать: на тёрне 17.0%, тёрн + ривер 31.5%"
        )
    ]
    turn = _draw([DrawKind.OPEN_ENDED], ["A", "6"], 8)
    assert _printed(draw=turn) == [
        "    дро: двусторонний стрит (A, 6) · 8 аутов · шанс собрать на ривере 17.4%"
    ]


def test_the_chance_to_hit_is_never_called_equity():
    flop = _draw([DrawKind.GUTSHOT], ["7"], 4, by_river=0.165)
    assert "эквити" not in _printed(draw=flop)[0]


@pytest.mark.parametrize(
    ("kinds", "ranks", "outs", "head"),
    [
        ([DrawKind.GUTSHOT], ["7"], 4, "гатшот (7) · 4 аута"),
        ([DrawKind.DOUBLE_GUTSHOT], ["J", "7"], 8, "двойной гатшот (J, 7) · 8 аутов"),
        ([DrawKind.FLUSH], [], 9, "флеш · 9 аутов"),
        ([DrawKind.FLUSH, DrawKind.OPEN_ENDED], ["J", "6"], 15, "флеш + двусторонний стрит (J, 6) · 15 аутов"),
        ([DrawKind.FLUSH], [], 12, "флеш · 12 аутов"),
    ],
)
def test_the_draw_kinds_and_the_outs_agree_with_their_counts(kinds, ranks, outs, head):
    (line,) = _printed(draw=_draw(kinds, ranks, outs))
    assert line.startswith(f"    дро: {head} · шанс собрать на ривере ")


# --- бэкдоры и оверкарты (§4.2.1) -------------------------------------------------------


def test_a_backdoor_line_marks_every_figure_as_approximate():
    flush = Backdoor(kind="flush", variants=None, approx=0.04)
    assert _printed(backdoors=[flush]) == ["    бэкдор: флеш ≈4%"]
    for variants, share, word in ((1, 0.015, "≈1.5%"), (2, 0.03, "≈3%"), (3, 0.045, "≈4.5%")):
        straight = Backdoor(kind="straight", variants=variants, approx=share)
        assert _printed(backdoors=[straight]) == [f"    бэкдор: стрит {word}"]
    assert _printed(backdoors=[flush, Backdoor(kind="straight", variants=2, approx=0.03)]) == [
        "    бэкдор: флеш ≈4% · стрит ≈3%"
    ]


def test_the_overcards_line_carries_the_condition_and_the_street_of_the_figure():
    two = Overcards(cards=["Ah", "Kd"], approx_by_river=0.12, approx_next=0.065)
    assert _printed(overcards=two) == [
        "    оверкарты: 2 (A, K) · ≈12% к риверу, если пара будет лучшей"
    ]
    one_flop = Overcards(cards=["Ah"], approx_by_river=0.065, approx_next=0.03)
    assert _printed(overcards=one_flop) == [
        "    оверкарты: 1 (A) · ≈6.5% к риверу, если пара будет лучшей"
    ]
    one_turn = Overcards(cards=["Ah"], approx_by_river=None, approx_next=0.03)
    assert _printed(overcards=one_turn) == [
        "    оверкарты: 1 (A) · ≈3% на ривере, если пара будет лучшей"
    ]


def test_a_flop_without_a_draw_prints_the_backdoor_and_the_overcards_from_the_analysis():
    """Тот же формат через настоящий расчёт: A♣K♣ на 9♣7♦2♠ — две оверкарты и бэкдор флеша."""
    en = _oop(
        [*_V_OPENS, _bet(F, "Hero", 300)],
        hero_cards=("Ac", "Kc"),
        boards={F: ["9c", "7d", "2s"], T: ["3h"], R: ["4h"]},
    )
    flop = _blocks(_text(en))[1]
    assert "    бэкдор: флеш ≈4%" in flop.splitlines()
    assert "    оверкарты: 2 (A, K) · ≈12% к риверу, если пара будет лучшей" in flop.splitlines()
    assert "дро:" not in flop


# --- линия ------------------------------------------------------------------------------


def test_the_line_names_the_kind_and_the_purpose():
    assert _printed(line=_line(ActionTag.PROBE, Purpose.SEMIBLUFF)) == [
        "    линия: проба · полублеф"
    ]
    assert _printed(line=_line(ActionTag.REPEAT_BET, Purpose.BLUFF)) == [
        "    линия: повторная ставка · блеф"
    ]


def test_a_plain_call_and_a_plain_bet_print_the_purpose_alone():
    assert _printed(line=_line(ActionTag.BET, Purpose.VALUE)) == ["    линия: вэлью"]
    assert _printed(line=_line(ActionTag.CALL, Purpose.CALL_STRONG)) == [
        "    линия: колл с сильной рукой"
    ]
    assert _printed(line=_line(ActionTag.CALL, Purpose.BLUFF_CATCH)) == ["    линия: ловля блефа"]
    assert _printed(line=_line(ActionTag.BET, Purpose.MEDIUM_HAND)) == [
        "    линия: ставка со средней рукой"
    ]


def test_a_barrel_is_named_by_its_number():
    assert _printed(line=_line(ActionTag.BARREL, Purpose.VALUE, barrel=2)) == [
        "    линия: второй баррель · вэлью"
    ]
    assert _printed(line=_line(ActionTag.BARREL, Purpose.BLUFF, barrel=3)) == [
        "    линия: третий баррель · блеф"
    ]


def test_a_line_of_unknown_cards_has_the_kind_and_no_purpose():
    assert _printed(line=_line(ActionTag.DONK, None)) == ["    линия: донк"]
    assert _printed(line=_line(ActionTag.BET, None)) == []


def test_a_fold_and_a_check_have_no_line():
    """Гейт 10: у чека и фолда строки «линия» нет."""
    assert _printed(line=None) == []
    assert _printed(line=_line(ActionTag.FOLD, None)) == []
    assert _printed(line=_line(ActionTag.CHECK_FOLD, None)) == []
    flop_check = _blocks(_text(_reference_hand()))[1]
    assert "линия" not in flop_check


def test_every_tag_and_every_purpose_has_its_own_word_or_is_deliberately_silent():
    assert set(_ACTION_TAG_WORD) | _UNPRINTED_TAGS | {ActionTag.BARREL} == set(ActionTag)
    assert not set(_ACTION_TAG_WORD) & _UNPRINTED_TAGS
    assert set(_PURPOSE_WORD) == set(Purpose)
    assert set(_DRAW_KIND_WORD) == set(DrawKind)
    for tag in set(ActionTag) - {ActionTag.FOLD, ActionTag.CHECK_FOLD}:
        printed = _printed(line=_line(tag, Purpose.VALUE, barrel=2 if tag is ActionTag.BARREL else None))
        assert printed and printed[0].startswith("    линия: "), tag


# --- «окупается» ------------------------------------------------------------------------


def test_a_bluff_and_a_semibluff_print_the_fold_threshold_in_whole_percents():
    bluff = _line(ActionTag.REPEAT_BET, Purpose.BLUFF)
    assert _printed(
        line=bluff, fold_threshold=FoldThreshold(bluff=0.4287, semibluff=None, semibluff_free=False)
    ) == ["    линия: повторная ставка · блеф", "    окупается: от 43% фолдов"]
    semi = _line(ActionTag.PROBE, Purpose.SEMIBLUFF)
    assert _printed(
        line=semi,
        fold_threshold=FoldThreshold(bluff=0.3597, semibluff=0.16141, semibluff_free=False),
    )[-1] == "    окупается: как чистый блеф от 36% фолдов; с учётом аутов от 16% фолдов"


def test_a_semibluff_that_pays_without_folds_says_so():
    semi = _line(ActionTag.PROBE, Purpose.SEMIBLUFF)
    (_, payoff) = _printed(
        line=semi, fold_threshold=FoldThreshold(bluff=0.3, semibluff=0.0, semibluff_free=True)
    )
    assert payoff == "    окупается: как чистый блеф от 30% фолдов; с учётом аутов окупается и без фолдов"


def test_a_semibluff_without_the_outs_threshold_claims_nothing_about_the_outs():
    semi = _line(ActionTag.PROBE, Purpose.SEMIBLUFF)
    (_, payoff) = _printed(
        line=semi, fold_threshold=FoldThreshold(bluff=0.3, semibluff=None, semibluff_free=False)
    )
    assert payoff == "    окупается: как чистый блеф от 30% фолдов"


def test_value_and_a_medium_hand_print_no_threshold():
    threshold = FoldThreshold(bluff=0.3, semibluff=None, semibluff_free=False)
    for purpose in (Purpose.VALUE, Purpose.MEDIUM_HAND):
        lines = _printed(line=_line(ActionTag.BET, purpose), fold_threshold=threshold)
        assert not [line for line in lines if "окупается" in line]


def test_a_draw_call_by_the_pot_odds_says_so_without_a_number():
    call = DrawCallDetail(
        required_equity=0.25, hit=0.33, by_pot_odds=True, implied_needed_chips=None, beyond_stack=False
    )
    assert _printed(draw_call=call) == ["    окупается: по шансам банка"]


def test_a_draw_call_short_of_the_pot_odds_names_the_amount_to_win_later():
    en = _gutshot_call(500_000)
    res = analyze_hand(en)
    detail = next(
        d
        for p in res.points
        if (d := postflop_line_detail(p)) is not None and d.draw_call is not None
    )
    assert detail.draw_call is not None and detail.draw_call.implied_needed_chips is not None
    chips = detail.draw_call.implied_needed_chips
    block = next(b for b in _blocks(_text(en)) if "окупается:" in b)
    printed = -(-10 * chips // en.hand.bb) / 10
    assert f"    окупается: нужно добрать позже {printed:.1f} ББ" in block.splitlines()


@pytest.mark.parametrize(
    ("chips", "printed"), [(1_234, "12.4"), (1_230, "12.3"), (1_201, "12.1"), (5, "0.1")]
)
def test_the_amount_to_win_later_rounds_up_to_a_tenth(chips, printed):
    """X — требование: 1 234 фишки при ББ 100 печатаются «12.4», а не «12.3» —
    округление к ближайшему занизило бы его на 0.04 ББ."""
    call = DrawCallDetail(
        required_equity=0.25,
        hit=0.1,
        by_pot_odds=False,
        implied_needed_chips=chips,
        beyond_stack=False,
    )
    assert _printed(draw_call=call) == [f"    окупается: нужно добрать позже {printed} ББ"]


def test_an_amount_beyond_the_stacks_is_a_line_and_not_a_verdict():
    en = _gutshot_call(_DEEP)
    block = next(b for b in _blocks(_text(en)) if "окупается:" in b)
    assert "    окупается: добрать столько нельзя — колл не окупается добором" in block.splitlines()
    assert "вердикт:" not in block, "точка остаётся без вердикта"
    assert "Лучше:" not in block


def test_the_draw_call_of_the_analysis_by_the_pot_odds():
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
    block = next(b for b in _blocks(_text(en)) if "сыграно: колл 3.0 ББ" in b)
    assert "    линия: чек-колл · колл с дро" in block.splitlines()
    assert "    окупается: по шансам банка" in block.splitlines()


# --- вердикта нет -----------------------------------------------------------------------


def test_an_empty_reason_prints_no_verdict_line_and_a_named_one_does():
    """Пустая причина (чек на постфлопе) — строки нет; названная — печатается."""
    blocks = _blocks(_text(_reference_hand()))
    assert "вердикта нет" not in blocks[1]
    assert "    вердикта нет: нет чарта защиты BB против опена UTG+1 на 37 ББ" in blocks[0]
    assert "    вердикта нет: частота фолдов оппонента зависит от его диапазона" in blocks[2]
    assert "    вердикта нет: частота фолдов оппонента зависит от его диапазона" in blocks[3]
    assert not [line for block in blocks for line in block.splitlines() if line == "    вердикта нет."]


def test_the_postflop_line_is_not_printed_as_a_raw_dictionary():
    """Ключ — в `skip` и не в таблице подписей: он печатается своими строками."""
    assert "postflop_line" not in _DETAIL_LABELS
    text = _text(_reference_hand())
    for fragment in ("postflop_line", "category:", "size_tag", "hit_next", "semibluff_free"):
        assert fragment not in text


# --- гейт 10: длина ---------------------------------------------------------------------


def test_a_hand_of_four_streets_goes_out_in_two_messages_by_the_existing_rule():
    """Рука из четырёх улиц с длинной линией: точки не режутся, уезжают второму целиком."""
    from harness.explanation import HandReplay, ReplaySpan

    en = _reference_hand()
    for filler in range(2_500, 3_700, 50):
        replay = HandReplay(spans=[ReplaySpan(text="а" * filler)])
        msgs = _messages(en, replay=replay)
        if len(msgs) == 2:
            break
    else:
        pytest.fail("не нашлось длины реплея, при которой точки делятся на два сообщения")
    first, second = msgs
    assert len(first.text) <= 4096 and len(second.text) <= 4096
    kept = _points_part(first.text).split("\n\n")
    moved = second.text.split("\n\n", 1)[1].split("\n\n")
    assert kept and moved, "обе половины несут хотя бы одну точку"
    assert "\n\n".join([*kept, *moved]) == _REFERENCE_POINTS, "ни одна точка не потеряна и не порезана"
    assert re.match(r"^\d+\. (Префлоп|Флоп|Тёрн|Ривер)", moved[0])


# --- гейт 11: деньги --------------------------------------------------------------------


def _money_the_hand_contains(en: EnrichedHand) -> set[float]:
    """Суммы в ББ, которые вправе стоять в разборе: руки, движка и `detail` ядра.

    К суммам руки и ядра добавлена одна производная сумма ядра — `X` из `detail`
    (сколько добрать позже, `DrawCall.implied_needed_chips`, в ББ вверх до
    десятой, как печатается), решение спеки постфлоп-линии, §9 гейт 11.
    """
    from harness.contracts.enriched import hero_stack_delta_bb

    hand = en.hand
    allowed = {round(v / hand.bb, 1) for v in en.report.pot_by_street.values()}
    allowed |= {round(en.report.final_pot / hand.bb, 1)}
    allowed |= {round(p.stack / hand.bb, 1) for p in hand.players}
    allowed |= {round(v / hand.bb, 1) for v in en.report.stacks_end.values()}
    allowed |= {round(pot.amount / hand.bb, 1) for pot in en.report.side_pots}
    allowed |= {abs(round(hero_stack_delta_bb(en), 1))}
    for dp in en.report.decision_points:
        allowed |= {
            round(dp.to_call / hand.bb, 1),
            round(dp.pot_before / hand.bb, 1),
            round(dp.eff_stack / hand.bb, 1),
            round(dp.eff_stack_bb, 1),
            round(dp.action.committed_after / hand.bb, 1),
        }
    for point in analyze_hand(en).points:
        detail = postflop_line_detail(point)
        if detail is not None and detail.draw_call is not None:
            chips = detail.draw_call.implied_needed_chips
            if chips is not None:
                allowed.add(-(-10 * chips // hand.bb) / 10)
    return allowed


@pytest.mark.parametrize(
    "make",
    [
        _reference_hand,
        lambda: _gutshot_call(500_000),
        lambda: _gutshot_call(_DEEP),
        lambda: _oop([*_V_OPENS, _check(F, "Hero"), _bet(F, "V", 300), _raise(F, "Hero", 900)]),
    ],
    ids=["reference", "amount-to-win-later", "beyond-the-stacks", "raise"],
)
def test_the_postflop_line_prints_no_money_the_hand_and_the_detail_do_not_contain(make):
    """Гейт 11: новые строки не печатают сумм вне руки, расчёта и `X` из `detail`."""
    en = make()
    text = _text(en)
    money = {float(n) for n in re.findall(r"(\d+(?:\.\d+)?)\s*ББ", text)}
    assert money, "в тексте нет ни одной суммы — тест ничего не значит"
    assert money <= _money_the_hand_contains(en) | _ev_money(en), (
        f"выдуманные суммы: {money - _money_the_hand_contains(en)}"
    )


def _ev_money(en: EnrichedHand) -> set[float]:
    """Цена и интервал ядра, если у руки есть судимые точки (у этих рук их нет)."""
    res = analyze_hand(en)
    allowed = {abs(round(p.ev_diff_bb, 1)) for p in res.points}
    allowed |= {abs(round(res.total_ev_loss_bb, 1))}
    return allowed


def test_the_amount_to_win_later_is_among_the_allowed_sums_and_not_by_accident():
    """`X` стоит в тексте и входит в разрешённые только как `X` из `detail`."""
    en = _gutshot_call(500_000)
    text = _text(en)
    (x_text,) = re.findall(r"нужно добрать позже (\d+\.\d) ББ", text)
    plain = {round(v / en.hand.bb, 1) for v in en.report.pot_by_street.values()}
    assert float(x_text) not in plain
    assert float(x_text) in _money_the_hand_contains(en)


# --- границы ---------------------------------------------------------------------------


def test_the_preflop_has_no_postflop_line_and_the_point_without_a_line_still_prints():
    en = _hand([*_V_OPENS], button="V", hero_cards=("As", "Ah"))
    res = analyze_hand(en)
    assert all(
        postflop_line_detail(p) is None for p in res.points if p.street is P
    ), "на префлопе ключа нет"
    assert "Префлоп" in _text(en)


def test_a_point_whose_detail_has_no_line_prints_no_new_lines():
    """Старая точка (до этой фичи) без ключа: числа и вердикт те же, новых строк нет."""
    en = _reference_hand()
    res = analyze_hand(en)
    stripped = res.model_copy(
        update={
            "points": [
                p.model_copy(update={"detail": {k: v for k, v in p.detail.items() if k != "postflop_line"}})
                for p in res.points
            ]
        }
    )
    (msg,) = hand_analysis_msgs(stripped, en, 12, None, 17, 50, replay=hand_replay(en))
    for word in ("рука:", "дро:", "линия:", "окупается:"):
        assert word not in msg.text
    assert "2. Флоп Q♥4♦2♦ · BB · сыграно: чек" in msg.text
