"""Реплей руки по улицам (задача 21, спека §5.6): чистый код, ноль токенов.

Синтетика проходит настоящий конвейер (`normalize` → `enrich`), как в
`test_tournament_report.py`: банк и стеки в реплее обязаны быть теми, что
посчитал движок, и подставлять их руками значило бы проверять реплей на входе,
которого конвейер никогда не произведёт.

Главные утверждения этого файла — не «текст выглядит так», а четыре запрета,
каждый из которых уже стоил бы игроку доверия: масти буквами, выдуманное число,
шапка длиннее двух строк и потерянная точка решения героя.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from harness.contracts import (
    ActionKind,
    EnrichedHand,
    Post,
    PostKind,
    Provenance,
    RawAction,
    RawHand,
    SeatInfo,
    ShowdownEntry,
    Street,
    ValidationStatus,
)
from harness.engine import enrich
from harness.explanation import hand_replay
from harness.normalizer import normalize

_START = datetime(2026, 8, 20, 21, 30, tzinfo=UTC)

# Шесть мест, кнопка на 6-м: позиции нормалайзера — SB, BB, UTG, HJ, CO, BTN.
_SEATS = [
    SeatInfo(seat=1, label="Hero", stack=1000),
    SeatInfo(seat=2, label="P2", stack=5000),
    SeatInfo(seat=3, label="P3", stack=5000),
    SeatInfo(seat=4, label="P4", stack=5000),
    SeatInfo(seat=5, label="P5", stack=5000),
    SeatInfo(seat=6, label="P6", stack=5000),
]
_POSTS = [Post(label=s.label, kind=PostKind.ANTE, amount=10) for s in _SEATS] + [
    Post(label="Hero", kind=PostKind.SMALL_BLIND, amount=50),
    Post(label="P2", kind=PostKind.BIG_BLIND, amount=100),
]


def _raw(
    *,
    actions: list[RawAction],
    dealt: dict[str, list[str]],
    boards: dict[Street, list[str]] | None = None,
    showdowns: list[ShowdownEntry] | None = None,
) -> RawHand:
    return RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="synthetic",
        hand_no="SYN1",
        tournament_id="TSYN",
        tournament_name="synthetic",
        level=12,
        sb=50,
        bb=100,
        ante=10,
        timestamp=_START,
        table_name="syn",
        max_seats=6,
        button_seat=6,
        seats=_SEATS,
        posts=_POSTS,
        dealt=dealt,
        actions=actions,
        boards=boards or {},
        showdowns=showdowns or [],
    )


def _enriched(raw: RawHand) -> EnrichedHand:
    en = enrich(normalize(raw))
    assert en.verdict.status is not ValidationStatus.REJECT, en.verdict.reasons
    return en


def _fold(label: str) -> RawAction:
    return RawAction(
        street=Street.PREFLOP, label=label, kind=ActionKind.FOLD, raw_line=f"{label}: folds"
    )


def _preflop_shove_hand() -> EnrichedHand:
    """Герой (SB, 10bb) шовит против рейза CO, тот коллирует. Постфлопа нет.

    Улицы после префлопа проходят без единого действия — на них и проверяется
    схлопывание тихих улиц в одну строку.
    """
    return _enriched(
        _raw(
            actions=[
                _fold("P3"),  # UTG
                _fold("P4"),  # HJ
                RawAction(
                    street=Street.PREFLOP,
                    label="P5",  # CO
                    kind=ActionKind.RAISE,
                    to_amount=250,
                    raw_line="P5: raises 150 to 250",
                ),
                _fold("P6"),  # BTN
                RawAction(
                    street=Street.PREFLOP,
                    label="Hero",
                    kind=ActionKind.RAISE,
                    to_amount=990,
                    is_all_in=True,
                    raw_line="Hero: raises 940 to 990 and is all-in",
                ),
                _fold("P2"),  # BB
                RawAction(
                    street=Street.PREFLOP,
                    label="P5",
                    kind=ActionKind.CALL,
                    amount=740,
                    raw_line="P5: calls 740",
                ),
            ],
            dealt={"Hero": ["Jh", "9h"], "P5": ["Ks", "Kd"]},
            boards={
                Street.FLOP: ["6s", "Jd", "Qd"],
                Street.TURN: ["7h"],
                Street.RIVER: ["Ah"],
            },
            showdowns=[
                ShowdownEntry(label="Hero", cards=["Jh", "9h"]),
                ShowdownEntry(label="P5", cards=["Ks", "Kd"]),
            ],
        )
    )


def _postflop_hand() -> EnrichedHand:
    """Герой доходит до флопа и пасует на ставку — улица с действиями героя."""
    return _enriched(
        _raw(
            actions=[
                _fold("P3"),
                _fold("P4"),
                RawAction(
                    street=Street.PREFLOP,
                    label="P5",
                    kind=ActionKind.RAISE,
                    to_amount=250,
                    raw_line="P5: raises 150 to 250",
                ),
                _fold("P6"),
                RawAction(
                    street=Street.PREFLOP,
                    label="Hero",
                    kind=ActionKind.CALL,
                    amount=200,
                    raw_line="Hero: calls 200",
                ),
                _fold("P2"),
                RawAction(
                    street=Street.FLOP,
                    label="Hero",
                    kind=ActionKind.CHECK,
                    raw_line="Hero: checks",
                ),
                RawAction(
                    street=Street.FLOP,
                    label="P5",
                    kind=ActionKind.BET,
                    amount=300,
                    raw_line="P5: bets 300",
                ),
                RawAction(
                    street=Street.FLOP,
                    label="Hero",
                    kind=ActionKind.FOLD,
                    raw_line="Hero: folds",
                ),
            ],
            dealt={"Hero": ["Jh", "9h"]},
            boards={Street.FLOP: ["6s", "Jd", "Qd"]},
        )
    )


def _preflop_fold_hand() -> EnrichedHand:
    """Герой пасует на префлопе, борда нет — самая короткая раздача из возможных."""
    return _enriched(
        _raw(
            actions=[
                _fold("P3"),
                _fold("P4"),
                RawAction(
                    street=Street.PREFLOP,
                    label="P5",
                    kind=ActionKind.RAISE,
                    to_amount=250,
                    raw_line="P5: raises 150 to 250",
                ),
                _fold("P6"),
                _fold("Hero"),
                _fold("P2"),
            ],
            dealt={"Hero": ["3c", "2d"]},
        )
    )


def _lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


# --- форма и объём ------------------------------------------------------------------


def test_header_is_two_lines_with_level_blinds_and_hero():
    """Шапка ровно в две строки (спека §5.6): турнир с уровнем и блайндами, затем
    герой с позицией, картами и стеком в фишках и bb."""
    text = hand_replay(_preflop_shove_hand()).plain
    head, hero, *_ = text.splitlines()
    assert "ур. 12" in head
    assert "50/100" in head
    assert head.startswith("TSYN")
    assert hero.startswith("Hero SB")
    assert "10.0bb" in hero


def test_a_preflop_hand_stays_within_the_size_budget():
    """Ориентир спеки: префлоп-рука — 4–5 строк. Длиннее — формат нарушен."""
    assert len(_lines(hand_replay(_preflop_fold_hand()).plain)) <= 5


def test_a_hand_with_postflop_stays_within_its_own_budget():
    """Тот же ориентир для руки с борда — 6–8 строк; шов с прогоном тоже сюда."""
    assert len(_lines(hand_replay(_postflop_hand()).plain)) <= 8
    assert len(_lines(hand_replay(_preflop_shove_hand()).plain)) <= 8


# --- масти -------------------------------------------------------------------------


def test_suits_are_symbols_with_colour_and_never_letters():
    """Масти — символ плюс цвет, никогда буквы (спека §5.6, требование дословно).

    Цвет в тексте Телеграма даёт только эмодзи-презентация, поэтому за каждым
    символом масти стоит селектор U+FE0F: ♠️♣️ тёмные, ♥️♦️ красные. Буквенная
    нотация не должна встречаться ни в одном виде — ни `Jh`, ни `J h`.
    """
    text = hand_replay(_preflop_shove_hand()).plain
    for suit in "♠♥♦♣":
        for position in (m.start() for m in re.finditer(suit, text)):
            assert text[position + 1] == "️", f"масть {suit} без цвета"
    assert "♥️" in text and "♠️" in text
    assert not re.search(r"\b[AKQJT2-9][shdc]\b", text)


# --- поток действий ------------------------------------------------------------------


def test_hero_decision_is_emphasised_inside_the_flow_not_on_its_own_line():
    """Точка решения героя выделяется прямо в потоке действий (спека §5.6)."""
    replay = hand_replay(_preflop_shove_hand())
    emphasised = [span.text for span in replay.spans if span.emphasis]
    assert emphasised, "точка решения героя не выделена вовсе"
    assert all("Hero" in span for span in emphasised)
    flow = next(line for line in replay.plain.splitlines() if "олл-ин" in line)
    assert "→" in flow, "выделенное действие вынесено из потока в отдельную строку"


def test_consecutive_folds_are_merged_into_one_token():
    """Однотипные фолды слипаются: `UTG/HJ фолд` вместо двух отдельных шагов."""
    assert "UTG/HJ фолд" in hand_replay(_preflop_shove_hand()).plain


def test_quiet_streets_collapse_into_a_single_line():
    """Улицы без действий и без ставок — одной строкой (`ТЁРН 7♥ · РИВЕР A♥`)."""
    line = next(
        line for line in _lines(hand_replay(_preflop_shove_hand()).plain) if "ТЁРН" in line
    )
    assert "РИВЕР" in line and "ФЛОП" in line


def test_a_street_with_action_gets_its_own_line_with_the_board():
    """Улица с действиями получает свой заголовок с бордом, а под ним — поток ходов."""
    lines = _lines(hand_replay(_postflop_hand()).plain)
    head = next(i for i, line in enumerate(lines) if line.startswith("ФЛОП"))
    assert "♠️" in lines[head] and "банк" in lines[head]
    assert "бет 300" in lines[head + 1] and "Hero" in lines[head + 1]


def test_raise_over_a_raise_is_called_a_3bet_preflop():
    """Второй рейз на префлопе — «3-бет»: счёт рейзов улицы, а не новый расчёт."""
    raw_actions = [
        _fold("P3"),
        _fold("P4"),
        RawAction(
            street=Street.PREFLOP,
            label="P5",
            kind=ActionKind.RAISE,
            to_amount=250,
            raw_line="P5: raises 150 to 250",
        ),
        RawAction(
            street=Street.PREFLOP,
            label="P6",
            kind=ActionKind.RAISE,
            to_amount=700,
            raw_line="P6: raises 450 to 700",
        ),
        _fold("Hero"),
        _fold("P2"),
        RawAction(
            street=Street.PREFLOP,
            label="P5",
            kind=ActionKind.FOLD,
            raw_line="P5: folds",
        ),
    ]
    en = _enriched(_raw(actions=raw_actions, dealt={"Hero": ["Jh", "9h"]}))
    text = hand_replay(en).plain
    assert "рейз 250" in text and "3-бет 700" in text


# --- числа: только то, что посчитал движок --------------------------------------------


def test_street_pot_is_the_engine_number_not_the_summary():
    """Банк улицы берётся из отчёта движка — единственного источника истины о деньгах."""
    en = _preflop_shove_hand()
    pot = en.report.pot_by_street[Street.PREFLOP]
    text = hand_replay(en).plain
    assert f"{pot:,}".replace(",", " ") in text


def test_the_replay_prints_no_number_the_hand_does_not_contain():
    """Ни одного выдуманного числа: каждое число реплея — либо сумма из руки, либо
    её глубина в bb, либо номер уровня.

    Проверка механическая ровно затем, что «реплей ничего не считает» — это
    утверждение о коде, а не о вкусе автора: любое новое число в шаблоне обязано
    сначала появиться в `EnrichedHand`.
    """
    en = _preflop_shove_hand()
    hand = en.hand
    allowed = {float(hand.level), float(hand.sb), float(hand.bb)}
    allowed |= {float(p.stack) for p in hand.players}
    allowed |= {round(p.stack_bb, 1) for p in hand.players}
    allowed |= {float(a.committed_after) for a in hand.actions}
    allowed |= {round(a.committed_after / hand.bb, 1) for a in hand.actions}
    allowed |= {float(v) for v in en.report.pot_by_street.values()}
    allowed |= {float(sum(post.amount for post in hand.posts))}
    allowed |= {float(sum(p.amount for p in hand.posts if p.kind is PostKind.ANTE))}
    # Ранги карт — не деньги: те же карты, что в `dealt` и `boards`, только
    # символом масти вместо буквы.
    allowed |= {
        float(card[0])
        for card in [*hand.dealt["Hero"], *(c for cs in hand.boards.values() for c in cs)]
        if card[0].isdigit()
    }
    text = re.sub(r"\s", "", hand_replay(en).plain)
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    assert numbers, "в реплее не осталось ни одного числа — тест перестал что-либо значить"
    assert set(numbers) <= allowed, f"выдуманные числа: {set(numbers) - allowed}"


def test_showdown_line_shows_the_cards_that_were_actually_shown():
    """Вскрытие — одной строкой, картами тех, кто их показал; комбинации не
    называются: назвать их значило бы оценить руку, а реплей ничего не считает."""
    line = next(
        line for line in _lines(hand_replay(_preflop_shove_hand()).plain) if "Вскрытие" in line
    )
    assert "Hero" in line and "K♠️K♦️" in line


def test_a_hand_without_a_showdown_says_nothing_about_one():
    assert "Вскрытие" not in hand_replay(_postflop_hand()).plain
