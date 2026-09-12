"""Реплей руки по улицам (задача 21, спека §5.6): чистый код, ноль токенов.

Синтетика проходит настоящий конвейер (`normalize` → `enrich`), как в
`test_tournament_report.py`: банк и стеки в реплее обязаны быть теми, что
посчитал движок, и подставлять их руками значило бы проверять реплей на входе,
которого конвейер никогда не произведёт.

Главные утверждения этого файла — не «текст выглядит так», а пять запретов,
каждый из которых уже стоил бы игроку доверия: масти буквами, выдуманное число,
фишки вместо ББ, «чек» там, где никто не ходил, и потерянная точка решения героя.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from harness.contracts import (
    ActionKind,
    Collected,
    EnrichedHand,
    PlayerStats,
    Post,
    PostKind,
    Provenance,
    RawAction,
    RawHand,
    SeatInfo,
    ShowdownEntry,
    Street,
    Uncalled,
    ValidationStatus,
    hero_stack_delta_bb,
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
    provenance: Provenance = Provenance.HAND_HISTORY,
) -> RawHand:
    return RawHand(
        provenance=provenance,
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


def _preflop_shove_hand(showdowns: list[ShowdownEntry] | None = None) -> EnrichedHand:
    """Герой (SB, 10bb) шовит против рейза CO, тот коллирует. Постфлопа нет.

    Улицы после префлопа проходят без единого действия — на них и проверяется
    прогон борда: печатается борд, и никаких приписанных игрокам ходов.

    `showdowns` — умолчание описывает настоящее вскрытие двоих дошедших; аргумент
    нужен случаю «спасовавший показал карту», где к ним добавляется третья запись.
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
            showdowns=showdowns
            or [
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


def _river_fold_hand(
    *, provenance: Provenance, showdowns: list[ShowdownEntry]
) -> EnrichedHand:
    """Герой доходит до ривера и пасует на ставку — до вскрытия не дошёл никто.

    Живых мест на ривере остаётся одно, поэтому ЛЮБАЯ запись в `showdowns` этой
    раздачи вскрытием быть не может, и различает их только провенанс: на скрине
    карманные карты героя зрение читает всегда, а в hand history запись
    появляется лишь тогда, когда игрок карту показал сам.
    """
    return _enriched(
        _raw(
            provenance=provenance,
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
                    street=Street.FLOP, label="Hero", kind=ActionKind.CHECK, raw_line="Hero: checks"
                ),
                RawAction(
                    street=Street.FLOP, label="P5", kind=ActionKind.CHECK, raw_line="P5: checks"
                ),
                RawAction(
                    street=Street.TURN, label="Hero", kind=ActionKind.CHECK, raw_line="Hero: checks"
                ),
                RawAction(
                    street=Street.TURN, label="P5", kind=ActionKind.CHECK, raw_line="P5: checks"
                ),
                RawAction(
                    street=Street.RIVER,
                    label="Hero",
                    kind=ActionKind.CHECK,
                    raw_line="Hero: checks",
                ),
                RawAction(
                    street=Street.RIVER,
                    label="P5",
                    kind=ActionKind.BET,
                    amount=300,
                    raw_line="P5: bets 300",
                ),
                RawAction(
                    street=Street.RIVER,
                    label="Hero",
                    kind=ActionKind.FOLD,
                    raw_line="Hero: folds",
                ),
            ],
            dealt={"Hero": ["Jh", "Ts"]},
            boards={
                Street.FLOP: ["6s", "Jd", "Qd"],
                Street.TURN: ["7h"],
                Street.RIVER: ["Ah"],
            },
            showdowns=showdowns,
        )
    )


def _river_fold_beside_a_showdown_hand() -> EnrichedHand:
    """Герой пасует на ривере, а двое соперников доходят до вскрытия.

    Заведена ради случая, где запись о показе стоит РЯДОМ с настоящим вскрытием:
    герой в ряд вскрывшихся не попадает, а его показ попадает в то же
    предложение и потому остаётся строчным.
    """
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
                RawAction(
                    street=Street.PREFLOP,
                    label="P6",
                    kind=ActionKind.CALL,
                    amount=250,
                    raw_line="P6: calls 250",
                ),
                RawAction(
                    street=Street.PREFLOP,
                    label="Hero",
                    kind=ActionKind.CALL,
                    amount=200,
                    raw_line="Hero: calls 200",
                ),
                _fold("P2"),
                *[
                    RawAction(
                        street=street,
                        label=label,
                        kind=ActionKind.CHECK,
                        raw_line=f"{label}: checks",
                    )
                    for street in (Street.FLOP, Street.TURN)
                    for label in ("Hero", "P5", "P6")
                ],
                RawAction(
                    street=Street.RIVER,
                    label="Hero",
                    kind=ActionKind.CHECK,
                    raw_line="Hero: checks",
                ),
                RawAction(
                    street=Street.RIVER,
                    label="P5",
                    kind=ActionKind.BET,
                    amount=300,
                    raw_line="P5: bets 300",
                ),
                RawAction(
                    street=Street.RIVER,
                    label="P6",
                    kind=ActionKind.CALL,
                    amount=300,
                    raw_line="P6: calls 300",
                ),
                RawAction(
                    street=Street.RIVER,
                    label="Hero",
                    kind=ActionKind.FOLD,
                    raw_line="Hero: folds",
                ),
            ],
            dealt={"Hero": ["Jh", "Ts"]},
            boards={
                Street.FLOP: ["6s", "Jd", "Qd"],
                Street.TURN: ["7h"],
                Street.RIVER: ["Ah"],
            },
            showdowns=[
                ShowdownEntry(label="P5", cards=["Ks", "Kd"]),
                ShowdownEntry(label="P6", cards=["8c", "8d"]),
                ShowdownEntry(label="Hero", cards=["Jh"]),
            ],
        )
    )


def _folded_through_shove_hand() -> EnrichedHand:
    """Рука владельца (2026-09-12) в синтетике: SB 24.6 ББ шовит, все пасуют.

    Цифры взяты с его скрина: банк 8 800 при bb 3 000 — те самые 2.9 ББ, которые
    блок обязан назвать. Сходятся они только так: 7 анте по 400 плюс большой
    блайнд 3 000 плюс уравненные героем 3 000. Остальные 70 400 его шова
    возвращаются непоколленной ставкой, и в банк не попадают.

    Стол свой, не общий (`_SEATS`): общий не даёт ни этих блайндов, ни семи мест.
    """
    seats = [
        SeatInfo(seat=1, label="Hero", stack=73_800),  # 24.6 ББ
        *[SeatInfo(seat=n, label=f"P{n}", stack=100_000) for n in range(2, 8)],
    ]
    raw = RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="synthetic",
        hand_no="SYN-SHOVE",
        tournament_id="TSYN",
        tournament_name="synthetic",
        level=20,
        sb=1_500,
        bb=3_000,
        ante=400,
        timestamp=_START,
        table_name="syn",
        max_seats=8,
        button_seat=7,
        seats=seats,
        posts=[Post(label=s.label, kind=PostKind.ANTE, amount=400) for s in seats]
        + [
            Post(label="Hero", kind=PostKind.SMALL_BLIND, amount=1_500),
            Post(label="P2", kind=PostKind.BIG_BLIND, amount=3_000),
        ],
        dealt={"Hero": ["Ah", "Qs"]},
        actions=[
            *[_fold(f"P{n}") for n in (3, 4, 5, 6, 7)],
            RawAction(
                street=Street.PREFLOP,
                label="Hero",
                kind=ActionKind.RAISE,
                to_amount=73_400,  # стек минус анте: больше поставить нечего
                is_all_in=True,
                raw_line="Hero: raises 71 900 to 73 400 and is all-in",
            ),
            _fold("P2"),
        ],
        uncalled=[Uncalled(label="Hero", amount=70_400)],
        collected=[Collected(label="Hero", amount=8_800)],
    )
    return _enriched(raw)


def _side_pot_loss_hand() -> EnrichedHand:
    """Герой забирает сайд-пот, но за раздачу теряет: олл-ин UTG выигрывает мейн.

    Ради знака фразы. У героя ЕСТЬ запись `collected` (380 фишек сайд-пота), а
    стек всё равно уменьшился на 110 — потому что в мейн-пот он вложил 290 и
    анте 10, а выиграл его не он. По записи выплат блок сказал бы «Забираете» на
    проигранной раздаче; по стеку — «Отдаёте», и это правда.
    """
    seats = [
        SeatInfo(seat=1, label="Hero", stack=1000),
        SeatInfo(seat=2, label="P2", stack=5000),
        SeatInfo(seat=3, label="P3", stack=300),  # короткий стек, олл-ин с UTG
        SeatInfo(seat=4, label="P4", stack=5000),
        SeatInfo(seat=5, label="P5", stack=5000),
        SeatInfo(seat=6, label="P6", stack=5000),
    ]
    raw = RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="synthetic",
        hand_no="SYN-SIDE",
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
        seats=seats,
        posts=[Post(label=s.label, kind=PostKind.ANTE, amount=10) for s in seats]
        + [
            Post(label="Hero", kind=PostKind.SMALL_BLIND, amount=50),
            Post(label="P2", kind=PostKind.BIG_BLIND, amount=100),
        ],
        dealt={"Hero": ["Kh", "Ks"], "P3": ["Ah", "As"], "P5": ["Qh", "Qs"]},
        actions=[
            RawAction(
                street=Street.PREFLOP,
                label="P3",
                kind=ActionKind.RAISE,
                to_amount=290,  # весь стек за вычетом анте
                is_all_in=True,
                raw_line="P3: raises 190 to 290 and is all-in",
            ),
            _fold("P4"),
            RawAction(
                street=Street.PREFLOP,
                label="P5",
                kind=ActionKind.RAISE,
                to_amount=480,
                raw_line="P5: raises 190 to 480",
            ),
            _fold("P6"),
            RawAction(
                street=Street.PREFLOP,
                label="Hero",
                kind=ActionKind.CALL,
                amount=430,
                raw_line="Hero: calls 430",
            ),
            _fold("P2"),
            *[
                RawAction(
                    street=street,
                    label=label,
                    kind=ActionKind.CHECK,
                    raw_line=f"{label}: checks",
                )
                for street in (Street.FLOP, Street.TURN, Street.RIVER)
                for label in ("Hero", "P5")
            ],
        ],
        boards={
            Street.FLOP: ["2c", "7d", "9h"],
            Street.TURN: ["3s"],
            Street.RIVER: ["4h"],
        },
        showdowns=[
            ShowdownEntry(label="P3", cards=["Ah", "As"]),
            ShowdownEntry(label="Hero", cards=["Kh", "Ks"]),
            ShowdownEntry(label="P5", cards=["Qh", "Qs"]),
        ],
        collected=[Collected(label="P3", amount=1030), Collected(label="Hero", amount=380)],
    )
    return _enriched(raw)


def _fabricated_showdown_hand() -> EnrichedHand:
    """Скрин: соперник дошёл до вскрытия, а его карт зрение не прочитало.

    Движок добирает неизвестные карты из остатка колоды и разыгрывает вскрытие
    ими, валидатор ставит пометку в `not_checked` и оставляет статус `pass`.
    Стек героя на конец такой руки — исход выдуманного вскрытия; строки «Победа»
    на экране нет, поэтому `collected` пуст и сверка выплат молчит.
    """
    return _enriched(
        _raw(
            provenance=Provenance.SCREENSHOT,
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
                    kind=ActionKind.RAISE,
                    to_amount=990,
                    is_all_in=True,
                    raw_line="Hero: raises 940 to 990 and is all-in",
                ),
                _fold("P2"),
                RawAction(
                    street=Street.PREFLOP,
                    label="P5",
                    kind=ActionKind.CALL,
                    amount=740,
                    raw_line="P5: calls 740",
                ),
            ],
            dealt={"Hero": ["Jh", "9h"]},  # карт соперника на экране не видно
            # Борд подобран так, чтобы выдуманное вскрытие герой ПРОИГРАЛ:
            # остаток колоды движок берёт по порядку (`replay._DECK`), и
            # непрочитанному сопернику достаётся 4♣️4♦️ — сет на этом борде.
            # Выиграй герой, блок промолчал бы и без страховки (банка рум не
            # записал), и тест ничего бы не проверял.
            boards={
                Street.FLOP: ["Kc", "Qd", "2s"],
                Street.TURN: ["3h"],
                Street.RIVER: ["4s"],
            },
            showdowns=[ShowdownEntry(label="Hero", cards=["Jh", "9h"])],
        )
    )


def _untouched_stack_hand() -> EnrichedHand:
    """Раздача без анте: герой на BTN пасует, и стек его не меняется ни на фишку.

    Простейший способ получить ровный ноль: анте платят все и всегда, поэтому на
    столе с анте пас героя стоит ему как минимум анте. Ноль бывает и там —
    например, сплит, вернувший ровно вложенное, — но такую руку пришлось бы
    подгонять до фишки, а проверяется здесь не она.
    """
    seats = [
        SeatInfo(seat=1, label="Hero", stack=1000),
        *[SeatInfo(seat=n, label=f"P{n}", stack=5000) for n in range(2, 7)],
    ]
    raw = RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="synthetic",
        hand_no="SYN-FLAT",
        tournament_id="TSYN",
        tournament_name="synthetic",
        level=1,
        sb=50,
        bb=100,
        ante=0,
        timestamp=_START,
        table_name="syn",
        max_seats=6,
        button_seat=1,  # кнопка у героя: блайнды платят P2 и P3
        seats=seats,
        posts=[
            Post(label="P2", kind=PostKind.SMALL_BLIND, amount=50),
            Post(label="P3", kind=PostKind.BIG_BLIND, amount=100),
        ],
        dealt={"Hero": ["7c", "2d"]},
        actions=[_fold("P4"), _fold("P5"), _fold("P6"), _fold("Hero"), _fold("P2")],
    )
    return _enriched(raw)


def _lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


# --- форма: проза, ББ, «вы» ----------------------------------------------------------


def test_header_is_one_line_with_position_cards_and_stack():
    """Шапка — одна строка (спека §5.6): позиция героя, карманные карты, стек в ББ."""
    assert hand_replay(_preflop_shove_hand()).plain.startswith("Вы на SB, J♥️9♥️, 10.0 ББ.\n")


def test_the_replay_speaks_in_big_blinds_not_chips():
    text = hand_replay(_postflop_hand()).plain
    assert "250" not in text and "300" not in text, "фишки остались в тексте"
    assert "ББ" in text


def test_the_replay_is_a_short_paragraph():
    """Построчный формат и был причиной, по которой блок прятали под кнопку."""
    assert len(_lines(hand_replay(_postflop_hand()).plain)) <= 3


def test_the_replay_addresses_the_player_as_you_everywhere():
    """Блок и текст под ним не имеют права говорить с игроком по-разному —
    включая строку вскрытия (`_preflop_shove_hand` её имеет)."""
    for en in (_postflop_hand(), _preflop_shove_hand()):
        text = hand_replay(en).plain
        assert "Hero" not in text
        assert "вы" in text.lower()


def test_a_street_sentence_starts_with_a_capital_even_when_it_is_you():
    text = hand_replay(_postflop_hand()).plain
    assert "банк 6.6. Вы чек" in text and "→ вы фолд" in text


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
    assert all("вы" in span.lower() for span in emphasised)
    flow = next(line for line in replay.plain.splitlines() if "олл-ин" in line)
    assert "→" in flow, "выделенное действие вынесено из потока в отдельную строку"


def test_consecutive_folds_are_merged_into_one_token():
    """Однотипные фолды слипаются: `UTG/HJ фолд` вместо двух отдельных шагов."""
    assert "UTG/HJ фолд" in hand_replay(_preflop_shove_hand()).plain


def test_a_run_out_street_prints_only_its_board():
    """После олл-ина никто не ходит. Печатается борд — и НИКАКИХ «чек-чек»:
    приписать игрокам действия, которых не было, значит выдумать ход руки."""
    text = hand_replay(_preflop_shove_hand()).plain
    assert "чек" not in text.lower()
    assert "Флоп 6♠️ J♦️ Q♦️ · Тёрн 7♥️ · Ривер A♥️." in text


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
    assert "опен 2.5" in text and "3-бет 7.0" in text


# --- числа: только то, что посчитал движок --------------------------------------------


def test_a_printed_amount_is_the_whole_bet_not_the_increment():
    """Спека §5.6: `бет 1.9` — 1.9 ББ от этого игрока на этой улице целиком.
    Берётся `committed_after`, а не разница с предыдущим действием."""
    en = _postflop_hand()
    raise_action = next(a for a in en.hand.actions if a.kind is ActionKind.RAISE)
    assert f"опен {raise_action.committed_after / en.hand.bb:.1f}" in hand_replay(en).plain


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
    allowed |= {round(p.stack / hand.bb, 1) for p in hand.players}
    allowed |= {float(a.committed_after) for a in hand.actions}
    allowed |= {round(a.committed_after / hand.bb, 1) for a in hand.actions}
    allowed |= {float(v) for v in en.report.pot_by_street.values()}
    allowed |= {round(v / hand.bb, 1) for v in en.report.pot_by_street.values()}
    allowed |= {float(sum(post.amount for post in hand.posts))}
    allowed |= {round(sum(post.amount for post in hand.posts) / hand.bb, 1)}
    allowed |= {float(sum(p.amount for p in hand.posts if p.kind is PostKind.ANTE))}
    # Исход раздачи — два числа из разных источников: убыль стека считает движок
    # (`stacks_end`), банк записал рум (`collected`). Оба в руке есть.
    allowed |= {abs(round(hero_stack_delta_bb(en), 1))}
    allowed |= {
        round(sum(c.amount for c in hand.collected if c.label == hand.hero_label) / hand.bb, 1)
    }
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


# --- вскрытие и исход раздачи ----------------------------------------------------------


def test_showdown_line_shows_the_cards_that_were_actually_shown():
    """Вскрытие — одной строкой, картами тех, кто их показал; комбинации не
    называются: назвать их значило бы оценить руку, а реплей ничего не считает."""
    line = next(
        line for line in _lines(hand_replay(_preflop_shove_hand()).plain) if "Вскрытие" in line
    )
    assert "вы J♥️9♥️" in line and "K♠️K♦️" in line


def test_players_who_reached_the_showdown_carry_no_show_mark():
    """Дошедшие до вскрытия стоят в ряд через ` vs ` и пометки показа не несут:
    пометка различает происхождение записи, а у этих двоих оно — вскрытие."""
    line = next(
        line for line in _lines(hand_replay(_preflop_shove_hand()).plain) if "Вскрытие" in line
    )
    assert "вы J♥️9♥️ vs CO K♠️K♦️" in line
    assert "показал" not in line


def test_a_card_shown_after_a_fold_is_marked_as_a_show():
    """Спасовавший, чью карту назвал рум, стоит с пометкой и не попадает в ряд
    вскрывшихся: `vs` между ним и ними утверждало бы, что он с ними мерился."""
    en = _preflop_shove_hand(
        [
            ShowdownEntry(label="Hero", cards=["Jh", "9h"]),
            ShowdownEntry(label="P5", cards=["Ks", "Kd"]),
            ShowdownEntry(label="P6", cards=["As"]),  # BTN спасовал на префлопе
        ]
    )
    line = next(line for line in _lines(hand_replay(en).plain) if "Вскрытие" in line)
    assert "вы J♥️9♥️ vs CO K♠️K♦️" in line
    assert "BTN A♠️ (игрок показал)" in line
    assert "vs BTN" not in line


def test_cards_of_a_folded_player_read_off_a_screenshot_are_not_printed():
    """На скрине запись о спасовавшем родилась из чтения карт героя, а не из
    показа: карт спасовавшего соперника на экране не видно. Строки нет вовсе, а
    карты героя и так стоят в шапке блока."""
    text = hand_replay(
        _river_fold_hand(
            provenance=Provenance.SCREENSHOT,
            showdowns=[ShowdownEntry(label="Hero", cards=["Jh", "Ts"])],
        )
    ).plain
    assert "Вскрытие" not in text and "показал" not in text
    assert text.count("J♥️T♠️") == 1, "карты героя удвоились строкой вскрытия"


def test_a_show_without_a_showdown_is_not_called_a_showdown():
    """Рум назвал карту спасовавшего в раздаче, где до вскрытия не дошёл никто:
    карта печатается, а слово «Вскрытие» — нет, вскрытия не было. Показ героя —
    фраза во втором лице и отдельное предложение, значит с заглавной буквы.
    Последним предложением абзаца она больше не стоит: за ней идёт исход раздачи."""
    text = hand_replay(
        _river_fold_hand(
            provenance=Provenance.HAND_HISTORY,
            showdowns=[ShowdownEntry(label="Hero", cards=["Jh"])],
        )
    ).plain
    assert "вы фолд. Вы показали J♥️." in text
    assert "Вскрытие" not in text


def test_a_hero_show_beside_a_real_showdown_stays_inside_the_line():
    """Герой спасовал и показал карту, а двое соперников вскрылись: их ряд — через
    ` vs `, показ героя — той же фразой во втором лице, но внутри предложения и
    потому со строчной."""
    text = hand_replay(_river_fold_beside_a_showdown_hand()).plain
    line = next(line for line in _lines(text) if "Вскрытие" in line)
    assert "Вскрытие: CO K♠️K♦️ vs BTN 8♣️8♦️; вы показали J♥️." in line


def test_a_hand_without_a_showdown_says_nothing_about_one():
    assert "Вскрытие" not in hand_replay(_postflop_hand()).plain


def test_a_won_hand_ends_with_the_whole_pot():
    """Рука владельца: банк 8 800 при bb 3 000 — 2.9 ББ, а не чистый прирост стека.

    Печатается банк ЦЕЛИКОМ, вместе с собственным вкладом героя (решение
    владельца 2026-09-12): по стекам он прибавил 1.8 ББ, потому что 3 400 фишек
    этого банка положил сам.
    """
    en = _folded_through_shove_hand()
    assert hand_replay(en).plain.rstrip().endswith("Забираете 2.9 ББ.")
    # Асимметрия пришпилена парой, а не одним числом: печатается банк, а прирост
    # стека у этой же руки другой, и докстринг `_outcome_line` ссылается на него.
    assert hero_stack_delta_bb(en) == 1.8


def test_a_lost_hand_ends_with_the_chips_that_left_the_stack():
    """Проигрыш — чистая убыль стека: колл до 250 плюс анте 10 при bb 100."""
    text = hand_replay(_postflop_hand()).plain
    assert text.rstrip().endswith("Отдаёте 2.6 ББ.")


def test_an_untouched_stack_ends_the_paragraph_without_an_outcome():
    """Ноль движения — фразы нет вовсе: «Отдаёте 0.0 ББ» не событие раздачи."""
    text = hand_replay(_untouched_stack_hand()).plain
    assert text.rstrip().endswith("вы фолд → SB фолд."), "абзац кончился не ходом раздачи"
    assert "Забираете" not in text and "Отдаёте" not in text


def test_the_sign_comes_from_the_stack_not_from_the_payout_record():
    """Сайд-пот герою, мейн-пот — олл-ину: запись `collected` есть, а раздача проиграна.

    Знак берётся у движка (стек уменьшился на 110 фишек), и потому блок говорит
    «Отдаёте». По наличию записи выплаты он сказал бы «Забираете 3.8 ББ» на
    раздаче, которую герой проиграл.
    """
    text = hand_replay(_side_pot_loss_hand()).plain
    assert text.rstrip().endswith("Отдаёте 1.1 ББ.")
    assert "Забираете" not in text


def test_a_win_the_source_did_not_record_stays_silent():
    """Стек вырос, а записи о банке нет: сказать нечем, и блок молчит.

    Взять сюда убыль по стеку значило бы напечатать под словом «Забираете»
    величину из другого источника; напечатать ноль — назвать выигранную раздачу
    нулевой. Обе цены хуже молчания (CLAUDE.md: никогда не выдумывать числа
    о деньгах).
    """
    en = _folded_through_shove_hand()
    en.hand.collected.clear()
    text = hand_replay(en).plain
    assert text.rstrip().endswith("BB фолд."), "абзац кончился не ходом раздачи"
    assert "Забираете" not in text and "Отдаёте" not in text


def test_an_unverified_showdown_leaves_the_outcome_unsaid():
    """Вскрытие решено на доукомплектованных картах — исход раздачи не факт.

    Движок добирает непрочитанную карту соперника из остатка колоды, и стек на
    конец руки становится исходом выдуманного вскрытия: «Отдаёте Z ББ» назвало
    бы проигрыш, которого могло не быть. Блок молчит по тому же признаку, по
    которому `worker.pipeline._hand_zone` понижает зону до «предполагая», —
    непустому `Verdict.not_checked`.
    """
    en = _fabricated_showdown_hand()
    assert en.verdict.not_checked, "фикстура перестала быть непроверенным входом"
    assert hero_stack_delta_bb(en) < 0, "без проигрыша по стекам блок промолчал бы и без страховки"
    text = hand_replay(en).plain
    assert text.rstrip().endswith("Вскрытие: вы J♥️9♥️.")
    assert "Забираете" not in text and "Отдаёте" not in text


def test_the_replay_no_longer_prints_the_cost_of_the_decision():
    """Цена решения ушла из блока в разбор точки (решение владельца 2026-09-12):
    она не движение фишек, а расчёт против диапазона, и месту «что было» чужая."""
    assert "Потеря" not in hand_replay(_postflop_hand()).plain


# --- оппонент: метка и частоты ---------------------------------------------------------


def test_an_opponent_carries_its_label_and_both_frequencies_once():
    stats = {"P5": PlayerStats(hands=40, vpip=10, pfr=7)}
    text = hand_replay(_postflop_hand(), stats=stats).plain
    assert "CO (P5, VPIP 25%, PFR 18%) опен 2.5" in text
    assert text.count("P5") == 1, "метка ставится один раз, не у каждого хода"


def test_an_opponent_without_a_sample_carries_no_brackets():
    """Скрин даёт одну руку, знаменателя нет. «VPIP 0%» никто не измерял;
    пустая скобка не печатается вовсе."""
    text = hand_replay(_postflop_hand(), stats={"P5": PlayerStats()}).plain
    assert "VPIP" not in text and "(" not in text


def test_a_measured_zero_is_printed_because_it_was_measured():
    """У VPIP и PFR знаменатель ОБЩИЙ (`PlayerStats.hands`): появляются и
    исчезают вместе. Ноль по сорока раздачам — измеренный."""
    stats = {"P5": PlayerStats(hands=40, vpip=10, pfr=0)}
    assert "VPIP 25%, PFR 0%" in hand_replay(_postflop_hand(), stats=stats).plain


def test_a_frequency_rounds_half_up():
    stats = {"P5": PlayerStats(hands=8, vpip=1, pfr=1)}  # 12.5%
    assert "VPIP 13%, PFR 13%" in hand_replay(_postflop_hand(), stats=stats).plain


def test_the_hero_gets_no_label_though_he_is_in_the_stats():
    """«Герой не в словаре» — не данность, а решение реплея.

    На пути hand history словарь строит `player_stats_by_label`, а он кладёт
    туда ВСЕХ за столом, включая героя, под его меткой. Иначе игрок читал бы
    про себя в третьем лице: `вы (Hero, VPIP 25%, PFR 18%)`.

    Запрет держат два места сразу, и каждого хватает поодиночке: `_street_flow`
    не считает метку герою, а `_action_text` приписывает скобку к позиции, а не
    к «вы». Тест краснеет, только когда сняты оба, — на то он и про печатаемый
    текст, а не про одно из условий. Метка оппонента проверяется здесь же:
    иначе тест был бы зелёным и от того, что статистика вообще не доехала.
    """
    row = PlayerStats(hands=40, vpip=10, pfr=7)
    text = hand_replay(_postflop_hand(), stats={"Hero": row, "P5": row}).plain
    assert "CO (P5, VPIP 25%, PFR 18%) опен 2.5" in text
    assert "Hero" not in text
    assert "вы (" not in text.lower()


def test_a_folding_opponent_gets_no_label():
    """Слипшиеся фолды (`UTG/HJ фолд`) не несут ни метки, ни частот."""
    stats = {"P3": PlayerStats(hands=40, vpip=10, pfr=7)}
    assert "P3" not in hand_replay(_postflop_hand(), stats=stats).plain
