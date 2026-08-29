from datetime import datetime

from harness.contracts import (
    ActionKind,
    CanonicalHand,
    Collected,
    Post,
    PostKind,
    Provenance,
    RawAction,
    RawHand,
    SeatInfo,
    Street,
    SummaryInfo,
    Uncalled,
)
from harness.engine import enrich
from harness.normalizer import normalize
from harness.parsers.hh_parser import parse_file, parse_hand
from tests.conftest import FIXTURE_DAILY
from tests.test_hh_parser import SAMPLE


def test_sample_hand_pot_and_stacks():
    en = enrich(normalize(parse_hand(SAMPLE, source_ref="x")))
    assert en.verdict.status == "pass"
    assert en.report.final_pot == 20391                      # сходится с SUMMARY
    # виллан: анте 750 + нетто-вклад 6000 (рейз до 69000, 63000 возвращено), забрал банк
    assert en.report.stacks_end["5553a2cd"] == 70471 - 6750 + 20391  # = 84112
    assert en.report.stacks_end["Hero"] == 0                 # проиграл олин
    # сохранение фишек
    start = 85440 + 25109 + 222896 + 3891 + 415055 + 70471 + 151005
    assert sum(en.report.stacks_end.values()) == start


def test_decision_point_for_hero():
    en = enrich(normalize(parse_hand(SAMPLE, source_ref="x")))
    dp = next(d for d in en.report.decision_points if d.label == "Hero")
    assert dp.street == "preflop"
    assert dp.to_call == 141                                  # доплата при стеке 141 за анте+SB
    assert dp.action.kind == "call" and dp.action.is_all_in
    assert (dp.live_total, dp.live_behind) == (3, 1)          # рейзер, Hero, BB; после Hero — BB


def test_validation_rejects_chip_mismatch():
    h = normalize(parse_hand(SAMPLE, source_ref="x"))
    bad = h.model_copy(deep=True)
    assert bad.summary is not None                           # сужение для pyright
    bad.summary.total_pot = 99999                            # банк не сходится
    en = enrich(bad)
    assert en.verdict.status == "reject"                     # HH = факт -> баг парсера, не эскалация
    assert any("pot" in r for r in en.verdict.reasons)


def test_validation_rejects_duplicate_cards():
    h = normalize(parse_hand(SAMPLE, source_ref="x"))
    bad = h.model_copy(deep=True)
    bad.dealt["Hero"] = ["Js", "Ah"]                         # дубль карт вскрытия оппонента
    en = enrich(bad)
    assert en.verdict.status == "reject"


def test_validation_escalates_for_screenshot():
    h = normalize(parse_hand(SAMPLE, source_ref="x"))
    scr = h.model_copy(deep=True)
    assert scr.summary is not None                           # сужение для pyright
    scr.provenance = Provenance.SCREENSHOT                   # энам вместо литерала — тоже для pyright
    scr.summary.total_pot = 99999
    en = enrich(scr)
    assert en.verdict.status == "escalate"                   # скрин = гипотеза -> спросить игрока
    assert en.verdict.fields                                  # какие поля переспрашивать


def _heads_up_hand() -> CanonicalHand:
    """Синтетика: хедз-ап, кнопка ставит малый блайнд и пасует.

    В фикстурах хедз-апа нет, а порядок мест для PokerKit в игре на двоих
    обратный порядку позиций нормалайзера — без этого теста ошибка в нём
    осталась бы незамеченной.
    """
    raw = RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="x",
        hand_no="T1",
        tournament_id="1",
        tournament_name="T",
        level=1,
        sb=3000,
        bb=6000,
        ante=0,
        timestamp=datetime(2026, 8, 20, 22, 22, 36),  # noqa: DTZ001
        table_name="1",
        max_seats=2,
        button_seat=1,
        seats=[
            SeatInfo(seat=1, label="Hero", stack=100_000),
            SeatInfo(seat=2, label="villain", stack=100_000),
        ],
        posts=[
            Post(label="Hero", kind=PostKind.SMALL_BLIND, amount=3000),
            Post(label="villain", kind=PostKind.BIG_BLIND, amount=6000),
        ],
        actions=[
            RawAction(street=Street.PREFLOP, label="Hero", kind=ActionKind.FOLD, raw_line="Hero: folds")
        ],
        uncalled=[Uncalled(label="villain", amount=3000)],
        collected=[Collected(label="villain", amount=6000)],
        summary=SummaryInfo(total_pot=6000, rake=0, jackpot=0, bingo=0, fortune=0, tax=0),
    )
    return normalize(raw)


def test_heads_up_button_posts_small_blind():
    en = enrich(_heads_up_hand())
    assert en.verdict.status == "pass"
    assert en.report.final_pot == 6000  # SB 3000 + столько же от BB, 3000 возвращено
    assert en.report.stacks_end == {"Hero": 97_000, "villain": 103_000}
    dp = next(d for d in en.report.decision_points if d.label == "Hero")
    assert dp.position == "BTN" and dp.to_call == 3000
    assert (dp.live_total, dp.live_behind) == (2, 1)


def test_decision_point_pot_is_only_what_hero_can_win():
    """`pot_before` — банк, за который герой играет, а не всё, что лежит на столе.

    Виллан поставил 69 000 при стеке героя 3891: 63 000 из них ему вернутся, и
    выиграть их герой не может — складывать их в банк значило бы завысить шансы
    банка. `pot_before + to_call` даёт 14 673, ровно тот мейн-пот, который рум
    записал строкой `5553a2cd collected 14,673`.
    """
    en = enrich(normalize(parse_hand(SAMPLE, source_ref="x")))
    dp = next(d for d in en.report.decision_points if d.label == "Hero")
    assert dp.pot_before == 14532
    assert dp.pot_before + dp.to_call == 14673


def test_forfeit_of_blind_all_in_is_honoured():
    """Реальная рука: рум пишет `folds` игроку, ушедшему в олл-ин блайндом.

    `e0fb14f7` со стеком 236 платит анте 100 и малый блайнд 136 — фишек за спиной
    не остаётся. По правилам NLHE он остаётся в руке и претендует на мейн-пот, но
    рум считает его вышедшим, а его 136 — мёртвыми деньгами, и отдаёт весь банк
    4136 победителю. Провенанс `hand_history` — это факт, движок его исполняет и
    отмечает поправку в `forfeits`.
    """
    raw = next(
        r
        for r in parse_file(FIXTURE_DAILY.read_text(encoding="utf-8"), source_ref="daily")
        if r.hand_no == "TM6315415787"
    )
    en = enrich(normalize(raw))
    assert en.verdict.status == "pass"
    assert en.report.forfeits == ["e0fb14f7"]  # поправка видна в трассе
    assert en.report.illegal_actions == []
    assert en.report.final_pot == 4136  # Total pot рума, вместе с мёртвыми 136
    assert en.report.stacks_end["e0fb14f7"] == 0  # назад не вернулось ничего
    assert en.report.stacks_end["ce115272"] == 38_247  # победитель забрал весь банк
