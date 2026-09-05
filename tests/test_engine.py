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
from tests.conftest import FIXTURE_DAILY, requires_fixtures
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


def _short_shove_against_deep_hand() -> CanonicalHand:
    """Короткий стек шовит, у Hero глубокий стек, и позади него ещё живой глубокий игрок.

    Раздача построена ровно под дефект: живой оппонент, которого решение не
    касается, здесь ГЛУБЖЕ шовера. Старая формула (максимум ОСТАТКОВ стеков) в
    ней обязана взять именно его — у шовера остаток нулевой, и максимум его
    никогда не выберет.
    """
    raw = RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="x",
        hand_no="S1",
        tournament_id="1",
        tournament_name="T",
        level=1,
        sb=1000,
        bb=2000,
        ante=200,
        timestamp=datetime(2026, 8, 21, 3, 4, 5),  # noqa: DTZ001
        table_name="1",
        max_seats=3,
        button_seat=1,
        seats=[
            SeatInfo(seat=1, label="shover", stack=6200),  # 3.1bb, из них 0.1bb анте
            SeatInfo(seat=2, label="Hero", stack=80_000),  # 40bb
            SeatInfo(seat=3, label="deep", stack=80_000),  # 40bb, ходит ПОСЛЕ Hero
        ],
        posts=[
            Post(label="shover", kind=PostKind.ANTE, amount=200),
            Post(label="Hero", kind=PostKind.ANTE, amount=200),
            Post(label="deep", kind=PostKind.ANTE, amount=200),
            Post(label="Hero", kind=PostKind.SMALL_BLIND, amount=1000),
            Post(label="deep", kind=PostKind.BIG_BLIND, amount=2000),
        ],
        actions=[
            RawAction(
                street=Street.PREFLOP,
                label="shover",
                kind=ActionKind.RAISE,
                amount=6000,
                to_amount=6000,
                is_all_in=True,
                raw_line="shover: raises 6,000 to 6,000 and is all-in",
            ),
            RawAction(
                street=Street.PREFLOP, label="Hero", kind=ActionKind.FOLD, raw_line="Hero: folds"
            ),
            RawAction(
                street=Street.PREFLOP, label="deep", kind=ActionKind.FOLD, raw_line="deep: folds"
            ),
        ],
        dealt={"Hero": ["Ks", "Qd"]},
        uncalled=[Uncalled(label="shover", amount=4000)],
        collected=[Collected(label="shover", amount=5600)],
        summary=SummaryInfo(total_pot=5600, rake=0, jackpot=0, bingo=0, fortune=0, tax=0),
    )
    return normalize(raw)


def test_eff_stack_is_measured_against_the_shover():
    """Глубина решения — против того, чью ставку Hero отвечает, а не против стола.

    Шов на 3bb здесь и есть весь спот: больше 3bb в этом решении не разыграть.
    Прежняя формула брала максимум ОСТАТКОВ живых оппонентов, а у шовера остаток
    нулевой — она выбирала глубокого игрока позади и объявляла тот же спот
    39-битовым, то есть «глубже пуш-фолд-зоны». На реальных данных так вылетели
    из разбора 23 точки, где Hero отвечал на олл-ин.

    Числа теста подобраны так, что подмена оппонента видна в самом значении:
    3.0bb против 38.9bb — не округление, а другой игрок.
    """
    en = enrich(_short_shove_against_deep_hand())
    assert en.verdict.status == "pass"
    assert en.report.illegal_actions == []
    dp = next(d for d in en.report.decision_points if d.label == "Hero")
    assert dp.to_call == 5000  # шов 6000 против 1000 малого блайнда Hero

    # Стек шовера за вычетом анте: 6200 - 200. Анте — мёртвые деньги, уже в
    # банке, и глубиной игры они не являются (та же величина, которой
    # индексируется равновесие: `classifier.SeatSnapshot.stack_after_ante`).
    assert dp.eff_stack == 6000
    assert dp.eff_stack_bb == 3.0

    # Контрольная величина: то, что дала бы старая формула. Живой глубокий
    # игрок позади здесь есть, и он в 13 раз глубже шовера — если тест
    # когда-нибудь снова начнёт мерить против него, он покажет 38.9bb.
    deep_behind = 80_000 - 200 - 2000
    assert deep_behind / 2000 > 15.0
    assert dp.eff_stack != deep_behind


@requires_fixtures
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


def test_validation_rejects_misdirected_pot():
    """Банк правильного размера, уехавший не тому игроку.

    Ни одна из остальных проверок такого не ловит: реплей легален, `final_pot`
    сходится с `Total pot`, карты не дублируются, фишки сохраняются. Ловит только
    сверка стеков с выплатами, записанными самим румом.
    """
    h = normalize(parse_hand(SAMPLE, source_ref="x"))
    bad = h.model_copy(deep=True)
    for entry in bad.collected:
        entry.label = "Hero"  # банк записан проигравшему
    en = enrich(bad)
    assert en.report.illegal_actions == []  # реплей прошёл без единой претензии
    assert en.report.final_pot == 20391  # и банк ровно тот же
    assert en.verdict.status == "reject"
    assert any("payout" in r for r in en.verdict.reasons)


def test_voluntary_all_in_fold_is_not_forfeited():
    """Лишний `folds` после ДОБРОВОЛЬНОГО олл-ина форфейтом не считается.

    Такую строку даёт сбой распознавания или баг парсера. Если принять её за
    форфейт, из руки вылетит живой игрок, а его банк молча уедет сопернику —
    ровно та подмена денег, ради которой правило и держится узким.
    """
    raw = RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="x",
        hand_no="P1",
        tournament_id="1",
        tournament_name="T",
        level=1,
        sb=1000,
        bb=2000,
        ante=0,
        timestamp=datetime(2026, 8, 20, 1, 2, 3),  # noqa: DTZ001
        table_name="1",
        max_seats=3,
        button_seat=3,
        seats=[
            SeatInfo(seat=1, label="A", stack=20_000),
            SeatInfo(seat=2, label="Hero", stack=40_000),
            SeatInfo(seat=3, label="C", stack=40_000),
        ],
        posts=[
            Post(label="A", kind=PostKind.SMALL_BLIND, amount=1000),
            Post(label="Hero", kind=PostKind.BIG_BLIND, amount=2000),
        ],
        actions=[
            RawAction(street=Street.PREFLOP, label="C", kind=ActionKind.FOLD, raw_line="C: folds"),
            RawAction(
                street=Street.PREFLOP,
                label="A",
                kind=ActionKind.RAISE,
                amount=19_000,
                to_amount=20_000,
                is_all_in=True,  # олл-ин добровольный, а не с блайнда
                raw_line="A: raises 19,000 to 20,000 and is all-in",
            ),
            RawAction(
                street=Street.PREFLOP,
                label="Hero",
                kind=ActionKind.CALL,
                amount=18_000,
                raw_line="Hero: calls 18,000",
            ),
            RawAction(street=Street.PREFLOP, label="A", kind=ActionKind.FOLD, raw_line="A: folds"),
        ],
        boards={Street.FLOP: ["2c", "7d", "9h"], Street.TURN: ["Jc"], Street.RIVER: ["4s"]},
        dealt={"A": ["Ac", "Ad"], "Hero": ["Kc", "Kd"]},
        collected=[Collected(label="A", amount=40_000)],
        summary=SummaryInfo(total_pot=40_000, rake=0, jackpot=0, bingo=0, fortune=0, tax=0),
    )
    en = enrich(normalize(raw))
    assert en.report.forfeits == []  # живого игрока из руки не вывели
    assert en.verdict.status == "reject"  # лишняя строка названа, а не проглочена
    assert any("A: folds" in r for r in en.verdict.reasons)


def _sample_with_moved_button() -> CanonicalHand:
    """SAMPLE с кнопкой на соседнем месте.

    Кнопка двигается в `RawHand` и рука нормализуется заново — так тест гоняет
    настоящий вывод позиций из кнопки, а не копию раскладки внутри теста.
    """
    raw = parse_hand(SAMPLE, source_ref="x")
    seats = sorted(s.seat for s in raw.seats)
    raw.button_seat = seats[(seats.index(raw.button_seat) + 1) % len(seats)]
    return normalize(raw)


def test_wrong_button_is_named_by_blind_check():
    """Чужая кнопка называется причиной, а не выводится из денежного каскада.

    Косвенно её ловит и движок — сломанный порядок хода даёт `illegal`, а следом
    расходятся банк, выплаты и сумма фишек. Но ни одно из этих сообщений не
    говорит, что сломана рассадка. Проверка блайндов говорит.
    """
    en = enrich(_sample_with_moved_button())
    assert en.verdict.status == "reject"
    named = [r for r in en.verdict.reasons if r.startswith("blind mismatch")]
    assert len(named) == 1, en.verdict.reasons
    assert "small_blind" in named[0] and "big_blind" in named[0]


def test_wrong_button_on_screenshot_asks_the_player():
    """Скрин — гипотеза: та же ошибка едет вопросом игроку, а не отказом.

    На скрине эта проверка единственная: истории улиц нет, проиграть руку и
    упереться в порядок хода нельзя.
    """
    hand = _sample_with_moved_button()
    hand.provenance = Provenance.SCREENSHOT
    verdict = enrich(hand).verdict
    assert verdict.status == "escalate"
    assert "button" in verdict.fields
    question = verdict.questions[verdict.fields.index("button")]
    assert "дилера" in question


def test_blind_check_is_silent_without_posts():
    """Нет записанных блайндов — нет и расхождения: выдумывать его нельзя.

    Та же оговорка, что у сверки выплат. Обратная сторона: чтобы проверке было
    что сверять, извлечение со скрина обязано отдавать блайнды по игрокам,
    а не только уровень из шапки.
    """
    hand = _sample_with_moved_button()
    hand.posts = []
    reasons = enrich(hand).verdict.reasons
    assert not any(r.startswith("blind mismatch") for r in reasons)
    assert reasons, "остальные проверки обязаны продолжать ловить сдвинутую кнопку"


@requires_fixtures
def test_blind_check_does_not_fire_on_real_hands():
    """Ноль ложных срабатываний на настоящих руках — иначе проверка вредна."""
    from harness.parsers.hh_parser import parse_file as _pf
    hands = _pf(FIXTURE_DAILY.read_text(encoding="utf-8"), source_ref=FIXTURE_DAILY.name)
    for raw in hands:
        verdict = enrich(normalize(raw)).verdict
        assert not any(r.startswith("blind mismatch") for r in verdict.reasons), raw.hand_no
