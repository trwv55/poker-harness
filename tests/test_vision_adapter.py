"""Скриншот -> `RawHand`: перевод единиц, рассадка, контрольные суммы, каскад.

Тесты ходят к ДВОЙНИКУ модели, а не к модели: софтверные тесты проверяют код,
качество чтения проверяют evals (EVALS.md, разделение этажей 1 и 2). Живой
прогон по настоящим скринам — `harness.platform.eval_runner vision`.

Эталонное чтение ниже — экспорт истории раздачи из `fixtures/hh/pko-bounty-172.txt`,
переписанный в числа, какими их печатает экран: в больших блайндах, с обрезкой до
двух знаков. Ники заменены на `N1…N8` — настоящие в репозиторий не едут
(docs/publishing-policy.md), а проверяемые величины от них не зависят.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar, cast

import pytest
from pydantic import BaseModel

from harness.contracts import (
    ActionKind,
    Completeness,
    Provenance,
    SeenAction,
    SeenPlayer,
    SeenPost,
    SeenWin,
    Street,
    Unit,
    ValidationStatus,
    VisionReading,
)
from harness.engine import enrich
from harness.normalizer import normalize
from harness.parsers.vision_adapter import (
    HERO_LABEL,
    PromptUnavailable,
    VisionOutcome,
    read_prompt,
    reading_to_raw,
    vision_extract,
)
from harness.parsers.vision_checks import (
    CHECK_BUTTON,
    CHECK_CARDS,
    CHECK_EQUITY,
    CHECK_HERO,
    CHECK_POT,
    match_hero,
)
from tests.conftest import requires_prompts

_T = TypeVar("_T", bound=BaseModel)

HERO_NICK = "N2"
BB_UNIT = 10_000  # столько «фишек» в одном ББ на экране, где фишек нет вовсе


def bb(value: float) -> int:
    return round(value * BB_UNIT)


def _nicknames(reading: VisionReading) -> list[str]:
    return [p.nickname for p in reading.players if p.nickname]


def _player(nick: str, stack: float, **over) -> SeenPlayer:
    base = {"nickname": nick, "stack": stack, "stack_unit": Unit.BB}
    base.update(over)
    return SeenPlayer(**base)


def export_reading(**over) -> VisionReading:
    """Чтение экспорта истории: рука с олл-ином, вскрытием и напечатанным эквити."""
    base = {
        "tournament_name": "Bounty Hunters Deepstack Turbo $5.40",
        "hand_no": "TM6292954855",
        "max_seats": 8,
        "pot_shown": 31.95,
        "pot_unit": Unit.BB,
        "board": ["6h", "9h", "Ah", "Jc", "2d"],
        "players": [
            _player("N3", 17.51, cards_in_log=["Tc", "Th"], equity_shown_pct=57.28),
            _player("N4", 14.89),
            _player("N5", 0.0, cards_in_log=["Ks", "Ad"], equity_shown_pct=42.72),
            _player("N6", 33.09),
            _player("N7", 60.85),
            _player("N8", 2.66, has_button=True),
            _player("N1", 11.58),
            _player(HERO_NICK, 12.52, cards_at_seat=["8d", "3h"]),
        ],
        "blinds_block": [
            SeenPost(label="All Ante", amount=1.20, unit=Unit.BB),
            SeenPost(label="SB", nickname="N1", amount=0.50, unit=Unit.BB),
            SeenPost(label="ББ", amount=1.0, unit=Unit.BB),
        ],
        "ante_pool_shown": 1.20,
        "ante_unit": Unit.BB,
        "actions": [
            SeenAction(
                street=Street.PREFLOP,
                nickname="N3",
                position="UTG",
                kind=ActionKind.RAISE,
                to_amount=32.14,
                is_all_in=True,
            ),
            SeenAction(
                street=Street.PREFLOP, nickname="N4", position="UTG+1", kind=ActionKind.FOLD
            ),
            SeenAction(
                street=Street.PREFLOP,
                nickname="N5",
                position="MP",
                kind=ActionKind.CALL,
                amount=14.62,
                is_all_in=True,
            ),
            SeenAction(
                street=Street.PREFLOP, nickname="N6", position="MP+1", kind=ActionKind.FOLD
            ),
            SeenAction(street=Street.PREFLOP, nickname="N7", position="CO", kind=ActionKind.FOLD),
            SeenAction(street=Street.PREFLOP, nickname="N8", position="BTN", kind=ActionKind.FOLD),
            SeenAction(street=Street.PREFLOP, nickname="N1", position="SB", kind=ActionKind.FOLD),
            SeenAction(street=Street.PREFLOP, kind=ActionKind.FOLD),  # строка героя — без подписи
        ],
        "showdown_seen": True,
        "result_seen": True,
        "winners": [SeenWin(nickname="N5", amount=31.95, unit=Unit.BB)],
    }
    base.update(over)
    return VisionReading.model_validate(base)


def built(reading: VisionReading | None = None, hero: str | None = HERO_NICK):
    return reading_to_raw(
        reading if reading is not None else export_reading(),
        hero_nickname=hero,
        source_ref="screenshot",
    )


# --- перевод единиц и деление ------------------------------------------------


def test_the_ante_pool_is_divided_by_the_code_not_by_the_model():
    """Экран показывает пул, движку нужна ставка одного игрока (реестр B2).

    1.20 ББ на восьмерых — 0.15 ББ каждому. Модель этого деления не делает, и в
    её схеме подушевого поля здесь нет вовсе.
    """
    raw, _ = built()
    assert raw.ante == bb(0.15)
    assert raw.ante_type == "per_player"


def test_everything_lands_in_one_unit_and_the_model_converts_nothing():
    """Единица приложена к каждой группе, приводит к одной — код (реестр A4)."""
    raw, _ = built()
    assert raw.bb == BB_UNIT
    assert raw.sb == bb(0.5)


def test_a_screen_printed_in_chips_keeps_its_own_scale():
    """Живой стол печатает уровень в фишках — тогда фишка и есть единица."""
    reading = export_reading(
        blinds_block=[
            SeenPost(label="All Ante", amount=6_000, unit=Unit.CHIPS),
            SeenPost(label="SB", nickname="N1", amount=2_500, unit=Unit.CHIPS),
            SeenPost(label="ББ", amount=5_000, unit=Unit.CHIPS),
        ],
        ante_pool_shown=6_000,
        ante_unit=Unit.CHIPS,
    )
    raw, _ = reading_to_raw(reading, hero_nickname=HERO_NICK, source_ref="s")
    assert (raw.sb, raw.bb, raw.ante) == (2_500, 5_000, 750)


# --- рассадка ----------------------------------------------------------------


def test_the_seating_is_rebuilt_from_the_order_of_the_preflop_log():
    """Порядок строк лога префлопа И ЕСТЬ порядок хода: первым UTG, последним ББ.

    Круг от малого блайнда получается как «SB, BB, затем все остальные в порядке
    лога», и кнопка оказывается последней. Сверено с текстом рума той же руки:
    место кнопки, малый и большой блайнд совпали.
    """
    raw, _ = built()
    assert [seat.label for seat in raw.seats][:2] == ["S1", HERO_LABEL]
    assert raw.button_seat == 8
    assert raw.seats[-1].label == "S8"


def test_the_dealer_chip_is_checked_against_the_seating_and_not_trusted_alone():
    """Фишка дилера — второе прочтение той же рассадки, и оно сверяется (реестр D3)."""
    _raw, checks = built()
    button = next(c for c in checks if c.name == CHECK_BUTTON)
    assert button.passed

    moved = export_reading(
        players=[
            p.model_copy(update={"has_button": p.nickname == "N7"})
            for p in export_reading().players
        ]
    )
    _raw2, checks2 = reading_to_raw(moved, hero_nickname=HERO_NICK, source_ref="s")
    assert not next(c for c in checks2 if c.name == CHECK_BUTTON).passed


# --- стартовые стеки ---------------------------------------------------------


def test_the_starting_stacks_are_rebuilt_from_the_shown_remainders():
    """Экран показывает остаток ПОСЛЕ ставок и ДО раздачи банка.

    Восстановление: `стартовый = показанный + анте + вложенное − возвращённое`.
    Сверено с текстом рума той же руки — совпадение до сотых ББ, расхождение
    только от обрезки экрана до двух знаков.
    """
    raw, _ = built()
    by_label = {seat.label: seat.stack for seat in raw.seats}
    # Рум: малый блайнд 61 158 фишек при блайнде 5 000 = 12.2316 ББ.
    assert by_label["S1"] == pytest.approx(bb(12.2316), abs=bb(0.02))
    # Рум: герой 68 360 = 13.672 ББ.
    assert by_label[HERO_LABEL] == pytest.approx(bb(13.672), abs=bb(0.02))
    # Рум: олл-ин-рейзер 161 473 = 32.2946 ББ.
    assert by_label["S3"] == pytest.approx(bb(32.2946), abs=bb(0.02))


def test_the_rebuilt_hand_plays_through_the_engine_and_passes_the_validator():
    """Главная проверка восстановления: рука обязана проигрываться и сходиться.

    Обрезка экрана до двух знаков делает суммы чуть меньше настоящих, и без
    поправки на неё движок упирался бы в «поставил больше, чем имел» на каждой
    руке с олл-ином. Поправка ограничена сверху (`_TRUNCATION_LIMIT_BB`).
    """
    en = enrich(normalize(built()[0]))
    assert en.report.illegal_actions == []
    assert en.verdict.status is ValidationStatus.PASS


def test_the_pot_of_the_rebuilt_hand_matches_what_the_screen_shows():
    en = enrich(normalize(built()[0]))
    assert en.report.final_pot == pytest.approx(bb(31.95), abs=bb(0.05))


# --- контрольные суммы -------------------------------------------------------


def test_all_four_checksums_pass_on_a_correctly_read_screen():
    from harness.parsers.vision_adapter import run_checks

    raw, extra = built()
    _hero, hero_check = match_hero(HERO_NICK, _nicknames(export_reading()))
    checks = run_checks(export_reading(), raw, hero_check, extra)
    failed = [c.name for c in checks if not c.passed]
    assert failed == [], [c.detail for c in checks if not c.passed]
    assert {CHECK_POT, CHECK_CARDS, CHECK_EQUITY, CHECK_HERO} <= {c.name for c in checks}


def test_a_missing_ante_pool_is_caught_by_the_pot_checksum():
    """Ровно тот отказ, которым анте и было доказано: банк не сошёлся (реестр D2)."""
    from harness.parsers.vision_adapter import run_checks

    reading = export_reading(ante_pool_shown=None)
    raw, extra = reading_to_raw(reading, hero_nickname=HERO_NICK, source_ref="s")
    _hero, hero_check = match_hero(HERO_NICK, _nicknames(reading))
    pot = next(c for c in run_checks(reading, raw, hero_check, extra) if c.name == CHECK_POT)
    assert not pot.passed
    # Расхождение равно ровно пропущенному пулу анте — так его и доказали.
    shown, summed = (float(value) for value in pot.options)
    assert shown - summed == pytest.approx(1.20, abs=0.02)


def test_a_suit_misread_under_the_win_banner_is_caught_by_the_equity_oracle():
    """Масть под баннером — измеренный класс ошибки, и эквити его ловит.

    Одномастность двигает эквити примерно на две процентных единицы (реестр:
    Q♣J♣ против A♠5♠ — 44.55%, против A♠5♣ — 46.60%), а на границе пуш-фолда
    этого хватает, чтобы поменять вердикт. Здесь прочитана масть короля: A♦K♠ —
    разномастные, A♦K♦ — одномастные, и напечатанный процент перестаёт сходиться.
    """
    from harness.parsers.vision_adapter import run_checks

    misread = export_reading(
        players=[
            _player("N3", 17.51, cards_in_log=["Tc", "Th"], equity_shown_pct=57.28),
            _player("N4", 14.89),
            _player("N5", 0.0, cards_in_log=["Kd", "Ad"], equity_shown_pct=42.72),
            _player("N6", 33.09),
            _player("N7", 60.85),
            _player("N8", 2.66, has_button=True),
            _player("N1", 11.58),
            _player(HERO_NICK, 12.52, cards_at_seat=["8d", "3h"]),
        ]
    )
    raw, extra = reading_to_raw(misread, hero_nickname=HERO_NICK, source_ref="s")
    _hero, hero_check = match_hero(HERO_NICK, _nicknames(misread))
    equity = next(c for c in run_checks(misread, raw, hero_check, extra) if c.name == CHECK_EQUITY)
    assert not equity.passed


def test_the_two_card_renderings_are_compared_and_a_disagreement_is_named():
    from harness.parsers.vision_checks import cards_check

    check = cards_check({"N5": ["Ks", "Ad"]}, {"N5": ["Kh", "Ad"]})
    assert not check.passed
    assert check.options == ["Ks Ad", "Kh Ad"]


# --- герой -------------------------------------------------------------------


def test_the_hero_is_found_by_the_profile_nickname_not_by_the_model():
    hero, check = match_hero("N2", ["N1", "N2", "N3"])
    assert (hero, check.passed) == ("N2", True)


def test_a_truncated_nickname_on_screen_still_matches_the_profile():
    """Экран режет длинные ники многоточием — отсюда сопоставление по префиксу."""
    hero, check = match_hero("FictionalNick46..", ["FictionalNick46..", "someone"])
    assert (hero, check.passed) == ("FictionalNick46..", True)


def test_no_match_and_several_matches_both_escalate_instead_of_guessing():
    """Ноль или больше одного совпадений — вопрос игроку, а не догадка.

    Дыру, которую это закрывает, не ловит ни одна контрольная сумма: банк, кнопка
    и эквити от того, кого назвали героем, не зависят вовсе.
    """
    assert match_hero("someone", ["N1", "N2"]) [0] is None
    assert match_hero("N", ["N1", "N2"])[0] is None


def test_the_hero_flag_read_by_the_model_is_not_the_source_of_truth():
    """Измерено: Sonnet назвал героем победителя на трёх экранах из трёх.

    Флаг модели остаётся наблюдением, но героя ставит совпадение по нику.
    """
    reading = export_reading()
    lying = reading.model_copy(
        update={
            "players": [
                p.model_copy(update={"is_hero": p.nickname == "N5"}) for p in reading.players
            ]
        }
    )
    raw, _ = reading_to_raw(lying, hero_nickname=HERO_NICK, source_ref="s")
    assert raw.vision is not None
    assert raw.vision.nicknames[HERO_LABEL] == HERO_NICK


# --- полнота -----------------------------------------------------------------


def test_a_screen_with_a_result_is_a_whole_hand():
    raw, _ = built()
    assert raw.completeness is Completeness.HAND
    assert raw.provenance is Provenance.SCREENSHOT


def test_a_screen_without_result_or_showdown_is_a_state_at_the_decision_point():
    """Признак полноты выводится из прочитанного, а не из типа экрана (решение C1)."""
    reading = export_reading(showdown_seen=False, result_seen=False, winners=[])
    raw, _ = reading_to_raw(reading, hero_nickname=HERO_NICK, source_ref="s")
    assert raw.completeness is Completeness.STATE


# --- каскад ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Meta:
    model: str


class FakeLLM:
    """Двойник фасада модели: отдаёт заранее заготовленные чтения по очереди.

    Реализует ровно протокол `VisionLLM` — если протокол разойдётся с фасадом,
    это увидит pyright на настоящем вызывающем (`worker.pipeline`), а не тест.
    """

    def __init__(self, *readings: VisionReading) -> None:
        self.readings = list(readings)
        self.purposes: list[str] = []
        self.prompt = ""

    async def __call__(
        self,
        purpose: Literal["vision_extract", "vision_extract_fallback"],
        schema: type[_T],
        *,
        prompt: str,
        images: Sequence[bytes] = (),
        trace_id: int,
    ) -> tuple[_T, _Meta]:
        self.purposes.append(purpose)
        self.prompt = prompt
        return cast("_T", self.readings.pop(0)), _Meta(model=purpose)


@pytest.fixture
def prompt_file(tmp_path: Path) -> Path:
    path = tmp_path / "vision.md"
    path.write_text("прочитай экран", encoding="utf-8")
    return path


async def _extract(llm: FakeLLM, prompt: Path, **over) -> VisionOutcome:
    kwargs = {"gg_nickname": HERO_NICK, "trace_id": 1, "prompt_path": prompt}
    kwargs.update(over)
    return await vision_extract(llm, b"\x89PNG", **kwargs)


async def test_a_clean_read_costs_one_call_and_never_reaches_the_expensive_model(prompt_file):
    llm = FakeLLM(export_reading())
    outcome = await _extract(llm, prompt_file)
    assert llm.purposes == ["vision_extract"]
    assert not outcome.escalate


async def test_a_failed_checksum_sends_the_screen_to_the_expensive_model(prompt_file):
    """Провал любой контрольной суммы — ступень каскада, а не тихая правка."""
    llm = FakeLLM(export_reading(ante_pool_shown=None), export_reading())
    outcome = await _extract(llm, prompt_file)
    assert llm.purposes == ["vision_extract", "vision_extract_fallback"]
    assert not outcome.escalate
    assert [hop.role for hop in outcome.hops] == ["primary", "fallback"]
    assert outcome.hops[0].failed_checks == [CHECK_POT]


async def test_a_failure_on_both_models_escalates_to_the_player(prompt_file):
    llm = FakeLLM(export_reading(ante_pool_shown=None), export_reading(ante_pool_shown=None))
    outcome = await _extract(llm, prompt_file)
    assert outcome.escalate
    assert [c.name for c in outcome.failed] == [CHECK_POT]
    assert outcome.raw is not None


async def test_without_a_second_model_configured_the_cascade_has_one_step(prompt_file):
    """Вторая переменная окружения необязательна: её нет — каскада нет, не отказ."""
    llm = FakeLLM(export_reading(ante_pool_shown=None))
    outcome = await _extract(llm, prompt_file, fallback_available=False)
    assert llm.purposes == ["vision_extract"]
    assert outcome.escalate


async def test_a_screen_that_is_not_a_hand_is_refused_without_paying_twice(prompt_file):
    """Честный отказ — это ответ, а не сбой: повторять его на дорогой модели незачем."""
    llm = FakeLLM(VisionReading(not_a_hand=True, refusal_reason="это лобби турнира"))
    outcome = await _extract(llm, prompt_file)
    assert llm.purposes == ["vision_extract"]
    assert outcome.raw is None
    assert outcome.refusal == "это лобби турнира"


async def test_every_hop_of_the_cascade_is_kept_with_the_hand(prompt_file):
    """Ступени едут в `hands.raw`: после эскалации видно, где расхождение возникло."""
    llm = FakeLLM(export_reading(ante_pool_shown=None), export_reading())
    outcome = await _extract(llm, prompt_file)
    assert outcome.raw is not None and outcome.raw.vision is not None
    assert [hop.model for hop in outcome.raw.vision.hops] == [
        "vision_extract",
        "vision_extract_fallback",
    ]


async def test_the_profile_nickname_never_reaches_the_prompt(prompt_file):
    """Подсказка уничтожила бы улику: подсказанный ответ не отличить от прочитанного."""
    llm = FakeLLM(export_reading())
    await _extract(llm, prompt_file)
    assert HERO_NICK not in llm.prompt


def test_a_missing_prompt_file_fails_loudly():
    """Промпт закрыт политикой публикации: в публичном клоне его нет.

    Пустой промпт вместо файла дал бы модели пустое задание, а контрольные суммы
    на пустом чтении молчат — тихая деградация вместо отказа.
    """
    with pytest.raises(PromptUnavailable):
        read_prompt(Path("/nonexistent/vision.md"))


@requires_prompts
def test_the_real_prompt_never_asks_the_model_to_compute_or_to_name_the_hero():
    """Промпт обязан запрещать ровно то, что делает код (реестры A1, B2, B3, A3).

    Читается настоящий файл: формулировки — половина качества чтения, и проверка
    «в промпте есть запрет считать» ловит правку, которая этот запрет потеряет.
    """
    text = read_prompt().casefold()
    assert "не дели" in text or "делить" in text
    assert "герой" in text
    assert "unsure_fields" in text


async def test_an_empty_reading_is_retried_and_not_called_a_refusal(prompt_file):
    """Пустая схема — сбой, а не ответ: повторяем, а не объявляем «это не раздача».

    Все поля `VisionReading` необязательны намеренно, поэтому пустой объект
    валиден и от честного отказа отличается только отсутствием причины.
    Измерено на живом прогоне датасета: одна и та же картинка на одном промпте
    отдаётся то полным чтением, то пустой схемой.
    """
    llm = FakeLLM(VisionReading(), export_reading())
    outcome = await _extract(llm, prompt_file)
    assert llm.purposes == ["vision_extract", "vision_extract"]
    assert outcome.raw is not None
    assert outcome.hops[0].error == "пустое чтение"


async def test_empty_readings_everywhere_fail_loudly_instead_of_inventing_a_refusal(
    prompt_file,
):
    """Пусто на всех ступенях — отказ станции, а не «это не раздача» игроку."""
    from harness.parsers.vision_adapter import VisionReadFailed

    llm = FakeLLM(*[VisionReading() for _ in range(4)])
    with pytest.raises(VisionReadFailed):
        await _extract(llm, prompt_file)
    assert llm.purposes == [
        "vision_extract",
        "vision_extract",
        "vision_extract_fallback",
        "vision_extract_fallback",
    ]
