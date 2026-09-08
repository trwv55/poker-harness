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
    VisionCheck,
    VisionMeta,
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
    EQUITY_TOLERANCE_PP,
    POT_TOLERANCE_BB,
    equity_check,
    match_hero,
    pot_check,
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


def test_a_pot_off_by_exactly_the_tolerance_still_passes():
    """Живой отказ: банк 23.00 при сумме вкладов 22.90 — расхождение ровно в допуск.

    Разность этих двух чисел в двоичной плавающей точке больше 0.1, поэтому
    `delta <= POT_TOLERANCE_BB` отвергал случай, который допуск обязан
    пропускать, — и печатал при этом «расхождение 0.10 ББ» при допуске 0.1.
    """
    assert 23.00 - 22.90 > POT_TOLERANCE_BB  # причина отказа, а не описка в числах
    check = pot_check(23.00, 22.90)
    assert check.passed, check.detail
    assert "расхождение 0.10 ББ" in check.detail


def test_a_pot_off_by_more_than_the_tolerance_still_fails():
    """Допуск остался допуском: пропущенное анте (величина блайнда) не проходит."""
    check = pot_check(24.10, 22.90)
    assert not check.passed


def test_an_equity_off_by_exactly_the_tolerance_still_passes(monkeypatch: pytest.MonkeyPatch):
    """То же на сверке эквити: 64.51% против 63.51% — ровно допуск в 1 п.п.

    Оракул подменён константой: Монте-Карло возвращает число, чьё десятичное
    представление заранее не известно, а проверяется здесь граница сравнения, а
    не расчёт эквити (его проверяют якорные тесты `test_equity.py`).
    """
    monkeypatch.setattr(
        "harness.analysis.tools.equity.equity_hand_vs_hand",
        lambda hero, villain, board: 0.6351,
    )
    assert 64.51 - 63.51 > EQUITY_TOLERANCE_PP  # причина отказа, а не описка в числах
    check = equity_check(64.51, ["Ad", "Ks"], ["Tc", "Th"], ["2c", "7d", "9s"])
    assert check.passed, check.detail
    assert "расхождение 1.00 п.п." in check.detail


def test_an_equity_off_by_more_than_the_tolerance_still_fails(monkeypatch: pytest.MonkeyPatch):
    """Допуск остался допуском: расхождение масти (порядка двух п.п.) не проходит."""
    monkeypatch.setattr(
        "harness.analysis.tools.equity.equity_hand_vs_hand",
        lambda hero, villain, board: 0.6351,
    )
    check = equity_check(65.56, ["Ad", "Ks"], ["Tc", "Th"], ["2c", "7d", "9s"])
    assert not check.passed


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
    hero, check = match_hero("длинный_ник_игрока", ["длинный_ник_иг..", "кто-то"])
    assert (hero, check.passed) == ("длинный_ник_иг..", True)


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


def test_a_showdown_with_an_unread_card_is_named_as_completed_by_the_engine():
    """Движок доукомплектовывает неизвестные карты — на скрине это опасно.

    В hand history безопасно (измерено: 0 из 111 оспариваемых шоудаунов решались
    на фабрикованных картах — рум показывает карты всех дошедших), на скрине
    карта соперника бывает не прочитана, и тогда банк может «выиграть» рука,
    которой не было. Сверка получателей не спасает: строк `collected` у скрина
    нет. Значит, такой шоудаун обязан быть назван.
    """
    from harness.engine.validation import _FABRICATED_SHOWDOWN

    hidden = export_reading(
        players=[
            p.model_copy(update={"cards_in_log": [], "cards_at_seat": []})
            if p.nickname == "N5"
            else p
            for p in export_reading().players
        ]
    )
    raw, _ = reading_to_raw(hidden, hero_nickname=HERO_NICK, source_ref="s")
    assert _FABRICATED_SHOWDOWN in enrich(normalize(raw)).verdict.not_checked
    # У полностью прочитанного вскрытия пометки нет — иначе она ничего не значит.
    assert _FABRICATED_SHOWDOWN not in enrich(normalize(built()[0])).verdict.not_checked


def test_the_win_banner_is_kept_as_a_fifth_independent_reading():
    """«Победа N» читается и доезжает до руки — иначе проверка выплат слепа.

    Раньше строка отбрасывалась, и валидатор проходил на любой руке. Отбросить
    прочитанное, чтобы проверка не срабатывала, — та же подгонка, что подправить
    число; разница только в том, что она молчаливая (ревью раунда 1, B).
    """
    raw, _ = built()
    assert [(c.label, c.amount) for c in raw.collected] == [("S5", bb(31.95))]


def test_the_payout_check_tolerates_the_truncation_the_screen_forces():
    """31.95 на экране против 31.94 восстановленных — обрезка, а не расхождение."""
    en = enrich(normalize(built()[0]))
    assert en.verdict.status is ValidationStatus.PASS


def test_the_payout_check_catches_a_showdown_decided_on_completed_cards():
    """Пятое независимое чтение ловит фабрикацию, а не только называет её.

    Карты одного из вскрывшихся не прочитаны: движок доукомплектует их из
    колоды и может отдать банк не тому, кого назвал экран. Расхождение с
    прочитанной строкой «Победа» — это и есть улика.
    """
    hidden = export_reading(
        players=[
            p.model_copy(update={"cards_in_log": [], "cards_at_seat": []})
            if p.nickname == "N5"
            else p
            for p in export_reading().players
        ]
    )
    raw, _ = reading_to_raw(hidden, hero_nickname=HERO_NICK, source_ref="s")
    verdict = enrich(normalize(raw)).verdict
    assert verdict.status is ValidationStatus.ESCALATE
    assert any("payout mismatch" in reason for reason in verdict.reasons)


def test_a_hand_history_payout_is_compared_without_any_tolerance():
    """У рума числа точные, и допуск там был бы дырой, а не поправкой."""
    from harness.engine.validation import _payout_mismatch
    from harness.parsers.hh_parser import parse_hand
    from tests.test_hh_parser import SAMPLE

    en = enrich(normalize(parse_hand(SAMPLE, source_ref="x")))
    assert _payout_mismatch(en.hand, en.report) is None
    off_by_one = en.report.model_copy(
        update={"stacks_end": {**en.report.stacks_end, "Hero": en.report.stacks_end["Hero"] + 1}}
    )
    assert _payout_mismatch(en.hand, off_by_one) is not None


def _with_phantom_actor() -> VisionReading:
    """Чтение, где служебный пузырёк принят за девятого участника.

    Измеренный на живом прогоне класс ошибки: между строками действий стоит
    банк времени («10s» с цифрой в кружке), модель приняла его за игрока, и вся
    рассадка уехала на одно место.
    """
    reading = export_reading()
    return export_reading(
        players=[*reading.players, reading.players[0].model_copy(update={"nickname": "N9"})],
        actions=[
            reading.actions[0],
            reading.actions[0].model_copy(update={"nickname": "N9", "position": "UTG+1"}),
            *reading.actions[1:],
        ],
    )


def _named_checks(reading: VisionReading) -> dict[str, bool]:
    from harness.parsers.vision_adapter import run_checks

    raw, extra = reading_to_raw(reading, hero_nickname=HERO_NICK, source_ref="s")
    _hero, hero_check = match_hero(HERO_NICK, _nicknames(reading))
    return {c.name: c.passed for c in run_checks(reading, raw, hero_check, extra)}


def test_a_phantom_actor_is_caught_by_the_table_size_and_by_the_printed_positions():
    """Два дешёвых чека ловят лишнего участника раньше всех денежных сверок.

    Ни банк, ни кнопка, ни эквити его не замечают: все три считаются по одному и
    тому же неверному чтению — слепой угол D4 реестра. Метки позиций и размер
    стола — независимые от него сигналы (ревью раунда 1, E).
    """
    from harness.parsers.vision_checks import CHECK_POSITIONS, CHECK_SEATS

    checks = _named_checks(_with_phantom_actor())
    assert checks[CHECK_SEATS] is False
    assert checks[CHECK_POSITIONS] is False


def test_both_new_checks_stay_quiet_on_a_correctly_read_screen():
    """Проверка, срабатывающая на верном чтении, — не проверка, а шум."""
    from harness.parsers.vision_checks import CHECK_POSITIONS, CHECK_SEATS

    checks = _named_checks(export_reading())
    assert checks[CHECK_SEATS] is True
    assert checks[CHECK_POSITIONS] is True


def test_the_printed_positions_are_compared_and_not_merely_stored():
    """Обещание контракта «код сверяет напечатанную метку» обязано быть правдой.

    До ревью раунда 1 метка попадала только в `raw_line` и не сверялась ни с чем.
    """
    from harness.parsers.vision_checks import CHECK_POSITIONS

    shifted = export_reading(
        actions=[
            a.model_copy(update={"position": "CO"}) if a.position == "UTG" else a
            for a in export_reading().actions
        ]
    )
    assert _named_checks(shifted)[CHECK_POSITIONS] is False


def test_the_table_size_check_is_silent_when_the_header_was_not_read():
    """Выдумывать расхождение из отсутствия данных нельзя — тот же принцип везде."""
    from harness.parsers.vision_checks import CHECK_SEATS

    assert _named_checks(export_reading(max_seats=None))[CHECK_SEATS] is True


def test_confirming_the_shown_pot_does_not_close_a_dispute_it_does_not_settle():
    """На полной руке `displayed_pot` не читает никто — ответ обязан СХОДИТЬСЯ.

    Иначе выходило так: модель пропустила пул анте, банк на экране разошёлся с
    суммой вкладов, игрок подтвердил показанное число — и проверка «закрывалась»,
    ничего в руке не изменив. Разбор уезжал игроку по руке с анте, равным нулю
    (ревью раунда 2, F1).
    """
    from harness.parsers.vision_adapter import apply_vision_answer, contributions_bb

    reading = export_reading(ante_pool_shown=None, winners=[])
    raw, _ = reading_to_raw(reading, hero_nickname=HERO_NICK, source_ref="s")
    raw.vision = (raw.vision or VisionMeta()).model_copy(
        update={"checks": [VisionCheck(name=CHECK_POT, passed=False, options=["31.95", "30.74"])]}
    )
    assert raw.completeness is Completeness.HAND

    confirmed = apply_vision_answer(raw, "pot", "31.95")
    assert confirmed is not None and confirmed.vision is not None
    assert [c.passed for c in confirmed.vision.checks] == [False]  # спор не закрыт
    assert confirmed.vision.displayed_pot == bb(31.95)  # но ответ игрока сохранён

    agreeing = apply_vision_answer(raw, "pot", f"{contributions_bb(raw):.2f}")
    assert agreeing is not None and agreeing.vision is not None
    assert [c.passed for c in agreeing.vision.checks] == [True]


def test_on_a_state_the_shown_pot_is_the_answer_and_closes_the_dispute():
    """На состоянии банк берётся именно из показанного — там ответ и есть данные."""
    from harness.parsers.vision_adapter import apply_vision_answer

    reading = export_reading(showdown_seen=False, result_seen=False, winners=[])
    raw, _ = reading_to_raw(reading, hero_nickname=HERO_NICK, source_ref="s")
    raw.vision = (raw.vision or VisionMeta()).model_copy(
        update={"checks": [VisionCheck(name=CHECK_POT, passed=False, options=["31.95", "30.74"])]}
    )
    assert raw.completeness is Completeness.STATE

    answered = apply_vision_answer(raw, "pot", "31.95")
    assert answered is not None and answered.vision is not None
    assert [c.passed for c in answered.vision.checks] == [True]


def test_a_position_label_that_cannot_occur_in_this_ring_is_reported_not_failed():
    """Метка, которой в круге этого стола не бывает, ничего не доказывает (F3).

    Словарь позиций у рума и у нормалайзера совпадает не весь, и набор меток
    зависит от числа мест. Роняя сверку на неопознанной метке, мы измеряли бы
    полноту своей таблицы соответствий, а не чтение.
    """
    from harness.parsers.vision_checks import positions_check

    check = positions_check({"N1": "MP+2"}, {"N1": "LJ", "N2": "HJ"})
    assert check.passed
    assert "не сопоставимо" in check.detail

    both = positions_check({"N1": "MP+2", "N2": "LJ"}, {"N1": "LJ", "N2": "HJ"})
    assert not both.passed  # сопоставимая метка всё-таки сравнивается
    assert "не сопоставимо" in both.detail


def test_the_room_middle_position_labels_are_only_aliased_where_measured():
    """`MP`/`MP+1` наблюдались на 8-max; на других размерах соответствие не измерено.

    На шестимаксе у нормалайзера нет `LJ` вовсе, и безусловный алиас ронял бы
    сверку на каждом таком экспорте.
    """
    from harness.parsers.vision_adapter import _aliased_position

    assert _aliased_position("MP", 8) == "LJ"
    assert _aliased_position("MP", 6) == "MP"  # не опознано — и не сравнивается
    for seats in (6, 8):
        assert _aliased_position("ББ", seats) == "BB"
        assert _aliased_position("МБ", seats) == "SB"
