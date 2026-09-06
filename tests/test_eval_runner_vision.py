"""Сверка прочитанного с текстом рума — механика раннера, без сети и без модели.

Проверяется КОД сравнения, а не качество чтения: качество меряет живой прогон
(`uv run python -m harness.platform.eval_runner vision`), и он стоит денег.
Разделение то же, что во всём проекте: софтверные тесты проверяют код, evals —
модель (EVALS.md).
"""

from __future__ import annotations

from harness.contracts import Completeness
from harness.normalizer import normalize
from harness.parsers.vision_adapter import reading_to_raw
from harness.platform.eval_runner import compare_reading
from tests.test_vision_adapter import HERO_NICK, export_reading


def _read(**over):
    raw, _checks = reading_to_raw(
        export_reading(**over), hero_nickname=HERO_NICK, source_ref="s"
    )
    return normalize(raw)


def _fields(read, truth) -> dict[str, bool]:
    return {r.field: r.matched for r in compare_reading(read, truth)}


def test_a_reading_compared_with_itself_matches_on_every_field():
    """Опора всей сверки: одинаковый вход обязан давать полное совпадение.

    Без этого теста любой перекос в самом сравнении (сортировка карт, единицы,
    привязка к позициям) читался бы как ошибка модели.
    """
    hand = _read()
    assert all(_fields(hand, hand).values())


def test_a_misread_suit_shows_up_as_a_card_field_and_nothing_else():
    """Масть — измеренный класс ошибки, и сверка обязана называть именно её."""
    truth = _read()
    misread = _read(
        players=[
            p.model_copy(update={"cards_in_log": ["Tc", "Td"]}) if p.nickname == "N3" else p
            for p in export_reading().players
        ]
    )
    fields = _fields(misread, truth)
    assert fields["showdown"] is False
    assert fields["stacks"] and fields["actions"] and fields["hand_no"]


def test_a_wrong_hand_number_is_the_first_and_cheapest_check():
    """Выдуманного номера в файле рума не существует, и сравнивать дальше нечего."""
    truth = _read()
    other = _read(hand_no="TM0000000000")
    assert _fields(other, truth)["hand_no"] is False


def test_a_phantom_player_shifts_the_seating_and_the_stacks_say_so():
    """Лишний игрок сдвигает раскладку целиком — это и ловит сверка по позициям.

    Найдено на живом прогоне: модель приняла служебный пузырёк банка времени за
    строку действия, и рассадка уехала на одно место. Ни одна контрольная сумма
    этого не поймала — поймал валидатор на сохранении фишек.
    """
    truth = _read()
    reading = export_reading()
    with_phantom = _read(
        players=[*reading.players, reading.players[0].model_copy(update={"nickname": "N9"})],
        actions=[
            *reading.actions[:1],
            reading.actions[1].model_copy(update={"nickname": "N9"}),
            *reading.actions[1:],
        ],
    )
    fields = _fields(with_phantom, truth)
    assert fields["players"] is False
    assert fields["stacks"] is False


def test_amounts_are_compared_with_the_tolerance_the_screen_forces():
    """Экран печатает два знака и обрезает: точное сравнение мерило бы округление.

    Первый полный прогон датасета так и вышел — шесть «расхождений» из
    девятнадцати оказались сотой доли ББ, а не ошибкой модели.
    """
    truth = _read()
    reading = export_reading()
    rounded = _read(
        actions=[
            a.model_copy(update={"amount": 14.63}) if a.amount == 14.62 else a
            for a in reading.actions
        ]
    )
    assert _fields(rounded, truth)["actions"] is True


def test_a_hand_read_from_a_screen_is_marked_as_a_whole_hand_before_comparison():
    """Сравнивать с текстом рума можно только руку целиком: у состояния итога нет."""
    assert _read().completeness is Completeness.HAND
