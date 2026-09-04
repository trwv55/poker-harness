"""Единый голос продукта: каждое сообщение, которое видит игрок, строится здесь.

«Правило единого голоса» (спека §4, CLAUDE.md): весь пользовательский текст
собирает только `presentation/`; `bot` и `worker` его лишь отправляют. Модуль —
чистые функции без побочных эффектов: ни Телеграм-API, ни БД, ни LLM (та же
конвейерная дисциплина, что у `contracts`/`analysis`/`explanation`) — входные
данные приходят аргументами, `Msg` возвращается значением.

**Регистр — по SESSIONS_UX.md.** Продукт продаётся незнакомым людям: ни
внутренней кухни («zone», «bracket-test», «assumption»), ни английских
токенов действий из контракта («fold»/«shove»/«call» — это язык движка, не
игрока). И главное слово этого модуля — «расхождение», никогда «ошибка»:
ядро судит решение против диапазона, а не против вскрытой карты, и решение,
проигравшее по случайности, ошибкой не считается (CLAUDE.md, дисциплина).

**Зона доверия — не подпись для галочки.** `Zone.ASSUMING` у `ScanItem`/
`PointVerdict` означает, что вывод опирается на угаданный диапазон оппонента;
`Zone.STRICT` — что нет. Пометка `_ASSUMING_MARKER` показывается ровно там,
где `zone is Zone.ASSUMING`, и ни разу больше — это и есть honesty-гарантия
задачи 17, проверенная тестом на обоих направлениях (строка есть/строки нет).

**Грамматика строки решения — намеренно без второго словаря (fix round 1).**
Первая версия писала `"{action} вместо {best}"`, а «вместо» управляет
родительным падежом («вместо шов**а**», не «вместо шов») — фраза читалась как
«колл вместо шов», и это была самая частая строка во всём продукте. Вместо
второго словаря словоформ (родительный параллельно именительному — источник
рассинхрона, стоивший проекту нескольких находок в других модулях) фраза
переписана как `"{action} (лучше: {best})"`: двоеточие после слова не
требует согласования падежа с существительным перед ним, поэтому одного
именительного падежа в `_ACTION_WORD` достаточно. `test_..._grammatically_correct`
пришпиливает буквальный рендер строки — регресс формулировки становится
красным тестом, а не тем, что заметит игрок раньше нас.

**«Лучше», не «верно» (fix round 2).** Первая правка round 1 заменила
«вместо» на «верно» и решила падеж, но не честность: «верно: {best}» ЗАЯВЛЯЕТ,
что сыгранное действие было неверным. В зоне `strict` это ещё защитимо (ядро
знает точный ответ), но у бОльшей части судимых точек продукта зона —
`assuming`: там ядро не знает, что альтернатива была верна, оно знает только,
что она выигрывает EV-ПРИ ДОГАДАННОМ диапазоне оппонента — а угаданный
диапазон стоит тремя строками выше маркером `_ASSUMING_MARKER` ровно за тем,
чтобы не выдавать догадку за факт. «Верно» в такой строке отменяет то, что
утверждает маркер рядом с ней — тот же манёвр, который запрещает правило
«расхождение, не ошибка» (CLAUDE.md), другими словами. «Лучше» утверждает
ровно то, что посчитано: у этой строки была выше EV — верно в обеих зонах,
не спорит с маркером и не требует знания, действительно ли сыгранное было
ошибкой.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from harness.analysis.scan import ScanSummary
from harness.contracts.analysis import AnalysisResult, SpotKind, Zone
from harness.contracts.raw import Street
from harness.presentation.keyboards import (
    Btn,
    deep_dive_button,
    escalation_buttons,
    verdict_buttons,
)

__all__ = [
    "Msg",
    "deep_dive_msg",
    "escalation_msg",
    "failed_msg",
    "hh_accepted_msg",
    "new_session_msg",
    "photo_soon_msg",
    "progress_text",
    "quota_exceeded_msg",
    "scan_summary_msg",
    "start_msg",
    "unsupported_document_msg",
]


class Msg(BaseModel):
    """Готовое к отправке сообщение: текст плюс раскладка инлайн-кнопок рядами."""

    text: str
    buttons: list[list[Btn]] = []


# --- Словари перевода внутренних токенов в слова игрока -------------------------

# `PointVerdict.action_taken`/`best_action` — токены движка (см. `preflop.py`),
# не то, что показывается игроку буквально.
_ACTION_WORD: dict[str, str] = {"fold": "фолд", "shove": "шов", "call": "колл"}

_SPOT_WORD: dict[SpotKind, str] = {
    SpotKind.PUSHFOLD_UNOPENED: "пуш-фолд",
    SpotKind.PUSHFOLD_FACING_SHOVE: "колл шова",
    SpotKind.PREFLOP_OTHER: "префлоп",
    SpotKind.POSTFLOP: "постфлоп",
}

_STREET_WORD: dict[Street, str] = {
    Street.PREFLOP: "Префлоп",
    Street.FLOP: "Флоп",
    Street.TURN: "Тёрн",
    Street.RIVER: "Ривер",
}

_ZONE_WORD: dict[Zone, str] = {Zone.STRICT: "строго", Zone.ASSUMING: "предполагая"}

# Пометка честности зоны (бриф задачи 17, дословно) — только у строк `assuming`.
_ASSUMING_MARKER = "по модели диапазонов"

_STATION_TEXT: dict[str, str] = {
    "parse": "Читаю стол…",
    "validate": "Проверяю руку…",
    "analyze": "Считаю эквити…",
    "explain": "Формулирую…",
}


def _action_word(action: str) -> str:
    return _ACTION_WORD.get(action, action)


def _spot_word(spot: SpotKind) -> str:
    return _SPOT_WORD.get(spot, spot.value)


def _fmt_bb(value_bb: float) -> str:
    """Знак минуса типографский (U+2212 «−»), не дефис — так задан бриф.

    Знак берётся ПОСЛЕ округления до 0.1, а не до: `-0.03` меньше нуля, но
    после округления до одного знака превращается в `0.0`, и если решать знак
    раньше округления, на экране игрока возникает «−0.0 bb» — читается как
    отдельная (мнимая) отрицательная величина вместо честного нуля (fix round 1).
    """
    magnitude = round(abs(value_bb), 1)
    sign = "−" if value_bb < 0 and magnitude != 0.0 else ""
    return f"{sign}{magnitude:.1f} bb"


def _quota_line(quota_left: int, quota_total: int) -> str:
    return f"разборов {quota_left}/{quota_total} за 24 ч"


def progress_text(station: Literal["parse", "validate", "analyze", "explain"]) -> str:
    """Строка прогресса, которой редактируется одно сообщение по станциям конвейера."""
    return _STATION_TEXT[station]


def scan_summary_msg(s: ScanSummary, quota_left: int, quota_total: int) -> Msg:
    """Сводка префлоп-скана: список расхождений по цене, кнопка разбора под каждым.

    «Расхождение», не «ошибка» (CLAUDE.md) — скан судит по равновесию и модельным
    диапазонам, не по факту выигрыша раздачи. Строки `zone is Zone.ASSUMING`
    несут `_ASSUMING_MARKER`, строки `strict` — нет.

    Заголовочное число — подписано ровно как «суммарная потеря по всем точкам
    разбора» (докстринг `ScanSummary.total_loss_bb`, дословно), а не как сумма
    списка ниже: `total_loss_bb` считает ВСЕ судимые точки файла, `items` —
    только те дороже порога 0.1bb, и на настоящем турнире первое число обычно
    ЧУТЬ БОЛЬШЕ суммы вторых. Два разных числа с одинаковой подписью — игрок
    решает, что мы ошиблись в счёте; разные подписи снимают это (fix round 1).

    `hands_failed` (руки, пропущенные политикой отказа скана) показывается
    только когда он не ноль — молчание о деградации ровно то, против чего
    спроектирован весь продукт (fix round 1, дискреционный пункт ревью).
    """
    lines = [f"Скан завершён: {s.hands_total} рук, {s.hands_with_decision} с решением."]
    if s.hands_failed:
        lines.append(f"Раздач не разобрано: {s.hands_failed} — не вошли в сводку.")
    lines.append(f"Суммарная потеря по всем точкам разбора: {_fmt_bb(s.total_loss_bb)}.")
    buttons: list[list[Btn]] = []

    if not s.items:
        lines.append("")
        lines.append("Расхождений дороже 0.1 bb не найдено.")
    else:
        lines.append("")
        lines.append("Топ расхождений:")
        for item in s.items:
            marker = f" ({_ASSUMING_MARKER})" if item.zone is Zone.ASSUMING else ""
            lines.append(
                f"№{item.hand_no} · {item.hero_class} · {_spot_word(item.spot)}: "
                f"{_action_word(item.action_taken)} (лучше: {_action_word(item.best_action)}) "
                f"— {_fmt_bb(item.ev_diff_bb)}{marker}"
            )
            buttons.append([deep_dive_button(item.hand_no)])

    lines.append("")
    lines.append(f"Доступно: {_quota_line(quota_left, quota_total)}.")
    return Msg(text="\n".join(lines), buttons=buttons)


def deep_dive_msg(
    res: AnalysisResult,
    elapsed_s: int,
    zone: Zone,
    quota_left: int,
    quota_total: int,
    dev_line: str | None = None,
) -> Msg:
    """Полный разбор раздачи: точки решения числами (текст LLM — задача 21) +
    статус-строка (⏱ время · зона доверия · остаток квоты) + три кнопки.

    Точки берутся в порядке `res.ranked` (самая дорогая первой) — это уже
    отфильтрованный и отранжированный список судимых точек (`error_cost.py`),
    без точек-пробелов, которым нечего показать честно.
    """
    lines = [f"Рука {res.hand_no}", ""]

    if not res.ranked:
        lines.append("По этой раздаче точек с вердиктом нет.")
    else:
        for idx in res.ranked:
            point = res.points[idx]
            marker = f" ({_ASSUMING_MARKER})" if point.zone is Zone.ASSUMING else ""
            lines.append(
                f"{_STREET_WORD.get(point.street, point.street.value)} · "
                f"{_spot_word(point.spot)}: {_action_word(point.action_taken)} "
                f"(лучше: {_action_word(point.best_action)}) — {_fmt_bb(point.ev_diff_bb)}{marker}"
            )

    lines.append("")
    status = f"⏱ {elapsed_s}с · зона: {_ZONE_WORD[zone]} · {_quota_line(quota_left, quota_total)}"
    lines.append(status)
    if dev_line is not None:
        lines.append(dev_line)

    return Msg(text="\n".join(lines), buttons=[verdict_buttons(res.hand_no)])


def escalation_msg(field: str, question: str, options: list[str]) -> Msg:
    """Эскалация валидатора — вопрос кнопками, не текстом (SESSIONS_UX): один тап."""
    return Msg(text=question, buttons=[escalation_buttons(field, options)])


def failed_msg(reason_public: str) -> Msg:
    """Разбор не удался — честная причина без внутренней кухни, без кнопок."""
    return Msg(text=f"Не получилось разобрать раздачу: {reason_public}")


def quota_exceeded_msg(hours_to_free: int) -> Msg:
    """Квота исчерпана — время возврата вместо остатка (он уже нулевой), без кнопок."""
    return Msg(
        text=(
            f"Дневной лимит разборов исчерпан. Следующий будет доступен через "
            f"{hours_to_free} ч — лимит считается за скользящие 24 ч."
        )
    )


# --- вход игрока: то, что говорит бот (задача 19) ----------------------------------


def start_msg() -> Msg:
    """`/start`: что это и что сделать прямо сейчас — без меню и без настроек.

    «Основное действие — не кнопка» (SESSIONS_UX): первый экран объясняет ровно
    один шаг (прислать файл), а не показывает карту продукта. Про сессии здесь
    не сказано ни слова намеренно — их создание молчаливое, и заставлять новичка
    думать о них до первого результата значило бы отменить это решение.
    """
    return Msg(
        text=(
            "Разбираю покерные раздачи с проверенным расчётом: точное считает код, "
            "словами объясняю отдельно.\n\n"
            "Пришлите файл раздач (.txt) из PokerCraft — сделаю префлоп-скан турнира и "
            "покажу расхождения по цене. Под каждой раздачей будет кнопка «разобрать».\n\n"
            "/new — начать новую сессию."
        )
    )


def hh_accepted_msg() -> Msg:
    """Файл принят — подтверждение приёма, ещё не результат.

    Ни числа рук, ни времени ожидания: ни того, ни другого бот в этот момент не
    знает (файл ещё не разобран), а называть их наугад запрещено (CLAUDE.md).
    """
    return Msg(text="Файл принят. Считаю префлоп-скан — пришлю сводку, когда закончу.")


def unsupported_document_msg() -> Msg:
    """Документ не `.txt` — отказ сразу, с называнием того, что сработает."""
    return Msg(
        text=(
            "Такой файл я не разберу. Нужен .txt с раздачами из PokerCraft — "
            "пришлите его, и запущу скан."
        )
    )


def photo_soon_msg() -> Msg:
    """Скриншот стола — заглушка до задачи 22 (vision).

    Честно называет, что именно ещё не готово, и тут же даёт рабочий путь: молчание
    в ответ на главное действие продукта («кинул скрин») было бы худшим из ответов.
    """
    return Msg(
        text=(
            "Разбор скриншотов ещё готовится. Пока пришлите файл раздач (.txt) из "
            "PokerCraft — по нему сделаю префлоп-скан турнира."
        )
    )


def new_session_msg(title: str, previous_closed: bool) -> Msg:
    """`/new`: новая сессия открыта. Про закрытие предыдущей — только если она была.

    `previous_closed` — не украшение: у первой сессии игрока закрывать нечего, и
    безусловная фраза «предыдущая закрыта» была бы сообщением о событии, которого
    не произошло.
    """
    lines = [f"Новая сессия: {title}."]
    if previous_closed:
        lines.append("Предыдущая закрыта — дальше всё пойдёт в новую.")
    else:
        lines.append("Всё, что пришлёте дальше, попадёт в неё.")
    return Msg(text=" ".join(lines))
