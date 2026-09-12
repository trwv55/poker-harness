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

**Постфлоп-точка — числа без цены.** У неё нет ни `ev_diff_bb`, ни интервала:
перебор борда отвечает на вопрос «сколько блефов нужно в его ставящем
диапазоне», а не «сколько стоило решение». Поэтому её строки стоят отдельно от
строк расхождений и не несут ни цены, ни слова «лучше» — кроме случая, когда
ядро назвало лучшую линию (`PointVerdict.best_action`), и тогда рядом названо
допущение, на котором она держится. Ривер, тёрн и флоп печатаются одними
словами и одной функцией (`_postflop_call_lines`): считают их разные
инструменты, но подписи у чисел одни и те же.

**Форма «около нуля» — вердикт, а не отказ.** У части точек интервал EV лежит
по обе стороны нуля: при одних моделях поведения оппонентов лучше входить, при
других пасовать. Раньше такая точка до игрока не доходила вовсе — ядро
отказывалось называть число, — и вместе с числом пропадали три вещи, которые у
нас на неё есть: знак и порядок величины, ширина интервала (она и есть мера
маргинальности решения, объяснять её словами не надо) и потолок цены выбора.

Сюда доходят только те из них, чей интервал узок: широкий отсеивается в ядре
(`analysis.preflop`), потому что «около нуля» при широком интервале значило бы
не «решение неважное», а «мы не знаем», — два противоположных сообщения одной
формой. Поэтому строка «около нуля» вправе обещать, что выбор дёшев, и список
таких строк отсортирован дешёвыми вперёд: обещание тем сильнее, чем меньше
названный в нём потолок.
Строка такой точки не содержит «лучше»: упрекать не за что, и грамматика
расхождения к ней не применяется. Потолок округляется ВВЕРХ
(`_fmt_ceiling_bb`) — строка обещает «не больше столько-то», и округление вниз
сделало бы обещание неверным.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from math import ceil, floor
from typing import Literal, NamedTuple

from pydantic import BaseModel, model_validator

from harness.contracts.analysis import (
    AllInEvent,
    AnalysisResult,
    ChipMove,
    EvInterval,
    EvSplit,
    Finding,
    PointVerdict,
    ScanItem,
    ScanSummary,
    SpotKind,
    StackTrajectory,
    TournamentReport,
    Zone,
    river_call_detail,
    turn_flop_call_detail,
)
from harness.contracts.calcs import (
    CalcName,
    CalcResult,
    CoverageResult,
    DefenseResult,
    FrequencyResult,
    FrequencyStat,
    LeaksResult,
    Measurement,
    Subject,
    ThresholdOutcome,
    ThresholdResult,
    Window,
)
from harness.contracts.explanation import TournamentTextOut, VerdictTextOut
from harness.contracts.history import (
    MAX_NOTE_TEXT_CHARS,
    NOTE_COLORS,
    LeaksOverview,
    LeakStat,
    NoteRecord,
    OpponentRecord,
    SessionLine,
    SessionSummary,
)
from harness.contracts.raw import Street
from harness.explanation.hand_replay import HandReplay, chips
from harness.presentation.keyboards import (
    MAIN_MENU,
    MENU_LEAKS,
    MENU_NOTES,
    MENU_SESSIONS,
    MENU_SETTINGS,
    MENU_TOURNAMENT,
    Btn,
    deep_dive_button,
    escalation_buttons,
    note_buttons_for_hand,
    note_color_buttons,
    note_row,
    session_buttons,
    set_nickname_button,
    verdict_buttons,
)

__all__ = [
    "Msg",
    "Photo",
    "analysis_unavailable_msg",
    "ask_gg_nickname_msg",
    "bot_failure_msg",
    "deep_dive_msg",
    "disagreement_saved_msg",
    "escalation_msg",
    "failed_msg",
    "gg_nickname_saved_msg",
    "gg_nickname_too_long_msg",
    "help_msg",
    "hh_accepted_msg",
    "hh_duplicate_msg",
    "hh_prompt_msg",
    "invite_accepted_msg",
    "invite_created_msg",
    "invite_required_msg",
    "leaks_msg",
    "new_session_msg",
    "not_a_hand_msg",
    "note_color_prompt_msg",
    "note_color_saved_msg",
    "note_deleted_msg",
    "note_gone_msg",
    "note_prompt_msg",
    "note_saved_msg",
    "notes_msg",
    "progress_text",
    "question_msg",
    "question_refusal_msg",
    "question_too_long_msg",
    "question_usage_msg",
    "quota_exceeded_msg",
    "range_image_title",
    "range_photos",
    "ranges_msg",
    "replay_msg",
    "replay_unavailable_msg",
    "scan_summary_msg",
    "screenshot_too_large_msg",
    "send_as_file_msg",
    "session_summary_msg",
    "session_unavailable_msg",
    "sessions_msg",
    "settings_msg",
    "start_msg",
    "tournament_report_msg",
    "tournament_story_msg",
    "unknown_button_msg",
    "unknown_text_msg",
    "unsupported_document_msg",
    "vision_answer_not_a_number_msg",
    "vision_answer_saved_msg",
    "vision_gave_up_msg",
    "vision_manual_entry_msg",
]


class Photo(BaseModel):
    """Картинка к сообщению: файл на диске и подпись под ним.

    Путь, а не байты: картинки диапазонов рисует и кладёт на диск воркер
    (`worker.pipeline._render_ranges`), в `analyses.range_images` лежат пути, и
    таскать мегабайты через слой текста незачем.
    """

    path: str
    caption: str


class Msg(BaseModel):
    """Готовое к отправке сообщение: текст, инлайн-кнопки, меню, картинки.

    **`buttons` и `menu` взаимно исключены** — не стилистикой, а Bot API: у
    сообщения ровно одно поле `reply_markup`, и положить туда обе клавиатуры
    нельзя. Проверяет это сам тип, а не вызывающий: ошибка иначе всплыла бы
    отказом Телеграма в проде (`test_a_message_cannot_carry_both_keyboards`).

    `menu` — постоянная нижняя клавиатура (SESSIONS_UX): список рядов подписей.
    Телеграм присылает нажатие такой кнопки обычным текстовым сообщением, и
    разбирает его `bot/menus.py`.
    """

    text: str
    buttons: list[list[Btn]] = []
    menu: list[list[str]] | None = None
    photos: list[Photo] = []
    # Разметка Телеграма для этого сообщения (`HTML`), либо `None` — текст как
    # есть. Ставится ровно там, где выделение несёт смысл: реплей выделяет точку
    # решения героя (спека §5.6). Везде, где разметки нет, экранировать текст не
    # требуется, и общий режим на всё подряд её бы и потребовал.
    parse_mode: str | None = None

    @model_validator(mode="after")
    def _one_keyboard_at_a_time(self) -> Msg:
        if self.buttons and self.menu is not None:
            raise ValueError(
                "у сообщения одна клавиатура: либо инлайн-кнопки, либо нижнее меню"
            )
        return self


# --- Словари перевода внутренних токенов в слова игрока -------------------------

# `PointVerdict.action_taken`/`best_action` — токены движка (см. `preflop.py`);
# словарь переводит их в слова игрока. Строка, которой в словаре нет,
# показывается как есть: так сюда приходит готовая формулировка развилки из
# `preflop._BEST_DEPENDS_ON_BEHIND`.
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

# Потолок строк сводки скана (round 5, Item J). `ScanSummary.items` ничем не
# ограничен — это правильно для данных (`tournaments.scan_summary` хранит их
# целиком), но у Телеграма `sendMessage` жёстко ограничен 4096 символами, а
# строка пункта вместе с кнопкой стоит ~85 символов: примерно с 48-го пункта
# сообщение перестало бы отправляться ВООБЩЕ. Отказ при этом не тихий и не
# дешёвый: `TelegramSender.send` делает `raise_for_status()`, задача уходит в
# ретрай и игрок платит тремя полными пересканами файла, прежде чем услышит
# хоть что-то. Измеренные фикстуры дают 9 пунктов — то есть само по себе это
# не выстрелит завтра, и именно поэтому ограничения тут и не было.
# 20 — с запасом внутри лимита (~1.8 тыс. символов) и всё ещё осмысленный
# список: сводка нужна, чтобы выбрать, что разбирать, а не чтобы прочитать
# турнир целиком.
_MAX_RENDERED_SCAN_ITEMS = 20

# Сколько строк «около нуля» показывается. Тот же ограничитель, что и у списка
# расхождений выше (жёсткий лимит `sendMessage` в 4096 символов), но потолок
# ниже: строка «около нуля» длиннее — в ней интервал и потолок цены, — а
# ценность списка другая. Список расхождений говорит, ЧТО разобрать; список
# «около нуля» говорит, что разбирать нечего, и десяти строк для этого хватает.
_MAX_RENDERED_CLOSE_CALLS = 10

# Что варьируется, когда мы говорим «по моделям»: в неоткрытом банке —
# готовность стола отвечать на шов, против чужого шова — то, с какими руками
# оппонент идёт олл-ин. Слово игрока, не внутренний термин (SESSIONS_UX).
_MODELS_WORD: dict[SpotKind, str] = {
    SpotKind.PUSHFOLD_UNOPENED: "по моделям колла",
    SpotKind.PUSHFOLD_FACING_SHOVE: "по моделям шова",
}

# Действие, которое сравнивается с пасом в форме «около нуля»: в неоткрытом
# банке это шов, против чужого шова — колл. Оба варианта называются целиком —
# «шов или фолд», — потому что вердикт здесь и есть «оба допустимы».
_ACTIVE_WORD: dict[SpotKind, str] = {
    SpotKind.PUSHFOLD_UNOPENED: "шов",
    SpotKind.PUSHFOLD_FACING_SHOVE: "колл",
}

# Две станции с одинаковой строкой — не дубль: игроку они и правда неразличимы
# («читаю стол»), а конвейеру нет. `read` — чтение скриншота моделью, `parse` —
# разбор файла раздач кодом; путать их в трейсе и в дедлайне задачи нельзя, а в
# сообщении игроку различать нечего.
_STATION_TEXT: dict[str, str] = {
    "ask": "Считаю по вашим раздачам…",
    "read": "Читаю стол…",
    "parse": "Читаю стол…",
    "validate": "Проверяю руку…",
    "analyze": "Считаю эквити…",
    "explain": "Формулирую…",
}


def _action_word(action: str) -> str:
    return _ACTION_WORD.get(action, action)


def _spot_word(spot: SpotKind) -> str:
    return _SPOT_WORD.get(spot, spot.value)


def _bb_number(value_bb: float) -> str:
    """Число в bb без единицы — для пар вида «34.0 → 12.5 bb».

    Знак минуса типографский (U+2212 «−»), не дефис — так задан бриф. Знак
    берётся ПОСЛЕ округления до 0.1, а не до: `-0.03` меньше нуля, но после
    округления до одного знака превращается в `0.0`, и если решать знак раньше
    округления, на экране игрока возникает «−0.0» — читается как отдельная
    (мнимая) отрицательная величина вместо честного нуля (fix round 1).
    """
    magnitude = round(abs(value_bb), 1)
    sign = "−" if value_bb < 0 and magnitude != 0.0 else ""
    return f"{sign}{magnitude:.1f}"


def _fmt_bb(value_bb: float) -> str:
    """То же число с единицей. Округление и знак — общие с `_bb_number`, а не
    вторая их копия: две формы одной величины обязаны округляться одинаково."""
    return f"{_bb_number(value_bb)} bb"


def _fmt_signed_bb(value_bb: float) -> str:
    """То же, что `_fmt_bb`, но плюс у положительного числа проговаривается.

    В интервале «−0.3 … +0.8» знак верхнего конца несёт смысл: он и говорит, что
    интервал пересекает ноль. Без явного плюса читатель видит два числа и должен
    сам заметить, что у одного знак есть, а у другого нет.
    """
    magnitude = round(abs(value_bb), 1)
    if magnitude == 0.0:
        return "0.0 bb"
    return f"{'−' if value_bb < 0 else '+'}{magnitude:.1f} bb"


def _tenth_down(value_bb: float) -> float:
    """Десятая ВНИЗ. `round(x * 10, 6)` — гвард от двоичного представления: без
    него `floor(-0.3 * 10)` даёт −4, то есть −0.4 вместо −0.3."""
    return floor(round(value_bb * 10, 6)) / 10


def _tenth_up(value_bb: float) -> float:
    """Десятая ВВЕРХ, с тем же гвардом, что и `_tenth_down`."""
    return ceil(round(value_bb * 10, 6)) / 10


def _fmt_ceiling_bb(value_bb: float) -> str:
    """Потолок цены — округлённый ВВЕРХ до той же десятой, что и остальные числа.

    Вверх, а не к ближайшему: строка обещает игроку «не больше столько-то», и
    округление вниз сделало бы обещание неверным на величину округления.
    """
    return f"{_tenth_up(value_bb):.1f} bb"


def _interval_words(spot: SpotKind, interval: EvInterval) -> str:
    """Интервал и потолок цены одной фразой — общая часть сводки и разбора.

    Концы округляются НАРУЖУ (нижний вниз, верхний вверх), а не к ближайшему.
    Иначе показанный интервал оказывается уже посчитанного, и рядом с ним
    появляется потолок, которого в нём не видно: «от −3.3 до +2.6, разница не
    больше 3.4» — читатель вправе счесть это опиской. При округлении наружу
    потолок равен модулю худшего из ПОКАЗАННЫХ концов, и строка сходится сама с
    собой (`test_the_close_call_line_agrees_with_itself_after_rounding`).
    """
    return (
        f"{_MODELS_WORD.get(spot, 'по моделям')} от "
        f"{_fmt_signed_bb(_tenth_down(interval.low_bb))} до "
        f"{_fmt_signed_bb(_tenth_up(interval.high_bb))}, разница между вариантами — "
        f"не больше {_fmt_ceiling_bb(interval.cost_ceiling_bb)}"
    )


def _close_call_line(item: ScanItem) -> str:
    """Строка точки «около нуля» в сводке: оба варианта, интервал, потолок цены."""
    assert item.interval is not None  # в `close_calls` попадают только точки с интервалом
    marker = f" ({_ASSUMING_MARKER})" if item.zone is Zone.ASSUMING else ""
    return (
        f"№{item.hand_no} · {item.hero_class} · {_spot_word(item.spot)}: "
        f"{_ACTIVE_WORD.get(item.spot, 'вход')} или фолд — около нуля, "
        f"{_interval_words(item.spot, item.interval)}{marker}"
    )


def _prose_lines(text: str | None) -> list[str]:
    """Абзац модели под строкой точки — с отступом, чтобы было видно, где чья речь.

    Пустой текст не даёт пустой строки: разбор без прозы (модель недоступна либо
    её текст не прошёл проверку верности) обязан выглядеть цельным, а не
    дырявым (`test_deep_dive_msg_without_prose_has_no_holes_in_it`).
    """
    if text is None or not text.strip():
        return []
    return [f"    {line.strip()}" for line in text.strip().splitlines() if line.strip()]


def _plural_form(count: int, one: str, few: str, many: str) -> str:
    """Форма слова по числу: 1 — `one`, 2–4 — `few`, остальное — `many`.

    Правило трёх форм, а не двух. Управляет и существительным, и глаголом: при
    пяти и больше подлежащее в родительном множественного, и сказуемое встаёт в
    средний род единственного числа — «Показано 5 абзацев», а не «Показаны»
    (`test_tournament_story_msg_counts_paragraphs_grammatically`).
    """
    tail_100 = abs(count) % 100
    tail_10 = abs(count) % 10
    if 11 <= tail_100 <= 14:
        return many
    if tail_10 == 1:
        return one
    if 2 <= tail_10 <= 4:
        return few
    return many


def _quota_line(quota_left: int, quota_total: int) -> str:
    return f"разборов {quota_left}/{quota_total} за 24 ч"


def progress_text(
    station: Literal["ask", "read", "parse", "validate", "analyze", "explain"],
) -> str:
    """Строка прогресса, которой редактируется одно сообщение по станциям конвейера."""
    return _STATION_TEXT[station]


def scan_summary_msg(s: ScanSummary, quota_left: int, quota_total: int) -> Msg:
    """Сводка префлоп-скана: список расхождений по цене, кнопка разбора под каждым.

    «Расхождение», не «ошибка» (CLAUDE.md) — скан судит по равновесию и модельным
    диапазонам, не по факту выигрыша раздачи. Строки `zone is Zone.ASSUMING`
    несут `_ASSUMING_MARKER`, строки `strict` — нет.

    Заголовочное число подписано тем множеством, по которому посчитано, —
    оценёнными решениями, то есть судимыми точками (`error_cost.is_judged`),
    теми же, что стоят числителем в строке покрытия строкой выше. «По всем
    точкам разбора» накрывало бы подписью и `points_total - points_judged`
    точек без вердикта, которых в сумме нет: ноль несчитанной точки означает
    «не посчитано», а не «сыграно верно» (докстринг модуля `error_cost`), и
    сумма, подписанная всем множеством, читается как «потерь не было»
    (`test_scan_summary_msg_scopes_the_loss_to_the_judged_points`).

    Подпись не пересказывает и список ниже: `total_loss_bb` считает все судимые
    точки файла, `items` — только те дороже порога 0.1bb, поэтому первое число
    и сумма вторых расходятся
    (`test_scan_summary_msg_total_loss_label_differs_from_items_sum`).

    `hands_failed` (руки, пропущенные политикой отказа скана) показывается
    только когда он не ноль — молчание о деградации ровно то, против чего
    спроектирован весь продукт (fix round 1, дискреционный пункт ревью).

    Строка покрытия (`points_judged` из `points_total`) печатается ВСЕГДА, а
    пустой список расхождений прямо говорит, что пустота относится к оценённым
    решениям, и называет число неоценённых. Точка без вердикта в список не
    попадает по построению, поэтому «расхождений не найдено» без покрытия рядом
    читается как «сыграно чисто» — то же молчание о деградации, что и
    умолчанный `hands_failed` абзацем выше, только на уровне точек.

    Список обрезается `_MAX_RENDERED_SCAN_ITEMS` (round 5, Item J — обоснование
    числа там же) и, если обрезан, говорит об этом прямо: сколько найдено и
    сколько показано. Молча показать 20 из 60 — та же деградация без огласки,
    что и `hands_failed` абзацем выше.
    """
    lines = [f"Скан завершён: {s.hands_total} рук, {s.hands_with_decision} с решением."]
    if s.hands_failed:
        lines.append(f"Раздач не разобрано: {s.hands_failed} — не вошли в сводку.")
    lines.append(f"Оценено решений: {s.points_judged} из {s.points_total}.")
    lines.append(f"Суммарная потеря в оценённых решениях: {_fmt_bb(s.total_loss_bb)}.")
    buttons: list[list[Btn]] = []

    if not s.items:
        lines.append("")
        # Числа стоят после двоеточия намеренно: «остальные 181 решение» требует
        # согласования с числительным, а строка собирается для любого числа.
        lines.append(
            f"Среди оценённых решений расхождений дороже 0.1 bb не найдено. "
            f"Решений без оценки: {s.points_total - s.points_judged} — про них "
            f"расчёт не говорит ничего."
        )
    else:
        shown = s.items[:_MAX_RENDERED_SCAN_ITEMS]
        lines.append("")
        if len(shown) < len(s.items):
            lines.append(
                f"Топ расхождений — показаны {len(shown)} самых дорогих "
                f"из {len(s.items)} найденных:"
            )
        else:
            lines.append("Топ расхождений:")
        for item in shown:
            marker = f" ({_ASSUMING_MARKER})" if item.zone is Zone.ASSUMING else ""
            lines.append(
                f"№{item.hand_no} · {item.hero_class} · {_spot_word(item.spot)}: "
                f"{_action_word(item.action_taken)} (лучше: {_action_word(item.best_action)}) "
                f"— {_fmt_bb(item.ev_diff_bb)}{marker}"
            )
            buttons.append([deep_dive_button(item.hand_no)])

    if s.close_calls:
        shown_close = s.close_calls[:_MAX_RENDERED_CLOSE_CALLS]
        lines.append("")
        head = "Решения около нуля — расчёт не спорит ни с одним из вариантов"
        if len(shown_close) < len(s.close_calls):
            head += (
                f" (показаны {len(shown_close)} из {len(s.close_calls)}, "
                f"самые дешёвые — первыми)"
            )
        lines.append(f"{head}:")
        lines.extend(_close_call_line(item) for item in shown_close)

    lines.append("")
    lines.append(f"Доступно: {_quota_line(quota_left, quota_total)}.")
    return Msg(text="\n".join(lines), buttons=buttons)


# Оговорка о непроверенном. «Проверить было нечем» — не то же, что «проверено и
# сошлось», и разница обязана быть видна игроку, а не только в трейсе.
_NOT_CHECKED_PREFIX = "Проверить на этом экране было нечем:"

# Вердикта нет ни по одной точке. Внутренние формулировки ядра написаны для
# разбора, а не для чтения вслух, и игроку не пересказываются.
_NO_VERDICT_LINE = "По этой раздаче точек с вердиктом нет."


# Слова про постфлоп-точку — одни и те же на всех трёх улицах. Требование к
# ставящему диапазону — число блефов на заданное вэлью — печатается вместе с
# долей, которую эти блефы в диапазоне занимают: одно число отвечает «сколько»,
# второе — «насколько это много».
#
# Чего в этих строках НЕТ и почему (решение владельца):
# * разложения борда по исходам (сколько комбо бьёт, проигрывает, делит) — это
#   не диапазон соперника, а полный перебор возможного, и читается как чужой
#   диапазон, которого мы не знаем;
# * строки «доказать фолд не удалось» и оговорок про то, чем тёрн и флоп
#   отличаются от ривера, — продукт не рассказывает о том, чего не умеет
#   (SESSIONS_UX).
_CALL_HEAD = "{street}: банк {pot}, доставить {to_call} — колл окупается от {equity} эквити."
_CALL_ASSUMPTION = "Допущение: сильнейшие руки он ставит."


def _rounded_bluffs(value: float) -> int:
    """Число блефов целым: к ближайшему, половина вверх.

    Вверх, а не по правилу `round` (оно округляет половину к чётному): требование
    к диапазону читается как «столько-то рук», и 0.5 обязана дать 1, а не 0
    (`test_half_a_bluff_is_rounded_up_to_one`).
    """
    return int(value + 0.5)


class _CallNumbers(NamedTuple):
    """Числа постфлоп-точки в форме, одинаковой для всех трёх улиц.

    Ривер и пара «тёрн, флоп» приходят из разных расчётов и лежат в `detail`
    под разными ключами, но показываются игроку одними словами. Общая форма
    здесь — то место, где две редакции этих слов не разойдутся.

    `bluffs` — `None`, когда требуемого числа блефов не существует (тёрн и
    флоп: не хватает и всего борда); `share` — `None` там же.
    `beyond_the_board` — требование превышает то, что борд вмещает.
    """

    pot_before: int
    to_call: int
    required_equity: float
    min_value_combos: int
    bluffs: float | None
    share: float | None
    beyond_the_board: bool


def _call_numbers(point: PointVerdict) -> _CallNumbers | None:
    """Числа постфлоп-точки из `detail` — или `None`, если их там нет."""
    river = river_call_detail(point)
    if river is not None:
        return _CallNumbers(
            pot_before=river.pot_before,
            to_call=river.to_call,
            required_equity=river.required_equity,
            min_value_combos=river.min_value_combos,
            bluffs=river.bluffs_needed_min_value,
            share=river.bluff_share,
            beyond_the_board=river.fold_proven,
        )
    early = turn_flop_call_detail(point)
    if early is not None:
        return _CallNumbers(
            pot_before=early.pot_before,
            to_call=early.to_call,
            required_equity=early.required_equity,
            min_value_combos=early.min_value_combos,
            bluffs=early.bluffs_needed_min_value,
            share=early.bluff_share,
            beyond_the_board=early.bluffs_needed_min_value is None,
        )
    return None


def _postflop_call_lines(point: PointVerdict) -> list[str]:
    """Разбор постфлоп-точки: цена решения, требование к диапазону, лучшая линия.

    Пустой список — у точки нет посчитанных чисел (`_call_numbers`), и печатать
    нечего. Строка про блефы пропускается, когда требования нет вовсе — вэлью
    старшего класса на борде не осталось или блефов нужно меньше одного:
    «нужно 0 блефов» не утверждение, а вырожденный случай
    (`test_a_degenerate_river_requirement_prints_only_the_price_of_the_call`).

    Числа блефов может не существовать вовсе — тогда строка называет вэлью и
    говорит, что столько блефов борд не вмещает
    (`test_a_requirement_beyond_the_board_prints_no_number_of_bluffs`).

    Лучшая линия называется ровно тогда, когда её назвало ядро
    (`PointVerdict.best_action`), то есть когда борд исчерпан; вместе с ней
    называется и единственное допущение, на котором она стоит.
    """
    numbers = _call_numbers(point)
    if numbers is None:
        return []
    lines = [
        _CALL_HEAD.format(
            street=_STREET_WORD.get(point.street, point.street.value),
            pot=chips(numbers.pot_before),
            to_call=chips(numbers.to_call),
            equity=_fmt_pct(100.0 * numbers.required_equity),
        )
    ]
    lines.extend(_bluff_line(numbers))
    if point.best_action:
        # Вывод отдельной строкой, а не хвостом предыдущей: строка про блефы у
        # вырожденного требования не печатается вовсе, и лучшая линия ушла бы
        # вместе с ней (`test_a_proven_fold_names_the_line_even_without_the_bluff_line`).
        lines.append(f"    Лучше: {_action_word(point.best_action)}. {_CALL_ASSUMPTION}")
    return lines


def _bluff_line(numbers: _CallNumbers) -> list[str]:
    """Строка требования к ставящему диапазону — или пустой список, если его нет."""
    if numbers.min_value_combos <= 0:
        return []
    combos_word = _plural_form(
        numbers.min_value_combos, "комбинацию", "комбинации", "комбинаций"
    )
    head = (
        f"    Чтобы колл вышел в ноль, на {numbers.min_value_combos} {combos_word} "
        f"несомненного вэлью ему "
    )
    if numbers.bluffs is None or numbers.share is None:
        return [f"{head}нужно больше блефов, чем на этом борде существует."]
    bluffs = _rounded_bluffs(numbers.bluffs)
    if bluffs <= 0:
        return []
    need_word = _plural_form(bluffs, "нужен", "нужно", "нужно")
    bluffs_word = _plural_form(bluffs, "блеф", "блефа", "блефов")
    tail = (
        "больше, чем на этом борде существует"
        if numbers.beyond_the_board
        else (
            f"то есть блефом должно быть {_fmt_pct(100.0 * numbers.share)} "
            f"его ставящего диапазона"
        )
    )
    return [f"{head}{need_word} {bluffs} {bluffs_word} — {tail}."]


# Обрезка называется вслух и одним и тем же словом везде, где она случается:
# экран заметок (`_fitted`) и разбор, которому не хватило места (`_shrink_prose`).
_MARKER = " […показано не целиком]"


def _shrink_prose(rows: list[tuple[str, bool]], budget: int) -> list[tuple[str, bool]]:
    """Уместить строки в бюджет, срезая ТОЛЬКО прозу модели (сырую, до экранирования).

    Слова необязательны, числа обязательны — то же правило, по которому
    `worker.pipeline` отдаёт разбор без прозы, когда модель не ответила.
    Строка, которая влезает целиком, не трогается; которой не хватает места
    даже на маркер — пропадает целиком (иначе `_fitted` вернул бы один маркер
    и пробил бюджет на его длину).
    """
    fixed = sum(len(text) + 1 for text, is_prose in rows if not is_prose)
    left = budget - fixed
    out: list[tuple[str, bool]] = []
    for text, is_prose in rows:
        if not is_prose:
            out.append((text, False))
            continue
        if len(text) + 1 <= left:
            out.append((text, True))
            left -= len(text) + 1
            continue
        if left <= len(_MARKER) + 1:
            continue
        cut = _fitted(text, left - 1)
        out.append((cut, True))
        left -= len(cut) + 1
    return out


def _render_html(head: str, rows: list[tuple[str, bool]]) -> str:
    """Блок уже с разметкой; остальные строки экранируются здесь, ПОСЛЕ усадки."""
    return "\n".join([head, *(_html_escape(text) for text, _ in rows)])


def _fit_html(head: str, rows: list[tuple[str, bool]], limit: int) -> str:
    """Итоговый текст не длиннее `limit` — меряется ПОСЛЕ экранирования и разметки.

    Режется сырая проза, а экранируется то, что осталось: разрез по
    экранированному попал бы внутрь `&amp;`, и что Телеграм сделает с `&am` —
    неизвестно.

    Отсюда и поиск бюджета. Сколько сырых символов прозы даст текст ровно в
    `limit`, заранее не знает никто: экранирование раздувает строку от нуля
    (обычная проза) до пятикратного (текст из одних `&`), и вычесть перелёт из
    бюджета значит выбросить всю прозу там, где влезала пятая её часть. Поэтому
    бюджет подбирается делением пополам — берётся самый щедрый, при котором
    ИТОГОВЫЙ текст ещё влезает (`test_the_model_prose_is_cut_before_the_
    replay_is`). Каждый кандидат проверяется по факту, а не по оценке.

    Ничего не влезло — остаются числа без прозы, даже если и они длиннее
    предела: резать их этой функции нельзя (спека §5.6).
    """
    text = _render_html(head, rows)
    if len(text) <= limit:
        return text
    best = _render_html(head, [(t, is_prose) for t, is_prose in rows if not is_prose])
    low, high = 0, limit
    while low <= high:
        budget = (low + high) // 2
        candidate = _render_html(head, _shrink_prose(rows, budget))
        if len(candidate) <= limit:
            best = candidate
            low = budget + 1
        else:
            high = budget - 1
    return best


def deep_dive_msg(
    res: AnalysisResult,
    elapsed_s: int,
    zone: Zone | None,
    quota_left: int,
    quota_total: int,
    dev_line: str | None = None,
    verdict: VerdictTextOut | None = None,
    replay: HandReplay | None = None,
    not_checked: Sequence[str] = (),
    note_nicks: Sequence[str] = (),
) -> Msg:
    """Полный разбор раздачи: точки решения числами (текст LLM — задача 21) +
    статус-строка (⏱ время · зона доверия · остаток квоты) + три кнопки.

    Точки с ценой берутся в порядке `res.ranked` (самая дорогая первой) — это
    уже отфильтрованный и отранжированный список судимых точек
    (`error_cost.py`), без точек-пробелов, которым нечего показать честно.

    **Под ними — постфлоп-точки, у которых цены нет** (`_postflop_call_lines`):
    их числа посчитаны перебором борда, а не против диапазона, и в ранжирование
    они не входят по построению (`SpotKind.POSTFLOP` вне `JUDGED_SPOTS`).
    Порядок такой, а не по улицам: сначала то, что стоило денег, потом то, что
    посчитано без цены; внутри второй половины улицы идут в порядке раздачи —
    том же, в каком лежат точки решения. Строка «точек с вердиктом нет»
    печатается, только когда нет ни одной из двух половин
    (`test_a_river_point_alone_is_not_a_hand_without_a_verdict`).

    **`zone=None` — «зоны нет», и тогда её нет и в строке (round 5, Item H).**
    Прежняя сигнатура требовала `Zone`, и вызывающий, которому нечего было
    сказать (ни одной судимой точки), подставлял `Zone.STRICT` — самую
    уверенную подпись продукта под сообщением «точек с вердиктом нет». CLAUDE.md
    разрешает `strict` только там, где вывод не опирается на угаданный диапазон;
    вывода в этом случае нет вообще, а значит нет и зоны. Молчание тут честнее
    любого слова, поэтому сегмент просто исчезает из статус-строки.

    Зона относится ко ВСЕЙ руке, поэтому вызывающий обязан выводить её из всех
    судимых точек, а не из первой (`worker.pipeline._hand_zone` — единственный
    такой вызывающий; там же и правило: «строго» только если строги все).

    **`not_checked` — то, что на этом входе проверить было нечем** (`Verdict.
    not_checked`), и оно обязано дойти до игрока. Скрин, где шоудаун решён на
    доукомплектованных картах, без этой строки показывал вскрытие, которого
    никто не видел, и подписывался «зона: строго» — то есть ровно то, что
    пометка обещала не допустить. Понижение зоны делает вызывающий
    (`_hand_zone`), а называет непроверенное эта строка: одно без другого
    оставляет либо неназванную оговорку, либо неоправданную уверенность.

    **Блок «Что было» — первым** (спека §5.6, план 2026-09-12): проза короче
    построчного реплея, ради которого его когда-то прятали за кнопку; в тесноте
    режется проза модели, не блок (`_shrink_prose`). `parse_mode="HTML"` только
    при наличии блока — иначе экранировать пришлось бы весь текст всюду.

    `note_nicks` — оппоненты, на которых можно записать заметку одним тапом
    (решение владельца 2026-09-04: путь заметки начинается ИЗ РАЗБОРА). Ники
    приходят только со скрина; на HH-пути список пуст, потому что там их нет.
    """
    # Строка и флаг «это проза модели»: в тесноте режется только она
    # (`_shrink_prose`), а числа и статус остаются на месте.
    rows: list[tuple[str, bool]] = []
    if replay is not None:
        rows.append(("", False))  # пустая строка после блока; сам блок — `head`
    rows.extend([(f"Рука {res.hand_no}", False), ("", False)])

    prose = {} if verdict is None else {point.dp_index: point.text for point in verdict.points}
    postflop = [line for point in res.points for line in _postflop_call_lines(point)]

    if not res.ranked and not postflop:
        rows.append((_NO_VERDICT_LINE, False))
    elif res.ranked:
        for idx in res.ranked:
            point = res.points[idx]
            marker = f" ({_ASSUMING_MARKER})" if point.zone is Zone.ASSUMING else ""
            street = _STREET_WORD.get(point.street, point.street.value)
            interval = point.interval
            if interval is not None and interval.near_zero:
                # Форма «около нуля»: ни одного «лучше» — упрёка тут нет, — зато
                # все три числа, которых не давал прежний отказ: точка, интервал
                # и потолок цены выбора.
                active = _ACTIVE_WORD.get(point.spot, "вход")
                headline = (
                    f"{street} · {_spot_word(point.spot)}: {active} или фолд — "
                    f"{point.best_action}{marker}"
                )
                numbers = (
                    f"    EV {active}а {_fmt_signed_bb(interval.point_bb)}, "
                    f"{_interval_words(point.spot, interval)}."
                )
                rows.append((headline, False))
                rows.append((numbers, False))
                rows.extend((line, True) for line in _prose_lines(prose.get(point.dp_index)))
                continue
            headline = (
                f"{street} · "
                f"{_spot_word(point.spot)}: {_action_word(point.action_taken)} "
                f"(лучше: {_action_word(point.best_action)}) — "
                f"{_fmt_bb(point.ev_diff_bb)}{marker}"
            )
            rows.append((headline, False))
            rows.extend((line, True) for line in _prose_lines(prose.get(point.dp_index)))

    if postflop:
        if res.ranked:
            rows.append(("", False))
        rows.extend((line, False) for line in postflop)

    if verdict is not None and verdict.summary.strip():
        rows.append(("", False))
        rows.append((verdict.summary.strip(), True))

    if not_checked:
        rows.append(("", False))
        rows.append((f"{_NOT_CHECKED_PREFIX} {', '.join(not_checked)}.", False))

    rows.append(("", False))
    zone_segment = "" if zone is None else f"зона: {_ZONE_WORD[zone]} · "
    status = f"⏱ {elapsed_s}с · {zone_segment}{_quota_line(quota_left, quota_total)}"
    rows.append((status, False))
    if dev_line is not None:
        rows.append((dev_line, False))

    buttons = [verdict_buttons(res.hand_no), *note_buttons_for_hand(res.hand_no, note_nicks)]
    if replay is None:
        return Msg(text="\n".join(text for text, _ in rows), buttons=buttons)
    return Msg(
        text=_fit_html(f"Что было\n{_replay_html(replay)}", rows, _TELEGRAM_TEXT_LIMIT),
        buttons=buttons,
        parse_mode="HTML",
    )


def range_image_title(point: PointVerdict) -> str:
    """Подпись НА картинке диапазона — тоже голос продукта, а не подпись из воркера.

    Называет ровно то, что нарисовано: чей это диапазон, к какому споту относится
    и какую долю всех рук занимает. Долю считает контракт (`Range.
    fraction_of_hands`), здесь она только печатается — второй формулы доли в
    продукте нет.

    Точка без допущения сюда не приходит: рисовать нечего (`worker.pipeline.
    _render_ranges`). Если всё же пришла, подпись честно говорит, что диапазон
    неизвестен, — вместо выдуманной доли.
    """
    if point.assumption is None:
        return f"{_spot_word(point.spot)}: диапазон оппонента не задан"
    share = 100.0 * point.assumption.range.fraction_of_hands()
    return (
        f"{_spot_word(point.spot)}: допущение о диапазоне оппонента — "
        f"{share:.1f}% всех рук"
    )


def escalation_msg(job_id: int, field: str, question: str, options: list[str]) -> Msg:
    """Эскалация валидатора — вопрос кнопками, не текстом (SESSIONS_UX): один тап."""
    return Msg(text=question, buttons=[escalation_buttons(job_id, field, options)])


def failed_msg(reason_public: str) -> Msg:
    """Разбор не удался — честная причина без внутренней кухни, без кнопок."""
    return Msg(text=f"Не получилось разобрать раздачу: {reason_public}")


def unknown_button_msg() -> Msg:
    """Ответ на нажатие, которое не разобрал ни один обработчик (round 5, Item G).

    Без ответа Телеграм крутит «часики» на кнопке, пока не свалится в ошибку —
    молчание, неотличимое от поломки. Текст короткий намеренно: он показывается
    всплывающим уведомлением callback-ответа, а у того жёсткий лимит около 200
    символов.
    """
    return Msg(text="Эта кнопка не работает.")


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
    """`/start`: что это и что сделать прямо сейчас — один шаг, без экрана настроек.

    «Основное действие — не кнопка» (SESSIONS_UX): первый экран объясняет ровно
    один шаг (прислать файл), а не показывает карту продукта. Про сессии здесь
    не сказано ни слова намеренно — их создание молчаливое, и заставлять новичка
    думать о них до первого результата значило бы отменить это решение.

    Нижнее меню приезжает вместе с этим сообщением (`menu`) — оно и есть «всё
    остальное» из того же правила: главное действие остаётся не кнопкой, а
    попасть в историю, лики и заметки больше неоткуда.
    """
    return Msg(
        text=(
            "Разбираю покерные раздачи с проверенным расчётом: точное считает код, "
            "словами объясняю отдельно.\n\n"
            "Пришлите скриншот стола — разберу раздачу. Или файл раздач (.txt) из "
            "PokerCraft — сделаю префлоп-скан турнира и покажу расхождения по цене. "
            "Под каждой раздачей будет кнопка «разобрать».\n\n"
            "/new — начать новую сессию."
        ),
        menu=MAIN_MENU,
    )


def hh_accepted_msg() -> Msg:
    """Файл принят — подтверждение приёма, ещё не результат.

    Ни числа рук, ни времени ожидания: ни того, ни другого бот в этот момент не
    знает (файл ещё не разобран), а называть их наугад запрещено (CLAUDE.md).
    """
    return Msg(text="Файл принят. Считаю префлоп-скан — пришлю сводку, когда закончу.")


def hh_duplicate_msg() -> Msg:
    """Тот же файл уже принят в эту сессию — считать второй раз незачем.

    Называет и путь дальше (`/new`): игрок, который ДЕЙСТВИТЕЛЬНО хочет разобрать
    тот же турнир заново, не должен упереться в тупик.
    """
    return Msg(
        text=(
            "Этот файл уже разбирается в текущей сессии — второй раз считать не буду. "
            "Нужен свежий разбор того же турнира — начните новую сессию: /new."
        )
    )


def bot_failure_msg() -> Msg:
    """Сбой на нашей стороне — короткое честное признание вместо молчания.

    Причина сюда не попадает никогда (ни `str(exc)`, ни путь файла, ни номер
    раздачи): она уходит в лог, как `jobs.error` у воркера. Игроку важно другое
    — что произошло не у него и что попытку имеет смысл повторить.
    """
    return Msg(text="Не получилось обработать запрос — это на нашей стороне. Попробуйте ещё раз.")


def unsupported_document_msg() -> Msg:
    """Документ ни на что не похож — отказ сразу, с называнием ОБЕИХ дверей.

    Дверей в продукт две, и текст обязан назвать обе: раздачи приходят файлом
    `.txt`, экран стола — картинкой (в том числе файлом, без сжатия). Пока текст
    предлагал только `.txt`, игрок, приславший скрин файлом, читал это как «зрения
    здесь нет» (`test_a_document_that_is_neither_hands_nor_a_picture_names_both_
    doors`).
    """
    return Msg(
        text=(
            "Такой файл я не разберу. Раздачи из PokerCraft присылайте файлом .txt "
            "— запущу скан. Скриншот стола — картинкой: обычной фотографией или "
            "файлом png, jpg, webp."
        )
    )


def screenshot_too_large_msg(limit_mb: int) -> Msg:
    """Картинка тяжелее предела — отказ до очереди, с названным пределом.

    Предел называется числом: «слишком большой файл» не говорит игроку, что
    делать, а «до N МБ» говорит. Отказ приходит сразу, потому что разбор занимает
    десятки секунд, и ждать их ради ошибки незачем.
    """
    return Msg(
        text=(
            f"Эта картинка слишком тяжёлая — я разбираю скрины до {limit_mb} МБ. "
            "Пришлите тот же экран обычной фотографией: со сжатием он станет легче."
        )
    )


def ask_gg_nickname_msg() -> Msg:
    """Разовый вопрос про ник в руме — без него скрин разобрать не на кого.

    Спрашивается ровно потому, что героя на экране определяет код, а не модель:
    подсветку и открытые карты экран рисует и победителю раздачи, и просить
    модель угадать «кто из них вы» значит получить уверенный неверный ответ.
    """
    return Msg(
        text=(
            "Чтобы разбирать скриншоты, мне нужен ваш ник в руме — тот, что "
            "написан у вашего места за столом. Пришлите его одним сообщением."
        )
    )


def gg_nickname_saved_msg(nickname: str) -> Msg:
    return Msg(text=f"Запомнил: {nickname}. Присылайте скриншот стола.")


def not_a_hand_msg(reason: str) -> Msg:
    """На экране не раздача — честный отказ, а не выдуманная из лобби рука.

    Причина показывается словами модели: она видела экран, а мы нет, и заменять
    её общей фразой значило бы отнять у игрока единственную подсказку, что
    именно прислать вместо этого.
    """
    return Msg(text=f"Это не похоже на раздачу: {reason}. Пришлите скриншот стола или руки.")


def hand_in_progress_msg() -> Msg:
    """Раздача на экране ещё идёт — отказ ДО разбора, а не пустой разбор после.

    Решение владельца 2026-09-09: скрин незавершённой руки не разбирается вовсе.

    Текст называет, что прислать вместо этого: отказ без следующего шага
    оставляет игрока с той же картинкой в руках.
    """
    return Msg(
        text=(
            "Похоже, раздача на этом скрине ещё идёт: не видно, чем она "
            "кончилась. Пришлите скриншот после того, как сходили и рука "
            "доиграла, — тогда будет что разбирать."
        )
    )


def send_as_file_msg() -> Msg:
    """Просьба переслать тот же скрин файлом — только по эскалации, не заранее.

    Сжатие Телеграма безопасно для чисел и опасно для мелких значков мастей, а
    файл втрое дороже в токенах (реестр, «Фото или файл»). Поэтому трение
    вводится там, где оно окупается, — после несошедшейся проверки карт, а не на
    каждой загрузке.
    """
    return Msg(
        text=(
            "Масти на этом скрине читаются плохо — Телеграм сжал картинку. "
            "Пришлите тот же скриншот ещё раз файлом: скрепка → «Файл» "
            "(на телефоне — «Документ»), тогда он придёт без сжатия."
        )
    )


def vision_manual_entry_msg(question: str) -> Msg:
    """Ввод числа вручную — вторая половина эскалации (спека §8.3)."""
    return Msg(text=f"{question}\nНапишите число одним сообщением — я подставлю его в разбор.")


def vision_answer_saved_msg() -> Msg:
    """Ответ принят: дальше снова говорит воркер, поэтому текст короткий."""
    return Msg(text="Принял, продолжаю разбор.")


def vision_answer_not_a_number_msg() -> Msg:
    return Msg(text="Это не похоже на число. Напишите только сумму, например 12.7.")


def vision_gave_up_msg() -> Msg:
    """Спрашивать больше нечего: расхождение не сошлось и после ответов игрока.

    Молчаливо разобрать такую руку нельзя — числа в ней спорные, и разбор поверх
    спорных чисел был бы уверенным выводом из неизвестного (CLAUDE.md).
    """
    return Msg(
        text=(
            "Не сходятся числа на этом скрине даже после уточнений — разбирать "
            "его я не возьмусь. Если раздача есть в выгрузке PokerCraft, "
            "пришлите файл: по тексту рума расчёт будет точным."
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


# --- отчёт по турниру (задача 23) --------------------------------------------------

# Потолки показа — тот же предел `sendMessage` в 4096 символов, что режет список
# расхождений выше, и та же цена отказа: `raise_for_status()` в отправителе,
# ретрай задачи, три пересчёта турнира вместо одного сообщения. Отчёт длиннее
# сводки (пять разделов вместо одного), поэтому потолки ниже; сумма всех
# разделов на максимуме проверена тестом
# (`test_tournament_report_msg_of_a_long_tournament_fits_one_telegram_message`).
_MAX_REPORT_LEVELS = 12
_MAX_REPORT_ALL_INS = 6
_MAX_REPORT_CHIP_MOVES = 8
_MAX_REPORT_FINDINGS = 4

# Строка, без которой таблица уровней читается как ошибка в счёте: стек на входе
# уровня МЕНЬШЕ, чем на выходе предыдущего, при тех же фишках. Объяснение
# кодовое, а не модельное — это факт («блайнды выросли»), а не суждение, и
# показывать его должен тот же голос, что печатает саму таблицу.
_BLIND_JUMP_LINE = (
    "Стек на входе уровня бывает меньше, чем на выходе предыдущего: фишки те же, "
    "выросли блайнды — в новых bb та же гора стоит меньше."
)

# Потолок текста рассказа по турниру: тот же предел `sendMessage` в 4096
# символов. Абзацы, которые в него не влезли, не выбрасываются молча — строка
# ниже говорит, сколько показано из скольких.
_MAX_STORY_CHARS = 3500

# Улица последнего действия героя — строчной буквой: она стоит внутри фразы
# («префлоп, олл-ин»), а не заголовком строки, как в разборе одной раздачи.
_STREET_WHERE: dict[Street, str] = {street: word.lower() for street, word in _STREET_WORD.items()}


def _hand_head(hand_no: str, hero_class: str, level: int) -> str:
    """«№TM123 · AKs · ур. 23» — общая шапка строк отчёта.

    Класс руки пропускается, если он неизвестен (карты героя в источнике не
    записаны): пустой сегмент оставил бы в строке две точки подряд, а выдумать
    вместо него что-либо нельзя
    (`test_tournament_report_msg_omits_a_hand_class_it_does_not_know`).
    """
    parts = [f"№{hand_no}", hero_class, f"ур. {level}"]
    return " · ".join(part for part in parts if part)


def _fmt_pct(value: float | None) -> str | None:
    """Доля одним знаком после запятой; `None` — доли нет, и печатать нечего.

    Возвращается `None`, а не «0.0%»: доля отсутствует ровно тогда, когда
    знаменатель нулевой (`PlayerStats`), и ноль процентов на этом месте был бы
    утверждением о том, чего не измеряли
    (`test_tournament_report_msg_does_not_print_a_missing_share_as_zero`).
    """
    return None if value is None else f"{value:.1f}%"


def _fmt_duration(minutes: int) -> str:
    """«1 ч 12 мин» для часа и дольше, «45 мин» — короче часа."""
    if minutes < 60:
        return f"{minutes} мин"
    return f"{minutes // 60} ч {minutes % 60:02d} мин"


def _levels_word(first: int, last: int) -> str:
    return f"{first}" if first == last else f"{first}–{last}"


def _share_line(title: str, share_pct: float | None, taken: int, chances: int) -> str:
    """Строка доли со счётчиками в скобках; при нулевом знаменателе — прямо об этом.

    Долю считает `PlayerStats` и передаёт сюда готовой — второй такой формулы в
    изложении нет и быть не должно. Счётчики показываются рядом с процентом
    намеренно: «8.3%» на двух десятках возможностей и на двух сотнях — разной
    силы утверждения, и отличить их можно только по знаменателю.
    """
    share = _fmt_pct(share_pct)
    if share is None:
        return f"{title}: таких развилок не было."
    return f"{title}: {share} ({taken} из {chances})."


def _stats_lines(report: TournamentReport) -> list[str]:
    """Статистика турнира и среднее игрока по всем его турнирам — или отказ сравнивать.

    При единственном турнире в базе среднее совпало бы с самим турниром, и
    сравнение было бы пустым: строка говорит об этом прямо, а не показывает два
    одинаковых числа (`TournamentReport.baseline`, бриф задачи).
    """
    stats = report.stats
    lines = [
        (
            f"Добровольный вход в банк: {_fmt_pct(stats.vpip_pct)}. "
            f"Повышение до флопа: {_fmt_pct(stats.pfr_pct)}."
        )
    ]
    baseline = report.baseline
    if baseline is None:
        lines.append("Сравнить не с чем: это первый турнир в базе.")
    else:
        lines.append(
            f"В среднем по всем турнирам (их {report.baseline_tournaments}): "
            f"вход {_fmt_pct(baseline.vpip_pct)}, повышение {_fmt_pct(baseline.pfr_pct)}."
        )
    lines.append(
        _share_line(
            "Ре-рейз до флопа", stats.reraise_pct, stats.reraise, stats.reraise_chances
        )
    )
    lines.append(
        _share_line(
            "Сдача на продолженную ставку",
            stats.fold_to_cbet_pct,
            stats.fold_to_cbet,
            stats.cbet_faced,
        )
    )
    return lines


def _trajectory_lines(trajectory: StackTrajectory) -> list[str]:
    """Стек по уровням и переломная точка.

    Обрезается по ПОСЛЕДНИМ уровням, а не по первым: обрезка нужна только очень
    длинному турниру, и в нём ближе к концу то, чем он кончился. Строка перелома
    печатается всегда — она и есть ответ на вопрос «где всё повернуло», и её
    уровень мог остаться за обрезкой.
    """
    shown = trajectory.levels[-_MAX_REPORT_LEVELS:]
    head = "Стек по уровням:"
    if len(shown) < len(trajectory.levels):
        head = (
            f"Стек по уровням (показаны последние {len(shown)} "
            f"из {len(trajectory.levels)}):"
        )
    lines = [head]
    lines += [
        f"Ур. {level.level}: {_bb_number(level.start_bb)} → {_fmt_bb(level.end_bb)}, "
        f"раздач: {level.hands}"
        for level in shown
    ]
    lines.append(
        f"Максимум: {_fmt_bb(trajectory.peak_bb)} на уровне {trajectory.peak_level} "
        f"(раздача №{trajectory.peak_hand_no}). После неё раздач: "
        f"{trajectory.hands_after_peak}, к концу турнира: {_fmt_bb(trajectory.final_bb)}."
    )
    if any(nxt.start_bb < prev.end_bb for prev, nxt in pairwise(shown)):
        lines.append(_BLIND_JUMP_LINE)
    return lines


def _all_in_lines(events: list[AllInEvent]) -> list[str]:
    """Олл-ины с исходом. Пустой список проговаривается, а не пропускается молча."""
    if not events:
        return ["Олл-инов в этом турнире не было."]
    shown = events[:_MAX_REPORT_ALL_INS]
    head = "Олл-ины:"
    if len(shown) < len(events):
        head = f"Олл-ины (показаны {len(shown)} самых крупных из {len(events)}):"
    return [head] + [
        f"{_hand_head(event.hand_no, event.hero_class, event.level)} · "
        f"вошёл с {_fmt_bb(event.stack_before_bb)} → {_fmt_signed_bb(event.delta_bb)}"
        f"{', вскрытие' if event.showdown else ''}"
        for event in shown
    ]


def _chip_move_lines(moves: list[ChipMove]) -> list[str]:
    """«Где ушли фишки»: раздача · что было · цена, дороже первой."""
    if not moves:
        return ["Раздач, в которых стек уменьшился, нет."]
    shown = moves[:_MAX_REPORT_CHIP_MOVES]
    head = "Где ушли фишки:"
    if len(shown) < len(moves):
        head = f"Где ушли фишки (показаны {len(shown)} самых дорогих из {len(moves)}):"
    lines = [head]
    for move in shown:
        what = _STREET_WHERE.get(move.last_street, move.last_street.value)
        if move.all_in:
            what += ", олл-ин"
        if move.showdown:
            what += ", вскрытие"
        lines.append(
            f"{_hand_head(move.hand_no, move.hero_class, move.level)} · {what} "
            f"— {_fmt_bb(move.cost_bb)}"
        )
    return lines


def _finding_lines(findings: list[Finding]) -> list[str]:
    """Находки — только по оценённым решениям, с ценой и историей игрока.

    Слово то же, что во всём модуле: «расхождение», не «ошибка», и «лучше», не
    «верно» — находка утверждает лишь, что у другой ветки была выше EV
    (`test_tournament_report_msg_never_calls_variance_a_mistake`).
    """
    if not findings:
        return ["Повторяющихся развилок среди оценённых решений не нашлось."]
    shown = findings[:_MAX_REPORT_FINDINGS]
    head = "Находки — только среди оценённых решений:"
    if len(shown) < len(findings):
        head = (
            f"Находки — только среди оценённых решений (показаны {len(shown)} "
            f"самых дорогих из {len(findings)}):"
        )
    lines = [head]
    for finding in shown:
        marker = f" ({_ASSUMING_MARKER})" if finding.zone is Zone.ASSUMING else ""
        lines.append(
            f"{_spot_word(finding.spot)}: {_action_word(finding.action_taken)} "
            f"(лучше: {_action_word(finding.best_action)}) — повторов: {finding.count}, "
            f"суммарно {_fmt_bb(finding.total_cost_bb)}{marker}"
        )
        if finding.seen_before:
            lines.append(
                f"    та же развилка в прошлых турнирах — точек: {finding.seen_before}, "
                f"турниров: {finding.seen_before_tournaments}"
            )
    return lines


def _ev_lines(ev: EvSplit) -> list[str]:
    """Честный счёт: цена расхождений, дисперсия и несудимое — тремя разными строками.

    Ни одно из чисел не подписано как «цена ошибок», и ни одно не складывается с
    соседним: EV расхождения посчитан против диапазона на момент решения, фишки
    — по факту раздачи. Последняя строка говорит это прямо, чтобы читатель не
    сложил их сам.
    """
    return [
        "Сколько это стоило:",
        f"Оценено решений: {ev.points_judged} из {ev.points_total}.",
        f"Цена расхождений в оценённых решениях: {_fmt_bb(ev.judged_loss_bb)}.",
        f"Фишки в раздачах с расхождением: {_fmt_bb(ev.chips_in_gap_hands_bb)}.",
        (
            f"Фишки в проигранных олл-инах без расхождения: "
            f"{_fmt_bb(ev.chips_in_lost_allins_bb)} — эти решения расчёт не оспаривает."
        ),
        (
            f"Фишки в остальных раздачах: {_fmt_bb(ev.chips_elsewhere_bb)} — про них "
            f"расчёт не говорит ничего."
        ),
        "Цена расхождений и потерянные фишки — разные величины: складывать их нельзя.",
    ]


def tournament_report_msg(report: TournamentReport) -> Msg:
    """Отчёт по турниру целиком: что случилось, где ушли фишки, находки, сколько стоило.

    Кнопок нет намеренно: «разобрать» стоит под пунктами сводки скана
    (`scan_summary_msg`), которая приходит следующим сообщением, и вторая копия
    тех же кнопок раздвоила бы одно действие на два места.

    Каждый список обрезан своим потолком и, если обрезан, говорит об этом
    прямо — молча показать восемь строк из ста было бы той же деградацией без
    огласки, что и умолчанный `hands_failed`.
    """
    lines = [
        (
            f"Турнир. Раздач: {report.hands_total}. "
            f"Уровни: {_levels_word(report.first_level, report.last_level)}. "
            f"Время: {_fmt_duration(report.duration_minutes)}."
        )
    ]
    if report.hands_failed:
        lines.append(f"Раздач не разобрано: {report.hands_failed} — они не вошли в оценку.")
    lines.append("")
    lines += _stats_lines(report)
    lines.append("")
    lines += _trajectory_lines(report.trajectory)
    lines.append("")
    lines += _all_in_lines(report.all_ins)
    lines.append("")
    lines += _chip_move_lines(report.chip_moves)
    lines.append("")
    lines += _finding_lines(report.findings)
    lines.append("")
    lines += _ev_lines(report.ev)
    return Msg(text="\n".join(lines))


def tournament_story_msg(narrative: TournamentTextOut) -> Msg:
    """Рассказ по турниру словами — отдельным сообщением ПЕРЕД отчётом с числами.

    Отдельным, а не абзацем внутри отчёта, по одной причине: вместе они не
    помещаются в `sendMessage` (4096 символов), и урезать пришлось бы либо
    числа, либо текст. Порядок «сначала рассказ, потом числа» — тот же, что у
    пары «отчёт → сводка со сводными кнопками»: кнопки должны остаться под
    последним сообщением.

    Абзацы, не влезшие в потолок, не пропадают молча — сообщение говорит,
    сколько абзацев показано из скольких
    (`test_tournament_story_msg_says_when_it_had_to_cut`).
    """
    shown: list[str] = []
    length = 0
    for paragraph in narrative.paragraphs:
        cleaned = paragraph.strip()
        if not cleaned:
            continue
        if length + len(cleaned) + 2 > _MAX_STORY_CHARS and shown:
            break
        shown.append(cleaned)
        length += len(cleaned) + 2

    total = len([p for p in narrative.paragraphs if p.strip()])
    if len(shown) < total:
        count = len(shown)
        verb = _plural_form(count, "Показан", "Показаны", "Показано")
        noun = _plural_form(count, "абзац", "абзаца", "абзацев")
        shown.append(f"{verb} {count} {noun} из {total} — текст не поместился целиком.")
    return Msg(text="\n\n".join(shown))


# --- экраны нижнего меню (задача 23) -------------------------------------------------

# Жёсткий предел `sendMessage`: сообщение длиннее 4096 символов Телеграм не
# отправляет вовсе. Для экрана «Заметки» цена отказа выше, чем для сводок: на
# этом же экране живут кнопки правки, цвета и удаления, и он единственный, откуда
# слишком длинную заметку можно убрать — не открывшись, он не оставляет выхода.
_TELEGRAM_TEXT_LIMIT = 4096

# Верхняя граница числа заметок на экране. Сколько поместится на самом деле,
# решает бюджет `_TELEGRAM_TEXT_LIMIT` в `notes_msg`
# (`test_notes_msg_of_the_longest_notes_still_fits_one_telegram_message`); это
# число только не даёт вырасти ряду кнопок под каждой заметкой.
_MAX_RENDERED_NOTES = 10

# Строка честности экрана «Мои лики». Список считается по вердиктам ядра, а те
# судят решение против диапазона, а не против вскрытой карты (CLAUDE.md): без
# этой строки игрок вправе прочесть список как «вот из-за чего я проиграл».
_LEAKS_DISCLAIMER = (
    "Считаю по решениям, а не по исходам: раздача, проигранная по случайности, "
    "в этот список не попадает."
)

# Что означает пустое покрытие. Судятся сегодня только префлоп-споты пуш-фолда
# (`JUDGED_SPOTS`), и молчание об этом превратило бы короткий список ликов в
# заявление «в остальном всё хорошо».
_COVERAGE_NOTE = "Остальные решения расчёт пока не судит — про них он не говорит ничего."


def _coverage_line(judged: int, total: int, scope: str) -> str:
    """Покрытие в точках решения. `scope` называет, за что оно посчитано.

    Область обязана быть названа: одна и та же строка печатается на экране
    ликов (вся история) и в сводке вечера (одна сессия), и «за всю историю» под
    числами одного вечера было бы неверным утверждением о чужом множестве.
    """
    return f"Оценено решений: {judged} из {total} {scope}."


def _times_word(count: int) -> str:
    return _plural_form(count, "раз", "раза", "раз")


def _leak_line(stat: LeakStat) -> str:
    """Строка типа лика: подпись, частота, цена. Ровно то, что просил владелец."""
    return (
        f"{stat.rule.title} — {stat.count} {_times_word(stat.count)}, "
        f"{_fmt_bb(-stat.loss_bb)}"
    )


def leaks_msg(overview: LeaksOverview) -> Msg:
    """Экран «Мои лики»: покрытие сверху, типы ликов по цене под ним.

    Группировка по ТИПУ ЛИКА (решение владельца 2026-09-07), а не по споту:
    «не шовит, где надо» и «шовит слишком широко» — противоположные привычки,
    и в одной строке они были бы бессмысленны. Точки «около нуля» и точки без
    вердикта сюда не попадают — их отсеивает сама таблица правил
    (`memory.repos.LeaksRepo`, `contracts.history`).

    Покрытие печатается ВСЕГДА и первым, как в сводке скана: короткий список
    ликов без него читается как «в остальном сыграно чисто».
    """
    if overview.points_total == 0:
        return Msg(
            text=(
                "Пока не из чего считать: разобранных раздач нет.\n\n"
                "Пришлите скриншот стола или файл раздач (.txt) из PokerCraft — "
                "лики копятся по всей истории разборов."
            )
        )
    lines = [
        "Мои лики — по всей истории разборов.",
        "",
        _coverage_line(overview.points_judged, overview.points_total, "за всю историю"),
    ]
    if overview.leaks:
        lines.append("")
        lines.extend(_leak_line(stat) for stat in overview.leaks)
    else:
        lines.append("")
        lines.append("Среди оценённых решений повторяющихся расхождений не нашлось.")
    lines.append("")
    lines.append(_LEAKS_DISCLAIMER)
    lines.append(_COVERAGE_NOTE)
    return Msg(text="\n".join(lines))


def sessions_msg(sessions: Sequence[SessionLine], total: int) -> Msg:
    """Экран «Сессии»: список вечеров, сводка — по нажатию на вечер.

    Сводка считается ПО ЗАПРОСУ (решение владельца 2026-09-07), поэтому в
    списке ни рук, ни цены: строка называет вечер и говорит, идёт ли он.
    Кнопка «Начать новую» — та же логика, что `/new` (SESSIONS_UX).

    `total` — сколько вечеров у игрока всего (`SessionsRepo.count_for_player`).
    Список приходит с потолком запроса, и обрезка называется вслух: молчать о
    ней этот файл запрещает себе дважды — в `scan_summary_msg` и в
    `tournament_report_msg`
    (`test_sessions_msg_says_when_the_history_did_not_fit`).
    """
    if not sessions:
        return Msg(
            text=(
                "Сессий пока нет. Сессия — это вечер игры: она откроется сама, "
                "как только пришлёте скриншот раздачи или файл из PokerCraft."
            ),
            buttons=session_buttons([]),
        )
    lines = ["Сессии — по одной на вечер игры.", ""]
    for line in sessions:
        lines.append(f"{line.title}{' · сейчас идёт' if line.is_active else ''}")
    lines.append("")
    if len(sessions) < total:
        lines.append(f"Показаны {len(sessions)} из {total} — самые свежие.")
    lines.append("Нажмите на сессию — покажу сводку вечера.")
    return Msg(
        text="\n".join(lines),
        buttons=session_buttons([(line.session_id, line.title) for line in sessions]),
    )


def session_summary_msg(summary: SessionSummary) -> Msg:
    """Сводка вечера: турниры, разобранные руки, цена расхождений, лик вечера.

    Число потери подписано теми же словами, что и в сводке скана («суммарная
    потеря в оценённых решениях»): это одна и та же величина, посчитанная по
    судимым точкам (`memory.repos.SessionsRepo._session_loss_bb`), и две разные
    подписи читались бы как два разных числа. Множество названо в подписи по
    той же причине, что и там: точки без вердикта в сумму не входят, а строка
    покрытия рядом говорит, сколько их
    (`test_session_summary_msg_scopes_the_loss_to_the_judged_points`).

    Строка лика подписана ЦЕНОЙ, а не частотой: `summary.top_leak` — первый
    элемент списка, который `LeaksRepo.by_type` упорядочивает по цене
    (`test_session_summary_msg_signs_the_leak_by_its_price_not_its_frequency`).
    """
    lines = [summary.title, ""]
    if summary.hands == 0:
        lines.append("За этот вечер ещё ничего не разобрано.")
        return Msg(text="\n".join(lines))
    lines.append(
        f"Турниров: {summary.tournaments} · разобрано раздач: {summary.hands}."
    )
    lines.append(_coverage_line(summary.points_judged, summary.points_total, "за этот вечер"))
    lines.append(
        f"Суммарная потеря в оценённых решениях: {_fmt_bb(-summary.loss_bb)}."
    )
    lines.append("")
    if summary.top_leak is None:
        lines.append("Повторяющегося расхождения за этот вечер расчёт не нашёл.")
    else:
        lines.append(f"Дороже всего за вечер: {_leak_line(summary.top_leak)}.")
    return Msg(text="\n".join(lines))


def _fitted(text: str, budget: int) -> str:
    """Текст не длиннее `budget` символов; обрезанный говорит об этом прямо.

    Заметка длиннее `MAX_NOTE_TEXT_CHARS` записаться не может, поэтому обрезка
    здесь — страховка на случай строки, записанной до этого предела; молчаливой
    она быть не вправе (`test_notes_msg_says_when_a_note_did_not_fit_whole`).
    """
    if len(text) <= budget:
        return text
    return text[: max(0, budget - len(_MARKER))] + _MARKER


def _note_lines(note: NoteRecord) -> list[str]:
    label = next(
        (color.label for color in NOTE_COLORS if color.key == note.color), note.color
    )
    return [f"{label} · {note.nick}", _fitted(note.text, MAX_NOTE_TEXT_CHARS)]


def _notes_cut_line(shown: int, total: int) -> str:
    return f"Показаны {shown} из {total} — самые свежие."


def notes_msg(notes: Sequence[NoteRecord], total: int) -> Msg:
    """Экран «Заметки»: наблюдения об оппонентах, свежие первыми.

    Экран показывает и правит, но НЕ заводит новых: заметка ценна скоростью
    записи в момент наблюдения, поэтому путь «добавить» начинается из разбора
    руки с уже подставленным оппонентом (решение владельца 2026-09-04). Об этом
    прямо сказано текстом — иначе экран выглядел бы сломанным.

    `total` — сколько заметок у игрока ВСЕГО (`NotesRepo.count_for_player`), а
    не длина `notes`: список приходит уже с потолком запроса, и знаменатель по
    нему называл бы размер страницы вечером игрока
    (`test_notes_msg_counts_the_notes_a_player_has_not_the_page_it_was_given`).

    Список режется бюджетом `_TELEGRAM_TEXT_LIMIT`, а не только числом заметок:
    десять заметок по 400 знаков не помещаются в `sendMessage`, и экран
    переставал открываться вместе с кнопкой удаления той заметки, из-за которой
    он и не открывался
    (`test_notes_msg_of_the_longest_notes_still_fits_one_telegram_message`).
    """
    head = "Заметки на оппонентов — то, чего не показывает HUD."
    tail = (
        "Новая заметка начинается из разбора раздачи: под вердиктом есть кнопка "
        "с ником оппонента. Заметки живут только на скринах — в файлах PokerCraft "
        "ники обезличены."
    )
    if not notes:
        return Msg(text=f"{head}\n\nПока пусто.\n\n{tail}")

    shown: list[NoteRecord] = []
    body: list[str] = []
    # Хвост и строка обрезки печатаются ПОСЛЕ списка, поэтому место под них
    # занимается до него: посчитать их постфактум значило бы вылезти за предел
    # ровно тогда, когда список и так пришлось резать. Длина строки обрезки
    # берётся по её же шаблону при самом длинном возможном числе показанных.
    length = len(head) + len(tail) + len(_notes_cut_line(_MAX_RENDERED_NOTES, total)) + 4
    for note in notes[:_MAX_RENDERED_NOTES]:
        block = [*_note_lines(note), ""]
        cost = sum(len(line) + 1 for line in block)
        if shown and length + cost > _TELEGRAM_TEXT_LIMIT:
            break
        shown.append(note)
        body.extend(block)
        length += cost

    lines = [head, "", *body]
    if len(shown) < total:
        lines.append(_notes_cut_line(len(shown), total))
        lines.append("")
    lines.append(tail)
    return Msg(
        text="\n".join(lines), buttons=[note_row(note.note_id) for note in shown]
    )


def note_prompt_msg(nick: str, existing: NoteRecord | None = None) -> Msg:
    """Просьба написать наблюдение об оппоненте — вход FSM заметки.

    Примеры в тексте — из решения владельца 2026-09-04 дословно: заметка
    фиксирует то, чего не выводится из счётчиков.

    Прежний текст показывается в бюджете `_TELEGRAM_TEXT_LIMIT`: экран правки —
    единственный способ заменить слишком длинную заметку, и упереться в предел
    `sendMessage` он не вправе
    (`test_note_prompt_msg_of_a_long_note_still_fits_one_telegram_message`).
    """
    head = f"Заметка на {nick}."
    ask = (
        "Напишите наблюдение одним сообщением — то, чего не покажет HUD: "
        "«фолдит на опен», «донкает флоп». Новый текст заменит прежний."
    )
    lines = [head]
    if existing is not None:
        shown_now = "Сейчас записано: "
        budget = _TELEGRAM_TEXT_LIMIT - len(head) - len(ask) - len(shown_now) - 4
        lines.append(f"{shown_now}{_fitted(existing.text, budget)}")
    lines.append(ask)
    return Msg(text="\n".join(lines))


def note_saved_msg(nick: str) -> Msg:
    return Msg(text=f"Записал заметку на {nick}.")


def note_deleted_msg(nick: str) -> Msg:
    return Msg(text=f"Удалил заметку на {nick}.")


def note_color_prompt_msg(note: NoteRecord) -> Msg:
    """Выбор цветового архетипа — вторая половина двухслойной разметки заметок."""
    return Msg(
        text=f"Цвет заметки на {note.nick}: выберите архетип.",
        buttons=note_color_buttons(
            note.note_id, [(color.key, color.label) for color in NOTE_COLORS]
        ),
    )


def note_color_saved_msg(nick: str, label: str) -> Msg:
    return Msg(text=f"{nick} — {label}.")


def note_too_long_msg(limit: int) -> Msg:
    """Заметка длиннее потолка: честный отказ вместо тихого обрезания.

    Обрезать нельзя по той же причине, что и ник: игрок увидел бы на экране не
    то, что написал. Потолок держит экран «Заметки» открываемым — а он
    единственное место, откуда заметку можно поправить или убрать.
    """
    return Msg(
        text=(
            f"Слишком длинная заметка: принимаю не длиннее {limit} символов. "
            f"Заметка — одно наблюдение об оппоненте; пришлите короче."
        )
    )


def note_gone_msg() -> Msg:
    """Кнопка нажата, а заметки уже нет (удалена соседним нажатием)."""
    return Msg(text="Этой заметки больше нет.")


def settings_msg(
    nickname: str | None, quota_left: int, quota_total: int, *, is_dev: bool = False
) -> Msg:
    """Экран «Настройки»: ник в руме, остаток дневного лимита, о боте.

    Ник вводится отсюда — явной кнопкой (решение владельца 2026-09-07). Раньше
    ником становилось первое же текстовое сообщение игрока, и случайная реплика
    молча оказывалась в профиле.

    Остаток квоты — те же числа и та же формулировка, что в подписи под
    разбором (`_quota_line`): второго счётчика в продукте нет.
    """
    lines = ["Настройки", ""]
    if nickname:
        lines.append(f"Ник в руме: {nickname} — по нему я нахожу вас за столом на скриншоте.")
    else:
        lines.append(
            "Ник в руме не задан. Без него скриншот разбирать не на кого: "
            "героя за столом определяет код, а не модель."
        )
    lines.append(f"Доступно: {_quota_line(quota_left, quota_total)}.")
    lines.append("")
    lines.append(
        "О боте: разбираю турнирные раздачи проверенным расчётом — точное считает "
        "код, словами объясняю отдельно."
    )
    if is_dev:
        lines.append("")
        lines.append("Режим разработчика включён. /invite — выпустить инвайт-код.")
    return Msg(text="\n".join(lines), buttons=[[set_nickname_button(bool(nickname))]])


def help_msg() -> Msg:
    """Экран «Help»: что прислать и что делает каждая кнопка нижнего меню.

    Несёт нижнее меню (`menu`) — это ещё и способ вернуть клавиатуру тому, кто
    её свернул: инлайн-кнопок здесь нет, и место `reply_markup` свободно.
    """
    return Msg(
        text=(
            "Как этим пользоваться\n\n"
            "Пришлите скриншот стола — разберу раздачу и покажу цену решений.\n"
            "Пришлите файл раздач (.txt) из PokerCraft — сделаю префлоп-скан турнира.\n\n"
            "Кнопки внизу:\n"
            f"{MENU_SESSIONS} — вечера игры и сводка по каждому.\n"
            f"{MENU_TOURNAMENT} — как прислать файл раздач.\n"
            f"{MENU_LEAKS} — типовые расхождения по всей истории.\n"
            f"{MENU_NOTES} — наблюдения об оппонентах.\n"
            f"{MENU_SETTINGS} — ник в руме и остаток дневного лимита.\n\n"
            "/new — начать новую сессию.\n"
            "/nick — указать ник в руме.\n"
            "/alias — назвать участника разбора ником в руме.\n"
            "/ask ВОПРОС — спросить о своей игре; отвечаю только посчитанным."
        ),
        menu=MAIN_MENU,
    )


def hh_prompt_msg() -> Msg:
    """Экран «Турнир (HH)»: откуда взять файл и что с ним будет.

    Отдельным экраном, а не одной строкой в help: путь до выгрузки в GG не
    очевиден, а без файла HH-вход недоступен вовсе.
    """
    return Msg(
        text=(
            "Разбор турнира по файлу раздач\n\n"
            "В GG откройте PokerCraft → «История рук» → выберите турнир → "
            "скачайте историю раздач (.txt) и пришлите файл сюда.\n\n"
            "Я сделаю префлоп-скан всего турнира и пришлю сводку: расхождения по "
            "цене, под каждым — кнопка «разобрать». Скан дневной лимит не тратит."
        )
    )


def range_photos(paths: Sequence[str], res: AnalysisResult) -> list[Photo]:
    """Картинки диапазонов с подписями — то, что уходит игроку по кнопке «Диапазоны».

    Порядок путей задан рисовальщиком (`worker.pipeline._render_ranges`): он
    идёт по `res.ranked` и пропускает точки без допущения. Здесь тот же обход
    повторён, чтобы подпись досталась своей картинке; лишние пути (сохранённые
    старой версией разбора) остаются без подписи, а не получают чужую
    (`test_range_photos_do_not_borrow_a_caption_from_another_point`).
    """
    points = [
        res.points[index]
        for index in res.ranked
        if res.points[index].assumption is not None
    ]
    photos: list[Photo] = []
    for position, path in enumerate(paths):
        caption = range_image_title(points[position]) if position < len(points) else ""
        photos.append(Photo(path=path, caption=caption))
    return photos


def ranges_msg(paths: Sequence[str], res: AnalysisResult) -> Msg:
    """Ответ на кнопку «Диапазоны»: картинки матриц либо честное «их нет».

    Диапазон рисуется только там, где вывод опирается на допущение о поле
    (зона «предполагая»). У строгой точки показывать нечего, и картинка
    подразумевала бы обратное — поэтому отказ называет причину, а не молчит.
    """
    photos = range_photos(paths, res)
    if not photos:
        return Msg(
            text=(
                "По этой раздаче диапазоны не рисуются: вывод не опирается на "
                "догадку о диапазоне оппонента."
            )
        )
    return Msg(text=f"Диапазоны к раздаче {res.hand_no}:", photos=photos)


def _html_escape(text: str) -> str:
    """Экранирование для `parse_mode=HTML` — только три обязательных символа.

    Телеграм требует экранировать `<`, `>` и `&`; ник оппонента и подписи карт
    приходят из внешнего мира, и незакрытый `<` уронил бы отправку сообщения
    целиком (`test_replay_msg_escapes_a_nickname_that_looks_like_a_tag`).
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _replay_html(replay: HandReplay) -> str:
    """Ход раздачи разметкой Телеграма: точка решения героя — жирным.

    Спека §5.6 требует выделить точку решения прямо в потоке действий;
    `explanation.hand_replay` отдаёт куски с флагом `emphasis`, а во что
    превратится выделение — решает этот модуль. Здесь это `<b>` при
    `parse_mode=HTML`, поэтому весь остальной текст экранируется
    (`_html_escape`). Одна разметка на оба места, где реплей показывается, —
    блоком в разборе (`deep_dive_msg`) и отдельным сообщением (`replay_msg`).
    """
    return "".join(
        f"<b>{_html_escape(span.text)}</b>" if span.emphasis else _html_escape(span.text)
        for span in replay.spans
    )


def replay_msg(replay: HandReplay, hand_no: str) -> Msg:
    """Ответ на кнопку «Подробнее»: ход раздачи отдельным сообщением."""
    return Msg(
        text=f"Ход раздачи {_html_escape(hand_no)}\n\n{_replay_html(replay)}",
        parse_mode="HTML",
    )


def replay_unavailable_msg() -> Msg:
    """Кнопка «Подробнее» нажата под раздачей, хода которой у нас нет.

    Так бывает у старых разборов: реплей строится из `hands.enriched`, а
    сохранённая рука могла остаться на чекпоинте ниже (спека §8.2).
    """
    return Msg(text="Хода этой раздачи у меня не сохранилось — показать нечего.")


def disagreement_saved_msg() -> Msg:
    """«Не согласен» принято: возражение уходит в eval-датасет (EVALS, этаж 4).

    Самая ценная кнопка продукта (SESSIONS_UX), поэтому ответ говорит, что
    именно произошло с нажатием, а не просто «спасибо».
    """
    return Msg(
        text=(
            "Записал возражение. Разбор от этого не меняется — но эта раздача "
            "уходит в набор, на котором мы проверяем расчёт."
        )
    )


def invite_required_msg() -> Msg:
    """Вход закрыт: без инвайта — вежливый отказ, а не молчание.

    Отказ называет, чего не хватает, и не намекает, что код можно подобрать:
    подбирать нечего, коды случайны (`memory.repos.InvitesRepo`).
    """
    return Msg(
        text=(
            "Пока я работаю по приглашениям. Если у вас есть код — пришлите "
            "команду /start и код одной строкой: /start ВАШ_КОД."
        )
    )


def invite_accepted_msg() -> Msg:
    """Код принят: приветствие и тот же первый экран, что у `/start`.

    Текст первого экрана берётся у `start_msg`, а не пишется второй раз: два
    приветствия разошлись бы при первой же правке одного из них.
    """
    start = start_msg()
    return Msg(text=f"Код принят — добро пожаловать.\n\n{start.text}", menu=start.menu)


def owner_admitted_msg() -> Msg:
    """Владельца впустило окружение сервера, а не код: первый экран плюс `/invite`.

    Отдельно от `invite_accepted_msg` затем, что тот начинается словами «Код
    принят», а кода на этом пути не было (`test_owner_admitted_msg_does_not_claim_
    a_code_was_used`). Текст первого экрана — у `start_msg`, как и там: два
    приветствия разошлись бы при первой же правке одного из них.

    Единственное, что здесь сказано сверх приветствия, — команда выдачи кодов:
    её видит только владелец, и ровно она превращает первый вход в возможность
    впустить остальных.
    """
    start = start_msg()
    return Msg(
        text=(
            "Вы вошли как владелец — код не потребовался.\n\n"
            f"{start.text}\n\n"
            "/invite — выпустить код для гостя."
        ),
        menu=start.menu,
    )


def gg_nickname_too_long_msg(limit: int) -> Msg:
    """Ник длиннее колонки в БД: честный отказ вместо тихого обрезания.

    Обрезанный ник не совпал бы с ником на экране, и опознание героя кодом
    перестало бы работать — молча и не в том месте, где ошиблись.
    """
    return Msg(text=f"Слишком длинный ник: принимаю не длиннее {limit} символов. Пришлите ещё раз.")


def session_unavailable_msg() -> Msg:
    """Нажали на сессию, которой у игрока нет: номер приехал из внешнего мира."""
    return Msg(text="Такой сессии у вас нет.")


def analysis_unavailable_msg() -> Msg:
    """Кнопка под разбором нажата, а самого разбора у нас не осталось."""
    return Msg(text="Разбора этой раздачи у меня не сохранилось.")


def invite_created_msg(code: str) -> Msg:
    """Выпущенный код — владельцу. Ссылку не собираем: имени бота модуль не знает."""
    return Msg(
        text=(
            f"Инвайт-код: {code}\n\n"
            f"Приглашённый начинает так: /start {code}"
        )
    )


def unknown_text_msg() -> Msg:
    """Текст, который ничего не значит: ни меню, ни ответ на вопрос бота.

    До задачи 23 такое сообщение молча становилось ником в руме. Молчание было
    бы вторым плохим ответом: игрок не знает, услышали ли его.
    """
    return Msg(
        text=(
            "Не понял. Пришлите скриншот раздачи или файл из PokerCraft — "
            "остальное в нижнем меню."
        )
    )


# --- псевдонимы: оппонент по нику и его метки в турнирах ----------------------------


def _alias_line(opponent: OpponentRecord) -> str:
    """Оппонент, число его турниров и — со второго — чем это число держится.

    Пометка стоит ровно там, где число перестаёт быть счётом одного турнира:
    два турнира и больше сложены СЛОВОМ владельца, а не данными. Ошибочная
    сшивка приписывает чужие раздачи одному человеку и оставляет неполным
    другого, а на экране выглядит как выросшая выборка
    (`test_aliases_msg_marks_the_numbers_that_stand_on_a_stitching`).
    """
    line = f"{opponent.nick} — турниров: {opponent.links}"
    return f"{line} · по вашей сшивке" if opponent.links > 1 else line


def _aliases_cut_line(shown: int, total: int) -> str:
    return f"Показаны {shown} из {total} — по алфавиту."


def aliases_msg(aliases: Sequence[OpponentRecord]) -> Msg:
    """Список оппонентов, которых игрок назвал по нику, и сколько турниров у каждого.

    Число турниров — единственный признак, что привязка состоялась: в файлах
    раздач участник обезличен, и увидеть за столом его ник негде.

    Список приходит целиком (`OpponentsRepo.list_for_player` без потолка), поэтому
    знаменатель строки обрезки — сколько оппонентов у игрока НА САМОМ ДЕЛЕ, а не
    размер страницы (`test_aliases_msg_counts_everyone_not_only_the_shown`).
    Режется бюджетом `_TELEGRAM_TEXT_LIMIT`: список длиннее одного сообщения
    Телеграма не отправился бы вовсе.
    """
    head = "Оппоненты, которых вы назвали по нику."
    tail = (
        "Привязать участника последнего разбора: /alias МЕСТО НИК — "
        "например /alias BTN Vasya. В файлах PokerCraft участники обезличены, "
        "и кто из них кто, знаете только вы."
    )
    if not aliases:
        return Msg(text=f"{head}\n\nПока пусто.\n\n{tail}")

    shown: list[str] = []
    length = len(head) + len(tail) + len(_aliases_cut_line(len(aliases), len(aliases))) + 4
    for alias in aliases:
        line = _alias_line(alias)
        if shown and length + len(line) + 1 > _TELEGRAM_TEXT_LIMIT:
            break
        shown.append(line)
        length += len(line) + 1

    lines = [head, "", *shown, ""]
    if len(shown) < len(aliases):
        lines.append(_aliases_cut_line(len(shown), len(aliases)))
        lines.append("")
    lines.append(tail)
    return Msg(text="\n".join(lines))


def alias_usage_msg() -> Msg:
    """Команда без второго слова: непонятно, кого и как звать."""
    return Msg(
        text=(
            "Нужно два слова: место за столом и ник в руме. "
            "Например: /alias BTN Vasya. Один /alias без слов покажет список."
        )
    )


def alias_bound_msg(nick: str, position: str) -> Msg:
    return Msg(text=f"Запомнил: {position} в последнем разборе — это {nick}.")


def alias_taken_msg(position: str, nick: str) -> Msg:
    """Место уже названо другим ником: молча переписать значило бы потерять сказанное.

    Ник в ответе — тот, что записан сейчас: без него игрок не знает, с чем
    именно спорит его команда.
    """
    return Msg(text=f"{position} в этом турнире уже записан как {nick}.")


def alias_tournament_taken_msg(nick: str, position: str) -> Msg:
    """У ника в этом турнире уже есть другое место.

    В турнире у участника один идентификатор, поэтому второе место того же ника
    в том же турнире означало бы, что в его статистику сложены двое.
    """
    return Msg(text=f"{nick} в этом турнире уже записан на другом месте: {position}.")


def alias_position_unknown_msg(positions: Sequence[str]) -> Msg:
    """Такого места в последнем разборе нет — перечисляем те, что есть.

    Гадать, кого имел в виду игрок, нельзя: связь участников утверждает он, и
    подставленное за него место записало бы статистику на чужого.
    """
    return Msg(
        text=(
            "Такого места в последнем разборе нет. Есть: "
            f"{', '.join(positions)}."
        )
    )


def alias_no_analysis_msg() -> Msg:
    """Привязывать не к чему: разборов у игрока ещё не было."""
    return Msg(
        text=(
            "Пока не к чему привязывать: разберите раздачу из файла PokerCraft, "
            "и участников этой раздачи можно будет назвать по нику."
        )
    )


def alias_screenshot_only_msg() -> Msg:
    """Последний разбор — скриншот: у него нет номера турнира комнаты.

    Привязка держится на номере турнира, а на скрине его нет. На скрине ники
    оппонентов видны и так — там работают заметки.
    """
    return Msg(
        text=(
            "Последний разбор — со скриншота, а у него нет номера турнира. "
            "Привязка нужна для файлов PokerCraft, где участники обезличены."
        )
    )


# --- Ответ на вопрос игрока ---------------------------------------------------------

# Имя расчёта словами игрока. Второй словарь рядом с `_CALC_BRIEF`
# (`explanation/question.py`) — намеренно: тот описывает величину модели,
# которая пишет прозу сама, а этот печатается игроку и подчиняется правилу
# единого голоса. То же решение и та же причина, что у пары
# `_SPOT_BRIEF`/`_SPOT_WORD`.
_CALC_WORD: dict[CalcName, str] = {
    CalcName.HERO_FREQUENCY: "ваша частота",
    CalcName.OPPONENT_FREQUENCY: "частота оппонента",
    CalcName.COVERAGE: "покрытие разбора и цена расхождений",
    CalcName.LEAKS: "типы расхождений",
    CalcName.DEFENSE_FREQUENCY: "требуемая частота защиты",
    CalcName.FREQUENCY_VS_THRESHOLD: "частота против порога",
}

_STAT_WORD: dict[FrequencyStat, str] = {
    FrequencyStat.VPIP: "добровольный вход в банк",
    FrequencyStat.PFR: "повышение до флопа",
    FrequencyStat.RERAISE: "ре-рейз до флопа",
    FrequencyStat.FOLD_TO_CBET: "сдача на продолженную ставку",
    FrequencyStat.CBET_FLOP: "продолженная ставка на флопе",
    FrequencyStat.BARREL_TURN: "второй баррель (ставка на тёрне)",
    FrequencyStat.BARREL_RIVER: "третий баррель (ставка на ривере)",
    FrequencyStat.SHOWDOWN: "доход до вскрытия",
}

_SUBJECT_WORD: dict[Subject, str] = {
    Subject.HERO: "у вас",
    Subject.OPPONENT: "у него",
    Subject.FIELD: "у поля",
}

_OUTCOME_WORD: dict[ThresholdOutcome, str] = {
    ThresholdOutcome.ABOVE: "Ваша величина выше порога.",
    ThresholdOutcome.BELOW: "Ваша величина ниже порога.",
    ThresholdOutcome.UNDECIDED: "",
}

_WINDOW_WORD = {True: "за этот вечер", False: "за всю историю разборов"}


def _measured(measurement: Measurement) -> str:
    """Величина со своим знаменателем — требование владельца, безусловное.

    Доля печатается только при ненулевом знаменателе: «0 из 0» — не ноль
    процентов, а отсутствие наблюдений (`Measurement.share`).
    """
    counted = f"{measurement.numerator} из {measurement.denominator}"
    share = measurement.share
    if share is None:
        return f"наблюдений нет ({counted})"
    return f"{_fmt_pct(100.0 * share)} ({counted})"


def _window_word(window: Window) -> str:
    return _WINDOW_WORD[window.session_id is not None]


def _measured_line(
    stat: FrequencyStat, subject: Subject, position: str | None, measurement: Measurement,
    window: Window,
) -> str:
    """Строка измеренной величины: что, у кого, где, сколько и из скольких."""
    where = f", позиция {position}" if position is not None else ""
    title = _STAT_WORD[stat]
    return (
        f"{title[0].upper()}{title[1:]} {_SUBJECT_WORD[subject]}{where}: "
        f"{_measured(measurement)}, {_window_word(window)}."
    )


def _frequency_answer(result: FrequencyResult) -> list[str]:
    return [
        _measured_line(
            result.stat, result.subject, result.position, result.measurement, result.window
        )
    ]


def _threshold_answer(result: ThresholdResult) -> list[str]:
    lines = [
        _measured_line(
            result.stat, result.subject, result.position, result.measurement, result.window
        )
    ]
    threshold = f"Порог — {_fmt_pct(100.0 * result.threshold)}"
    if result.reference is not None:
        reference = result.reference
        threshold += (
            f" (измерен по полю: {reference.numerator} из {reference.denominator})"
        )
    lines.append(threshold + ".")
    if result.outcome is ThresholdOutcome.UNDECIDED:
        needed = result.observations_needed
        lines.append(
            f"Данных пока мало: для вывода нужно около {needed} наблюдений."
            if needed is not None
            else "Данных пока мало, и при такой величине никакая выборка вывода не даст."
        )
    else:
        lines.append(_OUTCOME_WORD[result.outcome])
    return lines


def _coverage_answer(result: CoverageResult) -> list[str]:
    return [
        _coverage_line(
            result.judged.numerator,
            result.judged.denominator,
            f"{_window_word(result.filter.window)} под этим фильтром",
        ),
        (
            f"Цена посчитана у {result.priced.numerator} из {result.priced.denominator} "
            f"оценённых, всего {_fmt_bb(-result.loss_bb)}."
        ),
        _COVERAGE_NOTE,
    ]


def _leaks_answer(result: LeaksResult) -> list[str]:
    lines = [
        _coverage_line(
            result.judged.numerator,
            result.judged.denominator,
            _window_word(result.window),
        )
    ]
    if result.leaks:
        lines.extend(_leak_line(stat) for stat in result.leaks)
    else:
        lines.append("Повторяющихся расхождений среди оценённых решений не нашлось.")
    lines.append(_LEAKS_DISCLAIMER)
    return lines


def _defense_answer(result: DefenseResult) -> list[str]:
    return [
        (
            f"Против ставки {chips(result.bet)} в банк {chips(result.pot_before)} "
            f"защищаться нужно в {_fmt_pct(100.0 * result.defend_frequency)} случаев, "
            f"сдаваться допустимо в {_fmt_pct(100.0 * result.fold_frequency)}."
        ),
        f"Колл окупается от {_fmt_pct(100.0 * result.required_equity)} эквити.",
        "Ваших раздач здесь нет вовсе: это арифметика от размера ставки.",
    ]


def question_msg(result: CalcResult, prose: str | None) -> Msg:
    """Ответ на вопрос: слова модели (если они прошли проверку) и числа расчёта.

    **Подпись собирается по `CalcResult.calc`, а не по словам модели.** Неверно
    выбранный расчёт выглядит нормальным ответом — просто не на тот вопрос, — и
    подпись здесь единственное, что делает подмену видимой игроку
    (`test_the_answer_names_the_calculation_it_came_from`).

    `prose is None` — слова не прошли проверку чисел; числа при этом посчитаны
    кодом и показываются как есть. То же решение, что у разбора без текста
    вердикта: прятать посчитанное из-за фразы модели хуже, чем показать его без
    неё.
    """
    lines: list[str] = []
    if prose is not None and prose.strip():
        lines += [line.strip() for line in prose.strip().splitlines() if line.strip()]
        lines.append("")
    if isinstance(result, FrequencyResult):
        numbers = _frequency_answer(result)
    elif isinstance(result, ThresholdResult):
        numbers = _threshold_answer(result)
    elif isinstance(result, CoverageResult):
        numbers = _coverage_answer(result)
    elif isinstance(result, LeaksResult):
        numbers = _leaks_answer(result)
    else:
        numbers = _defense_answer(result)
    lines.append(f"Посчитано: {_CALC_WORD[result.calc]}.")
    lines.extend(line for line in numbers if line)
    return Msg(text="\n".join(lines))


def question_usage_msg() -> Msg:
    """Команда без вопроса — что написать, одной строкой и примером."""
    return Msg(
        text=(
            "Спросите о своей игре текстом после команды.\n\n"
            "Например: /ask плюсовой или минусовой я на BB?\n\n"
            + _WHAT_CAN_BE_COUNTED
        )
    )


def question_refusal_msg() -> Msg:
    """Честный отказ: расчёта под этот вопрос нет, и вот что есть.

    Показывается, когда ни один инструмент не был вызван. Текста модели в нём
    нет ни строки: ответ, не опирающийся на расчёт, — это общие знания о
    покере, а продукт продаёт посчитанное.
    """
    return Msg(text="На этот вопрос у меня нет расчёта.\n\n" + _WHAT_CAN_BE_COUNTED)


def question_too_long_msg(limit: int) -> Msg:
    return Msg(text=f"Вопрос длиннее {limit} символов — сократите его.")


# Список того, что продукт умеет считать. Один на отказ и на подсказку команды:
# два списка разошлись бы, и один из них начал бы обещать несуществующее.
_WHAT_CAN_BE_COUNTED = (
    "Посчитать могу:\n"
    "• ваши частоты — вход в банк, повышение, ре-рейз, продолженная ставка, "
    "бареллы, сдача на продолженную ставку, доход до вскрытия;\n"
    "• те же частоты у поля или у названного оппонента;\n"
    "• вашу частоту против порога — по размеру ставки или по полю;\n"
    "• покрытие разбора и цену расхождений;\n"
    "• типы повторяющихся расхождений;\n"
    "• требуемую частоту защиты против ставки.\n\n"
    "Можно сузить вопрос позицией и вечером."
)
