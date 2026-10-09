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

**Разбор раздачи — сырые числа расчёта, без единого слова модели** (решение
владельца 2026-09-12): блок «Что было», затем всё, что посчитал код, с подписью
у каждого числа. В этой половине модуля регистр другой и сознательно: там
печатаются и машинные ключи `detail`, потому что прятать посчитанное нельзя
(`_DETAIL_LABELS`, `_raw_hand_lines`). Правило «расхождение, не ошибка»
действует и там.

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
именительного падежа в `_ACTION_WORD` достаточно.
`test_scan_summary_msg_item_line_is_grammatically_correct` пришпиливает
буквальный рендер строки — регресс формулировки становится красным тестом, а не
тем, что заметит игрок раньше нас.

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

**Постфлоп-точка — числа без цены.** Перебор борда отвечает на вопрос «сколько
блефов нужно в его ставящем диапазоне», а не «сколько стоило решение», поэтому
у такой точки нет вердикта, а есть требование к диапазону. Слово «лучше» рядом с
ней появляется только тогда, когда лучшую линию назвало ядро
(`PointVerdict.best_action`), и тогда же названо допущение, на котором она
держится. Ривер, тёрн и флоп печатаются одними словами и одной функцией
(`_postflop_call_lines`): считают их разные инструменты, но подписи у чисел
одни и те же.

**Точка решения в разборе — макет владельца** (2026-10-09, спека постфлоп-линии, §2,
§7): заголовок «N. Улица карты · позиция · сыграно: …», строка чисел, шансы банка и
строки «рука → ценность на вскрытии → дро → бэкдор → оверкарты → линия → окупается →
вердикта нет», каждая только когда ей есть что сказать
(`test_the_reference_hand_prints_the_owners_layout_line_by_line`). Описание сыгранного,
а не вердикт: ключ `postflop_line` читается только через `postflop_line_detail`, сырым
словарём не печатается.

**Форма «около нуля» в сводке скана — вердикт, а не отказ.** У части точек
интервал EV лежит по обе стороны нуля: при одних моделях поведения оппонентов
лучше входить, при других пасовать. Раньше такая точка до игрока не доходила
вовсе — ядро отказывалось называть число, — и вместе с числом пропадали три
вещи, которые у нас на неё есть: знак и порядок величины, ширина интервала (она
и есть мера маргинальности решения, объяснять её словами не надо) и потолок
цены выбора.

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

from collections.abc import Mapping, Sequence
from math import ceil, floor
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, model_validator

from harness.contracts.analysis import (
    POSTFLOP_LINE_DETAIL,
    RIVER_CALL_DETAIL,
    TURN_FLOP_CALL_DETAIL,
    AnalysisResult,
    EvInterval,
    PointVerdict,
    PostflopLineDetail,
    ScanItem,
    ScanSummary,
    SpotKind,
    Zone,
    postflop_line_detail,
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
from harness.contracts.canonical import CanonicalHand
from harness.contracts.enriched import (
    DecisionPoint,
    EnrichedHand,
    ValidationStatus,
)
from harness.contracts.history import (
    MAX_NOTE_COLOR_MEANING_CHARS,
    MAX_NOTE_COLOR_NAME_CHARS,
    MAX_NOTE_COLORS,
    MAX_NOTE_TEXT_CHARS,
    LeaksOverview,
    LeakStat,
    NoteColorRecord,
    NoteRecord,
    OpponentRecord,
    SessionLine,
    SessionSummary,
    is_judged,
)
from harness.contracts.postflop import (
    ActionTag,
    Combination,
    DrawKind,
    HandCategory,
    HandStrength,
    Purpose,
)
from harness.contracts.raw import ActionKind, Street
from harness.explanation.hand_replay import HandReplay, bb, cards_text, chips, signed_bb
from harness.presentation.keyboards import (
    MAIN_MENU,
    MENU_LEAKS,
    MENU_NOTES,
    MENU_SESSIONS,
    MENU_SETTINGS,
    MENU_TOURNAMENT,
    NOTE_COLOR_UNSET,
    Btn,
    deep_dive_button,
    escalation_buttons,
    note_buttons_for_hand,
    note_color_buttons,
    note_color_delete_buttons,
    note_colors_button,
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
    "disagreement_saved_msg",
    "escalation_msg",
    "failed_msg",
    "gg_nickname_saved_msg",
    "gg_nickname_too_long_msg",
    "hand_analysis_msgs",
    "help_msg",
    "hh_accepted_msg",
    "hh_prompt_msg",
    "hh_scan_in_progress_msg",
    "invite_accepted_msg",
    "invite_created_msg",
    "invite_required_msg",
    "leaks_msg",
    "new_session_msg",
    "not_a_hand_msg",
    "note_appended_msg",
    "note_color_prompt_msg",
    "note_color_saved_msg",
    "note_colors_msg",
    "note_colors_refused_msg",
    "note_colors_too_many_msg",
    "note_deleted_msg",
    "note_gone_msg",
    "note_prompt_msg",
    "note_saved_msg",
    "note_usage_msg",
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
    "scan_summary_msg",
    "screenshot_too_large_msg",
    "screenshots_not_supported_msg",
    "send_as_file_msg",
    "session_summary_msg",
    "session_unavailable_msg",
    "sessions_msg",
    "settings_msg",
    "start_msg",
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

# `PointVerdict.action_taken`/`best_action` и `CanonicalAction.kind` — токены
# движка (см. `preflop.py`, `classifier.action_name`); словарь переводит их в
# слова игрока. Строка, которой в словаре нет, показывается как есть: так сюда
# приходит готовая формулировка развилки из `preflop._BEST_DEPENDS_ON_BEHIND`.
_ACTION_WORD: dict[str, str] = {
    "fold": "фолд",
    "shove": "шов",
    "call": "колл",
    "check": "чек",
    "bet": "бет",
    "raise": "рейз",
    "limp": "лимп",
}

_SPOT_WORD: dict[SpotKind, str] = {
    SpotKind.PUSHFOLD_UNOPENED: "пуш-фолд",
    SpotKind.PUSHFOLD_FACING_SHOVE: "колл шова",
    SpotKind.OPEN_CHART: "открытие по чарту",
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

# Сколько дверей в разбор показывать под сводкой. Предел ставит Телеграм (клавиатура
# из сотен кнопок не приходит вовсе), а не аналитика: на файле в 318 рук кнопка под
# каждой невозможна физически. Десять — столько же, сколько у решений около нуля:
# оба списка существуют, чтобы дать игроку куда нажать, а не чтобы перечислить всё.
_MAX_RENDERED_DOORS = 10

# Префикс `callback_data` кнопки разбора. Дублирует `bot.router.DEEP_DIVE_PREFIX`
# по букве, но не по зависимости: `presentation` не имеет права знать про роутер
# (правило зависимостей, CLAUDE.md). Сходство держит тест, а не импорт.
_DEEP_PREFIX = "deep:"

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


def _signed_bb_number(value_bb: float) -> str:
    """Число в bb со знаком у обоих направлений — без единицы.

    В интервале «−0.3 … +0.8» знак верхнего конца несёт смысл: он и говорит, что
    интервал пересекает ноль. Без явного плюса читатель видит два числа и должен
    сам заметить, что у одного знак есть, а у другого нет.
    """
    return signed_bb(value_bb)


def _fmt_signed_bb(value_bb: float) -> str:
    """То же, что `_fmt_bb`, но плюс у положительного числа проговаривается."""
    return f"{_signed_bb_number(value_bb)} bb"


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


def _fmt_pct(value: float | None) -> str | None:
    """Доля одним знаком после запятой; `None` — доли нет, и печатать нечего.

    Возвращается `None`, а не «0.0%»: доля отсутствует ровно тогда, когда
    знаменатель нулевой (`PlayerStats`), и ноль процентов на этом месте был бы
    утверждением о том, чего не измеряли.
    """
    return None if value is None else f"{value:.1f}%"


def _plural_form(count: int, one: str, few: str, many: str) -> str:
    """Форма слова по числу: 1 — `one`, 2–4 — `few`, остальное — `many`.

    Правило трёх форм, а не двух: «нужен 1 блеф», «нужно 2 блефа», «нужно 11
    блефов» (`test_the_bluff_count_agrees_with_its_own_plural_form`).
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
    station: Literal["ask", "read", "parse", "validate", "analyze"],
) -> str:
    """Строка прогресса, которой редактируется одно сообщение по станциям конвейера."""
    return _STATION_TEXT[station]


def _scan_item_line(item: ScanItem) -> str:
    """Строка расхождения сводки: ценовая — с ценой, по чарту — с частотой у чарта."""
    head = f"№{item.hand_no} · {item.hero_class} · {_spot_word(item.spot)}: "
    if item.taken_frequency is not None:
        return (
            f"{head}{_action_word(item.action_taken)} — у чарта "
            f"{_fmt_pct(100.0 * item.taken_frequency)}, чаще всего "
            f"{_action_word(item.best_action)}"
        )
    marker = f" ({_ASSUMING_MARKER})" if item.zone is Zone.ASSUMING else ""
    return (
        f"{head}{_action_word(item.action_taken)} (лучше: {_action_word(item.best_action)}) "
        f"— {_fmt_bb(item.ev_diff_bb)}{marker}"
    )


def scan_summary_msg(s: ScanSummary, quota_left: int, quota_total: int) -> Msg:
    """Сводка префлоп-скана: список расхождений по цене, кнопка разбора под каждым.

    «Расхождение», не «ошибка» (CLAUDE.md) — скан судит по равновесию и модельным
    диапазонам, не по факту выигрыша раздачи. Строки `zone is Zone.ASSUMING`
    несут `_ASSUMING_MARKER`, строки `strict` — нет.

    **Счётчиков покрытия в сводке нет** (решение владельца 2026-09-12): «оценено
    решений X из Y» и сумма цены расхождений говорили о движке, а не о турнире, и
    на файле, где судимых точек не нашлось, были единственным содержанием сводки.
    Пустой список расхождений теперь не комментируется ничем
    (`test_the_scan_summary_counts_nothing_it_cannot_judge`).

    `hands_failed` (руки, пропущенные политикой отказа скана) показывается
    только когда он не ноль — молчание о деградации ровно то, против чего
    спроектирован весь продукт.

    Список обрезается `_MAX_RENDERED_SCAN_ITEMS` и, если обрезан, говорит об этом
    прямо: сколько найдено и сколько показано. Молча показать 20 из 60 — та же
    деградация без огласки, что и умолчанный `hands_failed` абзацем выше.
    """
    hands_word = _plural_form(s.hands_total, "рука", "руки", "рук")
    lines = [f"Скан завершён: {s.hands_total} {hands_word}."]
    if s.hands_failed:
        lines.append(f"Раздач не разобрано: {s.hands_failed} — не вошли в сводку.")
    buttons: list[list[Btn]] = []

    if s.items:
        shown = s.items[:_MAX_RENDERED_SCAN_ITEMS]
        lines.append("")
        if len(shown) < len(s.items):
            lines.append(
                f"Топ расхождений — показаны первые {len(shown)} "
                f"из {len(s.items)} найденных:"
            )
        else:
            lines.append("Топ расхождений:")
        for item in shown:
            lines.append(_scan_item_line(item))
            buttons.append([deep_dive_button(item.hand_no)])
        # Расхождения по чарту идут в конце списка, и обрезка срезает их первыми;
        # сколько их не вошло, сказано прямо (решение владельца 2026-10-03).
        cut_chart = sum(1 for item in s.items[len(shown) :] if item.taken_frequency is not None)
        if cut_chart:
            lines.append(f"…и ещё {cut_chart} {_plural_form(cut_chart, 'расхождение', 'расхождения', 'расхождений')} по чарту.")

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

    # Дверь в разбор — под КАЖДОЙ раздачей файла, а не только под теми, где нашлось
    # расхождение (round 6). Движок v1 судит лишь пуш-фолд, поэтому на обычном
    # файле список расхождений пуст — и вместе с ним прежде исчезала единственная
    # кнопка `deep_dive_button` во всём продукте: разбор раздачи со всем, что в нём
    # есть (блок «Что было», числа расчёта), был из HH-входа недостижим в принципе.
    # Кнопка — дверь, а не награда за найденную ошибку
    # (`test_a_hand_without_a_disagreement_still_has_a_way_into_its_analysis`).
    already = {b.callback_data for row in buttons for b in row}
    rest = [no for no in s.hand_nos if f"{_DEEP_PREFIX}{no}" not in already]
    if rest:
        shown_rest = rest[:_MAX_RENDERED_DOORS]
        lines.append("")
        head = "Разобрать раздачу"
        if len(shown_rest) < len(rest):
            head += f" (первые {len(shown_rest)} из {len(rest)})"
        lines.append(f"{head}:")
        buttons.extend([deep_dive_button(no)] for no in shown_rest)

    lines.append("")
    lines.append(f"Доступно: {_quota_line(quota_left, quota_total)}.")
    return Msg(text="\n".join(lines), buttons=buttons)


# Оговорка о непроверенном. «Проверить было нечем» — не то же, что «проверено и
# сошлось», и разница обязана быть видна игроку, а не только в трейсе.
_NOT_CHECKED_PREFIX = "Проверить на этом экране было нечем:"

# Слова про постфлоп-точку — одни и те же на всех трёх улицах. Требование к
# ставящему диапазону — число блефов на заданное вэлью — печатается вместе с
# долей, которую эти блефы в диапазоне занимают: одно число отвечает «сколько»,
# второе — «насколько это много».
#
# Разложения борда по исходам (сколько комбо бьёт, проигрывает, делит) в этих
# строках нет (решение владельца): это не диапазон соперника, а полный перебор
# возможного, и читается как чужой диапазон, которого мы не знаем.
#
# Цены колла здесь тоже нет: банк, доплату и требуемую эквити печатает строка
# шансов банка той же точки (`_decision_lines`), и второй раз те же три числа
# были бы дублем.
_CALL_ASSUMPTION = "Допущение: сильнейшие руки он ставит."


def _rounded_bluffs(value: float) -> int:
    """Число блефов целым: к ближайшему, половина вверх.

    Вверх, а не по правилу `round` (оно округляет половину к чётному): требование
    к диапазону читается как «столько-то рук», и 0.5 обязана дать 1, а не 0
    (`test_half_a_bluff_is_rounded_up_to_one`).
    """
    return int(value + 0.5)


class _CallNumbers(NamedTuple):
    """Требование к ставящему диапазону в форме, одинаковой для всех трёх улиц.

    Ривер и пара «тёрн, флоп» приходят из разных расчётов и лежат в `detail`
    под разными ключами, но показываются игроку одними словами. Общая форма
    здесь — то место, где две редакции этих слов не разойдутся.

    `bluffs` — `None`, когда требуемого числа блефов не существует (тёрн и
    флоп: не хватает и всего борда); `share` — `None` там же.
    `beyond_the_board` — требование превышает то, что борд вмещает.
    """

    min_value_combos: int
    bluffs: float | None
    share: float | None
    beyond_the_board: bool


def _call_numbers(point: PointVerdict) -> _CallNumbers | None:
    """Требование к диапазону из `detail` — или `None`, если его там нет."""
    river = river_call_detail(point)
    if river is not None:
        return _CallNumbers(
            min_value_combos=river.min_value_combos,
            bluffs=river.bluffs_needed_min_value,
            share=river.bluff_share,
            beyond_the_board=river.fold_proven,
        )
    early = turn_flop_call_detail(point)
    if early is not None:
        return _CallNumbers(
            min_value_combos=early.min_value_combos,
            bluffs=early.bluffs_needed_min_value,
            share=early.bluff_share,
            beyond_the_board=early.bluffs_needed_min_value is None,
        )
    return None


def _postflop_call_lines(point: PointVerdict) -> list[str]:
    """Требование к ставящему диапазону и лучшая линия постфлоп-точки.

    Пустой список — у точки нет посчитанных чисел (`_call_numbers`), и печатать
    нечего. Строка про блефы пропускается, когда требования нет вовсе — вэлью
    старшего класса на борде не осталось или блефов нужно меньше одного:
    «нужно 0 блефов» не утверждение, а вырожденный случай
    (`test_a_degenerate_river_requirement_prints_no_requirement_at_all`).

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
    lines = _bluff_line(numbers)
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
# экран заметок (`_fitted`) и хвост разбора, не влезший даже во второе сообщение
# (`hand_analysis_msgs`).
_MARKER = " […показано не целиком]"


def _render_html(head: str, rows: list[str]) -> str:
    """Блок «Что было» уже с разметкой; остальные строки экранируются здесь.

    Две пустые строки подряд — дыра в сообщении, поэтому пустая строка идёт в
    текст, только если предыдущая непустая
    (`test_a_hand_too_long_for_one_message_goes_out_in_two`).
    """
    lines = [head]
    for text in rows:
        if not text and not lines[-1]:
            continue
        lines.append(_html_escape(text))
    return "\n".join(lines)


# --- сырые данные раздачи: всё, что посчитал код, с подписью у каждого числа ----------

def _raw_bb(value_chips: int, big_blind: int) -> str:
    """Сумма в ББ с единицей. Формат общий с блоком «Что было» (`hand_replay.bb`),
    а не вторая его копия: одна величина в двух местах одного сообщения обязана
    округляться одинаково, иначе разбор спорит сам с собой."""
    return f"{bb(value_chips, big_blind)} ББ"


def _raw_bb_value(value_bb: float) -> str:
    """Готовая величина в ББ — знак и округление общие с продуктовым `_fmt_bb`.

    Отличается от него ТОЛЬКО единицей: «ББ», как весь блок «Что было» (спека
    §5.6). Смешивать две записи в одном сообщении нельзя: читатель принимает их
    за разные величины.
    """
    return f"{_bb_number(value_bb)} ББ"


def _raw_signed_bb(value_bb: float) -> str:
    """То же в ББ, но плюс у положительного числа проговаривается: у цены и у
    концов интервала знак несёт смысл."""
    return f"{_signed_bb_number(value_bb)} ББ"


# Слова игрока для машинных значений `detail`. Значение без перевода печатается
# как есть — по той же причине, что и ключ без подписи.
_METHOD_WORD: dict[str, str] = {
    "full_deal_shove": "симуляция полной раздачи: шов",
    "full_deal_call": "симуляция полной раздачи: колл против диапазона шовера",
    "open_chart": "сверка с чартом солвера",
    "prefilter_chart_lookup": "лукап по чарту",
}
_BRACKET_WORD: dict[str, str] = {"stable": "устойчива", "unstable": "через ноль"}

# Та же пара значений у второй оси (`behind_axis`), но отвечает она на другой
# вопрос — «сдвинет ли вердикт вход живых за вами», — и словами «устойчива /
# через ноль» читалась бы как ответ первой.
_AXIS_WORD: dict[str, str] = {
    "stable": "вердикт не меняется",
    "unstable": "вердикт меняется",
}

# Чем кончилась сверка денег расчёта с суммами источника. Словарь накрывает весь
# `ValidationStatus` — это держит тест, а не внимательность правившего
# (`test_every_validation_status_has_a_word_of_its_own`).
_VALIDATION_WORD: dict[ValidationStatus, str] = {
    ValidationStatus.PASS: "сошлась",
    ValidationStatus.ESCALATE: "требует ответа игрока",
    ValidationStatus.REJECT: "не сошлась",
}

# Подписи ключей `detail` — того, что ядро положило в точку. Собраны по местам,
# где `detail` наполняется: `analysis/preflop.py` (шов в неоткрытый банк, ответ
# на шов, лукап по чарту, `_call_model_detail`, отказ солвера) и
# `analysis/classifier.py` (`unjudged_point`). Ключ без подписи печатается своим
# машинным именем — прятать посчитанное нельзя.
#
# Полноту таблицы держит `test_every_detail_key_the_analysis_produces_has_a_label`
# ДВУМЯ способами: ключи, которые ядро кладёт на фикстурах, он собирает прогоном
# настоящего `analyze_hand`, а ключи веток, до которых фикстуры не доходят
# (отказ солвера, лукап по чарту), перечислены в самом тесте списком.
_DETAIL_LABELS: dict[str, str] = {
    "method": "метод расчёта",
    "bracket": "вилка по ширине диапазона",
    "simulated_deals": "раздач в симуляции",
    "simulated_deals_by_width": "раздач в симуляции на каждую ширину",
    "simulation_seed": "сид симуляции",
    "fold_equity_ok": "фолд-эквити возможна",
    "ev_shove_bb": "EV шова, ББ",
    "ev_shove_tight_bb": "EV шова против узкого диапазона, ББ",
    "ev_shove_wide_bb": "EV шова против широкого диапазона, ББ",
    "ev_shove_by_width_bb": "EV шова по ширине диапазона, ББ",
    "ev_call_bb": "EV колла, ББ",
    "ev_call_tight_bb": "EV колла против узкого диапазона, ББ",
    "ev_call_wide_bb": "EV колла против широкого диапазона, ББ",
    "ev_call_by_width_bb": "EV колла по ширине диапазона, ББ",
    "ev_call_all_behind_bb": "EV колла, если входят все живые за вами, ББ",
    "hero_class": "класс вашей руки",
    "depths_bb": "глубины стеков, ББ",
    "dead_extra_bb": "мёртвых денег в банке, ББ",
    "zone_reason": "почему такая зона доверия",
    "model_within_bracket": "модель попала в вилку",
    "shove_range_fraction": "рук в диапазоне шова",
    "call_range_fractions": "рук в диапазонах колла",
    "overcall_range_fractions": "рук в диапазонах оверколла",
    "equilibrium_hand_regret_bb": "отклонение этой руки от равновесия, ББ",
    "p_all_fold": "вероятность, что все спасуют",
    "expected_callers": "ожидаемое число ответивших",
    "required_equity": "требуемая эквити",
    "shover_depth_bb": "глубина стека шовера, ББ",
    "live_others": "живых за вами в переборе",
    "behind_axis": "устойчивость к входу живых за вами",
    "rivals_when_shoved": "соперников на момент шова",
    "best_vs_one": "лучше по модели",
    "best_all_behind": "лучше, если входят все живые за вами",
    "push_weight": "вес руки в чарте шова",
    "lookup_depth_bb": "глубина лукапа по чарту, ББ",
    "solver_error": "сбой расчёта",
    "chart_depth_bb": "глубина чарта, ББ",
    "open_depth_bb": "ваша глубина в мерах чарта (стек до анте), ББ",
    "taken_frequency": "частота сыгранного по чарту",
    "chart_source": "источник чарта",
    "chart_revised_at": "чарт сверен с источником",
}

# Ключи `detail`, чьё значение — доля единицы: печатаются процентом, как все
# доли продукта. Дробь и процент в одном сообщении читатель принимает за разные
# величины (`test_a_share_is_printed_as_a_percentage_not_as_a_fraction`).
_SHARE_KEYS = frozenset(
    {
        "required_equity",
        "shove_range_fraction",
        "call_range_fractions",
        "overcall_range_fractions",
        "push_weight",
        "taken_frequency",
        "p_all_fold",
    }
)

# Ключи, чьё значение — токен действия движка (`fold`/`call`/`shove`).
_ACTION_KEYS = frozenset({"best_vs_one", "best_all_behind"})

# Ключи, чьё значение — свободный текст ядра, написанный для игрока, но с
# токенами движка внутри: «на узком конце лучше «shove»». Кавычки-ёлочки и есть
# та граница, по которой токен можно перевести, не трогая остальную фразу
# (`test_the_reason_for_the_zone_speaks_the_words_of_the_player`).
_PROSE_KEYS = frozenset({"zone_reason", "unmodelled"})

# Ключи, чьё значение — глубина стека в ББ: печатается одним знаком, как все
# стеки продукта. Два знака у глубины и один у стека в соседней строке читатель
# принимает за разную точность измерения.
_STACK_KEYS = frozenset(
    {"depths_bb", "shover_depth_bb", "lookup_depth_bb", "chart_depth_bb", "open_depth_bb"}
)

# Частоты чарта точки открытия печатаются в строке вердикта словами
# (`_chart_verdict_line`) и в общем переборе `detail` не повторяются.
_CHART_FREQUENCIES_KEY = "chart_frequencies"
_CHART_ACTION_ORDER = ("raise", "shove", "limp", "fold")

# Ключ, под которым ядро пишет причину отказа. Печатается отдельной строкой
# «вердикта нет: …» (`_verdict_lines`), а в общем переборе `detail`
# пропускается: дважды одно и то же — не диагностика.
_UNJUDGED_KEY = "unjudged"


def _required_equity(to_call: int, pot_before: int) -> float:
    """Доля банка, от которой колл окупается: `to_call / (pot_before + to_call)`.

    Копия `analysis.tools.pot_odds.required_equity` — не по лени, а по правилу
    зависимостей (CLAUDE.md): `presentation` не имеет права импортировать
    `analysis`, потому что пакет тянет в образ бота `eval7` и `pokerkit`
    (`test_bot_image_does_not_import_calculation_stack`). Две копии держит
    рядом тест `test_the_pot_odds_of_the_raw_block_agree_with_the_calculation_tool`,
    а не память правившего.
    """
    return to_call / (pot_before + to_call)


def _detail_number(value: float) -> str:
    """Число `detail`: два знака, мельче сотой — шесть, минус типографский.

    Шесть, а не четыре: ядро округляет доли до шестого знака (`preflop.py`), и
    на четырёх `0.000005` показалось бы нулём — то есть «не посчитано» вместо
    посчитанного. Экспоненциальной записи нет ни при каком значении: «5e-06» в
    тексте игрока не число, а сообщение об усталости формата
    (`test_a_detail_value_that_is_empty_prints_a_dash_not_a_blank`).
    """
    magnitude = abs(value)
    digits = 2 if magnitude >= 0.01 or magnitude == 0.0 else 6
    body = f"{magnitude:.{digits}f}"
    return f"−{body}" if value < 0 and float(body) != 0.0 else body


def _detail_value(value: Any) -> str:
    """Значение из `detail` в текст.

    `bool` проверяется раньше числа: он им и является. Пустота печатается прочерком,
    а не пустым местом: «рук в диапазонах колла:» без единого знака после
    двоеточия читается как потерянное значение, а не как «их не было»
    (`test_a_detail_value_that_is_empty_prints_a_dash_not_a_blank`).
    """
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, float):
        return _detail_number(value)
    if isinstance(value, dict):
        if not value:
            return "—"
        return " · ".join(f"{_width_key(key)}: {_detail_value(item)}" for key, item in value.items())
    if isinstance(value, list | tuple):
        if not value:
            return "—"
        return ", ".join(_detail_value(item) for item in value)
    return str(value)


def _translated_tokens(text: str) -> str:
    """Токены движка в кавычках-ёлочках — словами игрока; остальной текст как есть."""
    for token, word in _ACTION_WORD.items():
        text = text.replace(f"«{token}»", f"«{word}»")
    return text


def _width_key(key: str) -> str:
    """Ключ разбивки по ширине диапазона: `x0.35` — множитель, а не латинская
    буква перед числом."""
    return f"×{key[1:]}" if key.startswith("x") else key


def _keyed_value(key: str, value: Any) -> str:
    """Значение `detail` по правилам СВОЕГО ключа: доли процентом, токены
    действий словами, метод и вилка — словарями, остальное — общим правилом."""
    if key in _SHARE_KEYS:
        if isinstance(value, list):
            return ", ".join(str(_fmt_pct(100.0 * item)) for item in value) if value else "—"
        return "—" if value is None else str(_fmt_pct(100.0 * float(value)))
    if key in _ACTION_KEYS:
        return "—" if value is None else _action_word(str(value))
    if key in _PROSE_KEYS and isinstance(value, str):
        return _translated_tokens(value)
    if key in _STACK_KEYS:
        if isinstance(value, list):
            return ", ".join(f"{float(item):.1f}" for item in value) if value else "—"
        return "—" if value is None else f"{float(value):.1f}"
    if key == "method" and isinstance(value, str):
        return _METHOD_WORD.get(value, value)
    if key == "bracket" and isinstance(value, str):
        return _BRACKET_WORD.get(value, value)
    if key == "behind_axis" and isinstance(value, str):
        return _AXIS_WORD.get(value, value)
    return _detail_value(value)


def _detail_lines(point: PointVerdict) -> list[str]:
    """Всё содержимое `detail`, кроме того, что уже напечатано своими словами.

    Числа инструментов колла (`RIVER_CALL_DETAIL`, `TURN_FLOP_CALL_DETAIL`) идут
    первыми словами игрока; причина отказа ядра стоит строкой «вердикта нет: …»
    выше; постфлоп-линия (`POSTFLOP_LINE_DETAIL`) печатается своими строками в
    `_postflop_line_lines` и сырым словарём не печатается
    (`test_the_postflop_line_is_not_printed_as_a_raw_dictionary`); остальные ключи
    печатаются по таблице подписей.
    """
    skip = (
        RIVER_CALL_DETAIL,
        TURN_FLOP_CALL_DETAIL,
        POSTFLOP_LINE_DETAIL,
        _UNJUDGED_KEY,
        _CHART_FREQUENCIES_KEY,
    )
    lines = _postflop_call_lines(point)
    for key, value in point.detail.items():
        if key in skip:
            continue
        lines.append(f"    {_DETAIL_LABELS.get(key, key)}: {_keyed_value(key, value)}")
    return lines


def _half_up_percent(share: float) -> int:
    """Доля в целый процент: к ближайшему, половина вверх.

    Не `round` (он округляет половину к чётному) и не `:.0f`: «28.5% банка» обязано
    дать 29 (`test_a_percent_of_the_pot_rounds_half_up`). Поправка `1e-9` снимает
    двоичный хвост доли вроде `0.285 · 100 = 28.499999999999996`: доли здесь —
    отношения целых фишек, и до настоящей «ровно чуть меньше половины» им дальше.
    """
    return int(100.0 * share + 0.5 + 1e-9)


def _percent(share: float) -> str:
    return f"{_half_up_percent(share)}%"


def _approx_percent(share: float) -> str:
    """Константа владельца (§4.2.1) с `≈` и без хвоста нулей: `≈4%`, `≈6.5%`, `≈1.5%`."""
    return f"≈{round(100.0 * share, 1):g}%"


def _played_words(dp: DecisionPoint, big_blind: int, size_pct: float | None = None) -> str:
    """Что сыграно и на какую сумму — каждому действию своё число.

    Колл называет ДОПЛАТУ (`to_call`), бет и рейз — итог, до которого подняли
    (`CanonicalAction.committed_after` — накопленное за улицу), чек и фолд не
    несут суммы вовсе: денег в них нет
    (`test_the_action_of_a_point_prints_the_number_that_belongs_to_it`). Слово «до»
    стоит только у рейза; процент банка в скобках — у ставки и рейза, когда ядро его
    посчитало (`size_pct`: на префлопе его нет), тег размера не печатается
    (`test_a_bet_and_a_raise_print_the_percent_of_the_pot_and_not_the_size_tag`).
    """
    kind = dp.action.kind
    word = _action_word(kind.value)
    all_in = ", олл-ин" if dp.action.is_all_in else ""
    if kind is ActionKind.CALL:
        return f"{word} {_raw_bb(dp.to_call, big_blind)}{all_in}"
    if kind in (ActionKind.BET, ActionKind.RAISE):
        until = " до" if kind is ActionKind.RAISE else ""
        share = "" if size_pct is None else f" ({_percent(size_pct)} банка)"
        return f"{word}{until} {_raw_bb(dp.action.committed_after, big_blind)}{share}{all_in}"
    return f"{word}{all_in}"


def _point_title(
    dp: DecisionPoint, hand: CanonicalHand, detail: PostflopLineDetail | None
) -> str:
    """Заголовок точки: номер, улица, позиция, карты, что сыграно (макет §2 спеки).

    Префлоп несёт карты героя, постфлоп — карты, пришедшие на этой улице (флоп три
    подряд, тёрн и ривер одну); масти — форматом блока «Что было:». Карт, которых в
    руке нет, в заголовке нет.
    """
    size_pct = None if detail is None or detail.line is None else detail.line.size_pct
    played = f"сыграно: {_played_words(dp, hand.bb, size_pct)}"
    street = _STREET_WORD.get(dp.street, dp.street.value)
    if dp.street is Street.PREFLOP:
        hero_cards = hand.dealt.get(hand.hero_label, [])
        shown = cards_text(hero_cards) if hero_cards else ""
        head = f"{dp.index + 1}. {street}"
        return " · ".join(part for part in (head, dp.position, shown, played) if part)
    arrived = hand.boards.get(dp.street, [])
    head = f"{dp.index + 1}. {street}" + (f" {cards_text(arrived)}" if arrived else "")
    return f"{head} · {dp.position} · {played}"


def _decision_lines(
    dp: DecisionPoint, hand: CanonicalHand, detail: PostflopLineDetail | None = None
) -> list[str]:
    """Числа точки решения: заголовок, банк, доставить, эфф., SPR, шансы банка.

    Строка чисел несёт только то, что есть: «доставить» — при `to_call > 0`, SPR —
    на постфлопе и когда он посчитан; на префлопе стоит «банк до хода», на
    постфлопе «банк»; эффективный стек подписан «эфф.» (спека §4.7). Шансы банка —
    одной фразой, без повтора банка и доплаты, которые стоят строкой выше.
    """
    postflop = dp.street is not Street.PREFLOP
    big_blind = hand.bb
    numbers = [
        f"{'банк' if postflop else 'банк до хода'} {_raw_bb(dp.pot_before, big_blind)}"
    ]
    if dp.to_call > 0:
        numbers.append(f"доставить {_raw_bb(dp.to_call, big_blind)}")
    numbers.append(f"эфф. {_raw_bb(dp.eff_stack, big_blind)}")
    if postflop and dp.spr is not None:
        numbers.append(f"SPR {dp.spr:.1f}")
    numbers.append(f"живых {dp.live_total} (после вас {dp.live_behind})")
    lines = [_point_title(dp, hand, detail), "    " + " · ".join(numbers)]
    if dp.to_call > 0:
        equity = _fmt_pct(100.0 * _required_equity(dp.to_call, dp.pot_before))
        lines.append(f"    шансы банка: колл окупается от {equity} эквити")
    return lines


# --- постфлоп-линия: рука, дро, линия, окупается (спека 2026-10-09, §4, §7) -----------

# Ранги во множественном числе родительного падежа: «пара дам», «каре тузов».
_RANK_GENITIVE: dict[str, str] = {
    "A": "тузов",
    "K": "королей",
    "Q": "дам",
    "J": "валетов",
    "T": "десяток",
    "9": "девяток",
    "8": "восьмёрок",
    "7": "семёрок",
    "6": "шестёрок",
    "5": "пятёрок",
    "4": "четвёрок",
    "3": "троек",
    "2": "двоек",
}

# Вид дро. Стрит-дро несёт ранги аутов в скобках (`Draw.out_ranks`); флеш — нет.
_DRAW_KIND_WORD: dict[DrawKind, str] = {
    DrawKind.FLUSH: "флеш",
    DrawKind.OPEN_ENDED: "двусторонний стрит",
    DrawKind.DOUBLE_GUTSHOT: "двойной гатшот",
    DrawKind.GUTSHOT: "гатшот",
}

# Тип действия героя. Нет в словаре — тип не печатается: обычная ставка, обычный
# колл и пасы (спека §4.5: «у обычного колла и обычной ставки тип не печатается,
# только назначение»; у фолда строки «линия» нет вовсе). Баррель — отдельно:
# слово зависит от номера.
_ACTION_TAG_WORD: dict[ActionTag, str] = {
    ActionTag.CBET: "с-бет",
    ActionTag.REPEAT_BET: "повторная ставка",
    ActionTag.DELAYED_CBET: "отложенный с-бет",
    ActionTag.PROBE: "проба",
    ActionTag.DONK: "донк",
    ActionTag.BET_AFTER_CHECK: "ставка после чека",
    ActionTag.CHECK_CALL: "чек-колл",
    ActionTag.CHECK_RAISE: "чек-рейз",
    ActionTag.RAISE: "рейз",
    ActionTag.THREE_BET: "3-бет",
    ActionTag.RERAISE: "ререйз",
    ActionTag.FLOAT: "флоат",
}
_BARREL_WORD: dict[int, str] = {2: "второй баррель", 3: "третий баррель"}
_UNPRINTED_TAGS = frozenset({ActionTag.BET, ActionTag.CALL, ActionTag.FOLD, ActionTag.CHECK_FOLD})
_NO_LINE_TAGS = frozenset({ActionTag.FOLD, ActionTag.CHECK_FOLD})

_PURPOSE_WORD: dict[Purpose, str] = {
    Purpose.VALUE: "вэлью",
    Purpose.MEDIUM_HAND: "ставка со средней рукой",
    Purpose.SEMIBLUFF: "полублеф",
    Purpose.BLUFF: "блеф",
    Purpose.CALL_STRONG: "колл с сильной рукой",
    Purpose.BLUFF_CATCH: "ловля блефа",
    Purpose.CALL_WITH_DRAW: "колл с дро",
}


def _ranks_genitive(ranks: Sequence[str]) -> list[str]:
    return [_RANK_GENITIVE.get(rank, rank) for rank in ranks]


def _combination_words(combination: Combination, ranks: Sequence[str]) -> str:
    """Комбинация словами по рангам, которые ядро назвало (`HandStrength.ranks`)."""
    genitive = _ranks_genitive(ranks)
    top = ranks[0] if ranks else ""
    match combination:
        case Combination.PAIR:
            return f"пара {genitive[0]}"
        case Combination.TWO_PAIR:
            return f"две пары ({' и '.join(genitive)})"
        case Combination.TRIPS:
            return f"трипс {genitive[0]}"
        case Combination.STRAIGHT:
            return f"стрит (старшая {top})"
        case Combination.FLUSH:
            return f"флеш (старшая {top})"
        case Combination.FULL_HOUSE:
            return f"фулл-хаус (тройка {genitive[0]}, пара {genitive[1]})"
        case Combination.QUADS:
            return f"каре {genitive[0]}"
        case Combination.STRAIGHT_FLUSH:
            return f"стрит-флеш (старшая {top})"
        case Combination.HIGH_CARD:
            return f"старшая {top}, без пары"


def _hand_words(hand: HandStrength) -> str:
    """Рука героя одной фразой: «старшая 5, без пары», «топ-пара королей, слабый кикер»."""
    genitive = _ranks_genitive(hand.ranks[:1])
    named = genitive[0] if genitive else ""
    match hand.category:
        case HandCategory.NO_PAIR:
            return f"старшая {hand.ranks[0]}, без пары"
        case HandCategory.ON_BOARD:
            plays = f"играет кикер {hand.kicker}" if hand.plays == "kicker" else "играет борд"
            return f"{_combination_words(hand.combination, hand.ranks)} на борде, {plays}"
        case HandCategory.WEAK_FLUSH | HandCategory.WEAK_STRAIGHT:
            return f"слабый {_combination_words(hand.combination, hand.ranks)}"
        case HandCategory.SET:
            return f"сет {named}"
        case HandCategory.OVERPAIR:
            return f"оверпара {named}"
        case HandCategory.TOP_PAIR_STRONG_KICKER:
            return f"топ-пара {named}, сильный кикер"
        case HandCategory.TOP_PAIR_WEAK_KICKER:
            return f"топ-пара {named}, слабый кикер"
        case HandCategory.MIDDLE_PAIR:
            return f"средняя пара {named}"
        case HandCategory.WEAK_PAIR:
            return f"слабая пара {named}"
        case (
            HandCategory.STRAIGHT_FLUSH
            | HandCategory.QUADS
            | HandCategory.FULL_HOUSE
            | HandCategory.FLUSH
            | HandCategory.STRAIGHT
            | HandCategory.TRIPS
            | HandCategory.TWO_PAIR
        ):
            return _combination_words(hand.combination, hand.ranks)


def _hand_line(detail: PostflopLineDetail) -> str | None:
    """«рука: …» и, на ривере после несобранного дро, «· дро не закрылось»."""
    if detail.hand is None:
        return None
    missed = " · дро не закрылось" if detail.draw_missed else ""
    return f"    рука: {_hand_words(detail.hand)}{missed}"


def _showdown_line(detail: PostflopLineDetail) -> str | None:
    """Ценность на вскрытии — только крайние случаи, словами и без числа комбо (§4.8).

    Натс — когда проигрышей нет (раньше нуля выигрышей: рука, которой нечего
    проигрывать, не «нулевая»); нулевая — когда выигрышей нет. Иначе строки нет.
    Классы делящих рук называются один раз на пару рангов: «53o» и «53s» — «5-3».
    """
    value = detail.showdown
    if value is None:
        return None
    if value.losses == 0:
        return "    ценность на вскрытии: натс (не проигрывает ни одной руке)"
    if value.wins > 0:
        return None
    lost = "не выигрывает ни у одной руки"
    named: list[str] = []
    for hand_class in value.ties_with:
        ranks = hand_class[:2]
        text = ranks[0] + ranks[1] if ranks[0] == ranks[1] else f"{ranks[0]}-{ranks[1]}"
        if text not in named:
            named.append(text)
    shared = f", делит банк только с {', '.join(named)}" if named else ""
    return f"    ценность на вскрытии: нулевая ({lost}{shared})"


def _draw_line(detail: PostflopLineDetail) -> str | None:
    """«дро: вид (ранги аутов) · N аутов · шанс собрать …» — флоп и тёрн разными словами."""
    draw = detail.draw
    if draw is None:
        return None
    ranks = f" ({', '.join(draw.out_ranks)})" if draw.out_ranks else ""
    kinds = " + ".join(
        _DRAW_KIND_WORD[kind] + (ranks if kind is not DrawKind.FLUSH else "")
        for kind in draw.kinds
    )
    outs = len(draw.outs)
    outs_word = _plural_form(outs, "аут", "аута", "аутов")
    next_pct = _fmt_pct(100.0 * draw.hit_next)
    if draw.hit_by_river is None:
        chance = f"шанс собрать на ривере {next_pct}"
    else:
        chance = (
            f"шанс собрать: на тёрне {next_pct}, "
            f"тёрн + ривер {_fmt_pct(100.0 * draw.hit_by_river)}"
        )
    return f"    дро: {kinds} · {outs} {outs_word} · {chance}"


def _backdoor_line(detail: PostflopLineDetail) -> str | None:
    """«бэкдор: флеш ≈4% · стрит ≈3%» — цифры помечены `≈` (константы владельца, §4.2.1)."""
    if not detail.backdoors:
        return None
    kinds = {"flush": "флеш", "straight": "стрит"}
    parts = [f"{kinds[b.kind]} {_approx_percent(b.approx)}" for b in detail.backdoors]
    return f"    бэкдор: {' · '.join(parts)}"


def _overcards_line(detail: PostflopLineDetail) -> str | None:
    """«оверкарты: 2 (A, K) · ≈12% к риверу, если пара будет лучшей» (флоп); на тёрне
    — «≈3% на ривере»."""
    over = detail.overcards
    if over is None:
        return None
    named = ", ".join(card[0] for card in over.cards)
    if over.approx_by_river is not None:
        chance = f"{_approx_percent(over.approx_by_river)} к риверу"
    else:
        chance = f"{_approx_percent(over.approx_next)} на ривере"
    return f"    оверкарты: {len(over.cards)} ({named}) · {chance}, если пара будет лучшей"


def _line_line(detail: PostflopLineDetail) -> str | None:
    """«линия: проба · полублеф» — тип и назначение; у чека и фолда строки нет."""
    line = detail.line
    if line is None or line.action in _NO_LINE_TAGS:
        return None
    if line.action is ActionTag.BARREL:
        kind = _BARREL_WORD.get(line.barrel or 0, "баррель")
    else:
        kind = "" if line.action in _UNPRINTED_TAGS else _ACTION_TAG_WORD[line.action]
    purpose = "" if line.purpose is None else _PURPOSE_WORD[line.purpose]
    parts = [part for part in (kind, purpose) if part]
    return f"    линия: {' · '.join(parts)}" if parts else None


def _payoff_line(detail: PostflopLineDetail, big_blind: int) -> str | None:
    """«окупается: …» — порог фолдов у блефа и полублефа, доплата у колла с дро (§4.6).

    У вэлью и средней руки порога нет — ядро его не кладёт, и печатать нечего. Сумма
    «добрать позже» — фишки ядра в ББ, вверх до десятой: требование не занижается
    (`test_the_amount_to_win_later_rounds_up_to_a_tenth`).
    """
    line = detail.line
    fold = detail.fold_threshold
    if (
        fold is not None
        and line is not None
        and line.purpose in (Purpose.BLUFF, Purpose.SEMIBLUFF)
    ):
        if line.purpose is Purpose.BLUFF:
            return f"    окупается: от {_percent(fold.bluff)} фолдов"
        head = f"как чистый блеф от {_percent(fold.bluff)} фолдов"
        if fold.semibluff_free:
            return f"    окупается: {head}; с учётом аутов окупается и без фолдов"
        if fold.semibluff is not None:
            return f"    окупается: {head}; с учётом аутов от {_percent(fold.semibluff)} фолдов"
        return f"    окупается: {head}"
    call = detail.draw_call
    if call is None:
        return None
    if call.by_pot_odds:
        return "    окупается: по шансам банка"
    if call.beyond_stack:
        return "    окупается: добрать столько нельзя — колл не окупается добором"
    if call.implied_needed_chips is not None:
        return f"    окупается: нужно добрать позже {_bb_up(call.implied_needed_chips, big_blind)} ББ"
    return None


def _bb_up(value_chips: int, big_blind: int) -> str:
    """Фишки в ББ одним знаком, вверх до десятой: 1 234 при ББ 100 — «12.4»."""
    tenths = -(-10 * value_chips // big_blind)
    return f"{tenths / 10:.1f}"


def _postflop_line_lines(detail: PostflopLineDetail | None, big_blind: int) -> list[str]:
    """Строки постфлоп-линии в порядке макета: рука → ценность на вскрытии → дро →
    бэкдор → оверкарты → линия → окупается. Каждая — только когда ей есть что
    сказать (`test_an_empty_field_prints_no_line`)."""
    if detail is None:
        return []
    candidates = (
        _hand_line(detail),
        _showdown_line(detail),
        _draw_line(detail),
        _backdoor_line(detail),
        _overcards_line(detail),
        _line_line(detail),
        _payoff_line(detail, big_blind),
    )
    return [text for text in candidates if text is not None]


def _chart_frequencies_text(frequencies: Mapping[str, float]) -> str:
    """Ненулевые частоты чарта словами: «рейз 40%, фолд 60%»."""
    return ", ".join(
        f"{_action_word(action)} {_fmt_pct(100.0 * frequencies[action])}"
        for action in _CHART_ACTION_ORDER
        if frequencies.get(action, 0.0) > 0.0
    )


def _chart_verdict_line(point: PointVerdict) -> str:
    """Вердикт точки открытия по чарту: частоты вместо цены и «лучше».

    Цены у такой точки нет, и строки «цена» нет; «лучше» не печатается — у чарта
    нет лучшего, есть частоты (спека 2026-10-03-open-chart-verdict, §6). Частоты
    печатаются и тогда, когда расхождения нет: смешанная рука видна как смешанная.
    """
    depth = float(point.detail["chart_depth_bb"])
    verdict = "расхождение" if point.mismatch else "в пределах чарта"
    return (
        f"    вердикт: {_spot_word(point.spot)} {depth:.0f}bb · "
        f"зона {_ZONE_WORD.get(point.zone, str(point.zone))} · "
        f"сыграно {_action_word(point.action_taken)} · "
        f"по чарту: {_chart_frequencies_text(point.detail[_CHART_FREQUENCIES_KEY])} · "
        f"{verdict}"
    )


def _verdict_lines(point: PointVerdict) -> list[str]:
    """Вердикт точки — или одна строка о том, что его нет.

    **Зона, цена и «лучше» печатаются только у судимой точки** (`is_judged`). У
    остальных `unjudged_point` конструирует `zone=strict` и `ev_diff_bb=0.0` как
    заглушки — инвариант контракта требует заполнить поля, — и напечатать их
    значило бы выдать отсутствие расчёта за строгий нулевой вердикт
    (`test_a_point_without_a_verdict_gets_no_zone_no_price_and_no_better_line`).
    """
    if not is_judged(point):
        reason = point.detail.get(_UNJUDGED_KEY)
        if reason == "":
            # Пустая причина — ядро сказало, что говорить нечего (чек на постфлопе,
            # спека §4.9): строки нет. Ключа нет вовсе — причина не названа, и точка
            # честно говорит, что вердикта нет (`test_a_point_without_a_verdict_and_
            # without_a_reason_still_says_so`).
            return []
        return [f"    вердикта нет: {reason}" if reason else "    вердикта нет."]
    if point.mismatch is not None:
        return [_chart_verdict_line(point)]
    lines = [
        (
            f"    вердикт: спот {_spot_word(point.spot)} · "
            f"зона {_ZONE_WORD.get(point.zone, str(point.zone))} · "
            f"сыграно {_action_word(point.action_taken)} · "
            f"лучше {_action_word(point.best_action)} · "
            f"цена {_raw_signed_bb(point.ev_diff_bb)}"
        )
    ]
    interval = point.interval
    if interval is not None:
        # Потолок цены выбора отвечает на вопрос «сколько стоит ошибиться, когда
        # оба варианта допустимы», и у интервала по одну сторону нуля этого
        # вопроса нет (`test_the_ceiling_of_the_choice_is_named_only_where_the_
        # interval_crosses_zero`).
        tail = (
            f", интервал через ноль, потолок цены выбора "
            f"{_raw_bb_value(interval.cost_ceiling_bb)}"
            if interval.near_zero
            else ""
        )
        lines.append(
            f"    интервал EV: точка {_raw_signed_bb(interval.point_bb)}, "
            f"от {_raw_signed_bb(interval.low_bb)} до {_raw_signed_bb(interval.high_bb)}"
            f"{tail}"
        )
    assumption = point.assumption
    if assumption is not None:
        share = _fmt_pct(100.0 * assumption.range.fraction_of_hands())
        note = f" · {assumption.note}" if assumption.note else ""
        lines.append(
            f"    допущение: источник {assumption.source} · "
            f"доля рук в диапазоне {share}{note}"
        )
    if point.tools:
        lines.append(f"    инструменты: {', '.join(point.tools)}")
    return lines


def _raw_blocks(res: AnalysisResult, en: EnrichedHand) -> list[list[str]]:
    """Сырые числа раздачи, разбитые на блоки: по блоку на точку решения.

    Шапки раздачи здесь больше нет: номер, уровень, блайнды, стеки, борд, банк и
    исход стоят ровно один раз — в блоке «Что было» (`explanation.hand_replay`,
    решение владельца 2026-10-07,
    `test_the_lines_that_moved_into_the_block_are_not_printed_twice`).

    Блоками, а не одним списком, потому что при переполнении сообщение режется
    ПО ГРАНИЦЕ ТОЧКИ (`hand_analysis_msgs`): половина точки в одном сообщении и
    половина в другом — не диагностика.

    Подпись обязательна у каждого числа (`test_the_raw_data_block_names_what_
    each_number_means`). Ни одной величины, которой не было бы в `EnrichedHand`
    или `AnalysisResult` (`test_the_raw_data_block_prints_no_money_the_hand_
    does_not_contain`).

    Точки идут в порядке раздачи, без ранжирования: ранжирование отвечало на
    вопрос «что разбирать первым», а здесь показывается посчитанное, а не выбор
    из него.
    """
    hand = en.hand
    decisions = {dp.index: dp for dp in en.report.decision_points}
    blocks: list[list[str]] = []
    for point in res.points:
        dp = decisions.get(point.dp_index)
        street = _STREET_WORD.get(point.street, point.street.value)
        line = postflop_line_detail(point)
        block = (
            _decision_lines(dp, hand, line)
            if dp is not None
            else [f"{point.dp_index + 1}. {street}"]
        )
        blocks.append(
            [
                *block,
                *_postflop_line_lines(line, hand.bb),
                *_verdict_lines(point),
                *_detail_lines(point),
            ]
        )
    return blocks


def _raw_tail_lines(res: AnalysisResult, en: EnrichedHand) -> list[str]:
    """Итог раздачи: сумма цены расхождений и чем кончилась сверка денег.

    Сумма печатается только там, где есть хотя бы одна судимая точка С ЦЕНОЙ:
    ноль несчитанной точки означает «не посчитано», а не «сыграно верно», и
    подпись под ним читалась бы как «потерь не было»
    (`test_the_price_is_summed_only_where_something_was_judged`). Точка по чарту
    судима, но цены не имеет (`mismatch` не `None`) — одна она суммы не открывает.
    """
    lines: list[str] = []
    if any(is_judged(point) and point.mismatch is None for point in res.points):
        lines.append(
            f"Сумма цены расхождений: {_raw_signed_bb(res.total_ev_loss_bb)} — "
            f"только по оценённым точкам."
        )
    status = _VALIDATION_WORD.get(en.verdict.status, en.verdict.status.value)
    lines.append(f"Сверка денег с источником: {status}.")
    return lines


def _joined(blocks: Sequence[Sequence[str]]) -> list[str]:
    """Блоки в один список строк, разделённые пустой строкой."""
    lines: list[str] = []
    for block in blocks:
        if lines:
            lines.append("")
        lines.extend(block)
    return lines


def hand_analysis_msgs(
    res: AnalysisResult,
    en: EnrichedHand,
    elapsed_s: int,
    zone: Zone | None,
    quota_left: int,
    quota_total: int,
    *,
    replay: HandReplay,
    dev_line: str | None = None,
    not_checked: Sequence[str] = (),
    note_nicks: Sequence[str] = (),
) -> list[Msg]:
    """Разбор раздачи: блок «Что было», сырые числа расчёта, статус-строка, кнопки.

    Возвращает одно сообщение или два. Второе появляется ровно тогда, когда
    числа не влезли в предел `sendMessage` (4096 символов): первое несёт реплей,
    столько ЦЕЛЫХ точек, сколько поместилось, оговорки, статус и кнопки, второе
    — оставшиеся точки. Резать разбор внутри точки нельзя, а выбрасывать
    посчитанное — тем более
    (`test_a_hand_too_long_for_one_message_goes_out_in_two`).

    **Слов модели здесь нет** (решение владельца 2026-09-12): разбор состоит из
    того, что посчитал код, и ничего больше. Аргумента, куда передать текст
    модели, у функции нет вовсе
    (`test_the_deep_dive_carries_no_words_of_the_model`).

    **`zone=None` — «зоны нет», и тогда её нет и в строке.** Вызывающий,
    которому нечего сказать (ни одной судимой точки), прежде подставлял
    `Zone.STRICT` — самую уверенную подпись продукта. CLAUDE.md разрешает
    `strict` только там, где вывод не опирается на угаданный диапазон; вывода в
    этом случае нет вообще, а значит нет и зоны.

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

    **Блок «Что было:» — первым** (спека §5.6) и обязателен: шапка раздачи живёт
    только в нём, и разбор без блока остался бы без номера руки, стеков и
    исхода. Несжимаемы блок, сверка денег, оговорка и статус-строка; уезжают во
    второе сообщение только точки — при нужде все
    (`test_a_hand_too_long_for_one_message_goes_out_in_two`). Сообщение всегда
    в `parse_mode="HTML"`: выделение точки решения — разметка.

    `note_nicks` — оппоненты, на которых можно записать заметку одним тапом
    (решение владельца 2026-09-04: путь заметки начинается ИЗ РАЗБОРА). Ники
    приходят только со скрина; на HH-пути список пуст, потому что там их нет.
    """
    blocks = _raw_blocks(res, en)
    tail = ["", *_raw_tail_lines(res, en)]
    if not_checked:
        tail += ["", f"{_NOT_CHECKED_PREFIX} {', '.join(not_checked)}."]
    zone_segment = "" if zone is None else f"зона: {_ZONE_WORD[zone]} · "
    tail += ["", f"⏱ {elapsed_s}с · {zone_segment}{_quota_line(quota_left, quota_total)}"]
    if dev_line is not None:
        tail.append(dev_line)

    head = f"Что было:\n{_replay_html(replay)}"
    buttons = [verdict_buttons(res.hand_no), *note_buttons_for_hand(res.hand_no, note_nicks)]

    def rendered(kept: int) -> str:
        return _render_html(head, ["", *_joined(blocks[:kept]), *tail])

    kept = len(blocks)
    while kept > 0 and len(rendered(kept)) > _TELEGRAM_TEXT_LIMIT:
        kept -= 1
    first = Msg(text=rendered(kept), buttons=buttons, parse_mode="HTML")
    if kept == len(blocks):
        return [first]
    head_line = f"Рука {res.hand_no}, продолжение разбора:\n\n"
    rest = _fitted("\n".join(_joined(blocks[kept:])), _TELEGRAM_TEXT_LIMIT - len(head_line))
    return [first, Msg(text=head_line + rest)]


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


def hh_scan_in_progress_msg() -> Msg:
    """Тот же файл СЕЙЧАС считается — второй задачи на него не надо.

    Отказ ровно на время работы, и не дольше. Прежняя версия отклоняла файл при
    любом статусе кроме `failed`, то есть и после успешного разбора: на сессии,
    которая живёт неделями, это означало «файл, разобранный однажды, нельзя
    разобрать никогда», а выходом называла `/new` — разрыв истории ради повтора
    одного файла. Повтор законченного скана безопасен (турнир переиспользуется,
    сохранённые руки пропускают чекпоинты) и теперь просто принимается.

    Про `/new` здесь молчим: ждать нужно секунды, а не начинать что-то новое.
    """
    return Msg(text="Этот файл сейчас считаю — подождите, пришлю сводку, как закончу.")


def bot_failure_msg() -> Msg:
    """Сбой на нашей стороне — короткое честное признание вместо молчания.

    Причина сюда не попадает никогда (ни `str(exc)`, ни путь файла, ни номер
    раздачи): она уходит в лог, как `jobs.error` у воркера. Игроку важно другое
    — что произошло не у него и что попытку имеет смысл повторить.
    """
    return Msg(text="Не получилось обработать запрос — это на нашей стороне. Попробуйте ещё раз.")


def unsupported_document_msg() -> Msg:
    """Документ ни на что не похож — отказ сразу, с называнием ЕДИНСТВЕННОЙ двери.

    Дверь в продукт одна: раздачи приходят файлом `.txt` из PokerCraft. Вторую
    (скриншот стола картинкой) текст называл, пока она была открыта; со скрин-входом
    за границами v1 приглашать в неё значило бы обещать разбор, которого не будет —
    картинка получает свой отказ (`screenshots_not_supported_msg`).
    """
    return Msg(
        text=(
            "Такой файл я не разберу. Раздачи из PokerCraft присылайте файлом .txt "
            "— запущу скан."
        )
    )


def screenshots_not_supported_msg() -> Msg:
    """Экран стола прислан — разбора не будет, и причина названа вслух.

    Скрин-вход отложен решением владельца: ни одна протестированная модель не
    читает стол достаточно точно, чтобы на её числах строить расчёт
    (`docs/superpowers/specs/2026-09-04-vision-open-problems.md`). Отказ говорит
    об этом прямо, потому что продукт продаёт проверенный расчёт: молчаливое
    «не умею» игрок прочёл бы как поломку, а разбор по неверно прочитанным
    числам противоречил бы самому обещанию.

    Следующий шаг назван обязательно — отказ без него оставляет игрока с той же
    картинкой в руках (SESSIONS_UX).
    """
    return Msg(
        text=(
            "Скриншоты столов я не разбираю. Прочитать экран настолько точно, "
            "чтобы считать по нему, пока не выходит ни у одной модели, а считать "
            "по неверным числам хуже, чем не считать вовсе.\n\n"
            "Пришлите файл раздач из PokerCraft (.txt) — по нему разбор точный: "
            f"суммы и карты там записал сам рум. Где взять файл, покажет "
            f"{MENU_TOURNAMENT}."
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
    """Строка типа лика: подпись, частота, цена. Ровно то, что просил владелец.

    У лика по чарту цены нет, и «0.0 bb» читалось бы как «не потеряно»: вместо
    цены строка говорит, что это сверка с чартом.
    """
    head = f"{stat.rule.title} — {stat.count} {_times_word(stat.count)}"
    if stat.rule.spot is SpotKind.OPEN_CHART:
        return f"{head}, по чарту, без цены"
    return f"{head}, {_fmt_bb(-stat.loss_bb)}"


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
    Список приходит с потолком запроса, и обрезка называется вслух — так же,
    как в `scan_summary_msg`
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
        head = (
            "Чаще всего за вечер (по чарту)"
            if summary.top_leak.rule.spot is SpotKind.OPEN_CHART
            else "Дороже всего за вечер"
        )
        lines.append(f"{head}: {_leak_line(summary.top_leak)}.")
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


def _color_label(color: NoteColorRecord) -> str:
    return f"{color.name} — {color.meaning}"


def _note_lines(note: NoteRecord) -> list[str]:
    head = note.nick if note.color is None else f"{_color_label(note.color)} · {note.nick}"
    return [head, _fitted(note.text, MAX_NOTE_TEXT_CHARS)]


def _notes_cut_line(shown: int, total: int) -> str:
    return f"Показаны {shown} из {total} — самые свежие."


def notes_msg(notes: Sequence[NoteRecord], total: int) -> Msg:
    """Экран «Заметки»: наблюдения об оппонентах, свежие первыми.

    Экран показывает и правит, но НЕ заводит новых (решение владельца
    2026-09-04): новую заводит команда `/note`, и экран её называет
    (`test_notes_msg_says_where_a_new_note_starts`).

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
        "Новая заметка — командой /note Ник."
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


def note_prompt_msg(
    nick: str, existing: NoteRecord | None = None, *, append: bool = False
) -> Msg:
    """Просьба написать наблюдение об оппоненте — вход FSM заметки.

    Примеры в тексте — из решения владельца 2026-09-04 дословно: заметка
    фиксирует то, чего не выводится из счётчиков.

    Прежний текст показывается в бюджете `_TELEGRAM_TEXT_LIMIT`: экран правки —
    единственный способ заменить слишком длинную заметку, и упереться в предел
    `sendMessage` он не вправе
    (`test_note_prompt_msg_of_a_long_note_still_fits_one_telegram_message`).
    """
    head = f"Заметка на {nick}."
    effect = (
        "Запись встанет сверху с сегодняшней датой; если место кончится, "
        "уйдут самые старые записи."
        if append
        else "Новый текст заменит прежний."
    )
    ask = (
        "Напишите наблюдение одним сообщением — то, чего не покажет HUD: "
        f"«фолдит на опен», «донкает флоп». {effect}"
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


def note_appended_msg(nick: str) -> Msg:
    return Msg(text=f"Дописал в заметку на {nick}.")


def note_deleted_msg(nick: str) -> Msg:
    return Msg(text=f"Удалил заметку на {nick}.")


def note_color_prompt_msg(note: NoteRecord, colors: Sequence[NoteColorRecord]) -> Msg:
    """Выбор цвета заметки из цветов игрока плюс «без цвета».

    Цветов нет — ни одной кнопки и подсказка, где их задать
    (`test_the_colour_prompt_without_colours_points_at_settings`).
    """
    if not colors:
        return Msg(
            text=(
                f"Цвет заметки на {note.nick}.\n"
                f"Цвета ещё не заданы: задайте их в {MENU_SETTINGS} → "
                f"{note_colors_button().text}."
            )
        )
    choices = [(str(color.color_id), _color_label(color)) for color in colors]
    choices.append((NOTE_COLOR_UNSET, "⚪️ без цвета"))
    return Msg(
        text=f"Цвет заметки на {note.nick}: выберите.",
        buttons=note_color_buttons(note.note_id, choices),
    )


_NOTE_COLORS_HINT = (
    "Напишите свои цвета, по одному в строке: «цвет — что он значит».\n"
    "Например: зелёный — слабый, коллер. Тот же цвет новой строкой меняет подпись. "
    f"Не больше {MAX_NOTE_COLORS} цветов."
)


def note_colors_msg(colors: Sequence[NoteColorRecord], *, saved: int | None = None) -> Msg:
    """Экран «Цвета заметок»: цвета игрока, «🗑» на каждый и просьба написать свои.

    Экран открывает ввод цветов (`bot.handlers`), поэтому подсказка формата на
    нём и есть просьба. `saved` — сколько цветов записал только что принятый
    ввод. Двенадцать самых длинных цветов укладываются в одно сообщение
    (`test_the_colours_screen_of_the_longest_colours_fits_one_telegram_message`).
    """
    lines: list[str] = []
    if saved is not None:
        lines += [f"Записал {saved} {_plural_form(saved, 'цвет', 'цвета', 'цветов')}", ""]
    lines += ["Цвета заметок", ""]
    lines += [_color_label(color) for color in colors] or ["Цветов пока нет."]
    lines += ["", _NOTE_COLORS_HINT]
    return Msg(
        text="\n".join(lines),
        buttons=note_color_delete_buttons([(color.color_id, color.name) for color in colors]),
    )


_NOTE_COLOR_PROBLEMS = {
    "no_separator": "не вижу, где кончается цвет и начинается подпись — поставьте между ними «—»",
    "empty_half": "нужны и цвет, и подпись — одна половина пустая",
    "name_too_long": f"цвет длиннее {MAX_NOTE_COLOR_NAME_CHARS} знаков",
    "meaning_too_long": f"подпись длиннее {MAX_NOTE_COLOR_MEANING_CHARS} знаков",
}

# Сколько знаков отказанной строки показывать: строка приходит из сообщения
# игрока и сама может быть длиннее `_TELEGRAM_TEXT_LIMIT`.
_REFUSED_LINE_CHARS = 200


def note_colors_refused_msg(line_no: int, line: str, problem: str) -> Msg:
    """Строка цветов не разобралась: её номер и текст; не записано ничего."""
    reason = _NOTE_COLOR_PROBLEMS.get(problem, "не разобрал")
    return Msg(
        text=(
            f"Строка {line_no} «{_fitted(line, _REFUSED_LINE_CHARS)}»: {reason}. "
            "Ничего не записал — пришлите цвета ещё раз, исправив эту строку."
        )
    )


def note_colors_too_many_msg(limit: int) -> Msg:
    return Msg(
        text=(
            f"Цветов может быть не больше {limit}. Ничего не записал — удалите лишние "
            "кнопкой «🗑» или пришлите меньше."
        )
    )


def note_color_saved_msg(nick: str, color: NoteColorRecord | None) -> Msg:
    label = "без цвета" if color is None else _color_label(color)
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


def note_usage_msg() -> Msg:
    """`/note` без слов: формат обеих форм команды.

    Правило ника — то, по которому команду разбирает `handlers.handle_note_command`
    (`test_a_note_command_takes_the_whole_first_line_as_a_nick_of_several_words`,
    `test_a_note_command_without_a_line_break_splits_at_the_first_space`).
    """
    return Msg(
        text=(
            "Заметка на оппонента без скриншота.\n\n"
            "Показать заметку и дописать наблюдение следующим сообщением:\n"
            "/note Vasya\n\n"
            "Сразу дописать наблюдение:\n"
            "/note Vasya фолдит на опен\n\n"
            "Без переноса строки ник — первое слово после /note. Ник из нескольких "
            "слов — на первой строке, наблюдение — со второй:\n"
            "/note Big Fish\nфолдит на опен\n\n"
            "Перенос строки делает ником всю первую строку, поэтому наблюдение в "
            "несколько строк начинайте со второй строки, а на первой оставьте только ник."
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
    return Msg(
        text="\n".join(lines),
        buttons=[[set_nickname_button(bool(nickname))], [note_colors_button()]],
    )


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
            "/note НИК — заметка на оппонента без скриншота.\n"
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
    целиком (`test_the_deep_dive_escapes_a_nickname_that_looks_like_a_tag`).
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _replay_html(replay: HandReplay) -> str:
    """Ход раздачи разметкой Телеграма: точка решения героя и названия улиц — жирным.

    Спека §5.6 требует выделить точку решения прямо в потоке действий;
    `explanation.hand_replay` отдаёт куски с флагом `emphasis`, а во что
    превратится выделение — решает этот модуль. Здесь это `<b>` при
    `parse_mode=HTML`, поэтому весь остальной текст экранируется
    (`_html_escape`). Отдельно от `hand_analysis_msgs`, единственного
    вызывающего: разметка блока и сборка сообщения — два разных решения, и
    читаются они порознь.
    """
    return "".join(
        f"<b>{_html_escape(span.text)}</b>"
        if span.emphasis or span.title
        else _html_escape(span.text)
        for span in replay.spans
    )


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
