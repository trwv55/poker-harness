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

from collections.abc import Sequence
from math import ceil, floor
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, model_validator

from harness.contracts.analysis import (
    RIVER_CALL_DETAIL,
    TURN_FLOP_CALL_DETAIL,
    AnalysisResult,
    EvInterval,
    PointVerdict,
    ScanItem,
    ScanSummary,
    SpotKind,
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
from harness.contracts.enriched import (
    DecisionPoint,
    EnrichedHand,
    ValidationStatus,
    hero_stack_delta_bb,
)
from harness.contracts.history import (
    MAX_NOTE_TEXT_CHARS,
    NOTE_COLORS,
    LeaksOverview,
    LeakStat,
    NoteRecord,
    OpponentRecord,
    SessionLine,
    SessionSummary,
    is_judged,
)
from harness.contracts.raw import ActionKind, Street
from harness.explanation.hand_replay import HandReplay, bb, chips
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
    "scan_summary_msg",
    "screenshot_too_large_msg",
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
}

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
    magnitude = round(abs(value_bb), 1)
    if magnitude == 0.0:
        return "0.0"
    return f"{'−' if value_bb < 0 else '+'}{magnitude:.1f}"


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


# Слово игрока для типа анте. Значение, которого в словаре нет, печатается как
# есть: выдумать вместо него нечего, а спрятать нельзя.
_ANTE_TYPE_WORD: dict[str, str] = {"per_player": "с каждого"}

# Слова игрока для машинных значений `detail`. Значение без перевода печатается
# как есть — по той же причине, что и ключ без подписи.
_METHOD_WORD: dict[str, str] = {
    "subset_enumeration": "перебор подмножеств ответивших",
    "call_ev": "EV колла против диапазона шовера",
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
    "branches": "перебранных веток вскрытия",
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
    "equilibrium_hand_regret_bb": "отклонение этой руки от равновесия, ББ",
    "p_all_fold": "вероятность, что все спасуют",
    "expected_callers": "ожидаемое число ответивших",
    "required_equity": "требуемая эквити",
    "shover_depth_bb": "глубина стека шовера, ББ",
    "live_others": "живых за вами в переборе",
    "behind_axis": "устойчивость к входу живых за вами",
    "rivals_when_shoved": "соперников на момент шова",
    "best_vs_one": "лучше против одного диапазона",
    "best_all_behind": "лучше, если входят все живые за вами",
    "push_weight": "вес руки в чарте шова",
    "lookup_depth_bb": "глубина лукапа по чарту, ББ",
    "solver_error": "сбой расчёта",
}

# Ключи `detail`, чьё значение — доля единицы: печатаются процентом, как все
# доли продукта. Дробь и процент в одном сообщении читатель принимает за разные
# величины (`test_a_share_is_printed_as_a_percentage_not_as_a_fraction`).
_SHARE_KEYS = frozenset(
    {
        "required_equity",
        "shove_range_fraction",
        "call_range_fractions",
        "push_weight",
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
_STACK_KEYS = frozenset({"depths_bb", "shover_depth_bb", "lookup_depth_bb"})

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
    выше; остальные ключи печатаются по таблице подписей.
    """
    skip = (RIVER_CALL_DETAIL, TURN_FLOP_CALL_DETAIL, _UNJUDGED_KEY)
    lines = _postflop_call_lines(point)
    for key, value in point.detail.items():
        if key in skip:
            continue
        lines.append(f"    {_DETAIL_LABELS.get(key, key)}: {_keyed_value(key, value)}")
    return lines


def _hand_head_lines(en: EnrichedHand, hand_no: str) -> list[str]:
    """Шапка раздачи: уровень, блайнды, стол, ваши карты и стек, борд, банк, исход."""
    hand = en.hand
    rep = en.report
    ante = (
        f"{hand.ante} фишек ({_ANTE_TYPE_WORD.get(hand.ante_type, hand.ante_type)})"
        if hand.ante
        else "нет"
    )
    lines = [
        (
            f"Рука {hand_no} · уровень {hand.level} · "
            f"блайнды {hand.sb}/{hand.bb} фишек · анте {ante}"
        ),
        f"Игроков в раздаче: {len(hand.players)}.",
    ]
    hero = next((p for p in hand.players if p.label == hand.hero_label), None)
    if hero is not None:
        cards = " ".join(hand.dealt.get(hand.hero_label, [])) or "не известны"
        ended = rep.stacks_end.get(hand.hero_label)
        tail = "" if ended is None else f" → после раздачи {_raw_bb(ended, hand.bb)}"
        lines.append(
            f"Вы: {cards} · позиция {hero.position} · "
            f"стек до раздачи {_raw_bb(hero.stack, hand.bb)}{tail}"
        )
    if hand.boards:
        board = " · ".join(
            f"{_STREET_WORD.get(street, street.value).lower()} {' '.join(cards)}"
            for street, cards in hand.boards.items()
        )
        lines.append(f"Борд по улицам: {board}")
    pots = " · ".join(
        f"{_STREET_WORD.get(street, street.value).lower()} {_raw_bb(value, hand.bb)}"
        for street, value in rep.pot_by_street.items()
    )
    lines.append(f"Банк по улицам (сколько лежало в банке к концу улицы): {pots}.")
    lines.append(f"Конечный банк: {_raw_bb(rep.final_pot, hand.bb)}.")
    if len(rep.side_pots) > 1:
        # Движок кладёт в `side_pots` ВСЕ поты PokerKit, включая главный, поэтому
        # один элемент означает неделёный банк и печатать его нечем
        # (`test_the_hand_with_one_pot_says_nothing_about_side_pots`).
        parts = " · ".join(
            f"{_raw_bb(pot.amount, hand.bb)} (претендуют: {', '.join(pot.eligible)})"
            for pot in rep.side_pots
        )
        lines.append(f"Банк делится на части: {parts}.")
    if hero is not None:
        lines.append(f"Исход раздачи для вас: {_raw_signed_bb(hero_stack_delta_bb(en))}.")
    return lines


def _played_words(dp: DecisionPoint, big_blind: int) -> str:
    """Что сыграно и на какую сумму — каждому действию своё число.

    Колл называет ДОПЛАТУ (`to_call`), бет и рейз — итог, до которого подняли
    (`CanonicalAction.committed_after` — накопленное за улицу), чек и фолд не
    несут суммы вовсе: денег в них нет
    (`test_the_action_of_a_point_prints_the_number_that_belongs_to_it`).
    """
    kind = dp.action.kind
    word = _action_word(kind.value)
    all_in = ", олл-ин" if dp.action.is_all_in else ""
    if kind is ActionKind.CALL:
        return f"{word} {_raw_bb(dp.to_call, big_blind)}{all_in}"
    if kind in (ActionKind.BET, ActionKind.RAISE):
        return f"{word} до {_raw_bb(dp.action.committed_after, big_blind)}{all_in}"
    return f"{word}{all_in}"


def _decision_lines(dp: DecisionPoint, big_blind: int) -> list[str]:
    """Числа точки решения: что сыграно, сколько в банке, доставить, SPR, шансы банка."""
    spr = "—" if dp.spr is None else f"{dp.spr:.1f}"
    lines = [
        (
            f"{dp.index + 1}. {_STREET_WORD.get(dp.street, dp.street.value).lower()} · "
            f"позиция {dp.position} · сыграно: {_played_words(dp, big_blind)}"
        ),
        (
            f"    банк до хода {_raw_bb(dp.pot_before, big_blind)} · "
            f"доставить {_raw_bb(dp.to_call, big_blind)} · "
            f"эфф. стек {_raw_bb(dp.eff_stack, big_blind)} · SPR {spr} · "
            f"живых {dp.live_total} (после вас {dp.live_behind})"
        ),
    ]
    if dp.to_call > 0:
        equity = _fmt_pct(100.0 * _required_equity(dp.to_call, dp.pot_before))
        lines.append(
            f"    шансы банка: доставить {_raw_bb(dp.to_call, big_blind)} "
            f"в банк {_raw_bb(dp.pot_before, big_blind)} — "
            f"колл окупается от {equity} эквити"
        )
    return lines


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
        return [f"    вердикта нет: {reason}" if reason else "    вердикта нет."]
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
    """Сырые числа раздачи, разбитые на блоки: шапка, потом по блоку на точку.

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
    head = _hand_head_lines(en, res.hand_no)
    decisions = {dp.index: dp for dp in en.report.decision_points}
    blocks: list[list[str]] = []
    for point in res.points:
        dp = decisions.get(point.dp_index)
        street = _STREET_WORD.get(point.street, point.street.value).lower()
        block = (
            _decision_lines(dp, hand.bb)
            if dp is not None
            else [f"{point.dp_index + 1}. {street}"]
        )
        blocks.append([*block, *_verdict_lines(point), *_detail_lines(point)])
    return [head, *blocks]


def _raw_tail_lines(res: AnalysisResult, en: EnrichedHand) -> list[str]:
    """Итог раздачи: сумма цены расхождений и чем кончилась сверка денег.

    Сумма печатается только там, где есть хотя бы одна судимая точка: ноль
    несчитанной точки означает «не посчитано», а не «сыграно верно», и подпись
    под ним читалась бы как «потерь не было»
    (`test_the_price_is_summed_only_where_something_was_judged`).
    """
    lines: list[str] = []
    if any(is_judged(point) for point in res.points):
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
    dev_line: str | None = None,
    replay: HandReplay | None = None,
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

    **Блок «Что было» — первым** (спека §5.6). Несжимаемы реплей, сверка денег,
    оговорка и статус-строка; уезжают во второе сообщение только точки.
    `parse_mode="HTML"` только при наличии блока — иначе экранировать пришлось
    бы весь текст всюду.

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

    head = None if replay is None else f"Что было\n{_replay_html(replay)}"
    buttons = [verdict_buttons(res.hand_no), *note_buttons_for_hand(res.hand_no, note_nicks)]

    def rendered(kept: int) -> str:
        lines = ([""] if head is not None else []) + _joined(blocks[:kept]) + tail
        if head is None:
            return "\n".join(lines)
        return _render_html(head, lines)

    kept = len(blocks)
    while kept > 1 and len(rendered(kept)) > _TELEGRAM_TEXT_LIMIT:
        kept -= 1
    first = Msg(
        text=rendered(kept),
        buttons=buttons,
        parse_mode=None if head is None else "HTML",
    )
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
    целиком (`test_the_deep_dive_escapes_a_nickname_that_looks_like_a_tag`).
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _replay_html(replay: HandReplay) -> str:
    """Ход раздачи разметкой Телеграма: точка решения героя — жирным.

    Спека §5.6 требует выделить точку решения прямо в потоке действий;
    `explanation.hand_replay` отдаёт куски с флагом `emphasis`, а во что
    превратится выделение — решает этот модуль. Здесь это `<b>` при
    `parse_mode=HTML`, поэтому весь остальной текст экранируется
    (`_html_escape`). Отдельно от `hand_analysis_msgs`, единственного
    вызывающего: разметка блока и сборка сообщения — два разных решения, и
    читаются они порознь.
    """
    return "".join(
        f"<b>{_html_escape(span.text)}</b>" if span.emphasis else _html_escape(span.text)
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
