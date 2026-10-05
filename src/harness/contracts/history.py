"""Сквозное знание об игроке: сессии, лики, заметки — то, что копится по всей истории.

Не выход конвейера, а данные, которые память (`memory/repos.py`) собирает из уже
сохранённых разборов, а `presentation` печатает. Живут в контрактах по той же
причине, что `ScanItem`/`ScanSummary` (`analysis.py`, round 5, Item L): их читают
два пакета по разные стороны правила зависимостей — `memory` (SQL) и
`presentation` (текст), — и `presentation` не имеет права импортировать `memory`
(CLAUDE.md, правило зависимостей). Общий тип не даёт им разойтись.

**Таксономия ликов — таблица правил, а не код.** Тип лика определён тройкой
«спот · сыгранное действие · лучшее действие» над `PointVerdict`; новый тип —
строка в `LEAK_RULES`, а не ветка в `if`. Так задана постановка владельца
(решение 2026-09-07), и так же устроена находка турнира (`Finding`): та же
тройка, только внутри одного турнира.

**Почему тройка, а не спот.** Группировка по споту (первая версия плана) кладёт
в одну кучу «не шовит, где надо» и «шовит слишком широко» — противоположные
привычки с противоположным лечением. Тройка различает их, потому что в неё
входит направление ошибки.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel

from harness.contracts.analysis import PointVerdict, SpotKind

__all__ = [
    "JUDGED_SPOTS",
    "LEAK_RULES",
    "MAX_NOTE_COLORS",
    "MAX_NOTE_COLOR_MEANING_CHARS",
    "MAX_NOTE_COLOR_NAME_CHARS",
    "MAX_NOTE_ENTRY_CHARS",
    "MAX_NOTE_TEXT_CHARS",
    "MISMATCH_LOSS_BB",
    "LeakRule",
    "LeakStat",
    "LeaksOverview",
    "NoteColorLineError",
    "NoteColorRecord",
    "NoteRecord",
    "OpponentRecord",
    "SessionLine",
    "SessionSummary",
    "append_note_entry",
    "is_judged",
    "is_mismatch",
    "leak_rule_for",
    "leak_rule_of_point",
    "parse_note_colors",
]


class LeakRule(BaseModel, frozen=True):
    """Один тип лика: подпись игроку плюс тройка, по которой он опознаётся.

    `action_taken`/`best_action` — токены движка (`analysis/preflop.py`), не
    слова игрока: сравнение идёт с тем, что записано в `PointVerdict`, а перевод
    в слова делает `presentation`. Точка «около нуля» несёт в `best_action`
    русскую фразу («около нуля, оба варианта допустимы»), а точка без вердикта —
    пустую строку, поэтому ни та, ни другая не совпадают ни с одной строкой
    таблицы (`test_a_near_zero_point_matches_no_leak_rule`,
    `test_an_unjudged_point_matches_no_leak_rule`).
    """

    key: str
    title: str
    spot: SpotKind
    action_taken: str
    best_action: str


# Таксономия v1 — решение владельца 2026-09-07; правила спота `open_chart` —
# 2026-10-03 (спека 2026-10-03-open-chart-verdict, §6). У точки по чарту правило
# опознаётся тройкой только при `is_mismatch`: смешанная рука, сыгранная не самым
# частым действием, расхождением не является и в лики не идёт
# (`leak_rule_of_point`, `memory.repos.LeaksRepo.by_type`).
LEAK_RULES: tuple[LeakRule, ...] = (
    LeakRule(
        key="no_shove",
        title="Не шовит, где надо",
        spot=SpotKind.PUSHFOLD_UNOPENED,
        action_taken="fold",
        best_action="shove",
    ),
    LeakRule(
        key="shove_too_wide",
        title="Шовит слишком широко",
        spot=SpotKind.PUSHFOLD_UNOPENED,
        action_taken="shove",
        best_action="fold",
    ),
    LeakRule(
        key="fold_vs_shove",
        title="Сбрасывает против шова, где колл плюсовой",
        spot=SpotKind.PUSHFOLD_FACING_SHOVE,
        action_taken="fold",
        best_action="call",
    ),
    LeakRule(
        key="call_too_wide",
        title="Коллирует шов слишком широко",
        spot=SpotKind.PUSHFOLD_FACING_SHOVE,
        action_taken="call",
        best_action="fold",
    ),
    LeakRule(
        key="open_not_opened_raise",
        title="Сбрасывает руки, которые чарт открывает рейзом",
        spot=SpotKind.OPEN_CHART,
        action_taken="fold",
        best_action="raise",
    ),
    LeakRule(
        key="open_not_opened_shove",
        title="Сбрасывает руки, которые чарт шовит первым",
        spot=SpotKind.OPEN_CHART,
        action_taken="fold",
        best_action="shove",
    ),
    LeakRule(
        key="open_too_wide",
        title="Открывает рейзом шире чарта",
        spot=SpotKind.OPEN_CHART,
        action_taken="raise",
        best_action="fold",
    ),
    LeakRule(
        key="open_shove_too_wide",
        title="Шовит первым шире чарта",
        spot=SpotKind.OPEN_CHART,
        action_taken="shove",
        best_action="fold",
    ),
    LeakRule(
        key="open_limp_too_wide",
        title="Лимпует руки, которые чарт сбрасывает",
        spot=SpotKind.OPEN_CHART,
        action_taken="limp",
        best_action="fold",
    ),
    LeakRule(
        key="open_shove_instead_of_raise",
        title="Шовит там, где чарт рейзит",
        spot=SpotKind.OPEN_CHART,
        action_taken="shove",
        best_action="raise",
    ),
    LeakRule(
        key="open_raise_instead_of_shove",
        title="Рейзит там, где чарт шовит",
        spot=SpotKind.OPEN_CHART,
        action_taken="raise",
        best_action="shove",
    ),
    LeakRule(
        key="open_not_limped",
        title="Сбрасывает руки, которые чарт лимпует",
        spot=SpotKind.OPEN_CHART,
        action_taken="fold",
        best_action="limp",
    ),
    LeakRule(
        key="open_raise_instead_of_limp",
        title="Рейзит там, где чарт лимпует",
        spot=SpotKind.OPEN_CHART,
        action_taken="raise",
        best_action="limp",
    ),
    LeakRule(
        key="open_limp_instead_of_raise",
        title="Лимпует там, где чарт рейзит",
        spot=SpotKind.OPEN_CHART,
        action_taken="limp",
        best_action="raise",
    ),
)


# Споты, по которым ядро вообще выносит вердикт. Живёт здесь, а не в
# `analysis.error_cost`, потому что читателей у правила трое по разные стороны
# правила зависимостей: ядро (ранжирование), `memory` (колонка `judged`) и
# `presentation` (строка покрытия). `memory` не имеет права тянуть в образ бота
# расчётный стек (`test_bot_image_does_not_import_calculation_stack`), а
# `contracts` не тянут ничего.
JUDGED_SPOTS: frozenset[SpotKind] = frozenset(
    {SpotKind.PUSHFOLD_UNOPENED, SpotKind.PUSHFOLD_FACING_SHOVE, SpotKind.OPEN_CHART}
)

# Порог, с которого потеря ценовой точки — расхождение: дешевле него упрёк не
# стоит внимания игрока (задача 13). Живёт в контрактах, потому что читает его
# единый предикат `is_mismatch`, а за ним — скан и изложение.
MISMATCH_LOSS_BB = 0.1


def is_judged(point: PointVerdict) -> bool:
    """Есть ли по точке вердикт: пустой `best_action` означает «не посчитано».

    ЕДИНСТВЕННАЯ формулировка правила во всей системе. Ядро зовёт её же
    (`analysis.error_cost.is_judged` — это она), память вызывает её при записи
    точки и кладёт ответ в колонку `decision_points.judged`, а SQL читает
    колонку и правила не повторяет
    (`test_the_judged_column_is_written_by_the_one_predicate`).
    """
    return point.spot in JUDGED_SPOTS and point.best_action != ""


def is_mismatch(point: PointVerdict) -> bool:
    """Расхождение ли эта точка — ЕДИНСТВЕННАЯ формулировка правила.

    Точка по чарту несёт ответ сама (`PointVerdict.mismatch`): цены у неё нет.
    Ценовая точка — расхождение, если судима и потеряла больше `MISMATCH_LOSS_BB`.
    """
    if point.mismatch is not None:
        return point.mismatch
    return is_judged(point) and point.ev_diff_bb < -MISMATCH_LOSS_BB


def leak_rule_for(spot: SpotKind | str, action_taken: str, best_action: str) -> LeakRule | None:
    """Тип лика по тройке или `None`, если ни одно правило не совпало.

    Тройкой, а не готовой `PointVerdict`, потому что второй вызывающий — память:
    она группирует точки в SQL и получает те же три строки из jsonb
    (`memory.repos.LeaksRepo`). Одна реализация правила на оба пути; для точки в
    руках есть тонкая обёртка `leak_rule_of_point`.

    `SpotKind` — `StrEnum`, поэтому сравнение со строкой из jsonb работает без
    приведения типов (`test_a_leak_rule_is_found_by_the_raw_strings_of_jsonb`).
    """
    for rule in LEAK_RULES:
        if (
            rule.spot == spot
            and rule.action_taken == action_taken
            and rule.best_action == best_action
        ):
            return rule
    return None


def leak_rule_of_point(point: PointVerdict) -> LeakRule | None:
    """Тип лика этой точки решения — та же таблица, взятая по полям вердикта.

    Точка по чарту, сыгранная в пределах чарта (`mismatch is False`), лика не
    даёт, даже если её тройка есть в таблице.
    """
    if point.mismatch is False:
        return None
    return leak_rule_for(point.spot, point.action_taken, point.best_action)


class LeakStat(BaseModel):
    """Сколько раз тип лика встретился и во сколько обошёлся.

    `loss_bb` — ПОЛОЖИТЕЛЬНАЯ величина потери («столько ушло»), как в `EvSplit`:
    знак ставит изложение, а не контракт. Складывается из `ev_diff_bb` точек,
    которые совпали с правилом.
    """

    rule: LeakRule
    count: int
    loss_bb: float


class LeaksOverview(BaseModel):
    """Экран «Мои лики» целиком: покрытие сверху, типы ликов под ним.

    `points_judged`/`points_total` — то же покрытие, что у сводки скана
    (`ScanSummary`), только за всю историю игрока. Без него список ликов
    читается как полная картина игры, хотя судится сегодня лишь часть точек.
    """

    points_judged: int
    points_total: int
    leaks: list[LeakStat]


class SessionLine(BaseModel):
    """Строка списка сессий: чем она подписана и открыта ли она.

    Ни рук, ни цены: агрегат сессии считается ПО ЗАПРОСУ (решение владельца
    2026-09-07), и список нарочно не платит за него на каждой строке.
    """

    session_id: int
    title: str
    started_at: datetime
    is_active: bool


class SessionSummary(BaseModel):
    """Агрегат одного вечера: турниры, руки, цена расхождений, лик вечера.

    `loss_bb` — положительная величина, как у `LeakStat`. `top_leak` пуст, когда
    ни одна точка сессии не совпала ни с одним правилом таксономии.
    """

    session_id: int
    title: str
    tournaments: int
    hands: int
    loss_bb: float
    points_judged: int
    points_total: int
    top_leak: LeakStat | None = None


class NoteColorRecord(BaseModel, frozen=True):
    """Цвет заметки, названный игроком: имя и подпись его словами.

    Набора по умолчанию нет: у нового игрока цветов нет вовсе, каждый игрок
    называет свои (`memory.repos.NoteColorsRepo`).
    """

    color_id: int
    name: str
    meaning: str


# Пределы одного цвета — те же, что у колонок `note_colors.name`/`meaning`.
MAX_NOTE_COLOR_NAME_CHARS = 32
MAX_NOTE_COLOR_MEANING_CHARS = 120

# Сколько цветов у игрока самое большее: экран «Цвета заметок» с кнопкой
# удаления на каждый обязан уйти одним сообщением
# (`test_the_colours_screen_of_the_longest_colours_fits_one_telegram_message`).
MAX_NOTE_COLORS = 12

# Разделители имени и подписи. Дефис — только с пробелами по бокам: иначе
# «тёмно-зелёный — слабый» резалось бы по дефису
# (`test_a_hyphen_inside_a_word_does_not_split_the_colour`).
_NOTE_COLOR_SEPARATORS = ("—", "–", ":", " - ")


class NoteColorLineError(ValueError):
    """Строка цветов, которую разбор не принял: её номер, текст и что с ней не так.

    `problem` — `no_separator`, `empty_half`, `name_too_long` или
    `meaning_too_long`; слова игроку подбирает `presentation`.
    """

    def __init__(self, line_no: int, line: str, problem: str) -> None:
        super().__init__(f"строка {line_no} ({problem}): {line!r}")
        self.line_no = line_no
        self.line = line
        self.problem = problem


def parse_note_colors(text: str) -> list[tuple[str, str]]:
    """Цвета из сообщения игрока: по одному в строке, «имя — подпись».

    Пустые строки пропускаются; строку режет первое по месту вхождение любого
    из `_NOTE_COLOR_SEPARATORS` (`test_the_first_separator_of_a_line_wins`), обе
    половины обрезаются от пробелов. Плохая строка — `NoteColorLineError` на
    весь набор: из сообщения не возвращается ничего
    (`test_a_line_without_a_separator_is_refused_with_its_number_and_text`).

    Одно имя дважды без учёта регистра — последняя подпись, первые место и
    написание (`test_the_same_colour_twice_in_one_message_keeps_the_last_meaning`).
    """
    parsed: dict[str, tuple[str, str]] = {}
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        cuts = [
            (position, separator)
            for separator in _NOTE_COLOR_SEPARATORS
            if (position := line.find(separator)) >= 0
        ]
        if not cuts:
            raise NoteColorLineError(line_no, line, "no_separator")
        position, separator = min(cuts)
        name = line[:position].strip()
        meaning = line[position + len(separator) :].strip()
        if not name or not meaning:
            raise NoteColorLineError(line_no, line, "empty_half")
        if len(name) > MAX_NOTE_COLOR_NAME_CHARS:
            raise NoteColorLineError(line_no, line, "name_too_long")
        if len(meaning) > MAX_NOTE_COLOR_MEANING_CHARS:
            raise NoteColorLineError(line_no, line, "meaning_too_long")
        first = parsed.get(name.lower())
        parsed[name.lower()] = (first[0] if first else name, meaning)
    return list(parsed.values())


# Потолок длины одной заметки. Экран «Заметки» — одно сообщение Телеграма, а
# `sendMessage` жёстко ограничен 4096 символами: заметка, которая одна не влезает
# в этот предел, делает экран неоткрываемым вместе с кнопками правки и удаления,
# то есть неисправимым изнутри продукта. 500 держит и этот предел
# (`test_notes_msg_of_the_longest_notes_still_fits_one_telegram_message`), и то,
# чем заметка является: одно наблюдение об оппоненте, а не запись раздачи.
MAX_NOTE_TEXT_CHARS = 500

# Запись дополнения начинается с даты «ДД.ММ: » — семь знаков из того же потолка.
_NOTE_ENTRY_DATE_CHARS = len("00.00: ")

# Предел одной записи: столько остаётся от заметки после даты. Длиннее запись не
# поместилась бы в заметку даже одна, без единой старой строки рядом.
MAX_NOTE_ENTRY_CHARS = MAX_NOTE_TEXT_CHARS - _NOTE_ENTRY_DATE_CHARS


def append_note_entry(existing: str | None, entry: str, day: date) -> str:
    """Дописать наблюдение в заметку: новая строка с датой сверху, старые под ней.

    Запись сводится к одной строке (пробелы и переносы внутри — в один пробел),
    потому что вытесняются заметки построчно: перенос внутри записи разрезал бы
    её надвое, и половина ушла бы раньше другой
    (`test_a_note_entry_is_kept_on_one_line`).

    Не влезает в `MAX_NOTE_TEXT_CHARS` — уходят нижние строки, то есть самые
    старые записи, пока остальное не поместится
    (`test_the_oldest_lines_leave_when_the_note_outgrows_its_limit`). Текст,
    записанный заменой целиком, оказывается под датированными строками и
    вытесняется так же, построчно снизу — многострочный уходит по частям
    (`test_a_replaced_text_sits_under_the_entries_and_leaves_line_by_line`).

    Дата — день `day` как есть; какой это день, решает вызывающий.
    """
    line_body = " ".join(entry.split())
    if not line_body:
        raise ValueError("запись заметки не может быть пустой")
    if len(line_body) > MAX_NOTE_ENTRY_CHARS:
        raise ValueError(f"запись заметки длиннее {MAX_NOTE_ENTRY_CHARS} символов")
    lines = [f"{day:%d.%m}: {line_body}"]
    if existing:
        lines.extend(existing.splitlines())
    while len("\n".join(lines)) > MAX_NOTE_TEXT_CHARS:
        lines.pop()
    return "\n".join(lines)


class NoteRecord(BaseModel):
    """Заметка на оппонента: ник, цвет (`None` — без цвета), текст, когда обновлена."""

    note_id: int
    nick: str
    color: NoteColorRecord | None
    text: str
    updated_at: datetime


class OpponentRecord(BaseModel):
    """Оппонент, которого владелец опознал по нику в руме, и его привязки.

    `nick` — ник в руме: тот же ключ личности, что у заметки
    (`NoteRecord.nick`), а не произвольный ярлык. Ярлык вроде «агрессор слева»
    описывает место за столом, а не человека: в другом турнире слева сидит
    другой (решение владельца).

    `links` — сколько пар «турнир + идентификатор участника» к нику привязано.
    Число нужно на экране: в файлах раздач участник обезличен, и другого
    признака, что привязка состоялась, у владельца нет.

    Связь утверждает только владелец: ни одна привязка в этом проекте не
    заводится по догадке системы — ни по стилю, ни по стеку, ни по совпадению
    чего бы то ни было.
    """

    opponent_id: int
    nick: str
    links: int
