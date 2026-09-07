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

from datetime import datetime

from pydantic import BaseModel

from harness.contracts.analysis import PointVerdict, SpotKind

__all__ = [
    "JUDGED_SPOTS",
    "LEAK_RULES",
    "NOTE_COLORS",
    "NOTE_COLOR_NONE",
    "LeakRule",
    "LeakStat",
    "LeaksOverview",
    "NoteColor",
    "NoteRecord",
    "SessionLine",
    "SessionSummary",
    "leak_rule_for",
    "leak_rule_of_point",
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


# Таксономия v1 — решение владельца 2026-09-07, выведенное из того, что разбор
# умеет судить сегодня (`analysis/error_cost.py::is_judged`: только пуш-фолд в
# неоткрытом банке и колл чужого шова).
#
# Последняя строка — ЗАРЕЗЕРВИРОВАНА под чарты открытия: спот `preflop_other`
# сегодня не судится вовсе, поэтому правило не может совпасть ни с одной точкой,
# которую производит ядро (`test_the_reserved_open_raise_leak_matches_nothing_
# the_core_judges_today`). Она стоит здесь не как мёртвый код, а как образец
# формы: новый тип лика — строка, а не ветка.
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
        key="open_too_wide",
        title="Открывает слишком широко",
        spot=SpotKind.PREFLOP_OTHER,
        action_taken="raise",
        best_action="fold",
    ),
)


# Споты, по которым ядро вообще выносит вердикт (`analysis.error_cost.
# _JUDGED_SPOTS`). Продублировано строкой, а не импортом приватного имени чужого
# модуля, — тот же приём и та же причина, что у `_VALIDATOR_FIELD_BUTTON` в
# `worker/pipeline.py`, плюс своя: `memory` считает покрытие в SQL и не имеет
# права тянуть в образ бота расчётный стек (`test_bot_image_does_not_import_
# calculation_stack`). Разойдутся — покраснеет
# `test_the_judged_spots_of_history_agree_with_the_core`.
JUDGED_SPOTS: frozenset[SpotKind] = frozenset(
    {SpotKind.PUSHFOLD_UNOPENED, SpotKind.PUSHFOLD_FACING_SHOVE}
)


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
    """Тип лика этой точки решения — та же таблица, взятая по полям вердикта."""
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


class NoteColor(BaseModel, frozen=True):
    """Цветовой архетип заметки: ключ в БД и то, как он выглядит на экране."""

    key: str
    label: str


# Цветовая разметка заметок (ARCHITECTURE §6: «двухслойная разметка — цветовые
# архетипы плюс текстовые аннотации»). Набор — решение реализации, не владельца:
# четыре архетипа плюс «без цвета» покрывают то, ради чего цвет и заводится —
# узнать оппонента за секунду до решения. `NOTE_COLOR_NONE` — значение по
# умолчанию: заметка создаётся одним сообщением, цвет ставится потом.
NOTE_COLOR_NONE = "none"

NOTE_COLORS: tuple[NoteColor, ...] = (
    NoteColor(key=NOTE_COLOR_NONE, label="⚪️ без цвета"),
    NoteColor(key="red", label="🔴 агрессор"),
    NoteColor(key="yellow", label="🟡 лузовый"),
    NoteColor(key="green", label="🟢 слабый"),
    NoteColor(key="blue", label="🔵 тайтовый"),
)


class NoteRecord(BaseModel):
    """Заметка на оппонента: ник, цвет, текст, когда обновлена."""

    note_id: int
    nick: str
    color: str
    text: str
    updated_at: datetime
