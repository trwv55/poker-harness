"""Лукап опен-чартов: `(seats, position, ante_type)` и стек → стратегия ближайшей глубины.

Чарт — **эталон**, а не результат обучения на руках игрока: он известен до первой
загруженной руки, и выводить его из сыгранных рук нельзя (это зафиксировало бы
лики игрока как норму). Накопленные руки дают ДРУГОЕ — измеренные частоты поля
(§4 спеки `docs/superpowers/specs/2026-09-06-open-raise-model.md`), то есть то, с
чем эталон сравнивают, а не то, из чего его получают.

Механика ровно такая: структурный лукап по файлу в репозитории. Ни эмбеддингов,
ни поиска похожего, ни расчёта — файл читается, ключ ищется сравнением.

**Глубина — ближайшая снятая, всё остальное — точно** (схема 3, решение владельца
2026-10-02). Каждая запись несёт `depth_bb` — стек, на котором чарт снят в
солвере. Для стека игрока берётся запись той же раскладки, позиции и типа анте с
ближайшей `depth_bb`; при равном расстоянии — меньшая глубина. Чарт не
интерполируется: между 30 и 40bb ширина открытия у солвера не монотонна, и
усреднение двух чартов выдало бы за эталон стратегию, которой солвер не давал.
Стек глубже самого глубокого чарта судится самым глубоким.

**Подстановки соседней позиции, раскладки или типа анте нет и не будет.** Нет
записи с такими `(seats, position, ante_type)` — `ChartMissing`; вызывающая сторона
обязана отказаться от вердикта, а не получить «похожий» диапазон. Стол на 4–7 мест
лукап тоже не переносит сам: вызывающая сторона явно спрашивает 8-max ключ позиции с
тем же числом игроков позади (`open_chart.chart_table`, решение владельца 2026-10-10,
`test_the_chart_table_counts_the_players_behind`).

Чартов ниже 15bb нет: там эталон вычисляется равновесием (`analysis.tools.pushfold`).
Лукап судит стек от `MIN_LOOKUP_DEPTH_BB` (13bb, решение владельца 2026-10-03:
открытие на 13–15bb судит чарт 15bb); ниже поднимает `DepthNotCharted`, а не
возвращает самый мелкий чарт.

**Стратегия открытия — три действия, не одно** (схема 2, решение владельца
2026-10-02). На 15–20bb солвер открывает часть рук мин-рейзом, часть олл-ином, а
SB вдобавок лимпует; один диапазон «открытия» потерял бы, КАК рука открывается.
Запись несёт `raise`, `allin` и `limp` в компактной записи; рука, которой нет ни в
одном, — фолд. Доли одной руки по трём действиям в сумме не больше 1.

Что этот модуль НЕ проверяет: разумность самого чарта. Монотонность по глубине, по
позиции, ширина диапазона — не его дело: чарты владельца, код их не судит.
Проверяется только форма: известный ключ, известные классы, вес в [0,1], сумма
долей руки не больше 1, заявленный `open_pct` совпадает с посчитанным, заполненные
`source` и `revised_at` — список проверок пинит параметризованный
`test_malformed_file_is_rejected`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from harness.analysis.charts.notation import NotationError, parse_range
from harness.contracts import Range
from harness.normalizer import POSITIONS_BY_COUNT

SCHEMA_VERSION = 3

# Сумма долей руки по трём действиям допускает округление записи до сотых.
_SUM_TOLERANCE = 1e-9

# Заявленный `open_pct` сверяется с посчитанной долей «рейз + олл-ин» до десятой
# процента — так он записан в солвере. Это ловит опечатку при переносе чарта, а не
# судит сам чарт.
_OPEN_PCT_TOLERANCE = 0.1
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")

DEFAULT_CHART_PATH = Path(__file__).parent / "data" / "open_8max.json"

# Ниже — зона пуш-фолда: эталон там считается равновесием, а не берётся из чарта.
MIN_CHART_DEPTH_BB: float = 15.0

# С какого стека справочник вообще судит: ниже — эталон пуш-фолда. Ниже самой
# мелкой снятой глубины (15bb) стек судится ближайшим, то есть 15bb-чартом:
# решение владельца 2026-10-03 — чарт 15bb забирает открытие и на 13–15bb (позже
# граница опустится до 10–12bb). Классификатор берёт границу отсюда же
# (`classifier.OPEN_CHART_MIN_EFF_BB`), чтобы спот и лукап не разошлись.
MIN_LOOKUP_DEPTH_BB: float = 13.0


class ChartError(Exception):
    """Общий предок отказов справочника — чтобы отказ ловился одним `except`."""


class ChartFileError(ChartError):
    """Файл чартов не читается или не проходит проверку формы."""


class ChartMissing(ChartError):
    """Записи для таких раскладки, позиции и анте нет. Соседняя НЕ подставляется."""


class ChartPlaceholder(ChartError):
    """Запись помечена `status="example"` — образец формата, не чарт."""


class DepthNotCharted(ChartError):
    """Стек ниже 15bb — зона пуш-фолда, там эталон — равновесие, а не чарт."""


class ChartKey(NamedTuple):
    seats: int
    position: str
    depth_bb: float
    ante_type: str


@dataclass(frozen=True)
class OpenStrategy:
    """Эталонная стратегия открытия: доля каждого действия для каждого класса.

    Класс, не названный ни одним действием, — фолд целиком.
    """

    raise_range: Range
    allin_range: Range
    limp_range: Range

    @property
    def opening(self) -> Range:
        """Открытие с агрессией: рейз и олл-ин вместе (лимп сюда не входит)."""
        classes = {*self.raise_range.weights, *self.allin_range.weights}
        return Range(
            weights={
                cls: min(1.0, round(self.raise_range.weight(cls) + self.allin_range.weight(cls), 6))
                for cls in classes
            }
        )

    def fold_weight(self, cls: str) -> float:
        """Доля фолда класса: всё, что не рейз, не олл-ин и не лимп."""
        played = (
            self.raise_range.weight(cls) + self.allin_range.weight(cls) + self.limp_range.weight(cls)
        )
        return max(0.0, 1.0 - played)


class ChartEntry(BaseModel):
    """Одна запись файла: ключ, провенанс и три действия в компактной записи.

    `extra="forbid"`: опечатка в имени поля — ошибка загрузки, а не тихо
    проигнорированное поле (пин `test_malformed_file_is_rejected`, случай
    «опечатка в имени поля»). Поле `raise` в файле — `raise_` в коде: `raise` —
    ключевое слово Python.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    seats: int
    position: str
    depth_bb: float
    ante_type: str
    source: str
    revised_at: date
    status: Literal["chart", "example"] = "chart"
    raise_: str = Field(default="", alias="raise")
    allin: str = ""
    limp: str = ""
    open_pct: str | None = None

    @field_validator("ante_type", "source")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("поле обязательно и не может быть пустым")
        return value

    @field_validator("depth_bb")
    @classmethod
    def _charted_depth(cls, value: float) -> float:
        if not value >= MIN_CHART_DEPTH_BB:
            raise ValueError(
                f"глубина чарта {value}bb ниже {MIN_CHART_DEPTH_BB}bb — там эталон "
                f"считается равновесием пуш-фолда"
            )
        return value

    @model_validator(mode="after")
    def _check_shape(self) -> ChartEntry:
        if self.seats not in POSITIONS_BY_COUNT:
            raise ValueError(f"нет раскладки позиций для {self.seats} мест")
        if self.position not in POSITIONS_BY_COUNT[self.seats]:
            raise ValueError(
                f"позиция {self.position!r} не встречается за столом на {self.seats} мест: "
                f"{POSITIONS_BY_COUNT[self.seats]}"
            )
        if not (self.raise_.strip() or self.allin.strip()):
            raise ValueError("открытия нет: пусты и raise, и allin")
        return self

    @property
    def key(self) -> ChartKey:
        return ChartKey(self.seats, self.position, self.depth_bb, self.ante_type)


class ChartFileModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    readme: list[str] = []
    entries: list[ChartEntry]


class ChartBook:
    """Разобранный файл чартов: точный ключ либо ближайшая снятая глубина."""

    def __init__(
        self, path: Path, entries: dict[ChartKey, tuple[ChartEntry, OpenStrategy]]
    ) -> None:
        self.path = path
        self._entries = entries

    def all_keys(self) -> list[ChartKey]:
        """Все ключи файла, включая образцы формата (их `get` не отдаёт)."""
        return list(self._entries)

    def nearest(self, seats: int, position: str, eff_bb: float, ante_type: str) -> ChartKey:
        """Ключ чарта той же раскладки, позиции и анте с глубиной, ближайшей к стеку.

        При равном расстоянии — меньшая глубина. `DepthNotCharted` — стек ниже
        `MIN_LOOKUP_DEPTH_BB` (или не число); `ChartMissing` — для такой раскладки,
        позиции и анте нет ни одной глубины.
        """
        if not eff_bb >= MIN_LOOKUP_DEPTH_BB:
            raise DepthNotCharted(
                f"{eff_bb}bb ниже {MIN_LOOKUP_DEPTH_BB}bb — это зона пуш-фолда, "
                f"эталон там считается равновесием (analysis.tools.pushfold), а не чартом"
            )
        candidates = [
            key
            for key in self._entries
            if key.seats == seats and key.position == position and key.ante_type == ante_type
        ]
        if not candidates:
            raise ChartMissing(
                f"нет ни одного чарта для (seats={seats}, position={position!r}, "
                f"ante_type={ante_type!r}) в {self.path}; соседняя позиция, раскладка "
                f"или тип анте не подставляются"
            )
        return min(candidates, key=lambda key: (abs(key.depth_bb - eff_bb), key.depth_bb))

    def entry(self, key: ChartKey) -> ChartEntry:
        """Запись по ключу — вместе с провенансом. Отказы те же, что у `get`."""
        entry, _ = self._require(key)
        return entry

    def get(self, key: ChartKey) -> OpenStrategy:
        """Стратегия открытия по ТОЧНОМУ ключу.

        `ChartMissing` — записи нет (соседняя не подставляется);
        `ChartPlaceholder` — запись помечена образцом формата.
        """
        _, strategy = self._require(key)
        return strategy

    def _require(self, key: ChartKey) -> tuple[ChartEntry, OpenStrategy]:
        found = self._entries.get(key)
        if found is None:
            raise ChartMissing(
                f"нет чарта по ключу {tuple(key)} в {self.path}; "
                f"подстановка похожего чарта запрещена. "
                f"Для (seats={key.seats}, position={key.position!r}, "
                f"ante_type={key.ante_type!r}) в файле есть глубины: "
                f"{self._depths_for(key) or 'ни одной'}"
            )
        entry, strategy = found
        if entry.status == "example":
            raise ChartPlaceholder(
                f"запись {tuple(key)} помечена status='example' — это образец формата, "
                f"а не чарт (source: {entry.source!r}); загрузчик её не отдаёт"
            )
        return entry, strategy

    def _depths_for(self, key: ChartKey) -> list[float]:
        return sorted(
            k.depth_bb
            for k in self._entries
            if k.seats == key.seats
            and k.position == key.position
            and k.ante_type == key.ante_type
        )


_CACHE: dict[tuple[Path, int, int], ChartBook] = {}


def load_chart_book(path: Path | None = None) -> ChartBook:
    """Прочитать и проверить файл чартов (по умолчанию — файл в репозитории).

    Кэш держится по (путь, mtime, размер): перезаписанный файл перечитывается,
    а не отдаётся из памяти устаревшим.
    """
    target = (path or DEFAULT_CHART_PATH).resolve()
    try:
        stat = target.stat()
    except OSError as exc:
        raise ChartFileError(f"файл чартов не читается: {target} ({exc})") from exc
    cache_key = (target, stat.st_mtime_ns, stat.st_size)
    cached = _CACHE.get(cache_key)
    if cached is None:
        cached = _parse_file(target)
        _CACHE[cache_key] = cached
    return cached


def chart_keys(path: Path | None = None) -> list[ChartKey]:
    """Ключи файла — для диагностики и для отчётов о покрытии справочника."""
    return load_chart_book(path).all_keys()


def open_strategy(
    seats: int,
    position: str,
    eff_bb: float,
    ante_type: str,
    *,
    path: Path | None = None,
) -> OpenStrategy:
    """Эталонная стратегия открытия для стека: чарт ближайшей снятой глубины.

    Отказы — см. `ChartBook.nearest` и `ChartBook.get`. Какой чарт взят, называет
    `ChartBook.nearest` — вызывающая сторона показывает его глубину игроку.
    """
    book = load_chart_book(path)
    return book.get(book.nearest(seats, position, eff_bb, ante_type))


def open_range(
    seats: int,
    position: str,
    eff_bb: float,
    ante_type: str,
    *,
    path: Path | None = None,
) -> Range:
    """Открытие с агрессией (рейз + олл-ин) для стека — `OpenStrategy.opening`."""
    return open_strategy(seats, position, eff_bb, ante_type, path=path).opening


def _parse_file(path: Path) -> ChartBook:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ChartFileError(f"файл чартов не читается: {path} ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise ChartFileError(f"файл чартов не разобран как JSON: {path} ({exc})") from exc

    try:
        model = ChartFileModel.model_validate(payload)
    except ValidationError as exc:
        raise ChartFileError(f"файл чартов не прошёл проверку формы: {path}\n{exc}") from exc

    if model.schema_version != SCHEMA_VERSION:
        raise ChartFileError(
            f"версия схемы файла {model.schema_version}, код читает {SCHEMA_VERSION}: {path}"
        )

    entries: dict[ChartKey, tuple[ChartEntry, OpenStrategy]] = {}
    for entry in model.entries:
        if entry.key in entries:
            raise ChartFileError(f"ключ {tuple(entry.key)} встречается дважды: {path}")
        entries[entry.key] = (entry, _strategy_of(entry, path))
    if not entries:
        raise ChartFileError(f"в файле чартов нет ни одной записи: {path}")
    return ChartBook(path, entries)


def _strategy_of(entry: ChartEntry, path: Path) -> OpenStrategy:
    """Три действия записи — с проверкой суммы долей и заявленного процента."""
    where = f"запись {tuple(entry.key)} ({path})"
    strategy = OpenStrategy(
        raise_range=_range_of(entry.raise_, "raise", where),
        allin_range=_range_of(entry.allin, "allin", where),
        limp_range=_range_of(entry.limp, "limp", where),
    )
    over = sorted(
        cls
        for cls in {
            *strategy.raise_range.weights,
            *strategy.allin_range.weights,
            *strategy.limp_range.weights,
        }
        if strategy.raise_range.weight(cls)
        + strategy.allin_range.weight(cls)
        + strategy.limp_range.weight(cls)
        > 1.0 + _SUM_TOLERANCE
    )
    if over:
        raise ChartFileError(f"{where}: доли действий в сумме больше 1 у {', '.join(over)}")
    if entry.open_pct is not None:
        declared = _PERCENT.search(entry.open_pct)
        if declared is None:
            raise ChartFileError(f"{where}: open_pct {entry.open_pct!r} не содержит процента")
        counted = 100.0 * (
            strategy.raise_range.fraction_of_hands() + strategy.allin_range.fraction_of_hands()
        )
        if abs(float(declared.group(1)) - counted) > _OPEN_PCT_TOLERANCE:
            raise ChartFileError(
                f"{where}: заявлено открытие {declared.group(1)}%, по диапазонам "
                f"raise + allin выходит {counted:.2f}% — опечатка при переносе?"
            )
    return strategy


def _range_of(text: str, field: str, where: str) -> Range:
    """Диапазон одного действия; пустая строка — действие не используется.

    Неизвестный класс, вес вне [0,1], повтор класса — отказ загрузки (грамматика
    `parse_range`). Класс, не названный записью, имеет вес 0 (контракт `Range`).
    """
    if not text.strip():
        return Range(weights={})
    try:
        return parse_range(text)
    except (NotationError, ValidationError, ValueError) as exc:
        raise ChartFileError(f"{where}: диапазон {field} не принят: {exc}") from exc
