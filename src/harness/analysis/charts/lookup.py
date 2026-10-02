"""Лукап опен-чартов: точный ключ `(seats, position, depth_bucket, ante_type)` → стратегия.

Чарт — **эталон**, а не результат обучения на руках игрока: он известен до первой
загруженной руки, и выводить его из сыгранных рук нельзя (это зафиксировало бы
лики игрока как норму). Накопленные руки дают ДРУГОЕ — измеренные частоты поля
(§4 спеки `docs/superpowers/specs/2026-09-06-open-raise-model.md`), то есть то, с
чем эталон сравнивают, а не то, из чего его получают.

Механика ровно такая: структурный лукап по файлу в репозитории. Ни эмбеддингов,
ни поиска похожего, ни расчёта — файл читается, ключ ищется точным сравнением.

**Подстановки соседнего чарта нет и не будет.** Нет записи по ключу — `ChartMissing`
с названным ключом; вызывающая сторона обязана отказаться от вердикта, а не
получить «похожий» диапазон. Отдать чарт с соседней глубины или с соседней позиции
значило бы выдать за эталон то, чего в справочнике нет, — ровно тот отказ, ради
предотвращения которого продукт и существует.

Глубины ≤ 15bb здесь нет по построению: там эталон вычисляется равновесием
(`analysis.tools.pushfold`), справочник не нужен. `depth_bucket_for` на такой
глубине поднимает `DepthNotCharted`, а не возвращает нижнюю корзину.

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
from math import inf
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

SCHEMA_VERSION = 2

# Сумма долей руки по трём действиям допускает округление записи до сотых.
_SUM_TOLERANCE = 1e-9

# Заявленный `open_pct` сверяется с посчитанной долей «рейз + олл-ин» до десятой
# процента — так он записан в солвере. Это ловит опечатку при переносе чарта, а не
# судит сам чарт.
_OPEN_PCT_TOLERANCE = 0.1
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")

DEFAULT_CHART_PATH = Path(__file__).parent / "data" / "open_8max.json"

# Корзины глубины — из спецификации владельца, границы полуоткрытые: [lo, hi).
# Стек ровно на границе уходит в ВЕРХНЮЮ корзину (20.0 → "20-30"), иначе 20bb
# принадлежал бы двум корзинам сразу.
DEPTH_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("15-20", 15.0, 20.0),
    ("20-30", 20.0, 30.0),
    ("30-40", 30.0, 40.0),
    ("40-60", 40.0, 60.0),
    ("60+", 60.0, inf),
)
BUCKET_NAMES: tuple[str, ...] = tuple(name for name, _, _ in DEPTH_BUCKETS)
MIN_CHART_DEPTH_BB: float = DEPTH_BUCKETS[0][1]


class ChartError(Exception):
    """Общий предок отказов справочника — чтобы отказ ловился одним `except`."""


class ChartFileError(ChartError):
    """Файл чартов не читается или не проходит проверку формы."""


class ChartMissing(ChartError):
    """Записи по точному ключу нет. Похожая НЕ подставляется."""


class ChartPlaceholder(ChartError):
    """Запись помечена `status="example"` — образец формата, не чарт."""


class DepthNotCharted(ChartError):
    """Глубина вне корзин справочника (≤ 15bb — зона пуш-фолда, там равновесие)."""


class ChartKey(NamedTuple):
    seats: int
    position: str
    depth_bucket: str
    ante_type: str


def depth_bucket_for(eff_bb: float) -> str:
    """Имя корзины по эффективному стеку; границы полуоткрытые: [lo, hi).

    Пины теста `test_depth_bucket_edges_are_half_open`: 15.0 → "15-20", 19.99 →
    "15-20", 20.0 → "20-30", 60.0 и 1000.0 → "60+". Ниже 15bb — `DepthNotCharted`.
    """
    if eff_bb < MIN_CHART_DEPTH_BB:
        raise DepthNotCharted(
            f"{eff_bb}bb ниже {MIN_CHART_DEPTH_BB}bb — это зона пуш-фолда, "
            f"эталон там считается равновесием (analysis.tools.pushfold), а не чартом"
        )
    for name, lo, hi in DEPTH_BUCKETS:
        if lo <= eff_bb < hi:
            return name
    # Сюда попадает только не-число: NaN ложен во всех сравнениях выше.
    raise DepthNotCharted(f"глубина {eff_bb} не попала ни в одну корзину {BUCKET_NAMES}")


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
    depth_bucket: str
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

    @field_validator("depth_bucket")
    @classmethod
    def _known_bucket(cls, value: str) -> str:
        if value not in BUCKET_NAMES:
            raise ValueError(f"неизвестная корзина глубины {value!r}, известны {BUCKET_NAMES}")
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
        return ChartKey(self.seats, self.position, self.depth_bucket, self.ante_type)


class ChartFileModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    readme: list[str] = []
    entries: list[ChartEntry]


class ChartBook:
    """Разобранный файл чартов. Отдаёт диапазон только по точному ключу."""

    def __init__(
        self, path: Path, entries: dict[ChartKey, tuple[ChartEntry, OpenStrategy]]
    ) -> None:
        self.path = path
        self._entries = entries

    def all_keys(self) -> list[ChartKey]:
        """Все ключи файла, включая образцы формата (их `get` не отдаёт)."""
        return list(self._entries)

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
                f"ante_type={key.ante_type!r}) в файле есть корзины: "
                f"{self._buckets_for(key) or 'ни одной'}"
            )
        entry, strategy = found
        if entry.status == "example":
            raise ChartPlaceholder(
                f"запись {tuple(key)} помечена status='example' — это образец формата, "
                f"а не чарт (source: {entry.source!r}); загрузчик её не отдаёт"
            )
        return entry, strategy

    def _buckets_for(self, key: ChartKey) -> list[str]:
        return sorted(
            k.depth_bucket
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
    depth_bucket: str,
    ante_type: str,
    *,
    path: Path | None = None,
) -> OpenStrategy:
    """Эталонная стратегия открытия по точному ключу. Отказы — см. `ChartBook.get`."""
    return load_chart_book(path).get(ChartKey(seats, position, depth_bucket, ante_type))


def open_range(
    seats: int,
    position: str,
    depth_bucket: str,
    ante_type: str,
    *,
    path: Path | None = None,
) -> Range:
    """Открытие с агрессией (рейз + олл-ин) по точному ключу — `OpenStrategy.opening`."""
    return open_strategy(seats, position, depth_bucket, ante_type, path=path).opening


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
