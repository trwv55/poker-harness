"""Лукап опен-чартов: точный ключ `(seats, position, depth_bucket, ante_type)` → `Range`.

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

Что этот модуль НЕ проверяет: разумность самого чарта. Монотонность по глубине, по
позиции, ширина диапазона — не его дело: чарты владельца, код их не судит.
Проверяется только форма: известный ключ, известные классы, вес в [0,1],
заполненные `source` и `revised_at` — список проверок пинит параметризованный
`test_malformed_file_is_rejected`.
"""

from __future__ import annotations

import json
from datetime import date
from math import inf
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from harness.analysis.charts.notation import NotationError, parse_range
from harness.contracts import Range
from harness.normalizer import POSITIONS_BY_COUNT

SCHEMA_VERSION = 1

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


class ChartEntry(BaseModel):
    """Одна запись файла: ключ, провенанс и диапазон в одной из двух форм.

    `extra="forbid"`: опечатка в имени поля — ошибка загрузки, а не тихо
    проигнорированное поле (пин `test_malformed_file_is_rejected`, случай
    «опечатка в имени поля»).
    """

    model_config = ConfigDict(extra="forbid")

    seats: int
    position: str
    depth_bucket: str
    ante_type: str
    source: str
    revised_at: date
    status: Literal["chart", "example"] = "chart"
    hands: str | None = None
    weights: dict[str, float] | None = None

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
        if (self.hands is None) == (self.weights is None):
            raise ValueError("нужно ровно одно из полей: hands (запись) либо weights (карта весов)")
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

    def __init__(self, path: Path, entries: dict[ChartKey, tuple[ChartEntry, Range]]) -> None:
        self.path = path
        self._entries = entries

    def all_keys(self) -> list[ChartKey]:
        """Все ключи файла, включая образцы формата (их `get` не отдаёт)."""
        return list(self._entries)

    def entry(self, key: ChartKey) -> ChartEntry:
        """Запись по ключу — вместе с провенансом. Отказы те же, что у `get`."""
        entry, _ = self._require(key)
        return entry

    def get(self, key: ChartKey) -> Range:
        """Диапазон по ТОЧНОМУ ключу.

        `ChartMissing` — записи нет (соседняя не подставляется);
        `ChartPlaceholder` — запись помечена образцом формата.
        """
        _, rng = self._require(key)
        return rng

    def _require(self, key: ChartKey) -> tuple[ChartEntry, Range]:
        found = self._entries.get(key)
        if found is None:
            raise ChartMissing(
                f"нет чарта по ключу {tuple(key)} в {self.path}; "
                f"подстановка похожего чарта запрещена. "
                f"Для (seats={key.seats}, position={key.position!r}, "
                f"ante_type={key.ante_type!r}) в файле есть корзины: "
                f"{self._buckets_for(key) or 'ни одной'}"
            )
        entry, rng = found
        if entry.status == "example":
            raise ChartPlaceholder(
                f"запись {tuple(key)} помечена status='example' — это образец формата, "
                f"а не чарт (source: {entry.source!r}); загрузчик её не отдаёт"
            )
        return entry, rng

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


def open_range(
    seats: int,
    position: str,
    depth_bucket: str,
    ante_type: str,
    *,
    path: Path | None = None,
) -> Range:
    """Эталонный опен-диапазон по точному ключу. Отказы — см. `ChartBook.get`."""
    return load_chart_book(path).get(ChartKey(seats, position, depth_bucket, ante_type))


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

    entries: dict[ChartKey, tuple[ChartEntry, Range]] = {}
    for entry in model.entries:
        if entry.key in entries:
            raise ChartFileError(f"ключ {tuple(entry.key)} встречается дважды: {path}")
        entries[entry.key] = (entry, _range_of(entry, path))
    if not entries:
        raise ChartFileError(f"в файле чартов нет ни одной записи: {path}")
    return ChartBook(path, entries)


def _range_of(entry: ChartEntry, path: Path) -> Range:
    """Диапазон записи из любой из двух форм — с контекстом ключа в сообщении.

    Обе формы проходят валидатор `Range`: неизвестный класс и вес вне [0,1] —
    отказ загрузки. Класс, не названный записью, имеет вес 0 (контракт `Range`).
    """
    try:
        if entry.hands is not None:
            return parse_range(entry.hands)
        return Range(weights=entry.weights or {})
    except (NotationError, ValidationError, ValueError) as exc:
        raise ChartFileError(
            f"диапазон записи {tuple(entry.key)} не принят ({path}): {exc}"
        ) from exc
