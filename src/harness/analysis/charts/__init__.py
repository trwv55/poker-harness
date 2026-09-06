"""Справочник опен-диапазонов: файл в репозитории плюс структурный лукап по нему.

Два источника эталона в системе разные по природе и не смешиваются:

* **пуш-фолд (≤ 15bb)** — вычисляется равновесием (`analysis.tools.pushfold`),
  ввод владельца не требуется, зона доверия `strict`;
* **опен на глубоких стеках (15bb+)** — справочник: чарты пишет владелец, код их
  читает и никогда не выводит из сыгранных рук.

Публичный API — реэкспорт из `notation` (запись диапазона) и `lookup` (файл, ключ,
отказы), чтобы вызывающая сторона импортировала из `harness.analysis.charts`.

Пакет — не кэш. `analysis/tools/data/` рядом хранит производные артефакты
(равновесия и таблицу эквити; два из трёх файлов там вовсе в `.gitignore`,
потому что восстанавливаются пересчётом). Чарт пересчётом не восстанавливается:
это исходные данные с провенансом, версионируемые гитом, — поэтому у них
отдельный каталог `charts/data/`, а не соседство с кэшем.
"""

from __future__ import annotations

from harness.analysis.charts.lookup import (
    BUCKET_NAMES,
    DEFAULT_CHART_PATH,
    DEPTH_BUCKETS,
    MIN_CHART_DEPTH_BB,
    SCHEMA_VERSION,
    ChartBook,
    ChartEntry,
    ChartError,
    ChartFileError,
    ChartKey,
    ChartMissing,
    ChartPlaceholder,
    DepthNotCharted,
    chart_keys,
    depth_bucket_for,
    load_chart_book,
    open_range,
)
from harness.analysis.charts.notation import NotationError, parse_range, to_notation

__all__ = [
    "BUCKET_NAMES",
    "DEFAULT_CHART_PATH",
    "DEPTH_BUCKETS",
    "MIN_CHART_DEPTH_BB",
    "SCHEMA_VERSION",
    "ChartBook",
    "ChartEntry",
    "ChartError",
    "ChartFileError",
    "ChartKey",
    "ChartMissing",
    "ChartPlaceholder",
    "DepthNotCharted",
    "NotationError",
    "chart_keys",
    "depth_bucket_for",
    "load_chart_book",
    "open_range",
    "parse_range",
    "to_notation",
]
