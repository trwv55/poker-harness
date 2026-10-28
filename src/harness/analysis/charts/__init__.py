"""Справочник опен-диапазонов и диапазонов защиты: файлы в репозитории плюс
структурный лукап по ним.

Два источника эталона в системе разные по природе и не смешиваются:

* **пуш-фолд (≤ 15bb)** — вычисляется равновесием (`analysis.tools.pushfold`),
  ввод владельца не требуется, зона доверия `strict`;
* **опен на глубоких стеках (15bb+)** — справочник: чарты пишет владелец, код их
  читает и никогда не выводит из сыгранных рук; стек судится чартом ближайшей
  снятой глубины;
* **защита против опена (15bb+)** — справочник того же устройства и в том же
  каталоге `charts/data/`: ключ дополнен позицией открывшего, глубина — ближайшая
  по тому же правилу; нет файла — нет чартов защиты.

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
    DEFAULT_CHART_PATH,
    DEFAULT_DEFEND_PATH,
    MIN_CHART_DEPTH_BB,
    MIN_LOOKUP_DEPTH_BB,
    SCHEMA_VERSION,
    ChartBook,
    ChartEntry,
    ChartError,
    ChartFileError,
    ChartKey,
    ChartMissing,
    ChartPlaceholder,
    DefendBook,
    DefendEntry,
    DefendKey,
    DefendStrategy,
    DepthNotCharted,
    OpenStrategy,
    chart_keys,
    defend_strategy,
    load_chart_book,
    load_defend_book,
    open_range,
    open_strategy,
)
from harness.analysis.charts.notation import NotationError, parse_range, to_notation

__all__ = [
    "DEFAULT_CHART_PATH",
    "DEFAULT_DEFEND_PATH",
    "MIN_CHART_DEPTH_BB",
    "MIN_LOOKUP_DEPTH_BB",
    "SCHEMA_VERSION",
    "ChartBook",
    "ChartEntry",
    "ChartError",
    "ChartFileError",
    "ChartKey",
    "ChartMissing",
    "ChartPlaceholder",
    "DefendBook",
    "DefendEntry",
    "DefendKey",
    "DefendStrategy",
    "DepthNotCharted",
    "NotationError",
    "OpenStrategy",
    "chart_keys",
    "defend_strategy",
    "load_chart_book",
    "load_defend_book",
    "open_range",
    "open_strategy",
    "parse_range",
    "to_notation",
]
