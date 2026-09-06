"""Изложение: покерное содержание разбора — реплей руки, матрица диапазонов, текст.

Три выхода, из них два — чистый код (`hand_replay`, `range_render`) и один
опирается на модель (`verdict_text`, `tournament_text`). Как это выглядит в
Телеграме, решает `presentation`; пакет знает только про покер.

Публичный API реэкспортируется здесь по образцу `harness.contracts` и
`harness.presentation`: вызывающий (`worker`, `presentation`, eval-раннер)
импортирует из `harness.explanation`, а не из отдельных модулей.
"""

from __future__ import annotations

from harness.explanation.hand_replay import HandReplay, ReplaySpan, hand_replay

__all__ = ["HandReplay", "ReplaySpan", "hand_replay"]
