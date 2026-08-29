"""Нормализатор: приводит `RawHand` от любого источника к `CanonicalHand`.

Публичный API пакета — реэкспорт из `normalize`, чтобы дальнейшие задачи
импортировали из `harness.normalizer`, а не из внутреннего модуля.
"""

from __future__ import annotations

from harness.normalizer.normalize import POSITIONS_BY_COUNT, normalize

__all__ = ["POSITIONS_BY_COUNT", "normalize"]
