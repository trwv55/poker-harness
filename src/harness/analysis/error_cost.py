"""Оценщик расхождений: ранжирование точек по цене и суммарная потеря руки.

Считаются только те точки, по которым вердикт вынесен. Точка без вердикта
(постфлоп, спот вне пуш-фолд-зоны, спот, который модель не умеет оценить) несёт
`ev_diff_bb = 0` — но ноль здесь означает «не посчитано», а не «сыграно верно».
Пускать такие точки в ранжирование значило бы показывать игроку пробел как
подтверждение правильной игры.
"""

from __future__ import annotations

from collections.abc import Sequence

from harness.contracts import PointVerdict, SpotKind

_JUDGED_SPOTS = frozenset({SpotKind.PUSHFOLD_UNOPENED, SpotKind.PUSHFOLD_FACING_SHOVE})


def is_judged(point: PointVerdict) -> bool:
    """Есть ли по точке вердикт: пустой `best_action` означает «не посчитано»."""
    return point.spot in _JUDGED_SPOTS and point.best_action != ""


def rank_points(points: Sequence[PointVerdict]) -> list[int]:
    """Индексы точек с вердиктом по убыванию потери — самая дорогая первой.

    Вторичный ключ — сам индекс: при равной цене порядок остаётся тем, в каком
    решения были приняты в раздаче, и не зависит от устойчивости сортировки.
    """
    judged = [i for i, point in enumerate(points) if is_judged(point)]
    return sorted(judged, key=lambda i: (points[i].ev_diff_bb, i))


def total_ev_loss_bb(points: Sequence[PointVerdict]) -> float:
    """Суммарная потеря руки в bb — сумма отрицательных расхождений (<= 0)."""
    return sum(
        point.ev_diff_bb for point in points if is_judged(point) and point.ev_diff_bb < 0.0
    )
