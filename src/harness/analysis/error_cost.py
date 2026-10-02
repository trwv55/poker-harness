"""Оценщик расхождений: ранжирование точек по цене и суммарная потеря руки.

Считаются только те точки, по которым вердикт вынесен. Точка без вердикта
(постфлоп, ответ на чужой рейз, спот, который модель не умеет оценить) несёт
`ev_diff_bb = 0` — но ноль здесь означает «не посчитано», а не «сыграно верно».
Пускать такие точки в ранжирование значило бы показывать игроку пробел как
подтверждение правильной игры.

Само правило «точка судима» живёт в `contracts.history.is_judged` и здесь только
переэкспортировано: его читают ещё и память (колонка `decision_points.judged`), и
изложение, а импортировать ради него расчётный пакет им нельзя
(`test_bot_image_does_not_import_calculation_stack`).
"""

from __future__ import annotations

from collections.abc import Sequence

from harness.contracts import JUDGED_SPOTS, PointVerdict, is_judged

__all__ = ["JUDGED_SPOTS", "is_judged", "rank_points", "total_ev_loss_bb"]


def rank_points(points: Sequence[PointVerdict]) -> list[int]:
    """Индексы точек с вердиктом по убыванию потери — самая дорогая первой.

    Вторичный ключ — сам индекс: при равной цене порядок остаётся тем, в каком
    решения были приняты в раздаче, и не зависит от устойчивости сортировки.
    """
    judged = [i for i, point in enumerate(points) if is_judged(point)]

    def key(i: int) -> tuple[int, float, int]:
        point = points[i]
        if point.mismatch is None:
            return (0, point.ev_diff_bb, i)
        # Точка по чарту цены не имеет: она идёт после ценовых, расхождение раньше
        # совпадения, внутри — реже у солвера раньше.
        return (1 if point.mismatch else 2, float(point.detail["taken_frequency"]), i)

    return sorted(judged, key=key)


def total_ev_loss_bb(points: Sequence[PointVerdict]) -> float:
    """Суммарная потеря руки в bb — сумма отрицательных расхождений (<= 0).

    Точки по чарту (`mismatch` не `None`) цены не имеют и в сумму не входят: их
    `ev_diff_bb` — заглушка, а не «здесь не потеряно».
    """
    return sum(
        point.ev_diff_bb
        for point in points
        if is_judged(point) and point.mismatch is None and point.ev_diff_bb < 0.0
    )
