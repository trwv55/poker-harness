"""Рендер матрицы 13×13 (задача 21): сетка классов рук, вес — доля заливки клетки.

Интерфейс и три проверки — из плана дословно: ровно 169 подписанных клеток в
SVG, вес 0.5 даёт заливку в половину высоты клетки, картинка больше 10 КБ.
Проверяется SVG, а не пиксели: растеризацию делает cairosvg, и повторять его
работу тестом бессмысленно — а вот геометрия и подписи наши.
"""

from __future__ import annotations

import re

from harness.contracts import Range, all_classes
from harness.explanation import range_svg, render_range_png


def _full_range() -> Range:
    return Range(weights={cls: 1.0 for cls in all_classes()})


def _cell(svg: str, hand_class: str) -> dict[str, float]:
    """Геометрия заливки одной клетки: `data-hand` — ключ, по которому её видно."""
    match = re.search(rf'<rect class="fill" data-hand="{hand_class}"([^/]*)/>', svg)
    assert match is not None, f"клетка {hand_class} не найдена"
    return {
        name: float(value)
        for name, value in re.findall(r'(x|y|width|height)="([-\d.]+)"', match.group(1))
    }


def test_the_grid_has_exactly_169_labelled_cells():
    """169 классов — 13 пар, 78 одномастных, 78 разномастных, и ни одной клетки сверх."""
    svg = range_svg(Range(weights={"AKs": 1.0}), "Диапазон")
    labels = re.findall(r'<text class="label"[^>]*>([^<]+)</text>', svg)
    assert len(labels) == 169
    assert set(labels) == set(all_classes())


def test_the_diagonal_is_pairs_and_the_corners_are_suited_and_offsuit():
    """Раскладка клеток из плана: i==j — пара, i<j — одномастная, i>j — разномастная."""
    svg = range_svg(_full_range(), "Диапазон")
    aa, ako, a2s = _cell(svg, "AA"), _cell(svg, "AKo"), _cell(svg, "A2s")
    assert aa["x"] == ako["x"], "разномастные обязаны стоять под диагональю в том же столбце"
    assert aa["y"] == a2s["y"], "одномастные обязаны стоять правее диагонали в той же строке"
    assert ako["y"] > aa["y"] and a2s["x"] > aa["x"]


def test_weight_half_fills_exactly_half_the_cell_height():
    """Вес 0.5 — заливка в половину высоты, снизу: клетка читается как «пуш 50%»."""
    # Обе клетки в одной строке сетки (QQ и QJs), иначе их `y` несравнимы.
    svg = range_svg(Range(weights={"QJs": 1.0, "QQ": 0.5}), "Диапазон")
    full, half = _cell(svg, "QJs"), _cell(svg, "QQ")
    assert half["height"] == full["height"] / 2
    assert half["y"] == full["y"] + full["height"] / 2, "заливка растёт снизу вверх"


def test_a_class_without_weight_is_not_filled_at_all():
    """Отсутствие ключа = вес 0 (контракт `Range`), значит нулевая заливка."""
    assert _cell(range_svg(Range(weights={"AKs": 1.0}), "Диапазон"), "72o")["height"] == 0.0


def test_the_title_is_rendered_and_escaped():
    """Заголовок приходит из вызывающего кода, поэтому экранируется — иначе руками
    собранный SVG ломается на первом же `&` в подписи диапазона."""
    svg = range_svg(Range(), "Колл шова & допущение <поля>")
    assert "&amp;" in svg and "&lt;поля&gt;" in svg
    assert "<поля>" not in svg


def test_png_is_a_real_png_over_ten_kilobytes():
    """Картинка — настоящий PNG и не вырожденная: проверка из плана дословно."""
    png = render_range_png(_full_range(), "Диапазон шова")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 10_240
