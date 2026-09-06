"""Матрица 13×13 картинкой: сетка классов рук, вес — доля заливки клетки.

Второй кодовый выход изложения (рядом с `hand_replay`): диапазон, на который
опирается вывод, показывается, а не описывается словами. Это прямое следствие
правила зон — если вердикт стоит в зоне «предполагая», игрок вправе увидеть
само допущение, а не поверить ему на слово.

**Раскладка — из плана дословно:** ranks `AKQJT98765432`; клетка (i, j): i==j —
пара, i<j — одномастная (`ranks[i]+ranks[j]+"s"`), i>j — разномастная
(`ranks[j]+ranks[i]+"o"`). Вес 0..1 — вертикальная доля заливки, растущая снизу
вверх поверх фона; SVG растеризуется cairosvg.

**Почему SVG собирается строкой, а не библиотекой.** Проверяемая часть здесь —
геометрия и подписи (`test_range_render.py` читает именно SVG), а не пиксели;
единственное, ради чего берётся внешняя зависимость, — растеризация. Всё, что
приходит снаружи (заголовок), экранируется: SVG — это XML, и неэкранированный
`&` в подписи ломает картинку целиком.

**Cairo ищется явно, если не нашлась сама.** `cairocffi` берёт библиотеку через
`ctypes.util.find_library`; когда тот её нашёл, `_locate_cairo` не делает
ничего. Ветка нужна для macOS с Homebrew: там cairo установлена, но лежит вне
путей, которые `find_library` просматривает по умолчанию, — каталоги
дописываются в окружение процесса до импорта (обоснование, почему это работает
после старта процесса, — в докстринге самой функции).
"""

from __future__ import annotations

import os
import sys
from ctypes.util import find_library
from typing import Any

from harness.contracts import RANKS, Range

__all__ = ["range_svg", "render_range_png"]

# Геометрия в пикселях SVG: клетка 56 px, вся сетка (13 × 56 плюс поля) — меньше
# 800 px по стороне.
_CELL = 56
_PAD = 12
_TITLE_H = 44
_GRID = _CELL * len(RANKS)
_WIDTH = _GRID + 2 * _PAD
_HEIGHT = _GRID + _TITLE_H + 2 * _PAD

_BG = "#f4f4f2"
_GRID_LINE = "#d0d0cc"
_FILL = "#2f7d4f"  # доля веса
_EMPTY = "#ffffff"
_TEXT = "#1a1a1a"
_LABEL_ON_FILL = "#ffffff"

# Порог, с которого подпись пишется по заливке, а не по фону: у почти полностью
# закрашенной клетки тёмный текст на тёмном фоне нечитаем.
_LABEL_INVERT_FROM = 0.6

_HOMEBREW_LIB_DIRS = ("/opt/homebrew/lib", "/usr/local/lib")


def hand_class_at(row: int, col: int) -> str:
    """Класс руки в клетке (row, col) — правило раскладки из плана, одной функцией."""
    if row == col:
        return RANKS[row] + RANKS[col]
    if row < col:
        return RANKS[row] + RANKS[col] + "s"
    return RANKS[col] + RANKS[row] + "o"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _cell_svg(row: int, col: int, weight: float) -> str:
    """Клетка: фон, заливка долей веса снизу вверх, подпись класса.

    Заливка — отдельный `<rect class="fill">` с `data-hand`: по нему тест
    находит геометрию именно этой клетки, не разбирая картинку глазами.
    """
    x = _PAD + col * _CELL
    y = _PAD + _TITLE_H + row * _CELL
    height = round(_CELL * weight, 3)
    hand_class = hand_class_at(row, col)
    label_color = _LABEL_ON_FILL if weight >= _LABEL_INVERT_FROM else _TEXT
    return (
        f'<rect class="cell" x="{x}" y="{y}" width="{_CELL}" height="{_CELL}" '
        f'fill="{_EMPTY}" stroke="{_GRID_LINE}" stroke-width="1"/>'
        f'<rect class="fill" data-hand="{hand_class}" x="{x}" y="{round(y + _CELL - height, 3)}" '
        f'width="{_CELL}" height="{height}" fill="{_FILL}"/>'
        f'<text class="label" x="{x + _CELL // 2}" y="{y + _CELL // 2 + 5}" '
        f'text-anchor="middle" font-family="DejaVu Sans, Helvetica, Arial, sans-serif" '
        f'font-size="15" fill="{label_color}">{hand_class}</text>'
    )


def range_svg(rng: Range, title: str) -> str:
    """Сетка 13×13 в SVG: 169 подписанных клеток и заголовок.

    Доля рук диапазона в заголовок не дописывается: её считает `Range.
    fraction_of_hands()`, а как подписать картинку — решает вызывающий
    (`presentation`), у которого один голос на весь продукт.
    """
    cells = "".join(
        _cell_svg(row, col, rng.weight(hand_class_at(row, col)))
        for row in range(len(RANKS))
        for col in range(len(RANKS))
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" height="{_HEIGHT}" '
        f'viewBox="0 0 {_WIDTH} {_HEIGHT}">'
        f'<rect width="{_WIDTH}" height="{_HEIGHT}" fill="{_BG}"/>'
        f'<text class="title" x="{_PAD}" y="{_PAD + 26}" '
        f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="22" '
        f'fill="{_TEXT}">{_escape(title)}</text>'
        f"{cells}</svg>"
    )


def _locate_cairo() -> None:
    """Дать `find_library` шанс найти cairo из Homebrew.

    Работает после старта процесса вопреки первому впечатлению: `find_library`
    на macOS читает `DYLD_FALLBACK_LIBRARY_PATH` из `os.environ` в МОМЕНТ
    вызова, а не полагается на копию окружения, прочитанную dyld при запуске.
    Если библиотека уже нашлась, функция не делает ничего.
    """
    if sys.platform != "darwin" or find_library("cairo"):
        return
    existing = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    extra = [path for path in _HOMEBREW_LIB_DIRS if os.path.isdir(path)]
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join([*extra, existing]).strip(":")


def _svg2png() -> Any:
    """Ленивый импорт cairosvg: зависимость нужна только тому, кто рисует картинку.

    Импорт на уровне модуля потянул бы cairo в каждый процесс, который просто
    импортировал `harness.explanation` (бот, воркер до станции изложения), и
    падал бы на машине без системной библиотеки ещё до первого вызова.
    """
    _locate_cairo()
    import cairosvg

    return cairosvg.svg2png


def render_range_png(rng: Range, title: str) -> bytes:
    """Та же сетка, растеризованная в PNG, — то, что уходит игроку картинкой."""
    png = _svg2png()(bytestring=range_svg(rng, title).encode("utf-8"))
    if not isinstance(png, bytes):  # pragma: no cover — cairosvg возвращает bytes
        raise TypeError(f"cairosvg вернул {type(png)!r} вместо bytes")
    return png
