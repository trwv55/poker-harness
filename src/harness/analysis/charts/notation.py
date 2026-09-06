"""Компактная запись диапазона («66+, ATs+, KQs, AJo+») ↔ веса по 169 классам.

Запись существует ради человека: чарты пишет владелец руками, и таблица из 169
строк для этого непригодна. Разворачивает запись код — ровно один разбор, ровно
в тот же `Range`, который считают инструменты анализа.

Грамматика (пять строк, полностью):

    range  := token ("," token)*                 разделитель — запятая или пробел
    token  := hands (":" weight)?                weight — доля в [0,1], по умолчанию 1.0
    hands  := hand | hand "+" | hand "-" hand    "+" — вверх, "X-Y" — отрезок (концы любым порядком)
    hand   := R R | R R ("s"|"o")                пара либо две карты, СТАРШИЙ РАНГ ПЕРВЫМ
    R      := один из AKQJT98765432              регистр рангов и суффикса не важен

Что означает «вверх»: для пар `66+` — это 66..AA; для непары старшая карта
фиксирована, растёт кикер: `ATs+` — это ATs, AJs, AQs, AKs (KK и AA сюда не
попадают). Отрезок берётся только между однотипными записями: `TT-77` (пары),
`A5s-A2s` (та же старшая карта и та же мастность). Класс, названный дважды, —
ошибка, а не молчаливое переопределение веса: в чарте это опечатка.

Класс, не названный в записи, имеет вес 0 — это контракт `Range`, а не
умолчание парсера.
"""

from __future__ import annotations

import re

from harness.contracts import RANKS, Range, all_classes

_RANK_INDEX: dict[str, int] = {rank: i for i, rank in enumerate(RANKS)}
_SEPARATOR = re.compile(r"[,\s]+")


class NotationError(ValueError):
    """Запись диапазона не разобрана: сообщение называет проблемный токен."""


def parse_range(text: str) -> Range:
    """Развернуть компактную запись в `Range` (веса по классам, отсутствие = 0).

    Ошибку поднимает на всём, что не описано грамматикой в шапке модуля, включая
    повтор класса и вес вне [0,1]. Молчаливого исправления нет ни в одном случае.
    """
    tokens = [t for t in _SEPARATOR.split(text.strip()) if t]
    if not tokens:
        raise NotationError("пустая запись диапазона")

    weights: dict[str, float] = {}
    for token in tokens:
        hands, weight = _split_weight(token)
        for cls in _expand(hands, token):
            if cls in weights:
                raise NotationError(f"класс {cls} назван дважды (токен {token!r})")
            weights[cls] = weight
    return Range(weights=weights)


def to_notation(rng: Range) -> str:
    """Записать диапазон токенами по одному классу, в порядке `all_classes()`.

    Компактную форму («66+») НЕ восстанавливает — сворачивать разложенное обратно
    в интервалы значило бы угадывать намерение автора чарта. Пин теста
    `test_notation_round_trip_is_bit_exact`: `parse_range(to_notation(r))` даёт те
    же веса бит в бит, включая дробные (вес печатается через `repr`).
    """
    parts: list[str] = []
    for cls in all_classes():
        weight = rng.weight(cls)
        if weight == 0.0:
            continue
        parts.append(cls if weight == 1.0 else f"{cls}:{weight!r}")
    return ", ".join(parts)


def _split_weight(token: str) -> tuple[str, float]:
    if ":" not in token:
        return token, 1.0
    hands, _, tail = token.partition(":")
    if ":" in tail:
        raise NotationError(f"больше одного двоеточия в токене {token!r}")
    try:
        weight = float(tail)
    except ValueError:
        raise NotationError(f"вес не число в токене {token!r}") from None
    if not 0.0 <= weight <= 1.0:
        raise NotationError(f"вес вне [0,1] в токене {token!r}")
    return hands, weight


def _expand(hands: str, token: str) -> list[str]:
    if "-" in hands:
        left, _, right = hands.partition("-")
        return _span(left, right, token)
    if hands.endswith("+"):
        return _upward(hands[:-1], token)
    return [_hand(hands, token)]


def _hand(text: str, token: str) -> str:
    """Один класс из записи руки: проверяет ранги, порядок и суффикс."""
    raw = text.strip()
    if len(raw) == 2:
        hi, lo, suffix = raw[0].upper(), raw[1].upper(), ""
    elif len(raw) == 3:
        hi, lo, suffix = raw[0].upper(), raw[1].upper(), raw[2].lower()
        if suffix not in ("s", "o"):
            raise NotationError(f"суффикс должен быть s или o, получено {raw[2]!r} ({token!r})")
    else:
        raise NotationError(f"не запись руки: {raw!r} (токен {token!r})")

    for rank in (hi, lo):
        if rank not in _RANK_INDEX:
            raise NotationError(f"неизвестный ранг {rank!r} в токене {token!r}")
    if hi == lo:
        if suffix:
            raise NotationError(f"у пары не бывает суффикса: {raw!r} (токен {token!r})")
        return hi + lo
    if not suffix:
        raise NotationError(f"непаре нужен суффикс s или o: {raw!r} (токен {token!r})")
    if _RANK_INDEX[hi] > _RANK_INDEX[lo]:
        raise NotationError(
            f"старший ранг пишется первым: {raw!r} — это {lo}{hi}{suffix} (токен {token!r})"
        )
    return hi + lo + suffix


def _upward(text: str, token: str) -> list[str]:
    cls = _hand(text, token)
    if len(cls) == 2:
        top = _RANK_INDEX[cls[0]]
        return [RANKS[i] * 2 for i in range(top, -1, -1)]
    hi, lo, suffix = cls[0], cls[1], cls[2]
    return [hi + RANKS[i] + suffix for i in range(_RANK_INDEX[lo], _RANK_INDEX[hi], -1)]


def _span(left: str, right: str, token: str) -> list[str]:
    first, second = _hand(left, token), _hand(right, token)
    if len(first) != len(second):
        raise NotationError(f"концы отрезка разного вида: {token!r}")
    if len(first) == 2:
        lo_i, hi_i = sorted((_RANK_INDEX[first[0]], _RANK_INDEX[second[0]]))
        return [RANKS[i] * 2 for i in range(lo_i, hi_i + 1)]
    if first[0] != second[0] or first[2] != second[2]:
        raise NotationError(
            f"отрезок берётся при одной старшей карте и одной мастности: {token!r}"
        )
    lo_i, hi_i = sorted((_RANK_INDEX[first[1]], _RANK_INDEX[second[1]]))
    return [first[0] + RANKS[i] + first[2] for i in range(lo_i, hi_i + 1)]
