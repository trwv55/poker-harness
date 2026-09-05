"""Классификация карманных карт по 169 классам и диапазоны (`Range`).

Класс руки — это её представление без учёта конкретной масти: пара ("QQ"),
одномастная ("AKs") или разномастная ("AKo") комбинация двух рангов. `Range`
хранит веса по классам (0..1, отсутствие ключа = 0) и умеет считать долю
комбинаций руки от общего числа 1326 стартовых комбинаций.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator

RANKS = "AKQJT98765432"

_PAIR_COMBOS = 6
_SUITED_COMBOS = 4
_OFFSUIT_COMBOS = 12
_TOTAL_COMBOS = 1326


def class_of(card1: str, card2: str) -> str:
    """Класс двух карманных карт: "K3s", "AKo", "QQ". Порядок карт не важен."""
    rank1, suit1 = card1[0], card1[1]
    rank2, suit2 = card2[0], card2[1]
    if rank1 == rank2:
        return rank1 + rank2
    if RANKS.index(rank1) > RANKS.index(rank2):
        rank1, rank2 = rank2, rank1
        suit1, suit2 = suit2, suit1
    suited = "s" if suit1 == suit2 else "o"
    return rank1 + rank2 + suited


def all_classes() -> list[str]:
    """Ровно 169 классов стартовых рук: 13 пар + 78 suited + 78 offsuit."""
    classes: list[str] = [rank + rank for rank in RANKS]
    for i, hi in enumerate(RANKS):
        for lo in RANKS[i + 1 :]:
            classes.append(hi + lo + "s")
            classes.append(hi + lo + "o")
    return classes


def _combos_for(cls: str) -> int:
    if len(cls) == 2:
        return _PAIR_COMBOS
    return _SUITED_COMBOS if cls[2] == "s" else _OFFSUIT_COMBOS


class Range(BaseModel):
    weights: dict[str, float] = {}  # класс -> 0..1; отсутствие ключа = 0

    @field_validator("weights")
    @classmethod
    def _validate_weights(cls, value: dict[str, float]) -> dict[str, float]:
        valid_classes = set(all_classes())
        for hand_class, weight in value.items():
            if hand_class not in valid_classes:
                raise ValueError(f"unknown hand class: {hand_class!r}")
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"weight out of [0,1] for {hand_class!r}: {weight}")
        return value

    def weight(self, cls: str) -> float:
        return self.weights.get(cls, 0.0)

    def fraction_of_hands(self) -> float:
        """Взвешенная доля комбо от 1326 (пары 6, suited 4, offsuit 12 комбо на класс)."""
        total_combos = sum(_combos_for(cls) * w for cls, w in self.weights.items())
        return total_combos / _TOTAL_COMBOS
