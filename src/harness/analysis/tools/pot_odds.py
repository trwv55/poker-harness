"""Пот-оддсы: минимальная эквити, оправдывающая колл."""

from __future__ import annotations


def required_equity(to_call: int, pot_before: int) -> float:
    """Доля банка, которую нужно выигрывать не реже, чтобы колл был не -EV.

    `pot_before` уже включает ставку соперника, которую мы коллируем (то есть
    это размер банка на момент решения, до нашего колла).
    """
    return to_call / (pot_before + to_call)
