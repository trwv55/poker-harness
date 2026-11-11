"""Player-facing result of a verified, defaulted text scenario."""

from __future__ import annotations

from decimal import Decimal

from harness.contracts.natural_hand import ScenarioHand, ScenarioSnapshot
from harness.presentation.messages import Msg


def natural_read_unavailable_msg() -> Msg:
    return Msg(text="Сейчас не получилось прочитать описание раздачи. Попробуйте отправить его ещё раз позже.")


def natural_invalid_msg(reason: str) -> Msg:
    return Msg(text=reason + " Поправьте эту деталь одним сообщением.")


def _fmt(value: Decimal) -> str:
    result = format(value.quantize(Decimal("0.01")), "f").rstrip("0").rstrip(".")
    return (result or "0").replace(".", ",")


def natural_scenario_msg(
    hand: ScenarioHand, snapshot: ScenarioSnapshot, *,
    opinion: str = "", opponent_range: str | None = None, equity: float | None = None,
) -> Msg:
    """Show only code-calculated numbers and the assumptions that produced them."""
    street = {"preflop": "Префлоп", "flop": "Флоп", "turn": "Тёрн", "river": "Ривер"}[
        snapshot.street
    ]
    cards = "".join(hand.hero_cards) if hand.hero_cards else hand.hero_class or "карты не названы"
    board = " · " + "".join(hand.board) if hand.board else ""
    lines = [f"{street}{board} · {hand.hero_position} · {cards}.",
             f"Банк перед вашим ходом: {_fmt(snapshot.pot_before_bb)} BB."]
    if snapshot.to_call_bb > 0:
        lines.append(f"Для колла нужно добавить {_fmt(snapshot.to_call_bb)} BB.")
        if snapshot.to_call_bb > snapshot.remaining_bb[hand.hero_position]:
            lines.append("Стек героя меньше полной доплаты: нужен расчёт короткого олл-ина.")
        else:
            others = [p for p in snapshot.live_positions if p != hand.hero_position]
            later = [p for p in snapshot.positions_after_hero if p in others]
            if len(others) - len(later) == 1:
                threshold = snapshot.to_call_bb / (snapshot.pot_before_bb + snapshot.to_call_bb)
                condition = ("Если " + ", ".join(later) + " сбросят, " if later else "")
                lines.append(condition + "для колла нужно не меньше "
                             + _fmt(threshold * 100) + "% эквити против оставшегося соперника.")
                lines.append("Это порог по фишкам; вывод о колле зависит от диапазона соперника.")
            else:
                lines.append("Банк может стать мультивей: простой порог колла здесь недостаточен.")
    else:
        lines.append("Вы можете чекнуть без доплаты.")
        if snapshot.street != "preflop":
            bet = snapshot.pot_before_bb * Decimal("0.33")
            lines.append("Если решите поставить 33% банка, размер ставки — "
                         + _fmt(bet) + " BB. Выбор линии зависит от диапазона соперника.")
    if opponent_range and equity is not None and snapshot.to_call_bb > 0:
        threshold = snapshot.to_call_bb / (snapshot.pot_before_bb + snapshot.to_call_bb)
        gap = Decimal(str(equity)) - threshold
        decision = ("колл проходит порог" if gap > Decimal("0.01") else
                    "колл не проходит порог" if gap < Decimal("-0.01") else
                    "эквити близко к порогу")
        lines.append("Оценка модели: против предполагаемого диапазона " + opponent_range
                     + " эквити вашей руки " + _fmt(Decimal(str(equity)) * 100)
                     + "%; при названном условии " + decision + ".")
    if opinion:
        lines.append("Оценка модели: " + opinion)
    if snapshot.assumed_folds:
        lines.append("Неназванные пасы: " + ", ".join(snapshot.assumed_folds) + ".")
    if hand.assumptions:
        lines.append("Подставил: " + "; ".join(hand.assumptions) + ".")
    if hand.context:
        lines.append("Турнирный контекст из вашего описания: " + hand.context + ".")
    return Msg(text="\n".join(lines))
