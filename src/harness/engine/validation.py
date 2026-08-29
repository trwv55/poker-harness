"""Политика пригодности руки: вердикт по рапорту движка.

Валидатор читает готовый `EngineReport` и решает, годна ли рука для разбора.
Данные он **никогда не правит**: придумать «правильную» сумму значило бы
сгаллюцинировать про деньги игрока. Расхождение он только называет.

Как именно называет — решает провенанс. Скриншот — это гипотеза: vision мог
ошибиться, поэтому спорные поля возвращаются игроку вопросом (`escalate`).
Hand history — это факт, записанный самим румом: расхождение с ней означает
баг нашего парсера, а не ошибку игрока, поэтому рука уходит в лог
разработчику (`reject`), и игрока мы ни о чём не спрашиваем.
"""

from __future__ import annotations

from harness.contracts import (
    CanonicalHand,
    EngineReport,
    Provenance,
    ValidationStatus,
    Verdict,
)

_FIELD_STACKS = "stacks"
_FIELD_ACTIONS = "actions"
_FIELD_CARDS = "cards"

_QUESTIONS: dict[str, str] = {
    _FIELD_ACTIONS: "Ставки и повышения в раздаче распознаны верно?",
    _FIELD_CARDS: "Карты героя и карты на столе распознаны верно?",
}


def _cards_in_play(hand: CanonicalHand) -> list[str]:
    """Все карты руки: по одному экземпляру на игрока плюс борд.

    Карты игрока собираются в объединение — источник называет их дважды (раздача
    и вскрытие), и это не дубль. Дубль — это одна карта у разных игроков или у
    игрока и на столе.
    """
    by_label: dict[str, list[str]] = {}
    for label, cards in hand.dealt.items():
        by_label.setdefault(label, []).extend(cards)
    for entry in hand.showdowns:
        by_label.setdefault(entry.label, []).extend(entry.cards)

    cards = [card for street_cards in hand.boards.values() for card in street_cards]
    for player_cards in by_label.values():
        cards += dict.fromkeys(player_cards)
    return cards


def _has_duplicate_cards(hand: CanonicalHand) -> bool:
    cards = _cards_in_play(hand)
    return len(cards) != len(set(cards))


def _question_for(field: str, hand: CanonicalHand) -> str:
    """Короткий человеческий вопрос по полю — без покерного жаргона."""
    if field == _FIELD_STACKS:
        hero = next((p for p in hand.players if p.label == hand.hero_label), None)
        if hero is not None:
            return f"Стек героя {hero.stack_bb:.1f}bb — верно?"
        return "Стеки игроков распознаны верно?"
    return _QUESTIONS.get(field, f"Поле «{field}» распознано верно?")


def validate(hand: CanonicalHand, report: EngineReport) -> Verdict:
    """Вынести вердикт по руке: `pass` / `escalate` / `reject`."""
    reasons: list[str] = []
    fields: list[str] = []

    if report.illegal_actions:
        reasons.append(f"illegal: {report.illegal_actions}")
        fields.append(_FIELD_ACTIONS)
    if hand.summary is not None and report.final_pot != hand.summary.total_pot:
        reasons.append(
            f"pot mismatch: engine={report.final_pot} summary={hand.summary.total_pot}"
        )
        fields += [_FIELD_STACKS, _FIELD_ACTIONS]
    if _has_duplicate_cards(hand):
        reasons.append("duplicate cards")
        fields.append(_FIELD_CARDS)

    start = sum(p.stack for p in hand.players)
    rake = hand.summary.rake if hand.summary is not None else 0
    if sum(report.stacks_end.values()) + rake != start:
        reasons.append(
            f"chip conservation violated: start={start} "
            f"end={sum(report.stacks_end.values())} rake={rake}"
        )
        fields.append(_FIELD_STACKS)

    if not reasons:
        return Verdict(status=ValidationStatus.PASS)
    if hand.provenance == Provenance.SCREENSHOT:
        asked = sorted(set(fields))
        return Verdict(
            status=ValidationStatus.ESCALATE,
            fields=asked,
            questions=[_question_for(field, hand) for field in asked],
            reasons=reasons,
        )
    # Hand history — факт рума: чинить надо парсер, а не данные и не игрока.
    return Verdict(status=ValidationStatus.REJECT, reasons=reasons)
