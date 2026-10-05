"""Результат движка (пот/стеки/решения) поверх канонической руки плюс вердикт валидации.

`EnrichedHand` — вход для аналитического ядра: помимо самой руки несёт отчёт
движка (`EngineReport`) с точками решений (`DecisionPoint`) и вердикт
валидатора (`Verdict`), решающий, годна ли рука для дальнейшего анализа.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from harness.contracts.canonical import CanonicalAction, CanonicalHand
from harness.contracts.raw import Street


class SidePot(BaseModel):
    amount: int
    eligible: list[str]


class DecisionPoint(BaseModel):
    index: int
    street: Street
    label: str
    position: str
    to_call: int
    pot_before: int
    eff_stack: int
    eff_stack_bb: float
    spr: float | None = None
    # Сыгранное героем действие. Обязательное: точку решения строит только
    # `harness.engine.replay`, а он строит её ИЗ действия.
    action: CanonicalAction
    live_total: int = 0  # игроков ещё в руке на момент решения, включая Hero
    live_behind: int = 0  # из них ещё не действовавших после Hero — вход правила зоны


class ValidationStatus(StrEnum):
    PASS = "pass"
    ESCALATE = "escalate"
    REJECT = "reject"


class Verdict(BaseModel):
    status: ValidationStatus
    fields: list[str] = []
    questions: list[str] = []
    reasons: list[str] = []
    # Проверки, которые на этом входе выполнить НЕ ИЗ ЧЕГО, названные поимённо:
    # молчаливый `pass` неотличим от проверенного входа.
    not_checked: list[str] = []


class EngineReport(BaseModel):
    pot_by_street: dict[Street, int]
    final_pot: int
    side_pots: list[SidePot] = []
    stacks_end: dict[str, int]
    decision_points: list[DecisionPoint]
    illegal_actions: list[str] = []
    # Игроки, которым источник записал пас при нулевом стеке (олл-ин с
    # вынужденной ставки): движок исполняет такой пас, хотя правила NLHE
    # оставляют игрока в руке. Список делает эту поправку видимой в трассе.
    forfeits: list[str] = []


class EnrichedHand(BaseModel):
    schema_version: int = 1
    hand: CanonicalHand
    report: EngineReport
    verdict: Verdict


# Точность округления bb — та же, что у отчёта по турниру (`analysis.tournament.
# _BB_PRECISION`): величина приходит делением фишек на bb, и двоичный хвост
# деления не должен оседать в контракте.
_DELTA_PRECISION = 6


def hero_stack_delta_bb(en: EnrichedHand) -> float:
    """Изменение стека героя за раздачу, в bb её уровня. Знак — его же.

    Считается по стекам движка (`EngineReport.stacks_end`), а НЕ по строкам
    выплат источника: движок проигрывает руку сам и записанным суммам не верит
    (`engine.replay`, ARCHITECTURE.md). Расхождение между своим подсчётом и
    источником обязано остаться видимым — оно и есть сигнал о битых данных.

    Единственная формулировка на двух читателей: отчёт по турниру
    (`analysis.tournament._delta_bb` — фишечное разбиение `EvSplit`) и блок
    «Что было» (`explanation.hand_replay` — фраза исхода раздачи). Второй копии
    формулы в системе быть не должно: разъехавшись, они дали бы игроку два
    разных ответа на вопрос «сколько я потерял».

    Отсутствие героя за столом — не пробел данных, а чужая раздача, и потому
    исключение, а не ноль.
    """
    hand = en.hand
    start = next(
        (player.stack for player in hand.players if player.label == hand.hero_label), None
    )
    if start is None:
        raise ValueError(f"в раздаче {hand.hand_no} нет места героя ({hand.hero_label})")
    return round((en.report.stacks_end[hand.hero_label] - start) / hand.bb, _DELTA_PRECISION)
