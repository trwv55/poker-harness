"""Нормализованное представление руки — вход для движка и аналитического ядра.

В отличие от `RawHand`, здесь суммы действий — это накопленный итог,
поставленный игроком на текущей улице (`committed_after`), а не доплата;
позиции игроков вычислены явно (`PlayerState.position`); личность каждого
игрока классифицирована (`Identity`).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel

from harness.contracts.raw import (
    ActionKind,
    Collected,
    Completeness,
    Post,
    Provenance,
    ShowdownEntry,
    Street,
    SummaryInfo,
    Uncalled,
    VisionMeta,
)


class Identity(StrEnum):
    HERO = "hero"
    NICK = "nick"
    ANON = "anon"


class PlayerState(BaseModel):
    seat: int
    label: str
    identity: Identity
    position: str  # "BTN"/"SB"/"BB"/"UTG"/...
    stack: int
    stack_bb: float


class CanonicalAction(BaseModel):
    street: Street
    label: str
    kind: ActionKind
    committed_after: int  # ИТОГО поставлено игроком на этой улице после действия
    is_all_in: bool = False
    raw_line: str


class CanonicalHand(BaseModel):
    schema_version: int = 1
    provenance: Provenance
    # Полнота входа переносится из `RawHand` без изменений: маршрут в ядро
    # выбирает `harness.engine.enrich`, а он видит только каноническую руку.
    completeness: Completeness = Completeness.HAND
    tournament_id: str
    hand_no: str
    hand_index: int | None = None
    level: int
    sb: int
    bb: int
    ante: int
    ante_type: str = "per_player"
    timestamp: datetime
    button_seat: int
    hero_label: str = "Hero"
    players: list[PlayerState]
    # Переносится из `RawHand` без изменений — см. её докстринг поля.
    visible_bets: dict[str, int] = {}
    dealt: dict[str, list[str]] = {}
    # Посты анте и блайндов, как их записал источник. Деньги отсюда НЕ берутся:
    # блайнды уже сидят в `committed_after` первого круга, анте — в `ante`.
    # Поле держится как независимая улика: позиции выведены из кнопки, а тут
    # написано, кто блайнды поставил на самом деле — сверка ловит чужую кнопку.
    posts: list[Post] = []
    actions: list[CanonicalAction] = []
    boards: dict[Street, list[str]] = {}
    uncalled: list[Uncalled] = []
    showdowns: list[ShowdownEntry] = []
    collected: list[Collected] = []
    summary: SummaryInfo | None = None
    bounties: dict[str, int] | None = None
    bounty_source: str | None = None
    vision: VisionMeta | None = None


def went_to_showdown(hand: CanonicalHand, label: str) -> bool:
    """Дошло ли место до вскрытия: доска доехала до ривера, оно не пасовало, и оно не одно.

    Единственная формулировка правила для ДВУХ её читателей — статистики
    (`analysis.player_stats`, доля дошедших до вскрытия) и реплея
    (`explanation.hand_replay`, строка вскрытия): им разойтись негде. Второй
    предикат того же вопроса в системе всё же есть —
    `analysis.tournament._showdown` («герой есть в `hand.showdowns`»). Он
    заполняет поля отчёта по турниру (`AllInEvent.showdown`, `ChipMove.showdown`),
    а отчёт уходит и в рассказ словами (`worker/pipeline.py`,
    `_tournament_story`), и в сообщение с числами; сам рассказ описан в
    `.claude/ARCHITECTURE.md` и `.claude/SCALING.md` — тех-спеки у него нет. Эта
    задача его не трогала, и свести их — отдельная работа. Замер расхождения на
    2026-09-12: на 407 раздачах с чекпоинтом признаки совпали на 406,
    единственное расхождение — скриншотная раздача, а рассказ по турниру берёт
    руки турнира (`list_by_tournament`), куда скриншотные не попадают. Число
    живёт на той базе и той дате, не вечно.

    Считается по пасам и доске, а НЕ по строкам показа карт: источник пишет
    показ и за тем, кто спасовал, и такая строка вскрытием не является
    (`test_cards_shown_after_a_fold_are_not_a_showdown`). Двое непасовавших —
    условие того, что вскрытие вообще состоялось: когда последнюю ставку никто
    не уравнял, до вскрытия не дошёл никто
    (`test_a_river_fold_leaves_no_showdown_for_anyone`).

    Цена формулировки-функции: `hand.actions` обходится на каждый вызов, то есть
    на каждое место раздачи, а не один раз на раздачу, как считал предрасчёт
    `player_stats._HandView` до переноса. Владелец эту цену принял осознанно.
    """
    folded = {action.label for action in hand.actions if action.kind is ActionKind.FOLD}
    live = sum(1 for player in hand.players if player.label not in folded)
    return Street.RIVER in hand.boards and live >= 2 and label not in folded
