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

    ЕДИНСТВЕННАЯ формулировка правила во всей системе — как `is_judged`
    (`contracts.history`). Её зовут статистика (`analysis.player_stats`, доля
    дошедших до вскрытия) и реплей (`explanation.hand_replay`, строка вскрытия);
    двум формулировкам одного факта тут разойтись негде.

    Считается по пасам и доске, а НЕ по строкам показа карт: источник пишет
    показ и за тем, кто спасовал, и такая строка вскрытием не является
    (`test_cards_shown_after_a_fold_are_not_a_showdown`). Двое непасовавших —
    условие того, что вскрытие вообще состоялось: когда последнюю ставку никто
    не уравнял, до вскрытия не дошёл никто
    (`test_a_river_fold_leaves_no_showdown_for_anyone`).

    Множество спасовавших считается на каждый вызов: раздача — это десятки
    действий, а зовут предикат один раз на место. Против счёта эквити, который
    стоит рядом, это ничего не стоит.
    """
    folded = {action.label for action in hand.actions if action.kind is ActionKind.FOLD}
    live = sum(1 for player in hand.players if player.label not in folded)
    return Street.RIVER in hand.boards and live >= 2 and label not in folded
