"""Сырые данные, извлечённые парсером из источника (hand history / скриншот).

`RawHand` — это буквальное представление того, что написано в источнике: суммы
действий записаны как доплаты (как в GG hand history), улицы разложены по
секциям, а всё, что парсер не смог распознать, попадает в `unknown_lines`.
Нормализация в `CanonicalHand` (см. `canonical.py`) происходит позже.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class Provenance(StrEnum):
    HAND_HISTORY = "hand_history"
    SCREENSHOT = "screenshot"


class Completeness(StrEnum):
    """Рука целиком или незавершённая раздача — выводится из ПРОЧИТАННОГО.

    Решение владельца C1 отдельного типа экрана не вводит: классифицировать экран
    мы не беремся. Полнота — не тип экрана, а свойство того, что на нём удалось
    прочитать: есть результат или вскрытие — рука дошла до конца и её можно
    проиграть движком; нет — конца раздачи чтение не увидело (реестр E1).

    `STATE` — признак отказа, а не второй маршрут: такой экран станция чтения
    отвергает до сохранения руки и до разбора
    (`harness.parsers.vision_adapter.vision_extract`,
    `test_a_hand_still_in_progress_never_reaches_the_analysis`). В ядро идёт
    только `HAND` — реплеем и полной сверкой денег.

    `hand_history` всегда `HAND`: рум пишет руку целиком.
    """

    HAND = "hand"
    STATE = "state"


class Street(StrEnum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class ActionKind(StrEnum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


class PostKind(StrEnum):
    ANTE = "ante"
    SMALL_BLIND = "small_blind"
    BIG_BLIND = "big_blind"


class SeatInfo(BaseModel):
    seat: int
    label: str
    stack: int


class Post(BaseModel):
    label: str
    kind: PostKind
    amount: int


class RawAction(BaseModel):
    street: Street
    label: str
    kind: ActionKind
    amount: int | None = None  # calls/bets N — ДОПЛАТА, как в источнике
    to_amount: int | None = None  # raises X to Y -> Y
    is_all_in: bool = False
    raw_line: str


class Uncalled(BaseModel):
    label: str
    amount: int


class ShowdownEntry(BaseModel):
    label: str
    cards: list[str]


class Collected(BaseModel):
    label: str
    amount: int  # мейн/сайд не подписаны — как в GG


class SummaryInfo(BaseModel):
    total_pot: int
    rake: int
    jackpot: int
    bingo: int
    fortune: int
    tax: int
    board: list[str] = []
    seat_lines: list[str] = []


class VisionHop(BaseModel):
    """Одна ступень каскада моделей: кто читал и чем кончилось.

    Каскад — решение владельца 2026-09-06, п.4: чтение дешёвой моделью
    (`LLM_VISION_MODEL`), провал любой контрольной суммы — повтор на дорогой
    (`LLM_VISION_FALLBACK_MODEL`), провал и там — вопрос игроку. Ступени
    сохраняются здесь, чтобы после эскалации было видно, на какой из них
    расхождение появилось, а не только что оно есть.
    """

    role: str  # "primary" | "fallback"
    model: str
    failed_checks: list[str] = []
    error: str | None = None


class VisionCheck(BaseModel):
    """Итог одной контрольной суммы: имя, прошла ли, и два сравненных значения.

    `options` — то, что предъявляется игроку кнопками при эскалации: расхождение
    двух независимых прочтений даёт ровно два кандидата, и выбирать между ними
    игроку проще, чем вводить число (спека §8.3).
    """

    name: str
    passed: bool
    detail: str = ""
    options: list[str] = []
    # Кого касается расхождение — ник игрока, если проверка вообще про игрока.
    # Без него ответ «карты такие» некуда подставить: вариантов два, а чьи это
    # карты, знает только та проверка, которая их сравнивала.
    subject: str = ""


class VisionMeta(BaseModel):  # только для скринов
    confidence: dict[str, float] = {}
    needs_review: list[str] = []
    image_hash: str | None = None
    nicknames: dict[str, str] = {}
    bounties: dict[str, int] = {}
    displayed_pot: int | None = None
    # Ниже — поля задачи 22. Все необязательные: правило эволюции контрактов
    # (Global Constraints плана) разрешает только добавление необязательных
    # полей, иначе уже записанный jsonb перестал бы читаться новой моделью.
    hops: list[VisionHop] = []
    checks: list[VisionCheck] = []
    unsure_fields: list[str] = []
    # Ники, прочитанные моделью, среди которых код искал героя по нику из
    # профиля. Хранятся затем, чтобы эскалация «кто из них вы» могла предложить
    # тот же список, что видел код, а не перечитывать экран заново.
    hero_candidates: list[str] = []


class RawHand(BaseModel):
    schema_version: int = 1
    provenance: Provenance
    # Полнота входа (см. `Completeness`). Умолчание — рука целиком: так читается
    # весь уже записанный jsonb HH-пути, где иначе и не бывает.
    completeness: Completeness = Completeness.HAND
    source_ref: str
    hand_no: str
    tournament_id: str
    tournament_name: str
    level: int
    sb: int
    bb: int
    ante: int
    ante_type: str = "per_player"
    timestamp: datetime
    table_name: str
    max_seats: int
    button_seat: int
    seats: list[SeatInfo]
    # Фишки, стоящие перед игроками в момент снимка, — наблюдение живого стола
    # (метка игрока -> фишки). У полной руки пусто: там ставки восстанавливает
    # лог действий, и второй источник тех же денег только разошёлся бы с первым.
    visible_bets: dict[str, int] = {}
    posts: list[Post]
    dealt: dict[str, list[str]] = {}  # пустой список = Dealt to без карт
    actions: list[RawAction] = []
    boards: dict[Street, list[str]] = {}
    uncalled: list[Uncalled] = []
    showdowns: list[ShowdownEntry] = []
    collected: list[Collected] = []
    summary: SummaryInfo | None = None
    vision: VisionMeta | None = None
    unknown_lines: list[str] = []  # всё, что парсер не распознал
