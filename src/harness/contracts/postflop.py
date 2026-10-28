"""Ярлыки постфлоп-точки: сила руки, дро, бэкдоры, оверкарты, ценность на вскрытии.

Описывают карты героя на улице, а не решение: вердикта, зоны и цены здесь нет
(спека `docs/superpowers/specs/2026-10-09-postflop-line-design.md`, §4.1–§4.2.1,
§4.8). Считает их `analysis.tools.hand_class` и `analysis.tools.draws`; здесь они
только ОПИСАНЫ — по той же причине, что `ScanItem` (докстринг
`contracts.analysis`): читать их будет `presentation`, а импорт из `analysis`
втянул бы в образ бота `eval7` и `pokerkit`.

Ранги — однобуквенные, как в `contracts.ranges.RANKS` ("A", "T", "5"); карты —
двухбуквенные, как в колоде `analysis.tools.equity` ("Ah", "5c"). Доли — в долях
единицы.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, model_validator


class Combination(StrEnum):
    """Класс лучшей пятикарточной комбинации — то, что называет `eval7.handtype`."""

    HIGH_CARD = "high_card"
    PAIR = "pair"
    TWO_PAIR = "two_pair"
    TRIPS = "trips"
    STRAIGHT = "straight"
    FLUSH = "flush"
    FULL_HOUSE = "full_house"
    QUADS = "quads"
    STRAIGHT_FLUSH = "straight_flush"


class HandCategory(StrEnum):
    """Строка таблицы силы руки (спека, §4.1); одна строка — одно значение."""

    STRAIGHT_FLUSH = "straight_flush"
    QUADS = "quads"
    FULL_HOUSE = "full_house"
    FLUSH = "flush"
    STRAIGHT = "straight"
    WEAK_FLUSH = "weak_flush"
    WEAK_STRAIGHT = "weak_straight"
    SET = "set"
    TRIPS = "trips"
    TWO_PAIR = "two_pair"
    OVERPAIR = "overpair"
    TOP_PAIR_STRONG_KICKER = "top_pair_strong_kicker"
    TOP_PAIR_WEAK_KICKER = "top_pair_weak_kicker"
    MIDDLE_PAIR = "middle_pair"
    WEAK_PAIR = "weak_pair"
    ON_BOARD = "on_board"
    NO_PAIR = "no_pair"


class StrengthClass(StrEnum):
    """Класс силы готовой руки (спека, §4.1, правый столбец)."""

    STRONG = "strong"
    MEDIUM = "medium"
    WEAK = "weak"


Plays = Literal["hand", "kicker", "board"]


class HandStrength(BaseModel):
    """Сила готовой руки героя на улице.

    `combination` — класс лучшей пятикарточной комбинации из карт героя и борда;
    у категории `ON_BOARD` только он и говорит, что именно лежит на борде.

    `ranks` — что называть: ранг пары, обе пары старшей первой, ранг сета или
    трипса, старшая карта стрита или флеша, у фулл-хауса — тройка и пара; у
    `ON_BOARD` — то же для комбинации на борде; у `NO_PAIR` — старшая карта героя.

    `plays` — `"hand"`, когда комбинация героя (его карта входит в саму
    комбинацию), иначе `"kicker"` или `"board"`: рука на борде, и карта героя
    играет кикером или не играет вовсе. `kicker` — ранг этого кикера, ровно
    тогда, когда `plays == "kicker"`.
    """

    category: HandCategory
    strength: StrengthClass
    combination: Combination
    ranks: list[str]
    plays: Plays
    kicker: str | None

    @model_validator(mode="after")
    def _plays_follows_the_category(self) -> HandStrength:
        """Рука на борде и только она играет кикером или бордом; кикер — при `kicker`."""
        on_board = self.category is HandCategory.ON_BOARD
        if on_board == (self.plays == "hand"):
            raise ValueError(f"категория {self.category} при plays={self.plays}")
        if (self.plays == "kicker") != (self.kicker is not None):
            raise ValueError(f"кикер {self.kicker!r} при plays={self.plays}")
        return self


class DrawKind(StrEnum):
    """Вид дро (спека, §4.2). Комбо-дро — несколько видов в одном `Draw.kinds`."""

    FLUSH = "flush"
    OPEN_ENDED = "open_ended"
    DOUBLE_GUTSHOT = "double_gutshot"
    GUTSHOT = "gutshot"


class Draw(BaseModel):
    """Дро героя на флопе или тёрне: виды, ауты и точный шанс собрать.

    `outs` — карты-ауты, каждая один раз, даже если даёт и стрит, и флеш;
    `out_ranks` — ранги аутов на стрит, старший первым (туз — первым), пусто без
    стрит-дро. `unseen` — невидимые карты: 52 минус карты героя и борд.
    `hit_next` — `len(outs) / unseen`; `hit_by_river` — только на флопе,
    `1 − C(unseen − outs, 2) / C(unseen, 2)`.
    """

    kinds: list[DrawKind]
    out_ranks: list[str]
    outs: list[str]
    unseen: int
    hit_next: float
    hit_by_river: float | None


class Backdoor(BaseModel):
    """Бэкдор на флопе (спека, §4.2.1): вид и константа владельца, печатается с ≈.

    `variants` — число вариантов пары недостающих рангов `n` у стрита, `None` у
    флеша.
    """

    kind: Literal["flush", "straight"]
    variants: int | None
    approx: float


class Overcards(BaseModel):
    """Оверкарты героя на флопе или тёрне (спека, §4.2.1).

    `cards` — карты героя старше старшей карты борда, старшая первой (1 или 2).
    Обе цифры — константы владельца из таблицы, а не расчёт; `approx_by_river`
    только на флопе.
    """

    cards: list[str]
    approx_by_river: float | None
    approx_next: float


class ShowdownValue(BaseModel):
    """Перебор всех комбо соперника из невидимых карт на ривере (спека, §4.8).

    `wins` — комбо, у которых герой выигрывает; `ties_with` — классы рук,
    с которыми он делит банк ("53o", "53s"), по алфавиту.
    """

    wins: int
    ties: int
    losses: int
    ties_with: list[str]


class ActionTag(StrEnum):
    """Тип действия героя на постфлопе (спека, §4.3).

    Ставка — первая подошедшая строка таблицы §4.3: `CBET` … `BET`. Ответы и
    повышения — `CHECK_CALL` … `FOLD`. `RERAISE` — третье и дальнейшие повышения
    улицы: спека называет только первое («рейз») и второе («3-бет»).
    """

    CBET = "cbet"
    BARREL = "barrel"
    REPEAT_BET = "repeat_bet"
    DELAYED_CBET = "delayed_cbet"
    PROBE = "probe"
    DONK = "donk"
    BET_AFTER_CHECK = "bet_after_check"
    BET = "bet"
    CHECK_CALL = "check_call"
    CHECK_FOLD = "check_fold"
    CHECK_RAISE = "check_raise"
    RAISE = "raise"
    THREE_BET = "three_bet"
    RERAISE = "reraise"
    CALL = "call"
    FLOAT = "float"
    FOLD = "fold"


class SizeTag(StrEnum):
    """Тег размера ставки или рейза по полуоткрытым интервалам доли банка (§4.3).

    Блок [0, 33%), стандарт [33%, 75%), крупная [75%, 100%), овербет [100%, ∞).
    """

    BLOCK = "block"
    STANDARD = "standard"
    BIG = "big"
    OVERBET = "overbet"


class Purpose(StrEnum):
    """Назначение ставки, рейза или колла по силе руки и дро (спека, §4.5)."""

    VALUE = "value"
    MEDIUM_HAND = "medium_hand"
    SEMIBLUFF = "semibluff"
    BLUFF = "bluff"
    CALL_STRONG = "call_strong"
    BLUFF_CATCH = "bluff_catch"
    CALL_WITH_DRAW = "call_with_draw"


class Line(BaseModel):
    """Что сыграно героем в точке (спека, §4.3–§4.5).

    `barrel` — 2 или 3 у `ActionTag.BARREL`, иначе `None`. `size_pct` — точная
    доля банка у ставки и рейза (у рейза — повышение сверх колла к банку вместе с
    коллом), `None` у колла и фолда; `size_tag` — тег по ней. `purpose` — `None`,
    когда карты героя неизвестны, и у фолда.
    """

    action: ActionTag
    barrel: int | None
    size_pct: float | None
    size_tag: SizeTag | None
    purpose: Purpose | None


class FoldThreshold(BaseModel):
    """Сколько фолдов нужно ставке или рейзу (спека, §4.6), в долях единицы.

    `bluff` — `B / (P + B)`. `semibluff` — порог с учётом аутов, только у
    полублефа один на один; `None` — мультивей или назначение «блеф».
    `semibluff_free` — `E ≥ 0`: ставка окупается и без фолдов, тогда
    `semibluff` — ноль.
    """

    bluff: float
    semibluff: float | None
    semibluff_free: bool


class DrawCall(BaseModel):
    """Колл с дро (спека, §4.6): шансы банка против шанса собрать.

    `required_equity` — `C / (P + C)`; `hit` — шанс собрать, которым колл
    меряется; `by_pot_odds` — `hit ≥ required_equity`. Иначе
    `implied_needed_chips` — `X = C / hit − (P + C)` в фишках, округлённый вверх,
    а `beyond_stack` — `X` больше меньшего из стеков героя и соперника после колла.
    """

    required_equity: float
    hit: float
    by_pot_odds: bool
    implied_needed_chips: int | None
    beyond_stack: bool


class BetAlternatives(BaseModel):
    """Альтернативы последнему чеку или бету героя без ставки перед ним (решение
    владельца 2026-10-10, спека постфлоп-линии, §12): сколько фолдов нужно было бы
    ставке 50% банка и олл-ину, в долях единицы.

    Порог — `B / (P + B)`, как у блефа (§4.6): при колле герой проигрывает. `pot` —
    банк до хода `P`; `half_pot` — порог ставки `P / 2`, то есть ровно 1/3, и
    `None`, когда эффективный стек не больше половины банка (такая ставка и есть
    олл-ин) или сыгранный бет — от 45% до 55% банка (он и есть такая ставка);
    `all_in_chips` — эффективный стек героя `S` на решении;
    `all_in` — `S / (P + S)`.
    """

    pot: int
    half_pot: float | None
    all_in_chips: int
    all_in: float
