"""Что модель ВИДИТ на скриншоте — наблюдения, а не пересчитанная рука.

Контракт существует отдельно от `RawHand` по решению реестра
(`docs/superpowers/specs/2026-09-04-vision-open-problems.md`, «Схема извлечения
v1»): в `RawHand` анте подушевое, а экран показывает пул; стек там в фишках, а
экран может показывать большие блайнды. Отдать модели сразу `RawHand` значило бы
поручить ей делить и переводить — то есть считать, что запрещено границей проекта
(CLAUDE.md: «точное считает код»). Поэтому модель заполняет `VisionReading` в тех
единицах, в каких написано на экране, а перевод и деление делает
`harness.parsers.vision_adapter` в одном явном месте.

Три следствия этой границы видны прямо в полях:

* **Позиций здесь нет вовсе** (реестр A3). Ни один тип экрана их не подписывает у
  героя, единственный источник — `has_button`, дальше нормалайзер. Спросить
  позицию значит пригласить выдумать.
* **Анте двумя полями** (реестр B2): `ante_pool_shown` — то, что написано («Все
  анте: N»), `ante_per_player_shown` — только если экран показывает подушевое.
  Делит код, зная число игроков.
* **Карты спрашиваются дважды** (реестр «Карты отрисованы дважды»):
  `cards_at_seat` — у места за столом, `cards_in_log` — в колонке улицы.
  Измерено, что модель схлопывает эту избыточность, если спросить один раз;
  раздельные поля дают два независимых наблюдения, которые сверяет код.

Единица приложена к каждой числовой группе (реестр A4): в пределах одного экрана
блайнды бывают в фишках, а стеки — в больших блайндах, и модель, которая
«поможет» и приведёт к одной единице, выдаст неверные числа, выглядящие
безупречно.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from harness.contracts.raw import ActionKind, Street


class Unit(StrEnum):
    """В чём написано число на экране. `bb` — «большие блайнды»."""

    CHIPS = "chips"
    BB = "bb"


class SeenPlayer(BaseModel):
    """Один игрок за столом — так, как он выглядит на экране.

    Герой ВХОДИТ в список и помечен `is_hero` (реестр B1: расхождение «7 против
    8» между моделями было не ошибкой зрения, а неоговорённым заданием).
    `is_hero` при этом наблюдение модели, а не источник истины: героя определяет
    код по нику из профиля (`vision_adapter.identify_hero`).
    """

    seat: int
    nickname: str | None = None
    is_hero: bool = False
    stack: float | None = None
    stack_unit: Unit | None = None
    # Фишки, стоящие перед игроком в этот момент. Кто из них малый блайнд, а кто
    # большой, модель НЕ помечает (реестр B3) — это интерпретация, её делает код.
    bet: float | None = None
    bet_unit: Unit | None = None
    has_button: bool = False
    cards_at_seat: list[str] = []
    cards_in_log: list[str] = []
    # Видны ли перед игроком карты (рубашкой или открытые). `None` — по экрану не
    # определить. На живом столе это единственный видимый признак того, что игрок
    # ещё в руке; на экспорте истории он не значит ничего (там показаны все).
    has_hole_cards: bool | None = None
    bounty_usd: float | None = None
    # Процент, напечатанный GG рядом с игроком на олл-ин-экспорте. Оракул для
    # чтения карт: код считает эквити сам и сверяет (реестр, «Эквити с экрана GG»).
    equity_shown_pct: float | None = None


class SeenAction(BaseModel):
    """Строка лога улиц — есть только на экспорте истории, на живом столе её нет.

    Суммы — как на экране: у повышения написан итог улицы (`to_amount`), у колла и
    ставки — доплата (`amount`). Это те же две формы, что пишет hand history GG,
    и `RawAction` их различает так же.
    """

    street: Street
    seat: int
    kind: ActionKind
    amount: float | None = None
    to_amount: float | None = None
    is_all_in: bool = False


class VisionReading(BaseModel):
    """Полное наблюдение по одному экрану — выходная схема vision-вызова.

    `not_a_hand` с причиной — честный отказ на экране, который рукой не является
    (лобби, список результатов): выдуманная из лобби рука хуже отказа (реестр C3).
    """

    schema_version: int = 1

    not_a_hand: bool = False
    refusal_reason: str | None = None

    tournament_name: str | None = None
    hand_no: str | None = None
    level: int | None = None

    sb: float | None = None
    bb: float | None = None
    blind_unit: Unit | None = None

    ante_pool_shown: float | None = None
    ante_per_player_shown: float | None = None
    ante_unit: Unit | None = None

    max_seats: int | None = None
    players: list[SeenPlayer] = []

    board: list[str] = []
    pot_shown: float | None = None
    pot_unit: Unit | None = None

    actions: list[SeenAction] = []
    # Видно ли, чем рука кончилась: вскрытые карты соперников, баннер выигрыша,
    # строка «выиграл банк». Отсюда код выводит признак полноты
    # (`RawHand.completeness`) — тип экрана при этом не вводится, решение C1.
    showdown_seen: bool = False
    result_seen: bool = False
    winner_seats: list[int] = []

    unsure_fields: list[str] = []


__all__ = ["SeenAction", "SeenPlayer", "Unit", "VisionReading"]
