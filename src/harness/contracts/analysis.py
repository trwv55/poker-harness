"""Результат аналитического ядра: по одному вердикту на точку решения.

`Zone` различает точки, где применяется точное правило (`strict`, зона
пуш/фолд, шов и т.п.), от точек, где вывод строится на допущении о диапазоне
оппонента (`assuming`) — см. `Assumption`. `AnalysisResult` собирает вердикты
по всем точкам решения одной руки и суммарные потери в bb.

`ScanItem`/`ScanSummary` — тот же словарь, но на уровне турнира: результат
скана файла (`analysis/scan.py` его СЧИТАЕТ, здесь он только ОПИСАН). Живут
здесь, а не рядом со счётчиком (round 5, Item L): их читают `memory.repos`
(колонка `tournaments.scan_summary`) и `presentation.messages` (сводка
игроку) — оба вне пакета `analysis`, и импорт типа из `analysis.scan` тянул
за собой `pokerkit` и `eval7` в образ бота, которому считать нечего.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, model_validator

from harness.contracts.ranges import Range
from harness.contracts.raw import Street


class Zone(StrEnum):
    STRICT = "strict"
    ASSUMING = "assuming"


class SpotKind(StrEnum):
    PUSHFOLD_UNOPENED = "pushfold_unopened"
    PUSHFOLD_FACING_SHOVE = "pushfold_facing_shove"
    PREFLOP_OTHER = "preflop_other"
    POSTFLOP = "postflop"


class Assumption(BaseModel):
    range: Range
    source: str
    note: str = ""


class PointVerdict(BaseModel):
    dp_index: int
    street: Street
    spot: SpotKind
    zone: Zone
    action_taken: str
    best_action: str
    ev_diff_bb: float  # <0 = потеря
    assumption: Assumption | None = None
    tools: list[str] = []
    detail: dict[str, Any] = {}

    @model_validator(mode="after")
    def _assumption_matches_zone(self) -> PointVerdict:
        """Допущение заполнено тогда и только тогда, когда зона `assuming`.

        Правило держится в контракте, а не в договорённости: вывод в зоне
        «предполагая» обязан нести показанное игроку допущение, а вывод, помеченный
        «строго», не имеет права его нести — иначе продукт либо скрывает догадку,
        либо выдаёт догадку за точный расчёт. Инвариант касается всех, кто
        конструирует вердикты (ядро, скан, изложение), поэтому проверяет его сам
        тип, а не тест конкретного модуля.
        """
        if (self.assumption is not None) != (self.zone is Zone.ASSUMING):
            raise ValueError(
                f"допущение должно быть заполнено ровно при зоне assuming: "
                f"зона {self.zone}, допущение {'есть' if self.assumption else 'нет'}"
            )
        return self


class AnalysisResult(BaseModel):
    schema_version: int = 1
    hand_no: str
    points: list[PointVerdict]
    ranked: list[int] = []
    total_ev_loss_bb: float = 0.0


class ScanItem(BaseModel):
    """Одна точка расхождения в сводке скана — по цене, с зоной доверия рядом."""

    hand_no: str
    hand_index: int | None
    hero_class: str
    spot: SpotKind
    action_taken: str
    best_action: str
    ev_diff_bb: float  # < -0.1bb — иначе точка не попала бы в список
    zone: Zone


class ScanSummary(BaseModel):
    """Сводка по турниру: сколько рук, сколько с решением, список расхождений.

    `total_loss_bb` — суммарная потеря по ВСЕМ судимым точкам файла (то же, что
    дал бы `sum(total_ev_loss_bb(...))` по каждой руке), а не только по тем, что
    попали в `items`: список ограничен порогом 0.1bb ради актуальности, но общая
    цена турнира не должна тихо терять мелкие расхождения. Соответственно
    `total_loss_bb` по модулю обычно ЧУТЬ БОЛЬШЕ суммы `ev_diff_bb` из `items` —
    это не расхождение чисел, а сумма с порогом отображения против суммы без него.

    `hands_failed` — руки, пропущенные политикой отказа скана (см. докстринг
    модуля `analysis/scan.py`): расчёт разошёлся с движком по деньгам, и цену
    решения на такой руке доверять нельзя. Поле не в брифе задачи 13 дословно,
    но необходимо, чтобы пропуск был виден, а не тихим.

    `items` НЕ ограничен по длине — это данные, и `tournaments.scan_summary`
    хранит их целиком. Потолок показа живёт в изложении
    (`presentation.messages._MAX_RENDERED_SCAN_ITEMS`, round 5, Item J): предел
    ставит Телеграм, а не аналитика.
    """

    hands_total: int
    hands_with_decision: int
    items: list[ScanItem]
    # Полная цена турнира по ВСЕМ судимым точкам файла, включая те дешевле
    # 0.1bb, что не попали в `items` ниже, — НЕ сумма `ev_diff_bb` по `items`.
    # Названо явно здесь, а не только в докстринге класса: изложение (задача
    # 17) обязано подписать это число как «суммарная потеря по всем точкам
    # разбора», а не как «сумма списка ниже» — иначе игрок увидит два разных
    # числа рядом и решит, что одно из них ошибка.
    total_loss_bb: float
    hands_failed: int = 0
