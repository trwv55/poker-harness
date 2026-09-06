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


class EvInterval(BaseModel):
    """EV решения: точка, интервал по моделям колла и потолок цены ошибки.

    Отвечает на три вопроса разом, и ровно затем и заведён: какого знака и
    порядка величина (`point_bb`), насколько вывод зависит от допущения о поле
    (ширина `low_bb`..`high_bb` — она и есть мера маргинальности), и сколько
    максимум стоит неверный выбор (`cost_ceiling_bb`). Отказ «одного надёжного
    числа здесь нет» прятал все три.

    `near_zero` — интервал лежит по обе стороны нуля, то есть при одних моделях
    колла лучше входить, при других пасовать. Признак хранится, а не выводится
    из знаков `low_bb`/`high_bb`: форма, где диапазон оппонента взят из
    равновесия, а не угадан, получает обычный вердикт даже при интервале через
    ноль — там вывод держит равновесие, а не интервал (`preflop.zone_for`,
    `test_hu_equilibrium_shape_is_strict_even_when_bracket_unstable`).
    """

    point_bb: float
    low_bb: float
    high_bb: float
    near_zero: bool = False

    @property
    def cost_ceiling_bb(self) -> float:
        """Сколько максимум стоит неверный выбор — по модулю худшего конца.

        Пас при `high_bb > 0` стоит не больше `high_bb`, вход при `low_bb < 0` —
        не больше `|low_bb|`; потолок в любую сторону и есть максимум модулей.
        Свойство, а не поле: величина полностью определена концами интервала, и
        отдельно записанная она могла бы с ними разойтись.
        """
        return max(abs(self.low_bb), abs(self.high_bb))

    @model_validator(mode="after")
    def _point_lies_inside(self) -> EvInterval:
        """Точка обязана лежать внутри интервала, а нижний конец — не выше верхнего.

        Интервал строится как размах по моделям колла ВКЛЮЧАЯ саму модель, и
        точка вне собственного интервала означала бы, что одно из двух чисел
        посчитано не по тому набору.
        """
        if self.low_bb > self.high_bb:
            raise ValueError(
                f"нижний конец интервала {self.low_bb} выше верхнего {self.high_bb}"
            )
        if not self.low_bb <= self.point_bb <= self.high_bb:
            raise ValueError(
                f"точечная оценка {self.point_bb} лежит вне своего интервала "
                f"[{self.low_bb}, {self.high_bb}]"
            )
        return self


class PointVerdict(BaseModel):
    dp_index: int
    street: Street
    spot: SpotKind
    zone: Zone
    action_taken: str
    best_action: str
    ev_diff_bb: float  # <0 = потеря
    interval: EvInterval | None = None
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
    ev_diff_bb: float  # < -0.1bb для расхождения; 0.0 для точки «около нуля»
    zone: Zone
    interval: EvInterval | None = None


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

    `points_total`/`points_judged` — покрытие в точках решения, а не в руках:
    сколько точек героя было во всех разобранных руках файла и по скольким из
    них есть вердикт (`error_cost.is_judged`). Без этой пары пустой `items`
    читается как «сыграно чисто», хотя означать может «оценить почти ничего не
    удалось» — точка без вердикта в список не попадает по построению. Считает
    их `scan_tournament`, показывает — `presentation.scan_summary_msg` (строка
    печатается всегда). Руки, пропущенные политикой отказа, в `points_total` не
    входят: у них не посчитана ни одна точка, и они названы `hands_failed`.

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
    # Точки «около нуля»: интервал EV лежит по обе стороны нуля, оба варианта
    # допустимы, цена ноль. В `items` они не входят и входить не должны — там
    # список РАСХОЖДЕНИЙ, а тут расхождения нет, — но и молчать о них нельзя:
    # прежде такая точка не доходила до игрока вовсе (отказ «одного надёжного
    # числа здесь нет»), и вместе с числом пропадали знак, порядок величины и
    # потолок цены. Умолчание `[]` — ради сводок, записанных в
    # `tournaments.scan_summary` до появления поля.
    close_calls: list[ScanItem] = []
    hands_failed: int = 0
    # Умолчание 0 — ради сводок, записанных в `tournaments.scan_summary` до
    # появления этих полей: они читаются тем же типом.
    points_total: int = 0
    points_judged: int = 0


class PlayerStats(BaseModel):
    """Статистика игрока — только счётчики; доли выводятся из них свойствами.

    Хранятся ЧИСЛИТЕЛИ И ЗНАМЕНАТЕЛИ, а не готовые проценты: две статистики с
    разными знаменателями (VPIP считается от всех раздач, сдача на продолженную
    ставку — от числа таких ставок против героя) складываются в среднее по
    нескольким турнирам только через счётчики. Сложить проценты и поделить на
    число турниров значило бы взвесить турнир из 12 раздач наравне с турниром
    из 300.

    Формулы, по которым считается каждый счётчик, записаны в докстрингах
    `analysis/player_stats.py` — там же названы тесты, которые их пришпиливают.
    Доля отсутствует (`None`), когда знаменатель нулевой: «0 из 0» — это не
    ноль процентов, а отсутствие данных.
    """

    hands: int = 0
    vpip: int = 0
    pfr: int = 0
    reraise: int = 0
    reraise_chances: int = 0
    fold_to_cbet: int = 0
    cbet_faced: int = 0

    @staticmethod
    def _share(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else 100.0 * numerator / denominator

    @property
    def vpip_pct(self) -> float | None:
        return self._share(self.vpip, self.hands)

    @property
    def pfr_pct(self) -> float | None:
        return self._share(self.pfr, self.hands)

    @property
    def reraise_pct(self) -> float | None:
        return self._share(self.reraise, self.reraise_chances)

    @property
    def fold_to_cbet_pct(self) -> float | None:
        return self._share(self.fold_to_cbet, self.cbet_faced)
