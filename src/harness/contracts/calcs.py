"""Словарь параметризованных расчётов: имена, параметры, форма результата.

Закрытый набор: расчёт выбирается ИМЕНЕМ из `CalcName`, параметры приходят
типом, результат называет расчёт, которым получен. Живёт в контрактах по той же
причине, что `history.py`: типы читают два пакета по разные стороны правила
зависимостей — `memory` (выборка SQL) и `presentation` (печать), — а
композицию делает `harness.calcs`, который тянет и то и другое.

**Каждая величина приходит со своим знаменателем.** Доля наружу отдаётся
свойством `Measurement.share`, а хранятся числитель и знаменатель: результат, в
котором осталась только доля, невозможно ни сложить с другим окном, ни прочесть
как «5 из 7».

**Доверительного интервала здесь нет ни у одного типа.** Решение владельца
2026-09-09: интервал считается и решает, утверждать или нет, но наружу не
выводится. Отсутствие поля — не умолчание, а механизм: числа, которого нет в
результате, не может быть и в тексте (`explanation.faithfulness` сверяет числа
текста с выходом инструмента), поэтому граница держится формой типа, а не
договорённостью (`test_no_result_carries_an_interval_bound`).

**Оценка только там, где эталон вычислим.** `FrequencyResult` — измерение без
вердикта: постфлопного эталона в проекте нет. Вердикт даёт
`ThresholdResult`, и порог у него приходит либо арифметикой от размера ставки
(`BetSizeThreshold`), либо измерением поля (`FieldThreshold`) — третьего
источника в наборе нет.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from harness.contracts.analysis import SpotKind
from harness.contracts.history import LeakStat
from harness.contracts.raw import Street

__all__ = [
    "POSITIONS",
    "BetSizeThreshold",
    "CalcName",
    "CalcParams",
    "CalcResult",
    "CoverageParams",
    "CoverageResult",
    "DefenseParams",
    "DefenseResult",
    "FieldThreshold",
    "FrequencyResult",
    "FrequencyStat",
    "HeroFrequencyParams",
    "LeaksParams",
    "LeaksResult",
    "Measurement",
    "OpponentFrequencyParams",
    "PointFilter",
    "Subject",
    "ThresholdOutcome",
    "ThresholdParams",
    "ThresholdResult",
    "ThresholdSide",
    "Window",
]


# Позиции, по которым фильтруются расчёты. Список повторяет тот, которым
# нормализатор подписывает места (`normalizer.POSITIONS_BY_COUNT`), и повторяет
# его ЦЕЛИКОМ: равенство двух множеств пришпилено
# `test_positions_are_the_ones_the_normalizer_writes`, поэтому новая позиция в
# нормализаторе краснит этот файл, а не тихо перестаёт находиться фильтром.
# Копия, а не импорт: `contracts` не импортируют ни один пакет конвейера.
POSITIONS: frozenset[str] = frozenset(
    {"BTN", "SB", "BB", "UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO"}
)


class CalcName(StrEnum):
    """Имя расчёта. Закрытый набор: маршрутизация выбирает из него и только из него."""

    HERO_FREQUENCY = "hero_frequency"
    OPPONENT_FREQUENCY = "opponent_frequency"
    COVERAGE = "coverage"
    LEAKS = "leaks"
    DEFENSE_FREQUENCY = "defense_frequency"
    FREQUENCY_VS_THRESHOLD = "frequency_vs_threshold"


class FrequencyStat(StrEnum):
    """Названная частота — та, у которой в `PlayerStats` есть свой знаменатель.

    Улица и спот здесь не отдельные фильтры, а часть самой величины: у продолженной
    ставки флопа знаменатель — число возможностей поставить её на флопе, у второго
    барреля — число ставок на тёрне после своей ставки на флопе. Перемножить
    название частоты на улицу значило бы попросить знаменатель, которого никто не
    определял (`analysis/player_stats.py`, докстринги формул).
    """

    VPIP = "vpip"
    PFR = "pfr"
    RERAISE = "reraise"
    FOLD_TO_CBET = "fold_to_cbet"
    CBET_FLOP = "cbet_flop"
    BARREL_TURN = "barrel_turn"
    BARREL_RIVER = "barrel_river"
    SHOWDOWN = "showdown"


class Subject(StrEnum):
    """Чья частота измеряется.

    `FIELD` — все оппоненты героя в его же руках, сложенные в одно число: сшивка
    для него не нужна, и это измерение поля, а не теория (иерархия источников
    эталона, решение владельца 2026-09-09).
    """

    HERO = "hero"
    OPPONENT = "opponent"
    FIELD = "field"


class ThresholdSide(StrEnum):
    """Какая из двух арифметических частот берётся порогом.

    Порог из размера ставки — это пара чисел, дающая в сумме единицу, и сравнить
    частоту сдачи с частотой защиты значило бы сравнить её с дополнением к
    нужному числу. Сторона называется параметром, а не угадывается по имени
    статистики.
    """

    DEFEND = "defend"
    FOLD = "fold"


class ThresholdOutcome(StrEnum):
    """Итог сравнения с порогом.

    `UNDECIDED` — интервал накрывает порог. Это не «данных мало», а состояние,
    к которому прилагается число недостающих наблюдений
    (`ThresholdResult.observations_needed`).
    """

    ABOVE = "above"
    BELOW = "below"
    UNDECIDED = "undecided"


class Window(BaseModel):
    """Окно, из которого расчёт отвечает: вечер, начало отсчёта, или всё.

    Обе величины — колонки таблицы `sessions` (`id`, `started_at`), а не время
    внутри руки: вечер в этом продукте и есть единица времени
    (SESSIONS_UX, «сессия = вечер»), а у таблицы `hands` собственной отметки
    времени нет вовсе.

    Окно едет в результате каждого расчёта, который от него зависит: отчёт,
    не называющий окно, читается как ответ по всей игре.
    """

    session_id: int | None = None
    since: datetime | None = None


class PointFilter(BaseModel):
    """Фильтры по точкам решения — колонки `decision_points`, поле в поле."""

    street: Street | None = None
    spot: SpotKind | None = None
    position: str | None = None
    window: Window = Window()


class Measurement(BaseModel):
    """Числитель и знаменатель одной величины.

    Доля — свойство, а не поле: записанная рядом, она могла бы разойтись со
    своей парой, а расхождение подписи со знаменателем — та самая ошибка,
    ради которой счётчики и хранятся парами (`PlayerStats`).
    """

    numerator: int
    denominator: int

    @property
    def share(self) -> float | None:
        """Доля от нуля до единицы; `None` при нулевом знаменателе.

        «0 из 0» — не ноль процентов, а отсутствие наблюдений; то же правило и
        та же причина, что у `PlayerStats._share`.
        """
        return None if self.denominator == 0 else self.numerator / self.denominator


class FrequencyResult(BaseModel):
    """Измеренная частота — без оценки.

    Оценки здесь нет и не будет: эталона для постфлопных частот в проекте нет,
    а спрашивать его у модели запрещено (решение владельца). Сравнение с
    порогом — отдельный расчёт (`ThresholdResult`), и порог у него берётся из
    вычислимого источника.
    """

    calc: CalcName
    subject: Subject
    stat: FrequencyStat
    opponent_id: int | None = None
    position: str | None = None
    window: Window
    measurement: Measurement


class CoverageResult(BaseModel):
    """Сколько точек под фильтром, сколько судимых, во сколько обошлись.

    Три величины, три РАЗНЫХ знаменателя, и вложены они друг в друга:

    * `judged` — судимых из всех попавших под фильтр (`contracts.is_judged`);
    * `priced` — из судимых те, у которых расхождение отрицательно, то есть те,
      что и составили сумму;
    * `loss_bb` — сумма по `priced` и только по ним.

    Сумма и покрытие НЕ смешиваются намеренно: «разобрано» и «цена посчитана» —
    разные множества, и одно число на оба означало бы, что нули точек без цены
    поданы как «здесь не потеряно» (решение владельца о третьем исходе точки).
    """

    calc: CalcName
    filter: PointFilter
    judged: Measurement
    priced: Measurement
    loss_bb: float


class LeaksResult(BaseModel):
    """Типы ликов с ценой плюс окно, из которого они посчитаны.

    `judged` едет рядом со списком не для украшения: лики ранжируются по цене,
    значит в список попадают только точки с ценой, и без своего знаменателя
    список читается как полная картина игры.
    """

    calc: CalcName
    window: Window
    judged: Measurement
    leaks: list[LeakStat]


class DefenseResult(BaseModel):
    """Требуемая частота защиты из размера ставки — арифметика, данных не требует.

    `pot_before` — банк ДО ставки соперника, `bet` — сама ставка. Обозначение
    другое, чем у `pot_odds.required_equity` (там банк уже со ставкой внутри), и
    поэтому названо здесь явно: `required_equity` в этом результате посчитана
    тем самым вызовом, с приведённым к его виду банком.

    `defend_frequency` + `fold_frequency` = 1.
    """

    calc: CalcName
    pot_before: int
    bet: int
    defend_frequency: float
    fold_frequency: float
    required_equity: float


class ThresholdResult(BaseModel):
    """Частота против порога: утверждение либо число недостающих наблюдений.

    `outcome` — по доверительному интервалу измерения, а не по числу
    наблюдений: утверждение выдаётся, только когда интервал целиком по одну
    сторону порога. Сам интервал в результат не входит (см. докстринг модуля).

    `reference` заполнен, когда порог — измерение поля: у порога тогда свой
    знаменатель, и без него «порог 61%» выглядел бы величиной, которую кто-то
    знает точно.

    `observations_needed` — сколько наблюдений у ИЗМЕРЯЕМОГО нужно, чтобы при
    той же доле знак определился; заполнен ровно при `UNDECIDED`
    (`test_observations_needed_is_filled_exactly_when_undecided`), и `None`
    внутри `UNDECIDED` означает, что при этой доле не хватит никакого числа.
    """

    calc: CalcName
    subject: Subject
    stat: FrequencyStat
    opponent_id: int | None = None
    position: str | None = None
    window: Window
    measurement: Measurement
    threshold: float
    threshold_source: Literal["bet_size", "field"]
    reference: Measurement | None = None
    outcome: ThresholdOutcome
    observations_needed: int | None = None


class HeroFrequencyParams(BaseModel):
    calc: Literal[CalcName.HERO_FREQUENCY] = CalcName.HERO_FREQUENCY
    stat: FrequencyStat
    position: str | None = None
    window: Window = Window()


class OpponentFrequencyParams(BaseModel):
    """Частота оппонента: по сшивке (`opponent_id`) либо по полю целиком.

    `opponent_id is None` — поле: все оппоненты героя в его руках сложены в одно
    число. Это источник по умолчанию (решение владельца 2026-09-09): сшивка для
    него не нужна, копится быстро, и за столом против незнакомого места игрок
    тоже не знает, кто перед ним.
    """

    calc: Literal[CalcName.OPPONENT_FREQUENCY] = CalcName.OPPONENT_FREQUENCY
    stat: FrequencyStat
    opponent_id: int | None = None
    position: str | None = None
    window: Window = Window()


class CoverageParams(BaseModel):
    calc: Literal[CalcName.COVERAGE] = CalcName.COVERAGE
    filter: PointFilter = PointFilter()


class LeaksParams(BaseModel):
    calc: Literal[CalcName.LEAKS] = CalcName.LEAKS
    window: Window = Window()


class DefenseParams(BaseModel):
    """`pot_before` — банк ДО ставки соперника; `bet` — ставка. Оба в фишках."""

    calc: Literal[CalcName.DEFENSE_FREQUENCY] = CalcName.DEFENSE_FREQUENCY
    pot_before: int
    bet: int


class BetSizeThreshold(BaseModel):
    """Порог арифметикой от размера ставки — данных не требует вовсе.

    Одинаков для любого соперника, поставившего в этот размер, и верен при
    допущении, что тот ставит широко: против места, открывающего редко,
    пересбрасывать правильно, и вывод «вы сдаётесь слишком часто» был бы
    вредным (иерархия источников эталона, пункт 3).
    """

    source: Literal["bet_size"] = "bet_size"
    pot_before: int
    bet: int
    side: ThresholdSide


class FieldThreshold(BaseModel):
    """Порог — измеренная частота поля по той же статистике и тем же фильтрам.

    Неопределённостей тогда две, и сравнение идёт интервалом РАЗНОСТИ, а не
    измерения с точкой (`analysis.frequency.compare_to_reference`).
    """

    source: Literal["field"] = "field"


class ThresholdParams(BaseModel):
    calc: Literal[CalcName.FREQUENCY_VS_THRESHOLD] = CalcName.FREQUENCY_VS_THRESHOLD
    subject: Subject
    stat: FrequencyStat
    opponent_id: int | None = None
    position: str | None = None
    window: Window = Window()
    threshold: Annotated[BetSizeThreshold | FieldThreshold, Field(discriminator="source")]


CalcParams = Annotated[
    HeroFrequencyParams
    | OpponentFrequencyParams
    | CoverageParams
    | LeaksParams
    | DefenseParams
    | ThresholdParams,
    Field(discriminator="calc"),
]

CalcResult = (
    FrequencyResult | CoverageResult | LeaksResult | DefenseResult | ThresholdResult
)
