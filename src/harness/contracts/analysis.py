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


# Машинный ключ причины, по которой точка осталась без вердикта. Нужен ровно
# затем, чтобы `presentation` мог перевести причину в слова игрока, не разбирая
# внутренние формулировки ядра: те написаны для разбора, а не для чтения вслух.
UNJUDGED_DECISION_NOT_TAKEN = "decision_not_taken"


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
    # Названо явно здесь, а не только в докстринге класса: изложение обязано
    # подписать это число множеством, по которому оно посчитано, — судимыми
    # точками, — а не «всеми точками разбора» (тогда подпись накрывает и точки
    # без вердикта, чей ноль означает «не посчитано») и не «суммой списка ниже»
    # (тогда игрок видит два разных числа под одной подписью).
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
    ноль процентов, а отсутствие данных. Другого порога у долей нет: при любом
    ненулевом знаменателе возвращаются и доля, и он сам, и читатель видит, из
    скольких наблюдений она взята
    (`test_a_single_observation_is_returned_with_its_denominator`).

    Тип описывает счётчики ЛЮБОГО места за столом, а не только героя:
    `player_stats(hands, label)` и `player_stats_by_label(hands)` возвращают
    его же. Метки внутри нет — у одного места одна статистика, а кому она
    принадлежит, знает тот, кто её запросил (ключ словаря).
    """

    hands: int = 0
    vpip: int = 0
    pfr: int = 0
    reraise: int = 0
    reraise_chances: int = 0
    fold_to_cbet: int = 0
    cbet_faced: int = 0
    # Постфлоп-частоты. Каждая — пара «числитель, знаменатель», как и всё выше;
    # знаменатели у них РАЗНЫЕ и вложены друг в друга: возможность второго
    # барреля есть только у того, кто поставил на флопе, третьего — только у
    # поставившего на тёрне. Поэтому `barrel_river_chances` не больше
    # `barrel_turn_chances`, и это не потеря данных, а определение величины
    # (`test_a_barrel_needs_a_bet_on_the_street_before`,
    # `test_every_numerator_stays_within_its_own_denominator`).
    cbet_flop: int = 0
    cbet_flop_chances: int = 0
    barrel_turn: int = 0
    barrel_turn_chances: int = 0
    barrel_river: int = 0
    barrel_river_chances: int = 0
    showdowns: int = 0
    flops_seen: int = 0

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

    @property
    def cbet_flop_pct(self) -> float | None:
        return self._share(self.cbet_flop, self.cbet_flop_chances)

    @property
    def barrel_turn_pct(self) -> float | None:
        return self._share(self.barrel_turn, self.barrel_turn_chances)

    @property
    def barrel_river_pct(self) -> float | None:
        return self._share(self.barrel_river, self.barrel_river_chances)

    @property
    def showdown_pct(self) -> float | None:
        return self._share(self.showdowns, self.flops_seen)


class LevelLine(BaseModel):
    """Один уровень блайндов: сколько раздач и стек героя на входе и на выходе.

    Стек в bb того уровня, к которому строка относится: bb растут по ходу
    турнира, и один и тот же стек в фишках на разных уровнях — разная глубина.
    """

    level: int
    hands: int
    start_bb: float
    end_bb: float


class StackTrajectory(BaseModel):
    """Траектория стека по уровням и переломная точка — максимум стека.

    Переломная точка определена механически: раздача с наибольшим стеком героя
    НА ВХОДЕ (`peak_bb`, в bb своего уровня); при равенстве — самая ранняя.
    Всё, что после неё, — спуск с этого максимума, и `hands_after_peak` говорит,
    за сколько раздач он пройден. Никакого суждения «до этого шло хорошо» здесь
    нет: это арифметика максимума, а слова о ней — задача 21.
    """

    levels: list[LevelLine]
    start_bb: float
    final_bb: float
    peak_level: int
    peak_bb: float
    peak_hand_no: str
    hands_after_peak: int


class AllInEvent(BaseModel):
    """Раздача, решённая олл-ином с участием героя, — с исходом.

    Не только собственный шов героя: раздача, где герой уравнял чужой олл-ин
    (или покрыл его своей ставкой), — такой же олл-ин по сути и по цене. На
    измеренной фикстуре именно так и кончился турнир: герой с KK уравнял шов,
    имея стек больше, проиграл 18 bb и вылетел следующей раздачей — по
    признаку «пометка олл-ина на действии героя» эта раздача не была бы названа
    олл-ином вовсе. Точное правило — `analysis/tournament.py`,
    `_all_in_showdown`.

    `stack_before_bb` — с чем герой вошёл в раздачу, `delta_bb` — изменение
    стека за неё. Знак `delta_bb` и есть исход: отдельного поля «выиграл» в
    контракте нет, оно вычислялось бы из того же числа и могло бы с ним
    разойтись (то же правило, что у `EvInterval.cost_ceiling_bb`).
    """

    hand_no: str
    hand_index: int | None
    level: int
    hero_class: str
    stack_before_bb: float
    delta_bb: float
    showdown: bool


class ChipMove(BaseModel):
    """Строка «где ушли фишки»: раздача, что в ней было, цена в bb.

    `cost_bb` положительно и означает «столько стек потерял». Это ФИШКИ, а не
    EV: цена расхождения (`ScanItem.ev_diff_bb`) считается против диапазона на
    момент решения, потеря стека — по факту раздачи, и смешивать их нельзя
    (`EvSplit`).
    """

    hand_no: str
    hand_index: int | None
    level: int
    hero_class: str
    last_street: Street
    all_in: bool
    showdown: bool
    cost_bb: float


class Finding(BaseModel):
    """Повторяющийся паттерн среди ОЦЕНЁННЫХ точек — с ценой и покрытием рядом.

    Ключ паттерна — тройка «спот · сыгранное действие · лучшее действие»:
    находка утверждает, что одна и та же развилка сыграна одинаково несколько
    раз, а не что несколько разных рук чем-то похожи.

    `seen_before` — сколько раз тот же ключ встречался в прошлых турнирах этого
    игрока (`seen_before_tournaments` — в скольких именно). Это и есть «тот
    самый паттерн, который мы разбирали»: ссылка на его собственную историю, а
    не догадка о ней.
    """

    spot: SpotKind
    action_taken: str
    best_action: str
    zone: Zone
    count: int
    total_cost_bb: float
    hand_nos: list[str]
    seen_before: int = 0
    seen_before_tournaments: int = 0


class EvSplit(BaseModel):
    """Честный счёт: цена судимых расхождений ОТДЕЛЬНО от дисперсии и несудимого.

    Все четыре числа — положительные величины потерь («столько ушло»), чтобы
    рядом стоящие строки отчёта нельзя было прочитать со случайно разными
    знаками. Знак при них ставит изложение, а не контракт.

    Величины меряют РАЗНОЕ и потому не складываются между собой — ни здесь, ни
    в изложении:

    * `judged_loss_bb` — EV-цена расхождений на момент решения, по всем судимым
      точкам турнира. Это модуль `ScanSummary.total_loss_bb`
      (`test_judged_loss_is_the_scan_total_by_magnitude`) — единственное число
      здесь, посчитанное против диапазона, а не по факту раздачи.
    * `chips_in_gap_hands_bb` — фишки, потерянные в раздачах, где расхождение
      найдено. Равняться `judged_loss_bb` оно не обязано и обычно не равняется:
      расхождение стоит своей EV-цены, раздача — своих фишек.
    * `chips_in_lost_allins_bb` — фишки, потерянные в проигранных олл-инах, где
      расхождения нет: решение расчёт не оспаривает, а фишки ушли. Это
      дисперсия, и ошибкой она не считается (CLAUDE.md: правильный вход,
      проигравший по случайности, ошибкой не считается;
      `test_a_lost_all_in_without_a_gap_is_variance_not_error`).
    * `chips_elsewhere_bb` — всё остальное потерянное: блайнды, анте, постфлоп.
      Про эти фишки расчёт не говорит ничего.

    Три «фишечных» слагаемых — разбиение ВСЕХ потерянных фишек по раздачам:
    каждая раздача попадает ровно в одно из них, поэтому их сумма равна
    суммарной потере стека за турнир
    (`test_chip_buckets_are_a_partition_of_every_lost_chip`).

    `points_judged`/`points_total` — покрытие, перенесённое из сводки скана как
    есть: без него любая сумма выше читается как полная цена турнира.
    """

    judged_loss_bb: float
    points_judged: int
    points_total: int
    chips_in_gap_hands_bb: float
    chips_in_lost_allins_bb: float
    chips_elsewhere_bb: float


class TournamentReport(BaseModel):
    """Отчёт по турниру: факты и статистика, посчитанные кодом (задача 21 — слова).

    Собирается целиком из истории рук и сводки скана; ни одного числа отсюда не
    приходит от модели. Задача 21 получает этот объект на вход как есть.

    `baseline`/`baseline_tournaments` — среднее игрока по ВСЕМ его турнирам в
    базе, включая этот. При единственном турнире среднее совпало бы с самим
    турниром, и сравнивать не с чем: тогда `baseline` пуст (`None`), а не равен
    `stats` (`test_baseline_is_absent_when_the_player_has_a_single_tournament`).
    """

    schema_version: int = 1
    hands_total: int
    hands_failed: int
    levels_played: int
    first_level: int
    last_level: int
    duration_minutes: int
    stats: PlayerStats
    baseline: PlayerStats | None
    baseline_tournaments: int
    trajectory: StackTrajectory
    all_ins: list[AllInEvent]
    chip_moves: list[ChipMove]
    findings: list[Finding]
    ev: EvSplit
