"""Вердикт по префлоп-точке пуш-фолд-зоны: EV решения, зона доверия, цена.

Всё считается **против диапазона на момент решения**, а не против вскрытых карт:
верный вход, проигравший по случайности, ошибкой не является. Карты оппонентов
этот модуль не читает вообще — ни из `dealt`, ни из `showdowns`.

Вердикты в chip-EV (bb). ICM в v1 не применяется: в hand history GG нет призовой
структуры, а считать поправку не из чего — инструмент `icm_equities` ждёт
источника выплат.

**Анте стола входит в решаемую игру.** Продукт заявлен для MTT с анте, и брать
равновесие игры без анте значило бы судить не ту раздачу: на фикстуре семь анте
по 0.125bb дают 0.875bb мёртвых денег сверх 1.5bb блайндов, и равновесие без них
на 10bb теснее на 12.4 п.п. комбо по пушу и на 13.8 п.п. по коллу. Теснее — значит
верные шовы помечались бы ошибкой, а для тренажёра ложное обвинение хуже
пропущенной ошибки: оно учит пасовать там, где надо входить. Поэтому `nash_hu`
получает `dead_extra_bb` (сумма анте стола), а глубину — уже за вычетом анте.

**Правило зоны.** Точный расчёт есть только там, где колл-диапазон не угадан:
хедз-ап SB против BB решается равновесием `nash_hu`. При трёх и более живых
мультивей-равновесия у нас нет, колл-диапазоны игроков позади моделируются, и
вывод обязан быть помечен как опирающийся на модель — если только он от неё не
зависит. Последнее и проверяет bracket-тест: EV пересчитывается на СЕТКЕ ширин
диапазона, и зона `strict` ставится, только когда вердикт одинаков на всех её
точках и совпадает с вердиктом самой модели. Опрашивать два конца было бы
неверно: EV по ширине не монотонна, и вердикт умеет перевернуться внутри
интервала (см. `_SHOVE_CALL_WIDTHS`). Ошибка, которую правило предотвращает,
ровно одна и односторонняя: объявить устойчивым вывод, который на деле держится
на догадке, значит переоценить собственную уверенность. Решение о зоне целиком
живёт в `zone_for` — снаружи его переопределить негде.

**Живые игроки за героем при колле шова.** `call_shove_ev_bb` считает вскрытие
один на один. Если за героем остались живые, их возможный колл в модель не
входит, и эквити героя завышено — смещение направлено в сторону колла. Правильно
моделировать их — отдельная работа; до неё зона такой точки принудительно
`assuming`, как бы ни повела себя вилка диапазонов.

**Границы применимости, за которыми вердикта нет.** Модель описывает ровно две
формы: «шов или пас в неоткрытый банк» и «колл или пас против ОДНОГО открытого
шова». Всё остальное возвращается без вердикта с названной причиной
(`unpriced_reason`) — глубже 15bb, лимп, война повышений, ре-шов поверх чужого
опена, два олл-ина перед героем. Три последние границы найдены не рассуждением, а
прогоном по 318 реальным рукам: без них модель выдавала самые громкие цифры
разбора (до -11bb) на спотах, которых она не описывает, и все они смещали вердикт
в сторону колла.

**Фолд-эквити.** `shove_ev_bb` гейта не ставит (сигнатура заморожена задачей 11),
поэтому он стоит здесь: `fold_equity_ok` считается для каждого шова и попадает в
`detail`. Сама EV посчитана верно в любом случае — ветка «все сфолдили» получает
свою (нулевую) вероятность, — но изложение (задача 21) не имеет права объяснять
шов словами «оппоненты сфолдят», когда фолда в модели не существует.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from harness.analysis.classifier import (
    SeatSnapshot,
    TableState,
    action_name,
    spot_for,
    table_state,
    unpriced_reason,
)
from harness.analysis.tools.equity import equity_vs_ranges
from harness.analysis.tools.pot_odds import required_equity
from harness.analysis.tools.pushfold import (
    BRACKET_TIGHT,
    MAX_DEAD_EXTRA_BB,
    CallerModel,
    call_shove_ev_bb,
    equity_vs_range_classes,
    fold_equity_ok,
    nash_hu,
    range_of_width,
)
from harness.analysis.tools.pushfold import (
    shove_ev_bb as _shove_ev_bb,
)
from harness.contracts import (
    ActionKind,
    Assumption,
    CanonicalHand,
    DecisionPoint,
    EnrichedHand,
    PointVerdict,
    Range,
    SpotKind,
    Street,
    Zone,
    class_of,
)

# Шаг сетки глубин, на которой берутся равновесные диапазоны. Квантование нужно
# ради кэша: без него каждая рука с произвольной глубиной заводила бы собственный
# файл равновесия и собственный прогон fictitious play.
#
# Шаг выбран по ЗАМЕРУ смещения самой EV, а не по виду диапазона. Ширина
# равновесного диапазона на мелких стеках ходит быстро: между 3.5 и 4.0bb колл
# сужается с 83.6% до 73.6% комбо, то есть на 10 п.п. за полшага 0.5bb — прежнее
# утверждение «меньше процента комбо» было неверным. Но на EV это переносится
# слабее: при ошибке глубины 0.25bb (сетка 0.5) максимум по замеренным парам
# «класс × глубина» — 0.089bb, при ошибке 0.125bb (сетка 0.25) — 0.039bb,
# типичное значение около 0.01bb. Порог, ниже которого расхождение вообще не
# показывается игроку, — 0.1bb (скан турнира, задача 13), поэтому берётся 0.25:
# при нём квантование заведомо не двигает ни вердикт, ни строку сводки.
# Верхняя граница смещения зафиксирована тестом.
_DEPTH_STEP_BB = 0.25

# Окно, в котором пуш-фолд-равновесие вообще имеет смысл. Ниже 1.5bb колл
# равновесия — уже «любые две карты», и уточнять нечего; выше 25bb шов перестаёт
# быть моделью спота. `nash_hu` к тому же не принимает глубину <= 1bb.
_MIN_MODEL_DEPTH_BB = 1.5
_MAX_MODEL_DEPTH_BB = 25.0

# Перебор подмножеств коллеров в `shove_ev_bb` ограничен семью игроками позади
# (2^n веток). Спот с большим числом живых позади остаётся без вердикта — цена,
# посчитанная по урезанному составу оппонентов, была бы правдоподобно неверной.
_MAX_MODELLED_CALLERS = 7

# Шаг сетки мёртвых денег (сумма анте стола в bb). Нужен ровно затем же, зачем
# сетка глубин: без него каждый уровень блайндов заводил бы своё равновесие. Шаг
# 0.05bb — меньше половины типичного анте одного игрока (0.125-0.15bb), то есть
# заведомо мельче зернистости самой структуры.
_DEAD_STEP_BB = 0.05

# Ширины колл-диапазона, на которых опрашивается устойчивость вердикта о шове.
# Это НЕ два конца: EV по ширине не монотонна, и минимум лежит ВНУТРИ интервала.
# Замер на 10.25bb с анте фикстуры и двумя игроками позади — шов K5o: +0.50bb
# против 20% колла, -0.12 против 35%, -0.10 против 50%, -0.01 против 65%, +0.22
# против 80%, +0.64 против 100%. Опрос концов объявил бы вердикт устойчивым на
# интервале, внутри которого он дважды меняет знак. Поэтому опрашивается сетка, и
# `strict` требует одного ответа на ВСЕХ её точках, включая премиум-границу.
#
# ПОЧЕМУ УЗКИЙ КОНЕЦ ИМЕННО ЗДЕСЬ — «только премиум» (`BRACKET_TIGHT`, 3.02%
# комбо), а не что-то шире. Против такого поля шов плюсовой чем угодно: на
# 10.25bb с тремя игроками позади 72o даёт +1.76bb, 32o +1.79, 83o +1.76 — это
# не артефакт неудачно выбранного конца, а факт об игре. Когда оппоненты почти
# никогда не коллируют, фолд-эквити огромно.
#
# Отсюда следствие, которое выглядит как асимметрия, но ею не является: вердикт
# «пас был верен» действительно ЗАВИСИТ от того, что оппоненты коллируют
# достаточно часто, и `assuming` — честный для него ярлык. Пробовали поднять
# узкий конец до 20% — это не убирает допущение, а СОЗДАЁТ его: «никто никогда
# не коллирует теснее 20%». Правдоподобная нижняя граница колл-поведения —
# вопрос эмпирический, и ответят на него популяционные частоты (ступень 3
# порядка разработки). До тех пор широкий интервал — консервативный выбор, а
# сузить его потом можно будет по данным, а не по догадке. Ровно эту ошибку —
# неподтверждённая константа в границе вилки — уже приходилось убирать однажды,
# когда `BRACKET_WIDE` подбирался руками.
#
# Верхняя граница 1.00: там фолд-эквити отсутствует полностью, и шов держится
# только на вскрытии.
_SHOVE_CALL_WIDTHS: tuple[float, ...] = (0.20, 0.35, 0.50, 0.65, 0.80, 1.00)

# Ключ премиум-границы в разбивке EV по ширинам (`detail`). Отдельный, а не доля:
# это названный в спецификации набор JJ+/AK, а не выведенная из данных ширина.
_PREMIUM_KEY = "premium"

# Для колла против шова опрашивается ширина диапазона ШОВЕРА. Здесь узкий конец
# осмыслен и мал: нит, шовящий только премиум, — правдоподобный оппонент, в
# отличие от поля, коллирующего шов тремя процентами рук.
_SHOVER_RANGE_WIDTHS: tuple[float, ...] = (0.05, 0.20, 0.40, 0.70, 1.00)

# Итераций Монте-Карло на ветку с двумя и более коллерами. Хедз-ап-ветки идут по
# предвычисленной таблице эквити и Монте-Карло не трогают вовсе, поэтому платим
# только за настоящий мультивей. 20 000 дают стандартную ошибку эквити около
# 0.0035; ветка с двумя коллерами весит порядка 0.05 вероятности при банке
# порядка 20bb, то есть вклад шума в EV — единицы тысячных bb, на два порядка
# ниже шкалы, на которой вердикт меняется.
_MULTIWAY_ITERATIONS = 20_000

_ASSUMPTION_CALLERS = (
    "колл-диапазоны игроков позади смоделированы: мультивей-равновесия в v1 нет"
)
_ASSUMPTION_SHOVER = (
    "диапазон шова смоделирован равновесным пуш-диапазоном его глубины: "
    "мультивей-равновесия в v1 нет"
)

# Потолок памяти мультивей-кэша эквити. Воркер живёт долго и разбирает турнир за
# турниром, поэтому кэш без границы — утечка. При переполнении он сбрасывается
# целиком: вытеснять по возрасту незачем, повторный расчёт стоит доли секунды, а
# детерминизм от сброса не страдает — те же входы дают то же число.
_EQUITY_MEMO_LIMIT = 4096

_HeroCombo = tuple[str, str]
_RangeKey = tuple[tuple[str, float], ...]
_equity_memo: dict[tuple[_HeroCombo, tuple[_RangeKey, ...]], float] = {}


def _model_equity(hero: _HeroCombo, ranges: Sequence[Range]) -> float:
    """Эквити героя против набора диапазонов: таблицей, где она есть, иначе Монте-Карло.

    Против одного диапазона ответ берётся из той же таблицы 169x169, на которой
    посчитано само равновесие: считать Монте-Карло то, что уже посчитано точнее,
    значит вносить шум и расходиться с диапазоном, против которого судим.
    Мультивей-веток в таблице нет — они идут через сэмплер, и их результат
    запоминается: `shove_ev_bb` перебирает 2^n подмножеств, но разных наборов
    диапазонов среди них всего единицы.
    """
    if len(ranges) == 1:
        return equity_vs_range_classes(class_of(*hero), ranges[0])
    key = (hero, tuple(sorted(tuple(sorted(r.weights.items())) for r in ranges)))
    cached = _equity_memo.get(key)
    if cached is None:
        cached = equity_vs_ranges(hero, list(ranges), iterations=_MULTIWAY_ITERATIONS)
        if len(_equity_memo) >= _EQUITY_MEMO_LIMIT:
            _equity_memo.clear()
        _equity_memo[key] = cached
    return cached


def _depth_key(depth_bb: float) -> float:
    """Глубина, округлённая до сетки и зажатая в окно осмысленности пуш-фолда."""
    clamped = min(max(depth_bb, _MIN_MODEL_DEPTH_BB), _MAX_MODEL_DEPTH_BB)
    return round(clamped / _DEPTH_STEP_BB) * _DEPTH_STEP_BB


def _equilibrium_depth(depth_bb: float) -> float | None:
    """Та же сетка, но без зажима: вне окна равновесие не выдаётся за равновесие."""
    key = round(depth_bb / _DEPTH_STEP_BB) * _DEPTH_STEP_BB
    if not _MIN_MODEL_DEPTH_BB <= key <= _MAX_MODEL_DEPTH_BB:
        return None
    return key


def _dead_key(dead_bb: float) -> float:
    """Мёртвые деньги, округлённые до сетки и зажатые окном `nash_hu`.

    Верхний зажим — деградация вместо падения. Гвард в `nash_hu` стоит против
    перепутанных единиц (фишки вместо bb); здесь величина считается из настоящих
    анте и bb, перепутать нечего, поэтому структура с невероятным анте должна
    дать самое широкое равновесие из посчитанных, а не уронить разбор турнира.
    На фикстурах максимум — 1.2bb при потолке 5.
    """
    clamped = min(max(dead_bb, 0.0), MAX_DEAD_EXTRA_BB)
    return round(clamped / _DEAD_STEP_BB) * _DEAD_STEP_BB


def _table_dead_bb(state: TableState) -> float:
    """Сумма анте всего стола в bb — мёртвые деньги решаемого равновесия.

    Считаются анте ВСЕХ мест, а не только живых: анте спасовавших остаётся в
    банке и разыгрывается наравне с блайндами. Именно эта величина и была
    потеряна в игре без анте — на фикстуре она даёт 0.875bb против 1.5bb
    блайндов, и равновесие без неё оказывается на 12-14 п.п. комбо теснее того,
    в которое играет пользователь.
    """
    return _dead_key(sum(seat.ante for seat in state.seats) / state.bb)


def _call_model(depth_bb: float, dead_bb: float) -> Range:
    """Модель колл-диапазона игрока такой глубины — колл-сторона равновесия."""
    return nash_hu(_depth_key(depth_bb), dead_extra_bb=dead_bb)[1]


def _push_model(depth_bb: float, dead_bb: float) -> Range:
    """Модель диапазона шова игрока такой глубины — пуш-сторона равновесия."""
    return nash_hu(_depth_key(depth_bb), dead_extra_bb=dead_bb)[0]


def _average_range(ranges: Sequence[Range]) -> Range:
    """Диапазон «среднего» оппонента — то, что показывается игроку как допущение."""
    if len(ranges) == 1:
        return ranges[0]
    classes = {cls for rng in ranges for cls in rng.weights}
    weights = {cls: sum(rng.weight(cls) for rng in ranges) / len(ranges) for cls in classes}
    return Range(weights={cls: round(w, 6) for cls, w in weights.items() if w > 0.0})


def zone_for(
    best_tight: str,
    best_wide: str,
    *,
    live_total: int,
    equilibrium: bool = True,
    best_model: str | None = None,
    best_interior: Sequence[str] = (),
    best_behind: Sequence[str] = (),
    unmodelled: str = "",
) -> tuple[Zone, str]:
    """Зона доверия точки и причина, по которой она такая (спека §5.5).

    Здесь решается зона целиком: снаружи не осталось ни одного места, где её
    можно было бы переопределить. Это требование к читаемости, а не к стилю —
    изложение (задача 21) обязано объяснить игроку, почему вывод помечен так, а
    не иначе, и собирать объяснение из двух источников оно не должно.

    Что означают аргументы:

    * `equilibrium` при `live_total == 2` — настоящий хедз-ап либо SB против BB
      после пасов: колл-диапазон взят из равновесия, угадывать нечего. Флаг нужен
      потому, что двое живых сами по себе равновесия не дают: шов из ранней
      позиции, до которого спасовали все, кроме большого блайнда, оставляет за
      столом двоих, но это уже другая игра — там мёртвый малый блайнд и другой
      шовер;
    * `best_tight`, `best_interior`, `best_wide` — вердикты на СЕТКЕ ширин
      диапазона, от узкого конца до широкого. Опрашивается интервал, а не два его
      конца: EV по ширине не монотонна, и вердикт умеет перевернуться внутри
      (K5o на 10.25bb — см. `_SHOVE_CALL_WIDTHS`);
    * `best_model` — вердикт самой модели. Он обязан входить в проверку: вилка
      собрана из долей комбо и модель может лежать за её краем, и тогда концы
      совпали бы между собой, противореча выданному вердикту;
    * `best_behind` — вердикты по оси «живые за героем тоже входят в банк».
      Ось существует там, где этих игроков в модели нет вовсе;
    * `unmodelled` — непустая строка означает, что какое-то измерение задачи в
      модель не попало вообще (живой олл-ин позади, слишком много живых для
      перебора). Тогда проверять нечего и `strict` заявлять не о чем.
    """
    if unmodelled:
        return Zone.ASSUMING, unmodelled
    if live_total == 2 and equilibrium:
        return Zone.STRICT, "хедз-ап SB против BB: колл-диапазон взят из равновесия пуш-фолда"

    by_range = {best_tight, best_wide, *best_interior}
    if best_model is not None:
        by_range.add(best_model)
    by_behind = set(best_behind)

    if len(by_range | by_behind) == 1:
        stable = (
            f"вердикт «{best_tight}» не меняется ни на одной ширине модели диапазона"
            + (", ни если живые за героем тоже войдут в банк" if by_behind else "")
            + " — допущение на него не влияет"
        )
        return Zone.STRICT, stable
    if len(by_range) == 1 and by_behind - by_range:
        moved = ", ".join(sorted(by_behind - by_range))
        return Zone.ASSUMING, (
            f"вердикт «{best_tight}» держится на том, что живые позади не войдут в "
            f"банк: если войдут, лучше «{moved}»"
        )
    if best_model is not None and best_model not in {best_tight, best_wide, *best_interior}:
        return Zone.ASSUMING, (
            f"вердикт «{best_model}» держится на модели диапазона: ни одна ширина "
            f"вилки его не подтверждает"
        )
    if best_tight == best_wide and len(by_range) > 1:
        return Zone.ASSUMING, (
            f"на концах интервала вердикт «{best_tight}», но внутри него он "
            f"переворачивается — устойчивым он не является"
        )
    return Zone.ASSUMING, (
        f"вердикт зависит от модели диапазона: на узком конце лучше «{best_tight}», "
        f"на широком — «{best_wide}»"
    )


def _hero_class(hand: CanonicalHand) -> str | None:
    """Класс карманных карт героя. Вскрытие не читаем — только раздачу."""
    cards = hand.dealt.get(hand.hero_label, [])
    if len(cards) != 2:
        return None
    return class_of(*cards)


def _unjudged(dp: DecisionPoint, spot: SpotKind, reason: str) -> PointVerdict:
    """Точка без вердикта: спот размечен, цена не посчитана.

    Признак «вердикта нет» — пустой `best_action`; на такие точки не ссылается
    ранжирование и не опирается изложение. Зона здесь `strict` не потому, что
    вывод точен, а потому, что вывода нет вовсе: допущение не сделано, и
    инвариант «assumption заполнено тогда и только тогда, когда зона assuming»
    обязан выполняться и на таких точках.
    """
    return PointVerdict(
        dp_index=dp.index,
        street=dp.street,
        spot=spot,
        zone=Zone.STRICT,
        action_taken=action_name(dp),
        best_action="",
        ev_diff_bb=0.0,
        assumption=None,
        tools=[],
        detail={"unjudged": reason},
    )


def _shover(state: TableState) -> SeatSnapshot | None:
    """Кто поставил перед героем ставку, которую тот коллирует или сбрасывает."""
    if state.aggressor is not None and state.aggressor.live:
        return state.aggressor
    rivals = [s for s in state.seats if s.live and s.acted and s.label != state.hero.label]
    if not rivals:
        return None
    return max(rivals, key=lambda s: (s.street_committed, s.label))


def _unopened_verdict(dp: DecisionPoint, en: EnrichedHand, state: TableState) -> PointVerdict:
    """Шов или фолд в неоткрытый банк."""
    spot = SpotKind.PUSHFOLD_UNOPENED
    hero_cls = _hero_class(en.hand)
    if hero_cls is None:
        return _unjudged(dp, spot, "карты героя неизвестны")

    bb = en.hand.bb
    ceiling = state.hero.stack
    behind = [s for s in state.behind_hero if s.behind > 0]
    all_in_behind = len(state.behind_hero) - len(behind)
    if not behind:
        return _unjudged(dp, spot, "позади героя некому коллировать")
    if len(behind) > _MAX_MODELLED_CALLERS:
        return _unjudged(dp, spot, f"игроков позади {len(behind)} — перебор подмножеств ограничен")
    if state.hero.behind <= 0:
        return _unjudged(dp, spot, "у героя не осталось фишек за спиной")

    hero_behind_bb = state.hero.behind / bb
    hero_posted_bb = state.hero.contributed / bb
    pot_dead_bb = state.pot_before / bb
    dead_bb = _table_dead_bb(state)
    # Глубина равновесия — стек ПОСЛЕ анте: само анте входит в игру отдельным
    # слагаемым мёртвых денег, и складывать его в стек значило бы посчитать дважды.
    hero_eff = state.hero.stack_after_ante
    depths = [min(hero_eff, seat.stack_after_ante) / bb for seat in behind]

    def callers(ranges: Sequence[Range]) -> list[CallerModel]:
        return [
            CallerModel(
                call_range=rng,
                behind_bb=seat.behind / bb,
                posted_bb=min(seat.contributed, ceiling) / bb,
            )
            for seat, rng in zip(behind, ranges, strict=True)
        ]

    def ev(ranges: Sequence[Range]) -> float:
        return _shove_ev_bb(
            hero_cls,
            hero_behind_bb,
            pot_dead_bb,
            callers(ranges),
            hero_posted_bb=hero_posted_bb,
            equity_fn=_model_equity,
        )

    model_ranges = [_call_model(depth, dead_bb) for depth in depths]
    ev_model = ev(model_ranges)
    # Опрашивается вся сетка ширин колл-диапазона, а не два конца: EV по ширине
    # не монотонна (см. `_SHOVE_CALL_WIDTHS`). Узкий конец — премиум-граница.
    ev_by_width = {
        _PREMIUM_KEY: ev([BRACKET_TIGHT(_depth_key(depth)) for depth in depths]),
        **{
            str(width): ev(
                [
                    range_of_width(_depth_key(depth), width, dead_extra_bb=dead_bb)
                    for depth in depths
                ]
            )
            for width in _SHOVE_CALL_WIDTHS
        },
    }
    ev_tight = ev_by_width[_PREMIUM_KEY]
    ev_wide = ev_by_width[str(_SHOVE_CALL_WIDTHS[-1])]

    equilibrium_depth = None
    if dp.live_total == 2 and len(behind) == 1 and behind[0].position == "BB":
        equilibrium_depth = _equilibrium_depth(
            min(hero_eff, behind[0].stack_after_ante) / bb
        )

    by_width = [_best_of("shove", value) for value in ev_by_width.values()]
    best_tight, best_wide = by_width[0], by_width[-1]
    best = _best_of("shove", ev_model)
    bracket_shove = "stable" if {*by_width, best} == {best} else "unstable"
    # Вторая ось (входят ли живые позади) здесь уже внутри модели: перебор
    # подмножеств интегрирует их поведение с вероятностями, а сетка ширин двигает
    # сами вероятности от «коллирует каждый пятый» до «коллируют все». Отдельного
    # конца добавлять нечего — кроме случая, когда живой игрок позади в перебор не
    # попал вовсе. Тогда измерение вне модели, и решает это `zone_for`.
    zone, why = zone_for(
        best_tight,
        best_wide,
        live_total=dp.live_total,
        equilibrium=equilibrium_depth is not None,
        best_model=best,
        best_interior=by_width[1:-1],
        unmodelled=(
            f"позади героя {all_in_behind} живых уже в олл-ине: модель шова их не "
            f"перебирает, и их влияние на вердикт не проверено"
            if all_in_behind
            else ""
        ),
    )
    taken = "shove" if state.hero_all_in_after else "fold"
    folds_possible = fold_equity_ok(callers(model_ranges))

    detail: dict[str, object] = {
        "method": "subset_enumeration",
        "bracket": bracket_shove,
        "branches": 2 ** len(behind),
        "fold_equity_ok": folds_possible,
        "ev_shove_bb": round(ev_model, 4),
        "ev_shove_tight_bb": round(ev_tight, 4),
        "ev_shove_wide_bb": round(ev_wide, 4),
        "ev_shove_by_width_bb": {w: round(v, 4) for w, v in ev_by_width.items()},
        "hero_class": hero_cls,
        "depths_bb": [round(_depth_key(depth), 2) for depth in depths],
        "dead_extra_bb": round(dead_bb, 4),
        "zone_reason": why,
    }
    if all_in_behind:
        detail["all_in_behind_ignored"] = all_in_behind
    return PointVerdict(
        dp_index=dp.index,
        street=dp.street,
        spot=spot,
        zone=zone,
        action_taken=taken,
        best_action=best,
        ev_diff_bb=round(_taken_ev(taken, "shove", ev_model) - max(ev_model, 0.0), 6),
        assumption=(
            Assumption(
                range=_average_range(model_ranges),
                source="model:nash_hu_call",
                note=_ASSUMPTION_CALLERS,
            )
            if zone is Zone.ASSUMING
            else None
        ),
        tools=["nash_hu", "shove_ev_bb", "fold_equity_ok"],
        detail=detail,
    )


def _facing_shove_verdict(dp: DecisionPoint, en: EnrichedHand, state: TableState) -> PointVerdict:
    """Колл или фолд против ставки, которая доводит героя до олл-ина."""
    spot = SpotKind.PUSHFOLD_FACING_SHOVE
    hero_cls = _hero_class(en.hand)
    if hero_cls is None:
        return _unjudged(dp, spot, "карты героя неизвестны")

    shover = _shover(state)
    if shover is None:
        return _unjudged(dp, spot, "не удалось определить, кто поставил")
    bb = en.hand.bb
    if state.to_call <= 0 or state.hero.behind <= 0:
        return _unjudged(dp, spot, "доплаты нет либо у героя не осталось фишек")

    dead_bb = _table_dead_bb(state)
    rivals = [seat.stack_after_ante for seat in state.seats if seat.label != shover.label]
    shover_depth_bb = min(shover.stack_after_ante, max(rivals)) / bb
    pot_bb = state.pot_before / bb
    to_call_bb = state.to_call / bb
    hero_bb = state.hero.behind / bb

    def ev(rng: Range) -> float:
        return call_shove_ev_bb(
            hero_cls, hero_bb, rng, pot_bb, to_call_bb, equity_fn=_model_equity
        )

    model_range = _push_model(shover_depth_bb, dead_bb)
    ev_model = ev(model_range)
    width_ranges = {
        width: range_of_width(_depth_key(shover_depth_bb), width, dead_extra_bb=dead_bb)
        for width in _SHOVER_RANGE_WIDTHS
    }
    ev_by_width = {width: ev(rng) for width, rng in width_ranges.items()}
    ev_tight = ev_by_width[_SHOVER_RANGE_WIDTHS[0]]
    ev_wide = ev_by_width[_SHOVER_RANGE_WIDTHS[-1]]

    # Вторая ось вилки: живые за героем. `call_shove_ev_bb` считает вскрытие один
    # на один, поэтому их возможный вход в банк — величина, которой в модели нет
    # вовсе. Считаем противоположный конец: все они коллируют. Эквити героя тогда
    # делится на всех (хуже для него), но и банк растёт на их деньги (лучше), —
    # обе поправки берутся вместе, иначе конец вилки был бы искусственно мрачным.
    # Арифметика та же самая, из замороженного инструмента: подменяется только
    # набор диапазонов на вскрытии и размер банка.
    behind = [
        seat
        for seat in state.seats
        if seat.live and seat.label not in (state.hero.label, shover.label)
    ]
    behind_unmodelled = len(behind) > _MAX_MODELLED_CALLERS
    best_behind: list[str] = []
    ev_behind: float | None = None
    if behind and not behind_unmodelled:
        behind_ranges = [
            _call_model(min(seat.stack_after_ante, state.hero.stack_after_ante) / bb, dead_bb)
            for seat in behind
        ]
        pot_with_behind_bb = (state.pot_before + _extra_from_behind(state, behind)) / bb

        def ev_all(rng: Range) -> float:
            return call_shove_ev_bb(
                hero_cls,
                hero_bb,
                rng,
                pot_with_behind_bb,
                to_call_bb,
                equity_fn=_equity_with_behind(behind_ranges),
            )

        ev_behind = ev_all(model_range)
        ev_behind_by_width = {width: ev_all(rng) for width, rng in width_ranges.items()}
        best_behind = [_best_of("call", ev_behind), *(
            _best_of("call", value) for value in ev_behind_by_width.values()
        )]

    equilibrium_depth = None
    if (
        dp.live_total == 2
        and state.hero.position == "BB"
        and shover.position in ("SB", "BTN")
        and state.voluntary_actors == (shover.label,)
    ):
        equilibrium_depth = _equilibrium_depth(
            min(state.hero.stack_after_ante, shover.stack_after_ante) / bb
        )

    by_width = [_best_of("call", value) for value in ev_by_width.values()]
    best_tight, best_wide = by_width[0], by_width[-1]
    best = _best_of("call", ev_model)
    # Две оси разводятся в `detail` намеренно: они отвечают на разные вопросы —
    # «зависит ли вывод от угаданного диапазона» и «зависит ли он от того, войдут
    # ли живые позади». Слепить их в один флаг значило бы лишить изложение
    # возможности назвать причину.
    bracket_call = "stable" if {*by_width, best} == {best} else "unstable"
    # «Ось сдвинула вердикт» — значит расчёт с вошедшими в банк игроками позади
    # даёт не то, что выдала модель. Сравнивать надо именно с вердиктом модели, а
    # не с объединением концов вилки: иначе ось молчала бы всякий раз, когда
    # диапазонная вилка и так неустойчива, то есть ровно там, где смотреть на неё
    # интереснее всего.
    behind_moved = bool(set(best_behind) - {best})
    behind_axis = None if not best_behind else ("unstable" if behind_moved else "stable")
    zone, why = zone_for(
        best_tight,
        best_wide,
        live_total=dp.live_total,
        equilibrium=equilibrium_depth is not None,
        best_model=best,
        best_interior=by_width[1:-1],
        best_behind=best_behind,
        unmodelled=(
            f"живых за героем {len(behind)} — больше, чем модель вскрытия способна "
            f"перебрать, их влияние на вердикт не проверено"
            if behind_unmodelled
            else ""
        ),
    )
    taken = "call" if dp.action.kind is ActionKind.CALL else "fold"

    detail: dict[str, object] = {
        "method": "call_ev",
        "bracket": bracket_call,
        "ev_call_bb": round(ev_model, 4),
        "ev_call_tight_bb": round(ev_tight, 4),
        "ev_call_wide_bb": round(ev_wide, 4),
        "ev_call_by_width_bb": {str(w): round(v, 4) for w, v in ev_by_width.items()},
        "hero_class": hero_cls,
        "required_equity": round(required_equity(state.to_call, state.pot_before), 6),
        "shover_depth_bb": round(_depth_key(shover_depth_bb), 2),
        "dead_extra_bb": round(dead_bb, 4),
        # Считаются все живые, кроме героя и шовера: уже походивший опенер тоже
        # может ответить на шов, поэтому «ещё не действовавших» было бы занижением.
        "live_others": len(behind),
        "behind_axis": behind_axis,
        "ev_call_all_behind_bb": None if ev_behind is None else round(ev_behind, 4),
        "zone_reason": why,
    }
    return PointVerdict(
        dp_index=dp.index,
        street=dp.street,
        spot=spot,
        zone=zone,
        action_taken=taken,
        best_action=best,
        # Цена берётся по самому мягкому упрёку среди сценариев второй оси.
        # Обвинять игрока в потере 0.72bb за пас, когда собственный расчёт при
        # входе живого позади даёт -2.00bb, нельзя: это число ведёт и
        # ранжирование, и сумму потерь руки. Ярлык `assuming` предупреждает о
        # природе вывода, но не отменяет самого числа, а цифру игрок читает
        # первой. Сетка ширин диапазона в цену НЕ входит: там завышение
        # объявлено известным свойством модели и покрыто ярлыком зоны.
        ev_diff_bb=round(
            max(
                _taken_ev(taken, "call", value) - max(value, 0.0)
                for value in ([ev_model] if ev_behind is None else [ev_model, ev_behind])
            ),
            6,
        ),
        assumption=(
            Assumption(
                range=model_range,
                source="model:nash_hu_push",
                note=_ASSUMPTION_SHOVER
                + (
                    f"; вдобавок вердикт двигают {len(behind)} живых позади — "
                    f"если они тоже войдут в банк, лучше сыграть иначе"
                    if behind_moved
                    else ""
                ),
            )
            if zone is Zone.ASSUMING
            else None
        ),
        tools=["nash_hu", "call_shove_ev_bb", "required_equity"],
        detail=detail,
    )


def _extra_from_behind(state: TableState, behind: Sequence[SeatSnapshot]) -> int:
    """Сколько фишек добавят в банк героя живые позади, если тоже заколлируют.

    Каждый доколлирует до уровня ставки, но не больше своего стека; из этого
    герой способен выиграть лишь то, что покрыто его собственным вкладом, —
    потолок тот же, по которому движок считает `pot_before`.
    """
    ceiling = state.hero.stack
    level = max(seat.street_committed for seat in state.seats)
    total = 0
    for seat in behind:
        final = min(seat.stack, seat.ante + level)
        total += max(min(final, ceiling) - min(seat.contributed, ceiling), 0)
    return total


def _equity_with_behind(behind_ranges: Sequence[Range]) -> Callable[..., float]:
    """Эквити героя на вскрытии, куда добавлены диапазоны живых позади.

    Подменяет только эквити: сама арифметика банка остаётся в `call_shove_ev_bb`,
    иначе формулу денег пришлось бы написать второй раз — ровно тот дубль, на
    котором этот проект уже ловил расхождение.
    """

    def fn(hero: _HeroCombo, ranges: Sequence[Range]) -> float:
        return _model_equity(hero, [*ranges, *behind_ranges])

    return fn


def _best_of(active: str, ev: float) -> str:
    """Лучшее из двух действий: активное при плюсовой EV, иначе фолд (фолд = 0)."""
    return active if ev > 0.0 else "fold"


def _taken_ev(taken: str, active: str, ev: float) -> float:
    return ev if taken == active else 0.0


# Порог дешёвого префильтра (задача 13, SCALING.md §3): доля веса шова в
# равновесном чарте на САМОЙ КОРОТКОЙ глубине среди живых позади героя — той,
# что даёт равновесию САМЫЙ ШИРОКИЙ шов (короче эффективный стек — шире шов,
# см. замер в комментарии к `_DEPTH_STEP_BB`: 72o/32o/83o на 10.25bb с тремя
# позади дают ПОЛОЖИТЕЛЬНУЮ EV шова именно за счёт фолд-эквити более узкого
# поля). Если даже на этой, самой щедрой к шову глубине чарт не даёт классу
# героя веса больше десятой доли, дальше можно не считать: полный перебор
# подмножеств с сеткой ширин колл-диапазона (`_unopened_verdict`) способен
# только СУЗИТЬ обоснование шова относительно этой оценки — больше живых
# позади означает больше шансов быть отвеченным, а не меньше, — и развернуть
# вердикт в сторону шова он не может. Порог найден не подбором: 0.1 — это то
# же число, которым отсекается сама сводка скана (задача 13, `< -0.1bb`), и
# смысл тот же: цена ошибки ниже него всё равно не попала бы в список.
_PREFILTER_PUSH_WEIGHT_MAX = 0.1


def cheap_fold_verdict(dp: DecisionPoint, en: EnrichedHand) -> PointVerdict | None:
    """Дешёвый вердикт «фолд верен» для тривиального неоткрытого фолда — рычаг скана.

    Не альтернатива `verdict_for`, а обгон перед ним. Большинство рук турнирного
    файла — «сфолдил в неоткрытый банк, отдал блайнды» (SCALING.md §3), и для
    них дорогая часть расчёта (`shove_ev_bb`: перебор 2^n подмножеств коллеров
    на КАЖДОЙ из точек сетки ширин, с Монте-Карло на мультивее) ничего не меняет
    в выводе — чарт уже уверенно говорит «фолд». Эта функция закрывает именно
    такую точку одним попаданием в закешированный равновесный push-чарт
    (`_push_model`, тот же самый, которым уже пользуется `_unopened_verdict`),
    без единого вызова эквити-инструментов.

    Возвращает `None`, если условия не выполняются или чарт не даёт уверенного
    «фолд» (весом >= 0.9) — тогда точку обязан посчитать `verdict_for` целиком:
    отказ здесь не констатирует ошибку, а лишь снимает с себя право её судить
    дёшево. Это гарантирует отсутствие ложных «расхождения нет» на настоящих
    промахах: см. `_PREFILTER_PUSH_WEIGHT_MAX`.

    Зона проставляется тем же правилом, что и в `zone_for` для хедз-апа: точный
    расчёт (`strict`, без допущения) — только для SB против BB один на один,
    где колл-диапазон не угадан, а взят из равновесия. На мультивее полного
    перебора здесь нет, а значит нет и bracket-теста, который мог бы подтвердить
    устойчивость к ширине диапазона колла — зона обязана быть `assuming` с
    показанным допущением (контракт `PointVerdict._assumption_matches_zone`),
    как и для любого другого мультивей-вывода в этом модуле.
    """
    if dp.street is not Street.PREFLOP or dp.action.kind is not ActionKind.FOLD:
        return None
    state = table_state(dp, en)
    if spot_for(dp, state) is not SpotKind.PUSHFOLD_UNOPENED:
        return None

    hero_cls = _hero_class(en.hand)
    if hero_cls is None:
        return None
    behind_all = state.behind_hero
    behind = [s for s in behind_all if s.behind > 0]
    if len(behind) != len(behind_all):
        # Живой олл-ин позади (короткий блайнд ушёл в олл-ин постом) — вне
        # модели шова, как и в `_unopened_verdict` (там это `all_in_behind`,
        # принудительно ведущее к `assuming`). Дешёвый лукап такую точку не
        # обязан разбирать — она уходит на полный расчёт, который знает, что
        # с ней делать.
        return None
    if not behind or state.hero.behind <= 0:
        return None
    if len(behind) > _MAX_MODELLED_CALLERS:
        # Перебор подмножеств в `_unopened_verdict` на таком числе коллеров сам
        # не считается (2^n веток) и возвращает точку БЕЗ вердикта — дешёвый
        # лукап не имеет права быть увереннее полного расчёта там, где тот
        # прямо отказывается судить.
        return None

    bb = en.hand.bb
    hero_eff = state.hero.stack_after_ante
    depths = [min(hero_eff, seat.stack_after_ante) / bb for seat in behind]
    dead_bb = _table_dead_bb(state)
    shortest_depth = min(depths)
    push_range = _push_model(shortest_depth, dead_bb)
    push_weight = push_range.weight(hero_cls)
    if push_weight > _PREFILTER_PUSH_WEIGHT_MAX:
        return None

    equilibrium_depth = None
    if dp.live_total == 2 and len(behind) == 1 and behind[0].position == "BB":
        equilibrium_depth = _equilibrium_depth(min(hero_eff, behind[0].stack_after_ante) / bb)

    if equilibrium_depth is not None:
        zone = Zone.STRICT
        why = "хедз-ап SB против BB: колл-диапазон взят из равновесия пуш-фолда"
        assumption = None
    else:
        zone = Zone.ASSUMING
        why = (
            f"дешёвый префильтр: вес шова в равновесном чарте {push_weight:.3f} <= "
            f"{_PREFILTER_PUSH_WEIGHT_MAX} на самой короткой глубине среди живых "
            f"позади — полная сетка ширин колл-диапазона не проверялась"
        )
        assumption = Assumption(
            range=push_range,
            source="model:nash_hu_push_prefilter",
            note=(
                "дешёвый префильтр смотрит равновесный push-чарт на самой короткой "
                "глубине среди живых позади вместо полного перебора подмножеств "
                "коллеров и сетки ширин их диапазона"
            ),
        )

    detail: dict[str, object] = {
        "method": "prefilter_chart_lookup",
        "hero_class": hero_cls,
        "push_weight": round(push_weight, 6),
        "lookup_depth_bb": round(shortest_depth, 2),
        "dead_extra_bb": round(dead_bb, 4),
        "zone_reason": why,
    }
    return PointVerdict(
        dp_index=dp.index,
        street=dp.street,
        spot=SpotKind.PUSHFOLD_UNOPENED,
        zone=zone,
        action_taken="fold",
        best_action="fold",
        ev_diff_bb=0.0,
        assumption=assumption,
        tools=["nash_hu"],
        detail=detail,
    )


def verdict_for(dp: DecisionPoint, en: EnrichedHand) -> PointVerdict:
    """Вердикт по одной точке решения героя.

    Постфлоп и прочий префлоп в v1 не оцениваются и возвращаются без вердикта:
    инструмента, который посчитал бы их цену, у нас пока нет, а назвать
    неизвестную цену нулём и промолчать — значит выдать пробел за отсутствие
    ошибки.
    """
    if dp.street is not Street.PREFLOP:
        return _unjudged(dp, SpotKind.POSTFLOP, "постфлоп в v1 не оценивается")

    state = table_state(dp, en)
    spot = spot_for(dp, state)
    if spot is SpotKind.PUSHFOLD_UNOPENED:
        return _unopened_verdict(dp, en, state)
    if spot is SpotKind.PUSHFOLD_FACING_SHOVE:
        return _facing_shove_verdict(dp, en, state)
    return _unjudged(dp, spot, unpriced_reason(dp, state))
