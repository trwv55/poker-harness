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
зависит. Последнее и проверяет bracket-тест: EV пересчитывается против заведомо
узкой (`BRACKET_TIGHT`) и заведомо широкой (`BRACKET_WIDE`) модели, и зона
`strict` ставится, только когда оба конца вилки И сама модель дают ОДИН вердикт
(почему требуется третье совпадение — в докстринге `zone_for`). Ошибка, которую
правило предотвращает, ровно одна и односторонняя: объявить устойчивым вывод,
который на деле держится на догадке, значит переоценить собственную уверенность.

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

from collections.abc import Sequence

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
    BRACKET_WIDE,
    MAX_DEAD_EXTRA_BB,
    CallerModel,
    call_shove_ev_bb,
    equity_vs_range_classes,
    fold_equity_ok,
    nash_hu,
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
) -> tuple[Zone, str]:
    """Зона доверия точки и причина, по которой она такая (спека §5.5).

    `live_total == 2` при `equilibrium` — настоящий хедз-ап либо SB против BB
    после пасов: колл-диапазон берётся из равновесия, угадывать нечего.
    Флаг `equilibrium` нужен потому, что двое живых сами по себе равновесия не
    дают: шов из ранней позиции, до которого спасовали все, кроме большого
    блайнда, оставляет за столом двоих, но это уже не та игра, которую решает
    `nash_hu` — там мёртвый малый блайнд и другой шовер. Такая точка судится
    вилкой, как мультивей.

    При трёх и более живых зона `strict` выдаётся только тогда, когда вердикт от
    модели диапазонов не зависит: **все три** расчёта — узкий конец вилки, широкий
    конец и сама модель — дают один ответ.

    Требование про `best_model` появилось не из соображений строгости, а по факту
    с реальных рук. Вилка `BRACKET_TIGHT`/`BRACKET_WIDE` собрана как модель
    КОЛЛ-диапазона (top-40% на широком конце), а против шова моделью служит
    пуш-сторона равновесия — на 13.5bb это около 50% комбо, то есть ШИРЕ широкого
    конца. Вилка перестаёт накрывать модель, и её концы могут совпасть между собой,
    противореча при этом самому выданному вердикту: на фикстуре встретились точки,
    где модель говорила «колл», оба конца — «фолд», а зона выходила `strict`.
    Совпадение концов вилки доказывает устойчивость только того вердикта, который
    они и дают; вердикт, которого не подтверждает ни один из них, держится
    исключительно на модели и обязан быть помечен `assuming`.
    """
    if live_total == 2 and equilibrium:
        return Zone.STRICT, "хедз-ап SB против BB: колл-диапазон взят из равновесия пуш-фолда"
    verdicts = {best_tight, best_wide} | ({best_model} if best_model is not None else set())
    if len(verdicts) == 1:
        stable = (
            f"вердикт «{best_tight}» не меняется ни против узкой, ни против широкой "
            f"модели диапазона — допущение на него не влияет"
        )
        return Zone.STRICT, stable
    if best_model is not None and best_model not in (best_tight, best_wide):
        unsupported = (
            f"вердикт «{best_model}» держится на модели диапазона: вилка его не "
            f"подтверждает — против узкой лучше «{best_tight}», против широкой "
            f"«{best_wide}»"
        )
        return Zone.ASSUMING, unsupported
    unstable = (
        f"вердикт зависит от модели диапазона: против узкой лучше «{best_tight}», "
        f"против широкой — «{best_wide}»"
    )
    return Zone.ASSUMING, unstable


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
    ev_tight = ev([BRACKET_TIGHT(depth) for depth in depths])
    ev_wide = ev([BRACKET_WIDE(depth) for depth in depths])

    equilibrium_depth = None
    if dp.live_total == 2 and len(behind) == 1 and behind[0].position == "BB":
        equilibrium_depth = _equilibrium_depth(
            min(hero_eff, behind[0].stack_after_ante) / bb
        )

    best_tight, best_wide = _best_of("shove", ev_tight), _best_of("shove", ev_wide)
    best = _best_of("shove", ev_model)
    bracket_shove = "stable" if {best_tight, best_wide, best} == {best} else "unstable"
    zone, why = zone_for(
        best_tight,
        best_wide,
        live_total=dp.live_total,
        equilibrium=equilibrium_depth is not None,
        best_model=best,
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
    ev_tight = ev(BRACKET_TIGHT(shover_depth_bb))
    ev_wide = ev(BRACKET_WIDE(shover_depth_bb))

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

    best_tight, best_wide = _best_of("call", ev_tight), _best_of("call", ev_wide)
    best = _best_of("call", ev_model)
    bracket_call = "stable" if {best_tight, best_wide, best} == {best} else "unstable"
    zone, why = zone_for(
        best_tight,
        best_wide,
        live_total=dp.live_total,
        equilibrium=equilibrium_depth is not None,
        best_model=best,
    )
    taken = "call" if dp.action.kind is ActionKind.CALL else "fold"

    # Живые игроки, которые ещё могут ответить на шов после героя, в модель не
    # входят: `call_shove_ev_bb` считает вскрытие один на один. Если кто-то из
    # них тоже заколлирует, эквити героя окажется ниже посчитанной — смещение
    # направлено в сторону колла. Правильно моделировать их — отдельная работа;
    # честный ответ до неё — перестать заявлять точность, как бы ни повела себя
    # вилка. Недо-заявить безопасно, пере-заявить — тот единственный отказ,
    # ради которого правило зоны и существует.
    live_others = sum(
        1 for s in state.seats if s.live and s.label not in (state.hero.label, shover.label)
    )
    if live_others:
        zone = Zone.ASSUMING
        why = (
            f"за героем ещё {live_others} живых: модель считает вскрытие один на один, "
            f"их возможный колл в неё не входит — эквити героя завышено"
        )

    detail: dict[str, object] = {
        "method": "call_ev",
        "bracket": bracket_call,
        "ev_call_bb": round(ev_model, 4),
        "ev_call_tight_bb": round(ev_tight, 4),
        "ev_call_wide_bb": round(ev_wide, 4),
        "hero_class": hero_cls,
        "required_equity": round(required_equity(state.to_call, state.pot_before), 6),
        "shover_depth_bb": round(_depth_key(shover_depth_bb), 2),
        "dead_extra_bb": round(dead_bb, 4),
        # Считаются все живые, кроме героя и шовера: уже походивший опенер тоже
        # может ответить на шов, поэтому «ещё не действовавших» было бы занижением.
        "live_others": live_others,
        "zone_reason": why,
    }
    return PointVerdict(
        dp_index=dp.index,
        street=dp.street,
        spot=spot,
        zone=zone,
        action_taken=taken,
        best_action=best,
        ev_diff_bb=round(_taken_ev(taken, "call", ev_model) - max(ev_model, 0.0), 6),
        assumption=(
            Assumption(
                range=model_range,
                source="model:nash_hu_push",
                note=_ASSUMPTION_SHOVER
                + (
                    f"; кроме того, живых за героем {live_others} — их возможный колл "
                    f"модель вскрытия не учитывает"
                    if live_others
                    else ""
                ),
            )
            if zone is Zone.ASSUMING
            else None
        ),
        tools=["nash_hu", "call_shove_ev_bb", "required_equity"],
        detail=detail,
    )


def _best_of(active: str, ev: float) -> str:
    """Лучшее из двух действий: активное при плюсовой EV, иначе фолд (фолд = 0)."""
    return active if ev > 0.0 else "fold"


def _taken_ev(taken: str, active: str, ev: float) -> float:
    return ev if taken == active else 0.0


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
