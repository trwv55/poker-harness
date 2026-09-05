"""Классификатор спота и восстановление стола в точке решения.

Две вещи, которые нужны анализу и которых нет в `DecisionPoint`:

1. **Кто сколько уже вложил.** Формулы пуш-фолда (`shove_ev_bb`,
   `call_shove_ev_bb`) считают банк по ПОЛНЫМ вкладам игроков, а не по остаткам
   за спиной, и без постов систематически врут (см. докстринг
   `harness.analysis.tools.pushfold`). Движок постов по игрокам не отдаёт —
   `TableState` восстанавливает их из канонической руки.
2. **Форма спота.** Открыт ли банк добровольно, кто агрессор, доводит ли колл
   героя до олл-ина — из этого и складывается `SpotKind`.

Восстановление сверяется с движком на каждой точке: `pot_before` и `to_call`
обязаны совпасть с тем, что посчитал реплей. Расхождение — `ValueError`, а не
тихая поправка: подогнать банк значило бы соврать про деньги игрока.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.contracts import (
    ActionKind,
    CanonicalHand,
    DecisionPoint,
    EnrichedHand,
    SpotKind,
    Street,
)
from harness.engine.validation import forced_blind

# Порог пуш-фолд парадигмы (спека §5.5): глубже решение перестаёт сводиться к
# «шов или фолд», и модель к нему неприменима.
PUSHFOLD_MAX_EFF_BB = 15.0


@dataclass(frozen=True)
class SeatSnapshot:
    """Игрок на момент решения героя."""

    label: str
    position: str
    stack: int  # стартовый стек — потолок всего, что игрок способен вложить в руку
    ante: int  # уплаченное анте — мёртвые деньги, входящие в решаемую игру
    contributed: int  # вложено в руку к этому моменту: анте + блайнд + добровольное
    street_committed: int  # из этого — поставлено на текущей улице
    live: bool  # ещё в руке
    acted: bool  # уже действовал в этом круге до героя

    @property
    def behind(self) -> int:
        """Остаток стека за спиной."""
        return self.stack - self.contributed

    @property
    def stack_after_ante(self) -> int:
        """Стек за вычетом анте — глубина игры, которую решает `nash_hu`.

        Равновесие берёт анте отдельным слагаемым мёртвых денег, поэтому глубину
        ему надо давать уже без него: иначе анте посчитается дважды.
        """
        return self.stack - self.ante


@dataclass(frozen=True)
class TableState:
    """Стол в точке решения героя: деньги, живые игроки, форма спота."""

    bb: int
    hero: SeatSnapshot
    seats: tuple[SeatSnapshot, ...]
    pot_before: int  # Σ вкладов, урезанных потолком героя — как считает движок
    to_call: int
    opened_voluntarily: bool  # до героя кто-то вложился по своей воле
    voluntary_actors: tuple[str, ...]  # кто именно — в порядке хода
    aggressor: SeatSnapshot | None  # последний, кто ставил или повышал до героя
    hero_all_in_after: bool  # после своего действия герой остался без фишек

    @property
    def behind_hero(self) -> tuple[SeatSnapshot, ...]:
        """Живые игроки, которые ещё не действовали в этом круге."""
        return tuple(
            s for s in self.seats if s.live and not s.acted and s.label != self.hero.label
        )

    @property
    def call_is_all_in(self) -> bool:
        """Доплата съедает весь остаток героя — колл здесь и есть олл-ин."""
        return self.to_call > 0 and self.to_call >= self.hero.behind

    @property
    def aggressor_all_in(self) -> bool:
        return self.aggressor is not None and self.aggressor.behind == 0

    @property
    def all_in_before_hero(self) -> tuple[SeatSnapshot, ...]:
        """Живые игроки, уже ушедшие в олл-ин до решения героя."""
        return tuple(
            s
            for s in self.seats
            if s.live and s.acted and s.behind == 0 and s.label != self.hero.label
        )

    @property
    def opened_by_aggressor(self) -> bool:
        """Ставку перед героем поставил тот же, кто открыл банк.

        Иначе перед героем ре-шов поверх чужого опена, а его диапазон — не
        диапазон открытого шова, которым моделирует `nash_hu`.
        """
        if self.aggressor is None:
            return True  # банк открыт лимпами, повышений не было
        return bool(self.voluntary_actors) and self.aggressor.label == self.voluntary_actors[0]


def _action_index(hand: CanonicalHand, dp: DecisionPoint) -> int:
    """Позиция действия точки решения в списке действий руки.

    `DecisionPoint.index` нумерует точки решения героя, а не действия руки, и
    совпадает с порядковым номером действия героя — кроме рук, где рум записал
    герою пас при нулевом стеке (движок исполняет такой пас без точки решения).
    Поэтому номер проверяется сверкой самого действия, а не принимается на веру.
    """
    hero_actions = [i for i, action in enumerate(hand.actions) if action.label == hand.hero_label]
    for i in hero_actions[dp.index :]:
        if hand.actions[i] == dp.action:
            return i
    raise ValueError(
        f"действие точки решения {dp.index} не найдено среди действий героя: {dp.action.raw_line}"
    )


def table_state(dp: DecisionPoint, en: EnrichedHand) -> TableState:
    """Восстановить стол в префлоп-точке решения героя.

    Только префлоп: восстанавливать вклады постфлопа незачем, пока постфлоп не
    оценивается (ступень 2 порядка разработки), а частичная реализация тихо
    разошлась бы с движком.
    """
    if dp.street is not Street.PREFLOP:
        raise ValueError(
            f"восстановление стола реализовано только для префлопа, получено {dp.street}"
        )

    hand = en.hand
    ante = {p.label: min(hand.ante, p.stack) for p in hand.players}
    committed = {p.label: forced_blind(hand, p, ante[p.label]) for p in hand.players}
    live = dict.fromkeys(committed, True)
    acted = dict.fromkeys(committed, False)

    target = _action_index(hand, dp)
    voluntary: list[str] = []
    aggressor: str | None = None
    for action in hand.actions[:target]:
        acted[action.label] = True
        if action.kind is ActionKind.FOLD:
            live[action.label] = False
            continue
        if action.kind is not ActionKind.CHECK and action.label not in voluntary:
            voluntary.append(action.label)
        if action.kind in (ActionKind.BET, ActionKind.RAISE):
            aggressor = action.label
        committed[action.label] = action.committed_after

    hero = next(p for p in hand.players if p.label == hand.hero_label)
    ceiling = hero.stack  # больше стартового стека герой в банк не вложит
    seats = tuple(
        SeatSnapshot(
            label=p.label,
            position=p.position,
            stack=p.stack,
            ante=ante[p.label],
            contributed=ante[p.label] + committed[p.label],
            street_committed=committed[p.label],
            live=live[p.label],
            acted=acted[p.label],
        )
        for p in hand.players
    )
    by_label = {s.label: s for s in seats}
    hero_seat = by_label[hand.hero_label]

    pot_before = sum(min(s.contributed, ceiling) for s in seats)
    to_call = min(
        max(s.street_committed for s in seats) - hero_seat.street_committed, hero_seat.behind
    )
    hero_after = hero_seat.behind - (dp.action.committed_after - hero_seat.street_committed)

    state = TableState(
        bb=hand.bb,
        hero=hero_seat,
        seats=seats,
        pot_before=pot_before,
        to_call=to_call,
        opened_voluntarily=bool(voluntary),
        voluntary_actors=tuple(voluntary),
        aggressor=by_label[aggressor] if aggressor is not None else None,
        hero_all_in_after=hero_after <= 0,
    )
    _cross_check(state, dp)
    return state


def _cross_check(state: TableState, dp: DecisionPoint) -> None:
    """Сверить восстановленные деньги с посчитанными движком.

    Обе стороны считают одно и то же двумя независимыми путями: движок — реплеем
    в PokerKit, анализ — по строкам канонической руки. Расхождение означает, что
    одна из них врёт, и цена решения будет посчитана не от того банка.
    """
    if state.pot_before != dp.pot_before or state.to_call != dp.to_call:
        raise ValueError(
            f"восстановленный стол разошёлся с движком в точке {dp.index}: "
            f"банк {state.pot_before} против {dp.pot_before}, "
            f"доплата {state.to_call} против {dp.to_call}"
        )


def action_name(dp: DecisionPoint) -> str:
    """Человекочитаемое имя сыгранного действия — то, что показывается игроку."""
    kind = dp.action.kind
    if kind in (ActionKind.BET, ActionKind.RAISE) and dp.action.is_all_in:
        return "shove"
    return str(kind)


def unpriced_reason(dp: DecisionPoint, state: TableState) -> str:
    """Почему префлоп-точка осталась без вердикта — по-человечески, а не «прочее».

    Причина попадает в `detail` и дальше в сводку: игрок должен видеть не только
    то, что спот не разобран, но и чем именно он не подошёл. Без этого пробел в
    охвате выглядит как подтверждение правильной игры.
    """
    if dp.eff_stack_bb > PUSHFOLD_MAX_EFF_BB:
        return (
            f"глубже пуш-фолд-зоны: эффективный стек {dp.eff_stack_bb:.1f}bb "
            f"> {PUSHFOLD_MAX_EFF_BB:.0f}bb"
        )
    if state.hero.acted:
        return "герой уже вложился на этой улице: это война повышений, а не пуш-фолд"
    if not state.opened_voluntarily:
        return "неоткрытый банк, но сыгран не шов и не пас — лимп и мин-рейз модель не считает"
    if not (state.call_is_all_in or state.aggressor_all_in):
        return "банк открыт рейзом не в олл-ин, и колл героя олл-ином не был"
    if not state.opened_by_aggressor:
        return "перед героем ре-шов поверх чужого опена: его диапазон уже открытого шова"
    if len(state.all_in_before_hero) > 1:
        return (
            f"перед героем {len(state.all_in_before_hero)} олл-ина: сайд-поты и вскрытие "
            f"против нескольких диапазонов сразу"
        )
    return "перед героем олл-ин, но сыгран не колл и не пас"


def classify(dp: DecisionPoint, en: EnrichedHand) -> SpotKind:
    """Вид спота в точке решения героя.

    Класс отвечает на вопрос «какой моделью эта точка оценивается», а не «как она
    выглядит на столе». Поэтому лимп или мин-рейз в пуш-фолд-зоне — это
    `preflop_other`: моделью «шов или фолд» такое решение не оценивается, и
    выдавать его цену за посчитанную было бы враньём. Спот при этом не теряется —
    он попадает в результат без вердикта.
    """
    if dp.street is not Street.PREFLOP:
        return SpotKind.POSTFLOP
    return spot_for(dp, table_state(dp, en))


def spot_for(dp: DecisionPoint, state: TableState) -> SpotKind:
    """То же, что `classify`, но по уже восстановленному столу.

    Отдельная функция, чтобы анализ не переигрывал восстановление дважды: сначала
    ради класса спота, потом ради его оценки.
    """
    if dp.eff_stack_bb > PUSHFOLD_MAX_EFF_BB:
        return SpotKind.PREFLOP_OTHER
    # Герой, уже вложившийся на этой улице по своей воле, стоит не перед выбором
    # «шов или фолд», а внутри войны повышений: перед ним ре-шов, диапазон
    # которого с диапазоном открытого шова не имеет ничего общего. Замечено на
    # реальной руке — открытый рейз, колл героя, шов блайнда, ре-шов открывшего:
    # модель насчитывала герою потерю 11bb на пасе, приняв ре-шов за открытый шов.
    if state.hero.acted:
        return SpotKind.PREFLOP_OTHER

    folded = dp.action.kind is ActionKind.FOLD

    if state.opened_voluntarily:
        # Колл на весь остаток — тот же олл-ин, чем бы ни была ставка перед
        # героем. Так выглядит короткий блайнд против обычного опен-рейза
        # (доплата упирается в стек), и модель «колл или фолд против диапазона»
        # применима к нему полностью. Банк, открытый рейзом не в олл-ин, при
        # котором у героя ещё остаются фишки, — уже не пуш-фолд: там есть
        # третье действие (ре-шов), которого модель не считает.
        faces_shove = state.call_is_all_in or state.aggressor_all_in
        answered = folded or (dp.action.kind is ActionKind.CALL and state.hero_all_in_after)
        # Две границы применимости модели, обе найдены прогоном по реальным рукам.
        # `call_shove_ev_bb` меряет эквити против ОДНОГО диапазона, а пуш-сторона
        # `nash_hu` — это диапазон игрока, который шовит ПЕРВЫМ. Ре-шов поверх
        # чужого опена вчетверо уже открытого шова, а два уже вложившихся олл-ина
        # означают сайд-поты и вскрытие на троих. Оба нарушения завышают эквити
        # героя, то есть толкают вердикт в сторону колла.
        applicable = state.opened_by_aggressor and len(state.all_in_before_hero) <= 1
        if faces_shove and answered and applicable:
            return SpotKind.PUSHFOLD_FACING_SHOVE
        return SpotKind.PREFLOP_OTHER

    if folded or state.hero_all_in_after:
        return SpotKind.PUSHFOLD_UNOPENED
    return SpotKind.PREFLOP_OTHER
