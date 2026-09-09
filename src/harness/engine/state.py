"""Состояние в точке решения: то, что движок умеет на неполном входе.

Скриншот живого стола — не рука, а срез (реестр E1): истории улиц нет,
результата нет, вскрытия нет. Реплей здесь неприменим **по построению**, а не
потому, что мы его не написали: PokerKit проигрывает последовательность
действий, а последовательности на экране не напечатано. Валидатор такой вход
битым не считает (`harness.engine.validation.validate`, ветка `Completeness.
STATE`) — иначе разбирать его было бы нечем ещё до этого модуля.

**Из продакшена этот путь сегодня недостижим.** `Completeness.STATE` ставится в
одном месте на весь проект — `harness.parsers.vision_adapter.reading_to_raw`,
когда на экране не прочитано ни результата, ни вскрытия, ни победителя, — а
такой экран станция чтения отвергает до того, как рука сохранится
(`test_a_hand_in_progress_stops_the_cascade_on_the_first_hop`,
`test_a_hand_still_in_progress_never_reaches_the_analysis`). Разбор
незавершённой руки — решение владельца 2026-09-09, и оно продуктовое, а не
техническое: код оставлен, его вход строит `tests/test_engine_state.py`.

Отсюда второй путь в ядро. Он строит ровно одну точку решения — героя, на той
улице, которую застал экран, — и НИЧЕГО не утверждает про то, чего на экране
нет. Всё, что на полной руке проверяют три независимых факта рума (`Total pot`,
строки `collected`, воспроизведённый порядок хода), здесь не проверяется вовсе;
чтобы это не выглядело как пройденная проверка, каждая такая позиция названа в
`Verdict.not_checked` (`STATE_NOT_CHECKED`).

**Чего этот путь НЕ делает, и почему это принципиально:**

* не сверяет деньги с румом — у скрина нет ни одной строки от рума (реестр D1);
* не выводит `stacks_end` — рука не кончилась, конца стеков не существует;
* не судит сыгранное действие — герой ещё не сходил, и `DecisionPoint.action`
  остаётся `None`; ядро на такой точке отказывается с названной причиной
  (`harness.analysis.preflop.verdict_for`);
* не восстанавливает вклады прошлых улиц по игрокам: на постфлопе видно только
  фишки текущего круга, а деньги прошлых улиц лежат в середине общей кучей.
  Эта куча входит в банк как мёртвые деньги и никому не приписывается.

**Что такое `stack` на состоянии.** У полной руки это стартовый стек. Здесь
экран показывает остаток, поэтому адаптер (`harness.parsers.vision_adapter`)
восстанавливает `stack` как «остаток + выставленные фишки + уплаченное анте», и
на префлопе это в точности стартовый стек (реестр, «Проверка схемы на трёх
экранах»: сверено с текстом рума до фишки). На постфлопе вклады прошлых улиц в
него не входят — см. последний пункт выше и `STATE_NOT_CHECKED`.
"""

from __future__ import annotations

from harness.contracts import (
    ActionKind,
    CanonicalHand,
    Completeness,
    DecisionPoint,
    EngineReport,
    PlayerState,
    Street,
)
from harness.normalizer import POSITIONS_BY_COUNT

__all__ = ["STATE_NOT_CHECKED", "StateNotReadable", "state_not_checked", "state_report"]

# Порядок улиц — от префлопа к риверу. Текущая улица состояния определяется по
# числу карт на борде: три карты это флоп, четыре тёрн, пять ривер.
_STREET_BY_BOARD_SIZE: dict[int, Street] = {
    0: Street.PREFLOP,
    3: Street.FLOP,
    4: Street.TURN,
    5: Street.RIVER,
}

# Проверки, для которых на состоянии нет данных. Список попадает в
# `Verdict.not_checked` и дальше в трейс: молчаливый `pass` на входе, где
# проверять нечем, неотличим от проверенного входа.
STATE_NOT_CHECKED: tuple[str, ...] = (
    "сверка банка с итогом рума",
    "сверка получателей банка",
    "сохранение фишек",
    "воспроизведение порядка хода",
)

# Дополняется на постфлопе: там неизвестно, кто сколько положил на прошлых улицах.
NOT_CHECKED_PAST_STREETS = "вклады прошлых улиц по игрокам"


class StateNotReadable(ValueError):
    """Состояние не разобрать: нет героя, нет борда нужного размера, нет мест.

    Отдельный тип, а не голый `ValueError`: воркер обязан отличать «экран прочитан,
    но состояние не складывается» (вопрос игроку) от произвольной поломки в коде.
    """


def current_street(hand: CanonicalHand) -> Street:
    """Улица, на которой застали стол, — по числу карт на борде.

    Борд состояния хранится одной записью: карты, лежащие на столе сейчас. Числа
    карт, кроме 0, 3, 4 и 5, в холдеме не бывает, и придумывать улицу для него
    нельзя.
    """
    cards = [card for street_cards in hand.boards.values() for card in street_cards]
    street = _STREET_BY_BOARD_SIZE.get(len(cards))
    if street is None:
        raise StateNotReadable(f"на борде {len(cards)} карт — такой улицы в холдеме нет")
    return street


def _folded(hand: CanonicalHand) -> set[str]:
    """Кто выбыл из руки. На состоянии это записанные адаптером пасы.

    Единственный видимый на живом столе признак «не в руке» — отсутствие карт
    перед игроком; адаптер переводит это наблюдение в действие `fold`, потому что
    пас и есть «не в руке», и никакой другой суммы или намерения оно не несёт.
    """
    return {a.label for a in hand.actions if a.kind is ActionKind.FOLD}


def _ante_paid(hand: CanonicalHand, player: PlayerState) -> int:
    return min(hand.ante, player.stack)


def _dead_middle(hand: CanonicalHand, contributed: dict[str, int]) -> int:
    """Деньги в середине, которые не приписаны ни одному игроку.

    Разница между показанным на экране банком и суммой видимых вкладов. На
    префлопе она равна нулю всякий раз, когда никто не сбросил карты, уже
    вложившись, — именно на этом равенстве держится контрольная сумма банка
    (реестр D2, ею было доказано пропущенное анте). На постфлопе это деньги
    прошлых улиц: они в банке есть, а кто их положил, экран не говорит.

    Отрицательной разница быть не может: сумма вкладов больше показанного банка
    означает неверное чтение, и это ловит контрольная сумма ДО ядра
    (`harness.parsers.vision_checks`), а не здесь.
    """
    shown = hand.vision.displayed_pot if hand.vision is not None else None
    if shown is None:
        return 0
    return max(0, shown - sum(contributed.values()))


def _seats_in_action_order(hand: CanonicalHand) -> list[str]:
    """Метки игроков в порядке позиций за столом — от малого блайнда к кнопке."""
    order = POSITIONS_BY_COUNT.get(len(hand.players))
    if order is None:
        raise StateNotReadable(f"стол на {len(hand.players)} мест: порядка позиций для такого нет")
    rank = {position: i for i, position in enumerate(order)}
    return [p.label for p in sorted(hand.players, key=lambda p: rank[p.position])]


def state_report(hand: CanonicalHand) -> EngineReport:
    """Отчёт движка по состоянию в точке решения — одна точка, точка героя.

    `final_pot` здесь — банк, СЛОЖИВШИЙСЯ К ЭТОМУ МОМЕНТУ, а не итог руки:
    итога не существует, пока рука не сыграна. Имя поля общее с полной рукой
    потому, что общий и потребитель; сверка `final_pot` с итогом рума на этом
    входе не выполняется и названа в `Verdict.not_checked`.

    `stacks_end` пуст: конца стеков у несыгранной руки нет, а поставить туда
    текущие остатки значило бы выдать срез за результат.
    """
    if hand.completeness is not Completeness.STATE:
        raise StateNotReadable(
            f"путь состояния вызван на входе полноты {hand.completeness.value!r}"
        )

    street = current_street(hand)
    hero = next((p for p in hand.players if p.label == hand.hero_label), None)
    if hero is None:
        raise StateNotReadable(f"героя {hand.hero_label!r} нет среди мест состояния")

    folded = _folded(hand)
    bets = {p.label: hand.visible_bets.get(p.label, 0) for p in hand.players}
    contributed = {p.label: _ante_paid(hand, p) + bets[p.label] for p in hand.players}
    behind = {p.label: p.stack - contributed[p.label] for p in hand.players}

    ceiling = contributed[hero.label] + behind[hero.label]
    pot_before = sum(min(amount, ceiling) for amount in contributed.values())
    pot_before += _dead_middle(hand, contributed)
    to_call = min(max(bets.values()) - bets[hero.label], behind[hero.label])

    live = [p for p in hand.players if p.label not in folded]
    # Глубина решения — как у реплея (`replay._effective_stack`, ветка «агрессора
    # нет»): потолок руки по самому глубокому живому оппоненту. Агрессора здесь
    # выделить не из чего — экран не говорит, кто из выставивших фишки ставил, а
    # кто отвечал, и назначить его значило бы прочитать намерение, а не экран.
    playable = {p.label: behind[p.label] + bets[p.label] for p in hand.players}
    rivals = [playable[p.label] for p in live if p.label != hero.label]
    eff_stack = min(playable[hero.label], max(rivals)) if rivals else playable[hero.label]

    order = _seats_in_action_order(hand)
    hero_index = order.index(hero.label)
    after_hero = order[hero_index + 1 :] + order[:hero_index]
    forced = {p.label: _forced_bet(hand, p) for p in hand.players}
    live_behind = sum(
        1 for label in after_hero if label not in folded and bets[label] <= forced[label]
    )

    point = DecisionPoint(
        index=0,
        street=street,
        label=hero.label,
        position=hero.position,
        to_call=to_call,
        pot_before=pot_before,
        eff_stack=eff_stack,
        eff_stack_bb=eff_stack / hand.bb,
        spr=(eff_stack / pot_before if street is not Street.PREFLOP and pot_before else None),
        action=None,
        live_total=len(live),
        live_behind=live_behind,
    )
    return EngineReport(
        pot_by_street={street: pot_before},
        final_pot=pot_before,
        stacks_end={},
        decision_points=[point],
    )


def _forced_bet(hand: CanonicalHand, player: PlayerState) -> int:
    """Вынужденная ставка позиции — блайнд, урезанный остатком после анте.

    Тот же смысл, что у `harness.engine.validation.forced_blind`, но берётся
    отсюда: там формула считает по стартовому стеку полной руки, а здесь она
    нужна лишь как порог «игрок ещё не ходил» — фишки перед ним не больше
    вынужденных.
    """
    heads_up = len(hand.players) == 2
    if player.position == "BB":
        return hand.bb
    if player.position == "SB" or (heads_up and player.position == "BTN"):
        return hand.sb
    return 0


def state_not_checked(hand: CanonicalHand) -> list[str]:
    """Что на этом состоянии проверить не из чего — поимённо, для `Verdict`."""
    named = list(STATE_NOT_CHECKED)
    if current_street(hand) is not Street.PREFLOP:
        named.append(NOT_CHECKED_PAST_STREETS)
    return named
