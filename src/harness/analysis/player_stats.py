"""Статистика места по списку рук — чистые функции над действиями (задача 23).

Ни одного числа отсюда не приходит от модели: каждая величина — счётчик
раздач или ситуаций, а формула счётчика записана в докстринге его функции и
пришпилена тестом на синтетической руке (`tests/test_player_stats.py`).

**Считается для ЛЮБОГО места, не только героя.** `player_stats(hands, label)`
берёт одно место, `player_stats_by_label(hands)` — все места разом за тот же
проход по рукам. В GG-HH оппоненты обезличены, но метка места сквозная внутри
турнира, и статистика оппонента — такой же счётчик его действий, как и
статистика героя. Формулы ниже метку принимают аргументом; `hero_label`
остаётся только умолчанием `player_stats`.

**Почему счётчики, а не проценты.** Наружу отдаётся `PlayerStats` со
числителями и знаменателями (доли — свойства контракта): среднее игрока по
нескольким турнирам считается сложением счётчиков, а не усреднением процентов,
иначе турнир из 12 раздач весил бы столько же, сколько турнир из 300. Поэтому
и функции берут ЛЮБОЙ список рук — одного турнира или всех сразу.

**Порога на малую выборку здесь нет.** Доля отсутствует ровно при нулевом
знаменателе (`PlayerStats._share`) — то есть когда ситуация не встретилась ни
разу; при любом ненулевом знаменателе возвращается и доля, и сам знаменатель
(`test_a_single_observation_is_returned_with_its_denominator`). Скрыть долю от
трёх наблюдений значило бы скрыть вместе с ней и число 3, а именно оно и
говорит читателю, чего эта доля стоит.

**Вход — `CanonicalHand`, не `EnrichedHand`.** Всё, что нужно этим формулам, —
действия, разложенные по улицам, доска и состав мест; отчёт движка не нужен ни
одной из них. Это не экономия строки импорта: среднее по всем турнирам игрока
читает из базы только колонку `hands.canonical` (`memory.repos`), а `enriched`
там весит кратно больше и не пригодился бы.

**Блайнды и анте — не действия.** Нормализатор кладёт посты в
`CanonicalHand.posts` отдельно от `actions` (`normalizer/normalize.py`),
поэтому вынужденная ставка не может попасть в VPIP по построению формулы:
считаются только действия места. Чек большого блайнда — `ActionKind.CHECK`, и
в VPIP он тоже не входит (`test_a_posted_blind_is_not_a_voluntary_investment`).

**Ни одной оценки.** Здесь только частоты и счётчики: эталона, относительно
которого «много» или «мало» имело бы смысл, в проекте нет, и слово о норме
было бы выдумано.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from harness.contracts import (
    ActionKind,
    CanonicalAction,
    CanonicalHand,
    FrequencyStat,
    Measurement,
    PlayerStats,
    Street,
)

__all__ = [
    "measurement_of",
    "player_stats",
    "player_stats_across_tournaments",
    "player_stats_by_label",
    "player_stats_of_seats",
    "seat_position",
]

# Действия, которыми игрок добровольно кладёт фишки в банк. Пас и чек не кладут
# ничего, посты сюда не попадают вовсе (см. докстринг модуля).
_VOLUNTARY = frozenset({ActionKind.CALL, ActionKind.BET, ActionKind.RAISE})
# Действия, повышающие ставку. `BET` — префлоп-повышение в неоткрытом банке
# (источник может записать шов и так, и рейзом), `RAISE` — повышение поверх
# чужой ставки.
_AGGRESSIVE = frozenset({ActionKind.BET, ActionKind.RAISE})


@dataclass(frozen=True)
class _HandView:
    """Раздача, разобранная один раз: всё, что не зависит от метки.

    Заведена ради `player_stats_by_label`: те же величины — улицы,
    префлоп-агрессор, продолженная ставка, кто спасовал — спрашиваются у одной
    раздачи столько раз, сколько за столом мест, и пересчитывать их на каждое
    место незачем. Формулы ниже принимают вид, а не руку.

    `cbet_index` — позиция продолженной ставки в списке действий флопа либо
    `None`, если её нет. Определение продолженной ставки живёт здесь, в одном
    месте: и «поставил ли агрессор», и «сдался ли кто-то на его ставку»
    смотрят на это поле, а не заводят вторую формулу.
    """

    hand: CanonicalHand
    streets: dict[Street, list[CanonicalAction]]
    aggressor: str | None
    cbet_index: int | None
    folded_preflop: frozenset[str]
    folded: frozenset[str]
    live_at_end: int


def _view(hand: CanonicalHand) -> _HandView:
    """Разбор раздачи на величины, общие для всех мест за столом.

    Префлоп-агрессор — ПОСЛЕДНИЙ, кто повысил на префлопе. Продолженная ставка
    — первая ставка (`bet`) на флопе, если её сделал он; ставка от коллера,
    когда агрессор чекнул, продолженной не считается
    (`test_a_flop_bet_from_a_caller_is_not_a_continuation_bet`).
    """
    streets: dict[Street, list[CanonicalAction]] = {street: [] for street in Street}
    folded_preflop: set[str] = set()
    folded: set[str] = set()
    for action in hand.actions:
        streets[action.street].append(action)
        if action.kind is ActionKind.FOLD:
            folded.add(action.label)
            if action.street is Street.PREFLOP:
                folded_preflop.add(action.label)

    aggressor: str | None = None
    for action in streets[Street.PREFLOP]:
        if action.kind in _AGGRESSIVE:
            aggressor = action.label

    flop = streets[Street.FLOP]
    cbet_index = next((i for i, a in enumerate(flop) if a.kind is ActionKind.BET), None)
    if cbet_index is not None and flop[cbet_index].label != aggressor:
        cbet_index = None

    return _HandView(
        hand=hand,
        streets=streets,
        aggressor=aggressor,
        cbet_index=cbet_index,
        folded_preflop=frozenset(folded_preflop),
        folded=frozenset(folded),
        live_at_end=sum(1 for p in hand.players if p.label not in folded),
    )


def _seated(hand: CanonicalHand, label: str) -> bool:
    return any(p.label == label for p in hand.players)


def _is_vpip(view: _HandView, label: str) -> bool:
    """VPIP: есть ли у места на префлопе хоть одно действие вида колл/бет/рейз.

    Формула: `any(a.kind in {call, bet, raise} for a in preflop-действия места)`.
    Знаменатель — раздача (одна рука даёт не больше единицы, сколько бы раз
    место ни доложило).
    """
    return any(
        a.label == label and a.kind in _VOLUNTARY for a in view.streets[Street.PREFLOP]
    )


def _is_pfr(view: _HandView, label: str) -> bool:
    """PFR: есть ли у места на префлопе повышение (`bet` или `raise`).

    Формула: `any(a.kind in {bet, raise} for a in preflop-действия места)`.
    Колл в PFR не входит (`test_a_call_is_vpip_but_not_pfr`).
    """
    return any(
        a.label == label and a.kind in _AGGRESSIVE for a in view.streets[Street.PREFLOP]
    )


def _reraise(view: _HandView, label: str) -> tuple[bool, bool]:
    """Ре-рейз: (была ли возможность, воспользовалось ли место) — по префлопу.

    Возможность: перед ПЕРВЫМ действием места на префлопе кто-то другой уже
    повысил (`bet`/`raise`). То есть место отвечает на повышение, а не открывает
    банк (`test_an_open_raise_is_not_a_reraise_chance`).

    Засчитано: это первое действие места — само повышение
    (`test_raising_over_an_open_is_a_reraise`).

    Считается «3-бет и выше»: повышение поверх 3-бета (4-бет) попадает сюда же —
    формула смотрит на то, что место отвечает повышением на повышение, а не на
    номер круга торговли.
    """
    raised_before = False
    for action in view.streets[Street.PREFLOP]:
        if action.label == label:
            return raised_before, raised_before and action.kind in _AGGRESSIVE
        if action.kind in _AGGRESSIVE:
            raised_before = True
    return False, False


def _cbet_flop(view: _HandView, label: str) -> tuple[bool, bool]:
    """Продолженная ставка флопа: (была ли возможность, поставило ли место).

    Возможность есть у префлоп-агрессора
    (`test_only_the_preflop_aggressor_gets_a_cbet_chance`), у которого на флопе
    есть хоть одно действие: агрессор, вошедший во флоп олл-ином, не действует
    там вовсе и поставить не мог
    (`test_an_all_in_aggressor_has_no_flop_bet_chance`). Засчитано, когда первая
    ставка флопа — его (`_HandView.cbet_index`).
    """
    if view.aggressor != label:
        return False, False
    if not any(a.label == label for a in view.streets[Street.FLOP]):
        return False, False
    return True, view.cbet_index is not None


def _barrel(view: _HandView, label: str, street: Street, bet_before: bool) -> tuple[bool, bool]:
    """Ставка на следующей улице после своей ставки: (возможность, поставило ли место).

    Возможность считается только после того, как место поставило на предыдущей
    улице (`bet_before`): второй баррель без первого — не второй
    (`test_a_barrel_needs_a_bet_on_the_street_before`). Кроме того, у места
    должно быть действие на этой улице.

    Засчитано, когда ПЕРВАЯ ставка улицы — его: повышение поверх чужой ставки
    здесь не считается (`test_raising_someone_elses_turn_bet_is_not_a_barrel`).
    """
    if not bet_before:
        return False, False
    actions = view.streets[street]
    if not any(a.label == label for a in actions):
        return False, False
    first_bet = next((a for a in actions if a.kind is ActionKind.BET), None)
    return True, first_bet is not None and first_bet.label == label


def _fold_to_cbet(view: _HandView, label: str) -> tuple[bool, bool]:
    """Сдача на продолженную ставку: (была ли она против места, сдалось ли место).

    Против места она есть, если продолженная ставка в раздаче сделана
    (`_HandView.cbet_index`) не им самим (`test_hero_own_cbet_is_not_a_chance`)
    и у места есть действие на флопе после этой ставки. Сдачей считается, если
    это действие — пас.
    """
    if view.cbet_index is None:
        return False, False
    flop = view.streets[Street.FLOP]
    if flop[view.cbet_index].label == label:
        return False, False
    for action in flop[view.cbet_index + 1 :]:
        if action.label == label:
            return True, action.kind is ActionKind.FOLD
    return False, False


def _saw_flop(view: _HandView, label: str) -> bool:
    """Флоп раздан и место не спасовало до него.

    Пас НА флопе флоп увидеть не мешает — знаменатель считает дошедших до
    улицы, а не доигравших её (`test_folding_on_the_flop_still_counts_as_seeing_it`).
    """
    return Street.FLOP in view.hand.boards and label not in view.folded_preflop


def _went_to_showdown(view: _HandView, label: str) -> bool:
    """Место дошло до вскрытия: доска доехала до ривера, оно не пасовало, и оно не одно.

    Считается по пасам и доске, а не по строкам показа карт: источник пишет
    показ и за тем, кто спасовал, и такая строка вскрытием не является
    (`test_cards_shown_after_a_fold_are_not_a_showdown`). Двое непасовавших —
    условие того, что вскрытие вообще состоялось: когда последнюю ставку никто
    не уравнял, до вскрытия не дошёл никто
    (`test_a_river_fold_leaves_no_showdown_for_anyone`).
    """
    return (
        Street.RIVER in view.hand.boards
        and view.live_at_end >= 2
        and label not in view.folded
    )


def _accumulate(stats: PlayerStats, view: _HandView, label: str) -> None:
    """Прибавляет к счётчикам вклад одной раздачи для одного места."""
    stats.hands += 1
    stats.vpip += _is_vpip(view, label)
    stats.pfr += _is_pfr(view, label)

    chance, taken = _reraise(view, label)
    stats.reraise_chances += chance
    stats.reraise += taken

    faced, folded = _fold_to_cbet(view, label)
    stats.cbet_faced += faced
    stats.fold_to_cbet += folded

    flop_chance, flop_bet = _cbet_flop(view, label)
    stats.cbet_flop_chances += flop_chance
    stats.cbet_flop += flop_bet

    turn_chance, turn_bet = _barrel(view, label, Street.TURN, flop_bet)
    stats.barrel_turn_chances += turn_chance
    stats.barrel_turn += turn_bet

    river_chance, river_bet = _barrel(view, label, Street.RIVER, turn_bet)
    stats.barrel_river_chances += river_chance
    stats.barrel_river += river_bet

    if _saw_flop(view, label):
        stats.flops_seen += 1
        stats.showdowns += _went_to_showdown(view, label)


def player_stats(hands: Iterable[CanonicalHand], label: str | None = None) -> PlayerStats:
    """Счётчики одного места по списку рук; `label=None` — место героя каждой руки.

    Раздачи, где этого места за столом нет, не считаются вовсе — ни в
    числителе, ни в знаменателе: метка в такой руке не указывает ни на одного
    игрока, и любая из формул выше говорила бы о несуществующем месте. Для
    умолчания это раздачи чужого турнира, для метки оппонента — ещё и те, где
    он не сидел за столом героя
    (`test_hands_without_that_label_are_not_counted`).

    Список может быть и всей историей игрока сразу: счётчики складываются по
    рукам, поэтому турнир из 300 раздач весит в среднем ровно в 25 раз больше
    турнира из 12 (`test_hands_from_several_tournaments_are_weighed_by_hands`).
    Усреднить готовые проценты по турнирам дало бы другое число.
    """
    return player_stats_of_seats(
        hands, lambda hand: [hand.hero_label if label is None else label]
    )


def player_stats_by_label(hands: Iterable[CanonicalHand]) -> dict[str, PlayerStats]:
    """Те же счётчики сразу по всем местам: метка места → её статистика.

    Один проход по рукам: каждая раздача разбирается одним `_view` на все места
    за столом сразу. Отдельного пути для героя здесь нет — он в словаре под
    своей меткой, и его счётчики те же, что вернул бы запрос одного места
    (`test_by_label_matches_the_single_label_call`).

    Знаменатель у каждого места свой: место считается только в тех раздачах,
    где оно за столом (`test_by_label_counts_each_label_in_its_own_hands`).
    Ключи идут в порядке первого появления метки.
    """
    out: dict[str, PlayerStats] = {}
    for hand in hands:
        view = _view(hand)
        for player in hand.players:
            _accumulate(out.setdefault(player.label, PlayerStats()), view, player.label)
    return out


def player_stats_across_tournaments(
    hands: Iterable[CanonicalHand], labels: Mapping[str, str]
) -> PlayerStats:
    """Счётчики ОДНОГО игрока, у которого в каждом турнире своя метка.

    `labels` — «турнир комнаты (`CanonicalHand.tournament_id`) → метка, которая
    принадлежит ему в этом турнире». Такое соответствие приходит снаружи,
    целиком: в файлах раздач участники обезличены, метка сквозная только внутри
    турнира, и связать метки разных турниров может лишь тот, кто утверждает, что
    это один человек. Здесь эта связь ни выводится, ни проверяется — она
    аргумент.

    **Второй реализации формул нет.** Складываются те же счётчики тем же
    накопителем, что у `player_stats`: числители и знаменатели каждого турнира
    прибавляются к общим, поэтому турнир из 300 раздач весит в среднем ровно в
    25 раз больше турнира из 12 — как и в `player_stats`, и по той же причине
    (`test_stats_across_tournaments_add_up_what_each_tournament_counted`).

    **Почему соответствие, а не список пар «рука — метка».** Ключ здесь —
    турнир, поэтому одна раздача не может попасть в счётчики дважды под двумя
    метками, как бы ни выглядел вход: у раздачи один турнир, у турнира одна
    метка. Список пар такой гарантии не даёт, а её нарушение — не отказ, а
    молча удвоенное число.

    Раздача турнира, которого в соответствии нет, не считается вовсе — ни в
    числителе, ни в знаменателе: про этот турнир не сказано, кто в нём наш
    (`test_a_tournament_nobody_bound_is_not_counted`). Так же пропускается
    раздача, в которой названной метки нет за столом, — то же правило и та же
    причина, что в `player_stats`.
    """
    def seats(hand: CanonicalHand) -> list[str]:
        label = labels.get(hand.tournament_id)
        return [] if label is None else [label]

    return player_stats_of_seats(hands, seats)


def player_stats_of_seats(
    hands: Iterable[CanonicalHand], seats_of: Callable[[CanonicalHand], Iterable[str]]
) -> PlayerStats:
    """Счётчики по местам, которые `seats_of` называет в каждой раздаче.

    ОДИН накопитель на все входы пакета: `player_stats` и
    `player_stats_across_tournaments` — это он же с разными `seats_of`, второй
    реализации формул в модуле нет
    (`test_every_entry_point_shares_one_accumulator`).

    Ради чего заведено сверх них: словарь расчётов спрашивает частоту не только
    одного места, но и ПОЛЯ — всех оппонентов героя, сложенных в одно число, — и
    фильтрует места по позиции. И то и другое выражается выбором мест в раздаче,
    а не новой формулой.

    **Знаменатель — место-раздача, а не раздача.** Когда `seats_of` называет в
    одной руке несколько мест, рука прибавляет к `hands` столько же
    (`test_two_seats_in_one_hand_count_twice`). Для поля это и есть нужный
    знаменатель: доля мест, добровольно вложивших фишки, а не доля раздач, где
    это сделал хоть кто-то.

    Место, которого в этой раздаче нет за столом, пропускается — то же правило и
    та же причина, что в `player_stats`: метка тогда не указывает ни на одного
    игрока.
    """
    stats = PlayerStats()
    for hand in hands:
        view = _view(hand)
        for label in seats_of(hand):
            if _seated(hand, label):
                _accumulate(stats, view, label)
    return stats


def seat_position(hand: CanonicalHand, label: str) -> str | None:
    """Позиция места в этой раздаче или `None`, если места за столом нет.

    Позиция вычислена нормализатором от кнопки (`PlayerState.position`) — здесь
    она читается, а не выводится заново.
    """
    for player in hand.players:
        if player.label == label:
            return player.position
    return None


# Названная частота → пара «числитель, знаменатель» в `PlayerStats`. Таблица, а
# не ветки: знаменатели у этих величин РАЗНЫЕ, и пара сходится с долей, которую
# считает сам контракт, для каждого имени набора
# (`test_every_named_frequency_matches_the_share_the_contract_computes`).
_COUNTERS: dict[FrequencyStat, tuple[str, str]] = {
    FrequencyStat.VPIP: ("vpip", "hands"),
    FrequencyStat.PFR: ("pfr", "hands"),
    FrequencyStat.RERAISE: ("reraise", "reraise_chances"),
    FrequencyStat.FOLD_TO_CBET: ("fold_to_cbet", "cbet_faced"),
    FrequencyStat.CBET_FLOP: ("cbet_flop", "cbet_flop_chances"),
    FrequencyStat.BARREL_TURN: ("barrel_turn", "barrel_turn_chances"),
    FrequencyStat.BARREL_RIVER: ("barrel_river", "barrel_river_chances"),
    FrequencyStat.SHOWDOWN: ("showdowns", "flops_seen"),
}


def measurement_of(stats: PlayerStats, stat: FrequencyStat) -> Measurement:
    """Названная частота как пара «числитель, знаменатель».

    Доли здесь не считается вовсе: пара едет в результат расчёта целиком, и
    именно она не даёт подписи разойтись со знаменателем.
    """
    numerator, denominator = _COUNTERS[stat]
    return Measurement(
        numerator=getattr(stats, numerator), denominator=getattr(stats, denominator)
    )
