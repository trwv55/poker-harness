"""Требование к ставящему диапазону соперника на тёрне и флопе.

**Тот же вопрос, что на ривере, и другой ответ под ним.** `river_call` спрашивает
«сколько блефов должно быть в его ставящем диапазоне, чтобы колл вышел в ноль», и
отвечает перебором борда: на полном борде эквити героя против каждого комбо — ноль,
половина или единица. Здесь борд не полон, карта ещё идёт, и против конкретного
комбо у героя доля, а не исход. Поэтому вопрос ставится не через «бьёт/не бьёт», а
через СОСТАВ диапазона по эквити:

    пусть его ставящий диапазон — это `V` комбо минимального вэлью со средней
    эквити героя `e_V` плюс `B` комбо из остального; эквити героя против такого
    диапазона — `(V·e_V + сумма эквити добранных B) / (V + B)`, и колл выходит в
    ноль, когда она равна `required_equity`.

**Эквити каждого комбо — перебор доборов целиком.** На тёрне добор одной картой —
44 варианта на комбо, на флопе двумя — 990 пар; перебираются все
(`test_the_turn_enumerates_every_river_card`,
`test_the_flop_enumerates_every_pair_of_run_outs`,
`test_every_combo_is_dealt_the_same_number_of_run_outs`). Сэмплирования здесь
нет.

**Минимальное вэлью — правило ривера, применённое к ВИДИМОМУ борду.** Комбо входит
в минимальное вэлью, если на картах, лежащих на столе в момент решения, оно бьёт
героя И составляет комбинацию старшего класса (`river_call._HAND_TYPES`). Класс
берётся оттуда же, а не заводится второй раз: две таблицы классов разошлись бы
молча. Руки, которые бьют героя внутри его класса, в минимум не входят — они и
есть та часть вэлью, которую соперник может не ставить
(`test_min_value_on_the_turn_is_two_pair_and_better`,
`test_min_value_never_exceeds_the_combos_ahead`).

**Сколько блефов нужно — минимум по подмножествам, а не формула.** На ривере все
проигрывающие комбо дают героя одну и ту же эквити, и число блефов берётся из
замкнутой формулы. Здесь комбо «остального» по эквити неравноценны, а какие из
них он ставит, мы не угадываем, — поэтому берётся выбор, самый выгодный для
колла: комбо добираются в порядке убывания эквити героя, и `B` — первое число, на
котором средняя эквити достигает требуемой. При любом другом наборе той же
величины средняя эквити не выше, то есть занизить `B` нельзя
(`test_the_bluff_count_cannot_be_lowered_by_another_choice`).

`bluffs_needed_min_value = None` — требуемой эквити не даёт НИ ОДИН диапазон,
содержащий минимальное вэлью целиком: даже добрав всё остальное, что лежит на
борде, средняя эквити героя остаётся ниже требуемой
(`test_a_hopeless_hand_has_no_number_of_bluffs_that_helps`).

**Чего этот инструмент не считает.** Он отвечает на вопрос об эквити ставящего
диапазона к вскрытию. Доигрывание оставшихся улиц — ставки, которые ещё пойдут в
банк, — в перебор не входит, и лучшего действия по его числам не называется
(`analysis.turn_flop`).
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import eval7
from pydantic import BaseModel, model_validator

from harness.analysis.tools.equity import _FULL_DECK
from harness.analysis.tools.pot_odds import required_equity

# Класс комбинации берётся у риверного перебора: правило минимального вэлью —
# одно на все улицы, и вторая таблица классов означала бы две редакции правила.
from harness.analysis.tools.river_call import _type_index

_VISIBLE_BOARDS = (3, 4)
_FULL_BOARD = 5


class TurnFlopCallRequirement(BaseModel):
    """Разбор видимого борда и требование к ставящему диапазону соперника.

    `combos_total` — все комбо, оставшиеся в колоде после мёртвых карт (карты
    героя и видимый борд); `combos_ahead` — те из них, что бьют героя НА ЭТОМ
    БОРДЕ, `min_value_combos` — подмножество бьющих старшим классом.

    `min_value_equity` — средняя эквити героя против минимального вэлью,
    посчитанная перебором доборов; ноль, когда минимального вэлью нет вовсе.

    `bluffs_needed_min_value` — сколько комбо помимо минимального вэлью обязано
    быть в ставящем диапазоне, чтобы эквити героя против него дошла до
    `required_equity`. `None` — не хватает и всего борда.
    """

    required_equity: float
    combos_total: int
    combos_ahead: int
    min_value_combos: int
    min_value_equity: float
    bluffs_needed_min_value: int | None

    @model_validator(mode="after")
    def _min_value_is_a_subset(self) -> TurnFlopCallRequirement:
        """Минимальное вэлью — часть бьющих героя комбо, а не отдельный счёт."""
        if self.min_value_combos > self.combos_ahead:
            raise ValueError(
                f"минимальное вэлью {self.min_value_combos} шире, чем бьющих героя комбо "
                f"{self.combos_ahead}"
            )
        return self


def turn_flop_call_requirement(
    hero: tuple[str, str],
    board: list[str],
    pot_before: int,
    to_call: int,
) -> TurnFlopCallRequirement:
    """Требование к ставящему диапазону соперника на незавершённом борде.

    `board` — карты, лежащие на столе В МОМЕНТ РЕШЕНИЯ: три на флопе, четыре на
    тёрне. Пять карт — это ривер, и считает его `river_call`.

    `pot_before` — банк на момент решения, уже со ставкой соперника внутри (как в
    `pot_odds.required_equity`); `to_call` — цена колла. Обе величины в фишках и
    обе строго положительны.

    Карты героя и борда не пересекаются, все — из колоды в записи `equity`
    ("Ah", "Td"). Иначе `ValueError`.
    """
    if len(board) not in _VISIBLE_BOARDS:
        raise ValueError(f"борд тёрна или флопа — 3 или 4 карты, получено {len(board)}")
    if pot_before <= 0:
        raise ValueError(f"банк на решении должен быть положительным, получено {pot_before}")
    if to_call <= 0:
        raise ValueError(f"цена колла должна быть положительной, получено {to_call}")

    known = list(hero) + list(board)
    unknown = [card for card in known if card not in _FULL_DECK]
    if unknown:
        raise ValueError(f"не карты колоды: {unknown}")
    if len(set(known)) != len(known):
        raise ValueError(f"карта встречается дважды среди карт героя и борда: {known}")

    req = required_equity(to_call, pot_before)
    rest = [card for card in _FULL_DECK if card not in set(known)]
    cards = [eval7.Card(card) for card in rest]
    board_cards = [eval7.Card(card) for card in board]
    hero_cards = [eval7.Card(card) for card in hero]

    equity = _equity_by_combo(hero_cards, board_cards, cards, _FULL_BOARD - len(board))
    hero_score = eval7.evaluate(hero_cards + board_cards)
    hero_type = _type_index(hero_score)

    ahead = 0
    value: list[float] = []
    pool: list[float] = []
    for i, j in combinations(range(len(rest)), 2):
        score = eval7.evaluate([cards[i], cards[j]] + board_cards)
        if score > hero_score:
            ahead += 1
            if _type_index(score) > hero_type:
                value.append(equity[(i, j)])
                continue
        pool.append(equity[(i, j)])

    return TurnFlopCallRequirement(
        required_equity=req,
        combos_total=len(value) + len(pool),
        combos_ahead=ahead,
        min_value_combos=len(value),
        min_value_equity=sum(value) / len(value) if value else 0.0,
        bluffs_needed_min_value=_bluffs_needed(value, pool, req),
    )


def _equity_by_combo(
    hero_cards: list[eval7.Card],
    board_cards: list[eval7.Card],
    cards: list[eval7.Card],
    to_come: int,
) -> dict[tuple[int, int], float]:
    """Доля героя во вскрытии против каждого комбо — перебором всех доборов.

    Ключ — пара индексов в `cards` (i < j). Ничья считается половиной.

    Знаменатель один на все комбо: доборы, не задевающие двух карт комбо, — это
    сочетания из оставшихся `size - 2` карт, и их число от самого комбо не
    зависит (`test_every_combo_is_dealt_the_same_number_of_run_outs`).

    Цикл идёт по доборам, а не по комбо: рука героя на добранном борде считается
    тогда один раз на добор, а не заново на каждое комбо соперника.
    """
    size = len(cards)
    won = [[0.0] * size for _ in range(size)]
    evaluate = eval7.evaluate
    for run_out in combinations(range(size), to_come):
        taken = set(run_out)
        full_board = board_cards + [cards[x] for x in run_out]
        hero_score = evaluate(hero_cards + full_board)
        for i in range(size):
            if i in taken:
                continue
            row = won[i]
            with_first = full_board + [cards[i]]
            for j in range(i + 1, size):
                if j in taken:
                    continue
                score = evaluate(with_first + [cards[j]])
                if score < hero_score:
                    row[j] += 1.0
                elif score == hero_score:
                    row[j] += 0.5
    run_outs = comb(size - 2, to_come)
    return {
        (i, j): won[i][j] / run_outs for i in range(size) for j in range(i + 1, size)
    }


def _bluffs_needed(value: list[float], pool: list[float], req: float) -> int | None:
    """Наименьшее число комбо из `pool`, доводящее среднюю эквити до `req`.

    Ноль — вэлью нет вовсе (среднюю не от чего считать) или эквити героя против
    одного минимального вэлью уже не ниже требуемой: «блефов нужно не меньше
    нуля» — верное утверждение о любом диапазоне (`test_a_hand_ahead_of_the_value_needs_no_bluffs`).

    `None` — требуемой средней не даёт ни одно число комбо: добор идёт по
    убыванию эквити, поэтому исчерпанный `pool` означает, что и лучший выбор не
    дотягивает.
    """
    if not value:
        return 0
    total = sum(value)
    count = len(value)
    if total / count >= req:
        return 0
    for taken, combo_equity in enumerate(sorted(pool, reverse=True), start=1):
        total += combo_equity
        count += 1
        if total / count >= req:
            return taken
    return None
