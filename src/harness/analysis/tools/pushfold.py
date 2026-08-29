"""Пуш-фолд: EV шова и EV колла против шова, плюс проверка фолд-эквити.

Всё считается **относительно фолда**: фолд = 0. Уже поставленное в банк (блайнды,
анте) — sunk: в базлайне «я фолдю» эти деньги уже потеряны, поэтому забирая банк
без вскрытия герой получает `pot_dead_bb` целиком, включая собственные посты.

Мультивей считается **перебором подмножеств** коллеров, а не попарным
приближением: для n игроков позади перебираются все 2^n веток «кто заколлировал»,
вероятность ветки — произведение p_call по коллерам и (1 − p_call) по фолдерам.
Арифметика внутри ветки точная; приближение здесь ровно одно — сами колл-диапазоны
(их задаёт вызывающая сторона, и она же помечает вывод зоной доверия).

Ограничение модели, которое стоит знать. Стеки сравниваются в координатах «за
спиной» (`hero_behind_bb` против `behind_bb` коллера), а посты каждого игрока
по отдельности в API не передаются — известна только их сумма `pot_dead_bb`.
Поэтому формула банка `pot_dead + hero_behind + Σ min(behind_i, hero_behind)`
верна, когда суммарный вклад коллера не меньше суммарного вклада героя (обычный
случай: коллер покрывает героя либо стеки равны — тогда меньший `behind` коллера
компенсирован его большим постом, как у BB против SB). Если же коллер реально
короче героя по общему вкладу, непокрытый остаток шова герою должен был бы
вернуться, а модель считает его проигранным — оценка получается консервативной
(заниженной). Точный учёт требует знать посты по игрокам, то есть менять контракт.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import comb, prod

from harness.analysis.tools.equity import combos_of_class, equity_vs_ranges
from harness.contracts import Range

_DECK_SIZE = 52

# Больше 7 игроков позади за столом 9-max не бывает (герой + 7 = 8 мест до баттона),
# а 2^n веток растёт вдвое на каждого: кап делает стоимость расчёта предсказуемой.
_MAX_CALLERS = 7

# Ветки с вероятностью ниже этого порога отбрасываются: их вклад в EV меньше
# 1e-4 * (максимальный по модулю вклад ветки ~ десятки bb), то есть заведомо ниже
# точности самих колл-диапазонов, а каждая ветка стоит одного вызова Монте-Карло.
_MIN_BRANCH_PROB = 1e-4

# Порог фолд-эквити: проверяется не «мало ли фолдов», а «возможны ли они вообще».
# Диапазон колла, покрывающий все 1326 комбо, делает вероятность прохода шова
# структурно нулевой — тогда аргумент «шов ради фолдов» неприменим. Насколько
# фолд-эквити достаточно — вопрос к EV, а не к этому гейту.
_MIN_FOLD_PROB = 1e-6


@dataclass(frozen=True)
class CallerModel:
    """Игрок позади: его модель колл-диапазона и стек за спиной (после постов)."""

    call_range: Range
    behind_bb: float


def representative_combo(hero_cls: str) -> tuple[str, str]:
    """Детерминированный представитель класса руки в конкретных картах.

    Эквити считается по картам, а не по классам: блокеры (сколько комбо диапазона
    оппонента убивают карты героя) влияют и на эквити, и на вероятность колла.
    Выбор представителя внутри класса произволен ровно в том смысле, в каком масти
    симметричны — любые два комбо одного класса дают одинаковую эквити против
    масть-симметричного диапазона, поэтому берётся первое комбо в каноническом
    порядке `combos_of_class`.
    """
    combos = combos_of_class(hero_cls)
    if not combos:
        raise ValueError(f"неизвестный класс руки: {hero_cls!r}")
    return combos[0]


def _live_combos(rng: Range, dead: Sequence[str]) -> float:
    """Взвешенное число комбо диапазона, не задетых мёртвыми картами."""
    dead_set = set(dead)
    total = 0.0
    for cls, weight in rng.weights.items():
        if weight <= 0.0:
            continue
        alive = sum(
            1 for c1, c2 in combos_of_class(cls) if c1 not in dead_set and c2 not in dead_set
        )
        total += weight * alive
    return total


def default_call_prob(caller: CallerModel, hero_cls: str) -> float:
    """Вероятность колла = доля комбо колл-диапазона среди живых комбо оппонента.

    Карты героя сняты из колоды: у оппонента не 1326, а C(50,2) = 1225 комбо, и
    комбо, содержащие карты героя, из диапазона выпадают. Для узких диапазонов
    поправка велика (герой с AA убивает 5 из 6 комбо AA), поэтому она не факультативна.
    """
    hero_cards = representative_combo(hero_cls)
    total = comb(_DECK_SIZE - len(hero_cards), 2)
    return _live_combos(caller.call_range, hero_cards) / total


def shove_ev_bb(
    hero_cls: str,
    hero_behind_bb: float,
    pot_dead_bb: float,
    callers: list[CallerModel],
    *,
    equity_fn: Callable[..., float] = equity_vs_ranges,
    call_prob_fn: Callable[..., float] | None = None,
) -> float:
    """EV шова в bb относительно фолда (фолд = 0).

    `hero_behind_bb` — стек героя за спиной (после постов), `pot_dead_bb` — весь
    банк на момент решения, включая собственные посты героя.

    Ветка «все сфолдили» даёт `+pot_dead_bb`. Ветка, где заколлировало множество S,
    даёт `equity * банк − hero_behind_bb`, где банк =
    `pot_dead_bb + hero_behind_bb + Σ_{i∈S} min(behind_i, hero_behind_bb)`
    (коллер короче героя вносит только свой стек — остаток шова в контест не идёт).
    """
    if hero_behind_bb <= 0.0:
        raise ValueError(
            f"стек героя за спиной должен быть положительным, получено {hero_behind_bb}"
        )
    if pot_dead_bb < 0.0:
        raise ValueError(f"банк не может быть отрицательным, получено {pot_dead_bb}")
    if len(callers) > _MAX_CALLERS:
        raise ValueError(
            f"поддерживается не более {_MAX_CALLERS} игроков позади (перебор 2^n веток), "
            f"получено {len(callers)}"
        )
    if any(c.behind_bb <= 0.0 for c in callers):
        raise ValueError("стек коллера за спиной должен быть положительным")

    prob_fn = call_prob_fn or default_call_prob
    probs = [prob_fn(caller, hero_cls) for caller in callers]
    if any(not 0.0 <= p <= 1.0 for p in probs):
        raise ValueError(f"вероятность колла должна лежать в [0,1], получено {probs}")

    hero_combo = representative_combo(hero_cls)
    indices = range(len(callers))

    ev = 0.0
    for size in range(len(callers) + 1):
        for called in combinations(indices, size):
            called_set = set(called)
            branch_prob = prod(probs[i] if i in called_set else 1.0 - probs[i] for i in indices)
            if branch_prob < _MIN_BRANCH_PROB:
                continue
            if not called:
                ev += branch_prob * pot_dead_bb
                continue
            contested = (
                pot_dead_bb
                + hero_behind_bb
                + sum(min(callers[i].behind_bb, hero_behind_bb) for i in called)
            )
            equity = equity_fn(hero_combo, [callers[i].call_range for i in called])
            ev += branch_prob * (equity * contested - hero_behind_bb)
    return ev


def call_shove_ev_bb(
    hero_cls: str,
    hero_bb: float,
    shover_range: Range,
    pot_bb: float,
    to_call_bb: float,
    *,
    equity_fn: Callable[..., float] = equity_vs_ranges,
) -> float:
    """EV колла чужого шова в bb относительно фолда (фолд = 0).

    `pot_bb` — банк на момент решения, уже включая шов оппонента; `to_call_bb` —
    сколько герою доплатить. Если стек героя меньше доплаты, он коллирует на весь
    стек, а непокрытый остаток шова возвращается шоверу и в контест не входит.

    В точке безубыточности результат равен нулю ровно при эквити из
    `pot_odds.required_equity(to_call, pot_before)` — это одна и та же арифметика.
    """
    if hero_bb <= 0.0:
        raise ValueError(f"стек героя должен быть положительным, получено {hero_bb}")
    if to_call_bb <= 0.0:
        raise ValueError(f"доплата должна быть положительной, получено {to_call_bb}")

    call = min(to_call_bb, hero_bb)
    uncalled = to_call_bb - call
    contested = pot_bb - uncalled + call

    hero_combo = representative_combo(hero_cls)
    equity = equity_fn(hero_combo, [shover_range])
    return equity * contested - call


def fold_equity_ok(callers: list[CallerModel]) -> bool:
    """Есть ли у шова фолд-эквити вообще — то есть возможен ли проход без вскрытия.

    False только в структурно вырожденном случае: кто-то позади коллирует любые две
    карты, и вероятность, что все сфолдят, равна нулю. Тогда «шов ради фолдов» не
    аргумент, и вердикт о шове должен опираться только на эквити вскрытия.
    """
    prob_all_fold = prod(1.0 - c.call_range.fraction_of_hands() for c in callers)
    return prob_all_fold > _MIN_FOLD_PROB


# --- Bracket-модели колл-диапазона (вход bracket-теста зоны, задача 12) ---------
#
# Это не оценка «как коллируют на самом деле», а два намеренно грубых конца
# вилки: если вердикт одинаков и против премиум-онли, и против top-40%, он не
# зависит от угаданного диапазона. Обе модели не зависят от глубины: глубина
# принимается аргументом, потому что вызывающая сторона ею оперирует и потому
# что уточнение по глубине — вопрос данных, а не кода, и не должно менять места
# вызова. Сузить вилку без данных значило бы выдумать число.

_TIGHT_CLASSES: tuple[str, ...] = ("AA", "KK", "QQ", "JJ", "AKs", "AKo")

# Top-40% по комбо: 534 из 1326 (40.3%). Состав — стандартная форма широкого
# диапазона (пары, тузы, бродвеи, одномастные коннекторы), а не вычисленный
# порядок сил рук: это верхний конец вилки, и его задача — быть заведомо шире
# правдоподобного, а не точным.
_WIDE_CLASSES: tuple[str, ...] = (
    # пары 22+ (13 классов, 78 комбо)
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    # тузы одномастные A2s+ (12 классов, 48 комбо)
    "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    # тузы разномастные A5o+ (9 классов, 108 комбо)
    "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o", "A6o", "A5o",
    # короли одномастные K2s+ (11 классов, 44 комбо)
    "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s",
    # короли разномастные K7o+ (6 классов, 72 комбо)
    "KQo", "KJo", "KTo", "K9o", "K8o", "K7o",
    # дамы одномастные Q4s+ (8 классов, 32 комбо)
    "QJs", "QTs", "Q9s", "Q8s", "Q7s", "Q6s", "Q5s", "Q4s",
    # дамы разномастные Q9o+ (3 класса, 36 комбо)
    "QJo", "QTo", "Q9o",
    # валеты одномастные J6s+ (5 классов, 20 комбо)
    "JTs", "J9s", "J8s", "J7s", "J6s",
    # валеты разномастные J9o+ (2 класса, 24 комбо)
    "JTo", "J9o",
    # десятки одномастные T6s+ (4 класса, 16 комбо) и T9o (12 комбо)
    "T9s", "T8s", "T7s", "T6s", "T9o",
    # одномастные коннекторы и однозазорники ниже (8 классов * 4 = 32 комбо) и 98o (12)
    "98s", "97s", "96s", "98o", "87s", "86s", "76s", "75s", "65s",
)


def _bracket_tight(depth_bb: float) -> Range:
    """Узкий конец вилки: только премиум (JJ+, AK) на любой глубине."""
    return Range(weights=dict.fromkeys(_TIGHT_CLASSES, 1.0))


def _bracket_wide(depth_bb: float) -> Range:
    """Широкий конец вилки: top-40% по комбо на любой глубине."""
    return Range(weights=dict.fromkeys(_WIDE_CLASSES, 1.0))


BRACKET_TIGHT: Callable[[float], Range] = _bracket_tight
BRACKET_WIDE: Callable[[float], Range] = _bracket_wide
