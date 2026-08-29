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

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import combinations
from math import comb, prod
from operator import mul
from pathlib import Path

from harness.analysis.tools.equity import combos_of_class, equity_vs_ranges
from harness.contracts import Range, all_classes

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


# --- Таблица эквити 169x169 и хедз-ап равновесие --------------------------------

_DATA_DIR = Path(__file__).parent / "data"
_EQ169_PATH = _DATA_DIR / "eq169.json"

# Хедз-ап пуш-фолд: SB ставит 0.5, BB ставит 1.0, SB ходит первым — шов на весь
# эффективный стек или фолд; BB отвечает коллом или фолдом. Все EV — в bb
# относительно начала раздачи (до постов): фолд SB = -0.5, проход шова = +1.0,
# вскрытие на eff_bb с каждого = 2*equity*eff_bb - eff_bb.
_SB_BLIND = 0.5
_BB_BLIND = 1.0

# Fictitious play: чередование лучших ответов на СРЕДНЮЮ стратегию оппонента с
# усреднением по итерациям (для зеро-сум игры это сходится).
#
# Критерий остановки — эксплуатируемость, то есть сумма того, что каждый игрок
# выигрывает, отклонившись от средней стратегии против средней стратегии
# соперника. Это определение ε-равновесия, и оно считается внутри цикла даром:
# лучший ответ и так перебирает EV всех альтернатив.
#
# Изменения значения игры |ΔEV| как критерия НЕ хватает, и это измерено, а не
# предположено: при 1/t-усреднении значение меняется на O(1/t) само по себе, так
# что порог |ΔEV| < 0.001bb достигается уже на 3-19-й итерации, когда
# эксплуатируемость ещё 0.01-0.04bb, а диапазоны гуляют на 5-7% комбо. Порог по
# |ΔEV| оставлен как дополнительное (более слабое) условие, но решает первое.
#
# 1e-3 bb на двоих — с большим запасом ниже шкалы, на которой решения различимы
# (пуш и фолд расходятся на десятые доли bb). Замер: этот порог достигается на
# ~20 итерациях при 2bb, ~210 при 5bb, ~400 при 10bb, ~600 при 15bb; дальнейшие
# итерации двигают ширину диапазонов уже меньше чем на 0.001 доли комбо.
_FP_EXPLOITABILITY_BB = 1e-3
_FP_VALUE_TOLERANCE_BB = 1e-3
_FP_MAX_ITERATIONS = 20_000

# Минимум шагов усреднения — не запас "на всякий случай", а граница на вес
# произвольного стартового убеждения: первый лучший ответ считается против
# равномерного 0.5 и входит в среднее с весом 1/N. На мелких стеках игра почти
# тривиальна, и порог по эксплуатируемости достигается за 10 шагов — тогда в диапазоне
# остаются мусорные веса по 0.1 у рук, которые пушатся только против этого
# стартового убеждения. При 200 шагах остаток ограничен 0.005.
_FP_MIN_AVERAGING_STEPS = 200

_MIN_EFF_BB = 1.0
_MAX_EFF_BB = 100.0

# Веса равновесных диапазонов округляются перед сохранением, чтобы значение из
# кэша совпадало с пересчитанным бит в бит, а файл оставался читаемым.
_WEIGHT_PRECISION = 6


@cache
def _eq169() -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...]]:
    """Таблица эквити класс-против-класса, сгенерированная `scripts/build_eq169.py`."""
    if not _EQ169_PATH.exists():
        raise FileNotFoundError(
            f"нет таблицы эквити {_EQ169_PATH} — она генерируется один раз командой "
            f"`uv run python scripts/build_eq169.py` и коммитится как данные"
        )
    payload = json.loads(_EQ169_PATH.read_text(encoding="utf-8"))
    classes = tuple(payload["classes"])
    matrix = tuple(tuple(float(x) for x in row) for row in payload["equity"])
    expected = tuple(all_classes())
    if classes != expected:
        raise ValueError("порядок классов в eq169.json разошёлся с all_classes()")
    if len(matrix) != len(classes) or any(len(row) != len(classes) for row in matrix):
        raise ValueError(f"таблица эквити должна быть {len(classes)}x{len(classes)}")
    return classes, matrix


@cache
def _class_index() -> dict[str, int]:
    return {cls: i for i, cls in enumerate(_eq169()[0])}


def class_equity(hero_cls: str, villain_cls: str) -> float:
    """Префлоп-эквити класса против класса в хедз-апе — из предвычисленной таблицы.

    Значения получены Монте-Карло в `scripts/build_eq169.py` и лежат в репозитории
    как данные: считать их на лету нельзя (14 365 пар — минуты), а fictitious play
    зовёт эту функцию десятки миллионов раз.
    """
    index = _class_index()
    if hero_cls not in index or villain_cls not in index:
        unknown = hero_cls if hero_cls not in index else villain_cls
        raise ValueError(f"неизвестный класс руки: {unknown!r}")
    return _eq169()[1][index[hero_cls]][index[villain_cls]]


@cache
def _live_combo_counts() -> tuple[tuple[float, ...], ...]:
    """N[h][c] — сколько комбо класса c в среднем живы, когда у игрока рука класса h.

    Это учёт блокеров на уровне классов, и он не факультативен: держа AA, игрок
    оставляет оппоненту одно комбо AA из шести. Сумма N[h] по всем c равна ровно
    C(50,2) = 1225 — столько рук может быть у оппонента после снятия двух карт
    героя; это же и проверка корректности подсчёта.
    """
    classes = all_classes()
    combos = {cls: combos_of_class(cls) for cls in classes}
    per_card: list[dict[str, int]] = []
    as_set: list[set[frozenset[str]]] = []
    for cls in classes:
        counts: dict[str, int] = {}
        for c1, c2 in combos[cls]:
            counts[c1] = counts.get(c1, 0) + 1
            counts[c2] = counts.get(c2, 0) + 1
        per_card.append(counts)
        as_set.append({frozenset(combo) for combo in combos[cls]})

    matrix: list[tuple[float, ...]] = []
    for hero_cls in classes:
        hero_combos = combos[hero_cls]
        row: list[float] = []
        for j, villain_cls in enumerate(classes):
            total = 0
            for card1, card2 in hero_combos:
                blocked = per_card[j].get(card1, 0) + per_card[j].get(card2, 0)
                if frozenset((card1, card2)) in as_set[j]:
                    blocked -= 1  # комбо, состоящее ровно из карт героя, вычтено дважды
                total += len(combos[villain_cls]) - blocked
            row.append(total / len(hero_combos))
        if abs(sum(row) - comb(_DECK_SIZE - 2, 2)) > 1e-9:
            raise ValueError(f"живых комбо у оппонента против {hero_cls} не 1225: {sum(row)}")
        matrix.append(tuple(row))
    return tuple(matrix)


@cache
def _conditional_class_probs() -> tuple[tuple[float, ...], ...]:
    """P(у оппонента класс c | у игрока класс h) с учётом снятых карт."""
    total = float(comb(_DECK_SIZE - 2, 2))
    return tuple(tuple(n / total for n in row) for row in _live_combo_counts())


@cache
def _class_priors() -> tuple[float, ...]:
    """Безусловная вероятность класса руки: комбо класса от 1326."""
    return tuple(
        Range(weights={cls: 1.0}).fraction_of_hands() for cls in all_classes()
    )


def _solve_nash_hu(eff_bb: float) -> tuple[dict[str, float], dict[str, float], int, float]:
    """Fictitious play для хедз-ап пуш-фолда.

    Возвращает (диапазон шова, диапазон колла, число шагов усреднения, EV SB в bb).
    """
    classes = all_classes()
    _, equity = _eq169()
    conditional = _conditional_class_probs()
    priors = _class_priors()
    size = len(classes)

    # EV шова для SB с рукой h против средней стратегии BB:
    #   EV(h) = Σ_c P(c|h) * [ call_w(c) * (2*eq(h,c)-1)*eff + (1-call_w(c)) * (+1) ]
    #         = 1 + Σ_c push_term[h][c] * call_w(c),   т.к. Σ_c P(c|h) = 1.
    push_term = [
        tuple(
            conditional[h][c] * ((2.0 * equity[h][c] - 1.0) * eff_bb - _BB_BLIND)
            for c in range(size)
        )
        for h in range(size)
    ]
    # Разница "колл минус фолд" для BB с рукой b против шова руки h:
    #   2*eq(b,h)*eff - eff + 1 (фолд стоит BB его блайнда).
    call_term = [
        tuple(
            conditional[b][h] * (2.0 * equity[b][h] * eff_bb - eff_bb + _BB_BLIND)
            for h in range(size)
        )
        for b in range(size)
    ]

    avg_push = [0.5] * size  # стартовое убеждение, в среднее не входит: на шаге 1
    avg_call = [0.5] * size  # оно целиком заменяется первым лучшим ответом
    averaging_steps = 0
    previous_value = None
    value = 0.0

    for step in range(1, _FP_MAX_ITERATIONS + 1):
        shove_ev = [_BB_BLIND + sum(map(mul, push_term[h], avg_call)) for h in range(size)]
        call_minus_fold = [sum(map(mul, call_term[b], avg_push)) for b in range(size)]

        value = sum(
            priors[h] * (avg_push[h] * shove_ev[h] + (1.0 - avg_push[h]) * -_SB_BLIND)
            for h in range(size)
        )
        # Эксплуатируемость текущей пары средних стратегий: сколько даёт отклонение.
        gain_sb = sum(
            priors[h]
            * (
                max(shove_ev[h], -_SB_BLIND)
                - (avg_push[h] * shove_ev[h] + (1.0 - avg_push[h]) * -_SB_BLIND)
            )
            for h in range(size)
        )
        gain_bb = sum(
            priors[b] * (max(call_minus_fold[b], 0.0) - avg_call[b] * call_minus_fold[b])
            for b in range(size)
        )
        converged = (
            averaging_steps >= _FP_MIN_AVERAGING_STEPS
            and gain_sb + gain_bb <= _FP_EXPLOITABILITY_BB
            and previous_value is not None
            and abs(value - previous_value) < _FP_VALUE_TOLERANCE_BB
        )
        if converged:
            break
        previous_value = value

        br_push = [1.0 if ev > -_SB_BLIND else 0.0 for ev in shove_ev]
        br_call = [1.0 if diff > 0.0 else 0.0 for diff in call_minus_fold]
        averaging_steps += 1
        if averaging_steps == 1:
            avg_push, avg_call = br_push, br_call
        else:
            rate = 1.0 / averaging_steps
            avg_push = [w + (br - w) * rate for w, br in zip(avg_push, br_push, strict=True)]
            avg_call = [w + (br - w) * rate for w, br in zip(avg_call, br_call, strict=True)]

    push = {
        cls: round(w, _WEIGHT_PRECISION)
        for cls, w in zip(classes, avg_push, strict=True)
        if round(w, _WEIGHT_PRECISION) > 0.0
    }
    call = {
        cls: round(w, _WEIGHT_PRECISION)
        for cls, w in zip(classes, avg_call, strict=True)
        if round(w, _WEIGHT_PRECISION) > 0.0
    }
    return push, call, averaging_steps, value


def _read_nash_cache(path: Path) -> tuple[Range, Range] | None:
    """Кэш — ускорение, а не источник истины: любая порча файла ведёт к пересчёту."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return Range(weights=payload["push"]), Range(weights=payload["call"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_nash_cache(
    path: Path, eff_bb: float, solution: tuple[dict[str, float], dict[str, float], int, float]
) -> None:
    push, call, iterations, value = solution
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "eff_bb": eff_bb,
                    "fp_iterations": iterations,
                    "sb_ev_bb": round(value, 6),
                    "push": push,
                    "call": call,
                },
                ensure_ascii=False,
                indent=1,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass  # каталог пакета может быть только для чтения — считаем без кэша


def nash_hu(eff_bb: float, *, cache_dir: Path | None = None) -> tuple[Range, Range]:
    """Равновесие хедз-ап пуш-фолда на глубине `eff_bb`: (диапазон шова SB, колла BB).

    Игра: SB ставит 0.5, BB ставит 1.0, SB шовит на весь эффективный стек или
    фолдит, BB коллирует или фолдит. Решается fictitious play по предвычисленной
    таблице эквити (`class_equity`) — Монте-Карло внутри итераций не запускается,
    иначе сходимости не дождаться. Блокеры учтены: распределение руки оппонента
    условно по руке игрока.

    Результат кэшируется в `cache_dir` (по умолчанию — `data/` рядом с модулем).
    Кэш только ускоряет: при пустом или испорченном кэше считается заново и даёт
    тот же ответ — fictitious play здесь полностью детерминирован.
    """
    if not _MIN_EFF_BB < eff_bb <= _MAX_EFF_BB:
        raise ValueError(
            f"эффективный стек должен лежать в ({_MIN_EFF_BB}, {_MAX_EFF_BB}] bb: "
            f"на {eff_bb} bb шов SB не покрывает даже блайнд BB либо стек слишком глубок "
            f"для пуш-фолда"
        )

    directory = Path(cache_dir) if cache_dir is not None else _DATA_DIR
    path = directory / f"nash_hu_{eff_bb:.2f}.json"

    cached = _read_nash_cache(path)
    if cached is not None:
        return cached

    solution = _solve_nash_hu(eff_bb)
    _write_nash_cache(path, eff_bb, solution)
    return Range(weights=solution[0]), Range(weights=solution[1])
