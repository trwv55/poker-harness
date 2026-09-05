"""Пуш-фолд: EV шова и EV колла против шова, плюс проверка фолд-эквити.

Всё считается **относительно фолда**: фолд = 0. Уже поставленное в банк (блайнды,
анте) — sunk: в базлайне «я фолдю» эти деньги уже потеряны, поэтому забирая банк
без вскрытия герой получает `pot_dead_bb` целиком, включая собственные посты.

Мультивей считается **перебором подмножеств** коллеров, а не попарным
приближением: для n игроков позади перебираются все 2^n веток «кто заколлировал»,
вероятность ветки — произведение p_call по коллерам и (1 − p_call) по фолдерам.
Арифметика внутри ветки точная; приближение здесь ровно одно — сами колл-диапазоны
(их задаёт вызывающая сторона, и она же помечает вывод зоной доверия).

Стеки сравниваются по СУММАРНОМУ вкладу игрока (`posted + behind`), а не по
остатку за спиной. Это не педантизм: шов не может быть заколлирован на сумму
больше, чем есть у коллера, и непокрытый остаток герою возвращается. Модель,
сравнивавшая только остатки за спиной, систематически занижала EV шова против
короткого коллера (на примере «герой 10, коллер 4, мёртвых 1.5, эквити 0.5» —
-2.25bb вместо верных +0.75bb). Смещение было односторонним и удерживало от
правильных шовов, то есть порождало правдоподобно неверный вердикт — ровно тот
отказ, ради предотвращения которого этот продукт и существует. Поэтому посты
входят в контракт: `CallerModel.posted_bb` и `hero_posted_bb`.

Что в модели осталось приближением. При нескольких коллерах разной глубины
настоящая раздача образует основной банк и сайд-поты, у которых РАЗНЫЕ составы
вскрытия (в сайд-поте короткого коллера уже нет). Здесь весь контест считается
одним банком с эквити против всех заколлировавших сразу. Эквити против большего
числа оппонентов ниже, поэтому сайд-пот недооценивается — остаточное смещение
того же знака, но на порядок меньше снятого: оно возникает только когда коллеры
разной глубины и оба заколлировали.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from hashlib import sha256
from itertools import combinations
from math import comb, prod
from operator import mul
from pathlib import Path

from harness.analysis.tools.equity import combos_of_class, equity_vs_ranges
from harness.contracts import Range, all_classes

_DECK_SIZE = 52
_TOTAL_COMBOS = 1326  # C(52,2) — все стартовые комбо

# Больше 7 игроков позади за столом 9-max не бывает (герой + 7 = 8 мест до баттона),
# а 2^n веток растёт вдвое на каждого: кап делает стоимость расчёта предсказуемой.
_MAX_CALLERS = 7

# Ветки с вероятностью ниже этого порога отбрасываются: их вклад в EV меньше
# 1e-4 * (максимальный по модулю вклад ветки ~ десятки bb), то есть заведомо ниже
# точности самих колл-диапазонов, а каждая ветка стоит одного вызова Монте-Карло.
# Отброшенная масса НЕ перенормируется: сумма вероятностей учтённых веток может не
# дотягивать до единицы, и результат смещён на величину порядка отброшенной массы
# (не более 2^n * 1e-4, то есть <= 1.3% при семи игроках позади и много меньше на
# практике). Перенормировка размазала бы эту массу по оставшимся веткам, что не
# честнее: она приписала бы редким исходам вес, которого у них нет.
_MIN_BRANCH_PROB = 1e-4

# Порог фолд-эквити: проверяется не «мало ли фолдов», а «возможны ли они вообще».
# Диапазон колла, покрывающий все 1326 комбо, делает вероятность прохода шова
# структурно нулевой — тогда аргумент «шов ради фолдов» неприменим. Насколько
# фолд-эквити достаточно — вопрос к EV, а не к этому гейту.
_MIN_FOLD_PROB = 1e-6


@dataclass(frozen=True)
class CallerModel:
    """Игрок позади: модель колл-диапазона, стек за спиной и уже поставленное им.

    `behind_bb` — то, чем он ещё может заплатить; `posted_bb` — его блайнд/анте,
    уже лежащие в банке. Сумма этих двух и есть его вклад, которым меряется, на
    сколько он способен заколлировать шов. Ноль по умолчанию означает «весь банк
    состоит из чужих денег» — это осмысленный расклад, а не заглушка.
    """

    call_range: Range
    behind_bb: float
    posted_bb: float = 0.0


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
    hero_posted_bb: float = 0.0,
    equity_fn: Callable[..., float] = equity_vs_ranges,
    call_prob_fn: Callable[..., float] | None = None,
) -> float:
    """EV шова в bb относительно фолда (фолд = 0).

    `hero_behind_bb` — стек героя за спиной (после постов), `hero_posted_bb` — его
    собственные блайнд и анте, `pot_dead_bb` — весь банк на момент решения, включая
    посты всех, кто ещё в раздаче.

    Ветка «все сфолдили» даёт `+pot_dead_bb` (герой забирает банк целиком, свои
    посты в базлайне уже списаны). Ветка, где заколлировало множество S:

    - вклад игрока = `posted + behind`; матчится не больше, чем есть у соперника,
      поэтому герой рискует `at_risk = min(вклад героя, max вклад среди S)`,
      а каждый коллер вносит `min(вклад_i, at_risk)`;
    - непокрытый остаток шова возвращается герою, поэтому цена ветки — не весь
      стек за спиной, а `at_risk − hero_posted_bb`;
    - деньги игроков, которых в этой ветке нет (уже сбросивших до решения и
      сбросивших в этой ветке коллеров), остаются в банке:
      `pot_dead_bb − hero_posted_bb − Σ_{i∈S} posted_i`;
    - вклад ветки = `equity * банк − (at_risk − hero_posted_bb)`.
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
    if hero_posted_bb < 0.0 or any(c.posted_bb < 0.0 for c in callers):
        raise ValueError("уже поставленное не может быть отрицательным")
    posted_total = hero_posted_bb + sum(c.posted_bb for c in callers)
    if posted_total > pot_dead_bb + 1e-9:
        raise ValueError(
            f"сумма постов ({posted_total}) больше банка ({pot_dead_bb}): банк на момент "
            f"решения обязан включать посты всех, кто ещё в раздаче"
        )

    prob_fn = call_prob_fn or default_call_prob
    probs = [prob_fn(caller, hero_cls) for caller in callers]
    if any(not 0.0 <= p <= 1.0 for p in probs):
        raise ValueError(f"вероятность колла должна лежать в [0,1], получено {probs}")

    hero_combo = representative_combo(hero_cls)
    hero_total = hero_posted_bb + hero_behind_bb
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
            totals = [callers[i].posted_bb + callers[i].behind_bb for i in called]
            at_risk = min(hero_total, max(totals))
            others = pot_dead_bb - hero_posted_bb - sum(callers[i].posted_bb for i in called)
            contested = others + at_risk + sum(min(total, at_risk) for total in totals)
            equity = equity_fn(hero_combo, [callers[i].call_range for i in called])
            ev += branch_prob * (equity * contested - (at_risk - hero_posted_bb))
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



# --- Таблица эквити 169x169 и хедз-ап равновесие --------------------------------

_DATA_DIR = Path(__file__).parent / "data"
_EQ169_PATH = _DATA_DIR / "eq169.json"

# Хедз-ап пуш-фолд: SB ставит 0.5, BB ставит 1.0, SB ходит первым — шов на весь
# эффективный стек или фолд; BB отвечает коллом или фолдом.
#
# `dead_extra_bb` — мёртвые деньги сверх двух блайндов, то есть СУММА АНТЕ всего
# стола. Без этого параметра решалась не та игра, в которую играет пользователь:
# продукт заявлен для MTT с анте, а в фикстуре анте 0.125bb с семи игроков даёт
# 2.375bb мёртвых денег против 1.5bb, которые предполагает игра без анте — на
# 58% больше. Больше мёртвых денег — шире и пуш, и колл, поэтому равновесие без
# анте ТЕСНЕЕ реально разыгрываемого, и продукт помечал бы верные шовы ошибкой.
# Для тренажёра ложное обвинение хуже пропущенной ошибки: оно учит пасовать там,
# где надо входить.
#
# Соглашение о величинах (важно, иначе анте посчитается дважды). `eff_bb` — это
# эффективный стек ПОСЛЕ уплаты анте, то есть всё, что игрок может поставить
# сверх своего анте. Тогда:
#   * банк, который SB забирает при пасе BB   = 0.5 + 1.0 + dead;
#   * банк вскрытия                            = 2*eff_bb + dead
#     (свои анте обоих участников уже сидят внутри `dead`, а их ставки — внутри
#     eff_bb, так что каждое слагаемое учтено ровно один раз);
#   * риск SB относительно паса                = eff_bb - 0.5.
# Все EV — в bb относительно состояния «анте уплачено, блайнды не поставлены»:
# фолд SB = -0.5, проход шова = +1.0 + dead. Анте — константа в обеих ветках и
# на равновесную стратегию не влияет.
_SB_BLIND = 0.5
_BB_BLIND = 1.0

# Потолок мёртвых денег: 5bb сверх блайндов — это заведомо за пределами любой
# турнирной структуры (при анте 0.15bb с девяти игроков выходит 1.35bb). Гвард
# ловит перепутанные единицы (фишки вместо bb), а не защищает от структуры.
MAX_DEAD_EXTRA_BB = 5.0

# Fictitious play: чередование лучших ответов на СРЕДНЮЮ стратегию оппонента с
# усреднением по итерациям (для зеро-сум игры это сходится).
#
# Критериев остановки два, и они меряют разное.
#
# 1. Агрегатная эксплуатируемость — сумма выигрышей обоих игроков от отклонения,
#    взвешенная априорной вероятностью класса. Это определение ε-равновесия и
#    правильная мера сходимости игры в целом. Считается внутри цикла даром: шаг
#    лучшего ответа и так перебирает EV альтернатив.
#
#    Изменения значения игры |ΔEV| для этой роли НЕ хватает, и это измерено, а не
#    предположено: при 1/t-усреднении значение меняется на O(1/t) само по себе, так
#    что порог |ΔEV| < 0.001bb достигается уже на 3-19-й итерации, когда
#    эксплуатируемость ещё 0.01-0.04bb, а диапазоны гуляют на 5-7% комбо. Порог по
#    |ΔEV| оставлен как дополнительное (более слабое) условие.
#
# 2. Пер-хендовый регрет — МАКСИМУМ по классам, без взвешивания априорной
#    вероятностью. Он нужен потому, что агрегат сам по себе ничего не обещает
#    отдельному классу: при априорной вероятности ~0.009 класс может проигрывать
#    ~0.1bb и добавить к сумме всего ~1e-3. А читают этот модуль поклассно.
#
#    Что именно гарантирует порог 0.01bb. Регрет класса = (вес, стоящий на худшем
#    действии) * (отрыв действий). Отсюда: класс, вернувшийся с весом w на худшем
#    действии, имеет отрыв не больше 0.01/w. Значит вес 0.4 возможен только при
#    отрыве <= 0.025bb, то есть при настоящем безразличии, а рука с отрывом 0.1bb
#    обязана вернуться с весом не хуже 0.9 или не лучше 0.1. Остаток усреднения
#    больше не может выглядеть как смешанная стратегия на явно плюсовой руке.
#
#    Замер стоимости: порог 0.01bb достигается на 110 / 249 / 297 итерациях при
#    5 / 10 / 15bb — РАНЬШЕ, чем срабатывает агрегатный (204 / 398 / 559), поэтому
#    условие бесплатно и на практике не связывает. Фактически достигаемый регрет
#    на точке остановки: 0.0051 / 0.0063 / 0.0051bb (значения снимаются самим
#    решателем; 15bb не самая слабая из трёх, а самая тугая — она же дольше всех
#    крутится). Ужесточение до 0.001bb стоило бы 1083 / 2705 / 2853 итераций
#    (единицы секунд на глубину) и ширины почти не двигает: на 10 и 15bb сдвиг не
#    больше 0.03 п.п., на 5bb пуш уезжает с 72.07% на 71.49%. Если понадобится —
#    менять здесь одну константу.
#
# 1e-3 bb агрегатной эксплуатируемости — с запасом ниже шкалы, на которой решения
# различимы. Про пер-хендовую границу так сказать было НЕЛЬЗЯ, и прежний
# комментарий, утверждавший это про обе, был неверен: десятые доли bb — это как раз
# та шкала. Достигнутая граница отдаётся наружу (`nash_hu_regret_bb`), чтобы
# вызывающая сторона могла отказаться от строгого вердикта.
_FP_EXPLOITABILITY_BB = 1e-3
_FP_MAX_HAND_REGRET_BB = 1e-2
_FP_VALUE_TOLERANCE_BB = 1e-3
_FP_MAX_ITERATIONS = 20_000

# Минимум шагов усреднения — не запас "на всякий случай", а граница на вес
# произвольного стартового убеждения: первый лучший ответ считается против
# равномерного 0.5 и входит в среднее с весом 1/N. На мелких стеках игра почти
# тривиальна, и порог по эксплуатируемости достигается за 10 шагов — тогда в
# диапазоне остаются мусорные веса по 0.1 у рук, которые пушатся только против
# этого стартового убеждения. При 200 шагах остаток ограничен 0.005.
_FP_MIN_AVERAGING_STEPS = 200

# Версия формата кэша равновесий: меняется, когда меняется смысл записанного.
_NASH_CACHE_VERSION = 1

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


def _solve_nash_hu(
    eff_bb: float, dead_extra_bb: float
) -> tuple[dict[str, float], dict[str, float], int, float, float]:
    """Fictitious play для хедз-ап пуш-фолда с мёртвыми деньгами `dead_extra_bb`.

    Возвращает (диапазон шова, диапазон колла, число шагов усреднения, EV SB в bb,
    достигнутый максимальный пер-хендовый регрет в bb).

    Не сошлось за `_FP_MAX_ITERATIONS` — это `RuntimeError`, а не «последнее
    среднее молча». Правдоподобный результат вместо отказа здесь опаснее падения:
    вызывающая сторона примет его за равновесие.
    """
    classes = all_classes()
    _, equity = _eq169()
    conditional = _conditional_class_probs()
    priors = _class_priors()
    size = len(classes)

    # Проход шова: SB забирает блайнд BB и все мёртвые деньги.
    through = _BB_BLIND + dead_extra_bb
    # Банк вскрытия: по eff_bb с каждого плюс мёртвые деньги.
    showdown_pot = 2.0 * eff_bb + dead_extra_bb

    # EV шова для SB с рукой h против средней стратегии BB:
    #   EV(h) = Σ_c P(c|h) * [ call_w(c)*(eq*банк - eff) + (1-call_w(c))*through ]
    #         = through + Σ_c push_term[h][c] * call_w(c),   т.к. Σ_c P(c|h) = 1.
    push_term = [
        tuple(
            conditional[h][c] * (equity[h][c] * showdown_pot - eff_bb - through)
            for c in range(size)
        )
        for h in range(size)
    ]
    # Разница "колл минус фолд" для BB с рукой b против шова руки h:
    #   eq*банк - eff + 1 (фолд стоит BB его блайнда).
    call_term = [
        tuple(
            conditional[b][h] * (equity[b][h] * showdown_pot - eff_bb + _BB_BLIND)
            for h in range(size)
        )
        for b in range(size)
    ]

    avg_push = [0.5] * size  # стартовое убеждение, в среднее не входит: на шаге 1
    avg_call = [0.5] * size  # оно целиком заменяется первым лучшим ответом
    averaging_steps = 0
    previous_value = None
    value = 0.0
    hand_regret = 0.0
    gain_sb = gain_bb = float("inf")
    converged = False

    for _step in range(1, _FP_MAX_ITERATIONS + 1):
        shove_ev = [through + sum(map(mul, push_term[h], avg_call)) for h in range(size)]
        call_minus_fold = [sum(map(mul, call_term[b], avg_push)) for b in range(size)]

        value = sum(
            priors[h] * (avg_push[h] * shove_ev[h] + (1.0 - avg_push[h]) * -_SB_BLIND)
            for h in range(size)
        )
        # Регрет класса: сколько даёт переход на лучшее чистое действие ИМЕННО этой
        # рукой. Агрегат взвешивает их априорной вероятностью, пер-хендовая граница
        # берёт максимум — редкий класс не должен прятаться за своей редкостью.
        regret_sb = [
            max(shove_ev[h], -_SB_BLIND)
            - (avg_push[h] * shove_ev[h] + (1.0 - avg_push[h]) * -_SB_BLIND)
            for h in range(size)
        ]
        regret_bb = [
            max(call_minus_fold[b], 0.0) - avg_call[b] * call_minus_fold[b]
            for b in range(size)
        ]
        gain_sb = sum(priors[h] * regret_sb[h] for h in range(size))
        gain_bb = sum(priors[b] * regret_bb[b] for b in range(size))
        hand_regret = max(max(regret_sb), max(regret_bb))

        if (
            averaging_steps >= _FP_MIN_AVERAGING_STEPS
            and gain_sb + gain_bb <= _FP_EXPLOITABILITY_BB
            and hand_regret <= _FP_MAX_HAND_REGRET_BB
            and previous_value is not None
            and abs(value - previous_value) < _FP_VALUE_TOLERANCE_BB
        ):
            converged = True
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

    if not converged:
        raise RuntimeError(
            f"fictitious play не сошёлся на {eff_bb} bb (мёртвых {dead_extra_bb} bb) "
            f"за {_FP_MAX_ITERATIONS} итераций "
            f"(эксплуатируемость {gain_sb + gain_bb:.6f}, пер-хендовый регрет "
            f"{hand_regret:.6f}) — возвращать последнее среднее как равновесие нельзя"
        )

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
    return push, call, averaging_steps, value, hand_regret


@cache
def _nash_fingerprint(dead_extra_bb: float) -> str:
    """Отпечаток входных данных равновесия: таблица эквити, пороги, мёртвые деньги.

    Без него локальный кэш переживает и перегенерацию `eq169.json`, и смену
    порогов, и молча отдаёт равновесие, посчитанное при других входах. Имя файла
    от этого не защищает: в нём только глубина и мёртвые деньги.

    Мёртвые деньги входят в отпечаток наравне с порогами, и это не
    перестраховка: равновесие при другом анте — это равновесие ДРУГОЙ игры.
    Файлы, записанные до появления параметра, отпечатку не соответствуют и будут
    пересчитаны, а не прочитаны.
    """
    digest = sha256(_EQ169_PATH.read_bytes()).hexdigest()[:16]
    return (
        f"v{_NASH_CACHE_VERSION}"
        f"-eq{digest}"
        f"-x{_FP_EXPLOITABILITY_BB:g}"
        f"-r{_FP_MAX_HAND_REGRET_BB:g}"
        f"-d{_FP_VALUE_TOLERANCE_BB:g}"
        f"-m{_FP_MIN_AVERAGING_STEPS}"
        f"-D{dead_extra_bb:g}"
    )


def _read_nash_cache(path: Path, eff_bb: float, dead_extra_bb: float) -> tuple[Range, Range, float] | None:
    """Кэш — ускорение, а не источник истины: любое несовпадение ведёт к пересчёту.

    Проверяется и отпечаток входов, и записанные внутрь глубина с мёртвыми
    деньгами: имя файла округляет обе величины до сотых, поэтому само по себе
    оно ничего не гарантирует.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != _nash_fingerprint(dead_extra_bb):
            return None
        if abs(float(payload["eff_bb"]) - eff_bb) > 1e-9:
            return None
        if abs(float(payload.get("dead_extra_bb", 0.0)) - dead_extra_bb) > 1e-9:
            return None
        return (
            Range(weights=payload["push"]),
            Range(weights=payload["call"]),
            float(payload["hand_regret_bb"]),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_nash_cache(
    path: Path,
    eff_bb: float,
    dead_extra_bb: float,
    solution: tuple[dict[str, float], dict[str, float], int, float, float],
) -> None:
    push, call, iterations, value, hand_regret = solution
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "fingerprint": _nash_fingerprint(dead_extra_bb),
                    "eff_bb": eff_bb,
                    "dead_extra_bb": dead_extra_bb,
                    "fp_iterations": iterations,
                    "sb_ev_bb": round(value, 6),
                    "hand_regret_bb": hand_regret,
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


def _nash_solution(
    eff_bb: float, dead_extra_bb: float, cache_dir: Path | None
) -> tuple[Range, Range, float]:
    if not _MIN_EFF_BB < eff_bb <= _MAX_EFF_BB:
        raise ValueError(
            f"эффективный стек должен лежать в ({_MIN_EFF_BB}, {_MAX_EFF_BB}] bb: "
            f"на {eff_bb} bb шов SB не покрывает даже блайнд BB либо стек слишком глубок "
            f"для пуш-фолда"
        )
    if not 0.0 <= dead_extra_bb <= MAX_DEAD_EXTRA_BB:
        raise ValueError(
            f"мёртвые деньги сверх блайндов должны лежать в [0, {MAX_DEAD_EXTRA_BB}] bb, "
            f"получено {dead_extra_bb} — похоже на фишки вместо bb"
        )

    directory = Path(cache_dir) if cache_dir is not None else _DATA_DIR
    path = directory / f"nash_hu_{eff_bb:.2f}_d{dead_extra_bb:.2f}.json"

    cached = _read_nash_cache(path, eff_bb, dead_extra_bb)
    if cached is not None:
        return cached

    solution = _solve_nash_hu(eff_bb, dead_extra_bb)
    _write_nash_cache(path, eff_bb, dead_extra_bb, solution)
    return Range(weights=solution[0]), Range(weights=solution[1]), solution[4]


def nash_hu(
    eff_bb: float, *, dead_extra_bb: float = 0.0, cache_dir: Path | None = None
) -> tuple[Range, Range]:
    """Равновесие хедз-ап пуш-фолда на глубине `eff_bb`: (диапазон шова SB, колла BB).

    `eff_bb` — эффективный стек ПОСЛЕ уплаты анте; `dead_extra_bb` — мёртвые
    деньги сверх двух блайндов, то есть сумма анте всего стола (ноль для игры без
    анте). Разделение обязательно: иначе анте либо теряется, либо считается
    дважды — подробности в комментарии к `_SB_BLIND`.

    Игра: SB ставит 0.5, BB ставит 1.0, SB шовит на весь эффективный стек или
    фолдит, BB коллирует или фолдит. Решается fictitious play по предвычисленной
    таблице эквити (`class_equity`) — Монте-Карло внутри итераций не запускается,
    иначе сходимости не дождаться. Блокеры учтены: распределение руки оппонента
    условно по руке игрока.

    Веса в возвращённых диапазонах читаются поклассно, поэтому у них есть
    пер-хендовая гарантия, а не только средняя по игре: для каждого класса
    отклонение от его лучшего чистого действия стоит не больше
    `_FP_MAX_HAND_REGRET_BB`. Достигнутое значение отдаёт `nash_hu_regret_bb`.

    Результат кэшируется в `cache_dir` (по умолчанию — `data/` рядом с модулем).
    Кэш только ускоряет: при пустом, испорченном или посчитанном на других входах
    кэше считается заново и даёт тот же ответ — fictitious play детерминирован.
    """
    push, call, _ = _nash_solution(eff_bb, dead_extra_bb, cache_dir)
    return push, call


def nash_hu_regret_bb(
    eff_bb: float, *, dead_extra_bb: float = 0.0, cache_dir: Path | None = None
) -> float:
    """Достигнутый максимальный пер-хендовый регрет равновесия на этой глубине, в bb.

    Верхняя граница на то, сколько теряет класс, сыгранный по возвращённому весу
    вместо своего лучшего чистого действия. Вызывающая сторона может по ней
    решить, имеет ли право ставить строгий вердикт на пограничную руку.
    """
    return _nash_solution(eff_bb, dead_extra_bb, cache_dir)[2]


# --- Bracket-модели колл-диапазона (вход bracket-теста зоны, задача 12) ---------
#
# Это не оценка «как коллируют на самом деле», а два намеренно грубых конца
# вилки: если вердикт одинаков и против премиум-онли, и против top-40%, он не
# зависит от угаданного диапазона.
#
# Эти два диапазона решают, помечен вердикт как «строго» или как «предполагая»,
# поэтому ни одно число в них не выбирается на глаз, и — важнее — критерий
# упорядочения должен мерить ту величину, ради которой вилка существует.
# Широкий конец обязан ограничивать правдоподобное поведение оппонента СВЕРХУ:
# если он слишком узок, bracket-тест объявит вердикт устойчивым там, где тот на
# самом деле зависит от догадки, и продукт поставит «строго» на вывод, который
# этого не заслужил. Завышенная уверенность в себе — ровно тот отказ, который
# правило зоны и должно предотвращать.
#
# Отсюда критерий: коллер отвечает на ШОВ, а не играет против случайной руки,
# поэтому классы ранжируются по эквити против равновесного диапазона шова на той
# же глубине. Разница не косметическая — она меняет состав. Против случайной руки
# слабый туз сильнее двойки; против диапазона шова наоборот, потому что шов почти
# всегда содержит старшую карту, и пара впереди. Одномастные коннекторы остаются
# вне вилки по обоим критериям, и это правильно: колл шова — это вскрытие, а
# постфлоп-ценность, ради которой их держат, там не существует.
#
# Зацикливания нет: `nash_hu` брекеты не использует.

_TIGHT_CLASSES: tuple[str, ...] = ("AA", "KK", "QQ", "JJ", "AKs", "AKo")

_WIDE_TARGET_FRACTION = 0.40

# Глубина для диапазона шова зажимается в это окно. Ниже 2bb равновесный колл и
# так «любые две карты» (проверено тестом на замкнутом решении), выше 25bb пуш-фолд
# перестаёт быть моделью спота — в обоих концах ранжирование уже не двигается, а
# вилке важно не падать на вырожденном входе.
_BRACKET_MIN_DEPTH_BB = 2.0
_BRACKET_MAX_DEPTH_BB = 25.0


def equity_vs_range_classes(hero_cls: str, rng: Range) -> float:
    """Эквити класса против взвешенного диапазона классов — по таблице, с блокерами.

    Усреднение идёт не по номинальным весам диапазона, а по условному распределению
    руки оппонента после снятия карт героя: держа AA, герой сам делает AA у
    оппонента вшестеро реже, и в узких диапазонах эта поправка велика.
    """
    index = _class_index()
    if hero_cls not in index:
        raise ValueError(f"неизвестный класс руки: {hero_cls!r}")
    conditional = _conditional_class_probs()[index[hero_cls]]
    row = _eq169()[1][index[hero_cls]]

    weighted = 0.0
    mass = 0.0
    for cls, weight in rng.weights.items():
        if weight <= 0.0:
            continue
        if cls not in index:
            raise ValueError(f"неизвестный класс руки: {cls!r}")
        share = conditional[index[cls]] * weight
        weighted += share * row[index[cls]]
        mass += share
    if mass <= 0.0:
        raise ValueError(
            "диапазон пуст или полностью заблокирован картами героя — эквити не от чего считать"
        )
    return weighted / mass


def classes_by_equity_against(rng: Range) -> tuple[str, ...]:
    """169 классов по убыванию эквити против заданного диапазона.

    Имя класса — вторичный ключ сортировки, чтобы порядок был воспроизводим,
    когда эквити двух классов совпадает после округления таблицы.
    """
    strength = {cls: equity_vs_range_classes(cls, rng) for cls in all_classes()}
    return tuple(sorted(strength, key=lambda cls: (-strength[cls], cls)))


@cache
def _classes_of_width(depth_key: float, dead_key: float, fraction: float) -> tuple[str, ...]:
    """Верхние классы против равновесного шова, набирающие долю комбо ближе всего к `fraction`.

    Граница берётся по КОМБО, а не по числу классов (offsuit-класс весит 12 комбо,
    suited — 4), и выбирается тот префикс, который ближе к цели: заявленная ширина
    обязана быть проверяемой, а не приблизительной.

    Порядок классов считается против равновесного шова ТОЙ ЖЕ игры, включая
    мёртвые деньги: с анте шов шире, а против широкого шова порядок рук другой
    (пара против слабого туза переворачивается). Брать здесь безантевое
    равновесие значило бы ранжировать по игре, в которую никто не играет.
    """
    push_range, _ = nash_hu(depth_key, dead_extra_bb=dead_key)
    ordered = classes_by_equity_against(push_range)

    target = fraction * _TOTAL_COMBOS
    cumulative = 0
    best_prefix, best_gap = 0, target
    for position, cls in enumerate(ordered, start=1):
        cumulative += len(combos_of_class(cls))
        if abs(cumulative - target) < best_gap:
            best_gap, best_prefix = abs(cumulative - target), position
    return ordered[:best_prefix]


def range_of_width(
    depth_bb: float, fraction: float, *, dead_extra_bb: float = 0.0
) -> Range:
    """Диапазон заданной ширины: верхние `fraction` комбо по силе на этой глубине.

    Семейство, по которому задача 12 опрашивает ВЕСЬ интервал правдоподобных
    диапазонов, а не два его конца. Опрос концов неверен по существу: EV не
    монотонна по ширине. На 10.25bb с анте фикстуры шов K5o даёт +1.95bb против
    3% колла и +0.06bb против 51%, а между ними, против 40%, уходит в минус
    (-0.05bb) — оба конца сказали бы «шов», хотя внутри интервала вердикт
    переворачивается.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"доля комбо должна лежать в (0, 1], получено {fraction}")
    classes = _classes_of_width(
        _bracket_depth_key(depth_bb), round(dead_extra_bb, 4), round(fraction, 4)
    )
    return Range(weights=dict.fromkeys(classes, 1.0))


def _bracket_depth_key(depth_bb: float) -> float:
    """Глубина, округлённая до сотых и зажатая в окно, где пуш-фолд осмыслен."""
    clamped = min(max(depth_bb, _BRACKET_MIN_DEPTH_BB), _BRACKET_MAX_DEPTH_BB)
    return round(clamped, 2)


def _bracket_tight(depth_bb: float, dead_extra_bb: float = 0.0) -> Range:
    """Узкий конец вилки: только премиум (JJ+, AK).

    От глубины не зависит: это не выведенная граница, а названный в спецификации
    набор «только премиум», и его задача — быть заведомо уже правдоподобного.
    """
    return Range(weights=dict.fromkeys(_TIGHT_CLASSES, 1.0))


def _bracket_wide(depth_bb: float, dead_extra_bb: float = 0.0) -> Range:
    """Широкий конец вилки: top-40% по эквити против равновесного шова на этой глубине."""
    return range_of_width(depth_bb, _WIDE_TARGET_FRACTION, dead_extra_bb=dead_extra_bb)


BRACKET_TIGHT: Callable[..., Range] = _bracket_tight
BRACKET_WIDE: Callable[..., Range] = _bracket_wide
