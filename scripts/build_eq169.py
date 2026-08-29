"""Генерация таблицы префлоп-эквити 169x169 (класс против класса), хедз-ап.

Запускается вручную ОДИН раз, результат коммитится как данные. Ни импорт пакета,
ни тесты этот скрипт не зовут: 14 196 внедиагональных пар классов по 50 тыс.
розыгрышей — это минуты процессорного времени, а не миллисекунды. Замер на
8 ядрах (M-серия, Python 3.12): 280 c, ~50 пар/с, файл 194 КБ. Значения по
умолчанию — ровно те, которыми получен закоммиченный eq169.json, поэтому
повторный запуск без аргументов воспроизводит его.

    uv run python scripts/build_eq169.py                 # полная таблица
    uv run python scripts/build_eq169.py --limit 40      # смоук на 40 парах

Что именно считается. Эквити пары классов — среднее по всем НЕконфликтующим
парам комбо этих классов (для одинаковых классов комбо тоже не пересекаются):
каждой такой паре комбо достаётся равная доля итераций, и итог — невзвешенное
среднее их долей банка. Диагональ не сэмплируется: класс против самого себя даёт
ровно 0.5 по симметрии (распределение пары комбо обменно-симметрично), и MC тут
добавил бы только шум. Нижний треугольник получается как 1 − верхний: в хедз-апе
доли банка двух игроков в сумме дают единицу тождественно, поэтому считать обе
половины значило бы вносить в симметрию численный шум.

Воспроизводимость. Сид каждой пары выводится из её индексов, а не из порядка
обхода, поэтому результат не зависит ни от числа процессов, ни от того, в каком
порядке пул раздал задачи.

Сверка данных живёт в тестах (`test_class_equity_matches_independent_recomputation`):
случайные клетки пересчитываются через `equity_vs_range` другим сидом.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import eval7

from harness.analysis.tools.equity import combos_of_class
from harness.contracts import all_classes

DEFAULT_ITERATIONS = 50_000
DEFAULT_SEED = 20_260_830
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "src/harness/analysis/tools/data/eq169.json"

_RANKS = "AKQJT98765432"
_SUITS = "shdc"
_DECK: tuple[str, ...] = tuple(rank + suit for rank in _RANKS for suit in _SUITS)
_CARD: dict[str, eval7.Card] = {c: eval7.Card(c) for c in _DECK}

_CLASSES: list[str] = all_classes()
_COMBOS: list[list[tuple[str, str]]] = [combos_of_class(c) for c in _CLASSES]


def _pair_seed(base_seed: int, i: int, j: int) -> int:
    """Сид пары зависит только от её индексов — не от порядка обхода пула."""
    return base_seed * 1_000_003 + i * len(_CLASSES) + j


def pair_equity(task: tuple[int, int, int, int]) -> tuple[int, int, float]:
    """Эквити класса i против класса j: среднее по неконфликтующим парам комбо."""
    i, j, iterations, base_seed = task
    combo_pairs = [
        (a, b) for a in _COMBOS[i] for b in _COMBOS[j] if a[0] not in b and a[1] not in b
    ]
    if not combo_pairs:  # не бывает для 169 классов, но молча делить на ноль нельзя
        raise ValueError(f"нет неконфликтующих пар комбо для {_CLASSES[i]} и {_CLASSES[j]}")

    rand = random.Random(_pair_seed(base_seed, i, j))
    sample = rand.sample
    evaluate = eval7.evaluate

    per_pair, extra = divmod(iterations, len(combo_pairs))
    shares: list[float] = []
    for index, (hero_cards, villain_cards) in enumerate(combo_pairs):
        draws = per_pair + (1 if index < extra else 0)
        if draws == 0:
            continue
        hero = [_CARD[hero_cards[0]], _CARD[hero_cards[1]]]
        villain = [_CARD[villain_cards[0]], _CARD[villain_cards[1]]]
        deck = [_CARD[c] for c in _DECK if c not in hero_cards and c not in villain_cards]

        wins = 0.0
        for _ in range(draws):
            board = sample(deck, 5)
            hero_score = evaluate(hero + board)
            villain_score = evaluate(villain + board)
            if hero_score > villain_score:
                wins += 1.0
            elif hero_score == villain_score:
                wins += 0.5
        shares.append(wins / draws)

    return i, j, sum(shares) / len(shares)


def build(iterations: int, seed: int, workers: int, limit: int | None) -> list[list[float]]:
    size = len(_CLASSES)
    table = [[0.5] * size for _ in range(size)]  # диагональ = 0.5 по симметрии

    tasks = [(i, j, iterations, seed) for i in range(size) for j in range(i + 1, size)]
    if limit is not None:
        tasks = tasks[:limit]

    started = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, j, equity in pool.map(pair_equity, tasks, chunksize=32):
            value = round(equity, 4)
            table[i][j] = value
            table[j][i] = round(1.0 - value, 4)
            done += 1
            if done % 500 == 0 or done == len(tasks):
                elapsed = time.perf_counter() - started
                rate = done / elapsed
                left = (len(tasks) - done) / rate
                print(
                    f"  {done}/{len(tasks)} пар, {elapsed:.0f} c, "
                    f"{rate:.0f} пар/с, осталось ~{left:.0f} c",
                    flush=True,
                )
    return table


def dump(table: list[list[float]], iterations: int, seed: int, out: Path) -> None:
    """Одна строка JSON на класс — файл читаемый и грепается, а diff осмысленный."""
    rows = ",\n    ".join(json.dumps(row, separators=(",", ":")) for row in table)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "{\n"
        f'  "generated_by": "scripts/build_eq169.py",\n'
        f'  "iterations_per_pair": {iterations},\n'
        f'  "seed": {seed},\n'
        f'  "classes": {json.dumps(_CLASSES)},\n'
        '  "equity": [\n    ' + rows + "\n  ]\n}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--limit", type=int, default=None, help="считать только первые N пар")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    print(
        f"пар: {len(_CLASSES) * (len(_CLASSES) - 1) // 2}, итераций на пару: {args.iterations}, "
        f"процессов: {args.workers}, сид: {args.seed}",
        flush=True,
    )
    started = time.perf_counter()
    table = build(args.iterations, args.seed, args.workers, args.limit)
    elapsed = time.perf_counter() - started
    dump(table, args.iterations, args.seed, args.out)
    print(f"готово за {elapsed:.1f} c -> {args.out} ({args.out.stat().st_size / 1024:.0f} КБ)")


if __name__ == "__main__":
    main()
