"""Поклассная поправка трёхстороннего эквити решателя пуш-фолда (`multiway.py`).

Решатель с оверколлами считает долю банка героя при вскрытии втроём не
Монте-Карло, а из двух парных эквити x, y (герой против каждого диапазона) по
модели Брэдли–Терри: `1 / (1 + (1−x)/x + (1−y)/y)`. Модель ошибается до 4.5 п.п.,
и ошибка устойчива по классу героя: тузы переоцениваются (оба оппонента держат
тузов, и это коррелирует), связанные одномастные — недооцениваются (их дро
во втроём стоит дороже). Поэтому поправка одна на класс: среднее «MC − модель»
по калибровочным парам диапазонов ниже.

Замер при выборе конструкции (спека 2026-10-01-pushfold-overcalls-dead-cards,
подзадача 1): на парах, НЕ входивших в калибровку, остаточная ошибка с
поправкой ≤ 1.81 п.п. против 4.36 без неё. Сверка живёт в тестах
(`test_three_way_equity_matches_monte_carlo_on_held_out_ranges`).

Запускается вручную, результат коммитится как данные, как и eq169.json.
Значения по умолчанию — ровно те, которыми получен закоммиченный файл.

    uv run python scripts/build_eq3_correction.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from harness.analysis.charts.notation import parse_range
from harness.analysis.tools.equity import equity_vs_ranges
from harness.analysis.tools.multiway import bradley_terry_three_way
from harness.analysis.tools.pushfold import equity_vs_range_classes, representative_combo
from harness.contracts import all_classes

DEFAULT_ITERATIONS = 20_000
DEFAULT_SEED = 20_261_001
DEFAULT_OUT = (
    Path(__file__).resolve().parents[1] / "src/harness/analysis/tools/data/eq3_correction.json"
)

# Калибровочные диапазоны — типы, которые встречаются во вскрытии втроём:
# узкие холодные коллы, ещё более узкие оверколлы, широкие шовы.
CALIBRATION_RANGES: dict[str, str] = {
    "cold_8": "66+, ATs+, KQs, AJo+",
    "cold_16": "33+, A4s+, KTs+, QTs+, JTs:0.55, A9o+, KJo+",
    "over_4": "TT+, AQs+, AKo",
    "over_6": "88+, AJs+, AQo+",
    "push_17": "33+, A5s+, K9s+, Q9s+, J9s+, T9s, ATo+, KQo",
    "push_43": (
        "22+, A2s+, K2s+, Q5s+, J7s+, T7s+, 96s+, 86s+, 75s+, 65s, 54s, "
        "A2o+, K5o+, Q9o+, J9o+, T9o"
    ),
}
CALIBRATION_PAIRS: tuple[tuple[str, str], ...] = (
    ("cold_8", "over_4"),
    ("cold_16", "over_6"),
    ("push_17", "over_4"),
    ("push_17", "cold_16"),
    ("push_43", "over_6"),
    ("cold_8", "cold_16"),
)


def class_correction(task: tuple[int, str, int, int]) -> tuple[str, float, float]:
    """Среднее и наибольшее по модулю «MC − модель» для одного класса героя."""
    index, hero_cls, iterations, seed = task
    ranges = {name: parse_range(text) for name, text in CALIBRATION_RANGES.items()}
    hero = representative_combo(hero_cls)
    residuals = []
    for pair_index, (a, b) in enumerate(CALIBRATION_PAIRS):
        model = float(
            bradley_terry_three_way(
                np.asarray(equity_vs_range_classes(hero_cls, ranges[a])),
                np.asarray(equity_vs_range_classes(hero_cls, ranges[b])),
            )
        )
        measured = equity_vs_ranges(
            hero,
            [ranges[a], ranges[b]],
            iterations=iterations,
            seed=seed * 1_000_003 + index * len(CALIBRATION_PAIRS) + pair_index,
        )
        residuals.append(measured - model)
    mean = sum(residuals) / len(residuals)
    return hero_cls, mean, max(abs(r - mean) for r in residuals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Поклассная поправка трёхстороннего эквити")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    classes = all_classes()
    tasks = [(i, cls, args.iterations, args.seed) for i, cls in enumerate(classes)]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        results = {cls: (mean, spread) for cls, mean, spread in pool.map(class_correction, tasks)}

    payload = {
        "model": "bradley_terry_three_way",
        "iterations": args.iterations,
        "seed": args.seed,
        "calibration_ranges": CALIBRATION_RANGES,
        "calibration_pairs": [list(p) for p in CALIBRATION_PAIRS],
        "max_residual_after_correction": round(max(s for _, s in results.values()), 6),
        "classes": classes,
        "correction": [round(results[cls][0], 6) for cls in classes],
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        f"{len(classes)} классов за {time.monotonic() - started:.0f} c; наибольший остаток "
        f"на калибровке {100 * payload['max_residual_after_correction']:.2f} п.п. → {args.out}"
    )


if __name__ == "__main__":
    main()
