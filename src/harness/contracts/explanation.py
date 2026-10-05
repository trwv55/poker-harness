"""Выход изложения: то, что модель НАПИСАЛА словами поверх уже посчитанных чисел.

Контракт держится здесь, а не в `explanation/`, по той же причине, что и
`ScanSummary`: его читают вне пакета — `presentation` (собрать сообщение) и
`memory` (колонка `analyses.verdict_text`), и импорт типа из `explanation`
тянул бы за собой промпты и клиента модели туда, где ни того ни другого не
нужно.

**Метка вердикта — структурная, а не выведенная из текста.** `PointText.
verdict_label` ставит код по `ev_diff_bb` ядра (`explanation.faithfulness.
verdict_label_for`), модель её не выбирает. Это значит ровно одно: метка не
следует за тоном текста и не портится вместе с ним. Само по себе это НЕ
гарантирует, что текст не похвалит точку, которую ядро назвало расхождением, —
за это отвечают промпт и проверка направления (`faithfulness.
names_the_better_line`, `evals/verdict/checks.py`).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from harness.contracts.model_output import ModelOutput

VerdictLabel = Literal["ok", "mistake", "marginal"]


class PointText(BaseModel):
    """Пояснение к одной точке решения: чей это спот, какова метка, что сказано.

    `dp_index` — тот же индекс, что у `PointVerdict.dp_index`: текст обязан
    привязываться к точке расчёта, иначе его нечем сверить.
    """

    dp_index: int
    verdict_label: VerdictLabel
    text: str


class VerdictTextOut(BaseModel):
    """Текст разбора одной раздачи: по абзацу на точку плюс общий вывод."""

    points: list[PointText]
    summary: str


class TournamentTextOut(ModelOutput, BaseModel):
    """Текст отчёта по турниру — абзацами, в порядке следования в сообщении.

    Абзацами, а не одной строкой: `presentation` расставляет между ними пустые
    строки сама, и склеенный текст пришлось бы разбивать обратно по эвристике.
    """

    paragraphs: list[str]
