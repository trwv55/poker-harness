"""Изложение: покерное содержание разбора — реплей руки, матрица диапазонов, текст.

Три выхода, из них два — чистый код (`hand_replay`, `range_render`) и один
опирается на модель (`verdict_text`, `tournament_text`). Как это выглядит в
Телеграме, решает `presentation`; пакет знает только про покер.

Публичный API реэкспортируется здесь по образцу `harness.contracts` и
`harness.presentation`: вызывающий (`worker`, `presentation`, eval-раннер)
импортирует из `harness.explanation`, а не из отдельных модулей.
"""

from __future__ import annotations

from harness.explanation.faithfulness import (
    error_words_in,
    has_assumption_words,
    names_the_better_line,
    near_zero_reproach,
    numbers_in,
    unsupported_numbers,
    verdict_label_for,
)
from harness.explanation.hand_replay import HandReplay, ReplaySpan, hand_replay
from harness.explanation.range_render import range_svg, render_range_png
from harness.explanation.tournament_text import (
    TournamentLLM,
    tournament_digest,
    tournament_draft,
    tournament_text,
)
from harness.explanation.verdict_text import (
    Digest,
    PromptUnavailable,
    UnfaithfulText,
    VerdictDraft,
    VerdictLLM,
    verdict_digest,
    verdict_draft,
    verdict_text,
)

__all__ = [
    "Digest",
    "HandReplay",
    "PromptUnavailable",
    "ReplaySpan",
    "TournamentLLM",
    "UnfaithfulText",
    "VerdictDraft",
    "VerdictLLM",
    "error_words_in",
    "hand_replay",
    "has_assumption_words",
    "names_the_better_line",
    "near_zero_reproach",
    "numbers_in",
    "range_svg",
    "render_range_png",
    "tournament_digest",
    "tournament_draft",
    "tournament_text",
    "unsupported_numbers",
    "verdict_digest",
    "verdict_draft",
    "verdict_label_for",
    "verdict_text",
]
