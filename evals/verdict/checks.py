"""Этаж 3 (EVALS.md): верность изложения вердикта — проверки над ответом модели.

Три из плана (числа, метка, слова допущения) плюс четвёртая, добавленная после
первого живого прогона: решение игрока не названо ошибкой.

**Это не тесты.** Тесты проверяют код и гоняются на каждом коммите бесплатно;
этот файл проверяет МОДЕЛЬ, стоит денег и запускается руками — `uv run python -m
harness.platform.eval_runner verdict`. Смешивать нельзя: зелёный `pytest` ничего
не говорит о том, как отвечает `LLM_VERDICT_MODEL`, а провалившийся eval не
означает поломки кода.

**Прогонять обязательно** при смене `LLM_VERDICT_MODEL` или промпта
(`src/harness/explanation/prompts/verdict.md`) — это единственное место, где
видно, что новая модель начала выдумывать числа или молчать о допущении.

Сами предикаты живут в `harness.explanation.faithfulness` и используются ещё и в
проде: eval и продакшен обязаны понимать верность ОДИНАКОВО, иначе прогон меряет
не то, что защищает игрока.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.contracts import AnalysisResult, Zone
from harness.explanation.faithfulness import (
    has_assumption_words,
    unsupported_numbers,
    verdict_label_for,
)
from harness.explanation.verdict_text import VerdictDraft

__all__ = ["CheckResult", "run_checks"]


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Итог одной проверки одного кейса: имя, вердикт и что именно не сошлось."""

    name: str
    passed: bool
    detail: str = ""


def _check_numbers(
    draft: VerdictDraft, allowed: frozenset[float]
) -> CheckResult:
    """1. Все числа текста существуют в выходе ядра (EVALS этаж 3, пункт 1)."""
    invented: list[str] = []
    for point in draft.points:
        numbers = unsupported_numbers(point.text, allowed)
        if numbers:
            invented.append(f"точка {point.dp_index}: {numbers}")
    numbers = unsupported_numbers(draft.summary, allowed)
    if numbers:
        invented.append(f"вывод: {numbers}")
    return CheckResult("числа из расчёта", not invented, "; ".join(invented))


def _check_labels(draft: VerdictDraft, res: AnalysisResult) -> CheckResult:
    """2. Метка каждой точки совпадает с меткой ядра по порогам плана.

    Что эта проверка МОЖЕТ поймать: расхождение порогов между планом и кодом
    (`faithfulness.verdict_label_for`) и ответ модели про точки, которых разбор
    не судил. Чего она поймать НЕ может: похвалу в прозе там, где ядро видит
    расхождение, — метка модели не принадлежит вовсе, её ставит код, и потому
    противоречия «текст против метки» здесь не бывает по построению. Смысловое
    расхождение прозы и вердикта ловит человек, читающий вывод прогона.
    """
    core = {
        res.points[index].dp_index: verdict_label_for(res.points[index].ev_diff_bb)
        for index in res.ranked
    }
    problems = [
        f"точка {point.dp_index}: разбор её не судит"
        for point in draft.points
        if point.dp_index not in core
    ]
    expected_by_thresholds = {
        dp_index: ("ok" if ev >= -0.1 else "mistake" if ev < -0.5 else "marginal")
        for dp_index, ev in (
            (res.points[i].dp_index, res.points[i].ev_diff_bb) for i in res.ranked
        )
    }
    problems += [
        f"точка {dp_index}: код ставит {core[dp_index]}, пороги плана — {expected}"
        for dp_index, expected in expected_by_thresholds.items()
        if core[dp_index] != expected
    ]
    return CheckResult("метка совпадает с ядром", not problems, "; ".join(problems))


def _check_assumptions(draft: VerdictDraft, res: AnalysisResult) -> CheckResult:
    """3. Точка зоны «предполагая» названа словами допущения (EVALS, пункт 3)."""
    assuming = {
        res.points[index].dp_index
        for index in res.ranked
        if res.points[index].zone is Zone.ASSUMING
    }
    silent = [
        f"точка {point.dp_index}"
        for point in draft.points
        if point.dp_index in assuming and not has_assumption_words(point.text)
    ]
    return CheckResult("допущение названо словами", not silent, "; ".join(silent))


def _check_wording(draft: VerdictDraft) -> CheckResult:
    """4. Решение игрока не названо ошибкой (CLAUDE.md, дисциплина).

    Сверх трёх проверок плана, и намеренно ТОЛЬКО в eval, не в проде: слово —
    вопрос тона, а не верности, и отменять из-за него весь разбор было бы
    несоразмерно. Проверка появилась после первого живого прогона: модель
    написала игроку «единственное неверное решение» там, где расчёт судит
    решение против диапазона и об ошибке не говорит.
    """
    forbidden = ("ошибк", "неверн", "неправильн")
    hits = [
        f"точка {point.dp_index}: «{word}»"
        for point in draft.points
        for word in forbidden
        if word in point.text.lower()
    ]
    hits += [f"вывод: «{word}»" for word in forbidden if word in draft.summary.lower()]
    return CheckResult("решение не названо ошибкой", not hits, "; ".join(hits))


def run_checks(
    draft: VerdictDraft, res: AnalysisResult, allowed: frozenset[float]
) -> list[CheckResult]:
    """Все проверки одного кейса, в порядке важности."""
    return [
        _check_numbers(draft, allowed),
        _check_labels(draft, res),
        _check_assumptions(draft, res),
        _check_wording(draft),
    ]
