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

from harness.contracts import AnalysisResult, TournamentTextOut, Zone
from harness.explanation.faithfulness import (
    error_words_in,
    has_assumption_words,
    names_the_better_line,
    near_zero_reproach,
    unsupported_numbers,
    verdict_label_for,
)
from harness.explanation.verdict_text import VerdictDraft

__all__ = ["CheckResult", "run_checks", "run_story_checks"]


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


def _check_direction(draft: VerdictDraft, res: AnalysisResult) -> CheckResult:
    """2. Текст ведёт в ту же сторону, что и ядро (EVALS этаж 3, пункт 2).

    Прежняя версия этой проверки была тавтологией: она считала метку ядра дважды
    одной формулой и печатала «OK» всегда, ни разу не заглянув в текст модели
    (ревью, раздел B). Теперь проверяется то, ради чего пункт и написан:

    * точка, которую ядро НЕ считает сыгранной верно, обязана получить в тексте
      названную лучшую линию — «колл −1.2 bb» не имеет права стать рассказом
      вообще без упоминания того, что было лучше;
    * точка «около нуля» не имеет права получить упрёк: расчёт не спорит ни с
      одним из вариантов.

    Ответ про точку, которой разбор не судил, — тоже провал: сопоставлять текст
    с числами тогда не с чем.
    """
    by_dp = {res.points[i].dp_index: res.points[i] for i in res.ranked}
    problems: list[str] = []
    for draft_point in draft.points:
        point = by_dp.get(draft_point.dp_index)
        if point is None:
            problems.append(f"точка {draft_point.dp_index}: разбор её не судит")
            continue
        label = verdict_label_for(point.ev_diff_bb)
        if label != "ok" and not names_the_better_line(draft_point.text, point.best_action):
            problems.append(
                f"точка {draft_point.dp_index}: метка «{label}», а лучшая линия "
                f"«{point.best_action}» в тексте не названа"
            )
        if point.interval is not None and point.interval.near_zero:
            reproach = near_zero_reproach(draft_point.text)
            if reproach:
                problems.append(
                    f"точка {draft_point.dp_index}: «около нуля», а текст упрекает — {reproach}"
                )
    return CheckResult("текст ведёт туда же, куда ядро", not problems, "; ".join(problems))


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
    hits = [
        f"точка {point.dp_index}: «{word}»"
        for point in draft.points
        for word in error_words_in(point.text)
    ]
    hits += [f"вывод: «{word}»" for word in error_words_in(draft.summary)]
    return CheckResult("решение не названо ошибкой", not hits, "; ".join(hits))


def run_checks(
    draft: VerdictDraft, res: AnalysisResult, allowed: frozenset[float]
) -> list[CheckResult]:
    """Все проверки одного кейса, в порядке важности."""
    return [
        _check_numbers(draft, allowed),
        _check_direction(draft, res),
        _check_assumptions(draft, res),
        _check_wording(draft),
    ]


def run_story_checks(story: TournamentTextOut, allowed: frozenset[float]) -> list[CheckResult]:
    """Проверки рассказа по турниру — второго входа модели, у которого eval не было
    вовсе (ревью, раздел B).

    Здесь только числа и слова: точек решения у рассказа нет, а значит нет ни
    метки, ни лучшей линии. Слово «ошибка» в рассказе — не тон, а утверждение о
    НЕСУДИМЫХ раздачах, и прод его уже отвергает (`explanation.tournament_text`);
    проверка стоит рядом, чтобы прогон показывал такие случаи текстом, а не
    только фактом отказа.
    """
    invented = [
        f"абзац {index + 1}: {unsupported_numbers(paragraph, allowed)}"
        for index, paragraph in enumerate(story.paragraphs)
        if unsupported_numbers(paragraph, allowed)
    ]
    errors = [
        f"абзац {index + 1}: {error_words_in(paragraph)}"
        for index, paragraph in enumerate(story.paragraphs)
        if error_words_in(paragraph)
    ]
    return [
        CheckResult("числа из расчёта", not invented, "; ".join(invented)),
        CheckResult("ничего не названо ошибкой", not errors, "; ".join(errors)),
    ]
