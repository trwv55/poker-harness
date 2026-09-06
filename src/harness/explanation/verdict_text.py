"""Текст вердикта по одной раздаче — единственный второй вызов модели в системе.

Граница CLAUDE.md здесь проходит буквально: **модель излагает готовые числа и
не считает ничего**. Из этого три следствия, каждое реализовано, а не обещано:

1. **На вход идёт выжимка `AnalysisResult`, а не `EnrichedHand`.** Ход раздачи
   показывает `hand_replay` (код, ноль токенов), поэтому в промпте нет
   пересказа руки — класс ошибок «модель переврала ход руки» негде совершить, и
   вход с выходом короче. Функция физически не видит раздачу: в сигнатуре её нет.
2. **Метка вердикта отдаётся структурно.** Модель возвращает только тексты;
   `verdict_label` ставит код по `ev_diff_bb` (`faithfulness.verdict_label_for`).
   Модель не может «похвалить» точку, которую ядро назвало расхождением, потому
   что метку она не выбирает.
3. **Числа проверяются кодом.** Каждое число ответа обязано быть числом из
   выжимки (`digest.NumberBook`); иначе — `UnfaithfulText`, и текст игроку не
   уходит вовсе. То же для слов допущения у точек зоны «предполагая».

**Что происходит при отказе.** `UnfaithfulText` не роняет разбор: вызывающий
(`worker.pipeline`) ловит её и отправляет игроку разбор БЕЗ прозы — числа,
вердикты и зоны у него уже есть, они посчитаны кодом. Молча «подправить» текст
нельзя (CLAUDE.md: не выдумывать числа), а прятать весь разбор из-за одной
фразы модели — хуже, чем показать его без неё.

**Повтора вызова при отказе нет.** Повторить — значит заплатить второй раз за
ту же попытку и продлить ожидание игрока ради текста, без которого разбор уже
полон. Схема-ретраи (невалидный JSON) остаются на фасаде `platform/llm.py`, это
другой отказ.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel

from harness.contracts import (
    AnalysisResult,
    PointText,
    PointVerdict,
    SpotKind,
    Street,
    VerdictTextOut,
    Zone,
)
from harness.explanation.digest import NumberBook
from harness.explanation.faithfulness import (
    has_assumption_words,
    unsupported_numbers,
    verdict_label_for,
)

__all__ = ["UnfaithfulText", "VerdictLLM", "verdict_digest", "verdict_text"]

_T = TypeVar("_T", bound=BaseModel)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "verdict.md"

# Глоссарий промпта — НЕ словарь единого голоса (тот живёт в `presentation` и
# принадлежит ему целиком). Здесь описания спотов для модели: она пишет прозу
# сама, а не подставляет наши строки, поэтому дублирования формулировок нет и
# рассинхронизироваться нечему.
_SPOT_BRIEF: dict[SpotKind, str] = {
    SpotKind.PUSHFOLD_UNOPENED: "шов или фолд в неоткрытом банке",
    SpotKind.PUSHFOLD_FACING_SHOVE: "колл или фолд против чужого олл-ина",
    SpotKind.PREFLOP_OTHER: "префлоп вне пуш-фолда",
    SpotKind.POSTFLOP: "постфлоп",
}

_STREET_BRIEF: dict[Street, str] = {
    Street.PREFLOP: "префлоп",
    Street.FLOP: "флоп",
    Street.TURN: "тёрн",
    Street.RIVER: "ривер",
}

_ZONE_BRIEF: dict[Zone, str] = {
    Zone.STRICT: "строго (вывод не зависит от догадки о диапазоне)",
    Zone.ASSUMING: "предполагая (вывод опирается на допущение о диапазоне)",
}


class UnfaithfulText(Exception):
    """Текст модели не прошёл проверку верности — игроку он не уходит.

    Сообщение называет ровно то, что нашла проверка (какие числа выдуманы, у
    какой точки нет слов допущения): оно едет в лог и в `jobs.error`, поэтому
    обязано быть диагностикой, а не «модель ответила плохо».
    """


class VerdictLLM(Protocol):
    """Ровно та часть фасада `platform.llm.LLM`, которой пользуется изложение.

    Протокол, а не импорт класса: конвейерные пакеты не знают про `platform`
    (правило зависимостей CLAUDE.md), а тестам нужен двойник без сети и без
    Postgres. Настоящий `LLM` подходит под протокол структурно — у него шире
    `purpose` и конкретнее метаданные, и то и другое совместимо.
    """

    async def __call__(
        self,
        purpose: Literal["verdict_text"],
        schema: type[_T],
        *,
        prompt: str,
        images: Sequence[bytes] = (),
        trace_id: int,
    ) -> tuple[_T, Any]: ...


class _PointDraft(BaseModel):
    """Что модели РАЗРЕШЕНО вернуть про точку: привязка и слова. Метки здесь нет."""

    dp_index: int
    text: str


class _VerdictDraft(BaseModel):
    points: list[_PointDraft]
    summary: str


class Digest(BaseModel):
    """Выжимка: текст промпта и числа, которые он содержит (они же разрешённые)."""

    text: str
    allowed: frozenset[float]


def _point_lines(point: PointVerdict, ordinal: int, book: NumberBook) -> list[str]:
    """Одна точка выжимки. Каждое число — через `book`, иначе оно окажется
    запрещённым в ответе модели, хотя мы сами его и показали."""
    lines = [
        (
            f"Точка {book.count(ordinal)} (dp_index {book.count(point.dp_index)}): "
            f"{_STREET_BRIEF.get(point.street, point.street.value)}, "
            f"{_SPOT_BRIEF.get(point.spot, point.spot.value)}."
        ),
        (
            f"  сыграно: {point.action_taken or 'не названо'}; "
            f"лучше: {point.best_action or 'не названо'}; "
            f"цена расхождения: {book.bb(point.ev_diff_bb)} bb."
        ),
        f"  зона: {_ZONE_BRIEF[point.zone]}.",
    ]
    interval = point.interval
    if interval is not None:
        near = "да" if interval.near_zero else "нет"
        lines.append(
            f"  оценка EV: {book.bb(interval.point_bb)} bb, разброс по моделям поведения "
            f"оппонентов от {book.bb(interval.low_bb)} до {book.bb(interval.high_bb)} bb, "
            f"потолок цены выбора {book.bb(interval.cost_ceiling_bb)} bb; "
            f"около нуля: {near}."
        )
    assumption = point.assumption
    if assumption is not None:
        share = book.pct(100.0 * assumption.range.fraction_of_hands())
        note = f", {assumption.note}" if assumption.note else ""
        lines.append(
            f"  допущение: диапазон оппонента взят из источника «{assumption.source}»"
            f"{note}; в нём {share}% всех рук."
        )
    return lines


def verdict_digest(res: AnalysisResult) -> Digest:
    """Выжимка расчёта для промпта: только судимые точки, только числа ядра.

    Точки берутся в порядке `res.ranked` — то есть уже отфильтрованные ядром
    судимые точки, самая дорогая первой. Точка без вердикта в выжимку не
    попадает вовсе: модель не должна получить возможность сказать о ней хоть
    что-нибудь (`test_a_point_without_a_verdict_never_reaches_the_model`).
    """
    book = NumberBook()
    lines = [f"Раздача {book.token(res.hand_no)}."]
    lines.append(
        f"Судимых точек решения: {book.count(len(res.ranked))}; "
        f"суммарная цена расхождений: {book.bb(res.total_ev_loss_bb)} bb."
    )
    for ordinal, index in enumerate(res.ranked, start=1):
        lines.append("")
        lines.extend(_point_lines(res.points[index], ordinal, book))
    return Digest(text="\n".join(lines), allowed=book.allowed)


def _validate(draft: _VerdictDraft, res: AnalysisResult, allowed: frozenset[float]) -> None:
    """Три проверки разом; первая же непройденная — отказ от всего текста."""
    expected = {res.points[index].dp_index for index in res.ranked}
    got = {point.dp_index for point in draft.points}
    if got != expected:
        raise UnfaithfulText(
            f"модель ответила по точкам {sorted(got)}, а разбор судит {sorted(expected)}"
        )

    by_dp = {res.points[index].dp_index: res.points[index] for index in res.ranked}
    for draft_point in draft.points:
        invented = unsupported_numbers(draft_point.text, allowed)
        if invented:
            raise UnfaithfulText(
                f"точка {draft_point.dp_index}: числа не из расчёта — {invented}"
            )
        if by_dp[draft_point.dp_index].zone is Zone.ASSUMING and not has_assumption_words(
            draft_point.text
        ):
            raise UnfaithfulText(
                f"точка {draft_point.dp_index}: зона «предполагая», а допущение в тексте "
                f"не названо"
            )
    invented_summary = unsupported_numbers(draft.summary, allowed)
    if invented_summary:
        raise UnfaithfulText(f"вывод по раздаче: числа не из расчёта — {invented_summary}")


async def verdict_text(llm: VerdictLLM, res: AnalysisResult, *, trace_id: int) -> VerdictTextOut:
    """Слова к посчитанному разбору. Модель зовётся один раз и только если есть что
    излагать.

    `trace_id` — обязательный параметр фасада (`llm_calls.trace_id` NOT NULL),
    поэтому он есть и здесь, хотя в интерфейсе плана его нет: без него вызов
    модели невозможен вовсе.

    Пустой `ranked` (судить нечего) модель не видит: возвращается пустой текст
    без единого токена — сказать по такой раздаче нечего, и попытка что-то
    сказать была бы выдумкой (`test_no_judged_points_means_no_model_call`).
    """
    if not res.ranked:
        return VerdictTextOut(points=[], summary="")

    digest = verdict_digest(res)
    prompt = _PROMPT_PATH.read_text(encoding="utf-8").replace("{digest}", digest.text)
    draft, _meta = await llm("verdict_text", _VerdictDraft, prompt=prompt, trace_id=trace_id)
    _validate(draft, res, digest.allowed)

    by_dp = {res.points[index].dp_index: res.points[index] for index in res.ranked}
    order = [res.points[index].dp_index for index in res.ranked]
    texts = {point.dp_index: point.text for point in draft.points}
    return VerdictTextOut(
        points=[
            PointText(
                dp_index=dp_index,
                verdict_label=verdict_label_for(by_dp[dp_index].ev_diff_bb),
                text=texts[dp_index],
            )
            for dp_index in order
        ],
        summary=draft.summary,
    )
