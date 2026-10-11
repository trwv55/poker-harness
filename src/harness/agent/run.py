"""Один ход агента: инструменты, проверки, одна попытка переписать.

Модель маршрутизирует и излагает, считает код (спека агента №1, §1). Проверки —
`explanation.agent_checks`, та же реализация, что в evals. Попытка переписать —
второй вызов модели с тем же контекстом и списком нарушений; это не схема-ретрай
фасада (`_MAX_SCHEMA_ATTEMPTS`) — два разных механизма (спека §6).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage
from sqlalchemy.ext.asyncio import AsyncSession

from harness.agent import prompt as _prompt
from harness.agent.history import (
    REWRITE_HEAD,
    allowed_from_history,
    calcs_in_history,
    player_numbers,
    stats_in_history,
)
from harness.calcs.routing import CalcRun, tools_for
from harness.contracts import CalcName, CalcResult, ModelOutput
from harness.explanation.agent_checks import STAT_LABEL, AgentContext, check_agent_answer
from harness.platform.llm import LLM

__all__ = ["MAX_CALCS_PER_TURN", "AgentDraft", "AgentTurn", "run_agent_turn"]

MAX_CALCS_PER_TURN = 8


class AgentDraft(ModelOutput, BaseModel):
    """Что модели разрешено вернуть: слова. Подпись и числа собирает код."""

    answer: str


@dataclass(frozen=True)
class AgentTurn:
    """Итог хода. `from_context` — ответ стоит на сводке разбора, расчётов в
    ходе нет (`test_an_answer_from_the_context_alone_passes_without_a_call`).
    `drafts` — тексты модели, которые игрок не увидел: для журнала, не для
    игрока (`test_the_journal_keeps_the_rejected_drafts`)."""

    state: Literal["prose", "numbers", "refusal"]
    prose: str | None
    results: tuple[CalcResult, ...]
    from_context: bool
    messages: tuple[ModelMessage, ...]
    violations: tuple[tuple[str, ...], ...]
    calls: tuple[dict[str, Any], ...]
    drafts: tuple[str, ...] = ()

    def journal(self) -> dict[str, Any]:
        """Запись `agent_turn` в `traces.spans` (спека §8, «Журнал»;
        `test_the_journal_names_calls_checks_and_the_rewrite`)."""
        return {
            "state": self.state,
            "answer": self.prose,
            "calls": list(self.calls),
            "checks": [list(v) for v in self.violations],
            "rewrite": len(self.violations) == 2,
            "from_context": self.from_context,
            "drafts": list(self.drafts),
        }


def _violations(
    answer: str, messages: Sequence[ModelMessage], context: AgentContext | None
) -> list[str]:
    allowed = allowed_from_history(messages) | player_numbers(messages)
    labels = frozenset(STAT_LABEL + stat for stat in stats_in_history(messages))
    if context is not None:
        allowed |= context.allowed
        labels |= context.labels
    return check_agent_answer(
        answer,
        allowed=allowed,
        labels=labels,
        leaks_called=CalcName.LEAKS.value in calcs_in_history(messages),
        hero_class=context.hero_class if context else None,
        board_ranks=context.board_ranks if context else (),
    )


def _rewrite_request(violations: Sequence[str]) -> str:
    listed = "\n".join(f"- {v}" for v in violations)
    return f"{REWRITE_HEAD}\n{listed}\nПерепишите ответ, исправив только это."


async def run_agent_turn(
    db: AsyncSession,
    llm: LLM,
    *,
    player_id: int,
    session_id: int,
    text: str,
    trace_id: int,
    history: Sequence[ModelMessage],
    context: AgentContext | None,
    system_prompt: str | None = None,
) -> AgentTurn:
    """Ход: вызов модели, проверки, не более одной попытки переписать.

    Отказ — нет сводки разбора и ни одного расчёта в оставшейся истории вечера,
    включая этот ход
    (`test_an_answer_without_a_call_and_without_a_context_is_refused`). Числа
    без прозы — проза не прошла проверку дважды
    (`test_one_rewrite_then_numbers_without_prose`).
    """
    system = system_prompt if system_prompt is not None else _prompt.load_system_prompt()
    instructions = _prompt.instructions_for(system, context)
    run_state = CalcRun(max_calcs=MAX_CALCS_PER_TURN)
    tools = tools_for(db, player_id, session_id, run_state)

    draft, meta = await llm(
        "agent", AgentDraft, prompt=text, tools=tools,
        message_history=history, instructions=instructions, trace_id=trace_id,
    )
    messages = meta.messages
    grounded = context is not None or bool(calcs_in_history(messages))
    found = _violations(draft.answer, messages, context)
    violations: list[tuple[str, ...]] = [tuple(found)]
    prose: str | None = draft.answer.strip() if not found else None
    drafts: list[str] = [draft.answer] if found or not grounded else []

    if grounded and found:
        draft, meta = await llm(
            "agent", AgentDraft, prompt=_rewrite_request(found), tools=tools,
            message_history=messages, instructions=instructions, trace_id=trace_id,
        )
        messages = meta.messages
        second = _violations(draft.answer, messages, context)
        violations.append(tuple(second))
        prose = draft.answer.strip() if not second else None
        if second:
            drafts.append(draft.answer)

    state: Literal["prose", "numbers", "refusal"]
    if not grounded:
        state, prose = "refusal", None
    elif prose is not None:
        state = "prose"
    else:
        state = "numbers"
    return AgentTurn(
        state=state,
        prose=prose,
        results=tuple(run_state.results),
        from_context=context is not None and not run_state.results,
        messages=tuple(messages),
        violations=tuple(violations),
        calls=tuple(run_state.calls),
        drafts=tuple(drafts),
    )
