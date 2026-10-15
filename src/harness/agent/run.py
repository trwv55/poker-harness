"""Один ход агента: инструменты, проверки, одна попытка переписать.

Модель маршрутизирует и излагает, считает код (спека агента №1, §1). Проверки —
`explanation.agent_checks`, та же реализация, что в evals. Попытка переписать —
второй вызов модели с тем же контекстом и списком нарушений; это не схема-ретрай
фасада (`_MAX_SCHEMA_ATTEMPTS`) — два разных механизма (спека §6).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from sqlalchemy.ext.asyncio import AsyncSession

from harness.agent import prompt as _prompt
from harness.agent.history import (
    allowed_from_history,
    assumed_numbers_in,
    calcs_in_history,
    chart_numbers_in,
    frequency_numbers,
    hand_boundary,
    merged_requests,
    player_numbers,
    player_text_numbers,
    seen_history,
    stats_in_history,
    threshold_decided,
)
from harness.calcs.routing import CalcRun, tools_for
from harness.contracts import CalcName, CalcResult, ModelOutput
from harness.explanation.agent_checks import (
    STAT_LABEL,
    AgentContext,
    check_agent_answer,
    check_agent_assumption,
)
from harness.platform.llm import LLM

__all__ = ["MAX_CALCS_PER_TURN", "REWRITE_HEAD", "AgentDraft", "AgentTurn", "run_agent_turn"]

MAX_CALCS_PER_TURN = 8

# Начало просьбы переписать. Опознаётся она не по нему, а по месту в ходе
# (`AgentTurn.rewrite_at`): игрок может напечатать ту же фразу.
REWRITE_HEAD = "Ответ не прошёл проверку:"


# Разделитель блока «Оценка модели» в ответе простым текстом: в начале строки,
# регистр не важен; разметка markdown вокруг («**…:**», «> », «## ») не мешает
# (`test_a_markdown_separator_also_splits_the_text`).
_ASSUMPTION_SEPARATOR_RE = re.compile(
    r"^[ \t\u00a0*_#>]*оценка модели[*_]*:[*_]*", re.IGNORECASE | re.MULTILINE
)


class AgentDraft(ModelOutput, BaseModel):
    """Что модели разрешено вернуть: слова. Подпись и числа собирает код.

    `assumption` — блок «Оценка модели», чтение диапазона соперника (решение
    владельца 2026-10-15): код проверяет в нём только числа и словарные запреты
    (`check_agent_assumption`), смысл — нет."""

    answer: str = Field(
        description=(
            "Ответ игроку: всё, кроме чтения диапазона соперника. Сюда — то, что "
            "посчитано и что сказано в разборе раздачи."
        )
    )
    assumption: str | None = Field(
        default=None,
        description=(
            "Чтение диапазона соперника: стартовый диапазон по чарту, его сужение по "
            "линии, эквити против суженного из расчёта. Игрок увидит это под "
            "заголовком «Оценка модели». Нет такого чтения — не заполнять."
        ),
    )

    @classmethod
    def from_text(cls, text: str) -> AgentDraft:
        """Ответ простым текстом вместо вызова вывода: модель агента идёт с
        `tool_choice: auto` и делает и так, и так
        (`test_a_final_answer_in_plain_text_is_shown_and_saved`). Пустой текст —
        не ответ, а ошибка схемы (`test_a_blank_text_answer_is_a_schema_error`).
        Строка, начатая с «Оценка модели:» (регистр не важен), отделяет блок: всё
        после неё — `assumption`, до — `answer`; без такой строки всё — `answer`
        (`test_text_with_the_separator_splits_into_answer_and_block`,
        `test_text_without_the_separator_at_a_line_start_is_all_answer`)."""
        separator = _ASSUMPTION_SEPARATOR_RE.search(text)
        head = text if separator is None else text[: separator.start()]
        tail = None if separator is None else text[separator.end() :].strip()
        answer = head.strip()
        if not answer:
            raise ModelRetry("пустой ответ")
        return cls(answer=answer, assumption=tail or None)

    def as_text(self) -> str:
        """Черновик одной строкой для журнала: ответ и блок за разделителем."""
        if not self.assumption:
            return self.answer
        return f"{self.answer}\n\nОценка модели:\n{self.assumption}"


@dataclass(frozen=True)
class AgentTurn:
    """Итог хода. Расчётов в ходе нет — подпись называет, на чём ещё стоит
    ответ: `from_context` — сводка разбора есть, `from_history` — в истории
    вечера остались расчёты прошлых ходов
    (`test_the_signature_flags_name_the_sources_of_a_turn_without_calculations`).
    `drafts` — тексты модели, которые игрок не увидел: для журнала, не для
    игрока (`test_the_journal_keeps_the_rejected_drafts`)."""

    state: Literal["prose", "numbers", "refusal"]
    prose: str | None
    results: tuple[CalcResult, ...]
    from_context: bool
    from_history: bool
    messages: tuple[ModelMessage, ...]
    violations: tuple[tuple[str, ...], ...]
    calls: tuple[dict[str, Any], ...]
    drafts: tuple[str, ...] = ()
    # Где в `messages` начинается ход (реплика игрока; после попытки переписать она
    # может быть склеена с выходом вывода прошлого хода) и где лежит просьба
    # переписать — по ним `saved_messages` чистит ход, не читая текст реплик.
    history_len: int = 0
    rewrite_at: int | None = None
    # Указатель разбора хода — им помечается реплика в сохранённой истории.
    analysis_id: int | None = None
    # Блок «Оценка модели», который игрок увидит; `assumption_dropped` — блок был в
    # последнем черновике (или в первом, чей ответ показан), а игрок его не увидит
    # (`test_a_block_that_fails_twice_is_dropped_and_the_answer_is_shown`,
    # `test_a_failed_rewrite_keeps_the_first_answer_without_the_block`).
    assumption: str | None = None
    assumption_dropped: bool = False

    def saved_messages(self, final_answer: str | None) -> list[ModelMessage]:
        """Что сохранить в историю вечера (`history.seen_history`). Проза прошла,
        а блок выброшен — сохраняется проза без блока, как её увидел игрок
        (`test_a_block_that_fails_twice_is_dropped_and_the_answer_is_shown`)."""
        if final_answer is None and self.assumption_dropped:
            final_answer = self.prose
        return seen_history(
            self.messages,
            start=self.history_len,
            rewrite_at=self.rewrite_at,
            final_answer=final_answer,
            analysis_id=self.analysis_id,
        )

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
            "from_history": self.from_history,
            "drafts": list(self.drafts),
            "assumption": self.assumption,
            "assumption_dropped": self.assumption_dropped,
        }


def _violations(
    answer: str,
    messages: Sequence[ModelMessage],
    context: AgentContext | None,
    said: frozenset[float],
) -> list[str]:
    """Нарушения одного ответа, без блока «Оценка модели»."""
    return _draft_violations(AgentDraft(answer=answer), messages, context, said)[0]


def _draft_violations(
    draft: AgentDraft,
    messages: Sequence[ModelMessage],
    context: AgentContext | None,
    said: frozenset[float],
) -> tuple[list[str], list[str]]:
    """Нарушения ответа и блока «Оценка модели» — порознь: при повторном провале
    одного блока игрок получает прозу без него. Разрешённые числа у них одни."""
    calc_numbers = allowed_from_history(messages)
    allowed = calc_numbers | said
    labels = frozenset(STAT_LABEL + stat for stat in stats_in_history(messages))
    if context is not None:
        allowed |= context.allowed
        labels |= context.labels
        calc_numbers |= context.threshold_numbers
    leaks_called = CalcName.LEAKS.value in calcs_in_history(messages)
    block = (
        check_agent_assumption(
            draft.assumption,
            allowed=allowed,
            leaks_called=leaks_called,
            hero_class=context.hero_class if context else None,
            board_ranks=context.board_ranks if context else (),
        )
        if draft.assumption
        else []
    )
    return check_agent_answer(
        draft.answer,
        allowed=allowed,
        labels=labels,
        leaks_called=leaks_called,
        hero_class=context.hero_class if context else None,
        board_ranks=context.board_ranks if context else (),
        threshold_decided=threshold_decided(messages),
        profit_without_folds=context.profit_without_folds if context else False,
        calc_numbers=calc_numbers,
        frequency_numbers=frequency_numbers(messages),
        chart_numbers=chart_numbers_in(messages),
        assumed_numbers=assumed_numbers_in(messages),
    ), block


def _rewrite_index(messages: Sequence[ModelMessage], since: int, request: str) -> int:
    """Номер сообщения с просьбой переписать — первая реплика второго прогона.

    Ищется только среди сообщений, которых не было после первого прогона
    (`since` — его последнее сообщение, на случай если библиотека склеит
    просьбу с ним), и по точному тексту просьбы: реплик игрока там нет.
    """
    for index in range(max(since, 0), len(messages)):
        message = messages[index]
        if isinstance(message, ModelRequest) and any(
            isinstance(part, UserPromptPart) and request in (
                [part.content] if isinstance(part.content, str) else list(part.content)
            )
            for part in message.parts
        ):
            return index
    raise RuntimeError("просьба переписать не нашлась в сообщениях второго прогона")


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
    analysis_id: int | None = None,
) -> AgentTurn:
    """Ход: вызов модели, проверки, не более одной попытки переписать.

    Отказ — нет сводки разбора и ни одного расчёта в оставшейся истории вечера,
    включая этот ход
    (`test_an_answer_without_a_call_and_without_a_context_is_refused`). Числа
    без прозы — проза не прошла проверку дважды
    (`test_one_rewrite_then_numbers_without_prose`). Нарушения блока «Оценка
    модели» идут в тот же список и в ту же одну попытку переписать; не прошёл
    дважды только блок — проза без блока
    (`test_a_block_that_fails_twice_is_dropped_and_the_answer_is_shown`). В первом
    черновике прошёл ответ и не прошёл только блок, а переписанный ответ не прошёл —
    игрок получает ответ первого черновика без блока
    (`test_a_failed_rewrite_keeps_the_first_answer_without_the_block`); переписанный
    прошёл целиком — показывается он
    (`test_a_rewrite_that_passes_fully_replaces_the_first_answer`). `analysis_id` — указатель
    разбора руки вечера для `opponent_range` и `equity_vs_range`; без него они
    отвечают отказом (`test_without_a_hand_analysis_the_range_tool_says_so`); он же
    решает, ставить ли перед вопросом отметку смены руки (`history.hand_boundary`).
    """
    system = system_prompt if system_prompt is not None else _prompt.load_system_prompt()
    instructions = _prompt.instructions_for(system, context)
    board = context.board_ranks if context is not None else ()
    # Числа игрока (решение R1): его реплики в истории вечера и текущая, без
    # обозначений карт. Не из `messages`: после попытки переписать там и просьба
    # кода с выдуманными числами.
    said = player_numbers(history, board_ranks=board) | player_text_numbers(
        text, board_ranks=board
    )
    # Банк и ставку для расчётов защиты модель берёт из сводки, слов игрока или
    # выходов расчётов вечера, а не выводит сама (решение контроллера по I2).
    sizes_from = said | allowed_from_history(history)
    if context is not None:
        sizes_from |= context.allowed
    run_state = CalcRun(
        max_calcs=MAX_CALCS_PER_TURN, sizes_from=sizes_from, analysis_id=analysis_id
    )
    tools = tools_for(db, player_id, session_id, run_state)
    # Другая рука, чем у прошлого хода вечера, — отметка перед вопросом (решение
    # владельца 2026-10-15); она в реплике хода и уйдёт с ней в историю.
    boundary = hand_boundary(history, analysis_id)

    draft, meta = await llm(
        "agent", AgentDraft, prompt=text, preface=() if boundary is None else (boundary,),
        tools=tools,
        message_history=history, instructions=instructions,
        text_output=AgentDraft.from_text, trace_id=trace_id,
    )
    messages = meta.messages
    grounded = context is not None or bool(calcs_in_history(messages))
    answer_found, block_found = _draft_violations(draft, messages, context, said)
    found = answer_found + block_found
    violations: list[tuple[str, ...]] = [tuple(found)]
    prose: str | None = draft.answer.strip() if not found else None
    # Ответ первого черновика, прошедший проверку, — запасной, если переписанный
    # ответ не пройдёт, а в первом провалился только блок.
    first_answer = draft.answer.strip() if not answer_found else None
    fell_back = False
    drafts: list[str] = [draft.as_text()] if found or not grounded else []

    rewrite_at: int | None = None
    start = len(history)
    if grounded and found:
        request = _rewrite_request(found)
        # Склеено заранее, как склеила бы библиотека (выход вывода прошлого хода и
        # реплика этого стоят подряд): тогда она историю не меняет, и начало хода
        # известно по месту (`history.merged_requests`).
        start = len(merged_requests(messages[: start + 1])) - 1
        messages = merged_requests(messages)
        first_len = len(messages)
        draft, meta = await llm(
            "agent", AgentDraft, prompt=request, tools=tools,
            message_history=messages, instructions=instructions,
            text_output=AgentDraft.from_text, trace_id=trace_id,
        )
        messages = meta.messages
        rewrite_at = _rewrite_index(messages, first_len - 1, request)
        answer_found, block_found = _draft_violations(draft, messages, context, said)
        second = answer_found + block_found
        violations.append(tuple(second))
        prose = draft.answer.strip() if not answer_found else first_answer
        fell_back = bool(answer_found) and first_answer is not None
        if second:
            drafts.append(draft.as_text())

    state: Literal["prose", "numbers", "refusal"]
    if not grounded:
        state, prose = "refusal", None
    elif prose is not None:
        state = "prose"
    else:
        state = "numbers"
    # Блок показывается, только если прошёл сам и прошла проза; не прошёл только
    # блок — проза идёт без него (решение владельца 2026-10-15). Проза взята из
    # первого черновика — его блок не прошёл, а блок второго к ней не относится.
    block_text = draft.assumption.strip() if draft.assumption else None
    assumption = block_text if state == "prose" and not block_found and not fell_back else None
    return AgentTurn(
        state=state,
        prose=prose,
        results=tuple(run_state.results),
        from_context=context is not None and not run_state.results,
        from_history=bool(calcs_in_history(history)) and not run_state.results,
        messages=tuple(messages),
        violations=tuple(violations),
        calls=tuple(run_state.calls),
        drafts=tuple(drafts),
        history_len=start,
        rewrite_at=rewrite_at,
        analysis_id=analysis_id,
        assumption=assumption,
        assumption_dropped=fell_back
        or (grounded and block_text is not None and assumption is None),
    )
