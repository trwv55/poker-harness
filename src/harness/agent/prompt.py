"""Системный промпт агента и инструкции хода.

Файл промпта закрыт политикой публикации (`.githooks/pre-push`, путь
`src/harness/[^/]+/prompts/`), читается лениво (`read_prompt`): без файла падает
тот, кто зовёт модель, а не тот, кто импортировал пакет.
"""

from __future__ import annotations

from pathlib import Path

from harness.explanation.agent_checks import AgentContext
from harness.explanation.verdict_text import read_prompt

__all__ = ["SYSTEM_PROMPT_PATH", "instructions_for", "load_system_prompt"]

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"


def load_system_prompt() -> str:
    return read_prompt(SYSTEM_PROMPT_PATH)


def instructions_for(system: str, context: AgentContext | None) -> str:
    """Системный промпт плюс сводка последнего разбора — в инструкциях, а не в
    реплике: инструкции текущего запроса не копятся в истории
    (`test_instructions_carry_the_system_prompt_and_the_context`)."""
    if context is None:
        return f"{system}\n\nРазбора руки в этом вечере нет: отвечайте только расчётами."
    return (
        f"{system}\n\nСводка последнего разбора (числа — только отсюда или из расчётов):\n"
        f"{context.text}"
    )
