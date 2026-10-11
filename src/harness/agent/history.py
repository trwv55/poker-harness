"""История вечера агента: сериализация, ходы, обрезка, разрешённые числа.

История — сообщения pydantic-ai (`ModelMessagesTypeAdapter`), живёт в
`sessions.agent_history` до конца вечера. Разрешённые числа и вызванные расчёты
читаются из метаданных ответов инструментов (`calcs.routing._perform` кладёт
туда выжимку), а не из текста: что ушло из истории обрезкой, перестаёт быть
разрешённым (`test_trimming_drops_whole_turns_and_their_numbers`).

Числа, которые назвал сам игрок, тоже разрешены (решение владельца R1,
`test_a_number_the_player_typed_is_allowed`). Просьба переписать ответ уходит
модели такой же репликой, но игроку не принадлежит: её числа — это список
нарушений, и разрешать их нельзя
(`test_the_rewrite_request_does_not_allow_the_numbers_it_lists`). Она же не
открывает новый ход (`test_a_rewrite_stays_in_the_turn_of_its_question`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from harness.contracts import CalcName
from harness.explanation.faithfulness import numbers_in

__all__ = [
    "REWRITE_HEAD",
    "allowed_from_history",
    "calcs_in_history",
    "dump_history",
    "estimate_tokens",
    "load_history",
    "player_numbers",
    "seen_history",
    "stats_in_history",
    "trim_history",
    "turns",
]

# Начало просьбы переписать (`agent.run._rewrite_request`): по нему реплика кода
# отличается от реплики игрока.
REWRITE_HEAD = "Ответ не прошёл проверку:"

# Оценка, а не счёт токенизатора: байты JSON / 3. Кириллица в UTF-8 — два байта
# на букву и около половины токена на букву, так что оценка завышает — потолок
# срабатывает раньше, а не позже.
_BYTES_PER_TOKEN = 3

_CALC_TOOLS = frozenset(name.value for name in CalcName)


def load_history(raw: list[dict[str, Any]] | None) -> list[ModelMessage]:
    return ModelMessagesTypeAdapter.validate_python(raw) if raw else []


def _bare(messages: Sequence[ModelMessage]) -> list[ModelMessage]:
    """Сообщения без инструкций: системный промпт и сводка пересобираются на
    каждом ходе, хранить их копию в каждом запросе незачем."""
    return [
        dataclasses.replace(m, instructions=None) if isinstance(m, ModelRequest) else m
        for m in messages
    ]


def dump_history(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    """Сообщения в JSON — без инструкций (`_bare`)."""
    return ModelMessagesTypeAdapter.dump_python(_bare(messages), mode="json")


def estimate_tokens(messages: Sequence[ModelMessage]) -> int:
    """Размер того, что хранится, — без инструкций: системный промпт в каждом
    запросе хода съедал бы потолок, хотя в историю не попадает
    (`test_a_long_system_prompt_does_not_eat_the_history_budget`)."""
    return len(ModelMessagesTypeAdapter.dump_json(_bare(messages))) // _BYTES_PER_TOKEN


def _text(part: UserPromptPart) -> str:
    """Текст реплики: фасад шлёт её списком (`[prompt, *картинки]`), а не строкой."""
    if isinstance(part.content, str):
        return part.content
    return "\n".join(item for item in part.content if isinstance(item, str))


def _player_parts(message: ModelMessage) -> list[UserPromptPart]:
    if not isinstance(message, ModelRequest):
        return []
    return [
        part
        for part in message.parts
        if isinstance(part, UserPromptPart) and not _text(part).startswith(REWRITE_HEAD)
    ]


def turns(messages: list[ModelMessage]) -> list[list[ModelMessage]]:
    """Ходы: каждый начинается запросом с репликой игрока.

    Реплика игрока — отдельный запрос (pydantic-ai, как установлен), поэтому
    ответ на последний вызов хода — в том числе на вывод `final_result` —
    остаётся в своём ходе и уходит вместе с ним при обрезке.
    """
    result: list[list[ModelMessage]] = []
    for message in messages:
        if _player_parts(message) or not result:
            result.append([message])
        else:
            result[-1].append(message)
    return result


def trim_history(messages: Sequence[ModelMessage], max_tokens: int) -> list[ModelMessage]:
    """Самые старые ходы — прочь целиком, пока история не влезет в потолок.

    Ход режется только целиком (`turns`), поэтому вызов инструмента не
    отрывается от своего ответа, и оставшаяся история начинается репликой
    игрока (`test_trimming_leaves_no_orphan_tool_return`). Возвращается без
    инструкций — так, как будет храниться.
    """
    kept = turns(_bare(messages))
    while kept and estimate_tokens([m for turn in kept for m in turn]) > max_tokens:
        kept.pop(0)
    return [m for turn in kept for m in turn]


def _is_output_call(part: object) -> bool:
    """Вызов вывода (`final_result`), а не расчёта: инструменты агента — ровно
    имена `CalcName` (`test_the_tool_set_covers_every_calculation_name_exactly_once`)."""
    return isinstance(part, ToolCallPart) and part.tool_name not in _CALC_TOOLS


def seen_history(
    messages: Sequence[ModelMessage], *, final_answer: str | None
) -> list[ModelMessage]:
    """История к сохранению: только то, что игрок видел, плюс вызовы расчётов.

    Уходят просьба переписать и отвергнутые черновики — вызов вывода перед
    просьбой и ответ на него (решение контроллера по вопросу владельца 2,
    `test_the_saved_history_keeps_no_rejected_draft_and_no_rewrite_request`).
    `final_answer` — текст, который игрок получил вместо прозы (состояния
    «числа» и «отказ»): он подменяет ответ последнего вызова вывода; `None` —
    проза прошла и остаётся как есть.
    """
    rejected: set[str] = set()
    last_output: ToolCallPart | None = None
    for index, message in enumerate(messages):
        if not isinstance(message, ModelResponse):
            continue
        for part in message.parts:
            if not _is_output_call(part):
                continue
            assert isinstance(part, ToolCallPart)
            last_output = part
            after = messages[index + 1 : index + 3]
            if any(
                isinstance(m, ModelRequest)
                and any(
                    isinstance(p, UserPromptPart) and _text(p).startswith(REWRITE_HEAD)
                    for p in m.parts
                )
                for m in after
            ):
                rejected.add(part.tool_call_id)

    kept: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            parts = []
            for part in message.parts:
                if isinstance(part, ToolCallPart) and part.tool_call_id in rejected:
                    continue
                if part is last_output and final_answer is not None:
                    part = dataclasses.replace(part, args={"answer": final_answer})
                parts.append(part)
            if parts:
                kept.append(dataclasses.replace(message, parts=parts))
            continue
        request_parts = [
            part
            for part in message.parts
            if not (isinstance(part, ToolReturnPart) and part.tool_call_id in rejected)
            and not (isinstance(part, UserPromptPart) and _text(part).startswith(REWRITE_HEAD))
        ]
        if request_parts:
            kept.append(dataclasses.replace(message, parts=request_parts))
    return kept


def _metadata(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    return [
        part.metadata
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and isinstance(part.metadata, dict)
    ]


def allowed_from_history(messages: Sequence[ModelMessage]) -> frozenset[float]:
    return frozenset(float(n) for meta in _metadata(messages) for n in meta.get("allowed", []))


def calcs_in_history(messages: Sequence[ModelMessage]) -> frozenset[str]:
    return frozenset(str(meta["calc"]) for meta in _metadata(messages) if meta.get("calc"))


def stats_in_history(messages: Sequence[ModelMessage]) -> frozenset[str]:
    """Частоты (`FrequencyStat.value`) из выходов расчётов — без префикса
    `stat:`; метку с пространством имён собирает вызывающий."""
    return frozenset(str(meta["stat"]) for meta in _metadata(messages) if meta.get("stat"))


def player_numbers(messages: Sequence[ModelMessage]) -> frozenset[float]:
    """Числа из реплик игрока, оставшихся в истории (решение владельца R1)."""
    return frozenset(
        number
        for message in messages
        for part in _player_parts(message)
        for number in numbers_in(_text(part))
    )
