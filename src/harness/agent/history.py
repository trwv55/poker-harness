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
    ToolReturnPart,
    UserPromptPart,
)

from harness.explanation.faithfulness import numbers_in

__all__ = [
    "REWRITE_HEAD",
    "allowed_from_history",
    "calcs_in_history",
    "dump_history",
    "estimate_tokens",
    "load_history",
    "player_numbers",
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


def load_history(raw: list[dict[str, Any]] | None) -> list[ModelMessage]:
    return ModelMessagesTypeAdapter.validate_python(raw) if raw else []


def dump_history(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    """Сообщения в JSON без инструкций: системный промпт и сводка пересобираются
    на каждом ходе, хранить их копию в каждой строке незачем."""
    bare = [
        dataclasses.replace(m, instructions=None) if isinstance(m, ModelRequest) else m
        for m in messages
    ]
    return ModelMessagesTypeAdapter.dump_python(bare, mode="json")


def estimate_tokens(messages: Sequence[ModelMessage]) -> int:
    return len(ModelMessagesTypeAdapter.dump_json(list(messages))) // _BYTES_PER_TOKEN


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
    """Ходы: каждый начинается запросом с репликой игрока."""
    result: list[list[ModelMessage]] = []
    for message in messages:
        if _player_parts(message) or not result:
            result.append([message])
        else:
            result[-1].append(message)
    return result


def trim_history(messages: Sequence[ModelMessage], max_tokens: int) -> list[ModelMessage]:
    """Самые старые ходы — прочь целиком, пока история не влезет в потолок.

    Первый оставшийся запрос чистится от ответов инструментов: pydantic-ai
    склеивает соседние запросы, и ответ на вызов из выброшенного хода мог
    оказаться в одном запросе с новой репликой — провайдер такую историю
    отвергает (`test_trimming_leaves_no_orphan_tool_return`).
    """
    kept = turns(list(messages))
    while kept and estimate_tokens([m for turn in kept for m in turn]) > max_tokens:
        kept.pop(0)
    flat = [m for turn in kept for m in turn]
    if flat and isinstance(flat[0], ModelRequest):
        head = flat[0]
        parts = [p for p in head.parts if not isinstance(p, ToolReturnPart)]
        flat[0] = dataclasses.replace(head, parts=parts)
    return flat


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
