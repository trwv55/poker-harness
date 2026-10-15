"""История вечера агента: сериализация, ходы, обрезка, разрешённые числа.

История — сообщения pydantic-ai (`ModelMessagesTypeAdapter`), живёт в
`sessions.agent_history` до конца вечера. Разрешённые числа и вызванные расчёты
читаются из метаданных ответов инструментов (`calcs.routing._perform` кладёт
туда выжимку), а не из текста: что ушло из истории обрезкой, перестаёт быть
разрешённым (`test_trimming_drops_whole_turns_and_their_numbers`).

Числа, которые назвал сам игрок, тоже разрешены (решение владельца R1,
`test_a_number_the_player_typed_is_allowed`); обозначения карт в его репликах
числами не считаются (`player_text_numbers`). Просьба переписать ответ уходит
модели репликой, но игроку не принадлежит; в сохранённую историю она не
попадает (`seen_history`), а опознаётся по месту в ходе, а не по тексту
(`test_a_player_message_that_looks_like_a_rewrite_request_is_kept`).
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
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from harness.contracts import CalcName
from harness.explanation.agent_checks import strip_card_tokens
from harness.explanation.faithfulness import numbers_in

__all__ = [
    "allowed_from_history",
    "assumed_numbers_in",
    "calcs_in_history",
    "chart_numbers_in",
    "dump_history",
    "estimate_tokens",
    "frequency_numbers",
    "load_history",
    "player_numbers",
    "player_text_numbers",
    "seen_history",
    "stats_in_history",
    "threshold_decided",
    "trim_history",
    "turns",
]

# Оценка, а не счёт токенизатора: байты JSON / 3. Кириллица в UTF-8 — два байта
# на букву и около половины токена на букву, так что оценка завышает — потолок
# срабатывает раньше, а не позже.
_BYTES_PER_TOKEN = 3

_CALC_TOOLS = frozenset(name.value for name in CalcName)


def _without_thinking(messages: Sequence[ModelMessage]) -> list[ModelMessage]:
    """Сообщения без рассуждений модели. Подпись рассуждения привязана к системному
    промпту и инструментам, а инструкции агента меняются от хода к ходу — повтор
    рассуждения из истории провайдер отвергает (400). Ответ, в котором кроме
    рассуждения ничего не было, уходит целиком: вызовов в нём нет, выходов
    расчётов на него тоже
    (`test_a_stored_row_with_thinking_loads_without_it`)."""
    kept: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            parts = [part for part in message.parts if not isinstance(part, ThinkingPart)]
            if not parts:
                continue
            if len(parts) != len(message.parts):
                message = dataclasses.replace(message, parts=parts)
        kept.append(message)
    return kept


def load_history(raw: list[dict[str, Any]] | None) -> list[ModelMessage]:
    """История вечера из строки — без рассуждений (`_without_thinking`): строки,
    сохранённые прежним кодом, могли их содержать."""
    return _without_thinking(ModelMessagesTypeAdapter.validate_python(raw)) if raw else []


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
    """Реплики в запросе. В сохранённой истории это всегда реплики игрока:
    просьбу переписать `seen_history` туда не пускает."""
    if not isinstance(message, ModelRequest):
        return []
    return [part for part in message.parts if isinstance(part, UserPromptPart)]


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
    имена `CalcName` (`test_the_agent_tool_set_covers_every_calculation_name_exactly_once`)."""
    return isinstance(part, ToolCallPart) and part.tool_name not in _CALC_TOOLS


def _answer(message: ModelMessage) -> ToolCallPart | ModelResponse | None:
    """Ответ модели, которым кончился прогон, в двух формах: вызов вывода или
    ответ без единого вызова — простой текст, принятый фасадом наравне с
    выводом (`test_a_final_answer_in_plain_text_is_shown_and_saved`). Текст
    рядом с вызовами ответом не считается: библиотека тогда исполняет вызовы."""
    if not isinstance(message, ModelResponse):
        return None
    calls = [part for part in message.parts if isinstance(part, ToolCallPart)]
    if not calls:
        return message if any(isinstance(part, TextPart) for part in message.parts) else None
    outputs = [part for part in calls if _is_output_call(part)]
    return outputs[-1] if outputs else None


def seen_history(
    messages: Sequence[ModelMessage],
    *,
    start: int,
    rewrite_at: int | None,
    final_answer: str | None,
) -> list[ModelMessage]:
    """История к сохранению: реплики игрока, вызовы расчётов с ответами и один
    ответ модели на ход — тот, что ушёл игроку.

    `messages[:start]` — история вечера, уже прошедшая эту чистку, берётся как
    есть, кроме рассуждений: их в сохранённой истории нет ни в одном ответе
    (`_without_thinking`,
    `test_no_thinking_is_saved_on_any_path_and_the_next_turn_gets_none`). В сообщениях хода (`messages[start:]`) ответ модели — вызов вывода
    или ответ простым текстом без вызовов (`_answer`), и:

    * ответы модели сводятся к вызовам расчётов и последнему ответу — текст и
      рассуждения рядом с вызовами игрок не видел
      (`test_text_next_to_a_rejected_draft_is_not_saved`,
      `test_text_next_to_the_final_answer_is_not_saved`); у ответа текстом
      остаётся только текст (`test_a_final_answer_in_plain_text_is_shown_and_saved`);
    * прежние ответы — отвергнутые черновики в любой из двух форм — уходят
      вместе с ответами на вызов вывода, реплики в сообщении `rewrite_at`
      (просьба переписать — первое сообщение второго прогона) тоже
      (`test_the_saved_history_keeps_no_rejected_draft_and_no_rewrite_request`,
      `test_a_rejected_text_draft_is_not_saved_and_the_text_rewrite_is`,
      `test_a_text_draft_and_a_tool_rewrite_mix`);
    * `final_answer` — текст, который игрок получил вместо ответа модели
      (состояния «числа» и «отказ»; проза без выброшенного блока «Оценка модели»
      — `AgentTurn.saved_messages`), подменяет последний ответ; `None` — ответ
      прошёл и остаётся как есть
      (`test_a_refused_answer_is_saved_as_the_refusal_the_player_saw`,
      `test_a_refused_text_answer_is_saved_as_the_refusal_the_player_saw`).
    """
    turn = list(messages[start:])
    answers = [answer for message in turn if (answer := _answer(message)) is not None]
    final = answers[-1] if answers else None
    dropped = {
        part.tool_call_id for part in answers if part is not final and isinstance(part, ToolCallPart)
    }

    kept: list[ModelMessage] = list(messages[:start])
    for offset, message in enumerate(turn):
        if isinstance(message, ModelResponse):
            if message is final:
                text_parts: list[TextPart] = (
                    [TextPart(final_answer)]
                    if final_answer is not None
                    else [part for part in message.parts if isinstance(part, TextPart)]
                )
                kept.append(dataclasses.replace(message, parts=text_parts))
                continue
            if any(message is answer for answer in answers):
                continue
            parts: list[ToolCallPart] = []
            for part in message.parts:
                if not isinstance(part, ToolCallPart) or part.tool_call_id in dropped:
                    continue
                if part is final and final_answer is not None:
                    part = dataclasses.replace(part, args={"answer": final_answer})
                parts.append(part)
            if parts:
                kept.append(dataclasses.replace(message, parts=parts))
            continue
        is_rewrite = rewrite_at is not None and start + offset == rewrite_at
        request_parts = [
            part
            for part in message.parts
            if not (isinstance(part, ToolReturnPart) and part.tool_call_id in dropped)
            and not (is_rewrite and isinstance(part, UserPromptPart))
        ]
        if request_parts:
            kept.append(dataclasses.replace(message, parts=request_parts))
    return _without_thinking(kept)


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


def frequency_numbers(messages: Sequence[ModelMessage]) -> frozenset[float]:
    """Измеренные частоты из выходов расчётов (доля, числитель, знаменатель) — без
    порогов и долей защиты (`test_the_measured_frequency_is_kept_apart_from_other_calc_numbers`)."""
    return frozenset(float(n) for meta in _metadata(messages) for n in meta.get("frequency", []))


def chart_numbers_in(messages: Sequence[ModelMessage]) -> frozenset[float]:
    """Числа выхода `opponent_range`: доля рук в процентах и комбо
    (`test_the_agent_reads_the_range_by_the_chart_and_counts_equity_against_its_narrowing`)."""
    return frozenset(float(n) for meta in _metadata(messages) for n in meta.get("chart", []))


def assumed_numbers_in(messages: Sequence[ModelMessage]) -> frozenset[float]:
    """Числа выхода `equity_vs_range`: эквити и доля от стартового в процентах —
    числа расчёта, но не измеренная частота
    (`test_equity_numbers_are_calculation_numbers_and_never_a_measured_frequency`)."""
    return frozenset(float(n) for meta in _metadata(messages) for n in meta.get("assumed", []))


def threshold_decided(messages: Sequence[ModelMessage]) -> bool:
    """В сообщениях есть `frequency_vs_threshold` с выводом выше или ниже порога
    (не «не определено»): только тогда «в плюсе» / «в минусе» допустимы выводом
    (`test_a_decided_threshold_in_the_conversation_allows_a_profit_conclusion`)."""
    return any(meta.get("outcome") in ("above", "below") for meta in _metadata(messages))


def player_text_numbers(text: str, *, board_ranks: Sequence[str] = ()) -> frozenset[float]:
    """Числа из реплики игрока — без обозначений карт: «K9o» или «K-7-2» числа 9,
    7 и 2 не разрешают, как и на стороне ответа (`strip_card_tokens`;
    `test_a_card_rank_the_player_typed_is_not_an_allowed_number`)."""
    return frozenset(numbers_in(strip_card_tokens(text, board_ranks=board_ranks)))


def player_numbers(
    messages: Sequence[ModelMessage], *, board_ranks: Sequence[str] = ()
) -> frozenset[float]:
    """Числа из реплик игрока, оставшихся в истории (решение владельца R1), без
    обозначений карт (`player_text_numbers`)."""
    return frozenset(
        number
        for message in messages
        for part in _player_parts(message)
        for number in player_text_numbers(_text(part), board_ranks=board_ranks)
    )
