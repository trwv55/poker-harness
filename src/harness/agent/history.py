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

Граница руки (решение владельца 2026-10-15): реплика игрока в сохранённой
истории несёт в метаданных запроса указатель разбора своего хода
(`HAND_KEY`). Ход о другой руке, чем прошлый, — или переход между рукой и
вопросом без разбора — получает перед вопросом отметку (`hand_boundary`):
отдельный кусок той же реплики (`TextContent`), который модель видит, а числа
игрока и ходы не видят
(`test_a_new_hand_gets_a_boundary_before_the_question_and_the_same_hand_does_not`,
`test_the_boundary_has_no_numbers_and_is_not_the_players_text`).
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
    RetryPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from harness.contracts import CalcName, RangeNarrowing
from harness.explanation.agent_checks import strip_card_tokens
from harness.explanation.faithfulness import numbers_in

__all__ = [
    "HAND_KEY",
    "NEW_HAND_MARK",
    "NO_HAND_MARK",
    "allowed_from_history",
    "assumed_numbers_in",
    "calcs_in_history",
    "chart_numbers_in",
    "dump_history",
    "estimate_tokens",
    "frequency_numbers",
    "hand_boundary",
    "hand_player_texts",
    "load_history",
    "merged_requests",
    "narrowings_in",
    "player_numbers",
    "player_text_numbers",
    "seen_history",
    "stats_in_history",
    "threshold_decided",
    "threshold_numbers_in",
    "trim_history",
    "turns",
]

# Оценка, а не счёт токенизатора: байты JSON / 3. Кириллица в UTF-8 — два байта
# на букву и около половины токена на букву, так что оценка завышает — потолок
# срабатывает раньше, а не позже.
_BYTES_PER_TOKEN = 3

_CALC_TOOLS = frozenset(name.value for name in CalcName)

# Ключ метаданных запроса с репликой игрока: указатель разбора её хода (`None` —
# ход без разбора).
HAND_KEY = "analysis_id"
# Тексты отметки-границы — без чисел.
NEW_HAND_MARK = (
    "— Дальше разговор о другой руке. Ответы выше относились к прошлым рукам вечера; "
    "не пересматривай их. —"
)
NO_HAND_MARK = (
    "— Дальше вопрос не о конкретной руке. Ответы выше относились к прошлым рукам вечера; "
    "не пересматривай их. —"
)


def _is_tool_result(part: object) -> bool:
    """Ключ порядка в склеенном запросе — тот же, что у библиотеки."""
    return isinstance(part, ToolReturnPart | RetryPromptPart)


def _answers_a_call(part: object) -> bool:
    """Ответ на вызов инструмента — по тому же правилу, что `_is_tool_result_part`
    библиотеки: `RetryPromptPart` без имени инструмента ответом не считается."""
    return isinstance(part, ToolReturnPart) or (
        isinstance(part, RetryPromptPart) and part.tool_name is not None
    )


def merged_requests(messages: Sequence[ModelMessage]) -> list[ModelMessage]:
    """Запросы подряд — одним запросом, выходы инструментов впереди: так их склеивает
    библиотека перед прогоном. Склеенная заранее история проходит через неё без
    изменений, и её длина — то место, где в `all_messages()` начинается ход
    (`test_the_start_of_the_turn_holds_from_the_third_turn_with_output_answers`).
    Метаданные запросов не теряются: склейка библиотеки их бросает."""
    kept: list[ModelMessage] = []
    for message in messages:
        previous = kept[-1] if kept else None
        if isinstance(message, ModelRequest) and isinstance(previous, ModelRequest):
            tags = {**(previous.metadata or {}), **(message.metadata or {})}
            kept[-1] = dataclasses.replace(
                message,
                parts=sorted([*previous.parts, *message.parts], key=lambda p: not _is_tool_result(p)),
                instructions=previous.instructions or message.instructions,
                metadata=tags or None,
            )
            continue
        kept.append(message)
    return kept


def _without_thinking(messages: Sequence[ModelMessage]) -> list[ModelMessage]:
    """Сообщения без рассуждений модели, в том виде, в каком их склеила бы
    библиотека (`merged_requests`). Подпись рассуждения привязана к системному
    промпту и инструментам, а инструкции агента меняются от хода к ходу — повтор
    рассуждения из истории провайдер отвергает (400). Ответ, в котором кроме
    рассуждения ничего не было, уходит целиком: вызовов в нём нет, выходов
    расчётов на него тоже
    (`test_a_stored_row_with_thinking_loads_without_it`).

    Если такой ответ — последний ответ хода, ход игроку ничего не дал и уходит
    целиком (`turns`), с вызовами и выходами расчётов: иначе выход расчёта,
    склеенный с репликой следующего хода, после обрезки встал бы в начало истории
    (`test_dropped_thinking_only_answers_do_not_shift_the_start_of_the_turn`,
    `test_a_turn_ending_in_a_thinking_only_answer_leaves_no_orphan_tool_return`).
    Выходы инструментов прошлого хода, склеенные с репликой ушедшего хода,
    остаются: их вызовы в истории."""
    kept: list[ModelMessage] = []
    for turn in turns(list(messages)):
        cleaned: list[ModelMessage] = []
        answered = True
        for message in turn:
            if isinstance(message, ModelResponse):
                parts = [part for part in message.parts if not isinstance(part, ThinkingPart)]
                answered = bool(parts)
                if not parts:
                    continue
                if len(parts) != len(message.parts):
                    message = dataclasses.replace(message, parts=parts)
            cleaned.append(message)
        if answered:
            kept.extend(cleaned)
            continue
        head = turn[0]
        if isinstance(head, ModelRequest) and (
            results := [part for part in head.parts if _answers_a_call(part)]
        ):
            kept.append(ModelRequest(parts=results))
    return merged_requests(kept)


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
    """Текст реплики: фасад шлёт её списком (`[*отметка, prompt, *картинки]`), а не
    строкой. Отметка-граница (`TextContent`) — не текст игрока и сюда не входит."""
    if isinstance(part.content, str):
        return part.content
    return "\n".join(item for item in part.content if isinstance(item, str))


def _player_parts(message: ModelMessage) -> list[UserPromptPart]:
    """Реплики в запросе. В сохранённой истории это всегда реплики игрока:
    просьбу переписать `seen_history` туда не пускает."""
    if not isinstance(message, ModelRequest):
        return []
    return [part for part in message.parts if isinstance(part, UserPromptPart)]


def hand_boundary(
    history: Sequence[ModelMessage], analysis_id: int | None
) -> TextContent | None:
    """Отметка перед вопросом хода о руке `analysis_id`, если прошлый ход вечера
    был о другой руке или без разбора, либо наоборот
    (`test_a_turn_without_a_hand_and_back_gets_boundaries`). Первый ход вечера
    отметки не получает; не получает её и ход, чей прошлый ход не помечен рукой —
    строка, сохранённая до отметок
    (`test_an_old_row_without_hand_tags_loads_and_gets_no_boundary`)."""
    for message in reversed(history):
        if not isinstance(message, ModelRequest) or not _player_parts(message):
            continue
        tags = message.metadata
        if not isinstance(tags, dict) or HAND_KEY not in tags or tags[HAND_KEY] == analysis_id:
            return None
        text = NEW_HAND_MARK if analysis_id is not None else NO_HAND_MARK
        return TextContent(text)
    return None


def turns(messages: list[ModelMessage]) -> list[list[ModelMessage]]:
    """Ходы: каждый начинается запросом с репликой игрока.

    Выходы расчётов остаются в своём ходе: за ними внутри хода всегда идёт ответ
    модели. Выход вывода (`final_result`) в истории склеен с репликой следующего
    хода (`merged_requests`) и считается частью того хода
    (`test_an_old_row_with_separate_output_returns_loads_merged`).
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

    Ход режется только целиком (`turns`), поэтому вызов расчёта не отрывается
    от своего выхода, и оставшаяся история начинается репликой игрока
    (`test_trimming_leaves_no_orphan_tool_return`). Выход вывода, склеенный с этой
    репликой, после ушедшего хода остался бы без вызова — он снимается
    (`test_the_start_of_the_turn_holds_from_the_third_turn_with_output_answers`).
    Возвращается без инструкций — так, как будет храниться.
    """
    kept = turns(_bare(messages))
    while kept and estimate_tokens([m for turn in kept for m in turn]) > max_tokens:
        kept.pop(0)
    result = [m for turn in kept for m in turn]
    # Выходы прошлого хода, склеенные с репликой (`merged_requests`), без своего вызова
    # провайдер отвергает: в начале истории их не бывает.
    if result and isinstance(head := result[0], ModelRequest):
        parts = [part for part in head.parts if not _answers_a_call(part)]
        if len(parts) != len(head.parts):
            result[0] = dataclasses.replace(head, parts=parts)
    return result


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
    analysis_id: int | None = None,
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
      `test_a_refused_text_answer_is_saved_as_the_refusal_the_player_saw`);
    * реплика игрока, начавшая ход, помечается указателем разбора хода
      `analysis_id` (`HAND_KEY`) — по нему следующий ход ставит отметку-границу
      (`hand_boundary`), а сама отметка остаётся в реплике
      (`test_a_rewrite_on_a_new_hand_keeps_the_boundary_and_drops_the_request`).
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
        if not request_parts:
            continue
        changes: dict[str, Any] = {"parts": request_parts}
        if offset == 0 and _player_parts(message):
            changes["metadata"] = {**(message.metadata or {}), HAND_KEY: analysis_id}
        kept.append(dataclasses.replace(message, **changes))
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
    """Числа выхода `equity_vs_range`: эквити в процентах —
    числа расчёта, но не измеренная частота
    (`test_equity_numbers_are_calculation_numbers_and_never_a_measured_frequency`)."""
    return frozenset(float(n) for meta in _metadata(messages) for n in meta.get("assumed", []))


def threshold_numbers_in(messages: Sequence[ModelMessage]) -> frozenset[float]:
    """Порог выхода `frequency_vs_threshold` в процентах
    (`test_the_threshold_of_frequency_vs_threshold_reaches_the_check`)."""
    return frozenset(float(n) for meta in _metadata(messages) for n in meta.get("threshold", []))


def narrowings_in(messages: Sequence[ModelMessage]) -> list[RangeNarrowing]:
    """Сужения `equity_vs_range` из выходов расчётов, в порядке разговора: по ним
    сужение следующей улицы обязано быть подмножеством прошлой
    (`test_a_turn_in_the_next_turn_is_bound_by_the_flop_of_the_history`). Что ушло из
    истории обрезкой, здесь не возвращается
    (`test_a_narrowing_trimmed_from_the_history_refuses_the_turn`)."""
    return [
        RangeNarrowing.model_validate(meta["narrowed"])
        for meta in _metadata(messages)
        if isinstance(meta.get("narrowed"), dict)
    ]


def hand_player_texts(history: Sequence[ModelMessage], analysis_id: int | None) -> list[str]:
    """Реплики игрока о руке `analysis_id` с конца истории до первой реплики другого хода:
    о другой руке, без разбора или без пометки руки — так же, как ставится отметка смены
    руки (`hand_boundary`). Без руки — пусто
    (`test_the_player_texts_of_the_hand_stop_at_the_last_boundary`)."""
    if analysis_id is None:
        return []
    texts: list[str] = []
    for message in reversed(history):
        parts = _player_parts(message)
        if not parts:
            continue
        tags = message.metadata if isinstance(message, ModelRequest) else None
        if not isinstance(tags, dict) or tags.get(HAND_KEY) != analysis_id:
            break
        texts.extend(_text(part) for part in parts)
    return texts


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
