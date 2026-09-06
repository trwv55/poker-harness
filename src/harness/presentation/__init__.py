"""Единый голос продукта (спека §4): весь текст и все кнопки, которые видит игрок.

Публичный API пакета — реэкспорт из `messages` (тексты) и `keyboards` (кнопки),
по тому же принципу, что и `harness.contracts`: вызывающий код (`bot`, `worker`)
импортирует из `harness.presentation`, а не из отдельных модулей.
"""

from __future__ import annotations

from harness.presentation.keyboards import (
    Btn,
    deep_dive_button,
    escalation_buttons,
    verdict_buttons,
)
from harness.presentation.messages import (
    Msg,
    ask_gg_nickname_msg,
    bot_failure_msg,
    button_not_ready_msg,
    deep_dive_msg,
    escalation_msg,
    failed_msg,
    gg_nickname_saved_msg,
    hh_accepted_msg,
    hh_duplicate_msg,
    new_session_msg,
    not_a_hand_msg,
    progress_text,
    quota_exceeded_msg,
    range_image_title,
    scan_summary_msg,
    send_as_file_msg,
    start_msg,
    tournament_report_msg,
    tournament_story_msg,
    unsupported_document_msg,
    vision_answer_not_a_number_msg,
    vision_answer_saved_msg,
    vision_gave_up_msg,
    vision_manual_entry_msg,
)

__all__ = [
    "Btn",
    "Msg",
    "ask_gg_nickname_msg",
    "bot_failure_msg",
    "button_not_ready_msg",
    "deep_dive_button",
    "deep_dive_msg",
    "escalation_buttons",
    "escalation_msg",
    "failed_msg",
    "gg_nickname_saved_msg",
    "hh_accepted_msg",
    "hh_duplicate_msg",
    "new_session_msg",
    "not_a_hand_msg",
    "progress_text",
    "quota_exceeded_msg",
    "range_image_title",
    "scan_summary_msg",
    "send_as_file_msg",
    "start_msg",
    "tournament_report_msg",
    "tournament_story_msg",
    "unsupported_document_msg",
    "verdict_buttons",
    "vision_answer_not_a_number_msg",
    "vision_answer_saved_msg",
    "vision_gave_up_msg",
    "vision_manual_entry_msg",
]
