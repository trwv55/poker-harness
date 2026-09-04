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
    deep_dive_msg,
    escalation_msg,
    failed_msg,
    hh_accepted_msg,
    new_session_msg,
    photo_soon_msg,
    progress_text,
    quota_exceeded_msg,
    scan_summary_msg,
    start_msg,
    unsupported_document_msg,
)

__all__ = [
    "Btn",
    "Msg",
    "deep_dive_button",
    "deep_dive_msg",
    "escalation_buttons",
    "escalation_msg",
    "failed_msg",
    "hh_accepted_msg",
    "new_session_msg",
    "photo_soon_msg",
    "progress_text",
    "quota_exceeded_msg",
    "scan_summary_msg",
    "start_msg",
    "unsupported_document_msg",
    "verdict_buttons",
]
