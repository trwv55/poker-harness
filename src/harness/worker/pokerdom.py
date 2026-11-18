"""Temporary vision job; no repositories for gaming history are used here."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from harness.contracts.pokerdom_vision import (
    PokerdomResult,
    PokerdomTelegramReading,
    decode_pokerdom_reading,
)
from harness.parsers.pokerdom_vision import TG_PROMPT_PATH, process_pokerdom, read_pokerdom
from harness.presentation import Msg
from harness.presentation.pokerdom_vision import pokerdom_messages

if TYPE_CHECKING:
    from harness.memory.models import Job
    from harness.platform.trace import Trace
    from harness.worker.pipeline import Deps


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    temporary.replace(path)


async def run_pokerdom(job: Job, deps: Deps, trace: Trace) -> None:
    from harness.worker.pipeline import _chat_id, _send_idempotent
    image_path = Path(job.payload['source_file'])
    # Private service artifacts, separate from hands/analyses/sessions/eval truth.
    directory = image_path.parent / f'job-{job.id}'
    directory.mkdir(parents=True, exist_ok=True)
    reading_path = directory / 'reading.json'
    result_path = directory / 'result.json'
    async with deps.db_factory() as db:
        chat_id = await _chat_id(db, job.player_id)
        await _send_idempotent(deps, db, job.id, job.locked_by, 'progress_message_id', chat_id,
                               Msg(text="Читаю таблицу истории…"))
        await db.commit()
    if result_path.exists():
        result = PokerdomResult.model_validate_json(result_path.read_text())
    else:
        if reading_path.exists():
            reading = decode_pokerdom_reading(json.loads(reading_path.read_text()))
        else:
            image = image_path.read_bytes()
            if hashlib.sha256(image).hexdigest() != job.payload['image_sha256']:
                raise ValueError('image hash mismatch')
            if deps.pokerdom_llm is None:
                raise RuntimeError('Pokerdom model not configured')
            _atomic_json(directory / 'manifest.json', {
                'schema_version': 2,
                'schema_sha256': hashlib.sha256(json.dumps(PokerdomTelegramReading.model_json_schema(), sort_keys=True).encode()).hexdigest(),
                'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                    Path(__file__), TG_PROMPT_PATH.parent.parent / 'pokerdom_vision.py',
                    Path(__file__).parents[1] / 'contracts' / 'pokerdom_vision.py',
                )},
                'prompt_sha256': hashlib.sha256(TG_PROMPT_PATH.read_bytes()).hexdigest(),
                'image_sha256': job.payload['image_sha256'], 'host': deps.pokerdom_provider,
                'automatic_retries': False, 'cache': False, 'save_requested': False,
                'original_document': job.payload['original_document'],
            })
            reading, meta = await read_pokerdom(
                deps.pokerdom_llm, image, trace_id=trace.trace_id,
                provider=deps.pokerdom_provider, detect_format=True,
                observer=lambda answer: _atomic_json(directory / 'response.json', answer),
            )
            # Durable checkpoint BEFORE adaptation/send. Resume never rereads the model.
            _atomic_json(reading_path, reading.model_dump(mode='json'))
            _atomic_json(directory / 'call-meta.json', {'meta': str(meta)})
        observation = getattr(reading, 'format_observation', None)
        markers = ('player_rows', 'street_columns', 'numbered_actions', 'known_tokens',
                   'board_area', 'end_street_pots')
        accepted = (observation is not None and observation.kind == 'numbered_hh_table'
                    and bool(observation.evidence)
                    and all(getattr(observation, k) for k in markers))
        missing_pots = sorted({e.street.value for e in reading.events
                               if reading.pot_after_street.get(e.street) is None})
        if accepted and (missing_pots or not reading.events):
            result = PokerdomResult(reading=reading, read_status='incomplete', errors=[
                'Не прочитаны все номерные действия или банк сыгранной улицы: ' + ', '.join(missing_pots)
            ])
        elif accepted:
            result = process_pokerdom(reading, nickname=job.payload['nickname'])
        else:
            result = PokerdomResult(reading=reading, read_status='not_hand', errors=[
                'Не подтверждён поддерживаемый формат полной таблицы истории. Пришлите таблицу с игроками, номерами действий, улицами и Total Pot.'
            ])
        _atomic_json(result_path, result.model_dump(mode='json'))
    trace.record('pokerdom_read', read_status=result.read_status,
                 replay_status=result.replay_status, outcome_confirmed=result.outcome_confirmed,
                 artifacts=str(directory), persistent_hand=False)
    # Each chunk has its own checkpoint; never silently truncate a transcript.
    for i, msg in enumerate(pokerdom_messages(result)):
        async with deps.db_factory() as db:
            # The common fenced sender supports distinct checkpoint keys.
            await _send_idempotent(deps, db, job.id, job.locked_by,
                                   f'pokerdom_result_{i}', chat_id, msg)
            await db.commit()
