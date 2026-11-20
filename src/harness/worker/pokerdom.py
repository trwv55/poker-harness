"""Validated screenshot → permanent owned hand, real analysis and discussion context."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from harness.contracts.pokerdom_vision import (
    PokerdomResult,
    PokerdomTelegramReading,
    decode_pokerdom_reading,
)
from harness.memory.repos import AnalysesRepo, HandsRepo, SessionsRepo
from harness.memory.vision import (
    fenced_job,
    owned_image,
    owner_session,
    save_analysis,
    save_reading,
)
from harness.parsers.pokerdom_vision import TG_PROMPT_PATH, process_pokerdom, read_pokerdom
from harness.platform.source_storage import LocalSourceStorage, SourceRef, image_mime
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


async def run_pokerdom(job: Job, deps: Deps, trace: Trace) -> int | None:
    from harness.worker.pipeline import _chat_id, _send_idempotent
    root = deps.data_dir
    if root is None:
        raise ValueError('source storage data root not configured')
    store = deps.source_storage or LocalSourceStorage(root)
    if 'source_ref' in job.payload:
        ref = SourceRef.model_validate(job.payload['source_ref'])
    else:
        ref = LocalSourceStorage(root).legacy(job.player_id, Path(job.payload['source_file']), job.payload['image_sha256'])
    image = store.read(job.player_id, ref, 10 * 1024 * 1024)
    if ref.sha256 != job.payload['image_sha256'] or image_mime(image) != ref.mime:
        raise ValueError('image hash or MIME mismatch')
    # Diagnostics may be local; source bytes are always accessed through the port.
    directory = root / 'vision-jobs' / str(job.id)
    if 'source_file' in job.payload:
        legacy_directory = Path(job.payload['source_file']).parent / f'job-{job.id}'
        if legacy_directory.is_dir():
            directory = legacy_directory
    directory.mkdir(parents=True, exist_ok=True)
    superseded_ref = None
    async with deps.db_factory() as db:
        row = await fenced_job(db, job)
        if row.session_id is None:
            # Bind once. Never assign an old unbound job to a newer evening.
            from sqlalchemy import select

            from harness.memory.models import Session as SessionRow
            session = await db.scalar(select(SessionRow).where(
                SessionRow.player_id == job.player_id, SessionRow.started_at <= row.created_at,
                SessionRow.closed_at.is_(None)).order_by(SessionRow.id.desc()).limit(1))
            if session is None:
                if await db.scalar(select(SessionRow.id).where(SessionRow.player_id == job.player_id).limit(1)) is not None:
                    raise ValueError('legacy image predates current session; upload again')
                session = await SessionsRepo(db).active_or_create(job.player_id)
            row.session_id = session.id
        if 'context_revision' not in row.payload:
            revision = await SessionsRepo(db).select_context(row.session_id, job.player_id)
            row.payload = {**row.payload, 'context_revision': revision}
        row.payload = {**row.payload, 'source_ref': ref.model_dump()}
        job.session_id = row.session_id
        job.payload = dict(row.payload)
        found = await owned_image(db, job.player_id, ref.sha256)
        saved = None if found is None else await HandsRepo(db).get(found.id)
        if saved is not None:
            assert saved.enriched is not None
            saved_source = SourceRef.model_validate_json(saved.raw.source_ref)
            saved_bytes = store.read(job.player_id, saved_source, 10 * 1024 * 1024)
            if saved_source.sha256 != ref.sha256 or image_mime(saved_bytes) != saved_source.mime:
                raise ValueError('saved screenshot source mismatch; recovery required')
            if ref != saved_source:
                superseded_ref = ref
                ref = saved_source
                row.payload = {**row.payload, 'source_ref': ref.model_dump()}
                job.payload = dict(row.payload)
            nicknames = saved.raw.vision.nicknames if saved.raw.vision else {}
            if nicknames.get(saved.enriched.hand.hero_label) != job.payload['nickname']:
                raise ValueError('saved screenshot hero conflict; explicit correction required')
        await db.commit()
    if superseded_ref is not None:
        from harness.platform.source_lifecycle import release_source
        async with deps.db_factory() as db:
            await release_source(db, store, job.player_id, superseded_ref)
            await db.commit()
    cached_result = None
    if saved is not None:
        manifest_ref = saved.raw.vision.reading_ref if saved.raw.vision else None
        if not manifest_ref:
            raise ValueError('saved screenshot reading manifest missing; recovery required')
        manifest_data = json.loads(store.read(job.player_id, SourceRef.model_validate_json(manifest_ref), 20 * 1024 * 1024))
        cached_result = PokerdomResult.model_validate(manifest_data.get('result', manifest_data))
    reading_path = directory / 'reading.json'
    result_path = directory / 'result.json'
    async with deps.db_factory() as db:
        chat_id = await _chat_id(db, job.player_id)
        await _send_idempotent(deps, db, job.id, job.locked_by, 'progress_message_id', chat_id,
                               Msg(text="Читаю таблицу истории…"))
        await db.commit()
    if cached_result is not None:
        result = cached_result
    elif result_path.exists():
        result = PokerdomResult.model_validate_json(result_path.read_text())
    else:
        if reading_path.exists():
            reading = decode_pokerdom_reading(json.loads(reading_path.read_text()))
        else:
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
                'automatic_retries': False, 'cache': False, 'automatic_history': True,
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
    analysis_error = False
    hand_id = None
    if result.read_status == 'ok' and result.replay_status == 'passed' and result.raw is not None and result.enriched is not None:
        if cached_result is None:
            result.raw.source_ref = ref.encode()
            assert result.raw.vision is not None
            result.raw.vision.image_hash = ref.sha256
            # The manifest holds observations and conditional checks, not fabricated
            # source payout/showdown records. Preserve original response files too.
            request_path = directory / 'manifest.json'
            manifest_data = {'version': 1, 'result': result.model_dump(mode='json'),
                'request': json.loads(request_path.read_text()) if request_path.exists() else {}}
            manifest = store.put(job.player_id, json.dumps(manifest_data, ensure_ascii=False).encode(), 'application/json', f'reading:{job.id}')
            result.raw.vision.reading_ref = manifest.encode()
        async with deps.db_factory() as db:
            hand_id = await save_reading(db, job, result)
            await db.commit()
            existing = await AnalysesRepo(db).get_by_hand(hand_id)
            await db.commit()
        try:
            if existing is None:
                from harness.worker.pipeline import _analyze_hand_with_cache
                analysis, _delta = await asyncio.get_running_loop().run_in_executor(
                    deps.process_pool, _analyze_hand_with_cache, result.enriched, {})
                async with deps.db_factory() as db:
                    analysis_id = await save_analysis(db, job, hand_id, analysis)
                    await db.commit()
            else:
                analysis_id = existing.id
            async with deps.db_factory() as db:
                await fenced_job(db, job)
                await SessionsRepo(db).publish_analysis(owner_session(job), job.player_id,
                    job.payload['context_revision'], analysis_id)
                await db.commit()
        except Exception as exc:  # noqa: BLE001 — retain validated checkpoint; diagnose analysis failure
            analysis_error = True
            trace.record('pokerdom_analysis_unavailable', error_type=type(exc).__name__, error=str(exc))
    trace.record('pokerdom_read', read_status=result.read_status,
                 replay_status=result.replay_status, outcome_confirmed=result.outcome_confirmed,
                 artifacts=str(directory), persistent_hand=hand_id is not None, hand_id=hand_id)
    # Each chunk has its own checkpoint; never silently truncate a transcript.
    for i, msg in enumerate(pokerdom_messages(result)):
        async with deps.db_factory() as db:
            # The common fenced sender supports distinct checkpoint keys.
            await _send_idempotent(deps, db, job.id, job.locked_by,
                                   f'pokerdom_result_{i}', chat_id, msg)
            await db.commit()
    if analysis_error:
        async with deps.db_factory() as db:
            await _send_idempotent(deps, db, job.id, job.locked_by,
                'pokerdom_analysis_error', chat_id, Msg(text='Рука сохранена. Расчёт для обсуждения временно недоступен; повторите этот файл позже.'))
            await db.commit()
    return hand_id
