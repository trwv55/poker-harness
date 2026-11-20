"""One transactional boundary for owned screenshot save/resume, not OCR/math."""
from __future__ import annotations

import hashlib

from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from harness.contracts import AnalysisResult
from harness.contracts.pokerdom_vision import PokerdomResult
from harness.memory.models import Hand, Job
from harness.memory.models import Session as SessionRow
from harness.memory.repos import AnalysesRepo, HandsRepo


async def image_lock(db: AsyncSession, player_id: int, image_hash: str) -> None:
    key = int.from_bytes(hashlib.sha256(f'vision:{player_id}:{image_hash}'.encode()).digest()[:8], signed=True)
    await db.execute(text('SELECT pg_advisory_xact_lock(:key)'), {'key': key})


async def owned_image(db: AsyncSession, player_id: int, image_hash: str) -> Hand | None:
    return await db.scalar(select(Hand).join(SessionRow, SessionRow.id == Hand.session_id).where(
        SessionRow.player_id == player_id, Hand.image_hash == image_hash,
        Hand.provenance == 'screenshot', Hand.enriched['hand']['hero_label'].astext.is_not(None),
    ).order_by(Hand.id).limit(1))


async def fenced_job(db: AsyncSession, job: Job) -> Job:
    row = await db.scalar(select(Job).where(Job.id == job.id, Job.player_id == job.player_id,
        Job.status == 'running', Job.locked_by == job.locked_by).with_for_update())
    if row is None:
        raise ValueError('vision worker lost job ownership')
    return row


async def save_reading(db: AsyncSession, job: Job, result: PokerdomResult) -> int:
    if result.read_status != 'ok' or result.replay_status != 'passed' or result.raw is None or result.enriched is None:
        raise ValueError('unvalidated vision reading cannot enter history')
    if result.raw.vision is None:
        raise ValueError('vision metadata missing')
    await image_lock(db, job.player_id, job.payload['image_sha256'])
    row = await fenced_job(db, job)
    owner = await db.scalar(select(SessionRow.player_id).where(SessionRow.id == row.session_id))
    if owner != job.player_id:
        raise ValueError('vision session is not owned')
    found = await owned_image(db, job.player_id, job.payload['image_sha256'])
    if found is not None:
        # A changed profile must never reinterpret an existing hand silently.
        if found.raw.get('vision', {}).get('nicknames', {}).get(result.enriched.hand.hero_label) != result.raw.vision.nicknames.get(result.enriched.hand.hero_label):
            raise ValueError('saved screenshot hero conflict; explicit correction required')
        hand_id = found.id
    else:
        hands = HandsRepo(db)
        hand_id = await hands.save_raw(session_id=owner_session(row), raw=result.raw)
        await hands.save_canonical(hand_id, result.enriched.hand)
        await hands.save_enriched(hand_id, result.enriched)
    row.hand_id = hand_id
    row.payload = {**row.payload, 'saved_hand_id': hand_id}
    await db.flush()
    return hand_id


def owner_session(job: Job) -> int:
    if job.session_id is None:
        raise ValueError('vision session missing')
    return job.session_id


async def save_analysis(db: AsyncSession, job: Job, hand_id: int, result: AnalysisResult) -> int:
    # Lock one hand, not a transaction spanning CPU work. Recheck after acquiring.
    hand = await db.scalar(select(Hand).join(SessionRow, SessionRow.id == Hand.session_id)
        .where(Hand.id == hand_id, SessionRow.player_id == job.player_id).with_for_update(of=Hand))
    if hand is None:
        raise ValueError('vision hand is not owned')
    row = await fenced_job(db, job)
    if row.hand_id != hand_id:
        raise ValueError('vision hand checkpoint mismatch')
    analyses = AnalysesRepo(db)
    existing = await analyses.get_by_hand(hand_id)
    if existing is not None:
        return existing.id
    enriched = (await HandsRepo(db).get(hand_id)).enriched
    if enriched is None:
        raise ValueError('enriched checkpoint missing')
    analysis_id = await analyses.save(hand_id=hand_id, result=result,
                                    decision_points=enriched.report.decision_points)
    row.payload = {**row.payload, 'saved_analysis_id': analysis_id}
    await db.flush()
    return analysis_id


async def source_is_referenced(db: AsyncSession, encoded_ref: str, ref_data: dict) -> bool:
    """History/manifests, pending intake and active or explicitly retryable jobs."""
    from harness.memory.models import Player
    if await db.scalar(select(Hand.id).where(or_(Hand.raw['source_ref'].astext == encoded_ref, Hand.raw['vision']['reading_ref'].astext == encoded_ref)).limit(1)):
        return True
    if await db.scalar(select(Job.id).where(Job.payload['source_ref'] == ref_data, or_(Job.status.in_(['queued', 'running', 'awaiting_user']), (Job.status == 'failed') & (Job.attempts < Job.max_attempts))).limit(1)):
        return True
    return await db.scalar(select(Player.id).where(Player.pending_input['source_ref'] == ref_data).limit(1)) is not None
