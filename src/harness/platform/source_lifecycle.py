"""Owned, reference-aware source compensation/recovery. No filesystem assumptions."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from sqlalchemy import or_, select

from harness.memory.models import Job
from harness.memory.vision import image_lock, source_is_referenced

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from harness.platform.source_storage import SourceRef, SourceStorage

ORPHAN_GRACE_S = 3600


async def release_source(db: AsyncSession, store: SourceStorage, owner_id: int, ref: SourceRef) -> bool:
    await image_lock(db, owner_id, ref.sha256)
    if await source_is_referenced(db, ref.encode(), ref.model_dump()):
        return False
    # Only terminal, non-retryable refs can be detached; active/retryable refs
    # were checked under the same image lock used by admission.
    rows = await db.scalars(select(Job).where(Job.player_id == owner_id,
        Job.payload['source_ref'] == ref.model_dump(),
        or_(Job.status == 'done', (Job.status == 'failed') & (Job.attempts >= Job.max_attempts))).with_for_update())
    for row in rows:
        row.payload = {k: v for k, v in row.payload.items() if k != 'source_ref'} | {'source_released': True}
    await db.flush()
    store.delete(owner_id, ref, ref.sha256)
    return True


async def recover_sources(db: AsyncSession, store: SourceStorage, owner_id: int,
                          before: float | None = None) -> int:
    """Lazy grace recovery on the next owned intake. Optional adapter capability.

    No TTL for history. Missing capability means retain, never filesystem fallback.
    """
    candidates = getattr(store, 'orphan_candidates', None)
    if candidates is None:
        return 0
    count = 0
    for ref in candidates(owner_id, time.time() - ORPHAN_GRACE_S if before is None else before):
        count += await release_source(db, store, owner_id, ref)
    return count
