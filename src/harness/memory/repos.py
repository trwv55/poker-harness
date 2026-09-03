"""Репозитории: единственное место, где пайплайн-контракты встречаются со строками БД.

Конструктор каждого репозитория берёт `AsyncSession` (аргумент `db` — то же имя, что у
одноимённой pytest-фикстуры) и работает в её транзакции. Методы делают `flush()`, но
никогда не `commit()`/`rollback()`: коммитить — дело вызывающего (в тестах — фикстуры
`db`/`db_factory`, в проде — `run_job`, задача 18). Если бы репозиторий коммитил сам,
транзакционный откат теста (`db`) не смог бы отменить его запись, и фикстуры перестали
бы быть чистыми между тестами.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from harness.contracts import AnalysisResult, CanonicalHand, EnrichedHand, Provenance, RawHand
from harness.memory.models import Analysis, EvalCase, Hand, Player
from harness.memory.models import Session as SessionRow

_MONTHS_RU_ABBR = (
    "янв",
    "фев",
    "мар",
    "апр",
    "май",
    "июн",
    "июл",
    "авг",
    "сен",
    "окт",
    "ноя",
    "дек",
)


def _session_title(moment: datetime) -> str:
    """«Сессия 20 авг» — формат ровно тот, что задан примером в SESSIONS_UX.md."""
    return f"Сессия {moment.day} {_MONTHS_RU_ABBR[moment.month - 1]}"


class PlayersRepo:
    """`players`: `tg_user_id` уникален в БД, поэтому "get or create" ищет по нему."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_or_create(self, tg_user_id: int) -> Player:
        player = await self.db.scalar(select(Player).where(Player.tg_user_id == tg_user_id))
        if player is not None:
            return player
        player = Player(tg_user_id=tg_user_id)
        self.db.add(player)
        await self.db.flush()
        return player


class SessionsRepo:
    """`sessions`: молчаливый путь из SESSIONS_UX.md — скрин без активной сессии не
    отказывает игроку, а тихо открывает новую сессию. "Активная" — самая свежая
    строка игрока с `closed_at IS NULL`.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def active_or_create(self, player_id: int) -> SessionRow:
        stmt = (
            select(SessionRow)
            .where(SessionRow.player_id == player_id, SessionRow.closed_at.is_(None))
            .order_by(SessionRow.started_at.desc())
            .limit(1)
        )
        active = await self.db.scalar(stmt)
        if active is not None:
            return active
        now = datetime.now(UTC)
        record = SessionRow(player_id=player_id, started_at=now, title=_session_title(now))
        self.db.add(record)
        await self.db.flush()
        return record


@dataclass(frozen=True, slots=True)
class HandRecord:
    """Артефакты одной руки, десериализованные обратно в пайплайн-контракты.

    `canonical`/`enriched` отсутствуют (`None`), пока пайплайн не дошёл до этого
    чекпоинта (§8.2) — это не ошибка чтения, а нормальное промежуточное состояние.
    """

    id: int
    session_id: int
    tournament_id: int | None
    provenance: Provenance
    image_hash: str | None
    schema_version: int
    raw: RawHand
    canonical: CanonicalHand | None
    enriched: EnrichedHand | None


class HandsRepo:
    """`hands`: raw/canonical/enriched — три чекпоинта одной строки, не три таблицы."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def save_raw(
        self, *, session_id: int, raw: RawHand, tournament_id: int | None = None
    ) -> int:
        image_hash = raw.vision.image_hash if raw.vision is not None else None
        record = Hand(
            session_id=session_id,
            tournament_id=tournament_id,
            provenance=raw.provenance.value,
            image_hash=image_hash,
            raw=raw.model_dump(mode="json"),
            schema_version=raw.schema_version,
        )
        self.db.add(record)
        await self.db.flush()
        return record.id

    async def save_canonical(self, hand_id: int, canonical: CanonicalHand) -> None:
        record = await self._get_row(hand_id)
        record.canonical = canonical.model_dump(mode="json")
        await self.db.flush()

    async def save_enriched(self, hand_id: int, enriched: EnrichedHand) -> None:
        record = await self._get_row(hand_id)
        record.enriched = enriched.model_dump(mode="json")
        await self.db.flush()

    async def get(self, hand_id: int) -> HandRecord:
        record = await self._get_row(hand_id)
        return HandRecord(
            id=record.id,
            session_id=record.session_id,
            tournament_id=record.tournament_id,
            provenance=Provenance(record.provenance),
            image_hash=record.image_hash,
            schema_version=record.schema_version,
            raw=RawHand.model_validate(record.raw),
            canonical=(
                CanonicalHand.model_validate(record.canonical)
                if record.canonical is not None
                else None
            ),
            enriched=(
                EnrichedHand.model_validate(record.enriched)
                if record.enriched is not None
                else None
            ),
        )

    async def _get_row(self, hand_id: int) -> Hand:
        record = await self.db.get(Hand, hand_id)
        if record is None:
            raise LookupError(f"рука {hand_id} не найдена")
        return record


@dataclass(frozen=True, slots=True)
class AnalysisRecord:
    id: int
    hand_id: int
    result: AnalysisResult
    verdict_text: str | None
    range_images: list[str] | None


class AnalysesRepo:
    """`analyses`: выход ядра (`result`) и изложения (`verdict_text`, `range_images`)."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def save(
        self,
        *,
        hand_id: int,
        result: AnalysisResult,
        verdict_text: str | None = None,
        range_images: list[str] | None = None,
    ) -> int:
        record = Analysis(
            hand_id=hand_id,
            result=result.model_dump(mode="json"),
            verdict_text=verdict_text,
            range_images=range_images,
        )
        self.db.add(record)
        await self.db.flush()
        return record.id

    async def get_by_hand(self, hand_id: int) -> AnalysisRecord | None:
        record = await self.db.scalar(select(Analysis).where(Analysis.hand_id == hand_id))
        if record is None:
            return None
        return AnalysisRecord(
            id=record.id,
            hand_id=record.hand_id,
            result=AnalysisResult.model_validate(record.result),
            verdict_text=record.verdict_text,
            range_images=record.range_images,
        )


class EvalCasesRepo:
    """`eval_cases`: копится сам — подтверждения/возражения игрока, эскалации vision."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def add(
        self,
        *,
        kind: str,
        hand_id: int,
        ground_truth: dict,
        source: str,
        field: str | None = None,
    ) -> int:
        record = EvalCase(
            kind=kind, hand_id=hand_id, field=field, ground_truth=ground_truth, source=source
        )
        self.db.add(record)
        await self.db.flush()
        return record.id
