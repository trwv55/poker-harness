"""Репозитории: единственное место, где пайплайн-контракты встречаются со строками БД.

Конструктор каждого репозитория берёт `AsyncSession` (аргумент `db` — то же имя, что у
одноимённой pytest-фикстуры) и работает в её транзакции. Методы делают `flush()`, но
никогда не `commit()`/`rollback()`: коммитить — дело вызывающего (в тестах — фикстуры
`db`/`db_factory`, в проде — `run_job`, задача 18). Если бы репозиторий коммитил сам,
транзакционный откат теста (`db`) не смог бы отменить его запись, и фикстуры перестали
бы быть чистыми между тестами.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from harness.analysis.scan import ScanSummary
from harness.contracts import AnalysisResult, CanonicalHand, EnrichedHand, Provenance, RawHand
from harness.memory.models import Analysis, CalcCache, EvalCase, Hand, Player, Tournament
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
        return self._to_record(record)

    async def list_by_tournament(self, tournament_id: int) -> list[HandRecord]:
        """Все руки турнира, в порядке вставки (задача 18: тот же порядок, в
        котором их отдал `parse_file` — резюме читает `hand_index` руки N по
        позиции N в этом списке, не переразбирая файл заново, см. `run_job`).
        """
        stmt = select(Hand).where(Hand.tournament_id == tournament_id).order_by(Hand.id)
        rows = (await self.db.scalars(stmt)).all()
        return [self._to_record(row) for row in rows]

    async def count_by_tournament(self, tournament_id: int) -> int:
        """Сколько рук турнира уже сохранены — дешёвая проверка резюме без
        десериализации jsonb в контракты (в отличие от `list_by_tournament`).
        """
        stmt = select(func.count()).select_from(Hand).where(Hand.tournament_id == tournament_id)
        return int(await self.db.scalar(stmt) or 0)

    async def find_by_hand_no(self, session_id: int, hand_no: str) -> HandRecord | None:
        """Рука по номеру раздачи внутри сессии — вход станции `deep_dive`
        (задача 18): кнопка «разобрать» под строкой скана несёт только
        `hand_no` (`keyboards.deep_dive_button`), не `hand_id`, поэтому найти
        строку `hands` можно только по значению внутри `raw` (колонки-номера
        у таблицы нет — заводить её ради одного запроса дороже, чем прочитать
        jsonb: `hand_no` уникален не глобально, а в рамках источника, и поиск
        по `raw`, а не `canonical`, работает даже до чекпоинта нормализации).
        """
        stmt = select(Hand).where(
            Hand.session_id == session_id,
            Hand.raw["hand_no"].astext == hand_no,
        )
        record = await self.db.scalar(stmt)
        return self._to_record(record) if record is not None else None

    def _to_record(self, record: Hand) -> HandRecord:
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


class TournamentsRepo:
    """`tournaments`: HH-вход и сводка скана (задача 18, встык с `hh_scan`).

    `create()` — единственный чекпоинт, которого не было в схеме до этой задачи:
    `tournament_id` уходит в `jobs.payload` сразу после вставки (§8.2, `run_job`),
    и повторная попытка той же задачи находит его там же, а не заводит вторую
    строку `tournaments` на тот же файл.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, *, session_id: int, source_file: str) -> int:
        record = Tournament(session_id=session_id, source_file=source_file)
        self.db.add(record)
        await self.db.flush()
        return record.id

    async def save_scan_summary(self, tournament_id: int, summary: ScanSummary) -> None:
        record = await self._get_row(tournament_id)
        record.scan_summary = summary.model_dump(mode="json")
        await self.db.flush()

    async def _get_row(self, tournament_id: int) -> Tournament:
        record = await self.db.get(Tournament, tournament_id)
        if record is None:
            raise LookupError(f"турнир {tournament_id} не найден")
        return record


class CalcCacheRepo:
    """`calc_cache`: кэш расчётов, общий между турнирами и пользователями (§6).

    Ключ — сигнатура спота (уровень классов рук и квантованных глубин), не
    конкретная раздача, поэтому одна и та же строка годится любому будущему
    скану, который посчитает тот же спот (задача 13: 655× на прогретом кэше).
    Значения детерминированы сидом сэмплера (`analysis.preflop`, `_EQUITY_MC_
    SEED`) — конфликтов при параллельной записи одного и того же ключа в
    принципе не бывает (два воркера, посчитавшие один спот, посчитают ОДНО и
    то же число), поэтому `ON CONFLICT DO NOTHING` дешевле и настолько же
    корректен, как `DO UPDATE`: переписывать существующую строку нечем и
    незачем.

    `prefix` (не часть ключа спота, а версия/отпечаток сэмплера —
    `analysis.preflop.equity_cache_fingerprint()`) отделяет один набор входов
    (число итераций Монте-Карло, сид) от другого: смена итераций не должна
    тихо отдать число, посчитанное на старых — старые строки просто перестают
    совпадать по префиксу и остаются неиспользуемым, но безвредным мусором
    (тот же приём, что у дискового кэша в `preflop.py`, перенесённый в общее
    хранилище — контроллерский рулинг задачи 18, п.1: без него холодные
    минуты скана возвращались бы при каждом масштабировании воркера).
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_all(self, prefix: str) -> dict[str, float]:
        # `autoescape=True` (fix round 1, Minor): без него `startswith()` рендерит
        # голый `LIKE 'prefix%'`, а `_`/`%`/сам escape-символ внутри `prefix`
        # остаются активными спецсимволами LIKE, а не буквальным текстом — `_`
        # это "любой один символ". С одним префиксом сейчас безвредно; со вторым
        # (например `nash_hu:` — уже с подчёркиванием) станет тихим совпадением
        # чужого пространства ключей на первом совпавшем символе.
        stmt = select(CalcCache.key, CalcCache.value).where(
            CalcCache.key.startswith(prefix, autoescape=True)
        )
        rows = (await self.db.execute(stmt)).all()
        return {key[len(prefix) :]: float(value) for key, value in rows}

    async def upsert_many(self, prefix: str, entries: Mapping[str, float]) -> None:
        if not entries:
            return
        values: list[dict[str, Any]] = [
            {"key": f"{prefix}{key}", "value": value} for key, value in entries.items()
        ]
        stmt = pg_insert(CalcCache).values(values).on_conflict_do_nothing(index_elements=["key"])
        await self.db.execute(stmt)
        await self.db.flush()


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
