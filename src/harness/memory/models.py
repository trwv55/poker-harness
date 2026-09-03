"""ORM-модели БД: 12 таблиц спеки §6.

Источник — `docs/superpowers/specs/2026-08-28-poker-harness-tech-spec-design.md`,
раздел "6. Схема БД". Колонки — по табличным строкам спеки дословно; `?` у поля в
спеке значит nullable, отсутствие `?` — `NOT NULL`. Три отступления от этого
правила и почему они не нарушают "дословно":

1. `llm_calls.started_at` — таблица §6 её не называет среди "ключевых полей", но §7
   описывает ровно эту колонку: лимитер пишет строку "при старте вызова", а окно
   темпа считает `WHERE started_at > now() - interval '60 seconds'` (задача 16,
   `PgLimiter`). Индекс `llm_calls(started_at)`, тоже из этой задачи, без колонки
   строить не на чем — молчание таблицы §6 здесь пробел терминологии, а не запрет.
2. `llm_calls.tokens_in/tokens_out/cost/latency_ms` — по терзости строки таблицы
   были бы `NOT NULL`, но §7 явно описывает запись строки ДО завершения вызова
   (status='started'), когда этих чисел ещё нет: "статус обновляется по
   завершении". Nullable — не отступление от спеки, а следствие второй её части.
3. `invites.id` — таблица §6 не пишет "id" явно (только "code UNIQUE, issued_by,
   ..."), но UNIQUE у `code` рядом с обычным набором остальных таблиц ("id,
   ключ UNIQUE, ...", ср. `players`) читается как тот же паттерн: суррогатный id
   плюс отдельно помеченный уникальный бизнес-ключ. `calc_cache.key` — образец
   противоположного случая: там натуральный ключ и есть PK, и спека его никак не
   помечает (UNIQUE избыточен для PK). Инвайты собраны по образцу `players`.

jsonb-колонки хранят `model_dump(mode="json")` пайплайн-контрактов (`RawHand`,
`CanonicalHand`, `EnrichedHand`, `AnalysisResult`) — уже JSON-совместимые
примитивы (datetime и StrEnum-ключи словарей сериализуются в строки), поэтому
кастомный сериализатор не нужен: `model_validate` того же контракта читает jsonb
обратно один в один (проверено на реальной руке из `tests/test_hh_parser.py`).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Text,
    false,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Конвенция имён constraint'ов: без неё alembic autogenerate придумывает имена
# индексов/ограничений сам на каждый прогон, и диффы миграций захламляются
# переименованиями, которых по факту не было.
_NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=_NAMING_CONVENTION)


class Player(Base):
    """Сквозной уровень: `players`. Израсходованное за день выводится из `jobs` (§9)."""

    __tablename__ = "players"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tg_user_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False)
    # NULL = у игрока нет персонального переопределения, действует дефолт из
    # Config (спека §7: "квоты по умолчанию" — конфиг, не схема БД).
    quota_daily: Mapped[int | None] = mapped_column(Integer)
    subscription: Mapped[str] = mapped_column(String(32), nullable=False, server_default="free")
    is_dev: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=false())


class Invite(Base):
    """Доступ по инвайтам: `invites`. `id` — см. пункт 3 в модульном докстринге."""

    __tablename__ = "invites"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    issued_by: Mapped[int] = mapped_column(BigInteger, ForeignKey("players.id"), nullable=False)
    used_by: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("players.id"))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Session(Base):
    """«Сессия = вечер»: `sessions`. Закрывается `/new` (SESSIONS_UX.md), не таймером."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    player_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("players.id"), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    title: Mapped[str] = mapped_column(String, nullable=False)


class Tournament(Base):
    """HH-вход: `tournaments`. Файл лежит на диске в volume, здесь — путь и сводка скана."""

    __tablename__ = "tournaments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("sessions.id"), nullable=False
    )
    source_file: Mapped[str] = mapped_column(String, nullable=False)
    scan_summary: Mapped[Any | None] = mapped_column(JSONB)


class Hand(Base):
    """Контракты руки: `hands`. Nullable-колонки canonical/enriched — чекпоинты (§8.2),
    а не отсутствие данных: пайплайн мог дойти до этой строки и остановиться здесь.
    """

    __tablename__ = "hands"
    __table_args__ = (
        CheckConstraint(
            "provenance IN ('hand_history', 'screenshot')", name="provenance_allowed"
        ),
        Index("ix_hands_session_id", "session_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("sessions.id"), nullable=False
    )
    tournament_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("tournaments.id")
    )
    provenance: Mapped[str] = mapped_column(String(16), nullable=False)
    image_hash: Mapped[str | None] = mapped_column(String)
    raw: Mapped[Any] = mapped_column(JSONB, nullable=False)
    canonical: Mapped[Any | None] = mapped_column(JSONB)
    enriched: Mapped[Any | None] = mapped_column(JSONB)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)


class Analysis(Base):
    """Выход ядра и изложения: `analyses`."""

    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    hand_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("hands.id"), nullable=False)
    result: Mapped[Any] = mapped_column(JSONB, nullable=False)
    verdict_text: Mapped[str | None] = mapped_column(Text)
    range_images: Mapped[Any | None] = mapped_column(JSONB)


class Note(Base):
    """Заметки на игроков: `notes`. Только через vision — в HH ники анонимны."""

    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    owner_player_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("players.id"), nullable=False
    )
    opponent_nick: Mapped[str] = mapped_column(String, nullable=False)
    color: Mapped[str] = mapped_column(String(32), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EvalCase(Base):
    """Eval-датасет: `eval_cases`. Копится сам — этажи 2 и 4 EVALS.md."""

    __tablename__ = "eval_cases"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('vision_field', 'verdict_confirm', 'verdict_dispute')",
            name="kind_allowed",
        ),
        CheckConstraint(
            "source IN ('escalation', 'disagree_button', 'manual')", name="source_allowed"
        ),
        Index("ix_eval_cases_kind", "kind"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    hand_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("hands.id"), nullable=False)
    field: Mapped[str | None] = mapped_column(String)
    ground_truth: Mapped[Any] = mapped_column(JSONB, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Job(Base):
    """Очередь + журнал: `jobs` (§8). `session_id NOT NULL` с первой миграции —
    результату всегда есть куда лечь (молчаливое создание сессии — обязанность bot,
    задача 19).
    """

    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(
            "type IN ('screenshot_analyze', 'hh_scan', 'deep_dive', 'eval_run')",
            name="type_allowed",
        ),
        CheckConstraint(
            "status IN ('queued', 'running', 'awaiting_user', 'done', 'failed')",
            name="status_allowed",
        ),
        Index("ix_jobs_status_priority_created_at", "status", "priority", "created_at"),
        Index("ix_jobs_player_id_status", "player_id", "status"),
        # Партиционный уникальный индекс — структурная гарантия «одна активная
        # задача на игрока» (спека §8.1), а не только фильтр NOT EXISTS в SQL
        # захвата (`platform/queue.py`, `_CLAIM_SQL`). NOT EXISTS под READ
        # COMMITTED видит только закоммиченное и не останавливает два конкурентных
        # claim() над РАЗНЫМИ queued-задачами одного игрока — гонка (задача 15,
        # fix round 1, 300/300 на реальном Postgres). Индекс делает эту гонку
        # физически невозможной при любом уровне изоляции и для любого будущего
        # кода, который выставит status='running' в обход claim().
        Index(
            "uq_jobs_player_id_running",
            "player_id",
            unique=True,
            postgresql_where=text("status = 'running'"),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="queued")
    payload: Mapped[Any] = mapped_column(JSONB, nullable=False)
    session_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("sessions.id"), nullable=False
    )
    player_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("players.id"), nullable=False)
    hand_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("hands.id"))
    # Дефолты ниже — параметры устойчивости очереди (задача 15), а не продуктовые
    # числа о деньгах: 0 попыток на старте, 3 — стандартный потолок ретраев джобы.
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="3")
    priority: Mapped[int] = mapped_column(Integer, nullable=False, server_default="100")
    locked_by: Mapped[str | None] = mapped_column(String)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)


class Trace(Base):
    """Что вернул каждый сервис: `traces`. Одна строка на джобу, `spans` копится по ходу."""

    __tablename__ = "traces"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    job_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("jobs.id"), nullable=False)
    hand_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("hands.id"))
    spans: Mapped[Any] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))


class LlmCall(Base):
    """Учёт себестоимости и точка данных лимитера: `llm_calls` (§7).

    `started_at` и nullable-числовые колонки — см. пункты 1 и 2 модульного
    докстринга: строка пишется при старте вызова, до того как токены/цена/латентность
    известны.
    """

    __tablename__ = "llm_calls"
    __table_args__ = (
        CheckConstraint(
            "purpose IN ('vision_extract', 'verdict_text')", name="purpose_allowed"
        ),
        CheckConstraint(
            "status IN ('started', 'ok', 'error', 'schema_error')", name="status_allowed"
        ),
        Index("ix_llm_calls_started_at", "started_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    trace_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("traces.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    purpose: Mapped[str] = mapped_column(String(32), nullable=False)
    tokens_in: Mapped[int | None] = mapped_column(Integer)
    tokens_out: Mapped[int | None] = mapped_column(Integer)
    cost: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="started")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class CalcCache(Base):
    """Кэш расчётов: `calc_cache`. Общий между турнирами и пользователями (§6):
    655х на прогретом кэше, ключ — сигнатура спота, а не конкретной раздачи.
    """

    __tablename__ = "calc_cache"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[Any] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


def async_session_factory(dsn: str) -> async_sessionmaker[AsyncSession]:
    """Фабрика асинхронных сессий на заданном DSN (`postgresql+asyncpg://...`).

    Один вызов на процесс: возвращённая фабрика открывает новое соединение из
    пула движка на каждый вызов `factory()` — то, что нужно и воркеру (задачи
    15/16 гонят конкурентные запросы с разных соединений), и `db_factory` в
    тестах (контроллерский рулинг задачи 14: `FOR UPDATE SKIP LOCKED` и
    `pg_try_advisory_lock` требуют разных соединений, видящих закоммиченное).
    """
    engine = create_async_engine(dsn)
    return async_sessionmaker(engine, expire_on_commit=False)
