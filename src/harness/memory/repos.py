"""Репозитории: единственное место, где пайплайн-контракты встречаются со строками БД.

Конструктор каждого репозитория берёт `AsyncSession` (аргумент `db` — то же имя, что у
одноимённой pytest-фикстуры) и работает в её транзакции. Методы делают `flush()`, но
никогда не `commit()`/`rollback()`: коммитить — дело вызывающего (в тестах — фикстуры
`db`/`db_factory`, в проде — `run_job`, задача 18). Если бы репозиторий коммитил сам,
транзакционный откат теста (`db`) не смог бы отменить его запись, и фикстуры перестали
бы быть чистыми между тестами.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from harness.contracts import (
    AnalysisResult,
    CanonicalHand,
    EnrichedHand,
    Provenance,
    RawHand,
    ScanSummary,
)
from harness.memory.models import Analysis, CalcCache, EvalCase, Hand, Job, Player, Tournament
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
        """Найти игрока или завести — безопасно при гонке (fix round 1 задачи 19).

        «Прочитали — не нашли — вставили» перестало быть безобидным, как только
        появился первый вызывающий с настоящей конкурентностью: два файла,
        присланных незнакомым игроком подряд, обрабатываются двумя транзакциями
        сразу, обе не находят строку и обе вставляют — вторая получает
        `IntegrityError` на `players.tg_user_id`, и обработчик падает на ровном
        месте. `ON CONFLICT DO NOTHING` превращает это в ноль вернувшихся строк:
        проигравший ждёт коммита победителя (Postgres блокирует его на самом
        конфликте), после чего просто перечитывает готовую строку — под READ
        COMMITTED она ему уже видна.
        """
        player = await self.db.scalar(select(Player).where(Player.tg_user_id == tg_user_id))
        if player is not None:
            return player
        created_id = await self.db.scalar(
            pg_insert(Player)
            .values(tg_user_id=tg_user_id)
            .on_conflict_do_nothing(index_elements=["tg_user_id"])
            .returning(Player.id)
        )
        if created_id is None:
            player = await self.db.scalar(select(Player).where(Player.tg_user_id == tg_user_id))
            if player is None:  # pragma: no cover — конфликт был, а строки нет
                raise LookupError(f"игрок tg_user_id={tg_user_id} исчез после конфликта вставки")
            return player
        created = await self.db.scalar(select(Player).where(Player.id == created_id))
        if created is None:  # pragma: no cover — только что вставленная строка
            raise LookupError(f"игрок {created_id} не найден сразу после вставки")
        return created


# Пространство имён advisory-локов для `sessions` (первый аргумент двухаргументной
# формы — тот же приём, что `0x4C4C4D` у лимитера): ключ лока — пара
# (это пространство, player_id), поэтому за один и тот же КЛЮЧ с локами
# `platform/limiter.py` он не борется ни при каком player_id.
#
# **Пространства имён — не вся защита, и это существенно (round 5, Item M).**
# Лимитер на каждом входе в `slot()` выполняет `SELECT pg_advisory_unlock_all()`
# на соединении из ТОГО ЖЕ движка приложения, а `unlock_all` пространств имён не
# различает вовсе: он снимает все advisory-локи УРОВНЯ СЕССИИ, которые держит то
# соединение, чьи бы они ни были. Разные ключи от него не спасают.
#
# Спасает то, что этот лок — `pg_advisory_XACT_lock`: транзакционные локи живут
# отдельно, снимаются коммитом и для `pg_advisory_unlock_all()` невидимы. Отсюда
# инвариант всей системы, а не одного этого файла:
#
#     на движке приложения никто не берёт advisory-лок УРОВНЯ СЕССИИ, кроме
#     самого `PgLimiter` — тот держит для этого собственное закреплённое
#     соединение и сам же убирает за собой.
#
# Правка `pg_advisory_xact_lock` → `pg_advisory_lock` выглядела бы безобидной и
# по-прежнему согласовывалась бы с абзацем про пространства имён — а
# сериализация сессий тихо перестала бы работать при первом же вызове модели.
# Поэтому инвариант держит не комментарий, а тест:
# `test_sessions_lock_is_transaction_scoped` (tests/test_memory.py).
_SESSIONS_LOCK_NS = 0x53455353  # "SESS"


class SessionsRepo:
    """`sessions`: молчаливый путь из SESSIONS_UX.md — скрин без активной сессии не
    отказывает игроку, а тихо открывает новую сессию. "Активная" — самая свежая
    строка игрока с `closed_at IS NULL`.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def active_or_create(self, player_id: int) -> SessionRow:
        """Активная сессия игрока или новая. Сериализовано по игроку (fix round 1).

        У `sessions` нет уникального ключа, на который можно было бы повесить
        `ON CONFLICT` (одному игроку положено много сессий за жизнь), поэтому
        гонка «прочитали — не нашли — вставили» здесь не падает, а тихо
        расходится: два одновременных файла открыли бы ДВЕ сессии, и вечер игрока
        распался бы на два контейнера. Транзакционный advisory-лок по игроку
        (снимается коммитом; у лимитера в `platform/limiter.py` инструмент
        РОДСТВЕННЫЙ, но не тот же — там лок уровня СЕССИИ, и разница между ними
        несущая: см. комментарий к `_SESSIONS_LOCK_NS` выше) выстраивает такие
        транзакции в очередь: второй читает уже закоммиченную сессию первого.
        Ждут друг друга только транзакции ОДНОГО игрока — на чужие уплаты этот
        лок не влияет.
        """
        await self.db.execute(
            text("SELECT pg_advisory_xact_lock(:ns, :player_id)"),
            {"ns": _SESSIONS_LOCK_NS, "player_id": player_id},
        )
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

    async def close_active(self, player_id: int) -> bool:
        """Закрыть открытые сессии игрока; вернуть, было ли что закрывать.

        Это первая половина `/new` (задача 19): вторая — `active_or_create()`,
        который после закрытия неизбежно откроет новую. Отдельный метод, а не
        параметр `active_or_create`, потому что закрытие — самостоятельное
        событие с самостоятельным ответом игроку («предыдущая закрыта» говорится
        только когда предыдущая была).

        Закрываются ВСЕ открытые, а не только самая свежая: `active_or_create()`
        считает активной последнюю по `started_at`, поэтому вторая забытая
        открытая строка навсегда осталась бы невидимым мусором, который никакой
        `/new` больше не тронет. В норме она одна — инвариант поддерживается
        именно здесь.
        """
        result = await self.db.execute(
            update(SessionRow)
            .where(SessionRow.player_id == player_id, SessionRow.closed_at.is_(None))
            .values(closed_at=datetime.now(UTC))
            .execution_options(synchronize_session=False)
            .returning(SessionRow.id)
        )
        closed = result.all()
        await self.db.flush()
        return bool(closed)


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

    async def find_session_by_hand_no(self, player_id: int, hand_no: str) -> int | None:
        """В какой сессии ЭТОГО игрока лежит раздача с таким номером (fix round 1).

        Кнопка «разобрать» несёт только `hand_no` (`keyboards.deep_dive_button`),
        а разбор ищет руку как `find_by_hand_no(job.session_id, hand_no)` — значит
        задачу надо ставить в ту сессию, где рука ЛЕЖИТ, а не в ту, что сейчас
        активна. Иначе нажатие под сводкой вечера, закрытого командой `/new`,
        обречено на честный, но бессмысленный отказ.

        Область поиска — строго сессии этого игрока (JOIN по `sessions.player_id`):
        `hand_no` уникален в рамках источника, а не глобально, и без этого условия
        номер одного игрока мог бы разрешиться в чужую сессию.

        Одна и та же раздача может лежать в нескольких сессиях игрока (тот же файл
        загружен второй раз в другой вечер) — берём самую свежую по `hands.id`:
        содержимое раздачи идентично, поэтому выбор безопасен, а свежая сессия
        ближе к тому, на что игрок сейчас смотрит.
        """
        stmt = (
            select(Hand.session_id)
            .join(SessionRow, SessionRow.id == Hand.session_id)
            .where(SessionRow.player_id == player_id, Hand.raw["hand_no"].astext == hand_no)
            .order_by(Hand.id.desc())
            .limit(1)
        )
        return await self.db.scalar(stmt)

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

    async def find_in_session(self, session_id: int, source_file: str) -> int | None:
        """Турнир этого файла в этой сессии, если он уже заведён (fix round 1).

        Имя файла — sha256 содержимого (`bot/handlers.py`), поэтому совпадение
        пути значит совпадение байтов, а не просто похожее имя. Нужно, чтобы
        повторная загрузка не заводила вторую строку `tournaments` на тот же
        файл: продолжать разбор в уже существующей — это ещё и чекпоинты
        (`_run_hh_scan` пропускает руки, которые в ней уже сохранены).
        """
        stmt = select(Tournament.id).where(
            Tournament.session_id == session_id, Tournament.source_file == source_file
        )
        return await self.db.scalar(stmt)

    async def save_scan_summary(self, tournament_id: int, summary: ScanSummary) -> None:
        record = await self._get_row(tournament_id)
        record.scan_summary = summary.model_dump(mode="json")
        await self.db.flush()

    async def _get_row(self, tournament_id: int) -> Tournament:
        record = await self.db.get(Tournament, tournament_id)
        if record is None:
            raise LookupError(f"турнир {tournament_id} не найден")
        return record


class JobsRepo:
    """Чтение `jobs` для решений бота. Записью и жизненным циклом задач владеет
    `platform/queue.py` — сюда попадают только вопросы, на которые надо ответить
    внутри чужой, уже открытой транзакции (у очереди каждый метод открывает свою).
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def last_scan_status(self, session_id: int, source_file: str) -> str | None:
        """Статус последней задачи `hh_scan` по этому файлу в этой сессии; `None`
        — такой задачи не было.

        Поиск по `payload["source_file"]`, а не по колонке: связи `jobs` →
        `tournaments` в схеме нет, а путь в payload и есть то, по чему воркер
        читает файл. Тот же идиом обращения к jsonb, что у
        `HandsRepo.find_by_hand_no` (`raw["hand_no"].astext`).
        """
        stmt = (
            select(Job.status)
            .where(
                Job.session_id == session_id,
                Job.type == "hh_scan",
                Job.payload["source_file"].astext == source_file,
            )
            .order_by(Job.id.desc())
            .limit(1)
        )
        return await self.db.scalar(stmt)


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


# Квота по умолчанию (спека §9, пример дословно: «разборов 17/50 за 24 ч») — действует,
# когда у игрока нет персонального переопределения (`players.quota_daily IS NULL`).
QUOTA_DAILY_DEFAULT = 50

# Квоту тратят только ИНТЕРАКТИВНЫЕ задачи (спека §9). `hh_scan` в список не входит
# намеренно: он дёшев для нас и «поощряется щедрее» — это продуктовый рычаг
# (SCALING.md), а не недосмотр.
QUOTA_INTERACTIVE_JOB_TYPES = ("deep_dive", "screenshot_analyze")

# Скользящее окно, а не календарные сутки (спека §9 дословно: ни поля пояса, ни
# cron-сброса, ни полуночного «обнуления посреди ночной сессии»).
QUOTA_WINDOW = timedelta(hours=24)


@dataclass(frozen=True, slots=True)
class QuotaCheck:
    """Решение о допуске плюс числа для подписи сообщения.

    `hours_to_free` осмыслен только при `allowed is False` (иначе 0): это время
    до момента, когда самая старая задача В ОКНЕ из него выпадет и освободит
    место — то самое «через сколько», которое показывает `quota_exceeded_msg`.
    """

    allowed: bool
    left: int
    total: int
    hours_to_free: int


class QuotaRepo:
    """Квота игрока за скользящие 24 ч (спека §9) — ОДНА реализация на два процесса.

    Бот спрашивает «пускать ли» ДО постановки задачи в очередь (задача 19), воркер
    берёт отсюда же числа для строки «разборов X/Y за 24 ч» (задача 18) — он уже
    взял задачу в работу и отказывать не вправе. Соблазн держать по счётчику в
    каждом процессе этот проект уже наказывал (второй словарь словоформ, четыре
    расхождения с библиотекой по памяти): два SQL с одинаковым смыслом разошлись
    бы молча, и игрок увидел бы «осталось 3» ровно там, где ему отказали.

    Расход выводится из `jobs` (спека §6: «израсходованное выводится из jobs»), а
    не хранится счётчиком в `players`: счётчик пришлось бы сбрасывать по
    расписанию — ровно то, чего скользящее окно и не должно требовать.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def check(self, player_id: int) -> QuotaCheck:
        now = datetime.now(UTC)
        since = now - QUOTA_WINDOW
        player = await self.db.get(Player, player_id)
        total = (
            player.quota_daily
            if player is not None and player.quota_daily is not None
            else QUOTA_DAILY_DEFAULT
        )
        used_stmt = (
            select(func.count()).select_from(Job).where(*self._window_filter(player_id, since))
        )
        used = int(await self.db.scalar(used_stmt) or 0)
        left = max(total - used, 0)
        if used < total:
            return QuotaCheck(allowed=True, left=left, total=total, hours_to_free=0)
        hours = await self._hours_to_free(player_id, since=since, now=now, surplus=used - total)
        return QuotaCheck(allowed=False, left=left, total=total, hours_to_free=hours)

    @staticmethod
    def _window_filter(player_id: int, since: datetime) -> tuple[Any, ...]:
        """Один набор условий на оба запроса окна — чтобы «сколько потрачено» и
        «когда освободится» не могли начать считать по разным множествам задач.

        `status != 'failed'` — рулинг fix round 1: задача, упавшая по НАШЕЙ
        внутренней причине (`jobs.error` хранит её текст), не должна стоить
        игроку разбора из дневного лимита. Полоса наших сбоев иначе съедала бы
        чужой день целиком. Обратной стороны — «злоупотребление провалами» —
        здесь нет: провалившаяся задача не дала игроку никакого результата, так
        что выигрывать в этом размене нечего.
        """
        return (
            Job.player_id == player_id,
            Job.type.in_(QUOTA_INTERACTIVE_JOB_TYPES),
            Job.status != "failed",
            Job.created_at > since,
        )

    async def _hours_to_free(
        self, player_id: int, *, since: datetime, now: datetime, surplus: int
    ) -> int:
        """Через сколько часов освободится ПЕРВОЕ место.

        Это не всегда самая старая задача окна: если лимит успели понизить (или
        задачи проставили в обход бота), в окне может висеть больше задач, чем
        разрешено, — тогда первое место освободит `surplus`-я по возрасту, а не
        нулевая. `OFFSET surplus` выражает ровно это и в обычном случае
        (`used == total`) вырождается в «самая старая».

        Округление вверх: остаток в 10 минут — это «через 1 ч», а не «через 0»
        (сообщение с «через 0 ч» звучало бы как «уже можно», хотя нельзя).
        """
        oldest = await self.db.scalar(
            select(Job.created_at)
            .where(*self._window_filter(player_id, since))
            .order_by(Job.created_at)
            .offset(surplus)
            .limit(1)
        )
        if oldest is None:
            # Задач в окне нет, а место всё равно не даётся — значит лимит нулевой
            # (`quota_daily=0`, доступ отключён вручную). Освобождать нечему, и
            # честный ответ — полное окно, а не «через 0 ч, попробуйте ещё раз».
            return int(QUOTA_WINDOW.total_seconds() // 3600)
        remaining = (oldest + QUOTA_WINDOW) - now
        return max(1, math.ceil(remaining.total_seconds() / 3600))


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
