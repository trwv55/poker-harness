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
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, exists, func, literal, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from harness.contracts import (
    LEAK_RULES,
    MAX_NOTE_TEXT_CHARS,
    NOTE_COLOR_NONE,
    AnalysisResult,
    CalcName,
    CanonicalHand,
    CoverageResult,
    DecisionPoint,
    EnrichedHand,
    LeakRule,
    LeaksOverview,
    LeakStat,
    Measurement,
    NoteRecord,
    OpponentRecord,
    PointFilter,
    Provenance,
    RawHand,
    ScanSummary,
    SessionLine,
    SessionSummary,
    Window,
    is_judged,
    leak_rule_for,
)
from harness.memory.models import (
    Analysis,
    CalcCache,
    DecisionPointRow,
    EvalCase,
    Hand,
    Invite,
    Job,
    Note,
    Opponent,
    OpponentLink,
    Player,
    Tournament,
)
from harness.memory.models import Session as SessionRow

# Сколько случайных байт в инвайт-коде. 9 байт — 12 символов в base64url;
# перебором такой код не находится, а продиктовать его голосом всё ещё можно.
_INVITE_CODE_BYTES = 9

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

    async def set_gg_nickname(self, player_id: int, nickname: str) -> None:
        """Записать ник игрока в руме — вход опознания героя на скриншоте.

        Спрашивается один раз (задачи 19/23); здесь только запись. Пустой строкой
        не затирается: «ник неизвестен» это NULL, и превращать его в пустую
        строку значило бы завести второе значение с тем же смыслом.
        """
        if not nickname.strip():
            raise ValueError("ник в руме не может быть пустым")
        await self.db.execute(
            update(Player).where(Player.id == player_id).values(gg_nickname=nickname.strip())
        )
        await self.db.flush()

    async def find(self, tg_user_id: int) -> Player | None:
        """Игрок, если он уже заведён, — и НИКОГДА не заводит нового.

        Отдельно от `get_or_create` затем, что с задачи 23 вход закрыт инвайтом:
        обработчику надо уметь спросить «этот игрок уже наш?», не впуская
        незнакомца самим фактом вопроса.
        """
        return await self.db.scalar(select(Player).where(Player.tg_user_id == tg_user_id))

    async def set_pending_input(self, player_id: int, value: dict[str, Any] | None) -> None:
        """Запомнить (или снять) то, что означает следующее текстовое сообщение.

        `None` снимает ожидание. Состояние живёт в БД, а не в памяти бота, — см.
        комментарий к колонке `players.pending_input` (`memory/models.py`).
        """
        await self.db.execute(
            update(Player).where(Player.id == player_id).values(pending_input=value)
        )
        await self.db.flush()

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

    async def bootstrap_owner(self, tg_user_id: int) -> Player | None:
        """Завести владельца ПЕРВОЙ строкой `players` — или не завести ничего.

        Существует ради одного обстоятельства: коды выпускает только игрок с
        `is_dev`, а на чистой базе такого игрока нет, и продукт после деплоя
        недостижим никому. Здесь он появляется — ровно один раз на базу.

        `None` означает «таблица уже не пуста», и вызывающий обязан отказать
        обычным путём (`bot/handlers.py`): пустота — единственное условие, при
        котором id из окружения кого-то впускает, поэтому дверью после первого
        игрока эта переменная не остаётся
        (`test_the_owner_bootstrap_is_spent_once_per_database`).

        Проверка пустоты и вставка — ОДИН оператор (`INSERT ... WHERE NOT
        EXISTS`), а не «прочитали и записали»: два одновременных `/start` иначе
        разошлись бы между собой (тот же приём и та же причина, что у
        `InvitesRepo.redeem` и `get_or_create` выше;
        `test_two_owner_starts_at_once_admit_one_owner`).
        """
        created_id = await self.db.scalar(
            pg_insert(Player)
            .from_select(
                ["tg_user_id", "is_dev"],
                select(literal(tg_user_id), literal(True)).where(~exists(select(Player.id))),
            )
            .on_conflict_do_nothing(index_elements=["tg_user_id"])
            .returning(Player.id)
        )
        if created_id is None:
            return None
        created = await self.db.scalar(select(Player).where(Player.id == created_id))
        if created is None:  # pragma: no cover — только что вставленная строка
            raise LookupError(f"владелец {created_id} не найден сразу после вставки")
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

    async def count_for_player(self, player_id: int) -> int:
        """Сколько вечеров у игрока всего — знаменатель строки обрезки экрана.

        Отдельным запросом: длина `list_for_player` — размер страницы, а не
        история игрока (`test_the_session_count_does_not_depend_on_the_page_size`).
        """
        return int(
            await self.db.scalar(
                select(func.count())
                .select_from(SessionRow)
                .where(SessionRow.player_id == player_id)
            )
            or 0
        )

    async def list_for_player(self, player_id: int, *, limit: int = 10) -> list[SessionLine]:
        """Сессии игрока, свежие первыми, — список экрана «Сессии».

        Без агрегатов: сводка вечера считается ПО ЗАПРОСУ (решение владельца
        2026-09-07), и список не платит за неё на каждой строке.
        """
        stmt = (
            select(SessionRow)
            .where(SessionRow.player_id == player_id)
            .order_by(SessionRow.started_at.desc(), SessionRow.id.desc())
            .limit(limit)
        )
        return [
            SessionLine(
                session_id=row.id,
                title=row.title,
                started_at=row.started_at,
                is_active=row.closed_at is None,
            )
            for row in await self.db.scalars(stmt)
        ]

    async def summary(self, session_id: int, player_id: int) -> SessionSummary | None:
        """Агрегат вечера: турниры, разобранные руки, цена расхождений, лик вечера.

        `None` — сессии нет или она чужая: номер приезжает из `callback_data`,
        то есть из внешнего мира, и «показать по номеру» без сверки владельца
        отдало бы чужой вечер.

        `hands` — руки, по которым есть РАЗБОР (строка `analyses`), а не все
        сохранённые: сводка вечера отвечает на «сколько разобрано».

        `loss_bb` — сумма отрицательных расхождений по судимым точкам, взятая по
        модулю; ровно та величина, которую `ScanSummary.total_loss_bb` называет
        «суммарной потерей в оценённых решениях», только за сессию целиком.
        """
        row = await self.db.scalar(
            select(SessionRow).where(
                SessionRow.id == session_id, SessionRow.player_id == player_id
            )
        )
        if row is None:
            return None
        tournaments = int(
            await self.db.scalar(
                select(func.count())
                .select_from(Tournament)
                .where(Tournament.session_id == session_id)
            )
            or 0
        )
        hands = int(
            await self.db.scalar(
                select(func.count(func.distinct(Hand.id)))
                .select_from(Hand)
                .join(Analysis, Analysis.hand_id == Hand.id)
                .where(Hand.session_id == session_id)
            )
            or 0
        )
        leaks = LeaksRepo(self.db)
        evening = PointFilter(window=Window(session_id=session_id))
        judged, total = await leaks.coverage(player_id, evening)
        by_type = await leaks.by_type(player_id, evening)
        loss = await self._session_loss_bb(player_id, session_id)
        return SessionSummary(
            session_id=session_id,
            title=row.title,
            tournaments=tournaments,
            hands=hands,
            loss_bb=loss,
            points_judged=judged,
            points_total=total,
            top_leak=by_type[0] if by_type else None,
        )

    async def _session_loss_bb(self, player_id: int, session_id: int) -> float:
        """Цена расхождений вечера — сумма отрицательных `ev_diff_bb` судимых точек."""
        loss = await self.db.scalar(
            select(_negative_loss(DecisionPointRow.judged)).where(
                *_points_of(player_id, PointFilter(window=Window(session_id=session_id)))
            )
        )
        return -float(loss or 0.0)

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

    async def replace_raw(self, hand_id: int, raw: RawHand) -> None:
        """Переписать `hands.raw` и СБРОСИТЬ чекпоинты ниже по конвейеру.

        Спека §8.3, шаг 2: ответ игрока на вопрос валидатора патчит сырую руку, а
        `canonical`/`enriched` пересчитываются. Сброс здесь, а не у вызывающего, —
        потому что разъединить эти две записи некому: строка с новым `raw` и
        старым `enriched` описывает две разные руки сразу, и любой, кто прочитает
        её между двумя апдейтами, получит именно это.
        """
        record = await self._get_row(hand_id)
        record.raw = raw.model_dump(mode="json")
        record.canonical = None
        record.enriched = None
        await self.db.flush()

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

    async def player_hands_by_tournament(self, player_id: int) -> list[list[CanonicalHand]]:
        """Канонические руки игрока по всем его турнирам — списком на турнир.

        Вход «среднего по всем турнирам» в отчёте (задача 23). Группировка
        турнирами, а не одним плоским списком, — это не удобство вызывающего:
        по числу групп отчёт решает, есть ли с чем сравнивать вообще, и
        передать одно вместо другого он не сможет.

        Читается ровно одна колонка — `canonical`: `raw` и `enriched` весят
        кратно больше, а формулам статистики (`analysis/player_stats.py`) не
        нужны ни отчёт движка, ни исходные строки файла. Руки без чекпоинта
        `canonical` пропускаются: пайплайн до них не дошёл, и считать по ним
        нечего.

        Область — сессии этого игрока (JOIN по `sessions.player_id`), как и у
        `find_session_by_hand_no`: без этого условия в среднее попали бы чужие
        раздачи. Руки без турнира (скриншоты) не входят вовсе — сравнение
        заявлено «по турнирам».

        Стоимость растёт с историей игрока: это чтение ВСЕХ его рук на каждый
        отчёт. На порядках величин v1 (единицы турниров по паре сотен раздач)
        это дешевле, чем отдельная таблица агрегатов, которую пришлось бы
        держать в согласии с руками; кэш и агрегаты — после телеметрии
        (SCALING.md, «отложить до телеметрии»).
        """
        stmt = (
            select(Hand.tournament_id, Hand.canonical)
            .join(SessionRow, SessionRow.id == Hand.session_id)
            .where(
                SessionRow.player_id == player_id,
                Hand.tournament_id.is_not(None),
                Hand.canonical.is_not(None),
            )
            .order_by(Hand.tournament_id, Hand.id)
        )
        grouped: dict[int, list[CanonicalHand]] = {}
        for tournament_id, canonical in await self.db.execute(stmt):
            grouped.setdefault(tournament_id, []).append(
                CanonicalHand.model_validate(canonical)
            )
        return list(grouped.values())

    async def player_canonical(
        self, player_id: int, window: Window | None = None
    ) -> list[CanonicalHand]:
        """Канонические руки игрока в окне, плоским списком в порядке записи.

        Вход частот словаря расчётов. Плоским, а не по турнирам, в отличие от
        `player_hands_by_tournament`: там группы несут смысл (по их числу отчёт
        решает, есть ли с чем сравнивать), здесь считается один знаменатель на
        всё окно, и группировать нечего.

        Руки без турнира (скриншоты) ВХОДЯТ, в отличие от того же соседа:
        частота считается по действиям за столом, а они у скриншота такие же.
        Отсекается только отсутствие чекпоинта `canonical` — пайплайн до состава
        мест не дошёл, и считать по такой руке нечего.

        Окно — те же две колонки `sessions`, что у `_points_of`, и по той же
        причине (`contracts.calcs.Window`). Область всегда ограничена сессиями
        этого игрока, как у `player_hands_by_tournament`: без этого условия в
        знаменатель попали бы чужие раздачи.
        """
        window = window or Window()
        where = [SessionRow.player_id == player_id, Hand.canonical.is_not(None)]
        if window.session_id is not None:
            where.append(Hand.session_id == window.session_id)
        if window.since is not None:
            where.append(SessionRow.started_at >= window.since)
        stmt = (
            select(Hand.canonical)
            .join(SessionRow, SessionRow.id == Hand.session_id)
            .where(*where)
            .order_by(Hand.id)
        )
        return [
            CanonicalHand.model_validate(canonical)
            for canonical in await self.db.scalars(stmt)
        ]

    async def last_canonical(self, player_id: int) -> CanonicalHand | None:
        """Свежайшая рука игрока, дошедшая до чекпоинта `canonical`, — «последний
        разбор» для команды псевдонима (`bot/handlers.py`).

        Свежесть — по `hands.id`: он растёт с порядком записи, а времени
        разбора у таблицы нет вовсе. Руки без `canonical` пропускаются: до
        состава мест за столом пайплайн по ним не дошёл, и назвать в такой руке
        участника не по чему.

        Область — сессии этого игрока (JOIN по `sessions.player_id`), как у
        `find_session_by_hand_no` и `player_hands_by_tournament`: без этого
        условия «последним разбором» мог бы оказаться чужой.
        """
        stmt = (
            select(Hand.canonical)
            .join(SessionRow, SessionRow.id == Hand.session_id)
            .where(SessionRow.player_id == player_id, Hand.canonical.is_not(None))
            .order_by(Hand.id.desc())
            .limit(1)
        )
        canonical = await self.db.scalar(stmt)
        return None if canonical is None else CanonicalHand.model_validate(canonical)

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


# Поле `PointVerdict` → колонка `decision_points`. Одна карта на запись и на
# чтение: два списка полей, обязанных совпадать, — ровно то, чего эта задача
# избегает. Переименование единственное и вынужденное (`interval` в Postgres
# зарезервировано); полнота карты проверяется
# `test_every_field_of_a_point_verdict_has_its_column`.
_POINT_COLUMNS: Mapping[str, str] = {
    "dp_index": "dp_index",
    "street": "street",
    "spot": "spot",
    "zone": "zone",
    "action_taken": "action_taken",
    "best_action": "best_action",
    "ev_diff_bb": "ev_diff_bb",
    "interval": "ev_interval",
    "assumption": "assumption",
    "tools": "tools",
    "detail": "detail",
}

# Обстановка точки из `DecisionPoint` (`hands.enriched`): имя поля и имя колонки
# совпадают. Не весь контракт — только то, по чему фильтруют: улица и сыгранное
# действие уже приехали из вердикта, а метка места и живые оппоненты в фильтрах
# не участвуют.
_CONTEXT_COLUMNS: tuple[str, ...] = (
    "position",
    "to_call",
    "pot_before",
    "eff_stack",
    "eff_stack_bb",
    "spr",
)


class AnalysesRepo:
    """`analyses`: выход ядра (`result`) и изложения (`verdict_text`, `range_images`).

    Точки решения лежат НЕ здесь, а строками в `decision_points` (миграция
    0008): `result` хранит документ без них, `save` раскладывает массив по
    строкам, `get_by_hand` собирает его обратно. Второй копии вердикта в базе
    нет — значит, ей и не с чем расходиться
    (`test_the_saved_document_carries_no_points_of_its_own`).
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def save(
        self,
        *,
        hand_id: int,
        result: AnalysisResult,
        decision_points: Sequence[DecisionPoint],
        verdict_text: str | None = None,
        range_images: list[str] | None = None,
    ) -> int:
        """Записать разбор: документ в `analyses`, точки — строками, в одной транзакции.

        `decision_points` — точки решения движка из `hands.enriched` той же
        руки; из них берётся обстановка (позиция, банк, доплата, стек, SPR),
        которой в вердикте нет. Вердикт сопоставляется с обстановкой по
        `dp_index`; вердикту, которому в руке ничего не соответствует, колонки
        обстановки остаются пустыми, а не заполняются похожей точкой
        (`test_a_verdict_without_its_decision_point_is_stored_without_context`).
        """
        record = Analysis(
            hand_id=hand_id,
            result=result.model_dump(mode="json", exclude={"points"}),
            verdict_text=verdict_text,
            range_images=range_images,
        )
        self.db.add(record)
        await self.db.flush()
        await self._save_points(hand_id, result, decision_points)
        return record.id

    async def _save_points(
        self, hand_id: int, result: AnalysisResult, decision_points: Sequence[DecisionPoint]
    ) -> None:
        if not result.points:
            return
        address = (
            await self.db.execute(
                select(Hand.session_id, SessionRow.player_id)
                .join(SessionRow, SessionRow.id == Hand.session_id)
                .where(Hand.id == hand_id)
            )
        ).one()
        context = {point.index: point for point in decision_points}
        rows: list[dict[str, Any]] = []
        for point_no, point in enumerate(result.points):
            dumped = point.model_dump(mode="json")
            row: dict[str, Any] = {
                field_column: dumped[field] for field, field_column in _POINT_COLUMNS.items()
            }
            row |= {
                "hand_id": hand_id,
                "session_id": address.session_id,
                "player_id": address.player_id,
                "point_no": point_no,
                # Единственная формулировка правила судимости на всю систему —
                # `contracts.is_judged`; SQL после этого его не повторяет.
                "judged": is_judged(point),
            }
            source = context.get(point.dp_index)
            row |= {
                name: None if source is None else getattr(source, name)
                for name in _CONTEXT_COLUMNS
            }
            rows.append(row)
        await self.db.execute(pg_insert(DecisionPointRow), rows)
        await self.db.flush()

    async def set_explanation(
        self, *, hand_id: int, verdict_text: str | None = None, range_images: list[str]
    ) -> None:
        """Дописать изложение к УЖЕ сохранённому разбору — чекпоинт станции explain.

        Отдельным методом, а не вторым `save()`: разбор (`result`) и текст к нему
        считаются разными станциями конвейера и переживают разные падения (задача
        18, чекпоинты). Повторная попытка, у которой числа уже посчитаны, обязана
        дописать к ним слова, а не завести вторую строку на ту же руку.

        `verdict_text=None` — законный случай: картинки диапазонов рисует код, и
        сохранить их надо даже тогда, когда модель не ответила. Пустой текст при
        этом НЕ записывается поверх существующего — колонка просто не попадает в
        `UPDATE` (`test_set_explanation_without_text_keeps_the_saved_one`).
        """
        values: dict[str, Any] = {"range_images": range_images}
        if verdict_text is not None:
            values["verdict_text"] = verdict_text
        await self.db.execute(
            update(Analysis).where(Analysis.hand_id == hand_id).values(**values)
        )

    async def get_by_hand(self, hand_id: int) -> AnalysisRecord | None:
        """Разбор руки целиком: документ из `analyses` плюс точки из строк.

        Точки возвращаются в порядке `point_no` — в том же, в каком лежали в
        массиве: `AnalysisResult.ranked` индексирует массив, и перестановка
        сдвинула бы ранжирование на чужие точки
        (`test_the_analysis_document_returns_from_the_rows_as_it_went_in`).
        """
        record = await self.db.scalar(select(Analysis).where(Analysis.hand_id == hand_id))
        if record is None:
            return None
        rows = await self.db.scalars(
            select(DecisionPointRow)
            .where(DecisionPointRow.hand_id == hand_id)
            .order_by(DecisionPointRow.point_no)
        )
        points = [
            {field: getattr(row, field_column) for field, field_column in _POINT_COLUMNS.items()}
            for row in rows
        ]
        return AnalysisRecord(
            id=record.id,
            hand_id=record.hand_id,
            result=AnalysisResult.model_validate({**record.result, "points": points}),
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

    async def player_scan_summaries(
        self, player_id: int, *, exclude: int
    ) -> list[ScanSummary]:
        """Сводки сканов ПРОШЛЫХ турниров игрока — источник «этот паттерн уже был».

        `exclude` — турнир, по которому отчёт строится сейчас: его сводка к
        этому моменту уже сохранена, и без исключения каждая находка текущего
        турнира читалась бы как «то же самое было раньше»
        (`test_past_scan_summaries_exclude_the_tournament_being_reported`).
        Аргумент обязателен и именован: молчаливое умолчание «ничего не
        исключать» — ровно та ошибка, которую он предотвращает.

        Турниры без сохранённой сводки (скан не дошёл до конца) не попадают:
        отсутствующая сводка — не пустая.
        """
        stmt = (
            select(Tournament.scan_summary)
            .join(SessionRow, SessionRow.id == Tournament.session_id)
            .where(
                SessionRow.player_id == player_id,
                Tournament.id != exclude,
                Tournament.scan_summary.is_not(None),
            )
            .order_by(Tournament.id)
        )
        return [ScanSummary.model_validate(row) for row in await self.db.scalars(stmt)]

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

    async def get_awaiting(self, job_id: int, player_id: int) -> Job | None:
        """Ждущая ответа задача ПО НОМЕРУ — и только если она этого игрока.

        Номер приходит из нажатой кнопки; сверка с игроком нужна затем, что
        `callback_data` приходит из внешнего мира и номер в нём может быть
        любым. Ждущих задач у игрока бывает несколько сразу (спека §8.1:
        `awaiting_user` не считается активной), поэтому «самая свежая ждущая» —
        неверный ответ на вопрос «к какой руке относится этот ответ».
        """
        stmt = select(Job).where(
            Job.id == job_id, Job.player_id == player_id, Job.status == "awaiting_user"
        )
        return await self.db.scalar(stmt)

    async def awaiting_manual_entry(self, player_id: int) -> Job | None:
        """Задача игрока, ждущая ЧИСЛА, введённого вручную, — самая свежая из них.

        Обычное сообщение номера задачи не несёт, поэтому адресат ищется по
        признаку начатого ввода (`payload["manual_entry"]`), а не по одному лишь
        статусу: у соседней ждущей задачи ввод не начинали, и подставлять число
        в неё нельзя. Состояние ввода живёт в `jobs.payload`, а не в памяти
        процесса бота (спека §8.3 — точка возврата уже зафиксирована
        артефактами, и переживать она обязана перезапуск бота так же, как
        переживает его сама задача).
        """
        stmt = (
            select(Job)
            .where(
                Job.player_id == player_id,
                Job.status == "awaiting_user",
                Job.payload["manual_entry"].astext.isnot(None),
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
        """Записать пачку значений, разбивая её на куски по `_UPSERT_CHUNK` строк.

        Куски обязательны, а не оптимизация: одна строка стоит двух связанных
        параметров, а протокол Postgres их больше 32767 в одном запросе не
        принимает — с 16384-й строки asyncpg роняет запрос целиком
        («the number of query arguments cannot exceed 32767»). Кэш эквити растёт
        от турнира к турниру и этот рубеж переходит; поймано прогоном, где
        накопленный дисковый кэш дорос до 16884 записей и КАЖДАЯ задача воркера
        стала падать на записи в `calc_cache`. Закреплено
        `test_calc_cache_upsert_survives_more_rows_than_one_statement_allows`.
        """
        if not entries:
            return
        values: list[dict[str, Any]] = [
            {"key": f"{prefix}{key}", "value": value} for key, value in entries.items()
        ]
        for start in range(0, len(values), _UPSERT_CHUNK):
            stmt = (
                pg_insert(CalcCache)
                .values(values[start : start + _UPSERT_CHUNK])
                .on_conflict_do_nothing(index_elements=["key"])
            )
            await self.db.execute(stmt)
        await self.db.flush()


# Сколько строк уходит в БД одним INSERT. Потолок протокола Postgres — 32767
# связанных параметров на запрос, строка кэша стоит двух (ключ и значение),
# то есть жёсткий предел 16383 строки. 5000 — с запасом и без лишних рейсов.
_UPSERT_CHUNK = 5000


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


# Точки решения игрока — источник всех сквозных агрегатов («Мои лики», покрытие,
# цена вечера). Одно выражение вместо текста запроса: до миграции 0008 точки
# лежали массивом в `analyses.result`, разворачивались боковым соединением и
# каждое условие писалось по jsonb руками; теперь фильтры складываются обычным
# `where` и не требуют ни приведений типа, ни повторения правил на втором языке.
def _points_of(player_id: int, filters: PointFilter):
    """Условие «точки этого игрока, подходящие под фильтр» — списком для `where`.

    Каждый фильтр — обычное условие по колонке, и складываются они как угодно:
    именно ради этого точки переехали из массива `analyses.result` в таблицу
    (миграция 0008). Отдельного запроса на каждое сочетание фильтров нет
    (`test_every_filter_narrows_the_same_query`).

    Окно по времени выражено через `sessions.started_at`, а не через отметку
    внутри руки: у таблицы `hands` собственного времени нет вовсе, а вечер и
    есть единица времени продукта (`contracts.calcs.Window`). Условие игрока в
    подзапросе повторено намеренно — без него окно захватило бы чужие вечера,
    начавшиеся в те же дни.
    """
    where = [DecisionPointRow.player_id == player_id]
    if filters.window.session_id is not None:
        where.append(DecisionPointRow.session_id == filters.window.session_id)
    if filters.window.since is not None:
        where.append(
            DecisionPointRow.session_id.in_(
                select(SessionRow.id).where(
                    SessionRow.player_id == player_id,
                    SessionRow.started_at >= filters.window.since,
                )
            )
        )
    if filters.street is not None:
        where.append(DecisionPointRow.street == filters.street)
    if filters.spot is not None:
        where.append(DecisionPointRow.spot == filters.spot)
    if filters.position is not None:
        where.append(DecisionPointRow.position == filters.position)
    return where


# Сумма отрицательных расхождений — «столько ушло». Отдельным выражением,
# потому что складывают её два места (тип лика и цена вечера) и слагаемое у них
# одно; расходятся они только тем, что цена вечера берёт ещё и судимость, —
# отсюда `also`.
def _negative_loss(*also: Any):
    return func.coalesce(
        func.sum(DecisionPointRow.ev_diff_bb).filter(*also, DecisionPointRow.ev_diff_bb < 0.0),
        0.0,
    )


class LeaksRepo:
    """Сквозная статистика ошибок по всей истории игрока — экран «Мои лики».

    **Группировка по ТИПУ ЛИКА, а не по споту** (решение владельца 2026-09-07):
    тип — строка таблицы `LEAK_RULES` (`contracts/history.py`), то есть тройка
    «спот · сыграно · лучше». SQL группирует точки по этой тройке, Python
    сопоставляет тройки с таблицей правил; тройка, которой в таблице нет, в
    экран не попадает — этим же отсекаются точки «около нуля» (в `best_action`
    у них русская фраза ядра) и точки без вердикта (пустой `best_action`),
    которые в лики не входят по постановке.

    Читается по всей истории игрока, а не по сессии: «типовые ошибки копятся по
    всей истории, иначе „твой повторяющийся лик“ не вычислить» (SESSIONS_UX,
    раздел «Что сквозное»). Тот же агрегат с `session_id` — лик одного вечера.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def overview(self, player_id: int) -> LeaksOverview:
        """Экран целиком: покрытие за всю историю плюс типы ликов по цене."""
        judged, total = await self.coverage(player_id)
        return LeaksOverview(
            points_judged=judged,
            points_total=total,
            leaks=await self.by_type(player_id),
        )

    async def coverage(
        self, player_id: int, filters: PointFilter | None = None
    ) -> tuple[int, int]:
        """Сколько точек решения оценено из скольких — под заданным фильтром.

        Пара, которую экран печатает строкой «оценено N из M решений»: без неё
        список ликов читается как полная картина игры, хотя судится сегодня
        только префлоп-пуш-фолд. Судимость читается колонкой `judged`, а не
        условием: правило записано один раз, в `contracts.is_judged`.

        Фильтр по умолчанию пуст — вся история игрока.
        """
        row = (
            await self.db.execute(
                select(
                    func.count().label("points_total"),
                    func.count().filter(DecisionPointRow.judged).label("points_judged"),
                ).where(*_points_of(player_id, filters or PointFilter()))
            )
        ).one()
        return int(row.points_judged), int(row.points_total)

    async def coverage_and_cost(
        self, player_id: int, filters: PointFilter | None = None
    ) -> CoverageResult:
        """Покрытие и цена под фильтром — три величины с тремя знаменателями.

        Сумма в bb складывается ТОЛЬКО с судимых точек и только с тех из них, где
        расхождение отрицательно; `priced` называет, сколько их было. Точка
        разобранная, но без посчитанной цены, входит в знаменатель покрытия и не
        входит в сумму: сложить её ноль с ценами значило бы подать «здесь не
        потеряно» там, где не считали (`CoverageResult`,
        `test_points_without_a_price_do_not_enter_the_sum`).
        """
        where = _points_of(player_id, filters or PointFilter())
        priced = (DecisionPointRow.judged, DecisionPointRow.ev_diff_bb < 0.0)
        row = (
            await self.db.execute(
                select(
                    func.count().label("total"),
                    func.count().filter(DecisionPointRow.judged).label("judged"),
                    func.count().filter(*priced).label("priced"),
                    _negative_loss(DecisionPointRow.judged).label("loss"),
                ).where(*where)
            )
        ).one()
        return CoverageResult(
            calc=CalcName.COVERAGE,
            filter=filters or PointFilter(),
            judged=Measurement(numerator=int(row.judged), denominator=int(row.total)),
            priced=Measurement(numerator=int(row.priced), denominator=int(row.judged)),
            loss_bb=-float(row.loss),
        )

    async def by_type(
        self, player_id: int, filters: PointFilter | None = None
    ) -> list[LeakStat]:
        """Типы ликов с частотой и ценой, самый дорогой первым.

        `loss_bb` положителен («столько ушло»), как в `EvSplit`: складываются
        только отрицательные `ev_diff_bb`, потому что лик — это потеря, а не
        сальдо (положительных расхождений у судимой точки не бывает по
        построению ядра, и полагаться на это правило здесь не нужно).
        """
        stmt = (
            select(
                DecisionPointRow.spot,
                DecisionPointRow.action_taken,
                DecisionPointRow.best_action,
                func.count().label("n"),
                _negative_loss().label("loss"),
            )
            .where(*_points_of(player_id, filters or PointFilter()))
            .group_by(
                DecisionPointRow.spot,
                DecisionPointRow.action_taken,
                DecisionPointRow.best_action,
            )
        )
        counts: dict[str, int] = {}
        losses: dict[str, float] = {}
        rules: dict[str, LeakRule] = {}
        for row in (await self.db.execute(stmt)).all():
            rule = leak_rule_for(row.spot, row.action_taken, row.best_action)
            if rule is None:
                continue
            rules[rule.key] = rule
            counts[rule.key] = counts.get(rule.key, 0) + int(row.n)
            losses[rule.key] = losses.get(rule.key, 0.0) + float(row.loss)
        stats = [
            LeakStat(rule=rules[key], count=counts[key], loss_bb=-losses[key])
            for key in rules
        ]
        # Дорогой лик — первым, при равной цене — более частый. Третий ключ
        # (порядок в таблице правил) делает порядок полным: две строки с
        # одинаковой ценой и частотой иначе менялись бы местами от запроса к
        # запросу, и экран «Мои лики» выглядел бы по-разному без причины.
        order = {rule.key: index for index, rule in enumerate(LEAK_RULES)}
        stats.sort(key=lambda stat: (-stat.loss_bb, -stat.count, order[stat.rule.key]))
        return stats


class NotesRepo:
    """`notes`: заметки на оппонентов — сквозные, одна на оппонента.

    Живут только на vision-пути: в GG-HH оппоненты анонимизированы, и
    идентичность не переживает турнир (ARCHITECTURE §6, спека §5.2). Ник сюда
    приходит с экрана, поэтому все методы, кроме списка, сверяют владельца —
    номер заметки приезжает из `callback_data`, то есть из внешнего мира.

    Личность оппонента общая с частотами: заметка ссылается на строку
    `opponents` (`OpponentsRepo`), а не держит ник своей колонкой. Поэтому
    регистр ника здесь не различается — ровно так же, как в сшивках турниров
    (`test_a_note_and_a_link_on_the_same_nick_in_two_cases_meet_on_one_opponent`).
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def upsert(
        self, *, owner_player_id: int, nick: str, text_: str, color: str | None = None
    ) -> int:
        """Записать наблюдение об оппоненте; вторая запись на того же — правка.

        Заметка накапливается на оппоненте, а не на вечере (SESSIONS_UX), и
        уникальный индекс `notes(opponent_id)` (миграция 0007) делает это
        структурной гарантией, а не соглашением вызывающего. Оппонент по нику
        заводится тем же `get_or_create`, что и для сшивок, поэтому заметка на
        `Vasya` и заметка на `vasya` — одна заметка
        (`test_a_note_is_one_per_opponent_and_editing_keeps_its_colour`).

        `color=None` означает «цвет не трогать»: цвет ставится отдельной
        кнопкой, и правка текста не имеет права его стирать
        (`test_a_note_is_one_per_opponent_and_editing_keeps_its_colour`).
        """
        stripped = text_.strip()
        if not stripped:
            raise ValueError("заметка не может быть пустой")
        if len(stripped) > MAX_NOTE_TEXT_CHARS:
            # Инвариант хранилища, а не текст игроку: отказ словами выдаёт
            # `bot.handlers` до вызова, здесь стоит нижняя граница
            # (`test_a_note_longer_than_the_limit_is_refused_rather_than_cut`).
            raise ValueError(f"заметка длиннее {MAX_NOTE_TEXT_CHARS} символов")
        opponent_id = await OpponentsRepo(self.db).get_or_create(
            owner_player_id=owner_player_id, nick=nick
        )
        now = datetime.now(UTC)
        insert = pg_insert(Note).values(
            owner_player_id=owner_player_id,
            opponent_id=opponent_id,
            color=color if color is not None else NOTE_COLOR_NONE,
            text=stripped,
            updated_at=now,
        )
        updates: dict[str, Any] = {"text": stripped, "updated_at": now}
        if color is not None:
            updates["color"] = color
        stmt = insert.on_conflict_do_update(
            index_elements=["opponent_id"], set_=updates
        ).returning(Note.id)
        note_id = await self.db.scalar(stmt)
        await self.db.flush()
        if note_id is None:  # pragma: no cover — upsert всегда возвращает строку
            raise LookupError(f"заметка на {nick!r} не записалась")
        return int(note_id)

    async def set_color(self, note_id: int, owner_player_id: int, color: str) -> bool:
        """Поставить цветовой архетип; `False` — заметки нет или она чужая."""
        result = await self.db.execute(
            update(Note)
            .where(Note.id == note_id, Note.owner_player_id == owner_player_id)
            .values(color=color, updated_at=datetime.now(UTC))
            .returning(Note.id)
        )
        await self.db.flush()
        return result.first() is not None

    async def delete(self, note_id: int, owner_player_id: int) -> bool:
        """Удалить заметку; `False` — её нет или она чужая.

        Оппонент переживает свою заметку: сшивки турниров держатся на нём, и
        удалять его вместе с текстом наблюдения означало бы стереть привязки,
        о которых игрок не просил
        (`test_deleting_a_note_leaves_the_opponent_and_his_links`).
        """
        result = await self.db.execute(
            delete(Note)
            .where(Note.id == note_id, Note.owner_player_id == owner_player_id)
            .returning(Note.id)
        )
        await self.db.flush()
        return result.first() is not None

    async def get(self, note_id: int, owner_player_id: int) -> NoteRecord | None:
        row = (
            await self.db.execute(
                select(Note, Opponent.opponent_nick)
                .join(Opponent, Note.opponent_id == Opponent.id)
                .where(Note.id == note_id, Note.owner_player_id == owner_player_id)
            )
        ).first()
        return None if row is None else self._to_record(row[0], row[1])

    async def find_by_nick(self, owner_player_id: int, nick: str) -> NoteRecord | None:
        """Заметка на этого оппонента, если она уже есть, — для показа перед правкой."""
        row = (
            await self.db.execute(
                select(Note, Opponent.opponent_nick)
                .join(Opponent, Note.opponent_id == Opponent.id)
                .where(
                    Note.owner_player_id == owner_player_id,
                    func.lower(Opponent.opponent_nick) == func.lower(nick.strip()),
                )
            )
        ).first()
        return None if row is None else self._to_record(row[0], row[1])

    async def count_for_player(self, owner_player_id: int) -> int:
        """Сколько заметок у игрока всего — знаменатель строки обрезки экрана.

        Отдельным запросом, потому что `list_for_player` возвращает страницу:
        её длина — размер страницы, а не то, сколько заметок у игрока
        (`test_the_note_count_does_not_depend_on_the_page_size`).
        """
        return int(
            await self.db.scalar(
                select(func.count())
                .select_from(Note)
                .where(Note.owner_player_id == owner_player_id)
            )
            or 0
        )

    async def list_for_player(self, owner_player_id: int, *, limit: int = 50) -> list[NoteRecord]:
        """Заметки игрока, свежие первыми."""
        stmt = (
            select(Note, Opponent.opponent_nick)
            .join(Opponent, Note.opponent_id == Opponent.id)
            .where(Note.owner_player_id == owner_player_id)
            .order_by(Note.updated_at.desc(), Note.id.desc())
            .limit(limit)
        )
        return [self._to_record(note, nick) for note, nick in await self.db.execute(stmt)]

    @staticmethod
    def _to_record(record: Note, nick: str) -> NoteRecord:
        return NoteRecord(
            note_id=record.id,
            nick=nick,
            color=record.color,
            text=record.text,
            updated_at=record.updated_at,
        )


class InvitesRepo:
    """`invites`: доступ по коду. Выдаёт владелец (`players.is_dev`), гасит `/start`.

    Код — случайные байты из `secrets`, а не последовательность: инвайт это
    пропуск в закрытый продукт, и угадываемый код отменял бы саму его цель.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def mint(self, issued_by: int) -> str:
        """Создать неиспользованный код и вернуть его текст."""
        code = secrets.token_urlsafe(_INVITE_CODE_BYTES)
        record = Invite(code=code, issued_by=issued_by)
        self.db.add(record)
        await self.db.flush()
        return code

    async def redeem(self, code: str, used_by: int) -> bool:
        """Погасить код на игрока; `False` — кода нет или он уже использован.

        Одним `UPDATE ... WHERE used_by IS NULL`, а не «прочитать и записать»:
        два одновременных `/start` с одним кодом иначе прошли бы оба, и один
        инвайт впустил бы двоих (тот же приём, что `ON CONFLICT DO NOTHING` в
        `PlayersRepo.get_or_create`).
        """
        result = await self.db.execute(
            update(Invite)
            .where(Invite.code == code.strip(), Invite.used_by.is_(None))
            .values(used_by=used_by, used_at=datetime.now(UTC))
            .returning(Invite.id)
        )
        await self.db.flush()
        return result.first() is not None


class OpponentsRepo:
    """`opponents` + `opponent_links`: кто из участников турнира — какой оппонент.

    В файлах раздач участники обезличены, и метка участника сквозная только
    внутри турнира: между турнирами комната выдаёт новую. Связь между метками
    разных турниров утверждает ВЛАДЕЛЕЦ. Здесь нет ни одной функции, которая
    предлагала бы связь сама, — ни по стилю игры, ни по стеку, ни по совпадению
    чего бы то ни было; методы ниже только записывают и читают сказанное
    владельцем.

    Личность названа ником в руме и одна на весь продукт: на ту же строку
    ссылается заметка (`NotesRepo`).
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_or_create(self, *, owner_player_id: int, nick: str) -> int:
        """Номер оппонента по нику; второй раз тот же ник — тот же номер.

        Регистр не различается: `Vasya` и `vasya` — один оппонент
        (`test_the_same_nick_in_another_case_is_the_same_opponent`). Хранится
        написание, которым ник назвали в первый раз: переписывать его вторым
        вызовом значило бы менять то, что владелец уже видит на экране.

        Гонка закрыта тем же приёмом, что в `PlayersRepo.get_or_create`:
        `ON CONFLICT DO NOTHING` вместо «прочитали — не нашли — вставили», и
        проигравший перечитывает готовую строку.
        """
        stripped = nick.strip()
        if not stripped:
            raise ValueError("ник оппонента не может быть пустым")
        found = await self._find_by_nick(owner_player_id, stripped)
        if found is not None:
            return found
        created_id = await self.db.scalar(
            pg_insert(Opponent)
            .values(owner_player_id=owner_player_id, opponent_nick=stripped)
            .on_conflict_do_nothing(
                index_elements=[
                    Opponent.owner_player_id,
                    func.lower(Opponent.opponent_nick),
                ]
            )
            .returning(Opponent.id)
        )
        await self.db.flush()
        if created_id is not None:
            return int(created_id)
        found = await self._find_by_nick(owner_player_id, stripped)
        if found is None:  # pragma: no cover — конфликт был, а строки нет
            raise LookupError(f"оппонент {stripped!r} исчез после конфликта вставки")
        return found

    async def link(
        self,
        *,
        owner_player_id: int,
        opponent_id: int,
        room_tournament_id: str,
        participant_label: str,
    ) -> int | None:
        """Привязать метку участника В ЭТОМ ТУРНИРЕ к оппоненту.

        Возвращает номер оппонента, за которым пара «турнир + метка» закреплена
        ПОСЛЕ вызова: он же, если привязка состоялась, и ЧУЖОЙ, если эту пару
        уже занял другой ник — вызывающему нужно различать эти два исхода, а не
        получать «не получилось»
        (`test_a_participant_already_bound_keeps_his_first_nick`). `None` —
        такого оппонента у этого владельца нет.

        Один оператор, а не «прочитали — не нашли — вставили»: пара занимается
        первичным ключом `opponent_links`, и два одновременных вызова физически
        не могут развести её по двум оппонентам — проигравший ждёт коммита
        победителя на самом конфликте и возвращает победителя
        (`test_two_opponents_at_once_claim_one_participant_and_the_first_keeps_him`).
        `DO UPDATE`, а не `DO NOTHING`, именно ради этого: `DO NOTHING` вернул
        бы пустоту, неотличимую от «оппонента нет», а отдельным SELECT'ом после
        него чужую ещё не закоммиченную строку не увидеть.

        Владелец сверяется тем же оператором: строка берётся из `opponents`, и
        чужой номер просто не даёт ни одной строки на вставку
        (`test_an_opponent_of_another_player_takes_no_bindings`).
        """
        tournament = room_tournament_id.strip()
        label = participant_label.strip()
        if not tournament or not label:
            raise ValueError("турнир и метка участника не могут быть пустыми")
        taken = await self.db.scalar(
            select(OpponentLink.participant_label).where(
                OpponentLink.opponent_id == opponent_id,
                OpponentLink.room_tournament_id == tournament,
            )
        )
        if taken is not None and taken != label:
            # Пол, ниже которого не пускает уникальный индекс
            # `uq_opponent_links_opponent_tournament`; словами про это говорит
            # бот, до вызова
            # (`test_one_opponent_keeps_one_participant_per_tournament`).
            # Проверка читает уже закоммиченное, поэтому ДВЕ одновременные
            # привязки разных меток одного турнира к одному оппоненту доходят до
            # индекса, и проигравший получает `IntegrityError`, а не эти слова:
            # у команды владельца такой одновременности не бывает, а тихо
            # разойтись правило не имеет права.
            raise ValueError("у этого оппонента в этом турнире уже есть метка")
        source = select(
            literal(owner_player_id),
            literal(tournament),
            literal(label),
            Opponent.id,
        ).where(
            Opponent.id == opponent_id, Opponent.owner_player_id == owner_player_id
        )
        holder = await self.db.scalar(
            pg_insert(OpponentLink)
            .from_select(
                ["owner_player_id", "room_tournament_id", "participant_label", "opponent_id"],
                source,
            )
            .on_conflict_do_update(
                index_elements=[
                    OpponentLink.owner_player_id,
                    OpponentLink.room_tournament_id,
                    OpponentLink.participant_label,
                ],
                set_={"opponent_id": OpponentLink.opponent_id},
            )
            .returning(OpponentLink.opponent_id)
        )
        await self.db.flush()
        return None if holder is None else int(holder)

    async def unlink(
        self, *, owner_player_id: int, room_tournament_id: str, participant_label: str
    ) -> bool:
        """Снять привязку пары «турнир + метка»; `False` — её не было или она чужая."""
        result = await self.db.execute(
            delete(OpponentLink)
            .where(
                OpponentLink.owner_player_id == owner_player_id,
                OpponentLink.room_tournament_id == room_tournament_id.strip(),
                OpponentLink.participant_label == participant_label.strip(),
            )
            .returning(OpponentLink.opponent_id)
        )
        await self.db.flush()
        return result.first() is not None

    async def list_for_player(self, owner_player_id: int) -> list[OpponentRecord]:
        """Все оппоненты владельца по алфавиту, с числом привязок у каждого.

        Без потолка запроса: страницу пришлось бы сопровождать вторым запросом
        за общим числом (как у заметок, `NotesRepo.count_for_player`), а число
        оппонентов, которых владелец назвал руками, того же порядка, что число
        заметок. Обрезку по пределу сообщения делает `presentation`, и она
        печатает честный знаменатель, потому что видит весь список.
        """
        stmt = (
            select(
                Opponent.id,
                Opponent.opponent_nick,
                func.count(OpponentLink.opponent_id),
            )
            .outerjoin(OpponentLink, OpponentLink.opponent_id == Opponent.id)
            .where(Opponent.owner_player_id == owner_player_id)
            .group_by(Opponent.id, Opponent.opponent_nick)
            .order_by(func.lower(Opponent.opponent_nick), Opponent.id)
        )
        return [
            OpponentRecord(opponent_id=opponent_id, nick=nick, links=links)
            for opponent_id, nick, links in await self.db.execute(stmt)
        ]

    async def get(self, opponent_id: int, owner_player_id: int) -> OpponentRecord | None:
        """Оппонент по номеру — только свой; `None`, если номер чужой или его нет."""
        found = [
            record
            for record in await self.list_for_player(owner_player_id)
            if record.opponent_id == opponent_id
        ]
        return found[0] if found else None

    async def links(self, opponent_id: int, owner_player_id: int) -> dict[str, str]:
        """Все привязки оппонента: турнир комнаты → метка участника в нём.

        Форма словаря, а не списка пар, — вход `player_stats_across_tournaments`
        (`analysis/player_stats.py`) один в один. Она не теряет строк: у одного
        оппонента в одном турнире метка не больше одной, и это держит уникальный
        индекс `uq_opponent_links_opponent_tournament`, а не порядок обхода
        (`test_one_opponent_keeps_one_participant_per_tournament`).
        """
        stmt = select(
            OpponentLink.room_tournament_id, OpponentLink.participant_label
        ).where(
            OpponentLink.opponent_id == opponent_id,
            OpponentLink.owner_player_id == owner_player_id,
        )
        return {row[0]: row[1] for row in await self.db.execute(stmt)}

    async def _find_by_nick(self, owner_player_id: int, nick: str) -> int | None:
        return await self.db.scalar(
            select(Opponent.id).where(
                Opponent.owner_player_id == owner_player_id,
                func.lower(Opponent.opponent_nick) == func.lower(nick),
            )
        )
