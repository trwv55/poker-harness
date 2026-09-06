"""Память: миграция 0001 и репозитории — на настоящем Postgres (testcontainers).

`db` (одна транзакция, откатываемая после теста) покрывает все репозитории — их
контракт не зависит от того, коммитит вызывающий по-настоящему или нет. `db_factory`
(настоящие коммиты + TRUNCATE между тестами) в задаче 14 использует только его
собственный тест: первый реальный потребитель — `FOR UPDATE SKIP LOCKED` задачи 15 —
но без своего теста здесь контроллерский рулинг задачи 14 остался бы непроверенным
кодом.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from sqlalchemy import insert, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from harness.contracts import AnalysisResult, RawHand, ScanSummary
from harness.engine import enrich
from harness.memory.models import EvalCase, Job
from harness.memory.repos import (
    _SESSIONS_LOCK_NS,
    _UPSERT_CHUNK,
    AnalysesRepo,
    CalcCacheRepo,
    EvalCasesRepo,
    HandsRepo,
    JobsRepo,
    PlayersRepo,
    SessionsRepo,
    TournamentsRepo,
)
from harness.normalizer import normalize
from harness.parsers.hh_parser import parse_hand
from harness.platform.limiter import PgLimiter
from tests.test_contracts import make_min_raw
from tests.test_hh_parser import SAMPLE

# Ожидание конкурента после освобождения лока: обычные миллисекунды, потолок — на
# случай, если сериализация сломается так, что вторая корутина не проснётся вовсе.
_RIVAL_TIMEOUT_S = 10.0

_ALL_TABLES = {
    "players",
    "invites",
    "sessions",
    "tournaments",
    "hands",
    "analyses",
    "notes",
    "eval_cases",
    "jobs",
    "traces",
    "llm_calls",
    "calc_cache",
}


def _make_enriched():
    """Настоящая рука (не сшитая руками): парсер → нормализатор → движок на SAMPLE
    из `test_hh_parser.py` — тот же фикстур-приём, что и в `test_engine.py`.
    """
    return enrich(normalize(parse_hand(SAMPLE, source_ref="x")))


async def _make_session(db) -> int:
    """Валидный `session_id` для FK: `hands.session_id`/`jobs.session_id` — NOT NULL."""
    player = await PlayersRepo(db).get_or_create(tg_user_id=777)
    session_row = await SessionsRepo(db).active_or_create(player.id)
    return session_row.id


async def test_migration_applies(pg):
    """`alembic upgrade head` создал все 12 таблиц спеки §6 — не просто "не упал"."""
    engine = create_async_engine(pg.get_connection_url(driver="asyncpg"))
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text("select table_name from information_schema.tables where table_schema='public'")
            )
            tables = {row[0] for row in result}
    finally:
        await engine.dispose()
    assert tables >= _ALL_TABLES


async def test_hand_artifacts_roundtrip(db):
    session_id = await _make_session(db)
    raw = RawHand.model_validate(make_min_raw())
    hid = await HandsRepo(db).save_raw(session_id=session_id, raw=raw)
    got = await HandsRepo(db).get(hid)
    assert got.raw == raw and got.canonical is None  # nullable-колонки = чекпоинты


async def test_hand_checkpoints_progress_independently(db):
    session_id = await _make_session(db)
    en = _make_enriched()
    hid = await HandsRepo(db).save_raw(session_id=session_id, raw=RawHand.model_validate(make_min_raw()))

    await HandsRepo(db).save_canonical(hid, en.hand)
    mid = await HandsRepo(db).get(hid)
    assert mid.canonical == en.hand and mid.enriched is None  # ещё не enriched

    await HandsRepo(db).save_enriched(hid, en)
    done = await HandsRepo(db).get(hid)
    assert done.enriched == en


async def test_hands_get_missing_raises(db):
    with pytest.raises(LookupError):
        await HandsRepo(db).get(999_999)


async def test_jobs_required_fields(db):
    with pytest.raises(IntegrityError):
        await db.execute(insert(Job).values(type="hh_scan", status="queued", payload={}))
        await db.commit()  # session_id NOT NULL — спека §6


async def test_jobs_status_check_rejects_unknown_value(db):
    session_id = await _make_session(db)
    player = await PlayersRepo(db).get_or_create(tg_user_id=778)
    with pytest.raises(IntegrityError):
        await db.execute(
            insert(Job).values(
                type="hh_scan",
                status="bogus",  # не входит в CHECK-список статусов
                payload={},
                session_id=session_id,
                player_id=player.id,
            )
        )


async def test_players_get_or_create_is_idempotent(db):
    p1 = await PlayersRepo(db).get_or_create(tg_user_id=42)
    p2 = await PlayersRepo(db).get_or_create(tg_user_id=42)
    assert p1.id == p2.id
    assert p1.subscription == "free"
    assert p1.is_dev is False
    assert p1.quota_daily is None  # нет переопределения — дефолт из Config, не БД


async def test_players_get_or_create_distinguishes_users(db):
    p1 = await PlayersRepo(db).get_or_create(tg_user_id=1)
    p2 = await PlayersRepo(db).get_or_create(tg_user_id=2)
    assert p1.id != p2.id


async def test_sessions_active_or_create_reuses_open_session(db):
    player = await PlayersRepo(db).get_or_create(tg_user_id=10)
    s1 = await SessionsRepo(db).active_or_create(player.id)
    s2 = await SessionsRepo(db).active_or_create(player.id)
    assert s1.id == s2.id


async def test_sessions_active_or_create_ignores_closed_session(db):
    player = await PlayersRepo(db).get_or_create(tg_user_id=11)
    s1 = await SessionsRepo(db).active_or_create(player.id)
    s1.closed_at = datetime.now(UTC)
    await db.flush()

    s2 = await SessionsRepo(db).active_or_create(player.id)
    assert s2.id != s1.id
    assert s2.closed_at is None


async def test_analyses_save_and_get_by_hand(db):
    session_id = await _make_session(db)
    raw = RawHand.model_validate(make_min_raw())
    hid = await HandsRepo(db).save_raw(session_id=session_id, raw=raw)

    result = AnalysisResult(hand_no=raw.hand_no, points=[])
    aid = await AnalysesRepo(db).save(
        hand_id=hid, result=result, verdict_text="норм", range_images=["r1.png"]
    )

    got = await AnalysesRepo(db).get_by_hand(hid)
    assert got is not None
    assert got.id == aid
    assert got.result == result
    assert got.verdict_text == "норм"
    assert got.range_images == ["r1.png"]


async def test_set_explanation_without_text_keeps_the_saved_one(db):
    """Картинки диапазонов рисует код, и сохранить их надо даже когда модель не
    ответила — но пустой текст не имеет права затереть уже сказанное
    (`worker.pipeline`, станция explain)."""
    session_id = await _make_session(db)
    raw = RawHand.model_validate(make_min_raw())
    hid = await HandsRepo(db).save_raw(session_id=session_id, raw=raw)
    await AnalysesRepo(db).save(
        hand_id=hid, result=AnalysisResult(hand_no=raw.hand_no, points=[]), verdict_text="слова"
    )

    await AnalysesRepo(db).set_explanation(hand_id=hid, range_images=["r1.png"])

    got = await AnalysesRepo(db).get_by_hand(hid)
    assert got is not None
    assert got.range_images == ["r1.png"]
    assert got.verdict_text == "слова"


async def test_analyses_get_by_hand_missing_returns_none(db):
    session_id = await _make_session(db)
    raw = RawHand.model_validate(make_min_raw())
    hid = await HandsRepo(db).save_raw(session_id=session_id, raw=raw)
    assert await AnalysesRepo(db).get_by_hand(hid) is None


async def test_eval_cases_add(db):
    session_id = await _make_session(db)
    raw = RawHand.model_validate(make_min_raw())
    hid = await HandsRepo(db).save_raw(session_id=session_id, raw=raw)

    case_id = await EvalCasesRepo(db).add(
        kind="vision_field",
        hand_id=hid,
        ground_truth={"stack": 3891},
        source="escalation",
        field="stacks",
    )
    row = await db.get(EvalCase, case_id)
    assert row is not None
    assert row.kind == "vision_field"
    assert row.field == "stacks"
    assert row.ground_truth == {"stack": 3891}


async def test_db_factory_commits_are_visible_on_a_new_connection(db_factory):
    """Контроллерский рулинг задачи 14: db_factory коммитит по-настоящему, и это
    видно с ДРУГОГО соединения — то, что `db` (откатываемая транзакция)
    принципиально не может показать.
    """
    async with db_factory() as s1:
        player = await PlayersRepo(s1).get_or_create(tg_user_id=555)
        await s1.commit()
        player_id = player.id

    async with db_factory() as s2:
        again = await PlayersRepo(s2).get_or_create(tg_user_id=555)
        assert again.id == player_id  # тот же игрок, увиденный с нового соединения


async def test_db_factory_cleans_state_between_tests(db_factory):
    """Без TRUNCATE после предыдущего теста `tg_user_id=555` уже существовал бы, и
    `get_or_create` вернул бы старый id вместо создания нового — эта проверка ловит
    именно регресс уборки между тестами, а не что-то ещё.
    """
    async with db_factory() as s:
        player = await PlayersRepo(s).get_or_create(tg_user_id=555)
        await s.commit()
        assert player.id == 1  # RESTART IDENTITY — счётчик начат заново


async def test_sessions_lock_is_transaction_scoped(db_factory):
    """Round 5, Item M — инвариант, который комментарий назвать не может.

    `PgLimiter.slot()` первым делом выполняет `SELECT pg_advisory_unlock_all()`
    на соединении из ТОГО ЖЕ движка приложения, а `unlock_all` пространств имён
    не различает вовсе: он снимает все advisory-локи УРОВНЯ СЕССИИ этого
    соединения, чьи бы они ни были. Значит `SessionsRepo` защищён не тем, что у
    него свой namespace (так было написано), а тем, что его лок ТРАНЗАКЦИОННЫЙ
    (`pg_advisory_xact_lock`) — такие живут отдельно и для `unlock_all` невидимы.

    Правка `pg_advisory_xact_lock` → `pg_advisory_lock` согласовывалась бы с
    прежним комментарием слово в слово и молча отменила бы сериализацию сессий
    при первом же вызове модели. Этот тест её ловит: после коммита в `pg_locks`
    не должно остаться ни одного advisory-лока в namespace сессий — с
    session-scoped локом он пережил бы и коммит, и возврат соединения в пул
    (сброс пула — это `ROLLBACK`, а он такие локи не снимает).
    """
    async with db_factory() as session:
        player = await PlayersRepo(session).get_or_create(tg_user_id=90210)
        await SessionsRepo(session).active_or_create(player.id)
        await session.commit()

    async with db_factory() as probe:
        held = await probe.scalar(
            text(
                "SELECT count(*) FROM pg_locks "
                "WHERE locktype = 'advisory' AND classid = :ns"
            ),
            {"ns": _SESSIONS_LOCK_NS},
        )
    assert held == 0, "лок сессий пережил коммит — значит он уровня СЕССИИ, а не транзакции"


async def test_limiter_unlock_all_does_not_steal_the_sessions_lock(db_factory):
    """Вторая половина Item M — то самое последствие, ради которого инвариант и
    существует. Пока `SessionsRepo` держит свой лок в ОТКРЫТОЙ транзакции,
    `PgLimiter.slot()` (со своим `pg_advisory_unlock_all()`) работает на том же
    движке — и сериализация обязана уцелеть: второй писатель по-прежнему ждёт.
    """
    lim = PgLimiter(db_factory, max_concurrency=1, max_per_minute=1000)
    async with db_factory() as holder:
        player = await PlayersRepo(holder).get_or_create(tg_user_id=90211)
        await holder.commit()
        player_id = player.id

        # Лок взят и НЕ отпущен: транзакция ещё открыта.
        await SessionsRepo(holder).active_or_create(player_id)

        async with lim.slot():  # внутри — pg_advisory_unlock_all() на общем движке
            pass

        async with db_factory() as rival:
            waiting = asyncio.create_task(SessionsRepo(rival).active_or_create(player_id))
            done, _ = await asyncio.wait({waiting}, timeout=1.0)
            assert not done, "конкурент прошёл лок — сериализация сессий снята"

            await holder.commit()  # отпускаем лок
            await asyncio.wait_for(waiting, timeout=_RIVAL_TIMEOUT_S)
            await rival.rollback()


async def test_calc_cache_upsert_survives_more_rows_than_one_statement_allows(db):
    """Пачка длиннее одного INSERT записывается целиком, а не роняет задачу.

    Строка кэша стоит двух связанных параметров, а Postgres принимает не больше
    32767 на запрос: с 16384-й строки asyncpg роняет весь запрос. Кэш эквити
    растёт от турнира к турниру, и этот рубеж переходит — поймано прогоном, где
    накопленный дисковый кэш дорос до 16884 записей, и КАЖДАЯ задача воркера
    стала падать на записи в `calc_cache`, то есть отказ не деградация, а полная
    остановка обработки.

    Проверяется настоящий предел протокола (больше 16383 строк), а не
    `_UPSERT_CHUNK`: тест обязан краснеть на неразбитой реализации, а не на
    неудачно выбранном размере куска.
    """
    rows = 16_500
    assert rows > 32_767 // 2  # тот самый рубеж, а не произвольное большое число
    entries = {f"k{i}": float(i) for i in range(rows)}

    repo = CalcCacheRepo(db)
    await repo.upsert_many("equity_mc:test:", entries)

    stored = await repo.get_all("equity_mc:test:")
    assert len(stored) == rows
    assert stored["k0"] == 0.0 and stored[f"k{rows - 1}"] == float(rows - 1)
    assert _UPSERT_CHUNK * 2 <= 32_767  # кусок обязан помещаться в предел протокола


# --- история игрока для отчёта по турниру (задача 23) ------------------------------


async def _player_with_session(db, tg_user_id: int) -> tuple[int, int]:
    player = await PlayersRepo(db).get_or_create(tg_user_id=tg_user_id)
    session_row = await SessionsRepo(db).active_or_create(player.id)
    return player.id, session_row.id


async def _save_hand_in(db, *, session_id: int, tournament_id: int, hand_no: str) -> int:
    """Рука, доведённая до чекпоинта `canonical`, — та форма, которую читает отчёт.

    Из `SAMPLE`, а не из `make_min_raw()`: у той одно место за столом, и
    нормализатор такую руку не раскладывает по позициям вовсе.
    """
    raw = parse_hand(SAMPLE, source_ref="x").model_copy(update={"hand_no": hand_no})
    hid = await HandsRepo(db).save_raw(
        session_id=session_id, tournament_id=tournament_id, raw=raw
    )
    await HandsRepo(db).save_canonical(hid, normalize(raw))
    return hid


async def test_player_hands_come_grouped_by_tournament(db):
    player_id, session_id = await _player_with_session(db, tg_user_id=4001)
    first = await TournamentsRepo(db).create(session_id=session_id, source_file="a.txt")
    second = await TournamentsRepo(db).create(session_id=session_id, source_file="b.txt")
    await _save_hand_in(db, session_id=session_id, tournament_id=first, hand_no="A1")
    await _save_hand_in(db, session_id=session_id, tournament_id=first, hand_no="A2")
    await _save_hand_in(db, session_id=session_id, tournament_id=second, hand_no="B1")

    grouped = await HandsRepo(db).player_hands_by_tournament(player_id)

    assert [[h.hand_no for h in group] for group in grouped] == [["A1", "A2"], ["B1"]]


async def test_player_hands_do_not_leak_between_players(db):
    mine, my_session = await _player_with_session(db, tg_user_id=4002)
    _theirs, their_session = await _player_with_session(db, tg_user_id=4003)
    my_tournament = await TournamentsRepo(db).create(session_id=my_session, source_file="a.txt")
    their_tournament = await TournamentsRepo(db).create(
        session_id=their_session, source_file="b.txt"
    )
    await _save_hand_in(db, session_id=my_session, tournament_id=my_tournament, hand_no="MINE")
    await _save_hand_in(
        db, session_id=their_session, tournament_id=their_tournament, hand_no="THEIRS"
    )

    grouped = await HandsRepo(db).player_hands_by_tournament(mine)

    assert [h.hand_no for group in grouped for h in group] == ["MINE"]


async def test_player_hands_skip_a_hand_that_never_reached_canonical(db):
    """Рука на чекпоинте `raw` — не отсутствие данных, но статистику по ней не считают."""
    player_id, session_id = await _player_with_session(db, tg_user_id=4004)
    tournament_id = await TournamentsRepo(db).create(session_id=session_id, source_file="a.txt")
    await _save_hand_in(db, session_id=session_id, tournament_id=tournament_id, hand_no="DONE")
    await HandsRepo(db).save_raw(
        session_id=session_id,
        tournament_id=tournament_id,
        raw=parse_hand(SAMPLE, source_ref="x").model_copy(update={"hand_no": "RAW_ONLY"}),
    )

    grouped = await HandsRepo(db).player_hands_by_tournament(player_id)

    assert [[h.hand_no for h in group] for group in grouped] == [["DONE"]]


async def test_past_scan_summaries_exclude_the_tournament_being_reported(db):
    """Сводка текущего турнира уже сохранена к моменту отчёта — и в «прошлые» не идёт.

    Иначе каждая находка этого турнира выглядела бы как «то же самое было
    раньше», хотя раньше её не было.
    """
    player_id, session_id = await _player_with_session(db, tg_user_id=4005)
    old = await TournamentsRepo(db).create(session_id=session_id, source_file="old.txt")
    current = await TournamentsRepo(db).create(session_id=session_id, source_file="now.txt")
    old_summary = ScanSummary(
        hands_total=9, hands_with_decision=9, items=[], total_loss_bb=-1.0
    )
    current_summary = ScanSummary(
        hands_total=7, hands_with_decision=7, items=[], total_loss_bb=-2.0
    )
    await TournamentsRepo(db).save_scan_summary(old, old_summary)
    await TournamentsRepo(db).save_scan_summary(current, current_summary)

    past = await TournamentsRepo(db).player_scan_summaries(player_id, exclude=current)

    assert past == [old_summary]


async def test_past_scan_summaries_ignore_tournaments_without_one(db):
    """Турнир, скан которого не дошёл до сводки, в историю не попадает."""
    player_id, session_id = await _player_with_session(db, tg_user_id=4006)
    await TournamentsRepo(db).create(session_id=session_id, source_file="unfinished.txt")

    assert await TournamentsRepo(db).player_scan_summaries(player_id, exclude=0) == []


async def test_past_scan_summaries_do_not_leak_between_players(db):
    mine, _my_session = await _player_with_session(db, tg_user_id=4007)
    _theirs, their_session = await _player_with_session(db, tg_user_id=4008)
    their_tournament = await TournamentsRepo(db).create(
        session_id=their_session, source_file="theirs.txt"
    )
    await TournamentsRepo(db).save_scan_summary(
        their_tournament,
        ScanSummary(hands_total=1, hands_with_decision=1, items=[], total_loss_bb=0.0),
    )

    assert await TournamentsRepo(db).player_scan_summaries(mine, exclude=0) == []


# --- задача 22: ник в руме и патч сырой руки ---------------------------------


async def test_migration_0003_gives_players_a_room_nickname_column(pg):
    """Миграция 0003 прокатана тем же путём, что 0001 и 0002 — на живом Postgres.

    Колонка nullable: у игроков, заведённых раньше, ника нет, и разбор скрина у
    них упирается в вопрос игроку, а не в отказ.
    """
    engine = create_async_engine(pg.get_connection_url(driver="asyncpg"))
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "select column_name, is_nullable, character_maximum_length "
                    "from information_schema.columns "
                    "where table_name='players' and column_name='gg_nickname'"
                )
            )
            row = result.first()
    finally:
        await engine.dispose()
    assert row is not None
    assert (row[1], row[2]) == ("YES", 64)


async def test_the_room_nickname_is_written_once_and_read_back(db):
    player = await PlayersRepo(db).get_or_create(tg_user_id=4242)
    assert player.gg_nickname is None
    await PlayersRepo(db).set_gg_nickname(player.id, "  nick_on_screen  ")
    await db.refresh(player)
    assert player.gg_nickname == "nick_on_screen"


async def test_an_empty_room_nickname_is_refused_rather_than_stored(db):
    """«Ник неизвестен» — это NULL; пустая строка была бы вторым таким значением."""
    player = await PlayersRepo(db).get_or_create(tg_user_id=4243)
    with pytest.raises(ValueError, match="пустым"):
        await PlayersRepo(db).set_gg_nickname(player.id, "   ")


async def test_patching_the_raw_hand_resets_the_checkpoints_below_it(db):
    """Спека §8.3: ответ игрока патчит `raw`, а `canonical`/`enriched` пересчитываются.

    Строка с новым `raw` и старым `enriched` описывала бы две разные руки сразу,
    поэтому сброс идёт той же записью, что и патч.
    """
    session_id = await _make_session(db)
    en = _make_enriched()
    hid = await HandsRepo(db).save_raw(
        session_id=session_id, raw=RawHand.model_validate(make_min_raw())
    )
    await HandsRepo(db).save_canonical(hid, en.hand)
    await HandsRepo(db).save_enriched(hid, en)

    patched = RawHand.model_validate(make_min_raw(level=99))
    await HandsRepo(db).replace_raw(hid, patched)

    got = await HandsRepo(db).get(hid)
    assert got.raw.level == 99
    assert got.canonical is None and got.enriched is None


async def _awaiting(db, player_id: int, session_id: int, **payload) -> int:
    row = await db.execute(
        insert(Job)
        .values(
            type="screenshot_analyze",
            status="awaiting_user",
            payload=payload,
            session_id=session_id,
            player_id=player_id,
        )
        .returning(Job.id)
    )
    return row.scalar_one()


async def test_the_job_waiting_for_the_player_is_found_by_its_own_number(db):
    """Состояние ожидания живёт в `jobs`, а не в памяти бота: оно переживает перезапуск."""
    session_id = await _make_session(db)
    player = await PlayersRepo(db).get_or_create(tg_user_id=777)
    job_id = await _awaiting(db, player.id, session_id, escalation_field="pot")
    found = await JobsRepo(db).get_awaiting(job_id, player.id)
    assert found is not None and found.payload["escalation_field"] == "pot"


async def test_a_waiting_job_of_another_player_is_not_reachable_by_its_number(db):
    """Номер задачи приходит из внешнего мира, и одной его мало."""
    session_id = await _make_session(db)
    mine = await PlayersRepo(db).get_or_create(tg_user_id=777)
    stranger = await PlayersRepo(db).get_or_create(tg_user_id=779)
    job_id = await _awaiting(db, mine.id, session_id, escalation_field="pot")
    assert await JobsRepo(db).get_awaiting(job_id, stranger.id) is None


async def test_no_waiting_job_is_not_an_error(db):
    player = await PlayersRepo(db).get_or_create(tg_user_id=778)
    assert await JobsRepo(db).get_awaiting(1, player.id) is None
    assert await JobsRepo(db).awaiting_manual_entry(player.id) is None


async def test_manual_entry_is_looked_up_by_the_started_input_not_by_the_status(db):
    """Обычное сообщение номера задачи не несёт — адресат ищется по начатому вводу.

    У соседней ждущей задачи ввод не начинали, и подставлять число в неё нельзя
    (спека §8.1: ждущих задач у игрока бывает несколько сразу).
    """
    session_id = await _make_session(db)
    player = await PlayersRepo(db).get_or_create(tg_user_id=777)
    started = await _awaiting(db, player.id, session_id, escalation_field="pot", manual_entry="pot")
    await _awaiting(db, player.id, session_id, escalation_field="cards")
    found = await JobsRepo(db).awaiting_manual_entry(player.id)
    assert found is not None and found.id == started
