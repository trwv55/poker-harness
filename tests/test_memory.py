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
    "decision_points",
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
        hand_id=hid,
        result=result,
        decision_points=[],
        verdict_text="норм",
        range_images=["r1.png"],
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
        hand_id=hid,
        result=AnalysisResult(hand_no=raw.hand_no, points=[]),
        decision_points=[],
        verdict_text="слова",
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


# --- задача 23: лики, агрегат сессии, заметки, инвайты ------------------------


def _verdict(spot, taken: str, best: str, ev_diff_bb: float, **over):
    """Точка решения с заданной тройкой — вход таксономии ликов."""
    from harness.contracts import PointVerdict, Street, Zone

    return PointVerdict(
        dp_index=0,
        street=Street.PREFLOP,
        spot=spot,
        zone=Zone.STRICT,
        action_taken=taken,
        best_action=best,
        ev_diff_bb=ev_diff_bb,
        **over,
    )


async def _save_analysis(db, *, session_id: int, hand_no: str, points) -> int:
    """Рука с сохранённым разбором — то, из чего считаются лики и агрегат вечера."""
    from harness.contracts import AnalysisResult

    raw = RawHand.model_validate(make_min_raw(hand_no=hand_no))
    hand_id = await HandsRepo(db).save_raw(session_id=session_id, raw=raw)
    result = AnalysisResult(hand_no=hand_no, points=list(points))
    await AnalysesRepo(db).save(hand_id=hand_id, result=result, decision_points=[])
    return hand_id


async def test_leaks_are_grouped_by_type_over_the_whole_history(db):
    """Тип лика — тройка «спот · сыграно · лучше», и он копится по всем сессиям.

    Две одинаковых тройки из РАЗНЫХ вечеров обязаны сложиться в один тип с
    частотой два: «типовые ошибки копятся по всей истории» (SESSIONS_UX).
    """
    from harness.contracts import SpotKind
    from harness.memory.repos import LeaksRepo

    player_id, first = await _player_with_session(db, tg_user_id=5001)
    await _save_analysis(
        db,
        session_id=first,
        hand_no="L1",
        points=[_verdict(SpotKind.PUSHFOLD_UNOPENED, "fold", "shove", -2.0)],
    )
    await SessionsRepo(db).close_active(player_id)
    second = (await SessionsRepo(db).active_or_create(player_id)).id
    await _save_analysis(
        db,
        session_id=second,
        hand_no="L2",
        points=[
            _verdict(SpotKind.PUSHFOLD_UNOPENED, "fold", "shove", -1.0),
            _verdict(SpotKind.PUSHFOLD_FACING_SHOVE, "call", "fold", -5.0),
        ],
    )

    leaks = await LeaksRepo(db).by_type(player_id)

    assert [(stat.rule.key, stat.count, round(stat.loss_bb, 2)) for stat in leaks] == [
        ("call_too_wide", 1, 5.0),
        ("no_shove", 2, 3.0),
    ]


async def test_leaks_ignore_near_zero_and_unjudged_points(db):
    """В лики не входят ни точки «около нуля», ни точки без вердикта.

    У первых в `best_action` стоит русская фраза ядра, у вторых — пустая
    строка; ни то, ни другое не совпадает ни с одной тройкой таблицы правил.
    """
    from harness.contracts import EvInterval, SpotKind
    from harness.memory.repos import LeaksRepo

    player_id, session_id = await _player_with_session(db, tg_user_id=5002)
    await _save_analysis(
        db,
        session_id=session_id,
        hand_no="N1",
        points=[
            _verdict(
                SpotKind.PUSHFOLD_UNOPENED,
                "fold",
                "около нуля, оба варианта допустимы",
                0.0,
                interval=EvInterval(point_bb=0.0, low_bb=-0.2, high_bb=0.3, near_zero=True),
            ),
            _verdict(SpotKind.POSTFLOP, "call", "", 0.0),
        ],
    )

    assert await LeaksRepo(db).by_type(player_id) == []


async def test_leak_coverage_counts_every_point_of_the_history(db):
    """Строка «оценено N из M решений» — по тому же правилу, что покрытие скана."""
    from harness.contracts import SpotKind
    from harness.memory.repos import LeaksRepo

    player_id, session_id = await _player_with_session(db, tg_user_id=5003)
    await _save_analysis(
        db,
        session_id=session_id,
        hand_no="C1",
        points=[
            _verdict(SpotKind.PUSHFOLD_UNOPENED, "fold", "shove", -2.0),
            _verdict(SpotKind.POSTFLOP, "call", "", 0.0),
            _verdict(SpotKind.PREFLOP_OTHER, "raise", "", 0.0),
        ],
    )

    overview = await LeaksRepo(db).overview(player_id)

    assert (overview.points_judged, overview.points_total) == (1, 3)
    assert [stat.rule.key for stat in overview.leaks] == ["no_shove"]


async def test_leaks_do_not_leak_between_players(db):
    """Чужая история — не моя статистика: фильтр по игроку идёт через сессию."""
    from harness.contracts import SpotKind
    from harness.memory.repos import LeaksRepo

    mine, my_session = await _player_with_session(db, tg_user_id=5004)
    _theirs, their_session = await _player_with_session(db, tg_user_id=5005)
    await _save_analysis(
        db,
        session_id=their_session,
        hand_no="X1",
        points=[_verdict(SpotKind.PUSHFOLD_UNOPENED, "shove", "fold", -3.0)],
    )
    await _save_analysis(
        db,
        session_id=my_session,
        hand_no="M1",
        points=[_verdict(SpotKind.PUSHFOLD_UNOPENED, "fold", "shove", -1.0)],
    )

    assert [stat.rule.key for stat in await LeaksRepo(db).by_type(mine)] == ["no_shove"]


async def test_the_session_summary_counts_the_evening_and_names_its_leak(db):
    """Агрегат вечера: турниры, разобранные руки, цена расхождений, лик вечера."""
    from harness.contracts import SpotKind

    player_id, session_id = await _player_with_session(db, tg_user_id=5006)
    await TournamentsRepo(db).create(session_id=session_id, source_file="a.txt")
    await _save_analysis(
        db,
        session_id=session_id,
        hand_no="S1",
        points=[_verdict(SpotKind.PUSHFOLD_FACING_SHOVE, "fold", "call", -1.5)],
    )
    await _save_analysis(
        db,
        session_id=session_id,
        hand_no="S2",
        points=[_verdict(SpotKind.PUSHFOLD_FACING_SHOVE, "fold", "call", -0.5)],
    )
    # Рука без разбора: сохранена, но в «разобрано» не входит.
    await HandsRepo(db).save_raw(
        session_id=session_id, raw=RawHand.model_validate(make_min_raw(hand_no="S3"))
    )

    summary = await SessionsRepo(db).summary(session_id, player_id)

    assert summary is not None
    assert (summary.tournaments, summary.hands) == (1, 2)
    assert round(summary.loss_bb, 2) == 2.0
    assert summary.top_leak is not None and summary.top_leak.rule.key == "fold_vs_shove"
    assert (summary.points_judged, summary.points_total) == (2, 2)


async def test_a_session_summary_of_another_player_is_not_reachable_by_its_number(db):
    """Номер сессии приезжает из `callback_data` — то есть из внешнего мира."""
    mine, _my_session = await _player_with_session(db, tg_user_id=5007)
    _theirs, their_session = await _player_with_session(db, tg_user_id=5008)

    assert await SessionsRepo(db).summary(their_session, mine) is None


async def test_the_session_list_marks_the_open_evening(db):
    """Список сессий: свежие первыми, активная помечена — по ней и идёт разбор."""
    player_id, first = await _player_with_session(db, tg_user_id=5009)
    await SessionsRepo(db).close_active(player_id)
    second = (await SessionsRepo(db).active_or_create(player_id)).id

    lines = await SessionsRepo(db).list_for_player(player_id)

    assert [line.session_id for line in lines] == [second, first]
    assert [line.is_active for line in lines] == [True, False]


async def test_the_session_count_does_not_depend_on_the_page_size(db):
    """Счёт вечеров игрока — отдельный запрос, а не длина отданной страницы."""
    player_id, _first = await _player_with_session(db, tg_user_id=5017)
    for _ in range(4):
        await SessionsRepo(db).close_active(player_id)
        await SessionsRepo(db).active_or_create(player_id)

    assert len(await SessionsRepo(db).list_for_player(player_id, limit=2)) == 2
    assert await SessionsRepo(db).count_for_player(player_id) == 5


async def test_a_note_is_one_per_opponent_and_editing_keeps_its_colour(db):
    """Заметка накапливается на оппоненте: вторая запись — правка, а не дубль.

    Цвет ставится отдельной кнопкой, поэтому правка текста его не стирает.
    """
    from harness.memory.repos import NotesRepo

    player_id, _session_id = await _player_with_session(db, tg_user_id=5010)
    notes = NotesRepo(db)
    first = await notes.upsert(owner_player_id=player_id, nick="villain", text_="фолдит на опен")
    await notes.set_color(first, player_id, "red")
    again = await notes.upsert(owner_player_id=player_id, nick="villain", text_="донкает флоп")

    assert again == first
    stored = await notes.get(first, player_id)
    assert stored is not None
    assert (stored.text, stored.color) == ("донкает флоп", "red")
    assert len(await notes.list_for_player(player_id)) == 1


def test_the_note_upsert_docstring_points_at_a_test_that_exists():
    """Ссылка на тест обязана вести к тесту: иначе гарантия только на словах."""
    import re

    from harness.memory.repos import NotesRepo

    referenced = re.findall(r"`(test_\w+)`", NotesRepo.upsert.__doc__ or "")
    assert referenced
    assert all(name in globals() for name in referenced), referenced


async def test_an_empty_note_is_refused_rather_than_stored(db):
    from harness.memory.repos import NotesRepo

    player_id, _session_id = await _player_with_session(db, tg_user_id=5011)
    with pytest.raises(ValueError, match="пустой"):
        await NotesRepo(db).upsert(owner_player_id=player_id, nick="villain", text_="   ")


async def test_a_note_longer_than_the_limit_is_refused_rather_than_cut(db):
    """Потолок держит экран «Заметки» открываемым, поэтому он инвариант хранилища.

    Обрезать нельзя: игрок увидел бы на экране не то, что написал.
    """
    from harness.contracts import MAX_NOTE_TEXT_CHARS
    from harness.memory.repos import NotesRepo

    player_id, _session_id = await _player_with_session(db, tg_user_id=5015)
    with pytest.raises(ValueError, match="длиннее"):
        await NotesRepo(db).upsert(
            owner_player_id=player_id, nick="villain", text_="я" * (MAX_NOTE_TEXT_CHARS + 1)
        )
    assert await NotesRepo(db).count_for_player(player_id) == 0


async def test_the_note_count_does_not_depend_on_the_page_size(db):
    """Счёт заметок игрока — отдельный запрос, а не длина отданной страницы."""
    from harness.memory.repos import NotesRepo

    player_id, _session_id = await _player_with_session(db, tg_user_id=5016)
    notes = NotesRepo(db)
    for index in range(7):
        await notes.upsert(owner_player_id=player_id, nick=f"opp{index}", text_="фолдит")

    assert len(await notes.list_for_player(player_id, limit=3)) == 3
    assert await notes.count_for_player(player_id) == 7


async def test_a_note_of_another_player_is_neither_read_nor_deleted_by_its_number(db):
    """Номер заметки приезжает кнопкой из внешнего мира — владелец сверяется всегда."""
    from harness.memory.repos import NotesRepo

    mine, _my_session = await _player_with_session(db, tg_user_id=5012)
    theirs, _their_session = await _player_with_session(db, tg_user_id=5013)
    notes = NotesRepo(db)
    foreign = await notes.upsert(owner_player_id=theirs, nick="villain", text_="их заметка")

    assert await notes.get(foreign, mine) is None
    assert await notes.delete(foreign, mine) is False
    assert await notes.set_color(foreign, mine, "red") is False
    assert await notes.delete(foreign, theirs) is True


async def test_a_note_and_a_link_on_the_same_nick_in_two_cases_meet_on_one_opponent(db):
    """Заметка и сшивки турниров — про одного человека, и строка у него одна.

    До миграции 0007 личность жила в двух местах с разными правилами регистра:
    `Vasya` и `vasya` были одним оппонентом для сшивок и двумя для заметок.
    """
    from harness.memory.repos import NotesRepo, OpponentsRepo

    owner, _session = await _player_with_session(db, tg_user_id=5019)
    opponents = OpponentsRepo(db)
    opponent_id = await opponents.get_or_create(owner_player_id=owner, nick="Vasya")
    await opponents.link(
        owner_player_id=owner,
        opponent_id=opponent_id,
        room_tournament_id="T1",
        participant_label="p1",
    )
    notes = NotesRepo(db)
    note_id = await notes.upsert(owner_player_id=owner, nick="vASYA", text_="фолдит на опен")

    again = await notes.upsert(owner_player_id=owner, nick="  Vasya ", text_="донкает флоп")
    found = await notes.find_by_nick(owner, "VASYA")

    assert again == note_id
    assert found is not None and found.note_id == note_id
    # Написание — то, которым ника назвали первым: заметка второго не заводит.
    assert found.nick == "Vasya"
    assert [(o.nick, o.links) for o in await opponents.list_for_player(owner)] == [("Vasya", 1)]


async def test_deleting_a_note_leaves_the_opponent_and_his_links(db):
    """Удалена заметка — не человек: сшивки турниров о ней не знают."""
    from harness.memory.repos import NotesRepo, OpponentsRepo

    owner, _session = await _player_with_session(db, tg_user_id=5020)
    opponents = OpponentsRepo(db)
    opponent_id = await opponents.get_or_create(owner_player_id=owner, nick="Vasya")
    await opponents.link(
        owner_player_id=owner,
        opponent_id=opponent_id,
        room_tournament_id="T1",
        participant_label="p1",
    )
    note_id = await NotesRepo(db).upsert(
        owner_player_id=owner, nick="Vasya", text_="фолдит на опен"
    )

    assert await NotesRepo(db).delete(note_id, owner) is True
    assert await opponents.links(opponent_id, owner) == {"T1": "p1"}
    assert [(o.nick, o.links) for o in await opponents.list_for_player(owner)] == [("Vasya", 1)]


async def test_an_invite_code_opens_the_door_exactly_once(db):
    """Инвайт гасится одним UPDATE: второй игрок с тем же кодом внутрь не попадает."""
    from harness.memory.repos import InvitesRepo

    owner, _session_id = await _player_with_session(db, tg_user_id=5014)
    guest, _guest_session = await _player_with_session(db, tg_user_id=5015)
    latecomer, _late_session = await _player_with_session(db, tg_user_id=5016)
    invites = InvitesRepo(db)
    code = await invites.mint(owner)

    assert await invites.redeem(code, guest) is True
    assert await invites.redeem(code, latecomer) is False
    assert await invites.redeem("не-код", latecomer) is False


async def test_minted_invite_codes_differ(db):
    from harness.memory.repos import InvitesRepo

    owner, _session_id = await _player_with_session(db, tg_user_id=5017)
    invites = InvitesRepo(db)
    codes = {await invites.mint(owner) for _ in range(5)}
    assert len(codes) == 5


async def test_pending_input_is_remembered_and_cleared(db):
    """Что означает следующее текстовое сообщение, помнит БД, а не память бота."""
    players = PlayersRepo(db)
    player = await players.get_or_create(tg_user_id=5018)
    assert player.pending_input is None

    await players.set_pending_input(player.id, {"kind": "note", "nick": "villain"})
    await db.refresh(player)
    assert player.pending_input == {"kind": "note", "nick": "villain"}

    await players.set_pending_input(player.id, None)
    await db.refresh(player)
    assert player.pending_input is None


async def test_the_note_column_and_uniqueness_are_in_the_live_schema(pg):
    """Прокатано на живом Postgres: `players.pending_input` (0005) и одна
    заметка на оппонента — уникальный индекс по `notes.opponent_id` (0007).
    """
    engine = create_async_engine(pg.get_connection_url(driver="asyncpg"))
    try:
        async with engine.connect() as conn:
            column = (
                await conn.execute(
                    text(
                        "select is_nullable from information_schema.columns "
                        "where table_name='players' and column_name='pending_input'"
                    )
                )
            ).first()
            index = (
                await conn.execute(
                    text(
                        "select indexdef from pg_indexes where tablename='notes' "
                        "and indexname='uq_notes_opponent_id'"
                    )
                )
            ).first()
    finally:
        await engine.dispose()
    assert column is not None and column[0] == "YES"
    assert index is not None and "UNIQUE" in index[0]


# --- оппонент: ник в руме и его метки в турнирах ------------------------------------


async def _opponent(db, owner_player_id: int, nick: str) -> int:
    from harness.memory.repos import OpponentsRepo

    return await OpponentsRepo(db).get_or_create(owner_player_id=owner_player_id, nick=nick)


async def test_the_same_nick_in_another_case_is_the_same_opponent(db):
    """Регистр не различается, написание хранится первое (решение владельца).

    Иначе собственная опечатка в регистре развела бы одного человека на двоих, и
    его частоты посчитались бы по половине истории каждая.
    """
    from harness.memory.repos import OpponentsRepo

    owner, _session = await _player_with_session(db, tg_user_id=6001)
    first = await _opponent(db, owner, "Vasya")
    again = await _opponent(db, owner, "  vASYA ")

    assert again == first
    assert [(a.nick, a.links) for a in await OpponentsRepo(db).list_for_player(owner)] == [
        ("Vasya", 0)
    ]


async def test_an_empty_nick_is_refused_rather_than_stored(db):
    owner, _session = await _player_with_session(db, tg_user_id=6002)
    with pytest.raises(ValueError, match="пустым"):
        await _opponent(db, owner, "   ")


async def test_a_participant_already_bound_keeps_his_first_nick(db):
    """Метка участника в турнире принадлежит одному нику, и второй её не отнимает.

    Возвращается номер того ника, за которым метка закреплена: вызывающему надо
    сказать игроку, с чем именно спорит его команда.
    """
    from harness.memory.repos import OpponentsRepo

    owner, _session = await _player_with_session(db, tg_user_id=6003)
    first = await _opponent(db, owner, "Vasya")
    second = await _opponent(db, owner, "Petya")
    opponents = OpponentsRepo(db)

    mine = await opponents.link(
        owner_player_id=owner, opponent_id=first, room_tournament_id="T1", participant_label="p1"
    )
    stolen = await opponents.link(
        owner_player_id=owner, opponent_id=second, room_tournament_id="T1", participant_label="p1"
    )

    assert mine == first
    assert stolen == first
    assert await opponents.links(second, owner) == {}


async def test_binding_the_same_pair_again_changes_nothing(db):
    from harness.memory.repos import OpponentsRepo

    owner, _session = await _player_with_session(db, tg_user_id=6004)
    opponent_id = await _opponent(db, owner, "Vasya")
    opponents = OpponentsRepo(db)

    once = await opponents.link(
        owner_player_id=owner, opponent_id=opponent_id, room_tournament_id="T1", participant_label="p1"
    )
    twice = await opponents.link(
        owner_player_id=owner, opponent_id=opponent_id, room_tournament_id="T1", participant_label="p1"
    )

    assert (once, twice) == (opponent_id, opponent_id)
    assert await opponents.links(opponent_id, owner) == {"T1": "p1"}


async def test_one_opponent_keeps_one_participant_per_tournament(db):
    """В турнире у участника один идентификатор, поэтому второй у того же ника —
    отказ: иначе в статистику одного человека сложились бы двое, и раздачи этого
    турнира посчитались бы дважды.
    """
    from harness.memory.repos import OpponentsRepo

    owner, _session = await _player_with_session(db, tg_user_id=6005)
    opponent_id = await _opponent(db, owner, "Vasya")
    opponents = OpponentsRepo(db)
    await opponents.link(
        owner_player_id=owner, opponent_id=opponent_id, room_tournament_id="T1", participant_label="p1"
    )

    with pytest.raises(ValueError, match="уже есть метка"):
        await opponents.link(
            owner_player_id=owner,
            opponent_id=opponent_id,
            room_tournament_id="T1",
            participant_label="p2",
        )
    assert await opponents.links(opponent_id, owner) == {"T1": "p1"}


async def test_the_bindings_of_an_opponent_come_as_tournament_to_label(db):
    """Форма привязок — вход `player_stats_across_tournaments` один в один."""
    from harness.memory.repos import OpponentsRepo

    owner, _session = await _player_with_session(db, tg_user_id=6006)
    opponent_id = await _opponent(db, owner, "Vasya")
    opponents = OpponentsRepo(db)
    await opponents.link(
        owner_player_id=owner, opponent_id=opponent_id, room_tournament_id="T1", participant_label="p1"
    )
    await opponents.link(
        owner_player_id=owner, opponent_id=opponent_id, room_tournament_id="T2", participant_label="zz"
    )

    assert await opponents.links(opponent_id, owner) == {"T1": "p1", "T2": "zz"}
    assert [(a.nick, a.links) for a in await opponents.list_for_player(owner)] == [("Vasya", 2)]


async def test_an_opponent_of_another_player_takes_no_bindings(db):
    """Номер оппонента приходит аргументом — владелец сверяется тем же оператором,
    который вставляет: составной внешний ключ не даёт записать чужого.
    """
    from harness.memory.repos import OpponentsRepo

    mine, _my_session = await _player_with_session(db, tg_user_id=6007)
    theirs, _their_session = await _player_with_session(db, tg_user_id=6008)
    foreign = await _opponent(db, theirs, "Vasya")
    opponents = OpponentsRepo(db)

    assert (
        await opponents.link(
            owner_player_id=mine,
            opponent_id=foreign,
            room_tournament_id="T1",
            participant_label="p1",
        )
        is None
    )
    assert await opponents.links(foreign, mine) == {}
    assert await opponents.get(foreign, mine) is None
    assert await opponents.list_for_player(mine) == []


async def test_the_same_nick_of_two_players_is_two_opponents(db):
    """Ник уникален у ВЛАДЕЛЬЦА: один и тот же оппонент у двоих — две строки."""
    mine, _my_session = await _player_with_session(db, tg_user_id=6009)
    theirs, _their_session = await _player_with_session(db, tg_user_id=6010)

    assert await _opponent(db, mine, "Vasya") != await _opponent(db, theirs, "Vasya")


async def test_unbinding_frees_the_participant_for_another_nick(db):
    """Ошибку владельца можно отменить: отвязали — метка свободна."""
    from harness.memory.repos import OpponentsRepo

    owner, _session = await _player_with_session(db, tg_user_id=6011)
    wrong = await _opponent(db, owner, "Vasya")
    right = await _opponent(db, owner, "Petya")
    opponents = OpponentsRepo(db)
    await opponents.link(
        owner_player_id=owner, opponent_id=wrong, room_tournament_id="T1", participant_label="p1"
    )

    assert (
        await opponents.unlink(
            owner_player_id=owner, room_tournament_id="T1", participant_label="p1"
        )
        is True
    )
    assert (
        await opponents.link(
            owner_player_id=owner,
            opponent_id=right,
            room_tournament_id="T1",
            participant_label="p1",
        )
        == right
    )
    assert await opponents.links(wrong, owner) == {}


async def test_a_binding_of_another_player_is_not_removed_by_its_pair(db):
    from harness.memory.repos import OpponentsRepo

    mine, _my_session = await _player_with_session(db, tg_user_id=6012)
    theirs, _their_session = await _player_with_session(db, tg_user_id=6013)
    foreign = await _opponent(db, theirs, "Vasya")
    opponents = OpponentsRepo(db)
    await opponents.link(
        owner_player_id=theirs, opponent_id=foreign, room_tournament_id="T1", participant_label="p1"
    )

    assert (
        await opponents.unlink(
            owner_player_id=mine, room_tournament_id="T1", participant_label="p1"
        )
        is False
    )
    assert await opponents.links(foreign, theirs) == {"T1": "p1"}


async def test_deleting_an_opponent_takes_his_bindings_with_him(db):
    """`ON DELETE CASCADE`: привязка без ника не значит ничего и переживать его
    не должна — висящих ссылок после удаления не остаётся.
    """
    from sqlalchemy import delete as sql_delete
    from sqlalchemy import select as sql_select

    from harness.memory.models import Opponent, OpponentLink
    from harness.memory.repos import OpponentsRepo

    owner, _session = await _player_with_session(db, tg_user_id=6014)
    opponent_id = await _opponent(db, owner, "Vasya")
    await OpponentsRepo(db).link(
        owner_player_id=owner, opponent_id=opponent_id, room_tournament_id="T1", participant_label="p1"
    )

    await db.execute(sql_delete(Opponent).where(Opponent.id == opponent_id))
    await db.flush()

    left = (await db.execute(sql_select(OpponentLink.opponent_id))).all()
    assert left == []


async def test_a_binding_outlives_the_tournament_row_it_came_from(db):
    """Привязка держится за номер турнира КОМНАТЫ, а не за строку `tournaments`.

    Один турнир бывает загружен дважды и строк даёт две; удаление строки не
    оставляет висящей ссылки (внешнего ключа туда нет вовсе) и не стирает того,
    что владелец уже сказал.
    """
    from sqlalchemy import delete as sql_delete

    from harness.memory.models import Hand as HandRow
    from harness.memory.models import Tournament as TournamentRow
    from harness.memory.repos import OpponentsRepo

    owner, session_id = await _player_with_session(db, tg_user_id=6015)
    tournament_row = await TournamentsRepo(db).create(session_id=session_id, source_file="a.txt")
    hand_id = await _save_hand_in(
        db, session_id=session_id, tournament_id=tournament_row, hand_no="A1"
    )
    hand = await HandsRepo(db).get(hand_id)
    assert hand.canonical is not None
    opponent_id = await _opponent(db, owner, "Vasya")
    await OpponentsRepo(db).link(
        owner_player_id=owner,
        opponent_id=opponent_id,
        room_tournament_id=hand.canonical.tournament_id,
        participant_label=hand.canonical.players[0].label,
    )

    await db.execute(sql_delete(HandRow).where(HandRow.id == hand_id))
    await db.execute(sql_delete(TournamentRow).where(TournamentRow.id == tournament_row))
    await db.flush()

    assert await OpponentsRepo(db).links(opponent_id, owner) == {
        hand.canonical.tournament_id: hand.canonical.players[0].label
    }


async def test_two_opponents_at_once_claim_one_participant_and_the_first_keeps_him(db_factory):
    """Две привязки одной пары, идущие ОДНОВРЕМЕННО, дают одну строку и один ответ.

    Барьер делает одновременность предусловием теста, а не следствием таймингов
    (тот же приём, что `_rendezvous` в `test_bot_handlers.py`): пока обе
    корутины не вошли, ни одна не доходит до своего оператора. Проигравший
    ждёт коммита победителя на самом первичном ключе и возвращает победителя —
    «прочитали — не нашли — вставили» развело бы пару по двум оппонентам.
    """
    from harness.memory.repos import OpponentsRepo, PlayersRepo

    async with db_factory() as session:
        owner = await PlayersRepo(session).get_or_create(tg_user_id=6016)
        first = await OpponentsRepo(session).get_or_create(owner_player_id=owner.id, nick="Vasya")
        second = await OpponentsRepo(session).get_or_create(owner_player_id=owner.id, nick="Petya")
        owner_id = owner.id
        await session.commit()

    barrier = asyncio.Barrier(2)

    async def bind(opponent_id: int) -> int | None:
        async with db_factory() as session:
            await asyncio.wait_for(barrier.wait(), timeout=_RIVAL_TIMEOUT_S)
            holder = await OpponentsRepo(session).link(
                owner_player_id=owner_id,
                opponent_id=opponent_id,
                room_tournament_id="T1",
                participant_label="p1",
            )
            await session.commit()
            return holder

    answers = await asyncio.gather(bind(first), bind(second))

    assert len(set(answers)) == 1, f"одна пара досталась двум никам: {answers}"
    async with db_factory() as session:
        rows = (
            await session.execute(text("select opponent_id from opponent_links"))
        ).scalars().all()
    assert rows == [answers[0]]


async def test_the_last_analysis_is_the_freshest_hand_that_reached_canonical(db):
    """«Участник последнего разбора» — свежайшая рука игрока с чекпоинтом `canonical`.

    Рука, дошедшая только до `raw`, разбором не считается: состава мест за
    столом в ней ещё нет, и называть в ней участника не по чему.
    """
    player_id, session_id = await _player_with_session(db, tg_user_id=6017)
    tournament_id = await TournamentsRepo(db).create(session_id=session_id, source_file="a.txt")
    await _save_hand_in(db, session_id=session_id, tournament_id=tournament_id, hand_no="OLD")
    await _save_hand_in(db, session_id=session_id, tournament_id=tournament_id, hand_no="NEW")
    await HandsRepo(db).save_raw(
        session_id=session_id,
        tournament_id=tournament_id,
        raw=parse_hand(SAMPLE, source_ref="x").model_copy(update={"hand_no": "RAW_ONLY"}),
    )

    last = await HandsRepo(db).last_canonical(player_id)

    assert last is not None and last.hand_no == "NEW"


async def test_the_last_analysis_of_another_player_is_not_mine(db):
    mine, _my_session = await _player_with_session(db, tg_user_id=6018)
    _theirs, their_session = await _player_with_session(db, tg_user_id=6019)
    their_tournament = await TournamentsRepo(db).create(
        session_id=their_session, source_file="b.txt"
    )
    await _save_hand_in(
        db, session_id=their_session, tournament_id=their_tournament, hand_no="THEIRS"
    )

    assert await HandsRepo(db).last_canonical(mine) is None


async def test_the_opponent_rules_are_held_by_the_schema_not_by_the_code(pg):
    """Индекс по `lower(ник)`, первичный ключ пары и каскад по внешнему ключу —
    прокатаны на живом Postgres миграциями 0006 и 0007.
    """
    engine = create_async_engine(pg.get_connection_url(driver="asyncpg"))
    try:
        async with engine.connect() as conn:
            nick_index = (
                await conn.execute(
                    text(
                        "select indexdef from pg_indexes where tablename='opponents' "
                        "and indexname='uq_opponents_owner_player_id_lower_nick'"
                    )
                )
            ).scalar()
            pair_key = (
                await conn.execute(
                    text(
                        "select pg_get_constraintdef(oid) from pg_constraint "
                        "where conrelid='opponent_links'::regclass and contype='p'"
                    )
                )
            ).scalar()
            foreign_keys = (
                await conn.execute(
                    text(
                        "select pg_get_constraintdef(oid) from pg_constraint "
                        "where conrelid='opponent_links'::regclass and contype='f'"
                    )
                )
            ).scalars().all()
    finally:
        await engine.dispose()

    assert nick_index is not None
    assert "UNIQUE" in nick_index and "lower" in nick_index
    assert pair_key == (
        "PRIMARY KEY (owner_player_id, room_tournament_id, participant_label)"
    )
    # Ровно одна ссылка, и та составная: на `tournaments` внешнего ключа нет
    # вовсе — привязка живёт номером турнира комнаты.
    assert foreign_keys == [
        (
            "FOREIGN KEY (opponent_id, owner_player_id) REFERENCES "
            "opponents(id, owner_player_id) ON DELETE CASCADE"
        )
    ]


_PLANTED_NOTES = """
insert into notes (owner_player_id, opponent_nick, color, text, updated_at)
values (:owner, :nick, :color, :text, :moment)
returning id
"""


def test_migration_0007_collapses_notes_of_one_nick_in_two_cases():
    """Правило схлопывания и откат — прокатаны на базе С ДАННЫМИ, а не на пустой.

    В живой базе на день миграции заметок нет вовсе, поэтому спорного случая в
    переливке не будет ни одного. Правило всё равно обязано быть определённым:
    миграция переживёт этот день, а на чужой копии две заметки на один ник в
    разном регистре законны — их разрешал уникальный индекс 0005.

    Свой контейнер, а не сессионный `pg`: тест катает схему вперёд и назад, и
    делать это с базой, на которой стоят остальные тесты, нельзя.

    Ревизии названы номерами, а не `head`: тест про 0007, и следующая миграция
    не имеет права сдвинуть то, что он катает.
    """
    from alembic import command
    from sqlalchemy import create_engine
    from testcontainers.community.postgres import PostgresContainer

    from tests.conftest import alembic_config

    earlier = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    later = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
    with PostgresContainer("postgres:16-alpine") as container:
        dsn = container.get_connection_url(driver="psycopg")
        config = alembic_config(dsn)
        command.upgrade(config, "0006")
        engine = create_engine(dsn)
        try:
            with engine.begin() as conn:
                mine = conn.execute(
                    text("insert into players (tg_user_id) values (9001) returning id")
                ).scalar_one()
                theirs = conn.execute(
                    text("insert into players (tg_user_id) values (9002) returning id")
                ).scalar_one()
                # Оппонент, который у владельца уже назван, — написание берётся
                # отсюда, а не из заметки.
                named = conn.execute(
                    text(
                        "insert into player_aliases (owner_player_id, opponent_nick) "
                        "values (:owner, 'Vasya') returning id"
                    ),
                    {"owner": mine},
                ).scalar_one()
                conn.execute(
                    text(
                        "insert into player_alias_links (owner_player_id, room_tournament_id, "
                        "participant_label, alias_id) values (:owner, 'T1', 'p1', :alias)"
                    ),
                    {"owner": mine, "alias": named},
                )
                old_note = conn.execute(
                    text(_PLANTED_NOTES),
                    {
                        "owner": mine,
                        "nick": "vasya",
                        "color": "red",
                        "text": "фолдит на опен",
                        "moment": earlier,
                    },
                ).scalar_one()
                fresh_note = conn.execute(
                    text(_PLANTED_NOTES),
                    {
                        "owner": mine,
                        "nick": "VASYA",
                        "color": "green",
                        "text": "донкает флоп",
                        "moment": later,
                    },
                ).scalar_one()
                # Ник, которого в `player_aliases` нет вовсе, — оппонент заводится
                # миграцией; и тот же ник у ДРУГОГО владельца, который схлопнуться
                # с чужим не имеет права.
                lone_note = conn.execute(
                    text(_PLANTED_NOTES),
                    {
                        "owner": mine,
                        "nick": "Petya",
                        "color": "none",
                        "text": "лимпит",
                        "moment": earlier,
                    },
                ).scalar_one()
                foreign_note = conn.execute(
                    text(_PLANTED_NOTES),
                    {
                        "owner": theirs,
                        "nick": "vasya",
                        "color": "none",
                        "text": "их заметка",
                        "moment": later,
                    },
                ).scalar_one()

            command.upgrade(config, "0007")

            with engine.connect() as conn:
                merged = conn.execute(
                    text(
                        "select n.id, n.text, n.color, o.opponent_nick "
                        "from notes n join opponents o on o.id = n.opponent_id "
                        "where n.owner_player_id = :owner order by n.id"
                    ),
                    {"owner": mine},
                ).all()
                foreign = conn.execute(
                    text(
                        "select n.id, n.text, o.opponent_nick, o.owner_player_id "
                        "from notes n join opponents o on o.id = n.opponent_id "
                        "where n.owner_player_id = :owner"
                    ),
                    {"owner": theirs},
                ).one()

            # Выжила свежая строка; её id остался, поэтому кнопки уже
            # отправленных сообщений ведут к ней.
            survivor = next(row for row in merged if row.opponent_nick == "Vasya")
            assert survivor.id == fresh_note
            assert old_note not in {row.id for row in merged}
            # Ни один символ не потерян, порядок хронологический.
            assert survivor.text == "фолдит на опен\nдонкает флоп"
            # Цвет — от выжившей строки.
            assert survivor.color == "green"
            # Ник, которого не было в `opponents`, завёл оппонента своим написанием.
            assert {(row.id, row.opponent_nick) for row in merged} == {
                (fresh_note, "Vasya"),
                (lone_note, "Petya"),
            }
            # Чужая заметка на тот же ник — чужой оппонент, схлопывания нет.
            assert foreign.id == foreign_note
            assert (foreign.opponent_nick, foreign.owner_player_id) == ("vasya", theirs)

            command.downgrade(config, "0006")

            with engine.connect() as conn:
                back = conn.execute(
                    text(
                        "select id, opponent_nick, text from notes "
                        "where owner_player_id = :owner order by id"
                    ),
                    {"owner": mine},
                ).all()
                links = conn.execute(
                    text("select alias_id, room_tournament_id from player_alias_links")
                ).all()

            # Откат возвращает ник из связанного оппонента; схлопнутая СТРОКА не
            # воскресает, но её текст остался в выжившей.
            assert [(row.id, row.opponent_nick) for row in back] == [
                (fresh_note, "Vasya"),
                (lone_note, "Petya"),
            ]
            assert back[0].text == "фолдит на опен\nдонкает флоп"
            assert links == [(named, "T1")]

            command.upgrade(config, "0007")

            with engine.connect() as conn:
                again = conn.execute(
                    text(
                        "select n.id, o.opponent_nick from notes n "
                        "join opponents o on o.id = n.opponent_id "
                        "where n.owner_player_id = :owner order by n.id"
                    ),
                    {"owner": mine},
                ).all()
            assert [(row.id, row.opponent_nick) for row in again] == [
                (fresh_note, "Vasya"),
                (lone_note, "Petya"),
            ]
        finally:
            engine.dispose()


def test_the_models_say_the_same_as_the_migrations(pg):
    """Схема, собранная миграциями, совпадает с моделями — включая типы и дефолты.

    Тесты этого расхождения не ловят: базу они получают от `alembic upgrade
    head` (`conftest.pg`), поэтому «модель говорит одно, миграция другое»
    осталось бы незамеченным ровно до продакшена, где схему строит та же
    миграция, а читает её ORM.
    """
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from sqlalchemy import create_engine

    from harness.memory.models import Base

    engine = create_engine(pg.get_connection_url(driver="psycopg"))
    try:
        with engine.connect() as conn:
            context = MigrationContext.configure(
                conn, opts={"compare_type": True, "compare_server_default": True}
            )
            assert compare_metadata(context, Base.metadata) == []
    finally:
        engine.dispose()
