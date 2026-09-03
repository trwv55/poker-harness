"""Память: миграция 0001 и репозитории — на настоящем Postgres (testcontainers).

`db` (одна транзакция, откатываемая после теста) покрывает все репозитории — их
контракт не зависит от того, коммитит вызывающий по-настоящему или нет. `db_factory`
(настоящие коммиты + TRUNCATE между тестами) в задаче 14 использует только его
собственный тест: первый реальный потребитель — `FOR UPDATE SKIP LOCKED` задачи 15 —
но без своего теста здесь контроллерский рулинг задачи 14 остался бы непроверенным
кодом.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import insert, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from harness.contracts import AnalysisResult, RawHand
from harness.engine import enrich
from harness.memory.models import EvalCase, Job
from harness.memory.repos import AnalysesRepo, EvalCasesRepo, HandsRepo, PlayersRepo, SessionsRepo
from harness.normalizer import normalize
from harness.parsers.hh_parser import parse_hand
from tests.test_contracts import make_min_raw
from tests.test_hh_parser import SAMPLE

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
