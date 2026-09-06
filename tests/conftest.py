import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from harness.memory.models import Base, async_session_factory

FIXTURES = Path(__file__).parent.parent / "fixtures" / "hh"
FIXTURE_DAILY = FIXTURES / "daily-classic-146.txt"
FIXTURE_PKO = FIXTURES / "pko-bounty-172.txt"

# Реальные hand history — приватные данные игрока и в публичный репозиторий не попадают.
# Тесты, которым они нужны, пропускаются с явной причиной: без файлов регрессионная
# сетка на 318 руках НЕ выполняется, и зелёный прогон без них не означает, что конвейер
# проверен. Порядок сборки данных — в README.
FIXTURES_PRESENT = FIXTURE_DAILY.exists() and FIXTURE_PKO.exists()
requires_fixtures = pytest.mark.skipif(
    not FIXTURES_PRESENT,
    reason="нет приватных HH-фикстур в fixtures/hh/ — гейт на 318 руках не выполнен",
)

SRC = Path(__file__).parent.parent / "src" / "harness"
PROMPTS = SRC / "explanation" / "prompts"
PROMPT_VERDICT = PROMPTS / "verdict.md"
PROMPT_TOURNAMENT = PROMPTS / "tournament.md"
PROMPT_VISION = SRC / "parsers" / "prompts" / "vision.md"

# Промпты изложения закрыты политикой публикации — тем же решением владельца, что
# и vision-промпты, и по той же причине: они копируются за вечер. В публичном
# клоне этих файлов нет, и тесты, которым нужен НАСТОЯЩИЙ текст промпта (а не
# только код вокруг него), пропускаются с явной причиной — ровно та же
# дисциплина, что у HH-фикстур выше: пропуск обязан быть виден строкой
# `skipped` в `-ra`, а не выглядеть успехом.
#
# Пропускается при этом МЕНЬШИНСТВО: проверки верности, отказов и выжимки
# промпта не читают вовсе, а `PromptUnavailable` (громкий отказ при отсутствии
# файла) проверяется без него специально.
# Промпт зрения (задача 22) закрыт тем же решением и тем же путём в хуке, поэтому
# он входит в ТОТ ЖЕ признак, а не заводит второй: маркер отвечает на вопрос «есть
# ли в этом клоне промпты вообще», и клона с половиной промптов не бывает — они
# лежат рядом и приезжают вместе.
PROMPTS_PRESENT = (
    PROMPT_VERDICT.exists() and PROMPT_TOURNAMENT.exists() and PROMPT_VISION.exists()
)
requires_prompts = pytest.mark.skipif(
    not PROMPTS_PRESENT,
    reason="нет промптов моделей (explanation/prompts, parsers/prompts) — закрыты политикой публикации",
)


def _upgrade_head(sync_dsn: str) -> None:
    """Прогоняет `alembic upgrade head` на переданном DSN тем же путём, что и деплой
    (задача 20): `DATABASE_URL` в окружении, `migrations/env.py` читает его сам
    (`get_url()`). Мутирует `os.environ` процесса pytest, не родительского шелла —
    и переопределяется здесь же перед каждым использованием, так что даже если у
    разработчика уже есть свой `DATABASE_URL` (локальный дев-Postgres), тесты на
    него не попадут.
    """
    os.environ["DATABASE_URL"] = sync_dsn
    root = Path(__file__).parent.parent
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "migrations"))
    command.upgrade(cfg, "head")


@pytest.fixture(scope="session")
def pg():
    """Postgres 16 в Docker на весь прогон; схема мигрируется здесь же, один раз.

    Настоящий Postgres нужен нескольким задачам подряд (14-18): `FOR UPDATE SKIP
    LOCKED`, `pg_try_advisory_lock`, jsonb, CHECK-констрейнты — ничего из этого
    честно не эмулируется моком или sqlite. Без Docker фикстура падает явной
    ошибкой при старте контейнера, а не тихим skip: в отличие от приватных
    HH-фикстур (`requires_fixtures` выше) это не приватные данные, а требование
    к окружению разработки, и молчаливый пропуск был бы худшей подсказкой, чем
    трейсбек testcontainers.
    """
    with PostgresContainer("postgres:16-alpine") as container:
        _upgrade_head(container.get_connection_url(driver="psycopg"))
        yield container


@pytest.fixture
async def db(pg):
    """`AsyncSession` внутри одной транзакции, откатываемой в конце теста.

    `join_transaction_mode="create_savepoint"`: `commit()` внутри кода под тестом
    не фиксирует запись по-настоящему, а лишь снимает SAVEPOINT — внешняя
    транзакция откатывается в teardown, тест не оставляет следа для соседних.
    Для тестов репозиториев (задача 14) этого достаточно. Недостаточно для
    конкурентных тестов (задачи 15, 16) — им нужен `db_factory`.
    """
    engine = create_async_engine(pg.get_connection_url(driver="asyncpg"))
    async with engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(
            bind=conn, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        try:
            yield session
        finally:
            await session.close()
            await trans.rollback()
    await engine.dispose()


@pytest.fixture
async def db_factory(pg):
    """`async_sessionmaker`, где запись коммитится по-настоящему (не в `db` выше).

    Контроллерский рулинг задачи 14: `FOR UPDATE SKIP LOCKED` (задача 15) и
    `pg_try_advisory_lock` (задача 16, "advisory-локи — кросс-СОЕДИНЕНИЕ") должны
    быть видны с ДРУГОГО соединения на закоммиченные строки — внутри одной
    откатываемой транзакции (`db`) это невыразимо в принципе, а не просто
    неудобно. Каждый вызов фабрики открывает новое соединение из пула движка —
    то, что нужно обоим тестам. Чистое состояние между тестами — TRUNCATE всех
    таблиц (а не откат: тут в самом деле были коммиты).
    """
    factory = async_session_factory(pg.get_connection_url(driver="asyncpg"))
    try:
        yield factory
    finally:
        async with factory() as session:
            names = ", ".join(f'"{t.name}"' for t in Base.metadata.sorted_tables)
            await session.execute(text(f"TRUNCATE TABLE {names} RESTART IDENTITY CASCADE"))
            await session.commit()
        # sessionmaker.kw — публичный атрибут SQLAlchemy (хранит kwargs конструктора,
        # включая переданный движок); используем его, чтобы не заводить отдельную
        # переменную только ради dispose() в конце теста.
        await factory.kw["bind"].dispose()
