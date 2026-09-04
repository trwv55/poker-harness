"""Точка входа Alembic.

Синхронный движок нарочно: `create_async_engine`/asyncpg — рантайм приложения
(`harness.memory.models.async_session_factory`), а миграции гоняются
однократным шагом (`alembic upgrade head`, задача 20) и async-цикл им не нужен
— psycopg (sync) проще и не требует раннера для корутины внутри `env.py`.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from harness.memory.models import Base

config = context.config

if config.config_file_name is not None:
    # `disable_existing_loggers=False` — стандартный `True` из stdlib молча
    # отключает (`.disabled = True`) ЛЮБОЙ логгер, уже созданный к этому
    # моменту в процессе и не перечисленный в `[loggers]` alembic.ini (там
    # только root/sqlalchemy/alembic). Найдено на живом коде (задача 16, fix
    # round 2): `tests/conftest.py`, фикстура `pg`, вызывает `command.upgrade`
    # в этом же процессе — и логгеры `harness.platform.limiter`/`.llm`
    # (создаются `logging.getLogger(__name__)` при импорте модуля, до того как
    # эта фикстура вообще запускается) молча переставали писать куда бы то ни
    # было, без единой ошибки. Не только тестовая проблема: то же самое ждало
    # бы любой процесс, который вызовет `alembic.command.upgrade(...)`
    # программно (не отдельным CLI-шагом) после того, как что-то уже завело
    # свой логгер.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def get_url() -> str:
    """DATABASE_URL из окружения — прод (Docker Compose) и тесты (testcontainers,
    `tests/conftest.py`, фикстура `pg`) подставляют его каждый по-своему. Значение
    из alembic.ini — заведомо нерабочая заглушка, используется только если
    переменной окружения нет вовсе (тогда упадёт явной ошибкой подключения, а не
    тихо возьмёт что-то постороннее).
    """
    return os.environ.get("DATABASE_URL") or config.get_main_option("sqlalchemy.url", "")


def run_migrations_offline() -> None:
    """SQL-скрипт без подключения к БД — здесь не используется, но часть шаблона Alembic."""
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
