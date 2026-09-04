"""Точка входа бота: конфиг → зависимости → роутер → long polling.

Проводка и ничего больше — весь разум процесса в `handlers.py` (решения) и
`presentation/` (слова). Сам файл не покрыт тестами сознательно: проверять в нём
нечего, кроме того, что aiogram вызывает то, что мы ему передали, а для этого
нужен живой токен, которого у тестов нет и быть не должно (то же ограничение и
та же развязка, что у `TelegramSender` в `worker/main.py`).

`DATA_DIR` читается здесь, а не в `Config.from_env()` — по образцу
`WORKER_CONCURRENCY` (задача 18): `Config` описывает провайдер-слой (спека §7), а
это путь тома конкретного процесса, и делать его обязательной переменной для ВСЕХ
потребителей `Config` (воркер, evals) значило бы требовать от них знать то, что их
не касается. Дефолт `/data` — точка монтирования тома (спека §6: файлы на диске в
volume), а не угаданное продуктовое число.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from aiogram import Bot, Dispatcher

from harness.bot.handlers import BotDeps
from harness.bot.router import build_router
from harness.memory.models import async_session_factory
from harness.platform.config import Config
from harness.platform.logs import configure_logging
from harness.platform.queue import JobsQueue

__all__ = ["main"]

_DEFAULT_DATA_DIR = "/data"


async def main() -> None:
    configure_logging()
    cfg = Config.from_env()
    data_dir = Path(os.environ.get("DATA_DIR", _DEFAULT_DATA_DIR))

    db_factory = async_session_factory(cfg.database_url)
    deps = BotDeps(db_factory=db_factory, queue=JobsQueue(db_factory), data_dir=data_dir)

    bot = Bot(cfg.telegram_token)
    dispatcher = Dispatcher()
    dispatcher.include_router(build_router(deps))
    # Миграции бот не катит — их катает отдельный сервис `migrate` (спека §12,
    # задача 20), поэтому гонка двух процессов за схему исключена по построению.
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
