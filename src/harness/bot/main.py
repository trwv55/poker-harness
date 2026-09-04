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

**`DATA_DIR` читает только этот процесс** (правка ревью задачи 20: раньше здесь
было сказано, что переменную читают оба, — воркер её не читает вовсе, он берёт
готовый абсолютный путь из `jobs.payload`). Отсюда следствие, о котором должен
знать всякий, кто соберётся менять точку монтирования: путь, вычисленный ЗДЕСЬ,
уезжает в БД и живёт там столько же, сколько строка задачи и турнира. Держит
систему вместе не эта переменная, а совпадение точки монтирования тома у бота и
воркера (`docker-compose.yml`); переменная лишь решает, какой путь бот запишет.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from aiogram import Bot, Dispatcher

from harness.bot.handlers import BotDeps
from harness.bot.router import build_router
from harness.memory.models import async_session_factory
from harness.platform.config import Config, EnvVarError, optional_env
from harness.platform.logs import configure_logging
from harness.platform.queue import JobsQueue

__all__ = ["data_dir", "main"]

_DEFAULT_DATA_DIR = "/data"


def data_dir() -> Path:
    """Корень тома с файлами игроков — путь, который бот запишет в БД.

    Отдельной функцией, а не строкой внутри `main()` (ревью задачи 20, раунд 2):
    `main()` уходит в long polling и тестом не вызывается, поэтому откат этой
    строки к `os.environ.get(..., default)` оставлял бы прогон зелёным — притом
    что пустое значение из `env_file` дало бы `Path("")`, то есть `Path(".")`, и
    файлы игроков легли бы в рабочий каталог контейнера вместо тома.
    """
    return Path(optional_env("DATA_DIR", _DEFAULT_DATA_DIR))


async def main() -> None:
    configure_logging()
    cfg = Config.from_env()
    deps_data_dir = data_dir()

    db_factory = async_session_factory(cfg.database_url)
    deps = BotDeps(db_factory=db_factory, queue=JobsQueue(db_factory), data_dir=deps_data_dir)

    bot = Bot(cfg.telegram_token)
    dispatcher = Dispatcher()
    dispatcher.include_router(build_router(deps))
    # Миграции бот не катит — их катает отдельный сервис `migrate` (спека §12,
    # задача 20), поэтому гонка двух процессов за схему исключена по построению.
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except EnvVarError as exc:
        # Контейнер с неполным или испорченным конфигом обязан сказать ОДНОЙ
        # строкой, что именно не так (задача 20): трассировка на восемь кадров в
        # `docker compose logs` про непрочитанную переменную — шум, в котором
        # причина теряется, а `SystemExit` со строкой печатает её в stderr и
        # выходит с кодом 1, без стека. Ловится только `EnvVarError` (не задана
        # ИЛИ задана мусором, раунд 2 ревью): любое ДРУГОЕ исключение при старте
        # — настоящая поломка, и его трассировка нужна целиком.
        raise SystemExit(f"{exc}; заполните .env, образец — .env.example") from exc
