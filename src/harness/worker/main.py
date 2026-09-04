"""Точка входа воркера: `claim → run_job` в `WORKER_CONCURRENCY` корутинах, фоновый
`reap()` раз в минуту, единая настройка логирования (спека §2, §8.1).

**Логирование — одно место на весь проект (контроллерский рулинг задачи 18, п.4).**
`configure_logging()` жила здесь, пока процесс был один; с появлением бота (задача
19) она переехала в `platform/logs.py` — общую инфраструктуру обоих процессов — и
реэкспортируется отсюда, чтобы её вызывающие не переучивались. Рассуждение о том,
почему настройка обязана быть единственной (структлог воркера и stdlib-логгер
`platform/limiter.py` должны выйти одним форматтером), — в докстринге нового модуля.

**`worker_id` — уникален между РЕПЛИКАМИ, не только между корутинами одного
процесса.** Фенсинг `JobsQueue.complete()/fail()/await_user()` (задача 15/18, п.2)
сверяет `worker_id` с `jobs.locked_by`: если два контейнера (`docker compose up
--scale worker=2`, задача 20) присвоят своим корутинам одинаковые имена вроде
`"worker-0"`, зомби-корутина ОДНОГО контейнера может пройти фенсинг задачи, которую
`reap()` уже отдал корутине С ТЕМ ЖЕ ИМЕНЕМ в ДРУГОМ контейнере — фенсинг не спасёт
именно потому, что имена совпали не случайно, а по построению. Префикс из хостнейма
и PID процесса (в Docker Compose у каждой реплики свой hostname — обычно id
контейнера) делает `worker_id` уникальным между процессами, не только внутри одного.
"""

from __future__ import annotations

import asyncio
import os
import socket
from concurrent.futures import ProcessPoolExecutor

import httpx
import structlog

from harness.memory.models import async_session_factory
from harness.platform.config import Config, MissingEnvVar, optional_env
from harness.platform.llm import LLM
from harness.platform.logs import configure_logging
from harness.platform.queue import JobsQueue
from harness.presentation import Btn, Msg
from harness.worker.pipeline import Deps, run_job

__all__ = ["TelegramSender", "configure_logging", "main", "reap_loop", "worker_loop"]

_REAP_INTERVAL_S = 60.0
_POLL_INTERVAL_S = 1.0
# Уникален на процесс (см. модульный докстринг) — коротко и достаточно: PID переживает
# только сам процесс, а следующий деплой этого же контейнера получит новый PID.
_PROCESS_ID = f"{socket.gethostname()}-{os.getpid()}"


def _keyboard(buttons: list[list[Btn]]) -> dict[str, object] | None:
    if not buttons:
        return None
    return {
        "inline_keyboard": [
            [{"text": b.text, "callback_data": b.callback_data} for b in row] for row in buttons
        ]
    }


class TelegramSender:
    """`Sender` поверх Bot API — прямые HTTP-вызовы (`sendMessage`/`editMessageText`),
    без aiogram: тот приходит вместе с ботом (задача 19), а односторонней доставке
    результата из воркера не нужен весь SDK — только эти два эндпоинта. Не покрыт
    тестом на реальный Телеграм (в окружении этой задачи нет токена бота — то же
    ограничение, что у `platform/llm.py`, задача 16: тесты не имеют права стучаться
    в сеть); оркестрация (`run_job`) тестируется против `FakeSender`, эта тонкая
    обёртка намеренно вынесена в сторону от протестированной логики.
    """

    def __init__(self, token: str, *, client: httpx.AsyncClient | None = None) -> None:
        self._base_url = f"https://api.telegram.org/bot{token}"
        self._client = client if client is not None else httpx.AsyncClient(timeout=30.0)

    async def send(self, chat_id: int, msg: Msg) -> int:
        response = await self._client.post(
            f"{self._base_url}/sendMessage",
            json={"chat_id": chat_id, "text": msg.text, "reply_markup": _keyboard(msg.buttons)},
        )
        response.raise_for_status()
        return int(response.json()["result"]["message_id"])

    async def edit(self, chat_id: int, message_id: int, msg: Msg) -> None:
        response = await self._client.post(
            f"{self._base_url}/editMessageText",
            json={
                "chat_id": chat_id,
                "message_id": message_id,
                "text": msg.text,
                "reply_markup": _keyboard(msg.buttons),
            },
        )
        response.raise_for_status()


async def worker_loop(
    deps: Deps, worker_id: str, *, poll_interval_s: float = _POLL_INTERVAL_S
) -> None:
    """`claim → run_job`, вечно. Пустая очередь — обычное состояние (спека §8.1),
    не ошибка: `claim() is None` просто ждёт следующего опроса.
    """
    log = structlog.get_logger(__name__)
    while True:
        job = await deps.queue.claim(worker_id)
        if job is None:
            await asyncio.sleep(poll_interval_s)
            continue
        structlog.contextvars.bind_contextvars(worker_id=worker_id)
        try:
            await run_job(job, deps)
        except Exception:
            # `run_job` по контракту не бросает исключений наружу (см. её докстринг)
            # — этот `except` только на случай будущей регрессии контракта: одна
            # сломанная задача не имеет права остановить цикл воркера целиком.
            log.exception("run_job_raised_unexpectedly", job_id=job.id)
        finally:
            structlog.contextvars.unbind_contextvars("worker_id")


async def reap_loop(queue: JobsQueue, *, interval_s: float = _REAP_INTERVAL_S) -> None:
    """Раз в минуту (спека §8.1) подбирает задачи, зависшие дольше окна `reap()`."""
    log = structlog.get_logger(__name__)
    while True:
        await asyncio.sleep(interval_s)
        try:
            reaped = await queue.reap()
            if reaped:
                log.warning("jobs_reaped", count=reaped)
        except Exception:
            log.exception("reap_failed")


async def main() -> None:
    configure_logging()
    cfg = Config.from_env()
    # Не часть общего `Config`/`from_env()` (задача 16, `test_config_from_env_reads_
    # all_six_vars` дословно фиксирует их число): это тюнинг именно воркера, не
    # провайдер-слоя, и делать его обязательной переменной окружения для ВСЕХ
    # потребителей `Config` (бот, evals) значило бы требовать от них знать число,
    # которое их не касается. Дефолт — не угадывание лимита денег/провайдера
    # (то как раз запрещено, спека §7), а число корутин одного процесса.
    # `optional_env`, а не `os.environ.get(..., "4")`: `env_file` в Compose
    # отдаёт строку `WORKER_CONCURRENCY=` как ПУСТОЕ значение, дефолт бы не
    # применился, и `int("")` уронил бы воркер трассировкой в цикле рестартов
    # (ревью задачи 20; обоснование целиком — в докстринге `optional_env`).
    worker_concurrency = int(optional_env("WORKER_CONCURRENCY", "4"))

    db_factory = async_session_factory(cfg.database_url)
    queue = JobsQueue(db_factory)
    llm = LLM(cfg, db_factory)
    sender = TelegramSender(cfg.telegram_token)

    # МЕТОД СТАРТА ПРОЦЕССОВ ЗАДАЁТСЯ ОБРАЗОМ, А НЕ ЭТИМ ФАЙЛОМ. На Linux дефолт
    # `multiprocessing` — `fork` для 3.12/3.13 и `forkserver` начиная с 3.14;
    # `requires-python` у нас `>=3.12`, а `uv.lock` разрешён и под 3.14, поэтому
    # какой из двух в силе, решает строка `FROM python:3.12-slim` в Dockerfile
    # (задача 20, там же обоснование пина). Наблюдаемая разница — не в
    # корректности: код не имеет права ЗАВИСЕТЬ от метода старта (это отдельно
    # разобрано в докстринге `_enqueue_deep_dive_for_missing_hand`,
    # `tests/test_worker_pipeline.py`), и здесь через границу процесса едут
    # только пиклимые аргументы. Разница в ЦЕНЕ: под `fork` подпроцесс наследует
    # память родителя (прогретые модульные кэши, загруженная таблица эквити),
    # под `forkserver` каждый стартует с чистого импорта и греет своё заново.
    # Смена минорной версии образа меняет стоимость холодного пула молча.
    with ProcessPoolExecutor() as process_pool:
        deps = Deps(
            db_factory=db_factory,
            queue=queue,
            sender=sender,
            llm=llm,
            process_pool=process_pool,
        )
        async with asyncio.TaskGroup() as tg:
            for i in range(worker_concurrency):
                tg.create_task(worker_loop(deps, worker_id=f"{_PROCESS_ID}-{i}"))
            tg.create_task(reap_loop(queue))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except MissingEnvVar as exc:
        # Контейнер без `.env` обязан сказать ОДНОЙ строкой, чего именно не
        # хватает (задача 20): трассировка на восемь кадров в `docker compose
        # logs` про непрочитанную переменную — шум, в котором причина теряется, а
        # `SystemExit` со строкой печатает её в stderr и выходит с кодом 1, без
        # стека. Ловится только `MissingEnvVar`: любое ДРУГОЕ исключение при
        # старте — настоящая поломка, и его трассировка нужна целиком.
        raise SystemExit(f"{exc}; заполните .env, образец — .env.example") from exc
