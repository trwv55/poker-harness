"""Точка входа воркера: `claim → run_job` в `WORKER_CONCURRENCY` корутинах, фоновый
`reap()` раз в минуту, единая настройка логирования (спека §2, §8.1).

**Логирование — одно место на весь процесс (контроллерский рулинг задачи 18, п.4).**
`configure_logging()` — единственный вызов `structlog.configure()`/`logging.
basicConfig`-эквивалент во всём проекте: и структлог-события воркера (`_log =
structlog.get_logger(__name__)` в этом модуле и в `worker/pipeline.py`), и
stdlib-записи `platform/limiter.py` (задача 16 — `logging.getLogger(__name__).
warning(...)` на застревании слота лимитера) обязаны выйти через ОДИН и тот же
форматтер, а не жить каждый в своём формате/потоке — иначе предупреждение
лимитера о забитом слоте окажется невидимым именно тогда, когда оно важнее всего.
Собрано по стандартному рецепту интеграции structlog+stdlib logging (`structlog.
stdlib.ProcessorFormatter`, `foreign_pre_chain`), но не принято на веру: `test_
worker_pipeline.py::test_configure_logging_routes_limiter_warnings` реально вызывает
`logging.getLogger("harness.platform.limiter").warning(...)` и проверяет текст на
выходе — «настроено» и «действительно маршрутизирует» здесь разные утверждения
(`migrations/env.py`, найдено в этой же задаче, звало `fileConfig()` так, что тихо
гасило все логгеры, заведённые раньше в процессе, — этот случай уже исправлен, но
«логирование тихо отключено» остаётся живым классом бага, который стоит проверять,
а не предполагать).

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
import logging
import os
import socket
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import TextIO

import httpx
import structlog

from harness.memory.models import async_session_factory
from harness.platform.config import Config
from harness.platform.llm import LLM
from harness.platform.queue import JobsQueue
from harness.presentation import Btn, Msg
from harness.worker.pipeline import Deps, run_job

__all__ = ["TelegramSender", "configure_logging", "main", "reap_loop", "worker_loop"]

_REAP_INTERVAL_S = 60.0
_POLL_INTERVAL_S = 1.0
# Уникален на процесс (см. модульный докстринг) — коротко и достаточно: PID переживает
# только сам процесс, а следующий деплой этого же контейнера получит новый PID.
_PROCESS_ID = f"{socket.gethostname()}-{os.getpid()}"


def configure_logging(*, stream: TextIO | None = None, level: int = logging.INFO) -> None:
    """Единственное место настройки логирования (см. модульный докстринг). `stream`
    — тестовый крюк (прод — `sys.stderr` по умолчанию): тест подставляет `io.
    StringIO()` и проверяет фактический отрендеренный текст, а не полагается на
    то, что `caplog`/`capsys` не столкнутся с ручной заменой хендлеров рута,
    которую делает эта функция.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        foreign_pre_chain=shared_processors,
    )
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


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
    worker_concurrency = int(os.environ.get("WORKER_CONCURRENCY", "4"))

    db_factory = async_session_factory(cfg.database_url)
    queue = JobsQueue(db_factory)
    llm = LLM(cfg, db_factory)
    sender = TelegramSender(cfg.telegram_token)

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
    asyncio.run(main())
