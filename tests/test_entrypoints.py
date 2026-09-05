"""Точки входа бота и воркера — ровно в той части, где они читают окружение.

Сами `main()` тестами не покрыты и покрыты не будут: одна уходит в long polling,
другая в вечный цикл `claim`, и обеим нужен живой токен, которого у тестов нет и
быть не должно. Но ЧТЕНИЕ КОНФИГА — не «обвязка»: раунд 1 ревью задачи 20 чинил
именно его (пустое значение из `env_file` роняло воркер в цикл рестартов), а
раунд 2 показал, что гарантия висела на помощнике, а не на точке вызова — откат
строки в `main()` к `os.environ.get(name, default)` оставлял весь прогон зелёным.

Поэтому здесь два этажа проверки:

1. **Поведенческий** — вызываем сами функции чтения (`worker_concurrency()`,
   `data_dir()`) с испорченным окружением. Красный, если тело функции перестанет
   ходить через `optional_env`/`optional_int`.
2. **Структурный** — `main()` обеих точек входа не имеет права читать окружение
   напрямую. Красный, если кто-нибудь вернёт `os.environ` внутрь `main()` в
   обход вынесенной функции — дыра, которую поведенческий тест не видит.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import harness.bot.main as bot_main
import harness.worker.main as worker_main
from harness.platform.config import InvalidEnvVar
from harness.platform.queue import JobsQueue
from harness.worker.pipeline import Deps


def test_worker_concurrency_treats_empty_value_as_unset(monkeypatch):
    """РЕГРЕССИОННЫЙ СТОРОЖ раунда 1, перенесённый на точку вызова.

    `env_file` в Docker Compose отдаёт строку `WORKER_CONCURRENCY=` как пустое
    ЗНАЧЕНИЕ, а не как отсутствие переменной. Пока здесь стоял
    `int(os.environ.get("WORKER_CONCURRENCY", "4"))`, это был `int("")` →
    `ValueError` → трассировка в цикле рестартов при `restart: unless-stopped`.
    """
    monkeypatch.setenv("WORKER_CONCURRENCY", "")
    assert worker_main.worker_concurrency() == 4


def test_worker_concurrency_rejects_garbage_by_naming_the_variable(monkeypatch):
    """Другой триггер того же симптома (ревью задачи 20, раунд 2): значение
    задано, но числом не является. Оператор обязан увидеть имя переменной, а не
    `invalid literal for int()` восемью кадрами ниже.
    """
    monkeypatch.setenv("WORKER_CONCURRENCY", "abc")
    with pytest.raises(InvalidEnvVar, match="WORKER_CONCURRENCY"):
        worker_main.worker_concurrency()


def test_worker_concurrency_reads_the_value_when_it_is_set(monkeypatch):
    """Обратная сторона: заданное значение обязано побеждать дефолт — иначе
    «пустое значит незаданное» тихо превратилось бы в «переменная не читается».
    """
    monkeypatch.setenv("WORKER_CONCURRENCY", "8")
    assert worker_main.worker_concurrency() == 8


def test_data_dir_treats_empty_value_as_unset(monkeypatch):
    """Тот же класс на стороне бота, и цена ошибки здесь выше: `Path("")` — это
    `Path(".")`, то есть файлы игроков легли бы в рабочий каталог контейнера, а
    не в том. Молча: ошибки не будет, будут потерянные при рестарте файлы и
    воркер, который не найдёт их по записанному в БД пути.
    """
    monkeypatch.setenv("DATA_DIR", "")
    assert bot_main.data_dir() == Path("/data")


def test_data_dir_reads_the_value_when_it_is_set(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/mnt/hh")
    assert bot_main.data_dir() == Path("/mnt/hh")


@pytest.mark.parametrize("module", [bot_main, worker_main], ids=["bot", "worker"])
def test_main_does_not_read_environment_directly(module):
    """`main()` не читает окружение сама — только через вынесенные функции.

    Дыра, которую не видят тесты выше: можно оставить `data_dir()`/
    `worker_concurrency()` нетронутыми и при этом вписать `os.environ` обратно
    внутрь `main()`. Тогда поведенческие тесты остались бы зелёными, а прод
    получил бы ровно тот отказ, ради которого всё это писалось. Проверка
    структурная и потому грубая — но именно она делает инвариант («окружение
    читается в одном месте на модуль») исполняемым, а не пожеланием.
    """
    source = inspect.getsource(module.main)
    assert "os.environ" not in source
    assert "os.getenv" not in source


def test_bot_image_does_not_import_calculation_stack():
    """Round 5, Item L: типы скана (`ScanItem`/`ScanSummary`) объявлялись в
    `analysis/scan.py`, и ДВА модуля тянули их через границу процесса —
    `memory/repos.py` (колонка `tournaments.scan_summary`) и
    `presentation/messages.py` (сводка игроку). Импорт ТИПА затягивал в процесс
    бота весь расчётный стек: `pokerkit` и `eval7`. Теперь они живут в
    `harness.contracts`, где им и место — рядом с `AnalysisResult`/`Zone`.

    Проверка обязана идти в ОТДЕЛЬНОМ процессе: сам прогон тестов давно
    импортировал и `analysis`, и `pokerkit`, поэтому `sys.modules` внутри него
    не доказывает ничего.
    """
    code = (
        "import sys, harness.bot.main;"
        "print(sorted({m.split('.')[0] for m in sys.modules} & {'pokerkit', 'eval7'}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]", f"бот загрузил расчётный стек: {result.stdout}"


# --- останов по сигналу (round 5, Item F) -------------------------------------------


class _IdleQueue:
    """Очередь, в которой никогда ничего нет, и счётчик `reap()`. Достаточно:
    циклам воркера от `JobsQueue` нужны ровно эти два метода.
    """

    def __init__(self) -> None:
        self.claims = 0
        self.reaps = 0

    async def claim(self, worker_id: str) -> None:
        self.claims += 1

    async def reap(self) -> int:
        self.reaps += 1
        return 0


def _deps_with(queue: _IdleQueue) -> Deps:
    """`worker_loop` касается только `deps.queue` — остальное в этом тесте не
    существует и существовать не должно (ни БД, ни Телеграма, ни модели).
    """
    return cast(Deps, SimpleNamespace(queue=queue))


async def test_sigterm_stops_both_loops_instead_of_waiting_for_sigkill():
    """Round 5, Item F: воркер — PID 1 своего контейнера, а PID 1 не получает
    сигналов, на которые сам не подписался. Без обработчиков SIGTERM от `docker
    compose up -d` уходил в пустоту, и КАЖДЫЙ деплой заканчивался SIGKILL'ом:
    задачи висели `running` до `reap()` через десять минут, `except
    asyncio.CancelledError` в `platform/llm.py` (написанный ровно затем, чтобы
    строки `llm_calls` не зависали в `started`) не выполнялся никогда, и
    `atexit`-сброс эквити-кэша пропускался.

    Ключевое утверждение здесь — про `reap_loop` с интервалом в минуту: он
    обязан проснуться от сигнала, а не досидеть свою минуту. Иначе один он
    съедал бы весь `stop_grace_period`.
    """
    stop = worker_main._install_stop_handlers()
    loop = asyncio.get_running_loop()
    try:
        assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL, "обработчик не поставлен"
        queue = _IdleQueue()
        tasks = [
            asyncio.create_task(
                worker_main.worker_loop(
                    _deps_with(queue), "w1", poll_interval_s=0.01, stop=stop
                )
            ),
            asyncio.create_task(
                worker_main.reap_loop(cast(JobsQueue, queue), interval_s=60.0, stop=stop)
            ),
        ]
        await asyncio.sleep(0.05)  # дать циклам реально закрутиться
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)

        assert stop.is_set()
        assert queue.claims > 0, "цикл не работал — тест не о том"
        assert queue.reaps == 0, "минута не прошла, reap() звать было незачем"
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)


async def test_shutdown_cancels_work_that_outlives_the_grace_window(monkeypatch):
    """Вторая половина Item F: «дать доработать» — с потолком. Задача может идти
    до 480с (`pipeline._JOB_DEADLINE_S`), ждать столько на каждом деплое нельзя,
    поэтому по истечении окна остаток ОТМЕНЯЕТСЯ — намеренно, а не убивается
    SIGKILL'ом. Разница существенна: отмена доводит обработчики (`llm_calls`
    закрывается) и позволяет процессу выйти штатно, отработав `atexit`.
    """
    monkeypatch.setattr(worker_main, "_SHUTDOWN_GRACE_S", 0.05)
    cancelled = asyncio.Event()

    async def stubborn() -> None:
        # Три секунды, а не тридцать: достаточно, чтобы заведомо пережить окно,
        # и достаточно мало, чтобы фальсификация (убрать отмену) краснела
        # быстро, а не висела полминуты.
        try:
            await asyncio.sleep(3.0)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    stop = asyncio.Event()
    stop.set()
    task = asyncio.create_task(stubborn())
    await worker_main._run_until_stopped([task], stop)

    assert cancelled.is_set()
    assert task.cancelled()


async def test_a_dying_loop_takes_the_whole_process_down_with_its_traceback():
    """Свойство, унаследованное от `asyncio.TaskGroup`, которую Item F заменил:
    упавший цикл валит ВЕСЬ процесс (под `restart: unless-stopped` это
    перезапуск с чистого листа), а его исключение уходит наружу с трассировкой,
    а не тонет в `gather(..., return_exceptions=True)`. Воркер, тихо доживающий
    с тремя корутинами из четырёх, — худший из исходов: снаружи он здоров.
    """
    stop = asyncio.Event()

    async def doomed() -> None:
        raise RuntimeError("цикл воркера умер")

    async def patient() -> None:
        await stop.wait()

    tasks = [asyncio.create_task(doomed()), asyncio.create_task(patient())]
    with pytest.raises(RuntimeError, match="цикл воркера умер"):
        await worker_main._run_until_stopped(tasks, stop)
    assert stop.is_set(), "останов объявлен всем циклам, а не только упавшему"
