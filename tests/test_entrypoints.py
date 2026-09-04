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

import inspect
from pathlib import Path

import pytest

import harness.bot.main as bot_main
import harness.worker.main as worker_main
from harness.platform.config import InvalidEnvVar


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
