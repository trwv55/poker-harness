"""Конфигурация провайдер-слоя и инфраструктуры (спека §7): числа и строки, которые
меняются по окружению (модель, лимиты, DSN, токен бота), читаются один раз из
`os.environ` и дальше передаются значением — ни `llm.py`, ни `limiter.py` не читают
`os.environ` сами. Это то же самое разделение, что и у `session_factory` в `queue.py`
(задача 15): код пакета `platform` работает с уже готовыми значениями, а не с их
источником, и его тесты подставляют `Config` руками, не трогая переменные окружения
процесса.

Смена модели — это смена строки конфига, а не правка кода (спека §7: "требование
независимости от вендора — выполняется конфигом"), поэтому `llm_vision_model`/
`llm_verdict_model` не валидируются здесь на конкретный список провайдеров — формат
`"<provider>:<model>"` целиком в ведении PydanticAI (`Agent(model=...)`, 20+
провайдеров), а дублировать его список констант в этом файле означало бы рассинхрон
при каждом обновлении зависимости.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class MissingEnvVar(Exception):
    """Обязательная переменная окружения не задана. Отдельно от `KeyError`
    голого `os.environ[...]` — сообщение называет саму переменную и то, что она
    обязательна, а не индекс в отображении, который ничего не говорит о причине.
    """


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise MissingEnvVar(f"обязательная переменная окружения {name} не задана")
    return value


def _require_int(name: str) -> int:
    return int(_require(name))


def optional_env(name: str, default: str) -> str:
    """Необязательная переменная окружения, где ПУСТАЯ строка равна незаданной.

    Существует не ради симметрии с `_require`, а из-за конкретного отказа
    (ревью задачи 20): `env_file` в Docker Compose ставит строку
    `WORKER_CONCURRENCY=` как ПУСТОЕ ЗНАЧЕНИЕ, а не как отсутствие переменной.
    Голый `os.environ.get(name, default)` в этом случае вернёт `""`, дефолт не
    применится — и `int("")` роняет воркер трассировкой, а `restart:
    unless-stopped` превращает это в цикл рестартов. Путь к отказу — ровно тот,
    который предписывает наш собственный README: `cp .env.example .env`.

    Семантика та же, что у `_require` выше («пустое значит незаданное»), и
    держать её в одном месте важнее, чем сэкономить функцию: следующая
    необязательная переменная получит её бесплатно, а не повторит ту же ошибку.
    """
    return os.environ.get(name) or default


@dataclass(frozen=True, slots=True)
class Config:
    """Снимок конфигурации на момент запуска процесса — не читает окружение сам
    (см. модульный докстринг); `from_env()` — единственная точка, где `os.environ`
    вообще упоминается.
    """

    llm_vision_model: str
    llm_verdict_model: str
    llm_max_concurrency: int
    llm_max_per_minute: int
    database_url: str
    telegram_token: str

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            llm_vision_model=_require("LLM_VISION_MODEL"),
            llm_verdict_model=_require("LLM_VERDICT_MODEL"),
            llm_max_concurrency=_require_int("LLM_MAX_CONCURRENCY"),
            llm_max_per_minute=_require_int("LLM_MAX_PER_MINUTE"),
            database_url=_require("DATABASE_URL"),
            telegram_token=_require("TELEGRAM_TOKEN"),
        )
