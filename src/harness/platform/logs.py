"""Единая настройка логирования на процесс — одна на весь проект.

`configure_logging()` — единственная настройка логирования в коде продукта: и
структлог-события воркера (`worker/main.py`, `worker/pipeline.py`), и
stdlib-записи `platform/limiter.py` (`logging.getLogger(__name__).warning(...)`
на застревании слота лимитера) обязаны выйти через ОДИН форматтер, а не жить
каждый в своём формате и потоке — иначе предупреждение лимитера о забитом слоте
окажется невидимым именно тогда, когда оно важнее всего.

Ровно одно место за её пределами тоже трогает logging — `migrations/env.py` со
своим шаблонным `fileConfig()` (round 5, Item K.9: прежняя формулировка
называла эту функцию «единственным вызовом во всём репозитории» и тем самым
спорила с абзацем про `disable_existing_loggers` ниже в этом же докстринге).
Оно живёт в отдельном одноразовом процессе (`docker compose run --rm migrate`),
не внутри бота или воркера, — но обезврежено там явно и по делу, а не по
случайности: см. ниже.

**Почему в `platform`, а не в `worker/main.py` (задача 19).** Функция родилась
там (задача 18), пока процесс был один. Теперь их два, и бот обязан звать ту же
самую — а `import harness.worker.main` из бота означал бы зависимость бота от
воркера: два процесса, которые в проде даже не обязаны стоять на одной машине,
связались бы ради одной функции настройки. `platform` — общая инфраструктура
обоих, здесь этому и место; `worker/main.py` реэкспортирует имя, чтобы его
вызывающие (в том числе `test_configure_logging_routes_limiter_warnings`,
задача 18) не переучивались.

Довод «иначе бот тянет расчётный стек» здесь по-прежнему НЕ используется, но
уже по другой причине. До round 5 он был просто неверен: `bot/handlers.py`
тянул `memory/repos.py`, а тот импортировал `analysis.scan` ради `ScanSummary`
— и `pokerkit`/`eval7` грузились в процесс бота и так (а считая ещё и
`presentation/messages.py`, импортировавшую оттуда же, таких путей было два, не
один). Item L развязал это: типы скана переехали в `harness.contracts`, и
`import harness.bot.main` больше не загружает ни одного модуля `pokerkit` или
`eval7` (`test_bot_image_does_not_import_calculation_stack`). Довод не
используется потому, что аргумент про зависимость бота от воркера самодостаточен
и не зависит от того, что там с расчётным стеком.

Собрано по стандартному рецепту интеграции structlog+stdlib logging
(`structlog.stdlib.ProcessorFormatter`, `foreign_pre_chain`), но не принято на
веру: тест задачи 18 реально вызывает `logging.getLogger("harness.platform.
limiter").warning(...)` и проверяет текст на выходе — «настроено» и
«действительно маршрутизирует» здесь разные утверждения (`migrations/env.py`,
найдено в той же задаче, звал `fileConfig()` так, что тихо гасил все логгеры,
заведённые раньше в процессе).

**`format_exc_info` — почему он здесь и почему не `dict_tracebacks`
(fix round 2 задачи 19).** Без него в логах не было НИ ОДНОГО трейсбека, и это
не читалось глазами: цепочка выглядела законченной, а рендерились три разных
огрызка — `log.error(..., exc_info=exc)` давал `"exc_info":
"RuntimeError('...')"` (repr без места падения), `log.exception(...)` —
`"exc_info": true` (исключения нет вовсе), stdlib-путь — `["<class
'RuntimeError'>", ..., "<traceback object at 0x...>"]`. Под это попадали все
до единого пути отказа продукта (`worker/main.py`, `worker/pipeline.py`,
`bot/router.py`), то есть в первый же настоящий сбой в проде мы бы узнали, что
он случился, и ничего — где.

Оговорка, без которой абзац выше вводит в заблуждение (round 5, Item K.1):
починка ЗДЕСЬ была необходимой, но не достаточной. Рендерер начал печатать
трейсбек у всех, кто исключение ему передаёт, — а `worker/pipeline.py` в
своих двух ключевых точках (`job_attempt_failed_will_retry`, `job_failed`)
логировал один `repr(exc)` и не передавал ничего, так что путь смерти задачи
оставался без трейсбека ещё раунд. `exc_info` там появился только round 5;
до него называть этот файл среди «отремонтированных» было преждевременно.

Процессор стоит в `shared_processors`, а не в списке самого форматтера: так он
отрабатывает у структлог-событий в момент ВЫЗОВА (внутри `except`-блока, где
`sys.exc_info()` заведомо жив) и у stdlib-записей через `foreign_pre_chain` —
оба пути одним и тем же кодом.

`dict_tracebacks` (структурированный трейсбек, красивее для агрегаторов) не
взят сознательно: `ExceptionDictTransformer` по умолчанию идёт с
`show_locals=True`, а локальные переменные наших станций — это `raw_text` с
содержимым hand history игрока и путь к его файлу. Строчный трейсбек
(`traceback.format_exception`, никаких локалей) — единственный вариант, который
не превращает лог в новое место утечки приватных данных.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

import structlog

__all__ = ["configure_logging"]


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
        structlog.processors.format_exc_info,
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
