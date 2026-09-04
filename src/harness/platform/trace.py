"""Трейс попытки задачи: `open()` заводит строку `traces`, `span(name)` копит шаги
в памяти, `flush()` дописывает их в ту же строку (спека §6: "что вернул каждый
сервис").

Observability — "несущее с первого дня" (ARCHITECTURE.md §7 "Платформа"): тихий баг
вроде "парсер вернул 4% вместо 13.6%" без трейса всплывает уже в проде, а не на
ревью. Поэтому `Trace` пишет свою строку даже если попытка провалилась — `flush()`
в `run_job` (задача 18) вызывается и на успешном, и на неудачном пути.

**Почему строка заводится ДО станций, а не в конце (round 5, Item D).** `llm_calls.
trace_id` — FK NOT NULL на `traces.id`, и `LLM._log_started` вставляет свою строку
ПЕРЕД обращением к модели. Пока строки `traces` нет, любой вызов `deps.llm(...)` из
станции ложится нарушением внешнего ключа — на первом же вызове. Держалось это лишь
тем, что v1-HH модель не зовёт вовсе; задача 21 (станция `explain`) упёрлась бы в
это сразу. Отсюда порядок: `open()` в начале попытки создаёт строку и фиксирует
`trace_id`, `flush()` в конце её ОБНОВЛЯЕТ, а не вставляет вторую. Побочно это
усилило и исходное свойство: строка теперь остаётся даже после того, как процесс
убили посреди попытки, — раньше такая попытка не оставляла следа вообще.

`Trace` — не разделяемый между задачами долгоживущий объект вроде `JobsQueue`/`LLM`
(тех держит один на весь процесс воркера, задача 18, `Deps`): у него нет своего
поля в `Deps` именно потому, что новый экземпляр заводится на каждую ПОПЫТКУ
`run_job` — общий на несколько задач список `_spans` перемешал бы шаги разных
попыток в одной строке `traces`.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from harness.memory.models import Trace as TraceRow

Clock = Callable[[], float]

__all__ = ["Clock", "Trace"]


class Trace:
    """Копит `span`ы в памяти на протяжении одной попытки и дописывает их в свою
    строку `traces` через `flush()`. Саму строку заводит `open()` — в начале
    попытки, до первой станции (см. модульный докстринг: `llm_calls.trace_id`
    ссылается на неё и обязан иметь на что ссылаться).
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        clock: Clock = time.monotonic,
    ) -> None:
        self._session_factory = session_factory
        self._clock = clock
        self._spans: list[dict[str, Any]] = []
        self._trace_id: int | None = None

    @property
    def trace_id(self) -> int:
        """Идентификатор строки `traces` этой попытки — то, что станция передаёт
        в `deps.llm(..., trace_id=...)`. До `open()` его не существует, и молча
        отдать здесь, скажем, `0` значило бы обменять понятную ошибку на
        нарушение FK тремя уровнями глубже.
        """
        if self._trace_id is None:
            raise RuntimeError("Trace.open() ещё не вызывался — строки traces нет")
        return self._trace_id

    async def open(self, job_id: int) -> int:
        """Завести строку `traces` попытки — пустую, до первой станции. Своя
        короткая транзакция, как и у `flush()`: строка обязана быть ВИДНА другим
        соединениям (`LLM._log_started` пишет `llm_calls` из своей сессии, а
        незакоммиченная `traces` для неё не существует).
        """
        async with self._session_factory() as session:
            row = TraceRow(job_id=job_id, spans=[])
            session.add(row)
            await session.commit()
            self._trace_id = row.id
            return row.id

    @asynccontextmanager
    async def span(self, name: str) -> AsyncIterator[None]:
        """Засечь один именованный шаг конвейера; исключение внутри `with` не
        глотается (`raise` после записи ошибки в `_spans`) — трейс диагностирует
        провал, а не подменяет собой обработку ошибок вызывающего.
        """
        started = self._clock()
        error: str | None = None
        try:
            yield
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            entry: dict[str, Any] = {
                "name": name,
                "duration_s": round(self._clock() - started, 3),
            }
            if error is not None:
                entry["error"] = error
            self._spans.append(entry)

    async def flush(self, hand_id: int | None = None) -> int:
        """Дописать накопленные шаги в строку, заведённую `open()` — UPDATE, не
        INSERT (round 5, Item D: вторая строка на ту же попытку означала бы, что
        `llm_calls` этой попытки ссылаются на одну, а шаги лежат в другой). Своя
        короткая транзакция (тот же приём, что у `JobsQueue`/`LLM`, задачи 15/16):
        снаружи есть только `session_factory`, границу коммита некому больше
        держать, кроме этого метода.
        """
        trace_id = self.trace_id
        async with self._session_factory() as session:
            await session.execute(
                update(TraceRow)
                .where(TraceRow.id == trace_id)
                .values(hand_id=hand_id, spans=self._spans)
            )
            await session.commit()
        return trace_id
