"""Трейс попытки задачи: `Trace.span(name)` копит шаги, `flush()` пишет их одной
строкой `traces` (спека §6: "что вернул каждый сервис").

Observability — "несущее с первого дня" (ARCHITECTURE.md §7 "Платформа"): тихий баг
вроде "парсер вернул 4% вместо 13.6%" без трейса всплывает уже в проде, а не на
ревью. Поэтому `Trace` пишет свою строку даже если попытка провалилась — `flush()`
в `run_job` (задача 18) вызывается и на успешном, и на неудачном пути.

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

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from harness.memory.models import Trace as TraceRow

Clock = Callable[[], float]

__all__ = ["Clock", "Trace"]


class Trace:
    """Копит `span`ы в памяти на протяжении одной попытки, пишет их в `traces`
    одной строкой через `flush()` — не раньше, а то незавершённая попытка не
    оставила бы диагностического следа вовсе.
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

    async def flush(self, job_id: int, hand_id: int | None = None) -> int:
        """Одна строка `traces` на попытку — своя короткая транзакция (тот же
        приём, что у `JobsQueue`/`LLM`, задачи 15/16): снаружи есть только
        `session_factory`, границу коммита некому больше держать, кроме этого
        метода.
        """
        async with self._session_factory() as session:
            row = TraceRow(job_id=job_id, hand_id=hand_id, spans=self._spans)
            session.add(row)
            await session.commit()
            return row.id
