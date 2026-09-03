"""`PgLimiter` — кросс-процессный лимитер вызовов провайдера (спека §7).

Единственный источник истины о том, «можно ли сейчас», — Postgres, а не память
процесса: при `--scale worker=N` лимит на N процессов, посчитанный локальным
семафором, молча умножился бы на N (тот же класс бага, что гонка миграций —
см. спеку §7 и модульный докстринг `queue.py`). Отсюда два разных механизма
под одним контекстменеджером `slot()`, оба обязаны жить в БД:

- **Одновременность** — K "слотов" на advisory-локах Postgres
  (`pg_try_advisory_lock(0x4C4C4D, i)`, `i` из `range(K)`). Advisory-лок в
  Postgres — это лок уровня СЕССИИ (соединения), а не транзакции: держится,
  пока держится соединение, и снимается автоматически при его разрыве. Именно
  поэтому гарантия кросс-процессная — конкурирующие воркеры (и, отдельно,
  конкурирующие корутины одного воркера — `platform/llm.py`, задача 18,
  `WORKER_CONCURRENCY`) видят локи друг друга ЧЕРЕЗ СУБД, а не через
  разделяемую память одного процесса, — и не нужен отдельный reaper на смерть
  воркера, как у `jobs` (задача 15): застрявших слотов после падения соединения
  не бывает физически, снимает сама СУБД.
- **Темп** — не лок, а оконный счётчик: `SELECT count(*) FROM llm_calls WHERE
  started_at > now() - interval '60 seconds'`. Строка `llm_calls` пишется
  вызывающим (`LLM.__call__`, не этот модуль) ДО обращения к модели —
  `status='started'` — и обновляется по завершении; `slot()` считает такие
  in-flight строки внутри окна, иначе всплеск нагрузки, когда много вызовов
  одновременно "в полёте", тихо занижает счётчик именно в момент, когда лимит
  важнее всего (спека §7).

Внутрипроцессный `asyncio.Semaphore` в этом модуле НЕ живёт: он допустим лишь
как дешёвый предфильтр внутри `LLM.__call__`, источником истины не является —
эта обязанность целиком здесь, в Postgres.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# Namespace для pg_try_advisory_lock — первый аргумент без смысла сам по себе,
# нужен только чтобы не столкнуться с локами других подсистем на том же
# кластере Postgres, если они когда-нибудь тоже возьмутся за advisory-локи.
# Значение — из брифа задачи 16 дословно.
_ADVISORY_LOCK_CLASS = 0x4C4C4D

# Джиттер опроса — и окна темпа, и слотов одновременности: без него все
# ожидающие потоки просыпались бы синхронно и толкались за один и тот же
# освободившийся слот/окно (thundering herd).
_JITTER_MIN_S = 0.05
_JITTER_MAX_S = 0.15

_RATE_WINDOW_SQL = text(
    "SELECT count(*) FROM llm_calls WHERE started_at > now() - interval '60 seconds'"
)
_TRY_LOCK_SQL = text("SELECT pg_try_advisory_lock(:cls, :idx)")
_UNLOCK_SQL = text("SELECT pg_advisory_unlock(:cls, :idx)")


def _jitter() -> float:
    return random.uniform(_JITTER_MIN_S, _JITTER_MAX_S)


class PgLimiter:
    """Пул из `max_concurrency` слотов плюс окно `max_per_minute` вызовов в минуту,
    оба — над `session_factory` (задача 14: тот же `async_sessionmaker`, что и у
    `JobsQueue`, `db_factory` в тестах).
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        max_concurrency: int,
        max_per_minute: int,
    ) -> None:
        self._session_factory = session_factory
        self._max_concurrency = max_concurrency
        self._max_per_minute = max_per_minute

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Выделенное соединение на всё время удержания слота — то самое, у
        которого нужно спросить `pg_advisory_unlock` при выходе: advisory-лок
        привязан к СЕССИИ, снять его с другого соединения нельзя (в отличие от
        обычных блокировок строк). Порядок — сначала темп, потом одновременность
        (бриф задачи 16 дословно): нет смысла держать слот в очереди перед
        локом, если вызов и так упрётся в темп на следующем шаге.

        Ни в опросе темпа, ни в опросе слотов НЕТ промежуточных `commit()` —
        не только за ненадобностью (Postgres READ COMMITTED и так даёт свежий
        снапшот на каждый новый оператор внутри одной и той же транзакции, а
        функции advisory-локов вообще нетранзакционны), а потому что он ломает
        саму гарантию: `commit()` завершает текущую транзакцию сессии и
        возвращает её соединение в пул СРАЗУ — не по выходу из `async with`, а
        по факту коммита. Освободившееся соединение пул может тут же отдать
        ДРУГОЙ параллельной `slot()`, и её `pg_try_advisory_lock` на том же
        (физическом!) соединении, где секунду назад закрепился этот же лок,
        увидит его как СВОЙ (advisory-локи в Postgres реентерабельны на уровне
        сессии) и молча "заберёт" чужой слот. Воспроизведено и исправлено по
        живому: с `commit()` после каждого опроса `test_limiter_serializes_
        across_connections` ловил интерлив `["a-in", "b-in", "a-out", "b-out"]`
        стабильно — advisory-лок технически держался (два РАЗНЫХ соединения на
        старте, `pg_backend_pid()` отличался), но соединение, на котором он
        держался, не оставалось закреплено за одной логической `slot()` от
        входа до выхода.
        """
        async with self._session_factory() as session:
            await self._wait_for_rate_window(session)
            slot_index = await self._acquire_advisory_lock(session)
            try:
                yield
            finally:
                await session.execute(
                    _UNLOCK_SQL, {"cls": _ADVISORY_LOCK_CLASS, "idx": slot_index}
                )
                await session.commit()

    async def _wait_for_rate_window(self, session: AsyncSession) -> None:
        while True:
            count = await session.scalar(_RATE_WINDOW_SQL)
            if count is not None and count < self._max_per_minute:
                return
            await asyncio.sleep(_jitter())

    async def _acquire_advisory_lock(self, session: AsyncSession) -> int:
        while True:
            for slot_index in range(self._max_concurrency):
                acquired = await session.scalar(
                    _TRY_LOCK_SQL, {"cls": _ADVISORY_LOCK_CLASS, "idx": slot_index}
                )
                if acquired:
                    return slot_index
            await asyncio.sleep(_jitter())
