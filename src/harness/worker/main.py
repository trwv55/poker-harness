"""Точка входа воркера: `claim → run_job` в `WORKER_CONCURRENCY` корутинах, фоновый
`reap()` раз в минуту, единая настройка логирования (спека §2, §8.1).

**Логирование — одно место на весь проект (контроллерский рулинг задачи 18, п.4).**
`configure_logging()` жила здесь, пока процесс был один; с появлением бота (задача
19) она переехала в `platform/logs.py` — общую инфраструктуру обоих процессов — и
реэкспортируется отсюда, чтобы её вызывающие не переучивались. Рассуждение о том,
почему настройка обязана быть единственной (структлог воркера и stdlib-логгер
`platform/limiter.py` должны выйти одним форматтером), — в докстринге нового модуля.

**Останов — по сигналу, и это не украшение (round 5, Item F).** Воркер работает
PID 1 в своём контейнере, а PID 1 не получает сигналов, на которые сам не
подписался: без обработчиков SIGTERM от `docker compose stop`/`up -d` уходил в
пустоту, и каждый деплой заканчивался SIGKILL'ом по истечении grace-периода —
с зависшими до `reap()` задачами, незакрытыми строками `llm_calls` и
пропущенным сбросом эквити-кэша. Обоснование целиком — в докстринге
`_install_stop_handlers`, последовательность останова — в `_run_until_stopped`.

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
import signal
import socket
from asyncio import FIRST_COMPLETED
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress

import httpx
import structlog

from harness.bot.main import data_dir
from harness.memory.models import async_session_factory
from harness.platform.config import Config, EnvVarError, optional_int
from harness.platform.llm import LLM
from harness.platform.logs import configure_logging
from harness.platform.queue import JobsQueue
from harness.presentation import Btn, Msg
from harness.worker.pipeline import Deps, run_job

__all__ = [
    "TelegramDeliveryError",
    "TelegramSender",
    "configure_logging",
    "main",
    "reap_loop",
    "worker_concurrency",
    "worker_loop",
]

_DEFAULT_WORKER_CONCURRENCY = 4

_REAP_INTERVAL_S = 60.0
_POLL_INTERVAL_S = 1.0

# Сколько ждать завершения уже начатых задач после SIGTERM, прежде чем отменить их
# намеренно (round 5, Item F). Не «сколько может идти задача»: её бюджет — до 480с
# (`pipeline._JOB_DEADLINE_S`), и ждать столько на каждом деплое неприемлемо. Это
# окно для тех, кто и так почти закончил; остальные получают ОТМЕНУ, а не
# SIGKILL, и разница существенна — отмена доводит `except asyncio.CancelledError`
# в `platform/llm.py` (строка `llm_calls` закрывается, у неё нет reaper'а) и
# позволяет процессу выйти штатно, отработав `atexit`-сброс эквити-кэша
# (`analysis/preflop.py`). Сама задача при этом не теряется: строка остаётся
# `running`, и `reap()` вернёт её в очередь — ровно тот путь, что описан в
# `docker-compose.yml` у сервиса `worker`.
# Держать МЕНЬШЕ, чем `stop_grace_period` воркера в compose (30s), иначе Docker
# убьёт процесс раньше, чем эта последовательность доиграет.
_SHUTDOWN_GRACE_S = 20.0

# Уникален между ПРОЦЕССАМИ, живущими ОДНОВРЕМЕННО, — и ровно это от него
# требуется: фенсинг сравнивает `worker_id` с `jobs.locked_by` в моменте (см.
# модульный докстринг).
#
# Различает их целиком хостнейм, не PID (round 5, Item K.8: здесь стояло
# «следующий деплой этого же контейнера получит новый PID» — неверно). В образе
# нет ни `ENTRYPOINT`-обёртки, ни `init:` в compose, поэтому воркер стартует
# PID 1 ВСЕГДА, и вторая половина пары постоянна. Что до первой — измерено на
# Docker 29 при подготовке этой правки:
#   * реплики `--scale worker=2` получают РАЗНЫЕ хостнеймы (id контейнера), и
#     дыра, ради которой пара вообще собрана, закрыта;
#   * `up --force-recreate` (обычный деплой) даёт новый контейнер и новый
#     хостнейм — id тоже новый;
#   * а `restart` того же контейнера (и авто-рестарт по `restart:
#     unless-stopped`) переиспользует контейнер, то есть и хостнейм: `_PROCESS_
#     ID` повторяется в точности.
# Последнее не открывает дыры — задачи умершего процесса подбирает `reap()`,
# сбрасывая `locked_by`, — но означает, что «после перезапуска id заведомо
# другой» опорой быть не может. Если понадобится различать ЗАПУСКИ (а не живые
# процессы), сюда придётся добавить что-то ещё, например `uuid4()`.
#
# Явный `hostname:` в compose сломал бы главное свойство: все реплики получили
# бы одно имя. Его там нет и быть не должно.
_PROCESS_ID = f"{socket.gethostname()}-{os.getpid()}"


def _keyboard(buttons: list[list[Btn]]) -> dict[str, object] | None:
    """Раскладка кнопок или `None` — «ключа `reply_markup` в запросе быть не должно».

    `None` здесь означает именно ОТСУТСТВИЕ ключа, а не `null` в JSON. Живая
    приёмка 2026-09-05: значение `"reply_markup": null` Bot API отвергает —
    `400 Bad Request: object expected as reply markup`, — и первое же сообщение
    прогресса («Читаю стол…»), у которого кнопок нет, не доходило до игрока.
    Ту же полезную нагрузку без ключа он принимает (проверено против живого API).
    Разницу держит `_payload`, тесты — `test_sender_omits_reply_markup_when_
    there_are_no_buttons` и `test_sender_sends_the_keyboard_when_there_are_buttons`.
    """
    if not buttons:
        return None
    return {
        "inline_keyboard": [
            [{"text": b.text, "callback_data": b.callback_data} for b in row] for row in buttons
        ]
    }


def _payload(msg: Msg, **fields: object) -> dict[str, object]:
    """Тело запроса к Bot API: обязательные поля плюс `reply_markup` — но только
    когда кнопки действительно есть (см. `_keyboard`).
    """
    body: dict[str, object] = {**fields, "text": msg.text}
    markup = _keyboard(msg.buttons)
    if markup is not None:
        body["reply_markup"] = markup
    return body


class TelegramDeliveryError(RuntimeError):
    """Отказ Bot API, изложенный БЕЗ URL запроса — то есть без токена бота.

    Почему не `response.raise_for_status()` (живая приёмка 2026-09-05). Текст
    `httpx.HTTPStatusError` содержит полный URL запроса, а он у Bot API имеет вид
    `https://api.telegram.org/bot<ТОКЕН>/sendMessage`. Этот текст уходил в лог
    `job_attempt_failed_will_retry` как `error=repr(exc)` и в колонку
    `traces.spans[].error` в базе (`platform/trace.py`) — то есть секрет
    оказывался записан на диск в двух местах сразу. Здесь в сообщение попадают
    только метод, код ответа и `description` из тела ответа; ни `str(exc)`, ни
    `repr(exc)` (а в базу едет именно `repr`) URL не содержат.

    Вторая половина той же истории: тело ответа с `description` не логировалось
    нигде, поэтому причину 400 пришлось угадывать. Теперь она в тексте
    исключения.

    Закреплено `test_sender_error_names_the_status_and_description_without_the_token`
    и `test_sender_keeps_the_token_out_of_a_non_json_error_body`.
    """

    def __init__(self, method: str, status_code: int, description: str) -> None:
        super().__init__(f"{method} → {status_code}: {description}")
        self.method = method
        self.status_code = status_code
        self.description = description


# Кусок `description`, которым Bot API отвечает на редактирование, ничего не
# меняющее. Подстрока, а не полное совпадение: дальше в той же строке Телеграм
# дописывает подробности («specified new message content and reply markup are
# exactly the same as a current content and reply markup of the message»), и
# держать их дословно значило бы завязаться на формулировку целиком.
_MESSAGE_NOT_MODIFIED = "message is not modified"


class TelegramSender:
    """`Sender` поверх Bot API — прямые HTTP-вызовы (`sendMessage`/`editMessageText`),
    без aiogram: тот приходит вместе с ботом (задача 19), а односторонней доставке
    результата из воркера не нужен весь SDK — только эти два эндпоинта.

    В сеть тесты не ходят (то же ограничение, что у `platform/llm.py`, задача 16),
    но сама обёртка тестами покрыта: `httpx.MockTransport` подставляется через
    `client=` и позволяет проверить ФАКТИЧЕСКОЕ тело запроса и разбор ответа —
    ровно те два места, где живая приёмка нашла дефекты. Их набралось четыре
    подряд (`null` в `reply_markup`, токен в тексте исключения, потерянный
    `description`, отказ на неизменившемся сообщении), и все четыре — в коде,
    который до приёмки считался слишком тонким, чтобы его проверять.
    Оркестрация (`run_job`) по-прежнему тестируется против `FakeSender`.
    """

    def __init__(self, token: str, *, client: httpx.AsyncClient | None = None) -> None:
        self._token = token
        self._base_url = f"https://api.telegram.org/bot{token}"
        self._client = client if client is not None else httpx.AsyncClient(timeout=30.0)

    async def _call(self, method: str, payload: dict[str, object]) -> httpx.Response:
        """Один вызов Bot API. Не-2xx превращается в `TelegramDeliveryError` —
        см. её докстринг о том, почему не `raise_for_status()`.
        """
        response = await self._client.post(f"{self._base_url}/{method}", json=payload)
        if not response.is_success:
            raise TelegramDeliveryError(method, response.status_code, self._describe(response))
        return response

    def _describe(self, response: httpx.Response) -> str:
        """Что именно ответил Bot API, в виде, пригодном для лога и для колонки
        `traces.spans[].error`.

        Первый выбор — поле `description` из тела: это и есть внятная причина
        отказа (`Bad Request: object expected as reply markup`), которой до этой
        правки не было нигде. Если тело не Телеграма (например, HTML-страница от
        промежуточного прокси на 502), берётся оно само — но с вырезанным
        токеном: такая страница вполне может процитировать URL запроса, а в нём
        секрет. Замена идёт ДО обрезки, иначе разорванный обрезкой токен уцелел
        бы в остатке.
        """
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            description = body.get("description")
            if isinstance(description, str):
                return description
        return response.text.replace(self._token, "<токен>")[:200] or "<пустое тело ответа>"

    async def send(self, chat_id: int, msg: Msg) -> int:
        response = await self._call("sendMessage", _payload(msg, chat_id=chat_id))
        return int(response.json()["result"]["message_id"])

    async def edit(self, chat_id: int, message_id: int, msg: Msg) -> None:
        """Отредактировать сообщение. «Нечего менять» — это УСПЕХ, а не отказ.

        Живая приёмка 2026-09-06, job 1, попытка 3 из 3: попытка 2 уже перевела
        сообщение прогресса в «Считаю эквити…», ретрай вошёл в ту же станцию и
        отправил тот же текст. Bot API считает это ошибкой —
        `400 Bad Request: message is not modified: ...`, — а для нас цель
        достигнута: сообщение УЖЕ говорит то, что мы хотели сказать. Отказ здесь
        превращал здоровый ретрай в жёсткий `failed`, причём воспроизводимо: так
        кончался бы КАЖДЫЙ повтор любой станции, успевшей отредактировать
        прогресс.

        Отличается по `description`, не по коду ответа: 400 у Bot API — общий код
        на всё «запрос не принят», и глушить его целиком значило бы проглотить и
        реальные отказы. Что остальные 400 по-прежнему поднимают исключение,
        закреплено `test_edit_still_raises_on_any_other_bad_request`; сам случай
        — `test_edit_treats_an_unchanged_message_as_delivered`.
        """
        try:
            await self._call(
                "editMessageText", _payload(msg, chat_id=chat_id, message_id=message_id)
            )
        except TelegramDeliveryError as exc:
            if _MESSAGE_NOT_MODIFIED not in exc.description:
                raise


async def _sleep_or_stop(stop: asyncio.Event | None, seconds: float) -> None:
    """Пауза, которую прерывает сигнал останова. Без `stop` — обычный `sleep`
    (тесты и прямые вызовы циклов). С ним — останов не ждёт конца интервала:
    иначе `reap_loop` со своей минутой съедал бы весь бюджет `stop_grace_period`
    в одиночку.
    """
    if stop is None:
        await asyncio.sleep(seconds)
        return
    with suppress(TimeoutError):
        await asyncio.wait_for(stop.wait(), timeout=seconds)


async def worker_loop(
    deps: Deps,
    worker_id: str,
    *,
    poll_interval_s: float = _POLL_INTERVAL_S,
    stop: asyncio.Event | None = None,
) -> None:
    """`claim → run_job`, пока не попросят остановиться. Пустая очередь — обычное
    состояние (спека §8.1), не ошибка: `claim() is None` просто ждёт следующего
    опроса.

    `stop` (round 5, Item F) — «больше не брать новых задач». Проверяется ПЕРЕД
    `claim()`, а не после: задача, взятая уже во время останова, была бы взята
    только затем, чтобы через секунду быть отменённой, и до `reap()` пролежала бы
    `running` зря. Начатую задачу цикл при этом доводит до конца — принудительную
    отмену, если она не успевает, делает `main()` по истечении своего окна.
    """
    log = structlog.get_logger(__name__)
    while stop is None or not stop.is_set():
        job = await deps.queue.claim(worker_id)
        if job is None:
            await _sleep_or_stop(stop, poll_interval_s)
            continue
        structlog.contextvars.bind_contextvars(worker_id=worker_id)
        try:
            await run_job(job, deps)
        except Exception:
            # `run_job` не выпускает `Exception` наружу — это обеспечено её
            # внешним `except`, а не только обещано в докстринге (round 5, Item
            # E). Но выживание цикла воркера не имеет права держаться на
            # свойстве соседнего модуля: сюда же попадёт и любой будущий путь,
            # который это свойство нарушит. Сеть настоящая, а не декоративная —
            # одна сломанная задача не имеет права остановить воркер целиком.
            log.exception("run_job_raised_unexpectedly", job_id=job.id)
        finally:
            structlog.contextvars.unbind_contextvars("worker_id")


async def reap_loop(
    queue: JobsQueue,
    *,
    interval_s: float = _REAP_INTERVAL_S,
    stop: asyncio.Event | None = None,
) -> None:
    """Раз в минуту (спека §8.1) подбирает задачи, зависшие дольше окна `reap()`."""
    log = structlog.get_logger(__name__)
    while stop is None or not stop.is_set():
        await _sleep_or_stop(stop, interval_s)
        if stop is not None and stop.is_set():
            return
        try:
            reaped = await queue.reap()
            if reaped:
                log.warning("jobs_reaped", count=reaped)
        except Exception:
            log.exception("reap_failed")


def worker_concurrency() -> int:
    """Сколько корутин крутит ОДИН процесс воркера.

    Не часть общего `Config`/`from_env()` (задача 16, `test_config_from_env_
    reads_all_six_vars` дословно фиксирует их число): это тюнинг именно воркера,
    не провайдер-слоя, и делать его обязательной переменной окружения для ВСЕХ
    потребителей `Config` (бот, evals) значило бы требовать от них знать число,
    которое их не касается. Дефолт — не угадывание лимита денег/провайдера (то
    как раз запрещено, спека §7), а число корутин одного процесса.

    Отдельной функцией, а не строкой внутри `main()` (ревью задачи 20, раунд 2):
    саму `main()` тестом не позовёшь — она уходит в вечный цикл, — и потому
    откат ЭТОЙ строки к `os.environ.get(..., "4")` оставлял весь прогон зелёным,
    хотя чинили именно её. Вынесенная функция покрыта тестом напрямую.

    Разбор целого — в `optional_int`, а не здесь: пустое значение считается
    незаданным (`env_file` в Compose отдаёт `WORKER_CONCURRENCY=` пустой
    строкой), а мусор вроде `abc` даёт `InvalidEnvVar` с именем переменной, а не
    голый `ValueError` с трассировкой в цикле рестартов.
    """
    return optional_int("WORKER_CONCURRENCY", _DEFAULT_WORKER_CONCURRENCY)


def _install_stop_handlers() -> asyncio.Event:
    """SIGTERM/SIGINT → `asyncio.Event`, на который смотрят оба цикла.

    **Почему это обязательно, а не удобно (round 5, Item F).** Воркер — PID 1
    своего контейнера (`command:` в compose, никакого `init:`), а у PID 1 в Linux
    НЕТ диспозиции сигнала по умолчанию: ядро доставляет процессу с PID 1 только
    те сигналы, на которые он сам поставил обработчик. Без этой функции `docker
    compose stop`/`up -d` (то есть КАЖДЫЙ деплой) отправлял SIGTERM в пустоту,
    десять секунд ждал и убивал процесс SIGKILL'ом. Цена платилась трижды, и
    каждый раз — за уже написанную защиту: задачи оставались `running` до
    `reap()` через десять минут; `except asyncio.CancelledError` в
    `platform/llm.py`, написанный ровно затем, чтобы строки `llm_calls` не
    зависали в `started` (reaper'а у них нет), не выполнялся никогда;
    `atexit`-сброс дискового эквити-кэша (`analysis/preflop.py`) пропускался, и
    следующий контейнер грел кэш заново. Асимметрия была невидима, потому что
    бот на aiogram ставит обработчики сам и выходит штатно.

    `loop.add_signal_handler` (а не `signal.signal`) — обработчик исполняется в
    цикле событий, между шагами корутин, а не посреди произвольного байткода.
    """
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    return stop


async def _run_until_stopped(tasks: list[asyncio.Task[None]], stop: asyncio.Event) -> None:
    """Крутить циклы до сигнала останова, затем свернуться (round 5, Item F).

    Раньше здесь стояла `asyncio.TaskGroup`, и от неё унаследованы два свойства,
    которые обязаны сохраниться: упавший цикл валит ВЕСЬ процесс (под `restart:
    unless-stopped` это перезапуск с чистого листа, а не воркер, тихо доживающий с
    тремя корутинами из четырёх), и его исключение уходит наружу с трассировкой,
    а не тонет в `gather(..., return_exceptions=True)`. TaskGroup при этом не
    подходит: она ждёт завершения всех задач и не умеет «доиграть отведённое
    время, потом отменить».
    """
    log = structlog.get_logger(__name__)
    stop_waiter = asyncio.create_task(stop.wait())
    done, _pending = await asyncio.wait([*tasks, stop_waiter], return_when=FIRST_COMPLETED)
    stop_waiter.cancel()

    # Первым завершился не сигнал, а сам цикл — значит, он упал (штатно ни один
    # из них не заканчивается до `stop`). Останавливаем процесс целиком.
    died = [t for t in done if t is not stop_waiter and not t.cancelled()]
    failures = [exc for t in died if (exc := t.exception()) is not None]
    if died:
        stop.set()

    log.info("worker_shutdown_started", grace_s=_SHUTDOWN_GRACE_S)
    _finished, still_running = await asyncio.wait(tasks, timeout=_SHUTDOWN_GRACE_S)
    if still_running:
        # Отмена, а не «ждём дальше»: см. `_SHUTDOWN_GRACE_S`. Задача остаётся
        # `running` и вернётся в очередь через `reap()` — это дешевле, чем быть
        # убитым SIGKILL посреди вызова модели.
        log.warning("worker_shutdown_cancelling", tasks=len(still_running))
        for task in still_running:
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    log.info("worker_shutdown_done")

    if failures:
        raise failures[0]


async def main() -> None:
    configure_logging()
    cfg = Config.from_env()
    concurrency = worker_concurrency()

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
            # Тот же том, что у бота (`bot.main.data_dir`): туда ложатся картинки
            # диапазонов станции `explain`, в БД едут пути (спека §6).
            data_dir=data_dir(),
            # Флаг, а не имя модели: станции нужно одно решение — звать вторую
            # ступень каскада зрения или сразу спрашивать игрока. Переменная
            # необязательна (`.env.example`), и незаданная означает «каскада нет».
            vision_fallback=bool(cfg.llm_vision_fallback_model),
        )
        stop = _install_stop_handlers()
        tasks = [
            asyncio.create_task(worker_loop(deps, worker_id=f"{_PROCESS_ID}-{i}", stop=stop))
            for i in range(concurrency)
        ]
        tasks.append(asyncio.create_task(reap_loop(queue, stop=stop)))
        await _run_until_stopped(tasks, stop)


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
