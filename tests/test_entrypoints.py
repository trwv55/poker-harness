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

Кроме конфига здесь живёт то немногое из точек входа, что вообще проверяемо без
сети и без живого токена: останов по сигналу (round 5, Item F) и — с живой
приёмки 2026-09-05 — `TelegramSender`. Последний до неё считался «тонкой
обёрткой, вынесенной в сторону от протестированной логики», и ровно в нём нашлись
три дефекта подряд: `"reply_markup": null` вместо отсутствующего ключа (400 на
первом же сообщении прогресса), токен бота в тексте `HTTPStatusError` (а оттуда —
в лог и в колонку `traces.spans[].error`) и `description` из ответа Телеграма,
не попадавший никуда. `httpx.MockTransport` покрывает всё три, не выходя в сеть.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest

import harness.bot.main as bot_main
import harness.worker.main as worker_main
from harness.platform.config import InvalidEnvVar
from harness.platform.queue import JobsQueue
from harness.presentation import Btn, Msg
from harness.worker.main import TelegramDeliveryError
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


# --- доставка в Телеграм: тело запроса и текст отказа (живая приёмка 2026-09-05) ----

# Заметная строка вместо правдоподобного токена: если она хоть где-то просочится
# в сообщение исключения, это видно глазами в первом же `assert`.
_TEST_TOKEN = "SECRET-TOKEN-123"


def _sender_recording_into(
    requests: list[httpx.Request],
    response: httpx.Response,
) -> tuple[worker_main.TelegramSender, httpx.AsyncClient]:
    """`TelegramSender` на `MockTransport`: в сеть не ходит, но проходит весь свой
    код — сборку тела и разбор ответа. Возвращается и клиент, чтобы тест его
    закрыл (`filterwarnings = error` в pyproject не прощает незакрытых ресурсов).
    """

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return worker_main.TelegramSender(_TEST_TOKEN, client=client), client


_OK = {"ok": True, "result": {"message_id": 4242}}


async def test_sender_omits_reply_markup_when_there_are_no_buttons():
    """Дефект №2 живой приёмки: `_keyboard` отдавал `None` при пустых кнопках, и в
    JSON уходило `"reply_markup": null`. Bot API отвечал `400 Bad Request: object
    expected as reply markup` — то есть «Читаю стол…», первое сообщение прогресса
    и единственное без кнопок, не доходило до игрока НИКОГДА. Проверено против
    живого API: та же нагрузка без ключа принимается.

    Проверяются оба метода: `edit` собирал тело тем же способом и падал бы так же,
    как только редактируемое сообщение осталось бы без кнопок.
    """
    requests: list[httpx.Request] = []
    sender, client = _sender_recording_into(requests, httpx.Response(200, json=_OK))
    try:
        assert await sender.send(777, Msg(text="Читаю стол…")) == 4242
        await sender.edit(777, 4242, Msg(text="Считаю…"))
    finally:
        await client.aclose()

    assert [str(r.url).rsplit("/", 1)[-1] for r in requests] == ["sendMessage", "editMessageText"]
    for request in requests:
        body = json.loads(request.content)
        assert "reply_markup" not in body, "пустые кнопки не имеют права ехать как null"
        assert body["chat_id"] == 777
    assert json.loads(requests[1].content)["message_id"] == 4242


async def test_sender_sends_the_keyboard_when_there_are_buttons():
    """Обратная половина: убрать `null` можно и выбросив `reply_markup` совсем —
    кнопки перестали бы доезжать молча, а без них у игрока нет ни «разобрать», ни
    ответа на вопрос. Здесь пришпилена ровно та форма, которую ждёт Bot API
    (`inline_keyboard` — список РЯДОВ).
    """
    requests: list[httpx.Request] = []
    sender, client = _sender_recording_into(requests, httpx.Response(200, json=_OK))
    msg = Msg(
        text="Разобрать раздачу?",
        buttons=[
            [Btn(text="разобрать", callback_data="deep:TM1")],
            [Btn(text="нет", callback_data="skip")],
        ],
    )
    try:
        await sender.send(777, msg)
        await sender.edit(777, 4242, msg)
    finally:
        await client.aclose()

    for request in requests:
        assert json.loads(request.content)["reply_markup"] == {
            "inline_keyboard": [
                [{"text": "разобрать", "callback_data": "deep:TM1"}],
                [{"text": "нет", "callback_data": "skip"}],
            ]
        }


async def test_sender_error_names_the_status_and_description_without_the_token():
    """Дефект №1 живой приёмки, самый дорогой: `raise_for_status()` даёт
    `HTTPStatusError`, текст которого содержит URL запроса, а URL Bot API — это
    `https://api.telegram.org/bot<ТОКЕН>/sendMessage`. Этот текст записывался в
    лог (`job_attempt_failed_will_retry`, `error=repr(exc)`) и в колонку
    `traces.spans[].error` — секрет на диске в двух местах.

    Проверяется и `str`, и `repr`: в базу через `platform/trace.py` едет именно
    `repr`, и на нём одном тест бы не покраснел, если бы токен уехал в атрибут.
    Заодно дефект №3 — `description` из тела ответа, которого раньше не было
    нигде и из-за отсутствия которого причину 400 пришлось угадывать.
    """
    body = {
        "ok": False,
        "error_code": 400,
        "description": "Bad Request: object expected as reply markup",
    }
    sender, client = _sender_recording_into([], httpx.Response(400, json=body))
    try:
        with pytest.raises(TelegramDeliveryError) as caught:
            await sender.send(777, Msg(text="Читаю стол…"))
    finally:
        await client.aclose()

    exc = caught.value
    assert "400" in str(exc)
    assert "object expected as reply markup" in str(exc)
    assert "sendMessage" in str(exc)
    for rendered in (str(exc), repr(exc)):
        assert _TEST_TOKEN not in rendered
        assert "api.telegram.org" not in rendered


async def test_sender_keeps_the_token_out_of_a_non_json_error_body():
    """Ответ не от Телеграма, а от чего-то по дороге (прокси, балансировщик): тела
    с `description` нет, зато страница может процитировать URL запроса — вместе с
    токеном в пути. Причина отказа при этом нужна, поэтому тело берётся, но токен
    из него вырезается.
    """
    page = f"<html>502 upstream https://api.telegram.org/bot{_TEST_TOKEN}/sendMessage failed</html>"
    sender, client = _sender_recording_into([], httpx.Response(502, text=page))
    try:
        with pytest.raises(TelegramDeliveryError) as caught:
            await sender.send(777, Msg(text="Читаю стол…"))
    finally:
        await client.aclose()

    assert "502" in str(caught.value)
    assert "upstream" in str(caught.value), "причина отказа потеряна целиком"
    for rendered in (str(caught.value), repr(caught.value)):
        assert _TEST_TOKEN not in rendered


_NOT_MODIFIED = (
    "Bad Request: message is not modified: specified new message content and reply markup"
    " are exactly the same as a current content and reply markup of the message"
)


async def test_edit_treats_an_unchanged_message_as_delivered():
    """Четвёртый дефект той же обёртки, найденный уже ПОСЛЕ починки третьего —
    его и обнажил показанный `description` (живая приёмка, job 1, попытка 3/3).

    Попытка 2 успела перевести сообщение прогресса в «Считаю эквити…»; ретрай
    вошёл в ту же станцию и отправил ровно тот же текст. Для Bot API это 400, для
    нас — цель достигнута: сообщение уже говорит то, что мы хотели. Пока это
    считалось отказом, так кончался бы КАЖДЫЙ повтор любой станции, успевшей
    тронуть прогресс, — здоровый ретрай превращался в жёсткий `failed`.
    """
    requests: list[httpx.Request] = []
    body = {"ok": False, "error_code": 400, "description": _NOT_MODIFIED}
    sender, client = _sender_recording_into(requests, httpx.Response(400, json=body))
    try:
        await sender.edit(777, 4242, Msg(text="Считаю эквити…"))
    finally:
        await client.aclose()

    assert len(requests) == 1, "запрос всё равно должен быть отправлен — молчание не лечит"


async def test_edit_still_raises_on_any_other_bad_request():
    """Обратная половина: 400 у Bot API — общий код на всё «запрос не принят», и
    заглушить его целиком значило бы проглотить настоящие отказы вместе с
    безобидным. Различает `description`, а не код ответа.
    """
    body = {"ok": False, "error_code": 400, "description": "Bad Request: chat not found"}
    sender, client = _sender_recording_into([], httpx.Response(400, json=body))
    try:
        with pytest.raises(TelegramDeliveryError, match="chat not found"):
            await sender.edit(777, 4242, Msg(text="Считаю эквити…"))
    finally:
        await client.aclose()


def test_the_expensive_vision_model_is_optional_and_empty_means_absent(monkeypatch):
    """Необязательная переменная: пустая строка равна незаданной (правило `.env.example`).

    `env_file` в Compose передаёт строку `ИМЯ=` как ПУСТОЕ ЗНАЧЕНИЕ, а не как
    отсутствие переменной, — тот же отказ, что уронил воркер в ревью задачи 20,
    и закрыт он тем же `optional_env`.
    """
    from harness.platform.config import Config

    for name, value in {
        "LLM_VISION_MODEL": "anthropic:sonnet",
        "LLM_VERDICT_MODEL": "anthropic:haiku",
        "LLM_MAX_CONCURRENCY": "4",
        "LLM_MAX_PER_MINUTE": "60",
        "DATABASE_URL": "postgresql+asyncpg://x/y",
        "TELEGRAM_TOKEN": "t",
    }.items():
        monkeypatch.setenv(name, value)

    monkeypatch.delenv("LLM_VISION_FALLBACK_MODEL", raising=False)
    assert Config.from_env().llm_vision_fallback_model == ""
    monkeypatch.setenv("LLM_VISION_FALLBACK_MODEL", "")
    assert Config.from_env().llm_vision_fallback_model == ""
    monkeypatch.setenv("LLM_VISION_FALLBACK_MODEL", "anthropic:opus")
    assert Config.from_env().llm_vision_fallback_model == "anthropic:opus"


def test_the_env_example_documents_the_expensive_vision_model_commented_out():
    """Правило файла: всякая необязательная переменная приезжает закомментированной.

    Раскомментировал значит задал; строка `ИМЯ=` в `.env` означала бы пустое
    значение, а не отсутствие переменной.
    """
    from pathlib import Path

    text = (Path(__file__).parent.parent / ".env.example").read_text(encoding="utf-8")
    assert "#LLM_VISION_FALLBACK_MODEL=" in text
    assert "\nLLM_VISION_FALLBACK_MODEL=" not in text


async def test_sender_uploads_the_range_picture_as_a_file(tmp_path):
    """Задача 23: матрица диапазона уходит `sendPhoto` — телом запроса, не ссылкой.

    PNG лежит в томе воркера, публичного URL у него нет и не должно быть: это
    данные игрока. Пришпилено ровно то, что от запроса требуется — метод,
    multipart с файлом, `chat_id` и подпись рядом с ним.
    """
    from harness.presentation import Photo

    picture = tmp_path / "1-0.png"
    picture.write_bytes(b"\x89PNG\r\n\x1a\nbody of the picture")
    requests: list[httpx.Request] = []
    sender, client = _sender_recording_into(requests, httpx.Response(200, json=_OK))
    try:
        assert await sender.send_photo(777, Photo(path=str(picture), caption="колл шова")) == 4242
    finally:
        await client.aclose()

    request = requests[0]
    assert str(request.url).rsplit("/", 1)[-1] == "sendPhoto"
    assert request.headers["content-type"].startswith("multipart/form-data")
    body = request.content
    assert b"\x89PNG" in body
    assert "колл шова".encode() in body
    assert b'name="chat_id"' in body and b"777" in body


async def test_sender_photo_error_keeps_the_token_out_of_the_message(tmp_path):
    """У картинки свой `post`, а не общий `_call`, — и отказ обязан быть так же
    безопасен: ни URL, ни токена в тексте исключения (дефект №1 живой приёмки).
    """
    from harness.presentation import Photo

    picture = tmp_path / "1-0.png"
    picture.write_bytes(b"PNG")
    body = {"ok": False, "error_code": 400, "description": "Bad Request: PHOTO_INVALID_DIMENSIONS"}
    sender, client = _sender_recording_into([], httpx.Response(400, json=body))
    try:
        with pytest.raises(TelegramDeliveryError) as caught:
            await sender.send_photo(777, Photo(path=str(picture), caption="колл шова"))
    finally:
        await client.aclose()

    for rendered in (str(caught.value), repr(caught.value)):
        assert _TEST_TOKEN not in rendered
        assert "api.telegram.org" not in rendered
    assert "PHOTO_INVALID_DIMENSIONS" in str(caught.value)
    assert "sendPhoto" in str(caught.value)
