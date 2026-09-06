"""Обработчики входа игрока: функции с внедрёнными зависимостями, без aiogram.

Телеграм тут не упомянут ни разу — и это не стилистика. Aiogram (`bot/router.py`)
умеет только скачать файл и отправить `Msg`; всё, что можно сломать — молчаливое
создание сессии, переиспользование активной, `/new`, квота — живёт здесь, в
функциях, которые тест зовёт напрямую с настоящим Postgres и без токена бота.
Ровно поэтому `bot/main.py` и `bot/router.py` не покрыты тестами: там нечему
ломаться, кроме проводки.

**Текст игроку бот не сочиняет.** Каждый `Msg` — вызов конструктора из
`presentation` (правило единого голоса, спека §4). У этого правила есть вторая,
менее очевидная половина: `jobs.error` (задача 18 кладёт туда `str(exc)` —
путь файла на диске, номер руки, текст ошибки SQLAlchemy) игроку не показывается
никогда и ниоткуда. Это ops-данные того же класса, что запрещает публиковать
политика репозитория, и бот эту колонку не читает вовсе.

**Границы транзакций — как у `queue.py`, не как у репозиториев.** Снаружи
приходит `session_factory`, а не открытая сессия, поэтому открыть и закрыть
транзакцию больше некому: каждый блок `async with deps.db_factory()` здесь —
одна транзакция целиком, с явным `commit()`. `enqueue()` открывает свою
собственную (задача 15) и потому зовётся ПОСЛЕ коммита: задача обязана увидеть
уже закоммиченные `sessions`/`tournaments`, иначе воркер, взявший её мгновенно,
прочитает пустоту. Обратная сторона размена названа честно: если процесс умрёт
между коммитом и `enqueue`, останутся файл на диске и строка `tournaments` без
задачи — игрок не получит сводку и пришлёт файл заново, а не увидит
полуразобранный турнир.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from harness.memory.models import Job
from harness.memory.repos import (
    EvalCasesRepo,
    HandsRepo,
    JobsRepo,
    PlayersRepo,
    QuotaCheck,
    QuotaRepo,
    SessionsRepo,
    TournamentsRepo,
)
from harness.parsers.vision_adapter import apply_vision_answer
from harness.platform.queue import JobsQueue
from harness.presentation import (
    Msg,
    ask_gg_nickname_msg,
    gg_nickname_saved_msg,
    hh_accepted_msg,
    hh_duplicate_msg,
    new_session_msg,
    quota_exceeded_msg,
    start_msg,
    unsupported_document_msg,
    vision_answer_not_a_number_msg,
    vision_answer_saved_msg,
    vision_manual_entry_msg,
)

__all__ = [
    "ESCALATION_PREFIX",
    "BotDeps",
    "QuotaCheck",
    "check_quota",
    "handle_deep_dive_callback",
    "handle_document",
    "handle_escalation_callback",
    "handle_new_session",
    "handle_photo",
    "handle_start",
    "handle_text",
]

# Префикс `callback_data` кнопок эскалации — тот же, что собирает
# `presentation.keyboards.escalation_buttons`. Держится здесь строкой затем, что
# разбирает его этот модуль, а не роутер: роутер по контракту не решает ничего.
ESCALATION_PREFIX = "escalate:"

# Значение, которым кнопка «ввести вручную» отличается от кнопки с числом.
MANUAL_ANSWER = "manual"

# PokerCraft отдаёт историю раздач текстом; всё остальное сканировать нечем.
_HH_SUFFIX = ".txt"

# Скрины кладутся рядом с раздачами, тем же правилом имени: содержимое решает,
# как файл называется. Расширение условное — формат определяется по магическим
# байтам при вызове модели (`platform/llm.py`), а не по имени.
_SCREEN_SUFFIX = ".img"


@dataclass(frozen=True, slots=True)
class BotDeps:
    """Зависимости бот-процесса — общие на все обновления, не на одно.

    `data_dir` — корень тома с файлами игроков (спека §6: «файлы — на диске в
    volume, в БД путь и хэш»), а не путь к конкретному файлу: имя внутри него
    считается из содержимого, а не приходит снаружи.
    """

    db_factory: async_sessionmaker[AsyncSession]
    queue: JobsQueue
    data_dir: Path


def _store_hh_file(data_dir: Path, file_bytes: bytes) -> Path:
    """Сохранить файл под именем-хэшем содержимого: `DATA_DIR/hh/{sha256}.txt`.

    Имя от содержимого, а не от присланного игроком: имя из Телеграма — это
    произвольная строка из внешнего мира в пути файловой системы (`../` и всё
    остальное), а два одинаковых файла, присланных дважды, не должны занимать
    место дважды.

    Повторная загрузка перезаписывает файл ТЕМ ЖЕ содержимым — безопасно для
    диска, и только для него (fix round 1: раньше здесь стояло голое «по
    построению безопасно», и это читалось как утверждение обо всей операции).
    Строки БД от совпадения хэша не защищены ничем — за то, чтобы второй
    загрузкой не завелись второй турнир и второй скан, отвечает `handle_document`,
    а не эта функция.
    """
    directory = data_dir / "hh"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{hashlib.sha256(file_bytes).hexdigest()}{_HH_SUFFIX}"
    path.write_bytes(file_bytes)
    return path


def _store_screenshot(data_dir: Path, file_bytes: bytes) -> tuple[Path, str]:
    """Скрин на диск под именем-хэшем содержимого — как и файл раздач.

    Хэш возвращается отдельно: он же едет в `hands.image_hash`, и считать его
    дважды значило бы завести второй источник одного значения.
    """
    digest = hashlib.sha256(file_bytes).hexdigest()
    directory = data_dir / "screens"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}{_SCREEN_SUFFIX}"
    path.write_bytes(file_bytes)
    return path, digest


async def check_quota(deps: BotDeps, player_id: int) -> QuotaCheck:
    """Пускать ли игрока в интерактивный разбор — скользящее окно 24 ч (спека §9).

    Считает `QuotaRepo` (`memory/repos.py`) — та же реализация, из которой воркер
    берёт числа для подписи «разборов X/Y за 24 ч». Отказ принимается ЗДЕСЬ, до
    постановки задачи в очередь: воркер, уже взявший задачу, отказывать не вправе.
    """
    async with deps.db_factory() as db:
        return await QuotaRepo(db).check(player_id)


async def handle_start(deps: BotDeps, tg_user_id: int) -> Msg:
    """`/start`: завести игрока и объяснить один следующий шаг.

    Сессию НЕ открывает. Молчаливое создание привязано к присланному материалу
    (SESSIONS_UX: «скрин без активной сессии»), а не к нажатию «Start»: сессия,
    открытая на приветствии, к вечеру игры отношения не имеет и только
    испортила бы границу первого настоящего вечера.

    Инвайты — задача 23; до неё `players`-запись заводится без ограничений
    (закрытый догфудинг, бриф задачи 19 дословно).
    """
    async with deps.db_factory() as db:
        await PlayersRepo(db).get_or_create(tg_user_id)
        await db.commit()
    return start_msg()


async def handle_document(deps: BotDeps, tg_user_id: int, file_bytes: bytes, filename: str) -> Msg:
    """`.txt` из PokerCraft: файл на диск → сессия (молча) → турнир → `hh_scan`.

    **Молчаливое создание сессии — здесь** (спека §6/§13 шаг 6). Игрок, приславший
    файл, не просил открывать сессию и не должен быть к этому принуждён: сессия
    появляется потому, что результату нужно куда лечь (`jobs.session_id NOT NULL`
    с первой миграции), и подтверждение приёма о ней не упоминает.

    Строка `tournaments` заводится ДО постановки задачи и уходит в `payload`
    вместе с путём: `_run_hh_scan` (задача 18) читает `payload["tournament_id"]`
    и потому не создаёт вторую строку на тот же файл при повторной попытке.

    Квотой HH-скан не ограничен (спека §9: «поощряется щедрее», он дёшев для нас)
    — счётчик считает только интерактивные типы задач, см. `QuotaRepo`.

    **Повторная загрузка того же файла не считается второй раз** (fix round 1).
    Имя файла — sha256 содержимого, поэтому совпадение пути значит совпадение
    байтов: без этой проверки игрок, приславший файл дважды, получал второй
    `tournaments`, вторую копию всех `hands` в одной сессии и две одинаковые
    сводки. Исключение — файл, скан которого ПРОВАЛИЛСЯ: тогда повторная
    загрузка это законная повторная попытка, и она использует уже заведённую
    строку `tournaments` (её чекпоинты пропустят руки, сохранённые до сбоя).

    Что именно гарантировано (fix round 2 — граница названа, а не подразумевается):
    повторная загрузка не заводит второй турнир и не создаёт вторых строк `hands`.
    Чего гарантия НЕ покрывает: две загрузки одного файла, идущие ОДНОВРЕМЕННО,
    могут дать две задачи и, значит, две одинаковые сводки — проверка `jobs`
    читает уже закоммиченное, а `enqueue()` открывает свою транзакцию позже (см.
    модульный докстринг), так что обе успевают увидеть «сканов не было».

    Цена ограничена одним лишним сообщением, и держит эту границу не то, что тут
    стояло раньше (round 5, Item K.7). Прежний текст ссылался на счётчик-чекпоинт
    `_run_hh_scan`, но тот сравнивает число уже сохранённых рук с номером
    текущей ВНУТРИ своей попытки — от двух попыток, идущих одновременно, он не
    защищает вовсе. Защищает партиционный уникальный индекс
    `uq_jobs_player_id_running` (`memory.models.JOBS_RUNNING_UNIQUE_INDEX`,
    миграция `0002`, `on jobs(player_id) WHERE status='running'`): две задачи
    одного игрока физически не могут быть `running` одновременно, поэтому вторая
    ждёт своей очереди и заходит в станцию, когда руки первой уже закоммичены —
    вот тогда чекпоинт их и видит. Турнир у обеих задач при этом общий.

    Лечить это переносом дедупликации на `tournaments` нельзя — тогда сломается
    повторная попытка после смерти процесса между коммитом и `enqueue`, которую
    тот же модульный докстринг бережёт сознательно.
    """
    if not filename.lower().endswith(_HH_SUFFIX):
        # Отказ до всякой записи: ни файла на диске, ни сессии, ни задачи. Ждать
        # 20 секунд ради «не получилось разобрать» из воркера игроку незачем.
        return unsupported_document_msg()

    path = _store_hh_file(deps.data_dir, file_bytes)
    source_file = str(path)

    async with deps.db_factory() as db:
        player = await PlayersRepo(db).get_or_create(tg_user_id)
        session_row = await SessionsRepo(db).active_or_create(player.id)
        tournaments = TournamentsRepo(db)

        last_status = await JobsRepo(db).last_scan_status(session_row.id, source_file)
        if last_status is not None and last_status != "failed":
            await db.commit()  # игрок/сессия могли быть заведены выше — это не откатываем
            return hh_duplicate_msg()

        # Явная сверка с None, а не `... or ...`: `or` считает ложным и целый ноль,
        # а id турнира — число из последовательности, и молчаливая зависимость от
        # того, что она начинается с единицы, здесь ничем не оправдана.
        tournament_id = await tournaments.find_in_session(session_row.id, source_file)
        if tournament_id is None:
            tournament_id = await tournaments.create(
                session_id=session_row.id, source_file=source_file
            )
        await db.commit()
        player_id, session_id = player.id, session_row.id

    await deps.queue.enqueue(
        type="hh_scan",
        player_id=player_id,
        session_id=session_id,
        payload={"source_file": source_file, "tournament_id": tournament_id},
    )
    return hh_accepted_msg()


async def handle_deep_dive_callback(deps: BotDeps, tg_user_id: int, hand_no: str) -> Msg | None:
    """Кнопка `[разобрать]` под строкой сводки — задача `deep_dive` на эту раздачу.

    `None` в успешном случае — не «нечего сказать», а сознательное молчание:
    дальше говорит воркер (прогресс-сообщение, которое он же и редактирует по
    станциям, задача 18). Второй текст от бота стал бы дублем в том самом месте,
    где продукт обещал одно редактируемое сообщение.

    **Задача ставится в ту сессию, где лежит раздача** (рулинг fix round 1), а не
    в активную. Разбор ищет руку как `find_by_hand_no(job.session_id, hand_no)`
    (`worker/pipeline.py`), поэтому нажатие под сводкой вечера, который `/new` уже
    закрыл, при постановке в активную сессию было обречено на честный, но
    бессмысленный отказ — анализ принадлежит сессии, где рука живёт. Контракт
    кнопки при этом не меняется: `callback_data` по-прежнему несёт только
    `hand_no`, разрешение сессии — работа обработчика.

    Рука не нашлась ни в одной сессии игрока — ставим в активную и даём воркеру
    отказать своим единым текстом (`failed_msg`), а не заводим второй путь отказа
    с другой формулировкой на то же самое событие.

    Квота проверяется ДО того, как заводится сессия: игроку, которому отказали,
    не за чем оставлять пустой контейнер вечера.
    """
    async with deps.db_factory() as db:
        player = await PlayersRepo(db).get_or_create(tg_user_id)
        await db.commit()
        player_id = player.id

    # Между этой проверкой и `enqueue` ниже — граница транзакций (у очереди своя,
    # см. модульный докстринг), поэтому два нажатия на самом пределе квоты могут
    # пройти оба: проверка не блокирует строки, по которым считает. Лимит здесь
    # мягкий сознательно — жёсткий потребовал бы держать лок на игроке через всю
    # постановку задачи ради экономии одного разбора в сутки.
    quota = await check_quota(deps, player_id)
    if not quota.allowed:
        return quota_exceeded_msg(quota.hours_to_free)

    async with deps.db_factory() as db:
        session_id = await HandsRepo(db).find_session_by_hand_no(player_id, hand_no)
        if session_id is None:
            session_id = (await SessionsRepo(db).active_or_create(player_id)).id
        await db.commit()

    await deps.queue.enqueue(
        type="deep_dive",
        player_id=player_id,
        session_id=session_id,
        payload={"hand_no": hand_no},
    )
    return None


async def handle_new_session(deps: BotDeps, tg_user_id: int) -> Msg:
    """`/new` (и кнопка «Начать новую» — обработчик один, SESSIONS_UX).

    Закрыть активную и открыть новую — две операции репозитория, а не одна с
    флагом: закрывать бывает нечего (первая сессия игрока), и ответ игроку об
    этом честно молчит. Открывает `active_or_create()` — после закрытия активной
    не осталось, поэтому «создать новую» и «взять активную» здесь один и тот же
    вызов, а не две ветки, которые могли бы разойтись.
    """
    async with deps.db_factory() as db:
        sessions = SessionsRepo(db)
        player = await PlayersRepo(db).get_or_create(tg_user_id)
        previous_closed = await sessions.close_active(player.id)
        opened = await sessions.active_or_create(player.id)
        await db.commit()
        title = opened.title
    return new_session_msg(title, previous_closed=previous_closed)


async def handle_photo(deps: BotDeps, tg_user_id: int, file_bytes: bytes) -> Msg | None:
    """Скрин стола: файл на диск -> сессия (молча) -> задача `screenshot_analyze`.

    **Сначала ник в руме.** Героя на экране определяет код, сопоставляя
    прочитанные ники с ником из профиля; без него разбирать некого, и честнее
    спросить сразу, чем заплатить за чтение и упереться в вопрос после него.

    `None` в успешном случае — то же сознательное молчание, что у кнопки
    «разобрать»: дальше говорит воркер одним редактируемым сообщением прогресса
    (SESSIONS_UX), и второй текст от бота стал бы дублем.
    """
    async with deps.db_factory() as db:
        player = await PlayersRepo(db).get_or_create(tg_user_id)
        await db.commit()
        player_id, nickname = player.id, player.gg_nickname

    if not nickname:
        return ask_gg_nickname_msg()

    quota = await check_quota(deps, player_id)
    if not quota.allowed:
        return quota_exceeded_msg(quota.hours_to_free)

    path, digest = _store_screenshot(deps.data_dir, file_bytes)
    async with deps.db_factory() as db:
        session_row = await SessionsRepo(db).active_or_create(player_id)
        await db.commit()
        session_id = session_row.id

    await deps.queue.enqueue(
        type="screenshot_analyze",
        player_id=player_id,
        session_id=session_id,
        payload={"image_file": str(path), "image_hash": digest},
    )
    return None


def _parse_escalation(data: str) -> tuple[int, str, str] | None:
    """`escalate:{job_id}:{field}:{value}` — номер задачи, поле и выбор игрока.

    Номер задачи в кнопке обязателен: у игрока может ждать ответа больше одной
    задачи разом, и ответ без номера применялся бы к свежайшей — то есть к
    чужой руке (ревью раунда 1, R2).
    """
    parts = data.removeprefix(ESCALATION_PREFIX).split(":")
    if len(parts) != 3 or not parts[0].isdigit():
        return None
    return int(parts[0]), parts[1], parts[2]


def _chosen_option(job: Job, raw_value: str) -> str | None:
    """Вариант ответа по его ИНДЕКСУ в кнопке — сам список лежит в задаче.

    В `callback_data` Телеграма 64 байта, а вариантом бывает ник игрока; поэтому
    кнопка несёт номер, а не текст (`presentation.keyboards.escalation_buttons`).
    """
    options = list((job.payload or {}).get("escalation_options") or [])
    if not raw_value.isdigit():
        return None
    index = int(raw_value)
    return options[index] if 0 <= index < len(options) else None


async def handle_escalation_callback(deps: BotDeps, tg_user_id: int, data: str) -> Msg | None:
    """Ответ игрока на вопрос валидатора — спека §8.3, все четыре шага.

    Ground truth в `eval_cases` пишется ПЕРВЫМ и пишется всегда, даже когда
    подставить ответ в руку нечем: ответ игрока ценен сам по себе — это
    размеченный пример для vision-eval, и эскалация тем самым работает
    разметочной машиной (EVALS.md). Дальше патч `hands.raw`, сброс чекпоинтов
    ниже (одной записью, см. `HandsRepo.replace_raw`) и возврат задачи в очередь.

    Задача берётся ПО НОМЕРУ ИЗ КНОПКИ и сверяется с игроком: чужую задачу
    нажатием не тронуть, а свою — не перепутать с соседней.
    """
    parsed = _parse_escalation(data)
    if parsed is None:
        return None
    job_id, field, raw_value = parsed
    async with deps.db_factory() as db:
        player = await PlayersRepo(db).get_or_create(tg_user_id)
        job = await JobsRepo(db).get_awaiting(job_id, player.id)
        if job is None:
            await db.commit()
            return None
        if raw_value == MANUAL_ANSWER:
            payload = {**dict(job.payload), "manual_entry": field}
            await _remember_payload(db, job.id, payload)
            await db.commit()
            return vision_manual_entry_msg(_question_of(job))
        value = _chosen_option(job, raw_value)
        if value is None:
            await db.commit()
            return None
        await _apply_answer(db, job, field, value)
        await db.commit()

    await deps.queue.resume(job_id)
    return vision_answer_saved_msg()


async def handle_text(deps: BotDeps, tg_user_id: int, text: str) -> Msg | None:
    """Обычное сообщение: либо ник в руме, либо число, введённое вручную.

    Состояние ввода живёт в `jobs.payload`, а не в памяти процесса бота
    (`manual_entry`): точка возврата задачи и так зафиксирована артефактами
    (спека §8.2), и держать половину состояния рядом с ней, а половину в памяти,
    значило бы потерять эту половину при первом же перезапуске.
    """
    answer = text.strip()
    async with deps.db_factory() as db:
        player = await PlayersRepo(db).get_or_create(tg_user_id)
        # Ждущих задач у игрока может быть несколько, а ручного ввода ждёт та, у
        # которой он и был начат: искать «свежайшую ждущую» значило бы подставить
        # число в чужую руку (ревью раунда 1, R2).
        job = await JobsRepo(db).awaiting_manual_entry(player.id)
        field = (job.payload or {}).get("manual_entry") if job is not None else None
        if job is not None and field:
            try:
                float(answer.replace(",", "."))
            except ValueError:
                await db.commit()
                return vision_answer_not_a_number_msg()
            job_id = job.id
            await _apply_answer(db, job, field, answer)
            await db.commit()
            await deps.queue.resume(job_id)
            return vision_answer_saved_msg()

        if not player.gg_nickname and answer:
            await PlayersRepo(db).set_gg_nickname(player.id, answer)
            await db.commit()
            return gg_nickname_saved_msg(answer.strip())
        await db.commit()
    return None


def _question_of(job: Job) -> str:
    """Текст вопроса, на который игрок отвечает вручную, — из payload задачи."""
    payload = job.payload or {}
    field = payload.get("manual_entry") or payload.get("escalation_field") or ""
    return f"Поле «{field}»."


async def _remember_payload(db: AsyncSession, job_id: int, payload: dict) -> None:
    """Записать payload задачи, не трогая её статус: `awaiting_user` сохраняется.

    Мимо `JobsQueue` намеренно — та не даёт менять payload без смены статуса, а
    здесь состояние ввода дописывается к задаче, которая как ждала ответа, так и
    ждёт (тот же приём, что `worker.pipeline._sync_payload`).
    """
    await db.execute(update(Job).where(Job.id == job_id).values(payload=payload))


async def _apply_answer(db: AsyncSession, job: Job, field: str, value: str) -> None:
    """Ground truth в `eval_cases`, затем патч руки и сброс чекпоинтов ниже."""
    payload = dict(job.payload)
    hand_id = payload.get("hand_id")
    if hand_id is not None:
        await EvalCasesRepo(db).add(
            kind="vision_field",
            hand_id=hand_id,
            field=field,
            ground_truth={"field": field, "value": value},
            source="escalation",
        )
        hands = HandsRepo(db)
        record = await hands.get(hand_id)
        patched = apply_vision_answer(
            record.raw, field, value, subject=payload.get("escalation_subject", "")
        )
        if patched is not None:
            await hands.replace_raw(hand_id, patched)
    payload.pop("manual_entry", None)
    payload["last_answer"] = {"field": field, "value": value}
    await _remember_payload(db, job.id, payload)
