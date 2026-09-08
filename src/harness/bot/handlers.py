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

import structlog
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from harness.bot.menus import render_menu_screen, session_summary_screen
from harness.contracts import MAX_NOTE_TEXT_CHARS, NOTE_COLORS
from harness.explanation.hand_replay import hand_replay
from harness.memory.models import Job, Player
from harness.memory.repos import (
    AnalysesRepo,
    EvalCasesRepo,
    HandsRepo,
    InvitesRepo,
    JobsRepo,
    NotesRepo,
    PlayersRepo,
    QuotaCheck,
    QuotaRepo,
    SessionsRepo,
    TournamentsRepo,
)
from harness.parsers.vision_adapter import apply_vision_answer
from harness.platform.llm import MAX_IMAGE_BYTES, MAX_IMAGE_MB
from harness.platform.queue import JobsQueue
from harness.presentation import (
    DETAIL_PREFIX,
    DISAGREE_PREFIX,
    NEW_SESSION_DATA,
    NOTE_ADD_PREFIX,
    NOTE_COLOR_PREFIX,
    NOTE_COLOR_SET_PREFIX,
    NOTE_DELETE_PREFIX,
    NOTE_EDIT_PREFIX,
    RANGES_PREFIX,
    SESSION_PREFIX,
    SET_NICKNAME_DATA,
    Msg,
    analysis_unavailable_msg,
    ask_gg_nickname_msg,
    disagreement_saved_msg,
    gg_nickname_saved_msg,
    gg_nickname_too_long_msg,
    hh_accepted_msg,
    hh_duplicate_msg,
    invite_accepted_msg,
    invite_created_msg,
    invite_required_msg,
    new_session_msg,
    note_color_prompt_msg,
    note_color_saved_msg,
    note_deleted_msg,
    note_gone_msg,
    note_nicks_for_hand,
    note_prompt_msg,
    note_saved_msg,
    note_too_long_msg,
    owner_admitted_msg,
    quota_exceeded_msg,
    ranges_msg,
    replay_msg,
    replay_unavailable_msg,
    screenshot_too_large_msg,
    session_unavailable_msg,
    start_msg,
    unknown_text_msg,
    unsupported_document_msg,
    vision_answer_not_a_number_msg,
    vision_answer_saved_msg,
    vision_manual_entry_msg,
)

__all__ = [
    "ESCALATION_PREFIX",
    "UI_CALLBACK_PREFIXES",
    "BotDeps",
    "QuotaCheck",
    "check_quota",
    "handle_deep_dive_callback",
    "handle_document",
    "handle_escalation_callback",
    "handle_invite_command",
    "handle_new_session",
    "handle_nickname_command",
    "handle_photo",
    "handle_start",
    "handle_text",
    "handle_ui_callback",
]

# Префикс `callback_data` кнопок эскалации — тот же, что собирает
# `presentation.keyboards.escalation_buttons`. Держится здесь строкой затем, что
# разбирает его этот модуль, а не роутер: роутер по контракту не решает ничего.
ESCALATION_PREFIX = "escalate:"

# Лог бота — тот же структлог, что у роутера и воркера: одна настройка на процесс
# (`platform/logs.py`), одна строка на событие.
_log = structlog.get_logger(__name__)

# Значение, которым кнопка «ввести вручную» отличается от кнопки с числом.
MANUAL_ANSWER = "manual"

# Что означает следующее текстовое сообщение игрока (`players.pending_input`).
# Ввод открывается только явным действием — командой или кнопкой; молча в это
# состояние продукт больше не входит (решение владельца 2026-09-07).
_INPUT_NICKNAME = "gg_nickname"
_INPUT_NOTE = "note"

# Ник в руме не длиннее колонки `players.gg_nickname` (`String(64)`). Число здесь
# не второе определение предела, а ссылка на него: длиннее БД просто не примет.
_MAX_NICKNAME = 64

# Префиксы `callback_data`, которые разбирает `handle_ui_callback`. Роутер
# фильтрует нажатия ровно по этому кортежу, поэтому кнопка с любым другим
# префиксом по-прежнему доходит до общего ответа «ещё не работает» — а не
# проваливается в тишину (round 5, Item G).
UI_CALLBACK_PREFIXES: tuple[str, ...] = (
    SESSION_PREFIX,
    NEW_SESSION_DATA,
    NOTE_COLOR_SET_PREFIX,
    NOTE_COLOR_PREFIX,
    NOTE_EDIT_PREFIX,
    NOTE_DELETE_PREFIX,
    NOTE_ADD_PREFIX,
    SET_NICKNAME_DATA,
    RANGES_PREFIX,
    DETAIL_PREFIX,
    DISAGREE_PREFIX,
)

# PokerCraft отдаёт историю раздач текстом; всё остальное сканировать нечем.
_HH_SUFFIX = ".txt"

# Картинка, присланная документом («отправить без сжатия»), — тот же вход зрения,
# что и фотография. Набор УЖЕ, чем понимает `platform/llm.py` (там ещё и GIF):
# сюда попадает только то, чем бывает сохранённый экран.
_IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

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

    `owner_tg_user_id` — id владельца в Телеграме, если сервер его назвал
    (`OWNER_TG_USER_ID`, `bot/main.py`): единственный вход на ЧИСТУЮ базу, где
    инвайт выпустить некому. `None` — обычное состояние и безопасное умолчание:
    без кода не входит никто (`test_without_the_owner_variable_the_door_stays_shut`).
    """

    db_factory: async_sessionmaker[AsyncSession]
    queue: JobsQueue
    data_dir: Path
    owner_tg_user_id: int | None = None


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


async def handle_start(deps: BotDeps, tg_user_id: int, payload: str = "") -> Msg:
    """`/start [код]`: впустить по инвайту и объяснить один следующий шаг.

    Сессию НЕ открывает. Молчаливое создание привязано к присланному материалу
    (SESSIONS_UX: «скрин без активной сессии»), а не к нажатию «Start»: сессия,
    открытая на приветствии, к вечеру игры отношения не имеет и только
    испортила бы границу первого настоящего вечера.

    **Вход по инвайту (задача 23).** Уже заведённый игрок здоровается как
    прежде — код у него не спрашивают. Незнакомцу нужен код: `players`-строка
    заводится и код гасится в ОДНОЙ транзакции, поэтому непогашенный код не
    оставляет за собой игрока, которого никто не приглашал
    (`test_a_start_with_a_bad_code_leaves_no_player_behind`).

    Код приезжает из deep-link (`t.me/bot?start=КОД` — Телеграм отдаёт его
    аргументом команды), поэтому это внешние данные: гасит их `InvitesRepo`
    одним `UPDATE ... WHERE used_by IS NULL`, а не проверка «прочитали и
    записали».

    **Первый вход владельца на чистую базу.** Инвайт выпускает `/invite`, а он
    отвечает только игроку с `is_dev`; на пустой базе такого игрока нет, и после
    деплоя в продукт не может войти никто — включая того, кто его развернул.
    Круг разрывает id владельца из окружения сервера
    (`deps.owner_tg_user_id`): `/start` от него заводит ПЕРВОГО игрока с
    `is_dev` и без кода. Три свойства, которыми это не является постоянной
    дверью: срабатывает только пока `players` пуста (условие проверяет тем же
    оператором, что и вставляет, — `PlayersRepo.bootstrap_owner`), гостю код
    по-прежнему нужен (`test_after_the_owner_a_second_person_still_needs_an_
    invite`), и каждое срабатывание видно в логе сервера
    (`test_the_owner_bootstrap_leaves_a_line_in_the_log`).
    """
    async with deps.db_factory() as db:
        players = PlayersRepo(db)
        player = await players.find(tg_user_id)
        if player is not None:
            await db.commit()
            return start_msg()
        if deps.owner_tg_user_id is not None and tg_user_id == deps.owner_tg_user_id:
            owner = await players.bootstrap_owner(tg_user_id)
            if owner is not None:
                await db.commit()
                # После коммита: запись в логе означает состоявшийся вход, а не
                # намерение. Проигравший гонку сюда не попадает вовсе — у него
                # `bootstrap_owner` вернул `None`, и он уходит общим путём ниже.
                _log.info("owner_bootstrapped", player_id=owner.id, tg_user_id=tg_user_id)
                return owner_admitted_msg()
        code = payload.strip()
        if not code:
            await db.commit()
            return invite_required_msg()
        created = await players.get_or_create(tg_user_id)
        if not await InvitesRepo(db).redeem(code, created.id):
            await db.rollback()
            return invite_required_msg()
        await db.commit()
    return invite_accepted_msg()


async def _known_player(db: AsyncSession, tg_user_id: int) -> Player | None:
    """Игрок, уже впущенный в продукт, — либо `None`, и тогда вход закрыт.

    Проверяется на КАЖДОМ входе, а не только в `/start`: иначе незнакомец,
    приславший файл первым сообщением, заводил бы себе строку `players` в обход
    инвайта (`test_a_stranger_without_an_invite_is_refused_on_every_entry`).
    """
    return await PlayersRepo(db).find(tg_user_id)


async def handle_invite_command(deps: BotDeps, tg_user_id: int) -> Msg | None:
    """`/invite`: выпустить код. Только владельцу (`players.is_dev`).

    Чужому игроку команда не отвечает НИЧЕГО — не «вам нельзя»: сообщение о
    существовании команды и есть половина приглашения ею воспользоваться.
    Решение о раздаче кодов — владельца; здесь только механика (бриф задачи 23).
    """
    async with deps.db_factory() as db:
        player = await _known_player(db, tg_user_id)
        if player is None or not player.is_dev:
            await db.commit()
            return None
        code = await InvitesRepo(db).mint(player.id)
        await db.commit()
    return invite_created_msg(code)


async def handle_nickname_command(deps: BotDeps, tg_user_id: int) -> Msg | None:
    """`/nick`: открыть ввод ника в руме — та же FSM, что у кнопки в «Настройках».

    Явная команда вместо «первого текстового сообщения» (решение владельца
    2026-09-07): до этой задачи ником молча становилась любая случайная реплика.
    """
    async with deps.db_factory() as db:
        player = await _known_player(db, tg_user_id)
        if player is None:
            await db.commit()
            return invite_required_msg()
        await PlayersRepo(db).set_pending_input(player.id, {"kind": _INPUT_NICKNAME})
        await db.commit()
    return ask_gg_nickname_msg()


async def handle_document(
    deps: BotDeps,
    tg_user_id: int,
    file_bytes: bytes,
    filename: str,
    mime_type: str | None = None,
) -> Msg | None:
    """`.txt` из PokerCraft: файл на диск → сессия (молча) → турнир → `hh_scan`.

    **Документом приходит и картинка.** «Отправить без сжатия» — это документ, а
    не фото, и именно так продукт сам просит прислать скрин, когда масти не
    прочитались (`send_as_file_msg`). Такой документ уходит в тот же путь зрения,
    что и фотография (`_accept_screenshot`), и потому возвращает `Msg | None`:
    успешная постановка задачи молчит.

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
        if _is_image_document(mime_type, filename):
            return await _accept_screenshot(deps, tg_user_id, file_bytes)
        # Отказ до всякой записи: ни файла на диске, ни сессии, ни задачи. Ждать
        # 20 секунд ради «не получилось разобрать» из воркера игроку незачем.
        return unsupported_document_msg()

    async with deps.db_factory() as db:
        player = await _known_player(db, tg_user_id)
        if player is None:
            await db.commit()
            return invite_required_msg()
        # Байты ложатся на диск ТОЛЬКО за проверкой инвайта. Телеграм отдаёт
        # документы до 20 МБ, и запись до неё складывала в общий том файлы
        # любого, кто нашёл бота
        # (`test_a_stranger_without_an_invite_leaves_nothing_in_the_volume`).
        path = _store_hh_file(deps.data_dir, file_bytes)
        source_file = str(path)
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
        player = await _known_player(db, tg_user_id)
        await db.commit()
        if player is None:
            return invite_required_msg()
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
        player = await _known_player(db, tg_user_id)
        if player is None:
            await db.commit()
            return invite_required_msg()
        previous_closed = await sessions.close_active(player.id)
        opened = await sessions.active_or_create(player.id)
        await db.commit()
        title = opened.title
    return new_session_msg(title, previous_closed=previous_closed)


def _is_image_document(mime_type: str | None, filename: str) -> bool:
    """Документ — это картинка? Тип из Телеграма, имя файла — запасной признак.

    Оба признака приходят снаружи и оба бывают неверны: `mime_type` Телеграм
    ставит не всегда, а `application/octet-stream` он ставит охотно. Поэтому
    признаки складываются, а не заменяют друг друга: совпал любой — файл идёт в
    зрение. Что внутри на самом деле, решают магические байты у самой модели
    (`platform/llm.py`), а не эта функция.
    """
    if mime_type and mime_type.split(";")[0].strip().lower() in _IMAGE_MIME_TYPES:
        return True
    return filename.lower().endswith(_IMAGE_SUFFIXES)


async def _accept_screenshot(deps: BotDeps, tg_user_id: int, file_bytes: bytes) -> Msg | None:
    """Общий путь зрения: скрин на диск -> сессия (молча) -> `screenshot_analyze`.

    Один на оба входа — фотографию и документ-картинку. Дублировать эту
    последовательность на второй вход значило бы завести второе место, где
    порядок проверок может разойтись, а порядок здесь и есть гарантия: ни байта
    на диск раньше `_known_player`
    (`test_a_stranger_sending_a_picture_as_a_document_leaves_nothing_in_the_volume`).

    **Сначала ник в руме.** Героя на экране определяет код, сопоставляя
    прочитанные ники с ником из профиля; без него разбирать некого, и честнее
    спросить сразу, чем заплатить за чтение и упереться в вопрос после него.

    **Предел размера — до очереди.** Картинка тяжелее `MAX_IMAGE_BYTES` до модели
    не доедет (`platform/llm.py`), и узнать об этом игрок должен сразу, а не через
    двадцать секунд отказом воркера.

    `None` в успешном случае — то же сознательное молчание, что у кнопки
    «разобрать»: дальше говорит воркер одним редактируемым сообщением прогресса
    (SESSIONS_UX), и второй текст от бота стал бы дублем.
    """
    async with deps.db_factory() as db:
        player = await _known_player(db, tg_user_id)
        if player is None:
            await db.commit()
            return invite_required_msg()
        if len(file_bytes) > MAX_IMAGE_BYTES:
            # После проверки игрока, а не до неё: посторонний не должен узнавать
            # из отказа ничего сверх того, что узнавал раньше.
            await db.commit()
            return screenshot_too_large_msg(MAX_IMAGE_MB)
        player_id, nickname = player.id, player.gg_nickname
        if not nickname:
            # Вопрос задан — значит следующий текст и есть ответ на него. Это не
            # возврат к «первому тексту = ник» (тот срабатывал на любую реплику
            # без всякого вопроса): состояние ставится ровно потому, что бот
            # только что спросил, и снимается любым нажатием меню.
            await PlayersRepo(db).set_pending_input(player_id, {"kind": _INPUT_NICKNAME})
            await db.commit()
            return ask_gg_nickname_msg()
        await db.commit()

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


async def handle_photo(deps: BotDeps, tg_user_id: int, file_bytes: bytes) -> Msg | None:
    """Фотография в чат — главный вход продукта; вся работа в `_accept_screenshot`."""
    return await _accept_screenshot(deps, tg_user_id, file_bytes)


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
        player = await _known_player(db, tg_user_id)
        if player is None:
            await db.commit()
            return None
        job = await JobsRepo(db).get_awaiting(job_id, player.id)
        if job is None:
            await db.commit()
            return None
        if raw_value == MANUAL_ANSWER:
            payload = {**dict(job.payload), "manual_entry": field}
            await _remember_payload(db, job.id, payload)
            # Начатый ввод заметки или ника гасится здесь: `handle_text`
            # проверяет `pending_input` раньше эскалации, и без этой строки
            # число, набранное в ответ на только что заданный вопрос, ушло бы
            # в заметку
            # (`test_the_manual_entry_button_closes_an_input_that_was_started`).
            await PlayersRepo(db).set_pending_input(player.id, None)
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
    """Обычное сообщение: кнопка меню, начатый ввод, число по эскалации — в этом порядке.

    **Порядок не произволен.** Нижнее меню Телеграма присылает нажатие ОБЫЧНЫМ
    текстом, поэтому подпись кнопки проверяется первой: игрок, нажавший «Мои
    лики» посреди ввода заметки, хочет экран, а не заметку с таким текстом.
    Нажатие меню поэтому же и снимает начатый ввод — иначе следующая же реплика
    попала бы в него неожиданно для игрока.

    **Ввод из `pending_input` — раньше эскалации.** Эскалация ждёт в статусе
    `awaiting_user` без срока (SESSIONS_UX), а `pending_input` ставится ровно
    тем нажатием или командой, на которые игрок отвечает прямо сейчас: обратный
    порядок отдавал бы висящей задаче и текст заметки, и ник
    (`test_a_note_typed_while_an_escalation_waits_lands_in_the_note`).
    Встречный перехват закрыт в другом месте: открывая ручной ввод по
    эскалации, `handle_escalation_callback` гасит `pending_input`.

    Состояние ввода живёт в БД, а не в памяти процесса бота: число по эскалации
    — в `jobs.payload` (спека §8.3, задача 22), ник и текст заметки — в
    `players.pending_input` (задача 23). Перезапуск бота не теряет ни того, ни
    другого.

    **Ник больше не берётся из первого текста** (решение владельца 2026-09-07):
    ввод открывается только `/nick` или кнопкой в «Настройках», и текст без
    начатого ввода получает честное «не понял», а не тихо оказывается в профиле.
    """
    answer = text.strip()
    async with deps.db_factory() as db:
        player = await _known_player(db, tg_user_id)
        if player is None:
            await db.commit()
            return invite_required_msg()

        screen = await render_menu_screen(db, player, answer)
        if screen is not None:
            await PlayersRepo(db).set_pending_input(player.id, None)
            await db.commit()
            return screen

        pending = dict(player.pending_input or {})
        if pending:
            reply = await _apply_pending_input(db, player, pending, answer)
            await db.commit()
            return reply

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

        reply = await _apply_pending_input(db, player, pending, answer)
        await db.commit()
    return reply


async def _apply_pending_input(
    db: AsyncSession, player: Player, pending: dict, answer: str
) -> Msg:
    """Текст в начатый ввод: ник в руме или заметка на оппонента.

    Пустое сообщение ввод не закрывает и не сбрасывает: игрок, приславший одни
    пробелы, получает ту же просьбу, с которой ввод и начался
    (`test_a_blank_message_repeats_the_request_that_was_made`).
    """
    kind = pending.get("kind")
    if not answer:
        if kind == _INPUT_NICKNAME:
            return ask_gg_nickname_msg()
        if kind == _INPUT_NOTE:
            nick = str(pending.get("nick", ""))
            return note_prompt_msg(nick, await NotesRepo(db).find_by_nick(player.id, nick))
        return unknown_text_msg()
    if kind == _INPUT_NICKNAME:
        if len(answer) > _MAX_NICKNAME:
            return gg_nickname_too_long_msg(_MAX_NICKNAME)
        await PlayersRepo(db).set_gg_nickname(player.id, answer)
        await PlayersRepo(db).set_pending_input(player.id, None)
        return gg_nickname_saved_msg(answer)
    if kind == _INPUT_NOTE:
        nick = str(pending.get("nick", ""))
        if len(answer) > MAX_NOTE_TEXT_CHARS:
            # Ввод НЕ закрывается: игрок дописывает короче, а не начинает путь
            # заново
            # (`test_a_note_longer_than_the_screen_can_show_is_refused_in_words`).
            return note_too_long_msg(MAX_NOTE_TEXT_CHARS)
        await NotesRepo(db).upsert(owner_player_id=player.id, nick=nick, text_=answer)
        await PlayersRepo(db).set_pending_input(player.id, None)
        return note_saved_msg(nick)
    return unknown_text_msg()


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


async def handle_ui_callback(deps: BotDeps, tg_user_id: int, data: str) -> Msg | None:
    """Нажатия кнопок задачи 23: сессии, заметки, ник, диапазоны, реплей, возражение.

    Один обработчик на все префиксы `UI_CALLBACK_PREFIXES`, а не десять входов в
    роутер: роутер по контракту не решает ничего (`bot/router.py`), и разбор
    `callback_data` обязан жить там же, где решения, — то есть здесь.

    **Номера в кнопках приходят из внешнего мира.** Ни один из них не
    используется без сверки владельца: сессия — в `SessionsRepo.summary`,
    заметка — в `NotesRepo`, раздача — в `HandsRepo.find_session_by_hand_no`
    (`test_a_button_of_another_players_note_changes_nothing`).
    """
    async with deps.db_factory() as db:
        player = await _known_player(db, tg_user_id)
        if player is None:
            await db.commit()
            return invite_required_msg()
        reply = await _dispatch_ui(deps, db, player, data)
        await db.commit()
    if reply is _NEW_SESSION_REQUESTED:
        # «Начать новую» — та же логика, что `/new` (SESSIONS_UX: обработчик
        # один). Зовётся ПОСЛЕ коммита: `handle_new_session` открывает свою
        # транзакцию, и вложенной она быть не может.
        return await handle_new_session(deps, tg_user_id)
    return reply


# Метка «этот путь заканчивается вызовом `/new`»: сам вызов делается за границей
# транзакции, поэтому диспетчер возвращает признак, а не сообщение.
_NEW_SESSION_REQUESTED = Msg(text="")


async def _dispatch_ui(deps: BotDeps, db: AsyncSession, player: Player, data: str) -> Msg | None:
    """Разбор `callback_data` по префиксам. Порядок проверок значим: `notecolorset:`
    и `notecolor:` начинаются одинаково, и общий префикс обязан проверяться позже.
    """
    if data == NEW_SESSION_DATA:
        return _NEW_SESSION_REQUESTED
    if data == SET_NICKNAME_DATA:
        await PlayersRepo(db).set_pending_input(player.id, {"kind": _INPUT_NICKNAME})
        return ask_gg_nickname_msg()
    if data.startswith(SESSION_PREFIX):
        rest = data.removeprefix(SESSION_PREFIX)
        if not rest.isdigit():
            return session_unavailable_msg()
        screen = await session_summary_screen(db, player, int(rest))
        return session_unavailable_msg() if screen is None else screen
    if data.startswith(NOTE_COLOR_SET_PREFIX):
        return await _set_note_color(db, player, data.removeprefix(NOTE_COLOR_SET_PREFIX))
    if data.startswith(NOTE_COLOR_PREFIX):
        note = await _note_by_data(db, player, data.removeprefix(NOTE_COLOR_PREFIX))
        return note_gone_msg() if note is None else note_color_prompt_msg(note)
    if data.startswith(NOTE_EDIT_PREFIX):
        note = await _note_by_data(db, player, data.removeprefix(NOTE_EDIT_PREFIX))
        if note is None:
            return note_gone_msg()
        await PlayersRepo(db).set_pending_input(
            player.id, {"kind": _INPUT_NOTE, "nick": note.nick}
        )
        return note_prompt_msg(note.nick, note)
    if data.startswith(NOTE_DELETE_PREFIX):
        note = await _note_by_data(db, player, data.removeprefix(NOTE_DELETE_PREFIX))
        if note is None or not await NotesRepo(db).delete(note.note_id, player.id):
            return note_gone_msg()
        return note_deleted_msg(note.nick)
    if data.startswith(NOTE_ADD_PREFIX):
        return await _note_prompt_from_button(db, player, data.removeprefix(NOTE_ADD_PREFIX))
    if data.startswith(RANGES_PREFIX):
        return await _ranges_reply(db, player, data.removeprefix(RANGES_PREFIX))
    if data.startswith(DETAIL_PREFIX):
        return await _replay_reply(db, player, data.removeprefix(DETAIL_PREFIX))
    if data.startswith(DISAGREE_PREFIX):
        return await _disagree_reply(db, player, data.removeprefix(DISAGREE_PREFIX))
    return None


async def _note_prompt_from_button(db: AsyncSession, player: Player, rest: str) -> Msg:
    """Кнопка «заметка на оппонента» под разбором: `«{hand_no}:{index}»` → ник.

    Кнопка возит индекс, а не ник (`keyboards.note_buttons_for_hand`), поэтому
    ник берётся из сохранённой руки той же функцией, которая строила список
    кнопок: два разных порядка дали бы заметку не на того оппонента.
    """
    hand_no, _, raw_index = rest.rpartition(":")
    if not hand_no or not raw_index.isdigit():
        return analysis_unavailable_msg()
    hand, _analysis = await _hand_with_analysis(db, player, hand_no)
    if hand is None or hand.canonical is None:
        return analysis_unavailable_msg()
    nicks = note_nicks_for_hand(hand.canonical)
    index = int(raw_index)
    if index >= len(nicks):
        return analysis_unavailable_msg()
    nick = nicks[index]
    await PlayersRepo(db).set_pending_input(player.id, {"kind": _INPUT_NOTE, "nick": nick})
    return note_prompt_msg(nick, await NotesRepo(db).find_by_nick(player.id, nick))


async def _note_by_data(db: AsyncSession, player: Player, raw_id: str):
    """Заметка по номеру из кнопки — только своя. Нецифровой номер это не заметка."""
    if not raw_id.isdigit():
        return None
    return await NotesRepo(db).get(int(raw_id), player.id)


async def _set_note_color(db: AsyncSession, player: Player, rest: str) -> Msg:
    """`notecolorset:{id}:{цвет}` — цвет ставится только из известного набора.

    Ключ цвета сверяется с `NOTE_COLORS`, а не пишется как есть: `callback_data`
    приходит из внешнего мира, и в колонке `notes.color` не должно оказаться
    значения, которого экран не умеет показать.
    """
    raw_id, _, key = rest.partition(":")
    color = next((item for item in NOTE_COLORS if item.key == key), None)
    note = await _note_by_data(db, player, raw_id)
    if note is None or color is None:
        return note_gone_msg()
    await NotesRepo(db).set_color(note.note_id, player.id, color.key)
    return note_color_saved_msg(note.nick, color.label)


async def _hand_with_analysis(db: AsyncSession, player: Player, hand_no: str):
    """Рука игрока и её разбор по номеру раздачи — либо `(None, None)`.

    Ищется по ВСЕЙ истории игрока (`find_session_by_hand_no`), а не в активной
    сессии: кнопка живёт под сообщением, которое игрок может открыть спустя
    вечер, и разбор принадлежит той сессии, где рука лежит (тот же рулинг, что
    у кнопки «разобрать», задача 22).
    """
    session_id = await HandsRepo(db).find_session_by_hand_no(player.id, hand_no)
    if session_id is None:
        return None, None
    hand = await HandsRepo(db).find_by_hand_no(session_id, hand_no)
    if hand is None:
        return None, None
    return hand, await AnalysesRepo(db).get_by_hand(hand.id)


async def _ranges_reply(db: AsyncSession, player: Player, hand_no: str) -> Msg:
    """Кнопка «Диапазоны»: картинки матриц 13×13 из уже нарисованного разбора.

    Файлы рисует воркер и складывает пути в `analyses.range_images`
    (`worker.pipeline._render_ranges`); бот ничего не считает и не рисует — он
    отдаёт то, что посчитано, тем же путём, каким уходит текст.
    """
    _hand, analysis = await _hand_with_analysis(db, player, hand_no)
    if analysis is None:
        return analysis_unavailable_msg()
    return ranges_msg(analysis.range_images or [], analysis.result)


async def _replay_reply(db: AsyncSession, player: Player, hand_no: str) -> Msg:
    """Кнопка «Подробнее»: ход раздачи с выделенной точкой решения героя.

    Реплей считается из `hands.enriched` в момент нажатия, а не хранится: это
    чистая функция от сохранённой руки (`explanation.hand_replay` — ноль токенов
    и ноль расчётов), и держать её результат второй копией было бы вторым
    источником одного и того же.
    """
    hand, _analysis = await _hand_with_analysis(db, player, hand_no)
    if hand is None or hand.enriched is None:
        return replay_unavailable_msg()
    return replay_msg(hand_replay(hand.enriched), hand_no)


async def _disagree_reply(db: AsyncSession, player: Player, hand_no: str) -> Msg:
    """Кнопка «Не согласен»: возражение игрока уходит в eval-датасет.

    Самая ценная кнопка продукта (SESSIONS_UX) и вход четвёртого этажа EVALS:
    строка `eval_cases(kind="verdict_dispute")` — это будущий регрессионный
    случай. Разбор при этом не меняется: возражение — данные, а не правка
    вердикта.
    """
    hand, analysis = await _hand_with_analysis(db, player, hand_no)
    if hand is None or analysis is None:
        return analysis_unavailable_msg()
    await EvalCasesRepo(db).add(
        kind="verdict_dispute",
        hand_id=hand.id,
        ground_truth={"hand_no": hand_no, "disagreed_with": analysis.id},
        source="disagree_button",
    )
    return disagreement_saved_msg()
