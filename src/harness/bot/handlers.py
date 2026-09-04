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

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from harness.memory.repos import (
    HandsRepo,
    JobsRepo,
    PlayersRepo,
    QuotaCheck,
    QuotaRepo,
    SessionsRepo,
    TournamentsRepo,
)
from harness.platform.queue import JobsQueue
from harness.presentation import (
    Msg,
    hh_accepted_msg,
    hh_duplicate_msg,
    new_session_msg,
    quota_exceeded_msg,
    start_msg,
    unsupported_document_msg,
)

__all__ = [
    "BotDeps",
    "QuotaCheck",
    "check_quota",
    "handle_deep_dive_callback",
    "handle_document",
    "handle_new_session",
    "handle_start",
]

# PokerCraft отдаёт историю раздач текстом; всё остальное сканировать нечем.
_HH_SUFFIX = ".txt"


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
