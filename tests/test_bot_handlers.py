"""Бот (задача 19): приём файла, молчаливая сессия, `/new`, квота скользящих 24 ч —
на функциях-обработчиках с внедрёнными зависимостями и настоящем Postgres
(`db_factory`, задача 14). Aiogram здесь не участвует: он тонкая обвязка поверх
этих функций (`bot/router.py`), и в окружении задачи нет токена бота.

**Байты файла — синтетические, не реальная фикстура.** Бриф показывает
`FIXTURE_DAILY.read_bytes()`, но `handle_document` файл НЕ разбирает: он считает
sha256, пишет байты на диск и кладёт путь в `jobs.payload` — разбор начинается в
воркере (задача 18, `_run_hh_scan`). Требовать приватную HH-фикстуру ради этого
значило бы пропускать (`skipif`) все гарантии этого файла — молчаливое создание
сессии, переиспользование, `/new`, квоту — на любом клоне без приватных данных,
то есть ровно те тесты, которые контроллер требует уметь ронять. Стык с воркером
пришпилен иначе: `test_document_payload_matches_what_worker_reads` проверяет ключи
`payload` и содержимое файла на диске, то есть весь контракт, который воркер от
бота ждёт.

**Falsификация каждой гарантии — часть приёмки.** Для
`test_document_creates_session_silently`, `test_second_file_same_session`,
`test_new_closes_and_opens`, `test_quota_window_24h` и
`test_quota_window_is_rolling_not_calendar_day` соответствующее поведение было
временно сломано, и тест действительно краснел — см. отчёт задачи 19,
раздел «Falsификация».
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import text

from harness.bot.handlers import (
    BotDeps,
    check_quota,
    handle_deep_dive_callback,
    handle_document,
    handle_new_session,
    handle_start,
)
from harness.contracts import Provenance, RawHand
from harness.memory.models import Job
from harness.memory.repos import HandsRepo, PlayersRepo, SessionsRepo
from harness.platform.queue import JobsQueue
from harness.presentation import (
    hh_accepted_msg,
    hh_duplicate_msg,
    new_session_msg,
    quota_exceeded_msg,
    start_msg,
    unsupported_document_msg,
)

# Содержимое файла для бота непрозрачно (см. модульный докстринг) — любые байты.
_HH_BYTES = b"synthetic hand history payload, opaque to the bot\n"

_TG_USER_ID = 777


@pytest.fixture
def queue(db_factory) -> JobsQueue:
    return JobsQueue(db_factory)


@pytest.fixture
def deps(db_factory, queue, tmp_path: Path) -> BotDeps:
    """`data_dir` — `tmp_path` теста, а не `/data` прода: файлы игрока пишутся
    по-настоящему (гарантия «файл на диске» иначе непроверяема), но живут ровно
    столько, сколько тест.
    """
    return BotDeps(db_factory=db_factory, queue=queue, data_dir=tmp_path)


async def fetch_all(db_factory, sql: str) -> list[dict]:
    async with db_factory() as session:
        rows = (await session.execute(text(sql))).mappings().all()
        return [dict(row) for row in rows]


async def fetch_one(db_factory, sql: str) -> dict:
    rows = await fetch_all(db_factory, sql)
    assert len(rows) == 1, f"ожидали ровно одну строку, получили {len(rows)}"
    return rows[0]


async def _seed_player(
    db_factory, *, tg_user_id: int = _TG_USER_ID, quota_daily: int | None = None
) -> tuple[int, int]:
    """Игрок с активной сессией — валидные FK для `jobs` (тот же приём, что
    `_make_scope` в `test_worker_pipeline.py`)."""
    async with db_factory() as session:
        player = await PlayersRepo(session).get_or_create(tg_user_id=tg_user_id)
        if quota_daily is not None:
            player.quota_daily = quota_daily
        session_row = await SessionsRepo(session).active_or_create(player.id)
        await session.commit()
        return player.id, session_row.id


async def _seed_jobs(
    db_factory, *, player_id: int, session_id: int, ages: list[timedelta], type: str = "deep_dive"
) -> list[int]:
    """Закрытые задачи с ЗАДАННЫМ возрастом: окно квоты считается по
    `jobs.created_at`, и подделать его можно только явной записью (`now()` по
    умолчанию дал бы только «сейчас»). `status='done'` — не деталь: партиционный
    уникальный индекс `uq_jobs_player_id_running` не пустил бы вторую `running`.
    """
    now = datetime.now(UTC)
    ids: list[int] = []
    async with db_factory() as session:
        for age in ages:
            job = Job(
                type=type,
                status="done",
                payload={},
                player_id=player_id,
                session_id=session_id,
                created_at=now - age,
            )
            session.add(job)
            await session.flush()
            ids.append(job.id)
        await session.commit()
    return ids


def _synthetic_raw(hand_no: str) -> RawHand:
    """Минимальная, но НАСТОЯЩАЯ `RawHand` — не словарь и не сырой INSERT.

    Поиск сессии по номеру раздачи читает `hands.raw->>'hand_no'`, то есть то,
    что туда положил `model_dump(mode="json")` контракта. Подделать строку `hands`
    голым SQL значило бы проверять запрос против собственной выдумки о формате
    jsonb, а не против того, что пишет прод.
    """
    return RawHand(
        provenance=Provenance.HAND_HISTORY,
        source_ref="synthetic",
        hand_no=hand_no,
        tournament_id="T1",
        tournament_name="Synthetic",
        level=1,
        sb=10,
        bb=20,
        ante=2,
        timestamp=datetime(2026, 9, 4, 12, 0, tzinfo=UTC),
        table_name="1",
        max_seats=9,
        button_seat=1,
        seats=[],
        posts=[],
    )


async def _seed_hand(db_factory, *, session_id: int, hand_no: str) -> int:
    async with db_factory() as session:
        hand_id = await HandsRepo(session).save_raw(
            session_id=session_id, raw=_synthetic_raw(hand_no)
        )
        await session.commit()
        return hand_id


async def _warm_pool(db_factory, connections: int = 2) -> None:
    """Держать N соединений открытыми одновременно, чтобы пул их создал заранее.

    Без этого конкурентный тест не проверяет то, что обещает (этот проект уже
    ловил такое однажды): первая корутина получает готовое соединение из пула и
    успевает СДЕЛАТЬ ВСЁ И ЗАКОММИТИТЬ, пока вторая ждёт установления нового
    TCP-соединения с Postgres. Гонка не воспроизводится, тест зелёный при любой
    реализации — проверено falsификацией, см. отчёт fix round 1.
    """
    sessions = [db_factory() for _ in range(connections)]
    try:
        for session in sessions:
            await session.execute(text("SELECT 1"))
    finally:
        for session in sessions:
            await session.close()


async def _set_job_age(db_factory, job_id: int, age: timedelta) -> None:
    async with db_factory() as session:
        job = await session.get(Job, job_id)
        assert job is not None
        job.created_at = datetime.now(UTC) - age
        await session.commit()


# --- HH-путь: файл → молчаливая сессия → задача ------------------------------------


async def test_document_creates_session_silently(db_factory, deps):
    """Спека §6/§13 шаг 6: игрок не просил открывать сессию — она появляется
    молча, потому что результату нужно куда лечь (`jobs.session_id NOT NULL`).
    """
    msg = await handle_document(
        deps, tg_user_id=_TG_USER_ID, file_bytes=_HH_BYTES, filename="t.txt"
    )

    s = await fetch_one(db_factory, "select * from sessions")  # сессии не было — создана молча
    j = await fetch_one(db_factory, "select * from jobs")
    assert j["type"] == "hh_scan" and j["session_id"] == s["id"]  # NOT NULL по построению

    player = await fetch_one(db_factory, "select * from players")
    assert player["tg_user_id"] == _TG_USER_ID
    # «Молча» — это и про текст: подтверждение не рассказывает про сессию и не
    # просит её открыть. Регресс формулировки («Открыл новую сессию…») ронял бы
    # ровно ту продуктовую гарантию, ради которой сессия и создаётся молча.
    assert msg == hh_accepted_msg()
    assert "сесси" not in msg.text.lower()


async def test_document_payload_matches_what_worker_reads(db_factory, deps, tmp_path: Path):
    """Стык с задачей 18: `_run_hh_scan` читает `payload["source_file"]` и, если
    он есть, `payload["tournament_id"]` — второй строки `tournaments` на тот же
    файл не заводится. Имя файла на диске — sha256 содержимого (`DATA_DIR/hh/
    {hash}.txt`), а не присланное игроком имя.
    """
    await handle_document(deps, tg_user_id=_TG_USER_ID, file_bytes=_HH_BYTES, filename="t.txt")

    path = tmp_path / "hh" / f"{hashlib.sha256(_HH_BYTES).hexdigest()}.txt"
    assert path.read_bytes() == _HH_BYTES

    t = await fetch_one(db_factory, "select * from tournaments")
    j = await fetch_one(db_factory, "select * from jobs")
    assert j["payload"] == {"source_file": str(path), "tournament_id": t["id"]}
    assert t["source_file"] == str(path)
    assert t["session_id"] == j["session_id"]


async def test_second_file_same_session(db_factory, deps):
    """«Турнир — единица внутри сессии, а не сессия» (SESSIONS_UX): второй файл
    вечера прикрепляется к уже активной сессии, а не открывает новую.
    """
    await handle_document(deps, tg_user_id=_TG_USER_ID, file_bytes=b"first file", filename="a.txt")
    await handle_document(deps, tg_user_id=_TG_USER_ID, file_bytes=b"second file", filename="b.txt")

    sessions = await fetch_all(db_factory, "select * from sessions")
    jobs = await fetch_all(db_factory, "select * from jobs order by id")
    tournaments = await fetch_all(db_factory, "select * from tournaments order by id")

    assert len(sessions) == 1
    assert len(jobs) == 2
    assert len(tournaments) == 2
    assert {j["session_id"] for j in jobs} == {sessions[0]["id"]}
    assert {t["session_id"] for t in tournaments} == {sessions[0]["id"]}


async def test_same_file_twice_is_not_analysed_twice(db_factory, deps):
    """Тот же файл, присланный дважды: второй раз не заводится ни турнир, ни
    задача (fix round 1). Без этого игрок получал бы вторую копию всех `hands` в
    одной сессии и две одинаковые сводки.
    """
    first = await handle_document(
        deps, tg_user_id=_TG_USER_ID, file_bytes=_HH_BYTES, filename="t.txt"
    )
    second = await handle_document(
        deps, tg_user_id=_TG_USER_ID, file_bytes=_HH_BYTES, filename="снова-он.txt"
    )

    assert first == hh_accepted_msg()
    assert second == hh_duplicate_msg()
    assert len(await fetch_all(db_factory, "select * from jobs")) == 1
    assert len(await fetch_all(db_factory, "select * from tournaments")) == 1
    assert len(await fetch_all(db_factory, "select * from sessions")) == 1


async def test_different_files_in_one_session_are_both_accepted(db_factory, deps):
    """Защита от дубля не должна ловить РАЗНЫЕ файлы: имя на диске — хэш
    содержимого, и второй турнир вечера обязан приниматься как обычно.
    """
    await handle_document(deps, tg_user_id=_TG_USER_ID, file_bytes=b"file one", filename="a.txt")
    second = await handle_document(
        deps, tg_user_id=_TG_USER_ID, file_bytes=b"file two", filename="b.txt"
    )

    assert second == hh_accepted_msg()
    assert len(await fetch_all(db_factory, "select * from jobs")) == 2
    assert len(await fetch_all(db_factory, "select * from tournaments")) == 2


async def test_reupload_after_failed_scan_retries_on_the_same_tournament(db_factory, deps):
    """Скан провалился — повторная загрузка это законная повторная попытка, а не
    дубль: новая задача ставится, но турнир переиспользуется (его чекпоинты
    пропустят руки, сохранённые до сбоя).
    """
    await handle_document(deps, tg_user_id=_TG_USER_ID, file_bytes=_HH_BYTES, filename="t.txt")
    async with db_factory() as session:
        job = await session.get(Job, 1)
        assert job is not None
        job.status = "failed"
        job.error = "внутренняя причина, игроку не показывается"
        await session.commit()

    again = await handle_document(
        deps, tg_user_id=_TG_USER_ID, file_bytes=_HH_BYTES, filename="t.txt"
    )

    assert again == hh_accepted_msg()
    jobs = await fetch_all(db_factory, "select * from jobs order by id")
    tournaments = await fetch_all(db_factory, "select * from tournaments")
    assert len(jobs) == 2 and len(tournaments) == 1
    assert jobs[-1]["payload"]["tournament_id"] == tournaments[0]["id"]
    # Внутренний текст `jobs.error` не участвует ни в одном ответе игроку.
    assert "внутренняя причина" not in again.text


async def test_two_files_at_once_from_a_new_player_create_one_player(db_factory, deps):
    """Два файла подряд от НЕЗНАКОМОГО игрока обрабатываются одновременно:
    `players.tg_user_id` уникален, и «прочитали — не нашли — вставили» без
    `ON CONFLICT` роняло вторую транзакцию `IntegrityError` (fix round 1).

    Проверяет РОВНО это. Сериализация сессий сюда не входит: вставка игрока с
    `ON CONFLICT` сама заставляет второго ждать коммита первого, и к моменту его
    пробуждения сессия уже создана и видна — тест про сессии был бы зелёным и без
    лока (проверено falsификацией), поэтому он живёт отдельно, на ЗНАКОМОМ игроке.
    """
    await _warm_pool(db_factory)

    first, second = await asyncio.gather(
        handle_document(deps, tg_user_id=_TG_USER_ID, file_bytes=b"file one", filename="a.txt"),
        handle_document(deps, tg_user_id=_TG_USER_ID, file_bytes=b"file two", filename="b.txt"),
    )

    assert first == hh_accepted_msg() and second == hh_accepted_msg()
    assert len(await fetch_all(db_factory, "select * from players")) == 1
    assert len(await fetch_all(db_factory, "select * from jobs")) == 2


async def test_two_files_at_once_from_a_known_player_share_one_session(db_factory, deps):
    """Игрок уже заведён — вставки в `players` нет, и ничто, кроме лока в
    `active_or_create`, не мешает двум одновременным файлам открыть ДВЕ сессии и
    разложить вечер по двум контейнерам (fix round 1).
    """
    await _seed_player(db_factory)  # игрок и его сессия закрыты предыдущим коммитом
    async with db_factory() as session:
        await SessionsRepo(session).close_active(1)
        await session.commit()

    await _warm_pool(db_factory)

    await asyncio.gather(
        handle_document(deps, tg_user_id=_TG_USER_ID, file_bytes=b"file one", filename="a.txt"),
        handle_document(deps, tg_user_id=_TG_USER_ID, file_bytes=b"file two", filename="b.txt"),
    )

    opened = await fetch_all(db_factory, "select * from sessions where closed_at is null")
    assert len(opened) == 1, "гонка раздвоила вечер игрока"
    jobs = await fetch_all(db_factory, "select * from jobs")
    assert len(jobs) == 2
    assert {j["session_id"] for j in jobs} == {opened[0]["id"]}


async def test_non_txt_document_refused_without_side_effects(db_factory, deps, tmp_path: Path):
    """Не `.txt` — вежливый отказ сразу, а не 20 секунд ожидания и отказ из
    воркера: PokerCraft отдаёт `.txt`, всё остальное сканировать нечем.
    """
    msg = await handle_document(
        deps, tg_user_id=_TG_USER_ID, file_bytes=b"PK\x03\x04zip", filename="hands.zip"
    )

    assert msg == unsupported_document_msg()
    assert await fetch_all(db_factory, "select * from jobs") == []
    assert await fetch_all(db_factory, "select * from sessions") == []
    assert not (tmp_path / "hh").exists()


# --- /new и /start -----------------------------------------------------------------


async def test_new_closes_and_opens(db_factory, deps):
    """`/new` закрывает активную и открывает новую (SESSIONS_UX): следующий файл
    ложится уже в новую, а старая остаётся закрытой — контейнер прошлого вечера.
    """
    await handle_document(deps, tg_user_id=_TG_USER_ID, file_bytes=_HH_BYTES, filename="t.txt")
    first = await fetch_one(db_factory, "select * from sessions")

    msg = await handle_new_session(deps, tg_user_id=_TG_USER_ID)

    rows = await fetch_all(db_factory, "select * from sessions order by id")
    assert len(rows) == 2
    assert rows[0]["id"] == first["id"] and rows[0]["closed_at"] is not None
    assert rows[1]["closed_at"] is None
    assert msg == new_session_msg(rows[1]["title"], previous_closed=True)

    await handle_document(deps, tg_user_id=_TG_USER_ID, file_bytes=b"next file", filename="b.txt")
    jobs = await fetch_all(db_factory, "select * from jobs order by id")
    assert jobs[-1]["session_id"] == rows[1]["id"]


async def test_new_without_active_session_only_opens(db_factory, deps):
    """Закрывать нечего — и сообщение не заявляет, что что-то закрыто."""
    msg = await handle_new_session(deps, tg_user_id=_TG_USER_ID)

    rows = await fetch_all(db_factory, "select * from sessions")
    assert len(rows) == 1 and rows[0]["closed_at"] is None
    assert msg == new_session_msg(rows[0]["title"], previous_closed=False)


async def test_start_registers_player_without_session(db_factory, deps):
    """`/start` заводит игрока (инвайты — задача 23, до неё без ограничений), но
    сессию не открывает: молчаливое создание привязано к присланному материалу,
    а не к нажатию кнопки «Start».
    """
    msg = await handle_start(deps, tg_user_id=_TG_USER_ID)

    player = await fetch_one(db_factory, "select * from players")
    assert player["tg_user_id"] == _TG_USER_ID
    assert await fetch_all(db_factory, "select * from sessions") == []
    assert msg == start_msg()


# --- квота: скользящее окно 24 ч (спека §9) ----------------------------------------


async def test_quota_window_24h(db_factory, deps):
    """3 задачи: 2 внутри окна, 1 старше 24 ч; `quota_daily=2` → запрещено.
    Старение той, что внутри окна, освобождает место — без всякого сброса по
    расписанию.
    """
    player_id, session_id = await _seed_player(db_factory, quota_daily=2)
    job_ids = await _seed_jobs(
        db_factory,
        player_id=player_id,
        session_id=session_id,
        ages=[timedelta(minutes=5), timedelta(hours=20), timedelta(hours=30)],
    )

    quota = await check_quota(deps, player_id)
    assert quota.allowed is False
    assert (quota.left, quota.total) == (0, 2)
    # Место освободит самая старая ИЗ ОКНА (20 ч назад) — через 4 часа.
    assert quota.hours_to_free == 4

    await _set_job_age(db_factory, job_ids[1], timedelta(hours=25))

    freed = await check_quota(deps, player_id)
    assert freed.allowed is True
    assert (freed.left, freed.total, freed.hours_to_free) == (1, 2, 0)


async def test_quota_window_is_rolling_not_calendar_day(db_factory, deps):
    """Окно скользящее, а не «сегодня» (спека §9: ни cron-сброса, ни полуночного
    обнуления посреди ночной сессии).

    Задача заводится так, чтобы попасть внутрь скользящих 24 часов, но на
    ПРЕДЫДУЩИЕ календарные сутки: реализация, считающая «с полуночи», не увидела
    бы её и разрешила разбор. Возраст `(прошло_с_полуночи + 24 ч) / 2` даёт такую
    точку в любое время суток (она всегда до сегодняшней полуночи и всегда позже
    `now - 24 ч`), поэтому тест не зависит от того, когда его запустили.
    """
    now = datetime.now(UTC)
    since_midnight = now - now.replace(hour=0, minute=0, second=0, microsecond=0)
    age = (since_midnight + timedelta(hours=24)) / 2
    created_at = now - age
    assert created_at < now.replace(hour=0, minute=0, second=0, microsecond=0)  # вчера по календарю
    assert age < timedelta(hours=24)  # но внутри скользящего окна

    player_id, session_id = await _seed_player(db_factory, quota_daily=1)
    await _seed_jobs(db_factory, player_id=player_id, session_id=session_id, ages=[age])

    quota = await check_quota(deps, player_id)
    assert quota.allowed is False, "счёт «за сегодня» вместо скользящих 24 ч"
    assert quota.hours_to_free >= 1


async def test_quota_counts_only_interactive_jobs(db_factory, deps):
    """Считаются интерактивные задачи (спека §9); HH-скан дёшев для нас и
    «поощряется щедрее» — квоту он не тратит.
    """
    player_id, session_id = await _seed_player(db_factory, quota_daily=1)
    await _seed_jobs(
        db_factory,
        player_id=player_id,
        session_id=session_id,
        ages=[timedelta(minutes=1)],
        type="hh_scan",
    )
    assert (await check_quota(deps, player_id)).allowed is True

    await _seed_jobs(
        db_factory,
        player_id=player_id,
        session_id=session_id,
        ages=[timedelta(minutes=1)],
        type="screenshot_analyze",
    )
    assert (await check_quota(deps, player_id)).allowed is False


async def test_failed_jobs_do_not_spend_quota(db_factory, deps):
    """Задача, упавшая по НАШЕЙ причине, не стоит игроку разбора (рулинг fix
    round 1): полоса наших сбоев иначе съедала бы чужой день целиком.
    """
    player_id, session_id = await _seed_player(db_factory, quota_daily=1)
    job_ids = await _seed_jobs(
        db_factory, player_id=player_id, session_id=session_id, ages=[timedelta(minutes=5)]
    )
    assert (await check_quota(deps, player_id)).allowed is False  # пока задача успешна

    async with db_factory() as session:
        job = await session.get(Job, job_ids[0])
        assert job is not None
        job.status = "failed"
        job.error = "боевой текст ошибки с путём /data/hh/deadbeef.txt"
        await session.commit()

    freed = await check_quota(deps, player_id)
    assert freed.allowed is True and freed.left == 1


async def test_quota_default_total_without_personal_override(db_factory, deps):
    """`players.quota_daily IS NULL` — действует дефолт (спека §9, пример «17/50»),
    а не «безлимит» и не ноль."""
    player_id, _ = await _seed_player(db_factory)
    quota = await check_quota(deps, player_id)
    assert quota.allowed is True
    assert quota.total == 50 and quota.left == 50


# --- кнопка [разобрать] ------------------------------------------------------------


async def test_deep_dive_callback_enqueues_silently(db_factory, deps):
    """Нажатие кнопки под сводкой ставит `deep_dive` в сессию, где лежит раздача,
    и НИЧЕГО не отвечает: дальше говорит воркер (прогресс-сообщение, задача 18),
    а второй текст от бота был бы дублем.
    """
    await handle_document(deps, tg_user_id=_TG_USER_ID, file_bytes=_HH_BYTES, filename="t.txt")
    active = await fetch_one(db_factory, "select * from sessions")
    await _seed_hand(db_factory, session_id=active["id"], hand_no="RC1234")

    result = await handle_deep_dive_callback(deps, tg_user_id=_TG_USER_ID, hand_no="RC1234")

    assert result is None
    jobs = await fetch_all(db_factory, "select * from jobs order by id")
    assert jobs[-1]["type"] == "deep_dive"
    assert jobs[-1]["payload"] == {"hand_no": "RC1234"}
    assert jobs[-1]["session_id"] == active["id"]


async def test_deep_dive_goes_to_the_session_that_holds_the_hand(db_factory, deps):
    """Кнопка под сводкой ЗАКРЫТОГО вечера работает (рулинг fix round 1).

    Разбор ищет руку как `find_by_hand_no(job.session_id, hand_no)`, поэтому
    задача, поставленная в активную сессию, не нашла бы раздачу из прошлой и
    отказала бы честно и бессмысленно. Анализ принадлежит сессии, где рука живёт.
    """
    await handle_document(deps, tg_user_id=_TG_USER_ID, file_bytes=_HH_BYTES, filename="t.txt")
    old_session = await fetch_one(db_factory, "select * from sessions")
    await _seed_hand(db_factory, session_id=old_session["id"], hand_no="RC1234")

    await handle_new_session(deps, tg_user_id=_TG_USER_ID)  # прошлый вечер закрыт
    sessions = await fetch_all(db_factory, "select * from sessions order by id")
    assert sessions[-1]["id"] != old_session["id"] and sessions[-1]["closed_at"] is None

    assert await handle_deep_dive_callback(deps, tg_user_id=_TG_USER_ID, hand_no="RC1234") is None

    job = (await fetch_all(db_factory, "select * from jobs order by id"))[-1]
    assert job["type"] == "deep_dive"
    assert job["session_id"] == old_session["id"], "задача ушла в активную сессию, а не в свою"


async def test_deep_dive_never_resolves_into_another_players_session(db_factory, deps):
    """Область поиска — сессии ЭТОГО игрока: `hand_no` уникален в рамках
    источника, а не глобально, и одинаковый номер у двух игроков не должен
    отправлять разбор одного в сессию другого.
    """
    stranger_id, stranger_session = await _seed_player(db_factory, tg_user_id=999)
    await _seed_hand(db_factory, session_id=stranger_session, hand_no="RC1234")

    await handle_document(deps, tg_user_id=_TG_USER_ID, file_bytes=_HH_BYTES, filename="t.txt")
    own_session = await fetch_one(
        db_factory, f"select * from sessions where id <> {stranger_session}"
    )

    await handle_deep_dive_callback(deps, tg_user_id=_TG_USER_ID, hand_no="RC1234")

    job = (await fetch_all(db_factory, "select * from jobs where type = 'deep_dive'"))[-1]
    assert job["session_id"] == own_session["id"]
    assert job["session_id"] != stranger_session
    assert job["player_id"] != stranger_id


async def test_deep_dive_falls_back_to_active_session_when_hand_is_unknown(db_factory, deps):
    """Раздачи нет ни в одной сессии игрока — ставим в активную и даём воркеру
    отказать своим единым текстом, а не заводим второй путь отказа.
    """
    await handle_document(deps, tg_user_id=_TG_USER_ID, file_bytes=_HH_BYTES, filename="t.txt")
    active = await fetch_one(db_factory, "select * from sessions")

    assert await handle_deep_dive_callback(deps, tg_user_id=_TG_USER_ID, hand_no="НЕТ-ТАКОЙ") is None

    job = (await fetch_all(db_factory, "select * from jobs order by id"))[-1]
    assert job["type"] == "deep_dive" and job["session_id"] == active["id"]


async def test_deep_dive_callback_refuses_when_quota_exhausted(db_factory, deps):
    """Отказ приходит ДО постановки в очередь (воркер уже не вправе отказывать) и
    текстом из `presentation` — со временем возврата, а не с остатком.
    """
    player_id, session_id = await _seed_player(db_factory, quota_daily=1)
    await _seed_jobs(
        db_factory, player_id=player_id, session_id=session_id, ages=[timedelta(hours=1)]
    )

    msg = await handle_deep_dive_callback(deps, tg_user_id=_TG_USER_ID, hand_no="RC1234")

    assert msg == quota_exceeded_msg(23)
    jobs = await fetch_all(db_factory, "select * from jobs")
    assert len(jobs) == 1  # ничего не добавилось
