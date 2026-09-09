"""ORM-модели БД: 12 таблиц спеки §6, две таблицы оппонента и точки решения.

Источник — `docs/superpowers/specs/2026-08-28-poker-harness-tech-spec-design.md`,
раздел "6. Схема БД". Колонки — по табличным строкам спеки дословно; `?` у поля в
спеке значит nullable, отсутствие `?` — `NOT NULL`. Шесть отступлений от этого
правила и почему они не нарушают "дословно":

1. `llm_calls.started_at` — таблица §6 её не называет среди "ключевых полей", но §7
   описывает ровно эту колонку: лимитер пишет строку "при старте вызова", а окно
   темпа считает `WHERE started_at > now() - interval '60 seconds'` (задача 16,
   `PgLimiter`). Индекс `llm_calls(started_at)`, тоже из этой задачи, без колонки
   строить не на чем — молчание таблицы §6 здесь пробел терминологии, а не запрет.
2. `llm_calls.tokens_in/tokens_out/cost/latency_ms` — по терзости строки таблицы
   были бы `NOT NULL`, но §7 явно описывает запись строки ДО завершения вызова
   (status='started'), когда этих чисел ещё нет: "статус обновляется по
   завершении". Nullable — не отступление от спеки, а следствие второй её части.
3. `invites.id` — таблица §6 не пишет "id" явно (только "code UNIQUE, issued_by,
   ..."), но UNIQUE у `code` рядом с обычным набором остальных таблиц ("id,
   ключ UNIQUE, ...", ср. `players`) читается как тот же паттерн: суррогатный id
   плюс отдельно помеченный уникальный бизнес-ключ. `calc_cache.key` — образец
   противоположного случая: там натуральный ключ и есть PK, и спека его никак не
   помечает (UNIQUE избыточен для PK). Инвайты собраны по образцу `players`.
4. `players.pending_input` (задача 23, миграция 0005) — колонки нет в §6 вовсе.
   Она хранит не знание о покере, а незакрытый диалог бота: что означает
   следующее текстовое сообщение игрока. Спека §8.3 требует, чтобы состояние
   ожидания ответа жило в БД, а не в памяти процесса, — для задач это
   `jobs.payload`, но ввод ника и текста заметки задачей не сопровождается, и
   складывать его было некуда.
5. `opponents`/`opponent_links` (миграции 0006 и 0007) — таблиц нет в §6:
   на момент спеки сшивать личность между турнирами было нечем. Идентификатор
   участника в файлах раздач сквозной внутри турнира и не переживает его, а
   владелец хочет считать статистику по человеку, а не по турнирной метке.
   Связь между метками утверждает владелец; кода, который догадывался бы о ней
   сам, в проекте нет. Заметка (§6, `notes`) ссылается на ту же строку
   `opponents`: личность оппонента в продукте одна (миграция 0007).
6. `decision_points` (миграция 0008) — таблицы нет в §6: точки решения лежали
   массивом внутри `analyses.result`. Правило §6 («jsonb — для версионированных
   документов-контрактов; реляционные колонки — для всего, по чему ищем и
   джойним») ими же и нарушалось: по точкам ищут и группируют — лики, покрытие,
   цена вечера, — а каждое новое условие писалось выражением по jsonb руками.
   Отступление здесь в том, что таблицы нет в §6, а не в том, что она спорит с
   правилом.

jsonb-колонки хранят `model_dump(mode="json")` пайплайн-контрактов (`RawHand`,
`CanonicalHand`, `EnrichedHand`, `AnalysisResult`) — уже JSON-совместимые
примитивы (datetime и StrEnum-ключи словарей сериализуются в строки), поэтому
кастомный сериализатор не нужен: `model_validate` того же контракта читает jsonb
обратно один в один (проверено на реальной руке из `tests/test_hh_parser.py`).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Double,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    Numeric,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
    false,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Конвенция имён constraint'ов: без неё alembic autogenerate придумывает имена
# индексов/ограничений сам на каждый прогон, и диффы миграций захламляются
# переименованиями, которых по факту не было.
_NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=_NAMING_CONVENTION)


class Player(Base):
    """Сквозной уровень: `players`. Израсходованное за день выводится из `jobs` (§9)."""

    __tablename__ = "players"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tg_user_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False)
    # Ник в руме — единственный способ опознать героя на скриншоте КОДОМ, а не
    # моделью (реестр vision, решение 2026-09-05 «Герой определяется кодом»: ни
    # одна контрольная сумма подменённого героя не ловит). NULL — ника ещё не
    # спросили; тогда разбор скрина упирается в вопрос игроку, а не угадывает.
    gg_nickname: Mapped[str | None] = mapped_column(String(64))
    # NULL = у игрока нет персонального переопределения, действует дефолт из
    # Config (спека §7: "квоты по умолчанию" — конфиг, не схема БД).
    quota_daily: Mapped[int | None] = mapped_column(Integer)
    subscription: Mapped[str] = mapped_column(String(32), nullable=False, server_default="free")
    is_dev: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=false())
    # Что означает СЛЕДУЮЩЕЕ текстовое сообщение игрока: ник в руме, текст
    # заметки — либо ничего (NULL, обычное состояние). Состояние ввода живёт в
    # БД, а не в памяти процесса бота, по той же причине, что и состояние
    # эскалации в `jobs.payload` (спека §8.3): перезапуск бота не имеет права
    # терять половину диалога. Колонки в таблице §6 нет — это 4-е отступление,
    # см. пункт 4 модульного докстринга.
    pending_input: Mapped[Any | None] = mapped_column(JSONB)


class Invite(Base):
    """Доступ по инвайтам: `invites`. `id` — см. пункт 3 в модульном докстринге."""

    __tablename__ = "invites"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    issued_by: Mapped[int] = mapped_column(BigInteger, ForeignKey("players.id"), nullable=False)
    used_by: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("players.id"))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Session(Base):
    """«Сессия = вечер»: `sessions`. Закрывается `/new` (SESSIONS_UX.md), не таймером."""

    __tablename__ = "sessions"
    __table_args__ = (
        # Мишень составного внешнего ключа `decision_points(session_id,
        # player_id)` — см. `uq_hands_id_session`.
        UniqueConstraint("id", "player_id", name="uq_sessions_id_player"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    player_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("players.id"), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    title: Mapped[str] = mapped_column(String, nullable=False)


class Tournament(Base):
    """HH-вход: `tournaments`. Файл лежит на диске в volume, здесь — путь и сводка скана."""

    __tablename__ = "tournaments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("sessions.id"), nullable=False
    )
    source_file: Mapped[str] = mapped_column(String, nullable=False)
    scan_summary: Mapped[Any | None] = mapped_column(JSONB)


class Hand(Base):
    """Контракты руки: `hands`. Nullable-колонки canonical/enriched — чекпоинты (§8.2),
    а не отсутствие данных: пайплайн мог дойти до этой строки и остановиться здесь.
    """

    __tablename__ = "hands"
    __table_args__ = (
        CheckConstraint(
            "provenance IN ('hand_history', 'screenshot')", name="provenance_allowed"
        ),
        Index("ix_hands_session_id", "session_id"),
        # Не «вторая уникальность» руки, а мишень составного внешнего ключа
        # `decision_points(hand_id, session_id)`: без неё денормализованный
        # `session_id` точки некому проверить (тот же приём, что у
        # `uq_opponents_id_owner`).
        UniqueConstraint("id", "session_id", name="uq_hands_id_session"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("sessions.id"), nullable=False
    )
    tournament_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("tournaments.id")
    )
    provenance: Mapped[str] = mapped_column(String(16), nullable=False)
    image_hash: Mapped[str | None] = mapped_column(String)
    raw: Mapped[Any] = mapped_column(JSONB, nullable=False)
    canonical: Mapped[Any | None] = mapped_column(JSONB)
    enriched: Mapped[Any | None] = mapped_column(JSONB)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)


class Analysis(Base):
    """Выход ядра и изложения: `analyses`."""

    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    hand_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("hands.id"), nullable=False)
    result: Mapped[Any] = mapped_column(JSONB, nullable=False)
    verdict_text: Mapped[str | None] = mapped_column(Text)
    range_images: Mapped[Any | None] = mapped_column(JSONB)


class DecisionPointRow(Base):
    """Точка решения строкой: `decision_points`. ЕДИНСТВЕННОЕ место, где она лежит.

    До миграции 0008 массив точек жил в `analyses.result -> 'points'` и
    разворачивался `jsonb_array_elements` на каждом запросе. Теперь `result`
    хранит документ БЕЗ точек, а `AnalysesRepo` собирает его обратно из этих
    строк: двух копий одного вердикта в базе нет
    (`test_the_analysis_document_no_longer_carries_its_points`).

    Имя класса — `...Row`, а не `DecisionPoint`: контракт с таким именем уже
    есть (`contracts.enriched`), и строка не равна ему — она склеена из двух
    контрактов сразу. Что откуда:

    * `point_no`, `dp_index`, `street`, `spot`, `zone`, `action_taken`,
      `best_action`, `ev_diff_bb`, `ev_interval`, `assumption`, `tools`,
      `detail` — `PointVerdict` целиком, поле в поле
      (`test_every_field_of_a_point_verdict_has_its_column`). Единственное
      переименование — `interval` → `ev_interval`: `interval` в Postgres
      зарезервировано, и колонка с таким именем требовала бы кавычек в каждом
      запросе.
    * `position`, `to_call`, `pot_before`, `eff_stack`, `eff_stack_bb`, `spr` —
      обстановка точки из `DecisionPoint` (`hands.enriched`). Nullable: вердикт
      сопоставляется с обстановкой по `dp_index`, и вердикт без своей точки
      решения контракт не запрещает.
    * `judged` — ответ `contracts.is_judged` на момент записи. Колонка, а не
      выражение в SQL: иначе правило судимости существовало бы двумя текстами
      на двух языках, обязанными совпадать. Это СНИМОК правила, а не
      вычисляемое свойство: изменится правило — строки, записанные раньше,
      останутся с прежним ответом, и покрытие сложит две редакции. Правило
      целиком выписано в
      `test_the_judged_rule_is_pinned_because_the_column_freezes_it`, и правка
      правила краснит его вместе с требованием переливки.
    * `hand_id`, `session_id`, `player_id` — адрес точки. Сессия и игрок
      денормализованы (фильтры словаря расчётов складываются по игроку и по
      вечеру), и оба закрыты составными внешними ключами на
      `hands(id, session_id)` и `sessions(id, player_id)`: строка с чужой
      сессией или чужим игроком не вставляется вовсе
      (`test_a_point_cannot_claim_a_session_that_is_not_its_hands`).

    `point_no` — МЕСТО В МАССИВЕ `AnalysisResult.points`, а не `dp_index`:
    `AnalysisResult.ranked` индексирует именно массив, и порядок обязан
    пережить запись. Равенство `point_no == dp_index` в разборах ядра
    выполняется, но контракт его не требует, и ключом взято то, что нужно для
    сборки документа обратно.
    """

    __tablename__ = "decision_points"
    __table_args__ = (
        ForeignKeyConstraint(
            ["hand_id", "session_id"],
            ["hands.id", "hands.session_id"],
            name="fk_decision_points_hand_id_hands",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["session_id", "player_id"],
            ["sessions.id", "sessions.player_id"],
            name="fk_decision_points_session_id_sessions",
        ),
        UniqueConstraint("hand_id", "point_no", name="uq_decision_points_hand_id_point_no"),
        # Обе сквозные выборки идут по игроку: «Мои лики» — по всей истории,
        # агрегат вечера — по нему же плюс сессия.
        Index("ix_decision_points_player_id", "player_id"),
        Index("ix_decision_points_session_id", "session_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    hand_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    session_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    player_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    point_no: Mapped[int] = mapped_column(Integer, nullable=False)
    dp_index: Mapped[int] = mapped_column(Integer, nullable=False)
    street: Mapped[str] = mapped_column(String(16), nullable=False)
    spot: Mapped[str] = mapped_column(String(32), nullable=False)
    zone: Mapped[str] = mapped_column(String(16), nullable=False)
    action_taken: Mapped[str] = mapped_column(String, nullable=False)
    best_action: Mapped[str] = mapped_column(String, nullable=False)
    ev_diff_bb: Mapped[float] = mapped_column(Double, nullable=False)
    judged: Mapped[bool] = mapped_column(Boolean, nullable=False)
    position: Mapped[str | None] = mapped_column(String(16))
    to_call: Mapped[int | None] = mapped_column(BigInteger)
    pot_before: Mapped[int | None] = mapped_column(BigInteger)
    eff_stack: Mapped[int | None] = mapped_column(BigInteger)
    eff_stack_bb: Mapped[float | None] = mapped_column(Double)
    spr: Mapped[float | None] = mapped_column(Double)
    ev_interval: Mapped[Any | None] = mapped_column(JSONB)
    assumption: Mapped[Any | None] = mapped_column(JSONB)
    tools: Mapped[Any] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    detail: Mapped[Any] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))


class Note(Base):
    """Заметки на игроков: `notes`. Только через vision — в HH ники анонимны.

    Заметка на оппонента одна и накапливается (SESSIONS_UX: «оппонент
    встречается в разных сессиях, заметка должна накапливаться»), поэтому
    уникален `opponent_id` — не пара с владельцем: владелец у оппонента один и
    тот же, и второй раз в ключе он ничего не добавляет (миграция 0007).

    **Личность оппонента здесь не своя, а общая** — строка `opponents`. До
    миграции 0007 заметка держала ник строкой, и регистр в ней различался, а в
    `opponents` — нет: один и тот же ник в разном написании был одним
    оппонентом для статистики и двумя для заметок
    (`test_a_note_and_a_link_on_the_same_nick_in_two_cases_meet_on_one_opponent`).

    `owner_player_id` остаётся колонкой, хотя выводится из `opponents`:
    по нему сверяется владелец во всех методах `NotesRepo`, а составной внешний
    ключ на `opponents(id, owner_player_id)` не даёт ему разойтись с владельцем
    самого оппонента — тот же приём, что в `OpponentLink`.
    """

    __tablename__ = "notes"
    __table_args__ = (
        Index("uq_notes_opponent_id", "opponent_id", unique=True),
        ForeignKeyConstraint(
            ["opponent_id", "owner_player_id"],
            ["opponents.id", "opponents.owner_player_id"],
            name="fk_notes_opponent_opponents",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    owner_player_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("players.id"), nullable=False
    )
    opponent_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    color: Mapped[str] = mapped_column(String(32), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class Opponent(Base):
    """`opponents`: оппонент владельца, известный по нику в руме.

    Одна личность на весь продукт: сюда же ссылается заметка (`Note`), сюда же
    привязываются пары «турнир + идентификатор» (`OpponentLink`). Идентификатор
    участника в файлах раздач сквозной внутри турнира, но между турнирами не
    живёт — комната выдаёт новый; ник живёт.

    **Ключ личности — ник, а не произвольный ярлык** (решение владельца).

    **Регистр не различается, написание сохраняется.** Назвать тот же ник второй
    раз — единственный способ сказать «вот этот идентификатор из другого турнира
    это он же», и разъехаться на собственной опечатке в регистре этот путь не
    имеет права. Уникальность держит функциональный индекс по
    `(owner_player_id, lower(opponent_nick))`, а колонка хранит ник так, как его
    ввели в первый раз (`test_the_same_nick_in_another_case_is_the_same_opponent`).
    Имени индекса в коде нет и не нужно: `ON CONFLICT` в
    `OpponentsRepo.get_or_create` выводится по тем же выражениям, а не по имени.

    **Длина ника не ограничена схемой.** До миграции 0007 колонка повторяла
    `players.gg_nickname` (64), но ник оппонента приходит не из клавиатуры, а со
    стола: кнопка заметки возит индекс именно потому, что ник бывает длиннее 64
    байт (`test_a_long_nick_in_a_note_button_still_opens_the_right_note`), и
    заметка на такого оппонента писалась до этой миграции. Предел на ник,
    который владелец печатает РУКАМИ, остался в `bot.handlers`.

    `uq_opponents_id_owner` не проверяет ничего сам: это цель составных внешних
    ключей из `opponent_links` и `notes`, а Postgres требует уникальности на
    колонках, на которые ссылается FK.
    """

    __tablename__ = "opponents"
    __table_args__ = (
        Index(
            "uq_opponents_owner_player_id_lower_nick",
            "owner_player_id",
            func.lower(text("opponent_nick")),
            unique=True,
        ),
        UniqueConstraint("id", "owner_player_id", name="uq_opponents_id_owner"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    owner_player_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("players.id"), nullable=False
    )
    opponent_nick: Mapped[str] = mapped_column(String, nullable=False)


class OpponentLink(Base):
    """`opponent_links`: пара «турнир + идентификатор участника» → оппонент.

    **Турнир — тот, которым его нумерует комната** (`CanonicalHand.tournament_id`),
    а не строка `tournaments`: строк на один турнир бывает несколько (тот же
    файл, загруженный в другой вечер), и привязка не имеет права от этого
    раздваиваться. Внешнего ключа на `tournaments` поэтому нет вовсе — удаление
    строки турнира не оставляет висящей ссылки и не теряет привязку.

    **Ключ таблицы — сама уникальность.** «Один идентификатор в одном турнире у
    одного владельца привязан не более чем к одному оппоненту» — это первичный
    ключ, а не проверка в коде: два одновременных вызова `OpponentsRepo.link`
    физически не могут развести одну пару по двум оппонентам
    (`test_two_opponents_at_once_claim_one_participant_and_the_first_keeps_him`).
    Натуральный ключ вместо суррогатного `id` — тот же образец, что
    `calc_cache.key` (пункт 3 модульного докстринга).

    **Владелец в ключе — денормализация**, без которой это правило нельзя
    выразить индексом: владелец известен только через `opponents`, а индекс не
    умеет джойнить. Чтобы денормализованная колонка не разошлась с владельцем
    самого оппонента, ссылка составная — `(opponent_id, owner_player_id)` на
    `opponents(id, owner_player_id)`: строка с чужим владельцем не вставляется
    вовсе (`test_an_opponent_of_another_player_takes_no_bindings`).

    `ON DELETE CASCADE` — по той же логике: привязка без оппонента не значит
    ничего, и переживать его не должна.

    `uq_opponent_links_opponent_tournament` — «у одного оппонента в одном
    турнире один идентификатор»: в турнире у участника ровно один
    идентификатор, поэтому вторая метка того же оппонента в том же турнире
    означала бы, что в статистику одного человека сложены двое, и раздачи
    турнира посчитались бы дважды
    (`test_one_opponent_keeps_one_participant_per_tournament`).
    """

    __tablename__ = "opponent_links"
    __table_args__ = (
        PrimaryKeyConstraint(
            "owner_player_id",
            "room_tournament_id",
            "participant_label",
            name="pk_opponent_links",
        ),
        ForeignKeyConstraint(
            ["opponent_id", "owner_player_id"],
            ["opponents.id", "opponents.owner_player_id"],
            name="fk_opponent_links_opponent_opponents",
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "opponent_id", "room_tournament_id", name="uq_opponent_links_opponent_tournament"
        ),
    )

    owner_player_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    room_tournament_id: Mapped[str] = mapped_column(String, nullable=False)
    participant_label: Mapped[str] = mapped_column(String, nullable=False)
    opponent_id: Mapped[int] = mapped_column(BigInteger, nullable=False)


class EvalCase(Base):
    """Eval-датасет: `eval_cases`. Копится сам — этажи 2 и 4 EVALS.md."""

    __tablename__ = "eval_cases"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('vision_field', 'verdict_confirm', 'verdict_dispute')",
            name="kind_allowed",
        ),
        CheckConstraint(
            "source IN ('escalation', 'disagree_button', 'manual')", name="source_allowed"
        ),
        Index("ix_eval_cases_kind", "kind"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    hand_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("hands.id"), nullable=False)
    field: Mapped[str | None] = mapped_column(String)
    ground_truth: Mapped[Any] = mapped_column(JSONB, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


# Имя партиционного уникального индекса из `Job.__table_args__` — константа, а
# не только строковый литерал в `Index(...)` ниже, потому что `platform/queue.py`
# (fix round 2, Item 2) должен опознать именно ЭТОТ индекс по имени в тексте
# ошибки Postgres, чтобы отличить его от любого другого возможного нарушения
# уникальности на `jobs`. Общий источник строки — единственная защита от того,
# что переименование индекса здесь тихо разойдётся со строкой в `queue.py`.
JOBS_RUNNING_UNIQUE_INDEX = "uq_jobs_player_id_running"


class Job(Base):
    """Очередь + журнал: `jobs` (§8). `session_id NOT NULL` с первой миграции —
    результату всегда есть куда лечь (молчаливое создание сессии — обязанность bot,
    задача 19).
    """

    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(
            "type IN ('screenshot_analyze', 'hh_scan', 'deep_dive', 'eval_run')",
            name="type_allowed",
        ),
        CheckConstraint(
            "status IN ('queued', 'running', 'awaiting_user', 'done', 'failed')",
            name="status_allowed",
        ),
        Index("ix_jobs_status_priority_created_at", "status", "priority", "created_at"),
        Index("ix_jobs_player_id_status", "player_id", "status"),
        # Партиционный уникальный индекс — структурная гарантия «одна активная
        # задача на игрока» (спека §8.1), а не только фильтр NOT EXISTS в SQL
        # захвата (`platform/queue.py`, `_CLAIM_SQL`). NOT EXISTS под READ
        # COMMITTED видит только закоммиченное и не останавливает два конкурентных
        # claim() над РАЗНЫМИ queued-задачами одного игрока — гонка (задача 15,
        # fix round 1, 300/300 на реальном Postgres). Индекс делает эту гонку
        # физически невозможной при любом уровне изоляции и для любого будущего
        # кода, который выставит status='running' в обход claim().
        Index(
            JOBS_RUNNING_UNIQUE_INDEX,
            "player_id",
            unique=True,
            postgresql_where=text("status = 'running'"),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="queued")
    payload: Mapped[Any] = mapped_column(JSONB, nullable=False)
    session_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("sessions.id"), nullable=False
    )
    player_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("players.id"), nullable=False)
    hand_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("hands.id"))
    # Дефолты ниже — параметры устойчивости очереди (задача 15), а не продуктовые
    # числа о деньгах: 0 попыток на старте, 3 — стандартный потолок ретраев джобы.
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="3")
    priority: Mapped[int] = mapped_column(Integer, nullable=False, server_default="100")
    locked_by: Mapped[str | None] = mapped_column(String)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)


class Trace(Base):
    """Что вернул каждый сервис: `traces`. Одна строка на джобу, `spans` копится по ходу."""

    __tablename__ = "traces"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    job_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("jobs.id"), nullable=False)
    hand_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("hands.id"))
    spans: Mapped[Any] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))


class LlmCall(Base):
    """Учёт себестоимости и точка данных лимитера: `llm_calls` (§7).

    `started_at` и nullable-числовые колонки — см. пункты 1 и 2 модульного
    докстринга: строка пишется при старте вызова, до того как токены/цена/латентность
    известны.
    """

    __tablename__ = "llm_calls"
    __table_args__ = (
        CheckConstraint(
            # `vision_extract_fallback` — вторая ступень каскада зрения (задача
            # 22). Отдельное назначение, а не то же самое: по нему считается,
            # сколько раз дешёвого чтения не хватило, и сколько это стоило.
            "purpose IN ('vision_extract', 'vision_extract_fallback', 'verdict_text')",
            name="purpose_allowed",
        ),
        CheckConstraint(
            "status IN ('started', 'ok', 'error', 'schema_error')", name="status_allowed"
        ),
        Index("ix_llm_calls_started_at", "started_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    trace_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("traces.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    purpose: Mapped[str] = mapped_column(String(32), nullable=False)
    tokens_in: Mapped[int | None] = mapped_column(Integer)
    tokens_out: Mapped[int | None] = mapped_column(Integer)
    cost: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="started")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class CalcCache(Base):
    """Кэш расчётов: `calc_cache`. Общий между турнирами и пользователями (§6):
    655х на прогретом кэше, ключ — сигнатура спота, а не конкретной раздачи.
    """

    __tablename__ = "calc_cache"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[Any] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


def async_session_factory(dsn: str) -> async_sessionmaker[AsyncSession]:
    """Фабрика асинхронных сессий на заданном DSN (`postgresql+asyncpg://...`).

    Один вызов на процесс: возвращённая фабрика открывает новое соединение из
    пула движка на каждый вызов `factory()` — то, что нужно и воркеру (задачи
    15/16 гонят конкурентные запросы с разных соединений), и `db_factory` в
    тестах (контроллерский рулинг задачи 14: `FOR UPDATE SKIP LOCKED` и
    `pg_try_advisory_lock` требуют разных соединений, видящих закоммиченное).
    """
    engine = create_async_engine(dsn)
    return async_sessionmaker(engine, expire_on_commit=False)
