"""оппонент один: заметка и сшивки турниров ссылаются на одну строку

До этой миграции личность оппонента жила в двух таблицах и они уже разошлись:
`notes` держала ник строкой и РАЗЛИЧАЛА регистр (уникальный индекс 0005 по
`(owner_player_id, opponent_nick)`), `player_aliases` — НЕ различала
(функциональный индекс по `lower(opponent_nick)`). Один и тот же ник в разном
написании был одним оппонентом для статистики и двумя для заметок.

1. `player_aliases` → `opponents`, `player_alias_links` → `opponent_links`
   (колонка `alias_id` → `opponent_id`). Имя `player_alias_links` называло
   таблицу, которой после переименования нет.

2. Длина `opponents.opponent_nick` снята. Колонка повторяла
   `players.gg_nickname` (64), но ник оппонента приходит не с клавиатуры, а со
   стола, и `notes.opponent_nick` до этой миграции длину не ограничивала вовсе:
   заметки на ники длиннее 64 писались. Обратное сужение в `downgrade` откажет
   ошибкой Postgres, если такой ник записан, — обрезать его молча нельзя.

3. `notes.opponent_id` вместо `notes.opponent_nick`; уникален `opponent_id`
   (одна заметка на оппонента), владелец сверяется составным внешним ключом на
   `opponents(id, owner_player_id)`.

**Правило схлопывания.** Если у владельца оказалось несколько заметок на один
ник в разном регистре, они сливаются в ОДНУ строку:

- выживает самая свежая (`updated_at`, при равенстве — больший `id`); её `id`
  остаётся, поэтому кнопки уже отправленных сообщений ведут к ней;
- текст выжившей — тексты ВСЕХ схлопнутых строк, склеенные переводом строки в
  хронологическом порядке (`updated_at`, затем `id`). Ни один символ не
  отбрасывается: выбрать между наблюдениями владельца не на чем;
- цвет — от выжившей строки; цвет остальных не переносится (цвет — одна метка
  из набора, склеить их нельзя);
- написание ника — то, что уже стоит в `opponents`, а если оппонента ещё нет,
  то из самой ранней заметки: `OpponentsRepo.get_or_create` хранит первое
  написание, и миграция не имеет права заводить второе правило.

Правило закреплено `test_migration_0007_collapses_notes_of_one_nick_in_two_cases`.
Склейка может дать текст длиннее `MAX_NOTE_TEXT_CHARS`: предел стоит на том,
что игрок пишет за один раз, а правка такой заметки вернёт её под предел.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-08 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0007'
down_revision: str | Sequence[str] | None = '0006'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Оппонент для каждого ника, на который есть заметка, но нет строки в
# `opponents`. `DISTINCT ON (lower(...))` схлопывает регистр, а порядок внутри
# группы отдаёт написание САМОЙ РАННЕЙ заметки.
_CREATE_MISSING_OPPONENTS = """
INSERT INTO opponents (owner_player_id, opponent_nick)
SELECT DISTINCT ON (n.owner_player_id, lower(n.opponent_nick))
       n.owner_player_id, n.opponent_nick
  FROM notes n
 WHERE NOT EXISTS (
       SELECT 1 FROM opponents o
        WHERE o.owner_player_id = n.owner_player_id
          AND lower(o.opponent_nick) = lower(n.opponent_nick))
 ORDER BY n.owner_player_id, lower(n.opponent_nick), n.updated_at, n.id
"""

_POINT_NOTES_AT_OPPONENTS = """
UPDATE notes n
   SET opponent_id = o.id
  FROM opponents o
 WHERE o.owner_player_id = n.owner_player_id
   AND lower(o.opponent_nick) = lower(n.opponent_nick)
"""

# Тексты всех заметок оппонента — в выжившую строку. Считается ДО удаления,
# иначе склеивать было бы уже нечего.
_GLUE_TEXTS = """
UPDATE notes n
   SET text = m.glued
  FROM (SELECT opponent_id,
               (array_agg(id ORDER BY updated_at DESC, id DESC))[1] AS keep_id,
               string_agg(text, chr(10) ORDER BY updated_at, id) AS glued
          FROM notes
         GROUP BY opponent_id
        HAVING count(*) > 1) m
 WHERE n.id = m.keep_id
"""

_DROP_MERGED_NOTES = """
DELETE FROM notes n
 USING (SELECT opponent_id,
               (array_agg(id ORDER BY updated_at DESC, id DESC))[1] AS keep_id
          FROM notes
         GROUP BY opponent_id
        HAVING count(*) > 1) m
 WHERE n.opponent_id = m.opponent_id
   AND n.id <> m.keep_id
"""

_RESTORE_NICKS = """
UPDATE notes n
   SET opponent_nick = o.opponent_nick
  FROM opponents o
 WHERE o.id = n.opponent_id
"""


def upgrade() -> None:
    """Upgrade schema."""
    op.rename_table('player_aliases', 'opponents')
    op.execute(
        'ALTER INDEX uq_player_aliases_owner_player_id_lower_nick '
        'RENAME TO uq_opponents_owner_player_id_lower_nick'
    )
    op.execute('ALTER TABLE opponents RENAME CONSTRAINT pk_player_aliases TO pk_opponents')
    op.execute(
        'ALTER TABLE opponents RENAME CONSTRAINT uq_player_aliases_id_owner '
        'TO uq_opponents_id_owner'
    )
    op.execute(
        'ALTER TABLE opponents RENAME CONSTRAINT fk_player_aliases_owner_player_id_players '
        'TO fk_opponents_owner_player_id_players'
    )
    op.alter_column(
        'opponents',
        'opponent_nick',
        existing_type=sa.String(length=64),
        type_=sa.String(),
        existing_nullable=False,
    )

    op.rename_table('player_alias_links', 'opponent_links')
    op.alter_column('opponent_links', 'alias_id', new_column_name='opponent_id')
    op.execute(
        'ALTER TABLE opponent_links RENAME CONSTRAINT pk_player_alias_links '
        'TO pk_opponent_links'
    )
    op.execute(
        'ALTER TABLE opponent_links RENAME CONSTRAINT uq_player_alias_links_alias_tournament '
        'TO uq_opponent_links_opponent_tournament'
    )
    op.execute(
        'ALTER TABLE opponent_links RENAME CONSTRAINT fk_player_alias_links_alias_player_aliases '
        'TO fk_opponent_links_opponent_opponents'
    )

    op.add_column('notes', sa.Column('opponent_id', sa.BigInteger(), nullable=True))
    op.execute(_CREATE_MISSING_OPPONENTS)
    op.execute(_POINT_NOTES_AT_OPPONENTS)
    op.execute(_GLUE_TEXTS)
    op.execute(_DROP_MERGED_NOTES)
    op.alter_column('notes', 'opponent_id', existing_type=sa.BigInteger(), nullable=False)
    op.create_foreign_key(
        'fk_notes_opponent_opponents',
        'notes',
        'opponents',
        ['opponent_id', 'owner_player_id'],
        ['id', 'owner_player_id'],
    )
    op.create_index('uq_notes_opponent_id', 'notes', ['opponent_id'], unique=True)
    op.drop_index('uq_notes_owner_player_id_opponent_nick', table_name='notes')
    op.drop_column('notes', 'opponent_nick')


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column('notes', sa.Column('opponent_nick', sa.String(), nullable=True))
    op.execute(_RESTORE_NICKS)
    op.alter_column('notes', 'opponent_nick', existing_type=sa.String(), nullable=False)
    op.create_index(
        'uq_notes_owner_player_id_opponent_nick',
        'notes',
        ['owner_player_id', 'opponent_nick'],
        unique=True,
    )
    op.drop_index('uq_notes_opponent_id', table_name='notes')
    op.drop_constraint('fk_notes_opponent_opponents', 'notes', type_='foreignkey')
    op.drop_column('notes', 'opponent_id')

    op.execute(
        'ALTER TABLE opponent_links RENAME CONSTRAINT fk_opponent_links_opponent_opponents '
        'TO fk_player_alias_links_alias_player_aliases'
    )
    op.execute(
        'ALTER TABLE opponent_links RENAME CONSTRAINT uq_opponent_links_opponent_tournament '
        'TO uq_player_alias_links_alias_tournament'
    )
    op.execute(
        'ALTER TABLE opponent_links RENAME CONSTRAINT pk_opponent_links '
        'TO pk_player_alias_links'
    )
    op.alter_column('opponent_links', 'opponent_id', new_column_name='alias_id')
    op.rename_table('opponent_links', 'player_alias_links')

    op.alter_column(
        'opponents',
        'opponent_nick',
        existing_type=sa.String(),
        type_=sa.String(length=64),
        existing_nullable=False,
    )
    op.execute(
        'ALTER TABLE opponents RENAME CONSTRAINT fk_opponents_owner_player_id_players '
        'TO fk_player_aliases_owner_player_id_players'
    )
    op.execute(
        'ALTER TABLE opponents RENAME CONSTRAINT uq_opponents_id_owner '
        'TO uq_player_aliases_id_owner'
    )
    op.execute('ALTER TABLE opponents RENAME CONSTRAINT pk_opponents TO pk_player_aliases')
    op.execute(
        'ALTER INDEX uq_opponents_owner_player_id_lower_nick '
        'RENAME TO uq_player_aliases_owner_player_id_lower_nick'
    )
    op.rename_table('opponents', 'player_aliases')
