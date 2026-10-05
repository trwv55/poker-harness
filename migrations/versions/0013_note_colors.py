"""цвета заметок называет игрок: таблица `note_colors` вместо ключа в `notes.color`

До этой миграции цвет заметки был ключом из набора, зашитого в код (red,
yellow, green, blue и none), один на всех игроков. Теперь у каждого игрока свои
цвета — имя и подпись его словами, — а заметка ссылается на цвет своего
владельца составным внешним ключом `(color_id, owner_player_id)` →
`note_colors(id, player_id)`.

**Перенос вперёд.** Для каждой пары «владелец + ключ» кроме `none` владельцу
заводится один цвет с прежней подписью, и его заметки ссылаются на него:

- red → красный — агрессор;
- yellow → жёлтый — лузовый;
- green → зелёный — слабый;
- blue → синий — тайтовый;
- ключ вне этих четырёх → имя и подпись равны ключу.

`none` и пустая строка → `color_id` NULL.

**Откат теряет данные.** Заметке, чей цвет по `lower(name)` совпадает с одним из
четырёх имён выше, возвращается прежний ключ; любой другой цвет (названный
игроком после этой миграции или перенесённый из ключа вне набора) становится
`none`, а сами цвета и их подписи удаляются вместе с таблицей.

Перенос в обе стороны закреплён
`test_migration_0013_turns_colour_keys_into_colours_of_their_owners`.

Revision ID: 0013
Revises: 0012
Create Date: 2026-10-03 18:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0013'
down_revision: str | Sequence[str] | None = '0012'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Ключ → имя и подпись; последнее число — порядок, в котором цвета заводятся.
_KEYS = """
(VALUES ('red', 'красный', 'агрессор', 1),
        ('yellow', 'жёлтый', 'лузовый', 2),
        ('green', 'зелёный', 'слабый', 3),
        ('blue', 'синий', 'тайтовый', 4)) AS k(key, name, meaning, pos)
"""

_CREATE_COLOURS = f"""
INSERT INTO note_colors (player_id, name, meaning, created_at)
SELECT owner_player_id, name, meaning, now()
  FROM (SELECT DISTINCT n.owner_player_id,
               coalesce(k.name, n.color) AS name,
               coalesce(k.meaning, n.color) AS meaning,
               coalesce(k.pos, 5) AS pos
          FROM notes n
          LEFT JOIN {_KEYS} ON k.key = n.color
         WHERE n.color NOT IN ('none', '')) c
 ORDER BY owner_player_id, pos, name
ON CONFLICT (player_id, lower(name)) DO NOTHING
"""

_POINT_NOTES_AT_COLOURS = f"""
UPDATE notes n
   SET color_id = c.id
  FROM note_colors c
 WHERE c.player_id = n.owner_player_id
   AND n.color NOT IN ('none', '')
   AND lower(c.name) = lower(coalesce(
         (SELECT k.name FROM {_KEYS} WHERE k.key = n.color), n.color))
"""

_RESTORE_KEYS = f"""
UPDATE notes n
   SET color = k.key
  FROM note_colors c
  JOIN {_KEYS} ON k.name = lower(c.name)
 WHERE c.id = n.color_id
"""


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'note_colors',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('player_id', sa.BigInteger(), nullable=False),
        sa.Column('name', sa.String(length=32), nullable=False),
        sa.Column('meaning', sa.String(length=120), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ['player_id'], ['players.id'], name=op.f('fk_note_colors_player_id_players')
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_note_colors')),
        sa.UniqueConstraint('id', 'player_id', name='uq_note_colors_id_player'),
    )
    op.create_index(
        'uq_note_colors_player_id_lower_name',
        'note_colors',
        ['player_id', sa.literal_column('lower(name)')],
        unique=True,
    )
    op.add_column('notes', sa.Column('color_id', sa.BigInteger(), nullable=True))
    op.create_foreign_key(
        'fk_notes_color_note_colors',
        'notes',
        'note_colors',
        ['color_id', 'owner_player_id'],
        ['id', 'player_id'],
    )
    op.execute(_CREATE_COLOURS)
    op.execute(_POINT_NOTES_AT_COLOURS)
    op.drop_column('notes', 'color')


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        'notes',
        sa.Column('color', sa.String(length=32), nullable=False, server_default='none'),
    )
    op.alter_column('notes', 'color', existing_type=sa.String(length=32), server_default=None)
    op.execute(_RESTORE_KEYS)
    op.drop_constraint('fk_notes_color_note_colors', 'notes', type_='foreignkey')
    op.drop_column('notes', 'color_id')
    op.drop_index('uq_note_colors_player_id_lower_name', table_name='note_colors')
    op.drop_table('note_colors')
