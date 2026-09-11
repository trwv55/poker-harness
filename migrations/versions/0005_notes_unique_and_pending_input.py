"""players: незакрытый ввод бота; notes: одна заметка на оппонента

Задача 23. Две правки одной миграцией, потому что обе — про экраны бота.

1. `players.pending_input` — что означает СЛЕДУЮЩЕЕ текстовое сообщение игрока
   (ник в руме, текст заметки). До этой задачи роль первого текста была
   зашита в код («первый текст = ник»), и любое случайное сообщение молча
   становилось ником. Теперь ввод открывается явной кнопкой, а состояние живёт
   в БД, а не в памяти процесса бота — по той же причине, что и состояние
   эскалации в `jobs.payload` (спека §8.3): перезапуск бота не имеет права
   терять половину диалога. Колонка nullable: NULL = бот ничего не ждёт.

2. Уникальность `notes(owner_player_id, opponent_nick)` — заметка на оппонента
   одна и накапливается (SESSIONS_UX: «оппонент встречается в разных сессиях,
   заметка должна накапливаться»). Без индекса повторное «добавить заметку» на
   того же ника завело бы вторую строку, и экран «Заметки» показал бы одного
   игрока дважды. Индекс же делает возможным `ON CONFLICT DO UPDATE` в
   `NotesRepo.upsert`.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-07 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0005'
down_revision: str | Sequence[str] | None = '0004'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'players',
        sa.Column('pending_input', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index(
        'uq_notes_owner_player_id_opponent_nick',
        'notes',
        ['owner_player_id', 'opponent_nick'],
        unique=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('uq_notes_owner_player_id_opponent_nick', table_name='notes')
    op.drop_column('players', 'pending_input')
