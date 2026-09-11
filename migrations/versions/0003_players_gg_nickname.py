"""players: ник в руме — вход опознания героя на скриншоте

Задача 22. Героя на скрине определяет КОД: прочитанные моделью ники
сопоставляются по префиксу с ником из профиля игрока, и ноль или больше одного
совпадений уходят вопросом игроку. Модели этот вопрос не задают вовсе —
измерено, что она путает героя с победителем раздачи, а ни одна контрольная
сумма (банк, кнопка, эквити) от того, кого назвали героем, не зависит.

Колонка nullable: у игроков, заведённых до этой миграции, ника нет, и разбор
скрина у них упрётся в вопрос — это верное поведение, а не деградация.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-06 23:40:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0003'
down_revision: str | Sequence[str] | None = '0002'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('players', sa.Column('gg_nickname', sa.String(length=64), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('players', 'gg_nickname')
