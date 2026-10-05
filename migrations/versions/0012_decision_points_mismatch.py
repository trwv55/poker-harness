"""расхождение без цены — колонкой рядом с судимостью

Точка открытия по чарту (спот `open_chart`, спека 2026-10-03-open-chart-verdict)
цены не имеет: её `ev_diff_bb` — заглушка, а расхождение названо ядром явно
(`PointVerdict.mismatch`). Колонка нужна «Моим ликам»: точка по чарту, сыгранная в
пределах чарта, лика не даёт, и SQL обязан это видеть, не разбирая `detail`.

Nullable и без умолчания: у ценовых точек расхождение выводится из цены, и
NULL — это «не названо», а не «нет». Накопленные строки не переписываются: точек
по чарту до этой ревизии не было.

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-03 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0012'
down_revision: str | Sequence[str] | None = '0011'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("decision_points", sa.Column("mismatch", sa.Boolean(), nullable=True))


def downgrade() -> None:
    op.drop_column("decision_points", "mismatch")
