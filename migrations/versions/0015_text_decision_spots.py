"""Separate text decision spots and the active session context.

Revision ID: 0015
Revises: 0014
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0015"
down_revision: str | Sequence[str] | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "decision_spots",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("session_id", sa.BigInteger(), sa.ForeignKey("sessions.id"), nullable=False),
        sa.Column("player_id", sa.BigInteger(), sa.ForeignKey("players.id"), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("source_text", sa.Text(), nullable=False),
        sa.Column("raw", postgresql.JSONB(), nullable=False),
        sa.Column("canonical", postgresql.JSONB(), nullable=True),
        sa.Column("snapshot", postgresql.JSONB(), nullable=True),
        sa.CheckConstraint("status IN ('draft', 'verified')", name="status_allowed"),
    )
    op.create_index("ix_decision_spots_session_id", "decision_spots", ["session_id"])
    op.add_column("sessions", sa.Column("active_spot_id", sa.BigInteger(), nullable=True))
    op.create_foreign_key(
        "fk_sessions_active_spot_id_decision_spots",
        "sessions",
        "decision_spots",
        ["active_spot_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_sessions_active_spot_id_decision_spots", "sessions", type_="foreignkey")
    op.drop_column("sessions", "active_spot_id")
    op.drop_index("ix_decision_spots_session_id", table_name="decision_spots")
    op.drop_table("decision_spots")
