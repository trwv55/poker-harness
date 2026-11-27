"""Temporary Pokerdom jobs need no gaming session or saved hand.

Revision ID: 0017
Revises: 0016
"""
import sqlalchemy as sa
from alembic import op

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None

_OLD = "type IN ('screenshot_analyze', 'hh_scan', 'deep_dive', 'eval_run', 'question', 'agent')"
_NEW = "type IN ('screenshot_analyze', 'hh_scan', 'deep_dive', 'eval_run', 'question', 'agent', 'pokerdom_read')"


def upgrade() -> None:
    op.add_column("players", sa.Column("pokerdom_nickname", sa.String(64), nullable=True))
    op.drop_constraint('type_allowed', 'jobs', type_='check')
    op.create_check_constraint('type_allowed', 'jobs', _NEW)
    op.alter_column('jobs', 'session_id', nullable=True)
    op.create_check_constraint('session_required', 'jobs', "session_id IS NOT NULL OR type = 'pokerdom_read'")


def downgrade() -> None:
    # Traces reference jobs; follow existing migration downgrade policy explicitly.
    op.execute("DELETE FROM llm_calls WHERE trace_id IN (SELECT id FROM traces WHERE job_id IN (SELECT id FROM jobs WHERE type='pokerdom_read'))")
    op.execute("DELETE FROM traces WHERE job_id IN (SELECT id FROM jobs WHERE type='pokerdom_read')")
    op.execute("DELETE FROM jobs WHERE type='pokerdom_read'")
    op.drop_constraint('session_required', 'jobs', type_='check')
    op.alter_column('jobs', 'session_id', nullable=False)
    op.drop_constraint('type_allowed', 'jobs', type_='check')
    op.create_check_constraint('type_allowed', 'jobs', _OLD)
    op.drop_column("players", "pokerdom_nickname")
