"""Fence current-hand publication across image intake, spots and /new.

Revision ID: 0018
Revises: 0017
"""
import sqlalchemy as sa
from alembic import op

revision = '0018'
down_revision = '0017'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('sessions', sa.Column('context_revision', sa.Integer(), nullable=False, server_default='0'))
    op.create_index('ix_hands_image_hash', 'hands', ['image_hash'])


def downgrade() -> None:
    op.drop_index('ix_hands_image_hash', table_name='hands')
    op.drop_column('sessions', 'context_revision')
