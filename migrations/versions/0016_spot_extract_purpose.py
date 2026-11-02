"""Track text spot extraction separately from agent replies.

Revision ID: 0016
Revises: 0015
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0016"
down_revision: str | Sequence[str] | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD = (
    "purpose IN ('vision_extract', 'vision_extract_fallback', 'verdict_text', "
    "'tournament_text', 'question_answer', 'agent')"
)
_NEW = (
    "purpose IN ('vision_extract', 'vision_extract_fallback', 'verdict_text', "
    "'tournament_text', 'question_answer', 'agent', 'spot_extract')"
)


def upgrade() -> None:
    op.drop_constraint("purpose_allowed", "llm_calls", type_="check")
    op.create_check_constraint("purpose_allowed", "llm_calls", _NEW)


def downgrade() -> None:
    op.execute("DELETE FROM llm_calls WHERE purpose = 'spot_extract'")
    op.drop_constraint("purpose_allowed", "llm_calls", type_="check")
    op.create_check_constraint("purpose_allowed", "llm_calls", _OLD)
