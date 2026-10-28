"""агент: история вечера, указатель разбора, тип задачи и назначение вызова

Сессия получает историю диалога с агентом (`agent_history`, сериализованные
сообщения pydantic-ai) и указатель на последний разбор вечера
(`last_analysis_id`). `jobs.type` и `llm_calls.purpose` получают `agent` —
ДОБАВЛЕНИЕМ: `question`/`question_answer` остаются, пока `/ask` жив у игроков
без доступа к агенту (спека агента №1, §10). `eval_cases` принимает строку без
руки — «не то посчитано» на вопрос без разбора.

Откат удаляет строки, которых старые CHECK не примут (тот же приём, что 0009).
Вперёд и назад закреплено
`test_migration_0014_adds_the_agent_and_rolls_back_without_its_rows`.

Revision ID: 0014
Revises: 0013
Create Date: 2026-10-11 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0014'
down_revision: str | Sequence[str] | None = '0013'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_JOB_TYPES_OLD = "type IN ('screenshot_analyze', 'hh_scan', 'deep_dive', 'eval_run', 'question')"
_JOB_TYPES_NEW = (
    "type IN ('screenshot_analyze', 'hh_scan', 'deep_dive', 'eval_run', 'question', 'agent')"
)
_PURPOSE_OLD = (
    "purpose IN ('vision_extract', 'vision_extract_fallback', 'verdict_text', "
    "'tournament_text', 'question_answer')"
)
_PURPOSE_NEW = (
    "purpose IN ('vision_extract', 'vision_extract_fallback', 'verdict_text', "
    "'tournament_text', 'question_answer', 'agent')"
)
_KIND_OLD = "kind IN ('vision_field', 'verdict_confirm', 'verdict_dispute')"
_KIND_NEW = "kind IN ('vision_field', 'verdict_confirm', 'verdict_dispute', 'agent_routing')"
_SOURCE_OLD = "source IN ('escalation', 'disagree_button', 'manual')"
_SOURCE_NEW = "source IN ('escalation', 'disagree_button', 'manual', 'agent_miss_button')"


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'sessions',
        sa.Column('agent_history', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column('sessions', sa.Column('last_analysis_id', sa.BigInteger(), nullable=True))
    op.create_foreign_key(
        op.f('fk_sessions_last_analysis_id_analyses'),
        'sessions',
        'analyses',
        ['last_analysis_id'],
        ['id'],
        ondelete='SET NULL',
    )
    op.drop_constraint('type_allowed', 'jobs', type_='check')
    op.create_check_constraint('type_allowed', 'jobs', _JOB_TYPES_NEW)
    op.drop_constraint('purpose_allowed', 'llm_calls', type_='check')
    op.create_check_constraint('purpose_allowed', 'llm_calls', _PURPOSE_NEW)
    op.drop_constraint('kind_allowed', 'eval_cases', type_='check')
    op.create_check_constraint('kind_allowed', 'eval_cases', _KIND_NEW)
    op.drop_constraint('source_allowed', 'eval_cases', type_='check')
    op.create_check_constraint('source_allowed', 'eval_cases', _SOURCE_NEW)
    op.alter_column('eval_cases', 'hand_id', existing_type=sa.BigInteger(), nullable=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(
        "DELETE FROM eval_cases WHERE hand_id IS NULL OR kind = 'agent_routing' "
        "OR source = 'agent_miss_button'"
    )
    op.alter_column('eval_cases', 'hand_id', existing_type=sa.BigInteger(), nullable=False)
    op.drop_constraint('source_allowed', 'eval_cases', type_='check')
    op.create_check_constraint('source_allowed', 'eval_cases', _SOURCE_OLD)
    op.drop_constraint('kind_allowed', 'eval_cases', type_='check')
    op.create_check_constraint('kind_allowed', 'eval_cases', _KIND_OLD)
    # Порядок — по внешним ключам, как в 0009: `llm_calls.trace_id` → `traces`,
    # `traces.job_id` → `jobs`.
    op.execute(
        "DELETE FROM llm_calls WHERE purpose = 'agent' OR trace_id IN "
        "(SELECT id FROM traces WHERE job_id IN (SELECT id FROM jobs WHERE type = 'agent'))"
    )
    op.execute("DELETE FROM traces WHERE job_id IN (SELECT id FROM jobs WHERE type = 'agent')")
    op.execute("DELETE FROM jobs WHERE type = 'agent'")
    op.drop_constraint('purpose_allowed', 'llm_calls', type_='check')
    op.create_check_constraint('purpose_allowed', 'llm_calls', _PURPOSE_OLD)
    op.drop_constraint('type_allowed', 'jobs', type_='check')
    op.create_check_constraint('type_allowed', 'jobs', _JOB_TYPES_OLD)
    op.drop_constraint(
        op.f('fk_sessions_last_analysis_id_analyses'), 'sessions', type_='foreignkey'
    )
    op.drop_column('sessions', 'last_analysis_id')
    op.drop_column('sessions', 'agent_history')
