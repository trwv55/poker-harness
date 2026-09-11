"""вопрос игрока — свой тип задачи и своё назначение вызова модели

Два CHECK-констрейнта, оба перечисляющие закрытые наборы: `jobs.type_allowed`
не знал про тип `question`, `llm_calls.purpose_allowed` — про назначение
`question_answer`. Без обоих первая же поставленная задача падает нарушением
ограничения (тот же случай, что миграция 0004 на каскаде зрения).

Назначение вызова отдельное, а не `verdict_text`, хотя модель у них одна:
`llm_calls.purpose` — то, по чему считается, сколько стоит каждый вход
продукта, и один ключ на два разных входа сделал бы этот счёт невычислимым.

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-11 12:00:00.000000

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0009'
down_revision: str | Sequence[str] | None = '0008'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_JOB_TYPES_OLD = "type IN ('screenshot_analyze', 'hh_scan', 'deep_dive', 'eval_run')"
_JOB_TYPES_NEW = (
    "type IN ('screenshot_analyze', 'hh_scan', 'deep_dive', 'eval_run', 'question')"
)
_PURPOSE_OLD = "purpose IN ('vision_extract', 'vision_extract_fallback', 'verdict_text')"
_PURPOSE_NEW = (
    "purpose IN ('vision_extract', 'vision_extract_fallback', 'verdict_text', "
    "'question_answer')"
)


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint('type_allowed', 'jobs', type_='check')
    op.create_check_constraint('type_allowed', 'jobs', _JOB_TYPES_NEW)
    op.drop_constraint('purpose_allowed', 'llm_calls', type_='check')
    op.create_check_constraint('purpose_allowed', 'llm_calls', _PURPOSE_NEW)


def downgrade() -> None:
    """Downgrade schema."""
    # Порядок — по внешним ключам: `llm_calls.trace_id` ссылается на `traces`,
    # `traces.job_id` — на `jobs`. Строки, которых старый констрейнт не
    # допускает, удаляются, иначе его создание отвергнет уже накопленное.
    op.execute(
        "DELETE FROM llm_calls WHERE purpose = 'question_answer' OR trace_id IN "
        "(SELECT id FROM traces WHERE job_id IN (SELECT id FROM jobs WHERE type = 'question'))"
    )
    op.execute("DELETE FROM traces WHERE job_id IN (SELECT id FROM jobs WHERE type = 'question')")
    op.execute("DELETE FROM jobs WHERE type = 'question'")
    op.drop_constraint('purpose_allowed', 'llm_calls', type_='check')
    op.create_check_constraint('purpose_allowed', 'llm_calls', _PURPOSE_OLD)
    op.drop_constraint('type_allowed', 'jobs', type_='check')
    op.create_check_constraint('type_allowed', 'jobs', _JOB_TYPES_OLD)
