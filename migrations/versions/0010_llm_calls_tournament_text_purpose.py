"""рассказ по турниру — своё назначение вызова модели

`llm_calls.purpose` — то, по чему считается себестоимость каждого входа
продукта. Рассказ по турниру (`explanation/tournament_text.py`) писался в базу
под ключом `verdict_text`, то есть в одну строку со стоимостью разбора руки:
пока это так, цену скана и цену разбора из таблицы не разделить. Назначение
отдельное по той же причине и тем же способом, что `question_answer`
(миграция 0009), хотя модель у рассказа и у вердикта одна.

Уже накопленные строки не переписываются: какие из них были рассказом, а какие
вердиктом, в таблице не записано — восстановить это значило бы угадать. Ключ
меняется только для вызовов после наката.

Revision ID: 0010
Revises: 0009
Create Date: 2026-09-11 14:00:00.000000

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0010'
down_revision: str | Sequence[str] | None = '0009'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PURPOSE_OLD = (
    "purpose IN ('vision_extract', 'vision_extract_fallback', 'verdict_text', "
    "'question_answer')"
)
_PURPOSE_NEW = (
    "purpose IN ('vision_extract', 'vision_extract_fallback', 'verdict_text', "
    "'tournament_text', 'question_answer')"
)


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint('purpose_allowed', 'llm_calls', type_='check')
    op.create_check_constraint('purpose_allowed', 'llm_calls', _PURPOSE_NEW)


def downgrade() -> None:
    """Downgrade schema."""
    # Строки нового назначения удаляются, иначе старый CHECK не создастся на уже
    # накопленном (тот же порядок действий, что в 0009). Внешних ключей НА
    # `llm_calls` нет — удаление ограничено одной таблицей, трейсы и задачи
    # скана остаются: рассказ был их частью, а не отдельной задачей.
    op.execute("DELETE FROM llm_calls WHERE purpose = 'tournament_text'")
    op.drop_constraint('purpose_allowed', 'llm_calls', type_='check')
    op.create_check_constraint('purpose_allowed', 'llm_calls', _PURPOSE_OLD)
