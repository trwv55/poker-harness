"""llm_calls: назначение вызова второй ступени каскада зрения

Задача 22. Каскад зрения читает экран дешёвой моделью и, если не сошлась
контрольная сумма, перечитывает дорогой. Это ОТДЕЛЬНОЕ назначение вызова, а не
повтор прежнего: по нему считается, сколько раз дешёвого чтения не хватило и во
что это обошлось. CHECK-констрейнт `llm_calls.purpose` про него не знал, и
первый же вызов дорогой ступени падал нарушением ограничения — найдено на
живом прогоне eval-датасета, не рассуждением.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-06 23:55:00.000000

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0004'
down_revision: str | Sequence[str] | None = '0003'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD = "purpose IN ('vision_extract', 'verdict_text')"
_NEW = "purpose IN ('vision_extract', 'vision_extract_fallback', 'verdict_text')"


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint('purpose_allowed', 'llm_calls', type_='check')
    op.create_check_constraint('purpose_allowed', 'llm_calls', _NEW)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('purpose_allowed', 'llm_calls', type_='check')
    op.create_check_constraint('purpose_allowed', 'llm_calls', _OLD)
