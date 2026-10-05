"""кто обслужил вызов — отдельной колонкой от того, куда его послали

`llm_calls.provider` хранит разобранное из строки конфига: для `openrouter:qwen/…`
там лежит `openrouter`. Но у одной модели на OpenRouter несколько хостеров, и они
различаются тем, что нам критично: `seed` поддерживают не все, а
`tool_choice: required`, без которого у нас нет структурированного вывода, — не
все из оставшихся (замер 2026-09-12 на `qwen3-vl-235b`: пять эндпоинтов, один без
seed, один без `required`). Пока обслуживший не записан, разбор расхождений между
прогонами упирается в «неизвестно, кто отвечал».

Колонка, а не составное значение в `provider`: «куда слали» и «кто ответил» —
разные сущности, и одно поле на две уже один раз вышло дорого (поля анте, где схема
просила различить пул и подушевое, а на экране одно число).

Nullable и без умолчания: прямой вызов вендора хостера не называет вовсе, и
отсутствие имени — это факт, а не пропуск. Накопленные строки не переписываются:
кто обслуживал прошлые вызовы, в таблице не записано, и подставить туда что-либо
значило бы угадать.

Имя берётся из `provider_details.downstream_provider` ответа (проверено живым
вызовом), длина 64 — с запасом к самым длинным именам хостеров.

Revision ID: 0011
Revises: 0010
Create Date: 2026-09-12 17:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0011'
down_revision: str | Sequence[str] | None = '0010'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("llm_calls", sa.Column("served_by", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("llm_calls", "served_by")
