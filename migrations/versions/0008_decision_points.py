"""точка решения — строка таблицы, а не элемент массива в analyses.result

До этой миграции все сквозные счётчики (лики по типам, покрытие, цена вечера)
разворачивали `analyses.result -> 'points'` боковым соединением на каждом
запросе, и каждое новое условие писалось выражением по jsonb руками. Теперь
точки лежат строками, а `analyses.result` хранит документ БЕЗ них: копии
вердикта в базе не остаётся — `AnalysesRepo` собирает документ обратно из строк.

**Переливка.** Каждая точка каждого сохранённого разбора становится строкой;
порядок в массиве сохраняется в `point_no` (`AnalysisResult.ranked` индексирует
именно массив). Обстановка точки — позиция, банк до решения, доплата,
эффективный стек, SPR — берётся из `hands.enriched` по совпадению `dp_index`
с `index` точки решения движка; там, где руки нет до чекпоинта `enriched` или
такой точки в ней нет, колонки остаются пустыми, а не заполняются догадкой.
Проверено `test_migration_0008_moves_points_without_changing_a_single_number`:
агрегаты, посчитанные по jsonb ДО миграции, совпадают с посчитанными по
таблице ПОСЛЕ.

**Признак судимости** (`judged`) в переливке выражен SQL-условием — тем самым,
которое эта задача убирает из живого кода. Противоречия нет: миграция — снимок
одного момента, а не второй постоянный источник правила. После неё
единственная формулировка — `contracts.is_judged`, и она же пишет колонку у
всех новых точек.

**Откат** восстанавливает массив в `analyses.result` из тех же строк и роняет
таблицу. Восстановленный документ побайтово равен исходному
(`test_migration_0008_downgrade_gives_the_document_back_as_it_was`); колонки
обстановки и `judged` в него не возвращаются — их в `PointVerdict` нет.

Составные внешние ключи на `hands(id, session_id)` и `sessions(id, player_id)`
требуют уникальности в этих парах — отсюда два `UNIQUE`, добавленные здесь же.
Смысл тот же, что у `uq_opponents_id_owner`: денормализованная колонка не
проверяется джойном, поэтому её проверяет ключ.

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-09 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0008'
down_revision: str | Sequence[str] | None = '0007'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# `WITH ORDINALITY` даёт место элемента в массиве — оно и становится `point_no`.
# Обстановка точки подтягивается боковым LEFT JOIN: у руки, не дошедшей до
# чекпоинта `enriched`, его нет вовсе, и строка всё равно обязана появиться.
_FILL = """
INSERT INTO decision_points (
    hand_id, session_id, player_id, point_no, dp_index, street, spot, zone,
    action_taken, best_action, ev_diff_bb, judged,
    position, to_call, pot_before, eff_stack, eff_stack_bb, spr,
    ev_interval, assumption, tools, detail)
SELECT a.hand_id,
       h.session_id,
       s.player_id,
       (t.ord - 1)::int,
       (t.p ->> 'dp_index')::int,
       t.p ->> 'street',
       t.p ->> 'spot',
       t.p ->> 'zone',
       t.p ->> 'action_taken',
       t.p ->> 'best_action',
       (t.p ->> 'ev_diff_bb')::double precision,
       t.p ->> 'spot' IN ('pushfold_unopened', 'pushfold_facing_shove')
           AND t.p ->> 'best_action' <> '',
       d.p ->> 'position',
       (d.p ->> 'to_call')::bigint,
       (d.p ->> 'pot_before')::bigint,
       (d.p ->> 'eff_stack')::bigint,
       (d.p ->> 'eff_stack_bb')::double precision,
       (d.p ->> 'spr')::double precision,
       CASE WHEN t.p -> 'interval' = 'null'::jsonb THEN NULL ELSE t.p -> 'interval' END,
       CASE WHEN t.p -> 'assumption' = 'null'::jsonb THEN NULL ELSE t.p -> 'assumption' END,
       COALESCE(t.p -> 'tools', '[]'::jsonb),
       COALESCE(t.p -> 'detail', '{}'::jsonb)
  FROM analyses a
  JOIN hands h ON h.id = a.hand_id
  JOIN sessions s ON s.id = h.session_id
 CROSS JOIN LATERAL
       jsonb_array_elements(COALESCE(a.result -> 'points', '[]'::jsonb))
       WITH ORDINALITY AS t(p, ord)
  LEFT JOIN LATERAL (
       SELECT e AS p
         FROM jsonb_array_elements(
              COALESCE(h.enriched -> 'report' -> 'decision_points', '[]'::jsonb)) AS e
        WHERE (e ->> 'index')::int = (t.p ->> 'dp_index')::int
        LIMIT 1) d ON true
 ORDER BY a.hand_id, t.ord
"""

_DROP_POINTS_FROM_DOCUMENT = "UPDATE analyses SET result = result - 'points'"

# Обратная сборка: ключи и их порядок — поля `PointVerdict`; `NULL` попадает в
# документ как `null`, ровно так же, как его пишет `model_dump(mode="json")`.
_RESTORE_POINTS = """
UPDATE analyses a
   SET result = a.result || jsonb_build_object('points', COALESCE(m.points, '[]'::jsonb))
  FROM (SELECT an.id AS analysis_id,
               jsonb_agg(jsonb_build_object(
                   'dp_index', p.dp_index,
                   'street', p.street,
                   'spot', p.spot,
                   'zone', p.zone,
                   'action_taken', p.action_taken,
                   'best_action', p.best_action,
                   'ev_diff_bb', p.ev_diff_bb,
                   'interval', p.ev_interval,
                   'assumption', p.assumption,
                   'tools', p.tools,
                   'detail', p.detail) ORDER BY p.point_no) AS points
          FROM analyses an
          LEFT JOIN decision_points p ON p.hand_id = an.hand_id
         GROUP BY an.id) m
 WHERE a.id = m.analysis_id
"""


def upgrade() -> None:
    """Upgrade schema."""
    op.create_unique_constraint('uq_hands_id_session', 'hands', ['id', 'session_id'])
    op.create_unique_constraint('uq_sessions_id_player', 'sessions', ['id', 'player_id'])
    op.create_table(
        'decision_points',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('hand_id', sa.BigInteger(), nullable=False),
        sa.Column('session_id', sa.BigInteger(), nullable=False),
        sa.Column('player_id', sa.BigInteger(), nullable=False),
        sa.Column('point_no', sa.Integer(), nullable=False),
        sa.Column('dp_index', sa.Integer(), nullable=False),
        sa.Column('street', sa.String(length=16), nullable=False),
        sa.Column('spot', sa.String(length=32), nullable=False),
        sa.Column('zone', sa.String(length=16), nullable=False),
        sa.Column('action_taken', sa.String(), nullable=False),
        sa.Column('best_action', sa.String(), nullable=False),
        sa.Column('ev_diff_bb', sa.Double(), nullable=False),
        sa.Column('judged', sa.Boolean(), nullable=False),
        sa.Column('position', sa.String(length=16), nullable=True),
        sa.Column('to_call', sa.BigInteger(), nullable=True),
        sa.Column('pot_before', sa.BigInteger(), nullable=True),
        sa.Column('eff_stack', sa.BigInteger(), nullable=True),
        sa.Column('eff_stack_bb', sa.Double(), nullable=True),
        sa.Column('spr', sa.Double(), nullable=True),
        sa.Column('ev_interval', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('assumption', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            'tools',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            'detail',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['hand_id', 'session_id'],
            ['hands.id', 'hands.session_id'],
            name='fk_decision_points_hand_id_hands',
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['session_id', 'player_id'],
            ['sessions.id', 'sessions.player_id'],
            name='fk_decision_points_session_id_sessions',
        ),
        sa.PrimaryKeyConstraint('id', name='pk_decision_points'),
        sa.UniqueConstraint('hand_id', 'point_no', name='uq_decision_points_hand_id_point_no'),
    )
    op.create_index('ix_decision_points_player_id', 'decision_points', ['player_id'])
    op.create_index('ix_decision_points_session_id', 'decision_points', ['session_id'])

    op.execute(_FILL)
    op.execute(_DROP_POINTS_FROM_DOCUMENT)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(_RESTORE_POINTS)
    op.drop_index('ix_decision_points_session_id', table_name='decision_points')
    op.drop_index('ix_decision_points_player_id', table_name='decision_points')
    op.drop_table('decision_points')
    op.drop_constraint('uq_sessions_id_player', 'sessions', type_='unique')
    op.drop_constraint('uq_hands_id_session', 'hands', type_='unique')
