"""add_build_queue_and_worker_health

Revision ID: 013
Revises: 012
Create Date: 2026-02-21 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add build queue and recovery fields to strategy_builds
    op.add_column('strategy_builds', sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('strategy_builds', sa.Column('last_checkpoint', sa.JSON(), nullable=True))
    op.add_column('strategy_builds', sa.Column('assigned_worker_id', sa.String(100), nullable=True))
    op.add_column('strategy_builds', sa.Column('last_heartbeat', sa.DateTime(), nullable=True))
    op.add_column('strategy_builds', sa.Column('queue_position', sa.Integer(), nullable=True))
    
    # Create worker_health table
    op.create_table(
        'worker_health',
        sa.Column('worker_id', sa.String(100), nullable=False),
        sa.Column('hostname', sa.String(255), nullable=False),
        sa.Column('capacity', sa.Integer(), nullable=False),
        sa.Column('active_jobs', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_heartbeat', sa.DateTime(), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, server_default='active'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('worker_id'),
    )
    
    # Add indexes for performance
    op.create_index('idx_build_assigned_worker', 'strategy_builds', ['assigned_worker_id'])
    op.create_index('idx_build_queue_position', 'strategy_builds', ['queue_position'])
    op.create_index('idx_worker_status', 'worker_health', ['status'])
    op.create_index('idx_worker_last_heartbeat', 'worker_health', ['last_heartbeat'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_worker_last_heartbeat', table_name='worker_health')
    op.drop_index('idx_worker_status', table_name='worker_health')
    op.drop_index('idx_build_queue_position', table_name='strategy_builds')
    op.drop_index('idx_build_assigned_worker', table_name='strategy_builds')
    
    # Drop worker_health table
    op.drop_table('worker_health')
    
    # Drop columns from strategy_builds
    op.drop_column('strategy_builds', 'queue_position')
    op.drop_column('strategy_builds', 'last_heartbeat')
    op.drop_column('strategy_builds', 'assigned_worker_id')
    op.drop_column('strategy_builds', 'last_checkpoint')
    op.drop_column('strategy_builds', 'retry_count')

