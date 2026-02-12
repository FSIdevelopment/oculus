"""add_build_history_and_feature_tracker

Revision ID: 004
Revises: 003
Create Date: 2026-02-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create build_history table
    op.create_table(
        'build_history',
        sa.Column('uuid', sa.String(36), nullable=False),
        sa.Column('strategy_name', sa.String(255), nullable=False),
        sa.Column('asset_class', sa.String(50), nullable=False),
        sa.Column('timeframe', sa.String(50), nullable=False),
        sa.Column('target', sa.Float(), nullable=False),
        sa.Column('symbols', sa.JSON(), nullable=False),
        sa.Column('success', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('build_date', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('backtest_results', sa.JSON(), nullable=True),
        sa.Column('model_info', sa.JSON(), nullable=True),
        sa.Column('risk_params', sa.JSON(), nullable=True),
        sa.Column('features', sa.JSON(), nullable=True),
        sa.Column('iterations', sa.JSON(), nullable=True),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.uuid'], ),
        sa.PrimaryKeyConstraint('uuid')
    )
    op.create_index('idx_build_history_user_id', 'build_history', ['user_id'], unique=False)
    op.create_index('idx_build_history_strategy_name', 'build_history', ['strategy_name'], unique=False)
    op.create_index('idx_build_history_asset_class', 'build_history', ['asset_class'], unique=False)
    op.create_index('idx_build_history_build_date', 'build_history', ['build_date'], unique=False)
    
    # Create feature_tracker table
    op.create_table(
        'feature_tracker',
        sa.Column('uuid', sa.String(36), nullable=False),
        sa.Column('feature_name', sa.String(255), nullable=False, unique=True),
        sa.Column('times_used', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('performance_data', sa.JSON(), nullable=True),
        sa.Column('entry_rule_stats', sa.JSON(), nullable=True),
        sa.Column('exit_rule_stats', sa.JSON(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('uuid')
    )
    op.create_index('idx_feature_tracker_name', 'feature_tracker', ['feature_name'], unique=False)
    op.create_index('idx_feature_tracker_updated_at', 'feature_tracker', ['updated_at'], unique=False)


def downgrade() -> None:
    # Drop feature_tracker table
    op.drop_index('idx_feature_tracker_updated_at', table_name='feature_tracker')
    op.drop_index('idx_feature_tracker_name', table_name='feature_tracker')
    op.drop_table('feature_tracker')
    
    # Drop build_history table
    op.drop_index('idx_build_history_build_date', table_name='build_history')
    op.drop_index('idx_build_history_asset_class', table_name='build_history')
    op.drop_index('idx_build_history_strategy_name', table_name='build_history')
    op.drop_index('idx_build_history_user_id', table_name='build_history')
    op.drop_table('build_history')

