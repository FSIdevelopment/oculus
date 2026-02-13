"""add_build_iterations

Revision ID: 006
Revises: 005
Create Date: 2026-02-12 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create build_iterations table
    op.create_table(
        'build_iterations',
        sa.Column('uuid', sa.String(36), nullable=False),
        sa.Column('build_id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('iteration_number', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        # LLM Context
        sa.Column('llm_design', sa.JSON(), nullable=True),
        sa.Column('llm_thinking', sa.Text(), nullable=True),
        # Training Config
        sa.Column('training_config', sa.JSON(), nullable=True),
        # Label Configuration Results
        sa.Column('optimal_label_config', sa.JSON(), nullable=True),
        # Feature Data
        sa.Column('features', sa.JSON(), nullable=True),
        # Training Results Per Model Type
        sa.Column('hyperparameter_results', sa.JSON(), nullable=True),
        sa.Column('nn_training_results', sa.JSON(), nullable=True),
        sa.Column('lstm_training_results', sa.JSON(), nullable=True),
        sa.Column('model_evaluations', sa.JSON(), nullable=True),
        # Best Model
        sa.Column('best_model', sa.JSON(), nullable=True),
        # Extracted Rules
        sa.Column('entry_rules', sa.JSON(), nullable=True),
        sa.Column('exit_rules', sa.JSON(), nullable=True),
        # Backtest Results
        sa.Column('backtest_results', sa.JSON(), nullable=True),
        # Timestamps
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        # Foreign keys
        sa.ForeignKeyConstraint(['build_id'], ['strategy_builds.uuid']),
        sa.ForeignKeyConstraint(['user_id'], ['users.uuid']),
        # Primary key
        sa.PrimaryKeyConstraint('uuid'),
        # Unique constraint
        sa.UniqueConstraint('build_id', 'iteration_number', name='uq_build_iteration'),
    )

    # Create indexes
    op.create_index('idx_iteration_build_id', 'build_iterations', ['build_id'], unique=False)
    op.create_index('idx_iteration_user_id', 'build_iterations', ['user_id'], unique=False)
    op.create_index('idx_iteration_status', 'build_iterations', ['status'], unique=False)


def downgrade() -> None:
    # Drop indexes first
    op.drop_index('idx_iteration_status', table_name='build_iterations')
    op.drop_index('idx_iteration_user_id', table_name='build_iterations')
    op.drop_index('idx_iteration_build_id', table_name='build_iterations')
    # Drop table
    op.drop_table('build_iterations')

