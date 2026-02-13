"""add_strategy_creation_guides

Revision ID: 007
Revises: 006
Create Date: 2026-02-13 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create strategy_creation_guides table
    op.create_table(
        'strategy_creation_guides',
        sa.Column('uuid', sa.String(36), nullable=False),
        sa.Column('strategy_type', sa.String(100), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        # Primary key
        sa.PrimaryKeyConstraint('uuid'),
        # Unique constraint — one row per strategy type
        sa.UniqueConstraint('strategy_type', name='uq_strategy_creation_guide_type'),
    )

    # Create index for strategy_type lookups
    op.create_index('idx_guide_strategy_type', 'strategy_creation_guides', ['strategy_type'], unique=False)


def downgrade() -> None:
    # Drop index first
    op.drop_index('idx_guide_strategy_type', table_name='strategy_creation_guides')
    # Drop table
    op.drop_table('strategy_creation_guides')

