"""add_config_to_strategies

Revision ID: 011
Revises: 010
Create Date: 2026-02-21 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "011"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add config JSON column to strategies.
    # Stores the strategy config.json for SignalSynk integration.
    # Config is pulled from BuildIteration.strategy_files and cached here.
    op.add_column(
        'strategies',
        sa.Column('config', sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('strategies', 'config')

