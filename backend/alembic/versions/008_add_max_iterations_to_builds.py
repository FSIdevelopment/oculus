"""add_max_iterations_to_builds

Revision ID: 008
Revises: 007
Create Date: 2026-02-17 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add max_iterations column to strategy_builds table
    op.add_column('strategy_builds', sa.Column('max_iterations', sa.Integer(), nullable=True))


def downgrade() -> None:
    # Drop max_iterations column from strategy_builds table
    op.drop_column('strategy_builds', 'max_iterations')

