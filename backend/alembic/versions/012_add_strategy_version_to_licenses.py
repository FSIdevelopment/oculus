"""add_strategy_version_to_licenses

Revision ID: 012
Revises: 011
Create Date: 2026-02-21 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add strategy_version column to licenses table.
    # Stores the version number of the strategy that is being licensed.
    # This is used to pull the correct build from the docker registry.
    op.add_column(
        'licenses',
        sa.Column('strategy_version', sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('licenses', 'strategy_version')

