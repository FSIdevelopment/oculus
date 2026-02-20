"""add_marketplace_fields_to_strategies

Revision ID: 009
Revises: 008
Create Date: 2026-02-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add marketplace_listed and marketplace_price columns to strategies table
    op.add_column('strategies', sa.Column('marketplace_listed', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('strategies', sa.Column('marketplace_price', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('strategies', 'marketplace_price')
    op.drop_column('strategies', 'marketplace_listed')

