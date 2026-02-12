"""add_stripe_connect_account_id

Revision ID: 005
Revises: 004
Create Date: 2026-02-11 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add stripe_connect_account_id column to users table
    op.add_column('users', sa.Column('stripe_connect_account_id', sa.String(255), nullable=True))


def downgrade() -> None:
    # Remove stripe_connect_account_id column from users table
    op.drop_column('users', 'stripe_connect_account_id')

