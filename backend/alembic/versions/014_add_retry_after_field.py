"""Add retry_after field for exponential backoff

Revision ID: 014
Revises: 013
Create Date: 2026-02-21

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '014'
down_revision = '013'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add retry_after column to strategy_builds table
    op.add_column('strategy_builds', sa.Column('retry_after', sa.DateTime(), nullable=True))


def downgrade() -> None:
    # Remove retry_after column
    op.drop_column('strategy_builds', 'retry_after')

