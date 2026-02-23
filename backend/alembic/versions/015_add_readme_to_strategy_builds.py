"""Add readme field to strategy_builds

Revision ID: 015
Revises: 014
Create Date: 2026-02-23

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '015'
down_revision = '014'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add readme Text column to strategy_builds.
    # Stores the LLM-generated README.md content so it can be retrieved
    # from the database instead of the filesystem (which doesn't work in production).
    op.add_column(
        'strategy_builds',
        sa.Column('readme', sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('strategy_builds', 'readme')

