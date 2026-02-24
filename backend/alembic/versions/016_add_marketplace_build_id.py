"""Add marketplace_build_id to strategies

Revision ID: 016
Revises: 015
Create Date: 2026-02-24

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '016'
down_revision = '015'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add marketplace_build_id to strategies so owners can pin a specific
    # build version when listing on the marketplace.
    op.add_column(
        'strategies',
        sa.Column('marketplace_build_id', sa.String(36), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('strategies', 'marketplace_build_id')
