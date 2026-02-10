"""add_product_description

Revision ID: 003
Revises: 002
Create Date: 2026-02-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add description column to products table
    op.add_column('products', sa.Column('description', sa.String(), nullable=True))


def downgrade() -> None:
    # Remove description column from products table
    op.drop_column('products', 'description')

