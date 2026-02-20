"""add_strategy_files_to_build_iterations

Revision ID: 010
Revises: 009
Create Date: 2026-02-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add strategy_files JSON column to build_iterations.
    # Stores the worker-generated strategy files (filename → source code) so
    # the Docker-build phase can use the exact ML-rule implementation even when
    # training_results is reconstructed from the DB (Path 2 / best-iteration builds).
    op.add_column(
        'build_iterations',
        sa.Column('strategy_files', sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('build_iterations', 'strategy_files')

