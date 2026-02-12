"""additional_models

Revision ID: 002
Revises: 001
Create Date: 2025-02-10 15:35:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create licenses table
    op.create_table(
        'licenses',
        sa.Column('uuid', sa.String(36), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, server_default='active'),
        sa.Column('license_type', sa.String(50), nullable=False),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('webhook_url', sa.String(255), nullable=True),
        sa.Column('subscription_id', sa.String(255), nullable=True),
        sa.Column('strategy_id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('uuid'),
        sa.ForeignKeyConstraint(['strategy_id'], ['strategies.uuid'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.uuid'], ),
    )
    op.create_index('idx_license_strategy_id', 'licenses', ['strategy_id'])
    op.create_index('idx_license_user_id', 'licenses', ['user_id'])
    op.create_index('idx_license_status', 'licenses', ['status'])

    # Create subscriptions table
    op.create_table(
        'subscriptions',
        sa.Column('uuid', sa.String(36), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, server_default='active'),
        sa.Column('stripe_id', sa.String(255), nullable=True),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('strategy_id', sa.String(36), nullable=True),
        sa.Column('license_id', sa.String(36), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('uuid'),
        sa.ForeignKeyConstraint(['user_id'], ['users.uuid'], ),
        sa.ForeignKeyConstraint(['strategy_id'], ['strategies.uuid'], ),
        sa.ForeignKeyConstraint(['license_id'], ['licenses.uuid'], ),
    )
    op.create_index('idx_subscription_user_id', 'subscriptions', ['user_id'])
    op.create_index('idx_subscription_strategy_id', 'subscriptions', ['strategy_id'])
    op.create_index('idx_subscription_license_id', 'subscriptions', ['license_id'])
    op.create_index('idx_subscription_status', 'subscriptions', ['status'])

    # Create ratings table
    op.create_table(
        'ratings',
        sa.Column('uuid', sa.String(36), nullable=False),
        sa.Column('rating', sa.Integer(), nullable=False),
        sa.Column('review_text', sa.Text(), nullable=True),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('strategy_id', sa.String(36), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('uuid'),
        sa.ForeignKeyConstraint(['user_id'], ['users.uuid'], ),
        sa.ForeignKeyConstraint(['strategy_id'], ['strategies.uuid'], ),
        sa.UniqueConstraint('user_id', 'strategy_id', name='uq_rating_user_strategy'),
        sa.CheckConstraint('rating >= 1 AND rating <= 5', name='ck_rating_range'),
    )
    op.create_index('idx_rating_user_id', 'ratings', ['user_id'])
    op.create_index('idx_rating_strategy_id', 'ratings', ['strategy_id'])

    # Create strategy_builds table
    op.create_table(
        'strategy_builds',
        sa.Column('uuid', sa.String(36), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('phase', sa.String(100), nullable=True),
        sa.Column('logs', sa.Text(), nullable=True),
        sa.Column('tokens_consumed', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('iteration_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('strategy_id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('uuid'),
        sa.ForeignKeyConstraint(['strategy_id'], ['strategies.uuid'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.uuid'], ),
    )
    op.create_index('idx_build_strategy_id', 'strategy_builds', ['strategy_id'])
    op.create_index('idx_build_user_id', 'strategy_builds', ['user_id'])
    op.create_index('idx_build_status', 'strategy_builds', ['status'])


def downgrade() -> None:
    op.drop_index('idx_build_status', table_name='strategy_builds')
    op.drop_index('idx_build_user_id', table_name='strategy_builds')
    op.drop_index('idx_build_strategy_id', table_name='strategy_builds')
    op.drop_table('strategy_builds')
    op.drop_index('idx_rating_strategy_id', table_name='ratings')
    op.drop_index('idx_rating_user_id', table_name='ratings')
    op.drop_table('ratings')
    op.drop_index('idx_subscription_status', table_name='subscriptions')
    op.drop_index('idx_subscription_license_id', table_name='subscriptions')
    op.drop_index('idx_subscription_strategy_id', table_name='subscriptions')
    op.drop_index('idx_subscription_user_id', table_name='subscriptions')
    op.drop_table('subscriptions')
    op.drop_index('idx_license_status', table_name='licenses')
    op.drop_index('idx_license_user_id', table_name='licenses')
    op.drop_index('idx_license_strategy_id', table_name='licenses')
    op.drop_table('licenses')

