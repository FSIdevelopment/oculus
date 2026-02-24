"""Add earnings table for creator payout tracking.

Revision ID: 017
Revises: 016
Create Date: 2026-02-24
"""
from alembic import op
import sqlalchemy as sa

revision = '017'
down_revision = '016'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'earnings',
        sa.Column('uuid', sa.String(36), primary_key=True),
        sa.Column('strategy_id', sa.String(36), sa.ForeignKey('strategies.uuid'), nullable=False),
        sa.Column('creator_user_id', sa.String(36), sa.ForeignKey('users.uuid'), nullable=False),
        sa.Column('amount_gross', sa.Integer(), nullable=False),
        sa.Column('amount_creator', sa.Integer(), nullable=False),
        sa.Column('amount_platform', sa.Integer(), nullable=False),
        sa.Column('stripe_invoice_id', sa.String(255), nullable=False, unique=True),
        sa.Column('stripe_subscription_id', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )
    op.create_index('idx_earning_creator_user_id', 'earnings', ['creator_user_id'])
    op.create_index('idx_earning_strategy_id', 'earnings', ['strategy_id'])


def downgrade():
    op.drop_index('idx_earning_strategy_id', table_name='earnings')
    op.drop_index('idx_earning_creator_user_id', table_name='earnings')
    op.drop_table('earnings')
