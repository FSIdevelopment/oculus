"""initial_models

Revision ID: 001
Revises: 
Create Date: 2025-02-10 15:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('uuid', sa.String(36), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('phone_number', sa.String(20), nullable=True),
        sa.Column('balance', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('status', sa.String(50), nullable=False, server_default='active'),
        sa.Column('user_role', sa.String(50), nullable=False, server_default='user'),
        sa.Column('stripe_customer_id', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('uuid'),
        sa.UniqueConstraint('email'),
    )
    op.create_index('idx_user_email', 'users', ['email'])
    op.create_index('idx_user_status', 'users', ['status'])

    # Create strategies table
    op.create_table(
        'strategies',
        sa.Column('uuid', sa.String(36), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, server_default='draft'),
        sa.Column('strategy_type', sa.String(100), nullable=True),
        sa.Column('symbols', postgresql.JSON(), nullable=True),
        sa.Column('target_return', sa.Float(), nullable=True),
        sa.Column('backtest_results', postgresql.JSON(), nullable=True),
        sa.Column('docker_registry', sa.String(255), nullable=True),
        sa.Column('docker_image_url', sa.String(255), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('subscriber_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('rating', sa.Float(), nullable=True),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('uuid'),
        sa.ForeignKeyConstraint(['user_id'], ['users.uuid'], ),
    )
    op.create_index('idx_strategy_user_id', 'strategies', ['user_id'])
    op.create_index('idx_strategy_status', 'strategies', ['status'])

    # Create chat_history table
    op.create_table(
        'chat_history',
        sa.Column('uuid', sa.String(36), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('message_type', sa.String(50), nullable=False),
        sa.Column('content', postgresql.JSON(), nullable=True),
        sa.Column('strategy_id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('uuid'),
        sa.ForeignKeyConstraint(['strategy_id'], ['strategies.uuid'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.uuid'], ),
    )
    op.create_index('idx_chat_strategy_id', 'chat_history', ['strategy_id'])
    op.create_index('idx_chat_user_id', 'chat_history', ['user_id'])

    # Create balances table
    op.create_table(
        'balances',
        sa.Column('uuid', sa.String(36), nullable=False),
        sa.Column('tokens', sa.Float(), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('transaction_type', sa.String(50), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('uuid'),
        sa.ForeignKeyConstraint(['user_id'], ['users.uuid'], ),
    )
    op.create_index('idx_balance_user_id', 'balances', ['user_id'])
    op.create_index('idx_balance_transaction_type', 'balances', ['transaction_type'])

    # Create products table
    op.create_table(
        'products',
        sa.Column('uuid', sa.String(36), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('product_type', sa.String(50), nullable=False),
        sa.Column('token_amount', sa.Integer(), nullable=True),
        sa.Column('stripe_product_id', sa.String(255), nullable=True),
        sa.Column('stripe_price_id', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('uuid'),
    )
    op.create_index('idx_product_type', 'products', ['product_type'])
    op.create_index('idx_product_stripe_id', 'products', ['stripe_product_id'])

    # Create purchases table
    op.create_table(
        'purchases',
        sa.Column('uuid', sa.String(36), nullable=False),
        sa.Column('product_type', sa.String(50), nullable=False),
        sa.Column('stripe_id', sa.String(255), nullable=True),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('product_id', sa.String(36), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('purchased_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('uuid'),
        sa.ForeignKeyConstraint(['user_id'], ['users.uuid'], ),
        sa.ForeignKeyConstraint(['product_id'], ['products.uuid'], ),
    )
    op.create_index('idx_purchase_user_id', 'purchases', ['user_id'])
    op.create_index('idx_purchase_product_id', 'purchases', ['product_id'])
    op.create_index('idx_purchase_stripe_id', 'purchases', ['stripe_id'])


def downgrade() -> None:
    op.drop_index('idx_purchase_stripe_id', table_name='purchases')
    op.drop_index('idx_purchase_product_id', table_name='purchases')
    op.drop_index('idx_purchase_user_id', table_name='purchases')
    op.drop_table('purchases')
    op.drop_index('idx_product_stripe_id', table_name='products')
    op.drop_index('idx_product_type', table_name='products')
    op.drop_table('products')
    op.drop_index('idx_balance_transaction_type', table_name='balances')
    op.drop_index('idx_balance_user_id', table_name='balances')
    op.drop_table('balances')
    op.drop_index('idx_chat_user_id', table_name='chat_history')
    op.drop_index('idx_chat_strategy_id', table_name='chat_history')
    op.drop_table('chat_history')
    op.drop_index('idx_strategy_status', table_name='strategies')
    op.drop_index('idx_strategy_user_id', table_name='strategies')
    op.drop_table('strategies')
    op.drop_index('idx_user_status', table_name='users')
    op.drop_index('idx_user_email', table_name='users')
    op.drop_table('users')

