# Earnings & Payout System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire up real Stripe Connect revenue routing on license purchases, track earnings in a new `Earning` DB table via webhook, and replace the placeholder `$0.00` EarningsTab with live data.

**Architecture:** Destination charges route 65% of each subscription invoice to the creator's Stripe Connect account automatically. A new `Earning` model records each `invoice.payment_succeeded` event. The `/api/users/me/earnings` endpoint reads from this table and queries Stripe's Balance API. The EarningsTab frontend fetches and displays real numbers.

**Tech Stack:** FastAPI + SQLAlchemy + Alembic (backend), Stripe Python SDK, Next.js + TypeScript + Tailwind (frontend).

---

### Task 1: Add `Earning` model

**Files:**
- Create: `backend/app/models/earning.py`
- Modify: `backend/app/models/__init__.py`

**Context:** The project uses SQLAlchemy ORM with Alembic migrations. All models inherit from `Base = declarative_base()` in `backend/app/database.py`. Look at `backend/app/models/license.py` as a reference for the pattern (uuid PK as String(36), FKs to users/strategies, DateTime fields).

**Step 1: Create the model file**

Write `backend/app/models/earning.py`:

```python
"""Earning model for tracking creator payouts from strategy license payments."""
from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship

from app.database import Base


class Earning(Base):
    """Records each successful strategy license invoice payment.

    Created by the invoice.payment_succeeded webhook handler.
    All amounts stored in cents (USD).
    """
    __tablename__ = "earnings"

    uuid = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    strategy_id = Column(String(36), ForeignKey("strategies.uuid"), nullable=False)
    creator_user_id = Column(String(36), ForeignKey("users.uuid"), nullable=False)
    amount_gross = Column(Integer, nullable=False)    # cents — full invoice amount
    amount_creator = Column(Integer, nullable=False)  # cents — 65% share
    amount_platform = Column(Integer, nullable=False) # cents — 35% share
    stripe_invoice_id = Column(String(255), nullable=False, unique=True)
    stripe_subscription_id = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    strategy = relationship("Strategy", foreign_keys=[strategy_id])
    creator = relationship("User", foreign_keys=[creator_user_id])

    __table_args__ = (
        Index("ix_earnings_creator_user_id", "creator_user_id"),
        Index("ix_earnings_strategy_id", "strategy_id"),
    )
```

**Step 2: Register the model in `__init__.py`**

Open `backend/app/models/__init__.py`. Add `Earning` to the imports and `__all__` list, following the existing pattern for other models.

**Step 3: Verify the model imports cleanly**

```bash
source ~/Development/virtualenv/signalsynk/bin/activate && cd /Users/kevingriffiths/Development/OculusStrategy/backend && python3 -c "from app.models.earning import Earning; print('OK', Earning.__tablename__)"
```

Expected: `OK earnings`

**Step 4: Commit**

```bash
git add backend/app/models/earning.py backend/app/models/__init__.py
git commit -m "feat: add Earning model for tracking creator payout records"
```

---

### Task 2: Alembic migration for `earnings` table

**Files:**
- Create: `backend/alembic/versions/006_add_earnings_table.py`

**Context:** The project uses Alembic for DB migrations. Existing migrations are numbered `001`–`005`. The latest is `005_add_stripe_connect_account_id.py`. Open that file to find the `down_revision` value to use as the `down_revision` for this new migration.

**Step 1: Read the last migration to get its revision ID**

```bash
grep -n "^revision\|^down_revision" /Users/kevingriffiths/Development/OculusStrategy/backend/alembic/versions/005_add_stripe_connect_account_id.py
```

Note the `revision` value — you'll use it as the `down_revision` in the new migration.

**Step 2: Create `006_add_earnings_table.py`**

Write `backend/alembic/versions/006_add_earnings_table.py` (replace `<005_revision_id>` with the actual revision string from step 1):

```python
"""Add earnings table for creator payout tracking.

Revision ID: 006
Revises: <005_revision_id>
Create Date: 2026-02-24
"""
from alembic import op
import sqlalchemy as sa

revision = '006'
down_revision = '<005_revision_id>'
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
    op.create_index('ix_earnings_creator_user_id', 'earnings', ['creator_user_id'])
    op.create_index('ix_earnings_strategy_id', 'earnings', ['strategy_id'])


def downgrade():
    op.drop_index('ix_earnings_strategy_id', table_name='earnings')
    op.drop_index('ix_earnings_creator_user_id', table_name='earnings')
    op.drop_table('earnings')
```

**Step 3: Run the migration**

```bash
source ~/Development/virtualenv/signalsynk/bin/activate && cd /Users/kevingriffiths/Development/OculusStrategy/backend && alembic upgrade head 2>&1
```

Expected: output ending with `Running upgrade ... -> 006, Add earnings table for creator payout tracking.`

**Step 4: Commit**

```bash
git add backend/alembic/versions/006_add_earnings_table.py
git commit -m "feat: add Alembic migration 006 for earnings table"
```

---

### Task 3: Fix checkout webhook to store real Stripe subscription ID

**Files:**
- Modify: `backend/app/routers/payments.py`

**Context:** There is a bug in the existing `checkout.session.completed` webhook handler. Line 263 stores `subscription_id=session["id"]` (the checkout session ID, `cs_xxx`) but the `invoice.payment_succeeded` handler needs the actual Stripe subscription ID (`sub_xxx`). Line 241 already extracts it correctly as `subscription_id = session.get("subscription")` — but it's not being stored. Fix: store `subscription_id` (sub_xxx) in the license, and update the deduplication check to match.

**Step 1: Read the current webhook handler**

```bash
grep -n "subscription_id\|session\[.id.\]" /Users/kevingriffiths/Development/OculusStrategy/backend/app/routers/payments.py
```

**Step 2: Fix the `checkout.session.completed` handler**

In `backend/app/routers/payments.py`, find the `checkout.session.completed` block (around lines 233-273).

Change the deduplication check from:
```python
existing_result = await db.execute(
    select(License).where(License.subscription_id == session["id"])
)
```
to:
```python
existing_result = await db.execute(
    select(License).where(License.subscription_id == subscription_id)
)
```

Change the license creation from:
```python
license_obj = License(
    ...
    subscription_id=session["id"],
    ...
)
```
to:
```python
license_obj = License(
    ...
    subscription_id=subscription_id,
    ...
)
```

**Step 3: Verify the module imports cleanly**

```bash
source ~/Development/virtualenv/signalsynk/bin/activate && cd /Users/kevingriffiths/Development/OculusStrategy/backend && python3 -c "from app.routers.payments import router; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add backend/app/routers/payments.py
git commit -m "fix: store actual Stripe subscription ID (sub_xxx) in checkout webhook handler"
```

---

### Task 4: Add Connect routing to the checkout session

**Files:**
- Modify: `backend/app/routers/licenses.py`

**Context:** The `POST /api/strategies/{strategy_id}/license/checkout` handler (lines 119–224) creates a Stripe Checkout Session but does not route money to the strategy creator's Connect account. We need to look up the creator's `stripe_connect_account_id` and add `transfer_data` + `subscription_data.application_fee_percent` to the session. Guard: if creator has no Connect account, create the session without Connect (no failure).

The `CREATOR_EARNINGS_PERCENT = 0.65` → platform fee = 35%.

**Step 1: Read the imports at the top of `licenses.py`**

```bash
head -25 /Users/kevingriffiths/Development/OculusStrategy/backend/app/routers/licenses.py
```

Confirm `User` is imported. If not, add it.

**Step 2: Add creator Connect lookup before the checkout session call**

In `create_license_checkout`, after the `strategy` is fetched and before `stripe.checkout.Session.create(...)`, add:

```python
# Look up strategy creator's Connect account for revenue routing
creator_result = await db.execute(select(User).where(User.uuid == strategy.user_id))
creator = creator_result.scalar_one_or_none()
creator_connect_id = creator.stripe_connect_account_id if creator else None
```

**Step 3: Add Connect fields to the checkout session call**

Replace the `stripe.checkout.Session.create(...)` call. Add these fields alongside the existing ones (after `cancel_url`):

```python
        **({"transfer_data": {"destination": creator_connect_id}} if creator_connect_id else {}),
        subscription_data={
            "application_fee_percent": 35,
            "metadata": {"strategy_id": strategy_id},
        },
```

The full updated `Session.create()` should look like:

```python
    session = stripe.checkout.Session.create(
        customer=current_user.stripe_customer_id,
        mode="subscription",
        line_items=[{
            "price_data": {
                "currency": "usd",
                "product_data": {
                    "name": f"Oculus Strategy License — {strategy.name}",
                    "description": (
                        f"{license_data.license_type.capitalize()} license for "
                        f"the {strategy.name} strategy ({price_label})"
                    ),
                },
                "unit_amount": unit_amount,
                "recurring": {"interval": interval},
            },
            "quantity": 1,
        }],
        metadata={
            "event_type": "license_purchase",
            "user_id": current_user.uuid,
            "strategy_id": strategy_id,
            "license_type": license_data.license_type,
        },
        success_url=(
            f"{settings.FRONTEND_URL}/dashboard/strategies"
            "?license_success=1&session_id={CHECKOUT_SESSION_ID}"
        ),
        cancel_url=f"{settings.FRONTEND_URL}/dashboard/strategies?license_cancelled=1",
        **({"transfer_data": {"destination": creator_connect_id}} if creator_connect_id else {}),
        subscription_data={
            "application_fee_percent": 35,
            "metadata": {"strategy_id": strategy_id},
        },
    )
```

**Step 4: Verify imports and module loads**

```bash
source ~/Development/virtualenv/signalsynk/bin/activate && cd /Users/kevingriffiths/Development/OculusStrategy/backend && python3 -c "from app.routers.licenses import router; print('OK')"
```

Expected: `OK`

**Step 5: Commit**

```bash
git add backend/app/routers/licenses.py
git commit -m "feat: add Stripe Connect destination routing to license checkout session"
```

---

### Task 5: Add Connect routing to the direct subscribe endpoint

**Files:**
- Modify: `backend/app/routers/licenses.py`

**Context:** The `POST /api/strategies/{strategy_id}/license/subscribe` handler (lines 276–377) creates a Stripe Subscription directly (used when the user pays in-app via SetupIntent). It currently has no Connect routing. Add the same creator lookup and `transfer_data` / `application_fee_percent` to `stripe.Subscription.create()`.

**Step 1: Add creator lookup in `create_license_subscription`**

After `strategy` is fetched (around line 293), add:

```python
# Look up strategy creator's Connect account for revenue routing
creator_result = await db.execute(select(User).where(User.uuid == strategy.user_id))
creator = creator_result.scalar_one_or_none()
creator_connect_id = creator.stripe_connect_account_id if creator else None
```

**Step 2: Add Connect fields to `stripe.Subscription.create()`**

The existing call (around line 334) creates a subscription. Add these fields:

```python
        subscription = stripe.Subscription.create(
            customer=current_user.stripe_customer_id,
            items=[{
                "price_data": {
                    "currency": "usd",
                    "product": stripe_product.id,
                    "unit_amount": unit_amount,
                    "recurring": {"interval": interval},
                }
            }],
            default_payment_method=subscribe_data.payment_method_id,
            metadata={
                "event_type": "license_purchase",
                "user_id": current_user.uuid,
                "strategy_id": strategy_id,
                "license_type": subscribe_data.license_type,
            },
            **({"transfer_data": {"destination": creator_connect_id}} if creator_connect_id else {}),
            **({"application_fee_percent": 35} if creator_connect_id else {}),
        )
```

**Step 3: Verify module loads**

```bash
source ~/Development/virtualenv/signalsynk/bin/activate && cd /Users/kevingriffiths/Development/OculusStrategy/backend && python3 -c "from app.routers.licenses import router; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add backend/app/routers/licenses.py
git commit -m "feat: add Stripe Connect routing to direct license subscribe endpoint"
```

---

### Task 6: Add `invoice.payment_succeeded` webhook handler

**Files:**
- Modify: `backend/app/routers/payments.py`

**Context:** When a subscription invoice is paid (initial payment or renewal), Stripe fires `invoice.payment_succeeded`. We need to:
1. Look up the License by `subscription_id`
2. Look up the Strategy → creator User
3. Deduplicate on `stripe_invoice_id`
4. Write an `Earning` record

The `Earning` model was created in Task 1. Import it at the top of `payments.py`.
The `Strategy` model needs to be imported too.

**Step 1: Add imports to `payments.py`**

At the top of `backend/app/routers/payments.py`, after the existing model imports, add:

```python
from app.models.earning import Earning
from app.models.strategy import Strategy
```

**Step 2: Add the `invoice.payment_succeeded` handler**

In the webhook handler (`stripe_webhook` function), BEFORE the final `return {"status": "ignored"}` at the end of the function, add:

```python
    # Handle invoice.payment_succeeded for creator earnings tracking
    if event["type"] == "invoice.payment_succeeded":
        invoice = event["data"]["object"]
        subscription_id = invoice.get("subscription")

        if not subscription_id:
            return {"status": "ignored"}

        # Look up the license by subscription_id
        lic_result = await db.execute(
            select(License).where(License.subscription_id == subscription_id)
        )
        license_obj = lic_result.scalar_one_or_none()

        if not license_obj:
            return {"status": "ignored"}

        # Deduplicate — skip if already recorded this invoice
        dedup_result = await db.execute(
            select(Earning).where(Earning.stripe_invoice_id == invoice["id"])
        )
        if dedup_result.scalar_one_or_none():
            return {"status": "already_processed"}

        # Look up the strategy and its creator
        strat_result = await db.execute(
            select(Strategy).where(Strategy.uuid == license_obj.strategy_id)
        )
        strategy = strat_result.scalar_one_or_none()

        if not strategy:
            return {"status": "ignored"}

        creator_result = await db.execute(
            select(User).where(User.uuid == strategy.user_id)
        )
        creator = creator_result.scalar_one_or_none()

        # Only record if creator has a Connect account set up
        if not creator or not creator.stripe_connect_account_id:
            return {"status": "ignored"}

        amount_gross = invoice.get("amount_paid", 0)
        amount_platform = int(amount_gross * 0.35)
        amount_creator = amount_gross - amount_platform

        earning = Earning(
            strategy_id=strategy.uuid,
            creator_user_id=creator.uuid,
            amount_gross=amount_gross,
            amount_creator=amount_creator,
            amount_platform=amount_platform,
            stripe_invoice_id=invoice["id"],
            stripe_subscription_id=subscription_id,
        )
        db.add(earning)
        try:
            await db.commit()
            return {"status": "earning_recorded"}
        except Exception as e:
            await db.rollback()
            return {"status": "error", "message": str(e)}
```

**Step 3: Verify the module imports cleanly**

```bash
source ~/Development/virtualenv/signalsynk/bin/activate && cd /Users/kevingriffiths/Development/OculusStrategy/backend && python3 -c "from app.routers.payments import router; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add backend/app/routers/payments.py
git commit -m "feat: add invoice.payment_succeeded webhook handler to record creator earnings"
```

---

### Task 7: Update `EarningsResponse` schema and earnings endpoint

**Files:**
- Modify: `backend/app/schemas/subscriptions.py`
- Modify: `backend/app/routers/subscriptions.py`

**Context:** The current `EarningsResponse` schema uses placeholder fields (`subscriber_count`, `active_subscriptions`). Replace it with fields for real Stripe data. The `get_creator_earnings` handler uses placeholder math — replace it with DB queries on the `Earning` table and Stripe Balance API.

**Step 1: Replace `EarningsResponse` in `backend/app/schemas/subscriptions.py`**

Replace the existing `EarningsResponse` class with:

```python
class StrategyEarning(BaseModel):
    """Per-strategy earnings breakdown."""
    strategy_id: str
    strategy_name: str
    total_earned: float   # dollars
    payment_count: int


class EarningsResponse(BaseModel):
    """Schema for creator earnings dashboard."""
    total_earned: float         # lifetime earnings in dollars (from DB)
    balance_available: float    # available in Stripe Connect account (dollars)
    balance_pending: float      # pending in Stripe Connect account (dollars)
    by_strategy: list[StrategyEarning]

    class Config:
        from_attributes = True
```

Also add the `StrategyEarning` to the module — it can go directly above `EarningsResponse`.

**Step 2: Update imports in `backend/app/routers/subscriptions.py`**

Change the schema import line to include `StrategyEarning`:

```python
from app.schemas.subscriptions import (
    SubscriptionCreateRequest, SubscriptionResponse,
    SubscriptionListResponse, EarningsResponse, StrategyEarning
)
```

Add imports for `Earning` model and `func` aggregation (check if `func` is already imported — it is on line 4):

```python
from app.models.earning import Earning
```

**Step 3: Replace `get_creator_earnings` handler body**

Replace the entire body of `get_creator_earnings` in `backend/app/routers/subscriptions.py` with:

```python
    """
    Get creator earnings dashboard.
    - total_earned: sum of all Earning records for this creator (from DB)
    - balance_available / balance_pending: from Stripe Connect Balance API
    - by_strategy: breakdown per strategy from Earning records
    """
    # Query all earnings for this creator grouped by strategy
    earn_result = await db.execute(
        select(
            Earning.strategy_id,
            func.sum(Earning.amount_creator).label("total_cents"),
            func.count(Earning.uuid).label("payment_count"),
        )
        .where(Earning.creator_user_id == current_user.uuid)
        .group_by(Earning.strategy_id)
    )
    rows = earn_result.all()

    # Fetch strategy names
    strategy_ids = [row.strategy_id for row in rows]
    strat_result = await db.execute(
        select(Strategy).where(Strategy.uuid.in_(strategy_ids))
    )
    strategies_by_id = {s.uuid: s for s in strat_result.scalars().all()}

    total_earned_cents = sum(row.total_cents or 0 for row in rows)

    by_strategy = [
        StrategyEarning(
            strategy_id=row.strategy_id,
            strategy_name=strategies_by_id.get(row.strategy_id, Strategy()).name or "Unknown",
            total_earned=round((row.total_cents or 0) / 100, 2),
            payment_count=row.payment_count or 0,
        )
        for row in rows
    ]

    # Query Stripe Connect account balance (if creator has Connect set up)
    balance_available = 0.0
    balance_pending = 0.0

    if current_user.stripe_connect_account_id:
        try:
            balance = stripe.Balance.retrieve(
                stripe_account=current_user.stripe_connect_account_id
            )
            # Sum all currency available amounts (usually just USD)
            balance_available = round(
                sum(b["amount"] for b in balance["available"]) / 100, 2
            )
            balance_pending = round(
                sum(b["amount"] for b in balance["pending"]) / 100, 2
            )
        except stripe.error.StripeError:
            pass  # If Stripe call fails, return zeros for balance

    return EarningsResponse(
        total_earned=round(total_earned_cents / 100, 2),
        balance_available=balance_available,
        balance_pending=balance_pending,
        by_strategy=by_strategy,
    )
```

**Step 4: Verify module loads**

```bash
source ~/Development/virtualenv/signalsynk/bin/activate && cd /Users/kevingriffiths/Development/OculusStrategy/backend && python3 -c "from app.routers.subscriptions import router; print('OK')"
```

Expected: `OK`

**Step 5: Commit**

```bash
git add backend/app/schemas/subscriptions.py backend/app/routers/subscriptions.py
git commit -m "feat: replace placeholder earnings math with real DB + Stripe Balance queries"
```

---

### Task 8: Add `getEarnings` to the frontend API client

**Files:**
- Modify: `frontend/lib/api.ts`

**Context:** The `connectAPI` object in `frontend/lib/api.ts` has `connectStatus`, `connectOnboard`, and `connectDashboardLink`. Add `getEarnings` to call `GET /api/users/me/earnings`.

**Step 1: Find the `connectAPI` object**

```bash
grep -n "connectAPI\|connectStatus\|connectOnboard" /Users/kevingriffiths/Development/OculusStrategy/frontend/lib/api.ts | head -10
```

Read a few lines around it to understand the exact object structure.

**Step 2: Add `getEarnings` to `connectAPI`**

Add after `connectDashboardLink`:

```typescript
  getEarnings: async () => {
    const response = await api.get('/api/users/me/earnings')
    return response.data
  },
```

**Step 3: Verify TypeScript compiles**

```bash
cd /Users/kevingriffiths/Development/OculusStrategy/frontend && /opt/homebrew/opt/node@22/bin/npx tsc --noEmit 2>&1 | head -20
```

Expected: no output.

**Step 4: Commit**

```bash
git add frontend/lib/api.ts
git commit -m "feat: add getEarnings to connectAPI frontend client"
```

---

### Task 9: Update `EarningsTab.tsx` to display real earnings data

**Files:**
- Modify: `frontend/app/dashboard/account/components/EarningsTab.tsx`

**Context:** The `complete` state currently shows hardcoded `$0.00` in three cards and "No strategies with earnings yet". Replace with a `useEffect` that fetches from `connectAPI.getEarnings()` and renders real values. Also update the card labels to match the new schema: "Total Earnings", "Available in Stripe", "Pending".

**Step 1: Replace the entire file content**

Write `frontend/app/dashboard/account/components/EarningsTab.tsx`:

```tsx
'use client'

import { useEffect, useState } from 'react'
import { connectAPI } from '@/lib/api'
import { ExternalLink, CreditCard, CheckCircle, Loader, AlertCircle } from 'lucide-react'

interface ConnectStatus {
  status: 'not_started' | 'pending' | 'complete'
  account_id?: string
}

interface StrategyEarning {
  strategy_id: string
  strategy_name: string
  total_earned: number
  payment_count: number
}

interface Earnings {
  total_earned: number
  balance_available: number
  balance_pending: number
  by_strategy: StrategyEarning[]
}

function formatCurrency(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
  }).format(amount)
}

export default function EarningsTab() {
  const [loading, setLoading] = useState(true)
  const [connectStatus, setConnectStatus] = useState<ConnectStatus | null>(null)
  const [earnings, setEarnings] = useState<Earnings | null>(null)
  const [earningsLoading, setEarningsLoading] = useState(false)
  const [earningsError, setEarningsError] = useState<string | null>(null)
  const [onboarding, setOnboarding] = useState(false)

  useEffect(() => {
    fetchConnectStatus()
  }, [])

  const fetchConnectStatus = async () => {
    try {
      const status = await connectAPI.connectStatus()
      setConnectStatus(status)
      if (status.status === 'complete') {
        fetchEarnings()
      }
    } catch (error) {
      console.error('Failed to fetch connect status:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchEarnings = async () => {
    setEarningsLoading(true)
    setEarningsError(null)
    try {
      const data = await connectAPI.getEarnings()
      setEarnings(data)
    } catch (err: any) {
      setEarningsError(err.response?.data?.detail || 'Failed to load earnings')
    } finally {
      setEarningsLoading(false)
    }
  }

  const handleSetupPayouts = async () => {
    setOnboarding(true)
    try {
      const response = await connectAPI.connectOnboard()
      window.location.href = response.url
    } catch (error) {
      console.error('Failed to start onboarding:', error)
      setOnboarding(false)
    }
  }

  const handleViewDashboard = async () => {
    try {
      const response = await connectAPI.connectDashboardLink()
      window.location.href = response.url
    } catch (error) {
      console.error('Failed to get dashboard link:', error)
    }
  }

  if (loading) {
    return <div className="text-text-secondary">Loading earnings...</div>
  }

  // Not started
  if (connectStatus?.status === 'not_started') {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-blue-900/30 to-blue-800/30 border border-blue-700 rounded-lg p-8">
          <div className="flex items-start gap-4">
            <CreditCard className="text-blue-400 flex-shrink-0 mt-1" size={24} />
            <div className="flex-1">
              <h3 className="text-xl font-bold text-text mb-2">Set Up Marketplace Payouts</h3>
              <p className="text-text-secondary mb-4">
                Earn 65% revenue share from strategy subscriptions. Set up your payout account to start receiving payments.
              </p>
              <button
                onClick={handleSetupPayouts}
                disabled={onboarding}
                className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-600/50 text-white font-medium py-2 px-6 rounded-lg transition-colors"
              >
                {onboarding ? (
                  <>
                    <Loader size={18} className="animate-spin" />
                    Setting up...
                  </>
                ) : (
                  <>
                    <CreditCard size={18} />
                    Set Up Payouts
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
        <div className="bg-surface border border-border rounded-lg p-6">
          <h3 className="text-lg font-bold text-text mb-4">Earnings by Strategy</h3>
          <p className="text-text-secondary">No strategies with earnings yet</p>
        </div>
      </div>
    )
  }

  // Pending
  if (connectStatus?.status === 'pending') {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-yellow-900/30 to-yellow-800/30 border border-yellow-700 rounded-lg p-8">
          <div className="flex items-start gap-4">
            <Loader className="text-yellow-400 flex-shrink-0 mt-1 animate-spin" size={24} />
            <div className="flex-1">
              <h3 className="text-xl font-bold text-text mb-2">Account Setup In Progress</h3>
              <p className="text-text-secondary mb-4">
                Your payout account is being reviewed. This usually takes 1-2 business days.
              </p>
              <button
                onClick={handleSetupPayouts}
                className="flex items-center gap-2 bg-yellow-600 hover:bg-yellow-700 text-white font-medium py-2 px-6 rounded-lg transition-colors"
              >
                <ExternalLink size={18} />
                Complete Setup
              </button>
            </div>
          </div>
        </div>
        <div className="bg-surface border border-border rounded-lg p-6">
          <h3 className="text-lg font-bold text-text mb-4">Earnings by Strategy</h3>
          <p className="text-text-secondary">No strategies with earnings yet</p>
        </div>
      </div>
    )
  }

  // Complete — show real earnings
  return (
    <div className="space-y-6">
      {/* Connect active banner */}
      <div className="flex items-center justify-between bg-gradient-to-r from-green-900/30 to-green-800/30 border border-green-700 rounded-lg p-6">
        <div className="flex items-center gap-3">
          <CheckCircle className="text-green-400" size={24} />
          <div>
            <h3 className="text-lg font-bold text-text">Payout Account Active</h3>
            <p className="text-text-secondary text-sm">You&apos;re ready to receive payments</p>
          </div>
        </div>
        <button
          onClick={handleViewDashboard}
          className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-6 rounded-lg transition-colors"
        >
          <ExternalLink size={18} />
          View Dashboard
        </button>
      </div>

      {/* Earnings error */}
      {earningsError && (
        <div className="flex items-center gap-3 p-4 bg-red-900/20 border border-red-700 rounded-lg text-red-400">
          <AlertCircle size={20} />
          {earningsError}
        </div>
      )}

      {/* Summary cards */}
      {earningsLoading ? (
        <div className="flex items-center justify-center py-8">
          <Loader className="animate-spin text-primary" size={28} />
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-surface-hover border border-border rounded-lg p-6">
              <p className="text-text-secondary text-sm mb-2">Total Earnings</p>
              <p className="text-3xl font-bold text-primary">
                {formatCurrency(earnings?.total_earned ?? 0)}
              </p>
            </div>
            <div className="bg-surface-hover border border-border rounded-lg p-6">
              <p className="text-text-secondary text-sm mb-2">Available in Stripe</p>
              <p className="text-3xl font-bold text-green-400">
                {formatCurrency(earnings?.balance_available ?? 0)}
              </p>
            </div>
            <div className="bg-surface-hover border border-border rounded-lg p-6">
              <p className="text-text-secondary text-sm mb-2">Pending</p>
              <p className="text-3xl font-bold text-yellow-400">
                {formatCurrency(earnings?.balance_pending ?? 0)}
              </p>
            </div>
          </div>

          {/* Per-strategy breakdown */}
          <div className="bg-surface border border-border rounded-lg p-6">
            <h3 className="text-lg font-bold text-text mb-4">Earnings by Strategy</h3>
            {!earnings?.by_strategy?.length ? (
              <p className="text-text-secondary">No strategies with earnings yet</p>
            ) : (
              <div className="space-y-3">
                {earnings.by_strategy.map((s) => (
                  <div
                    key={s.strategy_id}
                    className="flex items-center justify-between py-3 border-b border-border last:border-0"
                  >
                    <div>
                      <p className="text-text font-medium">{s.strategy_name}</p>
                      <p className="text-text-secondary text-sm">
                        {s.payment_count} {s.payment_count === 1 ? 'payment' : 'payments'}
                      </p>
                    </div>
                    <p className="text-primary font-bold text-lg">
                      {formatCurrency(s.total_earned)}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
```

**Step 2: Verify TypeScript compiles**

```bash
cd /Users/kevingriffiths/Development/OculusStrategy/frontend && /opt/homebrew/opt/node@22/bin/npx tsc --noEmit 2>&1 | head -30
```

Expected: no output.

**Step 3: Commit**

```bash
git add frontend/app/dashboard/account/components/EarningsTab.tsx
git commit -m "feat: wire EarningsTab to real earnings API with per-strategy breakdown"
```

---

### Task 10: Register `invoice.payment_succeeded` in Stripe dashboard

**Context:** For the webhook to fire, `invoice.payment_succeeded` must be listed in the Stripe dashboard's webhook endpoint configuration. This is a manual step, but verify the current config and document what to check.

**Step 1: Verify webhook secret is configured**

```bash
grep -n "STRIPE_WEBHOOK_SECRET" /Users/kevingriffiths/Development/OculusStrategy/backend/app/config.py
```

Expected: `STRIPE_WEBHOOK_SECRET: str` is present.

**Step 2: Manual step — add event to Stripe dashboard**

In the Stripe Dashboard (https://dashboard.stripe.com/webhooks):
- Find your webhook endpoint pointing to `/api/webhooks/stripe`
- Add the event type `invoice.payment_succeeded` to the list of listened events
- No code change needed — the handler is already in place

**Step 3: Final smoke checks**

```bash
source ~/Development/virtualenv/signalsynk/bin/activate && cd /Users/kevingriffiths/Development/OculusStrategy/backend && python3 -c "
from app.models.earning import Earning
from app.routers.payments import router as pay_router
from app.routers.licenses import router as lic_router
from app.routers.subscriptions import router as sub_router
print('All modules OK')
print('Earning table:', Earning.__tablename__)
"
```

Expected:
```
All modules OK
Earning table: earnings
```

**Step 4: Final commit (any tweaks)**

```bash
git add -p
git commit -m "fix: earnings system smoke test tweaks"
```
