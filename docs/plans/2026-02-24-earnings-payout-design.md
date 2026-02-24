# Earnings & Payout System Design

**Date:** 2026-02-24
**Status:** Approved

## Problem

When a subscriber buys a strategy license, 100% of the payment goes to the platform Stripe account. The strategy creator's Connect account is stored but never used at checkout time. The Earnings tab shows hardcoded `$0.00` — earnings are fake placeholder math (`subscriber_count × $10 × 0.65`), not real Stripe data.

## Approach

**Option B — Destination charges + webhook earnings tracking (35% platform / 65% creator)**

- License checkout uses `transfer_data.destination` + `application_fee_percent=35` to auto-route 65% to the creator's Stripe Connect account on every invoice payment
- A new `Earning` DB table records each payment event from the `invoice.payment_succeeded` webhook
- The earnings API reads from this table for per-strategy totals; queries Stripe's Balance API for available/pending balance
- The EarningsTab frontend is wired to the real API

---

## Section 1 — New `Earning` Model

**File:** `backend/app/models/earning.py` (new)

```
Earning
├── uuid                   PK, String(36), default uuid4
├── strategy_id            FK → Strategy.uuid
├── creator_user_id        FK → User.uuid
├── amount_gross           Integer (cents)
├── amount_creator         Integer (cents) — 65% share
├── amount_platform        Integer (cents) — 35% share
├── stripe_invoice_id      String(255), unique — deduplication key
├── stripe_subscription_id String(255)
└── created_at             DateTime, default utcnow
```

No `status` field — all records represent confirmed earned payments (auto-transferred via destination charges).

---

## Section 2 — License Checkout: Connect Routing

**Files:** `backend/app/routers/licenses.py`

Both subscription creation paths need to look up the strategy creator's `stripe_connect_account_id` and add Connect routing if present. Guard: if creator has no Connect account or hasn't completed onboarding, create subscription without routing (no hard failure).

### `POST /api/strategies/{id}/license/checkout` (Checkout Session)

Add to `stripe.checkout.Session.create()`:
```python
transfer_data={"destination": creator_connect_id},   # only if connect_id present
subscription_data={
    "application_fee_percent": 35,
    "metadata": {"strategy_id": strategy_id},
},
```

### `POST /api/strategies/{id}/license/subscribe` (direct Subscription)

Add to `stripe.Subscription.create()`:
```python
transfer_data={"destination": creator_connect_id},   # only if connect_id present
application_fee_percent=35,
metadata={"strategy_id": strategy_id},
```

---

## Section 3 — Webhook: `invoice.payment_succeeded`

**File:** `backend/app/routers/payments.py`

New branch in the existing `POST /api/webhooks/stripe` handler:

```
invoice.payment_succeeded
  → get subscription_id from invoice
  → look up License by subscription_id
  → look up Strategy → creator User
  → if creator has no connect account: skip
  → deduplicate on stripe_invoice_id (skip if Earning already exists)
  → compute: amount_creator = amount_paid * 0.65
  → insert Earning record
  → commit
```

---

## Section 4 — Updated Earnings API & Schemas

**Files:**
- `backend/app/routers/subscriptions.py` — replace placeholder earnings handler
- `backend/app/schemas/earnings.py` (new)

### New schemas

```python
class StrategyEarning(BaseModel):
    strategy_id: str
    strategy_name: str
    total_earned: float   # dollars
    payment_count: int

class EarningsResponse(BaseModel):
    total_earned: float        # sum of all Earning.amount_creator, dollars
    balance_available: float   # from Stripe Connect Balance API
    balance_pending: float     # from Stripe Connect Balance API
    by_strategy: list[StrategyEarning]
```

### Updated `GET /api/users/me/earnings`

```
1. Query Earning WHERE creator_user_id = me → sum(amount_creator), group by strategy_id
2. Join Strategy for strategy_name
3. stripe.Balance.retrieve(stripe_account=user.stripe_connect_account_id) for available/pending
4. Return EarningsResponse
```

If user has no Connect account: return zeros for balance fields.

---

## Section 5 — Updated EarningsTab Frontend

**File:** `frontend/app/dashboard/account/components/EarningsTab.tsx`

The `complete` state currently shows hardcoded `$0.00`. Replace with:
- Fetch `GET /api/users/me/earnings` on mount (when status === "complete")
- Show real values in the three summary cards: Total Earnings, Available in Stripe, Pending
- Render per-strategy breakdown table: strategy name | total earned | payment count
- Loading spinner while fetching; error state if fetch fails
- "View Stripe Dashboard" button remains

**Add to `frontend/lib/api.ts`:** `connectAPI.getEarnings()` → `GET /api/users/me/earnings`

---

## Data Flow

```
Subscriber pays → Stripe Subscription invoice paid
  → Stripe fires invoice.payment_succeeded webhook
  → Our handler looks up License → Strategy → Creator
  → 65% auto-transferred to creator's Connect account (by Stripe)
  → Earning record written to DB

Creator opens Earnings tab (Connect complete)
  → GET /api/users/me/earnings
  → DB: sum Earning.amount_creator, group by strategy
  → Stripe API: Balance.retrieve for available/pending
  ← EarningsResponse with real data
  → Tab shows Total Earned, Available, Pending, per-strategy table
```

---

## Out of Scope

- Refund/chargeback reversal of Earning records
- Payout scheduling (Stripe handles automatic payouts)
- Admin earnings dashboard
- Historical payout reports
