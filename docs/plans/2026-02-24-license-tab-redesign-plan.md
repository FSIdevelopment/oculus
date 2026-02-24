# License Tab Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign the account page Licenses tab to show strategy name/version, display the license key securely with a masked reveal toggle, show a formatted expiry date, and allow users to cancel their subscription at period end.

**Architecture:** Enrich `LicenseResponse` with `strategy_name` by joining `Strategy` in the list query. Add a `DELETE /api/licenses/{license_id}` endpoint that calls `stripe.Subscription.modify(cancel_at_period_end=True)` and sets `status = "cancelling"`. Redesign the frontend card with masked UUID, show/hide toggle, copy button, and a cancel confirmation flow.

**Tech Stack:** FastAPI + SQLAlchemy (backend), Next.js + TypeScript + Tailwind CSS (frontend), Stripe Python SDK, Lucide React icons.

---

### Task 1: Enrich `LicenseResponse` schema with `strategy_name`

**Files:**
- Modify: `backend/app/schemas/licenses.py`

**Step 1: Add `strategy_name` to `LicenseResponse`**

Open `backend/app/schemas/licenses.py`. In the `LicenseResponse` class, add one field after `webhook_url`:

```python
strategy_name: Optional[str] = None
```

The class should now look like:

```python
class LicenseResponse(BaseModel):
    """Schema for license detail response."""
    uuid: str
    status: str
    license_type: str
    strategy_id: str
    user_id: str
    strategy_version: Optional[int] = None
    webhook_url: Optional[str] = None
    strategy_name: Optional[str] = None
    created_at: datetime
    expires_at: datetime

    class Config:
        from_attributes = True
```

**Step 2: Verify no tests break from schema change**

```bash
cd backend && python3 -m pytest tests/ -q 2>&1 | tail -20
```

Expected: all existing tests pass (adding an optional field is non-breaking).

**Step 3: Commit**

```bash
git add backend/app/schemas/licenses.py
git commit -m "feat: add strategy_name to LicenseResponse schema"
```

---

### Task 2: Populate `strategy_name` in the list query

**Files:**
- Modify: `backend/app/routers/licenses.py`

**Step 1: Find the `GET /api/users/me/licenses` handler**

Search for `users/me/licenses` in `backend/app/routers/licenses.py` — look for `@router.get("/api/users/me/licenses"`. Read the surrounding ~30 lines to understand the current query.

**Step 2: Update the query to join Strategy**

The current query selects `License` records and returns them directly. We need to join `Strategy` and manually build the response dict so `strategy_name` is populated.

Replace the existing handler body with:

```python
@router.get("/api/users/me/licenses", response_model=LicenseListResponse)
async def list_my_licenses(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """List all licenses belonging to the current user."""
    # Total count
    count_result = await db.execute(
        select(func.count(License.uuid)).where(License.user_id == current_user.uuid)
    )
    total = count_result.scalar_one()

    # Fetch licenses with strategy join
    result = await db.execute(
        select(License, Strategy.name.label("strategy_name"))
        .join(Strategy, License.strategy_id == Strategy.uuid, isouter=True)
        .where(License.user_id == current_user.uuid)
        .order_by(License.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    rows = result.all()

    items = []
    for license_obj, strategy_name in rows:
        item = LicenseResponse(
            uuid=license_obj.uuid,
            status=license_obj.status,
            license_type=license_obj.license_type,
            strategy_id=license_obj.strategy_id,
            user_id=license_obj.user_id,
            strategy_version=license_obj.strategy_version,
            webhook_url=license_obj.webhook_url,
            strategy_name=strategy_name,
            created_at=license_obj.created_at,
            expires_at=license_obj.expires_at,
        )
        items.append(item)

    return LicenseListResponse(items=items, total=total, skip=skip, limit=limit)
```

Note: `Strategy` is already imported in this file. Confirm `from app.models.strategy import Strategy` is present at the top.

**Step 3: Verify the app starts without errors**

```bash
cd backend && python3 -c "from app.routers.licenses import router; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add backend/app/routers/licenses.py
git commit -m "feat: join Strategy in license list query to populate strategy_name"
```

---

### Task 3: Add `DELETE /api/licenses/{license_id}` cancel endpoint

**Files:**
- Modify: `backend/app/routers/licenses.py`

**Step 1: Add the cancel endpoint after the renew endpoint**

Find the `PUT /api/licenses/{license_id}/renew` handler. Directly after it, add:

```python
@router.delete("/api/licenses/{license_id}", response_model=LicenseResponse)
async def cancel_license(
    license_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel a license subscription at period end.

    - Marks the Stripe subscription to cancel_at_period_end=True
    - Sets license status to 'cancelling'
    - The license remains active until expires_at
    """
    result = await db.execute(
        select(License).where(License.uuid == license_id)
    )
    license_obj = result.scalar_one_or_none()

    if not license_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="License not found")

    if license_obj.user_id != current_user.uuid:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    if license_obj.status not in ("active",):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel a license with status '{license_obj.status}'"
        )

    # Cancel Stripe subscription at period end (if one exists)
    if license_obj.subscription_id:
        try:
            stripe.Subscription.modify(
                license_obj.subscription_id,
                cancel_at_period_end=True
            )
        except stripe.error.StripeError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Stripe error: {str(e)}"
            )

    license_obj.status = "cancelling"
    await db.commit()
    await db.refresh(license_obj)

    # Fetch strategy_name for response
    strat_result = await db.execute(
        select(Strategy.name).where(Strategy.uuid == license_obj.strategy_id)
    )
    strategy_name = strat_result.scalar_one_or_none()

    return LicenseResponse(
        uuid=license_obj.uuid,
        status=license_obj.status,
        license_type=license_obj.license_type,
        strategy_id=license_obj.strategy_id,
        user_id=license_obj.user_id,
        strategy_version=license_obj.strategy_version,
        webhook_url=license_obj.webhook_url,
        strategy_name=strategy_name,
        created_at=license_obj.created_at,
        expires_at=license_obj.expires_at,
    )
```

**Step 2: Verify import of `stripe.error` is available**

`stripe` is already imported at the top of the file. `stripe.error.StripeError` is part of the stripe package — no new imports needed.

**Step 3: Confirm the app loads**

```bash
cd backend && python3 -c "from app.routers.licenses import router; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add backend/app/routers/licenses.py
git commit -m "feat: add DELETE /api/licenses/{id} endpoint to cancel subscription at period end"
```

---

### Task 4: Add `cancelLicense` to the frontend API client

**Files:**
- Modify: `frontend/lib/api.ts`

**Step 1: Find the licenses section in `frontend/lib/api.ts`**

Search for `renewLicense` or `listLicenses` in `frontend/lib/api.ts` to find the license API block.

**Step 2: Add the cancel function**

Add `cancelLicense` alongside the other license functions:

```typescript
cancelLicense: (licenseId: string) =>
  api.delete(`/api/licenses/${licenseId}`),
```

**Step 3: Commit**

```bash
git add frontend/lib/api.ts
git commit -m "feat: add cancelLicense API call to frontend client"
```

---

### Task 5: Redesign the `LicensesTab` component

**Files:**
- Modify: `frontend/app/dashboard/account/components/LicensesTab.tsx`

**Step 1: Replace the entire file with the new implementation**

```tsx
'use client'

import { useState, useEffect, useCallback } from 'react'
import api from '@/lib/api'
import { Loader, AlertCircle, Eye, EyeOff, Copy, Check } from 'lucide-react'

interface License {
  uuid: string
  status: string
  license_type: string
  strategy_id: string
  strategy_name: string | null
  strategy_version: number | null
  created_at: string
  expires_at: string
}

function maskLicenseKey(uuid: string): string {
  // Show last 6 chars, mask the rest
  const parts = uuid.split('-')
  const last = parts[parts.length - 1]
  const hint = last.slice(-6)
  return `••••••••-••••-••••-••••-${hint}`
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  })
}

function daysUntilExpiry(expiresAt: string): number {
  return Math.ceil((new Date(expiresAt).getTime() - Date.now()) / 86_400_000)
}

function StatusBadge({ status, expiresAt }: { status: string; expiresAt: string }) {
  const expired = new Date(expiresAt) < new Date()
  const effectiveStatus = expired && status === 'active' ? 'expired' : status

  const config: Record<string, { label: string; className: string }> = {
    active:      { label: 'Active',     className: 'bg-green-900/20 text-green-400 border border-green-700' },
    cancelling:  { label: `Cancels ${formatDate(expiresAt)}`, className: 'bg-yellow-900/20 text-yellow-400 border border-yellow-700' },
    expired:     { label: 'Expired',    className: 'bg-red-900/20 text-red-400 border border-red-700' },
    cancelled:   { label: 'Cancelled',  className: 'bg-surface text-text-secondary border border-border' },
  }

  const { label, className } = config[effectiveStatus] ?? config['cancelled']
  return (
    <span className={`px-3 py-1 rounded-full text-xs font-medium ${className}`}>
      {label}
    </span>
  )
}

function LicenseKeyRow({ uuid }: { uuid: string }) {
  const [revealed, setRevealed] = useState(false)
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(uuid)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="flex items-center gap-2 bg-black/20 rounded-lg px-3 py-2 font-mono text-sm">
      <span className="flex-1 text-text-secondary select-all">
        {revealed ? uuid : maskLicenseKey(uuid)}
      </span>
      <button
        onClick={() => setRevealed((v) => !v)}
        title={revealed ? 'Hide key' : 'Show key'}
        className="text-text-secondary hover:text-text transition-colors p-1"
      >
        {revealed ? <EyeOff size={15} /> : <Eye size={15} />}
      </button>
      <button
        onClick={handleCopy}
        title="Copy license key"
        className="text-text-secondary hover:text-text transition-colors p-1"
      >
        {copied ? <Check size={15} className="text-green-400" /> : <Copy size={15} />}
      </button>
    </div>
  )
}

function LicenseCard({
  license,
  onCancelled,
}: {
  license: License
  onCancelled: (id: string) => void
}) {
  const [confirming, setConfirming] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const daysLeft = daysUntilExpiry(license.expires_at)
  const showWarning = daysLeft > 0 && daysLeft <= 7

  const handleCancel = async () => {
    setCancelling(true)
    setError(null)
    try {
      await api.delete(`/api/licenses/${license.uuid}`)
      onCancelled(license.uuid)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to cancel subscription')
      setConfirming(false)
    } finally {
      setCancelling(false)
    }
  }

  return (
    <div className="bg-surface border border-border rounded-lg p-6 hover:border-primary transition-colors space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-base font-bold text-text">
            {license.strategy_name ?? 'Strategy License'}
            {license.strategy_version != null && (
              <span className="ml-2 text-text-secondary font-normal text-sm">
                v{license.strategy_version}
              </span>
            )}
          </h3>
        </div>
        <StatusBadge status={license.status} expiresAt={license.expires_at} />
      </div>

      {/* License key */}
      <div>
        <p className="text-text-secondary text-xs mb-1">License Key</p>
        <LicenseKeyRow uuid={license.uuid} />
      </div>

      {/* Info grid */}
      <div className="grid grid-cols-3 gap-4 text-sm">
        <div>
          <p className="text-text-secondary text-xs mb-1">Type</p>
          <p className="text-text font-medium capitalize">{license.license_type}</p>
        </div>
        <div>
          <p className="text-text-secondary text-xs mb-1">Expires</p>
          <p className="text-text font-medium flex items-center gap-2">
            {formatDate(license.expires_at)}
            {showWarning && (
              <span className="text-xs bg-yellow-900/20 text-yellow-400 border border-yellow-700 px-2 py-0.5 rounded-full">
                {daysLeft}d left
              </span>
            )}
          </p>
        </div>
        <div>
          <p className="text-text-secondary text-xs mb-1">Version</p>
          <p className="text-text font-medium">
            {license.strategy_version != null ? `v${license.strategy_version}` : '—'}
          </p>
        </div>
      </div>

      {/* Error */}
      {error && (
        <p className="text-red-400 text-sm flex items-center gap-2">
          <AlertCircle size={14} /> {error}
        </p>
      )}

      {/* Cancel */}
      {license.status === 'active' && (
        <div>
          {confirming ? (
            <div className="space-y-2">
              <p className="text-text-secondary text-sm">
                Your license stays active until{' '}
                <span className="text-text font-medium">{formatDate(license.expires_at)}</span>.
                Confirm cancel?
              </p>
              <div className="flex gap-2">
                <button
                  onClick={handleCancel}
                  disabled={cancelling}
                  className="px-4 py-2 bg-red-700 hover:bg-red-600 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50"
                >
                  {cancelling ? 'Cancelling…' : 'Yes, Cancel'}
                </button>
                <button
                  onClick={() => setConfirming(false)}
                  disabled={cancelling}
                  className="px-4 py-2 bg-surface border border-border hover:border-primary text-text text-sm font-medium rounded-lg transition-colors"
                >
                  Back
                </button>
              </div>
            </div>
          ) : (
            <button
              onClick={() => setConfirming(true)}
              className="text-sm text-text-secondary hover:text-red-400 transition-colors underline underline-offset-2"
            >
              Cancel Subscription
            </button>
          )}
        </div>
      )}
    </div>
  )
}

export default function LicensesTab() {
  const [licenses, setLicenses] = useState<License[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchLicenses = useCallback(async () => {
    try {
      const response = await api.get('/api/users/me/licenses')
      setLicenses(response.data.items || [])
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch licenses')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchLicenses()
  }, [fetchLicenses])

  const handleCancelled = useCallback((licenseId: string) => {
    setLicenses((prev) =>
      prev.map((l) => (l.uuid === licenseId ? { ...l, status: 'cancelling' } : l))
    )
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader className="animate-spin text-primary" size={32} />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {error && (
        <div className="flex items-center gap-3 p-4 bg-red-900/20 border border-red-700 rounded-lg text-red-400">
          <AlertCircle size={20} />
          {error}
        </div>
      )}

      {licenses.length === 0 ? (
        <div className="bg-surface border border-border rounded-lg p-8 text-center">
          <p className="text-text-secondary mb-4">No active licenses yet</p>
          <p className="text-sm text-text-secondary">
            Purchase a license to run strategies on the platform
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {licenses.map((license) => (
            <LicenseCard key={license.uuid} license={license} onCancelled={handleCancelled} />
          ))}
        </div>
      )}
    </div>
  )
}
```

**Step 2: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit 2>&1 | head -40
```

Expected: no errors.

**Step 3: Commit**

```bash
git add frontend/app/dashboard/account/components/LicensesTab.tsx
git commit -m "feat: redesign LicensesTab with masked key, strategy name/version, cancel flow"
```

---

### Task 6: Smoke test end-to-end

**Step 1: Start backend**

```bash
cd backend && source ~/Development/virtualenv/signalsynk/bin/activate
uvicorn main:app --port 8000 --log-level info &
```

**Step 2: Start frontend**

```bash
cd frontend && npm run dev &
```

**Step 3: Manual checks**

1. Log in and navigate to **Account → Licenses**
2. Confirm each card shows: strategy name, version badge, masked license key, type, formatted expiry date
3. Click the eye icon — confirm full UUID appears; click again to mask
4. Click copy icon — confirm clipboard contains the full UUID and "✓" appears briefly
5. For an active license, click "Cancel Subscription" — confirm the confirmation prompt appears with the expiry date
6. Click "Back" — confirm the prompt disappears without any API call
7. Click "Cancel Subscription" → "Yes, Cancel" — confirm status badge changes to amber "Cancels on {date}"
8. Confirm the "Cancel Subscription" link disappears for the now-cancelling license

**Step 4: Final commit (if any tweaks needed)**

```bash
git add -p
git commit -m "fix: license tab smoke test tweaks"
```
