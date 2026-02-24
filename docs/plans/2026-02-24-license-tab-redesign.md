# License Tab Redesign

**Date:** 2026-02-24
**Status:** Approved

## Overview

Redesign the account page Licenses tab to securely display the license key, show strategy name and version, show expiration as a date, and allow users to cancel their subscription (cancel at period end via Stripe).

## Approach

Option A — enrich the existing `LicenseResponse` schema with `strategy_name` and add a cancel endpoint. Single API call, no N+1 requests.

---

## Backend Changes

### 1. Enrich `LicenseResponse` schema
**File:** `backend/app/schemas/licenses.py`

Add `strategy_name: Optional[str] = None` to `LicenseResponse`. `strategy_version` already exists.

### 2. Update `GET /api/users/me/licenses` query
**File:** `backend/app/routers/licenses.py`

Join the `Strategy` model in the list query and populate `strategy_name` from `strategy.name` when building the response.

### 3. New cancel endpoint
**File:** `backend/app/routers/licenses.py`

```
DELETE /api/licenses/{license_id}
```

- Verify license belongs to current user (404 if not found, 403 if wrong user)
- If `license.subscription_id` exists: call `stripe.Subscription.modify(subscription_id, cancel_at_period_end=True)`
- Set `license.status = "cancelling"`
- Commit and return updated `LicenseResponse`

### 4. Add `"cancelling"` as a valid status
**File:** `backend/app/models/license.py`

Document/allow `"cancelling"` as a valid `status` value alongside `"active"`, `"expired"`, `"cancelled"`.

---

## Frontend Changes

### 5. Update `License` interface
**File:** `frontend/app/dashboard/account/components/LicensesTab.tsx`

Add `strategy_name: string` and ensure `strategy_version: number | null` is present.

### 6. Add cancel API call to `frontend/lib/api.ts`

```ts
cancelLicense(licenseId: string): Promise<LicenseResponse>
// DELETE /api/licenses/{licenseId}
```

### 7. Redesign the license card

Each card shows:
- **Header:** Strategy name + version (e.g. `My Strategy v3`), status badge
- **License key row:** Masked UUID (`••••••••-••••-••••-••••-a3f9b2`) with show/hide eye toggle and copy-to-clipboard button
- **Info grid:** License Type | Expiration (formatted date) | Version
- **Expiry warning:** Amber badge "X days left" when ≤ 7 days from expiry
- **Status badge colours:**
  - `active` → green
  - `cancelling` → amber, label "Cancels on {expiry date}"
  - `expired` → red
  - `cancelled` → grey
- **Cancel button:** Shown only for `active` licenses. Opens inline confirmation ("Your license stays active until {date}. Confirm cancel?") with Confirm/Back buttons. On confirm calls `cancelLicense()` and refreshes list.

---

## Data Flow

```
LicensesTab mounts
  → GET /api/users/me/licenses
  ← LicenseListResponse { items: [{ uuid, status, license_type, strategy_id,
                                     strategy_name, strategy_version, expires_at }] }

User clicks show/hide eye
  → local state toggle (no API call)

User clicks copy
  → navigator.clipboard.writeText(license.uuid)

User clicks Cancel Subscription → confirms
  → DELETE /api/licenses/{uuid}
  ← Updated LicenseResponse (status: "cancelling")
  → refresh license list
```

---

## Out of Scope

- Immediate cancellation (cancel at period end only)
- Refund flows
- Admin cancel (existing revoke endpoint covers that)
