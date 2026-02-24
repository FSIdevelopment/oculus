# Marketplace Detail Page Redesign — Design Document

**Date:** 2026-02-24
**Route:** `/dashboard/marketplace/[id]`

---

## Overview

Redesign the marketplace strategy detail page to show the full strategy — the same rich content as the owner's strategy detail page (KPI strip, equity curve, config, trade log) — alongside marketplace-specific UI: subscriber count, star rating, a subscribe modal with dynamic pricing, an inline license card on success, and a "Subscribed" indicator on both the detail page and the marketplace tiles.

---

## Goals

- Show the complete strategy detail (KPIs, equity curve, config, trade log) to any visitor, subscribed or not
- Detect subscription status client-side by matching user's active licenses against `strategy_id`
- Subscribe flow: click button → modal with plan picker → `LicensePaymentForm` → on success modal closes and inline license card appears
- "Subscribed" badge on the detail page header and on each marketplace listing tile
- No new backend endpoints required

---

## Access Model

All strategy detail sections (KPI, equity curve, config, trade log) are visible to any visitor — no paywall. A license unlocks webhook/deploy rights. This transparency drives subscriptions.

---

## Page Layout

### Header Card

```
┌───────────────────────────────────────────────────────────────┐
│ ← Back to Marketplace                                         │
│                                                               │
│ [Trend Following]  [✓ Subscribed]  (only when subscribed)    │
│                                       ★★★★☆ 4.2 (7 ratings) │
│                                                               │
│ Alpha Momentum Strategy                                       │
│ Short strategy description text                               │
│                                                               │
│ 👥 127 subscribers   📅 Created Jan 2026                      │
│                                                               │
│ ─────────────────────────────────────────────────────────── │
│                                                               │
│ [NOT SUBSCRIBED STATE]                                        │
│  $49 /mo  ·  $490 /yr                                        │
│  [ Subscribe Monthly — $49 ]  [ Subscribe Annual — $490 ]   │
│                                                               │
│ [SUBSCRIBED STATE — replaces above buttons]                   │
│ ┌─ Active License ────────────────────────────────────────┐  │
│ │ Plan: Monthly  ·  Renews: 2026-03-24  ·  ID: abc-123   │  │
│ │ Webhook URL: [___________________________] [ Save ]    │  │
│ └────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

**Subscribed badge:** `bg-green-500/20 text-green-400 border border-green-500/30 rounded px-2 py-0.5 text-xs font-medium` with a checkmark icon.

**Subscribe buttons:** Two side-by-side buttons — one for monthly, one for annual. Each opens the subscribe modal pre-selecting that plan.

**License card:** Shown instead of subscribe buttons when the user has an active license for this strategy. Shows plan type, renewal date, license ID, and an editable webhook URL field.

---

### KPI Strip

6-column grid from `backtest_results`:

| Total Return | Total Trades | Win Rate | Sharpe Ratio | Profit Factor | Max Drawdown |
|---|---|---|---|---|---|
| green/red | neutral | neutral | neutral | neutral | red |

All values show `N/A` in `text-text-secondary` when absent.

---

### Equity Curve

Full-width SVG chart from `backtest_results.equity_curve` (array of `{date, value}`).

- Same SVG technique as the marketplace featured card: polyline + linearGradient fill
- Enhanced for the detail page: taller (240px), horizontal grid lines at 25% intervals, Y-axis labels (e.g. `$100k`), 5 X-axis date labels
- Color: `#4ade80` (green) if total return ≥ 0, `#f87171` (red) otherwise
- Omitted entirely if `equity_curve` has < 2 points

---

### Secondary Metrics

4-column grid (below equity curve):

| Best Trade | Worst Trade | Avg Trade P&L | Avg Duration |
|---|---|---|---|
| green | red | green/red | neutral |

Source: `backtest_results.best_trade`, `.worst_trade`, `.avg_trade_pnl`, `.avg_duration_hours`. Omit row if all four are absent.

---

### Strategy Configuration

From `strategy.config`. Same three-section layout as the strategy detail page:

1. **Trading Parameters** — symbols, timeframe, max positions, position size %
2. **Indicators** — grid of indicator cards, each with name and parameters list
3. **Risk Management** — stop loss %, trailing stop, max position loss %, max daily loss %, max drawdown %

Section omitted if `config` is null/empty.

---

### Trade Log

From `backtest_results.trades` (array of trade objects). Paginated table, 20 trades per page.

Columns: `Date`, `Symbol`, `Action`, `Entry Price`, `Exit Price`, `P&L`, `P&L %`, `Duration`

- `P&L` and `P&L %` coloured green (positive) / red (negative)
- Sortable columns (click header to toggle asc/desc)
- Pagination controls below
- Section omitted entirely if `backtest_results.trades` is absent or empty

---

### Ratings & Reviews

Existing section — kept unchanged:
- Star rating input (1–5 stars) + optional review textarea
- Submit button
- List of existing ratings with review text and score

---

## Subscribe Modal

Triggered by clicking "Subscribe Monthly" or "Subscribe Annual" button. Opens as a centred modal overlay.

### Step 1: Plan picker (skippable — auto-select if user clicked a specific plan button)

```
┌─────────────────────────────────────────────────────────────┐
│  Subscribe to Alpha Momentum Strategy               [×]     │
│                                                             │
│  ┌──────────────────────┐   ┌──────────────────────────┐  │
│  │  Monthly             │   │  Annual                  │  │
│  │  $49 / month         │   │  $490 / year             │  │
│  │  Cancel any time     │   │  ~2 months free          │  │
│  └──────────────────────┘   └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

Selected plan gets a `border-primary` ring. Clicking one moves to Step 2.

### Step 2: Payment form

`LicensePaymentForm` renders inside the same modal with the chosen `licenseType`, `monthlyPrice`, `annualPrice`, `strategyId`, and `strategyName` props. The existing `LicensePaymentForm` component handles the Stripe SetupIntent → `confirmSetup` → `subscribeWithPaymentMethod` flow unchanged.

`onSuccess` callback: closes modal, sets local `activeLicense` state from the returned license object.
`onCancel` callback: returns user to Step 1 (or closes modal if they were already on Step 1).

---

## Subscribed Indicator on Marketplace Tiles

### Detection (marketplace listing page)

On `MarketplacePage` mount, after fetching strategies, also call `GET /api/users/me/licenses` (unauthenticated users skip this call). Build a `Set<string>` of subscribed strategy UUIDs. Pass it as `subscribedIds` prop to each tile/featured card render.

### Tile badge

In the top-right corner of each tile card and the featured card, render a small badge when `subscribedIds.has(strategy.uuid)`:

```
┌──────────────────────────────────────────────────────┐
│ [Trend Following]                   [✓ Subscribed]   │  ← badge top-right
│                                                      │
│ +142.3%          -8.4%                               │
│ Annual Return    Max Drawdown                        │
│ ...                                                  │
└──────────────────────────────────────────────────────┘
```

Badge style: same as detail page header — `bg-green-500/20 text-green-400 border border-green-500/30 text-xs rounded px-2 py-0.5`.

---

## Subscription Status Detection

**Detail page:** On page load, fetch `GET /api/users/me/licenses?limit=100` and find any license where `license.strategy_id === strategy.uuid` and `license.status === "active"`. If found, show the license card; if not, show subscribe buttons.

**Listing page:** Same fetch, build a `Set<string>` from matched strategy IDs, pass to each tile as a prop.

Both pages: If the request fails (e.g. user not logged in), treat as not subscribed — no error shown.

---

## Files to Change

| File | Change |
|---|---|
| `frontend/app/dashboard/marketplace/[id]/page.tsx` | Full rewrite — rich detail layout + subscribe modal + inline license card |
| `frontend/app/dashboard/marketplace/page.tsx` | Fetch user licenses on load; pass `subscribedIds` set to tiles and featured card; render "Subscribed" badge |

No backend changes needed. No new files needed.

---

## Out of Scope

- Renewing an existing license from the marketplace detail page (can be done via `/dashboard/account`)
- Marketplace detail page for the strategy owner (they use `/dashboard/strategies/[id]`)
- Admin marketplace management
