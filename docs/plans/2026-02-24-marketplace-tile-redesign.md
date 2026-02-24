# Marketplace Tile Redesign — Design Document

**Date:** 2026-02-24
**Route:** `/dashboard/marketplace`

---

## Overview

Redesign the marketplace strategy listing tiles to surface key financial metrics and pricing directly on the card, enabling users to evaluate strategies at a glance without navigating to the detail page. Add a featured strategy slot at the top of the grid.

---

## Goals

- Show backtest annual return, max drawdown, rating, subscriber count, strategy type, strategy name, and monthly/annual pricing on each tile
- Compute pricing server-side (no per-tile API calls)
- Auto-select the best-performing strategy as a featured hero card above the main grid
- Preserve existing search, filter, sort, and pagination behaviour

---

## Tile Design

### Layout (metrics-first card)

```
┌─────────────────────────────────────────────────┐
│ [Trend Following]              ★★★★☆  4.2  (7) │  ← type badge + star rating
│                                                 │
│ Alpha Momentum Strategy                         │  ← strategy name (bold, lg)
│                                                 │
│  ┌──────────────────┐  ┌──────────────────┐    │
│  │    +142.3%       │  │    -8.4%         │    │  ← metric panels
│  │  Annual Return   │  │  Max Drawdown    │    │
│  └──────────────────┘  └──────────────────┘    │
│                                                 │
│  👥 127 subscribers                             │
│                                                 │
│  ─────────────────────────────────────────────  │
│  $49 /mo · $490 /yr    [ View Details → ]      │
└─────────────────────────────────────────────────┘
```

### Metric panel colours
- **Annual Return:** `text-green-400` if positive, `text-red-400` if negative. `N/A` in `text-text-secondary` if absent.
- **Max Drawdown:** always `text-red-400`. `N/A` in `text-text-secondary` if absent.
- Panel background: `bg-background rounded-lg p-3` (darker than card surface)

### Price strip
- Separator: `border-t border-border`
- Monthly price: `text-primary font-semibold` (`$X /mo`)
- Annual price: `text-text-secondary` (`$Y /yr`)
- CTA button: `bg-primary hover:bg-primary-hover text-white` — full-width on mobile, `w-auto` on desktop aligned right

### Backtest field extraction
Extract from `backtest_results` dict:
- Annual return: `backtest_results.annual_return` (fallback: `backtest_results.total_return`)
- Max drawdown: `backtest_results.max_drawdown`

---

## Featured Strategy Slot

Above the main grid, display one wider card for the auto-selected "featured" strategy.

**Selection logic (frontend):** From the current page results, pick the strategy with the highest composite score:
```
score = (annual_return / 100) + (rating * 20) + log10(subscriber_count + 1) * 10
```

**Featured card layout:**
- Full-width (spans all columns) with a subtle `border-primary/40` border and `bg-primary/5` tint
- "Featured Strategy" badge in top-left corner
- Same metric panels as regular tiles but larger (`text-3xl`)
- Short description excerpt (2 lines, clamped)
- Pricing strip + "View Details" button

The featured card only renders when there are ≥ 2 strategies on the page (to avoid showing a lone result as "featured").

---

## API Changes

### `StrategyResponse` schema — computed pricing fields

Add two Pydantic `computed_field` properties to `backend/app/schemas/strategies.py`:

```python
from pydantic import computed_field
from backend.app.services.pricing import calculate_license_price

@computed_field
@property
def monthly_price(self) -> Optional[int]:
    if not self.backtest_results:
        return None
    monthly, _, _ = calculate_license_price(self.backtest_results)
    return monthly

@computed_field
@property
def annual_price(self) -> Optional[int]:
    if not self.backtest_results:
        return None
    _, annual, _ = calculate_license_price(self.backtest_results)
    return annual
```

This means `monthly_price` and `annual_price` are automatically included in every `StrategyResponse` (marketplace, detail, admin) with no router changes.

**Note:** Check Pydantic version — `computed_field` requires Pydantic v2. If the project uses v1, use a `@validator` / `root_validator` populating the fields instead, or compute in the router.

### Frontend `Strategy` interface additions

```typescript
interface Strategy {
  // existing fields ...
  backtest_results?: {
    annual_return?: number
    total_return?: number
    max_drawdown?: number
    [key: string]: any
  }
  monthly_price?: number
  annual_price?: number
}
```

---

## Files to Change

| File | Change |
|---|---|
| `backend/app/schemas/strategies.py` | Add `monthly_price` and `annual_price` computed fields to `StrategyResponse` |
| `frontend/app/dashboard/marketplace/page.tsx` | Full tile redesign + featured slot |

No new files needed. No other routes or components are affected.

---

## Out of Scope

- Marketplace detail page (`/dashboard/marketplace/[id]`) — not changed
- Admin marketplace management — not changed
- Filtering/sorting controls — unchanged
- Pagination — unchanged
