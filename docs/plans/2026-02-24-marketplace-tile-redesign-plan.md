# Marketplace Tile Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign the marketplace tile cards to show annual return, max drawdown, rating, subscriber count, strategy type, name and dynamic pricing; add a featured strategy hero slot above the grid.

**Architecture:** Add Pydantic v2 `computed_field` properties to `StrategyResponse` to auto-compute `monthly_price`/`annual_price` from `backtest_results`; then overhaul the frontend tile component in-place with the new metrics-first layout and a featured card slot rendered above the grid.

**Tech Stack:** Python 3.11 · FastAPI · Pydantic v2.5 · Next.js 14 · TypeScript · Tailwind CSS

---

## Task 1: Add `monthly_price` / `annual_price` computed fields to `StrategyResponse`

**Files:**
- Modify: `backend/app/schemas/strategies.py`
- Test: `backend/tests/test_strategies.py`

### Step 1: Write a failing test

Add this test at the bottom of `backend/tests/test_strategies.py`:

```python
def test_strategy_response_computed_prices_with_backtest():
    """StrategyResponse exposes monthly_price and annual_price when backtest_results present."""
    from app.schemas.strategies import StrategyResponse
    from datetime import datetime

    data = dict(
        uuid="abc",
        name="Test",
        status="complete",
        version=1,
        subscriber_count=0,
        user_id="user-1",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        backtest_results={"total_return": 80.0, "sharpe_ratio": 1.5},
    )
    resp = StrategyResponse(**data)
    assert resp.monthly_price is not None
    assert resp.monthly_price > 0
    assert resp.annual_price is not None
    assert resp.annual_price >= resp.monthly_price


def test_strategy_response_computed_prices_without_backtest():
    """StrategyResponse returns None prices when no backtest_results."""
    from app.schemas.strategies import StrategyResponse
    from datetime import datetime

    data = dict(
        uuid="abc",
        name="Test",
        status="complete",
        version=1,
        subscriber_count=0,
        user_id="user-1",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    resp = StrategyResponse(**data)
    assert resp.monthly_price is None
    assert resp.annual_price is None
```

### Step 2: Run test to verify it fails

```bash
cd /Users/kevingriffiths/Development/OculusStrategy/backend
source ~/Development/virtualenv/signalsynk/bin/activate
python3 -m pytest tests/test_strategies.py::test_strategy_response_computed_prices_with_backtest tests/test_strategies.py::test_strategy_response_computed_prices_without_backtest -v
```

Expected: `FAILED — AttributeError: 'StrategyResponse' object has no attribute 'monthly_price'`

### Step 3: Add the computed fields

Open `backend/app/schemas/strategies.py`. At the top, update the imports line to include `computed_field` and `model_validator`:

**Find:**
```python
from pydantic import BaseModel, Field
```

**Replace with:**
```python
from pydantic import BaseModel, Field, computed_field
from app.services.pricing import calculate_license_price
```

Then find the `StrategyResponse` class and add two `@computed_field` properties just before `class Config`:

**Find:**
```python
    class Config:
        from_attributes = True


class StrategyListResponse(BaseModel):
```

**Replace with:**
```python
    @computed_field  # type: ignore[misc]
    @property
    def monthly_price(self) -> int | None:
        if not self.backtest_results:
            return None
        monthly, _, _ = calculate_license_price(self.backtest_results)
        return monthly

    @computed_field  # type: ignore[misc]
    @property
    def annual_price(self) -> int | None:
        if not self.backtest_results:
            return None
        _, annual, _ = calculate_license_price(self.backtest_results)
        return annual

    class Config:
        from_attributes = True


class StrategyListResponse(BaseModel):
```

### Step 4: Run tests to verify they pass

```bash
python3 -m pytest tests/test_strategies.py::test_strategy_response_computed_prices_with_backtest tests/test_strategies.py::test_strategy_response_computed_prices_without_backtest -v
```

Expected: `2 passed`

### Step 5: Run the full test suite to check for regressions

```bash
python3 -m pytest tests/ -v --timeout=30 -x -q
```

Expected: all existing tests pass (some may be skipped due to missing env vars — that's fine).

### Step 6: Commit

```bash
git add backend/app/schemas/strategies.py backend/tests/test_strategies.py
git commit -m "feat: add computed monthly_price/annual_price to StrategyResponse"
```

---

## Task 2: Redesign the marketplace tile and add featured strategy slot

**Files:**
- Modify: `frontend/app/dashboard/marketplace/page.tsx`

### Step 1: Read the current file

Read `frontend/app/dashboard/marketplace/page.tsx` to confirm current state before editing.

### Step 2: Replace the `Strategy` interface

**Find:**
```typescript
interface Strategy {
  uuid: string
  name: string
  description?: string
  strategy_type?: string
  symbols?: Record<string, any>
  target_return?: number
  backtest_results?: Record<string, any>
  subscriber_count: number
  rating?: number
  created_at: string
}
```

**Replace with:**
```typescript
interface Strategy {
  uuid: string
  name: string
  description?: string
  strategy_type?: string
  symbols?: Record<string, any>
  target_return?: number
  backtest_results?: {
    total_return?: number
    total_return_pct?: number
    annual_return?: number
    max_drawdown?: number
    win_rate?: number
    [key: string]: any
  }
  subscriber_count: number
  rating?: number
  rating_count?: number
  monthly_price?: number
  annual_price?: number
  created_at: string
}
```

### Step 3: Add helper functions after the `renderStars` function

**Find:**
```typescript
  const getSymbolsDisplay = (symbols?: Record<string, any>) => {
    if (!symbols) return 'N/A'
    const symbolList = symbols.symbols || []
    return symbolList.slice(0, 3).join(', ') + (symbolList.length > 3 ? '...' : '')
  }
```

**Replace with:**
```typescript
  const getAnnualReturn = (strategy: Strategy): number | null => {
    const br = strategy.backtest_results
    if (!br) return null
    const val = br.annual_return ?? br.total_return_pct ?? br.total_return
    return val != null ? Number(val) : null
  }

  const getMaxDrawdown = (strategy: Strategy): number | null => {
    const br = strategy.backtest_results
    if (!br) return null
    const val = br.max_drawdown
    return val != null ? Number(val) : null
  }

  const getFeaturedStrategy = (list: Strategy[]): Strategy | null => {
    if (list.length < 2) return null
    return list.reduce((best, s) => {
      const annualReturn = getAnnualReturn(s) ?? 0
      const bestAnnualReturn = getAnnualReturn(best) ?? 0
      const score =
        annualReturn / 100 +
        (s.rating ?? 0) * 20 +
        Math.log10((s.subscriber_count ?? 0) + 1) * 10
      const bestScore =
        bestAnnualReturn / 100 +
        (best.rating ?? 0) * 20 +
        Math.log10((best.subscriber_count ?? 0) + 1) * 10
      return score > bestScore ? s : best
    })
  }
```

### Step 4: Replace the grid section with the new tile design

The section to replace starts at the `{/* Grid */}` comment and ends just before `{/* Pagination */}`.

**Find:**
```typescript
          {/* Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredStrategies.map(strategy => (
              <Link
                key={strategy.uuid}
                href={`/dashboard/marketplace/${strategy.uuid}`}
                className="bg-surface border border-border rounded-lg p-6 hover:border-primary transition-colors cursor-pointer"
              >
                <h3 className="text-lg font-bold text-text mb-2">{strategy.name}</h3>
                <p className="text-text-secondary text-sm mb-4 line-clamp-2">
                  {strategy.description || 'No description'}
                </p>

                {strategy.strategy_type && (
                  <div className="mb-3">
                    <span className="inline-block bg-primary/20 text-primary text-xs px-2 py-1 rounded">
                      {strategy.strategy_type}
                    </span>
                  </div>
                )}

                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-text-secondary">Symbols:</span>
                    <span className="text-text">{getSymbolsDisplay(strategy.symbols)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-secondary">Target Return:</span>
                    <span className="text-text">{strategy.target_return?.toFixed(1) || 'N/A'}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-text-secondary">Rating:</span>
                    {renderStars(strategy.rating)}
                  </div>
                  <div className="flex items-center gap-2 text-text-secondary">
                    <Users size={16} />
                    <span>{strategy.subscriber_count} subscribers</span>
                  </div>
                </div>

                <button className="w-full mt-4 bg-primary hover:bg-primary-hover text-white font-medium py-2 rounded-lg transition-colors">
                  View Details
                </button>
              </Link>
            ))}
          </div>
```

**Replace with:**
```typescript
          {/* Featured Strategy */}
          {(() => {
            const featured = getFeaturedStrategy(filteredStrategies)
            if (!featured) return null
            const annualReturn = getAnnualReturn(featured)
            const maxDrawdown = getMaxDrawdown(featured)
            return (
              <div>
                <p className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-3">Featured Strategy</p>
                <Link
                  href={`/dashboard/marketplace/${featured.uuid}`}
                  className="block bg-primary/5 border border-primary/40 rounded-lg p-6 hover:border-primary transition-colors"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="inline-block bg-primary/20 text-primary text-xs px-2 py-1 rounded font-medium">
                        Featured
                      </span>
                      {featured.strategy_type && (
                        <span className="inline-block bg-surface border border-border text-text-secondary text-xs px-2 py-1 rounded">
                          {featured.strategy_type}
                        </span>
                      )}
                    </div>
                    <div className="flex-shrink-0">{renderStars(featured.rating)}</div>
                  </div>

                  <h3 className="text-xl font-bold text-text mb-3">{featured.name}</h3>

                  <div className="grid grid-cols-2 gap-3 mb-4">
                    <div className="bg-background rounded-lg p-3 text-center">
                      <p className={`text-2xl font-bold ${annualReturn == null ? 'text-text-secondary' : annualReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {annualReturn == null ? 'N/A' : `${annualReturn >= 0 ? '+' : ''}${annualReturn.toFixed(1)}%`}
                      </p>
                      <p className="text-xs text-text-secondary mt-1">Annual Return</p>
                    </div>
                    <div className="bg-background rounded-lg p-3 text-center">
                      <p className={`text-2xl font-bold ${maxDrawdown == null ? 'text-text-secondary' : 'text-red-400'}`}>
                        {maxDrawdown == null ? 'N/A' : `-${Math.abs(maxDrawdown).toFixed(1)}%`}
                      </p>
                      <p className="text-xs text-text-secondary mt-1">Max Drawdown</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 text-text-secondary text-sm mb-4">
                    <Users size={15} />
                    <span>{featured.subscriber_count} subscribers</span>
                  </div>

                  <div className="flex items-center justify-between pt-4 border-t border-border">
                    <div className="flex items-center gap-3">
                      {featured.monthly_price != null ? (
                        <>
                          <span className="text-primary font-semibold">${featured.monthly_price} <span className="text-text-secondary font-normal text-sm">/mo</span></span>
                          {featured.annual_price != null && (
                            <span className="text-text-secondary text-sm">${featured.annual_price} /yr</span>
                          )}
                        </>
                      ) : (
                        <span className="text-text-secondary text-sm">Pricing unavailable</span>
                      )}
                    </div>
                    <span className="text-primary text-sm font-medium">View Details →</span>
                  </div>
                </Link>
              </div>
            )
          })()}

          {/* Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredStrategies.map(strategy => {
              const annualReturn = getAnnualReturn(strategy)
              const maxDrawdown = getMaxDrawdown(strategy)
              return (
                <Link
                  key={strategy.uuid}
                  href={`/dashboard/marketplace/${strategy.uuid}`}
                  className="bg-surface border border-border rounded-lg p-6 hover:border-primary transition-colors cursor-pointer flex flex-col"
                >
                  {/* Top row: type badge + stars */}
                  <div className="flex items-start justify-between mb-2 min-h-[28px]">
                    {strategy.strategy_type ? (
                      <span className="inline-block bg-primary/20 text-primary text-xs px-2 py-1 rounded">
                        {strategy.strategy_type}
                      </span>
                    ) : (
                      <span />
                    )}
                    <div className="flex-shrink-0">{renderStars(strategy.rating)}</div>
                  </div>

                  {/* Strategy name */}
                  <h3 className="text-lg font-bold text-text mb-4">{strategy.name}</h3>

                  {/* Metric panels */}
                  <div className="grid grid-cols-2 gap-2 mb-4">
                    <div className="bg-background rounded-lg p-3 text-center">
                      <p className={`text-xl font-bold ${annualReturn == null ? 'text-text-secondary' : annualReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {annualReturn == null ? 'N/A' : `${annualReturn >= 0 ? '+' : ''}${annualReturn.toFixed(1)}%`}
                      </p>
                      <p className="text-xs text-text-secondary mt-1">Annual Return</p>
                    </div>
                    <div className="bg-background rounded-lg p-3 text-center">
                      <p className={`text-xl font-bold ${maxDrawdown == null ? 'text-text-secondary' : 'text-red-400'}`}>
                        {maxDrawdown == null ? 'N/A' : `-${Math.abs(maxDrawdown).toFixed(1)}%`}
                      </p>
                      <p className="text-xs text-text-secondary mt-1">Max Drawdown</p>
                    </div>
                  </div>

                  {/* Subscriber count */}
                  <div className="flex items-center gap-2 text-text-secondary text-sm mb-4">
                    <Users size={15} />
                    <span>{strategy.subscriber_count} subscribers</span>
                  </div>

                  {/* Price strip + CTA */}
                  <div className="flex items-center justify-between pt-4 border-t border-border mt-auto">
                    <div className="flex items-center gap-2">
                      {strategy.monthly_price != null ? (
                        <>
                          <span className="text-primary font-semibold text-sm">${strategy.monthly_price}<span className="text-text-secondary font-normal"> /mo</span></span>
                          {strategy.annual_price != null && (
                            <span className="text-text-secondary text-xs">· ${strategy.annual_price} /yr</span>
                          )}
                        </>
                      ) : (
                        <span className="text-text-secondary text-xs">—</span>
                      )}
                    </div>
                    <span className="text-primary text-sm font-medium whitespace-nowrap">View Details →</span>
                  </div>
                </Link>
              )
            })}
          </div>
```

### Step 5: Remove unused `getSymbolsDisplay` function and unused import (if any)

After the replacements, the `getSymbolsDisplay` function is no longer used. Remove it:

**Find:**
```typescript
  const getSymbolsDisplay = (symbols?: Record<string, any>) => {
    if (!symbols) return 'N/A'
    const symbolList = symbols.symbols || []
    return symbolList.slice(0, 3).join(', ') + (symbolList.length > 3 ? '...' : '')
  }
```

**Replace with:** (delete the function — replace with empty string)

### Step 6: Manual verification

Start the backend and frontend, navigate to `/dashboard/marketplace`, and verify:

1. The featured strategy card appears above the grid when ≥ 2 results are shown
2. Each tile shows: strategy type badge, star rating, strategy name, annual return panel (green/red), max drawdown panel (red), subscriber count, price strip (`$X /mo · $Y /yr`), "View Details →" link
3. When `backtest_results` is absent the metric panels show "N/A"
4. When `monthly_price` is absent (strategy has no backtest) the price strip shows "—"
5. Clicking any tile navigates to the detail page
6. Search, filter, and sort still work
7. Pagination still works

```bash
# Backend (in one terminal)
cd /Users/kevingriffiths/Development/OculusStrategy/backend
source ~/Development/virtualenv/signalsynk/bin/activate
uvicorn main:app --port 8000 --log-level info

# Frontend (in another terminal)
cd /Users/kevingriffiths/Development/OculusStrategy/frontend
npm run dev
```

Open: http://localhost:3000/dashboard/marketplace

### Step 7: Commit

```bash
git add frontend/app/dashboard/marketplace/page.tsx
git commit -m "feat: redesign marketplace tiles with metrics, pricing and featured slot"
```

---

## Summary of changes

| File | What changes |
|---|---|
| `backend/app/schemas/strategies.py` | +2 `@computed_field` properties on `StrategyResponse` |
| `backend/tests/test_strategies.py` | +2 unit tests for computed price fields |
| `frontend/app/dashboard/marketplace/page.tsx` | Updated `Strategy` interface, new helper functions, new tile layout with featured slot |
