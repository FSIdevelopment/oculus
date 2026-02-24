# Marketplace Detail Page Redesign — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign the marketplace strategy detail page to show the full strategy (KPI strip, equity curve, config, trade log), add a subscribe modal using the existing LicensePaymentForm, show an inline license card on successful payment, and add "Subscribed" badges to both the detail page header and the marketplace listing tiles.

**Architecture:** Two frontend-only tasks. Task 1 fully rewrites `frontend/app/dashboard/marketplace/[id]/page.tsx`. Task 2 makes targeted additions to `frontend/app/dashboard/marketplace/page.tsx`. No backend changes needed — pricing comes from the `monthly_price`/`annual_price` computed fields already on `StrategyResponse`, and subscription status is detected by calling the existing `GET /api/users/me/licenses` endpoint.

**Tech Stack:** Next.js 14 App Router, TypeScript, Tailwind CSS (dark mode, CSS-variable colours), Stripe Elements via `LicensePaymentForm` component, inline SVG equity curve.

---

## Context

**Key files:**
- Detail page to rewrite: `frontend/app/dashboard/marketplace/[id]/page.tsx`
- Listing page to update: `frontend/app/dashboard/marketplace/page.tsx`
- LicensePaymentForm (used unchanged): `frontend/components/LicensePaymentForm.tsx`
- API functions: `frontend/lib/api.ts`
  - `marketplaceAPI.getMarketplaceStrategy(id)` → strategy with `backtest_results`, `config`, `monthly_price`, `annual_price`
  - `ratingsAPI.getStrategyRatings(id)` → `{ ratings: Rating[] }`
  - `licenseAPI.listLicenses(0, 100)` → `{ licenses: License[] }` — each has `strategy_id`, `status`, `expires_at`, `webhook_url`
  - `licenseAPI.updateWebhook(licenseId, webhookUrl)` → saves webhook URL on active license

**Design doc:** `docs/plans/2026-02-24-marketplace-detail-redesign.md`

**Tailwind theme tokens:** `bg-surface`, `bg-background`, `bg-primary`, `text-text`, `text-text-secondary`, `border-border`, `text-primary`, `bg-surface-hover`, `hover:bg-primary-hover`

**Subscribed badge style (used in both tasks):**
```tsx
<span className="inline-flex items-center gap-1 bg-green-500/20 text-green-400 border border-green-500/30 rounded px-2 py-0.5 text-xs font-medium">
  ✓ Subscribed
</span>
```

---

## Task 1: Rewrite marketplace detail page

**Files:**
- Modify: `frontend/app/dashboard/marketplace/[id]/page.tsx`

This is a complete rewrite. No tests needed — it is a pure UI render with no extractable logic. Verify manually by running `npm run dev` in the `frontend/` directory and viewing the page.

### Step 1: Replace the file with the new implementation

Replace the entire contents of `frontend/app/dashboard/marketplace/[id]/page.tsx` with the following:

```tsx
'use client'

import { useEffect, useState, useCallback } from 'react'
import { useParams } from 'next/navigation'
import Link from 'next/link'
import { marketplaceAPI, ratingsAPI, licenseAPI } from '@/lib/api'
import { useAuth } from '@/contexts/AuthContext'
import { ArrowLeft, Loader2, Users, CheckCircle2, CalendarDays } from 'lucide-react'
import LicensePaymentForm from '@/components/LicensePaymentForm'

// ─── Interfaces ───────────────────────────────────────────────────────────────

interface BacktestResults {
  total_return_pct?: number
  total_return?: number
  annual_return?: number
  max_drawdown_pct?: number
  max_drawdown?: number
  sharpe_ratio?: number
  win_rate_pct?: number
  win_rate?: number
  profit_factor?: number
  total_trades?: number
  winning_trades?: number
  losing_trades?: number
  avg_trade_pnl?: number
  best_trade_pnl?: number
  worst_trade_pnl?: number
  avg_trade_duration_hours?: number
  trades?: Trade[]
  equity_curve?: EquityPoint[]
}

interface Trade {
  entry_date: string
  exit_date?: string
  symbol: string
  action: string
  entry_price: number
  exit_price?: number
  pnl: number
  pnl_pct: number
  duration_hours: number
  exit_reason?: string
}

interface EquityPoint {
  date: string
  value: number
}

interface Strategy {
  uuid: string
  name: string
  description?: string
  strategy_type?: string
  symbols?: Record<string, any>
  target_return?: number
  backtest_results?: BacktestResults
  config?: Record<string, any>
  subscriber_count: number
  rating?: number
  monthly_price?: number
  annual_price?: number
  created_at: string
  user_id: string
}

interface License {
  uuid: string
  status: string
  license_type: string
  strategy_id: string
  webhook_url?: string
  expires_at: string
}

interface Rating {
  uuid: string
  user_id: string
  rating: number
  review_text?: string
  created_at: string
}

// ─── EquityCurve SVG ──────────────────────────────────────────────────────────

function EquityCurve({ data }: { data: EquityPoint[] }) {
  if (!data || data.length < 2) return (
    <div className="flex items-center justify-center h-40 text-text-secondary text-sm">No equity data</div>
  )
  const W = 800, H = 220, PAD = { top: 10, right: 10, bottom: 30, left: 70 }
  const values = data.map(d => d.value)
  const minV = Math.min(...values), maxV = Math.max(...values)
  const range = maxV - minV || 1
  const xScale = (i: number) => PAD.left + (i / (data.length - 1)) * (W - PAD.left - PAD.right)
  const yScale = (v: number) => PAD.top + (1 - (v - minV) / range) * (H - PAD.top - PAD.bottom)
  const pts = data.map((d, i) => `${xScale(i)},${yScale(d.value)}`).join(' ')
  const isUp = values[values.length - 1] >= values[0]
  const color = isUp ? '#22c55e' : '#ef4444'
  const labelIdxs = [0, Math.floor(data.length * 0.25), Math.floor(data.length * 0.5), Math.floor(data.length * 0.75), data.length - 1]

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: 220 }}>
      {[0, 0.25, 0.5, 0.75, 1].map(t => {
        const y = PAD.top + t * (H - PAD.top - PAD.bottom)
        const v = maxV - t * range
        return (
          <g key={t}>
            <line x1={PAD.left} y1={y} x2={W - PAD.right} y2={y} stroke="#374151" strokeDasharray="4,4" />
            <text x={PAD.left - 6} y={y + 4} textAnchor="end" fontSize={10} fill="#6b7280">
              ${(v / 1000).toFixed(0)}k
            </text>
          </g>
        )
      })}
      {labelIdxs.map(i => (
        <text key={i} x={xScale(i)} y={H - 6} textAnchor="middle" fontSize={10} fill="#6b7280">
          {data[i]?.date?.slice(5) ?? ''}
        </text>
      ))}
      <polygon
        points={`${xScale(0)},${H - PAD.bottom} ${pts} ${xScale(data.length - 1)},${H - PAD.bottom}`}
        fill={color} fillOpacity={0.1}
      />
      <polyline points={pts} fill="none" stroke={color} strokeWidth={2} />
    </svg>
  )
}

// ─── WinLossBar ───────────────────────────────────────────────────────────────

function WinLossBar({ wins, losses }: { wins: number; losses: number }) {
  const total = wins + losses || 1
  const winPct = Math.round((wins / total) * 100)
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm text-text-secondary mb-1">
        <span>Win Rate</span>
        <span className="text-green-400 font-semibold">{winPct}%</span>
      </div>
      <div className="flex h-4 rounded overflow-hidden">
        <div className="bg-green-500" style={{ width: `${winPct}%` }} />
        <div className="bg-red-500 flex-1" />
      </div>
      <div className="flex justify-between text-xs text-text-secondary">
        <span>{wins} wins</span>
        <span>{losses} losses</span>
      </div>
    </div>
  )
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function MarketplaceStrategyPage() {
  const params = useParams()
  const { user } = useAuth()
  const strategyId = params.id as string

  const [strategy, setStrategy] = useState<Strategy | null>(null)
  const [ratings, setRatings] = useState<Rating[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  // License state
  const [activeLicense, setActiveLicense] = useState<License | null>(null)
  const [webhookUrl, setWebhookUrl] = useState('')
  const [savingWebhook, setSavingWebhook] = useState(false)

  // Subscribe modal state
  const [showSubscribeModal, setShowSubscribeModal] = useState(false)
  const [selectedPlan, setSelectedPlan] = useState<'monthly' | 'annual'>('monthly')

  // Rating state
  const [ratingScore, setRatingScore] = useState(0)
  const [ratingReview, setRatingReview] = useState('')
  const [submittingRating, setSubmittingRating] = useState(false)

  // Trade sort + pagination state
  const [tradeSort, setTradeSort] = useState<{ col: keyof Trade; asc: boolean }>({ col: 'entry_date', asc: false })
  const [tradePage, setTradePage] = useState(0)
  const tradesPerPage = 20

  const loadLicenses = useCallback(async () => {
    try {
      const data = await licenseAPI.listLicenses(0, 100)
      const licenses: License[] = data.licenses || []
      const active = licenses.find(l => l.strategy_id === strategyId && l.status === 'active') ?? null
      setActiveLicense(active)
      setWebhookUrl(active?.webhook_url ?? '')
    } catch {
      // User may not be authenticated — treat as not subscribed
    }
  }, [strategyId])

  const loadData = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const [strategyData, ratingsData] = await Promise.all([
        marketplaceAPI.getMarketplaceStrategy(strategyId),
        ratingsAPI.getStrategyRatings(strategyId),
      ])
      setStrategy(strategyData)
      setRatings(ratingsData.ratings || [])
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to load strategy')
    } finally {
      setLoading(false)
    }
  }, [strategyId])

  useEffect(() => {
    loadData()
    loadLicenses()
  }, [loadData, loadLicenses])

  const handleSubscribeSuccess = async () => {
    setShowSubscribeModal(false)
    await loadLicenses()
  }

  const handleSaveWebhook = async () => {
    if (!activeLicense) return
    setSavingWebhook(true)
    try {
      await licenseAPI.updateWebhook(activeLicense.uuid, webhookUrl)
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to save webhook URL')
    } finally {
      setSavingWebhook(false)
    }
  }

  const handleSubmitRating = async () => {
    if (!user) { alert('Please log in to rate this strategy'); return }
    if (ratingScore === 0) { alert('Please select a rating'); return }
    setSubmittingRating(true)
    try {
      await ratingsAPI.createRating(strategyId, ratingScore, ratingReview)
      setRatingScore(0)
      setRatingReview('')
      loadData()
    } catch (err: any) {
      alert(err.response?.data?.detail || err.message || 'Failed to submit rating')
    } finally {
      setSubmittingRating(false)
    }
  }

  const renderStars = (rating?: number) => {
    if (!rating) return <span className="text-text-secondary text-sm">No ratings</span>
    const stars = Math.round(rating)
    return (
      <div className="flex items-center gap-1">
        {[...Array(5)].map((_, i) => (
          <span key={i} className={i < stars ? 'text-yellow-400' : 'text-gray-600'}>★</span>
        ))}
        <span className="text-text-secondary text-sm ml-1">({rating.toFixed(1)})</span>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="animate-spin text-text-secondary" size={24} />
      </div>
    )
  }

  if (error || !strategy) {
    return (
      <div className="space-y-4">
        <Link href="/dashboard/marketplace" className="flex items-center gap-2 text-primary hover:text-primary-hover">
          <ArrowLeft size={20} />
          Back to Marketplace
        </Link>
        <div className="bg-red-900/20 border border-red-700 rounded-lg p-4 text-red-400">
          {error || 'Strategy not found'}
        </div>
      </div>
    )
  }

  const br = strategy.backtest_results ?? {}
  const totalReturn = br.total_return_pct ?? br.total_return ?? null
  const maxDrawdown = br.max_drawdown_pct ?? br.max_drawdown ?? null
  const winRate = br.win_rate_pct ?? br.win_rate ?? null
  const hasKpis = totalReturn != null || br.total_trades != null || winRate != null ||
    br.sharpe_ratio != null || br.profit_factor != null || maxDrawdown != null
  const hasSecondary = br.best_trade_pnl != null || br.worst_trade_pnl != null ||
    br.avg_trade_pnl != null || br.avg_trade_duration_hours != null

  return (
    <div className="space-y-6">
      {/* Back Button */}
      <Link href="/dashboard/marketplace" className="flex items-center gap-2 text-text-secondary hover:text-text transition-colors">
        <ArrowLeft size={18} />
        Back to Marketplace
      </Link>

      {/* Header Card */}
      <div className="bg-surface border border-border rounded-lg p-6">
        {/* Badges row */}
        <div className="flex items-center gap-2 flex-wrap mb-3">
          {strategy.strategy_type && (
            <span className="inline-block bg-primary/20 text-primary text-xs px-2 py-1 rounded font-medium">
              {strategy.strategy_type}
            </span>
          )}
          {activeLicense && (
            <span className="inline-flex items-center gap-1 bg-green-500/20 text-green-400 border border-green-500/30 rounded px-2 py-0.5 text-xs font-medium">
              <CheckCircle2 size={11} />
              Subscribed
            </span>
          )}
          <div className="ml-auto flex-shrink-0">{renderStars(strategy.rating)}</div>
        </div>

        {/* Name + description */}
        <h1 className="text-3xl font-bold text-text mb-2">{strategy.name}</h1>
        {strategy.description && (
          <p className="text-text-secondary text-sm mb-4">{strategy.description}</p>
        )}

        {/* Metadata */}
        <div className="flex items-center gap-4 text-sm text-text-secondary mb-5 flex-wrap">
          <span className="flex items-center gap-1.5">
            <Users size={14} />
            {strategy.subscriber_count} subscribers
          </span>
          <span className="flex items-center gap-1.5">
            <CalendarDays size={14} />
            Created {new Date(strategy.created_at).toLocaleDateString()}
          </span>
        </div>

        {/* Subscribe CTA / License card */}
        <div className="pt-4 border-t border-border">
          {activeLicense ? (
            <div className="bg-green-900/20 border border-green-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle2 size={15} className="text-green-400" />
                <span className="text-sm font-semibold text-green-400">Active License</span>
                <span className="ml-auto text-xs text-green-400/70 capitalize">{activeLicense.license_type}</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-text-secondary mb-3">
                <CalendarDays size={12} />
                <span>Renews {new Date(activeLicense.expires_at).toLocaleDateString()}</span>
                <span className="ml-auto text-text-secondary/50">ID: {activeLicense.uuid.slice(0, 8)}…</span>
              </div>
              <div className="flex gap-2">
                <input
                  type="url"
                  value={webhookUrl}
                  onChange={e => setWebhookUrl(e.target.value)}
                  placeholder="Webhook URL (optional)"
                  className="flex-1 bg-surface border border-border rounded px-3 py-1.5 text-sm text-text placeholder-text-secondary focus:outline-none focus:border-primary"
                />
                <button
                  onClick={handleSaveWebhook}
                  disabled={savingWebhook}
                  className="px-3 py-1.5 text-sm bg-primary hover:bg-primary-hover text-white rounded transition-colors disabled:opacity-50"
                >
                  {savingWebhook ? '…' : 'Save'}
                </button>
              </div>
            </div>
          ) : strategy.monthly_price != null ? (
            <div>
              <p className="text-text-secondary text-sm mb-3">
                <span className="text-primary font-semibold text-lg">${strategy.monthly_price}</span>
                <span className="text-text-secondary"> /mo</span>
                {strategy.annual_price != null && (
                  <span className="text-text-secondary"> · ${strategy.annual_price} /yr</span>
                )}
              </p>
              <div className="flex gap-2 flex-wrap">
                <button
                  onClick={() => { setSelectedPlan('monthly'); setShowSubscribeModal(true) }}
                  className="px-4 py-2 bg-primary hover:bg-primary-hover text-white text-sm font-medium rounded-lg transition-colors"
                >
                  Subscribe Monthly — ${strategy.monthly_price}
                </button>
                {strategy.annual_price != null && (
                  <button
                    onClick={() => { setSelectedPlan('annual'); setShowSubscribeModal(true) }}
                    className="px-4 py-2 bg-surface hover:bg-surface-hover border border-border hover:border-primary text-text text-sm font-medium rounded-lg transition-colors"
                  >
                    Subscribe Annual — ${strategy.annual_price}
                  </button>
                )}
              </div>
            </div>
          ) : (
            <p className="text-text-secondary text-sm">Pricing unavailable</p>
          )}
        </div>
      </div>

      {/* KPI Strip */}
      {hasKpis && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <div className="bg-surface border border-border rounded-lg p-4">
            <p className="text-text-secondary text-xs mb-1">Total Return</p>
            <p className={`text-2xl font-bold ${totalReturn == null ? 'text-text-secondary' : totalReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {totalReturn != null ? `${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(1)}%` : '—'}
            </p>
          </div>
          <div className="bg-surface border border-border rounded-lg p-4">
            <p className="text-text-secondary text-xs mb-1">Total Trades</p>
            <p className="text-2xl font-bold text-text">{br.total_trades ?? '—'}</p>
          </div>
          <div className="bg-surface border border-border rounded-lg p-4">
            <p className="text-text-secondary text-xs mb-1">Win Rate</p>
            <p className="text-2xl font-bold text-text">{winRate != null ? `${winRate.toFixed(1)}%` : '—'}</p>
          </div>
          <div className="bg-surface border border-border rounded-lg p-4">
            <p className="text-text-secondary text-xs mb-1">Sharpe Ratio</p>
            <p className="text-2xl font-bold text-text">{br.sharpe_ratio != null ? br.sharpe_ratio.toFixed(2) : '—'}</p>
          </div>
          <div className="bg-surface border border-border rounded-lg p-4">
            <p className="text-text-secondary text-xs mb-1">Profit Factor</p>
            <p className="text-2xl font-bold text-text">{br.profit_factor != null ? br.profit_factor.toFixed(2) : '—'}</p>
          </div>
          <div className="bg-surface border border-border rounded-lg p-4">
            <p className="text-text-secondary text-xs mb-1">Max Drawdown</p>
            <p className="text-2xl font-bold text-red-400">{maxDrawdown != null ? `-${Math.abs(maxDrawdown).toFixed(1)}%` : '—'}</p>
          </div>
        </div>
      )}

      {/* Secondary Metrics */}
      {hasSecondary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-surface border border-border rounded-lg p-4">
            <p className="text-text-secondary text-xs mb-1">Best Trade</p>
            <p className="text-lg font-bold text-green-400">{br.best_trade_pnl != null ? `$${br.best_trade_pnl.toFixed(2)}` : '—'}</p>
          </div>
          <div className="bg-surface border border-border rounded-lg p-4">
            <p className="text-text-secondary text-xs mb-1">Worst Trade</p>
            <p className="text-lg font-bold text-red-400">{br.worst_trade_pnl != null ? `$${br.worst_trade_pnl.toFixed(2)}` : '—'}</p>
          </div>
          <div className="bg-surface border border-border rounded-lg p-4">
            <p className="text-text-secondary text-xs mb-1">Avg Trade P&L</p>
            <p className={`text-lg font-bold ${(br.avg_trade_pnl ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {br.avg_trade_pnl != null ? `$${br.avg_trade_pnl.toFixed(2)}` : '—'}
            </p>
          </div>
          <div className="bg-surface border border-border rounded-lg p-4">
            <p className="text-text-secondary text-xs mb-1">Avg Duration</p>
            <p className="text-lg font-bold text-text">{br.avg_trade_duration_hours != null ? `${br.avg_trade_duration_hours.toFixed(1)}h` : '—'}</p>
          </div>
        </div>
      )}

      {/* Strategy Configuration */}
      {strategy.config && Object.keys(strategy.config).length > 0 && (
        <div className="bg-surface border border-border rounded-lg p-6">
          <h2 className="text-lg font-semibold text-text mb-4">Strategy Configuration</h2>
          <div className="space-y-6">
            {strategy.config.description && (
              <p className="text-sm text-text-secondary leading-relaxed">{strategy.config.description}</p>
            )}
            {strategy.config.trading && (
              <div>
                <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-3">Trading Parameters</h3>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {([
                    { label: 'Symbols', value: Array.isArray(strategy.config.trading.symbols) ? strategy.config.trading.symbols.join(', ') : strategy.config.trading.symbols },
                    { label: 'Timeframe', value: strategy.config.trading.timeframe },
                    { label: 'Max Positions', value: strategy.config.trading.max_positions },
                    { label: 'Position Size', value: strategy.config.trading.position_size_pct != null ? `${(strategy.config.trading.position_size_pct * 100).toFixed(2)}%` : undefined },
                  ] as { label: string; value: any }[]).filter(item => item.value != null).map(item => (
                    <div key={item.label} className="bg-background rounded-lg p-3 border border-border/50">
                      <p className="text-xs text-text-secondary mb-1">{item.label}</p>
                      <p className="text-sm font-semibold text-text">{String(item.value)}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {strategy.config.indicators && Object.keys(strategy.config.indicators).length > 0 && (
              <div>
                <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-3">Indicators</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {Object.entries(strategy.config.indicators as Record<string, Record<string, any>>).map(([name, params]) => (
                    <div key={name} className="bg-background rounded-lg p-3 border border-border/50">
                      <p className="text-xs font-semibold text-text mb-2 uppercase tracking-wide">{name}</p>
                      <div className="space-y-1">
                        {Object.entries(params).map(([k, v]) => (
                          <div key={k} className="flex justify-between text-xs">
                            <span className="text-text-secondary">{k.replace(/_/g, ' ')}</span>
                            <span className="text-text font-medium">{String(v)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {strategy.config.risk_management && (
              <div>
                <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-3">Risk Management</h3>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {([
                    { label: 'Stop Loss', value: strategy.config.risk_management.stop_loss_pct != null ? `${strategy.config.risk_management.stop_loss_pct}%` : undefined },
                    { label: 'Trailing Stop', value: strategy.config.risk_management.trailing_stop_pct != null ? `${strategy.config.risk_management.trailing_stop_pct}%` : undefined },
                    { label: 'Max Position Loss', value: strategy.config.risk_management.max_position_loss_pct != null ? `${strategy.config.risk_management.max_position_loss_pct}%` : undefined },
                    { label: 'Max Daily Loss', value: strategy.config.risk_management.max_daily_loss_pct != null ? `${strategy.config.risk_management.max_daily_loss_pct}%` : (strategy.config.risk_management.max_daily_loss != null ? `${(strategy.config.risk_management.max_daily_loss * 100).toFixed(1)}%` : undefined) },
                    { label: 'Max Drawdown', value: strategy.config.risk_management.max_drawdown_pct != null ? `${strategy.config.risk_management.max_drawdown_pct}%` : (strategy.config.risk_management.max_drawdown != null ? `${(strategy.config.risk_management.max_drawdown * 100).toFixed(1)}%` : undefined) },
                  ] as { label: string; value: any }[]).filter(item => item.value != null).map(item => (
                    <div key={item.label} className="bg-background rounded-lg p-3 border border-border/50">
                      <p className="text-xs text-text-secondary mb-1">{item.label}</p>
                      <p className="text-sm font-semibold text-red-400">{String(item.value)}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Equity Curve */}
      {br.equity_curve && br.equity_curve.length > 1 && (
        <div className="bg-surface border border-border rounded-lg p-6">
          <h2 className="text-lg font-semibold text-text mb-4">Equity Curve</h2>
          <EquityCurve data={br.equity_curve} />
        </div>
      )}

      {/* Win/Loss Bar */}
      {(br.winning_trades != null || br.losing_trades != null) && (
        <div className="bg-surface border border-border rounded-lg p-6">
          <WinLossBar wins={br.winning_trades ?? 0} losses={br.losing_trades ?? 0} />
        </div>
      )}

      {/* Trade Log */}
      {br.trades && br.trades.length > 0 && (() => {
        const sortedTrades = [...br.trades].sort((a, b) => {
          const va = a[tradeSort.col], vb = b[tradeSort.col]
          if (va == null && vb == null) return 0
          if (va == null) return 1
          if (vb == null) return -1
          if (typeof va === 'number' && typeof vb === 'number') return tradeSort.asc ? va - vb : vb - va
          return tradeSort.asc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va))
        })
        const totalTrades = sortedTrades.length
        const totalPages = Math.ceil(totalTrades / tradesPerPage)
        const startIdx = tradePage * tradesPerPage
        const endIdx = Math.min(startIdx + tradesPerPage, totalTrades)
        const paginatedTrades = sortedTrades.slice(startIdx, endIdx)

        return (
          <div className="bg-surface border border-border rounded-lg p-6">
            <h2 className="text-lg font-semibold text-text mb-4">Trade Log ({totalTrades} trades)</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    {(['entry_date', 'exit_date', 'symbol', 'action', 'entry_price', 'exit_price', 'pnl', 'pnl_pct', 'duration_hours', 'exit_reason'] as (keyof Trade)[]).map(col => (
                      <th
                        key={col}
                        className="text-left py-2 px-3 text-text-secondary font-medium cursor-pointer select-none hover:text-text transition-colors whitespace-nowrap"
                        onClick={() => {
                          setTradeSort(s => s.col === col ? { col, asc: !s.asc } : { col, asc: true })
                          setTradePage(0)
                        }}
                      >
                        {col.replace(/_/g, ' ')}{tradeSort.col === col ? (tradeSort.asc ? ' ↑' : ' ↓') : ''}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {paginatedTrades.map((trade, idx) => (
                    <tr key={idx} className="border-b border-border/50 hover:bg-surface-hover transition-colors">
                      <td className="py-2 px-3 text-text-secondary whitespace-nowrap">{trade.entry_date?.slice(0, 10) ?? '—'}</td>
                      <td className="py-2 px-3 text-text-secondary whitespace-nowrap">{trade.exit_date?.slice(0, 10) ?? '—'}</td>
                      <td className="py-2 px-3 text-text font-medium">{trade.symbol}</td>
                      <td className={`py-2 px-3 font-medium ${trade.action === 'buy' ? 'text-green-400' : 'text-red-400'}`}>{trade.action}</td>
                      <td className="py-2 px-3 text-text">${trade.entry_price?.toFixed(2) ?? '—'}</td>
                      <td className="py-2 px-3 text-text">{trade.exit_price != null ? `$${trade.exit_price.toFixed(2)}` : '—'}</td>
                      <td className={`py-2 px-3 font-medium ${(trade.pnl ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>${trade.pnl?.toFixed(2) ?? '—'}</td>
                      <td className={`py-2 px-3 ${(trade.pnl_pct ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>{trade.pnl_pct?.toFixed(2) ?? '—'}%</td>
                      <td className="py-2 px-3 text-text-secondary">{trade.duration_hours?.toFixed(1) ?? '—'}h</td>
                      <td className="py-2 px-3 text-text-secondary">{trade.exit_reason ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {totalPages > 1 && (
              <div className="flex items-center justify-between mt-4 pt-4 border-t border-border">
                <button
                  onClick={() => setTradePage(Math.max(0, tradePage - 1))}
                  disabled={tradePage === 0}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors ${tradePage === 0 ? 'bg-surface-hover text-text-secondary cursor-not-allowed' : 'bg-surface border border-border hover:border-primary text-text'}`}
                >
                  Previous
                </button>
                <span className="text-text-secondary text-sm">
                  {startIdx + 1}–{endIdx} of {totalTrades} (Page {tradePage + 1} of {totalPages})
                </span>
                <button
                  onClick={() => setTradePage(Math.min(totalPages - 1, tradePage + 1))}
                  disabled={tradePage >= totalPages - 1}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors ${tradePage >= totalPages - 1 ? 'bg-surface-hover text-text-secondary cursor-not-allowed' : 'bg-surface border border-border hover:border-primary text-text'}`}
                >
                  Next
                </button>
              </div>
            )}
          </div>
        )
      })()}

      {/* Ratings & Reviews */}
      <div className="bg-surface border border-border rounded-lg p-6">
        <h2 className="text-lg font-semibold text-text mb-4">Ratings & Reviews</h2>
        {user && (
          <div className="mb-6 p-4 bg-background rounded-lg">
            <p className="text-text text-sm mb-3">Rate this strategy:</p>
            <div className="flex gap-1 mb-3">
              {[1, 2, 3, 4, 5].map(score => (
                <button
                  key={score}
                  onClick={() => setRatingScore(score)}
                  className={`text-2xl transition-colors ${score <= ratingScore ? 'text-yellow-400' : 'text-gray-600 hover:text-yellow-300'}`}
                >
                  ★
                </button>
              ))}
            </div>
            <textarea
              value={ratingReview}
              onChange={e => setRatingReview(e.target.value)}
              placeholder="Write a review (optional)..."
              className="w-full p-2 bg-surface border border-border rounded text-text placeholder-text-secondary focus:outline-none focus:border-primary mb-3"
              rows={3}
            />
            <button
              onClick={handleSubmitRating}
              disabled={submittingRating || ratingScore === 0}
              className="px-4 py-2 bg-primary hover:bg-primary-hover text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
            >
              {submittingRating && <Loader2 size={14} className="animate-spin" />}
              Submit Rating
            </button>
          </div>
        )}
        <div className="space-y-3">
          {ratings.length === 0 ? (
            <p className="text-text-secondary text-sm">No ratings yet</p>
          ) : (
            ratings.map(rating => (
              <div key={rating.uuid} className="p-4 bg-background rounded-lg">
                <div className="flex items-center gap-1 mb-2">
                  {[...Array(5)].map((_, i) => (
                    <span key={i} className={i < rating.rating ? 'text-yellow-400' : 'text-gray-600'}>★</span>
                  ))}
                </div>
                {rating.review_text && <p className="text-text text-sm">{rating.review_text}</p>}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Subscribe Modal */}
      {showSubscribeModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div className="bg-surface border border-border rounded-lg p-6 w-full max-w-md shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-text">Subscribe to {strategy.name}</h2>
              <button
                onClick={() => setShowSubscribeModal(false)}
                className="text-text-secondary hover:text-text text-2xl leading-none w-6 h-6 flex items-center justify-center"
              >
                ×
              </button>
            </div>
            <LicensePaymentForm
              strategyId={strategyId}
              strategyName={strategy.name}
              licenseType={selectedPlan}
              monthlyPrice={strategy.monthly_price ?? 0}
              annualPrice={strategy.annual_price ?? 0}
              onSuccess={handleSubscribeSuccess}
              onCancel={() => setShowSubscribeModal(false)}
            />
          </div>
        </div>
      )}
    </div>
  )
}
```

### Step 2: Check TypeScript compiles

Run from the `frontend/` directory:

```bash
npx tsc --noEmit
```

Expected: no errors. If TypeScript complains about `config` not existing on `StrategyResponse`, that means the backend schema doesn't include `config` in `StrategyResponse`. In that case the `strategy.config` block simply renders nothing (since `strategy.config` will be `undefined`) — no code change needed.

### Step 3: Commit

```bash
git add frontend/app/dashboard/marketplace/[id]/page.tsx
git commit -m "feat: redesign marketplace detail page with full strategy view and subscribe modal"
```

---

## Task 2: Add "Subscribed" badges to marketplace listing page

**Files:**
- Modify: `frontend/app/dashboard/marketplace/page.tsx`

Three targeted edits. No new state shape changes required.

### Step 1: Add `licenseAPI` to the import

Find line 5 in `frontend/app/dashboard/marketplace/page.tsx`:

```ts
import { marketplaceAPI } from '@/lib/api'
```

Replace with:

```ts
import { marketplaceAPI, licenseAPI } from '@/lib/api'
```

### Step 2: Add `subscribedIds` state and load effect

After the existing `const [total, setTotal] = useState(0)` line (around line 150), add:

```ts
const [subscribedIds, setSubscribedIds] = useState<Set<string>>(new Set())
```

After the existing `useEffect(() => { loadStrategies() }, [loadStrategies])` block, add a second independent effect:

```tsx
useEffect(() => {
  licenseAPI.listLicenses(0, 100)
    .then(data => {
      const ids = new Set<string>(
        (data.licenses || [])
          .filter((l: any) => l.status === 'active')
          .map((l: any) => l.strategy_id as string)
      )
      setSubscribedIds(ids)
    })
    .catch(() => { /* user not authenticated or request failed — treat as no subscriptions */ })
}, [])
```

### Step 3: Add "Subscribed" badge to featured card

In the featured card badges area, find this block (inside the `getFeaturedStrategy` IIFE):

```tsx
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
```

Replace with:

```tsx
<div className="flex items-center gap-2 flex-wrap">
  <span className="inline-block bg-primary/20 text-primary text-xs px-2 py-1 rounded font-medium">
    Featured
  </span>
  {featured.strategy_type && (
    <span className="inline-block bg-surface border border-border text-text-secondary text-xs px-2 py-1 rounded">
      {featured.strategy_type}
    </span>
  )}
  {subscribedIds.has(featured.uuid) && (
    <span className="inline-flex items-center gap-1 bg-green-500/20 text-green-400 border border-green-500/30 rounded px-2 py-0.5 text-xs font-medium">
      ✓ Subscribed
    </span>
  )}
</div>
```

### Step 4: Add "Subscribed" badge to each grid tile

Find the tile `<Link>` element (around line 358):

```tsx
<Link
  key={strategy.uuid}
  href={`/dashboard/marketplace/${strategy.uuid}`}
  className="bg-surface border border-border rounded-lg p-6 hover:border-primary transition-colors cursor-pointer flex flex-col"
>
```

Replace with:

```tsx
<Link
  key={strategy.uuid}
  href={`/dashboard/marketplace/${strategy.uuid}`}
  className="bg-surface border border-border rounded-lg p-6 hover:border-primary transition-colors cursor-pointer flex flex-col relative"
>
  {subscribedIds.has(strategy.uuid) && (
    <span className="absolute top-3 right-3 inline-flex items-center gap-1 bg-green-500/20 text-green-400 border border-green-500/30 rounded px-2 py-0.5 text-xs font-medium z-10">
      ✓ Subscribed
    </span>
  )}
```

### Step 5: Check TypeScript compiles

```bash
npx tsc --noEmit
```

Expected: no errors.

### Step 6: Commit

```bash
git add frontend/app/dashboard/marketplace/page.tsx
git commit -m "feat: add subscribed badges to marketplace listing tiles and featured card"
```

---

## Verification

Start the dev server:

```bash
cd frontend && npm run dev
```

Visit `http://localhost:3000/dashboard/marketplace`:
- [ ] Marketplace listing loads
- [ ] Strategies with an active license show a "✓ Subscribed" badge in the top-right corner of the tile
- [ ] The featured strategy card shows a "✓ Subscribed" badge if subscribed

Click a strategy to visit its detail page:
- [ ] Back button returns to `/dashboard/marketplace`
- [ ] Strategy name, type badge, rating, subscriber count, and created date appear in the header
- [ ] If subscribed: green "✓ Subscribed" badge + active license card with webhook field
- [ ] If not subscribed: pricing shown with "Subscribe Monthly" and "Subscribe Annual" buttons
- [ ] "Subscribe Monthly $X" button opens the payment modal pre-selecting monthly
- [ ] "Subscribe Annual $X" button opens the payment modal pre-selecting annual
- [ ] Payment modal shows LicensePaymentForm; `×` closes it
- [ ] After successful payment: modal closes, license card appears inline, subscribe buttons disappear
- [ ] KPI strip shows Total Return, Total Trades, Win Rate, Sharpe Ratio, Profit Factor, Max Drawdown
- [ ] Equity curve renders if `backtest_results.equity_curve` has ≥ 2 points
- [ ] Strategy config section renders indicators, trading params, risk management if `config` is set
- [ ] Trade log renders with pagination and sortable columns if `backtest_results.trades` is set
- [ ] Ratings & Reviews section works: stars selectable, review submittable
