'use client'

import { useEffect, useState, useCallback } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { strategyAPI, buildAPI, authAPI, licenseAPI, marketplaceAPI } from '@/lib/api'
import { ArrowLeft, RefreshCw, ChevronDown } from 'lucide-react'
import StrategyActionModal from '@/components/StrategyActionModal'

interface Strategy {
  uuid: string
  name: string
  description?: string
  status: string
  strategy_type?: string
  symbols?: string[] | Record<string, any>
  target_return?: number
  backtest_results?: Record<string, any>
  marketplace_listed?: boolean
  marketplace_price?: number
  subscriber_count: number
  rating?: number
  version: number
  created_at: string
  updated_at: string
}

interface License {
  uuid: string
  status: string
  license_type: string
  strategy_id: string
  webhook_url?: string
  expires_at: string
}

interface Build {
  uuid: string
  status: string
  phase?: string
  tokens_consumed: number
  iteration_count: number
  started_at: string
  completed_at?: string
  strategy_name?: string
}

interface BacktestResults {
  total_return_pct?: number
  total_return?: number
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
  start_date?: string
  end_date?: string
  period?: string
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

// ─── Equity Curve SVG ────────────────────────────────────────────────────────
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

  // Pick ~5 evenly-spaced date labels
  const labelIdxs = [0, Math.floor(data.length * 0.25), Math.floor(data.length * 0.5), Math.floor(data.length * 0.75), data.length - 1]

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: 220 }}>
      {/* Grid lines */}
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
      {/* Date labels */}
      {labelIdxs.map(i => (
        <text key={i} x={xScale(i)} y={H - 6} textAnchor="middle" fontSize={10} fill="#6b7280">
          {data[i]?.date?.slice(5) ?? ''}
        </text>
      ))}
      {/* Area fill */}
      <polygon
        points={`${xScale(0)},${H - PAD.bottom} ${pts} ${xScale(data.length - 1)},${H - PAD.bottom}`}
        fill={color} fillOpacity={0.1}
      />
      {/* Line */}
      <polyline points={pts} fill="none" stroke={color} strokeWidth={2} />
    </svg>
  )
}

// ─── Win/Loss bar ─────────────────────────────────────────────────────────────
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
export default function StrategyDetailPage() {
  const params = useParams()
  const router = useRouter()
  const strategyId = params.id as string

  const [strategy, setStrategy] = useState<Strategy | null>(null)
  const [builds, setBuilds] = useState<Build[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Version dropdown state
  const [selectedBuildId, setSelectedBuildId] = useState<string | null>(null)
  const [versionBacktest, setVersionBacktest] = useState<BacktestResults | null>(null)
  const [versionConfig, setVersionConfig] = useState<Record<string, any> | null>(null)
  const [versionLoading, setVersionLoading] = useState(false)

  // Retrain modal state
  const [retraining, setRetraining] = useState(false)
  const [showRetrainModal, setShowRetrainModal] = useState(false)
  const [retrainTargetReturn, setRetrainTargetReturn] = useState<number>(10)
  const [retrainMaxIterations, setRetrainMaxIterations] = useState<number>(5)
  const [tokensPerIteration, setTokensPerIteration] = useState<number | null>(null)
  const [userBalance, setUserBalance] = useState<number | null>(null)

  // Deploy modal state
  const [isDeployModalOpen, setIsDeployModalOpen] = useState(false)
  const [licenses, setLicenses] = useState<License[]>([])

  // Trade table sort state
  const [tradeSort, setTradeSort] = useState<{ col: keyof Trade; asc: boolean }>({ col: 'entry_date', asc: false })

  // Trade pagination state
  const [tradePage, setTradePage] = useState(0)
  const tradesPerPage = 20

  const loadStrategy = useCallback(async () => {
    try {
      const data = await strategyAPI.getStrategy(strategyId)
      setStrategy(data)
    } catch (err) {
      setError('Failed to load strategy')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }, [strategyId])

  const loadBuilds = useCallback(async () => {
    try {
      const data = await strategyAPI.getStrategyBuilds(strategyId, 0, 50)
      const items: Build[] = data.items || []
      setBuilds(items)
      // Default to most recent completed build for the version dropdown
      const completed = items.filter(b => b.status === 'complete')
      if (completed.length > 0 && !selectedBuildId) {
        setSelectedBuildId(completed[completed.length - 1].uuid)
      }
    } catch (err) {
      console.error('Failed to load builds:', err)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [strategyId])

  const loadLicenses = async () => {
    try {
      const data = await licenseAPI.listLicenses(0, 100)
      setLicenses(data.licenses || [])
    } catch (err) {
      // Licenses are non-critical; log but don't surface to user
      console.error('Failed to load licenses:', err)
    }
  }

  useEffect(() => {
    loadStrategy()
    loadBuilds()
    loadLicenses()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadStrategy, loadBuilds])

  // Load backtest + config for the selected version whenever it changes
  useEffect(() => {
    if (!selectedBuildId) return
    setVersionLoading(true)
    setTradePage(0) // Reset trade pagination when version changes
    Promise.all([
      strategyAPI.getBuildBestBacktest(strategyId, selectedBuildId).catch(() => null),
      strategyAPI.getBuildConfig(strategyId, selectedBuildId).catch(() => null),
    ]).then(([btRes, cfgRes]) => {
      setVersionBacktest(btRes?.backtest_results ?? null)
      setVersionConfig(cfgRes?.config ?? null)
    }).finally(() => setVersionLoading(false))
  }, [selectedBuildId, strategyId])

  const handleOpenRetrainModal = async () => {
    if (!strategy) return
    // Pre-fill target return from strategy if available
    setRetrainTargetReturn(strategy.target_return ?? 10)
    setRetrainMaxIterations(5)
    setUserBalance(null)
    setShowRetrainModal(true)
    // Fetch per-iteration cost and current balance in parallel
    try {
      const [pricing, me] = await Promise.all([
        buildAPI.getBuildPricing(),
        authAPI.getMe(),
      ])
      setTokensPerIteration(pricing.tokens_per_iteration)
      setUserBalance(me.balance ?? null)
    } catch {
      setTokensPerIteration(null)
      setUserBalance(null)
    }
  }

  const handleRetrainConfirm = async () => {
    if (!strategy) return
    setRetraining(true)
    try {
      const build = await strategyAPI.retrainStrategy(strategyId, {
        target_return: retrainTargetReturn,
        max_iterations: retrainMaxIterations,
      })
      setShowRetrainModal(false)
      router.push(`/dashboard/build?buildId=${build.uuid}`)
    } catch (err: any) {
      alert(err?.response?.data?.detail || 'Failed to start retrain')
      setRetraining(false)
    }
  }

  /**
   * Opens the deploy/action modal and ensures licenses are up-to-date.
   */
  const handleOpenDeployModal = async () => {
    await loadLicenses()
    setIsDeployModalOpen(true)
  }

  /**
   * Called by StrategyActionModal after the in-app license payment succeeds.
   * Refreshes the licenses list so the new "Active License" badge is shown
   * immediately without requiring the user to close and reopen the modal.
   */
  const handleLicensePurchase = async () => {
    await loadLicenses()
  }

  /**
   * Saves an updated webhook URL against the user's active license.
   */
  const handleWebhookUpdate = async (licenseId: string, webhookUrl: string) => {
    await licenseAPI.updateWebhook(licenseId, webhookUrl)
    await loadLicenses()
  }

  /**
   * Toggles the marketplace listing for the current strategy.
   */
  const handleMarketplaceSubmit = async (price: number) => {
    if (!strategy) return
    const currentlyListed = strategy.marketplace_listed ?? false
    await marketplaceAPI.updateMarketplaceListing(strategy.uuid, !currentlyListed, currentlyListed ? undefined : price)
    setIsDeployModalOpen(false)
    loadStrategy()
  }

  const getStatusColor = (s: string) => {
    switch (s) {
      case 'complete': return 'bg-green-900/30 text-green-400 border-green-700'
      case 'building': case 'queued': return 'bg-yellow-900/30 text-yellow-400 border-yellow-700'
      case 'failed': return 'bg-red-900/30 text-red-400 border-red-700'
      default: return 'bg-gray-900/30 text-gray-400 border-gray-700'
    }
  }

  // Completed builds sorted oldest→newest, so Version 1 = first build
  const completedBuilds = builds
    .filter(b => b.status === 'complete')
    .sort((a, b) => new Date(a.started_at).getTime() - new Date(b.started_at).getTime())

  // Display backtest: prefer version-specific results, fall back to strategy.backtest_results
  const displayBacktest: BacktestResults | null = versionBacktest ?? strategy?.backtest_results ?? null

  // Normalise fields that exist in both old and new formats
  const totalReturn = displayBacktest?.total_return_pct ?? displayBacktest?.total_return ?? null
  const maxDrawdown = displayBacktest?.max_drawdown_pct ?? displayBacktest?.max_drawdown ?? null
  const winRate = displayBacktest?.win_rate_pct ?? displayBacktest?.win_rate ?? null

  // Extract symbols list
  const symbolsList: string[] = Array.isArray(strategy?.symbols)
    ? strategy.symbols
    : strategy?.symbols
      ? Object.keys(strategy.symbols)
      : []

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-text-secondary">Loading strategy...</div>
      </div>
    )
  }

  if (error || !strategy) {
    return (
      <div className="space-y-4">
        <button
          onClick={() => router.back()}
          className="flex items-center gap-2 text-primary hover:underline"
        >
          <ArrowLeft size={18} />
          Back
        </button>
        <div className="bg-red-900/20 border border-red-700 rounded-lg p-4 text-red-400">
          {error || 'Strategy not found'}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Back button */}
      <button
        onClick={() => router.back()}
        className="flex items-center gap-2 text-text-secondary hover:text-text transition-colors"
      >
        <ArrowLeft size={18} />
        Back to Strategies
      </button>

      {/* Header card */}
      <div className="bg-surface border border-border rounded-lg p-6">
        <div className="flex items-start justify-between gap-4 mb-4">
          <div className="min-w-0">
            <h1 className="text-3xl font-bold text-text mb-1">{strategy.name}</h1>
            {strategy.description && (
              <p className="text-text-secondary text-sm">{strategy.description}</p>
            )}
            {symbolsList.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-3">
                {symbolsList.map(sym => (
                  <span key={sym} className="bg-surface-hover border border-border text-xs px-2 py-0.5 rounded text-text-secondary">{sym}</span>
                ))}
              </div>
            )}
          </div>
          <div className="flex flex-col items-end gap-3 flex-shrink-0">
            <span className={`text-xs font-medium px-3 py-1 rounded border ${getStatusColor(strategy.status)}`}>
              {strategy.status}
            </span>
            <div className="flex gap-2">
              <button
                onClick={handleOpenRetrainModal}
                disabled={retraining}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium bg-surface-hover hover:bg-border text-text border border-border rounded-lg transition-colors disabled:opacity-50"
              >
                <RefreshCw size={15} className={retraining ? 'animate-spin' : ''} />
                {retraining ? 'Retraining…' : 'Retrain'}
              </button>
              <button
                onClick={handleOpenDeployModal}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium bg-primary hover:bg-primary-hover text-white rounded-lg transition-colors"
              >
                Deploy
              </button>
            </div>
          </div>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-border text-sm">
          <div>
            <p className="text-text-secondary mb-0.5">Type</p>
            <p className="text-text font-medium">{strategy.strategy_type || 'N/A'}</p>
          </div>
          <div>
            <p className="text-text-secondary mb-0.5">Version</p>
            <p className="text-text font-medium">v{strategy.version}</p>
          </div>
          <div>
            <p className="text-text-secondary mb-0.5">Subscribers</p>
            <p className="text-text font-medium">{strategy.subscriber_count}</p>
          </div>
          <div>
            <p className="text-text-secondary mb-0.5">Last Updated</p>
            <p className="text-text font-medium">{new Date(strategy.updated_at).toLocaleDateString()}</p>
          </div>
        </div>
      </div>

      {/* Version selector */}
      {completedBuilds.length > 0 && (
        <div className="flex items-center gap-3">
          <span className="text-sm text-text-secondary">Viewing data for:</span>
          <div className="relative">
            <select
              value={selectedBuildId ?? ''}
              onChange={e => setSelectedBuildId(e.target.value)}
              className="appearance-none bg-surface border border-border rounded-lg px-4 py-2 pr-8 text-sm text-text focus:outline-none focus:border-primary cursor-pointer"
            >
              {completedBuilds.map((b, idx) => (
                <option key={b.uuid} value={b.uuid}>Version {idx + 1}</option>
              ))}
            </select>
            <ChevronDown size={14} className="absolute right-2 top-1/2 -translate-y-1/2 text-text-secondary pointer-events-none" />
          </div>
          {versionLoading && <span className="text-xs text-text-secondary animate-pulse">Loading…</span>}
        </div>
      )}

      {/* KPI Strip */}
      {displayBacktest ? (
        <>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <div className="bg-surface border border-border rounded-lg p-4">
              <p className="text-text-secondary text-xs mb-1">Total Return</p>
              <p className={`text-2xl font-bold ${(totalReturn ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {totalReturn != null ? `${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(1)}%` : '—'}
              </p>
            </div>
            <div className="bg-surface border border-border rounded-lg p-4">
              <p className="text-text-secondary text-xs mb-1">Total Trades</p>
              <p className="text-2xl font-bold text-text">{displayBacktest.total_trades ?? '—'}</p>
            </div>
            <div className="bg-surface border border-border rounded-lg p-4">
              <p className="text-text-secondary text-xs mb-1">Win Rate</p>
              <p className="text-2xl font-bold text-text">{winRate != null ? `${winRate.toFixed(1)}%` : '—'}</p>
            </div>
            <div className="bg-surface border border-border rounded-lg p-4">
              <p className="text-text-secondary text-xs mb-1">Sharpe Ratio</p>
              <p className="text-2xl font-bold text-text">{displayBacktest.sharpe_ratio != null ? displayBacktest.sharpe_ratio.toFixed(2) : '—'}</p>
            </div>
            <div className="bg-surface border border-border rounded-lg p-4">
              <p className="text-text-secondary text-xs mb-1">Profit Factor</p>
              <p className="text-2xl font-bold text-text">{displayBacktest.profit_factor != null ? displayBacktest.profit_factor.toFixed(2) : '—'}</p>
            </div>
            <div className="bg-surface border border-border rounded-lg p-4">
              <p className="text-text-secondary text-xs mb-1">Max Drawdown</p>
              <p className="text-2xl font-bold text-red-400">{maxDrawdown != null ? `-${Math.abs(maxDrawdown).toFixed(1)}%` : '—'}</p>
            </div>
          </div>

          {/* Secondary Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-surface border border-border rounded-lg p-4">
              <p className="text-text-secondary text-xs mb-1">Best Trade</p>
              <p className="text-lg font-bold text-green-400">{displayBacktest.best_trade_pnl != null ? `$${displayBacktest.best_trade_pnl.toFixed(2)}` : '—'}</p>
            </div>
            <div className="bg-surface border border-border rounded-lg p-4">
              <p className="text-text-secondary text-xs mb-1">Worst Trade</p>
              <p className="text-lg font-bold text-red-400">{displayBacktest.worst_trade_pnl != null ? `$${displayBacktest.worst_trade_pnl.toFixed(2)}` : '—'}</p>
            </div>
            <div className="bg-surface border border-border rounded-lg p-4">
              <p className="text-text-secondary text-xs mb-1">Avg Trade P&L</p>
              <p className={`text-lg font-bold ${(displayBacktest.avg_trade_pnl ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {displayBacktest.avg_trade_pnl != null ? `$${displayBacktest.avg_trade_pnl.toFixed(2)}` : '—'}
              </p>
            </div>
            <div className="bg-surface border border-border rounded-lg p-4">
              <p className="text-text-secondary text-xs mb-1">Avg Duration</p>
              <p className="text-lg font-bold text-text">{displayBacktest.avg_trade_duration_hours != null ? `${displayBacktest.avg_trade_duration_hours.toFixed(1)}h` : '—'}</p>
            </div>
          </div>

          {/* Strategy Configuration (config.json) */}
          <div className="bg-surface border border-border rounded-lg p-6">
            <h2 className="text-lg font-semibold text-text mb-4">Strategy Configuration</h2>
            {versionLoading ? (
              <p className="text-text-secondary animate-pulse text-sm">Loading config…</p>
            ) : versionConfig ? (
              <div className="space-y-6">
                {/* Description */}
                {versionConfig.description && (
                  <p className="text-sm text-text-secondary leading-relaxed">{versionConfig.description}</p>
                )}

                {/* Trading Parameters */}
                {versionConfig.trading && (
                  <div>
                    <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-3">Trading Parameters</h3>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      {([
                        { label: 'Symbols', value: Array.isArray(versionConfig.trading.symbols) ? versionConfig.trading.symbols.join(', ') : versionConfig.trading.symbols },
                        { label: 'Timeframe', value: versionConfig.trading.timeframe },
                        { label: 'Max Positions', value: versionConfig.trading.max_positions },
                        { label: 'Position Size', value: versionConfig.trading.position_size_pct != null ? `${(versionConfig.trading.position_size_pct * 100).toFixed(2)}%` : undefined },
                      ] as { label: string; value: any }[]).filter(item => item.value != null).map(item => (
                        <div key={item.label} className="bg-background rounded-lg p-3 border border-border/50">
                          <p className="text-xs text-text-secondary mb-1">{item.label}</p>
                          <p className="text-sm font-semibold text-text">{String(item.value)}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Indicators */}
                {versionConfig.indicators && Object.keys(versionConfig.indicators).length > 0 && (
                  <div>
                    <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-3">Indicators</h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                      {Object.entries(versionConfig.indicators as Record<string, Record<string, any>>).map(([name, params]) => (
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

                {/* Risk Management */}
                {versionConfig.risk_management && (
                  <div>
                    <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-3">Risk Management</h3>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      {([
                        {
                          label: 'Stop Loss',
                          value: versionConfig.risk_management.stop_loss_pct != null
                            ? `${versionConfig.risk_management.stop_loss_pct}%`
                            : undefined
                        },
                        {
                          label: 'Trailing Stop',
                          value: versionConfig.risk_management.trailing_stop_pct != null
                            ? `${versionConfig.risk_management.trailing_stop_pct}%`
                            : undefined
                        },
                        {
                          label: 'Max Position Loss',
                          value: versionConfig.risk_management.max_position_loss_pct != null
                            ? `${versionConfig.risk_management.max_position_loss_pct}%`
                            : undefined
                        },
                        {
                          label: 'Max Daily Loss',
                          value: versionConfig.risk_management.max_daily_loss_pct != null
                            ? `${versionConfig.risk_management.max_daily_loss_pct}%`
                            : (versionConfig.risk_management.max_daily_loss != null
                              ? `${(versionConfig.risk_management.max_daily_loss * 100).toFixed(1)}%`
                              : undefined)
                        },
                        {
                          label: 'Max Drawdown',
                          value: versionConfig.risk_management.max_drawdown_pct != null
                            ? `${versionConfig.risk_management.max_drawdown_pct}%`
                            : (versionConfig.risk_management.max_drawdown != null
                              ? `${(versionConfig.risk_management.max_drawdown * 100).toFixed(1)}%`
                              : undefined),
                          tooltip: 'Circuit breaker - trading pauses if portfolio drawdown exceeds this limit'
                        },
                      ] as { label: string; value: any; tooltip?: string }[]).filter(item => item.value != null).map(item => (
                        <div key={item.label} className="bg-background rounded-lg p-3 border border-border/50">
                          <p className="text-xs text-text-secondary mb-1" title={item.tooltip}>{item.label}</p>
                          <p className="text-sm font-semibold text-red-400">{String(item.value)}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-text-secondary text-sm">No configuration file found for this build.</p>
            )}
          </div>

          {/* Equity Curve */}
          {displayBacktest.equity_curve && displayBacktest.equity_curve.length > 1 && (
            <div className="bg-surface border border-border rounded-lg p-6">
              <h2 className="text-lg font-semibold text-text mb-4">Equity Curve</h2>
              <EquityCurve data={displayBacktest.equity_curve} />
            </div>
          )}

          {/* Win / Loss Bar */}
          {(displayBacktest.winning_trades != null || displayBacktest.losing_trades != null) && (
            <div className="bg-surface border border-border rounded-lg p-6">
              <WinLossBar wins={displayBacktest.winning_trades ?? 0} losses={displayBacktest.losing_trades ?? 0} />
            </div>
          )}

          {/* Trade Log */}
          {displayBacktest.trades && displayBacktest.trades.length > 0 && (() => {
            // Sort trades
            const sortedTrades = [...displayBacktest.trades].sort((a, b) => {
              const va = a[tradeSort.col], vb = b[tradeSort.col]
              if (va == null && vb == null) return 0
              if (va == null) return 1
              if (vb == null) return -1
              if (typeof va === 'number' && typeof vb === 'number') return tradeSort.asc ? va - vb : vb - va
              return tradeSort.asc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va))
            })

            // Paginate trades
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
                              setTradePage(0) // Reset to first page when sorting changes
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

                {/* Pagination Controls */}
                {totalPages > 1 && (
                  <div className="flex items-center justify-between mt-4 pt-4 border-t border-border">
                    <button
                      onClick={() => setTradePage(Math.max(0, tradePage - 1))}
                      disabled={tradePage === 0}
                      className={`px-4 py-2 rounded-lg font-medium transition-colors ${tradePage === 0
                        ? 'bg-surface-hover text-text-secondary cursor-not-allowed'
                        : 'bg-surface border border-border hover:border-primary text-text'
                        }`}
                    >
                      Previous
                    </button>

                    <span className="text-text-secondary text-sm">
                      Showing {startIdx + 1}-{endIdx} of {totalTrades} trades (Page {tradePage + 1} of {totalPages})
                    </span>

                    <button
                      onClick={() => setTradePage(Math.min(totalPages - 1, tradePage + 1))}
                      disabled={tradePage >= totalPages - 1}
                      className={`px-4 py-2 rounded-lg font-medium transition-colors ${tradePage >= totalPages - 1
                        ? 'bg-surface-hover text-text-secondary cursor-not-allowed'
                        : 'bg-surface border border-border hover:border-primary text-text'
                        }`}
                    >
                      Next
                    </button>
                  </div>
                )}
              </div>
            )
          })()}
        </>
      ) : (
        !versionLoading && (
          <div className="bg-surface border border-border rounded-lg p-6 text-center text-text-secondary">
            No backtest data available for this version yet.
          </div>
        )
      )}

      {/* Build History */}
      <div className="bg-surface border border-border rounded-lg p-6">
        <h2 className="text-lg font-semibold text-text mb-4">Build History</h2>
        {builds.length === 0 ? (
          <p className="text-text-secondary text-sm">No builds yet</p>
        ) : (
          <div className="space-y-3">
            {[...builds]
              .sort((a, b) => new Date(b.started_at).getTime() - new Date(a.started_at).getTime())
              .map((build, idx) => (
                <div key={build.uuid} className="border border-border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className={`text-xs font-medium px-2 py-0.5 rounded border ${getStatusColor(build.status)}`}>
                        {build.status}
                      </span>
                      <span className="text-sm text-text font-medium">Build #{builds.length - idx}</span>
                    </div>
                    <span className="text-xs text-text-secondary">{new Date(build.started_at).toLocaleString()}</span>
                  </div>
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <p className="text-text-secondary text-xs">Tokens Used</p>
                      <p className="text-text font-medium">{build.tokens_consumed?.toFixed(2) ?? '—'}</p>
                    </div>
                    <div>
                      <p className="text-text-secondary text-xs">Iterations</p>
                      <p className="text-text font-medium">{build.iteration_count}</p>
                    </div>
                    {build.phase && (
                      <div>
                        <p className="text-text-secondary text-xs">Phase</p>
                        <p className="text-text font-medium">{build.phase}</p>
                      </div>
                    )}
                  </div>
                </div>
              ))}
          </div>
        )}
      </div>

      {/* Retrain Modal */}
      {showRetrainModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-surface border border-border rounded-lg p-6 w-full max-w-md shadow-xl">
            <h2 className="text-xl font-bold text-text mb-2">Retrain Strategy</h2>
            <p className="text-text-secondary text-sm mb-6">
              Configure the retrain parameters. All context and results from previous builds will be fed into the new build to guide the AI.
            </p>

            <div className="space-y-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-text-secondary mb-1">
                  Target Return (%)
                </label>
                <input
                  type="number"
                  value={retrainTargetReturn}
                  onChange={(e) => setRetrainTargetReturn(Number(e.target.value))}
                  className="w-full bg-surface-hover border border-border rounded px-3 py-2 text-text focus:outline-none focus:border-primary"
                  min={1}
                  max={1000}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-text-secondary mb-1">
                  Max Iterations
                </label>
                <select
                  value={retrainMaxIterations}
                  onChange={(e) => setRetrainMaxIterations(Number(e.target.value))}
                  className="w-full bg-surface-hover border border-border rounded px-3 py-2 text-text focus:outline-none focus:border-primary"
                >
                  {[5, 10, 15, 25, 50, 100].map((n) => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
              </div>
              <div className="bg-surface-hover border border-border rounded p-3">
                <p className="text-xs text-text-secondary mb-1">Estimated Cost</p>
                {tokensPerIteration !== null ? (
                  <>
                    <p className="text-text font-semibold text-lg">
                      {(retrainMaxIterations * tokensPerIteration).toFixed(1)} tokens
                    </p>
                    <p className="text-xs text-text-secondary mt-1">
                      {tokensPerIteration} tokens × {retrainMaxIterations} iterations
                      {userBalance !== null && (
                        <span className="ml-2">· Balance: {userBalance.toFixed(1)} tokens</span>
                      )}
                    </p>
                  </>
                ) : (
                  <p className="text-text-secondary text-sm">Loading pricing…</p>
                )}
              </div>

              {/* Insufficient balance notice */}
              {tokensPerIteration !== null && userBalance !== null && userBalance < retrainMaxIterations * tokensPerIteration && (
                <div className="bg-yellow-900/20 border border-yellow-700 rounded p-3 flex items-start gap-2">
                  <span className="text-yellow-400 text-lg leading-none mt-0.5">⚠</span>
                  <div className="text-sm">
                    <p className="text-yellow-300 font-medium">Insufficient token balance</p>
                    <p className="text-yellow-400/80 mt-0.5">
                      You need {(retrainMaxIterations * tokensPerIteration).toFixed(1)} tokens but only have {userBalance.toFixed(1)}.{' '}
                      <a href="/dashboard/account" className="underline hover:text-yellow-300 transition-colors">
                        Purchase more tokens
                      </a>
                      {' '}to continue.
                    </p>
                  </div>
                </div>
              )}
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowRetrainModal(false)}
                disabled={retraining}
                className="flex-1 px-4 py-2 text-sm font-medium bg-surface-hover hover:bg-border text-text border border-border rounded-lg transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={handleRetrainConfirm}
                disabled={
                  retraining ||
                  tokensPerIteration === null ||
                  (userBalance !== null && userBalance < retrainMaxIterations * (tokensPerIteration ?? 0))
                }
                className="flex-1 px-4 py-2 text-sm font-medium bg-primary hover:bg-primary-hover text-white rounded-lg transition-colors disabled:opacity-50"
              >
                {retraining ? 'Starting…' : 'Confirm & Retrain'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Deploy / Action Modal */}
      {strategy && (
        <StrategyActionModal
          isOpen={isDeployModalOpen}
          onClose={() => setIsDeployModalOpen(false)}
          strategyId={strategy.uuid}
          strategyName={strategy.name}
          licenses={licenses.filter((l) => l.strategy_id === strategy.uuid)}
          onLicensePurchase={handleLicensePurchase}
          onWebhookUpdate={handleWebhookUpdate}
          onMarketplaceSubmit={handleMarketplaceSubmit}
          marketplaceListed={strategy.marketplace_listed ?? false}
        />
      )}
    </div>
  )
}

