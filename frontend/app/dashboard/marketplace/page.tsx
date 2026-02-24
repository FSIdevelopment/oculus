'use client'

import { useEffect, useState, useCallback } from 'react'
import Link from 'next/link'
import { marketplaceAPI } from '@/lib/api'
import { Users, Search } from 'lucide-react'

interface Strategy {
  uuid: string
  name: string
  description?: string
  strategy_type?: string
  symbols?: Record<string, any>
  target_return?: number
  backtest_results?: {
    total_return: number
    total_return_pct: number
    annual_return: number
    max_drawdown: number
    win_rate: number
    [key: string]: any
  }
  subscriber_count: number
  rating?: number
  rating_count?: number
  monthly_price?: number
  annual_price?: number
  created_at: string
}

const STRATEGY_TYPES = [
  'Bollinger Band Breakout',
  'Dual Moving Average Crossover',
  'Fibonacci Retracement',
  'MACD Divergence',
  'Mean Reversion',
  'Momentum',
  'Moving Average Convergence',
  'Pairs Trading',
  'Price Action',
  'RSI Overbought/Oversold',
  'Scalping',
  'Statistical Arbitrage',
  'Support/Resistance Breakout',
  'Swing Trading',
  'Trend Following',
  'VWAP',
  'Volatility Breakout',
  'Volume Profile',
  'Williams %R',
]

export default function MarketplacePage() {
  const [strategies, setStrategies] = useState<Strategy[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedType, setSelectedType] = useState('')
  const [sortBy, setSortBy] = useState('created_at')
  const [sortOrder] = useState('desc')
  const [skip, setSkip] = useState(0)
  const [total, setTotal] = useState(0)

  const limit = 12

  const loadStrategies = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const data = await marketplaceAPI.listMarketplace({
        skip,
        limit,
        strategy_type: selectedType,
        sort_by: sortBy,
        sort_order: sortOrder,
      })
      setStrategies(data.items)
      setTotal(data.total)
    } catch (err: any) {
      const message = err.response?.data?.detail || err.message || 'Failed to load strategies'
      setError(message)
    } finally {
      setLoading(false)
    }
  }, [skip, selectedType, sortBy, sortOrder])

  useEffect(() => {
    loadStrategies()
  }, [loadStrategies])

  const filteredStrategies = strategies.filter(s =>
    s.name.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const renderStars = (rating?: number) => {
    if (!rating) return <span className="text-text-secondary text-sm">No ratings</span>
    const stars = Math.round(rating)
    return (
      <div className="flex items-center gap-1">
        {[...Array(5)].map((_, i) => (
          <span key={i} className={i < stars ? 'text-yellow-400' : 'text-gray-600'}>
            ★
          </span>
        ))}
        <span className="text-text-secondary text-sm ml-1">({rating.toFixed(1)})</span>
      </div>
    )
  }

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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold text-text mb-2">Marketplace</h1>
        <p className="text-text-secondary">Browse and subscribe to trading strategies</p>
      </div>

      {/* Search & Filters */}
      <div className="bg-surface border border-border rounded-lg p-4 space-y-4">
        <div className="flex gap-4 flex-col md:flex-row">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-3 text-text-secondary" size={18} />
            <input
              type="text"
              placeholder="Search strategies..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-background border border-border rounded-lg text-text placeholder-text-secondary focus:outline-none focus:border-primary"
            />
          </div>

          {/* Type Filter */}
          <select
            value={selectedType}
            onChange={(e) => {
              setSelectedType(e.target.value)
              setSkip(0)
            }}
            className="px-4 py-2 bg-background border border-border rounded-lg text-text focus:outline-none focus:border-primary"
          >
            <option value="">All Types</option>
            {STRATEGY_TYPES.map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>

          {/* Sort */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="px-4 py-2 bg-background border border-border rounded-lg text-text focus:outline-none focus:border-primary"
          >
            <option value="created_at">Newest</option>
            <option value="rating">Highest Rated</option>
            <option value="subscriber_count">Most Popular</option>
            <option value="target_return">Best Return</option>
          </select>
        </div>
      </div>

      {error && (
        <div className="bg-red-900/20 border border-red-700 rounded-lg p-4 text-red-400">
          {error}
        </div>
      )}

      {loading ? (
        <div className="text-center py-12 text-text-secondary">Loading strategies...</div>
      ) : filteredStrategies.length === 0 ? (
        <div className="text-center py-12 text-text-secondary">
          No strategies available in the marketplace yet
        </div>
      ) : (
        <>
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
                            <span className="text-text-secondary text-sm">· ${featured.annual_price} /yr</span>
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

          {/* Pagination */}
          <div className="flex justify-between items-center">
            <button
              onClick={() => setSkip(Math.max(0, skip - limit))}
              disabled={skip === 0}
              className="px-4 py-2 bg-surface border border-border rounded-lg text-text hover:border-primary disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Previous
            </button>
            <span className="text-text-secondary">
              {skip + 1} - {Math.min(skip + limit, total)} of {total}
            </span>
            <button
              onClick={() => setSkip(skip + limit)}
              disabled={skip + limit >= total}
              className="px-4 py-2 bg-surface border border-border rounded-lg text-text hover:border-primary disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Next
            </button>
          </div>
        </>
      )}
    </div>
  )
}
