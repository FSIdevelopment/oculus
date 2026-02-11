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
  backtest_results?: Record<string, any>
  subscriber_count: number
  rating?: number
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
  'Sentiment Analysis',
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

  const getSymbolsDisplay = (symbols?: Record<string, any>) => {
    if (!symbols) return 'N/A'
    const symbolList = symbols.symbols || []
    return symbolList.slice(0, 3).join(', ') + (symbolList.length > 3 ? '...' : '')
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

