'use client'

import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import { strategyAPI } from '@/lib/api'
import { ArrowLeft } from 'lucide-react'

interface Strategy {
  uuid: string
  name: string
  description?: string
  status: string
  strategy_type?: string
  symbols?: Record<string, any>
  target_return?: number
  backtest_results?: Record<string, any>
  subscriber_count: number
  rating?: number
  version: number
  created_at: string
  updated_at: string
  docker_image_url?: string
}

interface Build {
  uuid: string
  status: string
  phase?: string
  tokens_consumed: number
  iteration_count: number
  started_at: string
  completed_at?: string
}

export default function StrategyDetailPage() {
  const params = useParams()
  const router = useRouter()
  const strategyId = params.id as string

  const [strategy, setStrategy] = useState<Strategy | null>(null)
  const [builds, setBuilds] = useState<Build[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadStrategy = async () => {
    try {
      const data = await strategyAPI.getStrategy(strategyId)
      setStrategy(data)
    } catch (err) {
      setError('Failed to load strategy')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const loadBuilds = async () => {
    try {
      const data = await strategyAPI.getStrategyBuilds(strategyId, 0, 20)
      setBuilds(data.items || [])
    } catch (err) {
      console.error('Failed to load builds:', err)
    }
  }

  useEffect(() => {
    loadStrategy()
    loadBuilds()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [strategyId])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'complete':
        return 'bg-green-900/30 text-green-400 border-green-700'
      case 'building':
        return 'bg-yellow-900/30 text-yellow-400 border-yellow-700'
      case 'failed':
        return 'bg-red-900/30 text-red-400 border-red-700'
      default:
        return 'bg-gray-900/30 text-gray-400 border-gray-700'
    }
  }

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
      {/* Header */}
      <button
        onClick={() => router.back()}
        className="flex items-center gap-2 text-primary hover:underline"
      >
        <ArrowLeft size={18} />
        Back to Strategies
      </button>

      {/* Strategy Info */}
      <div className="bg-surface border border-border rounded-lg p-6">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h1 className="text-4xl font-bold text-text mb-2">{strategy.name}</h1>
            {strategy.description && (
              <p className="text-text-secondary">{strategy.description}</p>
            )}
          </div>
          <span className={`text-sm font-medium px-3 py-1 rounded border ${getStatusColor(strategy.status)}`}>
            {strategy.status}
          </span>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-text-secondary text-sm mb-1">Type</p>
            <p className="text-text font-medium">{strategy.strategy_type || 'N/A'}</p>
          </div>
          <div>
            <p className="text-text-secondary text-sm mb-1">Version</p>
            <p className="text-text font-medium">v{strategy.version}</p>
          </div>
          <div>
            <p className="text-text-secondary text-sm mb-1">Subscribers</p>
            <p className="text-text font-medium">{strategy.subscriber_count}</p>
          </div>
          <div>
            <p className="text-text-secondary text-sm mb-1">Rating</p>
            <p className="text-text font-medium">{strategy.rating ? strategy.rating.toFixed(1) : 'N/A'}</p>
          </div>
        </div>
      </div>

      {/* Backtest Results */}
      {strategy.backtest_results && (
        <div className="bg-surface border border-border rounded-lg p-6">
          <h2 className="text-2xl font-bold text-text mb-4">Backtest Results</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {Object.entries(strategy.backtest_results).map(([key, value]) => (
              <div key={key} className="bg-surface-hover border border-border rounded-lg p-4">
                <p className="text-text-secondary text-sm mb-2 capitalize">
                  {key.replace(/_/g, ' ')}
                </p>
                <p className="text-2xl font-bold text-primary">
                  {typeof value === 'number' ? value.toFixed(2) : value}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Configuration */}
      {strategy.symbols && (
        <div className="bg-surface border border-border rounded-lg p-6">
          <h2 className="text-2xl font-bold text-text mb-4">Configuration</h2>
          <div className="space-y-3">
            {strategy.target_return && (
              <div>
                <p className="text-text-secondary text-sm">Target Return</p>
                <p className="text-text font-medium">{strategy.target_return}%</p>
              </div>
            )}
            <div>
              <p className="text-text-secondary text-sm mb-2">Trading Symbols</p>
              <div className="flex flex-wrap gap-2">
                {Object.keys(strategy.symbols).map((symbol) => (
                  <span
                    key={symbol}
                    className="bg-surface-hover border border-border rounded px-3 py-1 text-sm text-text"
                  >
                    {symbol}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Build History */}
      <div className="bg-surface border border-border rounded-lg p-6">
        <h2 className="text-2xl font-bold text-text mb-4">Build History</h2>
        {builds.length === 0 ? (
          <p className="text-text-secondary">No builds yet</p>
        ) : (
          <div className="space-y-3">
            {builds.map((build) => (
              <div key={build.uuid} className="border border-border rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-xs font-medium px-2 py-1 rounded border ${getStatusColor(build.status)}`}>
                    {build.status}
                  </span>
                  <span className="text-xs text-text-secondary">
                    {new Date(build.started_at).toLocaleString()}
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <p className="text-text-secondary">Tokens Used</p>
                    <p className="text-text font-medium">{build.tokens_consumed.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-text-secondary">Iterations</p>
                    <p className="text-text font-medium">{build.iteration_count}</p>
                  </div>
                  {build.phase && (
                    <div>
                      <p className="text-text-secondary">Phase</p>
                      <p className="text-text font-medium">{build.phase}</p>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3">
        <Link
          href={`/dashboard/strategies/${strategy.uuid}?action=deploy`}
          className="flex-1 bg-primary hover:bg-primary-hover text-white font-medium py-3 rounded-lg transition-colors text-center"
        >
          Deploy Strategy
        </Link>
        <Link
          href="/dashboard/strategies"
          className="flex-1 bg-surface-hover hover:bg-border text-text font-medium py-3 rounded-lg transition-colors text-center"
        >
          Back to List
        </Link>
      </div>
    </div>
  )
}

