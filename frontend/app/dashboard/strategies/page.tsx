'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { strategyAPI, licenseAPI, marketplaceAPI } from '@/lib/api'
import StrategyActionModal from '@/components/StrategyActionModal'
import { TrendingUp, Calendar, Users } from 'lucide-react'

interface Strategy {
  uuid: string
  name: string
  description?: string
  status: string
  strategy_type?: string
  backtest_results?: Record<string, any>
  subscriber_count: number
  rating?: number
  created_at: string
  docker_image_url?: string
}

interface DockerInfo {
  image_url: string
  pull_command: string
  run_command: string
  environment_variables: Record<string, string>
  terms_of_use: string
}

interface License {
  uuid: string
  status: string
  license_type: string
  strategy_id: string
  expires_at: string
}

export default function StrategiesPage() {
  const [strategies, setStrategies] = useState<Strategy[]>([])
  const [licenses, setLicenses] = useState<License[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null)
  const [dockerInfo, setDockerInfo] = useState<DockerInfo | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)

  const loadStrategies = async () => {
    try {
      setLoading(true)
      const data = await strategyAPI.listStrategies(0, 50)
      setStrategies(data.items || [])
    } catch (err) {
      setError('Failed to load strategies')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const loadLicenses = async () => {
    try {
      const data = await licenseAPI.listLicenses(0, 50)
      setLicenses(data.items || [])
    } catch (err) {
      console.error('Failed to load licenses:', err)
    }
  }

  useEffect(() => {
    loadStrategies()
    loadLicenses()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const openActionModal = async (strategy: Strategy) => {
    setSelectedStrategy(strategy)
    try {
      const info = await strategyAPI.getDockerInfo(strategy.uuid)
      setDockerInfo(info)
    } catch (err) {
      console.error('Failed to load docker info:', err)
    }
    setIsModalOpen(true)
  }

  const handleMarketplaceSubmit = async () => {
    if (!selectedStrategy) return
    try {
      await marketplaceAPI.subscribeToStrategy(selectedStrategy.uuid)
      setIsModalOpen(false)
      loadStrategies()
    } catch (err) {
      console.error('Failed to list on marketplace:', err)
    }
  }

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

  const getMetricValue = (results: Record<string, any> | undefined, key: string) => {
    if (!results) return 'N/A'
    const value = results[key]
    if (typeof value === 'number') return value.toFixed(2)
    return value || 'N/A'
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-text-secondary">Loading strategies...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold text-text mb-2">Your Strategies</h1>
          <p className="text-text-secondary">Manage and deploy your trading strategies</p>
        </div>
        <Link
          href="/dashboard/create"
          className="bg-primary hover:bg-primary-hover text-white font-medium px-6 py-2 rounded-lg transition-colors"
        >
          Create Strategy
        </Link>
      </div>

      {error && (
        <div className="bg-red-900/20 border border-red-700 rounded-lg p-4 text-red-400">
          {error}
        </div>
      )}

      {/* Strategies Grid */}
      {strategies.length === 0 ? (
        <div className="bg-surface border border-border rounded-lg p-12 text-center">
          <p className="text-text-secondary mb-4">No strategies yet</p>
          <Link
            href="/dashboard/create"
            className="text-primary hover:underline"
          >
            Create your first strategy
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {strategies.map((strategy) => (
            <div
              key={strategy.uuid}
              className="bg-surface border border-border rounded-lg p-6 hover:border-primary transition-colors"
            >
              {/* Header */}
              <div className="mb-4">
                <h3 className="text-lg font-bold text-text mb-2">{strategy.name}</h3>
                <div className="flex items-center gap-2">
                  <span className={`text-xs font-medium px-2 py-1 rounded border ${getStatusColor(strategy.status)}`}>
                    {strategy.status}
                  </span>
                  {strategy.strategy_type && (
                    <span className="text-xs text-text-secondary bg-surface-hover px-2 py-1 rounded">
                      {strategy.strategy_type}
                    </span>
                  )}
                </div>
              </div>

              {/* Metrics */}
              <div className="space-y-3 mb-6">
                {strategy.backtest_results && (
                  <>
                    <div className="flex items-center gap-2 text-sm">
                      <TrendingUp size={16} className="text-primary" />
                      <span className="text-text-secondary">Return:</span>
                      <span className="text-text font-medium">{getMetricValue(strategy.backtest_results, 'total_return')}%</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <TrendingUp size={16} className="text-primary" />
                      <span className="text-text-secondary">Win Rate:</span>
                      <span className="text-text font-medium">{getMetricValue(strategy.backtest_results, 'win_rate')}%</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <TrendingUp size={16} className="text-primary" />
                      <span className="text-text-secondary">Sharpe:</span>
                      <span className="text-text font-medium">{getMetricValue(strategy.backtest_results, 'sharpe_ratio')}</span>
                    </div>
                  </>
                )}
              </div>

              {/* Footer */}
              <div className="space-y-3 border-t border-border pt-4">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-1 text-text-secondary">
                    <Users size={14} />
                    <span>{strategy.subscriber_count} subscribers</span>
                  </div>
                  <div className="flex items-center gap-1 text-text-secondary">
                    <Calendar size={14} />
                    <span>{new Date(strategy.created_at).toLocaleDateString()}</span>
                  </div>
                </div>

                <div className="flex gap-2">
                  <Link
                    href={`/dashboard/strategies/${strategy.uuid}`}
                    className="flex-1 bg-surface-hover hover:bg-border text-text font-medium py-2 rounded-lg transition-colors text-center text-sm"
                  >
                    View Details
                  </Link>
                  <button
                    onClick={() => openActionModal(strategy)}
                    className="flex-1 bg-primary hover:bg-primary-hover text-white font-medium py-2 rounded-lg transition-colors text-sm"
                  >
                    Deploy
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Action Modal */}
      {selectedStrategy && (
        <StrategyActionModal
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          strategyName={selectedStrategy.name}
          strategyId={selectedStrategy.uuid}
          dockerInfo={dockerInfo || undefined}
          licenses={licenses.filter(l => l.strategy_id === selectedStrategy.uuid)}
          onMarketplaceSubmit={handleMarketplaceSubmit}
        />
      )}
    </div>
  )
}

