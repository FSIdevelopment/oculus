'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { useAuth } from '@/contexts/AuthContext'
import api from '@/lib/api'
import { Zap, Clock, CheckCircle, AlertCircle } from 'lucide-react'

interface Strategy {
  uuid: string
  name: string
  status: string
}

interface Build {
  uuid: string
  strategy_id: string
  status: string
  phase: string | null
  tokens_consumed: number
  iteration_count: number
  started_at: string
  completed_at: string | null
}

interface StrategyWithBuild extends Strategy {
  latestBuild?: Build
}

export default function BuildListPage() {
  const { isAuthenticated, isLoading } = useAuth()
  const [strategies, setStrategies] = useState<StrategyWithBuild[]>([])
  const [loadingStrategies, setLoadingStrategies] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!isLoading && isAuthenticated) {
      fetchStrategies()
    }
  }, [isAuthenticated, isLoading])

  const fetchStrategies = async () => {
    try {
      setLoadingStrategies(true)
      const response = await api.get('/api/strategies')
      const strategiesData = response.data.items || []

      // Fetch latest build for each strategy
      const strategiesWithBuilds = await Promise.all(
        strategiesData.map(async (strategy: Strategy) => {
          try {
            const buildsResponse = await api.get(`/api/strategies/${strategy.uuid}/builds?limit=1`)
            const latestBuild = buildsResponse.data.items?.[0]
            return { ...strategy, latestBuild }
          } catch (err) {
            console.error(`Failed to fetch builds for strategy ${strategy.uuid}:`, err)
            return strategy
          }
        })
      )

      setStrategies(strategiesWithBuilds)
      setError(null)
    } catch (err) {
      console.error('Failed to fetch strategies:', err)
      setError('Failed to load strategies')
    } finally {
      setLoadingStrategies(false)
    }
  }

  const getPhaseIcon = (phase: string | null) => {
    switch (phase) {
      case 'designing':
        return <Zap className="text-blue-400" size={16} />
      case 'training':
        return <Zap className="text-purple-400" size={16} />
      case 'optimizing':
        return <Zap className="text-yellow-400" size={16} />
      case 'building_docker':
        return <Zap className="text-green-400" size={16} />
      case 'complete':
        return <CheckCircle className="text-green-500" size={16} />
      default:
        return <Clock className="text-gray-400" size={16} />
    }
  }

  // Status badge colors compatible with both light and dark themes
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'queued':
        return 'bg-gray-100 dark:bg-gray-900 text-gray-600 dark:text-gray-300'
      case 'building':
      case 'designing':
      case 'training':
      case 'optimizing':
      case 'building_docker':
        return 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
      case 'complete':
        return 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
      case 'failed':
        return 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
      default:
        return 'bg-gray-100 dark:bg-gray-900 text-gray-600 dark:text-gray-300'
    }
  }

  if (loadingStrategies) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-text mb-2">Build Monitor</h1>
          <p className="text-text-secondary">Track your strategy builds in real-time</p>
        </div>
        <div className="text-text-secondary">Loading strategies...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-text mb-2">Build Monitor</h1>
        <p className="text-text-secondary">Track your strategy builds in real-time</p>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-100 dark:bg-red-900/20 border border-red-300 dark:border-red-700 rounded-lg p-4 text-red-700 dark:text-red-300">
          {error}
        </div>
      )}

      {/* Strategies List */}
      {strategies.length === 0 ? (
        <div className="bg-surface border border-border rounded-lg p-8 text-center">
          <AlertCircle className="mx-auto mb-4 text-text-secondary" size={32} />
          <p className="text-text-secondary mb-4">No strategies yet</p>
          <Link
            href="/dashboard/create"
            className="inline-block px-6 py-2 bg-primary hover:bg-primary-hover text-white rounded-lg transition-colors"
          >
            Create Your First Strategy
          </Link>
        </div>
      ) : (
        <div className="grid gap-4">
          {strategies.map((strategy) => {
            const build = strategy.latestBuild
            return (
              <Link
                key={strategy.uuid}
                href={`/dashboard/build/${strategy.uuid}`}
                className="bg-surface border border-border rounded-lg p-6 hover:border-primary hover:bg-surface-hover transition-all duration-300 group"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-text group-hover:text-primary transition-colors mb-2">
                      {strategy.name}
                    </h3>
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-2">
                        {getPhaseIcon(build?.phase || null)}
                        <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(build?.status || strategy.status)}`}>
                          {build ? build.status.charAt(0).toUpperCase() + build.status.slice(1) : 'No builds'}
                        </span>
                      </div>
                      {build && (
                        <div className="text-text-secondary text-sm">
                          Phase: <span className="text-text capitalize">{build.phase || 'N/A'}</span>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-text-secondary text-sm">Click to view details</p>
                  </div>
                </div>
              </Link>
            )
          })}
        </div>
      )}
    </div>
  )
}

