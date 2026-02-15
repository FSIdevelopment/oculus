'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { useAuth } from '@/contexts/AuthContext'
import { buildAPI } from '@/lib/api'
import { Zap, Clock, CheckCircle, AlertCircle, ChevronLeft, ChevronRight } from 'lucide-react'

interface BuildListItem {
  uuid: string
  strategy_id: string
  strategy_name: string
  status: string
  phase: string | null
  tokens_consumed: number
  iteration_count: number
  started_at: string
  completed_at: string | null
}

export default function BuildListPage() {
  const { isAuthenticated, isLoading } = useAuth()
  const [builds, setBuilds] = useState<BuildListItem[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const pageSize = 10

  useEffect(() => {
    if (!isLoading && isAuthenticated) {
      fetchBuilds()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated, isLoading, page])

  const fetchBuilds = async () => {
    try {
      setLoading(true)
      const skip = page * pageSize
      const response = await buildAPI.listBuilds(skip, pageSize)
      setBuilds(response.items || [])
      setTotal(response.total || 0)
      setError(null)
    } catch (err) {
      console.error('Failed to fetch builds:', err)
      setError('Failed to load builds')
    } finally {
      setLoading(false)
    }
  }

  const getPhaseIcon = (phase: string | null) => {
    switch (phase) {
      case 'designing':
        return <Zap className="text-blue-400" size={16} />
      case 'refining':
        return <Zap className="text-purple-400" size={16} />
      case 'selecting_best':
        return <Zap className="text-yellow-400" size={16} />
      case 'training':
        return <Zap className="text-purple-400" size={16} />
      case 'optimizing':
        return <Zap className="text-yellow-400" size={16} />
      case 'building_docker':
        return <Zap className="text-green-400" size={16} />
      case 'complete':
        return <CheckCircle className="text-green-500" size={16} />
      case 'completed':
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
      case 'refining':
      case 'selecting_best':
      case 'training':
      case 'optimizing':
      case 'building_docker':
        return 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
      case 'complete':
      case 'completed':
        return 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
      case 'failed':
        return 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
      default:
        return 'bg-gray-100 dark:bg-gray-900 text-gray-600 dark:text-gray-300'
    }
  }

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp)
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    })
  }

  const getPhaseDisplayName = (phase: string | null) => {
    const phaseDisplayNames: { [key: string]: string } = {
      'building_docker': 'writing algorithm',
    }
    return phaseDisplayNames[phase || ''] || phase?.replace(/_/g, ' ') || ''
  }

  const totalPages = Math.ceil(total / pageSize)
  const canGoPrevious = page > 0
  const canGoNext = page < totalPages - 1

  if (loading) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-text mb-2">Build Monitor</h1>
          <p className="text-text-secondary">Track your strategy builds in real-time</p>
        </div>
        <div className="text-text-secondary">Loading builds...</div>
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

      {/* Builds List */}
      {builds.length === 0 ? (
        <div className="bg-surface border border-border rounded-lg p-8 text-center">
          <AlertCircle className="mx-auto mb-4 text-text-secondary" size={32} />
          <p className="text-text-secondary mb-4">No active builds</p>
          <Link
            href="/dashboard/create"
            className="inline-block px-6 py-2 bg-primary hover:bg-primary-hover text-white rounded-lg transition-colors"
          >
            Create Your First Strategy
          </Link>
        </div>
      ) : (
        <>
          <div className="grid gap-4">
            {builds.map((build) => (
              <Link
                key={build.uuid}
                href={`/dashboard/build/${build.uuid}`}
                className="bg-surface border border-border rounded-lg p-6 hover:border-primary hover:bg-surface-hover transition-all duration-300 group"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-text group-hover:text-primary transition-colors mb-2">
                      {build.strategy_name}
                    </h3>
                    <div className="flex items-center gap-4 flex-wrap">
                      <div className="flex items-center gap-2">
                        {getPhaseIcon(build.phase)}
                        <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(build.status)}`}>
                          {build.status.charAt(0).toUpperCase() + build.status.slice(1)}
                        </span>
                      </div>
                      {build.phase && (
                        <div className="text-text-secondary text-sm">
                          Phase: <span className="text-text capitalize">{getPhaseDisplayName(build.phase)}</span>
                        </div>
                      )}
                      <div className="text-text-secondary text-sm">
                        Iterations: <span className="text-text">{build.iteration_count}</span>
                      </div>
                      <div className="text-text-secondary text-sm">
                        Tokens: <span className="text-text">{build.tokens_consumed.toLocaleString()}</span>
                      </div>
                    </div>
                    <div className="mt-2 text-text-secondary text-sm">
                      Started: {formatTimestamp(build.started_at)}
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>

          {/* Pagination Controls */}
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-4 mt-6">
              <button
                onClick={() => setPage(page - 1)}
                disabled={!canGoPrevious}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                  canGoPrevious
                    ? 'bg-surface border border-border hover:border-primary text-text'
                    : 'bg-surface-hover border border-border text-text-secondary cursor-not-allowed'
                }`}
              >
                <ChevronLeft size={16} />
                Previous
              </button>

              <span className="text-text-secondary">
                Page {page + 1} of {totalPages}
              </span>

              <button
                onClick={() => setPage(page + 1)}
                disabled={!canGoNext}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                  canGoNext
                    ? 'bg-surface border border-border hover:border-primary text-text'
                    : 'bg-surface-hover border border-border text-text-secondary cursor-not-allowed'
                }`}
              >
                Next
                <ChevronRight size={16} />
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}

