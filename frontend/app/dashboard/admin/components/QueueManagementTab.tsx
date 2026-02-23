'use client'

import { useState, useEffect } from 'react'
import { adminAPI } from '@/lib/api'
import { Clock, Activity, Server, AlertCircle, CheckCircle, Loader } from 'lucide-react'
import Link from 'next/link'

interface QueuedBuild {
  uuid: string
  strategy_id: string
  status: string
  queue_position: number
  retry_count: number
  started_at: string | null
}

interface ActiveBuild {
  uuid: string
  strategy_id: string
  status: string
  phase: string | null
  assigned_worker_id: string | null
  last_heartbeat: string | null
  iteration_count: number
  max_iterations: number
}

interface Worker {
  worker_id: string
  hostname: string
  status: string
  capacity: number
  active_jobs: number
  last_heartbeat: string | null
}

interface TrainingQueueBuild {
  uuid: string
  strategy_id: string
  status: string
  phase: string | null
  iteration_count: number
  max_iterations: number
  started_at: string | null
}

interface QueueStatus {
  queue_length: number
  active_builds_count: number
  training_queue_length: number
  total_workers: number
  active_workers: number
  total_capacity: number
  available_capacity: number
  queued_builds: QueuedBuild[]
  active_builds: ActiveBuild[]
  training_queue_builds: TrainingQueueBuild[]
  workers: Worker[]
}

export default function QueueManagementTab() {
  const [queueStatus, setQueueStatus] = useState<QueueStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)

  const fetchQueueStatus = async () => {
    try {
      setLoading(true)
      const data = await adminAPI.getQueueStatus()
      setQueueStatus(data)
      setError(null)
    } catch (err) {
      setError('Failed to load queue status')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchQueueStatus()
  }, [])

  // Auto-refresh every 5 seconds
  useEffect(() => {
    if (!autoRefresh) return

    const interval = setInterval(() => {
      fetchQueueStatus()
    }, 5000)

    return () => clearInterval(interval)
  }, [autoRefresh])

  if (loading && !queueStatus) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader className="animate-spin text-primary" size={32} />
      </div>
    )
  }

  if (error && !queueStatus) {
    return (
      <div className="bg-red-900/20 border border-red-500 rounded-lg p-6 text-center">
        <AlertCircle className="mx-auto mb-2 text-red-400" size={32} />
        <p className="text-red-400">{error}</p>
      </div>
    )
  }

  if (!queueStatus) return null

  return (
    <div className="space-y-6">
      {/* Header with Auto-Refresh Toggle */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-text">Build Queue Management</h2>
        <label className="flex items-center gap-2 text-text-secondary cursor-pointer">
          <input
            type="checkbox"
            checked={autoRefresh}
            onChange={(e) => setAutoRefresh(e.target.checked)}
            className="w-4 h-4 rounded border-border bg-surface text-primary focus:ring-primary"
          />
          <span className="text-sm">Auto-refresh (5s)</span>
        </label>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-surface border border-border rounded-lg p-6">
          <div className="flex items-center gap-3 mb-2">
            <Clock className="text-yellow-400" size={24} />
            <h3 className="text-lg font-semibold text-text">Build Queue</h3>
          </div>
          <p className="text-3xl font-bold text-text">{queueStatus.queue_length}</p>
          <p className="text-text-secondary text-sm mt-1">Waiting to start</p>
        </div>

        <div className="bg-surface border border-border rounded-lg p-6">
          <div className="flex items-center gap-3 mb-2">
            <Activity className="text-green-400" size={24} />
            <h3 className="text-lg font-semibold text-text">Active Builds</h3>
          </div>
          <p className="text-3xl font-bold text-text">{queueStatus.active_builds_count}</p>
          <p className="text-text-secondary text-sm mt-1">Currently running</p>
        </div>

        <div className="bg-surface border border-border rounded-lg p-6">
          <div className="flex items-center gap-3 mb-2">
            <Clock className="text-orange-400" size={24} />
            <h3 className="text-lg font-semibold text-text">Training Queue</h3>
          </div>
          <p className="text-3xl font-bold text-text">{queueStatus.training_queue_length}</p>
          <p className="text-text-secondary text-sm mt-1">Waiting for worker</p>
        </div>

        <div className="bg-surface border border-border rounded-lg p-6">
          <div className="flex items-center gap-3 mb-2">
            <Server className="text-blue-400" size={24} />
            <h3 className="text-lg font-semibold text-text">Workers</h3>
          </div>
          <p className="text-3xl font-bold text-text">
            {queueStatus.active_workers}/{queueStatus.total_workers}
          </p>
          <p className="text-text-secondary text-sm mt-1">
            Capacity: {queueStatus.total_capacity - queueStatus.available_capacity}/{queueStatus.total_capacity}
          </p>
        </div>
      </div>

      {/* Queued Builds */}
      <div className="bg-surface border border-border rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-border">
          <h3 className="text-lg font-semibold text-text flex items-center gap-2">
            <Clock size={20} className="text-yellow-400" />
            Queued Builds ({queueStatus.queue_length})
          </h3>
        </div>
        {queueStatus.queued_builds.length === 0 ? (
          <div className="p-8 text-center text-text-secondary">
            <CheckCircle className="mx-auto mb-2 text-green-400" size={32} />
            <p>No builds in queue</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-surface-hover">
                <tr>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Position</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Build ID</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Status</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Retry Count</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Queued At</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Actions</th>
                </tr>
              </thead>
              <tbody>
                {queueStatus.queued_builds.map((build, idx) => (
                  <tr
                    key={build.uuid}
                    className={`border-b border-border ${idx % 2 === 0 ? 'bg-surface' : 'bg-surface-hover'}`}
                  >
                    <td className="px-6 py-4">
                      <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-yellow-900/20 text-yellow-400 font-semibold">
                        {build.queue_position}
                      </span>
                    </td>
                    <td className="px-6 py-4 font-mono text-sm text-text">{build.uuid.slice(0, 8)}...</td>
                    <td className="px-6 py-4">
                      <span className="px-2 py-1 rounded-full bg-yellow-900/20 text-yellow-400 text-xs font-medium">
                        {build.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-text">{build.retry_count}</td>
                    <td className="px-6 py-4 text-text-secondary text-sm">
                      {build.started_at ? new Date(build.started_at).toLocaleString() : 'N/A'}
                    </td>
                    <td className="px-6 py-4">
                      <Link
                        href={`/dashboard/build/${build.uuid}`}
                        className="text-primary hover:text-primary-hover text-sm font-medium"
                      >
                        View
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Training Queue (Waiting for Worker) */}
      <div className="bg-surface border border-border rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-border">
          <h3 className="text-lg font-semibold text-text flex items-center gap-2">
            <Clock size={20} className="text-orange-400" />
            Training Queue ({queueStatus.training_queue_length})
          </h3>
          <p className="text-text-secondary text-sm mt-1">Builds waiting for worker availability</p>
        </div>
        {queueStatus.training_queue_builds.length === 0 ? (
          <div className="p-8 text-center text-text-secondary">
            <CheckCircle className="mx-auto mb-2 text-green-400" size={32} />
            <p>No builds waiting for workers</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-surface-hover">
                <tr>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Build ID</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Status</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Phase</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Progress</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Waiting Since</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Actions</th>
                </tr>
              </thead>
              <tbody>
                {queueStatus.training_queue_builds.map((build, idx) => (
                  <tr
                    key={build.uuid}
                    className={`border-b border-border ${idx % 2 === 0 ? 'bg-surface' : 'bg-surface-hover'}`}
                  >
                    <td className="px-6 py-4 font-mono text-sm text-text">{build.uuid.slice(0, 8)}...</td>
                    <td className="px-6 py-4">
                      <span className="px-2 py-1 rounded-full bg-orange-900/20 text-orange-400 text-xs font-medium">
                        {build.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-text capitalize">{build.phase || 'N/A'}</td>
                    <td className="px-6 py-4 text-text">
                      {build.iteration_count}/{build.max_iterations}
                    </td>
                    <td className="px-6 py-4 text-text-secondary text-sm">
                      {build.started_at ? new Date(build.started_at).toLocaleString() : 'N/A'}
                    </td>
                    <td className="px-6 py-4">
                      <Link
                        href={`/dashboard/build/${build.uuid}`}
                        className="text-primary hover:text-primary-hover text-sm font-medium"
                      >
                        View
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Active Builds */}
      <div className="bg-surface border border-border rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-border">
          <h3 className="text-lg font-semibold text-text flex items-center gap-2">
            <Activity size={20} className="text-green-400" />
            Active Builds ({queueStatus.active_builds_count})
          </h3>
        </div>
        {queueStatus.active_builds.length === 0 ? (
          <div className="p-8 text-center text-text-secondary">
            <AlertCircle className="mx-auto mb-2" size={32} />
            <p>No active builds</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-surface-hover">
                <tr>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Build ID</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Status</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Phase</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Progress</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Worker</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Last Heartbeat</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Actions</th>
                </tr>
              </thead>
              <tbody>
                {queueStatus.active_builds.map((build, idx) => (
                  <tr
                    key={build.uuid}
                    className={`border-b border-border ${idx % 2 === 0 ? 'bg-surface' : 'bg-surface-hover'}`}
                  >
                    <td className="px-6 py-4 font-mono text-sm text-text">{build.uuid.slice(0, 8)}...</td>
                    <td className="px-6 py-4">
                      <span className="px-2 py-1 rounded-full bg-green-900/20 text-green-400 text-xs font-medium">
                        {build.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-text capitalize">{build.phase || 'N/A'}</td>
                    <td className="px-6 py-4 text-text">
                      {build.iteration_count}/{build.max_iterations}
                    </td>
                    <td className="px-6 py-4 font-mono text-sm text-text-secondary">
                      {build.assigned_worker_id?.slice(0, 8) || 'N/A'}
                    </td>
                    <td className="px-6 py-4 text-text-secondary text-sm">
                      {build.last_heartbeat ? new Date(build.last_heartbeat).toLocaleString() : 'N/A'}
                    </td>
                    <td className="px-6 py-4">
                      <Link
                        href={`/dashboard/build/${build.uuid}`}
                        className="text-primary hover:text-primary-hover text-sm font-medium"
                      >
                        View
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Workers Status */}
      <div className="bg-surface border border-border rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-border">
          <h3 className="text-lg font-semibold text-text flex items-center gap-2">
            <Server size={20} className="text-blue-400" />
            Workers ({queueStatus.active_workers}/{queueStatus.total_workers} active)
          </h3>
        </div>
        {queueStatus.workers.length === 0 ? (
          <div className="p-8 text-center text-text-secondary">
            <AlertCircle className="mx-auto mb-2 text-red-400" size={32} />
            <p>No workers registered</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-surface-hover">
                <tr>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Worker ID</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Hostname</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Status</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Capacity</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Active Jobs</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Last Heartbeat</th>
                </tr>
              </thead>
              <tbody>
                {queueStatus.workers.map((worker, idx) => (
                  <tr
                    key={worker.worker_id}
                    className={`border-b border-border ${idx % 2 === 0 ? 'bg-surface' : 'bg-surface-hover'}`}
                  >
                    <td className="px-6 py-4 font-mono text-sm text-text">{worker.worker_id.slice(0, 8)}...</td>
                    <td className="px-6 py-4 text-text">{worker.hostname}</td>
                    <td className="px-6 py-4">
                      <span
                        className={`px-2 py-1 rounded-full text-xs font-medium ${worker.status === 'active'
                          ? 'bg-green-900/20 text-green-400'
                          : 'bg-red-900/20 text-red-400'
                          }`}
                      >
                        {worker.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-text">{worker.capacity}</td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <span className="text-text">{worker.active_jobs}</span>
                        <div className="flex-1 bg-surface-hover rounded-full h-2 max-w-[100px]">
                          <div
                            className="bg-primary h-2 rounded-full transition-all"
                            style={{ width: `${(worker.active_jobs / worker.capacity) * 100}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-text-secondary text-sm">
                      {worker.last_heartbeat ? new Date(worker.last_heartbeat).toLocaleString() : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

