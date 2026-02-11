'use client'

import { useState, useEffect } from 'react'
import { adminAPI } from '@/lib/api'
import { Search, Trash2 } from 'lucide-react'

interface Strategy {
  uuid: string
  name: string
  owner_id: string
  owner_name?: string
  type: string
  status: string
  rating?: number
  subscribers?: number
  created_at: string
}

export default function StrategyManagementTab() {
  const [strategies, setStrategies] = useState<Strategy[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [skip, setSkip] = useState(0)
  const [total, setTotal] = useState(0)

  const limit = 20

  useEffect(() => {
    fetchStrategies()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [skip, search, statusFilter])

  const fetchStrategies = async () => {
    try {
      setLoading(true)
      const data = await adminAPI.listStrategies({
        skip,
        limit,
        search: search || undefined,
        status: statusFilter !== 'all' ? statusFilter : undefined,
      })
      setStrategies(data.data || [])
      setTotal(data.total || 0)
    } catch (err) {
      setError('Failed to load strategies')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (strategyId: string) => {
    if (!confirm('Are you sure you want to delete this strategy?')) return
    try {
      await adminAPI.deleteStrategy(strategyId)
      setStrategies(strategies.filter((s) => s.uuid !== strategyId))
    } catch (err) {
      alert('Failed to delete strategy')
    }
  }

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex gap-2 flex-wrap">
        <div className="flex-1 min-w-[200px] relative">
          <Search className="absolute left-3 top-3 text-text-secondary" size={20} />
          <input
            type="text"
            placeholder="Search strategies..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value)
              setSkip(0)
            }}
            className="w-full pl-10 pr-4 py-2 bg-surface border border-border rounded-lg text-text placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-primary"
          />
        </div>
        <select
          value={statusFilter}
          onChange={(e) => {
            setStatusFilter(e.target.value)
            setSkip(0)
          }}
          className="px-4 py-2 bg-surface border border-border rounded-lg text-text focus:outline-none focus:ring-2 focus:ring-primary"
        >
          <option value="all">All Status</option>
          <option value="active">Active</option>
          <option value="inactive">Inactive</option>
          <option value="building">Building</option>
        </select>
      </div>

      {/* Strategies Table */}
      {loading ? (
        <div className="text-text-secondary">Loading strategies...</div>
      ) : error ? (
        <div className="text-red-400">{error}</div>
      ) : (
        <>
          <div className="overflow-x-auto bg-surface border border-border rounded-lg">
            <table className="w-full">
              <thead className="border-b border-border">
                <tr className="bg-surface-hover">
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Name</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Owner</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Type</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Status</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Rating</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Subscribers</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Actions</th>
                </tr>
              </thead>
              <tbody>
                {strategies.map((strategy, idx) => (
                  <tr
                    key={strategy.uuid}
                    className={`border-b border-border ${idx % 2 === 0 ? 'bg-surface' : 'bg-surface-hover'}`}
                  >
                    <td className="px-6 py-4 text-text font-medium">{strategy.name}</td>
                    <td className="px-6 py-4 text-text-secondary text-sm">{strategy.owner_name || strategy.owner_id}</td>
                    <td className="px-6 py-4 text-text capitalize">{strategy.type}</td>
                    <td className="px-6 py-4 text-text capitalize">{strategy.status}</td>
                    <td className="px-6 py-4 text-text">{strategy.rating?.toFixed(1) || 'N/A'}</td>
                    <td className="px-6 py-4 text-text">{strategy.subscribers || 0}</td>
                    <td className="px-6 py-4">
                      <div className="flex gap-2">
                        <button
                          onClick={() => handleDelete(strategy.uuid)}
                          className="p-2 hover:bg-red-900/20 rounded text-text-secondary hover:text-red-400"
                          title="Delete"
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex justify-between items-center">
            <p className="text-text-secondary text-sm">
              Showing {skip + 1} to {Math.min(skip + limit, total)} of {total}
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => setSkip(Math.max(0, skip - limit))}
                disabled={skip === 0}
                className="px-4 py-2 bg-surface border border-border rounded-lg text-text disabled:opacity-50 hover:bg-surface-hover"
              >
                Previous
              </button>
              <button
                onClick={() => setSkip(skip + limit)}
                disabled={skip + limit >= total}
                className="px-4 py-2 bg-surface border border-border rounded-lg text-text disabled:opacity-50 hover:bg-surface-hover"
              >
                Next
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

