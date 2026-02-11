'use client'

import { useState, useEffect } from 'react'
import { adminAPI } from '@/lib/api'
import { Users, TrendingUp, Zap, FileText } from 'lucide-react'

interface MetricCard {
  label: string
  value: number | string
  icon: React.ReactNode
  color: string
}

export default function OverviewTab() {
  const [metrics, setMetrics] = useState<MetricCard[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setLoading(true)
        const [usersData, strategiesData, licensesData] = await Promise.all([
          adminAPI.listUsers({ skip: 0, limit: 1 }),
          adminAPI.listStrategies({ skip: 0, limit: 1 }),
          adminAPI.listLicenses({ skip: 0, limit: 1 }),
        ])

        const totalUsers = usersData.total || 0
        const totalStrategies = strategiesData.total || 0
        const totalLicenses = licensesData.total || 0
        const activeBuilds = strategiesData.data?.filter((s: any) => s.status === 'building').length || 0

        setMetrics([
          {
            label: 'Total Users',
            value: totalUsers,
            icon: <Users size={24} />,
            color: 'text-blue-400',
          },
          {
            label: 'Total Strategies',
            value: totalStrategies,
            icon: <TrendingUp size={24} />,
            color: 'text-green-400',
          },
          {
            label: 'Active Builds',
            value: activeBuilds,
            icon: <Zap size={24} />,
            color: 'text-yellow-400',
          },
          {
            label: 'Active Licenses',
            value: totalLicenses,
            icon: <FileText size={24} />,
            color: 'text-purple-400',
          },
        ])
      } catch (err) {
        setError('Failed to load metrics')
        console.error(err)
      } finally {
        setLoading(false)
      }
    }

    fetchMetrics()
  }, [])

  if (loading) {
    return <div className="text-text-secondary">Loading metrics...</div>
  }

  if (error) {
    return <div className="text-red-400">{error}</div>
  }

  return (
    <div className="space-y-6">
      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((metric, idx) => (
          <div
            key={idx}
            className="bg-surface border border-border rounded-lg p-6 hover:bg-surface-hover transition-colors"
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="text-text-secondary text-sm mb-2">{metric.label}</p>
                <p className="text-3xl font-bold text-text">{metric.value}</p>
              </div>
              <div className={`${metric.color}`}>{metric.icon}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Placeholder for charts */}
      <div className="bg-surface border border-border rounded-lg p-6">
        <h3 className="text-lg font-semibold text-text mb-4">Platform Activity</h3>
        <div className="h-64 flex items-center justify-center text-text-secondary">
          <p>Activity charts coming soon</p>
        </div>
      </div>
    </div>
  )
}

