'use client'

import { useState, useEffect } from 'react'
import api from '@/lib/api'
import { Loader, AlertCircle } from 'lucide-react'

interface License {
  uuid: string
  status: string
  license_type: string
  strategy_id: string
  created_at: string
  expires_at: string
}

export default function LicensesTab() {
  const [licenses, setLicenses] = useState<License[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchLicenses()
  }, [])

  const fetchLicenses = async () => {
    try {
      const response = await api.get('/api/users/me/licenses')
      setLicenses(response.data.items || [])
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch licenses')
    } finally {
      setLoading(false)
    }
  }

  const handleRenew = async (licenseId: string, licenseType: string) => {
    try {
      await api.post(`/api/licenses/${licenseId}/renew`, {
        license_type: licenseType,
      })
      fetchLicenses()
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to renew license')
    }
  }

  const isExpired = (expiresAt: string) => {
    return new Date(expiresAt) < new Date()
  }

  const daysUntilExpiry = (expiresAt: string) => {
    const days = Math.ceil((new Date(expiresAt).getTime() - new Date().getTime()) / (1000 * 60 * 60 * 24))
    return days
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader className="animate-spin text-primary" size={32} />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {error && (
        <div className="flex items-center gap-3 p-4 bg-red-900/20 border border-red-700 rounded-lg text-red-400">
          <AlertCircle size={20} />
          {error}
        </div>
      )}

      {licenses.length === 0 ? (
        <div className="bg-surface border border-border rounded-lg p-8 text-center">
          <p className="text-text-secondary mb-4">No active licenses yet</p>
          <p className="text-sm text-text-secondary">Purchase a license to run strategies on the platform</p>
        </div>
      ) : (
        <div className="space-y-4">
          {licenses.map((license) => {
            const expired = isExpired(license.expires_at)
            const daysLeft = daysUntilExpiry(license.expires_at)

            return (
              <div
                key={license.uuid}
                className="bg-surface border border-border rounded-lg p-6 hover:border-primary transition-colors"
              >
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-bold text-text mb-1">Strategy License</h3>
                    <p className="text-text-secondary text-sm">ID: {license.strategy_id.substring(0, 8)}...</p>
                  </div>
                  <div
                    className={`px-3 py-1 rounded-full text-sm font-medium ${
                      expired
                        ? 'bg-red-900/20 text-red-400'
                        : license.status === 'active'
                          ? 'bg-green-900/20 text-green-400'
                          : 'bg-yellow-900/20 text-yellow-400'
                    }`}
                  >
                    {expired ? 'Expired' : license.status}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="text-text-secondary text-sm mb-1">License Type</p>
                    <p className="text-text font-medium capitalize">{license.license_type}</p>
                  </div>
                  <div>
                    <p className="text-text-secondary text-sm mb-1">Expires</p>
                    <p className="text-text font-medium">
                      {expired ? (
                        <span className="text-red-400">Expired</span>
                      ) : (
                        `${daysLeft} days`
                      )}
                    </p>
                  </div>
                </div>

                {!expired && (
                  <button
                    onClick={() => handleRenew(license.uuid, license.license_type)}
                    className="w-full px-4 py-2 bg-primary hover:bg-primary-hover text-white font-medium rounded-lg transition-colors"
                  >
                    Renew License
                  </button>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

