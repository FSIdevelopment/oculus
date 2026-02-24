'use client'

import { useState, useEffect, useCallback } from 'react'
import api from '@/lib/api'
import { Loader, AlertCircle, Eye, EyeOff, Copy, Check } from 'lucide-react'

interface License {
  uuid: string
  status: string
  license_type: string
  strategy_id: string
  strategy_name: string | null
  strategy_version: number | null
  created_at: string
  expires_at: string
}

function maskLicenseKey(uuid: string): string {
  const parts = uuid.split('-')
  const last = parts[parts.length - 1]
  const hint = last.slice(-6)
  return `\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022-\u2022\u2022\u2022\u2022-\u2022\u2022\u2022\u2022-\u2022\u2022\u2022\u2022-${hint}`
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  })
}

function daysUntilExpiry(expiresAt: string): number {
  return Math.ceil((new Date(expiresAt).getTime() - Date.now()) / 86_400_000)
}

function StatusBadge({ status, expiresAt }: { status: string; expiresAt: string }) {
  const expired = new Date(expiresAt) < new Date()
  const effectiveStatus = expired && status === 'active' ? 'expired' : status

  const config: Record<string, { label: string; className: string }> = {
    active:     { label: 'Active',    className: 'bg-green-900/20 text-green-400 border border-green-700' },
    cancelling: { label: `Cancels ${formatDate(expiresAt)}`, className: 'bg-yellow-900/20 text-yellow-400 border border-yellow-700' },
    expired:    { label: 'Expired',   className: 'bg-red-900/20 text-red-400 border border-red-700' },
    cancelled:  { label: 'Cancelled', className: 'bg-surface text-text-secondary border border-border' },
  }

  const { label, className } = config[effectiveStatus] ?? config['cancelled']
  return (
    <span className={`px-3 py-1 rounded-full text-xs font-medium ${className}`}>
      {label}
    </span>
  )
}

function LicenseKeyRow({ uuid }: { uuid: string }) {
  const [revealed, setRevealed] = useState(false)
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(uuid)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="flex items-center gap-2 bg-black/20 rounded-lg px-3 py-2 font-mono text-sm">
      <span className="flex-1 text-text-secondary select-all">
        {revealed ? uuid : maskLicenseKey(uuid)}
      </span>
      <button
        onClick={() => setRevealed((v) => !v)}
        title={revealed ? 'Hide key' : 'Show key'}
        className="text-text-secondary hover:text-text transition-colors p-1"
      >
        {revealed ? <EyeOff size={15} /> : <Eye size={15} />}
      </button>
      <button
        onClick={handleCopy}
        title="Copy license key"
        className="text-text-secondary hover:text-text transition-colors p-1"
      >
        {copied ? <Check size={15} className="text-green-400" /> : <Copy size={15} />}
      </button>
    </div>
  )
}

function LicenseCard({
  license,
  onCancelled,
}: {
  license: License
  onCancelled: (id: string) => void
}) {
  const [confirming, setConfirming] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const daysLeft = daysUntilExpiry(license.expires_at)
  const showWarning = daysLeft > 0 && daysLeft <= 7

  const handleCancel = async () => {
    setCancelling(true)
    setError(null)
    try {
      await api.delete(`/api/licenses/${license.uuid}`)
      onCancelled(license.uuid)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to cancel subscription')
      setConfirming(false)
    } finally {
      setCancelling(false)
    }
  }

  return (
    <div className="bg-surface border border-border rounded-lg p-6 hover:border-primary transition-colors space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-base font-bold text-text">
            {license.strategy_name ?? 'Strategy License'}
            {license.strategy_version != null && (
              <span className="ml-2 text-text-secondary font-normal text-sm">
                v{license.strategy_version}
              </span>
            )}
          </h3>
        </div>
        <StatusBadge status={license.status} expiresAt={license.expires_at} />
      </div>

      {/* License key */}
      <div>
        <p className="text-text-secondary text-xs mb-1">License Key</p>
        <LicenseKeyRow uuid={license.uuid} />
      </div>

      {/* Info grid */}
      <div className="grid grid-cols-3 gap-4 text-sm">
        <div>
          <p className="text-text-secondary text-xs mb-1">Type</p>
          <p className="text-text font-medium capitalize">{license.license_type}</p>
        </div>
        <div>
          <p className="text-text-secondary text-xs mb-1">Expires</p>
          <p className="text-text font-medium flex items-center gap-2">
            {formatDate(license.expires_at)}
            {showWarning && (
              <span className="text-xs bg-yellow-900/20 text-yellow-400 border border-yellow-700 px-2 py-0.5 rounded-full">
                {daysLeft}d left
              </span>
            )}
          </p>
        </div>
        <div>
          <p className="text-text-secondary text-xs mb-1">Version</p>
          <p className="text-text font-medium">
            {license.strategy_version != null ? `v${license.strategy_version}` : '\u2014'}
          </p>
        </div>
      </div>

      {/* Error */}
      {error && (
        <p className="text-red-400 text-sm flex items-center gap-2">
          <AlertCircle size={14} /> {error}
        </p>
      )}

      {/* Cancel */}
      {license.status === 'active' && (
        <div>
          {confirming ? (
            <div className="space-y-2">
              <p className="text-text-secondary text-sm">
                Your license stays active until{' '}
                <span className="text-text font-medium">{formatDate(license.expires_at)}</span>.
                Confirm cancel?
              </p>
              <div className="flex gap-2">
                <button
                  onClick={handleCancel}
                  disabled={cancelling}
                  className="px-4 py-2 bg-red-700 hover:bg-red-600 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50"
                >
                  {cancelling ? 'Cancelling\u2026' : 'Yes, Cancel'}
                </button>
                <button
                  onClick={() => setConfirming(false)}
                  disabled={cancelling}
                  className="px-4 py-2 bg-surface border border-border hover:border-primary text-text text-sm font-medium rounded-lg transition-colors"
                >
                  Back
                </button>
              </div>
            </div>
          ) : (
            <button
              onClick={() => setConfirming(true)}
              className="text-sm text-text-secondary hover:text-red-400 transition-colors underline underline-offset-2"
            >
              Cancel Subscription
            </button>
          )}
        </div>
      )}
    </div>
  )
}

export default function LicensesTab() {
  const [licenses, setLicenses] = useState<License[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchLicenses = useCallback(async () => {
    try {
      const response = await api.get('/api/users/me/licenses')
      setLicenses(response.data.items || [])
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch licenses')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchLicenses()
  }, [fetchLicenses])

  const handleCancelled = useCallback((licenseId: string) => {
    setLicenses((prev) =>
      prev.map((l) => (l.uuid === licenseId ? { ...l, status: 'cancelling' } : l))
    )
  }, [])

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
          <p className="text-sm text-text-secondary">
            Purchase a license to run strategies on the platform
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {licenses.map((license) => (
            <LicenseCard key={license.uuid} license={license} onCancelled={handleCancelled} />
          ))}
        </div>
      )}
    </div>
  )
}
