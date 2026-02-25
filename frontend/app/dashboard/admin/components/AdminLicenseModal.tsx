'use client'

import { useState, useEffect } from 'react'
import { adminAPI } from '@/lib/api'
import { X, Eye, EyeOff, Key, RefreshCw, XCircle, Copy, Check } from 'lucide-react'

interface License {
  uuid: string
  status: string
  license_type: string
  strategy_id: string
  user_id: string
  strategy_version?: number
  created_at: string
  expires_at: string
}

interface VersionInfo {
  version: number
  best_return_pct: number | null
}

interface VersionsData {
  versions: VersionInfo[]
  current_version: number
  total_builds: number
}

interface AdminLicenseModalProps {
  strategyId: string
  strategyName: string
  onClose: () => void
}

export default function AdminLicenseModal({ strategyId, strategyName, onClose }: AdminLicenseModalProps) {
  const [license, setLicense] = useState<License | null>(null)
  const [loading, setLoading] = useState(true)
  const [actionLoading, setActionLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [keyVisible, setKeyVisible] = useState(false)
  const [copied, setCopied] = useState(false)
  const [versions, setVersions] = useState<VersionsData | null>(null)
  const [selectedVersion, setSelectedVersion] = useState<number | null>(null)

  useEffect(() => {
    fetchLicense()
    fetchVersions()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [strategyId])

  const fetchLicense = async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await adminAPI.getAdminLicense(strategyId)
      setLicense(data)
    } catch (err) {
      setError('Failed to load license')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const fetchVersions = async () => {
    try {
      const data = await adminAPI.getStrategyVersions(strategyId)
      setVersions(data)
      // Set default selected version to current version
      if (data.versions.length > 0) {
        setSelectedVersion(data.current_version)
      }
    } catch (err) {
      console.error('Failed to load versions:', err)
    }
  }

  const handleGenerate = async () => {
    if (!selectedVersion) {
      setError('Please select a version')
      return
    }
    if (license && !confirm('This will revoke the current license and generate a new one. Continue?')) return
    try {
      setActionLoading(true)
      setError(null)
      setKeyVisible(false)
      const data = await adminAPI.generateAdminLicense(strategyId, selectedVersion)
      setLicense(data)
    } catch (err) {
      setError('Failed to generate license')
      console.error(err)
    } finally {
      setActionLoading(false)
    }
  }

  const handleRevoke = async () => {
    if (!license) return
    if (!confirm('Are you sure you want to revoke this license?')) return
    try {
      setActionLoading(true)
      setError(null)
      await adminAPI.revokeAdminLicense(license.uuid)
      setLicense(null)
      setKeyVisible(false)
    } catch (err) {
      setError('Failed to revoke license')
      console.error(err)
    } finally {
      setActionLoading(false)
    }
  }

  const handleCopy = async () => {
    if (!license) return
    await navigator.clipboard.writeText(license.uuid)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const maskedKey = license ? '••••••••-••••-••••-••••-••••••••••••' : ''

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-surface border border-border rounded-lg max-w-lg w-full">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border">
          <div className="flex items-center gap-2">
            <Key size={20} className="text-primary" />
            <h2 className="text-xl font-bold text-text">Admin License</h2>
          </div>
          <button onClick={onClose} className="p-1 hover:bg-surface-hover rounded text-text-secondary hover:text-text">
            <X size={20} />
          </button>
        </div>

        <div className="p-6 space-y-5">
          <p className="text-text-secondary text-sm">
            Strategy: <span className="text-text font-medium">{strategyName}</span>
          </p>

          {error && <div className="text-red-400 text-sm bg-red-900/20 px-3 py-2 rounded-lg">{error}</div>}

          {loading ? (
            <div className="text-text-secondary text-sm">Loading...</div>
          ) : license ? (
            <div className="space-y-4">
              {/* Status + expiry + version */}
              <div className="flex items-center gap-4 flex-wrap">
                <div className="flex items-center gap-2">
                  <span className="text-sm text-text-secondary">Status:</span>
                  <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-green-900/30 text-green-400 capitalize">
                    {license.status}
                  </span>
                </div>
                <div className="text-sm text-text-secondary">
                  Expires: <span className="text-text">{new Date(license.expires_at).toLocaleDateString()}</span>
                </div>
                {license.strategy_version && (
                  <div className="text-sm text-text-secondary">
                    Version: <span className="text-text font-semibold">v{license.strategy_version}</span>
                  </div>
                )}
              </div>

              {/* License key row */}
              <div>
                <label className="block text-sm font-medium text-text-secondary mb-1">License Key</label>
                <div className="flex items-center gap-2 bg-surface-hover border border-border rounded-lg px-3 py-2">
                  <span className="flex-1 font-mono text-sm text-text break-all select-all">
                    {keyVisible ? license.uuid : maskedKey}
                  </span>
                  {/* Eye toggle */}
                  <button
                    onClick={() => setKeyVisible((v) => !v)}
                    className="p-1 text-text-secondary hover:text-text transition-colors flex-shrink-0"
                    title={keyVisible ? 'Hide license key' : 'Show license key'}
                  >
                    {keyVisible ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                  {/* Copy */}
                  <button
                    onClick={handleCopy}
                    className="p-1 text-text-secondary hover:text-primary transition-colors flex-shrink-0"
                    title="Copy license key"
                  >
                    {copied ? <Check size={16} className="text-green-400" /> : <Copy size={16} />}
                  </button>
                </div>
              </div>

              {/* Version Selection */}
              <div className="space-y-2 pt-2">
                <label className="block text-sm font-medium text-text">Strategy Version</label>
                <select
                  value={selectedVersion || ''}
                  onChange={(e) => setSelectedVersion(Number(e.target.value))}
                  className="w-full px-3 py-2 bg-surface border border-border rounded-lg text-text text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                  disabled={!versions || versions.versions.length === 0}
                >
                  {versions?.versions.map(({ version, best_return_pct }) => {
                    const returnStr = best_return_pct != null
                      ? ` · ${best_return_pct >= 0 ? '+' : ''}${best_return_pct.toFixed(1)}% return`
                      : ''
                    return (
                      <option key={version} value={version}>
                        Version {version}{returnStr}{version === versions.current_version ? ' (Latest)' : ''}
                      </option>
                    )
                  })}
                </select>
                {versions && (
                  <p className="text-xs text-text-secondary">
                    {versions.total_builds} completed build{versions.total_builds !== 1 ? 's' : ''} available
                  </p>
                )}
              </div>

              {/* Action buttons */}
              <div className="flex gap-2 pt-2">
                <button
                  onClick={handleRevoke}
                  disabled={actionLoading}
                  className="flex items-center gap-1.5 px-4 py-2 bg-red-900/20 border border-red-800/40 text-red-400 rounded-lg hover:bg-red-900/40 disabled:opacity-50 text-sm transition-colors"
                >
                  <XCircle size={15} />
                  {actionLoading ? 'Revoking...' : 'Revoke'}
                </button>
                <button
                  onClick={handleGenerate}
                  disabled={actionLoading || !selectedVersion}
                  className="flex items-center gap-1.5 px-4 py-2 bg-surface border border-border text-text rounded-lg hover:bg-surface-hover disabled:opacity-50 text-sm transition-colors ml-auto"
                >
                  <RefreshCw size={15} />
                  {actionLoading ? 'Generating...' : 'Generate New'}
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <p className="text-text-secondary text-sm">No active admin license exists for this strategy.</p>

              {/* Version Selection */}
              <div className="space-y-2">
                <label className="block text-sm font-medium text-text">Strategy Version</label>
                <select
                  value={selectedVersion || ''}
                  onChange={(e) => setSelectedVersion(Number(e.target.value))}
                  className="w-full px-3 py-2 bg-surface border border-border rounded-lg text-text text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                  disabled={!versions || versions.versions.length === 0}
                >
                  {versions?.versions.map(({ version, best_return_pct }) => {
                    const returnStr = best_return_pct != null
                      ? ` · ${best_return_pct >= 0 ? '+' : ''}${best_return_pct.toFixed(1)}% return`
                      : ''
                    return (
                      <option key={version} value={version}>
                        Version {version}{returnStr}{version === versions.current_version ? ' (Latest)' : ''}
                      </option>
                    )
                  })}
                </select>
                {versions && (
                  <p className="text-xs text-text-secondary">
                    {versions.total_builds} completed build{versions.total_builds !== 1 ? 's' : ''} available
                  </p>
                )}
              </div>

              <button
                onClick={handleGenerate}
                disabled={actionLoading || !selectedVersion}
                className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-hover disabled:opacity-50 text-sm transition-colors"
              >
                <Key size={15} />
                {actionLoading ? 'Generating...' : 'Generate License'}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

