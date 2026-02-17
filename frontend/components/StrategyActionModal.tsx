'use client'

import { useState } from 'react'
import { X, Copy, Check } from 'lucide-react'

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

interface StrategyActionModalProps {
  isOpen: boolean
  onClose: () => void
  strategyName: string
  dockerInfo?: DockerInfo
  licenses?: License[]
  onMarketplaceSubmit?: (price: number) => Promise<void>
  onLicensePurchase?: (type: 'monthly' | 'annual') => Promise<void>
}

export default function StrategyActionModal({
  isOpen,
  onClose,
  strategyName,
  dockerInfo,
  licenses = [],
  onMarketplaceSubmit,
  onLicensePurchase,
}: StrategyActionModalProps) {
  const [activeTab, setActiveTab] = useState<'partner' | 'marketplace' | 'docker'>('partner')
  const [marketplacePrice, setMarketplacePrice] = useState(10)
  const [copiedCommand, setCopiedCommand] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  if (!isOpen) return null

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedCommand(id)
    setTimeout(() => setCopiedCommand(null), 2000)
  }

  const handleMarketplaceSubmit = async () => {
    if (!onMarketplaceSubmit) return
    setIsSubmitting(true)
    try {
      await onMarketplaceSubmit(marketplacePrice)
      onClose()
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleLicensePurchase = async (type: 'monthly' | 'annual') => {
    if (!onLicensePurchase) return
    setIsSubmitting(true)
    try {
      await onLicensePurchase(type)
    } finally {
      setIsSubmitting(false)
    }
  }

  const activeLicense = licenses.find(l => l.status === 'active')
  const licenseExpired = activeLicense && new Date(activeLicense.expires_at) < new Date()

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-surface border border-border rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border sticky top-0 bg-surface">
          <h2 className="text-2xl font-bold text-text">Deploy {strategyName}</h2>
          <button
            onClick={onClose}
            className="text-text-secondary hover:text-text transition-colors"
          >
            <X size={24} />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-border">
          {(['partner', 'marketplace', 'docker'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`flex-1 px-4 py-3 font-medium transition-colors ${
                activeTab === tab
                  ? 'text-primary border-b-2 border-primary'
                  : 'text-text-secondary hover:text-text'
              }`}
            >
              {tab === 'partner' && 'Partner Host'}
              {tab === 'marketplace' && 'Marketplace'}
              {tab === 'docker' && 'Docker'}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Partner Host Tab */}
          {activeTab === 'partner' && (
            <div className="space-y-4">
              <p className="text-text-secondary">
                Deploy your strategy to a partner host for automated trading execution.
              </p>
              <div className="bg-surface-hover border border-border rounded-lg p-4 space-y-3">
                <p className="text-sm font-medium text-text">Supported Partners:</p>
                <ul className="space-y-2 text-sm text-text-secondary">
                  <li>• <a href="https://www.atlastrade.ai" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">Atlas Trade AI</a></li>
                  <li>• <a href="https://www.signalsynk.com" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">Signal Synk</a></li>
                </ul>
              </div>
              <p className="text-sm text-text-secondary">
                <strong>Note:</strong> You must have an active account with the partner platform to deploy.
              </p>
            </div>
          )}

          {/* Marketplace Tab */}
          {activeTab === 'marketplace' && (
            <div className="space-y-4">
              <p className="text-text-secondary">
                List your strategy on the Oculus marketplace for other traders to subscribe.
              </p>
              <div className="space-y-3">
                <label className="block">
                  <span className="text-sm font-medium text-text mb-2 block">Monthly Subscription Price ($)</span>
                  <input
                    type="number"
                    min="1"
                    max="1000"
                    value={marketplacePrice}
                    onChange={(e) => setMarketplacePrice(Number(e.target.value))}
                    className="w-full bg-surface-hover border border-border rounded-lg px-4 py-2 text-text focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                </label>
                <div className="bg-surface-hover border border-border rounded-lg p-4">
                  <p className="text-sm text-text-secondary mb-2">Revenue Share:</p>
                  <div className="space-y-1 text-sm">
                    <p className="text-text">You receive: <span className="text-primary font-bold">${(marketplacePrice * 0.65).toFixed(2)}/month</span></p>
                    <p className="text-text-secondary">Platform fee: ${(marketplacePrice * 0.35).toFixed(2)}/month</p>
                  </div>
                </div>
              </div>
              <button
                onClick={handleMarketplaceSubmit}
                disabled={isSubmitting}
                className="w-full bg-gradient-to-r from-[#007cf0] to-[#00dfd8] disabled:opacity-50 text-white font-medium py-2 rounded-lg transition-all hover:shadow-lg"
              >
                {isSubmitting ? 'Listing...' : 'List on Marketplace'}
              </button>
            </div>
          )}

          {/* Docker Tab */}
          {activeTab === 'docker' && (
            <div className="space-y-4">
              <p className="text-text-secondary">
                Run your strategy locally using Docker. Requires a valid license.
              </p>

              {/* License Status */}
              <div className={`border rounded-lg p-4 ${
                activeLicense && !licenseExpired
                  ? 'bg-green-900/20 border-green-700'
                  : 'bg-yellow-900/20 border-yellow-700'
              }`}>
                <p className="text-sm font-medium text-text mb-1">License Status:</p>
                {activeLicense && !licenseExpired ? (
                  <p className="text-sm text-green-400">
                    ✓ Active ({activeLicense.license_type}) - Expires {new Date(activeLicense.expires_at).toLocaleDateString()}
                  </p>
                ) : (
                  <p className="text-sm text-yellow-400">
                    {activeLicense ? 'License expired' : 'No active license'}
                  </p>
                )}
              </div>

              {/* Docker Commands */}
              {dockerInfo && (
                <div className="space-y-3">
                  <div>
                    <p className="text-sm font-medium text-text mb-2">Pull Command:</p>
                    <div className="flex items-center gap-2">
                      <code className="flex-1 bg-surface-hover border border-border rounded px-3 py-2 text-sm text-text-secondary overflow-x-auto">
                        {dockerInfo.pull_command}
                      </code>
                      <button
                        onClick={() => copyToClipboard(dockerInfo.pull_command, 'pull')}
                        className="p-2 hover:bg-surface-hover rounded transition-colors"
                      >
                        {copiedCommand === 'pull' ? (
                          <Check size={18} className="text-green-400" />
                        ) : (
                          <Copy size={18} className="text-text-secondary" />
                        )}
                      </button>
                    </div>
                  </div>

                  <div>
                    <p className="text-sm font-medium text-text mb-2">Run Command:</p>
                    <div className="flex items-center gap-2">
                      <code className="flex-1 bg-surface-hover border border-border rounded px-3 py-2 text-sm text-text-secondary overflow-x-auto">
                        {dockerInfo.run_command}
                      </code>
                      <button
                        onClick={() => copyToClipboard(dockerInfo.run_command, 'run')}
                        className="p-2 hover:bg-surface-hover rounded transition-colors"
                      >
                        {copiedCommand === 'run' ? (
                          <Check size={18} className="text-green-400" />
                        ) : (
                          <Copy size={18} className="text-text-secondary" />
                        )}
                      </button>
                    </div>
                  </div>

                  <div>
                    <p className="text-sm font-medium text-text mb-2">Environment Variables:</p>
                    <div className="bg-surface-hover border border-border rounded p-3 space-y-2 text-sm">
                      {Object.entries(dockerInfo.environment_variables).map(([key, value]) => (
                        <div key={key}>
                          <p className="text-text-secondary">{key}</p>
                          <p className="text-text text-xs">{value}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* License Purchase */}
              {(!activeLicense || licenseExpired) && (
                <div className="space-y-3">
                  <p className="text-sm font-medium text-text">Purchase a License:</p>
                  <div className="grid grid-cols-2 gap-3">
                    <button
                      onClick={() => handleLicensePurchase('monthly')}
                      disabled={isSubmitting}
                      className="bg-gradient-to-r from-[#007cf0] to-[#00dfd8] disabled:opacity-50 text-white font-medium py-2 rounded-lg transition-all hover:shadow-lg text-sm"
                    >
                      Monthly
                    </button>
                    <button
                      onClick={() => handleLicensePurchase('annual')}
                      disabled={isSubmitting}
                      className="bg-gradient-to-r from-[#007cf0] to-[#00dfd8] disabled:opacity-50 text-white font-medium py-2 rounded-lg transition-all hover:shadow-lg text-sm"
                    >
                      Annual
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

