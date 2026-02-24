'use client'

import { useEffect, useState } from 'react'
import { connectAPI } from '@/lib/api'
import { ExternalLink, CreditCard, CheckCircle, Loader, AlertCircle } from 'lucide-react'

interface ConnectStatus {
  status: 'not_started' | 'pending' | 'complete'
  account_id?: string
  requirements_due?: string[]
  disabled_reason?: string | null
}

interface StrategyEarning {
  strategy_id: string
  strategy_name: string
  total_earned: number
  payment_count: number
}

interface Earnings {
  total_earned: number
  balance_available: number
  balance_pending: number
  by_strategy: StrategyEarning[]
}

function formatCurrency(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
  }).format(amount)
}

export default function EarningsTab() {
  const [loading, setLoading] = useState(true)
  const [connectStatus, setConnectStatus] = useState<ConnectStatus | null>(null)
  const [earnings, setEarnings] = useState<Earnings | null>(null)
  const [earningsLoading, setEarningsLoading] = useState(false)
  const [earningsError, setEarningsError] = useState<string | null>(null)
  const [onboarding, setOnboarding] = useState(false)

  useEffect(() => {
    fetchConnectStatus()
  }, [])

  const fetchConnectStatus = async () => {
    try {
      const status = await connectAPI.connectStatus()
      setConnectStatus(status)
      if (status.status === 'complete') {
        fetchEarnings()
      }
    } catch (error) {
      console.error('Failed to fetch connect status:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchEarnings = async () => {
    setEarningsLoading(true)
    setEarningsError(null)
    try {
      const data = await connectAPI.getEarnings()
      setEarnings(data)
    } catch (err: any) {
      setEarningsError(err.response?.data?.detail || 'Failed to load earnings')
    } finally {
      setEarningsLoading(false)
    }
  }

  const handleSetupPayouts = async () => {
    setOnboarding(true)
    try {
      const response = await connectAPI.connectOnboard()
      window.location.href = response.url
    } catch (error) {
      console.error('Failed to start onboarding:', error)
      setOnboarding(false)
    }
  }

  const handleViewDashboard = async () => {
    try {
      const response = await connectAPI.connectDashboardLink()
      window.location.href = response.url
    } catch (error) {
      console.error('Failed to get dashboard link:', error)
    }
  }

  const handleUpdateSettings = async () => {
    setOnboarding(true)
    try {
      const response = await connectAPI.connectOnboard()
      window.location.href = response.url
    } catch (error) {
      console.error('Failed to get settings link:', error)
      setOnboarding(false)
    }
  }

  if (loading) {
    return <div className="text-text-secondary">Loading earnings...</div>
  }

  // Not started
  if (connectStatus?.status === 'not_started') {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-blue-900/30 to-blue-800/30 border border-blue-700 rounded-lg p-8">
          <div className="flex items-start gap-4">
            <CreditCard className="text-blue-400 flex-shrink-0 mt-1" size={24} />
            <div className="flex-1">
              <h3 className="text-xl font-bold text-text mb-2">Set Up Marketplace Payouts</h3>
              <p className="text-text-secondary mb-4">
                Earn 65% revenue share from strategy subscriptions. Set up your payout account to start receiving payments.
              </p>
              <button
                onClick={handleSetupPayouts}
                disabled={onboarding}
                className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-600/50 text-white font-medium py-2 px-6 rounded-lg transition-colors"
              >
                {onboarding ? (
                  <>
                    <Loader size={18} className="animate-spin" />
                    Setting up...
                  </>
                ) : (
                  <>
                    <CreditCard size={18} />
                    Set Up Payouts
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
        <div className="bg-surface border border-border rounded-lg p-6">
          <h3 className="text-lg font-bold text-text mb-4">Earnings by Strategy</h3>
          <p className="text-text-secondary">No strategies with earnings yet</p>
        </div>
      </div>
    )
  }

  // Pending
  if (connectStatus?.status === 'pending') {
    const requirementsDue = connectStatus.requirements_due ?? []
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-yellow-900/30 to-yellow-800/30 border border-yellow-700 rounded-lg p-8">
          <div className="flex items-start gap-4">
            <Loader className="text-yellow-400 flex-shrink-0 mt-1 animate-spin" size={24} />
            <div className="flex-1">
              <h3 className="text-xl font-bold text-text mb-2">Account Setup In Progress</h3>
              <p className="text-text-secondary mb-4">
                Your payout account is being reviewed. This usually takes 1-2 business days.
              </p>
              <button
                onClick={handleSetupPayouts}
                className="flex items-center gap-2 bg-yellow-600 hover:bg-yellow-700 text-white font-medium py-2 px-6 rounded-lg transition-colors"
              >
                <ExternalLink size={18} />
                Complete Setup
              </button>
            </div>
          </div>
        </div>

        {requirementsDue.length > 0 && (
          <div className="flex items-start gap-3 p-4 bg-orange-900/20 border border-orange-700 rounded-lg">
            <AlertCircle className="text-orange-400 flex-shrink-0 mt-0.5" size={20} />
            <div>
              <p className="text-orange-300 font-medium mb-1">Stripe requires additional information</p>
              <p className="text-text-secondary text-sm">
                Complete Setup above to provide the required details so your account can be activated.
              </p>
            </div>
          </div>
        )}

        <div className="bg-surface border border-border rounded-lg p-6">
          <h3 className="text-lg font-bold text-text mb-4">Earnings by Strategy</h3>
          <p className="text-text-secondary">No strategies with earnings yet</p>
        </div>
      </div>
    )
  }

  // Complete — show real earnings
  const requirementsDue = connectStatus?.requirements_due ?? []
  return (
    <div className="space-y-6">
      {/* Connect active banner */}
      <div className="flex items-center justify-between bg-gradient-to-r from-green-900/30 to-green-800/30 border border-green-700 rounded-lg p-6">
        <div className="flex items-center gap-3">
          <CheckCircle className="text-green-400" size={24} />
          <div>
            <h3 className="text-lg font-bold text-text">Payout Account Active</h3>
            <p className="text-text-secondary text-sm">You&apos;re ready to receive payments</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={handleUpdateSettings}
            disabled={onboarding}
            className="flex items-center gap-2 border border-border text-text-secondary hover:text-text hover:border-text-secondary disabled:opacity-50 font-medium py-2 px-4 rounded-lg transition-colors text-sm"
          >
            {onboarding ? <Loader size={16} className="animate-spin" /> : <CreditCard size={16} />}
            Update Settings
          </button>
          <button
            onClick={handleViewDashboard}
            className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-6 rounded-lg transition-colors"
          >
            <ExternalLink size={18} />
            View Dashboard
          </button>
        </div>
      </div>

      {/* Requirements warning */}
      {requirementsDue.length > 0 && (
        <div className="flex items-start gap-3 p-4 bg-orange-900/20 border border-orange-700 rounded-lg">
          <AlertCircle className="text-orange-400 flex-shrink-0 mt-0.5" size={20} />
          <div className="flex-1">
            <p className="text-orange-300 font-medium mb-1">Action required — Stripe needs more information</p>
            <p className="text-text-secondary text-sm mb-3">
              Your payouts may be restricted until you provide the required details.
            </p>
            <button
              onClick={handleUpdateSettings}
              disabled={onboarding}
              className="flex items-center gap-2 bg-orange-600 hover:bg-orange-700 disabled:opacity-50 text-white font-medium py-1.5 px-4 rounded-lg transition-colors text-sm"
            >
              <ExternalLink size={14} />
              Update Payout Settings
            </button>
          </div>
        </div>
      )}

      {/* Earnings error */}
      {earningsError && (
        <div className="flex items-center gap-3 p-4 bg-red-900/20 border border-red-700 rounded-lg text-red-400">
          <AlertCircle size={20} />
          {earningsError}
        </div>
      )}

      {/* Summary cards */}
      {earningsLoading ? (
        <div className="flex items-center justify-center py-8">
          <Loader className="animate-spin text-primary" size={28} />
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-surface-hover border border-border rounded-lg p-6">
              <p className="text-text-secondary text-sm mb-2">Total Earnings</p>
              <p className="text-3xl font-bold text-primary">
                {formatCurrency(earnings?.total_earned ?? 0)}
              </p>
            </div>
            <div className="bg-surface-hover border border-border rounded-lg p-6">
              <p className="text-text-secondary text-sm mb-2">Available in Stripe</p>
              <p className="text-3xl font-bold text-green-400">
                {formatCurrency(earnings?.balance_available ?? 0)}
              </p>
            </div>
            <div className="bg-surface-hover border border-border rounded-lg p-6">
              <p className="text-text-secondary text-sm mb-2">Pending</p>
              <p className="text-3xl font-bold text-yellow-400">
                {formatCurrency(earnings?.balance_pending ?? 0)}
              </p>
            </div>
          </div>

          {/* Per-strategy breakdown */}
          <div className="bg-surface border border-border rounded-lg p-6">
            <h3 className="text-lg font-bold text-text mb-4">Earnings by Strategy</h3>
            {!earnings?.by_strategy?.length ? (
              <p className="text-text-secondary">No strategies with earnings yet</p>
            ) : (
              <div className="space-y-3">
                {earnings.by_strategy.map((s) => (
                  <div
                    key={s.strategy_id}
                    className="flex items-center justify-between py-3 border-b border-border last:border-0"
                  >
                    <div>
                      <p className="text-text font-medium">{s.strategy_name}</p>
                      <p className="text-text-secondary text-sm">
                        {s.payment_count} {s.payment_count === 1 ? 'payment' : 'payments'}
                      </p>
                    </div>
                    <p className="text-primary font-bold text-lg">
                      {formatCurrency(s.total_earned)}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
