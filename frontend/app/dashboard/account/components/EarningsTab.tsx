'use client'

import { useEffect, useState } from 'react'
import { connectAPI } from '@/lib/api'
import { ExternalLink, CreditCard, CheckCircle, Loader } from 'lucide-react'

interface ConnectStatus {
  status: 'not_started' | 'pending' | 'complete'
  account_id?: string
}

export default function EarningsTab() {
  const [loading, setLoading] = useState(true)
  const [connectStatus, setConnectStatus] = useState<ConnectStatus | null>(null)
  const [onboarding, setOnboarding] = useState(false)

  useEffect(() => {
    fetchConnectStatus()
  }, [])

  const fetchConnectStatus = async () => {
    try {
      const status = await connectAPI.connectStatus()
      setConnectStatus(status)
    } catch (error) {
      console.error('Failed to fetch connect status:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSetupPayouts = async () => {
    setOnboarding(true)
    try {
      const response = await connectAPI.connectOnboard()
      // Redirect to Stripe onboarding
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

  if (loading) {
    return <div className="text-text-secondary">Loading earnings...</div>
  }

  // Not started - show setup card
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

  // Pending - show incomplete message
  if (connectStatus?.status === 'pending') {
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

        <div className="bg-surface border border-border rounded-lg p-6">
          <h3 className="text-lg font-bold text-text mb-4">Earnings by Strategy</h3>
          <p className="text-text-secondary">No strategies with earnings yet</p>
        </div>
      </div>
    )
  }

  // Complete - show earnings cards and dashboard link
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between bg-gradient-to-r from-green-900/30 to-green-800/30 border border-green-700 rounded-lg p-6">
        <div className="flex items-center gap-3">
          <CheckCircle className="text-green-400" size={24} />
          <div>
            <h3 className="text-lg font-bold text-text">Payout Account Active</h3>
            <p className="text-text-secondary text-sm">You&apos;re ready to receive payments</p>
          </div>
        </div>
        <button
          onClick={handleViewDashboard}
          className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-6 rounded-lg transition-colors"
        >
          <ExternalLink size={18} />
          View Dashboard
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-surface-hover border border-border rounded-lg p-6">
          <p className="text-text-secondary text-sm mb-2">Total Earnings</p>
          <p className="text-3xl font-bold text-primary">$0.00</p>
        </div>
        <div className="bg-surface-hover border border-border rounded-lg p-6">
          <p className="text-text-secondary text-sm mb-2">Pending</p>
          <p className="text-3xl font-bold text-yellow-400">$0.00</p>
        </div>
        <div className="bg-surface-hover border border-border rounded-lg p-6">
          <p className="text-text-secondary text-sm mb-2">Paid Out</p>
          <p className="text-3xl font-bold text-green-400">$0.00</p>
        </div>
      </div>

      <div className="bg-surface border border-border rounded-lg p-6">
        <h3 className="text-lg font-bold text-text mb-4">Earnings by Strategy</h3>
        <p className="text-text-secondary">No strategies with earnings yet</p>
      </div>
    </div>
  )
}

