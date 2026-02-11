'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { connectAPI } from '@/lib/api'
import { CheckCircle, AlertCircle, Loader } from 'lucide-react'

interface ConnectStatus {
  status: 'not_started' | 'pending' | 'complete'
  account_id?: string
}

export default function ConnectReturnPage() {
  const router = useRouter()
  const [status, setStatus] = useState<ConnectStatus | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    checkStatus()
  }, [])

  const checkStatus = async () => {
    try {
      const result = await connectAPI.connectStatus()
      setStatus(result)
    } catch (error) {
      console.error('Failed to check status:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!loading && status) {
      // Auto-redirect after 3 seconds
      const timer = setTimeout(() => {
        router.push('/dashboard/account?tab=earnings')
      }, 3000)
      return () => clearTimeout(timer)
    }
    return
  }, [loading, status, router])

  if (loading) {
    return (
      <div className="min-h-screen bg-surface flex items-center justify-center">
        <div className="text-center">
          <Loader className="animate-spin text-primary mx-auto mb-4" size={48} />
          <p className="text-text-secondary">Setting up your payout account...</p>
        </div>
      </div>
    )
  }

  if (status?.status === 'complete') {
    return (
      <div className="min-h-screen bg-surface flex items-center justify-center p-4">
        <div className="bg-surface-hover border border-border rounded-lg p-8 max-w-md w-full text-center">
          <CheckCircle className="text-green-400 mx-auto mb-4" size={48} />
          <h1 className="text-2xl font-bold text-text mb-2">Setup Complete!</h1>
          <p className="text-text-secondary mb-6">
            Your payout account is ready. You can now receive payments from strategy subscriptions.
          </p>
          <button
            onClick={() => router.push('/dashboard/account?tab=earnings')}
            className="w-full bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-4 rounded-lg transition-colors"
          >
            Go to Earnings
          </button>
          <p className="text-text-secondary text-sm mt-4">
            Redirecting automatically in 3 seconds...
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-surface flex items-center justify-center p-4">
      <div className="bg-surface-hover border border-border rounded-lg p-8 max-w-md w-full text-center">
        <AlertCircle className="text-yellow-400 mx-auto mb-4" size={48} />
        <h1 className="text-2xl font-bold text-text mb-2">Almost Done!</h1>
        <p className="text-text-secondary mb-6">
          Your account is being reviewed. This usually takes 1-2 business days. We&apos;ll notify you when it&apos;s ready.
        </p>
        <button
          onClick={() => router.push('/dashboard/account?tab=earnings')}
          className="w-full bg-yellow-600 hover:bg-yellow-700 text-white font-medium py-2 px-4 rounded-lg transition-colors"
        >
          Go to Earnings
        </button>
        <p className="text-text-secondary text-sm mt-4">
          Redirecting automatically in 3 seconds...
        </p>
      </div>
    </div>
  )
}

