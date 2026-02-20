'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { loadStripe, Stripe } from '@stripe/stripe-js'
import { Elements, PaymentElement, useStripe, useElements } from '@stripe/react-stripe-js'
import { licenseAPI } from '@/lib/api'
import { AlertCircle, ChevronLeft } from 'lucide-react'

interface LicensePaymentFormProps {
  strategyId: string
  strategyName: string
  licenseType: 'monthly' | 'annual'
  monthlyPrice: number
  annualPrice: number
  onSuccess: () => void
  onCancel: () => void
}

export default function LicensePaymentForm({
  strategyId,
  strategyName,
  licenseType,
  monthlyPrice,
  annualPrice,
  onSuccess,
  onCancel,
}: LicensePaymentFormProps) {
  const [clientSecret, setClientSecret] = useState<string | null>(null)
  const [stripePromise, setStripePromise] = useState<Promise<Stripe | null> | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const intentCreated = useRef(false)

  const price = licenseType === 'annual' ? annualPrice : monthlyPrice
  const priceLabel = licenseType === 'annual'
    ? `$${annualPrice} / year (~2 months free)`
    : `$${monthlyPrice} / month`

  const initSetupIntent = useCallback(async () => {
    try {
      const data = await licenseAPI.createSetupIntent()
      setClientSecret(data.client_secret)
      setStripePromise(loadStripe(data.publishable_key))
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to initialise payment. Please try again.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (intentCreated.current) return
    intentCreated.current = true
    initSetupIntent()
  }, [initSetupIntent])

  if (loading) {
    return (
      <div className="rounded-lg border border-border bg-surface-hover p-6 text-center text-text-secondary text-sm">
        Loading payment form…
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-700/50 bg-red-900/10 p-4">
        <div className="flex items-center gap-2 text-red-400 mb-3">
          <AlertCircle size={16} />
          <p className="text-sm">{error}</p>
        </div>
        <button
          onClick={onCancel}
          className="text-sm text-text-secondary hover:text-text underline"
        >
          Go back
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Summary header */}
      <div className="flex items-center gap-3">
        <button
          onClick={onCancel}
          className="text-text-secondary hover:text-text transition-colors"
          aria-label="Back"
        >
          <ChevronLeft size={18} />
        </button>
        <div>
          <p className="text-sm font-semibold text-text capitalize">{licenseType} License — {strategyName}</p>
          <p className="text-primary font-bold">{priceLabel}</p>
        </div>
      </div>

      {clientSecret && stripePromise && (
        <Elements stripe={stripePromise} options={{ clientSecret }}>
          <SubscribeForm
            strategyId={strategyId}
            licenseType={licenseType}
            price={price}
            onSuccess={onSuccess}
          />
        </Elements>
      )}
    </div>
  )
}



// ─── Inner form rendered inside Stripe Elements ───────────────────────────────

function SubscribeForm({
  strategyId,
  licenseType,
  price,
  onSuccess,
}: {
  strategyId: string
  licenseType: 'monthly' | 'annual'
  price: number
  onSuccess: () => void
}) {
  const stripe = useStripe()
  const elements = useElements()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!stripe || !elements) return

    setLoading(true)
    setError(null)

    try {
      // 1. Validate the Elements form
      const { error: submitError } = await elements.submit()
      if (submitError) {
        setError(submitError.message || 'Card validation failed')
        setLoading(false)
        return
      }

      // 2. Confirm the SetupIntent to save the payment method
      const { setupIntent, error: confirmError } = await stripe.confirmSetup({
        elements,
        redirect: 'if_required',
      })

      if (confirmError) {
        setError(confirmError.message || 'Failed to confirm payment method')
        setLoading(false)
        return
      }

      if (!setupIntent?.payment_method) {
        setError('Payment method was not captured. Please try again.')
        setLoading(false)
        return
      }

      const paymentMethodId =
        typeof setupIntent.payment_method === 'string'
          ? setupIntent.payment_method
          : setupIntent.payment_method.id

      // 3. Call backend to create the subscription + license
      await licenseAPI.subscribeWithPaymentMethod(strategyId, licenseType, paymentMethodId)
      onSuccess()
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Subscription failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <PaymentElement />

      {error && (
        <div className="flex items-start gap-2 p-3 bg-red-900/20 border border-red-700/50 rounded-lg text-red-400">
          <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
          <p className="text-sm">{error}</p>
        </div>
      )}

      <button
        type="submit"
        disabled={loading || !stripe}
        className="w-full py-3 bg-gradient-to-r from-[#007cf0] to-[#00dfd8] hover:shadow-lg text-white font-semibold rounded-lg transition-all disabled:opacity-50"
      >
        {loading ? 'Processing…' : `Subscribe — $${price}`}
      </button>

      <p className="text-xs text-text-secondary text-center">
        Recurring {licenseType} subscription · Cancel any time from your account
      </p>
    </form>
  )
}
