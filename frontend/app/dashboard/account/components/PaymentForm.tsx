'use client'

import { useState, useEffect, useCallback } from 'react'
import { loadStripe } from '@stripe/stripe-js'
import { Elements, PaymentElement, useStripe, useElements } from '@stripe/react-stripe-js'
import api from '@/lib/api'
import { AlertCircle, X } from 'lucide-react'

interface Product {
  uuid: string
  name: string
  price: number
  token_amount: number
}

interface PaymentFormProps {
  product: Product
  onSuccess: () => void
  onCancel: () => void
}

const stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY!)

export default function PaymentForm({ product, onSuccess, onCancel }: PaymentFormProps) {
  const [clientSecret, setClientSecret] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const createPaymentIntent = useCallback(async () => {
    try {
      const response = await api.post('/api/payments/create-intent', {
        product_id: product.uuid,
      })
      setClientSecret(response.data.client_secret)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to create payment intent')
    } finally {
      setLoading(false)
    }
  }, [product.uuid])

  useEffect(() => {
    createPaymentIntent()
  }, [createPaymentIntent])

  if (loading) {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-surface rounded-lg p-8 max-w-md w-full mx-4">
          <p className="text-text-secondary">Loading payment form...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-surface rounded-lg p-8 max-w-md w-full mx-4">
          <div className="flex items-center gap-3 text-red-400 mb-4">
            <AlertCircle size={20} />
            <p>{error}</p>
          </div>
          <button
            onClick={onCancel}
            className="w-full px-4 py-2 bg-primary hover:bg-primary-hover text-white font-medium rounded-lg"
          >
            Close
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-surface rounded-lg p-8 max-w-md w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-text">Purchase Tokens</h2>
          <button onClick={onCancel} className="text-text-secondary hover:text-text">
            <X size={24} />
          </button>
        </div>

        <div className="bg-background rounded-lg p-4 mb-6">
          <p className="text-text-secondary text-sm mb-1">Package</p>
          <p className="text-lg font-bold text-text mb-3">{product.name}</p>
          <p className="text-text-secondary text-sm mb-1">Amount</p>
          <p className="text-2xl font-bold text-primary">${product.price.toFixed(2)}</p>
        </div>

        {clientSecret && (
          <Elements stripe={stripePromise} options={{ clientSecret }}>
            <CheckoutForm product={product} onSuccess={onSuccess} onCancel={onCancel} />
          </Elements>
        )}
      </div>
    </div>
  )
}

function CheckoutForm({
  product,
  onSuccess,
  onCancel,
}: {
  product: Product
  onSuccess: () => void
  onCancel: () => void
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
      const { error: submitError } = await elements.submit()
      if (submitError) {
        setError(submitError.message || 'Payment failed')
        setLoading(false)
        return
      }

      const { paymentIntent, error: confirmError } = await stripe.confirmPayment({
        elements,
        redirect: 'if_required',
      })

      if (confirmError) {
        setError(confirmError.message || 'Payment failed')
        setLoading(false)
        return
      }

      if (paymentIntent && paymentIntent.status === 'succeeded') {
        // Confirm payment on backend
        await api.post('/api/payments/confirm', {
          payment_intent_id: paymentIntent.id,
          product_id: product.uuid,
        })
        onSuccess()
      }
    } catch (err: any) {
      setError(err.message || 'Payment failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <PaymentElement />

      {error && (
        <div className="flex items-center gap-2 p-3 bg-red-900/20 border border-red-700 rounded-lg text-red-400">
          <AlertCircle size={18} />
          <p className="text-sm">{error}</p>
        </div>
      )}

      <div className="flex gap-3">
        <button
          type="button"
          onClick={onCancel}
          disabled={loading}
          className="flex-1 px-4 py-2 border border-border text-text-secondary hover:text-text rounded-lg transition-colors disabled:opacity-50"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={loading || !stripe}
          className="flex-1 px-4 py-2 bg-primary hover:bg-primary-hover text-white font-medium rounded-lg transition-colors disabled:opacity-50"
        >
          {loading ? 'Processing...' : `Pay $${product.price.toFixed(2)}`}
        </button>
      </div>
    </form>
  )
}

