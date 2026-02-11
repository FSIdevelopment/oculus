'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/contexts/AuthContext'
import api from '@/lib/api'
import { AlertCircle, Loader2, X } from 'lucide-react'

const STRATEGY_TYPES = [
  'Bollinger Band Breakout',
  'Dual Moving Average Crossover',
  'Fibonacci Retracement',
  'MACD Divergence',
  'Mean Reversion',
  'Momentum',
  'Moving Average Convergence',
  'Pairs Trading',
  'Price Action',
  'RSI Overbought/Oversold',
  'Scalping',
  'Sentiment Analysis',
  'Statistical Arbitrage',
  'Support/Resistance Breakout',
  'Swing Trading',
  'Trend Following',
  'VWAP',
  'Volatility Breakout',
  'Volume Profile',
  'Williams %R',
]

const ESTIMATED_TOKEN_COST = 50

export default function CreateStrategyPage() {
  const router = useRouter()
  const { user } = useAuth()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [symbolInput, setSymbolInput] = useState('')
  const [symbols, setSymbols] = useState<string[]>([])

  const [formData, setFormData] = useState({
    name: '',
    strategy_type: '',
    target_return: 15,
    comments: '',
    llm_enabled: true,
  })

  const handleAddSymbol = () => {
    const symbol = symbolInput.toUpperCase().trim()
    if (symbol && !symbols.includes(symbol)) {
      setSymbols([...symbols, symbol])
      setSymbolInput('')
    }
  }

  const handleRemoveSymbol = (sym: string) => {
    setSymbols(symbols.filter((s) => s !== sym))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    // Validation
    if (!formData.name.trim()) {
      setError('Strategy name is required')
      return
    }
    if (!formData.strategy_type) {
      setError('Strategy type is required')
      return
    }
    if (symbols.length === 0) {
      setError('At least one symbol is required')
      return
    }

    // Check token balance
    if (!user || user.balance < ESTIMATED_TOKEN_COST) {
      setError(
        `Insufficient tokens. You have ${user?.balance || 0}, but need at least ${ESTIMATED_TOKEN_COST}`
      )
      return
    }

    setLoading(true)

    try {
      // Step 1: Create strategy
      const strategyRes = await api.post('/api/strategies', {
        name: formData.name,
        description: formData.comments,
        config: {
          strategy_type: formData.strategy_type,
          symbols,
          target_return: formData.target_return,
          llm_enabled: formData.llm_enabled,
        },
      })

      const strategyId = strategyRes.data.uuid

      // Step 2: Trigger build
      await api.post('/api/builds/trigger', null, {
        params: {
          strategy_id: strategyId,
          description: formData.comments,
          target_return: formData.target_return,
        },
      })

      // Step 3: Redirect to build page
      router.push(`/dashboard/build/${strategyId}`)
    } catch (err: any) {
      const message = err.response?.data?.detail || err.message || 'Failed to create strategy'
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-text mb-2">Create Strategy</h1>
        <p className="text-text-secondary">
          Design a new trading strategy with AI assistance
        </p>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="bg-red-900/20 border border-red-700 rounded-lg p-4 flex gap-3">
          <AlertCircle className="text-red-500 flex-shrink-0" size={20} />
          <p className="text-red-200">{error}</p>
        </div>
      )}

      {/* Form */}
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="bg-surface border border-border rounded-lg p-6 space-y-6">
          {/* Strategy Name */}
          <div>
            <label className="block text-sm font-medium text-text mb-2">
              Strategy Name *
            </label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              placeholder="e.g., Tech Momentum Strategy"
              className="w-full bg-background border border-border rounded px-3 py-2 text-text placeholder-text-secondary focus:outline-none focus:border-primary"
            />
          </div>

          {/* Strategy Type */}
          <div>
            <label className="block text-sm font-medium text-text mb-2">
              Strategy Type *
            </label>
            <select
              value={formData.strategy_type}
              onChange={(e) => setFormData({ ...formData, strategy_type: e.target.value })}
              className="w-full bg-background border border-border rounded px-3 py-2 text-text focus:outline-none focus:border-primary"
            >
              <option value="">Select a strategy type</option>
              {STRATEGY_TYPES.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
          </div>

          {/* Symbols */}
          <div>
            <label className="block text-sm font-medium text-text mb-2">
              Trading Symbols *
            </label>
            <div className="flex gap-2 mb-3">
              <input
                type="text"
                value={symbolInput}
                onChange={(e) => setSymbolInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), handleAddSymbol())}
                placeholder="e.g., AAPL"
                className="flex-1 bg-background border border-border rounded px-3 py-2 text-text placeholder-text-secondary focus:outline-none focus:border-primary"
              />
              <button
                type="button"
                onClick={handleAddSymbol}
                className="px-4 py-2 bg-primary hover:bg-primary-hover text-white rounded font-medium transition-colors"
              >
                Add
              </button>
            </div>
            {symbols.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {symbols.map((sym) => (
                  <div
                    key={sym}
                    className="bg-primary/20 border border-primary rounded px-3 py-1 flex items-center gap-2"
                  >
                    <span className="text-primary font-medium">{sym}</span>
                    <button
                      type="button"
                      onClick={() => handleRemoveSymbol(sym)}
                      className="text-primary hover:text-primary-hover"
                    >
                      <X size={16} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Target Annual Return */}
          <div>
            <label className="block text-sm font-medium text-text mb-2">
              Target Annual Return (%) *
            </label>
            <input
              type="number"
              value={formData.target_return}
              onChange={(e) => setFormData({ ...formData, target_return: parseFloat(e.target.value) })}
              min="1"
              max="500"
              className="w-full bg-background border border-border rounded px-3 py-2 text-text focus:outline-none focus:border-primary"
            />
          </div>

          {/* Comments */}
          <div>
            <label className="block text-sm font-medium text-text mb-2">
              Comments / Customizations
            </label>
            <textarea
              value={formData.comments}
              onChange={(e) => setFormData({ ...formData, comments: e.target.value })}
              placeholder="Any additional instructions for the AI..."
              rows={4}
              className="w-full bg-background border border-border rounded px-3 py-2 text-text placeholder-text-secondary focus:outline-none focus:border-primary resize-none"
            />
          </div>

          {/* LLM Enabled */}
          <div className="flex items-center gap-3">
            <input
              type="checkbox"
              id="llm_enabled"
              checked={formData.llm_enabled}
              onChange={(e) => setFormData({ ...formData, llm_enabled: e.target.checked })}
              className="w-4 h-4 rounded border-border bg-background cursor-pointer"
            />
            <label htmlFor="llm_enabled" className="text-sm font-medium text-text cursor-pointer">
              Enable LLM-assisted design
            </label>
          </div>
        </div>

        {/* Token Cost */}
        <div className="bg-surface border border-border rounded-lg p-4">
          <p className="text-sm text-text-secondary mb-2">Estimated Token Cost</p>
          <p className="text-2xl font-bold text-primary">~{ESTIMATED_TOKEN_COST} tokens</p>
          <p className="text-xs text-text-secondary mt-2">
            Your balance: <span className="text-primary font-medium">{user?.balance || 0}</span> tokens
          </p>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading}
          className="w-full py-3 bg-primary hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold rounded-lg transition-colors flex items-center justify-center gap-2"
        >
          {loading && <Loader2 size={20} className="animate-spin" />}
          {loading ? 'Creating Strategy...' : 'Create & Build Strategy'}
        </button>
      </form>
    </div>
  )
}

