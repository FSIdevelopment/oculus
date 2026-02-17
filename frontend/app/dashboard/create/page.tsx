'use client'

import { useState, useEffect } from 'react'
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

const TIMEFRAME_OPTIONS = [
  { value: '5m', label: '5 Minutes' },
  { value: '15m', label: '15 Minutes' },
  { value: '30m', label: '30 Minutes' },
  { value: '1h', label: '1 Hour' },
  { value: '4h', label: '4 Hours' },
  { value: '1d', label: '1 Day' },
]

export default function CreateStrategyPage() {
  const router = useRouter()
  const { user } = useAuth()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [symbolInput, setSymbolInput] = useState('')
  const [symbols, setSymbols] = useState<string[]>([])
  const [tokensPerIteration, setTokensPerIteration] = useState<number | null>(null)

  // Fetch per-iteration pricing from the backend on page load
  useEffect(() => {
    const fetchPricing = async () => {
      try {
        const res = await api.get('/api/builds/pricing')
        setTokensPerIteration(res.data.tokens_per_iteration)
      } catch (err) {
        console.error('Failed to fetch pricing:', err)
        // Fallback — don't block the page
      }
    }
    fetchPricing()
  }, [])

  const [formData, setFormData] = useState({
    name: '',
    strategy_type: '',
    timeframe: '1d',
    target_return: 15,
    max_iterations: 5,
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

    // Check token balance against per-iteration cost
    const minRequired = tokensPerIteration || 10  // fallback default
    if (!user || user.balance < minRequired) {
      setError(
        `Insufficient tokens. You need at least ${minRequired} tokens per iteration. Your balance: ${user?.balance || 0}`
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
          timeframe: formData.timeframe,
          symbols,
          target_return: formData.target_return,
          llm_enabled: formData.llm_enabled,
        },
      })

      const strategyId = strategyRes.data.uuid

      // Step 2: Trigger build
      const buildRes = await api.post('/api/builds/trigger', null, {
        params: {
          strategy_id: strategyId,
          description: formData.comments,
          target_return: formData.target_return,
          timeframe: formData.timeframe,
          max_iterations: formData.max_iterations,
        },
      })

      // Step 3: Redirect to build page
      router.push(`/dashboard/build/${buildRes.data.uuid}`)
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

          {/* Timeframe */}
          <div>
            <label className="block text-sm font-medium text-text mb-2">
              Timeframe *
            </label>
            <select
              value={formData.timeframe}
              onChange={(e) => setFormData({ ...formData, timeframe: e.target.value })}
              className="w-full bg-background border border-border rounded px-3 py-2 text-text focus:outline-none focus:border-primary"
            >
              {TIMEFRAME_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
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

          {/* Max Iterations */}
          <div>
            <label className="block text-sm font-medium text-text mb-2">
              Max Iterations
            </label>
            <input
              type="number"
              value={formData.max_iterations}
              onChange={(e) => setFormData({ ...formData, max_iterations: parseInt(e.target.value) || 1 })}
              min="1"
              max="20"
              className="w-full bg-background border border-border rounded px-3 py-2 text-text focus:outline-none focus:border-primary"
            />
            <p className="text-xs text-text-secondary mt-1">
              Maximum number of AI design-train-refine cycles
            </p>
            <p className="text-xs text-text-secondary mt-1">
              Estimated build time: ~{formData.max_iterations * 3}–{formData.max_iterations * 5} minutes
            </p>
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
          <p className="text-sm text-text-secondary mb-2">Token Cost</p>
          {tokensPerIteration !== null ? (
            <>
              <p className="text-2xl font-bold text-primary">{tokensPerIteration} tokens / iteration</p>
              <p className="text-xs text-text-secondary mt-1">
                Tokens are charged per iteration of the AI build process
              </p>
            </>
          ) : (
            <p className="text-sm text-text-secondary">Loading pricing...</p>
          )}
          <p className="text-xs text-text-secondary mt-2">
            Your balance: <span className="text-primary font-medium">{user?.balance || 0}</span> tokens
          </p>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading}
          className="w-full py-3 bg-gradient-to-r from-[#007cf0] to-[#00dfd8] disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold rounded-lg transition-all hover:shadow-lg flex items-center justify-center gap-2"
        >
          {loading && <Loader2 size={20} className="animate-spin" />}
          {loading ? 'Creating Strategy...' : 'Create & Build Strategy'}
        </button>
      </form>
    </div>
  )
}

