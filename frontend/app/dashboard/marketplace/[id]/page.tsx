'use client'

import { useEffect, useState } from 'react'
import { useParams } from 'next/navigation'
import Link from 'next/link'
import { marketplaceAPI, ratingsAPI } from '@/lib/api'
import { useAuth } from '@/contexts/AuthContext'
import { ArrowLeft, Loader2 } from 'lucide-react'

interface Strategy {
  uuid: string
  name: string
  description?: string
  strategy_type?: string
  symbols?: Record<string, any>
  target_return?: number
  backtest_results?: Record<string, any>
  subscriber_count: number
  rating?: number
  created_at: string
  user_id: string
}

interface Rating {
  uuid: string
  user_id: string
  rating: number
  review_text?: string
  created_at: string
}

export default function MarketplaceStrategyPage() {
  const params = useParams()
  const { user } = useAuth()
  const strategyId = params.id as string

  const [strategy, setStrategy] = useState<Strategy | null>(null)
  const [ratings, setRatings] = useState<Rating[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [subscribing, setSubscribing] = useState(false)
  const [ratingScore, setRatingScore] = useState(0)
  const [ratingReview, setRatingReview] = useState('')
  const [submittingRating, setSubmittingRating] = useState(false)

  useEffect(() => {
    loadStrategy()
  }, [strategyId])

  const loadStrategy = async () => {
    setLoading(true)
    setError('')
    try {
      const data = await marketplaceAPI.getMarketplaceStrategy(strategyId)
      setStrategy(data)

      // Load ratings
      const ratingsData = await ratingsAPI.getStrategyRatings(strategyId)
      setRatings(ratingsData.ratings || [])
    } catch (err: any) {
      const message = err.response?.data?.detail || err.message || 'Failed to load strategy'
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  const handleSubscribe = async () => {
    setSubscribing(true)
    try {
      await marketplaceAPI.subscribeToStrategy(strategyId)
      alert('Successfully subscribed to strategy!')
      loadStrategy()
    } catch (err: any) {
      const message = err.response?.data?.detail || err.message || 'Failed to subscribe'
      alert(message)
    } finally {
      setSubscribing(false)
    }
  }

  const handleSubmitRating = async () => {
    if (!user) {
      alert('Please log in to rate this strategy')
      return
    }
    if (ratingScore === 0) {
      alert('Please select a rating')
      return
    }

    setSubmittingRating(true)
    try {
      await ratingsAPI.createRating(strategyId, ratingScore, ratingReview)
      setRatingScore(0)
      setRatingReview('')
      loadStrategy()
    } catch (err: any) {
      const message = err.response?.data?.detail || err.message || 'Failed to submit rating'
      alert(message)
    } finally {
      setSubmittingRating(false)
    }
  }

  const renderStars = (rating?: number) => {
    if (!rating) return <span className="text-text-secondary">No ratings</span>
    const stars = Math.round(rating)
    return (
      <div className="flex items-center gap-1">
        {[...Array(5)].map((_, i) => (
          <span key={i} className={`text-2xl ${i < stars ? 'text-yellow-400' : 'text-gray-600'}`}>
            ★
          </span>
        ))}
        <span className="text-text-secondary ml-2">({rating.toFixed(1)})</span>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-text-secondary">Loading strategy...</div>
      </div>
    )
  }

  if (error || !strategy) {
    return (
      <div className="space-y-4">
        <Link href="/dashboard/marketplace" className="flex items-center gap-2 text-primary hover:text-primary-hover">
          <ArrowLeft size={20} />
          Back to Marketplace
        </Link>
        <div className="bg-red-900/20 border border-red-700 rounded-lg p-4 text-red-400">
          {error || 'Strategy not found'}
        </div>
      </div>
    )
  }

  const backtestResults = strategy.backtest_results || {}
  const symbols = strategy.symbols?.symbols || []

  return (
    <div className="space-y-6">
      {/* Back Button */}
      <Link href="/dashboard/marketplace" className="flex items-center gap-2 text-primary hover:text-primary-hover">
        <ArrowLeft size={20} />
        Back to Marketplace
      </Link>

      {/* Header */}
      <div className="bg-surface border border-border rounded-lg p-6">
        <div className="flex justify-between items-start mb-4">
          <div>
            <h1 className="text-4xl font-bold text-text mb-2">{strategy.name}</h1>
            {strategy.strategy_type && (
              <span className="inline-block bg-primary/20 text-primary text-sm px-3 py-1 rounded mb-4">
                {strategy.strategy_type}
              </span>
            )}
          </div>
          <button
            onClick={handleSubscribe}
            disabled={subscribing}
            className="bg-primary hover:bg-primary-hover text-white font-medium px-6 py-2 rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
          >
            {subscribing && <Loader2 size={18} className="animate-spin" />}
            Subscribe - $29/mo
          </button>
        </div>

        <p className="text-text-secondary mb-4">{strategy.description || 'No description'}</p>

        {/* Key Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-text-secondary text-sm mb-1">Subscribers</p>
            <p className="text-2xl font-bold text-text">{strategy.subscriber_count}</p>
          </div>
          <div>
            <p className="text-text-secondary text-sm mb-1">Rating</p>
            {renderStars(strategy.rating)}
          </div>
          <div>
            <p className="text-text-secondary text-sm mb-1">Target Return</p>
            <p className="text-2xl font-bold text-text">{strategy.target_return?.toFixed(1) || 'N/A'}%</p>
          </div>
          <div>
            <p className="text-text-secondary text-sm mb-1">Symbols</p>
            <p className="text-text">{symbols.length} assets</p>
          </div>
        </div>
      </div>

      {/* Backtest Results */}
      {Object.keys(backtestResults).length > 0 && (
        <div className="bg-surface border border-border rounded-lg p-6">
          <h2 className="text-2xl font-bold text-text mb-4">Backtest Results</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {backtestResults.total_return !== undefined && (
              <div className="bg-background rounded p-4">
                <p className="text-text-secondary text-sm mb-1">Total Return</p>
                <p className="text-2xl font-bold text-green-400">{backtestResults.total_return.toFixed(2)}%</p>
              </div>
            )}
            {backtestResults.win_rate !== undefined && (
              <div className="bg-background rounded p-4">
                <p className="text-text-secondary text-sm mb-1">Win Rate</p>
                <p className="text-2xl font-bold text-text">{backtestResults.win_rate.toFixed(1)}%</p>
              </div>
            )}
            {backtestResults.sharpe_ratio !== undefined && (
              <div className="bg-background rounded p-4">
                <p className="text-text-secondary text-sm mb-1">Sharpe Ratio</p>
                <p className="text-2xl font-bold text-text">{backtestResults.sharpe_ratio.toFixed(2)}</p>
              </div>
            )}
            {backtestResults.max_drawdown !== undefined && (
              <div className="bg-background rounded p-4">
                <p className="text-text-secondary text-sm mb-1">Max Drawdown</p>
                <p className="text-2xl font-bold text-red-400">{backtestResults.max_drawdown.toFixed(2)}%</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Ratings Section */}
      <div className="bg-surface border border-border rounded-lg p-6">
        <h2 className="text-2xl font-bold text-text mb-4">Ratings & Reviews</h2>

        {/* Rating Form */}
        {user && (
          <div className="mb-6 p-4 bg-background rounded-lg">
            <p className="text-text mb-3">Rate this strategy:</p>
            <div className="flex gap-2 mb-3">
              {[1, 2, 3, 4, 5].map(score => (
                <button
                  key={score}
                  onClick={() => setRatingScore(score)}
                  className={`text-3xl transition-colors ${
                    score <= ratingScore ? 'text-yellow-400' : 'text-gray-600 hover:text-yellow-300'
                  }`}
                >
                  ★
                </button>
              ))}
            </div>
            <textarea
              value={ratingReview}
              onChange={(e) => setRatingReview(e.target.value)}
              placeholder="Write a review (optional)..."
              className="w-full p-2 bg-surface border border-border rounded text-text placeholder-text-secondary focus:outline-none focus:border-primary mb-3"
              rows={3}
            />
            <button
              onClick={handleSubmitRating}
              disabled={submittingRating || ratingScore === 0}
              className="bg-primary hover:bg-primary-hover text-white font-medium px-4 py-2 rounded transition-colors disabled:opacity-50 flex items-center gap-2"
            >
              {submittingRating && <Loader2 size={16} className="animate-spin" />}
              Submit Rating
            </button>
          </div>
        )}

        {/* Ratings List */}
        <div className="space-y-4">
          {ratings.length === 0 ? (
            <p className="text-text-secondary">No ratings yet</p>
          ) : (
            ratings.map(rating => (
              <div key={rating.uuid} className="p-4 bg-background rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  {[...Array(5)].map((_, i) => (
                    <span key={i} className={i < rating.rating ? 'text-yellow-400' : 'text-gray-600'}>
                      ★
                    </span>
                  ))}
                </div>
                {rating.review_text && (
                  <p className="text-text text-sm">{rating.review_text}</p>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

