'use client'

import { useState, useEffect } from 'react'
import { adminAPI } from '@/lib/api'
import { Trash2, Star } from 'lucide-react'

interface Rating {
  uuid: string
  strategy_id: string
  strategy_name?: string
  user_id: string
  user_name?: string
  score: number
  review?: string
  created_at: string
}

export default function RatingsManagementTab() {
  const [ratings, setRatings] = useState<Rating[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [skip, setSkip] = useState(0)
  const [total, setTotal] = useState(0)

  const limit = 20

  useEffect(() => {
    fetchRatings()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [skip])

  const fetchRatings = async () => {
    try {
      setLoading(true)
      const data = await adminAPI.listRatings({
        page: Math.floor(skip / limit) + 1,
        page_size: limit,
      })
      setRatings(data.data || [])
      setTotal(data.total || 0)
    } catch (err) {
      setError('Failed to load ratings')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (ratingId: string) => {
    if (!confirm('Are you sure you want to delete this rating?')) return
    try {
      await adminAPI.deleteRating(ratingId)
      setRatings(ratings.filter((r) => r.uuid !== ratingId))
    } catch (err) {
      alert('Failed to delete rating')
    }
  }

  const renderStars = (score: number) => {
    return (
      <div className="flex gap-1">
        {[1, 2, 3, 4, 5].map((i) => (
          <Star
            key={i}
            size={16}
            className={i <= score ? 'fill-yellow-400 text-yellow-400' : 'text-text-secondary'}
          />
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Ratings List */}
      {loading ? (
        <div className="text-text-secondary">Loading ratings...</div>
      ) : error ? (
        <div className="text-red-400">{error}</div>
      ) : (
        <>
          <div className="space-y-4">
            {ratings.map((rating) => (
              <div
                key={rating.uuid}
                className="bg-surface border border-border rounded-lg p-6 hover:bg-surface-hover transition-colors"
              >
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h3 className="text-lg font-semibold text-text">{rating.strategy_name}</h3>
                    <p className="text-text-secondary text-sm">by {rating.user_name}</p>
                  </div>
                  <button
                    onClick={() => handleDelete(rating.uuid)}
                    className="p-2 hover:bg-red-900/20 rounded text-text-secondary hover:text-red-400"
                    title="Delete"
                  >
                    <Trash2 size={18} />
                  </button>
                </div>

                <div className="mb-3">{renderStars(rating.score)}</div>

                {rating.review && (
                  <p className="text-text-secondary text-sm mb-3 italic">&quot;{rating.review}&quot;</p>
                )}

                <p className="text-text-secondary text-xs">
                  {new Date(rating.created_at).toLocaleDateString()}
                </p>
              </div>
            ))}
          </div>

          {/* Pagination */}
          <div className="flex justify-between items-center">
            <p className="text-text-secondary text-sm">
              Showing {skip + 1} to {Math.min(skip + limit, total)} of {total}
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => setSkip(Math.max(0, skip - limit))}
                disabled={skip === 0}
                className="px-4 py-2 bg-surface border border-border rounded-lg text-text disabled:opacity-50 hover:bg-surface-hover"
              >
                Previous
              </button>
              <button
                onClick={() => setSkip(skip + limit)}
                disabled={skip + limit >= total}
                className="px-4 py-2 bg-surface border border-border rounded-lg text-text disabled:opacity-50 hover:bg-surface-hover"
              >
                Next
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

