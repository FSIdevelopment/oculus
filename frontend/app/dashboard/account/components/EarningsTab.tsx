'use client'

import { useEffect, useState } from 'react'



export default function EarningsTab() {
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // TODO: Fetch earnings data from API
    setLoading(false)
  }, [])

  if (loading) {
    return <div className="text-text-secondary">Loading earnings...</div>
  }

  return (
    <div className="space-y-6">
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

