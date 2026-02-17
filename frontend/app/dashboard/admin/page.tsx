'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/contexts/AuthContext'
import { Settings } from 'lucide-react'
import OverviewTab from './components/OverviewTab'
import UserManagementTab from './components/UserManagementTab'
import StrategyManagementTab from './components/StrategyManagementTab'
import ProductManagementTab from './components/ProductManagementTab'
import RatingsManagementTab from './components/RatingsManagementTab'

export default function AdminPage() {
  const router = useRouter()
  const { user, isLoading } = useAuth()
  const [activeTab, setActiveTab] = useState('overview')

  // Check if user is admin
  useEffect(() => {
    if (!isLoading && user?.user_role !== 'admin') {
      router.push('/dashboard')
    }
  }, [user, isLoading, router])

  if (isLoading || user?.user_role !== 'admin') {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-text-secondary">Loading...</div>
      </div>
    )
  }

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'users', label: 'Users', icon: '👥' },
    { id: 'strategies', label: 'Strategies', icon: '📈' },
    { id: 'products', label: 'Products', icon: '📦' },
    { id: 'ratings', label: 'Ratings', icon: '⭐' },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Settings size={32} className="text-primary" />
        <div>
          <h1 className="text-4xl font-bold text-text">Admin Dashboard</h1>
          <p className="text-text-secondary">Manage platform users, strategies, products, and ratings</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-border">
        <div className="flex gap-8 overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-3 font-medium whitespace-nowrap border-b-2 transition-colors ${
                activeTab === tab.id
                  ? 'border-primary text-primary'
                  : 'border-transparent text-text-secondary hover:text-text'
              }`}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div className="mt-6">
        {activeTab === 'overview' && <OverviewTab />}
        {activeTab === 'users' && <UserManagementTab />}
        {activeTab === 'strategies' && <StrategyManagementTab />}
        {activeTab === 'products' && <ProductManagementTab />}
        {activeTab === 'ratings' && <RatingsManagementTab />}
      </div>
    </div>
  )
}

