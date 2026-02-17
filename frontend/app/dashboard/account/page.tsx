'use client'

import { useState } from 'react'
import ProfileTab from './components/ProfileTab'
import TokensTab from './components/TokensTab'
import LicensesTab from './components/LicensesTab'
import EarningsTab from './components/EarningsTab'

export default function AccountPage() {
  const [activeTab, setActiveTab] = useState('profile')

  const tabs = [
    { id: 'profile', label: 'Profile', icon: '👤' },
    { id: 'tokens', label: 'Tokens & Billing', icon: '💳' },
    { id: 'licenses', label: 'Licenses', icon: '📜' },
    { id: 'earnings', label: 'Earnings', icon: '💰' },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold text-text mb-2">Account Settings</h1>
        <p className="text-text-secondary">Manage your profile, tokens, and licenses</p>
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
        {activeTab === 'profile' && <ProfileTab />}
        {activeTab === 'tokens' && <TokensTab />}
        {activeTab === 'licenses' && <LicensesTab />}
        {activeTab === 'earnings' && <EarningsTab />}
      </div>
    </div>
  )
}

