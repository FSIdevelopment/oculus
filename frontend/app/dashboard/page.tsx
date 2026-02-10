'use client'

import Link from 'next/link'
import { useAuth } from '@/contexts/AuthContext'
import { Zap, BarChart3, ShoppingBag } from 'lucide-react'

export default function DashboardPage() {
  const { user } = useAuth()

  const quickActions = [
    {
      title: 'Create Strategy',
      description: 'Build a new trading strategy',
      icon: Zap,
      href: '/dashboard/create',
      color: 'from-blue-600 to-blue-400',
    },
    {
      title: 'View Strategies',
      description: 'Manage your existing strategies',
      icon: BarChart3,
      href: '/dashboard/strategies',
      color: 'from-purple-600 to-purple-400',
    },
    {
      title: 'Buy Tokens',
      description: 'Purchase more tokens',
      icon: ShoppingBag,
      href: '/dashboard/marketplace',
      color: 'from-green-600 to-green-400',
    },
  ]

  return (
    <div className="space-y-8">
      {/* Welcome Section */}
      <div>
        <h1 className="text-4xl font-bold text-text mb-2">
          Welcome to Oculus, {user?.name}
        </h1>
        <p className="text-text-secondary text-lg">
          Your AI-powered strategy design and optimization platform
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-surface border border-border rounded-lg p-6">
          <p className="text-text-secondary text-sm mb-2">Token Balance</p>
          <p className="text-3xl font-bold text-primary">
            🪙 {user?.token_balance || 0}
          </p>
        </div>
        <div className="bg-surface border border-border rounded-lg p-6">
          <p className="text-text-secondary text-sm mb-2">Account Status</p>
          <p className="text-3xl font-bold text-text capitalize">
            {user?.user_role || 'User'}
          </p>
        </div>
        <div className="bg-surface border border-border rounded-lg p-6">
          <p className="text-text-secondary text-sm mb-2">Email</p>
          <p className="text-lg font-medium text-text truncate">{user?.email}</p>
        </div>
      </div>

      {/* Quick Actions */}
      <div>
        <h2 className="text-2xl font-bold text-text mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {quickActions.map((action) => {
            const Icon = action.icon
            return (
              <Link
                key={action.href}
                href={action.href}
                className="group bg-surface border border-border rounded-lg p-6 hover:border-primary hover:bg-surface-hover transition-all duration-300"
              >
                <div className={`inline-block p-3 rounded-lg bg-gradient-to-br ${action.color} mb-4`}>
                  <Icon size={24} className="text-white" />
                </div>
                <h3 className="text-lg font-semibold text-text mb-2 group-hover:text-primary transition-colors">
                  {action.title}
                </h3>
                <p className="text-text-secondary text-sm">{action.description}</p>
              </Link>
            )
          })}
        </div>
      </div>

      {/* Getting Started */}
      <div className="bg-surface border border-border rounded-lg p-6">
        <h2 className="text-xl font-bold text-text mb-4">Getting Started</h2>
        <ul className="space-y-3 text-text-secondary">
          <li className="flex items-start gap-3">
            <span className="text-primary font-bold mt-1">1.</span>
            <span>Create your first strategy using the Strategy Creation tool</span>
          </li>
          <li className="flex items-start gap-3">
            <span className="text-primary font-bold mt-1">2.</span>
            <span>Configure parameters and test your strategy</span>
          </li>
          <li className="flex items-start gap-3">
            <span className="text-primary font-bold mt-1">3.</span>
            <span>Deploy and monitor your strategy in real-time</span>
          </li>
          <li className="flex items-start gap-3">
            <span className="text-primary font-bold mt-1">4.</span>
            <span>Optimize based on performance metrics</span>
          </li>
        </ul>
      </div>
    </div>
  )
}

