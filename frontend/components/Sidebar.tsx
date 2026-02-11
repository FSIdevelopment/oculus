'use client'

import { useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/contexts/AuthContext'
import {
  Menu,
  X,
  Zap,
  Hammer,
  BarChart3,
  ShoppingBag,
  User,
  FileText,
  LogOut,
  Settings,
} from 'lucide-react'

export default function Sidebar() {
  const router = useRouter()
  const { user, logout } = useAuth()
  const [isOpen, setIsOpen] = useState(false)

  const handleLogout = () => {
    logout()
    router.push('/login')
  }

  const menuItems = [
    { label: 'Strategy Creation', icon: Zap, href: '/dashboard/create' },
    { label: 'Build', icon: Hammer, href: '/dashboard/build' },
    { label: 'Strategies', icon: BarChart3, href: '/dashboard/strategies' },
    { label: 'Marketplace', icon: ShoppingBag, href: '/dashboard/marketplace' },
    { label: 'Account', icon: User, href: '/dashboard/account' },
    { label: 'Terms of Use', icon: FileText, href: '/terms' },
  ]

  const adminItems = user?.user_role === 'admin' ? [
    { label: 'Admin Dashboard', icon: Settings, href: '/dashboard/admin' },
  ] : []

  return (
    <>
      {/* Mobile menu button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed top-4 left-4 z-50 md:hidden bg-surface border border-border p-2 rounded-lg text-text hover:bg-surface-hover"
      >
        {isOpen ? <X size={24} /> : <Menu size={24} />}
      </button>

      {/* Sidebar */}
      <aside
        className={`fixed left-0 top-0 h-screen w-64 bg-surface border-r border-border flex flex-col transition-transform duration-300 z-40 ${
          isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        }`}
      >
        {/* Logo */}
        <div className="p-6 border-b border-border">
          <h1 className="text-2xl font-bold text-primary">OCULUS</h1>
        </div>

        {/* Token Balance */}
        <div className="px-6 py-4 border-b border-border">
          <p className="text-text-secondary text-sm mb-1">Token Balance</p>
          <p className="text-2xl font-bold text-primary">
            🪙 {user?.balance || 0}
          </p>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-4 py-6 space-y-2 overflow-y-auto">
          {menuItems.map((item) => {
            const Icon = item.icon
            return (
              <Link
                key={item.href}
                href={item.href}
                className="flex items-center gap-3 px-4 py-3 rounded-lg text-text-secondary hover:text-text hover:bg-surface-hover transition-colors"
                onClick={() => setIsOpen(false)}
              >
                <Icon size={20} />
                <span>{item.label}</span>
              </Link>
            )
          })}

          {/* Admin Section */}
          {adminItems.length > 0 && (
            <>
              <div className="my-4 border-t border-border" />
              <p className="px-4 py-2 text-xs font-semibold text-text-secondary uppercase">
                Admin
              </p>
              {adminItems.map((item) => {
                const Icon = item.icon
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="flex items-center gap-3 px-4 py-3 rounded-lg text-text-secondary hover:text-text hover:bg-surface-hover transition-colors"
                    onClick={() => setIsOpen(false)}
                  >
                    <Icon size={20} />
                    <span>{item.label}</span>
                  </Link>
                )
              })}
            </>
          )}
        </nav>

        {/* User Info & Logout */}
        <div className="border-t border-border p-4 space-y-3">
          <div className="px-2">
            <p className="text-sm font-medium text-text">{user?.name}</p>
            <p className="text-xs text-text-secondary">{user?.email}</p>
          </div>
          <button
            onClick={handleLogout}
            className="w-full flex items-center gap-2 px-4 py-2 rounded-lg bg-surface-hover hover:bg-red-900/20 text-text-secondary hover:text-red-400 transition-colors"
          >
            <LogOut size={18} />
            <span>Logout</span>
          </button>
        </div>
      </aside>

      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 md:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}
    </>
  )
}

