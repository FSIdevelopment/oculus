'use client'

import { useState } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import api from '@/lib/api'
import { AlertCircle, CheckCircle } from 'lucide-react'

export default function ProfileTab() {
  const { user } = useAuth()
  const [formData, setFormData] = useState({
    name: user?.name || '',
    email: user?.email || '',
    phone_number: user?.phone_number || '',
  })
  const [passwordData, setPasswordData] = useState({
    current_password: '',
    new_password: '',
    confirm_password: '',
  })
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  const handleProfileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
  }

  const handlePasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setPasswordData((prev) => ({ ...prev, [name]: value }))
  }

  const handleProfileSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setMessage(null)

    try {
      await api.put('/api/users/me', formData)
      setMessage({ type: 'success', text: 'Profile updated successfully' })
    } catch (error: any) {
      setMessage({ type: 'error', text: error.response?.data?.detail || 'Failed to update profile' })
    } finally {
      setLoading(false)
    }
  }

  const handlePasswordSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setMessage(null)

    if (passwordData.new_password !== passwordData.confirm_password) {
      setMessage({ type: 'error', text: 'Passwords do not match' })
      setLoading(false)
      return
    }

    try {
      await api.put('/api/users/me', { password: passwordData.new_password })
      setMessage({ type: 'success', text: 'Password changed successfully' })
      setPasswordData({ current_password: '', new_password: '', confirm_password: '' })
    } catch (error: any) {
      setMessage({ type: 'error', text: error.response?.data?.detail || 'Failed to change password' })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-8">
      {/* Messages */}
      {message && (
        <div
          className={`flex items-center gap-3 p-4 rounded-lg ${
            message.type === 'success'
              ? 'bg-green-900/20 border border-green-700 text-green-400'
              : 'bg-red-900/20 border border-red-700 text-red-400'
          }`}
        >
          {message.type === 'success' ? <CheckCircle size={20} /> : <AlertCircle size={20} />}
          {message.text}
        </div>
      )}

      {/* Profile Form */}
      <form onSubmit={handleProfileSubmit} className="space-y-4 bg-surface border border-border rounded-lg p-6">
        <h2 className="text-xl font-bold text-text mb-4">Profile Information</h2>

        <div>
          <label className="block text-sm font-medium text-text-secondary mb-2">Name</label>
          <input
            type="text"
            name="name"
            value={formData.name}
            onChange={handleProfileChange}
            className="w-full px-4 py-2 bg-background border border-border rounded-lg text-text focus:outline-none focus:border-primary"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-text-secondary mb-2">Email</label>
          <input
            type="email"
            name="email"
            value={formData.email}
            onChange={handleProfileChange}
            className="w-full px-4 py-2 bg-background border border-border rounded-lg text-text focus:outline-none focus:border-primary"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-text-secondary mb-2">Phone Number</label>
          <input
            type="tel"
            name="phone_number"
            value={formData.phone_number}
            onChange={handleProfileChange}
            className="w-full px-4 py-2 bg-background border border-border rounded-lg text-text focus:outline-none focus:border-primary"
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full px-4 py-2 bg-primary hover:bg-primary-hover text-white font-medium rounded-lg transition-colors disabled:opacity-50"
        >
          {loading ? 'Saving...' : 'Save Changes'}
        </button>
      </form>

      {/* Password Form */}
      <form onSubmit={handlePasswordSubmit} className="space-y-4 bg-surface border border-border rounded-lg p-6">
        <h2 className="text-xl font-bold text-text mb-4">Change Password</h2>

        <div>
          <label className="block text-sm font-medium text-text-secondary mb-2">New Password</label>
          <input
            type="password"
            name="new_password"
            value={passwordData.new_password}
            onChange={handlePasswordChange}
            placeholder="Must contain uppercase, lowercase, and number"
            className="w-full px-4 py-2 bg-background border border-border rounded-lg text-text focus:outline-none focus:border-primary"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-text-secondary mb-2">Confirm Password</label>
          <input
            type="password"
            name="confirm_password"
            value={passwordData.confirm_password}
            onChange={handlePasswordChange}
            className="w-full px-4 py-2 bg-background border border-border rounded-lg text-text focus:outline-none focus:border-primary"
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full px-4 py-2 bg-primary hover:bg-primary-hover text-white font-medium rounded-lg transition-colors disabled:opacity-50"
        >
          {loading ? 'Updating...' : 'Change Password'}
        </button>
      </form>
    </div>
  )
}

