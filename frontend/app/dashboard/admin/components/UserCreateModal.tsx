'use client'

import { useState } from 'react'
import { adminAPI } from '@/lib/api'
import { X } from 'lucide-react'

interface UserCreateModalProps {
  onClose: () => void
  onSuccess: () => void
}

export default function UserCreateModal({ onClose, onSuccess }: UserCreateModalProps) {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    phone_number: '',
    user_role: 'user',
    status: 'active',
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [passwordStrength, setPasswordStrength] = useState(false)

  const validatePassword = (password: string) => {
    const hasUpperCase = /[A-Z]/.test(password)
    const hasLowerCase = /[a-z]/.test(password)
    const hasNumber = /[0-9]/.test(password)
    const isLongEnough = password.length >= 8
    return hasUpperCase && hasLowerCase && hasNumber && isLongEnough
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
    
    if (name === 'password') {
      setPasswordStrength(validatePassword(value))
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    // Validate required fields
    if (!formData.name || !formData.email || !formData.password) {
      setError('Please fill in all required fields')
      return
    }

    // Validate password strength
    if (!validatePassword(formData.password)) {
      setError('Password must be at least 8 characters with uppercase, lowercase, and number')
      return
    }

    try {
      setLoading(true)
      setError(null)
      await adminAPI.createUser(formData)
      onSuccess()
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to create user')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-surface border border-border rounded-lg max-w-md w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border sticky top-0 bg-surface">
          <h2 className="text-xl font-bold text-text">Create User</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-surface-hover rounded text-text-secondary hover:text-text"
          >
            <X size={20} />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {error && <div className="text-red-400 text-sm">{error}</div>}

          <div>
            <label className="block text-sm font-medium text-text mb-2">Name *</label>
            <input
              type="text"
              name="name"
              value={formData.name}
              onChange={handleChange}
              required
              className="w-full px-3 py-2 bg-surface-hover border border-border rounded-lg text-text focus:outline-none focus:ring-2 focus:ring-primary"
              placeholder="John Doe"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text mb-2">Email *</label>
            <input
              type="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
              className="w-full px-3 py-2 bg-surface-hover border border-border rounded-lg text-text focus:outline-none focus:ring-2 focus:ring-primary"
              placeholder="user@example.com"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text mb-2">Password *</label>
            <input
              type="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              required
              className="w-full px-3 py-2 bg-surface-hover border border-border rounded-lg text-text focus:outline-none focus:ring-2 focus:ring-primary"
              placeholder="••••••••"
            />
            {formData.password && (
              <p className={`text-xs mt-2 ${passwordStrength ? 'text-green-400' : 'text-red-400'}`}>
                {passwordStrength ? '✓ Strong password' : 'Min 8 chars, uppercase, lowercase, number'}
              </p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-text mb-2">Phone (Optional)</label>
            <input
              type="tel"
              name="phone_number"
              value={formData.phone_number}
              onChange={handleChange}
              className="w-full px-3 py-2 bg-surface-hover border border-border rounded-lg text-text focus:outline-none focus:ring-2 focus:ring-primary"
              placeholder="+1 (555) 000-0000"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text mb-2">Role</label>
            <select
              name="user_role"
              value={formData.user_role}
              onChange={handleChange}
              className="w-full px-3 py-2 bg-surface-hover border border-border rounded-lg text-text focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="user">User</option>
              <option value="admin">Admin</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-text mb-2">Status</label>
            <select
              name="status"
              value={formData.status}
              onChange={handleChange}
              className="w-full px-3 py-2 bg-surface-hover border border-border rounded-lg text-text focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="active">Active</option>
              <option value="inactive">Inactive</option>
            </select>
          </div>

          {/* Buttons */}
          <div className="flex gap-2 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 bg-surface border border-border rounded-lg text-text hover:bg-surface-hover"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="flex-1 px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-hover disabled:opacity-50"
            >
              {loading ? 'Creating...' : 'Create User'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

