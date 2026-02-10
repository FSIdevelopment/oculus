'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { useAuth } from '@/contexts/AuthContext'

export default function RegisterPage() {
  const router = useRouter()
  const { register, isAuthenticated } = useAuth()
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    phone: '',
  })
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  // Redirect if already authenticated
  if (isAuthenticated) {
    router.push('/')
    return null
  }

  const validatePassword = (password: string) => {
    const hasUpperCase = /[A-Z]/.test(password)
    const hasLowerCase = /[a-z]/.test(password)
    const hasNumber = /[0-9]/.test(password)
    const isLongEnough = password.length >= 8

    return hasUpperCase && hasLowerCase && hasNumber && isLongEnough
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    // Validation
    if (!formData.name || !formData.email || !formData.password) {
      setError('Please fill in all required fields')
      return
    }

    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match')
      return
    }

    if (!validatePassword(formData.password)) {
      setError('Password must be at least 8 characters with uppercase, lowercase, and number')
      return
    }

    setIsLoading(true)

    try {
      await register(formData.name, formData.email, formData.password, formData.phone)
      router.push('/')
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Registration failed')
    } finally {
      setIsLoading(false)
    }
  }

  const passwordStrength = validatePassword(formData.password)

  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        <div className="bg-surface border border-border rounded-lg p-8 shadow-lg">
          <h1 className="text-3xl font-bold text-text mb-2">Create Account</h1>
          <p className="text-text-secondary mb-8">Join Oculus to start building strategies</p>

          {error && (
            <div className="bg-red-900/20 border border-red-700 text-red-400 px-4 py-3 rounded-lg mb-6">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="name" className="block text-sm font-medium text-text mb-2">
                Full Name
              </label>
              <input
                id="name"
                name="name"
                type="text"
                value={formData.name}
                onChange={handleChange}
                required
                className="w-full px-4 py-2 bg-background border border-border rounded-lg text-text placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-primary"
                placeholder="John Doe"
              />
            </div>

            <div>
              <label htmlFor="email" className="block text-sm font-medium text-text mb-2">
                Email
              </label>
              <input
                id="email"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                required
                className="w-full px-4 py-2 bg-background border border-border rounded-lg text-text placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-primary"
                placeholder="you@example.com"
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-text mb-2">
                Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                value={formData.password}
                onChange={handleChange}
                required
                className="w-full px-4 py-2 bg-background border border-border rounded-lg text-text placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-primary"
                placeholder="••••••••"
              />
              {formData.password && (
                <p className={`text-xs mt-2 ${passwordStrength ? 'text-green-400' : 'text-red-400'}`}>
                  {passwordStrength
                    ? '✓ Strong password'
                    : 'Min 8 chars, uppercase, lowercase, number'}
                </p>
              )}
            </div>

            <div>
              <label htmlFor="confirmPassword" className="block text-sm font-medium text-text mb-2">
                Confirm Password
              </label>
              <input
                id="confirmPassword"
                name="confirmPassword"
                type="password"
                value={formData.confirmPassword}
                onChange={handleChange}
                required
                className="w-full px-4 py-2 bg-background border border-border rounded-lg text-text placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-primary"
                placeholder="••••••••"
              />
            </div>

            <div>
              <label htmlFor="phone" className="block text-sm font-medium text-text mb-2">
                Phone (Optional)
              </label>
              <input
                id="phone"
                name="phone"
                type="tel"
                value={formData.phone}
                onChange={handleChange}
                className="w-full px-4 py-2 bg-background border border-border rounded-lg text-text placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-primary"
                placeholder="+1 (555) 000-0000"
              />
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-primary hover:bg-primary-hover text-white font-medium py-2 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Creating account...' : 'Create Account'}
            </button>
          </form>

          <p className="text-center text-text-secondary mt-6">
            Already have an account?{' '}
            <Link href="/login" className="text-primary hover:text-primary-hover font-medium">
              Sign In
            </Link>
          </p>
        </div>
      </div>
    </div>
  )
}

