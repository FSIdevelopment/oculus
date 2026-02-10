'use client'

import React, { createContext, useContext, useEffect, useState } from 'react'
import { authAPI } from '@/lib/api'
import Cookies from 'js-cookie'

export interface User {
  uuid: string
  name: string
  email: string
  phone_number?: string
  user_role: 'user' | 'admin'
  balance: number
}

interface AuthContextType {
  user: User | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
  logout: () => void
  register: (
    name: string,
    email: string,
    password: string,
    phone?: string
  ) => Promise<void>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // Check for existing token on mount
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const token = Cookies.get('auth_token')
        if (token) {
          const userData = await authAPI.getMe()
          setUser(userData)
        }
      } catch (error) {
        console.error('Auth check failed:', error)
        Cookies.remove('auth_token')
      } finally {
        setIsLoading(false)
      }
    }

    checkAuth()
  }, [])

  const login = async (email: string, password: string) => {
    try {
      const response = await authAPI.login(email, password)
      if (response.refresh_token) {
        Cookies.set('refresh_token', response.refresh_token, {
          expires: 7,
          secure: process.env.NODE_ENV === 'production',
          sameSite: 'strict',
        })
      }
      const userData = await authAPI.getMe()
      setUser(userData)
    } catch (error) {
      throw error
    }
  }

  const register = async (
    name: string,
    email: string,
    password: string,
    phone?: string
  ) => {
    try {
      const response = await authAPI.register(name, email, password, phone)
      if (response.refresh_token) {
        Cookies.set('refresh_token', response.refresh_token, {
          expires: 7,
          secure: process.env.NODE_ENV === 'production',
          sameSite: 'strict',
        })
      }
      const userData = await authAPI.getMe()
      setUser(userData)
    } catch (error) {
      throw error
    }
  }

  const logout = () => {
    authAPI.logout()
    setUser(null)
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        logout,
        register,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

