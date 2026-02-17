import axios, { AxiosInstance } from 'axios'
import Cookies from 'js-cookie'

const API_URL = process.env.NEXT_PUBLIC_API_URL || ''

const api: AxiosInstance = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor to attach JWT token
api.interceptors.request.use(
  (config) => {
    const token = Cookies.get('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for 401 errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Clear token and redirect to login
      Cookies.remove('auth_token')
      if (typeof window !== 'undefined') {
        window.location.href = '/login'
      }
    }
    return Promise.reject(error)
  }
)

// Auth API functions
export const authAPI = {
  login: async (email: string, password: string) => {
    const response = await api.post('/api/auth/login', { email, password })
    if (response.data.access_token) {
      Cookies.set('auth_token', response.data.access_token, {
        expires: 7,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'strict',
      })
    }
    return response.data
  },

  register: async (
    name: string,
    email: string,
    password: string,
    phone?: string
  ) => {
    const response = await api.post('/api/auth/register', {
      name,
      email,
      password,
      phone_number: phone,
    })
    if (response.data.access_token) {
      Cookies.set('auth_token', response.data.access_token, {
        expires: 7,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'strict',
      })
    }
    return response.data
  },

  getMe: async () => {
    const response = await api.get('/api/auth/me')
    return response.data
  },

  refreshToken: async () => {
    const response = await api.post('/api/auth/refresh', {
      refresh_token: Cookies.get('refresh_token'),
    })
    if (response.data.access_token) {
      Cookies.set('auth_token', response.data.access_token, {
        expires: 7,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'strict',
      })
    }
    return response.data
  },

  logout: () => {
    Cookies.remove('auth_token')
  },
}

// Strategy API functions
export const strategyAPI = {
  listStrategies: async (skip: number = 0, limit: number = 10, has_completed_build?: boolean) => {
    const params: any = { skip, limit }
    if (has_completed_build !== undefined) {
      params.has_completed_build = has_completed_build
    }
    const response = await api.get('/api/strategies', { params })
    return response.data
  },

  getStrategy: async (strategyId: string) => {
    const response = await api.get(`/api/strategies/${strategyId}`)
    return response.data
  },

  getDockerInfo: async (strategyId: string) => {
    const response = await api.get(`/api/strategies/${strategyId}/docker`)
    return response.data
  },

  getStrategyBuilds: async (strategyId: string, skip: number = 0, limit: number = 10) => {
    const response = await api.get(`/api/strategies/${strategyId}/builds`, {
      params: { skip, limit },
    })
    return response.data
  },

  getBuildIterations: async (buildId: string) => {
    const response = await api.get(`/api/builds/${buildId}/iterations`)
    return response.data
  },
}

// Build API functions
export const buildAPI = {
  listBuilds: async (skip: number = 0, limit: number = 10) => {
    const response = await api.get('/api/builds', { params: { skip, limit } })
    return response.data
  },
}

// License API functions
export const licenseAPI = {
  listLicenses: async (skip: number = 0, limit: number = 10) => {
    const response = await api.get('/api/users/me/licenses', {
      params: { skip, limit },
    })
    return response.data
  },

  purchaseLicense: async (strategyId: string, licenseType: 'monthly' | 'annual') => {
    const response = await api.post(`/api/strategies/${strategyId}/license`, {
      license_type: licenseType,
      strategy_id: strategyId,
    })
    return response.data
  },

  renewLicense: async (licenseId: string, licenseType: 'monthly' | 'annual') => {
    const response = await api.put(`/api/licenses/${licenseId}/renew`, {
      license_type: licenseType,
    })
    return response.data
  },
}

// Marketplace API functions
export const marketplaceAPI = {
  listMarketplace: async (params?: {
    skip?: number
    limit?: number
    strategy_type?: string
    sort_by?: string
    sort_order?: string
  }) => {
    const response = await api.get('/api/strategies/marketplace', { params })
    return response.data
  },

  getMarketplaceStrategy: async (strategyId: string) => {
    const response = await api.get(`/api/strategies/marketplace/${strategyId}`)
    return response.data
  },

  subscribeToStrategy: async (strategyId: string) => {
    const response = await api.post(`/api/marketplace/${strategyId}/subscribe`)
    return response.data
  },
}

// Ratings API functions
export const ratingsAPI = {
  getStrategyRatings: async (strategyId: string, page: number = 1, pageSize: number = 50) => {
    const response = await api.get(`/api/strategies/${strategyId}/ratings`, {
      params: { page, page_size: pageSize },
    })
    return response.data
  },

  createRating: async (strategyId: string, score: number, review?: string) => {
    const response = await api.post(`/api/strategies/${strategyId}/ratings`, {
      score,
      review,
    })
    return response.data
  },

  updateRating: async (ratingId: string, score?: number, review?: string) => {
    const response = await api.put(`/api/ratings/${ratingId}`, {
      score,
      review,
    })
    return response.data
  },
}

// Connect API functions
export const connectAPI = {
  connectOnboard: async () => {
    const response = await api.post('/api/connect/onboard')
    return response.data
  },

  connectStatus: async () => {
    const response = await api.get('/api/connect/status')
    return response.data
  },

  connectDashboardLink: async () => {
    const response = await api.get('/api/connect/dashboard-link')
    return response.data
  },
}

// Admin API functions
export const adminAPI = {
  listUsers: async (params?: any) => {
    const response = await api.get('/api/admin/users', { params })
    return response.data
  },

  createUser: async (data: any) => {
    const response = await api.post('/api/admin/users', data)
    return response.data
  },

  getUser: async (id: string) => {
    const response = await api.get(`/api/admin/users/${id}`)
    return response.data
  },

  updateUser: async (id: string, data: any) => {
    const response = await api.put(`/api/admin/users/${id}`, data)
    return response.data
  },

  deleteUser: async (id: string) => {
    const response = await api.delete(`/api/admin/users/${id}`)
    return response.data
  },

  resetPassword: async (id: string) => {
    const response = await api.post(`/api/admin/users/${id}/reset-password`)
    return response.data
  },

  impersonateUser: async (id: string) => {
    const response = await api.post(`/api/admin/users/${id}/impersonate`)
    return response.data
  },

  listStrategies: async (params?: any) => {
    const response = await api.get('/api/admin/strategies', { params })
    return response.data
  },

  updateStrategy: async (id: string, data: any) => {
    const response = await api.put(`/api/admin/strategies/${id}`, data)
    return response.data
  },

  deleteStrategy: async (id: string) => {
    const response = await api.delete(`/api/admin/strategies/${id}`)
    return response.data
  },

  listRatings: async (params?: any) => {
    const response = await api.get('/api/admin/ratings', { params })
    return response.data
  },

  deleteRating: async (id: string) => {
    const response = await api.delete(`/api/admin/ratings/${id}`)
    return response.data
  },

  createProduct: async (data: any) => {
    const response = await api.post('/api/admin/products', data)
    return response.data
  },

  updateProduct: async (id: string, data: any) => {
    const response = await api.put(`/api/admin/products/${id}`, data)
    return response.data
  },

  deleteProduct: async (id: string) => {
    const response = await api.delete(`/api/admin/products/${id}`)
    return response.data
  },

  listLicenses: async (params?: any) => {
    const response = await api.get('/api/admin/licenses', { params })
    return response.data
  },

  updateLicense: async (id: string, data: any) => {
    const response = await api.put(`/api/admin/licenses/${id}`, data)
    return response.data
  },

  listProducts: async () => {
    const response = await api.get('/api/products')
    return response.data
  },
}

export default api

