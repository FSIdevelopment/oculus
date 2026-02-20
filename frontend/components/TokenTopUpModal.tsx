'use client'

import { useState, useEffect } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import api from '@/lib/api'
import { ShoppingCart, X, Loader, Coins } from 'lucide-react'
import PaymentForm from '@/app/dashboard/account/components/PaymentForm'

interface Product {
  uuid: string
  name: string
  price: number
  token_amount: number
  product_type: string
  description?: string
}

interface TokenTopUpModalProps {
  isOpen: boolean
  onClose: () => void
}

export default function TokenTopUpModal({ isOpen, onClose }: TokenTopUpModalProps) {
  const { user, refreshUser } = useAuth()
  const [products, setProducts] = useState<Product[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null)

  useEffect(() => {
    if (isOpen) {
      fetchProducts()
    }
  }, [isOpen])

  const fetchProducts = async () => {
    try {
      setLoading(true)
      const response = await api.get('/api/products')
      const allProducts = Array.isArray(response.data) ? response.data : response.data.data || []
      const tokenProducts = allProducts.filter((p: any) => p.product_type === 'token_package')
      setProducts(tokenProducts)
    } catch (error) {
      console.error('Failed to fetch products:', error)
    } finally {
      setLoading(false)
    }
  }

  const handlePaymentSuccess = async () => {
    await refreshUser()
    setSelectedProduct(null)
    onClose()
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="bg-surface border border-border rounded-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <div className="flex items-center gap-2">
            <Coins size={20} className="text-primary" />
            <h2 className="text-lg font-semibold text-text">Top Up Tokens</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-surface-hover transition-colors text-text-secondary hover:text-text"
          >
            <X size={18} />
          </button>
        </div>

        {/* Balance Display */}
        <div className="px-6 py-4 bg-gradient-to-r from-blue-600/20 to-blue-400/10 border-b border-border">
          <p className="text-sm text-text-secondary mb-1">Current Balance</p>
          <p className="text-2xl font-bold text-text">🪙 {(user?.balance ?? 0).toLocaleString()} tokens</p>
        </div>

        {/* Payment Form (when a product is selected) */}
        {selectedProduct && (
          <div className="px-6 py-4">
            <PaymentForm
              product={selectedProduct}
              onSuccess={handlePaymentSuccess}
              onCancel={() => setSelectedProduct(null)}
            />
          </div>
        )}

        {/* Token Packages */}
        {!selectedProduct && (
          <div className="px-6 py-5">
            {loading ? (
              <div className="flex items-center justify-center py-10">
                <Loader className="animate-spin text-primary" size={28} />
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {products.map((product) => (
                  <div
                    key={product.uuid}
                    className="bg-surface-hover border border-border rounded-lg p-5 hover:border-primary transition-colors"
                  >
                    <h3 className="font-bold text-text mb-1">{product.name}</h3>
                    <p className="text-2xl font-bold text-primary mb-1">{product.token_amount.toLocaleString()} 🪙</p>
                    <p className="text-text-secondary text-xs mb-3">{product.description}</p>
                    <div className="flex items-baseline gap-1 mb-4">
                      <span className="text-xl font-bold text-text">${product.price.toFixed(2)}</span>
                      <span className="text-text-secondary text-sm">USD</span>
                    </div>
                    <button
                      onClick={() => setSelectedProduct(product)}
                      className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-primary hover:bg-primary-hover text-white font-medium rounded-lg transition-colors text-sm"
                    >
                      <ShoppingCart size={16} />
                      Buy Now
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

