'use client'

import { useState, useEffect } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import api from '@/lib/api'
import { ShoppingCart, Loader } from 'lucide-react'
import PaymentForm from './PaymentForm'

interface Product {
  uuid: string
  name: string
  price: number
  token_amount: number
  product_type: string
  description?: string
}

export default function TokensTab() {
  const { user } = useAuth()
  const [products, setProducts] = useState<Product[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null)
  const [balance, setBalance] = useState(user?.balance || 0)

  useEffect(() => {
    fetchProducts()
  }, [])

  const fetchProducts = async () => {
    try {
      const response = await api.get('/api/products')
      // Filter to only show token_package products, not licenses
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
    // Refresh balance
    try {
      const response = await api.get('/api/users/me/balance')
      setBalance(response.data.balance)
    } catch (error) {
      console.error('Failed to refresh balance:', error)
    }
    setSelectedProduct(null)
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader className="animate-spin text-primary" size={32} />
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Balance Display */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-400 rounded-lg p-8 text-white">
        <p className="text-sm font-medium opacity-90 mb-2">Current Token Balance</p>
        <p className="text-5xl font-bold">🪙 {balance.toLocaleString()}</p>
      </div>

      {/* Payment Form Modal */}
      {selectedProduct && (
        <PaymentForm
          product={selectedProduct}
          onSuccess={handlePaymentSuccess}
          onCancel={() => setSelectedProduct(null)}
        />
      )}

      {/* Token Packages */}
      <div>
        <h2 className="text-2xl font-bold text-text mb-4">Token Packages</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {products.map((product) => (
            <div
              key={product.uuid}
              className="bg-surface border border-border rounded-lg p-6 hover:border-primary transition-colors"
            >
              <h3 className="text-lg font-bold text-text mb-2">{product.name}</h3>
              <p className="text-3xl font-bold text-primary mb-4">{product.token_amount.toLocaleString()}</p>
              <p className="text-text-secondary text-sm mb-4">{product.description}</p>
              <div className="flex items-baseline gap-2 mb-6">
                <span className="text-2xl font-bold text-text">${product.price.toFixed(2)}</span>
                <span className="text-text-secondary text-sm">USD</span>
              </div>
              <button
                onClick={() => setSelectedProduct(product)}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-primary hover:bg-primary-hover text-white font-medium rounded-lg transition-colors"
              >
                <ShoppingCart size={18} />
                Buy Now
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Transaction History */}
      <TransactionHistory />
    </div>
  )
}

function TransactionHistory() {
  const [transactions, setTransactions] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchTransactions()
  }, [])

  const fetchTransactions = async () => {
    try {
      const response = await api.get('/api/users/me/balance/history?page=1&page_size=10')
      setTransactions(response.data.transactions)
    } catch (error) {
      console.error('Failed to fetch transactions:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) return <div className="text-text-secondary">Loading transactions...</div>

  return (
    <div className="bg-surface border border-border rounded-lg p-6">
      <h2 className="text-xl font-bold text-text mb-4">Transaction History</h2>
      {transactions.length === 0 ? (
        <p className="text-text-secondary">No transactions yet</p>
      ) : (
        <div className="space-y-3">
          {transactions.map((tx) => (
            <div key={tx.uuid} className="flex justify-between items-center py-3 border-b border-border last:border-0">
              <div>
                <p className="text-text font-medium capitalize">{tx.transaction_type}</p>
                <p className="text-text-secondary text-sm">{tx.description}</p>
              </div>
              <p className={`font-bold ${tx.tokens > 0 ? 'text-green-400' : 'text-red-400'}`}>
                {tx.tokens > 0 ? '+' : ''}{tx.tokens.toLocaleString()}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

