'use client'

import { useState, useEffect } from 'react'
import { adminAPI } from '@/lib/api'
import { Plus, Edit2, Trash2 } from 'lucide-react'
import ProductFormModal from './ProductFormModal'

interface Product {
  uuid: string
  name: string
  description?: string
  price: number
  product_type: string
  token_amount?: number
}

export default function ProductManagementTab() {
  const [products, setProducts] = useState<Product[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showModal, setShowModal] = useState(false)
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null)

  useEffect(() => {
    fetchProducts()
  }, [])

  const fetchProducts = async () => {
    try {
      setLoading(true)
      const data = await adminAPI.listProducts()
      setProducts(data.data || [])
    } catch (err) {
      setError('Failed to load products')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (productId: string) => {
    if (!confirm('Are you sure you want to delete this product?')) return
    try {
      await adminAPI.deleteProduct(productId)
      setProducts(products.filter((p) => p.uuid !== productId))
    } catch (err) {
      alert('Failed to delete product')
    }
  }

  const handleOpenModal = (product?: Product) => {
    setSelectedProduct(product || null)
    setShowModal(true)
  }

  const handleCloseModal = () => {
    setShowModal(false)
    setSelectedProduct(null)
  }

  const handleSave = () => {
    fetchProducts()
    handleCloseModal()
  }

  return (
    <div className="space-y-4">
      {/* Create Button */}
      <button
        onClick={() => handleOpenModal()}
        className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-hover"
      >
        <Plus size={20} />
        Create Product
      </button>

      {/* Products Grid */}
      {loading ? (
        <div className="text-text-secondary">Loading products...</div>
      ) : error ? (
        <div className="text-red-400">{error}</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {products.map((product) => (
            <div
              key={product.uuid}
              className="bg-surface border border-border rounded-lg p-6 hover:bg-surface-hover transition-colors"
            >
              <div className="space-y-3">
                <div>
                  <h3 className="text-lg font-semibold text-text">{product.name}</h3>
                  <p className="text-text-secondary text-sm">{product.description}</p>
                </div>

                <div className="space-y-1">
                  <p className="text-text">
                    <span className="text-text-secondary">Price:</span> ${(product.price / 100).toFixed(2)}
                  </p>
                  <p className="text-text capitalize">
                    <span className="text-text-secondary">Type:</span> {product.product_type}
                  </p>
                  {product.token_amount && (
                    <p className="text-text">
                      <span className="text-text-secondary">Tokens:</span> {product.token_amount}
                    </p>
                  )}
                </div>

                <div className="flex gap-2 pt-4">
                  <button
                    onClick={() => handleOpenModal(product)}
                    className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-surface-hover border border-border rounded-lg text-text hover:bg-border"
                  >
                    <Edit2 size={16} />
                    Edit
                  </button>
                  <button
                    onClick={() => handleDelete(product.uuid)}
                    className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-red-900/20 border border-red-900/50 rounded-lg text-red-400 hover:bg-red-900/30"
                  >
                    <Trash2 size={16} />
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Product Form Modal */}
      {showModal && (
        <ProductFormModal
          product={selectedProduct}
          onClose={handleCloseModal}
          onSave={handleSave}
        />
      )}
    </div>
  )
}

