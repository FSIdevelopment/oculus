'use client'

import { useState } from 'react'
import { adminAPI } from '@/lib/api'
import { X } from 'lucide-react'

interface Product {
  uuid: string
  name: string
  description?: string
  price: number
  product_type: string
  token_amount?: number
}

interface ProductFormModalProps {
  product: Product | null
  onClose: () => void
  onSave: () => void
}

export default function ProductFormModal({ product, onClose, onSave }: ProductFormModalProps) {
  const [formData, setFormData] = useState({
    name: product?.name || '',
    description: product?.description || '',
    price: product?.price || 0,
    product_type: product?.product_type || 'token_package',
    token_amount: product?.token_amount || 0,
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({
      ...prev,
      [name]: name === 'price' || name === 'token_amount' ? parseFloat(value) || 0 : value,
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      setLoading(true)
      setError(null)

      const submitData = {
        name: formData.name,
        description: formData.description,
        product_type: formData.product_type,
        token_amount: formData.token_amount || undefined,
        price_cents: Math.round(formData.price * 100),
      }

      if (product) {
        await adminAPI.updateProduct(product.uuid, submitData)
      } else {
        await adminAPI.createProduct(submitData)
      }
      onSave()
    } catch (err) {
      setError('Failed to save product')
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
          <h2 className="text-xl font-bold text-text">{product ? 'Edit Product' : 'Create Product'}</h2>
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
            <label className="block text-sm font-medium text-text mb-2">Name</label>
            <input
              type="text"
              name="name"
              value={formData.name}
              onChange={handleChange}
              required
              className="w-full px-3 py-2 bg-surface-hover border border-border rounded-lg text-text focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text mb-2">Description</label>
            <textarea
              name="description"
              value={formData.description}
              onChange={handleChange}
              rows={3}
              className="w-full px-3 py-2 bg-surface-hover border border-border rounded-lg text-text focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text mb-2">Price ($)</label>
            <input
              type="number"
              name="price"
              value={formData.price}
              onChange={handleChange}
              step="0.01"
              min="0"
              required
              className="w-full px-3 py-2 bg-surface-hover border border-border rounded-lg text-text focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text mb-2">Product Type</label>
            <select
              name="product_type"
              value={formData.product_type}
              onChange={handleChange}
              className="w-full px-3 py-2 bg-surface-hover border border-border rounded-lg text-text focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="token_package">Token Package</option>
              <option value="license">License</option>
            </select>
          </div>

          {formData.product_type === 'token_package' && (
            <div>
              <label className="block text-sm font-medium text-text mb-2">Token Amount</label>
              <input
                type="number"
                name="token_amount"
                value={formData.token_amount}
                onChange={handleChange}
                min="0"
                className="w-full px-3 py-2 bg-surface-hover border border-border rounded-lg text-text focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
          )}

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
              className="flex-1 px-4 py-2 bg-gradient-to-r from-[#007cf0] to-[#00dfd8] text-white rounded-lg hover:shadow-lg transition-all disabled:opacity-50"
            >
              {loading ? 'Saving...' : 'Save Product'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

