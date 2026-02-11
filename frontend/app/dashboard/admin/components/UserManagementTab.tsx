'use client'

import { useState, useEffect } from 'react'
import { adminAPI } from '@/lib/api'
import { Search, Edit2, Trash2, RotateCcw, Eye, Plus } from 'lucide-react'
import UserDetailModal from './UserDetailModal'
import UserCreateModal from './UserCreateModal'

interface User {
  uuid: string
  name: string
  email: string
  phone_number?: string
  user_role: string
  status: string
  balance: number
  created_at: string
}

export default function UserManagementTab() {
  const [users, setUsers] = useState<User[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [skip, setSkip] = useState(0)
  const [total, setTotal] = useState(0)
  const [selectedUser, setSelectedUser] = useState<User | null>(null)
  const [showModal, setShowModal] = useState(false)
  const [showCreateModal, setShowCreateModal] = useState(false)

  const limit = 20

  useEffect(() => {
    fetchUsers()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [skip, search])

  const fetchUsers = async () => {
    try {
      setLoading(true)
      const data = await adminAPI.listUsers({
        skip,
        limit,
        search: search || undefined,
      })
      setUsers(data.items || [])
      setTotal(data.total || 0)
    } catch (err) {
      setError('Failed to load users')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (userId: string) => {
    if (!confirm('Are you sure you want to delete this user?')) return
    try {
      await adminAPI.deleteUser(userId)
      setUsers(users.filter((u) => u.uuid !== userId))
    } catch (err) {
      alert('Failed to delete user')
    }
  }

  const handleResetPassword = async (userId: string) => {
    if (!confirm('Reset password for this user?')) return
    try {
      await adminAPI.resetPassword(userId)
      alert('Password reset email sent')
    } catch (err) {
      alert('Failed to reset password')
    }
  }

  const handleImpersonate = async (userId: string) => {
    try {
      const response = await adminAPI.impersonateUser(userId)
      if (response.impersonation_token) {
        window.open(`/dashboard?token=${response.impersonation_token}`, '_blank')
      }
    } catch (err) {
      alert('Failed to impersonate user')
    }
  }

  return (
    <div className="space-y-4">
      {/* Search Bar and Create Button */}
      <div className="flex gap-2">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-3 text-text-secondary" size={20} />
          <input
            type="text"
            placeholder="Search by name or email..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value)
              setSkip(0)
            }}
            className="w-full pl-10 pr-4 py-2 bg-surface border border-border rounded-lg text-text placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-primary"
          />
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-hover flex items-center gap-2"
        >
          <Plus size={20} />
          Create User
        </button>
      </div>

      {/* Users Table */}
      {loading ? (
        <div className="text-text-secondary">Loading users...</div>
      ) : error ? (
        <div className="text-red-400">{error}</div>
      ) : (
        <>
          <div className="overflow-x-auto bg-surface border border-border rounded-lg">
            <table className="w-full">
              <thead className="border-b border-border">
                <tr className="bg-surface-hover">
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Name</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Email</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Role</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Status</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Balance</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-text">Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.map((user, idx) => (
                  <tr
                    key={user.uuid}
                    className={`border-b border-border ${idx % 2 === 0 ? 'bg-surface' : 'bg-surface-hover'}`}
                  >
                    <td className="px-6 py-4 text-text">{user.name}</td>
                    <td className="px-6 py-4 text-text-secondary text-sm">{user.email}</td>
                    <td className="px-6 py-4 text-text capitalize">{user.user_role}</td>
                    <td className="px-6 py-4 text-text capitalize">{user.status}</td>
                    <td className="px-6 py-4 text-text">{user.balance}</td>
                    <td className="px-6 py-4">
                      <div className="flex gap-2">
                        <button
                          onClick={() => {
                            setSelectedUser(user)
                            setShowModal(true)
                          }}
                          className="p-2 hover:bg-surface-hover rounded text-text-secondary hover:text-text"
                          title="View/Edit"
                        >
                          <Edit2 size={16} />
                        </button>
                        <button
                          onClick={() => handleResetPassword(user.uuid)}
                          className="p-2 hover:bg-surface-hover rounded text-text-secondary hover:text-text"
                          title="Reset Password"
                        >
                          <RotateCcw size={16} />
                        </button>
                        <button
                          onClick={() => handleImpersonate(user.uuid)}
                          className="p-2 hover:bg-surface-hover rounded text-text-secondary hover:text-text"
                          title="Impersonate"
                        >
                          <Eye size={16} />
                        </button>
                        <button
                          onClick={() => handleDelete(user.uuid)}
                          className="p-2 hover:bg-red-900/20 rounded text-text-secondary hover:text-red-400"
                          title="Delete"
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex justify-between items-center">
            <p className="text-text-secondary text-sm">
              Showing {skip + 1} to {Math.min(skip + limit, total)} of {total}
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => setSkip(Math.max(0, skip - limit))}
                disabled={skip === 0}
                className="px-4 py-2 bg-surface border border-border rounded-lg text-text disabled:opacity-50 hover:bg-surface-hover"
              >
                Previous
              </button>
              <button
                onClick={() => setSkip(skip + limit)}
                disabled={skip + limit >= total}
                className="px-4 py-2 bg-surface border border-border rounded-lg text-text disabled:opacity-50 hover:bg-surface-hover"
              >
                Next
              </button>
            </div>
          </div>
        </>
      )}

      {/* User Detail Modal */}
      {showModal && selectedUser && (
        <UserDetailModal
          user={selectedUser}
          onClose={() => {
            setShowModal(false)
            setSelectedUser(null)
          }}
          onUpdate={() => {
            fetchUsers()
            setShowModal(false)
            setSelectedUser(null)
          }}
        />
      )}

      {/* User Create Modal */}
      {showCreateModal && (
        <UserCreateModal
          onClose={() => setShowCreateModal(false)}
          onSuccess={() => {
            fetchUsers()
            setShowCreateModal(false)
          }}
        />
      )}
    </div>
  )
}

