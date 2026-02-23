'use client'

import { useState, useEffect } from 'react'
import { X, FileText, Loader2 } from 'lucide-react'
import { strategyAPI } from '@/lib/api'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface StrategyExplainerModalProps {
  isOpen: boolean
  onClose: () => void
  strategyId: string
  buildId: string
  strategyName: string
}

export default function StrategyExplainerModal({
  isOpen,
  onClose,
  strategyId,
  buildId,
  strategyName,
}: StrategyExplainerModalProps) {
  const [readme, setReadme] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (isOpen && strategyId && buildId) {
      loadReadme()
    }
  }, [isOpen, strategyId, buildId])

  const loadReadme = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await strategyAPI.getBuildReadme(strategyId, buildId)
      setReadme(data.readme || '')
    } catch (err: any) {
      const message = err.response?.data?.detail || err.message || 'Failed to load strategy explainer'
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4">
      <div className="bg-surface border border-border rounded-xl max-w-5xl w-full max-h-[92vh] overflow-hidden shadow-2xl flex flex-col">
        
        {/* Header */}
        <div className="flex items-center justify-between px-8 py-6 border-b border-border bg-surface">
          <div className="flex items-center gap-3">
            <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
              <FileText size={20} className="text-primary" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-text">Strategy Explainer</h2>
              <p className="text-text-secondary text-sm mt-0.5">{strategyName}</p>
            </div>
          </div>
          <button onClick={onClose} className="text-text-secondary hover:text-text transition-colors">
            <X size={24} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-8 py-6">
          {loading && (
            <div className="flex items-center justify-center py-12">
              <Loader2 size={32} className="animate-spin text-primary" />
            </div>
          )}

          {error && (
            <div className="bg-red-900/20 border border-red-700 rounded-lg p-4 text-red-400">
              {error}
            </div>
          )}

          {!loading && !error && readme && (
            <div className="prose prose-invert prose-sm max-w-none
              prose-headings:text-text prose-headings:font-bold
              prose-h1:text-3xl prose-h1:mb-4 prose-h1:mt-6
              prose-h2:text-2xl prose-h2:mb-3 prose-h2:mt-5 prose-h2:border-b prose-h2:border-border prose-h2:pb-2
              prose-h3:text-xl prose-h3:mb-2 prose-h3:mt-4
              prose-p:text-text-secondary prose-p:leading-relaxed prose-p:mb-4
              prose-a:text-primary prose-a:no-underline hover:prose-a:underline
              prose-strong:text-text prose-strong:font-semibold
              prose-code:text-primary prose-code:bg-surface-hover prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:before:content-none prose-code:after:content-none
              prose-pre:bg-surface-hover prose-pre:border prose-pre:border-border prose-pre:rounded-lg prose-pre:p-4
              prose-ul:text-text-secondary prose-ul:list-disc prose-ul:ml-6
              prose-ol:text-text-secondary prose-ol:list-decimal prose-ol:ml-6
              prose-li:mb-1
              prose-table:text-text-secondary prose-table:border-collapse
              prose-th:border prose-th:border-border prose-th:bg-surface-hover prose-th:px-4 prose-th:py-2 prose-th:text-text prose-th:font-semibold
              prose-td:border prose-td:border-border prose-td:px-4 prose-td:py-2
              prose-blockquote:border-l-4 prose-blockquote:border-primary prose-blockquote:pl-4 prose-blockquote:italic prose-blockquote:text-text-secondary
              prose-hr:border-border prose-hr:my-6
              prose-img:rounded-lg prose-img:shadow-lg
            ">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {readme}
              </ReactMarkdown>
            </div>
          )}

          {!loading && !error && !readme && (
            <div className="text-center py-12 text-text-secondary">
              No strategy explainer available for this build.
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-8 py-4 border-t border-border bg-surface flex justify-end">
          <button
            onClick={onClose}
            className="px-6 py-2 bg-surface-hover hover:bg-border text-text rounded-lg transition-colors font-medium"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

