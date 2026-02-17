'use client'

import { useEffect, useState, useRef, useCallback } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { useAuth } from '@/contexts/AuthContext'
import api from '@/lib/api'
import { Clock, Wifi, WifiOff, ArrowLeft, Zap, Repeat2, AlertCircle, CheckCircle2, Loader, Square, RotateCcw, Brain, Trophy, ChevronDown, ChevronUp } from 'lucide-react'

// WebSocket reconnection constants
const WS_INITIAL_BACKOFF_MS = 2000
const WS_MAX_BACKOFF_MS = 30000
const WS_BACKOFF_MULTIPLIER = 2
const WS_MAX_RETRIES = 5

interface BuildStatus {
  uuid: string
  strategy_id: string
  status: string
  phase: string | null
  tokens_consumed: number
  iteration_count: number
  max_iterations?: number
  started_at: string
  completed_at: string | null
  strategy_type?: string
  strategy_name?: string
}

interface LogMessage {
  type: string
  message?: string
  data?: any
  timestamp?: string
}

// Per-iteration training result from WebSocket
interface IterationResult {
  iteration: number
  total_return: number | null
  win_rate: number | null
  sharpe_ratio: number | null
  model: string | null
}

// Full iteration data from API
interface BuildIteration {
  uuid: string
  build_id: string
  user_id: string
  iteration_number: number
  status: string
  duration_seconds: number | null
  llm_design: any
  llm_thinking: string | null
  training_config: any
  optimal_label_config: any
  features: any
  hyperparameter_results: any
  nn_training_results: any
  lstm_training_results: any
  model_evaluations: any
  best_model: {
    name: string
    metrics: {
      f1: number
      precision: number
      recall: number
      auc: number
    }
  } | null
  entry_rules: Array<{
    feature: string
    operator: string
    threshold: number
    description?: string
    importance?: number
  }> | null
  exit_rules: Array<{
    feature: string
    operator: string
    threshold: number
    description?: string
    importance?: number
  }> | null
  backtest_results: {
    total_return: number
    win_rate: number
    sharpe_ratio: number
    max_drawdown: number
    total_trades: number
    profit_factor: number
    avg_win?: number
    avg_loss?: number
  } | null
  created_at: string
  updated_at: string
}

interface ChatMessage {
  uuid: string
  role: string
  content: string
  created_at: string
  metadata?: { type?: string; iteration?: number; phase?: string } | null
}

const PHASES = ['Queued', 'Designing', 'Training', 'Optimizing', 'Writing Algorithm', 'Testing Algorithm', 'Complete']

// --- Parsed prompt section types and helpers ---

interface PromptSection {
  type: string
  title: string
  content: string
}

/**
 * Parse a raw LLM design prompt into structured sections by splitting on ## headings.
 * Returns an array of { type, title, content } objects for rendering as chat bubbles.
 */
function parseDesignPrompt(content: string): PromptSection[] {
  const sections: PromptSection[] = []
  const parts = content.split(/^## /m)

  // Text before the first ## heading is the role/system intro
  if (parts[0]?.trim()) {
    sections.push({ type: 'role', title: 'System', content: parts[0].trim() })
  }

  for (let i = 1; i < parts.length; i++) {
    const part = parts[i]
    const newlineIdx = part.indexOf('\n')
    const heading = (newlineIdx >= 0 ? part.slice(0, newlineIdx) : part).trim()
    const body = newlineIdx >= 0 ? part.slice(newlineIdx + 1).trim() : ''

    // Map heading text to a section type
    const h = heading.toLowerCase().replace(/:$/, '')
    let type: string
    if (h === 'task') {
      type = 'task'
    } else if (h.includes('available feature')) {
      type = 'features'
    } else if (h.includes('previous builds context')) {
      type = 'context'
    } else if (h.includes('important') || h.includes('critical instructions')) {
      type = 'important'
    } else if (h.includes('previous design')) {
      type = 'previous-design'
    } else if (h.includes('backtest results')) {
      type = 'backtest-results'
    } else if (
      h.includes('output format') ||
      h.includes('rules for entry') ||
      h.includes('ml training configuration') ||
      h.includes('analysis questions')
    ) {
      type = 'hidden'
    } else {
      type = 'other'
    }

    sections.push({ type, title: heading, content: body })
  }

  return sections
}

/** Extract labeled fields (e.g. "- Name: value") from the Task section content. */
function parseTaskFields(content: string): Record<string, string> {
  const fields: Record<string, string> = {}
  for (const line of content.split('\n')) {
    const match = line.match(/^-\s+(.+?):\s*(.*)$/)
    if (match) {
      fields[match[1].trim()] = match[2].trim()
    }
  }
  return fields
}

/** Extract feature names from indented list lines (e.g. "  - feature_name"). */
function parseFeatureNames(content: string): string[] {
  const features: string[] = []
  for (const line of content.split('\n')) {
    const match = line.match(/^\s+-\s+(.+)$/)
    if (match && match[1].trim()) {
      features.push(match[1].trim())
    }
  }
  return features
}

/**
 * Renders a user chat message by parsing the raw LLM prompt into
 * distinct, readable message bubbles. Falls back to original rendering
 * if the prompt cannot be meaningfully parsed.
 */
function DesignPromptBubbles({ msg }: { msg: ChatMessage }) {
  const [featuresExpanded, setFeaturesExpanded] = useState(false)
  const sections = parseDesignPrompt(msg.content)

  // Fallback: if parsing found ≤1 section, render as a single bubble
  if (sections.length <= 1) {
    return (
      <div className="flex gap-3 flex-row-reverse">
        <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 bg-primary/20 text-primary">
          👤
        </div>
        <div className="flex-1 text-right">
          <div className="inline-block max-w-md px-4 py-2 rounded-lg bg-primary/20 text-primary border border-primary/30">
            <p className="text-sm break-words">{msg.content}</p>
          </div>
          <p className="text-xs text-text-secondary mt-1">
            {new Date(msg.created_at).toLocaleTimeString()}
          </p>
        </div>
      </div>
    )
  }

  const isRefinement = sections.some(s => s.type === 'previous-design' || s.type === 'backtest-results')

  return (
    <div className="space-y-3">
      {/* User avatar + timestamp shown once at the top */}
      <div className="flex items-center gap-2 justify-end">
        <span className="text-xs text-text-secondary">
          {new Date(msg.created_at).toLocaleTimeString()}
        </span>
        <div className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 bg-primary/20 text-primary text-xs">
          👤
        </div>
      </div>

      {sections.filter(s => s.type !== 'hidden').map((section, idx) => {
        // Role intro — brief system message
        if (section.type === 'role') {
          const briefIntro = isRefinement
            ? 'Refining strategy based on previous iteration results...'
            : section.content.split('\n')[0]
          return (
            <div key={idx} className="text-center">
              <p className="text-xs text-text-secondary italic">{briefIntro}</p>
            </div>
          )
        }

        // Task section — card with labeled fields
        if (section.type === 'task') {
          const fields = parseTaskFields(section.content)
          return (
            <div key={idx} className="bg-surface-hover border border-border rounded-lg p-4">
              <p className="text-xs font-medium text-text-secondary uppercase tracking-wide mb-3">Strategy Task</p>
              <div className="grid grid-cols-2 gap-2 text-sm">
                {fields['Name'] && (
                  <div><span className="text-text-secondary">Name:</span> <span className="text-text font-medium">{fields['Name']}</span></div>
                )}
                {fields['Symbols'] && (
                  <div><span className="text-text-secondary">Symbols:</span> <span className="text-text font-medium">{fields['Symbols']}</span></div>
                )}
                {fields['Timeframe'] && (
                  <div><span className="text-text-secondary">Timeframe:</span> <span className="text-text font-medium">{fields['Timeframe']}</span></div>
                )}
                {fields['Target Return'] && (
                  <div><span className="text-text-secondary">Target Return:</span> <span className="text-text font-medium">{fields['Target Return']}</span></div>
                )}
                {fields['Description'] && (
                  <div className="col-span-2"><span className="text-text-secondary">Description:</span> <span className="text-text">{fields['Description']}</span></div>
                )}
              </div>
            </div>
          )
        }

        // Features section — collapsible tag chips
        if (section.type === 'features') {
          const features = parseFeatureNames(section.content)
          if (features.length === 0) return null
          return (
            <div key={idx} className="bg-surface-hover border border-border rounded-lg p-3">
              <button
                onClick={() => setFeaturesExpanded(!featuresExpanded)}
                className="flex items-center gap-2 text-xs font-medium text-text-secondary uppercase tracking-wide hover:text-text transition-colors w-full text-left"
              >
                <span>Available Features ({features.length})</span>
                <span>{featuresExpanded ? '▾' : '▸'}</span>
              </button>
              {featuresExpanded && (
                <div className="flex flex-wrap gap-1 mt-2">
                  {features.map(f => (
                    <span key={f} className="px-2 py-0.5 bg-surface rounded text-xs text-text-secondary">{f}</span>
                  ))}
                </div>
              )}
            </div>
          )
        }

        // Context section — show only if non-empty
        if (section.type === 'context') {
          if (!section.content.trim()) return null
          return (
            <div key={idx} className="bg-surface-hover border border-border rounded-lg p-3">
              <p className="text-xs font-medium text-text-secondary uppercase tracking-wide mb-2">Previous Builds Context</p>
              <p className="text-sm text-text whitespace-pre-wrap">{section.content}</p>
            </div>
          )
        }

        // Important / Critical Instructions — highlighted callout
        if (section.type === 'important') {
          return (
            <div key={idx} className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-3">
              <p className="text-xs font-medium text-amber-600 dark:text-amber-400 uppercase tracking-wide mb-1">Important</p>
              <p className="text-sm text-text">{section.content}</p>
            </div>
          )
        }

        // Previous design JSON (refinement prompts) — collapsed by default
        if (section.type === 'previous-design') {
          return (
            <div key={idx} className="bg-surface-hover border border-border rounded-lg p-3">
              <p className="text-xs font-medium text-text-secondary uppercase tracking-wide mb-2">{section.title}</p>
              <pre className="text-xs text-text-secondary overflow-x-auto max-h-32 overflow-y-auto"><code>{section.content}</code></pre>
            </div>
          )
        }

        // Backtest results (refinement prompts) — structured metrics
        if (section.type === 'backtest-results') {
          const metricsLines = section.content.split('\n').filter(l => l.startsWith('- '))
          return (
            <div key={idx} className="bg-surface-hover border border-border rounded-lg p-3">
              <p className="text-xs font-medium text-text-secondary uppercase tracking-wide mb-2">{section.title}</p>
              {metricsLines.length > 0 && (
                <div className="grid grid-cols-2 gap-2 text-sm">
                  {metricsLines.map((line, i) => {
                    const m = line.match(/^-\s+(.+?):\s+(.+)$/)
                    if (!m) return null
                    return (
                      <div key={i}>
                        <span className="text-text-secondary">{m[1]}:</span>{' '}
                        <span className="text-text font-medium">{m[2]}</span>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          )
        }

        // Catch-all for any unrecognized section
        if (section.type === 'other') {
          return (
            <div key={idx} className="bg-surface-hover border border-border rounded-lg p-3">
              <p className="text-xs font-medium text-text-secondary uppercase tracking-wide mb-2">{section.title}</p>
              <p className="text-sm text-text whitespace-pre-wrap">{section.content}</p>
            </div>
          )
        }

        return null
      })}
    </div>
  )
}

/**
 * IterationCard component - displays detailed iteration results
 */
interface IterationCardProps {
  iteration: BuildIteration
  isBest: boolean
  isCurrent: boolean
  isRunning: boolean
}

function IterationCard({ iteration, isBest, isCurrent, isRunning }: IterationCardProps) {
  const [configExpanded, setConfigExpanded] = useState(false)

  const getStatusBadge = (status: string) => {
    const statusLower = status.toLowerCase()
    if (statusLower === 'complete') {
      return <span className="px-2 py-1 text-xs font-medium bg-green-500/20 text-green-600 dark:text-green-400 rounded">Complete</span>
    }
    if (statusLower === 'training') {
      return <span className="px-2 py-1 text-xs font-medium bg-blue-500/20 text-blue-600 dark:text-blue-400 rounded">Training</span>
    }
    if (statusLower === 'designing') {
      return <span className="px-2 py-1 text-xs font-medium bg-purple-500/20 text-purple-600 dark:text-purple-400 rounded">Designing</span>
    }
    if (statusLower === 'failed') {
      return <span className="px-2 py-1 text-xs font-medium bg-red-500/20 text-red-600 dark:text-red-400 rounded">Failed</span>
    }
    return <span className="px-2 py-1 text-xs font-medium bg-surface-hover text-text-secondary rounded">{status}</span>
  }

  return (
    <div
      className={`rounded-lg p-6 border ${isBest
          ? 'bg-primary/5 border-primary/40'
          : 'bg-surface border-border'
        } ${isCurrent && isRunning ? 'animate-pulse-subtle' : ''}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <h3 className="text-lg font-semibold text-text">Iteration #{iteration.iteration_number + 1}</h3>
          {getStatusBadge(iteration.status)}
        </div>
        <div className="flex items-center gap-2">
          {isBest && (
            <span className="flex items-center gap-1 px-2 py-1 text-xs font-medium text-primary bg-primary/10 rounded">
              <Trophy size={14} />
              Best
            </span>
          )}
          {isCurrent && (
            <span className="flex items-center gap-1 px-2 py-1 text-xs font-medium text-text-secondary bg-surface-hover rounded">
              🔄 Current
            </span>
          )}
        </div>
      </div>

      {/* Backtest Results */}
      {iteration.backtest_results && typeof iteration.backtest_results.total_return === 'number' && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-text mb-3">Backtest Results</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            <div className="bg-surface-hover rounded-lg p-3 border border-border">
              <p className="text-xs text-text-secondary mb-1">Total Return</p>
              <p className={`text-lg font-bold ${(iteration.backtest_results?.total_return ?? 0) >= 0
                  ? 'text-green-600 dark:text-green-400'
                  : 'text-red-600 dark:text-red-400'
                }`}>
                {(iteration.backtest_results?.total_return ?? 0).toFixed(2)}%
              </p>
            </div>
            <div className="bg-surface-hover rounded-lg p-3 border border-border">
              <p className="text-xs text-text-secondary mb-1">Win Rate</p>
              <p className="text-sm font-semibold text-text">
                {(iteration.backtest_results?.win_rate ?? 0).toFixed(1)}%
              </p>
            </div>
            <div className="bg-surface-hover rounded-lg p-3 border border-border">
              <p className="text-xs text-text-secondary mb-1">Sharpe Ratio</p>
              <p className="text-sm font-semibold text-text">
                {(() => {
                  const storedSharpe = iteration.backtest_results?.sharpe_ratio ?? 0
                  if (storedSharpe !== 0) {
                    return storedSharpe.toFixed(2)
                  }
                  // Fallback: total_return / max_drawdown when stored value is 0
                  const totalReturn = iteration.backtest_results?.total_return ?? 0
                  const maxDrawdown = iteration.backtest_results?.max_drawdown ?? 0
                  if (totalReturn > 0 && maxDrawdown > 0) {
                    return (totalReturn / maxDrawdown).toFixed(2)
                  }
                  return '0.00'
                })()}
              </p>
            </div>
            <div className="bg-surface-hover rounded-lg p-3 border border-border">
              <p className="text-xs text-text-secondary mb-1">Max Drawdown</p>
              <p className="text-sm font-semibold text-red-600 dark:text-red-400">
                {(iteration.backtest_results?.max_drawdown ?? 0).toFixed(2)}%
              </p>
            </div>
            <div className="bg-surface-hover rounded-lg p-3 border border-border">
              <p className="text-xs text-text-secondary mb-1">Total Trades</p>
              <p className="text-sm font-semibold text-text">
                {iteration.backtest_results.total_trades}
              </p>
            </div>
            <div className="bg-surface-hover rounded-lg p-3 border border-border">
              <p className="text-xs text-text-secondary mb-1">Profit Factor</p>
              <p className="text-sm font-semibold text-text">
                {(() => {
                  const storedProfitFactor = iteration.backtest_results?.profit_factor ?? 0
                  if (storedProfitFactor !== 0) {
                    return storedProfitFactor.toFixed(2)
                  }
                  // Fallback: Math.abs(avg_win / avg_loss) when stored value is 0
                  const avgWin = iteration.backtest_results?.avg_win
                  const avgLoss = iteration.backtest_results?.avg_loss
                  if (avgWin !== undefined && avgLoss !== undefined && avgLoss !== 0) {
                    return Math.abs(avgWin / avgLoss).toFixed(2)
                  }
                  return '—'
                })()}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Best Model */}
      {iteration.best_model && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-text mb-3">Best Model</h4>
          <div className="bg-surface-hover rounded-lg p-4 border border-border">
            <p className="text-sm font-medium text-text mb-3">{iteration.best_model.name}</p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div>
                <p className="text-xs text-text-secondary mb-1">F1 Score</p>
                <p className="text-sm font-semibold text-text">
                  {(iteration.best_model?.metrics?.f1 ?? 0).toFixed(3)}
                </p>
              </div>
              <div>
                <p className="text-xs text-text-secondary mb-1">Precision</p>
                <p className="text-sm font-semibold text-text">
                  {(iteration.best_model?.metrics?.precision ?? 0).toFixed(3)}
                </p>
              </div>
              <div>
                <p className="text-xs text-text-secondary mb-1">Recall</p>
                <p className="text-sm font-semibold text-text">
                  {(iteration.best_model?.metrics?.recall ?? 0).toFixed(3)}
                </p>
              </div>
              <div>
                <p className="text-xs text-text-secondary mb-1">AUC</p>
                <p className="text-sm font-semibold text-text">
                  {(iteration.best_model?.metrics?.auc ?? 0).toFixed(3)}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Entry/Exit Rules */}
      {(iteration.entry_rules || iteration.exit_rules) && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-text mb-3">Trading Rules</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {iteration.entry_rules && iteration.entry_rules.length > 0 && (
              <div className="bg-surface-hover rounded-lg p-4 border border-border">
                <p className="text-xs font-medium text-text-secondary uppercase tracking-wide mb-2">
                  Entry Rules
                </p>
                <div className="space-y-2">
                  {iteration.entry_rules.map((rule, idx) => (
                    <div key={idx} className="text-xs text-text">
                      <span className="font-mono">{rule.feature} {rule.operator} {rule.threshold}</span>
                      {rule.description && (
                        <span className="text-text-secondary ml-2">— {rule.description}</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
            {iteration.exit_rules && iteration.exit_rules.length > 0 && (
              <div className="bg-surface-hover rounded-lg p-4 border border-border">
                <p className="text-xs font-medium text-text-secondary uppercase tracking-wide mb-2">
                  Exit Rules
                </p>
                <div className="space-y-2">
                  {iteration.exit_rules.map((rule, idx) => (
                    <div key={idx} className="text-xs text-text">
                      <span className="font-mono">{rule.feature} {rule.operator} {rule.threshold}</span>
                      {rule.description && (
                        <span className="text-text-secondary ml-2">— {rule.description}</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Training Config (Collapsed) */}
      <div>
        <button
          onClick={() => setConfigExpanded(!configExpanded)}
          className="flex items-center justify-between w-full text-left text-sm font-semibold text-text hover:text-primary transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-background rounded px-2 py-1"
          aria-expanded={configExpanded}
          aria-label="Toggle training configuration"
        >
          <span>Training Configuration</span>
          {configExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
        {configExpanded && (
          <div className="mt-3 bg-surface-hover rounded-lg p-4 border border-border">
            <div className="space-y-3 text-xs">
              {iteration.features && (
                <div>
                  <p className="font-medium text-text-secondary uppercase tracking-wide mb-1">Features</p>
                  <p className="text-text font-mono">{JSON.stringify(iteration.features, null, 2)}</p>
                </div>
              )}
              {iteration.optimal_label_config && (
                <div>
                  <p className="font-medium text-text-secondary uppercase tracking-wide mb-1">Label Config</p>
                  <p className="text-text font-mono">{JSON.stringify(iteration.optimal_label_config, null, 2)}</p>
                </div>
              )}
              {iteration.hyperparameter_results && (
                <div>
                  <p className="font-medium text-text-secondary uppercase tracking-wide mb-1">Hyperparameter Results</p>
                  <p className="text-text font-mono">{JSON.stringify(iteration.hyperparameter_results, null, 2)}</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default function BuildDetailPage() {
  const params = useParams()
  const router = useRouter()
  const buildId = params.id as string
  const { isAuthenticated, isLoading } = useAuth()

  const [build, setBuild] = useState<BuildStatus | null>(null)
  const [latestProgress, setLatestProgress] = useState<LogMessage | null>(null)
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([])
  const [progressMessages, setProgressMessages] = useState<{message: string, iteration?: number, timestamp: string}[]>([])
  const [wsConnected, setWsConnected] = useState(false)
  const [wsRetriesExhausted, setWsRetriesExhausted] = useState(false)
  const [loadingBuild, setLoadingBuild] = useState(true)
  const [loadingChat, setLoadingChat] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // New state for restart, iteration results, and max iterations
  const [restarting, setRestarting] = useState(false)
  // Keep for backward compatibility with WebSocket data (prefixed with _ to indicate intentionally unused)
  const [_iterationResults, setIterationResults] = useState<IterationResult[]>([])
  const [maxIterations, setMaxIterations] = useState<number | null>(null)
  const [tokensPerIteration, setTokensPerIteration] = useState<number | null>(null)

  // State for rich iteration cards from API
  const [buildIterations, setBuildIterations] = useState<BuildIteration[]>([])
  const [_loadingIterations, setLoadingIterations] = useState(false)

  // State for live thinking blocks (streaming from WebSocket)
  const [thinkingBlocks, setThinkingBlocks] = useState<{[blockNumber: number]: {content: string, isComplete: boolean}}>({})
  const [, setCurrentThinkingIteration] = useState<number | null>(null)


  const wsRef = useRef<WebSocket | null>(null)
  const statusIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const seenChatUuidsRef = useRef<Set<string>>(new Set())
  // Track the build UUID separately so WebSocket doesn't re-trigger on every poll update
  const [buildUuid, setBuildUuid] = useState<string | null>(null)
  const iterationsIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const [elapsedTick, setElapsedTick] = useState(0)

  // Tick every second for the Total Time counter
  useEffect(() => {
    const isTerminal = build?.status && ['completed', 'failed', 'stopped'].includes(build.status)
    if (isTerminal) return
    const timer = setInterval(() => setElapsedTick(t => t + 1), 1000)
    return () => clearInterval(timer)
  }, [build?.status])

  // Fetch per-iteration pricing from the backend on page load
  useEffect(() => {
    const fetchPricing = async () => {
      try {
        const res = await api.get('/api/builds/pricing')
        setTokensPerIteration(res.data.tokens_per_iteration)
      } catch (err) {
        console.error('Failed to fetch pricing:', err)
      }
    }
    fetchPricing()
  }, [])

  // Fetch initial build status
  useEffect(() => {
    if (!isLoading && isAuthenticated && buildId) {
      fetchLatestBuild()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated, isLoading, buildId])

  // Fetch chat history once build data is available
  useEffect(() => {
    if (!isLoading && isAuthenticated && build?.strategy_id) {
      fetchChatHistory()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated, isLoading, build?.strategy_id])

  // Polling: refresh build status every 5 seconds
  useEffect(() => {
    if (!isLoading && isAuthenticated && buildId) {
      statusIntervalRef.current = setInterval(() => {
        fetchLatestBuild()
      }, 5000)
    }

    return () => {
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated, isLoading, buildId])

  // Polling: refresh chat history every 5 seconds (only when strategy_id is available)
  const chatIntervalRef = useRef<NodeJS.Timeout | null>(null)
  useEffect(() => {
    if (!isLoading && isAuthenticated && build?.strategy_id) {
      chatIntervalRef.current = setInterval(() => {
        fetchChatHistory()
      }, 5000)
    }

    return () => {
      if (chatIntervalRef.current) {
        clearInterval(chatIntervalRef.current)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated, isLoading, build?.strategy_id])

  // Track build UUID changes — only update once when first discovered (prevents re-triggering on every poll)
  useEffect(() => {
    if (build?.uuid) {
      setBuildUuid((prev) => prev ?? build.uuid)
    }
  }, [build?.uuid])

  // Initialize maxIterations from build.max_iterations when build loads
  useEffect(() => {
    if (build?.max_iterations != null && maxIterations == null) {
      setMaxIterations(build.max_iterations)
    }
  }, [build?.max_iterations, maxIterations])

  // WebSocket connection: depends only on buildUuid state, not on the full build object
  useEffect(() => {
    if (!isLoading && isAuthenticated && buildUuid) {
      connectWebSocket(buildUuid)
    }

    return () => {
      // Clean up WebSocket and any pending reconnect timeout on unmount
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
        reconnectTimeoutRef.current = null
      }
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated, isLoading, buildUuid])

  // Fetch iterations initially and poll every 10 seconds while build is running
  useEffect(() => {
    if (!buildUuid) return

    // Fetch immediately
    fetchBuildIterations()

    // Poll every 10 seconds if build is not terminal
    const terminalStatuses = ['complete', 'failed', 'stopped']
    const isTerminal = build?.status ? terminalStatuses.includes(build.status.toLowerCase()) : false

    if (!isTerminal) {
      iterationsIntervalRef.current = setInterval(() => {
        fetchBuildIterations()
      }, 10000)
    }

    return () => {
      if (iterationsIntervalRef.current) {
        clearInterval(iterationsIntervalRef.current)
        iterationsIntervalRef.current = null
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [buildUuid, build?.status])

  const fetchLatestBuild = async () => {
    try {
      const response = await api.get(`/api/builds/${buildId}`)
      const buildData = response.data
      if (buildData) {
        setBuild(buildData)
        setError(null)
      } else {
        setError('Build not found')
      }
      setLoadingBuild(false)
    } catch (err) {
      console.error('Failed to fetch build status:', err)
      setError('Failed to load build status')
      setLoadingBuild(false)
    }
  }

  const fetchChatHistory = async () => {
    if (!build?.strategy_id) {
      return
    }

    try {
      const response = await api.get(`/api/strategies/${build.strategy_id}/chat`)
      const allMessages = response.data.items || response.data || []

      // Deduplicate by UUID: only keep first occurrence of each UUID
      const deduplicatedMessages = allMessages.filter((msg: ChatMessage) => {
        if (seenChatUuidsRef.current.has(msg.uuid)) {
          return false
        }
        seenChatUuidsRef.current.add(msg.uuid)
        return true
      })

      setChatHistory((prev) => [...prev, ...deduplicatedMessages])
      setLoadingChat(false)
    } catch (err) {
      console.error('Failed to fetch chat history:', err)
      setLoadingChat(false)
    }
  }

  const fetchBuildIterations = async () => {
    if (!buildUuid) return

    try {
      setLoadingIterations(true)
      const response = await api.get(`/api/builds/${buildUuid}/iterations`)
      setBuildIterations(response.data.iterations || [])
      setLoadingIterations(false)
    } catch (err) {
      console.error('Failed to fetch build iterations:', err)
      setLoadingIterations(false)
    }
  }

  // Helper: check if build is in a terminal state (no need for WebSocket)
  const isBuildTerminal = useCallback(() => {
    const terminalStatuses = ['complete', 'failed', 'stopped']
    return build?.status ? terminalStatuses.includes(build.status.toLowerCase()) : false
  }, [build?.status])

  // Schedule a WebSocket reconnect with exponential backoff + jitter
  const scheduleReconnect = useCallback((buildUuid: string) => {
    if (reconnectAttemptsRef.current >= WS_MAX_RETRIES) {
      console.warn(`WebSocket: max retries (${WS_MAX_RETRIES}) exhausted, giving up`)
      setWsRetriesExhausted(true)
      return
    }

    // Calculate backoff: 2s, 4s, 8s, 16s, 30s (capped)
    const baseDelay = WS_INITIAL_BACKOFF_MS * Math.pow(WS_BACKOFF_MULTIPLIER, reconnectAttemptsRef.current)
    const cappedDelay = Math.min(baseDelay, WS_MAX_BACKOFF_MS)
    // Add jitter: ±25% of the delay
    const jitter = cappedDelay * 0.25 * (Math.random() * 2 - 1)
    const delay = Math.round(cappedDelay + jitter)

    console.log(`WebSocket: scheduling reconnect attempt ${reconnectAttemptsRef.current + 1}/${WS_MAX_RETRIES} in ${delay}ms`)

    reconnectTimeoutRef.current = setTimeout(() => {
      reconnectTimeoutRef.current = null
      connectWebSocket(buildUuid)
    }, delay)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const connectWebSocket = useCallback((buildUuid: string) => {
    // Guard: don't connect if build is in a terminal state
    if (isBuildTerminal()) {
      console.log('WebSocket: build is in terminal state, skipping connection')
      return
    }

    // Guard: don't connect if already connected or connecting
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      console.log('WebSocket: already connected or connecting, skipping')
      return
    }

    // Guard: don't connect if retries exhausted
    if (reconnectAttemptsRef.current >= WS_MAX_RETRIES) {
      return
    }

    try {
      // Derive WebSocket URL from the API URL or current window location
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || ''
      let wsBase: string
      if (apiUrl) {
        // Absolute API URL provided — swap protocol from http(s) to ws(s)
        wsBase = apiUrl.replace(/^http/, 'ws')
      } else {
        // Relative/empty API URL — derive from current browser location
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        wsBase = `${protocol}//${window.location.host}`
      }
      const wsEndpoint = `${wsBase}/ws/builds/${buildUuid}/logs`

      const ws = new WebSocket(wsEndpoint)

      ws.onopen = () => {
        console.log('WebSocket connected')
        // Reset backoff on successful connection
        reconnectAttemptsRef.current = 0
        setWsRetriesExhausted(false)
        setWsConnected(true)
      }

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          const newLog: LogMessage = {
            ...message,
            timestamp: new Date().toISOString(),
          }

          // Handle different message types
          if (newLog.type === 'progress') {
            // Update progress in-place (replace previous, don't append)
            setLatestProgress(newLog)

            const progressData = newLog.data || {}

            // Handle thinking blocks (streaming)
            if (progressData.type === 'thinking') {
              const blockNumber = progressData.block_number ?? 1
              const content = progressData.content ?? ''
              const isComplete = progressData.is_complete ?? false
              const iteration = progressData.iteration

              console.log(`🧠 Received thinking block #${blockNumber}: ${content.length} chars, complete=${isComplete}`)

              // Clear thinking blocks when iteration changes (new iteration starting)
              setCurrentThinkingIteration((prevIteration) => {
                if (iteration !== undefined && prevIteration !== null && iteration !== prevIteration) {
                  console.log(`🔄 New iteration detected (${prevIteration} → ${iteration}), clearing thinking blocks`)
                  setThinkingBlocks({})
                }
                return iteration !== undefined ? iteration : prevIteration
              })

              setThinkingBlocks((prev) => {
                const updated = {
                  ...prev,
                  [blockNumber]: { content, isComplete }
                }
                console.log('📊 Updated thinkingBlocks state:', Object.keys(updated).length, 'blocks')

                // Auto-scroll to bottom when thinking is streaming (after state update)
                if (!isComplete) {
                  setTimeout(() => {
                    const messagesContainer = document.querySelector('.design-thinking-messages')
                    if (messagesContainer) {
                      messagesContainer.scrollTop = messagesContainer.scrollHeight
                    }
                  }, 100)
                }

                return updated
              })
            }

            // Accumulate progress messages for rendering as individual bubbles
            if (newLog.message) {
              const message = newLog.message // Capture in const to satisfy TypeScript
              setProgressMessages((prev) => [...prev, {
                message,
                iteration: progressData.iteration,
                timestamp: newLog.timestamp || new Date().toISOString()
              }])
            }

            // Track max_iterations from progress messages
            if (progressData.max_iterations != null) {
              setMaxIterations(progressData.max_iterations)
            }

            // Update tokens_consumed from WebSocket progress (real-time, no 5s delay)
            if (progressData.tokens_consumed != null) {
              setBuild(prev => prev ? { ...prev, tokens_consumed: progressData.tokens_consumed } : prev)
            }

            // Update iteration_count from WebSocket progress (real-time, no 5s delay)
            if (progressData.iteration_count != null) {
              setBuild(prev => prev ? { ...prev, iteration_count: progressData.iteration_count } : prev)
            }

            // Handle per-iteration training results
            if (progressData.phase === 'training_complete' && progressData.results) {
              const result: IterationResult = {
                iteration: progressData.iteration ?? 0,
                total_return: progressData.results.total_return ?? null,
                win_rate: progressData.results.win_rate ?? null,
                sharpe_ratio: progressData.results.sharpe_ratio ?? null,
                model: progressData.results.model ?? null,
              }
              setIterationResults((prev) => {
                // Avoid duplicates for the same iteration
                if (prev.some((r) => r.iteration === result.iteration)) return prev
                return [...prev, result]
              })
            }
          } else if (newLog.type === 'error') {
            // Show error in progress card
            setLatestProgress(newLog)
          }
          // 'status' messages are ignored for UI updates (connection status is handled separately)
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e)
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        setWsConnected(false)
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected')
        setWsConnected(false)
        wsRef.current = null

        // Only attempt reconnect if build is still active
        if (!isBuildTerminal()) {
          reconnectAttemptsRef.current += 1
          scheduleReconnect(buildUuid)
        }
      }

      wsRef.current = ws
    } catch (err) {
      console.error('Failed to connect WebSocket:', err)
      reconnectAttemptsRef.current += 1
      scheduleReconnect(buildUuid)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isBuildTerminal, scheduleReconnect])

  const getCurrentPhaseIndex = () => {
    if (!build?.phase) return 0
    const phaseMap: { [key: string]: number } = {
      'queued': 0,
      'designing': 1,
      'refining': 1,        // NEW: refining is same phase as designing (LLM design step)
      'selecting_best': 1,  // NEW: selecting best iteration (still in design phase)
      'training': 2,
      'optimizing': 3,
      'building_docker': 4,
      'testing_algorithm': 5,
      'complete': 6,
      'completed': 6,       // NEW: backward compat alias
    }
    return phaseMap[build.phase] || 0
  }

  const getPhaseStatus = (index: number) => {
    const currentIndex = getCurrentPhaseIndex()
    if (index < currentIndex) return 'complete'
    if (index === currentIndex) return 'active'
    return 'pending'
  }

  // Helper: Calculate elapsed time — never show negative, stop counting on terminal states
  const getElapsedTime = (tick?: number) => { // tick forces re-render each second
    void tick
    if (!build) return '0 00:00:00'
    const start = new Date(build.started_at).getTime()
    const end = build.completed_at ? new Date(build.completed_at).getTime() : Date.now()
    const totalSeconds = Math.max(0, Math.floor((end - start) / 1000))
    const days = Math.floor(totalSeconds / 86400)
    const hours = Math.floor((totalSeconds % 86400) / 3600)
    const minutes = Math.floor((totalSeconds % 3600) / 60)
    const seconds = totalSeconds % 60
    const pad = (n: number) => n.toString().padStart(2, '0')
    if (days > 0) return `${days} Days ${pad(hours)}:${pad(minutes)}:${pad(seconds)}`
    return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`
  }

  // Stop an active build
  const handleStopBuild = async () => {
    if (!build) return
    try {
      const response = await api.post(`/api/builds/${build.uuid}/stop`)
      // Update build with response data to immediately show "stopping" status
      if (response.data) {
        setBuild(response.data)
      }
    } catch (err: any) {
      console.error('Failed to stop build:', err)
      setError(err.response?.data?.detail || 'Failed to stop build')
    }
  }

  // Restart a failed/stopped build — resumes from where it left off
  const handleRestartBuild = async () => {
    if (!build) return
    setRestarting(true)
    try {
      const response = await api.post(`/api/builds/${build.uuid}/restart`)
      const updatedBuild = response.data

      // Update build state with the resumed build
      setBuild(updatedBuild)
      setRestarting(false)
      setError(null)

      // If it's the same build UUID, stay on this page (resume)
      // If it's a different UUID, redirect to the new build page (full restart)
      if (updatedBuild.uuid !== build.uuid) {
        router.push(`/dashboard/build/${updatedBuild.uuid}`)
      }
    } catch (err: any) {
      console.error('Failed to restart build:', err)
      setError(err.response?.data?.detail || 'Failed to restart build')
      setRestarting(false)
    }
  }

  // Helper: Get status badge color (light/dark theme compatible)
  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'complete':
        return 'bg-green-500/20 text-green-700 dark:text-green-400 border-green-500/30'
      case 'failed':
        return 'bg-red-500/20 text-red-700 dark:text-red-400 border-red-500/30'
      case 'building':
        return 'bg-blue-500/20 text-blue-700 dark:text-blue-400 border-blue-500/30'
      case 'stopping':
        return 'bg-amber-500/20 text-amber-600 dark:text-amber-400 border-amber-500/30'
      case 'stopped':
        return 'bg-yellow-500/20 text-yellow-700 dark:text-yellow-400 border-yellow-500/30'
      default:
        return 'bg-gray-500/20 text-gray-600 dark:text-gray-400 border-gray-500/30'
    }
  }

  // Helper: Get phase description
  const getPhaseDescription = (phase: string | null) => {
    const descriptions: { [key: string]: string } = {
      'queued': 'Queued for processing',
      'designing': 'Designing strategy with Oculus AI...',
      'refining': 'Refining strategy based on previous results...',     // NEW
      'selecting_best': 'Selecting the best iteration...',               // NEW
      'training': 'Training ML models...',
      'optimizing': 'Optimizing parameters...',
      'building_docker': 'Writing algorithm...',
      'testing_algorithm': 'Testing algorithm container...',
      'complete': 'Build complete',
      'completed': 'Build complete',                                     // NEW alias
    }
    return descriptions[phase?.toLowerCase() || ''] || 'Processing...'
  }

  // Helper: Get phase emoji
  const getPhaseEmoji = (phase: string | null) => {
    const emojis: { [key: string]: string } = {
      'queued': '⏳',
      'designing': '🧠',
      'refining': '🔄',          // NEW
      'selecting_best': '🏆',    // NEW
      'training': '🤖',
      'optimizing': '⚙️',
      'building_docker': '✍️',
      'testing_algorithm': '🧪',
      'complete': '✅',
      'completed': '✅',          // NEW alias
    }
    return emojis[phase?.toLowerCase() || ''] || '⏳'
  }

  // Helper: Get phase display name (uppercase)
  const getPhaseDisplayName = (phase: string | null) => {
    const phaseDisplayNames: { [key: string]: string } = {
      'queued': 'QUEUED',
      'designing': 'DESIGNING',
      'refining': 'REFINING',
      'selecting_best': 'SELECTING BEST',
      'training': 'TRAINING',
      'optimizing': 'OPTIMIZING',
      'building_docker': 'WRITING ALGORITHM',
      'testing_algorithm': 'TESTING ALGORITHM',
      'complete': 'COMPLETE',
      'completed': 'COMPLETE',
    }
    return phaseDisplayNames[phase?.toLowerCase() || ''] || phase?.replace(/_/g, ' ').toUpperCase() || ''
  }

  // Helper: Get seconds since timestamp
  const getSecondsSince = (timestamp: string | undefined) => {
    if (!timestamp) return null
    const elapsed = Math.floor((Date.now() - new Date(timestamp).getTime()) / 1000)
    if (elapsed < 60) return `${elapsed}s ago`
    const minutes = Math.floor(elapsed / 60)
    if (minutes < 60) return `${minutes}m ago`
    const hours = Math.floor(minutes / 60)
    return `${hours}h ago`
  }

  // Helper: Parse Claude's JSON design output
  const parseDesignOutput = (content: string) => {
    try {
      // Try to parse as JSON
      const design = JSON.parse(content)
      return design
    } catch {
      // If not JSON, return null
      return null
    }
  }

  // Helper: Render design output as readable card
  // Safely convert a value to a displayable string for JSX rendering
  const safeRenderValue = (value: unknown): string => {
    if (value === null || value === undefined) return ''
    if (typeof value === 'string') return value
    if (typeof value === 'number' || typeof value === 'boolean') return String(value)
    return JSON.stringify(value)
  }

  // Render a single feature item (could be a string or an object)
  const renderFeatureItem = (feature: unknown): string => {
    if (typeof feature === 'string') return feature
    if (typeof feature === 'object' && feature !== null) {
      const f = feature as Record<string, unknown>
      return String(f.feature || f.name || JSON.stringify(feature))
    }
    return String(feature)
  }

  // Render a single indicator item (could be a string or an object)
  const renderIndicatorItem = (indicator: unknown): string => {
    if (typeof indicator === 'string') return indicator
    if (typeof indicator === 'object' && indicator !== null) {
      const ind = indicator as Record<string, unknown>
      return String(ind.name || ind.indicator || JSON.stringify(indicator))
    }
    return String(indicator)
  }

  // Render entry/exit rules which may be a string or an array of rule objects
  const renderRules = (rules: unknown) => {
    if (typeof rules === 'string') {
      return <p className="text-xs text-text-secondary">{rules}</p>
    }
    if (Array.isArray(rules)) {
      return (
        <ul className="space-y-1">
          {rules.map((rule: any, idx: number) => (
            <li key={idx} className="text-xs text-text-secondary">
              <span className="font-medium">
                {safeRenderValue(rule.feature)} {safeRenderValue(rule.operator)} {safeRenderValue(rule.threshold)}
              </span>
              {rule.description && (
                <span className="block text-text-secondary/70 mt-0.5">{safeRenderValue(rule.description)}</span>
              )}
            </li>
          ))}
        </ul>
      )
    }
    // Fallback: stringify whatever it is
    return <p className="text-xs text-text-secondary">{safeRenderValue(rules)}</p>
  }

  const renderDesignCard = (design: any) => {
    if (!design) return null

    return (
      <div className="bg-gradient-to-br from-primary/10 to-primary/5 border border-primary/30 rounded-lg p-5 space-y-4">
        {/* Strategy Name/Description */}
        {design.name && (
          <div>
            <h3 className="text-lg font-bold text-primary mb-1">{safeRenderValue(design.name)}</h3>
            {design.description && (
              <p className="text-sm text-text-secondary">{safeRenderValue(design.description)}</p>
            )}
          </div>
        )}

        {/* Key Metrics */}
        {(design.expected_return || design.risk_level || design.win_rate) && (
          <div className="grid grid-cols-3 gap-3">
            {design.expected_return && (
              <div className="bg-surface rounded p-2">
                <p className="text-xs text-text-secondary">Expected Return</p>
                <p className="text-sm font-semibold text-primary">{safeRenderValue(design.expected_return)}</p>
              </div>
            )}
            {design.risk_level && (
              <div className="bg-surface rounded p-2">
                <p className="text-xs text-text-secondary">Risk Level</p>
                <p className="text-sm font-semibold text-text">{safeRenderValue(design.risk_level)}</p>
              </div>
            )}
            {design.win_rate && (
              <div className="bg-surface rounded p-2">
                <p className="text-xs text-text-secondary">Win Rate</p>
                <p className="text-sm font-semibold text-green-400">{safeRenderValue(design.win_rate)}</p>
              </div>
            )}
          </div>
        )}

        {/* Features */}
        {design.features && Array.isArray(design.features) && design.features.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-2">Features</p>
            <ul className="space-y-1">
              {design.features.slice(0, 5).map((feature: unknown, idx: number) => (
                <li key={idx} className="text-sm text-text flex items-start gap-2">
                  <span className="text-primary flex-shrink-0">•</span>
                  <span>{renderFeatureItem(feature)}</span>
                </li>
              ))}
              {design.features.length > 5 && (
                <li className="text-xs text-text-secondary italic">+{design.features.length - 5} more features</li>
              )}
            </ul>
          </div>
        )}

        {/* Indicators */}
        {design.indicators && Array.isArray(design.indicators) && design.indicators.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-2">Indicators</p>
            <div className="flex flex-wrap gap-2">
              {design.indicators.slice(0, 6).map((indicator: unknown, idx: number) => (
                <span key={idx} className="bg-surface px-2 py-1 rounded text-xs text-text">
                  {renderIndicatorItem(indicator)}
                </span>
              ))}
              {design.indicators.length > 6 && (
                <span className="text-xs text-text-secondary italic">+{design.indicators.length - 6} more</span>
              )}
            </div>
          </div>
        )}

        {/* Entry/Exit Rules */}
        {(design.entry_rules || design.exit_rules) && (
          <div className="grid grid-cols-2 gap-3">
            {design.entry_rules && (
              <div>
                <p className="text-xs font-semibold text-green-400 uppercase tracking-wide mb-2">Entry Rules</p>
                {renderRules(design.entry_rules)}
              </div>
            )}
            {design.exit_rules && (
              <div>
                <p className="text-xs font-semibold text-red-400 uppercase tracking-wide mb-2">Exit Rules</p>
                {renderRules(design.exit_rules)}
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  if (loadingBuild) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <Loader className="animate-spin text-primary" size={24} />
          <h1 className="text-3xl font-bold text-text">Loading Build...</h1>
        </div>
      </div>
    )
  }

  if (!build) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3 mb-4">
          <AlertCircle className="text-red-400" size={24} />
          <h1 className="text-3xl font-bold text-text">Build Not Found</h1>
        </div>
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-700 dark:text-red-300">
          The requested build could not be found.
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header with back button and status */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <button
              onClick={() => window.history.back()}
              className="p-2 hover:bg-surface rounded-lg transition-colors focus-visible:ring-2 focus-visible:ring-primary"
              aria-label="Go back"
            >
              <ArrowLeft size={20} className="text-text-secondary" />
            </button>
            <h1 className="text-3xl font-bold text-text">{build.strategy_name || 'Build Details'}</h1>
            {build.strategy_type && (
              <span className="px-3 py-1 rounded-full text-xs font-semibold bg-primary/10 text-primary border border-primary/20 uppercase tracking-wide">
                {build.strategy_type}
              </span>
            )}
          </div>
          <p className="text-text-secondary text-sm ml-11">Build ID: {build.uuid}</p>
        </div>

        <div className="flex items-center gap-3">
          {/* Stop button — visible only when build is active */}
          {['building', 'stopping'].includes(build.status.toLowerCase()) && (
            <button
              onClick={handleStopBuild}
              disabled={build.status.toLowerCase() === 'stopping'}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-500/20 text-red-700 dark:text-red-400 border border-red-500/30 hover:bg-red-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {build.status.toLowerCase() === 'stopping' ? <Loader size={16} className="animate-spin" /> : <Square size={16} />}
              <span className="text-sm font-medium">{build.status.toLowerCase() === 'stopping' ? 'Stopping...' : 'Stop Build'}</span>
            </button>
          )}

          {/* Restart button — visible when build is failed or stopped */}
          {(build.status.toLowerCase() === 'failed' || build.status.toLowerCase() === 'stopped') && (
            <button
              onClick={handleRestartBuild}
              disabled={restarting}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-500/20 text-blue-700 dark:text-blue-400 border border-blue-500/30 hover:bg-blue-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {restarting ? <Loader size={16} className="animate-spin" /> : <RotateCcw size={16} />}
              <span className="text-sm font-medium">{restarting ? 'Restarting...' : 'Restart Build'}</span>
            </button>
          )}

          {/* Connection status indicator */}
          <div
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-surface border border-border"
            title={wsConnected ? 'Connected to real-time logs' : wsRetriesExhausted ? 'Live updates unavailable — polling for status' : 'Polling for updates'}
          >
            {wsConnected ? (
              <Wifi size={16} className="text-green-400" />
            ) : wsRetriesExhausted ? (
              <WifiOff size={16} className="text-red-400" />
            ) : (
              <WifiOff size={16} className="text-yellow-400" />
            )}
          </div>
        </div>
      </div>

      {/* WebSocket retries exhausted banner */}
      {wsRetriesExhausted && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4 text-yellow-700 dark:text-yellow-300 flex items-start gap-3">
          <WifiOff size={20} className="flex-shrink-0 mt-0.5" />
          <div>Live updates unavailable. Still polling for build status every 5 seconds.</div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-700 dark:text-red-300 flex items-start gap-3">
          <AlertCircle size={20} className="flex-shrink-0 mt-0.5" />
          <div>{error}</div>
        </div>
      )}

      {/* Stats Bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Status Badge */}
        <div className="bg-surface border border-border rounded-lg p-4">
          <p className="text-text-secondary text-xs font-medium mb-2 uppercase tracking-wide">Status</p>
          <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full border text-sm font-semibold ${getStatusColor(build.status)}`}>
            {build.status.toLowerCase() === 'complete' && <CheckCircle2 size={16} />}
            {build.status.toLowerCase() === 'building' && <Loader size={16} className="animate-spin" />}
            {build.status.toLowerCase() === 'stopping' && <Loader size={16} className="animate-spin" />}
            {build.status.toLowerCase() === 'failed' && <AlertCircle size={16} />}
            {build.status.toLowerCase() === 'stopped' && <Square size={16} />}
            <span className="capitalize">{build.status}</span>
          </div>
        </div>

        {/* Tokens Used — shows total consumed with per-iteration cost breakdown */}
        <div className="bg-surface border border-border rounded-lg p-4">
          <p className="text-text-secondary text-xs font-medium mb-2 uppercase tracking-wide">Tokens Used</p>
          <div className="flex items-center gap-2">
            <Zap size={18} className="text-primary" />
            <p className="text-lg font-semibold text-text">{build.tokens_consumed.toFixed(0)} tokens</p>
          </div>
          {/* Cost breakdown: iterations × tokens per iteration */}
          {tokensPerIteration != null && build.iteration_count > 0 && (
            <p className="text-xs text-text-secondary mt-1">
              {build.iteration_count} × {tokensPerIteration} tokens/iter
            </p>
          )}
        </div>

        {/* Iterations — show X / Y when max is known */}
        <div className="bg-surface border border-border rounded-lg p-4">
          <p className="text-text-secondary text-xs font-medium mb-2 uppercase tracking-wide">Iterations</p>
          <div className="flex items-center gap-2">
            <Repeat2 size={18} className="text-primary" />
            <p className="text-lg font-semibold text-text">
              {build.iteration_count}{maxIterations != null ? ` / ${maxIterations}` : ''}
            </p>
          </div>
          {/* Visual progress bar for iterations */}
          {maxIterations != null && maxIterations > 0 && (
            <div className="mt-2 w-full bg-border rounded-full h-1.5">
              <div
                className="bg-primary h-1.5 rounded-full transition-all"
                style={{ width: `${Math.min(100, (build.iteration_count / maxIterations) * 100)}%` }}
              />
            </div>
          )}
        </div>

        {/* Build Duration - estimated total build time based on iterations x 6 min */}
        {maxIterations != null && maxIterations > 0 && (
          <div className="bg-surface border border-border rounded-lg p-4">
            <p className="text-text-secondary text-xs font-medium mb-2 uppercase tracking-wide">Build Duration</p>
            <div className="flex items-center gap-2">
              <Clock size={18} className="text-primary" />
              <p className="text-lg font-semibold text-text">
                ~{maxIterations * 6} min
              </p>
            </div>
          </div>
        )}

        {/* Total Time — fixed elapsed time, stops counting on terminal states */}
        <div className="bg-surface border border-border rounded-lg p-4">
          <p className="text-text-secondary text-xs font-medium mb-2 uppercase tracking-wide">Total Time</p>
          <div className="flex items-center gap-2">
            <Clock size={18} className="text-primary" />
            <p className="text-lg font-semibold text-text">{getElapsedTime(elapsedTick)}</p>
          </div>
        </div>
      </div>

      {/* Main Content: Vertical Timeline + Activity Feed */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Vertical Timeline (Left) */}
        <div className="lg:col-span-1">
          <div className="bg-surface border border-border rounded-lg p-6 sticky top-6">
            <h2 className="text-lg font-semibold text-text mb-6">Progress</h2>
            <div className="space-y-4">
              {PHASES.map((phase, index) => {
                const status = getPhaseStatus(index)
                const isActive = status === 'active'
                const isComplete = status === 'complete'

                return (
                  <div key={phase} className="flex gap-4">
                    {/* Timeline node */}
                    <div className="flex flex-col items-center">
                      <div
                        className={`w-8 h-8 rounded-full flex items-center justify-center transition-all ${isComplete
                            ? 'bg-green-500 text-white'
                            : isActive
                              ? 'bg-gradient-to-r from-[#007cf0] to-[#00dfd8] text-white animate-pulse'
                              : 'bg-border text-text-secondary'
                          }`}
                      >
                        {isComplete ? (
                          <CheckCircle2 size={16} />
                        ) : isActive ? (
                          <Loader size={16} className="animate-spin" />
                        ) : (
                          <span className="text-xs font-semibold">{index + 1}</span>
                        )}
                      </div>
                      {index < PHASES.length - 1 && (
                        <div
                          className={`w-0.5 h-8 mt-2 ${isComplete ? 'bg-green-500' : 'bg-border'
                            }`}
                        />
                      )}
                    </div>

                    {/* Phase label */}
                    <div className="pt-1">
                      <p className={`text-sm font-medium ${isActive ? 'text-primary' : isComplete ? 'text-green-400' : 'text-text-secondary'}`}>
                        {phase}
                      </p>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        {/* Build Status Card + Chat Section (Right) */}
        <div className="lg:col-span-3 space-y-6">
          {/* Build Status Card */}
          <div className="bg-gradient-to-br from-surface to-surface-hover border border-border rounded-lg p-6 shadow-sm">
            <div className="flex items-start justify-between mb-4">
              <h2 className="text-lg font-semibold text-text">Build Status</h2>
              {latestProgress?.timestamp && (
                <p className="text-xs text-text-secondary">Last updated: {getSecondsSince(latestProgress.timestamp)}</p>
              )}
            </div>

            {/* Current Phase Display */}
            {build.phase && (
              <div className="mb-6 pb-6 border-b border-border">
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-2xl">{getPhaseEmoji(build.phase)}</span>
                  <div>
                    <p className="text-sm font-medium text-text-secondary uppercase tracking-wide">Current Phase</p>
                    <p className="text-lg font-semibold text-text">{getPhaseDisplayName(build.phase)}</p>
                  </div>
                </div>
                <p className="text-sm text-text-secondary ml-11">{getPhaseDescription(build.phase)}</p>
              </div>
            )}

            {/* Latest Progress Data */}
            {latestProgress && latestProgress.type === 'progress' && latestProgress.data && (
              <div className="space-y-4">
                {latestProgress.data.details && (
                  <div>
                    <p className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-2">Details</p>
                    <p className="text-sm text-text">{latestProgress.data.details}</p>
                  </div>
                )}

                {latestProgress.data.metrics && Object.keys(latestProgress.data.metrics).length > 0 && (
                  <div>
                    <p className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-3">Metrics</p>
                    <div className="grid grid-cols-2 gap-3">
                      {Object.entries(latestProgress.data.metrics).map(([key, value]) => (
                        <div key={key} className="bg-surface rounded p-3 border border-border/50">
                          <p className="text-xs text-text-secondary mb-1 capitalize">{key.replace(/_/g, ' ')}</p>
                          <p className="text-sm font-semibold text-text">{String(value)}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Error State */}
            {latestProgress && latestProgress.type === 'error' && (
              <div className="bg-red-500/10 border border-red-500/30 rounded p-4">
                <p className="text-sm text-red-700 dark:text-red-300">
                  <span className="font-semibold">Error:</span> {latestProgress.message || 'An error occurred during the build'}
                </p>
              </div>
            )}

            {/* Success State */}
            {build.status.toLowerCase() === 'complete' && (
              <div className="bg-green-500/10 border border-green-500/30 rounded p-4">
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle2 size={18} className="text-green-500" />
                  <p className="font-semibold text-green-700 dark:text-green-300">Build Complete</p>
                </div>
                <p className="text-sm text-green-700 dark:text-green-300">
                  Your strategy has been successfully built and is ready for deployment.
                </p>
              </div>
            )}
          </div>

          {/* Rich Iteration Cards - Best and Current Only */}
          {buildIterations.length > 0 && (() => {
            // Calculate best iteration (highest total_return)
            const bestIteration = buildIterations.reduce((best, iter) => {
              const bestReturn = best.backtest_results?.total_return ?? -Infinity
              const iterReturn = iter.backtest_results?.total_return ?? -Infinity
              return iterReturn > bestReturn ? iter : best
            }, buildIterations[0])

            // Calculate current iteration (highest iteration_number)
            const currentIteration = buildIterations.reduce((current, iter) => {
              return iter.iteration_number > current.iteration_number ? iter : current
            }, buildIterations[0])

            // Check if best and current are the same
            const isSameIteration = bestIteration.uuid === currentIteration.uuid

            // Check if build is running
            const terminalStatuses = ['complete', 'failed', 'stopped']
            const isRunning = build?.status ? !terminalStatuses.includes(build.status.toLowerCase()) : false

            // If only 1 iteration, show it as Current only
            const showBest = buildIterations.length > 1 && !isSameIteration
            const showCurrent = true

            return (
              <div className="bg-surface border border-border rounded-lg p-6">
                <h2 className="text-lg font-semibold text-text mb-6">Iteration Results</h2>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Show single card with both badges if best === current */}
                  {isSameIteration ? (
                    <div className="lg:col-span-2">
                      <IterationCard
                        iteration={currentIteration}
                        isBest={buildIterations.length > 1}
                        isCurrent={true}
                        isRunning={isRunning}
                      />
                    </div>
                  ) : (
                    <>
                      {/* Best Iteration Card */}
                      {showBest && (
                        <IterationCard
                          iteration={bestIteration}
                          isBest={true}
                          isCurrent={false}
                          isRunning={false}
                        />
                      )}
                      {/* Current Iteration Card */}
                      {showCurrent && (
                        <IterationCard
                          iteration={currentIteration}
                          isBest={false}
                          isCurrent={true}
                          isRunning={isRunning}
                        />
                      )}
                    </>
                  )}
                </div>
              </div>
            )
          })()}

          {/* AI Chat Section — renamed heading (Change 1) */}
          <div className="bg-surface border border-border rounded-lg p-6">
            <h2 className="text-lg font-semibold text-text mb-6">Design Thinking</h2>
            <div className="design-thinking-messages space-y-4 max-h-[600px] overflow-y-auto">

              {/* Chat messages - design output is the hero content */}
              {loadingChat ? (
                <div className="text-center py-8">
                  <Loader className="animate-spin text-primary mx-auto mb-2" size={20} />
                  <p className="text-text-secondary text-sm">Loading chat history...</p>
                </div>
              ) : (() => {
                // Combine all messages: chat history + progress messages
                type CombinedMessage = ChatMessage | { role: 'progress', message: string, iteration?: number, timestamp: string, uuid: string }

                const allMessages: CombinedMessage[] = [
                  ...chatHistory,
                  ...progressMessages.map((pm, idx) => ({
                    role: 'progress' as const,
                    message: pm.message,
                    iteration: pm.iteration,
                    timestamp: pm.timestamp,
                    uuid: `progress-${idx}-${pm.timestamp}`
                  }))
                ]

                // Sort all messages chronologically by timestamp
                allMessages.sort((a, b) => {
                  const timeA = 'created_at' in a ? new Date(a.created_at).getTime() : new Date(a.timestamp).getTime()
                  const timeB = 'created_at' in b ? new Date(b.created_at).getTime() : new Date(b.timestamp).getTime()
                  return timeA - timeB
                })

                if (allMessages.length === 0 && Object.keys(thinkingBlocks).length === 0) {
                  return (
                    <div className="text-center py-12">
                      <p className="text-text-secondary">No design messages yet...</p>
                    </div>
                  )
                }

                // Render messages with iteration dividers
                let lastIteration: number | undefined = undefined

                return (
                  <div className="space-y-4">
                    {allMessages.map((msg) => {
                      const isProgress = msg.role === 'progress'
                      const isChatMessage = 'content' in msg
                      const currentIteration = isProgress
                        ? (msg as { role: 'progress', message: string, iteration?: number, timestamp: string, uuid: string }).iteration
                        : (isChatMessage ? (msg as ChatMessage).metadata?.iteration : undefined)

                      // Show iteration divider when iteration changes
                      const showDivider = currentIteration !== undefined && currentIteration !== lastIteration
                      if (currentIteration !== undefined) {
                        lastIteration = currentIteration
                      }

                      const designOutput = isChatMessage && msg.role === 'assistant' ? parseDesignOutput((msg as ChatMessage).content) : null

                      return (
                        <div key={msg.uuid}>
                          {/* Iteration divider */}
                          {showDivider && (
                            <div className="flex items-center gap-3 my-6">
                              <div className="flex-1 h-px bg-border"></div>
                              <span className="text-xs font-medium text-text-secondary uppercase tracking-wider">
                                Iteration #{(currentIteration ?? 0) + 1}
                              </span>
                              <div className="flex-1 h-px bg-border"></div>
                            </div>
                          )}

                          {/* Progress messages */}
                          {isProgress && (() => {
                            const messageText = isChatMessage
                              ? (msg as any).content
                              : (msg as any).message
                            return (
                              <div className="flex gap-3">
                                <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 bg-blue-500/20 text-blue-600 dark:text-blue-400">
                                  ⚙️
                                </div>
                                <div className="flex-1 bg-blue-500/10 border border-blue-500/20 rounded-lg p-3">
                                  <p className="text-xs text-blue-600 dark:text-blue-400">{messageText}</p>
                                </div>
                              </div>
                            )
                          })()}

                          {/* Thinking messages from chat history (role: "thinking") */}
                          {isChatMessage && msg.role === 'thinking' && (() => {
                            const chatMsg = msg as ChatMessage
                            return (
                              <div className="flex gap-3">
                                <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 bg-purple-500/20 text-purple-600 dark:text-purple-400">
                                  <Brain size={16} />
                                </div>
                                <div className="flex-1 bg-purple-500/10 border border-purple-500/20 rounded-lg p-4">
                                  <p className="text-xs font-medium text-purple-600 dark:text-purple-400 uppercase tracking-wide mb-1">Thinking</p>
                                  <p className="text-sm text-text break-words whitespace-pre-wrap">{chatMsg.content}</p>
                                  <p className="text-xs text-text-secondary mt-2">
                                    {new Date(chatMsg.created_at).toLocaleTimeString()}
                                  </p>
                                </div>
                              </div>
                            )
                          })()}

                          {/* User prompt - parsed into conversational bubbles */}
                          {isChatMessage && msg.role === 'user' && (
                            <DesignPromptBubbles msg={msg as ChatMessage} />
                          )}

                          {/* Design output - HERO content */}
                          {isChatMessage && msg.role === 'assistant' && (() => {
                            const chatMsg = msg as ChatMessage
                            return (
                              <div className="flex gap-3">
                                <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 bg-surface-hover text-text-secondary">
                                  🧠
                                </div>
                                <div className="flex-1">
                                  {designOutput ? (
                                    // Render as beautiful design card
                                    <div className="space-y-2">
                                      {renderDesignCard(designOutput)}
                                      <p className="text-xs text-text-secondary">
                                        {new Date(chatMsg.created_at).toLocaleTimeString()}
                                      </p>
                                    </div>
                                  ) : (
                                    // Fallback: render as text if not JSON
                                    <div className="space-y-2">
                                      <div className="bg-surface-hover border border-border rounded-lg p-4">
                                        <p className="text-sm text-text break-words whitespace-pre-wrap">{chatMsg.content}</p>
                                      </div>
                                      <p className="text-xs text-text-secondary">
                                        {new Date(chatMsg.created_at).toLocaleTimeString()}
                                      </p>
                                    </div>
                                  )}
                                </div>
                              </div>
                            )
                          })()}
                        </div>
                      )
                    })}

                    {/* Live streaming thinking blocks - always at bottom (current thinking) */}
                    {Object.keys(thinkingBlocks).length > 0 && (
                      <>
                        {Object.entries(thinkingBlocks)
                          .sort(([a], [b]) => Number(a) - Number(b))
                          .map(([blockNumber, block]) => (
                            <div key={`thinking-${blockNumber}`} className="flex gap-3">
                              <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 bg-purple-500/20 text-purple-600 dark:text-purple-400 ${!block.isComplete ? 'animate-pulse' : ''}`}>
                                <Brain size={16} />
                              </div>
                              <div className={`flex-1 bg-purple-500/10 border rounded-lg p-4 transition-all duration-300 ${!block.isComplete ? 'border-purple-500/40 shadow-lg shadow-purple-500/20' : 'border-purple-500/20'}`}>
                                <div className="flex items-center gap-2 mb-2">
                                  <p className="text-xs font-medium text-purple-600 dark:text-purple-400 uppercase tracking-wide">
                                    Thinking
                                  </p>
                                  {!block.isComplete && (
                                    <div className="flex items-center gap-1">
                                      <Loader className="animate-spin" size={12} />
                                      <span className="text-xs text-purple-500">streaming...</span>
                                    </div>
                                  )}
                                </div>
                                <p className="text-sm text-text break-words whitespace-pre-wrap">
                                  {block.content}
                                  {!block.isComplete && (
                                    <span className="inline-flex ml-1">
                                      <span className="animate-pulse text-purple-500 font-bold">...</span>
                                    </span>
                                  )}
                                </p>
                              </div>
                            </div>
                          ))}
                      </>
                    )}
                  </div>
                )
              })()}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

