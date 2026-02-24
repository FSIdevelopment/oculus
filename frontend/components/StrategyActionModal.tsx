'use client'

import { useState, useEffect } from 'react'
import { X, CheckCircle, Zap, Users, Globe, ArrowRight, ExternalLink } from 'lucide-react'
import { licenseAPI, strategyAPI } from '@/lib/api'
import LicensePaymentForm from '@/components/LicensePaymentForm'

/** Minimal license shape needed by the modal. */
interface License {
  uuid: string
  status: string
  license_type: string
  strategy_id: string
  webhook_url?: string
  expires_at: string
}

interface StrategyActionModalProps {
  isOpen: boolean
  onClose: () => void
  strategyId: string
  strategyName: string
  /** Licenses for the currently selected strategy (active only). */
  licenses?: License[]
  /**
   * Called after a license is successfully created (in-app payment complete).
   * The parent should use this to refresh its licenses list.
   */
  onLicensePurchase?: () => Promise<void>
  /** Called when the user saves a new webhook URL against an active license. */
  onWebhookUpdate?: (licenseId: string, webhookUrl: string) => Promise<void>
  onMarketplaceSubmit?: (price: number, buildId: string) => Promise<void>
  /** Whether the strategy is currently listed on the marketplace. */
  marketplaceListed?: boolean
}

export default function StrategyActionModal({
  isOpen,
  onClose,
  strategyId,
  strategyName,
  licenses = [],
  onLicensePurchase,
  onWebhookUpdate,
  onMarketplaceSubmit,
  marketplaceListed = false,
}: StrategyActionModalProps) {
  const [activeTab, setActiveTab] = useState<'partner' | 'webhook' | 'marketplace'>('partner')
  const [marketplacePrice, setMarketplacePrice] = useState(10)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [marketplaceError, setMarketplaceError] = useState('')
  const [webhookUrl, setWebhookUrl] = useState('')
  const [webhookSaved, setWebhookSaved] = useState(false)

  /** null = show purchase/renewal buttons; 'monthly'/'annual' = show in-app payment form */
  const [paymentFormType, setPaymentFormType] = useState<'monthly' | 'annual' | null>(null)

  // Renewal state (for users who already hold an active license)
  const [isRenewing, setIsRenewing] = useState(false)
  const [renewalSuccess, setRenewalSuccess] = useState(false)
  const [renewalError, setRenewalError] = useState('')

  // Dynamic license pricing based on strategy performance
  const [licensePrice, setLicensePrice] = useState<{ monthly_price: number; annual_price: number } | null>(null)
  const [priceLoading, setPriceLoading] = useState(false)

  // Completed builds for version selection on the Marketplace tab
  const [completedBuilds, setCompletedBuilds] = useState<Array<{ uuid: string; iteration_count: number; completed_at: string }>>([])
  const [buildsLoading, setBuildsLoading] = useState(false)
  const [selectedBuildId, setSelectedBuildId] = useState<string>('')

  useEffect(() => {
    if (!isOpen || !strategyId) return
    // Reset all transient state whenever the modal opens
    setPaymentFormType(null)
    setRenewalSuccess(false)
    setRenewalError('')
    setPriceLoading(true)
    licenseAPI
      .getLicensePrice(strategyId)
      .then((data) => setLicensePrice(data))
      .catch(() => {
        // Fall back to minimum prices if the request fails
        setLicensePrice({ monthly_price: 50, annual_price: 500 })
      })
      .finally(() => setPriceLoading(false))
  }, [isOpen, strategyId])

  // Fetch completed builds when the marketplace tab is opened (listing only)
  useEffect(() => {
    if (!isOpen || !strategyId || activeTab !== 'marketplace' || marketplaceListed) return
    setBuildsLoading(true)
    strategyAPI
      .getStrategyBuilds(strategyId, 0, 50)
      .then((data: any) => {
        const builds = (data.items ?? data).filter((b: any) => b.status === 'complete')
        setCompletedBuilds(builds)
        if (builds.length > 0 && !selectedBuildId) {
          setSelectedBuildId(builds[0].uuid)
        }
      })
      .catch(() => setCompletedBuilds([]))
      .finally(() => setBuildsLoading(false))
  }, [isOpen, strategyId, activeTab, marketplaceListed])

  if (!isOpen) return null

  const activeLicense = licenses.find((l) => l.status === 'active')
  const hasLicense = !!activeLicense

  /** Called after in-app payment succeeds — refresh parent licenses list. */
  const handlePaymentSuccess = async () => {
    // Refresh licenses first so hasLicense becomes true before clearing
    // paymentFormType — prevents a flash of the purchase buttons between renders.
    if (onLicensePurchase) {
      await onLicensePurchase()
    }
    setPaymentFormType(null)
  }

  /**
   * Renew or switch the billing plan for an existing active license.
   * Calls the backend renew endpoint which processes the charge via the
   * stored Stripe payment method — no new card entry required.
   */
  const handleRenew = async (type: 'monthly' | 'annual') => {
    if (!activeLicense) return
    setIsRenewing(true)
    setRenewalError('')
    setRenewalSuccess(false)
    try {
      await licenseAPI.renewLicense(activeLicense.uuid, type)
      setRenewalSuccess(true)
      // Refresh the parent license list so the updated expiry date is shown
      if (onLicensePurchase) await onLicensePurchase()
      setTimeout(() => setRenewalSuccess(false), 4000)
    } catch (err: any) {
      setRenewalError(
        err?.response?.data?.detail || err?.message || 'Renewal failed. Please try again.'
      )
    } finally {
      setIsRenewing(false)
    }
  }

  const handleMarketplaceSubmit = async () => {
    if (!onMarketplaceSubmit) return
    if (!marketplaceListed && !selectedBuildId) return
    setIsSubmitting(true)
    setMarketplaceError('')
    try {
      await onMarketplaceSubmit(marketplacePrice, selectedBuildId)
      onClose()
    } catch (err: any) {
      const message = err?.response?.data?.detail || err?.message || 'Failed to update marketplace listing'
      setMarketplaceError(message)
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleWebhookSave = async () => {
    if (!onWebhookUpdate || !activeLicense || !webhookUrl.trim()) return
    setIsSubmitting(true)
    try {
      await onWebhookUpdate(activeLicense.uuid, webhookUrl.trim())
      setWebhookSaved(true)
      setTimeout(() => setWebhookSaved(false), 3000)
    } finally {
      setIsSubmitting(false)
    }
  }

  const tabs = [
    { id: 'partner' as const, label: 'Partner Host', icon: <Users size={15} /> },
    { id: 'webhook' as const, label: 'Webhook Signals', icon: <Zap size={15} /> },
    { id: 'marketplace' as const, label: 'Marketplace', icon: <Globe size={15} /> },
  ]

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4">
      <div className="bg-surface border border-border rounded-xl max-w-4xl w-full max-h-[92vh] overflow-y-auto shadow-2xl">

        {/* Header */}
        <div className="flex items-center justify-between px-8 py-6 border-b border-border sticky top-0 bg-surface z-10">
          <div>
            <h2 className="text-2xl font-bold text-text">Deploy Strategy</h2>
            <p className="text-text-secondary text-sm mt-0.5">{strategyName}</p>
          </div>
          <button onClick={onClose} className="text-text-secondary hover:text-text transition-colors">
            <X size={24} />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-border px-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-5 py-3.5 text-sm font-medium transition-colors border-b-2 -mb-px ${activeTab === tab.id
                ? 'text-primary border-primary'
                : 'text-text-secondary border-transparent hover:text-text'
                }`}
            >
              {tab.icon}
              {tab.label}
              {tab.id !== 'marketplace' && (
                <span className="ml-1 text-[10px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary uppercase tracking-wide">
                  License
                </span>
              )}
            </button>
          ))}
        </div>

        {/* ── Partner Host ── */}
        {activeTab === 'partner' && (
          <div className="px-8 py-7 space-y-7">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                <Users size={20} className="text-primary" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-text mb-1">Partner Host</h3>
                <p className="text-text-secondary leading-relaxed">
                  Deploy your strategy to a partner platform for fully automated trading execution,
                  connected directly to a brokerage account. This is the most popular way to get
                  started — the partner handles order routing and risk management on your behalf.
                </p>
              </div>
            </div>

            {/* How it works */}
            <div>
              <h4 className="text-sm font-semibold text-text mb-3 uppercase tracking-wider">How it works</h4>
              <div className="space-y-3">
                {[
                  'Purchase an Oculus license below to activate your strategy.',
                  'Create an account at a supported partner platform.',
                  'Connect your strategy using your Oculus license key inside the partner platform.',
                  'The partner executes trades automatically through your linked brokerage account.',
                ].map((step, i) => (
                  <div key={i} className="flex items-start gap-3">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/15 text-primary text-xs font-bold flex items-center justify-center mt-0.5">
                      {i + 1}
                    </span>
                    <p className="text-text-secondary text-sm leading-relaxed">{step}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Supported partners */}
            <div>
              <h4 className="text-sm font-semibold text-text mb-3 uppercase tracking-wider">Supported Partners</h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {[
                  { name: 'Atlas Trade AI', url: 'https://www.atlastrade.ai', desc: 'AI-powered automated trading with multi-broker support.' },
                  { name: 'SignalSynk', url: 'https://www.signalsynk.com', desc: 'Real-time signal execution with advanced risk controls.' },
                ].map((p) => (
                  <a
                    key={p.name}
                    href={p.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-start justify-between gap-3 bg-surface-hover border border-border rounded-lg p-4 hover:border-primary transition-colors group"
                  >
                    <div>
                      <p className="font-semibold text-text group-hover:text-primary transition-colors">{p.name}</p>
                      <p className="text-xs text-text-secondary mt-0.5">{p.desc}</p>
                    </div>
                    <ExternalLink size={14} className="text-text-secondary flex-shrink-0 mt-0.5" />
                  </a>
                ))}
              </div>
            </div>

            {/* Active license banner — shown above the license section */}
            {hasLicense && activeLicense && (
              <ActiveLicenseBanner license={activeLicense} />
            )}

            {/* License section — purchase or renew */}
            <LicenseSection
              hasLicense={hasLicense}
              activeLicense={activeLicense}
              strategyId={strategyId}
              strategyName={strategyName}
              monthlyPrice={licensePrice?.monthly_price ?? 50}
              annualPrice={licensePrice?.annual_price ?? 500}
              priceLoading={priceLoading}
              paymentFormType={paymentFormType}
              onSetPaymentFormType={setPaymentFormType}
              onPaymentSuccess={handlePaymentSuccess}
              onRenew={handleRenew}
              isRenewing={isRenewing}
              renewalSuccess={renewalSuccess}
              renewalError={renewalError}
            />
          </div>
        )}

        {/* ── Webhook Signals ── */}
        {activeTab === 'webhook' && (
          <div className="px-8 py-7 space-y-7">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                <Zap size={20} className="text-primary" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-text mb-1">Webhook Signals</h3>
                <p className="text-text-secondary leading-relaxed">
                  Run your strategy on Oculus infrastructure and receive live trading signals as JSON
                  at any HTTP endpoint you choose — your own server, a third-party automation tool,
                  or a custom brokerage integration.
                </p>
              </div>
            </div>

            {/* How it works */}
            <div>
              <h4 className="text-sm font-semibold text-text mb-3 uppercase tracking-wider">How it works</h4>
              <div className="space-y-3">
                {[
                  'Purchase a license below. Your strategy container is deployed automatically on our Digital Ocean infrastructure.',
                  'Enter your webhook URL — the endpoint where signals should be sent.',
                  'Your strategy runs on a scheduled basis and POSTs JSON signal payloads to your URL in real time.',
                  'Use the signals to trigger trades via any brokerage API, alerting system, or custom application.',
                ].map((step, i) => (
                  <div key={i} className="flex items-start gap-3">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/15 text-primary text-xs font-bold flex items-center justify-center mt-0.5">
                      {i + 1}
                    </span>
                    <p className="text-text-secondary text-sm leading-relaxed">{step}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Example payload */}
            <div className="bg-surface-hover border border-border rounded-lg p-4">
              <p className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-2">Example Signal Payload</p>
              <pre className="text-xs text-primary font-mono leading-relaxed overflow-x-auto">{`{
  "signal":    "BUY",
  "symbol":    "AAPL",
  "price":     182.45,
  "quantity":  10,
  "timestamp": "2026-02-18T14:32:00Z",
  "strategy":  "${strategyName}"
}`}</pre>
            </div>

            {/* Webhook URL (only when licensed) */}
            {hasLicense && activeLicense && (
              <div className="space-y-3">
                <h4 className="text-sm font-semibold text-text mb-1 uppercase tracking-wider">Your Webhook URL</h4>
                <div className="flex gap-2">
                  <input
                    type="url"
                    placeholder="https://your-endpoint.com/signals"
                    defaultValue={activeLicense.webhook_url || ''}
                    onChange={(e) => setWebhookUrl(e.target.value)}
                    className="flex-1 bg-surface-hover border border-border rounded-lg px-4 py-2 text-text placeholder:text-text-secondary focus:outline-none focus:ring-2 focus:ring-primary text-sm"
                  />
                  <button
                    onClick={handleWebhookSave}
                    disabled={isSubmitting || !webhookUrl.trim()}
                    className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-hover disabled:opacity-50 text-white text-sm font-medium rounded-lg transition-colors"
                  >
                    {webhookSaved ? <><CheckCircle size={15} /> Saved</> : <>Save <ArrowRight size={15} /></>}
                  </button>
                </div>
                <p className="text-xs text-text-secondary">
                  Signals will be POSTed to this URL as JSON. Ensure your endpoint returns HTTP 200.
                </p>
              </div>
            )}

            {/* Active license banner — shown above the license section */}
            {hasLicense && activeLicense && (
              <ActiveLicenseBanner license={activeLicense} />
            )}

            {/* License section — purchase or renew */}
            <LicenseSection
              hasLicense={hasLicense}
              activeLicense={activeLicense}
              strategyId={strategyId}
              strategyName={strategyName}
              monthlyPrice={licensePrice?.monthly_price ?? 50}
              annualPrice={licensePrice?.annual_price ?? 500}
              priceLoading={priceLoading}
              paymentFormType={paymentFormType}
              onSetPaymentFormType={setPaymentFormType}
              onPaymentSuccess={handlePaymentSuccess}
              onRenew={handleRenew}
              isRenewing={isRenewing}
              renewalSuccess={renewalSuccess}
              renewalError={renewalError}
            />
          </div>
        )}

        {/* ── Marketplace ── */}
        {activeTab === 'marketplace' && (
          <div className="px-8 py-7 space-y-7">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                <Globe size={20} className="text-primary" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-text mb-1">Marketplace</h3>
                <p className="text-text-secondary leading-relaxed">
                  Publish your strategy to the Oculus Marketplace and earn revenue when other traders
                  subscribe. You keep <strong className="text-text">65%</strong> of every subscription
                  payment — no license required to list.
                </p>
              </div>
            </div>

            {/* How it works */}
            <div>
              <h4 className="text-sm font-semibold text-text mb-3 uppercase tracking-wider">How it works</h4>
              <div className="space-y-3">
                {[
                  'Set a monthly subscription price for your strategy.',
                  'Submit your strategy to the marketplace listing.',
                  'Traders discover and subscribe to your strategy.',
                  'You receive 65% of each subscription payment directly to your account.',
                ].map((step, i) => (
                  <div key={i} className="flex items-start gap-3">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/15 text-primary text-xs font-bold flex items-center justify-center mt-0.5">
                      {i + 1}
                    </span>
                    <p className="text-text-secondary text-sm leading-relaxed">{step}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Price + revenue */}
            <div className="space-y-4">
              <label className="block">
                <span className="text-sm font-medium text-text mb-2 block">Monthly Subscription Price (USD)</span>
                <div className="flex items-center gap-2">
                  <span className="text-text-secondary text-sm">$</span>
                  <input
                    type="number"
                    min="1"
                    max="1000"
                    value={marketplacePrice}
                    onChange={(e) => setMarketplacePrice(Number(e.target.value))}
                    className="w-40 bg-surface-hover border border-border rounded-lg px-4 py-2 text-text focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                  <span className="text-text-secondary text-sm">/ month per subscriber</span>
                </div>
              </label>

              <div className="bg-surface-hover border border-border rounded-lg p-5 grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-text-secondary uppercase tracking-wider mb-1">You receive</p>
                  <p className="text-2xl font-bold text-primary">${(marketplacePrice * 0.65).toFixed(2)}</p>
                  <p className="text-xs text-text-secondary mt-0.5">65% per subscriber / month</p>
                </div>
                <div>
                  <p className="text-xs text-text-secondary uppercase tracking-wider mb-1">Platform fee</p>
                  <p className="text-2xl font-bold text-text">${(marketplacePrice * 0.35).toFixed(2)}</p>
                  <p className="text-xs text-text-secondary mt-0.5">35% per subscriber / month</p>
                </div>
              </div>
            </div>

            {/* Version selector — only shown when listing (not when unlisting) */}
            {!marketplaceListed && (
              <div className="space-y-2">
                <label className="block">
                  <span className="text-sm font-medium text-text mb-2 block">Strategy Version</span>
                  {buildsLoading ? (
                    <p className="text-sm text-text-secondary">Loading builds…</p>
                  ) : completedBuilds.length === 0 ? (
                    <p className="text-sm text-red-400">No completed builds found. Build your strategy first.</p>
                  ) : (
                    <select
                      value={selectedBuildId}
                      onChange={(e) => setSelectedBuildId(e.target.value)}
                      className="w-full bg-surface-hover border border-border rounded-lg px-4 py-2 text-text focus:outline-none focus:ring-2 focus:ring-primary"
                    >
                      {completedBuilds.map((build, idx) => {
                        const date = new Date(build.completed_at).toLocaleDateString(undefined, {
                          year: 'numeric', month: 'short', day: 'numeric',
                        })
                        return (
                          <option key={build.uuid} value={build.uuid}>
                            Version {completedBuilds.length - idx} — {build.iteration_count} iterations · {date}
                          </option>
                        )
                      })}
                    </select>
                  )}
                </label>
                <p className="text-xs text-text-secondary">
                  Marketplace visitors will see the performance data and explainer for the selected version.
                </p>
              </div>
            )}

            {marketplaceError && (
              <div className="bg-red-900/20 border border-red-700 rounded-lg px-4 py-3 text-red-400 text-sm">
                {marketplaceError}
              </div>
            )}

            <button
              onClick={handleMarketplaceSubmit}
              disabled={isSubmitting || (!marketplaceListed && (buildsLoading || completedBuilds.length === 0 || !selectedBuildId))}
              className={`w-full disabled:opacity-50 text-white font-semibold py-3 rounded-lg transition-all hover:shadow-lg hover:opacity-90 ${marketplaceListed
                ? 'bg-red-600 hover:bg-red-700'
                : 'bg-gradient-to-r from-[#007cf0] to-[#00dfd8]'
                }`}
            >
              {isSubmitting
                ? marketplaceListed ? 'Removing...' : 'Listing...'
                : marketplaceListed ? 'Remove from Marketplace' : 'List on Marketplace'}
            </button>
          </div>
        )}

      </div>
    </div>
  )
}

// ─── Active License Banner ───────────────────────────────────────────────────

interface ActiveLicenseBannerProps {
  license: License
}

/**
 * Prominent banner displayed above the LicenseSection when the user already
 * holds an active license for this strategy. Shows the plan type, renewal
 * date, and the license UUID for reference.
 */
function ActiveLicenseBanner({ license }: ActiveLicenseBannerProps) {
  const planType = license.license_type.charAt(0).toUpperCase() + license.license_type.slice(1)
  const renewsAt = new Date(license.expires_at).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })

  return (
    <div className="flex items-start gap-3 bg-green-900/20 border border-green-700/50 rounded-lg p-4">
      <CheckCircle size={20} className="text-green-400 flex-shrink-0 mt-0.5" />
      <div className="min-w-0 flex-1">
        <p className="text-green-400 font-semibold text-sm">Active License</p>
        <p className="text-text-secondary text-xs mt-0.5">
          {planType} plan · Renews {renewsAt}
        </p>
        <p
          className="text-text-secondary text-xs mt-1 font-mono truncate"
          title={license.uuid}
        >
          ID: {license.uuid}
        </p>
      </div>
    </div>
  )
}

// ─── Shared license purchase / status section ───────────────────────────────

interface LicenseSectionProps {
  hasLicense: boolean
  activeLicense?: License
  strategyId: string
  strategyName: string
  /** Calculated monthly price in whole USD. */
  monthlyPrice: number
  /** Calculated annual price in whole USD. */
  annualPrice: number
  /** True while the price fetch is in flight. */
  priceLoading?: boolean
  /** null = show purchase/renewal buttons; set to open the inline payment form. */
  paymentFormType: 'monthly' | 'annual' | null
  onSetPaymentFormType: (type: 'monthly' | 'annual' | null) => void
  onPaymentSuccess: () => void
  /** Called when the user wants to renew/change their existing license plan. */
  onRenew?: (type: 'monthly' | 'annual') => Promise<void>
  /** True while a renewal request is in-flight. */
  isRenewing?: boolean
  /** True for a few seconds after a successful renewal. */
  renewalSuccess?: boolean
  /** Non-empty string when renewal fails. */
  renewalError?: string
}

function LicenseSection({
  hasLicense,
  activeLicense,
  strategyId,
  strategyName,
  monthlyPrice,
  annualPrice,
  priceLoading,
  paymentFormType,
  onSetPaymentFormType,
  onPaymentSuccess,
  onRenew,
  isRenewing = false,
  renewalSuccess = false,
  renewalError = '',
}: LicenseSectionProps) {
  const monthlyLabel = priceLoading ? 'Calculating…' : `$${monthlyPrice} / month`
  const annualLabel = priceLoading ? 'Calculating…' : `$${annualPrice} / year`

  return (
    <div className="border-t border-border pt-6 space-y-4">
      <h4 className="text-sm font-semibold text-text uppercase tracking-wider">License</h4>

      {hasLicense && activeLicense ? (
        /* ── User already has a license: show renew / extend options ── */
        <div className="space-y-3">
          <p className="text-sm text-text-secondary">
            Renew your subscription or switch between monthly and annual billing. Your strategy
            will continue running without interruption.
          </p>

          {renewalSuccess && (
            <div className="flex items-center gap-2 bg-green-900/20 border border-green-700/50 rounded-lg px-4 py-3 text-green-400 text-sm">
              <CheckCircle size={16} />
              License renewed successfully!
            </div>
          )}

          {renewalError && (
            <div className="bg-red-900/20 border border-red-700 rounded-lg px-4 py-3 text-red-400 text-sm">
              {renewalError}
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <button
              onClick={() => onRenew?.('monthly')}
              disabled={priceLoading || isRenewing}
              className="flex flex-col items-start px-5 py-4 bg-surface-hover border border-border rounded-lg hover:border-primary transition-colors disabled:opacity-50 text-left"
            >
              <span className="text-text font-semibold text-sm">
                {isRenewing ? 'Processing…' : 'Renew Monthly'}
              </span>
              <span className="text-text-secondary text-xs mt-0.5">Billed monthly · Cancel anytime</span>
              <span className="text-primary font-bold mt-2">{monthlyLabel}</span>
            </button>
            <button
              onClick={() => onRenew?.('annual')}
              disabled={priceLoading || isRenewing}
              className="flex flex-col items-start px-5 py-4 bg-surface-hover border border-primary/40 rounded-lg hover:border-primary transition-colors disabled:opacity-50 text-left relative overflow-hidden"
            >
              <span className="absolute top-2 right-2 text-[10px] font-bold px-2 py-0.5 bg-primary text-white rounded-full uppercase tracking-wide">
                Best value
              </span>
              <span className="text-text font-semibold text-sm">
                {activeLicense.license_type === 'monthly' ? 'Upgrade to Annual' : 'Renew Annual'}
              </span>
              <span className="text-text-secondary text-xs mt-0.5">Billed yearly · ~2 months free</span>
              <span className="text-primary font-bold mt-2">{annualLabel}</span>
            </button>
          </div>
          <p className="text-xs text-text-secondary">
            Renewal uses your saved payment method. Changes take effect immediately.
          </p>
        </div>
      ) : paymentFormType ? (
        /* ── Inline Stripe payment form ── */
        <LicensePaymentForm
          strategyId={strategyId}
          strategyName={strategyName}
          licenseType={paymentFormType}
          monthlyPrice={monthlyPrice}
          annualPrice={annualPrice}
          onSuccess={onPaymentSuccess}
          onCancel={() => onSetPaymentFormType(null)}
        />
      ) : (
        /* ── No license: purchase option buttons ── */
        <div className="space-y-3">
          <p className="text-sm text-text-secondary">
            An active Oculus license is required to use this deployment option.
            Licenses are billed as a recurring subscription and can be cancelled at any time.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <button
              onClick={() => onSetPaymentFormType('monthly')}
              disabled={priceLoading}
              className="flex flex-col items-start px-5 py-4 bg-surface-hover border border-border rounded-lg hover:border-primary transition-colors disabled:opacity-50 text-left"
            >
              <span className="text-text font-semibold text-sm">Monthly</span>
              <span className="text-text-secondary text-xs mt-0.5">Billed monthly · Cancel anytime</span>
              <span className="text-primary font-bold mt-2">{monthlyLabel}</span>
            </button>
            <button
              onClick={() => onSetPaymentFormType('annual')}
              disabled={priceLoading}
              className="flex flex-col items-start px-5 py-4 bg-surface-hover border border-primary/40 rounded-lg hover:border-primary transition-colors disabled:opacity-50 text-left relative overflow-hidden"
            >
              <span className="absolute top-2 right-2 text-[10px] font-bold px-2 py-0.5 bg-primary text-white rounded-full uppercase tracking-wide">
                Best value
              </span>
              <span className="text-text font-semibold text-sm">Annual</span>
              <span className="text-text-secondary text-xs mt-0.5">Billed yearly · ~2 months free</span>
              <span className="text-primary font-bold mt-2">{annualLabel}</span>
            </button>
          </div>
          <p className="text-xs text-text-secondary">
            Secure card payment powered by Stripe. Your license activates immediately after payment.
          </p>
        </div>
      )}
    </div>
  )
}

