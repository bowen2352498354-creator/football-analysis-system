import { useCallback, useEffect, useMemo, useState } from 'react'
import { createPortal } from 'react-dom'
import { motion } from 'framer-motion'
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  ImageOff,
  Loader2,
  RefreshCcw,
  ShieldCheck,
  Sparkles,
  Target,
  X,
} from 'lucide-react'
import BiomechanicalRadar from './BiomechanicalRadar'
import type {
  CoachAttemptReportDetail,
  CoachAttemptReportDetailResponse,
  CoachSanitizerRecord,
  SelfCheckTask,
} from '../types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

function apiAssetUrl(path: string | null | undefined): string | null {
  const value = String(path || '').trim()
  if (!value) return null
  if (/^(https?:|data:)/i.test(value)) return value
  return `${API_BASE_URL}${value.startsWith('/') ? value : `/${value}`}`
}

function groupLabel(type: string | undefined): string {
  if (type === 'realtime') return 'A组 · 实时反馈'
  if (type === 'delayed') return 'B组 · 延时反馈'
  if (type === 'control') return 'C组 · 无反馈采集'
  return type || '未归组'
}

function metricTone(status: string | null | undefined): string {
  const upper = String(status || '').toUpperCase()
  if (upper.includes('GREEN')) return 'border-emerald-400/25 bg-emerald-500/10'
  if (upper.includes('RED')) return 'border-rose-400/25 bg-rose-500/10'
  if (upper.includes('YELLOW')) return 'border-amber-400/25 bg-amber-500/10'
  return 'border-white/10 bg-white/[0.035]'
}

function SelfCheckIllustration({ kind }: { kind: string }) {
  const common = {
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 5,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
  }
  return (
    <svg viewBox="0 0 220 140" className="h-[140px] w-full text-slate-300" role="img" aria-label="动作自查示意图">
      <rect x="2" y="2" width="216" height="136" rx="20" fill="rgba(2,6,23,.38)" stroke="rgba(255,255,255,.08)" />
      {kind === 'support_target' && (
        <>
          <circle cx="145" cy="83" r="20" {...common} />
          <rect x="62" y="66" width="42" height="58" rx="18" fill="rgba(16,185,129,.18)" stroke="rgba(52,211,153,.8)" strokeWidth="3" strokeDasharray="7 5" />
          <path d="M72 43c12-8 30 0 28 13l-5 45c-2 14-25 15-29 2l-8-35c-3-12 3-20 14-25Z" {...common} />
          <path d="m113 94 15-8M119 105l15-7" stroke="#34d399" strokeWidth="4" strokeLinecap="round" />
        </>
      )}
      {kind === 'ankle_lock' && (
        <>
          <path d="M58 30v52c0 13 9 22 22 22h61" {...common} />
          <path d="m89 96 51-15" stroke="#34d399" strokeWidth="8" strokeLinecap="round" />
          <circle cx="161" cy="83" r="23" {...common} />
          <path d="m121 58 20 23-29 4" stroke="#fbbf24" strokeWidth="4" fill="none" />
          <path d="m119 54 5 8-9 1" stroke="#fbbf24" strokeWidth="3" fill="none" />
        </>
      )}
      {kind === 'leg_fold' && (
        <>
          <circle cx="74" cy="35" r="12" {...common} />
          <path d="M74 48v36M74 58l33 20M73 83l32 22 31-43" {...common} />
          <path d="M105 105a34 34 0 0 0 28-18" stroke="#34d399" strokeWidth="4" strokeDasharray="6 5" fill="none" />
          <path d="m128 86 7 2 1-8" stroke="#34d399" strokeWidth="4" fill="none" />
        </>
      )}
      {kind === 'follow_through' && (
        <>
          <circle cx="77" cy="30" r="11" {...common} />
          <path d="M77 43 89 78M86 58l31 7M89 78 63 112M89 78l62 25" {...common} />
          <circle cx="169" cy="101" r="18" {...common} />
          <path d="M105 44c30 4 47 20 51 42" stroke="#34d399" strokeWidth="4" fill="none" strokeDasharray="7 5" />
          <path d="m151 81 6 7 5-8" stroke="#34d399" strokeWidth="4" fill="none" />
        </>
      )}
      {kind === 'approach_steps' && (
        <>
          <ellipse cx="55" cy="96" rx="16" ry="29" transform="rotate(-18 55 96)" {...common} />
          <ellipse cx="107" cy="76" rx="18" ry="32" transform="rotate(12 107 76)" {...common} />
          <ellipse cx="165" cy="94" rx="21" ry="35" transform="rotate(-8 165 94)" stroke="#34d399" strokeWidth="5" fill="rgba(16,185,129,.12)" />
          <text x="49" y="103" fill="currentColor" fontSize="18">1</text>
          <text x="101" y="82" fill="currentColor" fontSize="18">2</text>
          <text x="159" y="100" fill="#34d399" fontSize="18">3</text>
        </>
      )}
    </svg>
  )
}

export interface SelfCheckTaskCardProps {
  updateEndpoint: string
  updateContext?: Record<string, unknown>
  scopeLabel?: string
  task: SelfCheckTask
  onChange: (task: SelfCheckTask) => void
}

export function SelfCheckTaskCard({
  updateEndpoint,
  updateContext,
  scopeLabel = '针对本次最低维度',
  task,
  onChange,
}: SelfCheckTaskCardProps) {
  const [savingKey, setSavingKey] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function persist(
    optimistic: SelfCheckTask,
    body: { slotNo?: number; checked?: boolean; coachVerified?: boolean },
    key: string,
  ) {
    const previous = task
    onChange(optimistic)
    setSavingKey(key)
    setError(null)
    try {
      const response = await fetch(updateEndpoint, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...updateContext, ...body }),
      })
      const data = (await response.json()) as { success?: boolean; task?: SelfCheckTask; detail?: string }
      if (!response.ok || !data.success || !data.task) {
        throw new Error(data.detail || `保存失败（${response.status}）`)
      }
      onChange(data.task)
    } catch (requestError) {
      onChange(previous)
      setError(requestError instanceof Error ? requestError.message : '打卡保存失败')
    } finally {
      setSavingKey(null)
    }
  }

  return (
    <section className="rounded-3xl border border-emerald-400/20 bg-gradient-to-br from-emerald-500/10 via-slate-900/70 to-sky-500/5 p-5">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-emerald-300/75">专属图式自查任务</p>
          <h4 className="mt-1 text-lg font-semibold text-white">{task.title}</h4>
        </div>
        <span className="rounded-full bg-emerald-400/10 px-2.5 py-1 text-[10px] text-emerald-200 ring-1 ring-emerald-400/20">
          {scopeLabel}
        </span>
      </div>

      <div className="grid gap-4 lg:grid-cols-[240px_1fr]">
        <SelfCheckIllustration kind={task.illustrationKey} />
        <div className="space-y-3">
          <div className="rounded-2xl bg-black/25 p-3">
            <p className="text-[10px] text-white/35">动作口令</p>
            <p className="mt-1 text-sm font-semibold leading-relaxed text-emerald-100">{task.instruction}</p>
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            <div className="rounded-2xl bg-black/25 p-3">
              <p className="text-[10px] text-white/35">达标标准</p>
              <p className="mt-1 text-xs leading-relaxed text-white/70">{task.successCriterion}</p>
            </div>
            <div className="rounded-2xl bg-black/25 p-3">
              <p className="text-[10px] text-white/35">练习剂量</p>
              <p className="mt-1 text-xs leading-relaxed text-white/70">{task.dosage}</p>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-white/8 pt-4">
        <div className="flex flex-wrap items-center gap-2">
          {task.checkins.map((checkin) => {
            const key = `slot-${checkin.slotNo}`
            return (
              <button
                key={key}
                type="button"
                disabled={savingKey !== null || task.persistenceAvailable === false}
                onClick={() =>
                  void persist(
                    {
                      ...task,
                      checkins: task.checkins.map((item) =>
                        item.slotNo === checkin.slotNo ? { ...item, checked: !item.checked } : item,
                      ),
                    },
                    { slotNo: checkin.slotNo, checked: !checkin.checked },
                    key,
                  )
                }
                className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-2 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-45 ${
                  checkin.checked
                    ? 'border-emerald-400/35 bg-emerald-500/20 text-emerald-100'
                    : 'border-white/10 bg-white/5 text-white/45 hover:bg-white/10'
                }`}
              >
                {savingKey === key ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <CheckCircle2 className="h-3.5 w-3.5" />
                )}
                第{checkin.slotNo}次
              </button>
            )
          })}
        </div>
        <button
          type="button"
          disabled={savingKey !== null || task.persistenceAvailable === false}
          onClick={() =>
            void persist(
              { ...task, coachVerified: !task.coachVerified },
              { coachVerified: !task.coachVerified },
              'coach',
            )
          }
          className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-2 text-xs font-semibold transition disabled:opacity-45 ${
            task.coachVerified
              ? 'border-sky-400/40 bg-sky-500/20 text-sky-100'
              : 'border-white/10 bg-white/5 text-white/45 hover:bg-white/10'
          }`}
        >
          {savingKey === 'coach' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ShieldCheck className="h-3.5 w-3.5" />}
          {task.coachVerified ? '教练已确认' : '教练确认'}
        </button>
      </div>
      {task.persistenceAvailable === false && (
        <p className="mt-3 text-xs text-amber-200/75">打卡存储暂不可用；任务可查看，但当前不能保存打卡。</p>
      )}
      {error && <p className="mt-3 text-xs text-rose-300">{error}</p>}
    </section>
  )
}

export interface AttemptReportDrawerProps {
  recordId: string
  fallbackRecord?: CoachSanitizerRecord | null
  onClose: () => void
}

export default function AttemptReportDrawer({ recordId, fallbackRecord, onClose }: AttemptReportDrawerProps) {
  const [report, setReport] = useState<CoachAttemptReportDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [imageFailed, setImageFailed] = useState(false)

  const loadReport = useCallback(async (signal?: AbortSignal) => {
    setLoading(true)
    setError(null)
    setImageFailed(false)
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/coach/records/${encodeURIComponent(recordId)}/detail`,
        { signal },
      )
      const data = (await response.json()) as CoachAttemptReportDetailResponse & { detail?: string }
      if (!response.ok || !data.success || !data.record) {
        throw new Error(data.detail || data.message || `详情加载失败（${response.status}）`)
      }
      setReport(data.record)
    } catch (requestError) {
      if (requestError instanceof DOMException && requestError.name === 'AbortError') return
      setReport(null)
      setError(requestError instanceof Error ? requestError.message : '详情加载失败')
    } finally {
      if (!signal?.aborted) setLoading(false)
    }
  }, [recordId])

  useEffect(() => {
    const controller = new AbortController()
    void loadReport(controller.signal)
    return () => controller.abort()
  }, [loadReport])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  const frameUrl = apiAssetUrl(report?.impactFrameUrl)
  const prescriptionCards = useMemo(() => {
    const prescription = report?.aigcPrescription
    if (!prescription) return []
    return [
      { title: '综合评价', text: prescription.overview, tone: 'emerald' },
      { title: '动力链问题与证据', text: prescription.biomechanicalAnalysis, tone: 'rose' },
      { title: '动作纠正口令', text: prescription.correctionCue, tone: 'amber' },
      { title: '训练处方', text: prescription.actionPlan, tone: 'sky' },
    ].filter((item) => item.text)
  }, [report])

  return createPortal(
    <motion.div
      className="fixed inset-0 z-[80] flex justify-end bg-black/65 backdrop-blur-sm"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
    >
      <motion.aside
        role="dialog"
        aria-modal="true"
        aria-label="个人单次测试完整报告"
        initial={{ x: 64, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        exit={{ x: 64, opacity: 0 }}
        transition={{ type: 'spring', stiffness: 330, damping: 32 }}
        className="flex h-full w-[min(1100px,94vw)] flex-col border-l border-white/10 bg-[#07101f]/98 shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="flex flex-shrink-0 items-start justify-between gap-4 border-b border-white/10 bg-slate-950/80 px-5 py-4 backdrop-blur-xl sm:px-7">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-lg font-semibold text-white">个人单次测试完整报告</h2>
              {report?.legacyReport && (
                <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] text-amber-200 ring-1 ring-amber-400/25">旧版报告</span>
              )}
            </div>
            <p className="mt-1 text-xs text-white/40">
              {report?.studentId || fallbackRecord?.studentId || '—'} · {report?.timestamp || fallbackRecord?.timestamp || '未知时间'}
            </p>
          </div>
          <button type="button" onClick={onClose} className="rounded-full p-2 text-white/40 transition hover:bg-white/10 hover:text-white" aria-label="关闭详情">
            <X className="h-4 w-4" />
          </button>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-5 scrollbar-thin scrollbar-thumb-slate-700 sm:px-7">
          {loading && (
            <div className="grid gap-4 lg:grid-cols-2">
              {[0, 1, 2, 3].map((item) => (
                <div key={item} className="h-64 animate-pulse rounded-3xl border border-white/8 bg-white/[0.04]" />
              ))}
            </div>
          )}

          {!loading && error && (
            <div className="flex min-h-[420px] flex-col items-center justify-center gap-4 rounded-3xl border border-rose-400/20 bg-rose-500/5 p-8 text-center">
              <AlertCircle className="h-9 w-9 text-rose-300" />
              <div>
                <p className="font-semibold text-white">完整报告加载失败</p>
                <p className="mt-1 text-sm text-white/45">{error}</p>
              </div>
              <button type="button" onClick={() => void loadReport()} className="inline-flex items-center gap-1.5 rounded-full bg-rose-500/20 px-4 py-2 text-xs text-rose-100 ring-1 ring-rose-400/30">
                <RefreshCcw className="h-3.5 w-3.5" />重试
              </button>
            </div>
          )}

          {!loading && report && (
            <div className="space-y-5 pb-8">
              <section className="flex flex-wrap items-center justify-between gap-4 rounded-3xl border border-white/10 bg-white/[0.04] p-4">
                <div className="flex items-center gap-3">
                  <div className="flex h-16 w-16 flex-col items-center justify-center rounded-2xl bg-amber-400/10 ring-1 ring-amber-400/25">
                    <span className="text-2xl font-bold text-amber-200">{typeof report.score === 'number' ? report.score : '—'}</span>
                    <span className="text-[9px] text-white/35">综合分 / 100</span>
                  </div>
                  <div>
                    <p className="font-semibold text-white">{report.studentId || '—'}</p>
                    <p className="mt-1 text-xs text-white/40">{report.school || '未设置学校'} · {report.classGroup || '未设置班级'}</p>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2 text-[11px]">
                  <span className="rounded-full bg-sky-500/10 px-2.5 py-1 text-sky-200 ring-1 ring-sky-400/20">{groupLabel(report.type)}</span>
                  <span className="rounded-full bg-white/5 px-2.5 py-1 text-white/55 ring-1 ring-white/10">数据质量 {report.qualityGrade || '未评级'}</span>
                  <span className="rounded-full bg-white/5 px-2.5 py-1 text-white/55 ring-1 ring-white/10">{report.reportStatus === 'formal' ? '正式报告' : report.reportStatus || '归档报告'}</span>
                </div>
              </section>

              <div className="grid gap-5 lg:grid-cols-2">
                <section className="rounded-3xl border border-white/10 bg-white/[0.035] p-4">
                  <div className="mb-3 flex items-center justify-between gap-2">
                    <h3 className="flex items-center gap-2 text-sm font-semibold text-white"><Activity className="h-4 w-4 text-sky-300" />击球瞬间骨骼定格</h3>
                    <span className="text-[10px] text-white/30">测试时自动采集 · 非后期模拟</span>
                  </div>
                  <div className="flex min-h-[315px] items-center justify-center overflow-hidden rounded-2xl border border-white/8 bg-black/35">
                    {frameUrl && !imageFailed ? (
                      <img src={frameUrl} alt={`${report.studentId} 击球瞬间骨骼定格图`} className="max-h-[420px] w-full object-contain" onError={() => setImageFailed(true)} />
                    ) : (
                      <div className="flex flex-col items-center gap-2 px-6 text-center text-white/30"><ImageOff className="h-8 w-8" /><p className="text-xs">该历史记录未保存可读取的骨骼定格图</p></div>
                    )}
                  </div>
                  {report.metrics.length > 0 && (
                    <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3">
                      {report.metrics.slice(0, 6).map((metric) => (
                        <div key={metric.key} className={`rounded-xl border bg-black/25 px-3 py-2 ${metricTone(metric.status)}`}>
                          <p className="truncate text-[10px] text-white/35">{metric.label}</p>
                          <p className="mt-0.5 text-sm font-semibold tabular-nums text-white/80">{metric.value} <span className="text-[10px] font-normal text-white/35">{metric.unit}</span></p>
                        </div>
                      ))}
                    </div>
                  )}
                </section>

                <section className="rounded-3xl border border-white/10 bg-white/[0.035] p-4">
                  <BiomechanicalRadar scores={report.fiveDimensionScores} primaryLabel="本次测试" />
                  {report.biomechanicalErrors.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {report.biomechanicalErrors.map((errorLabel) => (
                        <span key={errorLabel} className="rounded-lg bg-rose-500/12 px-2 py-1 text-[10px] text-rose-200 ring-1 ring-rose-400/20">{errorLabel}</span>
                      ))}
                    </div>
                  )}
                </section>
              </div>

              <section className="rounded-3xl border border-violet-400/20 bg-violet-500/[0.055] p-5">
                <div className="mb-4 flex items-center gap-2"><Sparkles className="h-4 w-4 text-violet-200" /><h3 className="text-sm font-semibold text-violet-100">AIGC 分析处方</h3></div>
                {prescriptionCards.length > 0 ? (
                  <div className="grid gap-3 md:grid-cols-2">
                    {prescriptionCards.map((item) => (
                      <article key={item.title} className="rounded-2xl border border-white/8 bg-black/20 p-4">
                        <p className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-white/40">{item.tone === 'rose' ? <Target className="h-3 w-3 text-rose-300" /> : <Sparkles className="h-3 w-3 text-emerald-300" />}{item.title}</p>
                        <p className="whitespace-pre-wrap text-sm leading-relaxed text-white/78">{item.text}</p>
                        {item.title === '训练处方' && report.aigcPrescription.dosage && <p className="mt-2 rounded-xl bg-sky-500/10 px-3 py-2 text-xs text-sky-100">建议剂量：{report.aigcPrescription.dosage}</p>}
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap rounded-2xl bg-black/20 p-4 text-sm leading-relaxed text-white/65">{report.aigcPrescription.legacyText || '该记录暂无 AIGC 分析处方。'}</p>
                )}
              </section>

              <SelfCheckTaskCard
                updateEndpoint={`${API_BASE_URL}/api/coach/records/${encodeURIComponent(report.id)}/self-check`}
                task={report.selfCheckTask}
                onChange={(task) => setReport((current) => current ? { ...current, selfCheckTask: task } : current)}
              />
            </div>
          )}
        </div>
      </motion.aside>
    </motion.div>,
    document.body
  )
}
