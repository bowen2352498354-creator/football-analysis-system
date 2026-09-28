import { useEffect } from 'react'
import { createPortal } from 'react-dom'
import { motion } from 'framer-motion'
import {
  Activity,
  BarChart3,
  CalendarDays,
  CheckCircle2,
  ClipboardList,
  ImageOff,
  ShieldCheck,
  Sparkles,
  Target,
  TrendingDown,
  TrendingUp,
  X,
} from 'lucide-react'
import type { IndividualSummaryReport, SelfCheckTask } from '../types'
import BiomechanicalRadar from './BiomechanicalRadar'
import { SelfCheckTaskCard } from './AttemptReportDrawer'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

function assetUrl(path: string | null | undefined): string | null {
  if (!path) return null
  if (/^(data:|https?:\/\/)/i.test(path)) return path
  return `${API_BASE_URL}${path.startsWith('/') ? path : `/${path}`}`
}

function scoreText(value: number | null | undefined, suffix = ''): string {
  return typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(1)}${suffix}` : '—'
}

function SummaryKpi({
  label,
  value,
  accent,
}: {
  label: string
  value: string
  accent: 'emerald' | 'sky' | 'amber'
}) {
  const color = accent === 'emerald' ? 'text-emerald-300' : accent === 'sky' ? 'text-sky-300' : 'text-amber-300'
  return (
    <div className="rounded-2xl border border-white/8 bg-white/[0.035] px-4 py-3">
      <p className="text-[10px] text-white/35">{label}</p>
      <p className={`mt-1 text-xl font-semibold tabular-nums ${color}`}>{value}</p>
    </div>
  )
}

export interface IndividualSummaryReportDrawerProps {
  report: IndividualSummaryReport
  studentId: string
  onClose: () => void
  onTaskChange: (task: SelfCheckTask) => void
}

export default function IndividualSummaryReportDrawer({
  report,
  studentId,
  onClose,
  onTaskChange,
}: IndividualSummaryReportDrawerProps) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  const summary = report.scoreSummary ?? {}
  const change = summary.change
  const representative = report.representativeAttempt
  const frameUrl = assetUrl(representative?.impactFrameUrl)
  const periodStart = report.period?.start || '最早有效记录'
  const periodEnd = report.period?.end || '最近有效记录'
  const requestedStart = report.requestedPeriod?.start || periodStart
  const requestedEnd = report.requestedPeriod?.end || periodEnd
  const summaryTaskSource = report.selfCheckTask?.sourceRecordId

  const narrativeSections = [
    { title: '总体评价', text: report.overallAssessment, icon: ClipboardList, tone: 'text-violet-200' },
    { title: '纵向变化', text: report.progressAnalysis, icon: Activity, tone: 'text-sky-200' },
    { title: '稳定优势', text: report.strengths, icon: CheckCircle2, tone: 'text-emerald-200' },
    { title: '习惯性盲区', text: report.weaknesses, icon: Target, tone: 'text-amber-200' },
  ].filter((item) => item.text)

  return createPortal(
    <motion.div
      className="fixed inset-0 z-[90] flex justify-end bg-black/70 backdrop-blur-sm"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
    >
      <motion.aside
        role="dialog"
        aria-modal="true"
        aria-label="LLM 个人总体分析完整报告"
        initial={{ x: 72, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        exit={{ x: 72, opacity: 0 }}
        transition={{ type: 'spring', stiffness: 320, damping: 32 }}
        className="flex h-full w-[min(1180px,96vw)] flex-col border-l border-violet-300/15 bg-[#070d1b]/98 shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="flex flex-shrink-0 items-start justify-between gap-4 border-b border-white/10 bg-slate-950/85 px-5 py-4 backdrop-blur-xl sm:px-7">
          <div>
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-violet-300" />
              <h2 className="text-base font-semibold text-white sm:text-lg">LLM 个人总体分析完整报告</h2>
            </div>
            <p className="mt-1 text-xs text-white/40">
              {studentId} · 分析范围 {requestedStart} 至 {requestedEnd}
            </p>
          </div>
          <button type="button" onClick={onClose} aria-label="关闭总体分析报告" className="rounded-xl p-2 text-white/45 transition hover:bg-white/10 hover:text-white">
            <X className="h-5 w-5" />
          </button>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-5 sm:px-7 sm:py-6">
          <div className="mx-auto max-w-[1080px] space-y-5 pb-10">
            <section className="rounded-3xl border border-violet-300/15 bg-gradient-to-br from-violet-500/12 via-slate-900/70 to-emerald-500/5 p-5">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-violet-300/75">个人纵向运动表现档案</p>
                  <h3 className="mt-1 text-xl font-semibold text-white">{studentId}</h3>
                  <p className="mt-1 text-xs text-white/40">生成于 {report.generatedAt || '—'}</p>
                  <p className="mt-1 text-[11px] text-white/30">实际A级数据覆盖：{periodStart} 至 {periodEnd}</p>
                </div>
                <div className="flex flex-wrap gap-2 text-[10px]">
                  <span className="rounded-full bg-emerald-500/12 px-3 py-1.5 text-emerald-200 ring-1 ring-emerald-400/20">A级有效 {report.formalAttemptCount ?? 0} 次</span>
                  <span className="rounded-full bg-white/5 px-3 py-1.5 text-white/50 ring-1 ring-white/10">排除 {report.excludedAttemptCount ?? 0} 次</span>
                  <span className="rounded-full bg-violet-500/12 px-3 py-1.5 text-violet-200 ring-1 ring-violet-400/20">结构化正式报告</span>
                </div>
              </div>
              <div className="mt-5 grid gap-3 sm:grid-cols-3">
                <SummaryKpi label="A级样本均分" value={scoreText(summary.mean)} accent="emerald" />
                <SummaryKpi label="最近一次" value={scoreText(summary.latest)} accent="sky" />
                <SummaryKpi label="首末变化" value={scoreText(change, typeof change === 'number' && change > 0 ? ' ↑' : typeof change === 'number' && change < 0 ? ' ↓' : '')} accent="amber" />
              </div>
            </section>

            <div className="grid gap-5 xl:grid-cols-[1.02fr_.98fr]">
              <section className="rounded-3xl border border-sky-400/15 bg-slate-900/60 p-5">
                <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2"><Activity className="h-4 w-4 text-sky-300" /><h3 className="text-sm font-semibold text-white">代表性击球瞬间骨骼定格</h3></div>
                  <span className="text-[10px] text-white/35">最近一次A级有效尝试</span>
                </div>
                {frameUrl ? (
                  <img src={frameUrl} alt={`${studentId} 代表性击球瞬间骨骼定格图`} className="aspect-video w-full rounded-2xl border border-white/8 bg-slate-950 object-contain" />
                ) : (
                  <div className="flex aspect-video w-full flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-white/10 bg-slate-950/70 text-white/30">
                    <ImageOff className="h-8 w-8" />
                    <p className="text-xs">该代表性记录暂无骨骼定格图</p>
                  </div>
                )}
                <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-white/45">
                  <span>{representative?.timestamp || representative?.testDate || '—'}</span>
                  <span>代表性得分 {scoreText(representative?.score)}</span>
                </div>
              </section>

              <section className="rounded-3xl border border-emerald-400/15 bg-slate-900/60 p-5">
                <div className="mb-2 flex items-center gap-2"><BarChart3 className="h-4 w-4 text-emerald-300" /><h3 className="text-sm font-semibold text-white">五维能力整体画像</h3></div>
                <p className="mb-2 text-[10px] text-white/35">仅对筛选范围内A级科研有效样本求均值</p>
                <BiomechanicalRadar scores={report.fiveDimensionScores} primaryLabel="总体均值" compact className="min-h-[310px]" />
              </section>
            </div>

            <section className="rounded-3xl border border-violet-400/18 bg-violet-500/[0.055] p-5">
              <div className="mb-4 flex items-center gap-2"><Sparkles className="h-4 w-4 text-violet-200" /><h3 className="text-sm font-semibold text-violet-100">AIGC 总体分析与处方</h3></div>
              <div className="grid gap-3 md:grid-cols-2">
                {narrativeSections.map((item) => {
                  const Icon = item.icon
                  return (
                    <article key={item.title} className="rounded-2xl border border-white/8 bg-black/20 p-4">
                      <p className={`mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider ${item.tone}`}><Icon className="h-3.5 w-3.5" />{item.title}</p>
                      <p className="whitespace-pre-wrap text-sm leading-relaxed text-white/78">{item.text}</p>
                    </article>
                  )
                })}
              </div>
              <article className="mt-3 rounded-2xl border border-sky-400/15 bg-sky-500/[0.06] p-4">
                <p className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-sky-200"><ShieldCheck className="h-3.5 w-3.5" />训练处方</p>
                <p className="whitespace-pre-wrap text-sm leading-relaxed text-white/80">{report.prescription || '暂无结构化训练处方。'}</p>
                {report.dosage && <p className="mt-3 rounded-xl bg-black/20 px-3 py-2 text-xs text-sky-100">建议剂量：{report.dosage}</p>}
              </article>
            </section>

            <section className="rounded-3xl border border-white/10 bg-slate-900/55 p-5">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2"><CalendarDays className="h-4 w-4 text-amber-300" /><h3 className="text-sm font-semibold text-white">结论证据摘要</h3></div>
                <span className="text-[10px] text-white/30">错误发生率 = 出现次数 ÷ A级有效尝试数</span>
              </div>
              {report.topErrors && report.topErrors.length > 0 ? (
                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                  {report.topErrors.map((item, index) => (
                    <div key={item.label} className="rounded-2xl border border-white/8 bg-black/20 p-3">
                      <p className="text-xs font-medium text-white/80">{index + 1}. {item.label}</p>
                      <p className="mt-1 text-[11px] text-white/38">出现 {item.count} 次 · 发生率 {(item.rate * 100).toFixed(0)}%</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-white/45">A级有效样本中未形成集中的错误分类。</p>
              )}
              <div className="mt-4 flex items-center gap-2 text-xs text-white/45">
                {typeof change === 'number' && change < 0 ? <TrendingDown className="h-4 w-4 text-rose-300" /> : <TrendingUp className="h-4 w-4 text-emerald-300" />}
                <span>首末评分变化：{scoreText(change)}；结论仅代表所选 {requestedStart} 至 {requestedEnd} 范围内的A级有效数据。</span>
              </div>
            </section>

            {report.selfCheckTask && summaryTaskSource && (
              <SelfCheckTaskCard
                updateEndpoint={`${API_BASE_URL}/api/coach/individual-summary/self-check`}
                updateContext={{ sourceRecordId: summaryTaskSource }}
                scopeLabel="针对总体最低维度"
                task={report.selfCheckTask}
                onChange={onTaskChange}
              />
            )}
          </div>
        </div>
      </motion.aside>
    </motion.div>,
    document.body
  )
}
