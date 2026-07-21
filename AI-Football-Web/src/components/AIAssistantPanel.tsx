import { useMemo, useState, type ReactNode } from 'react'
import { Filter, Sparkles } from 'lucide-react'
import { DeductionList, ERROR_CODE_LABELS, GOLDEN_METRIC_DEFS } from './ProStudioPanels'
import { TRAFFIC_CLASS, type TrafficLightLevel } from '../theme/trafficLight'
import type { FinalDiagnosisReport, ThresholdHitStats } from '../types'

export type DefectFilterId = 'all' | TrafficLightLevel

export interface AIAssistantPanelProps {
  /** DeepSeek 最终诊断；待机时展示引导文案 */
  report?: FinalDiagnosisReport | null
  /** 实时三级命中统计（用于缺陷筛选器计数） */
  hitStats?: ThresholdHitStats
  /** 命中的错误代码，驱动扣分清单 */
  errorCodes?: string[] | null
  /** 报告打字机展示文本（可选，缺省用 report.fullText） */
  displayText?: string
  /** 打字机是否仍在输出 */
  isTyping?: boolean
  /** 额外操作区（归档 / 导出按钮等） */
  actions?: ReactNode
  className?: string
}

const FILTER_OPTIONS: { id: DefectFilterId; label: string }[] = [
  { id: 'all', label: '全部' },
  { id: 'green', label: '达标' },
  { id: 'yellow', label: '接近' },
  { id: 'red', label: '偏离' },
]

/**
 * V2.5 右栏 AIAssistantPanel（28%）
 * AIGC 具身隐喻建议 + 缺陷筛选器，对标 Hudl 侧栏教练助手。
 */
export default function AIAssistantPanel({
  report = null,
  hitStats = { green: 0, yellow: 0, red: 0 },
  errorCodes = null,
  displayText,
  isTyping = false,
  actions,
  className = '',
}: AIAssistantPanelProps) {
  const [filter, setFilter] = useState<DefectFilterId>('all')

  const filteredCodes = useMemo(() => {
    const codes = errorCodes ?? []
    if (filter === 'all' || filter === 'green') {
      // 绿档：展示「无扣分」空态；有扣分时 all 展示全部，green 清空扣分列表
      if (filter === 'green') return []
      return codes
    }
    // 黄/红：按扣分严重度启发式过滤（penalty≥10 视为红档偏离，其余黄档接近）
    return codes.filter((code) => {
      const def = GOLDEN_METRIC_DEFS.find((d) => d.errorCode === code)
      const penalty = def?.penalty ?? 6
      if (filter === 'red') return penalty >= 10
      return penalty < 10
    })
  }, [errorCodes, filter])

  const bodyText = displayText ?? report?.fullText ?? ''
  const metaphorLead = report?.painPoint || report?.prescription || ''

  const filterCount = (id: DefectFilterId): number => {
    if (id === 'all') return hitStats.green + hitStats.yellow + hitStats.red
    if (id === 'pending') return 0
    return hitStats[id]
  }

  return (
    <aside
      className={`workbench-col workbench-card overflow-hidden ${className}`.trim()}
      aria-label="AI 教练助手面板"
    >
      <div className="flex h-full min-h-0 flex-col">
        <header className="flex flex-shrink-0 items-center gap-2 border-b border-slate-700/80 px-3 py-2.5">
          <span className="inline-flex flex-shrink-0 text-[var(--GREEN_OPTIMAL)]">
            <Sparkles className="h-4 w-4" />
          </span>
          <div className="min-w-0">
            <h2 className="truncate text-sm font-semibold text-slate-100">AI Assistant</h2>
            <p className="truncate text-[10px] text-slate-400">具身隐喻处方 · 缺陷筛选</p>
          </div>
        </header>

        {/* 缺陷筛选器 */}
        <div className="flex-shrink-0 border-b border-slate-700/60 px-3 py-2.5">
          <div className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold text-slate-400">
            <Filter className="h-3.5 w-3.5" />
            缺陷筛选器
          </div>
          <div className="flex flex-wrap gap-1.5">
            {FILTER_OPTIONS.map((opt) => {
              const active = filter === opt.id
              const tone =
                opt.id === 'green'
                  ? TRAFFIC_CLASS.green
                  : opt.id === 'yellow'
                    ? TRAFFIC_CLASS.yellow
                    : opt.id === 'red'
                      ? TRAFFIC_CLASS.red
                      : null
              return (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => setFilter(opt.id)}
                  className={`inline-flex items-center gap-1.5 rounded-lg border px-2 py-1 text-[11px] font-medium transition ${
                    active
                      ? tone
                        ? `${tone.border} ${tone.bg} ${tone.text} ${tone.glow}`
                        : 'border-slate-500 bg-slate-700/80 text-slate-100'
                      : 'border-slate-700 bg-slate-900/40 text-slate-400 hover:border-slate-500 hover:text-slate-200'
                  }`}
                >
                  {tone && <span className={`h-1.5 w-1.5 rounded-full ${tone.dot}`} />}
                  {opt.label}
                  <span className="tabular-nums text-[10px] opacity-70">{filterCount(opt.id)}</span>
                </button>
              )
            })}
          </div>
        </div>

        <div className="min-h-0 flex-1 space-y-3 overflow-y-auto overflow-x-hidden px-3 py-3">
          {/* 具身隐喻建议 */}
          <section className="rounded-xl border border-slate-700/70 bg-slate-900/35 p-3">
            <h3 className="mb-2 text-[11px] font-semibold text-slate-400">AIGC 具身隐喻建议</h3>
            {!report ? (
              <p className="text-xs leading-relaxed text-slate-500">
                结束分析后，DeepSeek 将在此生成具身隐喻化痛点与教练处方（例如「像拉满的弓弦再多蓄一点力」）。
              </p>
            ) : (
              <div className="space-y-2">
                {typeof report.score === 'number' && (
                  <div className="flex items-baseline gap-2">
                    <span className="text-2xl font-bold tabular-nums text-[var(--GREEN_OPTIMAL)]">
                      {report.score}
                    </span>
                    <span className="text-[10px] text-slate-500">发力稳定性评分</span>
                  </div>
                )}
                {metaphorLead && (
                  <p className="text-xs leading-relaxed text-slate-200">{metaphorLead}</p>
                )}
                {bodyText && (
                  <p className="whitespace-pre-line text-xs leading-relaxed text-slate-300">
                    {bodyText}
                    {isTyping && <span className="typewriter-caret">|</span>}
                  </p>
                )}
              </div>
            )}
          </section>

          {/* 扣分 / 缺陷清单（受筛选器驱动） */}
          <section>
            <h3 className="mb-2 text-[11px] font-semibold text-slate-400">量化扣分证据链</h3>
            {filter === 'green' && (errorCodes?.length ?? 0) > 0 ? (
              <div className="rounded-xl border border-[color-mix(in_srgb,var(--GREEN_OPTIMAL)_30%,transparent)] bg-[color-mix(in_srgb,var(--GREEN_OPTIMAL)_10%,transparent)] px-3 py-2.5 text-xs text-[var(--GREEN_OPTIMAL)]">
                当前筛选「达标」：已隐藏全部扣分项。切换至「全部 / 接近 / 偏离」查看证据链。
              </div>
            ) : (
              <DeductionList errorCodes={filteredCodes} />
            )}
            {filter !== 'all' && filter !== 'green' && filteredCodes.length === 0 && (
              <p className="mt-2 text-[10px] text-slate-500">
                当前筛选下无匹配项
                {Object.keys(ERROR_CODE_LABELS).length > 0 ? '（可切换「全部」查看完整清单）' : ''}
              </p>
            )}
          </section>

          {actions}
        </div>
      </div>
    </aside>
  )
}
