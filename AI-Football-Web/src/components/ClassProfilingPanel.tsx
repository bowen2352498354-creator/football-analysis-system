import { useEffect, useMemo, useRef, useState, type MutableRefObject, type RefObject } from 'react'
import * as echarts from 'echarts/core'
import { BarChart, RadarChart } from 'echarts/charts'
import {
  GridComponent,
  LegendComponent,
  RadarComponent,
  TooltipComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsCoreOption, EChartsType } from 'echarts/core'
import { AlertCircle, Bot, Inbox, Loader2, RefreshCcw, Sparkles } from 'lucide-react'
import { FIVE_D_DIMENSIONS, normalizeRadarScores } from './BiomechanicalRadar'
import { formatErrorCodeLabel } from './CohortComparePanel'
import type { ClassPrescriptionReport, GlobalTrainingRecord, Quantified5dScores } from '../types'

const API_BASE_URL = 'http://localhost:8000'

echarts.use([
  BarChart,
  RadarChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  RadarComponent,
  CanvasRenderer,
])

/** 历史中文标签 → ERR_*（兼容未写入 error_codes 的归档） */
const LABEL_TO_ERROR_CODE: Record<string, string> = {
  支撑脚位置偏离: 'ERR_A2_SUPPORT_WIDE',
  膝关节过度屈曲: 'ERR_KNEE_STIFF',
  随摆转髋不足: 'ERR_FOLLOW_THROUGH',
  身体重心偏移: 'ERR_TORSO_TILT',
}

const SCORE_BUCKETS = [
  { key: '<60', label: '<60', test: (s: number) => s < 60 },
  { key: '60-70', label: '60-70', test: (s: number) => s >= 60 && s < 70 },
  { key: '70-80', label: '70-80', test: (s: number) => s >= 70 && s < 80 },
  { key: '80-90', label: '80-90', test: (s: number) => s >= 80 && s < 90 },
  { key: '>90', label: '>90', test: (s: number) => s >= 90 },
] as const

function pushErrorCode(raw: unknown, seen: Set<string>, out: string[]): void {
  const text = String(raw ?? '').trim()
  if (!text || seen.has(text)) return
  if (text.startsWith('ERR_')) {
    seen.add(text)
    out.push(text)
    return
  }
  if (text === 'PASS_STANDARD') return
  const mapped = LABEL_TO_ERROR_CODE[text]
  if (mapped && !seen.has(mapped)) {
    seen.add(mapped)
    out.push(mapped)
  }
}

/** 从单条记录提取 ERR_* 列表（兼容 error_codes / biomechanicalErrors / 中文标签） */
function extractErrorCodes(record: GlobalTrainingRecord): string[] {
  const codes: string[] = []
  const seen = new Set<string>()
  const loose = record as GlobalTrainingRecord & {
    error_codes?: string[]
    errorCodes?: string[]
    biomechanical_errors?: string[]
    scoreDetail?: { error_codes?: string[]; errorCodes?: string[] }
    score_detail?: { error_codes?: string[]; errorCodes?: string[] }
  }

  for (const key of ['error_codes', 'errorCodes', 'biomechanicalErrors', 'biomechanical_errors'] as const) {
    const raw = loose[key]
    if (Array.isArray(raw)) {
      for (const item of raw) pushErrorCode(item, seen, codes)
    }
  }

  const detail = loose.scoreDetail || loose.score_detail
  if (detail && typeof detail === 'object') {
    for (const key of ['error_codes', 'errorCodes'] as const) {
      const raw = detail[key]
      if (Array.isArray(raw)) {
        for (const item of raw) pushErrorCode(item, seen, codes)
      }
    }
  }

  return codes
}

function ChartPlaceholder({ message }: { message: string }) {
  return (
    <div className="flex h-full min-h-[200px] flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-white/10 bg-black/20 px-4 text-center">
      <Inbox className="h-7 w-7 text-white/25" />
      <p className="text-sm font-medium text-white/45">{message}</p>
    </div>
  )
}

function useEChartLifecycle(
  hostRef: RefObject<HTMLDivElement | null>,
  chartRef: MutableRefObject<EChartsType | null>,
  buildOption: () => EChartsCoreOption | null,
  deps: unknown[],
) {
  useEffect(() => {
    const el = hostRef.current
    if (!el) return

    const option = buildOption()
    if (!option) return

    let chart = chartRef.current
    try {
      if (!chart || chart.isDisposed?.()) {
        chart = echarts.init(el, undefined, { renderer: 'canvas' })
        chartRef.current = chart
      }
    } catch {
      return
    }

    try {
      chart.setOption(option, { notMerge: true })
    } catch {
      /* ignore setOption failures */
    }

    const onResize = () => {
      try {
        chartRef.current?.resize()
      } catch {
        /* ignore */
      }
    }
    window.addEventListener('resize', onResize)
    // 数据切换后容器尺寸可能变化，下一帧补一次 resize
    const raf = window.requestAnimationFrame(onResize)
    return () => {
      window.cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  useEffect(() => {
    return () => {
      try {
        chartRef.current?.dispose()
      } catch {
        /* ignore */
      }
      chartRef.current = null
    }
  }, [chartRef])
}

function ClassAvgRadarChart({ averages, hasData }: { averages: number[]; hasData: boolean }) {
  const hostRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<EChartsType | null>(null)

  useEChartLifecycle(
    hostRef,
    chartRef,
    () => {
      if (!hasData) return null
      return {
        backgroundColor: 'transparent',
        animationDuration: 600,
        tooltip: {
          trigger: 'item',
          backgroundColor: 'rgba(0,0,0,0.82)',
          borderColor: 'rgba(255,255,255,0.12)',
          textStyle: { color: '#e2e8f0', fontSize: 11 },
        },
        radar: {
          indicator: FIVE_D_DIMENSIONS.map((d) => ({ name: d.short, max: 20 })),
          center: ['50%', '55%'],
          radius: '62%',
          axisName: { color: 'rgba(255,255,255,0.55)', fontSize: 11 },
          splitLine: { lineStyle: { color: 'rgba(255,255,255,0.08)' } },
          splitArea: {
            areaStyle: {
              color: ['rgba(16,185,129,0.04)', 'rgba(255,255,255,0.02)'],
            },
          },
          axisLine: { lineStyle: { color: 'rgba(255,255,255,0.12)' } },
        },
        series: [
          {
            type: 'radar',
            data: [
              {
                value: averages,
                name: '班级均值',
                lineStyle: { width: 2.2, color: '#34d399' },
                itemStyle: { color: '#34d399' },
                areaStyle: { color: 'rgba(52,211,153,0.28)' },
              },
            ],
          },
        ],
      }
    },
    [averages, hasData],
  )

  if (!hasData) {
    return <ChartPlaceholder message="当前筛选条件下暂无五维评分数据" />
  }
  return <div ref={hostRef} className="h-[240px] w-full" />
}

function CommonErrorBarChart({
  labels,
  counts,
}: {
  labels: string[]
  counts: number[]
}) {
  const hostRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<EChartsType | null>(null)
  const hasData = labels.length > 0

  useEChartLifecycle(
    hostRef,
    chartRef,
    () => {
      if (!hasData) return null
      return {
        backgroundColor: 'transparent',
        animationDuration: 600,
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'shadow' },
          backgroundColor: 'rgba(0,0,0,0.82)',
          borderColor: 'rgba(255,255,255,0.12)',
          textStyle: { color: '#e2e8f0', fontSize: 11 },
          formatter: (params: unknown) => {
            const list = Array.isArray(params) ? params : []
            if (!list.length) return ''
            const p = list[0] as { axisValue?: string; value?: number; marker?: string }
            const title = formatErrorCodeLabel(String(p.axisValue ?? ''))
            const v = typeof p.value === 'number' ? p.value : Number(p.value)
            return `<div style="font-weight:600;margin-bottom:4px">${title}</div>${p.marker ?? ''}频次：${Number.isFinite(v) ? v : '--'}`
          },
        },
        grid: { left: 128, right: 24, top: 16, bottom: 20, containLabel: false },
        xAxis: {
          type: 'value',
          minInterval: 1,
          axisLabel: { color: 'rgba(255,255,255,0.4)', fontSize: 10 },
          splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
        },
        yAxis: {
          type: 'category',
          data: labels,
          axisLabel: {
            color: 'rgba(255,255,255,0.55)',
            fontSize: 10,
            width: 110,
            overflow: 'truncate',
            formatter: (value: string) => formatErrorCodeLabel(value),
          },
        },
        series: [
          {
            type: 'bar',
            data: counts,
            barMaxWidth: 16,
            itemStyle: {
              borderRadius: [0, 6, 6, 0],
              color: 'rgba(251,113,133,0.78)',
            },
            label: {
              show: true,
              position: 'right',
              color: 'rgba(255,255,255,0.55)',
              fontSize: 10,
            },
          },
        ],
      }
    },
    [labels, counts, hasData],
  )

  if (!hasData) {
    return <ChartPlaceholder message="当前群体暂无共性错误记录" />
  }
  return <div ref={hostRef} className="h-[240px] w-full" />
}

function ScoreHistogramChart({
  labels,
  counts,
  hasScores,
}: {
  labels: string[]
  counts: number[]
  hasScores: boolean
}) {
  const hostRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<EChartsType | null>(null)

  useEChartLifecycle(
    hostRef,
    chartRef,
    () => {
      if (!hasScores) return null
      return {
        backgroundColor: 'transparent',
        animationDuration: 600,
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'shadow' },
          backgroundColor: 'rgba(0,0,0,0.82)',
          borderColor: 'rgba(255,255,255,0.12)',
          textStyle: { color: '#e2e8f0', fontSize: 11 },
          formatter: (params: unknown) => {
            const list = Array.isArray(params) ? params : []
            if (!list.length) return ''
            const p = list[0] as { axisValue?: string; value?: number; marker?: string }
            const v = typeof p.value === 'number' ? p.value : Number(p.value)
            return `${p.marker ?? ''}${p.axisValue ?? ''}：${Number.isFinite(v) ? v : '--'} 人`
          },
        },
        grid: { left: 42, right: 16, top: 20, bottom: 28 },
        xAxis: {
          type: 'category',
          data: labels,
          axisLabel: { color: 'rgba(255,255,255,0.45)', fontSize: 10 },
          axisLine: { lineStyle: { color: 'rgba(255,255,255,0.12)' } },
        },
        yAxis: {
          type: 'value',
          minInterval: 1,
          name: '人数',
          nameTextStyle: { color: 'rgba(255,255,255,0.35)', fontSize: 10 },
          axisLabel: { color: 'rgba(255,255,255,0.4)', fontSize: 10 },
          splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
        },
        series: [
          {
            type: 'bar',
            data: counts,
            barMaxWidth: 36,
            itemStyle: {
              borderRadius: [6, 6, 0, 0],
              color: 'rgba(56,189,248,0.78)',
            },
            label: {
              show: true,
              position: 'top',
              color: 'rgba(255,255,255,0.5)',
              fontSize: 10,
            },
          },
        ],
      }
    },
    [labels, counts, hasScores],
  )

  if (!hasScores) {
    return <ChartPlaceholder message="当前筛选条件下暂无有效总分" />
  }
  return <div ref={hostRef} className="h-[240px] w-full" />
}

function resolveCohortLabel(
  values: string[],
  filterValue: string | undefined,
  allToken: string,
  mixedFallback: string,
): string {
  if (filterValue && filterValue !== 'all') return filterValue
  const unique = Array.from(new Set(values.map((v) => v.trim()).filter(Boolean)))
  if (unique.length === 0) return allToken
  if (unique.length === 1) return unique[0]
  return mixedFallback
}

export interface ClassProfilingPanelProps {
  filteredDataset: GlobalTrainingRecord[]
  /** 全局过滤器当前学校（all = 未限定） */
  filterSchool?: string
  /** 全局过滤器当前班级（all = 未限定） */
  filterClass?: string
  className?: string
}

/**
 * 班级群体画像：五维均值雷达 + 共性错误 Top5 + 成绩分布 + AI 宏观备课卡
 */
export default function ClassProfilingPanel({
  filteredDataset,
  filterSchool = 'all',
  filterClass = 'all',
  className = '',
}: ClassProfilingPanelProps) {
  const profile = useMemo(() => {
    const dimSums = FIVE_D_DIMENSIONS.map(() => 0)
    const dimCounts = FIVE_D_DIMENSIONS.map(() => 0)
    const errorCounter = new Map<string, number>()
    const bucketCounts = SCORE_BUCKETS.map(() => 0)
    let scoreSum = 0
    let scoreSamples = 0
    const schools: string[] = []
    const classes: string[] = []

    for (const record of filteredDataset) {
      schools.push((record.school || '').trim() || '未设置学校')
      classes.push((record.classGroup || '').trim() || '未设置班级')

      const raw = record.quantified5dScores as Quantified5dScores | null | undefined
      if (raw && typeof raw === 'object') {
        const scores = normalizeRadarScores(raw)
        FIVE_D_DIMENSIONS.forEach((d, i) => {
          const r = raw
          const present =
            typeof r[d.key] === 'number' ||
            (d.key === 'approach_rhythm' && typeof r.approach_score === 'number') ||
            (d.key === 'support_stability' && typeof r.support_score === 'number') ||
            (d.key === 'backswing_folding' && typeof r.backswing_score === 'number') ||
            (d.key === 'ankle_rigidity' && typeof r.ankle_rigidity_score === 'number') ||
            (d.key === 'whipping_velocity' && typeof r.whipping_score === 'number')
          if (!present) return
          dimSums[i] += scores[d.key]
          dimCounts[i] += 1
        })
      }

      for (const code of extractErrorCodes(record)) {
        if (code === 'PASS_STANDARD') continue
        errorCounter.set(code, (errorCounter.get(code) || 0) + 1)
      }

      if (typeof record.score === 'number' && Number.isFinite(record.score)) {
        scoreSum += record.score
        scoreSamples += 1
        const idx = SCORE_BUCKETS.findIndex((b) => b.test(record.score as number))
        if (idx >= 0) bucketCounts[idx] += 1
      }
    }

    const averages = dimSums.map((sum, i) =>
      dimCounts[i] > 0 ? Math.round((sum / dimCounts[i]) * 10) / 10 : 0,
    )
    const hasRadar = dimCounts.some((c) => c > 0)
    const totalRecords = filteredDataset.length

    const topErrors = Array.from(errorCounter.entries())
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, 5)
    // 横向条形图：category 轴自下而上，反转使 Top1 在最上
    const topLabels = topErrors.map(([code]) => code).reverse()
    const topCounts = topErrors.map(([, count]) => count).reverse()

    // LLM 接口要求：中文错误标签 → 出现率百分比（0-100）
    const errorStats: Record<string, number> = {}
    for (const [code, count] of errorCounter.entries()) {
      const label = formatErrorCodeLabel(code)
      const pct = totalRecords > 0 ? Math.round((count / totalRecords) * 1000) / 10 : 0
      errorStats[label] = Math.max(errorStats[label] || 0, pct)
    }

    return {
      averages,
      hasRadar,
      topLabels,
      topCounts,
      bucketLabels: SCORE_BUCKETS.map((b) => b.label),
      bucketCounts,
      hasScores: scoreSamples > 0,
      sampleCount: totalRecords,
      avgScore: scoreSamples > 0 ? Math.round(scoreSum / scoreSamples) : null,
      errorStats,
      schoolLabel: resolveCohortLabel(schools, filterSchool, '全部学校', '多校混合'),
      classLabel: resolveCohortLabel(classes, filterClass, '全部班级', '多班混合'),
    }
  }, [filteredDataset, filterSchool, filterClass])

  if (filteredDataset.length === 0) {
    return (
      <div className={`rounded-2xl border border-dashed border-white/10 bg-black/20 p-8 ${className}`}>
        <ChartPlaceholder message="该筛选条件下暂无有效测试数据，请调整全局过滤器" />
      </div>
    )
  }

  return (
    <section className={`class-profiling-panel space-y-3 ${className}`}>
      <div className="flex flex-wrap items-center justify-between gap-2 px-0.5">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-white/85">
          <Sparkles className="h-4 w-4 text-violet-300" />
          班级群体画像
        </h3>
        <span className="rounded-full bg-black/30 px-2.5 py-0.5 text-[10px] text-white/35">
          有效样本 {profile.sampleCount} 条
        </span>
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        {/* A：五维能力整体画像 */}
        <article className="rounded-2xl border border-white/10 bg-slate-950/50 p-3">
          <h4 className="mb-1 text-xs font-semibold tracking-wide text-emerald-300/90">
            五维能力整体画像
          </h4>
          <p className="mb-2 text-[10px] text-white/30">当前过滤群体在五维上的平均分</p>
          <ClassAvgRadarChart averages={profile.averages} hasData={profile.hasRadar} />
        </article>

        {/* B：共性错误 Top 5 */}
        <article className="rounded-2xl border border-white/10 bg-slate-950/50 p-3">
          <h4 className="mb-1 text-xs font-semibold tracking-wide text-rose-300/90">
            班级共性错误频次 Top 5
          </h4>
          <p className="mb-2 text-[10px] text-white/30">按 ERR_* 出现总频次降序</p>
          <CommonErrorBarChart labels={profile.topLabels} counts={profile.topCounts} />
        </article>

        {/* C：成绩离散分布 */}
        <article className="rounded-2xl border border-white/10 bg-slate-950/50 p-3">
          <h4 className="mb-1 text-xs font-semibold tracking-wide text-sky-300/90">
            成绩离散分布区间
          </h4>
          <p className="mb-2 text-[10px] text-white/30">总分分桶人数统计</p>
          <ScoreHistogramChart
            labels={profile.bucketLabels}
            counts={profile.bucketCounts}
            hasScores={profile.hasScores}
          />
        </article>

        {/* D：AI 宏观教学建议（接通 /api/generate_class_prescription） */}
        <MacroTeachingAdviceCard
          school={profile.schoolLabel}
          classGroup={profile.classLabel}
          errorStats={profile.errorStats}
          totalRecords={profile.sampleCount}
          avgScore={profile.avgScore}
        />
      </div>
    </section>
  )
}

type AdviceStatus = 'idle' | 'loading' | 'ready' | 'error'

function MacroTeachingAdviceCard({
  school,
  classGroup,
  errorStats,
  totalRecords,
  avgScore,
}: {
  school: string
  classGroup: string
  errorStats: Record<string, number>
  totalRecords: number
  avgScore: number | null
}) {
  const [status, setStatus] = useState<AdviceStatus>('idle')
  const [report, setReport] = useState<ClassPrescriptionReport | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [refreshNonce, setRefreshNonce] = useState(0)

  const requestKey = useMemo(
    () =>
      JSON.stringify({
        school,
        classGroup,
        errorStats,
        totalRecords,
        avgScore,
      }),
    [school, classGroup, errorStats, totalRecords, avgScore],
  )

  useEffect(() => {
    if (totalRecords <= 0) {
      setStatus('idle')
      setReport(null)
      setErrorMessage(null)
      return
    }

    const controller = new AbortController()
    let cancelled = false

    setStatus('loading')
    setErrorMessage(null)

    void (async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/generate_class_prescription`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: controller.signal,
          body: JSON.stringify({
            school,
            classGroup,
            errorStats,
            totalRecords,
            avgScore,
          }),
        })
        if (!response.ok) {
          throw new Error(`接口返回状态码 ${response.status}`)
        }
        const data = (await response.json()) as ClassPrescriptionReport
        if (cancelled) return
        if (!data?.diagnosis && !data?.prescription) {
          throw new Error('后端未返回有效诊断内容')
        }
        setReport({
          diagnosis: data.diagnosis || '',
          prescription: data.prescription || '',
          fullText: data.fullText || `${data.diagnosis || ''}\n\n${data.prescription || ''}`,
          generatedAt: data.generatedAt || '',
        })
        setStatus('ready')
      } catch (error) {
        if (cancelled || (error instanceof DOMException && error.name === 'AbortError')) return
        setReport(null)
        setErrorMessage(error instanceof Error ? error.message : '生成失败，请稍后重试')
        setStatus('error')
      }
    })()

    return () => {
      cancelled = true
      controller.abort()
    }
    // requestKey 已序列化 school/classGroup/errorStats/totalRecords/avgScore
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestKey, refreshNonce])

  return (
    <article className="class-profiling-ai-card relative overflow-hidden rounded-2xl p-[1px]">
      <div className="relative flex h-full min-h-[280px] flex-col rounded-[15px] bg-slate-950/90 p-4">
        <div className="mb-3 flex items-start justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500/30 to-cyan-400/20 ring-1 ring-white/10">
              <Bot className="h-4 w-4 text-violet-200" />
            </span>
            <div>
              <h4 className="text-xs font-semibold tracking-wide text-violet-100">
                ✨ AI 宏观教学建议
              </h4>
              <p className="text-[10px] text-white/30">
                {status === 'loading'
                  ? '智能备课 · DeepSeek 生成中…'
                  : status === 'ready'
                    ? `智能备课 · ${report?.generatedAt || '已生成'}`
                    : '智能备课 · 班级共性错题宏观处方'}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setRefreshNonce((n) => n + 1)}
            disabled={status === 'loading' || totalRecords <= 0}
            className="inline-flex items-center gap-1 rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[10px] text-white/55 transition hover:bg-white/10 hover:text-white/80 disabled:opacity-40"
            title="重新生成宏观处方"
          >
            {status === 'loading' ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <RefreshCcw className="h-3 w-3" />
            )}
            重新生成
          </button>
        </div>

        <div className="flex min-h-0 flex-1 flex-col overflow-y-auto rounded-xl border border-violet-400/20 bg-violet-500/5 px-4 py-4 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
          {status === 'loading' && (
            <div className="flex flex-1 flex-col justify-center gap-3 py-4">
              <div className="flex items-center gap-2 text-violet-200/80">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-xs font-medium">分析中</span>
              </div>
              <p className="text-sm leading-relaxed text-white/45">
                正在分析班级共性错题，提取教学优化策略...
              </p>
              <p className="text-[11px] leading-relaxed text-white/25">
                {school} · {classGroup} · {totalRecords} 条样本
                {avgScore != null ? ` · 均分 ${avgScore}` : ''}
              </p>
            </div>
          )}

          {status === 'error' && (
            <div className="flex flex-1 flex-col justify-center gap-3 py-4">
              <div className="flex items-center gap-2 text-rose-300/90">
                <AlertCircle className="h-4 w-4" />
                <span className="text-xs font-medium">生成失败</span>
              </div>
              <p className="text-sm leading-relaxed text-white/55">
                {errorMessage || '无法连接宏观处方接口，请确认后端服务已启动。'}
              </p>
              <button
                type="button"
                onClick={() => setRefreshNonce((n) => n + 1)}
                className="inline-flex w-fit items-center gap-1.5 rounded-full border border-rose-400/30 bg-rose-500/10 px-3 py-1.5 text-[11px] font-medium text-rose-100 transition hover:bg-rose-500/20"
              >
                <RefreshCcw className="h-3 w-3" />
                重试
              </button>
            </div>
          )}

          {status === 'ready' && report && (
            <div className="flex flex-col gap-4">
              <section>
                <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-violet-300/70">
                  集体诊断
                </p>
                <p className="text-sm leading-relaxed text-white/80 whitespace-pre-wrap">
                  {report.diagnosis}
                </p>
              </section>
              <section className="border-t border-white/8 pt-3">
                <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-emerald-300/70">
                  教学处方
                </p>
                <p className="text-sm leading-relaxed text-white/80 whitespace-pre-wrap">
                  {report.prescription}
                </p>
              </section>
            </div>
          )}

          {status === 'idle' && (
            <div className="flex flex-1 flex-col items-center justify-center gap-2 py-6 text-center">
              <Inbox className="h-6 w-6 text-white/25" />
              <p className="text-sm text-white/40">暂无样本，无法生成宏观处方</p>
            </div>
          )}
        </div>
      </div>
    </article>
  )
}
