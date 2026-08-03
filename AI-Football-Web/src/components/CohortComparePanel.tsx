import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent as ReactMouseEvent,
} from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import * as echarts from 'echarts/core'
import { BarChart, LineChart, RadarChart } from 'echarts/charts'
import {
  GridComponent,
  LegendComponent,
  RadarComponent,
  TooltipComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsType } from 'echarts/core'
import {
  Check,
  ChevronDown,
  GitCompareArrows,
  Inbox,
  Loader2,
  Plus,
  X,
} from 'lucide-react'
import {
  loadCohortCompareOptions,
  removeCohortCompareOption,
  saveCustomCohortCompareName,
} from '../mockData'
import type {
  CohortCompareResponse,
  CohortErrorRateRow,
  CohortTrendPoint,
  RadarAverageScores,
} from '../types'

echarts.use([
  LineChart,
  BarChart,
  RadarChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  RadarComponent,
  CanvasRenderer,
])

const API_BASE_URL = 'http://localhost:8000'

/** 典型错误码 → 中文展示名（仅影响对比看板展示层；班级画像面板复用） */
export const ERROR_CODE_MAP: Record<string, string> = {
  ERR_TORSO_TILT: '躯干过度倾斜',
  ERR_KNEE_STIFF: '摆动腿膝盖僵硬',
  ERR_FOLLOW_THROUGH: '随前动作缺失',
  ERR_C1_LOOSE_ANKLE: '脚踝未锁死 (松弛)',
  ERR_B1_STRAIGHT_LEG: '直腿击球 (缺乏折叠)',
  ERR_A2_SUPPORT_WIDE: '支撑脚偏远 (距离过宽)',
  PASS_STANDARD: '动作达标无显著错误',
}

export function formatErrorCodeLabel(code: string): string {
  return ERROR_CODE_MAP[code] || code
}

const RADAR_KEYS = [
  'approach_rhythm',
  'support_stability',
  'backswing_folding',
  'ankle_rigidity',
  'whipping_velocity',
] as const

const RADAR_LABELS = ['助跑', '支撑', '后摆', '踝锁', '鞭打'] as const

function alignTrendSeries(
  dates: string[],
  points: CohortTrendPoint[] | undefined,
): { avg: (number | null)[]; upper: (number | null)[]; lower: (number | null)[] } {
  const map = new Map<string, CohortTrendPoint>()
  for (const p of points || []) {
    if (p?.date) map.set(p.date, p)
  }
  const avg: (number | null)[] = []
  const upper: (number | null)[] = []
  const lower: (number | null)[] = []
  for (const day of dates) {
    const hit = map.get(day)
    if (!hit || typeof hit.average_score !== 'number' || !Number.isFinite(hit.average_score)) {
      avg.push(null)
      upper.push(null)
      lower.push(null)
      continue
    }
    const mean = hit.average_score
    const variance =
      typeof hit.score_variance === 'number' && Number.isFinite(hit.score_variance)
        ? Math.max(0, hit.score_variance)
        : 0
    const band = Math.sqrt(variance)
    avg.push(mean)
    upper.push(Math.min(100, mean + band))
    lower.push(Math.max(0, mean - band))
  }
  return { avg, upper, lower }
}

function toRadarScores(
  values: Array<number | null | undefined> | undefined,
  keyed?: RadarAverageScores | null,
): RadarAverageScores | null {
  if (keyed && typeof keyed === 'object') {
    const has = RADAR_KEYS.some((k) => typeof keyed[k] === 'number' && Number.isFinite(keyed[k]!))
    if (has) return keyed
  }
  if (!Array.isArray(values) || values.length < 5) return null
  const out: RadarAverageScores = {}
  let hit = 0
  RADAR_KEYS.forEach((key, i) => {
    const v = values[i]
    if (typeof v === 'number' && Number.isFinite(v)) {
      out[key] = v
      hit += 1
    }
  })
  return hit > 0 ? out : null
}

function rateMap(rows: CohortErrorRateRow[] | undefined): Map<string, number> {
  const map = new Map<string, number>()
  for (const row of rows || []) {
    if (!row?.code) continue
    const pct =
      typeof row.percentage === 'number' && Number.isFinite(row.percentage)
        ? row.percentage
        : typeof row.rate === 'number' && Number.isFinite(row.rate)
          ? row.rate * 100
          : 0
    map.set(row.code, pct)
  }
  return map
}

function ChartPlaceholder({ message }: { message: string }) {
  return (
    <div className="flex h-full min-h-[200px] flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-white/10 bg-black/20 px-4 text-center">
      <Inbox className="h-7 w-7 text-white/25" />
      <p className="text-sm font-medium text-white/45">{message || '暂无足够数据对比'}</p>
    </div>
  )
}

function DualTrendChart({
  dates,
  seriesA,
  seriesB,
  labelA,
  labelB,
}: {
  dates: string[]
  seriesA: ReturnType<typeof alignTrendSeries>
  seriesB: ReturnType<typeof alignTrendSeries>
  labelA: string
  labelB: string
}) {
  const hostRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<EChartsType | null>(null)

  useEffect(() => {
    const el = hostRef.current
    if (!el || dates.length === 0) return

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
      chart.setOption(
        {
          backgroundColor: 'transparent',
          animationDuration: 700,
          tooltip: {
            trigger: 'axis',
            backgroundColor: 'rgba(0,0,0,0.82)',
            borderColor: 'rgba(255,255,255,0.12)',
            textStyle: { color: '#e2e8f0', fontSize: 11 },
          },
          legend: {
            top: 0,
            textStyle: { color: 'rgba(255,255,255,0.5)', fontSize: 10 },
            data: [labelA, labelB],
          },
          grid: { left: 42, right: 18, top: 36, bottom: 28 },
          xAxis: {
            type: 'category',
            data: dates,
            boundaryGap: false,
            axisLabel: { color: 'rgba(255,255,255,0.4)', fontSize: 10 },
            axisLine: { lineStyle: { color: 'rgba(255,255,255,0.12)' } },
          },
          yAxis: {
            type: 'value',
            min: 0,
            max: 100,
            name: '均分',
            nameTextStyle: { color: 'rgba(255,255,255,0.35)', fontSize: 10 },
            axisLabel: { color: 'rgba(255,255,255,0.4)', fontSize: 10 },
            splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
          },
          series: [
            // 波动带：stack = 下界 + (上界−下界) 半透明填充（由 score_variance 开方）
            {
              name: `${labelA}·下界`,
              type: 'line',
              data: seriesA.lower,
              lineStyle: { opacity: 0 },
              symbol: 'none',
              stack: 'bandA',
              tooltip: { show: false },
              silent: true,
            },
            {
              name: `${labelA}·波动`,
              type: 'line',
              data: seriesA.lower.map((lo, i) => {
                const hi = seriesA.upper[i]
                if (lo == null || hi == null) return null
                return Math.max(0, hi - lo)
              }),
              lineStyle: { opacity: 0 },
              symbol: 'none',
              stack: 'bandA',
              areaStyle: { color: 'rgba(16,185,129,0.18)' },
              tooltip: { show: false },
              silent: true,
            },
            {
              name: `${labelB}·下界`,
              type: 'line',
              data: seriesB.lower,
              lineStyle: { opacity: 0 },
              symbol: 'none',
              stack: 'bandB',
              tooltip: { show: false },
              silent: true,
            },
            {
              name: `${labelB}·波动`,
              type: 'line',
              data: seriesB.lower.map((lo, i) => {
                const hi = seriesB.upper[i]
                if (lo == null || hi == null) return null
                return Math.max(0, hi - lo)
              }),
              lineStyle: { opacity: 0 },
              symbol: 'none',
              stack: 'bandB',
              areaStyle: { color: 'rgba(251,191,36,0.14)' },
              tooltip: { show: false },
              silent: true,
            },
            {
              name: labelA,
              type: 'line',
              data: seriesA.avg,
              smooth: true,
              showSymbol: true,
              symbolSize: 6,
              z: 3,
              lineStyle: { width: 2.2, color: '#34d399' },
              itemStyle: { color: '#34d399' },
            },
            {
              name: labelB,
              type: 'line',
              data: seriesB.avg,
              smooth: true,
              showSymbol: true,
              symbolSize: 6,
              z: 3,
              lineStyle: { width: 2.2, color: '#fbbf24' },
              itemStyle: { color: '#fbbf24' },
            },
          ],
        },
        { notMerge: true },
      )
    } catch {
      /* ignore chart setOption failures */
    }

    const onResize = () => {
      try {
        chartRef.current?.resize()
      } catch {
        /* ignore */
      }
    }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [dates, seriesA, seriesB, labelA, labelB])

  useEffect(() => {
    return () => {
      try {
        chartRef.current?.dispose()
      } catch {
        /* ignore */
      }
      chartRef.current = null
    }
  }, [])

  return <div ref={hostRef} className="h-[240px] w-full" />
}

function OverlayRadarChart({
  scoresA,
  scoresB,
  labelA,
  labelB,
}: {
  scoresA: RadarAverageScores
  scoresB: RadarAverageScores
  labelA: string
  labelB: string
}) {
  const hostRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<EChartsType | null>(null)

  useEffect(() => {
    const el = hostRef.current
    if (!el) return

    let chart = chartRef.current
    try {
      if (!chart || chart.isDisposed?.()) {
        chart = echarts.init(el, undefined, { renderer: 'canvas' })
        chartRef.current = chart
      }
    } catch {
      return
    }

    const valuesA = RADAR_KEYS.map((k) => {
      const v = scoresA[k]
      return typeof v === 'number' && Number.isFinite(v) ? Math.max(0, Math.min(20, v)) : 0
    })
    const valuesB = RADAR_KEYS.map((k) => {
      const v = scoresB[k]
      return typeof v === 'number' && Number.isFinite(v) ? Math.max(0, Math.min(20, v)) : 0
    })

    try {
      chart.setOption(
        {
          backgroundColor: 'transparent',
          animationDuration: 900,
          legend: {
            bottom: 0,
            textStyle: { color: 'rgba(255,255,255,0.45)', fontSize: 10 },
            data: [labelA, labelB],
          },
          tooltip: {
            trigger: 'item',
            backgroundColor: 'rgba(0,0,0,0.82)',
            borderColor: 'rgba(255,255,255,0.12)',
            textStyle: { color: '#e2e8f0', fontSize: 11 },
          },
          radar: {
            indicator: RADAR_LABELS.map((name) => ({ name, max: 20 })),
            center: ['50%', '46%'],
            radius: '64%',
            axisName: { color: 'rgba(226,232,240,0.7)', fontSize: 11 },
            splitLine: { lineStyle: { color: 'rgba(71,85,105,0.55)' } },
            axisLine: { lineStyle: { color: 'rgba(100,116,139,0.45)' } },
            splitArea: {
              areaStyle: {
                color: ['rgba(15,23,42,0.15)', 'rgba(30,41,59,0.35)'],
              },
            },
          },
          series: [
            {
              type: 'radar',
              data: [
                {
                  value: valuesA,
                  name: labelA,
                  lineStyle: { width: 2, color: 'rgba(52,211,153,0.95)' },
                  itemStyle: { color: '#34d399' },
                  areaStyle: { color: 'rgba(52,211,153,0.32)' },
                },
                {
                  value: valuesB,
                  name: labelB,
                  lineStyle: { width: 2, color: 'rgba(251,191,36,0.95)' },
                  itemStyle: { color: '#fbbf24' },
                  areaStyle: { color: 'rgba(251,191,36,0.22)' },
                },
              ],
            },
          ],
        },
        { notMerge: true },
      )
    } catch {
      /* ignore */
    }

    const onResize = () => {
      try {
        chartRef.current?.resize()
      } catch {
        /* ignore */
      }
    }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [scoresA, scoresB, labelA, labelB])

  useEffect(() => {
    return () => {
      try {
        chartRef.current?.dispose()
      } catch {
        /* ignore */
      }
      chartRef.current = null
    }
  }, [])

  return <div ref={hostRef} className="h-[240px] w-full" />
}

function ErrorRateBars({
  codes,
  ratesA,
  ratesB,
  labelA,
  labelB,
}: {
  codes: string[]
  ratesA: Map<string, number>
  ratesB: Map<string, number>
  labelA: string
  labelB: string
}) {
  const hostRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<EChartsType | null>(null)

  useEffect(() => {
    const el = hostRef.current
    if (!el || codes.length === 0) return

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
      chart.setOption(
        {
          backgroundColor: 'transparent',
          tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'shadow' },
            backgroundColor: 'rgba(0,0,0,0.82)',
            borderColor: 'rgba(255,255,255,0.12)',
            textStyle: { color: '#e2e8f0', fontSize: 11 },
            formatter: (params: unknown) => {
              const list = Array.isArray(params) ? params : []
              if (!list.length) return ''
              const axisValue = String((list[0] as { axisValue?: string }).axisValue ?? '')
              const title = formatErrorCodeLabel(axisValue)
              const rows = list
                .map((item) => {
                  const p = item as { marker?: string; seriesName?: string; value?: number }
                  const v = typeof p.value === 'number' ? p.value : Number(p.value)
                  return `${p.marker ?? ''}${p.seriesName ?? ''}：${Number.isFinite(v) ? v.toFixed(1) : '--'}%`
                })
                .join('<br/>')
              return `<div style="font-weight:600;margin-bottom:4px">${title}</div>${rows}`
            },
          },
          legend: {
            top: 0,
            textStyle: { color: 'rgba(255,255,255,0.5)', fontSize: 10 },
          },
          // 中文标签较长：加宽左侧留白，避免被裁切
          grid: { left: 168, right: 18, top: 32, bottom: 24, containLabel: false },
          xAxis: {
            type: 'value',
            max: 100,
            axisLabel: {
              color: 'rgba(255,255,255,0.4)',
              fontSize: 10,
              formatter: '{value}%',
            },
            splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
          },
          yAxis: {
            type: 'category',
            data: codes,
            axisLabel: {
              color: 'rgba(255,255,255,0.55)',
              fontSize: 10,
              width: 150,
              overflow: 'truncate',
              formatter: (value: string) => formatErrorCodeLabel(value),
            },
          },
          series: [
            {
              name: labelA,
              type: 'bar',
              data: codes.map((c) => ratesA.get(c) ?? 0),
              itemStyle: { color: 'rgba(52,211,153,0.75)' },
              barMaxWidth: 14,
            },
            {
              name: labelB,
              type: 'bar',
              data: codes.map((c) => ratesB.get(c) ?? 0),
              itemStyle: { color: 'rgba(251,191,36,0.75)' },
              barMaxWidth: 14,
            },
          ],
        },
        { notMerge: true },
      )
    } catch {
      /* ignore */
    }

    const onResize = () => {
      try {
        chartRef.current?.resize()
      } catch {
        /* ignore */
      }
    }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [codes, ratesA, ratesB, labelA, labelB])

  useEffect(() => {
    return () => {
      try {
        chartRef.current?.dispose()
      } catch {
        /* ignore */
      }
      chartRef.current = null
    }
  }, [])

  return <div ref={hostRef} className="h-[240px] w-full" />
}

export interface CohortComparePanelProps {
  /** 可选班级/实验组名称列表（来自全局过滤器穿透后的数据集） */
  cohortOptions: string[]
  /** 全局过滤后的唯一数据源；对比菜单选项与空态与此联动 */
  filteredDataset?: Array<{ classGroup?: string | null }>
  className?: string
}

/** 对比下拉：支持选中 + 右侧删除幽灵班级选项 */
function CohortOptionPicker({
  label,
  accent,
  placeholder,
  value,
  options,
  disabledOption,
  onSelect,
  onDelete,
}: {
  label: string
  accent: 'emerald' | 'amber'
  placeholder: string
  value: string
  options: string[]
  disabledOption: string
  onSelect: (value: string) => void
  onDelete: (name: string, event: ReactMouseEvent) => void
}) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const accentText = accent === 'emerald' ? 'text-emerald-400/80' : 'text-amber-400/80'
  const selectedText = accent === 'emerald' ? 'text-emerald-300' : 'text-amber-300'

  useEffect(() => {
    if (!open) return
    function handleOutside(event: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleOutside)
    return () => document.removeEventListener('mousedown', handleOutside)
  }, [open])

  return (
    <div
      ref={rootRef}
      className="relative flex min-w-[180px] flex-1 items-center gap-2 rounded-2xl bg-black/25 px-3 py-1.5 text-xs text-white/50 sm:max-w-[300px]"
    >
      <span className={`shrink-0 ${accentText}`}>{label}</span>
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className="flex min-w-0 flex-1 items-center justify-between gap-1 bg-transparent text-left text-sm font-medium text-white outline-none"
      >
        <span className={`truncate ${value ? 'text-white' : 'text-white/35'}`}>
          {value || placeholder}
        </span>
        <ChevronDown
          className={`h-3.5 w-3.5 flex-shrink-0 text-white/35 transition-transform ${open ? 'rotate-180' : ''}`}
        />
      </button>

      <AnimatePresence>
        {open && (
          <motion.ul
            initial={{ opacity: 0, y: -6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.98 }}
            transition={{ type: 'spring', stiffness: 420, damping: 32 }}
            className="absolute left-0 right-0 top-[calc(100%+6px)] z-30 max-h-52 overflow-y-auto rounded-2xl border border-white/10 bg-zinc-900/95 py-1 shadow-xl backdrop-blur-xl"
            role="listbox"
          >
            <li
              role="option"
              aria-selected={!value}
              className="cursor-pointer px-3 py-2 text-sm text-white/40 transition hover:bg-white/10"
              onClick={() => {
                onSelect('')
                setOpen(false)
              }}
            >
              {placeholder}
            </li>
            {options.map((option) => {
              const disabled = Boolean(disabledOption) && option === disabledOption
              const selected = option === value
              return (
                <li
                  key={option}
                  role="option"
                  aria-selected={selected}
                  aria-disabled={disabled}
                  className={`group flex items-center justify-between gap-2 px-3 py-2 text-sm transition ${
                    disabled
                      ? 'cursor-not-allowed text-white/25'
                      : `cursor-pointer hover:bg-white/10 ${selected ? selectedText : 'text-white/85'}`
                  }`}
                  onClick={() => {
                    if (disabled) return
                    onSelect(option)
                    setOpen(false)
                  }}
                >
                  <span className="min-w-0 truncate">{option}</span>
                  <button
                    type="button"
                    title="删除此班级选项"
                    aria-label={`删除 ${option}`}
                    onClick={(e) => onDelete(option, e)}
                    className="inline-flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-md text-white/35 transition hover:bg-rose-500/20 hover:text-rose-300"
                  >
                    <X className="h-3.5 w-3.5" strokeWidth={2.25} />
                  </button>
                </li>
              )
            })}
            {options.length === 0 && (
              <li className="px-3 py-2 text-xs text-white/35">暂无可选项，请先新增班级</li>
            )}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  )
}

/**
 * 教练端「班级/实验组对比」科研分析模块：
 * 双下拉选择（支持新增 / 删除选项）→ 双轨趋势折线 + 叠影雷达 + 错误分布率。
 */
export default function CohortComparePanel({
  cohortOptions,
  filteredDataset,
  className = '',
}: CohortComparePanelProps) {
  const liveOptions = useMemo(() => {
    // 优先消费全局过滤器穿透后的数据集，保证对比雷达与看板同源
    if (Array.isArray(filteredDataset)) {
      return Array.from(
        new Set(filteredDataset.map((r) => (r.classGroup || '').trim()).filter(Boolean)),
      )
    }
    return Array.from(new Set(cohortOptions.map((n) => n.trim()).filter(Boolean)))
  }, [cohortOptions, filteredDataset])
  const [options, setOptions] = useState<string[]>(() => loadCohortCompareOptions(liveOptions))
  const [cohortA, setCohortA] = useState('')
  const [cohortB, setCohortB] = useState('')
  const [loading, setLoading] = useState(false)
  const [payload, setPayload] = useState<CohortCompareResponse | null>(null)
  const [isAdding, setIsAdding] = useState(false)
  const [draftClass, setDraftClass] = useState('')

  // 归档数据变化时，重新合并「活跃班级 + 自定义 − 已隐藏」
  useEffect(() => {
    setOptions(loadCohortCompareOptions(liveOptions))
  }, [liveOptions])

  useEffect(() => {
    if (cohortA && !options.includes(cohortA)) setCohortA('')
    if (cohortB && !options.includes(cohortB)) setCohortB('')
  }, [options, cohortA, cohortB])

  function handleAddClass() {
    const trimmed = draftClass.trim()
    if (!trimmed) return
    const next = saveCustomCohortCompareName(trimmed, liveOptions)
    setOptions(next)
    if (!cohortA) setCohortA(trimmed)
    else if (!cohortB && trimmed !== cohortA) setCohortB(trimmed)
    setDraftClass('')
    setIsAdding(false)
  }

  function handleDeleteClass(name: string, event: ReactMouseEvent) {
    event.preventDefault()
    event.stopPropagation()
    const next = removeCohortCompareOption(name, liveOptions)
    setOptions(next)
    if (cohortA === name) setCohortA('')
    if (cohortB === name) setCohortB('')
  }

  function handleDraftKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === 'Enter') {
      event.preventDefault()
      handleAddClass()
    } else if (event.key === 'Escape') {
      setIsAdding(false)
      setDraftClass('')
    }
  }

  useEffect(() => {
    if (!cohortA || !cohortB || cohortA === cohortB) {
      setPayload(null)
      setLoading(false)
      return
    }

    let cancelled = false
    const controller = new AbortController()

    async function run() {
      setLoading(true)
      try {
        const params = new URLSearchParams({ cohort_a: cohortA, cohort_b: cohortB })
        const response = await fetch(
          `${API_BASE_URL}/api/analytics/compare_cohorts?${params.toString()}`,
          { signal: controller.signal },
        )
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const data = (await response.json()) as CohortCompareResponse
        if (!cancelled) setPayload(data)
      } catch (err) {
        if (cancelled || (err instanceof DOMException && err.name === 'AbortError')) return
        if (!cancelled) {
          setPayload({
            success: false,
            sufficient_data: false,
            message: '暂无足够数据对比',
            cohort_a: cohortA,
            cohort_b: cohortB,
            sample_counts: { a: 0, b: 0 },
            trend: { dates: [], cohort_a: [], cohort_b: [] },
            radar: {
              dimensions: [...RADAR_LABELS],
              keys: [...RADAR_KEYS],
              cohort_a: [null, null, null, null, null],
              cohort_b: [null, null, null, null, null],
            },
            error_rates: { cohort_a: [], cohort_b: [], union_codes: [] },
          })
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void run()
    return () => {
      cancelled = true
      controller.abort()
    }
  }, [cohortA, cohortB])

  const ready = Boolean(cohortA && cohortB && cohortA !== cohortB)
  const sufficient = Boolean(payload?.sufficient_data)
  const placeholderMsg = payload?.message || '暂无足够数据对比'
  const filterEmpty = Array.isArray(filteredDataset) && filteredDataset.length === 0

  const dates = payload?.trend?.dates || []
  const seriesA = useMemo(
    () => alignTrendSeries(dates, payload?.trend?.cohort_a),
    [dates, payload?.trend?.cohort_a],
  )
  const seriesB = useMemo(
    () => alignTrendSeries(dates, payload?.trend?.cohort_b),
    [dates, payload?.trend?.cohort_b],
  )

  const radarA = toRadarScores(payload?.radar?.cohort_a, payload?.radar?.cohort_a_scores)
  const radarB = toRadarScores(payload?.radar?.cohort_b, payload?.radar?.cohort_b_scores)
  const hasRadar = Boolean(radarA && radarB)

  const errorCodes = useMemo(() => {
    const union = payload?.error_rates?.union_codes
    if (Array.isArray(union) && union.length > 0) return union.slice(0, 8)
    const merged = new Set<string>()
    for (const row of payload?.error_rates?.cohort_a || []) if (row.code) merged.add(row.code)
    for (const row of payload?.error_rates?.cohort_b || []) if (row.code) merged.add(row.code)
    return Array.from(merged).slice(0, 8)
  }, [payload])

  const ratesA = useMemo(() => rateMap(payload?.error_rates?.cohort_a), [payload])
  const ratesB = useMemo(() => rateMap(payload?.error_rates?.cohort_b), [payload])

  const shortA = cohortA || '班级 1'
  const shortB = cohortB || '班级 2'

  return (
    <section
      className={`rounded-2xl border border-white/10 bg-slate-900/40 px-3 py-3 sm:px-4 ${className}`.trim()}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-emerald-300/80">
          <GitCompareArrows className="h-3.5 w-3.5" />
          班级/实验组对比
        </span>

        <CohortOptionPicker
          label="对比班级 1"
          accent="emerald"
          placeholder="选择对比班级 1"
          value={cohortA}
          options={options}
          disabledOption={cohortB}
          onSelect={setCohortA}
          onDelete={handleDeleteClass}
        />

        <span className="rounded-full bg-white/10 px-2 py-0.5 text-[10px] font-bold tracking-wider text-white/50">
          VS
        </span>

        <CohortOptionPicker
          label="对比班级 2"
          accent="amber"
          placeholder="选择对比班级 2"
          value={cohortB}
          options={options}
          disabledOption={cohortA}
          onSelect={setCohortB}
          onDelete={handleDeleteClass}
        />

        {ready && payload?.sample_counts && (
          <span className="text-[11px] text-white/30">
            n={payload.sample_counts.a ?? 0} vs {payload.sample_counts.b ?? 0}
          </span>
        )}
        {loading && <Loader2 className="h-3.5 w-3.5 animate-spin text-emerald-400" />}
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-2">
        {!isAdding ? (
          <button
            type="button"
            onClick={() => setIsAdding(true)}
            className="inline-flex items-center gap-1.5 rounded-full border border-emerald-400/25 bg-emerald-500/10 px-2.5 py-1 text-[11px] font-medium text-emerald-200 transition hover:bg-emerald-500/20"
          >
            <Plus className="h-3.5 w-3.5" />
            新增对比班级
          </button>
        ) : (
          <div className="flex min-w-[220px] flex-1 items-center gap-2 sm:max-w-[360px]">
            <input
              autoFocus
              type="text"
              value={draftClass}
              onChange={(e) => setDraftClass(e.target.value)}
              onKeyDown={handleDraftKeyDown}
              placeholder="输入班级名称，如「五年级三班」"
              className="min-w-0 flex-1 rounded-xl border border-white/10 bg-black/30 px-3 py-1.5 text-xs text-white outline-none placeholder:text-white/30 focus:border-emerald-400/40"
            />
            <button
              type="button"
              onClick={handleAddClass}
              title="保存并加入选项"
              className="inline-flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-emerald-500 text-black transition hover:bg-emerald-400"
            >
              <Check className="h-4 w-4" />
            </button>
            <button
              type="button"
              onClick={() => {
                setIsAdding(false)
                setDraftClass('')
              }}
              title="取消"
              className="inline-flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-white/10 text-white/60 transition hover:bg-white/15 hover:text-white"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        )}
        <span className="text-[10px] text-white/25">
          删除仅隐藏对比选项，不会清除硬盘归档数据
        </span>
      </div>

      {filterEmpty ? (
        <div className="mt-3">
          <ChartPlaceholder message="该筛选条件下暂无有效测试数据，请调整右上角的全局过滤器" />
        </div>
      ) : !ready ? (
        <p className="mt-3 text-[11px] text-white/30">
          请在上方并排选择两个不同的班级/实验组，以渲染趋势、雷达与错误分布对比。
        </p>
      ) : loading && !payload ? (
        <div className="mt-3 flex h-[200px] items-center justify-center gap-2 text-white/40">
          <Loader2 className="h-5 w-5 animate-spin text-emerald-400" />
          <span className="text-sm">正在聚合对比数据……</span>
        </div>
      ) : !sufficient ? (
        <div className="mt-3">
          <ChartPlaceholder message={placeholderMsg} />
        </div>
      ) : (
        <div className="mt-3 grid gap-3 lg:grid-cols-3">
          <div className="rounded-2xl border border-white/8 bg-black/20 p-2 lg:col-span-1">
            <p className="mb-1 px-1 text-[11px] text-white/40">日趋势均分（双轨折线 · 含波动带）</p>
            {dates.length > 0 ? (
              <DualTrendChart
                dates={dates}
                seriesA={seriesA}
                seriesB={seriesB}
                labelA={shortA}
                labelB={shortB}
              />
            ) : (
              <ChartPlaceholder message="暂无足够数据对比" />
            )}
          </div>

          <div className="rounded-2xl border border-white/8 bg-black/20 p-2">
            <p className="mb-1 px-1 text-[11px] text-white/40">五维雷达叠影（助跑/支撑/后摆/踝锁/鞭打）</p>
            {hasRadar && radarA && radarB ? (
              <OverlayRadarChart
                scoresA={radarA}
                scoresB={radarB}
                labelA={shortA}
                labelB={shortB}
              />
            ) : (
              <ChartPlaceholder message="暂无足够数据对比" />
            )}
          </div>

          <div className="rounded-2xl border border-white/8 bg-black/20 p-2">
            <p className="mb-1 px-1 text-[11px] text-white/40">典型错误分布率</p>
            {errorCodes.length > 0 ? (
              <ErrorRateBars
                codes={errorCodes}
                ratesA={ratesA}
                ratesB={ratesB}
                labelA={shortA}
                labelB={shortB}
              />
            ) : (
              <ChartPlaceholder message="暂无足够数据对比" />
            )}
          </div>
        </div>
      )}
    </section>
  )
}
