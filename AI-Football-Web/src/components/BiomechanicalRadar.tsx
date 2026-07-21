import { useEffect, useRef } from 'react'
import * as echarts from 'echarts/core'
import { RadarChart } from 'echarts/charts'
import {
  LegendComponent,
  RadarComponent,
  TooltipComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsType } from 'echarts/core'
import type { Quantified5dScores, RadarScores } from '../types'

echarts.use([RadarChart, RadarComponent, TooltipComponent, LegendComponent, CanvasRenderer])

/** 五维维度元数据：与后端 radar_scores 键一一对应 */
export const FIVE_D_DIMENSIONS = [
  { key: 'approach_rhythm' as const, label: '助跑与进袭', short: '助跑' },
  { key: 'support_stability' as const, label: '支撑与稳固', short: '支撑' },
  { key: 'backswing_folding' as const, label: '蓄力与折叠', short: '折叠' },
  { key: 'ankle_rigidity' as const, label: '锁踝与刚性', short: '锁踝' },
  { key: 'whipping_velocity' as const, label: '鞭打与随摆', short: '鞭打' },
]

const EMPTY_SCORES: RadarScores = {
  approach_rhythm: 0,
  support_stability: 0,
  backswing_folding: 0,
  ankle_rigidity: 0,
  whipping_velocity: 0,
}

export type FiveDGrade = 'excellent' | 'reinforce' | 'critical' | 'empty'

export function gradeFiveDScore(score: number | null | undefined): FiveDGrade {
  if (typeof score !== 'number' || Number.isNaN(score)) return 'empty'
  if (score >= 16) return 'excellent'
  if (score >= 10) return 'reinforce'
  return 'critical'
}

export function fiveDGradeMeta(grade: FiveDGrade): { badge: string; barClass: string; textClass: string } {
  switch (grade) {
    case 'excellent':
      return { badge: '🟢 优秀', barClass: 'bg-emerald-400', textClass: 'text-emerald-300' }
    case 'reinforce':
      return { badge: '🟡 待强化', barClass: 'bg-amber-400', textClass: 'text-amber-300' }
    case 'critical':
      return { badge: '🔴 严重扣分/泄力', barClass: 'bg-rose-400', textClass: 'text-rose-300' }
    default:
      return { badge: '⚪ 暂无', barClass: 'bg-white/20', textClass: 'text-white/35' }
  }
}

function clampDim(v: unknown): number {
  return typeof v === 'number' && !Number.isNaN(v) ? Math.max(0, Math.min(20, v)) : 0
}

/**
 * 归一化后端 radar_scores / 旧版 quantified_5d_scores 为统一 RadarScores。
 */
export function normalizeRadarScores(raw: Quantified5dScores | RadarScores | null | undefined): RadarScores {
  if (!raw || typeof raw !== 'object') return { ...EMPTY_SCORES }
  const r = raw as Quantified5dScores
  return {
    approach_rhythm: clampDim(r.approach_rhythm ?? r.approach_score),
    support_stability: clampDim(r.support_stability ?? r.support_score),
    backswing_folding: clampDim(r.backswing_folding ?? r.backswing_score),
    ankle_rigidity: clampDim(r.ankle_rigidity ?? r.ankle_rigidity_score),
    whipping_velocity: clampDim(r.whipping_velocity ?? r.whipping_score),
  }
}

function hasRadarData(raw: Quantified5dScores | RadarScores | null | undefined): boolean {
  if (!raw || typeof raw !== 'object') return false
  return FIVE_D_DIMENSIONS.some((d) => {
    const r = raw as Quantified5dScores
    const v =
      r[d.key] ??
      (d.key === 'approach_rhythm'
        ? r.approach_score
        : d.key === 'support_stability'
          ? r.support_score
          : d.key === 'backswing_folding'
            ? r.backswing_score
            : d.key === 'ankle_rigidity'
              ? r.ankle_rigidity_score
              : r.whipping_score)
    return typeof v === 'number' && !Number.isNaN(v)
  })
}

export interface BiomechanicalRadarProps {
  /** 当前学员实际表现（Attempt 主数据 / 后端 radar_scores） */
  scores?: Quantified5dScores | RadarScores | null
  /** 可选：第二次尝试，用于雷达重叠对比 */
  compareScores?: Quantified5dScores | RadarScores | null
  /** 主数据标签，默认 Attempt #1 */
  primaryLabel?: string
  /** 对比数据标签，默认 Attempt #2 */
  compareLabel?: string
  /** 紧凑模式（Clinical Dock） */
  compact?: boolean
  className?: string
}

/**
 * V3.1 五维生物力学雷达图（ECharts）
 * 深色科幻雷达网 + 翡翠绿半透明填充 + 浮现动画。
 */
export default function BiomechanicalRadar({
  scores,
  compareScores = null,
  primaryLabel = 'Attempt #1',
  compareLabel = 'Attempt #2',
  compact = false,
  className = '',
}: BiomechanicalRadarProps) {
  const chartRef = useRef<HTMLDivElement>(null)
  const instanceRef = useRef<EChartsType | null>(null)

  const primary = normalizeRadarScores(scores)
  const compare = compareScores ? normalizeRadarScores(compareScores) : null
  const hasData = hasRadarData(scores)
  const total = hasData
    ? FIVE_D_DIMENSIONS.reduce((sum, d) => sum + primary[d.key], 0)
    : 0

  const primaryKey = JSON.stringify(primary)
  const compareKey = compare ? JSON.stringify(compare) : ''

  useEffect(() => {
    const el = chartRef.current
    if (!el) return

    const chart = instanceRef.current ?? echarts.init(el, undefined, { renderer: 'canvas' })
    instanceRef.current = chart

    if (!hasData) {
      chart.clear()
      return
    }

    const indicator = FIVE_D_DIMENSIONS.map((d) => ({
      name: d.short,
      max: 20,
    }))

    const primaryParsed = JSON.parse(primaryKey) as RadarScores
    const compareParsed = compareKey ? (JSON.parse(compareKey) as RadarScores) : null

    const seriesData: Array<Record<string, unknown>> = [
      {
        value: FIVE_D_DIMENSIONS.map((d) => primaryParsed[d.key]),
        name: primaryLabel,
        symbol: 'circle',
        symbolSize: 6,
        lineStyle: { width: 2, color: 'rgba(16, 185, 129, 0.95)' },
        itemStyle: { color: '#10b981' },
        areaStyle: { color: 'rgba(16, 185, 129, 0.4)' },
      },
    ]

    if (compareParsed) {
      seriesData.push({
        value: FIVE_D_DIMENSIONS.map((d) => compareParsed[d.key]),
        name: compareLabel,
        symbol: 'circle',
        symbolSize: 5,
        lineStyle: { width: 1.5, color: 'rgba(251, 191, 36, 0.9)' },
        itemStyle: { color: '#fbbf24' },
        areaStyle: { color: 'rgba(251, 191, 36, 0.22)' },
      })
    }

    chart.setOption(
      {
        backgroundColor: 'transparent',
        animation: true,
        animationDuration: 1100,
        animationEasing: 'cubicOut',
        animationDurationUpdate: 600,
        tooltip: {
          trigger: 'item',
          backgroundColor: 'rgba(0,0,0,0.82)',
          borderColor: 'rgba(255,255,255,0.12)',
          textStyle: { color: '#e2e8f0', fontSize: 11 },
          formatter: (params: unknown) => {
            const p = params as { name?: string; value?: number[]; seriesName?: string }
            const vals = Array.isArray(p.value) ? p.value : []
            const rows = FIVE_D_DIMENSIONS.map(
              (d, i) => `${d.label}：${vals[i] ?? 0}/20`,
            ).join('<br/>')
            return `<div style="font-weight:600;margin-bottom:4px">${p.seriesName ?? ''}</div>${rows}`
          },
        },
        legend: compareParsed
          ? {
              bottom: 0,
              textStyle: { color: 'rgba(255,255,255,0.45)', fontSize: 10 },
              data: [primaryLabel, compareLabel],
            }
          : undefined,
        radar: {
          indicator,
          center: ['50%', compareParsed ? '46%' : '50%'],
          radius: compact ? '62%' : '68%',
          startAngle: 90,
          splitNumber: 4,
          shape: 'polygon',
          axisName: {
            color: 'rgba(226, 232, 240, 0.7)',
            fontSize: 11,
            fontWeight: 500,
          },
          axisLine: {
            lineStyle: { color: 'rgba(100, 116, 139, 0.45)' },
          },
          splitLine: {
            lineStyle: { color: 'rgba(71, 85, 105, 0.55)', width: 1 },
          },
          splitArea: {
            show: true,
            areaStyle: {
              color: [
                'rgba(15, 23, 42, 0.15)',
                'rgba(30, 41, 59, 0.35)',
                'rgba(15, 23, 42, 0.25)',
                'rgba(30, 58, 138, 0.22)',
              ],
            },
          },
        },
        series: [
          {
            type: 'radar',
            name: '五维雷达',
            data: seriesData,
            emphasis: {
              lineStyle: { width: 3 },
              areaStyle: { color: 'rgba(16, 185, 129, 0.55)' },
            },
          },
        ],
      },
      { notMerge: true },
    )

    const onResize = () => chart.resize()
    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('resize', onResize)
    }
  }, [hasData, primaryKey, compareKey, primaryLabel, compareLabel, compact])

  useEffect(() => {
    return () => {
      instanceRef.current?.dispose()
      instanceRef.current = null
    }
  }, [])

  return (
    <div className={`biomech-radar ${compact ? 'biomech-radar--compact' : ''} ${className}`.trim()}>
      <div className="mb-3 flex items-end justify-between gap-2">
        <div>
          <p className="text-[11px] text-white/40">五维生物力学量化雷达 · V3.1</p>
          <p className="text-3xl font-bold tabular-nums text-emerald-300">
            {hasData ? Math.round(total * 10) / 10 : '--'}
            <span className="ml-1 text-sm font-medium text-white/35">/ 100</span>
          </p>
        </div>
        <div className="flex flex-col items-end gap-1 text-[10px] text-white/40">
          <span className="inline-flex items-center gap-1.5">
            <i className="inline-block h-2 w-2 rounded-sm bg-emerald-400/70" />
            {primaryLabel}
          </span>
          {compare && (
            <span className="inline-flex items-center gap-1.5">
              <i className="inline-block h-2 w-2 rounded-sm bg-amber-400" />
              {compareLabel}
            </span>
          )}
        </div>
      </div>

      <div
        className={`relative w-full overflow-hidden rounded-2xl border border-slate-700/50 bg-gradient-to-b from-slate-950 via-slate-900/90 to-[#0a1628] ${
          compact ? 'h-[200px]' : 'h-[260px]'
        }`}
      >
        {!hasData && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 px-4 text-center">
            <p className="text-sm text-white/45">暂无五维量化评分</p>
            <p className="text-[11px] leading-relaxed text-white/25">
              完成一次完整射门分析后，将在此渲染后端 radar_scores 五维雷达面。
            </p>
          </div>
        )}
        <div ref={chartRef} className={`h-full w-full ${hasData ? '' : 'invisible'}`} />
      </div>

      <div className={`mt-3 grid gap-2 ${compact ? 'grid-cols-1' : 'grid-cols-1 sm:grid-cols-5'}`}>
        {FIVE_D_DIMENSIONS.map((dim) => {
          const value = hasData ? primary[dim.key] : 0
          const grade = hasData ? gradeFiveDScore(value) : 'empty'
          const meta = fiveDGradeMeta(grade)
          const pct = Math.max(0, Math.min(100, (value / 20) * 100))
          return (
            <div
              key={dim.key}
              className="biomech-radar__capsule rounded-2xl border border-white/8 bg-black/25 px-3 py-2.5"
            >
              <div className="mb-1.5 flex items-center justify-between gap-2">
                <span className="truncate text-[11px] font-medium text-white/70">{dim.label}</span>
                <span className={`shrink-0 text-[10px] font-semibold ${meta.textClass}`}>{meta.badge}</span>
              </div>
              <div className="mb-1.5 flex items-baseline justify-between">
                <span className={`text-sm font-bold tabular-nums ${meta.textClass}`}>
                  {hasData ? value : '--'}
                  <span className="text-[10px] font-normal text-white/30"> / 20</span>
                </span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-white/10">
                <div
                  className={`h-full rounded-full transition-all duration-700 ease-out ${meta.barClass}`}
                  style={{ width: `${hasData ? pct : 0}%` }}
                />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
