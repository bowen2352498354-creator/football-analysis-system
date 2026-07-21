import { useEffect, useRef, useState } from 'react'
import * as echarts from 'echarts/core'
import { LineChart } from 'echarts/charts'
import {
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkPointComponent,
  TooltipComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsType } from 'echarts/core'
import { AlertTriangle, TrendingUp } from 'lucide-react'
import type { FatigueAlertPayload, GlobalTrainingRecord } from '../types'

echarts.use([
  LineChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkPointComponent,
  CanvasRenderer,
])

/** 触球瞬间膝关节夹角 · 儿童动作发展黄金区间（与 pose_tracker / report_generator 对齐） */
const GOLDEN_KNEE_MIN = 140
const GOLDEN_KNEE_MAX = 160
const KNEE_AXIS_MIN = 90
const KNEE_AXIS_MAX = 180

const MOTOR_WINDOW_N = 5
const MOTOR_VAR_AUTOMATED = 12
const MOTOR_VAR_EXPLORATORY = 35
const FATIGUE_DROP = 5
const KNEE_STIFF = 160
const ANKLE_RIGID_MIN = 10

const API_BASE_URL = 'http://localhost:8000'
const FATIGUE_POLL_MS = 2500

/** 熔断 UI 固定文案（科研伦理：防止负向肌肉记忆固化） */
export const FATIGUE_CIRCUIT_BREAKER_LABEL =
  '⚠️【动作衰减熔断】检测到下肢发力变形，请停止该被试当前轮次练习，防止负向肌肉记忆固化！'

/**
 * 前端轻量纵向 enrichment：当后端未注入模块四字段（离线缓存回退）时本地补算，
 * 算法与历史 LongitudinalProgressChart 对齐。
 */
export function enrichLongitudinalClient(records: GlobalTrainingRecord[]): GlobalTrainingRecord[] {
  if (!records.length) return []
  const sorted = [...records].sort((a, b) =>
    String(a.timestamp || '').localeCompare(String(b.timestamp || '')),
  )

  const kneeAngles = sorted.map((r) => pickKneeAngle(r))
  const rollingVars: (number | null)[] = []
  for (let i = 0; i < kneeAngles.length; i++) {
    const valid = kneeAngles
      .slice(0, i + 1)
      .filter((a): a is number => typeof a === 'number')
    if (valid.length < 2) {
      rollingVars.push(null)
      continue
    }
    const window = valid.length >= MOTOR_WINDOW_N ? valid.slice(-MOTOR_WINDOW_N) : valid
    const mean = window.reduce((s, v) => s + v, 0) / window.length
    rollingVars.push(window.reduce((s, v) => s + (v - mean) ** 2, 0) / window.length)
  }

  const overallVar = (() => {
    const valid = kneeAngles.filter((a): a is number => typeof a === 'number')
    if (valid.length < 2) return null
    const window = valid.length >= MOTOR_WINDOW_N ? valid.slice(-MOTOR_WINDOW_N) : valid
    const mean = window.reduce((s, v) => s + v, 0) / window.length
    return window.reduce((s, v) => s + (v - mean) ** 2, 0) / window.length
  })()

  const latestKnee = [...kneeAngles].reverse().find((a) => a != null) ?? null
  const inZone =
    latestKnee != null && latestKnee >= GOLDEN_KNEE_MIN && latestKnee <= GOLDEN_KNEE_MAX
  let automationStatus = '📊 样本不足·继续采集中'
  let phase = 'insufficient'
  if (overallVar != null) {
    if (overallVar <= MOTOR_VAR_AUTOMATED && inZone) {
      automationStatus = '🏆 动力链肌肉记忆已定型'
      phase = 'automated'
    } else if (overallVar > MOTOR_VAR_EXPLORATORY) {
      automationStatus = '🏃‍♂️ 步点探索与动平衡调整期'
      phase = 'exploratory'
    } else {
      automationStatus = '🏃‍♂️ 步点探索与动平衡调整期'
      phase = 'transitioning'
    }
  }

  let hasFatigue = false
  const enriched = sorted.map((record, i) => {
    const backendEnriched =
      'fatigue_alert_flag' in record ||
      'automation_status' in record ||
      typeof record.fatigueAlertFlag === 'boolean'
    if (backendEnriched) {
      if (record.fatigue_alert_flag ?? record.fatigueAlertFlag) hasFatigue = true
      return {
        ...record,
        automation_status: record.automation_status || record.automationStatus || automationStatus,
        motor_stability_phase: record.motor_stability_phase || record.motorStabilityPhase || phase,
      }
    }

    const curr = typeof record.score === 'number' ? record.score : null
    const prev = i > 0 && typeof sorted[i - 1].score === 'number' ? (sorted[i - 1].score as number) : null
    const knee = kneeAngles[i]
    const ankle =
      typeof record.quantified5dScores?.ankle_rigidity === 'number'
        ? record.quantified5dScores.ankle_rigidity
        : typeof record.quantified5dScores?.ankle_rigidity_score === 'number'
          ? record.quantified5dScores.ankle_rigidity_score
          : null
    const delta = curr != null && prev != null ? curr - prev : null
    const reasons: string[] = []
    if (delta != null && delta <= -FATIGUE_DROP) {
      if (knee != null && knee > KNEE_STIFF) reasons.push(`支撑膝直立代偿(${Math.round(knee)}°>160°)`)
      if (ankle != null && ankle < ANKLE_RIGID_MIN) reasons.push(`踝关节锁死失效(刚性${Math.round(ankle)}<10)`)
    }
    const fatigue = reasons.length > 0
    if (fatigue) hasFatigue = true

    const rolling = rollingVars[i]
    let bandUpper: number | null = null
    let bandLower: number | null = null
    if (rolling != null) {
      const valid = kneeAngles
        .slice(0, i + 1)
        .filter((a): a is number => typeof a === 'number')
      const window = valid.length >= MOTOR_WINDOW_N ? valid.slice(-MOTOR_WINDOW_N) : valid
      if (window.length >= 2) {
        const mean = window.reduce((s, v) => s + v, 0) / window.length
        const half = Math.sqrt(rolling)
        bandUpper = Math.round((mean + half) * 10) / 10
        bandLower = Math.round((mean - half) * 10) / 10
      }
    }

    return {
      ...record,
      support_knee_angle_resolved: knee,
      fatigue_alert_flag: fatigue,
      fatigue_alert_message: fatigue
        ? '⚠️ 预警：下肢疲劳动作衰减（踝膝刚性流失），建议强制休整轮换'
        : null,
      fatigue_reasons: reasons,
      score_delta: delta,
      motor_stability_index: rolling != null ? Math.round(rolling * 100) / 100 : null,
      band_upper: bandUpper,
      band_lower: bandLower,
      automation_status: automationStatus,
      motor_stability_phase: phase,
    }
  })

  if (hasFatigue && overallVar != null && overallVar > MOTOR_VAR_AUTOMATED) {
    return enriched.map((r) => ({ ...r, motor_stability_phase: 'fatigue' }))
  }
  return enriched
}

export interface LongitudinalProgressChartProps {
  records: GlobalTrainingRecord[]
  selectedIndex: number
  onSelectIndex: (index: number) => void
  /** 整组自动化定型状态；缺省时从最新记录读取 */
  automationStatus?: string | null
  /** 被试学号：用于轮询后台疲劳熔断接口 */
  studentId?: string | null
  /** 外部注入的实时疲劳报警（WebSocket / 父组件推送优先） */
  liveFatigueAlert?: FatigueAlertPayload | null
  className?: string
}

interface ChartPoint {
  index: number
  attemptLabel: string
  score: number | null
  kneeAngle: number | null
  fatigue: boolean
  fatigueMessage: string | null
  timeLabel: string
}

function getTimeLabel(record: GlobalTrainingRecord): string {
  const parts = (record.timestamp || '').split(' ')
  if (parts.length >= 2) return parts[1].slice(0, 5)
  return '--:--'
}

function pickKneeAngle(record: GlobalTrainingRecord): number | null {
  const resolved = record.support_knee_angle_resolved ?? record.supportKneeAngleResolved
  if (typeof resolved === 'number' && Number.isFinite(resolved)) return resolved
  const metrics = record.instepKickMetrics
  if (metrics && typeof metrics.support_knee_angle === 'number') {
    return metrics.support_knee_angle
  }
  if (typeof metrics?.impact_knee_angle === 'number') {
    return metrics.impact_knee_angle as number
  }
  if (typeof record.kneeFlexionAngle === 'number' && Number.isFinite(record.kneeFlexionAngle)) {
    return record.kneeFlexionAngle
  }
  return null
}

function isFatigue(record: GlobalTrainingRecord): boolean {
  return Boolean(record.fatigue_alert_flag ?? record.fatigueAlertFlag)
}

function fatigueMessage(record: GlobalTrainingRecord): string | null {
  return record.fatigue_alert_message ?? record.fatigueAlertMessage ?? null
}

function buildChartData(records: GlobalTrainingRecord[]): ChartPoint[] {
  return records.map((record, index) => ({
    index,
    attemptLabel: `Attempt ${index + 1}`,
    score: typeof record.score === 'number' && Number.isFinite(record.score) ? record.score : null,
    kneeAngle: pickKneeAngle(record),
    fatigue: isFatigue(record),
    fatigueMessage: fatigueMessage(record),
    timeLabel: getTimeLabel(record),
  }))
}

function isActiveFatigueAlert(alert: FatigueAlertPayload | null | undefined): boolean {
  return Boolean(alert && (alert.is_fatigue || alert.isFatigue))
}

/**
 * 【纵向双轴进化图谱】ECharts 科研级宽屏图
 * 左轴：五维综合评分（蓝实线） · 右轴：触球瞬间膝关节夹角（橙虚线）
 * markArea：140°–160° 淡绿黄金区间 · 右上角疲劳熔断闪烁卡
 */
export default function LongitudinalProgressChart({
  records,
  selectedIndex,
  onSelectIndex,
  automationStatus,
  studentId = null,
  liveFatigueAlert = null,
  className = '',
}: LongitudinalProgressChartProps) {
  const chartRef = useRef<HTMLDivElement>(null)
  const instanceRef = useRef<EChartsType | null>(null)
  const onSelectRef = useRef(onSelectIndex)
  onSelectRef.current = onSelectIndex

  const enriched = enrichLongitudinalClient(records)
  const chartData = buildChartData(enriched)
  const latest = enriched.length > 0 ? enriched[enriched.length - 1] : null
  const statusText =
    automationStatus || latest?.automation_status || latest?.automationStatus || null
  const phase = latest?.motor_stability_phase || latest?.motorStabilityPhase || 'insufficient'
  const isFormed = phase === 'automated' || (statusText?.includes('定型') ?? false)
  const archiveFatigue = chartData.length > 0 && Boolean(chartData[chartData.length - 1]?.fatigue)

  const [polledFatigue, setPolledFatigue] = useState<FatigueAlertPayload | null>(null)

  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      try {
        const qs = studentId ? `?student_id=${encodeURIComponent(studentId)}` : ''
        const res = await fetch(`${API_BASE_URL}/api/fatigue_alert${qs}`)
        if (!res.ok) return
        const data = (await res.json()) as FatigueAlertPayload
        if (!cancelled && isActiveFatigueAlert(data)) {
          setPolledFatigue(data)
        } else if (!cancelled) {
          setPolledFatigue(null)
        }
      } catch {
        /* 后端离线时静默：仍可用归档 fatigue_alert_flag */
      }
    }
    void poll()
    const timer = window.setInterval(() => void poll(), FATIGUE_POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [studentId])

  const activeFatigue: FatigueAlertPayload | null =
    (isActiveFatigueAlert(liveFatigueAlert) ? liveFatigueAlert : null) ||
    (isActiveFatigueAlert(polledFatigue) ? polledFatigue : null) ||
    (archiveFatigue
      ? {
          is_fatigue: true,
          reason: 'ARCHIVE_FATIGUE',
          message: FATIGUE_CIRCUIT_BREAKER_LABEL,
        }
      : null)

  const showCircuitBreaker = isActiveFatigueAlert(activeFatigue)
  const dataKey = JSON.stringify(
    chartData.map((p) => [p.score, p.kneeAngle, p.fatigue, p.index === selectedIndex]),
  )

  useEffect(() => {
    const el = chartRef.current
    if (!el || chartData.length === 0) return

    const chart = instanceRef.current ?? echarts.init(el, undefined, { renderer: 'canvas' })
    instanceRef.current = chart

    const categories = chartData.map((p) => p.attemptLabel)
    const scoreSeries = chartData.map((p) => p.score)
    const kneeSeries = chartData.map((p) => p.kneeAngle)

    const fatigueMarkPoints = chartData
      .filter((p) => p.fatigue && typeof p.kneeAngle === 'number')
      .map((p) => ({
        name: '疲劳拐点',
        coord: [p.attemptLabel, p.kneeAngle as number],
        value: p.kneeAngle,
        itemStyle: { color: '#fbbf24' },
        label: { show: true, formatter: '⚠️', color: '#fbbf24', fontSize: 12, fontWeight: 700 },
      }))

    chart.setOption(
      {
        backgroundColor: 'transparent',
        animation: true,
        animationDuration: 700,
        animationEasing: 'cubicOut',
        legend: {
          top: 4,
          right: showCircuitBreaker ? 220 : 12,
          textStyle: { color: 'rgba(255,255,255,0.55)', fontSize: 11 },
          data: ['五维综合评分', '触球瞬间膝关节夹角'],
        },
        tooltip: {
          trigger: 'axis',
          backgroundColor: 'rgba(10,14,12,0.94)',
          borderColor: 'rgba(255,255,255,0.12)',
          textStyle: { color: '#e2e8f0', fontSize: 12 },
          formatter: (params: unknown) => {
            const list = Array.isArray(params) ? params : []
            if (!list.length) return ''
            const idx = (list[0] as { dataIndex?: number }).dataIndex ?? 0
            const point = chartData[idx]
            if (!point) return ''
            const scoreText = point.score != null ? `${point.score} 分` : '暂无'
            const kneeText = point.kneeAngle != null ? `${point.kneeAngle}°` : '暂无'
            const inBand =
              point.kneeAngle != null &&
              point.kneeAngle >= GOLDEN_KNEE_MIN &&
              point.kneeAngle <= GOLDEN_KNEE_MAX
            const bandHint = inBand ? ' · 落入黄金区间' : ' · 偏离黄金区间'
            let html = `<div style="font-weight:600;margin-bottom:6px">${point.attemptLabel} · ${point.timeLabel}</div>`
            html += `<div>五维综合评分：<span style="color:#60a5fa">${scoreText}</span></div>`
            html += `<div>膝关节夹角：<span style="color:#fb923c">${kneeText}</span><span style="color:rgba(255,255,255,0.4)">${point.kneeAngle != null ? bandHint : ''}</span></div>`
            if (point.fatigue) {
              html += `<div style="margin-top:6px;color:#fbbf24;font-weight:600">${point.fatigueMessage || '⚠️ 下肢疲劳动作衰减'}</div>`
            }
            return html
          },
        },
        grid: {
          left: 56,
          right: 64,
          top: 48,
          bottom: 36,
          containLabel: false,
        },
        xAxis: {
          type: 'category',
          data: categories,
          boundaryGap: false,
          axisLine: { lineStyle: { color: 'rgba(255,255,255,0.12)' } },
          axisTick: { show: false },
          axisLabel: {
            color: 'rgba(255,255,255,0.45)',
            fontSize: 11,
          },
          splitLine: { show: false },
        },
        yAxis: [
          {
            type: 'value',
            name: '五维综合评分',
            nameTextStyle: { color: 'rgba(96,165,250,0.7)', fontSize: 10, padding: [0, 0, 0, 8] },
            min: 0,
            max: 100,
            position: 'left',
            axisLine: { show: true, lineStyle: { color: 'rgba(59,130,246,0.45)' } },
            axisLabel: { color: '#60a5fa', fontSize: 11 },
            splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
          },
          {
            type: 'value',
            name: '膝关节夹角 °',
            nameTextStyle: { color: 'rgba(251,146,60,0.75)', fontSize: 10, padding: [0, 8, 0, 0] },
            min: KNEE_AXIS_MIN,
            max: KNEE_AXIS_MAX,
            position: 'right',
            axisLine: { show: true, lineStyle: { color: 'rgba(249,115,22,0.45)' } },
            axisLabel: { color: '#fb923c', fontSize: 11, formatter: '{value}°' },
            splitLine: { show: false },
          },
        ],
        series: [
          {
            name: '五维综合评分',
            type: 'line',
            yAxisIndex: 0,
            data: scoreSeries,
            connectNulls: true,
            smooth: 0.15,
            symbol: 'circle',
            symbolSize: (_val: unknown, params: { dataIndex?: number }) =>
              params.dataIndex === selectedIndex ? 11 : 7,
            itemStyle: {
              color: '#3b82f6',
              borderColor: '#fff',
              borderWidth: 1,
            },
            lineStyle: { color: '#3b82f6', width: 2.6, type: 'solid' },
            emphasis: { focus: 'series', scale: 1.15 },
          },
          {
            name: '触球瞬间膝关节夹角',
            type: 'line',
            yAxisIndex: 1,
            data: kneeSeries,
            connectNulls: true,
            smooth: 0.15,
            symbol: 'circle',
            symbolSize: (_val: unknown, params: { dataIndex?: number }) =>
              params.dataIndex === selectedIndex ? 9 : 5,
            itemStyle: { color: '#f97316' },
            lineStyle: { color: '#f97316', width: 2.2, type: 'dashed' },
            emphasis: { focus: 'series' },
            // 淡绿半透明黄金区间：横穿整个图表，便于观察橙线何时跌入/逃出绿带
            markArea: {
              silent: true,
              itemStyle: {
                color: 'rgba(52, 211, 153, 0.20)',
              },
              label: {
                show: true,
                position: 'insideTopRight',
                formatter: `黄金区间 ${GOLDEN_KNEE_MIN}°–${GOLDEN_KNEE_MAX}°`,
                color: '#6ee7b7',
                fontSize: 10,
                fontWeight: 600,
              },
              data: [
                [
                  { yAxis: GOLDEN_KNEE_MIN, name: '黄金区间' },
                  { yAxis: GOLDEN_KNEE_MAX },
                ],
              ],
            },
            markPoint:
              fatigueMarkPoints.length > 0
                ? {
                    symbol: 'circle',
                    symbolSize: 10,
                    data: fatigueMarkPoints,
                  }
                : undefined,
          },
        ],
      },
      { notMerge: true },
    )

    const onClick = (params: { dataIndex?: number }) => {
      if (typeof params.dataIndex === 'number') {
        onSelectRef.current(params.dataIndex)
      }
    }
    chart.off('click')
    chart.on('click', onClick)

    const onResize = () => chart.resize()
    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('resize', onResize)
      chart.off('click', onClick)
    }
  }, [dataKey, chartData, selectedIndex, showCircuitBreaker])

  useEffect(() => {
    return () => {
      instanceRef.current?.dispose()
      instanceRef.current = null
    }
  }, [])

  if (records.length === 0) {
    return (
      <div
        className={`longitudinal-progress-chart flex h-80 flex-col items-center justify-center gap-2 rounded-2xl bg-black/20 text-center ${className}`}
      >
        <TrendingUp className="h-8 w-8 text-white/15" />
        <p className="text-sm text-white/35">暂无历史尝试，完成首次射门测验后即可生成纵向进化图谱</p>
        <p className="text-[11px] text-white/20">X 轴按 Attempt 1 → N 展开后，可对照黄金区间观察膝角漂移</p>
      </div>
    )
  }

  return (
    <div className={`longitudinal-progress-chart relative flex flex-col gap-3 ${className}`}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          {statusText && (
            <span
              className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-[11px] font-semibold backdrop-blur-xl ${
                isFormed
                  ? 'border-amber-300/40 bg-amber-400/15 text-amber-200'
                  : 'border-sky-400/30 bg-sky-400/10 text-sky-200'
              }`}
            >
              {statusText}
            </span>
          )}
          <span className="rounded-full border border-white/10 bg-black/30 px-2.5 py-1 text-[10px] text-white/40">
            双 Y 轴 · 评分 0–100 × 膝角 {KNEE_AXIS_MIN}°–{KNEE_AXIS_MAX}°
          </span>
        </div>

        {/* 疲劳熔断高亮：右上角剧烈闪烁警告卡 */}
        {showCircuitBreaker && (
          <div
            role="alert"
            className="max-w-md rounded-xl border-2 border-red-800 bg-yellow-500 px-3.5 py-2.5 shadow-lg animate-pulse text-red-900"
          >
            <p className="flex items-start gap-2 text-[12px] font-bold leading-snug">
              <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0" />
              <span>{FATIGUE_CIRCUIT_BREAKER_LABEL}</span>
            </p>
            {activeFatigue?.reason && activeFatigue.reason !== 'ARCHIVE_FATIGUE' && (
              <p className="mt-1 pl-6 text-[10px] font-semibold uppercase tracking-wide text-red-800/80">
                信号码 · {activeFatigue.reason}
              </p>
            )}
          </div>
        )}
      </div>

      <div className="w-full overflow-hidden rounded-2xl bg-black/20 p-2">
        <div ref={chartRef} className="h-[22rem] w-full min-h-[320px]" />
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[10px] text-white/30">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-0.5 w-4 rounded bg-blue-500" />
          五维综合评分（左轴 · 蓝实线）
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-0.5 w-4 rounded border-t border-dashed border-orange-400" />
          触球瞬间膝关节夹角（右轴 · 橙虚线）
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-4 rounded-sm bg-emerald-400/30" />
          学术合规黄金区间 {GOLDEN_KNEE_MIN}°–{GOLDEN_KNEE_MAX}°（markArea）
        </span>
        <span className="inline-flex items-center gap-1.5 text-amber-300/70">
          ⚠️ 疲劳变形拐点 · 橙线跌入/逃出绿带一目了然
        </span>
      </div>

      {records.length === 1 && (
        <p className="text-[11px] text-white/25">
          当前仅 1 次尝试。累计 ≥2 次后可观察膝角相对绿带的纵向漂移；点击拐点可联动时空胶囊。
        </p>
      )}
    </div>
  )
}
