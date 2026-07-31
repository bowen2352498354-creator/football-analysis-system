import { useMemo, useRef, useState } from 'react'
import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { AngularVelocityPoint, KineticChainDiagnosis } from '../types'
import { IMPACT_WINDOW_HALF } from './VideoWorkspace'

export interface KineticVelocityChartProps {
  /** 髋/膝/踝角速度时序；缺省或空数组时展示零状态引导 */
  profile?: AngularVelocityPoint[] | null
  /** 动力链诊断印章文案来源 */
  diagnosis?: KineticChainDiagnosis | null
  /**
   * 播控协同：当前高亮时间（相对触球的 time_ms）。
   * 可由视频播放进度或外部 Hover 驱动；图表自身 Hover 也会回调 onHoverTimeMs。
   */
  highlightTimeMs?: number | null
  onHoverTimeMs?: (timeMs: number | null) => void
  /**
   * 点击 / 拖拽波形点时回调窗口内索引 x_index（0~60）。
   * 调用方必须映射：absolute_frame = action_roi.start + x_index，
   * 再 video.currentTime = absolute_frame / 30.0 —— 绝不用 x_index 直接跳视频。
   */
  onSeekXIndex?: (xIndex: number) => void
  /**
   * 视频播放驱动的窗口内游标索引（0~60）；越界传 null 则隐藏高亮线。
   * 正确算法：x_index = floor(currentTime * 30) - action_roi.start
   */
  highlightXIndex?: number | null
  /**
   * 触球点在当前窗口内的索引（红色虚线）；缺省 30。
   * 对应后端 impact_index_in_window。
   */
  impactIndexInWindow?: number | null
  compact?: boolean
  className?: string
}

function sanitizeProfile(raw: AngularVelocityPoint[] | null | undefined): AngularVelocityPoint[] {
  if (!Array.isArray(raw) || raw.length === 0) return []
  const out: AngularVelocityPoint[] = []
  for (const row of raw) {
    if (!row || typeof row !== 'object') continue
    const time_ms = typeof row.time_ms === 'number' ? row.time_ms : Number(row.time_ms)
    if (!Number.isFinite(time_ms)) continue
    out.push({
      frame: typeof row.frame === 'number' ? row.frame : out.length,
      time_ms,
      hip_vel: Number.isFinite(Number(row.hip_vel)) ? Number(row.hip_vel) : 0,
      knee_vel: Number.isFinite(Number(row.knee_vel)) ? Number(row.knee_vel) : 0,
      ankle_vel: Number.isFinite(Number(row.ankle_vel)) ? Number(row.ankle_vel) : 0,
    })
  }
  return out
}

function diagnosisTone(status: string | undefined): { border: string; bg: string; text: string } {
  if (!status) {
    return { border: 'border-white/10', bg: 'bg-white/5', text: 'text-white/45' }
  }
  if (status.includes('🟢') || status.includes('达标')) {
    return { border: 'border-emerald-400/35', bg: 'bg-emerald-500/10', text: 'text-emerald-300' }
  }
  if (status.includes('🔴') || status.includes('错误')) {
    return { border: 'border-rose-400/35', bg: 'bg-rose-500/10', text: 'text-rose-300' }
  }
  if (status.includes('🟡') || status.includes('待强化')) {
    return { border: 'border-amber-400/35', bg: 'bg-amber-500/10', text: 'text-amber-300' }
  }
  return { border: 'border-white/10', bg: 'bg-white/5', text: 'text-white/50' }
}

function resolveXIndexFromPayload(
  state: { activeTooltipIndex?: number | string | null; activeLabel?: unknown } | null | undefined,
  data: AngularVelocityPoint[],
): number | null {
  if (!state || data.length === 0) return null
  const tipIdx = state.activeTooltipIndex
  if (typeof tipIdx === 'number' && tipIdx >= 0 && tipIdx < data.length) return tipIdx
  if (typeof tipIdx === 'string' && tipIdx !== '') {
    const parsed = Number(tipIdx)
    if (Number.isFinite(parsed) && parsed >= 0 && parsed < data.length) return Math.round(parsed)
  }
  const label = state.activeLabel
  const ms = typeof label === 'number' ? label : Number(label)
  if (!Number.isFinite(ms)) return null
  let best = 0
  let bestDist = Number.POSITIVE_INFINITY
  for (let i = 0; i < data.length; i += 1) {
    const dist = Math.abs(data[i].time_ms - ms)
    if (dist < bestDist) {
      bestDist = dist
      best = i
    }
  }
  return best
}

/**
 * 动力链多关节角速度时序图：三曲线同步 + 触球零点参考线 + 诊断印章。
 * 点击 / 拖拽仅回调窗口内 x_index（0~60）；绝对帧映射由 VideoWorkspace 完成。
 */
export default function KineticVelocityChart({
  profile,
  diagnosis = null,
  highlightTimeMs = null,
  onHoverTimeMs,
  onSeekXIndex,
  highlightXIndex = null,
  impactIndexInWindow = IMPACT_WINDOW_HALF,
  compact = false,
  className = '',
}: KineticVelocityChartProps) {
  const data = useMemo(() => sanitizeProfile(profile), [profile])
  const [localHoverMs, setLocalHoverMs] = useState<number | null>(null)
  const draggingRef = useRef(false)

  /** 优先使用绝对帧映射得到的窗口索引高亮；否则回退 time_ms Hover */
  const activeMsFromXIndex =
    typeof highlightXIndex === 'number' &&
    highlightXIndex >= 0 &&
    highlightXIndex < data.length
      ? data[highlightXIndex]?.time_ms ?? null
      : null

  const activeMs =
    typeof localHoverMs === 'number'
      ? localHoverMs
      : typeof activeMsFromXIndex === 'number'
        ? activeMsFromXIndex
        : typeof highlightTimeMs === 'number'
          ? highlightTimeMs
          : null

  const status = diagnosis?.status
  const tone = diagnosisTone(status)
  const hasData = data.length >= 2
  const impactX =
    typeof impactIndexInWindow === 'number' && Number.isFinite(impactIndexInWindow)
      ? Math.max(0, Math.min(data.length > 0 ? data.length - 1 : IMPACT_WINDOW_HALF, Math.round(impactIndexInWindow)))
      : IMPACT_WINDOW_HALF
  const impactTimeMs = data[impactX]?.time_ms ?? 0

  const emitSeek = (xIndex: number | null) => {
    if (xIndex == null || !onSeekXIndex) return
    onSeekXIndex(Math.max(0, Math.min(Math.max(0, data.length - 1), Math.round(xIndex))))
  }

  return (
    <div className={`kinetic-velocity-chart ${compact ? 'kinetic-velocity-chart--compact' : ''} ${className}`.trim()}>
      {/* 智能诊断印章 */}
      <div className={`mb-3 rounded-2xl border px-3 py-2.5 ${tone.border} ${tone.bg}`}>
        <p className="text-[10px] uppercase tracking-wide text-white/35">动力链鞭打诊断</p>
        <p className={`mt-0.5 text-sm font-semibold leading-snug ${tone.text}`}>
          {status || '⚪ 等待角速度时序矩阵…'}
        </p>
        {diagnosis && hasData && (
          <p className="mt-1 text-[10px] tabular-nums text-white/30">
            峰值 t<sub>髋</sub>={diagnosis.hip_peak_time_ms ?? '—'}ms · t<sub>膝</sub>=
            {diagnosis.knee_peak_time_ms ?? '—'}ms · t<sub>踝</sub>={diagnosis.ankle_peak_time_ms ?? '—'}ms
          </p>
        )}
      </div>

      <div className={`w-full ${compact ? 'h-[200px]' : 'h-[260px]'}`}>
        {!hasData ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-white/10 bg-black/20 px-4 text-center">
            <p className="text-sm text-white/45">暂无关节角速度时序</p>
            <p className="max-w-sm text-[11px] leading-relaxed text-white/25">
              老版本归档 JSON 可能不含 angularVelocityProfile。完成新一次分析后，将在此绘制髋（红虚线）/ 膝（黄）/
              踝（绿）三曲线，并以触球瞬间为绝对零点。点击 / 拖拽曲线可联动视频跳转到窗口帧。
            </p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={data}
              margin={{ top: 8, right: 12, left: 0, bottom: 4 }}
              onMouseDown={(state) => {
                draggingRef.current = true
                emitSeek(resolveXIndexFromPayload(state, data))
              }}
              onMouseMove={(state) => {
                const label = state?.activeLabel
                const ms = typeof label === 'number' ? label : Number(label)
                if (Number.isFinite(ms)) {
                  setLocalHoverMs(ms)
                  onHoverTimeMs?.(ms)
                }
                if (draggingRef.current) {
                  emitSeek(resolveXIndexFromPayload(state, data))
                }
              }}
              onMouseUp={(state) => {
                if (draggingRef.current) {
                  emitSeek(resolveXIndexFromPayload(state, data))
                }
                draggingRef.current = false
              }}
              onClick={(state) => {
                emitSeek(resolveXIndexFromPayload(state, data))
              }}
              onMouseLeave={() => {
                draggingRef.current = false
                setLocalHoverMs(null)
                onHoverTimeMs?.(null)
              }}
            >
              <CartesianGrid stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
              <XAxis
                dataKey="time_ms"
                type="number"
                domain={['dataMin', 'dataMax']}
                tick={{ fill: 'rgba(255,255,255,0.35)', fontSize: 10 }}
                tickFormatter={(v) => `${v}`}
                label={{
                  value: 'time_ms（触球=0）· 窗内 0~60',
                  position: 'insideBottomRight',
                  offset: -2,
                  style: { fill: 'rgba(255,255,255,0.25)', fontSize: 10 },
                }}
              />
              <YAxis
                tick={{ fill: 'rgba(255,255,255,0.35)', fontSize: 10 }}
                label={{
                  value: 'deg/s',
                  angle: -90,
                  position: 'insideLeft',
                  style: { fill: 'rgba(255,255,255,0.25)', fontSize: 10 },
                }}
              />
              <Tooltip
                contentStyle={{
                  background: 'rgba(0,0,0,0.85)',
                  border: '1px solid rgba(255,255,255,0.12)',
                  borderRadius: 12,
                  fontSize: 11,
                }}
                labelFormatter={(label) => `t = ${label} ms`}
                formatter={(value, name) => [`${Number(value ?? 0).toFixed(1)} °/s`, String(name)]}
              />
              <Legend
                wrapperStyle={{ fontSize: 11, color: 'rgba(255,255,255,0.55)' }}
                iconType="plainline"
              />
              {/* ⚽ 触球瞬间（红色虚线，对应 backend impact 窗内索引） */}
              <ReferenceLine
                x={impactTimeMs}
                stroke="rgba(239,68,68,0.95)"
                strokeWidth={1.75}
                strokeDasharray="5 4"
                label={{
                  value: '⚽ 触球瞬间',
                  position: 'insideTopLeft',
                  fill: 'rgba(248,113,113,0.9)',
                  fontSize: 11,
                }}
              />
              {/* 播控 / Hover 协同高亮指示线 */}
              {typeof activeMs === 'number' && (
                <ReferenceLine
                  x={activeMs}
                  stroke="rgba(56,189,248,0.75)"
                  strokeWidth={1.25}
                  strokeDasharray="4 3"
                />
              )}
              <Line
                type="monotone"
                dataKey="hip_vel"
                name="髋 hip_vel"
                stroke="rgba(248,113,113,0.9)"
                strokeWidth={1.5}
                strokeDasharray="6 4"
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="knee_vel"
                name="膝 knee_vel"
                stroke="rgba(250,204,21,0.95)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="ankle_vel"
                name="踝 ankle_vel"
                stroke="rgba(52,211,153,0.95)"
                strokeWidth={2.25}
                dot={false}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  )
}

/**
 * 将视频播放进度映射为窗口内 x_index / 相对触球 time_ms。
 * absolute_frame = floor(currentTime * fps)；x_index = absolute_frame - actionRoiStart。
 * 越出 0~60（或数据长度）返回 null —— 调用方应隐藏高亮游标。
 */
export function mapVideoTimeToWaveformXIndex(
  currentTimeSec: number,
  actionRoiStart: number,
  fps = 30,
  windowLength = IMPACT_WINDOW_HALF * 2 + 1,
): number | null {
  if (!Number.isFinite(currentTimeSec) || !Number.isFinite(actionRoiStart)) return null
  const xIndex = Math.floor(currentTimeSec * fps) - Math.round(actionRoiStart)
  if (xIndex < 0 || xIndex >= windowLength) return null
  return xIndex
}

/**
 * 将视频播放进度粗映射为相对触球的 time_ms，供播控协同高亮。
 * 必须传入 action_roi.start；禁止用整段视频进度线性拉伸到波形轴（会造成绝对错位）。
 */
export function estimateHighlightTimeMs(
  profile: AngularVelocityPoint[] | null | undefined,
  positionMs: number | null | undefined,
  _durationMs: number | null | undefined,
  actionRoiStart?: number | null,
  fps = 30,
): number | null {
  const data = sanitizeProfile(profile)
  if (data.length < 2) return null
  if (typeof positionMs !== 'number' || !Number.isFinite(positionMs)) return null
  if (typeof actionRoiStart !== 'number' || !Number.isFinite(actionRoiStart)) return null
  const xIndex = mapVideoTimeToWaveformXIndex(positionMs / 1000, actionRoiStart, fps, data.length)
  if (xIndex == null) return null
  return data[xIndex]?.time_ms ?? null
}
