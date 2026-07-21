import { useMemo, useState } from 'react'
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
      frame: typeof row.frame === 'number' ? row.frame : out.length + 1,
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

/**
 * 动力链多关节角速度时序图：三曲线同步 + 触球零点参考线 + 诊断印章。
 * 支持 Hover / 外部播控高亮指示线；老 JSON 无时序时平滑零状态，绝不抛错。
 */
export default function KineticVelocityChart({
  profile,
  diagnosis = null,
  highlightTimeMs = null,
  onHoverTimeMs,
  compact = false,
  className = '',
}: KineticVelocityChartProps) {
  const data = useMemo(() => sanitizeProfile(profile), [profile])
  const [localHoverMs, setLocalHoverMs] = useState<number | null>(null)

  const activeMs =
    typeof localHoverMs === 'number'
      ? localHoverMs
      : typeof highlightTimeMs === 'number'
        ? highlightTimeMs
        : null

  const status = diagnosis?.status
  const tone = diagnosisTone(status)
  const hasData = data.length >= 2

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
              踝（绿）三曲线，并以触球瞬间为绝对零点。
            </p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={data}
              margin={{ top: 8, right: 12, left: 0, bottom: 4 }}
              onMouseMove={(state) => {
                const label = state?.activeLabel
                const ms = typeof label === 'number' ? label : Number(label)
                if (!Number.isFinite(ms)) return
                setLocalHoverMs(ms)
                onHoverTimeMs?.(ms)
              }}
              onMouseLeave={() => {
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
                  value: 'time_ms（触球=0）',
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
              {/* ⚽ 触球瞬间绝对零点 */}
              <ReferenceLine
                x={0}
                stroke="rgba(255,255,255,0.85)"
                strokeWidth={1.5}
                label={{
                  value: '⚽ 触球瞬间',
                  position: 'insideTopLeft',
                  fill: 'rgba(255,255,255,0.75)',
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
 * 将视频播放进度粗映射为相对触球的 time_ms，供播控协同高亮。
 * 无时序数据时返回 null（调用方安全忽略）。
 */
export function estimateHighlightTimeMs(
  profile: AngularVelocityPoint[] | null | undefined,
  positionMs: number | null | undefined,
  durationMs: number | null | undefined,
): number | null {
  const data = sanitizeProfile(profile)
  if (data.length < 2) return null
  if (typeof positionMs !== 'number' || typeof durationMs !== 'number' || durationMs <= 0) return null
  const tMin = data[0].time_ms
  const tMax = data[data.length - 1].time_ms
  const progress = Math.max(0, Math.min(1, positionMs / durationMs))
  return Math.round(tMin + progress * (tMax - tMin))
}
