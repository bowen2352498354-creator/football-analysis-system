import { motion } from 'framer-motion'
import { Gauge, Pause, Play, Rewind, FastForward, StepBack, StepForward } from 'lucide-react'
import type { PlaybackRate } from '../types'
import { TRAFFIC_CLASS, type TrafficLightLevel } from '../theme/trafficLight'

/* ============================================================================
 * 【第二部分：Pro-Studio 职业视频工作台】共享子组件
 *
 * 与 api_server.py / pose_tracker.py 的「脚背内侧射门确定性算分引擎」8 大
 * 黄金指标 + error_codes 证据链严格一一对应，供 RealtimeWorkspace.tsx /
 * ZenWorkspace.tsx 的左栏「🚦 红绿信号灯 Dashboard」与右栏「扣分清单」
 * 复用，绝不在两个工作台里各写一份不一致的映射规则。
 * ========================================================================== */

/** 8 大黄金指标的展示配置：字段名、中文标签、单位、对应错误代码、扣分分值 */
export interface GoldenMetricDef {
  key: string
  label: string
  unit: string
  errorCode: string
  penalty: number
  /** 把原始数值格式化为展示文案（布尔型指标如"踝关节锁死"需要特殊处理） */
  format: (value: number | boolean | null | undefined) => string
}

export const GOLDEN_METRIC_DEFS: GoldenMetricDef[] = [
  {
    key: 'approach_angle',
    label: '助跑夹角',
    unit: '°',
    errorCode: 'ERR_APPROACH_ANGLE',
    penalty: 6,
    format: (v) => (typeof v === 'number' ? `${v.toFixed(1)}°` : '--'),
  },
  {
    key: 'support_lateral_dist_cm',
    label: '支撑横向距离',
    unit: 'cm',
    errorCode: 'ERR_A2_SUPPORT_WIDE',
    penalty: 8,
    format: (v) => (typeof v === 'number' ? `${v.toFixed(1)}cm` : '--'),
  },
  {
    key: 'support_ap_offset_cm',
    label: '支撑前后偏移',
    unit: 'cm',
    errorCode: 'ERR_A1_SUPPORT_BACK',
    penalty: 8,
    format: (v) => (typeof v === 'number' ? `${v.toFixed(1)}cm` : '--'),
  },
  {
    key: 'support_knee_angle',
    label: '支撑膝角',
    unit: '°',
    errorCode: 'ERR_KNEE_STIFF',
    penalty: 10,
    format: (v) => (typeof v === 'number' ? `${v.toFixed(1)}°` : '--'),
  },
  {
    key: 'swing_fold_angle',
    label: '蓄力膝角',
    unit: '°',
    errorCode: 'ERR_B1_STRAIGHT_LEG',
    penalty: 6,
    format: (v) => (typeof v === 'number' ? `${v.toFixed(1)}°` : '--'),
  },
  {
    key: 'torso_lateral_tilt',
    label: '躯干侧倾角',
    unit: '°',
    errorCode: 'ERR_TORSO_TILT',
    penalty: 6,
    format: (v) => (typeof v === 'number' ? `${v.toFixed(1)}°` : '--'),
  },
  {
    key: 'ankle_angle',
    label: '脚踝锁死',
    unit: '°',
    errorCode: 'ERR_C1_LOOSE_ANKLE',
    penalty: 15,
    format: (v) => (typeof v === 'number' ? `${v.toFixed(1)}°` : '--'),
  },
  {
    key: 'whipping_speed_peak',
    label: '随摆鞭打速度',
    unit: '°/s',
    errorCode: 'ERR_FOLLOW_THROUGH',
    penalty: 5,
    format: (v) => (typeof v === 'number' ? `${v.toFixed(0)}°/s` : '--'),
  },
]

/** 8 大黄金指标错误代码 -> 人类可读的技术扣分理由文案（与 llm_agent.py 保持一致的技术口径） */
export const ERROR_CODE_LABELS: Record<string, string> = {
  ERR_WARMUP_CLOSE: '支撑脚距球心过近（<5cm）',
  ERR_A1_SUPPORT_BACK: '支撑脚尖落后球心超过 10cm',
  ERR_A2_SUPPORT_WIDE: '支撑脚横向距离偏宽（>20cm）',
  ERR_B1_STRAIGHT_LEG: '蓄力窗内膝关节极值角 >170°（全程直腿）',
  ERR_B2_SHANK_ONLY: '小腿有折叠但大腿后伸≈0°（仅小腿弹射）',
  ERR_C1_LOOSE_ANKLE: '击球窗踝关节松弛泄力（方差/背屈骤降超标）',
  ERR_C2_TOE_POKE: '足背未外展，脚尖直捅球体',
  PASS_STANDARD: '各项指标落入合理区间',
  ERR_APPROACH_ANGLE: '助跑夹角未落在 20°-60° 黄金斜线区间',
  ERR_SUPPORT_LATERAL: '支撑脚横向距离超出 15-20cm 合理区间',
  ERR_SUPPORT_AP: '支撑脚尖相对球心前后位置不合理',
  ERR_KNEE_STIFF: '支撑腿膝关节过度绷直（超过 160°），缺乏缓冲',
  ERR_SWING_FOLD: '蓄力阶段膝折叠不足（直腿戳球）',
  ERR_TORSO_TILT: '躯干侧倾角度不合理（僵硬直立或侧倾失衡）',
  ERR_ANKLE_LOOSE: '触球瞬间踝关节未绷紧锁死，力量在此环节泄漏',
  ERR_FOLLOW_THROUGH: '随摆挥速不足或未完成跨体随摆，发力不连贯',
}

type MetricsRecord = Record<string, number | boolean | null> | null | undefined

/** 🚦 8 大黄金技术指标 Traffic-Light Dashboard：左栏 MetricPanel 核心组件 */
export function TrafficLightDashboard({
  metrics,
  errorCodes,
}: {
  metrics: MetricsRecord
  errorCodes: string[] | null | undefined
}) {
  const hasData = !!metrics && Object.keys(metrics).length > 0
  const safeErrorCodes = errorCodes ?? []

  return (
    <div className="flex flex-col gap-2">
      <h4 className="flex items-center gap-2 text-xs font-semibold text-slate-300">
        <span className="inline-flex flex-shrink-0">
          <Gauge className="h-3.5 w-3.5 text-[var(--GREEN_OPTIMAL)]" />
        </span>
        8 大黄金技术指标信号灯
      </h4>
      <div className="flex flex-col gap-1.5">
        {GOLDEN_METRIC_DEFS.map((def) => {
          const rawValue = hasData ? metrics?.[def.key] : null
          // 新典型错误码与旧码并存时，同一物理指标任一命中即亮红灯
          const relatedCodes: Record<string, string[]> = {
            ERR_A2_SUPPORT_WIDE: ['ERR_A2_SUPPORT_WIDE', 'ERR_SUPPORT_LATERAL', 'ERR_WARMUP_CLOSE'],
            ERR_A1_SUPPORT_BACK: ['ERR_A1_SUPPORT_BACK', 'ERR_SUPPORT_AP'],
            ERR_B1_STRAIGHT_LEG: ['ERR_B1_STRAIGHT_LEG', 'ERR_B2_SHANK_ONLY', 'ERR_SWING_FOLD'],
            ERR_C1_LOOSE_ANKLE: ['ERR_C1_LOOSE_ANKLE', 'ERR_C2_TOE_POKE', 'ERR_ANKLE_LOOSE'],
          }
          const aliases = relatedCodes[def.errorCode] ?? [def.errorCode]
          const isHit = aliases.some((c) => safeErrorCodes.includes(c))
          const level: TrafficLightLevel = !hasData ? 'pending' : isHit ? 'red' : 'green'
          const tone = TRAFFIC_CLASS[level]
          return (
            <div
              key={def.key}
              className={`flex items-center justify-between gap-2 rounded-xl border px-3 py-2 ${tone.border} ${tone.bg} ${tone.glow}`}
            >
              <span className="flex items-center gap-1.5 text-[11px] text-slate-400">
                <span className={`h-1.5 w-1.5 flex-shrink-0 rounded-full ${tone.dot}`} />
                {def.label}
              </span>
              <span className={`flex items-center gap-1 text-xs font-bold tabular-nums ${tone.text}`}>
                {def.format(rawValue)}
                {isHit &&
                  (def.key === 'support_lateral_dist_cm' || def.key === 'support_ap_offset_cm') && (
                    <span className="text-[9px] font-normal opacity-70">偏远</span>
                  )}
              </span>
            </div>
          )
        })}
      </div>
      {!hasData && (
        <p className="mt-1 text-[10px] leading-relaxed text-slate-500">
          等待本次分析结束后，确定性算分引擎将回填 8 大黄金指标真实实测数值。
        </p>
      )}
    </div>
  )
}

/** 四段彩色动作时序进度条各阶段的固定展示配置（比例仅用于视觉分段展示） */
const TIMELINE_SEGMENTS: { key: string; label: string; color: string; ratio: number }[] = [
  { key: 'approach', label: '助跑段', color: 'bg-sky-400', ratio: 30 },
  { key: 'fold', label: '折叠段', color: 'bg-amber-400', ratio: 20 },
  { key: 'contact', label: '锁踝触球核心段', color: 'bg-emerald-400', ratio: 20 },
  { key: 'followThrough', label: '随摆段', color: 'bg-rose-400', ratio: 30 },
]

/** 🟦助跑段 -> 🟨折叠段 -> 🟩锁踝触球核心段 -> 🟥随摆段：对标 Premiere Pro 的四段彩色动作时序进度条 */
export function FourSegmentTimeline({ progressPercent }: { progressPercent: number | null }) {
  const clamped = progressPercent === null ? null : Math.max(0, Math.min(100, progressPercent))
  return (
    <div className="flex flex-col gap-1.5">
      <div className="relative flex h-3 overflow-hidden rounded-full bg-black/30">
        {TIMELINE_SEGMENTS.map((segment) => (
          <div key={segment.key} className={`h-full ${segment.color} opacity-70`} style={{ width: `${segment.ratio}%` }} />
        ))}
        {clamped !== null && (
          <motion.div
            className="absolute top-0 h-full w-[3px] bg-white shadow-[0_0_8px_rgba(255,255,255,0.9)]"
            animate={{ left: `${clamped}%` }}
            transition={{ duration: 0.15, ease: 'linear' }}
          />
        )}
      </div>
      <div className="flex items-center justify-between text-[10px] text-white/40">
        {TIMELINE_SEGMENTS.map((segment) => (
          <span key={segment.key} className="flex items-center gap-1">
            <span className={`h-1.5 w-1.5 rounded-full ${segment.color}`} />
            {segment.label}
          </span>
        ))}
      </div>
    </div>
  )
}

const PLAYBACK_RATE_OPTIONS: PlaybackRate[] = [0.25, 0.5, 1]

/** 高精度视频播控台：0.25x / 0.5x / 1x 变速 + 逐帧前进/后退（仅本地视频分析模式生效） */
export function PlaybackControlBar({
  disabled,
  isPaused,
  playbackRate,
  onSetRate,
  onTogglePause,
  onStepFrame,
}: {
  disabled: boolean
  isPaused: boolean
  playbackRate: PlaybackRate
  onSetRate: (rate: PlaybackRate) => void
  onTogglePause: () => void
  onStepFrame: (direction: 'forward' | 'backward') => void
}) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-2 rounded-2xl bg-black/20 px-3 py-2">
      <div className="inline-flex items-center gap-1 rounded-full bg-black/30 p-0.5">
        {PLAYBACK_RATE_OPTIONS.map((rate) => (
          <button
            key={rate}
            type="button"
            disabled={disabled}
            onClick={() => onSetRate(rate)}
            className={`rounded-full px-2.5 py-1 text-[11px] font-semibold transition disabled:cursor-not-allowed disabled:opacity-40 ${
              playbackRate === rate ? 'bg-emerald-400 text-black' : 'text-white/50 hover:text-white/80'
            }`}
          >
            {rate}x
          </button>
        ))}
      </div>

      <div className="flex items-center gap-1">
        <button
          type="button"
          disabled={disabled}
          onClick={() => onStepFrame('backward')}
          title="逐帧后退"
          className="flex h-8 w-8 items-center justify-center rounded-full bg-white/10 text-white/70 transition hover:bg-white/20 active:scale-95 disabled:cursor-not-allowed disabled:opacity-30"
        >
          <StepBack className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={onTogglePause}
          title={isPaused ? '继续播放' : '暂停'}
          className="flex h-9 w-9 items-center justify-center rounded-full bg-emerald-400 text-black transition hover:bg-emerald-300 active:scale-95 disabled:cursor-not-allowed disabled:bg-white/10 disabled:text-white/30"
        >
          {isPaused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={() => onStepFrame('forward')}
          title="逐帧前进"
          className="flex h-8 w-8 items-center justify-center rounded-full bg-white/10 text-white/70 transition hover:bg-white/20 active:scale-95 disabled:cursor-not-allowed disabled:opacity-30"
        >
          <StepForward className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="flex items-center gap-1 text-white/25">
        <Rewind className="h-3 w-3" />
        <span className="text-[9px]">高精度播控台</span>
        <FastForward className="h-3 w-3" />
      </div>
    </div>
  )
}

/** 右栏「扣分项清单」：根据 error_codes 命中列表渲染逐条量化扣分依据 */
export function DeductionList({ errorCodes }: { errorCodes: string[] | null | undefined }) {
  const codes = errorCodes ?? []
  if (codes.length === 0) {
    return (
      <div className="rounded-2xl border border-emerald-400/20 bg-emerald-500/10 px-3.5 py-3 text-xs text-emerald-200">
        ✅ 本次分析未命中任何量化扣分项，8 大黄金指标全部合规！
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-1.5">
      {codes.map((code) => {
        const def = GOLDEN_METRIC_DEFS.find((item) => item.errorCode === code)
        return (
          <div
            key={code}
            className="flex items-center justify-between gap-2 rounded-xl border border-rose-400/20 bg-rose-500/10 px-3 py-2"
          >
            <span className="text-[11px] leading-relaxed text-rose-200">
              {ERROR_CODE_LABELS[code] ?? code}
            </span>
            <span className="flex-shrink-0 rounded-full bg-rose-500/25 px-2 py-0.5 text-[10px] font-bold text-rose-100">
              -{def?.penalty ?? '?'} 分
            </span>
          </div>
        )
      })}
    </div>
  )
}
