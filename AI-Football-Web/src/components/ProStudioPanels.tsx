import { motion } from 'framer-motion'
import { Gauge, Pause, Play, Rewind, FastForward, StepBack, StepForward } from 'lucide-react'
import type {
  BiomechIndicatorKey,
  BiomechIndicatorValue,
  OverallComplianceStatus,
  PlaybackRate,
  ScoreDetailPayload,
} from '../types'
import { TRAFFIC_CLASS, type TrafficLightLevel } from '../theme/trafficLight'

/** 8 大量纲 → 默认扣分错误码（右栏证据链回填；折叠/膝角按实测值再细分） */
const INDICATOR_ERROR_CODE: Partial<Record<BiomechIndicatorKey, string>> = {
  distance_cm: 'ERR_A2_SUPPORT_WIDE',
  toe_angle: 'ERR_C2_TOE_POKE',
  max_folding_angle: 'ERR_SWING_FOLD',
  whipping_velocity: 'ERR_FOLLOW_THROUGH',
  impact_knee_angle: 'ERR_KNEE_STIFF',
  ankle_rigidity: 'ERR_C1_LOOSE_ANKLE',
  support_knee_angle: 'ERR_KNEE_STIFF',
  hip_torsion_angle: 'ERR_TORSO_TILT',
}

/** 折叠深度 → 后摆膝内角；与后端 max_folding = 180 − swing_fold 同源 */
function foldingDepthToSwingInterior(foldingDepth: number): number {
  return Math.max(0, 180 - foldingDepth)
}

/**
 * 按实测值选择错误码，杜绝「测得 120° 却挂 >170° 直腿文案」的错位。
 * - max_folding_angle：深度过小（膝内角 >140）→ B1；过大（膝内角 <70）→ SWING_FOLD
 * - impact_knee_angle：仅 >165° → B1 直腿；其余 → KNEE_STIFF
 * - distance_cm：肩宽比 >0.9 → A2 严重偏宽；否则 SUPPORT_LATERAL
 */
function resolveIndicatorErrorCode(
  key: BiomechIndicatorKey,
  entry: BiomechIndicatorValue,
): string | undefined {
  const value = typeof entry.value === 'number' ? entry.value : null
  if (key === 'max_folding_angle' && value != null) {
    const interior = foldingDepthToSwingInterior(value)
    if (interior > 140) return 'ERR_B1_STRAIGHT_LEG'
    return 'ERR_SWING_FOLD'
  }
  if (key === 'impact_knee_angle' && value != null) {
    if (value > 165) return 'ERR_B1_STRAIGHT_LEG'
    return 'ERR_KNEE_STIFF'
  }
  if (key === 'distance_cm' && value != null) {
    // V3.8：肩宽归一化比例；>0.9 严重外挂，<0.25 过近
    const ratio =
      typeof entry.support_ratio === 'number' ? entry.support_ratio : value
    if (ratio > 0.9) return 'ERR_A2_SUPPORT_WIDE'
    if (ratio < 0.25) return 'ERR_SUPPORT_TOO_CLOSE'
    return 'ERR_SUPPORT_LATERAL'
  }
  return INDICATOR_ERROR_CODE[key]
}

function indicatorTrafficLevel(status: string | null | undefined): TrafficLightLevel {
  if (!status) return 'pending'
  const s = status.toUpperCase()
  if (s.includes('GREEN')) return 'green'
  if (s.includes('YELLOW')) return 'yellow'
  if (s.includes('RED')) return 'red'
  return 'pending'
}

/**
 * 从 scoreDetail / overall_status / 8 项指标严格判定是否「全部合规」。
 * 仅当 overall_status===PERFECT，或已有细分指标且无一为 RED/YELLOW 时返回 true。
 * 空数据 / 待机态一律视为未合规（禁止误亮绿色框）。
 */
export function isOverallPerfectCompliance(options: {
  overallStatus?: OverallComplianceStatus | null
  scoreDetail?: ScoreDetailPayload | null
  indicators?: Partial<Record<BiomechIndicatorKey, BiomechIndicatorValue>> | null
}): boolean {
  const explicit =
    options.overallStatus ??
    options.scoreDetail?.overall_status ??
    options.scoreDetail?.overallStatus ??
    null
  if (typeof explicit === 'string' && explicit.toUpperCase() === 'PERFECT') {
    return true
  }
  if (typeof explicit === 'string' && explicit.trim() !== '') {
    return false
  }

  const indicators = options.indicators ?? options.scoreDetail?.indicators ?? null
  if (!indicators || Object.keys(indicators).length === 0) return false

  for (const entry of Object.values(indicators)) {
    if (!entry) continue
    const level = indicatorTrafficLevel(entry.status)
    if (level === 'red' || level === 'yellow') return false
  }
  return true
}

/** 从 8 项 RED/YELLOW 指标推导扣分错误码（补全后端未下发 error_codes 的断层） */
export function deriveErrorCodesFromIndicators(
  indicators: Partial<Record<BiomechIndicatorKey, BiomechIndicatorValue>> | null | undefined,
): string[] {
  if (!indicators) return []
  const codes: string[] = []
  const seen = new Set<string>()
  for (const [key, entry] of Object.entries(indicators) as Array<
    [BiomechIndicatorKey, BiomechIndicatorValue | undefined]
  >) {
    if (!entry) continue
    const level = indicatorTrafficLevel(entry.status)
    if (level !== 'red' && level !== 'yellow') continue
    const code = resolveIndicatorErrorCode(key, entry)
    if (!code || seen.has(code)) continue
    seen.add(code)
    codes.push(code)
  }
  return codes
}

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
    penalty: 6,
    format: (v) => (typeof v === 'number' ? `${v.toFixed(1)}cm` : '--'),
  },
  {
    key: 'support_knee_angle',
    label: '支撑膝角',
    unit: '°',
    errorCode: 'ERR_KNEE_STIFF',
    penalty: 6,
    format: (v) => (typeof v === 'number' ? `${v.toFixed(1)}°` : '--'),
  },
  {
    key: 'swing_fold_angle',
    label: '蓄力膝角',
    unit: '°',
    errorCode: 'ERR_B1_STRAIGHT_LEG',
    penalty: 8,
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
    penalty: 8,
    format: (v) => (typeof v === 'number' ? `${v.toFixed(1)}°` : '--'),
  },
  {
    key: 'whipping_speed_peak',
    label: '随摆鞭打速度',
    unit: '°/s',
    errorCode: 'ERR_FOLLOW_THROUGH',
    penalty: 6,
    format: (v) => (typeof v === 'number' ? `${v.toFixed(0)}°/s` : '--'),
  },
]

/** 8 大黄金指标错误代码 -> 人类可读的技术扣分理由文案（V3.5 儿童/业余容错） */
export const ERROR_CODE_LABELS: Record<string, string> = {
  ERR_WARMUP_CLOSE: '支撑脚距球心过近（<5cm）',
  ERR_A1_SUPPORT_BACK: '支撑脚尖落后球心超过 10cm',
  ERR_A2_SUPPORT_WIDE: '支撑脚横距比例过远（>0.9 个肩宽，严重外挂）',
  ERR_B1_STRAIGHT_LEG: '后摆/触球膝角过大：后摆膝内角>140° 或触球膝角>165°（折叠不足/直腿）',
  ERR_B2_SHANK_ONLY: '浅折叠且大腿后伸≈0°（仅小腿弹射；90–130° 合理区不触发）',
  ERR_C1_LOOSE_ANKLE: '击球窗踝关节松弛泄力（方差/背屈骤降超标）',
  ERR_C2_TOE_POKE: '足背未外展，脚尖直捅球体',
  PASS_STANDARD: '各项指标落入合理区间',
  ERR_APPROACH_ANGLE: '助跑夹角未落在 20°-60° 黄金斜线区间',
  ERR_SUPPORT_LATERAL: '支撑脚横距比例略偏（理想约 0.4–0.7 个肩宽）',
  ERR_SUPPORT_TOO_CLOSE: '支撑脚横距比例过近（<0.25 个肩宽）',
  ERR_SUPPORT_AP: '支撑脚尖相对球心前后位置不合理',
  ERR_KNEE_STIFF: '触球/支撑膝角偏离缓冲带（触球直腿仅 >165° 触发）',
  ERR_SWING_FOLD: '后摆折叠偏离 90–130° 合理发力区（过度折叠或略浅）',
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

export interface DeductionListProps {
  errorCodes?: string[] | null
  /** 后端 overall_status；仅 PERFECT 可亮绿色合规框 */
  overallStatus?: OverallComplianceStatus | null
  /** DeterministicScorer 明细；用于二次校验 8 项是否含 RED/YELLOW */
  scoreDetail?: ScoreDetailPayload | null
  /** 是否已有分析结论（无结论时不亮绿色合规框） */
  hasAnalysisResult?: boolean
}

/** 后端 deductions 条目（与左侧展示变量同源） */
type ScorerDeduction = {
  metric_key?: string
  measured_value?: number
  unit?: string
  penalty?: number
  status?: string
  reason?: string
  error_code?: string
}

/** 右栏「扣分项清单」：优先后端 deductions（含实测值），否则回退 error_codes */
export function DeductionList({
  errorCodes,
  overallStatus = null,
  scoreDetail = null,
  hasAnalysisResult = false,
}: DeductionListProps) {
  const derived = deriveErrorCodesFromIndicators(scoreDetail?.indicators)
  const backendDeductions = (scoreDetail as { deductions?: ScorerDeduction[] } | null)
    ?.deductions
  const hasBackendDeductions = Array.isArray(backendDeductions) && backendDeductions.length > 0
  // 显式传入数组（含空数组）时尊重调用方筛选结果，不再回填 derived
  const codes = Array.isArray(errorCodes) ? errorCodes : derived
  const isPerfect =
    hasAnalysisResult &&
    isOverallPerfectCompliance({ overallStatus, scoreDetail }) &&
    (Array.isArray(errorCodes) ? errorCodes.length === 0 && derived.length === 0 : derived.length === 0)

  if (isPerfect) {
    return (
      <div className="rounded-2xl border border-emerald-400/20 bg-emerald-500/10 px-3.5 py-3 text-xs text-emerald-200">
        ✅ 本次分析未命中任何量化扣分项，8 大黄金指标全部合规！
      </div>
    )
  }

  // 优先渲染后端 deductions：reason 已绑定实测值，与左侧指标不可能错位
  if (hasBackendDeductions && !Array.isArray(errorCodes)) {
    return (
      <div className="flex flex-col gap-1.5">
        {backendDeductions!.map((item, idx) => (
          <div
            key={`${item.metric_key ?? 'd'}-${idx}`}
            className="flex items-center justify-between gap-2 rounded-xl border border-rose-400/20 bg-rose-500/10 px-3 py-2"
          >
            <span className="text-[11px] leading-relaxed text-rose-200">
              {item.reason ||
                (item.error_code ? ERROR_CODE_LABELS[item.error_code] : null) ||
                item.metric_key ||
                '扣分项'}
            </span>
            <span className="flex-shrink-0 rounded-full bg-rose-500/25 px-2 py-0.5 text-[10px] font-bold text-rose-100">
              -{typeof item.penalty === 'number' ? item.penalty : '?'} 分
            </span>
          </div>
        ))}
      </div>
    )
  }

  if (codes.length === 0) {
    if (!hasAnalysisResult) {
      return (
        <div className="rounded-2xl border border-slate-600/40 bg-slate-900/40 px-3.5 py-3 text-xs text-slate-400">
          等待分析完成后展示量化扣分证据链。
        </div>
      )
    }
    // 筛选器清空了列表，或合规未达标但当前视图无码可展示
    return null
  }

  const indicatorPenaltyByCode = new Map<string, number>()
  if (scoreDetail?.indicators) {
    for (const [key, entry] of Object.entries(scoreDetail.indicators) as Array<
      [BiomechIndicatorKey, BiomechIndicatorValue | undefined]
    >) {
      if (!entry || typeof entry.penalty !== 'number') continue
      const code = resolveIndicatorErrorCode(key, entry)
      if (code) indicatorPenaltyByCode.set(code, entry.penalty)
    }
  }

  return (
    <div className="flex flex-col gap-1.5">
      {codes.map((code) => {
        const def = GOLDEN_METRIC_DEFS.find((item) => item.errorCode === code)
        const livePenalty = indicatorPenaltyByCode.get(code)
        // 优先用指标实测 penalty；过滤视图下若有后端 deductions 也按码对齐
        const fromBackend = hasBackendDeductions
          ? backendDeductions!.find((d) => d.error_code === code)
          : undefined
        const penalty =
          typeof fromBackend?.penalty === 'number'
            ? fromBackend.penalty
            : typeof livePenalty === 'number'
              ? livePenalty
              : def?.penalty
        const label =
          fromBackend?.reason || ERROR_CODE_LABELS[code] || code
        return (
          <div
            key={code}
            className="flex items-center justify-between gap-2 rounded-xl border border-rose-400/20 bg-rose-500/10 px-3 py-2"
          >
            <span className="text-[11px] leading-relaxed text-rose-200">{label}</span>
            <span className="flex-shrink-0 rounded-full bg-rose-500/25 px-2 py-0.5 text-[10px] font-bold text-rose-100">
              -{penalty ?? '?'} 分
            </span>
          </div>
        )
      })}
    </div>
  )
}
