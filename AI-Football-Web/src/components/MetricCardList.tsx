import { useMemo, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  Crosshair,
  FileSpreadsheet,
  Footprints,
  Loader2,
  MapPinned,
  MoveDiagonal,
  Radar as RadarIcon,
  RotateCcw,
  Shield,
  Sparkles,
  Target,
  Wind,
  Zap,
} from 'lucide-react'
import BiomechanicalRadar, { normalizeRadarScores } from './BiomechanicalRadar'
import SpatialHeatmap from './SpatialHeatmap'
import { TRAFFIC_CLASS, type TrafficLightLevel } from '../theme/trafficLight'
import type {
  AcademicExportResult,
  BiomechIndicatorKey,
  BiomechIndicatorValue,
  BiomechStatusCode,
  MetricRenderMode,
  MetricSeekEvent,
  Quantified5dScores,
  ScoreDetailPayload,
} from '../types'

const API_BASE_URL = 'http://localhost:8000'

/** 8 大量纲展示元数据：图标、中文名、单位格式、具身隐喻（按 GREEN/YELLOW/RED） */
export const BIOMECH_CARD_DEFS: {
  key: BiomechIndicatorKey
  label: string
  shortLabel: string
  Icon: typeof Target
  format: (value: number | null | undefined, unit?: string) => string
  metaphors: Record<'green' | 'yellow' | 'red' | 'pending', string>
}[] = [
  {
    key: 'distance_cm',
    label: '支撑脚偏移',
    shortLabel: '支撑偏移',
    Icon: Footprints,
    format: (v) => (typeof v === 'number' ? `${v.toFixed(1)} cm` : '--'),
    metaphors: {
      green: '像大树的根扎稳了地面，支撑脚离球刚刚好！',
      yellow: '再靠近球心一点点，像把根扎在标志盘正中。',
      red: '下次把支撑脚挪到球旁，想象脚掌是树根要站稳。',
      pending: '等待本次分析回填支撑脚偏移实测值…',
    },
  },
  {
    key: 'toe_angle',
    label: '脚尖指向',
    shortLabel: '脚尖指向',
    Icon: MoveDiagonal,
    format: (v) => (typeof v === 'number' ? `${v.toFixed(1)}°` : '--'),
    metaphors: {
      green: '脚尖轻轻指向传球方向，像指南针对准目标！',
      yellow: '脚尖再微微外展一点，对准你想踢去的方向。',
      red: '触球前把脚尖转向目标，别让脚尖直直捅球。',
      pending: '等待脚尖指向角回填…',
    },
  },
  {
    key: 'max_folding_angle',
    label: '后摆折叠角',
    shortLabel: '后摆折叠',
    Icon: RotateCcw,
    format: (v) => (typeof v === 'number' ? `${v.toFixed(1)}°` : '--'),
    metaphors: {
      green: '摆动腿像拉满的弓弦，折叠蓄力刚刚好！',
      yellow: '后摆再多收一点小腿，把弓弦多拉满一格。',
      red: '先练原地折叠：小腿贴大腿，像拉满弓再放开。',
      pending: '等待后摆折叠极值回填…',
    },
  },
  {
    key: 'whipping_velocity',
    label: '鞭打速度',
    shortLabel: '鞭打速度',
    Icon: Zap,
    format: (v) => (typeof v === 'number' ? `${v.toFixed(0)} °/s` : '--'),
    metaphors: {
      green: '小腿像鞭子轻轻甩出去，速度又快又连贯！',
      yellow: '随摆再甩快一点，想象鞭梢甩到最远处。',
      red: '触球后别急刹车，让腿继续像鞭子甩过身体。',
      pending: '等待鞭打峰值速度回填…',
    },
  },
  {
    key: 'impact_knee_angle',
    label: '膝夹角',
    shortLabel: '触球膝角',
    Icon: Crosshair,
    format: (v) => (typeof v === 'number' ? `${v.toFixed(1)}°` : '--'),
    metaphors: {
      green: '触球时膝盖弯曲刚刚好，像弹簧蓄好了力！',
      yellow: '触球瞬间膝盖再微屈一点，别绷得太直。',
      red: '击球时保持膝盖微弯，像弹簧门轻轻弹出。',
      pending: '等待触球膝夹角回填…',
    },
  },
  {
    key: 'ankle_rigidity',
    label: '脚踝背屈锁紧度',
    shortLabel: '脚踝锁紧',
    Icon: Shield,
    format: (v, unit) =>
      typeof v === 'number'
        ? unit === 'variance' || unit === undefined
          ? `σ² ${v.toFixed(2)}`
          : `${v.toFixed(1)}${unit || ''}`
        : '--',
    metaphors: {
      green: '脚踝锁成坚硬的铁板，力量一点都不漏！',
      yellow: '触球时再绷紧脚面，像穿上一只结实的小靴子。',
      red: '触球瞬间把脚踝冻住，脚面变成坚硬的铁板。',
      pending: '等待脚踝锁紧度回填…',
    },
  },
  {
    key: 'support_knee_angle',
    label: '支撑膝角',
    shortLabel: '支撑膝角',
    Icon: Target,
    format: (v) => (typeof v === 'number' ? `${v.toFixed(1)}°` : '--'),
    metaphors: {
      green: '支撑腿膝盖微屈缓冲，落地又稳又软！',
      yellow: '支撑腿再微微屈膝，像坐在小凳子边缘。',
      red: '落地时别把支撑腿绷直，膝盖轻轻弯一下。',
      pending: '等待支撑膝角回填…',
    },
  },
  {
    key: 'hip_torsion_angle',
    label: '髋部扭转角',
    shortLabel: '髋扭转',
    Icon: Wind,
    format: (v) => (typeof v === 'number' ? `${v.toFixed(1)}°` : '--'),
    metaphors: {
      green: '髋部像灵活的陀螺转起来，带动整条腿甩出！',
      yellow: '转髋再多一点，肩膀和骨盆轻轻拧一下。',
      red: '踢球时转开髋部，想象身体在轻轻拧毛巾。',
      pending: '等待髋扭转角回填…',
    },
  },
]

function statusToLevel(status: string | null | undefined): TrafficLightLevel {
  if (!status) return 'pending'
  const s = status.toUpperCase()
  if (s.includes('GREEN')) return 'green'
  if (s.includes('YELLOW')) return 'yellow'
  if (s.includes('RED')) return 'red'
  return 'pending'
}

function resolveExtremeFrame(
  key: BiomechIndicatorKey,
  entry: BiomechIndicatorValue | undefined,
  scoreDetail: ScoreDetailPayload | null | undefined,
  fallbackImpact: number | null | undefined,
): number | null {
  if (typeof entry?.extreme_frame_index === 'number') return entry.extreme_frame_index
  const fromMap = scoreDetail?.metric_extreme_frames?.[key]
  if (typeof fromMap === 'number') return fromMap
  if (typeof fallbackImpact === 'number') {
    // 无极值元数据时的相位启发式：后摆/鞭打略早于触球，支撑类略早落地
    if (key === 'max_folding_angle') return Math.max(0, fallbackImpact - 8)
    if (key === 'whipping_velocity') return Math.max(0, fallbackImpact - 2)
    if (key === 'distance_cm' || key === 'toe_angle' || key === 'support_knee_angle') {
      return Math.max(0, fallbackImpact - 3)
    }
    return fallbackImpact
  }
  return null
}

/** GROUP_B 子面板：离线雷达 / 幽灵骨架 / 空间热力图 */
type GroupBPanel = 'radar' | 'ghost' | 'heatmap'

export interface MetricCardListProps {
  renderMode: MetricRenderMode
  /** DeterministicScorer scoreDetail；优先于 metrics 扁平字典 */
  scoreDetail?: ScoreDetailPayload | null
  /** 兼容旧扁平指标字典（无 scoreDetail 时降级） */
  metrics?: Record<string, number | boolean | null> | null
  /** 命中错误码（无 scoreDetail.status 时辅助推断红灯） */
  errorCodes?: string[] | null
  /** 五维雷达（GROUP_B 离线雷达面板） */
  radarScores?: Quantified5dScores | null
  /** 幽灵骨架对比的第二次 Attempt 雷达（可选） */
  compareRadarScores?: Quantified5dScores | null
  /** A 组全局处方文案（卡片级隐喻之外的大字号补充） */
  groupALeadMetaphor?: string | null
  /** 触球零点（无 scoreDetail 时用于极值帧启发式） */
  tImpact?: number | null
  /** 点击卡片 → VideoWorkspace Seek */
  onMetricSeek?: (event: MetricSeekEvent) => void
  /** COACH_CONSOLE 导出成功回调 */
  onExportSuccess?: (result: AcademicExportResult) => void
  /** COACH_CONSOLE 导出失败回调 */
  onExportError?: (message: string) => void
  /** Sprint 1：后端时空热力图 base64（也可从 scoreDetail.heatmap_base64 读取） */
  heatmapBase64?: string | null
  className?: string
}

/**
 * 左侧 8 大生物力学量纲卡片列表。
 * 支持 GROUP_A / GROUP_B / COACH_CONSOLE 三种实验组差异化渲染，
 * 并在点击时把对应物理极值帧回传给 VideoWorkspace。
 */
export default function MetricCardList({
  renderMode,
  scoreDetail = null,
  metrics = null,
  errorCodes = null,
  radarScores = null,
  compareRadarScores = null,
  groupALeadMetaphor = null,
  tImpact = null,
  onMetricSeek,
  onExportSuccess,
  onExportError,
  heatmapBase64 = null,
  className = '',
}: MetricCardListProps) {
  const [groupBPanel, setGroupBPanel] = useState<GroupBPanel>('radar')
  const [isExporting, setIsExporting] = useState(false)
  const [activeKey, setActiveKey] = useState<BiomechIndicatorKey | null>(null)

  const impactFrame = scoreDetail?.t_impact ?? tImpact ?? null
  const indicators = scoreDetail?.indicators
  /** 优先显式传入；否则回退 scoreDetail.radar_scores（V3.1） */
  const effectiveRadarScores = radarScores ?? scoreDetail?.radar_scores ?? null
  const effectiveCompareRadar = compareRadarScores
  const effectiveHeatmap =
    heatmapBase64 ?? scoreDetail?.heatmap_base64 ?? null

  const cards = useMemo(() => {
    return BIOMECH_CARD_DEFS.map((def) => {
      const entry = indicators?.[def.key]
      let value: number | null =
        typeof entry?.value === 'number'
          ? entry.value
          : typeof metrics?.[def.key] === 'number'
            ? (metrics[def.key] as number)
            : null

      // 兼容旧黄金指标字段名
      if (value === null && metrics) {
        const aliases: Partial<Record<BiomechIndicatorKey, string[]>> = {
          distance_cm: ['support_lateral_dist_cm', 'distance_cm'],
          toe_angle: ['support_toe_angle', 'toe_angle'],
          max_folding_angle: ['swing_fold_angle', 'max_folding_angle'],
          whipping_velocity: ['whipping_speed_peak', 'whipping_velocity'],
          impact_knee_angle: ['impact_knee_angle'],
          ankle_rigidity: ['ankle_angle', 'ankle_rigidity_variance'],
          support_knee_angle: ['support_knee_angle'],
          hip_torsion_angle: ['hip_torsion_angle', 'torso_lateral_tilt'],
        }
        for (const alias of aliases[def.key] ?? []) {
          const raw = metrics[alias]
          if (typeof raw === 'number') {
            // swing_fold_angle 是膝内角，折叠角 = 180 - 内角
            value =
              alias === 'swing_fold_angle' && def.key === 'max_folding_angle'
                ? Math.max(0, 180 - raw)
                : raw
            break
          }
        }
      }

      let level = statusToLevel(entry?.status as BiomechStatusCode | undefined)
      if (level === 'pending' && value !== null) {
        // 无 status 时：用 errorCodes 粗判（仅红/绿）
        const related: Partial<Record<BiomechIndicatorKey, string[]>> = {
          distance_cm: ['ERR_A2_SUPPORT_WIDE', 'ERR_SUPPORT_LATERAL', 'ERR_WARMUP_CLOSE'],
          toe_angle: ['ERR_C2_TOE_POKE'],
          max_folding_angle: ['ERR_B1_STRAIGHT_LEG', 'ERR_B2_SHANK_ONLY', 'ERR_SWING_FOLD'],
          whipping_velocity: ['ERR_FOLLOW_THROUGH'],
          impact_knee_angle: ['ERR_KNEE_STIFF'],
          ankle_rigidity: ['ERR_C1_LOOSE_ANKLE', 'ERR_ANKLE_LOOSE'],
          support_knee_angle: ['ERR_KNEE_STIFF'],
          hip_torsion_angle: ['ERR_TORSO_TILT'],
        }
        const hit = (related[def.key] ?? []).some((c) => (errorCodes ?? []).includes(c))
        level = hit ? 'red' : 'green'
      }

      const frameIndex = resolveExtremeFrame(def.key, entry, scoreDetail, impactFrame)

      return {
        ...def,
        value,
        unit: entry?.unit,
        level,
        frameIndex,
        metaphor: def.metaphors[level],
      }
    })
  }, [indicators, metrics, errorCodes, scoreDetail, impactFrame])

  function handleCardClick(card: (typeof cards)[number]) {
    setActiveKey(card.key)
    if (card.frameIndex == null || !onMetricSeek) return
    onMetricSeek({
      metricKey: card.key,
      frameIndex: card.frameIndex,
      label: card.label,
    })
  }

  async function handleExportSpss() {
    if (isExporting) return
    setIsExporting(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/export/spss_matrix`)
      const contentType = response.headers.get('content-type') || ''
      if (!response.ok) {
        let message = `导出失败（HTTP ${response.status}）`
        if (contentType.includes('application/json')) {
          const data = (await response.json()) as AcademicExportResult
          message = data.message || message
        }
        throw new Error(message)
      }
      if (contentType.includes('application/json')) {
        const data = (await response.json()) as AcademicExportResult
        throw new Error(data.message || '导出失败：后端返回了错误信息而非 CSV')
      }
      const blob = await response.blob()
      const objectUrl = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = objectUrl
      anchor.download = 'AI_Football_Research_Matrix_V3.csv'
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      URL.revokeObjectURL(objectUrl)
      onExportSuccess?.({
        success: true,
        filename: 'AI_Football_Research_Matrix_V3.csv',
        rowCount: Number(response.headers.get('X-Export-Row-Count') || 0) || undefined,
        columnCount: Number(response.headers.get('X-Export-Column-Count') || 0) || undefined,
        message: '✅ V3.1 全数字化科研宽表已下载：AI_Football_Research_Matrix_V3.csv',
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : '导出 SPSS 宽表失败，请检查后端服务'
      onExportError?.(message)
    } finally {
      setIsExporting(false)
    }
  }

  const showNumeric = renderMode !== 'GROUP_A'
  const isCoach = renderMode === 'COACH_CONSOLE'

  return (
    <div className={`flex flex-col gap-3 ${className}`.trim()} aria-label="生物力学指标卡片列表">
      {/* COACH_CONSOLE：顶部 SPSS 宽表导出 */}
      {isCoach && (
        <button
          type="button"
          onClick={() => void handleExportSpss()}
          disabled={isExporting}
          className="group relative flex w-full flex-col items-center justify-center gap-1.5 overflow-hidden rounded-2xl border-2 border-amber-400/50 bg-gradient-to-br from-amber-500/15 via-amber-400/5 to-transparent px-3 py-3.5 text-center shadow-[0_0_24px_rgba(251,191,36,0.16)] transition hover:border-amber-300/80 hover:shadow-[0_0_32px_rgba(251,191,36,0.28)] active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-60"
        >
          <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-amber-400/20 ring-1 ring-amber-300/40">
            {isExporting ? (
              <Loader2 className="h-4 w-4 animate-spin text-amber-200" />
            ) : (
              <FileSpreadsheet className="h-4 w-4 text-amber-200" />
            )}
          </span>
          <span className="text-[11px] font-bold leading-tight text-amber-100">
            {isExporting ? '正在生成宽表…' : '导出 SPSS 标准宽表 (.csv)'}
          </span>
          <span className="text-[9px] leading-tight text-amber-200/60">
            AI_Football_Research_Matrix_V3 · 全数字编码 · MSEM 直入
          </span>
        </button>
      )}

      {/* GROUP_A：大字号全局具身隐喻导语 */}
      {renderMode === 'GROUP_A' && (
        <div className="rounded-xl border border-[color-mix(in_srgb,var(--GREEN_OPTIMAL)_28%,transparent)] bg-[color-mix(in_srgb,var(--GREEN_OPTIMAL)_8%,transparent)] px-3 py-2.5">
          <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold text-[var(--GREEN_OPTIMAL)]">
            <Sparkles className="h-3 w-3" />
            AIGC 具身隐喻
          </div>
          <p className="text-sm font-semibold leading-snug text-slate-100">
            {groupALeadMetaphor ||
              cards.find((c) => c.level === 'red')?.metaphor ||
              cards.find((c) => c.level === 'yellow')?.metaphor ||
              '结束分析后，这里会用孩子听得懂的比喻告诉你下一步怎么练。'}
          </p>
        </div>
      )}

      {/* GROUP_B：离线雷达 / 幽灵骨架 / 空间热力图切换 */}
      {renderMode === 'GROUP_B' && (
        <div className="rounded-xl border border-slate-700/70 bg-slate-900/40 p-2.5">
          <div className="mb-2 flex items-center gap-1 rounded-lg bg-black/25 p-0.5">
            <button
              type="button"
              onClick={() => setGroupBPanel('radar')}
              className={`flex flex-1 items-center justify-center gap-1 rounded-md px-2 py-1.5 text-[10px] font-semibold transition ${
                groupBPanel === 'radar'
                  ? 'bg-[color-mix(in_srgb,var(--GREEN_OPTIMAL)_22%,transparent)] text-[var(--GREEN_OPTIMAL)]'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <RadarIcon className="h-3 w-3" />
              离线雷达图
            </button>
            <button
              type="button"
              onClick={() => setGroupBPanel('ghost')}
              className={`flex flex-1 items-center justify-center gap-1 rounded-md px-2 py-1.5 text-[10px] font-semibold transition ${
                groupBPanel === 'ghost'
                  ? 'bg-[color-mix(in_srgb,var(--GREEN_OPTIMAL)_22%,transparent)] text-[var(--GREEN_OPTIMAL)]'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Activity className="h-3 w-3" />
              幽灵骨架对比
            </button>
            <button
              type="button"
              onClick={() => setGroupBPanel('heatmap')}
              className={`flex flex-1 items-center justify-center gap-1 rounded-md px-2 py-1.5 text-[10px] font-semibold transition ${
                groupBPanel === 'heatmap'
                  ? 'bg-[color-mix(in_srgb,var(--GREEN_OPTIMAL)_22%,transparent)] text-[var(--GREEN_OPTIMAL)]'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <MapPinned className="h-3 w-3" />
              空间热力图
            </button>
          </div>

          {groupBPanel === 'radar' ? (
            <BiomechanicalRadar scores={effectiveRadarScores} compact className="!text-slate-100" />
          ) : groupBPanel === 'ghost' ? (
            <GhostSkeletonCompare primary={effectiveRadarScores} compare={effectiveCompareRadar} />
          ) : (
            <SpatialHeatmap
              heatmapBase64={effectiveHeatmap}
              landingPositions={
                scoreDetail?.spatial_trajectory?.support_rel
                  ? [scoreDetail.spatial_trajectory.support_rel]
                  : []
              }
              title="支撑脚与摆腿时空热力图"
              subtitle="球心 (0,0) · 1 px = 0.5 cm · [t_impact−15, t_impact] 摆腿光流"
              height={240}
              compact
            />
          )}
        </div>
      )}

      {/* COACH_CONSOLE：额外展示空间热力图选项卡式区块 */}
      {isCoach && (
        <SpatialHeatmap
          heatmapBase64={effectiveHeatmap}
          landingPositions={
            scoreDetail?.spatial_trajectory?.support_rel
              ? [scoreDetail.spatial_trajectory.support_rel]
              : []
          }
          title="空间热力图 · 支撑脚 / 摆腿轨迹"
          subtitle="OpenCV 单趟次云图 · 原点=触球球心"
          height={260}
          compact
        />
      )}

      {/* 8 大量纲卡片 */}
      <div className="flex flex-col gap-1.5">
        {cards.map((card) => {
          const tone = TRAFFIC_CLASS[card.level]
          const isAnkle = card.key === 'ankle_rigidity'
          const isActive = activeKey === card.key
          return (
            <button
              key={card.key}
              type="button"
              onClick={() => handleCardClick(card)}
              className={`w-full rounded-xl border px-2.5 py-2 text-left transition hover:brightness-110 active:scale-[0.99] ${tone.border} ${tone.bg} ${tone.glow} ${
                isActive ? 'ring-1 ring-white/25' : ''
              }`}
              title={
                card.frameIndex != null
                  ? `点击定位到极值帧 #${card.frameIndex}`
                  : '暂无极值帧，完成分析后可跳转'
              }
            >
              <div className="flex items-start gap-2">
                <span className={`mt-0.5 inline-flex flex-shrink-0 ${tone.text}`}>
                  {isAnkle && card.level === 'green' ? (
                    <span className="text-base leading-none" aria-label="脚踝已锁紧">
                      🛡️
                    </span>
                  ) : isAnkle && card.level === 'red' ? (
                    <span className="text-base leading-none" aria-label="脚踝未锁紧">
                      ⚠️
                    </span>
                  ) : (
                    <card.Icon className="h-3.5 w-3.5" />
                  )}
                </span>

                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-[11px] font-medium text-slate-200">{card.label}</span>
                    <span
                      className={`inline-flex flex-shrink-0 items-center gap-1 rounded-md px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wide ${tone.text}`}
                    >
                      <span className={`h-1.5 w-1.5 rounded-full ${tone.dot}`} />
                      {tone.label}
                    </span>
                  </div>

                  {showNumeric ? (
                    <div className={`mt-1 text-sm font-bold tabular-nums ${tone.text}`}>
                      {card.format(card.value, card.unit)}
                      {isAnkle && card.level === 'green' && (
                        <span className="ml-1.5 text-[10px] font-semibold opacity-80">已锁紧</span>
                      )}
                      {isAnkle && card.level === 'red' && (
                        <span className="ml-1.5 inline-flex items-center gap-0.5 text-[10px] font-semibold opacity-90">
                          <AlertTriangle className="h-3 w-3" />
                          未锁紧
                        </span>
                      )}
                    </div>
                  ) : (
                    <p className="mt-1.5 text-[12px] font-semibold leading-snug text-slate-100">
                      {card.metaphor}
                    </p>
                  )}

                  {card.frameIndex != null && (
                    <p className="mt-1 text-[9px] text-slate-500">
                      极值帧 #{card.frameIndex}
                      {onMetricSeek ? ' · 点击跳转视频' : ''}
                    </p>
                  )}
                </div>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}

/** GROUP_B「幽灵骨架对比」：标准模型剪影 + 学员雷达差速可视化 */
function GhostSkeletonCompare({
  primary,
  compare,
}: {
  primary?: Quantified5dScores | null
  compare?: Quantified5dScores | null
}) {
  const pNorm = normalizeRadarScores(primary)
  const cNorm = compare ? normalizeRadarScores(compare) : null
  const dims = [
    { key: 'approach_rhythm' as const, label: '助跑' },
    { key: 'support_stability' as const, label: '支撑' },
    { key: 'backswing_folding' as const, label: '折叠' },
    { key: 'ankle_rigidity' as const, label: '锁踝' },
    { key: 'whipping_velocity' as const, label: '鞭打' },
  ]
  const showPrimary = primary != null

  return (
    <div className="space-y-2">
      <div className="relative mx-auto flex h-36 w-full max-w-[200px] items-center justify-center overflow-hidden rounded-xl bg-gradient-to-br from-zinc-950 via-[#0a1612] to-black">
        <div className="pointer-events-none absolute inset-0 opacity-35 [background:radial-gradient(ellipse_at_center,rgba(52,211,153,0.2),transparent_65%)]" />
        <svg viewBox="0 0 200 260" className="absolute h-[90%] w-auto opacity-35">
          <circle cx="100" cy="48" r="16" fill="none" stroke="#34d399" strokeWidth="2.5" />
          <line x1="100" y1="64" x2="100" y2="130" stroke="#34d399" strokeWidth="3" />
          <line x1="100" y1="80" x2="60" y2="120" stroke="#34d399" strokeWidth="2.5" />
          <line x1="100" y1="80" x2="145" y2="125" stroke="#34d399" strokeWidth="2.5" />
          <line x1="100" y1="130" x2="70" y2="200" stroke="#34d399" strokeWidth="3" />
          <line x1="100" y1="130" x2="140" y2="175" stroke="#34d399" strokeWidth="3" />
          <line x1="140" y1="175" x2="165" y2="210" stroke="#34d399" strokeWidth="2.5" />
        </svg>
        <svg viewBox="0 0 200 260" className="relative h-[90%] w-auto drop-shadow-[0_0_12px_rgba(125,211,252,0.35)]">
          <circle cx="100" cy="48" r="16" fill="none" stroke="#7dd3fc" strokeWidth="2.5" />
          <line x1="100" y1="64" x2="100" y2="130" stroke="#7dd3fc" strokeWidth="3" />
          <line x1="100" y1="80" x2="55" y2="118" stroke="#7dd3fc" strokeWidth="2.5" />
          <line x1="100" y1="80" x2="148" y2="128" stroke="#7dd3fc" strokeWidth="2.5" />
          <line x1="100" y1="130" x2="68" y2="205" stroke="#7dd3fc" strokeWidth="3" />
          <line x1="100" y1="130" x2="135" y2="168" stroke="#7dd3fc" strokeWidth="3" />
          <line x1="135" y1="168" x2="158" y2="205" stroke="#38bdf8" strokeWidth="2.5" />
        </svg>
        <span className="absolute bottom-1.5 left-1/2 -translate-x-1/2 rounded bg-black/60 px-1.5 py-0.5 text-[8px] text-emerald-300/90">
          绿=标准 · 青=学员
        </span>
      </div>

      <div className="grid grid-cols-5 gap-1">
        {dims.map((d) => {
          const p = showPrimary ? pNorm[d.key] : null
          const c = cNorm ? cNorm[d.key] : null
          return (
            <div key={d.key} className="rounded-lg bg-black/25 px-1 py-1.5 text-center">
              <p className="text-[8px] text-slate-500">{d.label}</p>
              <p className="text-[10px] font-bold tabular-nums text-sky-300">
                {p != null ? Math.round(p) : '—'}
              </p>
              {c != null && (
                <p className="text-[8px] tabular-nums text-emerald-400/80">vs {Math.round(c)}</p>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
