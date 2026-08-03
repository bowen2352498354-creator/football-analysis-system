import { useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import * as echarts from 'echarts/core'
import { LineChart } from 'echarts/charts'
import {
  GridComponent,
  MarkAreaComponent,
  MarkLineComponent,
  TooltipComponent,
  AxisPointerComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsType } from 'echarts/core'
import { Camera, Clapperboard, Loader2, Pause, PenLine, Play, Trash2 } from 'lucide-react'
import type { JointHighlight } from '../types'
import JointHighlightOverlay, { pickFocusHighlight } from './JointHighlightOverlay'
import TelestrationCanvas, { type TelestrationCanvasHandle } from './TelestrationCanvas'

/** 子弹时间：触及错误绝对秒的触发阈值 */
const BULLET_TIME_THRESHOLD_SEC = 0.1
/** 子弹时间定格驻留（毫秒）：呼吸圈 + Emoji 展示后继续慢放 */
const BULLET_TIME_HOLD_MS = 3000
/** 离开错误窗多远后允许下一轮循环再触发（防止阈值内反复 pause） */
const BULLET_TIME_REARM_GAP_SEC = 0.5

const API_BASE_URL = 'http://localhost:8000'

/** 实验 A 组干预规程：强制 0.5x，延长错误动作视觉驻留 */
const INTERVENTION_PLAYBACK_RATE = 0.5

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('image load failed'))
    img.src = src
  })
}

function triggerJpegDownload(dataUrl: string, filename: string) {
  const anchor = document.createElement('a')
  anchor.href = dataUrl
  anchor.download = filename
  anchor.rel = 'noopener'
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
}

echarts.use([
  LineChart,
  GridComponent,
  MarkAreaComponent,
  MarkLineComponent,
  TooltipComponent,
  AxisPointerComponent,
  CanvasRenderer,
])

/** 单帧角速度采样：优先用 absolute_timestamp（秒）与视频 currentTime 对齐 */
export interface SyncVelocityPoint {
  frame_index: number
  omega: number
  /** 原视频绝对秒；缺省时由 (seriesFrameOffset + frame_index) / fps 推导 */
  absolute_timestamp?: number
}

/** 五段动作切片（图表 markArea 色块） */
export interface ActionPhaseSlice {
  key: 'approach' | 'support' | 'fold' | 'impact' | 'follow'
  label: string
  startFrame: number
  endFrame: number
  color: string
}

export interface SynchronizedVideoWorkspaceProps {
  /** HTML5 Video 源（本地 blob URL 或可播 URL）；缺省时仅展示 overlay/children */
  videoSrc?: string | null
  /** 摆动腿小腿角速度时序（deg/s）；窗口模式下 frame_index 为 0..N-1 */
  velocitySeries?: SyncVelocityPoint[] | null
  /**
   * 后端返回的 Action ROI 绝对秒数组，与 velocitySeries 等长。
   * ECharts X 轴绑定此数组；点击 `params.dataIndex` → video.currentTime。
   */
  absoluteTimestamps?: number[] | null
  /** 触球锁帧索引（绝对帧）；缺省时取 |ω| 峰值帧 */
  tImpact?: number | null
  /**
   * 触球点在当前波形窗口内的索引（优先用于 markLine）。
   * 对应后端 `impact_index_in_window`，正常为 30。
   */
  impactIndexInWindow?: number | null
  /**
   * velocitySeries[0] 对应的绝对视频帧号 = action_roi.start。
   * 仅在缺少 absoluteTimestamps 时作为回退：currentTime = (offset + i) / fps。
   */
  seriesFrameOffset?: number
  /** 自定义阶段切片；缺省时按 t_impact 自动切分五段 */
  phases?: ActionPhaseSlice[] | null
  /** 视频帧率，默认 30（无 absoluteTimestamps 时的回退换算） */
  fps?: number
  /** 点击波形跳转后，若视频暂停则自动 play（默认 true） */
  autoPlayOnSeek?: boolean
  /** 视频视口内叠层（实时推理帧 / HUD） */
  children?: ReactNode
  overlay?: ReactNode
  /** 分析进行中：优先显示 children 推理画面，隐藏本地原片 */
  preferLiveOverlay?: boolean
  className?: string
  title?: string
  subtitle?: string
  /**
   * 来自 MetricCardList 的外部极值帧 Seek。
   * token 递增时即使 frameIndex 相同也会重新定格。
   * frameIndex 为绝对视频帧。
   */
  externalSeek?: { frameIndex: number; token: number; label?: string } | null
  /**
   * 【V3.1 Sprint 3】Hudl 风格教练手绘电烙铁。
   * 默认开启；关闭后不渲染 Canvas 覆盖层。
   */
  enableTelestration?: boolean
  /** 关联 Attempt ID：保存批注时一并上传后端，写入诊断处方附件 */
  attemptId?: string | null
  /** 学号（可选），用于后端归档命名 */
  studentNumber?: string | null
  onTelestrationSaved?: (ok: boolean, message: string) => void
  /**
   * 具身隐喻：后端 scoreDetail.joint_highlights（T0 关节像素/归一化坐标 + 红绿灯）。
   * 在诊断帧附近或暂停回放时由透明 Canvas 叠层渲染。
   */
  jointHighlights?: JointHighlight[] | null
}

const DEFAULT_FPS = 30

const PHASE_SWATCH: Record<ActionPhaseSlice['key'], string> = {
  approach: '#38bdf8',
  support: '#a78bfa',
  fold: '#fbbf24',
  impact: '#ef4444',
  follow: '#10b981',
}


/** 相对 t_impact 自动切分：[助跑][支撑][折叠][触球][随摆] */
export function buildDefaultPhases(frameCount: number, tImpact: number): ActionPhaseSlice[] {
  const n = Math.max(1, frameCount)
  const t = Math.max(0, Math.min(n - 1, Math.round(tImpact)))
  const approachEnd = Math.max(0, Math.floor(t * 0.45))
  const supportEnd = Math.max(approachEnd, Math.floor(t * 0.7))
  const foldEnd = Math.max(supportEnd, Math.max(0, t - 1))
  const impactEnd = Math.min(n - 1, t + 2)

  return [
    {
      key: 'approach',
      label: '助跑',
      startFrame: 0,
      endFrame: approachEnd,
      color: 'rgba(56, 189, 248, 0.14)',
    },
    {
      key: 'support',
      label: '支撑',
      startFrame: approachEnd,
      endFrame: supportEnd,
      color: 'rgba(167, 139, 250, 0.14)',
    },
    {
      key: 'fold',
      label: '折叠',
      startFrame: supportEnd,
      endFrame: foldEnd,
      color: 'rgba(251, 191, 36, 0.16)',
    },
    {
      key: 'impact',
      label: '触球 t_impact',
      startFrame: foldEnd,
      endFrame: impactEnd,
      color: 'rgba(239, 68, 68, 0.20)',
    },
    {
      key: 'follow',
      label: '随摆',
      startFrame: impactEnd,
      endFrame: n - 1,
      color: 'rgba(16, 185, 129, 0.14)',
    },
  ]
}

function sanitizeTimestamps(raw: number[] | null | undefined, length: number): number[] {
  if (!Array.isArray(raw) || length <= 0) return []
  const out: number[] = []
  for (let i = 0; i < length; i += 1) {
    const v = Number(raw[i])
    if (!Number.isFinite(v)) return []
    out.push(v)
  }
  return out.length === length ? out : []
}

function sanitizeSeries(
  raw: SyncVelocityPoint[] | null | undefined,
  absoluteTimestamps?: number[] | null,
  seriesFrameOffset = 0,
  fps = DEFAULT_FPS,
): SyncVelocityPoint[] {
  if (!Array.isArray(raw) || raw.length === 0) return []
  const externalTs = sanitizeTimestamps(absoluteTimestamps, raw.length)
  const rate = fps > 0 ? fps : DEFAULT_FPS
  const offset = Number.isFinite(seriesFrameOffset) ? Math.max(0, Math.round(seriesFrameOffset)) : 0
  const out: SyncVelocityPoint[] = []
  for (let i = 0; i < raw.length; i += 1) {
    const row = raw[i]
    if (!row || typeof row !== 'object') continue
    const frame =
      typeof row.frame_index === 'number' && Number.isFinite(row.frame_index)
        ? Math.max(0, Math.round(row.frame_index))
        : i
    const omega = Number(row.omega)
    const fromPoint = Number(row.absolute_timestamp)
    const fromArray = externalTs[i]
    const absolute_timestamp = Number.isFinite(fromPoint)
      ? fromPoint
      : Number.isFinite(fromArray)
        ? fromArray
        : (offset + frame) / rate
    out.push({
      frame_index: frame,
      omega: Number.isFinite(omega) ? omega : 0,
      absolute_timestamp,
    })
  }
  out.sort((a, b) => a.frame_index - b.frame_index)
  return out
}

/** 将 video.currentTime 映射到最近的波形点 dataIndex；越界返回 null */
function nearestDataIndexByTime(
  series: SyncVelocityPoint[],
  currentTimeSec: number,
): number | null {
  if (!series.length || !Number.isFinite(currentTimeSec)) return null
  const first = series[0].absolute_timestamp ?? 0
  const last = series[series.length - 1].absolute_timestamp ?? first
  if (currentTimeSec < first - 1e-3 || currentTimeSec > last + 1e-3) return null
  let bestIdx = 0
  let bestDist = Infinity
  for (let i = 0; i < series.length; i += 1) {
    const t = series[i].absolute_timestamp
    if (typeof t !== 'number' || !Number.isFinite(t)) continue
    const dist = Math.abs(t - currentTimeSec)
    if (dist < bestDist) {
      bestDist = dist
      bestIdx = i
    }
  }
  return bestIdx
}

function resolveTImpact(series: SyncVelocityPoint[], tImpact: number | null | undefined): number {
  if (typeof tImpact === 'number' && Number.isFinite(tImpact) && series.length > 0) {
    const maxFrame = series[series.length - 1].frame_index
    return Math.max(0, Math.min(maxFrame, Math.round(tImpact)))
  }
  if (series.length === 0) return 0
  let best = series[0]
  for (const p of series) {
    if (Math.abs(p.omega) > Math.abs(best.omega)) best = p
  }
  return best.frame_index
}

/**
 * Kinovea 风格：视频 ↔ ECharts 角速度时序毫秒级双向联动工作区。
 * - X 轴绑定后端 absolute_timestamps（原视频绝对秒）
 * - 图表 click → video.currentTime = absolute_timestamps[dataIndex]
 * - video timeupdate → 按绝对秒就近对齐游标
 */
export default function SynchronizedVideoWorkspace({
  videoSrc = null,
  velocitySeries = null,
  absoluteTimestamps = null,
  tImpact = null,
  impactIndexInWindow = null,
  seriesFrameOffset = 0,
  phases = null,
  fps = DEFAULT_FPS,
  autoPlayOnSeek = true,
  children,
  overlay,
  preferLiveOverlay = false,
  className = '',
  title = 'Video Workspace',
  subtitle = 'Kinovea 毫秒级联动 · 小腿角速度时序',
  externalSeek = null,
  enableTelestration = true,
  attemptId = null,
  studentNumber = null,
  onTelestrationSaved,
  jointHighlights = null,
}: SynchronizedVideoWorkspaceProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const stageRef = useRef<HTMLDivElement | null>(null)
  const telestrationRef = useRef<TelestrationCanvasHandle>(null)
  const chartHostRef = useRef<HTMLDivElement | null>(null)
  const chartRef = useRef<EChartsType | null>(null)
  const scrubbingRef = useRef(false)
  const playheadFrameRef = useRef(0)
  const seriesRef = useRef<SyncVelocityPoint[]>([])
  const fpsRef = useRef(DEFAULT_FPS)
  const offsetRef = useRef(0)
  const autoPlayOnSeekRef = useRef(autoPlayOnSeek)
  /** 子弹时间：本轮循环内是否已对错误点 pause 过（防阈值内无限暂停卡死） */
  const hasPausedForErrorRef = useRef(false)
  const bulletTimeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const bulletFreezeActiveRef = useRef(false)
  const focusHighlightRef = useRef<JointHighlight | null>(null)
  const showNativeVideoRef = useRef(false)

  /** React 无法用 HTML 属性设倍速；在 metadata/canplay 等时机强制劫持 */
  const forceInterventionPlaybackRate = () => {
    const video = videoRef.current
    if (!video) return
    video.playbackRate = INTERVENTION_PLAYBACK_RATE
    video.defaultPlaybackRate = INTERVENTION_PLAYBACK_RATE
  }

  const [isPlaying, setIsPlaying] = useState(false)
  const [playheadFrame, setPlayheadFrame] = useState(0)
  /** 游标仅在 Action ROI 时间窗内显示；越界隐藏 */
  const [playheadVisible, setPlayheadVisible] = useState(true)
  const playheadVisibleRef = useRef(true)
  /** 子弹时间定格中：驱动 Canvas 呼吸圈 + Emoji */
  const [bulletFreezeActive, setBulletFreezeActive] = useState(false)
  const [seekBadge, setSeekBadge] = useState<string | null>(null)
  const [penActive, setPenActive] = useState(false)
  const [isSavingAnnotation, setIsSavingAnnotation] = useState(false)
  const [annotationHint, setAnnotationHint] = useState<string | null>(null)

  const safeOffset = Number.isFinite(seriesFrameOffset) ? Math.max(0, Math.round(seriesFrameOffset)) : 0
  const safeFps = fps > 0 ? fps : DEFAULT_FPS
  const series = useMemo(
    () => sanitizeSeries(velocitySeries, absoluteTimestamps, safeOffset, safeFps),
    [velocitySeries, absoluteTimestamps, safeOffset, safeFps],
  )
  const impactFrame = useMemo(() => {
    if (typeof impactIndexInWindow === 'number' && Number.isFinite(impactIndexInWindow)) {
      const maxIdx = series.length > 0 ? series.length - 1 : 0
      return Math.max(0, Math.min(maxIdx, Math.round(impactIndexInWindow)))
    }
    // resolveTImpact 返回 frame_index；窗口模式下通常等于 dataIndex
    const resolved = resolveTImpact(series, tImpact)
    const byFrame = series.findIndex((p) => p.frame_index === resolved)
    return byFrame >= 0 ? byFrame : resolved
  }, [series, tImpact, impactIndexInWindow])
  const frameCount = series.length
  const impactTimestamp =
    series[impactFrame]?.absolute_timestamp ??
    (safeOffset + impactFrame) / safeFps

  seriesRef.current = series
  fpsRef.current = safeFps
  offsetRef.current = safeOffset
  autoPlayOnSeekRef.current = autoPlayOnSeek

  const resolvedPhases = useMemo(() => {
    if (Array.isArray(phases) && phases.length > 0) return phases
    if (frameCount <= 0) return []
    return buildDefaultPhases(frameCount, impactFrame)
  }, [phases, frameCount, impactFrame])

  /** 强制跳转视频到绝对秒；可选自动播放 */
  const seekToTimestamp = (targetTimestamp: number, opts?: { autoPlay?: boolean }) => {
    const video = videoRef.current
    if (!Number.isFinite(targetTimestamp)) return
    const nextTime = Math.max(0, targetTimestamp)
    if (video) {
      if (Number.isFinite(video.duration) && video.duration > 0) {
        video.currentTime = Math.min(video.duration, nextTime)
      } else {
        video.currentTime = nextTime
      }
      video.playbackRate = INTERVENTION_PLAYBACK_RATE
      video.defaultPlaybackRate = INTERVENTION_PLAYBACK_RATE
      const shouldPlay = opts?.autoPlay ?? autoPlayOnSeekRef.current
      if (shouldPlay && video.paused) {
        void video.play().catch(() => {
          /* 自动播放被策略拦截时忽略 */
        })
      }
    }
    const idx = nearestDataIndexByTime(seriesRef.current, nextTime)
    if (idx != null) {
      playheadFrameRef.current = idx
      setPlayheadFrame(idx)
      playheadVisibleRef.current = true
      setPlayheadVisible(true)
    }
  }

  /** dataIndex → absolute_timestamps[i] → video.currentTime */
  const seekToDataIndex = (dataIndex: number, opts?: { autoPlay?: boolean }) => {
    const current = seriesRef.current
    if (!current.length || !Number.isFinite(dataIndex)) return
    const idx = Math.max(0, Math.min(current.length - 1, Math.round(dataIndex)))
    const ts =
      current[idx]?.absolute_timestamp ??
      (offsetRef.current + (current[idx]?.frame_index ?? idx)) / fpsRef.current
    playheadFrameRef.current = idx
    setPlayheadFrame(idx)
    playheadVisibleRef.current = true
    setPlayheadVisible(true)
    seekToTimestamp(ts, opts)
  }

  /** 兼容旧调用：localIndex 视为 dataIndex */
  const seekToLocalFrame = (localIndex: number) => {
    seekToDataIndex(localIndex, { autoPlay: false })
    setIsPlaying(false)
    videoRef.current?.pause()
  }

  /** 外部传入绝对帧 → 绝对秒再 seek */
  const seekToAbsoluteFrame = (absoluteFrame: number) => {
    const rate = fpsRef.current
    const abs = Math.round(absoluteFrame)
    // 优先在序列中找匹配帧，否则用 fps 换算
    const hit = seriesRef.current.findIndex(
      (p) => p.frame_index + offsetRef.current === abs || p.frame_index === abs,
    )
    if (hit >= 0) {
      seekToDataIndex(hit, { autoPlay: false })
    } else {
      seekToTimestamp(abs / rate, { autoPlay: false })
    }
    videoRef.current?.pause()
    setIsPlaying(false)
  }

  // MetricCardList → 物理极值帧 Seek；报告完成后定格触球瞬间
  useEffect(() => {
    if (!externalSeek) return
    seekToAbsoluteFrame(externalSeek.frameIndex)
    if (externalSeek.label) {
      setSeekBadge(`${externalSeek.label} · F#${externalSeek.frameIndex}`)
    }
  }, [externalSeek])

  // 报告带回触球索引后，对齐射门瞬间并以 0.5x 自动循环（A 组干预驻留）
  useEffect(() => {
    if (series.length < 2) return
    if (typeof impactIndexInWindow === 'number' && Number.isFinite(impactIndexInWindow)) {
      seekToDataIndex(impactIndexInWindow, { autoPlay: true })
      setIsPlaying(true)
      setSeekBadge(`射门瞬间 · 窗内 #${Math.round(impactIndexInWindow)}`)
      return
    }
    if (typeof tImpact === 'number' && Number.isFinite(tImpact)) {
      const rate = fpsRef.current
      const abs = Math.round(tImpact)
      const hit = seriesRef.current.findIndex(
        (p) => p.frame_index + offsetRef.current === abs || p.frame_index === abs,
      )
      if (hit >= 0) {
        seekToDataIndex(hit, { autoPlay: true })
      } else {
        seekToTimestamp(abs / rate, { autoPlay: true })
      }
      setIsPlaying(true)
      setSeekBadge(`射门瞬间 · F#${Math.round(tImpact)}`)
    }
  }, [tImpact, impactIndexInWindow, series.length])

  // 视频 loadeddata：强制 seek 到触球绝对秒（红色虚线），避免停在第 0 帧助跑
  useEffect(() => {
    const video = videoRef.current
    if (!video || !videoSrc) return

    const seekImpactOnReady = () => {
      const current = seriesRef.current
      let targetTs: number | null = null
      let badge = '射门瞬间'
      if (
        typeof impactIndexInWindow === 'number' &&
        Number.isFinite(impactIndexInWindow) &&
        current[Math.round(impactIndexInWindow)]
      ) {
        const idx = Math.round(impactIndexInWindow)
        targetTs = current[idx].absolute_timestamp ?? null
        badge = `射门瞬间 · ${targetTs != null ? `${targetTs.toFixed(3)}s` : `窗内 #${idx}`}`
      } else if (typeof tImpact === 'number' && Number.isFinite(tImpact)) {
        targetTs = Math.round(tImpact) / fpsRef.current
        badge = `射门瞬间 · F#${Math.round(tImpact)}`
      }
      if (targetTs == null || !Number.isFinite(targetTs)) return
      // A 组干预：对齐触球瞬间后以 0.5x 自动循环，而非定格暂停
      forceInterventionPlaybackRate()
      seekToTimestamp(targetTs, { autoPlay: true })
      void video.play().catch(() => {
        /* muted + autoPlay 仍可能被策略拦截 */
      })
      setIsPlaying(true)
      setSeekBadge(badge)
    }

    if (video.readyState >= 2) {
      seekImpactOnReady()
    }
    video.addEventListener('loadeddata', seekImpactOnReady)
    return () => {
      video.removeEventListener('loadeddata', seekImpactOnReady)
    }
  }, [videoSrc, tImpact, impactIndexInWindow, series])

  /** 从 ECharts 事件提取 dataIndex（优先），供 absolute_timestamps 查表 */
  const dataIndexFromChartEvent = (params: unknown): number | null => {
    const current = seriesRef.current
    if (!current.length) return null
    const p = params as {
      dataIndex?: number
      data?: unknown
      value?: unknown
      batch?: Array<{ dataIndex?: number }>
    }
    if (Array.isArray(p?.batch) && p.batch.length > 0) {
      const idx = p.batch[0]?.dataIndex
      if (typeof idx === 'number' && current[idx]) return idx
    }
    if (typeof p?.dataIndex === 'number' && current[p.dataIndex]) {
      return p.dataIndex
    }
    // 回退：data[0] 为绝对秒时，就近映射 dataIndex
    const data = p?.data
    if (Array.isArray(data) && typeof data[0] === 'number') {
      return nearestDataIndexByTime(current, data[0])
    }
    if (typeof p?.value === 'number' && Number.isFinite(p.value)) {
      return nearestDataIndexByTime(current, p.value)
    }
    return null
  }

  // 初始化 / 销毁 ECharts 实例；cleanup 中 off('click') 防止泄漏
  useEffect(() => {
    const host = chartHostRef.current
    if (!host) return
    const chart = echarts.init(host, undefined, { renderer: 'canvas' })
    chartRef.current = chart

    const onHighlight = (params: unknown) => {
      const dataIndex = dataIndexFromChartEvent(params)
      if (dataIndex === null) return
      scrubbingRef.current = true
      seekToDataIndex(dataIndex, { autoPlay: false })
      videoRef.current?.pause()
      setIsPlaying(false)
    }
    const onClick = (params: unknown) => {
      const dataIndex = dataIndexFromChartEvent(params)
      if (dataIndex === null) return
      scrubbingRef.current = true
      // 精准穿透：absolute_timestamps[dataIndex] → video.currentTime
      seekToDataIndex(dataIndex, { autoPlay: autoPlayOnSeekRef.current })
      window.setTimeout(() => {
        scrubbingRef.current = false
      }, 120)
    }
    const onGlobalOut = () => {
      scrubbingRef.current = false
    }

    chart.on('highlight', onHighlight)
    chart.on('click', onClick)
    chart.getZr().on('globalout', onGlobalOut)

    chart.getZr().on('mousedown', () => {
      scrubbingRef.current = true
    })
    chart.getZr().on('mouseup', () => {
      window.setTimeout(() => {
        scrubbingRef.current = false
      }, 80)
    })
    chart.getZr().on('mousemove', (e: { offsetX?: number }) => {
      if (!scrubbingRef.current || seriesRef.current.length === 0) return
      const pointInPixel = [e.offsetX ?? 0, 0]
      if (!chart.containPixel('grid', pointInPixel)) return
      const pointInGrid = chart.convertFromPixel({ seriesIndex: 0 }, pointInPixel)
      if (!pointInGrid || !Number.isFinite(pointInGrid[0])) return
      // X 轴已是绝对秒：就近 dataIndex 后 seek
      const idx = nearestDataIndexByTime(seriesRef.current, pointInGrid[0])
      if (idx == null) return
      seekToDataIndex(idx, { autoPlay: false })
      videoRef.current?.pause()
      setIsPlaying(false)
    })

    const onResize = () => chart.resize()
    window.addEventListener('resize', onResize)

    return () => {
      window.removeEventListener('resize', onResize)
      chart.off('highlight', onHighlight)
      chart.off('click', onClick)
      try {
        chart.getZr().off('globalout', onGlobalOut)
      } catch {
        /* dispose 前 zrender 可能已释放 */
      }
      chart.dispose()
      chartRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const impactMarkLine = {
    xAxis: impactTimestamp,
    label: {
      formatter: '触球瞬间 (t_impact)',
      position: 'insideEndTop' as const,
      color: '#f87171',
      fontSize: 11,
      fontWeight: 600 as const,
    },
    lineStyle: {
      color: '#ef4444',
      width: 2,
      type: 'dashed' as const,
      shadowBlur: 6,
      shadowColor: 'rgba(239,68,68,0.45)',
    },
  }

  // 刷新图表 option（含阶段色块、t_impact 锚线、播放游标）
  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return

    if (series.length < 2) {
      chart.clear()
      chart.setOption({
        backgroundColor: 'transparent',
        title: {
          text: '等待摆动腿小腿角速度时序…',
          left: 'center',
          top: 'middle',
          textStyle: { color: 'rgba(148,163,184,0.55)', fontSize: 12, fontWeight: 400 },
        },
      })
      return
    }

    const timeAt = (idx: number) =>
      series[Math.max(0, Math.min(series.length - 1, Math.round(idx)))]?.absolute_timestamp ??
      (safeOffset + idx) / safeFps

    const markAreaData = resolvedPhases.map((phase) => [
      {
        name: phase.label,
        xAxis: timeAt(phase.startFrame),
        itemStyle: { color: phase.color },
        label: {
          show: true,
          position: 'insideTop',
          color: 'rgba(226,232,240,0.55)',
          fontSize: 10,
          formatter: phase.label,
        },
      },
      { xAxis: timeAt(phase.endFrame) },
    ])

    const xMin = series[0].absolute_timestamp ?? 0
    const xMax = series[series.length - 1].absolute_timestamp ?? xMin
    const playheadTs =
      series[playheadFrameRef.current]?.absolute_timestamp ?? timeAt(playheadFrameRef.current)

    chart.setOption(
      {
        backgroundColor: 'transparent',
        animation: false,
        grid: { top: 28, right: 16, bottom: 28, left: 48 },
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'line', snap: true },
          backgroundColor: 'rgba(15,23,42,0.92)',
          borderColor: 'rgba(148,163,184,0.25)',
          textStyle: { color: '#e2e8f0', fontSize: 11 },
          formatter: (items: unknown) => {
            const arr = Array.isArray(items) ? items : [items]
            const first = arr[0] as {
              data?: [number, number]
              dataIndex?: number
              axisValue?: number | string
            }
            const tSec =
              Array.isArray(first?.data) && typeof first.data[0] === 'number'
                ? first.data[0]
                : Number(first?.axisValue)
            const omega =
              Array.isArray(first?.data) && typeof first.data[1] === 'number' ? first.data[1] : NaN
            const di =
              typeof first?.dataIndex === 'number' ? first.dataIndex : nearestDataIndexByTime(series, tSec)
            const localIdx = di != null ? di : '—'
            const absFrame =
              typeof di === 'number' ? Math.round(safeOffset + (series[di]?.frame_index ?? di)) : '—'
            const omegaText = Number.isFinite(omega) ? `${omega.toFixed(1)} deg/s` : '—'
            const tText = Number.isFinite(tSec) ? `${tSec.toFixed(3)} s` : '—'
            return `t = ${tText} · idx #${localIdx} · abs F#${absFrame}<br/>ω = ${omegaText}`
          },
        },
        axisPointer: {
          link: [{ xAxisIndex: 'all' }],
          label: { backgroundColor: '#1e293b' },
        },
        xAxis: {
          type: 'value',
          name: 't (s)',
          nameTextStyle: { color: 'rgba(148,163,184,0.45)', fontSize: 10 },
          min: xMin,
          max: xMax,
          axisLabel: {
            color: 'rgba(148,163,184,0.45)',
            fontSize: 10,
            formatter: (v: number) => (Number.isFinite(v) ? v.toFixed(2) : ''),
          },
          splitLine: { show: false },
          axisLine: { lineStyle: { color: 'rgba(51,65,85,0.9)' } },
        },
        yAxis: {
          type: 'value',
          name: 'deg/s',
          nameTextStyle: { color: 'rgba(148,163,184,0.45)', fontSize: 10 },
          axisLabel: { color: 'rgba(148,163,184,0.45)', fontSize: 10 },
          splitLine: { lineStyle: { color: 'rgba(51,65,85,0.55)', type: 'dashed' } },
          axisLine: { show: false },
        },
        series: [
          {
            id: 'shank-omega',
            name: '摆动腿小腿角速度',
            type: 'line',
            showSymbol: false,
            smooth: true,
            lineStyle: {
              width: 2.5,
              color: {
                type: 'linear',
                x: 0,
                y: 0,
                x2: 1,
                y2: 0,
                colorStops: [
                  { offset: 0, color: '#fef08a' },
                  { offset: 0.45, color: '#facc15' },
                  { offset: 1, color: '#eab308' },
                ],
              },
            },
            itemStyle: { color: '#facc15' },
            areaStyle: {
              color: {
                type: 'linear',
                x: 0,
                y: 0,
                x2: 0,
                y2: 1,
                colorStops: [
                  { offset: 0, color: 'rgba(250, 204, 21, 0.32)' },
                  { offset: 1, color: 'rgba(250, 204, 21, 0.02)' },
                ],
              },
            },
            emphasis: { focus: 'series' },
            // X = absolute_timestamps（秒），与 video.currentTime 同源
            data: series.map((p) => [p.absolute_timestamp ?? 0, p.omega]),
            markArea: {
              silent: true,
              data: markAreaData,
            },
            markLine: {
              symbol: 'none',
              animation: false,
              data: [
                impactMarkLine,
                {
                  xAxis: playheadTs,
                  label: { show: false },
                  lineStyle: {
                    color: 'rgba(125,211,252,0.95)',
                    width: 1.5,
                    type: 'dashed',
                  },
                },
              ],
            },
          },
        ],
      },
      { notMerge: true },
    )
  }, [series, resolvedPhases, impactFrame, impactTimestamp, safeFps, safeOffset])

  // 播放游标：仅更新 markLine，避免整表重绘；越界时只保留触球锚线
  useEffect(() => {
    const chart = chartRef.current
    if (!chart || series.length < 2) return
    const playheadTs =
      series[playheadFrame]?.absolute_timestamp ??
      (safeOffset + playheadFrame) / safeFps
    const markData = playheadVisible
      ? [
          impactMarkLine,
          {
            xAxis: playheadTs,
            label: { show: false },
            lineStyle: {
              color: 'rgba(125,211,252,0.95)',
              width: 1.5,
              type: 'dashed',
            },
          },
        ]
      : [impactMarkLine]
    chart.setOption({
      series: [
        {
          id: 'shank-omega',
          markLine: {
            data: markData,
          },
        },
      ],
    })
  }, [playheadFrame, playheadVisible, impactTimestamp, series, safeOffset, safeFps])

  const showNativeVideo = Boolean(videoSrc) && !preferLiveOverlay
  const focusHighlight = useMemo(
    () => pickFocusHighlight(jointHighlights),
    [jointHighlights],
  )
  showNativeVideoRef.current = showNativeVideo
  focusHighlightRef.current = focusHighlight
  bulletFreezeActiveRef.current = bulletFreezeActive

  const clearBulletTimeTimer = () => {
    if (bulletTimeTimerRef.current != null) {
      clearTimeout(bulletTimeTimerRef.current)
      bulletTimeTimerRef.current = null
    }
  }

  const resetBulletTimeCycle = () => {
    clearBulletTimeTimer()
    hasPausedForErrorRef.current = false
    bulletFreezeActiveRef.current = false
    setBulletFreezeActive(false)
  }

  // 视频源 / 焦点错误切换：重置子弹时间闩锁，避免沿用上一轮 hasPausedForError
  useEffect(() => {
    resetBulletTimeCycle()
    return () => {
      clearBulletTimeTimer()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- 仅在源/焦点时刻变化时重置
  }, [videoSrc, focusHighlight?.error_timestamp_sec, focusHighlight?.joint_name])

  // 切回录制/实时推理：立刻清画布并解除定格
  useEffect(() => {
    if (!showNativeVideo) {
      resetBulletTimeCycle()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showNativeVideo])

  // 视频 → 图表游标 + 具身隐喻「子弹时间」定格
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const onTimeUpdate = () => {
      if (scrubbingRef.current) return
      const currentTime = video.currentTime

      // —— 波形游标同步 ——
      const idx = nearestDataIndexByTime(seriesRef.current, currentTime)
      if (idx == null) {
        if (playheadVisibleRef.current) {
          playheadVisibleRef.current = false
          setPlayheadVisible(false)
        }
      } else {
        if (!playheadVisibleRef.current) {
          playheadVisibleRef.current = true
          setPlayheadVisible(true)
        }
        if (idx !== playheadFrameRef.current) {
          playheadFrameRef.current = idx
          setPlayheadFrame(idx)
        }
      }

      // —— 子弹时间：触及 error_timestamp_sec 则一次性 pause + 渲染 ——
      if (!showNativeVideoRef.current) return
      const focus = focusHighlightRef.current
      if (!focus) return
      const errorTs = focus.error_timestamp_sec
      if (typeof errorTs !== 'number' || !Number.isFinite(errorTs)) return

      // 循环回到错误点之前：重新武装，允许下一轮定格
      if (
        hasPausedForErrorRef.current &&
        !bulletFreezeActiveRef.current &&
        currentTime < errorTs - BULLET_TIME_REARM_GAP_SEC
      ) {
        hasPausedForErrorRef.current = false
      }

      if (hasPausedForErrorRef.current) return
      if (Math.abs(currentTime - errorTs) >= BULLET_TIME_THRESHOLD_SEC) return

      hasPausedForErrorRef.current = true
      try {
        // 钉死在临床错误绝对秒，避免 timeupdate 抖动偏帧
        if (Math.abs(video.currentTime - errorTs) > 0.02) {
          video.currentTime = errorTs
        }
      } catch {
        /* seek 中途忽略 */
      }
      video.pause()
      setIsPlaying(false)
      setBulletFreezeActive(true)
      setSeekBadge('子弹时间 · 错误定格')

      clearBulletTimeTimer()
      bulletTimeTimerRef.current = setTimeout(() => {
        bulletTimeTimerRef.current = null
        setBulletFreezeActive(false)
        bulletFreezeActiveRef.current = false
        const v = videoRef.current
        if (!v || !showNativeVideoRef.current) return
        forceInterventionPlaybackRate()
        void v.play().catch(() => undefined)
        setIsPlaying(true)
        setSeekBadge(null)
      }, BULLET_TIME_HOLD_MS)
    }

    const onPlay = () => setIsPlaying(true)
    const onPause = () => setIsPlaying(false)
    const onEnded = () => setIsPlaying(false)
    /** loop 回绕时重新武装子弹时间 */
    const onSeeked = () => {
      const focus = focusHighlightRef.current
      const errorTs = focus?.error_timestamp_sec
      if (
        typeof errorTs === 'number' &&
        Number.isFinite(errorTs) &&
        !bulletFreezeActiveRef.current &&
        video.currentTime < errorTs - BULLET_TIME_REARM_GAP_SEC
      ) {
        hasPausedForErrorRef.current = false
      }
    }

    video.addEventListener('timeupdate', onTimeUpdate)
    video.addEventListener('play', onPlay)
    video.addEventListener('pause', onPause)
    video.addEventListener('ended', onEnded)
    video.addEventListener('seeked', onSeeked)
    return () => {
      video.removeEventListener('timeupdate', onTimeUpdate)
      video.removeEventListener('play', onPlay)
      video.removeEventListener('pause', onPause)
      video.removeEventListener('ended', onEnded)
      video.removeEventListener('seeked', onSeeked)
      clearBulletTimeTimer()
    }
  }, [videoSrc, series])

  // 图表容器尺寸变化时 resize
  useEffect(() => {
    const host = chartHostRef.current
    const chart = chartRef.current
    if (!host || !chart || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => chart.resize())
    ro.observe(host)
    return () => ro.disconnect()
  }, [])

  const playheadTimestamp =
    series[playheadFrame]?.absolute_timestamp ?? (safeOffset + playheadFrame) / safeFps
  const absolutePlayhead = safeOffset + (series[playheadFrame]?.frame_index ?? playheadFrame)
  /** 仅在子弹时间定格窗口内渲染 Canvas（3 秒后清空） */
  const jointHighlightActive = showNativeVideo && bulletFreezeActive && Boolean(focusHighlight)

  const togglePlay = () => {
    const video = videoRef.current
    if (!video || !videoSrc) return
    // 手动播控时取消未完成的子弹时间计时，避免与用户操作打架
    if (bulletFreezeActive) {
      clearBulletTimeTimer()
      setBulletFreezeActive(false)
      bulletFreezeActiveRef.current = false
    }
    forceInterventionPlaybackRate()
    if (video.paused) void video.play()
    else video.pause()
  }

  /** 合并当前视频帧（或推理画面）与 Canvas 涂鸦 → JPEG Base64 */
  async function composeAnnotatedFrame(): Promise<string | null> {
    const stage = stageRef.current
    if (!stage) return null
    const width = stage.clientWidth
    const height = stage.clientHeight
    if (width <= 0 || height <= 0) return null

    videoRef.current?.pause()
    setIsPlaying(false)

    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const exportCanvas = document.createElement('canvas')
    exportCanvas.width = Math.floor(width * dpr)
    exportCanvas.height = Math.floor(height * dpr)
    const ctx = exportCanvas.getContext('2d')
    if (!ctx) return null
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.fillStyle = '#000000'
    ctx.fillRect(0, 0, width, height)

    const drawContain = (source: CanvasImageSource, sw: number, sh: number) => {
      if (sw <= 0 || sh <= 0) return
      const scale = Math.min(width / sw, height / sh)
      const dw = sw * scale
      const dh = sh * scale
      const dx = (width - dw) / 2
      const dy = (height - dh) / 2
      ctx.drawImage(source, dx, dy, dw, dh)
    }

    try {
      const video = videoRef.current
      const liveImg = stage.querySelector('img') as HTMLImageElement | null
      if (video && showNativeVideo && video.readyState >= 2 && video.videoWidth > 0) {
        drawContain(video, video.videoWidth, video.videoHeight)
      } else if (liveImg && liveImg.complete && liveImg.naturalWidth > 0) {
        drawContain(liveImg, liveImg.naturalWidth, liveImg.naturalHeight)
      }
    } catch {
      /* 底层帧缺失时仍导出涂鸦层 */
    }

    const layerUrl = telestrationRef.current?.exportLayerDataUrl()
    if (layerUrl) {
      try {
        const layer = await loadImage(layerUrl)
        ctx.drawImage(layer, 0, 0, width, height)
      } catch {
        /* ignore */
      }
    }

    return exportCanvas.toDataURL('image/jpeg', 0.92)
  }

  async function handleSaveAnnotation() {
    setIsSavingAnnotation(true)
    setAnnotationHint(null)
    try {
      const imageBase64 = await composeAnnotatedFrame()
      if (!imageBase64) throw new Error('无法合成批注截图')

      const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
      const filename = `telestration_${attemptId || studentNumber || 'clip'}_${stamp}.jpg`
      triggerJpegDownload(imageBase64, filename)

      let serverMsg = '批注已下载到本地'
      try {
        const response = await fetch(`${API_BASE_URL}/api/save_telestration_image`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            attemptId: attemptId || undefined,
            studentNumber: studentNumber || undefined,
            imageBase64,
          }),
        })
        const data = (await response.json()) as {
          success: boolean
          message?: string
          path?: string
        }
        if (data.success) {
          serverMsg = data.message || `批注已归档：${data.path || ''}`
        } else {
          serverMsg = `已本地下载；云端归档失败：${data.message || '未知错误'}`
        }
      } catch {
        serverMsg = '已本地下载；后端暂不可达，稍后可重试上传'
      }

      setAnnotationHint(serverMsg)
      onTelestrationSaved?.(true, serverMsg)
    } catch (error) {
      const msg = error instanceof Error ? error.message : '保存批注失败'
      setAnnotationHint(msg)
      onTelestrationSaved?.(false, msg)
    } finally {
      setIsSavingAnnotation(false)
    }
  }

  return (
    <section
      className={`workbench-col workbench-card overflow-hidden ${className}`.trim()}
      aria-label="同步视频工作区"
    >
      <div className="flex h-full min-h-0 flex-col">
        <header className="flex flex-shrink-0 items-center justify-between gap-2 border-b border-slate-700/80 px-3 py-2.5">
          <div className="flex min-w-0 items-center gap-2">
            <span className="inline-flex flex-shrink-0 text-[var(--GREEN_OPTIMAL)]">
              <Clapperboard className="h-4 w-4" />
            </span>
            <div className="min-w-0">
              <h2 className="truncate text-sm font-semibold text-slate-100">{title}</h2>
              <p className="truncate text-[10px] text-slate-400">{subtitle}</p>
            </div>
          </div>
          <div className="flex flex-shrink-0 flex-wrap items-center justify-end gap-1.5">
            {seekBadge && (
              <span className="rounded-lg border border-[color-mix(in_srgb,var(--GREEN_OPTIMAL)_35%,transparent)] bg-[color-mix(in_srgb,var(--GREEN_OPTIMAL)_12%,transparent)] px-2 py-1 text-[10px] font-semibold text-[var(--GREEN_OPTIMAL)]">
                {seekBadge}
              </span>
            )}
            {enableTelestration && (
              <>
                <button
                  type="button"
                  onClick={() => setPenActive((v) => !v)}
                  className={`inline-flex items-center gap-1 rounded-lg border px-2 py-1 text-[11px] font-medium transition ${
                    penActive
                      ? 'border-emerald-500/50 bg-emerald-500/20 text-emerald-200'
                      : 'border-slate-600/80 bg-slate-900/60 text-slate-200 hover:border-emerald-500/40 hover:text-emerald-300'
                  }`}
                  title={penActive ? '关闭画笔，恢复视频点击穿透' : '开启教练手绘电烙铁'}
                >
                  <PenLine className="h-3.5 w-3.5" />
                  {penActive ? '关闭画笔' : '✍️开启画笔'}
                </button>
                <button
                  type="button"
                  onClick={() => telestrationRef.current?.clearAll()}
                  className="inline-flex items-center gap-1 rounded-lg border border-slate-600/80 bg-slate-900/60 px-2 py-1 text-[11px] text-slate-200 transition hover:border-rose-500/40 hover:text-rose-300"
                  title="清空当前涂鸦"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  🗑️清除
                </button>
                <button
                  type="button"
                  onClick={() => void handleSaveAnnotation()}
                  disabled={isSavingAnnotation}
                  className="inline-flex items-center gap-1 rounded-lg border border-rose-500/40 bg-rose-500/15 px-2 py-1 text-[11px] font-medium text-rose-100 transition hover:border-rose-400/60 hover:bg-rose-500/25 disabled:cursor-not-allowed disabled:opacity-55"
                  title="合并视频帧与涂鸦并下载 / 上传"
                >
                  {isSavingAnnotation ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Camera className="h-3.5 w-3.5" />
                  )}
                  📸保存批注
                </button>
              </>
            )}
            {videoSrc && (
              <button
                type="button"
                onClick={togglePlay}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-600/80 bg-slate-900/60 px-2.5 py-1 text-[11px] text-slate-200 transition hover:border-emerald-500/40 hover:text-emerald-300"
              >
                {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                {isPlaying ? '暂停' : '播放'}
              </button>
            )}
          </div>
        </header>

        <div ref={stageRef} className="relative min-h-0 flex-[1.35] overflow-hidden bg-black/40">
          {/* 视频 + 具身隐喻 Canvas 叠层：相对定位父容器，Canvas 与视频同尺寸 */}
          <div className="absolute inset-0">
            {videoSrc && (
              <video
                ref={videoRef}
                src={videoSrc}
                className={`absolute inset-0 h-full w-full bg-black object-contain ${
                  showNativeVideo ? 'opacity-100' : 'pointer-events-none opacity-0'
                }`}
                playsInline
                preload="auto"
                autoPlay
                muted
                loop
                onLoadedMetadata={forceInterventionPlaybackRate}
                onCanPlay={forceInterventionPlaybackRate}
                onPlay={forceInterventionPlaybackRate}
                onRateChange={() => {
                  const video = videoRef.current
                  if (video && video.playbackRate !== INTERVENTION_PLAYBACK_RATE) {
                    video.playbackRate = INTERVENTION_PLAYBACK_RATE
                  }
                }}
              />
            )}
            <JointHighlightOverlay
              containerRef={stageRef}
              videoRef={videoRef}
              highlights={focusHighlight ? [focusHighlight] : null}
              active={jointHighlightActive}
            />
          </div>
          {/* 实时推理画面：分析中优先；无本地视频源时作为主视口 */}
          {(preferLiveOverlay || !videoSrc) && (
            <div className="absolute inset-0 z-[1]">{children}</div>
          )}
          {!preferLiveOverlay && !videoSrc && !children && (
            <div className="absolute inset-0 flex items-center justify-center text-xs text-slate-500">
              请选择本地视频或启动实时分析
            </div>
          )}
          {/* HUD / 角标始终可叠在视频之上；画笔激活时让路给 Canvas */}
          {overlay && <div className="pointer-events-none absolute inset-0 z-10">{overlay}</div>}
          {enableTelestration && (
            <TelestrationCanvas
              ref={telestrationRef}
              drawingEnabled={penActive}
              onDrawingEnabledChange={setPenActive}
              showToolbar={penActive}
            />
          )}
          {annotationHint && (
            <div className="pointer-events-none absolute bottom-2 left-1/2 z-30 max-w-[90%] -translate-x-1/2 rounded-lg border border-white/10 bg-black/75 px-3 py-1.5 text-center text-[10px] text-slate-200 backdrop-blur">
              {annotationHint}
            </div>
          )}
        </div>

        <div className="flex min-h-0 flex-shrink-0 flex-col border-t border-slate-700/80 bg-slate-950/40">
          <div className="flex items-center justify-between gap-2 px-3 pt-2">
            <p className="text-[10px] uppercase tracking-wide text-slate-500">
              鞭打发力角速度 (deg/s)
            </p>
            <p className="text-[10px] tabular-nums text-slate-500">
              t={playheadTimestamp.toFixed(3)}s
              {` · idx #${playheadFrame}`}
              {safeOffset > 0 ? ` · abs F#${absolutePlayhead}` : ''}
              {typeof impactIndexInWindow === 'number' || typeof tImpact === 'number' || series.length > 0
                ? ` · 触球 @${impactTimestamp.toFixed(3)}s`
                : ''}
              {` · ${safeFps} fps`}
            </p>
          </div>
          <div ref={chartHostRef} className="h-[168px] w-full px-1 pb-1" />
          {resolvedPhases.length > 0 && (
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 pb-2 text-[10px] text-slate-500">
              {resolvedPhases.map((phase) => (
                <span key={phase.key} className="inline-flex items-center gap-1.5">
                  <span
                    className="inline-block h-2 w-2 rounded-sm"
                    style={{ backgroundColor: PHASE_SWATCH[phase.key] }}
                  />
                  [{phase.label}]
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
