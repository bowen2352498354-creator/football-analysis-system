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
import TelestrationCanvas, { type TelestrationCanvasHandle } from './TelestrationCanvas'

const API_BASE_URL = 'http://localhost:8000'

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

/** 单帧角速度采样：frame_index 与视频 30fps 时间轴一一对应 */
export interface SyncVelocityPoint {
  frame_index: number
  omega: number
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
  /** 触球锁帧索引（绝对帧）；缺省时取 |ω| 峰值帧 */
  tImpact?: number | null
  /**
   * 触球点在当前波形窗口内的索引（优先用于 markLine）。
   * 对应后端 `impact_index_in_window`，正常为 30。
   */
  impactIndexInWindow?: number | null
  /**
   * velocitySeries[0] 对应的绝对视频帧号。
   * 窗口模式（0..60）下用于 scrub：videoFrame = offset + localIndex。
   */
  seriesFrameOffset?: number
  /** 自定义阶段切片；缺省时按 t_impact 自动切分五段 */
  phases?: ActionPhaseSlice[] | null
  /** 视频帧率，默认 30（currentTime = absoluteFrame / fps） */
  fps?: number
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

function sanitizeSeries(raw: SyncVelocityPoint[] | null | undefined): SyncVelocityPoint[] {
  if (!Array.isArray(raw) || raw.length === 0) return []
  const out: SyncVelocityPoint[] = []
  for (let i = 0; i < raw.length; i += 1) {
    const row = raw[i]
    if (!row || typeof row !== 'object') continue
    const frame =
      typeof row.frame_index === 'number' && Number.isFinite(row.frame_index)
        ? Math.max(0, Math.round(row.frame_index))
        : i
    const omega = Number(row.omega)
    out.push({ frame_index: frame, omega: Number.isFinite(omega) ? omega : 0 })
  }
  out.sort((a, b) => a.frame_index - b.frame_index)
  return out
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
 * - 波形图下方五段动作色块 + t_impact 红色锚线
 * - 图表 highlight/click → video.currentTime = frame_index / fps
 * - video timeupdate → 游标随播放平滑右移
 */
export default function SynchronizedVideoWorkspace({
  videoSrc = null,
  velocitySeries = null,
  tImpact = null,
  impactIndexInWindow = null,
  seriesFrameOffset = 0,
  phases = null,
  fps = DEFAULT_FPS,
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

  const [isPlaying, setIsPlaying] = useState(false)
  const [playheadFrame, setPlayheadFrame] = useState(0)
  const [seekBadge, setSeekBadge] = useState<string | null>(null)
  const [penActive, setPenActive] = useState(false)
  const [isSavingAnnotation, setIsSavingAnnotation] = useState(false)
  const [annotationHint, setAnnotationHint] = useState<string | null>(null)

  const series = useMemo(() => sanitizeSeries(velocitySeries), [velocitySeries])
  const safeOffset = Number.isFinite(seriesFrameOffset) ? Math.max(0, Math.round(seriesFrameOffset)) : 0
  const impactFrame = useMemo(() => {
    if (typeof impactIndexInWindow === 'number' && Number.isFinite(impactIndexInWindow)) {
      const maxIdx = series.length > 0 ? series[series.length - 1].frame_index : 0
      return Math.max(0, Math.min(maxIdx, Math.round(impactIndexInWindow)))
    }
    return resolveTImpact(series, tImpact)
  }, [series, tImpact, impactIndexInWindow])
  const frameCount = series.length > 0 ? series[series.length - 1].frame_index + 1 : 0
  const safeFps = fps > 0 ? fps : DEFAULT_FPS

  seriesRef.current = series
  fpsRef.current = safeFps
  offsetRef.current = safeOffset

  const resolvedPhases = useMemo(() => {
    if (Array.isArray(phases) && phases.length > 0) return phases
    if (frameCount <= 0) return []
    return buildDefaultPhases(frameCount, impactFrame)
  }, [phases, frameCount, impactFrame])

  /** localIndex：波形 X 轴帧；映射到绝对视频帧后再 seek */
  const seekToLocalFrame = (localIndex: number) => {
    const video = videoRef.current
    const rate = fpsRef.current
    if (!Number.isFinite(localIndex)) return
    const clampedLocal = Math.max(0, Math.round(localIndex))
    const absoluteFrame = clampedLocal + offsetRef.current
    if (video) {
      video.pause()
      const nextTime = absoluteFrame / rate
      if (Math.abs(video.currentTime - nextTime) > 1 / (rate * 2)) {
        video.currentTime = nextTime
      }
    }
    playheadFrameRef.current = clampedLocal
    setPlayheadFrame(clampedLocal)
    setIsPlaying(false)
  }

  /** 外部传入绝对帧 → 转成窗口内 localIndex 再定格 */
  const seekToAbsoluteFrame = (absoluteFrame: number) => {
    seekToLocalFrame(Math.round(absoluteFrame) - offsetRef.current)
  }

  // MetricCardList → 物理极值帧 Seek
  useEffect(() => {
    if (!externalSeek) return
    seekToAbsoluteFrame(externalSeek.frameIndex)
    if (externalSeek.label) {
      setSeekBadge(`${externalSeek.label} · F#${externalSeek.frameIndex}`)
    }
  }, [externalSeek])

  const frameFromChartEvent = (params: unknown): number | null => {
    const current = seriesRef.current
    const p = params as {
      dataIndex?: number
      data?: unknown
      value?: unknown
      batch?: Array<{ dataIndex?: number }>
    }
    if (Array.isArray(p?.batch) && p.batch.length > 0) {
      const idx = p.batch[0]?.dataIndex
      if (typeof idx === 'number' && current[idx]) return current[idx].frame_index
    }
    if (typeof p?.dataIndex === 'number' && current[p.dataIndex]) {
      return current[p.dataIndex].frame_index
    }
    const data = p?.data
    if (Array.isArray(data) && typeof data[0] === 'number') return Math.round(data[0])
    if (data && typeof data === 'object' && 'frame_index' in data) {
      const fi = Number((data as { frame_index: number }).frame_index)
      if (Number.isFinite(fi)) return Math.round(fi)
    }
    if (typeof p?.value === 'number' && Number.isFinite(p.value)) {
      return Math.round(p.value)
    }
    return null
  }

  // 初始化 / 销毁 ECharts 实例（事件经 ref 读取最新 series / fps）
  useEffect(() => {
    const host = chartHostRef.current
    if (!host) return
    const chart = echarts.init(host, undefined, { renderer: 'canvas' })
    chartRef.current = chart

    const onHighlight = (params: unknown) => {
      const frame = frameFromChartEvent(params)
      if (frame === null) return
      scrubbingRef.current = true
      seekToLocalFrame(frame)
    }
    const onClick = (params: unknown) => {
      const frame = frameFromChartEvent(params)
      if (frame === null) return
      scrubbingRef.current = true
      seekToLocalFrame(frame)
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
      seekToLocalFrame(pointInGrid[0])
    })

    const onResize = () => chart.resize()
    window.addEventListener('resize', onResize)

    return () => {
      window.removeEventListener('resize', onResize)
      chart.off('highlight', onHighlight)
      chart.off('click', onClick)
      chart.dispose()
      chartRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const impactMarkLine = {
    xAxis: impactFrame,
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

    const markAreaData = resolvedPhases.map((phase) => [
      {
        name: phase.label,
        xAxis: phase.startFrame,
        itemStyle: { color: phase.color },
        label: {
          show: true,
          position: 'insideTop',
          color: 'rgba(226,232,240,0.55)',
          fontSize: 10,
          formatter: phase.label,
        },
      },
      { xAxis: phase.endFrame },
    ])

    const xMin = series[0].frame_index
    const xMax = series[series.length - 1].frame_index

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
            const first = arr[0] as { data?: [number, number]; axisValue?: number | string }
            const frame =
              Array.isArray(first?.data) && typeof first.data[0] === 'number'
                ? first.data[0]
                : Number(first?.axisValue)
            const omega =
              Array.isArray(first?.data) && typeof first.data[1] === 'number' ? first.data[1] : NaN
            const absFrame = Number.isFinite(frame) ? Math.round(frame) + safeOffset : NaN
            const tMs = Number.isFinite(absFrame) ? Math.round((absFrame / safeFps) * 1000) : '—'
            const omegaText = Number.isFinite(omega) ? `${omega.toFixed(1)} deg/s` : '—'
            return `Frame ${Number.isFinite(frame) ? frame : '—'} · abs #${
              Number.isFinite(absFrame) ? absFrame : '—'
            } · ${tMs} ms<br/>ω = ${omegaText}`
          },
        },
        axisPointer: {
          link: [{ xAxisIndex: 'all' }],
          label: { backgroundColor: '#1e293b' },
        },
        xAxis: {
          type: 'value',
          name: 'frame',
          nameTextStyle: { color: 'rgba(148,163,184,0.45)', fontSize: 10 },
          min: xMin,
          max: xMax,
          axisLabel: { color: 'rgba(148,163,184,0.45)', fontSize: 10 },
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
            data: series.map((p) => [p.frame_index, p.omega]),
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
                  xAxis: playheadFrameRef.current,
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
  }, [series, resolvedPhases, impactFrame, safeFps, safeOffset])

  // 播放游标：仅更新 markLine，避免整表重绘
  useEffect(() => {
    const chart = chartRef.current
    if (!chart || series.length < 2) return
    chart.setOption({
      series: [
        {
          id: 'shank-omega',
          markLine: {
            data: [
              impactMarkLine,
              {
                xAxis: playheadFrame,
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
    })
  }, [playheadFrame, impactFrame, series.length])

  // 视频 → 图表：timeupdate 驱动游标（绝对帧 → 窗口 localIndex）
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const onTimeUpdate = () => {
      if (scrubbingRef.current) return
      const absoluteFrame = Math.max(0, Math.round(video.currentTime * safeFps))
      const localFrame = absoluteFrame - safeOffset
      if (localFrame !== playheadFrameRef.current) {
        playheadFrameRef.current = localFrame
        setPlayheadFrame(localFrame)
      }
    }
    const onPlay = () => setIsPlaying(true)
    const onPause = () => setIsPlaying(false)
    const onEnded = () => setIsPlaying(false)

    video.addEventListener('timeupdate', onTimeUpdate)
    video.addEventListener('play', onPlay)
    video.addEventListener('pause', onPause)
    video.addEventListener('ended', onEnded)
    return () => {
      video.removeEventListener('timeupdate', onTimeUpdate)
      video.removeEventListener('play', onPlay)
      video.removeEventListener('pause', onPause)
      video.removeEventListener('ended', onEnded)
    }
  }, [videoSrc, safeFps, safeOffset])

  // 图表容器尺寸变化时 resize
  useEffect(() => {
    const host = chartHostRef.current
    const chart = chartRef.current
    if (!host || !chart || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => chart.resize())
    ro.observe(host)
    return () => ro.disconnect()
  }, [])

  const showNativeVideo = Boolean(videoSrc) && !preferLiveOverlay
  const absolutePlayhead = playheadFrame + safeOffset

  const togglePlay = () => {
    const video = videoRef.current
    if (!video || !videoSrc) return
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
          {videoSrc && (
            <video
              ref={videoRef}
              src={videoSrc}
              className={`absolute inset-0 h-full w-full bg-black object-contain ${
                showNativeVideo ? 'opacity-100' : 'pointer-events-none opacity-0'
              }`}
              playsInline
              preload="auto"
            />
          )}
          {/* 实时推理画面：分析中优先；无本地视频源时作为主视口 */}
          {(preferLiveOverlay || !videoSrc) && (
            <div className="absolute inset-0">{children}</div>
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
              Frame {playheadFrame}
              {safeOffset > 0 ? ` · abs #${absolutePlayhead}` : ''}
              {typeof impactIndexInWindow === 'number' || typeof tImpact === 'number' || series.length > 0
                ? ` · 触球 @${impactFrame}`
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
