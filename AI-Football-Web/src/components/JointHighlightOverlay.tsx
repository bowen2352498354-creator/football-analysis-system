import { useEffect, useRef, type RefObject } from 'react'
import type { JointHighlight } from '../types'

/** 具身语义：问题部位 → 一眼能懂的 Emoji 隐喻 */
export type EmbodiedJointKind = 'knee' | 'ankle' | 'support_foot' | 'hip' | 'other'

const EMBODIED_EMOJI: Record<EmbodiedJointKind, string> = {
  knee: '🧊', // 僵硬 / 直腿
  ankle: '🔓', // 未锁死 / 松弛
  support_foot: '📏', // 支撑距过远 / 站位偏差
  hip: '🔄', // 髋扭转 / 躯干拧转
  other: '⚠️',
}

const CORE_RADIUS_PX = 17
const PULSE_MIN_RADIUS_PX = 18
const PULSE_MAX_RADIUS_PX = 36
const PULSE_PERIOD_MS = 1400
const EMOJI_FONT = '28px "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif'

export interface JointHighlightOverlayProps {
  containerRef: RefObject<HTMLElement | null>
  videoRef: RefObject<HTMLVideoElement | null>
  highlights?: JointHighlight[] | null
  /** false 时清空画布（录制模式 / 无诊断数据） */
  active?: boolean
  className?: string
}

function trafficRank(colorCode: string | undefined): number {
  const key = String(colorCode || '')
    .trim()
    .toUpperCase()
  if (key.includes('RED')) return 2
  if (key.includes('YELLOW')) return 1
  return 0 // GREEN / unknown
}

/**
 * 单点聚焦：丢掉 GREEN，全屏最多保留 1 个最严重警告（优先首个 RED）。
 */
export function pickFocusHighlight(
  highlights: JointHighlight[] | null | undefined,
): JointHighlight | null {
  if (!Array.isArray(highlights) || highlights.length === 0) return null
  const warnings = highlights.filter((h) => trafficRank(h.color_code) > 0)
  if (warnings.length === 0) return null
  const firstRed = warnings.find((h) => trafficRank(h.color_code) === 2)
  return firstRed ?? warnings[0]
}

/** 根据 joint_name / metric_key 判定具身部位种类 */
export function resolveEmbodiedKind(highlight: JointHighlight): EmbodiedJointKind {
  const joint = String(highlight.joint_name || '').toLowerCase()
  const metric = String(highlight.metric_key || '').toLowerCase()

  if (
    metric.includes('distance') ||
    metric.includes('toe_angle') ||
    joint.includes('foot_index') ||
    (joint.includes('ankle') && metric.includes('distance'))
  ) {
    return 'support_foot'
  }
  if (metric.includes('ankle') || joint.includes('ankle')) return 'ankle'
  if (metric.includes('knee') || metric.includes('folding') || joint.includes('knee')) {
    return 'knee'
  }
  if (metric.includes('hip') || joint.includes('hip')) return 'hip'
  if (joint.includes('foot')) return 'support_foot'
  return 'other'
}

function looksNormalized(x: number, y: number, space?: string | null): boolean {
  const explicit = String(space || '').toLowerCase()
  if (explicit === 'normalized') return true
  if (explicit === 'pixel') return false
  return x >= 0 && x <= 1.0001 && y >= 0 && y <= 1.0001
}

function mapJointToCanvas(
  highlight: JointHighlight,
  video: HTMLVideoElement,
  canvasCssW: number,
  canvasCssH: number,
): { x: number; y: number } | null {
  const vw = video.videoWidth || 0
  const vh = video.videoHeight || 0
  if (vw <= 0 || vh <= 0 || canvasCssW <= 0 || canvasCssH <= 0) return null
  if (!Number.isFinite(highlight.x) || !Number.isFinite(highlight.y)) return null

  let srcX = highlight.x
  let srcY = highlight.y
  if (looksNormalized(srcX, srcY, highlight.coordinate_space)) {
    srcX *= vw
    srcY *= vh
  }

  const scale = Math.min(canvasCssW / vw, canvasCssH / vh)
  const offsetX = (canvasCssW - vw * scale) / 2
  const offsetY = (canvasCssH - vh * scale) / 2
  return { x: offsetX + srcX * scale, y: offsetY + srcY * scale }
}

function syncCanvasSize(canvas: HTMLCanvasElement, cssW: number, cssH: number): number {
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  const tw = Math.max(1, Math.floor(cssW * dpr))
  const th = Math.max(1, Math.floor(cssH * dpr))
  if (canvas.width !== tw || canvas.height !== th) {
    canvas.width = tw
    canvas.height = th
  }
  canvas.style.width = `${cssW}px`
  canvas.style.height = `${cssH}px`
  return dpr
}

function clearCanvas(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.clearRect(0, 0, canvas.width, canvas.height)
}

/**
 * 绘制单一焦点：雷达波呼吸外圈 + 中心具身 Emoji。
 * phase ∈ [0, 1) 由 sin 驱动半径与透明度消散。
 */
function paintFocus(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  emoji: string,
  phase: number,
  isRed: boolean,
) {
  const pulse = 0.5 + 0.5 * Math.sin(phase * Math.PI * 2)
  const ringR = PULSE_MIN_RADIUS_PX + (PULSE_MAX_RADIUS_PX - PULSE_MIN_RADIUS_PX) * pulse
  const ringAlpha = 0.75 * (1 - pulse * 0.85)
  const tone = isRed ? '255, 60, 60' : '255, 190, 40'

  // 第二道滞后波纹，增强「雷达报警」感
  const pulse2 = 0.5 + 0.5 * Math.sin((phase + 0.35) * Math.PI * 2)
  const ringR2 = PULSE_MIN_RADIUS_PX + (PULSE_MAX_RADIUS_PX - PULSE_MIN_RADIUS_PX) * pulse2
  const ringAlpha2 = 0.45 * (1 - pulse2 * 0.9)

  ctx.beginPath()
  ctx.arc(x, y, ringR2, 0, Math.PI * 2)
  ctx.strokeStyle = `rgba(${tone}, ${ringAlpha2.toFixed(3)})`
  ctx.lineWidth = 2
  ctx.stroke()

  ctx.beginPath()
  ctx.arc(x, y, ringR, 0, Math.PI * 2)
  ctx.strokeStyle = `rgba(${tone}, ${ringAlpha.toFixed(3)})`
  ctx.lineWidth = 2.5
  ctx.stroke()

  // 小实心核：精准钉在关节上（15–20px）
  ctx.beginPath()
  ctx.arc(x, y, CORE_RADIUS_PX, 0, Math.PI * 2)
  ctx.fillStyle = `rgba(${tone}, 0.22)`
  ctx.fill()
  ctx.lineWidth = 2
  ctx.strokeStyle = `rgba(${tone}, 0.9)`
  ctx.stroke()

  ctx.font = EMOJI_FONT
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(emoji, x, y + 1)
}

/**
 * 具身隐喻极简叠层：全屏最多 1 个焦点（呼吸波纹 + Emoji）。
 * ``active`` 由父级「子弹时间」定格窗口驱动：触及 error_timestamp_sec 时开启，
 * 3 秒后关闭并 clearRect；非定格时段不绘制。
 */
export default function JointHighlightOverlay({
  containerRef,
  videoRef,
  highlights = null,
  active = false,
  className = '',
}: JointHighlightOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const focusRef = useRef<JointHighlight | null>(null)
  const rafRef = useRef<number>(0)
  const startRef = useRef<number>(0)

  useEffect(() => {
    focusRef.current = pickFocusHighlight(highlights)
  }, [highlights])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const stop = () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = 0
      }
      clearCanvas(canvas)
    }

    if (!active) {
      stop()
      return stop
    }

    startRef.current = performance.now()

    const frame = (now: number) => {
      const stage = containerRef.current
      const video = videoRef.current
      const layer = canvasRef.current
      const focus = focusRef.current
      if (!layer || !stage) {
        rafRef.current = requestAnimationFrame(frame)
        return
      }

      const cssW = stage.clientWidth
      const cssH = stage.clientHeight
      if (cssW <= 0 || cssH <= 0 || !video || !focus) {
        clearCanvas(layer)
        layer.style.width = cssW > 0 ? `${cssW}px` : layer.style.width
        layer.style.height = cssH > 0 ? `${cssH}px` : layer.style.height
        rafRef.current = requestAnimationFrame(frame)
        return
      }

      const mapped = mapJointToCanvas(focus, video, cssW, cssH)
      if (!mapped) {
        clearCanvas(layer)
        rafRef.current = requestAnimationFrame(frame)
        return
      }

      const dpr = syncCanvasSize(layer, cssW, cssH)
      const ctx = layer.getContext('2d')
      if (!ctx) {
        rafRef.current = requestAnimationFrame(frame)
        return
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, cssW, cssH)

      const elapsed = now - startRef.current
      const phase = (elapsed % PULSE_PERIOD_MS) / PULSE_PERIOD_MS
      const kind = resolveEmbodiedKind(focus)
      const emoji = EMBODIED_EMOJI[kind]
      const isRed = trafficRank(focus.color_code) === 2
      paintFocus(ctx, mapped.x, mapped.y, emoji, phase, isRed)

      rafRef.current = requestAnimationFrame(frame)
    }

    rafRef.current = requestAnimationFrame(frame)

    let ro: ResizeObserver | null = null
    const container = containerRef.current
    if (container && typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(() => {
        /* 下一帧 paint 会读取新尺寸 */
      })
      ro.observe(container)
    }

    return () => {
      ro?.disconnect()
      stop()
    }
  }, [containerRef, videoRef, active, highlights])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      className={`pointer-events-none absolute top-0 left-0 z-[5] h-full w-full ${className}`.trim()}
    />
  )
}
