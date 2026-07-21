import { useEffect, useMemo, useRef } from 'react'
import { motion } from 'framer-motion'
import { Crosshair, MapPinned } from 'lucide-react'
import type { TrajectoryPoint2D } from '../types'

/* ============================================================================
 * 【V3·模块一 / Sprint 1】2D 虚拟球场时空热力图 SpatialHeatmap
 *
 * 两种渲染路径：
 *   1) 优先：后端 OpenCV 生成的 `heatmapBase64`（800×800，原点=球心，1 px=0.5 cm）
 *      → `<img src="data:image/png;base64,..." />` 高科技光斑云图
 *   2) 回退：前端 Canvas 落点热力斑（深绿草坪 + 学术合格站位带）
 * ========================================================================== */

/** 学术合格站位带：横向 20~25 cm、前后 0~-10 cm（相对球心） */
const IDEAL_LATERAL_MIN = 20
const IDEAL_LATERAL_MAX = 25
const IDEAL_AP_MIN = -10
const IDEAL_AP_MAX = 0

/** 离散度阈值（cm²）：低于此值判定步点定型稳定 */
const DISPERSION_STABLE_THRESHOLD = 28

/** 画布逻辑坐标系覆盖范围（厘米） */
const VIEW_HALF_X = 45
const VIEW_Z_MIN = -30
const VIEW_Z_MAX = 20

export interface SpatialHeatmapProps {
  /** 支撑脚落点数组，每项为 [x_cm, z_cm]；空数组时展示优雅空草坪占位 */
  landingPositions?: TrajectoryPoint2D[]
  /**
   * 【Sprint 1】后端 OpenCV 生成的时空热力云图（纯 base64 或完整 data URI）。
   * 若提供，优先以 `<img>` 展示高科技光斑图，而非前端 Canvas 落点估算。
   */
  heatmapBase64?: string | null
  /** 标题文案 */
  title?: string
  /** 副标题 / 范围说明 */
  subtitle?: string
  /** 画布高度（px） */
  height?: number
  /** 紧凑模式（左栏窄容器） */
  compact?: boolean
}

/** 规范化热力图 src：兼容纯 base64 与已带 data URI 前缀两种后端写法 */
export function resolveHeatmapDataUri(heatmapBase64: string | null | undefined): string | null {
  if (!heatmapBase64 || typeof heatmapBase64 !== 'string') return null
  const trimmed = heatmapBase64.trim()
  if (!trimmed) return null
  if (trimmed.startsWith('data:image/')) return trimmed
  return `data:image/png;base64,${trimmed}`
}

function isValidPoint(point: TrajectoryPoint2D | null | undefined): point is TrajectoryPoint2D {
  return (
    Array.isArray(point) &&
    point.length >= 2 &&
    typeof point[0] === 'number' &&
    typeof point[1] === 'number' &&
    Number.isFinite(point[0]) &&
    Number.isFinite(point[1])
  )
}

/** 空间离散度指数 = 落点相对质心的均方距离（cm²） */
export function computeDispersionIndex(points: TrajectoryPoint2D[]): number {
  const valid = points.filter(isValidPoint)
  if (valid.length < 2) return 0
  const meanX = valid.reduce((sum, p) => sum + p[0], 0) / valid.length
  const meanZ = valid.reduce((sum, p) => sum + p[1], 0) / valid.length
  const variance =
    valid.reduce((sum, p) => sum + (p[0] - meanX) ** 2 + (p[1] - meanZ) ** 2, 0) / valid.length
  return Math.round(variance * 10) / 10
}

function worldToCanvas(
  xCm: number,
  zCm: number,
  width: number,
  height: number,
): { x: number; y: number } {
  const x = ((xCm + VIEW_HALF_X) / (VIEW_HALF_X * 2)) * width
  const y = ((VIEW_Z_MAX - zCm) / (VIEW_Z_MAX - VIEW_Z_MIN)) * height
  return { x, y }
}

function heatColor(intensity: number): string {
  // 低频冷青 → 中频黄 → 高频暖红
  const t = Math.max(0, Math.min(1, intensity))
  if (t < 0.35) {
    const k = t / 0.35
    return `rgba(${Math.round(34 + 40 * k)}, ${Math.round(197 + 30 * k)}, ${Math.round(180 - 40 * k)}, ${0.35 + 0.35 * k})`
  }
  if (t < 0.7) {
    const k = (t - 0.35) / 0.35
    return `rgba(${Math.round(74 + 180 * k)}, ${Math.round(222 - 60 * k)}, ${Math.round(140 - 100 * k)}, ${0.55 + 0.25 * k})`
  }
  const k = (t - 0.7) / 0.3
  return `rgba(${Math.round(251 - 20 * k)}, ${Math.round(146 - 80 * k)}, ${Math.round(40 - 20 * k)}, ${0.75 + 0.2 * k})`
}

function drawPitch(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  dpr: number,
) {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, width, height)

  // 深绿草坪底 + 细腻光晕
  const grass = ctx.createLinearGradient(0, 0, width, height)
  grass.addColorStop(0, '#0a1f14')
  grass.addColorStop(0.45, '#0d2b1a')
  grass.addColorStop(1, '#071510')
  ctx.fillStyle = grass
  ctx.fillRect(0, 0, width, height)

  // 微妙条纹
  ctx.save()
  ctx.globalAlpha = 0.06
  for (let i = 0; i < width; i += 14) {
    ctx.fillStyle = i % 28 === 0 ? '#ffffff' : '#86efac'
    ctx.fillRect(i, 0, 7, height)
  }
  ctx.restore()

  // 发光白色标线（外框 + 中线）
  ctx.strokeStyle = 'rgba(255,255,255,0.55)'
  ctx.lineWidth = 1.5
  ctx.shadowColor = 'rgba(255,255,255,0.45)'
  ctx.shadowBlur = 8
  ctx.strokeRect(10, 10, width - 20, height - 20)
  ctx.beginPath()
  ctx.moveTo(width / 2, 10)
  ctx.lineTo(width / 2, height - 10)
  ctx.stroke()
  ctx.shadowBlur = 0

  // 学术合格站位阴影带：左右两侧 |x|∈[20,25]、z∈[-10,0]
  for (const side of [-1, 1] as const) {
    const a = worldToCanvas(side * IDEAL_LATERAL_MIN, IDEAL_AP_MAX, width, height)
    const b = worldToCanvas(side * IDEAL_LATERAL_MAX, IDEAL_AP_MIN, width, height)
    const x0 = Math.min(a.x, b.x)
    const x1 = Math.max(a.x, b.x)
    const y0 = Math.min(a.y, b.y)
    const y1 = Math.max(a.y, b.y)
    const band = ctx.createLinearGradient(x0, y0, x1, y1)
    band.addColorStop(0, 'rgba(52, 211, 153, 0.08)')
    band.addColorStop(0.5, 'rgba(52, 211, 153, 0.28)')
    band.addColorStop(1, 'rgba(52, 211, 153, 0.08)')
    ctx.fillStyle = band
    ctx.fillRect(x0, y0, Math.max(2, x1 - x0), Math.max(2, y1 - y0))
    ctx.strokeStyle = 'rgba(110, 231, 183, 0.45)'
    ctx.lineWidth = 1
    ctx.setLineDash([4, 3])
    ctx.strokeRect(x0, y0, Math.max(2, x1 - x0), Math.max(2, y1 - y0))
    ctx.setLineDash([])
  }

  // 刻度标注
  ctx.fillStyle = 'rgba(167, 243, 208, 0.55)'
  ctx.font = '10px "SF Pro Display", "PingFang SC", sans-serif'
  ctx.fillText('20–25 cm', worldToCanvas(22.5, 2, width, height).x - 18, worldToCanvas(22.5, 2, width, height).y)
  ctx.fillText('0 ~ −10 cm', 14, worldToCanvas(-VIEW_HALF_X, -5, width, height).y)

  // 球心 (0,0) 发光足球
  const origin = worldToCanvas(0, 0, width, height)
  const glow = ctx.createRadialGradient(origin.x, origin.y, 2, origin.x, origin.y, 28)
  glow.addColorStop(0, 'rgba(255,255,255,0.95)')
  glow.addColorStop(0.35, 'rgba(251, 191, 36, 0.55)')
  glow.addColorStop(1, 'rgba(251, 191, 36, 0)')
  ctx.fillStyle = glow
  ctx.beginPath()
  ctx.arc(origin.x, origin.y, 28, 0, Math.PI * 2)
  ctx.fill()

  ctx.beginPath()
  ctx.arc(origin.x, origin.y, 7, 0, Math.PI * 2)
  ctx.fillStyle = '#fafafa'
  ctx.shadowColor = 'rgba(255,255,255,0.9)'
  ctx.shadowBlur = 12
  ctx.fill()
  ctx.shadowBlur = 0
  ctx.strokeStyle = 'rgba(0,0,0,0.35)'
  ctx.lineWidth = 1
  ctx.stroke()

  ctx.fillStyle = 'rgba(255,255,255,0.7)'
  ctx.font = '600 11px "SF Pro Display", "PingFang SC", sans-serif'
  ctx.fillText('(0, 0)', origin.x + 12, origin.y - 10)
}

function drawHeatBlobs(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  points: TrajectoryPoint2D[],
) {
  if (points.length === 0) return

  // 简易密度估计：每个点贡献径向核，再按局部强度上色
  const density = new Float32Array(points.length)
  const sigma = 8 // cm
  for (let i = 0; i < points.length; i += 1) {
    let d = 0
    for (let j = 0; j < points.length; j += 1) {
      const dx = points[i][0] - points[j][0]
      const dz = points[i][1] - points[j][1]
      d += Math.exp(-(dx * dx + dz * dz) / (2 * sigma * sigma))
    }
    density[i] = d
  }
  const maxD = Math.max(...density, 1)

  points.forEach((point, index) => {
    const { x, y } = worldToCanvas(point[0], point[1], width, height)
    const intensity = density[index] / maxD
    const radius = 10 + 16 * intensity
    const grad = ctx.createRadialGradient(x, y, 0, x, y, radius)
    const color = heatColor(intensity)
    grad.addColorStop(0, color)
    grad.addColorStop(1, 'rgba(0,0,0,0)')
    ctx.fillStyle = grad
    ctx.beginPath()
    ctx.arc(x, y, radius, 0, Math.PI * 2)
    ctx.fill()

    // 核心高亮点
    ctx.beginPath()
    ctx.arc(x, y, 2.2, 0, Math.PI * 2)
    ctx.fillStyle = intensity > 0.55 ? 'rgba(254, 240, 138, 0.95)' : 'rgba(165, 243, 252, 0.9)'
    ctx.fill()
  })
}

export default function SpatialHeatmap({
  landingPositions = [],
  heatmapBase64 = null,
  title = '2D 虚拟球场时空热力图',
  subtitle,
  height = 280,
  compact = false,
}: SpatialHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const heatmapSrc = useMemo(() => resolveHeatmapDataUri(heatmapBase64), [heatmapBase64])
  const validPoints = useMemo(
    () => landingPositions.filter(isValidPoint),
    [landingPositions],
  )
  const dispersion = useMemo(() => computeDispersionIndex(validPoints), [validPoints])
  const isStable = validPoints.length >= 2 ? dispersion <= DISPERSION_STABLE_THRESHOLD : null
  const showBackendHeatmap = Boolean(heatmapSrc)

  useEffect(() => {
    if (showBackendHeatmap) return
    const canvas = canvasRef.current
    if (!canvas) return
    const parent = canvas.parentElement
    const cssWidth = parent?.clientWidth || 360
    const cssHeight = height
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = Math.floor(cssWidth * dpr)
    canvas.height = Math.floor(cssHeight * dpr)
    canvas.style.width = `${cssWidth}px`
    canvas.style.height = `${cssHeight}px`

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    drawPitch(ctx, cssWidth, cssHeight, dpr)
    drawHeatBlobs(ctx, cssWidth, cssHeight, validPoints)
  }, [validPoints, height, showBackendHeatmap])

  return (
    <section
      className={`spatial-heatmap flex flex-col gap-3 rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-xl ${
        compact ? 'p-3' : 'p-5'
      }`}
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="flex items-center gap-2 text-sm font-semibold text-white/85">
            <span className="inline-flex flex-shrink-0">
              <MapPinned className="h-4 w-4 text-emerald-400" />
            </span>
            {title}
          </h3>
          {subtitle && <p className="mt-1 text-[11px] text-white/35">{subtitle}</p>}
        </div>
        <span className="rounded-full bg-black/30 px-2.5 py-1 text-[10px] text-white/40">
          {showBackendHeatmap
            ? 'OpenCV 光斑云图 · 球心 (0,0)'
            : `${validPoints.length} 个落点 · 球心 (0,0)`}
        </span>
      </div>

      <div className="relative overflow-hidden rounded-2xl border border-emerald-400/15 bg-black/40 shadow-[inset_0_0_40px_rgba(16,185,129,0.08)]">
        {showBackendHeatmap ? (
          <img
            src={heatmapSrc!}
            alt="支撑脚与摆腿时空运动轨迹热力图"
            className="block w-full bg-black object-contain"
            style={{ minHeight: height, maxHeight: Math.max(height, 420) }}
          />
        ) : (
          <canvas ref={canvasRef} className="block w-full" />
        )}

        {!showBackendHeatmap && validPoints.length === 0 && (
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-2 bg-gradient-to-t from-black/50 via-black/20 to-transparent px-6 text-center">
            <motion.div
              initial={{ opacity: 0.4, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 1.2, repeat: Infinity, repeatType: 'reverse' }}
              className="flex h-12 w-12 items-center justify-center rounded-full border border-emerald-400/30 bg-emerald-500/10"
            >
              <Crosshair className="h-5 w-5 text-emerald-300/80" />
            </motion.div>
            <p className="text-sm font-medium text-white/70">空草坪待机</p>
            <p className="max-w-xs text-[11px] leading-relaxed text-white/40">
              完成射门分析后，支撑脚落点与摆腿光流轨迹将沉淀于此。原点为触球瞬间球心 (0,0)，比例尺 1 px = 0.5 cm。
            </p>
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-[11px] text-white/45">
          {showBackendHeatmap ? (
            <>
              🎯 触球原点锚定 · 支撑脚热力核 + 摆腿 15 帧光流
              <span className="ml-2 font-semibold tabular-nums text-white/80">1 px = 0.5 cm</span>
            </>
          ) : (
            <>
              🎯 空间离散度指数 (Dispersion Index)
              <span className="ml-2 font-semibold tabular-nums text-white/80">
                {validPoints.length >= 2 ? dispersion.toFixed(1) : '—'}
                {validPoints.length >= 2 ? ' cm²' : ''}
              </span>
            </>
          )}
        </div>
        {!showBackendHeatmap && (
          <span
            className={`inline-flex items-center rounded-full px-3 py-1 text-[11px] font-semibold ring-1 ${
              isStable === null
                ? 'bg-white/5 text-white/35 ring-white/10'
                : isStable
                  ? 'bg-emerald-500/15 text-emerald-300 ring-emerald-400/30'
                  : 'bg-rose-500/15 text-rose-300 ring-rose-400/30'
            }`}
          >
            {isStable === null
              ? '⏳ 等待多脚沉淀'
              : isStable
                ? '🟢 步点定型稳定'
                : '🔴 重心落点杂乱'}
          </span>
        )}
      </div>
    </section>
  )
}
