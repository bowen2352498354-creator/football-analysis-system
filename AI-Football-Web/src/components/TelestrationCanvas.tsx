import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from 'react'
import {
  Highlighter,
  MoveRight,
  PenLine,
  Ruler,
  Undo2,
  Trash2,
} from 'lucide-react'

/* ============================================================================
 * 【V3.1 Sprint 3】Hudl Sportscode 风格教练端手绘电烙铁 TelestrationCanvas
 *
 * 绝对定位覆盖视频视口；默认 pointer-events: none，教练「开启画笔」后激活。
 * 坐标系与视口 CSS 像素 1:1（内部按 DPR 缩放）。
 * 工具：自由画笔 · 力学方向箭头 · 3 点动态量角器 · 撤销 / 清空。
 * ========================================================================== */

export type TelestrationTool = 'pen' | 'arrow' | 'protractor' | 'none'
/** Sprint 3 规范：醒目亮红 / 荧光绿 */
export type PenColor = '#EF4444' | '#10B981'

export interface TelestrationCanvasHandle {
  /** 导出与视口同尺寸的透明 PNG（仅涂鸦层） */
  exportLayerDataUrl: () => string | null
  /** 是否存在未清空的笔画 */
  hasStrokes: () => boolean
  clearAll: () => void
  /** 程序化开关画笔（与工具栏「开启画笔」等效） */
  setDrawingEnabled: (enabled: boolean) => void
}

interface Point {
  x: number
  y: number
}

type Stroke =
  | { kind: 'pen'; color: PenColor; width: number; points: Point[] }
  | { kind: 'arrow'; color: string; from: Point; to: Point }
  | { kind: 'protractor'; a: Point; b: Point; c: Point; angleDeg: number }

export interface TelestrationCanvasProps {
  className?: string
  /**
   * 涂鸦交互总开关。关闭时 canvas 为 pointer-events: none，便于播控穿透。
   * 不传则组件内部自行管理（工具栏「开启画笔」）。
   */
  drawingEnabled?: boolean
  onDrawingEnabledChange?: (enabled: boolean) => void
  /** 是否渲染右上角浮动工具条（父级自带工具栏时可关） */
  showToolbar?: boolean
}

const PEN_WIDTH = 4
const ARROW_COLOR = '#38BDF8'
const PROTRACTOR_COLOR = '#FFFFFF'
const DEFAULT_PEN: PenColor = '#EF4444'

function dist(a: Point, b: Point): number {
  return Math.hypot(b.x - a.x, b.y - a.y)
}

function angleAtVertex(a: Point, vertex: Point, c: Point): number {
  const v1 = { x: a.x - vertex.x, y: a.y - vertex.y }
  const v2 = { x: c.x - vertex.x, y: c.y - vertex.y }
  const dot = v1.x * v2.x + v1.y * v2.y
  const cross = v1.x * v2.y - v1.y * v2.x
  const rad = Math.atan2(cross, dot)
  return Math.abs((rad * 180) / Math.PI)
}

function drawArrowHead(
  ctx: CanvasRenderingContext2D,
  from: Point,
  to: Point,
  size = 14,
) {
  const angle = Math.atan2(to.y - from.y, to.x - from.x)
  ctx.beginPath()
  ctx.moveTo(to.x, to.y)
  ctx.lineTo(to.x - size * Math.cos(angle - Math.PI / 6), to.y - size * Math.sin(angle - Math.PI / 6))
  ctx.lineTo(to.x - size * Math.cos(angle + Math.PI / 6), to.y - size * Math.sin(angle + Math.PI / 6))
  ctx.closePath()
  ctx.fill()
}

function paintStroke(ctx: CanvasRenderingContext2D, stroke: Stroke) {
  if (stroke.kind === 'pen') {
    if (stroke.points.length < 2) return
    ctx.save()
    ctx.lineJoin = 'round'
    ctx.lineCap = 'round'
    ctx.strokeStyle = stroke.color
    ctx.globalAlpha = 0.92
    ctx.lineWidth = stroke.width
    ctx.beginPath()
    ctx.moveTo(stroke.points[0].x, stroke.points[0].y)
    for (let i = 1; i < stroke.points.length; i += 1) {
      ctx.lineTo(stroke.points[i].x, stroke.points[i].y)
    }
    ctx.stroke()
    ctx.restore()
    return
  }

  if (stroke.kind === 'arrow') {
    ctx.save()
    ctx.strokeStyle = stroke.color
    ctx.fillStyle = stroke.color
    ctx.lineWidth = 3.2
    ctx.lineCap = 'round'
    ctx.beginPath()
    ctx.moveTo(stroke.from.x, stroke.from.y)
    ctx.lineTo(stroke.to.x, stroke.to.y)
    ctx.stroke()
    drawArrowHead(ctx, stroke.from, stroke.to)
    ctx.restore()
    return
  }

  const { a, b, c, angleDeg } = stroke
  ctx.save()
  ctx.strokeStyle = PROTRACTOR_COLOR
  ctx.fillStyle = PROTRACTOR_COLOR
  ctx.lineWidth = 2.4
  ctx.lineCap = 'round'

  ctx.beginPath()
  ctx.moveTo(a.x, a.y)
  ctx.lineTo(b.x, b.y)
  ctx.lineTo(c.x, c.y)
  ctx.stroke()

  for (const p of [a, b, c]) {
    ctx.beginPath()
    ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2)
    ctx.fill()
  }

  const start = Math.atan2(a.y - b.y, a.x - b.x)
  const end = Math.atan2(c.y - b.y, c.x - b.x)
  let delta = end - start
  while (delta <= -Math.PI) delta += Math.PI * 2
  while (delta > Math.PI) delta -= Math.PI * 2
  const radius = Math.min(36, dist(a, b) * 0.35, dist(c, b) * 0.35)
  ctx.beginPath()
  ctx.arc(b.x, b.y, radius, start, start + delta, delta < 0)
  ctx.stroke()

  const mid = start + delta / 2
  const labelR = radius + 18
  const lx = b.x + Math.cos(mid) * labelR
  const ly = b.y + Math.sin(mid) * labelR
  const label = `${angleDeg.toFixed(1)}°`
  ctx.font = '700 15px "SF Pro Display", "PingFang SC", system-ui, sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillStyle = 'rgba(0,0,0,0.55)'
  const metrics = ctx.measureText(label)
  const padX = 8
  const padY = 5
  ctx.fillRect(
    lx - metrics.width / 2 - padX,
    ly - 10 - padY,
    metrics.width + padX * 2,
    20 + padY * 2,
  )
  ctx.fillStyle = '#FBBF24'
  ctx.fillText(label, lx, ly)
  ctx.restore()
}

const TelestrationCanvas = forwardRef<TelestrationCanvasHandle, TelestrationCanvasProps>(
  function TelestrationCanvas(
    {
      className = '',
      drawingEnabled: drawingEnabledProp,
      onDrawingEnabledChange,
      showToolbar = true,
    },
    ref,
  ) {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const containerRef = useRef<HTMLDivElement>(null)
    const strokesRef = useRef<Stroke[]>([])
    const [strokes, setStrokes] = useState<Stroke[]>([])
    const [internalDrawing, setInternalDrawing] = useState(false)
    const drawingEnabled = drawingEnabledProp ?? internalDrawing
    const [tool, setTool] = useState<TelestrationTool>('pen')
    const [penColor, setPenColor] = useState<PenColor>(DEFAULT_PEN)
    const [protractorDraft, setProtractorDraft] = useState<Point[]>([])

    const drawingRef = useRef(false)
    const draftPenRef = useRef<Point[]>([])
    const draftArrowFromRef = useRef<Point | null>(null)
    const draftArrowToRef = useRef<Point | null>(null)
    const sizeRef = useRef({ w: 0, h: 0, dpr: 1 })

    const setDrawingEnabled = useCallback(
      (enabled: boolean) => {
        if (drawingEnabledProp === undefined) setInternalDrawing(enabled)
        onDrawingEnabledChange?.(enabled)
        if (enabled) setTool((t) => (t === 'none' ? 'pen' : t))
        else {
          drawingRef.current = false
          draftPenRef.current = []
          draftArrowFromRef.current = null
          draftArrowToRef.current = null
        }
      },
      [drawingEnabledProp, onDrawingEnabledChange],
    )

    const canDraw = drawingEnabled && tool !== 'none'

    const redraw = useCallback(
      (ctx?: CanvasRenderingContext2D | null) => {
        const canvas = canvasRef.current
        if (!canvas) return
        const context = ctx ?? canvas.getContext('2d')
        if (!context) return
        const { w, h, dpr } = sizeRef.current
        context.setTransform(dpr, 0, 0, dpr, 0, 0)
        context.clearRect(0, 0, w, h)
        for (const stroke of strokesRef.current) paintStroke(context, stroke)

        if (draftPenRef.current.length >= 2) {
          paintStroke(context, {
            kind: 'pen',
            color: penColor,
            width: PEN_WIDTH,
            points: draftPenRef.current,
          })
        }
        if (draftArrowFromRef.current && draftArrowToRef.current) {
          paintStroke(context, {
            kind: 'arrow',
            color: ARROW_COLOR,
            from: draftArrowFromRef.current,
            to: draftArrowToRef.current,
          })
        }
        const draft = protractorDraft
        if (draft.length > 0) {
          context.save()
          context.fillStyle = '#FBBF24'
          for (const p of draft) {
            context.beginPath()
            context.arc(p.x, p.y, 5, 0, Math.PI * 2)
            context.fill()
          }
          if (draft.length >= 2) {
            context.strokeStyle = 'rgba(255,255,255,0.7)'
            context.lineWidth = 2
            context.beginPath()
            context.moveTo(draft[0].x, draft[0].y)
            for (let i = 1; i < draft.length; i += 1) {
              context.lineTo(draft[i].x, draft[i].y)
            }
            context.stroke()
          }
          context.restore()
        }
      },
      [penColor, protractorDraft],
    )

    const syncSize = useCallback(() => {
      const canvas = canvasRef.current
      const host = containerRef.current
      if (!canvas || !host) return
      const w = host.clientWidth
      const h = host.clientHeight
      if (w <= 0 || h <= 0) return
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      sizeRef.current = { w, h, dpr }
      canvas.width = Math.floor(w * dpr)
      canvas.height = Math.floor(h * dpr)
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      redraw(ctx)
    }, [redraw])

    useEffect(() => {
      strokesRef.current = strokes
      redraw()
    }, [strokes, redraw])

    useEffect(() => {
      syncSize()
      const host = containerRef.current
      if (!host) return
      const ro = new ResizeObserver(() => syncSize())
      ro.observe(host)
      return () => ro.disconnect()
    }, [syncSize])

    useEffect(() => {
      redraw()
    }, [protractorDraft, redraw])

    useImperativeHandle(ref, () => ({
      exportLayerDataUrl: () => {
        const canvas = canvasRef.current
        if (!canvas) return null
        redraw()
        return canvas.toDataURL('image/png')
      },
      hasStrokes: () => strokesRef.current.length > 0,
      clearAll: () => {
        setStrokes([])
        setProtractorDraft([])
        draftPenRef.current = []
        draftArrowFromRef.current = null
        draftArrowToRef.current = null
      },
      setDrawingEnabled,
    }))

    function clientToLocal(clientX: number, clientY: number): Point {
      const canvas = canvasRef.current
      if (!canvas) return { x: 0, y: 0 }
      const rect = canvas.getBoundingClientRect()
      return {
        x: ((clientX - rect.left) / rect.width) * sizeRef.current.w,
        y: ((clientY - rect.top) / rect.height) * sizeRef.current.h,
      }
    }

    function commitStroke(stroke: Stroke) {
      setStrokes((prev) => [...prev, stroke])
    }

    function handlePointerDown(event: ReactPointerEvent<HTMLCanvasElement>) {
      if (!canDraw) return
      event.preventDefault()
      event.currentTarget.setPointerCapture(event.pointerId)
      const p = clientToLocal(event.clientX, event.clientY)

      if (tool === 'pen') {
        drawingRef.current = true
        draftPenRef.current = [p]
        redraw()
        return
      }
      if (tool === 'arrow') {
        drawingRef.current = true
        draftArrowFromRef.current = p
        draftArrowToRef.current = p
        redraw()
        return
      }
      if (tool === 'protractor') {
        setProtractorDraft((prev) => {
          const next = [...prev, p]
          if (next.length >= 3) {
            const [a, b, c] = next
            const angleDeg = angleAtVertex(a, b, c)
            commitStroke({ kind: 'protractor', a, b, c, angleDeg })
            return []
          }
          return next
        })
      }
    }

    function handlePointerMove(event: ReactPointerEvent<HTMLCanvasElement>) {
      if (!canDraw || !drawingRef.current) return
      event.preventDefault()
      const p = clientToLocal(event.clientX, event.clientY)
      if (tool === 'pen') {
        draftPenRef.current = [...draftPenRef.current, p]
        redraw()
      } else if (tool === 'arrow') {
        draftArrowToRef.current = p
        redraw()
      }
    }

    function handlePointerUp(event: ReactPointerEvent<HTMLCanvasElement>) {
      if (!canDraw) return
      event.preventDefault()
      if (tool === 'pen' && drawingRef.current) {
        const points = draftPenRef.current
        if (points.length >= 2) {
          commitStroke({ kind: 'pen', color: penColor, width: PEN_WIDTH, points })
        }
        draftPenRef.current = []
      }
      if (tool === 'arrow' && drawingRef.current) {
        const from = draftArrowFromRef.current
        const to = draftArrowToRef.current
        if (from && to && dist(from, to) > 8) {
          commitStroke({ kind: 'arrow', color: ARROW_COLOR, from, to })
        }
        draftArrowFromRef.current = null
        draftArrowToRef.current = null
      }
      drawingRef.current = false
      try {
        event.currentTarget.releasePointerCapture(event.pointerId)
      } catch {
        /* ignore */
      }
    }

    function handleUndo() {
      setStrokes((prev) => prev.slice(0, -1))
      setProtractorDraft([])
    }

    function handleClear() {
      setStrokes([])
      setProtractorDraft([])
      draftPenRef.current = []
      draftArrowFromRef.current = null
      draftArrowToRef.current = null
    }

    const toolBtn = (active: boolean) =>
      `flex h-9 w-9 items-center justify-center rounded-xl border transition active:scale-95 ${
        active
          ? 'border-white/40 bg-white/20 text-white shadow-[0_0_12px_rgba(255,255,255,0.15)]'
          : 'border-white/10 bg-black/45 text-white/55 hover:bg-white/10 hover:text-white/85'
      }`

    return (
      <div
        ref={containerRef}
        className={`telestration-layer absolute inset-0 z-[15] ${className}`}
        style={{ touchAction: canDraw ? 'none' : 'auto' }}
      >
        <canvas
          ref={canvasRef}
          className={`absolute inset-0 h-full w-full ${canDraw ? 'cursor-crosshair' : ''}`}
          style={{ pointerEvents: canDraw ? 'auto' : 'none' }}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
        />

        {showToolbar && (
          <div className="telestration-toolbar pointer-events-auto absolute right-2 top-2 z-20 flex flex-col items-end gap-1.5">
            <button
              type="button"
              title={drawingEnabled ? '关闭画笔（穿透播控）' : '开启画笔'}
              onClick={() => setDrawingEnabled(!drawingEnabled)}
              className={`inline-flex items-center gap-1.5 rounded-xl border px-2.5 py-1.5 text-[11px] font-semibold backdrop-blur-xl transition active:scale-95 ${
                drawingEnabled
                  ? 'border-emerald-400/50 bg-emerald-500/25 text-emerald-100'
                  : 'border-white/15 bg-black/55 text-white/80 hover:bg-white/10'
              }`}
            >
              <PenLine className="h-3.5 w-3.5" />
              {drawingEnabled ? '关闭画笔' : '✍️开启画笔'}
            </button>

            {drawingEnabled && (
              <div className="flex items-center gap-1 rounded-2xl border border-white/15 bg-black/55 p-1 backdrop-blur-xl">
                <button
                  type="button"
                  title="自由画笔"
                  className={toolBtn(tool === 'pen')}
                  onClick={() => setTool('pen')}
                >
                  <Highlighter className="h-4 w-4" />
                </button>
                <button
                  type="button"
                  title="亮红色 #EF4444"
                  className={`h-7 w-7 rounded-full border-2 transition ${
                    penColor === '#EF4444' ? 'border-white scale-110' : 'border-transparent opacity-70'
                  }`}
                  style={{ backgroundColor: '#EF4444' }}
                  onClick={() => {
                    setPenColor('#EF4444')
                    setTool('pen')
                  }}
                />
                <button
                  type="button"
                  title="荧光绿 #10B981"
                  className={`h-7 w-7 rounded-full border-2 transition ${
                    penColor === '#10B981' ? 'border-white scale-110' : 'border-transparent opacity-70'
                  }`}
                  style={{ backgroundColor: '#10B981' }}
                  onClick={() => {
                    setPenColor('#10B981')
                    setTool('pen')
                  }}
                />
                <button
                  type="button"
                  title="力学方向箭头"
                  className={toolBtn(tool === 'arrow')}
                  onClick={() => setTool('arrow')}
                >
                  <MoveRight className="h-4 w-4" />
                </button>
                <button
                  type="button"
                  title="3点动态量角器（依次点 3 点，顶点为第 2 点）"
                  className={toolBtn(tool === 'protractor')}
                  onClick={() => {
                    setTool('protractor')
                    setProtractorDraft([])
                  }}
                >
                  <Ruler className="h-4 w-4" />
                </button>
                <button type="button" title="撤销" className={toolBtn(false)} onClick={handleUndo}>
                  <Undo2 className="h-4 w-4" />
                </button>
                <button
                  type="button"
                  title="🗑️清除"
                  className={toolBtn(false)}
                  onClick={handleClear}
                >
                  <Trash2 className="h-4 w-4 text-rose-300" />
                </button>
              </div>
            )}
            {drawingEnabled && tool === 'protractor' && (
              <span className="rounded-full border border-amber-400/30 bg-black/60 px-2.5 py-1 text-[10px] text-amber-200 backdrop-blur">
                量角器 · 已点 {protractorDraft.length}/3（第 2 点为夹角顶点）
              </span>
            )}
          </div>
        )}
      </div>
    )
  },
)

export default TelestrationCanvas
