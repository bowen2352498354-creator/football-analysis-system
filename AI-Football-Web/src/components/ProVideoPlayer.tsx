import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import {
  Pause,
  Play,
  StepBack,
  StepForward,
  Crosshair,
  Clapperboard,
  UserRound,
  Scissors,
  Loader2,
} from 'lucide-react'
import type { PlaybackRate, SpatialTrajectoryData, TrajectoryPoint2D } from '../types'
import TelestrationCanvas, { type TelestrationCanvasHandle } from './TelestrationCanvas'

/* ============================================================================
 * 【V3·模块一 + 模块五】大师 vs 学员「双屏对齐播放器」ProVideoPlayer
 *
 * 对标 Dartfish：Side-by-Side 同步 · 触球绝对零点帧对齐 · 光流轨迹 Canvas
 * · 教师透明涂鸦层 / 手绘量角器 · 处方截图保存至 Word。
 * ========================================================================== */

const API_BASE_URL = 'http://localhost:8000'

const RATE_OPTIONS: { value: PlaybackRate; label: string }[] = [
  { value: 0.25, label: '0.25x 极慢速' },
  { value: 0.5, label: '0.5x 慢速' },
  { value: 1, label: '1.0x 正常' },
]

function estimateContactTimeSec(
  spatial: SpatialTrajectoryData | null | undefined,
  durationSec: number,
): number {
  if (!spatial || durationSec <= 0) return Math.max(0, durationSec * 0.55)
  const contact = spatial.contact_frame_index
  const total = spatial.sample_frame_count
  if (
    typeof contact === 'number' &&
    typeof total === 'number' &&
    total > 1 &&
    contact >= 0
  ) {
    return Math.max(0, Math.min(durationSec, (contact / Math.max(1, total - 1)) * durationSec))
  }
  return Math.max(0, durationSec * 0.55)
}

function drawPolyline(
  ctx: CanvasRenderingContext2D,
  points: TrajectoryPoint2D[],
  progress: number,
  stroke: string,
  glow: string,
  lineWidth: number,
) {
  if (points.length < 2) return
  const count = Math.max(2, Math.floor(points.length * Math.max(0.02, Math.min(1, progress))))
  const visible = points.slice(0, count)

  ctx.save()
  ctx.lineJoin = 'round'
  ctx.lineCap = 'round'
  ctx.shadowColor = glow
  ctx.shadowBlur = 12
  ctx.strokeStyle = stroke
  ctx.lineWidth = lineWidth
  ctx.beginPath()
  ctx.moveTo(visible[0][0], visible[0][1])
  for (let i = 1; i < visible.length; i += 1) {
    ctx.lineTo(visible[i][0], visible[i][1])
  }
  ctx.stroke()

  const tip = visible[visible.length - 1]
  const tipGlow = ctx.createRadialGradient(tip[0], tip[1], 0, tip[0], tip[1], 10)
  tipGlow.addColorStop(0, stroke)
  tipGlow.addColorStop(1, 'rgba(0,0,0,0)')
  ctx.fillStyle = tipGlow
  ctx.beginPath()
  ctx.arc(tip[0], tip[1], 10, 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()
}

function scalePathToViewport(
  path: TrajectoryPoint2D[],
  width: number,
  height: number,
): TrajectoryPoint2D[] {
  if (path.length === 0 || width <= 0 || height <= 0) return []
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity
  path.forEach(([x, y]) => {
    minX = Math.min(minX, x)
    maxX = Math.max(maxX, x)
    minY = Math.min(minY, y)
    maxY = Math.max(maxY, y)
  })
  const spanX = Math.max(40, maxX - minX)
  const spanY = Math.max(40, maxY - minY)
  const pad = 0.12
  const usableW = width * (1 - pad * 2)
  const usableH = height * (1 - pad * 2)
  const scale = Math.min(usableW / spanX, usableH / spanY)

  return path.map(([x, y]) => {
    const nx = width * pad + (x - minX) * scale + (usableW - spanX * scale) / 2
    const ny = height * pad + (y - minY) * scale + (usableH - spanY * scale) / 2
    return [nx, ny]
  })
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('image load failed'))
    img.src = src
  })
}

/** 左侧大师示范：无外部示范片时，用同步骨骼剪影动画占位 */
function MasterSkeletonStage({
  progress,
  contactProgress,
}: {
  progress: number
  contactProgress: number
}) {
  const phase = progress - contactProgress
  const swing = Math.sin(Math.max(-1.2, Math.min(1.2, phase * 4)) * Math.PI) * 38
  const fold = phase < 0 ? 28 + Math.abs(phase) * 40 : Math.max(8, 28 - phase * 50)

  return (
    <div className="relative flex h-full w-full items-center justify-center overflow-hidden bg-gradient-to-br from-zinc-950 via-[#0a1612] to-black">
      <div className="pointer-events-none absolute inset-0 opacity-40 [background:radial-gradient(ellipse_at_center,rgba(52,211,153,0.18),transparent_65%)]" />
      <svg viewBox="0 0 200 260" className="h-[88%] w-auto drop-shadow-[0_0_24px_rgba(52,211,153,0.25)]">
        <defs>
          <linearGradient id="boneGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#6ee7b7" />
            <stop offset="100%" stopColor="#34d399" />
          </linearGradient>
        </defs>
        <circle cx="100" cy="48" r="16" fill="none" stroke="url(#boneGrad)" strokeWidth="2.5" opacity="0.85" />
        <line x1="100" y1="64" x2="100" y2="130" stroke="url(#boneGrad)" strokeWidth="3" strokeLinecap="round" />
        <line x1="70" y1="80" x2="130" y2="80" stroke="url(#boneGrad)" strokeWidth="2.5" strokeLinecap="round" />
        <line x1="100" y1="130" x2="78" y2="175" stroke="url(#boneGrad)" strokeWidth="3" strokeLinecap="round" />
        <line x1="78" y1="175" x2="72" y2="220" stroke="url(#boneGrad)" strokeWidth="3" strokeLinecap="round" />
        <circle cx="70" cy="228" r="5" fill="#a7f3d0" />
        <g transform={`rotate(${swing} 100 130)`}>
          <line x1="100" y1="130" x2={118} y2={160 - fold * 0.3} stroke="#34d399" strokeWidth="3.2" strokeLinecap="round" />
          <line
            x1={118}
            y1={160 - fold * 0.3}
            x2={130 + swing * 0.15}
            y2={210 - fold}
            stroke="#6ee7b7"
            strokeWidth="3.2"
            strokeLinecap="round"
          />
          <circle cx={132 + swing * 0.15} cy={218 - fold} r="5" fill="#fbbf24" />
        </g>
        <circle cx="155" cy="222" r="9" fill="#fafafa" opacity="0.9" />
        <circle cx="155" cy="222" r="9" fill="none" stroke="#fbbf24" strokeWidth="1.5" opacity="0.7" />
      </svg>
      <span className="absolute left-3 top-3 rounded-full border border-emerald-400/25 bg-black/50 px-2.5 py-1 text-[10px] font-medium text-emerald-300 backdrop-blur-md">
        标准职业示范 · 骨骼模型
      </span>
    </div>
  )
}

export interface ProVideoPlayerProps {
  studentVideoUrl?: string | null
  masterVideoUrl?: string | null
  studentPosterUrl?: string | null
  spatialTrajectoryData?: SpatialTrajectoryData | null
  title?: string
  /** 【V3·模块五】当前尝试 ID，用于手绘处方写盘 */
  attemptId?: string | null
  /** 是否启用涂鸦层（默认 true） */
  enableTelestration?: boolean
  /** 保存成功/失败回调（可选，供父级 Toast） */
  onTelestrationSaved?: (ok: boolean, message: string) => void
}

export default function ProVideoPlayer({
  studentVideoUrl = null,
  masterVideoUrl = null,
  studentPosterUrl = null,
  spatialTrajectoryData = null,
  title = '大师 vs 学员 · 双屏对齐播控',
  attemptId = null,
  enableTelestration = true,
  onTelestrationSaved,
}: ProVideoPlayerProps) {
  const studentVideoRef = useRef<HTMLVideoElement>(null)
  const masterVideoRef = useRef<HTMLVideoElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const studentStageRef = useRef<HTMLDivElement>(null)
  const posterImgRef = useRef<HTMLImageElement>(null)
  const telestrationRef = useRef<TelestrationCanvasHandle>(null)

  const [rate, setRate] = useState<PlaybackRate>(0.5)
  const [isPlaying, setIsPlaying] = useState(false)
  const [duration, setDuration] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [ready, setReady] = useState(false)
  const [isSavingTelestration, setIsSavingTelestration] = useState(false)
  const [saveHint, setSaveHint] = useState<string | null>(null)

  const contactTime = estimateContactTimeSec(spatialTrajectoryData, duration || 1)
  const contactProgress = duration > 0 ? contactTime / duration : 0.55
  const playProgress = duration > 0 ? currentTime / duration : 0

  useEffect(() => {
    const canvas = overlayRef.current
    const stage = studentStageRef.current
    if (!canvas || !stage) return

    const width = stage.clientWidth
    const height = stage.clientHeight
    if (width <= 0 || height <= 0) return

    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = Math.floor(width * dpr)
    canvas.height = Math.floor(height * dpr)
    canvas.style.width = `${width}px`
    canvas.style.height = `${height}px`

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, width, height)

    const swing = spatialTrajectoryData?.swing_leg_path ?? []
    const ball = spatialTrajectoryData?.ball_flight_path ?? []
    if (swing.length === 0 && ball.length === 0) return

    const relative = playProgress - contactProgress
    const swingProgress = Math.max(0, Math.min(1, (relative + 0.45) / 0.9))
    const ballProgress = relative < 0 ? 0 : Math.max(0, Math.min(1, relative / 0.35))

    const scaledSwing = scalePathToViewport(swing, width, height)
    const scaledBall = scalePathToViewport(ball, width, height)

    drawPolyline(ctx, scaledSwing, swingProgress, 'rgba(52, 211, 153, 0.95)', 'rgba(52, 211, 153, 0.85)', 3.2)
    if (ballProgress > 0) {
      drawPolyline(ctx, scaledBall, ballProgress, 'rgba(251, 191, 36, 0.95)', 'rgba(251, 191, 36, 0.8)', 2.8)
    }
  }, [spatialTrajectoryData, playProgress, contactProgress])

  useEffect(() => {
    if (studentVideoRef.current) studentVideoRef.current.playbackRate = rate
    if (masterVideoRef.current) masterVideoRef.current.playbackRate = rate
  }, [rate])

  function syncMasterToStudent() {
    const student = studentVideoRef.current
    const master = masterVideoRef.current
    if (!student || !master || !Number.isFinite(student.duration) || student.duration <= 0) return
    if (!Number.isFinite(master.duration) || master.duration <= 0) return
    const studentContact = estimateContactTimeSec(spatialTrajectoryData, student.duration)
    const masterContact = master.duration * (studentContact / Math.max(student.duration, 0.001))
    const offset = student.currentTime - studentContact
    const target = Math.max(0, Math.min(master.duration, masterContact + offset))
    if (Math.abs(master.currentTime - target) > 0.04) {
      master.currentTime = target
    }
  }

  async function handleTogglePlay() {
    const student = studentVideoRef.current
    if (!student) {
      setIsPlaying((prev) => !prev)
      return
    }
    if (student.paused) {
      try {
        await student.play()
        await masterVideoRef.current?.play().catch(() => undefined)
        setIsPlaying(true)
      } catch {
        setIsPlaying(false)
      }
    } else {
      student.pause()
      masterVideoRef.current?.pause()
      setIsPlaying(false)
    }
  }

  function seekBoth(nextTime: number) {
    const student = studentVideoRef.current
    if (student && Number.isFinite(student.duration)) {
      student.currentTime = Math.max(0, Math.min(student.duration, nextTime))
      setCurrentTime(student.currentTime)
      syncMasterToStudent()
    } else {
      setCurrentTime(Math.max(0, nextTime))
    }
  }

  function handleStep(direction: 'forward' | 'backward') {
    const student = studentVideoRef.current
    student?.pause()
    masterVideoRef.current?.pause()
    setIsPlaying(false)
    const frameDt = 1 / 30
    const base = student?.currentTime ?? currentTime
    seekBoth(base + (direction === 'forward' ? frameDt : -frameDt))
  }

  function handleFreezeContact() {
    const student = studentVideoRef.current
    student?.pause()
    masterVideoRef.current?.pause()
    setIsPlaying(false)
    const dur = student?.duration || duration || 1
    const t = estimateContactTimeSec(spatialTrajectoryData, dur)
    seekBoth(t)
  }

  /** 合并视频定格帧 + 轨迹层 + 涂鸦层 → Base64 PNG */
  async function composeTelestrationPrescription(): Promise<string | null> {
    const stage = studentStageRef.current
    if (!stage) return null
    const width = stage.clientWidth
    const height = stage.clientHeight
    if (width <= 0 || height <= 0) return null

    // 定格：暂停播放，保证帧一致
    studentVideoRef.current?.pause()
    masterVideoRef.current?.pause()
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

    const video = studentVideoRef.current
    const poster = posterImgRef.current

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
      if (video && video.readyState >= 2 && video.videoWidth > 0) {
        drawContain(video, video.videoWidth, video.videoHeight)
      } else if (poster && poster.complete && poster.naturalWidth > 0) {
        drawContain(poster, poster.naturalWidth, poster.naturalHeight)
      } else if (studentPosterUrl) {
        const img = await loadImage(studentPosterUrl)
        drawContain(img, img.naturalWidth, img.naturalHeight)
      }
    } catch {
      /* 底层帧缺失时仍导出涂鸦层 */
    }

    // 轨迹叠加层
    if (overlayRef.current) {
      ctx.drawImage(overlayRef.current, 0, 0, width, height)
    }

    // 涂鸦层
    const telestrationUrl = telestrationRef.current?.exportLayerDataUrl()
    if (telestrationUrl) {
      try {
        const layer = await loadImage(telestrationUrl)
        ctx.drawImage(layer, 0, 0, width, height)
      } catch {
        /* ignore layer decode */
      }
    }

    return exportCanvas.toDataURL('image/png')
  }

  async function handleSaveTelestration() {
    if (!attemptId) {
      const msg = '请先选中一条已归档的 Attempt，再保存手绘涂鸦处方'
      setSaveHint(msg)
      onTelestrationSaved?.(false, msg)
      return
    }
    setIsSavingTelestration(true)
    setSaveHint(null)
    try {
      const imageBase64 = await composeTelestrationPrescription()
      if (!imageBase64) throw new Error('无法合成涂鸦处方图')

      const response = await fetch(`${API_BASE_URL}/api/save_telestration_image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          attemptId,
          imageBase64,
        }),
      })
      const data = (await response.json()) as {
        success: boolean
        message?: string
        path?: string
      }
      if (!data.success) throw new Error(data.message || '保存失败')
      const msg = data.message || '手绘涂鸦处方已写入 Word 诊断单'
      setSaveHint(msg)
      onTelestrationSaved?.(true, msg)
    } catch (error) {
      const msg = error instanceof Error ? error.message : '保存手绘处方失败'
      setSaveHint(msg)
      onTelestrationSaved?.(false, msg)
    } finally {
      setIsSavingTelestration(false)
    }
  }

  const hasStudentMedia = Boolean(studentVideoUrl || studentPosterUrl)
  const showEmpty = !hasStudentMedia && !spatialTrajectoryData

  return (
    <section className="pro-video-player flex flex-col gap-3 rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-xl">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-white/85">
          <span className="inline-flex flex-shrink-0">
            <Clapperboard className="h-4 w-4 text-amber-300" />
          </span>
          {title}
        </h3>
        <span className="rounded-full bg-black/30 px-2.5 py-1 text-[10px] text-white/40">
          触球绝对零点 · Frame {spatialTrajectoryData?.contact_frame_index ?? '—'}
        </span>
      </div>

      {showEmpty ? (
        <div className="flex min-h-[220px] flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-white/10 bg-black/30 px-6 text-center">
          <UserRound className="h-8 w-8 text-white/25" />
          <p className="text-sm text-white/50">等待学员实测视频与时空轨迹就绪</p>
          <p className="text-[11px] text-white/30">
            完成本地视频分析后，此处将自动进入双屏对齐与光流轨迹涂鸦模式
          </p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <div className="relative aspect-video overflow-hidden rounded-2xl border border-white/10 bg-black shadow-inner">
              {masterVideoUrl ? (
                <video
                  ref={masterVideoRef}
                  src={masterVideoUrl}
                  muted
                  playsInline
                  className="h-full w-full object-contain"
                  onLoadedMetadata={(e) => {
                    const v = e.currentTarget
                    const contact = v.duration * contactProgress
                    v.currentTime = Math.max(0, contact - 0.35)
                  }}
                />
              ) : (
                <MasterSkeletonStage progress={playProgress} contactProgress={contactProgress} />
              )}
              <span className="absolute bottom-2 left-2 rounded-full bg-black/55 px-2 py-0.5 text-[10px] text-emerald-200 backdrop-blur">
                左屏 · 标准示范
              </span>
            </div>

            {/* 右屏：学员实测 + 轨迹 + 教师涂鸦层 */}
            <div
              ref={studentStageRef}
              className="relative aspect-video overflow-hidden rounded-2xl border border-amber-400/20 bg-black shadow-inner"
            >
              {studentVideoUrl ? (
                <video
                  ref={studentVideoRef}
                  src={studentVideoUrl}
                  playsInline
                  className="h-full w-full object-contain"
                  onLoadedMetadata={(e) => {
                    const v = e.currentTarget
                    setDuration(v.duration || 0)
                    setReady(true)
                    const t = estimateContactTimeSec(spatialTrajectoryData, v.duration || 1)
                    v.currentTime = Math.max(0, t - 0.35)
                    setCurrentTime(v.currentTime)
                    syncMasterToStudent()
                  }}
                  onTimeUpdate={(e) => {
                    setCurrentTime(e.currentTarget.currentTime)
                    syncMasterToStudent()
                  }}
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onEnded={() => setIsPlaying(false)}
                />
              ) : studentPosterUrl ? (
                <img
                  ref={posterImgRef}
                  src={studentPosterUrl}
                  alt="学员击球关键帧"
                  className="h-full w-full object-contain"
                  onLoad={() => {
                    setDuration(1)
                    setCurrentTime(contactProgress)
                    setReady(true)
                  }}
                />
              ) : (
                <div className="flex h-full items-center justify-center text-xs text-white/30">无学员画面</div>
              )}
              <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" />
              {enableTelestration && (
                <TelestrationCanvas
                  ref={telestrationRef}
                  drawingEnabled
                  showToolbar
                />
              )}
              <span className="pointer-events-none absolute bottom-2 left-2 z-10 rounded-full bg-black/55 px-2 py-0.5 text-[10px] text-amber-200 backdrop-blur">
                右屏 · 学员实测 · 手绘层
              </span>
            </div>
          </div>

          <div className="flex flex-col gap-2 rounded-2xl bg-black/25 px-3 py-3">
            <div className="relative h-1.5 overflow-hidden rounded-full bg-white/10">
              <motion.div
                className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-emerald-400 via-amber-300 to-amber-400"
                animate={{ width: `${Math.max(0, Math.min(100, playProgress * 100))}%` }}
                transition={{ duration: 0.1, ease: 'linear' }}
              />
              <div
                className="absolute top-1/2 h-3 w-0.5 -translate-y-1/2 bg-white shadow-[0_0_8px_rgba(255,255,255,0.9)]"
                style={{ left: `${contactProgress * 100}%` }}
                title="触球绝对零点"
              />
            </div>

            <div className="flex flex-wrap items-center justify-center gap-2">
              <div className="inline-flex items-center gap-1 rounded-full bg-black/40 p-0.5">
                {RATE_OPTIONS.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => setRate(option.value)}
                    className={`rounded-full px-2.5 py-1 text-[11px] font-semibold transition ${
                      rate === option.value
                        ? 'bg-emerald-400 text-black'
                        : 'text-white/50 hover:text-white/80'
                    }`}
                  >
                    {option.label}
                  </button>
                ))}
              </div>

              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => handleStep('backward')}
                  title="逐帧后退"
                  className="flex h-8 w-8 items-center justify-center rounded-full bg-white/10 text-white/70 transition hover:bg-white/20 active:scale-95"
                >
                  <StepBack className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => void handleTogglePlay()}
                  disabled={!ready && !studentPosterUrl}
                  title={isPlaying ? '暂停' : '播放'}
                  className="flex h-9 w-9 items-center justify-center rounded-full bg-amber-400 text-black transition hover:bg-amber-300 active:scale-95 disabled:cursor-not-allowed disabled:bg-white/10 disabled:text-white/30"
                >
                  {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                </button>
                <button
                  type="button"
                  onClick={() => handleStep('forward')}
                  title="逐帧前进"
                  className="flex h-8 w-8 items-center justify-center rounded-full bg-white/10 text-white/70 transition hover:bg-white/20 active:scale-95"
                >
                  <StepForward className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  onClick={handleFreezeContact}
                  title="定格触球帧"
                  className="ml-1 inline-flex items-center gap-1 rounded-full border border-white/15 bg-white/10 px-3 py-1.5 text-[11px] font-semibold text-white/80 transition hover:bg-white/20 active:scale-95"
                >
                  <Crosshair className="h-3 w-3 text-amber-300" />
                  定格触球帧
                </button>
              </div>
            </div>

            <p className="text-center text-[10px] text-white/30">
              绝对零点对齐 · 亮绿摆动腿弧线 / 金黄球路射线 · 右上角教师手绘工具条
              {duration > 0 && (
                <span className="ml-2 tabular-nums text-white/45">
                  {currentTime.toFixed(2)}s / {duration.toFixed(2)}s
                </span>
              )}
            </p>
          </div>

          {/* 【V3·模块五】重磅：保存手绘涂鸦处方至 Word */}
          {enableTelestration && (
            <div className="flex flex-col items-center gap-2">
              <button
                type="button"
                onClick={() => void handleSaveTelestration()}
                disabled={isSavingTelestration}
                className="telestration-save-btn group relative flex w-full max-w-xl items-center justify-center gap-2 overflow-hidden rounded-2xl border-2 border-rose-400/45 bg-gradient-to-r from-rose-500/25 via-amber-400/15 to-transparent px-5 py-3.5 text-sm font-bold text-rose-50 shadow-[0_0_28px_rgba(255,59,48,0.18)] transition hover:border-rose-300/70 hover:shadow-[0_0_36px_rgba(255,59,48,0.28)] active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-55"
              >
                <span className="pointer-events-none absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent opacity-0 transition-opacity group-hover:opacity-100" />
                {isSavingTelestration ? (
                  <Loader2 className="relative h-4 w-4 animate-spin" />
                ) : (
                  <Scissors className="relative h-4 w-4 text-rose-200" />
                )}
                <span className="relative">✂️ 保存手绘涂鸦处方至 Word</span>
              </button>
              {saveHint && (
                <p className="max-w-xl text-center text-[11px] leading-relaxed text-white/45">{saveHint}</p>
              )}
              {!attemptId && (
                <p className="text-[10px] text-white/25">
                  提示：在教练看板选中 Attempt 后，手绘处方将写入该次诊断单主图
                </p>
              )}
            </div>
          )}
        </>
      )}
    </section>
  )
}
