import { useEffect, useRef, type ReactNode } from 'react'
import { Clapperboard, Crosshair } from 'lucide-react'
import type { MetricSeekEvent } from '../types'

/** 触球窗口半宽：波形 X 轴 0..60 对应 [t_impact-30, t_impact+30] */
export const IMPACT_WINDOW_HALF = 30

export interface VideoSeekRequest {
  /** 物理极值帧索引（0-based，绝对视频帧） */
  frameIndex: number
  /** 采样总帧数（用于把帧映射到时间轴比例） */
  sampleFrameCount?: number | null
  /** 假定帧率，默认 30 */
  fps?: number
  /** 指标中文名（HUD 提示） */
  label?: string
  /** 递增 token，保证同一帧重复点击也能触发 */
  token: number
}

export interface ChartSeekRequest {
  /** 波形窗口内索引 x_index（0~60） */
  xIndex: number
  /** 递增 token，保证同一点重复点击也能触发 */
  token: number
}

export interface VideoWorkspaceProps {
  /** 核心视频视口（实时帧 / Pro 播放器） */
  children: ReactNode
  /** 底部时序波形区（ECharts / KineticVelocityChart） */
  waveform?: ReactNode
  /** 可选：视口上方 HUD / 工具条 */
  overlay?: ReactNode
  className?: string
  title?: string
  subtitle?: string
  /**
   * 来自 MetricCardList 的 Seek 请求。
   * VideoWorkspace 会尝试定位内部 <video>；若找不到则把请求转发给 onSeekRequest。
   */
  seekRequest?: VideoSeekRequest | null
  /**
   * 来自 KineticVelocityChart 的窗口内索引 Seek。
   * absolute_frame = actionRoiStart + xIndex（切勿直接用 xIndex 跳视频）
   */
  chartSeekRequest?: ChartSeekRequest | null
  /** 后端触球绝对帧（红色虚线锚点） */
  backendImpactFrameIdx?: number | null
  /**
   * Action ROI 窗口起点绝对帧（scoreDetail.action_roi.start）。
   * 优先用于波形 x_index → 视频映射；缺省回退 backendImpactFrameIdx - 30。
   */
  actionRoiStart?: number | null
  /** 视频帧率；currentTime = absolute_frame / fps（默认 30） */
  videoFps?: number
  /** 当内部无法直接 Seek 时（例如仅有 WebSocket 帧流），由父组件接管 */
  onSeekRequest?: (request: VideoSeekRequest) => void
  /** 当前定格提示（HUD） */
  seekHud?: MetricSeekEvent | null
  /**
   * 视频自然播放时，把窗口内游标索引回传给波形图。
   * x_index = floor(currentTime * fps) - actionRoiStart；越界时传 null（隐藏游标）。
   */
  onWaveformCursorXIndex?: (xIndex: number | null) => void
}

function clampTime(video: HTMLVideoElement, seconds: number): number {
  const max = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : seconds
  return Math.max(0, Math.min(max, seconds))
}

/**
 * V2.5 中栏 VideoWorkspace（44%）
 * 核心视频播放器 + 底部时序波形；支持指标卡片 / 波形图驱动的帧 Seek。
 */
export default function VideoWorkspace({
  children,
  waveform,
  overlay,
  className = '',
  title = 'Video Workspace',
  subtitle = '主监视器 · 动能链时序',
  seekRequest = null,
  chartSeekRequest = null,
  backendImpactFrameIdx = null,
  actionRoiStart = null,
  videoFps = 30,
  onSeekRequest,
  seekHud = null,
  onWaveformCursorXIndex,
}: VideoWorkspaceProps) {
  const stageRef = useRef<HTMLDivElement>(null)
  const fps = videoFps > 0 ? videoFps : 30

  /** absolute_frame 窗口起点：优先 action_roi.start，否则 t_impact - 30 */
  const resolveRoiStart = (): number | null => {
    if (typeof actionRoiStart === 'number' && Number.isFinite(actionRoiStart)) {
      return Math.max(0, Math.round(actionRoiStart))
    }
    if (typeof backendImpactFrameIdx === 'number' && Number.isFinite(backendImpactFrameIdx)) {
      return Math.max(0, Math.round(backendImpactFrameIdx) - IMPACT_WINDOW_HALF)
    }
    return null
  }

  const findVideo = (): HTMLVideoElement | null => {
    const stage = stageRef.current
    return (stage?.querySelector('video') as HTMLVideoElement | null) ?? null
  }

  const seekAbsoluteFrame = (absoluteFrame: number, label?: string) => {
    const video = findVideo()
    if (video && Number.isFinite(video.duration) && video.duration > 0) {
      video.pause()
      // 精确跳转：currentTime = absolute_frame / 30.0
      video.currentTime = clampTime(video, absoluteFrame / fps)
      return true
    }
    if (label != null || onSeekRequest) {
      onSeekRequest?.({
        frameIndex: Math.max(0, Math.round(absoluteFrame)),
        fps,
        label,
        token: Date.now(),
      })
    }
    return false
  }

  // 指标卡片 → 绝对帧 Seek
  useEffect(() => {
    if (!seekRequest) return

    const requestFps = seekRequest.fps && seekRequest.fps > 0 ? seekRequest.fps : fps
    const sampleCount =
      typeof seekRequest.sampleFrameCount === 'number' && seekRequest.sampleFrameCount > 1
        ? seekRequest.sampleFrameCount
        : null

    const video = findVideo()
    if (video && Number.isFinite(video.duration) && video.duration > 0) {
      video.pause()
      let targetSec: number
      if (sampleCount != null) {
        targetSec = (seekRequest.frameIndex / Math.max(1, sampleCount - 1)) * video.duration
      } else {
        targetSec = seekRequest.frameIndex / requestFps
      }
      video.currentTime = clampTime(video, targetSec)
      return
    }

    onSeekRequest?.(seekRequest)
  }, [seekRequest, onSeekRequest, fps])

  // 波形图 x_index → absolute_frame = action_roi.start + x_index（绝不用局部索引直接跳）
  useEffect(() => {
    if (!chartSeekRequest) return
    const roiStart = resolveRoiStart()
    if (roiStart == null) return
    const xIndex = Math.max(0, Math.round(chartSeekRequest.xIndex))
    const absoluteFrame = roiStart + xIndex
    seekAbsoluteFrame(Math.max(0, absoluteFrame), `波形 #${xIndex}`)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chartSeekRequest, actionRoiStart, backendImpactFrameIdx, fps])

  // 视频 timeupdate → 波形游标：x_index = floor(t*fps) - action_roi.start；越界隐藏
  useEffect(() => {
    if (!onWaveformCursorXIndex) return
    const video = findVideo()
    if (!video) return

    const onTimeUpdate = () => {
      const roiStart = resolveRoiStart()
      if (roiStart == null) {
        onWaveformCursorXIndex(null)
        return
      }
      const xIndex = Math.floor(video.currentTime * fps) - roiStart
      if (xIndex < 0 || xIndex > IMPACT_WINDOW_HALF * 2) {
        onWaveformCursorXIndex(null)
        return
      }
      onWaveformCursorXIndex(xIndex)
    }

    video.addEventListener('timeupdate', onTimeUpdate)
    return () => {
      video.removeEventListener('timeupdate', onTimeUpdate)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onWaveformCursorXIndex, actionRoiStart, backendImpactFrameIdx, fps, children])

  // 视频首次 loadeddata：强制 seek 到触球帧（红色虚线），避免停在第 0 帧助跑
  useEffect(() => {
    if (typeof backendImpactFrameIdx !== 'number' || !Number.isFinite(backendImpactFrameIdx)) {
      return
    }
    const video = findVideo()
    if (!video) return

    const seekToImpact = () => {
      video.pause()
      video.currentTime = clampTime(video, Math.round(backendImpactFrameIdx) / fps)
    }

    if (video.readyState >= 2) {
      seekToImpact()
      return
    }

    video.addEventListener('loadeddata', seekToImpact)
    return () => {
      video.removeEventListener('loadeddata', seekToImpact)
    }
  }, [backendImpactFrameIdx, fps, children])

  return (
    <section
      className={`workbench-col workbench-card overflow-hidden ${className}`.trim()}
      aria-label="视频工作区"
    >
      <div className="flex h-full min-h-0 flex-col">
        <header className="flex flex-shrink-0 items-center gap-2 border-b border-slate-700/80 px-3 py-2.5">
          <span className="inline-flex flex-shrink-0 text-[var(--GREEN_OPTIMAL)]">
            <Clapperboard className="h-4 w-4" />
          </span>
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-sm font-semibold text-slate-100">{title}</h2>
            <p className="truncate text-[10px] text-slate-400">{subtitle}</p>
          </div>
          {seekHud && (
            <span className="inline-flex flex-shrink-0 items-center gap-1 rounded-lg border border-[color-mix(in_srgb,var(--GREEN_OPTIMAL)_35%,transparent)] bg-[color-mix(in_srgb,var(--GREEN_OPTIMAL)_12%,transparent)] px-2 py-1 text-[10px] font-semibold text-[var(--GREEN_OPTIMAL)]">
              <Crosshair className="h-3 w-3" />
              {seekHud.label} · F#{seekHud.frameIndex}
            </span>
          )}
        </header>

        <div ref={stageRef} className="relative min-h-0 flex-[1.35] overflow-hidden bg-black/40">
          {children}
          {overlay}
        </div>

        {waveform != null && (
          <div className="min-h-0 flex-shrink-0 overflow-hidden border-t border-slate-700/80">
            {waveform}
          </div>
        )}
      </div>
    </section>
  )
}
