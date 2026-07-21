import { useEffect, useRef, type ReactNode } from 'react'
import { Clapperboard, Crosshair } from 'lucide-react'
import type { MetricSeekEvent } from '../types'

export interface VideoSeekRequest {
  /** 物理极值帧索引（0-based） */
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

export interface VideoWorkspaceProps {
  /** 核心视频视口（实时帧 / Pro 播放器） */
  children: ReactNode
  /** 底部时序波形区（Recharts / SVG 动能链监控） */
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
  /** 当内部无法直接 Seek 时（例如仅有 WebSocket 帧流），由父组件接管 */
  onSeekRequest?: (request: VideoSeekRequest) => void
  /** 当前定格提示（HUD） */
  seekHud?: MetricSeekEvent | null
}

/**
 * V2.5 中栏 VideoWorkspace（44%）
 * 核心视频播放器 + 底部时序波形；支持指标卡片驱动的极值帧 Seek。
 */
export default function VideoWorkspace({
  children,
  waveform,
  overlay,
  className = '',
  title = 'Video Workspace',
  subtitle = '主监视器 · 动能链时序',
  seekRequest = null,
  onSeekRequest,
  seekHud = null,
}: VideoWorkspaceProps) {
  const stageRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!seekRequest) return

    const fps = seekRequest.fps && seekRequest.fps > 0 ? seekRequest.fps : 30
    const sampleCount =
      typeof seekRequest.sampleFrameCount === 'number' && seekRequest.sampleFrameCount > 1
        ? seekRequest.sampleFrameCount
        : null

    const stage = stageRef.current
    const video = stage?.querySelector('video') as HTMLVideoElement | null

    if (video && Number.isFinite(video.duration) && video.duration > 0) {
      video.pause()
      let targetSec: number
      if (sampleCount != null) {
        targetSec = (seekRequest.frameIndex / Math.max(1, sampleCount - 1)) * video.duration
      } else {
        targetSec = seekRequest.frameIndex / fps
      }
      video.currentTime = Math.max(0, Math.min(video.duration, targetSec))
      return
    }

    // 无本地 video 元素（实时 WS 帧流等）：交给父组件
    onSeekRequest?.(seekRequest)
  }, [seekRequest, onSeekRequest])

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
