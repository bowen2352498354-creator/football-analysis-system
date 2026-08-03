import { Lock, Images } from 'lucide-react'
import type { ComparisonFrames } from '../types'

export interface ActionMirrorCompareProps {
  /** 后端 / 本地回退的双关键帧；null 表示尚未解锁 */
  comparisonFrames: ComparisonFrames | null | undefined
  /** 本节有效尝试数；< 2 时展示解锁空态 */
  attemptCount?: number
  className?: string
}

function formatScore(score: number | null | undefined): string {
  if (typeof score !== 'number' || Number.isNaN(score)) return '—'
  return Number.isInteger(score) ? String(score) : score.toFixed(1)
}

function FrameImage({
  src,
  alt,
  glowClass,
}: {
  src?: string | null
  alt: string
  glowClass: string
}) {
  if (!src) {
    return (
      <div className="flex aspect-[4/3] w-full flex-col items-center justify-center gap-2 bg-black/35 text-white/30">
        <Images className="h-9 w-9 opacity-50" />
        <p className="px-4 text-center text-[11px]">暂无触球瞬间关键帧</p>
      </div>
    )
  }
  return (
    <div className={`overflow-hidden rounded-2xl ${glowClass}`}>
      <img src={src} alt={alt} className="aspect-[4/3] w-full object-cover" />
    </div>
  )
}

/**
 * B 组复盘：「动作定型镜像对比」Twin-Card。
 * 左右绝对对称：最佳（绿/金） vs 待改进（橙），建立静态视觉参照系。
 */
export default function ActionMirrorCompare({
  comparisonFrames,
  attemptCount = 0,
  className = '',
}: ActionMirrorCompareProps) {
  const unlocked = Boolean(
    comparisonFrames?.best &&
      comparisonFrames?.improve &&
      typeof comparisonFrames.best.score === 'number' &&
      typeof comparisonFrames.improve.score === 'number',
  )

  return (
    <section
      className={`w-full rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-xl ${className}`}
    >
      <h4 className="mb-4 text-sm font-semibold text-white/85">📸 动作定型镜像对比</h4>

      {!unlocked ? (
        <div className="flex min-h-[140px] flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-white/15 bg-black/25 px-6 py-8 text-center">
          <Lock className="h-6 w-6 text-white/25" />
          <p className="text-sm text-white/55">数据积攒中，下一次尝试后解锁对比功能</p>
          <p className="text-[11px] text-white/30">
            本节已记录有效尝试 {attemptCount} 次 · 需至少 2 次才能镜像对照
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 md:gap-5">
          {/* 左：最佳表现 */}
          <article className="flex flex-col gap-3 rounded-2xl border border-emerald-400/25 bg-gradient-to-b from-emerald-500/10 via-amber-400/5 to-transparent p-3.5">
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center rounded-full border border-amber-300/35 bg-emerald-500/15 px-2.5 py-1 text-[11px] font-semibold text-amber-100">
                🌟 最佳表现 (Best)
              </span>
            </div>
            <FrameImage
              src={comparisonFrames!.best.image_url}
              alt="最佳表现触球关键帧"
              glowClass="border border-emerald-400/45 shadow-[0_0_22px_rgba(52,211,153,0.35)] ring-1 ring-amber-300/20"
            />
            <p className="text-[12px] leading-relaxed text-emerald-50/90">
              得分: {formatScore(comparisonFrames!.best.score)} - 这是你今天表现最完美的一次，记住这个身体感觉！
            </p>
          </article>

          {/* 右：待改进表现 */}
          <article className="flex flex-col gap-3 rounded-2xl border border-orange-400/30 bg-gradient-to-b from-orange-500/10 via-transparent to-transparent p-3.5">
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center rounded-full border border-orange-300/40 bg-orange-500/15 px-2.5 py-1 text-[11px] font-semibold text-orange-100">
                💪 待改进表现 (Needs Improvement)
              </span>
            </div>
            <FrameImage
              src={comparisonFrames!.improve.image_url}
              alt="待改进触球关键帧"
              glowClass="border border-orange-400/40 shadow-[0_0_18px_rgba(251,146,60,0.22)]"
            />
            <p className="text-[12px] leading-relaxed text-orange-50/90">
              得分: {formatScore(comparisonFrames!.improve.score)} - 主要痛点：
              {comparisonFrames!.improve.main_error?.trim()
                ? comparisonFrames!.improve.main_error
                : '继续观察动作细节'}
            </p>
          </article>
        </div>
      )}
    </section>
  )
}
