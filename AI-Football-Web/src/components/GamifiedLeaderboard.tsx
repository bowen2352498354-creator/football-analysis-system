import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Loader2, Sparkles, Trophy } from 'lucide-react'
import type { WeeklyAchievementBadge, WeeklyAchievementsResponse } from '../types'

/* ============================================================================
 * 【SDT 自我决定理论】多维度成就印章榜（拒绝总分排名）
 * 后端 GET /api/achievements/weekly → 三张 Glassmorphism Bento 卡片
 * 🛡️ 钢铁锁踝王 / 🌳 最稳底盘奖 / 🚀 最快进步奖
 * ========================================================================== */

const API_BASE_URL = 'http://localhost:8000'

const BADGE_ACCENT: Record<
  string,
  { glow: string; ring: string; chip: string; value: string }
> = {
  iron_ankle: {
    glow: 'shadow-[0_0_40px_rgba(56,189,248,0.22)]',
    ring: 'ring-sky-400/25',
    chip: 'from-sky-400/25 to-cyan-500/10 text-sky-200',
    value: 'text-sky-300',
  },
  stable_chassis: {
    glow: 'shadow-[0_0_40px_rgba(52,211,153,0.22)]',
    ring: 'ring-emerald-400/25',
    chip: 'from-emerald-400/25 to-teal-500/10 text-emerald-200',
    value: 'text-emerald-300',
  },
  fastest_progress: {
    glow: 'shadow-[0_0_40px_rgba(251,146,60,0.22)]',
    ring: 'ring-orange-400/25',
    chip: 'from-orange-400/25 to-amber-500/10 text-orange-200',
    value: 'text-orange-300',
  },
}

const FALLBACK_BADGES: WeeklyAchievementBadge[] = [
  {
    id: 'iron_ankle',
    title: '钢铁锁踝王',
    emoji: '🛡️',
    anonymousId: null,
    value: null,
    valueLabel: '脚踝刚性方差',
    unit: 'σ²',
    praise: '踝关节稳如泰山，力量毫无流失！',
    hasWinner: false,
  },
  {
    id: 'stable_chassis',
    title: '最稳底盘奖',
    emoji: '🌳',
    anonymousId: null,
    value: null,
    valueLabel: '支撑脚横纵偏差',
    unit: 'cm',
    praise: '支撑脚扎根大地，底盘稳如磐石！',
    hasWinner: false,
  },
  {
    id: 'fastest_progress',
    title: '最快进步奖',
    emoji: '🚀',
    anonymousId: null,
    value: null,
    valueLabel: '五维均分周环比',
    unit: 'Δ',
    praise: '本周飞跃成长，高反应者实至名归！',
    hasWinner: false,
  },
]

function formatValue(badge: WeeklyAchievementBadge): string {
  if (badge.value === null || badge.value === undefined) return '—'
  const n = Number(badge.value)
  if (!Number.isFinite(n)) return '—'
  if (badge.id === 'fastest_progress') {
    return `+${n.toFixed(1)}`
  }
  return n.toFixed(2)
}

export interface GamifiedLeaderboardProps {
  /** 与教练端筛选器对齐；'all' / 空表示全库 */
  school?: string
  classGroup?: string
  scopeLabel?: string
}

export default function GamifiedLeaderboard({
  school = 'all',
  classGroup = 'all',
  scopeLabel = '全班',
}: GamifiedLeaderboardProps) {
  const [badges, setBadges] = useState<WeeklyAchievementBadge[]>(FALLBACK_BADGES)
  const [weekLabel, setWeekLabel] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const params = new URLSearchParams()
    if (school && school !== 'all') params.set('school', school)
    if (classGroup && classGroup !== 'all') params.set('classGroup', classGroup)
    const qs = params.toString()

    async function load() {
      setLoading(true)
      setError(null)
      try {
        const res = await fetch(
          `${API_BASE_URL}/api/achievements/weekly${qs ? `?${qs}` : ''}`,
        )
        const data = (await res.json()) as WeeklyAchievementsResponse
        if (!res.ok || !data.success) {
          throw new Error(data.message || `接口返回 ${res.status}`)
        }
        if (cancelled) return
        const list = data.badges?.length ? data.badges : data.achievements || []
        setBadges(list.length ? list : FALLBACK_BADGES)
        if (data.weekStart && data.weekEnd) {
          setWeekLabel(`${data.weekStart} ~ ${data.weekEnd}`)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : '加载成就失败')
          setBadges(FALLBACK_BADGES)
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void load()
    return () => {
      cancelled = true
    }
  }, [school, classGroup])

  return (
    <section className="gamified-leaderboard space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h3 className="flex items-center gap-2 text-base font-semibold text-white/90">
            <Trophy className="h-4.5 w-4.5 text-amber-300" />
            成就印章 · SDT 内在动机激励
          </h3>
          <p className="mt-1 text-xs text-white/35">
            {scopeLabel}
            {weekLabel ? ` · 本周 ${weekLabel}` : ''} · 多维度独立王者，拒绝总分排名
          </p>
        </div>
        <span className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[10px] text-white/40">
          <Sparkles className="h-3 w-3 text-amber-300/80" />
          Self-Determination Theory
        </span>
      </div>

      {loading ? (
        <div className="flex min-h-[160px] items-center justify-center gap-2 rounded-3xl border border-white/10 bg-white/5 text-white/40 backdrop-blur-xl">
          <Loader2 className="h-5 w-5 animate-spin text-amber-300" />
          <span className="text-sm">正在点亮本周成就印章……</span>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          {badges.map((badge, index) => {
            const accent = BADGE_ACCENT[badge.id] ?? BADGE_ACCENT.iron_ankle
            const lit = Boolean(badge.hasWinner && badge.anonymousId)
            return (
              <motion.article
                key={badge.id}
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.08, type: 'spring', stiffness: 260, damping: 24 }}
                className={`relative flex flex-col overflow-hidden rounded-[28px] border border-white/12 bg-gradient-to-br from-white/[0.12] via-white/[0.04] to-black/40 p-5 backdrop-blur-2xl ring-1 ${accent.ring} ${lit ? accent.glow : ''}`}
              >
                <div
                  className="pointer-events-none absolute -right-8 -top-10 h-36 w-36 rounded-full bg-white/5 blur-2xl"
                  aria-hidden
                />
                <div
                  className={`mb-3 inline-flex w-fit items-center gap-1.5 rounded-full bg-gradient-to-r px-2.5 py-1 text-[11px] font-semibold ${accent.chip}`}
                >
                  {badge.title}
                </div>

                <div className="mb-3 flex items-start justify-between gap-3">
                  <span
                    className="select-none text-5xl leading-none drop-shadow-[0_8px_24px_rgba(0,0,0,0.45)]"
                    aria-hidden
                  >
                    {badge.emoji}
                  </span>
                  <div className="min-w-0 text-right">
                    <p className="text-[10px] uppercase tracking-wider text-white/30">
                      {badge.valueLabel}
                    </p>
                    <p className={`text-xl font-black tabular-nums ${accent.value}`}>
                      {formatValue(badge)}
                      {lit && badge.unit ? (
                        <span className="ml-1 text-[11px] font-medium text-white/35">
                          {badge.unit}
                        </span>
                      ) : null}
                    </p>
                  </div>
                </div>

                {lit ? (
                  <>
                    <p className="text-2xl font-black tracking-wide text-white/95">
                      {badge.anonymousId}
                    </p>
                    <p className="mt-1 text-[11px] text-white/35">匿名学员编号 · 荣誉独占本周</p>
                    <p className="mt-4 text-sm leading-relaxed text-white/70">{badge.praise}</p>
                  </>
                ) : (
                  <div className="mt-2 flex flex-1 flex-col justify-center gap-2 py-4">
                    <p className="text-lg font-semibold text-white/40">虚位以待</p>
                    <p className="text-xs leading-relaxed text-white/30">
                      {error
                        ? `暂无法拉取成就（${error}）`
                        : '本周尚无达标学员，继续稳扎稳打探索动作奥秘！'}
                    </p>
                  </div>
                )}
              </motion.article>
            )
          })}
        </div>
      )}
    </section>
  )
}
