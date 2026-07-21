import { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import {
  Tablet,
  CheckCircle2,
  Loader2,
  X,
  Users,
} from 'lucide-react'
import BiomechanicalRadar from './BiomechanicalRadar'
import type { GlobalTrainingRecord, PeerReviewData, PeerReviewTagId } from '../types'

/* ============================================================================
 * 【V3·模块五】B组延时反馈 · iPad 平板结对互评触控工作台
 * 左栏：慢放关键帧 + 五维雷达 · 右栏：大号同伴裁判印章 + 综合分滑块
 * ========================================================================== */

const API_BASE_URL = 'http://localhost:8000'

const SUPPORT_TAGS: { id: PeerReviewTagId; label: string; tone: 'good' | 'bad' | 'warn' }[] = [
  { id: 'knee_spring', label: '👍 膝盖弯得像弹簧！', tone: 'good' },
  { id: 'knee_straight', label: '❌ 膝盖太直了', tone: 'bad' },
  { id: 'stand_far', label: '⚠️ 离球站太远', tone: 'warn' },
]

const ANKLE_TAGS: { id: PeerReviewTagId; label: string; tone: 'good' | 'bad' | 'warn' }[] = [
  { id: 'ankle_iron', label: '🔒 脚踝踩得很硬像铁板！', tone: 'good' },
  { id: 'ankle_soft', label: '❌ 触球时脚踝软了', tone: 'bad' },
  { id: 'whip_fast', label: '🚀 鞭打速度很快！', tone: 'good' },
]

const TONE_CLASS: Record<'good' | 'bad' | 'warn', string> = {
  good: 'border-emerald-400/40 bg-emerald-500/15 text-emerald-100 data-[on=true]:bg-emerald-400 data-[on=true]:text-black',
  bad: 'border-rose-400/40 bg-rose-500/15 text-rose-100 data-[on=true]:bg-rose-400 data-[on=true]:text-black',
  warn: 'border-amber-400/40 bg-amber-500/15 text-amber-100 data-[on=true]:bg-amber-400 data-[on=true]:text-black',
}

export interface TabletPeerReviewProps {
  records: GlobalTrainingRecord[]
  onClose: () => void
  onSubmitted?: (recordId: string, data: PeerReviewData) => void
  onToast?: (message: string, success: boolean) => void
}

export default function TabletPeerReview({
  records,
  onClose,
  onSubmitted,
  onToast,
}: TabletPeerReviewProps) {
  const students = useMemo(() => {
    const map = new Map<string, { key: string; studentId: string; school: string; classGroup: string; latest: GlobalTrainingRecord }>()
    const sorted = [...records].sort((a, b) => (a.timestamp || '').localeCompare(b.timestamp || ''))
    for (const record of sorted) {
      const key = `${record.school}__${record.classGroup}__${record.studentId}`
      map.set(key, {
        key,
        studentId: record.studentId,
        school: record.school,
        classGroup: record.classGroup,
        latest: record,
      })
    }
    return Array.from(map.values())
  }, [records])

  const [selectedKey, setSelectedKey] = useState<string | null>(students[0]?.key ?? null)
  const selected = students.find((s) => s.key === selectedKey) ?? students[0] ?? null
  const attempt = selected?.latest ?? null

  const [selectedTags, setSelectedTags] = useState<PeerReviewTagId[]>([])
  const [peerScore, setPeerScore] = useState(72)
  const [reviewerId, setReviewerId] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  function toggleTag(id: PeerReviewTagId) {
    setSelectedTags((prev) => (prev.includes(id) ? prev.filter((t) => t !== id) : [...prev, id]))
  }

  async function handleSubmit() {
    if (!attempt?.id) {
      onToast?.('请先选择一位学员的 Attempt', false)
      return
    }
    if (selectedTags.length === 0) {
      onToast?.('请至少点选一枚同伴互评印章', false)
      return
    }
    setIsSubmitting(true)
    const payload: PeerReviewData = {
      tags: selectedTags,
      peerScore,
      reviewerId: reviewerId.trim() || '同伴匿名',
      submittedAt: new Date().toISOString().replace('T', ' ').slice(0, 19),
    }
    try {
      const response = await fetch(`${API_BASE_URL}/api/save_peer_review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          attemptId: attempt.id,
          peerReviewData: payload,
        }),
      })
      const data = (await response.json()) as { success: boolean; message?: string }
      if (!data.success) throw new Error(data.message || '提交失败')
      onSubmitted?.(attempt.id, payload)
      onToast?.(data.message || '同伴互评裁判单已存档', true)
      setSelectedTags([])
    } catch (error) {
      onToast?.(error instanceof Error ? error.message : '提交互评失败', false)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="tablet-peer-review fixed inset-0 z-[70] flex flex-col bg-black/95 backdrop-blur-2xl">
      <header className="flex flex-shrink-0 items-center justify-between gap-3 border-b border-white/10 px-5 py-4">
        <div className="flex items-center gap-3">
          <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-sky-500/20 ring-1 ring-sky-400/35">
            <Tablet className="h-5 w-5 text-sky-300" />
          </span>
          <div>
            <h2 className="text-lg font-semibold text-white/95">📱 B组平板互评模式</h2>
            <p className="text-xs text-white/40">结对触控裁判台 · 大字印章 · 适合 iPad 横向握持</p>
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="flex h-11 w-11 items-center justify-center rounded-2xl bg-white/10 text-white/70 transition hover:bg-white/20 active:scale-95"
          title="退出平板互评"
        >
          <X className="h-5 w-5" />
        </button>
      </header>

      <div className="flex min-h-0 flex-1 flex-col gap-4 p-4 lg:flex-row">
        {/* 左栏 50%：视频 + 雷达 */}
        <section className="flex min-h-0 w-full flex-col gap-4 lg:w-1/2">
          <label className="flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
            <Users className="h-4 w-4 text-emerald-300" />
            <span className="text-sm text-white/50">选定学员</span>
            <select
              value={selected?.key ?? ''}
              onChange={(e) => {
                setSelectedKey(e.target.value)
                setSelectedTags([])
              }}
              className="ml-2 flex-1 rounded-xl bg-black/40 px-3 py-2 text-base font-semibold text-white outline-none [&>option]:bg-zinc-900"
            >
              {students.length === 0 && <option value="">暂无归档学员</option>}
              {students.map((s) => (
                <option key={s.key} value={s.key}>
                  {s.studentId} · {s.classGroup}
                </option>
              ))}
            </select>
          </label>

          <div className="relative min-h-[220px] flex-1 overflow-hidden rounded-3xl border border-amber-400/25 bg-black shadow-inner">
            {attempt?.impactFrameBase64 ? (
              <img
                src={attempt.impactFrameBase64}
                alt="学员发力关键帧"
                className="h-full w-full object-contain"
              />
            ) : (
              <div className="flex h-full min-h-[220px] items-center justify-center text-sm text-white/35">
                该学员暂无高清关键帧（完成分析归档后即可互评）
              </div>
            )}
            <span className="absolute left-3 top-3 rounded-full bg-black/60 px-3 py-1 text-xs text-amber-200 backdrop-blur">
              高清慢放关键帧 · {attempt?.score ?? '—'} 分
            </span>
          </div>

          <div className="rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-xl">
            <BiomechanicalRadar
              scores={attempt?.quantified5dScores}
              primaryLabel={attempt ? `学号 ${attempt.studentId}` : '待选学员'}
            />
          </div>
        </section>

        {/* 右栏 50%：伙伴裁判卡 */}
        <section className="flex w-full flex-col gap-4 rounded-3xl border border-sky-400/25 bg-gradient-to-br from-sky-500/10 via-white/5 to-transparent p-5 backdrop-blur-xl lg:w-1/2">
          <h3 className="text-xl font-bold tracking-tight text-white/95">伙伴裁判卡</h3>
          <p className="text-sm text-white/40">用大拇指点选印章 · 可多选 · 再拖动综合得分</p>

          <div className="space-y-3">
            <p className="text-sm font-semibold text-white/70">支撑与膝角</p>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              {SUPPORT_TAGS.map((tag) => {
                const on = selectedTags.includes(tag.id)
                return (
                  <button
                    key={tag.id}
                    type="button"
                    data-on={on}
                    onClick={() => toggleTag(tag.id)}
                    className={`peer-tag-card min-h-[72px] rounded-2xl border px-3 py-4 text-center text-[15px] font-bold leading-snug transition active:scale-[0.97] ${TONE_CLASS[tag.tone]}`}
                  >
                    {tag.label}
                  </button>
                )
              })}
            </div>
          </div>

          <div className="space-y-3">
            <p className="text-sm font-semibold text-white/70">脚踝与发力</p>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              {ANKLE_TAGS.map((tag) => {
                const on = selectedTags.includes(tag.id)
                return (
                  <button
                    key={tag.id}
                    type="button"
                    data-on={on}
                    onClick={() => toggleTag(tag.id)}
                    className={`peer-tag-card min-h-[72px] rounded-2xl border px-3 py-4 text-center text-[15px] font-bold leading-snug transition active:scale-[0.97] ${TONE_CLASS[tag.tone]}`}
                  >
                    {tag.label}
                  </button>
                )
              })}
            </div>
          </div>

          <div className="mt-2 rounded-2xl bg-black/30 p-4">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-sm font-semibold text-white/70">同伴互评综合得分</span>
              <motion.span
                key={peerScore}
                initial={{ scale: 1.15 }}
                animate={{ scale: 1 }}
                className="text-3xl font-black tabular-nums text-amber-300"
              >
                {peerScore}
                <span className="ml-1 text-sm font-medium text-white/40">分</span>
              </motion.span>
            </div>
            <input
              type="range"
              min={40}
              max={100}
              step={1}
              value={peerScore}
              onChange={(e) => setPeerScore(Number(e.target.value))}
              className="peer-score-slider w-full"
            />
            <div className="mt-1 flex justify-between text-[11px] text-white/30">
              <span>40</span>
              <span>100</span>
            </div>
          </div>

          <label className="flex items-center gap-2 rounded-2xl bg-black/25 px-4 py-3">
            <span className="whitespace-nowrap text-sm text-white/45">裁判同伴学号</span>
            <input
              value={reviewerId}
              onChange={(e) => setReviewerId(e.target.value)}
              placeholder="选填"
              className="flex-1 bg-transparent text-base text-white outline-none placeholder:text-white/25"
            />
          </label>

          <button
            type="button"
            onClick={() => void handleSubmit()}
            disabled={isSubmitting || !attempt}
            className="mt-auto flex min-h-[56px] items-center justify-center gap-2 rounded-2xl bg-emerald-400 px-6 text-lg font-bold text-black transition hover:bg-emerald-300 active:scale-[0.98] disabled:cursor-not-allowed disabled:bg-white/15 disabled:text-white/35"
          >
            {isSubmitting ? <Loader2 className="h-5 w-5 animate-spin" /> : <CheckCircle2 className="h-5 w-5" />}
            ✅ 提交互评裁判单
          </button>
        </section>
      </div>
    </div>
  )
}
