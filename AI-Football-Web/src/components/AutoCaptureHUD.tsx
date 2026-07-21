import { AnimatePresence, motion } from 'framer-motion'
import { Gamepad2, Loader2 } from 'lucide-react'
import type { AutoCaptureAttempt, AutoCaptureFsmState } from '../types'

export interface AutoCaptureHUDProps {
  /** 后端广播三态：IDLE / RECORDING / PROCESSING */
  fsmState: AutoCaptureFsmState
  /** 已确认归档的趟次总数（含进行中卡片时由父组件拼装） */
  attemptCount: number
  /** 最近一脚得分；尚无成绩时为 null */
  latestScore: number | null
  /** Attempt Chain Dock 数据（含「抓取中…」占位卡） */
  attempts: AutoCaptureAttempt[]
  /** 当前倒带选中的趟次编号；null 表示跟随最新 live 画面 */
  selectedAttemptNumber: number | null
  /** 是否正在分析（未开摄像头时 HUD 半透明待机） */
  isLive: boolean
  /** 点击历史 Attempt：中栏视口 + 右侧雷达倒带 */
  onSelectAttempt: (attempt: AutoCaptureAttempt) => void
  /** 人工介入防线：手动强制截取该脚 */
  onForceCapture: () => void
  /** 强制截取按钮冷却中（防连点） */
  forceCaptureBusy?: boolean
}

const FSM_PILL: Record<
  AutoCaptureFsmState,
  { label: string; className: string; dotClass: string; icon: 'idle' | 'rec' | 'proc' }
> = {
  IDLE: {
    label: 'AI 零感侦测监听中（无需按键）',
    className: 'border-emerald-400/35 bg-emerald-500/15 text-emerald-200',
    dotClass: 'bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.85)]',
    icon: 'idle',
  },
  RECORDING: {
    label: '锁定发力 · 自动捕获中…',
    className: 'border-rose-400/40 bg-rose-500/20 text-rose-100',
    dotClass: 'bg-rose-500 shadow-[0_0_12px_rgba(244,63,94,0.9)] ac-hud-breath',
    icon: 'rec',
  },
  PROCESSING: {
    label: '静默切片与五维评分中…',
    className: 'border-amber-400/40 bg-amber-500/15 text-amber-100',
    dotClass: 'bg-amber-400 shadow-[0_0_10px_rgba(251,191,36,0.85)]',
    icon: 'proc',
  },
}

/**
 * 【V3·模块三】零感捕获 HUD 抬头显示器
 * 悬浮于 RealtimeWorkspace 中心大屏视口上方：顶部状态脉冲灯 +
 * 底部 Attempt Chain Dock + 右下角人工强制截取备用按钮。
 */
export default function AutoCaptureHUD({
  fsmState,
  attemptCount,
  latestScore,
  attempts,
  selectedAttemptNumber,
  isLive,
  onSelectAttempt,
  onForceCapture,
  forceCaptureBusy = false,
}: AutoCaptureHUDProps) {
  const pill = FSM_PILL[fsmState] ?? FSM_PILL.IDLE

  return (
    <div
      className={`pointer-events-none absolute inset-0 z-20 ${isLive ? 'opacity-100' : 'opacity-70'}`}
      aria-label="零感自动捕获抬头显示器"
    >
      {/* —— 顶部状态脉冲胶囊 —— */}
      <div className="pointer-events-none absolute left-1/2 top-4 z-30 flex -translate-x-1/2 flex-col items-center gap-1.5">
        <motion.div
          layout
          className={`ac-hud-pill flex items-center gap-2.5 rounded-full border px-4 py-2 backdrop-blur-2xl ${pill.className}`}
        >
          {pill.icon === 'proc' ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin text-amber-300" />
          ) : (
            <span className={`inline-block h-2.5 w-2.5 rounded-full ${pill.dotClass}`} />
          )}
          <span className="text-[11px] font-semibold tracking-wide">{pill.label}</span>
          {attemptCount > 0 && (
            <span className="rounded-full bg-black/30 px-2 py-0.5 text-[10px] tabular-nums text-white/70">
              #{attemptCount}
              {latestScore != null ? ` · ${latestScore}分` : ''}
            </span>
          )}
        </motion.div>
      </div>

      {/* —— 底部 Attempt Chain Dock —— */}
      <div className="pointer-events-auto absolute bottom-3 left-1/2 z-30 w-[92%] max-w-3xl -translate-x-1/2">
        <div className="ac-hud-dock overflow-x-auto rounded-2xl border border-white/10 bg-black/55 px-3 py-2.5 backdrop-blur-2xl">
          <div className="mb-1.5 flex items-center justify-between px-1">
            <p className="text-[10px] font-medium uppercase tracking-[0.14em] text-white/35">
              Attempt Chain
            </p>
            <p className="text-[10px] text-white/30">点击历史趟次倒带复盘</p>
          </div>
          <div className="flex items-center gap-2">
            <AnimatePresence initial={false} mode="popLayout">
              {attempts.length === 0 ? (
                <motion.span
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="px-2 py-1 text-[11px] text-white/30"
                >
                  等待学员自然发力 · 系统将自动挂载趟次卡片
                </motion.span>
              ) : (
                attempts.map((attempt, index) => {
                  const selected = selectedAttemptNumber === attempt.attemptNumber
                  const isLiveCard =
                    attempt.status === 'capturing' || attempt.status === 'processing'
                  return (
                    <div key={attempt.id} className="flex shrink-0 items-center gap-2">
                      {index > 0 && (
                        <span className="text-[10px] text-white/25" aria-hidden>
                          ➜
                        </span>
                      )}
                      <motion.button
                        type="button"
                        layout
                        initial={{ opacity: 0, scale: 0.92, y: 6 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.9 }}
                        transition={{ type: 'spring', stiffness: 380, damping: 28 }}
                        disabled={isLiveCard}
                        onClick={() => onSelectAttempt(attempt)}
                        className={`group flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-left transition ${
                          selected
                            ? 'border-sky-400/50 bg-sky-500/20 text-sky-100'
                            : isLiveCard
                              ? 'border-rose-400/35 bg-rose-500/15 text-rose-100'
                              : 'border-white/10 bg-white/5 text-white/80 hover:border-white/25 hover:bg-white/10'
                        }`}
                      >
                        <span className="text-sm leading-none" aria-hidden>
                          {isLiveCard ? '🔴' : '⚽'}
                        </span>
                        <span className="text-[11px] font-semibold tabular-nums">
                          #{attempt.attemptNumber}
                          {attempt.score != null
                            ? `: ${attempt.score}分`
                            : isLiveCard
                              ? ': 抓取中…'
                              : ': --'}
                        </span>
                      </motion.button>
                    </div>
                  )
                })
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>

      {/* —— 右下角人工介入防线 —— */}
      <div className="pointer-events-auto absolute bottom-3 right-3 z-30">
        <button
          type="button"
          disabled={!isLive || forceCaptureBusy || fsmState === 'PROCESSING'}
          onClick={onForceCapture}
          title="光线过暗等极端特例下，手动强制截取当前发力脚"
          className="ac-hud-override flex items-center gap-1.5 rounded-full border border-white/15 bg-black/55 px-3 py-2 text-[10px] font-medium text-white/70 shadow-lg backdrop-blur-xl transition hover:border-white/30 hover:bg-black/70 hover:text-white disabled:cursor-not-allowed disabled:opacity-35"
        >
          <Gamepad2 className="h-3.5 w-3.5" />
          <span>手动强制截取该脚</span>
        </button>
      </div>
    </div>
  )
}
