import { useCallback, useEffect, useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  CalendarDays,
  Filter,
  Loader2,
  PencilLine,
  Search,
  ShieldAlert,
  Trash2,
  Users,
  X,
} from 'lucide-react'
import type { CoachCalibrateMetricKey, CoachSanitizerRecord } from '../types'

const API_BASE_URL = 'http://localhost:8000'

interface CoachDataSanitizerGridProps {
  /** 软删除成功后回调，便于父级同步刷新 KPI / 曲线 */
  onRecordSoftDeleted?: (recordId: string) => void
  /** 人工标定成功后回调 */
  onRecordCalibrated?: (recordId: string) => void
  showToast?: (message: string, success: boolean) => void
  /** 仅展示该被试的历史尝试（右侧清道夫面板） */
  studentId?: string | null
  /** 紧凑三列：时间 / 总分 / 删除 */
  compact?: boolean
  /** 父级数据世代号：软删除或刷新后递增，强制重新拉取 */
  refreshKey?: number
  className?: string
}

type GroupFilter = 'all' | 'realtime' | 'delayed'

const CALIBRATE_OPTIONS: Array<{ key: CoachCalibrateMetricKey; label: string; unit: string }> = [
  { key: 'distance_cm', label: '支撑脚横距', unit: 'cm' },
  { key: 'max_folding_angle', label: '后摆折叠角', unit: '°' },
  { key: 'ankle_rigidity', label: '踝刚度方差', unit: 'var' },
]

function groupLabel(record: CoachSanitizerRecord): string {
  const type = String(record.type || '').toLowerCase()
  if (type === 'realtime' || record.groupTypeCode === 1) return '实时 A 组'
  if (type === 'delayed' || record.groupTypeCode === 2) return '延时 B 组'
  return record.classGroup || '未分组'
}

function currentMetricValue(
  record: CoachSanitizerRecord,
  key: CoachCalibrateMetricKey,
): number | null {
  if (key === 'distance_cm') {
    return typeof record.supportFootDistance === 'number' ? record.supportFootDistance : null
  }
  if (key === 'max_folding_angle') {
    return typeof record.max_folding_angle === 'number' ? record.max_folding_angle : null
  }
  return typeof record.ankle_rigidity === 'number' ? record.ankle_rigidity : null
}

export default function CoachDataSanitizerGrid({
  onRecordSoftDeleted,
  onRecordCalibrated,
  showToast,
  studentId = null,
  compact = false,
  refreshKey = 0,
  className = '',
}: CoachDataSanitizerGridProps) {
  const [records, setRecords] = useState<CoachSanitizerRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [calibrating, setCalibrating] = useState(false)
  const [fadingIds, setFadingIds] = useState<Set<string>>(new Set())
  const [confirmTarget, setConfirmTarget] = useState<CoachSanitizerRecord | null>(null)
  const [calibrateTarget, setCalibrateTarget] = useState<CoachSanitizerRecord | null>(null)
  const [calibrateMetric, setCalibrateMetric] = useState<CoachCalibrateMetricKey>('distance_cm')
  const [calibrateValue, setCalibrateValue] = useState('')
  const [calibrateNote, setCalibrateNote] = useState('')

  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [studentQuery, setStudentQuery] = useState('')
  const [groupFilter, setGroupFilter] = useState<GroupFilter>('all')

  const scopedStudentId = studentId?.trim() || ''

  const fetchRecords = useCallback(async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams()
      if (dateFrom) params.set('date_from', dateFrom)
      if (dateTo) params.set('date_to', dateTo)
      const sid = scopedStudentId || studentQuery.trim()
      if (sid) params.set('student_id', sid)
      if (groupFilter !== 'all') params.set('group', groupFilter)

      const response = await fetch(`${API_BASE_URL}/api/coach/records?${params.toString()}`)
      if (!response.ok) throw new Error(`状态码 ${response.status}`)
      const data = (await response.json()) as {
        success: boolean
        records?: CoachSanitizerRecord[]
        message?: string
      }
      if (!data.success) throw new Error(data.message || '拉取失败')
      setRecords(Array.isArray(data.records) ? data.records : [])
    } catch (error) {
      setRecords([])
      showToast?.(
        `⚠️ 数据清道夫列表加载失败：${error instanceof Error ? error.message : '请检查后端'}`,
        false,
      )
    } finally {
      setLoading(false)
    }
  }, [dateFrom, dateTo, studentQuery, groupFilter, scopedStudentId, showToast])

  useEffect(() => {
    void fetchRecords()
  }, [fetchRecords, refreshKey])

  const visibleRecords = useMemo(
    () => records.filter((r) => !fadingIds.has(r.id)),
    [records, fadingIds],
  )

  function openCalibrate(record: CoachSanitizerRecord) {
    setCalibrateTarget(record)
    setCalibrateMetric('distance_cm')
    const current = currentMetricValue(record, 'distance_cm')
    setCalibrateValue(current != null ? String(current) : '')
    setCalibrateNote('')
  }

  async function confirmCalibrate() {
    if (!calibrateTarget || calibrating) return
    const parsed = Number(calibrateValue)
    if (!Number.isFinite(parsed)) {
      showToast?.('请输入有效数值', false)
      return
    }
    setCalibrating(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/coach/calibrate_metric`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: calibrateTarget.id,
          metric_key: calibrateMetric,
          value: parsed,
          note: calibrateNote.trim() || undefined,
          coach_id: 'coach',
        }),
      })
      const data = (await response.json()) as { success: boolean; message?: string }
      if (!response.ok || !data.success) {
        throw new Error(data.message || `状态码 ${response.status}`)
      }
      showToast?.(data.message || '已写入 calibrated', true)
      onRecordCalibrated?.(calibrateTarget.id)
      setCalibrateTarget(null)
      await fetchRecords()
    } catch (error) {
      showToast?.(
        `⚠️ 标定失败：${error instanceof Error ? error.message : '未知错误'}`,
        false,
      )
    } finally {
      setCalibrating(false)
    }
  }

  async function confirmSoftDelete() {
    if (!confirmTarget || deletingId) return
    const targetId = confirmTarget.id
    setDeletingId(targetId)
    try {
      const response = await fetch(`${API_BASE_URL}/api/coach/delete_record`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: targetId }),
      })
      const data = (await response.json()) as { success: boolean; message?: string }
      if (!response.ok || !data.success) {
        throw new Error(data.message || `状态码 ${response.status}`)
      }

      setConfirmTarget(null)
      setFadingIds((prev) => new Set(prev).add(targetId))
      window.setTimeout(() => {
        setRecords((prev) => prev.filter((r) => r.id !== targetId))
        setFadingIds((prev) => {
          const next = new Set(prev)
          next.delete(targetId)
          return next
        })
      }, 380)

      onRecordSoftDeleted?.(targetId)
      showToast?.(data.message || '已标记为无效，不参与科研统计', true)
    } catch (error) {
      showToast?.(
        `⚠️ 软删除失败：${error instanceof Error ? error.message : '未知错误'}`,
        false,
      )
    } finally {
      setDeletingId(null)
    }
  }

  const colSpan = compact ? 3 : 7

  return (
    <section
      className={`flex min-h-0 flex-col rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-xl ${
        compact ? 'h-full' : ''
      } ${className}`.trim()}
    >
      <div className="mb-3 flex flex-shrink-0 flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="flex items-center gap-2 text-sm font-semibold text-white">
            <span className="inline-flex h-7 w-7 items-center justify-center rounded-xl bg-rose-500/15 ring-1 ring-rose-400/30">
              <ShieldAlert className="h-3.5 w-3.5 text-rose-300" />
            </span>
            {compact ? '清道夫 · 尝试日志' : '数据清道夫 · 误测记录治理'}
          </h2>
          <p className="mt-1 text-[11px] text-white/40">
            {compact
              ? scopedStudentId
                ? `仅显示被试 ${scopedStudentId} 的历史尝试 · 删除后中栏曲线自动自愈`
                : '请先在左侧选择被试'
              : '软删除仅标记无效；人工标定写入 provenance=calibrated，可进入科研实测过滤。'}
          </p>
        </div>
        <span className="text-[11px] text-white/35">有效 {visibleRecords.length} 条</span>
      </div>

      {!compact && (
        <div className="mb-4 flex flex-shrink-0 flex-col gap-2.5 sm:flex-row sm:flex-wrap sm:items-center">
          <label className="flex items-center gap-2 rounded-2xl bg-black/25 px-3 py-2 text-xs text-white/45">
            <CalendarDays className="h-3.5 w-3.5 text-amber-400" />
            <span className="whitespace-nowrap">起</span>
            <input
              type="date"
              value={dateFrom}
              onChange={(e) => setDateFrom(e.target.value)}
              className="bg-transparent text-sm text-white outline-none [color-scheme:dark]"
            />
          </label>
          <label className="flex items-center gap-2 rounded-2xl bg-black/25 px-3 py-2 text-xs text-white/45">
            <CalendarDays className="h-3.5 w-3.5 text-amber-400" />
            <span className="whitespace-nowrap">止</span>
            <input
              type="date"
              value={dateTo}
              onChange={(e) => setDateTo(e.target.value)}
              className="bg-transparent text-sm text-white outline-none [color-scheme:dark]"
            />
          </label>
          <label className="flex min-w-[160px] flex-1 items-center gap-2 rounded-2xl bg-black/25 px-3 py-2 text-xs text-white/45 sm:max-w-xs">
            <Search className="h-3.5 w-3.5 text-sky-400" />
            <input
              type="search"
              value={studentQuery}
              onChange={(e) => setStudentQuery(e.target.value)}
              placeholder="搜索被试编号…"
              className="w-full bg-transparent text-sm text-white outline-none placeholder:text-white/25"
            />
          </label>
          <label className="flex items-center gap-2 rounded-2xl bg-black/25 px-3 py-2 text-xs text-white/45">
            <Filter className="h-3.5 w-3.5 text-emerald-400" />
            <span className="whitespace-nowrap">组别</span>
            <select
              value={groupFilter}
              onChange={(e) => setGroupFilter(e.target.value as GroupFilter)}
              className="rounded-lg bg-transparent text-sm font-medium text-white outline-none [&>option]:bg-zinc-900"
            >
              <option value="all">全部组别</option>
              <option value="realtime">实时反馈 A 组</option>
              <option value="delayed">延时反馈 B 组</option>
            </select>
          </label>
        </div>
      )}

      {!scopedStudentId && compact ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-white/10 bg-black/20 px-4 py-16 text-center">
          <Users className="h-8 w-8 text-white/20" />
          <p className="text-xs text-white/35">选择左侧被试后，此处显示其可治理的尝试记录</p>
        </div>
      ) : (
        <div
          className={`min-h-0 flex-1 overflow-y-auto overflow-x-hidden rounded-2xl border border-white/8 bg-black/20 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent ${
            compact ? '' : 'max-h-[480px]'
          }`}
        >
          <table className="min-w-full border-collapse text-left text-sm">
            <thead className="sticky top-0 z-10 bg-slate-900/95 backdrop-blur-md">
              <tr className="border-b border-white/10 text-[11px] uppercase tracking-wide text-white/40">
                <th className="px-3 py-2.5 font-medium">时间</th>
                {!compact && <th className="px-3 py-2.5 font-medium">被试编号</th>}
                {!compact && <th className="px-3 py-2.5 font-medium">组别</th>}
                <th className="px-3 py-2.5 font-medium">总分</th>
                {!compact && <th className="px-3 py-2.5 font-medium">诊断快照</th>}
                {!compact && <th className="px-3 py-2.5 text-center font-medium">标定</th>}
                <th className="px-3 py-2.5 text-center font-medium">删除</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={colSpan} className="px-4 py-12 text-center text-white/40">
                    <span className="inline-flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin text-emerald-400" />
                      正在加载归档记录…
                    </span>
                  </td>
                </tr>
              ) : visibleRecords.length === 0 ? (
                <tr>
                  <td colSpan={colSpan} className="px-4 py-12 text-center text-white/35">
                    <span className="inline-flex flex-col items-center gap-2">
                      <Users className="h-5 w-5 text-white/25" />
                      当前没有可治理的记录
                    </span>
                  </td>
                </tr>
              ) : (
                <AnimatePresence initial={false}>
                  {records.map((record) => {
                    const isFading = fadingIds.has(record.id)
                    return (
                      <motion.tr
                        key={record.id}
                        layout
                        initial={{ opacity: 1 }}
                        animate={{ opacity: isFading ? 0 : 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.35 }}
                        className={`border-b border-white/5 text-white/80 last:border-0 hover:bg-white/[0.03] ${
                          isFading ? 'pointer-events-none bg-rose-500/10' : ''
                        }`}
                      >
                        <td className="whitespace-nowrap px-3 py-2.5 align-middle text-xs text-white/55">
                          {record.timestamp || record.testDate || '—'}
                        </td>
                        {!compact && (
                          <td className="whitespace-nowrap px-3 py-2.5 align-middle font-medium text-emerald-200/90">
                            {record.studentId || '—'}
                          </td>
                        )}
                        {!compact && (
                          <td className="whitespace-nowrap px-3 py-2.5 align-middle text-xs">
                            <span className="rounded-lg bg-white/5 px-2 py-0.5 text-white/70 ring-1 ring-white/10">
                              {groupLabel(record)}
                            </span>
                          </td>
                        )}
                        <td className="whitespace-nowrap px-3 py-2.5 align-middle tabular-nums">
                          {typeof record.score === 'number' ? (
                            <span className="font-semibold text-amber-200/90">{record.score}</span>
                          ) : (
                            <span className="text-white/30">—</span>
                          )}
                        </td>
                        {!compact && (
                          <td className="max-w-[280px] px-3 py-2.5 align-middle text-xs leading-relaxed text-white/50">
                            <span className="line-clamp-2">
                              {record.diagnosisSnapshot || record.aiFeedback || '（无诊断批注）'}
                            </span>
                            {record.supportFootDistanceProvenance === 'calibrated' && (
                              <span className="mt-1 inline-block text-[10px] text-sky-300/80">
                                横距已标定
                              </span>
                            )}
                          </td>
                        )}
                        {!compact && (
                          <td className="px-3 py-2.5 text-center align-middle">
                            <button
                              type="button"
                              title="人工标定实测值"
                              onClick={() => openCalibrate(record)}
                              className="inline-flex h-8 w-8 items-center justify-center rounded-xl bg-sky-500/10 text-sky-300 ring-1 ring-sky-400/25 transition hover:bg-sky-500/25 hover:text-sky-100 active:scale-95"
                            >
                              <PencilLine className="h-3.5 w-3.5" />
                            </button>
                          </td>
                        )}
                        <td className="px-3 py-2.5 text-center align-middle">
                          <button
                            type="button"
                            title="标记为无效误测"
                            disabled={deletingId === record.id}
                            onClick={() => setConfirmTarget(record)}
                            className="inline-flex h-8 w-8 items-center justify-center rounded-xl bg-rose-500/10 text-rose-400 ring-1 ring-rose-400/25 transition hover:bg-rose-500/25 hover:text-rose-200 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            {deletingId === record.id ? (
                              <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            ) : (
                              <Trash2 className="h-3.5 w-3.5" />
                            )}
                          </button>
                        </td>
                      </motion.tr>
                    )
                  })}
                </AnimatePresence>
              )}
            </tbody>
          </table>
        </div>
      )}

      <AnimatePresence>
        {confirmTarget && (
          <motion.div
            className="fixed inset-0 z-[70] flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => (deletingId ? null : setConfirmTarget(null))}
          >
            <motion.div
              role="dialog"
              aria-modal="true"
              initial={{ opacity: 0, scale: 0.94, y: 8 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.94, y: 8 }}
              transition={{ type: 'spring', stiffness: 380, damping: 28 }}
              className="relative w-full max-w-md rounded-3xl border border-rose-400/25 bg-zinc-950/95 p-5 shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              <button
                type="button"
                className="absolute top-3 right-3 rounded-full p-1.5 text-white/40 hover:bg-white/10 hover:text-white"
                onClick={() => setConfirmTarget(null)}
                disabled={!!deletingId}
              >
                <X className="h-4 w-4" />
              </button>
              <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-2xl bg-rose-500/15 ring-1 ring-rose-400/30">
                <Trash2 className="h-4.5 w-4.5 text-rose-300" />
              </div>
              <h3 className="text-base font-semibold text-white">确认标记无效？</h3>
              <p className="mt-2 text-sm leading-relaxed text-white/55">
                确定将该条误测数据标记为无效吗？此数据将不参与最终科研统计，中栏曲线将自动重绘。
              </p>
              <p className="mt-2 rounded-xl bg-black/30 px-3 py-2 text-xs text-white/40">
                被试 <span className="text-emerald-300">{confirmTarget.studentId || '—'}</span>
                {' · '}
                {confirmTarget.timestamp || confirmTarget.testDate || '未知时间'}
                {' · '}
                总分{' '}
                <span className="text-amber-200">
                  {typeof confirmTarget.score === 'number' ? confirmTarget.score : '—'}
                </span>
              </p>
              <div className="mt-5 flex justify-end gap-2">
                <button
                  type="button"
                  disabled={!!deletingId}
                  onClick={() => setConfirmTarget(null)}
                  className="rounded-full px-4 py-2 text-xs font-medium text-white/60 transition hover:bg-white/10 hover:text-white disabled:opacity-50"
                >
                  取消
                </button>
                <button
                  type="button"
                  disabled={!!deletingId}
                  onClick={() => void confirmSoftDelete()}
                  className="inline-flex items-center gap-1.5 rounded-full bg-rose-500/90 px-4 py-2 text-xs font-semibold text-white transition hover:bg-rose-500 disabled:opacity-60"
                >
                  {deletingId ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                  确认标记无效
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {calibrateTarget && (
          <motion.div
            className="fixed inset-0 z-[70] flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => (calibrating ? null : setCalibrateTarget(null))}
          >
            <motion.div
              role="dialog"
              aria-modal="true"
              initial={{ opacity: 0, scale: 0.94, y: 8 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.94, y: 8 }}
              transition={{ type: 'spring', stiffness: 380, damping: 28 }}
              className="relative w-full max-w-md rounded-3xl border border-sky-400/25 bg-zinc-950/95 p-5 shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              <button
                type="button"
                className="absolute top-3 right-3 rounded-full p-1.5 text-white/40 hover:bg-white/10 hover:text-white"
                onClick={() => setCalibrateTarget(null)}
                disabled={calibrating}
              >
                <X className="h-4 w-4" />
              </button>
              <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-2xl bg-sky-500/15 ring-1 ring-sky-400/30">
                <PencilLine className="h-4.5 w-4.5 text-sky-300" />
              </div>
              <h3 className="text-base font-semibold text-white">人工标定实测值</h3>
              <p className="mt-2 text-sm leading-relaxed text-white/55">
                将写入 provenance=calibrated，可供 AIGC 复述与 measured_only 科研导出；不会伪装为传感器
                measured。
              </p>
              <label className="mt-4 block text-xs text-white/45">
                指标
                <select
                  value={calibrateMetric}
                  onChange={(e) => {
                    const key = e.target.value as CoachCalibrateMetricKey
                    setCalibrateMetric(key)
                    const current = calibrateTarget
                      ? currentMetricValue(calibrateTarget, key)
                      : null
                    setCalibrateValue(current != null ? String(current) : '')
                  }}
                  className="mt-1 w-full rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-sm text-white outline-none"
                >
                  {CALIBRATE_OPTIONS.map((opt) => (
                    <option key={opt.key} value={opt.key} className="bg-zinc-900">
                      {opt.label}（{opt.unit}）
                    </option>
                  ))}
                </select>
              </label>
              <label className="mt-3 block text-xs text-white/45">
                标定值
                <input
                  type="number"
                  step="any"
                  value={calibrateValue}
                  onChange={(e) => setCalibrateValue(e.target.value)}
                  className="mt-1 w-full rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-sm text-white outline-none"
                />
              </label>
              <label className="mt-3 block text-xs text-white/45">
                备注（可选）
                <input
                  type="text"
                  value={calibrateNote}
                  onChange={(e) => setCalibrateNote(e.target.value)}
                  placeholder="如：地标盘复核 17.5cm"
                  className="mt-1 w-full rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-sm text-white outline-none placeholder:text-white/25"
                />
              </label>
              <div className="mt-5 flex justify-end gap-2">
                <button
                  type="button"
                  disabled={calibrating}
                  onClick={() => setCalibrateTarget(null)}
                  className="rounded-full px-4 py-2 text-xs font-medium text-white/60 transition hover:bg-white/10 hover:text-white disabled:opacity-50"
                >
                  取消
                </button>
                <button
                  type="button"
                  disabled={calibrating}
                  onClick={() => void confirmCalibrate()}
                  className="inline-flex items-center gap-1.5 rounded-full bg-sky-500/90 px-4 py-2 text-xs font-semibold text-white transition hover:bg-sky-500 disabled:opacity-60"
                >
                  {calibrating ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <PencilLine className="h-3.5 w-3.5" />
                  )}
                  确认写入 calibrated
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  )
}
