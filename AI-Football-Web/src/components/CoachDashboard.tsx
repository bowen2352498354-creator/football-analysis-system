import { useEffect, useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Users,
  Radio,
  Clock3,
  Gauge,
  RefreshCcw,
  Loader2,
  Inbox,
  School as SchoolIcon,
  Layers,
  CheckCircle2,
  XCircle,
  FileSpreadsheet,
  CalendarDays,
  TrendingUp,
  Crosshair,
  MousePointerClick,
} from 'lucide-react'
import { loadGlobalRecordsFromLocalStorage, saveGlobalRecordsToLocalStorage } from '../mockData'
import LongitudinalProgressChart from './LongitudinalProgressChart'
import BiomechanicalRadar from './BiomechanicalRadar'
import CoachDataSanitizerGrid from './CoachDataSanitizerGrid'
import type {
  AcademicExportResult,
  GlobalTrainingRecord,
  ProgressHistoryPoint,
  ProgressHistoryResponse,
} from '../types'

const API_BASE_URL = 'http://localhost:8000'
const SCHOOL_FALLBACK = '未设置学校'
const CLASS_FALLBACK = '未设置班级'
const DOSE_WARNING_THRESHOLD = 10

function getRecordTestDate(record: GlobalTrainingRecord): string {
  if (record.testDate && record.testDate.length >= 10) return record.testDate
  return (record.timestamp || '').slice(0, 10) || '未知日期'
}

interface DashboardToastState {
  id: number
  message: string
  success: boolean
}

let dashboardToastSeq = 0

type LoadState = 'loading' | 'ready'

/** 花名册条目：按被试聚合 */
interface StudentAggregate {
  key: string
  studentId: string
  school: string
  classGroup: string
  group: 'GROUP_A' | 'GROUP_B' | 'OTHER'
  records: GlobalTrainingRecord[]
}

function resolveGroup(record: GlobalTrainingRecord): StudentAggregate['group'] {
  if (record.type === 'realtime' || record.groupTypeCode === 1) return 'GROUP_A'
  if (record.type === 'delayed' || record.groupTypeCode === 2) return 'GROUP_B'
  const label = (record.classGroup || '').toUpperCase()
  if (label.includes('A') || label.includes('实时')) return 'GROUP_A'
  if (label.includes('B') || label.includes('延时')) return 'GROUP_B'
  return 'OTHER'
}

/**
 * 教练端科研指挥中心（Catapult / Hudl 宏观→微观三层下钻）
 *
 * 左栏花名册（剂量） → 中栏个人图谱（时序+雷达） → 右栏清道夫尝试日志
 * 三栏各自 `h-[calc(100vh-80px)] overflow-y-auto` 局部滚动，互不干扰。
 */
export default function CoachDashboard() {
  const [records, setRecords] = useState<GlobalTrainingRecord[]>([])
  const [loadState, setLoadState] = useState<LoadState>('loading')
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [toast, setToast] = useState<DashboardToastState | null>(null)

  const [selectedSchool, setSelectedSchool] = useState<string>('all')
  const [selectedClassGroup, setSelectedClassGroup] = useState<string>('all')
  const [selectedTestDate, setSelectedTestDate] = useState<string>('all')

  const [isExportingMatrix, setIsExportingMatrix] = useState(false)
  const [selectedStudent, setSelectedStudent] = useState<StudentAggregate | null>(null)
  const [selectedAttemptIndex, setSelectedAttemptIndex] = useState(0)
  /** 软删除后递增：驱动中栏曲线 / 雷达 / 右栏列表 re-fetch */
  const [dataEpoch, setDataEpoch] = useState(0)
  const [progressHistory, setProgressHistory] = useState<ProgressHistoryPoint[] | null>(null)

  function showToast(message: string, success: boolean) {
    const id = ++dashboardToastSeq
    setToast({ id, message, success })
    window.setTimeout(() => {
      setToast((current) => (current?.id === id ? null : current))
    }, 3200)
  }

  async function fetchAllRecords(isManualRefresh = false) {
    if (isManualRefresh) setIsRefreshing(true)
    const localRecords = loadGlobalRecordsFromLocalStorage()
    try {
      const response = await fetch(`${API_BASE_URL}/api/get_all_records`)
      if (!response.ok) throw new Error(`接口返回状态码 ${response.status}`)
      const data = (await response.json()) as { success: boolean; records?: GlobalTrainingRecord[] }
      const backendRecords = Array.isArray(data.records) ? data.records : []

      const backendIds = new Set(backendRecords.map((r) => r.id))
      const mergedMap = new Map<string, GlobalTrainingRecord>()
      for (const record of backendRecords) mergedMap.set(record.id, record)
      for (const record of localRecords) {
        if (backendIds.has(record.id)) continue
        if (record.is_deleted || record.isDeleted) continue
        mergedMap.set(record.id, record)
      }
      const merged = Array.from(mergedMap.values())

      setRecords(merged)
      saveGlobalRecordsToLocalStorage(merged)
      if (isManualRefresh) showToast(`✅ 已刷新，共加载 ${merged.length} 条历史归档记录`, true)
    } catch {
      setRecords(localRecords)
      if (isManualRefresh) {
        showToast(
          localRecords.length > 0
            ? `⚠️ 后端服务未响应，已回退展示本地缓存的 ${localRecords.length} 条记录`
            : '⚠️ 后端服务未响应，且本地缓存暂无历史数据',
          false,
        )
      }
    } finally {
      setLoadState('ready')
      if (isManualRefresh) setIsRefreshing(false)
    }
  }

  useEffect(() => {
    void fetchAllRecords()
  }, [])

  const schoolOptions = useMemo(() => {
    const set = new Set<string>()
    records.forEach((record) => set.add(record.school || SCHOOL_FALLBACK))
    return Array.from(set).sort()
  }, [records])

  const classGroupOptions = useMemo(() => {
    const set = new Set<string>()
    records
      .filter((record) => selectedSchool === 'all' || (record.school || SCHOOL_FALLBACK) === selectedSchool)
      .forEach((record) => set.add(record.classGroup || CLASS_FALLBACK))
    return Array.from(set).sort()
  }, [records, selectedSchool])

  useEffect(() => {
    if (selectedClassGroup !== 'all' && !classGroupOptions.includes(selectedClassGroup)) {
      setSelectedClassGroup('all')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSchool, classGroupOptions])

  const dateOptions = useMemo(() => {
    const set = new Set<string>()
    records.forEach((record) => set.add(getRecordTestDate(record)))
    return Array.from(set).sort((a, b) => (a < b ? 1 : -1))
  }, [records])

  useEffect(() => {
    if (selectedTestDate === 'all') return
    const stillExists = records
      .filter((record) => selectedSchool === 'all' || (record.school || SCHOOL_FALLBACK) === selectedSchool)
      .filter((record) => selectedClassGroup === 'all' || (record.classGroup || CLASS_FALLBACK) === selectedClassGroup)
      .some((record) => getRecordTestDate(record) === selectedTestDate)
    if (!stillExists) setSelectedTestDate('all')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSchool, selectedClassGroup, records])

  const filteredRecords = useMemo(() => {
    return records
      .filter((record) => selectedSchool === 'all' || (record.school || SCHOOL_FALLBACK) === selectedSchool)
      .filter((record) => selectedClassGroup === 'all' || (record.classGroup || CLASS_FALLBACK) === selectedClassGroup)
      .filter((record) => selectedTestDate === 'all' || getRecordTestDate(record) === selectedTestDate)
      .sort((a, b) => (a.timestamp < b.timestamp ? 1 : -1))
  }, [records, selectedSchool, selectedClassGroup, selectedTestDate])

  async function handleExportAcademicMatrix() {
    if (isExportingMatrix) return
    setIsExportingMatrix(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/export/spss_matrix`)
      const contentType = response.headers.get('content-type') || ''
      if (!response.ok) {
        let message = `接口返回状态码 ${response.status}`
        if (contentType.includes('application/json')) {
          const data = (await response.json()) as AcademicExportResult
          message = data.message || message
        }
        throw new Error(message)
      }
      if (contentType.includes('application/json')) {
        const data = (await response.json()) as AcademicExportResult
        throw new Error(data.message || '导出失败：后端返回了错误信息而非 CSV')
      }
      const blob = await response.blob()
      const objectUrl = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = objectUrl
      anchor.download = 'AI_Football_Research_Matrix_V3.csv'
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      URL.revokeObjectURL(objectUrl)
      const rows = response.headers.get('X-Export-Row-Count')
      showToast(
        rows
          ? `✅ V3.1 科研宽表已下载（${rows} 名被试）· AI_Football_Research_Matrix_V3.csv`
          : '✅ V3.1 全数字化科研宽表已下载：AI_Football_Research_Matrix_V3.csv',
        true,
      )
    } catch (error) {
      showToast(
        `⚠️ 导出学术统计矩阵失败：${error instanceof Error ? error.message : '请检查后端服务是否已启动'}`,
        false,
      )
    } finally {
      setIsExportingMatrix(false)
    }
  }

  const kpi = useMemo(() => {
    const uniqueStudents = new Set(records.map((r) => `${r.school}__${r.classGroup}__${r.studentId}`))
    const realtimeStudents = new Set(
      records.filter((r) => r.type === 'realtime').map((r) => `${r.school}__${r.classGroup}__${r.studentId}`),
    )
    const delayedStudents = new Set(
      records.filter((r) => r.type === 'delayed').map((r) => `${r.school}__${r.classGroup}__${r.studentId}`),
    )
    const validScores = records.filter((r) => typeof r.score === 'number') as (GlobalTrainingRecord & { score: number })[]
    const avgScore =
      validScores.length > 0
        ? Math.round(validScores.reduce((sum, r) => sum + r.score, 0) / validScores.length)
        : null

    return {
      totalStudents: uniqueStudents.size,
      realtimeStudents: realtimeStudents.size,
      delayedStudents: delayedStudents.size,
      avgScore,
    }
  }, [records])

  const studentAggregates: StudentAggregate[] = useMemo(() => {
    const map = new Map<string, StudentAggregate>()
    filteredRecords.forEach((record) => {
      const school = record.school || SCHOOL_FALLBACK
      const classGroup = record.classGroup || CLASS_FALLBACK
      const studentId = record.studentId || '未填写编号'
      const key = `${school}__${classGroup}__${studentId}`
      if (!map.has(key)) {
        map.set(key, {
          key,
          studentId,
          school,
          classGroup,
          group: resolveGroup(record),
          records: [],
        })
      }
      map.get(key)!.records.push(record)
    })
    map.forEach((aggregate) => {
      aggregate.records.sort((a, b) => (a.timestamp < b.timestamp ? -1 : 1))
      if (aggregate.records.length > 0) {
        aggregate.group = resolveGroup(aggregate.records[aggregate.records.length - 1])
      }
    })
    return Array.from(map.values()).sort((a, b) => a.studentId.localeCompare(b.studentId))
  }, [filteredRecords])

  const rosterGroups = useMemo(() => {
    const groupA = studentAggregates.filter((s) => s.group === 'GROUP_A')
    const groupB = studentAggregates.filter((s) => s.group === 'GROUP_B')
    const other = studentAggregates.filter((s) => s.group === 'OTHER')
    return { groupA, groupB, other }
  }, [studentAggregates])

  // 筛选变化或数据洗净后，同步刷新当前选中被试的 records 引用
  useEffect(() => {
    if (!selectedStudent) return
    const fresh = studentAggregates.find((s) => s.key === selectedStudent.key) ?? null
    setSelectedStudent(fresh)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [studentAggregates, dataEpoch])

  useEffect(() => {
    if (selectedStudent && selectedStudent.records.length > 0) {
      setSelectedAttemptIndex(selectedStudent.records.length - 1)
    } else {
      setSelectedAttemptIndex(0)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedStudent?.key, selectedStudent?.records.length, dataEpoch])

  const clampedAttemptIndex = selectedStudent
    ? Math.min(Math.max(0, selectedAttemptIndex), Math.max(0, selectedStudent.records.length - 1))
    : 0

  const selectedAttempt: GlobalTrainingRecord | null =
    selectedStudent && selectedStudent.records.length > 0
      ? selectedStudent.records[clampedAttemptIndex]
      : null

  const radarScores =
    selectedAttempt?.quantified5dScores ??
    selectedStudent?.records[selectedStudent.records.length - 1]?.quantified5dScores ??
    null

  // 拉取纵向科研节点历史；dataEpoch 变化时强制重拉（洗净后曲线自愈）
  useEffect(() => {
    if (!selectedStudent) {
      setProgressHistory(null)
      return
    }
    let cancelled = false
    const sid = selectedStudent.studentId
    const school = selectedStudent.school || ''
    const classGroup = selectedStudent.classGroup || ''
    const qs = new URLSearchParams({
      student_id: sid,
      ...(school && school !== SCHOOL_FALLBACK ? { school } : {}),
      ...(classGroup && classGroup !== CLASS_FALLBACK ? { classGroup } : {}),
    })
    void (async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/progress/history?${qs.toString()}`)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = (await res.json()) as ProgressHistoryResponse
        if (!cancelled) {
          setProgressHistory(data.success && Array.isArray(data.points) ? data.points : [])
        }
      } catch {
        if (!cancelled) setProgressHistory(null)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [selectedStudent?.key, selectedStudent?.studentId, selectedStudent?.school, selectedStudent?.classGroup, dataEpoch])

  function handleRecordSoftDeleted(recordId: string) {
    setRecords((prev) => {
      const next = prev.filter((r) => r.id !== recordId)
      saveGlobalRecordsToLocalStorage(next)
      return next
    })
    setDataEpoch((n) => n + 1)
  }

  const panelScrollClass =
    'coach-panel-scroll h-[calc(100vh-80px)] overflow-y-auto overflow-x-hidden scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent'

  return (
    <div className="coach-dashboard-shell flex h-full min-h-0 flex-col overflow-hidden">
      {/* ============================ 顶栏：KPI + 筛选（紧凑，不锁死页面滚动） ============================ */}
      <div className="flex-shrink-0 space-y-3 border-b border-white/5 px-3 py-3 sm:px-4">
        <div className="flex flex-wrap items-center gap-2">
          <MiniKpi icon={Users} label="总人数" value={kpi.totalStudents} accent="emerald" />
          <MiniKpi icon={Radio} label="A 组" value={kpi.realtimeStudents} accent="sky" />
          <MiniKpi icon={Clock3} label="B 组" value={kpi.delayedStudents} accent="teal" />
          <MiniKpi
            icon={Gauge}
            label="均分"
            value={kpi.avgScore ?? '--'}
            accent="amber"
          />

          <div className="ml-auto flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => void handleExportAcademicMatrix()}
              disabled={isExportingMatrix}
              className="inline-flex items-center gap-1.5 rounded-full border border-amber-400/40 bg-amber-500/10 px-3 py-1.5 text-[11px] font-semibold text-amber-100 transition hover:bg-amber-500/20 disabled:opacity-60"
            >
              {isExportingMatrix ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <FileSpreadsheet className="h-3.5 w-3.5" />}
              导出 SPSS
            </button>
            <button
              type="button"
              onClick={() => void fetchAllRecords(true)}
              disabled={isRefreshing}
              className="inline-flex items-center gap-1.5 rounded-full bg-white/10 px-3 py-1.5 text-[11px] font-medium text-white transition hover:bg-white/20 disabled:opacity-60"
            >
              {isRefreshing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCcw className="h-3.5 w-3.5" />}
              刷新
            </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <label className="flex items-center gap-2 rounded-2xl bg-black/25 px-3 py-1.5 text-xs text-white/50">
            <SchoolIcon className="h-3.5 w-3.5 text-emerald-400" />
            <select
              value={selectedSchool}
              onChange={(e) => setSelectedSchool(e.target.value)}
              className="bg-transparent text-sm font-medium text-white outline-none [&>option]:bg-zinc-900"
            >
              <option value="all">全部学校</option>
              {schoolOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 rounded-2xl bg-black/25 px-3 py-1.5 text-xs text-white/50">
            <Layers className="h-3.5 w-3.5 text-sky-400" />
            <select
              value={selectedClassGroup}
              onChange={(e) => setSelectedClassGroup(e.target.value)}
              className="bg-transparent text-sm font-medium text-white outline-none [&>option]:bg-zinc-900"
            >
              <option value="all">全部班级/组别</option>
              {classGroupOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 rounded-2xl bg-black/25 px-3 py-1.5 text-xs text-white/50">
            <CalendarDays className="h-3.5 w-3.5 text-amber-400" />
            <select
              value={selectedTestDate}
              onChange={(e) => setSelectedTestDate(e.target.value)}
              className="bg-transparent text-sm font-medium text-white outline-none [&>option]:bg-zinc-900"
            >
              <option value="all">全部日期</option>
              {dateOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <span className="text-[11px] text-white/30">
            花名册 {studentAggregates.length} 人 · 三栏局部滚动
          </span>
        </div>
      </div>

      {/* ============================ 三栏主工作区 ============================ */}
      {loadState === 'loading' ? (
        <div className="flex flex-1 items-center justify-center gap-3 text-white/40">
          <Loader2 className="h-7 w-7 animate-spin text-emerald-400" />
          <p className="text-sm">正在加载全量历史归档数据……</p>
        </div>
      ) : records.length === 0 ? (
        <div className="flex flex-1 items-center justify-center p-6">
          <EmptyStateCard />
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 gap-3 overflow-hidden px-3 pb-3 pt-2 sm:px-4">
          {/* -------- 左栏：花名册与剂量监控 -------- */}
          <aside
            className={`w-[320px] flex-shrink-0 rounded-2xl border border-white/10 bg-slate-900/40 p-3 ${panelScrollClass}`}
          >
            <LeftSidebar
              rosterGroups={rosterGroups}
              selectedKey={selectedStudent?.key ?? null}
              onSelect={(student) => setSelectedStudent(student)}
            />
          </aside>

          {/* -------- 中栏：个人时序 + 雷达 -------- */}
          <main
            className={`min-w-0 flex-1 rounded-2xl border border-white/10 bg-slate-900/30 p-4 ${panelScrollClass}`}
          >
            <MainCanvas
              selectedStudent={selectedStudent}
              selectedAttemptIndex={clampedAttemptIndex}
              onSelectAttemptIndex={setSelectedAttemptIndex}
              progressHistory={progressHistory}
              radarScores={radarScores}
              dataEpoch={dataEpoch}
            />
          </main>

          {/* -------- 右栏：清道夫尝试日志 -------- */}
          <aside
            className={`flex w-[400px] flex-shrink-0 flex-col rounded-2xl border border-white/10 bg-slate-900/40 p-2 ${panelScrollClass}`}
          >
            <CoachDataSanitizerGrid
              compact
              studentId={selectedStudent?.studentId ?? null}
              refreshKey={dataEpoch}
              onRecordSoftDeleted={handleRecordSoftDeleted}
              showToast={showToast}
              className="min-h-0 flex-1"
            />
          </aside>
        </div>
      )}

      <AnimatePresence>
        {toast && (
          <motion.div
            key={toast.id}
            initial={{ opacity: 0, y: -16, scale: 0.94 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -16, scale: 0.94 }}
            transition={{ type: 'spring', stiffness: 300, damping: 26 }}
            className={`fixed top-20 right-6 z-[60] flex max-w-md items-start gap-3 rounded-2xl border px-5 py-3.5 shadow-2xl backdrop-blur-2xl ${
              toast.success
                ? 'border-emerald-400/30 bg-emerald-950/90 text-emerald-100'
                : 'border-rose-400/30 bg-rose-950/90 text-rose-100'
            }`}
          >
            <span className="mt-0.5 inline-flex flex-shrink-0">
              {toast.success ? (
                <CheckCircle2 className="h-4 w-4 text-emerald-300" />
              ) : (
                <XCircle className="h-4 w-4 text-rose-300" />
              )}
            </span>
            <p className="text-sm leading-relaxed break-all">{toast.message}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/* ============================================================================
 * 子组件
 * ========================================================================== */

const KPI_ACCENT: Record<string, string> = {
  emerald: 'text-emerald-300 bg-emerald-500/15',
  sky: 'text-sky-300 bg-sky-500/15',
  teal: 'text-teal-300 bg-teal-500/15',
  amber: 'text-amber-300 bg-amber-500/15',
}

function MiniKpi({
  icon: Icon,
  label,
  value,
  accent,
}: {
  icon: typeof Users
  label: string
  value: number | string
  accent: keyof typeof KPI_ACCENT
}) {
  return (
    <div className="inline-flex items-center gap-2 rounded-2xl border border-white/8 bg-white/5 px-2.5 py-1.5">
      <span className={`flex h-7 w-7 items-center justify-center rounded-xl ${KPI_ACCENT[accent]}`}>
        <Icon className="h-3.5 w-3.5" />
      </span>
      <div className="leading-tight">
        <p className="text-[10px] text-white/35">{label}</p>
        <p className="text-sm font-bold tabular-nums text-white/90">{value}</p>
      </div>
    </div>
  )
}

function EmptyStateCard() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex max-w-lg flex-col items-center gap-4 rounded-3xl border border-white/10 bg-gradient-to-br from-white/5 via-white/[0.02] to-transparent p-10 text-center"
    >
      <div className="flex h-16 w-16 items-center justify-center rounded-[24px] bg-gradient-to-br from-emerald-400/20 to-sky-500/20 ring-1 ring-white/10">
        <Inbox className="h-8 w-8 text-emerald-300" />
      </div>
      <h3 className="text-lg font-semibold text-white/90">暂无任何历史归档数据</h3>
      <p className="text-sm leading-relaxed text-white/40">
        前往「实时反馈系统」或「延时反馈系统」完成测试并开启本地落盘归档后，数据将同步至此处三栏指挥台。
      </p>
    </motion.div>
  )
}

function LeftSidebar({
  rosterGroups,
  selectedKey,
  onSelect,
}: {
  rosterGroups: {
    groupA: StudentAggregate[]
    groupB: StudentAggregate[]
    other: StudentAggregate[]
  }
  selectedKey: string | null
  onSelect: (student: StudentAggregate) => void
}) {
  const sections: Array<{ id: string; title: string; badge: string; students: StudentAggregate[] }> = [
    { id: 'GROUP_A', title: 'GROUP_A · 实时反馈', badge: 'A', students: rosterGroups.groupA },
    { id: 'GROUP_B', title: 'GROUP_B · 延时反馈', badge: 'B', students: rosterGroups.groupB },
  ]
  if (rosterGroups.other.length > 0) {
    sections.push({ id: 'OTHER', title: '未归组', badge: '?', students: rosterGroups.other })
  }

  const total =
    rosterGroups.groupA.length + rosterGroups.groupB.length + rosterGroups.other.length

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-white/85">
          <Users className="h-4 w-4 text-emerald-300" />
          花名册 · 剂量监控
        </h3>
        <span className="rounded-full bg-black/30 px-2 py-0.5 text-[10px] text-white/40">{total} 人</span>
      </div>

      {total === 0 ? (
        <p className="rounded-2xl border border-dashed border-white/10 bg-black/20 px-3 py-8 text-center text-xs text-white/35">
          当前筛选条件下暂无被试
        </p>
      ) : (
        sections.map((section) =>
          section.students.length === 0 ? null : (
            <div key={section.id} className="flex flex-col gap-2">
              <div className="flex items-center gap-2 px-1">
                <span
                  className={`flex h-5 w-5 items-center justify-center rounded-md text-[10px] font-bold ${
                    section.id === 'GROUP_A'
                      ? 'bg-sky-500/20 text-sky-300'
                      : section.id === 'GROUP_B'
                        ? 'bg-teal-500/20 text-teal-300'
                        : 'bg-white/10 text-white/50'
                  }`}
                >
                  {section.badge}
                </span>
                <p className="text-[11px] font-semibold tracking-wide text-white/45">{section.title}</p>
                <span className="text-[10px] text-white/25">{section.students.length}</span>
              </div>
              <div className="flex flex-col gap-1.5">
                {section.students.map((student) => {
                  const dose = student.records.length
                  const underdosed = dose < DOSE_WARNING_THRESHOLD
                  const isActive = student.key === selectedKey
                  return (
                    <button
                      key={student.key}
                      type="button"
                      onClick={() => onSelect(student)}
                      className={`flex items-center justify-between rounded-xl border px-3 py-2.5 text-left transition ${
                        isActive
                          ? 'border-emerald-400/45 bg-emerald-400/15 text-white shadow-[0_0_20px_rgba(16,185,129,0.12)]'
                          : 'border-white/5 bg-black/25 text-white/65 hover:border-white/15 hover:bg-white/8'
                      }`}
                    >
                      <span className="min-w-0">
                        <span className="block truncate text-sm font-medium">{student.studentId}</span>
                        <span className="block truncate text-[10px] text-white/30">
                          {student.school} · {student.classGroup}
                        </span>
                      </span>
                      <span
                        className={`ml-2 flex-shrink-0 rounded-md px-1.5 py-0.5 text-[10px] font-bold tabular-nums ${
                          underdosed
                            ? 'bg-rose-500/90 text-white shadow-[0_0_10px_rgba(239,68,68,0.35)]'
                            : 'bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-400/30'
                        }`}
                        title={underdosed ? `已练 ${dose} 次，未达 10 次剂量` : `已练 ${dose} 次`}
                      >
                        {dose}次
                      </span>
                    </button>
                  )
                })}
              </div>
            </div>
          ),
        )
      )}
    </div>
  )
}

function MainCanvas({
  selectedStudent,
  selectedAttemptIndex,
  onSelectAttemptIndex,
  progressHistory,
  radarScores,
  dataEpoch,
}: {
  selectedStudent: StudentAggregate | null
  selectedAttemptIndex: number
  onSelectAttemptIndex: (index: number) => void
  progressHistory: ProgressHistoryPoint[] | null
  radarScores: GlobalTrainingRecord['quantified5dScores']
  dataEpoch: number
}) {
  if (!selectedStudent) {
    return (
      <div className="flex h-full min-h-[420px] flex-col items-center justify-center gap-5 px-6 text-center">
        <div className="relative flex h-24 w-24 items-center justify-center rounded-[28px] bg-gradient-to-br from-emerald-500/15 via-sky-500/10 to-transparent ring-1 ring-white/10">
          <MousePointerClick className="h-10 w-10 text-emerald-300/80" />
          <span className="absolute -left-2 top-1/2 -translate-y-1/2 text-2xl opacity-60">👈</span>
        </div>
        <div className="max-w-md space-y-2">
          <h3 className="text-lg font-semibold text-white/90">
            请在左侧选择被试以查看科研分析图谱
          </h3>
          <p className="text-sm leading-relaxed text-white/40">
            Catapult / Hudl 式三层下钻：花名册剂量 → 个人时序与五维雷达 → 右侧尝试级清道夫日志。
          </p>
        </div>
      </div>
    )
  }

  const latest = selectedStudent.records[selectedStudent.records.length - 1]
  const latestScore = latest?.score ?? '--'

  return (
    <div className="flex flex-col gap-5">
      <section className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="flex h-14 w-14 flex-col items-center justify-center rounded-2xl bg-emerald-400/15 ring-1 ring-emerald-400/25">
              <span className="text-xl font-bold text-emerald-300">{latestScore}</span>
              <span className="text-[9px] text-emerald-300/60">最新分</span>
            </div>
            <div>
              <p className="text-lg font-semibold text-white">{selectedStudent.studentId}</p>
              <p className="text-xs text-white/40">
                {selectedStudent.school} · {selectedStudent.classGroup} · {selectedStudent.group}
              </p>
            </div>
          </div>
          <p className="text-xs text-white/35">
            全周期 {selectedStudent.records.length} 次 · Attempt #1 → #{selectedStudent.records.length}
          </p>
        </div>
      </section>

      <section className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
        <h4 className="mb-3 flex items-center gap-2 text-sm font-semibold text-white/80">
          <TrendingUp className="h-4 w-4 text-sky-300" />
          个人进步趋势图 · 真实日期 × 科研节点
        </h4>
        {selectedStudent.records.length === 0 ? (
          <div className="flex h-56 items-center justify-center rounded-2xl bg-black/20 text-xs text-white/25">
            暂无尝试数据
          </div>
        ) : (
          <LongitudinalProgressChart
            key={`chart-${selectedStudent.key}-${dataEpoch}`}
            records={selectedStudent.records}
            historyPoints={progressHistory}
            selectedIndex={selectedAttemptIndex}
            onSelectIndex={onSelectAttemptIndex}
            studentId={selectedStudent.studentId}
          />
        )}
      </section>

      <section className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
        <h4 className="mb-3 flex items-center gap-2 text-sm font-semibold text-white/80">
          <Crosshair className="h-4 w-4 text-emerald-300" />
          五维生物力学雷达 · Attempt #{selectedAttemptIndex + 1}
        </h4>
        <BiomechanicalRadar
          key={`radar-${selectedStudent.key}-${dataEpoch}-${selectedAttemptIndex}`}
          scores={radarScores}
          primaryLabel={`Attempt #${selectedAttemptIndex + 1}`}
        />
      </section>
    </div>
  )
}
