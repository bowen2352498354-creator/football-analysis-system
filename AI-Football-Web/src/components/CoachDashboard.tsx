import { useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react'
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
  Search,
  ChevronDown,
  ChevronRight,
  FlaskConical,
  Filter,
  RotateCcw,
  Activity,
  UsersRound,
  UserRound,
  GitCompareArrows,
  X,
  Trash2,
  ChevronsUpDown,
  Sparkles,
  Upload,
  AlertCircle,
  ThumbsUp,
  Target,
  ClipboardList,
} from 'lucide-react'
import {
  loadGlobalRecordsFromLocalStorage,
  loadHiddenClassGroupNames,
  loadHiddenSchoolNames,
  removeClassGroupOption,
  removeSchoolOption,
  saveGlobalRecordsToLocalStorage,
} from '../mockData'
import LongitudinalProgressChart from './LongitudinalProgressChart'
import BiomechanicalRadar from './BiomechanicalRadar'
import CoachDataSanitizerGrid from './CoachDataSanitizerGrid'
import CohortComparePanel from './CohortComparePanel'
import ClassProfilingPanel from './ClassProfilingPanel'
import InterventionDosageMonitor from './InterventionDosageMonitor'
import type {
  AcademicExportResult,
  GlobalTrainingRecord,
  IndividualSummaryReport,
  ProgressHistoryPoint,
  ProgressHistoryResponse,
  Quantified5dScores,
  RadarAverageScores,
} from '../types'

type CoachAnalysisTab = 'classProfile' | 'individual' | 'cohortCompare' | 'dosage'

const COACH_ANALYSIS_TABS: Array<{ id: CoachAnalysisTab; label: string }> = [
  { id: 'classProfile', label: '班级群体画像' },
  { id: 'individual', label: '个体复盘' },
  { id: 'cohortCompare', label: '班级对比' },
  { id: 'dosage', label: '实验干预剂量' },
]

const TAB_ICONS: Record<CoachAnalysisTab, typeof Activity> = {
  classProfile: UsersRound,
  individual: UserRound,
  cohortCompare: GitCompareArrows,
  dosage: Activity,
}

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

type ExperimentGroupId = 'GROUP_A' | 'GROUP_B' | 'OTHER'

/** CTMS 级全局实验过滤器（学校 / 班级 / 组别 / 日期区间） */
interface GlobalFilter {
  school: string
  class: string
  group: string
  dateRange: [string | null, string | null]
}

const DEFAULT_GLOBAL_FILTER: GlobalFilter = {
  school: 'all',
  class: 'all',
  group: 'all',
  dateRange: [null, null],
}

function isDateInRange(dateStr: string, range: [string | null, string | null]): boolean {
  const [start, end] = range
  if (!start && !end) return true
  if (dateStr === '未知日期') return false
  if (start && dateStr < start) return false
  if (end && dateStr > end) return false
  return true
}

/** 花名册条目：按被试聚合 */
interface StudentAggregate {
  key: string
  studentId: string
  school: string
  classGroup: string
  group: ExperimentGroupId
  records: GlobalTrainingRecord[]
}

/** 树状花名册：班级 → 实验组 → 学生 */
interface RosterGroupNode {
  group: ExperimentGroupId
  label: string
  badge: string
  students: StudentAggregate[]
}

interface RosterClassNode {
  classGroup: string
  groups: RosterGroupNode[]
  studentCount: number
}

const GROUP_META: Record<
  ExperimentGroupId,
  { label: string; badge: string; shortLabel: string }
> = {
  GROUP_A: { label: 'A组 · 实时反馈', badge: 'A', shortLabel: 'A组' },
  GROUP_B: { label: 'B组 · 延时反馈', badge: 'B', shortLabel: 'B组' },
  OTHER: { label: '未归组', badge: '?', shortLabel: '未归组' },
}

const GROUP_ORDER: ExperimentGroupId[] = ['GROUP_A', 'GROUP_B', 'OTHER']

function resolveGroup(record: GlobalTrainingRecord): ExperimentGroupId {
  if (record.type === 'realtime' || record.groupTypeCode === 1) return 'GROUP_A'
  if (record.type === 'delayed' || record.groupTypeCode === 2) return 'GROUP_B'
  const label = (record.classGroup || '').toUpperCase()
  if (label.includes('A') || label.includes('实时')) return 'GROUP_A'
  if (label.includes('B') || label.includes('延时')) return 'GROUP_B'
  return 'OTHER'
}

function isActiveRecord(record: GlobalTrainingRecord | null | undefined): record is GlobalTrainingRecord {
  return Boolean(record && typeof record.id === 'string' && !(record.is_deleted || record.isDeleted))
}

function normalizeSchool(value: string | undefined | null): string {
  const trimmed = (value || '').trim()
  return trimmed || SCHOOL_FALLBACK
}

function normalizeClassGroup(value: string | undefined | null): string {
  const trimmed = (value || '').trim()
  return trimmed || CLASS_FALLBACK
}

/**
 * 教练端科研指挥中心（互斥 Tab 路由布局）
 *
 * 固定头部：KPI + 全局控制台 + Tab 栏
 * 动态区按 activeTab 互斥渲染：班级群体画像 | 个体复盘 | 班级对比 | 实验干预剂量
 * 整页锁在一屏高度内，仅动态区内部滚动。
 */
export default function CoachDashboard() {
  const [records, setRecords] = useState<GlobalTrainingRecord[]>([])
  const [loadState, setLoadState] = useState<LoadState>('loading')
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [toast, setToast] = useState<DashboardToastState | null>(null)

  const [globalFilter, setGlobalFilter] = useState<GlobalFilter>(DEFAULT_GLOBAL_FILTER)
  const [activeTab, setActiveTab] = useState<CoachAnalysisTab>('classProfile')
  /** 学校/班级选项被删除（隐藏）后递增，驱动下拉列表重算 */
  const [filterOptionsEpoch, setFilterOptionsEpoch] = useState(0)

  const [isExportingMatrix, setIsExportingMatrix] = useState(false)
  const [selectedStudent, setSelectedStudent] = useState<StudentAggregate | null>(null)
  const [selectedAttemptIndex, setSelectedAttemptIndex] = useState(0)
  /** 软删除后递增：驱动中栏曲线 / 雷达 / 右栏列表 re-fetch */
  const [dataEpoch, setDataEpoch] = useState(0)
  const [progressHistory, setProgressHistory] = useState<ProgressHistoryPoint[] | null>(null)
  const [radarAverage, setRadarAverage] = useState<RadarAverageScores | null>(null)
  /** 个人详细数据默认最小化，点击或选中被试后再展开 */
  const [detailPanelOpen, setDetailPanelOpen] = useState(false)

  function showToast(message: string, success: boolean) {
    const id = ++dashboardToastSeq
    setToast({ id, message, success })
    window.setTimeout(() => {
      setToast((current) => (current?.id === id ? null : current))
    }, 3200)
  }

  async function fetchAllRecords(isManualRefresh = false) {
    if (isManualRefresh) setIsRefreshing(true)
    const localRecords = loadGlobalRecordsFromLocalStorage().filter(isActiveRecord)
    try {
      const response = await fetch(`${API_BASE_URL}/api/get_all_records`)
      if (!response.ok) throw new Error(`接口返回状态码 ${response.status}`)
      const data = (await response.json()) as { success: boolean; records?: GlobalTrainingRecord[] }
      // 后端在线时以后端为唯一真相源，禁止用 localStorage 幽灵记录复活已删除数据
      const backendRecords = (Array.isArray(data.records) ? data.records : []).filter(isActiveRecord)

      setRecords(backendRecords)
      saveGlobalRecordsToLocalStorage(backendRecords)
      if (!backendRecords.length) {
        setSelectedStudent(null)
        setProgressHistory(null)
        setRadarAverage(null)
      }
      if (isManualRefresh) showToast(`✅ 已刷新，共加载 ${backendRecords.length} 条历史归档记录`, true)
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

  /** 唯一真相源：当前未删除的完整尝试记录（已剔除软删除） */
  const activeRecords = useMemo(() => records.filter(isActiveRecord), [records])

  /** 从最新活跃数据动态提取学校列表，并剔除教师主动隐藏的幽灵项 */
  const schoolOptions = useMemo(() => {
    const hidden = new Set(loadHiddenSchoolNames())
    const set = new Set<string>()
    activeRecords.forEach((record) => {
      const name = normalizeSchool(record.school)
      if (!hidden.has(name)) set.add(name)
    })
    return Array.from(set).sort((a, b) => a.localeCompare(b, 'zh-CN'))
    // filterOptionsEpoch：删除选项后强制重读 localStorage 隐藏名单
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeRecords, filterOptionsEpoch])

  /** 班级列表：随所选学校级联，剔除已隐藏项 */
  const classGroupOptions = useMemo(() => {
    const hidden = new Set(loadHiddenClassGroupNames())
    const set = new Set<string>()
    activeRecords
      .filter((record) => globalFilter.school === 'all' || normalizeSchool(record.school) === globalFilter.school)
      .forEach((record) => {
        const name = normalizeClassGroup(record.classGroup)
        if (!hidden.has(name)) set.add(name)
      })
    return Array.from(set).sort((a, b) => a.localeCompare(b, 'zh-CN'))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeRecords, globalFilter.school, filterOptionsEpoch])

  /** 实验组别列表：随学校/班级级联，仅含 A/B/未归组中实际存在的项 */
  const groupOptions = useMemo(() => {
    const set = new Set<ExperimentGroupId>()
    activeRecords
      .filter((record) => globalFilter.school === 'all' || normalizeSchool(record.school) === globalFilter.school)
      .filter(
        (record) =>
          globalFilter.class === 'all' || normalizeClassGroup(record.classGroup) === globalFilter.class,
      )
      .forEach((record) => set.add(resolveGroup(record)))
    return GROUP_ORDER.filter((g) => set.has(g))
  }, [activeRecords, globalFilter.school, globalFilter.class])

  /**
   * 核心数据过滤管道：全量活跃记录 → school → class → group → dateRange 漏斗。
   * 下游花名册 / 进度图 / 对比雷达均只消费此数据集。
   * 已从花名册/控制台删除（隐藏）的班级不再进入看板。
   */
  const filteredDataset = useMemo(() => {
    const hiddenClasses = new Set(loadHiddenClassGroupNames())
    return activeRecords
      .filter((record) => globalFilter.school === 'all' || normalizeSchool(record.school) === globalFilter.school)
      .filter(
        (record) =>
          globalFilter.class === 'all' || normalizeClassGroup(record.classGroup) === globalFilter.class,
      )
      .filter((record) => !hiddenClasses.has(normalizeClassGroup(record.classGroup)))
      .filter((record) => globalFilter.group === 'all' || resolveGroup(record) === globalFilter.group)
      .filter((record) => isDateInRange(getRecordTestDate(record), globalFilter.dateRange))
      .sort((a, b) => (a.timestamp < b.timestamp ? 1 : -1))
    // filterOptionsEpoch：花名册删除班级后强制重算
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeRecords, globalFilter, filterOptionsEpoch])

  /** 班级对比模块：仅从过滤后数据集提取班级选项，保证与全局过滤器联动 */
  const cohortOptions = useMemo(() => {
    const set = new Set<string>()
    filteredDataset.forEach((record) => set.add(normalizeClassGroup(record.classGroup)))
    return Array.from(set).sort((a, b) => a.localeCompare(b, 'zh-CN'))
  }, [filteredDataset])

  const hasActiveDateRange = Boolean(globalFilter.dateRange[0] || globalFilter.dateRange[1])
  const hasActiveFilter =
    globalFilter.school !== 'all' ||
    globalFilter.class !== 'all' ||
    globalFilter.group !== 'all' ||
    hasActiveDateRange

  // 筛选值一旦不再存在于动态选项中，立即回退，避免幽灵选中态
  useEffect(() => {
    if (globalFilter.school !== 'all' && !schoolOptions.includes(globalFilter.school)) {
      setGlobalFilter((prev) => ({ ...prev, school: 'all', class: 'all', group: 'all' }))
    }
  }, [globalFilter.school, schoolOptions])

  useEffect(() => {
    if (globalFilter.class !== 'all' && !classGroupOptions.includes(globalFilter.class)) {
      setGlobalFilter((prev) => ({ ...prev, class: 'all', group: 'all' }))
    }
  }, [globalFilter.class, classGroupOptions])

  useEffect(() => {
    if (globalFilter.group !== 'all' && !groupOptions.includes(globalFilter.group as ExperimentGroupId)) {
      setGlobalFilter((prev) => ({ ...prev, group: 'all' }))
    }
  }, [globalFilter.group, groupOptions])

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

  /** KPI 与花名册同源：仅反映全局过滤器穿透后的数据集 */
  const kpi = useMemo(() => {
    const uniqueStudents = new Set(
      filteredDataset.map((r) => `${normalizeSchool(r.school)}__${normalizeClassGroup(r.classGroup)}__${r.studentId}`),
    )
    const realtimeStudents = new Set(
      filteredDataset
        .filter((r) => resolveGroup(r) === 'GROUP_A')
        .map((r) => `${normalizeSchool(r.school)}__${normalizeClassGroup(r.classGroup)}__${r.studentId}`),
    )
    const delayedStudents = new Set(
      filteredDataset
        .filter((r) => resolveGroup(r) === 'GROUP_B')
        .map((r) => `${normalizeSchool(r.school)}__${normalizeClassGroup(r.classGroup)}__${r.studentId}`),
    )
    const validScores = filteredDataset.filter((r) => typeof r.score === 'number') as (GlobalTrainingRecord & {
      score: number
    })[]
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
  }, [filteredDataset])

  const studentAggregates: StudentAggregate[] = useMemo(() => {
    const map = new Map<string, StudentAggregate>()
    filteredDataset.forEach((record) => {
      const school = normalizeSchool(record.school)
      const classGroup = normalizeClassGroup(record.classGroup)
      const studentId = (record.studentId || '').trim() || '未填写编号'
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
    return Array.from(map.values()).sort((a, b) => a.studentId.localeCompare(b.studentId, 'zh-CN'))
  }, [filteredDataset])

  /** 班级 → 实验组 (A/B) 树状分组，供左侧手风琴花名册消费 */
  const rosterTree: RosterClassNode[] = useMemo(() => {
    const byClass = new Map<string, Map<ExperimentGroupId, StudentAggregate[]>>()
    studentAggregates.forEach((student) => {
      if (!byClass.has(student.classGroup)) {
        byClass.set(student.classGroup, new Map())
      }
      const groupMap = byClass.get(student.classGroup)!
      if (!groupMap.has(student.group)) groupMap.set(student.group, [])
      groupMap.get(student.group)!.push(student)
    })

    return Array.from(byClass.entries())
      .sort(([a], [b]) => a.localeCompare(b, 'zh-CN'))
      .map(([classGroup, groupMap]) => {
        const groups: RosterGroupNode[] = GROUP_ORDER.filter((g) => groupMap.has(g)).map((group) => ({
          group,
          label: GROUP_META[group].label,
          badge: GROUP_META[group].badge,
          students: (groupMap.get(group) || []).slice().sort((a, b) => a.studentId.localeCompare(b.studentId, 'zh-CN')),
        }))
        return {
          classGroup,
          groups,
          studentCount: groups.reduce((sum, g) => sum + g.students.length, 0),
        }
      })
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

  // 拉取纵向科研节点历史；dataEpoch 变化时强制重拉（洗净后曲线自愈）
  useEffect(() => {
    if (!selectedStudent) {
      setProgressHistory(null)
      setRadarAverage(null)
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
    // 强制与后端对齐，避免 localStorage / 花名册残留幽灵学员
    void fetchAllRecords()
  }

  function handleRecordsBatchDeleted(recordIds: string[]) {
    if (!Array.isArray(recordIds) || recordIds.length === 0) return
    const idSet = new Set(recordIds)
    setRecords((prev) => {
      const next = prev.filter((r) => !idSet.has(r.id))
      saveGlobalRecordsToLocalStorage(next)
      return next
    })
    setDataEpoch((n) => n + 1)
    void fetchAllRecords()
  }

  /** 将后端 radar_average 安全转为雷达组件可用结构 */
  const portraitRadarScores: Quantified5dScores | null = (() => {
    if (!radarAverage || typeof radarAverage !== 'object') return null
    const keys = [
      'approach_rhythm',
      'support_stability',
      'backswing_folding',
      'ankle_rigidity',
      'whipping_velocity',
    ] as const
    const out: Quantified5dScores = {}
    let hit = 0
    for (const key of keys) {
      const v = radarAverage[key]
      if (typeof v === 'number' && Number.isFinite(v)) {
        out[key] = v
        hit += 1
      }
    }
    return hit > 0 ? out : null
  })()

  const panelScrollClass =
    'coach-panel-scroll min-h-0 overflow-y-auto overflow-x-hidden scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent'

  const isIndividualTab = activeTab === 'individual'

  return (
    <div className="coach-dashboard-shell flex h-full min-h-0 flex-col overflow-hidden">
      {/* ============================ 固定头部：KPI + 全局控制台 + Tab 栏 ============================ */}
      <header className="flex-shrink-0 space-y-3 border-b border-white/5 px-3 py-3 sm:px-4">
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

        {/* CTMS 全局控制台：统管 school / class / group / dateRange */}
        <div className="coach-global-control-panel z-20 flex flex-wrap items-center gap-2 rounded-2xl border border-white/10 bg-slate-950/80 px-3 py-2.5 shadow-[0_8px_32px_rgba(0,0,0,0.35)] backdrop-blur-xl">
          <span className="mr-1 inline-flex items-center gap-1.5 text-[11px] font-semibold tracking-wide text-emerald-300/90">
            <Filter className="h-3.5 w-3.5" />
            全局控制台
          </span>

          <FilterOptionPicker
            icon={SchoolIcon}
            iconClassName="text-emerald-400"
            allLabel="全部学校"
            value={globalFilter.school}
            options={schoolOptions}
            onSelect={(school) => {
              setGlobalFilter((prev) => ({
                ...prev,
                school,
                class: 'all',
                group: 'all',
              }))
            }}
            onDelete={(school, event) => {
              event.preventDefault()
              event.stopPropagation()
              removeSchoolOption(school)
              setFilterOptionsEpoch((n) => n + 1)
              setGlobalFilter((prev) =>
                prev.school === school
                  ? { ...prev, school: 'all', class: 'all', group: 'all' }
                  : prev,
              )
            }}
          />

          <FilterOptionPicker
            icon={Layers}
            iconClassName="text-sky-400"
            allLabel="全部班级"
            value={globalFilter.class}
            options={classGroupOptions}
            onSelect={(classGroup) => {
              setGlobalFilter((prev) => ({
                ...prev,
                class: classGroup,
                group: 'all',
              }))
            }}
            onDelete={(classGroup, event) => {
              event.preventDefault()
              event.stopPropagation()
              removeClassGroupOption(classGroup)
              setFilterOptionsEpoch((n) => n + 1)
              setGlobalFilter((prev) =>
                prev.class === classGroup ? { ...prev, class: 'all', group: 'all' } : prev,
              )
            }}
          />

          <label className="flex items-center gap-2 rounded-2xl bg-black/35 px-3 py-1.5 text-xs text-white/50">
            <FlaskConical className="h-3.5 w-3.5 text-violet-300" />
            <select
              value={globalFilter.group}
              onChange={(e) =>
                setGlobalFilter((prev) => ({
                  ...prev,
                  group: e.target.value,
                }))
              }
              className="bg-transparent text-sm font-medium text-white outline-none [&>option]:bg-zinc-900"
            >
              <option value="all">全部组别</option>
              {groupOptions.map((option) => (
                <option key={option} value={option}>
                  {GROUP_META[option].shortLabel}
                </option>
              ))}
            </select>
          </label>

          <label className="flex items-center gap-2 rounded-2xl bg-black/35 px-3 py-1.5 text-xs text-white/50">
            <CalendarDays className="h-3.5 w-3.5 text-amber-400" />
            <span className="whitespace-nowrap text-white/40">起</span>
            <input
              type="date"
              value={globalFilter.dateRange[0] ?? ''}
              onChange={(e) => {
                const start = e.target.value || null
                setGlobalFilter((prev) => {
                  const end = prev.dateRange[1]
                  const nextEnd = start && end && end < start ? start : end
                  return { ...prev, dateRange: [start, nextEnd] }
                })
              }}
              className="bg-transparent text-sm font-medium text-white outline-none [color-scheme:dark]"
            />
            <span className="text-white/30">→</span>
            <span className="whitespace-nowrap text-white/40">止</span>
            <input
              type="date"
              value={globalFilter.dateRange[1] ?? ''}
              min={globalFilter.dateRange[0] ?? undefined}
              onChange={(e) => {
                const end = e.target.value || null
                setGlobalFilter((prev) => ({
                  ...prev,
                  dateRange: [prev.dateRange[0], end],
                }))
              }}
              className="bg-transparent text-sm font-medium text-white outline-none [color-scheme:dark]"
            />
          </label>

          {hasActiveFilter && (
            <button
              type="button"
              onClick={() => setGlobalFilter(DEFAULT_GLOBAL_FILTER)}
              className="inline-flex items-center gap-1 rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[11px] text-white/55 transition hover:bg-white/10 hover:text-white/80"
              title="重置全局过滤器"
            >
              <RotateCcw className="h-3 w-3" />
              重置
            </button>
          )}

          <span className="ml-auto text-[11px] text-white/30">
            有效样本 {filteredDataset.length} 条 · 花名册 {studentAggregates.length} 人 · {rosterTree.length} 个班级
          </span>
        </div>

        {/* Tab 切换栏（仅导航，内容在下方动态区互斥渲染） */}
        <div
          className="flex flex-wrap items-center gap-1 rounded-2xl border border-white/10 bg-black/30 p-1"
          role="tablist"
          aria-label="教练端分析视图"
        >
          {COACH_ANALYSIS_TABS.map((tab) => {
            const isActive = activeTab === tab.id
            const Icon = TAB_ICONS[tab.id]
            return (
              <button
                key={tab.id}
                type="button"
                role="tab"
                aria-selected={isActive}
                onClick={() => setActiveTab(tab.id)}
                className={`inline-flex items-center gap-1.5 rounded-xl px-3 py-2 text-[11px] font-semibold transition ${
                  isActive
                    ? 'bg-emerald-500/20 text-emerald-100 shadow-[0_0_18px_rgba(16,185,129,0.15)] ring-1 ring-emerald-400/35'
                    : 'text-white/45 hover:bg-white/5 hover:text-white/75'
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                {tab.label}
              </button>
            )
          })}
        </div>
      </header>

      {/* ============================ 动态渲染区：按 activeTab 严格互斥 ============================ */}
      <div
        className={`flex min-h-0 flex-1 flex-col p-4 ${
          isIndividualTab ? 'overflow-hidden' : 'overflow-y-auto'
        }`}
      >
        {loadState === 'loading' ? (
          <div className="flex flex-1 items-center justify-center gap-3 text-white/40">
            <Loader2 className="h-7 w-7 animate-spin text-emerald-400" />
            <p className="text-sm">正在加载全量历史归档数据……</p>
          </div>
        ) : activeRecords.length === 0 ? (
          <div className="flex flex-1 items-center justify-center">
            <EmptyStateCard />
          </div>
        ) : filteredDataset.length === 0 ? (
          <div className="flex flex-1 items-center justify-center">
            <FilterEmptyState onReset={() => setGlobalFilter(DEFAULT_GLOBAL_FILTER)} />
          </div>
        ) : activeTab === 'classProfile' ? (
          /* 班级群体画像：仅 2×2 图表面板 */
          <ClassProfilingPanel
            filteredDataset={filteredDataset}
            filterSchool={globalFilter.school}
            filterClass={globalFilter.class}
          />
        ) : activeTab === 'cohortCompare' ? (
          /* 班级对比：多维数据对比面板，隐藏花名册与尝试日志 */
          <CohortComparePanel
            cohortOptions={cohortOptions}
            filteredDataset={filteredDataset}
          />
        ) : activeTab === 'dosage' ? (
          /* 实验干预剂量：全屏热力图 */
          <InterventionDosageMonitor filteredDataset={filteredDataset} />
        ) : (
          /* 个体复盘：左花名册 + 右（个人图谱 / 可展开详细数据） */
          <div className="flex min-h-0 flex-1 flex-row gap-4 overflow-hidden">
            <aside
              className={`w-80 flex-shrink-0 rounded-2xl border border-white/10 bg-slate-900/40 p-3 ${panelScrollClass}`}
            >
              <LeftSidebar
                rosterTree={rosterTree}
                filteredDataset={filteredDataset}
                selectedKey={selectedStudent?.key ?? null}
                onSelect={(student) => {
                  setSelectedStudent(student)
                  setDetailPanelOpen(true)
                }}
                onDeleteClass={(classGroup) => {
                  const confirmed = window.confirm(
                    `确定从花名册删除班级「${classGroup}」？\n删除后该班级将从看板隐藏（数据仍保留在归档中）。`,
                  )
                  if (!confirmed) return
                  removeClassGroupOption(classGroup)
                  setFilterOptionsEpoch((n) => n + 1)
                  setGlobalFilter((prev) =>
                    prev.class === classGroup ? { ...prev, class: 'all', group: 'all' } : prev,
                  )
                  setSelectedStudent((prev) => (prev?.classGroup === classGroup ? null : prev))
                  showToast(`已删除班级「${classGroup}」`, true)
                }}
              />
            </aside>

            <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-4 overflow-hidden">
              <main
                className={`min-h-0 flex-1 rounded-2xl border border-white/10 bg-slate-900/30 p-4 ${panelScrollClass}`}
              >
                <MainCanvas
                  selectedStudent={selectedStudent}
                  filteredDataset={filteredDataset}
                  selectedAttemptIndex={clampedAttemptIndex}
                  onSelectAttemptIndex={setSelectedAttemptIndex}
                  progressHistory={progressHistory}
                  radarScores={portraitRadarScores}
                  dataEpoch={dataEpoch}
                />
              </main>

              {/* 个人详细数据：默认最小化，选中被试或点击后展开 */}
              {!detailPanelOpen ? (
                <button
                  type="button"
                  onClick={() => setDetailPanelOpen(true)}
                  className="group flex h-11 flex-shrink-0 items-center justify-between gap-3 rounded-2xl border border-white/10 bg-slate-900/50 px-3.5 text-left transition hover:border-sky-400/35 hover:bg-sky-500/10"
                >
                  <span className="inline-flex items-center gap-2 text-sm font-medium text-white/70 group-hover:text-sky-100">
                    <span className="inline-flex h-7 w-7 items-center justify-center rounded-xl bg-sky-500/15 ring-1 ring-sky-400/30">
                      <ClipboardList className="h-3.5 w-3.5 text-sky-300" />
                    </span>
                    个人详细数据
                    {selectedStudent ? (
                      <span className="rounded-md bg-white/8 px-1.5 py-0.5 text-[10px] text-white/40">
                        {selectedStudent.studentId} · {selectedStudent.records.length} 次
                      </span>
                    ) : (
                      <span className="text-[11px] font-normal text-white/30">先选左侧被试</span>
                    )}
                  </span>
                  <span className="inline-flex items-center gap-1 text-[11px] text-white/35 group-hover:text-sky-200/80">
                    点击查看
                    <ChevronsUpDown className="h-3.5 w-3.5" />
                  </span>
                </button>
              ) : (
                <aside className="flex h-[min(42vh,360px)] min-h-[240px] flex-shrink-0 flex-col overflow-hidden rounded-2xl border border-white/10 bg-slate-900/40 p-2">
                  <div className="mb-1 flex flex-shrink-0 items-center justify-end px-1">
                    <button
                      type="button"
                      onClick={() => setDetailPanelOpen(false)}
                      className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] text-white/40 transition hover:bg-white/10 hover:text-white/80"
                      title="最小化个人详细数据"
                    >
                      最小化
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                  <CoachDataSanitizerGrid
                    compact
                    studentId={selectedStudent?.studentId ?? null}
                    refreshKey={dataEpoch}
                    onRecordSoftDeleted={handleRecordSoftDeleted}
                    onRecordsBatchDeleted={handleRecordsBatchDeleted}
                    onRadarAverageChange={setRadarAverage}
                    showToast={showToast}
                    className="h-full min-h-0"
                  />
                </aside>
              )}
            </div>
          </div>
        )}
      </div>

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

/** 全局控制台学校/班级下拉：支持选中 + 右侧删除（隐藏）幽灵选项 */
function FilterOptionPicker({
  icon: Icon,
  iconClassName,
  allLabel,
  value,
  options,
  onSelect,
  onDelete,
}: {
  icon: typeof SchoolIcon
  iconClassName: string
  allLabel: string
  value: string
  options: string[]
  onSelect: (value: string) => void
  onDelete: (name: string, event: ReactMouseEvent) => void
}) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const displayLabel = value === 'all' ? allLabel : value

  useEffect(() => {
    if (!open) return
    function handleOutside(event: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleOutside)
    return () => document.removeEventListener('mousedown', handleOutside)
  }, [open])

  return (
    <div
      ref={rootRef}
      className="relative flex min-w-[140px] items-center gap-2 rounded-2xl bg-black/35 px-3 py-1.5 text-xs text-white/50"
    >
      <Icon className={`h-3.5 w-3.5 flex-shrink-0 ${iconClassName}`} />
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className="flex min-w-0 flex-1 items-center justify-between gap-1 bg-transparent text-left text-sm font-medium text-white outline-none"
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className="truncate">{displayLabel}</span>
        <ChevronDown
          className={`h-3.5 w-3.5 flex-shrink-0 text-white/35 transition-transform ${open ? 'rotate-180' : ''}`}
        />
      </button>

      <AnimatePresence>
        {open && (
          <motion.ul
            initial={{ opacity: 0, y: -6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.98 }}
            transition={{ type: 'spring', stiffness: 420, damping: 32 }}
            className="absolute left-0 right-0 top-[calc(100%+6px)] z-40 max-h-52 min-w-[180px] overflow-y-auto rounded-2xl border border-white/10 bg-zinc-900/95 py-1 shadow-xl backdrop-blur-xl"
            role="listbox"
          >
            <li
              role="option"
              aria-selected={value === 'all'}
              className={`cursor-pointer px-3 py-2 text-sm transition hover:bg-white/10 ${
                value === 'all' ? 'text-emerald-300' : 'text-white/40'
              }`}
              onClick={() => {
                onSelect('all')
                setOpen(false)
              }}
            >
              {allLabel}
            </li>
            {options.map((option) => {
              const selected = option === value
              return (
                <li
                  key={option}
                  role="option"
                  aria-selected={selected}
                  className={`group flex items-center justify-between gap-2 px-3 py-2 text-sm transition hover:bg-white/10 ${
                    selected ? 'text-emerald-300' : 'text-white/85'
                  }`}
                  onClick={() => {
                    onSelect(option)
                    setOpen(false)
                  }}
                >
                  <span className="min-w-0 truncate">{option}</span>
                  <button
                    type="button"
                    title={`删除「${option}」选项`}
                    aria-label={`删除 ${option}`}
                    onClick={(e) => onDelete(option, e)}
                    className="inline-flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-md text-white/35 opacity-70 transition hover:bg-rose-500/20 hover:text-rose-300 group-hover:opacity-100"
                  >
                    <X className="h-3.5 w-3.5" strokeWidth={2.25} />
                  </button>
                </li>
              )
            })}
            {options.length === 0 && (
              <li className="px-3 py-2 text-xs text-white/35">暂无可选项</li>
            )}
          </motion.ul>
        )}
      </AnimatePresence>
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
        前往「实时反馈系统」或「延时反馈系统」完成测试并开启本地落盘归档后，数据将同步至教练端看板。
      </p>
    </motion.div>
  )
}

function FilterEmptyState({ onReset }: { onReset: () => void }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex max-w-lg flex-col items-center gap-4 rounded-3xl border border-dashed border-amber-400/25 bg-gradient-to-br from-amber-500/10 via-white/[0.02] to-transparent p-10 text-center"
    >
      <div className="flex h-16 w-16 items-center justify-center rounded-[24px] bg-gradient-to-br from-amber-400/20 to-emerald-500/15 ring-1 ring-white/10">
        <Filter className="h-8 w-8 text-amber-300" />
      </div>
      <h3 className="text-lg font-semibold text-white/90">该筛选条件下暂无有效测试数据</h3>
      <p className="text-sm leading-relaxed text-white/40">
        请调整右上角的全局过滤器（学校 / 班级 / 组别 / 日期区间），或重置后重新查看全量样本。
      </p>
      <button
        type="button"
        onClick={onReset}
        className="inline-flex items-center gap-1.5 rounded-full border border-amber-400/35 bg-amber-500/15 px-4 py-2 text-xs font-semibold text-amber-100 transition hover:bg-amber-500/25"
      >
        <RotateCcw className="h-3.5 w-3.5" />
        重置全局过滤器
      </button>
    </motion.div>
  )
}

function matchesRosterQuery(student: StudentAggregate, query: string): boolean {
  if (!query) return true
  const haystack = `${student.studentId} ${student.school} ${student.classGroup}`.toLowerCase()
  return haystack.includes(query)
}

function LeftSidebar({
  rosterTree,
  filteredDataset,
  selectedKey,
  onSelect,
  onDeleteClass,
}: {
  rosterTree: RosterClassNode[]
  /** 全局过滤后的唯一数据源（剂量统计与花名册树均由此衍生） */
  filteredDataset: GlobalTrainingRecord[]
  selectedKey: string | null
  onSelect: (student: StudentAggregate) => void
  onDeleteClass: (classGroup: string) => void
}) {
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedClasses, setExpandedClasses] = useState<Set<string>>(new Set())
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set())

  const normalizedQuery = searchQuery.trim().toLowerCase()
  const isSearching = normalizedQuery.length > 0
  const attemptCount = filteredDataset.length

  const filteredTree = useMemo(() => {
    if (!isSearching) return rosterTree
    return rosterTree
      .map((classNode) => {
        const groups = classNode.groups
          .map((groupNode) => ({
            ...groupNode,
            students: groupNode.students.filter((s) => matchesRosterQuery(s, normalizedQuery)),
          }))
          .filter((g) => g.students.length > 0)
        return {
          ...classNode,
          groups,
          studentCount: groups.reduce((sum, g) => sum + g.students.length, 0),
        }
      })
      .filter((c) => c.studentCount > 0)
  }, [rosterTree, isSearching, normalizedQuery])

  // 树结构变化时：默认展开首个班级；搜索时自动展开全部命中节点
  useEffect(() => {
    if (isSearching) {
      setExpandedClasses(new Set(filteredTree.map((c) => c.classGroup)))
      setExpandedGroups(
        new Set(filteredTree.flatMap((c) => c.groups.map((g) => `${c.classGroup}__${g.group}`))),
      )
      return
    }
    setExpandedClasses((prev) => {
      const valid = new Set(rosterTree.map((c) => c.classGroup))
      const next = new Set([...prev].filter((key) => valid.has(key)))
      if (next.size === 0 && rosterTree[0]) next.add(rosterTree[0].classGroup)
      return next
    })
    setExpandedGroups((prev) => {
      const valid = new Set(rosterTree.flatMap((c) => c.groups.map((g) => `${c.classGroup}__${g.group}`)))
      return new Set([...prev].filter((key) => valid.has(key)))
    })
  }, [rosterTree, filteredTree, isSearching])

  // 选中被试所属班级/组别自动展开，便于回看定位
  useEffect(() => {
    if (!selectedKey) return
    for (const classNode of rosterTree) {
      for (const groupNode of classNode.groups) {
        if (groupNode.students.some((s) => s.key === selectedKey)) {
          setExpandedClasses((prev) => new Set(prev).add(classNode.classGroup))
          setExpandedGroups((prev) => new Set(prev).add(`${classNode.classGroup}__${groupNode.group}`))
          return
        }
      }
    }
  }, [selectedKey, rosterTree])

  const total = filteredTree.reduce((sum, c) => sum + c.studentCount, 0)

  function toggleClass(classGroup: string) {
    if (isSearching) return
    setExpandedClasses((prev) => {
      const next = new Set(prev)
      if (next.has(classGroup)) next.delete(classGroup)
      else next.add(classGroup)
      return next
    })
  }

  function toggleGroup(classGroup: string, group: ExperimentGroupId) {
    if (isSearching) return
    const key = `${classGroup}__${group}`
    setExpandedGroups((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  return (
    <div className="花名册列表 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-white/85">
          <Users className="h-4 w-4 text-emerald-300" />
          花名册
        </h3>
        <span className="rounded-full bg-black/30 px-2 py-0.5 text-[10px] text-white/40">
          {total} 人 · {attemptCount} 次
        </span>
      </div>

      <label className="flex items-center gap-2 rounded-xl border border-white/10 bg-black/30 px-2.5 py-2">
        <Search className="h-3.5 w-3.5 flex-shrink-0 text-white/35" />
        <input
          type="search"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="搜索学号 / 姓名 / 班级…"
          className="min-w-0 flex-1 bg-transparent text-xs text-white/85 outline-none placeholder:text-white/30"
        />
      </label>

      {total === 0 ? (
        <p className="rounded-2xl border border-dashed border-white/10 bg-black/20 px-3 py-8 text-center text-xs text-white/35">
          {isSearching ? '未找到匹配的学号或班级' : '当前筛选条件下暂无被试'}
        </p>
      ) : (
        <div className="flex flex-col gap-1.5">
          {filteredTree.map((classNode) => {
            const classOpen = expandedClasses.has(classNode.classGroup)
            return (
              <div
                key={classNode.classGroup}
                className="overflow-hidden rounded-xl border border-white/8 bg-black/20"
              >
                <div className="group/class flex w-full items-center gap-1 px-1.5 py-1.5">
                  <button
                    type="button"
                    onClick={() => toggleClass(classNode.classGroup)}
                    className="flex min-w-0 flex-1 items-center gap-2 rounded-lg px-1.5 py-1 text-left transition hover:bg-white/5"
                  >
                    {classOpen ? (
                      <ChevronDown className="h-3.5 w-3.5 flex-shrink-0 text-emerald-300/80" />
                    ) : (
                      <ChevronRight className="h-3.5 w-3.5 flex-shrink-0 text-white/35" />
                    )}
                    <span className="min-w-0 flex-1 truncate text-sm font-semibold text-white/85">
                      {classNode.classGroup}
                    </span>
                    <span className="flex-shrink-0 rounded-md bg-white/8 px-1.5 py-0.5 text-[10px] tabular-nums text-white/40">
                      {classNode.studentCount}人
                    </span>
                  </button>
                  <button
                    type="button"
                    title={`删除班级「${classNode.classGroup}」`}
                    aria-label={`删除班级 ${classNode.classGroup}`}
                    onClick={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                      onDeleteClass(classNode.classGroup)
                    }}
                    className="inline-flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-md text-white/25 opacity-70 transition hover:bg-rose-500/20 hover:text-rose-300 group-hover/class:opacity-100"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>

                <AnimatePresence initial={false}>
                  {classOpen && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.18 }}
                      className="overflow-hidden border-t border-white/5"
                    >
                      <div className="flex flex-col gap-1 p-1.5">
                        {classNode.groups.map((groupNode) => {
                          const groupKey = `${classNode.classGroup}__${groupNode.group}`
                          const groupOpen = expandedGroups.has(groupKey)
                          return (
                            <div key={groupKey} className="rounded-lg bg-white/[0.02]">
                              <button
                                type="button"
                                onClick={() => toggleGroup(classNode.classGroup, groupNode.group)}
                                className="flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left transition hover:bg-white/5"
                              >
                                {groupOpen ? (
                                  <ChevronDown className="h-3 w-3 flex-shrink-0 text-white/40" />
                                ) : (
                                  <ChevronRight className="h-3 w-3 flex-shrink-0 text-white/25" />
                                )}
                                <span
                                  className={`flex h-5 w-5 items-center justify-center rounded-md text-[10px] font-bold ${
                                    groupNode.group === 'GROUP_A'
                                      ? 'bg-sky-500/20 text-sky-300'
                                      : groupNode.group === 'GROUP_B'
                                        ? 'bg-teal-500/20 text-teal-300'
                                        : 'bg-white/10 text-white/50'
                                  }`}
                                >
                                  {groupNode.badge}
                                </span>
                                <span className="min-w-0 flex-1 truncate text-[11px] font-medium text-white/55">
                                  {groupNode.label}
                                </span>
                                <span className="text-[10px] text-white/25">{groupNode.students.length}</span>
                              </button>

                              <AnimatePresence initial={false}>
                                {groupOpen && (
                                  <motion.div
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: 'auto', opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    transition={{ duration: 0.16 }}
                                    className="overflow-hidden"
                                  >
                                    <div className="flex flex-col gap-1 px-1.5 pb-1.5">
                                      {groupNode.students.map((student) => {
                                        const dose = student.records.length
                                        const underdosed = dose < DOSE_WARNING_THRESHOLD
                                        const isActive = student.key === selectedKey
                                        return (
                                          <button
                                            key={student.key}
                                            type="button"
                                            onClick={() => onSelect(student)}
                                            className={`flex items-center justify-between rounded-lg border px-2.5 py-2 text-left transition ${
                                              isActive
                                                ? 'border-emerald-400/45 bg-emerald-400/15 text-white shadow-[0_0_20px_rgba(16,185,129,0.12)]'
                                                : 'border-white/5 bg-black/25 text-white/65 hover:border-white/15 hover:bg-white/8'
                                            }`}
                                          >
                                            <span className="min-w-0">
                                              <span className="block truncate text-sm font-medium">
                                                {student.studentId}
                                              </span>
                                              <span className="block truncate text-[10px] text-white/30">
                                                {student.school}
                                              </span>
                                            </span>
                                            <span
                                              className={`ml-2 flex-shrink-0 rounded-md px-1.5 py-0.5 text-[10px] font-bold tabular-nums ${
                                                underdosed
                                                  ? 'bg-rose-500/90 text-white shadow-[0_0_10px_rgba(239,68,68,0.35)]'
                                                  : 'bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-400/30'
                                              }`}
                                              title={
                                                underdosed
                                                  ? `已练 ${dose} 次，未达 10 次剂量`
                                                  : `已练 ${dose} 次`
                                              }
                                            >
                                              {dose}次
                                            </span>
                                          </button>
                                        )
                                      })}
                                    </div>
                                  </motion.div>
                                )}
                              </AnimatePresence>
                            </div>
                          )
                        })}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function MainCanvas({
  selectedStudent,
  filteredDataset,
  selectedAttemptIndex,
  onSelectAttemptIndex,
  progressHistory,
  radarScores,
  dataEpoch,
}: {
  selectedStudent: StudentAggregate | null
  /** 全局过滤后的唯一数据源；进度图与雷达仅消费其子集 */
  filteredDataset: GlobalTrainingRecord[]
  selectedAttemptIndex: number
  onSelectAttemptIndex: (index: number) => void
  progressHistory: ProgressHistoryPoint[] | null
  radarScores: GlobalTrainingRecord['quantified5dScores']
  dataEpoch: number
}) {
  // 强制联动：个人图谱只展示落在 filteredDataset 内的尝试
  const scopedRecords = useMemo(() => {
    if (!selectedStudent) return [] as GlobalTrainingRecord[]
    const allow = new Set(filteredDataset.map((r) => r.id))
    return selectedStudent.records.filter((r) => allow.has(r.id))
  }, [selectedStudent, filteredDataset])

  if (!selectedStudent) {
    return (
      <div className="flex h-full min-h-0 flex-col items-center justify-center gap-5 px-6 text-center">
        <div className="relative flex h-24 w-24 items-center justify-center rounded-[28px] bg-gradient-to-br from-emerald-500/15 via-sky-500/10 to-transparent ring-1 ring-white/10">
          <MousePointerClick className="h-10 w-10 text-emerald-300/80" />
          <span className="absolute -left-2 top-1/2 -translate-y-1/2 text-2xl opacity-60">👈</span>
        </div>
        <div className="max-w-md space-y-2">
          <h3 className="text-lg font-semibold text-white/90">
            请在左侧选择被试以查看科研分析图谱
          </h3>
          <p className="text-sm leading-relaxed text-white/40">
            从花名册挑选学生后，可查看个人进步趋势、五维雷达与底部「个人详细数据」；亦可上传数据给 LLM 分析总体情况。
          </p>
        </div>
      </div>
    )
  }

  const latest = scopedRecords[scopedRecords.length - 1]
  const latestScore = latest?.score ?? '--'
  const clampedIndex =
    scopedRecords.length > 0
      ? Math.min(Math.max(0, selectedAttemptIndex), scopedRecords.length - 1)
      : 0

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
            筛选内 {scopedRecords.length} 次 · Attempt #1 → #{Math.max(scopedRecords.length, 1)}
          </p>
        </div>
      </section>

      <section className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
        <h4 className="mb-3 flex items-center gap-2 text-sm font-semibold text-white/80">
          <TrendingUp className="h-4 w-4 text-sky-300" />
          个人进步趋势图 · 真实日期 × 科研节点
        </h4>
        {scopedRecords.length === 0 ? (
          <div className="flex h-56 flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-white/10 bg-black/20 px-4 text-center">
            <Filter className="h-6 w-6 text-white/25" />
            <p className="text-xs text-white/35">
              该筛选条件下暂无有效测试数据，请调整右上角的全局过滤器
            </p>
          </div>
        ) : (
          <LongitudinalProgressChart
            key={`chart-${selectedStudent.key}-${dataEpoch}-${scopedRecords.length}`}
            records={scopedRecords}
            historyPoints={progressHistory}
            selectedIndex={clampedIndex}
            onSelectIndex={onSelectAttemptIndex}
            studentId={selectedStudent.studentId}
          />
        )}
      </section>

      <section className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
        <h4 className="mb-3 flex items-center gap-2 text-sm font-semibold text-white/80">
          <Crosshair className="h-4 w-4 text-emerald-300" />
          综合均值能力画像
        </h4>
        <BiomechanicalRadar
          key={`radar-${selectedStudent.key}-${dataEpoch}`}
          scores={radarScores}
          primaryLabel="综合均值"
        />
      </section>

      <IndividualLlmSummaryCard
        key={`llm-summary-${selectedStudent.key}-${dataEpoch}`}
        studentId={selectedStudent.studentId}
        records={scopedRecords}
      />
    </div>
  )
}

type LlmSummaryStatus = 'idle' | 'loading' | 'ready' | 'error'

/** 从被试归档记录汇总评分序列与错误次数，供 /api/generate_individual_summary 消费 */
function buildIndividualSummaryPayload(records: GlobalTrainingRecord[]): {
  scoreHistory: number[]
  errorCounter: Record<string, number>
} {
  const chronological = records
    .slice()
    .sort((a, b) => (a.timestamp < b.timestamp ? -1 : a.timestamp > b.timestamp ? 1 : 0))
  const scoreHistory = chronological
    .map((r) => r.score)
    .filter((s): s is number => typeof s === 'number' && Number.isFinite(s))
  const errorCounter: Record<string, number> = {}
  for (const record of chronological) {
    for (const err of record.biomechanicalErrors ?? []) {
      const label = String(err).trim()
      if (!label) continue
      errorCounter[label] = (errorCounter[label] || 0) + 1
    }
  }
  return { scoreHistory, errorCounter }
}

/**
 * 个体复盘：将当前筛选内的个人数据上传给 LLM，生成总体优缺点总结。
 * 默认待命；需教练主动点击「上传并分析」。
 */
function IndividualLlmSummaryCard({
  studentId,
  records,
}: {
  studentId: string
  records: GlobalTrainingRecord[]
}) {
  const [status, setStatus] = useState<LlmSummaryStatus>('idle')
  const [report, setReport] = useState<IndividualSummaryReport | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const { scoreHistory, errorCounter } = useMemo(
    () => buildIndividualSummaryPayload(records),
    [records],
  )
  const sampleCount = records.length
  const canAnalyze = scoreHistory.length > 0

  async function handleUploadAndAnalyze() {
    if (!canAnalyze || status === 'loading') return
    setStatus('loading')
    setErrorMessage(null)
    try {
      const response = await fetch(`${API_BASE_URL}/api/generate_individual_summary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          studentId,
          scoreHistory,
          errorCounter,
        }),
      })
      if (!response.ok) {
        throw new Error(`接口返回状态码 ${response.status}`)
      }
      const data = (await response.json()) as IndividualSummaryReport
      if (!data?.strengths && !data?.weaknesses) {
        throw new Error('后端未返回有效分析内容')
      }
      setReport({
        strengths: data.strengths || '',
        weaknesses: data.weaknesses || '',
        generatedAt: data.generatedAt || '',
      })
      setStatus('ready')
    } catch (error) {
      setReport(null)
      setErrorMessage(error instanceof Error ? error.message : '分析失败，请稍后重试')
      setStatus('error')
    }
  }

  const topErrors = Object.entries(errorCounter)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)

  return (
    <section className="rounded-2xl border border-violet-400/20 bg-gradient-to-br from-violet-500/10 via-white/[0.03] to-transparent p-4">
      <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-violet-500/20 ring-1 ring-violet-400/30">
            <Sparkles className="h-4 w-4 text-violet-200" />
          </span>
          <div>
            <h4 className="text-sm font-semibold text-violet-100">LLM 个人总体分析</h4>
            <p className="text-[11px] text-white/35">
              上传筛选内 {sampleCount} 次尝试
              {scoreHistory.length > 0 ? ` · ${scoreHistory.length} 个有效评分` : ''}
              ，生成优势 / 盲区总结
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => void handleUploadAndAnalyze()}
          disabled={!canAnalyze || status === 'loading'}
          className="inline-flex items-center gap-1.5 rounded-full border border-violet-400/35 bg-violet-500/20 px-3.5 py-1.5 text-xs font-semibold text-violet-100 transition hover:bg-violet-500/30 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {status === 'loading' ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Upload className="h-3.5 w-3.5" />
          )}
          {status === 'loading' ? '分析中…' : status === 'ready' ? '重新上传分析' : '上传数据并分析'}
        </button>
      </div>

      {topErrors.length > 0 && status === 'idle' && (
        <p className="mb-3 text-[11px] text-white/30">
          待上传高频偏差：
          {topErrors.map(([label, count]) => `${label}×${count}`).join(' · ')}
        </p>
      )}

      {status === 'idle' && (
        <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-white/10 bg-black/20 px-4 py-8 text-center">
          <Upload className="h-6 w-6 text-violet-300/50" />
          <p className="text-sm text-white/55">
            {canAnalyze
              ? '点击上方按钮，将本生历史评分与错误统计上传给 LLM 分析个人总体情况'
              : '当前筛选内暂无有效评分，无法上传分析'}
          </p>
        </div>
      )}

      {status === 'loading' && (
        <div className="flex flex-col gap-2 rounded-xl border border-violet-400/20 bg-violet-500/5 px-4 py-6">
          <div className="flex items-center gap-2 text-violet-200/90">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-xs font-medium">DeepSeek 正在解读个人纵向表现…</span>
          </div>
          <p className="text-[11px] text-white/30">
            学号 {studentId} · 评分序列 {scoreHistory.join(' → ') || '—'}
          </p>
        </div>
      )}

      {status === 'error' && (
        <div className="flex flex-col gap-3 rounded-xl border border-rose-400/25 bg-rose-500/10 px-4 py-5">
          <div className="flex items-center gap-2 text-rose-300">
            <AlertCircle className="h-4 w-4" />
            <span className="text-xs font-medium">分析失败</span>
          </div>
          <p className="text-sm text-white/60">{errorMessage || '请确认后端服务已启动后重试'}</p>
          <button
            type="button"
            onClick={() => void handleUploadAndAnalyze()}
            className="inline-flex w-fit items-center gap-1.5 rounded-full border border-rose-400/30 bg-rose-500/15 px-3 py-1.5 text-[11px] font-medium text-rose-100 transition hover:bg-rose-500/25"
          >
            <RefreshCcw className="h-3 w-3" />
            重试
          </button>
        </div>
      )}

      {status === 'ready' && report && (
        <div className="flex flex-col gap-3">
          <article className="rounded-xl border border-emerald-400/20 bg-emerald-500/8 px-4 py-3">
            <p className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-emerald-300/80">
              <ThumbsUp className="h-3 w-3" />
              稳定发力优势
            </p>
            <p className="text-sm leading-relaxed text-white/85 whitespace-pre-wrap">{report.strengths}</p>
          </article>
          <article className="rounded-xl border border-amber-400/20 bg-amber-500/8 px-4 py-3">
            <p className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-amber-300/80">
              <Target className="h-3 w-3" />
              习惯性盲区
            </p>
            <p className="text-sm leading-relaxed text-white/85 whitespace-pre-wrap">{report.weaknesses}</p>
          </article>
          {report.generatedAt && (
            <p className="text-right text-[10px] text-white/25">生成于 {report.generatedAt}</p>
          )}
        </div>
      )}
    </section>
  )
}
