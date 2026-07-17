import { useEffect, useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  Users,
  Radio,
  Clock3,
  Gauge,
  RefreshCcw,
  Loader2,
  Inbox,
  Printer,
  FolderOpen,
  Crosshair,
  ScanFace,
  School as SchoolIcon,
  Layers,
  CheckCircle2,
  XCircle,
  Sparkles,
  TrendingUp,
  Award,
  ChevronRight,
  Flame,
  GraduationCap,
  LayoutGrid,
  UserSearch,
  ThumbsUp,
  ShieldAlert,
  Wand2,
  CalendarDays,
  FileSpreadsheet,
  MessageSquareText,
} from 'lucide-react'
import { BIOMECH_ERROR_TAXONOMY, loadGlobalRecordsFromLocalStorage, saveGlobalRecordsToLocalStorage } from '../mockData'
import type {
  AcademicExportResult,
  BiomechFaultStat,
  ClassPrescriptionReport,
  CoachDashboardPerspective,
  GlobalTrainingRecord,
  IndividualSummaryReport,
} from '../types'

/** 【v4.0 新增】科研级膝角学术合规发力区间：135°-155°（用于双轴成长期刊图绿色阴影带） */
const OPTIMAL_KNEE_ANGLE_MIN = 135
const OPTIMAL_KNEE_ANGLE_MAX = 155

/** 从记录中安全提取 "YYYY-MM-DD" 测试日期：优先使用后端回填的 testDate 字段，
 * 历史旧记录缺失时退化为从 timestamp 字符串前 10 位截取。 */
function getRecordTestDate(record: GlobalTrainingRecord): string {
  if (record.testDate && record.testDate.length >= 10) return record.testDate
  return (record.timestamp || '').slice(0, 10) || '未知日期'
}

/** 从 "YYYY-MM-DD HH:mm:ss" 时间戳字符串中提取 "HH:mm" 时间段，供时空胶囊标签展示 */
function getRecordTimeLabel(record: GlobalTrainingRecord): string {
  const parts = (record.timestamp || '').split(' ')
  if (parts.length >= 2) return parts[1].slice(0, 5)
  return '--:--'
}

/* ============================================================================
 * 后台服务网关地址，与 RealtimeWorkspace.tsx / ZenWorkspace.tsx 保持完全一致。
 * ========================================================================== */
const API_BASE_URL = 'http://localhost:8000'

const SCHOOL_FALLBACK = '未设置学校'
const CLASS_FALLBACK = '未设置班级'

/** 右上角浮动 Toast 提示条状态 */
interface DashboardToastState {
  id: number
  message: string
  success: boolean
}

let dashboardToastSeq = 0

/** 数据加载状态：正在加载 / 已就绪（无论最终是否为空数据） */
type LoadState = 'loading' | 'ready'

/** 某一位学生的聚合视图数据：分组 key + 展示用编号 + 全部历史记录（按时间升序） */
interface StudentAggregate {
  key: string
  studentId: string
  school: string
  classGroup: string
  records: GlobalTrainingRecord[]
}

/**
 * 教练端数据看板（v3.0 科研指挥中心）：
 *
 * 数据源读取策略：组件加载时优先请求后端 GET /api/get_all_records 拉取全量历史
 * 归档数据；请求失败（例如后端服务未启动）时自动回退读取 localStorage 里的
 * 极速双保险缓存，两边取"更完整"的一份数据展示，绝不会因为后端离线就出现
 * "看板看不到"的痛点。
 *
 * 核心结构：① 学校 -> 班级 三级级联精准筛选器；② 视角切换胶囊（全班集体宏观
 * 诊断 / 个体纵向进化追踪）；③ 集体诊断模式下渲染生物力学错误热力图 +
 * AIGC 全班教学处方；④ 个体追踪模式下渲染纵向成长曲线 + AI 优缺点总结 +
 * 最佳关键帧 + 打印/归档操作。
 */
export default function CoachDashboard() {
  const [records, setRecords] = useState<GlobalTrainingRecord[]>([])
  const [loadState, setLoadState] = useState<LoadState>('loading')
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [toast, setToast] = useState<DashboardToastState | null>(null)

  /* ---------------------------- 三级级联精准筛选器状态 ---------------------------- */
  const [selectedSchool, setSelectedSchool] = useState<string>('all')
  const [selectedClassGroup, setSelectedClassGroup] = useState<string>('all')
  /** 【v4.0 新增】📅 动态日期级联筛选器：'all' 表示全部日期，否则为具体 "YYYY-MM-DD" */
  const [selectedTestDate, setSelectedTestDate] = useState<string>('all')
  const [perspective, setPerspective] = useState<CoachDashboardPerspective>('classOverview')

  /* ---------------------------- 📥 学术统计矩阵一键导出状态 ---------------------------- */
  const [isExportingMatrix, setIsExportingMatrix] = useState(false)

  /* ---------------------------- ⏱️ 时空胶囊尝试时间轴：当前选中的尝试序号（从 0 开始） ---------------------------- */
  const [selectedAttemptIndex, setSelectedAttemptIndex] = useState(0)

  /* ---------------------------- 全班 AIGC 教学处方状态 ---------------------------- */
  const [classPrescription, setClassPrescription] = useState<ClassPrescriptionReport | null>(null)
  const [isPrescriptionLoading, setIsPrescriptionLoading] = useState(false)

  /* ---------------------------- 个体纵向进化追踪状态 ---------------------------- */
  const [selectedStudentKey, setSelectedStudentKey] = useState<string | null>(null)
  const [individualSummaries, setIndividualSummaries] = useState<Record<string, IndividualSummaryReport>>({})
  const [loadingSummaryKey, setLoadingSummaryKey] = useState<string | null>(null)

  function showToast(message: string, success: boolean) {
    const id = ++dashboardToastSeq
    setToast({ id, message, success })
    window.setTimeout(() => {
      setToast((current) => (current?.id === id ? null : current))
    }, 3200)
  }

  /** 拉取全量历史归档数据：后端优先，localStorage 兜底/合并，两边按 id 去重合并展示 */
  async function fetchAllRecords(isManualRefresh = false) {
    if (isManualRefresh) setIsRefreshing(true)
    const localRecords = loadGlobalRecordsFromLocalStorage()
    try {
      const response = await fetch(`${API_BASE_URL}/api/get_all_records`)
      if (!response.ok) throw new Error(`接口返回状态码 ${response.status}`)
      const data = (await response.json()) as { success: boolean; records?: GlobalTrainingRecord[] }
      const backendRecords = Array.isArray(data.records) ? data.records : []

      // 按 id 去重合并：后端数据库为唯一权威真源，localStorage 仅补充后端可能
      // 还未来得及同步、但浏览器本地已经落盘成功的极新记录。
      const mergedMap = new Map<string, GlobalTrainingRecord>()
      for (const record of localRecords) mergedMap.set(record.id, record)
      for (const record of backendRecords) mergedMap.set(record.id, record)
      const merged = Array.from(mergedMap.values())

      setRecords(merged)
      saveGlobalRecordsToLocalStorage(merged)
      if (isManualRefresh) showToast(`✅ 已刷新，共加载 ${merged.length} 条历史归档记录`, true)
    } catch {
      // 后端服务未启动/网络异常：优雅回退到本地缓存，绝不让看板直接白屏报错
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

  /* ---------------------------- 三级级联筛选器：可选项与筛选结果 ---------------------------- */

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

  // 切换学校后，若原先选中的班级不再属于新学校的可选范围，自动重置为「全部班级」
  useEffect(() => {
    if (selectedClassGroup !== 'all' && !classGroupOptions.includes(selectedClassGroup)) {
      setSelectedClassGroup('all')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSchool, classGroupOptions])

  /** 【v4.0 新增】📅 动态抽取全部去重测试日期序列（最新日期排在最前面） */
  const dateOptions = useMemo(() => {
    const set = new Set<string>()
    records.forEach((record) => set.add(getRecordTestDate(record)))
    return Array.from(set).sort((a, b) => (a < b ? 1 : -1))
  }, [records])

  // 切换学校/班级后，若原先选中的具体日期在新范围内已不存在任何记录，自动重置为「全部日期」
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

  const filterScopeLabel = useMemo(() => {
    const schoolText = selectedSchool === 'all' ? '全部学校' : selectedSchool
    const classText = selectedClassGroup === 'all' ? '全部班级' : selectedClassGroup
    const dateText = selectedTestDate === 'all' ? '全部日期' : selectedTestDate
    return `${schoolText} · ${classText} · ${dateText}`
  }, [selectedSchool, selectedClassGroup, selectedTestDate])

  // 切换筛选范围后，之前生成的全班处方/个体总结上下文已经过期，清空避免误导教练
  useEffect(() => {
    setClassPrescription(null)
  }, [selectedSchool, selectedClassGroup, selectedTestDate])

  /** 📥 一键导出科研论文数据矩阵：调用后端清洗转换 + 落盘，成功后弹出 Apple 风格提示 */
  async function handleExportAcademicMatrix() {
    if (isExportingMatrix) return
    setIsExportingMatrix(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/export_academic_matrix`, { method: 'POST' })
      const data = (await response.json()) as AcademicExportResult
      if (!response.ok || !data.success) throw new Error(data.message || `接口返回状态码 ${response.status}`)
      showToast(
        data.message ||
          `✅ 科研数据矩阵已清洗完毕并数字化编码！文件已存入：${data.path ?? 'academic_data_export/'}`,
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

  /* ---------------------------- 顶栏核心指标矩阵 (KPI Metrics) ---------------------------- */

  const kpi = useMemo(() => {
    const uniqueStudents = new Set(records.map((record) => `${record.school}__${record.classGroup}__${record.studentId}`))
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

  /* ---------------------------- 集体错误热力图：错误分类统计 ---------------------------- */

  const faultStats: BiomechFaultStat[] = useMemo(() => {
    const counter = new Map<string, number>()
    BIOMECH_ERROR_TAXONOMY.forEach((label) => counter.set(label, 0))
    let recordsWithErrorField = 0
    filteredRecords.forEach((record) => {
      const errors = record.biomechanicalErrors
      if (!Array.isArray(errors)) return
      recordsWithErrorField += 1
      errors.forEach((label) => {
        if (counter.has(label)) counter.set(label, (counter.get(label) ?? 0) + 1)
      })
    })
    const denominator = Math.max(1, recordsWithErrorField)
    return BIOMECH_ERROR_TAXONOMY.map((label) => {
      const count = counter.get(label) ?? 0
      return { label, count, percentage: (count / denominator) * 100 }
    }).sort((a, b) => b.percentage - a.percentage)
  }, [filteredRecords])

  const classAvgScore = useMemo(() => {
    const validScores = filteredRecords.filter((r) => typeof r.score === 'number') as (GlobalTrainingRecord & {
      score: number
    })[]
    return validScores.length > 0
      ? Math.round(validScores.reduce((sum, r) => sum + r.score, 0) / validScores.length)
      : null
  }, [filteredRecords])

  /** ✨ 召唤 AI 生成全班改进教案：把当前筛选范围内的错误分布统计交给 DeepSeek */
  async function handleGenerateClassPrescription() {
    if (isPrescriptionLoading) return
    setIsPrescriptionLoading(true)
    try {
      const errorStatsPayload: Record<string, number> = {}
      faultStats.forEach((stat) => {
        errorStatsPayload[stat.label] = Math.round(stat.percentage * 10) / 10
      })
      const response = await fetch(`${API_BASE_URL}/api/generate_class_prescription`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          school: selectedSchool === 'all' ? '' : selectedSchool,
          classGroup: selectedClassGroup === 'all' ? '' : selectedClassGroup,
          errorStats: errorStatsPayload,
          totalRecords: filteredRecords.length,
          avgScore: classAvgScore,
        }),
      })
      if (!response.ok) throw new Error(`接口返回状态码 ${response.status}`)
      const data = (await response.json()) as ClassPrescriptionReport
      setClassPrescription(data)
    } catch (error) {
      showToast(
        `⚠️ 生成全班教学处方失败：${error instanceof Error ? error.message : '请检查后端服务是否已启动'}`,
        false,
      )
    } finally {
      setIsPrescriptionLoading(false)
    }
  }

  /* ---------------------------- 个体纵向进化追踪：按学生分组聚合 ---------------------------- */

  const studentAggregates: StudentAggregate[] = useMemo(() => {
    const map = new Map<string, StudentAggregate>()
    filteredRecords.forEach((record) => {
      const school = record.school || SCHOOL_FALLBACK
      const classGroup = record.classGroup || CLASS_FALLBACK
      const studentId = record.studentId || '未填写编号'
      const key = `${school}__${classGroup}__${studentId}`
      if (!map.has(key)) {
        map.set(key, { key, studentId, school, classGroup, records: [] })
      }
      map.get(key)!.records.push(record)
    })
    // 每位学生内部按时间升序排列，方便直接绘制 Attempt #1 -> #N 的成长曲线
    map.forEach((aggregate) => {
      aggregate.records.sort((a, b) => (a.timestamp < b.timestamp ? -1 : 1))
    })
    return Array.from(map.values()).sort((a, b) => a.studentId.localeCompare(b.studentId))
  }, [filteredRecords])

  useEffect(() => {
    if (perspective !== 'individual') return
    if (studentAggregates.length === 0) {
      setSelectedStudentKey(null)
      return
    }
    if (!selectedStudentKey || !studentAggregates.some((s) => s.key === selectedStudentKey)) {
      setSelectedStudentKey(studentAggregates[0].key)
    }
  }, [perspective, studentAggregates, selectedStudentKey])

  const selectedStudent = studentAggregates.find((s) => s.key === selectedStudentKey) ?? null

  /** 【v4.0 新增：时空胶囊尝试时间轴】切换学生，或当前筛选范围内该生的尝试次数发生变化
   * （例如切换了「📅 测试日期」筛选器）时，自动把选中的尝试胶囊定位到「最新一次」。 */
  useEffect(() => {
    if (selectedStudent && selectedStudent.records.length > 0) {
      setSelectedAttemptIndex(selectedStudent.records.length - 1)
    } else {
      setSelectedAttemptIndex(0)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedStudent?.key, selectedStudent?.records.length])

  const clampedAttemptIndex = selectedStudent
    ? Math.min(Math.max(0, selectedAttemptIndex), Math.max(0, selectedStudent.records.length - 1))
    : 0

  /** 当前「时空胶囊」选中的那一次尝试的真实临床数据：关键帧 / 实测角度 / AI 单次批注 */
  const selectedAttempt: GlobalTrainingRecord | null =
    selectedStudent && selectedStudent.records.length > 0 ? selectedStudent.records[clampedAttemptIndex] : null

  const selectedStudentScoreSeries = useMemo(() => {
    if (!selectedStudent) return []
    return selectedStudent.records.map((r) => (typeof r.score === 'number' ? r.score : null))
  }, [selectedStudent])

  const selectedStudentBestRecord = useMemo(() => {
    if (!selectedStudent || selectedStudent.records.length === 0) return null
    let best = selectedStudent.records[0]
    selectedStudent.records.forEach((record) => {
      if (typeof record.score === 'number' && (typeof best.score !== 'number' || record.score > best.score)) {
        best = record
      }
    })
    return best
  }, [selectedStudent])

  const selectedStudentErrorCounter = useMemo(() => {
    const counter: Record<string, number> = {}
    if (!selectedStudent) return counter
    selectedStudent.records.forEach((record) => {
      ;(record.biomechanicalErrors ?? []).forEach((label) => {
        counter[label] = (counter[label] ?? 0) + 1
      })
    })
    return counter
  }, [selectedStudent])

  const selectedStudentSummary = selectedStudentKey ? individualSummaries[selectedStudentKey] : null

  /** 懒加载调用后端生成「个体优缺点总结」，结果按学生 key 缓存，避免重复请求 */
  async function handleGenerateIndividualSummary(student: StudentAggregate) {
    if (loadingSummaryKey === student.key) return
    setLoadingSummaryKey(student.key)
    try {
      const scoreHistory = student.records
        .map((r) => r.score)
        .filter((s): s is number => typeof s === 'number')
      const errorCounter: Record<string, number> = {}
      student.records.forEach((record) => {
        ;(record.biomechanicalErrors ?? []).forEach((label) => {
          errorCounter[label] = (errorCounter[label] ?? 0) + 1
        })
      })
      const response = await fetch(`${API_BASE_URL}/api/generate_individual_summary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ studentId: student.studentId, scoreHistory, errorCounter }),
      })
      if (!response.ok) throw new Error(`接口返回状态码 ${response.status}`)
      const data = (await response.json()) as IndividualSummaryReport
      setIndividualSummaries((prev) => ({ ...prev, [student.key]: data }))
    } catch (error) {
      showToast(
        `⚠️ 生成个体总结失败：${error instanceof Error ? error.message : '请检查后端服务是否已启动'}`,
        false,
      )
    } finally {
      setLoadingSummaryKey(null)
    }
  }

  // 选中一位新学生、且尚无缓存总结时，自动懒加载调用一次生成
  useEffect(() => {
    if (perspective !== 'individual' || !selectedStudent) return
    if (individualSummaries[selectedStudent.key]) return
    if (loadingSummaryKey === selectedStudent.key) return
    void handleGenerateIndividualSummary(selectedStudent)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [perspective, selectedStudentKey])

  /** 打印单份报告：打开一个新窗口，用极简排版渲染后自动唤起浏览器打印对话框 */
  function handlePrintRecord(record: GlobalTrainingRecord) {
    const printWindow = window.open('', '_blank', 'width=900,height=1000')
    if (!printWindow) {
      showToast('⚠️ 浏览器拦截了打印窗口，请允许弹出窗口后重试', false)
      return
    }
    const typeLabel = record.type === 'realtime' ? '实时反馈 (A组)' : '延时反馈 (B组)'
    printWindow.document.write(`<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<title>${record.studentId || '未填写编号'} · 诊断处方</title>
<style>
  body { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; padding: 40px; color: #111; }
  h1 { font-size: 22px; text-align: center; margin-bottom: 4px; }
  .subtitle { text-align: center; color: #666; font-size: 12px; margin-bottom: 24px; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 24px; }
  td { border: 1px solid #ccc; padding: 8px 12px; font-size: 13px; }
  td.label { font-weight: bold; background: #f5f5f5; width: 140px; }
  img { display: block; max-width: 100%; margin: 0 auto 8px; border-radius: 8px; }
  .caption { text-align: center; color: #888; font-size: 11px; margin-bottom: 24px; }
  h2 { font-size: 15px; color: #1f6f4a; margin-top: 20px; }
  p { font-size: 13px; line-height: 1.7; white-space: pre-line; }
</style>
</head>
<body>
  <h1>《AI 可视化足球教学 - 生物力学诊断报告》</h1>
  <p class="subtitle">${typeLabel} - 系统自动归档生成</p>
  <table>
    <tr><td class="label">测试时间</td><td>${record.timestamp}</td></tr>
    <tr><td class="label">学校班级</td><td>${record.school} - ${record.classGroup}</td></tr>
    <tr><td class="label">学生编号</td><td>${record.studentId}</td></tr>
    <tr><td class="label">发力综合评分</td><td>${record.score ?? '暂无评分'}</td></tr>
  </table>
  ${record.impactFrameBase64 ? `<img src="${record.impactFrameBase64}" alt="击球关键帧" /><p class="caption">击球瞬间生物力学关键帧标注图</p>` : ''}
  <h2>AI 诊断批注与改进建议</h2>
  <p>${(record.aiFeedback || '暂无批注内容').replace(/</g, '&lt;')}</p>
</body>
</html>`)
    printWindow.document.close()
    printWindow.focus()
    window.setTimeout(() => printWindow.print(), 300)
  }

  /** 🖨️ 打印该生纵向对比报告：汇总全周期所有尝试的评分趋势 + 最新 AI 总结 + 最佳关键帧 */
  function handlePrintStudentReport(student: StudentAggregate) {
    const printWindow = window.open('', '_blank', 'width=900,height=1100')
    if (!printWindow) {
      showToast('⚠️ 浏览器拦截了打印窗口，请允许弹出窗口后重试', false)
      return
    }
    const summary = individualSummaries[student.key]
    const bestRecord = selectedStudentBestRecord
    const attemptRows = student.records
      .map(
        (record, index) =>
          `<tr><td>Attempt #${index + 1}</td><td>${record.timestamp}</td><td>${record.score ?? '--'}</td><td>${(record.biomechanicalErrors ?? []).join('、') || '无明显错误'}</td></tr>`,
      )
      .join('')

    printWindow.document.write(`<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<title>${student.studentId} · 纵向进化对比报告</title>
<style>
  body { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; padding: 40px; color: #111; }
  h1 { font-size: 22px; text-align: center; margin-bottom: 4px; }
  .subtitle { text-align: center; color: #666; font-size: 12px; margin-bottom: 24px; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 24px; }
  td, th { border: 1px solid #ccc; padding: 8px 12px; font-size: 12px; text-align: left; }
  th { background: #f5f5f5; }
  img { display: block; max-width: 60%; margin: 0 auto 8px; border-radius: 8px; }
  .caption { text-align: center; color: #888; font-size: 11px; margin-bottom: 24px; }
  h2 { font-size: 15px; color: #1f6f4a; margin-top: 20px; }
  p { font-size: 13px; line-height: 1.7; white-space: pre-line; }
</style>
</head>
<body>
  <h1>《个体纵向进化画像 - 生物力学诊断报告》</h1>
  <p class="subtitle">${student.school} - ${student.classGroup} · ${student.studentId} · 系统自动归档生成</p>
  ${bestRecord?.impactFrameBase64 ? `<img src="${bestRecord.impactFrameBase64}" alt="最佳关键帧" /><p class="caption">全周期最佳关键力学特征捕获帧</p>` : ''}
  <table>
    <tr><th>尝试序号</th><th>测试时间</th><th>评分</th><th>命中错误分类</th></tr>
    ${attemptRows}
  </table>
  <h2>✨ 稳定发力优势</h2>
  <p>${(summary?.strengths || '暂无 AI 总结').replace(/</g, '&lt;')}</p>
  <h2>⚠️ 需克服习惯性盲区</h2>
  <p>${(summary?.weaknesses || '暂无 AI 总结').replace(/</g, '&lt;')}</p>
</body>
</html>`)
    printWindow.document.close()
    printWindow.focus()
    window.setTimeout(() => printWindow.print(), 300)
  }

  /** 打开电脑文件夹：调用后端 /api/open_folder，在本机文件管理器中直接定位该份报告 */
  async function handleOpenFolder(record: GlobalTrainingRecord | null) {
    const targetPath = record?.directory || record?.path
    if (!targetPath) {
      showToast('⚠️ 该记录缺少本地文件路径信息', false)
      return
    }
    try {
      const response = await fetch(`${API_BASE_URL}/api/open_folder`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: targetPath }),
      })
      const data = (await response.json()) as { success: boolean; message?: string }
      if (!response.ok || !data.success) throw new Error(data.message || `接口返回状态码 ${response.status}`)
      showToast('📁 已在电脑文件管理器中打开该文件夹', true)
    } catch (error) {
      showToast(
        `⚠️ 打开文件夹失败：${error instanceof Error ? error.message : '请检查后端服务是否已启动'}`,
        false,
      )
    }
  }

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
      {/* ============================ 顶栏核心指标矩阵 (KPI Metrics) + 📥 学术矩阵导出重磅按钮 ============================ */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 lg:col-span-4">
          <KpiCard icon={Users} label="已完成测试总人数" value={kpi.totalStudents} suffix="人" accent="emerald" />
          <KpiCard icon={Radio} label="实时反馈 A 组人数" value={kpi.realtimeStudents} suffix="人" accent="sky" />
          <KpiCard icon={Clock3} label="延时反馈 B 组人数" value={kpi.delayedStudents} suffix="人" accent="teal" />
          <KpiCard
            icon={Gauge}
            label="平均发力质量评分"
            value={kpi.avgScore ?? '--'}
            suffix={kpi.avgScore !== null ? '分' : ''}
            accent="amber"
          />
        </div>

        {/* 📥 一键导出科研论文数据矩阵：金黄色/科技感边框重磅按钮 */}
        <button
          type="button"
          onClick={() => void handleExportAcademicMatrix()}
          disabled={isExportingMatrix}
          className="group relative flex flex-col items-center justify-center gap-1.5 overflow-hidden rounded-3xl border-2 border-amber-400/50 bg-gradient-to-br from-amber-500/15 via-amber-400/5 to-transparent p-4 text-center shadow-[0_0_28px_rgba(251,191,36,0.18)] transition hover:border-amber-300/80 hover:shadow-[0_0_36px_rgba(251,191,36,0.32)] active:scale-95 disabled:cursor-not-allowed disabled:opacity-60 lg:col-span-1"
        >
          <span className="pointer-events-none absolute inset-0 bg-gradient-to-r from-transparent via-amber-200/10 to-transparent opacity-0 transition-opacity group-hover:opacity-100" />
          <span className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-2xl bg-amber-400/20 ring-1 ring-amber-300/40">
            {isExportingMatrix ? (
              <Loader2 className="h-4.5 w-4.5 animate-spin text-amber-200" />
            ) : (
              <FileSpreadsheet className="h-4.5 w-4.5 text-amber-200" />
            )}
          </span>
          <span className="text-[11px] font-bold leading-tight text-amber-100">
            📥 一键导出科研论文数据矩阵
          </span>
          <span className="text-[9px] leading-tight text-amber-200/60">SPSS / Excel 宽表 · 学术数值编码</span>
        </button>
      </div>

      {/* ============================ 三级级联精准筛选器 (Filter Bar) ============================ */}
      <section className="flex flex-col gap-4 rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-xl sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
        <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center sm:gap-3">
          {/* ① 学校选择器 */}
          <label className="flex items-center gap-2 rounded-2xl bg-black/20 px-3.5 py-2.5 text-xs text-white/50">
            <span className="inline-flex flex-shrink-0">
              <SchoolIcon className="h-3.5 w-3.5 text-emerald-400" />
            </span>
            <span className="whitespace-nowrap">学校</span>
            <select
              value={selectedSchool}
              onChange={(e) => setSelectedSchool(e.target.value)}
              className="rounded-lg bg-transparent text-sm font-medium text-white outline-none [&>option]:bg-zinc-900"
            >
              <option value="all">全部学校</option>
              {schoolOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>

          {/* ② 班级/组别选择器：自动联动，仅显示已选学校下的班级 */}
          <label className="flex items-center gap-2 rounded-2xl bg-black/20 px-3.5 py-2.5 text-xs text-white/50">
            <span className="inline-flex flex-shrink-0">
              <Layers className="h-3.5 w-3.5 text-sky-400" />
            </span>
            <span className="whitespace-nowrap">班级/组别</span>
            <select
              value={selectedClassGroup}
              onChange={(e) => setSelectedClassGroup(e.target.value)}
              className="rounded-lg bg-transparent text-sm font-medium text-white outline-none [&>option]:bg-zinc-900"
            >
              <option value="all">全部班级/组别</option>
              {classGroupOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>

          {/* ③ 📅 动态日期级联筛选器：从全量归档记录中去重抽取的测试日期序列 */}
          <label className="flex items-center gap-2 rounded-2xl bg-black/20 px-3.5 py-2.5 text-xs text-white/50">
            <span className="inline-flex flex-shrink-0">
              <CalendarDays className="h-3.5 w-3.5 text-amber-400" />
            </span>
            <span className="whitespace-nowrap">测试日期</span>
            <select
              value={selectedTestDate}
              onChange={(e) => setSelectedTestDate(e.target.value)}
              className="rounded-lg bg-transparent text-sm font-medium text-white outline-none [&>option]:bg-zinc-900"
            >
              <option value="all">📅 全部日期</option>
              {dateOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>

          {/* ④ 视角切换胶囊：全班集体宏观诊断 vs 个体纵向进化追踪 */}
          <div className="inline-flex items-center gap-1 self-start rounded-full bg-black/30 p-1">
            {(
              [
                { id: 'classOverview' as CoachDashboardPerspective, label: '📊 全班集体宏观诊断', icon: LayoutGrid },
                { id: 'individual' as CoachDashboardPerspective, label: '🏃‍♂️ 个体纵向进化追踪', icon: UserSearch },
              ] as const
            ).map((option) => {
              const active = option.id === perspective
              const Icon = option.icon
              return (
                <button
                  key={option.id}
                  type="button"
                  onClick={() => setPerspective(option.id)}
                  className={`relative rounded-full px-3.5 py-2 text-xs font-medium transition-colors sm:text-sm ${
                    active ? 'text-white' : 'text-white/50 hover:text-white/80'
                  }`}
                >
                  {active && (
                    <motion.span
                      layoutId="coach-perspective-pill"
                      className="absolute inset-0 rounded-full bg-emerald-500/25 ring-1 ring-emerald-400/40"
                      transition={{ type: 'spring', stiffness: 400, damping: 32 }}
                    />
                  )}
                  <span className="relative z-10 flex items-center gap-1.5 whitespace-nowrap">
                    <span className="inline-flex flex-shrink-0">
                      <Icon className="h-3.5 w-3.5" />
                    </span>
                    {option.label}
                  </span>
                </button>
              )
            })}
          </div>
        </div>

        {/* 手动刷新按钮 */}
        <button
          type="button"
          onClick={() => void fetchAllRecords(true)}
          disabled={isRefreshing}
          className="flex items-center gap-2 self-start rounded-full bg-white/10 px-4 py-2 text-xs font-medium text-white transition hover:bg-white/20 active:scale-95 disabled:cursor-not-allowed disabled:opacity-60 sm:text-sm"
        >
          <span className="inline-flex flex-shrink-0">
            {isRefreshing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCcw className="h-3.5 w-3.5" />}
          </span>
          刷新数据
        </button>
      </section>

      {/* ============================ 主体内容区 ============================ */}
      {loadState === 'loading' ? (
        <div className="flex min-h-[360px] flex-col items-center justify-center gap-3 rounded-3xl border border-white/10 bg-white/5 backdrop-blur-xl">
          <Loader2 className="h-8 w-8 animate-spin text-emerald-400" />
          <p className="text-sm text-white/40">正在加载全量历史归档数据……</p>
        </div>
      ) : records.length === 0 ? (
        <EmptyStateCard />
      ) : perspective === 'classOverview' ? (
        <ClassOverviewSection
          scopeLabel={filterScopeLabel}
          faultStats={faultStats}
          totalRecords={filteredRecords.length}
          avgScore={classAvgScore}
          prescription={classPrescription}
          isPrescriptionLoading={isPrescriptionLoading}
          onGeneratePrescription={() => void handleGenerateClassPrescription()}
          onPrintRecord={handlePrintRecord}
          onOpenFolder={(record) => void handleOpenFolder(record)}
          recentRecords={filteredRecords.slice(0, 6)}
        />
      ) : (
        <IndividualDrilldownSection
          students={studentAggregates}
          selectedStudent={selectedStudent}
          onSelectStudent={(key) => setSelectedStudentKey(key)}
          scoreSeries={selectedStudentScoreSeries}
          bestRecord={selectedStudentBestRecord}
          summary={selectedStudentSummary}
          isSummaryLoading={selectedStudent ? loadingSummaryKey === selectedStudent.key : false}
          errorCounter={selectedStudentErrorCounter}
          onRegenerateSummary={() => selectedStudent && void handleGenerateIndividualSummary(selectedStudent)}
          onPrintStudentReport={() => selectedStudent && handlePrintStudentReport(selectedStudent)}
          onOpenFolder={() => void handleOpenFolder(selectedStudentBestRecord)}
          selectedAttempt={selectedAttempt}
          selectedAttemptIndex={clampedAttemptIndex}
          onSelectAttemptIndex={setSelectedAttemptIndex}
          onOpenAttemptFolder={() => void handleOpenFolder(selectedAttempt)}
        />
      )}

      {/* ============================ 右上角浮动 Toast 提示条 ============================ */}
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
              {toast.success ? <CheckCircle2 className="h-4 w-4 text-emerald-300" /> : <XCircle className="h-4 w-4 text-rose-300" />}
            </span>
            <p className="text-sm leading-relaxed break-all">{toast.message}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/** KPI 指标卡配色主题 */
const KPI_ACCENT_STYLE: Record<string, { icon: string; ring: string; value: string }> = {
  emerald: { icon: 'text-emerald-300 bg-emerald-500/15', ring: 'ring-emerald-400/20', value: 'text-emerald-300' },
  sky: { icon: 'text-sky-300 bg-sky-500/15', ring: 'ring-sky-400/20', value: 'text-sky-300' },
  teal: { icon: 'text-teal-300 bg-teal-500/15', ring: 'ring-teal-400/20', value: 'text-teal-300' },
  amber: { icon: 'text-amber-300 bg-amber-500/15', ring: 'ring-amber-400/20', value: 'text-amber-300' },
}

interface KpiCardProps {
  icon: typeof Users
  label: string
  value: number | string
  suffix?: string
  accent: keyof typeof KPI_ACCENT_STYLE
}

/** 顶栏核心指标卡：Apple 风格圆角磨砂玻璃质感数字大卡 */
function KpiCard({ icon: Icon, label, value, suffix, accent }: KpiCardProps) {
  const style = KPI_ACCENT_STYLE[accent]
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex items-center gap-4 rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-xl ring-1 ${style.ring}`}
    >
      <span className={`flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-2xl ${style.icon}`}>
        <Icon className="h-5 w-5" />
      </span>
      <div className="min-w-0">
        <p className="truncate text-[11px] text-white/40">{label}</p>
        <p className={`mt-0.5 text-2xl font-bold tabular-nums ${style.value}`}>
          {value}
          {suffix && <span className="ml-1 text-xs font-normal text-white/30">{suffix}</span>}
        </p>
      </div>
    </motion.div>
  )
}

/** 极美的 Apple 零数据引导占位卡片：当全局训练数据库完全为空时展示 */
function EmptyStateCard() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex min-h-[420px] flex-col items-center justify-center gap-5 rounded-3xl border border-white/10 bg-gradient-to-br from-white/5 via-white/[0.02] to-transparent p-10 text-center backdrop-blur-xl"
    >
      <div className="flex h-20 w-20 items-center justify-center rounded-[28px] bg-gradient-to-br from-emerald-400/20 to-sky-500/20 ring-1 ring-white/10">
        <Inbox className="h-9 w-9 text-emerald-300" />
      </div>
      <div className="max-w-md">
        <h3 className="text-lg font-semibold text-white/90">暂无任何历史归档数据</h3>
        <p className="mt-2 text-sm leading-relaxed text-white/40">
          前往「实时反馈系统 (实验A组)」或「延时反馈系统 (实验B组)」完成一次测试，并确保 Navbar
          全局环境设置中的「💾 本次训练数据本地落盘归档」开关处于开启状态，测试结果将自动生成 Word
          报告并实时同步至这里。
        </p>
      </div>
    </motion.div>
  )
}

/** 三级阈值评分区间对应的展示样式 */
function getScoreStyle(score: number | null | undefined): { text: string; bg: string } {
  if (score === null || score === undefined) return { text: 'text-white/40', bg: 'bg-white/10' }
  if (score >= 75) return { text: 'text-emerald-300', bg: 'bg-emerald-500/15' }
  if (score >= 55) return { text: 'text-amber-300', bg: 'bg-amber-500/15' }
  return { text: 'text-rose-300', bg: 'bg-rose-500/15' }
}

/* ============================================================================
 * 📊 全班集体宏观诊断中心 (Class Overview Mode)
 * ========================================================================== */

interface ClassOverviewSectionProps {
  scopeLabel: string
  faultStats: BiomechFaultStat[]
  totalRecords: number
  avgScore: number | null
  prescription: ClassPrescriptionReport | null
  isPrescriptionLoading: boolean
  onGeneratePrescription: () => void
  onPrintRecord: (record: GlobalTrainingRecord) => void
  onOpenFolder: (record: GlobalTrainingRecord) => void
  recentRecords: GlobalTrainingRecord[]
}

/** 生物力学错误分布对应的告警配色（出现率越高，颜色越警示） */
function getFaultBarStyle(percentage: number): { bar: string; text: string; badge: string } {
  if (percentage >= 40) return { bar: 'bg-rose-500', text: 'text-rose-300', badge: '🔴 高发警报' }
  if (percentage >= 20) return { bar: 'bg-amber-500', text: 'text-amber-300', badge: '🟡 需要关注' }
  if (percentage > 0) return { bar: 'bg-sky-500', text: 'text-sky-300', badge: '🔵 偶发个例' }
  return { bar: 'bg-emerald-500', text: 'text-emerald-300', badge: '🟢 表现良好' }
}

function ClassOverviewSection({
  scopeLabel,
  faultStats,
  totalRecords,
  avgScore,
  prescription,
  isPrescriptionLoading,
  onGeneratePrescription,
  onPrintRecord,
  onOpenFolder,
  recentRecords,
}: ClassOverviewSectionProps) {
  if (totalRecords === 0) {
    return (
      <div className="flex min-h-[280px] flex-col items-center justify-center gap-3 rounded-3xl border border-white/10 bg-white/5 p-8 text-center backdrop-blur-xl">
        <span className="inline-flex flex-shrink-0">
          <Layers className="h-10 w-10 text-white/20" />
        </span>
        <p className="text-sm text-white/40">当前筛选范围「{scopeLabel}」下暂无历史测试记录，请尝试切换学校/班级筛选条件。</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-5">
      {/* Bento Grid：左侧生物力学错误热力图 + 右侧 AIGC 全班教学处方 */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-5">
        {/* 🔥 生物力学错误分布热力图 */}
        <section className="flex flex-col gap-4 rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-xl lg:col-span-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="flex items-center gap-2 text-sm font-semibold text-white/80">
              <span className="inline-flex flex-shrink-0">
                <Flame className="h-4 w-4 text-rose-400" />
              </span>
              🔥 生物力学错误分布热力图
            </h3>
            <span className="rounded-full bg-black/30 px-2.5 py-1 text-[10px] text-white/40">
              范围：{scopeLabel} · 共 {totalRecords} 条记录 · 均分 {avgScore ?? '--'}
            </span>
          </div>

          <div className="flex flex-col gap-3.5">
            {faultStats.map((stat, index) => {
              const style = getFaultBarStyle(stat.percentage)
              return (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="flex flex-col gap-1.5"
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium text-white/80">{stat.label}</span>
                    <span className={`flex items-center gap-1.5 font-semibold ${style.text}`}>
                      {stat.percentage.toFixed(0)}% 高发率
                      <span className="rounded-full bg-black/30 px-2 py-0.5 text-[10px]">{style.badge}</span>
                    </span>
                  </div>
                  <div className="h-3 overflow-hidden rounded-full bg-black/30">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(100, stat.percentage)}%` }}
                      transition={{ duration: 0.7, ease: 'easeOut' }}
                      className={`h-full rounded-full ${style.bar}`}
                    />
                  </div>
                </motion.div>
              )
            })}
          </div>

          <p className="mt-1 text-[11px] leading-relaxed text-white/30">
            统计口径：基于当前筛选范围内每条历史测试记录命中的生物力学错误分类（支撑脚稳定性 /
            膝关节屈曲角 / 发力摆腿转髋速度 / 身体重心偏移），一目了然看清全班通病。
          </p>
        </section>

        {/* 🧠 DeepSeek 全班教案改进处方 */}
        <section className="flex flex-col gap-4 rounded-3xl border border-emerald-400/20 bg-gradient-to-br from-emerald-500/10 via-white/5 to-transparent p-5 backdrop-blur-xl lg:col-span-2">
          <h3 className="flex items-center gap-2 text-sm font-semibold text-white/80">
            <span className="inline-flex flex-shrink-0">
              <Sparkles className="h-4 w-4 text-emerald-300" />
            </span>
            🧠 DeepSeek 全班教案改进处方
          </h3>

          {!prescription && !isPrescriptionLoading && (
            <p className="flex-1 text-xs leading-relaxed text-white/40">
              点击下方按钮，系统将把左侧热力图统计的全班生物力学错误分布数据交给 DeepSeek 大模型，
              自动生成一段结构严谨的集体诊断提示与下节课教学重点建议。
            </p>
          )}

          {isPrescriptionLoading && (
            <div className="flex flex-1 flex-col items-center justify-center gap-2 text-white/40">
              <Loader2 className="h-6 w-6 animate-spin text-emerald-300" />
              <p className="text-xs">正在请求 DeepSeek 大模型生成全班教学处方……</p>
            </div>
          )}

          {prescription && !isPrescriptionLoading && (
            <div className="flex-1 space-y-3 overflow-y-auto">
              <div>
                <p className="mb-1 text-[11px] font-semibold text-emerald-300">集体诊断提示</p>
                <p className="whitespace-pre-line rounded-2xl bg-black/20 p-3 text-xs leading-relaxed text-white/85">
                  {prescription.diagnosis}
                </p>
              </div>
              <div>
                <p className="mb-1 text-[11px] font-semibold text-sky-300">下一步教学重点</p>
                <p className="whitespace-pre-line rounded-2xl bg-black/20 p-3 text-xs leading-relaxed text-white/85">
                  {prescription.prescription}
                </p>
              </div>
              <p className="text-right text-[10px] text-white/25">生成时间：{prescription.generatedAt}</p>
            </div>
          )}

          <button
            type="button"
            onClick={onGeneratePrescription}
            disabled={isPrescriptionLoading}
            className="flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-emerald-400 to-teal-400 py-3 text-sm font-bold text-black shadow-lg shadow-emerald-500/30 transition hover:brightness-110 active:scale-95 disabled:cursor-not-allowed disabled:opacity-70"
          >
            <span className="inline-flex flex-shrink-0">
              {isPrescriptionLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Wand2 className="h-4 w-4" />}
            </span>
            {prescription ? '🔄 重新召唤 AI 生成' : '✨ 召唤 AI 生成全班改进教案'}
          </button>
        </section>
      </div>

      {/* 最近归档记录流水线便当盒（缩略版），方便快速定位单条记录 */}
      <section className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-xl">
        <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-white/80">
          <span className="inline-flex flex-shrink-0">
            <Clock3 className="h-4 w-4 text-sky-400" />
          </span>
          最近归档记录
        </h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {recentRecords.map((record) => (
            <ReportCard
              key={record.id}
              record={record}
              onPrint={() => onPrintRecord(record)}
              onOpenFolder={() => onOpenFolder(record)}
            />
          ))}
        </div>
      </section>
    </div>
  )
}

interface ReportCardProps {
  record: GlobalTrainingRecord
  onPrint: () => void
  onOpenFolder: () => void
}

/** 全班学号报告流水线便当盒卡片：左图右文结构 + 底部打印/打开文件夹交互 */
function ReportCard({ record, onPrint, onOpenFolder }: ReportCardProps) {
  const scoreStyle = getScoreStyle(record.score)
  const typeLabel = record.type === 'realtime' ? '实时组 (A)' : '延时组 (B)'
  const typeBadgeStyle =
    record.type === 'realtime' ? 'bg-sky-500/15 text-sky-300' : 'bg-teal-500/15 text-teal-300'

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col overflow-hidden rounded-3xl border border-white/10 bg-white/5 backdrop-blur-xl"
    >
      <div className="flex flex-1 gap-3 p-4">
        <div className="relative h-28 w-28 flex-shrink-0 overflow-hidden rounded-2xl border border-white/10 bg-black/40">
          {record.impactFrameBase64 ? (
            <img src={record.impactFrameBase64} alt="击球关键帧" className="h-full w-full object-cover" />
          ) : (
            <div className="flex h-full w-full flex-col items-center justify-center gap-1 text-white/20">
              <ScanFace className="h-6 w-6" />
              <span className="text-[9px]">无截图</span>
            </div>
          )}
          <span className="absolute bottom-1 left-1 flex items-center gap-0.5 rounded-full bg-black/60 px-1.5 py-0.5 text-[8px] text-emerald-300">
            <Crosshair className="h-2.5 w-2.5" />
          </span>
        </div>

        <div className="flex min-w-0 flex-1 flex-col gap-1.5">
          <div className="flex items-center justify-between gap-2">
            <p className="truncate text-sm font-semibold text-white/90">{record.studentId || '未填写编号'}</p>
            <span className={`flex-shrink-0 rounded-full px-2 py-0.5 text-[9px] font-medium ${typeBadgeStyle}`}>
              {typeLabel}
            </span>
          </div>
          <p className="truncate text-[11px] text-white/35">{record.timestamp}</p>
          <p className="truncate text-[11px] text-white/35">
            {record.school} · {record.classGroup}
          </p>
          <span className={`inline-flex w-fit items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-bold ${scoreStyle.bg} ${scoreStyle.text}`}>
            {record.score ?? '--'} 分
          </span>
          <p className="mt-0.5 line-clamp-2 flex-1 text-[11px] leading-relaxed text-white/50">
            {record.aiFeedback || '暂无 AI 批注内容'}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2 border-t border-white/10 p-3">
        <button
          type="button"
          onClick={onPrint}
          className="flex flex-1 items-center justify-center gap-1.5 rounded-xl bg-white/10 py-2 text-[11px] font-medium text-white/70 transition hover:bg-white/20 active:scale-95"
        >
          <Printer className="h-3.5 w-3.5" />
          打印该报告
        </button>
        <button
          type="button"
          onClick={onOpenFolder}
          className="flex flex-1 items-center justify-center gap-1.5 rounded-xl bg-emerald-500/15 py-2 text-[11px] font-medium text-emerald-300 transition hover:bg-emerald-500/25 active:scale-95"
        >
          <FolderOpen className="h-3.5 w-3.5" />
          打开电脑文件夹
        </button>
      </div>
    </motion.div>
  )
}

/* ============================================================================
 * 🏃‍♂️ 个体纵向进化追踪档案 (Individual Drill-down Mode)
 * ========================================================================== */

interface IndividualDrilldownSectionProps {
  students: StudentAggregate[]
  selectedStudent: StudentAggregate | null
  onSelectStudent: (key: string) => void
  scoreSeries: (number | null)[]
  bestRecord: GlobalTrainingRecord | null
  summary: IndividualSummaryReport | null
  isSummaryLoading: boolean
  errorCounter: Record<string, number>
  onRegenerateSummary: () => void
  onPrintStudentReport: () => void
  onOpenFolder: () => void
  /** 【v4.0 新增：时空胶囊尝试时间轴】当前选中的那一次尝试的真实临床数据 */
  selectedAttempt: GlobalTrainingRecord | null
  selectedAttemptIndex: number
  onSelectAttemptIndex: (index: number) => void
  onOpenAttemptFolder: () => void
}

function IndividualDrilldownSection({
  students,
  selectedStudent,
  onSelectStudent,
  scoreSeries: _scoreSeries,
  bestRecord: _bestRecord,
  summary,
  isSummaryLoading,
  errorCounter,
  onRegenerateSummary,
  onPrintStudentReport,
  onOpenFolder: _onOpenFolder,
  selectedAttempt,
  selectedAttemptIndex,
  onSelectAttemptIndex,
  onOpenAttemptFolder,
}: IndividualDrilldownSectionProps) {
  if (students.length === 0) {
    return (
      <div className="flex min-h-[280px] flex-col items-center justify-center gap-3 rounded-3xl border border-white/10 bg-white/5 p-8 text-center backdrop-blur-xl">
        <span className="inline-flex flex-shrink-0">
          <UserSearch className="h-10 w-10 text-white/20" />
        </span>
        <p className="text-sm text-white/40">当前筛选条件下暂无学生记录，请尝试切换学校/班级筛选条件。</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-6 lg:flex-row lg:items-start">
      {/* -------- 左侧：学生编号池导航（约 22% 宽度） -------- */}
      <aside className="w-full flex-shrink-0 rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-xl lg:w-[22%]">
        <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-white/80">
          <span className="inline-flex flex-shrink-0">
            <Users className="h-4 w-4 text-emerald-300" />
          </span>
          学生编号池 · 共 {students.length} 人
        </h3>
        <div className="flex max-h-[560px] flex-col gap-2 overflow-y-auto pr-1">
          {students.map((student) => {
            const isActive = student.key === selectedStudent?.key
            const latestScore = student.records[student.records.length - 1]?.score ?? null
            return (
              <button
                key={student.key}
                type="button"
                onClick={() => onSelectStudent(student.key)}
                className={`flex items-center justify-between rounded-2xl border px-3.5 py-3 text-left transition ${
                  isActive
                    ? 'border-emerald-400/40 bg-emerald-400/15 text-white'
                    : 'border-white/5 bg-black/20 text-white/60 hover:bg-white/10'
                }`}
              >
                <span className="flex flex-col">
                  <span className="text-sm font-medium">{student.studentId}</span>
                  <span className="text-[11px] text-white/35">共 {student.records.length} 次测试</span>
                </span>
                <span className="flex items-center gap-1.5">
                  <span className={`text-xs font-bold ${getScoreStyle(latestScore).text}`}>{latestScore ?? '--'}</span>
                  <ChevronRight className={`h-4 w-4 ${isActive ? 'text-emerald-300' : 'text-white/20'}`} />
                </span>
              </button>
            )
          })}
        </div>
      </aside>

      {/* -------- 右侧：便当盒档案单（约 78% 宽度） -------- */}
      <div className="flex w-full flex-col gap-5 lg:w-[78%]">
        {!selectedStudent ? (
          <section className="flex min-h-[400px] flex-col items-center justify-center gap-3 rounded-3xl border border-white/10 bg-white/5 p-8 text-center backdrop-blur-xl">
            <ScanFace className="h-12 w-12 text-white/20" />
            <p className="text-sm text-white/40">请先在左侧选择一位同学，查看其个体纵向进化档案。</p>
          </section>
        ) : (
          <>
            {/* 档案头信息 */}
            <section className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-xl">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-4">
                  <div className="flex h-16 w-16 flex-shrink-0 flex-col items-center justify-center rounded-2xl bg-emerald-400/20">
                    <span className="text-2xl font-bold text-emerald-300">
                      {selectedStudent.records[selectedStudent.records.length - 1]?.score ?? '--'}
                    </span>
                    <span className="text-[9px] text-emerald-300/70">最新评分</span>
                  </div>
                  <div>
                    <p className="text-lg font-semibold text-white">{selectedStudent.studentId}</p>
                    <p className="flex items-center gap-1.5 text-xs text-white/40">
                      <GraduationCap className="h-3.5 w-3.5" />
                      {selectedStudent.school} · {selectedStudent.classGroup}
                    </p>
                  </div>
                </div>
                <div className="text-right text-xs text-white/40">
                  <p>全周期共完成 {selectedStudent.records.length} 次测试</p>
                  <p>Attempt #1 → #{selectedStudent.records.length}</p>
                </div>
              </div>
            </section>

            {/* ⏱️ 时空胶囊尝试时间轴 (Attempt #N Drill-down) */}
            <section className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-xl">
              <h4 className="mb-3 flex items-center gap-2 text-sm font-semibold text-white/80">
                <span className="inline-flex flex-shrink-0">
                  <Clock3 className="h-4 w-4 text-sky-300" />
                </span>
                ⏱️ 时空胶囊尝试时间轴 · 指哪打哪，点击任意胶囊立即追溯该次真实数据
              </h4>
              <div className="flex gap-2.5 overflow-x-auto pb-1">
                {selectedStudent.records.map((record, idx) => {
                  const isActive = idx === selectedAttemptIndex
                  const style = getScoreStyle(record.score)
                  return (
                    <button
                      key={record.id}
                      type="button"
                      onClick={() => onSelectAttemptIndex(idx)}
                      className={`flex flex-shrink-0 flex-col items-start gap-0.5 rounded-2xl border px-4 py-2.5 text-left transition ${
                        isActive
                          ? 'border-sky-400/50 bg-sky-400/15 ring-1 ring-sky-300/40'
                          : 'border-white/10 bg-black/20 hover:bg-white/10'
                      }`}
                    >
                      <span className={`text-[10px] font-semibold ${isActive ? 'text-sky-200' : 'text-white/40'}`}>
                        ⚽ {getRecordTimeLabel(record)} · Attempt #{idx + 1}
                      </span>
                      <span className={`text-sm font-bold ${style.text}`}>{record.score ?? '--'} 分</span>
                    </button>
                  )
                })}
              </div>
            </section>

            {/* 📈 双轴互动的运动学成长期刊图 (Biomechanical Dual-Axis Curve) */}
            <section className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-xl">
              <h4 className="mb-3 flex items-center gap-2 text-sm font-semibold text-white/80">
                <span className="inline-flex flex-shrink-0">
                  <TrendingUp className="h-4 w-4 text-emerald-300" />
                </span>
                📈 双轴运动学成长期刊图 · 综合评分 (左轴) × 击球瞬间右膝屈曲角度 (右轴)
              </h4>
              {selectedStudent.records.length === 0 ? (
                <div className="flex h-64 w-full items-center justify-center rounded-2xl bg-black/20 text-xs text-white/25">
                  暂无任何尝试数据，无法绘制成长期刊图
                </div>
              ) : (
                <BiomechDualAxisChart
                  records={selectedStudent.records}
                  selectedIndex={selectedAttemptIndex}
                  onSelectIndex={onSelectAttemptIndex}
                />
              )}
              <p className="mt-2 text-[11px] leading-relaxed text-white/30">
                🎯 学术合规发力区间 (135°-155°)：图中绿色阴影带即为触球瞬间膝关节屈曲角度的学术合规区间，一眼看清哪一次尝试精准落入标准区间。
              </p>
            </section>

            {/* 下方三栏：本次尝试关键帧 / DeepSeek 单次批注 / AI 全周期总结 */}
            <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
              {/* ① 选中尝试的关键帧 + OpenCV 实测角度数值 */}
              <section className="flex flex-col gap-3 rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-xl">
                <h4 className="flex items-center gap-2 text-sm font-semibold text-white/80">
                  <span className="inline-flex flex-shrink-0">
                    <Award className="h-4 w-4 text-amber-300" />
                  </span>
                  Attempt #{selectedAttemptIndex + 1} 关键帧
                </h4>
                <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-black/40">
                  {selectedAttempt?.impactFrameBase64 ? (
                    <img
                      src={selectedAttempt.impactFrameBase64}
                      alt="该次尝试关键帧"
                      className="h-full w-full object-contain"
                    />
                  ) : (
                    <div className="flex aspect-video flex-col items-center justify-center gap-2 text-white/30">
                      <ScanFace className="h-10 w-10" />
                      <p className="max-w-[16rem] text-center text-xs">该次尝试暂无有效击球关键帧截图。</p>
                    </div>
                  )}
                  <span className="absolute left-3 top-3 flex items-center gap-1.5 rounded-full border border-white/10 bg-black/60 px-2.5 py-1 text-[10px] text-emerald-300 backdrop-blur-xl">
                    <Crosshair className="h-3 w-3" />
                    矢量标注 · 髋-膝-踝动力链
                  </span>
                  {selectedAttempt && (
                    <span className="absolute right-3 top-3 flex items-center gap-1 rounded-full border border-amber-300/30 bg-black/60 px-2.5 py-1 text-[10px] text-amber-200 backdrop-blur-xl">
                      <Award className="h-3 w-3" />
                      {selectedAttempt.score ?? '--'} 分
                    </span>
                  )}
                </div>

                {/* OpenCV 动力链实测角度数值 */}
                <div className="grid grid-cols-2 gap-2">
                  <div className="rounded-xl bg-black/20 p-2.5 text-center">
                    <p className="text-[10px] text-white/35">膝角实测</p>
                    <p className="text-base font-bold text-sky-300">
                      {typeof selectedAttempt?.kneeFlexionAngle === 'number'
                        ? `${selectedAttempt.kneeFlexionAngle}°`
                        : '--'}
                    </p>
                  </div>
                  <div className="rounded-xl bg-black/20 p-2.5 text-center">
                    <p className="text-[10px] text-white/35">支撑脚离球</p>
                    <p className="text-base font-bold text-teal-300">
                      {typeof selectedAttempt?.supportFootDistance === 'number'
                        ? `${selectedAttempt.supportFootDistance}cm`
                        : '--'}
                    </p>
                  </div>
                </div>

                <div className="mt-1 flex items-center gap-2">
                  <button
                    type="button"
                    onClick={onPrintStudentReport}
                    className="flex flex-1 items-center justify-center gap-1.5 rounded-xl bg-white/10 py-2.5 text-xs font-medium text-white/70 transition hover:bg-white/20 active:scale-95"
                  >
                    <Printer className="h-3.5 w-3.5" />
                    🖨️ 打印纵向对比报告
                  </button>
                  <button
                    type="button"
                    onClick={onOpenAttemptFolder}
                    className="flex flex-1 items-center justify-center gap-1.5 rounded-xl bg-emerald-500/15 py-2.5 text-xs font-medium text-emerald-300 transition hover:bg-emerald-500/25 active:scale-95"
                  >
                    <FolderOpen className="h-3.5 w-3.5" />
                    📁 打开该次归档文件夹
                  </button>
                </div>
              </section>

              {/* ② DeepSeek AI 单次优缺点批注（严格对应当前选中的这一次尝试） */}
              <section className="flex flex-col gap-3 rounded-3xl border border-sky-400/20 bg-gradient-to-br from-sky-500/10 via-white/5 to-transparent p-5 backdrop-blur-xl">
                <h4 className="flex items-center gap-2 text-sm font-semibold text-white/80">
                  <span className="inline-flex flex-shrink-0">
                    <MessageSquareText className="h-4 w-4 text-sky-300" />
                  </span>
                  DeepSeek AI 单次优缺点批注
                </h4>
                <div className="flex-1 rounded-2xl bg-black/20 p-3.5">
                  <p className="whitespace-pre-line text-xs leading-relaxed text-white/85">
                    {selectedAttempt?.aiFeedback || '该次尝试暂无 AI 批注内容（可能是历史归档数据缺失该字段）。'}
                  </p>
                </div>
                {selectedAttempt && (selectedAttempt.biomechanicalErrors ?? []).length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {(selectedAttempt.biomechanicalErrors ?? []).map((label) => (
                      <span key={label} className="rounded-full bg-rose-500/15 px-2.5 py-1 text-[10px] text-rose-300">
                        {label}
                      </span>
                    ))}
                  </div>
                )}
              </section>

              {/* ③ AI 个体优缺点总结（全周期视角） */}
              <section className="flex flex-col gap-4 rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-xl">
                <h4 className="flex items-center gap-2 text-sm font-semibold text-white/80">
                  <span className="inline-flex flex-shrink-0">
                    <Sparkles className="h-4 w-4 text-emerald-300" />
                  </span>
                  AI 全周期优缺点总结
                </h4>

                {isSummaryLoading ? (
                  <div className="flex flex-1 flex-col items-center justify-center gap-2 text-white/40">
                    <Loader2 className="h-6 w-6 animate-spin text-emerald-300" />
                    <p className="text-xs">正在基于该生全周期数据生成 AI 总结……</p>
                  </div>
                ) : (
                  <div className="flex flex-1 flex-col gap-3">
                    <div className="rounded-2xl border border-emerald-400/20 bg-emerald-500/10 p-3.5">
                      <p className="mb-1.5 flex items-center gap-1.5 text-[11px] font-semibold text-emerald-300">
                        <ThumbsUp className="h-3.5 w-3.5" />
                        ✨ 稳定发力优势
                      </p>
                      <p className="text-xs leading-relaxed text-white/85">
                        {summary?.strengths || '暂无 AI 总结（可能后端服务未启动）。'}
                      </p>
                    </div>
                    <div className="rounded-2xl border border-amber-400/20 bg-amber-500/10 p-3.5">
                      <p className="mb-1.5 flex items-center gap-1.5 text-[11px] font-semibold text-amber-300">
                        <ShieldAlert className="h-3.5 w-3.5" />
                        ⚠️ 需克服习惯性盲区
                      </p>
                      <p className="text-xs leading-relaxed text-white/85">
                        {summary?.weaknesses || '暂无 AI 总结（可能后端服务未启动）。'}
                      </p>
                    </div>

                    {Object.keys(errorCounter).length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {Object.entries(errorCounter).map(([label, count]) => (
                          <span key={label} className="rounded-full bg-black/30 px-2.5 py-1 text-[10px] text-white/50">
                            {label} × {count}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                <button
                  type="button"
                  onClick={onRegenerateSummary}
                  disabled={isSummaryLoading}
                  className="flex items-center justify-center gap-2 rounded-2xl border border-white/10 bg-white/5 py-2.5 text-xs font-medium text-white/70 transition hover:bg-white/10 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <Sparkles className="h-3.5 w-3.5" />
                  重新生成 AI 总结
                </button>
              </section>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

/* ============================================================================
 * 📈 双轴互动的运动学成长期刊图 (Biomechanical Dual-Axis Curve)：
 * 左轴（绿色实线）= 综合动作发力评分 Total Score；
 * 右轴（蓝色虚线）= 击球瞬间右膝屈曲角度 Knee Flexion Angle；
 * 绿色半透明阴影带 = 🎯 学术合规发力区间 (135°-155°)。
 * ========================================================================== */

interface BiomechDualAxisChartProps {
  records: GlobalTrainingRecord[]
  selectedIndex: number
  onSelectIndex: (index: number) => void
}

/** 自定义可点击圆点：点击图表上任意数据点，立即联动下方「时空胶囊」切换到对应尝试 */
function ClickableDot(
  props: {
    cx?: number
    cy?: number
    index?: number
    payload?: { index: number }
    fill: string
  } & { onDotClick: (index: number) => void; isActive: boolean },
) {
  const { cx, cy, fill, onDotClick, isActive } = props
  const pointIndex = props.payload?.index ?? props.index ?? 0
  if (typeof cx !== 'number' || typeof cy !== 'number') return null
  return (
    <circle
      cx={cx}
      cy={cy}
      r={isActive ? 6 : 4}
      fill={fill}
      stroke={isActive ? '#fff' : 'transparent'}
      strokeWidth={isActive ? 2 : 0}
      style={{ cursor: 'pointer' }}
      onClick={() => onDotClick(pointIndex)}
    />
  )
}

function BiomechDualAxisChart({ records, selectedIndex, onSelectIndex }: BiomechDualAxisChartProps) {
  const chartData = records.map((record, index) => ({
    index,
    attempt: `#${index + 1}`,
    label: `Attempt #${index + 1} · ${getRecordTimeLabel(record)}`,
    score: typeof record.score === 'number' ? record.score : null,
    kneeAngle: typeof record.kneeFlexionAngle === 'number' ? record.kneeFlexionAngle : null,
  }))

  const hasKneeAngleData = chartData.some((point) => typeof point.kneeAngle === 'number')

  return (
    <div className="h-72 w-full rounded-2xl bg-black/20 p-2">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={chartData} margin={{ top: 16, right: 18, bottom: 4, left: 4 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
          <XAxis dataKey="attempt" tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 11 }} axisLine={false} tickLine={false} />
          <YAxis
            yAxisId="left"
            domain={[0, 100]}
            tick={{ fill: '#6ee7b7', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={32}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            domain={[95, 185]}
            tick={{ fill: '#7dd3fc', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={38}
          />
          {/* 🎯 学术合规发力区间 (135°-155°) 绿色半透明阴影带 */}
          <ReferenceArea
            yAxisId="right"
            y1={OPTIMAL_KNEE_ANGLE_MIN}
            y2={OPTIMAL_KNEE_ANGLE_MAX}
            fill="#34d399"
            fillOpacity={0.14}
            stroke="#34d399"
            strokeOpacity={0.3}
            strokeDasharray="4 4"
            label={{
              value: '🎯 学术合规发力区间 (135°-155°)',
              position: 'insideTopRight',
              fill: '#6ee7b7',
              fontSize: 10,
            }}
          />
          <Tooltip
            contentStyle={{
              background: 'rgba(10,14,12,0.92)',
              border: '1px solid rgba(255,255,255,0.12)',
              borderRadius: 12,
              fontSize: 12,
            }}
            labelFormatter={(_label, payload) => payload?.[0]?.payload?.label ?? _label}
            formatter={(value, name) => {
              const isScore = name === 'score'
              if (value === null || value === undefined) return ['暂无数据', isScore ? '综合评分' : '膝角']
              return isScore ? [`${value} 分`, '综合评分'] : [`${value}°`, '膝角']
            }}
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="score"
            name="score"
            stroke="#34d399"
            strokeWidth={2.5}
            connectNulls
            dot={(dotProps: unknown) => (
              <ClickableDot
                {...(dotProps as { cx?: number; cy?: number; payload?: { index: number } })}
                fill="#34d399"
                onDotClick={onSelectIndex}
                isActive={(dotProps as { payload?: { index: number } }).payload?.index === selectedIndex}
              />
            )}
            activeDot={{ r: 6 }}
          />
          {hasKneeAngleData && (
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="kneeAngle"
              name="kneeAngle"
              stroke="#38bdf8"
              strokeWidth={2}
              strokeDasharray="6 4"
              connectNulls
              dot={(dotProps: unknown) => (
                <ClickableDot
                  {...(dotProps as { cx?: number; cy?: number; payload?: { index: number } })}
                  fill="#38bdf8"
                  onDotClick={onSelectIndex}
                  isActive={(dotProps as { payload?: { index: number } }).payload?.index === selectedIndex}
                />
              )}
              activeDot={{ r: 6 }}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
