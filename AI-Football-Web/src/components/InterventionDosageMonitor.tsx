import { useMemo } from 'react'
import { Activity, AlertTriangle, CheckCircle2, Info } from 'lucide-react'
import type { GlobalTrainingRecord } from '../types'

const PHASE_ORDER = ['T0', 'T1', 'T2', 'T3', 'T4'] as const

interface InterventionDosageMonitorProps {
  /** 全局过滤器穿透后的唯一数据源 */
  filteredDataset: GlobalTrainingRecord[]
  className?: string
}

interface SessionColumn {
  key: string
  date: string
  phase: string
  label: string
  shortDate: string
}

interface DosageRow {
  key: string
  studentId: string
  school: string
  classGroup: string
  groupBadge: string
  /** 各课时 attempt count，与 sessions 对齐 */
  counts: number[]
  total: number
}

function getRecordTestDate(record: GlobalTrainingRecord): string {
  if (record.testDate && record.testDate.length >= 10) return record.testDate.slice(0, 10)
  const ts = record.timestamp || ''
  if (ts.length >= 10 && ts[4] === '-' && ts[7] === '-') return ts.slice(0, 10)
  return '未知日期'
}

function parsePhase(raw: unknown): string | null {
  if (raw == null) return null
  let text = String(raw).trim().toUpperCase()
  if (!text) return null
  if (/^\d+$/.test(text)) text = `T${text}`
  return (PHASE_ORDER as readonly string[]).includes(text) ? text : null
}

function resolveGroupBadge(record: GlobalTrainingRecord): string {
  if (record.type === 'realtime' || record.groupTypeCode === 1) return 'A'
  if (record.type === 'delayed' || record.groupTypeCode === 2) return 'B'
  const label = (record.classGroup || '').toUpperCase()
  if (label.includes('A') || label.includes('实时')) return 'A'
  if (label.includes('B') || label.includes('延时')) return 'B'
  return '?'
}

function formatShortDate(date: string): string {
  if (date.length >= 10 && date[4] === '-') return `${date.slice(5, 7)}/${date.slice(8, 10)}`
  return date
}

/** 具身反馈剂量色阶：0 缺勤 / 1–2 不足 / ≥3 达标 */
function doseCellClass(count: number): string {
  if (count <= 0) {
    return 'bg-gray-100 text-gray-500 border border-amber-400/70 shadow-[inset_0_0_0_1px_rgba(251,191,36,0.35)]'
  }
  if (count <= 2) {
    return 'bg-yellow-100 text-yellow-800 border border-yellow-300/60'
  }
  return 'bg-green-500 text-white font-bold border border-green-600/40'
}

/**
 * 实验干预剂量监控板（Intervention Dosage Monitor）
 * 将 filteredDataset 变换为「学生 × 课时」attempt-count 矩阵，证明 A/B 组依从性。
 */
export default function InterventionDosageMonitor({
  filteredDataset,
  className = '',
}: InterventionDosageMonitorProps) {
  const { sessions, rows, columnAverages, complianceRate, totalAttempts } = useMemo(() => {
    const dateBuckets = new Map<string, GlobalTrainingRecord[]>()
    for (const record of filteredDataset) {
      const day = getRecordTestDate(record)
      if (day === '未知日期') continue
      const list = dateBuckets.get(day) ?? []
      list.push(record)
      dateBuckets.set(day, list)
    }

    const sortedDates = Array.from(dateBuckets.keys()).sort((a, b) => a.localeCompare(b))
    const inferredPhase = new Map<string, string>()
    sortedDates.forEach((day, i) => {
      inferredPhase.set(day, PHASE_ORDER[Math.min(i, PHASE_ORDER.length - 1)])
    })

    const sessions: SessionColumn[] = sortedDates.map((date, index) => {
      const bucket = dateBuckets.get(date) || []
      const explicit = bucket.map((r) => parsePhase(r.timepoint ?? r.phase)).find(Boolean)
      const phase = explicit || inferredPhase.get(date) || `T${index}`
      const shortDate = formatShortDate(date)
      return {
        key: date,
        date,
        phase,
        shortDate,
        label: `${phase} · ${shortDate}`,
      }
    })

    const studentMap = new Map<
      string,
      {
        studentId: string
        school: string
        classGroup: string
        groupBadge: string
        byDate: Map<string, number>
      }
    >()

    for (const record of filteredDataset) {
      const day = getRecordTestDate(record)
      if (day === '未知日期') continue
      const school = (record.school || '').trim() || '未设置学校'
      const classGroup = (record.classGroup || '').trim() || '未设置班级'
      const studentId = (record.studentId || '').trim() || '未填写编号'
      const key = `${school}__${classGroup}__${studentId}`
      if (!studentMap.has(key)) {
        studentMap.set(key, {
          studentId,
          school,
          classGroup,
          groupBadge: resolveGroupBadge(record),
          byDate: new Map(),
        })
      }
      const entry = studentMap.get(key)!
      entry.byDate.set(day, (entry.byDate.get(day) || 0) + 1)
      entry.groupBadge = resolveGroupBadge(record)
    }

    const rows: DosageRow[] = Array.from(studentMap.entries())
      .map(([key, entry]) => {
        const counts = sessions.map((session) => entry.byDate.get(session.date) || 0)
        const total = counts.reduce((sum, n) => sum + n, 0)
        return {
          key,
          studentId: entry.studentId,
          school: entry.school,
          classGroup: entry.classGroup,
          groupBadge: entry.groupBadge,
          counts,
          total,
        }
      })
      .sort((a, b) => {
        const g = a.groupBadge.localeCompare(b.groupBadge, 'zh-CN')
        if (g !== 0) return g
        return a.studentId.localeCompare(b.studentId, 'zh-CN')
      })

    const columnAverages = sessions.map((_, colIdx) => {
      if (rows.length === 0) return 0
      const sum = rows.reduce((acc, row) => acc + row.counts[colIdx], 0)
      return Math.round((sum / rows.length) * 10) / 10
    })

    let compliantCells = 0
    let scoredCells = 0
    for (const row of rows) {
      for (const count of row.counts) {
        scoredCells += 1
        if (count >= 3) compliantCells += 1
      }
    }
    const complianceRate =
      scoredCells > 0 ? Math.round((compliantCells / scoredCells) * 1000) / 10 : 0
    const totalAttempts = rows.reduce((sum, row) => sum + row.total, 0)

    return { sessions, rows, columnAverages, complianceRate, totalAttempts }
  }, [filteredDataset])

  if (filteredDataset.length === 0) {
    return (
      <section
        className={`rounded-2xl border border-dashed border-white/10 bg-slate-900/30 px-4 py-6 ${className}`.trim()}
      >
        <div className="flex flex-col items-center gap-2 text-center">
          <AlertTriangle className="h-5 w-5 text-amber-300/70" />
          <p className="text-sm text-white/45">
            该筛选条件下暂无有效测试数据，请调整右上角的全局过滤器
          </p>
        </div>
      </section>
    )
  }

  if (sessions.length === 0 || rows.length === 0) {
    return (
      <section
        className={`rounded-2xl border border-white/10 bg-slate-900/40 px-4 py-6 ${className}`.trim()}
      >
        <HeaderBar complianceRate={0} studentCount={0} sessionCount={0} totalAttempts={0} />
        <p className="mt-3 text-center text-xs text-white/35">暂无可聚合的课时日期</p>
      </section>
    )
  }

  return (
    <section
      className={`rounded-2xl border border-white/10 bg-slate-900/40 px-3 py-3 sm:px-4 ${className}`.trim()}
    >
      <HeaderBar
        complianceRate={complianceRate}
        studentCount={rows.length}
        sessionCount={sessions.length}
        totalAttempts={totalAttempts}
      />

      <LegendBar />

      <div className="mt-3 overflow-x-auto overscroll-contain rounded-xl border border-white/8 bg-[#0d1117] scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
        <table className="dosage-matrix min-w-full border-collapse text-left">
          <thead>
            <tr className="border-b border-white/10 bg-[#161b22]">
              <th className="sticky left-0 z-10 min-w-[148px] bg-[#161b22] px-3 py-2.5 text-[11px] font-semibold tracking-wide text-white/55">
                被试（学号）
              </th>
              {sessions.map((session) => (
                <th
                  key={session.key}
                  className="min-w-[72px] px-1.5 py-2.5 text-center"
                  title={`${session.date} · ${session.phase}`}
                >
                  <div className="text-[11px] font-bold tabular-nums text-emerald-300/90">
                    {session.phase}
                  </div>
                  <div className="text-[10px] tabular-nums text-white/35">{session.shortDate}</div>
                </th>
              ))}
              <th className="min-w-[88px] px-2 py-2.5 text-center text-[11px] font-semibold tracking-wide text-white/55">
                总完成次数
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIdx) => (
              <tr
                key={row.key}
                className={`border-b border-white/[0.06] ${
                  rowIdx % 2 === 0 ? 'bg-[#0d1117]' : 'bg-[#010409]/60'
                }`}
              >
                <td className="sticky left-0 z-10 bg-inherit px-3 py-1.5">
                  <div className="flex min-w-0 items-center gap-2">
                    <span
                      className={`flex h-5 w-5 flex-shrink-0 items-center justify-center rounded text-[10px] font-bold ${
                        row.groupBadge === 'A'
                          ? 'bg-sky-500/25 text-sky-300'
                          : row.groupBadge === 'B'
                            ? 'bg-teal-500/25 text-teal-300'
                            : 'bg-white/10 text-white/45'
                      }`}
                      title={row.groupBadge === 'A' ? 'A组 · 实时反馈' : row.groupBadge === 'B' ? 'B组 · 延时反馈' : '未归组'}
                    >
                      {row.groupBadge}
                    </span>
                    <div className="min-w-0">
                      <p className="truncate text-xs font-medium text-white/85">{row.studentId}</p>
                      <p className="truncate text-[10px] text-white/30">
                        {row.classGroup}
                      </p>
                    </div>
                  </div>
                </td>
                {row.counts.map((count, colIdx) => (
                  <td key={`${row.key}-${sessions[colIdx].key}`} className="px-1.5 py-1.5 text-center">
                    <span
                      className={`inline-flex h-8 w-8 items-center justify-center rounded-md text-xs tabular-nums ${doseCellClass(count)}`}
                      title={`${row.studentId} · ${sessions[colIdx].label}：${count} 次有效尝试`}
                    >
                      {count}
                    </span>
                  </td>
                ))}
                <td className="px-2 py-1.5 text-center">
                  <span
                    className={`inline-flex min-w-[2.25rem] items-center justify-center rounded-md px-2 py-1 text-xs font-semibold tabular-nums ${
                      row.total >= sessions.length * 3
                        ? 'bg-green-500/90 text-white'
                        : row.total === 0
                          ? 'bg-gray-100 text-gray-500'
                          : 'bg-yellow-100 text-yellow-800'
                    }`}
                  >
                    {row.total}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr className="border-t border-white/15 bg-[#161b22]">
              <td className="sticky left-0 z-10 bg-[#161b22] px-3 py-2.5 text-[11px] font-semibold text-white/60">
                该课时组内平均完成次数
              </td>
              {columnAverages.map((avg, colIdx) => (
                <td key={`avg-${sessions[colIdx].key}`} className="px-1.5 py-2.5 text-center">
                  <span className="inline-flex h-8 min-w-[2rem] items-center justify-center rounded-md border border-white/10 bg-white/5 px-1.5 text-xs font-semibold tabular-nums text-emerald-200/90">
                    {avg.toFixed(1)}
                  </span>
                </td>
              ))}
              <td className="px-2 py-2.5 text-center">
                <span className="text-[10px] text-white/30">—</span>
              </td>
            </tr>
          </tfoot>
        </table>
      </div>

      <p className="mt-2 flex items-start gap-1.5 text-[10px] leading-relaxed text-white/30">
        <Info className="mt-0.5 h-3 w-3 flex-shrink-0" />
        单元格 = 该被试在对应课时的有效尝试次数（Attempt Count）。科研干预达标阈值：单课时 ≥ 3
        次；缺勤（0）以警告边框标出，用于 A/B 组依从性（Compliance）报告。
      </p>
    </section>
  )
}

function HeaderBar({
  complianceRate,
  studentCount,
  sessionCount,
  totalAttempts,
}: {
  complianceRate: number
  studentCount: number
  sessionCount: number
  totalAttempts: number
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-emerald-300/85">
        <Activity className="h-3.5 w-3.5" />
        Intervention Dosage Monitor · 实验干预剂量监控板
      </span>
      <span className="rounded-full bg-black/30 px-2 py-0.5 text-[10px] tabular-nums text-white/40">
        {studentCount} 名被试 · {sessionCount} 课时 · Σ {totalAttempts} 次
      </span>
      <span
        className={`ml-auto inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-semibold tabular-nums ${
          complianceRate >= 70
            ? 'bg-green-500/20 text-green-300'
            : complianceRate >= 40
              ? 'bg-yellow-500/15 text-yellow-200'
              : 'bg-amber-500/15 text-amber-200'
        }`}
        title="单元格中达到 ≥3 次干预标准的比例"
      >
        <CheckCircle2 className="h-3 w-3" />
        达标率 {complianceRate.toFixed(1)}%
      </span>
    </div>
  )
}

function LegendBar() {
  return (
    <div className="mt-2 flex flex-wrap items-center gap-3 text-[10px] text-white/45">
      <span className="text-white/30">剂量色阶</span>
      <span className="inline-flex items-center gap-1.5">
        <span className={`inline-flex h-5 w-5 items-center justify-center rounded-sm text-[10px] ${doseCellClass(0)}`}>
          0
        </span>
        缺勤
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className={`inline-flex h-5 w-5 items-center justify-center rounded-sm text-[10px] ${doseCellClass(1)}`}>
          1
        </span>
        样本不足 (1–2)
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className={`inline-flex h-5 w-5 items-center justify-center rounded-sm text-[10px] ${doseCellClass(3)}`}>
          3
        </span>
        符合科研干预标准 (≥3)
      </span>
      <span className="ml-auto hidden items-center gap-1 sm:inline-flex">
        <span className="h-2.5 w-2.5 rounded-sm bg-gray-100" />
        <span className="h-2.5 w-2.5 rounded-sm bg-yellow-100" />
        <span className="h-2.5 w-2.5 rounded-sm bg-[#9be9a8]" />
        <span className="h-2.5 w-2.5 rounded-sm bg-[#40c463]" />
        <span className="h-2.5 w-2.5 rounded-sm bg-green-500" />
        <span className="text-white/25">GitHub Contribution 风格</span>
      </span>
    </div>
  )
}
