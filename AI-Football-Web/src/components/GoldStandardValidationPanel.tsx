import { useEffect, useRef, useState } from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  Database,
  Download,
  FileCheck2,
  FlaskConical,
  Loader2,
  Play,
  Upload,
} from 'lucide-react'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

interface PreviewResult {
  success: boolean
  canCommit: boolean
  previewToken: string
  sourceFilename: string
  sourceSha256: string
  totalRows: number
  validRowCount: number
  invalidRowCount: number
  duplicateCount: number
  invalidRows: Array<{ rowNumber: number; errors: string[] }>
  warnings: Array<{ rowNumber: number; warnings: string[] }>
}

interface DatasetManifest {
  datasetId: string
  createdAt: string
  sourceFilename: string
  sourceSha256: string
  validRowCount: number
  excludedRowCount: number
  thresholdVersion: string
}

interface MetricValidation {
  n: number
  mae: number
  rmse: number
  bias: number
  iccAbsoluteAgreement: number | null
  levelAccuracy: number | null
  impactFrameMae: number | null
  impactWithin1FrameRate: number | null
}

interface ValidationReport {
  datasetId: string
  generatedAt: string
  thresholdVersion: string
  alignedSampleCount: number
  excludedSampleCount: number
  metrics: Record<string, MetricValidation>
  interRater: {
    multiAnnotatorSampleCount: number
    meanAbsoluteDifference: number | null
    maxAbsoluteDifference: number | null
  }
  hashVerification: {
    verifiedCount: number
    unavailableCount: number
    mismatchCount: number
  }
}

interface ThresholdProfile {
  thresholdVersion: string
  impactKneeAngleDeg: {
    greenLow: number
    greenHigh: number
    yellowLow: number
    yellowHigh: number
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : '请求失败，请检查后端服务'
}

async function readApiError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string }
    return body.detail || `接口返回状态码 ${response.status}`
  } catch {
    return `接口返回状态码 ${response.status}`
  }
}

export default function GoldStandardValidationPanel() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [preview, setPreview] = useState<PreviewResult | null>(null)
  const [datasets, setDatasets] = useState<DatasetManifest[]>([])
  const [threshold, setThreshold] = useState<ThresholdProfile | null>(null)
  const [report, setReport] = useState<ValidationReport | null>(null)
  const [busy, setBusy] = useState<'preview' | 'commit' | string | null>(null)
  const [notice, setNotice] = useState<{ text: string; ok: boolean } | null>(null)

  async function refreshDatasets() {
    const response = await fetch(`${API_BASE_URL}/api/research/gold-standard/datasets`)
    if (!response.ok) throw new Error(await readApiError(response))
    const body = (await response.json()) as { datasets?: DatasetManifest[] }
    setDatasets(Array.isArray(body.datasets) ? body.datasets : [])
  }

  useEffect(() => {
    void Promise.all([
      refreshDatasets(),
      fetch(`${API_BASE_URL}/api/research/threshold-profile`)
        .then((response) => {
          if (!response.ok) throw new Error(`阈值接口返回 ${response.status}`)
          return response.json() as Promise<ThresholdProfile>
        })
        .then(setThreshold),
    ]).catch((error) => setNotice({ text: errorMessage(error), ok: false }))
  }, [])

  async function handleFile(file: File) {
    setBusy('preview')
    setPreview(null)
    setReport(null)
    setNotice(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const response = await fetch(`${API_BASE_URL}/api/research/gold-standard/import/preview`, {
        method: 'POST',
        body: form,
      })
      if (!response.ok) throw new Error(await readApiError(response))
      const body = (await response.json()) as PreviewResult
      setPreview(body)
      setNotice({
        text: `预检完成：${body.validRowCount} 行有效，${body.invalidRowCount} 行需排除`,
        ok: body.invalidRowCount === 0,
      })
    } catch (error) {
      setNotice({ text: errorMessage(error), ok: false })
    } finally {
      setBusy(null)
    }
  }

  async function commitPreview() {
    if (!preview?.canCommit) return
    setBusy('commit')
    try {
      const response = await fetch(`${API_BASE_URL}/api/research/gold-standard/import/commit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ previewToken: preview.previewToken }),
      })
      if (!response.ok) throw new Error(await readApiError(response))
      const body = (await response.json()) as { dataset: DatasetManifest }
      setNotice({ text: `已提交数据集 ${body.dataset.datasetId}`, ok: true })
      setPreview(null)
      await refreshDatasets()
    } catch (error) {
      setNotice({ text: errorMessage(error), ok: false })
    } finally {
      setBusy(null)
    }
  }

  async function validateDataset(datasetId: string) {
    setBusy(datasetId)
    setReport(null)
    try {
      const response = await fetch(`${API_BASE_URL}/api/research/gold-standard/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ datasetId }),
      })
      if (!response.ok) throw new Error(await readApiError(response))
      const body = (await response.json()) as ValidationReport
      setReport(body)
      setNotice({
        text: `效度计算完成：对齐 ${body.alignedSampleCount} 个样本，排除 ${body.excludedSampleCount} 个`,
        ok: body.alignedSampleCount > 0,
      })
    } catch (error) {
      setNotice({ text: errorMessage(error), ok: false })
    } finally {
      setBusy(null)
    }
  }

  function downloadTemplate() {
    const header = [
      'sample_id',
      'attempt_id',
      'video_hash',
      'source_type',
      'annotator_id',
      'metric_key',
      'reference_value',
      'unit',
      'reference_frame',
      'fps',
      'swing_leg',
      'camera_view',
      'annotation_version',
      'system_value',
      'system_frame',
    ].join(',')
    const example = [
      'S001-A01',
      'attempt-record-id',
      '填写视频文件SHA256',
      'kinovea',
      'RATER01',
      'impact_knee_angle',
      '151.2',
      'deg',
      '42',
      '30',
      'right',
      'lateral',
      'KINOVEA_V1',
      '',
      '',
    ].join(',')
    const blob = new Blob([`\ufeff${header}\r\n${example}\r\n`], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'gold_standard_import_template.csv'
    anchor.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-5">
      <section className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div>
          <h2 className="flex items-center gap-2 text-base font-semibold text-white/90">
            <FlaskConical className="h-5 w-5 text-cyan-300" />
            金标准效度验证
          </h2>
          <p className="mt-1 text-xs text-white/40">
            Kinovea、双人盲评与运动捕捉数据统一预检、版本化和配对验证
          </p>
        </div>
        <div className="text-right text-xs text-white/45">
          <p>{threshold?.thresholdVersion || '正在读取阈值版本'}</p>
          {threshold && (
            <p className="mt-1 text-emerald-300/80">
              绿 {threshold.impactKneeAngleDeg.greenLow}–{threshold.impactKneeAngleDeg.greenHigh}° · 黄外沿{' '}
              {threshold.impactKneeAngleDeg.yellowLow}–{threshold.impactKneeAngleDeg.yellowHigh}°
            </p>
          )}
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-[1fr_auto] lg:items-center">
        <div className="rounded-lg border border-dashed border-cyan-400/25 bg-cyan-400/[0.04] p-5">
          <div className="flex flex-wrap items-center gap-3">
            <input
              ref={inputRef}
              type="file"
              accept=".csv,.xlsx,.json"
              className="hidden"
              onChange={(event) => {
                const file = event.target.files?.[0]
                if (file) void handleFile(file)
                event.currentTarget.value = ''
              }}
            />
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              disabled={busy !== null}
              className="inline-flex items-center gap-2 rounded-md bg-cyan-500 px-3 py-2 text-xs font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:opacity-50"
            >
              {busy === 'preview' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
              选择标注文件
            </button>
            <button
              type="button"
              onClick={downloadTemplate}
              className="inline-flex items-center gap-2 rounded-md border border-white/10 bg-white/5 px-3 py-2 text-xs text-white/70 transition hover:bg-white/10"
            >
              <Download className="h-4 w-4" />
              下载导入模板
            </button>
            <span className="text-xs text-white/35">支持 CSV、XLSX、JSON，单文件不超过 10 MB</span>
          </div>
        </div>
        {preview?.canCommit && (
          <button
            type="button"
            onClick={() => void commitPreview()}
            disabled={busy !== null}
            className="inline-flex items-center justify-center gap-2 rounded-md bg-emerald-500 px-4 py-3 text-xs font-semibold text-slate-950 disabled:opacity-50"
          >
            {busy === 'commit' ? <Loader2 className="h-4 w-4 animate-spin" /> : <FileCheck2 className="h-4 w-4" />}
            确认提交有效行
          </button>
        )}
      </section>

      {notice && (
        <div
          className={`flex items-center gap-2 rounded-md border px-3 py-2 text-xs ${
            notice.ok
              ? 'border-emerald-400/25 bg-emerald-400/10 text-emerald-200'
              : 'border-amber-400/25 bg-amber-400/10 text-amber-100'
          }`}
        >
          {notice.ok ? <CheckCircle2 className="h-4 w-4" /> : <AlertTriangle className="h-4 w-4" />}
          {notice.text}
        </div>
      )}

      {preview && (
        <section className="grid gap-3 border-y border-white/10 py-4 sm:grid-cols-4">
          {[
            ['总行数', preview.totalRows],
            ['有效', preview.validRowCount],
            ['排除', preview.invalidRowCount],
            ['重复', preview.duplicateCount],
          ].map(([label, value]) => (
            <div key={label} className="rounded-md bg-white/[0.04] px-3 py-3">
              <p className="text-[11px] text-white/40">{label}</p>
              <p className="mt-1 text-xl font-semibold tabular-nums text-white/90">{value}</p>
            </div>
          ))}
          {preview.invalidRows.length > 0 && (
            <div className="sm:col-span-4 max-h-36 overflow-auto rounded-md bg-rose-500/[0.06] p-3 text-xs text-rose-100/80">
              {preview.invalidRows.map((row) => (
                <p key={row.rowNumber}>第 {row.rowNumber} 行：{row.errors.join('；')}</p>
              ))}
            </div>
          )}
        </section>
      )}

      <section>
        <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-white/75">
          <Database className="h-4 w-4 text-emerald-300" />
          已提交数据集
        </h3>
        {datasets.length === 0 ? (
          <p className="rounded-md border border-dashed border-white/10 py-8 text-center text-xs text-white/35">
            暂无金标准数据集
          </p>
        ) : (
          <div className="overflow-x-auto rounded-md border border-white/10">
            <table className="w-full min-w-[820px] text-left text-xs">
              <thead className="bg-white/[0.04] text-white/40">
                <tr>
                  <th className="px-3 py-2 font-medium">数据集</th>
                  <th className="px-3 py-2 font-medium">来源文件</th>
                  <th className="px-3 py-2 font-medium">有效 / 排除</th>
                  <th className="px-3 py-2 font-medium">阈值版本</th>
                  <th className="px-3 py-2 text-right font-medium">操作</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {datasets.map((dataset) => (
                  <tr key={dataset.datasetId} className="text-white/70">
                    <td className="px-3 py-3 font-mono text-[11px] text-cyan-200">{dataset.datasetId}</td>
                    <td className="px-3 py-3">{dataset.sourceFilename}</td>
                    <td className="px-3 py-3 tabular-nums">{dataset.validRowCount} / {dataset.excludedRowCount}</td>
                    <td className="px-3 py-3 text-[11px]">{dataset.thresholdVersion}</td>
                    <td className="px-3 py-3 text-right">
                      <button
                        type="button"
                        onClick={() => void validateDataset(dataset.datasetId)}
                        disabled={busy !== null}
                        title="与本地归档 attempt 配对并生成效度报告"
                        className="inline-flex items-center gap-1.5 rounded-md border border-emerald-400/25 bg-emerald-400/10 px-2.5 py-1.5 text-emerald-200 disabled:opacity-50"
                      >
                        {busy === dataset.datasetId ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
                        计算效度
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {report && (
        <section className="border-t border-white/10 pt-4">
          <h3 className="mb-3 text-sm font-semibold text-white/75">效度报告 · {report.datasetId}</h3>
          {Object.entries(report.metrics).map(([metric, values]) => (
            <div key={metric} className="grid gap-3 sm:grid-cols-4 lg:grid-cols-7">
              {[
                ['配对样本', values.n],
                ['MAE', values.mae],
                ['RMSE', values.rmse],
                ['偏倚', values.bias],
                ['ICC(A,1)', values.iccAbsoluteAgreement ?? '样本不足'],
                ['分级一致率', values.levelAccuracy === null ? '无' : `${(values.levelAccuracy * 100).toFixed(1)}%`],
                ['触球帧 MAE', values.impactFrameMae ?? '无'],
              ].map(([label, value]) => (
                <div key={label} className="rounded-md bg-white/[0.04] p-3">
                  <p className="text-[10px] text-white/35">{label}</p>
                  <p className="mt-1 text-sm font-semibold tabular-nums text-white/85">{value}</p>
                </div>
              ))}
            </div>
          ))}
          <p className="mt-3 text-xs text-white/40">
            对齐 {report.alignedSampleCount} 个，排除 {report.excludedSampleCount} 个；双标注样本{' '}
            {report.interRater.multiAnnotatorSampleCount} 个，标注员平均绝对差{' '}
            {report.interRater.meanAbsoluteDifference ?? '样本不足'}°；视频哈希已核验{' '}
            {report.hashVerification.verifiedCount} 个、缺少历史哈希 {report.hashVerification.unavailableCount} 个。
          </p>
        </section>
      )}
    </div>
  )
}
