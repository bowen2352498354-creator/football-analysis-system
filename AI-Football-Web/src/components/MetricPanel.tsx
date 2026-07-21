import { Gauge } from 'lucide-react'
import MetricCardList from './MetricCardList'
import BiomechanicalRadar from './BiomechanicalRadar'
import type {
  MetricRenderMode,
  MetricSeekEvent,
  Quantified5dScores,
  ScoreDetailPayload,
  AcademicExportResult,
} from '../types'

export interface MetricPanelProps {
  /** 实验组差异化渲染模式 */
  renderMode?: MetricRenderMode
  /** DeterministicScorer 完整 scoreDetail（优先） */
  scoreDetail?: ScoreDetailPayload | null
  /** 兼容旧扁平 8 大指标字典 */
  metrics?: Record<string, number | boolean | null> | null
  /** 命中的错误代码列表 */
  errorCodes?: string[] | null
  /** 五维雷达评分 */
  radarScores?: Quantified5dScores | null
  /** GROUP_B 幽灵骨架对比用第二次 Attempt */
  compareRadarScores?: Quantified5dScores | null
  /** GROUP_A 大字号具身隐喻导语 */
  groupALeadMetaphor?: string | null
  /** 触球零点帧 */
  tImpact?: number | null
  /** 点击卡片 → VideoWorkspace Seek */
  onMetricSeek?: (event: MetricSeekEvent) => void
  onExportSuccess?: (result: AcademicExportResult) => void
  onExportError?: (message: string) => void
  /** Sprint 1：时空热力图 base64 */
  heatmapBase64?: string | null
  className?: string
}

/**
 * V2.5 左栏 MetricPanel（28%）
 * 委托 MetricCardList 渲染 8 大量纲 + 三实验组模式；
 * GROUP_A/COACH 下额外保留紧凑雷达（GROUP_B 雷达已内置于切换面板）。
 */
export default function MetricPanel({
  renderMode = 'GROUP_A',
  scoreDetail = null,
  metrics = null,
  errorCodes = null,
  radarScores = null,
  compareRadarScores = null,
  groupALeadMetaphor = null,
  tImpact = null,
  onMetricSeek,
  onExportSuccess,
  onExportError,
  heatmapBase64 = null,
  className = '',
}: MetricPanelProps) {
  const showFooterRadar = renderMode !== 'GROUP_B'

  return (
    <aside
      className={`workbench-col workbench-card overflow-hidden ${className}`.trim()}
      aria-label="生物力学指标面板"
    >
      <div className="flex h-full min-h-0 flex-col">
        <header className="flex flex-shrink-0 items-center gap-2 border-b border-slate-700/80 px-3 py-2.5">
          <span className="inline-flex flex-shrink-0 text-[var(--GREEN_OPTIMAL)]">
            <Gauge className="h-4 w-4" />
          </span>
          <div className="min-w-0">
            <h2 className="truncate text-sm font-semibold text-slate-100">Metric Panel</h2>
            <p className="truncate text-[10px] text-slate-400">
              {renderMode === 'GROUP_A' && '实验A组 · 具身隐喻极简卡'}
              {renderMode === 'GROUP_B' && '实验B组 · 雷达 / 幽灵骨架 / 空间热力图'}
              {renderMode === 'COACH_CONSOLE' && '教练科研控制台 · SPSS 宽表'}
            </p>
          </div>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden px-3 py-3">
          <MetricCardList
            renderMode={renderMode}
            scoreDetail={scoreDetail}
            metrics={metrics}
            errorCodes={errorCodes}
            radarScores={radarScores}
            compareRadarScores={compareRadarScores}
            groupALeadMetaphor={groupALeadMetaphor}
            tImpact={tImpact}
            onMetricSeek={onMetricSeek}
            onExportSuccess={onExportSuccess}
            onExportError={onExportError}
            heatmapBase64={
              heatmapBase64 ?? scoreDetail?.heatmap_base64 ?? null
            }
          />

          {showFooterRadar && (
            <div className="mt-4 border-t border-slate-700/60 pt-4">
              <div className="mb-2 text-[11px] font-semibold text-slate-400">五维量化雷达</div>
              <BiomechanicalRadar
                scores={radarScores ?? scoreDetail?.radar_scores ?? null}
                compact
                className="!text-slate-100"
              />
            </div>
          )}
        </div>
      </div>
    </aside>
  )
}
