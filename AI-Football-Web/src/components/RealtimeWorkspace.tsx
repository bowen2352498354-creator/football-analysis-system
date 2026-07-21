import { useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Play,
  Square,
  ScanFace,
  Bone,
  Camera,
  FileVideo,
  UploadCloud,
  User,
  Download,
  GraduationCap,
  Loader2,
  Wifi,
  WifiOff,
  AlertTriangle,
  Gauge,
  Crosshair,
  Save,
  CheckCircle2,
  XCircle,
} from 'lucide-react'
import { useTypewriter } from '../hooks/useTypewriter'
import {
  appendGlobalRecordToLocalStorage,
  getClassGroupDisplayName,
  getSchoolDisplayName,
  getSmoothAngleBackground,
  LEVEL_COLOR_MAP,
  LEVEL_LABEL_MAP,
  MOCK_RADAR_SCORES,
  MOCK_RADAR_SCORES_COMPARE,
  MOCK_SCORE_DETAIL_V31,
} from '../mockData'
import type {
  AnalysisStatus,
  FinalDiagnosisReport,
  GlobalSettings,
  GlobalTrainingRecord,
  ThresholdHitStats,
  ThresholdLevel,
  VideoSourceMode,
} from '../types'
import MetricPanel from './MetricPanel'
import SynchronizedVideoWorkspace, {
  type SyncVelocityPoint,
} from './SynchronizedVideoWorkspace'
import AIAssistantPanel from './AIAssistantPanel'
import { TRAFFIC_LIGHT } from '../theme/trafficLight'

interface RealtimeWorkspaceProps {
  /** 来自 Navbar 的全局教学环境设置（学校 + 班级/组别），本工作台只读消费 */
  globalSettings: GlobalSettings
}

/* ============================================================================
 * 【真刀真枪联调配置】后台服务网关地址。
 * api_server.py 默认监听 8000 端口，启动方式见该文件顶部注释：
 *     python api_server.py
 * 或   uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
 * ========================================================================== */
const API_BASE_URL = 'http://localhost:8000'
const WS_ANALYZE_URL = 'ws://localhost:8000/ws/analyze'

/** 后台推送的三级容错状态字符串（与 pose_tracker.py judge_knee_status 保持一致） */
type BackendStatus = 'Green' | 'Yellow' | 'Red'

/** 后端 WebSocket 推送的消息结构（与 api_server.py 的协议保持一致） */
interface WsFrameMessage {
  type: 'frame'
  image: string
  angle: number | null
  status: BackendStatus | null
  /** 【新增】后端实时计算出的右膝角速度（deg/s），用于左侧动力链角速度监控波形图 */
  angular_velocity: number | null
  /** 【新增】后端基于滑动窗口角速度离散程度计算出的动平衡稳定指数（0-100） */
  stability_index: number | null
  timestamp: number
}
interface WsStartedMessage {
  type: 'started'
  session_id: string
}
interface WsStoppedMessage {
  type: 'stopped'
  session_id: string
  total_records: number
}
interface WsErrorMessage {
  type: 'error'
  message: string
}
/** 【新增】非致命诊断提醒（例如自动检测到摄像头持续输出全黑画面），不会中断分析会话 */
interface WsNoticeMessage {
  type: 'notice'
  message: string
}
type WsMessage = WsFrameMessage | WsStartedMessage | WsStoppedMessage | WsErrorMessage | WsNoticeMessage

/** 「自动归档并生成 Word 报告」按钮的当前状态 */
type WordSaveStatus = 'idle' | 'saving' | 'success' | 'error'

/** 右上角浮动 Toast 提示条的展示内容（绿色成功 / 红色失败） */
interface WordSaveToastState {
  id: number
  message: string
  success: boolean
}

let wordSaveToastSeq = 0

/** 三级阈值命中次数柱状条对应的背景色类名（对齐 Traffic-Light） */
const HIT_BAR_BG: Record<ThresholdLevel, string> = {
  green: 'bg-[var(--GREEN_OPTIMAL)]',
  yellow: 'bg-[var(--YELLOW_APPROACHING)]',
  red: 'bg-[var(--RED_DEVIATED)]',
}

/** 环形图各分段对应的描边颜色（十六进制，供 SVG stroke 使用） */
const RING_STROKE: Record<ThresholdLevel, string> = {
  green: TRAFFIC_LIGHT.GREEN_OPTIMAL,
  yellow: TRAFFIC_LIGHT.YELLOW_APPROACHING,
  red: TRAFFIC_LIGHT.RED_DEVIATED,
}

/** 运行状态展示文案与颜色映射 */
const STATUS_META: Record<AnalysisStatus, { label: string; className: string }> = {
  idle: { label: '待机中', className: 'text-white/50' },
  analyzing: { label: '分析进行中', className: 'text-amber-300' },
  stopping: { label: '正在结束并生成报告…', className: 'text-amber-300' },
  finished: { label: '本次分析已结束', className: 'text-emerald-300' },
}

/** 把后端返回的 Green/Yellow/Red 状态字符串，映射为前端展示用的小写等级标识 */
function statusToLevel(status: BackendStatus | null): ThresholdLevel | null {
  if (status === 'Green') return 'green'
  if (status === 'Yellow') return 'yellow'
  if (status === 'Red') return 'red'
  return null
}

/**
 * 实时反馈系统工作台（实验A组）：
 * 左侧 ~62% 视觉分析展示区 + 右侧 ~38% 结构化诊断报告便当盒。
 *
 * 【v1.1 全栈联调版】本组件已彻底移除所有前端 Mock 模拟逻辑：
 *   - 左侧视口通过 WebSocket 订阅 api_server.py 实时推送的、已经在后端完成
 *     骨骼渲染 + 三级容错染色 + 面部高斯模糊打码的真实视频帧（Base64 JPEG）；
 *   - 右膝角度监控卡直接绑定后端实时计算出的真实角度与状态；
 *   - 「结束分析」会真正调用后端 /api/generate_report 接口，由 DeepSeek 大模型
 *     生成本次训练的真实综合诊断报告。
 */
export default function RealtimeWorkspace({ globalSettings }: RealtimeWorkspaceProps) {
  /* ---------------------------- 顶栏控制区状态 ---------------------------- */
  const [videoSourceMode, setVideoSourceMode] = useState<VideoSourceMode>('webcam')
  const [localVideoFile, setLocalVideoFile] = useState<File | null>(null)
  const [uploadedVideoPath, setUploadedVideoPath] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [studentNumber, setStudentNumber] = useState('B004')
  const [analysisStatus, setAnalysisStatus] = useState<AnalysisStatus>('idle')
  const fileInputRef = useRef<HTMLInputElement>(null)

  /* ---------------------------- 后端实时连接与画面状态 ---------------------------- */
  const wsRef = useRef<WebSocket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  /** 【新增】非致命诊断提醒（如摄像头疑似全黑画面），单独用黄色提示条展示，不影响分析状态 */
  const [diagnosticNotice, setDiagnosticNotice] = useState<string | null>(null)
  const [frameImage, setFrameImage] = useState<string | null>(null)
  const [kneeAngle, setKneeAngle] = useState<number | null>(null)
  const [backendStatus, setBackendStatus] = useState<BackendStatus | null>(null)

  /* ---------------------------- 实时动力链角速度监控状态 ---------------------------- */
  /** 全程角速度轨迹（frame_index 递增），供 Kinovea 联动波形图使用 */
  const [omegaSeries, setOmegaSeries] = useState<SyncVelocityPoint[]>([])
  const omegaFrameRef = useRef(0)
  const [stabilityIndex, setStabilityIndex] = useState<number | null>(null)
  /** 本地视频 blob URL，供 HTML5 Video 与波形图毫秒级 scrub */
  const [localVideoObjectUrl, setLocalVideoObjectUrl] = useState<string | null>(null)

  /* ---------------------------- 诊断统计与最终报告状态 ---------------------------- */
  const [hitStats, setHitStats] = useState<ThresholdHitStats>({ green: 0, yellow: 0, red: 0 })
  const [finalReport, setFinalReport] = useState<FinalDiagnosisReport | null>(null)
  const [isGeneratingReport, setIsGeneratingReport] = useState(false)
  /** 最近一次分析会话 ID，供手绘批注归档关联 */
  const [lastSessionId, setLastSessionId] = useState<string | null>(null)

  /* ---------------------------- 本地归档 + Word 报告生成状态 ---------------------------- */
  const [wordSaveStatus, setWordSaveStatus] = useState<WordSaveStatus>('idle')
  const [wordSaveToast, setWordSaveToast] = useState<WordSaveToastState | null>(null)

  const isAnalyzing = analysisStatus === 'analyzing'
  const level = useMemo(() => statusToLevel(backendStatus), [backendStatus])
  const smoothBackground = useMemo(
    () => getSmoothAngleBackground(kneeAngle ?? 150),
    [kneeAngle],
  )
  const { displayText: reportDisplayText, isDone: isReportDone } = useTypewriter(finalReport?.fullText ?? '', 22)

  const totalAttempts = hitStats.green + hitStats.yellow + hitStats.red

  /** 报告返回的触球窗口内索引；分析中仍用全程绝对帧 */
  const impactIndexInWindow = useMemo(() => {
    const idx = finalReport?.impact_index_in_window ?? finalReport?.impactIndexInWindow
    return typeof idx === 'number' && Number.isFinite(idx) ? idx : null
  }, [finalReport])

  /** 窗口序列第 0 帧对应的绝对视频帧 = t_impact - impact_index_in_window */
  const seriesFrameOffset = useMemo(() => {
    if (impactIndexInWindow === null) return 0
    const tAbs = finalReport?.t_impact ?? finalReport?.tImpact
    if (typeof tAbs !== 'number' || !Number.isFinite(tAbs)) return 0
    return Math.max(0, Math.round(tAbs) - Math.round(impactIndexInWindow))
  }, [finalReport, impactIndexInWindow])

  /**
   * 统一的 WebSocket 消息处理：每一条推理帧都会实时更新角度/状态/画面，
   * 收到 "stopped"（无论是用户主动结束，还是本地视频自然播放完毕）后，
   * 自动触发真实的 DeepSeek 综合报告生成请求。
   */
  function handleWsMessage(event: MessageEvent<string>) {
    let message: WsMessage
    try {
      message = JSON.parse(event.data) as WsMessage
    } catch {
      return
    }

    if (message.type === 'frame') {
      // 【容错防呆】只有拿到规范的 "data:image/..." 格式字符串才更新画面；
      // 万一某一条消息的 image 字段异常（空字符串/undefined/格式不对），
      // 直接忽略这一帧、保留上一帧画面，绝不能让 <img src="" /> 这种空/非法
      // src 把画面"闪"成黑屏——这正是之前排查"点击开始分析后黑屏"问题的关键点。
      if (typeof message.image === 'string' && message.image.startsWith('data:image')) {
        setFrameImage(message.image)
      }

      if (message.angle !== null && message.status !== null) {
        setKneeAngle(message.angle)
        setBackendStatus(message.status)
        const nextLevel = statusToLevel(message.status)
        if (nextLevel) {
          setHitStats((stats) => ({ ...stats, [nextLevel]: stats[nextLevel] + 1 }))
        }
      }

      // Kinovea 联动：累积全程角速度时序（横轴 = frame_index）
      if (typeof message.angular_velocity === 'number' && Number.isFinite(message.angular_velocity)) {
        const nextVelocity = message.angular_velocity
        const frameIndex = omegaFrameRef.current
        omegaFrameRef.current = frameIndex + 1
        setOmegaSeries((prev) => [...prev, { frame_index: frameIndex, omega: nextVelocity }])
      }
      if (typeof message.stability_index === 'number' && Number.isFinite(message.stability_index)) {
        setStabilityIndex(message.stability_index)
      }
      return
    }

    if (message.type === 'started') {
      setLastSessionId(message.session_id)
      return
    }

    if (message.type === 'stopped') {
      wsRef.current?.close()
      wsRef.current = null
      setIsConnected(false)
      setLastSessionId(message.session_id)
      void fetchGeneratedReport(message.session_id)
      return
    }

    if (message.type === 'error') {
      setConnectionError(message.message)
      setAnalysisStatus('idle')
      wsRef.current?.close()
      wsRef.current = null
      setIsConnected(false)
      return
    }

    if (message.type === 'notice') {
      // 非致命提醒：只展示提示条，绝不中断当前分析会话、绝不关闭 WebSocket 连接
      setDiagnosticNotice(message.message)
    }
  }

  /** 真正调用后端 /api/generate_report 接口，由 DeepSeek 生成本次综合诊断报告 */
  async function fetchGeneratedReport(sessionId: string) {
    setAnalysisStatus('stopping')
    setIsGeneratingReport(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/generate_report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, student_number: studentNumber }),
      })
      if (!response.ok) throw new Error(`报告接口返回状态码 ${response.status}`)
      const report = (await response.json()) as FinalDiagnosisReport & { hitStats: ThresholdHitStats }
      setFinalReport(report)
      setHitStats(report.hitStats)

      // Sprint 1：用 Action ROI 鞭打发力窗口替换实时累积的全程序列
      const windowSeries = report.time_series_velocity ?? report.timeSeriesVelocity
      if (Array.isArray(windowSeries) && windowSeries.length > 0) {
        setOmegaSeries(
          windowSeries.map((omega, index) => ({
            frame_index: index,
            omega: typeof omega === 'number' && Number.isFinite(omega) ? omega : 0,
          })),
        )
      }
    } catch (error) {
      setConnectionError(error instanceof Error ? error.message : '生成诊断报告失败，请检查后端服务是否已启动。')
    } finally {
      setIsGeneratingReport(false)
      setAnalysisStatus('finished')
    }
  }

  /** 组件卸载时，确保 WebSocket 连接被妥善关闭，不留下悬空连接 */
  useEffect(() => {
    return () => {
      wsRef.current?.close()
      wsRef.current = null
    }
  }, [])

  /** 本地视频文件 → Object URL，供 HTML5 Video 与波形图 scrub 同步 */
  useEffect(() => {
    if (!localVideoFile) {
      setLocalVideoObjectUrl(null)
      return
    }
    const url = URL.createObjectURL(localVideoFile)
    setLocalVideoObjectUrl(url)
    return () => {
      URL.revokeObjectURL(url)
    }
  }, [localVideoFile])

  /**
   * 【全局归档总闸】只要 Navbar 全局环境设置中的「本地落盘归档」开关处于开启状态，
   * 一旦 DeepSeek 生成的最终诊断报告就位，就无需等待教练手动点击，
   * 自动静默调用 handleSaveWordReport() 完成本机硬盘写盘 + 全局数据库同步。
   */
  useEffect(() => {
    if (finalReport && globalSettings.enableDataArchiving && wordSaveStatus === 'idle') {
      void handleSaveWordReport()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finalReport, globalSettings.enableDataArchiving])

  function handleSelectVideoFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return
    setLocalVideoFile(file)
    setUploadedVideoPath(null)
    void uploadVideoFile(file)
  }

  /** 把用户选择的本地 MP4 文件真正上传到后端 /api/upload_video，换回后端文件路径 */
  async function uploadVideoFile(file: File) {
    setIsUploading(true)
    setConnectionError(null)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const response = await fetch(`${API_BASE_URL}/api/upload_video`, { method: 'POST', body: formData })
      if (!response.ok) throw new Error(`上传接口返回状态码 ${response.status}`)
      const data = (await response.json()) as { video_path: string }
      setUploadedVideoPath(data.video_path)
    } catch (error) {
      setConnectionError(error instanceof Error ? error.message : '视频上传失败，请检查后端服务是否已启动。')
    } finally {
      setIsUploading(false)
    }
  }

  function handleStart() {
    setConnectionError(null)
    setDiagnosticNotice(null)
    setFinalReport(null)
    setHitStats({ green: 0, yellow: 0, red: 0 })
    setKneeAngle(null)
    setBackendStatus(null)
    setFrameImage(null)
    setOmegaSeries([])
    omegaFrameRef.current = 0
    setStabilityIndex(null)
    setWordSaveStatus('idle')

    const socket = new WebSocket(WS_ANALYZE_URL)
    wsRef.current = socket

    socket.onopen = () => {
      setIsConnected(true)
      socket.send(
        JSON.stringify({
          action: 'start',
          source: videoSourceMode === 'file' ? 'file' : 'webcam',
          video_path: videoSourceMode === 'file' ? uploadedVideoPath : undefined,
        }),
      )
      setAnalysisStatus('analyzing')
    }
    socket.onmessage = handleWsMessage
    socket.onerror = () => {
      setConnectionError('无法连接到后台服务，请确认 api_server.py 已在 8000 端口启动。')
      setAnalysisStatus('idle')
      setIsConnected(false)
    }
    socket.onclose = () => {
      wsRef.current = null
      setIsConnected(false)
    }
  }

  function handleStop() {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'stop' }))
      setAnalysisStatus('stopping')
    }
  }

  function handleExportReport() {
    if (!finalReport) return
    const payload = {
      exportedAt: new Date().toISOString(),
      studentNumber,
      school: getSchoolDisplayName(globalSettings),
      classGroup: getClassGroupDisplayName(globalSettings),
      hitStats,
      report: finalReport,
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `诊断报告_${studentNumber || 'unknown'}_${Date.now()}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  /** 显示右上角浮动 Toast 提示，3.2 秒后自动淡出消失 */
  function showWordSaveToast(message: string, success: boolean) {
    const id = ++wordSaveToastSeq
    setWordSaveToast({ id, message, success })
    window.setTimeout(() => {
      setWordSaveToast((current) => (current?.id === id ? null : current))
    }, 3200)
  }

  /**
   * 「自动归档并生成 Word 报告」核心操作：把当前学生档案 + AI 诊断报告 + 关键帧图片
   * Base64 一并 POST 给后台 /api/save_word_report，由 word_reporter.py 在本机硬盘上
   * 完成"建文件夹 + 写 .docx"两件事，绝不依赖浏览器的直接下载。
   */
  async function handleSaveWordReport() {
    if (!finalReport) return
    setWordSaveStatus('saving')
    try {
      const response = await fetch(`${API_BASE_URL}/api/save_word_report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: 'realtime',
          school: getSchoolDisplayName(globalSettings),
          classGroup: getClassGroupDisplayName(globalSettings),
          studentNumber: studentNumber || '未填写编号',
          score: finalReport.score,
          totalAttempts: finalReport.totalAttempts,
          painPoint: finalReport.painPoint,
          prescription: finalReport.prescription,
          generatedAt: finalReport.generatedAt,
          impactFrameImage: finalReport.impactFrameImage ?? null,
          heatmapBase64:
            finalReport.heatmap_base64 ??
            finalReport.heatmapBase64 ??
            finalReport.scoreDetail?.heatmap_base64 ??
            null,
          heatmap_base64:
            finalReport.heatmap_base64 ??
            finalReport.heatmapBase64 ??
            finalReport.scoreDetail?.heatmap_base64 ??
            null,
          hitStats: finalReport.hitStats ?? hitStats,
          kneeFlexionAngle: finalReport.avgKneeAngle ?? null,
          scoreDetail: finalReport.scoreDetail ?? null,
        }),
      })
      const data = (await response.json()) as {
        success: boolean
        message: string
        path?: string
        record?: GlobalTrainingRecord
      }
      if (!response.ok || !data.success) throw new Error(data.message || `接口返回状态码 ${response.status}`)
      setWordSaveStatus('success')
      showWordSaveToast(`✅ ${data.message}`, true)
      // 【双向同步全局数据库】后端已经把这条记录追加进 global_training_db.json，
      // 这里把同一份记录同步写进 localStorage 作为极速双保险，供教练端看板兜底读取。
      if (data.record) {
        appendGlobalRecordToLocalStorage(data.record)
      }
    } catch (error) {
      setWordSaveStatus('error')
      const message = error instanceof Error ? error.message : '保存 Word 报告失败，请检查后端服务是否已启动。'
      showWordSaveToast(`⚠️ 自动归档失败：${message}`, false)
    }
  }

  const isStartDisabled =
    isAnalyzing ||
    analysisStatus === 'stopping' ||
    (videoSourceMode === 'file' && (!uploadedVideoPath || isUploading))

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      {/* ============================ 顶栏控制区（紧凑工具条） ============================ */}
      <section className="workbench-toolbar workbench-card mx-3 mt-2 flex flex-shrink-0 flex-col gap-2 px-3 py-2 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
        <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center sm:gap-2">
          <div className="inline-flex items-center gap-1 self-start rounded-full bg-slate-900/60 p-0.5">
            {(
              [
                { id: 'webcam' as VideoSourceMode, label: '实时摄像头', icon: Camera },
                { id: 'file' as VideoSourceMode, label: '本地视频', icon: FileVideo },
              ] as const
            ).map((option) => {
              const active = option.id === videoSourceMode
              const Icon = option.icon
              return (
                <button
                  key={option.id}
                  type="button"
                  disabled={isAnalyzing}
                  onClick={() => setVideoSourceMode(option.id)}
                  className={`relative rounded-full px-2.5 py-1.5 text-[11px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 sm:text-xs ${
                    active ? 'text-slate-100' : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {active && (
                    <motion.span
                      layoutId="video-source-pill"
                      className="absolute inset-0 rounded-full bg-[color-mix(in_srgb,var(--GREEN_OPTIMAL)_22%,transparent)] ring-1 ring-[color-mix(in_srgb,var(--GREEN_OPTIMAL)_40%,transparent)]"
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

          {videoSourceMode === 'file' && (
            <>
              <button
                type="button"
                disabled={isAnalyzing || isUploading}
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900/40 px-3 py-1.5 text-[11px] text-slate-300 transition hover:bg-slate-800 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50 sm:text-xs"
              >
                <span className="inline-flex flex-shrink-0">
                  {isUploading ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-sky-400" />
                  ) : (
                    <UploadCloud className="h-3.5 w-3.5 text-sky-400" />
                  )}
                </span>
                <span className="max-w-[9rem] truncate sm:max-w-[14rem]">
                  {isUploading
                    ? '上传中…'
                    : localVideoFile
                      ? `${localVideoFile.name}${uploadedVideoPath ? ' ✓' : ''}`
                      : '选择本地 MP4'}
                </span>
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="video/mp4,video/*"
                className="hidden"
                onChange={handleSelectVideoFile}
              />
            </>
          )}

          <div className="flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900/40 px-3 py-1.5">
            <User className="h-3.5 w-3.5 flex-shrink-0 text-slate-500" />
            <input
              type="text"
              value={studentNumber}
              onChange={(e) => setStudentNumber(e.target.value)}
              placeholder='编号/姓名'
              className="w-28 bg-transparent text-xs text-slate-100 outline-none placeholder:text-slate-500 sm:w-40"
            />
          </div>

          <span className={`text-[11px] ${STATUS_META[analysisStatus].className}`}>
            {STATUS_META[analysisStatus].label}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {!isAnalyzing && analysisStatus !== 'stopping' ? (
            <button
              type="button"
              onClick={handleStart}
              disabled={isStartDisabled}
              title={isStartDisabled ? '请先选择并等待本地 MP4 文件上传完成' : undefined}
              className="flex items-center gap-1.5 rounded-full bg-[var(--GREEN_OPTIMAL)] px-4 py-1.5 text-xs font-semibold text-slate-950 transition hover:brightness-110 active:scale-95 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
            >
              <Play className="h-3.5 w-3.5" />
              开始分析
            </button>
          ) : (
            <button
              type="button"
              onClick={handleStop}
              disabled={analysisStatus === 'stopping'}
              className="flex items-center gap-1.5 rounded-full bg-[var(--RED_DEVIATED)] px-4 py-1.5 text-xs font-semibold text-white transition hover:brightness-110 active:scale-95 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {analysisStatus === 'stopping' ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Square className="h-3.5 w-3.5" />
              )}
              结束分析
            </button>
          )}
        </div>
      </section>

      {(connectionError || diagnosticNotice) && (
        <div className="mx-3 mt-2 flex flex-shrink-0 flex-col gap-1.5">
          {connectionError && (
            <div className="flex items-center gap-2 rounded-xl border border-[color-mix(in_srgb,var(--RED_DEVIATED)_35%,transparent)] bg-[color-mix(in_srgb,var(--RED_DEVIATED)_12%,transparent)] px-3 py-2 text-xs text-[var(--RED_DEVIATED)]">
              <AlertTriangle className="h-3.5 w-3.5 flex-shrink-0" />
              <p className="min-w-0 truncate">{connectionError}</p>
            </div>
          )}
          {diagnosticNotice && (
            <div className="flex items-start gap-2 rounded-xl border border-[color-mix(in_srgb,var(--YELLOW_APPROACHING)_35%,transparent)] bg-[color-mix(in_srgb,var(--YELLOW_APPROACHING)_12%,transparent)] px-3 py-2 text-xs text-[var(--YELLOW_APPROACHING)]">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
              <p className="min-w-0 leading-relaxed">{diagnosticNotice}</p>
            </div>
          )}
        </div>
      )}

      {/* ============================ V2.5 三栏沉浸式 Grid：28% / 44% / 28% ============================ */}
      <div className="workbench-grid">
        <MetricPanel
          renderMode="GROUP_B"
          scoreDetail={finalReport?.scoreDetail ?? MOCK_SCORE_DETAIL_V31}
          metrics={null}
          errorCodes={null}
          radarScores={finalReport?.scoreDetail?.radar_scores ?? MOCK_RADAR_SCORES}
          compareRadarScores={MOCK_RADAR_SCORES_COMPARE}
          tImpact={finalReport?.tImpact ?? finalReport?.t_impact ?? null}
          heatmapBase64={
            finalReport?.heatmap_base64 ??
            finalReport?.heatmapBase64 ??
            finalReport?.scoreDetail?.heatmap_base64 ??
            null
          }
        />

        <SynchronizedVideoWorkspace
          videoSrc={videoSourceMode === 'file' ? localVideoObjectUrl : null}
          velocitySeries={omegaSeries}
          tImpact={finalReport?.tImpact ?? finalReport?.t_impact ?? null}
          impactIndexInWindow={impactIndexInWindow}
          seriesFrameOffset={seriesFrameOffset}
          fps={30}
          preferLiveOverlay={isAnalyzing || analysisStatus === 'stopping'}
          title="Video Workspace"
          subtitle="鞭打发力角速度时序 · 触球窗口 t_impact±30 · 教练手绘电烙铁"
          studentNumber={studentNumber}
          attemptId={lastSessionId}
          overlay={
            <>
              <motion.div
                animate={{ backgroundColor: smoothBackground }}
                transition={{ duration: 0.8, ease: 'easeInOut' }}
                className={`pointer-events-auto absolute top-3 left-3 rounded-xl border border-white/10 px-3 py-2.5 shadow-lg backdrop-blur-xl ${
                  level ? LEVEL_COLOR_MAP[level].glow : ''
                }`}
              >
                <p className="text-[10px] font-medium text-white/70">右膝角度</p>
                <p className="text-2xl font-bold tabular-nums text-white">
                  {kneeAngle !== null ? `${kneeAngle}°` : '--'}
                </p>
                <span className="mt-0.5 inline-block rounded-full bg-black/30 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-white/85">
                  {level ? LEVEL_LABEL_MAP[level] : '等待人体'}
                </span>
              </motion.div>

              <div className="absolute top-3 right-3 flex flex-col items-end gap-1.5">
                <span className="flex items-center gap-1.5 rounded-full border border-white/10 bg-[color-mix(in_srgb,var(--GREEN_OPTIMAL)_18%,transparent)] px-2.5 py-1 text-[10px] text-[var(--GREEN_OPTIMAL)] backdrop-blur-xl">
                  <ScanFace className="h-3 w-3" />
                  隐私打码
                </span>
                <span className="flex items-center gap-1.5 rounded-full border border-white/10 bg-sky-500/20 px-2.5 py-1 text-[10px] text-sky-300 backdrop-blur-xl">
                  <Bone className="h-3 w-3" />
                  骨骼渲染
                </span>
                <span
                  className={`flex items-center gap-1.5 rounded-full border border-white/10 px-2.5 py-1 text-[10px] backdrop-blur-xl ${
                    isConnected
                      ? 'bg-[color-mix(in_srgb,var(--GREEN_OPTIMAL)_12%,transparent)] text-[var(--GREEN_OPTIMAL)]'
                      : 'bg-black/30 text-slate-400'
                  }`}
                >
                  {isConnected ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}
                  {isConnected ? '已连接' : '未连接'}
                </span>
                {stabilityIndex !== null && (
                  <span className="flex items-center gap-1.5 rounded-full border border-white/10 bg-black/40 px-2.5 py-1 text-[10px] text-slate-300 backdrop-blur-xl">
                    <Gauge className="h-3 w-3 text-[var(--GREEN_OPTIMAL)]" />
                    稳定指数 {stabilityIndex}
                  </span>
                )}
              </div>

              {finalReport?.impactFrameImage && (
                <div className="absolute bottom-3 left-3 w-28 overflow-hidden rounded-lg border border-slate-600/60 bg-black/70 shadow-lg">
                  <img
                    src={finalReport.impactFrameImage}
                    alt="击球关键帧"
                    className="aspect-video w-full object-cover"
                  />
                  <span className="absolute left-1 top-1 flex items-center gap-1 rounded bg-black/70 px-1.5 py-0.5 text-[8px] text-[var(--GREEN_OPTIMAL)]">
                    <Crosshair className="h-2.5 w-2.5" />
                    Impact
                  </span>
                </div>
              )}

              <AnimatePresence>
                {isGeneratingReport && (
                  <motion.div
                    initial={{ opacity: 0, y: 20, scale: 0.9 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 20, scale: 0.9 }}
                    transition={{ type: 'spring', stiffness: 260, damping: 24 }}
                    className="pointer-events-auto absolute bottom-4 left-1/2 flex w-[88%] max-w-md -translate-x-1/2 items-center gap-2 rounded-full border border-white/10 bg-black/75 px-4 py-2.5 shadow-2xl backdrop-blur-xl"
                  >
                    <Loader2 className="h-3.5 w-3.5 flex-shrink-0 animate-spin text-[var(--GREEN_OPTIMAL)]" />
                    <p className="text-xs text-slate-100">DeepSeek 正在生成诊断报告…</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </>
          }
        >
          <div className="relative h-full min-h-0 w-full overflow-hidden bg-gradient-to-br from-slate-900 via-black to-slate-950">
            {frameImage ? (
              <img
                src={frameImage}
                alt="实时推理画面"
                className="absolute inset-0 h-full w-full bg-black object-contain"
                onError={() => setFrameImage(null)}
              />
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-slate-500">
                <span className="inline-flex flex-shrink-0">
                  {videoSourceMode === 'file' ? (
                    <FileVideo className="h-12 w-12" />
                  ) : (
                    <Camera className="h-12 w-12" />
                  )}
                </span>
                <p className="max-w-xs px-4 text-center text-xs leading-relaxed">
                  {videoSourceMode === 'file'
                    ? uploadedVideoPath
                      ? localVideoObjectUrl && analysisStatus === 'finished'
                        ? '分析结束：可在下方波形图拖拽/点击，同步跳转视频帧'
                        : '视频已就绪，点击「开始分析」后显示后端实时推理画面'
                      : '请先选择本地 MP4 并等待上传完成'
                    : '点击「开始分析」后，后端将推送实时推理画面'}
                </p>
              </div>
            )}
          </div>
        </SynchronizedVideoWorkspace>

        <AIAssistantPanel
          report={finalReport}
          hitStats={hitStats}
          errorCodes={null}
          displayText={
            finalReport
              ? reportDisplayText
              : isAnalyzing
                ? level
                  ? `右膝角度 ${kneeAngle}°，判定为「${LEVEL_LABEL_MAP[level]}」，累计已采集 ${totalAttempts} 次有效数据。`
                  : '正在等待后端检测到完整人体姿态……'
                : undefined
          }
          isTyping={!!finalReport && !isReportDone}
          actions={
            finalReport ? (
              <div className="flex flex-col gap-2">
                <button
                  type="button"
                  onClick={handleSaveWordReport}
                  disabled={wordSaveStatus === 'saving'}
                  className={`flex items-center justify-center gap-2 rounded-xl py-2.5 text-xs font-bold transition active:scale-95 disabled:cursor-not-allowed disabled:opacity-70 ${
                    wordSaveStatus === 'success'
                      ? 'bg-[var(--GREEN_OPTIMAL)] text-slate-950'
                      : wordSaveStatus === 'error'
                        ? 'bg-[var(--RED_DEVIATED)] text-white'
                        : 'bg-[var(--GREEN_OPTIMAL)] text-slate-950 hover:brightness-110'
                  }`}
                >
                  {wordSaveStatus === 'saving' ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : wordSaveStatus === 'success' ? (
                    <CheckCircle2 className="h-3.5 w-3.5" />
                  ) : wordSaveStatus === 'error' ? (
                    <XCircle className="h-3.5 w-3.5" />
                  ) : (
                    <Save className="h-3.5 w-3.5" />
                  )}
                  {wordSaveStatus === 'saving'
                    ? '写入 Word…'
                    : wordSaveStatus === 'success'
                      ? '已归档 Word'
                      : wordSaveStatus === 'error'
                        ? '归档失败，重试'
                        : '归档 Word 报告'}
                </button>
                <button
                  type="button"
                  onClick={handleExportReport}
                  className="flex items-center justify-center gap-2 rounded-xl bg-slate-700/60 py-2 text-xs font-semibold text-slate-200 transition hover:bg-slate-600/70 active:scale-95"
                >
                  <Download className="h-3.5 w-3.5" />
                  导出 JSON
                </button>
                <div className="rounded-xl border border-slate-700/70 bg-slate-900/35 p-2.5">
                  <p className="mb-2 text-[10px] font-semibold text-slate-500">档案快照</p>
                  <div className="grid grid-cols-2 gap-1.5 text-[10px]">
                    <span className="text-slate-500">学号</span>
                    <span className="truncate text-right text-slate-200">{studentNumber || '未填写'}</span>
                    <span className="text-slate-500">学校</span>
                    <span className="truncate text-right text-slate-200">
                      {getSchoolDisplayName(globalSettings)}
                    </span>
                    <span className="text-slate-500">班级</span>
                    <span className="truncate text-right text-slate-200">
                      {getClassGroupDisplayName(globalSettings)}
                    </span>
                  </div>
                  <div className="mt-2 flex justify-center">
                    <ThresholdRing stats={hitStats} />
                  </div>
                </div>
              </div>
            ) : (
              <div className="rounded-xl border border-slate-700/70 bg-slate-900/35 p-2.5">
                <div className="mb-2 flex items-center gap-1.5 text-[10px] font-semibold text-slate-500">
                  <GraduationCap className="h-3 w-3" />
                  实时命中分布
                </div>
                <div className="flex justify-center">
                  <ThresholdRing stats={hitStats} />
                </div>
                <div className="mt-2 flex flex-col gap-1.5">
                  {(['green', 'yellow', 'red'] as ThresholdLevel[]).map((lvl) => {
                    const count = hitStats[lvl]
                    const ratio = totalAttempts > 0 ? (count / totalAttempts) * 100 : 0
                    const colorStyle = LEVEL_COLOR_MAP[lvl]
                    return (
                      <div key={lvl}>
                        <div className="mb-0.5 flex items-center justify-between text-[10px]">
                          <span className={`font-medium ${colorStyle.text}`}>{LEVEL_LABEL_MAP[lvl]}</span>
                          <span className="text-slate-500">
                            {count} · {ratio.toFixed(0)}%
                          </span>
                        </div>
                        <div className="h-1.5 overflow-hidden rounded-full bg-slate-900">
                          <motion.div
                            animate={{ width: `${ratio}%` }}
                            transition={{ duration: 0.6, ease: 'easeOut' }}
                            className={`h-full rounded-full ${HIT_BAR_BG[lvl]}`}
                          />
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )
          }
        />
      </div>

      <AnimatePresence>
        {wordSaveToast && (
          <motion.div
            key={wordSaveToast.id}
            initial={{ opacity: 0, y: -16, scale: 0.94 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -16, scale: 0.94 }}
            transition={{ type: 'spring', stiffness: 300, damping: 26 }}
            className={`fixed top-6 right-6 z-[60] flex max-w-md items-start gap-3 rounded-2xl border px-5 py-3.5 shadow-2xl backdrop-blur-2xl ${
              wordSaveToast.success
                ? 'border-[color-mix(in_srgb,var(--GREEN_OPTIMAL)_35%,transparent)] bg-slate-950/90 text-[var(--GREEN_OPTIMAL)]'
                : 'border-[color-mix(in_srgb,var(--RED_DEVIATED)_35%,transparent)] bg-slate-950/90 text-[var(--RED_DEVIATED)]'
            }`}
          >
            <span className="mt-0.5 inline-flex flex-shrink-0">
              {wordSaveToast.success ? (
                <CheckCircle2 className="h-4 w-4" />
              ) : (
                <XCircle className="h-4 w-4" />
              )}
            </span>
            <p className="text-sm leading-relaxed break-all text-slate-100">{wordSaveToast.message}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/** 三级阈值命中占比环形图（纯 SVG 实现，无需图表库依赖） */
function ThresholdRing({ stats }: { stats: ThresholdHitStats }) {
  const total = stats.green + stats.yellow + stats.red
  const radius = 42
  const strokeWidth = 10
  const circumference = 2 * Math.PI * radius

  let cumulativeOffset = 0
  const segments = (['green', 'yellow', 'red'] as ThresholdLevel[]).map((lvl) => {
    const ratio = total > 0 ? stats[lvl] / total : 0
    const length = ratio * circumference
    const segment = { lvl, length, offset: cumulativeOffset }
    cumulativeOffset += length
    return segment
  })

  const greenRatioPercent = total > 0 ? Math.round((stats.green / total) * 100) : 0

  return (
    <div className="relative flex h-32 w-32 items-center justify-center">
      <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90">
        <circle cx="50" cy="50" r={radius} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={strokeWidth} />
        {segments.map(
          (segment) =>
            segment.length > 0 && (
              <circle
                key={segment.lvl}
                cx="50"
                cy="50"
                r={radius}
                fill="none"
                stroke={RING_STROKE[segment.lvl]}
                strokeWidth={strokeWidth}
                strokeDasharray={`${segment.length} ${circumference - segment.length}`}
                strokeDashoffset={-segment.offset}
                strokeLinecap="round"
              />
            ),
        )}
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="text-2xl font-bold text-white">{total}</span>
        <span className="text-[10px] text-white/40">次采样 · 达标{greenRatioPercent}%</span>
      </div>
    </div>
  )
}
