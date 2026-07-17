import { useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Play,
  Square,
  ScanFace,
  Bone,
  Sparkles,
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

/** 三级阈值命中次数柱状条对应的背景色类名 */
const HIT_BAR_BG: Record<ThresholdLevel, string> = {
  green: 'bg-emerald-400',
  yellow: 'bg-amber-400',
  red: 'bg-rose-400',
}

/** 环形图各分段对应的描边颜色（十六进制，供 SVG stroke 使用） */
const RING_STROKE: Record<ThresholdLevel, string> = {
  green: '#34d399',
  yellow: '#fbbf24',
  red: '#fb7185',
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

/** 「实时动力链角速度监控」单个采样点：客户端接收时间戳 + 后端计算出的角速度值 */
interface VelocitySample {
  t: number
  v: number
}

/** 波形图展示的滚动时间窗口长度（毫秒），对应需求中的「5 秒时序波形」 */
const VELOCITY_WINDOW_MS = 5000

/** 波形纵轴裁剪范围（deg/s）：超出该范围的偶发异常尖峰会被裁剪，避免压扁整张图 */
const VELOCITY_CLAMP_RANGE = 420

/** 动平衡稳定指数分级展示文案与颜色（对应右上角「XX/100 优秀」徽标） */
function getStabilityMeta(index: number | null): { text: string; className: string } {
  if (index === null) return { text: '--/100', className: 'text-white/40' }
  if (index >= 90) return { text: `${index}/100 优秀`, className: 'text-emerald-300' }
  if (index >= 75) return { text: `${index}/100 良好`, className: 'text-sky-300' }
  if (index >= 60) return { text: `${index}/100 一般`, className: 'text-amber-300' }
  return { text: `${index}/100 待提升`, className: 'text-rose-300' }
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
  const [velocityHistory, setVelocityHistory] = useState<VelocitySample[]>([])
  const [stabilityIndex, setStabilityIndex] = useState<number | null>(null)

  /* ---------------------------- 诊断统计与最终报告状态 ---------------------------- */
  const [hitStats, setHitStats] = useState<ThresholdHitStats>({ green: 0, yellow: 0, red: 0 })
  const [finalReport, setFinalReport] = useState<FinalDiagnosisReport | null>(null)
  const [isGeneratingReport, setIsGeneratingReport] = useState(false)

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

      // 实时动力链角速度监控：把新样本追加进滚动窗口，并淘汰掉 5 秒之前的旧样本。
      // 【容错防呆】额外用 Number.isFinite 校验，避免后端在极端情况下（例如第一帧
      // 尚未计算出角速度、或除法结果出现 NaN/Infinity）推来非法数值，导致后续
      // SVG 波形图渲染出无法解析的坐标，静默影响整张卡片渲染。
      if (typeof message.angular_velocity === 'number' && Number.isFinite(message.angular_velocity)) {
        const now = Date.now()
        const nextVelocity = message.angular_velocity
        setVelocityHistory((prev) => {
          const next = [...prev, { t: now, v: nextVelocity }]
          const cutoff = now - VELOCITY_WINDOW_MS
          return next.filter((sample) => sample.t >= cutoff)
        })
      }
      if (typeof message.stability_index === 'number' && Number.isFinite(message.stability_index)) {
        setStabilityIndex(message.stability_index)
      }
      return
    }

    if (message.type === 'stopped') {
      wsRef.current?.close()
      wsRef.current = null
      setIsConnected(false)
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
    setVelocityHistory([])
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
          hitStats: finalReport.hitStats ?? hitStats,
          kneeFlexionAngle: finalReport.avgKneeAngle ?? null,
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
    <div className="mx-auto flex max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
      {/* ============================ 顶栏控制区 ============================ */}
      <section className="flex flex-col gap-4 rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-xl sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
        <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center sm:gap-3">
          {/* 视频源分段控制器 */}
          <div className="inline-flex items-center gap-1 self-start rounded-full bg-black/30 p-1">
            {(
              [
                { id: 'webcam' as VideoSourceMode, label: '实时摄像头', icon: Camera },
                { id: 'file' as VideoSourceMode, label: '本地视频分析', icon: FileVideo },
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
                  className={`relative rounded-full px-3.5 py-2 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 sm:text-sm ${
                    active ? 'text-white' : 'text-white/50 hover:text-white/80'
                  }`}
                >
                  {active && (
                    <motion.span
                      layoutId="video-source-pill"
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

          {/* 本地 MP4 文件选择胶囊：仅在「本地视频分析」模式下显示，选中文件后会立即真实上传到后端 */}
          {videoSourceMode === 'file' && (
            <>
              <button
                type="button"
                disabled={isAnalyzing || isUploading}
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs text-white/70 backdrop-blur-xl transition hover:bg-white/10 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50 sm:text-sm"
              >
                <span className="inline-flex flex-shrink-0">
                  {isUploading ? (
                    <Loader2 className="h-4 w-4 animate-spin text-sky-400" />
                  ) : (
                    <UploadCloud className="h-4 w-4 text-sky-400" />
                  )}
                </span>
                <span className="max-w-[10rem] truncate sm:max-w-[16rem]">
                  {isUploading
                    ? '正在上传至后端…'
                    : localVideoFile
                      ? `${localVideoFile.name}${uploadedVideoPath ? '（已上传）' : ''}`
                      : '点击选择本地 MP4 文件（如 test_video.mp4）'}
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

          {/* 受试者编号/姓名自由输入：100% 自定义文本，支持任意格式（如 "NO. 07" / "实验测试张同学"），
              全程实时透传给全页各卡片与最终诊断报告，不做任何格式限制或校验 */}
          <div className="flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-2 backdrop-blur-xl">
            <User className="h-4 w-4 flex-shrink-0 text-white/40" />
            <input
              type="text"
              value={studentNumber}
              onChange={(e) => setStudentNumber(e.target.value)}
              placeholder='自由填写编号/姓名，如 "NO. 07"'
              className="w-36 bg-transparent text-sm text-white outline-none placeholder:text-white/30 sm:w-48"
            />
          </div>
        </div>

        {/* 操作主按钮 */}
        <div className="flex items-center gap-3">
          {!isAnalyzing && analysisStatus !== 'stopping' ? (
            <button
              type="button"
              onClick={handleStart}
              disabled={isStartDisabled}
              title={isStartDisabled ? '请先选择并等待本地 MP4 文件上传完成' : undefined}
              className="flex items-center gap-2 rounded-full bg-emerald-500 px-5 py-2.5 text-sm font-semibold text-black transition hover:bg-emerald-400 active:scale-95 disabled:cursor-not-allowed disabled:bg-white/10 disabled:text-white/30"
            >
              <span className="inline-flex flex-shrink-0">
                <Play className="h-4 w-4" />
              </span>
              开始分析 / 训练
            </button>
          ) : (
            <button
              type="button"
              onClick={handleStop}
              disabled={analysisStatus === 'stopping'}
              className="flex items-center gap-2 rounded-full bg-rose-500 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-rose-400 active:scale-95 disabled:cursor-not-allowed disabled:opacity-60"
            >
              <span className="inline-flex flex-shrink-0">
                {analysisStatus === 'stopping' ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Square className="h-4 w-4" />
                )}
              </span>
              结束分析
            </button>
          )}
        </div>
      </section>

      {/* 后端连接错误提示条 */}
      {connectionError && (
        <div className="flex items-center gap-2 rounded-2xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          <span className="inline-flex flex-shrink-0">
            <AlertTriangle className="h-4 w-4" />
          </span>
          <p>{connectionError}</p>
        </div>
      )}

      {/* 【新增】非致命诊断提醒条：例如自动检测到摄像头持续输出全黑画面，
          用黄色样式与上方红色的连接错误条区分，且不会中断当前分析会话 */}
      {diagnosticNotice && (
        <div className="flex items-start gap-2 rounded-2xl border border-amber-400/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
          <span className="mt-0.5 inline-flex flex-shrink-0">
            <AlertTriangle className="h-4 w-4" />
          </span>
          <p className="leading-relaxed">{diagnosticNotice}</p>
        </div>
      )}

      {/* ============================ 主体左右分栏 ============================ */}
      <div className="flex flex-col gap-6 lg:flex-row lg:items-start">
        {/* -------- 左侧：视觉分析展示区（约 62% 宽度） -------- */}
        <div className="w-full lg:w-[62%] lg:flex-shrink-0">
          <div className="relative min-h-[520px] overflow-hidden rounded-3xl bg-gradient-to-br from-zinc-900 via-black to-zinc-900 shadow-2xl">
            {/* 画面渲染主体：真实订阅 api_server.py 通过 WebSocket 推来的、
                已经在后端完成骨骼渲染 + 三级染色 + 面部打码的 Base64 JPEG 帧 */}
            {frameImage ? (
              <img
                src={frameImage}
                alt="实时推理画面"
                className="absolute inset-0 h-full w-full bg-black object-contain"
                // 【容错防呆】万一某一帧的 Base64 数据在传输/解码过程中意外损坏，
                // 浏览器会触发 onError；此时主动清空 frameImage 回退到占位提示，
                // 避免长期停留在一张"损坏的图片图标"上，被误认为是黑屏卡死。
                onError={() => setFrameImage(null)}
              />
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-white/30">
                <span className="inline-flex flex-shrink-0">
                  {videoSourceMode === 'file' ? <FileVideo className="h-16 w-16" /> : <Camera className="h-16 w-16" />}
                </span>
                <p className="max-w-sm text-center text-sm">
                  {videoSourceMode === 'file'
                    ? uploadedVideoPath
                      ? '视频已就绪，点击「开始分析 / 训练」后将显示后端实时推理画面'
                      : '请先在上方选择本地 MP4 文件并等待上传完成'
                    : '点击「开始分析 / 训练」后，后端将打开本机摄像头并推送实时推理画面'}
                </p>
              </div>
            )}

            {/* 右膝角度监控卡：直接绑定后端实时计算出的真实角度与状态 */}
            <motion.div
              animate={{ backgroundColor: smoothBackground }}
              transition={{ duration: 0.8, ease: 'easeInOut' }}
              className="absolute top-5 left-5 rounded-2xl border border-white/10 px-5 py-4 shadow-lg backdrop-blur-xl"
            >
              <p className="text-xs font-medium text-white/70">右膝角度监控（后端实时计算）</p>
              <p className="mt-1 text-4xl font-bold tabular-nums text-white">
                {kneeAngle !== null ? `${kneeAngle}°` : '--'}
              </p>
              <span className="mt-1 inline-block rounded-full bg-black/30 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-white/85">
                {level ? LEVEL_LABEL_MAP[level] : '等待检测到人体'}
              </span>
            </motion.div>

            {/* 后端强制开启的隐私保护 / 骨骼渲染状态徽标（伦理红线，不可由前端关闭） */}
            <div className="absolute top-5 right-5 flex flex-col items-end gap-2">
              <span className="flex items-center gap-2 rounded-full border border-white/10 bg-emerald-500/20 px-3 py-1.5 text-xs text-emerald-300 backdrop-blur-xl">
                <span className="inline-flex flex-shrink-0">
                  <ScanFace className="h-3.5 w-3.5" />
                </span>
                后端强制隐私打码
              </span>
              <span className="flex items-center gap-2 rounded-full border border-white/10 bg-sky-500/20 px-3 py-1.5 text-xs text-sky-300 backdrop-blur-xl">
                <span className="inline-flex flex-shrink-0">
                  <Bone className="h-3.5 w-3.5" />
                </span>
                后端实时骨骼渲染
              </span>
              <span
                className={`flex items-center gap-2 rounded-full border border-white/10 px-3 py-1.5 text-xs backdrop-blur-xl ${
                  isConnected ? 'bg-emerald-500/10 text-emerald-300' : 'bg-black/30 text-white/40'
                }`}
              >
                <span className="inline-flex flex-shrink-0">
                  {isConnected ? <Wifi className="h-3.5 w-3.5" /> : <WifiOff className="h-3.5 w-3.5" />}
                </span>
                {isConnected ? '已连接后端服务' : '未连接'}
              </span>
            </div>

            {/* 灵动岛风格提示：正在等待后端生成真实诊断报告时的加载态 */}
            <AnimatePresence>
              {isGeneratingReport && (
                <motion.div
                  initial={{ opacity: 0, y: 20, scale: 0.9 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 20, scale: 0.9 }}
                  transition={{ type: 'spring', stiffness: 260, damping: 24 }}
                  className="absolute bottom-6 left-1/2 flex w-[88%] max-w-md -translate-x-1/2 items-center gap-3 rounded-full border border-white/10 bg-black/70 px-5 py-3 shadow-2xl backdrop-blur-xl"
                >
                  <span className="inline-flex flex-shrink-0">
                    <Loader2 className="h-4 w-4 animate-spin text-emerald-400" />
                  </span>
                  <p className="text-sm text-white/90">正在请求 DeepSeek 大模型生成真实诊断报告……</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* 【核心新增】实时动力链角速度 (deg/s) 与动平衡稳定指数监控卡片：
              随视频推移实时滚动的 5 秒时序波形，右上角同步给出动平衡稳定指数评级 */}
          <KineticChainMonitor history={velocityHistory} stabilityIndex={stabilityIndex} />
        </div>

        {/* -------- 右侧：反馈报告与诊断中心 Bento Grid（约 38% 宽度） -------- */}
        <aside className="flex w-full flex-col gap-5 lg:w-[38%]">
          {/* 便当盒一：档案与状态 */}
          <section className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-xl">
            <div className="mb-4 flex items-center gap-2">
              <span className="inline-flex flex-shrink-0">
                <GraduationCap className="h-4 w-4 text-emerald-400" />
              </span>
              <h3 className="text-sm font-semibold text-white/80">档案与状态</h3>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <InfoCell label="学生编号" value={studentNumber || '未填写'} />
              <InfoCell
                label="运行状态"
                value={STATUS_META[analysisStatus].label}
                valueClassName={STATUS_META[analysisStatus].className}
              />
              <InfoCell label="所属学校" value={getSchoolDisplayName(globalSettings)} />
              <InfoCell label="所属班级" value={getClassGroupDisplayName(globalSettings)} />
            </div>
          </section>

          {/* 便当盒二：实时动作指标 */}
          <section className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-xl">
            <h3 className="mb-4 text-sm font-semibold text-white/80">实时动作指标</h3>

            {/* 三色命中占比环形图 */}
            <div className="mb-5 flex items-center justify-center">
              <ThresholdRing stats={hitStats} />
            </div>

            {/* 各等级命中次数进度条 */}
            <div className="flex flex-col gap-3">
              {(['green', 'yellow', 'red'] as ThresholdLevel[]).map((lvl) => {
                const count = hitStats[lvl]
                const ratio = totalAttempts > 0 ? (count / totalAttempts) * 100 : 0
                const colorStyle = LEVEL_COLOR_MAP[lvl]
                return (
                  <div key={lvl}>
                    <div className="mb-1 flex items-center justify-between text-xs">
                      <span className={`font-medium ${colorStyle.text}`}>{LEVEL_LABEL_MAP[lvl]}</span>
                      <span className="text-white/40">
                        {count} 次 · {ratio.toFixed(0)}%
                      </span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-black/30">
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
          </section>
        </aside>
      </div>

      {/* ============================ 便当盒三（重构）：图文并茂的击球瞬间关键帧生物力学诊断报告单 ============================
          扩展为全宽双栏「科研诊断报告单」：左栏展示后端 OpenCV 矢量标注的击球关键帧，右栏展示评分卡 + DeepSeek 打字机文字处方 */}
      <section
        className={`relative flex flex-col gap-5 rounded-3xl border p-5 backdrop-blur-xl transition-colors sm:p-6 ${
          finalReport
            ? 'border-emerald-400/30 bg-gradient-to-br from-emerald-500/10 via-white/5 to-transparent'
            : 'border-white/10 bg-white/5'
        }`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="inline-flex flex-shrink-0">
              <Sparkles className="h-4 w-4 text-emerald-400" />
            </span>
            <h3 className="text-sm font-semibold text-white/80">AI 综合反馈报告 · 双栏科研诊断报告单</h3>
          </div>
          <span
            className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold ${
              isAnalyzing || isGeneratingReport
                ? 'bg-amber-500/20 text-amber-300'
                : finalReport
                  ? 'bg-emerald-500/20 text-emerald-300'
                  : 'bg-white/10 text-white/40'
            }`}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                isAnalyzing || isGeneratingReport
                  ? 'animate-pulse bg-amber-400'
                  : finalReport
                    ? 'bg-emerald-400'
                    : 'bg-white/30'
              }`}
            />
            {isGeneratingReport ? 'DeepSeek 生成中' : isAnalyzing ? '分析中' : finalReport ? '报告已生成' : '待机'}
          </span>
        </div>

        {!finalReport && !isAnalyzing && !isGeneratingReport && (
          <p className="text-xs leading-relaxed text-white/40">
            点击「开始分析 / 训练」后，系统将通过后端真实推理持续采集触球瞬间关节角度与角速度数据，并自动捕捉
            右膝角速度峰值所在的「击球关键帧」；点击「结束分析」后，后端会真正调用 DeepSeek 大模型 + OpenCV
            矢量标注引擎，在此处生成左图右文的结构化本次综合练习诊断报告。
          </p>
        )}

        {isAnalyzing && !finalReport && (
          <div className="flex flex-col gap-2">
            <p className="text-xs text-white/50">实时监测中，当前判定状态：</p>
            <p className="min-h-[3rem] rounded-2xl bg-black/20 p-3 text-sm text-white/80">
              {level
                ? `右膝角度 ${kneeAngle}°，判定为「${LEVEL_LABEL_MAP[level]}」，累计已采集 ${totalAttempts} 次有效数据。`
                : '正在等待后端检测到完整人体姿态……'}
            </p>
          </div>
        )}

        {finalReport && (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* 左栏：生物力学关键帧诊断图 —— 后端 OpenCV 矢量标注的击球瞬间截图 */}
            <div className="flex flex-col gap-2">
              <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-black/40">
                {finalReport.impactFrameImage ? (
                  <img
                    src={finalReport.impactFrameImage}
                    alt="击球瞬间生物力学诊断关键帧"
                    className="h-full w-full object-contain"
                  />
                ) : (
                  <div className="flex aspect-video flex-col items-center justify-center gap-2 text-white/30">
                    <span className="inline-flex flex-shrink-0">
                      <ScanFace className="h-10 w-10" />
                    </span>
                    <p className="max-w-[16rem] text-center text-xs">
                      本次分析未能捕捉到有效击球关键帧（可能全程未检测到完整人体姿态）。
                    </p>
                  </div>
                )}
                <span className="absolute left-3 top-3 flex items-center gap-1.5 rounded-full border border-white/10 bg-black/60 px-2.5 py-1 text-[10px] text-emerald-300 backdrop-blur-xl">
                  <span className="inline-flex flex-shrink-0">
                    <Crosshair className="h-3 w-3" />
                  </span>
                  矢量标注 · 髋-膝-踝动力链
                </span>
              </div>
              <p className="text-center text-xs text-white/40">关键力学特征捕获帧 (Impact Moment)</p>
            </div>

            {/* 右栏：AIGC 专家处方 —— 评分卡 + 打字机动效呈现的 DeepSeek 文字指导 */}
            <div className="flex flex-col gap-4">
              <div className="flex items-center gap-4">
                <div className="flex h-16 w-16 flex-shrink-0 flex-col items-center justify-center rounded-2xl bg-emerald-500/20">
                  <span className="text-2xl font-bold text-emerald-300">{finalReport.score}</span>
                  <span className="text-[9px] text-emerald-300/70">评分</span>
                </div>
                <div className="text-xs text-white/40">
                  <p>共采集 {finalReport.totalAttempts} 次有效触球数据</p>
                  <p>{finalReport.generatedAt}</p>
                </div>
              </div>
              <p className="flex-1 whitespace-pre-line rounded-2xl bg-black/20 p-4 text-sm leading-relaxed text-white/85">
                {reportDisplayText}
                {!isReportDone && <span className="typewriter-caret">|</span>}
              </p>
              {/* 醒目的主操作按钮：调用后台接口，在本机硬盘完成建文件夹 + 写 Word 文档 */}
              <button
                type="button"
                onClick={handleSaveWordReport}
                disabled={wordSaveStatus === 'saving'}
                className={`flex items-center justify-center gap-2 rounded-2xl py-3 text-sm font-bold shadow-lg transition active:scale-95 disabled:cursor-not-allowed disabled:opacity-70 ${
                  wordSaveStatus === 'success'
                    ? 'bg-emerald-500 text-black shadow-emerald-500/30'
                    : wordSaveStatus === 'error'
                      ? 'bg-rose-500 text-white shadow-rose-500/30'
                      : 'bg-gradient-to-r from-emerald-400 to-teal-400 text-black shadow-emerald-500/30 hover:brightness-110'
                }`}
              >
                <span className="inline-flex flex-shrink-0">
                  {wordSaveStatus === 'saving' ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : wordSaveStatus === 'success' ? (
                    <CheckCircle2 className="h-4 w-4" />
                  ) : wordSaveStatus === 'error' ? (
                    <XCircle className="h-4 w-4" />
                  ) : (
                    <Save className="h-4 w-4" />
                  )}
                </span>
                {wordSaveStatus === 'saving'
                  ? '正在写入本地 Word 文档…'
                  : wordSaveStatus === 'success'
                    ? '已成功归档为 Word 文档'
                    : wordSaveStatus === 'error'
                      ? '归档失败，点击重试'
                      : '💾 自动归档并生成 Word 报告'}
              </button>
              <button
                type="button"
                onClick={handleExportReport}
                className="flex items-center justify-center gap-2 rounded-2xl bg-white/10 py-2.5 text-sm font-semibold text-white/80 transition hover:bg-white/20 active:scale-95"
              >
                <span className="inline-flex flex-shrink-0">
                  <Download className="h-4 w-4" />
                </span>
                导出本次反馈报告 (JSON)
              </button>
            </div>
          </div>
        )}
      </section>

      {/* ============================ 右上角浮动 Toast 提示条：Word 报告归档结果 ============================ */}
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
                ? 'border-emerald-400/30 bg-emerald-950/90 text-emerald-100'
                : 'border-rose-400/30 bg-rose-950/90 text-rose-100'
            }`}
          >
            <span className="mt-0.5 inline-flex flex-shrink-0">
              {wordSaveToast.success ? (
                <CheckCircle2 className="h-4 w-4 text-emerald-300" />
              ) : (
                <XCircle className="h-4 w-4 text-rose-300" />
              )}
            </span>
            <p className="text-sm leading-relaxed break-all">{wordSaveToast.message}</p>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex items-center gap-2 rounded-3xl border border-white/10 bg-white/5 p-4 text-xs text-white/40 backdrop-blur-xl">
        <span className="inline-flex flex-shrink-0">
          <Bone className="h-4 w-4" />
        </span>
        <p>
          全边缘计算：所有推理与渲染均在 api_server.py 本地内存中完成，画面通过局域网 WebSocket 直传浏览器，不上传公有云端。
        </p>
      </div>
    </div>
  )
}

/** 档案信息小格子 */
function InfoCell({ label, value, valueClassName }: { label: string; value: string; valueClassName?: string }) {
  return (
    <div className="rounded-2xl bg-black/20 px-3.5 py-3">
      <p className="text-[11px] text-white/40">{label}</p>
      <p className={`mt-1 truncate text-sm font-semibold ${valueClassName ?? 'text-white/90'}`}>{value}</p>
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

/**
 * 【核心新增】实时动力链角速度 (deg/s) 与动平衡稳定指数监控卡片。
 *
 * 纯 SVG 实现的滚动时序波形图：横轴为最近 5 秒的滚动时间窗口，纵轴为后端
 * 实时计算出的右膝角速度（正负号代表伸展/屈曲方向），发力瞬间角速度会
 * 骤然拉出尖峰，直观呈现"动力链传导是否顺畅、是否存在多余抖动"；
 * 右上角同步展示后端基于滑动窗口角速度离散程度换算出的动平衡稳定指数。
 */
function KineticChainMonitor({
  history,
  stabilityIndex,
}: {
  history: VelocitySample[]
  stabilityIndex: number | null
}) {
  const stabilityMeta = getStabilityMeta(stabilityIndex)

  const viewWidth = 600
  const viewHeight = 140
  const midY = viewHeight / 2
  const now = Date.now()

  // 把每个样本的"距今毫秒数"映射到 [0, viewWidth] 的横坐标（越靠右代表越接近当前时刻）。
  // 【容错防呆】先过滤掉任何非有限数值的样本（NaN/Infinity 等异常数据），
  // 避免拼出非法的 SVG path 坐标字符串，导致整张波形图静默不渲染。
  const points = history
    .filter((sample) => Number.isFinite(sample.t) && Number.isFinite(sample.v))
    .map((sample) => {
      const elapsed = now - sample.t
      const x = viewWidth - (elapsed / VELOCITY_WINDOW_MS) * viewWidth
      const clampedV = Math.max(-VELOCITY_CLAMP_RANGE, Math.min(VELOCITY_CLAMP_RANGE, sample.v))
      const y = midY - (clampedV / VELOCITY_CLAMP_RANGE) * (midY - 12)
      return { x, y }
    })

  const linePath = points.length > 1 ? `M ${points.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' L ')}` : ''
  const areaPath =
    points.length > 1
      ? `${linePath} L ${points[points.length - 1].x.toFixed(1)},${midY} L ${points[0].x.toFixed(1)},${midY} Z`
      : ''

  const latestSample = history.length > 0 ? history[history.length - 1] : null

  return (
    <section className="mt-6 rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-xl">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="inline-flex flex-shrink-0">
            <Gauge className="h-4 w-4 text-emerald-400" />
          </span>
          <h3 className="text-sm font-semibold text-white/80">实时动力链角速度监控</h3>
          <span className="rounded-full bg-black/30 px-2 py-0.5 text-[10px] text-white/40">5s 滚动波形</span>
        </div>
        <div className="text-right">
          <p className="text-[10px] text-white/40">动平衡稳定指数</p>
          <p className={`text-sm font-bold tabular-nums ${stabilityMeta.className}`}>{stabilityMeta.text}</p>
        </div>
      </div>

      <div className="relative overflow-hidden rounded-2xl bg-black/30">
        <svg viewBox={`0 0 ${viewWidth} ${viewHeight}`} preserveAspectRatio="none" className="h-28 w-full">
          {/* 零基准线：代表角速度为 0 deg/s，即摆动腿瞬时静止的参考线 */}
          <line x1="0" y1={midY} x2={viewWidth} y2={midY} stroke="rgba(255,255,255,0.12)" strokeWidth={1} strokeDasharray="4 4" />
          {areaPath && <path d={areaPath} fill="rgba(52,211,153,0.14)" stroke="none" />}
          {linePath && <path d={linePath} fill="none" stroke="#34d399" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />}
          {/* 当前帧尖端高亮点：标记波形最新推进到的位置，模拟"实时扫描笔尖" */}
          {points.length > 0 && (
            <circle cx={points[points.length - 1].x} cy={points[points.length - 1].y} r={4} fill="#a7f3d0" />
          )}
        </svg>
        <div className="pointer-events-none absolute bottom-2 left-3 text-[10px] text-white/30">
          实时角速度：{latestSample ? `${latestSample.v.toFixed(0)} deg/s` : '--'}
        </div>
      </div>

      <p className="mt-2 text-[11px] leading-relaxed text-white/30">
        波形来自后端逐帧计算的右膝关节角速度（deg/s），发力瞬间会出现尖峰；动平衡稳定指数基于滑动窗口内角速度的
        离散程度换算，指数越高代表触球前后身体动力链传导越连贯稳定。
      </p>
    </section>
  )
}
