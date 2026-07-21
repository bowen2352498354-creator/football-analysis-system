import { useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Play,
  Square,
  Camera,
  FileVideo,
  UploadCloud,
  User,
  Loader2,
  Flag,
  ArrowLeft,
  CheckCircle2,
  Users,
  ChevronRight,
  Download,
  Sparkles,
  Crosshair,
  ScanFace,
  GraduationCap,
  Clock,
  Lock,
  Unlock,
  SkipForward,
  Trash2,
  RotateCcw,
  TrendingUp,
  Award,
  Save,
} from 'lucide-react'
import {
  appendGlobalRecordToLocalStorage,
  getClassGroupDisplayName,
  getSchoolDisplayName,
  loadZenSessionsFromLocalStorage,
  saveZenSessionsToLocalStorage,
  clearZenSessionsFromLocalStorage,
  MOCK_RADAR_SCORES,
  MOCK_RADAR_SCORES_COMPARE,
  MOCK_SCORE_DETAIL_V31,
} from '../mockData'
import type {
  AggregateDiagnosisReport,
  FinalDiagnosisReport,
  GlobalSettings,
  GlobalTrainingRecord,
  VideoSourceMode,
  ZenAttemptRecord,
  ZenSessionRecord,
  ZenViewMode,
} from '../types'
import MetricCardList from './MetricCardList'

interface ZenWorkspaceProps {
  /** 来自 Navbar 的全局教学环境设置（学校 + 班级/组别），本工作台只读消费 */
  globalSettings: GlobalSettings
}

/* ============================================================================
 * 后台服务网关地址，与 RealtimeWorkspace.tsx 保持完全一致的联调配置。
 * ========================================================================== */
const API_BASE_URL = 'http://localhost:8000'
const WS_ANALYZE_URL = 'ws://localhost:8000/ws/analyze'

/** 后端 WebSocket 推送的消息结构（B组只关心 image/session_id/error，不消费角度三色判定字段） */
interface WsFrameMessage {
  type: 'frame'
  image: string
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
interface WsNoticeMessage {
  type: 'notice'
  message: string
}
type WsMessage = WsFrameMessage | WsStartedMessage | WsStoppedMessage | WsErrorMessage | WsNoticeMessage

/** 当前采集状态：待机 / 正在录制单次尝试 / 正在为这一次尝试生成报告 */
type CaptureStatus = 'idle' | 'recording' | 'archiving'

/** Apple 风格浮动 Toast 提示条的展示内容 */
interface ToastState {
  id: number
  message: string
}

let toastSeq = 0

/** 根据发力稳定性评分，判断该次尝试落在红/黄/绿哪个区间（评分缺失时判定为 unknown） */
type ScoreBucket = 'green' | 'yellow' | 'red' | 'unknown'

function getScoreBucket(score: number | null | undefined): ScoreBucket {
  if (typeof score !== 'number' || Number.isNaN(score)) return 'unknown'
  if (score >= 75) return 'green'
  if (score >= 55) return 'yellow'
  return 'red'
}

const SCORE_BUCKET_STYLE: Record<ScoreBucket, { text: string; bg: string; ring: string; dot: string; label: string }> = {
  green: { text: 'text-emerald-300', bg: 'bg-emerald-500/15', ring: 'ring-emerald-400/40', dot: 'bg-emerald-400', label: '合格区间' },
  yellow: { text: 'text-amber-300', bg: 'bg-amber-500/15', ring: 'ring-amber-400/40', dot: 'bg-amber-400', label: '接近合格' },
  red: { text: 'text-rose-300', bg: 'bg-rose-500/15', ring: 'ring-rose-400/40', dot: 'bg-rose-400', label: '需要关注' },
  unknown: { text: 'text-white/40', bg: 'bg-white/5', ring: 'ring-white/10', dot: 'bg-white/30', label: '数据缺失' },
}

/**
 * 【聚合计算核心】根据同一位学生多趟尝试的评分序列，计算「动作表现稳定性得分」。
 * 设计思路与后端 /api/generate_aggregate_report 保持一致：各趟评分越接近，
 * 说明动作表现越稳定，得分越高；忽高忽低（标准差偏大）则相应扣分。
 * 这里在前端做一份镜像计算，用于在等待后端聚合诊断返回之前先行展示预览值，
 * 后端返回结果后会以后端计算的 stabilityScore 为准（两者算法一致，理论上数值相同）。
 */
function computeClientStabilityScore(scores: number[]): number {
  if (scores.length === 0) return 0
  if (scores.length === 1) return Math.round(Math.max(0, Math.min(100, scores[0])))
  const mean = scores.reduce((sum, s) => sum + s, 0) / scores.length
  const variance = scores.reduce((sum, s) => sum + (s - mean) ** 2, 0) / scores.length
  const stdDev = Math.sqrt(variance)
  return Math.max(0, Math.min(100, Math.round(100 - stdDev * 1.5)))
}

/**
 * 延时反馈系统工作台（实验B组）· v3.0 跨课时聚合复盘版：
 *
 * 阶段一「课中静默采集模式」（viewMode === 'capture'）：
 *   学号锁定后，教练可连续点击"记录本次尝试"2~3 次，每次都会静默调用后台
 *   生成一份独立诊断报告，暂存进当前学生的临时尝试列表 currentAttempts；
 *   点击"完成该生测试"后，把这 2~3 次尝试打包成一个完整实体归档进
 *   sessionQueue，并同步落盘到 localStorage + 后端 JSON，实现跨课时双重持久化。
 *
 * 阶段二「课后集中复盘看板模式」（viewMode === 'review'）：
 *   左侧列出所有已完成测试的学生编号，右侧多趟聚合便当盒展示该生 2~3 次
 *   尝试的趋势变化、自动挑选的最佳击球关键帧，以及 DeepSeek 生成的跨次
 *   聚合诊断处方。
 */
export default function ZenWorkspace({ globalSettings }: ZenWorkspaceProps) {
  /* ---------------------------- 阶段切换与归档池核心状态 ---------------------------- */
  const [viewMode, setViewMode] = useState<ZenViewMode>('capture')
  // 归档池：初始化时优先从 localStorage 读取历史课堂数据，确保跨课时/跨周不丢失；
  // 任何异常情况（首次打开/存储损坏）都由 loadZenSessionsFromLocalStorage 内部兜底为空数组。
  const [sessionQueue, setSessionQueue] = useState<ZenSessionRecord[]>(() => loadZenSessionsFromLocalStorage())

  /* ---------------------------- 学号锁定 + 当前学生临时尝试列表 ---------------------------- */
  const [studentId, setStudentId] = useState('')
  const [isLocked, setIsLocked] = useState(false)
  const [currentAttempts, setCurrentAttempts] = useState<ZenAttemptRecord[]>([])

  /* ---------------------------- 采集控制栏状态（对齐A组输入） ---------------------------- */
  const [videoSourceMode, setVideoSourceMode] = useState<VideoSourceMode>('webcam')
  const [localVideoFile, setLocalVideoFile] = useState<File | null>(null)
  const [uploadedVideoPath, setUploadedVideoPath] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  /* ---------------------------- 后端连接与静默采集画面状态 ---------------------------- */
  const wsRef = useRef<WebSocket | null>(null)
  const sessionIdRef = useRef<string | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const [frameImage, setFrameImage] = useState<string | null>(null)
  const [captureStatus, setCaptureStatus] = useState<CaptureStatus>('idle')

  /* ---------------------------- Apple 风格 Toast 提示条 ---------------------------- */
  const [toast, setToast] = useState<ToastState | null>(null)

  /* ---------------------------- 复盘看板：选中的学生归档记录 / 手动指定的最佳尝试 ---------------------------- */
  const [selectedRecordId, setSelectedRecordId] = useState<string | null>(null)
  const [bestAttemptOverride, setBestAttemptOverride] = useState<Record<string, number>>({})
  const [aggregateLoadingId, setAggregateLoadingId] = useState<string | null>(null)

  /* ---------------------------- 批量导出全班 Word 报告单 ---------------------------- */
  const [isBatchExportingWord, setIsBatchExportingWord] = useState(false)

  const isRecording = captureStatus === 'recording'
  const isArchiving = captureStatus === 'archiving'

  /** 显示一条 Toast 提示，2.6 秒后自动淡出消失 */
  function showToast(message: string) {
    const id = ++toastSeq
    setToast({ id, message })
    window.setTimeout(() => {
      setToast((current) => (current?.id === id ? null : current))
    }, 2600)
  }

  /** 组件卸载时，确保 WebSocket 连接被妥善关闭，不留下悬空连接 */
  useEffect(() => {
    return () => {
      wsRef.current?.close()
      wsRef.current = null
    }
  }, [])

  /* ---------------------------- 跨课时双重持久化：前端 localStorage 半重保险 ---------------------------- */
  // 归档池发生任何变化（新增归档/清空/重新导入）都自动同步序列化保存到 localStorage，
  // 确保教练下周重新打开页面时，能自动读取到上节课的完整历史数据，不会丢失。
  useEffect(() => {
    saveZenSessionsToLocalStorage(sessionQueue)
  }, [sessionQueue])

  /** 跨课时双重持久化：后端 JSON 落盘半重保险，静默调用，失败也不打断教练操作 */
  async function persistSessionsToBackend(sessions: ZenSessionRecord[]) {
    try {
      await fetch(`${API_BASE_URL}/api/save_session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessions }),
      })
    } catch {
      // 静默忽略：后端落盘失败不影响 localStorage 这一重保险，不应打断教练的正常操作流程
    }
  }

  /**
   * 【核心新增】本地归档 + Word 报告生成：把某位同学某一次尝试的诊断数据 POST 给
   * 后台 /api/save_word_report，由 word_reporter.py 在本机硬盘上完成"建文件夹 +
   * 写 .docx"，绝不依赖浏览器直接下载。返回是否保存成功，供调用方汇总统计。
   */
  async function saveAttemptToWord(studentIdForReport: string, attempt: ZenAttemptRecord): Promise<boolean> {
    const report = attempt.reportData
    try {
      const response = await fetch(`${API_BASE_URL}/api/save_word_report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: 'delayed',
          school: getSchoolDisplayName(globalSettings),
          classGroup: getClassGroupDisplayName(globalSettings),
          studentNumber: studentIdForReport || '未填写编号',
          score: report?.score ?? null,
          totalAttempts: report?.totalAttempts ?? null,
          painPoint: report?.painPoint ?? '',
          prescription: report?.prescription ?? '',
          generatedAt: report?.generatedAt ?? null,
          impactFrameImage: attempt.impactFrameBase64 ?? report?.impactFrameImage ?? null,
          heatmapBase64:
            report?.heatmap_base64 ??
            report?.heatmapBase64 ??
            report?.scoreDetail?.heatmap_base64 ??
            null,
          heatmap_base64:
            report?.heatmap_base64 ??
            report?.heatmapBase64 ??
            report?.scoreDetail?.heatmap_base64 ??
            null,
          hitStats: report?.hitStats ?? null,
          kneeFlexionAngle: report?.avgKneeAngle ?? null,
          scoreDetail: report?.scoreDetail ?? null,
        }),
      })
      const data = (await response.json()) as { success: boolean; message?: string; record?: GlobalTrainingRecord }
      const ok = Boolean(response.ok && data.success)
      // 【双向同步全局数据库】后端已经把这条记录追加进 global_training_db.json，
      // 这里把同一份记录同步写进 localStorage 作为极速双保险，供教练端看板兜底读取。
      if (ok && data.record) {
        appendGlobalRecordToLocalStorage(data.record)
      }
      return ok
    } catch {
      return false
    }
  }

  function handleSelectVideoFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return
    setLocalVideoFile(file)
    setUploadedVideoPath(null)
    void uploadVideoFile(file)
  }

  /** 把用户选择的本地 MP4 文件上传到后端 /api/upload_video，换回后端文件路径 */
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

  /** 统一的 WebSocket 消息处理：B组只关心画面帧 + 会话号，绝不展示任何即时干预信息 */
  function handleWsMessage(event: MessageEvent<string>) {
    let message: WsMessage
    try {
      message = JSON.parse(event.data) as WsMessage
    } catch {
      return
    }

    if (message.type === 'frame') {
      if (typeof message.image === 'string' && message.image.startsWith('data:image')) {
        setFrameImage(message.image)
      }
      return
    }

    if (message.type === 'started') {
      sessionIdRef.current = message.session_id
      return
    }

    if (message.type === 'stopped') {
      wsRef.current?.close()
      wsRef.current = null
      setIsConnected(false)
      void finalizeCurrentAttempt(message.session_id)
      return
    }

    if (message.type === 'error') {
      setConnectionError(message.message)
      setCaptureStatus('idle')
      wsRef.current?.close()
      wsRef.current = null
      setIsConnected(false)
      return
    }

    // notice：非致命诊断提醒（如摄像头疑似全黑），B组不展示任何即时视觉打扰，静默忽略即可
  }

  /** 点击「锁定学号」：锁定后下方主按钮切换为"记录本次尝试"，输入框禁止再编辑 */
  function handleLockStudentId() {
    const trimmed = studentId.trim()
    if (!trimmed) return
    setStudentId(trimmed)
    setIsLocked(true)
  }

  /** 点击「解锁」：若已录入尝试但尚未归档，需二次确认，避免误触导致数据丢失 */
  function handleUnlockStudentId() {
    if (currentAttempts.length > 0) {
      const confirmed = window.confirm(
        `当前学号已静默录入 ${currentAttempts.length} 次尝试但尚未归档，解锁将清空这些临时数据，确认要解锁吗？`,
      )
      if (!confirmed) return
      setCurrentAttempts([])
    }
    setIsLocked(false)
  }

  /** 点击「记录本次尝试」：建立本次单趟采集的 WebSocket 会话 */
  function handleStartAttemptRecording() {
    if (!isLocked) return
    setConnectionError(null)
    setFrameImage(null)
    sessionIdRef.current = null

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
      setCaptureStatus('recording')
    }
    socket.onmessage = handleWsMessage
    socket.onerror = () => {
      setConnectionError('无法连接到后台服务，请确认 api_server.py 已在 8000 端口启动。')
      setCaptureStatus('idle')
      setIsConnected(false)
    }
    socket.onclose = () => {
      wsRef.current = null
      setIsConnected(false)
    }
  }

  /** 点击「结束本次尝试」：请求后台结束本次分析，后续报告生成在 finalizeCurrentAttempt 中静默完成 */
  function handleFinishAttemptRecording() {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'stop' }))
      setCaptureStatus('archiving')
    }
  }

  /**
   * 单次尝试分析结束后，静默调用 /api/generate_report 获取 DeepSeek 诊断报告 + 关键帧矢量截图，
   * 结果 push 进当前学生的临时尝试列表 currentAttempts（而非直接归档），支持连续攒 2~3 次。
   */
  async function finalizeCurrentAttempt(sessionId: string) {
    setCaptureStatus('archiving')
    const attemptNumber = currentAttempts.length + 1
    const activeStudentId = studentId.trim() || '未填写编号'

    try {
      const response = await fetch(`${API_BASE_URL}/api/generate_report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, student_number: activeStudentId }),
      })
      if (!response.ok) throw new Error(`报告接口返回状态码 ${response.status}`)
      const report = (await response.json()) as FinalDiagnosisReport

      const newAttempt: ZenAttemptRecord = {
        attemptNumber,
        timestamp: Date.now(),
        videoSource: videoSourceMode,
        reportData: report,
        impactFrameBase64: report.impactFrameImage ?? null,
      }

      setCurrentAttempts((prev) => {
        const next = [...prev, newAttempt]
        showToast(`${activeStudentId}：已静默录入 ${next.length} 次尝试 (Attempt #${next.length})`)
        return next
      })
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '生成诊断报告失败，请检查后端服务是否已启动。'
      setConnectionError(errorMessage)
      showToast(`⚠️ 第 ${attemptNumber} 次尝试数据生成失败，请检查后端服务`)
    } finally {
      setCaptureStatus('idle')
      setFrameImage(null)
      sessionIdRef.current = null
    }
  }

  /** 点击「完成该生测试 (换下一位)」：把当前学生的 2~3 次尝试打包归档，双重持久化保存后复位表单 */
  function handleFinishCurrentStudent() {
    if (currentAttempts.length === 0) return
    const finishedStudentId = studentId.trim() || '未填写编号'
    const newRecord: ZenSessionRecord = {
      id: `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      studentId: finishedStudentId,
      timestamp: Date.now(),
      attempts: currentAttempts,
    }

    const nextQueue = [...sessionQueue, newRecord]
    setSessionQueue(nextQueue)
    void persistSessionsToBackend(nextQueue)
    showToast(`✅ ${finishedStudentId} 已完成本节课测试并归档，共记录 ${newRecord.attempts.length} 次尝试`)

    // 【全局归档总闸】只有 Navbar 全局环境设置中的「本地落盘归档」开关开启时，
    // 才强制自动触发本地归档 + Word 报告生成：静默为该生本节课的每一次尝试
    // 各生成一份规范排版的 Word 处方单，写入 student feedback report/延时反馈/... 对应文件夹，
    // 并同步汇入全局训练数据库，供教练端看板实时消费。
    if (globalSettings.enableDataArchiving) {
      void (async () => {
        let successCount = 0
        for (const attempt of newRecord.attempts) {
          const ok = await saveAttemptToWord(finishedStudentId, attempt)
          if (ok) successCount += 1
        }
        if (successCount === newRecord.attempts.length) {
          showToast(`💾 ${finishedStudentId} 的 ${successCount} 份 Word 报告已自动归档至本地硬盘`)
        } else {
          showToast(`⚠️ ${finishedStudentId} 仅成功归档 ${successCount}/${newRecord.attempts.length} 份 Word 报告，请检查后端服务`)
        }
      })()
    }

    // 复位：清空当前学生的临时尝试列表与学号锁定状态，方便迅速换下一位同学
    setCurrentAttempts([])
    setStudentId('')
    setIsLocked(false)
    setFrameImage(null)
    setLocalVideoFile(null)
    setUploadedVideoPath(null)
  }

  /** 点击「所有人测试完成」：先做一次双重持久化落盘，再进入课前集中复盘看板 */
  function handleEnterReviewMode() {
    void persistSessionsToBackend(sessionQueue)
    setViewMode('review')
  }

  const isStartAttemptDisabled =
    !isLocked || isRecording || isArchiving || (videoSourceMode === 'file' && (!uploadedVideoPath || isUploading))
  const isFinishStudentDisabled = currentAttempts.length === 0 || isRecording || isArchiving

  /* ============================ 课后集中复盘看板：数据准备 ============================ */

  // 进入复盘模式时，若尚未选中任何学生记录，默认选中归档池中第一位（空数据态兜底为 null）
  useEffect(() => {
    if (viewMode !== 'review') return
    if (sessionQueue.length === 0) {
      setSelectedRecordId(null)
      return
    }
    const stillExists = sessionQueue.some((record) => record.id === selectedRecordId)
    if (!stillExists) {
      setSelectedRecordId(sessionQueue[0].id)
    }
  }, [viewMode, sessionQueue, selectedRecordId])

  const selectedRecord = sessionQueue.find((record) => record.id === selectedRecordId) ?? null
  const selectedAttempts = selectedRecord?.attempts ?? []

  // 该同学各趟尝试的发力稳定性评分序列（缺失数据兜底为空数组，绝不空指针）
  const attemptScores = useMemo(
    () => selectedAttempts.map((attempt) => attempt.reportData?.score ?? null),
    [selectedAttempts],
  )
  const validScores = useMemo(() => attemptScores.filter((s): s is number => typeof s === 'number'), [attemptScores])
  const clientStabilityScore = useMemo(() => computeClientStabilityScore(validScores), [validScores])

  // 自动挑选"最标准/合规"的那一次尝试（评分最高者）；教师也可在下方缩略图手动点选覆盖
  const autoBestAttemptIndex = useMemo(() => {
    if (selectedAttempts.length === 0) return 0
    let bestIdx = 0
    let bestScore = -Infinity
    selectedAttempts.forEach((attempt, idx) => {
      const score = attempt.reportData?.score
      if (typeof score === 'number' && score > bestScore) {
        bestScore = score
        bestIdx = idx
      }
    })
    return bestIdx
  }, [selectedAttempts])

  const effectiveBestAttemptIndex = selectedRecord
    ? bestAttemptOverride[selectedRecord.id] ?? autoBestAttemptIndex
    : 0
  const bestAttempt = selectedAttempts[effectiveBestAttemptIndex] ?? null

  function handleSelectRecord(id: string) {
    setSelectedRecordId(id)
  }

  function handleOverrideBestAttempt(recordId: string, attemptIndex: number) {
    setBestAttemptOverride((prev) => ({ ...prev, [recordId]: attemptIndex }))
  }

  /** 懒加载调用后端生成「跨次尝试聚合诊断报告」，结果缓存进该学生的归档记录里 */
  async function handleGenerateAggregateReport(record: ZenSessionRecord) {
    if (record.attempts.length === 0) return
    setAggregateLoadingId(record.id)
    try {
      const attemptsPayload = record.attempts.map((attempt) => ({
        attemptNumber: attempt.attemptNumber,
        score: attempt.reportData?.score ?? null,
        hitStats: attempt.reportData?.hitStats ?? null,
      }))
      const response = await fetch(`${API_BASE_URL}/api/generate_aggregate_report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ student_number: record.studentId, attempts: attemptsPayload }),
      })
      if (!response.ok) throw new Error(`聚合诊断接口返回状态码 ${response.status}`)
      const aggregate = (await response.json()) as AggregateDiagnosisReport

      setSessionQueue((prev) => {
        const next = prev.map((item) => (item.id === record.id ? { ...item, aggregateReport: aggregate } : item))
        void persistSessionsToBackend(next)
        return next
      })
    } catch {
      showToast('⚠️ 生成聚合诊断报告失败，请检查后端服务是否已启动')
    } finally {
      setAggregateLoadingId(null)
    }
  }

  // 选中一位新学生、且该生尚无聚合诊断缓存时，自动懒加载调用一次生成
  useEffect(() => {
    if (viewMode !== 'review' || !selectedRecord) return
    if (selectedRecord.attempts.length === 0) return
    if (selectedRecord.aggregateReport) return
    if (aggregateLoadingId === selectedRecord.id) return
    void handleGenerateAggregateReport(selectedRecord)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewMode, selectedRecordId, selectedRecord?.aggregateReport])

  /** 一键清空全部历史课堂归档数据（前端内存 + localStorage），需二次确认避免误触 */
  function handleClearHistory() {
    if (sessionQueue.length === 0) return
    const confirmed = window.confirm('确认清空全部历史课堂归档数据吗？此操作不可撤销。')
    if (!confirmed) return
    setSessionQueue([])
    clearZenSessionsFromLocalStorage()
    setSelectedRecordId(null)
    setBestAttemptOverride({})
    showToast('已清空全部历史课堂归档数据')
  }

  /** 一键重新导入历史课堂数据：优先读取本地缓存，本地为空时尝试从后端 JSON 落盘恢复 */
  async function handleReimportHistory() {
    const local = loadZenSessionsFromLocalStorage()
    if (local.length > 0) {
      setSessionQueue(local)
      showToast(`已从本地缓存重新导入 ${local.length} 位同学的历史数据`)
      return
    }
    try {
      const response = await fetch(`${API_BASE_URL}/api/load_sessions`)
      if (!response.ok) throw new Error(`读取归档接口返回状态码 ${response.status}`)
      const data = (await response.json()) as { sessions?: ZenSessionRecord[] }
      const sessions = Array.isArray(data.sessions) ? data.sessions : []
      setSessionQueue(sessions)
      showToast(
        sessions.length > 0
          ? `已从后台服务器恢复 ${sessions.length} 位同学的历史数据`
          : '本地缓存与后台服务器均暂无历史课堂数据',
      )
    } catch {
      showToast('⚠️ 重新导入历史数据失败，请检查后端服务是否已启动')
    }
  }

  /**
   * 【核心新增】"一键导出全班 Word 报告单"：循环遍历归档池中每一位同学的每一次
   * 尝试，依次调用后台 /api/save_word_report，静默完成本地建文件夹 + 写 .docx，
   * 全部处理完毕后用 Toast 汇总展示成功/失败统计，绝不依赖浏览器直接下载。
   */
  async function handleExportAllWordReports() {
    if (sessionQueue.length === 0 || isBatchExportingWord) return
    setIsBatchExportingWord(true)
    let successCount = 0
    let totalCount = 0
    try {
      for (const record of sessionQueue) {
        for (const attempt of record.attempts) {
          totalCount += 1
          const ok = await saveAttemptToWord(record.studentId, attempt)
          if (ok) successCount += 1
        }
      }
      if (totalCount === 0) {
        showToast('⚠️ 归档池中暂无任何尝试数据，无法导出 Word 报告单')
      } else if (successCount === totalCount) {
        showToast(`💾 已为全班 ${sessionQueue.length} 位同学成功导出 ${successCount} 份 Word 报告单！`)
      } else {
        showToast(`⚠️ 全班导出完成：${successCount}/${totalCount} 份成功，请检查后端服务后重试失败的部分`)
      }
    } finally {
      setIsBatchExportingWord(false)
    }
  }

  /** 导出当前选定同学的完整聚合诊断报告（JSON 结构化数据，PDF 版本可后续接入打印排版） */
  function handleExportAggregateReport() {
    if (!selectedRecord) return
    const payload = {
      exportedAt: new Date().toISOString(),
      school: getSchoolDisplayName(globalSettings),
      classGroup: getClassGroupDisplayName(globalSettings),
      studentId: selectedRecord.studentId,
      archivedAt: new Date(selectedRecord.timestamp).toLocaleString('zh-CN', { hour12: false }),
      totalAttempts: selectedRecord.attempts.length,
      attempts: selectedRecord.attempts,
      aggregateReport: selectedRecord.aggregateReport ?? null,
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `延时反馈聚合诊断_${selectedRecord.studentId || 'unknown'}_${selectedRecord.timestamp}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  /** 生成简易折线图坐标点：把评分序列（0-100）映射到 300x72 的 SVG viewBox 内 */
  const trendPolylinePoints = useMemo(() => {
    if (validScores.length < 2) return ''
    const width = 300
    const height = 72
    const step = width / (attemptScores.length - 1 || 1)
    return attemptScores
      .map((score, idx) => {
        const safeScore = typeof score === 'number' ? score : 50
        const x = Math.round(idx * step)
        const y = Math.round(height - (safeScore / 100) * height)
        return `${x},${y}`
      })
      .join(' ')
  }, [attemptScores, validScores.length])

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
      <AnimatePresence mode="wait">
        {viewMode === 'capture' ? (
          <motion.div
            key="capture"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
            className="flex flex-col gap-6"
          >
            {/* ============================ 顶栏控制区：对齐A组输入 + 核心复盘触发开关 ============================ */}
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
                        disabled={isRecording || isArchiving}
                        onClick={() => setVideoSourceMode(option.id)}
                        className={`relative rounded-full px-3.5 py-2 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 sm:text-sm ${
                          active ? 'text-white' : 'text-white/50 hover:text-white/80'
                        }`}
                      >
                        {active && (
                          <motion.span
                            layoutId="zen-video-source-pill"
                            className="absolute inset-0 rounded-full bg-teal-500/25 ring-1 ring-teal-400/40"
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

                {/* 本地 MP4 文件选择胶囊：仅在「本地视频分析」模式下显示 */}
                {videoSourceMode === 'file' && (
                  <>
                    <button
                      type="button"
                      disabled={isRecording || isArchiving || isUploading}
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
                            : '点击选择本地 MP4 文件'}
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

                {/* 学号锁定胶囊：锁定后输入框禁止编辑，下方主按钮切换为「记录本次尝试」 */}
                <div className="flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-2 backdrop-blur-xl">
                  <User className="h-4 w-4 flex-shrink-0 text-white/40" />
                  <input
                    type="text"
                    value={studentId}
                    onChange={(e) => setStudentId(e.target.value)}
                    disabled={isLocked || isRecording || isArchiving}
                    placeholder='请输入学生编号/学号，如 "No.07"'
                    className="w-40 bg-transparent text-sm text-white outline-none placeholder:text-white/30 disabled:opacity-60 sm:w-52"
                  />
                  {!isLocked ? (
                    <button
                      type="button"
                      onClick={handleLockStudentId}
                      disabled={!studentId.trim()}
                      title="锁定学号，开始为该生连续记录多次尝试"
                      className="flex items-center gap-1 rounded-full bg-teal-400/90 px-2.5 py-1 text-xs font-semibold text-black transition hover:bg-teal-300 active:scale-95 disabled:cursor-not-allowed disabled:bg-white/10 disabled:text-white/30"
                    >
                      <Lock className="h-3 w-3" />
                      锁定
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={handleUnlockStudentId}
                      disabled={isRecording || isArchiving}
                      title="解锁学号，可重新填写"
                      className="flex items-center gap-1 rounded-full bg-white/10 px-2.5 py-1 text-xs font-medium text-white/70 transition hover:bg-white/20 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      <Unlock className="h-3 w-3" />
                      解锁
                    </button>
                  )}
                </div>
              </div>

              {/* 核心主触发开关：所有人测试完成 -> 保存归档 + 进入集中复盘看板 */}
              <button
                type="button"
                disabled={sessionQueue.length === 0}
                onClick={handleEnterReviewMode}
                className={`flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-semibold transition active:scale-95 ${
                  sessionQueue.length > 0
                    ? 'bg-gradient-to-r from-teal-400 to-sky-500 text-black shadow-[0_0_30px_rgba(45,212,191,0.45)] hover:brightness-110'
                    : 'cursor-not-allowed bg-white/10 text-white/30'
                }`}
              >
                <span className="inline-flex flex-shrink-0">
                  <Flag className="h-4 w-4" />
                </span>
                🏁 所有人测试完成 (进入课前复盘看板)
                {sessionQueue.length > 0 && (
                  <span className="ml-1 rounded-full bg-black/25 px-2 py-0.5 text-xs">{sessionQueue.length}</span>
                )}
              </button>
            </section>

            {connectionError && (
              <div className="rounded-2xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
                {connectionError}
              </div>
            )}

            {/* ============================ 静默采集视口：低调呼吸荧光边框，绝无即时干预 ============================ */}
            <div className="relative">
              <motion.div
                animate={{
                  boxShadow: [
                    '0 0 20px rgba(45,212,191,0.25), inset 0 0 20px rgba(45,212,191,0.08)',
                    '0 0 46px rgba(56,189,248,0.45), inset 0 0 30px rgba(56,189,248,0.14)',
                    '0 0 20px rgba(45,212,191,0.25), inset 0 0 20px rgba(45,212,191,0.08)',
                  ],
                }}
                transition={{ duration: 3.2, repeat: Infinity, ease: 'easeInOut' }}
                className="relative min-h-[520px] overflow-hidden rounded-3xl border border-teal-400/30 bg-gradient-to-br from-zinc-900 via-black to-zinc-900"
              >
                {frameImage ? (
                  <img
                    src={frameImage}
                    alt="静默采集画面"
                    className="absolute inset-0 h-full w-full bg-black object-contain"
                    onError={() => setFrameImage(null)}
                  />
                ) : (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-white/25">
                    <span className="inline-flex flex-shrink-0">
                      {videoSourceMode === 'file' ? <FileVideo className="h-16 w-16" /> : <Camera className="h-16 w-16" />}
                    </span>
                    <p className="max-w-sm text-center text-sm">
                      {!isLocked
                        ? '请先在上方输入学号并点击「锁定」，锁定后即可开始为该生连续记录 2~3 次尝试'
                        : isRecording
                          ? '数据采集中，请专注于动作发力本身……'
                          : `学号已锁定，点击下方「记录本次尝试」开始 Attempt #${currentAttempts.length + 1}`}
                    </p>
                  </div>
                )}

                {/* 【B组科研红线】刻意不展示任何角度红/黄/绿即时警告卡，也不展示AI聊天气泡，
                    仅保留一个极简的低调状态徽标，提示"当前处于静默采集中" */}
                <div className="absolute right-5 top-5 flex items-center gap-2 rounded-full border border-white/10 bg-black/40 px-3 py-1.5 text-xs text-teal-200 backdrop-blur-xl">
                  <span className={`h-1.5 w-1.5 rounded-full ${isRecording ? 'animate-pulse bg-teal-400' : 'bg-white/30'}`} />
                  {isRecording ? '静默采集中' : isArchiving ? '正在静默生成本次尝试报告…' : isConnected ? '已连接' : '待机'}
                </div>

                {/* 学号锁定状态徽标：左上角显示当前正在为哪位同学连续记录 */}
                {isLocked && (
                  <div className="absolute left-5 top-5 flex items-center gap-2 rounded-full border border-teal-400/30 bg-black/40 px-3 py-1.5 text-xs text-white/80 backdrop-blur-xl">
                    <Lock className="h-3 w-3 text-teal-300" />
                    {studentId || '未填写编号'} · 已录入 {currentAttempts.length} 次尝试
                  </div>
                )}
              </motion.div>
            </div>

            {/* ============================ 采集操作按钮区 ============================ */}
            <section className="flex flex-col items-center gap-4 rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-xl sm:flex-row sm:flex-wrap sm:justify-center sm:gap-4">
              {!isRecording ? (
                <button
                  type="button"
                  onClick={handleStartAttemptRecording}
                  disabled={isStartAttemptDisabled}
                  title={!isLocked ? '请先锁定学号' : isStartAttemptDisabled ? '请先选择并等待本地 MP4 文件上传完成' : undefined}
                  className="flex items-center gap-2 rounded-full bg-teal-400 px-6 py-3 text-sm font-semibold text-black transition hover:bg-teal-300 active:scale-95 disabled:cursor-not-allowed disabled:bg-white/10 disabled:text-white/30"
                >
                  <span className="inline-flex flex-shrink-0">
                    {isArchiving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                  </span>
                  {isArchiving ? '正在静默生成本次尝试报告…' : `⏺️ 记录本次尝试 (Attempt #${currentAttempts.length + 1})`}
                </button>
              ) : (
                <button
                  type="button"
                  onClick={handleFinishAttemptRecording}
                  disabled={isArchiving}
                  className="flex items-center gap-2 rounded-full bg-sky-500 px-6 py-3 text-sm font-semibold text-white transition hover:bg-sky-400 active:scale-95 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <span className="inline-flex flex-shrink-0">
                    <Square className="h-4 w-4" />
                  </span>
                  ⏹️ 结束本次尝试
                </button>
              )}

              <button
                type="button"
                onClick={handleFinishCurrentStudent}
                disabled={isFinishStudentDisabled}
                className="flex items-center gap-2 rounded-full border border-amber-400/40 bg-amber-400/15 px-6 py-3 text-sm font-semibold text-amber-200 transition hover:bg-amber-400/25 active:scale-95 disabled:cursor-not-allowed disabled:border-white/10 disabled:bg-white/5 disabled:text-white/30"
              >
                <span className="inline-flex flex-shrink-0">
                  <SkipForward className="h-4 w-4" />
                </span>
                ⏭️ 完成该生测试 (换下一位)
              </button>

              <p className="max-w-md text-center text-xs leading-relaxed text-white/30 sm:text-left">
                课中严格执行"结构性沉默"：本页面绝不弹出即时角度判定或 AI 对话提示，所有诊断数据静默存入本地归档池，待课后统一复盘。
              </p>
            </section>

            {/* 右下角低调的脉冲提示胶囊堆栈 */}
            <div className="fixed bottom-6 right-6 flex flex-col items-end gap-2">
              {isLocked && currentAttempts.length > 0 && (
                <div className="flex items-center gap-3 rounded-2xl border border-teal-400/20 bg-black/60 px-4 py-2.5 backdrop-blur-xl">
                  <span className="relative flex h-2.5 w-2.5">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-sky-400 opacity-75" />
                    <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-sky-400" />
                  </span>
                  <span className="text-xs tabular-nums text-white/60">
                    {studentId || '未填写编号'}：已静默录入 {currentAttempts.length} 次尝试
                  </span>
                </div>
              )}
              <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-2.5 backdrop-blur-xl">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-teal-400 opacity-75" />
                  <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-teal-400" />
                </span>
                <span className="text-xs tabular-nums text-white/40">已归档 {sessionQueue.length} 人次测试记录</span>
              </div>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="review"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
            className="flex flex-col gap-6"
          >
            {/* ============================ 复盘看板顶栏：返回按钮 ============================ */}
            <section className="flex items-center justify-between rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-xl">
              <button
                type="button"
                onClick={() => setViewMode('capture')}
                className="flex items-center gap-2 rounded-full bg-white/10 px-4 py-2 text-sm font-medium text-white transition hover:bg-white/20 active:scale-95"
              >
                <span className="inline-flex flex-shrink-0">
                  <ArrowLeft className="h-4 w-4" />
                </span>
                返回继续采集
              </button>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2 text-sm text-white/50">
                  <span className="inline-flex flex-shrink-0">
                    <Users className="h-4 w-4 text-teal-300" />
                  </span>
                  课后集中复盘看板 · 共 {sessionQueue.length} 位同学
                </div>
                {/* 【核心新增】一键导出全班 Word 报告单：循环为全班每名同学生成规范的本地 Word 处方单 */}
                <button
                  type="button"
                  onClick={() => void handleExportAllWordReports()}
                  disabled={sessionQueue.length === 0 || isBatchExportingWord}
                  className="flex items-center gap-2 rounded-full bg-gradient-to-r from-teal-400 to-sky-500 px-4 py-2 text-sm font-semibold text-black shadow-[0_0_24px_rgba(45,212,191,0.35)] transition hover:brightness-110 active:scale-95 disabled:cursor-not-allowed disabled:from-white/10 disabled:to-white/10 disabled:text-white/30 disabled:shadow-none"
                >
                  <span className="inline-flex flex-shrink-0">
                    {isBatchExportingWord ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
                  </span>
                  {isBatchExportingWord ? '正在批量生成中…' : '💾 一键导出全班 Word 报告单'}
                </button>
              </div>
            </section>

            {/* ============================ 便当盒主体：左25%编号池 + 右75%多趟聚合复盘 ============================ */}
            <div className="flex flex-col gap-6 lg:flex-row lg:items-start">
              {/* -------- 左侧便当盒：编号池导航 + 清空/重新导入历史数据（约 25% 宽度） -------- */}
              <aside className="w-full flex-shrink-0 rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-xl lg:w-[25%]">
                <div className="mb-3 flex items-center justify-between">
                  <h3 className="flex items-center gap-2 text-sm font-semibold text-white/80">
                    <span className="inline-flex flex-shrink-0">
                      <Users className="h-4 w-4 text-teal-300" />
                    </span>
                    学生编号池
                  </h3>
                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => void handleReimportHistory()}
                      title="重新导入历史课堂数据"
                      className="flex h-7 w-7 items-center justify-center rounded-full bg-white/10 text-white/60 transition hover:bg-white/20 hover:text-white active:scale-95"
                    >
                      <RotateCcw className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={handleClearHistory}
                      disabled={sessionQueue.length === 0}
                      title="清空历史课堂数据"
                      className="flex h-7 w-7 items-center justify-center rounded-full bg-white/10 text-white/60 transition hover:bg-rose-500/30 hover:text-rose-200 active:scale-95 disabled:cursor-not-allowed disabled:opacity-30"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>

                {sessionQueue.length === 0 ? (
                  <p className="rounded-2xl bg-black/20 p-4 text-center text-xs text-white/30">
                    暂无任何静默采集记录，请返回采集模式先完成至少一位同学的测试。
                  </p>
                ) : (
                  <div className="flex flex-col gap-2">
                    {sessionQueue.map((record, index) => {
                      const isActive = record.id === selectedRecordId
                      return (
                        <button
                          key={record.id}
                          type="button"
                          onClick={() => handleSelectRecord(record.id)}
                          className={`flex items-center justify-between rounded-2xl border px-3.5 py-3 text-left transition ${
                            isActive
                              ? 'border-teal-400/40 bg-teal-400/15 text-white'
                              : 'border-white/5 bg-black/20 text-white/60 hover:bg-white/10'
                          }`}
                        >
                          <span className="flex flex-col">
                            <span className="text-sm font-medium">
                              No.{String(index + 1).padStart(2, '0')} {record.studentId}
                            </span>
                            <span className="text-[11px] text-white/35">共计 {record.attempts.length} 次尝试</span>
                          </span>
                          <span className="inline-flex flex-shrink-0">
                            <ChevronRight className={`h-4 w-4 ${isActive ? 'text-teal-300' : 'text-white/20'}`} />
                          </span>
                        </button>
                      )
                    })}
                  </div>
                )}
              </aside>

              {/* -------- 右侧主便当盒：多趟聚合诊断中心（约 75% 宽度） -------- */}
              <div className="flex w-full flex-col gap-5 lg:w-[75%]">
                {!selectedRecord || selectedAttempts.length === 0 ? (
                  <section className="flex min-h-[400px] flex-col items-center justify-center gap-3 rounded-3xl border border-white/10 bg-white/5 p-8 text-center backdrop-blur-xl">
                    <span className="inline-flex flex-shrink-0">
                      <ScanFace className="h-12 w-12 text-white/20" />
                    </span>
                    <p className="text-sm text-white/40">请先在左侧选择一位同学，查看其多趟聚合诊断报告。</p>
                  </section>
                ) : (
                  <>
                    {/* 档案头信息 */}
                    <section className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-xl">
                      <div className="flex flex-wrap items-center justify-between gap-4">
                        <div className="flex items-center gap-4">
                          <div className="flex h-16 w-16 flex-shrink-0 flex-col items-center justify-center rounded-2xl bg-teal-400/20">
                            <span className="text-2xl font-bold text-teal-300">
                              {selectedRecord.aggregateReport?.stabilityScore ?? clientStabilityScore}
                            </span>
                            <span className="text-[9px] text-teal-300/70">稳定性得分</span>
                          </div>
                          <div>
                            <p className="text-lg font-semibold text-white">{selectedRecord.studentId}</p>
                            <p className="flex items-center gap-1.5 text-xs text-white/40">
                              <GraduationCap className="h-3.5 w-3.5" />
                              {getSchoolDisplayName(globalSettings)} · {getClassGroupDisplayName(globalSettings)}
                            </p>
                          </div>
                        </div>
                        <div className="flex flex-col items-end gap-1 text-xs text-white/40">
                          <span className="flex items-center gap-1.5">
                            <Clock className="h-3.5 w-3.5" />
                            {new Date(selectedRecord.timestamp).toLocaleString('zh-CN', { hour12: false })}
                          </span>
                          <span>本节课共完成 {selectedAttempts.length} 次尝试</span>
                        </div>
                      </div>
                    </section>

                    {/* 上方盒：发力稳定性与趋势 */}
                    <section className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-xl">
                      <h4 className="mb-3 flex items-center gap-2 text-sm font-semibold text-white/80">
                        <span className="inline-flex flex-shrink-0">
                          <TrendingUp className="h-4 w-4 text-teal-300" />
                        </span>
                        发力稳定性与趋势 · Attempt 1 → Attempt {selectedAttempts.length}
                      </h4>

                      <div className="flex flex-col gap-4 lg:flex-row lg:items-center">
                        {/* 横向对齐的每趟尝试评分卡片 */}
                        <div className="flex flex-1 flex-wrap items-center gap-2">
                          {selectedAttempts.map((attempt, idx) => {
                            const score = attempt.reportData?.score ?? null
                            const bucket = getScoreBucket(score)
                            const style = SCORE_BUCKET_STYLE[bucket]
                            return (
                              <div key={attempt.attemptNumber} className="flex items-center gap-2">
                                <div className={`flex flex-col items-center gap-0.5 rounded-2xl px-3.5 py-2.5 ring-1 ${style.bg} ${style.ring}`}>
                                  <span className="text-[10px] text-white/40">Attempt #{attempt.attemptNumber}</span>
                                  <span className={`text-lg font-bold ${style.text}`}>{score ?? '--'}</span>
                                  <span className="flex items-center gap-1 text-[10px] text-white/40">
                                    <span className={`h-1.5 w-1.5 rounded-full ${style.dot}`} />
                                    {style.label}
                                  </span>
                                </div>
                                {idx < selectedAttempts.length - 1 && <ChevronRight className="h-4 w-4 flex-shrink-0 text-white/20" />}
                              </div>
                            )
                          })}
                        </div>

                        {/* 简易趋势折线图：至少 2 趟有效评分数据才绘制 */}
                        {trendPolylinePoints ? (
                          <svg viewBox="0 0 300 72" className="h-16 w-full max-w-[300px] flex-shrink-0 rounded-xl bg-black/20 lg:w-[300px]">
                            <polyline points={trendPolylinePoints} fill="none" stroke="#2dd4bf" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                        ) : (
                          <div className="flex h-16 w-full max-w-[300px] flex-shrink-0 items-center justify-center rounded-xl bg-black/20 text-[11px] text-white/25 lg:w-[300px]">
                            至少完成 2 次尝试才能绘制趋势折线图
                          </div>
                        )}
                      </div>
                    </section>

                    {/* 下方左右分栏：左下大盒（最佳击球关键帧） + 右下长立盒（AIGC聚合处方） */}
                    <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
                      {/* 左下大盒：教师可点选切换的最佳临床击球截图 */}
                      <section className="flex flex-col gap-2 rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-xl">
                        <h4 className="mb-1 flex items-center gap-2 text-sm font-semibold text-white/80">
                          <span className="inline-flex flex-shrink-0">
                            <Crosshair className="h-4 w-4 text-teal-300" />
                          </span>
                          最佳临床击球截图 · 生物力学标注图
                        </h4>
                        <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-black/40">
                          {bestAttempt?.impactFrameBase64 ? (
                            <img
                              src={bestAttempt.impactFrameBase64}
                              alt="击球瞬间生物力学诊断关键帧"
                              className="h-full w-full object-contain"
                            />
                          ) : (
                            <div className="flex aspect-video flex-col items-center justify-center gap-2 text-white/30">
                              <span className="inline-flex flex-shrink-0">
                                <ScanFace className="h-10 w-10" />
                              </span>
                              <p className="max-w-[16rem] text-center text-xs">
                                该次尝试未能捕捉到有效击球关键帧（可能全程未检测到完整人体姿态）。
                              </p>
                            </div>
                          )}
                          <span className="absolute left-3 top-3 flex items-center gap-1.5 rounded-full border border-white/10 bg-black/60 px-2.5 py-1 text-[10px] text-teal-300 backdrop-blur-xl">
                            <Crosshair className="h-3 w-3" />
                            矢量标注 · 髋-膝-踝动力链
                          </span>
                          {effectiveBestAttemptIndex === autoBestAttemptIndex && (
                            <span className="absolute right-3 top-3 flex items-center gap-1 rounded-full border border-amber-300/30 bg-black/60 px-2.5 py-1 text-[10px] text-amber-200 backdrop-blur-xl">
                              <Award className="h-3 w-3" />
                              系统自动推荐
                            </span>
                          )}
                        </div>

                        {/* 教师可点选任意一趟尝试，覆盖系统自动挑选的"最佳" */}
                        {selectedAttempts.length > 1 && (
                          <div className="mt-1 flex flex-wrap items-center gap-1.5">
                            <span className="text-[11px] text-white/30">点选切换：</span>
                            {selectedAttempts.map((attempt, idx) => (
                              <button
                                key={attempt.attemptNumber}
                                type="button"
                                onClick={() => handleOverrideBestAttempt(selectedRecord.id, idx)}
                                className={`rounded-full px-3 py-1 text-[11px] font-medium transition ${
                                  idx === effectiveBestAttemptIndex
                                    ? 'bg-teal-400/25 text-teal-200 ring-1 ring-teal-400/40'
                                    : 'bg-black/20 text-white/40 hover:bg-white/10'
                                }`}
                              >
                                Attempt #{attempt.attemptNumber}
                              </button>
                            ))}
                          </div>
                        )}

                        {/* 实验B组：离线雷达 / 幽灵骨架 + 8 大量纲卡片 */}
                        <div className="mt-3 rounded-2xl border border-white/10 bg-black/25 p-3">
                          <MetricCardList
                            renderMode="GROUP_B"
                            scoreDetail={bestAttempt?.reportData?.scoreDetail ?? MOCK_SCORE_DETAIL_V31}
                            tImpact={
                              bestAttempt?.reportData?.t_impact ??
                              bestAttempt?.reportData?.tImpact ??
                              MOCK_SCORE_DETAIL_V31.t_impact ??
                              null
                            }
                            radarScores={
                              bestAttempt?.reportData?.scoreDetail?.radar_scores ?? MOCK_RADAR_SCORES
                            }
                            compareRadarScores={MOCK_RADAR_SCORES_COMPARE}
                          />
                        </div>
                      </section>

                      {/* 右下长立盒：DeepSeek 跨课时聚合诊断处方 */}
                      <section className="flex flex-col gap-4 rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-xl">
                        <h4 className="flex items-center gap-2 text-sm font-semibold text-white/80">
                          <span className="inline-flex flex-shrink-0">
                            <Sparkles className="h-4 w-4 text-teal-300" />
                          </span>
                          DeepSeek 跨课时聚合诊断处方
                        </h4>
                        <div className="flex-1 whitespace-pre-line rounded-2xl bg-black/20 p-4 text-sm leading-relaxed text-white/85">
                          {aggregateLoadingId === selectedRecord.id ? (
                            <span className="flex items-center gap-2 text-white/40">
                              <Loader2 className="h-4 w-4 animate-spin" />
                              正在基于该生本节课 {selectedAttempts.length} 次尝试生成跨课时诊断建议…
                            </span>
                          ) : (
                            selectedRecord.aggregateReport?.fullText ?? '暂无聚合诊断报告（可能后端服务未启动，或该生尝试数据不足）。'
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => void handleGenerateAggregateReport(selectedRecord)}
                            disabled={aggregateLoadingId === selectedRecord.id}
                            className="flex flex-1 items-center justify-center gap-2 rounded-2xl border border-white/10 bg-white/5 py-2.5 text-xs font-medium text-white/70 transition hover:bg-white/10 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40"
                          >
                            <RotateCcw className="h-3.5 w-3.5" />
                            重新生成
                          </button>
                          <button
                            type="button"
                            onClick={handleExportAggregateReport}
                            className="flex flex-1 items-center justify-center gap-2 rounded-2xl bg-teal-400 py-2.5 text-sm font-semibold text-black transition hover:bg-teal-300 active:scale-95"
                          >
                            <span className="inline-flex flex-shrink-0">
                              <Download className="h-4 w-4" />
                            </span>
                            📥 导出聚合诊断 (PDF/JSON)
                          </button>
                        </div>
                      </section>
                    </div>
                  </>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ============================ Apple 风格浮动 Toast 提示条 ============================ */}
      <AnimatePresence>
        {toast && (
          <motion.div
            key={toast.id}
            initial={{ opacity: 0, y: 24, scale: 0.92 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 24, scale: 0.92 }}
            transition={{ type: 'spring', stiffness: 300, damping: 26 }}
            className="fixed bottom-24 left-1/2 z-[60] flex max-w-md -translate-x-1/2 items-center gap-3 rounded-full border border-white/10 bg-black/80 px-5 py-3.5 shadow-2xl backdrop-blur-2xl"
          >
            <span className="inline-flex flex-shrink-0">
              <CheckCircle2 className="h-4 w-4 text-teal-300" />
            </span>
            <p className="text-sm text-white/90">{toast.message}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
