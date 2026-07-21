// Mock 数据文件：为界面开发提供模拟数据，后续可替换为真实后端接口数据

import type {
  AngleTimeSeriesPoint,
  ClassInfo,
  ErrorTypeDistribution,
  ExperimentGroup,
  FinalDiagnosisReport,
  GlobalSettings,
  GlobalTrainingRecord,
  RadarScores,
  School,
  ScoreDetailPayload,
  ScoreRecord,
  StudentInfo,
  TeachingSuggestion,
  ThresholdHitStats,
  ThresholdLevel,
  ZenSessionRecord,
} from './types'

/** 根据膝关节角度计算三级阈值等级（140-160 绿；130-140 或 160-170 黄；其余红） */
export function getThresholdLevel(angle: number): ThresholdLevel {
  if (angle >= 140 && angle <= 160) return 'green'
  if ((angle >= 130 && angle < 140) || (angle > 160 && angle <= 170)) return 'yellow'
  return 'red'
}

/** 各等级对应的主题色（对齐 V2.5 Traffic-Light System） */
export const LEVEL_COLOR_MAP: Record<ThresholdLevel, { text: string; bg: string; ring: string; glow: string }> = {
  green: {
    text: 'text-[var(--GREEN_OPTIMAL)]',
    bg: 'bg-[color-mix(in_srgb,var(--GREEN_OPTIMAL)_18%,transparent)]',
    ring: 'ring-[color-mix(in_srgb,var(--GREEN_OPTIMAL)_50%,transparent)]',
    glow: 'traffic-glow-green',
  },
  yellow: {
    text: 'text-[var(--YELLOW_APPROACHING)]',
    bg: 'bg-[color-mix(in_srgb,var(--YELLOW_APPROACHING)_18%,transparent)]',
    ring: 'ring-[color-mix(in_srgb,var(--YELLOW_APPROACHING)_50%,transparent)]',
    glow: 'traffic-glow-yellow',
  },
  red: {
    text: 'text-[var(--RED_DEVIATED)]',
    bg: 'bg-[color-mix(in_srgb,var(--RED_DEVIATED)_18%,transparent)]',
    ring: 'ring-[color-mix(in_srgb,var(--RED_DEVIATED)_50%,transparent)]',
    glow: 'traffic-glow-red',
  },
}

export const LEVEL_LABEL_MAP: Record<ThresholdLevel, string> = {
  green: '达标',
  yellow: '接近',
  red: '错误',
}

/**
 * V3.1 Sprint 1 五维雷达测试数据（对齐后端 DeterministicScorer.radar_scores）。
 * 模拟一次偏「支撑稳、折叠佳、鞭打偏弱、踝锁中等」的儿童射门。
 */
export const MOCK_RADAR_SCORES: RadarScores = {
  support_stability: 18.2,
  backswing_folding: 16.5,
  ankle_rigidity: 15.0,
  whipping_velocity: 12.4,
  approach_rhythm: 18.0,
}

/** 对比 Attempt：鞭打提升、折叠略降 */
export const MOCK_RADAR_SCORES_COMPARE: RadarScores = {
  support_stability: 17.5,
  backswing_folding: 14.0,
  ankle_rigidity: 20.0,
  whipping_velocity: 18.7,
  approach_rhythm: 19.0,
}

/**
 * 模拟完整 scoreDetail（含 radar_scores），供 MetricPanel / MetricCardList 联调。
 */
export const MOCK_SCORE_DETAIL_V31: ScoreDetailPayload = {
  TotalScore: 80.1,
  t_impact: 42,
  base_score: 100,
  total_penalty: 19.9,
  scoring_engine: 'DeterministicScorer_V3.1',
  llm_participated: false,
  radar_scores: MOCK_RADAR_SCORES,
  indicators: {
    distance_cm: {
      value: 17.2,
      unit: 'cm',
      status: 'GREEN_OPTIMAL',
      penalty: 0,
      extreme_frame_index: 39,
    },
    toe_angle: {
      value: 8.0,
      unit: 'deg',
      status: 'GREEN_OPTIMAL',
      penalty: 0,
      extreme_frame_index: 39,
    },
    max_folding_angle: {
      value: 78.0,
      unit: 'deg',
      status: 'GREEN_OPTIMAL',
      penalty: 0,
      extreme_frame_index: 34,
    },
    whipping_velocity: {
      value: 279.0,
      unit: 'deg/s',
      status: 'YELLOW_APPROACHING',
      penalty: 4.2,
      extreme_frame_index: 40,
    },
    impact_knee_angle: {
      value: 148.0,
      unit: 'deg',
      status: 'GREEN_OPTIMAL',
      penalty: 0,
      extreme_frame_index: 42,
    },
    ankle_rigidity: {
      value: 3.2,
      variance: 3.2,
      unit: 'variance',
      status: 'YELLOW_APPROACHING',
      penalty: 4.1,
      extreme_frame_index: 42,
    },
    support_knee_angle: {
      value: 152.0,
      unit: 'deg',
      status: 'GREEN_OPTIMAL',
      penalty: 0,
      extreme_frame_index: 39,
    },
    hip_torsion_angle: {
      value: 22.0,
      unit: 'deg',
      status: 'GREEN_OPTIMAL',
      penalty: 0,
      extreme_frame_index: 42,
    },
  },
}

/** DeepSeek 模拟指导语列表（具身隐喻、积极意图原则） */
export const MOCK_GUIDANCE_TEXTS: string[] = [
  '很棒的尝试！下次触球时，试着让摆动腿像拉满的弓弦一样再多蓄一点力～',
  '支撑脚落地很稳，像大树的根扎稳了地面，继续保持这种感觉！',
  '触球瞬间腿部可以再放松一点，想象小腿是一条鞭子轻轻甩出去。',
  '髋部转动的幅度刚刚好，身体像一个灵活的陀螺在旋转，非常棒！',
  '踝关节锁得很稳，像穿了一只结实的小靴子，下次再试试保持住这个感觉。',
]

/** 实时评分历史模拟数据 */
export const MOCK_SCORE_HISTORY: ScoreRecord[] = [
  { id: 's1', kneeAngle: 146, level: 'green', timestamp: '10:02:11' },
  { id: 's2', kneeAngle: 137, level: 'yellow', timestamp: '10:02:34' },
  { id: 's3', kneeAngle: 152, level: 'green', timestamp: '10:03:02' },
  { id: 's4', kneeAngle: 124, level: 'red', timestamp: '10:03:28' },
  { id: 's5', kneeAngle: 158, level: 'green', timestamp: '10:03:55' },
  { id: 's6', kneeAngle: 165, level: 'yellow', timestamp: '10:04:20' },
]

/** 学校列表 */
export const MOCK_SCHOOLS: School[] = [
  { id: 'school-1', name: '学校一' },
  { id: 'school-2', name: '学校二' },
]

/** 实验组别列表 */
export const MOCK_GROUPS: ExperimentGroup[] = [
  { id: 'group-a', name: '实验A组' },
  { id: 'group-b', name: '实验B组' },
  { id: 'group-c', name: '常规C组' },
]

/** 班级列表 */
export const MOCK_CLASSES: ClassInfo[] = [
  { id: 'class-1', name: '五年一班', schoolId: 'school-1', groupId: 'group-a' },
  { id: 'class-2', name: '五年二班', schoolId: 'school-1', groupId: 'group-b' },
  { id: 'class-3', name: '五年三班', schoolId: 'school-1', groupId: 'group-c' },
  { id: 'class-4', name: '五年四班', schoolId: 'school-2', groupId: 'group-a' },
  { id: 'class-5', name: '五年五班', schoolId: 'school-2', groupId: 'group-b' },
]

/** 学生列表 */
export const MOCK_STUDENTS: StudentInfo[] = [
  { id: 'stu-1', studentNumber: 'B001', name: '张三', classId: 'class-1' },
  { id: 'stu-2', studentNumber: 'B002', name: '李四', classId: 'class-1' },
  { id: 'stu-3', studentNumber: 'B003', name: '王五', classId: 'class-2' },
  { id: 'stu-4', studentNumber: 'B004', name: '赵六', classId: 'class-2' },
]

/** 班级合规率（百分比） */
export const MOCK_COMPLIANCE_RATE = 78

/** 错误类型分布（用于柱状图/饼图） */
export const MOCK_ERROR_DISTRIBUTION: ErrorTypeDistribution[] = [
  { level: 'green', label: '达标 (140°-160°)', count: 156 },
  { level: 'yellow', label: '接近 (130°-140° / 160°-170°)', count: 64 },
  { level: 'red', label: '错误 (<130° 或 >170°)', count: 32 },
]

/** 生成模拟的时间序列膝关节角度数据（用于折线图，Y轴 100-180） */
export function generateMockTimeSeries(pointCount = 40): AngleTimeSeriesPoint[] {
  const points: AngleTimeSeriesPoint[] = []
  let base = 148
  for (let i = 0; i < pointCount; i++) {
    // 模拟自然波动，偶尔出现较大偏差
    const noise = (Math.random() - 0.5) * 14
    const spike = Math.random() < 0.12 ? (Math.random() - 0.5) * 30 : 0
    base = Math.min(178, Math.max(102, base * 0.6 + (148 + noise + spike) * 0.4))
    const minutes = String(Math.floor(i / 2)).padStart(2, '0')
    const seconds = String((i % 2) * 30).padStart(2, '0')
    points.push({ time: `10:${minutes}:${seconds}`, angle: Math.round(base) })
  }
  return points
}

export const MOCK_TIME_SERIES: AngleTimeSeriesPoint[] = generateMockTimeSeries()

/** AI 教学处方建议模拟数据 */
export const MOCK_TEACHING_SUGGESTIONS: TeachingSuggestion[] = [
  {
    id: 'sug-1',
    title: '强化摆动腿鞭打感知',
    content:
      '该班级近三成动作膝关节角度低于130°，建议增加"弹力带抗阻摆腿"专项练习，帮助学生建立"腿部像鞭子甩出"的具身感知，逐步扩大鞭打幅度。',
    tag: '膝关节屈曲角度',
  },
  {
    id: 'sug-2',
    title: '巩固支撑脚稳定性',
    content:
      '部分学生支撑脚落位偏离球心较远，建议课堂增设"标志盘定点支撑"游戏化练习，强化"像大树扎根"的稳定站姿记忆。',
    tag: '支撑脚落位',
  },
  {
    id: 'sug-3',
    title: '髋部旋转节奏引导',
    content:
      '髋关节旋转普遍略显僵硬，建议采用慢速分解教学，引导学生想象"身体像陀螺轻盈转动"，逐步提升旋转流畅度。',
    tag: '髋关节旋转角度',
  },
]

/* ------------------------------------------------------------------ */
/* 以下为「全局教学环境设置」与「实时反馈工作台升级」新增 Mock 数据与工具函数 */
/* ------------------------------------------------------------------ */

/** 学校预设常用选项（教师仍可在此基础上自由新增自定义学校/机构名称） */
export const PRESET_SCHOOL_NAMES: string[] = ['学校一', '学校二']

/** 班级 / 实验组别预设常用选项（教师仍可在此基础上自由新增自定义分组/班级名称） */
export const PRESET_CLASS_GROUP_NAMES: string[] = ['四年级1班-实验A组', '四年级2班-实验B组', '四年级3班-常规C组']

/** 默认全局教学环境设置：默认学校一 + 四年级1班-实验A组 + 默认开启本地落盘归档总闸 */
export const DEFAULT_GLOBAL_SETTINGS: GlobalSettings = {
  schoolName: PRESET_SCHOOL_NAMES[0],
  classGroupName: PRESET_CLASS_GROUP_NAMES[0],
  enableDataArchiving: true,
}

/* ------------------------------------------------------------------ */
/* 100% 自定义学校 / 班级分组：localStorage 持久化工具函数              */
/* ------------------------------------------------------------------ */

const CUSTOM_SCHOOLS_STORAGE_KEY = 'aiff_custom_schools_v1'
const CUSTOM_CLASS_GROUPS_STORAGE_KEY = 'aiff_custom_class_groups_v1'

/** 从 localStorage 安全读取一份字符串数组，任何异常（未支持/解析失败）都静默兜底为空数组 */
function readStringListFromStorage(storageKey: string): string[] {
  try {
    const raw = window.localStorage.getItem(storageKey)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === 'string') : []
  } catch {
    return []
  }
}

/** 把一份字符串数组安全写入 localStorage，写入失败（例如隐私模式禁用存储）时静默忽略 */
function writeStringListToStorage(storageKey: string, list: string[]): void {
  try {
    window.localStorage.setItem(storageKey, JSON.stringify(list))
  } catch {
    // 静默忽略：不应因为本地存储写入失败而中断教师的正常操作流程
  }
}

/** 读取教师历史新增并持久化保存过的自定义学校/机构名称列表 */
export function loadCustomSchoolNames(): string[] {
  return readStringListFromStorage(CUSTOM_SCHOOLS_STORAGE_KEY)
}

/** 新增一个自定义学校/机构名称并持久化保存（自动去重、去除首尾空白） */
export function saveCustomSchoolName(name: string): string[] {
  const trimmed = name.trim()
  const existing = loadCustomSchoolNames()
  if (!trimmed || existing.includes(trimmed) || PRESET_SCHOOL_NAMES.includes(trimmed)) return existing
  const next = [...existing, trimmed]
  writeStringListToStorage(CUSTOM_SCHOOLS_STORAGE_KEY, next)
  return next
}

/** 读取教师历史新增并持久化保存过的自定义班级/分组名称列表 */
export function loadCustomClassGroupNames(): string[] {
  return readStringListFromStorage(CUSTOM_CLASS_GROUPS_STORAGE_KEY)
}

/** 新增一个自定义班级/分组名称并持久化保存（自动去重、去除首尾空白） */
export function saveCustomClassGroupName(name: string): string[] {
  const trimmed = name.trim()
  const existing = loadCustomClassGroupNames()
  if (!trimmed || existing.includes(trimmed) || PRESET_CLASS_GROUP_NAMES.includes(trimmed)) return existing
  const next = [...existing, trimmed]
  writeStringListToStorage(CUSTOM_CLASS_GROUPS_STORAGE_KEY, next)
  return next
}

/** 学校展示名称：现已 100% 自定义，直接返回用户填写/选择的名称，未填写时给出占位提示 */
export function getSchoolDisplayName(settings: GlobalSettings): string {
  return settings.schoolName.trim() || '未设置学校'
}

/** 班级 / 组别展示名称：现已 100% 自定义，直接返回用户填写/选择的名称，未填写时给出占位提示 */
export function getClassGroupDisplayName(settings: GlobalSettings): string {
  return settings.classGroupName.trim() || '未设置班级'
}

/** RGB 颜色线性插值 */
function lerpRgb(from: [number, number, number], to: [number, number, number], t: number): [number, number, number] {
  const clampT = Math.min(1, Math.max(0, t))
  return [
    Math.round(from[0] + (to[0] - from[0]) * clampT),
    Math.round(from[1] + (to[1] - from[1]) * clampT),
    Math.round(from[2] + (to[2] - from[2]) * clampT),
  ]
}

const COLOR_GREEN: [number, number, number] = [16, 130, 90] // 深沉的达标绿（作为卡片背景基底色）
const COLOR_YELLOW: [number, number, number] = [146, 104, 12] // 接近阈值的暖黄
const COLOR_RED: [number, number, number] = [140, 30, 50] // 错误阈值的警示红

/**
 * 根据触球瞬间膝关节角度，计算实时遥测卡的「平滑过渡」背景色。
 * 以黄金区间 [140°,160°] 的中心 150° 为基准，按偏离量分段插值，
 * 避免离散三色跳变，呈现更接近真实生理反馈的渐变效果。
 * 后续可将此函数替换为读取真实推理引擎输出的角度值。
 */
export function getSmoothAngleBackground(angle: number): string {
  const deviation = Math.abs(angle - 150)
  let rgb: [number, number, number]
  if (deviation <= 10) {
    // 140°-160° 黄金区间：纯绿基底
    rgb = COLOR_GREEN
  } else if (deviation <= 30) {
    // 绿 -> 黄 过渡带
    rgb = lerpRgb(COLOR_GREEN, COLOR_YELLOW, (deviation - 10) / 20)
  } else if (deviation <= 50) {
    // 黄 -> 红 过渡带
    rgb = lerpRgb(COLOR_YELLOW, COLOR_RED, (deviation - 30) / 20)
  } else {
    // 严重偏离：纯红警示
    rgb = COLOR_RED
  }
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`
}

/**
 * 模拟真实推理引擎输出的下一帧膝关节角度。
 * 大概率落在黄金区间附近，偶发较大偏离以模拟儿童动作的不稳定性。
 * @param previousAngle 上一帧角度，用于保持帧间连贯性（避免跳变过于突兀）
 */
export function generateNextKneeAngle(previousAngle: number): number {
  const noise = (Math.random() - 0.5) * 16
  const spikeTriggered = Math.random() < 0.18
  const spike = spikeTriggered ? (Math.random() - 0.5) * 36 : 0
  const target = 150 + noise + spike
  // 与上一帧做加权平均，模拟连续视频帧之间的运动惯性
  const next = previousAngle * 0.45 + target * 0.55
  return Math.round(Math.min(180, Math.max(100, next)))
}

/** 命中次数统计对应的教练处方文案库（按主要痛点等级挑选） */
const PRESCRIPTION_BY_LEVEL: Record<ThresholdLevel, string> = {
  green:
    '教练处方：当前发力节奏已非常稳定，建议保持射门前支撑脚膝盖微屈、身体重心前倾的准备姿势，可尝试提升摆动腿加速度以进一步巩固动作自动化。',
  yellow:
    '教练处方：建议保持射门前支撑脚膝盖微屈，想象摆动腿像拉满的弓弦一样蓄力后再释放，减少触球瞬间的角度偏移。',
  red: '教练处方：建议从静态分解动作开始练习，先固定支撑脚站位，反复体会"腿部像鞭子甩出"的具身感觉，待动作稳定后再逐步加入完整助跑衔接。',
}

/* ------------------------------------------------------------------ */
/* 延时反馈系统 (实验B组) · 跨课时双重持久化：localStorage 读写工具函数    */
/* ------------------------------------------------------------------ */

/** 延时反馈系统「学生归档池」的 localStorage 存储键名 */
export const DELAYED_FEEDBACK_SESSIONS_STORAGE_KEY = 'delayed_feedback_sessions'

/**
 * 从 localStorage 安全读取延时反馈系统的学生归档池（跨课时持久化）。
 * 【边界情况防呆】任何异常情况（首次打开尚无历史数据 / 浏览器禁用本地存储 /
 * 存储内容被意外污染成非数组 JSON）都必须静默兜底为空数组，绝不能因为
 * 一份脏数据就让整个延时反馈系统白屏崩溃。
 */
export function loadZenSessionsFromLocalStorage(): ZenSessionRecord[] {
  try {
    const raw = window.localStorage.getItem(DELAYED_FEEDBACK_SESSIONS_STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    // 再做一层字段级防呆：过滤掉缺失关键字段的脏记录，attempts 缺失时兜底为空数组
    return parsed
      .filter((item): item is ZenSessionRecord => !!item && typeof item === 'object' && typeof item.studentId === 'string')
      .map((item) => ({ ...item, attempts: Array.isArray(item.attempts) ? item.attempts : [] }))
  } catch {
    return []
  }
}

/**
 * 把当前完整的学生归档池同步写入 localStorage，实现「跨课时持久化」的
 * 前端半重保险。写入失败（例如隐私模式禁用存储、容量超限）时静默忽略，
 * 不应该因为本地存储异常而打断教练正在进行的课中采集流程。
 */
export function saveZenSessionsToLocalStorage(sessions: ZenSessionRecord[]): void {
  try {
    window.localStorage.setItem(DELAYED_FEEDBACK_SESSIONS_STORAGE_KEY, JSON.stringify(sessions))
  } catch {
    // 静默忽略：本地存储写入失败不应中断教师的正常操作流程
  }
}

/** 一键清空本地保存的历史课堂归档数据（左侧面板「清空历史数据」按钮使用） */
export function clearZenSessionsFromLocalStorage(): void {
  try {
    window.localStorage.removeItem(DELAYED_FEEDBACK_SESSIONS_STORAGE_KEY)
  } catch {
    // 静默忽略
  }
}

/* ------------------------------------------------------------------ */
/* 全局训练数据库：localStorage 极速双保险读写工具函数                    */
/* 与后端 global_training_db.json 存储完全同构的记录列表，供 App 内任意   */
/* 工作台在归档成功的第一时间本地同步一份，教练端看板优先读后端接口，       */
/* 接口不可用时回退到这份本地缓存，确保"看板看不到"这一痛点被彻底解决。     */
/* ------------------------------------------------------------------ */

/** 全局训练数据库的 localStorage 存储键名（与需求文档命名保持一致） */
export const GLOBAL_RECORDS_STORAGE_KEY = 'global_football_records'

/** 从 localStorage 安全读取全局训练数据库的完整记录列表，任何异常都静默兜底为空数组 */
export function loadGlobalRecordsFromLocalStorage(): GlobalTrainingRecord[] {
  try {
    const raw = window.localStorage.getItem(GLOBAL_RECORDS_STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.filter(
      (item): item is GlobalTrainingRecord => !!item && typeof item === 'object' && typeof item.id === 'string',
    )
  } catch {
    return []
  }
}

/** 把当前完整的全局训练数据库记录列表整体覆盖写入 localStorage */
export function saveGlobalRecordsToLocalStorage(records: GlobalTrainingRecord[]): void {
  try {
    window.localStorage.setItem(GLOBAL_RECORDS_STORAGE_KEY, JSON.stringify(records))
  } catch {
    // 静默忽略：本地存储写入失败不应中断教师/教练的正常操作流程
  }
}

/**
 * 【v3.0 新增：科研指挥中心】生物力学错误分类标签体系，与后端
 * api_server.py 的 BIOMECH_ERROR_TAXONOMY 保持完全一致，供教练端看板
 * 「集体错误热力图」按固定顺序展示各维度，即便某项分类在当前筛选范围内
 * 出现次数为 0，也依然展示一条 0% 的基线条目，保持热力图维度齐整。
 */
export const BIOMECH_ERROR_TAXONOMY: string[] = ['支撑脚位置偏离', '膝关节过度屈曲', '随摆转髋不足', '身体重心偏移']

/**
 * 追加一条新的归档记录到本地全局训练数据库缓存（自动按 id 去重，
 * 避免因网络重试等原因导致同一条记录被写入两次）。
 */
export function appendGlobalRecordToLocalStorage(record: GlobalTrainingRecord): void {
  const existing = loadGlobalRecordsFromLocalStorage()
  const next = [...existing.filter((item) => item.id !== record.id), record]
  saveGlobalRecordsToLocalStorage(next)
}

/**
 * 根据本次分析的三级阈值命中统计，生成 DeepSeek 风格的「本次综合练习诊断报告」。
 * 真实环境中，该函数应替换为对后端 AIGC 转译接口（GLM-4 / 智谱AI，temperature=0.3）的调用，
 * 当前使用规则化模板拼接以模拟低随机性、结构稳定的输出。
 */
export function buildFinalDiagnosisReport(stats: ThresholdHitStats, studentNumber: string): FinalDiagnosisReport {
  const totalAttempts = stats.green + stats.yellow + stats.red
  const safeTotal = Math.max(1, totalAttempts)
  const greenRatio = stats.green / safeTotal

  // 发力稳定性评分：以绿色占比为主，红色命中做额外扣分，贴近教研评估口径
  const score = Math.max(35, Math.min(98, Math.round(greenRatio * 100 - stats.red * 3)))

  // 判定本次主要痛点等级：红色优先，其次黄色，否则视为整体稳定
  const dominantLevel: ThresholdLevel = stats.red >= stats.yellow && stats.red > 0 ? 'red' : stats.yellow > 0 ? 'yellow' : 'green'

  const painPoint =
    dominantLevel === 'green'
      ? '本次练习动作整体稳定，未出现显著偏离黄金区间的情况。'
      : dominantLevel === 'yellow'
        ? `主要痛点：后摆腿触球瞬间膝关节屈曲角偏小（触发 ${stats.yellow} 次黄色警示）。`
        : `主要痛点：触球瞬间膝关节屈曲角显著偏离黄金区间（触发 ${stats.red} 次红色警示，${stats.yellow} 次黄色警示）。`

  const prescription = PRESCRIPTION_BY_LEVEL[dominantLevel]

  const fullText = `学号 ${studentNumber || '未填写'} 本次综合练习诊断报告\n\n发力稳定性评分：${score} 分（共采集 ${totalAttempts} 次有效触球数据）。\n${painPoint}\n${prescription}`

  return {
    score,
    totalAttempts,
    painPoint,
    prescription,
    fullText,
    generatedAt: new Date().toLocaleString('zh-CN', { hour12: false }),
  }
}
