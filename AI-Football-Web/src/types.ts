// 全局类型定义：足球AI可视化反馈系统

/** 顶部导航三个视图模式 */
export type ViewMode = 'realtime' | 'zen' | 'coach'

/** 三级阈值容错等级：绿(达标) / 黄(接近) / 红(错误) */
export type ThresholdLevel = 'green' | 'yellow' | 'red'

/** 单次动作评分历史记录条目 */
export interface ScoreRecord {
  id: string
  /** 触球瞬间膝关节屈曲角度 */
  kneeAngle: number
  /** 判定等级 */
  level: ThresholdLevel
  /** 记录时间（HH:mm:ss） */
  timestamp: string
}

/** 学校信息 */
export interface School {
  id: string
  name: string
}

/** 实验组别 */
export interface ExperimentGroup {
  id: string
  name: string
}

/** 班级信息 */
export interface ClassInfo {
  id: string
  name: string
  schoolId: string
  groupId: string
}

/** 学生信息 */
export interface StudentInfo {
  id: string
  studentNumber: string
  name: string
  classId: string
}

/** 时间序列中单个数据点（用于折线图） */
export interface AngleTimeSeriesPoint {
  /** 采样时间点标签 */
  time: string
  /** 膝关节角度 */
  angle: number
}

/** 错误类型分布（用于柱状图/饼图） */
export interface ErrorTypeDistribution {
  level: ThresholdLevel
  label: string
  count: number
}

/** AI 教学处方建议卡片 */
export interface TeachingSuggestion {
  id: string
  title: string
  content: string
  /** 关联的生物力学参数标签 */
  tag: string
}

/** API 连接状态 */
export type ApiStatus = 'online' | 'offline' | 'connecting'

/* ------------------------------------------------------------------ */
/* 以下为「全局教学环境设置」与「实时反馈工作台升级」新增类型定义         */
/* ------------------------------------------------------------------ */

/**
 * 全局教学环境设置：贯穿 Navbar 与各工作台的统一教学上下文。
 * 【100% 自定义升级】不再依赖固定的预设 ID 枚举，学校与班级/组别均直接
 * 存储用户实际填写/选择的名称字符串，配合 mockData.ts 中的 localStorage
 * 持久化工具函数，允许教师自由录入任意学校、班级、实验分组名称。
 */
export interface GlobalSettings {
  /** 学校 / 机构名称（预设选项或用户自定义输入均直接存名称本身） */
  schoolName: string
  /** 班级 / 实验组别名称（预设选项或用户自定义输入均直接存名称本身） */
  classGroupName: string
  /**
   * 【全局归档总闸】默认开启：开启后，实时反馈(A组)与延时反馈(B组)在完成一次
   * 测试后，都会静默自动调用后端 /api/save_word_report 在本机硬盘写盘生成 Word
   * 报告，并同步汇入全局训练数据库（global_training_db.json + 教练端看板）。
   */
  enableDataArchiving: boolean
}

/** 视频源模式：实时摄像头 或 本地视频文件回放分析 */
export type VideoSourceMode = 'webcam' | 'file'

/** 单次分析任务的运行状态：待机 / 分析中 / 正在结束（等待后端收尾并生成报告） / 已结束 */
export type AnalysisStatus = 'idle' | 'analyzing' | 'stopping' | 'finished'

/** 三级阈值命中次数统计（用于右侧诊断便当盒可视化） */
export interface ThresholdHitStats {
  green: number
  yellow: number
  red: number
}

/** DeepSeek 生成的本次综合练习诊断报告（分析结束后展示） */
export interface FinalDiagnosisReport {
  /** 发力稳定性综合评分（0-100） */
  score: number
  /** 本次分析总触球/采样次数 */
  totalAttempts: number
  /** 主要痛点描述（具身隐喻化表达） */
  painPoint: string
  /** 教练处方建议 */
  prescription: string
  /** 拼接完整的报告正文，用于打字机展示与导出 */
  fullText: string
  /** 报告生成时间 */
  generatedAt: string
  /**
   * 【核心新增】击球瞬间关键帧生物力学诊断标注图（Base64 data URL 字符串）。
   * 由后端 api_server.py 在整趟练习中自动捕捉右膝角速度峰值/屈曲极值帧，
   * 并用 OpenCV 叠加髋-膝-踝矢量箭头、角度弧线与重心垂直虚线后生成；
   * 若本次分析全程未能捕捉到有效人体姿态，则为 null。
   */
  impactFrameImage?: string | null
  /**
   * 【聚合诊断新增】本次分析的三级容错命中次数统计（Green/Yellow/Red），
   * 后端 /api/generate_report 一直有返回这个字段，这里补上类型声明，
   * 供延时反馈系统课后复盘时，判断"这一趟测试整体偏向红/黄/绿哪个区间"。
   */
  hitStats?: ThresholdHitStats
  /**
   * 【v4.0 科研级数据矩阵新增】本次分析全程真实测得的膝关节屈曲角度均值
   * （pose_tracker.py 逐帧真实测量值的平均，并非估算）。用于「双轴互动运动学
   * 成长期刊图」右侧蓝色虚线轴，以及保存 Word 报告时随请求体一并回填进
   * global_training_db.json 的 kneeFlexionAngle 字段。可能为 null（本次分析
   * 全程未采集到任何有效姿态数据）。
   */
  avgKneeAngle?: number | null
}

/* ------------------------------------------------------------------ */
/* 以下为「延时反馈系统 (实验B组) · 课中静默采集 -> 课后集中复盘」新增类型定义 */
/* ------------------------------------------------------------------ */

/** B组工作台当前所处的页面阶段：'capture' 静默采集模式 / 'review' 集中复盘看板模式 */
export type ZenViewMode = 'capture' | 'review'

/**
 * 【v3.0 新增】单次尝试记录：对应"某位同学连续 2~3 次踢球中的其中一次"。
 * 一位同学在课中会连续攒 2~3 次尝试，全部记录在 ZenSessionRecord.attempts[] 里，
 * 直到教练点击"完成该生测试"才整体打包归档，绝不会出现"半条尝试数据"的情况。
 */
export interface ZenAttemptRecord {
  /** 本条尝试在该生本节课内的序号，从 1 开始（Attempt #1 / #2 / #3） */
  attemptNumber: number
  /** 本次尝试完成分析的时间戳（毫秒） */
  timestamp: number
  /** 本次尝试所用的视频源（实时摄像头 / 本地视频文件） */
  videoSource: VideoSourceMode
  /** 后台 /api/generate_report 真实返回的完整诊断报告；采集失败时可能为 null，渲染前必须做空值兜底 */
  reportData: FinalDiagnosisReport | null
  /** 冗余保存一份击球关键帧 Base64 图片，等同于 reportData.impactFrameImage */
  impactFrameBase64?: string | null
}

/**
 * B组本地归档池中的一条记录：对应"某位同学本节课完整的一次测试实体"，
 * 内部打包了该生连续完成的 2~3 次尝试（attempts[]），这是「跨课时双重持久化」
 * 与「课前集中复盘」共同消费的最小单位。
 * 【字段说明，方便后续按需调整】
 *   - id：本条记录的唯一标识（前端生成，用于 React key 与选中态匹配）；
 *   - studentId：学生编号/学号（自由文本，由采集时输入框填写，用于课后归档分组）；
 *   - timestamp：教练点击"完成该生测试"完成归档的时间戳（毫秒）；
 *   - attempts：该生本节课全部尝试记录（1~N 条），必须至少含 1 条才允许归档；
 *   - aggregateReport：课后复盘看板里，DeepSeek 基于多趟尝试生成的跨次趋势诊断，
 *     首次查看时才会懒加载调用后端生成，因此初始为空，需要做空值兜底。
 */
export interface ZenSessionRecord {
  id: string
  studentId: string
  timestamp: number
  attempts: ZenAttemptRecord[]
  aggregateReport?: AggregateDiagnosisReport | null
}

/**
 * 【v3.0 新增】跨次尝试聚合诊断报告：由后端 /api/generate_aggregate_report
 * 调用 DeepSeek，基于该生本节课 2~3 次尝试的评分/三级命中趋势生成。
 */
export interface AggregateDiagnosisReport {
  /** 动作表现稳定性得分（0-100），综合多趟尝试评分的离散程度计算得出 */
  stabilityScore: number
  /** 跨次趋势总结（具身隐喻化，例如"越踢越稳"或"后段体力下降造成的变形"） */
  trendDescription: string
  /** 面向下节课的具身隐喻化教学处方建议 */
  prescription: string
  /** 拼接完整的报告正文，用于展示与导出 */
  fullText: string
  /** 报告生成时间 */
  generatedAt: string
}

/* ------------------------------------------------------------------ */
/* 以下为「全局训练数据库」新增类型定义：数据落地闭环 + 教练端看板消费          */
/* ------------------------------------------------------------------ */

/** 单条归档记录所属的测试模式：实时反馈(A组) / 延时反馈(B组) */
export type FeedbackRecordType = 'realtime' | 'delayed'

/**
 * 【核心新增】全局训练数据库单条记录：每当一份 Word 报告成功写盘归档，
 * 后端 /api/save_word_report 都会自动追加一条这样的完整记录进
 * global_training_db.json，同时前端把同一份记录同步写进
 * localStorage['global_football_records'] 作为极速双保险，
 * 供教练端数据看板 (CoachDashboard.tsx) 统一读取消费。
 */
export interface GlobalTrainingRecord {
  /** 记录唯一标识（后端生成的 UUID） */
  id: string
  /** 记录生成时间（"YYYY-MM-DD HH:mm:ss" 格式字符串） */
  timestamp: string
  /** 学校 / 机构名称 */
  school: string
  /** 班级 / 实验组别名称 */
  classGroup: string
  /** 学生编号 / 学号 */
  studentId: string
  /** 测试模式：实时反馈(A组) / 延时反馈(B组) */
  type: FeedbackRecordType
  /** 发力综合评分（0-100），可能为空 */
  score: number | null
  /** DeepSeek 大模型给出的动作批注与改进建议（痛点分析 + 教练处方拼接） */
  aiFeedback: string
  /**
   * 【v3.0 新增：科研指挥中心】本条记录命中的生物力学错误分类标签列表
   * （例如 ["支撑脚位置偏离", "膝关节过度屈曲"]），由后端根据本次三级容错
   * 命中统计与综合评分启发式推导，供教练端看板「集体错误热力图」统计
   * 全班高频失误分布使用。可能为空数组（本次表现完全达标）或字段缺失
   * （历史旧数据，未回填该字段），前端消费时必须做空值兜底。
   */
  biomechanicalErrors?: string[]
  /** 后端 OpenCV 矢量标注过的击球关键帧截图（Base64 data URI），可能为空 */
  impactFrameBase64?: string | null
  /** 生成的 Word 报告文件绝对物理路径 */
  path?: string | null
  /** Word 报告所在的文件夹绝对物理路径（供"打开电脑文件夹"按钮使用） */
  directory?: string | null
  /**
   * 【v4.0 科研级数据矩阵新增】以下字段与后端 academic_exporter.py 导出的
   * 学术统计矩阵长表列严格对齐，供教练端「时空胶囊尝试时间轴」「双轴互动
   * 运动学成长期刊图」直接消费。历史旧记录（v4.0 升级前归档）可能缺失这些
   * 字段，前端消费时必须做空值兜底（例如回退到启发式估算或不渲染对应曲线段）。
   */
  /** 测试日期，YYYY-MM-DD 格式字符串，用于顶栏「📅 测试日期」级联筛选器 */
  testDate?: string
  /** 击球瞬间膝关节屈曲角度（度），优先为真实测量均值，缺失历史记录会退化为估算值 */
  kneeFlexionAngle?: number | null
  /** 支撑脚离球距离（cm），当前版本为启发式估算值（尚未接入真实多点位测量） */
  supportFootDistance?: number | null
  /** 实验对照组别编码：1 = 实时反馈 A 组，2 = 延时反馈 B 组 */
  groupTypeCode?: 1 | 2
  /** 主要错误分类编码：0=合规，1=支撑脚偏离，2=膝角不足，3=重心后坐 */
  primaryErrorCode?: 0 | 1 | 2 | 3
}

/* ------------------------------------------------------------------ */
/* 以下为「教练端科研指挥中心」v3.0 新增类型定义                          */
/* ------------------------------------------------------------------ */

/** 集体错误热力图单条统计项：某项生物力学错误分类 + 出现次数/百分比 */
export interface BiomechFaultStat {
  label: string
  count: number
  percentage: number
}

/** ✨ 全班 AIGC 教学处方（DeepSeek 生成的集体诊断简报） */
export interface ClassPrescriptionReport {
  diagnosis: string
  prescription: string
  fullText: string
  generatedAt: string
}

/** 个体纵向进化画像：AI 优缺点总结 */
export interface IndividualSummaryReport {
  strengths: string
  weaknesses: string
  generatedAt: string
}

/** 教练端看板视角切换：全班集体宏观诊断 / 个体纵向进化追踪 */
export type CoachDashboardPerspective = 'classOverview' | 'individual'

/* ------------------------------------------------------------------ */
/* 以下为「v4.0 科研级数据矩阵大升级」新增类型定义                        */
/* ------------------------------------------------------------------ */

/** POST /api/export_academic_matrix 的响应结构 */
export interface AcademicExportResult {
  success: boolean
  message?: string
  path?: string
  filename?: string
  rowCount?: number
  studentCount?: number
}
