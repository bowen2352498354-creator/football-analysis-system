# -*- coding: utf-8 -*-
"""
research_models.py
V2.5 Cluster-RCT 科研管理后台 —— 数据模型层（Pydantic Schema）

与「被试组别隔离 / 射门行为日志 / 时间节点干预进度」三张核心表对齐，
供 ResearchDashboardService、AcademicDataExporter 等下游模块统一消费。

伦理合规：业务表仅使用 anonymous_id，不落真实姓名或学号；
身份映射如需存在，必须独立于本模块的存储文件。
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# --------------------------------------------------------------------------
# 常量：16 周追踪实验时间节点 + 单课标准练习剂量
# --------------------------------------------------------------------------

class Timepoint(str, Enum):
    """16 周追踪实验中的固定观测/干预节点。"""

    T0 = "T0"
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"


class ExperimentalGroup(str, Enum):
    """整群随机对照试验三臂组别（不可变枚举）。"""

    GROUP_A_REALTIME = "GROUP_A_REALTIME"
    GROUP_B_DELAYED = "GROUP_B_DELAYED"
    GROUP_C_CONTROL = "GROUP_C_CONTROL"


# 单次干预课标准射门剂量（次）；剂量异常判定带宽为 ±20%
STANDARD_SHOT_DOSE: int = 15
DOSE_TOLERANCE_RATIO: float = 0.20

# DeterministicScorer 对齐的 8 大生物力学扁平字段（便于聚合 / SPSS 宽表展开）
BIOMECH_METRIC_FIELDS: tuple[str, ...] = (
    "distance_cm",
    "toe_angle",
    "max_folding_angle",
    "whipping_velocity",
    "impact_knee_angle",
    "ankle_rigidity",
    "support_knee_angle",
    "hip_torsion_angle",
)


# --------------------------------------------------------------------------
# 1. StudentProfile —— 被试档案（组别严格隔离）
# --------------------------------------------------------------------------


class StudentProfile(BaseModel):
    """受试者匿名档案。真实姓名/学号不得写入本模型。"""

    model_config = ConfigDict(extra="ignore")

    anonymous_id: str = Field(..., description="唯一匿名编号，例如 Sub_001")
    cluster_id: str = Field(..., description="行政班集群，例如 Class_1")
    experimental_group: ExperimentalGroup

    @field_validator("anonymous_id", "cluster_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("anonymous_id / cluster_id 不可为空")
        return text


# --------------------------------------------------------------------------
# 2. ShotAttemptLog —— 单次射门行为日志（扁平 8 大量纲）
# --------------------------------------------------------------------------


class ShotAttemptLog(BaseModel):
    """一次有效射门的结构化落盘记录。"""

    model_config = ConfigDict(extra="ignore")

    anonymous_id: str
    session_date: date
    timepoint: Timepoint
    # 帧索引理应为 int；JS 可能传 60.0 —— before 校验取整，避免 int_from_float
    impact_frame_index: int = Field(..., ge=0)
    cluster_id: str = ""
    experimental_group: Optional[ExperimentalGroup] = None

    # 8 大生物力学核心测量值（扁平数值字段）——全部 float，禁止 int
    distance_cm: Optional[float] = None
    toe_angle: Optional[float] = None
    max_folding_angle: Optional[float] = None
    whipping_velocity: Optional[float] = None
    impact_knee_angle: Optional[float] = None
    ankle_rigidity: Optional[float] = None
    support_knee_angle: Optional[float] = None
    hip_torsion_angle: Optional[float] = None

    # 综合得分（DeterministicScorer TotalScore，0–100）
    composite_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)

    # 前端归档可能附带的雷达小分 / 临时状态（落盘桥接时忽略未知键）
    radar_scores: Optional[Dict[str, float]] = Field(default_factory=dict)
    scores: Optional[Dict[str, float]] = Field(default_factory=dict)
    comment: Optional[str] = ""
    task_status: Optional[str] = None
    score_detail: Optional[Dict[str, Any]] = Field(default_factory=dict)

    # Sprint 5：软删除；True 表示误测脏数据，默认不参与科研统计
    is_deleted: bool = False

    @field_validator("impact_frame_index", mode="before")
    @classmethod
    def _coerce_impact_frame_index(cls, value):
        if value is None or isinstance(value, bool):
            raise ValueError("impact_frame_index 必须为非负整数")
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            value = value.item()
        return int(round(float(value)))

    @field_validator(
        "distance_cm",
        "toe_angle",
        "max_folding_angle",
        "whipping_velocity",
        "impact_knee_angle",
        "ankle_rigidity",
        "support_knee_angle",
        "hip_torsion_angle",
        "composite_score",
        mode="before",
    )
    @classmethod
    def _coerce_metric_floats(cls, value):
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            raise ValueError("业务指标不能为布尔值")
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            value = value.item()
        return float(value)

    @field_validator("session_date", mode="before")
    @classmethod
    def _parse_session_date(cls, value):
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        text = str(value or "").strip()
        if len(text) >= 10 and text[4] == "-" and text[7] == "-":
            return date.fromisoformat(text[:10])
        return date.today()

    def metric_values(self) -> dict[str, Optional[float]]:
        return {key: getattr(self, key) for key in BIOMECH_METRIC_FIELDS}


# --------------------------------------------------------------------------
# 3. TimepointSession —— 干预课次节点标记
# --------------------------------------------------------------------------


class TimepointSession(BaseModel):
    """一次干预/测试课在 16 周时间轴上的节点登记。"""

    model_config = ConfigDict(extra="ignore")

    session_id: str
    cluster_id: str
    timepoint: Timepoint
    session_date: date
    planned_dose: int = Field(default=STANDARD_SHOT_DOSE, ge=1)
    notes: str = ""
    comment: Optional[str] = ""

    @field_validator("planned_dose", mode="before")
    @classmethod
    def _coerce_planned_dose(cls, value):
        if value is None or value == "":
            return STANDARD_SHOT_DOSE
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            value = value.item()
        return int(round(float(value)))

    @field_validator("session_date", mode="before")
    @classmethod
    def _parse_session_date(cls, value):
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        text = str(value or "").strip()
        if len(text) >= 10 and text[4] == "-" and text[7] == "-":
            return date.fromisoformat(text[:10])
        return date.today()

    def dose_band(self) -> tuple[float, float]:
        """返回本课「标准剂量 ±20%」合法区间 [low, high]。"""
        low = self.planned_dose * (1.0 - DOSE_TOLERANCE_RATIO)
        high = self.planned_dose * (1.0 + DOSE_TOLERANCE_RATIO)
        return low, high

    def is_dose_compliant(self, shot_count: int) -> bool:
        low, high = self.dose_band()
        return low <= float(shot_count) <= high


# --------------------------------------------------------------------------
# 教练端 API 响应片段（轻量 DTO）
# --------------------------------------------------------------------------


class DoseAnomalySubject(BaseModel):
    model_config = ConfigDict(extra="ignore")

    anonymous_id: str
    cluster_id: str
    timepoint: str
    shot_count: int
    standard_dose: int
    dose_low: float
    dose_high: float
    deviation_ratio: float
    anomaly_type: str  # "under_dose" | "over_dose"


class ExtremeCaseSubject(BaseModel):
    model_config = ConfigDict(extra="ignore")

    anonymous_id: str
    cluster_id: str
    experimental_group: Optional[str] = None
    score_t1: float
    score_t2: float
    slope: float
    mean_level: float
    responder_type: str  # "high_responder" | "low_responder"


# --------------------------------------------------------------------------
# 具身隐喻：问题关节红绿灯高亮（前端 Canvas 叠加）
# --------------------------------------------------------------------------


class JointHighlightColor(str, Enum):
    """关节高亮红绿灯色码（前端 Canvas 直接消费）。"""

    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


class JointHighlight(BaseModel):
    """问题关节的 2D 像素坐标 + 红绿灯色码 + 临床绝对时间戳。

    ``x`` / ``y`` 为图像平面绝对像素（与 pose_tracker.serialize_pose_frame_record
    一致：归一化坐标 × 视频宽高）。``error_timestamp_sec`` 为该错误发生帧在
    原视频中的绝对秒（与 HTML5 ``video.currentTime`` / ``absolute_timestamps`` 对齐），
    前端仅在播放头靠近该时刻时渲染，避免全时段常亮。
    """

    model_config = ConfigDict(extra="ignore")

    joint_name: str = Field(..., description='关节点名，例如 "right_knee"')
    x: float = Field(..., description="2D 像素 X（或 0–1 归一化，见 coordinate_space）")
    y: float = Field(..., description="2D 像素 Y（或 0–1 归一化，见 coordinate_space）")
    color_code: str = Field(
        ...,
        description='红绿灯色码："RED" | "YELLOW" | "GREEN"',
    )
    error_timestamp_sec: float = Field(
        ...,
        ge=0.0,
        description="错误发生帧的临床绝对时间戳（秒），与 absolute_timestamps 同源",
    )
    metric_key: Optional[str] = Field(
        default=None,
        description="触发该高亮的八大量纲键（可选，便于前端联调）",
    )
    coordinate_space: str = Field(
        default="pixel",
        description='"pixel"=绝对像素；"normalized"=MediaPipe 0–1 归一化',
    )

    @field_validator("joint_name", mode="before")
    @classmethod
    def _strip_joint_name(cls, value):
        text = str(value or "").strip()
        if not text:
            raise ValueError("joint_name 不可为空")
        return text

    @field_validator("color_code", mode="before")
    @classmethod
    def _normalize_color_code(cls, value):
        text = str(value or "").strip().upper()
        if "RED" in text:
            return JointHighlightColor.RED.value
        if "YELLOW" in text:
            return JointHighlightColor.YELLOW.value
        if "GREEN" in text:
            return JointHighlightColor.GREEN.value
        raise ValueError("color_code 必须为 RED / YELLOW / GREEN")

    @field_validator("x", "y", "error_timestamp_sec", mode="before")
    @classmethod
    def _coerce_finite_float(cls, value):
        if value is None or value == "" or isinstance(value, bool):
            raise ValueError("数值字段必须为有限浮点数")
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            value = value.item()
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            raise ValueError("数值字段必须为有限浮点数")
        return number


class MetricsData(BaseModel):
    """诊断结果中的生物力学指标载荷片段（含具身隐喻关节高亮）。"""

    model_config = ConfigDict(extra="ignore")

    # 八大扁平量纲（可选；与 ShotAttemptLog / DeterministicScorer 对齐）
    distance_cm: Optional[float] = None
    toe_angle: Optional[float] = None
    max_folding_angle: Optional[float] = None
    whipping_velocity: Optional[float] = None
    impact_knee_angle: Optional[float] = None
    ankle_rigidity: Optional[float] = None
    support_knee_angle: Optional[float] = None
    hip_torsion_angle: Optional[float] = None

    # 具身隐喻：T0 问题关节红绿灯高亮列表
    joint_highlights: List[Dict[str, Any]] = Field(default_factory=list)


class FeedbackReport(BaseModel):
    """单次动作诊断反馈报告（API / 前端 finalReport 对齐的轻量片段）。"""

    model_config = ConfigDict(extra="ignore")

    TotalScore: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    t_impact: Optional[int] = Field(default=None, ge=0)
    score_detail: Optional[Dict[str, Any]] = Field(default_factory=dict)
    metrics: Optional[Dict[str, Any]] = Field(default_factory=dict)
    # 具身隐喻：与 score_detail.joint_highlights 同源，便于顶层直取
    joint_highlights: List[Dict[str, Any]] = Field(default_factory=list)


# --------------------------------------------------------------------------
# 波形图 / Action ROI 时序（相对帧索引 + 原视频绝对秒）
# --------------------------------------------------------------------------


class KineticWaveformSeries(BaseModel):
    """Action ROI（约 0~60 帧）角速度/角度波形载荷。

    ``absolute_timestamps[i] = (action_roi_start + i) / fps``，单位秒，
    与 HTML5 video.currentTime 对齐，消除波形图与视频的时空脱节。
    """

    model_config = ConfigDict(extra="ignore")

    time_series_velocity: List[float] = Field(default_factory=list)
    absolute_timestamps: List[float] = Field(default_factory=list)
    impact_index_in_window: int = 0
    action_roi_start: int = Field(default=0, ge=0)
    fps: float = Field(default=30.0, gt=0.0)
    # 前端可能附带的临时状态 / 雷达分，缺失时忽略
    radar_scores: Optional[Dict[str, float]] = Field(default_factory=dict)
    scores: Optional[Dict[str, float]] = Field(default_factory=dict)
    task_status: Optional[str] = None
    comment: Optional[str] = ""

    @field_validator(
        "time_series_velocity",
        "absolute_timestamps",
        mode="before",
    )
    @classmethod
    def _coerce_float_lists(cls, value):
        if value is None or value == "":
            return []
        if not isinstance(value, list):
            return []
        out: list[float] = []
        for item in value:
            if item is None or item == "" or isinstance(item, bool):
                continue
            try:
                if hasattr(item, "item") and not isinstance(item, (str, bytes)):
                    item = item.item()
                out.append(float(item))
            except (TypeError, ValueError):
                continue
        return out

    @field_validator("impact_index_in_window", "action_roi_start", mode="before")
    @classmethod
    def _coerce_nonneg_int(cls, value):
        if value is None or value == "":
            return 0
        try:
            if hasattr(value, "item") and not isinstance(value, (str, bytes)):
                value = value.item()
            return max(0, int(round(float(value))))
        except (TypeError, ValueError):
            return 0

    @field_validator("fps", mode="before")
    @classmethod
    def _coerce_fps(cls, value):
        if value is None or value == "":
            return 30.0
        try:
            if hasattr(value, "item") and not isinstance(value, (str, bytes)):
                value = value.item()
            fps = float(value)
            return fps if fps > 1e-6 else 30.0
        except (TypeError, ValueError):
            return 30.0

    @field_validator("radar_scores", "scores", mode="before")
    @classmethod
    def _coerce_score_maps(cls, value):
        if value is None or value == "" or not isinstance(value, dict):
            return {}
        out: dict[str, float] = {}
        for key, raw in value.items():
            if raw is None or raw == "" or isinstance(raw, bool):
                continue
            try:
                if hasattr(raw, "item") and not isinstance(raw, (str, bytes, dict, list)):
                    raw = raw.item()
                out[str(key)] = float(raw)
            except (TypeError, ValueError):
                continue
        return out


class GenerateReportWaveformResponse(BaseModel):
    """``/api/generate_report`` 波形相关字段片段（Request/Response 共用校验）。"""

    model_config = ConfigDict(extra="ignore")

    time_series_velocity: Optional[List[float]] = Field(default_factory=list)
    timeSeriesVelocity: Optional[List[float]] = Field(default_factory=list)
    absolute_timestamps: Optional[List[float]] = Field(default_factory=list)
    absoluteTimestamps: Optional[List[float]] = Field(default_factory=list)
    impact_index_in_window: Optional[int] = None
    impactIndexInWindow: Optional[int] = None
    radar_scores: Optional[Dict[str, float]] = Field(default_factory=dict)
    scores: Optional[Dict[str, float]] = Field(default_factory=dict)
    comment: Optional[str] = ""
    task_status: Optional[str] = None

    @field_validator(
        "time_series_velocity",
        "timeSeriesVelocity",
        "absolute_timestamps",
        "absoluteTimestamps",
        mode="before",
    )
    @classmethod
    def _coerce_optional_float_lists(cls, value):
        if value is None or value == "":
            return []
        if not isinstance(value, list):
            return []
        out: list[float] = []
        for item in value:
            if item is None or item == "" or isinstance(item, bool):
                continue
            try:
                if hasattr(item, "item") and not isinstance(item, (str, bytes)):
                    item = item.item()
                out.append(float(item))
            except (TypeError, ValueError):
                continue
        return out

    @field_validator(
        "impact_index_in_window",
        "impactIndexInWindow",
        mode="before",
    )
    @classmethod
    def _coerce_optional_int(cls, value):
        if value is None or value == "":
            return None
        try:
            if hasattr(value, "item") and not isinstance(value, (str, bytes)):
                value = value.item()
            return int(round(float(value)))
        except (TypeError, ValueError):
            return None

    @field_validator("radar_scores", "scores", mode="before")
    @classmethod
    def _coerce_score_maps(cls, value):
        if value is None or value == "" or not isinstance(value, dict):
            return {}
        out: dict[str, float] = {}
        for key, raw in value.items():
            if raw is None or raw == "" or isinstance(raw, bool):
                continue
            try:
                if hasattr(raw, "item") and not isinstance(raw, (str, bytes, dict, list)):
                    raw = raw.item()
                out[str(key)] = float(raw)
            except (TypeError, ValueError):
                continue
        return out
