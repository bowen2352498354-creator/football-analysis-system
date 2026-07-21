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
from typing import Optional

from pydantic import BaseModel, Field, field_validator


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

    anonymous_id: str
    session_date: date
    timepoint: Timepoint
    impact_frame_index: int = Field(..., ge=0)
    cluster_id: str = ""
    experimental_group: Optional[ExperimentalGroup] = None

    # 8 大生物力学核心测量值（扁平数值字段）
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

    session_id: str
    cluster_id: str
    timepoint: Timepoint
    session_date: date
    planned_dose: int = Field(default=STANDARD_SHOT_DOSE, ge=1)
    notes: str = ""

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
    anonymous_id: str
    cluster_id: str
    experimental_group: Optional[str] = None
    score_t1: float
    score_t2: float
    slope: float
    mean_level: float
    responder_type: str  # "high_responder" | "low_responder"
