# -*- coding: utf-8 -*-
"""Pydantic Schema：API 入参/出参与 ORM 解耦。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models.enums import (
    DEFAULT_MIN_SHOTS_PER_SESSION,
    ExperimentalGroup,
    StudyTimepoint,
)


class StudentProfileCreate(BaseModel):
    """注册匿名被试（禁止携带真实姓名/学号）。"""

    model_config = ConfigDict(extra="forbid")

    anonymous_id: str = Field(..., min_length=1, max_length=64, examples=["Sub_001"])
    cluster_id: str = Field(..., min_length=1, max_length=64, examples=["Class_1"])
    experimental_group: ExperimentalGroup

    @field_validator("anonymous_id", "cluster_id")
    @classmethod
    def _strip_ids(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("字段不可为空")
        return text


class StudentProfileRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    anonymous_id: str
    cluster_id: str
    experimental_group: ExperimentalGroup
    enrolled_at: datetime
    is_active: bool
    stores_identifying_pii: bool = False


class EthicsIdentityMappingCreate(BaseModel):
    """仅伦理管理员路径使用；不得进入科研分析 API。"""

    model_config = ConfigDict(extra="forbid")

    anonymous_id: str = Field(..., min_length=1, max_length=64)
    real_full_name: Optional[str] = Field(None, max_length=128)
    real_student_number: Optional[str] = Field(None, max_length=64)
    consent_version: Optional[str] = Field(None, max_length=32)
    notes: Optional[str] = Field(None, max_length=512)


class EthicsIdentityMappingRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    anonymous_id: str
    real_full_name: Optional[str] = None
    real_student_number: Optional[str] = None
    consent_version: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime


class ShotAttemptLogCreate(BaseModel):
    """单次射门扁平写入；8 大量纲均为标量。"""

    model_config = ConfigDict(extra="forbid")

    anonymous_id: str = Field(..., min_length=1, max_length=64)
    session_date: date
    impact_frame_index: int = Field(..., ge=0)
    timepoint_session_id: Optional[int] = None

    distance_cm: Optional[float] = None
    toe_angle: Optional[float] = None
    max_folding_angle: Optional[float] = None
    whipping_velocity: Optional[float] = None
    impact_knee_angle: Optional[float] = None
    ankle_rigidity: Optional[float] = None
    support_knee_angle: Optional[float] = None
    hip_torsion_angle: Optional[float] = None
    total_score: Optional[float] = Field(None, ge=0, le=100)


class ShotAttemptLogRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    anonymous_id: str
    session_date: date
    impact_frame_index: int
    timepoint_session_id: Optional[int] = None
    distance_cm: Optional[float] = None
    toe_angle: Optional[float] = None
    max_folding_angle: Optional[float] = None
    whipping_velocity: Optional[float] = None
    impact_knee_angle: Optional[float] = None
    ankle_rigidity: Optional[float] = None
    support_knee_angle: Optional[float] = None
    hip_torsion_angle: Optional[float] = None
    total_score: Optional[float] = None
    recorded_at: datetime


class TimepointSessionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str = Field(..., min_length=1, max_length=64)
    timepoint: StudyTimepoint
    session_date: date
    experimental_group: Optional[ExperimentalGroup] = None
    planned_min_shots: int = Field(DEFAULT_MIN_SHOTS_PER_SESSION, ge=1)
    notes: Optional[str] = Field(None, max_length=512)


class TimepointSessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    cluster_id: str
    timepoint: StudyTimepoint
    session_date: date
    experimental_group: Optional[ExperimentalGroup] = None
    planned_min_shots: int
    notes: Optional[str] = None
    created_at: datetime


class SubjectDoseStatusSchema(BaseModel):
    anonymous_id: str
    shot_count: int
    meets_threshold: bool


class DoseComplianceReportSchema(BaseModel):
    cluster_id: str
    timepoint: StudyTimepoint
    session_date: date
    min_shots: int
    subjects: list[SubjectDoseStatusSchema]
    missing_anonymous_ids: list[str] = []
    all_meet_threshold: bool
    compliant_count: int
    noncompliant_anonymous_ids: list[str] = []
