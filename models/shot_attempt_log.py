# -*- coding: utf-8 -*-
"""射门行为日志：扁平化 8 大生物力学量纲。"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Mapping, Optional

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, validates

from models.base import Base
from models.enums import BIOMECH_CORE_METRIC_KEYS


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ShotAttemptLog(Base):
    """单次射门尝试的科研级落盘记录。

    绑定键：``anonymous_id`` + ``session_date`` + ``impact_frame_index``。
    8 大量纲一律扁平 Float 列，禁止深层 JSON blob，便于按集群/时点做
    SQL 聚合（均值、方差、达标率）。
    """

    __tablename__ = "shot_attempt_logs"
    __table_args__ = (
        UniqueConstraint(
            "anonymous_id",
            "session_date",
            "impact_frame_index",
            name="uq_shot_anon_date_impact",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    anonymous_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("student_profiles.anonymous_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    session_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    impact_frame_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="抛物线锁帧得到的触球绝对零点帧索引 t_impact",
    )

    # ---- 可选：挂接到干预课时间节点 ----
    timepoint_session_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("timepoint_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # ---- 8 大生物力学核心测量值（扁平数值列）----
    distance_cm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, doc="支撑脚距球心横距 (cm)"
    )
    toe_angle: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, doc="支撑脚尖朝向角 (°)"
    )
    max_folding_angle: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, doc="摆动腿最大折叠角 (°)"
    )
    whipping_velocity: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, doc="鞭打峰值角速度 (°/s)"
    )
    impact_knee_angle: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, doc="触球瞬间膝关节角 (°)"
    )
    ankle_rigidity: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, doc="踝关节跖屈刚度方差"
    )
    support_knee_angle: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, doc="支撑腿膝关节角 (°)"
    )
    hip_torsion_angle: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, doc="髋部相对扭转角 (°)"
    )

    total_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, doc="DeterministicScorer 总分 (0-100)"
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
    )

    @validates("anonymous_id")
    def _normalize_anonymous_id(self, _key: str, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("anonymous_id 不可为空")
        return text

    @validates("impact_frame_index")
    def _validate_impact_frame(self, _key: str, value: Any) -> int:
        idx = int(value)
        if idx < 0:
            raise ValueError("impact_frame_index 不可为负")
        return idx

    @classmethod
    def from_score_detail(
        cls,
        *,
        anonymous_id: str,
        session_date: date,
        impact_frame_index: int,
        score_detail: Mapping[str, Any] | None = None,
        timepoint_session_id: int | None = None,
        total_score: float | None = None,
    ) -> "ShotAttemptLog":
        """从 DeterministicScorer ``scoreDetail.indicators`` 扁平化为 ORM 行。"""
        indicators = {}
        if isinstance(score_detail, Mapping):
            raw = score_detail.get("indicators")
            if isinstance(raw, Mapping):
                indicators = raw
            if total_score is None and score_detail.get("TotalScore") is not None:
                total_score = float(score_detail["TotalScore"])

        flat: dict[str, float | None] = {}
        for key in BIOMECH_CORE_METRIC_KEYS:
            entry = indicators.get(key)
            if isinstance(entry, Mapping) and entry.get("value") is not None:
                flat[key] = float(entry["value"])
            elif isinstance(entry, (int, float)):
                flat[key] = float(entry)
            else:
                flat[key] = None

        return cls(
            anonymous_id=anonymous_id,
            session_date=session_date,
            impact_frame_index=int(impact_frame_index),
            timepoint_session_id=timepoint_session_id,
            total_score=total_score,
            **flat,
        )

    def as_metric_dict(self) -> dict[str, float | None]:
        """返回仅含 8 大量纲的扁平字典，便于聚合/导出。"""
        return {key: getattr(self, key) for key in BIOMECH_CORE_METRIC_KEYS}

    def __repr__(self) -> str:
        return (
            f"<ShotAttemptLog {self.anonymous_id} "
            f"{self.session_date} frame={self.impact_frame_index}>"
        )
