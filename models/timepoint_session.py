# -*- coding: utf-8 -*-
"""干预课时间节点与练习剂量达标校验。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Date,
    DateTime,
    Enum,
    Integer,
    String,
    UniqueConstraint,
    func,
    select,
)
from sqlalchemy.orm import Mapped, Session, mapped_column, validates

from models.base import Base
from models.enums import (
    DEFAULT_MIN_SHOTS_PER_SESSION,
    ExperimentalGroup,
    StudyTimepoint,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SubjectDoseStatus:
    """单个被试在某次干预课上的剂量状态。"""

    anonymous_id: str
    shot_count: int
    meets_threshold: bool


@dataclass(frozen=True)
class DoseComplianceReport:
    """集群在指定时间节点下的练习剂量合规报告。"""

    cluster_id: str
    timepoint: StudyTimepoint
    session_date: date
    min_shots: int
    subjects: tuple[SubjectDoseStatus, ...]
    missing_anonymous_ids: tuple[str, ...] = ()

    @property
    def all_meet_threshold(self) -> bool:
        if self.missing_anonymous_ids:
            return False
        if not self.subjects:
            return False
        return all(item.meets_threshold for item in self.subjects)

    @property
    def compliant_count(self) -> int:
        return sum(1 for item in self.subjects if item.meets_threshold)

    @property
    def noncompliant_anonymous_ids(self) -> tuple[str, ...]:
        return tuple(
            item.anonymous_id for item in self.subjects if not item.meets_threshold
        )


class TimepointSession(Base):
    """16 周追踪实验中的一次干预/测评课程记录。

    一行 = 某集群在某日的某个标准时间节点（T0–T4）上课。
    """

    __tablename__ = "timepoint_sessions"
    __table_args__ = (
        UniqueConstraint(
            "cluster_id",
            "timepoint",
            "session_date",
            name="uq_timepoint_cluster_tp_date",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    cluster_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    timepoint: Mapped[StudyTimepoint] = mapped_column(
        Enum(
            StudyTimepoint,
            name="study_timepoint_enum",
            native_enum=False,
            length=8,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        index=True,
    )
    session_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    experimental_group: Mapped[Optional[ExperimentalGroup]] = mapped_column(
        Enum(
            ExperimentalGroup,
            name="timepoint_experimental_group_enum",
            native_enum=False,
            length=32,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=True,
        doc="冗余记录当课所属试验臂，便于过滤；以 StudentProfile 为准",
    )
    planned_min_shots: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=DEFAULT_MIN_SHOTS_PER_SESSION,
        doc="本课剂量阈值（默认 15 次射门）",
    )
    notes: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
    )

    @validates("cluster_id")
    def _normalize_cluster(self, _key: str, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("cluster_id 不可为空")
        return text

    @validates("timepoint")
    def _coerce_timepoint(self, _key: str, value: Any) -> StudyTimepoint:
        if isinstance(value, StudyTimepoint):
            return value
        return StudyTimepoint(str(value))

    @validates("planned_min_shots")
    def _validate_min_shots(self, _key: str, value: Any) -> int:
        n = int(value)
        if n < 1:
            raise ValueError("planned_min_shots 必须 >= 1")
        return n

    def check_cluster_dose_compliance(
        self,
        db: Session,
        *,
        min_shots: int | None = None,
        only_active_subjects: bool = True,
    ) -> DoseComplianceReport:
        """校验本课所属集群内每位被试的射门总剂量是否达标。

        统计口径：``ShotAttemptLog`` 中
        ``timepoint_session_id == self.id`` 的记录数；
        若尚无挂接 session_id，则回退为同日同集群被试的
        ``session_date == self.session_date`` 计数。

        Args:
            db: SQLAlchemy Session
            min_shots: 覆盖阈值；默认使用 ``self.planned_min_shots``
            only_active_subjects: 仅统计 ``StudentProfile.is_active`` 被试

        Returns:
            DoseComplianceReport
        """
        from models.shot_attempt_log import ShotAttemptLog
        from models.student_profile import StudentProfile

        threshold = int(min_shots if min_shots is not None else self.planned_min_shots)

        profile_stmt = select(StudentProfile).where(
            StudentProfile.cluster_id == self.cluster_id
        )
        if only_active_subjects:
            profile_stmt = profile_stmt.where(StudentProfile.is_active.is_(True))
        profiles = list(db.scalars(profile_stmt).all())
        expected_ids = [p.anonymous_id for p in profiles]

        # 优先按 timepoint_session_id 精确归属
        linked_counts = dict(
            db.execute(
                select(
                    ShotAttemptLog.anonymous_id,
                    func.count(ShotAttemptLog.id),
                )
                .where(ShotAttemptLog.timepoint_session_id == self.id)
                .group_by(ShotAttemptLog.anonymous_id)
            ).all()
        )

        # 回退：同日 + 本集群被试（未挂接 session 的历史数据）
        fallback_counts: dict[str, int] = {}
        if expected_ids:
            fallback_counts = dict(
                db.execute(
                    select(
                        ShotAttemptLog.anonymous_id,
                        func.count(ShotAttemptLog.id),
                    )
                    .where(
                        ShotAttemptLog.session_date == self.session_date,
                        ShotAttemptLog.anonymous_id.in_(expected_ids),
                    )
                    .group_by(ShotAttemptLog.anonymous_id)
                ).all()
            )

        statuses: list[SubjectDoseStatus] = []
        for anonymous_id in expected_ids:
            count = int(
                linked_counts.get(anonymous_id, fallback_counts.get(anonymous_id, 0))
            )
            statuses.append(
                SubjectDoseStatus(
                    anonymous_id=anonymous_id,
                    shot_count=count,
                    meets_threshold=count >= threshold,
                )
            )

        # 有射门记录但不在当前活跃档案中的匿名 ID（数据完整性提示）
        observed = set(linked_counts) | set(fallback_counts)
        missing_from_roster = tuple(sorted(observed - set(expected_ids)))

        return DoseComplianceReport(
            cluster_id=self.cluster_id,
            timepoint=self.timepoint,
            session_date=self.session_date,
            min_shots=threshold,
            subjects=tuple(statuses),
            missing_anonymous_ids=missing_from_roster,
        )

    @classmethod
    def verify_dose_for_cluster_timepoint(
        cls,
        db: Session,
        *,
        cluster_id: str,
        timepoint: StudyTimepoint | str,
        session_date: date | None = None,
        min_shots: int = DEFAULT_MIN_SHOTS_PER_SESSION,
    ) -> DoseComplianceReport:
        """按集群 + 时间节点快速定位课程行并校验剂量。

        若同一 (cluster, timepoint) 有多行，优先匹配 ``session_date``；
        未指定日期时取最新一行。
        """
        tp = (
            timepoint
            if isinstance(timepoint, StudyTimepoint)
            else StudyTimepoint(str(timepoint))
        )
        stmt = select(cls).where(
            cls.cluster_id == cluster_id,
            cls.timepoint == tp,
        )
        if session_date is not None:
            stmt = stmt.where(cls.session_date == session_date)
        else:
            stmt = stmt.order_by(cls.session_date.desc())

        row = db.scalars(stmt).first()
        if row is None:
            raise LookupError(
                f"未找到 TimepointSession: cluster={cluster_id!r} "
                f"timepoint={tp.value} date={session_date}"
            )
        return row.check_cluster_dose_compliance(db, min_shots=min_shots)

    def __repr__(self) -> str:
        return (
            f"<TimepointSession {self.cluster_id} "
            f"{self.timepoint.value} {self.session_date}>"
        )
