# -*- coding: utf-8 -*-
"""被试档案与伦理身份映射。

``StudentProfile`` 仅承载 Cluster-RCT 隔离所需的匿名字段；真实姓名/学号
严禁写入本表。必要时由 ``EthicsIdentityMapping`` 独立维护双向映射，
该表不得出现在学术导出、看板聚合或默认 ORM relationship 联表中。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import DateTime, Enum, String, UniqueConstraint, event, inspect as sa_inspect
from sqlalchemy.orm import Mapped, mapped_column, validates

from models.base import Base
from models.enums import ExperimentalGroup

# StudentProfile 上视为「随机化后不可变」的列名
_IMMUTABLE_PROFILE_FIELDS = frozenset(
    {"anonymous_id", "cluster_id", "experimental_group"}
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StudentProfile(Base):
    """匿名被试档案：组别隔离的最小不可变单元。

    设计约束：
    - 不含 real_name / student_number / 任何可直接识别字段；
    - ``anonymous_id`` / ``cluster_id`` / ``experimental_group`` 在持久化后只读；
    - 整群随机化语义上 ``experimental_group`` 应与同 ``cluster_id`` 一致
      （由注册流程保证，本模型不跨行强制）。
    """

    __tablename__ = "student_profiles"
    __table_args__ = (
        UniqueConstraint("anonymous_id", name="uq_student_anonymous_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    anonymous_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="唯一匿名编号，如 Sub_001",
    )
    cluster_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="行政班集群 ID，如 Class_1（整群随机化单位）",
    )
    experimental_group: Mapped[ExperimentalGroup] = mapped_column(
        Enum(
            ExperimentalGroup,
            name="experimental_group_enum",
            native_enum=False,
            length=32,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        index=True,
        doc="A 实时 / B 延时 / C 对照",
    )

    enrolled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
    )
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)

    @property
    def stores_identifying_pii(self) -> bool:
        """伦理合规探针：主档案永不持有可识别 PII。"""
        return False

    @validates("anonymous_id", "cluster_id")
    def _normalize_ids(self, key: str, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"{key} 不可为空")
        return text

    @validates("experimental_group")
    def _coerce_group(self, _key: str, value: Any) -> ExperimentalGroup:
        if isinstance(value, ExperimentalGroup):
            return value
        return ExperimentalGroup(str(value))

    def __repr__(self) -> str:
        return (
            f"<StudentProfile anonymous_id={self.anonymous_id!r} "
            f"cluster_id={self.cluster_id!r} "
            f"group={self.experimental_group.value}>"
        )


@event.listens_for(StudentProfile, "before_update")
def _reject_immutable_field_mutation(mapper, connection, target) -> None:  # noqa: ARG001
    """拦截对随机化键的 UPDATE，保证组别隔离字段不可变。"""
    state = sa_inspect(target)
    for field in _IMMUTABLE_PROFILE_FIELDS:
        hist = state.attrs[field].history
        if hist.has_changes():
            raise ValueError(
                f"StudentProfile.{field} 为 Cluster-RCT 不可变字段，禁止修改 "
                f"(attempted change on {target.anonymous_id!r})"
            )


class EthicsIdentityMapping(Base):
    """真实身份 ↔ 匿名编号映射（独立伦理表）。

    仅在知情同意、紧急联系或撤回研究等合规场景按需访问。
    科研分析管线、SPSS 导出、教练看板默认路径不得 JOIN 本表。
    """

    __tablename__ = "ethics_identity_mappings"
    __table_args__ = (
        UniqueConstraint("anonymous_id", name="uq_ethics_anonymous_id"),
        UniqueConstraint("real_student_number", name="uq_ethics_real_student_number"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    anonymous_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="对应 StudentProfile.anonymous_id",
    )
    # 以下字段仅存在于本表；绝不复制到 StudentProfile
    real_full_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    real_student_number: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    consent_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
    )

    def __repr__(self) -> str:
        return f"<EthicsIdentityMapping anonymous_id={self.anonymous_id!r}>"
