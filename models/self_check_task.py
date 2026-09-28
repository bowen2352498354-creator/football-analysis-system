# -*- coding: utf-8 -*-
"""Per-attempt visual self-check tasks and persistent completion slots."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.base import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SelfCheckTask(Base):
    __tablename__ = "self_check_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_record_id: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, index=True
    )
    dimension_key: Mapped[str] = mapped_column(String(64), nullable=False)
    illustration_key: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(128), nullable=False)
    instruction: Mapped[str] = mapped_column(String(256), nullable=False)
    success_criterion: Mapped[str] = mapped_column(String(256), nullable=False)
    dosage: Mapped[str] = mapped_column(String(128), nullable=False)
    target_count: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    template_version: Mapped[str] = mapped_column(String(32), nullable=False, default="1.0")
    coach_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utc_now, onupdate=_utc_now
    )

    checkins: Mapped[list["SelfCheckCheckin"]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="SelfCheckCheckin.slot_no",
    )


class SelfCheckCheckin(Base):
    __tablename__ = "self_check_checkins"
    __table_args__ = (
        UniqueConstraint("task_id", "slot_no", name="uq_self_check_task_slot"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(
        ForeignKey("self_check_tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    slot_no: Mapped[int] = mapped_column(Integer, nullable=False)
    checked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    checked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    note: Mapped[Optional[str]] = mapped_column(String(500))

    task: Mapped[SelfCheckTask] = relationship(back_populates="checkins")

