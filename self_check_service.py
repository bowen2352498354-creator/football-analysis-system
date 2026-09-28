"""Deterministic self-check prescription for one archived shot attempt."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.engine import Engine

from db import init_db, session_scope
from models.self_check_task import SelfCheckCheckin, SelfCheckTask


TEMPLATE_VERSION = "1.0"
_task_lock = threading.Lock()

_DIMENSION_ALIASES: dict[str, tuple[str, ...]] = {
    "approach_rhythm": ("approach_rhythm", "approach_score"),
    "support_stability": ("support_stability", "support_score"),
    "backswing_folding": ("backswing_folding", "backswing_score"),
    "ankle_rigidity": ("ankle_rigidity", "ankle_rigidity_score"),
    "whipping_velocity": ("whipping_velocity", "whipping_score"),
}

_METRIC_TO_DIMENSION = {
    "distance_cm": "support_stability",
    "support_lateral_dist_cm": "support_stability",
    "support_ap_offset_cm": "support_stability",
    "max_folding_angle": "backswing_folding",
    "impact_knee_angle": "backswing_folding",
    "ankle_rigidity": "ankle_rigidity",
    "toe_angle": "ankle_rigidity",
    "whipping_velocity": "whipping_velocity",
    "hip_torsion_angle": "whipping_velocity",
}

_TEMPLATES: dict[str, dict[str, Any]] = {
    "approach_rhythm": {
        "illustrationKey": "approach_steps",
        "title": "最后三步节奏自查",
        "instruction": "小—大—稳：最后一步缩短，支撑脚轻快落稳。",
        "successCriterion": "连续3次助跑节奏一致，支撑脚落地后身体不晃动。",
        "dosage": "每组5次，共3组；组间休息30秒",
    },
    "support_stability": {
        "illustrationKey": "support_target",
        "title": "支撑脚落点自查",
        "instruction": "脚落球侧，膝盖微屈，先踩稳再摆腿。",
        "successCriterion": "连续3次支撑脚进入目标区，落地后躯干保持稳定。",
        "dosage": "无球定位5次＋带球击球5次，共3组",
    },
    "backswing_folding": {
        "illustrationKey": "leg_fold",
        "title": "摆动腿折叠自查",
        "instruction": "脚跟靠近臀部，膝盖先领，随后快速打开小腿。",
        "successCriterion": "连续3次后摆时小腿完成清晰折叠，击球前不提前伸膝。",
        "dosage": "慢动作摆腿8次＋正常击球5次，共3组",
    },
    "ankle_rigidity": {
        "illustrationKey": "ankle_lock",
        "title": "脚尖下压锁踝自查",
        "instruction": "脚尖下压，脚背绷直，触球前后保持脚踝形状。",
        "successCriterion": "连续3次触球时脚尖不翘起，脚背正面稳定接触足球。",
        "dosage": "靠墙锁踝保持10秒×3次，再击球5次",
    },
    "whipping_velocity": {
        "illustrationKey": "follow_through",
        "title": "向前送髋随摆自查",
        "instruction": "踢穿足球，不急刹腿，让膝盖和脚尖指向目标。",
        "successCriterion": "连续3次击球后摆动腿自然越过支撑腿，身体继续向前。",
        "dosage": "半程助跑击球5次，共3组",
    },
}


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _radar_dict(record: dict[str, Any]) -> dict[str, Any]:
    for key in ("quantified5dScores", "radar_scores", "radarScores"):
        value = record.get(key)
        if isinstance(value, dict):
            return value
    detail = record.get("scoreDetail")
    if isinstance(detail, dict) and isinstance(detail.get("radar_scores"), dict):
        return detail["radar_scores"]
    return {}


def choose_dimension(record: dict[str, Any]) -> str:
    priority = record.get("priorityTarget")
    if isinstance(priority, dict):
        metric_key = str(priority.get("metricKey") or priority.get("metric_key") or "")
        if metric_key in _METRIC_TO_DIMENSION:
            return _METRIC_TO_DIMENSION[metric_key]

    radar = _radar_dict(record)
    scores: list[tuple[float, str]] = []
    for dimension, aliases in _DIMENSION_ALIASES.items():
        for alias in aliases:
            number = _finite_number(radar.get(alias))
            if number is not None:
                scores.append((number, dimension))
                break
    if scores:
        return min(scores, key=lambda item: item[0])[1]

    error_text = " ".join(str(item) for item in (record.get("biomechanicalErrors") or []))
    if "支撑" in error_text or "重心" in error_text:
        return "support_stability"
    if "膝" in error_text or "折叠" in error_text:
        return "backswing_folding"
    if "踝" in error_text or "脚尖" in error_text:
        return "ankle_rigidity"
    if "随摆" in error_text or "转髋" in error_text:
        return "whipping_velocity"
    return "approach_rhythm"


def build_task_preview(record: dict[str, Any]) -> dict[str, Any]:
    """Build the deterministic task payload without touching persistence."""

    dimension = choose_dimension(record)
    template = _TEMPLATES[dimension]
    return {
        "taskId": None,
        "sourceRecordId": str(record.get("id") or ""),
        "dimensionKey": dimension,
        "illustrationKey": str(template["illustrationKey"]),
        "title": str(template["title"]),
        "instruction": str(template["instruction"]),
        "successCriterion": str(template["successCriterion"]),
        "dosage": str(template["dosage"]),
        "targetCount": 3,
        "templateVersion": TEMPLATE_VERSION,
        "coachVerified": False,
        "checkins": [
            {"slotNo": slot, "checked": False, "checkedAt": None, "note": None}
            for slot in range(1, 4)
        ],
        "createdAt": None,
        "updatedAt": None,
        "persistenceAvailable": False,
    }


def _serialize_datetime(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if isinstance(value, datetime) else None


def serialize_task(task: SelfCheckTask) -> dict[str, Any]:
    existing = {row.slot_no: row for row in task.checkins}
    checkins = []
    for slot_no in range(1, task.target_count + 1):
        row = existing.get(slot_no)
        checkins.append(
            {
                "slotNo": slot_no,
                "checked": bool(row.checked) if row else False,
                "checkedAt": _serialize_datetime(row.checked_at) if row else None,
                "note": row.note if row else None,
            }
        )
    return {
        "taskId": task.id,
        "sourceRecordId": task.source_record_id,
        "dimensionKey": task.dimension_key,
        "illustrationKey": task.illustration_key,
        "title": task.title,
        "instruction": task.instruction,
        "successCriterion": task.success_criterion,
        "dosage": task.dosage,
        "targetCount": task.target_count,
        "templateVersion": task.template_version,
        "coachVerified": bool(task.coach_verified),
        "checkins": checkins,
        "createdAt": _serialize_datetime(task.created_at),
        "updatedAt": _serialize_datetime(task.updated_at),
    }


def ensure_self_check_task(
    record: dict[str, Any],
    *,
    bind: Optional[Engine] = None,
) -> dict[str, Any]:
    record_id = str(record.get("id") or "").strip()
    if not record_id:
        raise ValueError("source record id is required")
    init_db(bind)
    with _task_lock, session_scope(bind) as session:
        task = session.scalar(
            select(SelfCheckTask).where(SelfCheckTask.source_record_id == record_id)
        )
        if task is None:
            preview = build_task_preview(record)
            task = SelfCheckTask(
                id=str(uuid.uuid4()),
                source_record_id=record_id,
                dimension_key=str(preview["dimensionKey"]),
                illustration_key=str(preview["illustrationKey"]),
                title=str(preview["title"]),
                instruction=str(preview["instruction"]),
                success_criterion=str(preview["successCriterion"]),
                dosage=str(preview["dosage"]),
                target_count=3,
                template_version=TEMPLATE_VERSION,
            )
            session.add(task)
            session.flush()
        # Materialize relationship before the transaction closes.
        _ = list(task.checkins)
        return serialize_task(task)


def update_self_check_task(
    record: dict[str, Any],
    *,
    slot_no: Optional[int] = None,
    checked: Optional[bool] = None,
    coach_verified: Optional[bool] = None,
    note: Optional[str] = None,
    bind: Optional[Engine] = None,
) -> dict[str, Any]:
    ensured = ensure_self_check_task(record, bind=bind)
    task_id = str(ensured["taskId"])
    now = datetime.now(timezone.utc)

    with _task_lock, session_scope(bind) as session:
        task = session.get(SelfCheckTask, task_id)
        if task is None:
            raise LookupError("self-check task not found")
        if coach_verified is not None:
            task.coach_verified = bool(coach_verified)
        if slot_no is not None:
            if slot_no < 1 or slot_no > task.target_count:
                raise ValueError("slotNo is outside the task target range")
            row = session.scalar(
                select(SelfCheckCheckin).where(
                    SelfCheckCheckin.task_id == task.id,
                    SelfCheckCheckin.slot_no == slot_no,
                )
            )
            if row is None:
                row = SelfCheckCheckin(task_id=task.id, slot_no=slot_no)
                session.add(row)
            if checked is not None:
                row.checked = bool(checked)
                row.checked_at = now if checked else None
            if note is not None:
                row.note = str(note).strip()[:500] or None
        task.updated_at = now
        session.flush()
        _ = list(task.checkins)
        return serialize_task(task)
