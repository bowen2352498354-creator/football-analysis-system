# -*- coding: utf-8 -*-
"""Research attempt ledger for the A/B/C, T0-T4 training loop.

The ledger is intentionally metadata-only. It records one row per completed
analysis session, including rejected captures, without persisting video frames
or report images. Formal research exports continue to use the quality-gated
training archive; this file provides dose and invalid-attempt accountability.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from experimental_group_router import (
    GROUP_A,
    GROUP_B,
    GROUP_C,
    normalize_experimental_group,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LEDGER_PATH = os.path.join(SCRIPT_DIR, "experiment_attempt_ledger.json")
DEFAULT_PLANNED_ATTEMPTS = 15
VALID_TIMEPOINTS = ("T0", "T1", "T2", "T3", "T4")
RESEARCH_PROTOCOL_VERSION = os.environ.get(
    "AIFF_RESEARCH_PROTOCOL_VERSION", "THESIS_2026_DRAFT_V1"
)

_ledger_lock = threading.RLock()


def normalize_timepoint(raw: Optional[str], default: str = "T0") -> str:
    """Return a canonical T0-T4 label; unknown input falls back safely."""
    text = str(raw or "").strip().upper()
    if text in VALID_TIMEPOINTS:
        return text
    if text.isdigit() and f"T{text}" in VALID_TIMEPOINTS:
        return f"T{text}"
    return default if default in VALID_TIMEPOINTS else "T0"


def canonical_group(raw: Optional[str], default: str = GROUP_A) -> str:
    """Return the long research enum used by archives and SPSS exports."""
    short = normalize_experimental_group(raw, default)
    return {
        GROUP_A: "GROUP_A_REALTIME",
        GROUP_B: "GROUP_B_DELAYED",
        GROUP_C: "GROUP_C_CONTROL",
    }[short]


def _read_rows(path: str) -> list[dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    return [row for row in raw if isinstance(row, dict)] if isinstance(raw, list) else []


def _write_rows(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(list(rows), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    finally:
        try:
            if temp.exists():
                temp.unlink()
        except OSError:
            pass


def _same_scope(
    row: Mapping[str, Any],
    *,
    student_number: str,
    experimental_group: str,
    timepoint: str,
    lesson_id: str,
) -> bool:
    row_lesson = str(
        row.get("lessonId") or str(row.get("timestamp") or "")[:10]
    ).strip()
    return (
        str(row.get("studentNumber") or "").strip() == student_number
        and canonical_group(str(row.get("experimentalGroup") or "")) == experimental_group
        and normalize_timepoint(str(row.get("timepoint") or "")) == timepoint
        and row_lesson == lesson_id
    )


def summarize_attempts(
    rows: Sequence[Mapping[str, Any]],
    *,
    student_number: str,
    experimental_group: str,
    timepoint: str,
    lesson_id: str = "",
    planned_attempts: int = DEFAULT_PLANNED_ATTEMPTS,
) -> dict[str, Any]:
    """Summarize dose compliance for one student, group, and timepoint."""
    student = str(student_number or "").strip()
    group = canonical_group(experimental_group)
    point = normalize_timepoint(timepoint)
    lesson = str(lesson_id or time.strftime("%Y-%m-%d")).strip()[:64]
    target = max(1, min(999, int(planned_attempts or DEFAULT_PLANNED_ATTEMPTS)))
    scoped = [
        row
        for row in rows
        if _same_scope(
            row,
            student_number=student,
            experimental_group=group,
            timepoint=point,
            lesson_id=lesson,
        )
    ]
    valid = sum(bool(row.get("valid")) for row in scoped)
    invalid = len(scoped) - valid
    return {
        "studentNumber": student,
        "experimentalGroup": group,
        "timepoint": point,
        "lessonId": lesson,
        "totalAttempts": len(scoped),
        "validAttempts": valid,
        "invalidAttempts": invalid,
        "plannedAttempts": target,
        "remainingAttempts": max(0, target - valid),
        "doseComplete": valid >= target,
    }


def record_attempt(
    *,
    session_id: str,
    student_number: str,
    school: str,
    class_group: str,
    experimental_group: str,
    timepoint: str,
    lesson_id: str = "",
    source: str,
    quality_gate: Optional[Mapping[str, Any]],
    score: Optional[float],
    planned_attempts: int = DEFAULT_PLANNED_ATTEMPTS,
    ledger_path: Optional[str] = None,
) -> dict[str, Any]:
    """Append one idempotent attempt and return the current scoped summary."""
    path = ledger_path or DEFAULT_LEDGER_PATH
    session_key = str(session_id or "").strip()
    student = str(student_number or "").strip()
    group = canonical_group(experimental_group)
    point = normalize_timepoint(timepoint)
    lesson = str(lesson_id or time.strftime("%Y-%m-%d")).strip()[:64]
    gate = dict(quality_gate or {})
    grade = str(gate.get("grade") or "C").strip().upper()
    valid = grade == "A" and bool(gate.get("researchEligible", True))
    issue_rows = gate.get("issues") if isinstance(gate.get("issues"), list) else []
    invalid_reasons = [
        {
            "code": str(item.get("code") or "quality_gate"),
            "label": str(item.get("label") or item.get("message") or "采集质量未达标"),
        }
        for item in issue_rows
        if isinstance(item, Mapping)
    ]
    if not valid and not invalid_reasons:
        invalid_reasons = [{"code": "quality_gate", "label": str(gate.get("summary") or "采集质量未达标")}]

    with _ledger_lock:
        rows = _read_rows(path)
        existing = next(
            (row for row in rows if str(row.get("sessionId") or "") == session_key),
            None,
        )
        if existing is None:
            scoped_before = summarize_attempts(
                rows,
                student_number=student,
                experimental_group=group,
                timepoint=point,
                lesson_id=lesson,
                planned_attempts=planned_attempts,
            )
            row = {
                "id": str(uuid.uuid4()),
                "sessionId": session_key,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "school": str(school or "").strip(),
                "classGroup": str(class_group or "").strip(),
                "studentNumber": student,
                "experimentalGroup": group,
                "timepoint": point,
                "lessonId": lesson,
                "source": str(source or "unknown"),
                "attemptOrdinal": int(scoped_before["totalAttempts"]) + 1,
                "valid": valid,
                "invalidReasons": invalid_reasons if not valid else [],
                "qualityGrade": grade,
                "qualityScore": gate.get("score"),
                "researchEligible": valid,
                "score": float(score) if valid and score is not None else None,
                "protocolVersion": RESEARCH_PROTOCOL_VERSION,
                "interventionPolicy": (
                    "immediate_feedback"
                    if group == "GROUP_A_REALTIME"
                    else "delayed_feedback"
                    if group == "GROUP_B_DELAYED"
                    else "no_participant_feedback"
                ),
                "feedbackSuppressionExpected": group
                in {"GROUP_B_DELAYED", "GROUP_C_CONTROL"},
            }
            rows.append(row)
            _write_rows(path, rows)
            existing = row

        summary = summarize_attempts(
            rows,
            student_number=student,
            experimental_group=group,
            timepoint=point,
            lesson_id=lesson,
            planned_attempts=planned_attempts,
        )
        return {"attempt": dict(existing), "summary": summary}


def read_attempts(
    *,
    student_number: str = "",
    experimental_group: str = "",
    timepoint: str = "",
    lesson_id: str = "",
    planned_attempts: int = DEFAULT_PLANNED_ATTEMPTS,
    ledger_path: Optional[str] = None,
) -> dict[str, Any]:
    """Read a filtered ledger snapshot without exposing media or prompts."""
    path = ledger_path or DEFAULT_LEDGER_PATH
    student = str(student_number or "").strip()
    group = canonical_group(experimental_group) if experimental_group else ""
    point = normalize_timepoint(timepoint) if timepoint else ""
    lesson = str(lesson_id or "").strip()[:64]
    with _ledger_lock:
        rows = _read_rows(path)
    filtered = [
        row
        for row in rows
        if (not student or str(row.get("studentNumber") or "").strip() == student)
        and (not group or canonical_group(str(row.get("experimentalGroup") or "")) == group)
        and (not point or normalize_timepoint(str(row.get("timepoint") or "")) == point)
        and (
            not lesson
            or str(row.get("lessonId") or str(row.get("timestamp") or "")[:10]).strip()
            == lesson
        )
    ]
    result: dict[str, Any] = {"records": filtered, "count": len(filtered)}
    if student and group and point:
        result["summary"] = summarize_attempts(
            rows,
            student_number=student,
            experimental_group=group,
            timepoint=point,
            lesson_id=lesson or time.strftime("%Y-%m-%d"),
            planned_attempts=planned_attempts,
        )
    return result
