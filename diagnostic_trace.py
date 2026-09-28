# -*- coding: utf-8 -*-
"""Lightweight diagnostic tracing for the feedback/report pipeline.

The trace is intentionally a side channel: failures here must never affect the
training session, scoring result, or LLM report returned to the frontend.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from datetime import datetime
from typing import Any, Optional


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACE_DIR = os.getenv(
    "FEEDBACK_DIAGNOSTIC_TRACE_DIR",
    os.path.join(PROJECT_DIR, "diagnostic_traces"),
)
TRACE_ENABLED = os.getenv("FEEDBACK_DIAGNOSTIC_TRACE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

MAX_DEPTH = 6
MAX_LIST_ITEMS = 60
MAX_STRING_CHARS = 4000
PREVIEW_CHARS = 500

_LOCK = threading.Lock()
_REDACT_KEY_FRAGMENTS = (
    "api_key",
    "authorization",
    "base64",
    "canvas",
    "data_uri",
    "frame",
    "heatmap",
    "image",
    "password",
    "secret",
    "thumbnail",
    "token",
)


def text_fingerprint(text: Any) -> str:
    """Return a stable short fingerprint without storing long raw text."""
    raw = "" if text is None else str(text)
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]


def _should_redact(key: Optional[str]) -> bool:
    if not key:
        return False
    lower = key.lower()
    return any(fragment in lower for fragment in _REDACT_KEY_FRAGMENTS)


def _safe_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _numeric_series_summary(values: list[Any]) -> dict[str, Any]:
    numbers = [_safe_number(v) for v in values]
    clean = [v for v in numbers if v is not None]
    if not clean:
        return {"count": len(values), "numeric_count": 0}
    return {
        "count": len(values),
        "numeric_count": len(clean),
        "min": round(min(clean), 4),
        "max": round(max(clean), 4),
        "mean": round(sum(clean) / len(clean), 4),
        "first": round(clean[0], 4),
        "last": round(clean[-1], 4),
    }


def sanitize_for_trace(value: Any, *, key: Optional[str] = None, depth: int = 0) -> Any:
    """Convert arbitrary runtime objects into bounded, privacy-aware JSON."""
    # Keep scalar counters and flags visible even when their key contains words
    # such as "frame" or "image" (for example frame_count / has_image).
    if value is None or isinstance(value, (bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if _should_redact(key):
        try:
            size = len(value)  # type: ignore[arg-type]
        except Exception:
            size = None
        return {"redacted": True, "reason": "sensitive_or_large_media", "size": size}

    if depth > MAX_DEPTH:
        return {"truncated": True, "reason": "max_depth"}

    if isinstance(value, str):
        if value.startswith("data:image/"):
            return {
                "redacted": True,
                "reason": "data_image",
                "length": len(value),
                "fingerprint": text_fingerprint(value),
            }
        if len(value) > MAX_STRING_CHARS:
            return {
                "truncated": True,
                "length": len(value),
                "fingerprint": text_fingerprint(value),
                "preview": value[:PREVIEW_CHARS],
            }
        return value

    if isinstance(value, dict):
        return {
            str(k): sanitize_for_trace(v, key=str(k), depth=depth + 1)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        items = list(value)
        if len(items) > 12 and all(_safe_number(v) is not None for v in items):
            return _numeric_series_summary(items)
        if len(items) > MAX_LIST_ITEMS:
            return {
                "truncated": True,
                "count": len(items),
                "items": [
                    sanitize_for_trace(v, depth=depth + 1)
                    for v in items[:MAX_LIST_ITEMS]
                ],
            }
        return [sanitize_for_trace(v, depth=depth + 1) for v in items]

    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"type": type(value).__name__, "shape": tuple(shape)}

    return str(value)


def summarize_records(records: list[dict]) -> dict[str, Any]:
    records = records or []
    statuses: dict[str, int] = {}
    knee_angles: list[Any] = []
    angular_velocities: list[Any] = []
    stability_values: list[Any] = []
    for record in records:
        status = record.get("status")
        if status:
            statuses[str(status)] = statuses.get(str(status), 0) + 1
        if record.get("knee_angle") is not None:
            knee_angles.append(record.get("knee_angle"))
        if record.get("angular_velocity") is not None:
            angular_velocities.append(record.get("angular_velocity"))
        if record.get("stability_index") is not None:
            stability_values.append(record.get("stability_index"))

    return {
        "count": len(records),
        "status_counts": statuses,
        "knee_angle": _numeric_series_summary(knee_angles),
        "angular_velocity": _numeric_series_summary(angular_velocities),
        "stability_index": _numeric_series_summary(stability_values),
        "first": sanitize_for_trace(records[0]) if records else None,
        "last": sanitize_for_trace(records[-1]) if records else None,
    }


def summarize_capture_quality(session: Any) -> dict[str, Any]:
    """Combine pipeline counters with MediaPipe visibility statistics."""
    if session is None:
        return {}
    try:
        base = dict(session.get_capture_diagnostics() or {})
    except Exception:
        base = {}

    try:
        pose_frames = list(getattr(session, "_trajectory_pose_frames", []) or [])
    except Exception:
        pose_frames = []

    all_visibility: list[float] = []
    critical_frame_scores: list[float] = []
    critical_joints = {
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    }
    valid_pose_frames = 0
    full_body_frames = 0
    edge_clipped_frames = 0
    body_height_ratios: list[float] = []
    resolution = base.get("resolution") if isinstance(base.get("resolution"), dict) else {}
    frame_width = _safe_number(resolution.get("width")) or 0.0
    frame_height = _safe_number(resolution.get("height")) or 0.0
    full_body_joints = critical_joints | {
        "left_foot_index",
        "right_foot_index",
    }
    for frame in pose_frames:
        if not isinstance(frame, dict):
            continue
        visibility = frame.get("visibility")
        if not isinstance(visibility, dict):
            continue
        values = [
            number
            for number in (_safe_number(v) for v in visibility.values())
            if number is not None
        ]
        if values and max(values) > 0.0:
            valid_pose_frames += 1
        all_visibility.extend(values)
        critical_values = [
            number
            for name, raw in visibility.items()
            if name in critical_joints
            for number in [_safe_number(raw)]
            if number is not None
        ]
        if critical_values:
            critical_frame_scores.append(sum(critical_values) / len(critical_values))

        if frame_width > 0 and frame_height > 0:
            visible_points: list[tuple[float, float]] = []
            full_body_ok = True
            for name in full_body_joints:
                confidence = _safe_number(visibility.get(name))
                point = frame.get(name)
                if (
                    confidence is None
                    or confidence < 0.5
                    or not isinstance(point, (list, tuple))
                    or len(point) < 2
                ):
                    full_body_ok = False
                    continue
                x = _safe_number(point[0])
                y = _safe_number(point[1])
                if x is None or y is None:
                    full_body_ok = False
                    continue
                if not (0.0 <= x <= frame_width and 0.0 <= y <= frame_height):
                    full_body_ok = False
                    continue
                visible_points.append((x, y))
            if full_body_ok:
                full_body_frames += 1

            all_visible_points: list[tuple[float, float]] = []
            for name, raw_confidence in visibility.items():
                confidence = _safe_number(raw_confidence)
                point = frame.get(name)
                if (
                    confidence is None
                    or confidence < 0.5
                    or not isinstance(point, (list, tuple))
                    or len(point) < 2
                ):
                    continue
                x = _safe_number(point[0])
                y = _safe_number(point[1])
                if x is not None and y is not None:
                    all_visible_points.append((x, y))
            if all_visible_points:
                ys = [point[1] for point in all_visible_points]
                body_height_ratios.append(
                    max(0.0, min(1.5, (max(ys) - min(ys)) / frame_height))
                )
                margin_x = frame_width * 0.03
                margin_y = frame_height * 0.03
                if any(
                    x <= margin_x
                    or x >= frame_width - margin_x
                    or y <= margin_y
                    or y >= frame_height - margin_y
                    for x, y in all_visible_points
                ):
                    edge_clipped_frames += 1

    total_frames = int(base.get("read_frame_count") or len(pose_frames) or 0)
    pose_ratio = valid_pose_frames / total_frames if total_frames > 0 else None
    low_confidence_frames = sum(v < 0.5 for v in critical_frame_scores)
    confidence_frame_count = len(critical_frame_scores)
    base["pose_visibility"] = {
        "valid_pose_frame_count": valid_pose_frames,
        "valid_pose_frame_ratio": round(pose_ratio, 4)
        if pose_ratio is not None
        else None,
        "landmark_sample_count": len(all_visibility),
        "landmark_visibility_mean": round(
            sum(all_visibility) / len(all_visibility), 4
        )
        if all_visibility
        else None,
        "landmark_visibility_min": round(min(all_visibility), 4)
        if all_visibility
        else None,
        "critical_visibility_frame_count": confidence_frame_count,
        "critical_visibility_mean": round(
            sum(critical_frame_scores) / confidence_frame_count, 4
        )
        if confidence_frame_count
        else None,
        "critical_low_confidence_frame_count": low_confidence_frames,
        "critical_low_confidence_frame_ratio": round(
            low_confidence_frames / confidence_frame_count, 4
        )
        if confidence_frame_count
        else None,
        "low_confidence_threshold": 0.5,
    }
    geometry_denominator = confidence_frame_count or valid_pose_frames
    sorted_height_ratios = sorted(body_height_ratios)
    median_height_ratio = (
        sorted_height_ratios[len(sorted_height_ratios) // 2]
        if sorted_height_ratios
        else None
    )
    base["body_geometry"] = {
        "evaluated_frame_count": geometry_denominator,
        "full_body_frame_count": full_body_frames,
        "full_body_frame_ratio": round(
            full_body_frames / geometry_denominator, 4
        )
        if geometry_denominator > 0
        else None,
        "body_height_ratio_median": round(median_height_ratio, 4)
        if median_height_ratio is not None
        else None,
        "edge_clipped_frame_count": edge_clipped_frames,
        "edge_clipped_frame_ratio": round(
            edge_clipped_frames / len(body_height_ratios), 4
        )
        if body_height_ratios
        else None,
        "visibility_threshold": 0.5,
        "edge_margin_ratio": 0.03,
    }
    ball_detection = base.get("ball_detection")
    if isinstance(ball_detection, dict):
        base["ball_visibility"] = dict(ball_detection)
    return sanitize_for_trace(base)


def _summarize_indicators(indicators: Any) -> dict[str, Any]:
    if not isinstance(indicators, dict):
        return {}
    status_counts: dict[str, int] = {}
    summary: dict[str, Any] = {}
    for name, detail in indicators.items():
        if not isinstance(detail, dict):
            summary[str(name)] = sanitize_for_trace(detail)
            continue
        status = detail.get("status")
        if status:
            status_counts[str(status)] = status_counts.get(str(status), 0) + 1
        summary[str(name)] = {
            key: sanitize_for_trace(detail.get(key), key=key)
            for key in (
                "value",
                "status",
                "penalty",
                "provenance",
                "confidence",
                "threshold",
                "measured_value",
                "target_value",
                "unit",
            )
            if key in detail
        }
    return {
        "count": len(indicators),
        "status_counts": status_counts,
        "items": summary,
    }


def summarize_score_detail(score_detail: Optional[dict]) -> Optional[dict[str, Any]]:
    if not isinstance(score_detail, dict):
        return None
    action_roi = score_detail.get("action_roi")
    if isinstance(action_roi, dict):
        action_roi_summary = {
            "start": action_roi.get("start"),
            "end": action_roi.get("end"),
            "frame_count": action_roi.get("frame_count"),
            "impact_index_in_window": action_roi.get("impact_index_in_window"),
            "absolute_timestamps": sanitize_for_trace(
                action_roi.get("absolute_timestamps"),
                key="absolute_timestamps",
            ),
        }
    else:
        action_roi_summary = None

    deductions = score_detail.get("deductions")
    if isinstance(deductions, list):
        deduction_summary = [sanitize_for_trace(item) for item in deductions[:8]]
    else:
        deduction_summary = sanitize_for_trace(deductions)

    return {
        "TotalScore": score_detail.get("TotalScore"),
        "t_impact": score_detail.get("t_impact"),
        "t0_index": score_detail.get("t0_index"),
        "primary_error_code": score_detail.get("primary_error_code"),
        "primary_error_description": score_detail.get("primary_error_description"),
        "indicators": _summarize_indicators(score_detail.get("indicators")),
        "deductions": deduction_summary,
        "radar_scores": sanitize_for_trace(score_detail.get("radar_scores")),
        "action_roi": action_roi_summary,
        "has_heatmap_base64": bool(score_detail.get("heatmap_base64")),
        "has_spatial_trajectory": bool(score_detail.get("spatial_trajectory")),
        "baseline_session_id": score_detail.get("baseline_session_id"),
        "is_baseline_trusted": score_detail.get("is_baseline_trusted"),
        "class_id": score_detail.get("class_id"),
        "camera_height_cm": score_detail.get("camera_height_cm"),
        "calibrator_status": score_detail.get("calibrator_status"),
    }


def summarize_llm_result(result: Optional[dict]) -> Optional[dict[str, Any]]:
    if not isinstance(result, dict):
        return None
    fields = (
        "overview",
        "biomechanical_analysis",
        "magic_metaphor",
        "action_plan",
        "painPoint",
        "prescription",
        "correction_metaphor",
        "praise_encouragement",
        "clinical_echo",
    )
    summary: dict[str, Any] = {
        "score": result.get("score"),
        "aigc_source": result.get("aigc_source") or result.get("aigcSource"),
    }
    for field in fields:
        text = result.get(field)
        if text is None:
            continue
        text = str(text)
        summary[field] = {
            "length": len(text),
            "fingerprint": text_fingerprint(text),
            "preview": text[:PREVIEW_CHARS],
            "contains_number": any(ch.isdigit() for ch in text),
        }
    return summary


def record_feedback_trace(
    stage: str,
    *,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    source: Optional[str] = None,
    payload: Optional[dict[str, Any]] = None,
) -> bool:
    """Append a bounded JSONL trace event. Returns False on any trace failure."""
    if not TRACE_ENABLED:
        return False
    try:
        os.makedirs(TRACE_DIR, exist_ok=True)
        event = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "epoch_ms": int(time.time() * 1000),
            "stage": stage,
            "trace_id": trace_id,
            "session_id": session_id,
            "source": source,
            "payload": sanitize_for_trace(payload or {}),
        }
        path = os.path.join(
            TRACE_DIR,
            f"feedback_trace_{time.strftime('%Y-%m-%d')}.jsonl",
        )
        line = json.dumps(event, ensure_ascii=False, default=str)
        with _LOCK:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return True
    except Exception:
        return False
