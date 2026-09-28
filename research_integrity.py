# -*- coding: utf-8 -*-
"""Research-integrity helpers shared by archive, reports, and analytics.

This module never scores an action. It records how a score was produced,
builds deterministic longitudinal evidence, and audits intervention exposure.
"""

from __future__ import annotations

import math
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

from empirical_thresholds import load_empirical_thresholds
from experiment_ledger import canonical_group, normalize_timepoint


RESEARCH_PROTOCOL_VERSION = os.environ.get(
    "AIFF_RESEARCH_PROTOCOL_VERSION", "THESIS_2026_DRAFT_V1"
)
ANALYSIS_VERSION = "ShotAnalysisPipeline_V3.11"
MEASUREMENT_VERSION = "MediaPipeTasks_XY2D_V3.11"
MODEL_VERSION = "pose_landmarker_tasks+optional_yolov8n"
LEGACY_VERSION = "legacy_unknown"

FORMAL_PROVENANCE = frozenset({"measured", "calibrated"})


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _record_date(record: Mapping[str, Any]) -> str:
    return str(record.get("testDate") or record.get("timestamp") or "")[:10]


def _record_group(record: Mapping[str, Any]) -> str:
    return canonical_group(
        str(
            record.get("experimentalGroup")
            or record.get("experimental_group")
            or record.get("type")
            or ""
        )
    )


def is_explicit_formal_record(record: Mapping[str, Any]) -> bool:
    """Return True only for explicit A-grade, eligible, non-deleted rows."""
    if bool(record.get("is_deleted") or record.get("isDeleted")):
        return False
    gate = record.get("qualityGate") or record.get("quality_gate") or {}
    grade = record.get("qualityGrade") or record.get("quality_grade")
    if not grade and isinstance(gate, Mapping):
        grade = gate.get("grade")
    eligible = record.get("researchEligible", record.get("research_eligible"))
    return str(grade or "").upper() == "A" and eligible is True


def threshold_version() -> str:
    cfg = load_empirical_thresholds()
    explicit = str(cfg.get("threshold_version") or "").strip()
    if explicit:
        return explicit
    schema = cfg.get("schema_version", "unknown")
    population = str(cfg.get("population") or "unknown").strip()
    return f"empirical-v{schema}:{population}"


def build_version_metadata(score_detail: Optional[Mapping[str, Any]] = None) -> dict[str, str]:
    detail = score_detail if isinstance(score_detail, Mapping) else {}
    return {
        "protocolVersion": RESEARCH_PROTOCOL_VERSION,
        "analysisVersion": str(detail.get("scoring_engine") or ANALYSIS_VERSION),
        "thresholdVersion": threshold_version(),
        "measurementVersion": MEASUREMENT_VERSION,
        "modelVersion": MODEL_VERSION,
    }


def build_measurement_provenance_summary(
    score_detail: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create a compact, auditable provenance inventory for all indicators."""
    detail = score_detail if isinstance(score_detail, Mapping) else {}
    indicators = detail.get("indicators") if isinstance(detail.get("indicators"), Mapping) else {}
    metrics: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for key, raw in indicators.items():
        if not isinstance(raw, Mapping):
            continue
        provenance = str(raw.get("provenance") or "unknown").strip().lower()
        counts[provenance] += 1
        metrics[str(key)] = {
            "provenance": provenance,
            "formalMeasurement": provenance in FORMAL_PROVENANCE,
            "method": raw.get("method"),
            "confidence": _safe_float(raw.get("confidence")),
        }
    return {
        "metrics": metrics,
        "counts": dict(counts),
        "allFormalMeasurements": bool(metrics)
        and all(item["formalMeasurement"] for item in metrics.values()),
    }


def build_intervention_audit(
    experimental_group: str,
    *,
    feedback_suppressed: bool,
    supplied: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Normalize exposure facts and flag violations of the A/B/C protocol."""
    group = canonical_group(experimental_group)
    raw = dict(supplied or {})
    feedback_shown = bool(raw.get("feedbackShown", False))
    traffic_shown = bool(raw.get("trafficLightShown", False))
    replay_shown = bool(raw.get("replayShown", False))
    ai_shown = bool(raw.get("aiAdviceShown", False))
    delayed = bool(raw.get("delayedPresentation", False))
    violations: list[str] = []

    if group == "GROUP_A_REALTIME":
        if feedback_suppressed or not feedback_shown:
            violations.append("A_REALTIME_FEEDBACK_NOT_SHOWN")
        if not traffic_shown:
            violations.append("A_TRAFFIC_LIGHT_NOT_SHOWN")
    elif group == "GROUP_B_DELAYED":
        if not delayed:
            violations.append("B_DELAYED_PRESENTATION_NOT_CONFIRMED")
        if bool(raw.get("immediateFeedbackShown", False)):
            violations.append("B_IMMEDIATE_FEEDBACK_LEAK")
    elif group == "GROUP_C_CONTROL":
        if feedback_shown or traffic_shown or replay_shown or ai_shown:
            violations.append("C_FEEDBACK_LEAK")
        if not feedback_suppressed:
            violations.append("C_SUPPRESSION_NOT_CONFIRMED")

    return {
        "experimentalGroup": group,
        "feedbackShown": feedback_shown,
        "feedbackShownAt": raw.get("feedbackShownAt"),
        "feedbackLatencyMs": _safe_float(raw.get("feedbackLatencyMs")),
        "trafficLightShown": traffic_shown,
        "replayShown": replay_shown,
        "aiAdviceShown": ai_shown,
        "immediateFeedbackShown": bool(raw.get("immediateFeedbackShown", False)),
        "delayedPresentation": delayed,
        "operatorPreviewOnly": bool(raw.get("operatorPreviewOnly", False)),
        "feedbackSuppressed": bool(feedback_suppressed),
        "violations": violations,
        "protocolDeviation": bool(violations),
    }


def _extract_indicators(record: Mapping[str, Any]) -> Mapping[str, Any]:
    detail = record.get("scoreDetail") or record.get("score_detail") or {}
    if not isinstance(detail, Mapping):
        return {}
    indicators = detail.get("indicators")
    return indicators if isinstance(indicators, Mapping) else {}


_FIVE_DIMENSION_ALIASES: dict[str, tuple[str, ...]] = {
    "approach_rhythm": ("approach_rhythm", "approach_rhythm_score", "approach_score"),
    "support_stability": ("support_stability", "support_stability_score", "support_score"),
    "backswing_folding": ("backswing_folding", "backswing_folding_score", "backswing_score"),
    "ankle_rigidity": ("ankle_rigidity", "ankle_rigidity_score"),
    "whipping_velocity": ("whipping_velocity", "whipping_velocity_score", "whipping_score"),
}


def _extract_five_dimension_scores(record: Mapping[str, Any]) -> dict[str, float]:
    raw: Any = None
    for key in ("quantified5dScores", "radar_scores", "radarScores"):
        candidate = record.get(key)
        if isinstance(candidate, Mapping):
            raw = candidate
            break
    if raw is None:
        detail = record.get("scoreDetail") or record.get("score_detail") or {}
        if isinstance(detail, Mapping):
            candidate = detail.get("radar_scores") or detail.get("radarScores")
            if isinstance(candidate, Mapping):
                raw = candidate
    if not isinstance(raw, Mapping):
        return {}

    normalized: dict[str, float] = {}
    for dimension, aliases in _FIVE_DIMENSION_ALIASES.items():
        for alias in aliases:
            value = _safe_float(raw.get(alias))
            if value is not None:
                normalized[dimension] = round(value, 3)
                break
    return normalized


def normalize_individual_attempt(record: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce one archive row to the evidence allowed into an LLM prompt."""
    indicators = _extract_indicators(record)
    metrics: dict[str, Any] = {}
    for key, raw in indicators.items():
        if not isinstance(raw, Mapping):
            continue
        metrics[str(key)] = {
            "value": _safe_float(raw.get("value", raw.get("scoring_value"))),
            "unit": raw.get("unit"),
            "status": raw.get("status"),
            "penalty": _safe_float(raw.get("penalty")),
            "provenance": str(raw.get("provenance") or "unknown").lower(),
            "confidence": _safe_float(raw.get("confidence")),
        }
    return {
        "attemptId": str(record.get("attemptId") or record.get("id") or ""),
        "timestamp": str(record.get("timestamp") or ""),
        "testDate": _record_date(record),
        "timepoint": normalize_timepoint(str(record.get("timepoint") or "T0")),
        "experimentalGroup": _record_group(record),
        "score": _safe_float(record.get("score")),
        "fiveDimensionScores": _extract_five_dimension_scores(record),
        "qualityGrade": str(record.get("qualityGrade") or "").upper(),
        "researchEligible": record.get("researchEligible") is True,
        "metrics": metrics,
        "deductions": list(
            (record.get("scoreDetail") or {}).get("deductions") or []
        )
        if isinstance(record.get("scoreDetail"), Mapping)
        else [],
        "errors": [str(item) for item in (record.get("biomechanicalErrors") or []) if item],
        "protocolVersion": str(record.get("protocolVersion") or LEGACY_VERSION),
    }


def filter_individual_attempts_by_date(
    attempts: Sequence[Mapping[str, Any]],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> list[Mapping[str, Any]]:
    """Apply an inclusive ISO-date range before research-quality gating."""

    start = str(date_from or "").strip() or None
    end = str(date_to or "").strip() or None
    for label, value in (("dateFrom", start), ("dateTo", end)):
        if value is None:
            continue
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"{label} must use YYYY-MM-DD") from exc
    if start and end and start > end:
        raise ValueError("dateFrom must not be later than dateTo")

    selected: list[Mapping[str, Any]] = []
    for attempt in attempts:
        day = _record_date(attempt)
        if not day or day == "unknown":
            if start or end:
                continue
        if start and day < start:
            continue
        if end and day > end:
            continue
        selected.append(attempt)
    return selected


def _linear_slope(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    x_mean = (len(values) - 1) / 2.0
    y_mean = statistics.fmean(values)
    denominator = sum((i - x_mean) ** 2 for i in range(len(values)))
    if denominator <= 0:
        return None
    return sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values)) / denominator


def summarize_individual_attempts(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build deterministic longitudinal evidence before any LLM wording."""
    normalized = [normalize_individual_attempt(item) for item in attempts]
    formal = [item for item in normalized if item["qualityGrade"] == "A" and item["researchEligible"]]
    formal.sort(key=lambda item: (item["timestamp"], item["attemptId"]))
    scores = [item["score"] for item in formal if item["score"] is not None]
    error_counts: Counter[str] = Counter()
    metric_values: dict[str, list[float]] = defaultdict(list)
    metric_statuses: dict[str, Counter[str]] = defaultdict(Counter)
    five_dimension_values: dict[str, list[float]] = defaultdict(list)
    for item in formal:
        error_counts.update(item["errors"])
        for key, metric in item["metrics"].items():
            value = metric.get("value")
            if value is not None:
                metric_values[key].append(float(value))
            status = str(metric.get("status") or "")
            if status:
                metric_statuses[key][status] += 1
        for key, value in item.get("fiveDimensionScores", {}).items():
            five_dimension_values[key].append(float(value))

    metric_trends: dict[str, Any] = {}
    for key, values in metric_values.items():
        metric_trends[key] = {
            "n": len(values),
            "first": round(values[0], 3),
            "latest": round(values[-1], 3),
            "change": round(values[-1] - values[0], 3),
            "mean": round(statistics.fmean(values), 3),
            "slopePerAttempt": round(_linear_slope(values), 4) if len(values) >= 2 else None,
            "statusCounts": dict(metric_statuses.get(key, {})),
        }

    mean = statistics.fmean(scores) if scores else None
    sd = statistics.pstdev(scores) if len(scores) >= 2 else 0.0 if scores else None
    versions = sorted({item["protocolVersion"] for item in formal})
    five_dimension_average = {
        key: round(statistics.fmean(values), 2)
        for key, values in five_dimension_values.items()
        if values
    }
    return {
        "formalAttemptCount": len(formal),
        "excludedAttemptCount": len(normalized) - len(formal),
        "scoreSummary": {
            "history": [round(v, 2) for v in scores],
            "mean": round(mean, 2) if mean is not None else None,
            "sd": round(sd, 2) if sd is not None else None,
            "first": round(scores[0], 2) if scores else None,
            "latest": round(scores[-1], 2) if scores else None,
            "change": round(scores[-1] - scores[0], 2) if scores else None,
            "slopePerAttempt": round(_linear_slope(scores), 4) if len(scores) >= 2 else None,
        },
        "errorCounts": dict(error_counts.most_common()),
        "errorRates": {
            key: round(count / max(len(formal), 1), 4)
            for key, count in error_counts.items()
        },
        "metricTrends": metric_trends,
        "fiveDimensionScores": five_dimension_average,
        "period": {
            "start": formal[0]["testDate"] if formal else None,
            "end": formal[-1]["testDate"] if formal else None,
        },
        "timepoints": sorted({item["timepoint"] for item in formal}),
        "protocolVersions": versions,
        "mixedProtocolVersions": len(versions) > 1,
        "attempts": formal,
    }


def research_exclusion_reasons(
    record: Mapping[str, Any], *, require_version: bool = True
) -> list[str]:
    reasons: list[str] = []
    if bool(record.get("is_deleted") or record.get("isDeleted")):
        reasons.append("soft_deleted")
    if not is_explicit_formal_record(record):
        reasons.append("not_explicit_A_research_eligible")
    if not str(record.get("studentId") or record.get("student_id") or "").strip():
        reasons.append("missing_student_id")
    raw_timepoint = str(record.get("timepoint") or "").strip().upper()
    if raw_timepoint not in {"T0", "T1", "T2", "T3", "T4"}:
        reasons.append("missing_or_invalid_timepoint")
    audit = record.get("interventionAudit") or {}
    if bool(record.get("protocolDeviation")) or (
        isinstance(audit, Mapping) and bool(audit.get("protocolDeviation"))
    ):
        reasons.append("protocol_deviation")
    if require_version and str(record.get("protocolVersion") or LEGACY_VERSION) == LEGACY_VERSION:
        reasons.append("legacy_unknown_protocol_version")
    return list(dict.fromkeys(reasons))
