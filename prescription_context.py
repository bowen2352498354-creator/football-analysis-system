# -*- coding: utf-8 -*-
"""Phase 4: deterministic evidence and longitudinal context for prescriptions."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Optional


METRIC_LABELS = {
    "distance_cm": "支撑脚横距",
    "toe_angle": "支撑脚尖方向",
    "max_folding_angle": "后摆折叠",
    "whipping_velocity": "鞭打速度",
    "impact_knee_angle": "触球膝角",
    "ankle_rigidity": "脚踝锁定",
    "support_knee_angle": "支撑膝缓冲",
    "hip_torsion_angle": "转髋幅度",
    "trunk_lean_angle": "躯干倾角",
}

DEFAULT_GREEN_BANDS = {
    "distance_cm": (0.40, 0.70),
    "toe_angle": (0.0, 15.0),
    "max_folding_angle": (70.0, 100.0),
    "whipping_velocity": (450.0, None),
    "impact_knee_angle": (135.0, 165.0),
    "ankle_rigidity": (0.0, 10.0),
    "support_knee_angle": (135.0, 170.0),
    "hip_torsion_angle": (15.0, 40.0),
    "trunk_lean_angle": (5.0, 15.0),
}

UNIT_LABELS = {
    "ratio": "×肩宽",
    "deg": "°",
    "degree": "°",
    "deg/s": "°/s",
    "variance": "σ²",
}

TRAINING_PROTOCOLS = {
    "distance_cm": {
        "exercise": "支撑脚定点落位",
        "lowCue": "在球侧标出目标区，支撑脚向外落入目标区后再触球",
        "highCue": "在球侧标出目标区，支撑脚向球靠近后再触球",
        "defaultCue": "让支撑脚稳定落在球侧目标区",
        "dosage": "3组×5次慢速落位",
    },
    "toe_angle": {
        "exercise": "脚尖对靶落位",
        "highCue": "支撑脚尖对准球门目标线，落稳后再摆腿",
        "defaultCue": "支撑脚尖沿目标线落地",
        "dosage": "3组×5次定点落位",
    },
    "max_folding_angle": {
        "exercise": "后摆折叠定格",
        "lowCue": "后摆时主动收小腿，达到目标折叠后停顿1秒",
        "highCue": "减少过度收腿，在目标折叠区停顿1秒",
        "defaultCue": "后摆时把小腿收进目标折叠区",
        "dosage": "3组×6次慢动作",
    },
    "whipping_velocity": {
        "exercise": "小腿鞭打加速",
        "lowCue": "髋带动大腿后再快速伸膝，触球后继续随摆",
        "defaultCue": "保持近端带动与触球后的完整随摆",
        "dosage": "4组×4次，组间休息30秒",
    },
    "impact_knee_angle": {
        "exercise": "触球膝角定格",
        "lowCue": "触球时适度伸膝，避免膝关节过度弯曲",
        "highCue": "触球时保留膝部缓冲，避免完全伸直",
        "defaultCue": "把触球膝角控制在目标区",
        "dosage": "3组×5次半速触球",
    },
    "ankle_rigidity": {
        "exercise": "锁踝触球",
        "highCue": "脚趾内收并固定踝关节，触球前后保持脚型不散",
        "defaultCue": "触球窗内保持脚踝稳定",
        "dosage": "3组×6次近距离触球",
    },
    "support_knee_angle": {
        "exercise": "支撑膝缓冲定格",
        "lowCue": "减少支撑膝过度屈曲，落地后保持稳定",
        "highCue": "支撑膝微屈吸收冲击，避免僵直落地",
        "defaultCue": "让支撑膝落入目标缓冲角度",
        "dosage": "3组×5次落位定格",
    },
    "hip_torsion_angle": {
        "exercise": "转髋带腿",
        "lowCue": "先转髋再带动摆动腿，保持胸口朝向目标",
        "highCue": "减少过度旋髋，保持躯干与目标线稳定",
        "defaultCue": "把转髋幅度控制在目标区",
        "dosage": "3组×6次无球分解",
    },
    "trunk_lean_angle": {
        "exercise": "躯干中轴控制",
        "lowCue": "重心微向前送，避免后仰够球",
        "highCue": "收紧核心，减少过度前倾折腰",
        "defaultCue": "保持躯干倾角稳定在目标区",
        "dosage": "3组×5次镜前定格",
    },
}


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _round(value: Any, digits: int = 2) -> Optional[float]:
    number = _finite_float(value)
    return round(number, digits) if number is not None else None


def _is_deleted(record: dict) -> bool:
    value = record.get("is_deleted", record.get("isDeleted", False))
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _quality_source(record: dict) -> Optional[str]:
    grade = str(
        record.get("qualityGrade")
        or (record.get("qualityGate") or {}).get("grade")
        or ""
    ).upper()
    if grade in {"B", "C"} or record.get("researchEligible") is False:
        return None
    return "formal" if grade == "A" or record.get("researchEligible") is True else "legacy"


def _record_indicators(record: dict) -> dict:
    detail = record.get("scoreDetail") or record.get("score_detail") or {}
    indicators = detail.get("indicators") if isinstance(detail, dict) else {}
    return indicators if isinstance(indicators, dict) else {}


def _metric_value(indicators: dict, metric_key: str) -> Optional[float]:
    item = indicators.get(metric_key)
    if not isinstance(item, dict):
        return None
    value = item.get("value")
    if value is None:
        value = item.get("scoring_value")
    if value is None and metric_key == "ankle_rigidity":
        value = item.get("variance")
    return _finite_float(value)


def _mean(values: Iterable[float]) -> Optional[float]:
    clean = [float(value) for value in values if _finite_float(value) is not None]
    return sum(clean) / len(clean) if clean else None


def _clean_identity(value: Any) -> str:
    return str(value or "").strip().casefold()


def _latest_records(records: list[dict], limit: int = 8) -> list[dict]:
    return sorted(records, key=lambda row: str(row.get("timestamp") or ""), reverse=True)[:limit]


def build_history_context(
    records: list[dict],
    *,
    student_number: str,
    school: str = "",
    class_group: str = "",
    current_score_detail: Optional[dict] = None,
) -> dict:
    """Build personal and peer references without allowing B/C samples into means."""
    student_key = _clean_identity(student_number)
    school_key = _clean_identity(school)
    class_key = _clean_identity(class_group)
    accepted: list[dict] = []
    source_counts = {"formal": 0, "legacy": 0, "excluded": 0}
    for raw in records if isinstance(records, list) else []:
        if not isinstance(raw, dict) or _is_deleted(raw):
            continue
        if school_key and _clean_identity(raw.get("school")) != school_key:
            continue
        if class_key and _clean_identity(raw.get("classGroup")) != class_key:
            continue
        source = _quality_source(raw)
        if source is None:
            source_counts["excluded"] += 1
            continue
        accepted.append(raw)
        source_counts[source] += 1

    personal_candidates = [
        row for row in accepted if student_key and _clean_identity(row.get("studentId")) == student_key
    ]
    personal_formal = [row for row in personal_candidates if _quality_source(row) == "formal"]
    personal = personal_formal or personal_candidates
    peer_candidates = [
        row for row in accepted if not student_key or _clean_identity(row.get("studentId")) != student_key
    ]
    peer_rows_by_student: dict[str, list[dict]] = defaultdict(list)
    for row in peer_candidates:
        peer_id = _clean_identity(row.get("studentId"))
        if peer_id:
            peer_rows_by_student[peer_id].append(row)
    peers = []
    for peer_rows in peer_rows_by_student.values():
        formal_rows = [row for row in peer_rows if _quality_source(row) == "formal"]
        peers.extend(formal_rows or peer_rows)
    current_indicators = (
        current_score_detail.get("indicators")
        if isinstance(current_score_detail, dict)
        and isinstance(current_score_detail.get("indicators"), dict)
        else {}
    )

    metric_keys = list(current_indicators.keys())
    comparisons: dict[str, dict] = {}
    latest_personal = _latest_records(personal)
    for metric_key in metric_keys:
        current_value = _metric_value(current_indicators, metric_key)
        personal_values = [
            value
            for row in latest_personal
            if (value := _metric_value(_record_indicators(row), metric_key)) is not None
        ]

        peer_values_by_student: dict[str, list[float]] = defaultdict(list)
        for row in peers:
            peer_id = _clean_identity(row.get("studentId"))
            value = _metric_value(_record_indicators(row), metric_key)
            if peer_id and value is not None:
                peer_values_by_student[peer_id].append(value)
        peer_student_means = [
            value
            for values in peer_values_by_student.values()
            if (value := _mean(values)) is not None
        ]

        personal_mean = _mean(personal_values)
        peer_mean = _mean(peer_student_means)
        previous = personal_values[0] if personal_values else None
        comparisons[metric_key] = {
            "currentValue": _round(current_value),
            "previousValue": _round(previous),
            "personalMean": _round(personal_mean),
            "deltaFromPrevious": _round(current_value - previous)
            if current_value is not None and previous is not None
            else None,
            "deltaFromPersonalMean": _round(current_value - personal_mean)
            if current_value is not None and personal_mean is not None
            else None,
            "classPeerMean": _round(peer_mean),
            "deltaFromClassPeerMean": _round(current_value - peer_mean)
            if current_value is not None and peer_mean is not None
            else None,
            "personalSampleCount": len(personal_values),
            "peerStudentCount": len(peer_student_means),
        }

    current_score = _finite_float((current_score_detail or {}).get("TotalScore"))
    personal_scores = [
        value
        for row in latest_personal
        if (value := _finite_float(row.get("score"))) is not None
    ]
    peer_scores_by_student: dict[str, list[float]] = defaultdict(list)
    for row in peers:
        peer_id = _clean_identity(row.get("studentId"))
        value = _finite_float(row.get("score"))
        if peer_id and value is not None:
            peer_scores_by_student[peer_id].append(value)
    peer_score_mean = _mean(
        value
        for values in peer_scores_by_student.values()
        if (value := _mean(values)) is not None
    )
    personal_score_mean = _mean(personal_scores)

    return {
        "policy": "A-grade formal records preferred; legacy records retained for compatibility; B/C excluded",
        "studentNumber": student_number,
        "school": school,
        "classGroup": class_group,
        "sourceCounts": source_counts,
        "personalAttemptCount": len(personal),
        "personalHistorySource": "formal"
        if personal_formal
        else "legacy"
        if personal_candidates
        else "none",
        "peerStudentCount": len(peer_scores_by_student),
        "scoreComparison": {
            "currentScore": _round(current_score),
            "previousScore": _round(personal_scores[0]) if personal_scores else None,
            "personalMean": _round(personal_score_mean),
            "deltaFromPersonalMean": _round(current_score - personal_score_mean)
            if current_score is not None and personal_score_mean is not None
            else None,
            "classPeerMean": _round(peer_score_mean),
            "deltaFromClassPeerMean": _round(current_score - peer_score_mean)
            if current_score is not None and peer_score_mean is not None
            else None,
        },
        "metricComparisons": comparisons,
    }


def _green_band(item: dict, metric_key: str) -> tuple[Optional[float], Optional[float]]:
    raw = item.get("green_band") if isinstance(item, dict) else None
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return _finite_float(raw[0]), _finite_float(raw[1])
    return DEFAULT_GREEN_BANDS.get(metric_key, (None, None))


def _unit(item: dict) -> str:
    raw = str(item.get("unit") or "").strip()
    return UNIT_LABELS.get(raw.lower(), raw)


def _format_number(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "未测"
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".")


def _band_text(low: Optional[float], high: Optional[float], unit: str) -> str:
    if low is not None and high is not None:
        return f"{_format_number(low)}–{_format_number(high)}{unit}"
    if low is not None:
        return f"≥{_format_number(low)}{unit}"
    if high is not None:
        return f"≤{_format_number(high)}{unit}"
    return "暂无标准区间"


def _direction(value: Optional[float], low: Optional[float], high: Optional[float]) -> str:
    if value is None:
        return "未知"
    if low is not None and value < low:
        return "偏低"
    if high is not None and value > high:
        return "偏高"
    return "区间内"


def _priority_target(metric: dict) -> dict:
    key = str(metric.get("metricKey") or "")
    protocol = TRAINING_PROTOCOLS.get(
        key,
        {
            "exercise": "核心动作分解练习",
            "defaultCue": "围绕本次最大扣分项做慢动作控制",
            "dosage": "3组×5次",
        },
    )
    direction = metric.get("deviationDirection")
    cue_key = "lowCue" if direction == "偏低" else "highCue" if direction == "偏高" else "defaultCue"
    cue = protocol.get(cue_key) or protocol.get("defaultCue")
    return {
        "metricKey": key,
        "label": metric.get("label") or key,
        "measuredValue": metric.get("measuredValue"),
        "unit": metric.get("unit") or "",
        "deviationDirection": direction,
        "standardRange": metric.get("standardRange"),
        "penalty": metric.get("penalty"),
        "reason": metric.get("reason"),
        "exercise": protocol.get("exercise"),
        "cue": cue,
        "dosage": protocol.get("dosage"),
        "retestCriterion": f"复测时进入{(metric.get('standardRange') or {}).get('text') or '目标区'}",
    }


def build_prescription_evidence(
    diagnosis: Optional[dict],
    *,
    history_context: Optional[dict] = None,
    quality_gate: Optional[dict] = None,
) -> dict:
    """Create the auditable Top-3 evidence pack and one deterministic priority target."""
    diagnosis = diagnosis if isinstance(diagnosis, dict) else {}
    detail = diagnosis.get("score_detail") or {}
    indicators = detail.get("indicators") if isinstance(detail, dict) else {}
    indicators = indicators if isinstance(indicators, dict) else {}
    deductions = detail.get("deductions") if isinstance(detail, dict) else []
    deductions = deductions if isinstance(deductions, list) else []

    reason_by_metric = {}
    error_by_metric = {}
    for row in deductions:
        if not isinstance(row, dict):
            continue
        key = row.get("metric_key") or row.get("metric") or row.get("key")
        if key:
            reason_by_metric[str(key)] = str(row.get("reason") or "").strip()
            error_by_metric[str(key)] = row.get("error_code")

    rows = []
    for metric_key, item in indicators.items():
        if not isinstance(item, dict):
            continue
        penalty = _finite_float(item.get("penalty")) or 0.0
        if penalty <= 0:
            continue
        value = _finite_float(item.get("value"))
        if value is None:
            value = _finite_float(item.get("scoring_value"))
        low, high = _green_band(item, metric_key)
        unit = _unit(item)
        comparison = (history_context or {}).get("metricComparisons", {}).get(metric_key, {})
        rows.append(
            {
                "metricKey": metric_key,
                "label": METRIC_LABELS.get(metric_key, metric_key),
                "measuredValue": _round(value),
                "unit": unit,
                "standardRange": {
                    "low": _round(low),
                    "high": _round(high),
                    "text": _band_text(low, high, unit),
                },
                "deviationDirection": _direction(value, low, high),
                "penalty": _round(penalty),
                "status": item.get("status"),
                "provenance": item.get("provenance") or "unknown",
                "confidence": _round(item.get("confidence"), 3),
                "method": item.get("method"),
                "errorCode": error_by_metric.get(metric_key),
                "reason": reason_by_metric.get(metric_key)
                or f"{METRIC_LABELS.get(metric_key, metric_key)}偏离目标区",
                "historyComparison": comparison,
            }
        )
    rows.sort(key=lambda row: (-float(row.get("penalty") or 0.0), str(row.get("metricKey"))))
    top_three = rows[:3]
    target = _priority_target(top_three[0]) if top_three else None
    gate = quality_gate or detail.get("quality_gate") or {}
    return {
        "version": "phase4-v1",
        "selectionRule": "highest deterministic penalty; exactly one training target",
        "topDeductions": top_three,
        "priorityTarget": target,
        "quality": {
            "grade": gate.get("grade"),
            "score": gate.get("score"),
            "researchEligible": gate.get("researchEligible"),
            "summary": gate.get("summary"),
        },
        "historyContext": history_context or {},
    }
