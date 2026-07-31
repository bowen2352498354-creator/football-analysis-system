# -*- coding: utf-8 -*-
"""coach_calibration.py — Phase 4 教练人工标定（写入 calibrated provenance）。

防幻觉：人工覆写必须显式 provenance=calibrated，并保留审计轨迹；
不得把教练填写值伪装成传感器 measured。
"""

from __future__ import annotations

import time
from typing import Any, Optional

from error_diagnoser import (
    AIGC_FOCUS_METRIC_KEYS,
    PROVENANCE_CALIBRATED,
    is_aigc_measurable_provenance,
)

# 归档顶层字段映射（camelCase 与 snake 双写，兼容导出）
_METRIC_ARCHIVE_FIELDS: dict[str, tuple[str, ...]] = {
    "distance_cm": (
        "supportFootDistance",
        "distance_cm",
        "support_lateral_dist_cm",
        "support_foot_distance",
    ),
    "max_folding_angle": (
        "max_folding_angle",
        "maxFoldingAngle",
    ),
    "ankle_rigidity": (
        "ankle_rigidity",
        "ankle_rigidity_variance",
        "ankleRigidity",
    ),
}

_METRIC_PROVENANCE_FIELDS: dict[str, tuple[str, ...]] = {
    "distance_cm": (
        "supportFootDistanceProvenance",
        "distance_cm_provenance",
        "support_foot_distance_provenance",
    ),
    "max_folding_angle": (
        "maxFoldingAngleProvenance",
        "max_folding_angle_provenance",
    ),
    "ankle_rigidity": (
        "ankleRigidityProvenance",
        "ankle_rigidity_provenance",
    ),
}


def allowed_calibrate_metrics() -> frozenset[str]:
    return frozenset(AIGC_FOCUS_METRIC_KEYS)


def apply_coach_calibration(
    record: dict[str, Any],
    *,
    metric_key: str,
    value: float,
    coach_id: Optional[str] = None,
    note: Optional[str] = None,
) -> dict[str, Any]:
    """就地更新归档记录：焦点指标写为 calibrated，并追加审计条目。

    返回 ``{"ok": True, "record": ..., "audit": ...}`` 或 ``{"ok": False, "message": ...}``。
    """
    key = str(metric_key or "").strip()
    if key not in allowed_calibrate_metrics():
        return {
            "ok": False,
            "message": f"不支持标定指标：{key}（仅允许 {sorted(allowed_calibrate_metrics())}）",
        }
    try:
        number = float(value)
    except (TypeError, ValueError):
        return {"ok": False, "message": "标定值必须是有限数值"}
    if number != number:  # NaN
        return {"ok": False, "message": "标定值不可为 NaN"}

    # 合理物理范围护栏（防误填）
    ranges = {
        "distance_cm": (0.0, 60.0),
        "max_folding_angle": (0.0, 160.0),
        "ankle_rigidity": (0.0, 200.0),
    }
    lo, hi = ranges[key]
    if not (lo <= number <= hi):
        return {
            "ok": False,
            "message": f"{key} 标定值超出合理范围 [{lo}, {hi}]",
        }

    rounded = round(number, 4 if key == "ankle_rigidity" else 2)
    for field in _METRIC_ARCHIVE_FIELDS[key]:
        record[field] = rounded
    for field in _METRIC_PROVENANCE_FIELDS[key]:
        record[field] = PROVENANCE_CALIBRATED

    # 同步 scoreDetail.indicators（若存在）供 AIGC / 看板复用
    detail = record.get("scoreDetail") or record.get("score_detail")
    if not isinstance(detail, dict):
        detail = {"indicators": {}}
        record["scoreDetail"] = detail
    indicators = detail.get("indicators")
    if not isinstance(indicators, dict):
        indicators = {}
        detail["indicators"] = indicators
    entry = indicators.get(key) if isinstance(indicators.get(key), dict) else {}
    entry = dict(entry)
    entry["value"] = rounded
    entry["scoring_value"] = rounded
    entry["provenance"] = PROVENANCE_CALIBRATED
    entry["method"] = "coach_manual"
    if key == "ankle_rigidity":
        entry["variance"] = rounded
        entry["scoring_variance"] = rounded
    indicators[key] = entry
    detail["indicators"] = indicators
    record["scoreDetail"] = detail
    if "score_detail" in record:
        record["score_detail"] = detail

    audit = {
        "metric_key": key,
        "value": rounded,
        "provenance": PROVENANCE_CALIBRATED,
        "coach_id": str(coach_id or "coach").strip() or "coach",
        "note": str(note or "").strip()[:200],
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    trail = record.get("calibration_audit")
    if not isinstance(trail, list):
        trail = []
    trail.append(audit)
    record["calibration_audit"] = trail
    record["lastCalibratedAt"] = audit["calibrated_at"]
    record["lastCalibratedMetric"] = key

    assert is_aigc_measurable_provenance(PROVENANCE_CALIBRATED)
    return {"ok": True, "record": record, "audit": audit}
