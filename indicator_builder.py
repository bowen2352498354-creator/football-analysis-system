# -*- coding: utf-8 -*-
"""
indicator_builder.py
焦点指标打包 / 溯源常量 / 八大量纲默认值与清洗函数。

此模块为纯数据层：零 I/O、零 LLM、无对 error_diagnoser 的循环依赖。
可被 deterministic_scorer / error_diagnoser / api_server 等直接导入。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from biomech_primitives import (
    ANKLE_STIFFNESS_LOCKED,
    ANKLE_STIFFNESS_SLIGHT_DEFORMATION,
    ANKLE_STIFFNESS_YIELDING,
)

# --------------------------------------------------------------------------
# 【Phase 1】实测值溯源：AIGC 仅允许 measured / calibrated 进入复述载荷
# --------------------------------------------------------------------------
PROVENANCE_MEASURED = "measured"
PROVENANCE_CALIBRATED = "calibrated"
PROVENANCE_ESTIMATED = "estimated"
PROVENANCE_DEFAULT = "default"
PROVENANCE_MISSING = "missing"
AIGC_ALLOWED_PROVENANCE = frozenset({PROVENANCE_MEASURED, PROVENANCE_CALIBRATED})
AIGC_FOCUS_METRIC_KEYS = frozenset(
    {"distance_cm", "max_folding_angle", "ankle_rigidity"}
)

# --------------------------------------------------------------------------
# 【全链路透明化】UI / 报告用 PROVENANCE 血统层级（大写枚举）
# 与小写 provenance（AIGC 复述门禁）并存：tier 描述测量方法学置信度。
# --------------------------------------------------------------------------
PROVENANCE_TIER_MEASURED = "MEASURED"
PROVENANCE_TIER_CALIBRATED = "CALIBRATED"
PROVENANCE_TIER_ESTIMATED = "ESTIMATED"
PROVENANCE_TIERS = frozenset(
    {
        PROVENANCE_TIER_MEASURED,
        PROVENANCE_TIER_CALIBRATED,
        PROVENANCE_TIER_ESTIMATED,
    }
)

# 指标键 → 默认血统层级（方法学分类；可被 method / 显式字段覆盖）
_DEFAULT_PROVENANCE_TIER_BY_KEY: dict[str, str] = {
    # 纯几何 / 物理像素实测（含 XY-2D Z 坍缩膝角）
    "whipping_velocity": PROVENANCE_TIER_MEASURED,
    "trunk_lean_angle": PROVENANCE_TIER_MEASURED,
    "ball_speed_kmh": PROVENANCE_TIER_MEASURED,
    "launch_angle_deg": PROVENANCE_TIER_MEASURED,
    "impact_knee_angle": PROVENANCE_TIER_MEASURED,
    "max_folding_angle": PROVENANCE_TIER_MEASURED,
    "toe_angle": PROVENANCE_TIER_MEASURED,
    "ankle_rigidity": PROVENANCE_TIER_MEASURED,
    "support_knee_angle": PROVENANCE_TIER_MEASURED,
    # 肩宽基准 / 单应性标定
    "distance_cm": PROVENANCE_TIER_CALIBRATED,
    "support_ratio": PROVENANCE_TIER_CALIBRATED,
    # 直接依赖 MediaPipe 原始 3D（Z）推断
    "hip_torsion_angle": PROVENANCE_TIER_ESTIMATED,
}


def resolve_provenance_tier(
    metric_key: str, entry: Optional[dict] = None
) -> str:
    """为单条指标解析 ``provenance_tier``（MEASURED|CALIBRATED|ESTIMATED）。"""
    item = entry if isinstance(entry, dict) else {}
    for field in ("provenance_tier", "provenanceTier"):
        raw = item.get(field)
        if raw is None:
            continue
        tier = str(raw).strip().upper()
        if tier in PROVENANCE_TIERS:
            return tier

    method = str(item.get("method") or "").strip().lower()
    if any(
        token in method
        for token in (
            "shoulder",
            "homography",
            "calibrat",
            "ratio",
            "pcr",
        )
    ):
        return PROVENANCE_TIER_CALIBRATED
    if any(
        token in method
        for token in (
            "world_z",
            "mediapipe_z",
            "raw_3d",
            "hip_torsion",
            "xz_plane",
            "horizontal_3d",
        )
    ):
        return PROVENANCE_TIER_ESTIMATED
    # 显式 3D 且非 2D/Z-collapse 路径 → 估算
    if "3d" in method and "2d" not in method and "z_collapse" not in method:
        return PROVENANCE_TIER_ESTIMATED

    key = str(metric_key or "").strip()
    if key in _DEFAULT_PROVENANCE_TIER_BY_KEY:
        return _DEFAULT_PROVENANCE_TIER_BY_KEY[key]

    # 未知键：用小写 provenance 回退；否则保守 ESTIMATED
    prov = str(item.get("provenance") or "").strip().lower()
    if prov == PROVENANCE_CALIBRATED:
        return PROVENANCE_TIER_CALIBRATED
    if prov == PROVENANCE_MEASURED:
        return PROVENANCE_TIER_MEASURED
    return PROVENANCE_TIER_ESTIMATED


def apply_provenance_tiers(indicators: Optional[dict]) -> dict:
    """强制为 indicators 中每一条 metric 写入 ``provenance_tier``。"""
    out: dict[str, Any] = dict(indicators or {})
    for metric_key, entry in list(out.items()):
        if not isinstance(entry, dict):
            continue
        tier = resolve_provenance_tier(str(metric_key), entry)
        entry["provenance_tier"] = tier
        out[metric_key] = entry
    return out

# --------------------------------------------------------------------------
# 指标状态等级（Green/Yellow/Red）
# --------------------------------------------------------------------------
STATUS_GREEN = "GREEN_OPTIMAL"
STATUS_YELLOW = "YELLOW_APPROACHING"
STATUS_RED = "RED_DEVIATED"

# 阶段窗口最短安全长度：保证后续 min()/max()/argmin 永不为空序列
MIN_PHASE_WINDOW_FRAMES = 3


def is_aigc_measurable_provenance(provenance: Optional[str]) -> bool:
    """AIGC 可否复述该指标数值（禁止 default/estimated/missing 伪装实测）。"""
    return str(provenance or "").strip().lower() in AIGC_ALLOWED_PROVENANCE


def indicator_scoring_number(entry: Optional[dict], *prefer_keys: str) -> Optional[float]:
    """读取指标数值：优先对外 value，其次 scoring_value / variance（供评分与归档回退）。"""
    if not isinstance(entry, dict):
        return None
    keys = prefer_keys or ("value", "scoring_value", "variance")
    for key in keys:
        raw = entry.get(key)
        if raw is None:
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return float(number)
    return None


def pack_focus_indicator(
    *,
    scoring_value: float,
    provenance: str,
    unit: str,
    status: str,
    penalty: float,
    green_band: list,
    extreme_frame_index: int,
    method: Optional[str] = None,
    confidence: Optional[float] = None,
    decimals: int = 2,
    extra: Optional[dict] = None,
) -> dict:
    """打包焦点指标：仅 measured/calibrated 写入对外 ``value``，评分始终有 ``scoring_value``。"""
    scoring = float(scoring_value)
    if not np.isfinite(scoring):
        scoring = 0.0
    rounded = round(scoring, int(decimals))
    measured = is_aigc_measurable_provenance(provenance)
    payload: dict[str, Any] = {
        "value": rounded if measured else None,
        "scoring_value": rounded,
        "unit": unit,
        "status": status,
        "penalty": penalty,
        "green_band": green_band,
        "extreme_frame_index": int(extreme_frame_index),
        "provenance": str(provenance),
    }
    if method:
        payload["method"] = str(method)
    if confidence is not None and np.isfinite(float(confidence)):
        payload["confidence"] = round(float(confidence), 3)
    if extra:
        payload.update(extra)
    return payload


# --------------------------------------------------------------------------
# 八大量纲计算失败时的安全默认值（禁止 null / NaN 外泄给 LLM / 前端）
# --------------------------------------------------------------------------
_EIGHT_DIMENSION_FALLBACK_VALUES: dict[str, float] = {
    "distance_cm": 17.5,
    "toe_angle": 0.0,
    "max_folding_angle": 80.0,
    "whipping_velocity": 320.0,
    "impact_knee_angle": 150.0,
    "ankle_rigidity": 2.0,  # 形变落差默认 2°（锁踝带内）
    "support_knee_angle": 155.0,
    "hip_torsion_angle": 25.0,
}
_EIGHT_DIMENSION_FALLBACK_UNITS: dict[str, str] = {
    "distance_cm": "cm",
    "toe_angle": "deg",
    "max_folding_angle": "deg",
    "whipping_velocity": "deg/s",
    "impact_knee_angle": "deg",
    "ankle_rigidity": "deg",
    "support_knee_angle": "deg",
    "hip_torsion_angle": "deg",
}

# 【V2.5 科研级】触球核心动作窗口：前后各 30 帧（约 1s@30fps），固定约 60 帧
# 所有极值/方差量纲只准在此窗口内解算，杜绝「300 帧截断 vs 414 帧完整」漂移。
ACTION_ROI_HALF_FRAMES: int = 30


def _metric_value_is_dirty(value: Any) -> bool:
    """None / 非有限浮点 → 视为计算失败，必须回填默认值。"""
    if value is None:
        return True
    try:
        return not bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return True


def sanitize_eight_dimension_indicators(indicators: Optional[dict]) -> dict:
    """组装后强制校验：任一量纲 None/NaN → 合理默认值 + YELLOW_APPROACHING。

    绝对不允许返回给大模型 / 前端的 indicators JSON 含 null 或未计算值。
    """
    out: dict[str, Any] = dict(indicators or {})
    for key, default_val in _EIGHT_DIMENSION_FALLBACK_VALUES.items():
        item = out.get(key)
        if not isinstance(item, dict):
            item = {}

        # provenance 先行确定：它决定 value / variance 是否允许对外暴露数字。
        item.setdefault("provenance", PROVENANCE_DEFAULT)
        measurable = is_aigc_measurable_provenance(item.get("provenance"))

        primary = item.get("value")
        if key == "ankle_rigidity" and _metric_value_is_dirty(primary):
            # 踝刚度对外常落在 variance 字段
            primary = item.get("variance")
        dirty = _metric_value_is_dirty(primary)

        # 非 measured 量纲：value=None 是契约要求的正确状态，不是脏数据。
        # 此时只需保证 scoring_value 等评分侧字段可用，不得回填对外 value。
        if dirty and not measurable:
            decimals = 2
            filled = round(float(default_val), decimals)
            item["value"] = None
            item["scoring_value"] = (
                round(float(item["scoring_value"]), decimals)
                if not _metric_value_is_dirty(item.get("scoring_value"))
                else filled
            )
            item["status"] = item.get("status") or STATUS_YELLOW
            item["unit"] = item.get("unit") or _EIGHT_DIMENSION_FALLBACK_UNITS[key]
            item.setdefault("penalty", 0.0)
            # 【PCR 质量标记传播】若已有来自兜底 PCR 的 method 标记则保留，
            # 否则才填 "fallback_default"；再在 precision_note 里告知下游精度低。
            _pcr_fallback_methods = {"fallback_body_pcr", "fallback_empirical_pcr"}
            existing_method = item.get("method") or ""
            if existing_method not in _pcr_fallback_methods:
                item.setdefault("method", "fallback_default")
            if existing_method in _pcr_fallback_methods or item.get("provenance") == PROVENANCE_ESTIMATED:
                item.setdefault("precision_note", "estimated_value_low_accuracy")
            item.setdefault("extreme_frame_index", 0)
            item.setdefault("confidence", 0.0)
            if key == "ankle_rigidity":
                # variance 是 value 的对外镜像 → 同样保持 None；
                # scoring_variance 是评分侧镜像 → 必须有数。
                item["variance"] = None
                item["scoring_variance"] = float(item["scoring_value"])
                item.setdefault("stiffness_status", ANKLE_STIFFNESS_LOCKED)
                item.setdefault("dorsiflex_drop_deg", None)
                item.setdefault("ankle_angles_window", [])
            gb = item.get("green_band")
            if isinstance(gb, list):
                item["green_band"] = [
                    (0.0 if i == 0 else 99999.0) if v is None else v for i, v in enumerate(gb)
                ]
            out[key] = item
            continue

        if dirty:
            print(
                f"【Warning】DeterministicScorer 量纲 `{key}` 计算失败（None/NaN），"
                f"已回填默认值 {default_val}，状态强制 {STATUS_YELLOW}。"
            )
            decimals = 2
            filled = round(float(default_val), decimals)
            # AIGC 防幻觉契约：先确认 provenance，再决定是否写入 value。
            # provenance 为 default / missing / estimated 时 value 必须保持 None，
            # 仅 measured / calibrated 才允许对外暴露 value。
            item.setdefault("provenance", PROVENANCE_DEFAULT)
            if is_aigc_measurable_provenance(item["provenance"]):
                item["value"] = filled
            # else: item["value"] 保持 None，不写入
            item["scoring_value"] = (
                round(float(item["scoring_value"]), decimals)
                if not _metric_value_is_dirty(item.get("scoring_value"))
                else filled
            )
            item["status"] = STATUS_YELLOW
            item["unit"] = item.get("unit") or _EIGHT_DIMENSION_FALLBACK_UNITS[key]
            item.setdefault("penalty", 0.0)
            item.setdefault("provenance", PROVENANCE_DEFAULT)
            # 【PCR 质量标记传播】measurable dirty 路径同样保留兜底 PCR 来源标记
            _pcr_fallback_methods = {"fallback_body_pcr", "fallback_empirical_pcr"}
            existing_method = item.get("method") or ""
            if existing_method not in _pcr_fallback_methods:
                item.setdefault("method", "fallback_default")
            if existing_method in _pcr_fallback_methods or item.get("provenance") == PROVENANCE_ESTIMATED:
                item.setdefault("precision_note", "estimated_value_low_accuracy")
            item.setdefault("extreme_frame_index", 0)
            item.setdefault("confidence", 0.0)
            if key == "ankle_rigidity":
                item["variance"] = filled
                item["scoring_variance"] = filled
                item["stiffness_status"] = ANKLE_STIFFNESS_LOCKED
                item["dorsiflex_drop_deg"] = 0.0
                item.setdefault("ankle_angles_window", [])

        # 嵌套字段禁止残留 null（此处只处理 measurable 路径；非 measurable 已 continue）
        if key == "ankle_rigidity":
            if _metric_value_is_dirty(item.get("variance")):
                item["variance"] = float(item["value"]) if not _metric_value_is_dirty(item.get("value")) else float(item.get("scoring_value", default_val))
            if _metric_value_is_dirty(item.get("scoring_variance")):
                sv = item.get("scoring_value")
                item["scoring_variance"] = float(sv) if not _metric_value_is_dirty(sv) else float(default_val)
            if item.get("stiffness_status") is None:
                item["stiffness_status"] = ANKLE_STIFFNESS_LOCKED
            if item.get("dorsiflex_drop_deg") is None:
                item["dorsiflex_drop_deg"] = 0.0
            if item.get("ankle_angles_window") is None:
                item["ankle_angles_window"] = []

        gb = item.get("green_band")
        if isinstance(gb, list):
            cleaned_gb = []
            for i, v in enumerate(gb):
                if v is None:
                    cleaned_gb.append(0.0 if i == 0 else 99999.0)
                else:
                    cleaned_gb.append(v)
            item["green_band"] = cleaned_gb

        out[key] = item
    # 全链路血统标签：清洗后强制为每个量纲写入 provenance_tier
    return apply_provenance_tiers(out)

