# -*- coding: utf-8 -*-
"""empirical_thresholds.py — Phase 4 经验阈值配置（默认同产线，可 JSON 覆盖）。

【防幻觉】无伦理审批与足量五年级样本前，禁止擅自改动绿带中心。
本模块仅提供可复现的阈值读取；默认值与现网 DeterministicScorer / 原语一致。
可选文件：项目根目录 ``empirical_thresholds.json``。
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_THRESHOLDS_PATH = os.path.join(SCRIPT_DIR, "empirical_thresholds.json")

# 与现网评分绿带 / 踝分档完全同源的默认值（勿凭空改动）
DEFAULT_THRESHOLDS: dict[str, Any] = {
    "schema_version": 1,
    "population": "production_defaults_v31",
    "notes": (
        "五年级样本经验校准需伦理与样本量达标后方可替换本文件；"
        "未替换前一律使用 production_defaults_v31。"
    ),
    "support_foot_distance_cm": {
        "green_low": 15.0,
        "green_high": 20.0,
        "yellow_low": 10.0,
        "yellow_high": 25.0,
        "ideal_center": 17.5,
    },
    "max_folding_angle_deg": {
        "green_low": 70.0,
        "green_high": 90.0,
        "yellow_low": 55.0,
        "yellow_high": 105.0,
        "ideal_center": 80.0,
    },
    "ankle_rigidity_variance": {
        # 单位：无量纲角方差 σ²（度²口径下的样本方差）——前端与 AIGC 统一标注 "variance"
        "locked_max": 2.0,
        "slight_max": 5.0,
    },
    # ── 以下区块自 error_diagnoser.DeterministicScorer 函数内联字面量迁移而来 ──
    # 依据: 教研标准化底库「支撑脚尖对准前方目标」判定口径（原 _score 内联 15/25）
    # 校准日期: 2026-07-31 | 数据集: production_defaults_v31 | 状态: 未经五年级样本重标定
    "toe_angle_deg": {
        "green_high": 15.0,  # ≤15° 全绿（脚尖基本对准出球方向）
        "red_low": 25.0,  # >25° 直接扣满（脚尖明显外展/后撤）
    },
    # 依据: 小腿鞭打峰值角速度经验档（原 _score 内联 450/320，黄带线性系数 0.55）
    # 校准日期: 2026-07-31 | 数据集: production_defaults_v31 | 备注: 与雷达满分线同源
    "whipping_velocity_deg_s": {
        "green_low": 450.0,  # ≥450°/s 视为充分鞭打
        "yellow_low": 320.0,  # [320,450) 线性递增惩罚；<320 判红
        "yellow_penalty_ratio": 0.55,  # 黄带最大惩罚占该维满分比例
    },
    # 依据: 触球瞬间摆动腿伸展态经验区间（原 _linear_band_penalty 内联 140/160/125/175）
    # 校准日期: 2026-07-31 | 数据集: production_defaults_v31
    "impact_knee_angle_deg": {
        "green_low": 140.0,
        "green_high": 160.0,
        "yellow_low": 125.0,
        "yellow_high": 175.0,
        "fallback": 150.0,  # 解算异常时的中性兜底（provenance 必须标 default/missing）
    },
    # 依据: 支撑腿微屈缓冲经验区间（原 _linear_band_penalty 内联 140/165/125/175）
    # 校准日期: 2026-07-31 | 数据集: production_defaults_v31
    "support_knee_angle_deg": {
        "green_low": 140.0,
        "green_high": 165.0,
        "yellow_low": 125.0,
        "yellow_high": 175.0,
        "fallback": 155.0,
    },
    # 依据: 髋关节相对扭转（转髋充分度）经验区间（原内联 15/40/5/55）
    # 校准日期: 2026-07-31 | 数据集: production_defaults_v31
    "hip_torsion_angle_deg": {
        "green_low": 15.0,
        "green_high": 40.0,
        "yellow_low": 5.0,
        "yellow_high": 55.0,
        "fallback": 25.0,
    },
    # 依据: ERR_B2「只用小腿弹射」判定上界（原 _detect_error_code 内联 165.0）
    # 校准日期: 2026-07-31 | 数据集: production_defaults_v31
    # 语义: fold < 该值 且 大腿后伸≈0° → 认定髋未参与发力
    "shank_only_fold_max_deg": 165.0,
    # 依据: V3.1 五维雷达各维满分 20 的离散/线性映射（原 _compose_radar_scores 内联）
    # 校准日期: 2026-07-31 | 数据集: production_defaults_v31
    "radar": {
        "ankle_locked_score": 20.0,  # σ² < locked_max
        "ankle_slight_score": 15.0,  # locked_max ≤ σ² ≤ slight_max
        "ankle_yielding_score": 5.0,  # σ² > slight_max
        "whipping_full_score_deg_s": 450.0,  # 与 whipping_velocity_deg_s.green_low 同源
        "approach_rhythm_floor": 16.0,  # 助跑占位维鼓励性保底
        "approach_rhythm_ceiling": 20.0,
        "approach_penalty_slope": 0.05,  # 每 1 分总惩罚下拉的雷达分
    },
    "ankle_impact_half_window_ms": 50.0,
    "pcr_mae_max_cm": 2.0,
    "fold_mae_max_deg": 5.0,
}


_cached: Optional[dict[str, Any]] = None
_cached_path: Optional[str] = None


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_empirical_thresholds(
    path: Optional[str] = None,
    *,
    force_reload: bool = False,
) -> dict[str, Any]:
    """读取经验阈值；文件缺失或损坏时回退默认（绝不抛崩）。"""
    global _cached, _cached_path
    target = os.path.abspath(path or DEFAULT_THRESHOLDS_PATH)
    if (
        not force_reload
        and _cached is not None
        and _cached_path == target
    ):
        return deepcopy(_cached)

    merged = deepcopy(DEFAULT_THRESHOLDS)
    if os.path.isfile(target):
        try:
            with open(target, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                merged = _deep_merge(merged, raw)
                merged["loaded_from"] = target
            else:
                merged["loaded_from"] = None
                merged["load_warning"] = "empirical_thresholds.json 根节点非 object"
        except Exception as exc:  # noqa: BLE001
            merged["loaded_from"] = None
            merged["load_warning"] = f"读取失败，已回退默认：{exc}"
    else:
        merged["loaded_from"] = None

    _cached = merged
    _cached_path = target
    return deepcopy(merged)


def get_support_distance_green_band(path: Optional[str] = None) -> tuple[float, float]:
    cfg = load_empirical_thresholds(path)
    block = cfg.get("support_foot_distance_cm") or {}
    return float(block.get("green_low", 15.0)), float(block.get("green_high", 20.0))


def get_folding_green_band(path: Optional[str] = None) -> tuple[float, float]:
    cfg = load_empirical_thresholds(path)
    block = cfg.get("max_folding_angle_deg") or {}
    return float(block.get("green_low", 70.0)), float(block.get("green_high", 90.0))


def get_support_distance_bands(
    path: Optional[str] = None,
) -> tuple[float, float, float, float, float]:
    """返回 (green_low, green_high, yellow_low, yellow_high, ideal_center)。"""
    cfg = load_empirical_thresholds(path)
    block = cfg.get("support_foot_distance_cm") or {}
    return (
        float(block.get("green_low", 15.0)),
        float(block.get("green_high", 20.0)),
        float(block.get("yellow_low", 10.0)),
        float(block.get("yellow_high", 25.0)),
        float(block.get("ideal_center", 17.5)),
    )


def get_folding_bands(path: Optional[str] = None) -> tuple[float, float, float, float, float]:
    """返回 (green_low, green_high, yellow_low, yellow_high, ideal_center)。"""
    cfg = load_empirical_thresholds(path)
    block = cfg.get("max_folding_angle_deg") or {}
    return (
        float(block.get("green_low", 70.0)),
        float(block.get("green_high", 90.0)),
        float(block.get("yellow_low", 55.0)),
        float(block.get("yellow_high", 105.0)),
        float(block.get("ideal_center", 80.0)),
    )


def get_ankle_variance_gates(path: Optional[str] = None) -> tuple[float, float]:
    cfg = load_empirical_thresholds(path)
    block = cfg.get("ankle_rigidity_variance") or {}
    return float(block.get("locked_max", 2.0)), float(block.get("slight_max", 5.0))


def get_ankle_half_window_ms(path: Optional[str] = None) -> float:
    cfg = load_empirical_thresholds(path)
    return float(cfg.get("ankle_impact_half_window_ms", 50.0))


def get_toe_angle_thresholds(path: Optional[str] = None) -> tuple[float, float]:
    """返回 (green_high, red_low)：toe_angle ≤ green_high → 绿，> red_low → 红。"""
    cfg = load_empirical_thresholds(path)
    block = cfg.get("toe_angle_deg") or {}
    return float(block.get("green_high", 15.0)), float(block.get("red_low", 25.0))


def get_whipping_thresholds(path: Optional[str] = None) -> tuple[float, float, float]:
    """返回 (green_low, yellow_low, yellow_penalty_ratio)。"""
    cfg = load_empirical_thresholds(path)
    block = cfg.get("whipping_velocity_deg_s") or {}
    return (
        float(block.get("green_low", 450.0)),
        float(block.get("yellow_low", 320.0)),
        float(block.get("yellow_penalty_ratio", 0.55)),
    )


def get_impact_knee_thresholds(path: Optional[str] = None) -> tuple[float, float, float, float, float]:
    """返回 (green_low, green_high, yellow_low, yellow_high, fallback)。"""
    cfg = load_empirical_thresholds(path)
    block = cfg.get("impact_knee_angle_deg") or {}
    return (
        float(block.get("green_low", 140.0)),
        float(block.get("green_high", 160.0)),
        float(block.get("yellow_low", 125.0)),
        float(block.get("yellow_high", 175.0)),
        float(block.get("fallback", 150.0)),
    )


def get_support_knee_thresholds(path: Optional[str] = None) -> tuple[float, float, float, float, float]:
    """返回 (green_low, green_high, yellow_low, yellow_high, fallback)。"""
    cfg = load_empirical_thresholds(path)
    block = cfg.get("support_knee_angle_deg") or {}
    return (
        float(block.get("green_low", 140.0)),
        float(block.get("green_high", 165.0)),
        float(block.get("yellow_low", 125.0)),
        float(block.get("yellow_high", 175.0)),
        float(block.get("fallback", 155.0)),
    )


def get_hip_torsion_thresholds(path: Optional[str] = None) -> tuple[float, float, float, float, float]:
    """返回 (green_low, green_high, yellow_low, yellow_high, fallback)。"""
    cfg = load_empirical_thresholds(path)
    block = cfg.get("hip_torsion_angle_deg") or {}
    return (
        float(block.get("green_low", 15.0)),
        float(block.get("green_high", 40.0)),
        float(block.get("yellow_low", 5.0)),
        float(block.get("yellow_high", 55.0)),
        float(block.get("fallback", 25.0)),
    )


def get_shank_only_fold_max(path: Optional[str] = None) -> float:
    """ERR_B2 判定上界：fold < 该值 且 大腿后伸≈0° → 小腿弹射。"""
    cfg = load_empirical_thresholds(path)
    return float(cfg.get("shank_only_fold_max_deg", 165.0))


def get_radar_config(path: Optional[str] = None) -> dict:
    """五维雷达映射参数；返回完整 radar 子块（含默认值）。"""
    cfg = load_empirical_thresholds(path)
    block = cfg.get("radar") or {}
    return {
        "ankle_locked_score": float(block.get("ankle_locked_score", 20.0)),
        "ankle_slight_score": float(block.get("ankle_slight_score", 15.0)),
        "ankle_yielding_score": float(block.get("ankle_yielding_score", 5.0)),
        "whipping_full_score_deg_s": float(block.get("whipping_full_score_deg_s", 450.0)),
        "approach_rhythm_floor": float(block.get("approach_rhythm_floor", 16.0)),
        "approach_rhythm_ceiling": float(block.get("approach_rhythm_ceiling", 20.0)),
        "approach_penalty_slope": float(block.get("approach_penalty_slope", 0.05)),
    }


def assert_defaults_match_production() -> None:
    """单元测试用：默认绿带必须与现网常数一致。"""
    assert DEFAULT_THRESHOLDS["support_foot_distance_cm"]["green_low"] == 15.0
    assert DEFAULT_THRESHOLDS["support_foot_distance_cm"]["green_high"] == 20.0
    assert DEFAULT_THRESHOLDS["max_folding_angle_deg"]["green_low"] == 70.0
    assert DEFAULT_THRESHOLDS["max_folding_angle_deg"]["green_high"] == 90.0
    assert DEFAULT_THRESHOLDS["ankle_rigidity_variance"]["locked_max"] == 2.0
    assert DEFAULT_THRESHOLDS["ankle_rigidity_variance"]["slight_max"] == 5.0
