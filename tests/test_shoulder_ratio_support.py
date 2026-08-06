# -*- coding: utf-8 -*-
"""V3.6 肩宽归一化支撑横距：消除 world 绝对尺度漂移。"""

from __future__ import annotations

from biomech_primitives import (
    AVERAGE_CHILD_SHOULDER_WIDTH_CM,
    calculate_support_offset_by_shoulder_ratio,
)
from deterministic_scorer import calculate_biomechanical_score


def test_shoulder_ratio_cancels_world_scale_inflation():
    """尺度放大 2.5× 后，比例与估计 cm 应保持不变。"""
    # 真值：横距 0.20m，肩宽 0.30m → ratio=0.667 → est≈20cm
    base = calculate_support_offset_by_shoulder_ratio(
        support_ankle=(0.20, 0.0, 0.0),
        ball_center=(0.0, 0.0, 0.0),
        left_shoulder=(-0.15, 0.0, 0.0),
        right_shoulder=(0.15, 0.0, 0.0),
    )
    assert base["ok"] is True
    assert abs(float(base["support_ratio"]) - (0.20 / 0.30)) < 1e-3
    assert abs(float(base["distance_cm_estimate"]) - 20.0) < 0.5

    # MediaPipe 尺度漂 2.5×（本次事故同型）
    k = 2.5
    inflated = calculate_support_offset_by_shoulder_ratio(
        support_ankle=(0.20 * k, 0.0, 0.0),
        ball_center=(0.0, 0.0, 0.0),
        left_shoulder=(-0.15 * k, 0.0, 0.0),
        right_shoulder=(0.15 * k, 0.0, 0.0),
    )
    assert inflated["ok"] is True
    assert abs(float(inflated["support_ratio"]) - float(base["support_ratio"])) < 1e-3
    assert abs(
        float(inflated["distance_cm_estimate"]) - float(base["distance_cm_estimate"])
    ) < 0.5
    # 裸 ×100 会得到 50cm；归一化后必须仍≈20
    assert float(inflated["distance_cm_estimate"]) < 30.0


def test_scorer_uses_shoulder_ratio_not_bare_world_cm():
    """上游若只给漂移的绝对 cm，Scorer 应从帧内肩宽重算，不得当 measured 满扣。"""
    # world：ΔX=0.508（裸×100=50.8），肩宽=0.76 → ratio≈0.67 → ~20cm
    impact_rec = {
        "left_ankle": [100.0, 200.0, 0.0],
        "right_foot_index": [120.0, 200.0, 0.0],
        "left_shoulder": [90.0, 80.0, 0.0],
        "right_shoulder": [150.0, 80.0, 0.0],
        "left_hip": [100.0, 140.0, 0.0],
        "left_knee": [100.0, 170.0, 0.0],
        "right_hip": [130.0, 140.0, 0.0],
        "right_knee": [130.0, 170.0, 0.0],
        "right_ankle": [130.0, 200.0, 0.0],
        "world": {
            "left_ankle": [0.508, 0.0, 0.0],
            "right_foot_index": [0.0, 0.0, 0.0],
            "right_ankle": [0.05, 0.0, 0.0],
            "left_shoulder": [-0.38, 0.0, 0.0],
            "right_shoulder": [0.38, 0.0, 0.0],
        },
    }
    impact = {
        "t_impact": 0,
        "frames": [impact_rec],
        # 故意注入旧版漂移 cm，验证不再被当成 measured 红灯
        "support_lateral_dist_cm": 50.82,
        "support_distance_method": "legacy_world_x100",
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "ankle_angles_window": [140.0, 140.1, 140.0],
    }
    trajectory = {"max_folding_angle": 70.0, "whipping_velocity": 500.0}
    score, detail = calculate_biomechanical_score(impact, trajectory)
    dist = detail["indicators"]["distance_cm"]
    assert dist["method"] == "shoulder_width_ratio"
    assert dist["unit"] == "ratio"
    assert dist["status"] == "GREEN_OPTIMAL"
    assert float(dist["scoring_value"]) < 1.0  # ~0.67
    assert float(dist.get("support_ratio") or dist["scoring_value"]) < 1.0
    assert score >= 90.0


def test_calculate_support_ratio_image_plane():
    """图像平面算子：support_dist_px / shoulder_width_px。"""
    from biomech_primitives import calculate_support_ratio

    landmarks = {
        "left_shoulder": (100.0, 80.0),
        "right_shoulder": (160.0, 80.0),  # 肩宽 60px
        "left_ankle": (70.0, 200.0),
        "right_ankle": (140.0, 200.0),
    }
    # 支撑左踝 x=70，球心 x=100 → |Δx|=30 → ratio=30/60=0.5（绿带）
    detail = calculate_support_ratio(landmarks, (100.0, 210.0), support_ankle_key="left_ankle")
    assert detail["ok"] is True
    assert abs(float(detail["support_ratio"]) - 0.5) < 1e-3
    assert abs(float(detail["shoulder_width_px"]) - 60.0) < 1e-6
    assert abs(float(detail["support_dist_px"]) - 30.0) < 1e-6


def test_ratio_bands_red_above_0_9():
    impact_rec = {
        "world": {
            "left_ankle": [0.60, 0.0, 0.0],
            "right_foot_index": [0.0, 0.0, 0.0],
            "left_shoulder": [-0.15, 0.0, 0.0],
            "right_shoulder": [0.15, 0.0, 0.0],  # 肩宽 0.30 → ratio=2.0
        },
        "left_ankle": [0, 0, 0],
        "right_foot_index": [0, 0, 0],
        "left_shoulder": [0, 0, 0],
        "right_shoulder": [1, 0, 0],
        "left_hip": [0, 0, 0],
        "left_knee": [0, 0, 0],
        "right_hip": [0, 0, 0],
        "right_knee": [0, 0, 0],
        "right_ankle": [0, 0, 0],
    }
    impact = {
        "t_impact": 0,
        "frames": [impact_rec],
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "ankle_angles_window": [140.0, 140.1, 140.0],
    }
    trajectory = {"max_folding_angle": 70.0, "whipping_velocity": 500.0}
    _, detail = calculate_biomechanical_score(impact, trajectory)
    dist = detail["indicators"]["distance_cm"]
    assert dist["status"] == "RED_DEVIATED"
    assert float(dist["scoring_value"]) > 0.9
    assert dist["unit"] == "ratio"
    reasons = " ".join(d["reason"] for d in detail.get("deductions") or [])
    assert "支撑脚横距比例" in reasons


def test_foot_proxy_ignores_anterior_stride_inflation():
    """摆动脚尖代理球心时：前后步幅大、体侧横距正常 → 不得报 2×肩宽夸张站位。

    复现现场事故：水平斜距 ≈65cm（ratio≈2.16），但体侧分量仅 ≈20cm（ratio≈0.67）。
    """
    # 肩宽轴沿 X；支撑踝相对摆动足：ΔX=0.20（横距），ΔZ=0.60（前后步幅）
    # 旧算法 hypot → ratio≈2.16；新算法沿肩宽投影 → ratio≈0.67
    base = calculate_support_offset_by_shoulder_ratio(
        support_ankle=(0.20, 0.0, 0.60),
        ball_center=(0.0, 0.0, 0.0),
        left_shoulder=(-0.15, 0.0, 0.0),
        right_shoulder=(0.15, 0.0, 0.0),
        distance_mode="lateral",
    )
    assert base["ok"] is True
    assert abs(float(base["support_ratio"]) - (0.20 / 0.30)) < 0.05
    assert float(base["support_ratio"]) < 1.0
    assert float(base["distance_cm_estimate"]) < 30.0
    # 全水平斜距仍很大，但不得用于站位评分
    assert float(base["horizontal_distance"]) > float(base["lateral_distance"]) * 2.0

    impact_rec = {
        "world": {
            "left_ankle": [0.20, 0.0, 0.60],
            "right_foot_index": [0.0, 0.0, 0.0],
            "right_ankle": [0.02, 0.0, 0.02],
            "left_shoulder": [-0.15, 0.0, 0.0],
            "right_shoulder": [0.15, 0.0, 0.0],
        },
        "left_ankle": [0, 0, 0],
        "right_foot_index": [0, 0, 0],
        "left_shoulder": [0, 0, 0],
        "right_shoulder": [1, 0, 0],
        "left_hip": [0, 0, 0],
        "left_knee": [0, 0, 0],
        "right_hip": [0, 0, 0],
        "right_knee": [0, 0, 0],
        "right_ankle": [0, 0, 0],
    }
    impact = {
        "t_impact": 0,
        "frames": [impact_rec],
        "swing_leg": "right",
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "ankle_angles_window": [140.0, 140.1, 140.0],
    }
    trajectory = {
        "max_folding_angle": 70.0,
        "whipping_velocity": 500.0,
        "swing_leg": "right",
    }
    _, detail = calculate_biomechanical_score(impact, trajectory)
    dist = detail["indicators"]["distance_cm"]
    assert dist["method"] == "shoulder_width_ratio"
    assert float(dist["scoring_value"]) < 1.0
    assert dist["status"] != "RED_DEVIATED"
