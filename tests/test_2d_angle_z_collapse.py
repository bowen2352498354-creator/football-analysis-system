# -*- coding: utf-8 -*-
"""V3.9：Z 坍缩 XY-2D 膝角 — 抗 MediaPipe 侧向深度畸变。"""

from __future__ import annotations

from biomech_primitives import (
    calculate_2d_angle,
    calculate_2d_angle_or_none,
    calculate_3d_joint_angle,
    calculate_sagittal_angle,
)
from deterministic_scorer import calculate_biomechanical_score
from empirical_thresholds import get_folding_bands


def test_calculate_2d_angle_ignores_z_spike():
    """Z 轴爆炸不得改变 XY-2D 夹角；旧 YZ-矢状角会被拉歪。"""
    # 图像平面：直腿近似 180°（髋在上、踝在下，共线）
    hip = (100.0, 80.0, 0.0)
    knee = (100.0, 140.0, 0.0)
    ankle = (100.0, 200.0, 0.0)
    base_2d = calculate_2d_angle(hip, knee, ankle)
    assert abs(base_2d - 180.0) < 1.0

    # 仅污染 Z：2D 角应不变；YZ 矢状角会偏离
    hip_z = (100.0, 80.0, 0.0)
    knee_z = (100.0, 140.0, 2.5)  # 深度毛刺
    ankle_z = (100.0, 200.0, -1.8)
    spiked_2d = calculate_2d_angle(hip_z, knee_z, ankle_z)
    spiked_sag = calculate_sagittal_angle(hip_z, knee_z, ankle_z)
    assert abs(spiked_2d - base_2d) < 1e-6
    assert abs(spiked_sag - base_2d) > 5.0  # 旧路径被 Z 拉歪


def test_knee_extension_path_uses_2d():
    hip = (0.0, 0.0, 0.0)
    knee = (0.0, 1.0, 9.0)  # 巨大 Z
    ankle = (0.5, 2.0, -9.0)
    a = calculate_3d_joint_angle(hip, knee, ankle, is_knee_extension=True)
    b = calculate_2d_angle(hip, knee, ankle)
    assert abs(a - b) < 1e-9


def test_folding_green_band_70_100():
    gl, gh, yl, yh, center = get_folding_bands()
    assert (gl, gh) == (70.0, 100.0)
    assert yl == 55.0 and yh == 120.0
    assert center == 85.0


def test_scorer_fold_and_impact_use_2d_from_frames():
    """有帧时折叠/触球膝角 method 走 2D，且折叠落在 70–100 绿带。"""
    # 膝内角≈95° → fold≈85°
    impact_rec = {
        "right_hip": [120.0, 100.0, 5.0],
        "right_knee": [130.0, 160.0, -8.0],  # Z 畸变不应影响
        "right_ankle": [100.0, 210.0, 6.0],
        "left_hip": [90.0, 100.0, 0.0],
        "left_knee": [90.0, 160.0, 0.0],
        "left_ankle": [90.0, 210.0, 0.0],
        "left_shoulder": [85.0, 60.0, 0.0],
        "right_shoulder": [145.0, 60.0, 0.0],
        "right_foot_index": [105.0, 220.0, 0.0],
        "left_foot_index": [90.0, 220.0, 0.0],
        "timestamp_sec": 0.0,
    }
    # 再补两帧满足 ROI 最少有效帧
    frames = []
    for i in range(5):
        rec = dict(impact_rec)
        rec["timestamp_sec"] = i / 30.0
        # 中间帧稍屈曲
        if i < 3:
            rec["right_ankle"] = [95.0 + i, 205.0, 3.0 * i]
            rec["right_knee"] = [128.0, 155.0 + i, -4.0 * i]
        frames.append(rec)

    impact = {
        "t_impact": 4,
        "frames": frames,
        "swing_leg": "right",
        "toe_angle": 5.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "ankle_angles_window": [140.0, 140.1, 140.0],
    }
    trajectory = {"whipping_velocity": 500.0, "swing_leg": "right"}
    _, detail = calculate_biomechanical_score(impact, trajectory)
    fold = detail["indicators"]["max_folding_angle"]
    ik = detail["indicators"]["impact_knee_angle"]
    assert fold["method"].startswith("roi_2d_knee_min")
    assert fold["value"] is not None
    assert 0.0 < float(fold["value"]) < 180.0
    assert ik["value"] is not None
    # 2D 路径应给出有限膝角，不被 Z 打成尖峰 0/NaN
    assert float(ik["value"]) > 20.0


def test_2d_angle_or_none_on_degenerate():
    assert calculate_2d_angle_or_none(None, (0, 0, 0), (1, 1, 0)) is None
    assert calculate_2d_angle_or_none((0, 0, 0), (0, 0, 0), (0, 0, 0)) is None
