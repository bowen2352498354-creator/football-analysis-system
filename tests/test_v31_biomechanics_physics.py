# -*- coding: utf-8 -*-
"""V3.1 核心生物力学实测原语：3D 折叠角 / PCR 横距 / 踝刚度方差。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from error_diagnoser import (
    ANKLE_STIFFNESS_LOCKED,
    ANKLE_STIFFNESS_SLIGHT_DEFORMATION,
    ANKLE_STIFFNESS_YIELDING,
    DEFAULT_EMPIRICAL_PCR,
    STANDARD_BALL_DIAMETER_CM,
    calculate_3d_joint_angle,
    calculate_ankle_stiffness_variance,
    calculate_support_foot_offset_cm,
)
from biomech_primitives import calculate_support_foot_offset_detailed
from pose_tracker import (
    calculate_3d_joint_angle as pt_calculate_3d_joint_angle,
    calculate_ankle_stiffness_variance as pt_calculate_ankle_stiffness_variance,
    calculate_support_foot_offset_cm as pt_calculate_support_foot_offset_cm,
)


def test_calculate_3d_joint_angle_right_angle():
    """正交向量 → 90°（纯 3D 点乘，非 arctan2）。"""
    p1 = (0.0, 1.0, 0.0)
    p2 = (0.0, 0.0, 0.0)
    p3 = (1.0, 0.0, 0.0)
    ang = calculate_3d_joint_angle(p1, p2, p3)
    assert abs(ang - 90.0) < 1e-6
    assert abs(pt_calculate_3d_joint_angle(p1, p2, p3) - 90.0) < 1e-6


def test_calculate_3d_joint_angle_rejects_perspective_flat_collapse():
    """含深度的真实夹角 ≠ 仅投影到 XY 的 2D 角。"""
    p1 = (1.0, 1.0, 1.0)
    p2 = (0.0, 0.0, 0.0)
    p3 = (1.0, 0.0, 0.0)
    ang_3d = calculate_3d_joint_angle(p1, p2, p3)
    ba_2d = np.array([1.0, 1.0], dtype=np.float64)
    bc_2d = np.array([1.0, 0.0], dtype=np.float64)
    cos_2d = float(np.dot(ba_2d, bc_2d) / (np.linalg.norm(ba_2d) * np.linalg.norm(bc_2d)))
    ang_2d = float(np.degrees(np.arccos(np.clip(cos_2d, -1.0, 1.0))))
    assert abs(ang_2d - 45.0) < 1e-6
    assert abs(ang_3d - math.degrees(math.acos(1.0 / math.sqrt(3.0)))) < 1e-6
    assert abs(ang_3d - ang_2d) > 5.0


def test_calculate_3d_joint_angle_depth_changes_result():
    """同一 XY、不同 Z → 3D 角必须变化（证明用了 z）。"""
    p1 = (0.0, 1.0, 1.0)
    p2 = (0.0, 0.0, 0.0)
    p3_a = (1.0, 0.0, 0.0)
    p3_b = (1.0, 0.0, 2.0)
    a0 = calculate_3d_joint_angle(p1, p2, p3_a)
    a1 = calculate_3d_joint_angle(p1, p2, p3_b)
    assert abs(a0 - a1) > 5.0


def test_calculate_3d_joint_angle_zero_vector_safe():
    assert calculate_3d_joint_angle((0, 0, 0), (0, 0, 0), (1, 0, 0)) == 0.0
    assert calculate_3d_joint_angle((1, 0, 0), (0, 0, 0), (0, 0, 0)) == 0.0


def test_calculate_3d_joint_angle_collinear_180():
    p1 = (-1.0, 0.0, 0.0)
    p2 = (0.0, 0.0, 0.0)
    p3 = (2.0, 0.0, 0.0)
    assert abs(calculate_3d_joint_angle(p1, p2, p3) - 180.0) < 1e-6


def test_support_foot_offset_pcr_exact():
    """球径 84px → PCR=0.25；踝距球心 68px → 17.0 cm。"""
    ball_bbox = [100.0, 200.0, 184.0, 284.0]  # 84×84
    # 球心 x = 142；踝 x = 142 + 68 = 210
    ankle = (210.0, 250.0)
    offset = calculate_support_foot_offset_cm(ankle, ball_bbox)
    assert abs(offset - 17.0) < 1e-9
    assert abs(pt_calculate_support_foot_offset_cm(ankle, ball_bbox) - 17.0) < 1e-9


def test_support_foot_offset_uses_max_side_against_blur():
    """宽高取 max，防止运动模糊单边缩水低估直径。"""
    # 宽 40（模糊缩水）、高 80 → diameter=80，PCR=21/80
    ball_bbox = [0.0, 0.0, 40.0, 80.0]
    ball_center_x = 20.0
    ankle = (20.0 + 40.0, 40.0)  # delta_x = 40
    offset = calculate_support_foot_offset_cm(ankle, ball_bbox)
    expected = 40.0 * (STANDARD_BALL_DIAMETER_CM / 80.0)
    assert abs(offset - expected) < 1e-9


def test_support_foot_offset_missing_ball_uses_empirical_pcr():
    """球未检出：无球心 → 横距 0；非法 bbox 同理。"""
    offset = calculate_support_foot_offset_cm((100.0, 50.0), None)
    assert offset == 0.0
    # 非法 bbox 同样无法取球心
    offset2 = calculate_support_foot_offset_cm((100.0, 50.0), [1.0, 2.0])  # 不足 4 维
    assert offset2 == 0.0
    assert DEFAULT_EMPIRICAL_PCR == pytest.approx(STANDARD_BALL_DIAMETER_CM / 84.0)


def test_support_foot_offset_rejects_tiny_bbox_uses_fallback_and_clamp():
    """max(w,h)<10 严禁用球框比例尺；fallback + 钳制防横距爆炸。"""
    # 极小框（遮挡）：若误用 PCR=21/5=4.2，Δx=40 → 168cm 爆炸
    tiny = [100.0, 100.0, 105.0, 105.0]  # 5×5，中心 102.5
    ankle = (142.5, 102.5)  # Δx=40
    # 无身高：经验 PCR → 40*(21/84)=10.0
    offset = calculate_support_foot_offset_cm(ankle, tiny)
    assert abs(offset - 10.0) < 1e-9
    # 身高 fallback：body_h=290px → PCR=0.5 → 20cm
    offset_body = calculate_support_foot_offset_cm(ankle, tiny, body_h_px=290.0)
    assert abs(offset_body - 20.0) < 1e-9
    detail = calculate_support_foot_offset_detailed(ankle, tiny, body_h_px=290.0)
    assert detail["ok"] is False
    assert detail["method"] == "fallback_body_pcr"
    # 钳制：超大像素距 × 经验 PCR 不得 > 60
    far_ankle = (102.5 + 400.0, 102.5)  # 400*0.25=100 → clamp 60
    assert calculate_support_foot_offset_cm(far_ankle, tiny) == 60.0


def test_sagittal_knee_angle_stable_near_extension():
    """矢状面 atan2：近 180° 直腿在 X 轴抖动下不翻成锐角。"""
    from biomech_primitives import calculate_sagittal_angle

    knee = (0.0, 0.0, 0.0)
    # 矢状面近乎伸直，但左右（X）同向偏移——3D arccos 会压到 ~118°
    hip = (0.3, 0.5, -0.05)
    ankle = (0.3, -0.5, 0.05)
    sag = calculate_3d_joint_angle(hip, knee, ankle, is_knee_extension=True)
    raw_3d = calculate_3d_joint_angle(hip, knee, ankle, is_knee_extension=False)
    assert sag > 170.0
    assert raw_3d < 130.0

    # 无 X 噪声的直腿基准仍 ≈ 180°
    clean = calculate_sagittal_angle((0.0, 0.5, -0.05), knee, (0.0, -0.5, 0.05))
    assert clean > 170.0
    assert abs(sag - clean) < 5.0

    # 矢状面真实屈曲（Y-Z 上约 90°）保持锐角，不再做 180-angle 误补
    flexed = calculate_3d_joint_angle(
        (0.0, 1.0, 0.0), knee, (0.0, 0.0, 1.0), is_knee_extension=True
    )
    assert abs(flexed - 90.0) < 1.0

    signed = calculate_sagittal_angle(hip, knee, ankle, signed=True)
    assert abs(abs(signed) - sag) < 1e-6


def test_ankle_stiffness_rejects_visibility_jump_and_rounds():
    """低可见度/空值跳变帧不得拉爆方差；结果 round(..., 2)。"""
    # 中帧塌缩到 0，若计入 → var 极大（≈299 量级）
    series = [140.0, 0.0, 140.2]
    vis = [0.9, 0.1, 0.9]
    var, status = calculate_ankle_stiffness_variance(
        series, t_impact_index=1, landmark_visibility_series=vis
    )
    assert status == ANKLE_STIFFNESS_LOCKED
    assert var < 2.0
    assert var == round(var, 2)
    # 无 visibility 时，0 值本身也视为跳变剔除
    var2, status2 = calculate_ankle_stiffness_variance(series, t_impact_index=1)
    assert status2 == ANKLE_STIFFNESS_LOCKED
    assert var2 < 2.0


def test_ankle_stiffness_locked():
    series = [140.0, 140.2, 140.1, 139.9, 140.0]
    var, status = calculate_ankle_stiffness_variance(series, t_impact_index=2)
    assert status == ANKLE_STIFFNESS_LOCKED
    assert var < 2.0
    var2, status2 = pt_calculate_ankle_stiffness_variance(series, 2)
    assert status2 == ANKLE_STIFFNESS_LOCKED
    assert abs(var - var2) < 1e-12


def test_ankle_stiffness_slight_deformation():
    # 持续渐变（非单帧尖峰）：中值滤波后冲击窗方差仍落在 [2, 5]
    series = [128.0, 130.0, 133.0, 136.0, 134.0]
    var, status = calculate_ankle_stiffness_variance(series, t_impact_index=2)
    assert 2.0 <= var <= 5.0
    assert status == ANKLE_STIFFNESS_SLIGHT_DEFORMATION


def test_ankle_stiffness_yielding():
    # 大幅持续波动，中值后仍 > 5
    series = [100.0, 110.0, 130.0, 95.0, 105.0]
    var, status = calculate_ankle_stiffness_variance(series, t_impact_index=2)
    assert var > 5.0
    assert status == ANKLE_STIFFNESS_YIELDING


def test_ankle_stiffness_median_rejects_single_frame_spike():
    """单帧极值噪点不得拉爆方差；中值滤波后应仍 LOCKED。"""
    series = [140.0, 140.1, 200.0, 139.9, 140.0]
    var, status = calculate_ankle_stiffness_variance(series, t_impact_index=2)
    assert status == ANKLE_STIFFNESS_LOCKED
    assert var < 2.0
    # 空数组安全
    assert calculate_ankle_stiffness_variance([], 0)[0] == 0.0
    assert calculate_ankle_stiffness_variance(None, 0)[0] == 0.0


def test_ankle_stiffness_boundary_safe_truncate():
    """索引越界安全截断，不抛异常。"""
    series = [140.0, 141.0]
    var, status = calculate_ankle_stiffness_variance(series, t_impact_index=0)
    assert math.isfinite(var)
    assert status in (
        ANKLE_STIFFNESS_LOCKED,
        ANKLE_STIFFNESS_SLIGHT_DEFORMATION,
        ANKLE_STIFFNESS_YIELDING,
    )
    var_end, _ = calculate_ankle_stiffness_variance(series, t_impact_index=99)
    assert math.isfinite(var_end)


def test_ankle_stiffness_empty_series():
    var, status = calculate_ankle_stiffness_variance([], 0)
    assert var == 0.0
    assert status == ANKLE_STIFFNESS_LOCKED
