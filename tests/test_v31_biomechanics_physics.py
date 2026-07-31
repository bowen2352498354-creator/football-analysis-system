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


def test_knee_extension_anatomical_correction():
    """触球伸展语境：angle < 130 视为补角反转，翻转为 180-angle。"""
    # 构造内角 ≈ 52° 的三点（锐角）→ 补角 128°
    hip = (0.0, 1.0, 0.0)
    knee = (0.0, 0.0, 0.0)
    rad = np.radians(52.0)
    ankle = (float(np.sin(rad)), float(np.cos(rad)), 0.0)
    raw = calculate_3d_joint_angle(hip, knee, ankle, is_knee_extension=False)
    assert abs(raw - 52.0) < 1e-6
    fixed = calculate_3d_joint_angle(hip, knee, ankle, is_knee_extension=True)
    assert abs(fixed - 128.0) < 1e-6
    # 半伸展假象 110° 也应在触球语境下翻转为 70°（阈值 130）
    rad110 = np.radians(110.0)
    ankle110 = (float(np.sin(rad110)), float(np.cos(rad110)), 0.0)
    assert abs(calculate_3d_joint_angle(hip, knee, ankle110, is_knee_extension=False) - 110.0) < 1e-6
    assert abs(calculate_3d_joint_angle(hip, knee, ankle110, is_knee_extension=True) - 70.0) < 1e-6
    # 折叠解算保持默认：不翻转
    assert abs(calculate_3d_joint_angle(hip, knee, ankle) - 52.0) < 1e-6
    # 已在伸展带内（≥130）不再二次翻转
    rad150 = np.radians(150.0)
    ankle150 = (float(np.sin(rad150)), float(np.cos(rad150)), 0.0)
    assert abs(calculate_3d_joint_angle(hip, knee, ankle150, is_knee_extension=True) - 150.0) < 1e-6


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
    # 人为构造 t±1 三帧方差落在 [2, 5]
    series = [130.0, 135.0, 132.0]
    var, status = calculate_ankle_stiffness_variance(series, t_impact_index=1)
    assert 2.0 <= var <= 5.0
    assert status == ANKLE_STIFFNESS_SLIGHT_DEFORMATION


def test_ankle_stiffness_yielding():
    series = [100.0, 120.0, 90.0]
    var, status = calculate_ankle_stiffness_variance(series, t_impact_index=1)
    assert var > 5.0
    assert status == ANKLE_STIFFNESS_YIELDING


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
