# -*- coding: utf-8 -*-
"""V3.9：脚踝最大形变落差角（Max Angular Deflection）单测。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biomech_primitives import (
    ANKLE_STIFFNESS_LOCKED,
    ANKLE_STIFFNESS_SLIGHT_DEFORMATION,
    ANKLE_STIFFNESS_YIELDING,
    calculate_ankle_deflection,
    calculate_ankle_stiffness_variance,
)
from error_diagnoser import (
    ANKLE_DEFLECTION_FAIL_DEG,
    DeterministicErrorEngine,
    ERR_C1_LOOSE_ANKLE,
)


def test_deflection_locked_under_10deg():
    series = [140.0, 140.2, 140.1, 139.9, 140.0]
    d, status = calculate_ankle_deflection(series, 2)
    assert status == ANKLE_STIFFNESS_LOCKED
    assert d < 10.0
    assert d == round(d, 2)


def test_deflection_slight_10_to_20():
    # T0±2 窗内落差约 15°
    series = [130.0, 135.0, 145.0, 140.0, 132.0]
    d, status = calculate_ankle_deflection(series, 2)
    assert 10.0 <= d <= 20.0
    assert status == ANKLE_STIFFNESS_SLIGHT_DEFORMATION


def test_deflection_yielding_over_20():
    series = [100.0, 120.0, 140.0, 110.0, 90.0]
    d, status = calculate_ankle_deflection(series, 2)
    assert d > 20.0
    assert status == ANKLE_STIFFNESS_YIELDING


def test_deflection_rejects_visibility_spike():
    """低可见度塌缩帧不得拉爆落差。"""
    series = [140.0, 0.0, 140.2, 140.1, 139.9]
    vis = [0.9, 0.1, 0.9, 0.9, 0.9]
    d, status = calculate_ankle_deflection(
        series, 2, landmark_visibility_series=vis
    )
    assert status == ANKLE_STIFFNESS_LOCKED
    assert d < 10.0


def test_legacy_variance_api_delegates_to_deflection():
    series = [140.0, 140.1, 140.0, 139.9, 140.05]
    d0, s0 = calculate_ankle_deflection(series, 2)
    d1, s1 = calculate_ankle_stiffness_variance(series, 2)
    assert s0 == s1
    assert abs(d0 - d1) < 1e-12


def test_error_engine_c1_on_red_deflection():
    engine = DeterministicErrorEngine()
    hit = engine.evaluate(
        {
            "support_lateral_dist_cm": 17.0,
            "support_ap_offset_cm": 0.0,
            "support_ball_dist_cm": 17.0,
            "swing_fold_angle": 100.0,
            "thigh_retraction_deg": 20.0,
            "ankle_deflection_deg": 28.0,
            "ankle_stiffness_status": ANKLE_STIFFNESS_YIELDING,
            "ankle_locked": False,
            "instep_abduction_deg": 45.0,
            "approach_angle": 35.0,
            "support_foot_ratio": 1.0,
            "support_ratio": 0.55,
        }
    )
    assert hit["primary_error_code"] == ERR_C1_LOOSE_ANKLE
    assert ANKLE_DEFLECTION_FAIL_DEG == 20.0


def test_error_engine_no_c1_on_yellow_deflection_alone():
    """黄灯卸力（10–20°）不单独触发 C1；需 >20°。"""
    engine = DeterministicErrorEngine()
    result = engine.evaluate(
        {
            "support_lateral_dist_cm": 17.0,
            "support_ap_offset_cm": 0.0,
            "support_ball_dist_cm": 17.0,
            "swing_fold_angle": 100.0,
            "thigh_retraction_deg": 20.0,
            "ankle_deflection_deg": 15.0,
            "ankle_stiffness_status": ANKLE_STIFFNESS_SLIGHT_DEFORMATION,
            "ankle_locked": False,
            "instep_abduction_deg": 45.0,
            "approach_angle": 35.0,
            "support_foot_ratio": 1.0,
            "support_ratio": 0.55,
        }
    )
    assert result["primary_error_code"] != ERR_C1_LOOSE_ANKLE


def test_empty_series_safe():
    d, status = calculate_ankle_deflection([], 0)
    assert d == 0.0
    assert status == ANKLE_STIFFNESS_LOCKED
    assert math.isfinite(calculate_ankle_deflection(None, 0)[0])


def test_kinematic_guard_no_longer_clamps_ankle():
    from deterministic_scorer import apply_kinematic_physical_guards

    g = apply_kinematic_physical_guards(
        impact_knee_angle=150.0,
        support_knee_angle=155.0,
        distance_cm=17.5,
        ankle_variance=80.0,  # 旧暴走值；现为 deflection 原样透传
    )
    assert abs(g["ankle_variance"] - 80.0) < 1e-9
    assert g["ankle_clamped"] is False
    assert abs(g["ankle_deflection_deg"] - 80.0) < 1e-9
