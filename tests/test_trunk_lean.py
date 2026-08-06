# -*- coding: utf-8 -*-
"""V3.10：躯干倾角 trunk_lean_angle（纯 2D）单测。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biomech_primitives import (
    calculate_trunk_lean,
    classify_trunk_lean_status,
)
from error_diagnoser import (
    DeterministicErrorEngine,
    ERR_A2_SUPPORT_WIDE,
    ERR_D1_TRUNK_LEAN,
)


def _landmarks(*, shoulder_x: float, hip_x: float = 0.0) -> dict:
    """构造直立骨架；shoulder_x 相对 hip 的水平偏移决定倾角符号。"""
    return {
        "left_shoulder": [shoulder_x - 0.1, 0.2, 0.0],
        "right_shoulder": [shoulder_x + 0.1, 0.2, 0.0],
        "left_hip": [hip_x - 0.1, 0.5, 0.0],
        "right_hip": [hip_x + 0.1, 0.5, 0.0],
    }


def test_trunk_lean_upright_near_zero():
    lean = calculate_trunk_lean(_landmarks(shoulder_x=0.0, hip_x=0.0))
    assert lean is not None
    assert abs(lean) < 1.0
    assert classify_trunk_lean_status(lean) == "GREEN"


def test_trunk_lean_forward_positive():
    """肩相对髋偏 +X → 前倾为正。"""
    lean = calculate_trunk_lean(_landmarks(shoulder_x=0.12, hip_x=0.0))
    assert lean is not None and lean > 0.0
    assert classify_trunk_lean_status(lean) in ("GREEN", "YELLOW", "RED")


def test_trunk_lean_backward_negative():
    lean = calculate_trunk_lean(_landmarks(shoulder_x=-0.15, hip_x=0.0))
    assert lean is not None and lean < 0.0
    assert classify_trunk_lean_status(lean) in ("YELLOW", "RED")


def test_trunk_lean_ball_flips_anterior():
    """球在髋左侧时前向翻转：同一几何应得到相反符号。"""
    lm = _landmarks(shoulder_x=0.12, hip_x=0.0)
    lean_right = calculate_trunk_lean(lm, ball_center=(0.5, 0.5))
    lean_left = calculate_trunk_lean(lm, ball_center=(-0.5, 0.5))
    assert lean_right is not None and lean_left is not None
    assert lean_right > 0
    assert lean_left < 0


def test_classify_bands():
    assert classify_trunk_lean_status(8.0) == "GREEN"
    assert classify_trunk_lean_status(-5.0) == "YELLOW"
    assert classify_trunk_lean_status(20.0) == "YELLOW"
    assert classify_trunk_lean_status(-15.0) == "RED"
    assert classify_trunk_lean_status(30.0) == "RED"


def test_missing_landmarks_returns_none():
    assert calculate_trunk_lean({}) is None
    assert calculate_trunk_lean({"left_shoulder": [0, 0, 0]}) is None


def test_error_engine_d1_on_severe_back_lean():
    engine = DeterministicErrorEngine()
    hit = engine.evaluate(
        {
            "support_lateral_dist_cm": 17.0,
            "support_ap_offset_cm": 0.0,
            "support_ball_dist_cm": 17.0,
            "swing_fold_angle": 100.0,
            "thigh_retraction_deg": 20.0,
            "ankle_deflection_deg": 2.0,
            "ankle_stiffness_status": "LOCKED",
            "ankle_locked": True,
            "instep_abduction_deg": 45.0,
            "approach_angle": 35.0,
            "support_foot_ratio": 1.0,
            "support_ratio": 0.55,
            "trunk_lean_angle": -18.0,
            "trunk_lean_status": "RED",
        }
    )
    assert hit["primary_error_code"] == ERR_D1_TRUNK_LEAN


def test_error_engine_wide_stance_links_back_lean():
    """支撑过远优先于 D1，但文案须指出后仰代偿关联。"""
    engine = DeterministicErrorEngine()
    hit = engine.evaluate(
        {
            "support_lateral_dist_cm": 40.0,
            "support_ap_offset_cm": 0.0,
            "support_ball_dist_cm": 40.0,
            "swing_fold_angle": 100.0,
            "thigh_retraction_deg": 20.0,
            "ankle_deflection_deg": 2.0,
            "ankle_stiffness_status": "LOCKED",
            "ankle_locked": True,
            "instep_abduction_deg": 45.0,
            "approach_angle": 35.0,
            "support_foot_ratio": 2.0,
            "support_ratio": 1.1,
            "support_stance_code": ERR_A2_SUPPORT_WIDE,
            "trunk_lean_angle": -12.0,
            "trunk_lean_status": "RED",
            "trunk_lean_linked_to_wide_stance": True,
        }
    )
    assert hit["primary_error_code"] == ERR_A2_SUPPORT_WIDE
    assert "代偿" in hit["decision_reason"]


def test_scorer_trunk_red_deducts():
    from error_diagnoser import calculate_biomechanical_score

    impact = {
        "t_impact": 1,
        "distance_cm": 17.5,
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "ankle_angles_window": [140.0, 140.1, 140.0],
        "trunk_lean_angle": -18.0,
    }
    trajectory = {"max_folding_angle": 80.0, "whipping_velocity": 500.0}
    score, detail = calculate_biomechanical_score(impact, trajectory)
    trunk = detail["indicators"]["trunk_lean_angle"]
    assert trunk["status"] == "RED_DEVIATED"
    assert float(trunk["penalty"]) == 8.0
    assert score <= 92.0
