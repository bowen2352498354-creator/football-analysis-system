# -*- coding: utf-8 -*-
"""V3.5 儿童/业余评分校准：阈值放宽、单项封顶、扣分文案与实测值对齐。"""

from __future__ import annotations

from deterministic_scorer import calculate_biomechanical_score
from empirical_thresholds import assert_defaults_match_production, get_folding_bands, get_support_distance_bands
from error_diagnoser import (
    BACKSWING_STRAIGHT_LEG_DEG,
    DeterministicErrorEngine,
    ERR_A2_SUPPORT_WIDE,
    ERR_B1_STRAIGHT_LEG,
    SUPPORT_WIDE_CM,
)


def test_empirical_youth_bands():
    assert_defaults_match_production()
    d_gl, d_gh, d_yl, d_yh, _ = get_support_distance_bands()
    assert d_gh == 20.0 and d_yh == 35.0
    f_gl, f_gh, f_yl, f_yh, _ = get_folding_bands()
    assert (f_gl, f_gh, f_yl, f_yh) == (70.0, 100.0, 55.0, 120.0)


def test_near_standard_youth_shot_stays_above_70():
    """趋近标准的儿童动作：肩宽比≈0.80（略偏远黄灯）、折叠≈85、触球膝 162 → >70。"""
    # 折叠深度 85° 落在绿带 70–100（XY-2D / Z 坍缩口径）
    # 横距比 0.80 落在肩宽黄带 (0.70, 0.90]
    impact = {
        "t_impact": 1,
        "support_ratio": 0.80,
        "support_lateral_dist_cm": 24.0,
        "support_distance_method": "shoulder_width_ratio",
        "toe_angle": 8.0,
        "impact_knee_angle": 162.0,  # ≤165 不扣直腿
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "ankle_angles_window": [140.0, 140.2, 140.1],
    }
    trajectory = {
        "max_folding_angle": 85.0,
        "whipping_velocity": 400.0,
        "support_ratio": 0.80,
        "support_distance_method": "shoulder_width_ratio",
    }
    score, detail = calculate_biomechanical_score(impact, trajectory)
    assert score >= 70.0, f"趋近标准动作不应跌破 70，实际 {score}"
    fold = detail["indicators"]["max_folding_angle"]
    assert fold["status"] == "GREEN_OPTIMAL"
    assert fold["penalty"] == 0.0
    ik = detail["indicators"]["impact_knee_angle"]
    assert ik["status"] == "GREEN_OPTIMAL"
    dist = detail["indicators"]["distance_cm"]
    assert dist["status"] == "YELLOW_APPROACHING"
    assert dist["penalty"] <= 5.5  # 黄灯轻微扣分约 3–5
    assert dist["unit"] == "ratio"
    assert "支撑脚横距比例" in " ".join(
        d["reason"] for d in (detail.get("deductions") or [])
    )


def test_deduction_reason_uses_same_measured_fold_angle():
    """折叠扣分文案必须引用实测膝内角，禁止出现与 120° 矛盾的 >170° 文案。"""
    # 折叠深度 30° → 膝内角 150° > 140 → B1
    impact = {
        "t_impact": 1,
        "distance_cm": 17.5,
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "ankle_angles_window": [140.0, 140.1, 140.0],
    }
    trajectory = {"max_folding_angle": 30.0, "whipping_velocity": 500.0}
    _, detail = calculate_biomechanical_score(impact, trajectory)
    deductions = detail.get("deductions") or []
    fold_d = next(d for d in deductions if d["metric_key"] == "max_folding_angle")
    assert "150.0°" in fold_d["reason"] or "150°" in fold_d["reason"].replace(".0", "")
    assert "170" not in fold_d["reason"]
    assert fold_d["error_code"] == "ERR_B1_STRAIGHT_LEG"
    assert fold_d["penalty"] <= 8.0


def test_impact_knee_straight_leg_only_above_165():
    impact = {
        "t_impact": 1,
        "distance_cm": 17.5,
        "toe_angle": 5.0,
        "impact_knee_angle": 163.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "ankle_angles_window": [140.0, 140.1, 140.0],
    }
    trajectory = {"max_folding_angle": 70.0, "whipping_velocity": 500.0}
    score_ok, detail_ok = calculate_biomechanical_score(impact, trajectory)
    assert detail_ok["indicators"]["impact_knee_angle"]["penalty"] == 0.0

    impact["impact_knee_angle"] = 168.0
    score_bad, detail_bad = calculate_biomechanical_score(impact, trajectory)
    assert detail_bad["indicators"]["impact_knee_angle"]["penalty"] > 0.0
    assert detail_bad["indicators"]["impact_knee_angle"]["penalty"] <= 8.0
    assert score_bad < score_ok


def test_support_wide_err_requires_ratio_above_0_9():
    """ERR_A2 必须以肩宽比 >0.9 为准；黄灯区 0.80 不得触发严重偏宽。"""
    assert SUPPORT_WIDE_CM == 35.0  # 遗留 cm 兜底常数仍保留
    assert BACKSWING_STRAIGHT_LEG_DEG == 140.0
    engine = DeterministicErrorEngine()
    mild = engine.evaluate(
        {
            "support_ratio": 0.80,
            "support_lateral_dist_cm": 24.0,
            "support_ap_offset_cm": 0.0,
            "support_ball_dist_cm": 24.0,
            "swing_fold_angle": 110.0,
            "thigh_retraction_deg": 20.0,
            "ankle_variance": 1.0,
            "ankle_dorsiflex_drop_deg": 2.0,
            "instep_abduction_deg": 45.0,
            "ankle_locked": True,
            "approach_angle": 35.0,
            "support_foot_ratio": 1.1,
            "support_stance_code": "PASS_SUPPORT_OK",
            "early_deceleration": False,
            "foot_len_m": 0.22,
        }
    )
    assert mild["primary_error_code"] != ERR_A2_SUPPORT_WIDE

    wide = engine.evaluate(
        {
            "support_ratio": 1.05,
            "support_lateral_dist_cm": 31.5,
            "support_ap_offset_cm": 0.0,
            "support_ball_dist_cm": 31.5,
            "swing_fold_angle": 110.0,
            "thigh_retraction_deg": 20.0,
            "ankle_variance": 1.0,
            "ankle_dorsiflex_drop_deg": 2.0,
            "instep_abduction_deg": 45.0,
            "ankle_locked": True,
            "approach_angle": 35.0,
            "support_foot_ratio": 1.4,
            "support_stance_code": ERR_A2_SUPPORT_WIDE,
            "early_deceleration": False,
            "foot_len_m": 0.22,
        }
    )
    assert wide["primary_error_code"] == ERR_A2_SUPPORT_WIDE
    assert "支撑脚横距比例" in wide["decision_reason"]


def test_b1_uses_swing_fold_not_phantom_170():
    engine = DeterministicErrorEngine()
    # 120° 膝内角绝不能触发 B1（旧 bug：文案写 >170）
    ok = engine.evaluate(
        {
            "support_lateral_dist_cm": 17.5,
            "support_ap_offset_cm": 0.0,
            "support_ball_dist_cm": 17.5,
            "swing_fold_angle": 120.0,
            "thigh_retraction_deg": 25.0,
            "ankle_variance": 1.0,
            "ankle_dorsiflex_drop_deg": 2.0,
            "instep_abduction_deg": 45.0,
            "ankle_locked": True,
            "approach_angle": 35.0,
            "support_foot_ratio": 1.0,
            "early_deceleration": False,
            "foot_len_m": 0.22,
        }
    )
    assert ok["primary_error_code"] != ERR_B1_STRAIGHT_LEG

    bad = engine.evaluate(
        {
            "support_lateral_dist_cm": 17.5,
            "support_ap_offset_cm": 0.0,
            "support_ball_dist_cm": 17.5,
            "swing_fold_angle": 145.0,
            "thigh_retraction_deg": 25.0,
            "ankle_variance": 1.0,
            "ankle_dorsiflex_drop_deg": 2.0,
            "instep_abduction_deg": 45.0,
            "ankle_locked": True,
            "approach_angle": 35.0,
            "support_foot_ratio": 1.0,
            "early_deceleration": False,
            "foot_len_m": 0.22,
        }
    )
    assert bad["primary_error_code"] == ERR_B1_STRAIGHT_LEG
    assert "145" in bad["decision_reason"]
    assert "170" not in bad["decision_reason"]


def test_single_metric_penalty_cap():
    """单项最高扣分：支撑距 ≤10，其余 ≤8。"""
    impact = {
        "t_impact": 1,
        "support_ratio": 1.5,
        "support_distance_method": "shoulder_width_ratio",
        "toe_angle": 40.0,
        "impact_knee_angle": 174.0,
        "support_knee_angle": 110.0,
        "hip_torsion_angle": 70.0,
        "ankle_angles_window": [100.0, 140.0, 180.0],
    }
    trajectory = {
        "max_folding_angle": 20.0,
        "whipping_velocity": 50.0,
        "support_ratio": 1.5,
        "support_distance_method": "shoulder_width_ratio",
    }
    score, detail = calculate_biomechanical_score(impact, trajectory)
    for key, entry in detail["indicators"].items():
        cap = 10.0 if key == "distance_cm" else 8.0
        assert float(entry.get("penalty") or 0) <= cap + 1e-9, f"{key} 超单项上限"
    assert score >= 100.0 - (10.0 + 8.0 * 7)  # 理论最差下限
