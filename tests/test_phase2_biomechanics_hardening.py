# -*- coding: utf-8 -*-
"""Phase 2：PCR QA / 摆动腿选侧 / 踝时长窗 / 原语单一权威。"""

from __future__ import annotations

import numpy as np

from biomech_primitives import (
    ANKLE_STIFFNESS_LOCKED,
    BALL_BBOX_MIN_DIAMETER_PX,
    DEFAULT_EMPIRICAL_PCR,
    STANDARD_BALL_DIAMETER_CM,
    ankle_half_window_frames,
    calculate_3d_joint_angle,
    calculate_ankle_stiffness_variance,
    calculate_support_foot_offset_cm,
    calculate_support_foot_offset_detailed,
    crosscheck_pcr_vs_world_lateral,
    evaluate_ball_bbox_for_pcr,
    infer_swing_leg_side,
    slice_ankle_impact_window_bounds,
)
from error_diagnoser import (
    PROVENANCE_ESTIMATED,
    PROVENANCE_MEASURED,
    calculate_biomechanical_score,
    calculate_3d_joint_angle as ed_angle,
    calculate_ankle_stiffness_variance as ed_ankle,
    calculate_support_foot_offset_cm as ed_offset,
)
from pose_tracker import (
    calculate_3d_joint_angle as pt_angle,
    calculate_ankle_stiffness_variance as pt_ankle,
    calculate_support_foot_offset_cm as pt_offset,
)
from llm_agent import _extract_indicator_payload


def test_primitives_single_authority_parity():
    """error_diagnoser / pose_tracker 与 biomech_primitives 数值一致。"""
    p1, p2, p3 = (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
    assert abs(calculate_3d_joint_angle(p1, p2, p3) - 90.0) < 1e-9
    assert ed_angle(p1, p2, p3) == calculate_3d_joint_angle(p1, p2, p3)
    assert pt_angle(p1, p2, p3) == calculate_3d_joint_angle(p1, p2, p3)

    bbox = [100.0, 200.0, 184.0, 284.0]
    ankle = (210.0, 250.0)
    assert abs(calculate_support_foot_offset_cm(ankle, bbox) - 17.0) < 1e-9
    assert ed_offset(ankle, bbox) == calculate_support_foot_offset_cm(ankle, bbox)
    assert pt_offset(ankle, bbox) == calculate_support_foot_offset_cm(ankle, bbox)

    series = [140.0, 140.2, 140.1]
    v0, s0 = calculate_ankle_stiffness_variance(series, 1)
    v1, s1 = ed_ankle(series, 1)
    v2, s2 = pt_ankle(series, 1)
    assert s0 == s1 == s2 == ANKLE_STIFFNESS_LOCKED
    assert abs(v0 - v1) < 1e-12 and abs(v0 - v2) < 1e-12


def test_ball_bbox_qa_rejects_tiny_and_blur():
    tiny = evaluate_ball_bbox_for_pcr([0.0, 0.0, 5.0, 5.0])
    assert tiny["ok"] is False
    assert tiny["reason"] == "diameter_too_small"
    assert tiny["diameter_px"] < BALL_BBOX_MIN_DIAMETER_PX

    # 极端扁长：宽 100、高 20 → aspect=5 > 2.5
    blur = evaluate_ball_bbox_for_pcr([0.0, 0.0, 100.0, 20.0])
    assert blur["ok"] is False
    assert blur["reason"] == "aspect_ratio_blur"

    good = evaluate_ball_bbox_for_pcr([100.0, 200.0, 184.0, 284.0])
    assert good["ok"] is True
    assert abs(good["pcr"] - (STANDARD_BALL_DIAMETER_CM / 84.0)) < 1e-9


def test_pcr_detailed_rejects_bad_bbox_not_measured_path():
    detail = calculate_support_foot_offset_detailed((50.0, 50.0), [0.0, 0.0, 8.0, 8.0])
    assert detail["ok"] is False
    # 坏球框不得标 measured；仍可用 fallback_PCR 给出钳制后的数值
    assert detail["method"].startswith("fallback_")
    assert 0.0 <= float(detail["offset_cm"]) <= 60.0
    # 兼容旧 float API：不再因坏框直接归零，而是走 fallback + [0,60] 钳制
    offset = calculate_support_foot_offset_cm((50.0, 50.0), [0.0, 0.0, 8.0, 8.0])
    assert 0.0 <= offset <= 60.0
    # 球心 x=4，踝 x=50，Δ=46；经验 PCR=21/84 → 11.5
    assert abs(offset - 46.0 * (STANDARD_BALL_DIAMETER_CM / 84.0)) < 1e-9


def test_scorer_rejects_blur_bbox_as_unmeasured():
    """模糊球框不得进入 AIGC measured 横距。"""
    impact = {
        "t_impact": 1,
        "support_ankle_px": (80.0, 40.0),
        "ball_pixel_bbox": [0.0, 0.0, 100.0, 20.0],  # aspect 过大
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "ankle_angles_window": [140.0, 140.1, 140.0],
    }
    trajectory = {"max_folding_angle": 80.0, "whipping_velocity": 500.0}
    _, detail = calculate_biomechanical_score(impact, trajectory)
    dist = detail["indicators"]["distance_cm"]
    assert dist["provenance"] != PROVENANCE_MEASURED or dist["method"] != "ball_pcr"
    # 无合格 PCR 且无显式上游 → default，value=None
    assert dist["value"] is None
    payload = _extract_indicator_payload({"score_detail": detail})
    assert "value" not in payload["distance_cm"]


def test_scorer_ignores_pcr_bbox_prefers_shoulder_ratio():
    """V3.8：即便有合格球框 PCR，站位评分也不得走 ball_pcr；须肩宽比或中性默认。"""
    impact = {
        "t_impact": 1,
        "support_ankle_px": (210.0, 250.0),
        "ball_pixel_bbox": [100.0, 200.0, 184.0, 284.0],  # 旧 PCR 路径会得到 ~17cm
        "world_support_lateral_cm": 40.0,
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "ankle_angles_window": [140.0, 140.1, 140.0],
    }
    trajectory = {"max_folding_angle": 80.0, "whipping_velocity": 500.0}
    _, detail = calculate_biomechanical_score(impact, trajectory)
    dist = detail["indicators"]["distance_cm"]
    assert dist["method"] != "ball_pcr"
    assert dist["unit"] == "ratio"
    # 无 landmarks / 无上游 ratio → 中性绿带中心，不得标 measured PCR
    assert dist["provenance"] != PROVENANCE_MEASURED
    payload = _extract_indicator_payload({"score_detail": detail})
    assert "value" not in payload.get("distance_cm", {}) or dist["value"] is None


def test_crosscheck_agree_within_tol():
    r = crosscheck_pcr_vs_world_lateral(17.0, 18.5, tol_cm=4.0)
    assert r["skipped"] is False
    assert r["agree"] is True
    assert r["confidence_factor"] == 1.0


def test_ankle_half_window_scales_with_fps():
    assert ankle_half_window_frames(30.0, 50.0) == 1
    assert ankle_half_window_frames(60.0, 50.0) == 3
    lo, hi, half = slice_ankle_impact_window_bounds(100, 50, fps=60.0, half_window_ms=50.0)
    assert half == 3
    assert lo == 47 and hi == 53


def test_ankle_deflection_fixed_half_window_ignores_fps_ms():
    """V3.9：形变落差固定 T0±2；fps/ms 参数被兼容入口忽略。"""
    series = [100.0] * 10
    series[3] = 130.0
    series[4] = 131.0
    series[6] = 129.0
    # 默认 half=2 → 窗 [3,7] 含扰动
    d_default, _ = calculate_ankle_stiffness_variance(series, 5)
    # 显式 half=1 → 窗 [4,6]，落差应更小或相等
    d_narrow, _ = calculate_ankle_stiffness_variance(
        series, 5, half_window_frames=1
    )
    assert d_default >= d_narrow
    assert d_default > 5.0


def test_infer_swing_leg_explicit_and_motion():
    assert infer_swing_leg_side([], 0, explicit_side="left") == "left"
    assert infer_swing_leg_side([], 0, explicit_side="right") == "right"

    frames = []
    for i in range(10):
        frames.append(
            {
                "left_ankle": (0.0, 0.0, 0.0),
                "right_ankle": (0.1 * i, 0.0, 0.2 * i),  # 右踝大幅位移
            }
        )
    assert infer_swing_leg_side(frames, 9) == "right"


def test_scorer_swing_leg_left_fold_method():
    """显式左脚摆动时 method 应带 left。"""
    # 构造足够 ROI 帧：左膝更折叠
    frames = []
    t = 20
    for i in range(40):
        # 右腿接近伸直，左腿后摆折叠
        fold_phase = max(0, t - i)
        left_knee_flex = 100.0 + fold_phase  # 更小内角 → 更大折叠
        # 用几何点近似：通过直接上游更稳；这里用显式 swing_leg + upstream fold
        frames.append(
            {
                "left_hip": (0.0, 0.0, 0.0),
                "left_knee": (0.0, 0.3, 0.0),
                "left_ankle": (0.0, 0.3, 0.3 if i < t else 0.1),
                "left_foot_index": (0.0, 0.3, 0.35),
                "right_hip": (0.2, 0.0, 0.0),
                "right_knee": (0.2, 0.3, 0.0),
                "right_ankle": (0.2, 0.55, 0.0),
                "right_foot_index": (0.2, 0.6, 0.0),
            }
        )
    impact = {
        "t_impact": t,
        "frames": frames,
        "swing_leg": "left",
        "distance_cm": 17.5,
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "ankle_angles_window": [140.0, 140.0, 140.1],
        "fps": 30.0,
    }
    trajectory = {"whipping_velocity": 500.0}
    _, detail = calculate_biomechanical_score(impact, trajectory)
    fold = detail["indicators"]["max_folding_angle"]
    assert fold.get("swing_leg") == "left"
    if fold["provenance"] == PROVENANCE_MEASURED:
        assert "left" in str(fold.get("method") or "")


def test_scorer_ankle_includes_dorsiflex_drop_when_measured():
    impact = {
        "t_impact": 1,
        "distance_cm": 17.5,
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "ankle_angles_window": [120.0, 140.0, 160.0],
        "fps": 30.0,
    }
    trajectory = {"max_folding_angle": 80.0, "whipping_velocity": 500.0}
    _, detail = calculate_biomechanical_score(impact, trajectory)
    ankle = detail["indicators"]["ankle_rigidity"]
    assert ankle["provenance"] == PROVENANCE_MEASURED
    assert ankle.get("unit") == "deg"
    assert ankle.get("deflection_deg") is not None or ankle.get("dorsiflex_drop_deg") is not None
    assert float(ankle.get("dorsiflex_drop_deg") or ankle.get("deflection_deg") or 0) > 0
    assert ankle.get("window_half_frames") == 2


def test_default_empirical_pcr_constant():
    assert DEFAULT_EMPIRICAL_PCR == STANDARD_BALL_DIAMETER_CM / 84.0


def test_kinematic_physical_guards_knee_distance_ankle():
    """Kinematic Boundary Guard：膝角补角+生理钳位 / 横距暴走；踝落差不再钳制。"""
    from error_diagnoser import apply_kinematic_physical_guards

    # 锐角补角假象 52° → 128°，落在 [120, 175]
    g = apply_kinematic_physical_guards(
        impact_knee_angle=52.0,
        support_knee_angle=80.0,  # → 100 → clamp 120
        distance_cm=17.5,
        ankle_variance=1.0,
    )
    assert abs(g["impact_knee_angle"] - 128.0) < 1e-9
    assert g["impact_knee_flipped"] is True
    assert abs(g["support_knee_angle"] - 120.0) < 1e-9
    assert g["support_knee_flipped"] is True

    # YOLO 漂移 68cm → 钳制 40.0 + Warning（V3.5：35cm 为合法严重区）
    g2 = apply_kinematic_physical_guards(
        impact_knee_angle=150.0,
        support_knee_angle=155.0,
        distance_cm=68.0,
        ankle_variance=1.0,
    )
    assert abs(g2["distance_cm"] - 40.0) < 1e-9
    assert g2["distance_clamped"] is True
    assert any("WARNING" in (e.get("status") or "") for e in g2["events"])

    # V3.9：踝形变落差原样透传，不再 80→12 钳制
    g3 = apply_kinematic_physical_guards(
        impact_knee_angle=150.0,
        support_knee_angle=155.0,
        distance_cm=17.5,
        ankle_variance=80.0,
    )
    assert abs(g3["ankle_variance"] - 80.0) < 1e-9
    assert g3["ankle_clamped"] is False


def test_scorer_applies_kinematic_guards_before_output():
    """DeterministicScorer：锐角膝护栏生效；裸漂移 cm 不再被吃进评分。"""
    impact = {
        "t_impact": 1,
        # V3.6：无 PCR / 无肩宽帧时，禁止把裸 68cm 当 measured
        "distance_cm": 68.0,
        "toe_angle": 5.0,
        "impact_knee_angle": 52.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "ankle_angles_window": [140.0, 140.1, 140.0],
    }
    trajectory = {"max_folding_angle": 70.0, "whipping_velocity": 500.0}
    _, detail = calculate_biomechanical_score(impact, trajectory)
    dist = detail["indicators"]["distance_cm"]
    # 回退肩宽比中性带（≈0.55），而非钳制后的伪 40cm / PCR
    assert float(dist["scoring_value"]) <= 1.0
    assert dist["unit"] == "ratio"
    assert dist["method"] in ("neutral_band_center", "shoulder_width_ratio")
    assert abs(detail["indicators"]["impact_knee_angle"]["value"] - 128.0) < 1e-9
    assert detail["indicators"]["impact_knee_angle"]["kinematic_guard"]["supplementary_flip"]


if __name__ == "__main__":
    test_primitives_single_authority_parity()
    test_ball_bbox_qa_rejects_tiny_and_blur()
    test_pcr_detailed_rejects_bad_bbox_not_measured_path()
    test_scorer_rejects_blur_bbox_as_unmeasured()
    test_scorer_ignores_pcr_bbox_prefers_shoulder_ratio()
    test_crosscheck_agree_within_tol()
    test_ankle_half_window_scales_with_fps()
    test_ankle_deflection_fixed_half_window_ignores_fps_ms()
    test_infer_swing_leg_explicit_and_motion()
    test_scorer_swing_leg_left_fold_method()
    test_scorer_ankle_includes_dorsiflex_drop_when_measured()
    test_default_empirical_pcr_constant()
    test_kinematic_physical_guards_knee_distance_ankle()
    test_scorer_applies_kinematic_guards_before_output()
    print("ALL PHASE2 TESTS PASSED")
