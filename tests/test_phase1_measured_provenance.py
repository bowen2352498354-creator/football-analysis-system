# -*- coding: utf-8 -*-
"""Phase 1：焦点三指标 provenance 契约 + AIGC 载荷防幻觉回归。"""

from __future__ import annotations

from error_diagnoser import (
    PROVENANCE_DEFAULT,
    PROVENANCE_MEASURED,
    PROVENANCE_MISSING,
    DeterministicScorer,
    calculate_biomechanical_score,
    is_aigc_measurable_provenance,
)
from llm_agent import _build_clinical_fallback_markdown, _extract_indicator_payload


def test_is_aigc_measurable_provenance_gate():
    assert is_aigc_measurable_provenance("measured") is True
    assert is_aigc_measurable_provenance("calibrated") is True
    assert is_aigc_measurable_provenance("MEASURED") is True
    assert is_aigc_measurable_provenance("default") is False
    assert is_aigc_measurable_provenance("missing") is False
    assert is_aigc_measurable_provenance("estimated") is False
    assert is_aigc_measurable_provenance(None) is False


def test_scorer_default_distance_not_exported_as_measured():
    """无 PCR / 无显式横距时：评分可用 17.5，但对外 value 必须为 None。"""
    impact = {
        "t_impact": 1,
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "ankle_angles_window": [140.0, 140.1, 140.0],
    }
    trajectory = {"max_folding_angle": 80.0, "whipping_velocity": 500.0}
    _, detail = calculate_biomechanical_score(impact, trajectory)
    dist = detail["indicators"]["distance_cm"]
    assert dist["provenance"] == PROVENANCE_DEFAULT
    assert dist["value"] is None
    assert abs(float(dist["scoring_value"]) - 17.5) < 1e-9


def test_scorer_pcr_distance_is_measured():
    """YOLO 球框 PCR 路径必须标为 measured，并写出 value。"""
    # 球径 84px → PCR=0.25；踝距球心 68px → 17.0 cm
    impact = {
        "t_impact": 1,
        "support_ankle_px": (210.0, 250.0),
        "ball_pixel_bbox": [100.0, 200.0, 184.0, 284.0],
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "ankle_angles_window": [140.0, 140.1, 140.0],
    }
    trajectory = {"max_folding_angle": 82.0, "whipping_velocity": 500.0}
    _, detail = calculate_biomechanical_score(impact, trajectory)
    dist = detail["indicators"]["distance_cm"]
    assert dist["provenance"] == PROVENANCE_MEASURED
    assert dist["method"] == "ball_pcr"
    assert dist["value"] is not None
    assert abs(float(dist["value"]) - 17.0) < 1e-6


def test_scorer_default_folding_without_frames():
    """无帧且无上游折叠角：80° 仅作 scoring_value，不得进入 value。"""
    impact = {
        "t_impact": 1,
        "distance_cm": 17.5,
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "ankle_angles_window": [140.0, 140.1, 140.0],
    }
    trajectory = {"whipping_velocity": 500.0}
    _, detail = calculate_biomechanical_score(impact, trajectory)
    fold = detail["indicators"]["max_folding_angle"]
    assert fold["provenance"] == PROVENANCE_DEFAULT
    assert fold["value"] is None
    assert abs(float(fold["scoring_value"]) - 80.0) < 1e-9


def test_scorer_missing_ankle_window_not_locked_for_aigc():
    """无踝窗：provenance=missing，value/stiffness_status 不得伪装 LOCKED 实测。"""
    impact = {
        "t_impact": 1,
        "distance_cm": 17.5,
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        # 故意不提供 ankle_angles_window / frames
    }
    trajectory = {"max_folding_angle": 80.0, "whipping_velocity": 500.0}
    _, detail = calculate_biomechanical_score(impact, trajectory)
    ankle = detail["indicators"]["ankle_rigidity"]
    assert ankle["provenance"] == PROVENANCE_MISSING
    assert ankle["value"] is None
    assert ankle.get("variance") is None
    assert ankle.get("stiffness_status") is None
    assert ankle["scoring_value"] is not None


def test_aigc_payload_omits_unmeasured_focus_values():
    """_extract_indicator_payload 不得向 AIGC 注入 default/missing 数值。"""
    diagnosis = {
        "score_detail": {
            "indicators": {
                "distance_cm": {
                    "value": None,
                    "scoring_value": 17.5,
                    "status": "GREEN_OPTIMAL",
                    "provenance": PROVENANCE_DEFAULT,
                    "unit": "cm",
                },
                "max_folding_angle": {
                    "value": None,
                    "scoring_value": 80.0,
                    "status": "GREEN_OPTIMAL",
                    "provenance": PROVENANCE_DEFAULT,
                    "unit": "deg",
                },
                "ankle_rigidity": {
                    "value": None,
                    "scoring_value": 0.0,
                    "variance": None,
                    "stiffness_status": None,
                    "status": "GREEN_OPTIMAL",
                    "provenance": PROVENANCE_MISSING,
                    "unit": "variance",
                },
                "impact_knee_angle": {
                    "value": 150.0,
                    "status": "GREEN_OPTIMAL",
                    "provenance": PROVENANCE_MEASURED,
                    "unit": "deg",
                },
            }
        }
    }
    payload = _extract_indicator_payload(diagnosis)
    assert "value" not in payload["distance_cm"]
    assert payload["distance_cm"]["measured"] is False
    assert "未提供实测值" in payload["distance_cm"]["note"]
    assert "value" not in payload["max_folding_angle"]
    assert "value" not in payload["ankle_rigidity"]
    assert "stiffness_status" not in payload["ankle_rigidity"]
    # 非焦点指标仍可带 value
    assert payload["impact_knee_angle"]["value"] == 150.0


def test_aigc_payload_includes_measured_focus_values():
    diagnosis = {
        "score_detail": {
            "indicators": {
                "distance_cm": {
                    "value": 28.5,
                    "scoring_value": 28.5,
                    "status": "RED_DEVIATED",
                    "provenance": PROVENANCE_MEASURED,
                    "method": "ball_pcr",
                    "unit": "cm",
                },
                "max_folding_angle": {
                    "value": 55.0,
                    "scoring_value": 55.0,
                    "status": "RED_DEVIATED",
                    "provenance": "calibrated",
                    "unit": "deg",
                },
                "ankle_rigidity": {
                    "value": 6.2,
                    "variance": 6.2,
                    "stiffness_status": "YIELDING",
                    "status": "RED_DEVIATED",
                    "provenance": PROVENANCE_MEASURED,
                    "unit": "variance",
                },
            }
        }
    }
    payload = _extract_indicator_payload(diagnosis)
    assert payload["distance_cm"]["value"] == 28.5
    assert payload["distance_cm"]["measured"] is True
    assert payload["max_folding_angle"]["value"] == 55.0
    assert payload["ankle_rigidity"]["value"] == 6.2
    assert payload["ankle_rigidity"]["stiffness_status"] == "YIELDING"


def test_aigc_payload_never_backfills_scoring_value_from_metrics():
    """即使 metrics 里有横距，缺测 provenance 也不得回填进 AIGC。"""
    diagnosis = {
        "metrics": {"support_lateral_dist_cm": 17.5, "distance_cm": 17.5},
        "score_detail": {
            "indicators": {
                "distance_cm": {
                    "value": None,
                    "scoring_value": 17.5,
                    "provenance": PROVENANCE_DEFAULT,
                    "status": "GREEN_OPTIMAL",
                    "unit": "cm",
                }
            }
        },
    }
    payload = _extract_indicator_payload(diagnosis)
    assert "value" not in payload["distance_cm"]


def test_fallback_markdown_does_not_invent_focus_numbers():
    diagnosis = {
        "score_detail": {
            "indicators": {
                "distance_cm": {
                    "value": None,
                    "scoring_value": 17.5,
                    "provenance": PROVENANCE_DEFAULT,
                    "status": "GREEN_OPTIMAL",
                    "unit": "cm",
                },
                "max_folding_angle": {
                    "value": None,
                    "scoring_value": 80.0,
                    "provenance": PROVENANCE_DEFAULT,
                    "status": "GREEN_OPTIMAL",
                    "unit": "deg",
                },
                "ankle_rigidity": {
                    "value": None,
                    "provenance": PROVENANCE_MISSING,
                    "status": "GREEN_OPTIMAL",
                    "unit": "variance",
                },
            }
        }
    }
    text = _build_clinical_fallback_markdown(diagnosis)
    assert "未提供实测值" in text
    assert "17.5" not in text
    assert "80.0" not in text
    assert "80°" not in text


def test_pack_focus_roundtrip_via_scorer_explicit_upstream():
    """显式上游实测值应进入 AIGC payload。"""
    impact = {
        "t_impact": 1,
        "distance_cm": 18.2,
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "ankle_angles_window": [139.0, 140.0, 141.0],
    }
    trajectory = {"max_folding_angle": 78.0, "whipping_velocity": 500.0}
    score, detail = DeterministicScorer().calculate_biomechanical_score(impact, trajectory)
    assert isinstance(score, float)
    payload = _extract_indicator_payload({"score_detail": detail})
    assert payload["distance_cm"]["value"] == 18.2
    assert payload["max_folding_angle"]["value"] == 78.0
    assert payload["ankle_rigidity"]["measured"] is True
    assert "value" in payload["ankle_rigidity"]


if __name__ == "__main__":
    test_is_aigc_measurable_provenance_gate()
    test_scorer_default_distance_not_exported_as_measured()
    test_scorer_pcr_distance_is_measured()
    test_scorer_default_folding_without_frames()
    test_scorer_missing_ankle_window_not_locked_for_aigc()
    test_aigc_payload_omits_unmeasured_focus_values()
    test_aigc_payload_includes_measured_focus_values()
    test_aigc_payload_never_backfills_scoring_value_from_metrics()
    test_fallback_markdown_does_not_invent_focus_numbers()
    test_pack_focus_roundtrip_via_scorer_explicit_upstream()
    print("ALL PHASE1 MEASURED PROVENANCE TESTS PASSED")
