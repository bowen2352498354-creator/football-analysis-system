# -*- coding: utf-8 -*-
"""Phase 4 + Phase1–4 综合验收：标定 / fps / 经验阈值 / 全链路防幻觉。"""

from __future__ import annotations

import json

import academic_exporter as ae
from calibration_protocol import run_full_calibration_suite
from coach_calibration import apply_coach_calibration
from empirical_thresholds import (
    DEFAULT_THRESHOLDS,
    assert_defaults_match_production,
    get_ankle_half_window_ms,
    get_support_distance_green_band,
    load_empirical_thresholds,
)
from error_diagnoser import (
    PROVENANCE_CALIBRATED,
    PROVENANCE_DEFAULT,
    PROVENANCE_MEASURED,
    PROVENANCE_MISSING,
    calculate_biomechanical_score,
    is_aigc_measurable_provenance,
)
from llm_agent import build_aigc_safe_payload, build_aigc_user_message
from biomech_primitives import ankle_half_window_frames


def test_empirical_defaults_match_production_bands():
    assert_defaults_match_production()
    low, high = get_support_distance_green_band()
    assert (low, high) == (15.0, 20.0)
    assert get_ankle_half_window_ms() == 50.0
    cfg = load_empirical_thresholds(force_reload=True)
    assert cfg["population"] == "production_defaults_v31"
    assert DEFAULT_THRESHOLDS["ankle_impact_half_window_ms"] == 50.0


def test_coach_calibration_writes_calibrated_and_audit():
    record = {
        "id": "rec-1",
        "supportFootDistance": 30.0,
        "supportFootDistanceProvenance": "estimated",
        "scoreDetail": {
            "indicators": {
                "distance_cm": {
                    "value": None,
                    "scoring_value": 17.5,
                    "provenance": PROVENANCE_DEFAULT,
                }
            }
        },
    }
    result = apply_coach_calibration(
        record,
        metric_key="distance_cm",
        value=17.5,
        coach_id="coach_A",
        note="地标盘复核",
    )
    assert result["ok"] is True
    assert record["supportFootDistance"] == 17.5
    assert record["supportFootDistanceProvenance"] == PROVENANCE_CALIBRATED
    assert record["distance_cm"] == 17.5
    entry = record["scoreDetail"]["indicators"]["distance_cm"]
    assert entry["value"] == 17.5
    assert entry["provenance"] == PROVENANCE_CALIBRATED
    assert entry["method"] == "coach_manual"
    assert len(record["calibration_audit"]) == 1
    assert record["calibration_audit"][0]["note"] == "地标盘复核"


def test_coach_calibration_rejects_out_of_range_and_unknown_metric():
    record = {"id": "x"}
    bad = apply_coach_calibration(record, metric_key="distance_cm", value=999.0)
    assert bad["ok"] is False
    unknown = apply_coach_calibration(record, metric_key="toe_angle", value=5.0)
    assert unknown["ok"] is False


def test_calibrated_flows_to_aigc_and_measured_only_export():
    record = {"id": "rec-2", "studentId": "S100", "score": 88.0, "timestamp": "2026-07-22 12:00:00"}
    apply_coach_calibration(record, metric_key="distance_cm", value=18.0, note="教练复核")
    apply_coach_calibration(record, metric_key="max_folding_angle", value=80.0)
    apply_coach_calibration(record, metric_key="ankle_rigidity", value=1.2)

    # AIGC：calibrated 可复述
    diagnosis = {"score_detail": record["scoreDetail"]}
    payload = build_aigc_safe_payload(diagnosis)
    assert payload["indicators"]["distance_cm"]["value"] == 18.0
    assert payload["indicators"]["distance_cm"]["provenance"] == "calibrated"
    assert is_aigc_measurable_provenance("calibrated")
    msg = build_aigc_user_message(diagnosis)
    assert "18.0" in msg

    # 导出：calibrated 可通过 measured_only
    df_all = ae.build_long_format_dataframe(
        [
            {
                **record,
                "school": "S",
                "classGroup": "C",
                "type": "realtime",
            }
        ],
        measured_only=False,
    )
    df_m = ae.build_long_format_dataframe(
        [
            {
                **record,
                "school": "S",
                "classGroup": "C",
                "type": "realtime",
            }
        ],
        measured_only=True,
    )
    assert len(df_all) == 1
    assert len(df_m) == 1
    assert df_m.loc[0, "support_foot_distance_provenance"] == ae.PROVENANCE_CALIBRATED


def test_fps_passthrough_widens_ankle_window_in_scorer():
    """60fps 时半窗帧数=3，indicator 记录 window_half_frames。"""
    assert ankle_half_window_frames(60.0, 50.0) == 3
    impact = {
        "t_impact": 5,
        "fps": 60.0,
        "distance_cm": 17.5,
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "ankle_angles_time_series": [140.0] * 11,
    }
    trajectory = {"max_folding_angle": 80.0, "whipping_velocity": 500.0, "fps": 60.0}
    _, detail = calculate_biomechanical_score(impact, trajectory)
    ankle = detail["indicators"]["ankle_rigidity"]
    assert ankle["window_half_frames"] == 3
    assert abs(float(ankle["fps"]) - 60.0) < 1e-6


def test_phase1_to4_full_pipeline_no_hallucinated_defaults():
    """综合：PCR 实测 → Scorer → AIGC；缺测不注入；标定可进导出。"""
    # 1) PCR measured
    impact = {
        "t_impact": 1,
        "support_ankle_px": (210.0, 250.0),
        "ball_pixel_bbox": [100.0, 200.0, 184.0, 284.0],
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "ankle_angles_window": [140.0, 140.1, 140.0],
        "fps": 30.0,
    }
    trajectory = {"max_folding_angle": 78.0, "whipping_velocity": 500.0, "fps": 30.0}
    _, detail = calculate_biomechanical_score(impact, trajectory)
    assert detail["indicators"]["distance_cm"]["provenance"] == PROVENANCE_MEASURED
    assert detail["indicators"]["distance_cm"]["value"] == 17.0
    msg = build_aigc_user_message({"score_detail": detail})
    assert "17.0" in msg

    # 2) 缺测横距不得进 AIGC
    _, detail_miss = calculate_biomechanical_score(
        {
            "t_impact": 1,
            "toe_angle": 5.0,
            "impact_knee_angle": 150.0,
            "support_knee_angle": 155.0,
            "hip_torsion_angle": 25.0,
            "ankle_angles_window": [140.0, 140.0, 140.1],
        },
        {"max_folding_angle": 80.0, "whipping_velocity": 500.0},
    )
    assert detail_miss["indicators"]["distance_cm"]["provenance"] == PROVENANCE_DEFAULT
    payload_miss = build_aigc_safe_payload({"score_detail": detail_miss})
    assert "value" not in payload_miss["indicators"]["distance_cm"]
    assert payload_miss["indicators"]["distance_cm"].get("measured") is False

    # 3) 教室标定夹具仍 PASS
    assert run_full_calibration_suite()["pass"] is True

    # 4) 教练标定后可过滤导出
    rec = {
        "id": "full-1",
        "school": "S",
        "classGroup": "C",
        "studentId": "Z9",
        "timestamp": "2026-07-22 13:00:00",
        "type": "realtime",
        "score": 90.0,
    }
    apply_coach_calibration(rec, metric_key="distance_cm", value=16.0)
    df = ae.build_long_format_dataframe([rec], measured_only=True)
    assert len(df) == 1
    assert df.loc[0, "support_foot_distance"] == 16.0


def test_missing_ankle_still_not_locked_for_aigc_after_phase4():
    impact = {
        "t_impact": 1,
        "distance_cm": 17.5,
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "fps": 30.0,
    }
    trajectory = {"max_folding_angle": 80.0, "whipping_velocity": 500.0}
    _, detail = calculate_biomechanical_score(impact, trajectory)
    ankle = detail["indicators"]["ankle_rigidity"]
    assert ankle["provenance"] == PROVENANCE_MISSING
    assert ankle["value"] is None
    payload = build_aigc_safe_payload({"score_detail": detail})
    assert "value" not in payload["indicators"]["ankle_rigidity"]
    assert payload["indicators"]["ankle_rigidity"].get("measured") is False


def test_generate_report_must_wrap_score_detail_for_aigc():
    """回归：综合报告若只传 sample_angles、不传 score_detail，AIGC 会空 indicators。

    api_server 必须调用::
        generate_session_report(..., diagnosis_json={"score_detail": score_detail})
    """
    import inspect

    import api_server as server

    src = inspect.getsource(server.generate_report)
    assert 'diagnosis_json=diagnosis_for_aigc' in src or 'diagnosis_json={' in src
    assert "score_detail" in src
    assert "generate_session_report" in src


def test_pose_frames_feed_measured_fold_ankle_and_world_lateral_to_aigc():
    """有姿态帧时：折叠角/踝刚度/世界横距应进入 AIGC 可复述载荷（非默认中心值）。"""
    frames = []
    for i in range(20):
        # 后摆：膝角逐步减小；触球附近踝角稳定
        knee_interior = 100.0 - i * 1.5  # → max_folding ≈ 180-70 = 110 at late frames
        fold = max(70.0, knee_interior)
        # 右摆腿：hip / knee / ankle 构成折角；足尖用于踝刚度
        rec = {
            "timestamp_sec": i / 30.0,
            "right_hip": [0.0, 0.0, 0.0],
            "right_knee": [0.0, 0.4, 0.0],
            "right_ankle": [0.0, 0.4 + 0.4 * __import__("math").cos(__import__("math").radians(fold)),
                            0.4 * __import__("math").sin(__import__("math").radians(fold))],
            "right_foot_index": [0.05, 0.85, 0.05],
            "left_ankle": [-0.18, 0.9, 0.0],
            "left_hip": [-0.1, 0.0, 0.0],
            "left_knee": [-0.1, 0.4, 0.0],
            "left_foot_index": [-0.18, 1.0, 0.05],
            "visibility": {k: 1.0 for k in (
                "right_hip", "right_knee", "right_ankle", "right_foot_index",
                "left_ankle", "left_hip", "left_knee", "left_foot_index",
            )},
            "world": {},
        }
        # 世界坐标：左踝与右足尖横距 0.17m = 17cm
        rec["world"] = {
            "left_ankle": [-0.17, 0.0, 0.0],
            "right_foot_index": [0.0, 0.0, 0.0],
            "right_ankle": [0.0, 0.0, 0.0],
            "right_hip": [0.0, -0.4, 0.0],
            "right_knee": [0.0, -0.2, 0.0],
            "left_hip": [-0.1, -0.4, 0.0],
            "left_knee": [-0.12, -0.2, 0.0],
            "left_foot_index": [-0.17, 0.05, 0.0],
        }
        frames.append(rec)

    t_impact = 15
    impact = {
        "t_impact": t_impact,
        "frames": frames,
        "fps": 30.0,
        "ball_center": frames[t_impact]["world"]["right_foot_index"],
        "support_lateral_dist_cm": 17.0,
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
    }
    trajectory = {
        "frames": frames,
        "fps": 30.0,
        "knee_angles": [150.0] * len(frames),
        "angular_velocities": [0.0] * len(frames),
        "whipping_velocity": 500.0,
        "support_lateral_dist_cm": 17.0,
    }
    _, detail = calculate_biomechanical_score(impact, trajectory)
    # 与 api_server 相同包装
    payload = build_aigc_safe_payload({"score_detail": detail})
    dist = payload["indicators"]["distance_cm"]
    fold = payload["indicators"]["max_folding_angle"]
    ankle = payload["indicators"]["ankle_rigidity"]
    assert dist.get("measured") is True and dist.get("value") is not None
    assert fold.get("measured") is True and fold.get("value") is not None
    assert ankle.get("measured") is True and ankle.get("value") is not None
    assert is_aigc_measurable_provenance(dist["provenance"])
    assert is_aigc_measurable_provenance(fold["provenance"])
    assert is_aigc_measurable_provenance(ankle["provenance"])


if __name__ == "__main__":
    test_empirical_defaults_match_production_bands()
    test_coach_calibration_writes_calibrated_and_audit()
    test_coach_calibration_rejects_out_of_range_and_unknown_metric()
    test_calibrated_flows_to_aigc_and_measured_only_export()
    test_fps_passthrough_widens_ankle_window_in_scorer()
    test_phase1_to4_full_pipeline_no_hallucinated_defaults()
    test_missing_ankle_still_not_locked_for_aigc_after_phase4()
    test_generate_report_must_wrap_score_detail_for_aigc()
    test_pose_frames_feed_measured_fold_ankle_and_world_lateral_to_aigc()
    print("ALL PHASE4 + INTEGRATION TESTS PASSED")
