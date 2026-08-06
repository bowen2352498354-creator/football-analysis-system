# -*- coding: utf-8 -*-
"""Phase 3：教室标定门槛 + AIGC E2E + 启发式 provenance 导出过滤。"""

from __future__ import annotations

import json

import academic_exporter as ae
from calibration_protocol import (
    FOLD_MAE_MAX_DEG,
    PCR_MAE_MAX_CM,
    format_calibration_report,
    run_full_calibration_suite,
)
from error_diagnoser import (
    PROVENANCE_DEFAULT,
    PROVENANCE_MEASURED,
    PROVENANCE_MISSING,
    calculate_biomechanical_score,
)
from llm_agent import (
    _build_clinical_fallback_markdown,
    build_aigc_safe_payload,
    build_aigc_user_message,
    generate_feedback,
)


def test_calibration_suite_meets_acceptance_gates():
    suite = run_full_calibration_suite()
    assert suite["pass"] is True
    assert suite["pcr"]["mae_cm"] <= PCR_MAE_MAX_CM
    assert suite["fold"]["mae_deg"] <= FOLD_MAE_MAX_DEG
    assert suite["ankle"]["accuracy"] >= 1.0
    report = format_calibration_report(suite)
    assert "PASS" in report
    assert "PCR" in report


def test_aigc_e2e_user_message_contains_exact_measured_numbers():
    diagnosis = {
        "primary_error_code": "ERR_SUPPORT_TOO_WIDE",
        "t_impact": 42,
        "score_detail": {
            "indicators": {
                "distance_cm": {
                    "value": 28.5,
                    "scoring_value": 28.5,
                    "status": "RED_DEVIATED",
                    "provenance": PROVENANCE_MEASURED,
                    "unit": "cm",
                    "method": "ball_pcr",
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
        },
    }
    payload = build_aigc_safe_payload(diagnosis)
    assert payload["indicators"]["distance_cm"]["value"] == 28.5
    assert payload["indicators"]["max_folding_angle"]["value"] == 55.0
    assert payload["indicators"]["ankle_rigidity"]["value"] == 6.2
    assert "primary_error_description" in payload
    msg = build_aigc_user_message(diagnosis)
    assert "28.5" in msg
    assert "55.0" in msg or "55" in msg
    assert "6.2" in msg
    assert "【ClinicalBrief 首要事实】" in msg or "【首要错误锁定】" in msg
    # 全量测量上下文必须注入（总分 / 雷达 / 扣分）
    assert "TotalScore" in msg
    assert "overview" in msg
    assert "biomechanical_analysis" in msg
    assert "主要扣分病灶" in msg


def test_aigc_e2e_fallback_quotes_measured_and_blocks_defaults():
    diagnosis = {
        "score_detail": {
            "indicators": {
                "distance_cm": {
                    "value": 28.5,
                    "scoring_value": 28.5,
                    "provenance": PROVENANCE_MEASURED,
                    "status": "RED_DEVIATED",
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
                    "scoring_value": 0.0,
                    "provenance": PROVENANCE_MISSING,
                    "status": "GREEN_OPTIMAL",
                    "unit": "variance",
                },
            }
        }
    }
    text = _build_clinical_fallback_markdown(diagnosis)
    assert "28.5" in text
    assert "未提供实测值" in text
    assert "80.0" not in text
    assert "17.5" not in text


def test_aigc_e2e_generate_feedback_offline_fallback_optimal_dual(monkeypatch):
    """强制 LLM 失败走 OPTIMAL 双段兜底（零网络）；实测值仍由 clinical helper 复述。"""
    diagnosis = {
        "score_detail": {
            "indicators": {
                "distance_cm": {
                    "value": 0.55,
                    "scoring_value": 0.55,
                    "provenance": "calibrated",
                    "status": "GREEN_OPTIMAL",
                    "unit": "ratio",
                    "support_ratio": 0.55,
                    "method": "shoulder_width_ratio",
                },
                "max_folding_angle": {
                    "value": 82.0,
                    "scoring_value": 82.0,
                    "provenance": PROVENANCE_MEASURED,
                    "status": "GREEN_OPTIMAL",
                    "unit": "deg",
                },
                "ankle_rigidity": {
                    "value": 1.1,
                    "variance": 1.1,
                    "stiffness_status": "LOCKED",
                    "provenance": PROVENANCE_MEASURED,
                    "status": "GREEN_OPTIMAL",
                    "unit": "variance",
                },
            }
        }
    }

    class _Boom:
        def create(self, *args, **kwargs):
            raise RuntimeError("forced offline")

    class _Client:
        chat = type("C", (), {"completions": _Boom()})()

    import llm_agent as la

    monkeypatch.setattr(la, "client", _Client())
    dual = la.generate_optimal_dual_feedback(diagnosis)
    assert dual["correction_metaphor"]
    assert dual["praise_encouragement"]
    assert dual["correction_metaphor"].startswith("你刚才")
    assert "就像" in dual["correction_metaphor"] and "下次试试" in dual["correction_metaphor"]
    assert len(dual["correction_metaphor"]) <= la._CORRECTION_MAX_CHARS
    # action_plan / praise 动态兜底允许略超短句上限（见 _clamp_report_phrase +25）
    assert len(dual["praise_encouragement"]) <= la._ACTION_MAX_CHARS + 25
    text = generate_feedback(diagnosis)
    assert "【魔法指令】" in text and "【闪光点发现】" in text
    # 实测值复述仍由 clinical helper 负责（不塞进孩子话术）
    clinical = _build_clinical_fallback_markdown(diagnosis)
    assert "0.55" in clinical or "支撑脚横距比例" in clinical
    assert "82.0" in clinical or "82" in clinical
    assert "1.1" in clinical


def test_scorer_to_aigc_e2e_shoulder_ratio_roundtrip():
    """肩宽比实测 → Scorer → AIGC payload 数值一致（废除 PCR cm）。"""
    impact = {
        "t_impact": 0,
        "frames": [
            {
                "left_ankle": [70.0, 200.0, 0.0],
                "right_foot_index": [100.0, 200.0, 0.0],
                "left_shoulder": [100.0, 80.0, 0.0],
                "right_shoulder": [160.0, 80.0, 0.0],
                "left_hip": [100.0, 140.0, 0.0],
                "left_knee": [100.0, 170.0, 0.0],
                "right_hip": [130.0, 140.0, 0.0],
                "right_knee": [130.0, 170.0, 0.0],
                "right_ankle": [130.0, 200.0, 0.0],
            }
        ],
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "ankle_angles_window": [140.0, 140.1, 140.0],
    }
    trajectory = {"max_folding_angle": 78.0, "whipping_velocity": 500.0}
    _, detail = calculate_biomechanical_score(impact, trajectory)
    msg = build_aigc_user_message({"score_detail": detail})
    dist = detail["indicators"]["distance_cm"]
    assert dist["unit"] == "ratio"
    assert dist["value"] is not None
    assert "78.0" in msg or "78" in msg
    raw_json = msg[msg.index("{") :]
    payload = json.loads(raw_json)
    assert abs(float(payload["indicators"]["distance_cm"]["value"]) - float(dist["value"])) < 1e-6


def test_long_format_tags_heuristic_as_estimated():
    records = [
        {
            "school": "S1",
            "classGroup": "C1",
            "studentId": "U1",
            "timestamp": "2026-07-22 10:00:00",
            "type": "realtime",
            "score": 80.0,
            # 故意不给 supportFootDistance → 导出侧启发式 + estimated
        }
    ]
    df = ae.build_long_format_dataframe(records)
    assert len(df) == 1
    assert df.loc[0, "support_foot_distance_provenance"] == ae.PROVENANCE_ESTIMATED
    assert df.loc[0, "knee_flexion_angle_provenance"] == ae.PROVENANCE_ESTIMATED
    assert "support_foot_distance_provenance" in ae.LONG_FORMAT_COLUMNS


def test_long_format_preserves_measured_provenance():
    records = [
        {
            "school": "S1",
            "classGroup": "C1",
            "studentId": "U2",
            "timestamp": "2026-07-22 10:01:00",
            "type": "realtime",
            "score": 90.0,
            "supportFootDistance": 18.2,
            "supportFootDistanceProvenance": "measured",
            "kneeFlexionAngle": 148.0,
            "kneeFlexionAngleProvenance": "measured",
        }
    ]
    df = ae.build_long_format_dataframe(records)
    assert df.loc[0, "support_foot_distance"] == 18.2
    assert df.loc[0, "support_foot_distance_provenance"] == ae.PROVENANCE_MEASURED
    assert df.loc[0, "knee_flexion_angle_provenance"] == ae.PROVENANCE_MEASURED


def test_long_format_legacy_value_without_provenance_is_unknown_not_measured():
    records = [
        {
            "school": "S1",
            "classGroup": "C1",
            "studentId": "U3",
            "timestamp": "2026-07-22 10:02:00",
            "score": 70.0,
            "supportFootDistance": 17.5,  # 历史无 provenance
        }
    ]
    df = ae.build_long_format_dataframe(records)
    assert df.loc[0, "support_foot_distance_provenance"] == ae.PROVENANCE_UNKNOWN


def test_measured_only_filters_estimated_and_unknown():
    records = [
        {
            "school": "S1",
            "classGroup": "C1",
            "studentId": "A",
            "timestamp": "2026-07-22 10:00:00",
            "score": 80.0,
            "supportFootDistance": 18.0,
            "supportFootDistanceProvenance": "measured",
        },
        {
            "school": "S1",
            "classGroup": "C1",
            "studentId": "B",
            "timestamp": "2026-07-22 10:01:00",
            "score": 80.0,
            # 启发式
        },
        {
            "school": "S1",
            "classGroup": "C1",
            "studentId": "C",
            "timestamp": "2026-07-22 10:02:00",
            "score": 80.0,
            "supportFootDistance": 17.5,
            # legacy unknown
        },
    ]
    all_df = ae.build_long_format_dataframe(records, measured_only=False)
    measured_df = ae.build_long_format_dataframe(records, measured_only=True)
    assert len(all_df) == 3
    assert len(measured_df) == 1
    assert measured_df.loc[0, "student_id"] == "A"
    assert measured_df.loc[0, "support_foot_distance_provenance"] == "measured"


if __name__ == "__main__":
    test_calibration_suite_meets_acceptance_gates()
    test_aigc_e2e_user_message_contains_exact_measured_numbers()
    test_aigc_e2e_fallback_quotes_measured_and_blocks_defaults()
    test_long_format_tags_heuristic_as_estimated()
    test_long_format_preserves_measured_provenance()
    test_long_format_legacy_value_without_provenance_is_unknown_not_measured()
    test_measured_only_filters_estimated_and_unknown()
    test_scorer_to_aigc_e2e_shoulder_ratio_roundtrip()
    print("ALL PHASE3 TESTS PASSED (offline subset)")
