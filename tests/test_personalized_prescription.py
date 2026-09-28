# -*- coding: utf-8 -*-

import llm_agent as la
from prescription_context import build_history_context, build_prescription_evidence


def _detail(primary: str = "ankle_rigidity") -> dict:
    indicators = {
        "ankle_rigidity": {
            "value": 24.0,
            "unit": "deg",
            "green_band": [0.0, 10.0],
            "penalty": 8.0 if primary == "ankle_rigidity" else 3.0,
            "status": "RED_DEVIATED",
            "provenance": "measured",
            "confidence": 0.91,
        },
        "max_folding_angle": {
            "value": 62.0,
            "unit": "deg",
            "green_band": [70.0, 100.0],
            "penalty": 8.0 if primary == "max_folding_angle" else 4.0,
            "status": "RED_DEVIATED",
            "provenance": "measured",
        },
        "support_knee_angle": {
            "value": 174.0,
            "unit": "deg",
            "green_band": [135.0, 170.0],
            "penalty": 2.0,
            "status": "YELLOW_APPROACHING",
            "provenance": "measured",
        },
    }
    return {
        "TotalScore": 78.0,
        "indicators": indicators,
        "deductions": [
            {
                "metric_key": key,
                "penalty": item["penalty"],
                "reason": f"{key} test reason",
            }
            for key, item in indicators.items()
        ],
    }


def test_evidence_ranks_top_three_and_keeps_direction():
    evidence = build_prescription_evidence({"score_detail": _detail("ankle_rigidity")})
    top = evidence["topDeductions"]
    assert [item["metricKey"] for item in top] == [
        "ankle_rigidity",
        "max_folding_angle",
        "support_knee_angle",
    ]
    assert top[0]["deviationDirection"] == "偏高"
    assert top[1]["deviationDirection"] == "偏低"
    assert evidence["priorityTarget"]["metricKey"] == "ankle_rigidity"


def test_different_primary_defects_produce_different_protocols():
    ankle = build_prescription_evidence({"score_detail": _detail("ankle_rigidity")})
    fold = build_prescription_evidence({"score_detail": _detail("max_folding_angle")})
    assert ankle["priorityTarget"]["exercise"] == "锁踝触球"
    assert fold["priorityTarget"]["exercise"] == "后摆折叠定格"
    assert ankle["priorityTarget"]["cue"] != fold["priorityTarget"]["cue"]


def test_history_uses_a_and_legacy_but_excludes_low_quality_and_peers_are_per_student():
    def record(student, value, score, *, grade=None, eligible=None):
        row = {
            "studentId": student,
            "school": "学校一",
            "classGroup": "四年级1班-实验A组",
            "score": score,
            "timestamp": f"2026-09-0{int(score) % 8 + 1} 10:00:00",
            "scoreDetail": {"indicators": {"ankle_rigidity": {"value": value}}},
        }
        if grade is not None:
            row["qualityGrade"] = grade
        if eligible is not None:
            row["researchEligible"] = eligible
        return row

    history = build_history_context(
        [
            record("S01", 20, 70, grade="A", eligible=True),
            record("S01", 16, 74),
            record("S01", 99, 10, grade="B", eligible=False),
            record("S02", 12, 80, grade="A", eligible=True),
            record("S02", 14, 82, grade="A", eligible=True),
            record("S03", 10, 84),
        ],
        student_number="S01",
        school="学校一",
        class_group="四年级1班-实验A组",
        current_score_detail={
            "TotalScore": 78,
            "indicators": {"ankle_rigidity": {"value": 18}},
        },
    )
    comparison = history["metricComparisons"]["ankle_rigidity"]
    assert history["sourceCounts"]["excluded"] == 1
    assert comparison["personalMean"] == 20.0
    assert history["personalHistorySource"] == "formal"
    assert comparison["classPeerMean"] == 11.5
    assert comparison["peerStudentCount"] == 2


def test_missing_digits_append_evidence_without_replacing_model_prose():
    diagnosis = {"score_detail": _detail("ankle_rigidity")}
    evidence = build_prescription_evidence(diagnosis)
    diagnosis["prescription_evidence"] = evidence
    original = {
        "overview": "动作节奏稳定，但触球末端仍有改进空间。",
        "biomechanical_analysis": "近端传导基本完整，末端脚型松散削弱了力量输出。",
        "magic_metaphor": "把脚面固定成一块结实的拍面。",
        "action_plan": "触球前先固定脚型。",
        "aigc_source": "llm",
    }
    enriched = la._ensure_report_cites_measurements(original, diagnosis)
    assert enriched["overview"].startswith(original["overview"])
    assert enriched["biomechanical_analysis"].startswith(original["biomechanical_analysis"])
    assert enriched["magic_metaphor"] == original["magic_metaphor"]
    assert "锁踝触球" in enriched["action_plan"]
    assert enriched["aigc_source"] == "llm_augmented"
    assert enriched["postprocess_audit"]["fallbackUsed"] is False


def test_action_that_mentions_secondary_target_is_restricted_to_primary():
    diagnosis = {"score_detail": _detail("ankle_rigidity")}
    diagnosis["prescription_evidence"] = build_prescription_evidence(diagnosis)
    report = {
        "overview": "本次总分78分。",
        "biomechanical_analysis": "脚踝锁定24度是主要问题。",
        "magic_metaphor": "把脚面固定住。",
        "action_plan": "先练后摆折叠，再练脚踝锁定。",
        "aigc_source": "llm",
    }
    enriched = la._ensure_report_cites_measurements(report, diagnosis)
    assert "本次只练脚踝锁定" in enriched["action_plan"]
    assert "后摆折叠" not in enriched["action_plan"]
    assert "action_plan:restricted_to_priority_target" in enriched["postprocess_audit"]["changedFields"]


def test_invalid_llm_json_records_fallback_reason():
    diagnosis = {"score_detail": _detail("ankle_rigidity")}
    diagnosis["prescription_evidence"] = build_prescription_evidence(diagnosis)
    parsed = la._parse_optimal_dual_feedback("not-json", diagnosis)
    assert parsed["aigc_source"] == "fallback"
    assert parsed["fallback_reason"].startswith("invalid_llm_json:")
