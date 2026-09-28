from __future__ import annotations

import json

import academic_exporter as ae
from db import compare_cohorts
from measurement_validation import validate_measurement_pairs
from research_integrity import (
    build_intervention_audit,
    build_measurement_provenance_summary,
    build_version_metadata,
    filter_individual_attempts_by_date,
    summarize_individual_attempts,
)


def _formal_record(student: str, cohort: str, score: float, **extra):
    record = {
        "id": f"{cohort}-{student}-{score}",
        "timestamp": "2026-09-01 10:00:00",
        "testDate": "2026-09-01",
        "timepoint": "T1",
        "school": "School-1",
        "classGroup": cohort,
        "studentId": student,
        "experimentalGroup": "GROUP_A_REALTIME",
        "qualityGrade": "A",
        "researchEligible": True,
        "score": score,
        "protocolVersion": "THESIS_2026_DRAFT_V1",
        "quantified5dScores": {
            "approach_rhythm": score / 5,
            "support_stability": score / 5,
            "backswing_folding": score / 5,
            "ankle_rigidity": score / 5,
            "whipping_velocity": score / 5,
        },
    }
    record.update(extra)
    return record


def test_individual_summary_uses_only_explicit_formal_attempts_and_metrics():
    rows = [
        _formal_record(
            "S1",
            "Class-A",
            70,
            scoreDetail={
                "indicators": {
                    "impact_knee_angle": {
                        "value": 150,
                        "unit": "deg",
                        "status": "GREEN_OPTIMAL",
                        "penalty": 0,
                        "provenance": "measured",
                    }
                }
            },
            biomechanicalErrors=[],
        ),
        _formal_record(
            "S1",
            "Class-A",
            82,
            timestamp="2026-09-08 10:00:00",
            scoreDetail={
                "indicators": {
                    "impact_knee_angle": {
                        "value": 156,
                        "unit": "deg",
                        "status": "GREEN_OPTIMAL",
                        "penalty": 0,
                        "provenance": "measured",
                    }
                }
            },
            biomechanicalErrors=["支撑脚位置偏离"],
        ),
        {
            **_formal_record("S1", "Class-A", 99),
            "qualityGrade": "B",
            "researchEligible": False,
        },
    ]
    result = summarize_individual_attempts(rows)
    assert result["formalAttemptCount"] == 2
    assert result["excludedAttemptCount"] == 1
    assert result["scoreSummary"]["change"] == 12
    assert result["metricTrends"]["impact_knee_angle"]["change"] == 6
    assert result["errorRates"]["支撑脚位置偏离"] == 0.5
    assert result["fiveDimensionScores"]["support_stability"] == 15.2
    assert result["period"] == {"start": "2026-09-01", "end": "2026-09-01"}


def test_individual_summary_date_range_is_inclusive_and_validated():
    rows = [
        _formal_record("S1", "Class-A", 70, testDate="2026-09-01"),
        _formal_record("S1", "Class-A", 76, testDate="2026-09-05"),
        _formal_record("S1", "Class-A", 82, testDate="2026-09-09"),
    ]
    selected = filter_individual_attempts_by_date(rows, "2026-09-05", "2026-09-09")
    assert [row["score"] for row in selected] == [76, 82]

    try:
        filter_individual_attempts_by_date(rows, "2026-09-10", "2026-09-01")
    except ValueError as exc:
        assert "dateFrom" in str(exc)
    else:
        raise AssertionError("reversed date range must be rejected")


def test_cohort_comparison_is_student_weighted_and_strictly_scoped():
    records = [
        _formal_record("A1", "Class-A", 100),
        _formal_record("A1", "Class-A", 100, id="a1-second"),
        _formal_record("A2", "Class-A", 0),
        _formal_record("B1", "Class-B", 60),
        _formal_record("B2", "Class-B", 80),
        {**_formal_record("A3", "Class-A", 100), "qualityGrade": "B", "researchEligible": False},
        _formal_record("X1", "Class-A", 100, school="Other-School"),
    ]
    result = compare_cohorts(
        "Class-A",
        "Class-B",
        records=records,
        school="School-1",
        strict_quality=True,
    )
    assert result["sufficient_data"] is True
    assert result["sample_counts"] == {"a": 2, "b": 2}
    assert result["attempt_counts"] == {"a": 3, "b": 2}
    assert result["descriptive"]["cohort_a"]["mean"] == 50
    assert result["filter_scope"]["aggregation_unit"] == "student"


def test_provenance_inventory_and_intervention_audit():
    summary = build_measurement_provenance_summary(
        {
            "indicators": {
                "impact_knee_angle": {"provenance": "measured", "method": "xy_2d"},
                "hip_torsion_angle": {"provenance": "estimated", "method": "xz_proxy"},
            }
        }
    )
    assert summary["counts"] == {"measured": 1, "estimated": 1}
    assert summary["allFormalMeasurements"] is False

    control = build_intervention_audit(
        "GROUP_C_CONTROL",
        feedback_suppressed=True,
        supplied={"feedbackShown": False, "operatorPreviewOnly": True},
    )
    assert control["protocolDeviation"] is False
    leaked = build_intervention_audit(
        "GROUP_C_CONTROL",
        feedback_suppressed=True,
        supplied={"feedbackShown": True},
    )
    assert "C_FEEDBACK_LEAK" in leaked["violations"]


def test_measurement_validation_reports_error_agreement_and_frame_accuracy():
    result = validate_measurement_pairs(
        [
            {
                "metric": "impact_knee_angle",
                "systemValue": 151,
                "referenceValue": 150,
                "systemLevel": "GREEN",
                "referenceLevel": "GREEN",
                "systemFrame": 31,
                "referenceFrame": 30,
            },
            {
                "metric": "impact_knee_angle",
                "systemValue": 157,
                "referenceValue": 155,
                "systemLevel": "GREEN",
                "referenceLevel": "YELLOW",
                "systemFrame": 42,
                "referenceFrame": 40,
            },
        ]
    )
    metric = result["metrics"]["impact_knee_angle"]
    assert metric["mae"] == 1.5
    assert metric["levelAccuracy"] == 0.5
    assert metric["impactWithin1FrameRate"] == 0.5


def test_formal_export_requires_versions_and_writes_exclusion_reasons(tmp_path, monkeypatch):
    monkeypatch.setattr(ae, "EXPORT_DIR", str(tmp_path))
    versioned = _formal_record("S1", "Class-A", 80)
    versioned.update(build_version_metadata({"scoring_engine": "test-engine"}))
    legacy = _formal_record("S2", "Class-A", 90)
    legacy.pop("protocolVersion")

    result = ae.export_academic_matrix([versioned, legacy])
    assert result["success"] is True
    assert result["rowCount"] == 1
    exclusions = json.loads(
        open(result["exclusionLogPath"], encoding="utf-8").read()
    )
    assert exclusions[0]["studentId"] == "S2"
    assert "legacy_unknown_protocol_version" in exclusions[0]["reasons"]
