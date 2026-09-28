# -*- coding: utf-8 -*-
"""Phase 5 regression tests for the research training loop."""

from __future__ import annotations

import json

import academic_exporter as ae
import word_reporter
from experiment_ledger import (
    canonical_group,
    normalize_timepoint,
    read_attempts,
    record_attempt,
)


def _gate(grade: str) -> dict:
    eligible = grade == "A"
    return {
        "grade": grade,
        "score": 92.0 if eligible else 38.0,
        "researchEligible": eligible,
        "issues": []
        if eligible
        else [{"code": "pose_visibility", "label": "关键关节可见度不足"}],
    }


def test_group_and_timepoint_normalization():
    assert canonical_group("A") == "GROUP_A_REALTIME"
    assert canonical_group("GROUP_B") == "GROUP_B_DELAYED"
    assert canonical_group("GROUP_C_CONTROL") == "GROUP_C_CONTROL"
    assert normalize_timepoint("3") == "T3"
    assert normalize_timepoint("bad") == "T0"


def test_attempt_ledger_is_idempotent_and_counts_invalid(tmp_path):
    path = str(tmp_path / "ledger.json")
    common = {
        "student_number": "S001",
        "school": "School",
        "class_group": "Class-C",
        "experimental_group": "GROUP_C_CONTROL",
        "timepoint": "T2",
        "lesson_id": "lesson-01",
        "source": "webcam",
        "planned_attempts": 2,
        "ledger_path": path,
    }
    first = record_attempt(session_id="session-valid", quality_gate=_gate("A"), score=86.0, **common)
    duplicate = record_attempt(session_id="session-valid", quality_gate=_gate("A"), score=86.0, **common)
    rejected = record_attempt(session_id="session-invalid", quality_gate=_gate("C"), score=None, **common)

    assert first["summary"]["validAttempts"] == 1
    assert duplicate["summary"]["totalAttempts"] == 1
    assert rejected["summary"] == {
        "studentNumber": "S001",
        "experimentalGroup": "GROUP_C_CONTROL",
        "timepoint": "T2",
        "lessonId": "lesson-01",
        "totalAttempts": 2,
        "validAttempts": 1,
        "invalidAttempts": 1,
        "plannedAttempts": 2,
        "remainingAttempts": 1,
        "doseComplete": False,
    }
    rows = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert rows[1]["score"] is None
    assert rows[1]["invalidReasons"][0]["code"] == "pose_visibility"

    snapshot = read_attempts(
        student_number="S001",
        experimental_group="C",
        timepoint="T2",
        lesson_id="lesson-01",
        planned_attempts=2,
        ledger_path=path,
    )
    assert snapshot["count"] == 2
    assert snapshot["summary"]["remainingAttempts"] == 1


def test_academic_export_uses_explicit_group_timepoint_and_quality_gate():
    exporter = ae.AcademicDataExporter.from_global_records(
        [
            {
                "studentId": "C001",
                "classGroup": "Class-C",
                "experimental_group": "GROUP_C_CONTROL",
                "timepoint": "T4",
                "score": 82.0,
                "kneeFlexionAngle": 146.0,
                "qualityGrade": "A",
                "researchEligible": True,
            },
            {
                "studentId": "C001",
                "classGroup": "Class-C",
                "experimental_group": "GROUP_C_CONTROL",
                "timepoint": "T4",
                "score": 10.0,
                "qualityGrade": "C",
                "researchEligible": False,
            },
        ]
    )
    assert len(exporter.shot_logs) == 1
    assert exporter.shot_logs[0]["experimental_group"] == 3
    assert exporter.shot_logs[0]["timepoint"] == "T4"


def test_official_json_export_rejects_legacy_ungraded_rows(tmp_path):
    path = tmp_path / "global_training_db.json"
    path.write_text(
        json.dumps(
            [
                {"studentId": "legacy", "score": 99, "timepoint": "T0"},
                {
                    "studentId": "C002",
                    "classGroup": "Class-C",
                    "experimental_group": "GROUP_C_CONTROL",
                    "timepoint": "T1",
                    "score": 84,
                    "qualityGrade": "A",
                    "researchEligible": True,
                },
            ]
        ),
        encoding="utf-8",
    )
    exporter = ae.AcademicDataExporter.from_global_json(str(path))
    assert len(exporter.shot_logs) == 1
    assert exporter.shot_logs[0]["anonymous_id"] == "C002"
    assert exporter.shot_logs[0]["experimental_group"] == 3


def test_control_mode_has_its_own_archive_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(word_reporter, "REPORT_ROOT_DIR", str(tmp_path))
    target = word_reporter.build_target_directory("control", "School", "Class-C", "C001")
    assert target.exists()
    assert target.parts[-3] == "无反馈采集"
