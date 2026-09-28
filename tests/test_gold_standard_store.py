from __future__ import annotations

import hashlib
import json
from pathlib import Path

from empirical_thresholds import (
    classify_impact_knee_angle,
    get_impact_knee_thresholds,
    get_threshold_profile,
)
from gold_standard_store import (
    commit_preview,
    generate_validation_report,
    latest_validation_report,
    list_datasets,
    preview_import,
)


def _csv_payload(video_hash: str) -> bytes:
    header = (
        "sample_id,attempt_id,video_hash,source_type,annotator_id,metric_key,"
        "reference_value,unit,reference_frame,fps,swing_leg,camera_view,annotation_version\n"
    )
    rows = [
        f"S1,ATTEMPT-1,{video_hash},kinovea,R1,impact_knee_angle,150,deg,30,30,right,lateral,KINOVEA_V1",
        f"S1,ATTEMPT-1,{video_hash},manual,R2,impact_knee_angle,152,deg,32,30,right,lateral,MANUAL_V1",
        f"S1,ATTEMPT-1,{video_hash},kinovea,R1,impact_knee_angle,151,deg,31,30,right,lateral,KINOVEA_V1",
    ]
    return (header + "\n".join(rows) + "\n").encode("utf-8")


def test_frozen_impact_knee_boundaries_are_single_source():
    assert get_impact_knee_thresholds() == (140.0, 160.0, 130.0, 170.0, 150.0)
    assert get_threshold_profile()["thresholdVersion"] == "THESIS_KNEE_140_160_V1"
    expected = {
        129.9: "RED_DEVIATED",
        130.0: "YELLOW_APPROACHING",
        139.9: "YELLOW_APPROACHING",
        140.0: "GREEN_OPTIMAL",
        160.0: "GREEN_OPTIMAL",
        160.1: "YELLOW_APPROACHING",
        170.0: "YELLOW_APPROACHING",
        170.1: "RED_DEVIATED",
    }
    assert {value: classify_impact_knee_angle(value) for value in expected} == expected


def test_gold_standard_preview_commit_and_validation_report(tmp_path: Path):
    video_hash = hashlib.sha256(b"video-1").hexdigest()
    preview = preview_import("kinovea.csv", _csv_payload(video_hash), store_root=tmp_path)
    assert preview["canCommit"] is True
    assert preview["validRowCount"] == 2
    assert preview["invalidRowCount"] == 1
    assert preview["duplicateCount"] == 1

    committed = commit_preview(preview["previewToken"], store_root=tmp_path)
    manifest = committed["dataset"]
    assert manifest["immutable"] is True
    assert manifest["thresholdVersion"] == "THESIS_KNEE_140_160_V1"
    assert Path(manifest["sourcePath"]).read_bytes() == _csv_payload(video_hash)
    assert len(list_datasets(store_root=tmp_path)) == 1

    system_records = [
        {
            "id": "ATTEMPT-1",
            "videoHash": video_hash,
            "scoreDetail": {
                "t_impact": 31,
                "indicators": {
                    "impact_knee_angle": {
                        "value": 151,
                        "status": "GREEN_OPTIMAL",
                    }
                },
            },
        }
    ]
    report = generate_validation_report(
        manifest["datasetId"], system_records, store_root=tmp_path
    )
    metric = report["metrics"]["impact_knee_angle"]
    assert report["alignedSampleCount"] == 1
    assert report["excludedSampleCount"] == 0
    assert metric["mae"] == 0.0
    assert metric["impactFrameMae"] == 0.0
    assert metric["levelAccuracy"] == 1.0
    assert report["interRater"]["multiAnnotatorSampleCount"] == 1
    assert report["interRater"]["meanAbsoluteDifference"] == 2.0
    assert report["hashVerification"] == {
        "verifiedCount": 1,
        "unavailableCount": 0,
        "mismatchCount": 0,
        "policy": "mismatch_excluded; unavailable_retained_and_flagged",
    }
    assert latest_validation_report(manifest["datasetId"], store_root=tmp_path) == report


def test_front_view_knee_annotation_is_rejected(tmp_path: Path):
    video_hash = "a" * 64
    content = _csv_payload(video_hash).replace(b",lateral,", b",front,")
    preview = preview_import("manual.csv", content, store_root=tmp_path)
    assert preview["canCommit"] is False
    assert preview["validRowCount"] == 0
    assert any("正面机位" in message for row in preview["invalidRows"] for message in row["errors"])
