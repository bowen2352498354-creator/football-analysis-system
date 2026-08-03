# -*- coding: utf-8 -*-
"""B 组复盘双关键帧：极值筛选 + student_review_summary。"""

from __future__ import annotations

from db import build_comparison_frames, student_review_summary


def _attempt(
    number: int,
    score: float,
    *,
    image: str | None = None,
    errors: list[str] | None = None,
    red_indicator: str | None = None,
) -> dict:
    indicators = {}
    if red_indicator:
        indicators[red_indicator] = {"value": 99.0, "status": "RED"}
    return {
        "attemptNumber": number,
        "timestamp": 1_700_000_000_000 + number * 1000,
        "impactFrameBase64": image,
        "reportData": {
            "score": score,
            "impactFrameImage": image,
            "biomechanicalErrors": errors or [],
            "scoreDetail": {"indicators": indicators} if indicators else {},
            "painPoint": "占位痛点文案",
        },
    }


def test_build_comparison_frames_skips_when_fewer_than_two():
    assert build_comparison_frames([_attempt(1, 80.0, image="data:image/jpeg;base64,AAA")]) is None
    assert build_comparison_frames([]) is None
    assert build_comparison_frames(None) is None


def test_build_comparison_frames_picks_best_and_improve():
    attempts = [
        _attempt(1, 72.0, image="data:image/jpeg;base64,LOW", errors=["支撑脚位置偏离"]),
        _attempt(2, 88.5, image="data:image/jpeg;base64,BEST"),
        _attempt(3, 52.0, image="data:image/jpeg;base64,WORST", red_indicator="distance_cm"),
    ]
    frames = build_comparison_frames(attempts)
    assert frames is not None
    assert frames["best"]["score"] == 88.5
    assert frames["best"]["attempt_id"] == 2
    assert frames["best"]["image_url"] == "data:image/jpeg;base64,BEST"
    assert frames["improve"]["score"] == 52.0
    assert frames["improve"]["attempt_id"] == 3
    assert frames["improve"]["image_url"] == "data:image/jpeg;base64,WORST"
    assert frames["improve"]["main_error"] == "支撑脚过宽"


def test_build_comparison_frames_normalizes_bare_base64():
    attempts = [
        _attempt(1, 40.0, image="YWJjZGVmZ2hpams=" * 4),
        _attempt(2, 90.0, image="data:image/png;base64,ZZZ"),
    ]
    frames = build_comparison_frames(attempts)
    assert frames is not None
    assert frames["best"]["image_url"].startswith("data:image/png;base64,")
    assert frames["improve"]["image_url"].startswith("data:image/jpeg;base64,")


def test_student_review_summary_from_sessions():
    sessions = [
        {
            "id": "sess-a",
            "studentId": "B004",
            "timestamp": 1_754_185_200_000,  # ~2025-08-03 local depends; use explicit date field
            "session_date": "2026-08-03",
            "attempts": [
                _attempt(1, 60.0, image="data:image/jpeg;base64,A", errors=["膝关节过度屈曲"]),
                _attempt(2, 91.0, image="data:image/jpeg;base64,B"),
            ],
        },
        {
            "id": "sess-b",
            "studentId": "B004",
            "session_date": "2026-08-02",
            "timestamp": 1_754_000_000_000,
            "attempts": [_attempt(1, 10.0, image="data:image/jpeg;base64,OLD")],
        },
    ]
    summary = student_review_summary(
        "B004",
        session_date="2026-08-03",
        sessions=sessions,
        global_records=[],
    )
    assert summary["success"] is True
    assert summary["attempt_count"] == 2
    assert summary["comparison_available"] is True
    assert summary["comparison_frames"]["best"]["score"] == 91.0
    assert summary["comparison_frames"]["improve"]["score"] == 60.0
    assert summary["comparison_frames"]["improve"]["main_error"] == "膝关节过度屈曲"


def test_student_review_summary_global_fallback():
    records = [
        {
            "id": "r1",
            "studentId": "a101",
            "type": "delayed",
            "score": 55.0,
            "testDate": "2026-08-03",
            "timestamp": "2026-08-03 08:00:00",
            "impactFrameBase64": "data:image/jpeg;base64,X",
            "biomechanicalErrors": ["支撑脚位置偏离"],
            "is_deleted": False,
        },
        {
            "id": "r2",
            "studentId": "a101",
            "type": "delayed",
            "score": 82.0,
            "testDate": "2026-08-03",
            "timestamp": "2026-08-03 08:10:00",
            "impactFrameBase64": "data:image/jpeg;base64,Y",
            "biomechanicalErrors": [],
            "is_deleted": False,
        },
    ]
    summary = student_review_summary(
        "a101",
        session_date="2026-08-03",
        sessions=[],
        global_records=records,
    )
    assert summary["comparison_available"] is True
    assert summary["comparison_frames"]["best"]["score"] == 82.0
    assert summary["comparison_frames"]["improve"]["main_error"] == "支撑脚过宽"
