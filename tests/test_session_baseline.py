# -*- coding: utf-8 -*-
"""实验防干扰：SessionCheckpoint / SessionMetadataStore 基线水印契约。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from session_baseline import (
    SessionMetadataStore,
    stamp_baseline_watermark,
)
import academic_exporter as ae
from shot_analysis_service import ShotAnalysisPipeline


def test_lock_baseline_generates_unique_session_id():
    store = SessionMetadataStore()
    assert store.session_locked is False
    cp = store.lock_baseline(
        class_id="class-4-1",
        camera_height_cm=120.0,
        calibrator_status="homography_ok",
    )
    assert store.session_locked is True
    assert cp.session_id
    assert cp.class_id == "class-4-1"
    assert cp.camera_height_cm == 120.0
    assert cp.calibrator_status == "homography_ok"
    assert cp.locked_at


def test_stamp_payload_trusted_when_locked():
    store = SessionMetadataStore()
    store.lock_baseline(
        class_id="C1",
        camera_height_cm=110.5,
        calibrator_status="locked",
        session_id="baseline-fixed-001",
    )
    detail = stamp_baseline_watermark(
        {"TotalScore": 88.0},
        analysis_session_id="analysis-abc",
        store=store,
    )
    assert detail["baseline_session_id"] == "baseline-fixed-001"
    assert detail["class_id"] == "C1"
    assert detail["camera_height_cm"] == 110.5
    assert detail["calibrator_status"] == "locked"
    assert detail["is_baseline_trusted"] is True
    assert detail["analysis_session_id"] == "analysis-abc"
    assert detail["session_checkpoint"]["session_id"] == "baseline-fixed-001"


def test_stamp_payload_untrusted_when_unlocked():
    store = SessionMetadataStore()
    detail = stamp_baseline_watermark({"TotalScore": 70.0}, store=store)
    assert detail["is_baseline_trusted"] is False
    assert detail["calibrator_status"] == "unlocked"
    assert detail["baseline_session_id"] is None


def test_warn_if_unlocked_prints_strong_warning(capsys):
    store = SessionMetadataStore()
    ok = store.warn_if_unlocked()
    assert ok is False
    captured = capsys.readouterr().out
    assert "[Baseline Warning]: Analysis running without locked session baseline!" in (
        captured
    )


def test_pipeline_stamp_session_baseline_uses_store_api():
    """ShotAnalysisPipeline.stamp_session_baseline 与 stamp_baseline_watermark 同源。"""
    pipe = ShotAnalysisPipeline(
        session_id="analysis-1",
        source="file",
        push_fn=lambda _msg: None,
    )
    # 未锁定全局基线时：仍应写出 is_baseline_trusted=False（不抛异常）
    stamped = pipe.stamp_session_baseline({"TotalScore": 91.0})
    assert "is_baseline_trusted" in stamped
    assert stamped["analysis_session_id"] == "analysis-1"
    assert stamped["TotalScore"] == 91.0


def test_spss_long_format_includes_baseline_columns():
    records = [
        {
            "school": "S1",
            "classGroup": "四年级1班",
            "studentId": "B001",
            "timestamp": "2026-08-04 09:00:00",
            "type": "realtime",
            "score": 85.0,
            "supportFootDistance": 0.55,
            "supportFootDistanceProvenance": "calibrated",
            "kneeFlexionAngle": 150.0,
            "kneeFlexionAngleProvenance": "measured",
            "baseline_session_id": "bl-spss-1",
            "class_id": "class-4-1",
            "camera_height_cm": 125.0,
            "calibrator_status": "homography_ok",
            "is_baseline_trusted": True,
        },
        {
            "school": "S1",
            "classGroup": "四年级1班",
            "studentId": "B002",
            "timestamp": "2026-08-04 09:05:00",
            "type": "realtime",
            "score": 70.0,
            # 无基线水印 → trusted=0
        },
    ]
    df = ae.build_long_format_dataframe(records)
    assert "baseline_session_id" in ae.LONG_FORMAT_COLUMNS
    assert "is_baseline_trusted" in ae.LONG_FORMAT_COLUMNS
    assert df.loc[0, "baseline_session_id"] == "bl-spss-1"
    assert df.loc[0, "class_id"] == "class-4-1"
    assert float(df.loc[0, "camera_height_cm"]) == 125.0
    assert df.loc[0, "calibrator_status"] == "homography_ok"
    assert int(df.loc[0, "is_baseline_trusted"]) == 1
    assert int(df.loc[1, "is_baseline_trusted"]) == 0


def test_unlock_clears_trusted_flag():
    store = SessionMetadataStore()
    store.lock_baseline(class_id="X", calibrator_status="ok")
    store.unlock_baseline()
    assert store.session_locked is False
    stamped = stamp_baseline_watermark({}, store=store)
    assert stamped["is_baseline_trusted"] is False
