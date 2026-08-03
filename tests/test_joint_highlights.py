# -*- coding: utf-8 -*-
"""具身隐喻 joint_highlights：像素坐标 + 红绿灯 + 临床绝对时间戳。"""

from __future__ import annotations

import pytest

from research_models import FeedbackReport, JointHighlight, MetricsData
from error_diagnoser import (
    STATUS_GREEN,
    STATUS_RED,
    STATUS_YELLOW,
    build_joint_highlights,
    status_to_color_code,
)


def _pixel_frame(timestamp_sec: float = 1.0, *, right_knee=(420.0, 310.0), left_ankle=(280.0, 480.0)) -> dict:
    """构造已像素化的帧（对齐 serialize_pose_frame_record）。"""
    return {
        "timestamp_sec": float(timestamp_sec),
        "left_shoulder": [300.0, 120.0, 0.0],
        "right_shoulder": [420.0, 120.0, 0.0],
        "left_hip": [320.0, 260.0, 0.0],
        "right_hip": [400.0, 260.0, 0.0],
        "left_knee": [300.0, 380.0, 0.0],
        "right_knee": [float(right_knee[0]), float(right_knee[1]), 0.0],
        "left_ankle": [float(left_ankle[0]), float(left_ankle[1]), 0.0],
        "right_ankle": [440.0, 470.0, 0.0],
        "left_foot_index": [270.0, 510.0, 0.0],
        "right_foot_index": [450.0, 500.0, 0.0],
        "visibility": {
            "right_knee": 0.95,
            "left_ankle": 0.9,
            "right_ankle": 0.9,
            "left_knee": 0.9,
            "right_hip": 0.9,
            "left_hip": 0.9,
            "left_foot_index": 0.9,
            "right_foot_index": 0.9,
        },
    }


def test_status_to_color_code_maps_traffic_light():
    assert status_to_color_code(STATUS_RED) == "RED"
    assert status_to_color_code(STATUS_YELLOW) == "YELLOW"
    assert status_to_color_code(STATUS_GREEN) == "GREEN"
    assert status_to_color_code("RED_DEVIATED") == "RED"


def test_build_joint_highlights_stamps_absolute_error_timestamp():
    frames = [_pixel_frame(i / 30.0) for i in range(10)]
    frames[5]["right_knee"] = [512.0, 384.0, 0.0]
    frames[5]["left_ankle"] = [200.0, 500.0, 0.0]
    # 后摆折叠极值帧（早于 T0）
    frames[2]["right_knee"] = [480.0, 300.0, 0.0]

    absolute_timestamps = [round(i / 30.0, 4) for i in range(10)]
    indicators = {
        "impact_knee_angle": {
            "value": 120.0,
            "status": STATUS_RED,
            "extreme_frame_index": 5,  # 直腿击球 → T0
        },
        "max_folding_angle": {
            "value": 40.0,
            "status": STATUS_YELLOW,
            "extreme_frame_index": 2,  # 后摆
        },
        "distance_cm": {
            "value": 28.0,
            "status": STATUS_RED,
            "extreme_frame_index": 5,
        },
        "ankle_rigidity": {"value": 1.0, "status": STATUS_GREEN, "extreme_frame_index": 5},
    }

    highlights = build_joint_highlights(
        frames,
        t0_index=5,
        indicators=indicators,
        swing_side="right",
        absolute_timestamps=absolute_timestamps,
        fps=30.0,
    )
    by_name = {h["joint_name"]: h for h in highlights}

    assert "right_knee" in by_name
    # RED(impact) 覆盖同关节 YELLOW(folding) → 时间戳锚定 T0=5
    assert by_name["right_knee"]["color_code"] == "RED"
    assert by_name["right_knee"]["error_timestamp_sec"] == pytest.approx(5 / 30.0, abs=1e-4)
    assert by_name["right_knee"]["error_frame_index"] == 5
    assert by_name["right_knee"]["x"] == 512.0

    assert by_name["left_ankle"]["error_timestamp_sec"] == pytest.approx(5 / 30.0, abs=1e-4)
    assert "error_timestamp_sec" in by_name["right_ankle"]


def test_build_joint_highlights_follow_through_phase_after_t0():
    """随前类错误应锚定 T0 之后的绝对秒。"""
    frames = [_pixel_frame(i / 30.0) for i in range(12)]
    abs_ts = [i / 30.0 for i in range(12)]
    indicators = {
        "whipping_velocity": {
            "value": 100.0,
            "status": STATUS_RED,
            "extreme_frame_index": 8,  # 模拟随前窗
        },
    }
    highlights = build_joint_highlights(
        frames,
        t0_index=5,
        indicators=indicators,
        absolute_timestamps=abs_ts,
        fps=30.0,
    )
    assert len(highlights) == 1
    assert highlights[0]["error_frame_index"] == 8
    assert highlights[0]["error_timestamp_sec"] == pytest.approx(8 / 30.0, abs=1e-4)


def test_build_joint_highlights_normalizes_when_landmarks_are_0_1():
    frame = {
        "timestamp_sec": 0.5,
        "left_shoulder": [0.40, 0.20, 0.0],
        "right_shoulder": [0.55, 0.20, 0.0],
        "left_hip": [0.42, 0.45, 0.0],
        "right_hip": [0.53, 0.45, 0.0],
        "left_knee": [0.40, 0.65, 0.0],
        "right_knee": [0.60, 0.70, 0.0],
        "left_ankle": [0.38, 0.85, 0.0],
        "right_ankle": [0.62, 0.88, 0.0],
        "left_foot_index": [0.37, 0.90, 0.0],
        "right_foot_index": [0.63, 0.92, 0.0],
        "visibility": {k: 0.9 for k in (
            "right_knee", "left_ankle", "right_ankle", "left_knee",
            "right_hip", "left_hip", "left_foot_index", "right_foot_index",
        )},
    }
    highlights = build_joint_highlights(
        [frame],
        t0_index=0,
        indicators={"impact_knee_angle": {"status": STATUS_RED, "extreme_frame_index": 0}},
        swing_side="right",
        frame_width=1280,
        frame_height=720,
        absolute_timestamps=[0.5],
    )
    assert len(highlights) == 1
    h = highlights[0]
    assert h["error_timestamp_sec"] == pytest.approx(0.5)
    assert abs(h["x"] - 0.60 * 1280) < 1e-6


def test_research_models_require_error_timestamp_sec():
    item = JointHighlight(
        joint_name="right_knee",
        x=100.0,
        y=200.0,
        color_code="RED_DEVIATED",
        error_timestamp_sec=1.234,
    )
    assert item.color_code == "RED"
    assert item.error_timestamp_sec == pytest.approx(1.234)

    with pytest.raises(Exception):
        JointHighlight(
            joint_name="right_knee",
            x=100.0,
            y=200.0,
            color_code="RED",
        )

    metrics = MetricsData(joint_highlights=[item.model_dump()])
    report = FeedbackReport(TotalScore=72.5, joint_highlights=metrics.joint_highlights)
    assert report.joint_highlights[0]["error_timestamp_sec"] == pytest.approx(1.234)
