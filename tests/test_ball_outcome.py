# -*- coding: utf-8 -*-
"""V3.11：出球初速度 / 发射仰角 calculate_ball_outcome 单测。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biomech_primitives import (
    DEFAULT_EMPIRICAL_PCR,
    calculate_ball_outcome,
)


def test_ball_outcome_pcr_horizontal_flight():
    """水平飞行：仰角≈0，速度由 PCR 位移决定。"""
    # 每帧 +84 px（= 默认球径）→ 每帧 0.21 m；3 间隔 @30fps → dt=0.1s
    # 总位移 0.63 m → 6.3 m/s → 22.68 km/h
    traj = [(0.0, 100.0), (84.0, 100.0), (168.0, 100.0), (252.0, 100.0)]
    out = calculate_ball_outcome(traj, fps=30.0)
    assert out["ok"] is True
    assert out["scale_method"] == "default_pcr"
    assert abs(float(out["ball_speed_kmh"]) - 22.68) < 0.05
    assert abs(float(out["launch_angle_deg"])) < 1.0


def test_ball_outcome_upward_launch_angle():
    """图像 Y 向下：终点 y 更小 → 仰角为正。"""
    traj = [(0.0, 200.0), (30.0, 180.0), (60.0, 160.0), (90.0, 140.0)]
    out = calculate_ball_outcome(traj, fps=30.0)
    assert out["ok"] is True
    assert float(out["launch_angle_deg"]) > 0.0


def test_ball_outcome_homography_calibrator():
    """单位单应性：像素即米；验证标定路径。"""
    H = np.eye(3, dtype=np.float64)
    traj = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    out = calculate_ball_outcome(traj, fps=30.0, calibrator=H)
    assert out["ok"] is True
    assert out["scale_method"] == "homography"
    # 3 m / 0.1 s = 30 m/s = 108 km/h
    assert abs(float(out["ball_speed_kmh"]) - 108.0) < 0.1


def test_ball_outcome_callable_calibrator():
    def pix2m(x, y):
        return (x * 0.01, y * 0.01)

    traj = [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0), (300.0, 0.0)]
    out = calculate_ball_outcome(traj, fps=30.0, calibrator=pix2m)
    assert out["ok"] is True
    assert out["scale_method"] == "calibrator_callable"
    assert abs(float(out["ball_speed_kmh"]) - 108.0) < 0.1


def test_ball_outcome_insufficient_points():
    out = calculate_ball_outcome([(1.0, 2.0)], fps=30.0)
    assert out["ok"] is False
    assert out["ball_speed_kmh"] is None
    assert out["launch_angle_deg"] is None


def test_ball_outcome_skips_none_holes():
    traj = [(0.0, 100.0), None, (168.0, 100.0), (252.0, 100.0)]
    out = calculate_ball_outcome(traj, fps=30.0)
    assert out["ok"] is True
    assert out["sample_count"] == 3


def test_pipeline_inject_ball_outcome():
    from shot_analysis_service import ShotAnalysisPipeline

    pipe = ShotAnalysisPipeline.__new__(ShotAnalysisPipeline)
    pipe.t_impact = 10
    pipe.ball_speed_kmh = 45.5
    pipe.launch_angle_deg = 12.3
    pipe.ball_outcome_meta = {
        "ok": True,
        "scale_method": "default_pcr",
        "displacement_m": 1.2,
        "dt_sec": 0.1,
        "reason": "ok",
    }
    detail = pipe.inject_ball_outcome_into_score_detail({"TotalScore": 88.0})
    assert detail["ball_speed_kmh"] == 45.5
    assert detail["launch_angle_deg"] == 12.3
    assert detail["indicators"]["ball_speed_kmh"]["value"] == 45.5
    assert detail["indicators"]["ball_speed_kmh"]["unit"] == "km/h"
    assert detail["indicators"]["launch_angle_deg"]["unit"] == "deg"


def test_collect_window_and_finalize_with_synthetic_track():
    from shot_analysis_service import ShotAnalysisPipeline

    pipe = ShotAnalysisPipeline.__new__(ShotAnalysisPipeline)
    pipe.t_impact = 2
    pipe._video_fps = 30.0
    pipe._fixed_frame_dt = 1.0 / 30.0
    pipe._field_calibrator = None
    pipe.ball_speed_kmh = None
    pipe.launch_angle_deg = None
    pipe.ball_outcome_meta = {}
    # 6 帧轨迹；T0=2 → 取 2,3,4,5
    pipe._trajectory_ball_px = [
        (0.0, 100.0),
        (10.0, 100.0),
        (0.0, 100.0),
        (84.0, 100.0),
        (168.0, 100.0),
        (252.0, 100.0),
    ]
    pipe._trajectory_ball_diameter_px = [84.0] * 6
    pipe._finalize_ball_outcome()
    assert pipe.ball_speed_kmh is not None
    assert abs(float(pipe.ball_speed_kmh) - 22.68) < 0.1
