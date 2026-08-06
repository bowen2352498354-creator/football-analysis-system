# -*- coding: utf-8 -*-
"""V2.6：CubicSpline 120Hz 多模态亚像素触球锁帧单测。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pose_tracker as pt


def _synth_contact_window(
    n: int = 25,
    t_hit: int = 12,
    *,
    moving_ball: bool = True,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """构造触球窗：踝逼近→急刹，球在 t_hit 后突然加速，距离谷在 t_hit。"""
    ball_x: list[float] = []
    ball_y: list[float] = []
    ankle_x: list[float] = []
    ankle_y: list[float] = []
    for i in range(n):
        # 摆动踝：沿 +x 逼近球心 (0,0)；触球前减速逼近，触球后急刹小幅随前
        if i <= t_hit:
            ax = -40.0 * (1.0 - (i / max(1, t_hit)) ** 1.2)
        else:
            ax = 0.5 * (i - t_hit)
        ay = 0.0
        ankle_x.append(float(ax))
        ankle_y.append(float(ay))

        if moving_ball:
            # 球：触球前静止，触球后沿 +x 突然飞出（速度阶跃）
            if i < t_hit:
                bx, by = 0.0, 0.0
            else:
                bx = 25.0 * (i - t_hit) ** 1.15
                by = 0.0
        else:
            bx, by = 0.0, 0.0
        ball_x.append(float(bx))
        ball_y.append(float(by))
    return ball_x, ball_y, ankle_x, ankle_y


def test_find_precise_impact_subframe_locks_near_true_contact():
    t_hit = 12
    window_start = 40
    bx, by, ax, ay = _synth_contact_window(n=25, t_hit=t_hit, moving_ball=True)
    locked = pt.find_precise_impact_subframe(
        bx, by, ax, ay, window_start_frame=window_start
    )
    assert locked is not None
    # 允许 ±1 帧（亚像素四舍五入到离散 30fps）
    assert abs(int(locked) - (window_start + t_hit)) <= 1


def test_find_precise_impact_subframe_static_ball_still_works():
    """无 YOLO 球轨迹（静止代理球心）时仍应靠踝急刹+距离极小锁帧。"""
    t_hit = 10
    bx, by, ax, ay = _synth_contact_window(n=21, t_hit=t_hit, moving_ball=False)
    locked = pt.find_precise_impact_subframe(bx, by, ax, ay, window_start_frame=0)
    assert locked is not None
    assert abs(int(locked) - t_hit) <= 2


def test_find_precise_impact_subframe_dirty_data_returns_none():
    """全 NaN / 过短序列：必须返回 None，绝不抛异常。"""
    assert pt.find_precise_impact_subframe([], [], [], []) is None
    assert (
        pt.find_precise_impact_subframe(
            [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]
        )
        is None
    )
    nan = [float("nan")] * 12
    assert pt.find_precise_impact_subframe(nan, nan, nan, nan) is None


def test_locate_impact_frame_prefers_cubic_or_falls_back():
    """端到端：有清晰鞭打峰+球-踝接近时，锁帧成功且质量为升频或抛物线。"""
    n = 60
    t_true = 30
    omega = [0.0] * n
    # 鞭打峰落在真触球附近
    for i in range(n):
        omega[i] = 200.0 * np.exp(-0.5 * ((i - t_true) / 3.0) ** 2)

    ankles = []
    balls = []
    for i in range(n):
        if i <= t_true:
            ax = -30.0 * (1.0 - i / t_true)
        else:
            ax = 0.8 * (i - t_true)
        ankles.append((float(ax), 0.0))
        if i < t_true:
            balls.append((0.0, 0.0))
        else:
            balls.append((20.0 * (i - t_true), 0.0))

    t_lock, quality = pt.locate_impact_frame_with_quality(omega, ankles, balls)
    assert 0 <= int(t_lock) < n
    assert abs(int(t_lock) - t_true) <= 2
    assert quality in (
        pt.T0_QUALITY_CUBIC_120HZ,
        pt.T0_QUALITY_SUBFRAME,
        pt.T0_QUALITY_FALLBACK,
    )


def test_locate_impact_frame_survives_constant_junk_series():
    """脏常数轨迹：管线不崩溃，仍返回安全索引。"""
    omega = [1.0, 2.0, 50.0, 3.0, 1.0] + [0.0] * 10
    ankles = [(0.0, 0.0)] * len(omega)
    balls = [(0.0, 0.0)] * len(omega)
    t_lock = pt.locate_impact_frame(omega, ankles, balls)
    assert isinstance(t_lock, int)
    assert 0 <= t_lock < len(omega)


def test_find_precise_subframe_deterministic():
    """同一输入连续调用结果位级一致。"""
    bx, by, ax, ay = _synth_contact_window()
    first = pt.find_precise_impact_subframe(bx, by, ax, ay, window_start_frame=5)
    for _ in range(50):
        assert (
            pt.find_precise_impact_subframe(bx, by, ax, ay, window_start_frame=5)
            == first
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_partial_nan_interpolated_not_crash(bad: float):
    bx, by, ax, ay = _synth_contact_window(n=16, t_hit=8)
    bx[3] = bad
    ay[7] = bad
    locked = pt.find_precise_impact_subframe(bx, by, ax, ay, window_start_frame=0)
    # 局部脏点应被填补后仍可锁，或安全返回 None——绝不能抛
    assert locked is None or isinstance(locked, int)
