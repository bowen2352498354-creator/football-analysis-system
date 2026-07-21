# -*- coding: utf-8 -*-
"""V3.1 Sprint 2：AutoShotCaptureEngine 滚动缓冲 / FSM / 切片窗口单测。"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from workers.auto_shot_capture import (
    AutoShotCaptureEngine,
    POST_IMPACT_FRAMES,
    PRE_IMPACT_FRAMES,
    ROLLING_BUFFER_MAXLEN,
    RollingBuffer,
    ShotFsmState,
)


def _fake_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)


def test_rolling_buffer_maxlen_keeps_only_recent_frames():
    engine = AutoShotCaptureEngine(rolling_maxlen=150)
    for i in range(200):
        engine.push_frame(_fake_frame(i), i)
    assert engine.buffer_len == ROLLING_BUFFER_MAXLEN
    # 最旧应已被挤出：当前窗口为 [50, 199]
    assert engine._buffer[0].frame_index == 50
    assert engine._buffer[-1].frame_index == 199


def test_rolling_buffer_eviction_releases_ndarray_refs():
    """挤出的旧帧必须 release()，切断 bgr 引用以便 GC。"""
    from workers.auto_shot_capture import BufferedFrame

    buf = RollingBuffer(maxlen=3)
    first = BufferedFrame(0, _fake_frame(0), 0.0)
    buf.append(first)
    buf.append(BufferedFrame(1, _fake_frame(1), 0.1))
    buf.append(BufferedFrame(2, _fake_frame(2), 0.2))
    assert len(buf) == 3
    assert first.bgr is not None
    # 第 4 帧挤出 first → release
    buf.append(BufferedFrame(3, _fake_frame(3), 0.3))
    assert len(buf) == 3
    assert first.bgr is None
    assert buf[0].frame_index == 1
    buf.clear()
    assert len(buf) == 0


def test_fsm_idle_approach_impact_cooldown_idle(tmp_path: Path):
    states: list[str] = []
    saved: list[dict] = []

    engine = AutoShotCaptureEngine(
        output_dir=str(tmp_path),
        fps=30.0,
        cooldown_sec=0.15,
        on_state_change=lambda o, n: states.append(n.value),
        on_clip_saved=lambda info: saved.append(info),
    )
    assert engine.state == ShotFsmState.IDLE

    # 先灌入足够前窗
    t_impact = 80
    for i in range(0, t_impact + 1):
        engine.push_frame(_fake_frame(i), i)

    assert engine.notify_approach(omega=120.0) is True
    assert engine.state == ShotFsmState.APPROACH

    assert engine.notify_impact_locked(t_impact) is True
    assert engine.state == ShotFsmState.IMPACT_LOCKED

    # 后窗凑齐 → 异步落盘 → COOLDOWN
    for i in range(t_impact + 1, t_impact + POST_IMPACT_FRAMES + 1):
        engine.push_frame(_fake_frame(i), i)

    assert engine.state == ShotFsmState.COOLDOWN
    assert engine.accepts_impact_triggers() is False

    # 冷却期间忽略新触发
    assert engine.notify_approach(omega=200.0) is False
    assert engine.notify_impact_locked(t_impact + 10) is False

    # 等待写盘 + 冷却
    deadline = time.time() + 3.0
    while time.time() < deadline and (not saved or engine.state != ShotFsmState.IDLE):
        engine.push_frame(_fake_frame(900), 900)
        time.sleep(0.05)

    assert saved, "异步落盘回调未触发"
    assert saved[0]["ok"] is True
    assert saved[0]["attempt_number"] == 1
    assert os.path.isfile(saved[0]["path"])
    assert saved[0]["path"].endswith("attempt_1.mp4")
    assert "session_" in os.path.basename(saved[0]["path"])

    # 切片帧数应约为 pre+1+post（边界齐备时）
    expected = PRE_IMPACT_FRAMES + 1 + POST_IMPACT_FRAMES
    assert saved[0]["frame_count"] == expected

    assert engine.state == ShotFsmState.IDLE
    assert ShotFsmState.APPROACH.value in states
    assert ShotFsmState.IMPACT_LOCKED.value in states
    assert ShotFsmState.COOLDOWN.value in states


def test_discard_returns_to_idle():
    engine = AutoShotCaptureEngine()
    engine.notify_approach()
    engine.notify_discard("flat_omega")
    assert engine.state == ShotFsmState.IDLE
    assert engine.attempt_count == 0


def test_finalize_flushes_partial_post_window(tmp_path: Path):
    saved: list[dict] = []
    engine = AutoShotCaptureEngine(
        output_dir=str(tmp_path),
        fps=30.0,
        cooldown_sec=0.05,
        on_clip_saved=lambda info: saved.append(info),
    )
    t_impact = 40
    for i in range(0, t_impact + 5):  # 仅 5 帧后窗，不足 30
        engine.push_frame(_fake_frame(i), i)
    engine.notify_approach()
    engine.notify_impact_locked(t_impact)
    engine.finalize()

    deadline = time.time() + 2.0
    while time.time() < deadline and not saved:
        time.sleep(0.05)

    assert saved and saved[0]["ok"] is True
    # range(0, t_impact+5) → indices 0..44；前窗不足时取全部可得帧
    assert saved[0]["frame_count"] == t_impact + 5
