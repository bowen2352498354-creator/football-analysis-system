# -*- coding: utf-8 -*-
"""V3.1 Sprint 2：AutoShotCaptureEngine 滚动缓冲 / FSM / 切片窗口单测。"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from workers.auto_shot_capture import (
    AutoShotCaptureEngine,
    BUFFER_HEADROOM_FRAMES,
    POST_IMPACT_FRAMES,
    PRE_IMPACT_FRAMES,
    ROLLING_BUFFER_MAXLEN,
    ROLLING_BUFFER_SECONDS,
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


# ----------------------------------------------------------------------
# 帧率适应（24 / 30 / 60 fps）
# ----------------------------------------------------------------------


def _expected_maxlen(fps: float) -> int:
    """缓冲容量 = max(≈ROLLING_BUFFER_SECONDS 秒帧数, 前窗+后窗+余量)。"""
    floor = PRE_IMPACT_FRAMES + POST_IMPACT_FRAMES + BUFFER_HEADROOM_FRAMES
    return max(round(ROLLING_BUFFER_SECONDS * fps), floor)


def test_buffer_capacity_derived_from_fps():
    """maxlen 不再是固定 150，而是按 fps 换算成 ≈5s 的帧数。"""
    assert _expected_maxlen(24.0) == 120
    assert _expected_maxlen(30.0) == 150
    assert _expected_maxlen(60.0) == 300

    for fps in (24.0, 30.0, 60.0):
        engine = AutoShotCaptureEngine(fps=fps)
        cap = _expected_maxlen(fps)
        # 灌入超量帧，验证环容量真的等于推导值
        for i in range(cap + 40):
            engine.push_frame(_fake_frame(i % 7), i)
        assert engine.buffer_len == cap, f"fps={fps} 缓冲容量应为 {cap}"
        assert engine._buffer[-1].frame_index == cap + 39
        assert engine._buffer[0].frame_index == 40  # 最旧的已被挤出

    # 30fps 下必须与旧常量完全等价（向后兼容）
    assert _expected_maxlen(30.0) == ROLLING_BUFFER_MAXLEN


def test_cooldown_snapped_to_frame_boundary():
    """cooldown_sec 对齐到整帧：round(sec * fps) / fps。"""
    for fps, expect_frames in ((24.0, 84), (30.0, 105), (60.0, 210)):
        engine = AutoShotCaptureEngine(fps=fps, cooldown_sec=3.5)
        assert engine.fps == fps
        assert engine._cooldown_frames == expect_frames
        assert engine.cooldown_sec == expect_frames / fps

    # 24fps 下 3.5s = 84 帧，正好整除；取一个会被舍入的值验证对齐生效
    engine = AutoShotCaptureEngine(fps=24.0, cooldown_sec=0.1)
    assert engine._cooldown_frames == 2  # round(0.1 * 24) = 2
    assert abs(engine.cooldown_sec - 2 / 24) < 1e-9


def test_fps_setter_recomputes_buffer_and_cooldown():
    """视频源真实帧率在打开后才知道 → 赋值必须即时重算，且保留最近帧。"""
    engine = AutoShotCaptureEngine(fps=30.0, cooldown_sec=3.5)
    assert engine.buffer_capacity == 150

    for i in range(150):
        engine.push_frame(_fake_frame(i % 5), i)
    assert engine.buffer_len == 150

    # 扩容（30 → 60）：容量翻倍，已有帧全部保留
    engine.fps = 60.0
    assert engine.buffer_capacity == 300
    assert engine.buffer_len == 150
    assert engine._buffer[0].frame_index == 0
    assert engine._buffer[-1].frame_index == 149
    assert engine._cooldown_frames == 210
    assert engine.cooldown_sec == 210 / 60.0

    # 缩容（60 → 24）：保留最近 120 帧，被截掉的旧帧必须 release()
    stale = engine._buffer[0]
    assert stale.bgr is not None
    engine.fps = 24.0
    assert engine.buffer_capacity == 120
    assert engine.buffer_len == 120
    assert engine._buffer[0].frame_index == 30
    assert engine._buffer[-1].frame_index == 149
    assert stale.bgr is None, "缩容截掉的帧未 release()，ndarray 会悬挂"
    assert engine._cooldown_frames == 84


def test_invalid_fps_falls_back_to_default():
    """0 / None / 负值等无效帧率回落到 30fps，避免除零与容量塌缩。"""
    for bad in (0.0, 1.0, -5.0, None):
        engine = AutoShotCaptureEngine(fps=bad)
        assert engine.fps == 30.0
        assert engine.buffer_capacity == 150

    engine = AutoShotCaptureEngine(fps=60.0)
    engine.fps = 0.0
    assert engine.fps == 30.0
    assert engine.buffer_capacity == 150


def test_explicit_rolling_maxlen_overrides_fps_derivation():
    """显式传入 rolling_maxlen 时固定容量，不随 fps 变化（旧调用方语义）。"""
    engine = AutoShotCaptureEngine(fps=60.0, rolling_maxlen=150)
    assert engine.buffer_capacity == 150
    engine.fps = 24.0
    assert engine.buffer_capacity == 150
    engine.fps = 60.0
    assert engine.buffer_capacity == 150


def test_buffer_capacity_never_below_slice_window():
    """极低帧率下容量也不能小于 前窗+后窗，否则切片永远凑不齐。"""
    engine = AutoShotCaptureEngine(fps=10.0)
    floor = PRE_IMPACT_FRAMES + POST_IMPACT_FRAMES + BUFFER_HEADROOM_FRAMES
    assert engine.buffer_capacity == floor  # round(5*10)=50 < 100 → 取下界
    assert engine.buffer_capacity >= PRE_IMPACT_FRAMES + POST_IMPACT_FRAMES


def test_checkpoint_meta_reports_fps_derived_values(tmp_path: Path):
    """断点续传元数据须反映当前帧率下的真实容量与冷却，且不含像素帧。"""
    engine = AutoShotCaptureEngine(output_dir=str(tmp_path), fps=60.0, cooldown_sec=3.5)
    meta = engine.export_checkpoint_meta()
    assert meta["fps"] == 60.0
    assert meta["buffer_maxlen"] == 300
    assert abs(meta["cooldown_sec"] - 210 / 60.0) < 1e-9
    # 隐私红线：checkpoint 绝不落像素——显式声明 + 全部字段必须是标量
    assert meta.get("buffer_frames_persisted") is False
    for key, value in meta.items():
        assert isinstance(value, (int, float, str, bool)), f"{key} 疑似承载像素数据"
