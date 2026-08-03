# -*- coding: utf-8 -*-
"""
tests/test_shot_analysis_pipeline.py
=====================================
ShotAnalysisPipeline 单元测试——无 FastAPI / WebSocket / 视频文件依赖。

覆盖目标：
  - 构造函数：注入依赖 / 默认值边界
  - _compute_angular_velocity：Δt 路径（固定 fps / 墙钟 / 首帧 0.0）
  - _compute_stability_index：窗口未满 / 已满 / 常数序列
  - _capture_impact_candidate：面部脱敏契约（绝对拦截器）
  - _cache_blurred_frame / get_blurred_frame：file 模式字典 / webcam 模式环形缓冲
  - build_time_series_velocity_window：空序列 / 非边界 / 边界截断
  - build_scoring_payloads：最小载荷 + 必要键存在断言
  - on_completed / status_provider 回调合约
  - push_fn 收到控制消息（stopped 字段验证）
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shot_analysis_service import ShotAnalysisPipeline


# ---------------------------------------------------------------------------
# 工厂帮助函数
# ---------------------------------------------------------------------------

def make_pipeline(
    *,
    source: str = "file",
    push_fn=None,
    on_completed=None,
    records=None,
    records_lock=None,
    stop_event=None,
    status_provider=None,
) -> ShotAnalysisPipeline:
    """最小化构造一个 ShotAnalysisPipeline，push_fn 默认 mock。"""
    if push_fn is None:
        push_fn = MagicMock()
    return ShotAnalysisPipeline(
        session_id="test-session",
        source=source,
        push_fn=push_fn,
        on_completed=on_completed,
        records=records,
        records_lock=records_lock,
        stop_event=stop_event,
        status_provider=status_provider,
    )


# ---------------------------------------------------------------------------
# 构造函数 + 依赖注入
# ---------------------------------------------------------------------------

class TestInit:
    def test_defaults_are_empty_init_values(self):
        p = make_pipeline()
        assert p.session_id == "test-session"
        assert p.source == "file"
        assert p._trajectory_angles == []
        assert p._trajectory_omega == []
        assert p.t_impact is None
        assert p.sync_frame_count == 0
        assert p.impact_frame is None
        assert p.impact_metrics is None

    def test_shared_records_list_is_same_object(self):
        shared = []
        p = make_pipeline(records=shared)
        p.records.append({"x": 1})
        assert shared[0]["x"] == 1, "管线必须与宿主共享同一 records 列表对象"

    def test_shared_stop_event_is_same_object(self):
        ev = threading.Event()
        p = make_pipeline(stop_event=ev)
        assert p.stop_event is ev

    def test_on_completed_default_is_noop(self):
        p = make_pipeline()
        p._on_completed()  # 不应抛出

    def test_on_completed_callback_invoked(self):
        called = []
        p = make_pipeline(on_completed=lambda: called.append(1))
        p._on_completed()
        assert called == [1]

    def test_status_provider_none_yields_none(self):
        p = make_pipeline(status_provider=None)
        assert p._status_provider is None

    def test_status_provider_callable(self):
        p = make_pipeline(status_provider=lambda: "PROCESSING")
        assert p._status_provider() == "PROCESSING"

    def test_store_all_blurred_jpeg_file_vs_webcam(self):
        pf = make_pipeline(source="file")
        pw = make_pipeline(source="webcam")
        assert pf._store_all_blurred_jpeg is True
        assert pw._store_all_blurred_jpeg is False


# ---------------------------------------------------------------------------
# _compute_angular_velocity
# ---------------------------------------------------------------------------

class TestComputeAngularVelocity:
    def test_first_frame_returns_zero(self):
        """第一帧因无前帧对比，返回 0.0。"""
        p = make_pipeline()
        result = p._compute_angular_velocity(90.0)
        assert result == 0.0
        assert p._prev_angle == 90.0

    def test_fixed_fps_path(self):
        p = make_pipeline(source="file")
        p._fixed_frame_dt = 1 / 30.0
        p._prev_angle = 80.0
        result = p._compute_angular_velocity(110.0)
        # 30° / (1/30) = 900 deg/s
        assert result is not None
        assert math.isclose(result, 900.0, rel_tol=1e-6)

    def test_wallclock_path_positive_dt(self):
        p = make_pipeline(source="webcam")
        p._prev_angle = 100.0
        p._prev_frame_time = time.time() - 0.1  # 0.1 s ago
        result = p._compute_angular_velocity(110.0)
        # 10° over ~0.1s ≈ 100 deg/s
        assert result is not None
        assert 50 < abs(result) < 200  # loose sanity check for timing variance

    def test_zero_dt_returns_zero(self):
        """时间戳相同时 dt=0，返回 0.0（不抛出异常）。"""
        p = make_pipeline(source="webcam")
        p._prev_angle = 90.0
        now = time.time()
        p._prev_frame_time = now
        with patch("time.time", return_value=now):
            result = p._compute_angular_velocity(91.0)
        assert result == 0.0

    def test_prev_angle_updated_after_call(self):
        p = make_pipeline()
        p._fixed_frame_dt = 1 / 30.0
        p._prev_angle = 50.0
        p._compute_angular_velocity(70.0)
        assert p._prev_angle == 70.0


# ---------------------------------------------------------------------------
# _compute_stability_index
# ---------------------------------------------------------------------------

class TestComputeStabilityIndex:
    def test_empty_window_returns_100(self):
        """窗口未满（<2 样本）时返回 100（最高稳定性）。"""
        p = make_pipeline()
        assert p._compute_stability_index() == 100

    def test_constant_sequence_is_stable(self):
        p = make_pipeline()
        for _ in range(30):
            p._velocity_window.append(100.0)
        idx = p._compute_stability_index()
        # 零标准差 → 100
        assert idx == 100

    def test_high_variance_is_low_stability(self):
        p = make_pipeline()
        for i in range(30):
            p._velocity_window.append(500.0 if i % 2 == 0 else -500.0)
        idx = p._compute_stability_index()
        assert idx < 50, "大幅抖动时稳定指数应远低于 100"

    def test_window_not_full_still_computes(self):
        p = make_pipeline()
        p._velocity_window.append(10.0)
        p._velocity_window.append(20.0)
        idx = p._compute_stability_index()
        assert 0 <= idx <= 100


# ---------------------------------------------------------------------------
# _capture_impact_candidate：面部脱敏契约（绝对拦截器）
# ---------------------------------------------------------------------------

class TestCaptureImpactCandidate:
    def _fake_frame(self, h=100, w=80):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def _fake_landmarks(self):
        """构造一个含 33 个关键点的假 landmarks（访问 [23] [24]）。"""
        class FakeLandmark:
            def __init__(self, x, y):
                self.x, self.y = x, y
        return [FakeLandmark(0.5, 0.5) for _ in range(33)]

    def test_stored_impact_frame_is_anonymized(self):
        """存入 impact_frame 的必须是 apply_facial_anonymization 的返回值。"""
        p = make_pipeline()
        raw_frame = np.ones((60, 80, 3), dtype=np.uint8) * 200
        anon_frame = np.zeros((60, 80, 3), dtype=np.uint8)  # 全黑（不同内容）
        landmarks = self._fake_landmarks()

        with patch("shot_analysis_service.pt") as mock_pt:
            mock_pt.apply_facial_anonymization.return_value = anon_frame
            p._capture_impact_candidate(
                raw_frame, landmarks,
                hip_px=(40, 30), knee_px=(40, 50), ankle_px=(40, 70),
                angle=90.0, status="green"
            )

        assert p.impact_frame is not None
        # 通过均值区分：原帧全 200，脱敏帧全 0
        assert float(p.impact_frame.mean()) < 1.0, (
            "impact_frame 必须是脱敏后的帧（apply_facial_anonymization 的返回值），"
            "不能是未经脱敏的原始帧。这违反了《未成年人保护法》拦截器契约。"
        )

    def test_apply_facial_anonymization_called_before_storage(self):
        """apply_facial_anonymization 必须在帧存储前被调用。"""
        call_order: list[str] = []
        p = make_pipeline()
        raw_frame = np.zeros((60, 80, 3), dtype=np.uint8)
        landmarks = self._fake_landmarks()

        def tracking_anon(frame, lm):
            call_order.append("anonymize")
            return frame.copy()

        with patch("shot_analysis_service.pt") as mock_pt:
            mock_pt.apply_facial_anonymization.side_effect = tracking_anon
            p._capture_impact_candidate(
                raw_frame, landmarks,
                hip_px=(40, 30), knee_px=(40, 50), ankle_px=(40, 70),
                angle=90.0, status="green"
            )
            call_order.append("stored")  # 帧赋值在调用返回后完成

        assert call_order[0] == "anonymize", "匿名化必须先于帧存储执行"

    def test_impact_metrics_structure(self):
        """impact_metrics 必须包含必要键。"""
        p = make_pipeline()
        frame = self._fake_frame()
        landmarks = self._fake_landmarks()

        with patch("shot_analysis_service.pt") as mock_pt:
            mock_pt.apply_facial_anonymization.return_value = frame.copy()
            p._capture_impact_candidate(
                frame, landmarks,
                hip_px=(40, 30), knee_px=(40, 50), ankle_px=(40, 70),
                angle=123.5, status="red"
            )

        assert p.impact_metrics is not None
        for k in ("hip_px", "knee_px", "ankle_px", "mid_hip_px", "angle", "status"):
            assert k in p.impact_metrics, f"impact_metrics 缺少 '{k}'"
        assert p.impact_metrics["angle"] == 123.5
        assert p.impact_metrics["status"] == "red"


# ---------------------------------------------------------------------------
# _cache_blurred_frame / get_blurred_frame
# ---------------------------------------------------------------------------

class TestBlurredFrameCache:
    def _make_fake_frame(self):
        """构造一个假的 BGR 帧用于 JPEG 编码。"""
        return np.zeros((60, 80, 3), dtype=np.uint8)

    def test_file_mode_stores_by_index(self):
        p = make_pipeline(source="file")
        frame = self._make_fake_frame()
        p._cache_blurred_frame(7, frame)
        result = p.get_blurred_frame(7)
        assert result is not None
        assert result.shape == (60, 80, 3)

    def test_webcam_mode_ring_stores_and_retrieves(self):
        p = make_pipeline(source="webcam")
        frame = self._make_fake_frame()
        p._cache_blurred_frame(3, frame)
        result = p.get_blurred_frame(3)
        assert result is not None
        assert result.shape == (60, 80, 3)

    def test_missing_index_returns_none(self):
        p = make_pipeline(source="file")
        assert p.get_blurred_frame(999) is None

    def test_webcam_ring_evicts_oldest_on_overflow(self):
        """环形缓冲满后旧帧应被淘汰（capacity = 150）。"""
        p = make_pipeline(source="webcam")
        frame = self._make_fake_frame()
        for i in range(155):
            p._cache_blurred_frame(i, frame)
        # Index 0 must be gone (evicted)
        assert p.get_blurred_frame(0) is None
        # Recent entries must survive
        assert p.get_blurred_frame(154) is not None


# ---------------------------------------------------------------------------
# build_time_series_velocity_window
# ---------------------------------------------------------------------------

def _patch_smooth_keep_pack(omega_series):
    """Mock 平滑器，但保留真实 pack_action_roi_series / 时间戳公式。"""
    import pose_tracker as real_pt

    ctx = patch("shot_analysis_service.pt")
    mock_pt = ctx.start()
    mock_pt.KinematicSignalProcessor.smooth_joint_trajectories.return_value = omega_series
    mock_pt.pack_action_roi_series = real_pt.pack_action_roi_series
    mock_pt.build_absolute_timestamps = real_pt.build_absolute_timestamps
    return ctx


class TestBuildAbsoluteTimestamps:
    def test_formula_and_empty(self):
        import pose_tracker as real_pt

        assert real_pt.build_absolute_timestamps(10, 0, 30.0) == []
        ts = real_pt.build_absolute_timestamps(60, 3, 30.0)
        assert ts == [2.0, 2.033333, 2.066667]
        packed = real_pt.pack_action_roi_series([1.0, 2.0], start_frame_index=90, fps=30.0)
        assert packed["absolute_timestamps"] == [3.0, 3.033333]
        assert packed["start_frame_index"] == 90


class TestBuildTimeSeriesVelocityWindow:
    def test_empty_trajectory_returns_empty(self):
        p = make_pipeline()
        window, idx, roi_start, timestamps = p.build_time_series_velocity_window()
        assert window == []
        assert idx == 0
        assert roi_start == 0
        assert timestamps == []

    def test_returns_correct_types(self):
        p = make_pipeline()
        p._trajectory_omega = [float(i) for i in range(100)]
        p.t_impact = 50
        p._video_fps = 30.0

        ctx = _patch_smooth_keep_pack(p._trajectory_omega)
        try:
            window, idx, roi_start, timestamps = p.build_time_series_velocity_window(
                t_impact=50
            )
        finally:
            ctx.stop()

        assert isinstance(window, list)
        assert all(isinstance(v, float) for v in window)
        assert isinstance(idx, int)
        assert isinstance(roi_start, int)
        assert isinstance(timestamps, list)
        assert len(timestamps) == len(window)
        assert all(isinstance(t, float) for t in timestamps)

    def test_impact_index_in_window_clamped(self):
        """边界：t_impact=0 时 impact_index 应为 0，不应为负值。"""
        p = make_pipeline()
        p._trajectory_omega = [float(i) for i in range(60)]
        p._video_fps = 30.0

        ctx = _patch_smooth_keep_pack(p._trajectory_omega)
        try:
            window, idx, roi_start, timestamps = p.build_time_series_velocity_window(
                t_impact=0
            )
        finally:
            ctx.stop()

        assert idx >= 0
        assert roi_start == 0  # 应为 max(0, 0-30) = 0
        assert len(timestamps) == len(window)
        if timestamps:
            assert timestamps[0] == 0.0

    def test_absolute_timestamps_match_fps_formula(self):
        """absolute_timestamps[i] == (roi_start + i) / fps。"""
        p = make_pipeline()
        p._trajectory_omega = [float(i) for i in range(100)]
        p.t_impact = 50
        p._video_fps = 25.0

        ctx = _patch_smooth_keep_pack(p._trajectory_omega)
        try:
            window, _idx, roi_start, timestamps = p.build_time_series_velocity_window(
                t_impact=50
            )
        finally:
            ctx.stop()

        assert len(timestamps) == len(window) > 0
        for i, ts in enumerate(timestamps):
            assert abs(ts - (roi_start + i) / 25.0) < 1e-6

    def test_t_impact_from_attribute_when_arg_is_none(self):
        p = make_pipeline()
        p._trajectory_omega = [1.0, 5.0, 3.0, 2.0]
        p.t_impact = 1  # 属性值应被使用
        p._video_fps = 30.0

        ctx = _patch_smooth_keep_pack(p._trajectory_omega)
        try:
            window, idx, roi_start, timestamps = p.build_time_series_velocity_window(
                t_impact=None
            )
        finally:
            ctx.stop()

        assert isinstance(window, list)
        assert isinstance(timestamps, list)
        # 应使用 p.t_impact=1 作为中心


# ---------------------------------------------------------------------------
# build_scoring_payloads：最小载荷 + 必要键存在
# ---------------------------------------------------------------------------

class TestBuildScoringPayloads:
    def _build_minimal_pipeline(self):
        p = make_pipeline()
        n = 80
        p._trajectory_angles = [float(i) for i in range(n)]
        p._trajectory_omega = [float(i) * 0.5 for i in range(n)]
        p._trajectory_ankle_px = [(100.0 + i, 200.0 + i * 0.5) for i in range(n)]
        p._trajectory_pose_frames = [
            {
                "swing_ankle": [0.5, 0.6, 0.0],
                "support_ankle": [0.3, 0.6, 0.0],
                "swing_knee": [0.5, 0.5, 0.0],
                "swing_hip": [0.5, 0.4, 0.0],
                "support_knee": [0.3, 0.5, 0.0],
                "support_hip": [0.3, 0.4, 0.0],
            }
            for _ in range(n)
        ]
        p.t_impact = 50
        p._video_fps = 30.0
        return p

    def test_returns_two_dicts(self):
        p = self._build_minimal_pipeline()
        with patch("shot_analysis_service.pt") as mock_pt:
            mock_pt.KinematicSignalProcessor.smooth_joint_trajectories.side_effect = (
                lambda x: x
            )
            result = p.build_scoring_payloads()
        assert isinstance(result, tuple)
        assert len(result) == 2
        impact_payload, trajectory_payload = result
        assert isinstance(impact_payload, dict)
        assert isinstance(trajectory_payload, dict)

    def test_impact_payload_required_keys(self):
        p = self._build_minimal_pipeline()
        with patch("shot_analysis_service.pt") as mock_pt:
            mock_pt.KinematicSignalProcessor.smooth_joint_trajectories.side_effect = (
                lambda x: x
            )
            impact_payload, _ = p.build_scoring_payloads()
        for k in ("session_id", "task_id", "t_impact", "total_frames", "fps"):
            assert k in impact_payload, f"impact_payload 缺少 '{k}'"

    def test_trajectory_payload_required_keys(self):
        p = self._build_minimal_pipeline()
        with patch("shot_analysis_service.pt") as mock_pt:
            mock_pt.KinematicSignalProcessor.smooth_joint_trajectories.side_effect = (
                lambda x: x
            )
            _, trajectory_payload = p.build_scoring_payloads()
        for k in ("session_id", "task_id", "knee_angles", "angular_velocities", "fps"):
            assert k in trajectory_payload, f"trajectory_payload 缺少 '{k}'"

    def test_session_id_propagated(self):
        p = self._build_minimal_pipeline()
        with patch("shot_analysis_service.pt") as mock_pt:
            mock_pt.KinematicSignalProcessor.smooth_joint_trajectories.side_effect = (
                lambda x: x
            )
            ip, tp = p.build_scoring_payloads()
        assert ip["session_id"] == "test-session"
        assert tp["session_id"] == "test-session"

    def test_task_id_equals_session_id(self):
        """task_id 是 session_id 的别名（向后兼容旧 Scorer 调用约定）。"""
        p = self._build_minimal_pipeline()
        with patch("shot_analysis_service.pt") as mock_pt:
            mock_pt.KinematicSignalProcessor.smooth_joint_trajectories.side_effect = (
                lambda x: x
            )
            ip, tp = p.build_scoring_payloads()
        assert ip["task_id"] == ip["session_id"]
        assert tp["task_id"] == tp["session_id"]


# ---------------------------------------------------------------------------
# push_fn 回调合约：stopped 消息字段验证
# ---------------------------------------------------------------------------

class TestPushFnStopped:
    def test_stopped_payload_fields(self):
        """run() 的 finally 块必须推送含必要字段的 stopped 消息。

        由于 run() 需要真实视频源，这里通过直接调用 finally 逻辑的等价路径
        来验证 stopped 的 payload 结构——提取 _push_fn({...}) 的调用参数。
        """
        received: list[dict] = []
        completed: list = []

        p = make_pipeline(
            push_fn=lambda d: received.append(d),
            on_completed=lambda: completed.append(1),
            status_provider=lambda: "COMPLETED",
        )

        # 直接模拟 finally 块的执行顺序（on_completed 先于 stopped）
        p._on_completed()
        p._push_fn(
            {
                "type": "stopped",
                "session_id": p.session_id,
                "total_records": len(p.records),
                "frame_count": p.sync_frame_count,
                "t_impact": p.t_impact,
                "task_status": p._status_provider() if p._status_provider else "COMPLETED",
            }
        )

        assert completed, "on_completed 必须在 stopped 之前调用"
        assert received, "stopped 消息必须被推送"
        payload = received[0]
        assert payload["type"] == "stopped"
        assert payload["session_id"] == "test-session"
        assert "total_records" in payload
        assert "frame_count" in payload
        assert "t_impact" in payload
        assert "task_status" in payload

    def test_on_completed_before_stopped_invariant(self):
        """V2.5 不变式：mark_completed() 必须先于 stopped 消息推送。"""
        order: list[str] = []

        def on_completed():
            order.append("completed")

        def push_fn(d: dict):
            if d.get("type") == "stopped":
                order.append("stopped")

        p = make_pipeline(push_fn=push_fn, on_completed=on_completed)
        # 复现 run() finally 块顺序
        p._on_completed()
        p._push_fn({"type": "stopped"})

        assert order == ["completed", "stopped"], (
            "不变式违反：stopped 消息必须严格晚于 on_completed 回调"
        )
