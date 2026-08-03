# -*- coding: utf-8 -*-
"""
workers/inference_worker.py
【V3.1 Sprint 4】桌面端原生并发架构 —— 独立计算线程 InferenceWorker

职责边界（硬性约束）：
    - 本模块运行在 QThread 从线程：独占 cv2.VideoCapture / MediaPipe 推理 /
      角速度平滑 / ImpactFrameLocator 锁帧 / DeterministicScorer 打分。
    - 通过无锁 pyqtSignal 总线把结果推给 GUI 主线程；
      主线程严禁出现任何 ``cv2.read()`` 或推理调用。
    - RollingBuffer / 轨迹 deque 均为有界结构，旧帧矩阵释放引用后交 GC。

【V3.1 Sprint 2】无感全自动切片：
    - ``AutoShotCaptureEngine`` 维护 RollingBuffer(maxlen=150)；
    - 射门 FSM：IDLE → APPROACH → IMPACT_LOCKED → COOLDOWN；
    - 锁帧成功后异步 VideoWriter 落盘 [t_impact−60, t_impact+30]，不阻塞本线程。
"""

from __future__ import annotations

import copy
import csv
import gc
import json
import os
import random
import threading
import time
from collections import deque
from queue import Empty, Full, Queue
from typing import Any, Deque, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

import llm_agent
import pose_tracker as pt
from error_diagnoser import DeterministicScorer
from experimental_group_router import (
    GROUP_A,
    GROUP_B,
    GROUP_C,
    ExperimentalGroupRouter,
    normalize_experimental_group,
)
from session_monitor import FatigueMonitor
from session_checkpoint import (
    STATUS_ACTIVE,
    SessionCheckpointStore,
    SessionSnapshot,
    default_store,
)
from .auto_shot_capture import AutoShotCaptureEngine, ShotFsmState

# 摄像头实时触球检测：角速度峰值回落判定阈值与冷却
_LIVE_OMEGA_PEAK_THRESHOLD: float = float(
    getattr(pt, "SHOT_OMEGA_PEAK_THRESHOLD", 80.0)
)
_LIVE_POST_PEAK_FRAMES: int = 8  # 峰值后再观察若干帧再锁帧，避免半截动作
_REPLAY_BUFFER_MAX: int = 60  # GROUP_A 时空胶囊半速回放缓冲
# ~10s @30fps：运动学轨迹有界，防止整堂课无限 append 导致 OOM
_TRAJECTORY_MAXLEN: int = 300
# 【Sprint 5】连续空帧 / 读取失败阈值：达到后发 camera_lost_signal 并自愈重开，绝不裸崩溃
_CAMERA_READ_FAIL_LIMIT: int = 50
_CAMERA_REOPEN_SLEEP_SEC: float = 0.45
# 推理结果有界队列：满则丢弃过期帧，优先保障最新帧送达主线程
_FRAME_EMIT_QUEUE_MAXSIZE: int = 2
_DIAGNOSTICS_EMIT_QUEUE_MAXSIZE: int = 1
# 每隔 N 帧显式触发一次轻量 GC，缓解长时间推理的帧缓冲滞留
_GC_EVERY_N_FRAMES: int = 90


def _is_empty_or_failed_frame(ret: bool, frame: Any) -> bool:
    """判定 cap.read() 是否得到可用画面（空帧 / None / 零尺寸均视为失败）。"""
    if not ret or frame is None:
        return True
    if not isinstance(frame, np.ndarray):
        return True
    if frame.size == 0:
        return True
    return False


class InferenceWorker(QThread):
    """后台推理计算线程：打开视频源 → 姿态提取 → 运动学清洗 → 锁帧打分。

    通信总线（无锁信号，跨线程 QueuedConnection）—— Sprint 4 强类型：
        frame_ready_signal(np.ndarray)
            已画骨骼线且已脱敏的 BGR 画面，供 UI 实时渲染。
        diagnostics_ready_signal(dict)
            每次射门锁定后的 8 大量纲 JSON（生物力学打分结果）。
        fatigue_warning_signal(dict)
            课堂疲劳变形熔断报警（由 FatigueMonitor 转发）。
        error_occurred_signal(str)
            帧异常 / 计算越界时的安全报错，绝不让线程裸崩溃拖死 GUI。
        camera_lost_signal(str)
            连续空帧达到阈值时上抛；主线程提示「摄像头信号丢失，尝试重连...」，
            Worker 内部执行 cap.release()+cap.open() 自愈，不结束训练会话。
        log_message(str)
            运行日志（兼容原主窗口日志区）。
    """

    # 【Sprint 4】主信号总线（正名）
    frame_ready_signal = pyqtSignal(np.ndarray)
    diagnostics_ready_signal = pyqtSignal(dict)
    fatigue_warning_signal = pyqtSignal(dict)
    capture_discarded_signal = pyqtSignal(str)
    error_occurred_signal = pyqtSignal(str)
    # 【Sprint 5】摄像头丢信号自愈（非致命，不断开会话）
    camera_lost_signal = pyqtSignal(str)
    log_message = pyqtSignal(str)
    # 【V3.1 Sprint 2】无感自动切片：FSM 状态 / 切片落盘完成（均不阻塞 GUI）
    fsm_state_changed_signal = pyqtSignal(str)
    clip_saved_signal = pyqtSignal(dict)

    # 兼容旧槽名（Sprint 3 / 早期接线）
    frame_processed_signal = frame_ready_signal
    impact_detected_signal = diagnostics_ready_signal
    frame_ready = frame_ready_signal

    def __init__(
        self,
        data_source: str,
        video_path: str,
        group: str,
        camera_index: int = 0,
        experimental_group: Optional[str] = None,
        parent=None,
        *,
        class_group: str = "",
        student_id: str = "",
        student_name: str = "",
        checkpoint_store: Optional[SessionCheckpointStore] = None,
        restore_snapshot: Optional[SessionSnapshot] = None,
    ):
        super().__init__(parent)
        self.data_source = data_source
        self.video_path = video_path
        self.camera_index = camera_index

        raw = experimental_group if experimental_group is not None else group
        self.experimental_group = normalize_experimental_group(raw, GROUP_A)
        self.group = {GROUP_A: "A", GROUP_B: "B", GROUP_C: "C"}.get(
            self.experimental_group, "A"
        )
        self._router = ExperimentalGroupRouter(self.experimental_group)
        self._policy = self._router.policy()

        self._running = True
        self._paused_for_capsule = False
        self._capsule_lock = threading.Lock()

        # 教学身份（断点快照主键上下文）
        self.class_group = str(class_group or "").strip()
        self.student_id = str(student_id or "").strip()
        self.student_name = str(student_name or "").strip()

        # 断点续传仓库
        self._checkpoint_store = checkpoint_store or default_store
        self._session_snapshot: Optional[SessionSnapshot] = None
        self._clip_paths: List[str] = []
        self._checkpoint_clean_exit = False

        # A 组 AIGC 防抖状态（每次新建 Worker 均重置）
        self._red_streak_start_time: Optional[float] = None
        self._red_streak_already_triggered = False
        self._feedback_lock = threading.Lock()
        self._feedback_text: Optional[str] = None
        self._feedback_show_until = 0.0
        self._is_calling_llm = False

        # B 组静默落盘缓冲 + 本地宽表行
        self._b_group_new_records: List[dict] = []
        self._b_group_wide_rows: List[dict] = []

        # 运动学轨迹（有界 deque，与帧索引对齐，供锁帧 / 打分；防整堂课 OOM）
        self._trajectory_angles: Deque[float] = deque(maxlen=_TRAJECTORY_MAXLEN)
        self._trajectory_omega: Deque[float] = deque(maxlen=_TRAJECTORY_MAXLEN)
        self._trajectory_ankle_px: Deque[Tuple[float, float]] = deque(
            maxlen=_TRAJECTORY_MAXLEN
        )
        self._prev_angle: Optional[float] = None
        self._fixed_frame_dt: Optional[float] = None
        self._video_fps: float = 30.0
        self._replay_buffer: Deque[np.ndarray] = deque(maxlen=_REPLAY_BUFFER_MAX)

        # 实时触球检测状态
        self._peak_omega_abs = 0.0
        self._peak_frame_index = -1
        self._frames_since_peak = 0
        self._awaiting_post_peak = False
        self._last_impact_emit_time = 0.0
        self._impact_emitted_for_segment = False

        # 有界发射队列：主线程消费慢时丢弃过期帧 / 旧诊断，避免内存蓄积卡死 GUI
        self._frame_emit_queue: Queue = Queue(maxsize=_FRAME_EMIT_QUEUE_MAXSIZE)
        self._diagnostics_emit_queue: Queue = Queue(maxsize=_DIAGNOSTICS_EMIT_QUEUE_MAXSIZE)
        self._frames_since_gc = 0

        self._scorer = DeterministicScorer()

        # 课堂疲劳监控：本堂课历次 8 大量纲时序（报警经本 Worker 信号上抛）
        self.fatigue_monitor = FatigueMonitor()
        sid = self.student_id or None
        self.fatigue_monitor.reset_session(student_id=sid)

        # 【V3.1 Sprint 2】滚动时间机器 + 射门 FSM（异步落盘，不阻塞本 QThread）
        # rolling_maxlen 不再传入，由引擎按 fps 自动推导（≈5s 帧数）；
        # 真实 fps 在打开摄像头/视频文件后由 _apply_fps() setter 即时修正。
        self._auto_capture = AutoShotCaptureEngine(
            output_dir=getattr(pt, "AUTO_CAPTURE_CLIPS_DIR", None),
            fps=self._video_fps,
            pre_frames=int(getattr(pt, "SHOT_PRE_IMPACT_FRAMES", 60)),
            post_frames=int(getattr(pt, "SHOT_POST_IMPACT_FRAMES", 30)),
            cooldown_sec=float(getattr(pt, "SHOT_IMPACT_COOLDOWN_SEC", 3.5)),
            on_log=lambda msg: self.log_message.emit(msg),
            on_state_change=self._on_auto_capture_state_change,
            on_clip_saved=self._on_auto_clip_saved,
        )

        # 热重启：若带入未结案快照，立刻还原 attempts_history / attempt_count
        if restore_snapshot is not None:
            self._apply_recovery_snapshot(restore_snapshot)
        else:
            self._begin_checkpoint_session()

    def _begin_checkpoint_session(self) -> None:
        """新开一堂课：创建 active 快照骨架（尚无射门时不打扰恢复对话框）。"""
        self._session_snapshot = SessionSnapshot.new_session(
            class_group=self.class_group,
            student_id=self.student_id,
            student_name=self.student_name,
            experimental_group=self.experimental_group,
            data_source=self.data_source,
            video_path=self.video_path,
        )
        self._clip_paths = []

    def _apply_recovery_snapshot(self, snap: SessionSnapshot) -> None:
        """把异常中断的教学记录灌回 FatigueMonitor + AutoShotCapture。"""
        self._session_snapshot = snap
        self._session_snapshot.status = STATUS_ACTIVE
        self.class_group = snap.class_group or self.class_group
        self.student_id = snap.student_id or self.student_id
        self.student_name = snap.student_name or self.student_name
        self._clip_paths = list(snap.clip_paths or [])

        self.fatigue_monitor.restore_state(
            attempts_history=list(snap.attempts_history or []),
            student_id=self.student_id or None,
            last_fatigue_warning=snap.last_fatigue_warning,
        )
        self._auto_capture.restore_from_checkpoint(
            attempt_count=int(snap.attempt_count or 0),
            meta=snap.auto_capture_meta,
        )
        self.log_message.emit(
            f"【断点恢复】已载入 {snap.display_label()} 的 "
            f"{int(snap.attempt_count)} 次射门记录"
            f"{'（含疲劳预警）' if snap.fatigue_triggered else ''}。"
        )

    def mark_clean_exit(self) -> None:
        """主线程在老师主动结束训练时调用，避免被当成崩溃。"""
        self._checkpoint_clean_exit = True

    def _persist_checkpoint(self, *, clip_info: Optional[dict] = None) -> None:
        """事务级写快照：IMPACT_LOCKED 完成且切片落盘后立刻调用。"""
        if self._session_snapshot is None:
            self._begin_checkpoint_session()
        assert self._session_snapshot is not None

        if isinstance(clip_info, dict) and clip_info.get("ok") and clip_info.get("path"):
            path = str(clip_info["path"])
            if path and path not in self._clip_paths:
                self._clip_paths.append(path)

        fatigue_state = self.fatigue_monitor.export_state()
        snap = self._session_snapshot
        snap.status = "active"
        snap.class_group = self.class_group
        snap.student_id = self.student_id
        snap.student_name = self.student_name
        snap.experimental_group = self.experimental_group
        snap.data_source = self.data_source
        snap.video_path = self.video_path
        history = list(fatigue_state.get("attempts_history") or [])
        # 后窗未凑齐时 attempt_count 可能尚未 +1；以 history 长度为下限保真
        snap.attempt_count = max(int(self._auto_capture.attempt_count), len(history))
        snap.attempts_history = history
        snap.fatigue_triggered = bool(fatigue_state.get("fatigue_triggered"))
        snap.last_fatigue_warning = fatigue_state.get("last_fatigue_warning")
        snap.clip_paths = list(self._clip_paths)
        snap.auto_capture_meta = self._auto_capture.export_checkpoint_meta()
        # 元数据里的计数与快照对齐，便于恢复时还原切片序号
        snap.auto_capture_meta["attempt_count"] = int(snap.attempt_count)

        try:
            self._checkpoint_store.save_snapshot(snap)
            self.log_message.emit(
                f"【断点快照】已原子写入 "
                f"{snap.display_label()} · 第 {snap.attempt_count} 脚"
                f"{' · 疲劳已触发' if snap.fatigue_triggered else ''}。"
            )
        except Exception as exc:  # noqa: BLE001 — 快照失败绝不能拖死教学
            self.log_message.emit(f"【断点快照】写入失败（教学继续）：{exc}")

    def _finalize_checkpoint(self) -> None:
        """正常结束 → completed；异常退出则保持 active 供下次启动恢复。"""
        if self._session_snapshot is None:
            return
        sid = self._session_snapshot.session_id
        try:
            if self._checkpoint_clean_exit:
                # 最后再刷一次完整状态，再标 completed
                self._persist_checkpoint()
                self._checkpoint_store.mark_completed(sid)
                self.log_message.emit("【断点快照】本堂课已正常结案，无需恢复。")
            # 非 clean_exit：保留 active，下次启动弹恢复对话框
        except Exception as exc:  # noqa: BLE001
            self.log_message.emit(f"【断点快照】结案失败：{exc}")

    def request_stop(self) -> None:
        """主线程调用：请求从线程自然退出（非强杀）。"""
        self._running = False
        self.resume_after_capsule()

    def pause_for_capsule(self) -> None:
        """GROUP_A：时空胶囊播放期间暂停摄像头读取。"""
        with self._capsule_lock:
            self._paused_for_capsule = True

    def resume_after_capsule(self) -> None:
        """时空胶囊结束后恢复摄像头读取。"""
        with self._capsule_lock:
            self._paused_for_capsule = False

    def _is_paused_for_capsule(self) -> bool:
        with self._capsule_lock:
            return bool(self._paused_for_capsule)

    # ------------------------------------------------------------------
    # A 组：AIGC 防抖（内层 threading.Thread，避免阻塞本 QThread 读环）
    # ------------------------------------------------------------------

    def _call_llm_in_background(self, angle: float, status: str) -> None:
        self.log_message.emit(
            f"检测到连续 {pt.RED_DEBOUNCE_SECONDS} 秒 Red 状态，正在请求 DeepSeek 大模型……"
        )
        feedback_text = llm_agent.generate_feedback(angle, status)
        self.log_message.emit(f"DeepSeek 大模型返回：{feedback_text}")
        with self._feedback_lock:
            self._feedback_text = feedback_text
            self._feedback_show_until = time.time() + pt.FEEDBACK_DISPLAY_SECONDS
            self._is_calling_llm = False

    def _update_llm_trigger_state(self, angle: float, status: str) -> None:
        current_time = time.time()
        if status != "Red":
            self._red_streak_start_time = None
            self._red_streak_already_triggered = False
            return
        if self._red_streak_start_time is None:
            self._red_streak_start_time = current_time
            return
        if (current_time - self._red_streak_start_time) < pt.RED_DEBOUNCE_SECONDS:
            return
        if self._red_streak_already_triggered:
            return
        with self._feedback_lock:
            if self._is_calling_llm:
                return
            self._is_calling_llm = True
        self._red_streak_already_triggered = True
        threading.Thread(
            target=self._call_llm_in_background,
            args=(angle, status),
            daemon=True,
        ).start()

    def _draw_feedback_text_if_needed(self, frame_bgr: np.ndarray) -> np.ndarray:
        with self._feedback_lock:
            text = self._feedback_text
            show_until = self._feedback_show_until
        if not text or time.time() > show_until:
            return frame_bgr
        return pt.draw_chinese_text_with_backdrop(
            frame_bgr, text, pt.FEEDBACK_FONT, anchor="bottom"
        )

    # ------------------------------------------------------------------
    # B 组落盘
    # ------------------------------------------------------------------

    def _record_b_group_data(self, angle: float, status: str) -> None:
        self._b_group_new_records.append(
            {
                "timestamp": time.time(),
                "knee_angle": round(float(angle), 1),
                "status": status,
                "experimental_group": self.experimental_group,
            }
        )

    def _flush_b_group_data_to_disk(self) -> None:
        if self._b_group_new_records:
            existing: List[dict] = []
            if os.path.exists(pt.B_GROUP_LOG_PATH):
                try:
                    with open(pt.B_GROUP_LOG_PATH, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = []
                except Exception:  # noqa: BLE001
                    existing = []
            all_records = existing + self._b_group_new_records
            try:
                with open(pt.B_GROUP_LOG_PATH, "w", encoding="utf-8") as f:
                    json.dump(all_records, f, ensure_ascii=False, indent=2)
                self.log_message.emit(
                    f"B组数据已静默保存：本次新增 {len(self._b_group_new_records)} 条，"
                    f"文件累计 {len(all_records)} 条，路径：{pt.B_GROUP_LOG_PATH}"
                )
            except Exception as exc:  # noqa: BLE001
                self.error_occurred_signal.emit(f"B组数据保存失败：{exc}")

        if self._b_group_wide_rows:
            fieldnames = [
                "timestamp",
                "experimental_group",
                "t_impact",
                "knee_angle",
                "status",
                "total_score",
                "omega_peak",
                "discarded",
            ]
            write_header = not os.path.exists(pt.B_GROUP_WIDE_TABLE_PATH)
            try:
                with open(
                    pt.B_GROUP_WIDE_TABLE_PATH, "a", encoding="utf-8-sig", newline=""
                ) as f:
                    writer = csv.DictWriter(
                        f, fieldnames=fieldnames, extrasaction="ignore"
                    )
                    if write_header:
                        writer.writeheader()
                    for row in self._b_group_wide_rows:
                        writer.writerow(row)
                self.log_message.emit(
                    f"B组宽表已追加 {len(self._b_group_wide_rows)} 行："
                    f"{pt.B_GROUP_WIDE_TABLE_PATH}"
                )
            except Exception as exc:  # noqa: BLE001
                self.error_occurred_signal.emit(f"B组宽表保存失败：{exc}")

    # ------------------------------------------------------------------
    # 运动学轨迹 + 触球锁帧 + 确定性打分
    # ------------------------------------------------------------------

    def _clear_replay_buffer(self) -> None:
        """切断回放缓冲中所有 ndarray 引用，交 GC。"""
        self._replay_buffer.clear()

    def _append_neutral_trajectory(self) -> None:
        last_angle = (
            float(self._trajectory_angles[-1]) if self._trajectory_angles else 150.0
        )
        last_ankle = (
            self._trajectory_ankle_px[-1] if self._trajectory_ankle_px else (0.0, 0.0)
        )
        self._trajectory_angles.append(last_angle)
        self._trajectory_omega.append(0.0)
        self._trajectory_ankle_px.append(last_ankle)
        self._prev_angle = last_angle

    def _compute_angular_velocity(self, angle: float) -> float:
        omega = 0.0
        if self._prev_angle is not None:
            if self._fixed_frame_dt is not None and self._fixed_frame_dt > 0:
                dt = self._fixed_frame_dt
            else:
                dt = self._fixed_frame_dt or (1.0 / 30.0)
            if dt > 0:
                omega = (float(angle) - float(self._prev_angle)) / dt
        self._prev_angle = float(angle)
        return float(omega)

    def _append_pose_trajectory(
        self, angle: float, ankle_px: Tuple[int, int]
    ) -> float:
        omega = self._compute_angular_velocity(angle)
        self._trajectory_angles.append(float(angle))
        self._trajectory_omega.append(float(omega))
        self._trajectory_ankle_px.append((float(ankle_px[0]), float(ankle_px[1])))
        return omega

    @staticmethod
    def _enqueue_drop_oldest(q: Queue, item: Any) -> None:
        """有界队列：满则丢弃最旧项再放入最新项，优先保障实时性。"""
        try:
            q.put_nowait(item)
            return
        except Full:
            pass
        try:
            q.get_nowait()
        except Empty:
            pass
        try:
            q.put_nowait(item)
        except Full:
            pass

    def _drain_emit_queues(self) -> None:
        """把有界队列中的最新帧/诊断经信号发给主线程（仅深拷贝后的独立对象）。"""
        latest_frame = None
        while True:
            try:
                latest_frame = self._frame_emit_queue.get_nowait()
            except Empty:
                break
        if latest_frame is not None and isinstance(latest_frame, np.ndarray):
            # ndarray 已在入队前 .copy()；再 copy 一次切断与队列残留的共享
            self.frame_ready_signal.emit(latest_frame.copy())

        latest_diag = None
        while True:
            try:
                latest_diag = self._diagnostics_emit_queue.get_nowait()
            except Empty:
                break
        if isinstance(latest_diag, dict):
            self.diagnostics_ready_signal.emit(copy.deepcopy(latest_diag))

    def _emit_frame_ready(self, frame_bgr: np.ndarray) -> None:
        """向主线程投递一帧脱敏画面；仅传 copy，原矩阵可被本环覆盖/回收。

        前置条件：调用方必须已通过 ``apply_facial_anonymization`` 覆盖原图；
        本方法本身不再接触未脱敏帧。
        经有界队列中转：主线程慢时主动丢弃过期帧。
        """
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            return
        self._enqueue_drop_oldest(self._frame_emit_queue, frame_bgr.copy())
        self._drain_emit_queues()

    def _emit_diagnostics_ready(self, payload: dict) -> None:
        """跨线程诊断结果：深拷贝后入有界队列再发射，禁止主子线程共享可变对象。"""
        if not isinstance(payload, dict):
            return
        safe = copy.deepcopy(payload)
        self._enqueue_drop_oldest(self._diagnostics_emit_queue, safe)
        self._drain_emit_queues()

    def _try_reopen_camera(self, cap: Any) -> Tuple[Any, bool]:
        """Sprint 5 自愈：释放后重新 open 摄像头。返回 (新 cap, 是否成功)。"""
        try:
            if cap is not None:
                cap.release()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(_CAMERA_REOPEN_SLEEP_SEC)
        try:
            new_cap, video_fps, _reported = pt.open_video_capture_deterministic(
                "", is_camera=True, camera_index=self.camera_index
            )
            if video_fps and video_fps > 1:
                self._video_fps = float(video_fps)
                self._fixed_frame_dt = 1.0 / float(self._video_fps)
                self._auto_capture.fps = float(self._video_fps)
            if new_cap is not None and new_cap.isOpened():
                self.log_message.emit(
                    f"摄像头自愈成功：已重新打开设备 #{self.camera_index}。"
                )
                return new_cap, True
        except Exception as reopen_exc:  # noqa: BLE001
            self.log_message.emit(f"摄像头自愈打开失败：{reopen_exc}")
        return None, False

    def _build_biomechanics_payload(self, t_impact: int) -> dict[str, Any]:
        """KinematicSignalProcessor → ImpactFrameLocator → DeterministicScorer。"""
        n = len(self._trajectory_omega)
        if n <= 0:
            return {
                "t_impact": 0,
                "total_score": 0.0,
                "scoring_engine": "DeterministicScorer_V2.5",
                "error": "empty_trajectory",
            }

        t_impact = int(max(0, min(n - 1, t_impact)))
        omega_smooth = pt.KinematicSignalProcessor.smooth_joint_trajectories(
            list(self._trajectory_omega[:n])
        )
        # 若调用方尚未锁帧，在此用 ImpactFrameLocator 再锁一次（幂等）
        locked_t, _ball = pt.ImpactFrameLocator.locate_with_ball_proxy(
            omega_smooth, list(self._trajectory_ankle_px[:n])
        )
        t_impact = int(max(0, min(n - 1, locked_t)))

        impact_knee = float(self._trajectory_angles[t_impact])
        status_text, _color = pt.judge_knee_status(impact_knee)
        dt = float(self._fixed_frame_dt) if self._fixed_frame_dt else (1.0 / 30.0)

        impact_frame_data = {
            "t_impact": t_impact,
            "total_frames": n,
            "impact_knee_angle": impact_knee,
            "contact_frame_index": t_impact,
            # 【Phase 4】真实采集帧率传入 Scorer，踝冲击窗按时长自适应
            "fps": float(self._video_fps) if self._video_fps and self._video_fps > 1 else 30.0,
        }
        trajectory_data = {
            "knee_angles": list(self._trajectory_angles[:n]),
            "angular_velocities": list(omega_smooth),
            "timestamps_sec": [i * dt for i in range(n)],
            "total_frames": n,
            "t_impact": t_impact,
            "fps": float(self._video_fps) if self._video_fps and self._video_fps > 1 else 30.0,
            "whipping_velocity": float(
                max((abs(v) for v in omega_smooth), default=0.0)
            ),
        }

        total_score, score_detail = self._scorer.calculate_biomechanical_score(
            impact_frame_data, trajectory_data
        )

        return {
            "t_impact": t_impact,
            "total_frames": n,
            "impact_knee_angle": round(impact_knee, 2),
            "knee_status": status_text,
            "total_score": float(total_score),
            "score_detail": score_detail,
            "whipping_velocity": trajectory_data["whipping_velocity"],
            "group": self.group,
            "scoring_engine": "DeterministicScorer_V2.5",
            "t0_method": "ImpactFrameLocator.locate_with_ball_proxy",
        }

    def _ingest_fatigue_monitor(self, payload: dict[str, Any]) -> None:
        """打分成功后写入 FatigueMonitor；熔断时由 fatigue_warning_signal 上抛。"""
        try:
            detail = payload.get("score_detail")
            if isinstance(detail, dict):
                self.fatigue_monitor.record_attempt(detail)
            else:
                self.fatigue_monitor.record_attempt(payload)
            # 在 Worker 线程求值，经本线程信号 Queued 投递主窗口（避免嵌套 QObject 转发）
            warning = self.fatigue_monitor.check_fatigue_deformation(emit_signal=False)
            if isinstance(warning, dict) and warning.get("is_fatigue"):
                self.log_message.emit(
                    f"【疲劳预警】{warning.get('reason')}: {warning.get('message')}"
                )
                payload["fatigue_warning"] = warning
                self.fatigue_warning_signal.emit(copy.deepcopy(dict(warning)))
        except Exception as exc:  # noqa: BLE001 — 监控失败不得中断教学
            self.log_message.emit(f"疲劳监控跳过：{exc}")

    def _peek_feedback_subtitle(self, knee_angle: float, status: str) -> str:
        with self._feedback_lock:
            text = self._feedback_text
            show_until = self._feedback_show_until
        if text and time.time() <= show_until:
            return str(text)
        try:
            return str(llm_agent.generate_feedback(knee_angle, status))
        except Exception:  # noqa: BLE001
            return "触球瞬间已锁定——请感受支撑腿与摆动腿的时空节奏，下一球继续！"

    def _on_auto_capture_state_change(
        self, _old: ShotFsmState, new: ShotFsmState
    ) -> None:
        try:
            self.fsm_state_changed_signal.emit(new.value)
        except Exception:  # noqa: BLE001
            pass

    def _on_auto_clip_saved(self, info: dict) -> None:
        try:
            self.clip_saved_signal.emit(copy.deepcopy(dict(info)))
        except Exception:  # noqa: BLE001
            pass
        # 【灾难恢复】IMPACT_LOCKED 完成 + 切片落盘 → 立刻原子写快照
        if isinstance(info, dict) and info.get("ok"):
            self._persist_checkpoint(clip_info=info)

    def _discard_shot_cycle(self, reason: str) -> None:
        """抛物线锁帧无解：Discard 本轮打分，绝不抛致死异常中断教学。"""
        hint = getattr(pt, "CAPTURE_DISCARD_HINT", "本轮捕获失败，请准备下一球")
        self.capture_discarded_signal.emit(hint)
        self.log_message.emit(f"Discard：本轮锁帧无解（{reason}），跳过打分，请准备下一球。")
        self._impact_emitted_for_segment = True
        self._last_impact_emit_time = time.time()
        self._auto_capture.notify_discard(reason)

    def _emit_impact_if_ready(self, force: bool = False) -> None:
        """在轨迹足够时执行锁帧+打分；无解则 Discard，成功则按组别分流。"""
        n = len(self._trajectory_omega)
        if n < getattr(pt, "MIN_FRAMES_FOR_IMPACT_LOCK", 15) and not force:
            return
        if n < 5:
            return
        # Sprint 2：COOLDOWN / IMPACT_LOCKED 期间忽略新触发，防抖连踢
        if not force and not self._auto_capture.accepts_impact_triggers():
            return
        now = time.time()
        if not force and self._impact_emitted_for_segment:
            return

        try:
            omega_smooth = pt.KinematicSignalProcessor.smooth_joint_trajectories(
                list(self._trajectory_omega[:n])
            )
            ankles = list(self._trajectory_ankle_px[:n])
            # 无独立球检：足/踝代理球心
            ball_coords = list(ankles)
            t_impact, meta = pt.ImpactFrameLocator.try_locate_or_discard(
                omega_smooth, ankles, ball_coords
            )
            if t_impact is None or meta.get("discarded"):
                self._discard_shot_cycle(str(meta.get("discard_reason") or "unsolvable"))
                return

            payload = self._build_biomechanics_payload(int(t_impact))
            payload["experimental_group"] = self.experimental_group
            payload["lock_meta"] = {
                k: v for k, v in (meta or {}).items() if k not in ("ok",)
            }
            duration = float(
                random.uniform(
                    getattr(pt, "CAPSULE_DURATION_MIN_SEC", 8.0),
                    getattr(pt, "CAPSULE_DURATION_MAX_SEC", 12.0),
                )
            )
            knee = float(payload.get("impact_knee_angle") or 150.0)
            status = str(payload.get("knee_status") or "Yellow")
            payload["subtitle"] = self._peek_feedback_subtitle(knee, status)
            payload["duration_sec"] = duration
            payload["fps"] = float(self._video_fps)
            # 回放帧深拷贝后立刻清空本地缓冲，避免与 payload 双持有
            replay_frames = [f.copy() for f in self._replay_buffer]
            self._clear_replay_buffer()
            payload["replay_frames"] = replay_frames

            # 【Sprint 2】锁帧成功 → IMPACT_LOCKED，后台异步切片落盘
            self._auto_capture.notify_impact_locked(int(t_impact))
            payload["auto_capture"] = self._auto_capture.snapshot()

            self._last_impact_emit_time = now
            self._impact_emitted_for_segment = True

            # 课堂大脑：写入 8 大量纲时序并检查疲劳变形（各组别通用）
            self._ingest_fatigue_monitor(payload)
            # 射门数据点已入内存 → 立即快照（不等切片线程；切片成功后再补 clip_path）
            self._persist_checkpoint()

            # GROUP_B：静默写宽表，不向主预览抛触球模态
            if self.experimental_group == GROUP_B or self._policy.silent_recording:
                self._b_group_wide_rows.append(
                    {
                        "timestamp": time.time(),
                        "experimental_group": self.experimental_group,
                        "t_impact": payload.get("t_impact"),
                        "knee_angle": payload.get("impact_knee_angle"),
                        "status": payload.get("knee_status"),
                        "total_score": payload.get("total_score"),
                        "omega_peak": payload.get("whipping_velocity"),
                        "discarded": False,
                    }
                )
                # B 组不把回放帧送 UI，立刻释放副本
                payload.pop("replay_frames", None)
                del replay_frames
                self.log_message.emit(
                    f"GROUP_B：触球锁帧成功 t_impact={payload.get('t_impact')}，"
                    "已写入内存宽表（课中不展示骨骼/反馈）。"
                )
                return

            if self.experimental_group == GROUP_A and self._policy.connect_impact_lock:
                if self._policy.pause_camera_for_capsule:
                    self.pause_for_capsule()
                # 【Sprint 4】diagnostics_ready_signal → 主线程图表 / 时空胶囊
                # 深拷贝 + 有界队列，避免主子线程共享 replay_frames 等可变对象
                self._emit_diagnostics_ready(payload)
                self.log_message.emit(
                    f"GROUP_A：触球锁定 t_impact={payload.get('t_impact')}，"
                    f"时空胶囊 {duration:.1f}s 半速回放已触发。"
                )
            else:
                # GROUP_C：仅记日志，不弹模态；释放回放帧
                payload.pop("replay_frames", None)
                del replay_frames
                self.log_message.emit(
                    f"触球锁帧完成：t_impact={payload.get('t_impact')}，"
                    f"膝角={payload.get('impact_knee_angle')}°，"
                    f"总分={payload.get('total_score')}"
                )
        except Exception as exc:  # noqa: BLE001 - 绝不致死中断教学
            self._discard_shot_cycle(f"exception:{exc}")

    def _update_live_impact_detector(self, omega: float) -> None:
        """摄像头模式：角速度峰值回落后触发一次确定性锁帧。"""
        if not self._auto_capture.accepts_impact_triggers():
            return

        abs_omega = abs(float(omega))
        frame_idx = len(self._trajectory_omega) - 1

        if abs_omega >= _LIVE_OMEGA_PEAK_THRESHOLD and abs_omega >= self._peak_omega_abs:
            self._peak_omega_abs = abs_omega
            self._peak_frame_index = frame_idx
            self._awaiting_post_peak = True
            self._frames_since_peak = 0
            self._impact_emitted_for_segment = False
            self._auto_capture.notify_approach(omega=abs_omega)
            return

        if not self._awaiting_post_peak:
            return

        self._frames_since_peak += 1
        # 峰值后角速度明显回落，或已过观察窗 → 锁帧
        dropped = abs_omega < (self._peak_omega_abs * 0.45)
        if dropped or self._frames_since_peak >= _LIVE_POST_PEAK_FRAMES:
            self._awaiting_post_peak = False
            self._peak_omega_abs = 0.0
            self._emit_impact_if_ready(force=False)

    # ------------------------------------------------------------------
    # QThread 入口
    # ------------------------------------------------------------------

    def run(self) -> None:
        """从线程主体：所有 cv2.read / MediaPipe / 打分均在此执行。"""
        landmarker = None
        cap = None
        sync_frame_count = 0

        try:
            is_video_file_mode = self.data_source == "video"

            if is_video_file_mode:
                if not os.path.exists(self.video_path):
                    self.error_occurred_signal.emit(
                        f"未找到本地视频文件：{self.video_path}"
                    )
                    return
                cap, video_fps, _reported = pt.open_video_capture_deterministic(
                    self.video_path, is_camera=False
                )
            else:
                cap, video_fps, _reported = pt.open_video_capture_deterministic(
                    "", is_camera=True, camera_index=self.camera_index
                )

            if not cap.isOpened():
                msg = (
                    f"无法打开本地视频文件：{self.video_path}"
                    if is_video_file_mode
                    else "无法打开摄像头，请检查是否被其他程序占用。"
                )
                self.error_occurred_signal.emit(msg)
                return

            self._video_fps = float(video_fps) if video_fps and video_fps > 1 else 30.0
            self._auto_capture.fps = float(self._video_fps)
            if is_video_file_mode:
                self._fixed_frame_dt = 1.0 / float(self._video_fps)
                frame_delay_seconds = self._fixed_frame_dt
                self.log_message.emit(
                    f"本地视频文件已打开：{self.video_path}"
                    f"（原始帧率 {self._video_fps:.1f} FPS，同步顺序帧模式｜"
                    f"Active_Group={self.experimental_group}）"
                )
            else:
                self._fixed_frame_dt = 1.0 / float(self._video_fps)
                frame_delay_seconds = 0.0
                self.log_message.emit(
                    f"摄像头已成功打开（Active_Group={self.experimental_group}），"
                    "正在实时检测……"
                )

            need_pose_detection = self.group in ("A", "B", "C")
            if need_pose_detection:
                task_handles = pt.start_analysis_task(reset_yolo=True)
                landmarker = task_handles["pose_landmarker"]

            if self.group == "B":
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
                black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                black_frame = pt.draw_chinese_text_with_backdrop(
                    black_frame,
                    "静默采集中，请专心练习",
                    pt.BLACKSCREEN_FONT,
                    text_color=(230, 230, 230),
                    anchor="center",
                )
                self.frame_ready_signal.emit(black_frame.copy())
                self.log_message.emit(
                    "GROUP_B：骨骼渲染已屏蔽，后台静默写入本地宽表；"
                    "请课后通过「离线课后报告查看」入口查阅。"
                )

            frame_interval_ms = int(round(1000.0 / float(video_fps)))
            frame_timestamp_ms = 0
            consecutive_read_fails = 0
            camera_lost_emitted = False

            while self._running and cap is not None and cap.isOpened():
                if self._is_paused_for_capsule():
                    time.sleep(0.05)
                    continue

                loop_start_time = time.time()

                try:
                    ret, frame = cap.read()
                except Exception as read_exc:  # noqa: BLE001
                    ret, frame = False, None
                    self.log_message.emit(f"读取视频帧异常（计入空帧计数）：{read_exc}")

                if _is_empty_or_failed_frame(ret, frame):
                    if is_video_file_mode:
                        self.log_message.emit(
                            f"本地视频文件已播放完毕（同步读入 {sync_frame_count} 帧），"
                            "训练自动结束。"
                        )
                        break

                    # 【Sprint 5】摄像头空帧 / 读取失败：累计至阈值后发信号并自愈，绝不裸崩
                    consecutive_read_fails += 1
                    if consecutive_read_fails >= _CAMERA_READ_FAIL_LIMIT:
                        if not camera_lost_emitted:
                            self.camera_lost_signal.emit(
                                "摄像头信号丢失，尝试重连..."
                            )
                            camera_lost_emitted = True
                            self.log_message.emit(
                                f"连续 {consecutive_read_fails} 帧 cap.read() 失败，"
                                "触发摄像头自愈（release → open）。"
                            )
                        cap, reopened = self._try_reopen_camera(cap)
                        consecutive_read_fails = 0
                        if not reopened:
                            self.error_occurred_signal.emit(
                                "摄像头自愈失败：无法重新打开设备，请检查连接后重试。"
                            )
                            break
                        # 自愈后继续循环，等待下一帧；不中断 MediaPipe / 训练会话
                        continue
                    time.sleep(0.01)
                    continue

                consecutive_read_fails = 0
                if camera_lost_emitted:
                    camera_lost_emitted = False
                    self.log_message.emit("摄像头信号已恢复，继续静默/实时推流。")

                sync_frame_count += 1
                frame_index = sync_frame_count - 1

                if not is_video_file_mode:
                    frame = cv2.flip(frame, 1)

                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=rgb_frame
                    )
                    frame_timestamp_ms += frame_interval_ms
                    results = landmarker.detect_for_video(
                        mp_image, frame_timestamp_ms
                    )

                    # ----------------------------------------------------------
                    # 【绝对拦截器 / Choke Point】管道最前端：MediaPipe 提完关键点后、
                    # 任何 frame_ready_signal 发射或 Attempt_XX.mp4 切片入缓冲之前，
                    # 必须先用面部高斯模糊覆盖原图。
                    # 这是符合《未成年人保护法》与科研伦理审查的物理级脱敏，任何人不得在此行代码之前进行原图转存。
                    # ----------------------------------------------------------
                    if results.pose_landmarks:
                        frame = pt.apply_facial_anonymization(
                            frame, results.pose_landmarks
                        )

                    if self.group == "C":
                        if results.pose_landmarks:
                            landmarks = results.pose_landmarks[0]
                            try:
                                angle, status, _c, _h, knee_px, ankle_px = (
                                    pt.compute_right_knee_diagnosis(frame, landmarks)
                                )
                                omega = self._append_pose_trajectory(angle, ankle_px)
                                if not is_video_file_mode:
                                    self._update_live_impact_detector(omega)
                            except Exception as traj_exc:  # noqa: BLE001
                                self.error_occurred_signal.emit(
                                    f"C组轨迹计算越界已跳过：{traj_exc}"
                                )
                                self._append_neutral_trajectory()
                        else:
                            self._append_neutral_trajectory()
                        self._emit_frame_ready(frame)

                    elif results.pose_landmarks:
                        landmarks = results.pose_landmarks[0]
                        angle, status, color, hip_px, knee_px, ankle_px = (
                            pt.compute_right_knee_diagnosis(frame, landmarks)
                        )
                        omega = self._append_pose_trajectory(angle, ankle_px)

                        if self.group == "A" and self._policy.render_skeleton:
                            pt.draw_pose_landmarks(frame, results.pose_landmarks)
                            pt.draw_right_knee_overlay(
                                frame, hip_px, knee_px, ankle_px, color, angle, status
                            )
                            self._update_llm_trigger_state(angle, status)
                            frame = self._draw_feedback_text_if_needed(frame)
                            if not self._is_paused_for_capsule():
                                self._emit_frame_ready(frame)
                                self._replay_buffer.append(frame.copy())
                            if not is_video_file_mode:
                                self._update_live_impact_detector(omega)

                        elif self.group == "B":
                            # 断开骨骼连线：仅静默采集；画面已在拦截器处打码
                            # Ghost Monitor：仍向主线程推送脱敏实况，供右下角画中画
                            self._record_b_group_data(angle, status)
                            if not is_video_file_mode:
                                self._update_live_impact_detector(omega)
                            self._emit_frame_ready(frame)
                    else:
                        self._append_neutral_trajectory()
                        if self.group in ("A", "B"):
                            # B 组无姿态时也推幽灵帧，证明引擎仍在跑
                            self._emit_frame_ready(frame)

                    # 【Sprint 2】时间机器：仅接受拦截器之后的安全帧入滚动缓冲
                    # （Attempt_XX.mp4 由此缓冲切片落盘，绝不可写入原图）
                    self._auto_capture.push_frame(frame, frame_index)

                except Exception as frame_exc:  # noqa: BLE001
                    # 单帧异常：上报后继续下一帧，保障操场恶劣环境下的高可用
                    self.error_occurred_signal.emit(
                        f"单帧处理异常（已跳过，画面继续）：{frame_exc}"
                    )
                    self._append_neutral_trajectory()
                    if self.group in ("A", "B", "C"):
                        try:
                            self._emit_frame_ready(frame)
                        except Exception:  # noqa: BLE001
                            pass
                    # 异常帧仍入缓冲，保证索引与轨迹对齐（若已脱敏则仍为安全帧）
                    try:
                        self._auto_capture.push_frame(frame, frame_index)
                    except Exception:  # noqa: BLE001
                        pass

                # 本轮局部引用结束：帮助 GC（cap.read 下一帧会覆盖，显式 del 更稳）
                try:
                    del frame
                except Exception:  # noqa: BLE001
                    pass
                try:
                    del rgb_frame
                    del mp_image
                    del results
                except Exception:  # noqa: BLE001
                    pass

                self._frames_since_gc += 1
                if self._frames_since_gc >= _GC_EVERY_N_FRAMES:
                    self._frames_since_gc = 0
                    gc.collect()

                if is_video_file_mode and frame_delay_seconds > 0:
                    elapsed = time.time() - loop_start_time
                    remaining = frame_delay_seconds - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

            # 录像 EOF 或训练结束：对整段轨迹做一次确定性锁帧打分
            if sync_frame_count > 0 and len(self._trajectory_omega) >= 5:
                self._emit_impact_if_ready(force=True)
            # 若锁帧后仍缺后窗帧（EOF），用缓冲内可得画面立即落盘
            self._auto_capture.finalize()

        except Exception as exc:  # noqa: BLE001
            self.error_occurred_signal.emit(f"后台处理线程发生异常：{exc}")

        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:  # noqa: BLE001
                    pass
            if landmarker is not None:
                try:
                    pt.destroy_pose_landmarker(landmarker)
                except Exception:  # noqa: BLE001
                    pass
            # 释放可能残留的 OpenCV 高窗句柄（摄像头预览调试路径）
            try:
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass
            if self.group == "B":
                self._flush_b_group_data_to_disk()
            try:
                self._auto_capture.finalize()
            except Exception:  # noqa: BLE001
                pass
            attempt_count = int(self._auto_capture.attempt_count)
            # 【Sprint 4】会话收尾：切断所有帧/轨迹引用，防止泄漏跨会话
            self._clear_replay_buffer()
            self._trajectory_angles.clear()
            self._trajectory_omega.clear()
            self._trajectory_ankle_px.clear()
            # 排空有界队列，切断残留 ndarray
            for q in (self._frame_emit_queue, self._diagnostics_emit_queue):
                while True:
                    try:
                        q.get_nowait()
                    except Empty:
                        break
            try:
                self._auto_capture.reset()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._finalize_checkpoint()
            except Exception:  # noqa: BLE001
                pass
            gc.collect()
            self.log_message.emit(
                f"后台处理线程已安全退出，训练结束（本次同步帧数={sync_frame_count}，"
                f"Active_Group={self.experimental_group}，"
                f"自动切片次数={attempt_count}）。"
            )


# 兼容旧名：Sprint 3 及更早代码仍可 from workers import VideoWorker
VideoWorker = InferenceWorker
