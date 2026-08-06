"""shot_analysis_service.py — 射门分析管线（纯业务逻辑，无 FastAPI/WebSocket 依赖）

ShotAnalysisPipeline 接收 push_fn 和 on_completed 两个回调，与传输层完全解耦，
可在无网络/无 WebSocket 的单测环境中直接实例化并调用 run()。

《未成年人保护法》脱敏要求：
  面部匿名化（apply_facial_anonymization）在关键点提取后立即执行，
  先于任何缓存写入、击球关键帧捕捉和骨骼线叠加——此顺序不可更改。
"""
from __future__ import annotations

import sys

# --------------------------------------------------------------------------
# UTF-8 双保险：防止 Windows 默认 cp936 控制台编码导致中文打印崩溃
# --------------------------------------------------------------------------
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:
    pass


def safe_print(*args, **kwargs) -> None:
    """打印的同时处理 UnicodeEncodeError（第二道防线）。"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_args = [
            arg.encode(encoding, errors="replace").decode(encoding, errors="replace")
            if isinstance(arg, str) else arg
            for arg in args
        ]
        try:
            print(*safe_args, **kwargs)
        except Exception:
            pass


# --------------------------------------------------------------------------
# 标准库
# --------------------------------------------------------------------------
import base64
import collections
import os
import queue
import threading
import time
from typing import Any, Callable, Optional

# --------------------------------------------------------------------------
# 第三方 / 项目内部
# --------------------------------------------------------------------------
import cv2
import mediapipe as mp
import numpy as np

import error_diagnoser
import pose_tracker as pt
from session_baseline import stamp_baseline_watermark, get_session_metadata_store

# --------------------------------------------------------------------------
# 常量（从 api_server.py 迁移；api_server.py 保留同名引用以向下兼容）
# --------------------------------------------------------------------------
MAX_TRANSMIT_WIDTH = 800
JPEG_QUALITY = 75
STABILITY_WINDOW_SIZE = 30
IMPACT_FRAME_JPEG_QUALITY = 90
BLACK_FRAME_MEAN_BRIGHTNESS_THRESHOLD = 6.0
BLACK_FRAME_CONSECUTIVE_LIMIT = 15
FRAME_PROGRESS_LOG_INTERVAL = 60
CAMERA_READ_FAIL_LIMIT = 50
CAMERA_REOPEN_SLEEP_SEC = 0.45

# --------------------------------------------------------------------------
# 【动作相位时序切分】五阶段帧索引区间，全部以 t_impact 为绝对基准点做相对偏移。
#
# 关键帧与实际视频错位的根因是各阶段各自用毫秒阈值/时间戳反查下标，
# 一旦 fps 抖动或时间戳插值就会漂移。此处改为纯整数相对帧偏移：
# t_impact 由空间距离极小值锁定（pose_tracker.locate_impact_frame_with_quality
# 内 w_prox 距离极小门控），其余四段一律 t_impact ± N 帧，零漂移可复现。
#
# 取值为 (start_offset, end_offset)，闭区间，相对 t_impact，
# 且相邻两段端点互不重叠（-16/-15、-11/-10 边界各归一段），避免同一帧被
# 两个相位重复计数。
#
# 注意：下表的帧数是以 PHASE_OFFSET_REFERENCE_FPS（30fps）为基准标定的，
# 代表的是固定的生理时长（助跑 -1.5s..-0.533s、支撑 -0.5s..-0.367s、
# 折叠 -0.333s..-0.033s、随前 +0.033s..+0.5s）。实际视频 fps 不等于 30 时，
# build_phase_windows() 会按 fps/30 整数取整重缩放，使各相位覆盖的时长恒定，
# 同时保持"纯整数帧偏移、零漂移可复现"的原始约束（不引入时间戳反查）。
# --------------------------------------------------------------------------
PHASE_OFFSET_REFERENCE_FPS: float = 30.0
PHASE_FRAME_OFFSETS: tuple[tuple[str, int, int], ...] = (
    ("approach", -45, -16),        # 助跑
    ("plant", -15, -11),           # 支撑
    ("fold", -10, -1),             # 折叠
    ("impact", 0, 0),              # 射门
    ("follow_through", 1, 15),     # 随前
)


# --------------------------------------------------------------------------
# 工具函数（同步保留在 workers/inference_worker.py；此处独立定义，供管线内部使用）
# --------------------------------------------------------------------------

def _is_empty_or_failed_frame(ret: bool, frame: Any) -> bool:
    """判断 cap.read() 的返回值是否代表一帧有效画面。"""
    if not ret or frame is None:
        return True
    try:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return True
    except Exception:  # noqa: BLE001
        return True
    return False


# --------------------------------------------------------------------------
# pose_tracker 兼容转发（api_server.py 亦保留同名转发，供既有调用方使用）
# --------------------------------------------------------------------------

def resolve_leg_annotation_target(
    score_detail: Optional[dict], *, t_impact: Optional[int] = None,
) -> tuple[int, str, str]:
    return pt.resolve_leg_annotation_target(score_detail, t_impact=t_impact)


def draw_biomechanics_annotation(frame, metrics: dict):
    return pt.draw_biomechanics_annotation(frame, metrics)


class ShotAnalysisPipeline:
    """射门分析计算管线：读帧 → 姿态检测 → 力学诊断 → 脱敏 → 渲染 → push_fn。

    传输层通过两个回调注入，管线本身不认识 WebSocket / FastAPI / queue：
      push_fn(payload: dict)  —— 输出一份帧数据或控制消息；
      on_completed()          —— 标记任务完成（必须早于 stopped 消息推送）。
    """

    def __init__(
        self,
        *,
        session_id: str,
        source: str,
        video_path: Optional[str] = None,
        camera_index: int = 0,
        push_fn: Callable[[dict], None],
        on_completed: Optional[Callable[[], None]] = None,
        records: Optional[list] = None,
        records_lock: Optional[threading.Lock] = None,
        stop_event: Optional[threading.Event] = None,
        status_provider: Optional[Callable[[], str]] = None,
    ) -> None:
        self.session_id = session_id
        self.source = source  # "webcam" | "file"
        self.video_path = video_path
        self.camera_index = camera_index

        # 传输/生命周期回调（由 AnalysisSession 或单测注入）
        self._push_fn = push_fn
        self._on_completed = on_completed or (lambda: None)
        self._status_provider = status_provider
        self.stop_event = stop_event if stop_event is not None else threading.Event()

        # 诊断记录：可由宿主 AnalysisSession 共享同一 list + 锁
        self.records: list[dict] = records if records is not None else []
        self._records_lock = records_lock if records_lock is not None else threading.Lock()

        # 【实时动力链角速度监控】上一帧角度与时间戳，用于逐帧计算角速度（deg/s）
        self._prev_angle: Optional[float] = None
        self._prev_frame_time: Optional[float] = None
        # 录像分析：用固定 fps 推导 Δt，杜绝墙钟抖动导致角速度/触球帧跳变
        self._fixed_frame_dt: Optional[float] = None
        # 【Phase 4】真实采集帧率；综合报告打分时传入 Scorer（踝冲击窗按时长自适应）
        self._video_fps: float = 30.0
        # 最近 STABILITY_WINDOW_SIZE 帧的角速度滑动窗口，用于计算"动平衡稳定指数"
        self._velocity_window: "collections.deque[float]" = collections.deque(
            maxlen=STABILITY_WINDOW_SIZE
        )

        # 【击球关键帧自动捕捉】整趟练习中"右膝角速度绝对值最大"的那一帧
        self.impact_frame = None  # 已完成面部打码、未叠加骨骼线的 numpy 数组
        self.impact_metrics: Optional[dict] = None
        self._best_impact_score: float = -1.0
        # 【V2.5/V2.6】逐帧轨迹缓存；结束后抛物线粗锁 + CubicSpline 120Hz 精修
        self._trajectory_angles: list[float] = []
        self._trajectory_omega: list[float] = []
        self._trajectory_ankle_px: list[tuple] = []
        self._trajectory_frames_blurred: list = []
        # 【Sprint 1】逐帧姿态关键点（支撑踝 / 摆腿踝等），供时空热力图坐标映射
        self._trajectory_pose_frames: list[dict] = []
        # 【V3.11】YOLO 足球中心轨迹（与帧索引对齐；未检出为 None）
        self._trajectory_ball_px: list[Optional[tuple]] = []
        self._trajectory_ball_diameter_px: list[Optional[float]] = []
        # 场地单应性 / pixel→meter 标定器（可选）
        self._field_calibrator: Any = None
        # 射门结果闭环：出球初速度 / 发射仰角
        self.ball_speed_kmh: Optional[float] = None
        self.launch_angle_deg: Optional[float] = None
        self.ball_outcome_meta: dict = {}
        # 脱敏帧 JPEG 缓存：报告阶段按折叠极值帧重标定大小腿夹角标注
        self._blurred_jpeg_by_index: dict[int, bytes] = {}
        self._blurred_jpeg_ring: "collections.deque[tuple[int, bytes]]" = collections.deque(
            maxlen=150
        )
        self._store_all_blurred_jpeg: bool = source == "file"
        self.t_impact: Optional[int] = None
        self.sync_frame_count: int = 0

        # 【黑屏问题自动诊断】累计推送帧数 + 连续疑似全黑帧计数
        self._pushed_frame_count = 0
        self._consecutive_dark_frames = 0
        self._dark_frame_warning_sent = False

    # ----------------------------------------------------------------------
    # 运动学计算
    # ----------------------------------------------------------------------
    def _compute_angular_velocity(self, angle: float) -> float:
        """根据"当前帧角度 - 上一帧角度"除以帧间时间差，计算右膝角速度（deg/s）。
        第一帧因为没有"上一帧"可比较，约定角速度为 0。

        【V2.5】录像模式优先使用固定 fps 的 Δt，避免 wall-clock 抖动导致分数跳变。
        """
        angular_velocity = 0.0
        if self._prev_angle is not None:
            if self._fixed_frame_dt is not None and self._fixed_frame_dt > 0:
                dt = self._fixed_frame_dt
            else:
                now = time.time()
                dt = (now - self._prev_frame_time) if self._prev_frame_time is not None else 0.0
            if dt > 0:
                angular_velocity = (angle - self._prev_angle) / dt
        self._prev_angle = angle
        self._prev_frame_time = time.time()
        self._velocity_window.append(angular_velocity)
        return angular_velocity

    def _compute_stability_index(self) -> int:
        """根据最近滑动窗口内角速度的离散程度（标准差）换算「动平衡稳定指数」（0-100）。

        设计思路：角速度标准差越小，说明摆动腿发力节奏越连贯、动作越"不抖"，
        对应稳定指数越高；一旦出现忽快忽慢的剧烈波动，标准差变大，指数随之下降。
        """
        if len(self._velocity_window) < 2:
            return 100
        values = list(self._velocity_window)
        mean_value = sum(values) / len(values)
        variance = sum((v - mean_value) ** 2 for v in values) / len(values)
        std_dev = variance ** 0.5
        # 经验系数：标准差每增加 4 deg/s 扣 1 分，兜底裁剪到 [0, 100] 区间
        index = 100 - std_dev / 4.0
        return int(max(0, min(100, round(index))))

    # ----------------------------------------------------------------------
    # 击球关键帧捕捉 / 脱敏帧缓存
    # ----------------------------------------------------------------------
    def _capture_impact_candidate(self, frame, landmarks, hip_px, knee_px, ankle_px, angle, status):
        """【击球关键帧自动捕捉】把当前帧记录为"击球关键帧"候选：
        整趟练习结束后，self.impact_frame 会一直保留角速度绝对值最大（即冲击最
        剧烈、最贴近真实触球瞬间）的那一帧，供 /api/generate_report 生成矢量标注图。

        重要：必须在 pt.apply_facial_anonymization()（或兼容别名 apply_face_blur）
        之后、pt.draw_pose_landmarks() / pt.draw_right_knee_overlay() 之前调用——
        既要保证脸部已经打码（隐私红线），又要保证存下来的是一张"干净"的画面，
        方便后续单独叠加矢量标注，不会与实时预览用的白色骨骼线/粗染色线互相干扰。
        """
        height, width = frame.shape[:2]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        mid_hip_px = (
            int((left_hip.x + right_hip.x) / 2 * width),
            int((left_hip.y + right_hip.y) / 2 * height),
        )

        # 【隐私红线】落盘/缓存前再次强制脱敏，杜绝任何旁路漏网的原始带脸帧
        safe_frame = pt.apply_facial_anonymization(frame, landmarks)
        self.impact_frame = safe_frame.copy()
        self.impact_metrics = {
            "hip_px": hip_px,
            "knee_px": knee_px,
            "ankle_px": ankle_px,
            "mid_hip_px": mid_hip_px,
            "angle": angle,
            "status": status,
            "overlay_ok": True,
            "label": "大小腿夹角",
            "swing_side": "right",
        }

    def _cache_blurred_frame(self, frame_index: int, frame_bgr) -> None:
        """缓存脱敏帧 JPEG，供报告阶段按折叠/触球索引取回对齐画面。"""
        try:
            ok, buf = cv2.imencode(
                ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, IMPACT_FRAME_JPEG_QUALITY]
            )
            if not ok:
                return
            raw = buf.tobytes()
            idx = int(frame_index)
            if self._store_all_blurred_jpeg:
                self._blurred_jpeg_by_index[idx] = raw
            else:
                self._blurred_jpeg_ring.append((idx, raw))
        except Exception:  # noqa: BLE001
            return

    def get_blurred_frame(self, frame_index: int):
        """按帧索引取回脱敏 BGR；缺失返回 None。"""
        idx = int(frame_index)
        raw = self._blurred_jpeg_by_index.get(idx)
        if raw is None:
            for ring_idx, ring_raw in self._blurred_jpeg_ring:
                if int(ring_idx) == idx:
                    raw = ring_raw
                    break
        if raw is None:
            return None
        try:
            arr = np.frombuffer(raw, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            return None

    # ----------------------------------------------------------------------
    # 大小腿夹角标注重建
    # ----------------------------------------------------------------------
    def rebuild_leg_annotation(
        self,
        score_detail: Optional[dict] = None,
        *,
        t_impact: Optional[int] = None,
    ) -> tuple:
        """按折叠极值帧 + 摆动腿重建大小腿夹角标注（画面与关键点同源）。

        返回 ``(frame_bgr|None, metrics|None)``；失败时回退流式 impact_frame。
        """
        from biomech_primitives import infer_swing_leg_side

        frames = list(self._trajectory_pose_frames)
        t_ref = int(t_impact) if t_impact is not None else int(self.t_impact or 0)
        frame_idx, side, label = resolve_leg_annotation_target(
            score_detail, t_impact=t_ref
        )
        if isinstance(score_detail, dict) and score_detail.get("swing_leg") in (
            "left",
            "right",
        ):
            side = str(score_detail["swing_leg"])
        elif frames:
            side = infer_swing_leg_side(frames, frame_idx, explicit_side=side)

        n = len(frames)
        candidates: list[int] = []
        for base in (frame_idx, t_ref):
            b = int(base)
            if b not in candidates:
                candidates.append(b)
            for delta in (-2, -1, 1, 2, -4, 4):
                c = b + delta
                if c not in candidates:
                    candidates.append(c)

        best_metrics = None
        best_frame = None
        for cand in candidates:
            if n <= 0 or not (0 <= cand < n):
                continue
            metrics = pt.build_annotation_metrics_from_pose_record(
                frames[cand], side=side, label=label
            )
            if metrics is None:
                continue
            img = self.get_blurred_frame(cand)
            if img is None:
                continue
            metrics["annotation_frame_index"] = int(cand)
            if metrics.get("overlay_ok"):
                return img, metrics
            if best_metrics is None:
                best_metrics = metrics
                best_frame = img

        if best_frame is not None and best_metrics is not None:
            return best_frame, best_metrics

        if self.impact_frame is not None and self.impact_metrics is not None:
            legacy = dict(self.impact_metrics)
            qa = pt.evaluate_leg_overlay_geometry(
                legacy["hip_px"], legacy["knee_px"], legacy["ankle_px"]
            )
            legacy["overlay_ok"] = bool(qa.get("ok"))
            legacy["overlay_qa"] = qa
            legacy.setdefault("label", label)
            legacy.setdefault("swing_side", side)
            return self.impact_frame, legacy
        return None, None

    # ----------------------------------------------------------------------
    # 抛物线锁帧
    # ----------------------------------------------------------------------
    def _finalize_impact_with_parabolic_lock(self) -> None:
        """分析结束后用 locate_impact_frame（抛物线插值）覆写流式峰值候选，零漂移锁帧。"""
        n = len(self._trajectory_omega)
        if n < 3 or len(self._trajectory_ankle_px) < n:
            if n > 0:
                self.t_impact = int(max(range(n), key=lambda i: abs(self._trajectory_omega[i])))
            return

        # 球心代理：若无独立球检测，用整段踝坐标中位数作为静止球近似（操场固定机位）
        ankles = self._trajectory_ankle_px[:n]
        xs = [float(a[0]) for a in ankles]
        ys = [float(a[1]) for a in ankles]
        # 取踝轨迹 Y 较大的 15% 分位中位数作为触地球区近似球心
        order = sorted(range(n), key=lambda i: ys[i])
        tail = order[int(n * 0.85):] or order[-1:]
        ball_x = float(sum(xs[i] for i in tail) / len(tail))
        ball_y = float(sum(ys[i] for i in tail) / len(tail))
        ball_coords = [(ball_x, ball_y) for _ in range(n)]

        omega_smooth = pt.KinematicSignalProcessor.smooth_joint_trajectories(
            list(self._trajectory_omega[:n])
        )
        t_impact, t0_quality = pt.locate_impact_frame_with_quality(
            omega_smooth, ankles, ball_coords
        )
        self.t_impact = int(t_impact)
        safe_print(
            f"【shot_analysis_service】[V2.6] 触球锁帧 t_impact={self.t_impact} "
            f"t0_quality={t0_quality} "
            f"（同步总帧数 frame_count={self.sync_frame_count}）",
            flush=True,
        )
        # T0 锁定后立刻结算出球初速度 / 发射仰角
        self._finalize_ball_outcome()

    def set_field_calibrator(self, calibrator: Any) -> None:
        """注入场地单应性矩阵或 pixel_to_meter 标定器（可选）。"""
        self._field_calibrator = calibrator

    def _append_ball_detection(self, frame_bgr, yolo_model: Any) -> None:
        """逐帧写入 YOLO 球心；失败时对齐填 None，保证与帧索引等长。"""
        center = None
        diameter = None
        if yolo_model is not None and frame_bgr is not None:
            try:
                results = pt.yolo_detect_frame(yolo_model, frame_bgr)
                extracted = pt.extract_sports_ball_center(results)
                if extracted is not None:
                    center = (float(extracted[0]), float(extracted[1]))
                    diameter = float(extracted[2])
            except Exception:  # noqa: BLE001
                center = None
                diameter = None
        self._trajectory_ball_px.append(center)
        self._trajectory_ball_diameter_px.append(diameter)

    def _collect_ball_window_after_t0(self, t0: int, n_frames: int = 4) -> list:
        """从轨迹缓冲（Rolling 全长序列）取 ``[T0, T0+n_frames)`` 球心。"""
        balls = list(self._trajectory_ball_px)
        n = len(balls)
        if n <= 0 or t0 < 0 or t0 >= n:
            return []
        window = []
        for i in range(t0, min(n, t0 + int(n_frames))):
            window.append(balls[i])
        return window

    def _finalize_ball_outcome(self) -> None:
        """T0 锁定后：取 T0..T0+3 球心 → calculate_ball_outcome。"""
        self.ball_speed_kmh = None
        self.launch_angle_deg = None
        self.ball_outcome_meta = {"ok": False, "reason": "not_computed"}
        try:
            from biomech_primitives import (
                DEFAULT_EMPIRICAL_PCR,
                STANDARD_BALL_DIAMETER_CM,
                calculate_ball_outcome,
            )

            t0 = int(self.t_impact) if self.t_impact is not None else -1
            window = self._collect_ball_window_after_t0(t0, n_frames=4)
            if sum(1 for p in window if p is not None) < 2:
                self.ball_outcome_meta = {
                    "ok": False,
                    "reason": "insufficient_yolo_ball_points",
                    "window_len": len(window),
                }
                return

            fps = float(self._video_fps) if self._video_fps and self._video_fps > 1 else 30.0
            if self._fixed_frame_dt and self._fixed_frame_dt > 0:
                fps = float(1.0 / self._fixed_frame_dt)

            # T0 球框直径 → PCR 覆盖；否则默认经验 PCR
            pcr = None
            if 0 <= t0 < len(self._trajectory_ball_diameter_px):
                diam = self._trajectory_ball_diameter_px[t0]
                if diam is not None and float(diam) >= 10.0:
                    pcr = float(STANDARD_BALL_DIAMETER_CM) / float(diam)

            outcome = calculate_ball_outcome(
                window,
                fps=fps,
                calibrator=self._field_calibrator,
                pcr_cm_per_px=pcr if pcr is not None else float(DEFAULT_EMPIRICAL_PCR),
            )
            self.ball_outcome_meta = dict(outcome)
            if outcome.get("ok"):
                self.ball_speed_kmh = outcome.get("ball_speed_kmh")
                self.launch_angle_deg = outcome.get("launch_angle_deg")
                safe_print(
                    f"【shot_analysis_service】[V3.11] 出球结果 "
                    f"speed={self.ball_speed_kmh} km/h "
                    f"launch={self.launch_angle_deg}° "
                    f"scale={outcome.get('scale_method')}",
                    flush=True,
                )
            else:
                safe_print(
                    f"【shot_analysis_service】[V3.11] 出球结果未解算："
                    f"{outcome.get('reason')}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            self.ball_outcome_meta = {"ok": False, "reason": f"exception:{exc}"}

    def get_ball_outcome(self) -> dict:
        """返回出球初速度 / 仰角（供 generate_report 注入 score_detail）。"""
        return {
            "ball_speed_kmh": self.ball_speed_kmh,
            "launch_angle_deg": self.launch_angle_deg,
            "meta": dict(self.ball_outcome_meta or {}),
        }

    def inject_ball_outcome_into_score_detail(self, score_detail: Optional[dict]) -> dict:
        """把球速 / 仰角写入 score_detail 顶层与 indicators。"""
        detail = dict(score_detail or {})
        speed = self.ball_speed_kmh
        angle = self.launch_angle_deg
        meta = dict(self.ball_outcome_meta or {})
        detail["ball_speed_kmh"] = speed
        detail["launch_angle_deg"] = angle
        detail["ball_outcome"] = {
            "ball_speed_kmh": speed,
            "launch_angle_deg": angle,
            "ok": bool(meta.get("ok")),
            "scale_method": meta.get("scale_method"),
            "displacement_m": meta.get("displacement_m"),
            "dt_sec": meta.get("dt_sec"),
            "reason": meta.get("reason"),
        }
        indicators = dict(detail.get("indicators") or {})
        t0 = int(self.t_impact) if self.t_impact is not None else 0
        if speed is not None:
            indicators["ball_speed_kmh"] = {
                "value": round(float(speed), 2),
                "unit": "km/h",
                "status": "GREEN_OPTIMAL",
                "penalty": 0.0,
                "provenance": "measured",
                "method": str(meta.get("scale_method") or "ball_outcome"),
                "extreme_frame_index": t0,
            }
        else:
            indicators["ball_speed_kmh"] = {
                "value": None,
                "unit": "km/h",
                "status": "YELLOW_APPROACHING",
                "penalty": 0.0,
                "provenance": "missing",
                "method": "ball_outcome_unavailable",
                "note": meta.get("reason") or "未检出出球轨迹",
                "extreme_frame_index": t0,
            }
        if angle is not None:
            indicators["launch_angle_deg"] = {
                "value": round(float(angle), 2),
                "unit": "deg",
                "status": "GREEN_OPTIMAL",
                "penalty": 0.0,
                "provenance": "measured",
                "method": str(meta.get("scale_method") or "ball_outcome"),
                "extreme_frame_index": t0,
            }
        else:
            indicators["launch_angle_deg"] = {
                "value": None,
                "unit": "deg",
                "status": "YELLOW_APPROACHING",
                "penalty": 0.0,
                "provenance": "missing",
                "method": "ball_outcome_unavailable",
                "note": meta.get("reason") or "未检出出球轨迹",
                "extreme_frame_index": t0,
            }
        detail["indicators"] = indicators
        # 实验防干扰：强制打上基线环境水印（含 is_baseline_trusted）
        return self.stamp_session_baseline(detail)

    def stamp_session_baseline(self, score_detail: Optional[dict]) -> dict:
        """将锁定的 SessionCheckpoint 环境参数写入 score_detail。"""
        return stamp_baseline_watermark(
            score_detail,
            analysis_session_id=getattr(self, "session_id", None),
        )

    # ------------------------------------------------------------------
    # 评分/时序载荷构造
    # ------------------------------------------------------------------
    def build_scoring_payloads(self) -> tuple[dict, dict]:
        """从本会话轨迹构造 DeterministicScorer 所需的 impact / trajectory 载荷。"""
        knee_angles = list(self._trajectory_angles)
        omega = list(self._trajectory_omega)
        pose_frames = list(self._trajectory_pose_frames)
        n = len(knee_angles)
        t_impact = int(self.t_impact) if self.t_impact is not None else (
            int(max(range(n), key=lambda i: abs(omega[i]))) if n > 0 else 0
        )
        if n > 0:
            t_impact = int(max(0, min(n - 1, t_impact)))

        fps = float(self._video_fps) if self._video_fps and self._video_fps > 1 else 30.0
        if self._fixed_frame_dt and self._fixed_frame_dt > 0:
            fps = float(1.0 / self._fixed_frame_dt)

        # 触球帧：按摆动腿推断支撑踝；无球框时用摆动足尖代理球心，但只计量「体侧横距」
        ball_center = None
        support_lateral_dist_cm = None
        support_ratio = None
        support_distance_method = None
        support_ankle_px = None
        swing_side = "right"
        if pose_frames and 0 <= t_impact < len(pose_frames):
            rec = pose_frames[t_impact]
            world = rec.get("world") if isinstance(rec, dict) else None
            try:
                from biomech_primitives import (
                    calculate_support_offset_by_shoulder_ratio,
                    infer_swing_leg_side,
                )

                swing_side = infer_swing_leg_side(pose_frames, t_impact)
            except Exception:  # noqa: BLE001
                swing_side = "right"
            support_key = "left_ankle" if swing_side == "right" else "right_ankle"
            swing_foot_key = (
                "right_foot_index" if swing_side == "right" else "left_foot_index"
            )
            swing_ankle_key = "right_ankle" if swing_side == "right" else "left_ankle"

            if isinstance(world, dict):
                ball_center = (
                    world.get(swing_foot_key)
                    or world.get(swing_ankle_key)
                    or world.get("right_foot_index")
                    or world.get("right_ankle")
                )
            elif isinstance(rec, dict):
                ball_center = (
                    rec.get(swing_foot_key)
                    or rec.get(swing_ankle_key)
                    or rec.get("right_foot_index")
                    or rec.get("right_ankle")
                )
            # 【V3.7】禁止 world ΔX×100；无球框时肩宽归一化 + 体侧向横距（不含前后步幅）
            if isinstance(world, dict):
                try:
                    from biomech_primitives import calculate_support_offset_by_shoulder_ratio

                    support_w = world.get(support_key)
                    ball_w = (
                        world.get(swing_foot_key)
                        or world.get(swing_ankle_key)
                        or ball_center
                    )
                    detail = calculate_support_offset_by_shoulder_ratio(
                        support_w,
                        ball_w,
                        world.get("left_shoulder"),
                        world.get("right_shoulder"),
                        coord_space="world_m",
                        distance_mode="lateral",
                    )
                    if detail.get("ok"):
                        support_lateral_dist_cm = float(detail["distance_cm_estimate"])
                        support_ratio = float(detail["support_ratio"])
                        support_distance_method = "shoulder_width_ratio"
                except Exception:  # noqa: BLE001
                    support_lateral_dist_cm = None
                    support_ratio = None
                    support_distance_method = None
            if isinstance(rec, dict):
                support_px = rec.get(support_key)
                if support_px is not None:
                    try:
                        support_ankle_px = (float(support_px[0]), float(support_px[1]))
                    except (TypeError, ValueError, IndexError):
                        support_ankle_px = None

        impact_metrics = self.impact_metrics or {}
        impact_frame_data = {
            "t_impact": t_impact,
            "task_id": self.session_id,
            "session_id": self.session_id,
            "total_frames": n,
            "frames": pose_frames,
            "ball_center": ball_center,
            "fps": fps,
            "impact_knee_angle": impact_metrics.get("angle"),
            "distance_cm": impact_metrics.get("distance_cm"),
            "toe_angle": impact_metrics.get("toe_angle"),
            "support_knee_angle": impact_metrics.get("support_knee_angle"),
            "hip_torsion_angle": impact_metrics.get("hip_torsion_angle"),
        }
        if support_lateral_dist_cm is not None:
            # 仅作估计 cm（肩宽归一化）；禁止再标成绝对 world 实测
            impact_frame_data["support_lateral_dist_cm"] = round(
                float(support_lateral_dist_cm), 2
            )
            impact_frame_data["distance_cm_estimate"] = round(
                float(support_lateral_dist_cm), 2
            )
            impact_frame_data["support_distance_method"] = (
                support_distance_method or "shoulder_width_ratio"
            )
        if support_ratio is not None:
            impact_frame_data["support_ratio"] = round(float(support_ratio), 4)
        if support_ankle_px is not None:
            impact_frame_data["support_ankle_px"] = support_ankle_px
        impact_frame_data["swing_leg"] = swing_side
        dt = float(self._fixed_frame_dt) if self._fixed_frame_dt else (1.0 / fps)
        trajectory_data = {
            "task_id": self.session_id,
            "session_id": self.session_id,
            "knee_angles": knee_angles,
            "angular_velocities": omega,
            "timestamps_sec": [i * dt for i in range(n)],
            "total_frames": n,
            "t_impact": t_impact,
            "frames": pose_frames,
            "ball_center": ball_center,
            "fps": fps,
            "whipping_velocity": float(max((abs(v) for v in omega), default=0.0)),
            "swing_leg": swing_side,
        }
        if support_lateral_dist_cm is not None:
            trajectory_data["support_lateral_dist_cm"] = round(
                float(support_lateral_dist_cm), 2
            )
            trajectory_data["support_distance_method"] = (
                support_distance_method or "shoulder_width_ratio"
            )
        if support_ratio is not None:
            trajectory_data["support_ratio"] = round(float(support_ratio), 4)

        # 【V3.11】出球结果：若尚未结算（例如摄像头峰值锁帧）则此处补算
        if self.ball_speed_kmh is None and self.t_impact is not None:
            self._finalize_ball_outcome()
        if self.ball_speed_kmh is not None:
            impact_frame_data["ball_speed_kmh"] = float(self.ball_speed_kmh)
            trajectory_data["ball_speed_kmh"] = float(self.ball_speed_kmh)
        if self.launch_angle_deg is not None:
            impact_frame_data["launch_angle_deg"] = float(self.launch_angle_deg)
            trajectory_data["launch_angle_deg"] = float(self.launch_angle_deg)
        if self.ball_outcome_meta:
            impact_frame_data["ball_outcome"] = dict(self.ball_outcome_meta)
            trajectory_data["ball_outcome"] = dict(self.ball_outcome_meta)

        return impact_frame_data, trajectory_data

    def build_time_series_velocity_window(
        self, t_impact: Optional[int] = None
    ) -> tuple[list[float], int, int, list[float]]:
        """裁剪 Action ROI 内的摆动腿小腿连续角速度序列（KinematicSignalProcessor 平滑后）。

        窗口为 ``[t_impact-30, t_impact+30)``（最长约 60 帧）。返回：
            (time_series_velocity, impact_index_in_window, roi_start, absolute_timestamps)
        其中 ``absolute_timestamps[i] = (roi_start + i) / fps``，为原视频绝对秒。
        边界未截断时 ``impact_index_in_window`` 恒为 30（数组中心）。
        """
        omega_raw = list(self._trajectory_omega)
        n = len(omega_raw)
        if n <= 0:
            return [], 0, 0, []

        omega_smooth = pt.KinematicSignalProcessor.smooth_joint_trajectories(omega_raw)
        if t_impact is None:
            t_impact = int(self.t_impact) if self.t_impact is not None else int(
                max(range(n), key=lambda i: abs(float(omega_smooth[i])))
            )
        t = int(max(0, min(n - 1, int(t_impact))))
        roi_start, roi_end = error_diagnoser.slice_action_roi_bounds(t, n)
        window = [round(float(v), 2) for v in omega_smooth[roi_start:roi_end]]
        impact_index_in_window = int(t - roi_start)
        fps = float(self._video_fps) if self._video_fps and self._video_fps > 1 else 30.0
        packed = pt.pack_action_roi_series(window, int(roi_start), fps)
        absolute_timestamps = list(packed.get("absolute_timestamps") or [])
        return window, impact_index_in_window, int(roi_start), absolute_timestamps

    # ------------------------------------------------------------------
    # 动作相位时序切分（以 t_impact 为绝对基准点）
    # ------------------------------------------------------------------
    def build_phase_windows(
        self,
        t_impact: Optional[int] = None,
        total_frames: Optional[int] = None,
    ) -> dict[str, dict]:
        """按 :data:`PHASE_FRAME_OFFSETS` 切分五个动作相位的帧索引闭区间。

        所有阶段一律以 ``t_impact`` 为绝对基准点做整数帧相对偏移，不再经由
        毫秒阈值反查时间戳下标，因此同一段视频必然零漂移复现，前端按
        ``start_frame / end_frame`` 直接 seek 即与实际画面对齐。

        :data:`PHASE_FRAME_OFFSETS` 的帧数以 :data:`PHASE_OFFSET_REFERENCE_FPS`
        （30fps）标定，代表固定生理时长。实际 fps 不等于 30 时，本方法按
        ``scale = fps / 30`` 对每个偏移四舍五入重缩放，使各相位覆盖的时长恒定；
        缩放全程只做整数帧运算，不引入时间戳反查。缩放后还会做两项归一：
        ``impact`` 恒为 ``(0, 0)``，非零偏移不允许被取整成 0（即不跨越触球帧），
        且按时间顺序强制 ``start > 上一段 end``，保证五段仍严格互不重叠。

        参数:
            t_impact: 触球帧绝对索引；``None`` 时取 ``self.t_impact``，
                仍为空则回退到 |ω| 峰值帧。
            total_frames: 序列总帧数；``None`` 时取轨迹长度。

        返回:
            ``{phase_name: {...}}``。每个相位含：

            - ``start_frame`` / ``end_frame``：钳制到 ``[0, total_frames-1]``
              的闭区间下标，可直接切片消费；
            - ``start_offset`` / ``end_offset``：相对 ``t_impact`` 实际生效的
              偏移（已按 fps 重缩放），便于前端标注"触球前 45 帧"等文案；
            - ``reference_start_offset`` / ``reference_end_offset``：重缩放前的
              30fps 基准偏移，便于排查缩放结果；
            - ``clamped``：该区间是否因视频边界被裁剪；
            - ``truncated``：区间是否已完全落在视频之外（越界空窗）。

            ``_meta`` 额外给出 ``reference_fps`` / ``effective_fps`` /
            ``offset_scale``，用于核对本次实际使用的缩放系数。

        边界处理：相减后 < 0 或超出总帧数的下标一律钳制进合法范围；若整段
        区间都在视频之外（例如只录到触球后 3 帧却要 +15 帧的随前），则退化为
        贴边单帧并标记 ``truncated=True``，绝不返回负下标或倒序区间。
        """
        n = int(total_frames) if total_frames is not None else len(self._trajectory_omega)
        if n <= 0:
            return {}

        if t_impact is None:
            t_impact = self.t_impact
        if t_impact is None:
            omega = list(self._trajectory_omega)
            t_impact = (
                int(max(range(len(omega)), key=lambda i: abs(float(omega[i]))))
                if omega
                else 0
            )
        t0 = int(max(0, min(n - 1, int(t_impact))))

        # --- fps 重缩放：把 30fps 基准标定的整数偏移换算到本视频的实际帧率 ---
        # 仍然是纯整数帧运算（不做时间戳反查），因此"零漂移可复现"的约束不变。
        fps = float(self._video_fps) if self._video_fps and self._video_fps > 1 else 30.0
        reference_fps = float(PHASE_OFFSET_REFERENCE_FPS) or 30.0
        scale = fps / reference_fps

        def _scale_offset(value: int) -> int:
            """按 fps 比例缩放单个偏移，且不允许跨越触球帧（符号必须保持）。"""
            if value == 0:
                return 0
            scaled = int(round(float(value) * scale))
            return min(-1, scaled) if value < 0 else max(1, scaled)

        # 取整后相邻段可能出现端点重叠或倒序：按原表顺序（时间递增）逐段归一，
        # 保证每段 start <= end 且严格晚于前一段的 end，维持互不重叠。
        applied_offsets: list[tuple[str, int, int]] = []
        prev_end: Optional[int] = None
        for name, ref_start, ref_end in PHASE_FRAME_OFFSETS:
            if name == "impact":
                start_offset = end_offset = 0
            else:
                start_offset = _scale_offset(int(ref_start))
                end_offset = _scale_offset(int(ref_end))
            if prev_end is not None and start_offset <= prev_end:
                start_offset = prev_end + 1
            if end_offset < start_offset:
                end_offset = start_offset
            applied_offsets.append((name, start_offset, end_offset))
            prev_end = end_offset

        last = n - 1
        windows: dict[str, dict] = {}
        for (name, start_offset, end_offset), (_, ref_start, ref_end) in zip(
            applied_offsets, PHASE_FRAME_OFFSETS
        ):
            raw_start = t0 + int(start_offset)
            raw_end = t0 + int(end_offset)

            start = max(0, min(last, raw_start))
            end = max(0, min(last, raw_end))
            # 整段越界（如 raw_start/raw_end 同时 < 0 或同时 > last）：贴边单帧
            truncated = raw_end < 0 or raw_start > last
            if end < start:
                start = end
            windows[name] = {
                "start_frame": int(start),
                "end_frame": int(end),
                "start_offset": int(start_offset),
                "end_offset": int(end_offset),
                "reference_start_offset": int(ref_start),
                "reference_end_offset": int(ref_end),
                "frame_count": int(end - start + 1),
                "clamped": bool(raw_start != start or raw_end != end),
                "truncated": bool(truncated),
            }

        windows["_meta"] = {
            "t_impact": int(t0),
            "total_frames": int(n),
            "reference": "t_impact",
            "index_mode": "inclusive_closed_interval",
            "reference_fps": float(reference_fps),
            "effective_fps": float(fps),
            "offset_scale": float(scale),
        }
        return windows

    def build_phase_windows_rich(self) -> dict:
        """调用 error_diagnoser.diagnose_with_temporal_isolation 对本会话轨迹
        执行 T0 锁定 + 相位隔离诊断，返回包含每相位测量指标和 keyframe_index 的
        enriched phase_windows 字典。

        若轨迹不足（< 3 帧）或 diagnose 失败，透明降级为空字典，由调用方决定
        是否 fallback 到 build_phase_windows() 的简化版本。

        返回值字段（每个相位 key 如 "approach_phase"、"support_phase" 等）：
            start_index / end_index / start_ms_rel / end_ms_rel / frame_count
            keyframe_index
            metrics  (仅含 provenance in {"measured","calibrated"} 的指标)
        """
        frames = list(self._trajectory_pose_frames)
        if len(frames) < 3:
            return {}

        # 将 T0 前最近检出的球心作为辅助信号传给 diagnose
        t0 = int(self.t_impact) if self.t_impact is not None else 0
        ball_center = None
        balls = list(self._trajectory_ball_px)
        for idx in range(min(t0, len(balls) - 1), -1, -1):
            if balls[idx] is not None:
                import numpy as _np
                ball_center = _np.array(balls[idx], dtype=float)
                break

        try:
            result = error_diagnoser.diagnose_with_temporal_isolation(
                frames, ball_center=ball_center
            )
        except Exception:  # noqa: BLE001 — 富相位诊断失败不阻断主流程
            return {}

        phase_windows = result.get("phase_windows")
        if not phase_windows:
            return {}
        return phase_windows

    # ------------------------------------------------------------------
    # 主分析循环
    # ------------------------------------------------------------------
    def run(self) -> None:
        """后台工作线程主体：逐帧读取视频源 -> 姿态检测 -> 力学诊断 ->
        骨骼染色 + 面部打码渲染 -> 编码成 Base64 JPEG -> 推入队列。

        这里的每一步算法逻辑，都是直接调用 pose_tracker.py (pt 模块) 里
        已经写好、且已经在桌面版软件里验证过的函数，完全没有重新实现。
        """
        cap = None
        landmarker = None
        yolo_model = None
        is_video_file_mode = self.source == "file"
        video_fps = 30.0

        # 实验防干扰：未锁定基线时强烈警告（跨日摄像头偏移污染 SPSS）
        get_session_metadata_store().warn_if_unlocked(log_fn=safe_print)

        try:
            pt.ensure_model_downloaded()

            if is_video_file_mode:
                if not self.video_path or not os.path.exists(self.video_path):
                    safe_print(f"【shot_analysis_service】错误：未找到视频文件：{self.video_path}")
                    self._push_fn({
                        "type": "error",
                        "message": f"未找到视频文件：{self.video_path}",
                    })
                    return
                cap, video_fps, _reported = pt.open_video_capture_deterministic(
                    self.video_path, is_camera=False
                )
            else:
                cap, video_fps, _reported = pt.open_video_capture_deterministic(
                    "", is_camera=True, camera_index=self.camera_index
                )

            if not cap.isOpened():
                safe_print("【shot_analysis_service】错误：无法打开视频源（本地视频文件损坏，或摄像头被其他程序占用/无摄像头设备）。")
                self._push_fn({
                    "type": "error",
                    "message": "无法打开视频源（本地视频文件损坏，或摄像头被其他程序占用/无摄像头设备）。",
                })
                return

            self._video_fps = float(video_fps) if video_fps and video_fps > 1 else 30.0
            if is_video_file_mode:
                # 【V2.5】录像分析：固定 Δt，角速度与 MediaPipe 时间戳完全由 fps 决定
                self._fixed_frame_dt = 1.0 / float(self._video_fps)
                frame_delay_seconds = self._fixed_frame_dt
            else:
                self._fixed_frame_dt = None
                frame_delay_seconds = 0.0

            # 【V2.5】每次分析任务：销毁旧 PoseLandmarker/YOLO 记忆并重建干净实例
            # 【V3.11】尝试加载 YOLO 以采集足球中心序列（出球初速度闭环）
            task_handles = pt.start_analysis_task(
                reset_yolo=True, yolo_weights="yolov8n.pt"
            )
            landmarker = task_handles["pose_landmarker"]
            yolo_model = task_handles.get("yolo_model")
            if yolo_model is None:
                # 权重路径失败时再试一次显式创建；仍失败则球速降级为 missing
                try:
                    yolo_model = pt.create_fresh_yolo_model("yolov8n.pt")
                except Exception:  # noqa: BLE001
                    yolo_model = None
            if yolo_model is None:
                safe_print(
                    "【shot_analysis_service】提示：YOLO 未就绪，出球初速度将标记为未检出。",
                    flush=True,
                )

            frame_interval_ms = int(round(1000.0 / float(video_fps)))
            frame_timestamp_ms = 0
            self.sync_frame_count = 0
            consecutive_read_fails = 0
            camera_lost_emitted = False
            fps_timestamps: collections.deque = collections.deque(maxlen=60)

            # 【V2.5】同步阻断式 while cap.read()：录像路径严禁跳帧/丢帧
            # 录像文件模式：忽略 stop_event，必须读到 EOF，避免报告竞态截断在 300/414 帧。
            # 摄像头模式：允许 stop_event 提前结束；连续空帧触发 Sprint 5 自愈。
            while cap is not None and cap.isOpened():
                if (not is_video_file_mode) and self.stop_event.is_set():
                    break

                loop_start_time = time.time()

                try:
                    ret, frame = cap.read()
                except Exception as read_exc:  # noqa: BLE001
                    ret, frame = False, None
                    safe_print(f"【shot_analysis_service】读取视频帧异常（计入空帧计数）：{read_exc}")

                if _is_empty_or_failed_frame(ret, frame):
                    if is_video_file_mode:
                        if self._pushed_frame_count == 0:
                            failure_reason = (
                                f"未能从本地视频文件读取到任何一帧画面数据："
                                f"cv2.VideoCapture 显示已成功打开，但第一次 cap.read() 就直接失败。"
                                f" 视频文件路径：{self.video_path}。请确认该文件本身没有损坏，"
                                f"且编码格式受本机 OpenCV 支持（推荐使用 H.264 编码的 .mp4）。"
                            )
                            safe_print(f"【shot_analysis_service】错误：{failure_reason}", flush=True)
                            self._push_fn({"type": "error", "message": failure_reason})
                        else:
                            safe_print(
                                f"【shot_analysis_service】提示：视频源已读取完毕"
                                f"（本次分析同步读入 frame_count={self.sync_frame_count}，"
                                f"成功推送 {self._pushed_frame_count} 帧），分析自然结束。",
                                flush=True,
                            )
                        break

                    # 【Sprint 5】摄像头空帧保护：连续失败达阈值 → camera_lost + release/open 自愈
                    consecutive_read_fails += 1
                    if self._pushed_frame_count == 0 and consecutive_read_fails >= CAMERA_READ_FAIL_LIMIT:
                        # 开场就读不到任何帧：仍按原逻辑报错结束（设备不可用）
                        failure_reason = (
                            "未能从摄像头读取到任何一帧画面数据：cv2.VideoCapture 显示已成功打开，"
                            "但连续多次 cap.read() 均失败。"
                            " 常见原因：摄像头正被其他程序独占使用（如视频会议软件/OBS）、"
                            "摄像头驱动异常，或该摄像头编号在系统里并不是真正可用的设备。"
                            "请先关闭其他可能占用摄像头的程序，或重启电脑后重试。"
                        )
                        safe_print(f"【shot_analysis_service】错误：{failure_reason}", flush=True)
                        self._push_fn({"type": "error", "message": failure_reason})
                        break

                    if consecutive_read_fails >= CAMERA_READ_FAIL_LIMIT:
                        if not camera_lost_emitted:
                            lost_msg = "摄像头信号丢失，尝试重连..."
                            safe_print(
                                f"【shot_analysis_service】警告：连续 {consecutive_read_fails} 帧空读，"
                                f"触发摄像头自愈（release → open）。",
                                flush=True,
                            )
                            self._push_fn(
                                {"type": "camera_lost", "message": lost_msg}
                            )
                            camera_lost_emitted = True
                        try:
                            cap.release()
                        except Exception:  # noqa: BLE001
                            pass
                        time.sleep(CAMERA_REOPEN_SLEEP_SEC)
                        try:
                            cap, video_fps, _reported = pt.open_video_capture_deterministic(
                                "", is_camera=True, camera_index=self.camera_index
                            )
                            frame_interval_ms = int(round(1000.0 / float(video_fps or 30.0)))
                        except Exception as reopen_exc:  # noqa: BLE001
                            safe_print(f"【shot_analysis_service】摄像头自愈打开失败：{reopen_exc}")
                            cap = None
                        consecutive_read_fails = 0
                        if cap is None or not cap.isOpened():
                            self._push_fn({
                                "type": "error",
                                "message": "摄像头自愈失败：无法重新打开设备，请检查连接后重试。",
                            })
                            break
                        safe_print(
                            f"【shot_analysis_service】摄像头自愈成功：已重新打开设备 #{self.camera_index}",
                            flush=True,
                        )
                        continue

                    time.sleep(0.01)
                    continue

                consecutive_read_fails = 0
                if camera_lost_emitted:
                    camera_lost_emitted = False
                    safe_print("【shot_analysis_service】摄像头信号已恢复，继续推流。", flush=True)

                self.sync_frame_count += 1

                if not is_video_file_mode:
                    frame = cv2.flip(frame, 1)

                # 【V3.11】逐帧 YOLO 球心入库（与 sync_frame_count 对齐）
                self._append_ball_detection(frame, yolo_model)

                # 【新增：黑屏问题自动诊断】统计推送帧数 + 检测"疑似全黑帧"。
                # 这一步只做统计判断，绝不修改 frame 本身，不影响后续任何画面处理。
                self._pushed_frame_count += 1
                mean_brightness = float(frame.mean())

                if self._pushed_frame_count == 1:
                    safe_print(
                        f"【shot_analysis_service】[OK] 已成功读取到第 1 帧原始画面"
                        f"（平均亮度 {mean_brightness:.1f}/255，数值越接近 0 代表画面越黑），"
                        f"视频推理管线已正常启动（V2.5 同步顺序帧 / 模型热重置）。"
                    )
                elif self._pushed_frame_count % FRAME_PROGRESS_LOG_INTERVAL == 0:
                    safe_print(
                        f"【shot_analysis_service】进度：已累计推送 {self._pushed_frame_count} 帧画面"
                        f"（本帧平均亮度 {mean_brightness:.1f}/255）。"
                    )

                if mean_brightness < BLACK_FRAME_MEAN_BRIGHTNESS_THRESHOLD:
                    self._consecutive_dark_frames += 1
                else:
                    self._consecutive_dark_frames = 0

                if (
                    not self._dark_frame_warning_sent
                    and self._consecutive_dark_frames >= BLACK_FRAME_CONSECUTIVE_LIMIT
                ):
                    self._dark_frame_warning_sent = True
                    dark_frame_hint = (
                        "检测到摄像头已连续读取到多帧近乎全黑的画面（程序本身运行正常，没有发生异常）。"
                        "在 Windows 系统上，这通常不是代码问题，而是以下几种情况之一："
                        "① Windows 设置 -> 隐私和安全性 -> 相机，未授权「桌面应用」访问摄像头；"
                        "② 摄像头正被其他程序占用（例如视频会议软件、OBS，请先关闭它们再重试）；"
                        "③ 摄像头物理镜头被遮挡，或笔记本电脑的摄像头隐私挡片处于关闭状态。"
                        "请检查以上几点后重新点击「开始分析」。"
                    )
                    safe_print(f"【shot_analysis_service】警告：{dark_frame_hint}")
                    # 用独立的 "notice" 消息类型推送提示：这是"非致命的诊断提醒"，
                    # 不应该像 "error" 一样中断分析会话或关闭连接，只是让前端弹出
                    # 一条醒目的黄色提示，方便老师/学生第一时间知道该去检查什么。
                    self._push_fn({"type": "notice", "message": dark_frame_hint})

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                # MediaPipe VIDEO 时间戳按真实 fps 递增，杜绝写死 33ms 造成的跨次漂移
                frame_timestamp_ms += frame_interval_ms
                results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                angle_value = None
                status_value = None
                angular_velocity_value = None
                stability_index_value = None

                # 【容错防呆】把"姿态诊断 + 角速度/稳定指数计算 + 骨骼渲染"这一整段
                # 逐帧处理逻辑包在独立的 try/except 里：万一某一帧因为异常姿态数据
                # （例如极端角度、瞬时坐标缺失等）导致计算异常，也只会跳过这一帧的
                # 诊断信息渲染，绝不能让整条视频推理循环因此直接崩溃退出——否则
                # 前端会表现为"点击开始分析后画面很快就变成一片黑屏，且没有任何
                # 明确报错"，因为循环提前 return 之后就再也没有新的画面帧推送过来了。
                try:
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks[0]
                        angle, status, color, hip_px, knee_px, ankle_px = pt.compute_right_knee_diagnosis(
                            frame, landmarks
                        )

                        # 【新增】逐帧计算右膝角速度（deg/s）与动平衡稳定指数，
                        # 供前端「实时动力链角速度监控」波形图与稳定指数徽标使用。
                        # 第一帧因为还没有"上一帧角度"可供比较，_compute_angular_velocity
                        # 内部已经做好防呆（约定角速度为 0），这里不会出现除以零的情况。
                        angular_velocity = self._compute_angular_velocity(angle)
                        stability_index_value = self._compute_stability_index()

                        # 轨迹缓存（供结束后抛物线锁帧）
                        self._trajectory_angles.append(float(angle))
                        self._trajectory_omega.append(float(angular_velocity))
                        self._trajectory_ankle_px.append(
                            (float(ankle_px[0]), float(ankle_px[1]))
                        )
                        # Sprint 1：支撑脚 / 摆腿时空热力图所需逐帧关键点
                        try:
                            world_lms = None
                            if getattr(results, "pose_world_landmarks", None):
                                world_lms = results.pose_world_landmarks[0]
                            ts_sec = (
                                float(self.sync_frame_count - 1) * float(self._fixed_frame_dt)
                                if self._fixed_frame_dt
                                else float(self.sync_frame_count - 1) / 30.0
                            )
                            self._trajectory_pose_frames.append(
                                pt.serialize_pose_frame_record(
                                    landmarks,
                                    frame.shape,
                                    timestamp_sec=ts_sec,
                                    world_landmarks=world_lms,
                                )
                            )
                        except Exception:  # noqa: BLE001 - 热力图序列化失败不阻断主诊断链路
                            ts_sec = (
                                float(self.sync_frame_count - 1) * float(self._fixed_frame_dt)
                                if self._fixed_frame_dt
                                else float(self.sync_frame_count - 1) / 30.0
                            )
                            self._trajectory_pose_frames.append(pt.empty_pose_frame_record(ts_sec))

                        # 【绝对拦截器/ Choke Point】关键点提取后立即强制替换为脱敏安全帧，再画骨骼线；
                        # 这是符合《未成年人保护法》与科研伦理审查的物理级脱敏，任何人不得在此行代码之前进行原图转存。
                        #顺序严格保持：先打码，再捕捉击球关键帧，最后叠加染色骨骼线。
                        frame = pt.apply_facial_anonymization(frame, landmarks)

                        # 脱敏帧入缓存（报告阶段按折叠极值帧重取画面）
                        self._cache_blurred_frame(self.sync_frame_count - 1, frame)

                        # 【击球关键帧自动捕捉】在打码之后、骨骼线绘制之前，
                        # 用"角速度绝对值是否为整趟练习目前最大值"来判定是否更新击球关键帧候选，
                        # 角速度越大代表这一帧越接近真实的"发力冲击瞬间"。
                        # 写入 impact_frame 的一定是无脸安全图像（错题本/对比照同理）。
                        impact_score = abs(angular_velocity)
                        if impact_score > self._best_impact_score:
                            # 关键点几何可信才更新峰值候选，避免把塌缩骨架锁进报告图
                            _qa = pt.evaluate_leg_overlay_geometry(hip_px, knee_px, ankle_px)
                            if _qa.get("ok"):
                                self._best_impact_score = impact_score
                                self._capture_impact_candidate(
                                    frame, landmarks, hip_px, knee_px, ankle_px, angle, status
                                )
                            elif self.impact_frame is None:
                                # 尚无任何候选时仍收下，后续报告阶段会再 QA / 邻域搜索
                                self._best_impact_score = impact_score
                                self._capture_impact_candidate(
                                    frame, landmarks, hip_px, knee_px, ankle_px, angle, status
                                )

                        pt.draw_pose_landmarks(frame, results.pose_landmarks)
                        pt.draw_right_knee_overlay(frame, hip_px, knee_px, ankle_px, color, angle, status)

                        angle_value = round(float(angle), 1)
                        status_value = status
                        angular_velocity_value = round(float(angular_velocity), 1)

                        record = {
                            "timestamp": time.time(),
                            "knee_angle": angle_value,
                            "status": status_value,
                            "angular_velocity": angular_velocity_value,
                            "frame_index": self.sync_frame_count - 1,
                        }
                        with self._records_lock:
                            self.records.append(record)
                    else:
                        # 无姿态帧：仍计入同步帧序列长度，用中性值填轨迹以保持索引对齐
                        self._trajectory_angles.append(
                            float(self._trajectory_angles[-1]) if self._trajectory_angles else 150.0
                        )
                        self._trajectory_omega.append(0.0)
                        self._trajectory_ankle_px.append(
                            self._trajectory_ankle_px[-1] if self._trajectory_ankle_px else (0.0, 0.0)
                        )
                        ts_sec = (
                            float(self.sync_frame_count - 1) * float(self._fixed_frame_dt)
                            if self._fixed_frame_dt
                            else float(self.sync_frame_count - 1) / 30.0
                        )
                        self._trajectory_pose_frames.append(pt.empty_pose_frame_record(ts_sec))

                except Exception as diagnosis_exc:  # noqa: BLE001 - 单帧诊断异常绝不能打断整条视频流
                    safe_print(f"【shot_analysis_service】单帧姿态诊断/角速度计算发生异常（已跳过该帧诊断信息，画面仍会继续推送）：{diagnosis_exc}")
                    angle_value = None
                    status_value = None
                    angular_velocity_value = None
                    stability_index_value = None
                    self._trajectory_angles.append(
                        float(self._trajectory_angles[-1]) if self._trajectory_angles else 150.0
                    )
                    self._trajectory_omega.append(0.0)
                    self._trajectory_ankle_px.append(
                        self._trajectory_ankle_px[-1] if self._trajectory_ankle_px else (0.0, 0.0)
                    )
                    ts_sec = (
                        float(self.sync_frame_count - 1) * float(self._fixed_frame_dt)
                        if self._fixed_frame_dt
                        else float(self.sync_frame_count - 1) / 30.0
                    )
                    self._trajectory_pose_frames.append(pt.empty_pose_frame_record(ts_sec))

                # 传输前按最大宽度等比例缩小，减轻 Base64 + WebSocket 的带宽压力
                height, width = frame.shape[:2]
                if width > MAX_TRANSMIT_WIDTH:
                    scale = MAX_TRANSMIT_WIDTH / width
                    frame = cv2.resize(frame, (MAX_TRANSMIT_WIDTH, int(height * scale)))

                ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok:
                    safe_print("【shot_analysis_service】警告：本帧 JPEG 编码失败，已跳过，不影响后续帧的实时推送。")
                    continue
                base64_jpeg = base64.b64encode(buffer).decode("ascii")

                # 【规范格式防呆】显式拼接标准的 data URI 前缀，确保前端 <img src={...}>
                # 拿到的永远是浏览器能够直接识别渲染的合法 "data:image/jpeg;base64,xxxx" 格式。
                image_data_uri = f"data:image/jpeg;base64,{base64_jpeg}"

                # Sprint 5：近 1 秒到达帧数 → 推流 FPS，供 Ghost Monitor 浮层显示
                now_ts = time.time()
                fps_timestamps.append(now_ts)
                while fps_timestamps and (now_ts - fps_timestamps[0]) > 1.0:
                    fps_timestamps.popleft()
                live_fps = int(len(fps_timestamps))

                self._push_fn({
                    "type": "frame",
                    "image": image_data_uri,
                    "angle": angle_value,
                    "status": status_value,
                    "angular_velocity": angular_velocity_value,
                    "stability_index": stability_index_value,
                    "frame_index": self.sync_frame_count - 1,
                    "timestamp": now_ts,
                    "fps": live_fps,
                })

                if is_video_file_mode and frame_delay_seconds > 0:
                    elapsed = time.time() - loop_start_time
                    remaining = frame_delay_seconds - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

            # 录像跑完后：抛物线插值锁定全局唯一 t_impact（内含出球结果结算）
            if is_video_file_mode:
                self._finalize_impact_with_parabolic_lock()
            elif self.t_impact is not None and not self.ball_outcome_meta.get("ok"):
                # 摄像头路径：若已有触球帧则补算出球初速度
                self._finalize_ball_outcome()

        except Exception as exc:  # noqa: BLE001 - 后台线程内的任何异常都不能让服务崩溃
            safe_print(f"【shot_analysis_service】后台推理线程发生异常，本次分析会话将提前结束：{exc}")
            self._push_fn({"type": "error", "message": f"后台推理线程发生异常：{exc}"})

        finally:
            if cap is not None:
                cap.release()
            if landmarker is not None:
                pt.destroy_pose_landmarker(landmarker)
            try:
                pt.destroy_yolo_tracker(yolo_model)
            except Exception:  # noqa: BLE001
                pass
            # 【V2.5】必须在推送 stopped 之前标记 COMPLETED，唤醒挂起的 generate_report
            self._on_completed()
            self._push_fn({
                "type": "stopped",
                "session_id": self.session_id,
                "total_records": len(self.records),
                "frame_count": self.sync_frame_count,
                "t_impact": self.t_impact,
                "task_status": self._status_provider() if self._status_provider else "COMPLETED",
            })

