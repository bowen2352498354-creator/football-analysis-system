# -*- coding: utf-8 -*-
"""
workers/auto_shot_capture.py
【V3.1 Sprint 2】无感全自动切片与死时间剥离
【V3.1 Sprint 4】RollingBuffer 防 OOM：满员挤出的旧帧立即失去引用，交 GC 回收。

核心能力：
  1. RollingBuffer：``deque(maxlen=150)`` 只保留最近约 5 秒画面；
  2. 射门 FSM：IDLE → APPROACH → IMPACT_LOCKED → COOLDOWN → IDLE；
  3. 触球锁帧成功后，异步 ``cv2.VideoWriter`` 落盘
     ``[t_impact-60, t_impact+30]``（共 90 帧核心画面），剔除捡球死时间；
  4. 写盘在独立 daemon 线程完成，绝不阻塞 GUI / 推理主环。
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Callable, Deque, Iterator, List, Optional, Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 时间机器 / FSM 常量（与 pose_tracker 对齐，可被外部覆盖）
# ---------------------------------------------------------------------------

ROLLING_BUFFER_MAXLEN: int = 150  # ~5s @30fps
PRE_IMPACT_FRAMES: int = 60
POST_IMPACT_FRAMES: int = 30  # 与前窗合计 90 帧核心切片
COOLDOWN_SEC: float = 3.5  # 强制冷却 3–5s 中位值，防抖
DEFAULT_CLIP_FPS: float = 30.0

_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_AUTO_CLIP_DIR = os.path.join(_SCRIPT_DIR, "auto_capture_clips")


class ShotFsmState(str, Enum):
    """射门动作有限状态机四态。"""

    IDLE = "IDLE"  # 待机 / 无人 / 走动捡球
    APPROACH = "APPROACH"  # 检测到助跑发力
    IMPACT_LOCKED = "IMPACT_LOCKED"  # 触球瞬间已锁定，等待后窗凑齐后落盘
    COOLDOWN = "COOLDOWN"  # 保存冷却中，忽略乱动


@dataclass
class BufferedFrame:
    """滚动缓冲中的单帧快照。

    非 frozen：挤出/清空时可把 ``bgr`` 置 ``None``，切断 ndarray 引用以利 GC。
    """

    frame_index: int
    bgr: Optional[np.ndarray]
    timestamp: float

    def release(self) -> None:
        """主动切断图像矩阵引用，便于立刻被 GC 回收。"""
        self.bgr = None


LogFn = Callable[[str], None]
StateChangeFn = Callable[[ShotFsmState, ShotFsmState], None]
ClipSavedFn = Callable[[dict], None]


class RollingBuffer:
    """防 OOM 的有界帧环：满员时挤出最旧帧并 ``release()``，不持有无限历史。"""

    def __init__(self, maxlen: int = ROLLING_BUFFER_MAXLEN) -> None:
        cap = max(1, int(maxlen))
        self._buf: Deque[BufferedFrame] = deque(maxlen=cap)
        self._maxlen = cap

    def __len__(self) -> int:
        return len(self._buf)

    def __bool__(self) -> bool:
        return bool(self._buf)

    def __iter__(self) -> Iterator[BufferedFrame]:
        return iter(self._buf)

    def __getitem__(self, index: int) -> BufferedFrame:
        return self._buf[index]

    @property
    def maxlen(self) -> int:
        return self._maxlen

    @property
    def latest(self) -> Optional[BufferedFrame]:
        return self._buf[-1] if self._buf else None

    def append(self, item: BufferedFrame) -> None:
        """入队；若已满，被 ``deque`` 挤出的旧项立即 ``release()``。"""
        if self._buf.maxlen is not None and len(self._buf) >= self._buf.maxlen:
            evicted = self._buf[0]
        else:
            evicted = None
        self._buf.append(item)
        if evicted is not None and evicted is not item:
            # deque 已丢弃最左引用；再显式断 ndarray，双保险防 OOM
            evicted.release()

    def clear(self) -> None:
        """会话复位：逐帧 release 后再清空，确保旧矩阵立刻可被 GC。"""
        for item in self._buf:
            item.release()
        self._buf.clear()


class AutoShotCaptureEngine:
    """无感自动切片引擎：滚动缓冲 + FSM + 异步落盘。

    线程模型：
      - ``push_frame`` / ``notify_*`` 由推理线程调用（同步、轻量）；
      - ``cv2.VideoWriter`` 写盘在 daemon 线程执行，完成后回调 ``on_clip_saved``。
    """

    def __init__(
        self,
        *,
        output_dir: Optional[str] = None,
        fps: float = DEFAULT_CLIP_FPS,
        session_date: Optional[date] = None,
        rolling_maxlen: int = ROLLING_BUFFER_MAXLEN,
        pre_frames: int = PRE_IMPACT_FRAMES,
        post_frames: int = POST_IMPACT_FRAMES,
        cooldown_sec: float = COOLDOWN_SEC,
        on_log: Optional[LogFn] = None,
        on_state_change: Optional[StateChangeFn] = None,
        on_clip_saved: Optional[ClipSavedFn] = None,
    ) -> None:
        self.output_dir = output_dir or DEFAULT_AUTO_CLIP_DIR
        self.fps = float(fps) if fps and fps > 1.0 else DEFAULT_CLIP_FPS
        self.session_date = session_date or date.today()
        self.pre_frames = int(pre_frames)
        self.post_frames = int(post_frames)
        self.cooldown_sec = float(cooldown_sec)

        self._buffer = RollingBuffer(maxlen=int(rolling_maxlen))
        self._state = ShotFsmState.IDLE
        self._state_lock = threading.Lock()

        self._locked_t_impact: Optional[int] = None
        self._cooldown_until: float = 0.0
        self._attempt_count: int = 0
        self._save_in_flight: bool = False

        self._on_log = on_log
        self._on_state_change = on_state_change
        self._on_clip_saved = on_clip_saved

        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------

    @property
    def state(self) -> ShotFsmState:
        with self._state_lock:
            return self._state

    @property
    def attempt_count(self) -> int:
        return int(self._attempt_count)

    @property
    def buffer_len(self) -> int:
        return len(self._buffer)

    def accepts_impact_triggers(self) -> bool:
        """COOLDOWN / IMPACT_LOCKED 期间忽略新的触球触发，防止连踢误判。"""
        with self._state_lock:
            return self._state in (ShotFsmState.IDLE, ShotFsmState.APPROACH)

    # ------------------------------------------------------------------
    # 帧入队（时间机器）
    # ------------------------------------------------------------------

    def push_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        *,
        timestamp: Optional[float] = None,
    ) -> ShotFsmState:
        """无脑存入滚动缓冲；在 IMPACT_LOCKED 时检查后窗是否凑齐并触发落盘。

        【隐私红线】入参 ``frame_bgr`` 必须已是
        ``apply_facial_anonymization`` 覆盖后的安全帧。
        Attempt 切片由此缓冲异步写出，绝不可接收未脱敏原图。
        """
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            return self.state

        ts = float(timestamp) if timestamp is not None else time.time()
        # 必须 copy：推理环会原地改写同一块内存
        self._buffer.append(
            BufferedFrame(
                frame_index=int(frame_index),
                bgr=frame_bgr.copy(),
                timestamp=ts,
            )
        )

        with self._state_lock:
            state = self._state
            locked_t = self._locked_t_impact

        if state == ShotFsmState.COOLDOWN:
            if time.time() >= self._cooldown_until:
                self._transition(ShotFsmState.IDLE)
                self._locked_t_impact = None
                self._log("COOLDOWN 结束 → IDLE，继续无感侦测。")
            return self.state

        if state == ShotFsmState.IMPACT_LOCKED and locked_t is not None:
            if int(frame_index) >= int(locked_t) + self.post_frames:
                self._dispatch_clip_save(int(locked_t))

        return self.state

    def finalize(self) -> None:
        """会话结束 / 录像 EOF：若仍停在 IMPACT_LOCKED，用缓冲内可得帧立即落盘。"""
        with self._state_lock:
            state = self._state
            locked_t = self._locked_t_impact
        if state == ShotFsmState.IMPACT_LOCKED and locked_t is not None:
            self._log(
                f"finalize：EOF 不等后窗，立即切片 t_impact=#{locked_t} "
                f"（缓冲内可得帧）。"
            )
            self._dispatch_clip_save(int(locked_t))
        elif state == ShotFsmState.APPROACH:
            self.notify_discard("session_end")

    def notify_approach(self, *, omega: Optional[float] = None) -> bool:
        """检测到助跑发力（角速度峰）→ IDLE → APPROACH。"""
        if not self.accepts_impact_triggers():
            return False
        with self._state_lock:
            if self._state != ShotFsmState.IDLE:
                return False
        tip = f"（ω={omega:.1f}）" if omega is not None else ""
        self._transition(ShotFsmState.APPROACH)
        self._log(f"FSM APPROACH：检测到助跑发力{tip}")
        return True

    def notify_impact_locked(self, t_impact: int) -> bool:
        """原有锁帧算法成功锁定 ``t_impact`` → IMPACT_LOCKED。"""
        if not self.accepts_impact_triggers():
            return False

        t = int(t_impact)
        self._locked_t_impact = t
        self._transition(ShotFsmState.IMPACT_LOCKED)
        self._log(
            f"FSM IMPACT_LOCKED：t_impact=#{t}，"
            f"等待后窗 +{self.post_frames} 帧后异步落盘"
            f"（前窗 −{self.pre_frames}）。"
        )

        # 若缓冲里已经攒够后窗（录像 EOF / 延迟锁帧），立即落盘
        latest_bf = self._buffer.latest
        if latest_bf is not None:
            if int(latest_bf.frame_index) >= t + self.post_frames:
                self._dispatch_clip_save(t)
        return True

    def notify_discard(self, reason: str = "") -> None:
        """锁帧无解：从 APPROACH 退回 IDLE，不落盘。"""
        with self._state_lock:
            if self._state not in (ShotFsmState.APPROACH, ShotFsmState.IMPACT_LOCKED):
                return
        self._locked_t_impact = None
        self._transition(ShotFsmState.IDLE)
        hint = f"（{reason}）" if reason else ""
        self._log(f"FSM 丢弃本轮{hint} → IDLE")

    def reset(self) -> None:
        """会话结束 / 强制复位（保留 attempt_count，供日志统计）。"""
        with self._state_lock:
            self._state = ShotFsmState.IDLE
        self._locked_t_impact = None
        self._cooldown_until = 0.0
        self._save_in_flight = False
        self._buffer.clear()

    def export_checkpoint_meta(self) -> dict[str, Any]:
        """轻量元数据（不含 BGR 像素）——供灾难恢复快照。"""
        return {
            "attempt_count": int(self._attempt_count),
            "fsm_state": ShotFsmState.IDLE.value,
            "buffer_maxlen": int(self._buffer.maxlen),
            "pre_frames": int(self.pre_frames),
            "post_frames": int(self.post_frames),
            "cooldown_sec": float(self.cooldown_sec),
            "fps": float(self.fps),
            # 像素帧不落盘：恢复后缓冲为空，仅保留射门计数连续性
            "buffer_frames_persisted": False,
        }

    def restore_from_checkpoint(
        self,
        *,
        attempt_count: int = 0,
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        """热重启：还原射门计数，清空 RollingBuffer，FSM 回到 IDLE。

        中断前的像素帧无法在断电后可靠/合规地恢复；此处保证
        ``attempt_count`` 与切片命名序号无缝衔接，缓冲重新开始攒帧。
        """
        del meta  # 预留扩展；当前仅需 attempt_count
        with self._state_lock:
            self._state = ShotFsmState.IDLE
        self._locked_t_impact = None
        self._cooldown_until = 0.0
        self._save_in_flight = False
        self._attempt_count = max(0, int(attempt_count))
        self._buffer.clear()
        self._log(
            f"灾难恢复：RollingBuffer 已清空并回到 IDLE，"
            f"attempt_count={self._attempt_count}（像素帧不续传，仅保留计数）。"
        )

    # ------------------------------------------------------------------
    # 切片 + 异步写盘
    # ------------------------------------------------------------------

    def _dispatch_clip_save(self, t_impact: int) -> None:
        """从 RollingBuffer 截取核心 90 帧，交给后台线程写 MP4，并进入 COOLDOWN。"""
        if self._save_in_flight:
            return
        self._save_in_flight = True

        frames = self._slice_core_window(t_impact)
        if not frames:
            self._save_in_flight = False
            self._locked_t_impact = None
            self._transition(ShotFsmState.IDLE)
            self._log("落盘取消：滚动缓冲中无可用帧（可能会话刚启动）。")
            return

        self._attempt_count += 1
        attempt_n = self._attempt_count
        filename = self._build_clip_filename(attempt_n)
        out_path = os.path.join(self.output_dir, filename)

        # 深拷贝帧列表，避免写盘期间滚动缓冲回收/改写同一块内存
        frames_copy = [f.copy() for f in frames]
        # 切断对缓冲内矩阵的临时引用，只保留 frames_copy
        del frames
        fps = self.fps

        self._cooldown_until = time.time() + self.cooldown_sec
        self._transition(ShotFsmState.COOLDOWN)
        self._locked_t_impact = None
        self._log(
            f"异步落盘启动：attempt={attempt_n}，帧数={len(frames_copy)}，"
            f"文件={filename}；进入 COOLDOWN {self.cooldown_sec:.1f}s。"
        )

        threading.Thread(
            target=self._write_clip_thread,
            args=(frames_copy, out_path, fps, attempt_n, t_impact),
            daemon=True,
            name=f"AutoShotWriter-{attempt_n}",
        ).start()

    def _slice_core_window(self, t_impact: int) -> List[np.ndarray]:
        """截取 ``[t_impact-pre, t_impact+post]``；边界不足时取缓冲内可得帧。"""
        lo = int(t_impact) - self.pre_frames
        hi = int(t_impact) + self.post_frames
        selected: List[np.ndarray] = []
        for bf in self._buffer:
            if bf.bgr is None:
                continue
            if lo <= int(bf.frame_index) <= hi:
                selected.append(bf.bgr)
        return selected

    def _build_clip_filename(self, attempt_n: int) -> str:
        date_token = self.session_date.strftime("%Y%m%d")
        return f"session_{date_token}_attempt_{attempt_n}.mp4"

    def _write_clip_thread(
        self,
        frames: Sequence[np.ndarray],
        out_path: str,
        fps: float,
        attempt_n: int,
        t_impact: int,
    ) -> None:
        ok = False
        error: Optional[str] = None
        frame_count = len(frames)
        try:
            ok = self._write_mp4(frames, out_path, fps)
            if not ok:
                error = "VideoWriter 打开或写入失败"
        except Exception as exc:  # noqa: BLE001 - 写盘失败绝不能拖死推理环
            error = str(exc)
            ok = False
        finally:
            self._save_in_flight = False
            # 写盘结束：立刻丢掉写盘线程持有的帧副本，交 GC
            if isinstance(frames, list):
                frames.clear()
            del frames

        if ok:
            self._log(f"切片已静默落盘：{out_path}（剥离捡球死时间，仅保留核心动作）。")
        else:
            self._log(f"切片落盘失败：{error or 'unknown'} → {out_path}")

        if self._on_clip_saved is not None:
            try:
                self._on_clip_saved(
                    {
                        "ok": ok,
                        "path": out_path if ok else None,
                        "attempt_number": attempt_n,
                        "attempt_count": self._attempt_count,
                        "t_impact": t_impact,
                        "frame_count": frame_count,
                        "error": error,
                        "fsm_state": self.state.value,
                    }
                )
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _write_mp4(frames: Sequence[np.ndarray], out_path: str, fps: float) -> bool:
        """将缓冲帧写入 Attempt 切片。

        调用方保证 ``frames`` 均已通过管道最前端的面部脱敏拦截器；
        本方法只负责编码，不再接触原始带脸画面。
        """
        if not frames:
            return False
        h, w = frames[0].shape[:2]
        if h <= 0 or w <= 0:
            return False
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, float(fps), (int(w), int(h)))
        if not writer.isOpened():
            return False
        try:
            for frame in frames:
                if frame is None or frame.size == 0:
                    continue
                if frame.shape[0] != h or frame.shape[1] != w:
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)
        finally:
            writer.release()
        return os.path.isfile(out_path) and os.path.getsize(out_path) > 0

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _transition(self, new_state: ShotFsmState) -> None:
        with self._state_lock:
            old = self._state
            if old == new_state:
                return
            self._state = new_state
        if self._on_state_change is not None:
            try:
                self._on_state_change(old, new_state)
            except Exception:  # noqa: BLE001
                pass

    def _log(self, message: str) -> None:
        if self._on_log is not None:
            try:
                self._on_log(f"【AutoCapture】{message}")
            except Exception:  # noqa: BLE001
                pass

    def snapshot(self) -> dict[str, Any]:
        """供 GUI / WebSocket 广播的轻量状态快照。"""
        return {
            "fsm_state": self.state.value,
            "attempt_count": self._attempt_count,
            "buffer_len": len(self._buffer),
            "locked_t_impact": self._locked_t_impact,
            "cooldown_remaining_sec": max(0.0, self._cooldown_until - time.time()),
        }
