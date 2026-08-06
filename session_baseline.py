# -*- coding: utf-8 -*-
"""
session_baseline.py
实验防干扰机制 —— SessionCheckpoint 基线元数据水印。

户外/跨日测试时摄像头高度漂移、标定矩阵失效会导致 SPSS 污染。
本模块在「锁定实验环境」后为每一次拍摄结果打上基线水印：
    baseline_session_id / class_id / camera_height_cm / calibrator_status /
    is_baseline_trusted

与 ``session_checkpoint.py``（断点续传）互不干扰：后者管崩溃恢复，
本模块管实验环境血统。
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class SessionCheckpoint:
    """单次实验环境基线快照（水印载体）。"""

    session_id: str
    class_id: str = ""
    camera_height_cm: Optional[float] = None
    calibrator_status: str = "unknown"
    locked_at: str = ""
    school: str = ""
    class_group: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # 扁平化时保留嵌套 extra，供调试；导出侧只取标量字段
        return payload

    def to_stamp(self, *, trusted: bool = True) -> dict[str, Any]:
        """写入 score_detail / 归档 JSON / SPSS 的标准水印字段。"""
        height = self.camera_height_cm
        try:
            height_out = float(height) if height is not None else None
        except (TypeError, ValueError):
            height_out = None
        return {
            "baseline_session_id": str(self.session_id),
            "baseline_class_id": str(self.class_id or ""),
            "class_id": str(self.class_id or ""),
            "camera_height_cm": height_out,
            "calibrator_status": str(self.calibrator_status or "unknown"),
            "baseline_locked_at": str(self.locked_at or ""),
            "is_baseline_trusted": bool(trusted),
            "session_checkpoint": {
                "session_id": str(self.session_id),
                "class_id": str(self.class_id or ""),
                "camera_height_cm": height_out,
                "calibrator_status": str(self.calibrator_status or "unknown"),
                "locked_at": str(self.locked_at or ""),
                "school": str(self.school or ""),
                "class_group": str(self.class_group or ""),
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> Optional["SessionCheckpoint"]:
        if not isinstance(data, Mapping):
            return None
        sid = str(data.get("session_id") or "").strip()
        if not sid:
            return None
        height = data.get("camera_height_cm")
        try:
            height_f = float(height) if height is not None and height != "" else None
        except (TypeError, ValueError):
            height_f = None
        extra = data.get("extra") if isinstance(data.get("extra"), dict) else {}
        return cls(
            session_id=sid,
            class_id=str(data.get("class_id") or ""),
            camera_height_cm=height_f,
            calibrator_status=str(data.get("calibrator_status") or "unknown"),
            locked_at=str(data.get("locked_at") or ""),
            school=str(data.get("school") or ""),
            class_group=str(data.get("class_group") or data.get("classGroup") or ""),
            extra=dict(extra),
        )


class SessionMetadataStore:
    """进程内全局基线状态管理器（线程安全）。"""

    BASELINE_WARNING = (
        "[Baseline Warning]: Analysis running without locked session baseline!"
    )

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._checkpoint: Optional[SessionCheckpoint] = None
        self._session_locked: bool = False

    @property
    def session_locked(self) -> bool:
        with self._lock:
            return bool(self._session_locked and self._checkpoint is not None)

    def get_checkpoint(self) -> Optional[SessionCheckpoint]:
        with self._lock:
            return self._checkpoint

    def lock_baseline(
        self,
        *,
        class_id: str = "",
        camera_height_cm: Optional[float] = None,
        calibrator_status: str = "unknown",
        school: str = "",
        class_group: str = "",
        session_id: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> SessionCheckpoint:
        """锁定当前实验环境，生成全局唯一 session_id。"""
        sid = str(session_id or "").strip() or str(uuid.uuid4())
        try:
            height = float(camera_height_cm) if camera_height_cm is not None else None
        except (TypeError, ValueError):
            height = None
        status = str(calibrator_status or "unknown").strip() or "unknown"
        checkpoint = SessionCheckpoint(
            session_id=sid,
            class_id=str(class_id or "").strip(),
            camera_height_cm=height,
            calibrator_status=status,
            locked_at=_now_iso(),
            school=str(school or "").strip(),
            class_group=str(class_group or "").strip(),
            extra=dict(extra or {}),
        )
        with self._lock:
            self._checkpoint = checkpoint
            self._session_locked = True
        return checkpoint

    def unlock_baseline(self) -> None:
        """显式解除锁定（换班 / 跨日重标定前调用）。"""
        with self._lock:
            self._checkpoint = None
            self._session_locked = False

    def status_dict(self) -> dict[str, Any]:
        with self._lock:
            cp = self._checkpoint
            locked = bool(self._session_locked and cp is not None)
            body: dict[str, Any] = {
                "session_locked": locked,
                "is_baseline_trusted": locked,
            }
            if cp is not None:
                body.update(cp.to_dict())
            else:
                body.update(
                    {
                        "session_id": None,
                        "class_id": "",
                        "camera_height_cm": None,
                        "calibrator_status": "unlocked",
                        "locked_at": "",
                    }
                )
            return body

    def warn_if_unlocked(self, *, log_fn=None) -> bool:
        """若未锁定则打印强烈警告；返回 True 表示已锁定。"""
        if self.session_locked:
            return True
        printer = log_fn or print
        try:
            printer(self.BASELINE_WARNING)
        except Exception:  # noqa: BLE001
            print(self.BASELINE_WARNING)
        return False

    def stamp_payload(
        self, target: Optional[dict], *, analysis_session_id: Optional[str] = None
    ) -> dict:
        """将基线水印强制写入目标 dict（score_detail / 归档记录）。

        未锁定时仍写入 ``is_baseline_trusted=False``，便于后期研究者过滤。
        """
        out = dict(target or {})
        trusted = self.session_locked
        cp = self.get_checkpoint()
        if cp is not None:
            stamp = cp.to_stamp(trusted=trusted)
        else:
            stamp = {
                "baseline_session_id": None,
                "baseline_class_id": "",
                "class_id": "",
                "camera_height_cm": None,
                "calibrator_status": "unlocked",
                "baseline_locked_at": "",
                "is_baseline_trusted": False,
                "session_checkpoint": None,
            }
        out.update(stamp)
        # 保留分析会话 id（WebSocket 单趟），不覆盖基线 session_id
        if analysis_session_id:
            out["analysis_session_id"] = str(analysis_session_id)
        return out


# 进程级单例：api_server / shot_analysis_service / academic_exporter 共用
SESSION_METADATA_STORE = SessionMetadataStore()


def get_session_metadata_store() -> SessionMetadataStore:
    return SESSION_METADATA_STORE


def stamp_baseline_watermark(
    payload: Optional[dict],
    *,
    analysis_session_id: Optional[str] = None,
    store: Optional[SessionMetadataStore] = None,
) -> dict:
    """模块级便捷入口：给任意结果 dict 打基线水印。"""
    active = store or SESSION_METADATA_STORE
    return active.stamp_payload(payload, analysis_session_id=analysis_session_id)
