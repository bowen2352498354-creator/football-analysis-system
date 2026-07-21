# -*- coding: utf-8 -*-
"""
session_checkpoint.py
断点续传与灾难恢复 —— 事务级 Session 状态快照。

户外体育课场景：断电 / 休眠 / 进程崩溃时，保证「当前学生 + 已完成射门
次数 + attempts_history + 疲劳预警」不丢。

持久化策略：
    - 本地 sqlite3（``session_checkpoint.db``），每次射门切片落盘后
      ``BEGIN IMMEDIATE`` 原子 UPSERT；
    - 同时写一份同结构的 JSON 旁路（``session_checkpoint.json``），
      经临时文件 + ``os.replace`` 原子替换，便于人工排查。
    - 绝不序列化 RollingBuffer 内的 BGR 像素（体积大 + 隐私红线）；
      恢复时仅还原 attempt_count / FSM→IDLE，并清空缓冲准备新帧。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(SCRIPT_DIR, "session_checkpoint.db")
DEFAULT_JSON_PATH = os.path.join(SCRIPT_DIR, "session_checkpoint.json")

STATUS_ACTIVE = "active"
STATUS_COMPLETED = "completed"
STATUS_ABANDONED = "abandoned"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS session_snapshots (
    session_id   TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_status
    ON session_snapshots(status);
"""


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _json_safe(value: Any) -> Any:
    """递归剥离不可 JSON 序列化的值（ndarray / 异常对象等）。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return None
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError, OverflowError):
        return str(value)


@dataclass
class SessionSnapshot:
    """内存核心状态的强类型快照（可 JSON / sqlite 往返）。"""

    session_id: str
    status: str = STATUS_ACTIVE
    class_group: str = ""
    student_id: str = ""
    student_name: str = ""
    experimental_group: str = "GROUP_A"
    attempt_count: int = 0
    attempts_history: list[dict] = field(default_factory=list)
    fatigue_triggered: bool = False
    last_fatigue_warning: Optional[dict] = None
    clip_paths: list[str] = field(default_factory=list)
    auto_capture_meta: dict = field(default_factory=dict)
    data_source: str = "webcam"
    video_path: str = ""
    created_at: str = ""
    updated_at: str = ""

    def display_label(self) -> str:
        """人可读标签：班级-姓名/学号。"""
        who = (self.student_name or self.student_id or "未命名学员").strip()
        klass = (self.class_group or "未设置班级").strip()
        return f"{klass}-{who}"

    def recovery_prompt(self) -> str:
        n = int(self.attempt_count)
        fatigue = "；已触发疲劳预警" if self.fatigue_triggered else ""
        return (
            f"检测到上次异常退出的教学记录（{self.display_label()}，"
            f"已完成 {n} 次射门{fatigue}），是否恢复进度？"
        )

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        return _json_safe(raw)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> Optional["SessionSnapshot"]:
        if not isinstance(data, Mapping):
            return None
        try:
            history = data.get("attempts_history") or []
            if not isinstance(history, list):
                history = []
            clips = data.get("clip_paths") or []
            if not isinstance(clips, list):
                clips = []
            meta = data.get("auto_capture_meta") or {}
            if not isinstance(meta, dict):
                meta = {}
            warning = data.get("last_fatigue_warning")
            if warning is not None and not isinstance(warning, dict):
                warning = None
            return cls(
                session_id=str(data.get("session_id") or ""),
                status=str(data.get("status") or STATUS_ACTIVE),
                class_group=str(data.get("class_group") or ""),
                student_id=str(data.get("student_id") or ""),
                student_name=str(data.get("student_name") or ""),
                experimental_group=str(data.get("experimental_group") or "GROUP_A"),
                attempt_count=int(data.get("attempt_count") or 0),
                attempts_history=[dict(row) for row in history if isinstance(row, dict)],
                fatigue_triggered=bool(data.get("fatigue_triggered")),
                last_fatigue_warning=dict(warning) if warning else None,
                clip_paths=[str(p) for p in clips if p],
                auto_capture_meta=dict(meta),
                data_source=str(data.get("data_source") or "webcam"),
                video_path=str(data.get("video_path") or ""),
                created_at=str(data.get("created_at") or ""),
                updated_at=str(data.get("updated_at") or ""),
            )
        except (TypeError, ValueError):
            return None

    @classmethod
    def new_session(
        cls,
        *,
        class_group: str = "",
        student_id: str = "",
        student_name: str = "",
        experimental_group: str = "GROUP_A",
        data_source: str = "webcam",
        video_path: str = "",
    ) -> "SessionSnapshot":
        stamp = _now_iso()
        return cls(
            session_id=str(uuid.uuid4()),
            status=STATUS_ACTIVE,
            class_group=class_group,
            student_id=student_id,
            student_name=student_name,
            experimental_group=experimental_group,
            data_source=data_source,
            video_path=video_path,
            created_at=stamp,
            updated_at=stamp,
        )


class SessionCheckpointStore:
    """事务级快照仓库：sqlite 主存 + JSON 旁路，线程安全。"""

    def __init__(
        self,
        db_path: str | None = None,
        json_path: str | None = None,
    ) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        self.json_path = json_path or DEFAULT_JSON_PATH
        self._lock = threading.RLock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _ensure_schema(self) -> None:
        with self._lock:
            parent = os.path.dirname(self.db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            conn = self._connect()
            try:
                conn.executescript(_SCHEMA_SQL)
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # 原子写
    # ------------------------------------------------------------------

    def save_snapshot(self, snapshot: SessionSnapshot) -> SessionSnapshot:
        """原子 UPSERT：BEGIN IMMEDIATE → 写 sqlite → 写 JSON → COMMIT。"""
        if not snapshot.session_id:
            snapshot.session_id = str(uuid.uuid4())
        snapshot.updated_at = _now_iso()
        if not snapshot.created_at:
            snapshot.created_at = snapshot.updated_at
        payload = snapshot.to_dict()
        payload_text = json.dumps(payload, ensure_ascii=False)

        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    INSERT INTO session_snapshots
                        (session_id, status, payload_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        status=excluded.status,
                        payload_json=excluded.payload_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        snapshot.session_id,
                        snapshot.status,
                        payload_text,
                        snapshot.created_at,
                        snapshot.updated_at,
                    ),
                )
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:  # noqa: BLE001
                    pass
                raise
            finally:
                conn.close()
            self._atomic_write_json(payload)
        return snapshot

    def _atomic_write_json(self, payload: Mapping[str, Any]) -> None:
        """临时文件 + os.replace，断电时不会留下半截 JSON。"""
        parent = os.path.dirname(self.json_path) or "."
        os.makedirs(parent, exist_ok=True)
        tmp_path = f"{self.json_path}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.json_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # 读 / 结案
    # ------------------------------------------------------------------

    def load_active_session(self) -> Optional[SessionSnapshot]:
        """启动时调用：返回最近一条未正常结束的 active 快照。"""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT payload_json FROM session_snapshots
                    WHERE status = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (STATUS_ACTIVE,),
                ).fetchone()
            finally:
                conn.close()
        if not row:
            # sqlite 空时回退 JSON 旁路（兼容仅旁路存活的极端情况）
            return self._load_json_if_active()
        snap = SessionSnapshot.from_dict(json.loads(row[0]))
        if snap is None or snap.status != STATUS_ACTIVE:
            return None
        if snap.attempt_count <= 0 and not snap.attempts_history:
            # 空会话不算「异常中断」——直接清掉，避免打扰老师
            self.mark_abandoned(snap.session_id)
            return None
        return snap

    def _load_json_if_active(self) -> Optional[SessionSnapshot]:
        if not os.path.isfile(self.json_path):
            return None
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        snap = SessionSnapshot.from_dict(data)
        if snap is None or snap.status != STATUS_ACTIVE:
            return None
        if snap.attempt_count <= 0 and not snap.attempts_history:
            return None
        return snap

    def mark_completed(self, session_id: str) -> None:
        self._set_status(session_id, STATUS_COMPLETED)

    def mark_abandoned(self, session_id: str) -> None:
        self._set_status(session_id, STATUS_ABANDONED)

    def _set_status(self, session_id: str, status: str) -> None:
        if not session_id:
            return
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT payload_json FROM session_snapshots WHERE session_id=?",
                    (session_id,),
                ).fetchone()
                if row:
                    payload = json.loads(row[0])
                    payload["status"] = status
                    payload["updated_at"] = _now_iso()
                    conn.execute(
                        """
                        UPDATE session_snapshots
                        SET status=?, payload_json=?, updated_at=?
                        WHERE session_id=?
                        """,
                        (
                            status,
                            json.dumps(payload, ensure_ascii=False),
                            payload["updated_at"],
                            session_id,
                        ),
                    )
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:  # noqa: BLE001
                    pass
                raise
            finally:
                conn.close()
            # 同步旁路：若当前 JSON 就是该会话，改写 status
            if os.path.isfile(self.json_path):
                try:
                    with open(self.json_path, "r", encoding="utf-8") as f:
                        current = json.load(f)
                    if (
                        isinstance(current, dict)
                        and current.get("session_id") == session_id
                    ):
                        current["status"] = status
                        current["updated_at"] = _now_iso()
                        self._atomic_write_json(current)
                except (OSError, json.JSONDecodeError, TypeError):
                    pass


# 模块级默认仓库（桌面端 / Worker 共用）
default_store = SessionCheckpointStore()
