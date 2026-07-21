# -*- coding: utf-8 -*-
"""断点续传 / 灾难恢复：SessionCheckpoint 原子写与热重启还原。"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt5.QtCore import QCoreApplication

from models.enums import BIOMECH_CORE_METRIC_KEYS
from session_checkpoint import (
    STATUS_ABANDONED,
    STATUS_ACTIVE,
    STATUS_COMPLETED,
    SessionCheckpointStore,
    SessionSnapshot,
)
from session_monitor import FatigueMonitor
from workers.auto_shot_capture import AutoShotCaptureEngine, ShotFsmState


def _ensure_app() -> QCoreApplication:
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


def _flat(*, ankle_rigidity: float = 1.0, support_knee_angle: float = 150.0) -> dict:
    row = {key: 0.0 for key in BIOMECH_CORE_METRIC_KEYS}
    row["ankle_rigidity"] = float(ankle_rigidity)
    row["support_knee_angle"] = float(support_knee_angle)
    return row


def test_snapshot_roundtrip_sqlite_and_json():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "cp.db")
        js = os.path.join(tmp, "cp.json")
        store = SessionCheckpointStore(db_path=db, json_path=js)

        snap = SessionSnapshot.new_session(
            class_group="四年级2班",
            student_id="S01",
            student_name="张三",
            experimental_group="GROUP_A",
        )
        snap.attempt_count = 6
        snap.attempts_history = [
            {"attempt_index": i, "ankle_rigidity": 1.0 + i * 0.1}
            for i in range(1, 7)
        ]
        snap.fatigue_triggered = True
        snap.last_fatigue_warning = {
            "is_fatigue": True,
            "reason": "ANKLE_FATIGUE",
            "message": "脚踝发力出现松软卸力，建议立即叫停轮换！",
        }
        store.save_snapshot(snap)

        assert os.path.isfile(db)
        assert os.path.isfile(js)

        loaded = store.load_active_session()
        assert loaded is not None
        assert loaded.session_id == snap.session_id
        assert loaded.attempt_count == 6
        assert len(loaded.attempts_history) == 6
        assert loaded.fatigue_triggered is True
        assert "四年级2班-张三" in loaded.display_label()
        assert "已完成 6 次射门" in loaded.recovery_prompt()
        assert "疲劳预警" in loaded.recovery_prompt()


def test_empty_active_session_is_auto_abandoned():
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionCheckpointStore(
            db_path=os.path.join(tmp, "cp.db"),
            json_path=os.path.join(tmp, "cp.json"),
        )
        empty = SessionSnapshot.new_session(student_name="空会话")
        store.save_snapshot(empty)
        assert store.load_active_session() is None


def test_mark_completed_hides_from_recovery():
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionCheckpointStore(
            db_path=os.path.join(tmp, "cp.db"),
            json_path=os.path.join(tmp, "cp.json"),
        )
        snap = SessionSnapshot.new_session(
            class_group="四年级2班", student_name="张三"
        )
        snap.attempt_count = 3
        snap.attempts_history = [{"attempt_index": 1}]
        store.save_snapshot(snap)
        store.mark_completed(snap.session_id)
        assert store.load_active_session() is None


def test_mark_abandoned():
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionCheckpointStore(
            db_path=os.path.join(tmp, "cp.db"),
            json_path=os.path.join(tmp, "cp.json"),
        )
        snap = SessionSnapshot.new_session(student_name="李四")
        snap.attempt_count = 2
        snap.attempts_history = [{"attempt_index": 1}, {"attempt_index": 2}]
        store.save_snapshot(snap)
        store.mark_abandoned(snap.session_id)
        assert store.load_active_session() is None


def test_fatigue_monitor_export_restore():
    _ensure_app()
    mon = FatigueMonitor()
    mon.reset_session(student_id="S01")
    for _ in range(3):
        mon.record_attempt(_flat(ankle_rigidity=1.0))
    for _ in range(2):
        mon.record_attempt(_flat(ankle_rigidity=3.0))
    for _ in range(3):
        mon.record_attempt(_flat(ankle_rigidity=6.5))
    warning = mon.check_fatigue_deformation(emit_signal=False)
    assert warning is not None
    assert mon.last_fatigue_warning is not None

    state = mon.export_state()
    assert state["fatigue_triggered"] is True
    assert len(state["attempts_history"]) == 8

    mon2 = FatigueMonitor()
    mon2.restore_state(
        attempts_history=state["attempts_history"],
        student_id="S01",
        last_fatigue_warning=state["last_fatigue_warning"],
    )
    assert len(mon2.attempts_history) == 8
    assert mon2.student_id == "S01"
    assert mon2.last_fatigue_warning["reason"] == "ANKLE_FATIGUE"
    # 恢复后立刻能再次算出同一熔断
    again = mon2.check_fatigue_deformation(emit_signal=False)
    assert again is not None
    assert again["reason"] == "ANKLE_FATIGUE"


def test_rolling_buffer_restore_keeps_attempt_count():
    engine = AutoShotCaptureEngine(output_dir=tempfile.mkdtemp())
    # 模拟已踢 6 脚
    engine._attempt_count = 6
    engine.notify_approach()
    assert engine.state == ShotFsmState.APPROACH

    meta = engine.export_checkpoint_meta()
    assert meta["attempt_count"] == 6
    assert meta["buffer_frames_persisted"] is False

    engine.restore_from_checkpoint(attempt_count=6, meta=meta)
    assert engine.attempt_count == 6
    assert engine.state == ShotFsmState.IDLE
    assert engine.buffer_len == 0


def test_status_constants():
    assert STATUS_ACTIVE == "active"
    assert STATUS_COMPLETED == "completed"
    assert STATUS_ABANDONED == "abandoned"


if __name__ == "__main__":
    test_snapshot_roundtrip_sqlite_and_json()
    test_empty_active_session_is_auto_abandoned()
    test_mark_completed_hides_from_recovery()
    test_mark_abandoned()
    test_fatigue_monitor_export_restore()
    test_rolling_buffer_restore_keeps_attempt_count()
    test_status_constants()
    print("test_session_checkpoint: ALL PASSED")
