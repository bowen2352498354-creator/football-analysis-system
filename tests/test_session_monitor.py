# -*- coding: utf-8 -*-
"""FatigueMonitor：脚踝卸力 / 支撑腿僵直熔断金标准。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt5.QtCore import QCoreApplication

from models.enums import BIOMECH_CORE_METRIC_KEYS
from session_monitor import FatigueMonitor, flatten_eight_metrics


def _ensure_app() -> QCoreApplication:
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


def _flat(
    *,
    ankle_rigidity: float = 1.0,
    support_knee_angle: float = 150.0,
) -> dict:
    row = {key: 0.0 for key in BIOMECH_CORE_METRIC_KEYS}
    row["ankle_rigidity"] = float(ankle_rigidity)
    row["support_knee_angle"] = float(support_knee_angle)
    return row


def test_flatten_from_indicators():
    detail = {
        "indicators": {
            "ankle_rigidity": {"value": 2.5, "variance": 2.5},
            "support_knee_angle": {"value": 155.0},
            "distance_cm": {"value": 18.0},
        }
    }
    flat = flatten_eight_metrics(detail)
    assert flat["ankle_rigidity"] == 2.5
    assert flat["support_knee_angle"] == 155.0
    assert flat["distance_cm"] == 18.0
    assert flat["hip_torsion_angle"] is None


def test_no_monitor_before_eight_attempts():
    _ensure_app()
    mon = FatigueMonitor()
    for _ in range(7):
        mon.record_attempt(_flat(ankle_rigidity=6.0, support_knee_angle=170.0))
    assert mon.check_fatigue_deformation(emit_signal=False) is None


def test_ankle_fatigue_fuse():
    """前 3 脚硬踝基线，后段脚踝方差崩坏 → ANKLE_FATIGUE。"""
    _ensure_app()
    mon = FatigueMonitor()
    # baseline：硬踝
    for _ in range(3):
        mon.record_attempt(_flat(ankle_rigidity=1.0, support_knee_angle=150.0))
    # 中间过渡
    for _ in range(2):
        mon.record_attempt(_flat(ankle_rigidity=3.0, support_knee_angle=152.0))
    # recent：严重松软且相对基线抬升 ≥ 2.0
    for _ in range(3):
        mon.record_attempt(_flat(ankle_rigidity=6.5, support_knee_angle=153.0))

    assert len(mon.attempts_history) == 8
    warning = mon.check_fatigue_deformation(emit_signal=False)
    assert warning is not None
    assert warning["is_fatigue"] is True
    assert warning["reason"] == "ANKLE_FATIGUE"
    assert "松软卸力" in warning["message"]


def test_knee_stiffness_fuse():
    """前 3 脚可屈曲，后段支撑膝僵直 → KNEE_STIFFNESS。"""
    _ensure_app()
    mon = FatigueMonitor()
    for _ in range(3):
        mon.record_attempt(_flat(ankle_rigidity=1.2, support_knee_angle=150.0))
    for _ in range(2):
        mon.record_attempt(_flat(ankle_rigidity=1.5, support_knee_angle=158.0))
    for _ in range(3):
        mon.record_attempt(_flat(ankle_rigidity=1.8, support_knee_angle=172.0))

    warning = mon.check_fatigue_deformation(emit_signal=False)
    assert warning is not None
    assert warning["reason"] == "KNEE_STIFFNESS"
    assert warning["recent_mean"] > 165.0
    assert warning["delta"] >= 15.0


def test_reset_session_clears_history():
    _ensure_app()
    mon = FatigueMonitor()
    mon.record_attempt(_flat())
    mon.last_fatigue_warning = {"is_fatigue": True, "reason": "ANKLE_FATIGUE"}
    mon.reset_session(student_id="S001")
    assert mon.attempts_history == []
    assert mon.student_id == "S001"
    assert mon.last_fatigue_warning is None


if __name__ == "__main__":
    test_flatten_from_indicators()
    test_no_monitor_before_eight_attempts()
    test_ankle_fatigue_fuse()
    test_knee_stiffness_fuse()
    test_reset_session_clears_history()
    print("test_session_monitor: ALL PASSED")
