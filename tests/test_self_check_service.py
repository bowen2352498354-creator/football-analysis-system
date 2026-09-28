from __future__ import annotations

from sqlalchemy import create_engine

import self_check_service


def _engine():
    return create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
    )


def test_choose_dimension_prefers_priority_metric_then_lowest_radar():
    prioritized = {
        "priorityTarget": {"metricKey": "ankle_rigidity"},
        "quantified5dScores": {
            "approach_rhythm": 4,
            "ankle_rigidity": 18,
        },
    }
    assert self_check_service.choose_dimension(prioritized) == "ankle_rigidity"

    radar_only = {
        "quantified5dScores": {
            "approach_rhythm": 14,
            "support_stability": 7,
            "backswing_folding": 10,
            "ankle_rigidity": 16,
            "whipping_velocity": 12,
        }
    }
    assert self_check_service.choose_dimension(radar_only) == "support_stability"


def test_task_is_stable_and_checkins_persist():
    bind = _engine()
    record = {
        "id": "record-001",
        "quantified5dScores": {
            "approach_rhythm": 15,
            "support_stability": 11,
            "backswing_folding": 9,
            "ankle_rigidity": 17,
            "whipping_velocity": 14,
        },
    }

    first = self_check_service.ensure_self_check_task(record, bind=bind)
    second = self_check_service.ensure_self_check_task(record, bind=bind)

    assert first["taskId"] == second["taskId"]
    assert first["dimensionKey"] == "backswing_folding"
    assert len(first["checkins"]) == 3

    updated = self_check_service.update_self_check_task(
        record,
        slot_no=2,
        checked=True,
        coach_verified=True,
        bind=bind,
    )
    assert updated["coachVerified"] is True
    assert updated["checkins"][1]["checked"] is True
    assert updated["checkins"][1]["checkedAt"]

    persisted = self_check_service.ensure_self_check_task(record, bind=bind)
    assert persisted["coachVerified"] is True
    assert persisted["checkins"][1]["checked"] is True


def test_invalid_checkin_slot_is_rejected():
    bind = _engine()
    record = {"id": "record-002", "biomechanicalErrors": ["支撑脚位置偏离"]}
    try:
        self_check_service.update_self_check_task(
            record,
            slot_no=4,
            checked=True,
            bind=bind,
        )
    except ValueError as exc:
        assert "slotNo" in str(exc)
    else:
        raise AssertionError("out-of-range self-check slot should fail")

