"""Regression checks for Phase 1 diagnostic observability."""

from diagnostic_trace import sanitize_for_trace, summarize_capture_quality
from llm_agent import build_aigc_safe_payload


class _FakeSession:
    _trajectory_pose_frames = [
        {
            "visibility": {
                "left_shoulder": 0.9,
                "right_shoulder": 0.8,
                "left_hip": 0.7,
                "right_hip": 0.6,
                "left_knee": 0.5,
                "right_knee": 0.4,
                "left_ankle": 0.3,
                "right_ankle": 0.2,
            }
        },
        {"visibility": {"left_shoulder": 0.0, "right_shoulder": 0.0}},
    ]

    def get_capture_diagnostics(self):
        return {
            "declared_fps": 30.0,
            "effective_processing_fps": 24.0,
            "resolution": {"width": 1280, "height": 720},
            "reported_frame_count": 2,
            "read_frame_count": 2,
        }


class _FakeLandmark:
    def __init__(self, visibility):
        self.x = 0.5
        self.y = 0.5
        self.z = 0.0
        self.visibility = visibility


def test_scalar_frame_and_media_flags_are_not_redacted():
    safe = sanitize_for_trace(
        {"frame_count": 42, "frame_index": 7, "has_image": True, "image": "raw"}
    )
    assert safe["frame_count"] == 42
    assert safe["frame_index"] == 7
    assert safe["has_image"] is True
    assert safe["image"]["redacted"] is True


def test_capture_quality_summarizes_visibility_and_resolution():
    summary = summarize_capture_quality(_FakeSession())
    assert summary["resolution"] == {"width": 1280, "height": 720}
    assert summary["pose_visibility"]["valid_pose_frame_count"] == 1
    assert summary["pose_visibility"]["valid_pose_frame_ratio"] == 0.5
    assert summary["pose_visibility"]["critical_low_confidence_frame_count"] == 1
    assert summary["pose_visibility"]["critical_low_confidence_frame_ratio"] == 0.5


def test_pose_serialization_preserves_zero_visibility():
    from pose_tracker import HEATMAP_JOINT_INDICES, serialize_pose_frame_record

    landmark_count = max(HEATMAP_JOINT_INDICES.values()) + 1
    landmarks = [_FakeLandmark(0.0) for _ in range(landmark_count)]
    record = serialize_pose_frame_record(landmarks, (720, 1280, 3))
    assert record["visibility"]
    assert all(value == 0.0 for value in record["visibility"].values())


def test_aigc_payload_preserves_nested_impact_and_deduction_fields():
    payload = build_aigc_safe_payload(
        {
            "score_detail": {
                "TotalScore": 82.5,
                "t_impact": 260,
                "indicators": {
                    "max_folding_angle": {
                        "value": 125.0,
                        "unit": "deg",
                        "status": "RED_DEVIATED",
                        "penalty": 5.0,
                        "provenance": "measured",
                    }
                },
                "deductions": [
                    {
                        "metric_key": "max_folding_angle",
                        "measured_value": 125.0,
                        "penalty": 5.0,
                        "reason": "folding deviation",
                        "error_code": "ERR_SWING_FOLD",
                    }
                ],
            }
        }
    )
    assert payload["t_impact"] == 260
    primary = payload["primary_deduction"]
    assert primary["metric"] == "max_folding_angle"
    assert primary["measured_value"] == 125.0
    assert primary["error_code"] == "ERR_SWING_FOLD"
