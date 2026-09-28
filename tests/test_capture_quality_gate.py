"""Phase 3 regression tests for capture quality and research eligibility."""

from pathlib import Path

from academic_exporter import _is_research_eligible
from capture_quality import evaluate_capture_quality


ROOT = Path(__file__).resolve().parents[1]


def _summary(
    *,
    frames: int = 75,
    pose_ratio: float = 0.96,
    critical_visibility: float = 0.88,
    full_body_ratio: float = 0.86,
    occupancy: float = 0.62,
    edge_clip_ratio: float = 0.03,
    brightness: float = 105.0,
    focus: float = 140.0,
    peak_omega: float = 240.0,
    t0_quality: str = "subframe_cubic_120hz",
) -> dict:
    return {
        "declared_fps": 30.0,
        "effective_processing_fps": 25.0,
        "read_frame_count": frames,
        "frame_timing": {"interval_cv": 0.08},
        "brightness": {"mean": brightness},
        "focus": {"laplacian_variance_mean": focus},
        "pose_visibility": {
            "valid_pose_frame_ratio": pose_ratio,
            "critical_visibility_mean": critical_visibility,
        },
        "body_geometry": {
            "full_body_frame_ratio": full_body_ratio,
            "body_height_ratio_median": occupancy,
            "edge_clipped_frame_ratio": edge_clip_ratio,
        },
        "impact_signal": {
            "peak_angular_velocity_abs": peak_omega,
            "t0_quality": t0_quality,
            "impact_is_interior": True,
        },
        "ball_visibility": {
            "model_available": False,
            "detected_frame_ratio": 0.0,
        },
    }


def _evaluate(attempt: dict) -> dict:
    return evaluate_capture_quality(
        capture=_summary(),
        attempt=attempt,
        source="webcam",
        precision_analysis_used=True,
        replay_available=True,
    )


def test_complete_attempt_is_grade_a_and_research_eligible():
    result = _evaluate(_summary())
    assert result["grade"] == "A"
    assert result["formalReportAllowed"] is True
    assert result["researchEligible"] is True


def test_borderline_attempt_is_reference_only_grade_b():
    result = _evaluate(
        _summary(
            pose_ratio=0.75,
            critical_visibility=0.62,
            full_body_ratio=0.55,
            peak_omega=65.0,
            t0_quality="fallback_midframe",
        )
    )
    assert result["grade"] == "B"
    assert result["referenceFeedbackAllowed"] is True
    assert result["formalReportAllowed"] is False
    assert result["researchEligible"] is False


def test_seated_or_lower_body_missing_attempt_is_grade_c():
    result = _evaluate(
        _summary(
            critical_visibility=0.25,
            full_body_ratio=0.0,
            occupancy=0.28,
            peak_omega=5.0,
        )
    )
    assert result["grade"] == "C"
    assert result["referenceFeedbackAllowed"] is False
    assert any(issue["code"] == "critical_joint_visibility" for issue in result["issues"])


def test_quality_gate_runs_before_scorer_and_llm():
    source = (ROOT / "api_server.py").read_text(encoding="utf-8")
    gate = source.index('"capture_quality_gate"')
    rejection = source.index('quality_gate.get("grade") == "C"', gate)
    scoring = source.index("session.build_scoring_payloads()", rejection)
    llm = source.index("llm_agent.generate_session_report(", scoring)
    assert gate < rejection < scoring < llm


def test_research_export_excludes_explicit_b_and_c_but_keeps_legacy():
    assert _is_research_eligible({"qualityGrade": "A", "researchEligible": True})
    assert not _is_research_eligible({"qualityGrade": "B", "researchEligible": False})
    assert not _is_research_eligible({"qualityGate": {"grade": "C"}})
    assert _is_research_eligible({"score": 88.0})
