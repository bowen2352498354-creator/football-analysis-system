# -*- coding: utf-8 -*-
"""大小腿夹角标注：几何 QA + 折叠极值帧选点 + 降级绘制。"""

from __future__ import annotations

import numpy as np

import pose_tracker as pt


def test_geometry_qa_rejects_collapsed_shank():
    qa = pt.evaluate_leg_overlay_geometry(
        (100, 100), (100, 200), (102, 205), knee_visibility=0.9, ankle_visibility=0.9
    )
    assert qa["ok"] is False
    assert "shank" in qa["reason"]


def test_geometry_qa_accepts_plausible_leg():
    qa = pt.evaluate_leg_overlay_geometry(
        (100, 80), (110, 200), (140, 340), knee_visibility=0.9, ankle_visibility=0.85
    )
    assert qa["ok"] is True


def test_geometry_qa_rejects_low_visibility():
    qa = pt.evaluate_leg_overlay_geometry(
        (100, 80), (110, 200), (140, 340), knee_visibility=0.2, ankle_visibility=0.9
    )
    assert qa["ok"] is False
    assert qa["reason"] == "knee_low_visibility"


def test_build_metrics_from_pose_marks_collapsed_not_ok():
    rec = {
        "right_hip": [100.0, 80.0, 0.0],
        "right_knee": [110.0, 120.0, 0.0],
        "right_ankle": [112.0, 125.0, 0.0],
        "left_hip": [80.0, 80.0, 0.0],
        "visibility": {"right_knee": 0.9, "right_ankle": 0.9},
    }
    metrics = pt.build_annotation_metrics_from_pose_record(rec, side="right")
    assert metrics is not None
    assert metrics["overlay_ok"] is False


def test_build_metrics_plausible_fold_pose():
    rec = {
        "right_hip": [200.0, 100.0, 0.0],
        "right_knee": [260.0, 220.0, 0.0],
        "right_ankle": [180.0, 280.0, 0.0],
        "left_hip": [160.0, 100.0, 0.0],
        "visibility": {"right_knee": 0.95, "right_ankle": 0.9},
        "world": {
            "right_hip": [0.0, 0.0, 0.0],
            "right_knee": [0.0, 0.4, 0.0],
            "right_ankle": [-0.25, 0.55, 0.0],
        },
    }
    metrics = pt.build_annotation_metrics_from_pose_record(rec, side="right")
    assert metrics is not None
    assert metrics["overlay_ok"] is True
    assert metrics["angle"] > 0
    assert metrics["swing_side"] == "right"


def test_resolve_prefers_fold_extreme_when_measured():
    detail = {
        "swing_leg": "right",
        "indicators": {
            "max_folding_angle": {
                "value": 95.0,
                "provenance": "measured",
                "extreme_frame_index": 42,
                "method": "roi_3d_knee_min_right",
                "swing_leg": "right",
            }
        },
    }
    idx, side, _label = pt.resolve_leg_annotation_target(detail, t_impact=80)
    assert idx == 42
    assert side == "right"


def test_rebuild_leg_annotation_forces_t_impact_not_fold():
    """射门瞬间分析帧必须定格 t_impact，不得被折叠极值帧劫持。"""
    from shot_analysis_service import ShotAnalysisPipeline

    pipe = ShotAnalysisPipeline(
        session_id="ann-force-t0",
        source="file",
        video_path=None,
        push_fn=lambda _m: None,
    )
    fold_rec = {
        "right_hip": [100.0, 80.0, 0.0],
        "right_knee": [110.0, 140.0, 0.0],
        "right_ankle": [90.0, 220.0, 0.0],
        "visibility": {"right_hip": 0.99, "right_knee": 0.99, "right_ankle": 0.99},
    }
    impact_rec = {
        "right_hip": [100.0, 80.0, 0.0],
        "right_knee": [120.0, 150.0, 0.0],
        "right_ankle": [140.0, 230.0, 0.0],
        "visibility": {"right_hip": 0.99, "right_knee": 0.99, "right_ankle": 0.99},
    }
    pipe._trajectory_pose_frames = [fold_rec, impact_rec]
    pipe._cache_blurred_frame(0, np.zeros((240, 320, 3), dtype=np.uint8))
    pipe._cache_blurred_frame(1, np.zeros((240, 320, 3), dtype=np.uint8) + 10)
    detail = {
        "swing_leg": "right",
        "indicators": {
            "max_folding_angle": {
                "value": 95.0,
                "provenance": "measured",
                "extreme_frame_index": 0,
                "method": "roi_3d_knee_min_right",
                "swing_leg": "right",
            }
        },
    }
    frame, metrics = pipe.rebuild_leg_annotation(detail, t_impact=1, force_impact_frame=True)
    assert frame is not None and metrics is not None
    assert int(metrics.get("annotation_frame_index", -1)) == 1


def test_draw_skips_arc_when_overlay_not_ok():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    metrics = {
        "hip_px": (100, 60),
        "knee_px": (110, 90),
        "ankle_px": (112, 95),
        "mid_hip_px": (100, 60),
        "angle": 52.3,
        "status": "Red",
        "overlay_ok": False,
        "label": "大小腿夹角",
        "swing_side": "right",
    }
    out = pt.draw_biomechanics_annotation(frame, metrics)
    assert out.shape == frame.shape
    assert int(out[20, 30].sum()) > 0


def test_draw_ok_places_vertex_near_knee():
    frame = np.zeros((360, 480, 3), dtype=np.uint8)
    knee = (240, 200)
    metrics = {
        "hip_px": (220, 80),
        "knee_px": knee,
        "ankle_px": (200, 320),
        "mid_hip_px": (210, 80),
        "angle": 95.0,
        "status": "Red",
        "overlay_ok": True,
        "fold_depth_deg": 85.0,
        "label": "大小腿夹角",
        "swing_side": "right",
    }
    out = pt.draw_biomechanics_annotation(frame, metrics)
    patch = out[knee[1] - 3 : knee[1] + 4, knee[0] - 3 : knee[0] + 4]
    assert int(patch.max()) > 200


if __name__ == "__main__":
    test_geometry_qa_rejects_collapsed_shank()
    test_geometry_qa_accepts_plausible_leg()
    test_geometry_qa_rejects_low_visibility()
    test_build_metrics_from_pose_marks_collapsed_not_ok()
    test_build_metrics_plausible_fold_pose()
    test_resolve_prefers_fold_extreme_when_measured()
    test_draw_skips_arc_when_overlay_not_ok()
    test_draw_ok_places_vertex_near_knee()
    print("ALL LEG ANNOTATION TESTS PASSED")
