"""Deterministic capture-quality gate for formal biomechanics reports.

The gate is intentionally independent from scoring and AIGC.  It answers one
question only: is the captured attempt reliable enough to support a formal
research report?
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


QUALITY_GATE_VERSION = "capture_quality_v1"

# Centralized thresholds so later field calibration changes policy in one place.
QUALITY_THRESHOLDS = {
    "min_frames_a": 45,
    "min_frames_b": 20,
    "pose_ratio_a": 0.85,
    "pose_ratio_b": 0.60,
    "critical_visibility_a": 0.70,
    "critical_visibility_b": 0.50,
    "full_body_ratio_a": 0.70,
    "full_body_ratio_b": 0.35,
    "fps_a": 18.0,
    "fps_b": 10.0,
    "fps_cv_a": 0.20,
    "fps_cv_b": 0.40,
    "brightness_a": (45.0, 210.0),
    "brightness_b": (25.0, 235.0),
    "focus_a": 60.0,
    "focus_b": 25.0,
    "occupancy_a": (0.30, 0.92),
    "occupancy_b": (0.18, 0.98),
    "edge_clip_a": 0.10,
    "edge_clip_b": 0.35,
    "impact_peak_a": 80.0,
    "impact_peak_b": 30.0,
}


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _nested(data: Mapping[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _range_state(value: Optional[float], good: tuple[float, float], usable: tuple[float, float]) -> str:
    if value is None:
        return "unknown"
    if good[0] <= value <= good[1]:
        return "good"
    if usable[0] <= value <= usable[1]:
        return "warning"
    return "critical"


def evaluate_capture_quality(
    *,
    capture: Mapping[str, Any],
    attempt: Mapping[str, Any],
    source: str,
    precision_analysis_used: bool,
    replay_available: bool,
) -> dict[str, Any]:
    """Return an explainable A/B/C gate result for one analyzed attempt.

    ``capture`` describes the continuous camera/file input. ``attempt``
    describes the exact clip used for scoring. The latter owns pose and body
    quality; the former contributes acquisition stability evidence.
    """
    t = QUALITY_THRESHOLDS
    issues: list[dict[str, Any]] = []
    penalties = 0.0
    hard_fail = False

    def issue(code: str, label: str, severity: str, actual: Any, requirement: str, penalty: float) -> None:
        nonlocal penalties, hard_fail
        issues.append(
            {
                "code": code,
                "label": label,
                "severity": severity,
                "actual": actual,
                "requirement": requirement,
            }
        )
        penalties += penalty
        if severity == "critical":
            hard_fail = True

    frames = int(_number(attempt.get("read_frame_count")) or 0)
    pose_ratio = _number(_nested(attempt, "pose_visibility", "valid_pose_frame_ratio"))
    critical_visibility = _number(_nested(attempt, "pose_visibility", "critical_visibility_mean"))
    full_body_ratio = _number(_nested(attempt, "body_geometry", "full_body_frame_ratio"))
    occupancy = _number(_nested(attempt, "body_geometry", "body_height_ratio_median"))
    edge_clip_ratio = _number(_nested(attempt, "body_geometry", "edge_clipped_frame_ratio"))
    fps = _number(
        capture.get("declared_fps")
        if source == "file"
        else capture.get("effective_processing_fps")
    )
    fps_cv = (
        None
        if source == "file"
        else _number(_nested(capture, "frame_timing", "interval_cv"))
    )
    brightness = _number(_nested(attempt, "brightness", "mean"))
    focus = _number(_nested(attempt, "focus", "laplacian_variance_mean"))
    peak_omega = _number(_nested(attempt, "impact_signal", "peak_angular_velocity_abs"))
    impact_quality = str(_nested(attempt, "impact_signal", "t0_quality") or "unknown")
    impact_interior = bool(_nested(attempt, "impact_signal", "impact_is_interior"))
    ball_ratio = _number(_nested(attempt, "ball_visibility", "detected_frame_ratio"))
    ball_model_available = bool(
        _nested(attempt, "ball_visibility", "model_available")
    )

    if frames < t["min_frames_b"]:
        issue("too_few_frames", "有效分析帧不足", "critical", frames, f">={t['min_frames_b']} 帧", 45)
    elif frames < t["min_frames_a"]:
        issue("short_attempt", "动作片段偏短", "warning", frames, f">={t['min_frames_a']} 帧", 8)

    if pose_ratio is None or pose_ratio < t["pose_ratio_b"]:
        issue("pose_continuity", "人体姿态连续率不足", "critical", pose_ratio, f">={t['pose_ratio_b']:.0%}", 35)
    elif pose_ratio < t["pose_ratio_a"]:
        issue("pose_continuity", "人体姿态存在断帧", "warning", pose_ratio, f">={t['pose_ratio_a']:.0%}", 8)

    if critical_visibility is None or critical_visibility < t["critical_visibility_b"]:
        issue("critical_joint_visibility", "髋、膝、踝关键点置信度不足", "critical", critical_visibility, f">={t['critical_visibility_b']:.2f}", 40)
    elif critical_visibility < t["critical_visibility_a"]:
        issue("critical_joint_visibility", "下肢关键点置信度偏低", "warning", critical_visibility, f">={t['critical_visibility_a']:.2f}", 10)

    if full_body_ratio is None or full_body_ratio < t["full_body_ratio_b"]:
        issue("full_body_visibility", "人体或双侧下肢未完整入镜", "critical", full_body_ratio, f">={t['full_body_ratio_b']:.0%}", 40)
    elif full_body_ratio < t["full_body_ratio_a"]:
        issue("full_body_visibility", "部分帧下肢不完整", "warning", full_body_ratio, f">={t['full_body_ratio_a']:.0%}", 10)

    occupancy_state = _range_state(occupancy, t["occupancy_a"], t["occupancy_b"])
    if occupancy_state == "critical":
        issue("body_scale", "人体在画面中过大或过小", "critical", occupancy, f"{t['occupancy_b'][0]:.0%}-{t['occupancy_b'][1]:.0%}", 28)
    elif occupancy_state == "warning":
        issue("body_scale", "人体画面占比不理想", "warning", occupancy, f"{t['occupancy_a'][0]:.0%}-{t['occupancy_a'][1]:.0%}", 6)

    if edge_clip_ratio is not None and edge_clip_ratio > t["edge_clip_b"]:
        issue("edge_clipping", "关键关节频繁贴近画面边缘", "critical", edge_clip_ratio, f"<={t['edge_clip_b']:.0%}", 28)
    elif edge_clip_ratio is not None and edge_clip_ratio > t["edge_clip_a"]:
        issue("edge_clipping", "部分帧存在边缘裁切风险", "warning", edge_clip_ratio, f"<={t['edge_clip_a']:.0%}", 6)

    brightness_state = _range_state(brightness, t["brightness_a"], t["brightness_b"])
    if brightness_state == "critical":
        issue("exposure", "画面严重过暗或过曝", "critical", brightness, f"{t['brightness_b'][0]:.0f}-{t['brightness_b'][1]:.0f}", 30)
    elif brightness_state == "warning":
        issue("exposure", "画面亮度接近可用边界", "warning", brightness, f"{t['brightness_a'][0]:.0f}-{t['brightness_a'][1]:.0f}", 6)

    if focus is not None and focus < t["focus_b"]:
        issue("blur", "画面明显模糊", "critical", round(focus, 2), f">={t['focus_b']:.0f}", 30)
    elif focus is not None and focus < t["focus_a"]:
        issue("blur", "画面清晰度偏低", "warning", round(focus, 2), f">={t['focus_a']:.0f}", 6)

    if fps is not None and fps < t["fps_b"]:
        issue("low_fps", "采集处理帧率过低", "critical", fps, f">={t['fps_b']:.0f} fps", 25)
    elif fps is not None and fps < t["fps_a"]:
        issue("low_fps", "采集处理帧率偏低", "warning", fps, f">={t['fps_a']:.0f} fps", 6)

    if fps_cv is not None and fps_cv > t["fps_cv_b"]:
        issue("fps_instability", "帧间隔波动过大", "critical", fps_cv, f"<={t['fps_cv_b']:.2f}", 24)
    elif fps_cv is not None and fps_cv > t["fps_cv_a"]:
        issue("fps_instability", "帧间隔存在波动", "warning", fps_cv, f"<={t['fps_cv_a']:.2f}", 5)

    impact_good = impact_quality in {"subframe_cubic_120hz", "subframe_interp"}
    if peak_omega is None or peak_omega < t["impact_peak_b"] or not impact_interior:
        issue("impact_confidence", "触球时刻缺少可信运动峰值", "critical", peak_omega, f">={t['impact_peak_b']:.0f} deg/s 且不在边界", 40)
    elif peak_omega < t["impact_peak_a"] or not impact_good:
        issue("impact_confidence", "触球时刻仅达到参考级可信度", "warning", peak_omega, "插值锁帧且峰值充分", 10)

    if source == "webcam" and (not precision_analysis_used or not replay_available):
        issue("precision_replay", "摄像头精分析片段或回放不可用", "critical", False, "精分析与回放均可用", 40)

    # Ball detection is useful corroborating evidence, but lack of a YOLO model
    # must not automatically invalidate otherwise sound pose biomechanics.
    if ball_model_available and ball_ratio is not None and ball_ratio <= 0:
        issue("ball_not_detected", "未检测到足球，触球点采用运动学代理", "warning", ball_ratio, ">0", 4)

    score = max(0.0, min(100.0, 100.0 - penalties))
    if hard_fail or score < 55.0:
        grade = "C"
    elif issues or score < 85.0:
        grade = "B"
    else:
        grade = "A"

    summary = {
        "A": "采集质量达标，可生成正式报告并进入科研数据。",
        "B": "采集质量基本可用，仅生成低置信度参考反馈，不进入科研统计。",
        "C": "采集质量不达标，已停止正式评分与 AI 处方，请调整机位后重新采集。",
    }[grade]
    recommendations = []
    recommendation_by_code = {
        "pose_continuity": "保持全身持续位于画面内，避免快速离开取景范围。",
        "critical_joint_visibility": "确保髋、双膝、双踝和双脚无遮挡。",
        "full_body_visibility": "后退或调整机位，使头部到双脚完整入镜。",
        "body_scale": "调整人与摄像头距离，使人体高度约占画面三至九成。",
        "edge_clipping": "将运动员置于画面中央，并为助跑和随摆留出空间。",
        "exposure": "调整环境照明，避免逆光、过曝或暗部丢失。",
        "blur": "固定摄像头并改善照明，减少运动模糊。",
        "low_fps": "关闭占用摄像头或算力的程序，并使用稳定帧率采集。",
        "fps_instability": "降低后台负载或固定视频分辨率后重新采集。",
        "impact_confidence": "完成一次包含助跑、摆腿、触球和随摆的完整射门。",
        "precision_replay": "确认自动截取成功后再结束本次分析。",
        "ball_not_detected": "让足球保持清晰可见，避免被脚或器材长时间遮挡。",
        "too_few_frames": "延长动作采集时间，保留完整动作前后文。",
        "short_attempt": "保留触球前后更完整的动作片段。",
    }
    for item in issues:
        recommendation = recommendation_by_code.get(item["code"])
        if recommendation and recommendation not in recommendations:
            recommendations.append(recommendation)

    return {
        "schemaVersion": QUALITY_GATE_VERSION,
        "grade": grade,
        "score": round(score, 1),
        "summary": summary,
        "formalReportAllowed": grade == "A",
        "referenceFeedbackAllowed": grade in {"A", "B"},
        "researchEligible": grade == "A",
        "issues": issues,
        "recommendations": recommendations[:4],
        "metrics": {
            "analysisFrameCount": frames,
            "poseFrameRatio": pose_ratio,
            "criticalJointVisibility": critical_visibility,
            "fullBodyFrameRatio": full_body_ratio,
            "bodyHeightRatioMedian": occupancy,
            "edgeClippedFrameRatio": edge_clip_ratio,
            "effectiveFps": fps,
            "frameIntervalCv": fps_cv,
            "brightnessMean": brightness,
            "focusLaplacianVariance": focus,
            "impactPeakAngularVelocity": peak_omega,
            "impactLockQuality": impact_quality,
            "ballDetectedFrameRatio": ball_ratio,
        },
    }
