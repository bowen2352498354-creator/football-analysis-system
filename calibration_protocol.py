# -*- coding: utf-8 -*-
"""calibration_protocol.py — Phase 3 教室标定协议与误差门槛（可复现夹具）。

用途：
    - 单元测试：用已知真值验证 PCR / 折叠角 / 踝刚度原语误差带；
    - 教室点检：``python -m calibration_protocol`` 输出误差表，不依赖实拍视频。

防幻觉铁律：本模块只做「真值 → 算法输出 → 误差」对照，绝不生成给 AIGC 的假实测。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from biomech_primitives import (
    ANKLE_STIFFNESS_LOCKED,
    ANKLE_STIFFNESS_YIELDING,
    STANDARD_BALL_DIAMETER_CM,
    calculate_3d_joint_angle,
    calculate_ankle_stiffness_variance,
    calculate_support_foot_offset_detailed,
    evaluate_ball_bbox_for_pcr,
)

# --------------------------------------------------------------------------
# 验收门槛（与 Phase 1 设计文档对齐）
# --------------------------------------------------------------------------
PCR_MAE_MAX_CM = 2.0
PCR_LANDMARKS_CM: tuple[float, ...] = (15.0, 17.5, 20.0, 25.0)

FOLD_MAE_MAX_DEG = 5.0
FOLD_TARGETS_DEG: tuple[float, ...] = (70.0, 80.0, 90.0)

DEFAULT_BALL_DIAMETER_PX = 84.0


def synthesize_pcr_case(
    true_offset_cm: float,
    *,
    ball_diameter_px: float = DEFAULT_BALL_DIAMETER_PX,
    ball_center_x: float = 200.0,
    ball_center_y: float = 200.0,
) -> dict[str, Any]:
    """由真值横距反推踝像素与合格球框（正方形，通过 QA）。"""
    diameter = float(ball_diameter_px)
    pcr = float(STANDARD_BALL_DIAMETER_CM) / diameter
    half = diameter / 2.0
    bbox = [
        ball_center_x - half,
        ball_center_y - half,
        ball_center_x + half,
        ball_center_y + half,
    ]
    delta_px = float(true_offset_cm) / pcr
    ankle = (ball_center_x + delta_px, ball_center_y)
    return {
        "true_offset_cm": float(true_offset_cm),
        "ankle_pixel": ankle,
        "ball_pixel_bbox": bbox,
        "pcr": pcr,
    }


def run_pcr_landmark_calibration(
    landmarks_cm: tuple[float, ...] = PCR_LANDMARKS_CM,
    *,
    ball_diameter_px: float = DEFAULT_BALL_DIAMETER_PX,
) -> dict[str, Any]:
    """地标盘式 PCR 标定：返回逐点误差与 MAE。"""
    rows: list[dict[str, Any]] = []
    errors: list[float] = []
    for true_cm in landmarks_cm:
        case = synthesize_pcr_case(true_cm, ball_diameter_px=ball_diameter_px)
        qa = evaluate_ball_bbox_for_pcr(case["ball_pixel_bbox"])
        detail = calculate_support_foot_offset_detailed(
            case["ankle_pixel"], case["ball_pixel_bbox"]
        )
        measured = float(detail.get("offset_cm") or 0.0)
        err = abs(measured - float(true_cm))
        errors.append(err)
        rows.append(
            {
                "true_cm": float(true_cm),
                "measured_cm": round(measured, 4),
                "abs_error_cm": round(err, 4),
                "bbox_ok": bool(qa.get("ok")),
                "detail_ok": bool(detail.get("ok")),
            }
        )
    mae = float(sum(errors) / len(errors)) if errors else float("inf")
    return {
        "metric": "support_foot_pcr",
        "n": len(rows),
        "mae_cm": round(mae, 4),
        "max_abs_error_cm": round(max(errors) if errors else float("inf"), 4),
        "pass": bool(mae <= PCR_MAE_MAX_CM and all(r["detail_ok"] for r in rows)),
        "threshold_mae_cm": PCR_MAE_MAX_CM,
        "rows": rows,
    }


def synthesize_fold_points(fold_depth_deg: float) -> tuple[tuple, tuple, tuple]:
    """构造髋-膝-踝三点，使膝内角 = 180 - fold_depth（屈曲深度）。

    髋在原点上方，膝在原点，踝在 XY 平面按内角旋转。
    """
    interior = 180.0 - float(fold_depth_deg)
    rad = np.radians(interior)
    hip = (0.0, 1.0, 0.0)
    knee = (0.0, 0.0, 0.0)
    ankle = (float(np.sin(rad)), float(np.cos(rad)), 0.0)
    return hip, knee, ankle


def run_fold_angle_calibration(
    targets_deg: tuple[float, ...] = FOLD_TARGETS_DEG,
) -> dict[str, Any]:
    """静态折叠角标定：真值屈曲深度 vs 3D 解算。"""
    rows: list[dict[str, Any]] = []
    errors: list[float] = []
    for true_fold in targets_deg:
        hip, knee, ankle = synthesize_fold_points(true_fold)
        interior = calculate_3d_joint_angle(hip, knee, ankle)
        measured_fold = max(0.0, 180.0 - float(interior))
        err = abs(measured_fold - float(true_fold))
        errors.append(err)
        rows.append(
            {
                "true_fold_deg": float(true_fold),
                "measured_fold_deg": round(measured_fold, 4),
                "interior_deg": round(float(interior), 4),
                "abs_error_deg": round(err, 4),
            }
        )
    mae = float(sum(errors) / len(errors)) if errors else float("inf")
    return {
        "metric": "max_folding_angle",
        "n": len(rows),
        "mae_deg": round(mae, 4),
        "max_abs_error_deg": round(max(errors) if errors else float("inf"), 4),
        "pass": bool(mae <= FOLD_MAE_MAX_DEG),
        "threshold_mae_deg": FOLD_MAE_MAX_DEG,
        "rows": rows,
    }


def run_ankle_stiffness_calibration() -> dict[str, Any]:
    """锁踝 / 松踝分档标定（确定性序列，不依赖视频）。"""
    locked_series = [140.0, 140.1, 140.0, 139.9, 140.05]
    yielding_series = [100.0, 130.0, 90.0, 140.0, 80.0]
    v_lock, s_lock = calculate_ankle_stiffness_variance(locked_series, t_impact_index=2)
    v_yield, s_yield = calculate_ankle_stiffness_variance(yielding_series, t_impact_index=2)
    rows = [
        {
            "label": "locked",
            "variance": round(float(v_lock), 4),
            "status": s_lock,
            "expected_status": ANKLE_STIFFNESS_LOCKED,
            "pass": s_lock == ANKLE_STIFFNESS_LOCKED and float(v_lock) < 2.0,
        },
        {
            "label": "yielding",
            "variance": round(float(v_yield), 4),
            "status": s_yield,
            "expected_status": ANKLE_STIFFNESS_YIELDING,
            "pass": s_yield == ANKLE_STIFFNESS_YIELDING and float(v_yield) > 5.0,
        },
    ]
    return {
        "metric": "ankle_rigidity",
        "n": len(rows),
        "pass": all(r["pass"] for r in rows),
        "accuracy": float(sum(1 for r in rows if r["pass"]) / len(rows)),
        "rows": rows,
    }


def run_full_calibration_suite() -> dict[str, Any]:
    """汇总三项标定，供测试与 CLI。"""
    pcr = run_pcr_landmark_calibration()
    fold = run_fold_angle_calibration()
    ankle = run_ankle_stiffness_calibration()
    return {
        "pass": bool(pcr["pass"] and fold["pass"] and ankle["pass"]),
        "pcr": pcr,
        "fold": fold,
        "ankle": ankle,
    }


def format_calibration_report(suite: Optional[dict[str, Any]] = None) -> str:
    suite = suite or run_full_calibration_suite()
    lines = [
        "=== AI-Football Phase3 Calibration Report ===",
        f"OVERALL: {'PASS' if suite.get('pass') else 'FAIL'}",
        "",
        f"[PCR] MAE={suite['pcr']['mae_cm']}cm "
        f"(max {suite['pcr']['max_abs_error_cm']}, "
        f"gate <={suite['pcr']['threshold_mae_cm']}) "
        f"{'PASS' if suite['pcr']['pass'] else 'FAIL'}",
    ]
    for row in suite["pcr"]["rows"]:
        lines.append(
            f"  true={row['true_cm']:.1f} measured={row['measured_cm']:.4f} "
            f"err={row['abs_error_cm']:.4f}"
        )
    lines.append(
        f"[FOLD] MAE={suite['fold']['mae_deg']}deg "
        f"(max {suite['fold']['max_abs_error_deg']}, "
        f"gate <={suite['fold']['threshold_mae_deg']}) "
        f"{'PASS' if suite['fold']['pass'] else 'FAIL'}"
    )
    for row in suite["fold"]["rows"]:
        lines.append(
            f"  true={row['true_fold_deg']:.1f} measured={row['measured_fold_deg']:.4f} "
            f"err={row['abs_error_deg']:.4f}"
        )
    lines.append(
        f"[ANKLE] accuracy={suite['ankle']['accuracy']:.0%} "
        f"{'PASS' if suite['ankle']['pass'] else 'FAIL'}"
    )
    for row in suite["ankle"]["rows"]:
        lines.append(
            f"  {row['label']}: var={row['variance']} status={row['status']} "
            f"expected={row['expected_status']} {'PASS' if row['pass'] else 'FAIL'}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    report = format_calibration_report()
    print(report)
    raise SystemExit(0 if run_full_calibration_suite()["pass"] else 1)
