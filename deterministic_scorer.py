# -*- coding: utf-8 -*-
"""
deterministic_scorer.py
V2.5/V3.1 确定性科研级纯数学评分引擎（LLM 零参与、零随机源）。

自 ``error_diagnoser.py`` 拆出，职责边界：
    - 八大量纲扣分 + 五维雷达量化（``DeterministicScorer``）；
    - Kinematic Boundary Guard 人体生理学硬拦截；
    - Action ROI 切片与 ROI 内极值/角速度解算；
    - 经验阈值中央化读取（唯一权威源仍为 ``empirical_thresholds.py``）。

指标封装 / 消毒层在 ``indicator_builder.py``；ERR 码诊断、几何 helper 与
时空热力图仍在 ``error_diagnoser.py``。后者按需被本模块以惰性导入方式引用，
以打破 ``error_diagnoser -> deterministic_scorer -> error_diagnoser`` 环。

同一输入必须位级可复现：禁止引入随机数、时间戳或 LLM 调用。
"""

from __future__ import annotations

# 【V2.5】与视觉管线一致：在可能触发 CUDA 的依赖前写入 CUBLAS 配置
import os
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from typing import Any, Optional

import numpy as np

from biomech_primitives import (
    ANKLE_DEFLECTION_HALF_WINDOW_FRAMES,
    ANKLE_IMPACT_HALF_WINDOW_MS,
    ANKLE_STIFFNESS_LOCKED,
    ANKLE_STIFFNESS_SLIGHT_DEFORMATION,
    ANKLE_STIFFNESS_YIELDING,
    AVERAGE_CHILD_SHOULDER_WIDTH_CM,
    DEFAULT_VIDEO_FPS,
    FOLD_ROI_MIN_VALID_FRAMES,
    LANDMARK_CONFIDENCE_MIN,
    TRUNK_LEAN_GREEN_HIGH_DEG,
    TRUNK_LEAN_GREEN_LOW_DEG,
    TRUNK_LEAN_RED_FORWARD_DEG,
    TRUNK_LEAN_YELLOW_BACK_DEG,
    ankle_half_window_frames,
    ankle_window_dorsiflex_drop_deg,
    calculate_2d_angle,
    calculate_2d_angle_or_none,
    calculate_3d_joint_angle,
    calculate_3d_joint_angle_or_none,
    calculate_ankle_deflection,
    calculate_ankle_stiffness_variance,
    calculate_support_offset_by_shoulder_ratio,
    calculate_support_ratio,
    calculate_trunk_lean,
    classify_trunk_lean_status,
    gap_fill_scalar_series,
    infer_swing_leg_side,
    is_valid_joint_point,
    log_2d_vs_sagittal_shift,
    slice_ankle_impact_window_bounds,
    swing_leg_joint_keys,
)

# --------------------------------------------------------------------------
# FSM Timeout Guard 常量（射门 FSM 实现见 workers/auto_shot_capture.py）
# --------------------------------------------------------------------------
# 中间态（如 APPROACH / 准备击球）超过该帧数未完成转换 → 强制回 IDLE
FSM_APPROACH_TIMEOUT_FRAMES = 60  # ≈2s @30fps
FSM_IMPACT_LOCKED_TIMEOUT_FRAMES = 120
from indicator_builder import (
    ACTION_ROI_HALF_FRAMES,
    PROVENANCE_CALIBRATED,
    PROVENANCE_DEFAULT,
    PROVENANCE_ESTIMATED,
    PROVENANCE_MEASURED,
    PROVENANCE_MISSING,
    is_aigc_measurable_provenance,
    pack_focus_indicator,
    sanitize_eight_dimension_indicators,
)


# --------------------------------------------------------------------------
# error_diagnoser 侧几何 / 热力图 helper 的惰性代理
# --------------------------------------------------------------------------
# 【为何惰性】error_diagnoser 顶层会 import 本模块做符号回导出；若本模块在顶层
# 反向 import error_diagnoser 即成环。这些 helper 只在评分执行期被调用，
# 届时 error_diagnoser 已完成加载，惰性取用完全安全且零性能负担。
def calculate_angle(a, b, c, *, is_knee_extension: bool = False) -> float:
    """以 b 为顶点的关节夹角（度）；与 error_diagnoser.calculate_angle 同源。

    髋/膝屈伸（``is_knee_extension=True``）走 XY-2D（Z 坍缩）。
    """
    if bool(is_knee_extension):
        return float(calculate_2d_angle(a, b, c))
    return calculate_3d_joint_angle(a, b, c, is_knee_extension=False)


def _landmark_visibility(frame_record: dict, joint: str) -> float:
    from error_diagnoser import _landmark_visibility as _impl

    return _impl(frame_record, joint)


def _hip_relative_torsion_deg(frame_record: dict) -> float:
    from error_diagnoser import _hip_relative_torsion_deg as _impl

    return _impl(frame_record)


def _support_toe_angle_deg(frame_record: dict, ball_center=None) -> float:
    from error_diagnoser import _support_toe_angle_deg as _impl

    return _impl(frame_record, ball_center)


def build_spatial_heatmap_payload(*args, **kwargs):
    from error_diagnoser import build_spatial_heatmap_payload as _impl

    return _impl(*args, **kwargs)


def build_joint_highlights(*args, **kwargs):
    """惰性代理：具身隐喻关节红绿灯高亮（实现见 error_diagnoser）。"""
    from error_diagnoser import build_joint_highlights as _impl

    return _impl(*args, **kwargs)


# --------------------------------------------------------------------------
# V2.5 确定性科研级纯数学评分引擎（LLM 零参与）
# --------------------------------------------------------------------------
STATUS_GREEN = "GREEN_OPTIMAL"
STATUS_YELLOW = "YELLOW_APPROACHING"
STATUS_RED = "RED_DEVIATED"

# 各项最高扣分（V3.8：支撑距 RED 8–10；黄灯线性约 3–5）
_MAX_PENALTY_DISTANCE_CM = 10.0
_MAX_PENALTY_TOE_ANGLE = 6.0
_MAX_PENALTY_FOLDING = 8.0
_MAX_PENALTY_WHIPPING = 6.0
_MAX_PENALTY_IMPACT_KNEE = 8.0
_MAX_PENALTY_ANKLE = 8.0  # 脚踝锁紧度：方差 > 5.0 直接扣该项满分
_MAX_PENALTY_SUPPORT_KNEE = 6.0
_MAX_PENALTY_HIP_TORSION = 6.0
_MAX_PENALTY_TRUNK_LEAN = 8.0  # 后仰/过度折腰严重扣分；直立黄灯约 0.55×

# 【V3.9】脚踝最大形变落差角阈值（°）——取代旧方差 σ²
ANKLE_DEFLECTION_GREEN = 10.0  # < 10° 锁踝极佳
ANKLE_DEFLECTION_YELLOW_HIGH = 20.0  # 10–20° 轻微卸力；>20° 严重松弛
# 兼容旧名（数值已切换为 deflection 阈值）
ANKLE_VARIANCE_GREEN = ANKLE_DEFLECTION_GREEN
ANKLE_VARIANCE_YELLOW_HIGH = ANKLE_DEFLECTION_YELLOW_HIGH

# --------------------------------------------------------------------------
# Kinematic Boundary Guard（特斯拉 FSD 风格人体生理学硬拦截）
# --------------------------------------------------------------------------
# 触球瞬间摆动腿必为伸展态；点乘锐角假象阈值 + 绝对生理钳位区间
KINEMATIC_KNEE_SUPPLEMENT_THRESHOLD_DEG = 100.0
KINEMATIC_KNEE_PHYSIO_MIN_DEG = 120.0
KINEMATIC_KNEE_PHYSIO_MAX_DEG = 175.0
# YOLO 球框漂移导致的离谱横距：>55cm 才触发降级钳制（V3.5：35cm 已是合法严重区）
KINEMATIC_DISTANCE_RUNAWAY_CM = 55.0
KINEMATIC_DISTANCE_CLAMP_CM = 40.0
# 【V3.9】踝形变落差有物理自然上限，移除旧 variance→12.0 防暴走钳制
KINEMATIC_ANKLE_VARIANCE_RUNAWAY = float("inf")  # 兼容导入；不再触发
KINEMATIC_ANKLE_VARIANCE_MAX_ALLOWED = ANKLE_DEFLECTION_YELLOW_HIGH


# --------------------------------------------------------------------------
# 经验阈值中央化读取（empirical_thresholds.py 为唯一权威源）
# --------------------------------------------------------------------------
# 【设计约束】评分函数内不得再出现绿/黄带字面量。所有区间一律经本函数取值，
# 便于凭实际教学数据迭代（改 empirical_thresholds.json 即可，无需动评分代码）。
# 依据/校准日期注释随值写在 empirical_thresholds.DEFAULT_THRESHOLDS 各区块。
# 模块缺失或 JSON 损坏时回退到与现网同源的硬编码字面量，绝不抛崩。
_SCORING_BAND_FALLBACK: dict[str, tuple[float, ...]] = {
    # 【遗留】PCR cm 带；站位评分已全面切到 support_ratio
    "distance_cm": (15.0, 20.0, 10.0, 35.0, 17.5),
    # 肩宽比黄金标准：0.4–0.7 绿；0.25–0.4 / 0.7–0.9 黄；外红
    "support_ratio": (0.40, 0.70, 0.25, 0.90, 0.55),
    "toe_angle": (15.0, 25.0),
    # folding depth（XY-2D）：70–100 绿；55–120 黄带；外红
    "max_folding_angle": (70.0, 100.0, 55.0, 120.0, 85.0),
    "whipping_velocity": (450.0, 320.0, 0.55),
    # 触球膝角：仅 >165° 进入直腿黄/红带
    "impact_knee_angle": (135.0, 165.0, 120.0, 172.0, 150.0),
    "support_knee_angle": (135.0, 170.0, 120.0, 175.0, 155.0),
    "hip_torsion_angle": (15.0, 40.0, 5.0, 55.0, 25.0),
}
_RADAR_CONFIG_FALLBACK: dict[str, float] = {
    "ankle_locked_score": 20.0,
    "ankle_slight_score": 15.0,
    "ankle_yielding_score": 5.0,
    "whipping_full_score_deg_s": 450.0,
    "approach_rhythm_floor": 16.0,
    "approach_rhythm_ceiling": 20.0,
    "approach_penalty_slope": 0.05,
}
_SHANK_ONLY_FOLD_MAX_FALLBACK = 140.0


def _scoring_bands() -> dict[str, tuple[float, ...]]:
    """取全部评分带阈值；任一环节异常即整体回退现网默认（可复现，零随机）。"""
    bands = dict(_SCORING_BAND_FALLBACK)
    try:
        from empirical_thresholds import (
            get_folding_bands,
            get_hip_torsion_thresholds,
            get_impact_knee_thresholds,
            get_support_distance_bands,
            get_support_knee_thresholds,
            get_support_ratio_bands,
            get_toe_angle_thresholds,
            get_whipping_thresholds,
        )

        loaded = {
            "distance_cm": get_support_distance_bands(),
            "support_ratio": get_support_ratio_bands(),
            "toe_angle": get_toe_angle_thresholds(),
            "max_folding_angle": get_folding_bands(),
            "whipping_velocity": get_whipping_thresholds(),
            "impact_knee_angle": get_impact_knee_thresholds(),
            "support_knee_angle": get_support_knee_thresholds(),
            "hip_torsion_angle": get_hip_torsion_thresholds(),
        }
    except Exception:  # noqa: BLE001
        return bands

    for key, values in loaded.items():
        expected = len(_SCORING_BAND_FALLBACK[key])
        try:
            coerced = tuple(float(v) for v in values)
        except (TypeError, ValueError):
            continue
        if len(coerced) == expected and all(np.isfinite(v) for v in coerced):
            bands[key] = coerced
    return bands


def _radar_config() -> dict[str, float]:
    """取五维雷达映射参数；异常回退现网默认。"""
    cfg = dict(_RADAR_CONFIG_FALLBACK)
    try:
        from empirical_thresholds import get_radar_config

        loaded = get_radar_config() or {}
    except Exception:  # noqa: BLE001
        return cfg
    for key, default in _RADAR_CONFIG_FALLBACK.items():
        try:
            value = float(loaded.get(key, default))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            cfg[key] = value
    return cfg


def _shank_only_fold_max() -> float:
    """ERR_B2 浅折叠膝内角上界；异常回退 140.0。"""
    try:
        from empirical_thresholds import get_shank_only_fold_max

        value = float(get_shank_only_fold_max())
    except Exception:  # noqa: BLE001
        return _SHANK_ONLY_FOLD_MAX_FALLBACK
    if not np.isfinite(value) or value <= 0:
        return _SHANK_ONLY_FOLD_MAX_FALLBACK
    return value


def _guard_knee_extension_angle(angle_deg: float) -> tuple[float, bool]:
    """触球伸展语境下的膝角解剖学矫正 + 生理区间钳位。

    Returns:
        (guarded_angle_deg, did_supplementary_flip)
    """
    angle = float(angle_deg)
    if not np.isfinite(angle):
        return float(KINEMATIC_KNEE_PHYSIO_MIN_DEG), False
    flipped = False
    # 3D 向量点乘常把伸展钝角算成锐角补角 → 强制翻转为伸展角
    if angle < KINEMATIC_KNEE_SUPPLEMENT_THRESHOLD_DEG:
        angle = 180.0 - angle
        flipped = True
    angle = min(
        max(angle, KINEMATIC_KNEE_PHYSIO_MIN_DEG),
        KINEMATIC_KNEE_PHYSIO_MAX_DEG,
    )
    return float(angle), flipped


def apply_kinematic_physical_guards(
    *,
    impact_knee_angle: float,
    support_knee_angle: float,
    distance_cm: float,
    ankle_variance: float,
) -> dict[str, Any]:
    """绝对安全的人体生理学硬拦截器（Kinematic Boundary Guard）。

    在 DeterministicScorer 扣分 / 雷达 / 对外 JSON 组装之前调用，防止离谱量纲
    （锐角补角假象、YOLO 球框漂移横距）击穿评分与雷达。

    【V3.9】踝指标已改为最大形变落差角（物理有界），不再对 ankle 做
    variance→12.0 的防暴走钳制；``ankle_variance`` 参数名保留兼容，语义为
    ``deflection_deg``。

    设计对标特斯拉 FSD：物理运动学边界约束优先于上层决策。
    """
    warnings: list[str] = []
    events: list[dict[str, Any]] = []

    raw_impact = float(impact_knee_angle)
    raw_support = float(support_knee_angle)
    raw_distance = float(distance_cm)
    raw_ankle = float(ankle_variance)

    impact_knee, impact_flipped = _guard_knee_extension_angle(raw_impact)
    if impact_flipped or abs(impact_knee - raw_impact) > 1e-9:
        msg = (
            f"impact_knee_angle {raw_impact:.2f}° → {impact_knee:.2f}° "
            f"(supplement={'yes' if impact_flipped else 'no'}, "
            f"physio=[{KINEMATIC_KNEE_PHYSIO_MIN_DEG}, {KINEMATIC_KNEE_PHYSIO_MAX_DEG}])"
        )
        warnings.append(msg)
        events.append(
            {
                "metric": "impact_knee_angle",
                "action": "knee_anatomy_guard",
                "raw": round(raw_impact, 4),
                "guarded": round(impact_knee, 4),
                "supplementary_flip": bool(impact_flipped),
            }
        )

    support_knee, support_flipped = _guard_knee_extension_angle(raw_support)
    if support_flipped or abs(support_knee - raw_support) > 1e-9:
        msg = (
            f"support_knee_angle {raw_support:.2f}° → {support_knee:.2f}° "
            f"(supplement={'yes' if support_flipped else 'no'}, "
            f"physio=[{KINEMATIC_KNEE_PHYSIO_MIN_DEG}, {KINEMATIC_KNEE_PHYSIO_MAX_DEG}])"
        )
        warnings.append(msg)
        events.append(
            {
                "metric": "support_knee_angle",
                "action": "knee_anatomy_guard",
                "raw": round(raw_support, 4),
                "guarded": round(support_knee, 4),
                "supplementary_flip": bool(support_flipped),
            }
        )

    distance = raw_distance if np.isfinite(raw_distance) else 17.5
    distance_clamped = False
    if np.isfinite(raw_distance) and raw_distance > KINEMATIC_DISTANCE_RUNAWAY_CM:
        distance = float(min(raw_distance, KINEMATIC_DISTANCE_CLAMP_CM))
        distance_clamped = True
        msg = (
            f"distance_cm {raw_distance:.2f}cm > {KINEMATIC_DISTANCE_RUNAWAY_CM}cm "
            f"→ clamped to {distance:.2f}cm (Warning)"
        )
        warnings.append(msg)
        events.append(
            {
                "metric": "distance_cm",
                "action": "distance_runaway_clamp",
                "raw": round(raw_distance, 4),
                "guarded": round(distance, 4),
                "status": "WARNING",
            }
        )
        print(f"【Warning】KinematicBoundaryGuard 支撑脚横距暴走：{msg}")

    # 形变落差原样透传（有限化）；不再钳制到 12.0
    ankle = raw_ankle if np.isfinite(raw_ankle) else 0.0
    ankle_clamped = False

    return {
        "impact_knee_angle": float(impact_knee),
        "support_knee_angle": float(support_knee),
        "distance_cm": float(distance),
        "ankle_variance": float(ankle),
        "ankle_deflection_deg": float(ankle),
        "distance_clamped": bool(distance_clamped),
        "ankle_clamped": bool(ankle_clamped),
        "impact_knee_flipped": bool(impact_flipped),
        "support_knee_flipped": bool(support_flipped),
        "warnings": warnings,
        "events": events,
        "raw": {
            "impact_knee_angle": raw_impact if np.isfinite(raw_impact) else None,
            "support_knee_angle": raw_support if np.isfinite(raw_support) else None,
            "distance_cm": raw_distance if np.isfinite(raw_distance) else None,
            "ankle_variance": raw_ankle if np.isfinite(raw_ankle) else None,
            "ankle_deflection_deg": raw_ankle if np.isfinite(raw_ankle) else None,
        },
    }

def slice_action_roi_bounds(
    impact_frame_idx: int,
    total_frames: int,
    half_window: int = ACTION_ROI_HALF_FRAMES,
) -> tuple[int, int]:
    """以 t_impact 为中心裁剪核心动作窗口 [start, end)（半开区间，最长 60 帧）。

    action_window = [max(0, t-30), min(N, t+30))
    """
    n = max(0, int(total_frames))
    t = int(impact_frame_idx)
    if n <= 0:
        return 0, 0
    t = int(max(0, min(n - 1, t)))
    start = max(0, t - int(half_window))
    end = min(n, t + int(half_window))
    if end <= start:
        end = min(n, start + 1)
    return int(start), int(end)


def _frame_joint_visibility(rec: dict, joint: str) -> float:
    """读取帧内关节点 visibility；缺省视为 1.0（旧数据无该字段）。"""
    vis = rec.get("visibility") if isinstance(rec, dict) else None
    if isinstance(vis, dict) and joint in vis:
        try:
            return float(vis.get(joint) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 1.0


def _roi_knee_angles_with_gap_fill(
    frames: list[dict],
    roi_start: int,
    roi_end_inclusive: int,
    *,
    swing_side: str = "right",
    min_visibility: float = LANDMARK_CONFIDENCE_MIN,
) -> list[Optional[float]]:
    """ROI 内逐帧膝角；单帧关键点缺失时用前后帧运动趋势补帧（max_gap=2）。"""
    hip_k, knee_k, ankle_k, _foot_k = swing_leg_joint_keys(swing_side)
    raw: list[Optional[float]] = []
    lo = int(max(0, roi_start))
    hi = int(min(len(frames) - 1, roi_end_inclusive)) if frames else -1
    if hi < lo:
        return []
    for i in range(lo, hi + 1):
        try:
            rec = frames[i]
            if not isinstance(rec, dict):
                raw.append(None)
                continue
            hip, knee, ankle = rec.get(hip_k), rec.get(knee_k), rec.get(ankle_k)
            vis_ok = (
                _frame_joint_visibility(rec, hip_k) >= min_visibility
                and _frame_joint_visibility(rec, knee_k) >= min_visibility
                and _frame_joint_visibility(rec, ankle_k) >= min_visibility
            )
            if not vis_ok or not (
                is_valid_joint_point(hip)
                and is_valid_joint_point(knee)
                and is_valid_joint_point(ankle)
            ):
                raw.append(None)
                continue
            ang = calculate_2d_angle_or_none(hip, knee, ankle)
            raw.append(float(ang) if ang is not None else None)
        except Exception:  # noqa: BLE001
            raw.append(None)
    return gap_fill_scalar_series(raw, max_gap=2)


def _roi_max_folding_angle(
    frames: list[dict],
    t_impact: int,
    roi_start: int,
    roi_end: int,
    *,
    swing_side: str = "right",
    min_valid_frames: int = FOLD_ROI_MIN_VALID_FRAMES,
) -> tuple[Optional[float], int, bool]:
    """【V3.9】后摆最大折叠角：ROI 内触球前摆动腿 XY-2D 膝角绝对极小值。

    对外 ``max_folding_angle`` = ``180 - min(膝内角)``（Z 坍缩，抗深度畸变）。
    返回 ``(max_folding_angle|None, 极值帧索引, ok)``。
    ``ok=False``：有效帧不足 ``min_valid_frames`` 或无有限膝角。

    【补帧】单帧关键点丢失时用前后帧趋势插值，降低状态/极值漏检率。
    """
    fallback_idx = int(max(roi_start, min(max(roi_start, roi_end - 1), t_impact)))
    if not frames:
        return None, fallback_idx, False
    t = int(max(roi_start, min(roi_end - 1, t_impact)))
    filled = _roi_knee_angles_with_gap_fill(
        frames, roi_start, t, swing_side=swing_side
    )
    best_interior = None
    best_idx = fallback_idx
    valid_count = 0
    for offset, knee in enumerate(filled):
        if knee is None:
            continue
        try:
            kv = float(knee)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(kv) or kv <= 0.0:
            continue
        valid_count += 1
        if best_interior is None or kv < best_interior:
            best_interior = kv
            best_idx = int(roi_start + offset)
    if best_interior is None or valid_count < int(min_valid_frames):
        return None, fallback_idx, False
    max_folding = float(max(0.0, 180.0 - float(best_interior)))
    # 与旧 YZ-矢状角对比：若整体平移 ≥5° 打日志，供 GREEN 70–100 带后期微调
    try:
        hip_k, knee_k, ankle_k, _fk = swing_leg_joint_keys(swing_side)
        rec = frames[int(best_idx)] if 0 <= int(best_idx) < len(frames) else None
        if isinstance(rec, dict):
            log_2d_vs_sagittal_shift(
                rec.get(hip_k),
                rec.get(knee_k),
                rec.get(ankle_k),
                tag=f"max_folding@{best_idx}",
            )
            print(
                f"【Biomech/2D投影】max_folding_angle={max_folding:.1f}° "
                f"(膝内角={float(best_interior):.1f}°, method=xy_2d_z_collapse, "
                f"green_band=70–100)",
                flush=True,
            )
    except Exception:  # noqa: BLE001
        pass
    return max_folding, int(best_idx), True


def fsm_should_timeout_reset(
    state_name: str,
    frames_in_state: int,
    *,
    approach_timeout: int = FSM_APPROACH_TIMEOUT_FRAMES,
    impact_timeout: int = FSM_IMPACT_LOCKED_TIMEOUT_FRAMES,
) -> bool:
    """判定射门 FSM 中间态是否应超时强制回 IDLE（供测试与外部编排复用）。"""
    name = str(state_name or "").strip().upper()
    n = int(frames_in_state)
    if name in ("APPROACH", "PREPARE", "准备击球"):
        return n >= int(approach_timeout)
    if name in ("IMPACT_LOCKED", "IMPACT"):
        return n >= int(impact_timeout)
    return False


def _roi_whipping_velocity(
    frames: list[dict],
    trajectory_data: dict,
    roi_start: int,
    roi_end: int,
) -> tuple[float, int]:
    """仅在 ROI 内取小腿/膝角速度 |ω| 峰值。

    返回 (|ω|_peak, 峰值所在帧索引)。
    """
    fallback_idx = int(max(roi_start, min(max(roi_start, roi_end - 1), (roi_start + roi_end) // 2)))
    omega_series = trajectory_data.get("angular_velocities") or trajectory_data.get(
        "knee_angular_velocities"
    )
    if omega_series is not None and len(omega_series) > 0:
        lo = max(0, min(len(omega_series), roi_start))
        hi = max(lo, min(len(omega_series), roi_end))
        peak = 0.0
        peak_idx = fallback_idx
        for i in range(lo, hi):
            mag = abs(float(omega_series[i]))
            if mag >= peak:
                peak = mag
                peak_idx = int(i)
        if peak > 0.0:
            return float(peak), int(peak_idx)

    knee_angles = trajectory_data.get("knee_angles")
    timestamps = trajectory_data.get("timestamps_sec")
    if knee_angles is not None and len(knee_angles) >= 2:
        lo = max(0, min(len(knee_angles), roi_start))
        hi = max(lo + 1, min(len(knee_angles), roi_end))
        peak = 0.0
        peak_idx = fallback_idx
        for i in range(lo + 1, hi):
            if timestamps is not None and len(timestamps) > i:
                dt = float(timestamps[i]) - float(timestamps[i - 1])
            else:
                dt = 1.0 / 30.0
            if dt <= 1e-9:
                continue
            mag = abs(float(knee_angles[i]) - float(knee_angles[i - 1])) / dt
            if mag >= peak:
                peak = mag
                peak_idx = int(i)
        if peak > 0.0:
            return float(peak), int(peak_idx)

    # 从 ROI 帧几何反推膝角再差分
    if frames and roi_end - roi_start >= 2:
        angles: list[float] = []
        times: list[float] = []
        for i in range(roi_start, roi_end):
            rec = frames[i]
            try:
                angles.append(
                    float(
                        calculate_angle(
                            rec["right_hip"],
                            rec["right_knee"],
                            rec["right_ankle"],
                            is_knee_extension=True,
                        )
                    )
                )
            except Exception:
                angles.append(angles[-1] if angles else 150.0)
            times.append(float(rec.get("timestamp_sec", i / 30.0)))
        peak = 0.0
        peak_idx = fallback_idx
        for i in range(1, len(angles)):
            dt = times[i] - times[i - 1]
            if dt <= 1e-9:
                dt = 1.0 / 30.0
            mag = abs(angles[i] - angles[i - 1]) / dt
            if mag >= peak:
                peak = mag
                peak_idx = int(roi_start + i)
        return float(peak), int(peak_idx)
    return 0.0, fallback_idx

def _linear_band_penalty(
    value: float,
    green_low: float,
    green_high: float,
    yellow_low: float,
    yellow_high: float,
    max_penalty: float,
) -> tuple[float, str]:
    """区间型指标：GREEN 内扣 0；YELLOW 带线性扣分；RED 外线性至满分。"""
    v = float(value)
    if green_low <= v <= green_high:
        return 0.0, STATUS_GREEN
    if yellow_low <= v < green_low:
        span = max(1e-9, green_low - yellow_low)
        ratio = (green_low - v) / span
        return round(min(max_penalty, max_penalty * 0.55 * ratio), 2), STATUS_YELLOW
    if green_high < v <= yellow_high:
        span = max(1e-9, yellow_high - green_high)
        ratio = (v - green_high) / span
        return round(min(max_penalty, max_penalty * 0.55 * ratio), 2), STATUS_YELLOW
    # RED：超出黄带，继续线性爬升至满分
    if v < yellow_low:
        span = max(1e-9, yellow_low - (yellow_low - (green_high - green_low)))
        excess = yellow_low - v
        ratio = min(1.0, 0.55 + 0.45 * (excess / max(span, green_high - green_low, 1.0)))
    else:
        excess = v - yellow_high
        span = max(1e-9, green_high - green_low)
        ratio = min(1.0, 0.55 + 0.45 * (excess / span))
    return round(min(max_penalty, max_penalty * ratio), 2), STATUS_RED


def _extract_ankle_window_angles(
    frames: list[dict],
    t_impact: int,
    precomputed: Optional[list] = None,
) -> tuple[list[float], bool]:
    """提取 t_impact 前后各 1 帧（共 3 点）的踝关节夹角。

    返回 ``(angles, ok)``。``ok=False`` 表示无有效实测窗（不得把假 140° 标成 measured）。
    """
    if precomputed is not None and len(precomputed) >= 3:
        vals = [float(precomputed[0]), float(precomputed[1]), float(precomputed[2])]
        if all(np.isfinite(v) for v in vals):
            return vals, True
        return [], False
    n = len(frames) if frames else 0
    if n == 0:
        return [], False
    t = int(max(0, min(n - 1, t_impact)))
    idxs = [max(0, t - 1), t, min(n - 1, t + 1)]
    angles: list[float] = []
    real_count = 0
    for i in idxs:
        rec = frames[i]
        try:
            ang = float(
                calculate_angle(rec["right_knee"], rec["right_ankle"], rec["right_foot_index"])
            )
            if np.isfinite(ang):
                angles.append(ang)
                real_count += 1
            else:
                angles.append(float("nan"))
        except Exception:
            angles.append(float("nan"))
    finite = [a for a in angles if np.isfinite(a)]
    if real_count <= 0 or not finite:
        return [], False
    # 短窗：用有限值填补至 3 点（仍视为实测窗，因至少有 1 个真实角）
    while len(finite) < 3:
        finite.append(finite[-1])
    return finite[:3], True


def _ankle_series_has_valid_window(
    series: list,
    t_impact: int,
    *,
    fps: float = DEFAULT_VIDEO_FPS,
    half_window_ms: float = ANKLE_IMPACT_HALF_WINDOW_MS,
    half_window_frames: Optional[int] = None,
) -> bool:
    """判断踝角序列在冲击窗内是否存在有限实测点。"""
    if not series:
        return False
    n = len(series)
    if half_window_frames is not None:
        half = int(max(1, half_window_frames))
        t = int(max(0, min(n - 1, int(t_impact))))
        lo, hi = max(0, t - half), min(n - 1, t + half)
    else:
        lo, hi, _half = slice_ankle_impact_window_bounds(
            n, t_impact, fps=fps, half_window_ms=half_window_ms
        )
    if hi < lo:
        return False
    for i in range(lo, hi + 1):
        try:
            if np.isfinite(float(series[i])):
                return True
        except (TypeError, ValueError):
            continue
    return False

class DeterministicScorer:
    """V2.5 纯数学生物力学评分器。

    严禁 LLM / 随机源参与任何扣分或等级判定。总分自 100.00 起按量纲线性扣减，
    保留两位小数；同一输入保证浮点误差为 0.0 的位级可复现。

    【V2.5 Action ROI】所有极值/方差量纲仅在 t_impact ± 30 帧核心窗口内解算。
    【Kinematic Boundary Guard】扣分前经 ``apply_kinematic_physical_guards`` 硬拦截。
    """

    def calculate_biomechanical_score(
        self,
        impact_frame_data: dict,
        trajectory_data: dict,
    ) -> tuple[float, dict]:
        """主入口：在固定 Action ROI 内纯数学解算 8 大量纲，返回 (TotalScore, detail)。

        detail 同时携带 V3.1 ``radar_scores`` 五维独立量化（每维满分 20）。
        """
        impact_frame_data = impact_frame_data or {}
        trajectory_data = trajectory_data or {}

        # 【Item 8】评分带阈值统一自 empirical_thresholds.py 取，禁止散落魔数
        _bands = _scoring_bands()
        ratio_gl, ratio_gh, ratio_yl, ratio_yh, ratio_center = _bands["support_ratio"]
        dist_gh, dist_yh = ratio_gh, ratio_yh  # 扣分明细签名兼容（原 cm 带已废弃）
        toe_green_high, toe_red_low = _bands["toe_angle"]
        fold_gl, fold_gh, fold_yl, fold_yh, fold_center = _bands["max_folding_angle"]
        whip_green_low, whip_yellow_low, whip_yellow_ratio = _bands["whipping_velocity"]
        ik_gl, ik_gh, ik_yl, ik_yh, ik_fallback = _bands["impact_knee_angle"]
        sk_gl, sk_gh, sk_yl, sk_yh, sk_fallback = _bands["support_knee_angle"]
        hip_gl, hip_gh, hip_yl, hip_yh, hip_fallback = _bands["hip_torsion_angle"]

        t_impact = int(
            impact_frame_data.get(
                "t_impact",
                impact_frame_data.get("contact_frame_index", trajectory_data.get("t_impact", 0)),
            )
            or 0
        )
        frames = impact_frame_data.get("frames") or trajectory_data.get("frames") or []
        knee_angles_full = trajectory_data.get("knee_angles") or impact_frame_data.get("knee_angles")

        # ---------- Action ROI 裁剪（数学保护层）----------
        if frames:
            total_n = len(frames)
        elif knee_angles_full is not None:
            total_n = len(knee_angles_full)
        else:
            total_n = int(
                trajectory_data.get("total_frames")
                or impact_frame_data.get("total_frames")
                or 0
            )
        if total_n <= 0:
            total_n = max(1, t_impact + 1)
        t_impact = int(max(0, min(total_n - 1, t_impact)))
        roi_start, roi_end = slice_action_roi_bounds(t_impact, total_n)
        roi_frames = frames[roi_start:roi_end] if frames else []

        # 瞬时量纲锚点：全局 t_impact 对应帧（必然落在 ROI 内）
        impact_rec = None
        if frames:
            impact_rec = frames[t_impact]

        # ---- a) 支撑脚偏移：全面肩宽归一化比例（废除 PCR 绝对厘米主路径）----
        distance_cm = float(ratio_center) * float(AVERAGE_CHILD_SHOULDER_WIDTH_CM)
        support_ratio: Optional[float] = None
        distance_provenance = PROVENANCE_DEFAULT
        distance_method = "neutral_band_center"
        distance_confidence = 0.0
        distance_qa: dict[str, Any] = {}
        distance_unit = "ratio"
        score_on_ratio = True
        video_fps = float(
            impact_frame_data.get("fps")
            or trajectory_data.get("fps")
            or DEFAULT_VIDEO_FPS
        )
        try:
            from empirical_thresholds import get_ankle_half_window_ms

            ankle_half_ms = float(get_ankle_half_window_ms())
        except Exception:  # noqa: BLE001
            ankle_half_ms = float(ANKLE_IMPACT_HALF_WINDOW_MS)
        if not np.isfinite(ankle_half_ms) or ankle_half_ms <= 0:
            ankle_half_ms = float(ANKLE_IMPACT_HALF_WINDOW_MS)

        # 上游肩宽比——仅作后备；禁止裸 world cm / PCR cm 当 measured
        upstream_ratio = impact_frame_data.get("support_ratio")
        if upstream_ratio is None:
            upstream_ratio = trajectory_data.get("support_ratio")
        upstream_method = (
            impact_frame_data.get("support_distance_method")
            or trajectory_data.get("support_distance_method")
        )

        shoulder_ok = False
        if impact_rec is not None:
            world = impact_rec.get("world") if isinstance(impact_rec.get("world"), dict) else None
            try:
                swing_side = infer_swing_leg_side(
                    frames if frames else [impact_rec],
                    int(t_impact),
                    explicit_side=(
                        impact_frame_data.get("swing_leg")
                        or trajectory_data.get("swing_leg")
                    ),
                )
                support_key = "left_ankle" if swing_side == "right" else "right_ankle"
                swing_foot_key = (
                    "right_foot_index" if swing_side == "right" else "left_foot_index"
                )
                swing_ankle_key = (
                    "right_ankle" if swing_side == "right" else "left_ankle"
                )
                if world:
                    ball_w = (
                        world.get(swing_foot_key)
                        or world.get(swing_ankle_key)
                        or impact_frame_data.get("ball_center")
                    )
                    sdetail = calculate_support_offset_by_shoulder_ratio(
                        world.get(support_key),
                        ball_w,
                        world.get("left_shoulder"),
                        world.get("right_shoulder"),
                        coord_space="world_m",
                        distance_mode="lateral",
                    )
                else:
                    ball_i = (
                        impact_rec.get(swing_foot_key)
                        or impact_rec.get(swing_ankle_key)
                        or impact_frame_data.get("ball_center")
                    )
                    # 图像平面权威算子：support_dist_px / shoulder_width_px
                    rdetail = calculate_support_ratio(
                        impact_rec, ball_i, support_ankle_key=support_key
                    )
                    if rdetail.get("ok"):
                        sdetail = {
                            "ok": True,
                            "support_ratio": rdetail["support_ratio"],
                            "distance_cm_estimate": float(rdetail["support_ratio"])
                            * float(AVERAGE_CHILD_SHOULDER_WIDTH_CM),
                            "shoulder_width": rdetail.get("shoulder_width_px"),
                            "raw_distance": rdetail.get("support_dist_px"),
                            "lateral_distance": rdetail.get("support_dist_px"),
                            "horizontal_distance": rdetail.get("support_dist_px"),
                            "ref_shoulder_cm": float(AVERAGE_CHILD_SHOULDER_WIDTH_CM),
                            "ratio_clamped": rdetail.get("ratio_clamped"),
                        }
                    else:
                        sdetail = calculate_support_offset_by_shoulder_ratio(
                            impact_rec.get(support_key),
                            ball_i,
                            impact_rec.get("left_shoulder"),
                            impact_rec.get("right_shoulder"),
                            coord_space="image_px",
                            distance_mode="lateral",
                        )
                if sdetail.get("ok"):
                    support_ratio = float(sdetail["support_ratio"])
                    distance_cm = float(
                        sdetail.get("distance_cm_estimate")
                        or (support_ratio * AVERAGE_CHILD_SHOULDER_WIDTH_CM)
                    )
                    distance_method = "shoulder_width_ratio"
                    distance_provenance = PROVENANCE_CALIBRATED
                    distance_confidence = 0.85
                    distance_unit = "ratio"
                    score_on_ratio = True
                    shoulder_ok = True
                    distance_qa = {
                        **distance_qa,
                        "shoulder_width": sdetail.get("shoulder_width"),
                        "raw_distance": sdetail.get("raw_distance"),
                        "lateral_distance": sdetail.get("lateral_distance"),
                        "horizontal_distance": sdetail.get("horizontal_distance"),
                        "ref_shoulder_cm": sdetail.get("ref_shoulder_cm"),
                        "distance_mode": "lateral",
                        "ratio_clamped": sdetail.get("ratio_clamped"),
                        "swing_leg": swing_side,
                        "ball_proxy": "swing_foot",
                    }
            except Exception:
                shoulder_ok = False
        if not shoulder_ok:
            if upstream_ratio is not None and (
                str(upstream_method or "").startswith("shoulder")
                or str(upstream_method or "") == "shoulder_width_ratio"
                or upstream_method is None
            ):
                try:
                    support_ratio = float(upstream_ratio)
                    distance_cm = float(support_ratio * AVERAGE_CHILD_SHOULDER_WIDTH_CM)
                    distance_method = "shoulder_width_ratio"
                    distance_provenance = PROVENANCE_CALIBRATED
                    distance_confidence = 0.7
                    distance_unit = "ratio"
                    score_on_ratio = True
                except (TypeError, ValueError):
                    pass
            # 否则保持绿带中心比例，不吃漂移的绝对 cm / PCR
            if support_ratio is None:
                support_ratio = float(ratio_center)
                distance_cm = float(support_ratio * AVERAGE_CHILD_SHOULDER_WIDTH_CM)
                distance_unit = "ratio"
                score_on_ratio = True
        # pen_dist 延后至 Kinematic Boundary Guard 之后

        # ---- b) 支撑脚尖指向角 toe_angle：[0, 15] GREEN；>25 RED 满分 ----
        toe_angle = float(
            impact_frame_data.get("toe_angle", trajectory_data.get("toe_angle", 0.0)) or 0.0
        )
        if "toe_angle" not in impact_frame_data and "toe_angle" not in trajectory_data and impact_rec is not None:
            try:
                ball = impact_frame_data.get("ball_center")
                toe_angle = _support_toe_angle_deg(impact_rec, ball)
            except Exception:
                toe_angle = 0.0
        if 0.0 <= toe_angle <= toe_green_high:
            pen_toe, st_toe = 0.0, STATUS_GREEN
        elif toe_angle > toe_red_low:
            pen_toe, st_toe = float(_MAX_PENALTY_TOE_ANGLE), STATUS_RED
        else:
            ratio = (toe_angle - toe_green_high) / max(1e-9, toe_red_low - toe_green_high)
            pen_toe = round(min(_MAX_PENALTY_TOE_ANGLE, _MAX_PENALTY_TOE_ANGLE * ratio), 2)
            st_toe = STATUS_YELLOW

        # ---- c) 摆动腿后摆折叠角：【仅 ROI 内】解算；无帧时回退预计算标量 ----
        # 【Phase 1/2】缺测不得把 80° 缺省标成 measured；按摆动腿选侧 + ROI 有效帧门槛
        fold_extreme_idx = int(t_impact)
        fold_provenance = PROVENANCE_DEFAULT
        fold_method = "neutral_band_center"
        fold_confidence = 0.0
        max_folding = fold_center
        swing_side = infer_swing_leg_side(
            frames,
            t_impact,
            ball_center=impact_frame_data.get("ball_center")
            or trajectory_data.get("ball_center"),
            explicit_side=impact_frame_data.get("swing_leg")
            or impact_frame_data.get("kicking_foot")
            or trajectory_data.get("swing_leg"),
        )
        if frames:
            fold_val, fold_extreme_idx, fold_ok = _roi_max_folding_angle(
                frames,
                t_impact,
                roi_start,
                roi_end,
                swing_side=swing_side,
            )
            if fold_ok and fold_val is not None:
                max_folding = float(fold_val)
                fold_provenance = PROVENANCE_MEASURED
                fold_method = f"roi_2d_knee_min_{swing_side}"
                fold_confidence = 0.85
            else:
                upstream_fold = trajectory_data.get("max_folding_angle")
                if upstream_fold is None and trajectory_data.get("swing_fold_angle") is not None:
                    upstream_fold = max(0.0, 180.0 - float(trajectory_data["swing_fold_angle"]))
                if upstream_fold is not None:
                    max_folding = float(upstream_fold)
                    fold_provenance = PROVENANCE_MEASURED
                    fold_method = "upstream_explicit"
                    fold_confidence = 0.75
        else:
            max_folding_raw = trajectory_data.get("max_folding_angle")
            if max_folding_raw is None and trajectory_data.get("swing_fold_angle") is not None:
                max_folding_raw = max(0.0, 180.0 - float(trajectory_data["swing_fold_angle"]))
            if max_folding_raw is not None:
                max_folding = float(max_folding_raw)
                fold_provenance = PROVENANCE_MEASURED
                fold_method = "upstream_explicit"
                fold_confidence = 0.75
            fold_extreme_idx = int(
                trajectory_data.get(
                    "backswing_extreme_frame_index",
                    impact_frame_data.get("backswing_extreme_frame_index", max(0, t_impact - 8)),
                )
                or max(0, t_impact - 8)
            )
        pen_fold, st_fold = _linear_band_penalty(
            max_folding, fold_gl, fold_gh, fold_yl, fold_yh, _MAX_PENALTY_FOLDING
        )

        # ---- d) 小腿鞭打速度：【仅 ROI 内】|ω| 峰值 ----
        whip_extreme_idx = int(max(0, t_impact - 2))
        if frames or knee_angles_full is not None or trajectory_data.get("angular_velocities"):
            whipping, whip_extreme_idx = _roi_whipping_velocity(
                frames, trajectory_data, roi_start, roi_end
            )
            # 若 ROI 差分得到 0 且外部给了标量，仅在完全无序列时才回退
            if whipping <= 0.0 and not frames and knee_angles_full is None:
                whipping = float(
                    trajectory_data.get(
                        "whipping_velocity",
                        trajectory_data.get(
                            "whipping_speed_peak",
                            impact_frame_data.get("whipping_velocity", 0.0),
                        ),
                    )
                    or 0.0
                )
        else:
            whipping = float(
                trajectory_data.get(
                    "whipping_velocity",
                    trajectory_data.get(
                        "whipping_speed_peak", impact_frame_data.get("whipping_velocity", 0.0)
                    ),
                )
                or 0.0
            )
        if whipping >= whip_green_low:
            pen_whip, st_whip = 0.0, STATUS_GREEN
        elif whipping >= whip_yellow_low:
            ratio = (whip_green_low - whipping) / max(1e-9, whip_green_low - whip_yellow_low)
            pen_whip = round(min(_MAX_PENALTY_WHIPPING, _MAX_PENALTY_WHIPPING * whip_yellow_ratio * ratio), 2)
            st_whip = STATUS_YELLOW
        else:
            ratio = min(1.0, (whip_yellow_low - whipping) / max(1e-9, whip_yellow_low))
            pen_whip = round(
                min(_MAX_PENALTY_WHIPPING, _MAX_PENALTY_WHIPPING * (whip_yellow_ratio + (1.0 - whip_yellow_ratio) * ratio)), 2
            )
            st_whip = STATUS_RED

        # ---- e) 触球瞬间膝关节夹角（触球帧，属 ROI；XY-2D / Z 坍缩）----
        _hip_k, knee_k, ankle_k, foot_k = swing_leg_joint_keys(swing_side)
        impact_knee = impact_frame_data.get("impact_knee_angle")
        if impact_knee is None and impact_rec is not None:
            try:
                impact_knee = calculate_2d_angle(
                    impact_rec[_hip_k],
                    impact_rec[knee_k],
                    impact_rec[ankle_k],
                )
                log_2d_vs_sagittal_shift(
                    impact_rec.get(_hip_k),
                    impact_rec.get(knee_k),
                    impact_rec.get(ankle_k),
                    tag=f"impact_knee@{t_impact}",
                )
            except Exception:
                impact_knee = ik_fallback
        if impact_knee is None:
            impact_knee = ik_fallback
        impact_knee = float(impact_knee)
        # 锐角补角 / 生理钳位统一由 apply_kinematic_physical_guards 处理

        # ---- f) 脚踝锁紧度：【V3.9】T0±2 最大形变落差角 + LOCKED/SLIGHT/YIELDING ----
        # 【Phase 1】空窗/假填充不得标成 measured LOCKED
        ankle_provenance = PROVENANCE_MISSING
        ankle_method = "impact_window_deflection"
        ankle_confidence = 0.0
        ankle_angles: list[float] = []
        ankle_deflection = 0.0
        stiffness_status = ANKLE_STIFFNESS_LOCKED
        ankle_dorsi_drop: Optional[float] = None
        ankle_half_frames = int(ANKLE_DEFLECTION_HALF_WINDOW_FRAMES)

        precomputed_window = impact_frame_data.get("ankle_angles_window") or trajectory_data.get(
            "ankle_angles_window"
        )
        ankle_series = impact_frame_data.get("ankle_angles_time_series") or trajectory_data.get(
            "ankle_angles_time_series"
        )
        series_from_frames = False
        ankle_vis_series = impact_frame_data.get(
            "ankle_landmark_visibility_series"
        ) or trajectory_data.get("ankle_landmark_visibility_series")
        if ankle_series is None and frames:
            built: list[float] = []
            built_vis: list[float] = []
            real_pts = 0
            for i in range(len(frames)):
                try:
                    ang = float(
                        calculate_3d_joint_angle(
                            frames[i][knee_k],
                            frames[i][ankle_k],
                            frames[i][foot_k],
                        )
                    )
                    if np.isfinite(ang):
                        built.append(ang)
                        real_pts += 1
                    else:
                        built.append(float("nan"))
                except Exception:
                    built.append(float("nan"))
                try:
                    built_vis.append(
                        min(
                            _landmark_visibility(frames[i], knee_k),
                            _landmark_visibility(frames[i], ankle_k),
                            _landmark_visibility(frames[i], foot_k),
                        )
                    )
                except Exception:
                    built_vis.append(0.0)
            if real_pts > 0:
                ankle_series = built
                ankle_vis_series = built_vis
                series_from_frames = True

        if ankle_series is not None and _ankle_series_has_valid_window(
            ankle_series,
            t_impact,
            fps=video_fps,
            half_window_ms=ankle_half_ms,
            half_window_frames=ankle_half_frames,
        ):
            ankle_deflection, stiffness_status = calculate_ankle_deflection(
                ankle_series,
                t_impact,
                half_window_frames=ankle_half_frames,
                landmark_visibility_series=ankle_vis_series,
            )
            n_ser = len(ankle_series)
            lo_a = max(0, int(t_impact) - ankle_half_frames)
            hi_a = min(n_ser - 1, int(t_impact) + ankle_half_frames)
            window_vals = []
            for i in range(lo_a, hi_a + 1):
                try:
                    v = float(ankle_series[i])
                    if np.isfinite(v):
                        window_vals.append(v)
                except (TypeError, ValueError):
                    continue
            ankle_angles = list(window_vals) if window_vals else []
            ankle_dorsi_drop = float(ankle_deflection)
            ankle_provenance = PROVENANCE_MEASURED
            ankle_method = (
                f"impact_window_deflection_frames_{swing_side}"
                if series_from_frames
                else "impact_window_deflection"
            )
            ankle_confidence = 0.8
        else:
            ankle_angles, ankle_ok = _extract_ankle_window_angles(
                frames,
                t_impact,
                precomputed=precomputed_window,
            )
            if ankle_ok and ankle_angles:
                local_t = min(ankle_half_frames, max(0, len(ankle_angles) // 2))
                ankle_deflection, stiffness_status = calculate_ankle_deflection(
                    ankle_angles,
                    local_t,
                    half_window_frames=min(ankle_half_frames, max(1, len(ankle_angles) // 2)),
                )
                ankle_dorsi_drop = float(ankle_deflection)
                ankle_provenance = PROVENANCE_MEASURED
                ankle_method = (
                    "impact_window_deflection_precomputed"
                    if precomputed_window is not None
                    else "impact_window_deflection"
                )
                ankle_confidence = 0.8
            else:
                # 缺测：评分沿用中性 0 落差，但 provenance=missing
                ankle_deflection = 0.0
                stiffness_status = ANKLE_STIFFNESS_LOCKED
                ankle_angles = []
                ankle_dorsi_drop = None
                ankle_provenance = PROVENANCE_MISSING
                ankle_method = "missing_window"
                ankle_confidence = 0.0

        # 兼容下游仍读 ankle_variance 的路径（语义 = deflection_deg）
        ankle_variance = float(ankle_deflection)

        # ---- g1) 支撑腿膝关节角度（触球帧）----
        support_knee = impact_frame_data.get(
            "support_knee_angle", trajectory_data.get("support_knee_angle")
        )
        if support_knee is None and impact_rec is not None:
            try:
                support_knee = calculate_angle(
                    impact_rec["left_hip"],
                    impact_rec["left_knee"],
                    impact_rec["left_ankle"],
                    is_knee_extension=True,
                )
            except Exception:
                support_knee = sk_fallback
        if support_knee is None:
            support_knee = sk_fallback
        support_knee = float(support_knee)

        # ---- Kinematic Boundary Guard：输出 / 扣分前的人体生理学硬拦截 ----
        kinematic_guards = apply_kinematic_physical_guards(
            impact_knee_angle=impact_knee,
            support_knee_angle=support_knee,
            distance_cm=distance_cm,
            ankle_variance=ankle_deflection,
        )
        impact_knee = float(kinematic_guards["impact_knee_angle"])
        support_knee = float(kinematic_guards["support_knee_angle"])
        distance_cm = float(kinematic_guards["distance_cm"])
        ankle_deflection = float(
            kinematic_guards.get("ankle_deflection_deg", kinematic_guards["ankle_variance"])
        )
        ankle_variance = float(ankle_deflection)
        if kinematic_guards.get("distance_clamped"):
            # 离谱横距降级：方法标注 + 置信下调，保留 Warning 痕迹
            distance_method = f"{distance_method}|kinematic_clamp"
            distance_confidence = round(min(float(distance_confidence), 0.55), 3)

        if support_ratio is None:
            support_ratio = float(ratio_center)
        pen_dist, st_dist = _linear_band_penalty(
            float(support_ratio),
            ratio_gl,
            ratio_gh,
            ratio_yl,
            ratio_yh,
            _MAX_PENALTY_DISTANCE_CM,
        )
        # 护栏仍用估计 cm；评分与对外 value 一律为肩宽比
        distance_cm = float(support_ratio) * float(AVERAGE_CHILD_SHOULDER_WIDTH_CM)
        score_on_ratio = True
        distance_unit = "ratio"
        if kinematic_guards.get("distance_clamped") and st_dist == STATUS_GREEN:
            st_dist = STATUS_YELLOW

        # 【T0 精度等级】fallback_midframe 时 T0 定位精度不足（无抛物线三点），
        # 放宽膝关节黄灯阈值 ±5°，避免因帧级抖动触发误扣分
        _T0_FALLBACK_KNEE_SLACK = 5.0
        _t0_quality = (impact_frame_data.get("t0_quality") or "")
        if _t0_quality == "fallback_midframe":
            ik_yl = ik_yl - _T0_FALLBACK_KNEE_SLACK
            ik_yh = ik_yh + _T0_FALLBACK_KNEE_SLACK
            sk_yl = sk_yl - _T0_FALLBACK_KNEE_SLACK
            sk_yh = sk_yh + _T0_FALLBACK_KNEE_SLACK

        pen_iknee, st_iknee = _linear_band_penalty(
            impact_knee, ik_gl, ik_gh, ik_yl, ik_yh, _MAX_PENALTY_IMPACT_KNEE
        )
        pen_sknee, st_sknee = _linear_band_penalty(
            support_knee, sk_gl, sk_gh, sk_yl, sk_yh, _MAX_PENALTY_SUPPORT_KNEE
        )

        if ankle_deflection < ANKLE_DEFLECTION_GREEN:
            pen_ankle, st_ankle = 0.0, STATUS_GREEN
        elif ankle_deflection <= ANKLE_DEFLECTION_YELLOW_HIGH:
            ratio = (ankle_deflection - ANKLE_DEFLECTION_GREEN) / (
                ANKLE_DEFLECTION_YELLOW_HIGH - ANKLE_DEFLECTION_GREEN
            )
            pen_ankle = round(min(_MAX_PENALTY_ANKLE, _MAX_PENALTY_ANKLE * 0.55 * ratio), 2)
            st_ankle = STATUS_YELLOW
        else:
            pen_ankle, st_ankle = float(_MAX_PENALTY_ANKLE), STATUS_RED

        # ---- g2) 髋关节相对扭转角（触球帧）----
        hip_torsion = impact_frame_data.get(
            "hip_torsion_angle", trajectory_data.get("hip_torsion_angle")
        )
        if hip_torsion is None and impact_rec is not None:
            try:
                hip_torsion = _hip_relative_torsion_deg(impact_rec)
            except Exception:
                hip_torsion = hip_fallback
        if hip_torsion is None:
            hip_torsion = hip_fallback
        hip_torsion = float(hip_torsion)
        pen_hip, st_hip = _linear_band_penalty(
            hip_torsion, hip_gl, hip_gh, hip_yl, hip_yh, _MAX_PENALTY_HIP_TORSION
        )

        # ---- g3) 躯干倾角（近端稳定性）：T0 优先，缺省 T₋₁ / 上游透传 ----
        trunk_lean = impact_frame_data.get(
            "trunk_lean_angle", trajectory_data.get("trunk_lean_angle")
        )
        if trunk_lean is None and impact_rec is not None:
            try:
                ball_for_lean = (
                    impact_frame_data.get("ball_center")
                    or trajectory_data.get("ball_center")
                )
                trunk_lean = calculate_trunk_lean(
                    impact_rec, ball_center=ball_for_lean
                )
            except Exception:
                trunk_lean = None
        if trunk_lean is None and frames and 0 <= int(t_impact) - 1 < len(frames):
            try:
                trunk_lean = calculate_trunk_lean(frames[int(t_impact) - 1])
            except Exception:
                trunk_lean = None
        trunk_color = classify_trunk_lean_status(
            float(trunk_lean) if trunk_lean is not None else None
        )
        if trunk_lean is None:
            trunk_lean = 8.0  # 绿带中性默认（微前倾）
            trunk_color = "GREEN"
            trunk_provenance = PROVENANCE_MISSING
        else:
            trunk_lean = float(trunk_lean)
            trunk_provenance = PROVENANCE_MEASURED
        if trunk_color == "GREEN":
            pen_trunk, st_trunk = 0.0, STATUS_GREEN
        elif trunk_color == "YELLOW":
            # 轻微失衡：直立偏后或前倾偏大 → 约半档惩罚
            pen_trunk = round(float(_MAX_PENALTY_TRUNK_LEAN) * 0.4, 2)
            st_trunk = STATUS_YELLOW
        else:
            pen_trunk, st_trunk = float(_MAX_PENALTY_TRUNK_LEAN), STATUS_RED

        # 决策树关联：支撑过远 + 后仰 → 在扣分明细中标注代偿（不重复叠满扣）
        trunk_linked = bool(
            impact_frame_data.get("trunk_lean_linked_to_wide_stance")
            or trajectory_data.get("trunk_lean_linked_to_wide_stance")
        )
        if (
            not trunk_linked
            and st_dist == STATUS_RED
            and trunk_lean < float(TRUNK_LEAN_GREEN_LOW_DEG)
        ):
            trunk_linked = True

        # 单项上限二次钳制（防止阈值/线性系数回归导致单项击穿）
        pen_dist = min(float(pen_dist), float(_MAX_PENALTY_DISTANCE_CM))
        pen_toe = min(float(pen_toe), float(_MAX_PENALTY_TOE_ANGLE))
        pen_fold = min(float(pen_fold), float(_MAX_PENALTY_FOLDING))
        pen_whip = min(float(pen_whip), float(_MAX_PENALTY_WHIPPING))
        pen_iknee = min(float(pen_iknee), float(_MAX_PENALTY_IMPACT_KNEE))
        pen_ankle = min(float(pen_ankle), float(_MAX_PENALTY_ANKLE))
        pen_sknee = min(float(pen_sknee), float(_MAX_PENALTY_SUPPORT_KNEE))
        pen_hip = min(float(pen_hip), float(_MAX_PENALTY_HIP_TORSION))
        pen_trunk = min(float(pen_trunk), float(_MAX_PENALTY_TRUNK_LEAN))

        total_penalty = (
            pen_dist
            + pen_toe
            + pen_fold
            + pen_whip
            + pen_iknee
            + pen_ankle
            + pen_sknee
            + pen_hip
            + pen_trunk
        )
        total_score = round(max(0.0, 100.00 - float(total_penalty)), 2)

        # 【错位修复】扣分明细必须引用与左侧展示完全相同的实测变量
        swing_fold_interior = float(max(0.0, 180.0 - float(max_folding)))
        deductions = self._build_deduction_reasons(
            distance_cm=distance_cm,
            support_ratio=support_ratio,
            score_on_ratio=score_on_ratio,
            pen_dist=pen_dist,
            st_dist=st_dist,
            dist_gh=dist_gh,
            dist_yh=dist_yh,
            ratio_gh=ratio_gh,
            ratio_yh=ratio_yh,
            max_folding=max_folding,
            swing_fold_interior=swing_fold_interior,
            pen_fold=pen_fold,
            st_fold=st_fold,
            fold_gl=fold_gl,
            fold_gh=fold_gh,
            fold_yh=fold_yh,
            impact_knee=impact_knee,
            pen_iknee=pen_iknee,
            st_iknee=st_iknee,
            ik_gh=ik_gh,
            ankle_variance=ankle_variance,
            pen_ankle=pen_ankle,
            st_ankle=st_ankle,
            toe_angle=toe_angle,
            pen_toe=pen_toe,
            st_toe=st_toe,
            whipping=whipping,
            pen_whip=pen_whip,
            st_whip=st_whip,
            support_knee=support_knee,
            pen_sknee=pen_sknee,
            st_sknee=st_sknee,
            hip_torsion=hip_torsion,
            pen_hip=pen_hip,
            st_hip=st_hip,
            trunk_lean=float(trunk_lean),
            pen_trunk=pen_trunk,
            st_trunk=st_trunk,
            trunk_linked=trunk_linked,
        )

        # ---------- V3.1 五维独立量化雷达（每维满分 20，保底 0，1 位小数）----------
        radar_scores = self._compose_radar_scores(
            pen_dist=pen_dist,
            pen_sknee=pen_sknee,
            pen_fold=pen_fold,
            ankle_variance=ankle_variance,
            whipping=whipping,
            total_penalty=float(total_penalty),
        )

        landing_idx = int(
            impact_frame_data.get(
                "landing_frame_index",
                trajectory_data.get("landing_frame_index", max(0, t_impact - 3)),
            )
            or max(0, t_impact - 3)
        )

        ankle_measured = is_aigc_measurable_provenance(ankle_provenance)
        dist_extra = {}
        if distance_qa:
            dist_extra["ball_bbox_qa"] = {
                "ok": distance_qa.get("ok"),
                "reason": distance_qa.get("reason"),
                "diameter_px": distance_qa.get("diameter_px"),
                "aspect_ratio": distance_qa.get("aspect_ratio"),
            }
            if distance_qa.get("world_crosscheck") is not None:
                dist_extra["ball_bbox_qa"]["world_crosscheck"] = distance_qa.get(
                    "world_crosscheck"
                )
        if kinematic_guards.get("distance_clamped"):
            dist_extra["kinematic_guard"] = {
                "status": "WARNING",
                "action": "distance_runaway_clamp",
                "raw": (kinematic_guards.get("raw") or {}).get("distance_cm"),
                "guarded": round(distance_cm, 2),
            }
        if support_ratio is not None:
            dist_extra["support_ratio"] = round(float(support_ratio), 4)
            dist_extra["distance_cm_estimate"] = round(float(distance_cm), 2)
            dist_extra["ref_shoulder_cm"] = float(AVERAGE_CHILD_SHOULDER_WIDTH_CM)
        if distance_qa.get("shoulder_width") is not None:
            dist_extra["shoulder_width"] = distance_qa.get("shoulder_width")
        fold_extra = {"swing_leg": swing_side}
        ankle_extra = {
            "deflection_deg": round(ankle_deflection, 2) if ankle_measured else None,
            # 兼容旧消费者：variance 字段现承载 deflection_deg
            "variance": round(ankle_deflection, 4) if ankle_measured else None,
            "scoring_variance": round(ankle_deflection, 4),
            "stiffness_status": stiffness_status if ankle_measured else None,
            "ankle_angles_window": (
                [round(a, 2) for a in ankle_angles] if ankle_measured else []
            ),
            "dorsiflex_drop_deg": (
                round(float(ankle_dorsi_drop), 2)
                if ankle_measured and ankle_dorsi_drop is not None
                else None
            ),
            "window_half_frames": int(ankle_half_frames),
            "fps": round(float(video_fps), 3),
            "swing_leg": swing_side,
        }
        indicators = {
            "distance_cm": pack_focus_indicator(
                scoring_value=float(support_ratio),
                provenance=distance_provenance,
                method=distance_method,
                confidence=distance_confidence,
                unit="ratio",
                status=st_dist,
                penalty=pen_dist,
                green_band=[ratio_gl, ratio_gh],
                extreme_frame_index=int(landing_idx),
                decimals=4,
                extra=dist_extra or None,
            ),
            "toe_angle": {
                "value": round(toe_angle, 2),
                "unit": "deg",
                "status": st_toe,
                "penalty": pen_toe,
                "green_band": [0.0, toe_green_high],
                "extreme_frame_index": int(landing_idx),
                "provenance": PROVENANCE_MEASURED,
                "method": "impact_frame",
            },
            "max_folding_angle": pack_focus_indicator(
                scoring_value=max_folding,
                provenance=fold_provenance,
                method=fold_method,
                confidence=fold_confidence,
                unit="deg",
                status=st_fold,
                penalty=pen_fold,
                green_band=[fold_gl, fold_gh],
                extreme_frame_index=int(fold_extreme_idx),
                extra=fold_extra,
            ),
            "whipping_velocity": {
                "value": round(whipping, 2),
                "unit": "deg/s",
                "status": st_whip,
                "penalty": pen_whip,
                "green_band": [whip_green_low, None],
                "extreme_frame_index": int(whip_extreme_idx),
                "provenance": PROVENANCE_MEASURED,
                "method": "roi_peak",
            },
            "impact_knee_angle": {
                "value": round(impact_knee, 2),
                "unit": "deg",
                "status": st_iknee,
                "penalty": pen_iknee,
                "green_band": [ik_gl, ik_gh],
                "extreme_frame_index": int(t_impact),
                "provenance": PROVENANCE_MEASURED,
                "method": "impact_frame_xy_2d",
            },
            "ankle_rigidity": pack_focus_indicator(
                scoring_value=ankle_deflection,
                provenance=ankle_provenance,
                method=ankle_method,
                confidence=ankle_confidence,
                unit="deg",
                status=st_ankle,
                penalty=pen_ankle,
                green_band=[0.0, ANKLE_DEFLECTION_GREEN],
                extreme_frame_index=int(t_impact),
                decimals=2,
                extra=ankle_extra,
            ),
            "support_knee_angle": {
                "value": round(support_knee, 2),
                "unit": "deg",
                "status": st_sknee,
                "penalty": pen_sknee,
                "green_band": [sk_gl, sk_gh],
                "extreme_frame_index": int(landing_idx),
                "provenance": PROVENANCE_MEASURED,
                "method": "impact_frame",
            },
            "hip_torsion_angle": {
                "value": round(hip_torsion, 2),
                "unit": "deg",
                "status": st_hip,
                "penalty": pen_hip,
                "green_band": [hip_gl, hip_gh],
                "extreme_frame_index": int(t_impact),
                "provenance": PROVENANCE_ESTIMATED,
                "method": "hip_torsion_xz_plane",
            },
            "trunk_lean_angle": {
                "value": round(float(trunk_lean), 2),
                "unit": "deg",
                "status": st_trunk,
                "penalty": pen_trunk,
                "green_band": [
                    float(TRUNK_LEAN_GREEN_LOW_DEG),
                    float(TRUNK_LEAN_GREEN_HIGH_DEG),
                ],
                "extreme_frame_index": int(t_impact),
                "provenance": trunk_provenance,
                "method": "trunk_lean_2d_t0",
                "linked_to_wide_stance": bool(trunk_linked),
                "yellow_band": [
                    float(TRUNK_LEAN_YELLOW_BACK_DEG),
                    float(TRUNK_LEAN_GREEN_LOW_DEG),
                ],
                "red_outside": [
                    float(TRUNK_LEAN_YELLOW_BACK_DEG),
                    float(TRUNK_LEAN_RED_FORWARD_DEG),
                ],
            },
        }
        # 膝角解剖学矫正痕迹写入指标（仅在实际发生翻转/钳位时）
        raw_map = kinematic_guards.get("raw") or {}
        if kinematic_guards.get("impact_knee_flipped") or (
            raw_map.get("impact_knee_angle") is not None
            and abs(float(raw_map["impact_knee_angle"]) - float(impact_knee)) > 1e-6
        ):
            indicators["impact_knee_angle"]["kinematic_guard"] = {
                "action": "knee_anatomy_guard",
                "raw": raw_map.get("impact_knee_angle"),
                "guarded": round(impact_knee, 2),
                "supplementary_flip": bool(kinematic_guards.get("impact_knee_flipped")),
            }
        if kinematic_guards.get("support_knee_flipped") or (
            raw_map.get("support_knee_angle") is not None
            and abs(float(raw_map["support_knee_angle"]) - float(support_knee)) > 1e-6
        ):
            indicators["support_knee_angle"]["kinematic_guard"] = {
                "action": "knee_anatomy_guard",
                "raw": raw_map.get("support_knee_angle"),
                "guarded": round(support_knee, 2),
                "supplementary_flip": bool(kinematic_guards.get("support_knee_flipped")),
            }

        # 【脏数据拦截】任一量纲 None/NaN → 默认值 + YELLOW，禁止 null 流向 LLM/前端
        indicators = sanitize_eight_dimension_indicators(indicators)

        metric_extreme_frames = {
            key: int(item["extreme_frame_index"]) for key, item in indicators.items()
        }

        # ---- Sprint 1：支撑脚 / 摆腿时空热力图（有完整帧序列时生成）----
        heatmap_base64 = None
        spatial_trajectory = None
        if frames:
            ball_center = (
                impact_frame_data.get("ball_center")
                or trajectory_data.get("ball_center")
                or (impact_rec.get("right_foot_index") if impact_rec else None)
            )
            try:
                heat_payload = build_spatial_heatmap_payload(
                    frames, t_impact, ball_center_t_impact=ball_center
                )
                heat_payload.pop("_canvas_bgr", None)
                heatmap_base64 = heat_payload.get("heatmap_base64")
                spatial_trajectory = {
                    k: v
                    for k, v in heat_payload.items()
                    if k not in ("heatmap_base64", "heatmap_data_uri", "_canvas_bgr")
                }
            except Exception:
                heatmap_base64 = None
                spatial_trajectory = None

        # 具身隐喻：问题关节高亮（像素坐标 + 临床绝对时间戳 → 前端 Canvas）
        joint_highlights: list = []
        if frames:
            try:
                abs_ts = [
                    float(rec.get("timestamp_sec", i / max(video_fps, 1e-6)))
                    if isinstance(rec, dict)
                    else float(i / max(video_fps, 1e-6))
                    for i, rec in enumerate(frames)
                ]
                joint_highlights = build_joint_highlights(
                    frames,
                    int(t_impact),
                    indicators,
                    swing_side=swing_side,
                    absolute_timestamps=abs_ts,
                    fps=float(video_fps),
                    roi_start=int(roi_start) if roi_start is not None else None,
                )
            except Exception:
                joint_highlights = []

        detail = {
            "TotalScore": total_score,
            "t_impact": int(t_impact),
            "base_score": 100.00,
            "total_penalty": round(float(total_penalty), 2),
            "indicators": indicators,
            "deductions": deductions,
            "metric_extreme_frames": metric_extreme_frames,
            "radar_scores": radar_scores,
            "scoring_engine": "DeterministicScorer_V3.5",
            "llm_participated": False,
            "swing_leg": swing_side,
            "kinematic_guards": {
                "warnings": list(kinematic_guards.get("warnings") or []),
                "events": list(kinematic_guards.get("events") or []),
                "distance_clamped": bool(kinematic_guards.get("distance_clamped")),
                "ankle_clamped": bool(kinematic_guards.get("ankle_clamped")),
            },
            "action_roi": {
                "start": int(roi_start),
                "end": int(roi_end),
                "half_window": int(ACTION_ROI_HALF_FRAMES),
                "length": int(max(0, roi_end - roi_start)),
                "roi_frame_count": int(len(roi_frames)) if roi_frames else int(max(0, roi_end - roi_start)),
            },
            # Sprint 1：单趟次支撑脚 / 摆腿时空热力图（PNG base64，无 data URI 前缀）
            "heatmap_base64": heatmap_base64,
            "spatial_trajectory": spatial_trajectory,
            # 具身隐喻：问题关节 2D 像素 + RED/YELLOW/GREEN
            "joint_highlights": joint_highlights,
        }
        return total_score, detail

    @staticmethod
    def _clamp_radar(value: float) -> float:
        """雷达维分数：[0, 20]，保留 1 位小数。"""
        return round(max(0.0, min(20.0, float(value))), 1)

    @staticmethod
    def _build_deduction_reasons(
        *,
        distance_cm: float,
        support_ratio: Optional[float] = None,
        score_on_ratio: bool = False,
        pen_dist: float,
        st_dist: str,
        dist_gh: float,
        dist_yh: float,
        ratio_gh: float = 0.70,
        ratio_yh: float = 0.90,
        max_folding: float,
        swing_fold_interior: float,
        pen_fold: float,
        st_fold: str,
        fold_gl: float,
        fold_gh: float,
        fold_yh: float,
        impact_knee: float,
        pen_iknee: float,
        st_iknee: str,
        ik_gh: float,
        ankle_variance: float,
        pen_ankle: float,
        st_ankle: str,
        toe_angle: float,
        pen_toe: float,
        st_toe: str,
        whipping: float,
        pen_whip: float,
        st_whip: str,
        support_knee: float,
        pen_sknee: float,
        st_sknee: str,
        hip_torsion: float,
        pen_hip: float,
        st_hip: str,
        trunk_lean: float = 8.0,
        pen_trunk: float = 0.0,
        st_trunk: str = STATUS_GREEN,
        trunk_linked: bool = False,
    ) -> list[dict[str, Any]]:
        """生成扣分明细：条件判断所引用的变量 ≡ 左侧展示变量。"""
        out: list[dict[str, Any]] = []
        # 参数名保留 ankle_variance；V3.9 起语义为形变落差角 deflection_deg
        ankle_deflection = float(ankle_variance)

        def _push(
            metric_key: str,
            measured: float,
            unit: str,
            penalty: float,
            status: str,
            reason: str,
            *,
            error_code: Optional[str] = None,
        ) -> None:
            if float(penalty) <= 0.0 and status == STATUS_GREEN:
                return
            if status == STATUS_GREEN:
                return
            entry: dict[str, Any] = {
                "metric_key": metric_key,
                "measured_value": round(float(measured), 2),
                "unit": unit,
                "penalty": round(float(penalty), 2),
                "status": status,
                "reason": reason,
            }
            if error_code:
                entry["error_code"] = error_code
            out.append(entry)

        if st_dist != STATUS_GREEN:
            r = float(support_ratio) if support_ratio is not None else float(
                distance_cm / max(AVERAGE_CHILD_SHOULDER_WIDTH_CM, 1e-6)
            )
            if r > ratio_yh:
                reason = f"支撑脚横距比例 {r:.2f}（严重外挂，>{ratio_yh:.2f}）"
                code = "ERR_A2_SUPPORT_WIDE"
            elif r > ratio_gh:
                reason = f"支撑脚横距比例 {r:.2f}（略偏远，理想 {ratio_gh:.2f} 附近）"
                code = "ERR_SUPPORT_LATERAL"
            elif r < 0.25:
                reason = f"支撑脚横距比例 {r:.2f}（过近，<{0.25:.2f}）"
                code = "ERR_SUPPORT_TOO_CLOSE"
            else:
                reason = f"支撑脚横距比例 {r:.2f}（略偏近，理想带 0.40–0.70）"
                code = "ERR_SUPPORT_LATERAL"
            _push(
                "distance_cm",
                r,
                "ratio",
                pen_dist,
                st_dist,
                reason,
                error_code=code,
            )

        if st_fold != STATUS_GREEN:
            # 判断与文案一律用 swing_fold_interior（与前端蓄力膝角同源），
            # 禁止出现「测得 120° 却写 >170°」的错位。
            if swing_fold_interior > 140.0:
                reason = (
                    f"后摆膝内角 {swing_fold_interior:.1f}° > 140°（几乎没折叠；"
                    f"折叠深度 {max_folding:.1f}°）"
                )
                code = "ERR_B1_STRAIGHT_LEG"
            elif swing_fold_interior < 70.0:
                reason = (
                    f"后摆膝内角 {swing_fold_interior:.1f}° < 70°（过度折叠；"
                    f"折叠深度 {max_folding:.1f}° > {fold_yh:.0f}°）"
                )
                code = "ERR_SWING_FOLD"
            else:
                reason = (
                    f"后摆膝内角 {swing_fold_interior:.1f}° 略偏离 90–130° 合理发力区"
                    f"（折叠深度 {max_folding:.1f}°，绿带深度 {fold_gl:.0f}–"
                    f"{fold_gh:.0f}°）"
                )
                code = "ERR_SWING_FOLD"
            _push(
                "max_folding_angle",
                max_folding,
                "deg",
                pen_fold,
                st_fold,
                reason,
                error_code=code,
            )

        if st_iknee != STATUS_GREEN:
            if impact_knee > ik_gh:
                reason = (
                    f"触球瞬间膝夹角 {impact_knee:.1f}° > {ik_gh:.0f}°（绝对直腿）"
                )
                code = "ERR_B1_STRAIGHT_LEG"
            else:
                reason = f"触球瞬间膝夹角 {impact_knee:.1f}° 偏离理想伸展带"
                code = "ERR_KNEE_STIFF"
            _push(
                "impact_knee_angle",
                impact_knee,
                "deg",
                pen_iknee,
                st_iknee,
                reason,
                error_code=code,
            )

        if st_ankle != STATUS_GREEN:
            _push(
                "ankle_rigidity",
                ankle_deflection,
                "deg",
                pen_ankle,
                st_ankle,
                f"击球窗踝形变落差 {ankle_deflection:.1f}°，锁踝不足"
                + (
                    "（严重松弛）"
                    if st_ankle == STATUS_RED
                    else "（轻微卸力）"
                ),
                error_code="ERR_C1_LOOSE_ANKLE",
            )
        if st_toe != STATUS_GREEN:
            _push(
                "toe_angle",
                toe_angle,
                "deg",
                pen_toe,
                st_toe,
                f"支撑脚尖指向偏角 {toe_angle:.1f}°",
                error_code="ERR_C2_TOE_POKE",
            )
        if st_whip != STATUS_GREEN:
            _push(
                "whipping_velocity",
                whipping,
                "deg/s",
                pen_whip,
                st_whip,
                f"鞭打峰值角速度 {whipping:.1f}°/s 不足",
                error_code="ERR_FOLLOW_THROUGH",
            )
        if st_sknee != STATUS_GREEN:
            _push(
                "support_knee_angle",
                support_knee,
                "deg",
                pen_sknee,
                st_sknee,
                f"支撑膝角 {support_knee:.1f}° 缓冲不足或过度屈曲",
                error_code="ERR_KNEE_STIFF",
            )
        if st_hip != STATUS_GREEN:
            _push(
                "hip_torsion_angle",
                hip_torsion,
                "deg",
                pen_hip,
                st_hip,
                f"髋扭转角 {hip_torsion:.1f}° 偏离转髋充分区",
                error_code="ERR_TORSO_TILT",
            )
        if st_trunk != STATUS_GREEN:
            lean_reason = f"躯干倾角 {float(trunk_lean):.1f}°"
            if float(trunk_lean) < float(TRUNK_LEAN_GREEN_LOW_DEG):
                lean_reason += "（后仰/过于直立，近端失稳）"
            else:
                lean_reason += "（前倾过大，核心代偿折腰）"
            if trunk_linked:
                lean_reason += "；与支撑脚过远联动——后仰多为够球代偿"
            _push(
                "trunk_lean_angle",
                float(trunk_lean),
                "deg",
                pen_trunk,
                st_trunk,
                lean_reason,
                error_code="ERR_D1_TRUNK_LEAN",
            )
        return out

    def _compose_radar_scores(
        self,
        *,
        pen_dist: float,
        pen_sknee: float,
        pen_fold: float,
        ankle_variance: float,
        whipping: float,
        total_penalty: float,
    ) -> dict[str, float]:
        """
        V3.1 五维儿童游戏化雷达：与单一 TotalScore 并行输出。

        - support_stability：支撑脚偏移 + 支撑膝缓冲惩罚折算
        - backswing_folding：后摆最大折叠角惩罚折算
        - ankle_rigidity：脚踝形变落差分档（阈值源自 _radar_config()['ankle_*_score']）
        - whipping_velocity：小腿峰值角速度（>= radar['whipping_full_score_deg_s'] → 20，否则线性递减）
        - approach_rhythm：助跑占位，由整体流畅度映射到 [floor, ceiling]（确定性，零随机）
        """
        rc = _radar_config()
        ankle_locked_score: float = rc["ankle_locked_score"]
        ankle_slight_score: float = rc["ankle_slight_score"]
        ankle_yielding_score: float = rc["ankle_yielding_score"]
        whip_full: float = rc["whipping_full_score_deg_s"]
        rhythm_floor: float = rc["approach_rhythm_floor"]
        rhythm_ceiling: float = rc["approach_rhythm_ceiling"]
        rhythm_slope: float = rc["approach_penalty_slope"]

        # 支撑与稳固：两路惩罚按各自满分权重折算到 20 分制
        support_denom = _MAX_PENALTY_DISTANCE_CM + _MAX_PENALTY_SUPPORT_KNEE
        support_stability = self._clamp_radar(
            20.0 * (1.0 - (float(pen_dist) + float(pen_sknee)) / support_denom)
        )

        # 蓄力与折叠：折叠惩罚满扣 → 0 分
        backswing_folding = self._clamp_radar(
            20.0 * (1.0 - float(pen_fold) / _MAX_PENALTY_FOLDING)
        )

        # 锁踝与刚性：离散档位（与 ANKLE_DEFLECTION_* 阈值对齐）
        if ankle_variance < ANKLE_DEFLECTION_GREEN:
            ankle_rigidity = ankle_locked_score
        elif ankle_variance <= ANKLE_DEFLECTION_YELLOW_HIGH:
            ankle_rigidity = ankle_slight_score
        else:
            ankle_rigidity = ankle_yielding_score

        # 鞭打与随摆：>= whip_full → 满分，否则按比例递减至 0
        if whipping >= whip_full:
            whipping_velocity = 20.0
        else:
            whipping_velocity = self._clamp_radar((float(whipping) / whip_full) * 20.0)

        # 助跑与进袭：占位符 —— 用总惩罚映射流畅度到 [floor, ceiling]，保持确定性可复现
        # total_penalty≈0 → ceiling；惩罚升高逐步贴近 floor；永不低于 floor（鼓励性保底）
        approach_rhythm = self._clamp_radar(
            max(rhythm_floor, min(rhythm_ceiling, rhythm_ceiling - float(total_penalty) * rhythm_slope))
        )

        return {
            "support_stability": support_stability,
            "backswing_folding": backswing_folding,
            "ankle_rigidity": ankle_rigidity,
            "whipping_velocity": whipping_velocity,
            "approach_rhythm": approach_rhythm,
        }


def calculate_biomechanical_score(
    impact_frame_data: dict,
    trajectory_data: dict,
) -> tuple[float, dict]:
    """模块级入口：委托 DeterministicScorer，便于测试与外部直接调用。"""
    return DeterministicScorer().calculate_biomechanical_score(impact_frame_data, trajectory_data)
