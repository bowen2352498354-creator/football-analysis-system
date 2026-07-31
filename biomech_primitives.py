# -*- coding: utf-8 -*-
"""biomech_primitives.py — V3.1/Phase2 生物力学实测原语（唯一权威实现）。

error_diagnoser / pose_tracker 均应委托本模块，禁止再复制 PCR / 3D 夹角 / 踝方差算法。
本模块零依赖业务层，严禁 LLM / 启发式评分干预。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# --------------------------------------------------------------------------
# 物理常数与刚度分档
# --------------------------------------------------------------------------
STANDARD_BALL_DIAMETER_CM = 21.0
DEFAULT_EMPIRICAL_PCR = STANDARD_BALL_DIAMETER_CM / 84.0
AVERAGE_CHILD_HEIGHT_CM = 145.0
SUPPORT_FOOT_OFFSET_MAX_CM = 60.0

ANKLE_STIFFNESS_LOCKED = "LOCKED"
ANKLE_STIFFNESS_SLIGHT_DEFORMATION = "SLIGHT_DEFORMATION"
ANKLE_STIFFNESS_YIELDING = "YIELDING"
ANKLE_STIFFNESS_LOCKED_MAX_VAR = 2.0
ANKLE_STIFFNESS_SLIGHT_MAX_VAR = 5.0
ANKLE_LANDMARK_VISIBILITY_MIN = 0.5

# 【Phase 2 / 横距防抖】YOLO 球框：max(w,h) < 10px 严禁参与比例尺
BALL_BBOX_MIN_DIAMETER_PX = 10.0
BALL_BBOX_MAX_ASPECT_RATIO = 2.5  # max(w,h)/min(w,h)
PCR_WORLD_CROSSCHECK_TOL_CM = 4.0

# 【Phase 2】踝冲击窗：默认总宽约 100ms（半窗 50ms）
ANKLE_IMPACT_HALF_WINDOW_MS = 50.0
DEFAULT_VIDEO_FPS = 30.0

# 【Phase 2】折叠角 ROI 最少有效帧
FOLD_ROI_MIN_VALID_FRAMES = 3


def _as_vec3(point) -> np.ndarray:
    arr = np.asarray(point, dtype=np.float64).reshape(-1)
    if arr.size >= 3:
        return arr[:3].astype(np.float64, copy=False)
    if arr.size == 2:
        return np.array([float(arr[0]), float(arr[1]), 0.0], dtype=np.float64)
    if arr.size == 1:
        return np.array([float(arr[0]), 0.0, 0.0], dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def estimate_fallback_pcr(body_h_px: Optional[float] = None) -> float:
    """备用比例尺：身高像素 → cm/px；缺省退回经验球径 PCR。"""
    try:
        if body_h_px is not None and np.isfinite(float(body_h_px)) and float(body_h_px) > 1e-6:
            return float(AVERAGE_CHILD_HEIGHT_CM) / float(body_h_px)
    except (TypeError, ValueError):
        pass
    return float(DEFAULT_EMPIRICAL_PCR)


def estimate_body_height_px(
    nose_xy=None,
    heel_xy=None,
    *,
    landmarks: Optional[dict] = None,
) -> Optional[float]:
    """鼻子→脚后跟像素身高；可从 landmarks 字典自动取点。"""
    nose = nose_xy
    heel = heel_xy
    if landmarks and isinstance(landmarks, dict):
        if nose is None:
            nose = landmarks.get("nose")
        if heel is None:
            heel = (
                landmarks.get("left_heel")
                or landmarks.get("right_heel")
                or landmarks.get("left_ankle")
                or landmarks.get("right_ankle")
            )
        if nose is None:
            ls = landmarks.get("left_shoulder")
            rs = landmarks.get("right_shoulder")
            if ls is not None and rs is not None:
                try:
                    a = _as_vec3(ls)
                    b = _as_vec3(rs)
                    nose = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
                except Exception:  # noqa: BLE001
                    nose = None
    if nose is None or heel is None:
        return None
    try:
        n = _as_vec3(nose)
        h = _as_vec3(heel)
        body_h = float(np.linalg.norm(n[:2] - h[:2]))
        if not np.isfinite(body_h) or body_h <= 1e-6:
            return None
        return body_h
    except Exception:  # noqa: BLE001
        return None


def calculate_3d_joint_angle(p1, p2, p3, *, is_knee_extension: bool = False) -> float:
    """基于 3D 向量点乘的真实关节夹角（度）；禁止 2D arctan2。

    ``is_knee_extension=True``（触球/impact 伸展语境）：矢状面射门若点乘得到
    ``angle < 130``，视为取成了锐角/半伸展假象，强制补角 ``180 - angle``，
    使摆动腿膝角落在正常 140°–160° 伸展带而非满屏爆红的 50° 类内角反转。
    后摆折叠角解算请保持默认 ``False``（需要真实内角极小值）。
    """
    try:
        ba = _as_vec3(p1) - _as_vec3(p2)
        bc = _as_vec3(p3) - _as_vec3(p2)
        na = float(np.linalg.norm(ba))
        nb = float(np.linalg.norm(bc))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        if not (np.isfinite(na) and np.isfinite(nb)):
            return 0.0
        cos_v = float(np.dot(ba, bc) / (na * nb))
        if not np.isfinite(cos_v):
            return 0.0
        cos_v = float(np.clip(cos_v, -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cos_v)))
        # 触球瞬间伸展态：angle < 130 强制补角，避免锐角误报
        if bool(is_knee_extension) and angle < 130.0:
            angle = 180.0 - angle
        return float(angle)
    except Exception:  # noqa: BLE001
        return 0.0


def evaluate_ball_bbox_for_pcr(ball_pixel_bbox) -> dict[str, Any]:
    """评估 YOLO 足球框是否足以做 PCR 标定。

    不合格时 ``ok=False``，调用方不得把 PCR 结果标成 measured。
    """
    result: dict[str, Any] = {
        "ok": False,
        "width": 0.0,
        "height": 0.0,
        "diameter_px": 0.0,
        "aspect_ratio": 0.0,
        "pcr": float(DEFAULT_EMPIRICAL_PCR),
        "ball_center_x": None,
        "ball_center_y": None,
        "reason": "missing_bbox",
    }
    if ball_pixel_bbox is None:
        return result
    try:
        bbox = np.asarray(ball_pixel_bbox, dtype=np.float64).reshape(-1)
        if bbox.size < 4 or not bool(np.all(np.isfinite(bbox[:4]))):
            result["reason"] = "invalid_bbox"
            return result
        x_min, y_min, x_max, y_max = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        width = abs(x_max - x_min)
        height = abs(y_max - y_min)
        diameter = max(width, height)
        short = min(width, height)
        aspect = float(diameter / short) if short > 1e-9 else float("inf")
        center_x = 0.5 * (x_min + x_max)
        center_y = 0.5 * (y_min + y_max)
        result.update(
            {
                "width": float(width),
                "height": float(height),
                "diameter_px": float(diameter),
                "aspect_ratio": float(aspect) if np.isfinite(aspect) else 0.0,
                "ball_center_x": float(center_x),
                "ball_center_y": float(center_y),
            }
        )
        if diameter < float(BALL_BBOX_MIN_DIAMETER_PX):
            result["reason"] = "diameter_too_small"
            return result
        if not np.isfinite(aspect) or aspect > float(BALL_BBOX_MAX_ASPECT_RATIO):
            result["reason"] = "aspect_ratio_blur"
            return result
        pcr = float(STANDARD_BALL_DIAMETER_CM) / float(diameter)
        if not np.isfinite(pcr) or pcr <= 0.0:
            result["reason"] = "invalid_pcr"
            return result
        result["pcr"] = pcr
        result["ok"] = True
        result["reason"] = "ok"
        return result
    except Exception:  # noqa: BLE001
        result["reason"] = "exception"
        return result


def calculate_support_foot_offset_cm(
    ankle_pixel_coords,
    ball_pixel_bbox,
    body_h_px: Optional[float] = None,
) -> float:
    """支撑脚横距（厘米）：合格球框用 PCR；否则 fallback_PCR；结果钳制在 [0, 60] cm。"""
    detail = calculate_support_foot_offset_detailed(
        ankle_pixel_coords, ball_pixel_bbox, body_h_px=body_h_px
    )
    return float(detail.get("offset_cm") or 0.0)


def calculate_support_foot_offset_detailed(
    ankle_pixel_coords,
    ball_pixel_bbox,
    *,
    body_h_px: Optional[float] = None,
) -> dict[str, Any]:
    """PCR 横距详单：含 QA / ok，供 Scorer 写 provenance。

    - ``max(ball_w, ball_h) < 10``：严禁用球框算比例尺，改走 ``fallback_PCR``
      （``145 / body_h_px``，缺省经验 PCR）。
    - 最终 ``offset_cm = min(max(offset, 0), 60)``，防止横距爆炸。
    - 仅合格球框 PCR 时 ``ok=True``（AIGC measured）；fallback 仍给出数值但 ok=False。
    """
    qa = evaluate_ball_bbox_for_pcr(ball_pixel_bbox)
    out: dict[str, Any] = {
        "offset_cm": 0.0,
        "ok": False,
        "qa": qa,
        "method": "ball_pcr",
        "pcr": float(qa.get("pcr") or DEFAULT_EMPIRICAL_PCR),
        "fallback_pcr": None,
    }
    try:
        ankle = np.asarray(ankle_pixel_coords, dtype=np.float64).reshape(-1)
        if ankle.size < 1 or not np.isfinite(ankle[0]):
            out["qa"] = {**qa, "reason": qa.get("reason") or "invalid_ankle"}
            return out
        ankle_x = float(ankle[0])

        ball_center_x = qa.get("ball_center_x")
        if ball_center_x is None or not np.isfinite(float(ball_center_x)):
            # 完全无球心：无法定义横距
            return out

        ball_w = float(qa.get("width") or 0.0)
        ball_h = float(qa.get("height") or 0.0)
        ball_span = max(ball_w, ball_h)
        use_ball_pcr = bool(qa.get("ok")) and ball_span >= float(BALL_BBOX_MIN_DIAMETER_PX)

        if use_ball_pcr:
            pcr = float(qa["pcr"])
            method = "ball_pcr"
            measured_ok = True
            fallback_pcr = None
        else:
            # 遮挡/误识别：绝对不用坏球框比例尺
            fallback_pcr = float(estimate_fallback_pcr(body_h_px))
            pcr = fallback_pcr
            method = (
                "fallback_body_pcr"
                if body_h_px is not None and np.isfinite(float(body_h_px)) and float(body_h_px) > 1e-6
                else "fallback_empirical_pcr"
            )
            measured_ok = False

        offset = abs(ankle_x - float(ball_center_x)) * float(pcr)
        if not np.isfinite(offset):
            out["qa"] = {**qa, "ok": False, "reason": "non_finite_offset"}
            out["method"] = method
            out["pcr"] = float(pcr)
            out["fallback_pcr"] = fallback_pcr
            return out

        offset_cm = float(min(max(float(offset), 0.0), float(SUPPORT_FOOT_OFFSET_MAX_CM)))
        out["offset_cm"] = offset_cm
        out["ok"] = bool(measured_ok)
        out["method"] = method
        out["pcr"] = float(pcr)
        out["fallback_pcr"] = fallback_pcr
        if not measured_ok:
            out["qa"] = {
                **qa,
                "ok": False,
                "reason": qa.get("reason") or "fallback_pcr",
                "used_fallback_pcr": True,
            }
        return out
    except Exception:  # noqa: BLE001
        out["qa"] = {**qa, "ok": False, "reason": "exception"}
        return out


def crosscheck_pcr_vs_world_lateral(
    pcr_cm: float,
    world_lateral_cm: Optional[float],
    *,
    tol_cm: float = PCR_WORLD_CROSSCHECK_TOL_CM,
) -> dict[str, Any]:
    """PCR 与世界坐标横距交叉校验。无世界量时 ``skipped=True``。"""
    result = {
        "skipped": True,
        "agree": True,
        "delta_cm": None,
        "confidence_factor": 1.0,
        "world_lateral_cm": None,
    }
    if world_lateral_cm is None:
        return result
    try:
        world = float(world_lateral_cm)
        pcr = float(pcr_cm)
        if not (np.isfinite(world) and np.isfinite(pcr)):
            return result
        delta = abs(pcr - world)
        agree = delta <= float(tol_cm)
        # 偏差越大置信越低；超过 2*tol 时置信压到 0.35
        if agree:
            factor = 1.0
        else:
            factor = float(max(0.35, 1.0 - (delta - tol_cm) / max(tol_cm, 1e-9) * 0.4))
        return {
            "skipped": False,
            "agree": bool(agree),
            "delta_cm": round(float(delta), 3),
            "confidence_factor": round(factor, 3),
            "world_lateral_cm": round(world, 3),
        }
    except (TypeError, ValueError):
        return result


def ankle_half_window_frames(
    fps: float = DEFAULT_VIDEO_FPS,
    half_window_ms: float = ANKLE_IMPACT_HALF_WINDOW_MS,
) -> int:
    """由帧率与半窗毫秒换算帧数；至少 1 帧。

    使用向零截断：30fps×50ms → 1 帧（与旧 t±1 对齐）；60fps×50ms → 3 帧。
    """
    try:
        fps_v = float(fps) if np.isfinite(fps) and float(fps) > 1e-6 else DEFAULT_VIDEO_FPS
        half_ms = float(half_window_ms) if np.isfinite(half_window_ms) else ANKLE_IMPACT_HALF_WINDOW_MS
        return int(max(1, int(half_ms * fps_v / 1000.0)))
    except Exception:  # noqa: BLE001
        return 1


def slice_ankle_impact_window_bounds(
    n: int,
    t_impact_index: int,
    *,
    fps: float = DEFAULT_VIDEO_FPS,
    half_window_ms: float = ANKLE_IMPACT_HALF_WINDOW_MS,
    half_window_frames: Optional[int] = None,
) -> tuple[int, int, int]:
    """返回 ``(lo, hi, half)``，闭区间索引（含 hi）。"""
    if n <= 0:
        return 0, -1, 1
    half = (
        int(half_window_frames)
        if half_window_frames is not None
        else ankle_half_window_frames(fps, half_window_ms)
    )
    half = int(max(1, half))
    t = int(t_impact_index) if np.isfinite(t_impact_index) else 0
    t = int(max(0, min(n - 1, t)))
    lo = max(0, t - half)
    hi = min(n - 1, t + half)
    return lo, hi, half


def _interp_invalid_ankle_angles(
    values: list[float],
    valid: list[bool],
) -> list[float]:
    """对无效帧做线性插值；仅一端有邻域时用最近有效值；全无效返回空。"""
    n = len(values)
    if n == 0 or not any(valid):
        return []
    out = [float(v) for v in values]
    for i in range(n):
        if valid[i]:
            continue
        left = None
        right = None
        for j in range(i - 1, -1, -1):
            if valid[j]:
                left = j
                break
        for j in range(i + 1, n):
            if valid[j]:
                right = j
                break
        if left is not None and right is not None and right > left:
            t = (i - left) / float(right - left)
            out[i] = float(values[left] + t * (values[right] - values[left]))
        elif left is not None:
            out[i] = float(values[left])
        elif right is not None:
            out[i] = float(values[right])
        else:
            return []
    return out


def _ankle_window_validity(
    window: np.ndarray,
    visibility_window: Optional[np.ndarray],
    *,
    min_visibility: float,
) -> list[bool]:
    """可见度极低 / 非有限 / 空值跳变(≈0) → 无效，严禁直接进方差。"""
    valid: list[bool] = []
    for i, raw in enumerate(window.tolist()):
        try:
            v = float(raw)
        except (TypeError, ValueError):
            valid.append(False)
            continue
        if not np.isfinite(v):
            valid.append(False)
            continue
        # 空值/塌缩跳变：角度≈0 通常来自丢失关键点，不得计入
        if abs(v) < 1e-6:
            valid.append(False)
            continue
        if visibility_window is not None and i < int(visibility_window.size):
            try:
                vis = float(visibility_window[i])
            except (TypeError, ValueError):
                vis = 0.0
            if not np.isfinite(vis) or vis < float(min_visibility):
                valid.append(False)
                continue
        valid.append(True)
    return valid


def calculate_ankle_stiffness_variance(
    ankle_angles_time_series,
    t_impact_index,
    *,
    fps: float = DEFAULT_VIDEO_FPS,
    half_window_ms: float = ANKLE_IMPACT_HALF_WINDOW_MS,
    half_window_frames: Optional[int] = None,
    landmark_visibility_series=None,
    min_visibility: float = ANKLE_LANDMARK_VISIBILITY_MIN,
) -> tuple[float, str]:
    """踝关节刚度：冲击窗角方差 + 刚度状态。

    【Phase 2】窗长由帧率与毫秒半窗决定（默认 ~100ms）；
    30fps 时 half=1，行为与旧 t±1 三帧一致。

    【可见度门控】冲击窗内 visibility 极低或角度空值跳变的帧严禁计入方差；
    先线性插值，再用剩余有效点 ``np.var``，结果 ``round(..., 2)``。
    """
    try:
        series = np.asarray(ankle_angles_time_series, dtype=np.float64).reshape(-1)
        n = int(series.size)
        if n <= 0:
            return 0.0, ANKLE_STIFFNESS_LOCKED

        lo, hi, _half = slice_ankle_impact_window_bounds(
            n,
            t_impact_index,
            fps=fps,
            half_window_ms=half_window_ms,
            half_window_frames=half_window_frames,
        )
        if hi < lo:
            return 0.0, ANKLE_STIFFNESS_LOCKED
        window = series[lo : hi + 1]

        vis_window = None
        if landmark_visibility_series is not None:
            vis_arr = np.asarray(landmark_visibility_series, dtype=np.float64).reshape(-1)
            if int(vis_arr.size) == n:
                vis_window = vis_arr[lo : hi + 1]
            elif int(vis_arr.size) == int(window.size):
                vis_window = vis_arr

        valid = _ankle_window_validity(
            window, vis_window, min_visibility=float(min_visibility)
        )
        raw_vals = [float(v) for v in window.tolist()]
        cleaned = _interp_invalid_ankle_angles(raw_vals, valid)
        if not cleaned:
            # 插值失败：仅用剩余有效帧差值/方差
            cleaned = [raw_vals[i] for i, ok in enumerate(valid) if ok]
        cleaned = [float(v) for v in cleaned if np.isfinite(float(v))]
        if not cleaned:
            return 0.0, ANKLE_STIFFNESS_LOCKED

        vals = list(cleaned)
        # 与旧语义对齐：至少用 3 点估计 var（短窗端点复制）
        while len(vals) < 3 and vals:
            vals.append(vals[-1])
        variance = float(np.var(np.asarray(vals, dtype=np.float64)))
        if not np.isfinite(variance):
            variance = 0.0
        variance = float(round(max(0.0, variance), 2))

        if variance < ANKLE_STIFFNESS_LOCKED_MAX_VAR:
            status = ANKLE_STIFFNESS_LOCKED
        elif variance <= ANKLE_STIFFNESS_SLIGHT_MAX_VAR:
            status = ANKLE_STIFFNESS_SLIGHT_DEFORMATION
        else:
            status = ANKLE_STIFFNESS_YIELDING
        return variance, status
    except Exception:  # noqa: BLE001
        return 0.0, ANKLE_STIFFNESS_LOCKED


def ankle_window_dorsiflex_drop_deg(ankle_angles_window) -> Optional[float]:
    """冲击窗背屈骤降幅度 = max - min；无效窗返回 None。"""
    try:
        vals = [float(v) for v in (ankle_angles_window or []) if np.isfinite(float(v))]
        if len(vals) < 2:
            return None
        return float(max(vals) - min(vals))
    except (TypeError, ValueError):
        return None


def infer_swing_leg_side(
    frames: list,
    t_impact: int,
    ball_center=None,
    *,
    explicit_side: Optional[str] = None,
) -> str:
    """推断摆动腿侧：``left`` / ``right``。

    优先级：显式字段 → 触球前踝位移更大侧 → 距球更近侧 → 默认 right（固定机位右脚惯例）。
    """
    side = str(explicit_side or "").strip().lower()
    if side in ("left", "l", "left_foot"):
        return "left"
    if side in ("right", "r", "right_foot"):
        return "right"

    if not frames:
        return "right"
    n = len(frames)
    t = int(max(0, min(n - 1, int(t_impact))))

    def _ankle_xz(rec: dict, which: str) -> Optional[np.ndarray]:
        key = "left_ankle" if which == "left" else "right_ankle"
        if rec.get(key) is None:
            return None
        try:
            v = _as_vec3(rec[key])
            return np.array([float(v[0]), float(v[2])], dtype=np.float64)
        except Exception:
            return None

    # 触球前短窗位移幅度
    t0 = max(0, t - 5)
    left_travel = 0.0
    right_travel = 0.0
    prev_l = _ankle_xz(frames[t0], "left")
    prev_r = _ankle_xz(frames[t0], "right")
    for i in range(t0 + 1, t + 1):
        cur_l = _ankle_xz(frames[i], "left")
        cur_r = _ankle_xz(frames[i], "right")
        if prev_l is not None and cur_l is not None:
            left_travel += float(np.linalg.norm(cur_l - prev_l))
        if prev_r is not None and cur_r is not None:
            right_travel += float(np.linalg.norm(cur_r - prev_r))
        if cur_l is not None:
            prev_l = cur_l
        if cur_r is not None:
            prev_r = cur_r

    if left_travel > right_travel * 1.15:
        return "left"
    if right_travel > left_travel * 1.15:
        return "right"

    # 距球更近侧为摆动腿
    if ball_center is not None:
        try:
            ball = _as_vec3(ball_center)
            ball_xz = np.array([float(ball[0]), float(ball[2])], dtype=np.float64)
            la = _ankle_xz(frames[t], "left")
            ra = _ankle_xz(frames[t], "right")
            if la is not None and ra is not None:
                dl = float(np.linalg.norm(la - ball_xz))
                dr = float(np.linalg.norm(ra - ball_xz))
                if dl + 1e-9 < dr:
                    return "left"
                if dr + 1e-9 < dl:
                    return "right"
        except Exception:
            pass

    return "right"


def swing_leg_joint_keys(side: str) -> tuple[str, str, str, str]:
    """返回摆动腿 (hip, knee, ankle, foot_index) 字段名。"""
    if str(side).lower().startswith("l"):
        return "left_hip", "left_knee", "left_ankle", "left_foot_index"
    return "right_hip", "right_knee", "right_ankle", "right_foot_index"
