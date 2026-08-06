# -*- coding: utf-8 -*-
"""biomech_primitives.py — V3.1/Phase2 生物力学实测原语（唯一权威实现）。

error_diagnoser / pose_tracker 均应委托本模块，禁止再复制 PCR / 3D 夹角 /
踝形变落差（Max Angular Deflection）算法。
本模块零依赖业务层，严禁 LLM / 启发式评分干预。
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------
# 物理常数与刚度分档
# --------------------------------------------------------------------------
STANDARD_BALL_DIAMETER_CM = 21.0
DEFAULT_EMPIRICAL_PCR = STANDARD_BALL_DIAMETER_CM / 84.0
AVERAGE_CHILD_HEIGHT_CM = 145.0
AVERAGE_CHILD_FOOT_LEN_CM = 22.0  # 基准脚长（cm），熔断降级时用于比例换算
AVERAGE_CHILD_SHOULDER_WIDTH_CM = 30.0  # 肩宽归一化参考尺（cm）
SUPPORT_FOOT_OFFSET_MAX_CM = 50.0
SUPPORT_FOOT_LATERAL_FUSE_CM = 45.0  # PCR 横距超过此值时触发 3D 脚长防线熔断
# 肩宽归一化比例阈值（黄金标准：约半个肩宽；废除 PCR 绝对厘米）
# GREEN 0.4–0.7；YELLOW 0.25–0.4 / 0.7–0.9；RED <0.25 / >0.9
SUPPORT_RATIO_GREEN_LOW = 0.40
SUPPORT_RATIO_GREEN_HIGH = 0.70
SUPPORT_RATIO_YELLOW_LOW = 0.25
SUPPORT_RATIO_YELLOW_HIGH = 0.90
SUPPORT_RATIO_IDEAL_CENTER = 0.55  # 绿带中心 ≈半个肩宽

ANKLE_STIFFNESS_LOCKED = "LOCKED"
ANKLE_STIFFNESS_SLIGHT_DEFORMATION = "SLIGHT_DEFORMATION"
ANKLE_STIFFNESS_YIELDING = "YIELDING"
# 【V3.9】最大形变落差角（Max Angular Deflection）分档 —— 取代方差 σ²
# GREEN:  < 10°（击球瞬间脚腕几乎未偏转）
# YELLOW: 10°–20°（轻微卸力）
# RED:    > 20°（严重松弛 → ERR_C1_LOOSE_ANKLE）
ANKLE_DEFLECTION_GREEN_MAX_DEG = 10.0
ANKLE_DEFLECTION_YELLOW_MAX_DEG = 20.0
ANKLE_DEFLECTION_HALF_WINDOW_FRAMES = 2  # T0 ± 2 → 共 5 帧
# 兼容旧名：数值语义已切换为 deflection_deg 阈值
ANKLE_STIFFNESS_LOCKED_MAX_VAR = ANKLE_DEFLECTION_GREEN_MAX_DEG
ANKLE_STIFFNESS_SLIGHT_MAX_VAR = ANKLE_DEFLECTION_YELLOW_MAX_DEG
ANKLE_LANDMARK_VISIBILITY_MIN = 0.5
# 通用关节点置信度门槛（遮挡 / 出框时 MediaPipe 可能回 None 或低 visibility）
LANDMARK_CONFIDENCE_MIN = 0.5
# 关节点坐标 EMA 平滑系数（越大越跟随新帧；0.35 兼顾抑抖与响应）
LANDMARK_EMA_ALPHA = 0.35
# 突发跳变防护：单帧位移超过该像素阈值时改用上一帧缓存
LANDMARK_JUMP_MAX_PX = 80.0

# 【Phase 2 / 横距防抖】YOLO 球框：max(w,h) < 10px 严禁参与比例尺
BALL_BBOX_MIN_DIAMETER_PX = 10.0
BALL_BBOX_MAX_ASPECT_RATIO = 2.5  # max(w,h)/min(w,h)
PCR_WORLD_CROSSCHECK_TOL_CM = 4.0

# 【Phase 2】踝冲击窗：默认总宽约 100ms（半窗 50ms）；形变落差改用固定 ±2 帧
ANKLE_IMPACT_HALF_WINDOW_MS = 50.0
DEFAULT_VIDEO_FPS = 30.0

# 【Phase 2】折叠角 ROI 最少有效帧
FOLD_ROI_MIN_VALID_FRAMES = 3


def _as_vec3(point) -> np.ndarray:
    if point is None:
        return np.full(3, np.nan, dtype=np.float64)
    try:
        arr = np.asarray(point, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return np.full(3, np.nan, dtype=np.float64)
    if arr.size >= 3:
        return arr[:3].astype(np.float64, copy=False)
    if arr.size == 2:
        return np.array([float(arr[0]), float(arr[1]), 0.0], dtype=np.float64)
    if arr.size == 1:
        return np.array([float(arr[0]), 0.0, 0.0], dtype=np.float64)
    return np.full(3, np.nan, dtype=np.float64)


def is_valid_joint_point(
    point,
    visibility: Optional[float] = None,
    *,
    min_visibility: float = LANDMARK_CONFIDENCE_MIN,
) -> bool:
    """判空防护：关节点存在、坐标有限，且（若提供）置信度 ≥ 门槛。"""
    if point is None:
        return False
    try:
        vec = _as_vec3(point)
    except Exception:  # noqa: BLE001
        return False
    if not bool(np.all(np.isfinite(vec))):
        return False
    if visibility is not None:
        try:
            vis = float(visibility)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(vis) or vis < float(min_visibility):
            return False
    return True


class LandmarkEMASmoother:
    """关节点坐标滑动窗口指数移动平均（EMA），抑制遮挡恢复时的突发跳变。

    缺失 / 低置信度帧：返回上一帧有效缓存（若无缓存则 None）。
    单帧跳变超过 ``jump_max_px``：拒绝采纳，继续用缓存。
    """

    def __init__(
        self,
        *,
        alpha: float = LANDMARK_EMA_ALPHA,
        jump_max_px: float = LANDMARK_JUMP_MAX_PX,
        min_visibility: float = LANDMARK_CONFIDENCE_MIN,
    ) -> None:
        self.alpha = float(min(1.0, max(1e-3, alpha)))
        self.jump_max_px = float(max(1.0, jump_max_px))
        self.min_visibility = float(min_visibility)
        self._state: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._state.clear()

    def update(
        self,
        name: str,
        point,
        visibility: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """更新并返回平滑后的 xyz；无效输入时回退上一帧缓存。"""
        key = str(name)
        if not is_valid_joint_point(
            point, visibility, min_visibility=self.min_visibility
        ):
            cached = self._state.get(key)
            return cached.copy() if cached is not None else None

        raw = _as_vec3(point)
        prev = self._state.get(key)
        if prev is None:
            self._state[key] = raw.copy()
            return raw.copy()

        # 突发跳变：平面位移过大则拒绝本帧，防 TypeError 上游噪声击穿
        jump = float(np.linalg.norm(raw[:2] - prev[:2]))
        if np.isfinite(jump) and jump > self.jump_max_px:
            return prev.copy()

        a = self.alpha
        smoothed = a * raw + (1.0 - a) * prev
        self._state[key] = smoothed
        return smoothed.copy()

    def get(self, name: str) -> Optional[np.ndarray]:
        cached = self._state.get(str(name))
        return cached.copy() if cached is not None else None


def gap_fill_scalar_series(
    values: Sequence[Optional[float]],
    *,
    max_gap: int = 2,
) -> list[Optional[float]]:
    """前后帧运动趋势补帧：单帧/短间隙缺失用线性插值填补，降低漏检率。

    ``max_gap`` 控制最大可补空洞长度（默认 2，对应丢 1~2 帧）。
    两端无邻域或空洞过长时保持 None，绝不编造长程轨迹。
    """
    n = len(values) if values is not None else 0
    if n == 0:
        return []
    out: list[Optional[float]] = []
    raw: list[Optional[float]] = []
    for v in values:
        try:
            if v is None:
                raw.append(None)
            else:
                fv = float(v)
                raw.append(fv if np.isfinite(fv) else None)
        except (TypeError, ValueError):
            raw.append(None)

    for i in range(n):
        if raw[i] is not None:
            out.append(raw[i])
            continue
        left = None
        right = None
        for j in range(i - 1, -1, -1):
            if raw[j] is not None:
                left = j
                break
        for j in range(i + 1, n):
            if raw[j] is not None:
                right = j
                break
        if left is None or right is None:
            out.append(None)
            continue
        gap = right - left
        if gap > int(max_gap) + 1:
            out.append(None)
            continue
        t = (i - left) / float(right - left)
        filled = float(raw[left]) + t * (float(raw[right]) - float(raw[left]))
        out.append(filled)
    return out


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


def calculate_2d_angle(p1, p2, p3) -> float:
    """纯 2D 关节夹角（度）：强制坍缩 Z，仅用图像/平面 ``(X, Y)``。

    MediaPipe 侧向踢球时 Z 深度估计畸变严重，会把后摆折叠角打出极端毛刺。
    本算子忽略 Z，以 p2 为顶点用 2D 点积：

        θ = arccos( (v1·v2) / (|v1|·|v2|) )，θ ∈ [0, 180]

    任一关节点无效 / 投影退化 → 返回 0.0。
    """
    try:
        if not (
            is_valid_joint_point(p1)
            and is_valid_joint_point(p2)
            and is_valid_joint_point(p3)
        ):
            return 0.0
        a = _as_vec3(p1)
        b = _as_vec3(p2)
        c = _as_vec3(p3)
        v1 = np.array([float(a[0] - b[0]), float(a[1] - b[1])], dtype=np.float64)
        v2 = np.array([float(c[0] - b[0]), float(c[1] - b[1])], dtype=np.float64)
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        if not (np.isfinite(n1) and np.isfinite(n2)):
            return 0.0
        cos_v = float(np.dot(v1, v2) / (n1 * n2))
        if not np.isfinite(cos_v):
            return 0.0
        cos_v = float(np.clip(cos_v, -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cos_v)))
        if not np.isfinite(angle):
            return 0.0
        return float(angle)
    except Exception:  # noqa: BLE001
        return 0.0


def calculate_2d_angle_or_none(p1, p2, p3) -> Optional[float]:
    """与 ``calculate_2d_angle`` 同源；关节点缺失/投影退化时返回 ``None``。"""
    if not (
        is_valid_joint_point(p1)
        and is_valid_joint_point(p2)
        and is_valid_joint_point(p3)
    ):
        return None
    a = _as_vec3(p1)
    b = _as_vec3(p2)
    c = _as_vec3(p3)
    n1 = float(np.hypot(float(a[0] - b[0]), float(a[1] - b[1])))
    n2 = float(np.hypot(float(c[0] - b[0]), float(c[1] - b[1])))
    if n1 < 1e-12 or n2 < 1e-12:
        return None
    angle = calculate_2d_angle(p1, p2, p3)
    if not np.isfinite(angle) or angle <= 0.0:
        return None
    return float(angle)


def log_2d_vs_sagittal_shift(p1, p2, p3, *, tag: str = "knee") -> Optional[float]:
    """对比 XY-2D 与旧 YZ-矢状角；若 |Δ|≥5° 打日志，供后期阈值微调。"""
    try:
        ang_2d = calculate_2d_angle_or_none(p1, p2, p3)
        ang_sag = calculate_sagittal_angle_or_none(p1, p2, p3)
        if ang_2d is None or ang_sag is None:
            return None
        delta = float(ang_2d) - float(ang_sag)
        if abs(delta) >= 5.0:
            print(
                f"【Biomech/2D投影】{tag}: 2d={ang_2d:.1f}° sagittal_yz={ang_sag:.1f}° "
                f"Δ={delta:+.1f}°（Z坍缩平移，供阈值微调）",
                flush=True,
            )
        return delta
    except Exception:  # noqa: BLE001
        return None


def calculate_sagittal_angle(p1, p2, p3, *, signed: bool = False) -> float:
    """【遗留】矢状面（Y-Z）髋/膝关节屈伸角（度）；atan2 锁定方向。

    侧向踢球时 MediaPipe Z 深度畸变严重，膝屈伸请改用 ``calculate_2d_angle``
    （强制坍缩 Z，仅用 X-Y）。

    ``signed=False``（默认）：返回内角 ``|θ| ∈ [0, 180]``，直腿≈180°，屈曲更小。
    ``signed=True``：返回 ``θ ∈ (-180, 180]``，符号标记屈伸侧。
    任一关节点无效 / 投影退化 → 返回 0.0。
    """
    try:
        if not (
            is_valid_joint_point(p1)
            and is_valid_joint_point(p2)
            and is_valid_joint_point(p3)
        ):
            return 0.0
        v1 = _as_vec3(p1) - _as_vec3(p2)  # p2 → p1（如膝→髋）
        v2 = _as_vec3(p3) - _as_vec3(p2)  # p2 → p3（如膝→踝）
        # 矢状面 Y-Z：丢弃左右横轴 X，抑制侧向抖动
        v1_yz = np.array([float(v1[1]), float(v1[2])], dtype=np.float64)
        v2_yz = np.array([float(v2[1]), float(v2[2])], dtype=np.float64)
        n1 = float(np.linalg.norm(v1_yz))
        n2 = float(np.linalg.norm(v2_yz))
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        if not (np.isfinite(n1) and np.isfinite(n2)):
            return 0.0
        # atan2(y, x) 原理：x=dot、y=2D cross，符号锁定旋转方向
        cross = float(v1_yz[0] * v2_yz[1] - v1_yz[1] * v2_yz[0])
        dot = float(v1_yz[0] * v2_yz[0] + v1_yz[1] * v2_yz[1])
        if not (np.isfinite(cross) and np.isfinite(dot)):
            return 0.0
        signed_deg = float(np.degrees(np.arctan2(cross, dot)))
        if not np.isfinite(signed_deg):
            return 0.0
        if bool(signed):
            return float(signed_deg)
        # 内角：直腿时 |±180|≈180，噪声只会在 ±180 分支附近抖动而不会跳到锐角
        return float(abs(signed_deg))
    except Exception:  # noqa: BLE001
        return 0.0


def calculate_sagittal_angle_or_none(
    p1, p2, p3, *, signed: bool = False
) -> Optional[float]:
    """与 ``calculate_sagittal_angle`` 同源；关节点缺失/投影退化时返回 ``None``。"""
    if not (
        is_valid_joint_point(p1)
        and is_valid_joint_point(p2)
        and is_valid_joint_point(p3)
    ):
        return None
    v1 = _as_vec3(p1) - _as_vec3(p2)
    v2 = _as_vec3(p3) - _as_vec3(p2)
    n1 = float(np.hypot(float(v1[1]), float(v1[2])))
    n2 = float(np.hypot(float(v2[1]), float(v2[2])))
    if n1 < 1e-12 or n2 < 1e-12:
        return None
    angle = calculate_sagittal_angle(p1, p2, p3, signed=signed)
    if not np.isfinite(angle):
        return None
    return float(angle)


def calculate_3d_joint_angle(p1, p2, p3, *, is_knee_extension: bool = False) -> float:
    """关节夹角（度）。

    ``is_knee_extension=True``（髋/膝屈伸、触球伸展语境）：走 ``calculate_2d_angle``
    （强制坍缩 Z，仅用 X-Y），消除侧向踢球时 MediaPipe 深度畸变毛刺。

    ``is_knee_extension=False``：保留纯 3D 点乘 ``arccos``（踝等非屈伸角）。
    防御性：任一关节点为 None / 非有限 → 返回 0.0；``cos`` 钳制到 ``[-1, 1]``。
    """
    if bool(is_knee_extension):
        return float(calculate_2d_angle(p1, p2, p3))
    try:
        if not (
            is_valid_joint_point(p1)
            and is_valid_joint_point(p2)
            and is_valid_joint_point(p3)
        ):
            return 0.0
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
        if not np.isfinite(angle):
            return 0.0
        return float(angle)
    except Exception:  # noqa: BLE001
        return 0.0


def calculate_3d_joint_angle_or_none(
    p1, p2, p3, *, is_knee_extension: bool = False
) -> Optional[float]:
    """与 ``calculate_3d_joint_angle`` 同源；关节点缺失/无效时返回 ``None``。"""
    if bool(is_knee_extension):
        return calculate_2d_angle_or_none(p1, p2, p3)
    if not (
        is_valid_joint_point(p1)
        and is_valid_joint_point(p2)
        and is_valid_joint_point(p3)
    ):
        return None
    ba = _as_vec3(p1) - _as_vec3(p2)
    bc = _as_vec3(p3) - _as_vec3(p2)
    if float(np.linalg.norm(ba)) < 1e-12 or float(np.linalg.norm(bc)) < 1e-12:
        return None
    angle = calculate_3d_joint_angle(p1, p2, p3, is_knee_extension=False)
    if not np.isfinite(angle):
        return None
    return float(angle)


def _apply_homography_xy(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """3×3 单应性：像素 (x,y) → 平面坐标（米）。"""
    vec = np.array([float(x), float(y), 1.0], dtype=np.float64)
    mapped = H @ vec
    w = float(mapped[2])
    if abs(w) < 1e-12:
        raise ValueError("homography_degenerate")
    return float(mapped[0] / w), float(mapped[1] / w)


def _trajectory_to_meters(
    ball_trajectory_px: Sequence,
    calibrator=None,
    *,
    pcr_cm_per_px: Optional[float] = None,
) -> tuple[list[tuple[float, float]], str, bool]:
    """将像素轨迹映射为米制平面坐标。

    返回 ``(points_m, scale_method, y_up)``：
        - ``y_up=True``：物理 Y 向上（单应性 / 场地标定）
        - ``y_up=False``：仍为图像 Y 向下（PCR 降级），仰角计算时取 ``−Δy``
    """
    pts_px: list[tuple[float, float]] = []
    for pt in ball_trajectory_px or []:
        if pt is None:
            continue
        try:
            arr = np.asarray(pt, dtype=np.float64).reshape(-1)
            if arr.size < 2 or not bool(np.all(np.isfinite(arr[:2]))):
                continue
            pts_px.append((float(arr[0]), float(arr[1])))
        except (TypeError, ValueError):
            continue

    if len(pts_px) < 2:
        return [], "insufficient_points", True

    # 1) 场地单应性 / 可调用标定器
    if calibrator is not None:
        try:
            if isinstance(calibrator, np.ndarray) and calibrator.shape == (3, 3):
                H = np.asarray(calibrator, dtype=np.float64)
                pts_m = [_apply_homography_xy(H, x, y) for x, y in pts_px]
                return pts_m, "homography", True
            if hasattr(calibrator, "pixel_to_meter") and callable(
                calibrator.pixel_to_meter
            ):
                pts_m = []
                for x, y in pts_px:
                    mx, my = calibrator.pixel_to_meter(x, y)
                    pts_m.append((float(mx), float(my)))
                return pts_m, "calibrator.pixel_to_meter", True
            if callable(calibrator):
                pts_m = []
                for x, y in pts_px:
                    out = calibrator(x, y)
                    pts_m.append((float(out[0]), float(out[1])))
                return pts_m, "calibrator_callable", True
        except Exception:  # noqa: BLE001
            pass  # 降级 PCR

    # 2) 默认 PCR：cm/px → m/px
    pcr = (
        float(pcr_cm_per_px)
        if pcr_cm_per_px is not None and np.isfinite(float(pcr_cm_per_px)) and float(pcr_cm_per_px) > 0
        else float(DEFAULT_EMPIRICAL_PCR)
    )
    m_per_px = pcr / 100.0
    pts_m = [(x * m_per_px, y * m_per_px) for x, y in pts_px]
    return pts_m, "default_pcr", False


def calculate_ball_outcome(
    ball_trajectory_px,
    fps: float = DEFAULT_VIDEO_FPS,
    calibrator=None,
    *,
    pcr_cm_per_px: Optional[float] = None,
) -> dict[str, Any]:
    """出球初速度与发射仰角（射门结果闭环）。

    参数:
        ball_trajectory_px: ``T0`` 及其后 3 帧（共 4 点）的 YOLO 球心像素序列，
            元素为 ``(x, y)`` / ``[x, y]``；允许夹杂 ``None``（将被跳过）。
        fps: 采样帧率（默认 30）。
        calibrator: 可选场地标定——``3×3`` 单应性矩阵、
            ``pixel_to_meter(x, y)`` 对象、或 ``(x, y) -> (X_m, Y_m)`` 可调用对象。
            缺失时降级为默认球径 PCR（``DEFAULT_EMPIRICAL_PCR`` cm/px）。
        pcr_cm_per_px: 可选覆盖 PCR（例如 T0 帧实测球框直径推得）。

    返回:
        ``{ok, ball_speed_kmh, launch_angle_deg, ball_speed_mps, ...}``；
        点数不足 / 脏数据时 ``ok=False``，速度与仰角为 ``None``（不抛异常）。

    公式:
        - ``V0_mps = 物理路径总位移 / ((N-1) / fps)``，再 ×3.6 → km/h
        - ``launch_angle_deg = atan2(ΔY_up, ΔX)``，水平地面为 ``0°``
    """
    out: dict[str, Any] = {
        "ok": False,
        "ball_speed_kmh": None,
        "ball_speed_mps": None,
        "launch_angle_deg": None,
        "displacement_m": None,
        "dt_sec": None,
        "sample_count": 0,
        "scale_method": None,
        "reason": "init",
    }
    try:
        fps_v = float(fps) if fps is not None and np.isfinite(float(fps)) and float(fps) > 1e-6 else DEFAULT_VIDEO_FPS
        pts_m, scale_method, y_up = _trajectory_to_meters(
            ball_trajectory_px, calibrator, pcr_cm_per_px=pcr_cm_per_px
        )
        out["scale_method"] = scale_method
        out["sample_count"] = int(len(pts_m))
        if len(pts_m) < 2:
            out["reason"] = "insufficient_points"
            return out

        # 路径总位移（米）
        path_m = 0.0
        for i in range(1, len(pts_m)):
            dx = pts_m[i][0] - pts_m[i - 1][0]
            dy = pts_m[i][1] - pts_m[i - 1][1]
            seg = math.hypot(dx, dy)
            if np.isfinite(seg):
                path_m += float(seg)

        n_intervals = len(pts_m) - 1
        dt = float(n_intervals) / float(fps_v)
        if dt <= 1e-12 or not np.isfinite(path_m) or path_m < 0.0:
            out["reason"] = "invalid_dt_or_displacement"
            return out

        v_mps = float(path_m / dt)
        v_kmh = float(v_mps * 3.6)

        # 仰角：起终点相对水平；图像 Y 向下时取 −Δy 作为向上
        dx_net = float(pts_m[-1][0] - pts_m[0][0])
        dy_net = float(pts_m[-1][1] - pts_m[0][1])
        dy_up = float(dy_net) if y_up else float(-dy_net)
        launch_deg = float(math.degrees(math.atan2(dy_up, dx_net)))

        if not (np.isfinite(v_kmh) and np.isfinite(launch_deg)):
            out["reason"] = "non_finite_outcome"
            return out

        out.update(
            {
                "ok": True,
                "ball_speed_kmh": round(v_kmh, 2),
                "ball_speed_mps": round(v_mps, 3),
                "launch_angle_deg": round(launch_deg, 2),
                "displacement_m": round(float(path_m), 4),
                "dt_sec": round(dt, 4),
                "reason": "ok",
            }
        )
        return out
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"exception:{exc}"
        return out


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


def calculate_support_ratio(
    landmarks,
    ball_center_px,
    *,
    support_ankle_key: Optional[str] = None,
) -> dict[str, Any]:
    """肩宽归一化支撑脚横距比例（权威算子；废除 PCR 绝对厘米）。

    从 ``landmarks`` 提取左右肩 ``(X, Y)`` → 欧氏距离 ``shoulder_width_px``；
    支撑脚踝与 ``ball_center_px`` 的水平（X 轴）像素绝对距离 ``support_dist_px``；
    返回无量纲系数 ``support_ratio = support_dist_px / shoulder_width_px``。
    """
    out: dict[str, Any] = {
        "ok": False,
        "support_ratio": None,
        "support_dist_px": None,
        "shoulder_width_px": None,
        "method": "shoulder_width_ratio",
        "reason": "missing_landmarks",
    }
    if not isinstance(landmarks, dict):
        return out
    try:
        ls = landmarks.get("left_shoulder")
        rs = landmarks.get("right_shoulder")
        if not (is_valid_joint_point(ls) and is_valid_joint_point(rs)):
            out["reason"] = "missing_shoulders"
            return out
        ls_v = _as_vec3(ls)
        rs_v = _as_vec3(rs)
        shoulder_width_px = float(np.hypot(float(rs_v[0] - ls_v[0]), float(rs_v[1] - ls_v[1])))
        if not np.isfinite(shoulder_width_px) or shoulder_width_px < 1e-6:
            out["reason"] = "degenerate_shoulder_width"
            return out

        ankle_key = str(support_ankle_key or "").strip() or None
        if ankle_key is None:
            # 默认右脚踢球 → 左踝为支撑脚；显式字段优先
            if landmarks.get("left_ankle") is not None:
                ankle_key = "left_ankle"
            elif landmarks.get("right_ankle") is not None:
                ankle_key = "right_ankle"
            else:
                out["reason"] = "missing_support_ankle"
                return out
        ankle = landmarks.get(ankle_key) or landmarks.get(f"{ankle_key}_px")
        if not is_valid_joint_point(ankle):
            out["reason"] = "invalid_support_ankle"
            return out
        ankle_v = _as_vec3(ankle)

        if ball_center_px is None:
            out["reason"] = "missing_ball_center"
            return out
        ball = _as_vec3(ball_center_px)
        if not np.isfinite(ball[0]):
            out["reason"] = "invalid_ball_center"
            return out

        support_dist_px = float(abs(float(ankle_v[0]) - float(ball[0])))
        if not np.isfinite(support_dist_px):
            out["reason"] = "non_finite_support_dist"
            return out

        ratio = float(support_dist_px / shoulder_width_px)
        if not np.isfinite(ratio) or ratio < 0.0:
            out["reason"] = "invalid_ratio"
            return out

        # 软钳制：避免遮挡噪声把比例冲到非物理上界（相对约 1.7×肩宽）
        ratio_max = float(SUPPORT_FOOT_OFFSET_MAX_CM) / max(
            float(AVERAGE_CHILD_SHOULDER_WIDTH_CM), 1e-6
        )
        ratio_clamped = bool(ratio > ratio_max)
        ratio = float(min(ratio, ratio_max))
        out.update(
            {
                "ok": True,
                "support_ratio": round(ratio, 4),
                "support_dist_px": round(support_dist_px, 3),
                "shoulder_width_px": round(shoulder_width_px, 3),
                "support_ankle_key": ankle_key,
                "ratio_clamped": ratio_clamped,
                "reason": "ok",
            }
        )
        return out
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"exception:{exc}"
        return out


def calculate_support_foot_offset_cm(
    ankle_pixel_coords,
    ball_pixel_bbox,
    body_h_px: Optional[float] = None,
    world_lateral_cm: Optional[float] = None,
    world_foot_len_m: Optional[float] = None,
) -> float:
    """【遗留】PCR 横距（厘米）。站位评分请改用 ``calculate_support_ratio``。"""
    detail = calculate_support_foot_offset_detailed(
        ankle_pixel_coords,
        ball_pixel_bbox,
        body_h_px=body_h_px,
        world_lateral_cm=world_lateral_cm,
        world_foot_len_m=world_foot_len_m,
    )
    return float(detail.get("offset_cm") or 0.0)


def _horizontal_plane_distance(
    a,
    b,
    *,
    plane: str = "xz",
) -> float:
    """两点在水平面的欧氏距离（world 用 X-Z；图像近似用 X-Y）。"""
    va = _as_vec3(a)
    vb = _as_vec3(b)
    if not (np.all(np.isfinite(va)) and np.all(np.isfinite(vb))):
        return float("nan")
    if plane == "xy":
        d0 = float(va[0] - vb[0])
        d1 = float(va[1] - vb[1])
    else:
        # MediaPipe world：Y 为垂直轴，水平面为 X-Z
        d0 = float(va[0] - vb[0])
        d1 = float(va[2] - vb[2])
    return float(np.hypot(d0, d1))


def _horizontal_delta(
    a,
    b,
    *,
    plane: str = "xz",
) -> tuple[float, float]:
    """水平面位移 (d0, d1)；world=X-Z，image=X-Y。"""
    va = _as_vec3(a)
    vb = _as_vec3(b)
    if not (np.all(np.isfinite(va)) and np.all(np.isfinite(vb))):
        return float("nan"), float("nan")
    if plane == "xy":
        return float(va[0] - vb[0]), float(va[1] - vb[1])
    return float(va[0] - vb[0]), float(va[2] - vb[2])


def _lateral_distance_along_shoulder(
    support_ankle,
    ball_center,
    left_shoulder,
    right_shoulder,
    *,
    plane: str = "xz",
) -> tuple[float, float, float]:
    """支撑脚相对球的「体侧向横距」（沿肩宽轴投影），而非全水平斜距。

    无球检测时球常被摆动脚尖代理；若用 ||ankle−foot||_horizontal，会把助跑
    前后步幅一并算进「支撑脚偏宽」，出现 2×肩宽 / 65cm 的夸张值。
    正确口径：只取相对肩宽方向（体侧）的分量。

    Returns:
        (lateral_raw, horizontal_raw, shoulder_width)
    """
    d0, d1 = _horizontal_delta(support_ankle, ball_center, plane=plane)
    horizontal = float(np.hypot(d0, d1))
    ls = _as_vec3(left_shoulder)
    rs = _as_vec3(right_shoulder)
    if plane == "xy":
        shoulder_h = np.array(
            [float(rs[0] - ls[0]), float(rs[1] - ls[1])], dtype=np.float64
        )
    else:
        shoulder_h = np.array(
            [float(rs[0] - ls[0]), float(rs[2] - ls[2])], dtype=np.float64
        )
    shoulder_width = float(np.linalg.norm(shoulder_h))
    if not np.isfinite(shoulder_width) or shoulder_width < 1e-6:
        # 肩宽退化：退回 |主轴分量|（world 偏 X，image 偏 X）
        lateral = abs(d0) if np.isfinite(d0) else float("nan")
        return lateral, horizontal, shoulder_width
    shoulder_unit = shoulder_h / shoulder_width
    delta = np.array([d0, d1], dtype=np.float64)
    lateral = float(abs(np.dot(delta, shoulder_unit)))
    return lateral, horizontal, shoulder_width


def calculate_support_offset_by_shoulder_ratio(
    support_ankle,
    ball_center,
    left_shoulder,
    right_shoulder,
    *,
    ref_shoulder_cm: float = AVERAGE_CHILD_SHOULDER_WIDTH_CM,
    coord_space: str = "world_m",
    distance_mode: str = "lateral",
) -> dict[str, Any]:
    """肩宽归一化支撑横距：消除 MediaPipe world 绝对尺度漂移。

    算法（默认 ``distance_mode="lateral"``）：
      raw_distance = |(ankle − ball) · shoulder_unit|   # 体侧向横距，不含前后步幅
      shoulder_width = ||right_shoulder − left_shoulder||_horizontal
      support_ratio = raw_distance / shoulder_width
      distance_cm_estimate = support_ratio × ref_shoulder_cm

    ``distance_mode="horizontal"`` 保留旧的全水平欧氏距离（仅调试/对比）。

    【禁止】对 world 坐标做裸 ``×100`` 当真实厘米。
    ``coord_space``：``world_m`` 用 X-Z 水平面；``image_px`` 用 X-Y 近似。
    """
    out: dict[str, Any] = {
        "ok": False,
        "support_ratio": None,
        "distance_cm_estimate": None,
        "raw_distance": None,
        "horizontal_distance": None,
        "shoulder_width": None,
        "ref_shoulder_cm": float(ref_shoulder_cm),
        "method": "shoulder_width_ratio",
        "coord_space": str(coord_space),
        "distance_mode": str(distance_mode or "lateral"),
        "scale_factor_to_cm": None,
    }
    try:
        plane = "xy" if str(coord_space).startswith("image") else "xz"
        lateral, horizontal, shoulder_width = _lateral_distance_along_shoulder(
            support_ankle,
            ball_center,
            left_shoulder,
            right_shoulder,
            plane=plane,
        )
        mode = str(distance_mode or "lateral").strip().lower()
        raw = horizontal if mode == "horizontal" else lateral
        if not (
            np.isfinite(raw)
            and np.isfinite(horizontal)
            and np.isfinite(shoulder_width)
            and shoulder_width >= 1e-6
        ):
            out["reason"] = "invalid_landmarks"
            return out
        ratio = float(raw / shoulder_width)
        if not np.isfinite(ratio) or ratio < 0.0:
            out["reason"] = "invalid_ratio"
            return out
        ref = float(ref_shoulder_cm) if ref_shoulder_cm and ref_shoulder_cm > 0 else float(
            AVERAGE_CHILD_SHOULDER_WIDTH_CM
        )
        # 比例与估计 cm 同步软钳制，避免脚尖代理把步幅算进站位后冲出物理上界
        ratio_max = float(SUPPORT_FOOT_OFFSET_MAX_CM) / max(ref, 1e-6)
        ratio_clamped = bool(ratio > ratio_max)
        ratio = float(min(ratio, ratio_max))
        est_cm = float(max(0.0, min(ratio * ref, float(SUPPORT_FOOT_OFFSET_MAX_CM))))
        scale = float(ref / shoulder_width)  # world/image 原始单位 → 归一化 cm
        out.update(
            {
                "ok": True,
                "support_ratio": round(ratio, 4),
                "distance_cm_estimate": round(est_cm, 2),
                "raw_distance": round(float(raw), 6),
                "horizontal_distance": round(float(horizontal), 6),
                "lateral_distance": round(float(lateral), 6),
                "shoulder_width": round(shoulder_width, 6),
                "ref_shoulder_cm": ref,
                "scale_factor_to_cm": round(scale, 6),
                "ratio_clamped": ratio_clamped,
                "reason": "ok",
            }
        )
        return out
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"exception:{exc}"
        return out


def world_delta_to_shoulder_normalized_cm(
    delta_xy,
    shoulder_width: float,
    *,
    ref_shoulder_cm: float = AVERAGE_CHILD_SHOULDER_WIDTH_CM,
) -> tuple[float, float]:
    """把 world/image 平面位移向量按肩宽归一化为等效厘米。"""
    try:
        arr = np.asarray(delta_xy, dtype=np.float64).reshape(-1)
        dx = float(arr[0]) if arr.size >= 1 else 0.0
        dy = float(arr[1]) if arr.size >= 2 else 0.0
    except (TypeError, ValueError):
        return 0.0, 0.0
    sw = float(shoulder_width)
    if not np.isfinite(sw) or sw < 1e-9:
        # 无肩宽时禁止假装绝对米制；退回 0
        return 0.0, 0.0
    ref = float(ref_shoulder_cm) if ref_shoulder_cm and ref_shoulder_cm > 0 else float(
        AVERAGE_CHILD_SHOULDER_WIDTH_CM
    )
    scale = ref / sw
    return float(dx * scale), float(dy * scale)


def calculate_support_foot_offset_detailed(
    ankle_pixel_coords,
    ball_pixel_bbox,
    *,
    body_h_px: Optional[float] = None,
    world_lateral_cm: Optional[float] = None,
    world_foot_len_m: Optional[float] = None,
) -> dict[str, Any]:
    """PCR 横距详单：含 QA / ok，供 Scorer 写 provenance。

    - ``max(ball_w, ball_h) < 10``：严禁用球框算比例尺，改走 ``fallback_PCR``。
    - 熔断逻辑：PCR 结果 > 45 cm 且有效 world_lateral_cm / world_foot_len_m 时，
      降级为 ``(world_lateral_cm / foot_len_cm) * AVERAGE_CHILD_FOOT_LEN_CM``，
      method 写 "world_foot_ratio_fuse"。
    - 极值截断：最终 ``offset_cm`` 钳制在 [0, 50] cm。
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

        calc_dist = float(offset)

        # 熔断：PCR 结果爆炸时降级到 3D 脚长比例换算
        if calc_dist > float(SUPPORT_FOOT_LATERAL_FUSE_CM):
            wl = world_lateral_cm if world_lateral_cm is not None else None
            wf = world_foot_len_m if world_foot_len_m is not None else None
            if (
                wl is not None
                and wf is not None
                and np.isfinite(float(wl))
                and np.isfinite(float(wf))
                and float(wf) > 1e-9
            ):
                foot_len_cm = float(wf) * 100.0
                fused = (float(wl) / foot_len_cm) * float(AVERAGE_CHILD_FOOT_LEN_CM)
                calc_dist = float(fused)
                method = "world_foot_ratio_fuse"
                measured_ok = False

        offset_cm = float(max(0.0, min(calc_dist, float(SUPPORT_FOOT_OFFSET_MAX_CM))))
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
    """可见度极低 / 非有限 / 空值跳变(≈0) → 无效，严禁直接进形变落差。"""
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


def _median_filter_kernel3(values: Sequence[float]) -> list[float]:
    """滑动中值滤波（kernel_size=3），剔除单帧飞点。

    优先 ``scipy.signal.medfilt``（边缘复制填充，避免默认零填充拉歪边界）；
    scipy 不可用时手写窗口=3 的一维中值。空输入 → ``[]``。
    """
    try:
        if values is None:
            return []
        arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
        n = int(arr.size)
        if n <= 0:
            return []
        if n < 3:
            return [float(x) for x in arr.tolist()]
        try:
            from scipy.signal import medfilt

            # medfilt 默认零填充会把边缘角拉向 0°；先 edge-pad 再截回
            padded = np.pad(arr, (1, 1), mode="edge")
            filtered = medfilt(padded, kernel_size=3)
            return [float(x) for x in np.asarray(filtered[1:-1], dtype=np.float64).tolist()]
        except Exception:  # noqa: BLE001
            out: list[float] = [0.0] * n
            raw = [float(x) for x in arr.tolist()]
            for i in range(n):
                lo = max(0, i - 1)
                hi = min(n - 1, i + 1)
                out[i] = float(np.median(raw[lo : hi + 1]))
            return out
    except Exception:  # noqa: BLE001
        try:
            return [float(x) for x in list(values)]
        except Exception:  # noqa: BLE001
            return []


def _savgol_smooth_ankle_series(values: Sequence[float]) -> list[float]:
    """对踝角全序列做 Savitzky-Golay 平滑；失败时降级中值滤波。"""
    try:
        arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
        n = int(arr.size)
        if n <= 0:
            return []
        if n < 5:
            return _median_filter_kernel3(arr.tolist())
        try:
            from scipy.signal import savgol_filter

            window = 5 if n >= 5 else (n if n % 2 == 1 else n - 1)
            window = int(max(3, window))
            if window % 2 == 0:
                window -= 1
            if window >= n:
                window = n if n % 2 == 1 else n - 1
            if window < 3:
                return [float(x) for x in arr.tolist()]
            poly = 2 if window > 2 else 1
            smoothed = savgol_filter(arr, window_length=window, polyorder=poly, mode="interp")
            return [float(x) for x in np.asarray(smoothed, dtype=np.float64).tolist()]
        except Exception:  # noqa: BLE001
            return _median_filter_kernel3(arr.tolist())
    except Exception:  # noqa: BLE001
        try:
            return [float(x) for x in list(values)]
        except Exception:  # noqa: BLE001
            return []


def _classify_ankle_deflection(deflection_deg: float) -> str:
    """按形变落差角分档：LOCKED / SLIGHT_DEFORMATION / YIELDING。"""
    d = float(deflection_deg)
    if not np.isfinite(d):
        return ANKLE_STIFFNESS_LOCKED
    if d < float(ANKLE_DEFLECTION_GREEN_MAX_DEG):
        return ANKLE_STIFFNESS_LOCKED
    if d <= float(ANKLE_DEFLECTION_YELLOW_MAX_DEG):
        return ANKLE_STIFFNESS_SLIGHT_DEFORMATION
    return ANKLE_STIFFNESS_YIELDING


def calculate_ankle_deflection(
    ankle_angles_sequence,
    t_impact_idx,
    *,
    half_window_frames: int = ANKLE_DEFLECTION_HALF_WINDOW_FRAMES,
    landmark_visibility_series=None,
    min_visibility: float = ANKLE_LANDMARK_VISIBILITY_MIN,
    already_smoothed: bool = False,
) -> tuple[float, str]:
    """踝关节刚性：最大形变落差角（Max Angular Deflection）。

    在（可选）Savitzky-Golay 平滑后的踝角序列上，截取 ``T0 ± half_window_frames``
    （默认前后 2 帧，共 5 帧）时间窗，计算：

        deflection_deg = max(window) − min(window)

    分档：
        - LOCKED（GREEN）：``deflection_deg < 10°``
        - SLIGHT_DEFORMATION（YELLOW）：``10° ≤ deflection_deg ≤ 20°``
        - YIELDING（RED）：``deflection_deg > 20°``

    空序列 / 脏数据 → ``(0.0, LOCKED)``；结果 ``round(..., 2)``。
    相对旧 ``np.var`` 指标：形变落差有物理自然上限，对 30fps 动态模糊更稳健。
    """
    try:
        if ankle_angles_sequence is None:
            return 0.0, ANKLE_STIFFNESS_LOCKED
        series = np.asarray(ankle_angles_sequence, dtype=np.float64).reshape(-1)
        n = int(series.size)
        if n <= 0:
            return 0.0, ANKLE_STIFFNESS_LOCKED

        half = int(half_window_frames) if half_window_frames is not None else (
            ANKLE_DEFLECTION_HALF_WINDOW_FRAMES
        )
        half = int(max(1, half))
        t = int(t_impact_idx) if np.isfinite(t_impact_idx) else 0
        t = int(max(0, min(n - 1, t)))
        lo = max(0, t - half)
        hi = min(n - 1, t + half)
        if hi < lo:
            return 0.0, ANKLE_STIFFNESS_LOCKED

        # 可见度门控 + 插值清洗（全序列），再 SG 平滑，最后截窗求落差
        vis_arr = None
        if landmark_visibility_series is not None:
            vis_arr = np.asarray(landmark_visibility_series, dtype=np.float64).reshape(-1)
            if int(vis_arr.size) != n:
                vis_arr = None

        valid = _ankle_window_validity(series, vis_arr, min_visibility=float(min_visibility))
        raw_vals = [float(v) for v in series.tolist()]
        cleaned = _interp_invalid_ankle_angles(raw_vals, valid)
        if not cleaned:
            cleaned = [raw_vals[i] for i, ok in enumerate(valid) if ok]
        cleaned = [float(v) for v in cleaned if np.isfinite(float(v))]
        if not cleaned:
            return 0.0, ANKLE_STIFFNESS_LOCKED

        # 插值后长度可能因全窗剔除而变短；若长度仍等于 n，按全序列平滑后切片
        if len(cleaned) == n and not already_smoothed:
            smoothed = _savgol_smooth_ankle_series(cleaned)
        elif len(cleaned) == n and already_smoothed:
            smoothed = list(cleaned)
        else:
            # 有效点被压缩：对清洗后短序列平滑，落差直接取全域 max-min
            smoothed = (
                list(cleaned)
                if already_smoothed
                else _savgol_smooth_ankle_series(cleaned)
            )
            vals = [float(v) for v in smoothed if np.isfinite(float(v))]
            if not vals:
                return 0.0, ANKLE_STIFFNESS_LOCKED
            deflection = float(max(vals) - min(vals))
            if not np.isfinite(deflection):
                deflection = 0.0
            deflection = float(round(max(0.0, deflection), 2))
            return deflection, _classify_ankle_deflection(deflection)

        if not smoothed or len(smoothed) != n:
            smoothed = _median_filter_kernel3(cleaned) if len(cleaned) == n else list(cleaned)
        if not smoothed:
            return 0.0, ANKLE_STIFFNESS_LOCKED

        vals = [
            float(v)
            for v in smoothed[lo : hi + 1]
            if np.isfinite(float(v))
        ]
        if not vals:
            return 0.0, ANKLE_STIFFNESS_LOCKED

        deflection = float(max(vals) - min(vals))
        if not np.isfinite(deflection):
            deflection = 0.0
        deflection = float(round(max(0.0, deflection), 2))
        return deflection, _classify_ankle_deflection(deflection)
    except Exception:  # noqa: BLE001
        return 0.0, ANKLE_STIFFNESS_LOCKED


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
    """【已废弃 API 名】兼容入口 → :func:`calculate_ankle_deflection`。

    返回值第一项现为 ``deflection_deg``（最大形变落差角，单位 °），
    **不再**是 ``np.var`` 方差。``fps`` / ``half_window_ms`` 被忽略；
    默认固定 ``T0 ± 2`` 帧。若显式传入 ``half_window_frames`` 则尊重之。
    """
    del fps, half_window_ms  # 形变落差改用固定帧窗，保留签名兼容
    half = (
        int(half_window_frames)
        if half_window_frames is not None
        else int(ANKLE_DEFLECTION_HALF_WINDOW_FRAMES)
    )
    return calculate_ankle_deflection(
        ankle_angles_time_series,
        t_impact_index,
        half_window_frames=half,
        landmark_visibility_series=landmark_visibility_series,
        min_visibility=min_visibility,
    )


def ankle_window_dorsiflex_drop_deg(ankle_angles_window) -> Optional[float]:
    """冲击窗角度落差 = max - min（与 deflection 同构）；无效窗返回 None。"""
    try:
        vals = [float(v) for v in (ankle_angles_window or []) if np.isfinite(float(v))]
        if len(vals) < 2:
            return None
        return float(max(vals) - min(vals))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# 【V3.10】躯干前倾 / 后仰角（纯 2D 投影）——近端核心稳定性
# --------------------------------------------------------------------------
# GREEN:  0° ~ +15°（微微前倾压住重心）
# YELLOW: -10° ~ 0°（过于直立）或 +15° ~ +25°（前倾偏大）
# RED:    < -10°（明显后仰）或 > +25°（过度折腰）
TRUNK_LEAN_GREEN_LOW_DEG = 0.0
TRUNK_LEAN_GREEN_HIGH_DEG = 15.0
TRUNK_LEAN_YELLOW_BACK_DEG = -10.0
TRUNK_LEAN_RED_FORWARD_DEG = 25.0


def _pick_landmark_xy(landmarks: dict, key: str) -> Optional[np.ndarray]:
    """从 landmarks / landmarks['world'] 取 2D (x, y)；无效返回 None。"""
    if not isinstance(landmarks, dict):
        return None
    src = landmarks.get(key)
    if src is None and isinstance(landmarks.get("world"), dict):
        src = landmarks["world"].get(key)
    if src is None:
        # 兼容像素别名
        src = landmarks.get(f"{key}_px")
    if not is_valid_joint_point(src):
        return None
    v = _as_vec3(src)
    xy = np.array([float(v[0]), float(v[1])], dtype=np.float64)
    if not np.all(np.isfinite(xy)):
        return None
    return xy


def calculate_trunk_lean(
    landmarks,
    *,
    ball_center=None,
) -> Optional[float]:
    """纯 2D 躯干倾角（度）：mid_hip → mid_shoulder 相对竖直向上的有符号夹角。

    算法：
        1) mid_shoulder = mean(LEFT_SHOULDER, RIGHT_SHOULDER)
        2) mid_hip = mean(LEFT_HIP, RIGHT_HIP)
        3) trunk = mid_shoulder − mid_hip（图像平面 XY，强制丢弃 Z）
        4) 相对图像竖直向上 ``(0, -1)`` 的有符号角：
               lean = atan2(dx, −dy)
           其中图像 Y 向下为正；``−dy`` 为向上分量。

    符号约定（侧向机位，前向为出球方向）：
        - **正值** = 身体前倾（压重心）
        - **负值** = 身体后仰（核心失稳 / 代偿）

    若提供 ``ball_center`` 且球心在髋中点左侧，则前向为 −X，自动翻转符号。

    关节点缺失 / 退化 → ``None``（调用方降级，不抛异常）。
    """
    try:
        if not isinstance(landmarks, dict):
            return None
        ls = _pick_landmark_xy(landmarks, "left_shoulder")
        rs = _pick_landmark_xy(landmarks, "right_shoulder")
        lh = _pick_landmark_xy(landmarks, "left_hip")
        rh = _pick_landmark_xy(landmarks, "right_hip")
        if ls is None or rs is None or lh is None or rh is None:
            return None

        mid_shoulder = 0.5 * (ls + rs)
        mid_hip = 0.5 * (lh + rh)
        trunk = mid_shoulder - mid_hip  # (dx, dy) 图像坐标
        dx = float(trunk[0])
        dy = float(trunk[1])
        # 竖直向上分量 = -dy（图像 Y 向下）；退化（零向量）无法定义倾角
        up = -dy
        if abs(dx) < 1e-12 and abs(up) < 1e-12:
            return None
        if not (np.isfinite(dx) and np.isfinite(up)):
            return None

        # atan2(horizontal, vertical_up)：直立 → 0°；+X 侧倾 → 正角
        lean = float(np.degrees(np.arctan2(dx, up)))
        if not np.isfinite(lean):
            return None

        # 前向对齐：球在髋左侧 → 出球方向为 -X，翻转符号使「前倾」仍为正
        if ball_center is not None:
            try:
                bx = float(np.asarray(ball_center, dtype=np.float64).reshape(-1)[0])
                if np.isfinite(bx) and bx < float(mid_hip[0]):
                    lean = -lean
            except (TypeError, ValueError, IndexError):
                pass

        return float(round(lean, 2))
    except Exception:  # noqa: BLE001
        return None


def classify_trunk_lean_status(trunk_lean_deg: Optional[float]) -> str:
    """躯干倾角分档：返回 GREEN / YELLOW / RED（简写色码）。"""
    if trunk_lean_deg is None:
        return "GREEN"
    try:
        a = float(trunk_lean_deg)
    except (TypeError, ValueError):
        return "GREEN"
    if not np.isfinite(a):
        return "GREEN"
    if TRUNK_LEAN_GREEN_LOW_DEG <= a <= TRUNK_LEAN_GREEN_HIGH_DEG:
        return "GREEN"
    if TRUNK_LEAN_YELLOW_BACK_DEG <= a < TRUNK_LEAN_GREEN_LOW_DEG:
        return "YELLOW"
    if TRUNK_LEAN_GREEN_HIGH_DEG < a <= TRUNK_LEAN_RED_FORWARD_DEG:
        return "YELLOW"
    # a < -10 或 a > +25
    return "RED"


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
