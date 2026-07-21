# -*- coding: utf-8 -*-
"""
error_diagnoser.py
绝对零点帧 (T0) 时序隔离 + 确定性典型错误判定引擎。

设计目标：
    彻底消除「把第五阶段随前的惯性弯曲当成第三阶段蓄力折叠」的致命误判。
    所有折叠角 / 大腿后缩角等后摆指标，必须且只能在
        backswing_phase = [T0 - 350ms, T0 - 60ms]
    闭区间内计算；一旦向后跨越到 T0 之后，立即拦截报错。

V3.4 科研级升级：
    - 助跑窗动态切片 [0, T0-10] + CoM 进袭角；
    - T0 由 pose_tracker.detect_contact_frame 纯运动学锁定
      （V2.5：Savitzky-Golay 平滑 ω + locate_impact_frame 抛物线插值锁帧，
       球心=T0 右足尖）；
    - 支撑距全量使用 pose_world_landmarks（米制）+ 右脚长比例尺（>1.3→ERR_A2）；
    - visibility≥0.7 置信过滤 + [T0±15] 膝角速度提前减速检测。

【V2.5 Vision Pipeline Determinism】：
    评分侧 DeterministicScorer 保持纯数学零随机；视觉侧由 pose_tracker /
    api_server 负责同步顺序帧、模型热重置与 CUDA 确定性锁死。

固定机位前提（操场标准化底库）：
    右侧前方 3.5 m、高 1.3 m；十字定位标记；球体中心在画面中近似固定。
"""

from __future__ import annotations

# 【V2.5】与视觉管线一致：在可能触发 CUDA 的依赖前写入 CUBLAS 配置
import os
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# --------------------------------------------------------------------------
# 错误代码（按优先级从高到低：自下而上、由早到晚的动力链顺序）
# --------------------------------------------------------------------------
ERR_APPROACH_TOO_STRAIGHT = "ERR_APPROACH_TOO_STRAIGHT"
ERR_APPROACH_TOO_WIDE = "ERR_APPROACH_TOO_WIDE"
ERR_WARMUP_CLOSE = "ERR_WARMUP_CLOSE"
ERR_SUPPORT_TOO_CLOSE = "ERR_SUPPORT_TOO_CLOSE"
ERR_A1_SUPPORT_BACK = "ERR_A1_SUPPORT_BACK"
ERR_SUPPORT_TOO_WIDE = "ERR_SUPPORT_TOO_WIDE"
ERR_A2_SUPPORT_WIDE = "ERR_A2_SUPPORT_WIDE"  # 兼容别名 → 宽站
ERR_EARLY_DECELERATION = "ERR_EARLY_DECELERATION"
ERR_B1_STRAIGHT_LEG = "ERR_B1_STRAIGHT_LEG"
ERR_B2_SHANK_ONLY = "ERR_B2_SHANK_ONLY"
ERR_C1_LOOSE_ANKLE = "ERR_C1_LOOSE_ANKLE"
ERR_C2_TOE_POKE = "ERR_C2_TOE_POKE"
PASS_SUPPORT_OK = "PASS_SUPPORT_OK"
PASS_STANDARD = "PASS_STANDARD"

ERROR_PRIORITY: tuple[str, ...] = (
    ERR_APPROACH_TOO_STRAIGHT,
    ERR_APPROACH_TOO_WIDE,
    ERR_WARMUP_CLOSE,
    ERR_SUPPORT_TOO_CLOSE,
    ERR_A1_SUPPORT_BACK,
    ERR_SUPPORT_TOO_WIDE,
    ERR_A2_SUPPORT_WIDE,
    ERR_EARLY_DECELERATION,
    ERR_B1_STRAIGHT_LEG,
    ERR_B2_SHANK_ONLY,
    ERR_C1_LOOSE_ANKLE,
    ERR_C2_TOE_POKE,
    PASS_STANDARD,
)

# 阈值（厘米 / 度 / 脚长比）——与教研标准化底库判定口径一致
SUPPORT_WARMUP_CLOSE_CM = 5.0
SUPPORT_BACK_OFFSET_CM = 10.0  # 支撑脚尖落后球心 > 10 cm → A1
SUPPORT_WIDE_CM = 20.0  # 横向绝对距离 > 20 cm → A2（兼容旧阈值）
SUPPORT_LATERAL_IDEAL_LOW_CM = 15.0
SUPPORT_LATERAL_IDEAL_HIGH_CM = 20.0
SUPPORT_FOOT_RATIO_LOW = 0.8  # lateral_dist / foot_len
SUPPORT_FOOT_RATIO_HIGH = 1.3
APPROACH_STRAIGHT_DEG = 20.0
APPROACH_WIDE_DEG = 55.0
APPROACH_IDEAL_LOW_DEG = 30.0
APPROACH_IDEAL_HIGH_DEG = 45.0
BACKSWING_STRAIGHT_LEG_DEG = 170.0  # min(knee) > 170° → B1
BACKSWING_FOLD_IDEAL_LOW_DEG = 140.0
BACKSWING_FOLD_IDEAL_HIGH_DEG = 160.0
THIGH_RETRACTION_NEAR_ZERO_DEG = 8.0  # 大腿后伸 ≈ 0° → B2
ANKLE_VARIANCE_FAIL = 18.0  # impact 窗口踝角方差超标 → C1
ANKLE_DORSIFLEX_DROP_DEG = 12.0  # 背屈角骤降幅度
TOE_POKE_INSTEP_ALIGN_DEG = 35.0  # 足背外展不足阈值
LANDMARK_VISIBILITY_MIN = 0.7
TEMPORAL_WINDOW_HALF_FRAMES = 15
EARLY_DECEL_BEFORE_T0_FRAMES = 5
EARLY_DECEL_DROP_RATIO = 0.55  # 峰值后衰减至 55% 以下视为显著减速

# 相位窗口（相对 T0，毫秒）
PHASE_APPROACH_START_BEFORE_T0_MS = 600.0
PHASE_APPROACH_END_BEFORE_T0_MS = 300.0
PHASE_SUPPORT_PRE_MS = 150.0
PHASE_BACKSWING_PRE_MS = 350.0
PHASE_BACKSWING_END_PRE_MS = 60.0
PHASE_IMPACT_HALF_MS = 20.0
PHASE_FOLLOW_START_MS = 40.0

AVERAGE_CHILD_SHOULDER_WIDTH_CM = 30.0
AVERAGE_CHILD_FOOT_LEN_M = 0.22  # 世界坐标缺失时的脚长兜底（米）

# 阶段窗口最短安全长度：保证后续 min()/max()/argmin 永不为空序列
MIN_PHASE_WINDOW_FRAMES = 3


class BackswingWindowViolation(RuntimeError):
    """后摆指标采样越界：索引落入 T0 之后或离开 backswing 闭区间。"""


def clamp_phase_indices(
    start_idx: int,
    end_idx: int,
    total_frames: int,
    *,
    prefer_before_t0: Optional[int] = None,
    min_frames: int = MIN_PHASE_WINDOW_FRAMES,
) -> tuple[int, int]:
    """对阶段切片下标做边界限幅，并在空窗时自动扩展至少 min_frames 帧。

    规则：
      start = max(0, int(start))
      end   = min(n-1, int(end))
      若 end < start 或长度不足，围绕 prefer_before_t0（或窗口中点）向两侧扩展，
      确保后续 min()/max() 绝不抛空序列异常。
    """
    n = int(total_frames)
    if n <= 0:
        return 0, 0
    last = n - 1
    start = max(0, min(last, int(start_idx)))
    end = max(0, min(last, int(end_idx)))

    if end < start:
        if prefer_before_t0 is not None:
            anchor = max(0, min(last, int(prefer_before_t0) - 1))
        else:
            anchor = max(0, min(last, (start + end) // 2))
        start = max(0, anchor - max(0, min_frames // 2))
        end = min(last, start + max(1, min_frames) - 1)
        if end < start:
            start = end = anchor

    length = end - start + 1
    need = max(1, int(min_frames))
    if length < need:
        if n < need:
            return 0, last
        # 优先向左（更靠近触球前）扩展，再向右补齐
        missing = need - length
        left = min(start, missing)
        start -= left
        missing -= left
        if missing > 0:
            end = min(last, end + missing)
        length = end - start + 1
        if length < need:
            start = max(0, end - need + 1)
        if prefer_before_t0 is not None and end >= int(prefer_before_t0) and int(prefer_before_t0) > 0:
            end = min(end, max(0, int(prefer_before_t0) - 1))
            start = max(0, min(start, end))
            if end - start + 1 < min(need, int(prefer_before_t0)):
                start = max(0, end - min(need, int(prefer_before_t0)) + 1)

    return int(start), int(end)


# --------------------------------------------------------------------------
# 基础几何
# --------------------------------------------------------------------------
def _as_vec3(point) -> np.ndarray:
    arr = np.asarray(point, dtype=np.float64).reshape(-1)
    if arr.size >= 3:
        return arr[:3].copy()
    if arr.size == 2:
        return np.array([arr[0], arr[1], 0.0], dtype=np.float64)
    raise ValueError(f"point 维度不足: {arr.shape}")


def calculate_angle(a, b, c) -> float:
    """以 b 为顶点的空间夹角（度）。"""
    ba = _as_vec3(a) - _as_vec3(b)
    bc = _as_vec3(c) - _as_vec3(b)
    na = np.linalg.norm(ba)
    nb = np.linalg.norm(bc)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cos_v = float(np.clip(np.dot(ba, bc) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_v)))


def _time_derivative_series(values, timestamps_sec) -> list[float]:
    n = len(values)
    out = [0.0] * n
    for i in range(n):
        if n == 1:
            continue
        if i == 0:
            j0, j1 = 0, 1
        elif i == n - 1:
            j0, j1 = n - 2, n - 1
        else:
            j0, j1 = i - 1, i + 1
        dt = float(timestamps_sec[j1] - timestamps_sec[j0])
        if dt <= 0:
            out[i] = 0.0
        else:
            out[i] = float((values[j1] - values[j0]) / dt)
    return out


def _estimate_cm_per_pixel(frame_record: dict) -> float:
    left = _as_vec3(frame_record["left_shoulder"])[:2]
    right = _as_vec3(frame_record["right_shoulder"])[:2]
    shoulder_px = float(np.linalg.norm(right - left))
    if shoulder_px <= 1e-6:
        return 1.0
    return AVERAGE_CHILD_SHOULDER_WIDTH_CM / shoulder_px


def _landmark_visibility(frame_record: dict, joint: str) -> float:
    vis = frame_record.get("visibility") or {}
    if isinstance(vis, dict) and joint in vis:
        try:
            return float(vis[joint])
        except (TypeError, ValueError):
            return 1.0
    return 1.0  # 旧帧无 visibility 字段时视为可信


def _frame_passes_visibility(
    frame_record: dict,
    joints: tuple[str, ...],
    min_vis: float = LANDMARK_VISIBILITY_MIN,
) -> bool:
    return all(_landmark_visibility(frame_record, j) >= min_vis for j in joints)


def _world_joint(frame_record: dict, joint: str) -> Optional[np.ndarray]:
    """读取 pose_world_landmarks 米制坐标；缺失返回 None。"""
    world = frame_record.get("world")
    if not isinstance(world, dict) or joint not in world:
        return None
    try:
        return _as_vec3(world[joint])
    except (TypeError, ValueError):
        return None


def _joint_prefer_world(frame_record: dict, joint: str) -> np.ndarray:
    """优先世界坐标（米），否则回退像素近似坐标。"""
    w = _world_joint(frame_record, joint)
    if w is not None and np.all(np.isfinite(w)):
        return w
    return _as_vec3(frame_record[joint])


def _has_world_landmarks(frame_record: dict) -> bool:
    world = frame_record.get("world")
    return isinstance(world, dict) and "left_ankle" in world and "right_ankle" in world


def _dist3d(a, b) -> float:
    return float(np.linalg.norm(_as_vec3(a) - _as_vec3(b)))


def _compute_foot_len_m(frame_record: dict, side: str = "right") -> float:
    """动态脚长：dist3d(HEEL, FOOT_INDEX)，优先世界坐标（米）。

    默认取摆动侧右脚（right_heel → right_foot_index），与支撑横向归一化口径一致。
    """
    heel_key = f"{side}_heel"
    toe_key = f"{side}_foot_index"
    heel_w = _world_joint(frame_record, heel_key)
    toe_w = _world_joint(frame_record, toe_key)
    if heel_w is not None and toe_w is not None:
        length = _dist3d(heel_w, toe_w)
        if length > 0.05:  # 儿童脚长合理下限 ~5cm
            return float(length)
    # 同坐标系像素回退：用像素脚长 / 像素肩宽 * 真实肩宽 → 米（仅作比例尺兜底）
    heel = frame_record.get(heel_key)
    toe = frame_record.get(toe_key)
    if heel is not None and toe is not None:
        try:
            cm_per_px = _estimate_cm_per_pixel(frame_record)
            length_cm = _dist3d(heel, toe) * cm_per_px
            if length_cm > 5.0:
                return float(length_cm / 100.0)
        except Exception:
            pass
    return float(AVERAGE_CHILD_FOOT_LEN_M)


def _foot_dorsum_center(frame_record: dict) -> np.ndarray:
    """右脚足背边界框中心：踝 + 足尖 的中点（像素近似 3D）。"""
    ankle = _as_vec3(frame_record["right_ankle"])
    toe = _as_vec3(frame_record["right_foot_index"])
    return 0.5 * (ankle + toe)


def _shank_forward_angle_deg(frame_record: dict) -> float:
    """小腿向量 knee→ankle 相对竖直向下方向的前向倾角（度）。

    画面 X 增大视为「向前」（固定右侧前方机位下，踢球腿前摆主分量）。
    """
    knee = _as_vec3(frame_record["right_knee"])
    ankle = _as_vec3(frame_record["right_ankle"])
    shank = ankle - knee
    # 竖直向下参考（图像 Y 向下为正）
    vertical = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    # 投影到 X-Y 平面后求与竖直的有符号偏角：X 正方向为正（向前）
    sx, sy = float(shank[0]), float(shank[1])
    if abs(sx) < 1e-9 and abs(sy) < 1e-9:
        return 0.0
    # atan2(forward, down)
    return float(np.degrees(np.arctan2(sx, max(sy, 1e-6))))


def _thigh_retraction_deg(frame_record: dict) -> float:
    """大腿后伸角：髋→膝 相对躯干竖直向下，在矢状相关平面上的后向偏角。

    后伸越大（膝在髋后方）数值越大；≈0 表示大腿几乎不后拉。
    """
    hip = _as_vec3(frame_record["right_hip"])
    knee = _as_vec3(frame_record["right_knee"])
    thigh = knee - hip
    # 后向：画面 X 减小（相对前摆方向取反）
    tx, ty = float(thigh[0]), float(thigh[1])
    if abs(tx) < 1e-9 and abs(ty) < 1e-9:
        return 0.0
    # 后伸角 = max(0, -atan2(tx, ty)) —— 膝落后于髋（tx 为负）时为正
    signed = float(np.degrees(np.arctan2(-tx, max(ty, 1e-6))))
    return float(max(0.0, signed))


def _instep_abduction_proxy_deg(frame_record: dict, ball_center: np.ndarray) -> float:
    """足背外展代理角：踝→足尖 与 踝→球心 的夹角。

    脚背内侧射门要求足背外展对准球体；夹角过小 ≈ 脚尖直捅。
    """
    ankle = _as_vec3(frame_record["right_ankle"])
    toe = _as_vec3(frame_record["right_foot_index"])
    ball = _as_vec3(ball_center)
    return calculate_angle(toe, ankle, ball)


# --------------------------------------------------------------------------
# 相位窗口
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PhaseWindow:
    name: str
    start_ms_rel: float
    end_ms_rel: float
    start_index: int
    end_index: int  # inclusive

    def contains_index(self, index: int) -> bool:
        return self.start_index <= int(index) <= self.end_index

    def indices(self) -> range:
        if self.end_index < self.start_index:
            return range(0, 0)
        return range(self.start_index, self.end_index + 1)


@dataclass
class TemporalIsolationPlan:
    t0_index: int
    t0_timestamp_sec: float
    ball_center: np.ndarray
    cm_per_pixel: float
    phases: dict[str, PhaseWindow] = field(default_factory=dict)

    def require_backswing_index(self, index: int) -> None:
        phase = self.phases["backswing_phase"]
        if int(index) > self.t0_index:
            raise BackswingWindowViolation(
                f"后摆指标索引 {index} 已越过绝对零点 T0={self.t0_index}，严禁跨阶段计算！"
            )
        if not phase.contains_index(index):
            raise BackswingWindowViolation(
                f"后摆指标索引 {index} 不在 backswing_phase "
                f"[{phase.start_index}, {phase.end_index}] 闭区间内！"
            )


def _index_at_or_before(timestamps_sec: list[float], t0_sec: float, offset_ms: float) -> int:
    """返回时间戳 <= t0_sec + offset_ms 的最大合法下标。"""
    target = t0_sec + offset_ms / 1000.0
    idx = 0
    for i, t in enumerate(timestamps_sec):
        if t <= target:
            idx = i
        else:
            break
    return int(idx)


def _index_at_or_after(timestamps_sec: list[float], t0_sec: float, offset_ms: float) -> int:
    """返回时间戳 >= t0_sec + offset_ms 的最小合法下标。"""
    target = t0_sec + offset_ms / 1000.0
    n = len(timestamps_sec)
    for i, t in enumerate(timestamps_sec):
        if t >= target:
            return int(i)
    return int(n - 1)


def build_phase_windows(
    timestamps_sec: list[float],
    t0_index: int,
) -> dict[str, PhaseWindow]:
    """严格隔离五大搜索窗口（时间闭区间 → 帧下标闭区间）。

    所有切片下标均经 clamp_phase_indices 限幅；空窗自动扩展至少 3 帧。
    """
    n = len(timestamps_sec)
    if n == 0:
        raise ValueError("空帧序列，无法构建相位窗口")
    t0_index = int(max(0, min(n - 1, int(t0_index) if t0_index is not None else int(round((n - 1) * 0.60)))))
    t0_sec = float(timestamps_sec[t0_index])

    # ① 助跑环节动态窗：[0, max(0, T0-10)] —— 废弃死板 600~300ms
    approach_start = 0
    approach_end = max(0, t0_index - 10)
    if approach_end < approach_start:
        approach_end = approach_start
    if approach_end == approach_start and t0_index > 0:
        approach_end = max(0, t0_index - 1)
    support_start = _index_at_or_after(timestamps_sec, t0_sec, -PHASE_SUPPORT_PRE_MS)
    support_end = t0_index

    back_start = _index_at_or_after(timestamps_sec, t0_sec, -PHASE_BACKSWING_PRE_MS)
    back_end = _index_at_or_before(timestamps_sec, t0_sec, -PHASE_BACKSWING_END_PRE_MS)
    # 硬锁：后摆终点绝不能 >= T0
    back_end = min(back_end, max(0, t0_index - 1))
    if back_end < back_start:
        # 极短切片：退化为 T0 前尽可能靠近但不超过 T0 的窄窗
        back_start = max(0, t0_index - 2)
        back_end = max(0, t0_index - 1)
        if back_end < back_start:
            back_start = back_end = max(0, t0_index - 1)

    impact_start = _index_at_or_after(timestamps_sec, t0_sec, -PHASE_IMPACT_HALF_MS)
    impact_end = _index_at_or_before(timestamps_sec, t0_sec, PHASE_IMPACT_HALF_MS)
    impact_start = min(impact_start, t0_index)
    impact_end = max(impact_end, t0_index)

    follow_start = _index_at_or_after(timestamps_sec, t0_sec, PHASE_FOLLOW_START_MS)
    follow_end = n - 1

    # 边界限幅；助跑终点硬钳制在 T0-10（或 T0-1）
    ap_s = int(max(0, min(n - 1, approach_start)))
    ap_e = int(max(0, min(n - 1, approach_end)))
    if t0_index > 0:
        ap_e = min(ap_e, max(0, t0_index - 1))
    if ap_e < ap_s:
        ap_s = ap_e
    sp_s, sp_e = clamp_phase_indices(support_start, support_end, n, prefer_before_t0=t0_index + 1)
    bk_s, bk_e = clamp_phase_indices(back_start, back_end, n, prefer_before_t0=t0_index)
    # 后摆终点仍不得越过 T0（若 T0=0 则允许退化为 [0,0]）
    if t0_index > 0:
        bk_e = min(bk_e, t0_index - 1)
        bk_s = min(bk_s, bk_e)
        bk_s, bk_e = clamp_phase_indices(bk_s, bk_e, n, prefer_before_t0=t0_index)
        bk_e = min(bk_e, t0_index - 1)
        bk_s = min(bk_s, bk_e)
    im_s, im_e = clamp_phase_indices(impact_start, impact_end, n)
    # 击球窗必须包含 T0
    im_s = min(im_s, t0_index)
    im_e = max(im_e, t0_index)
    fl_s, fl_e = clamp_phase_indices(follow_start, follow_end, n)
    if fl_s <= t0_index < n - 1:
        fl_s = min(n - 1, max(fl_s, t0_index + 1))
        fl_s, fl_e = clamp_phase_indices(fl_s, fl_e, n)

    return {
        "approach_phase": PhaseWindow(
            "approach_phase",
            float("-inf"),  # 动态窗：自序列起点
            -10.0,  # 相对 T0 约 -10 帧（非毫秒）
            ap_s,
            ap_e,
        ),
        "support_phase": PhaseWindow(
            "support_phase",
            -PHASE_SUPPORT_PRE_MS,
            0.0,
            sp_s,
            sp_e,
        ),
        "backswing_phase": PhaseWindow(
            "backswing_phase",
            -PHASE_BACKSWING_PRE_MS,
            -PHASE_BACKSWING_END_PRE_MS,
            bk_s,
            bk_e,
        ),
        "impact_phase": PhaseWindow(
            "impact_phase",
            -PHASE_IMPACT_HALF_MS,
            PHASE_IMPACT_HALF_MS,
            im_s,
            im_e,
        ),
        "follow_through_phase": PhaseWindow(
            "follow_through_phase",
            PHASE_FOLLOW_START_MS,
            float("inf"),
            fl_s,
            fl_e,
        ),
    }


# --------------------------------------------------------------------------
# T0 绝对零点锁定
# --------------------------------------------------------------------------
def estimate_fixed_ball_center(frames: list[dict]) -> np.ndarray:
    """固定机位下球体中心近似：取足背轨迹「最低前伸簇」的空间中位数。

    球在画面中静止，踢球腿足背会在触球瞬间逼近球心；对足背中心序列
    取 Y 较大（更靠近地面）的 15% 分位样本的中位数，作为固定球心。
    """
    if not frames:
        return np.zeros(3, dtype=np.float64)
    centers = np.stack([_foot_dorsum_center(f) for f in frames], axis=0)
    y_vals = centers[:, 1]
    threshold = float(np.percentile(y_vals, 85.0))
    near_ground = centers[y_vals >= threshold]
    if near_ground.shape[0] == 0:
        near_ground = centers
    return np.median(near_ground, axis=0)


def lock_absolute_t0(
    frames: list[dict],
    ball_center: Optional[np.ndarray] = None,
) -> tuple[int, np.ndarray, dict[str, Any]]:
    """锁定击球瞬间绝对零点帧 T0。

    【V3.4 / V2.5】权威实现委托 pose_tracker.detect_contact_frame：
      ① 膝/小腿角速度 Savitzky-Golay 平滑后取 |ω| 峰值；
      ② 峰邻域踝-球欧氏距离极小 + 三点抛物线插值细化 → t0_index（零漂移）；
      ③ ball_center = t0 帧 right_foot_index（绝对锚点，禁止左脚猜球心）。

    【强容错】绝不返回 None：异常时回退小腿 |ω| 峰值或 60% 时间线，
    球心仍强制取该帧 right_foot_index。
    """
    del ball_center  # 禁止外部/静态球心覆盖 T0 足尖锚点
    n = len(frames)
    if n == 0:
        raise ValueError("空帧序列，无法锁定 T0")

    fallback_sixty = int(max(0, min(n - 1, round((n - 1) * 0.60))))

    def _t0_foot_anchor(idx: int) -> np.ndarray:
        idx = int(max(0, min(n - 1, idx)))
        rec = frames[idx]
        w = _world_joint(rec, "right_foot_index")
        if w is not None and np.all(np.isfinite(w)):
            return _as_vec3(w)
        try:
            return _as_vec3(rec["right_foot_index"])
        except Exception:
            return _foot_dorsum_center(rec)

    try:
        # 延迟导入：避免与 pose_tracker 顶层互相 import 形成环
        from pose_tracker import detect_contact_frame

        t0_index, ball, meta = detect_contact_frame(frames, None)
        t0_index = int(max(0, min(n - 1, int(t0_index))))
        # 强制锚点：无论 detect 返回何值，球心死锁为 T0 右足尖
        ball = _t0_foot_anchor(t0_index)
        meta = dict(meta or {})
        meta.setdefault("t0_method", "locate_impact_frame_parabolic_v25")
        meta["ball_estimate"] = "t0_right_foot_index_anchor"
        return t0_index, _as_vec3(ball), meta
    except Exception as exc:  # noqa: BLE001 - 任何异常都必须安全兜底
        meta: dict[str, Any] = {
            "t0_fallback": "exception_sixty_percent",
            "t0_exception": str(exc),
            "t0_method": "legacy_shank_omega_fallback",
            "ball_estimate": "t0_right_foot_index_anchor",
        }
        try:
            timestamps = [float(f.get("timestamp_sec", i / 30.0)) for i, f in enumerate(frames)]
            shank_angles = [_shank_forward_angle_deg(f) for f in frames]
            shank_omega = _time_derivative_series(shank_angles, timestamps)
            omega_arr = np.asarray(shank_omega, dtype=np.float64)
            pos = omega_arr.copy()
            pos[pos <= 0.0] = -np.inf
            if bool(np.any(np.isfinite(pos) & (pos > 0.0))):
                peak = int(np.argmax(pos))
                t0_index = int(min(n - 1, peak + 2))
                meta["t0_fallback"] = "exception_shank_omega_positive_peak_plus2"
            else:
                abs_omega = np.abs(omega_arr)
                if float(np.max(abs_omega)) > 1e-6:
                    t0_index = int(np.argmax(abs_omega))
                    meta["t0_fallback"] = "exception_shank_omega_abs_peak"
                else:
                    t0_index = fallback_sixty
        except Exception:
            t0_index = fallback_sixty
        t0_index = int(max(0, min(n - 1, t0_index)))
        ball = _t0_foot_anchor(t0_index)
        meta.update(
            {
                "shank_omega_peak_index": t0_index,
                "shank_omega_peak_value": 0.0,
                "shank_omega_decay_index": t0_index,
                "min_foot_ball_dist_index": t0_index,
                "min_foot_ball_dist_px": 0.0,
                "t0_search_range": [t0_index, t0_index],
            }
        )
        return t0_index, _as_vec3(ball), meta


# --------------------------------------------------------------------------
# 相位内指标（严禁跨阶段）
# --------------------------------------------------------------------------
def _support_metrics(frames, phase: PhaseWindow, ball_center, cm_per_pixel) -> dict[str, Any]:
    """支撑站位：强制世界坐标（米）+ 右脚长比例归一化。

    严禁 2D 像素 × cm_per_pixel 作为主路径。
    lateral_dist / foot_len > 1.3 → ERR_A2_SUPPORT_WIDE。
    """
    del cm_per_pixel  # 主路径禁用像素比例尺
    n = len(frames)
    support_joints = ("left_hip", "left_knee", "left_ankle", "left_foot_index")
    if phase.end_index < phase.start_index:
        idx = max(0, min(n - 1, phase.end_index))
    else:
        # 支撑脚落地：支撑踝垂直速度最小（跳过低置信帧）
        best_i = int(max(0, min(n - 1, phase.start_index)))
        best_speed = float("inf")
        for i in phase.indices():
            if i <= 0 or i >= n:
                continue
            if not _frame_passes_visibility(frames[i], support_joints):
                continue
            try:
                dt = float(frames[i]["timestamp_sec"] - frames[i - 1]["timestamp_sec"])
            except Exception:
                continue
            if dt <= 0:
                continue
            y_i = float(_joint_prefer_world(frames[i], "left_ankle")[1])
            y_p = float(_joint_prefer_world(frames[i - 1], "left_ankle")[1])
            speed = abs((y_i - y_p) / dt)
            if speed < best_speed:
                best_speed = speed
                best_i = i
        idx = best_i

    idx = int(max(0, min(n - 1, idx)))
    rec = frames[idx]
    use_world = _has_world_landmarks(rec)
    ankle = _joint_prefer_world(rec, "left_ankle")
    toe = _joint_prefer_world(rec, "left_foot_index")
    hip = _joint_prefer_world(rec, "left_hip")
    knee = _joint_prefer_world(rec, "left_knee")
    ball = _as_vec3(ball_center)

    # 球心必须与关节同坐标系：优先世界系右足尖锚点
    ball_w = _world_joint(rec, "right_foot_index")
    if ball_w is None:
        # 回退到序列中 T0 邻域：调用方传入的 ball 若为世界系则直接用
        if use_world and float(np.linalg.norm(ball)) > 5.0:
            ball_w = _world_joint(rec, "right_ankle")
    if ball_w is not None:
        ball = ball_w

    # 强制右脚 3D 脚长
    foot_len_m = _compute_foot_len_m(rec, side="right")
    if foot_len_m < 0.05:
        foot_len_m = float(AVERAGE_CHILD_FOOT_LEN_M)

    # 左踝—球心 3D 横向距离（同坐标系比值）
    lateral_raw = abs(float(ankle[0] - ball[0]))
    ap_raw = float(toe[2] - ball[2])
    dist_xz_raw = float(
        np.linalg.norm(
            np.array([ankle[0] - ball[0], ankle[2] - ball[2]], dtype=np.float64)
        )
    )
    if lateral_raw < 1e-9 and dist_xz_raw > 1e-9:
        lateral_raw = dist_xz_raw

    ratio = float(lateral_raw / foot_len_m) if foot_len_m > 1e-9 else 0.0

    if use_world or foot_len_m < 1.0:
        # 米制输出 cm
        lateral = float(lateral_raw * 100.0)
        ap = float(ap_raw * 100.0)
        dist_xz = float(dist_xz_raw * 100.0)
        coord_space = "world_m"
    else:
        # 无可靠世界坐标：脚长归一化反推等效 cm（禁止 cm_per_pixel）
        lateral = float(ratio * AVERAGE_CHILD_FOOT_LEN_M * 100.0)
        ap = float((ap_raw / foot_len_m) * AVERAGE_CHILD_FOOT_LEN_M * 100.0) if foot_len_m > 1e-9 else 0.0
        dist_xz = float((dist_xz_raw / foot_len_m) * AVERAGE_CHILD_FOOT_LEN_M * 100.0) if foot_len_m > 1e-9 else lateral
        coord_space = "foot_len_normalized"

    if SUPPORT_FOOT_RATIO_LOW <= ratio <= SUPPORT_FOOT_RATIO_HIGH:
        support_code = PASS_SUPPORT_OK
    elif ratio < SUPPORT_FOOT_RATIO_LOW:
        support_code = ERR_SUPPORT_TOO_CLOSE
    else:
        # 严格：脚长比 > 1.3 → ERR_A2_SUPPORT_WIDE
        support_code = ERR_A2_SUPPORT_WIDE

    try:
        support_knee = calculate_angle(hip, knee, ankle)
    except Exception:
        support_knee = 150.0

    return {
        "landing_frame_index": int(idx),
        "support_lateral_dist_cm": round(lateral, 1),
        "support_ap_offset_cm": round(ap, 1),
        "support_ball_dist_cm": round(dist_xz, 1),
        "support_knee_angle": round(float(support_knee), 1),
        "foot_len_m": round(float(foot_len_m), 4),
        "support_foot_ratio": round(ratio, 3),
        "support_stance_code": support_code,
        "support_coord_space": coord_space,
    }


def _backswing_metrics(frames, plan: TemporalIsolationPlan) -> dict[str, Any]:
    phase = plan.phases["backswing_phase"]
    n = len(frames)
    start_i, end_i = clamp_phase_indices(
        phase.start_index,
        phase.end_index,
        n,
        prefer_before_t0=plan.t0_index,
    )
    if plan.t0_index > 0:
        end_i = min(end_i, plan.t0_index - 1)
        start_i = min(start_i, end_i)
    if end_i < start_i:
        # T0=0 等极端短视频：退化为首帧安全窗，绝不抛空序列
        start_i = end_i = 0

    knee_angles = []
    thigh_angles = []
    for i in range(start_i, end_i + 1):
        if i > plan.t0_index:
            continue
        rec = frames[i]
        knee_angles.append(
            calculate_angle(rec["right_hip"], rec["right_knee"], rec["right_ankle"])
        )
        thigh_angles.append(_thigh_retraction_deg(rec))

    if not knee_angles:
        # 最终兜底：用 T0 前最多 3 帧，保证 min() 可调用
        safe_end = max(0, min(n - 1, plan.t0_index - 1 if plan.t0_index > 0 else 0))
        safe_start = max(0, safe_end - MIN_PHASE_WINDOW_FRAMES + 1)
        for i in range(safe_start, safe_end + 1):
            rec = frames[i]
            knee_angles.append(
                calculate_angle(rec["right_hip"], rec["right_knee"], rec["right_ankle"])
            )
            thigh_angles.append(_thigh_retraction_deg(rec))
        start_i, end_i = safe_start, safe_end

    if not knee_angles:
        # 理论不可达（n>=1）；仍返回安全默认值，杜绝崩溃
        idx = max(0, min(n - 1, plan.t0_index))
        return {
            "swing_fold_angle": 180.0,
            "thigh_retraction_deg": 0.0,
            "backswing_extreme_frame_index": int(idx),
            "backswing_window": [int(idx), int(idx)],
        }

    local_argmin = int(np.argmin(np.asarray(knee_angles, dtype=np.float64)))
    extreme_idx = start_i + local_argmin
    if extreme_idx >= plan.t0_index and plan.t0_index > 0:
        extreme_idx = max(0, plan.t0_index - 1)
    extreme_idx = int(max(0, min(n - 1, extreme_idx)))

    swing_fold = float(knee_angles[local_argmin])
    thigh_ret = float(max(thigh_angles)) if thigh_angles else 0.0
    return {
        "swing_fold_angle": round(swing_fold, 1),
        "thigh_retraction_deg": round(thigh_ret, 1),
        "backswing_extreme_frame_index": int(extreme_idx),
        "backswing_window": [int(start_i), int(end_i)],
    }


def _impact_metrics(frames, plan: TemporalIsolationPlan) -> dict[str, Any]:
    phase = plan.phases["impact_phase"]
    idxs = list(phase.indices())
    if not idxs:
        idxs = [plan.t0_index]
    ankle_angles = [
        calculate_angle(
            frames[i]["right_knee"], frames[i]["right_ankle"], frames[i]["right_foot_index"]
        )
        for i in idxs
    ]
    t0_local = idxs.index(plan.t0_index) if plan.t0_index in idxs else len(idxs) // 2
    ankle_at_t0 = float(ankle_angles[t0_local])
    variance = float(np.var(ankle_angles)) if len(ankle_angles) >= 2 else 0.0
    # 背屈角骤降：窗口内 max - min
    dorsi_drop = float(max(ankle_angles) - min(ankle_angles)) if ankle_angles else 0.0
    abduction = _instep_abduction_proxy_deg(frames[plan.t0_index], plan.ball_center)
    return {
        "ankle_angle": round(ankle_at_t0, 1),
        "ankle_variance": round(variance, 2),
        "ankle_dorsiflex_drop_deg": round(dorsi_drop, 1),
        "ankle_locked": bool(variance < 6.0 and ankle_at_t0 > 130.0),
        "instep_abduction_deg": round(float(abduction), 1),
    }


def _follow_through_metrics(frames, plan: TemporalIsolationPlan) -> dict[str, Any]:
    """随前期：仅计算跨越中线，严禁计算折叠角。"""
    phase = plan.phases["follow_through_phase"]
    contact = frames[plan.t0_index]
    mid_hip_x = float(
        (_as_vec3(contact["left_hip"])[0] + _as_vec3(contact["right_hip"])[0]) * 0.5
    )
    side0 = float(_as_vec3(contact["right_foot_index"])[0] - mid_hip_x)
    crossed = False
    for i in phase.indices():
        # 显式拒绝在此读取膝角做折叠判定
        mid = float(
            (_as_vec3(frames[i]["left_hip"])[0] + _as_vec3(frames[i]["right_hip"])[0]) * 0.5
        )
        side = float(_as_vec3(frames[i]["right_foot_index"])[0] - mid)
        if side0 != 0.0 and side * side0 < 0:
            crossed = True
            break
    return {"cross_body_follow_through": bool(crossed)}


def _com_xz(frame_record: dict) -> Optional[np.ndarray]:
    """左右髋中点 CoM 在 X-Z 平面的投影；低置信返回 None。"""
    if not _frame_passes_visibility(frame_record, ("left_hip", "right_hip")):
        return None
    left = _joint_prefer_world(frame_record, "left_hip")
    right = _joint_prefer_world(frame_record, "right_hip")
    mid = 0.5 * (left + right)
    return np.array([float(mid[0]), float(mid[2])], dtype=np.float64)


def _fit_approach_vector(com_points: np.ndarray) -> np.ndarray:
    """最小二乘 / SVD 拟合助跑进袭向量 V_approach（X-Z 平面单位向量）。"""
    if com_points.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=np.float64)
    centered = com_points - np.mean(com_points, axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0].astype(np.float64)
    except np.linalg.LinAlgError:
        direction = com_points[-1] - com_points[0]
    chord = com_points[-1] - com_points[0]
    if float(np.dot(direction, chord)) < 0.0:
        direction = -direction
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return direction / norm


def _approach_metrics(frames, plan: TemporalIsolationPlan, support: dict) -> dict[str, Any]:
    """助跑环节动态窗 [0, T0-10]：CoM 轨迹 → V_approach → θ_approach。

    权威实现与 pose_tracker.analyze_approach_phase 对齐；此处保留供诊断管线内联调用。
    """
    # 委托 pose_tracker，保证与 analyze() 强制合并口径一致
    try:
        from pose_tracker import analyze_approach_phase

        return analyze_approach_phase(
            frames, plan.t0_index, ball_center=plan.ball_center
        )
    except Exception:
        pass

    n = len(frames)
    t0 = int(max(0, min(n - 1, plan.t0_index)))
    start_idx = 0
    end_idx = max(0, t0 - 10)
    if end_idx < start_idx:
        end_idx = start_idx

    com_list: list[np.ndarray] = []
    for i in range(start_idx, end_idx + 1):
        com = _com_xz(frames[i])
        if com is not None and np.all(np.isfinite(com)):
            com_list.append(com)

    if len(com_list) < 2:
        start = frames[start_idx]
        end = frames[end_idx]
        c0 = _com_xz(start)
        c1 = _com_xz(end)
        if c0 is None:
            c0 = np.array([0.0, 0.0], dtype=np.float64)
        if c1 is None:
            c1 = c0 + np.array([1.0, 0.0], dtype=np.float64)
        com_list = [c0, c1]

    com_pts = np.stack(com_list, axis=0)
    v_approach = _fit_approach_vector(com_pts)

    ball = _as_vec3(plan.ball_center)
    if _has_world_landmarks(frames[t0]) and float(np.linalg.norm(ball)) > 5.0:
        ball_w = _world_joint(frames[t0], "right_foot_index")
        if ball_w is not None:
            ball = ball_w
    ball_xz = np.array([float(ball[0]), float(ball[2])], dtype=np.float64)
    v_goal = ball_xz - com_pts[0]
    nb = float(np.linalg.norm(v_goal))
    if nb < 1e-9:
        land_idx = int(support.get("landing_frame_index", t0))
        land_idx = max(0, min(n - 1, land_idx))
        ankle = _joint_prefer_world(frames[land_idx], "left_ankle")
        v_goal = np.array([float(ball[0] - ankle[0]), float(ball[2] - ankle[2])], dtype=np.float64)
        nb = float(np.linalg.norm(v_goal))
    if nb < 1e-9:
        v_goal = np.array([1.0, 0.0], dtype=np.float64)
    else:
        v_goal = v_goal / nb

    cos_v = float(np.clip(np.dot(v_approach, v_goal), -1.0, 1.0))
    raw = float(np.degrees(np.arccos(cos_v)))
    angle = float(min(raw, 180.0 - raw))
    angle = float(max(0.0, min(90.0, angle)))

    if angle < APPROACH_STRAIGHT_DEG:
        approach_code = ERR_APPROACH_TOO_STRAIGHT
    elif angle > APPROACH_WIDE_DEG:
        approach_code = ERR_APPROACH_TOO_WIDE
    else:
        approach_code = None

    return {
        "approach_angle": round(angle, 1),
        "approach_vector": [round(float(v_approach[0]), 4), round(float(v_approach[1]), 4)],
        "approach_goal_vector": [round(float(v_goal[0]), 4), round(float(v_goal[1]), 4)],
        "approach_error_code": approach_code,
        "approach_ideal": bool(APPROACH_IDEAL_LOW_DEG <= angle <= APPROACH_IDEAL_HIGH_DEG),
        "approach_sample_count": int(len(com_list)),
        "approach_window": [int(start_idx), int(end_idx)],
    }


def _early_deceleration_metrics(frames, t0_index: int) -> dict[str, Any]:
    """[T0±15] 滑动窗：膝角速度微积分；T0-5 前显著衰减 → ERR_EARLY_DECELERATION。"""
    n = len(frames)
    if n < 3:
        return {
            "early_deceleration": False,
            "knee_omega_series": [],
            "knee_omega_peak_index": 0,
            "knee_omega_peak_value": 0.0,
        }

    lo = max(0, int(t0_index) - TEMPORAL_WINDOW_HALF_FRAMES)
    hi = min(n - 1, int(t0_index) + TEMPORAL_WINDOW_HALF_FRAMES)
    knee_joints = ("right_hip", "right_knee", "right_ankle")

    timestamps: list[float] = []
    knee_angles: list[float] = []
    index_map: list[int] = []
    for i in range(lo, hi + 1):
        if not _frame_passes_visibility(frames[i], knee_joints):
            continue
        hip = _joint_prefer_world(frames[i], "right_hip")
        knee = _joint_prefer_world(frames[i], "right_knee")
        ankle = _joint_prefer_world(frames[i], "right_ankle")
        knee_angles.append(calculate_angle(hip, knee, ankle))
        timestamps.append(float(frames[i]["timestamp_sec"]))
        index_map.append(i)

    if len(knee_angles) < 3:
        # 置信过滤过严时回退全窗像素坐标
        timestamps = [float(frames[i]["timestamp_sec"]) for i in range(lo, hi + 1)]
        knee_angles = [
            calculate_angle(frames[i]["right_hip"], frames[i]["right_knee"], frames[i]["right_ankle"])
            for i in range(lo, hi + 1)
        ]
        index_map = list(range(lo, hi + 1))

    omega = _time_derivative_series(knee_angles, timestamps)
    abs_omega = [abs(float(w)) for w in omega]
    if not abs_omega:
        return {
            "early_deceleration": False,
            "knee_omega_series": [],
            "knee_omega_peak_index": int(t0_index),
            "knee_omega_peak_value": 0.0,
        }

    local_peak = int(np.argmax(np.asarray(abs_omega, dtype=np.float64)))
    peak_val = float(abs_omega[local_peak])
    peak_global = int(index_map[local_peak])
    early_cutoff = int(t0_index) - EARLY_DECEL_BEFORE_T0_FRAMES

    early = False
    decay_idx = peak_global
    if peak_val > 1e-6 and peak_global < early_cutoff:
        thresh = peak_val * EARLY_DECEL_DROP_RATIO
        for k in range(local_peak, len(abs_omega)):
            if abs_omega[k] <= thresh and index_map[k] < early_cutoff:
                early = True
                decay_idx = int(index_map[k])
                break
        # 峰值本身已在 T0-5 之前且之后持续低于峰值 55%
        if not early and peak_global <= early_cutoff:
            post = [abs_omega[k] for k in range(local_peak, len(abs_omega)) if index_map[k] < int(t0_index)]
            if post and float(np.mean(post)) <= thresh:
                early = True
                decay_idx = peak_global

    series = [
        {
            "frame_index": int(index_map[k]),
            "knee_angle": round(float(knee_angles[k]), 2),
            "knee_omega_deg_s": round(float(omega[k]), 2),
        }
        for k in range(len(index_map))
    ]
    return {
        "early_deceleration": bool(early),
        "early_deceleration_code": ERR_EARLY_DECELERATION if early else None,
        "knee_omega_series": series,
        "knee_omega_peak_index": int(peak_global),
        "knee_omega_peak_value": round(peak_val, 2),
        "knee_omega_decay_index": int(decay_idx),
        "temporal_window": [int(lo), int(hi)],
    }


# --------------------------------------------------------------------------
# 确定性错误引擎
# --------------------------------------------------------------------------
@dataclass
class DeterministicErrorEngine:
    """按固定优先级输出唯一错误代码（或 PASS_STANDARD）。

    优先级遵循动力链「自下而上、由早到晚」：助跑 → 支撑 → 时序减速 → 蓄力 → 击球。
    """

    def evaluate(self, metrics: dict[str, Any]) -> dict[str, Any]:
        lateral = float(metrics.get("support_lateral_dist_cm", 99.0) or 99.0)
        ap = float(metrics.get("support_ap_offset_cm", 0.0) or 0.0)
        ball_dist = float(metrics.get("support_ball_dist_cm", lateral) or lateral)
        fold = float(metrics.get("swing_fold_angle", 180.0) or 180.0)
        thigh = float(metrics.get("thigh_retraction_deg", 0.0) or 0.0)
        ankle_var = float(metrics.get("ankle_variance", 99.0) or 99.0)
        dorsi_drop = float(metrics.get("ankle_dorsiflex_drop_deg", 0.0) or 0.0)
        abduction = float(metrics.get("instep_abduction_deg", 0.0) or 0.0)
        ankle_locked = bool(metrics.get("ankle_locked", False))
        approach = float(metrics.get("approach_angle", 35.0) or 35.0)
        foot_ratio = float(metrics.get("support_foot_ratio", 1.0) or 1.0)
        stance_code = metrics.get("support_stance_code")
        approach_code = metrics.get("approach_error_code")
        early_decel = bool(metrics.get("early_deceleration", False))
        foot_len_m = float(metrics.get("foot_len_m", AVERAGE_CHILD_FOOT_LEN_M) or AVERAGE_CHILD_FOOT_LEN_M)

        too_close = (
            stance_code == ERR_SUPPORT_TOO_CLOSE
            or foot_ratio < SUPPORT_FOOT_RATIO_LOW
            or ball_dist < SUPPORT_WARMUP_CLOSE_CM
            or lateral < SUPPORT_WARMUP_CLOSE_CM
        )
        too_wide = (
            stance_code in (ERR_SUPPORT_TOO_WIDE, ERR_A2_SUPPORT_WIDE)
            or foot_ratio > SUPPORT_FOOT_RATIO_HIGH
        )

        checks: list[tuple[str, bool, str]] = [
            (
                ERR_APPROACH_TOO_STRAIGHT,
                approach_code == ERR_APPROACH_TOO_STRAIGHT or approach < APPROACH_STRAIGHT_DEG,
                f"助跑进袭角 {approach:.1f}° < {APPROACH_STRAIGHT_DEG:.0f}°，直冲冲门线",
            ),
            (
                ERR_APPROACH_TOO_WIDE,
                approach_code == ERR_APPROACH_TOO_WIDE or approach > APPROACH_WIDE_DEG,
                f"助跑进袭角 {approach:.1f}° > {APPROACH_WIDE_DEG:.0f}°，绕圈跑偏离斜线",
            ),
            (
                ERR_WARMUP_CLOSE,
                ball_dist < SUPPORT_WARMUP_CLOSE_CM and foot_ratio >= SUPPORT_FOOT_RATIO_LOW,
                f"支撑脚距球心 {ball_dist:.1f}cm（横距 {lateral:.1f}cm），小于 {SUPPORT_WARMUP_CLOSE_CM:.0f}cm 安全间距",
            ),
            (
                ERR_SUPPORT_TOO_CLOSE,
                too_close and foot_ratio < SUPPORT_FOOT_RATIO_LOW,
                f"支撑横距/脚长比 {foot_ratio:.2f} < {SUPPORT_FOOT_RATIO_LOW:.1f}"
                f"（横距 {lateral:.1f}cm，脚长 {foot_len_m * 100:.1f}cm），踩球槛过近",
            ),
            (
                ERR_A1_SUPPORT_BACK,
                ap < -SUPPORT_BACK_OFFSET_CM,
                f"支撑脚尖落后球心 {abs(ap):.1f}cm，超过 {SUPPORT_BACK_OFFSET_CM:.0f}cm",
            ),
            (
                ERR_A2_SUPPORT_WIDE,
                too_wide,
                f"支撑横距/脚长比 {foot_ratio:.2f} > {SUPPORT_FOOT_RATIO_HIGH:.1f}"
                f"（横距 {lateral:.1f}cm = {foot_ratio:.2f}×脚长），站位偏宽易诱发横向扫把踢",
            ),
            (
                ERR_EARLY_DECELERATION,
                early_decel,
                "摆动腿膝角速度在 T0−5 帧前已显著衰减，动力未彻底释放（提前减速）",
            ),
            (
                ERR_B1_STRAIGHT_LEG,
                fold > BACKSWING_STRAIGHT_LEG_DEG,
                f"backswing_phase 内膝关节极值角 {fold:.1f}°，大于 {BACKSWING_STRAIGHT_LEG_DEG:.0f}°（全程直腿）",
            ),
            (
                ERR_B2_SHANK_ONLY,
                fold <= BACKSWING_STRAIGHT_LEG_DEG
                and fold < 165.0
                and thigh <= THIGH_RETRACTION_NEAR_ZERO_DEG,
                f"backswing_phase 内小腿折叠角 {fold:.1f}° 但大腿后伸（髋后伸代理）仅 {thigh:.1f}°≈0°，只用小腿弹射",
            ),
            (
                ERR_C1_LOOSE_ANKLE,
                (not ankle_locked)
                or ankle_var >= ANKLE_VARIANCE_FAIL
                or dorsi_drop >= ANKLE_DORSIFLEX_DROP_DEG,
                f"击球窗踝角方差 σ²={ankle_var:.2f}，背屈骤降 {dorsi_drop:.1f}°，踝关节锁死={ankle_locked}",
            ),
            (
                ERR_C2_TOE_POKE,
                abduction < TOE_POKE_INSTEP_ALIGN_DEG,
                f"足背外展代理角仅 {abduction:.1f}°，足背未外展、脚尖直捅球体",
            ),
        ]

        for code, hit, reason in checks:
            if hit:
                return {
                    "primary_error_code": code,
                    "error_codes": [code],
                    "pass_standard": False,
                    "decision_reason": reason,
                }

        reason = (
            f"助跑角 {approach:.1f}°，支撑比 {foot_ratio:.2f}（"
            f"{SUPPORT_FOOT_RATIO_LOW:.1f}~{SUPPORT_FOOT_RATIO_HIGH:.1f}），"
            f"膝折叠 {fold:.1f}°，大腿后伸 {thigh:.1f}°，踝锁死={ankle_locked}，"
            f"足背外展 {abduction:.1f}°"
        )
        return {
            "primary_error_code": PASS_STANDARD,
            "error_codes": [],
            "pass_standard": True,
            "decision_reason": reason,
        }


# 错误代码 → 必须截取的相位 / 默认关键帧字段
KEYFRAME_PHASE_BY_ERROR: dict[str, str] = {
    ERR_APPROACH_TOO_STRAIGHT: "approach_phase",
    ERR_APPROACH_TOO_WIDE: "approach_phase",
    ERR_WARMUP_CLOSE: "support_phase",
    ERR_SUPPORT_TOO_CLOSE: "support_phase",
    ERR_A1_SUPPORT_BACK: "support_phase",
    ERR_SUPPORT_TOO_WIDE: "support_phase",
    ERR_A2_SUPPORT_WIDE: "support_phase",
    ERR_EARLY_DECELERATION: "backswing_phase",
    ERR_B1_STRAIGHT_LEG: "backswing_phase",
    ERR_B2_SHANK_ONLY: "backswing_phase",
    ERR_C1_LOOSE_ANKLE: "impact_phase",
    ERR_C2_TOE_POKE: "impact_phase",
    PASS_SUPPORT_OK: "support_phase",
    PASS_STANDARD: "impact_phase",
}


def resolve_keyframe_index(diagnosis: dict[str, Any]) -> int:
    """按错误代码死锁关键帧：B1/B2 必须取 backswing 极值后摆帧。"""
    code = diagnosis.get("primary_error_code") or PASS_STANDARD
    metrics = diagnosis.get("metrics") or {}
    t0 = int(diagnosis.get("t0_index", metrics.get("contact_frame_index", 0)) or 0)

    if code in (ERR_B1_STRAIGHT_LEG, ERR_B2_SHANK_ONLY, ERR_EARLY_DECELERATION):
        idx = int(metrics.get("backswing_extreme_frame_index", t0) or t0)
        if code == ERR_EARLY_DECELERATION:
            idx = int(metrics.get("knee_omega_decay_index", metrics.get("knee_omega_peak_index", idx)) or idx)
        # 硬闸：绝禁止随前帧
        if idx >= t0:
            back_win = metrics.get("backswing_window") or [max(0, t0 - 2), max(0, t0 - 1)]
            idx = int(back_win[1]) if isinstance(back_win, (list, tuple)) and back_win else max(0, t0 - 1)
        return max(0, idx)

    if code in (
        ERR_WARMUP_CLOSE,
        ERR_SUPPORT_TOO_CLOSE,
        ERR_A1_SUPPORT_BACK,
        ERR_SUPPORT_TOO_WIDE,
        ERR_A2_SUPPORT_WIDE,
    ):
        return int(metrics.get("landing_frame_index", t0) or t0)

    if code in (ERR_APPROACH_TOO_STRAIGHT, ERR_APPROACH_TOO_WIDE):
        win = metrics.get("approach_window") or [max(0, t0 - 18), max(0, t0 - 9)]
        if isinstance(win, (list, tuple)) and len(win) >= 2:
            return int(max(0, (int(win[0]) + int(win[1])) // 2))
        return max(0, t0 - 12)

    return t0


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def diagnose_with_temporal_isolation(
    frames: list[dict],
    ball_center: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """对 InstepKickAnalyzer 同构的帧序列执行 T0 锁定 + 相位隔离诊断。

    【强容错】即便有效帧 < 3，也返回带安全 T0（60% 时间线）的结构，
    绝不把 t0_index=None 传给下游。
    """
    if not frames:
        return {
            "ok": False,
            "primary_error_code": None,
            "error_codes": [],
            "metrics": None,
            "t0_index": 0,
            "keyframe_frame_index": 0,
            "keyframe_phase": "impact_phase",
            "phase_windows": None,
            "t0_meta": {"t0_fallback": "empty_frames"},
            "decision_reason": "空帧序列，无法进行时序隔离诊断",
        }

    n = len(frames)
    if n < 3:
        # 极短序列：仍锁定安全 T0，供前端雷达降级渲染
        t0_index = int(max(0, min(n - 1, round((n - 1) * 0.60))))
        return {
            "ok": False,
            "primary_error_code": None,
            "error_codes": [],
            "metrics": None,
            "t0_index": t0_index,
            "keyframe_frame_index": t0_index,
            "keyframe_phase": "impact_phase",
            "phase_windows": None,
            "t0_meta": {"t0_fallback": "sixty_percent_short_clip", "sample_frame_count": n},
            "decision_reason": "有效帧不足，无法进行时序隔离诊断",
        }

    timestamps = [float(f["timestamp_sec"]) for f in frames]
    t0_index, ball, t0_meta = lock_absolute_t0(frames, ball_center=ball_center)
    # 双重保险：lock_absolute_t0 已保证非 None，此处再夹紧一次
    t0_index = int(max(0, min(n - 1, int(t0_index) if t0_index is not None else round((n - 1) * 0.60))))
    phases = build_phase_windows(timestamps, t0_index)
    cm = _estimate_cm_per_pixel(frames[t0_index])
    plan = TemporalIsolationPlan(
        t0_index=t0_index,
        t0_timestamp_sec=float(timestamps[t0_index]),
        ball_center=ball,
        cm_per_pixel=cm,
        phases=phases,
    )

    support = _support_metrics(frames, phases["support_phase"], ball, cm)
    try:
        backswing = _backswing_metrics(frames, plan)
    except BackswingWindowViolation as exc:
        # 不再因后摆窗异常整单失败：用 T0 前安全 3 帧降级指标继续诊断
        safe_end = max(0, t0_index - 1)
        safe_start = max(0, safe_end - MIN_PHASE_WINDOW_FRAMES + 1)
        knee_fallback = [
            calculate_angle(
                frames[i]["right_hip"], frames[i]["right_knee"], frames[i]["right_ankle"]
            )
            for i in range(safe_start, safe_end + 1)
        ] or [180.0]
        local_i = int(np.argmin(np.asarray(knee_fallback, dtype=np.float64)))
        backswing = {
            "swing_fold_angle": round(float(knee_fallback[local_i]), 1),
            "thigh_retraction_deg": 0.0,
            "backswing_extreme_frame_index": int(safe_start + local_i),
            "backswing_window": [int(safe_start), int(safe_end)],
            "backswing_degraded": True,
            "backswing_degrade_reason": str(exc),
        }

    impact = _impact_metrics(frames, plan)
    follow = _follow_through_metrics(frames, plan)
    approach = _approach_metrics(frames, plan, support)
    early = _early_deceleration_metrics(frames, t0_index)

    # 鞭打摆速：仅用 T0 前角速度峰值，避免随前干扰（跳过低置信帧）
    knee_series = []
    for f in frames:
        if _frame_passes_visibility(f, ("right_hip", "right_knee", "right_ankle")):
            hip = _joint_prefer_world(f, "right_hip")
            knee = _joint_prefer_world(f, "right_knee")
            ankle = _joint_prefer_world(f, "right_ankle")
            knee_series.append(calculate_angle(hip, knee, ankle))
        else:
            knee_series.append(
                calculate_angle(f["right_hip"], f["right_knee"], f["right_ankle"])
            )
    omega = _time_derivative_series(knee_series, timestamps)
    pre_t0_omega = [abs(omega[i]) for i in range(0, t0_index + 1)]
    whip = float(max(pre_t0_omega)) if pre_t0_omega else 0.0

    torso_mid_shoulder = 0.5 * (
        _joint_prefer_world(frames[t0_index], "left_shoulder")
        + _joint_prefer_world(frames[t0_index], "right_shoulder")
    )
    torso_mid_hip = 0.5 * (
        _joint_prefer_world(frames[t0_index], "left_hip")
        + _joint_prefer_world(frames[t0_index], "right_hip")
    )
    dx = float(torso_mid_shoulder[0] - torso_mid_hip[0])
    dy = float(torso_mid_shoulder[1] - torso_mid_hip[1])
    if dx == 0.0 and dy == 0.0:
        torso_tilt = 0.0
    elif dy == 0.0:
        torso_tilt = 90.0
    else:
        torso_tilt = float(np.degrees(np.arctan2(abs(dx), abs(dy))))

    metrics: dict[str, Any] = {
        **approach,
        **support,
        **backswing,
        **impact,
        **follow,
        **early,
        "torso_lateral_tilt": round(torso_tilt, 1),
        "whipping_speed_peak": round(whip, 1),
        "contact_frame_index": int(t0_index),
        "sample_frame_count": int(len(frames)),
        "ball_center_px": [round(float(ball[0]), 1), round(float(ball[1]), 1), round(float(ball[2]), 1)],
        "cm_per_pixel": round(float(cm), 4),
    }

    decision = DeterministicErrorEngine().evaluate(metrics)
    diagnosis = {
        "ok": True,
        "primary_error_code": decision["primary_error_code"],
        "error_codes": list(decision["error_codes"]),
        "pass_standard": decision["pass_standard"],
        "decision_reason": decision["decision_reason"],
        "metrics": metrics,
        "t0_index": int(t0_index),
        "t0_meta": t0_meta,
        "phase_windows": {
            name: {
                "start_index": p.start_index,
                "end_index": p.end_index,
                "start_ms_rel": p.start_ms_rel,
                "end_ms_rel": p.end_ms_rel,
            }
            for name, p in phases.items()
        },
    }
    key_idx = resolve_keyframe_index(diagnosis)
    # B1/B2 最终闸门
    if decision["primary_error_code"] in (ERR_B1_STRAIGHT_LEG, ERR_B2_SHANK_ONLY) and key_idx >= t0_index:
        key_idx = int(backswing["backswing_extreme_frame_index"])
        if key_idx >= t0_index:
            key_idx = max(0, t0_index - 1)
    diagnosis["keyframe_frame_index"] = int(key_idx)
    diagnosis["keyframe_phase"] = KEYFRAME_PHASE_BY_ERROR.get(
        decision["primary_error_code"], "impact_phase"
    )

    # V2.5：纯数学确定性总分（LLM 绝不可触碰）
    impact_payload = {
        "t_impact": int(t0_index),
        "frames": frames,
        "distance_cm": metrics.get("support_lateral_dist_cm"),
        "toe_angle": metrics.get("support_toe_angle"),
        "impact_knee_angle": metrics.get("impact_knee_angle"),
        "support_knee_angle": metrics.get("support_knee_angle"),
        "hip_torsion_angle": metrics.get("hip_torsion_angle"),
        "ankle_angles_window": metrics.get("ankle_angles_window"),
    }
    trajectory_payload = {
        "max_folding_angle": metrics.get("max_folding_angle"),
        "whipping_velocity": metrics.get("whipping_speed_peak"),
        "swing_fold_angle": metrics.get("swing_fold_angle"),
    }
    # 补齐触球膝角 / 脚尖角 / 髋扭转 / 折叠角（若 metrics 尚未写入）
    try:
        t0_rec = frames[t0_index]
        if impact_payload.get("impact_knee_angle") is None:
            impact_payload["impact_knee_angle"] = calculate_angle(
                t0_rec["right_hip"], t0_rec["right_knee"], t0_rec["right_ankle"]
            )
        if impact_payload.get("hip_torsion_angle") is None:
            impact_payload["hip_torsion_angle"] = _hip_relative_torsion_deg(t0_rec)
        if impact_payload.get("toe_angle") is None:
            impact_payload["toe_angle"] = _support_toe_angle_deg(t0_rec, ball)
        fold_interior = float(metrics.get("swing_fold_angle", 180.0) or 180.0)
        # 后摆折叠角 = 相对伸直的屈曲量（度）
        trajectory_payload["max_folding_angle"] = float(
            metrics.get("max_folding_angle")
            if metrics.get("max_folding_angle") is not None
            else max(0.0, 180.0 - fold_interior)
        )
    except Exception:
        pass

    total_score, score_detail = DeterministicScorer().calculate_biomechanical_score(
        impact_payload, trajectory_payload
    )
    diagnosis["TotalScore"] = total_score
    diagnosis["score_detail"] = score_detail
    diagnosis["t_impact"] = int(t0_index)
    metrics["max_folding_angle"] = round(
        float(score_detail["indicators"]["max_folding_angle"]["value"]), 1
    )
    metrics["toe_angle"] = round(float(score_detail["indicators"]["toe_angle"]["value"]), 1)
    metrics["ankle_rigidity_variance"] = round(
        float(score_detail["indicators"]["ankle_rigidity"]["variance"]), 2
    )
    metrics["impact_knee_angle"] = round(
        float(score_detail["indicators"]["impact_knee_angle"]["value"]), 1
    )
    metrics["hip_torsion_angle"] = round(
        float(score_detail["indicators"]["hip_torsion_angle"]["value"]), 1
    )
    diagnosis["metrics"] = metrics
    return diagnosis


# --------------------------------------------------------------------------
# V2.5 确定性科研级纯数学评分引擎（LLM 零参与）
# --------------------------------------------------------------------------
STATUS_GREEN = "GREEN_OPTIMAL"
STATUS_YELLOW = "YELLOW_APPROACHING"
STATUS_RED = "RED_DEVIATED"

# 各项最高扣分（合计 91，最差分不低于 9.00）
_MAX_PENALTY_DISTANCE_CM = 12.0
_MAX_PENALTY_TOE_ANGLE = 10.0
_MAX_PENALTY_FOLDING = 12.0
_MAX_PENALTY_WHIPPING = 10.0
_MAX_PENALTY_IMPACT_KNEE = 12.0
_MAX_PENALTY_ANKLE = 15.0  # 脚踝锁紧度：方差 > 5.0 直接扣满分
_MAX_PENALTY_SUPPORT_KNEE = 10.0
_MAX_PENALTY_HIP_TORSION = 10.0

# 脚踝锁紧方差阈值（用户规格）
ANKLE_VARIANCE_GREEN = 2.0
ANKLE_VARIANCE_YELLOW_HIGH = 5.0

# 【V2.5 科研级】触球核心动作窗口：前后各 30 帧（约 1s@30fps），固定约 60 帧
# 所有极值/方差量纲只准在此窗口内解算，杜绝「300 帧截断 vs 414 帧完整」漂移。
ACTION_ROI_HALF_FRAMES: int = 30


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


def _roi_max_folding_angle(
    frames: list[dict], t_impact: int, roi_start: int, roi_end: int
) -> tuple[float, int]:
    """仅在 ROI 内、触球前（含触球）计算后摆最大折叠角 = 180 - min(膝内角)。

    返回 (折叠角, 膝内角最小的物理极值帧索引)。
    """
    fallback_idx = int(max(roi_start, min(max(roi_start, roi_end - 1), t_impact)))
    if not frames:
        return 80.0, fallback_idx
    t = int(max(roi_start, min(roi_end - 1, t_impact)))
    best_fold = None
    best_idx = fallback_idx
    for i in range(roi_start, min(roi_end, t + 1)):
        try:
            rec = frames[i]
            knee = float(calculate_angle(rec["right_hip"], rec["right_knee"], rec["right_ankle"]))
            fold = float(max(0.0, 180.0 - knee))
            if best_fold is None or fold > best_fold:
                best_fold = fold
                best_idx = int(i)
        except Exception:
            continue
    if best_fold is None:
        return 80.0, fallback_idx
    return float(best_fold), int(best_idx)


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
                    float(calculate_angle(rec["right_hip"], rec["right_knee"], rec["right_ankle"]))
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


def print_golden_audit_log(
    task_id: str,
    knee_angle_count: int,
    impact_frame_idx: int,
    final_score: float,
) -> None:
    """控制台黄金审计日志：验证同一视频反复测试的确定性。"""
    lines = [
        f"=== [V2.5 审计日志] 任务 ID: {task_id} ===",
        f"1. 有效参与推理总帧数: {knee_angle_count} 帧",
        f"2. 绝对锁定触球帧索引 (t_impact): Frame #{impact_frame_idx}",
        f"3. 确定性打分引擎最终得分: {final_score:.2f} 分",
        "==========================================",
    ]
    for line in lines:
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            print(line.encode(encoding, errors="replace").decode(encoding, errors="replace"), flush=True)


def _hip_relative_torsion_deg(frame_record: dict) -> float:
    """髋关节相对扭转角：骨盆连线与肩带连线在水平面的夹角（度）。"""
    lh = _as_vec3(frame_record["left_hip"])
    rh = _as_vec3(frame_record["right_hip"])
    ls = _as_vec3(frame_record["left_shoulder"])
    rs = _as_vec3(frame_record["right_shoulder"])
    pelvis = np.array([float(rh[0] - lh[0]), float(rh[2] - lh[2])], dtype=np.float64)
    shoulder = np.array([float(rs[0] - ls[0]), float(rs[2] - ls[2])], dtype=np.float64)
    np_ = float(np.linalg.norm(pelvis))
    ns = float(np.linalg.norm(shoulder))
    if np_ < 1e-9 or ns < 1e-9:
        # 像素平面回退（无可靠 Z 时用 X-Y）
        pelvis = np.array([float(rh[0] - lh[0]), float(rh[1] - lh[1])], dtype=np.float64)
        shoulder = np.array([float(rs[0] - ls[0]), float(rs[1] - ls[1])], dtype=np.float64)
        np_ = float(np.linalg.norm(pelvis))
        ns = float(np.linalg.norm(shoulder))
        if np_ < 1e-9 or ns < 1e-9:
            return 0.0
    cos_v = float(np.clip(np.dot(pelvis, shoulder) / (np_ * ns), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_v)))


def _support_toe_angle_deg(frame_record: dict, ball_center=None) -> float:
    """支撑脚尖指向角：支撑足（左）跟→尖 相对指向球心方向的偏角（度）。"""
    heel = _as_vec3(frame_record["left_heel"] if "left_heel" in frame_record else frame_record["left_ankle"])
    toe = _as_vec3(frame_record["left_foot_index"])
    foot_dir = toe - heel
    if ball_center is not None:
        target = _as_vec3(ball_center) - heel
    else:
        # 无球心：以画面前向（+X）为参考
        target = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    # 投影到水平面（X-Z，缺失 Z 则用 X-Y）
    fd = np.array([float(foot_dir[0]), float(foot_dir[2])], dtype=np.float64)
    tg = np.array([float(target[0]), float(target[2])], dtype=np.float64)
    if float(np.linalg.norm(fd)) < 1e-9 or float(np.linalg.norm(tg)) < 1e-9:
        fd = np.array([float(foot_dir[0]), float(foot_dir[1])], dtype=np.float64)
        tg = np.array([float(target[0]), float(target[1])], dtype=np.float64)
    nf = float(np.linalg.norm(fd))
    nt = float(np.linalg.norm(tg))
    if nf < 1e-9 or nt < 1e-9:
        return 0.0
    cos_v = float(np.clip(np.dot(fd, tg) / (nf * nt), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_v)))


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
) -> list[float]:
    """提取 t_impact 前后各 1 帧（共 3 点）的踝关节夹角。"""
    if precomputed is not None and len(precomputed) >= 3:
        return [float(precomputed[0]), float(precomputed[1]), float(precomputed[2])]
    n = len(frames) if frames else 0
    if n == 0:
        return [140.0, 140.0, 140.0]
    t = int(max(0, min(n - 1, t_impact)))
    idxs = [max(0, t - 1), t, min(n - 1, t + 1)]
    angles: list[float] = []
    for i in idxs:
        rec = frames[i]
        try:
            angles.append(
                calculate_angle(rec["right_knee"], rec["right_ankle"], rec["right_foot_index"])
            )
        except Exception:
            angles.append(140.0)
    # 保证恰好 3 点（极端短序列时复制）
    while len(angles) < 3:
        angles.append(angles[-1] if angles else 140.0)
    return angles[:3]


class DeterministicScorer:
    """V2.5 纯数学生物力学评分器。

    严禁 LLM / 随机源参与任何扣分或等级判定。总分自 100.00 起按量纲线性扣减，
    保留两位小数；同一输入保证浮点误差为 0.0 的位级可复现。

    【V2.5 Action ROI】所有极值/方差量纲仅在 t_impact ± 30 帧核心窗口内解算。
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

        # ---- a) 支撑脚偏移 distance_cm：[15, 20] GREEN ----
        # 瞬时几何量：取触球帧（ROI 内），不依赖全长序列极值
        distance_cm = 17.5
        for key_src in (impact_frame_data, trajectory_data):
            if key_src.get("distance_cm") is not None:
                distance_cm = float(key_src["distance_cm"])
                break
            if key_src.get("support_lateral_dist_cm") is not None:
                distance_cm = float(key_src["support_lateral_dist_cm"])
                break
        pen_dist, st_dist = _linear_band_penalty(
            distance_cm, 15.0, 20.0, 10.0, 25.0, _MAX_PENALTY_DISTANCE_CM
        )

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
        if 0.0 <= toe_angle <= 15.0:
            pen_toe, st_toe = 0.0, STATUS_GREEN
        elif toe_angle > 25.0:
            pen_toe, st_toe = float(_MAX_PENALTY_TOE_ANGLE), STATUS_RED
        else:
            ratio = (toe_angle - 15.0) / 10.0
            pen_toe = round(min(_MAX_PENALTY_TOE_ANGLE, _MAX_PENALTY_TOE_ANGLE * ratio), 2)
            st_toe = STATUS_YELLOW

        # ---- c) 摆动腿后摆折叠角：【仅 ROI 内】解算；无帧时回退预计算标量 ----
        fold_extreme_idx = int(t_impact)
        if frames:
            max_folding, fold_extreme_idx = _roi_max_folding_angle(
                frames, t_impact, roi_start, roi_end
            )
        else:
            max_folding = trajectory_data.get("max_folding_angle")
            if max_folding is None and trajectory_data.get("swing_fold_angle") is not None:
                max_folding = max(0.0, 180.0 - float(trajectory_data["swing_fold_angle"]))
            if max_folding is None:
                max_folding = 80.0
            max_folding = float(max_folding)
            fold_extreme_idx = int(
                trajectory_data.get(
                    "backswing_extreme_frame_index",
                    impact_frame_data.get("backswing_extreme_frame_index", max(0, t_impact - 8)),
                )
                or max(0, t_impact - 8)
            )
        pen_fold, st_fold = _linear_band_penalty(
            max_folding, 70.0, 90.0, 55.0, 105.0, _MAX_PENALTY_FOLDING
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
        if whipping >= 450.0:
            pen_whip, st_whip = 0.0, STATUS_GREEN
        elif whipping >= 320.0:
            ratio = (450.0 - whipping) / 130.0
            pen_whip = round(min(_MAX_PENALTY_WHIPPING, _MAX_PENALTY_WHIPPING * 0.55 * ratio), 2)
            st_whip = STATUS_YELLOW
        else:
            ratio = min(1.0, (320.0 - whipping) / 320.0)
            pen_whip = round(
                min(_MAX_PENALTY_WHIPPING, _MAX_PENALTY_WHIPPING * (0.55 + 0.45 * ratio)), 2
            )
            st_whip = STATUS_RED

        # ---- e) 触球瞬间膝关节夹角（触球帧，属 ROI）----
        impact_knee = impact_frame_data.get("impact_knee_angle")
        if impact_knee is None and impact_rec is not None:
            try:
                impact_knee = calculate_angle(
                    impact_rec["right_hip"], impact_rec["right_knee"], impact_rec["right_ankle"]
                )
            except Exception:
                impact_knee = 150.0
        if impact_knee is None:
            impact_knee = 150.0
        impact_knee = float(impact_knee)
        pen_iknee, st_iknee = _linear_band_penalty(
            impact_knee, 140.0, 160.0, 125.0, 175.0, _MAX_PENALTY_IMPACT_KNEE
        )

        # ---- f) 脚踝锁紧度：t±1 三帧（落在 ROI 内）----
        ankle_angles = _extract_ankle_window_angles(
            frames,
            t_impact,
            precomputed=impact_frame_data.get("ankle_angles_window")
            or trajectory_data.get("ankle_angles_window"),
        )
        ankle_variance = float(np.var(np.asarray(ankle_angles, dtype=np.float64)))
        if ankle_variance < ANKLE_VARIANCE_GREEN:
            pen_ankle, st_ankle = 0.0, STATUS_GREEN
        elif ankle_variance <= ANKLE_VARIANCE_YELLOW_HIGH:
            ratio = (ankle_variance - ANKLE_VARIANCE_GREEN) / (
                ANKLE_VARIANCE_YELLOW_HIGH - ANKLE_VARIANCE_GREEN
            )
            pen_ankle = round(min(_MAX_PENALTY_ANKLE, _MAX_PENALTY_ANKLE * 0.55 * ratio), 2)
            st_ankle = STATUS_YELLOW
        else:
            pen_ankle, st_ankle = float(_MAX_PENALTY_ANKLE), STATUS_RED

        # ---- g1) 支撑腿膝关节角度（触球帧）----
        support_knee = impact_frame_data.get(
            "support_knee_angle", trajectory_data.get("support_knee_angle")
        )
        if support_knee is None and impact_rec is not None:
            try:
                support_knee = calculate_angle(
                    impact_rec["left_hip"], impact_rec["left_knee"], impact_rec["left_ankle"]
                )
            except Exception:
                support_knee = 155.0
        if support_knee is None:
            support_knee = 155.0
        support_knee = float(support_knee)
        pen_sknee, st_sknee = _linear_band_penalty(
            support_knee, 140.0, 165.0, 125.0, 175.0, _MAX_PENALTY_SUPPORT_KNEE
        )

        # ---- g2) 髋关节相对扭转角（触球帧）----
        hip_torsion = impact_frame_data.get(
            "hip_torsion_angle", trajectory_data.get("hip_torsion_angle")
        )
        if hip_torsion is None and impact_rec is not None:
            try:
                hip_torsion = _hip_relative_torsion_deg(impact_rec)
            except Exception:
                hip_torsion = 25.0
        if hip_torsion is None:
            hip_torsion = 25.0
        hip_torsion = float(hip_torsion)
        pen_hip, st_hip = _linear_band_penalty(
            hip_torsion, 15.0, 40.0, 5.0, 55.0, _MAX_PENALTY_HIP_TORSION
        )

        total_penalty = (
            pen_dist
            + pen_toe
            + pen_fold
            + pen_whip
            + pen_iknee
            + pen_ankle
            + pen_sknee
            + pen_hip
        )
        total_score = round(max(0.0, 100.00 - float(total_penalty)), 2)

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

        indicators = {
            "distance_cm": {
                "value": round(distance_cm, 2),
                "unit": "cm",
                "status": st_dist,
                "penalty": pen_dist,
                "green_band": [15.0, 20.0],
                "extreme_frame_index": int(landing_idx),
            },
            "toe_angle": {
                "value": round(toe_angle, 2),
                "unit": "deg",
                "status": st_toe,
                "penalty": pen_toe,
                "green_band": [0.0, 15.0],
                "extreme_frame_index": int(landing_idx),
            },
            "max_folding_angle": {
                "value": round(max_folding, 2),
                "unit": "deg",
                "status": st_fold,
                "penalty": pen_fold,
                "green_band": [70.0, 90.0],
                "extreme_frame_index": int(fold_extreme_idx),
            },
            "whipping_velocity": {
                "value": round(whipping, 2),
                "unit": "deg/s",
                "status": st_whip,
                "penalty": pen_whip,
                "green_band": [450.0, None],
                "extreme_frame_index": int(whip_extreme_idx),
            },
            "impact_knee_angle": {
                "value": round(impact_knee, 2),
                "unit": "deg",
                "status": st_iknee,
                "penalty": pen_iknee,
                "green_band": [140.0, 160.0],
                "extreme_frame_index": int(t_impact),
            },
            "ankle_rigidity": {
                "value": round(ankle_variance, 4),
                "variance": round(ankle_variance, 4),
                "ankle_angles_window": [round(a, 2) for a in ankle_angles],
                "unit": "variance",
                "status": st_ankle,
                "penalty": pen_ankle,
                "green_band": [0.0, ANKLE_VARIANCE_GREEN],
                "extreme_frame_index": int(t_impact),
            },
            "support_knee_angle": {
                "value": round(support_knee, 2),
                "unit": "deg",
                "status": st_sknee,
                "penalty": pen_sknee,
                "green_band": [140.0, 165.0],
                "extreme_frame_index": int(landing_idx),
            },
            "hip_torsion_angle": {
                "value": round(hip_torsion, 2),
                "unit": "deg",
                "status": st_hip,
                "penalty": pen_hip,
                "green_band": [15.0, 40.0],
                "extreme_frame_index": int(t_impact),
            },
        }

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

        detail = {
            "TotalScore": total_score,
            "t_impact": int(t_impact),
            "base_score": 100.00,
            "total_penalty": round(float(total_penalty), 2),
            "indicators": indicators,
            "metric_extreme_frames": metric_extreme_frames,
            "radar_scores": radar_scores,
            "scoring_engine": "DeterministicScorer_V3.1",
            "llm_participated": False,
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
        }
        return total_score, detail

    @staticmethod
    def _clamp_radar(value: float) -> float:
        """雷达维分数：[0, 20]，保留 1 位小数。"""
        return round(max(0.0, min(20.0, float(value))), 1)

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
        - ankle_rigidity：脚踝方差分档（<2→20，[2,5]→15，>5→5）
        - whipping_velocity：小腿峰值角速度（>=450→20，否则线性递减）
        - approach_rhythm：助跑占位，由整体流畅度映射到 16–20（确定性，零随机）
        """
        # 支撑与稳固：两路惩罚按各自满分权重折算到 20 分制
        support_denom = _MAX_PENALTY_DISTANCE_CM + _MAX_PENALTY_SUPPORT_KNEE
        support_stability = self._clamp_radar(
            20.0 * (1.0 - (float(pen_dist) + float(pen_sknee)) / support_denom)
        )

        # 蓄力与折叠：折叠惩罚满扣 → 0 分
        backswing_folding = self._clamp_radar(
            20.0 * (1.0 - float(pen_fold) / _MAX_PENALTY_FOLDING)
        )

        # 锁踝与刚性：离散档位（与 ANKLE_VARIANCE_* 阈值对齐）
        if ankle_variance < ANKLE_VARIANCE_GREEN:
            ankle_rigidity = 20.0
        elif ankle_variance <= ANKLE_VARIANCE_YELLOW_HIGH:
            ankle_rigidity = 15.0
        else:
            ankle_rigidity = 5.0

        # 鞭打与随摆：>=450 → 满分，否则按比例递减至 0
        if whipping >= 450.0:
            whipping_velocity = 20.0
        else:
            whipping_velocity = self._clamp_radar((float(whipping) / 450.0) * 20.0)

        # 助跑与进袭：占位符 —— 用总惩罚映射流畅度到 [16, 20]，保持确定性可复现
        # total_penalty≈0 → 20；惩罚升高逐步贴近 16；永不低于 16（鼓励性保底）
        approach_rhythm = self._clamp_radar(
            max(16.0, min(20.0, 20.0 - float(total_penalty) * 0.05))
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


# ==========================================================================
# Sprint 1 · 支撑脚与摆腿时空运动轨迹热力图
# --------------------------------------------------------------------------
# 坐标系约定（俯视虚拟球场，单位：厘米）：
#   - 触球瞬间 t_impact 的球心 ball_center 为绝对原点 (0, 0)
#   - dx > 0：相对球心向右（运动员横向外侧，取决于机位）；dy > 0：相对球心向前
#   - 世界坐标（米）→ cm：×100；优先使用 pose_world_landmarks 的水平面 (X, Z)
#   - 像素坐标 → cm：× cm_per_pixel（由肩宽估算），水平面用 (X, Y) 近似
#
# 画布映射（OpenCV 热力图，数学必须自洽）：
#   - 画布尺寸 HEATMAP_CANVAS_SIZE × HEATMAP_CANVAS_SIZE（默认 800×800）
#   - 像素原点 HEATMAP_ORIGIN_PX = (400, 400) 对应物理原点 (0, 0) cm（球心）
#   - 比例尺 HEATMAP_CM_PER_PX = 0.5  ⇒  1 px = 0.5 cm  ⇒  1 cm = 2 px
#   - 正向映射：
#         px = ORIGIN_X + dx_cm / HEATMAP_CM_PER_PX
#         py = ORIGIN_Y - dy_cm / HEATMAP_CM_PER_PX   # 图像 Y 向下，故取负号使 +dy 朝上
#   - 逆映射（校验用）：
#         dx_cm = (px - ORIGIN_X) * HEATMAP_CM_PER_PX
#         dy_cm = (ORIGIN_Y - py) * HEATMAP_CM_PER_PX
# ==========================================================================

# 摆动腿轨迹回溯帧数：闭区间 [t_impact - N, t_impact]
SWING_TRAJECTORY_PRE_FRAMES: int = 15

# 热力图画布与比例尺
HEATMAP_CANVAS_SIZE: int = 800
HEATMAP_ORIGIN_PX: tuple[int, int] = (400, 400)
HEATMAP_CM_PER_PX: float = 0.5  # 1 px = 0.5 cm
HEATMAP_PX_PER_CM: float = 1.0 / HEATMAP_CM_PER_PX  # = 2.0 px/cm

# 支撑脚发光点 / 摆腿光流样式（BGR）
HEATMAP_SUPPORT_COLOR_BGR: tuple[int, int, int] = (40, 80, 255)  # 暖红热力核
HEATMAP_SWING_COLOR_BGR: tuple[int, int, int] = (80, 255, 120)  # 青绿出腿光流
HEATMAP_BALL_COLOR_BGR: tuple[int, int, int] = (220, 220, 220)  # 球心中性白
HEATMAP_SUPPORT_RADIUS_PX: int = 18
HEATMAP_SWING_THICKNESS_PX: int = 3
HEATMAP_GAUSSIAN_KSIZE: tuple[int, int] = (31, 31)
HEATMAP_GAUSSIAN_SIGMA: float = 9.0


def _joint_xy_for_heatmap(frame_record: dict, joint: str) -> Optional[np.ndarray]:
    """取关节点用于俯视热力图的水平 2D 坐标（原始单位：米或像素）。

    优先世界坐标 (x, z)；无世界坐标时回退图像平面 (x, y)。
    """
    if not isinstance(frame_record, dict):
        return None
    world = _world_joint(frame_record, joint)
    if world is not None and np.all(np.isfinite(world)):
        # 俯视：MediaPipe 世界系 X=横向，Z=前后（Y 为垂直轴，丢弃）
        return np.array([float(world[0]), float(world[2])], dtype=np.float64)
    try:
        raw = _as_vec3(frame_record[joint])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite(raw)):
        return None
    # 像素 / 归一化平面：用 (x, y) 近似俯视投影
    return np.array([float(raw[0]), float(raw[1])], dtype=np.float64)


def _scale_delta_to_cm(
    delta_xy: np.ndarray,
    *,
    coord_space: str,
    cm_per_pixel: float,
) -> tuple[float, float]:
    """把相对位移向量从原始单位换算为厘米。

    - world_m：原始单位为米 → ×100
    - image_px：原始单位为像素 → × cm_per_pixel
    - normalized：近似按像素比例尺处理（调用方应尽量避免）
    """
    dx, dy = float(delta_xy[0]), float(delta_xy[1])
    if coord_space == "world_m":
        return dx * 100.0, dy * 100.0
    scale = float(cm_per_pixel) if cm_per_pixel > 1e-12 else 1.0
    return dx * scale, dy * scale


def _detect_heatmap_coord_space(frame_record: dict) -> str:
    """判定本帧热力图应使用的坐标空间。"""
    if _has_world_landmarks(frame_record):
        return "world_m"
    return "image_px"


def physical_cm_to_heatmap_px(
    dx_cm: float,
    dy_cm: float,
    *,
    origin_px: tuple[int, int] = HEATMAP_ORIGIN_PX,
    cm_per_px: float = HEATMAP_CM_PER_PX,
) -> tuple[int, int]:
    """物理相对坐标 (dx_cm, dy_cm) → 热力图像素 (px, py)。

    数学：
        px = origin_x + dx_cm / cm_per_px
        py = origin_y - dy_cm / cm_per_px
    其中 cm_per_px=0.5 ⇒ /0.5 ≡ ×2，即 1 cm = 2 px。
    """
    ox, oy = int(origin_px[0]), int(origin_px[1])
    inv = 1.0 / float(cm_per_px) if cm_per_px > 1e-12 else HEATMAP_PX_PER_CM
    px = int(round(ox + float(dx_cm) * inv))
    py = int(round(oy - float(dy_cm) * inv))
    return px, py


def heatmap_px_to_physical_cm(
    px: int,
    py: int,
    *,
    origin_px: tuple[int, int] = HEATMAP_ORIGIN_PX,
    cm_per_px: float = HEATMAP_CM_PER_PX,
) -> tuple[float, float]:
    """热力图像素 → 物理相对坐标（逆映射，供单测 / 注释校验）。"""
    ox, oy = float(origin_px[0]), float(origin_px[1])
    dx_cm = (float(px) - ox) * float(cm_per_px)
    dy_cm = (oy - float(py)) * float(cm_per_px)
    return dx_cm, dy_cm


def generate_spatial_trajectory(
    pose_landmarks_sequence: list,
    t_impact: int,
    ball_center_t_impact,
) -> dict[str, Any]:
    """以触球瞬间球心为绝对原点，提取支撑脚相对坐标与摆动腿 15 帧轨迹。

    参数：
        pose_landmarks_sequence：逐帧关键点字典列表（与 error_diagnoser 帧结构兼容；
            至少含 left_ankle / right_ankle；可选 world 子字典）。
        t_impact：触球绝对零点帧索引。
        ball_center_t_impact：t_impact 时球心绝对坐标（与关节点同坐标系）。
            若为 None，则回退为该帧 right_foot_index（足背触球锚点）。

    返回：
        {
          "dx_support": float,          # 支撑脚踝相对球心 X（cm）
          "dy_support": float,          # 支撑脚踝相对球心 Y/Z（cm）
          "support_rel": [dx, dy],
          "swing_trajectory": [[dx, dy], ...],  # [t_impact-15, t_impact] 共至多 16 点
          "ball_origin_cm": [0.0, 0.0],
          "t_impact": int,
          "window": [start, end],       # 闭区间帧下标
          "coord_space": str,
          "cm_per_pixel": float | None,
          "scale": {
              "cm_per_px": 0.5,
              "px_per_cm": 2.0,
              "canvas_size": 800,
              "origin_px": [400, 400],
          },
        }
    """
    frames = list(pose_landmarks_sequence or [])
    n = len(frames)
    empty = {
        "dx_support": 0.0,
        "dy_support": 0.0,
        "support_rel": [0.0, 0.0],
        "swing_trajectory": [],
        "ball_origin_cm": [0.0, 0.0],
        "t_impact": int(t_impact) if t_impact is not None else 0,
        "window": [0, 0],
        "coord_space": "empty",
        "cm_per_pixel": None,
        "scale": {
            "cm_per_px": HEATMAP_CM_PER_PX,
            "px_per_cm": HEATMAP_PX_PER_CM,
            "canvas_size": HEATMAP_CANVAS_SIZE,
            "origin_px": [HEATMAP_ORIGIN_PX[0], HEATMAP_ORIGIN_PX[1]],
        },
    }
    if n <= 0:
        return empty

    t = int(max(0, min(n - 1, int(t_impact))))
    impact_rec = frames[t] if isinstance(frames[t], dict) else {}
    coord_space = _detect_heatmap_coord_space(impact_rec)
    cm_per_pixel = float(_estimate_cm_per_pixel(impact_rec)) if impact_rec else 1.0

    # ---- 球心绝对原点：强制锚定为 t_impact 时刻 ----
    ball_abs = None
    if ball_center_t_impact is not None:
        try:
            ball_vec = _as_vec3(ball_center_t_impact)
            if coord_space == "world_m":
                ball_abs = np.array([float(ball_vec[0]), float(ball_vec[2])], dtype=np.float64)
            else:
                ball_abs = np.array([float(ball_vec[0]), float(ball_vec[1])], dtype=np.float64)
        except (TypeError, ValueError):
            ball_abs = None
    if ball_abs is None:
        # 与 lock_absolute_t0 一致：右足尖作为球心逼近锚点
        ball_abs = _joint_xy_for_heatmap(impact_rec, "right_foot_index")
        if ball_abs is None:
            ball_abs = _joint_xy_for_heatmap(impact_rec, "right_ankle")
    if ball_abs is None or not np.all(np.isfinite(ball_abs)):
        return {**empty, "t_impact": t, "coord_space": coord_space, "cm_per_pixel": round(cm_per_pixel, 4)}

    # ---- 支撑脚踝 @ t_impact：相对球心 (dx_support, dy_support) ----
    support_abs = _joint_xy_for_heatmap(impact_rec, "left_ankle")
    if support_abs is None:
        dx_support, dy_support = 0.0, 0.0
    else:
        dx_support, dy_support = _scale_delta_to_cm(
            support_abs - ball_abs, coord_space=coord_space, cm_per_pixel=cm_per_pixel
        )

    # ---- 摆动腿踝轨迹：[t_impact-15, t_impact] 相对同一球心 ----
    win_start = max(0, t - int(SWING_TRAJECTORY_PRE_FRAMES))
    win_end = t  # 闭区间右端
    swing_trajectory: list[list[float]] = []
    for i in range(win_start, win_end + 1):
        rec = frames[i] if isinstance(frames[i], dict) else {}
        swing_abs = _joint_xy_for_heatmap(rec, "right_ankle")
        if swing_abs is None or not np.all(np.isfinite(swing_abs)):
            continue
        dx_s, dy_s = _scale_delta_to_cm(
            swing_abs - ball_abs, coord_space=coord_space, cm_per_pixel=cm_per_pixel
        )
        swing_trajectory.append([round(float(dx_s), 2), round(float(dy_s), 2)])

    return {
        "dx_support": round(float(dx_support), 2),
        "dy_support": round(float(dy_support), 2),
        "support_rel": [round(float(dx_support), 2), round(float(dy_support), 2)],
        "swing_trajectory": swing_trajectory,
        "ball_origin_cm": [0.0, 0.0],
        "t_impact": int(t),
        "window": [int(win_start), int(win_end)],
        "coord_space": coord_space,
        "cm_per_pixel": round(float(cm_per_pixel), 4) if coord_space != "world_m" else None,
        "scale": {
            "cm_per_px": HEATMAP_CM_PER_PX,
            "px_per_cm": HEATMAP_PX_PER_CM,
            "canvas_size": HEATMAP_CANVAS_SIZE,
            "origin_px": [HEATMAP_ORIGIN_PX[0], HEATMAP_ORIGIN_PX[1]],
        },
    }


def render_spatial_heatmap_base64(
    dx_support: float,
    dy_support: float,
    swing_trajectory: list,
    *,
    accumulate_on: Optional[np.ndarray] = None,
    canvas_size: int = HEATMAP_CANVAS_SIZE,
) -> tuple[str, np.ndarray]:
    """用 OpenCV 将单趟次支撑点 + 摆腿轨迹渲染为高斯模糊热力云图，并 base64 编码。

    - 空白底图：纯黑 ``uint8`` 零矩阵 ``(H, W, 3)``
    - 球心 (0,0) cm → 像素 (400, 400)；比例尺 1 px = 0.5 cm
    - 支撑脚：暖红发光圆，经高斯模糊形成热力斑
    - 摆腿轨迹：折线/平滑曲线（青色），再整体高斯模糊模拟出腿光流
    - ``accumulate_on``：若传入既有画布，则在其上叠加（多趟次亮度累加）

    返回 ``(heatmap_base64, canvas_bgr)``；base64 为纯 PNG 字节串（无 data URI 前缀）。
    """
    import base64

    import cv2

    size = int(max(64, canvas_size))
    if accumulate_on is not None and isinstance(accumulate_on, np.ndarray) and accumulate_on.shape[:2] == (size, size):
        canvas = accumulate_on.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    else:
        # 纯黑底图：NumPy zero array，原点稍后映射到中心像素
        canvas = np.zeros((size, size, 3), dtype=np.uint8)

    origin = (
        int(round(size * HEATMAP_ORIGIN_PX[0] / float(HEATMAP_CANVAS_SIZE))),
        int(round(size * HEATMAP_ORIGIN_PX[1] / float(HEATMAP_CANVAS_SIZE))),
    )
    cm_per_px = HEATMAP_CM_PER_PX

    # ---- 图层：支撑脚热力核（先画到独立层再模糊，避免污染轨迹层）----
    support_layer = np.zeros_like(canvas)
    spx, spy = physical_cm_to_heatmap_px(dx_support, dy_support, origin_px=origin, cm_per_px=cm_per_px)
    if 0 <= spx < size and 0 <= spy < size:
        cv2.circle(
            support_layer,
            (spx, spy),
            HEATMAP_SUPPORT_RADIUS_PX,
            HEATMAP_SUPPORT_COLOR_BGR,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        # 内核高亮：叠加多次测试时亮度自然饱和向白
        cv2.circle(
            support_layer,
            (spx, spy),
            max(3, HEATMAP_SUPPORT_RADIUS_PX // 3),
            (180, 200, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    support_blur = cv2.GaussianBlur(support_layer, HEATMAP_GAUSSIAN_KSIZE, HEATMAP_GAUSSIAN_SIGMA)

    # ---- 图层：摆动腿出腿光流轨迹 ----
    swing_layer = np.zeros_like(canvas)
    pts: list[tuple[int, int]] = []
    for point in swing_trajectory or []:
        if not point or len(point) < 2:
            continue
        try:
            qx, qy = physical_cm_to_heatmap_px(
                float(point[0]), float(point[1]), origin_px=origin, cm_per_px=cm_per_px
            )
        except (TypeError, ValueError):
            continue
        if 0 <= qx < size and 0 <= qy < size:
            pts.append((qx, qy))
            cv2.circle(swing_layer, (qx, qy), 4, HEATMAP_SWING_COLOR_BGR, thickness=-1, lineType=cv2.LINE_AA)

    if len(pts) >= 2:
        # 折线连接；点足够时用近似平滑（polylines + 轻度膨胀）
        arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            swing_layer,
            [arr],
            isClosed=False,
            color=HEATMAP_SWING_COLOR_BGR,
            thickness=HEATMAP_SWING_THICKNESS_PX,
            lineType=cv2.LINE_AA,
        )
        # 触球端点加亮，强调 t_impact 踝关节落点
        cv2.circle(swing_layer, pts[-1], 6, (200, 255, 200), thickness=-1, lineType=cv2.LINE_AA)
    swing_blur = cv2.GaussianBlur(swing_layer, HEATMAP_GAUSSIAN_KSIZE, HEATMAP_GAUSSIAN_SIGMA)

    # ---- 叠加（饱和加法，多趟次反复叠加会提高亮度）----
    canvas = cv2.add(canvas, support_blur)
    canvas = cv2.add(canvas, swing_blur)

    # 球心十字准星（不模糊，保持几何锚点清晰）
    ox, oy = origin
    cv2.drawMarker(
        canvas,
        (ox, oy),
        HEATMAP_BALL_COLOR_BGR,
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=1,
        line_type=cv2.LINE_AA,
    )
    cv2.circle(canvas, (ox, oy), 5, HEATMAP_BALL_COLOR_BGR, thickness=1, lineType=cv2.LINE_AA)

    ok, buffer = cv2.imencode(".png", canvas)
    if not ok:
        return "", canvas
    heatmap_base64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    return heatmap_base64, canvas


def build_spatial_heatmap_payload(
    pose_landmarks_sequence: list,
    t_impact: int,
    ball_center_t_impact=None,
    *,
    accumulate_on: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """一站式：相对坐标提取 + OpenCV 热力图 → 含 ``heatmap_base64`` 的 API 载荷。"""
    spatial = generate_spatial_trajectory(pose_landmarks_sequence, t_impact, ball_center_t_impact)
    b64, canvas = render_spatial_heatmap_base64(
        float(spatial.get("dx_support", 0.0) or 0.0),
        float(spatial.get("dy_support", 0.0) or 0.0),
        list(spatial.get("swing_trajectory") or []),
        accumulate_on=accumulate_on,
    )
    return {
        **spatial,
        "heatmap_base64": b64,
        "heatmap_data_uri": f"data:image/png;base64,{b64}" if b64 else None,
        "_canvas_bgr": canvas,  # 仅供服务端多趟次累加；序列化前应 pop
    }
