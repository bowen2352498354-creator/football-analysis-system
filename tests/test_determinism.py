# -*- coding: utf-8 -*-
"""
金标准回归测试：DeterministicScorer 必须在相同输入下 1000 次输出位级一致。

验证目标：
    TotalScore 与 t_impact 连续 1000 次计算结果绝对相同，浮点误差为 0.0。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# 允许直接从项目根目录运行：python tests/test_determinism.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from error_diagnoser import DeterministicScorer, calculate_biomechanical_score


def _joint(x: float, y: float, z: float = 0.0) -> list[float]:
    return [float(x), float(y), float(z)]


def build_fake_instep_kick_frames(n_frames: int = 100, t_impact: int = 60) -> list[dict]:
    """构造 100 帧脚背内侧射门假数据（确定性，无随机源）。"""
    frames: list[dict] = []
    for i in range(n_frames):
        t = i / 30.0
        # 相位参数：助跑 → 后摆蓄力 → 击球 → 随前
        phase = (i - t_impact) / max(1, t_impact)

        # 摆动腿（右）：后摆时膝屈曲加深，触球时打开到 ~150°
        if i < t_impact - 15:
            swing_knee = 165.0 - 5.0 * (i / max(1, t_impact - 15))
        elif i <= t_impact:
            # 后摆极值折叠：膝内角 ~100° → 折叠角 ~80°（落在 [70,90] GREEN）
            local = (i - (t_impact - 15)) / 15.0
            swing_knee = 165.0 - 65.0 * local  # → 100°
            if i == t_impact:
                swing_knee = 150.0
        else:
            swing_knee = 150.0 + min(20.0, (i - t_impact) * 1.5)

        # 由膝角反推关节坐标（简化平面几何，保证 calculate_angle 可复现）
        rh = _joint(0.40, 0.50, 0.0)
        # 髋→膝长度 0.20，膝→踝长度 0.22
        knee_flex_rad = math.radians(180.0 - swing_knee)
        rk = _joint(0.40 + 0.05 * phase, 0.70, -0.10 * max(0.0, -phase))
        ra_x = rk[0] + 0.22 * math.sin(knee_flex_rad)
        ra_y = rk[1] + 0.22 * math.cos(knee_flex_rad)
        ra = _joint(ra_x, ra_y, rk[2])
        # 踝锁紧：触球邻域三帧踝角几乎不变（方差 << 2）
        rfi = _joint(ra[0] + 0.08, ra[1] + 0.02, ra[2])

        # 支撑腿（左）：横距约 17.5cm（世界坐标米制 0.175）
        lh = _joint(0.20, 0.50, 0.0)
        lk = _joint(0.20, 0.72, 0.0)
        la = _joint(0.175, 0.95, 0.0)  # 相对球/右足尖横向 ~17.5cm
        lfi = _joint(0.175 + 0.06, 0.95, 0.02)  # 脚尖略指向前方
        lheel = _joint(0.175 - 0.05, 0.95, -0.01)

        ls = _joint(0.18, 0.20, 0.0)
        rs = _joint(0.42, 0.20, 0.05)  # 肩带相对骨盆轻微扭转

        world = {
            "left_hip": lh,
            "right_hip": rh,
            "left_knee": lk,
            "right_knee": rk,
            "left_ankle": la,
            "right_ankle": ra,
            "left_foot_index": lfi,
            "right_foot_index": rfi,
            "left_heel": lheel,
            "right_heel": _joint(ra[0] - 0.05, ra[1], ra[2] - 0.01),
            "left_shoulder": ls,
            "right_shoulder": rs,
        }

        frames.append(
            {
                "timestamp_sec": t,
                "left_hip": lh,
                "right_hip": rh,
                "left_knee": lk,
                "right_knee": rk,
                "left_ankle": la,
                "right_ankle": ra,
                "left_foot_index": lfi,
                "right_foot_index": rfi,
                "left_heel": lheel,
                "right_heel": world["right_heel"],
                "left_shoulder": ls,
                "right_shoulder": rs,
                "world": world,
                "visibility": {k: 1.0 for k in world},
            }
        )
    return frames


def build_score_inputs(frames: list[dict], t_impact: int = 60) -> tuple[dict, dict]:
    """从假帧构造 DeterministicScorer 所需的 impact / trajectory 输入。"""
    # 后摆折叠角：取 T0 前膝内角最小值对应的屈曲量
    knee_pre = []
    for i in range(max(0, t_impact - 20), t_impact):
        rec = frames[i]
        # 使用与引擎一致的三点角
        from error_diagnoser import calculate_angle

        knee_pre.append(calculate_angle(rec["right_hip"], rec["right_knee"], rec["right_ankle"]))
    min_knee = float(min(knee_pre)) if knee_pre else 100.0
    max_folding = max(0.0, 180.0 - min_knee)

    impact_frame_data = {
        "t_impact": int(t_impact),
        "frames": frames,
        "distance_cm": 17.5,
        "toe_angle": 8.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
    }
    trajectory_data = {
        "max_folding_angle": max_folding,
        "whipping_velocity": 520.0,
    }
    return impact_frame_data, trajectory_data


def test_deterministic_scorer_bit_identical_across_1000_runs():
    """金标准：同一假数据连续 1000 次，TotalScore 与 t_impact 浮点误差必须为 0.0。"""
    t_impact = 60
    frames = build_fake_instep_kick_frames(n_frames=100, t_impact=t_impact)
    impact_frame_data, trajectory_data = build_score_inputs(frames, t_impact=t_impact)

    scorer = DeterministicScorer()
    first_score, first_detail = scorer.calculate_biomechanical_score(
        impact_frame_data, trajectory_data
    )
    first_t = int(first_detail["t_impact"])

    assert first_t == t_impact
    assert isinstance(first_score, float)
    assert abs(first_score - round(first_score, 2)) == 0.0

    for i in range(1000):
        score, detail = calculate_biomechanical_score(impact_frame_data, trajectory_data)
        assert score == first_score, (
            f"第 {i + 1} 次 TotalScore 漂移：{score!r} != {first_score!r}"
        )
        assert detail["TotalScore"] == first_score
        assert detail["t_impact"] == first_t
        assert abs(float(score) - float(first_score)) == 0.0
        assert abs(float(detail["t_impact"]) - float(first_t)) == 0.0

        # 8 大量纲状态字段必须存在且可复现
        indicators = detail["indicators"]
        for key in (
            "distance_cm",
            "toe_angle",
            "max_folding_angle",
            "whipping_velocity",
            "impact_knee_angle",
            "ankle_rigidity",
            "support_knee_angle",
            "hip_torsion_angle",
        ):
            assert key in indicators
            assert indicators[key]["status"] in (
                "GREEN_OPTIMAL",
                "YELLOW_APPROACHING",
                "RED_DEVIATED",
            )
            assert indicators[key] == first_detail["indicators"][key]

        assert detail["radar_scores"] == first_detail["radar_scores"]

    # LLM 零参与标记
    assert first_detail["llm_participated"] is False
    assert first_detail["scoring_engine"] == "DeterministicScorer_V3.1"

    # V3.1 五维雷达：键齐全、范围合法
    radar = first_detail["radar_scores"]
    for key in (
        "support_stability",
        "backswing_folding",
        "ankle_rigidity",
        "whipping_velocity",
        "approach_rhythm",
    ):
        assert key in radar
        assert 0.0 <= float(radar[key]) <= 20.0
    assert 16.0 <= float(radar["approach_rhythm"]) <= 20.0


def test_ankle_rigidity_red_deducts_full_15():
    """脚踝方差 > 5.0 必须 RED 且直接扣满分 15。"""
    impact = {
        "t_impact": 1,
        "ankle_angles_window": [120.0, 140.0, 160.0],  # 方差远大于 5
        "distance_cm": 17.5,
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
    }
    trajectory = {"max_folding_angle": 80.0, "whipping_velocity": 500.0}
    score, detail = calculate_biomechanical_score(impact, trajectory)
    ankle = detail["indicators"]["ankle_rigidity"]
    assert ankle["status"] == "RED_DEVIATED"
    assert ankle["penalty"] == 15.0
    assert abs(score - 85.0) == 0.0  # 仅脚踝扣 15，其余 GREEN
    # 雷达锁踝档位：方差 > 5 → 5.0 分
    assert detail["radar_scores"]["ankle_rigidity"] == 5.0


def test_radar_scores_whipping_and_ankle_buckets():
    """鞭打线性映射 + 脚踝三档分桶金标准。"""
    # 鞭打 225 → 10.0；脚踝方差落在 [2,5] → 15
    impact = {
        "t_impact": 1,
        "ankle_angles_window": [100.0, 102.0, 104.5],  # var ≈ 3.39
        "distance_cm": 17.5,
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
    }
    trajectory = {"max_folding_angle": 80.0, "whipping_velocity": 225.0}
    _, detail = calculate_biomechanical_score(impact, trajectory)
    radar = detail["radar_scores"]
    assert radar["whipping_velocity"] == 10.0
    assert radar["ankle_rigidity"] == 15.0
    assert radar["support_stability"] == 20.0
    assert radar["backswing_folding"] == 20.0
    assert 16.0 <= radar["approach_rhythm"] <= 20.0


if __name__ == "__main__":
    test_deterministic_scorer_bit_identical_across_1000_runs()
    test_ankle_rigidity_red_deducts_full_15()
    test_radar_scores_whipping_and_ankle_buckets()
    print("ALL DETERMINISM GOLDEN TESTS PASSED")
