# -*- coding: utf-8 -*-
"""Cluster-RCT 枚举与生物力学量纲常量。"""

from __future__ import annotations

import enum


class ExperimentalGroup(str, enum.Enum):
    """整群随机对照试验三臂组别（班级级随机化后写入被试，不可变）。"""

    GROUP_A_REALTIME = "GROUP_A_REALTIME"
    GROUP_B_DELAYED = "GROUP_B_DELAYED"
    GROUP_C_CONTROL = "GROUP_C_CONTROL"


class StudyTimepoint(str, enum.Enum):
    """16 周纵向追踪的五个标准测评/干预节点。"""

    T0 = "T0"  # 基线
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"  # 期末/保持测


# 与 error_diagnoser.DeterministicScorer / llm_agent._RED_DEFECT_PRIORITY 对齐的 8 大量纲
BIOMECH_CORE_METRIC_KEYS: tuple[str, ...] = (
    "distance_cm",
    "toe_angle",
    "max_folding_angle",
    "whipping_velocity",
    "impact_knee_angle",
    "ankle_rigidity",
    "support_knee_angle",
    "hip_torsion_angle",
)

# 单次干预课默认练习剂量下限（射门次数）
DEFAULT_MIN_SHOTS_PER_SESSION: int = 15
