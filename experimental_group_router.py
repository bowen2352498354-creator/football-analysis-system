# -*- coding: utf-8 -*-
"""实验组别路由器 —— Active_Group 条件渲染策略（Cluster-RCT 三臂）。

在启动分析任务前注入 ``experimental_group``（GROUP_A / GROUP_B / GROUP_C），
由主窗体控制器按策略决定：
  - 是否连接触球锁定信号与时空胶囊阻塞模态（A）
  - 是否屏蔽骨骼预览、静默写本地宽表并暴露离线报告入口（B）
  - 是否保持干净对照预览（C）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# 前端 / Worker 统一使用的短码（与 UI 单选、信号路由对齐）
GROUP_A = "GROUP_A"
GROUP_B = "GROUP_B"
GROUP_C = "GROUP_C"

# 兼容历史短码与科研枚举长码
_GROUP_ALIASES: dict[str, str] = {
    "A": GROUP_A,
    "GROUP_A": GROUP_A,
    "GROUP_A_REALTIME": GROUP_A,
    "B": GROUP_B,
    "GROUP_B": GROUP_B,
    "GROUP_B_DELAYED": GROUP_B,
    "C": GROUP_C,
    "GROUP_C": GROUP_C,
    "GROUP_C_CONTROL": GROUP_C,
}


def normalize_experimental_group(raw: Optional[str], default: str = GROUP_A) -> str:
    """将任意组别标识归一化为 GROUP_A / GROUP_B / GROUP_C。"""
    if raw is None:
        return default
    key = str(raw).strip().upper()
    if not key:
        return default
    return _GROUP_ALIASES.get(key, default)


@dataclass(frozen=True)
class GroupRenderPolicy:
    """某一实验臂在主 UI / Worker 上的条件渲染与接线策略。"""

    experimental_group: str
    connect_impact_lock: bool
    render_skeleton: bool
    silent_recording: bool
    show_offline_report_button: bool
    emit_live_preview: bool
    pause_camera_for_capsule: bool


class ExperimentalGroupRouter:
    """基于 ``Active_Group`` 的实验组别路由器。

    主窗体在 ``start_training`` 前注入组别，随后通过本路由器决定信号连接与 UI 分支。
    """

    def __init__(self, experimental_group: Optional[str] = None):
        self._active_group = normalize_experimental_group(experimental_group, GROUP_A)

    @property
    def Active_Group(self) -> str:
        """当前激活的实验组别（只读属性名对齐需求文档）。"""
        return self._active_group

    @property
    def experimental_group(self) -> str:
        return self._active_group

    def set_active_group(self, experimental_group: str) -> str:
        """启动任务前注入 / 切换 Active_Group，返回归一化后的组别。"""
        self._active_group = normalize_experimental_group(experimental_group, GROUP_A)
        return self._active_group

    def policy(self) -> GroupRenderPolicy:
        """按 Active_Group 返回条件渲染策略。"""
        g = self._active_group
        if g == GROUP_A:
            return GroupRenderPolicy(
                experimental_group=GROUP_A,
                connect_impact_lock=True,
                render_skeleton=True,
                silent_recording=False,
                show_offline_report_button=False,
                emit_live_preview=True,
                pause_camera_for_capsule=True,
            )
        if g == GROUP_B:
            return GroupRenderPolicy(
                experimental_group=GROUP_B,
                connect_impact_lock=False,
                render_skeleton=False,
                silent_recording=True,
                show_offline_report_button=True,
                emit_live_preview=False,
                pause_camera_for_capsule=False,
            )
        # GROUP_C：干净对照
        return GroupRenderPolicy(
            experimental_group=GROUP_C,
            connect_impact_lock=False,
            render_skeleton=False,
            silent_recording=False,
            show_offline_report_button=False,
            emit_live_preview=True,
            pause_camera_for_capsule=False,
        )

    def is_group_a(self) -> bool:
        return self._active_group == GROUP_A

    def is_group_b(self) -> bool:
        return self._active_group == GROUP_B

    def is_group_c(self) -> bool:
        return self._active_group == GROUP_C
