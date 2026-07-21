# -*- coding: utf-8 -*-
"""桌面端后台 Worker 包：主/从线程分离的计算侧入口。"""

from .auto_shot_capture import AutoShotCaptureEngine, RollingBuffer, ShotFsmState
from .inference_worker import InferenceWorker, VideoWorker

__all__ = [
    "InferenceWorker",
    "VideoWorker",  # 兼容别名
    "AutoShotCaptureEngine",
    "RollingBuffer",
    "ShotFsmState",
]
