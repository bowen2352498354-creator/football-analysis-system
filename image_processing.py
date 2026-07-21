# -*- coding: utf-8 -*-
"""
image_processing.py
图像处理公共模块 —— 未成年人隐私保护红线能力集中于此。

本模块对外暴露物理级面部脱敏拦截器 ``apply_facial_anonymization``：
在 MediaPipe 返回有效的 33 个姿态关键点后，用 0~10 号面部关键点
算出包围盒，向外扩充约 15% Padding，再以高强度 ``cv2.GaussianBlur``
涂抹该矩形区域，确保任何下游渲染、WebSocket 推流、Attempt 切片落盘
都只能接触到"无脸"安全图像。
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import cv2
import numpy as np

# MediaPipe Pose 0~10 号关键点覆盖头部/面部（鼻、眼、耳、嘴角）
FACE_LANDMARK_INDICES = list(range(0, 11))

# 包围盒向外扩充比例（开题报告要求覆盖整个头部）
FACE_ANON_EXPAND_RATIO = 0.15

# 固定高强度高斯核：人脸区域完全模糊不可辨认（核尺寸必须为奇数）
FACE_ANON_KERNEL_SIZE: Tuple[int, int] = (45, 45)

# SigmaX 取较大值，配合 (45, 45) 核彻底抹去面部可识别细节
FACE_ANON_SIGMA_X: float = 25.0

# 兼容旧导入名（历史代码 ``FACE_ANON_MIN_KERNEL_SIZE``）
FACE_ANON_MIN_KERNEL_SIZE = FACE_ANON_KERNEL_SIZE[0]

# 完整姿态关键点数量；不足则视为无效，跳过脱敏（避免误涂抹）
POSE_LANDMARK_COUNT = 33


def _normalize_landmark_list(pose_landmarks: Any) -> Sequence[Any] | None:
    """把调用方传入的 pose_landmarks 规整为"单人 33 点"序列。

    兼容两种常见形态：
      1) 单人 landmarks（len == 33）
      2) MediaPipe ``results.pose_landmarks``（多人列表，取第一人）
    """
    if pose_landmarks is None:
        return None

    # 多人列表：第一层元素本身还是 landmarks 序列
    try:
        first = pose_landmarks[0]
    except (TypeError, IndexError, KeyError):
        return None

    if hasattr(first, "x") and hasattr(first, "y"):
        # 已是单人 NormalizedLandmark 序列
        landmarks = pose_landmarks
    else:
        # 多人：取检测到的第一个人
        try:
            landmarks = pose_landmarks[0]
        except (TypeError, IndexError):
            return None

    try:
        if len(landmarks) < POSE_LANDMARK_COUNT:
            return None
    except TypeError:
        return None

    return landmarks


def apply_facial_anonymization(
    image: np.ndarray,
    pose_landmarks: Any,
) -> np.ndarray:
    """物理级面部高斯模糊脱敏拦截器。

    当检测到姿态关键点后，提取索引 0~10（面部），按绝对像素坐标计算
    Bounding Box，上下左右外扩 15% 以包裹完整头部，再对 ROI 施加
    ``cv2.GaussianBlur``（核 ``(45, 45)``，较大 SigmaX），原地强烈涂抹后
    返回同一图像矩阵。

    必须在 ``draw_pose_landmarks`` / 任何诊断标注绘制之前调用，
    且在发射 ``frame_ready_signal`` 或把 ``Attempt_XX.mp4`` 写入硬盘之前，
    强制用返回值覆盖原始帧。

    参数：
        image: BGR 图像（``np.ndarray``），会被原地修改面部 ROI
        pose_landmarks: 单人 33 点，或 ``results.pose_landmarks`` 多人列表

    返回：
        脱敏后的图像（与入参为同一 ndarray，便于 ``frame = apply_facial_anonymization(...)``）
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return image

    landmarks = _normalize_landmark_list(pose_landmarks)
    if landmarks is None:
        return image

    height, width = image.shape[:2]

    face_xs = [landmarks[idx].x * width for idx in FACE_LANDMARK_INDICES]
    face_ys = [landmarks[idx].y * height for idx in FACE_LANDMARK_INDICES]
    min_x, max_x = min(face_xs), max(face_xs)
    min_y, max_y = min(face_ys), max(face_ys)

    box_width = max_x - min_x
    box_height = max_y - min_y
    expand_x = box_width * FACE_ANON_EXPAND_RATIO
    expand_y = box_height * FACE_ANON_EXPAND_RATIO

    x1 = int(max(0, min_x - expand_x))
    y1 = int(max(0, min_y - expand_y))
    x2 = int(min(width, max_x + expand_x))
    y2 = int(min(height, max_y + expand_y))

    if x2 <= x1 or y2 <= y1:
        return image

    face_roi = image[y1:y2, x1:x2]
    # 固定核 (45, 45) + 大 SigmaX：小学生面部完全不可辨认
    blurred_face_roi = cv2.GaussianBlur(
        face_roi,
        FACE_ANON_KERNEL_SIZE,
        FACE_ANON_SIGMA_X,
    )
    image[y1:y2, x1:x2] = blurred_face_roi

    return image
