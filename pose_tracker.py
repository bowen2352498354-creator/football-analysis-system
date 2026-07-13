# -*- coding: utf-8 -*-
"""
pose_tracker.py
v0.1 基础视觉捕捉阶段脚本

功能说明：
    1. 使用 OpenCV 打开电脑默认摄像头，读取实时视频流；
    2. 使用 Google MediaPipe 最新的 Tasks API
       （mediapipe.tasks.python.vision.PoseLandmarker）对每一帧画面进行人体姿态检测；
    3. 检测得到人体 33 个关键点后，使用 OpenCV 手动将骨架连线实时绘制在画面上；
    4. 按下键盘上的 'q' 键即可退出程序并关闭窗口。

【重要说明】
    新版 mediapipe（0.10.x 及以上）已经彻底移除了 `mp.solutions.pose` 这种旧写法，
    官方全面转向了新的 Tasks API（`mediapipe.tasks.python.vision.PoseLandmarker`）。
    本脚本完全不使用 `mp.solutions`，包括绘图部分也不依赖
    `mp.solutions.drawing_utils` / `mp.solutions.drawing_styles`，
    而是自己用 OpenCV 的 cv2.circle / cv2.line 手动绘制关键点与骨架连线。

    新写法需要额外下载一个模型文件（.task 文件），本脚本会在启动时自动检测
    并下载 pose_landmarker_full.task（对应旧版 model_complexity=1，
    即精度与速度均衡的"full"模型）到脚本所在目录下。

这是整个"小学足球AI可视化反馈系统"五阶段开发蓝图中的第一步：
跑通本地摄像头 + MediaPipe Pose 实时绘制 33 点火柴人骨架，建立基础开发环境。
"""

import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_vision

# --------------------------------------------------------------------------
# 第〇步：模型文件路径与自动下载逻辑
# --------------------------------------------------------------------------

# 模型文件路径：与本脚本放在同一目录下的 pose_landmarker_full.task
# "full" 对应旧版 API 中的 model_complexity=1（精度与速度较均衡）
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_full.task")

# 官方模型文件下载地址（Google 官方托管，Tasks API 文档中给出的标准地址）
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


def _download_progress_hook(block_num, block_size, total_size):
    """urllib.request.urlretrieve 的下载进度回调，在终端打印下载百分比。"""
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 // total_size)
    print(f"\r正在下载模型文件……{percent}%", end="", flush=True)


def ensure_model_downloaded():
    """如果本地不存在模型文件，就自动从官方地址下载到脚本所在目录。"""
    if os.path.exists(MODEL_PATH):
        return

    print(f"未找到模型文件：{MODEL_PATH}")
    print(f"正在从官方地址自动下载：{MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=_download_progress_hook)
        print("\n模型文件下载完成！")
    except Exception as exc:  # noqa: BLE001 - 下载失败时需要给用户明确提示
        # 下载失败时清理掉可能残留的不完整文件，避免下次误判为"已存在"
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        raise RuntimeError(
            f"模型文件自动下载失败：{exc}\n"
            f"请检查网络连接，或手动从以下地址下载并放到脚本所在目录：\n{MODEL_URL}"
        ) from exc


ensure_model_downloaded()

# --------------------------------------------------------------------------
# 第一步：初始化 MediaPipe Pose Landmarker（Tasks API，绝不使用 mp.solutions）
# --------------------------------------------------------------------------

# 创建 PoseLandmarker 的配置项
#   base_options.model_asset_path：指定本地模型文件路径
#   running_mode：使用 VIDEO 模式（专门针对"逐帧连续视频"场景，
#                 相比 IMAGE 模式，内部会做帧间关键点跟踪优化，效果更稳定；
#                 该模式要求每一帧都传入一个不断递增的时间戳）
#   num_poses：最多检测 1 个人（我们只关心单人踢球动作）
#   min_pose_detection_confidence / min_tracking_confidence：检测与跟踪的置信度阈值
pose_options = mp_vision.PoseLandmarkerOptions(
    base_options=mp_tasks_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# 创建 PoseLandmarker 检测器实例
landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)

# 33 个关键点之间的连接关系（骨架连线规则）。
# 因为脚本不允许使用 mp.solutions，所以这里直接把官方拓扑关系写成常量列表，
# 每一对数字表示"哪两个关键点编号之间要连一条线"。
POSE_CONNECTIONS = [
    (0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (11, 23), (12, 14), (12, 24),
    (13, 15), (14, 16), (15, 17), (15, 19), (15, 21), (16, 18),
    (16, 20), (16, 22), (17, 19), (18, 20), (23, 24), (23, 25),
    (24, 26), (25, 27), (26, 28), (27, 29), (27, 31), (28, 30),
    (28, 32), (29, 31), (30, 32),
]

# 绘图颜色（BGR 格式）
LANDMARK_COLOR = (0, 255, 0)    # 关键点：绿色
CONNECTION_COLOR = (255, 255, 255)  # 连线：白色


def draw_pose_landmarks(image, pose_landmarks_list):
    """手动使用 OpenCV 把检测到的关键点和骨架连线画在画面上（不依赖 mp.solutions）。

    参数：
        image：要绘制的目标 BGR 画面（会被直接就地修改）
        pose_landmarks_list：results.pose_landmarks，即"人员列表"，
                              每个元素是这个人的 33 个关键点（归一化坐标 0~1）
    """
    height, width = image.shape[:2]

    for single_person_landmarks in pose_landmarks_list:
        # 先画连线，再画关键点，这样关键点会盖在连线上面，视觉效果更清晰
        for start_idx, end_idx in POSE_CONNECTIONS:
            start_lm = single_person_landmarks[start_idx]
            end_lm = single_person_landmarks[end_idx]
            start_point = (int(start_lm.x * width), int(start_lm.y * height))
            end_point = (int(end_lm.x * width), int(end_lm.y * height))
            cv2.line(image, start_point, end_point, CONNECTION_COLOR, 2)

        for landmark in single_person_landmarks:
            center = (int(landmark.x * width), int(landmark.y * height))
            cv2.circle(image, center, 4, LANDMARK_COLOR, -1)


# --------------------------------------------------------------------------
# 第二步：打开电脑默认摄像头
# --------------------------------------------------------------------------

# 参数 0 表示使用电脑的第一个（默认）摄像头
# 如果你的电脑有多个摄像头，可以尝试改成 1、2 等数字
cap = cv2.VideoCapture(0)

# 判断摄像头是否成功打开，如果失败就提示用户并退出程序
if not cap.isOpened():
    print("错误：无法打开摄像头，请检查摄像头是否被其他程序占用，或摄像头驱动是否正常。")
    exit()

print("摄像头已成功打开，正在实时检测人体骨架……")
print("按下键盘上的 'q' 键可以退出程序。")

# VIDEO 运行模式要求我们为每一帧提供一个不断递增的时间戳（单位：毫秒）
frame_timestamp_ms = 0

# --------------------------------------------------------------------------
# 第三步：循环读取摄像头画面，逐帧进行姿态检测与绘制
# --------------------------------------------------------------------------

while cap.isOpened():
    # cap.read() 会返回两个值：
    #   ret：是否成功读取到一帧画面（True/False）
    #   frame：读取到的这一帧图像数据（本质是一个 numpy 数组）
    ret, frame = cap.read()

    if not ret:
        print("警告：读取画面失败，可能是摄像头已断开，程序即将退出。")
        break

    # 为了提升"自拍视角"下的观感，这里将画面进行左右镜像翻转
    # （可选步骤，如果你不需要镜像效果，可以删掉这一行）
    frame = cv2.flip(frame, 1)

    # MediaPipe 要求输入图像是 RGB 格式，
    # 而 OpenCV 读取到的画面默认是 BGR 格式，因此这里需要做一次颜色空间转换
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 将 numpy 数组包装成 MediaPipe 自己的 Image 对象
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 时间戳需要不断递增，这里用摄像头的实际时间来计算毫秒时间戳，
    # 也可以简单用一个自增计数器模拟，只要保证严格递增即可
    frame_timestamp_ms += 33  # 近似按 30 fps 递增（1000ms / 30 ≈ 33ms）

    # 调用 PoseLandmarker 对当前帧进行姿态检测，得到 33 个关键点坐标信息
    results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    # 如果成功检测到了人体（results.pose_landmarks 是一个"人员列表"，
    # 每个元素对应一个人的 33 个关键点；因为 num_poses=1，最多只有 1 个元素）
    if results.pose_landmarks:
        draw_pose_landmarks(frame, results.pose_landmarks)

    # 将处理好（画上骨架）的画面显示在一个窗口中
    cv2.imshow("Pose Tracker - v0.1 (press 'q' to quit)", frame)

    # cv2.waitKey(1) 会等待 1 毫秒，检测是否有键盘按键被按下
    # 0xFF 是为了兼容不同操作系统的按键编码写法
    # 如果按下的是 'q' 键，就跳出循环，结束程序
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("检测到 'q' 键，正在退出程序……")
        break

# --------------------------------------------------------------------------
# 第四步：释放资源，关闭窗口
# --------------------------------------------------------------------------

# 释放摄像头资源，交还给系统，方便其他程序使用摄像头
cap.release()

# 关闭 MediaPipe Pose Landmarker 检测器
landmarker.close()

# 关闭所有由 OpenCV 打开的窗口
cv2.destroyAllWindows()

print("程序已安全退出。")
