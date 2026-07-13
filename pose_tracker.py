# -*- coding: utf-8 -*-
"""
pose_tracker.py
v0.2 体育力学诊断引擎阶段脚本（在 v0.1 基础视觉捕捉之上迭代）

功能说明：
    1. 使用 OpenCV 打开电脑默认摄像头，读取实时视频流；
    2. 使用 Google MediaPipe 最新的 Tasks API
       （mediapipe.tasks.python.vision.PoseLandmarker）对每一帧画面进行人体姿态检测；
    3. 检测得到人体 33 个关键点后，使用 OpenCV 手动将骨架连线实时绘制在画面上；
    4. 【v0.2 新增】提取右髋(24)-右膝(26)-右踝(28) 三点坐标，用通用的空间向量
       夹角公式实时计算右膝关节屈曲角度，并按文档规定的三级容错阈值
       （Green/Yellow/Red）判定当前动作质量，将结果以颜色骨骼连线 + 大号文字
       的形式实时叠加渲染在画面上；
    5. 按下键盘上的 'q' 键即可退出程序并关闭窗口。

【重要说明】
    新版 mediapipe（0.10.x 及以上）已经彻底移除了 `mp.solutions.pose` 这种旧写法，
    官方全面转向了新的 Tasks API（`mediapipe.tasks.python.vision.PoseLandmarker`）。
    本脚本完全不使用 `mp.solutions`，包括绘图部分也不依赖
    `mp.solutions.drawing_utils` / `mp.solutions.drawing_styles`，
    而是自己用 OpenCV 的 cv2.circle / cv2.line 手动绘制关键点与骨架连线。

    新写法需要额外下载一个模型文件（.task 文件），本脚本会在启动时自动检测
    并下载 pose_landmarker_full.task（对应旧版 model_complexity=1，
    即精度与速度均衡的"full"模型）到脚本所在目录下。

这是整个"小学足球AI可视化反馈系统"五阶段开发蓝图中的第二步：
在 v0.1 跑通的摄像头 + 33 点骨架基础上，编写向量数学计算模块，
实时输出膝关节角度并渲染红/黄/绿三色容错框。
"""

import os
import urllib.request

import cv2
import numpy as np
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

# --------------------------------------------------------------------------
# 【v0.2 新增】体育力学诊断引擎：右膝关节屈曲角度计算与三级容错阈值判定
# --------------------------------------------------------------------------

# MediaPipe 33 个关键点中，右腿三个关键点的编号：
#   24 = 右髋（RIGHT_HIP）
#   26 = 右膝（RIGHT_KNEE）
#   28 = 右踝（RIGHT_ANKLE）
RIGHT_HIP_IDX = 24
RIGHT_KNEE_IDX = 26
RIGHT_ANKLE_IDX = 28

# 三种诊断状态对应的 BGR 颜色（用于骨骼连线染色 + 文字提示）
COLOR_GREEN = (0, 255, 0)      # 达标：绿色
COLOR_YELLOW = (0, 255, 255)   # 接近：黄色
COLOR_RED = (0, 0, 255)        # 错误：红色


def calculate_angle(a, b, c):
    """通用的空间夹角计算函数：给定三个空间坐标点 a、b、c，计算出以 b 为顶点、
    由 a→b 和 c→b 两条向量夹出的角度（单位：度）。

    这是一个纯数学函数，不依赖 MediaPipe 或 OpenCV，可以用来计算人体任意
    三个关键点组成的关节角度（例如膝关节角度：a=髋，b=膝，c=踝）。

    参数：
        a, b, c：三个坐标点，可以是 (x, y) 二维坐标，也可以是 (x, y, z) 三维坐标，
                 类型可以是列表、元组或 numpy 数组。

    返回：
        角度值（0~180 之间的浮点数，单位：度）。
    """
    # 统一转换成 numpy 数组，方便做向量运算
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)

    # 构造以 b（关节顶点）为起点，分别指向 a 和 c 的两个向量
    vector_ba = a - b
    vector_bc = c - b

    # 用向量点积公式反推夹角：
    #   cos(θ) = (ba · bc) / (|ba| * |bc|)
    dot_product = np.dot(vector_ba, vector_bc)
    norm_product = np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc)

    # 极端情况下（两点重合导致向量长度为 0）避免除以零
    if norm_product == 0:
        return 0.0

    cos_angle = dot_product / norm_product
    # 由于浮点数计算误差，cos_angle 可能会略微超出 [-1, 1] 的合法范围，
    # 这里用 np.clip 做一次安全裁剪，避免 arccos 报错
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # arccos 得到的是弧度（radians），再转换成角度（degrees）
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def judge_knee_status(angle):
    """根据文档规定的三级容错阈值，判定当前膝关节角度所处的状态。

    判定规则（严格按照 project_plan.md 中"核心生物力学诊断参数"章节）：
        Green（达标）：140° <= 角度 <= 160°
        Yellow（接近）：130° <= 角度 < 140° 或 160° < 角度 <= 170°
        Red（错误）：角度 < 130° 或 角度 > 170°

    参数：
        angle：当前实时计算出的膝关节屈曲角度（度）

    返回：
        一个二元组 (status_text, status_color)：
            status_text：字符串 "Green" / "Yellow" / "Red"
            status_color：对应的 BGR 颜色元组，用于绘制文字和骨骼连线
    """
    if 140 <= angle <= 160:
        return "Green", COLOR_GREEN
    elif (130 <= angle < 140) or (160 < angle <= 170):
        return "Yellow", COLOR_YELLOW
    else:
        return "Red", COLOR_RED


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


def diagnose_and_draw_right_knee(image, single_person_landmarks):
    """【v0.2 核心新增】提取右腿三点坐标，计算右膝角度，并把诊断结果渲染到画面上。

    具体做法：
        1. 从 33 个关键点中取出右髋(24)、右膝(26)、右踝(28) 三个点；
        2. 调用 calculate_angle() 算出右膝关节的实时屈曲角度；
        3. 调用 judge_knee_status() 按三级容错阈值判定 Green/Yellow/Red；
        4. 用 cv2.line 把"右髋-右膝"和"右膝-右踝"这两根骨骼连线，重新画成
           对应的红/黄/绿色，覆盖掉前面 draw_pose_landmarks 画的白色默认连线；
        5. 在画面左上角用大号字体显示当前角度数值与状态文字。

    参数：
        image：要绘制的目标 BGR 画面（会被直接就地修改）
        single_person_landmarks：某一个人的 33 个关键点（归一化坐标 0~1）

    返回：
        (angle, status_text)：本次计算出的角度值与状态文字，方便调用方做其他扩展
        （例如后续 v0.3 阶段把这个结果传给 AIGC 转译模块）
    """
    height, width = image.shape[:2]

    hip = single_person_landmarks[RIGHT_HIP_IDX]
    knee = single_person_landmarks[RIGHT_KNEE_IDX]
    ankle = single_person_landmarks[RIGHT_ANKLE_IDX]

    # 用 (x, y, z) 三维归一化坐标计算角度，z 轴（深度信息）能让角度计算
    # 在身体侧对镜头等场景下更加准确，不会因为只看二维投影而失真
    hip_point = (hip.x, hip.y, hip.z)
    knee_point = (knee.x, knee.y, knee.z)
    ankle_point = (ankle.x, ankle.y, ankle.z)

    # 计算右膝关节屈曲角度（以膝盖为顶点，髋-膝-踝三点夹角）
    knee_angle = calculate_angle(hip_point, knee_point, ankle_point)

    # 按三级容错阈值判定当前状态与对应颜色
    status_text, status_color = judge_knee_status(knee_angle)

    # 把归一化坐标换算成像素坐标，用于绘制
    hip_px = (int(hip.x * width), int(hip.y * height))
    knee_px = (int(knee.x * width), int(knee.y * height))
    ankle_px = (int(ankle.x * width), int(ankle.y * height))

    # 【重点 UI 修改】用诊断出的颜色重新画一遍右腿的两根骨骼连线，
    # 线条加粗（厚度 6），盖住 draw_pose_landmarks 之前画的白色细线，
    # 让学生一眼就能看出自己右腿姿态是"对/接近/错"
    cv2.line(image, hip_px, knee_px, status_color, 6)
    cv2.line(image, knee_px, ankle_px, status_color, 6)

    # 在左上角用大号字体实时显示角度数值与状态文字，颜色与状态保持一致
    angle_text = f"Right Knee Angle: {knee_angle:.1f} deg"
    status_line = f"Status: {status_text}"

    cv2.putText(
        image, angle_text, (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3, cv2.LINE_AA,
    )
    cv2.putText(
        image, status_line, (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX, 1.4, status_color, 3, cv2.LINE_AA,
    )

    return knee_angle, status_text


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

        # 【v0.2 新增】num_poses=1，最多只有一个人，取第一个人的 33 个关键点，
        # 提取右腿三点坐标、计算右膝角度，并把红/黄/绿诊断结果实时叠加到画面上
        diagnose_and_draw_right_knee(frame, results.pose_landmarks[0])

    # 将处理好（画上骨架）的画面显示在一个窗口中
    cv2.imshow("Pose Tracker - v0.2 (press 'q' to quit)", frame)

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
