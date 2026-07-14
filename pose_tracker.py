# -*- coding: utf-8 -*-
"""
pose_tracker.py
v0.4 差异化双界面开发阶段脚本（在 v0.3 AIGC 认知转译接入基础上迭代）

功能说明（本版本核心变化）：
    本版本把之前"一个从头跑到尾的 OpenCV 脚本"彻底重构成了一个真正的
    PyQt5 桌面软件程序，界面左右分栏：
        左侧【控制面板】：选择数据源（摄像头 / 本地视频文件）、
                          选择实验组别（A组-实时 / B组-延时 / C组-对照）、
                          以及"开始训练" / "结束训练"两个按钮；
        右侧【视频展示区】：一个足够大的 QLabel，用来实时显示处理后的画面。

    三个实验组别的核心业务逻辑严格按照 project_plan.md 文档"第3节 差异化
    教学路径与交互界面规范"来区分：
        A组（实时反馈）：完整继承 v0.1~v0.3 的逻辑——检测骨架 -> 计算右膝角度
                        -> 红/黄/绿染色骨骼线 -> 连续 Red 触发 DeepSeek 大模型
                        -> 画面上叠加中文指导语，处理后的画面实时显示在右侧。
        B组（延时反馈）：右侧视频区【绝不显示摄像头画面】，只居中显示一句提示
                        文字"正在采集中，请专心练习"；后台仍然正常提取关键点、
                        计算右膝角度、判断红绿灯状态，但完全不调用大模型、
                        也不渲染任何骨骼线条，只是把每一条有效数据（时间戳、
                        右膝角度、判定状态）静默追加保存到本地 JSON 文件
                        （B_group_data_log.json）里。
        C组（常规对照）：右侧只显示"干净"的原始视频画面，不画骨骼线、
                        不显示颜色框、也不显示任何文字；【v1.0 变更】但为了
                        满足"面部绝对脱敏"的伦理红线，C组同样会在后台跑一次
                        姿态检测，仅用于定位面部区域并做高斯模糊打码。

技术重构要点（极度重要，务必保持）：
    1. 原来写在主线程里的 `while True` 摄像头循环，现在被整个搬进了一个
       继承自 QThread 的后台线程类 VideoWorker 里运行。主线程（GUI 线程）
       里绝对不会出现任何"死循环读摄像头"的代码，否则界面会卡死甚至崩溃。
    2. 后台线程处理好每一帧画面后，通过自定义的 pyqtSignal 信号
       （携带一个 QImage 对象）"发送"给主线程，主线程收到信号后，
       只需要把 QImage 转成 QPixmap 塞进 QLabel 里显示即可，非常轻量，
       完全不会阻塞或拖慢 GUI 的响应。
    3. 【v0.3 遗留设计延续】即便是在后台线程 VideoWorker 内部，调用 DeepSeek
       大模型这种"耗时的网络请求"依然会被丢进一个更内层的 threading.Thread
       子线程去执行，避免网络请求的等待时间拖慢 VideoWorker 本身的摄像头
       画面处理节奏（也就是"双层线程"结构：QThread 负责摄像头循环，
       内部再派生 threading.Thread 负责耗时的大模型请求）。

【重要说明】
    新版 mediapipe（0.10.x 及以上）已经彻底移除了 `mp.solutions.pose` 这种旧写法，
    官方全面转向了新的 Tasks API（`mediapipe.tasks.python.vision.PoseLandmarker`）。
    本脚本完全不使用 `mp.solutions`，包括绘图部分也不依赖
    `mp.solutions.drawing_utils` / `mp.solutions.drawing_styles`，
    而是自己用 OpenCV 的 cv2.circle / cv2.line 手动绘制关键点与骨架连线。

    另外，OpenCV 的 cv2.putText 并不支持中文字符（会画出乱码方块），
    所以本脚本在需要显示中文文字（大模型生成的指导语、B组黑屏提示语）时，
    改用 PIL（Pillow）库的 ImageDraw + ImageFont 来绘制中文，
    画好之后再转换回 OpenCV 的图像格式，最终再转成 PyQt5 的 QImage 显示。

这是整个"小学足球AI可视化反馈系统"五阶段开发蓝图中的第四步：
在 v0.1~v0.3 打通的"姿态检测 + 力学诊断 + AIGC 认知转译"技术底座之上，
用 PyQt5 搭建出能同时服务于 A/B/C 三个实验组别、满足 Cluster-RCT 科研
实验设计要求的差异化双界面桌面软件。

【v1.0 补丁说明】面部绝对脱敏（科技伦理与隐私保护红线）：
    在 A 组（实时反馈）和 C 组（常规对照）两个"会把画面显示出来"的
    实验组别里，新增了 apply_face_blur() 高斯模糊打码逻辑——利用
    MediaPipe 输出的 0~10 号头部/面部关键点算出边界框、向外扩展 15%
    后，对该区域做高斯模糊，确保无论哪个组别，学生的脸部特征在实时
    展示画面中都彻底不可辨识。B 组本来就是黑屏，天然满足这一要求，
    不需要额外处理。
"""

import json
import os
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageDraw, ImageFont

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# 【v0.3 遗留】引入我们自己写的 AIGC 代理模块，里面封装了对 DeepSeek 大模型的调用
import llm_agent

# 脚本所在目录，后面模型文件、默认视频文件、B组数据日志文件都以此为基准路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# 第〇步：模型文件路径与自动下载逻辑（与 v0.1~v0.3 完全一致）
# --------------------------------------------------------------------------

# 模型文件路径：与本脚本放在同一目录下的 pose_landmarker_full.task
# "full" 对应旧版 API 中的 model_complexity=1（精度与速度较均衡）
MODEL_PATH = os.path.join(SCRIPT_DIR, "pose_landmarker_full.task")

# 官方模型文件下载地址（Google 官方托管，Tasks API 文档中给出的标准地址）
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

# 默认本地视频文件路径（用户在界面上选择"本地视频"数据源时的默认候选路径）
DEFAULT_VIDEO_FILE_PATH = os.path.join(SCRIPT_DIR, "test_video.mp4")

# 【v0.4 新增】B组"延时反馈"实验数据的本地落盘文件路径
B_GROUP_LOG_PATH = os.path.join(SCRIPT_DIR, "B_group_data_log.json")


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


# --------------------------------------------------------------------------
# 第一步：骨架连线拓扑关系与绘图颜色常量（与 v0.1~v0.3 完全一致）
# --------------------------------------------------------------------------

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
LANDMARK_COLOR = (0, 255, 0)       # 关键点：绿色
CONNECTION_COLOR = (255, 255, 255)  # 连线：白色

# --------------------------------------------------------------------------
# 【v1.0 新增】科技伦理与隐私保护：面部绝对脱敏所需的关键点范围与参数
# --------------------------------------------------------------------------

# MediaPipe Pose 的 0~10 号关键点，全部落在头部/面部区域
# （0=鼻尖，1~6=左右眼内外眼角，7~8=左右耳，9~10=嘴角左右两端），
# 用这 11 个点就足够画出一个能完整覆盖头部的边界框。
FACE_LANDMARK_INDICES = list(range(0, 11))

# 边界框向四周扩展的比例：项目开题报告要求至少覆盖到整个头部，
# 这里按照最新需求把扩展比例从 10% 提高到 15%，留更充足的余量。
FACE_BLUR_EXPAND_RATIO = 0.15

# 高斯模糊核尺寸的下限（必须是奇数），保证即使脸部在画面中很小，
# 模糊强度依然足够让面部特征彻底不可辨识。
FACE_BLUR_MIN_KERNEL_SIZE = 21

# --------------------------------------------------------------------------
# 第二步：体育力学诊断引擎：右膝关节屈曲角度计算与三级容错阈值判定
#         （纯数学/纯计算函数，与 v0.2 完全一致，不掺杂任何绘图代码）
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

# 【v0.3 遗留】防抖阈值：必须连续处于 Red 状态达到这么多秒，才允许触发一次大模型调用
RED_DEBOUNCE_SECONDS = 1.5

# 大模型生成的中文指导语，在画面上持续展示的时长（单位：秒）
FEEDBACK_DISPLAY_SECONDS = 8.0

# 中文字体文件路径：cv2.putText 本身不支持中文（会显示成乱码方块），
# 所以这里改用 PIL 加载 Windows 系统自带的中文字体（微软雅黑），
# 如果这个字体不存在，就依次尝试黑体、宋体作为备选。
_CANDIDATE_FONT_PATHS = [
    r"C:\Windows\Fonts\msyh.ttc",    # 微软雅黑（首选，字形清晰美观）
    r"C:\Windows\Fonts\simhei.ttf",  # 黑体（备选一）
    r"C:\Windows\Fonts\simsun.ttc",  # 宋体（备选二）
]


def _load_chinese_font(font_size):
    """按优先级依次尝试加载系统自带的中文字体，全部失败时回退到 PIL 默认字体
    （默认字体不支持中文，会显示成方块，但至少不会让程序崩溃）。
    """
    for font_path in _CANDIDATE_FONT_PATHS:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception:  # noqa: BLE001 - 字体文件损坏等极端情况下继续尝试下一个
                continue
    print("警告：未能在系统中找到可用的中文字体文件，中文提示语可能无法正常显示。")
    return ImageFont.load_default()


# 提前把字体加载好（只需要加载一次，避免每一帧都重新读取字体文件影响性能）
FEEDBACK_FONT = _load_chinese_font(font_size=36)      # A组：大模型反馈语字号
BLACKSCREEN_FONT = _load_chinese_font(font_size=42)   # B组：黑屏提示语字号


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
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)

    vector_ba = a - b
    vector_bc = c - b

    dot_product = np.dot(vector_ba, vector_bc)
    norm_product = np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc)

    if norm_product == 0:
        return 0.0

    cos_angle = dot_product / norm_product
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def judge_knee_status(angle):
    """根据文档规定的三级容错阈值，判定当前膝关节角度所处的状态。

    判定规则（严格按照 project_plan.md 中"核心生物力学诊断参数"章节）：
        Green（达标）：140° <= 角度 <= 160°
        Yellow（接近）：130° <= 角度 < 140° 或 160° < 角度 <= 170°
        Red（错误）：角度 < 130° 或 角度 > 170°

    返回：
        (status_text, status_color)：状态文字与对应的 BGR 颜色元组。
    """
    if 140 <= angle <= 160:
        return "Green", COLOR_GREEN
    elif (130 <= angle < 140) or (160 < angle <= 170):
        return "Yellow", COLOR_YELLOW
    else:
        return "Red", COLOR_RED


def compute_right_knee_diagnosis(frame, single_person_landmarks):
    """【纯计算，不做任何绘制】提取右腿三点坐标，计算右膝角度并判定三级状态。

    之所以把"计算"和"绘制"拆成两个独立函数，是因为 v0.4 三个实验组别对
    "是否需要把诊断结果画到画面上"的需求完全不同：
        A组：既要计算，也要把结果画在画面上；
        B组：只需要计算结果用于静默落盘，绝对不能画在画面上；
        C组：压根不需要姿态检测（在 VideoWorker 里就不会调用本函数）。

    参数：
        frame：当前 BGR 画面（只用来读取宽高，不会被修改）
        single_person_landmarks：某个人的 33 个关键点（归一化坐标 0~1）

    返回：
        (angle, status_text, status_color, hip_px, knee_px, ankle_px)
    """
    height, width = frame.shape[:2]

    hip = single_person_landmarks[RIGHT_HIP_IDX]
    knee = single_person_landmarks[RIGHT_KNEE_IDX]
    ankle = single_person_landmarks[RIGHT_ANKLE_IDX]

    # 用 (x, y, z) 三维归一化坐标计算角度，z 轴（深度信息）能让角度计算
    # 在身体侧对镜头等场景下更加准确，不会因为只看二维投影而失真
    hip_point = (hip.x, hip.y, hip.z)
    knee_point = (knee.x, knee.y, knee.z)
    ankle_point = (ankle.x, ankle.y, ankle.z)

    knee_angle = calculate_angle(hip_point, knee_point, ankle_point)
    status_text, status_color = judge_knee_status(knee_angle)

    hip_px = (int(hip.x * width), int(hip.y * height))
    knee_px = (int(knee.x * width), int(knee.y * height))
    ankle_px = (int(ankle.x * width), int(ankle.y * height))

    return knee_angle, status_text, status_color, hip_px, knee_px, ankle_px


def apply_face_blur(frame, single_person_landmarks):
    """【v1.0 新增：科技伦理与隐私保护核心函数】把画面中学生的整个头部/面部区域
    用高斯模糊彻底打码，确保无论是 A 组还是 C 组的实时展示画面，学生的脸部特征
    都完全不可辨识（符合未成年人隐私保护与科研伦理规范）。

    重要：这个函数必须在 draw_pose_landmarks（画骨骼连线）之前调用！
    因为骨骼连线在人脸区域也会经过几个点，如果先画骨骼线再打码，
    骨骼线条会被一起模糊掉；反过来"先打码、再画骨骼线"则完全没有问题，
    还能让学生一眼看出脸部已经被保护起来。

    参数：
        frame：当前 BGR 画面，会被【原地修改】（直接在人脸区域画上模糊效果）
        single_person_landmarks：某个人的 33 个关键点（归一化坐标 0~1）

    返回：
        修改后的 frame（其实是同一个 numpy 数组，返回值只是为了方便链式调用）。
    """
    height, width = frame.shape[:2]

    # 第一步：取出 0~10 号面部关键点的像素坐标，计算出能框住整个头部的边界框
    face_xs = [single_person_landmarks[idx].x * width for idx in FACE_LANDMARK_INDICES]
    face_ys = [single_person_landmarks[idx].y * height for idx in FACE_LANDMARK_INDICES]
    min_x, max_x = min(face_xs), max(face_xs)
    min_y, max_y = min(face_ys), max(face_ys)

    # 第二步：把边界框向四周扩展 15%，确保额头、下巴、耳朵之外的头部边缘也被完整覆盖
    box_width = max_x - min_x
    box_height = max_y - min_y
    expand_x = box_width * FACE_BLUR_EXPAND_RATIO
    expand_y = box_height * FACE_BLUR_EXPAND_RATIO

    x1 = int(max(0, min_x - expand_x))
    y1 = int(max(0, min_y - expand_y))
    x2 = int(min(width, max_x + expand_x))
    y2 = int(min(height, max_y + expand_y))

    # 极端情况下（比如检测抖动导致宽高为 0）直接跳过，避免 cv2.GaussianBlur 报错
    if x2 <= x1 or y2 <= y1:
        return frame

    # 第三步：对扩展后的区域做高斯模糊。核尺寸根据人脸框大小自适应放大，
    # 保证脸部离摄像头很近（人脸框很大）时模糊强度依然足够，
    # 同时用 FACE_BLUR_MIN_KERNEL_SIZE 兜底，避免人脸框太小时模糊强度不够。
    roi_width = x2 - x1
    roi_height = y2 - y1
    kernel_w = max(FACE_BLUR_MIN_KERNEL_SIZE, roi_width // 2)
    kernel_h = max(FACE_BLUR_MIN_KERNEL_SIZE, roi_height // 2)
    kernel_w = kernel_w if kernel_w % 2 == 1 else kernel_w + 1  # cv2.GaussianBlur 要求核尺寸必须是奇数
    kernel_h = kernel_h if kernel_h % 2 == 1 else kernel_h + 1

    face_roi = frame[y1:y2, x1:x2]
    blurred_face_roi = cv2.GaussianBlur(face_roi, (kernel_w, kernel_h), 0)
    frame[y1:y2, x1:x2] = blurred_face_roi

    return frame


def draw_pose_landmarks(image, pose_landmarks_list):
    """手动使用 OpenCV 把检测到的关键点和骨架连线画在画面上（不依赖 mp.solutions）。
    只有 A 组（实时反馈）才会调用这个函数。
    """
    height, width = image.shape[:2]

    for single_person_landmarks in pose_landmarks_list:
        for start_idx, end_idx in POSE_CONNECTIONS:
            start_lm = single_person_landmarks[start_idx]
            end_lm = single_person_landmarks[end_idx]
            start_point = (int(start_lm.x * width), int(start_lm.y * height))
            end_point = (int(end_lm.x * width), int(end_lm.y * height))
            cv2.line(image, start_point, end_point, CONNECTION_COLOR, 2)

        for landmark in single_person_landmarks:
            center = (int(landmark.x * width), int(landmark.y * height))
            cv2.circle(image, center, 4, LANDMARK_COLOR, -1)


def draw_right_knee_overlay(image, hip_px, knee_px, ankle_px, status_color, angle, status_text):
    """【只有 A 组会调用】把右膝诊断结果（红/黄/绿染色骨骼线 + 角度文字）画到画面上。"""
    # 用诊断出的颜色重新画一遍右腿的两根骨骼连线，线条加粗（厚度 6），
    # 盖住 draw_pose_landmarks 之前画的白色细线，让学生一眼就能看出
    # 自己右腿姿态是"对/接近/错"
    cv2.line(image, hip_px, knee_px, status_color, 6)
    cv2.line(image, knee_px, ankle_px, status_color, 6)

    angle_text = f"Right Knee Angle: {angle:.1f} deg"
    status_line = f"Status: {status_text}"

    cv2.putText(
        image, angle_text, (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3, cv2.LINE_AA,
    )
    cv2.putText(
        image, status_line, (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX, 1.4, status_color, 3, cv2.LINE_AA,
    )


def draw_chinese_text_with_backdrop(frame_bgr, text, font, text_color=(255, 220, 80),
                                     anchor="bottom"):
    """用 PIL 在画面上画一句带半透明深色底纹的中文文字（cv2.putText 无法正确显示中文）。

    参数：
        frame_bgr：OpenCV 格式（BGR）的当前帧画面
        text：要绘制的中文文本
        font：PIL 的 ImageFont 对象
        text_color：文字颜色（RGB 格式，因为最终是用 PIL 画的）
        anchor：文字锚定位置，"bottom" 表示画在画面底部居中，
                "center" 表示画在画面正中间（B组黑屏提示语用这个）

    返回：
        画好文字之后的 BGR 画面（numpy 数组）。
    """
    height, width = frame_bgr.shape[:2]

    rgb_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    text_x = max(0, (width - text_width) // 2)
    if anchor == "center":
        text_y = max(0, (height - text_height) // 2)
    else:
        text_y = height - text_height - 40

    padding = 16
    overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [
            text_x - padding,
            text_y - padding,
            text_x + text_width + padding,
            text_y + text_height + padding,
        ],
        fill=(0, 0, 0, 160),
    )
    pil_image = Image.alpha_composite(pil_image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    draw.text((text_x, text_y), text, font=font, fill=text_color)

    result_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return result_bgr


def bgr_frame_to_qimage(frame_bgr):
    """把 OpenCV 的 BGR numpy 画面，转换成 PyQt5 能直接显示的 QImage 对象。

    注意：这里必须调用 .copy()！因为 QImage 默认只是"包了一层壳"指向
    numpy 数组底层的内存，如果不拷贝一份，等这一帧的 numpy 数组被后续
    循环覆盖或被垃圾回收之后，界面上显示的画面就可能变成花屏或崩溃。
    """
    rgb_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb_image.shape
    bytes_per_line = channels * width
    qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return qimage.copy()


# --------------------------------------------------------------------------
# 第三步：VideoWorker —— 真正的"后台工作线程"，继承自 QThread
#         【技术重构核心】原来主线程里的 while 循环，全部搬到这里执行
# --------------------------------------------------------------------------


class VideoWorker(QThread):
    """负责"打开视频源 -> 逐帧姿态检测 -> 按实验组别分流处理 -> 通过信号把
    处理好的画面发回主线程显示"的后台线程。

    绝对不能把这里的 while 循环直接写在主线程（GUI 线程）里，否则整个
    界面会被摄像头读取、MediaPipe 推理这些耗时操作卡死，用户点击按钮、
    拖动窗口都会没有任何反应，甚至被操作系统判定为"未响应"而强制关闭。
    """

    # 【核心信号一】每处理好一帧画面，就把这一帧打包成 QImage 发给主线程
    frame_ready = pyqtSignal(QImage)

    # 【核心信号二】把运行过程中的关键事件（开始/结束/报错/保存文件等）
    # 以文字形式发给主线程，主线程会把这些文字显示在界面下方的日志区域
    log_message = pyqtSignal(str)

    def __init__(self, data_source, video_path, group, camera_index=0, parent=None):
        """
        参数：
            data_source：数据源开关，"webcam"（摄像头）或 "video"（本地视频文件）
            video_path：当 data_source="video" 时，本地视频文件的路径
            group：实验组别，"A"（实时反馈）/ "B"（延时反馈）/ "C"（常规对照）
            camera_index：摄像头设备编号，默认 0（电脑默认摄像头）
        """
        super().__init__(parent)
        self.data_source = data_source
        self.video_path = video_path
        self.group = group
        self.camera_index = camera_index

        # 【线程停止开关】主线程调用 request_stop() 时只是把这个标志位设为 False，
        # 线程内部的 while 循环每一轮都会检查这个标志位，检测到 False 就自然退出
        # 循环、清理资源、结束线程，绝不会在 GUI 线程里做任何"强杀"操作。
        self._running = True

        # 【v0.3 遗留：AIGC 联动的防抖状态】
        # 之前是模块级全局变量，现在改成实例属性——这样每次点击"开始训练"
        # 新建一个 VideoWorker 时，防抖状态都是全新的，不会受到上一次运行的影响。
        self._red_streak_start_time = None
        self._red_streak_already_triggered = False
        self._feedback_lock = threading.Lock()
        self._feedback_text = None
        self._feedback_show_until = 0.0
        self._is_calling_llm = False

        # 【B组专用：本次训练新采集到的数据记录列表】
        self._b_group_new_records = []

    def request_stop(self):
        """供主线程调用：请求这个后台线程尽快自然结束（非强制杀死）。"""
        self._running = False

    # ----------------------------------------------------------------
    # 【v0.3 遗留】AIGC 认知转译联动逻辑：防抖触发 + 后台子线程调用 + 中文叠加
    #              （只有 A 组会用到这一整套逻辑）
    # ----------------------------------------------------------------

    def _call_llm_in_background(self, angle, status):
        """在更内层的 threading.Thread 子线程中实际执行：调用大模型接口拿到
        中文反馈文本，写入共享状态，供 VideoWorker 主循环渲染到画面上。
        """
        self.log_message.emit(f"检测到连续 {RED_DEBOUNCE_SECONDS} 秒 Red 状态，正在请求 DeepSeek 大模型……")
        feedback_text = llm_agent.generate_feedback(angle, status)
        self.log_message.emit(f"DeepSeek 大模型返回：{feedback_text}")

        with self._feedback_lock:
            self._feedback_text = feedback_text
            self._feedback_show_until = time.time() + FEEDBACK_DISPLAY_SECONDS
            self._is_calling_llm = False

    def _update_llm_trigger_state(self, angle, status):
        """【防抖核心逻辑】每一帧都要调用一次：判断是否已连续 Red 状态达到
        RED_DEBOUNCE_SECONDS 秒，如果是，就派生一个子线程去调用大模型。
        """
        current_time = time.time()

        if status != "Red":
            self._red_streak_start_time = None
            self._red_streak_already_triggered = False
            return

        if self._red_streak_start_time is None:
            self._red_streak_start_time = current_time
            return

        red_duration = current_time - self._red_streak_start_time
        if red_duration < RED_DEBOUNCE_SECONDS:
            return

        if self._red_streak_already_triggered:
            return

        with self._feedback_lock:
            if self._is_calling_llm:
                return
            self._is_calling_llm = True

        self._red_streak_already_triggered = True

        worker_thread = threading.Thread(
            target=self._call_llm_in_background, args=(angle, status), daemon=True
        )
        worker_thread.start()

    def _draw_feedback_text_if_needed(self, frame_bgr):
        """如果当前有还在"有效展示期"内的大模型反馈文本，就画在画面底部。"""
        with self._feedback_lock:
            text = self._feedback_text
            show_until = self._feedback_show_until

        if not text or time.time() > show_until:
            return frame_bgr

        return draw_chinese_text_with_backdrop(frame_bgr, text, FEEDBACK_FONT, anchor="bottom")

    # ----------------------------------------------------------------
    # B组专用：静默数据落盘逻辑
    # ----------------------------------------------------------------

    def _record_b_group_data(self, angle, status):
        """把这一帧的有效诊断数据，静默追加进内存里的记录列表（尚未写文件）。"""
        record = {
            "timestamp": time.time(),
            "knee_angle": round(float(angle), 1),
            "status": status,
        }
        self._b_group_new_records.append(record)

    def _flush_b_group_data_to_disk(self):
        """把本次训练新采集到的所有 B 组数据，追加合并进本地 JSON 文件并保存。

        为了保留历次训练的历史数据（方便后续 v1.0 阶段做离线报告聚合分析），
        这里采取"读取旧数据 -> 追加新数据 -> 整体重新写回"的方式，而不是
        每次开训练都覆盖掉之前的记录。
        """
        if not self._b_group_new_records:
            return

        existing_records = []
        if os.path.exists(B_GROUP_LOG_PATH):
            try:
                with open(B_GROUP_LOG_PATH, "r", encoding="utf-8") as f:
                    existing_records = json.load(f)
                if not isinstance(existing_records, list):
                    existing_records = []
            except Exception:  # noqa: BLE001 - 旧文件损坏时不影响本次新数据保存
                existing_records = []

        all_records = existing_records + self._b_group_new_records

        try:
            with open(B_GROUP_LOG_PATH, "w", encoding="utf-8") as f:
                json.dump(all_records, f, ensure_ascii=False, indent=2)
            self.log_message.emit(
                f"B组数据已静默保存：本次新增 {len(self._b_group_new_records)} 条，"
                f"文件累计 {len(all_records)} 条，路径：{B_GROUP_LOG_PATH}"
            )
        except Exception as exc:  # noqa: BLE001 - 写文件失败时提示但不崩溃
            self.log_message.emit(f"错误：B组数据保存失败：{exc}")

    # ----------------------------------------------------------------
    # 线程主体：run() —— QThread 的约定入口，start() 之后会自动在新线程里执行这里
    # ----------------------------------------------------------------

    def run(self):
        """后台线程真正执行的地方。原来 v0.1~v0.3 脚本里那个不断读摄像头的
        while 循环，现在完整地"搬家"到了这里，彻底和 GUI 主线程隔离开。
        """
        landmarker = None
        cap = None

        try:
            # ---------------- 1. 打开视频源（摄像头 或 本地视频文件） ----------------
            is_video_file_mode = (self.data_source == "video")

            if is_video_file_mode:
                if not os.path.exists(self.video_path):
                    self.log_message.emit(f"错误：未找到本地视频文件：{self.video_path}")
                    return
                cap = cv2.VideoCapture(self.video_path)
            else:
                cap = cv2.VideoCapture(self.camera_index)

            if not cap.isOpened():
                if is_video_file_mode:
                    self.log_message.emit(f"错误：无法打开本地视频文件：{self.video_path}")
                else:
                    self.log_message.emit("错误：无法打开摄像头，请检查是否被其他程序占用。")
                return

            # 本地视频文件模式下，按视频原始 FPS 控制播放速度，避免"快进"效果
            if is_video_file_mode:
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                if video_fps is None or video_fps <= 0:
                    video_fps = 30.0
                frame_delay_seconds = 1.0 / video_fps
                self.log_message.emit(
                    f"本地视频文件已打开：{self.video_path}（原始帧率 {video_fps:.1f} FPS）"
                )
            else:
                frame_delay_seconds = 0.0
                self.log_message.emit("摄像头已成功打开，正在实时检测……")

            # ---------------- 2. A/B/C 三组现在都需要用到 MediaPipe 姿态检测器 ----------------
            # 【v1.0 变更】C组虽然仍是"干净对照组"（不画骨骼线、不显示颜色框、
            # 不显示文字），但为了满足"面部绝对脱敏"的科技伦理红线，C组也必须
            # 跑一次姿态检测来定位面部关键点，用于后续的高斯模糊打码。
            need_pose_detection = self.group in ("A", "B", "C")

            if need_pose_detection:
                pose_options = mp_vision.PoseLandmarkerOptions(
                    base_options=mp_tasks_python.BaseOptions(model_asset_path=MODEL_PATH),
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)

            # ---------------- 3. B组：一开始就发送一次"黑屏 + 提示语"画面 ----------------
            # B组的核心实验设计要求是"课中黑屏"，右侧视频区绝不能出现摄像头/视频画面，
            # 所以这里只需要发送一次静态的黑屏提示画面即可，不用每一帧都重新生成发送。
            if self.group == "B":
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
                black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                black_frame = draw_chinese_text_with_backdrop(
                    black_frame, "正在采集中，请专心练习", BLACKSCREEN_FONT,
                    text_color=(230, 230, 230), anchor="center",
                )
                self.frame_ready.emit(bgr_frame_to_qimage(black_frame))
                self.log_message.emit("B组模式：右侧画面已切换为黑屏，后台静默采集数据中……")

            frame_timestamp_ms = 0

            # ---------------- 4. 主循环：逐帧读取、处理、分流 ----------------
            while self._running and cap.isOpened():
                loop_start_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    if is_video_file_mode:
                        self.log_message.emit("本地视频文件已播放完毕，训练自动结束。")
                    else:
                        self.log_message.emit("警告：读取画面失败，摄像头可能已断开。")
                    break

                # 摄像头模式下做左右镜像翻转，提升"自拍视角"的观感；
                # 本地视频文件的动作方向是固定的，不能镜像，否则左右腿会认错
                if not is_video_file_mode:
                    frame = cv2.flip(frame, 1)

                # ------------------- A/B/C 三组都先跑一次姿态检测 -------------------
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                frame_timestamp_ms += 33  # 近似按 30 fps 递增
                results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                if self.group == "C":
                    # ---- C组：干净对照组，不画骨骼线/颜色框/文字，但面部必须打码 ----
                    # 【v1.0 新增】科技伦理红线：即便是常规对照组，脸部特征也必须
                    # 彻底不可辨识，所以这里检测到人体后，先做一次面部高斯模糊，
                    # 再把画面发出去；检测不到人体时画面本身就没有脸，直接原样显示。
                    if results.pose_landmarks:
                        apply_face_blur(frame, results.pose_landmarks[0])
                    self.frame_ready.emit(bgr_frame_to_qimage(frame))

                elif results.pose_landmarks:
                    landmarks = results.pose_landmarks[0]
                    angle, status, color, hip_px, knee_px, ankle_px = (
                        compute_right_knee_diagnosis(frame, landmarks)
                    )

                    if self.group == "A":
                        # ---- A组：实时反馈，完整渲染骨骼 + 颜色框 + AIGC 中文指导语 ----
                        # 【v1.0 新增】必须先做面部打码，再画骨骼连线/关键点，
                        # 否则骨骼线会把模糊后的脸部区域再"划破"露出线条，
                        # 顺序绝不能颠倒（函数名里也特意强调了这一点）。
                        apply_face_blur(frame, landmarks)
                        draw_pose_landmarks(frame, results.pose_landmarks)
                        draw_right_knee_overlay(
                            frame, hip_px, knee_px, ankle_px, color, angle, status
                        )
                        self._update_llm_trigger_state(angle, status)
                        frame = self._draw_feedback_text_if_needed(frame)
                        self.frame_ready.emit(bgr_frame_to_qimage(frame))

                    elif self.group == "B":
                        # ---- B组：后台静默运行，绝不渲染骨骼/颜色/文字，绝不显示画面 ----
                        # 只把计算出的有效数据记录下来，右侧画面依然维持黑屏
                        # （黑屏画面已经在循环外发送过一次，这里不需要重复发送）
                        self._record_b_group_data(angle, status)
                else:
                    # 没检测到人体的帧，A组也需要把原始画面显示出来（否则画面会卡住）
                    if self.group == "A":
                        self.frame_ready.emit(bgr_frame_to_qimage(frame))

                # ------------------- 播放速度控制（仅本地视频文件模式需要） -------------------
                if is_video_file_mode and frame_delay_seconds > 0:
                    elapsed = time.time() - loop_start_time
                    remaining = frame_delay_seconds - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

        except Exception as exc:  # noqa: BLE001 - 后台线程内的任何异常都不能让程序崩溃
            self.log_message.emit(f"错误：后台处理线程发生异常：{exc}")

        finally:
            # ---------------- 5. 收尾：释放资源、B组落盘保存 ----------------
            if cap is not None:
                cap.release()
            if landmarker is not None:
                landmarker.close()
            if self.group == "B":
                self._flush_b_group_data_to_disk()
            self.log_message.emit("后台处理线程已安全退出，训练结束。")


# --------------------------------------------------------------------------
# 第四步：MainWindow —— PyQt5 主界面（GUI 线程），左右分栏布局
# --------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """程序主窗口：左侧控制面板 + 右侧视频展示区。

    这个类本身只负责"界面搭建"和"响应用户点击"，所有耗时的摄像头/视频
    处理逻辑全部委托给上面的 VideoWorker 后台线程完成，主窗口只是通过
    信号与槽（signal/slot）机制被动接收处理好的画面并显示出来，绝不直接操作摄像头。
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("小学足球AI可视化反馈系统 v0.4 —— 差异化双界面训练终端")
        self.resize(1280, 800)

        self.video_worker = None  # 当前正在运行的后台线程，未开始训练时为 None

        self._build_ui()

    # ----------------------------------------------------------------
    # 界面搭建
    # ----------------------------------------------------------------

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)

        # ============== 左侧：控制面板 ==============
        control_panel = self._build_control_panel()
        control_panel.setFixedWidth(340)
        root_layout.addWidget(control_panel)

        # ============== 右侧：视频展示区 ==============
        self.video_label = QLabel("请选择数据源与实验组别后，点击「开始训练」")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(880, 660)
        self.video_label.setStyleSheet(
            "background-color: #202020; color: #cccccc; font-size: 16px; border: 1px solid #444;"
        )
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root_layout.addWidget(self.video_label, stretch=1)

    def _build_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)

        # ---------- 数据源选择 ----------
        source_group_box = QGroupBox("① 数据源选择")
        source_layout = QVBoxLayout()

        self.radio_source_webcam = QRadioButton("摄像头（实时拍摄）")
        self.radio_source_video = QRadioButton("本地视频文件")
        self.radio_source_webcam.setChecked(True)

        self.source_button_group = QButtonGroup(self)
        self.source_button_group.addButton(self.radio_source_webcam)
        self.source_button_group.addButton(self.radio_source_video)

        # 本地视频文件路径输入框 + 浏览按钮（只有选中"本地视频文件"时才可用）
        video_path_layout = QHBoxLayout()
        self.video_path_edit = QLineEdit(DEFAULT_VIDEO_FILE_PATH)
        self.video_path_edit.setEnabled(False)
        self.browse_button = QPushButton("浏览…")
        self.browse_button.setEnabled(False)
        self.browse_button.clicked.connect(self._on_browse_video_file)
        video_path_layout.addWidget(self.video_path_edit)
        video_path_layout.addWidget(self.browse_button)

        # 根据单选按钮的选中状态，联动启用/禁用视频路径输入框
        self.radio_source_video.toggled.connect(self.video_path_edit.setEnabled)
        self.radio_source_video.toggled.connect(self.browse_button.setEnabled)

        source_layout.addWidget(self.radio_source_webcam)
        source_layout.addWidget(self.radio_source_video)
        source_layout.addLayout(video_path_layout)
        source_group_box.setLayout(source_layout)
        layout.addWidget(source_group_box)

        # ---------- 实验组别选择 ----------
        group_group_box = QGroupBox("② 实验组别选择")
        group_layout = QVBoxLayout()

        self.radio_group_a = QRadioButton("实验A组 —— 实时反馈")
        self.radio_group_b = QRadioButton("实验B组 —— 延时反馈")
        self.radio_group_c = QRadioButton("常规C组 —— 对照组")
        self.radio_group_a.setChecked(True)

        self.group_button_group = QButtonGroup(self)
        self.group_button_group.addButton(self.radio_group_a)
        self.group_button_group.addButton(self.radio_group_b)
        self.group_button_group.addButton(self.radio_group_c)

        group_layout.addWidget(self.radio_group_a)
        group_layout.addWidget(self.radio_group_b)
        group_layout.addWidget(self.radio_group_c)
        group_group_box.setLayout(group_layout)
        layout.addWidget(group_group_box)

        # ---------- 开始 / 结束按钮 ----------
        self.start_button = QPushButton("▶ 开始训练")
        self.start_button.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; font-size: 16px; "
            "padding: 10px; border-radius: 6px; } QPushButton:disabled { background-color: #777; }"
        )
        self.start_button.clicked.connect(self._on_start_clicked)

        self.stop_button = QPushButton("■ 结束训练")
        self.stop_button.setStyleSheet(
            "QPushButton { background-color: #c62828; color: white; font-size: 16px; "
            "padding: 10px; border-radius: 6px; } QPushButton:disabled { background-color: #777; }"
        )
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._on_stop_clicked)

        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        # ---------- 运行日志区域 ----------
        log_group_box = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setStyleSheet("font-size: 12px;")
        log_layout.addWidget(self.log_text_edit)
        log_group_box.setLayout(log_layout)
        layout.addWidget(log_group_box, stretch=1)

        return panel

    # ----------------------------------------------------------------
    # 事件响应
    # ----------------------------------------------------------------

    def _on_browse_video_file(self):
        """点击"浏览…"按钮：弹出文件选择对话框，让用户挑选本地视频文件。"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择本地视频文件", SCRIPT_DIR,
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)",
        )
        if file_path:
            self.video_path_edit.setText(file_path)

    def _on_start_clicked(self):
        """点击"开始训练"：读取当前界面上的选择，创建并启动后台线程 VideoWorker。"""
        if self.video_worker is not None and self.video_worker.isRunning():
            return  # 防止重复点击导致启动多个线程

        data_source = "webcam" if self.radio_source_webcam.isChecked() else "video"
        video_path = self.video_path_edit.text().strip()

        if self.radio_group_a.isChecked():
            group = "A"
        elif self.radio_group_b.isChecked():
            group = "B"
        else:
            group = "C"

        if data_source == "video" and not os.path.exists(video_path):
            QMessageBox.warning(self, "文件不存在", f"未找到本地视频文件：\n{video_path}")
            return

        self._append_log(f"开始训练：数据源={'摄像头' if data_source == 'webcam' else '本地视频'}，"
                          f"实验组别={group} 组")

        # 创建后台线程，把处理好的画面/日志信号连接到主窗口对应的槽函数
        self.video_worker = VideoWorker(
            data_source=data_source, video_path=video_path, group=group,
        )
        self.video_worker.frame_ready.connect(self._update_video_frame)
        self.video_worker.log_message.connect(self._append_log)
        # 线程自然结束（比如本地视频播放完毕）时，自动把界面恢复成"未运行"状态
        self.video_worker.finished.connect(self._on_worker_finished)
        self.video_worker.start()

        # 训练进行中，锁定所有选择控件，避免中途切换导致状态混乱
        self._set_controls_enabled(is_training=True)

    def _on_stop_clicked(self):
        """点击"结束训练"：只发出停止请求，真正的资源清理在 VideoWorker.run() 里完成。"""
        if self.video_worker is not None:
            self._append_log("正在请求结束训练，请稍候……")
            self.video_worker.request_stop()
        self.stop_button.setEnabled(False)  # 避免重复点击

    def _on_worker_finished(self):
        """VideoWorker 线程真正结束后触发（无论是用户主动停止，还是视频自然播放完毕）。"""
        self._set_controls_enabled(is_training=False)
        self.video_label.clear()
        self.video_label.setText("训练已结束，请选择数据源与实验组别后，点击「开始训练」")

    def _update_video_frame(self, qimage):
        """接收后台线程发来的处理好的画面，缩放后显示到右侧 QLabel 上。"""
        pixmap = QPixmap.fromImage(qimage)
        # 按 QLabel 当前尺寸等比例缩放，避免画面变形，同时避免超出可视区域
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def _append_log(self, message):
        """把一条日志文字追加显示到左下角的日志区域。"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text_edit.append(f"[{timestamp}] {message}")

    def _set_controls_enabled(self, is_training):
        """训练进行中/未进行中，切换各控件的可用状态。"""
        self.start_button.setEnabled(not is_training)
        self.stop_button.setEnabled(is_training)
        self.radio_source_webcam.setEnabled(not is_training)
        self.radio_source_video.setEnabled(not is_training)
        self.radio_group_a.setEnabled(not is_training)
        self.radio_group_b.setEnabled(not is_training)
        self.radio_group_c.setEnabled(not is_training)
        self.browse_button.setEnabled((not is_training) and self.radio_source_video.isChecked())
        self.video_path_edit.setEnabled((not is_training) and self.radio_source_video.isChecked())

    def closeEvent(self, event):
        """用户直接关闭窗口时，确保后台线程也被妥善停止，避免留下僵尸线程。"""
        if self.video_worker is not None and self.video_worker.isRunning():
            self.video_worker.request_stop()
            self.video_worker.wait(3000)  # 最多等待 3 秒，给线程收尾时间
        event.accept()


# --------------------------------------------------------------------------
# 第五步：程序入口
# --------------------------------------------------------------------------


def main():
    """程序主入口：先确保模型文件已下载，再启动 PyQt5 应用主循环。"""
    ensure_model_downloaded()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
