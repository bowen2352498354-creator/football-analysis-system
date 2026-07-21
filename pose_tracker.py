# -*- coding: utf-8 -*-
"""
pose_tracker.py
v0.4 差异化双界面开发阶段脚本（在 v0.3 AIGC 认知转译接入基础上迭代）
【V2.5】新增底层运动学清洗：KinematicSignalProcessor（Savitzky-Golay）
        + locate_impact_frame（双向交叉触球锁帧），消除单目帧间抖动导致的
        角度/触球时间戳跳变。

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
       继承自 QThread 的后台线程类 InferenceWorker 里运行。主线程（GUI 线程）
       里绝对不会出现任何"死循环读摄像头"的代码，否则界面会卡死甚至崩溃。
    2. 后台线程处理好每一帧画面后，通过自定义的 pyqtSignal 信号
       （frame_ready_signal / diagnostics_ready_signal）"发送"给主线程，
       主线程收到信号后只刷新 QLabel / 图表，非常轻量，完全不会阻塞 GUI。
    3. 【v0.3 遗留设计延续】即便是在后台线程 InferenceWorker 内部，调用 DeepSeek
       大模型这种"耗时的网络请求"依然会被丢进一个更内层的 threading.Thread
       子线程去执行，避免网络请求的等待时间拖慢 InferenceWorker 本身的摄像头
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

【v1.0 / 物理级面部脱敏红线】：
    公共图像模块 image_processing.apply_facial_anonymization 是不可逾越的
    拦截器：紧挨关键点提取之后、任何 frame_ready 发射或诊断关键帧落盘之前，
    必须强制用脱敏后的安全图像替换原始帧。A/C 组实时画面与击球关键帧均
    走此路径；B 组黑屏采集不渲染画面，天然满足。历史别名 apply_face_blur
    仍保留，内部直接委托给 apply_facial_anonymization。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
from typing import Any, Generator, List, Optional, Sequence, Tuple, Union

# --------------------------------------------------------------------------
# 【V2.5 Vision Pipeline Determinism Lockdown】必须在任何 CUDA/PyTorch/YOLO
# 初始化之前写入 CUBLAS 工作区配置，否则非确定性内核仍会生效。
# --------------------------------------------------------------------------
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def lock_vision_pipeline_determinism() -> dict[str, Any]:
    """锁死 PyTorch / cuDNN / CUBLAS 推理非确定性（像素级抖动源头）。

    若本机未安装 torch，静默跳过并返回状态字典，绝不抛异常。
    """
    status: dict[str, Any] = {
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_available": False,
        "cuda_available": False,
        "deterministic": False,
    }
    try:
        import torch  # noqa: WPS457 - 延迟导入，避免无 GPU 环境强依赖

        status["torch_available"] = True
        if torch.cuda.is_available():
            status["cuda_available"] = True
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                # 旧版 torch 无 warn_only 参数
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:  # noqa: BLE001
                    pass
            status["deterministic"] = True
        else:
            # CPU 路径同样尽量锁死（对部分算子仍有意义）
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                status["deterministic"] = True
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001 - torch 缺失或初始化失败时忽略
        status["error"] = str(exc)
    return status


# 模块 import 即执行一次确定性锁死
_DETERMINISM_STATUS: dict[str, Any] = lock_vision_pipeline_determinism()

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import savgol_filter

from PyQt5.QtGui import QImage

from image_processing import (
    FACE_ANON_EXPAND_RATIO,
    FACE_ANON_KERNEL_SIZE,
    FACE_ANON_MIN_KERNEL_SIZE,
    FACE_ANON_SIGMA_X,
    FACE_LANDMARK_INDICES,
    apply_facial_anonymization,
)

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

# 【实验组路由】B 组静默采集的本地生物力学宽表（CSV，按射门回合追加）
B_GROUP_WIDE_TABLE_PATH = os.path.join(SCRIPT_DIR, "B_group_wide_table.csv")

# 触球锁帧 / 射门周期 / 时空胶囊常量
# 【V3.1 Sprint 2】滚动时间机器：deque(maxlen=150) ≈ 5s @30fps
SHOT_FRAME_BUFFER_MAX = 150
SHOT_PRE_IMPACT_FRAMES = 60  # 触球前核心窗
SHOT_POST_IMPACT_FRAMES = 30  # 触球后核心窗（合计 90 帧切片）
SHOT_OMEGA_PEAK_THRESHOLD = 80.0  # deg/s，判定疑似鞭打峰 / APPROACH
SHOT_IMPACT_COOLDOWN_SEC = 3.5  # COOLDOWN 防抖（3–5s）
MIN_FRAMES_FOR_IMPACT_LOCK = 15
CAPSULE_DURATION_MIN_SEC = 8.0
CAPSULE_DURATION_MAX_SEC = 12.0
CAPSULE_HALF_SPEED_FACTOR = 2.0  # 半速：每帧停留时间为正常的 2 倍
CAPTURE_DISCARD_HINT = "本轮捕获失败，请准备下一球"
AUTO_CAPTURE_CLIPS_DIR = os.path.join(SCRIPT_DIR, "auto_capture_clips")

# 抛物线锁帧判定为「无解」的质量阈值（反光 / 遮挡 / 信号塌缩）
_IMPACT_OMEGA_FLAT_EPS = 1e-3
_IMPACT_DIST_REL_RANGE_MIN = 0.02  # 搜索窗内距离相对极差过小 → 无清晰触球谷
_IMPACT_MAX_VALID_DIST = 1e6


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
# 【物理级面部脱敏】常量由 image_processing 公共模块统一持有；此处再导出，
# 保持旧代码 ``from pose_tracker import FACE_LANDMARK_INDICES`` 等写法可用。
# --------------------------------------------------------------------------
FACE_BLUR_EXPAND_RATIO = FACE_ANON_EXPAND_RATIO
FACE_BLUR_MIN_KERNEL_SIZE = FACE_ANON_MIN_KERNEL_SIZE
FACE_BLUR_KERNEL_SIZE = FACE_ANON_KERNEL_SIZE
FACE_BLUR_SIGMA_X = FACE_ANON_SIGMA_X

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

# 支撑脚 / 摆腿热力图所需关节点编号（与 MediaPipe Pose 拓扑一致）
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
LEFT_HIP_IDX = 23
LEFT_KNEE_IDX = 25
LEFT_ANKLE_IDX = 27
LEFT_HEEL_IDX = 29
RIGHT_HEEL_IDX = 30
LEFT_FOOT_INDEX_IDX = 31
RIGHT_FOOT_INDEX_IDX = 32

# 热力图 / 诊断帧序列：关节名 → MediaPipe 下标
HEATMAP_JOINT_INDICES: dict[str, int] = {
    "left_shoulder": LEFT_SHOULDER_IDX,
    "right_shoulder": RIGHT_SHOULDER_IDX,
    "left_hip": LEFT_HIP_IDX,
    "right_hip": RIGHT_HIP_IDX,
    "left_knee": LEFT_KNEE_IDX,
    "right_knee": RIGHT_KNEE_IDX,
    "left_ankle": LEFT_ANKLE_IDX,
    "right_ankle": RIGHT_ANKLE_IDX,
    "left_heel": LEFT_HEEL_IDX,
    "right_heel": RIGHT_HEEL_IDX,
    "left_foot_index": LEFT_FOOT_INDEX_IDX,
    "right_foot_index": RIGHT_FOOT_INDEX_IDX,
}

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


def serialize_pose_frame_record(
    single_person_landmarks,
    frame_shape,
    *,
    timestamp_sec: float = 0.0,
    world_landmarks=None,
) -> dict:
    """把 MediaPipe 单人关键点序列化为 error_diagnoser 兼容的逐帧字典。

    - 图像平面坐标：像素 (x*W, y*H, z*W)
    - 若提供 pose_world_landmarks：写入 ``world`` 子字典（米制），供热力图优先使用
    """
    height = int(frame_shape[0]) if frame_shape is not None else 1
    width = int(frame_shape[1]) if frame_shape is not None and len(frame_shape) > 1 else 1
    width = max(1, width)
    height = max(1, height)

    def _image_xyz(lm) -> list[float]:
        return [float(lm.x) * width, float(lm.y) * height, float(getattr(lm, "z", 0.0)) * width]

    def _world_xyz(lm) -> list[float]:
        return [float(lm.x), float(lm.y), float(lm.z)]

    record: dict = {"timestamp_sec": float(timestamp_sec)}
    visibility: dict = {}
    for name, idx in HEATMAP_JOINT_INDICES.items():
        try:
            lm = single_person_landmarks[idx]
        except (IndexError, TypeError, KeyError):
            continue
        record[name] = _image_xyz(lm)
        try:
            visibility[name] = float(getattr(lm, "visibility", 1.0) or 1.0)
        except (TypeError, ValueError):
            visibility[name] = 1.0
    record["visibility"] = visibility

    if world_landmarks is not None:
        world: dict = {}
        for name, idx in HEATMAP_JOINT_INDICES.items():
            try:
                wlm = world_landmarks[idx]
            except (IndexError, TypeError, KeyError):
                continue
            world[name] = _world_xyz(wlm)
        if world:
            record["world"] = world
    return record


def empty_pose_frame_record(timestamp_sec: float = 0.0) -> dict:
    """无姿态帧占位：保持轨迹下标与 sync_frame_count 对齐。"""
    zero = [0.0, 0.0, 0.0]
    joints = list(HEATMAP_JOINT_INDICES.keys())
    record = {name: list(zero) for name in joints}
    record["timestamp_sec"] = float(timestamp_sec)
    record["visibility"] = {name: 0.0 for name in joints}
    return record


# --------------------------------------------------------------------------
# 【V2.5 底层架构】运动学时序信号清洗 + 双向交叉触球锁帧
# --------------------------------------------------------------------------
# 单目摄像头帧间抖动会导致同一视频多次重跑时，关节角度与触球时间戳发生跳变。
# 本段通过 Savitzky-Golay 时序滤波压低高频抖动，再用「角速度极值 + 踝-球
# 欧氏距离」双向交叉锁定全局唯一的确定性触球帧 t_impact。
# --------------------------------------------------------------------------

# Savitzky-Golay 默认参数（写死，保证跨次复现结果一致）
SAVGOL_WINDOW_LENGTH: int = 7
SAVGOL_POLYORDER: int = 3

# 触球锁帧：以角速度极值点为中心的前后搜索半窗（帧）
IMPACT_SEARCH_HALF_WINDOW: int = 5

# 坐标点类型：二维 (x, y) 或三维 (x, y, z)，元素为浮点或可转浮点
PointLike = Union[Tuple[float, ...], Sequence[float]]


class KinematicSignalProcessor:
    """【V2.5】运动学时序信号清洗器。

    对连续帧关节角度等一维时序信号施加 Savitzky-Golay 滤波，抑制单目姿态
    估计的帧间高频抖动，同时尽量保留踢球动作中的真实加速度峰值形态。

    设计约束：
        - 默认窗口 / 阶数写死为 window_length=7、polyorder=3，保证确定性复现；
        - 短序列自动降级（奇数窗或移动平均），严禁因长度不足抛出异常。
    """

    DEFAULT_WINDOW_LENGTH: int = SAVGOL_WINDOW_LENGTH
    DEFAULT_POLYORDER: int = SAVGOL_POLYORDER

    @staticmethod
    def _to_float64_array(raw_angles_sequence: Sequence[float]) -> np.ndarray:
        """将任意可迭代角度序列安全转为 float64 一维向量；非法值按 0.0 填补。"""
        if raw_angles_sequence is None:
            return np.asarray([], dtype=np.float64)
        values: List[float] = []
        for item in raw_angles_sequence:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                values.append(0.0)
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def _simple_moving_average(signal: np.ndarray, window: int) -> np.ndarray:
        """短序列降级方案：奇数窗中心对齐的简单移动平均（边界用边缘填充）。"""
        n: int = int(signal.shape[0])
        if n == 0:
            return signal.copy()
        # 保证 window 为 >=1 的奇数，且不超过序列长度
        window = max(1, min(int(window), n))
        if window % 2 == 0:
            window = window - 1 if window > 1 else 1
        if window <= 1:
            return signal.copy()

        half: int = window // 2
        padded: np.ndarray = np.pad(signal, (half, half), mode="edge")
        kernel: np.ndarray = np.ones(window, dtype=np.float64) / float(window)
        # 有效卷积长度 = n，与原序列一一对应
        smoothed: np.ndarray = np.convolve(padded, kernel, mode="valid")
        return smoothed.astype(np.float64, copy=False)

    @classmethod
    def _resolve_savgol_params(cls, n: int) -> Tuple[int, int] | None:
        """根据序列长度解析可用的 (window_length, polyorder)。

        返回 None 表示应降级为简单移动平均（或原样返回）。
        """
        if n < 3:
            return None

        if n >= cls.DEFAULT_WINDOW_LENGTH:
            window_length: int = cls.DEFAULT_WINDOW_LENGTH
            polyorder: int = cls.DEFAULT_POLYORDER
        else:
            # 长度 < 7：取不超过 n 的最大奇数窗（且 >= 3）
            window_length = n if (n % 2 == 1) else (n - 1)
            if window_length < 3:
                return None
            # polyorder 必须严格小于 window_length
            polyorder = min(cls.DEFAULT_POLYORDER, window_length - 1)

        if polyorder < 1 or window_length <= polyorder:
            return None
        return window_length, polyorder

    @classmethod
    def smooth_joint_trajectories(
        cls,
        raw_angles_sequence: List[float],
    ) -> List[float]:
        """对连续帧关节角度时序做 Savitzky-Golay 平滑。

        参数:
            raw_angles_sequence: 原始角度序列（如摆动腿膝关节 / 髋关节逐帧角度）。

        返回:
            与输入等长的平滑后角度列表（float）。短序列自动降级，永不抛异常。
        """
        try:
            signal: np.ndarray = cls._to_float64_array(raw_angles_sequence)
            n: int = int(signal.shape[0])

            if n == 0:
                return []
            if n < 3:
                # 不足以构成任何有意义的奇数窗，原样返回（拷贝，避免外部被原地修改）
                return [float(v) for v in signal]

            params = cls._resolve_savgol_params(n)
            if params is None:
                # 降级：用长度允许的最大奇数窗做简单移动平均
                fallback_window: int = n if (n % 2 == 1) else max(1, n - 1)
                smoothed = cls._simple_moving_average(signal, fallback_window)
                return [float(v) for v in smoothed]

            window_length, polyorder = params
            try:
                smoothed = savgol_filter(
                    signal,
                    window_length=window_length,
                    polyorder=polyorder,
                    mode="interp",
                )
            except Exception:  # noqa: BLE001 - 滤波失败时降级，严禁向上抛出
                smoothed = cls._simple_moving_average(signal, window_length)

            # 保证输出长度与输入严格一致
            if smoothed.shape[0] != n:
                smoothed = cls._simple_moving_average(signal, window_length)

            return [float(v) for v in np.asarray(smoothed, dtype=np.float64)]
        except Exception:  # noqa: BLE001 - 顶层兜底：任何意外都退回原始可解析序列
            try:
                return [float(v) for v in cls._to_float64_array(raw_angles_sequence)]
            except Exception:  # noqa: BLE001
                return []


def _euclidean_distance(point_a: PointLike, point_b: PointLike) -> float:
    """高精度欧氏距离（float64）。维度取两者较短公共维；无效点返回 +inf。"""
    try:
        a = np.asarray(point_a, dtype=np.float64).reshape(-1)
        b = np.asarray(point_b, dtype=np.float64).reshape(-1)
        if a.size == 0 or b.size == 0:
            return float("inf")
        dim: int = int(min(a.size, b.size))
        delta: np.ndarray = a[:dim] - b[:dim]
        # 使用 hypot 链式累加，避免大坐标下平方和溢出，保持数值稳定
        dist: float = float(np.linalg.norm(delta, ord=2))
        if not np.isfinite(dist):
            return float("inf")
        return dist
    except Exception:  # noqa: BLE001
        return float("inf")


def _parabolic_vertex_frame_offset(
    dist_prev: float,
    dist_center: float,
    dist_next: float,
) -> float:
    """对三点距离做二次多项式拟合，返回相对中心帧的亚帧顶点偏移。

    模型：以中心为 t=0，拟合 d(t)=a t² + b t + c。
    顶点 t* = -b/(2a)。仅当 a>0（开口向上=局部极小）时采用；否则返回 0。
    """
    y0 = float(dist_prev)
    y1 = float(dist_center)
    y2 = float(dist_next)
    if not (np.isfinite(y0) and np.isfinite(y1) and np.isfinite(y2)):
        return 0.0
    # a = (y0 + y2)/2 - y1 ; b = (y2 - y0)/2
    a_coef: float = 0.5 * (y0 + y2) - y1
    b_coef: float = 0.5 * (y2 - y0)
    if a_coef <= 1e-15:
        # 非严格凹极小（平坦或开口向下）：不偏移，保持离散中心
        return 0.0
    vertex: float = -b_coef / (2.0 * a_coef)
    # 亚帧偏移钳制在邻域 (-1, 1)，避免数值噪声跳出三点窗
    if vertex < -1.0:
        return -1.0
    if vertex > 1.0:
        return 1.0
    return float(vertex)


def _nearest_omega_zero_crossing(
    omega: np.ndarray,
    center_idx: int,
    search_lo: int,
    search_hi: int,
) -> Optional[int]:
    """在搜索窗内寻找最靠近 center_idx 的角速度零点交叉帧（符号翻转后一帧）。

    用于邻帧距离极接近时的确定性平局裁决，杜绝 ±1 帧翻转。
    """
    best_cross: Optional[int] = None
    best_rank: Tuple[int, int] = (10**9, 10**9)  # (|Δi|, i) 字典序
    for i in range(max(search_lo + 1, 1), search_hi + 1):
        w0 = float(omega[i - 1])
        w1 = float(omega[i])
        if not (np.isfinite(w0) and np.isfinite(w1)):
            continue
        if w0 == 0.0 or w1 == 0.0 or (w0 > 0.0) != (w1 > 0.0):
            # 零点或符号翻转：取交叉后帧 i
            rank = (abs(i - center_idx), i)
            if rank < best_rank:
                best_rank = rank
                best_cross = i
    return best_cross


def locate_impact_frame(
    smoothed_knee_angular_velocities: List[float],
    ankle_coordinates: List[Tuple],
    ball_coordinates: List[Tuple],
) -> int:
    """【V2.5+】双向交叉触球锁帧 + 抛物线插值平滑：确定性零漂移 t_impact。

    算法步骤（同一视频序列必须零漂移复现）:
        1) 在 ``smoothed_knee_angular_velocities`` 中定位摆动腿小腿鞭打的
           最高速度极值点索引 ``idx_max_omega``（取 |ω| 最大帧）；
        2) 以该索引为中心，在闭区间搜索窗
           ``[idx_max_omega - 5, idx_max_omega + 5]`` 内，逐帧计算脚踝坐标与
           球心坐标的欧氏距离，得到离散极小值候选 ``idx_min_dist``；
        3) 对 ``[idx-1, idx, idx+1]`` 三点距离做二次多项式（抛物线）拟合，
           估算曲率最低点的亚帧顶点，再四舍五入到整数帧；若邻帧距离极接近，
           再结合角速度零点交叉做确定性平局裁决；
        4) 返回全局唯一触球时间戳 ``t_impact``。

    参数:
        smoothed_knee_angular_velocities: 已平滑的摆动腿膝关节角速度时序（度/秒等）。
        ankle_coordinates: 与帧对齐的脚踝坐标序列，元素为 (x, y[, z])。
        ball_coordinates: 与帧对齐的球心坐标序列，元素为 (x, y[, z])。

    返回:
        触球帧的全局整数索引 ``t_impact``。输入异常时尽量回退到安全索引，不抛异常。
    """
    # ---------- 输入归一化（float64） ----------
    try:
        omega = np.asarray(
            KinematicSignalProcessor._to_float64_array(smoothed_knee_angular_velocities),
            dtype=np.float64,
        )
    except Exception:  # noqa: BLE001
        omega = np.asarray([], dtype=np.float64)

    n_omega: int = int(omega.shape[0])
    n_ankle: int = len(ankle_coordinates) if ankle_coordinates is not None else 0
    n_ball: int = len(ball_coordinates) if ball_coordinates is not None else 0

    # 有效对齐长度：三者取最短，保证索引不会越界
    n: int = int(min(n_omega, n_ankle, n_ball)) if (n_omega > 0 and n_ankle > 0 and n_ball > 0) else 0

    if n <= 0:
        if n_omega > 0:
            return int(np.argmax(np.abs(omega)))
        return 0

    omega_valid: np.ndarray = omega[:n]

    # ---------- 第一步：摆动腿鞭打最高速度极值点 ----------
    idx_max_omega: int = int(np.argmax(np.abs(omega_valid)))

    # ---------- 第二步：局部搜索窗内踝-球欧氏距离 ----------
    search_lo: int = max(0, idx_max_omega - IMPACT_SEARCH_HALF_WINDOW)
    search_hi: int = min(n - 1, idx_max_omega + IMPACT_SEARCH_HALF_WINDOW)

    distances: dict[int, float] = {}
    best_idx: int = idx_max_omega
    best_dist: float = float("inf")

    for i in range(search_lo, search_hi + 1):
        try:
            ankle_pt: PointLike = ankle_coordinates[i]
            ball_pt: PointLike = ball_coordinates[i]
        except (IndexError, TypeError):
            continue

        dist: float = _euclidean_distance(ankle_pt, ball_pt)
        distances[i] = dist
        # 严格 < ：并列时保留先出现帧（更靠近扫描起点 / 角速度峰侧的确定性）
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    # ---------- 第三步：抛物线插值细化，杜绝邻帧 ±1 翻转 ----------
    idx_min: int = int(best_idx)
    if (idx_min - 1) in distances and (idx_min + 1) in distances:
        d_prev = distances[idx_min - 1]
        d_mid = distances[idx_min]
        d_next = distances[idx_min + 1]
        offset = _parabolic_vertex_frame_offset(d_prev, d_mid, d_next)
        # 四舍五入到最近整数帧（banker's rounding 对 *.5 用「远离零」保证确定性：
        # 这里用 floor(x+0.5) 对非负偏移、ceil(x-0.5) 对负偏移，等价于远离零的半入）
        if offset >= 0.0:
            rounded_off = int(np.floor(offset + 0.5))
        else:
            rounded_off = int(np.ceil(offset - 0.5))
        refined = int(np.clip(idx_min + rounded_off, search_lo, search_hi))

        # 邻帧距离极接近时：用角速度零交叉做二次确定性裁决
        near_eps = max(1e-9, 1e-4 * max(d_mid, 1.0))
        neighbors_near = (
            abs(d_prev - d_mid) <= near_eps or abs(d_next - d_mid) <= near_eps
        )
        if neighbors_near:
            cross = _nearest_omega_zero_crossing(
                omega_valid, refined, search_lo, search_hi
            )
            if cross is not None:
                # 在 refined 与 cross 之间，取距角速度峰更近者；仍平局则取较小索引
                cand = (refined, cross)
                refined = min(
                    cand,
                    key=lambda j: (abs(j - idx_max_omega), j),
                )
        idx_min = int(refined)

    # ---------- 第四步：全局唯一触球时间戳 t_impact ----------
    t_impact: int = int(np.clip(idx_min, 0, n - 1))
    return t_impact


class ImpactFrameLocator:
    """【V2.5】触球锁帧定位器：对 ``locate_impact_frame`` 的面向对象封装。

    桌面端 ``VideoWorker`` 与 Web ``api_server`` 均应通过本类调用锁帧逻辑，
    避免在 GUI 主线程直接散落调用底层函数。
    """

    SEARCH_HALF_WINDOW: int = IMPACT_SEARCH_HALF_WINDOW

    @classmethod
    def locate(
        cls,
        smoothed_knee_angular_velocities: List[float],
        ankle_coordinates: List[Tuple],
        ball_coordinates: List[Tuple],
    ) -> int:
        """返回确定性触球帧索引 ``t_impact``（输入异常时安全回退，不抛异常）。"""
        return locate_impact_frame(
            smoothed_knee_angular_velocities,
            ankle_coordinates,
            ball_coordinates,
        )

    @classmethod
    def locate_with_ball_proxy(
        cls,
        smoothed_omega: List[float],
        ankle_coordinates: List[Tuple],
    ) -> Tuple[int, List[Tuple]]:
        """无独立球检测时：用踝轨迹高 Y 分位中位数近似静止球心，再锁帧。

        返回:
            (t_impact, ball_coordinates) —— ball_coordinates 与 ankle 等长。
        """
        n = min(len(smoothed_omega), len(ankle_coordinates))
        if n <= 0:
            return 0, []
        ankles = list(ankle_coordinates[:n])
        ys = [float(a[1]) if len(a) > 1 else 0.0 for a in ankles]
        xs = [float(a[0]) if len(a) > 0 else 0.0 for a in ankles]
        order = sorted(range(n), key=lambda i: ys[i])
        tail = order[int(n * 0.85) :] or order[-1:]
        ball_x = float(sum(xs[i] for i in tail) / len(tail))
        ball_y = float(sum(ys[i] for i in tail) / len(tail))
        ball_coords: List[Tuple] = [(ball_x, ball_y) for _ in range(n)]
        t_impact = cls.locate(list(smoothed_omega[:n]), ankles, ball_coords)
        return int(t_impact), ball_coords

    @classmethod
    def try_locate_or_discard(
        cls,
        smoothed_knee_angular_velocities: List[float],
        ankle_coordinates: List[Tuple],
        ball_coordinates: List[Tuple],
    ) -> Tuple[Optional[int], dict[str, Any]]:
        """抛物线锁帧灾备入口：无解时 Discard，返回 ``(None, meta)``，绝不抛致死异常。"""
        return try_locate_impact_or_discard(
            smoothed_knee_angular_velocities,
            ankle_coordinates,
            ball_coordinates,
        )


def try_locate_impact_or_discard(
    smoothed_knee_angular_velocities: List[float],
    ankle_coordinates: List[Tuple],
    ball_coordinates: List[Tuple],
) -> Tuple[Optional[int], dict[str, Any]]:
    """抛物线触球锁帧的灾备入口：无解时 Discard，绝不抛致死异常。

    强烈反光 / 严重遮挡 / 信号塌缩时返回 ``(None, meta)``，由 Worker
    跳过本轮打分并通过 ``capture_discarded_signal`` 提示前端继续下一球。
    """
    meta: dict[str, Any] = {
        "ok": False,
        "discarded": False,
        "discard_reason": None,
        "t_impact": None,
    }
    try:
        omega = np.asarray(
            KinematicSignalProcessor._to_float64_array(smoothed_knee_angular_velocities),
            dtype=np.float64,
        )
    except Exception as exc:  # noqa: BLE001
        meta.update({"discarded": True, "discard_reason": f"omega_parse_error:{exc}"})
        return None, meta

    n_omega = int(omega.shape[0])
    n_ankle = len(ankle_coordinates) if ankle_coordinates is not None else 0
    n_ball = len(ball_coordinates) if ball_coordinates is not None else 0
    n = int(min(n_omega, n_ankle, n_ball)) if (n_omega > 0 and n_ankle > 0 and n_ball > 0) else 0

    if n < MIN_FRAMES_FOR_IMPACT_LOCK:
        meta.update(
            {
                "discarded": True,
                "discard_reason": "insufficient_frames",
                "sample_frame_count": n,
            }
        )
        return None, meta

    omega_valid = omega[:n]
    abs_peak = float(np.max(np.abs(omega_valid))) if n > 0 else 0.0
    if not np.isfinite(abs_peak) or abs_peak <= _IMPACT_OMEGA_FLAT_EPS:
        meta.update({"discarded": True, "discard_reason": "no_whip_signal_glare_or_occlusion"})
        return None, meta

    idx_max_omega = int(np.argmax(np.abs(omega_valid)))
    search_lo = max(0, idx_max_omega - IMPACT_SEARCH_HALF_WINDOW)
    search_hi = min(n - 1, idx_max_omega + IMPACT_SEARCH_HALF_WINDOW)

    distances: dict[int, float] = {}
    for i in range(search_lo, search_hi + 1):
        try:
            ankle_pt: PointLike = ankle_coordinates[i]
            ball_pt: PointLike = ball_coordinates[i]
        except (IndexError, TypeError):
            continue
        dist = _euclidean_distance(ankle_pt, ball_pt)
        if not np.isfinite(dist) or dist >= _IMPACT_MAX_VALID_DIST:
            continue
        try:
            ax, ay = float(ankle_pt[0]), float(ankle_pt[1])
            bx, by = float(ball_pt[0]), float(ball_pt[1])
        except (TypeError, ValueError, IndexError):
            continue
        if abs(ax) + abs(ay) < 1e-9 and abs(bx) + abs(by) < 1e-9:
            continue
        distances[i] = dist

    if len(distances) < 3:
        meta.update(
            {
                "discarded": True,
                "discard_reason": "occlusion_or_glare_missing_landmarks",
                "valid_distance_samples": len(distances),
            }
        )
        return None, meta

    dist_vals = list(distances.values())
    d_min, d_max = float(min(dist_vals)), float(max(dist_vals))
    rel_range = (d_max - d_min) / max(d_max, 1.0)
    if rel_range < _IMPACT_DIST_REL_RANGE_MIN:
        meta.update(
            {
                "discarded": True,
                "discard_reason": "ambiguous_parabola_no_contact_valley",
                "distance_rel_range": rel_range,
            }
        )
        return None, meta

    try:
        t_impact = int(
            locate_impact_frame(
                list(omega_valid),
                list(ankle_coordinates[:n]),
                list(ball_coordinates[:n]),
            )
        )
        t_impact = int(max(0, min(n - 1, t_impact)))
    except Exception as exc:  # noqa: BLE001 - 锁帧引擎任何异常均 Discard
        meta.update({"discarded": True, "discard_reason": f"parabolic_lock_exception:{exc}"})
        return None, meta

    meta.update(
        {
            "ok": True,
            "discarded": False,
            "t_impact": t_impact,
            "shank_omega_peak_value": abs_peak,
            "distance_rel_range": rel_range,
            "t0_method": "try_locate_impact_or_discard_v1",
        }
    )
    return t_impact, meta


# --------------------------------------------------------------------------
# 【V2.5】视觉推理层确定性锁死：模型生命周期 + 同步顺序帧读取
# --------------------------------------------------------------------------
# MediaPipe PoseLandmarker / YOLO 均带跨帧跟踪记忆。若复用旧实例，同一视频
# 第二次分析的第 0 帧会继承上一次尾部状态，导致角度与 t_impact 漂移。
# 规则：每次 start_analysis_task() 必须销毁旧实例并重建；视频文件必须同步
# 阻断式逐帧读取，禁止丢帧缓冲队列。
# --------------------------------------------------------------------------

# 进程内当前活跃的姿态/球检测器句柄（仅允许通过本模块 API 访问）
_ACTIVE_POSE_LANDMARKER: Any = None
_ACTIVE_YOLO_MODEL: Any = None
_ANALYSIS_TASK_LOCK = threading.Lock()


def destroy_pose_landmarker(landmarker: Any = None) -> None:
    """显式销毁 PoseLandmarker，清空跨帧跟踪记忆。"""
    global _ACTIVE_POSE_LANDMARKER
    target = landmarker if landmarker is not None else _ACTIVE_POSE_LANDMARKER
    if target is None:
        return
    try:
        close_fn = getattr(target, "close", None)
        if callable(close_fn):
            close_fn()
    except Exception:  # noqa: BLE001
        pass
    if target is _ACTIVE_POSE_LANDMARKER or landmarker is None:
        _ACTIVE_POSE_LANDMARKER = None


def destroy_yolo_tracker(model: Any = None) -> None:
    """显式销毁 / 重置 YOLO 跟踪器状态（若项目接入了 ultralytics）。"""
    global _ACTIVE_YOLO_MODEL
    target = model if model is not None else _ACTIVE_YOLO_MODEL
    if target is None:
        return
    try:
        # ultralytics：predictor/trackers 可能挂在 model.predictor 上
        predictor = getattr(target, "predictor", None)
        if predictor is not None:
            for attr in ("trackers", "tracker", "vid_path", "vid_stride"):
                if hasattr(predictor, attr):
                    try:
                        setattr(predictor, attr, None)
                    except Exception:  # noqa: BLE001
                        pass
            reset_fn = getattr(predictor, "reset", None)
            if callable(reset_fn):
                try:
                    reset_fn()
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        pass
    if target is _ACTIVE_YOLO_MODEL or model is None:
        _ACTIVE_YOLO_MODEL = None


def create_fresh_pose_landmarker(
    *,
    model_path: Optional[str] = None,
    num_poses: int = 1,
    min_pose_detection_confidence: float = 0.5,
    min_pose_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> Any:
    """创建全新 PoseLandmarker（VIDEO 模式）。调用前会销毁进程内旧实例。"""
    global _ACTIVE_POSE_LANDMARKER
    ensure_model_downloaded()
    destroy_pose_landmarker()
    asset = model_path or MODEL_PATH
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks_python.BaseOptions(model_asset_path=asset),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)
    _ACTIVE_POSE_LANDMARKER = landmarker
    return landmarker


def create_fresh_yolo_model(weights: str = "yolov8n.pt") -> Any:
    """创建全新 YOLO 模型；禁用跨帧轨迹记忆（tracker=None）。

    未安装 ultralytics 时返回 None（不抛异常）。
    """
    global _ACTIVE_YOLO_MODEL
    destroy_yolo_tracker()
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:  # noqa: BLE001
        _ACTIVE_YOLO_MODEL = None
        return None
    model = YOLO(weights)
    _ACTIVE_YOLO_MODEL = model
    return model


def yolo_detect_frame(model: Any, frame_bgr: np.ndarray) -> Any:
    """对单帧做 YOLO 检测：强制 tracker=None / persist=False，杜绝轨迹残留。"""
    if model is None:
        return None
    try:
        return model.predict(
            source=frame_bgr,
            verbose=False,
            tracker=None,
            persist=False,
        )
    except TypeError:
        # 旧版 ultralytics 可能不接受 tracker/persist
        try:
            return model.predict(source=frame_bgr, verbose=False)
        except Exception:  # noqa: BLE001
            return None
    except Exception:  # noqa: BLE001
        return None


def start_analysis_task(
    *,
    reset_yolo: bool = True,
    yolo_weights: Optional[str] = None,
) -> dict[str, Any]:
    """【V2.5】每次新建分析任务的唯一入口：销毁旧模型并重建干净实例。

    严禁复用上一次的 PoseLandmarker / YOLO 跟踪器。返回新建句柄字典：
        {"pose_landmarker": ..., "yolo_model": ... | None, "determinism": ...}
    """
    with _ANALYSIS_TASK_LOCK:
        lock_vision_pipeline_determinism()
        destroy_pose_landmarker()
        if reset_yolo:
            destroy_yolo_tracker()
        landmarker = create_fresh_pose_landmarker()
        yolo_model = None
        if reset_yolo and yolo_weights is not None:
            yolo_model = create_fresh_yolo_model(yolo_weights)
        elif reset_yolo:
            # 不强制加载权重：仅清空旧 YOLO 状态，由调用方按需 create
            yolo_model = None
        return {
            "pose_landmarker": landmarker,
            "yolo_model": yolo_model,
            "determinism": dict(_DETERMINISM_STATUS),
        }


def open_video_capture_deterministic(
    video_path: str,
    *,
    is_camera: bool = False,
    camera_index: int = 0,
) -> Tuple[Any, float, int]:
    """打开视频源并配置为确定性读取。

    返回 (cap, fps, reported_frame_count)。
    - 录像文件：禁止依赖缓冲区取「最新帧」；缓冲区尽量置 0/1。
    - 摄像头：BUFFERSIZE=1，仅降低实时延迟，分析仍按读到的每一帧顺序处理。
    """
    if is_camera:
        cap = cv2.VideoCapture(camera_index)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001
            pass
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        return cap, fps, -1

    cap = cv2.VideoCapture(video_path)
    # 部分后端支持将内部缓冲压到最小，避免「跳到最新帧」语义
    for buf_size in (0, 1):
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, buf_size)
            break
        except Exception:  # noqa: BLE001
            continue
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 1e-6:
        fps = 30.0
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, float(fps), reported


def iter_video_frames_sync(
    video_path: str,
) -> Generator[Tuple[int, np.ndarray, float], None, None]:
    """【严格同步顺序帧】阻断式逐帧遍历本地视频，严禁丢帧/跳帧。

    Yields:
        (frame_index, frame_bgr, timestamp_sec)
        其中 timestamp_sec = frame_index / fps（与墙钟无关，保证跨次复现）。
    """
    cap, fps, _reported = open_video_capture_deterministic(video_path, is_camera=False)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"无法打开视频文件：{video_path}")
    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp_sec = float(frame_index) / float(fps)
            yield frame_index, frame, timestamp_sec
            frame_index += 1
    finally:
        cap.release()


def detect_contact_frame(
    frames: List[dict],
    ball_center: Any = None,
) -> Tuple[int, np.ndarray, dict[str, Any]]:
    """从逐帧关键点字典序列锁定触球帧（供 error_diagnoser.lock_absolute_t0 调用）。

    使用平滑膝角速度 + 踝/球坐标，经 locate_impact_frame（含抛物线插值）得到
    确定性 t_impact；球心默认锚定为该帧 right_foot_index。
    """
    del ball_center  # 外部静态球心不得覆盖锁帧交叉验证
    n = len(frames) if frames else 0
    if n == 0:
        return 0, np.zeros(3, dtype=np.float64), {"t0_method": "empty_frames"}

    knee_angles: List[float] = []
    timestamps: List[float] = []
    ankle_coords: List[Tuple] = []
    ball_coords: List[Tuple] = []

    for i, rec in enumerate(frames):
        t = float(rec.get("timestamp_sec", i / 30.0))
        timestamps.append(t)
        try:
            ang = float(
                calculate_angle(rec["right_hip"], rec["right_knee"], rec["right_ankle"])
            )
        except Exception:  # noqa: BLE001
            ang = float(knee_angles[-1]) if knee_angles else 150.0
        knee_angles.append(ang)

        try:
            ankle = rec.get("right_ankle") or [0.0, 0.0, 0.0]
            ankle_coords.append(tuple(float(x) for x in ankle[:3]))
        except Exception:  # noqa: BLE001
            ankle_coords.append((0.0, 0.0, 0.0))

        # 球心：优先显式 ball_center 字段，否则用右足尖作为逼近锚点
        try:
            if rec.get("ball_center") is not None:
                bc = rec["ball_center"]
                ball_coords.append(tuple(float(x) for x in list(bc)[:3]))
            else:
                toe = rec.get("right_foot_index") or ankle_coords[-1]
                ball_coords.append(tuple(float(x) for x in list(toe)[:3]))
        except Exception:  # noqa: BLE001
            ball_coords.append(ankle_coords[-1])

    # 角速度（deg/s）：用时间戳差分，再 Savitzky-Golay 平滑
    omega_raw: List[float] = [0.0]
    for i in range(1, n):
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 1e-9:
            omega_raw.append(0.0)
        else:
            omega_raw.append((knee_angles[i] - knee_angles[i - 1]) / dt)
    omega_smooth = KinematicSignalProcessor.smooth_joint_trajectories(omega_raw)

    t_impact = locate_impact_frame(omega_smooth, ankle_coords, ball_coords)
    t_impact = int(max(0, min(n - 1, t_impact)))

    toe = frames[t_impact].get("right_foot_index") or frames[t_impact].get("right_ankle")
    try:
        ball = np.asarray(list(toe)[:3], dtype=np.float64)
    except Exception:  # noqa: BLE001
        ball = np.zeros(3, dtype=np.float64)

    meta: dict[str, Any] = {
        "t0_method": "locate_impact_frame_parabolic_v25",
        "ball_estimate": "t0_right_foot_index_anchor",
        "shank_omega_peak_index": int(np.argmax(np.abs(np.asarray(omega_smooth, dtype=np.float64)))),
        "shank_omega_peak_value": float(np.max(np.abs(np.asarray(omega_smooth, dtype=np.float64))))
        if omega_smooth
        else 0.0,
        "min_foot_ball_dist_index": t_impact,
        "t0_search_half_window": IMPACT_SEARCH_HALF_WINDOW,
    }
    return t_impact, ball, meta


def apply_face_blur(frame, single_person_landmarks):
    """【兼容别名】委托给 image_processing.apply_facial_anonymization。

    新代码请直接调用 ``apply_facial_anonymization``。本函数保留是为了不打断
    api_server / 报告链路里既有的 ``pt.apply_face_blur(...)`` 调用。

    重要：必须在 draw_pose_landmarks（画骨骼连线）之前、以及任何原图转存之前调用。
    """
    # 这是符合《未成年人保护法》与科研伦理审查的物理级脱敏，任何人不得在此行代码之前进行原图转存。
    return apply_facial_anonymization(frame, single_person_landmarks)


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
# 【第四部分】桌面 GUI / Worker 已拆分至独立模块（主/从线程分离）
#   - workers/inference_worker.py  → InferenceWorker(QThread) 计算从线程
#   - main_window.py               → MainWindow 仅信号槽刷新 UI
#   - main.py                      → 推荐桌面入口
# 本文件保留视觉算法与运动学工具；此处仅做兼容转发，避免循环导入。
# --------------------------------------------------------------------------


def __getattr__(name: str):
    """惰性导出 InferenceWorker / VideoWorker / MainWindow。"""
    if name in ("InferenceWorker", "VideoWorker"):
        from workers.inference_worker import InferenceWorker, VideoWorker

        return InferenceWorker if name == "InferenceWorker" else VideoWorker
    if name == "MainWindow":
        from main_window import MainWindow

        return MainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main():
    """兼容入口：转发到 main_window.main()（推荐直接 python main.py）。"""
    from main_window import main as _run_desktop_app

    _run_desktop_app()


if __name__ == "__main__":
    main()
