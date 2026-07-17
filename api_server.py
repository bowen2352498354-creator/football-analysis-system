# -*- coding: utf-8 -*-
"""
api_server.py
v1.1 前后端全栈联调阶段：后台服务网关（FastAPI + Uvicorn）

功能说明：
    本脚本是连接「React 前端网页 (AI-Football-Web)」与「Python 视觉/大模型算法
    (pose_tracker.py + llm_agent.py)」之间的唯一桥梁。它彻底废除了前端的假数据
    (mockData.ts) 生成逻辑，让网页真正显示后台实时推理出来的画面与数据。

    本文件【完全复用】pose_tracker.py 里已经写好的核心算法函数（角度计算、
    三级容错判定、骨骼连线绘制、面部高斯模糊打码），不重复实现任何算法逻辑，
    只是把原来"画在 PyQt5 QLabel 上"的输出通道，换成了"通过 WebSocket 推给
    浏览器"的输出通道；也完全复用 llm_agent.py 里封装好的 DeepSeek 调用逻辑。

核心接口一览：
    POST /api/upload_video   ：上传本地 MP4 文件（例如 test_video.mp4），
                                保存到项目根目录 uploads/ 临时目录，返回文件路径。
    WS   /ws/analyze         ：核心实时推理通道。浏览器通过这条 WebSocket 连接：
                                1) 发送 {"action": "start", "source": "webcam"/"file"/"default",
                                   "video_path": "..."} 来启动一次分析会话；
                                2) 持续收到 {"type": "frame", "image": "data:image/jpeg;base64,...",
                                   "angle": 142.3, "status": "Green", "angular_velocity": 186.4,
                                   "stability_index": 92, ...} 这样的实时推理结果（新增的
                                   angular_velocity/stability_index 字段供前端「实时动力链
                                   角速度监控」波形图与稳定指数徽标使用）；
                                3) 发送 {"action": "stop"} 来结束这次分析会话（本地视频播放完毕
                                   也会自动结束）；
                                4) 【新增】偶尔可能收到一条 {"type": "notice", "message": "..."}，
                                   这是非致命的诊断提醒（例如自动检测到摄像头持续输出全黑画面），
                                   不会中断分析会话，只用来提示用户去检查摄像头权限/占用/遮挡。
    POST /api/generate_report：分析结束后，前端带着 session_id 调用这个接口，
                                后台真正调用 llm_agent.generate_session_report()
                                请求 DeepSeek 大模型，生成结构化诊断报告 JSON 返回给前端；
                                同时会用 OpenCV 在整趟练习中自动捕捉到的"击球关键帧"上
                                叠加髋-膝-踝矢量标注，以 impactFrameImage（Base64 JPEG）
                                字段随文字报告一起返回，供前端左栏展示。
    POST /api/save_word_report：前端带着学生档案 + AI 诊断报告 + 关键帧图片 Base64 +
                                模式类型 (realtime/delayed) 调用这个接口，后台真正调用
                                word_reporter.save_feedback_to_word()，在本机硬盘上按
                                "一级测试类型 -> 二级学校-班级/组别 -> 三级学生编号"
                                的规则建好文件夹树，并把规范排版的 Word (.docx) 报告
                                写入其中，返回成功消息与生成文件的绝对物理路径。

【科技伦理与隐私保护红线】（与 pose_tracker.py 完全一致）：
    所有视频帧的姿态推理、骨骼绘制、面部高斯模糊打码全部在服务端内存中实时完成，
    处理完的画面通过 WebSocket 直接推给浏览器展示，不会把原始帧或处理后的帧写入
    磁盘做长期持久化保存；uploads/ 目录仅临时存放用户主动上传的本地视频文件，
    用于本次分析读取帧数据，不属于"实时展示视频"的范畴。

============================================================================
【如何启动这个后端服务】（请在终端里执行，而不是直接用 F5 调试运行）：

    1. 确保依赖已安装（项目根目录下）：
           pip install fastapi "uvicorn[standard]" python-multipart
       （opencv-python / mediapipe / numpy / Pillow / openai / python-dotenv
        应该已经在 requirements.txt 里装过，如果没装：pip install -r requirements.txt）

    2. 确保项目根目录下的 .env 文件已配置好 DEEPSEEK_API_KEY
       （llm_agent.py 启动时会自动加载，缺失会直接抛错退出）。

    3. 在项目根目录（与 pose_tracker.py 同级目录）下执行以下任意一条命令启动服务：

           python api_server.py

       或者（推荐，支持代码改动后自动重载，开发调试更方便）：

           uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

    4. 服务启动后会监听 http://localhost:8000 ，
       浏览器可以直接访问 http://localhost:8000/docs 查看自动生成的接口文档。

    【极其重要，请务必注意】如果你是用 `python api_server.py` 这种方式启动的，
    这个进程【不会自动感知代码改动】！每次修改完 api_server.py / pose_tracker.py /
    llm_agent.py 之后，必须先在终端按 Ctrl+C 停掉旧进程，再重新执行一次
    `python api_server.py`，新代码才会真正生效——前端 Vite 开发服务器有 HMR
    热更新，但这个 Python 后端没有（除非你用的是下面这条带 --reload 的命令）。

    5. 保持这个终端窗口一直运行，再另开一个终端窗口，进入 AI-Football-Web 目录，
       执行 npm run dev 启动前端 Vite 开发服务器（默认 http://localhost:5173），
       前端会自动通过 http://localhost:8000 与 ws://localhost:8000 访问本服务。
============================================================================
"""

import asyncio
import base64
import collections
import io
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from typing import Optional

# --------------------------------------------------------------------------
# 【Windows 编码兼容性修复：第一防线】强制把标准输出/标准错误流重新包装成
# UTF-8 编码。
#
# 根因：Windows 默认控制台代码页是 GBK（cp936），当 Python 进程的 stdout 没有
# 被显式指定编码时，print() 内部会尝试用 GBK 去编码字符串。一旦调试日志里出现
# Emoji（✅ ❌ 💾 🖨️ 等）或任何 GBK 字符集之外的字符，就会直接抛出
# UnicodeEncodeError（'gbk' codec can't encode character ...），而这里的
# print() 调用大多发生在后台工作线程（AnalysisSession._run）里，未被外层
# try/except 兜住的话会直接把整条后台视频处理/归档线程干掉，前端表现为
# "分析莫名其妙卡死/黑屏"。
#
# 这里在所有其他逻辑执行之前，把 sys.stdout / sys.stderr 重新包装成一个
# encoding='utf-8'、errors='replace' 的 TextIOWrapper：
#   - 强制使用 UTF-8，不再依赖操作系统的默认代码页，从根源上避免 GBK 编码不了
#     Emoji/生僻字的问题；
#   - errors='replace' 作为最后一道保险——即使真的遇到 UTF-8 也编码不了的
#     极端字符，也只会把它替换成 "?"，绝不会再抛异常炸掉后台线程。
# --------------------------------------------------------------------------
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except (AttributeError, ValueError):
        pass
if sys.stderr.encoding is None or sys.stderr.encoding.lower() != "utf-8":
    try:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except (AttributeError, ValueError):
        pass


def safe_print(*args, **kwargs) -> None:
    """【Windows 编码兼容性修复：第二防线】封装打印函数：即使上面的 stdout
    强制 UTF-8 重组因为某些极端环境（例如 stdout 被第三方库/IDE 再次替换成
    没有 .buffer 属性的对象）没能生效，这里也兜底捕获 UnicodeEncodeError，
    自动把无法编码的字符替换掉再重试打印一次，确保任何一条日志语句都
    绝对不会让后台视频处理/归档线程崩溃退出。
    """
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_args = [
            arg.encode(encoding, errors="replace").decode(encoding, errors="replace")
            if isinstance(arg, str)
            else arg
            for arg in args
        ]
        try:
            print(*safe_args, **kwargs)
        except Exception:
            pass

import cv2
import mediapipe as mp
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 【核心复用】直接把 pose_tracker.py 当作一个模块导入，复用里面已经写好的
# 骨骼绘制 / 角度计算 / 三级容错判定 / 面部打码函数，绝不重复实现算法逻辑。
import pose_tracker as pt

# 【核心复用】直接复用 llm_agent.py 里封装好的 DeepSeek 调用逻辑。
import llm_agent

# 【核心复用】直接复用 word_reporter.py 里封装好的本地归档 + Word 报告生成逻辑。
import word_reporter

# 【v4.0 核心复用】直接复用 academic_exporter.py 里封装好的「论文专供：学术统计
# 矩阵一键自动导出」清洗 + 落盘逻辑，完全不在本文件重复实现任何转换算法。
import academic_exporter

# 【重要防呆】当 Python 进程的标准输出没有连接到一个真正的交互式终端时
# （例如被某些 IDE/工具通过管道重定向捕获），CPython 默认会切换成"整块缓冲"
# 而不是"逐行缓冲"——这意味着我们用 print() 打印的调试日志有可能长时间停留
# 在缓冲区里、迟迟不显示在终端窗口上，制造出"代码明明在跑，终端却什么都没有"
# 的假象，非常容易误导排查方向。这里强制切回逐行缓冲，确保每一条 print()
# 都能第一时间刷新显示出来（配合下面新增诊断日志里的 flush=True 双重保险）。
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

# --------------------------------------------------------------------------
# 第〇步：基础路径与全局状态
# --------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(SCRIPT_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 【v2.0 新增：跨课时双重持久化】延时反馈系统 (Web) 归档池的落盘文件路径。
# 【重要说明，避免和旧版桌面工具混淆】项目根目录下原本已经存在一个
# B_group_data_log.json，那个文件是 pose_tracker.py (PyQt5 桌面版) 与
# report_generator.py 配套使用的"扁平化单帧记录"格式（每条记录只有
# timestamp/knee_angle/status 三个字段），如果这里直接复用同名文件、
# 又写入完全不同的"按学生分组、每人 2~3 次尝试"的结构化格式，会直接
# 冲毁桌面版工具的历史数据、导致 report_generator.py 解析失败。
# 因此 Web 端延时反馈系统使用一个独立的新文件专门承载"学生归档池"数据，
# 与桌面版工具的数据文件互不干扰、可以并存。
WEB_SESSION_LOG_PATH = os.path.join(SCRIPT_DIR, "B_group_web_sessions_log.json")

# 【v2.0 新增：全局训练数据库】每当一份 Word 报告成功写盘归档（无论是实时反馈 A
# 组的单次分析，还是延时反馈 B 组的单趟/批量尝试），都会自动往这个文件追加一条
# 完整的结构化记录，供教练端数据看板 (CoachDashboard.tsx) 通过
# GET /api/get_all_records 一键拉取全量历史归档数据进行可视化复盘。
GLOBAL_DB_PATH = os.path.join(SCRIPT_DIR, "global_training_db.json")
_global_db_lock = threading.Lock()

# 传输给前端时，把画面等比例缩放到这个最大宽度以内，减少 WebSocket 传输的
# Base64 数据量，避免大分辨率视频/摄像头把浏览器和网络带宽拖垮。
MAX_TRANSMIT_WIDTH = 800

# JPEG 编码质量（0-100），在清晰度与传输体积之间取一个比较均衡的数值。
JPEG_QUALITY = 75

# 【新增：实时动力链角速度监控】计算"动平衡稳定指数"时使用的滑动窗口长度
# （按帧数计，约等于最近 1 秒左右的角速度样本），窗口内角速度越离散（标准差越大），
# 说明动作抖动越明显，稳定指数就越低。
STABILITY_WINDOW_SIZE = 30

# 击球关键帧标注图输出的 JPEG 质量：报告场景只需要生成一张静态图，
# 可以用更高的质量换取更清晰的矢量标注展示效果。
IMPACT_FRAME_JPEG_QUALITY = 90

# 【新增：黑屏问题自动诊断】画面平均亮度（0-255）低于这个阈值，判定为"疑似全黑帧"。
# 这是 Windows 上最常见的一类"黑屏但无任何报错"的真实原因：cv2.VideoCapture 明明
# isOpened() 为 True、cap.read() 也返回 ret=True，但因为系统隐私设置未授权摄像头
# 权限，或摄像头被其他程序占用/物理遮挡，读出来的每一帧画面数据本身就是纯黑色的。
BLACK_FRAME_MEAN_BRIGHTNESS_THRESHOLD = 6.0

# 连续多少帧都被判定为"疑似全黑"，才正式弹出一次诊断提示（避免开场一两帧还没对好焦
# 就误报，同时也不会拖太久才提示，让用户能尽快定位问题）。
BLACK_FRAME_CONSECUTIVE_LIMIT = 15

# 每推送这么多帧，就在服务端终端打印一次进度日志，方便直接从终端确认
# "画面到底有没有在真实产生、真实推送"，而不是盲猜。
FRAME_PROGRESS_LOG_INTERVAL = 60

# 全局会话表：session_id -> AnalysisSession。
# 【设计说明】本项目是面向单个课堂/单台设备的教学工具，同一时刻通常只有
# 一个学生在做分析，这里用一个简单的全局字典即可满足需求；如果未来要支持
# 多教室并发使用，可以在这里升级为按 classroom_id 分片的会话管理。
SESSIONS: dict[str, "AnalysisSession"] = {}


# --------------------------------------------------------------------------
# 【核心新增】击球瞬间关键帧生物力学诊断标注：OpenCV 矢量绘图引擎
#
#         这里只在"分析结束、前端调用 /api/generate_report"这一次性场景下，
#         对捕捉到的那一张静态关键帧做一次绘制，绝不在实时推理的逐帧循环里
#         调用，避免拖慢实时画面的推送节奏。
# --------------------------------------------------------------------------


def draw_biomechanics_annotation(frame, metrics: dict):
    """在"击球关键帧"静态截图上叠加专业运动生物力学矢量标注，用于图文并茂的
    诊断报告左栏展示。

    标注内容（严格对应生物力学诊断报告的可视化需求）：
        ① 髋->膝、膝->踝 两段带方向箭头的矢量连线，清晰呈现"髋-膝-踝"这条
           动力链的发力传导路径；颜色随三级容错状态变化——Green/Yellow（达标/
           接近）用醒目的亮绿色，Red（明显偏离）用醒目的亮红色，一眼可辨；
        ② 在膝关节顶点画一段角度弧线，并用清晰的白底黑边数字标注出实际测量
           得到的膝关节屈曲夹角，让"角度数字"与"画面上的真实夹角"直接对应；
        ③ 从髋部中心点向下画一条身体重心参考垂直虚线，辅助教练/学生判断
           触球瞬间身体是否存在明显的前倾或后仰，是评估动作稳定性的重要
           辅助参考线。

    参数：
        frame：击球关键帧的 BGR 画面（已完成面部高斯模糊打码，函数内部会先
               .copy() 一份，不会修改传入的原始帧）。
        metrics：dict，包含 hip_px / knee_px / ankle_px / mid_hip_px / angle /
                 status 六个字段（由 AnalysisSession._capture_impact_candidate
                 采集时一并记录下来）。

    返回：
        画好全部矢量标注之后的新 BGR 画面（numpy 数组）。
    """
    annotated = frame.copy()

    hip_px = metrics["hip_px"]
    knee_px = metrics["knee_px"]
    ankle_px = metrics["ankle_px"]
    mid_hip_px = metrics["mid_hip_px"]
    angle = metrics["angle"]
    status = metrics["status"]

    # 颜色策略：达标(Green)/接近(Yellow)用醒目亮绿色，明显偏离(Red)用醒目亮红色，
    # 让教练/学生一眼就能从颜色上判断出这次触球瞬间的发力质量。
    vector_color = (60, 235, 60) if status in ("Green", "Yellow") else (40, 40, 245)
    line_thickness = 4

    # ① 髋->膝、膝->踝 方向箭头矢量线：tipLength 取一个比较优雅、不过分夸张的比例，
    # 线宽 4px 在 800px 宽度的传输画面上清晰又不会显得笨重。
    cv2.arrowedLine(annotated, hip_px, knee_px, vector_color, line_thickness, cv2.LINE_AA, tipLength=0.14)
    cv2.arrowedLine(annotated, knee_px, ankle_px, vector_color, line_thickness, cv2.LINE_AA, tipLength=0.14)

    # 髋/膝/踝三个诊断关键点画成"白底描边 + 彩色描边"的双层圆点，在任何背景色下都清晰可见
    for point in (hip_px, knee_px, ankle_px):
        cv2.circle(annotated, point, 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(annotated, point, 7, vector_color, 2, cv2.LINE_AA)

    # ② 在膝关节顶点画一段角度弧线：以膝关节为圆心，利用 hip/ankle 两个方向向量
    # 相对膝关节的极角（atan2），让 cv2.ellipse 自动画出这两条矢量夹出的那一段弧线
    vector_knee_to_hip = (hip_px[0] - knee_px[0], hip_px[1] - knee_px[1])
    vector_knee_to_ankle = (ankle_px[0] - knee_px[0], ankle_px[1] - knee_px[1])
    angle_towards_hip_deg = math.degrees(math.atan2(vector_knee_to_hip[1], vector_knee_to_hip[0]))
    angle_towards_ankle_deg = math.degrees(math.atan2(vector_knee_to_ankle[1], vector_knee_to_ankle[0]))

    arc_radius = 46
    cv2.ellipse(
        annotated,
        knee_px,
        (arc_radius, arc_radius),
        0,
        angle_towards_hip_deg,
        angle_towards_ankle_deg,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 角度数字标注：先画一层加粗的深色描边，再叠一层白色正文，保证在任何画面背景
    # （亮/暗、复杂纹理）下都清晰可读，字体选用 cv2 内置的 FONT_HERSHEY_SIMPLEX，
    # 字号 0.85 在 800px 宽度的画面上清晰不拥挤。
    angle_label = f"{angle:.1f} deg"
    label_pos = (knee_px[0] + arc_radius + 10, knee_px[1] + 6)
    cv2.putText(annotated, angle_label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.85, (15, 15, 15), 5, cv2.LINE_AA)
    cv2.putText(annotated, angle_label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # ③ 身体重心参考垂直虚线：从髋部中心点开始，向下用"短线段+间隔"的方式手绘一条
    # 虚线直到画面底部，颜色选用醒目的琥珀色，与绿/红矢量线形成鲜明区分。
    frame_height = annotated.shape[0]
    dash_length, gap_length = 10, 8
    y_cursor = mid_hip_px[1]
    while y_cursor < frame_height:
        y_end = min(frame_height, y_cursor + dash_length)
        cv2.line(annotated, (mid_hip_px[0], y_cursor), (mid_hip_px[0], y_end), (0, 200, 255), 2, cv2.LINE_AA)
        y_cursor += dash_length + gap_length

    return annotated


# --------------------------------------------------------------------------
# 第一步：AnalysisSession —— 后台分析会话（每次"开始分析"对应一个实例）
#
#         这里的职责跟 pose_tracker.py 里的 VideoWorker(QThread) 几乎一模一样，
#         区别只是：VideoWorker 用 pyqtSignal 把处理好的帧发给 PyQt5 主线程，
#         而这里用一个线程安全的 queue.Queue 把处理好的帧交给 FastAPI 的
#         异步协程，再由协程通过 WebSocket 推送给浏览器。
# --------------------------------------------------------------------------


class AnalysisSession:
    """代表一次"开始分析 -> 持续推理 -> 结束分析"的完整生命周期。"""

    def __init__(self, session_id: str, source: str, video_path: Optional[str], camera_index: int = 0):
        self.session_id = session_id
        self.source = source  # "webcam" | "file"
        self.video_path = video_path
        self.camera_index = camera_index

        # 后台线程与主协程之间通过这两个线程安全的对象通信：
        #   frame_queue：后台线程处理好一帧就 put 一份结果字典进去；
        #   stop_event：协程收到前端"结束分析"指令时 set()，后台线程每一轮
        #               循环都会检查，检测到就自然退出（跟 VideoWorker 的
        #               request_stop() 设计思路完全一致）。
        self.frame_queue: "queue.Queue[dict]" = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        # 【B组/科研数据落盘的联调等价物】本次分析全过程的有效诊断记录，
        # 结束分析后会被 /api/generate_report 读取，喂给 DeepSeek 生成真实报告。
        self.records: list[dict] = []
        self._records_lock = threading.Lock()

        # 【新增：实时动力链角速度监控】上一帧角度与时间戳，用于逐帧计算角速度（deg/s）
        self._prev_angle: Optional[float] = None
        self._prev_frame_time: Optional[float] = None
        # 最近 STABILITY_WINDOW_SIZE 帧的角速度滑动窗口，用于计算"动平衡稳定指数"
        self._velocity_window: "collections.deque[float]" = collections.deque(maxlen=STABILITY_WINDOW_SIZE)

        # 【新增：击球关键帧自动捕捉】记录整趟练习中"右膝角速度绝对值最大"的那一帧
        # （即冲击最剧烈、最接近真实触球瞬间的画面），供生成报告时叠加矢量标注。
        self.impact_frame = None  # numpy 数组：命中的击球关键帧（已完成面部打码，未叠加骨骼线）
        self.impact_metrics: Optional[dict] = None  # 该帧对应的 hip/knee/ankle 像素坐标与角度/状态
        self._best_impact_score: float = -1.0

        # 【新增：黑屏问题自动诊断】累计推送帧数（用于终端进度日志）+ 连续疑似全黑帧计数
        # （用于自动检测"摄像头权限被禁用/被占用/被遮挡"这一类无报错但画面全黑的情况）。
        self._pushed_frame_count = 0
        self._consecutive_dark_frames = 0
        self._dark_frame_warning_sent = False

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def request_stop(self):
        self.stop_event.set()

    def get_records_snapshot(self) -> list[dict]:
        with self._records_lock:
            return list(self.records)

    def _compute_angular_velocity(self, angle: float) -> float:
        """根据"当前帧角度 - 上一帧角度"除以帧间时间差，计算右膝角速度（deg/s）。
        第一帧因为没有"上一帧"可比较，约定角速度为 0。
        """
        now = time.time()
        angular_velocity = 0.0
        if self._prev_angle is not None and self._prev_frame_time is not None:
            dt = now - self._prev_frame_time
            if dt > 0:
                angular_velocity = (angle - self._prev_angle) / dt
        self._prev_angle = angle
        self._prev_frame_time = now
        self._velocity_window.append(angular_velocity)
        return angular_velocity

    def _compute_stability_index(self) -> int:
        """根据最近滑动窗口内角速度的离散程度（标准差）换算「动平衡稳定指数」（0-100）。

        设计思路：角速度标准差越小，说明摆动腿发力节奏越连贯、动作越"不抖"，
        对应稳定指数越高；一旦出现忽快忽慢的剧烈波动，标准差变大，指数随之下降。
        """
        if len(self._velocity_window) < 2:
            return 100
        values = list(self._velocity_window)
        mean_value = sum(values) / len(values)
        variance = sum((v - mean_value) ** 2 for v in values) / len(values)
        std_dev = variance ** 0.5
        # 经验系数：标准差每增加 4 deg/s 扣 1 分，兜底裁剪到 [0, 100] 区间
        index = 100 - std_dev / 4.0
        return int(max(0, min(100, round(index))))

    def _capture_impact_candidate(self, frame, landmarks, hip_px, knee_px, ankle_px, angle, status):
        """【击球关键帧自动捕捉】把当前帧记录为"击球关键帧"候选：
        整趟练习结束后，self.impact_frame 会一直保留角速度绝对值最大（即冲击最
        剧烈、最贴近真实触球瞬间）的那一帧，供 /api/generate_report 生成矢量标注图。

        重要：必须在 pt.apply_face_blur() 之后、pt.draw_pose_landmarks() /
        pt.draw_right_knee_overlay() 之前调用——既要保证脸部已经打码（隐私红线），
        又要保证存下来的是一张"干净"的画面，方便后续单独叠加矢量标注，不会与
        实时预览用的白色骨骼线/粗染色线互相干扰。
        """
        height, width = frame.shape[:2]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        mid_hip_px = (
            int((left_hip.x + right_hip.x) / 2 * width),
            int((left_hip.y + right_hip.y) / 2 * height),
        )

        self.impact_frame = frame.copy()
        self.impact_metrics = {
            "hip_px": hip_px,
            "knee_px": knee_px,
            "ankle_px": ankle_px,
            "mid_hip_px": mid_hip_px,
            "angle": angle,
            "status": status,
        }

    def _push_frame_payload(self, payload: dict):
        """把一份处理好的帧数据放进队列；如果消费端（WebSocket 协程）来不及
        取走导致队列满了，就直接丢弃最旧的一帧，保证画面始终贴近"实时"，
        而不是攒积压帧导致画面越播越"卡"、越滞后。
        """
        try:
            self.frame_queue.put_nowait(payload)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(payload)
            except queue.Full:
                pass

    def _run(self):
        """后台工作线程主体：逐帧读取视频源 -> 姿态检测 -> 力学诊断 ->
        骨骼染色 + 面部打码渲染 -> 编码成 Base64 JPEG -> 推入队列。

        这里的每一步算法逻辑，都是直接调用 pose_tracker.py (pt 模块) 里
        已经写好、且已经在桌面版软件里验证过的函数，完全没有重新实现。
        """
        cap = None
        landmarker = None
        is_video_file_mode = self.source == "file"

        try:
            pt.ensure_model_downloaded()

            if is_video_file_mode:
                if not self.video_path or not os.path.exists(self.video_path):
                    safe_print(f"【api_server】错误：未找到视频文件：{self.video_path}")
                    self._push_frame_payload({
                        "type": "error",
                        "message": f"未找到视频文件：{self.video_path}",
                    })
                    return
                cap = cv2.VideoCapture(self.video_path)
            else:
                cap = cv2.VideoCapture(self.camera_index)

            if not cap.isOpened():
                # 【新增】终端同步打印一条明确的错误提示：这是"点击开始分析后一直黑屏，
                # 但终端看起来又没有任何异常"最常见的真实原因之一——摄像头被其他程序
                # 占用、摄像头编号不对，或者根本没有摄像头设备。
                safe_print("【api_server】错误：无法打开视频源（本地视频文件损坏，或摄像头被其他程序占用/无摄像头设备）。")
                self._push_frame_payload({
                    "type": "error",
                    "message": "无法打开视频源（本地视频文件损坏，或摄像头被其他程序占用/无摄像头设备）。",
                })
                return

            if is_video_file_mode:
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                if not video_fps or video_fps <= 0:
                    video_fps = 30.0
                frame_delay_seconds = 1.0 / video_fps
            else:
                frame_delay_seconds = 0.0

            # 与 pose_tracker.py A组逻辑完全一致：VIDEO 运行模式的姿态检测器
            pose_options = pt.mp_vision.PoseLandmarkerOptions(
                base_options=pt.mp_tasks_python.BaseOptions(model_asset_path=pt.MODEL_PATH),
                running_mode=pt.mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            landmarker = pt.mp_vision.PoseLandmarker.create_from_options(pose_options)

            frame_timestamp_ms = 0

            while not self.stop_event.is_set() and cap.isOpened():
                loop_start_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    if self._pushed_frame_count == 0:
                        # 【核心修复：黑屏"零帧"问题】cv2.VideoCapture(...) 的 isOpened()
                        # 检查在 Windows/DirectShow(MSMF) 环境下经常会"虚报成功"——
                        # 明明返回 True，但紧接着第一次 cap.read() 就直接失败（ret=False）。
                        # 之前这里只有一个静默的 break，既不打印任何日志，也不往前端
                        # 推送任何 error/notice 消息，效果等同于"正常播放完毕自动结束"，
                        # 导致前端永远等不到第一帧画面、也看不到任何报错，
                        # 表现为一开始就直接黑屏（也就是"视频流在第 0 秒就流产"）。
                        failure_reason = (
                            f"未能从{'本地视频文件' if is_video_file_mode else '摄像头'}"
                            f"读取到任何一帧画面数据：cv2.VideoCapture 显示已成功打开，"
                            f"但第一次 cap.read() 就直接失败。"
                        )
                        if is_video_file_mode:
                            failure_reason += (
                                f" 视频文件路径：{self.video_path}。请确认该文件本身没有损坏，"
                                f"且编码格式受本机 OpenCV 支持（推荐使用 H.264 编码的 .mp4）。"
                            )
                        else:
                            failure_reason += (
                                " 常见原因：摄像头正被其他程序独占使用（如视频会议软件/OBS）、"
                                "摄像头驱动异常，或该摄像头编号在系统里并不是真正可用的设备。"
                                "请先关闭其他可能占用摄像头的程序，或重启电脑后重试。"
                            )
                        safe_print(f"【api_server】错误：{failure_reason}", flush=True)
                        self._push_frame_payload({"type": "error", "message": failure_reason})
                    else:
                        safe_print(
                            f"【api_server】提示：视频源已读取完毕或已断开"
                            f"（本次分析累计成功推送 {self._pushed_frame_count} 帧），分析自然结束。",
                            flush=True,
                        )
                    break

                if not is_video_file_mode:
                    frame = cv2.flip(frame, 1)

                # 【新增：黑屏问题自动诊断】统计推送帧数 + 检测"疑似全黑帧"。
                # 这一步只做统计判断，绝不修改 frame 本身，不影响后续任何画面处理。
                self._pushed_frame_count += 1
                mean_brightness = float(frame.mean())

                if self._pushed_frame_count == 1:
                    safe_print(
                        f"【api_server】[OK] 已成功读取到第 1 帧原始画面"
                        f"（平均亮度 {mean_brightness:.1f}/255，数值越接近 0 代表画面越黑），"
                        f"视频推理管线已正常启动。"
                    )
                elif self._pushed_frame_count % FRAME_PROGRESS_LOG_INTERVAL == 0:
                    safe_print(
                        f"【api_server】进度：已累计推送 {self._pushed_frame_count} 帧画面"
                        f"（本帧平均亮度 {mean_brightness:.1f}/255）。"
                    )

                if mean_brightness < BLACK_FRAME_MEAN_BRIGHTNESS_THRESHOLD:
                    self._consecutive_dark_frames += 1
                else:
                    self._consecutive_dark_frames = 0

                if (
                    not self._dark_frame_warning_sent
                    and self._consecutive_dark_frames >= BLACK_FRAME_CONSECUTIVE_LIMIT
                ):
                    self._dark_frame_warning_sent = True
                    dark_frame_hint = (
                        "检测到摄像头已连续读取到多帧近乎全黑的画面（程序本身运行正常，没有发生异常）。"
                        "在 Windows 系统上，这通常不是代码问题，而是以下几种情况之一："
                        "① Windows 设置 -> 隐私和安全性 -> 相机，未授权「桌面应用」访问摄像头；"
                        "② 摄像头正被其他程序占用（例如视频会议软件、OBS，请先关闭它们再重试）；"
                        "③ 摄像头物理镜头被遮挡，或笔记本电脑的摄像头隐私挡片处于关闭状态。"
                        "请检查以上几点后重新点击「开始分析」。"
                    )
                    safe_print(f"【api_server】警告：{dark_frame_hint}")
                    # 用独立的 "notice" 消息类型推送提示：这是"非致命的诊断提醒"，
                    # 不应该像 "error" 一样中断分析会话或关闭连接，只是让前端弹出
                    # 一条醒目的黄色提示，方便老师/学生第一时间知道该去检查什么。
                    self._push_frame_payload({"type": "notice", "message": dark_frame_hint})

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                frame_timestamp_ms += 33
                results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                angle_value = None
                status_value = None
                angular_velocity_value = None
                stability_index_value = None

                # 【容错防呆】把"姿态诊断 + 角速度/稳定指数计算 + 骨骼渲染"这一整段
                # 逐帧处理逻辑包在独立的 try/except 里：万一某一帧因为异常姿态数据
                # （例如极端角度、瞬时坐标缺失等）导致计算异常，也只会跳过这一帧的
                # 诊断信息渲染，绝不能让整条视频推理循环因此直接崩溃退出——否则
                # 前端会表现为"点击开始分析后画面很快就变成一片黑屏，且没有任何
                # 明确报错"，因为循环提前 return 之后就再也没有新的画面帧推送过来了。
                try:
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks[0]
                        angle, status, color, hip_px, knee_px, ankle_px = pt.compute_right_knee_diagnosis(
                            frame, landmarks
                        )

                        # 【新增】逐帧计算右膝角速度（deg/s）与动平衡稳定指数，
                        # 供前端「实时动力链角速度监控」波形图与稳定指数徽标使用。
                        # 第一帧因为还没有"上一帧角度"可供比较，_compute_angular_velocity
                        # 内部已经做好防呆（约定角速度为 0），这里不会出现除以零的情况。
                        angular_velocity = self._compute_angular_velocity(angle)
                        stability_index_value = self._compute_stability_index()

                        # 顺序严格保持与 pose_tracker.py 一致：先打码，再画骨骼线，
                        # 最后叠加红/黄/绿容错染色骨骼线，避免打码把骨骼线又"擦掉"。
                        pt.apply_face_blur(frame, landmarks)

                        # 【新增：击球关键帧自动捕捉】在打码之后、骨骼线绘制之前，
                        # 用"角速度绝对值是否为整趟练习目前最大值"来判定是否更新击球关键帧候选，
                        # 角速度越大代表这一帧越接近真实的"发力冲击瞬间"。
                        impact_score = abs(angular_velocity)
                        if impact_score > self._best_impact_score:
                            self._best_impact_score = impact_score
                            self._capture_impact_candidate(
                                frame, landmarks, hip_px, knee_px, ankle_px, angle, status
                            )

                        pt.draw_pose_landmarks(frame, results.pose_landmarks)
                        pt.draw_right_knee_overlay(frame, hip_px, knee_px, ankle_px, color, angle, status)

                        angle_value = round(float(angle), 1)
                        status_value = status
                        angular_velocity_value = round(float(angular_velocity), 1)

                        record = {
                            "timestamp": time.time(),
                            "knee_angle": angle_value,
                            "status": status_value,
                            "angular_velocity": angular_velocity_value,
                        }
                        with self._records_lock:
                            self.records.append(record)
                except Exception as diagnosis_exc:  # noqa: BLE001 - 单帧诊断异常绝不能打断整条视频流
                    safe_print(f"【api_server】单帧姿态诊断/角速度计算发生异常（已跳过该帧诊断信息，画面仍会继续推送）：{diagnosis_exc}")
                    angle_value = None
                    status_value = None
                    angular_velocity_value = None
                    stability_index_value = None

                # 传输前按最大宽度等比例缩小，减轻 Base64 + WebSocket 的带宽压力
                height, width = frame.shape[:2]
                if width > MAX_TRANSMIT_WIDTH:
                    scale = MAX_TRANSMIT_WIDTH / width
                    frame = cv2.resize(frame, (MAX_TRANSMIT_WIDTH, int(height * scale)))

                ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok:
                    # 极少数情况下编码失败：跳过这一帧但绝不中断循环，下一帧会正常继续推送
                    safe_print("【api_server】警告：本帧 JPEG 编码失败，已跳过，不影响后续帧的实时推送。")
                    continue
                base64_jpeg = base64.b64encode(buffer).decode("ascii")

                # 【规范格式防呆】显式拼接标准的 data URI 前缀，确保前端 <img src={...}>
                # 拿到的永远是浏览器能够直接识别渲染的合法 "data:image/jpeg;base64,xxxx" 格式。
                image_data_uri = f"data:image/jpeg;base64,{base64_jpeg}"

                self._push_frame_payload({
                    "type": "frame",
                    "image": image_data_uri,
                    "angle": angle_value,
                    "status": status_value,
                    "angular_velocity": angular_velocity_value,
                    "stability_index": stability_index_value,
                    "timestamp": time.time(),
                })

                if is_video_file_mode and frame_delay_seconds > 0:
                    elapsed = time.time() - loop_start_time
                    remaining = frame_delay_seconds - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

        except Exception as exc:  # noqa: BLE001 - 后台线程内的任何异常都不能让服务崩溃
            # 【新增】同步把异常信息打印到服务端终端：之前这里只把错误通过 WebSocket
            # 发给前端，终端里完全看不到任何报错痕迹，导致排查"黑屏"问题时无从下手。
            safe_print(f"【api_server】后台推理线程发生异常，本次分析会话将提前结束：{exc}")
            self._push_frame_payload({"type": "error", "message": f"后台推理线程发生异常：{exc}"})

        finally:
            if cap is not None:
                cap.release()
            if landmarker is not None:
                landmarker.close()
            self._push_frame_payload({
                "type": "stopped",
                "session_id": self.session_id,
                "total_records": len(self.records),
            })


# --------------------------------------------------------------------------
# 第二步：FastAPI 应用初始化 + CORS 跨域配置
# --------------------------------------------------------------------------

app = FastAPI(title="小学足球AI可视化反馈系统 - 后台服务网关", version="1.1.0")

# 开启 CORS：允许本地 Vite 开发服务器（5173/5183 等常见端口）跨域访问。
# 开发阶段直接放开所有来源，避免因为 Vite 随机切换端口而反复改配置；
# 生产部署时应该把 allow_origins 收紧为真实的前端域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"service": "AI-Football-Feedback API Gateway", "status": "running"}


# --------------------------------------------------------------------------
# 第三步：核心接口一（上）—— 本地视频文件上传
# --------------------------------------------------------------------------


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """接收前端上传的本地 MP4 文件（例如 test_video.mp4），保存到项目根目录
    uploads/ 临时目录下，返回后端可以直接用 cv2.VideoCapture 打开的绝对路径。

    前端拿到 video_path 后，在下一步通过 WebSocket 发送
    {"action": "start", "source": "file", "video_path": video_path} 即可启动分析。
    """
    file_extension = os.path.splitext(file.filename or "")[1] or ".mp4"
    saved_filename = f"{uuid.uuid4().hex}{file_extension}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)

    with open(saved_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return {"video_path": saved_path, "original_filename": file.filename}


@app.get("/api/default_test_video")
def get_default_test_video():
    """快捷方式：直接返回 pose_tracker.py 里约定的默认本地测试视频
    （项目根目录下的 test_video.mp4）路径，方便前端一键联调，不用每次都手动上传。
    """
    exists = os.path.exists(pt.DEFAULT_VIDEO_FILE_PATH)
    return {"video_path": pt.DEFAULT_VIDEO_FILE_PATH, "exists": exists}


# --------------------------------------------------------------------------
# 第四步：核心接口一（下）—— 实时推理 WebSocket 通道
# --------------------------------------------------------------------------


@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """浏览器通过这条 WebSocket 连接，驱动一次完整的"开始分析 -> 实时收帧 ->
    结束分析"流程。协议非常简单，全部使用 JSON 文本消息：

    浏览器 -> 服务端：
        {"action": "start", "source": "webcam" | "file",
         "video_path": "...", "camera_index": 0}
        {"action": "stop"}

    服务端 -> 浏览器：
        {"type": "started", "session_id": "..."}
        {"type": "frame", "image": "data:image/jpeg;base64,...",
         "angle": 142.3, "status": "Green", "timestamp": 1234567.89}
        {"type": "stopped", "session_id": "...", "total_records": 87}
        {"type": "error", "message": "..."}
    """
    await websocket.accept()

    current_session: Optional[AnalysisSession] = None
    pump_task: Optional[asyncio.Task] = None

    async def pump_frames(session: AnalysisSession):
        """持续从后台线程的队列里取出处理好的帧，转发给浏览器，
        直到收到 "stopped" 这一条收尾消息为止。
        """
        loop = asyncio.get_event_loop()
        while True:
            payload = await loop.run_in_executor(None, session.frame_queue.get)
            try:
                await websocket.send_text(json.dumps(payload, ensure_ascii=False))
            except Exception:
                # 浏览器端已经断开连接，直接停止转发即可，不需要抛出异常
                break
            if payload.get("type") in ("stopped", "error"):
                break

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                data = json.loads(raw_message)
            except json.JSONDecodeError:
                continue

            action = data.get("action")

            if action == "start":
                # 如果上一个会话还没结束，先请求它停止，避免同一个连接里
                # 同时跑两个后台线程互相抢摄像头/视频资源。
                if current_session is not None:
                    current_session.request_stop()
                if pump_task is not None:
                    pump_task.cancel()

                session_id = str(uuid.uuid4())
                source = data.get("source", "webcam")
                video_path = data.get("video_path")
                camera_index = int(data.get("camera_index", 0))

                current_session = AnalysisSession(
                    session_id=session_id, source=source, video_path=video_path, camera_index=camera_index
                )
                SESSIONS[session_id] = current_session
                current_session.start()

                await websocket.send_text(json.dumps({"type": "started", "session_id": session_id}))
                pump_task = asyncio.create_task(pump_frames(current_session))

            elif action == "stop":
                if current_session is not None:
                    current_session.request_stop()

    except WebSocketDisconnect:
        if current_session is not None:
            current_session.request_stop()
    finally:
        if pump_task is not None:
            pump_task.cancel()


# --------------------------------------------------------------------------
# 第五步：核心接口二 —— 调用 DeepSeek 生成真实综合诊断报告
# --------------------------------------------------------------------------


class GenerateReportRequest(BaseModel):
    session_id: str
    student_number: str = ""


@app.post("/api/generate_report")
def generate_report(payload: GenerateReportRequest):
    """分析结束后，前端带着刚才那次分析的 session_id 调用本接口。

    后台会：
        1) 从内存里取出这次分析全过程采集到的真实诊断记录；
        2) 汇总出 Green / Yellow / Red 三级命中次数统计；
        3) 真正调用 llm_agent.generate_session_report()，把统计数据交给
           DeepSeek 大模型，换回结构化的评分 + 痛点 + 处方；
        4) 拼接成前端 FinalDiagnosisReport 类型所需的完整 JSON 返回。
    """
    session = SESSIONS.get(payload.session_id)
    records = session.get_records_snapshot() if session is not None else []

    hit_stats = {"green": 0, "yellow": 0, "red": 0}
    for record in records:
        status = record.get("status")
        if status == "Green":
            hit_stats["green"] += 1
        elif status == "Yellow":
            hit_stats["yellow"] += 1
        elif status == "Red":
            hit_stats["red"] += 1

    total_attempts = hit_stats["green"] + hit_stats["yellow"] + hit_stats["red"]
    sample_angles = [record["knee_angle"] for record in records if record.get("knee_angle") is not None]

    # 【科研指挥中心新增】本次分析全程真实测得的膝关节屈曲角度均值——这是
    # pose_tracker.py 逐帧真实计算出的物理测量值（并非启发式估算），供教练端
    # 「双轴互动运动学成长期刊图」右侧蓝色虚线轴与学术统计矩阵导出直接消费。
    avg_knee_angle = round(sum(sample_angles) / len(sample_angles), 1) if sample_angles else None

    ai_result = llm_agent.generate_session_report(
        hit_stats=hit_stats, student_number=payload.student_number, sample_angles=sample_angles
    )

    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    full_text = (
        f"学号 {payload.student_number or '未填写'} 本次综合练习诊断报告\n\n"
        f"发力稳定性评分：{ai_result['score']} 分（共采集 {total_attempts} 次有效触球数据）。\n"
        f"{ai_result['painPoint']}\n"
        f"{ai_result['prescription']}"
    )

    # 【核心新增】图文并茂诊断报告：在这次分析全程自动捕捉到的"击球关键帧"上，
    # 用 OpenCV 叠加髋-膝-踝矢量箭头 + 角度弧线 + 身体重心垂直虚线，
    # 编码成 Base64 JPEG 字符串，随文字报告一起返回给前端左栏展示。
    impact_frame_image = None
    if session is not None and session.impact_frame is not None and session.impact_metrics is not None:
        annotated_frame = draw_biomechanics_annotation(session.impact_frame, session.impact_metrics)
        ok, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, IMPACT_FRAME_JPEG_QUALITY])
        if ok:
            impact_frame_image = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('ascii')}"

    # 报告已生成完毕，主动清理这份会话（连同内存中持有的击球关键帧画面），
    # 严格遵守"不长期持久化保存任何视频帧"的科技伦理红线，同时避免内存持续累积。
    SESSIONS.pop(payload.session_id, None)

    return {
        "score": ai_result["score"],
        "totalAttempts": total_attempts,
        "painPoint": ai_result["painPoint"],
        "prescription": ai_result["prescription"],
        "fullText": full_text,
        "generatedAt": generated_at,
        "hitStats": hit_stats,
        "impactFrameImage": impact_frame_image,
        "avgKneeAngle": avg_knee_angle,
    }


# --------------------------------------------------------------------------
# 第五步半：核心接口三 —— 跨课时双重持久化「保存归档池」+「读取归档池」
#
#         前端 ZenWorkspace.tsx 每次归档一位同学（换人）或点击"所有人测试
#         完成"时，都会把当前完整的 sessionQueue（每位同学 + 该生 2~3 次
#         尝试的完整实体）POST 到这里，后端直接整体覆盖写入本地 JSON 文件，
#         形成"前端 localStorage + 后端 JSON 落盘"的双保险，防止老师不小心
#         清空浏览器缓存导致本节课数据全部丢失。
# --------------------------------------------------------------------------


class SaveSessionRequest(BaseModel):
    # 直接接收前端 sessionQueue 的原始 JSON 结构（每一项是一位学生的归档实体），
    # 不在后端做强类型校验——前端已经用 TypeScript 类型约束过结构，这里只管落盘。
    sessions: list[dict]


@app.post("/api/save_session")
def save_session(payload: SaveSessionRequest):
    """把前端当前完整的学生归档池，整体覆盖写入 B_group_web_sessions_log.json。

    【核心新增：跨课时双重持久化】前端已经把同一份数据同步写入了浏览器
    localStorage，这里再做一次后端 JSON 落盘，两边互为备份：即使老师的
    浏览器缓存被意外清空，下节课前也能通过 /api/load_sessions 从服务器
    这份 JSON 文件里把上节课的完整归档数据找回来。
    """
    try:
        payload_to_write = {
            "savedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sessions": payload.sessions,
        }
        with open(WEB_SESSION_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(payload_to_write, f, ensure_ascii=False, indent=2)
        return {"success": True, "savedCount": len(payload.sessions), "path": WEB_SESSION_LOG_PATH}
    except Exception as exc:  # noqa: BLE001 - 磁盘写入异常不应导致前端崩溃，只返回失败信息
        safe_print(f"【api_server】保存学生归档池到本地 JSON 失败：{exc}")
        return {"success": False, "error": str(exc)}


@app.get("/api/load_sessions")
def load_sessions():
    """读取后端本地 JSON 落盘的学生归档池，供前端在 localStorage 为空
    （例如换了一台电脑，或浏览器缓存被清空）时，作为"第二重保险"找回上节课数据。
    """
    if not os.path.exists(WEB_SESSION_LOG_PATH):
        return {"success": True, "sessions": [], "savedAt": None}
    try:
        with open(WEB_SESSION_LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        sessions = data.get("sessions", []) if isinstance(data, dict) else []
        saved_at = data.get("savedAt") if isinstance(data, dict) else None
        return {"success": True, "sessions": sessions, "savedAt": saved_at}
    except Exception as exc:  # noqa: BLE001 - 文件损坏时不影响前端正常使用，只返回空归档池
        safe_print(f"【api_server】读取学生归档池 JSON 文件失败：{exc}")
        return {"success": False, "sessions": [], "savedAt": None, "error": str(exc)}


# --------------------------------------------------------------------------
# 第五步再半：核心接口四 —— 调用 DeepSeek 生成「跨次尝试聚合诊断报告」
#
#         课后集中复盘看板里，教练查看某位同学 2~3 次尝试的整体趋势时，
#         前端会把这几次尝试各自的评分/三级命中统计打包发给这个接口，
#         后台真正调用 llm_agent.generate_aggregate_diagnosis() 请求
#         DeepSeek 大模型，生成"这几脚球之间发生了什么变化"的诊断建议。
# --------------------------------------------------------------------------


class AggregateAttemptSummary(BaseModel):
    attemptNumber: int
    score: Optional[int] = None
    hitStats: Optional[dict] = None


class GenerateAggregateReportRequest(BaseModel):
    student_number: str = ""
    attempts: list[AggregateAttemptSummary]


@app.post("/api/generate_aggregate_report")
def generate_aggregate_report(payload: GenerateAggregateReportRequest):
    """基于同一位学生本节课 2~3 次尝试的评分/三级命中统计，生成跨次趋势诊断。"""
    attempts_summary = [item.model_dump() for item in payload.attempts]

    ai_result = llm_agent.generate_aggregate_diagnosis(
        student_number=payload.student_number, attempts_summary=attempts_summary
    )

    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")

    # 【聚合稳定性得分计算】以各次尝试评分的离散程度（标准差）换算稳定性得分：
    # 各趟评分越接近，说明动作表现越稳定，得分越高；忽高忽低则相应扣分。
    scores = [item.score for item in payload.attempts if isinstance(item.score, (int, float))]
    if len(scores) >= 2:
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
        stability_score = int(max(0, min(100, round(100 - std_dev * 1.5))))
    elif len(scores) == 1:
        # 只有一次尝试时，没有"趋势"可言，直接沿用这一次的评分作为稳定性参考
        stability_score = int(scores[0])
    else:
        stability_score = 0

    full_text = (
        f"学号 {payload.student_number or '未填写'} 本节课多趟聚合诊断报告"
        f"（共 {len(attempts_summary)} 次尝试，动作表现稳定性得分 {stability_score} 分）\n\n"
        f"{ai_result['trendDescription']}\n"
        f"{ai_result['prescription']}"
    )

    return {
        "stabilityScore": stability_score,
        "trendDescription": ai_result["trendDescription"],
        "prescription": ai_result["prescription"],
        "fullText": full_text,
        "generatedAt": generated_at,
    }


# --------------------------------------------------------------------------
# 第五步再再半：核心接口五 —— 本地文件夹归档 + Word (.docx) 报告自动生成
#
#         前端「实时反馈」卡片的"自动归档并生成 Word 报告"按钮，以及
#         「延时反馈」单人/批量导出按钮，都会调用这个接口，完全在服务端
#         本地磁盘完成"建文件夹 + 写 .docx"两件事，绝不依赖浏览器的
#         直接下载。核心排版/建目录逻辑全部复用 word_reporter.py，本接口
#         只负责校验请求体、调用核心函数、把结果（含绝对物理路径）吐给前端。
# --------------------------------------------------------------------------


class SaveWordReportRequest(BaseModel):
    # "realtime" | "delayed" —— 对应一级归档子文件夹「实时反馈」/「延时反馈」
    mode: str = "realtime"
    # 学校/机构名称、班级/实验组别名称 —— 拼接成二级归档子文件夹
    school: str = ""
    classGroup: str = ""
    # 学生编号/学号 —— 三级归档子文件夹，也用于 Word 文件命名
    studentNumber: str = ""
    # AI 诊断报告核心字段：发力综合评分、有效采样次数、痛点分析、改进建议
    score: Optional[int] = None
    totalAttempts: Optional[int] = None
    painPoint: str = ""
    prescription: str = ""
    # 报告生成时间戳（前端已格式化好的字符串），缺省时后端自动补当前时间
    generatedAt: Optional[str] = None
    # 后端 OpenCV 矢量标注过的击球关键帧截图，Base64/data URI 字符串，可为空
    impactFrameImage: Optional[str] = None
    # 【v3.0 新增：集体错误热力图数据源】本次分析的三级容错命中次数统计
    # （Green/Yellow/Red），前端 finalReport.hitStats / attempt.reportData.hitStats
    # 原样转发过来，后端据此推导出本条记录归属的生物力学错误分类标签，
    # 供教练端看板的「集体错误热力图」统计全班高频失误分布使用。
    hitStats: Optional[dict] = None
    # 【v4.0 新增：科研级数据矩阵】本次分析全程真实测得的膝关节屈曲角度均值
    # （来自 /api/generate_report 返回的 avgKneeAngle，是 pose_tracker.py 逐帧
    # 真实计算出的物理测量值）。缺失时（例如历史联调数据、前端尚未回填）后端
    # 会自动退化为基于评分的启发式估算，确保导出的学术矩阵绝不出现空值。
    kneeFlexionAngle: Optional[float] = None


# 【v3.0 新增：生物力学错误分类体系】
#
# 项目当前真实落地的传感诊断参数只有「摆动腿触球瞬间膝关节屈曲角度」这一个
# （见 project_plan.md 第2节），支撑脚落位/髋关节旋转/踝关节锁定等维度尚未
# 接入真实的多点位姿测量。为了让教练端「集体错误热力图」在多维度呈现全班
# 通病分布的同时不过度虚构不存在的数据，这里采用一套启发式规则：
# 以红/黄命中率与综合评分作为唯一的真实信号，映射到运动生物力学教研领域
# 常见的四个动作诊断维度上——未来 pose_tracker.py 接入更多关节点测量后，
# 可以直接在这里替换为真正基于多维坐标计算出的独立分类，接口/看板侧完全
# 不需要改动。
BIOMECH_ERROR_TAXONOMY = [
    "支撑脚位置偏离",
    "膝关节过度屈曲",
    "随摆转髋不足",
    "身体重心偏移",
]


def _classify_biomechanical_errors(hit_stats: Optional[dict], score: Optional[int]) -> list[str]:
    """根据本次尝试的三级命中统计与综合评分，启发式推导出本条记录命中的
    生物力学错误分类标签列表（可能为空，也可能同时命中多个维度）。
    """
    if not hit_stats:
        return []
    try:
        green = float(hit_stats.get("green", 0) or 0)
        yellow = float(hit_stats.get("yellow", 0) or 0)
        red = float(hit_stats.get("red", 0) or 0)
    except (TypeError, ValueError):
        return []

    total = green + yellow + red
    if total <= 0:
        return []

    red_rate = red / total
    yellow_rate = yellow / total

    errors: list[str] = []
    if red_rate >= 0.30:
        errors.append("支撑脚位置偏离")
    if red_rate >= 0.15:
        errors.append("膝关节过度屈曲")
    if yellow_rate >= 0.25:
        errors.append("随摆转髋不足")
    if isinstance(score, (int, float)) and score < 60:
        errors.append("身体重心偏移")
    return errors


# 【v4.0 新增：学术统计矩阵数值编码】把生物力学错误分类标签映射为 SPSS/Mplus
# 友好的整数编码：0=合规，1=支撑脚偏离，2=膝角不足，3=重心后坐。按下面的
# 优先级顺序取"本条记录最主要的一个"错误分类（与 painPoint 单一焦点原则一致，
# 一条记录只落地一个主要错误编码，避免长表格式里出现无法二次编码的多值字段）。
PRIMARY_ERROR_CODE_PRIORITY: list[tuple[str, int]] = [
    ("支撑脚位置偏离", 1),
    ("膝关节过度屈曲", 2),
    ("身体重心偏移", 3),
]


def _derive_primary_error_code(errors: Optional[list]) -> int:
    """把 biomechanicalErrors 标签列表折算成单一的主要错误编码（0-3）。"""
    if not errors:
        return 0
    for label, code in PRIMARY_ERROR_CODE_PRIORITY:
        if label in errors:
            return code
    return 0


# 【v4.0 新增：科研级数据矩阵启发式补全】项目当前真实落地的传感诊断参数只有
# 「摆动腿触球瞬间膝关节屈曲角度」这一个是逐帧真实测量值（来自 pose_tracker.py），
# 「支撑脚离球距离」尚未接入真实的多点位坐标测量。为了保证导出给 SPSS/Excel 的
# 学术宽表严格做到"完全无缺失值"，这里用与 _classify_biomechanical_errors 完全
# 同源的启发式规则，基于综合评分反推一个物理上合理、单调对应的估算值——分数越
# 接近满分，估算角度越贴近 140°-160° 黄金区间中心、估算支撑脚距离越贴近
# 15-20cm 理想区间中心；分数越低，两个估算值都相应地朝越界方向偏移。
# 【重要说明】任何时候真实测量值可用（例如 /api/generate_report 返回的
# avgKneeAngle），都必须优先使用真实值，只有在真实值缺失时才退化到这里的估算。
_KNEE_ANGLE_OPTIMAL_CENTER = 150.0
_SUPPORT_FOOT_DISTANCE_IDEAL_CENTER = 17.5


def _estimate_knee_flexion_angle(score: Optional[float]) -> float:
    safe_score = score if isinstance(score, (int, float)) else 50.0
    angle = _KNEE_ANGLE_OPTIMAL_CENTER - (100.0 - safe_score) * 0.35
    return round(max(95.0, min(185.0, angle)), 1)


def _estimate_support_foot_distance(score: Optional[float]) -> float:
    safe_score = score if isinstance(score, (int, float)) else 50.0
    distance = _SUPPORT_FOOT_DISTANCE_IDEAL_CENTER + (100.0 - safe_score) * 0.15
    return round(max(5.0, min(45.0, distance)), 1)


def _extract_test_date(timestamp_text: Optional[str]) -> str:
    """从 "YYYY-MM-DD HH:mm:ss" 格式的时间戳字符串里安全提取出 "YYYY-MM-DD" 日期段，
    格式异常时兜底返回当前系统日期，确保导出的学术矩阵 test_date 列绝不出现空值。
    """
    text = (timestamp_text or "").strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return time.strftime("%Y-%m-%d")


def _load_global_records() -> list[dict]:
    """安全读取全局训练数据库的完整记录列表，文件不存在/损坏时静默兜底为空列表。"""
    if not os.path.exists(GLOBAL_DB_PATH):
        return []
    try:
        with open(GLOBAL_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as exc:  # noqa: BLE001 - 文件损坏不应阻断新记录的追加写入
        safe_print(f"【api_server】读取全局训练数据库失败（将视为空库继续追加）：{exc}")
        return []


def _append_global_record(record: dict) -> None:
    """把一条新记录追加进全局训练数据库并整体覆盖落盘，用锁保证并发写入安全。"""
    with _global_db_lock:
        records = _load_global_records()
        records.append(record)
        with open(GLOBAL_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


@app.post("/api/save_word_report")
def save_word_report(payload: SaveWordReportRequest):
    """接收前端组装好的学生档案 + AI 诊断报告 + 关键帧图片 Base64 + 模式类型，
    真正调用 word_reporter.save_feedback_to_word() 完成本地目录树建立与
    Word (.docx) 文档落地，把生成文件的绝对物理路径连同成功消息一并返回。

    【v2.0 新增：双向同步全局数据库】写盘成功后，会自动把这笔完整记录（id、
    时间、学校班级、学号、模式类型、评分、AI 批注、关键帧截图、文件路径）
    追加保存进项目根目录的 global_training_db.json，供教练端看板统一消费；
    同时把这条记录原样返回给前端，前端再同步写入 localStorage 作为极速双保险。

    【健壮性说明】word_reporter.save_feedback_to_word() 内部已经对 Base64 图片
    解码异常、Windows 非法文件名字符做了完整防呆处理，本接口这里只需要再兜底
    捕获一层意料之外的异常（例如磁盘写满/权限不足等系统级错误），确保接口
    永远返回结构化的 JSON 结果，绝不会给前端返回一个裸的 500 错误页面。
    """
    try:
        result = word_reporter.save_feedback_to_word(payload.model_dump())
    except Exception as exc:  # noqa: BLE001 - 任何未预料的系统级异常都不应让接口直接崩溃
        safe_print(f"【api_server】保存 Word 报告时发生未预料的异常：{exc}")
        return {"success": False, "message": f"保存 Word 报告失败：{exc}"}

    if result.get("success"):
        record_type = "delayed" if payload.mode == "delayed" else "realtime"
        ai_feedback_text = "\n".join(
            part.strip() for part in (payload.painPoint, payload.prescription) if part and part.strip()
        )

        record_timestamp = payload.generatedAt or time.strftime("%Y-%m-%d %H:%M:%S")
        biomechanical_errors = _classify_biomechanical_errors(payload.hitStats, payload.score)

        record = {
            "id": str(uuid.uuid4()),
            "timestamp": record_timestamp,
            "school": payload.school or "",
            "classGroup": payload.classGroup or "",
            "studentId": payload.studentNumber or "",
            "type": record_type,
            "score": payload.score,
            "biomechanicalErrors": biomechanical_errors,
            "aiFeedback": ai_feedback_text,
            "impactFrameBase64": payload.impactFrameImage,
            "path": result["path"],
            "directory": result.get("directory"),
            # 【v4.0 新增：科研级数据矩阵字段】详见 project_plan.md 第4节新增需求——
            # 供教练端「双轴运动学成长期刊图」与后台 /api/export_academic_matrix
            # 学术统计矩阵导出直接消费，写入时一次性补全，避免看板/导出侧重复计算。
            "testDate": _extract_test_date(record_timestamp),
            "groupTypeCode": 1 if record_type == "realtime" else 2,
            "kneeFlexionAngle": (
                round(float(payload.kneeFlexionAngle), 1)
                if isinstance(payload.kneeFlexionAngle, (int, float))
                else _estimate_knee_flexion_angle(payload.score)
            ),
            "supportFootDistance": _estimate_support_foot_distance(payload.score),
            "primaryErrorCode": _derive_primary_error_code(biomechanical_errors),
        }
        try:
            _append_global_record(record)
        except Exception as exc:  # noqa: BLE001 - 数据库追加失败不应影响 Word 本身已经保存成功的结果
            safe_print(f"【api_server】追加记录到全局训练数据库失败（Word 文件已正常保存）：{exc}")

        return {
            "success": True,
            "message": f"报告已自动保存成 Word！文件已存入：{result['path']}",
            "path": result["path"],
            "directory": result.get("directory"),
            "filename": result.get("filename"),
            "record": record,
        }

    return {
        "success": False,
        "message": f"保存 Word 报告失败：{result.get('error', '未知错误')}",
    }


@app.get("/api/get_all_records")
def get_all_records():
    """供教练端数据看板一键拉取全量历史归档数据（实时反馈 A 组 + 延时反馈 B 组）。"""
    records = _load_global_records()
    return {"success": True, "records": records, "count": len(records)}


# --------------------------------------------------------------------------
# 第五步再再再再半：核心接口七 —— 「论文专供：学术统计矩阵一键自动导出」
#
#         教练端看板顶栏「📥 一键导出科研论文数据矩阵」按钮调用本接口。全部
#         数据清洗、长表格式转换、数值编码逻辑都封装在 academic_exporter.py，
#         本接口只负责读取全局训练数据库、调用核心导出函数、把落盘的物理
#         路径与统计信息吐给前端弹窗展示。
# --------------------------------------------------------------------------


@app.post("/api/export_academic_matrix")
def export_academic_matrix():
    """一键清洗 global_training_db.json 全量记录，导出规范的 SPSS/Excel 长表格式
    学术统计矩阵 CSV，落盘存入项目根目录 academic_data_export/ 文件夹。
    """
    records = _load_global_records()
    try:
        result = academic_exporter.export_academic_matrix(records)
    except Exception as exc:  # noqa: BLE001 - 导出失败不应让接口直接抛 500，交由前端友好提示
        safe_print(f"【api_server】导出学术统计矩阵失败：{exc}")
        return {"success": False, "message": f"导出学术统计矩阵失败：{exc}"}

    if not result.get("success"):
        return result

    return {
        "success": True,
        "message": (
            f"✅ 科研数据矩阵已清洗完毕并数字化编码！文件已存入："
            f"{result['path']}，可直接导入 SPSS、Excel 或 Mplus 跑方差分析！"
        ),
        "path": result["path"],
        "filename": result["filename"],
        "rowCount": result["rowCount"],
        "studentCount": result["studentCount"],
    }


# --------------------------------------------------------------------------
# 第五步再再再半：核心接口六 —— 教练端科研指挥中心：
# 「全班集体宏观诊断」AIGC 处方 + 「个体纵向进化画像」AI 优缺点总结
# --------------------------------------------------------------------------


class GenerateClassPrescriptionRequest(BaseModel):
    school: str = ""
    classGroup: str = ""
    # 键为错误分类标签，值为该分类在全班记录中的出现百分比（0-100）
    errorStats: dict[str, float] = {}
    totalRecords: int = 0
    avgScore: Optional[float] = None


@app.post("/api/generate_class_prescription")
def generate_class_prescription_endpoint(payload: GenerateClassPrescriptionRequest):
    """✨「召唤 AI 生成全班改进教案」：基于该班级全部历史记录的生物力学错误
    分布统计，调用 DeepSeek 生成一份结构严谨的集体教学诊断简报 + 处方。
    """
    ai_result = llm_agent.generate_class_prescription(
        school=payload.school,
        class_group=payload.classGroup,
        error_stats=payload.errorStats,
        total_records=payload.totalRecords,
        avg_score=payload.avgScore,
    )
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    full_text = f"{ai_result['diagnosis']}\n\n{ai_result['prescription']}"
    return {
        "diagnosis": ai_result["diagnosis"],
        "prescription": ai_result["prescription"],
        "fullText": full_text,
        "generatedAt": generated_at,
    }


class GenerateIndividualSummaryRequest(BaseModel):
    studentId: str = ""
    scoreHistory: list[float] = []
    # 键为错误分类标签，值为该生历史记录中该分类出现的次数
    errorCounter: dict[str, int] = {}


@app.post("/api/generate_individual_summary")
def generate_individual_summary_endpoint(payload: GenerateIndividualSummaryRequest):
    """「个体纵向进化追踪」档案：基于该生全周期历史评分与错误分类统计，
    调用 DeepSeek 生成结构化的「稳定发力优势」与「需克服习惯性盲区」总结。
    """
    ai_result = llm_agent.generate_individual_summary(
        student_id=payload.studentId,
        score_history=payload.scoreHistory,
        error_counter=payload.errorCounter,
    )
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "strengths": ai_result["strengths"],
        "weaknesses": ai_result["weaknesses"],
        "generatedAt": generated_at,
    }


class OpenFolderRequest(BaseModel):
    path: str


@app.post("/api/open_folder")
def open_folder(payload: OpenFolderRequest):
    """供教练端数据看板「📁 打开电脑文件夹」按钮调用：在本机文件管理器中，
    直接定位并打开某份 Word 报告所在的文件夹。跨平台兼容 Windows / macOS / Linux。
    """
    target_path = (payload.path or "").strip()
    if not target_path:
        return {"success": False, "message": "缺少文件夹路径参数"}
    if not os.path.exists(target_path):
        return {"success": False, "message": f"路径不存在（文件可能已被移动或删除）：{target_path}"}

    try:
        if sys.platform.startswith("win"):
            os.startfile(target_path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", target_path])
        else:
            subprocess.Popen(["xdg-open", target_path])
        return {"success": True, "message": "已在文件管理器中打开该文件夹"}
    except Exception as exc:  # noqa: BLE001 - 打开文件夹失败不应让接口抛出 500
        safe_print(f"【api_server】打开本地文件夹失败：{exc}")
        return {"success": False, "message": f"打开文件夹失败：{exc}"}


# --------------------------------------------------------------------------
# 第六步：程序入口 —— 支持直接 `python api_server.py` 启动
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
