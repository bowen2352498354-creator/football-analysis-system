# -*- coding: utf-8 -*-
"""
api_server.py
v1.1 前后端全栈联调阶段：后台服务网关（FastAPI + Uvicorn）

功能说明：
    本脚本是连接「React 前端网页 (AI-Football-Web)」与「Python 视觉/大模型算法
    (pose_tracker.py + llm_agent.py)」之间的唯一桥梁。它彻底废除了前端的假数据
    (mockData.ts) 生成逻辑，让网页真正显示后台实时推理出来的画面与数据。

    本文件【完全复用】pose_tracker.py / image_processing.py 里已经写好的核心算法
    （角度计算、三级容错判定、骨骼连线绘制、物理级面部脱敏 apply_facial_anonymization），
    不重复实现任何算法逻辑；只是把原来"画在 PyQt5 QLabel 上"的输出通道，换成了
    "通过 WebSocket 推给浏览器"的输出通道；也完全复用 llm_agent.py 里封装好的
    DeepSeek 调用逻辑。关键点提取后、推流编码与击球关键帧缓存前，强制用脱敏安全帧
    替换原始画面，确保错题本/对比照也是彻底无脸的。

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
    GET  /api/fatigue_alert  ：课堂疲劳熔断轮询（ANKLE_FATIGUE / KNEE_STIFFNESS）。
                                generate_report 写入时序后命中规则即缓存；教练端
                                「纵向双轴进化图谱」每 2.5s 拉取并渲染熔断闪烁卡。
    GET  /api/achievements/weekly ：SDT 游戏化周成就印章（钢铁锁踝王 / 最稳底盘奖 /
                                最快进步奖），拒绝总分排名，返回匿名学员编号与指标。

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

# 【V2.5】必须在任何可能触发 CUDA/PyTorch 的 import 之前锁死 CUBLAS 工作区
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import asyncio
import base64
import collections
import io
import json
import math
import queue
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

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
from fastapi.responses import Response
from pydantic import BaseModel

# 【核心复用】直接把 pose_tracker.py 当作一个模块导入，复用里面已经写好的
# 骨骼绘制 / 角度计算 / 三级容错判定 / 面部打码函数，绝不重复实现算法逻辑。
import pose_tracker as pt

# 【V2.5】导入时再次确认确定性锁死（pose_tracker 模块级已执行；此处幂等加固）
pt.lock_vision_pipeline_determinism()

# 【核心复用】直接复用 llm_agent.py 里封装好的 DeepSeek 调用逻辑。
import llm_agent

# 【V2.5】确定性评分 + Action ROI + 黄金审计日志
import error_diagnoser

# 【核心复用】直接复用 word_reporter.py 里封装好的本地归档 + Word 报告生成逻辑。
import word_reporter

# 【v4.0 核心复用】直接复用 academic_exporter.py 里封装好的「论文专供：学术统计
# 矩阵一键自动导出」清洗 + 落盘逻辑，完全不在本文件重复实现任何转换算法。
import academic_exporter

# 【V2.5 Cluster-RCT】教练端科研控制台：干预剂量监控 + 极端个案目的性抽样。
# 业务逻辑全部封装在 ResearchDashboardService，本文件只挂路由与透传查询参数。
import research_dashboard_service
from research_models import STANDARD_SHOT_DOSE

# 【疲劳熔断】复用 session_monitor 判定阈值与静态评估函数（不实例化 QObject）。
from session_monitor import (
    FATIGUE_MESSAGES,
    MIN_ATTEMPTS_FOR_MONITOR,
    BASELINE_WINDOW,
    RECENT_WINDOW,
    FatigueMonitor,
    flatten_eight_metrics,
)

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

# 【疲劳熔断】课堂时序监控状态（纯 Python，不依赖 PyQt QObject 信号总线）
# 供教练端 / 延时组看板轮询 GET /api/fatigue_alert；generate_report 成功后写入。
_fatigue_history_lock = threading.Lock()
_fatigue_attempts: dict[str, list] = {}  # student_id -> flatten_eight_metrics 行
_latest_fatigue_alerts: dict[str, dict] = {}  # student_id -> 报警字典
_global_latest_fatigue: Optional[dict] = None

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

# 【V2.5 竞态防护】/api/generate_report 若早于分析完成到达，最长挂起等待秒数
REPORT_WAIT_TIMEOUT_SEC = 600.0

# 任务状态常量（AnalysisSession.task_status）
TASK_STATUS_PROCESSING = "PROCESSING"
TASK_STATUS_COMPLETED = "COMPLETED"


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
        #
        # 【V2.5 确定性】录像分析禁用「满则丢最旧帧」语义：
        #   - file 模式：无界队列，保证推理过的每一帧都送达消费端，frame_count 绝对相等；
        #   - webcam 模式：仍用 maxsize=2 丢旧帧保实时性（实时流允许丢显示帧）。
        is_file = source == "file"
        self.frame_queue: "queue.Queue[dict]" = (
            queue.Queue() if is_file else queue.Queue(maxsize=2)
        )
        self._drop_frames_on_backpressure: bool = not is_file
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        # 【B组/科研数据落盘的联调等价物】本次分析全过程的有效诊断记录，
        # 结束分析后会被 /api/generate_report 读取，喂给 DeepSeek 生成真实报告。
        self.records: list[dict] = []
        self._records_lock = threading.Lock()

        # 【新增：实时动力链角速度监控】上一帧角度与时间戳，用于逐帧计算角速度（deg/s）
        self._prev_angle: Optional[float] = None
        self._prev_frame_time: Optional[float] = None
        # 录像分析：用固定 fps 推导 Δt，杜绝墙钟抖动导致角速度/触球帧跳变
        self._fixed_frame_dt: Optional[float] = None
        # 最近 STABILITY_WINDOW_SIZE 帧的角速度滑动窗口，用于计算"动平衡稳定指数"
        self._velocity_window: "collections.deque[float]" = collections.deque(maxlen=STABILITY_WINDOW_SIZE)

        # 【新增：击球关键帧自动捕捉】记录整趟练习中"右膝角速度绝对值最大"的那一帧
        # （即冲击最剧烈、最接近真实触球瞬间的画面），供生成报告时叠加矢量标注。
        self.impact_frame = None  # numpy 数组：命中的击球关键帧（已完成面部打码，未叠加骨骼线）
        self.impact_metrics: Optional[dict] = None  # 该帧对应的 hip/knee/ankle 像素坐标与角度/状态
        self._best_impact_score: float = -1.0
        # 【V2.5】逐帧轨迹缓存，分析结束后用 locate_impact_frame 抛物线锁帧覆写
        self._trajectory_angles: list[float] = []
        self._trajectory_omega: list[float] = []
        self._trajectory_ankle_px: list[tuple] = []
        self._trajectory_frames_blurred: list = []  # 可选：仅保留候选邻域，控制内存
        # 【Sprint 1】逐帧姿态关键点（支撑踝 / 摆腿踝等），供时空热力图坐标映射
        self._trajectory_pose_frames: list[dict] = []
        self.t_impact: Optional[int] = None
        self.sync_frame_count: int = 0

        # 【V2.5 竞态锁】PROCESSING → COMPLETED；generate_report 必须等 COMPLETED
        self.task_status: str = TASK_STATUS_PROCESSING
        self._completed_event = threading.Event()

        # 【新增：黑屏问题自动诊断】累计推送帧数（用于终端进度日志）+ 连续疑似全黑帧计数
        # （用于自动检测"摄像头权限被禁用/被占用/被遮挡"这一类无报错但画面全黑的情况）。
        self._pushed_frame_count = 0
        self._consecutive_dark_frames = 0
        self._dark_frame_warning_sent = False

    def start(self):
        self.task_status = TASK_STATUS_PROCESSING
        self._completed_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def request_stop(self):
        self.stop_event.set()

    def wait_until_completed(self, timeout: float = REPORT_WAIT_TIMEOUT_SEC) -> bool:
        """阻塞直到分析线程将状态标为 COMPLETED（或超时）。返回是否已完成。"""
        if self.task_status == TASK_STATUS_COMPLETED:
            return True
        return bool(self._completed_event.wait(timeout=timeout))

    def mark_completed(self) -> None:
        """幂等：标记任务完成并唤醒所有等待 generate_report 的调用方。"""
        self.task_status = TASK_STATUS_COMPLETED
        self._completed_event.set()

    def get_records_snapshot(self) -> list[dict]:
        with self._records_lock:
            return list(self.records)

    def build_scoring_payloads(self) -> tuple[dict, dict]:
        """从本会话轨迹构造 DeterministicScorer 所需的 impact / trajectory 载荷。"""
        knee_angles = list(self._trajectory_angles)
        omega = list(self._trajectory_omega)
        pose_frames = list(self._trajectory_pose_frames)
        n = len(knee_angles)
        t_impact = int(self.t_impact) if self.t_impact is not None else (
            int(max(range(n), key=lambda i: abs(omega[i]))) if n > 0 else 0
        )
        if n > 0:
            t_impact = int(max(0, min(n - 1, t_impact)))

        # 触球帧球心锚点：优先该帧右足尖（与 lock_absolute_t0 口径一致）
        ball_center = None
        if pose_frames and 0 <= t_impact < len(pose_frames):
            rec = pose_frames[t_impact]
            world = rec.get("world") if isinstance(rec, dict) else None
            if isinstance(world, dict) and world.get("right_foot_index") is not None:
                ball_center = world["right_foot_index"]
            elif isinstance(rec, dict):
                ball_center = rec.get("right_foot_index") or rec.get("right_ankle")

        impact_metrics = self.impact_metrics or {}
        impact_frame_data = {
            "t_impact": t_impact,
            "task_id": self.session_id,
            "session_id": self.session_id,
            "total_frames": n,
            "frames": pose_frames,
            "ball_center": ball_center,
            "impact_knee_angle": impact_metrics.get("angle"),
            "distance_cm": impact_metrics.get("distance_cm"),
            "toe_angle": impact_metrics.get("toe_angle"),
            "support_knee_angle": impact_metrics.get("support_knee_angle"),
            "hip_torsion_angle": impact_metrics.get("hip_torsion_angle"),
        }
        dt = float(self._fixed_frame_dt) if self._fixed_frame_dt else (1.0 / 30.0)
        trajectory_data = {
            "task_id": self.session_id,
            "session_id": self.session_id,
            "knee_angles": knee_angles,
            "angular_velocities": omega,
            "timestamps_sec": [i * dt for i in range(n)],
            "total_frames": n,
            "t_impact": t_impact,
            "frames": pose_frames,
            "ball_center": ball_center,
            "whipping_velocity": float(max((abs(v) for v in omega), default=0.0)),
        }
        return impact_frame_data, trajectory_data

    def build_time_series_velocity_window(
        self, t_impact: Optional[int] = None
    ) -> tuple[list[float], int, int]:
        """裁剪 Action ROI 内的摆动腿小腿连续角速度序列（KinematicSignalProcessor 平滑后）。

        窗口为 ``[t_impact-30, t_impact+30)``（最长约 60 帧）。返回：
            (time_series_velocity, impact_index_in_window, roi_start)
        边界未截断时 ``impact_index_in_window`` 恒为 30（数组中心）。
        """
        omega_raw = list(self._trajectory_omega)
        n = len(omega_raw)
        if n <= 0:
            return [], 0, 0

        omega_smooth = pt.KinematicSignalProcessor.smooth_joint_trajectories(omega_raw)
        if t_impact is None:
            t_impact = int(self.t_impact) if self.t_impact is not None else int(
                max(range(n), key=lambda i: abs(float(omega_smooth[i])))
            )
        t = int(max(0, min(n - 1, int(t_impact))))
        roi_start, roi_end = error_diagnoser.slice_action_roi_bounds(t, n)
        window = [round(float(v), 2) for v in omega_smooth[roi_start:roi_end]]
        impact_index_in_window = int(t - roi_start)
        return window, impact_index_in_window, int(roi_start)

    def _compute_angular_velocity(self, angle: float) -> float:
        """根据"当前帧角度 - 上一帧角度"除以帧间时间差，计算右膝角速度（deg/s）。
        第一帧因为没有"上一帧"可比较，约定角速度为 0。

        【V2.5】录像模式优先使用固定 fps 的 Δt，避免 wall-clock 抖动导致分数跳变。
        """
        angular_velocity = 0.0
        if self._prev_angle is not None:
            if self._fixed_frame_dt is not None and self._fixed_frame_dt > 0:
                dt = self._fixed_frame_dt
            else:
                now = time.time()
                dt = (now - self._prev_frame_time) if self._prev_frame_time is not None else 0.0
            if dt > 0:
                angular_velocity = (angle - self._prev_angle) / dt
        self._prev_angle = angle
        self._prev_frame_time = time.time()
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

        重要：必须在 pt.apply_facial_anonymization()（或兼容别名 apply_face_blur）
        之后、pt.draw_pose_landmarks() / pt.draw_right_knee_overlay() 之前调用——
        既要保证脸部已经打码（隐私红线），又要保证存下来的是一张"干净"的画面，
        方便后续单独叠加矢量标注，不会与实时预览用的白色骨骼线/粗染色线互相干扰。
        """
        height, width = frame.shape[:2]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        mid_hip_px = (
            int((left_hip.x + right_hip.x) / 2 * width),
            int((left_hip.y + right_hip.y) / 2 * height),
        )

        # 【隐私红线】落盘/缓存前再次强制脱敏，杜绝任何旁路漏网的原始带脸帧
        safe_frame = pt.apply_facial_anonymization(frame, landmarks)
        self.impact_frame = safe_frame.copy()
        self.impact_metrics = {
            "hip_px": hip_px,
            "knee_px": knee_px,
            "ankle_px": ankle_px,
            "mid_hip_px": mid_hip_px,
            "angle": angle,
            "status": status,
        }

    def _push_frame_payload(self, payload: dict):
        """把一份处理好的帧数据放进队列。

        【V2.5】录像分析（file）严禁丢帧：阻塞式 put，保证 frame_count 绝对相等。
        仅实时摄像头模式允许在背压时丢弃最旧显示帧。
        """
        if not self._drop_frames_on_backpressure:
            self.frame_queue.put(payload)
            return
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

    def _finalize_impact_with_parabolic_lock(self) -> None:
        """分析结束后用 locate_impact_frame（抛物线插值）覆写流式峰值候选，零漂移锁帧。"""
        n = len(self._trajectory_omega)
        if n < 3 or len(self._trajectory_ankle_px) < n:
            if n > 0:
                self.t_impact = int(max(range(n), key=lambda i: abs(self._trajectory_omega[i])))
            return

        # 球心代理：若无独立球检测，用整段踝坐标中位数作为静止球近似（操场固定机位）
        ankles = self._trajectory_ankle_px[:n]
        xs = [float(a[0]) for a in ankles]
        ys = [float(a[1]) for a in ankles]
        # 取踝轨迹 Y 较大的 15% 分位中位数作为触地球区近似球心
        order = sorted(range(n), key=lambda i: ys[i])
        tail = order[int(n * 0.85) :] or order[-1:]
        ball_x = float(sum(xs[i] for i in tail) / len(tail))
        ball_y = float(sum(ys[i] for i in tail) / len(tail))
        ball_coords = [(ball_x, ball_y) for _ in range(n)]

        omega_smooth = pt.KinematicSignalProcessor.smooth_joint_trajectories(
            list(self._trajectory_omega[:n])
        )
        t_impact = pt.locate_impact_frame(omega_smooth, ankles, ball_coords)
        self.t_impact = int(t_impact)
        safe_print(
            f"【api_server】[V2.5] 抛物线触球锁帧 t_impact={self.t_impact} "
            f"（同步总帧数 frame_count={self.sync_frame_count}）",
            flush=True,
        )

    def _run(self):
        """后台工作线程主体：逐帧读取视频源 -> 姿态检测 -> 力学诊断 ->
        骨骼染色 + 面部打码渲染 -> 编码成 Base64 JPEG -> 推入队列。

        这里的每一步算法逻辑，都是直接调用 pose_tracker.py (pt 模块) 里
        已经写好、且已经在桌面版软件里验证过的函数，完全没有重新实现。
        """
        cap = None
        landmarker = None
        is_video_file_mode = self.source == "file"
        video_fps = 30.0

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
                cap, video_fps, _reported = pt.open_video_capture_deterministic(
                    self.video_path, is_camera=False
                )
            else:
                cap, video_fps, _reported = pt.open_video_capture_deterministic(
                    "", is_camera=True, camera_index=self.camera_index
                )

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
                # 【V2.5】录像分析：固定 Δt，角速度与 MediaPipe 时间戳完全由 fps 决定
                self._fixed_frame_dt = 1.0 / float(video_fps)
                frame_delay_seconds = self._fixed_frame_dt
            else:
                self._fixed_frame_dt = None
                frame_delay_seconds = 0.0

            # 【V2.5】每次分析任务：销毁旧 PoseLandmarker/YOLO 记忆并重建干净实例
            task_handles = pt.start_analysis_task(reset_yolo=True)
            landmarker = task_handles["pose_landmarker"]

            frame_interval_ms = int(round(1000.0 / float(video_fps)))
            frame_timestamp_ms = 0
            self.sync_frame_count = 0

            # 【V2.5】同步阻断式 while cap.read()：录像路径严禁跳帧/丢帧
            # 录像文件模式：忽略 stop_event，必须读到 EOF，避免报告竞态截断在 300/414 帧。
            # 摄像头模式：允许 stop_event 提前结束。
            while cap.isOpened():
                if (not is_video_file_mode) and self.stop_event.is_set():
                    break

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
                            f"（本次分析同步读入 frame_count={self.sync_frame_count}，"
                            f"成功推送 {self._pushed_frame_count} 帧），分析自然结束。",
                            flush=True,
                        )
                    break

                self.sync_frame_count += 1

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
                        f"视频推理管线已正常启动（V2.5 同步顺序帧 / 模型热重置）。"
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
                # MediaPipe VIDEO 时间戳按真实 fps 递增，杜绝写死 33ms 造成的跨次漂移
                frame_timestamp_ms += frame_interval_ms
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

                        # 轨迹缓存（供结束后抛物线锁帧）
                        self._trajectory_angles.append(float(angle))
                        self._trajectory_omega.append(float(angular_velocity))
                        self._trajectory_ankle_px.append(
                            (float(ankle_px[0]), float(ankle_px[1]))
                        )
                        # Sprint 1：支撑脚 / 摆腿时空热力图所需逐帧关键点
                        try:
                            world_lms = None
                            if getattr(results, "pose_world_landmarks", None):
                                world_lms = results.pose_world_landmarks[0]
                            ts_sec = (
                                float(self.sync_frame_count - 1) * float(self._fixed_frame_dt)
                                if self._fixed_frame_dt
                                else float(self.sync_frame_count - 1) / 30.0
                            )
                            self._trajectory_pose_frames.append(
                                pt.serialize_pose_frame_record(
                                    landmarks,
                                    frame.shape,
                                    timestamp_sec=ts_sec,
                                    world_landmarks=world_lms,
                                )
                            )
                        except Exception:  # noqa: BLE001 - 热力图序列化失败不阻断主诊断链路
                            ts_sec = (
                                float(self.sync_frame_count - 1) * float(self._fixed_frame_dt)
                                if self._fixed_frame_dt
                                else float(self.sync_frame_count - 1) / 30.0
                            )
                            self._trajectory_pose_frames.append(pt.empty_pose_frame_record(ts_sec))

                        # 【绝对拦截器 / Choke Point】关键点提取后立即强制替换为脱敏安全帧，再画骨骼线；
                        # 这是符合《未成年人保护法》与科研伦理审查的物理级脱敏，任何人不得在此行代码之前进行原图转存。
                        # 顺序严格保持：先打码，再捕捉击球关键帧，最后叠加染色骨骼线。
                        frame = pt.apply_facial_anonymization(frame, landmarks)

                        # 【击球关键帧自动捕捉】在打码之后、骨骼线绘制之前，
                        # 用"角速度绝对值是否为整趟练习目前最大值"来判定是否更新击球关键帧候选，
                        # 角速度越大代表这一帧越接近真实的"发力冲击瞬间"。
                        # 写入 impact_frame 的一定是无脸安全图像（错题本/对比照同理）。
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
                            "frame_index": self.sync_frame_count - 1,
                        }
                        with self._records_lock:
                            self.records.append(record)
                    else:
                        # 无姿态帧：仍计入同步帧序列长度，用中性值填轨迹以保持索引对齐
                        self._trajectory_angles.append(
                            float(self._trajectory_angles[-1]) if self._trajectory_angles else 150.0
                        )
                        self._trajectory_omega.append(0.0)
                        self._trajectory_ankle_px.append(
                            self._trajectory_ankle_px[-1] if self._trajectory_ankle_px else (0.0, 0.0)
                        )
                        ts_sec = (
                            float(self.sync_frame_count - 1) * float(self._fixed_frame_dt)
                            if self._fixed_frame_dt
                            else float(self.sync_frame_count - 1) / 30.0
                        )
                        self._trajectory_pose_frames.append(pt.empty_pose_frame_record(ts_sec))
                except Exception as diagnosis_exc:  # noqa: BLE001 - 单帧诊断异常绝不能打断整条视频流
                    safe_print(f"【api_server】单帧姿态诊断/角速度计算发生异常（已跳过该帧诊断信息，画面仍会继续推送）：{diagnosis_exc}")
                    angle_value = None
                    status_value = None
                    angular_velocity_value = None
                    stability_index_value = None
                    self._trajectory_angles.append(
                        float(self._trajectory_angles[-1]) if self._trajectory_angles else 150.0
                    )
                    self._trajectory_omega.append(0.0)
                    self._trajectory_ankle_px.append(
                        self._trajectory_ankle_px[-1] if self._trajectory_ankle_px else (0.0, 0.0)
                    )
                    ts_sec = (
                        float(self.sync_frame_count - 1) * float(self._fixed_frame_dt)
                        if self._fixed_frame_dt
                        else float(self.sync_frame_count - 1) / 30.0
                    )
                    self._trajectory_pose_frames.append(pt.empty_pose_frame_record(ts_sec))

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
                    "frame_index": self.sync_frame_count - 1,
                    "timestamp": time.time(),
                })

                if is_video_file_mode and frame_delay_seconds > 0:
                    elapsed = time.time() - loop_start_time
                    remaining = frame_delay_seconds - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

            # 录像跑完后：抛物线插值锁定全局唯一 t_impact
            if is_video_file_mode:
                self._finalize_impact_with_parabolic_lock()

        except Exception as exc:  # noqa: BLE001 - 后台线程内的任何异常都不能让服务崩溃
            # 【新增】同步把异常信息打印到服务端终端：之前这里只把错误通过 WebSocket
            # 发给前端，终端里完全看不到任何报错痕迹，导致排查"黑屏"问题时无从下手。
            safe_print(f"【api_server】后台推理线程发生异常，本次分析会话将提前结束：{exc}")
            self._push_frame_payload({"type": "error", "message": f"后台推理线程发生异常：{exc}"})

        finally:
            if cap is not None:
                cap.release()
            if landmarker is not None:
                pt.destroy_pose_landmarker(landmarker)
            # 【V2.5】必须在推送 stopped 之前标记 COMPLETED，唤醒挂起的 generate_report
            self.mark_completed()
            self._push_frame_payload({
                "type": "stopped",
                "session_id": self.session_id,
                "total_records": len(self.records),
                "frame_count": self.sync_frame_count,
                "t_impact": self.t_impact,
                "task_status": self.task_status,
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

    【V2.5 竞态防护】若 WebSocket 仍在推帧 / MediaPipe 尚未跑完，本接口会挂起
    等待 AnalysisSession.task_status == COMPLETED，禁止用半截轨迹打分。

    后台会：
        1) 等待分析 COMPLETED；
        2) 用 DeterministicScorer + Action ROI（t_impact±30）解算确定性总分；
        3) 打印黄金审计日志；
        4) 汇总三级命中统计并调用 llm_agent 生成文字痛点/处方；
        5) 返回结构化报告 JSON（分数以确定性引擎为准）。
    """
    session = SESSIONS.get(payload.session_id)

    # ---------- 竞态锁：未完成则挂起等待 ----------
    if session is not None and session.task_status != TASK_STATUS_COMPLETED:
        safe_print(
            f"【api_server】[竞态防护] generate_report 早到：session={payload.session_id} "
            f"status={session.task_status}，挂起等待 COMPLETED（超时 {REPORT_WAIT_TIMEOUT_SEC:.0f}s）…",
            flush=True,
        )
        finished = session.wait_until_completed(timeout=REPORT_WAIT_TIMEOUT_SEC)
        if not finished:
            safe_print(
                f"【api_server】[竞态防护] 等待超时：session={payload.session_id} "
                f"仍为 {session.task_status}，将基于当前已采集轨迹继续报告（可能不完整）。",
                flush=True,
            )
        else:
            safe_print(
                f"【api_server】[竞态防护] 分析已 COMPLETED：frame_count={session.sync_frame_count}，"
                f"t_impact={session.t_impact}，开始确定性打分。",
                flush=True,
            )

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

    # ---------- V2.5 确定性打分（Action ROI）+ 黄金审计 ----------
    deterministic_score = None
    score_detail = None
    t_impact_locked = None
    heatmap_base64 = None
    spatial_trajectory = None
    if session is not None and len(session._trajectory_angles) > 0:
        impact_payload, trajectory_payload = session.build_scoring_payloads()
        deterministic_score, score_detail = error_diagnoser.calculate_biomechanical_score(
            impact_payload, trajectory_payload
        )
        t_impact_locked = int(score_detail.get("t_impact", session.t_impact or 0))
        heatmap_base64 = score_detail.get("heatmap_base64")
        spatial_trajectory = score_detail.get("spatial_trajectory")
        # 若打分路径未产出热力图（极短序列等），再显式用姿态序列补一次
        if not heatmap_base64 and session._trajectory_pose_frames:
            try:
                heat = error_diagnoser.build_spatial_heatmap_payload(
                    session._trajectory_pose_frames,
                    t_impact_locked,
                    ball_center_t_impact=impact_payload.get("ball_center"),
                )
                heat.pop("_canvas_bgr", None)
                heatmap_base64 = heat.get("heatmap_base64")
                spatial_trajectory = {
                    k: v
                    for k, v in heat.items()
                    if k not in ("heatmap_base64", "heatmap_data_uri", "_canvas_bgr")
                }
                if isinstance(score_detail, dict):
                    score_detail["heatmap_base64"] = heatmap_base64
                    score_detail["spatial_trajectory"] = spatial_trajectory
            except Exception as heat_exc:  # noqa: BLE001
                safe_print(f"【api_server】时空热力图生成失败（不影响评分）：{heat_exc}")
        error_diagnoser.print_golden_audit_log(
            task_id=payload.session_id,
            knee_angle_count=len(trajectory_payload.get("knee_angles") or []),
            impact_frame_idx=t_impact_locked,
            final_score=float(deterministic_score),
        )

    ai_result = llm_agent.generate_session_report(
        hit_stats=hit_stats, student_number=payload.student_number, sample_angles=sample_angles
    )

    # 分数以确定性引擎为准；无轨迹时回退 LLM 分
    final_score = (
        float(deterministic_score)
        if deterministic_score is not None
        else float(ai_result["score"])
    )

    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    full_text = (
        f"学号 {payload.student_number or '未填写'} 本次综合练习诊断报告\n\n"
        f"发力稳定性评分：{final_score:.2f} 分（共采集 {total_attempts} 次有效触球数据）。\n"
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

    # V2.5 Kinovea 联动：全程角速度 + Action ROI 鞭打发力窗口（须在 pop 前读完）
    angular_velocities_out = None
    frame_count_out = len(sample_angles)
    time_series_velocity: Optional[list] = None
    impact_index_in_window: Optional[int] = None
    if session is not None:
        angular_velocities_out = [float(v) for v in session._trajectory_omega]
        frame_count_out = int(
            getattr(session, "sync_frame_count", None) or len(session._trajectory_omega) or len(sample_angles)
        )
        if len(session._trajectory_omega) > 0:
            time_series_velocity, impact_index_in_window, _roi_start = (
                session.build_time_series_velocity_window(t_impact=t_impact_locked)
            )

    # 报告已生成完毕，主动清理这份会话（连同内存中持有的击球关键帧画面），
    # 严格遵守"不长期持久化保存任何视频帧"的科技伦理红线，同时避免内存持续累积。
    SESSIONS.pop(payload.session_id, None)

    # 【疲劳熔断】将本趟确定性打分写入课堂时序；命中 ANKLE_FATIGUE 等时缓存供看板轮询
    fatigue_warning = None
    if isinstance(score_detail, dict):
        try:
            fatigue_warning = _ingest_web_fatigue_attempt(
                payload.student_number or "",
                score_detail,
            )
        except Exception as fatigue_exc:  # noqa: BLE001
            safe_print(f"【api_server】疲劳熔断写入失败（不影响报告）：{fatigue_exc}")

    return {
        "score": final_score,
        "totalAttempts": total_attempts,
        "painPoint": ai_result["painPoint"],
        "prescription": ai_result["prescription"],
        "fullText": full_text,
        "generatedAt": generated_at,
        "hitStats": hit_stats,
        "impactFrameImage": impact_frame_image,
        "avgKneeAngle": avg_knee_angle,
        "t_impact": t_impact_locked,
        "tImpact": t_impact_locked,
        "frame_count": frame_count_out,
        "frameCount": frame_count_out,
        "angular_velocities": angular_velocities_out,
        "angularVelocities": angular_velocities_out,
        # Sprint 1：鞭打发力窗口 [t_impact±30] 角速度时序 + 触球点窗口内索引
        "time_series_velocity": time_series_velocity,
        "timeSeriesVelocity": time_series_velocity,
        "impact_index_in_window": impact_index_in_window,
        "impactIndexInWindow": impact_index_in_window,
        "task_status": TASK_STATUS_COMPLETED,
        "scoreDetail": score_detail,
        "scoringEngine": "DeterministicScorer_V2.5" if deterministic_score is not None else "llm_fallback",
        # Sprint 1：支撑脚 / 摆腿时空热力图（纯 PNG base64，前端拼 data URI）
        "heatmap_base64": heatmap_base64,
        "heatmapBase64": heatmap_base64,
        "spatial_trajectory": spatial_trajectory,
        "spatialTrajectory": spatial_trajectory,
        "fatigue_warning": fatigue_warning,
        "fatigueWarning": fatigue_warning,
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
    # Sprint 1：支撑脚 / 摆腿时空热力图 PNG base64（可带或不带 data URI 前缀）
    heatmapBase64: Optional[str] = None
    heatmap_base64: Optional[str] = None
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
    # 【SDT 成就印章】前端原样转发 generate_report 的 scoreDetail，供落盘时
    # 抽出脚踝刚性方差 / 支撑脚横纵偏差 / 五维雷达，供周成就引擎消费。
    scoreDetail: Optional[dict] = None
    score_detail: Optional[dict] = None


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


# --------------------------------------------------------------------------
# SDT 游戏化成就印章引擎（拒绝总分排名，只发多维度独立王者）
# --------------------------------------------------------------------------

_SUPPORT_LATERAL_IDEAL_CENTER_CM = 17.5
_RADAR_DIM_KEYS = (
    "support_stability",
    "backswing_folding",
    "ankle_rigidity",
    "whipping_velocity",
    "approach_rhythm",
)

_ACHIEVEMENT_PRAISE = {
    "iron_ankle": "踝关节稳如泰山，力量毫无流失！",
    "stable_chassis": "支撑脚扎根大地，底盘稳如磐石！",
    "fastest_progress": "本周飞跃成长，高反应者实至名归！",
}


def _parse_record_datetime(record: dict) -> Optional[datetime]:
    """从 testDate / timestamp 解析记录时间；失败返回 None。"""
    raw = (record.get("timestamp") or "").strip()
    if not raw:
        date_only = (record.get("testDate") or "").strip()
        if len(date_only) >= 10:
            raw = date_only[:10] + " 12:00:00"
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw[:19] if len(raw) >= 19 else raw[:10], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00").replace(" ", "T")[:19])
    except ValueError:
        return None


def _week_window(now: Optional[datetime] = None) -> tuple[datetime, datetime, datetime, datetime]:
    """返回 (本周一起点, 本周结束, 上周一起点, 上周结束)，周一 00:00 为界。"""
    anchor = now or datetime.now()
    this_monday = anchor.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
        days=anchor.weekday()
    )
    next_monday = this_monday + timedelta(days=7)
    last_monday = this_monday - timedelta(days=7)
    return this_monday, next_monday, last_monday, this_monday


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    return None


def _nested_metric_value(container: Any, *keys: str) -> Optional[float]:
    """从扁平字段 / indicators 嵌套 {value|variance} 中取第一个可用标量。"""
    if not isinstance(container, dict):
        return None
    for key in keys:
        entry = container.get(key)
        if isinstance(entry, dict):
            for sub in ("variance", "value"):
                num = _safe_float(entry.get(sub))
                if num is not None:
                    return num
        else:
            num = _safe_float(entry)
            if num is not None:
                return num
    return None


def _extract_ankle_rigidity_variance(record: dict) -> Optional[float]:
    """脚踝刚性方差（越小越锁踝稳固）。优先真实字段，缺失时用综合分启发式。"""
    direct = _nested_metric_value(
        record,
        "ankle_rigidity",
        "ankle_rigidity_variance",
        "ankleRigidity",
        "ankleRigidityVariance",
    )
    if direct is not None:
        return max(0.0, direct)

    metrics = record.get("instepKickMetrics") or record.get("metrics") or {}
    if isinstance(metrics, dict):
        from_metrics = _nested_metric_value(
            metrics,
            "ankle_rigidity",
            "ankle_rigidity_variance",
            "ankle_variance",
        )
        if from_metrics is not None:
            return max(0.0, from_metrics)

    detail = record.get("scoreDetail") or record.get("score_detail") or {}
    if isinstance(detail, dict):
        indicators = detail.get("indicators") if isinstance(detail.get("indicators"), dict) else detail
        from_detail = _nested_metric_value(
            indicators,
            "ankle_rigidity",
            "ankle_rigidity_variance",
        )
        if from_detail is not None:
            return max(0.0, from_detail)

    score = _safe_float(record.get("score"))
    if score is None:
        return None
    # 启发式：高分 → 低方差（与 ANKLE_VARIANCE_* 量级对齐）
    return round(max(0.0, (100.0 - score) / 12.0), 3)


def _support_lateral_deviation(lateral_cm: float) -> float:
    """相对 [15, 20] cm 理想带的横向偏差（带内为 0）。"""
    low, high = 15.0, 20.0
    if low <= lateral_cm <= high:
        return 0.0
    if lateral_cm < low:
        return low - lateral_cm
    return lateral_cm - high


def _extract_support_chassis_deviation(record: dict) -> Optional[float]:
    """支撑脚横纵向位移偏差综合值 = 横向偏离理想带 + |纵向 AP 偏移|。"""
    metrics = record.get("instepKickMetrics") or record.get("metrics") or {}
    detail = record.get("scoreDetail") or record.get("score_detail") or {}
    indicators: dict = {}
    if isinstance(detail, dict) and isinstance(detail.get("indicators"), dict):
        indicators = detail["indicators"]

    lateral = (
        _nested_metric_value(record, "support_lateral_dist_cm", "supportLateralDistCm", "distance_cm")
        or _nested_metric_value(
            metrics if isinstance(metrics, dict) else {},
            "support_lateral_dist_cm",
            "distance_cm",
        )
        or _nested_metric_value(indicators, "distance_cm", "support_lateral_dist_cm")
    )
    ap = (
        _nested_metric_value(record, "support_ap_offset_cm", "supportApOffsetCm")
        or _nested_metric_value(
            metrics if isinstance(metrics, dict) else {},
            "support_ap_offset_cm",
        )
        or _nested_metric_value(indicators, "support_ap_offset_cm")
    )

    if lateral is None:
        foot = _safe_float(record.get("supportFootDistance"))
        if foot is not None:
            lateral = foot
    if lateral is None:
        score = _safe_float(record.get("score"))
        if score is None:
            return None
        lateral = _SUPPORT_LATERAL_IDEAL_CENTER_CM + (100.0 - score) * 0.15
    if ap is None:
        score = _safe_float(record.get("score"))
        # 启发式：低分时略增大前后偏差
        ap = 0.0 if score is None else max(0.0, (70.0 - score) * 0.2)

    return round(_support_lateral_deviation(float(lateral)) + abs(float(ap)), 3)


def _sum_radar_scores(radar: Any) -> Optional[float]:
    if not isinstance(radar, dict):
        return None
    vals: list[float] = []
    for key in _RADAR_DIM_KEYS:
        num = _safe_float(radar.get(key))
        if num is None:
            # 兼容旧版 *_score 别名
            num = _safe_float(radar.get(f"{key}_score"))
        if num is not None:
            vals.append(num)
    # 也兼容 quantified5dScores 的简写键
    if len(vals) < 3:
        alias_map = {
            "support_stability": ("support_stability_score", "support"),
            "backswing_folding": ("backswing_folding_score", "folding"),
            "ankle_rigidity": ("ankle_rigidity_score",),
            "whipping_velocity": ("whipping_velocity_score", "whipping"),
            "approach_rhythm": ("approach_rhythm_score", "approach"),
        }
        vals = []
        for key, aliases in alias_map.items():
            num = _safe_float(radar.get(key))
            if num is None:
                for a in aliases:
                    num = _safe_float(radar.get(a))
                    if num is not None:
                        break
            if num is not None:
                vals.append(num)
    if not vals:
        return None
    return round(sum(vals), 2)


def _extract_five_dim_total(record: dict) -> Optional[float]:
    """五维雷达总分（每维满分 20，合计满分 100）；缺失时回退综合分 score。"""
    for key in ("quantified5dScores", "radar_scores", "radarScores"):
        total = _sum_radar_scores(record.get(key))
        if total is not None:
            return total
    detail = record.get("scoreDetail") or record.get("score_detail") or {}
    if isinstance(detail, dict):
        total = _sum_radar_scores(detail.get("radar_scores"))
        if total is not None:
            return total
    return _safe_float(record.get("score"))


def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _pack_winner(
    badge_id: str,
    title: str,
    emoji: str,
    student_id: Optional[str],
    value: Optional[float],
    value_label: str,
    unit: str = "",
    attempt_count: int = 0,
) -> dict:
    return {
        "id": badge_id,
        "title": title,
        "emoji": emoji,
        "anonymousId": student_id,
        "studentId": student_id,
        "value": None if value is None else round(float(value), 3),
        "valueLabel": value_label,
        "unit": unit,
        "attemptCount": attempt_count,
        "praise": _ACHIEVEMENT_PRAISE.get(badge_id, "太棒了，继续探索身体的超级力量！"),
        "hasWinner": student_id is not None and value is not None,
    }


def calculate_achievements(
    records: Optional[list] = None,
    *,
    school: str = "",
    class_group: str = "",
    now: Optional[datetime] = None,
) -> dict:
    """基于 SDT 的多维度成就印章计算引擎（无总分排名）。

    三个独立王者：
      a) 钢铁锁踝王 —— 本周脚踝刚性方差均值最小（趋近 0）
      b) 最稳底盘奖 —— 本周支撑脚横纵向位移偏差综合最小
      c) 最快进步奖 —— 本周五维总分均值 − 上周五维总分均值，正向差值最大
    """
    source = records if isinstance(records, list) else _load_global_records()
    school_q = (school or "").strip()
    class_q = (class_group or "").strip()
    this_start, this_end, last_start, last_end = _week_window(now)

    # studentId -> {"this": [...], "last": [...]} 指标包
    buckets: dict[str, dict[str, list]] = {}

    for record in source:
        if not isinstance(record, dict):
            continue
        sid = str(record.get("studentId") or record.get("anonymous_id") or "").strip()
        if not sid:
            continue
        if school_q and school_q not in ("all", "全部") and (record.get("school") or "") != school_q:
            continue
        if class_q and class_q not in ("all", "全部") and (record.get("classGroup") or "") != class_q:
            continue

        ts = _parse_record_datetime(record)
        if ts is None:
            continue

        ankle = _extract_ankle_rigidity_variance(record)
        chassis = _extract_support_chassis_deviation(record)
        five_dim = _extract_five_dim_total(record)
        row = {
            "ankle": ankle,
            "chassis": chassis,
            "five_dim": five_dim,
            "timestamp": ts.isoformat(sep=" ", timespec="seconds"),
        }

        if this_start <= ts < this_end:
            buckets.setdefault(sid, {"this": [], "last": []})["this"].append(row)
        elif last_start <= ts < last_end:
            buckets.setdefault(sid, {"this": [], "last": []})["last"].append(row)

    iron_best: tuple[Optional[str], Optional[float], int] = (None, None, 0)
    chassis_best: tuple[Optional[str], Optional[float], int] = (None, None, 0)
    progress_best: tuple[Optional[str], Optional[float], int] = (None, None, 0)

    for sid, pack in buckets.items():
        this_rows = pack.get("this") or []
        last_rows = pack.get("last") or []

        ankle_vals = [r["ankle"] for r in this_rows if isinstance(r.get("ankle"), (int, float))]
        ankle_mean = _mean(ankle_vals)
        if ankle_mean is not None:
            if iron_best[1] is None or ankle_mean < iron_best[1] or (
                ankle_mean == iron_best[1] and len(ankle_vals) > iron_best[2]
            ):
                iron_best = (sid, ankle_mean, len(ankle_vals))

        chassis_vals = [
            r["chassis"] for r in this_rows if isinstance(r.get("chassis"), (int, float))
        ]
        chassis_mean = _mean(chassis_vals)
        if chassis_mean is not None:
            if chassis_best[1] is None or chassis_mean < chassis_best[1] or (
                chassis_mean == chassis_best[1] and len(chassis_vals) > chassis_best[2]
            ):
                chassis_best = (sid, chassis_mean, len(chassis_vals))

        this_five = [
            r["five_dim"] for r in this_rows if isinstance(r.get("five_dim"), (int, float))
        ]
        last_five = [
            r["five_dim"] for r in last_rows if isinstance(r.get("five_dim"), (int, float))
        ]
        this_avg = _mean(this_five)
        last_avg = _mean(last_five)
        if this_avg is not None and last_avg is not None:
            delta = this_avg - last_avg
            if delta > 0 and (
                progress_best[1] is None
                or delta > progress_best[1]
                or (delta == progress_best[1] and len(this_five) > progress_best[2])
            ):
                progress_best = (sid, delta, len(this_five))

    badges = [
        _pack_winner(
            "iron_ankle",
            "钢铁锁踝王",
            "🛡️",
            iron_best[0],
            iron_best[1],
            "脚踝刚性方差",
            unit="σ²",
            attempt_count=iron_best[2],
        ),
        _pack_winner(
            "stable_chassis",
            "最稳底盘奖",
            "🌳",
            chassis_best[0],
            chassis_best[1],
            "支撑脚横纵偏差",
            unit="cm",
            attempt_count=chassis_best[2],
        ),
        _pack_winner(
            "fastest_progress",
            "最快进步奖",
            "🚀",
            progress_best[0],
            progress_best[1],
            "五维均分周环比",
            unit="Δ",
            attempt_count=progress_best[2],
        ),
    ]

    return {
        "success": True,
        "weekStart": this_start.strftime("%Y-%m-%d"),
        "weekEnd": (this_end - timedelta(seconds=1)).strftime("%Y-%m-%d"),
        "lastWeekStart": last_start.strftime("%Y-%m-%d"),
        "lastWeekEnd": (last_end - timedelta(seconds=1)).strftime("%Y-%m-%d"),
        "subjectCount": len(buckets),
        "badges": badges,
        "achievements": badges,  # 别名，便于前端消费
    }


def _metrics_snapshot_from_score_detail(score_detail: Optional[dict]) -> dict:
    """从 scoreDetail 抽出成就引擎与看板可复用的轻量指标快照（无大图）。"""
    if not isinstance(score_detail, dict):
        return {}
    flat = flatten_eight_metrics(score_detail)
    indicators = score_detail.get("indicators") if isinstance(score_detail.get("indicators"), dict) else {}
    ankle = flat.get("ankle_rigidity")
    if ankle is None and isinstance(indicators.get("ankle_rigidity"), dict):
        ankle = _safe_float(indicators["ankle_rigidity"].get("variance"))
        if ankle is None:
            ankle = _safe_float(indicators["ankle_rigidity"].get("value"))

    lateral = _nested_metric_value(indicators, "distance_cm", "support_lateral_dist_cm")
    if lateral is None:
        lateral = _nested_metric_value(score_detail, "support_lateral_dist_cm", "distance_cm")
    ap = _nested_metric_value(score_detail, "support_ap_offset_cm")
    if ap is None and isinstance(score_detail.get("spatial_trajectory"), dict):
        # 若有相对坐标点，不在此强解；保持 None
        pass

    radar = score_detail.get("radar_scores")
    snapshot: dict[str, Any] = {}
    if ankle is not None:
        snapshot["ankle_rigidity"] = round(float(ankle), 3)
        snapshot["ankle_rigidity_variance"] = round(float(ankle), 3)
    if lateral is not None:
        snapshot["support_lateral_dist_cm"] = round(float(lateral), 2)
        snapshot["supportFootDistance"] = round(float(lateral), 2)
    if ap is not None:
        snapshot["support_ap_offset_cm"] = round(float(ap), 2)
    # V3.1：支撑脚相对坐标写入归档，供 Heatmap_Dispersion_Index 结算
    spatial = score_detail.get("spatial_trajectory") or score_detail.get("spatialTrajectory")
    if isinstance(spatial, dict):
        dx = _safe_float(spatial.get("dx_support"))
        dy = _safe_float(spatial.get("dy_support"))
        if dx is None or dy is None:
            rel = spatial.get("support_rel")
            if isinstance(rel, (list, tuple)) and len(rel) >= 2:
                dx = _safe_float(rel[0]) if dx is None else dx
                dy = _safe_float(rel[1]) if dy is None else dy
        if dx is not None and dy is not None:
            snapshot["dx_support"] = round(float(dx), 2)
            snapshot["dy_support"] = round(float(dy), 2)
            snapshot["support_rel"] = [round(float(dx), 2), round(float(dy), 2)]
            snapshot["spatial_trajectory"] = {
                "dx_support": round(float(dx), 2),
                "dy_support": round(float(dy), 2),
                "support_rel": [round(float(dx), 2), round(float(dy), 2)],
            }
    # 脚踝锁紧状态数字编码（GREEN=3 / YELLOW=2 / RED=1）供宽表直接读取
    ankle_entry = indicators.get("ankle_rigidity") if isinstance(indicators, dict) else None
    if isinstance(ankle_entry, dict) and ankle_entry.get("status") is not None:
        snapshot["ankle_lock_status"] = ankle_entry.get("status")
    if isinstance(radar, dict):
        snapshot["quantified5dScores"] = radar
        snapshot["radar_scores"] = radar
    # 保留精简 indicators 数值，避免把 heatmap_base64 等巨字段写入 JSON DB
    slim_indicators: dict[str, Any] = {}
    for key, entry in (indicators or {}).items():
        if isinstance(entry, dict):
            slim: dict[str, Any] = {}
            for sub in ("value", "variance", "status", "penalty"):
                if sub in entry:
                    slim[sub] = entry[sub]
            if slim:
                slim_indicators[key] = slim
    if slim_indicators or radar:
        snapshot["scoreDetail"] = {
            "indicators": slim_indicators,
            "radar_scores": radar if isinstance(radar, dict) else None,
            "t_impact": score_detail.get("t_impact"),
        }
    return snapshot


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
            "heatmapBase64": payload.heatmapBase64 or payload.heatmap_base64,
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
        # 【SDT】合并 scoreDetail 轻量快照（脚踝刚性 / 支撑横纵 / 五维雷达）
        detail_payload = payload.scoreDetail or payload.score_detail
        snapshot = _metrics_snapshot_from_score_detail(
            detail_payload if isinstance(detail_payload, dict) else None
        )
        if snapshot.get("supportFootDistance") is not None:
            record["supportFootDistance"] = snapshot["supportFootDistance"]
        for key, value in snapshot.items():
            if key == "supportFootDistance":
                continue
            record[key] = value
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


@app.get("/api/achievements/weekly")
def get_weekly_achievements(school: str = "", classGroup: str = ""):
    """SDT 游戏化周成就印章：返回三个维度的独立王者（无总分排名）。

    可选 query：school / classGroup —— 与教练端筛选器对齐；空则全库遍历。
    """
    try:
        return calculate_achievements(school=school, class_group=classGroup)
    except Exception as exc:  # noqa: BLE001
        safe_print(f"【api_server】计算周成就失败：{exc}")
        return {
            "success": False,
            "message": f"计算周成就失败：{exc}",
            "badges": [],
            "achievements": [],
        }


# --------------------------------------------------------------------------
# 疲劳熔断报警 —— 供教练端 / 延时组「纵向双轴进化图谱」轮询
# --------------------------------------------------------------------------


def _eval_fatigue_from_history(history: list) -> Optional[dict]:
    """基线 vs 近期对比；命中 ANKLE_FATIGUE / KNEE_STIFFNESS 时返回报警字典。"""
    if len(history) < MIN_ATTEMPTS_FOR_MONITOR:
        return None
    baseline = history[:BASELINE_WINDOW]
    recent = history[-RECENT_WINDOW:]
    warning = FatigueMonitor._eval_ankle_fatigue(baseline, recent)
    if warning is None:
        warning = FatigueMonitor._eval_knee_stiffness(baseline, recent)
    return warning


def _ingest_web_fatigue_attempt(
    student_id: str,
    score_detail: Optional[dict],
) -> Optional[dict]:
    """将一次确定性打分写入疲劳时序，并在命中熔断时缓存最新报警。"""
    global _global_latest_fatigue
    sid = (student_id or "").strip() or "_anonymous"
    if not isinstance(score_detail, dict):
        return None
    flat = flatten_eight_metrics(score_detail)
    with _fatigue_history_lock:
        rows = _fatigue_attempts.setdefault(sid, [])
        row = dict(flat)
        row["attempt_index"] = len(rows) + 1
        row["student_id"] = sid
        rows.append(row)
        warning = _eval_fatigue_from_history(rows)
        if isinstance(warning, dict) and warning.get("is_fatigue"):
            payload = {
                **warning,
                "student_id": sid,
                "studentId": sid,
                "isFatigue": True,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "updatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
                "attempt_count": len(rows),
                "message": warning.get("message")
                or FATIGUE_MESSAGES.get(str(warning.get("reason") or ""), "疲劳熔断"),
            }
            _latest_fatigue_alerts[sid] = payload
            _global_latest_fatigue = payload
            safe_print(
                f"【api_server】⚠️ 疲劳熔断 [{payload.get('reason')}] student={sid} "
                f"attempts={len(rows)}",
                flush=True,
            )
            return payload
    return None


@app.get("/api/fatigue_alert")
def get_fatigue_alert(student_id: str = ""):
    """轮询最新疲劳熔断信号。

    - 指定 student_id：返回该被试最近一次熔断（无则 is_fatigue=false）
    - 不传：返回全局最近一次熔断（供教练端总览）
    """
    sid = (student_id or "").strip()
    with _fatigue_history_lock:
        if sid:
            alert = _latest_fatigue_alerts.get(sid)
            history_len = len(_fatigue_attempts.get(sid, []))
        else:
            alert = _global_latest_fatigue
            history_len = sum(len(v) for v in _fatigue_attempts.values())

    if isinstance(alert, dict) and alert.get("is_fatigue"):
        return {**alert, "success": True, "history_len": history_len}
    return {
        "success": True,
        "is_fatigue": False,
        "isFatigue": False,
        "reason": None,
        "message": None,
        "student_id": sid or None,
        "history_len": history_len,
    }


@app.post("/api/fatigue_alert/reset")
def reset_fatigue_alert(student_id: str = ""):
    """换人 / 新开轮次时清空疲劳时序（可选）。"""
    global _global_latest_fatigue
    sid = (student_id or "").strip()
    with _fatigue_history_lock:
        if sid:
            _fatigue_attempts.pop(sid, None)
            _latest_fatigue_alerts.pop(sid, None)
            if _global_latest_fatigue and (
                _global_latest_fatigue.get("student_id") == sid
                or _global_latest_fatigue.get("studentId") == sid
            ):
                _global_latest_fatigue = None
        else:
            _fatigue_attempts.clear()
            _latest_fatigue_alerts.clear()
            _global_latest_fatigue = None
    return {"success": True, "cleared": sid or "all"}


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
    """【V3.1】一键导出全数字化 SPSS 标准宽表（JSON 元信息 + 落盘）。

    优先走 AcademicDataExporter 宽表主路径；同时保留长表旁路落盘供 ANOVA。
    前端若需浏览器直接下载，请改用 GET ``/api/export/spss_matrix``。
    """
    try:
        exporter = academic_exporter.AcademicDataExporter.from_db()
        result = exporter.export_spss_matrix_file()
    except Exception as exc:  # noqa: BLE001
        safe_print(f"【api_server】导出 V3.1 科研宽表失败：{exc}")
        # 回退：旧成长表，避免教练端完全无法导出
        records = _load_global_records()
        try:
            result = academic_exporter.export_academic_matrix(records)
        except Exception as long_exc:  # noqa: BLE001
            return {"success": False, "message": f"导出学术统计矩阵失败：{long_exc}"}

    if not result.get("success"):
        return result

    return {
        "success": True,
        "message": (
            f"✅ V3.1 全数字化科研宽表已生成！文件："
            f"{result.get('filename', academic_exporter.RESEARCH_MATRIX_V3_FILENAME)}，"
            f"已存入：{result['path']}，可直接导入 SPSS / Mplus 跑 MSEM！"
        ),
        "path": result["path"],
        "filename": result["filename"],
        "rowCount": result["rowCount"],
        "columnCount": result.get("columnCount"),
        "studentCount": result["studentCount"],
        "downloadUrl": "/api/export/spss_matrix",
    }


@app.get("/api/export/spss_matrix")
def export_spss_wide_matrix():
    """【V3.1 Cluster-RCT · MSEM】导出全数字化 SPSS 标准宽表 CSV（浏览器直接下载）。

    数据源优先级：``cluster_rct.db`` → 桥接 ``global_training_db.json``。
    主键 ``anonymous_id`` 一行一人；T0–T4 前缀展平；组别/疲劳/锁踝全数字编码；
    含 ``Heatmap_Dispersion_Index`` / ``Ankle_Rigidity_Score`` 衍生中介；
    表尾 ``Class_Dummy_1``…``Class_Dummy_5`` 群聚固定效应哑变量。

    固定文件名：``AI_Football_Research_Matrix_V3.csv``。
    """
    filename = academic_exporter.RESEARCH_MATRIX_V3_FILENAME
    try:
        exporter = academic_exporter.AcademicDataExporter.from_db()
        wide_df = exporter.generate_wide_format_matrix()
        # 同步落盘一份到 academic_data_export/，便于教练本地归档
        exporter.export_spss_matrix_file(filename=filename)
    except Exception as exc:  # noqa: BLE001 - 导出失败返回结构化错误，避免裸 500
        safe_print(f"【api_server】导出 V3.1 科研宽表失败：{exc}")
        return {"success": False, "message": f"导出 V3.1 科研宽表失败：{exc}"}

    csv_bytes = exporter.to_csv_bytes(wide_df)

    safe_print(
        f"【api_server】V3.1 科研宽表已生成：{filename}"
        f"（{len(wide_df)} 行 × {len(wide_df.columns)} 列）",
        flush=True,
    )
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Export-Row-Count": str(len(wide_df)),
            "X-Export-Column-Count": str(len(wide_df.columns)),
        },
    )


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
# 第五步再再再再再半：核心接口八 —— 教练端 / 科研控制台
# 「干预进度与剂量异常监控」+「极端个案目的性抽样」
#
#         全部聚合 / 斜率 / 百分位逻辑封装在
#         research_dashboard_service.ResearchDashboardService，本处只做
#         查询参数解析与结果透传。数据优先读 research_shot_logs.json，
#         不存在时自动桥接 global_training_db.json。
# --------------------------------------------------------------------------


@app.get("/api/coach/progress_monitor")
def coach_progress_monitor(
    timepoint: Optional[str] = None,
    cluster_id: Optional[str] = None,
    standard_dose: int = STANDARD_SHOT_DOSE,
):
    """【干预进度与缺失值监控】

    分组聚合当前（或指定 T 节点）所有被试的射门完成次数，计算组内均值，
    并返回射门次数偏离「标准剂量 ±20%」的剂量异常被试名单，供教练课上
    及时人工干预。

    Query 参数：
        timepoint     —— 可选，T0/T1/T2/T3/T4；缺省则汇总全部节点
        cluster_id    —— 可选，行政班集群过滤（如 Class_1）
        standard_dose —— 可选，标准射门剂量，默认 15
    """
    try:
        service = research_dashboard_service.get_dashboard_service(reload=True)
        return service.get_progress_monitor(
            timepoint=timepoint,
            cluster_id=cluster_id,
            standard_dose=standard_dose,
        )
    except Exception as exc:  # noqa: BLE001
        safe_print(f"【api_server】progress_monitor 失败：{exc}")
        return {"success": False, "message": f"干预进度监控失败：{exc}", "dose_anomalies": []}


@app.get("/api/coach/extreme_cases")
def coach_extreme_cases(
    cluster_id: Optional[str] = None,
    baseline: str = "T1",
    followup: str = "T2",
    percentile: float = 0.20,
):
    """【极端个案捕捉 · Purposive Sampling Extractor】

    对比 baseline（默认 T1）与 followup（默认 T2）阶段被试在 8 大生物力学
    综合得分上的变化斜率（Slope），自动识别：
      - 高反应者 (High Responders)：斜率最高的前 20%
      - 低反应者 (Low Responders)：得分一直处于低位且改善斜率最平缓的后 20%

    返回名单可作为后续现象学深度访谈的客观抽样基础。
    """
    try:
        service = research_dashboard_service.get_dashboard_service(reload=True)
        return service.extract_extreme_cases(
            cluster_id=cluster_id,
            baseline=baseline,
            followup=followup,
            percentile=percentile,
        )
    except Exception as exc:  # noqa: BLE001
        safe_print(f"【api_server】extreme_cases 失败：{exc}")
        return {
            "success": False,
            "message": f"极端个案捕捉失败：{exc}",
            "high_responders": [],
            "low_responders": [],
        }


# --------------------------------------------------------------------------
# 【V3.1 Sprint 3】教练端手绘电烙铁批注截图归档
# --------------------------------------------------------------------------

TELESTRATION_DIR = os.path.join(SCRIPT_DIR, "telestration_annotations")
os.makedirs(TELESTRATION_DIR, exist_ok=True)


class SaveTelestrationImageRequest(BaseModel):
    """前端合并「视频定格帧 + Canvas 涂鸦」后的 JPEG/PNG Base64。"""

    imageBase64: str
    attemptId: Optional[str] = None
    studentNumber: Optional[str] = None
    studentId: Optional[str] = None


def _decode_data_url_bytes(data_url: str) -> bytes:
    """支持 data:image/...;base64,XXX 或纯 base64。"""
    raw = (data_url or "").strip()
    if not raw:
        raise ValueError("imageBase64 为空")
    if "," in raw and raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    return base64.b64decode(raw)


@app.post("/api/save_telestration_image")
def save_telestration_image(payload: SaveTelestrationImageRequest):
    """
    接收教练手绘批注合成图，写入 telestration_annotations/，
    若 attemptId 命中 global_training_db.json 则回填 telestrationImagePath 字段，
    供后续 Word 诊断处方附加。
    """
    try:
        image_bytes = _decode_data_url_bytes(payload.imageBase64)
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "message": f"Base64 解码失败：{exc}"}

    if not image_bytes:
        return {"success": False, "message": "图像数据为空"}

    student_key = (payload.studentNumber or payload.studentId or "unknown").strip() or "unknown"
    # Windows 非法文件名字符清理
    for ch in '<>:"/\\|?*':
        student_key = student_key.replace(ch, "_")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    attempt_part = (payload.attemptId or uuid.uuid4().hex[:8]).strip()
    for ch in '<>:"/\\|?*':
        attempt_part = attempt_part.replace(ch, "_")

    # 根据 data URI 头或内容简单判定扩展名
    lower = (payload.imageBase64 or "")[:64].lower()
    ext = ".png" if "image/png" in lower else ".jpg"
    filename = f"telestration_{student_key}_{attempt_part}_{stamp}{ext}"
    abs_path = os.path.join(TELESTRATION_DIR, filename)

    try:
        with open(abs_path, "wb") as fh:
            fh.write(image_bytes)
    except Exception as exc:  # noqa: BLE001
        safe_print(f"【api_server】写入手绘批注失败：{exc}")
        return {"success": False, "message": f"写盘失败：{exc}"}

    # 可选：回填全局训练库记录，便于报告附加
    linked = False
    if payload.attemptId:
        try:
            if os.path.isfile(GLOBAL_DB_PATH):
                with open(GLOBAL_DB_PATH, "r", encoding="utf-8") as f:
                    records = json.load(f)
                if isinstance(records, list):
                    for row in records:
                        if isinstance(row, dict) and str(row.get("id", "")) == str(payload.attemptId):
                            row["telestrationImagePath"] = abs_path
                            row["telestrationImageFilename"] = filename
                            linked = True
                            break
                    if linked:
                        with open(GLOBAL_DB_PATH, "w", encoding="utf-8") as f:
                            json.dump(records, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            safe_print(f"【api_server】回填手绘批注到全局库失败（文件已保存）：{exc}")

    msg = f"手绘批注已归档：{abs_path}"
    if linked:
        msg += "（已关联 Attempt 诊断记录）"

    return {
        "success": True,
        "message": msg,
        "path": abs_path,
        "filename": filename,
        "linked": linked,
    }


# --------------------------------------------------------------------------
# 第六步：程序入口 —— 支持直接 `python api_server.py` 启动
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
