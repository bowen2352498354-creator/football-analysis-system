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
    POST /api/lock_baseline  ：实验防干扰——锁定课堂环境基线（class_id /
                                camera_height_cm / calibrator_status），生成全局
                                session_id；后续每次 score_detail / SPSS 导出均
                                打上该水印与 is_baseline_trusted。
    GET  /api/baseline_status：查询当前基线是否已锁定及环境参数。
    GET  /api/fatigue_alert  ：课堂疲劳熔断轮询（ANKLE_FATIGUE / KNEE_STIFFNESS）。
                                generate_report 写入时序后命中规则即缓存；教练端
                                「纵向双轴进化图谱」每 2.5s 拉取并渲染熔断闪烁卡。
    GET  /api/achievements/weekly ：SDT 游戏化周成就印章（钢铁锁踝王 / 最稳底盘奖 /
                                最快进步奖），拒绝总分排名，返回匿名学员编号与指标。
    GET  /api/progress/history ：个人纵向进步图谱（按测试日聚合：date / phase /
                                score / 正负向诊断高亮），供教练端 Catapult 风格
                                ECharts 趋势图消费。
    GET  /api/coach/records    ：学员成绩列表 + radar_average（所选日期内五维均值）。
    GET  /api/analytics/compare_cohorts ：班级/实验组三维对比（日趋势均分+方差、
                                五维雷达均值、ERR_* 错误分布率），供教练看板科研模块。
    GET  /api/review/student_summary ：B 组课后复盘聚合（极值双关键帧
                                comparison_frames：best vs improve）。
    DELETE|POST /api/records/batch ：批量软删除归档记录（is_deleted=True，禁止物理删除）。

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
import traceback
import uuid
from contextlib import asynccontextmanager
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
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
from session_baseline import (
    SESSION_METADATA_STORE,
    stamp_baseline_watermark,
)

# 【第 6 点重构】射门分析的全部计算逻辑已迁移到 shot_analysis_service.ShotAnalysisPipeline。
# 本文件的 AnalysisSession 退化为「传输层 / 生命周期适配器」：只负责队列、WebSocket 边界、
# 线程管理与 task_status 状态机，不再持有任何逐帧计算状态。
# safe_print / _is_empty_or_failed_frame 在下方重新导出，保证既有 `from api_server import ...`
# 的调用方零改动继续可用。
from shot_analysis_service import (
    ShotAnalysisPipeline,
    safe_print,
    _is_empty_or_failed_frame,
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

# 【Sprint 5】摄像头连续空帧 / cap.read() 失败阈值：达到后推送 camera_lost 并自愈重开
CAMERA_READ_FAIL_LIMIT = 50
CAMERA_REOPEN_SLEEP_SEC = 0.45


def _is_empty_or_failed_frame(ret: bool, frame: Any) -> bool:
    """判定 cap.read() 是否得到可用画面（空帧 / None / 零尺寸均视为失败）。"""
    if not ret or frame is None:
        return True
    try:
        import numpy as _np

        if not isinstance(frame, _np.ndarray) or frame.size == 0:
            return True
    except Exception:  # noqa: BLE001
        return True
    return False


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
    """兼容转发：大小腿夹角矢量标注实现位于 pose_tracker。"""
    return pt.draw_biomechanics_annotation(frame, metrics)


def resolve_leg_annotation_target(
    score_detail: Optional[dict],
    *,
    t_impact: Optional[int] = None,
) -> tuple[int, str, str]:
    """兼容转发：标注目标帧选择实现位于 pose_tracker。"""
    return pt.resolve_leg_annotation_target(score_detail, t_impact=t_impact)


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

        # 【V2.5 竞态锁】PROCESSING → COMPLETED；generate_report 必须等 COMPLETED
        self.task_status: str = TASK_STATUS_PROCESSING
        self._completed_event = threading.Event()

        # 【重构】全部计算逻辑（姿态推理 / 脱敏 / 角速度 / 锁帧 / 打分载荷）已迁移到
        # shot_analysis_service.ShotAnalysisPipeline。本类只负责传输与生命周期：
        # 队列背压、WebSocket 边界、线程管理、任务状态机。
        # 管线实例在 start() 时构造（run 前所有轨迹属性访问都走下方转发代理）。
        self.pipeline: Optional[ShotAnalysisPipeline] = None

    def start(self):
        self.task_status = TASK_STATUS_PROCESSING
        self._completed_event.clear()
        # 管线与宿主共享 records / 锁 / stop_event，并通过回调回写传输层与状态机：
        #   push_fn        -> 本类的队列背压策略（file 不丢帧 / webcam 丢最旧显示帧）
        #   on_completed   -> mark_completed()，保证 stopped 之前状态已是 COMPLETED
        #   status_provider -> 让 stopped 载荷读到宿主真实 task_status
        self.pipeline = ShotAnalysisPipeline(
            session_id=self.session_id,
            source=self.source,
            video_path=self.video_path,
            camera_index=self.camera_index,
            push_fn=self._push_frame_payload,
            on_completed=self.mark_completed,
            records=self.records,
            records_lock=self._records_lock,
            stop_event=self.stop_event,
            status_provider=lambda: self.task_status,
        )
        self.thread = threading.Thread(target=self.pipeline.run, daemon=True)
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

    def _push_frame_payload(self, payload: dict):
        """把一份处理好的帧数据放进队列。

        【V2.5】录像分析（file）严禁丢帧：阻塞式 put，保证 frame_count 绝对相等。
        仅实时摄像头模式允许在背压时丢弃最旧显示帧。
        【Sprint 5】camera_lost / error / notice / stopped 等控制消息永不丢弃。
        """
        msg_type = payload.get("type")
        if (not self._drop_frames_on_backpressure) or msg_type in (
            "camera_lost",
            "error",
            "notice",
            "stopped",
        ):
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

    # ----------------------------------------------------------------------
    # 转发代理：/api/generate_report 等既有调用方无需任何改动即可继续工作。
    # 计算状态的唯一所有者是 ShotAnalysisPipeline；本类只做只读转发，
    # 并在管线尚未构造（start() 之前）时返回与旧实现一致的空初值。
    # ----------------------------------------------------------------------

    @property
    def _trajectory_angles(self) -> list:
        return self.pipeline._trajectory_angles if self.pipeline else []

    @property
    def _trajectory_omega(self) -> list:
        return self.pipeline._trajectory_omega if self.pipeline else []

    @property
    def _trajectory_ankle_px(self) -> list:
        return self.pipeline._trajectory_ankle_px if self.pipeline else []

    @property
    def _trajectory_pose_frames(self) -> list:
        return self.pipeline._trajectory_pose_frames if self.pipeline else []

    @property
    def sync_frame_count(self) -> int:
        return self.pipeline.sync_frame_count if self.pipeline else 0

    @property
    def t_impact(self) -> Optional[int]:
        return self.pipeline.t_impact if self.pipeline else None

    @property
    def impact_frame(self):
        """击球关键帧（已完成面部打码）。generate_report 会用重标定后的帧覆写。"""
        return self.pipeline.impact_frame if self.pipeline else None

    @impact_frame.setter
    def impact_frame(self, value) -> None:
        if self.pipeline is not None:
            self.pipeline.impact_frame = value

    @property
    def impact_metrics(self) -> Optional[dict]:
        return self.pipeline.impact_metrics if self.pipeline else None

    @impact_metrics.setter
    def impact_metrics(self, value: Optional[dict]) -> None:
        if self.pipeline is not None:
            self.pipeline.impact_metrics = value

    def build_scoring_payloads(self) -> tuple[dict, dict]:
        if self.pipeline is None:
            raise RuntimeError("分析管线尚未启动，无法构造打分载荷")
        return self.pipeline.build_scoring_payloads()

    def get_ball_outcome(self) -> dict:
        if self.pipeline is None:
            return {"ball_speed_kmh": None, "launch_angle_deg": None, "meta": {}}
        return self.pipeline.get_ball_outcome()

    def inject_ball_outcome_into_score_detail(self, score_detail=None) -> dict:
        if self.pipeline is None:
            return dict(score_detail or {})
        return self.pipeline.inject_ball_outcome_into_score_detail(score_detail)

    def build_time_series_velocity_window(
        self, t_impact: Optional[int] = None
    ) -> tuple[list, int, int, list]:
        if self.pipeline is None:
            return [], 0, 0, []
        return self.pipeline.build_time_series_velocity_window(t_impact=t_impact)

    def build_phase_windows(
        self, t_impact: Optional[int] = None, total_frames: Optional[int] = None
    ) -> dict:
        if self.pipeline is None:
            return {}
        # 优先使用 T0 锁定 + 相位隔离的富化版本（含每相位测量指标和 keyframe_index）；
        # 若富化版本无法构建（轨迹不足 / diagnose 异常），降级到固定偏移简化版。
        rich = self.pipeline.build_phase_windows_rich()
        if rich:
            return rich
        return self.pipeline.build_phase_windows(
            t_impact=t_impact, total_frames=total_frames
        )

    def rebuild_leg_annotation(self, score_detail=None, t_impact=None, force_impact_frame=True):
        if self.pipeline is None:
            return None, None
        return self.pipeline.rebuild_leg_annotation(
            score_detail, t_impact=t_impact, force_impact_frame=force_impact_frame
        )

    def get_blurred_frame(self, index: Optional[int]):
        if self.pipeline is None:
            return None
        return self.pipeline.get_blurred_frame(index)


# --------------------------------------------------------------------------
# 第二步：FastAPI 应用初始化 + CORS 跨域配置
# --------------------------------------------------------------------------

@asynccontextmanager
async def _app_lifespan(_app: FastAPI):
    """启动时初始化软删除列，并拉起 12 小时自动备份守护线程。"""
    try:
        from db import init_db, start_auto_backup_daemon

        init_db()
        started = start_auto_backup_daemon()
        safe_print(
            "【api_server】数据安全守护已就绪"
            + ("（已启动自动备份线程）" if started else "（自动备份线程已在运行）")
        )
    except Exception as exc:  # noqa: BLE001 - 备份守护失败不得阻断主服务
        safe_print(f"【api_server】启动数据安全守护失败（主服务继续运行）：{exc}")
    yield


app = FastAPI(
    title="小学足球AI可视化反馈系统 - 后台服务网关",
    version="1.1.0",
    lifespan=_app_lifespan,
)

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


@app.exception_handler(RequestValidationError)
async def _pydantic_request_validation_handler(request, exc: RequestValidationError):
    """非法 Payload → 标准 HTTP 422（Pydantic 强类型校验失败）。"""
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": getattr(exc, "body", None),
            "message": "请求体字段类型或约束不合法",
        },
    )


@app.get("/")
def read_root():
    return {"service": "AI-Football-Feedback API Gateway", "status": "running"}


# --------------------------------------------------------------------------
# 第三步：核心接口一（上）—— 本地视频文件上传
# --------------------------------------------------------------------------


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """接收前端上传的本地视频文件（推荐 MP4），保存到项目根目录 uploads/ 临时目录，
    并用 OpenCV 读取第一帧做解码可用性校验，再返回可直接用于分析的绝对路径。

    前端拿到 video_path 后，在下一步通过 WebSocket 发送
    {"action": "start", "source": "file", "video_path": video_path} 即可启动分析。

    解码失败（如部分 .MOV 编码不被本机 OpenCV 支持）时返回标准 JSON HTTP 500，
    避免进程异常退出导致前端只看到含糊的 Failed to fetch。
    """
    saved_path: Optional[str] = None
    try:
        file_extension = os.path.splitext(file.filename or "")[1] or ".mp4"
        saved_filename = f"{uuid.uuid4().hex}{file_extension}"
        saved_path = os.path.join(UPLOAD_DIR, saved_filename)

        with open(saved_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 第一帧验证：尽早暴露不受支持的容器/编码（常见于 iPhone .MOV）
        cap = cv2.VideoCapture(saved_path)
        try:
            if not cap.isOpened():
                raise RuntimeError(
                    f"无法打开视频文件（格式可能不受支持）：{file.filename or saved_filename}"
                )
            ret, _frame = cap.read()
            if not ret:
                raise RuntimeError(
                    f"视频第一帧读取失败（ret == False），本机 OpenCV 可能不支持该编码"
                    f"（如 .MOV）。请改用标准 H.264 .mp4。文件：{file.filename or saved_filename}"
                )
        finally:
            cap.release()

        return {"video_path": saved_path, "original_filename": file.filename}
    except Exception as e:
        # 最外层兜底：绝不让未捕获异常把服务进程打挂；统一回传可读 JSON 500
        if saved_path and os.path.exists(saved_path):
            try:
                os.remove(saved_path)
            except OSError:
                pass
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": (
                    "视频解码失败或程序异常，请尽量使用标准的 .mp4 格式。"
                    f"详细错误: {e}"
                ),
            },
        )


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
         "angle": 142.3, "status": "Green", "timestamp": 1234567.89, "fps": 28}
        {"type": "camera_lost", "message": "摄像头信号丢失，尝试重连..."}
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
            # camera_lost / notice / frame 均不结束泵送；仅 stopped / error 收尾
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

                # 实验防干扰：未锁定基线时强烈警告（跨日摄像头偏移污染 SPSS）
                SESSION_METADATA_STORE.warn_if_unlocked(log_fn=safe_print)

                session_id = str(uuid.uuid4())
                source = data.get("source", "webcam")
                video_path = data.get("video_path")
                camera_index = int(data.get("camera_index", 0))

                current_session = AnalysisSession(
                    session_id=session_id, source=source, video_path=video_path, camera_index=camera_index
                )
                SESSIONS[session_id] = current_session
                current_session.start()

                baseline_meta = SESSION_METADATA_STORE.status_dict()
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "started",
                            "session_id": session_id,
                            "baseline_session_id": baseline_meta.get("session_id"),
                            "is_baseline_trusted": bool(
                                baseline_meta.get("session_locked")
                            ),
                        }
                    )
                )
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
    """生成诊断报告入参：非法类型由 FastAPI/Pydantic 自动返回 HTTP 422。"""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    session_id: str = Field(..., min_length=1, max_length=128)
    student_number: str = Field(default="", max_length=64)

    @field_validator("session_id")
    @classmethod
    def _session_id_non_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("session_id 不可为空")
        return text


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

    # 实验防干扰：报告生成路径再次校验基线锁定
    SESSION_METADATA_STORE.warn_if_unlocked(log_fn=safe_print)

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
        # 【V3.11】射门结果闭环：出球初速度 / 发射仰角注入 score_detail
        try:
            score_detail = session.inject_ball_outcome_into_score_detail(score_detail)
        except Exception as ball_exc:  # noqa: BLE001
            safe_print(f"【api_server】出球结果注入失败（不影响评分）：{ball_exc}")

    # 实验防干扰：score_detail 成功生成时强制打基线水印
    if isinstance(score_detail, dict):
        score_detail = stamp_baseline_watermark(
            score_detail, analysis_session_id=payload.session_id
        )

    # 【关键接线】必须把 DeterministicScorer 的 score_detail 交给 AIGC（已含脏数据 fallback）。
    diagnosis_for_aigc = None
    if isinstance(score_detail, dict):
        diagnosis_for_aigc = {"score_detail": score_detail}
    print(
        "【api_server】即将发给大模型的完整诊断 JSON：\n"
        + json.dumps(diagnosis_for_aigc, indent=4, ensure_ascii=False, default=str)
    )
    ai_result = llm_agent.generate_session_report(
        hit_stats=hit_stats,
        student_number=payload.student_number,
        sample_angles=sample_angles,
        deterministic_score=deterministic_score,
        diagnosis_json=diagnosis_for_aigc,
    )

    # 分数以确定性引擎为准；无轨迹时回退 LLM 分
    final_score = (
        float(deterministic_score)
        if deterministic_score is not None
        else float(ai_result["score"])
    )

    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    clinical_echo = ai_result.get("clinical_echo") or ai_result.get("clinicalEcho") or ""
    overview = ai_result.get("overview") or clinical_echo
    biomechanical_analysis = ai_result.get("biomechanical_analysis") or ""
    magic_metaphor = (
        ai_result.get("magic_metaphor")
        or ai_result.get("correction_metaphor")
        or ai_result.get("painPoint")
        or ""
    )
    action_plan = (
        ai_result.get("action_plan")
        or ai_result.get("praise_encouragement")
        or ai_result.get("prescription")
        or ""
    )
    aigc_source = ai_result.get("aigc_source") or ai_result.get("aigcSource") or "fallback"
    clinical_brief = ai_result.get("clinical_brief") or ai_result.get("clinicalBrief")
    full_text = (
        f"学号 {payload.student_number or '未填写'} 本次综合练习诊断报告\n\n"
        f"发力稳定性评分：{final_score:.2f} 分（共采集 {total_attempts} 次有效触球数据）。\n"
        + (f"【综合评价】{overview}\n" if overview else "")
        + (f"【动力链病理分析】{biomechanical_analysis}\n" if biomechanical_analysis else "")
        + (f"【具身隐喻处方】{magic_metaphor}\n" if magic_metaphor else "")
        + (f"【下一步训练指令】{action_plan}" if action_plan else "")
    )

    # 【大小腿夹角可视化】优先折叠极值帧 + 摆动腿关键点重标定；几何不合格则降级提示
    impact_frame_image = None
    if session is not None:
        try:
            ann_frame, ann_metrics = session.rebuild_leg_annotation(
                score_detail if isinstance(score_detail, dict) else None,
                t_impact=t_impact_locked if t_impact_locked is not None else session.t_impact,
            )
            if ann_frame is not None and ann_metrics is not None:
                session.impact_frame = ann_frame
                session.impact_metrics = ann_metrics
                annotated_frame = draw_biomechanics_annotation(ann_frame, ann_metrics)
                ok, buffer = cv2.imencode(
                    ".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, IMPACT_FRAME_JPEG_QUALITY]
                )
                if ok:
                    impact_frame_image = (
                        f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('ascii')}"
                    )
        except Exception as ann_exc:  # noqa: BLE001
            safe_print(f"【api_server】大小腿夹角标注重建失败，回退旧链路：{ann_exc}")
            if session.impact_frame is not None and session.impact_metrics is not None:
                annotated_frame = draw_biomechanics_annotation(
                    session.impact_frame, session.impact_metrics
                )
                ok, buffer = cv2.imencode(
                    ".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, IMPACT_FRAME_JPEG_QUALITY]
                )
                if ok:
                    impact_frame_image = (
                        f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('ascii')}"
                    )

    # V2.5 Kinovea 联动：全程角速度 + Action ROI 鞭打发力窗口（须在 pop 前读完）
    angular_velocities_out = None
    frame_count_out = len(sample_angles)
    time_series_velocity: Optional[list] = None
    impact_index_in_window: Optional[int] = None
    absolute_timestamps: Optional[list] = None
    if session is not None:
        angular_velocities_out = [float(v) for v in session._trajectory_omega]
        frame_count_out = int(
            getattr(session, "sync_frame_count", None) or len(session._trajectory_omega) or len(sample_angles)
        )
        if len(session._trajectory_omega) > 0:
            (
                time_series_velocity,
                impact_index_in_window,
                _roi_start,
                absolute_timestamps,
            ) = session.build_time_series_velocity_window(t_impact=t_impact_locked)
            # 将绝对时间戳一并写入 scoreDetail.action_roi，供前端 scrub 直连视频秒
            if isinstance(score_detail, dict) and absolute_timestamps is not None:
                roi = score_detail.get("action_roi")
                if not isinstance(roi, dict):
                    roi = {}
                    score_detail["action_roi"] = roi
                roi["absolute_timestamps"] = list(absolute_timestamps)
                roi["start"] = int(_roi_start)

    # 【动作相位切分】五阶段帧索引区间，全部以 t_impact 为绝对基准点相对偏移。
    # 必须在 SESSIONS.pop 之前算完（pop 后管线轨迹即被释放）。
    phase_windows = None
    if session is not None:
        try:
            phase_windows = session.build_phase_windows(
                t_impact=t_impact_locked, total_frames=frame_count_out
            ) or None
            if isinstance(score_detail, dict) and phase_windows:
                score_detail["phase_windows"] = phase_windows
        except Exception as phase_exc:  # noqa: BLE001 - 相位切分失败不阻断报告
            safe_print(f"【api_server】动作相位切分失败（不影响评分）：{phase_exc}")
            phase_windows = None

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
        "overview": overview,
        "biomechanical_analysis": biomechanical_analysis,
        "magic_metaphor": magic_metaphor,
        "action_plan": action_plan,
        "painPoint": ai_result.get("painPoint") or biomechanical_analysis or magic_metaphor,
        "prescription": ai_result.get("prescription") or action_plan,
        "correction_metaphor": ai_result.get(
            "correction_metaphor", magic_metaphor or ai_result.get("painPoint")
        ),
        "praise_encouragement": ai_result.get(
            "praise_encouragement", action_plan or ai_result.get("prescription")
        ),
        "clinical_echo": overview or clinical_echo,
        "clinicalEcho": overview or clinical_echo,
        "aigc_source": aigc_source,
        "aigcSource": aigc_source,
        "clinical_brief": clinical_brief,
        "clinicalBrief": clinical_brief,
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
        # absolute_timestamps：切片内每帧在原视频中的绝对秒，消除波形-视频时空脱节
        "time_series_velocity": time_series_velocity,
        "timeSeriesVelocity": time_series_velocity,
        "absolute_timestamps": absolute_timestamps,
        "absoluteTimestamps": absolute_timestamps,
        "impact_index_in_window": impact_index_in_window,
        "impactIndexInWindow": impact_index_in_window,
        # 动作相位切分。优先为 T0 锁定 + 相位隔离的富化版本，key 为
        # approach_phase / support_phase / backswing_phase / impact_phase /
        # follow_through_phase，每段含 start_index、end_index（闭区间）、
        # start_ms_rel、end_ms_rel（相对 T0 毫秒）、frame_count、
        # keyframe_index（该相位代表帧）以及 metrics（仅 provenance 为
        # measured/calibrated 的实测指标，估计值与缺省值不外泄）。
        # 轨迹不足或诊断异常时降级为固定偏移简化版，key 为
        # approach / plant / fold / impact / follow_through，
        # 每段为 {start_frame, end_frame} 闭区间，均以 t_impact 相对偏移得出。
        "phase_windows": phase_windows,
        "phaseWindows": phase_windows,
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
        # 实验防干扰：基线水印摘要（完整字段已写入 scoreDetail）
        "baseline_session_id": (
            score_detail.get("baseline_session_id")
            if isinstance(score_detail, dict)
            else None
        ),
        "is_baseline_trusted": bool(
            isinstance(score_detail, dict)
            and score_detail.get("is_baseline_trusted")
        ),
        "class_id": (
            score_detail.get("class_id") if isinstance(score_detail, dict) else ""
        ),
        "camera_height_cm": (
            score_detail.get("camera_height_cm")
            if isinstance(score_detail, dict)
            else None
        ),
        "calibrator_status": (
            score_detail.get("calibrator_status")
            if isinstance(score_detail, dict)
            else "unlocked"
        ),
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
    """归档池落盘：sessions 必须为对象列表，非法类型 → HTTP 422。"""

    model_config = ConfigDict(extra="ignore")

    sessions: list[dict] = Field(default_factory=list)

    @field_validator("sessions")
    @classmethod
    def _sessions_are_dicts(cls, value: list) -> list:
        if not isinstance(value, list):
            raise ValueError("sessions 必须为数组")
        for i, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(f"sessions[{i}] 必须为对象")
        return value


class LockBaselineRequest(BaseModel):
    """锁定实验环境基线：class_id / 摄像头高度 / 标定矩阵状态。"""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    class_id: str = Field(default="", max_length=128)
    camera_height_cm: Optional[float] = Field(default=None, ge=0, le=500)
    calibrator_status: str = Field(default="unknown", max_length=64)
    school: str = Field(default="", max_length=128)
    class_group: str = Field(default="", max_length=128)
    session_id: Optional[str] = Field(default=None, max_length=128)


@app.post("/api/lock_baseline")
def lock_baseline(payload: LockBaselineRequest):
    """实验防干扰：锁定当前课堂环境参数，生成全局唯一 baseline session_id。

    锁定后，每一次 ``score_detail`` / ``global_training_db.json`` / SPSS 导出
    都会强制写入该水印；未锁定时分析仍可运行，但会打强烈警告且
    ``is_baseline_trusted=false``。
    """
    try:
        checkpoint = SESSION_METADATA_STORE.lock_baseline(
            class_id=payload.class_id,
            camera_height_cm=payload.camera_height_cm,
            calibrator_status=payload.calibrator_status,
            school=payload.school,
            class_group=payload.class_group,
            session_id=payload.session_id,
        )
        safe_print(
            f"【api_server】基线已锁定 session_id={checkpoint.session_id} "
            f"class_id={checkpoint.class_id} "
            f"camera_height_cm={checkpoint.camera_height_cm} "
            f"calibrator_status={checkpoint.calibrator_status}"
        )
        body = checkpoint.to_dict()
        body.update(
            {
                "success": True,
                "session_locked": True,
                "is_baseline_trusted": True,
            }
        )
        return body
    except Exception as exc:  # noqa: BLE001
        safe_print(f"【api_server】lock_baseline 失败：{exc}")
        raise HTTPException(status_code=500, detail=f"锁定基线失败: {exc}") from exc


@app.get("/api/baseline_status")
def baseline_status():
    """查询当前实验基线是否已锁定及环境参数。"""
    status = SESSION_METADATA_STORE.status_dict()
    status["success"] = True
    return status


@app.post("/api/unlock_baseline")
def unlock_baseline():
    """解除基线锁定（换班 / 跨日重标定前调用）。"""
    SESSION_METADATA_STORE.unlock_baseline()
    safe_print("【api_server】基线锁定已解除（session_locked=False）")
    return {"success": True, "session_locked": False, "is_baseline_trusted": False}


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
# B 组课后复盘：学生摘要 + 最佳 vs 待改进 双关键帧
# --------------------------------------------------------------------------


@app.get("/api/review/student_summary")
def review_student_summary(
    student_id: str = "",
    session_date: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """B 组课后复盘聚合：按 total_score 极值筛选 best / improve 触球关键帧。

    Query：
        student_id —— 学号（必填）
        session_date —— YYYY-MM-DD，限定本节课日期（可选）
        session_id —— 归档池中的课时实体 id（可选，优先于日期）

    有效尝试数 < 2 时 ``comparison_frames`` 为 null，``comparison_available=False``。
    """
    sid = (student_id or "").strip()
    if not sid:
        return {
            "success": False,
            "student_id": "",
            "session_date": (session_date or "").strip()[:10] or None,
            "session_id": (session_id or "").strip() or None,
            "attempt_count": 0,
            "attempts": [],
            "comparison_frames": None,
            "comparison_available": False,
            "message": "缺少 student_id",
        }
    try:
        from db import student_review_summary

        return student_review_summary(
            sid,
            session_date=session_date,
            session_id=session_id,
            web_session_path=WEB_SESSION_LOG_PATH,
            global_db_path=GLOBAL_DB_PATH,
        )
    except Exception as exc:  # noqa: BLE001
        safe_print(f"【api_server】review/student_summary 失败：{exc}")
        return {
            "success": False,
            "student_id": sid,
            "session_date": (session_date or "").strip()[:10] or None,
            "session_id": (session_id or "").strip() or None,
            "attempt_count": 0,
            "attempts": [],
            "comparison_frames": None,
            "comparison_available": False,
            "message": f"复盘聚合失败：{exc}",
        }


# --------------------------------------------------------------------------
# 第五步再半：核心接口四 —— 调用 DeepSeek 生成「跨次尝试聚合诊断报告」
#
#         课后集中复盘看板里，教练查看某位同学 2~3 次尝试的整体趋势时，
#         前端会把这几次尝试各自的评分/三级命中统计打包发给这个接口，
#         后台真正调用 llm_agent.generate_aggregate_diagnosis() 请求
#         DeepSeek 大模型，生成"这几脚球之间发生了什么变化"的诊断建议。
# --------------------------------------------------------------------------


def _coerce_json_float(value, *, field_name: str = "value", allow_none: bool = True):
    """将 JSON 数值（含 numpy 标量 / 数字字符串）统一转为 Python float。

    Pydantic v2 对 int 字段拒绝带小数部分的 number（int_from_float → HTTP 422）；
    业务指标一律走 float，避免嵌套模型叶子节点再踩坑。
    """
    if value is None or value == "":
        if allow_none:
            return None
        return 0.0
    if isinstance(value, bool):
        raise ValueError(f"{field_name} 不能为布尔值")
    try:
        # numpy.float64 等：优先 item()/float()，避免 isinstance(x, float) 漏判
        if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list)):
            value = value.item()
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须为数值") from exc


def _coerce_float_dict(value, *, field_name: str = "metrics") -> Optional[dict]:
    """Dict[str, int] 风格载荷 → Dict[str, float]；非 dict 视为非法。"""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} 必须为对象")
    out: dict = {}
    for key, raw in value.items():
        if raw is None or raw == "":
            continue
        out[str(key)] = _coerce_json_float(raw, field_name=f"{field_name}.{key}", allow_none=False)
    return out


def _coerce_float_dict_soft(value, *, default_empty: bool = False) -> Optional[dict]:
    """宽松 dict[str, float]：非法整体/叶子一律跳过，绝不抛错触发 422。"""
    if value is None or value == "":
        return {} if default_empty else None
    if not isinstance(value, dict):
        return {} if default_empty else None
    out: dict = {}
    for key, raw in value.items():
        if raw is None or raw == "" or isinstance(raw, bool):
            continue
        try:
            if hasattr(raw, "item") and not isinstance(raw, (str, bytes, dict, list)):
                raw = raw.item()
            out[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return out


def _coerce_float_list_soft(value) -> Optional[list]:
    """宽松 float 列表：非数组或坏叶子 → None / 跳过，避免可选字段 422。"""
    if value is None or value == "":
        return None
    if not isinstance(value, list):
        return None
    out: list = []
    for item in value:
        if item is None or item == "" or isinstance(item, bool):
            continue
        try:
            if hasattr(item, "item") and not isinstance(item, (str, bytes, dict, list)):
                item = item.item()
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def _coerce_optional_float_soft(value) -> Optional[float]:
    """可选 float：无法解析时回落 None，不抛 ValidationError。"""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list)):
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return None


class AggregateHitStats(BaseModel):
    """三级命中统计：全部 float，兼容均值/百分比等小数，杜绝 int_from_float。"""

    model_config = ConfigDict(extra="ignore")

    green: float = 0.0
    yellow: float = 0.0
    red: float = 0.0

    @field_validator("green", "yellow", "red", mode="before")
    @classmethod
    def _coerce_hit_counts(cls, value):
        coerced = _coerce_optional_float_soft(value)
        return 0.0 if coerced is None else coerced


class AggregateAttemptSummary(BaseModel):
    """单次尝试摘要。所有业务数值均为 float；禁止任何指标字段声明为 int。"""

    model_config = ConfigDict(extra="ignore")

    # 序号也用 float：JS/聚合链路可能传 2.0；下游 dump 时再 round 成展示序号
    attemptNumber: float = 0.0
    score: Optional[float] = None
    hitStats: Optional[AggregateHitStats] = None
    # 前端若误带完整报告字段，按 float 接收后忽略用途，避免再引入 int 子模型
    totalAttempts: Optional[float] = None
    avgKneeAngle: Optional[float] = None
    kneeFlexionAngle: Optional[float] = None
    stabilityScore: Optional[float] = None
    radar_scores: Optional[dict[str, float]] = Field(default_factory=dict)
    radarScores: Optional[dict[str, float]] = Field(default_factory=dict)
    scores: Optional[dict[str, float]] = Field(default_factory=dict)
    comment: Optional[str] = ""
    task_status: Optional[str] = None

    @field_validator(
        "attemptNumber",
        "score",
        "totalAttempts",
        "avgKneeAngle",
        "kneeFlexionAngle",
        "stabilityScore",
        mode="before",
    )
    @classmethod
    def _coerce_metric_floats(cls, value, info):
        coerced = _coerce_optional_float_soft(value)
        if info.field_name == "attemptNumber":
            return 0.0 if coerced is None else max(0.0, coerced)
        return coerced

    @field_validator("radar_scores", "radarScores", "scores", mode="before")
    @classmethod
    def _coerce_radar_maps(cls, value):
        return _coerce_float_dict_soft(value, default_empty=True)

    @field_validator("hitStats", mode="before")
    @classmethod
    def _coerce_hit_stats(cls, value):
        if value is None or value == "":
            return None
        if isinstance(value, AggregateHitStats):
            return value
        if not isinstance(value, dict):
            return None
        return _coerce_float_dict_soft(value, default_empty=True)


class GenerateAggregateReportRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    student_number: str = Field(default="", max_length=64)
    attempts: list[AggregateAttemptSummary] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_attempts_tree(cls, data):
        """深度清洗 attempts 树：叶子数值全部 float 化，杜绝嵌套 int 校验。"""
        if not isinstance(data, dict):
            return data
        attempts = data.get("attempts")
        if not isinstance(attempts, list):
            return data

        def _walk(node):
            if isinstance(node, dict):
                out = {}
                for key, raw in node.items():
                    if isinstance(raw, (dict, list)):
                        out[key] = _walk(raw)
                    elif isinstance(raw, bool) or raw is None or isinstance(raw, str):
                        out[key] = raw
                    else:
                        try:
                            out[key] = _coerce_json_float(raw, field_name=str(key), allow_none=True)
                        except ValueError:
                            out[key] = raw
                return out
            if isinstance(node, list):
                return [_walk(item) for item in node]
            return node

        return {**data, "attempts": [_walk(item) if isinstance(item, dict) else item for item in attempts]}


@app.post("/api/generate_aggregate_report")
def generate_aggregate_report(payload: GenerateAggregateReportRequest):
    """基于同一位学生历史有效尝试（≥2 次，可跨课时/跨天）生成跨次趋势诊断。"""
    attempts_summary = []
    for item in payload.attempts:
        row = item.model_dump()
        # LLM prompt 使用整数序号；校验层已改为 float 以防 422
        try:
            row["attemptNumber"] = int(round(float(row.get("attemptNumber") or 0)))
        except (TypeError, ValueError):
            row["attemptNumber"] = 0
        attempts_summary.append(row)
    scores = [item.score for item in payload.attempts if isinstance(item.score, (int, float))]

    # 阈值放宽：只要历史有效评分 ≥ 2（不论是否同一天）即可请求聚合诊断
    if len(scores) < 2:
        raise HTTPException(
            status_code=400,
            detail="历史有效数据不足 2 次，暂无法生成聚合诊断（可不在同一天累计）",
        )

    try:
        ai_result = llm_agent.generate_aggregate_diagnosis(
            student_number=payload.student_number, attempts_summary=attempts_summary
        )
    except Exception as exc:  # noqa: BLE001
        print(f"【api_server】聚合诊断接口异常：{exc}")
        raise HTTPException(
            status_code=504,
            detail="大模型生成超时，请重试",
        ) from exc

    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")

    # 【聚合稳定性得分计算】以各次尝试评分的离散程度（标准差）换算稳定性得分：
    # 各趟评分越接近，说明动作表现越稳定，得分越高；忽高忽低则相应扣分。
    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    std_dev = variance ** 0.5
    stability_score = int(max(0, min(100, round(100 - std_dev * 1.5))))

    llm_error = ai_result.get("error")
    full_text = (
        f"学号 {payload.student_number or '未填写'} 跨课时聚合诊断报告"
        f"（共 {len(attempts_summary)} 次历史尝试，动作表现稳定性得分 {stability_score} 分）\n\n"
        f"{ai_result['trendDescription']}\n"
        f"{ai_result['prescription']}"
    )
    if llm_error:
        full_text = f"⚠️ {llm_error}\n\n{full_text}"

    # 极值双关键帧：复用同一批 attempts（≥2）拼装 best vs improve
    comparison_frames = None
    try:
        from db import build_comparison_frames

        comparison_frames = build_comparison_frames(
            [
                {
                    "attemptNumber": row.get("attemptNumber"),
                    "attempt_id": row.get("attemptNumber"),
                    "score": row.get("score"),
                    "total_score": row.get("score"),
                    "impactFrameBase64": row.get("impactFrameImage")
                    or row.get("impactFrameBase64"),
                    "biomechanicalErrors": row.get("biomechanicalErrors") or [],
                    "scoreDetail": row.get("scoreDetail"),
                    "reportData": {
                        "score": row.get("score"),
                        "impactFrameImage": row.get("impactFrameImage")
                        or row.get("impactFrameBase64"),
                        "scoreDetail": row.get("scoreDetail"),
                        "hitStats": row.get("hitStats"),
                        "painPoint": row.get("painPoint"),
                    },
                }
                for row in attempts_summary
            ]
        )
    except Exception as cmp_exc:  # noqa: BLE001
        safe_print(f"【api_server】聚合报告 comparison_frames 组装失败：{cmp_exc}")
        comparison_frames = None

    return {
        "stabilityScore": stability_score,
        "trendDescription": ai_result["trendDescription"],
        "prescription": ai_result["prescription"],
        "fullText": full_text,
        "generatedAt": generated_at,
        "llmError": llm_error,
        "comparison_frames": comparison_frames,
        "comparison_available": comparison_frames is not None,
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
    """前端归档载荷。V3.1 新增字段一律 Optional，避免严格校验导致 422。"""

    model_config = ConfigDict(extra="ignore")

    # "realtime" | "delayed" —— 对应一级归档子文件夹「实时反馈」/「延时反馈」
    mode: str = "realtime"
    # 学校/机构名称、班级/实验组别名称 —— 拼接成二级归档子文件夹
    school: str = ""
    classGroup: str = ""
    # 学生编号/学号 —— 三级归档子文件夹，也用于 Word 文件命名
    studentNumber: str = ""
    # AI 诊断报告核心字段：发力综合评分、有效采样次数、痛点分析、改进建议
    # score 必须接受 float：DeterministicScorer 常返回 72.35 等小数，
    # 若写成 Optional[int] 会在 Pydantic v2 直接 422（int_from_float）。
    score: Optional[float] = None
    totalAttempts: Optional[float] = None
    # 四维深度诊断（大模型主字段）；缺失时 word_reporter 回落旧字段
    overview: Optional[str] = ""
    biomechanical_analysis: Optional[str] = ""
    magic_metaphor: Optional[str] = ""
    action_plan: Optional[str] = ""
    # 旧字段兼容（painPoint≈病理分析/隐喻，prescription≈训练指令）
    painPoint: str = ""
    prescription: str = ""
    correction_metaphor: Optional[str] = ""
    praise_encouragement: Optional[str] = ""
    clinical_echo: Optional[str] = ""
    clinical_brief: Optional[Any] = None
    aigc_source: Optional[str] = None
    comment: Optional[str] = ""
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
    hitStats: Optional[AggregateHitStats] = None
    # 【v4.0 新增：科研级数据矩阵】本次分析全程真实测得的膝关节屈曲角度均值
    # （来自 /api/generate_report 返回的 avgKneeAngle，是 pose_tracker.py 逐帧
    # 真实计算出的物理测量值）。缺失时（例如历史联调数据、前端尚未回填）后端
    # 会自动退化为基于评分的启发式估算，确保导出的学术矩阵绝不出现空值。
    kneeFlexionAngle: Optional[float] = None
    # 【SDT 成就印章】前端原样转发 generate_report 的 scoreDetail，供落盘时
    # 抽出脚踝刚性方差 / 支撑脚横纵偏差 / 五维雷达，供周成就引擎消费。
    # 保持宽松 dict，避免嵌套叶子 int 校验；写入前不做 int 强约束。
    scoreDetail: Optional[dict[str, Any]] = Field(default_factory=dict)
    score_detail: Optional[dict[str, Any]] = Field(default_factory=dict)
    # 【V3.1】雷达小分 / 临时分数字典：缺失或脏值一律软降级，绝不 422
    radar_scores: Optional[dict[str, float]] = Field(default_factory=dict)
    radarScores: Optional[dict[str, float]] = Field(default_factory=dict)
    scores: Optional[dict[str, float]] = Field(default_factory=dict)
    quantified5dScores: Optional[dict[str, float]] = Field(default_factory=dict)
    quantified_5d_scores: Optional[dict[str, float]] = Field(default_factory=dict)
    spatial_trajectory: Optional[dict[str, Any]] = Field(default_factory=dict)
    spatialTrajectory: Optional[dict[str, Any]] = Field(default_factory=dict)
    avgKneeAngle: Optional[float] = None
    t_impact: Optional[float] = None
    tImpact: Optional[float] = None
    time_series_velocity: Optional[list[float]] = Field(default_factory=list)
    timeSeriesVelocity: Optional[list[float]] = Field(default_factory=list)
    # 切片内每帧在原视频中的绝对秒：[ (roi_start+i)/fps, ... ]
    absolute_timestamps: Optional[list[float]] = Field(default_factory=list)
    absoluteTimestamps: Optional[list[float]] = Field(default_factory=list)
    impact_index_in_window: Optional[float] = None
    impactIndexInWindow: Optional[float] = None
    frame_count: Optional[float] = None
    frameCount: Optional[float] = None
    angular_velocities: Optional[list[float]] = Field(default_factory=list)
    angularVelocities: Optional[list[float]] = Field(default_factory=list)
    action_roi: Optional[dict[str, Any]] = Field(default_factory=dict)
    fullText: Optional[str] = ""
    scoringEngine: Optional[str] = ""
    task_status: Optional[str] = None
    fatigue_warning: Optional[Any] = None
    fatigueWarning: Optional[Any] = None

    @field_validator(
        "score",
        "totalAttempts",
        "kneeFlexionAngle",
        "avgKneeAngle",
        "t_impact",
        "tImpact",
        "impact_index_in_window",
        "impactIndexInWindow",
        "frame_count",
        "frameCount",
        mode="before",
    )
    @classmethod
    def _coerce_word_report_floats(cls, value):
        return _coerce_optional_float_soft(value)

    @field_validator(
        "radar_scores",
        "radarScores",
        "scores",
        "quantified5dScores",
        "quantified_5d_scores",
        mode="before",
    )
    @classmethod
    def _coerce_word_radar(cls, value):
        return _coerce_float_dict_soft(value, default_empty=True)

    @field_validator(
        "scoreDetail",
        "score_detail",
        "spatial_trajectory",
        "spatialTrajectory",
        "action_roi",
        mode="before",
    )
    @classmethod
    def _coerce_optional_dicts(cls, value):
        if value is None or value == "":
            return {}
        return value if isinstance(value, dict) else {}

    @field_validator("hitStats", mode="before")
    @classmethod
    def _coerce_word_hit_stats(cls, value):
        if value is None or value == "":
            return None
        if isinstance(value, AggregateHitStats):
            return value
        if not isinstance(value, dict):
            return None
        return _coerce_float_dict_soft(value, default_empty=True)

    @field_validator(
        "time_series_velocity",
        "timeSeriesVelocity",
        "absolute_timestamps",
        "absoluteTimestamps",
        "angular_velocities",
        "angularVelocities",
        mode="before",
    )
    @classmethod
    def _coerce_float_lists(cls, value):
        coerced = _coerce_float_list_soft(value)
        return [] if coerced is None else coerced


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


def _classify_biomechanical_errors(hit_stats: Optional[Any], score: Optional[float]) -> list[str]:
    """根据本次尝试的三级命中统计与综合评分，启发式推导出本条记录命中的
    生物力学错误分类标签列表（可能为空，也可能同时命中多个维度）。
    """
    if not hit_stats:
        return []
    if isinstance(hit_stats, BaseModel):
        hit_stats = hit_stats.model_dump()
    if not isinstance(hit_stats, dict):
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


def _provenance_from_indicator(entry: Any) -> Optional[str]:
    """从 scoreDetail.indicators 条目读取 provenance。"""
    if not isinstance(entry, dict):
        return None
    raw = entry.get("provenance")
    if raw is None:
        return None
    text = str(raw).strip().lower()
    return text or None


def _resolve_archive_knee_flexion(
    payload_knee: Any,
    score: Any,
    score_detail: Optional[dict],
) -> tuple[float, str]:
    """归档膝角：实测优先，否则启发式并打标 estimated。"""
    indicators = {}
    if isinstance(score_detail, dict) and isinstance(score_detail.get("indicators"), dict):
        indicators = score_detail["indicators"]
    impact_entry = indicators.get("impact_knee_angle")
    if isinstance(impact_entry, dict) and impact_entry.get("value") is not None:
        try:
            return round(float(impact_entry["value"]), 1), (
                _provenance_from_indicator(impact_entry) or "measured"
            )
        except (TypeError, ValueError):
            pass
    if isinstance(payload_knee, (int, float)):
        return round(float(payload_knee), 1), "measured"
    return _estimate_knee_flexion_angle(score), "estimated"


def _resolve_archive_support_foot(
    score: Any,
    score_detail: Optional[dict],
    snapshot: Optional[dict],
) -> tuple[Optional[float], str]:
    """归档支撑脚横距：仅 scoreDetail/snapshot 实测写入数值+provenance；否则估算并标 estimated。"""
    indicators = {}
    if isinstance(score_detail, dict) and isinstance(score_detail.get("indicators"), dict):
        indicators = score_detail["indicators"]
    dist_entry = indicators.get("distance_cm")
    if isinstance(dist_entry, dict):
        prov = _provenance_from_indicator(dist_entry)
        value = dist_entry.get("value")
        if value is None:
            value = dist_entry.get("scoring_value")
        # 仅 measured/calibrated 的对外 value 视为科研实测；否则若仅有 scoring_value 仍标 estimated
        if prov in ("measured", "calibrated") and dist_entry.get("value") is not None:
            try:
                return round(float(dist_entry["value"]), 2), prov
            except (TypeError, ValueError):
                pass
        if prov in ("default", "estimated", "missing") and value is not None:
            try:
                # 评分中性值可写入看板，但 provenance 不得伪装 measured
                return round(float(value), 2), (
                    "estimated" if prov in ("default", "estimated") else "missing"
                )
            except (TypeError, ValueError):
                pass
    if isinstance(snapshot, dict) and snapshot.get("supportFootDistance") is not None:
        snap_prov = str(snapshot.get("supportFootDistanceProvenance") or "").strip().lower()
        try:
            val = round(float(snapshot["supportFootDistance"]), 2)
        except (TypeError, ValueError):
            val = None
        if val is not None:
            if snap_prov in ("measured", "calibrated"):
                return val, snap_prov
            # snapshot 来自 _nested_metric_value（可能含 scoring_value）→ 无明确实测则 unknown
            return val, snap_prov or "unknown"
    return _estimate_support_foot_distance(score), "estimated"


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


def _is_soft_deleted_record(record: dict) -> bool:
    """Sprint 5：判断全局 JSON 记录是否已软删除。缺省字段视为未删除。"""
    raw = record.get("is_deleted", record.get("isDeleted", False))
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    return str(raw or "").strip().lower() in {"1", "true", "yes", "y"}


def _active_global_records(records: Optional[list] = None) -> list[dict]:
    """返回 ``is_deleted == False`` 的活跃归档（软删记录仍保留在硬盘上）。"""
    source = records if isinstance(records, list) else _load_global_records()
    return [
        record
        for record in source
        if isinstance(record, dict) and not _is_soft_deleted_record(record)
    ]


def _record_test_date(record: dict) -> str:
    """从记录提取 YYYY-MM-DD 测试日期。"""
    for key in ("testDate", "test_date", "session_date"):
        value = str(record.get(key) or "").strip()
        if len(value) >= 10 and value[4] == "-" and value[7] == "-":
            return value[:10]
    ts = str(record.get("timestamp") or "").strip()
    if len(ts) >= 10 and ts[4] == "-" and ts[7] == "-":
        return ts[:10]
    return ""


def _save_global_records(records: list[dict]) -> None:
    """整体覆盖写回全局训练数据库（调用方须已持有 ``_global_db_lock``）。"""
    with open(GLOBAL_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _append_global_record(record: dict) -> None:
    """把一条新记录追加进全局训练数据库并整体覆盖落盘，用锁保证并发写入安全。"""
    # Sprint 5：新归档默认未删除，供教练端软删除体系消费
    if "is_deleted" not in record and "isDeleted" not in record:
        record["is_deleted"] = False
    with _global_db_lock:
        records = _load_global_records()
        records.append(record)
        _save_global_records(records)


def _soft_delete_orm_shot_by_json_id(record_id: str) -> bool:
    """若 cluster_rct.db 中存在可映射的射门行，同步置 is_deleted=True。

    全局 JSON 以 UUID 为主键；ORM 以自增 int 为主键。此处尝试：
      record_id 本身为数字 → 直接按 ORM id 软删。
    匹配失败不视为错误（JSON 软删仍成功）。
    """
    try:
        from db import init_db, session_scope
        from models.shot_attempt_log import ShotAttemptLog

        init_db()
        with session_scope() as session:
            if record_id.isdigit():
                row = session.get(ShotAttemptLog, int(record_id))
                if row is not None and not row.is_deleted:
                    row.is_deleted = True
                    return True
                return False
            return False
    except Exception as exc:  # noqa: BLE001
        safe_print(f"【api_server】同步 ORM 软删除失败（JSON 侧已处理）：{exc}")
        return False


def _soft_delete_orm_shot_matching(record: dict) -> bool:
    """按被试 + 日期 + 总分在 ORM 中定位并软删（尽力匹配）。"""
    try:
        from datetime import date as date_cls

        from db import init_db, session_scope
        from models.shot_attempt_log import ShotAttemptLog
        from sqlalchemy import select

        anon = str(
            record.get("anonymous_id")
            or record.get("studentId")
            or record.get("student_id")
            or ""
        ).strip()
        day_text = _record_test_date(record)
        if not anon or not day_text:
            return False
        try:
            session_day = date_cls.fromisoformat(day_text[:10])
        except ValueError:
            return False
        score = record.get("score")
        score_f = float(score) if isinstance(score, (int, float)) else None

        init_db()
        with session_scope() as session:
            candidates = list(
                session.scalars(
                    select(ShotAttemptLog).where(
                        ShotAttemptLog.anonymous_id == anon,
                        ShotAttemptLog.session_date == session_day,
                        ShotAttemptLog.is_deleted.is_(False),
                    )
                ).all()
            )
            if not candidates:
                return False
            target = None
            if score_f is not None:
                for row in candidates:
                    if row.total_score is not None and abs(float(row.total_score) - score_f) < 0.51:
                        target = row
                        break
            if target is None:
                target = candidates[-1]
            target.is_deleted = True
            return True
    except Exception as exc:  # noqa: BLE001
        safe_print(f"【api_server】ORM 近似软删除失败：{exc}")
        return False


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
    """从扁平字段 / indicators 嵌套 {value|scoring_value|variance} 中取第一个可用标量。"""
    if not isinstance(container, dict):
        return None
    for key in keys:
        entry = container.get(key)
        if isinstance(entry, dict):
            for sub in ("value", "scoring_value", "variance", "scoring_variance"):
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


_RADAR_DIM_ALIASES: dict[str, tuple[str, ...]] = {
    "approach_rhythm": (
        "approach_rhythm",
        "approach_rhythm_score",
        "approach_score",
        "approach",
    ),
    "support_stability": (
        "support_stability",
        "support_stability_score",
        "support_score",
        "support",
    ),
    "backswing_folding": (
        "backswing_folding",
        "backswing_folding_score",
        "backswing_score",
        "folding",
    ),
    "ankle_rigidity": (
        "ankle_rigidity",
        "ankle_rigidity_score",
    ),
    "whipping_velocity": (
        "whipping_velocity",
        "whipping_velocity_score",
        "whipping_score",
        "whipping",
    ),
}


def _pick_radar_dim(radar: dict, dim_key: str) -> Optional[float]:
    for alias in _RADAR_DIM_ALIASES.get(dim_key, (dim_key,)):
        num = _safe_float(radar.get(alias))
        if num is not None:
            return num
    return None


def _extract_radar_dict(record: dict) -> Optional[dict]:
    """从归档记录中提取五维雷达字典（兼容多字段名）。"""
    for key in ("quantified5dScores", "radar_scores", "radarScores"):
        raw = record.get(key)
        if isinstance(raw, dict) and raw:
            return raw
    detail = record.get("scoreDetail") or record.get("score_detail") or {}
    if isinstance(detail, dict):
        raw = detail.get("radar_scores") or detail.get("radarScores")
        if isinstance(raw, dict) and raw:
            return raw
    return None


def _sum_radar_scores(radar: Any) -> Optional[float]:
    if not isinstance(radar, dict):
        return None
    vals: list[float] = []
    for key in _RADAR_DIM_KEYS:
        num = _pick_radar_dim(radar, key)
        if num is not None:
            vals.append(num)
    if not vals:
        return None
    return round(sum(vals), 2)


def _extract_five_dim_total(record: dict) -> Optional[float]:
    """五维雷达总分（每维满分 20，合计满分 100）；缺失时回退综合分 score。"""
    radar = _extract_radar_dict(record)
    if radar is not None:
        total = _sum_radar_scores(radar)
        if total is not None:
            return total
    return _safe_float(record.get("score"))


def _aggregate_radar_average(records: list[dict]) -> dict[str, Optional[float]]:
    """对所选日期内记录按五维求均值，供教练端综合能力画像。"""
    buckets: dict[str, list[float]] = {key: [] for key in _RADAR_DIM_KEYS}
    for record in records:
        if not isinstance(record, dict):
            continue
        radar = _extract_radar_dict(record)
        if not isinstance(radar, dict):
            continue
        for key in _RADAR_DIM_KEYS:
            num = _pick_radar_dim(radar, key)
            if num is not None:
                buckets[key].append(num)

    average: dict[str, Optional[float]] = {}
    for key in _RADAR_DIM_KEYS:
        vals = buckets[key]
        average[key] = round(sum(vals) / len(vals), 2) if vals else None
    return average


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
        if _is_soft_deleted_record(record):
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
        ankle_entry = indicators["ankle_rigidity"]
        for sub in ("value", "scoring_value", "variance", "scoring_variance"):
            ankle = _safe_float(ankle_entry.get(sub))
            if ankle is not None:
                break

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
            for sub in (
                "value",
                "scoring_value",
                "variance",
                "status",
                "penalty",
                "provenance",
                "provenance_tier",
                "method",
                "stiffness_status",
            ):
                if sub in entry:
                    slim[sub] = entry[sub]
            if slim:
                slim_indicators[key] = slim
    if slim_indicators or radar:
        # 保留总分与扣分明细，供 Word 四维报告 / 复盘二次生成引用真实数据
        deductions = score_detail.get("deductions")
        if not isinstance(deductions, list):
            deductions = []
        snapshot["scoreDetail"] = {
            "TotalScore": score_detail.get("TotalScore"),
            "indicators": slim_indicators,
            "radar_scores": radar if isinstance(radar, dict) else None,
            "t_impact": score_detail.get("t_impact"),
            "deductions": deductions,
            # 实验防干扰水印（供归档 / SPSS 二次读取）
            "baseline_session_id": score_detail.get("baseline_session_id"),
            "class_id": score_detail.get("class_id"),
            "camera_height_cm": score_detail.get("camera_height_cm"),
            "calibrator_status": score_detail.get("calibrator_status"),
            "is_baseline_trusted": score_detail.get("is_baseline_trusted"),
            "session_checkpoint": score_detail.get("session_checkpoint"),
        }
    # 顶层快照同步基线字段（即使无 indicators 也保留）
    for bk in (
        "baseline_session_id",
        "class_id",
        "camera_height_cm",
        "calibrator_status",
        "is_baseline_trusted",
        "session_checkpoint",
        "baseline_locked_at",
        "analysis_session_id",
    ):
        if bk in score_detail and score_detail.get(bk) is not None:
            snapshot[bk] = score_detail[bk]
    # Phase 3：若横距指标带实测 provenance，同步到快照顶层
    dist_entry = indicators.get("distance_cm") if isinstance(indicators, dict) else None
    if isinstance(dist_entry, dict):
        prov = str(dist_entry.get("provenance") or "").strip().lower()
        if prov in ("measured", "calibrated") and dist_entry.get("value") is not None:
            try:
                snapshot["supportFootDistance"] = round(float(dist_entry["value"]), 2)
                snapshot["supportFootDistanceProvenance"] = prov
            except (TypeError, ValueError):
                pass
    return snapshot


def _word_report_error_response(
    message: str,
    *,
    status_code: int = 500,
    detail: Optional[str] = None,
) -> JSONResponse:
    """Word 导出失败时的统一 JSON：兼容前端 success 字段，并带 status=error。"""
    body = {
        "success": False,
        "status": "error",
        "message": message,
    }
    if detail:
        body["detail"] = detail
    return JSONResponse(status_code=status_code, content=body)


@app.post("/api/save_word_report")
def save_word_report(payload: SaveWordReportRequest):
    """接收前端组装好的学生档案 + AI 诊断报告 + 关键帧图片 Base64 + 模式类型，
    真正调用 word_reporter.save_feedback_to_word() 完成本地目录树建立与
    Word (.docx) 文档落地，把生成文件的绝对物理路径连同成功消息一并返回。

    【v2.0 新增：双向同步全局数据库】写盘成功后，会自动把这笔完整记录（id、
    时间、学校班级、学号、模式类型、评分、AI 批注、关键帧截图、文件路径）
    追加保存进项目根目录的 global_training_db.json，供教练端看板统一消费；
    同时把这条记录原样返回给前端，前端再同步写入 localStorage 作为极速双保险。

    【异常安全网】整段核心逻辑包在 try/except 中：字段缺失 (KeyError)、
    类型异常 (TypeError/ValueError) 或其它未预料错误都不得拖垮进程，
    一律打详细日志并以 400/500 JSON 返回前端。
    """
    try:
        try:
            payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload)
        except Exception as dump_exc:  # noqa: BLE001
            safe_print(f"【api_server】save_word_report 序列化请求体失败：{dump_exc}")
            safe_print(traceback.format_exc())
            return _word_report_error_response(
                f"报告生成失败，请求数据无法解析: {dump_exc}",
                status_code=400,
                detail=str(dump_exc),
            )

        try:
            result = word_reporter.save_feedback_to_word(payload_dict if isinstance(payload_dict, dict) else {})
        except (KeyError, TypeError, ValueError, AttributeError) as gen_exc:
            safe_print(
                f"【api_server】save_word_report Word 生成阶段数据异常"
                f"（{type(gen_exc).__name__}）：{gen_exc}"
            )
            safe_print(traceback.format_exc())
            return _word_report_error_response(
                f"报告生成失败，部分数据缺失: {gen_exc}",
                status_code=400,
                detail=f"{type(gen_exc).__name__}: {gen_exc}",
            )
        except Exception as gen_exc:  # noqa: BLE001
            safe_print(f"【api_server】save_word_report Word 生成未预料异常：{gen_exc}")
            safe_print(traceback.format_exc())
            return _word_report_error_response(
                f"报告生成失败: {gen_exc}",
                status_code=500,
                detail=str(gen_exc),
            )

        if not isinstance(result, dict):
            return _word_report_error_response(
                "报告生成失败，生成模块返回格式异常",
                status_code=500,
            )

        if not result.get("success"):
            err_msg = result.get("error") or result.get("message") or "未知错误"
            return _word_report_error_response(
                f"报告生成失败，部分数据缺失: {err_msg}",
                status_code=400,
                detail=str(err_msg),
            )

        saved_path = result.get("path") or ""
        saved_directory = result.get("directory")
        saved_filename = result.get("filename")

        # ---- 写盘已成功：后续归档库同步单独兜底，失败不影响 Word 成功响应 ----
        record = None
        try:
            record_type = "delayed" if getattr(payload, "mode", None) == "delayed" else "realtime"
            overview = getattr(payload, "overview", None) or ""
            biomech = getattr(payload, "biomechanical_analysis", None) or ""
            magic = getattr(payload, "magic_metaphor", None) or ""
            action = getattr(payload, "action_plan", None) or ""
            pain = getattr(payload, "painPoint", None) or ""
            prescription = getattr(payload, "prescription", None) or ""
            ai_parts = [
                overview,
                biomech or pain,
                magic,
                action or prescription,
            ]
            ai_feedback_text = "\n".join(
                str(part).strip() for part in ai_parts if part and str(part).strip()
            )

            record_timestamp = getattr(payload, "generatedAt", None) or time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            biomechanical_errors = _classify_biomechanical_errors(
                getattr(payload, "hitStats", None),
                getattr(payload, "score", None),
            )

            detail_payload = getattr(payload, "scoreDetail", None) or getattr(
                payload, "score_detail", None
            )
            detail_dict = detail_payload if isinstance(detail_payload, dict) else None
            snapshot = _metrics_snapshot_from_score_detail(detail_dict) or {}
            if isinstance(detail_dict, dict):
                ind = (
                    detail_dict.get("indicators")
                    if isinstance(detail_dict.get("indicators"), dict)
                    else {}
                )
                dist_ind = ind.get("distance_cm") if isinstance(ind, dict) else None
                if isinstance(dist_ind, dict) and dist_ind.get("provenance") in (
                    "measured",
                    "calibrated",
                ):
                    if dist_ind.get("value") is not None:
                        try:
                            snapshot["supportFootDistance"] = round(float(dist_ind["value"]), 2)
                            snapshot["supportFootDistanceProvenance"] = str(
                                dist_ind.get("provenance")
                            )
                        except (TypeError, ValueError):
                            pass

            knee_val, knee_prov = _resolve_archive_knee_flexion(
                getattr(payload, "kneeFlexionAngle", None),
                getattr(payload, "score", None),
                detail_dict,
            )
            support_val, support_prov = _resolve_archive_support_foot(
                getattr(payload, "score", None),
                detail_dict,
                snapshot,
            )

            record = {
                "id": str(uuid.uuid4()),
                "timestamp": record_timestamp,
                "school": getattr(payload, "school", None) or "",
                "classGroup": getattr(payload, "classGroup", None) or "",
                "studentId": getattr(payload, "studentNumber", None) or "",
                "type": record_type,
                "score": getattr(payload, "score", None),
                "biomechanicalErrors": biomechanical_errors,
                "aiFeedback": ai_feedback_text,
                "overview": str(overview or "").strip() or None,
                "biomechanical_analysis": str(biomech or pain or "").strip() or None,
                "magic_metaphor": str(magic or "").strip() or None,
                "action_plan": str(action or prescription or "").strip() or None,
                "painPoint": str(biomech or pain or "").strip() or None,
                "prescription": str(action or prescription or "").strip() or None,
                "aigc_source": getattr(payload, "aigc_source", None),
                "impactFrameBase64": getattr(payload, "impactFrameImage", None),
                "heatmapBase64": getattr(payload, "heatmapBase64", None)
                or getattr(payload, "heatmap_base64", None),
                "path": saved_path,
                "directory": saved_directory,
                "testDate": _extract_test_date(record_timestamp),
                "groupTypeCode": 1 if record_type == "realtime" else 2,
                "kneeFlexionAngle": knee_val,
                "kneeFlexionAngleProvenance": knee_prov,
                "supportFootDistance": support_val,
                "supportFootDistanceProvenance": support_prov,
                "primaryErrorCode": _derive_primary_error_code(biomechanical_errors),
            }
            if snapshot.get("supportFootDistance") is not None and support_prov in (
                "measured",
                "calibrated",
            ):
                record["supportFootDistance"] = snapshot["supportFootDistance"]
                record["supportFootDistanceProvenance"] = support_prov
            for key, value in snapshot.items():
                if key in ("supportFootDistance", "supportFootDistanceProvenance"):
                    continue
                record[key] = value
            # 实验防干扰：归档 JSON 强制写入基线水印（优先 scoreDetail，再回退全局锁）
            if isinstance(detail_dict, dict) and detail_dict.get("baseline_session_id"):
                for bk in (
                    "baseline_session_id",
                    "class_id",
                    "camera_height_cm",
                    "calibrator_status",
                    "baseline_locked_at",
                    "is_baseline_trusted",
                    "session_checkpoint",
                    "analysis_session_id",
                ):
                    if bk in detail_dict:
                        record[bk] = detail_dict[bk]
            else:
                record = stamp_baseline_watermark(record)
            try:
                _append_global_record(record)
            except Exception as db_exc:  # noqa: BLE001
                safe_print(
                    f"【api_server】追加记录到全局训练数据库失败（Word 文件已正常保存）：{db_exc}"
                )
                safe_print(traceback.format_exc())
        except (KeyError, TypeError, ValueError, AttributeError) as archive_exc:
            safe_print(
                f"【api_server】save_word_report 归档字段组装失败"
                f"（Word 已保存，跳过全局库同步）：{archive_exc}"
            )
            safe_print(traceback.format_exc())
            record = None
        except Exception as archive_exc:  # noqa: BLE001
            safe_print(
                f"【api_server】save_word_report 归档阶段未预料异常"
                f"（Word 已保存）：{archive_exc}"
            )
            safe_print(traceback.format_exc())
            record = None

        response_body = {
            "success": True,
            "status": "ok",
            "message": f"报告已自动保存成 Word！文件已存入：{saved_path}",
            "path": saved_path,
            "directory": saved_directory,
            "filename": saved_filename,
        }
        if record is not None:
            response_body["record"] = record
        return response_body

    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        safe_print(
            f"【api_server】save_word_report 外层捕获数据异常"
            f"（{type(exc).__name__}）：{exc}"
        )
        safe_print(traceback.format_exc())
        return _word_report_error_response(
            f"报告生成失败，部分数据缺失: {exc}",
            status_code=400,
            detail=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - 绝对禁止未捕获异常拖垮服务进程
        safe_print(f"【api_server】save_word_report 外层未预料异常：{exc}")
        safe_print(traceback.format_exc())
        return _word_report_error_response(
            f"报告生成失败: {exc}",
            status_code=500,
            detail=str(exc),
        )


@app.get("/api/get_all_records")
def get_all_records(include_deleted: bool = False):
    """供教练端数据看板一键拉取全量历史归档数据（实时反馈 A 组 + 延时反馈 B 组）。

    默认仅返回 ``is_deleted == False``；软删记录继续躺在硬盘 JSON 中，
    绝不物理清除。传 ``include_deleted=true`` 可审计全量（含已软删）。
    """
    with _global_db_lock:
        records = [r for r in _load_global_records() if isinstance(r, dict)]

    if include_deleted:
        return {"success": True, "records": records, "count": len(records)}

    active = _active_global_records(records)
    return {"success": True, "records": active, "count": len(active)}


class DeleteCoachRecordRequest(BaseModel):
    """教练端单条软删除请求（``is_deleted=True``，禁止物理删除）。"""

    model_config = ConfigDict(extra="ignore")

    id: str = ""
    recordId: Optional[str] = None


class BatchDeleteRecordsRequest(BaseModel):
    """批量软删除：接收多个归档记录 ID（``is_deleted=True``）。"""

    model_config = ConfigDict(extra="ignore")

    ids: list[str] = Field(default_factory=list)
    recordIds: Optional[list[str]] = Field(default_factory=list)


class CalibrateCoachMetricRequest(BaseModel):
    """Phase 4：教练人工标定焦点指标 → provenance=calibrated。"""

    model_config = ConfigDict(extra="ignore")

    id: str = ""
    recordId: Optional[str] = None
    metric_key: str = ""
    value: float = 0.0
    coach_id: Optional[str] = "coach"
    note: Optional[str] = ""

    @field_validator("value", mode="before")
    @classmethod
    def _coerce_calibrate_value(cls, value):
        coerced = _coerce_optional_float_soft(value)
        return 0.0 if coerced is None else coerced


def _normalize_batch_ids(payload: BatchDeleteRecordsRequest) -> list[str]:
    raw_ids = list(payload.ids or []) + list(payload.recordIds or [])
    seen: set[str] = set()
    ids: list[str] = []
    for raw in raw_ids:
        rid = str(raw or "").strip()
        if not rid or rid in seen:
            continue
        seen.add(rid)
        ids.append(rid)
    return ids


def _soft_delete_records_by_ids(ids: list[str]) -> dict:
    """软删除：将 ``is_deleted`` 置 True，记录继续保留在硬盘；同步 ORM。

    绝对禁止 ``del`` / 覆盖式剔除 / DROP。前端清道夫看到的「删除」只是隐身。
    """
    if not ids:
        return {
            "success": False,
            "message": "缺少记录 ID 列表",
            "deletedIds": [],
            "alreadyDeletedIds": [],
            "missingIds": [],
            "count": 0,
            "ormDeleted": 0,
        }

    deleted_ids: list[str] = []
    already_deleted_ids: list[str] = []
    missing_ids: list[str] = []
    matched_records: list[dict] = []

    with _global_db_lock:
        records = _load_global_records()
        wanted = set(ids)
        found: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                continue
            rid = str(record.get("id") or "").strip()
            if not rid or rid not in wanted:
                continue
            found.add(rid)
            if _is_soft_deleted_record(record):
                already_deleted_ids.append(rid)
                continue
            record["is_deleted"] = True
            record["isDeleted"] = True
            deleted_ids.append(rid)
            matched_records.append(record)
        for rid in ids:
            if rid not in found:
                missing_ids.append(rid)
        # 全量写回（含软删行），绝不物理剔除
        _save_global_records([r for r in records if isinstance(r, dict)])

    orm_deleted = 0
    try:
        from db import soft_delete_shot_attempt_matching, soft_delete_shot_attempts_by_ids

        orm_deleted += soft_delete_shot_attempts_by_ids(deleted_ids)
        for record in matched_records:
            rid = str(record.get("id") or "")
            if rid.isdigit():
                continue
            if _soft_delete_orm_shot_by_json_id(rid):
                orm_deleted += 1
                continue
            anon = str(
                record.get("anonymous_id")
                or record.get("studentId")
                or record.get("student_id")
                or ""
            ).strip()
            day = _record_test_date(record)
            score = record.get("score")
            score_f = float(score) if isinstance(score, (int, float)) else None
            orm_deleted += soft_delete_shot_attempt_matching(
                anonymous_id=anon,
                session_date=day,
                total_score=score_f,
            )
    except Exception as exc:  # noqa: BLE001
        safe_print(f"【api_server】批量 ORM 软删除失败：{exc}")

    return {
        "success": True,
        "message": f"已软删除 {len(deleted_ids)} 条记录（数据仍保留在硬盘）",
        "deletedIds": deleted_ids,
        "alreadyDeletedIds": already_deleted_ids,
        "missingIds": missing_ids,
        "count": len(deleted_ids),
        "ormDeleted": orm_deleted,
    }


# 兼容旧内部调用名：一律转发软删除，禁止物理删除
def _hard_delete_records_by_ids(ids: list[str]) -> dict:
    return _soft_delete_records_by_ids(ids)


@app.get("/api/coach/records")
def coach_list_records(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    student_id: Optional[str] = None,
    group: Optional[str] = None,
    class_group: Optional[str] = None,
    include_deleted: bool = False,
):
    """【Sprint 5】教练端数据清道夫列表：支持日期范围 / 被试编号 / 组别过滤。

    Query：
        date_from / date_to —— YYYY-MM-DD，闭区间；
        student_id —— 被试编号模糊匹配；
        group —— 实验组别：realtime | delayed | A | B（亦接受 classGroup 中文别名）；
        class_group —— 行政班/组别精确过滤；
        include_deleted —— 默认 False，隐藏软删除废记录。

    额外返回 ``radar_average``：当前筛选结果集内五维雷达各维平均分。
    """
    records = _load_global_records()
    date_from_s = (date_from or "").strip()[:10]
    date_to_s = (date_to or "").strip()[:10]
    student_q = (student_id or "").strip().lower()
    group_q = (group or "").strip().lower()
    class_q = (class_group or "").strip()

    group_aliases = {
        "realtime": "realtime",
        "a": "realtime",
        "group_a": "realtime",
        "group_a_realtime": "realtime",
        "实验a组": "realtime",
        "delayed": "delayed",
        "b": "delayed",
        "group_b": "delayed",
        "group_b_delayed": "delayed",
        "实验b组": "delayed",
    }
    group_norm = group_aliases.get(group_q, group_q) if group_q else ""

    filtered: list[dict] = []
    source_for_radar: list[dict] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if not include_deleted and _is_soft_deleted_record(record):
            continue

        day = _record_test_date(record)
        if date_from_s and (not day or day < date_from_s):
            continue
        if date_to_s and (not day or day > date_to_s):
            continue

        if student_q:
            sid = str(
                record.get("studentId")
                or record.get("student_id")
                or record.get("anonymous_id")
                or ""
            ).strip().lower()
            if student_q not in sid:
                continue

        if group_norm in ("realtime", "delayed"):
            rtype = str(record.get("type") or "").strip().lower()
            code = record.get("groupTypeCode")
            inferred = (
                "realtime"
                if rtype == "realtime" or code == 1
                else "delayed"
                if rtype == "delayed" or code == 2
                else ""
            )
            if inferred != group_norm:
                continue
        elif group_q and group_norm not in ("realtime", "delayed"):
            # 非 A/B 别名时，按 classGroup 子串匹配
            cg = str(record.get("classGroup") or record.get("cluster_id") or "")
            if group_q not in cg.lower():
                continue

        if class_q:
            cg = str(record.get("classGroup") or record.get("cluster_id") or "").strip()
            if cg != class_q:
                continue

        # 列表轻量投影：诊断快照截断，避免把 Base64 大图塞进 Data Grid
        feedback = str(record.get("aiFeedback") or "").strip()
        snapshot = feedback.replace("\n", " ")
        if len(snapshot) > 120:
            snapshot = snapshot[:117] + "…"

        radar = _extract_radar_dict(record)
        errors = record.get("biomechanicalErrors") or record.get("biomechanical_errors")
        if not isinstance(errors, list):
            errors = []

        source_for_radar.append(record)
        filtered.append(
            {
                "id": record.get("id"),
                "timestamp": record.get("timestamp") or "",
                "testDate": day or record.get("testDate") or "",
                "studentId": record.get("studentId")
                or record.get("anonymous_id")
                or "",
                "school": record.get("school") or "",
                "classGroup": record.get("classGroup") or "",
                "type": record.get("type") or "",
                "groupTypeCode": record.get("groupTypeCode"),
                "score": record.get("score"),
                "diagnosisSnapshot": snapshot,
                "aiFeedback": feedback,
                "biomechanicalErrors": [str(e) for e in errors if e],
                "quantified5dScores": radar,
                "radar_scores": radar,
                "kneeFlexionAngle": record.get("kneeFlexionAngle")
                or record.get("knee_flexion_angle"),
                "supportFootDistance": record.get("supportFootDistance"),
                "supportFootDistanceProvenance": record.get(
                    "supportFootDistanceProvenance"
                ),
                "max_folding_angle": record.get("max_folding_angle")
                or record.get("maxFoldingAngle"),
                "maxFoldingAngleProvenance": record.get("maxFoldingAngleProvenance")
                or record.get("max_folding_angle_provenance"),
                "ankle_rigidity": record.get("ankle_rigidity")
                or record.get("ankle_rigidity_variance"),
                "ankleRigidityProvenance": record.get("ankleRigidityProvenance")
                or record.get("ankle_rigidity_provenance"),
                "lastCalibratedAt": record.get("lastCalibratedAt"),
                "lastCalibratedMetric": record.get("lastCalibratedMetric"),
                "is_deleted": _is_soft_deleted_record(record),
                "path": record.get("path"),
                "directory": record.get("directory"),
            }
        )

    filtered.sort(key=lambda r: str(r.get("timestamp") or ""), reverse=True)
    radar_average = _aggregate_radar_average(source_for_radar)
    return {
        "success": True,
        "records": filtered,
        "count": len(filtered),
        "radar_average": radar_average,
    }


@app.api_route("/api/records/batch", methods=["DELETE", "POST"])
def batch_delete_records(payload: BatchDeleteRecordsRequest):
    """批量软删除归档记录：``is_deleted=True``，同步 ORM；禁止物理删除。

    同时支持 DELETE / POST，避免旧进程或某些客户端对 DELETE+body 不兼容导致 404。
    """
    return _soft_delete_records_by_ids(_normalize_batch_ids(payload))


@app.post("/api/coach/delete_record")
def coach_delete_record(payload: DeleteCoachRecordRequest):
    """单条软删除：仅标记 ``is_deleted=True``，记录仍保留在硬盘 JSON / ORM。"""
    record_id = (payload.id or payload.recordId or "").strip()
    if not record_id:
        return {"success": False, "message": "缺少记录 ID"}

    result = _soft_delete_records_by_ids([record_id])
    if not result.get("success"):
        return result
    if record_id in (result.get("missingIds") or []):
        return {"success": False, "message": f"未找到记录：{record_id}"}
    if record_id in (result.get("alreadyDeletedIds") or []) and record_id not in (
        result.get("deletedIds") or []
    ):
        return {
            "success": True,
            "message": "该记录此前已软删除",
            "id": record_id,
            "deleted": True,
            "alreadyDeleted": True,
            "ormDeleted": result.get("ormDeleted", 0),
        }
    return {
        "success": True,
        "message": "已软删除该记录（数据仍保留在硬盘）",
        "id": record_id,
        "deleted": True,
        "ormDeleted": result.get("ormDeleted", 0),
    }


@app.post("/api/coach/calibrate_metric")
def coach_calibrate_metric(payload: CalibrateCoachMetricRequest):
    """【Phase 4】教练人工覆写焦点指标，强制 provenance=calibrated 并写审计。"""
    from coach_calibration import apply_coach_calibration

    record_id = (payload.id or payload.recordId or "").strip()
    if not record_id:
        return {"success": False, "message": "缺少记录 ID"}

    with _global_db_lock:
        records = _load_global_records()
        target: Optional[dict] = None
        for record in records:
            if isinstance(record, dict) and str(record.get("id") or "") == record_id:
                target = record
                break
        if target is None:
            return {"success": False, "message": f"未找到记录：{record_id}"}
        if _is_soft_deleted_record(target):
            return {"success": False, "message": "已删除记录不可标定"}

        result = apply_coach_calibration(
            target,
            metric_key=payload.metric_key,
            value=payload.value,
            coach_id=payload.coach_id,
            note=payload.note,
        )
        if not result.get("ok"):
            return {"success": False, "message": result.get("message") or "标定失败"}
        _save_global_records(records)

    audit = result.get("audit") or {}
    return {
        "success": True,
        "message": (
            f"已人工标定 {audit.get('metric_key')}={audit.get('value')} "
            f"（provenance=calibrated）"
        ),
        "id": record_id,
        "audit": audit,
        "supportFootDistance": target.get("supportFootDistance"),
        "supportFootDistanceProvenance": target.get("supportFootDistanceProvenance"),
        "max_folding_angle": target.get("max_folding_angle"),
        "ankle_rigidity": target.get("ankle_rigidity"),
    }


# --------------------------------------------------------------------------
# 个人纵向进步图谱：按测试日聚合 + 科研节点 (T0..T4) + 正负向诊断高亮
# --------------------------------------------------------------------------

_PROGRESS_PHASES = ("T0", "T1", "T2", "T3", "T4")

_ERROR_PROGRESS_COPY = {
    "支撑脚位置偏离": "支撑脚落位有进步",
    "膝关节过度屈曲": "膝角控制有进步",
    "随摆转髋不足": "转髋随摆有进步",
    "身体重心偏移": "重心控制有进步",
}

_ERROR_DEFICIT_COPY = {
    "支撑脚位置偏离": "支撑脚落位仍需校准",
    "膝关节过度屈曲": "膝角仍偏屈曲",
    "随摆转髋不足": "随摆转髋仍不足",
    "身体重心偏移": "身体重心仍有偏移",
}


def _progress_record_date(record: dict) -> str:
    """提取 YYYY-MM-DD；优先 testDate，其次 timestamp。"""
    test_date = (record.get("testDate") or record.get("test_date") or "").strip()
    if len(test_date) >= 10 and test_date[4] == "-" and test_date[7] == "-":
        return test_date[:10]
    return _extract_test_date(str(record.get("timestamp") or ""))


def _progress_parse_phase(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().upper()
    if not text:
        return None
    if text.isdigit():
        text = f"T{text}"
    return text if text in _PROGRESS_PHASES else None


def _progress_axis_label(date_text: str, phase: str) -> str:
    """Catapult 风格横轴：MM/DD (Tn)。"""
    md = date_text[5:7] + "/" + date_text[8:10] if len(date_text) >= 10 else date_text
    return f"{md} ({phase})"


def _progress_first_clause(text: str, max_len: int = 28) -> str:
    cleaned = " ".join((text or "").replace("\n", " ").split()).strip()
    if not cleaned:
        return ""
    for sep in ("。", "！", "？", ".", "!", "?"):
        if sep in cleaned:
            cleaned = cleaned.split(sep, 1)[0].strip()
            break
    if len(cleaned) > max_len:
        return cleaned[: max_len - 1] + "…"
    return cleaned


def _progress_pick_knee(record: dict) -> Optional[float]:
    for key in (
        "support_knee_angle_resolved",
        "supportKneeAngleResolved",
        "kneeFlexionAngle",
    ):
        num = _safe_float(record.get(key))
        if num is not None:
            return num
    metrics = record.get("instepKickMetrics") or record.get("instep_kick_metrics")
    if isinstance(metrics, dict):
        for key in ("support_knee_angle", "impact_knee_angle"):
            num = _safe_float(metrics.get(key))
            if num is not None:
                return num
    return None


def _progress_build_highlights(
    *,
    score: Optional[float],
    prev_score: Optional[float],
    errors: list[str],
    prev_errors: list[str],
    ai_feedback: str,
    knee_angle: Optional[float] = None,
    prev_knee_angle: Optional[float] = None,
) -> tuple[Optional[str], Optional[str]]:
    """返回 (positive_highlight, negative_highlight)。"""
    positive: Optional[str] = None
    negative: Optional[str] = None
    knee_gold_min, knee_gold_max = 140.0, 160.0

    cleared = [e for e in prev_errors if e and e not in errors]
    if cleared:
        lead = cleared[0]
        positive = _ERROR_PROGRESS_COPY.get(lead, f"{lead}有改善")
    elif (
        prev_knee_angle is not None
        and knee_angle is not None
        and not (knee_gold_min <= prev_knee_angle <= knee_gold_max)
        and knee_gold_min <= knee_angle <= knee_gold_max
    ):
        positive = "膝角进入黄金区间"
    elif (
        prev_score is not None
        and score is not None
        and score - prev_score >= 1.5
    ):
        delta = int(round(score - prev_score))
        if any("踝" in e for e in prev_errors) or not errors:
            positive = "脚踝锁紧有进步"
        else:
            positive = f"总分提升 {max(delta, 1)} 分"
    elif score is not None and score >= 80 and not errors:
        positive = "动力链传导更顺畅"
    elif prev_errors and len(errors) < len(prev_errors):
        positive = "动作缺陷维度收敛"

    # 处方第二句常含正向动作暗示，仅在有前序节点时作兜底正向高亮
    if not positive and prev_score is not None and ai_feedback:
        parts = [p.strip() for p in ai_feedback.replace("\r", "").split("\n") if p.strip()]
        if len(parts) >= 2:
            cue = _progress_first_clause(parts[1], 24)
            if cue:
                positive = cue

    if errors:
        lead = errors[0]
        negative = _ERROR_DEFICIT_COPY.get(lead, lead)
    else:
        clause = _progress_first_clause(ai_feedback)
        if clause and (score is None or score < 85):
            negative = clause

    if (
        prev_score is not None
        and score is not None
        and score - prev_score <= -5
        and not negative
    ):
        negative = f"总分回落 {int(round(prev_score - score))} 分"

    return positive, negative


def _aggregate_progress_history(
    records: list[dict],
    *,
    student_id: str,
    school: str = "",
    class_group: str = "",
) -> list[dict]:
    """按测试日聚合个人进步点：score 均值 + phase + 诊断高亮。"""
    sid = (student_id or "").strip()
    if not sid:
        return []

    school_f = (school or "").strip()
    class_f = (class_group or "").strip()
    filtered: list[dict] = []
    for raw in records:
        if not isinstance(raw, dict):
            continue
        if _is_soft_deleted_record(raw):
            continue
        rid = str(raw.get("studentId") or raw.get("student_id") or "").strip()
        if rid != sid:
            continue
        if school_f and str(raw.get("school") or "").strip() != school_f:
            continue
        if class_f and str(raw.get("classGroup") or raw.get("class_group") or "").strip() != class_f:
            continue
        filtered.append(raw)

    if not filtered:
        return []

    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    for raw in filtered:
        buckets[_progress_record_date(raw)].append(raw)

    ordered_dates = sorted(buckets.keys())
    explicit_phase_by_date: dict[str, str] = {}
    for day in ordered_dates:
        for raw in buckets[day]:
            phase = _progress_parse_phase(raw.get("timepoint") or raw.get("timePoint") or raw.get("phase"))
            if phase:
                explicit_phase_by_date[day] = phase
                break

    inferred: dict[str, str] = {}
    for index, day in enumerate(ordered_dates):
        inferred[day] = _PROGRESS_PHASES[min(index, len(_PROGRESS_PHASES) - 1)]

    points: list[dict] = []
    prev_score: Optional[float] = None
    prev_errors: list[str] = []
    prev_knee: Optional[float] = None

    for day in ordered_dates:
        day_rows = sorted(
            buckets[day],
            key=lambda r: str(r.get("timestamp") or ""),
        )
        scores = [
            float(s)
            for s in (_safe_float(r.get("score")) for r in day_rows)
            if s is not None
        ]
        mean_score = round(sum(scores) / len(scores), 1) if scores else None

        knees = [
            float(k)
            for k in (_progress_pick_knee(r) for r in day_rows)
            if k is not None
        ]
        mean_knee = round(sum(knees) / len(knees), 1) if knees else None

        # 错误标签：取当日出现频次最高的若干项（保持稳定顺序）
        error_counter: collections.Counter[str] = collections.Counter()
        for raw in day_rows:
            errs = raw.get("biomechanicalErrors") or raw.get("biomechanical_errors") or []
            if isinstance(errs, list):
                for label in errs:
                    text = str(label).strip()
                    if text:
                        error_counter[text] += 1
        top_errors = [label for label, _ in error_counter.most_common(3)]

        # 代表性反馈：取当日最高分尝试的 aiFeedback；同分取最新
        best_row = None
        best_score = None
        for raw in day_rows:
            s = _safe_float(raw.get("score"))
            if best_row is None:
                best_row = raw
                best_score = s
                continue
            if s is None:
                continue
            if best_score is None or s > best_score or (
                s == best_score and str(raw.get("timestamp") or "") >= str(best_row.get("timestamp") or "")
            ):
                best_row = raw
                best_score = s

        ai_feedback = str((best_row or {}).get("aiFeedback") or (best_row or {}).get("ai_feedback") or "")
        phase = explicit_phase_by_date.get(day) or inferred[day]
        positive, negative = _progress_build_highlights(
            score=mean_score,
            prev_score=prev_score,
            errors=top_errors,
            prev_errors=prev_errors,
            ai_feedback=ai_feedback,
            knee_angle=mean_knee,
            prev_knee_angle=prev_knee,
        )

        latest_ts = str((day_rows[-1].get("timestamp") or f"{day} 12:00:00"))
        points.append(
            {
                "date": day,
                "phase": phase,
                "timestamp": latest_ts,
                "score": mean_score,
                "kneeAngle": mean_knee,
                "attemptCount": len(day_rows),
                "label": _progress_axis_label(day, phase),
                "positiveHighlight": positive,
                "negativeHighlight": negative,
                "biomechanicalErrors": top_errors,
                "representativeRecordId": (best_row or {}).get("id"),
                "aiFeedbackSnippet": _progress_first_clause(ai_feedback, 48) or None,
            }
        )
        prev_score = mean_score
        prev_errors = top_errors
        prev_knee = mean_knee

    return points


@app.get("/api/progress/history")
def progress_history(
    student_id: str = "",
    studentId: str = "",
    school: str = "",
    classGroup: str = "",
    class_group: str = "",
):
    """【个人进步图谱】按测试日聚合分数，并附带标准日期与科研节点 (T0..T4)。

    Query:
        student_id / studentId —— 必填学号
        school                 —— 可选学校过滤
        classGroup / class_group —— 可选班级过滤

    返回 points[]：date / phase / timestamp / score / label /
    positiveHighlight / negativeHighlight / kneeAngle / attemptCount
    """
    sid = (student_id or studentId or "").strip()
    if not sid:
        return {
            "success": False,
            "message": "缺少 student_id 参数",
            "studentId": "",
            "points": [],
            "count": 0,
        }

    school_f = (school or "").strip()
    class_f = (classGroup or class_group or "").strip()
    try:
        points = _aggregate_progress_history(
            _active_global_records(),
            student_id=sid,
            school=school_f,
            class_group=class_f,
        )
        return {
            "success": True,
            "studentId": sid,
            "school": school_f or None,
            "classGroup": class_f or None,
            "points": points,
            "count": len(points),
        }
    except Exception as exc:  # noqa: BLE001
        safe_print(f"【api_server】progress/history 失败：{exc}")
        return {
            "success": False,
            "message": f"拉取个人进步历史失败：{exc}",
            "studentId": sid,
            "points": [],
            "count": 0,
        }


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
def export_academic_matrix(measured_only: bool = False):
    """【V3.1】一键导出全数字化 SPSS 标准宽表（JSON 元信息 + 落盘）。

    优先走 AcademicDataExporter 宽表主路径；同时保留长表旁路落盘供 ANOVA。
    【Phase 3】``measured_only=true`` 时宽表失败回退长表仅导出实测横距行。
    """
    try:
        exporter = academic_exporter.AcademicDataExporter.from_db()
        result = exporter.export_spss_matrix_file()
        if measured_only:
            # 宽表暂无逐字段 provenance；并行落盘实测长表供科研过滤
            records = _load_global_records()
            long_result = academic_exporter.export_academic_matrix(
                records, measured_only=True
            )
            if long_result.get("success"):
                result = {
                    **result,
                    "longFormatPath": long_result.get("path"),
                    "longFormatFilename": long_result.get("filename"),
                    "longFormatRowCount": long_result.get("rowCount"),
                    "measuredOnly": True,
                }
    except Exception as exc:  # noqa: BLE001
        safe_print(f"【api_server】导出 V3.1 科研宽表失败：{exc}")
        # 回退：旧成长表，避免教练端完全无法导出
        records = _load_global_records()
        try:
            result = academic_exporter.export_academic_matrix(
                records, measured_only=measured_only
            )
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
        "measuredOnly": bool(result.get("measuredOnly", measured_only)),
        "longFormatPath": result.get("longFormatPath"),
        "longFormatFilename": result.get("longFormatFilename"),
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
    model_config = ConfigDict(extra="ignore")

    school: str = ""
    classGroup: str = ""
    # 键为错误分类标签，值为该分类在全班记录中的出现百分比（0-100）——必须 float
    errorStats: dict[str, float] = Field(default_factory=dict)
    # 记录总数偶发以 12.0 传入；用 float 接收，下游再 int(round(...))
    totalRecords: float = 0.0
    avgScore: Optional[float] = None

    @field_validator("errorStats", mode="before")
    @classmethod
    def _coerce_error_stats(cls, value):
        return _coerce_float_dict_soft(value, default_empty=True) or {}

    @field_validator("totalRecords", "avgScore", mode="before")
    @classmethod
    def _coerce_class_metrics(cls, value, info):
        coerced = _coerce_optional_float_soft(value)
        if info.field_name == "totalRecords":
            return 0.0 if coerced is None else coerced
        return coerced


@app.post("/api/generate_class_prescription")
def generate_class_prescription_endpoint(payload: GenerateClassPrescriptionRequest):
    """✨「召唤 AI 生成全班改进教案」：基于该班级全部历史记录的生物力学错误
    分布统计，调用 DeepSeek 生成一份结构严谨的集体教学诊断简报 + 处方。
    """
    ai_result = llm_agent.generate_class_prescription(
        school=payload.school,
        class_group=payload.classGroup,
        error_stats=payload.errorStats,
        total_records=int(round(float(payload.totalRecords or 0))),
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
    model_config = ConfigDict(extra="ignore")

    studentId: str = ""
    scoreHistory: list[float] = Field(default_factory=list)
    # 禁止 dict[str, int]：聚合占比/均值可能带小数 → 一律 float
    errorCounter: dict[str, float] = Field(default_factory=dict)

    @field_validator("scoreHistory", mode="before")
    @classmethod
    def _coerce_score_history(cls, value):
        coerced = _coerce_float_list_soft(value)
        return [] if coerced is None else coerced

    @field_validator("errorCounter", mode="before")
    @classmethod
    def _coerce_error_counter(cls, value):
        return _coerce_float_dict_soft(value, default_empty=True) or {}


@app.post("/api/generate_individual_summary")
def generate_individual_summary_endpoint(payload: GenerateIndividualSummaryRequest):
    """「个体纵向进化追踪」档案：基于该生全周期历史评分与错误分类统计，
    调用 DeepSeek 生成结构化的「稳定发力优势」与「需克服习惯性盲区」总结。
    """
    # LLM 侧按「次数」展示时取整；校验层保持 float 以免 422
    error_counter_int = {
        str(k): int(round(float(v))) for k, v in (payload.errorCounter or {}).items()
    }
    ai_result = llm_agent.generate_individual_summary(
        student_id=payload.studentId,
        score_history=payload.scoreHistory,
        error_counter=error_counter_int,
    )
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "strengths": ai_result["strengths"],
        "weaknesses": ai_result["weaknesses"],
        "generatedAt": generated_at,
    }


class OpenFolderRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str = ""


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


@app.get("/api/analytics/compare_cohorts")
def analytics_compare_cohorts(cohort_a: str = "", cohort_b: str = ""):
    """【班级/实验组对比】科研分析聚合。

    Query：
        cohort_a / cohort_b —— 班级/实验组名称（如「四年级1班-实验A组」）

    返回三维对比：
      ① trend：按日期 average_score + score_variance（波动区间）
      ② radar：五维（助跑/支撑/后摆/踝锁/鞭打）均值
      ③ error_rates：高频 ERR_* 发生占比

    样本为空或不足时 ``sufficient_data=False``，前端渲染「暂无足够数据对比」。
    """
    try:
        from db import compare_cohorts

        payload = compare_cohorts(
            cohort_a,
            cohort_b,
            records=_active_global_records(),
        )
        return payload
    except Exception as exc:  # noqa: BLE001
        safe_print(f"【api_server】compare_cohorts 失败：{exc}")
        try:
            from db import empty_compare_cohorts_payload

            empty = empty_compare_cohorts_payload(
                (cohort_a or "").strip(),
                (cohort_b or "").strip(),
                message="暂无足够数据对比",
            )
            empty["success"] = False
            return empty
        except Exception:  # noqa: BLE001
            return {
                "success": False,
                "sufficient_data": False,
                "message": "暂无足够数据对比",
                "cohort_a": (cohort_a or "").strip(),
                "cohort_b": (cohort_b or "").strip(),
                "sample_counts": {"a": 0, "b": 0},
                "trend": {"dates": [], "cohort_a": [], "cohort_b": []},
                "radar": {
                    "dimensions": ["助跑", "支撑", "后摆", "踝锁", "鞭打"],
                    "keys": [
                        "approach_rhythm",
                        "support_stability",
                        "backswing_folding",
                        "ankle_rigidity",
                        "whipping_velocity",
                    ],
                    "cohort_a": [None, None, None, None, None],
                    "cohort_b": [None, None, None, None, None],
                    "cohort_a_scores": {},
                    "cohort_b_scores": {},
                },
                "error_rates": {"cohort_a": [], "cohort_b": [], "union_codes": []},
            }


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

    model_config = ConfigDict(extra="ignore")

    imageBase64: str = ""
    attemptId: Optional[str] = None
    studentNumber: Optional[str] = None
    studentId: Optional[str] = None
    comment: Optional[str] = ""
    scores: Optional[dict[str, float]] = Field(default_factory=dict)
    radar_scores: Optional[dict[str, float]] = Field(default_factory=dict)
    task_status: Optional[str] = None


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
