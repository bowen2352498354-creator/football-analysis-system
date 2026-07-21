# -*- coding: utf-8 -*-
"""小学足球 AI 可视化反馈系统 —— 桌面端入口。

启动后进入主/从线程分离架构（V3.1 Sprint 4）：
    主线程 = ``main_window.MainWindow``（仅 UI + 信号槽，严禁 cv2.read / 推理）
    从线程 = ``workers.inference_worker.InferenceWorker``（视频采集与全部 AI 计算）

启动序：优先检查本地未结案 Session 快照（断点续传 / 灾难恢复），
再进入 Qt 事件循环。
"""

from main_window import main

if __name__ == "__main__":
    main()
