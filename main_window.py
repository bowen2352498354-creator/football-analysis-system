# -*- coding: utf-8 -*-
"""
main_window.py
【控制器层】桌面端 GUI 主线程 —— 实验组别路由器 + 时空胶囊模态 + 灾备提示。

【V3.1 Sprint 4】所有视频采集 / MediaPipe / 运动学清洗 / 触球锁帧 / 确定性打分
均在 ``workers.inference_worker.InferenceWorker`` 从线程中完成；本窗口仅通过
``frame_ready_signal`` / ``diagnostics_ready_signal`` / ``camera_lost_signal``
无锁总线被动更新 UI，并按 Active_Group 条件接线。主线程严禁 ``cv2.read()``
或任何推理调用。

【Sprint 5】GROUP_B 主画面保持黑屏静默；右下角 Ghost Monitor 画中画消费
``frame_ready_signal`` 实时帧 + FPS，证明引擎存活但不干扰场上学员。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtGui import QDesktopServices, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QFrame,
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

from experimental_group_router import (
    GROUP_A,
    GROUP_B,
    GROUP_C,
    ExperimentalGroupRouter,
)
from pose_tracker import (
    B_GROUP_LOG_PATH,
    B_GROUP_WIDE_TABLE_PATH,
    CAPSULE_HALF_SPEED_FACTOR,
    DEFAULT_VIDEO_FILE_PATH,
    SCRIPT_DIR,
    ensure_model_downloaded,
)
from session_checkpoint import (
    SessionCheckpointStore,
    SessionSnapshot,
    default_store,
)
from workers.inference_worker import InferenceWorker


_FATIGUE_REASON_TITLES = {
    "ANKLE_FATIGUE": "疲劳预警 · 脚踝卸力",
    "KNEE_STIFFNESS": "疲劳预警 · 支撑腿僵直",
}


class SpaceTimeCapsuleOverlay(QFrame):
    """GROUP_A：阻塞式时空胶囊模态 —— AIGC 字幕 + 半速回放，结束后自动关闭。"""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("SpaceTimeCapsuleOverlay")
        self.setStyleSheet(
            "#SpaceTimeCapsuleOverlay {"
            "  background-color: rgba(8, 12, 20, 230);"
            "}"
            "QLabel#CapsuleTitle { color: #f5f5f5; font-size: 22px; font-weight: 700; }"
            "QLabel#CapsuleSubtitle { color: #ffe082; font-size: 28px; font-weight: 600; }"
            "QLabel#CapsuleHint { color: #90a4ae; font-size: 13px; }"
        )
        self.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(16)

        self.title_label = QLabel("时空胶囊 · 即时反馈")
        self.title_label.setObjectName("CapsuleTitle")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.replay_label = QLabel()
        self.replay_label.setAlignment(Qt.AlignCenter)
        self.replay_label.setMinimumHeight(360)
        self.replay_label.setStyleSheet(
            "background-color: #111; border: 1px solid #455a64;"
        )

        self.subtitle_label = QLabel("")
        self.subtitle_label.setObjectName("CapsuleSubtitle")
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setWordWrap(True)

        self.hint_label = QLabel("半速回放中……结束后将自动恢复摄像头")
        self.hint_label.setObjectName("CapsuleHint")
        self.hint_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.title_label)
        layout.addWidget(self.replay_label, stretch=1)
        layout.addWidget(self.subtitle_label)
        layout.addWidget(self.hint_label)

        self._frames: list = []
        self._frame_index = 0
        self._replay_timer = QTimer(self)
        self._replay_timer.timeout.connect(self._on_replay_tick)
        self._close_timer = QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.timeout.connect(self._finish)
        self._on_finished = None

    def start_capsule(self, payload: dict, on_finished) -> None:
        self._on_finished = on_finished
        # 接管回放帧所有权后立刻从 payload 弹出，避免双持有导致 OOM
        self._frames = list(payload.pop("replay_frames", None) or [])
        self._frame_index = 0
        subtitle = str(payload.get("subtitle") or "触球瞬间已锁定")
        self.subtitle_label.setText(subtitle)
        duration = float(payload.get("duration_sec") or 10.0)
        duration = max(8.0, min(12.0, duration))
        fps = float(payload.get("fps") or 30.0)
        interval_ms = int(
            max(40, round(1000.0 / max(fps, 1.0) * float(CAPSULE_HALF_SPEED_FACTOR)))
        )

        self.setGeometry(self.parent().rect())
        self.raise_()
        self.show()

        if self._frames:
            self._show_frame(self._frames[0])
            self._replay_timer.start(interval_ms)
        else:
            self.replay_label.setText("（本轮无回放帧，仅展示 AIGC 字幕）")

        self.hint_label.setText(f"半速回放中……约 {duration:.0f} 秒后自动恢复摄像头")
        self._close_timer.start(int(duration * 1000))

    def _show_frame(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            return
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
            return
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        h, w, c = rgb.shape
        qimage = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.replay_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.replay_label.setPixmap(pixmap)

    def _on_replay_tick(self) -> None:
        if not self._frames:
            return
        self._frame_index = (self._frame_index + 1) % len(self._frames)
        self._show_frame(self._frames[self._frame_index])

    def _finish(self) -> None:
        self._replay_timer.stop()
        self.hide()
        cb = self._on_finished
        self._on_finished = None
        # 切断回放矩阵引用，交 GC（防多轮胶囊堆积）
        self._frames.clear()
        self._frames = []
        self.replay_label.clear()
        if callable(cb):
            cb()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.isVisible() and self.parent() is not None:
            self.setGeometry(self.parent().rect())


class MainWindow(QMainWindow):
    """主窗口控制器：按 Active_Group 条件渲染与信号接线。"""

    def __init__(
        self,
        *,
        checkpoint_store: SessionCheckpointStore | None = None,
        pending_recovery: SessionSnapshot | None = None,
    ):
        super().__init__()
        self.setWindowTitle(
            "小学足球AI可视化反馈系统 v0.4 —— 实验组别路由训练终端"
        )
        self.resize(1280, 800)
        self.video_worker: InferenceWorker | None = None
        self.group_router = ExperimentalGroupRouter(GROUP_A)
        self._capsule_active = False
        self._checkpoint_store = checkpoint_store or default_store
        self._pending_recovery: SessionSnapshot | None = pending_recovery
        # Sprint 5：Ghost Monitor 画中画可见性 + FPS 滑动窗口
        self._ghost_monitor_visible = True
        self._fps_timestamps: list[float] = []
        self._build_ui()
        self._apply_group_ui_policy()
        if self._pending_recovery is not None:
            self._prefill_from_recovery(self._pending_recovery)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QHBoxLayout(central_widget)

        control_panel = self._build_control_panel()
        control_panel.setFixedWidth(340)
        root_layout.addWidget(control_panel)

        self.video_host = QWidget()
        video_host_layout = QVBoxLayout(self.video_host)
        video_host_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel("请选择数据源与实验组别后，点击「开始训练」")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(880, 660)
        self.video_label.setStyleSheet(
            "background-color: #202020; color: #cccccc; font-size: 16px; "
            "border: 1px solid #444;"
        )
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_host_layout.addWidget(self.video_label)
        root_layout.addWidget(self.video_host, stretch=1)

        self.capsule_overlay = SpaceTimeCapsuleOverlay(self.video_host)

        # 【Sprint 5】GROUP_B Ghost Monitor PiP：绝对定位于 video_host 右下角
        self.camera_lost_banner = QLabel(self.video_host)
        self.camera_lost_banner.setAlignment(Qt.AlignCenter)
        self.camera_lost_banner.setStyleSheet(
            "QLabel {"
            "  background-color: rgba(180, 120, 20, 210);"
            "  color: #fff8e1;"
            "  font-size: 12px;"
            "  padding: 6px 14px;"
            "  border-radius: 8px;"
            "  border: 1px solid #ffb74d;"
            "}"
        )
        self.camera_lost_banner.hide()

        self.ghost_toggle_btn = QPushButton("[隐藏监控]", self.video_host)
        self.ghost_toggle_btn.setFixedHeight(24)
        self.ghost_toggle_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: rgba(0,0,0,170);"
            "  color: #80cbc4;"
            "  font-size: 10px;"
            "  border: 1px solid #455a64;"
            "  border-radius: 4px;"
            "  padding: 2px 8px;"
            "}"
            "QPushButton:hover { border-color: #26a69a; color: #b2dfdb; }"
        )
        self.ghost_toggle_btn.clicked.connect(self._toggle_ghost_monitor)
        self.ghost_toggle_btn.hide()

        self.ghost_pip_label = QLabel(self.video_host)
        self.ghost_pip_label.setAlignment(Qt.AlignCenter)
        self.ghost_pip_label.setFixedSize(280, 158)
        self.ghost_pip_label.setStyleSheet(
            "QLabel {"
            "  background-color: #0a0a0a;"
            "  color: #666;"
            "  font-size: 10px;"
            "  border: 1px solid #26a69a;"
            "}"
        )
        self.ghost_pip_label.setText("Ghost Monitor\n等待推流…")
        self.ghost_pip_label.hide()

        self.ghost_fps_label = QLabel("-- FPS", self.video_host)
        self.ghost_fps_label.setStyleSheet(
            "QLabel {"
            "  background-color: rgba(0,0,0,200);"
            "  color: #4db6ac;"
            "  font-size: 9px;"
            "  font-weight: 700;"
            "  padding: 2px 6px;"
            "  border-radius: 3px;"
            "}"
        )
        self.ghost_fps_label.hide()

        QTimer.singleShot(0, self._layout_ghost_monitor)

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)

        source_group_box = QGroupBox("① 数据源选择")
        source_layout = QVBoxLayout()
        self.radio_source_webcam = QRadioButton("摄像头（实时拍摄）")
        self.radio_source_video = QRadioButton("本地视频文件")
        self.radio_source_webcam.setChecked(True)
        self.source_button_group = QButtonGroup(self)
        self.source_button_group.addButton(self.radio_source_webcam)
        self.source_button_group.addButton(self.radio_source_video)

        video_path_layout = QHBoxLayout()
        self.video_path_edit = QLineEdit(DEFAULT_VIDEO_FILE_PATH)
        self.video_path_edit.setEnabled(False)
        self.browse_button = QPushButton("浏览…")
        self.browse_button.setEnabled(False)
        self.browse_button.clicked.connect(self._on_browse_video_file)
        video_path_layout.addWidget(self.video_path_edit)
        video_path_layout.addWidget(self.browse_button)
        self.radio_source_video.toggled.connect(self.video_path_edit.setEnabled)
        self.radio_source_video.toggled.connect(self.browse_button.setEnabled)

        source_layout.addWidget(self.radio_source_webcam)
        source_layout.addWidget(self.radio_source_video)
        source_layout.addLayout(video_path_layout)
        source_group_box.setLayout(source_layout)
        layout.addWidget(source_group_box)

        group_group_box = QGroupBox("② 实验组别（Active_Group）")
        group_layout = QVBoxLayout()
        self.radio_group_a = QRadioButton("GROUP_A —— 实时反馈 / 时空胶囊")
        self.radio_group_b = QRadioButton("GROUP_B —— 静默录制 / 课后报告")
        self.radio_group_c = QRadioButton("GROUP_C —— 常规对照")
        self.radio_group_a.setChecked(True)
        self.group_button_group = QButtonGroup(self)
        self.group_button_group.addButton(self.radio_group_a)
        self.group_button_group.addButton(self.radio_group_b)
        self.group_button_group.addButton(self.radio_group_c)
        self.radio_group_a.toggled.connect(self._on_group_radio_changed)
        self.radio_group_b.toggled.connect(self._on_group_radio_changed)
        self.radio_group_c.toggled.connect(self._on_group_radio_changed)
        group_layout.addWidget(self.radio_group_a)
        group_layout.addWidget(self.radio_group_b)
        group_layout.addWidget(self.radio_group_c)
        group_group_box.setLayout(group_layout)
        layout.addWidget(group_group_box)

        identity_box = QGroupBox("③ 学员身份（断点续传）")
        identity_layout = QVBoxLayout()
        self.class_group_edit = QLineEdit()
        self.class_group_edit.setPlaceholderText("班级，例如：四年级2班")
        self.student_name_edit = QLineEdit()
        self.student_name_edit.setPlaceholderText("姓名，例如：张三")
        self.student_id_edit = QLineEdit()
        self.student_id_edit.setPlaceholderText("学号（可选），例如：B004")
        identity_layout.addWidget(QLabel("班级"))
        identity_layout.addWidget(self.class_group_edit)
        identity_layout.addWidget(QLabel("姓名"))
        identity_layout.addWidget(self.student_name_edit)
        identity_layout.addWidget(QLabel("学号"))
        identity_layout.addWidget(self.student_id_edit)
        identity_box.setLayout(identity_layout)
        layout.addWidget(identity_box)

        self.start_button = QPushButton("▶ 开始训练")
        self.start_button.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; font-size: 16px; "
            "padding: 10px; border-radius: 6px; } "
            "QPushButton:disabled { background-color: #777; }"
        )
        self.start_button.clicked.connect(self._on_start_clicked)

        self.stop_button = QPushButton("■ 结束训练")
        self.stop_button.setStyleSheet(
            "QPushButton { background-color: #c62828; color: white; font-size: 16px; "
            "padding: 10px; border-radius: 6px; } "
            "QPushButton:disabled { background-color: #777; }"
        )
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._on_stop_clicked)

        self.offline_report_button = QPushButton("离线课后报告查看")
        self.offline_report_button.setStyleSheet(
            "QPushButton { background-color: #1565c0; color: white; font-size: 14px; "
            "padding: 8px; border-radius: 6px; } "
            "QPushButton:disabled { background-color: #777; }"
        )
        self.offline_report_button.clicked.connect(self._on_open_offline_report)

        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.offline_report_button)

        log_group_box = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setStyleSheet("font-size: 12px;")
        log_layout.addWidget(self.log_text_edit)
        log_group_box.setLayout(log_layout)
        layout.addWidget(log_group_box, stretch=1)

        return panel

    def _selected_experimental_group(self) -> str:
        if self.radio_group_a.isChecked():
            return GROUP_A
        if self.radio_group_b.isChecked():
            return GROUP_B
        return GROUP_C

    def _on_group_radio_changed(self, checked: bool = False) -> None:
        del checked
        if self.video_worker is not None and self.video_worker.isRunning():
            return
        self.group_router.set_active_group(self._selected_experimental_group())
        self._apply_group_ui_policy()

    def _apply_group_ui_policy(self) -> None:
        policy = self.group_router.policy()
        self.offline_report_button.setVisible(policy.show_offline_report_button)
        self.offline_report_button.setEnabled(policy.show_offline_report_button)

    # ------------------------------------------------------------------
    # 事件
    # ------------------------------------------------------------------

    def _on_browse_video_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择本地视频文件",
            SCRIPT_DIR,
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)",
        )
        if file_path:
            self.video_path_edit.setText(file_path)

    def _prefill_from_recovery(self, snap: SessionSnapshot) -> None:
        """恢复对话框确认后：把班级/姓名填回表单，并提示老师点开始训练。"""
        if snap.class_group:
            self.class_group_edit.setText(snap.class_group)
        if snap.student_name:
            self.student_name_edit.setText(snap.student_name)
        if snap.student_id:
            self.student_id_edit.setText(snap.student_id)
        eg = str(snap.experimental_group or "")
        if eg == GROUP_B:
            self.radio_group_b.setChecked(True)
        elif eg == GROUP_C:
            self.radio_group_c.setChecked(True)
        else:
            self.radio_group_a.setChecked(True)
        self.group_router.set_active_group(self._selected_experimental_group())
        self._apply_group_ui_policy()
        if snap.data_source == "video" and snap.video_path:
            self.radio_source_video.setChecked(True)
            self.video_path_edit.setText(snap.video_path)
        self._append_log(
            f"待恢复：{snap.display_label()}，已完成 {snap.attempt_count} 次射门。"
            "点击「开始训练」即可无缝续课。"
        )

    def _on_start_clicked(self) -> None:
        if self.video_worker is not None and self.video_worker.isRunning():
            return

        data_source = "webcam" if self.radio_source_webcam.isChecked() else "video"
        video_path = self.video_path_edit.text().strip()
        class_group = self.class_group_edit.text().strip()
        student_name = self.student_name_edit.text().strip()
        student_id = self.student_id_edit.text().strip()

        # 启动任务前注入 experimental_group / Active_Group
        experimental_group = self.group_router.set_active_group(
            self._selected_experimental_group()
        )
        policy = self.group_router.policy()
        self._apply_group_ui_policy()

        if data_source == "video" and not os.path.exists(video_path):
            QMessageBox.warning(
                self, "文件不存在", f"未找到本地视频文件：\n{video_path}"
            )
            return

        restore_snap = self._pending_recovery
        if restore_snap is not None:
            self._append_log(
                f"热重启恢复：{restore_snap.display_label()}，"
                f"attempts_history={len(restore_snap.attempts_history)}，"
                f"attempt_count={restore_snap.attempt_count}"
            )

        self._append_log(
            f"开始训练：数据源="
            f"{'摄像头' if data_source == 'webcam' else '本地视频'}，"
            f"Active_Group={experimental_group}，"
            f"学员={class_group or '未设班级'}-"
            f"{student_name or student_id or '未命名'}"
        )

        short_group = {"GROUP_A": "A", "GROUP_B": "B", "GROUP_C": "C"}[
            experimental_group
        ]
        self.video_worker = InferenceWorker(
            data_source=data_source,
            video_path=video_path,
            group=short_group,
            experimental_group=experimental_group,
            class_group=class_group,
            student_id=student_id,
            student_name=student_name,
            checkpoint_store=self._checkpoint_store,
            restore_snapshot=restore_snap,
        )
        # 恢复快照只消费一次
        self._pending_recovery = None

        # 【Sprint 4】无锁信号总线：主线程只 connect，收到数据后立刻刷新控件
        self.video_worker.frame_ready_signal.connect(self._on_frame_ready)
        self.video_worker.error_occurred_signal.connect(self._on_worker_error)
        self.video_worker.camera_lost_signal.connect(self._on_camera_lost)
        self.video_worker.log_message.connect(self._append_log)
        self.video_worker.capture_discarded_signal.connect(self._on_capture_discarded)
        self.video_worker.finished.connect(self._on_worker_finished)

        if policy.connect_impact_lock:
            self.video_worker.diagnostics_ready_signal.connect(
                self._on_diagnostics_ready
            )
            self._append_log(
                "路由：已连接 diagnostics_ready_signal → 时空胶囊 / 8 大量纲"
            )
        else:
            self._append_log("路由：未连接诊断信号（非 GROUP_A）")

        # 课堂大脑：若非恢复模式则清空时序；恢复模式已在 Worker 内还原
        if restore_snap is None:
            self.video_worker.fatigue_monitor.reset_session(
                student_id=student_id or None
            )
        self.video_worker.fatigue_warning_signal.connect(self._on_fatigue_warning)
        self._append_log("路由：已连接 fatigue_warning_signal → 疲劳变形预警")

        if policy.silent_recording:
            self._append_log("路由：GROUP_B 静默录制，骨骼指示已屏蔽")
            self._ensure_b_group_black_canvas()
            self._show_ghost_monitor_chrome(True)
            self._fps_timestamps.clear()
        else:
            self._show_ghost_monitor_chrome(False)
            self.video_label.setStyleSheet(
                "background-color: #202020; color: #cccccc; font-size: 16px; "
                "border: 1px solid #444;"
            )

        self.video_worker.start()
        self._set_controls_enabled(is_training=True)

    def _on_stop_clicked(self) -> None:
        if self.video_worker is not None:
            self._append_log("正在请求结束训练，请稍候……")
            # 老师主动结束 → 标记 clean exit，避免下次误弹恢复框
            self.video_worker.mark_clean_exit()
            self.video_worker.request_stop()
        self.stop_button.setEnabled(False)

    def _on_worker_finished(self) -> None:
        self._capsule_active = False
        self._set_controls_enabled(is_training=False)
        self._show_ghost_monitor_chrome(False)
        self.camera_lost_banner.hide()
        self._fps_timestamps.clear()
        self.video_label.clear()
        self.video_label.setStyleSheet(
            "background-color: #202020; color: #cccccc; font-size: 16px; "
            "border: 1px solid #444;"
        )
        self.video_label.setText(
            "训练已结束，请选择数据源与实验组别后，点击「开始训练」"
        )

    def _on_open_offline_report(self) -> None:
        """GROUP_B 入口：打开本地宽表 / JSON / 报告图（按存在优先级）。"""
        candidates = [
            os.path.join(SCRIPT_DIR, "B_group_training_report.png"),
            B_GROUP_WIDE_TABLE_PATH,
            B_GROUP_LOG_PATH,
        ]
        target = next((p for p in candidates if os.path.exists(p)), None)
        if target is None:
            # 尝试即时生成报告
            try:
                import report_generator

                report_generator.generate_report()
                target = next((p for p in candidates if os.path.exists(p)), None)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.information(
                    self,
                    "暂无课后报告",
                    f"尚未找到 B 组离线数据。\n请先完成一次 GROUP_B 静默训练。\n\n{exc}",
                )
                return
        if target is None:
            QMessageBox.information(
                self,
                "暂无课后报告",
                "尚未找到 B 组离线数据。请先完成一次 GROUP_B 静默训练。",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(target)))
        self._append_log(f"已打开离线课后报告：{target}")

    # ------------------------------------------------------------------
    # 信号槽（仅 UI 刷新，无任何 cv2 / 推理）
    # ------------------------------------------------------------------

    def _on_frame_ready(self, frame_bgr: np.ndarray) -> None:
        """frame_ready_signal：立刻把脱敏骨骼画面刷到 QLabel。

        GROUP_B：主画面保持黑屏静默，实时帧仅刷入右下角 Ghost Monitor 画中画。
        """
        if self._capsule_active:
            return
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            return
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
            return

        # 收到有效帧 → 清除丢信号提示，并刷新 FPS
        if self.camera_lost_banner.isVisible():
            self.camera_lost_banner.hide()
        self._note_frame_for_fps()

        if self.group_router.Active_Group == GROUP_B:
            if self.video_label.pixmap() is not None or not self.video_label.text():
                self._ensure_b_group_black_canvas()
            if self._ghost_monitor_visible:
                self._paint_ndarray_to_label(self.ghost_pip_label, frame_bgr)
                self.ghost_pip_label.show()
                self.ghost_fps_label.show()
                self.ghost_toggle_btn.show()
                self._layout_ghost_monitor()
            return

        self._paint_ndarray_to_label(self.video_label, frame_bgr)

    def _paint_ndarray_to_label(self, label: QLabel, frame_bgr: np.ndarray) -> None:
        """把 BGR 帧缩放到目标 QLabel（槽返回后局部引用交 GC）。"""
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        qimage = QImage(
            rgb.data, width, height, bytes_per_line, QImage.Format_RGB888
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(
            label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        label.setPixmap(scaled)

    def _ensure_b_group_black_canvas(self) -> None:
        """B 组主视口固定黑屏文案，避免全尺寸实况暴露给场上学员。"""
        self.video_label.clear()
        self.video_label.setText("静默采集中，请专心练习")
        self.video_label.setStyleSheet(
            "background-color: #000000; color: #e0e0e0; font-size: 18px; "
            "border: 1px solid #333;"
        )

    def _note_frame_for_fps(self) -> None:
        now = time.time()
        self._fps_timestamps = [t for t in self._fps_timestamps if now - t < 1.0]
        self._fps_timestamps.append(now)
        self.ghost_fps_label.setText(f"{len(self._fps_timestamps)} FPS")

    def _toggle_ghost_monitor(self) -> None:
        self._ghost_monitor_visible = not self._ghost_monitor_visible
        self.ghost_toggle_btn.setText(
            "[隐藏监控]" if self._ghost_monitor_visible else "[显示监控]"
        )
        if self._ghost_monitor_visible and self.group_router.Active_Group == GROUP_B:
            self.ghost_pip_label.show()
            self.ghost_fps_label.show()
        else:
            self.ghost_pip_label.hide()
            self.ghost_fps_label.hide()
        self._layout_ghost_monitor()

    def _show_ghost_monitor_chrome(self, visible: bool) -> None:
        if visible and self.group_router.Active_Group == GROUP_B:
            self.ghost_toggle_btn.show()
            if self._ghost_monitor_visible:
                self.ghost_pip_label.show()
                self.ghost_fps_label.show()
            self._layout_ghost_monitor()
        else:
            self.ghost_toggle_btn.hide()
            self.ghost_pip_label.hide()
            self.ghost_fps_label.hide()

    def _layout_ghost_monitor(self) -> None:
        """把幽灵监控控件钉在 video_host 右下角（宽不超过 300px）。"""
        host = self.video_host
        if host is None:
            return
        margin = 14
        pip_w = min(280, max(160, host.width() - 40))
        pip_h = int(pip_w * 9 / 16)
        self.ghost_pip_label.setFixedSize(pip_w, pip_h)

        btn_w = self.ghost_toggle_btn.sizeHint().width() + 8
        btn_h = 24
        x = max(margin, host.width() - pip_w - margin)
        y_pip = max(margin, host.height() - pip_h - margin)
        y_btn = max(4, y_pip - btn_h - 4)

        self.ghost_toggle_btn.setGeometry(x + pip_w - btn_w, y_btn, btn_w, btn_h)
        self.ghost_pip_label.move(x, y_pip)
        self.ghost_fps_label.adjustSize()
        self.ghost_fps_label.move(x + 6, y_pip + 6)
        self.ghost_toggle_btn.raise_()
        self.ghost_pip_label.raise_()
        self.ghost_fps_label.raise_()

        banner_w = min(420, max(200, host.width() - 80))
        self.camera_lost_banner.adjustSize()
        self.camera_lost_banner.setFixedWidth(banner_w)
        self.camera_lost_banner.move(
            max(20, (host.width() - banner_w) // 2),
            48,
        )
        self.camera_lost_banner.raise_()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._layout_ghost_monitor()
        if hasattr(self, "capsule_overlay") and self.capsule_overlay.isVisible():
            self.capsule_overlay.setGeometry(self.video_host.rect())

    def _on_camera_lost(self, message: str) -> None:
        """camera_lost_signal：非致命提示，会话继续，等待 Worker 自愈重开。"""
        text = message or "摄像头信号丢失，尝试重连..."
        self._append_log(f"【摄像头自愈】{text}")
        self.camera_lost_banner.setText(text)
        self.camera_lost_banner.show()
        self.ghost_fps_label.setText("-- FPS")
        self._layout_ghost_monitor()

    def _on_diagnostics_ready(self, payload: dict) -> None:
        """diagnostics_ready_signal：射门锁定后的 8 大量纲 → 时空胶囊模态。"""
        if not isinstance(payload, dict):
            return
        if self.group_router.Active_Group != GROUP_A:
            return

        t_impact = payload.get("t_impact", "?")
        knee = payload.get("impact_knee_angle", "?")
        status = payload.get("knee_status", "?")
        score = payload.get("total_score", "?")
        self._append_log(
            f"【时空胶囊】t_impact=#{t_impact}  膝角={knee}° ({status})  "
            f"确定性总分={score}"
        )

        self._capsule_active = True
        if self.video_worker is not None:
            self.video_worker.pause_for_capsule()

        self.capsule_overlay.start_capsule(payload, on_finished=self._on_capsule_finished)

    # 兼容旧槽名（若外部仍按 Sprint 3 名称连接）
    _on_frame_processed = _on_frame_ready
    _on_impact_detected = _on_diagnostics_ready

    def _on_capsule_finished(self) -> None:
        self._capsule_active = False
        if self.video_worker is not None:
            self.video_worker.resume_after_capsule()
        self._append_log("时空胶囊结束，已恢复摄像头读取。")

    def _on_capture_discarded(self, message: str) -> None:
        """灾备提示：本轮捕获失败，教学不中断。"""
        text = message or "本轮捕获失败，请准备下一球"
        self._append_log(f"【Discard】{text}")
        # 显著但不阻塞整课：短暂覆盖提示
        if not self._capsule_active:
            self.video_label.setText(text)

    def _on_fatigue_warning(self, warning: dict) -> None:
        """FatigueMonitor 熔断：日志 + 非模态提示，建议教练叫停轮换。"""
        if not isinstance(warning, dict) or not warning.get("is_fatigue"):
            return
        reason = str(warning.get("reason") or "FATIGUE")
        message = str(
            warning.get("message")
            or "检测到疲劳导致的动作变形，建议立即叫停轮换！"
        )
        title = _FATIGUE_REASON_TITLES.get(reason, "疲劳预警")
        n = 0
        if self.video_worker is not None:
            n = len(self.video_worker.fatigue_monitor.attempts_history)
        self._append_log(f"【疲劳预警】第 {n} 脚 · {reason} · {message}")

        detail_bits = []
        if warning.get("baseline_mean") is not None and warning.get("recent_mean") is not None:
            detail_bits.append(
                f"基线均值={warning['baseline_mean']} → "
                f"近期均值={warning['recent_mean']} "
                f"(Δ={warning.get('delta', '?')})"
            )
        body = message if not detail_bits else f"{message}\n\n{'；'.join(detail_bits)}"
        QMessageBox.warning(self, title, body)

    def _on_worker_error(self, message: str) -> None:
        self._append_log(f"错误：{message}")

    def _append_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text_edit.append(f"[{timestamp}] {message}")

    def _set_controls_enabled(self, is_training: bool) -> None:
        self.start_button.setEnabled(not is_training)
        self.stop_button.setEnabled(is_training)
        self.radio_source_webcam.setEnabled(not is_training)
        self.radio_source_video.setEnabled(not is_training)
        self.radio_group_a.setEnabled(not is_training)
        self.radio_group_b.setEnabled(not is_training)
        self.radio_group_c.setEnabled(not is_training)
        self.class_group_edit.setEnabled(not is_training)
        self.student_name_edit.setEnabled(not is_training)
        self.student_id_edit.setEnabled(not is_training)
        self.browse_button.setEnabled(
            (not is_training) and self.radio_source_video.isChecked()
        )
        self.video_path_edit.setEnabled(
            (not is_training) and self.radio_source_video.isChecked()
        )
        # 训练中 GROUP_B 仍可查看离线报告入口
        policy = self.group_router.policy()
        self.offline_report_button.setEnabled(policy.show_offline_report_button)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "capsule_overlay") and self.capsule_overlay.isVisible():
            self.capsule_overlay.setGeometry(self.video_host.rect())

    def closeEvent(self, event) -> None:
        if self.video_worker is not None and self.video_worker.isRunning():
            # 关闭窗口视为老师主动结案，避免误报异常退出
            self.video_worker.mark_clean_exit()
            self.video_worker.request_stop()
            self.video_worker.wait(3000)
        event.accept()


def prompt_recovery_if_needed(
    store: SessionCheckpointStore | None = None,
    parent: QWidget | None = None,
) -> SessionSnapshot | None:
    """启动时优先检查未结案 Session；确认则返回快照，放弃则标 abandoned。"""
    checkpoint_store = store or default_store
    orphan = checkpoint_store.load_active_session()
    if orphan is None:
        return None

    reply = QMessageBox.question(
        parent,
        "教学进度恢复",
        orphan.recovery_prompt()
        + "\n\n选择「是」将还原已完成射门与疲劳监控状态；"
        "选择「否」将丢弃该断点并开始新课。",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.Yes,
    )
    if reply == QMessageBox.Yes:
        return orphan

    try:
        checkpoint_store.mark_abandoned(orphan.session_id)
    except Exception:  # noqa: BLE001
        pass
    return None


def main() -> None:
    """桌面端入口：确保模型就绪 → 检查灾难恢复 → 启动 Qt 事件循环。"""
    ensure_model_downloaded()
    app = QApplication(sys.argv)
    pending = prompt_recovery_if_needed(default_store)
    window = MainWindow(
        checkpoint_store=default_store,
        pending_recovery=pending,
    )
    window.show()
    if pending is not None:
        window._append_log(
            f"已准备恢复：{pending.recovery_prompt().replace('，是否恢复进度？', '')}"
        )
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
