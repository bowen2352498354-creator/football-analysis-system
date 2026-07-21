# -*- coding: utf-8 -*-
"""
session_monitor.py
课堂时序疲劳监控 —— 对比「体力充沛基线」与「近期状态」，识别疲劳导致的动作变形。

职责边界：
    - 维护单堂课内历次射门的 8 大量纲扁平记录；
    - 在样本足够（≥8 脚）后执行脚踝卸力 / 支撑腿僵直熔断判定；
    - 通过 ``fatigue_warning_signal`` 向主程序抛出报警字典（不阻塞打分主路径）；
    - 提供 ``export_state`` / ``restore_state``，供断点续传灾难恢复使用。
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from PyQt5.QtCore import QObject, pyqtSignal

from models.enums import BIOMECH_CORE_METRIC_KEYS

# ---------------------------------------------------------------------------
# 熔断阈值（与 DeterministicScorer 脚踝 RED 阈值 / 支撑膝绿带上沿对齐）
# ---------------------------------------------------------------------------
MIN_ATTEMPTS_FOR_MONITOR: int = 8
BASELINE_WINDOW: int = 3
RECENT_WINDOW: int = 3

ANKLE_SOFT_MEAN: float = 5.0  # Recent 脚踝刚性方差均值超过此值 → 严重松软
ANKLE_SOFT_DELTA: float = 2.0  # Recent − Baseline 至少抬升此值才判疲劳

KNEE_STIFF_MEAN: float = 165.0  # Recent 支撑膝角均值超过此值 → 无法下蹲缓冲
KNEE_STIFF_DELTA: float = 15.0  # Recent − Baseline 至少抬升此值才判僵直

FATIGUE_MESSAGES: dict[str, str] = {
    "ANKLE_FATIGUE": "脚踝发力出现松软卸力，建议立即叫停轮换！",
    "KNEE_STIFFNESS": "支撑腿膝关节僵直无法缓冲，下肢疲软迹象明显，建议立即叫停轮换！",
}


def flatten_eight_metrics(source: Mapping[str, Any] | None) -> dict[str, Optional[float]]:
    """从 score_detail / indicators / 已扁平字典中提取 8 大量纲标量。"""
    flat: dict[str, Optional[float]] = {key: None for key in BIOMECH_CORE_METRIC_KEYS}
    if not isinstance(source, Mapping):
        return flat

    # 优先 indicators 嵌套结构（DeterministicScorer detail）
    indicators = source.get("indicators") if "indicators" in source else None
    pool: Mapping[str, Any]
    if isinstance(indicators, Mapping):
        pool = indicators
    else:
        pool = source

    for key in BIOMECH_CORE_METRIC_KEYS:
        entry = pool.get(key)
        if isinstance(entry, Mapping):
            raw = entry.get("value")
            if raw is None and key == "ankle_rigidity":
                raw = entry.get("variance")
        else:
            raw = entry
        if isinstance(raw, (int, float)):
            flat[key] = float(raw)
        else:
            flat[key] = None
    return flat


def _mean_of(values: list[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


class FatigueMonitor(QObject):
    """单堂课疲劳与动作变形监控器。

    使用方式：
        1. 每次射门打分完毕调用 ``record_attempt(metrics)``（或 ``ingest_score_detail``）；
        2. 调用 ``check_fatigue_deformation()``；若命中熔断规则则自动 emit
           ``fatigue_warning_signal``，并返回同一报警字典。
    """

    fatigue_warning_signal = pyqtSignal(dict)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.attempts_history: list[dict] = []
        self.student_id: Optional[str] = None
        self.last_fatigue_warning: Optional[dict] = None

    # ------------------------------------------------------------------
    # 时序记录
    # ------------------------------------------------------------------

    def reset_session(self, student_id: Optional[str] = None) -> None:
        """清空本堂课时序记录（换人 / 新开一堂课时调用）。"""
        self.attempts_history.clear()
        self.student_id = student_id
        self.last_fatigue_warning = None

    def record_attempt(self, metrics: Mapping[str, Any]) -> dict[str, Optional[float]]:
        """将一次射门的 8 大量纲分析结果 append 进 ``attempts_history``。"""
        flat = flatten_eight_metrics(metrics)
        row = dict(flat)
        row["attempt_index"] = len(self.attempts_history) + 1
        if self.student_id is not None:
            row["student_id"] = self.student_id
        self.attempts_history.append(row)
        return flat

    def export_state(self) -> dict[str, Any]:
        """导出可序列化状态，供断点快照使用。"""
        warning = self.last_fatigue_warning
        return {
            "student_id": self.student_id,
            "attempts_history": [dict(row) for row in self.attempts_history],
            "fatigue_triggered": bool(warning and warning.get("is_fatigue")),
            "last_fatigue_warning": dict(warning) if isinstance(warning, dict) else None,
        }

    def restore_state(
        self,
        *,
        attempts_history: list[dict] | None = None,
        student_id: Optional[str] = None,
        last_fatigue_warning: Optional[dict] = None,
    ) -> None:
        """热重启：把 attempts_history / 疲劳预警重置回中断前时刻。"""
        self.attempts_history = [
            dict(row) for row in (attempts_history or []) if isinstance(row, dict)
        ]
        if student_id is not None:
            self.student_id = student_id
        self.last_fatigue_warning = (
            dict(last_fatigue_warning)
            if isinstance(last_fatigue_warning, dict)
            else None
        )

    def check_fatigue_deformation(self, *, emit_signal: bool = True) -> Optional[dict]:
        """基线 vs 近期对比；命中规则时可选 emit 报警并返回字典，否则返回 None。

        触发条件：``len(attempts_history) >= 8``。
        基线 = 前 3 脚；近期 = 最后 3 脚。

        emit_signal：
            True  —— 直接通过本对象的 ``fatigue_warning_signal`` 抛出（独立使用时）；
            False —— 仅返回字典，由调用方（如 VideoWorker）再经总线转发。
        """
        history = self.attempts_history
        if len(history) < MIN_ATTEMPTS_FOR_MONITOR:
            return None

        baseline = history[:BASELINE_WINDOW]
        recent = history[-RECENT_WINDOW:]

        warning = self._eval_ankle_fatigue(baseline, recent)
        if warning is None:
            warning = self._eval_knee_stiffness(baseline, recent)
        if warning is None:
            return None

        self.last_fatigue_warning = dict(warning)
        if emit_signal:
            try:
                self.fatigue_warning_signal.emit(dict(warning))
            except Exception:  # noqa: BLE001 — 信号异常不得中断教学主路径
                pass
        return warning

    def ingest_score_detail(
        self, score_detail: Mapping[str, Any] | None, *, emit_signal: bool = True
    ) -> Optional[dict]:
        """便捷入口：从 DeterministicScorer ``score_detail`` 写入并立即检查疲劳。"""
        if not isinstance(score_detail, Mapping):
            return None
        self.record_attempt(score_detail)
        return self.check_fatigue_deformation(emit_signal=emit_signal)

    @staticmethod
    def _eval_ankle_fatigue(
        baseline: list[dict], recent: list[dict]
    ) -> Optional[dict]:
        """规则 1：脚踝卸力熔断 → ANKLE_FATIGUE。"""
        base_mean = _mean_of([row.get("ankle_rigidity") for row in baseline])
        recent_mean = _mean_of([row.get("ankle_rigidity") for row in recent])
        if base_mean is None or recent_mean is None:
            return None
        if recent_mean > ANKLE_SOFT_MEAN and (recent_mean - base_mean) >= ANKLE_SOFT_DELTA:
            reason = "ANKLE_FATIGUE"
            return {
                "is_fatigue": True,
                "reason": reason,
                "message": FATIGUE_MESSAGES[reason],
                "baseline_mean": round(base_mean, 4),
                "recent_mean": round(recent_mean, 4),
                "delta": round(recent_mean - base_mean, 4),
                "metric": "ankle_rigidity",
            }
        return None

    @staticmethod
    def _eval_knee_stiffness(
        baseline: list[dict], recent: list[dict]
    ) -> Optional[dict]:
        """规则 2：支撑腿僵直熔断 → KNEE_STIFFNESS。"""
        base_mean = _mean_of([row.get("support_knee_angle") for row in baseline])
        recent_mean = _mean_of([row.get("support_knee_angle") for row in recent])
        if base_mean is None or recent_mean is None:
            return None
        if (
            recent_mean > KNEE_STIFF_MEAN
            and (recent_mean - base_mean) >= KNEE_STIFF_DELTA
        ):
            reason = "KNEE_STIFFNESS"
            return {
                "is_fatigue": True,
                "reason": reason,
                "message": FATIGUE_MESSAGES[reason],
                "baseline_mean": round(base_mean, 2),
                "recent_mean": round(recent_mean, 2),
                "delta": round(recent_mean - base_mean, 2),
                "metric": "support_knee_angle",
            }
        return None
