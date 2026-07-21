# -*- coding: utf-8 -*-
"""
research_dashboard_service.py
教练端 / 科研控制台 —— 宏观监控与个案追踪业务逻辑

ResearchDashboardService 职责：
  1) 干预进度与缺失值监控：按 T 节点聚合各被试射门完成次数，
     标出偏离标准剂量 ±20% 的剂量异常名单；
  2) 极端个案捕捉（Purposive Sampling）：对比 T1→T2 八大生物力学
     综合得分斜率，自动抽出高反应者 / 低反应者，供现象学访谈抽样。

数据源优先级：
  ① 显式传入的 ShotAttemptLog 列表；
  ② 科研结构化 JSON（research_shot_logs.json，若存在）；
  ③ 兼容桥接 global_training_db.json（历史联调归档）。
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from datetime import date
from typing import Any, Optional

from research_models import (
    DOSE_TOLERANCE_RATIO,
    STANDARD_SHOT_DOSE,
    DoseAnomalySubject,
    ExperimentalGroup,
    ExtremeCaseSubject,
    ShotAttemptLog,
    StudentProfile,
    Timepoint,
)

# --------------------------------------------------------------------------
# Windows 控制台编码兼容（与 api_server / academic_exporter 同源防线）
# --------------------------------------------------------------------------
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    try:
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except (AttributeError, ValueError):
        pass


def _safe_print(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        try:
            print(message.encode(encoding, errors="replace").decode(encoding, errors="replace"))
        except Exception:
            pass


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GLOBAL_DB_PATH = os.path.join(SCRIPT_DIR, "global_training_db.json")
DEFAULT_RESEARCH_LOG_PATH = os.path.join(SCRIPT_DIR, "research_shot_logs.json")

# 极端个案抽样比例
EXTREME_PERCENTILE = 0.20

# 将历史归档 type 映射为实验组枚举
_TYPE_TO_GROUP = {
    "realtime": ExperimentalGroup.GROUP_A_REALTIME,
    "delayed": ExperimentalGroup.GROUP_B_DELAYED,
    "control": ExperimentalGroup.GROUP_C_CONTROL,
}


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(float(value)):
            return None
        return float(value)
    return None


def _parse_timepoint(raw: Any) -> Optional[Timepoint]:
    if raw is None:
        return None
    text = str(raw).strip().upper()
    if not text:
        return None
    if text.isdigit():
        text = f"T{text}"
    try:
        return Timepoint(text)
    except ValueError:
        return None


def _infer_timepoints_by_date(sorted_dates: list[date]) -> dict[date, Timepoint]:
    """将一名被试（或一个集群）按日历日排序的测试日，映射到 T0..T4。

    规则：第 1 个不同日期 → T0，第 2 个 → T1，……，超出 T4 的仍归入 T4。
    """
    mapping: dict[date, Timepoint] = {}
    ordered = sorted(set(sorted_dates))
    nodes = list(Timepoint)
    for index, day in enumerate(ordered):
        mapping[day] = nodes[min(index, len(nodes) - 1)]
    return mapping


class ResearchDashboardService:
    """教练端科研控制台业务服务：剂量监控 + 目的性抽样。"""

    def __init__(
        self,
        global_db_path: Optional[str] = None,
        research_log_path: Optional[str] = None,
        standard_dose: int = STANDARD_SHOT_DOSE,
        dose_tolerance_ratio: float = DOSE_TOLERANCE_RATIO,
    ):
        self.global_db_path = global_db_path or DEFAULT_GLOBAL_DB_PATH
        self.research_log_path = research_log_path or DEFAULT_RESEARCH_LOG_PATH
        self.standard_dose = int(standard_dose)
        self.dose_tolerance_ratio = float(dose_tolerance_ratio)
        self._shots: list[ShotAttemptLog] = []
        self._profiles: dict[str, StudentProfile] = {}
        self._sessions: list[TimepointSession] = []

    # ------------------------------------------------------------------
    # 数据装载
    # ------------------------------------------------------------------

    def load(self, shots: Optional[list[ShotAttemptLog]] = None) -> "ResearchDashboardService":
        """装载射门日志。未显式传入时，优先科研 JSON，再桥接全局训练库。"""
        if shots is not None:
            self._shots = list(shots)
        else:
            self._shots = self._load_research_logs()
            if not self._shots:
                self._shots = self._bridge_from_global_db()
        self._rebuild_profiles()
        return self

    def _load_json_list(self, path: str) -> list[dict]:
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for key in ("shots", "records", "logs"):
                    if isinstance(data.get(key), list):
                        return data[key]
            return []
        except Exception as exc:  # noqa: BLE001
            _safe_print(f"【ResearchDashboardService】读取 {path} 失败：{exc}")
            return []

    def _load_research_logs(self) -> list[ShotAttemptLog]:
        rows = self._load_json_list(self.research_log_path)
        shots: list[ShotAttemptLog] = []
        for row in rows:
            try:
                shots.append(ShotAttemptLog.model_validate(row))
            except Exception:
                continue
        return shots

    def _bridge_from_global_db(self) -> list[ShotAttemptLog]:
        """把 global_training_db.json 的归档记录桥接为 ShotAttemptLog。

        兼容策略：
          - studentId → anonymous_id
          - classGroup → cluster_id
          - type → experimental_group
          - score → composite_score
          - 若记录带 timepoint 字段则直接使用，否则按该生测试日期序推断 T0..Tn
          - 8 大量纲若缺失则尽量从同源字段回填，否则留空（综合分仍可用）
        """
        records = self._load_json_list(self.global_db_path)
        if not records:
            return []

        # 先按被试收集测试日期，再推断 timepoint
        dates_by_student: dict[str, list[date]] = defaultdict(list)
        normalized: list[dict] = []

        for raw in records:
            if not isinstance(raw, dict):
                continue
            anonymous_id = str(
                raw.get("anonymous_id") or raw.get("studentId") or raw.get("student_id") or ""
            ).strip()
            if not anonymous_id:
                continue

            date_text = (
                raw.get("session_date")
                or raw.get("testDate")
                or raw.get("test_date")
                or (str(raw.get("timestamp") or "")[:10])
            )
            try:
                session_date = date.fromisoformat(str(date_text)[:10])
            except ValueError:
                session_date = date.today()

            dates_by_student[anonymous_id].append(session_date)
            normalized.append({**raw, "_anonymous_id": anonymous_id, "_session_date": session_date})

        tp_maps = {
            sid: _infer_timepoints_by_date(days) for sid, days in dates_by_student.items()
        }

        shots: list[ShotAttemptLog] = []
        for raw in normalized:
            anonymous_id = raw["_anonymous_id"]
            session_date = raw["_session_date"]
            explicit_tp = _parse_timepoint(raw.get("timepoint") or raw.get("timePoint"))
            timepoint = explicit_tp or tp_maps[anonymous_id].get(session_date, Timepoint.T0)

            cluster_id = str(
                raw.get("cluster_id") or raw.get("classGroup") or raw.get("class_name") or "Unknown"
            ).strip() or "Unknown"

            group_raw = raw.get("experimental_group") or raw.get("type") or ""
            group: Optional[ExperimentalGroup] = None
            if isinstance(group_raw, ExperimentalGroup):
                group = group_raw
            else:
                key = str(group_raw).strip()
                if key in ExperimentalGroup.__members__:
                    group = ExperimentalGroup[key]
                else:
                    group = _TYPE_TO_GROUP.get(key.lower())

            impact_idx = raw.get("impact_frame_index")
            if impact_idx is None:
                impact_idx = raw.get("impactFrameIndex") or raw.get("t_impact") or 0
            try:
                impact_idx = int(impact_idx)
            except (TypeError, ValueError):
                impact_idx = 0

            metrics = raw.get("biomechanical_metrics") or raw.get("indicators") or {}
            if not isinstance(metrics, dict):
                metrics = {}

            def pick(*keys: str) -> Optional[float]:
                for key in keys:
                    if key in raw and _as_float(raw.get(key)) is not None:
                        return _as_float(raw.get(key))
                    nested = metrics.get(key)
                    if isinstance(nested, dict) and _as_float(nested.get("value")) is not None:
                        return _as_float(nested.get("value"))
                    if _as_float(nested) is not None:
                        return _as_float(nested)
                return None

            composite = _as_float(
                raw.get("composite_score")
                or raw.get("TotalScore")
                or raw.get("total_score")
                or raw.get("score")
            )

            # 膝角：历史库常用 kneeFlexionAngle
            impact_knee = pick("impact_knee_angle", "kneeFlexionAngle", "knee_flexion_angle")
            distance = pick("distance_cm", "supportFootDistance", "support_foot_distance")

            payload = {
                "anonymous_id": anonymous_id,
                "session_date": session_date,
                "timepoint": timepoint,
                "impact_frame_index": max(0, impact_idx),
                "cluster_id": cluster_id,
                "experimental_group": group,
                "distance_cm": distance,
                "toe_angle": pick("toe_angle"),
                "max_folding_angle": pick("max_folding_angle"),
                "whipping_velocity": pick("whipping_velocity"),
                "impact_knee_angle": impact_knee,
                "ankle_rigidity": pick("ankle_rigidity", "ankle_rigidity_variance"),
                "support_knee_angle": pick("support_knee_angle"),
                "hip_torsion_angle": pick("hip_torsion_angle"),
                "composite_score": composite,
            }
            try:
                shots.append(ShotAttemptLog.model_validate(payload))
            except Exception as exc:  # noqa: BLE001
                _safe_print(f"【ResearchDashboardService】跳过无法解析的归档记录：{exc}")
                continue

        _safe_print(
            f"【ResearchDashboardService】已从 global_training_db 桥接 {len(shots)} 条射门日志。"
        )
        return shots

    def _rebuild_profiles(self) -> None:
        profiles: dict[str, StudentProfile] = {}
        for shot in self._shots:
            if shot.anonymous_id in profiles:
                continue
            group = shot.experimental_group or ExperimentalGroup.GROUP_A_REALTIME
            try:
                profiles[shot.anonymous_id] = StudentProfile(
                    anonymous_id=shot.anonymous_id,
                    cluster_id=shot.cluster_id or "Unknown",
                    experimental_group=group,
                )
            except Exception:
                continue
        self._profiles = profiles

    # ------------------------------------------------------------------
    # 1) 干预进度与缺失值 / 剂量异常监控
    # ------------------------------------------------------------------

    def dose_band(self, standard_dose: Optional[int] = None) -> tuple[float, float]:
        dose = int(standard_dose if standard_dose is not None else self.standard_dose)
        low = dose * (1.0 - self.dose_tolerance_ratio)
        high = dose * (1.0 + self.dose_tolerance_ratio)
        return low, high

    def aggregate_shot_counts(
        self,
        timepoint: Optional[str] = None,
        cluster_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """按被试聚合指定 T 节点（或当前全部）的射门完成次数。"""
        tp = _parse_timepoint(timepoint) if timepoint else None
        cluster_filter = (cluster_id or "").strip() or None

        counts: dict[str, dict[str, Any]] = {}
        for shot in self._shots:
            if tp is not None and shot.timepoint != tp:
                continue
            if cluster_filter and shot.cluster_id != cluster_filter:
                continue
            bucket = counts.setdefault(
                shot.anonymous_id,
                {
                    "anonymous_id": shot.anonymous_id,
                    "cluster_id": shot.cluster_id,
                    "experimental_group": (
                        shot.experimental_group.value if shot.experimental_group else None
                    ),
                    "timepoint": (tp.value if tp else "ALL"),
                    "shot_count": 0,
                },
            )
            bucket["shot_count"] += 1
            # 若未指定单一 T，标注该生实际覆盖的节点集合（逗号拼接，便于前端展示）
            if tp is None:
                existing = bucket.get("_tps") or set()
                existing.add(shot.timepoint.value)
                bucket["_tps"] = existing

        rows: list[dict[str, Any]] = []
        for item in counts.values():
            tps = item.pop("_tps", None)
            if tps is not None:
                item["timepoints_covered"] = sorted(tps)
            rows.append(item)
        rows.sort(key=lambda r: (r.get("cluster_id") or "", r["anonymous_id"]))
        return rows

    def get_progress_monitor(
        self,
        timepoint: Optional[str] = None,
        cluster_id: Optional[str] = None,
        standard_dose: Optional[int] = None,
    ) -> dict[str, Any]:
        """干预进度监控主入口。

        返回：
          - subjects: 全员射门次数明细
          - mean_shot_count: 组内均值
          - standard_dose / dose_low / dose_high
          - dose_anomalies: 偏离标准剂量 ±20% 的异常名单
        """
        dose = int(standard_dose if standard_dose is not None else self.standard_dose)
        low, high = self.dose_band(dose)
        subjects = self.aggregate_shot_counts(timepoint=timepoint, cluster_id=cluster_id)

        counts = [int(s["shot_count"]) for s in subjects]
        mean_shot = round(sum(counts) / len(counts), 2) if counts else 0.0

        anomalies: list[dict[str, Any]] = []
        for subject in subjects:
            shot_count = int(subject["shot_count"])
            if shot_count < low:
                anomaly_type = "under_dose"
            elif shot_count > high:
                anomaly_type = "over_dose"
            else:
                continue
            deviation_ratio = (shot_count - dose) / dose if dose > 0 else 0.0
            anomaly = DoseAnomalySubject(
                anonymous_id=subject["anonymous_id"],
                cluster_id=subject.get("cluster_id") or "",
                timepoint=str(subject.get("timepoint") or timepoint or "ALL"),
                shot_count=shot_count,
                standard_dose=dose,
                dose_low=round(low, 2),
                dose_high=round(high, 2),
                deviation_ratio=round(deviation_ratio, 4),
                anomaly_type=anomaly_type,
            )
            anomalies.append(anomaly.model_dump())

        anomalies.sort(key=lambda a: abs(a["deviation_ratio"]), reverse=True)

        resolved_tp = (_parse_timepoint(timepoint).value if timepoint and _parse_timepoint(timepoint) else None)

        return {
            "success": True,
            "timepoint": resolved_tp or (timepoint or "ALL"),
            "cluster_id": cluster_id or "ALL",
            "standard_dose": dose,
            "dose_tolerance_ratio": self.dose_tolerance_ratio,
            "dose_low": round(low, 2),
            "dose_high": round(high, 2),
            "subject_count": len(subjects),
            "mean_shot_count": mean_shot,
            "subjects": subjects,
            "dose_anomalies": anomalies,
            "anomaly_count": len(anomalies),
        }

    # ------------------------------------------------------------------
    # 2) 极端个案捕捉（Purposive Sampling Extractor）
    # ------------------------------------------------------------------

    def _subject_timepoint_composite(
        self,
        cluster_id: Optional[str] = None,
    ) -> dict[str, dict[str, float]]:
        """计算每名被试在各 T 节点的 8 大生物力学综合得分均值。

        综合分优先使用 composite_score；若缺失，则对可用的 8 大量纲做
        简单归一化均值回退（仅作联调兜底，正式实验应以 DeterministicScorer 为准）。
        """
        cluster_filter = (cluster_id or "").strip() or None
        buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for shot in self._shots:
            if cluster_filter and shot.cluster_id != cluster_filter:
                continue
            score = shot.composite_score
            if score is None:
                score = self._fallback_composite_from_metrics(shot)
            if score is None:
                continue
            buckets[shot.anonymous_id][shot.timepoint.value].append(float(score))

        result: dict[str, dict[str, float]] = {}
        for anonymous_id, tp_map in buckets.items():
            result[anonymous_id] = {
                tp: round(sum(vals) / len(vals), 2) for tp, vals in tp_map.items() if vals
            }
        return result

    @staticmethod
    def _fallback_composite_from_metrics(shot: ShotAttemptLog) -> Optional[float]:
        """当没有 composite_score 时，用已有量纲的简易启发式合成分（0–100）。"""
        values = []
        # 膝角越接近 150 越好
        if shot.impact_knee_angle is not None:
            values.append(max(0.0, 100.0 - abs(shot.impact_knee_angle - 150.0) * 2.0))
        if shot.support_knee_angle is not None:
            values.append(max(0.0, 100.0 - abs(shot.support_knee_angle - 152.5) * 2.0))
        if shot.distance_cm is not None:
            values.append(max(0.0, 100.0 - abs(shot.distance_cm - 17.5) * 4.0))
        if shot.hip_torsion_angle is not None:
            mid = 27.5
            values.append(max(0.0, 100.0 - abs(shot.hip_torsion_angle - mid) * 3.0))
        if shot.toe_angle is not None:
            values.append(max(0.0, 100.0 - max(0.0, shot.toe_angle - 15.0) * 4.0))
        if shot.max_folding_angle is not None:
            values.append(max(0.0, 100.0 - abs(shot.max_folding_angle - 80.0) * 2.0))
        if shot.whipping_velocity is not None:
            values.append(min(100.0, max(0.0, shot.whipping_velocity / 5.0)))
        if shot.ankle_rigidity is not None:
            values.append(max(0.0, 100.0 - shot.ankle_rigidity * 20.0))
        if not values:
            return None
        return round(sum(values) / len(values), 2)

    def extract_extreme_cases(
        self,
        cluster_id: Optional[str] = None,
        baseline: str = "T1",
        followup: str = "T2",
        percentile: float = EXTREME_PERCENTILE,
    ) -> dict[str, Any]:
        """Purposive Sampling：基于 baseline→followup 综合分斜率抽取极端个案。

        - High Responders：斜率最高的前 percentile（默认 20%）
        - Low Responders：得分一直处于低位（均值 ≤ 队列中位数）且斜率最平缓的后 percentile
        """
        tp_from = _parse_timepoint(baseline) or Timepoint.T1
        tp_to = _parse_timepoint(followup) or Timepoint.T2
        ratio = max(0.01, min(0.5, float(percentile)))

        score_map = self._subject_timepoint_composite(cluster_id=cluster_id)
        cohort: list[dict[str, Any]] = []

        for anonymous_id, tp_scores in score_map.items():
            if tp_from.value not in tp_scores or tp_to.value not in tp_scores:
                continue
            score_t1 = float(tp_scores[tp_from.value])
            score_t2 = float(tp_scores[tp_to.value])
            # 相邻节点间距按 1 个实验单位计，斜率 = Δscore
            slope = round(score_t2 - score_t1, 4)
            mean_level = round((score_t1 + score_t2) / 2.0, 2)
            profile = self._profiles.get(anonymous_id)
            cohort.append(
                {
                    "anonymous_id": anonymous_id,
                    "cluster_id": profile.cluster_id if profile else "",
                    "experimental_group": (
                        profile.experimental_group.value if profile else None
                    ),
                    "score_t1": score_t1,
                    "score_t2": score_t2,
                    "slope": slope,
                    "mean_level": mean_level,
                }
            )

        n = len(cohort)
        if n == 0:
            return {
                "success": True,
                "baseline": tp_from.value,
                "followup": tp_to.value,
                "cluster_id": cluster_id or "ALL",
                "percentile": ratio,
                "eligible_count": 0,
                "high_responders": [],
                "low_responders": [],
                "message": f"暂无同时具备 {tp_from.value} 与 {tp_to.value} 综合得分的被试。",
            }

        k = max(1, int(math.ceil(n * ratio)))

        by_slope_desc = sorted(cohort, key=lambda x: (x["slope"], x["score_t2"]), reverse=True)
        high_raw = by_slope_desc[:k]

        median_level = sorted(c["mean_level"] for c in cohort)[n // 2]
        low_pool = [c for c in cohort if c["mean_level"] <= median_level]
        if not low_pool:
            low_pool = list(cohort)
        by_slope_asc = sorted(low_pool, key=lambda x: (x["slope"], x["mean_level"]))
        low_raw = by_slope_asc[:k]

        # 避免同一人同时出现在两侧（斜率极高且又低位的边界情况优先归高反应）
        high_ids = {item["anonymous_id"] for item in high_raw}
        low_raw = [item for item in low_raw if item["anonymous_id"] not in high_ids]

        high_responders = [
            ExtremeCaseSubject(**item, responder_type="high_responder").model_dump()
            for item in high_raw
        ]
        low_responders = [
            ExtremeCaseSubject(**item, responder_type="low_responder").model_dump()
            for item in low_raw
        ]

        return {
            "success": True,
            "baseline": tp_from.value,
            "followup": tp_to.value,
            "cluster_id": cluster_id or "ALL",
            "percentile": ratio,
            "eligible_count": n,
            "mean_level_median": median_level,
            "high_responders": high_responders,
            "low_responders": low_responders,
            "high_responder_count": len(high_responders),
            "low_responder_count": len(low_responders),
            "sampling_note": (
                "高反应者 = 斜率最高前 20%；"
                "低反应者 = 均值处于低位（≤中位数）且斜率最平缓的后 20%。"
                "名单可用于后续现象学深度访谈的目的性抽样。"
            ),
        }


# 模块级便捷入口，供 api_server 直接调用
_default_service: Optional[ResearchDashboardService] = None


def get_dashboard_service(reload: bool = True) -> ResearchDashboardService:
    """获取（并按需重新装载）默认 ResearchDashboardService 单例。"""
    global _default_service
    if _default_service is None:
        _default_service = ResearchDashboardService()
    if reload or not _default_service._shots:
        _default_service.load()
    return _default_service
