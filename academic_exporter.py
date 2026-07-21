# -*- coding: utf-8 -*-
"""
academic_exporter.py
V3.1 「论文专供：全数字化 SPSS 标准宽表一键导出 (MSEM / HLM / CRSE)」核心模块。

双轨导出能力：
    A) 长表（Long Format）—— ``export_academic_matrix`` / ``build_long_format_dataframe``
       兼容旧版 global_training_db.json，每行 = 一人一脚尝试，供 ANOVA 等。

    B) 宽表（Wide Format）—— ``AcademicDataExporter.generate_wide_format_matrix``
       Cluster-RCT 科研主路径：每行 = 一名匿名被试 (anonymous_id)，
       全数字编码（组别 / 疲劳 / 锁踝状态）、SEM 衍生中介变量、
       T0–T4 前缀展平，表尾 ``Class_Dummy_*`` 群聚固定效应哑变量。
       默认落盘文件名：``AI_Football_Research_Matrix_V3.csv``。

宽表列示例：
    anonymous_id, experimental_group, cluster_code,
    T0_Ankle_Rigidity, T0_Ankle_Rigidity_Score, T0_Heatmap_Dispersion_Index,
    T0_Fatigue_Alert, T0_Ankle_Lock_Status, …,
    T1_Ankle_Rigidity, …,
    Class_Dummy_1, Class_Dummy_2, Class_Dummy_3, Class_Dummy_4, Class_Dummy_5

伦理红线：绝不 JOIN EthicsIdentityMapping；导出矩阵仅含匿名编号。
全数字铁律：状态列禁止中文/英文字符串，一律 0/1/2/3 整数编码。

健壮性说明：
    - Windows GBK 打印零报错：终端日志统一走 _safe_print()。
    - CSV 使用 utf-8-sig（带 BOM），Excel / SPSS 直接打开不乱码。
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional

import pandas as pd

try:
    from models.enums import BIOMECH_CORE_METRIC_KEYS, StudyTimepoint
except ImportError:  # 允许在无 models 包的精简环境下单独测长表逻辑
    BIOMECH_CORE_METRIC_KEYS = (
        "distance_cm",
        "toe_angle",
        "max_folding_angle",
        "whipping_velocity",
        "impact_knee_angle",
        "ankle_rigidity",
        "support_knee_angle",
        "hip_torsion_angle",
    )

    class StudyTimepoint:  # type: ignore[no-redef]
        T0 = "T0"
        T1 = "T1"
        T2 = "T2"
        T3 = "T3"
        T4 = "T4"

# --------------------------------------------------------------------------
# Windows 控制台编码兼容性修复：与 api_server.py / word_reporter.py 保持完全
# 一致的第一/第二防线，确保本模块被后台线程调用时绝不会因为终端打印中文/
# Emoji 而抛出 UnicodeEncodeError 炸掉导出流程。
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

# 学术数据导出物理落地文件夹：项目根目录 / academic_data_export /
EXPORT_DIR = os.path.join(SCRIPT_DIR, "academic_data_export")

# 长表最终导出的标准列顺序（严格对应课题组变量命名规范）
LONG_FORMAT_COLUMNS = [
    "student_id",
    "school_name",
    "class_name",
    "group_type_code",
    "test_date",
    "attempt_sequence",
    "total_score",
    "knee_flexion_angle",
    "support_foot_distance",
    "fatigue_drop_flag",
    "primary_error_code",
]

# 综合评分缺失时的中性兜底值（保证 total_score 列绝不出现空值/NaN，
# 又不会因为极端的 0 分/100 分而扭曲后续的方差分析结果）
_NEUTRAL_SCORE_FALLBACK = 60.0

# 膝角 / 支撑脚距离启发式估算的物理参考中心点，与 api_server.py 里
# _estimate_knee_flexion_angle / _estimate_support_foot_distance 完全同源，
# 保证「实时看板显示的估算值」与「导出的学术矩阵里的估算值」永远一致。
_KNEE_ANGLE_OPTIMAL_CENTER = 150.0
_SUPPORT_FOOT_DISTANCE_IDEAL_CENTER = 17.5

# 主要错误分类标签 -> 数值编码，优先级从高到低（一条记录只保留一个主要编码）
_PRIMARY_ERROR_CODE_PRIORITY: list[tuple[str, int]] = [
    ("支撑脚位置偏离", 1),
    ("膝关节过度屈曲", 2),
    ("身体重心偏移", 3),
]

# 疲劳/动作变形预警的评分下降阈值：与上一次尝试相比下降达到或超过这个分数即触发
FATIGUE_DROP_THRESHOLD = 5.0


def _estimate_knee_flexion_angle(score: Optional[float]) -> float:
    safe_score = score if isinstance(score, (int, float)) else _NEUTRAL_SCORE_FALLBACK
    angle = _KNEE_ANGLE_OPTIMAL_CENTER - (100.0 - safe_score) * 0.35
    return round(max(95.0, min(185.0, angle)), 1)


def _estimate_support_foot_distance(score: Optional[float]) -> float:
    safe_score = score if isinstance(score, (int, float)) else _NEUTRAL_SCORE_FALLBACK
    distance = _SUPPORT_FOOT_DISTANCE_IDEAL_CENTER + (100.0 - safe_score) * 0.15
    return round(max(5.0, min(45.0, distance)), 1)


def _derive_primary_error_code(errors) -> int:
    if not isinstance(errors, list) or not errors:
        return 0
    for label, code in _PRIMARY_ERROR_CODE_PRIORITY:
        if label in errors:
            return code
    return 0


def _extract_test_date(timestamp_text: Optional[str]) -> str:
    text = (timestamp_text or "").strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return time.strftime("%Y-%m-%d")


def _clean_score(raw_score) -> float:
    if isinstance(raw_score, (int, float)):
        return round(float(raw_score), 1)
    return _NEUTRAL_SCORE_FALLBACK


def build_long_format_dataframe(records: list[dict]) -> pd.DataFrame:
    """把 global_training_db.json 的原始记录列表，清洗转换成标准长表格式
    的 pandas DataFrame（不做任何文件落盘，方便单独单元测试/复用）。

    清洗步骤：
        1) 逐条记录补全 test_date / total_score / knee_flexion_angle /
           support_foot_distance / primary_error_code / group_type_code
           （历史旧记录缺失的字段一律走启发式规则兜底补全，绝不留空）；
        2) 按 (school, classGroup, studentId) 分组，组内按时间戳升序排列，
           计算 attempt_sequence（从 1 开始）与 fatigue_drop_flag（对比
           组内上一条记录的评分）；
        3) 按标准列顺序整理输出，绝不包含任何多余的内部字段。
    """
    if not records:
        return pd.DataFrame(columns=LONG_FORMAT_COLUMNS)

    rows: list[dict] = []
    for record in records:
        if not isinstance(record, dict):
            continue

        school = str(record.get("school") or "未设置学校")
        class_group = str(record.get("classGroup") or "未设置班级")
        student_id = str(record.get("studentId") or "未填写编号")
        timestamp = record.get("timestamp") or ""
        record_type = record.get("type")

        score = _clean_score(record.get("score"))

        group_type_code = record.get("groupTypeCode")
        if group_type_code not in (1, 2):
            group_type_code = 1 if record_type == "realtime" else 2

        test_date = record.get("testDate") or _extract_test_date(timestamp)

        knee_flexion_angle = record.get("kneeFlexionAngle")
        if not isinstance(knee_flexion_angle, (int, float)):
            knee_flexion_angle = _estimate_knee_flexion_angle(record.get("score"))

        support_foot_distance = record.get("supportFootDistance")
        if not isinstance(support_foot_distance, (int, float)):
            support_foot_distance = _estimate_support_foot_distance(record.get("score"))

        primary_error_code = record.get("primaryErrorCode")
        if primary_error_code not in (0, 1, 2, 3):
            primary_error_code = _derive_primary_error_code(record.get("biomechanicalErrors"))

        rows.append(
            {
                "student_id": student_id,
                "school_name": school,
                "class_name": class_group,
                "group_type_code": int(group_type_code),
                "test_date": test_date,
                "total_score": round(float(score), 1),
                "knee_flexion_angle": round(float(knee_flexion_angle), 1),
                "support_foot_distance": round(float(support_foot_distance), 1),
                "primary_error_code": int(primary_error_code),
                "_timestamp_sort_key": timestamp,
                "_group_key": f"{school}__{class_group}__{student_id}",
            }
        )

    if not rows:
        return pd.DataFrame(columns=LONG_FORMAT_COLUMNS)

    df = pd.DataFrame(rows)

    # 组内按时间先后排序，推导 attempt_sequence（第几次尝试）与
    # fatigue_drop_flag（本次评分比上一次尝试下降 >= 5 分则为 1）。
    df.sort_values(by=["_group_key", "_timestamp_sort_key"], inplace=True, kind="stable")
    df["attempt_sequence"] = df.groupby("_group_key").cumcount() + 1
    df["_prev_score"] = df.groupby("_group_key")["total_score"].shift(1)
    df["fatigue_drop_flag"] = (
        (df["_prev_score"].notna()) & ((df["_prev_score"] - df["total_score"]) >= FATIGUE_DROP_THRESHOLD)
    ).astype(int)

    # 恢复成"每位学生内部按尝试先后顺序、跨学生按学号排列"的直观阅读顺序
    df.sort_values(by=["_group_key", "attempt_sequence"], inplace=True, kind="stable")

    df = df[LONG_FORMAT_COLUMNS].reset_index(drop=True)
    return df


def export_academic_matrix(records: list[dict]) -> dict:
    """核心入口：清洗全量记录 -> 生成长表 DataFrame -> 落盘写入 CSV，
    返回结构化的导出结果字典，供 api_server.py 直接转发给前端。

    返回：
        成功：{"success": True, "path": 绝对路径, "filename": 文件名,
               "rowCount": 行数, "studentCount": 涉及的受试者人数}
        失败：{"success": False, "message": 错误说明}
    """
    try:
        os.makedirs(EXPORT_DIR, exist_ok=True)

        df = build_long_format_dataframe(records)

        if df.empty:
            return {
                "success": False,
                "message": "全局训练数据库当前为空，暂无任何历史归档记录可供导出，请先完成至少一次测试。",
            }

        timestamp_label = time.strftime("%Y%m%d")
        filename = f"Academic_SPSS_Matrix_{timestamp_label}.csv"
        full_path = os.path.join(EXPORT_DIR, filename)

        # utf-8-sig：带 BOM 的 UTF-8，确保 Windows 上直接双击用 Excel 打开时，
        # 中文列取值（学校名/班级名）不会被误判成 GBK 编码而显示成乱码问号。
        df.to_csv(full_path, index=False, encoding="utf-8-sig")

        student_count = df["student_id"].nunique() if "student_id" in df.columns else 0

        _safe_print(f"[academic_exporter] 学术统计矩阵已导出：{full_path}（共 {len(df)} 行）")

        return {
            "success": True,
            "path": os.path.abspath(full_path),
            "filename": filename,
            "rowCount": int(len(df)),
            "studentCount": int(student_count),
        }
    except Exception as exc:  # noqa: BLE001 - 导出失败必须结构化返回，绝不抛出让接口 500
        _safe_print(f"[academic_exporter] 导出学术统计矩阵失败：{exc}")
        return {"success": False, "message": f"导出学术统计矩阵失败：{exc}"}


# ==========================================================================
# AcademicDataExporter —— V3.1 全数字化 SPSS 宽表（MSEM / HLM / CRSE）
# ==========================================================================

# 时间节点顺序（与 StudyTimepoint / 16 周追踪协议对齐）
STUDY_TIMEPOINTS: tuple[str, ...] = ("T0", "T1", "T2", "T3", "T4")

# 浏览器 / 落盘统一文件名
RESEARCH_MATRIX_V3_FILENAME = "AI_Football_Research_Matrix_V3.csv"

# SPSS 友好量纲名 → ShotAttemptLog 扁平字段名
# 宽表列名：{T}_{Metric}（时点均值，全数字）
WIDE_METRIC_FIELD_MAP: dict[str, str] = {
    "Knee_Flexion": "impact_knee_angle",
    "Ankle_Rigidity": "ankle_rigidity",
    "Impact_Score": "total_score",
    "Support_Distance": "distance_cm",
    "Toe_Angle": "toe_angle",
    "Max_Folding": "max_folding_angle",
    "Whipping_Velocity": "whipping_velocity",
    "Support_Knee": "support_knee_angle",
    "Hip_Torsion": "hip_torsion_angle",
}

# ---- 全数字编码字典（Digital Encoding）----
# 组别：GROUP_A → 1, GROUP_B → 2, GROUP_C → 3
GROUP_CODE_MAP: dict[str, int] = {
    "GROUP_A": 1,
    "GROUP_A_REALTIME": 1,
    "A": 1,
    "REALTIME": 1,
    "1": 1,
    "GROUP_B": 2,
    "GROUP_B_DELAYED": 2,
    "B": 2,
    "DELAYED": 2,
    "2": 2,
    "GROUP_C": 3,
    "GROUP_C_CONTROL": 3,
    "C": 3,
    "CONTROL": 3,
    "3": 3,
}

# 脚踝锁紧状态：GREEN_OPTIMAL → 3, YELLOW → 2, RED → 1
ANKLE_STATUS_CODE_MAP: dict[str, int] = {
    "GREEN_OPTIMAL": 3,
    "GREEN": 3,
    "G": 3,
    "YELLOW_APPROACHING": 2,
    "YELLOW": 2,
    "Y": 2,
    "RED_DEVIATED": 1,
    "RED": 1,
    "R": 1,
}

# 与 error_diagnoser 脚踝方差阈值对齐（用于由方差反推状态码 / 归一化刚性）
ANKLE_VARIANCE_GREEN = 2.0
ANKLE_VARIANCE_YELLOW_HIGH = 5.0
# 方差 → [0,1] 刚性分数的软上界（> 此值视为刚性 0）
ANKLE_VARIANCE_NORM_CEILING = 10.0

# 整群固定效应：Class_1…Class_6，以 Class_6 为参照类 → Class_Dummy_1…Class_Dummy_5
CLUSTER_LEVELS: tuple[str, ...] = tuple(f"Class_{i}" for i in range(1, 7))
CLUSTER_REFERENCE: str = "Class_6"
CLUSTER_DUMMY_COLUMNS: tuple[str, ...] = tuple(
    f"Class_Dummy_{i}" for i in range(1, 6)
)


def _as_mapping(obj: Any) -> dict[str, Any]:
    """把 ORM 行 / Pydantic / dict 统一成普通 dict。"""
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return dict(obj.model_dump())
    if hasattr(obj, "__dict__"):
        raw = {
            key: value
            for key, value in vars(obj).items()
            if not key.startswith("_")
        }
        for key in (
            "id",
            "anonymous_id",
            "cluster_id",
            "experimental_group",
            "timepoint",
            "timepoint_session_id",
            "session_date",
            "impact_frame_index",
            "total_score",
            "dx_support",
            "dy_support",
            "fatigue_alert_flag",
            "ankle_lock_status",
            *BIOMECH_CORE_METRIC_KEYS,
        ):
            if hasattr(obj, key):
                try:
                    raw[key] = getattr(obj, key)
                except Exception:  # noqa: BLE001
                    pass
        return raw
    return {}


def _coerce_timepoint_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = str(value).strip().upper()
    if text in STUDY_TIMEPOINTS:
        return text
    if text.isdigit():
        candidate = f"T{text}"
        if candidate in STUDY_TIMEPOINTS:
            return candidate
    for label in STUDY_TIMEPOINTS:
        if text.endswith(label):
            return label
    return None


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number != number:  # NaN
            return None
        return number
    try:
        number = float(value)
        if number != number:
            return None
        return number
    except (TypeError, ValueError):
        return None


def encode_experimental_group(value: Any) -> int:
    """组别全数字编码：GROUP_A→1, GROUP_B→2, GROUP_C→3；未知→0。"""
    if value is None:
        return 0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        code = int(value)
        return code if code in (1, 2, 3) else 0
    if hasattr(value, "value"):
        value = value.value
    text = str(value).strip().upper().replace(" ", "_")
    if text in GROUP_CODE_MAP:
        return GROUP_CODE_MAP[text]
    # 兼容 type=realtime / delayed / control
    if "REALTIME" in text or text.endswith("_A") or text == "A":
        return 1
    if "DELAY" in text or text.endswith("_B") or text == "B":
        return 2
    if "CONTROL" in text or text.endswith("_C") or text == "C":
        return 3
    if text.startswith("GROUP_A"):
        return 1
    if text.startswith("GROUP_B"):
        return 2
    if text.startswith("GROUP_C"):
        return 3
    return 0


def encode_fatigue_alert(value: Any) -> int:
    """疲劳预警：True→1, False→0。"""
    if value is None:
        return 0
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 1 if float(value) != 0.0 else 0
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "fatigue", "alert"):
        return 1
    return 0


def encode_ankle_lock_status(value: Any) -> int:
    """脚踝锁紧状态：GREEN_OPTIMAL→3, YELLOW→2, RED→1；未知→0。"""
    if value is None:
        return 0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        code = int(value)
        return code if code in (1, 2, 3) else 0
    text = str(value).strip().upper().replace(" ", "_")
    if text in ANKLE_STATUS_CODE_MAP:
        return ANKLE_STATUS_CODE_MAP[text]
    if "GREEN" in text:
        return 3
    if "YELLOW" in text:
        return 2
    if "RED" in text:
        return 1
    return 0


def ankle_status_from_variance(variance: Optional[float]) -> int:
    """由脚踝方差反推锁踝状态码（与 error_diagnoser 阈值一致）。"""
    if variance is None:
        return 0
    if variance < ANKLE_VARIANCE_GREEN:
        return 3
    if variance <= ANKLE_VARIANCE_YELLOW_HIGH:
        return 2
    return 1


def normalize_ankle_rigidity_score(variance: Optional[float]) -> Optional[float]:
    """脚踝方差 → [0, 1] 综合锁踝刚性（越高越锁死定型）。"""
    if variance is None:
        return None
    v = max(0.0, float(variance))
    score = 1.0 - min(v / ANKLE_VARIANCE_NORM_CEILING, 1.0)
    return round(score, 4)


def heatmap_dispersion_index(
    points: Sequence[tuple[float, float]],
) -> Optional[float]:
    """热力图散度指数：支撑脚相对坐标 (dx, dy) 的 2D 标准距离。

    SD = sqrt( mean( (x-x̄)² + (y-ȳ)² ) )；点数 < 1 返回 None，单点返回 0。
    数值越小代表支撑脚落点越集中、动作定型越好。
    """
    cleaned: list[tuple[float, float]] = []
    for point in points:
        if point is None or len(point) < 2:
            continue
        dx, dy = _safe_float(point[0]), _safe_float(point[1])
        if dx is None or dy is None:
            continue
        cleaned.append((dx, dy))
    n = len(cleaned)
    if n == 0:
        return None
    if n == 1:
        return 0.0
    mean_x = sum(p[0] for p in cleaned) / n
    mean_y = sum(p[1] for p in cleaned) / n
    sq = sum((p[0] - mean_x) ** 2 + (p[1] - mean_y) ** 2 for p in cleaned) / n
    return round(float(sq ** 0.5), 4)


def _extract_support_xy(shot: Mapping[str, Any]) -> Optional[tuple[float, float]]:
    """从射门日志 / 嵌套 spatial_trajectory 抽取支撑脚相对坐标 (dx, dy)。"""
    dx = _safe_float(shot.get("dx_support"))
    dy = _safe_float(shot.get("dy_support"))
    if dx is not None and dy is not None:
        return (dx, dy)

    support_rel = shot.get("support_rel")
    if isinstance(support_rel, (list, tuple)) and len(support_rel) >= 2:
        dx = _safe_float(support_rel[0])
        dy = _safe_float(support_rel[1])
        if dx is not None and dy is not None:
            return (dx, dy)

    spatial = shot.get("spatial_trajectory") or shot.get("spatialTrajectory")
    if isinstance(spatial, Mapping):
        dx = _safe_float(spatial.get("dx_support"))
        dy = _safe_float(spatial.get("dy_support"))
        if dx is not None and dy is not None:
            return (dx, dy)
        rel = spatial.get("support_rel")
        if isinstance(rel, (list, tuple)) and len(rel) >= 2:
            dx = _safe_float(rel[0])
            dy = _safe_float(rel[1])
            if dx is not None and dy is not None:
                return (dx, dy)

    # 回退：横距 + 前后偏移（cm）近似俯视相对坐标
    lateral = _safe_float(
        shot.get("support_lateral_dist_cm")
        or shot.get("supportLateralDistCm")
        or shot.get("distance_cm")
        or shot.get("supportFootDistance")
    )
    ap = _safe_float(
        shot.get("support_ap_offset_cm") or shot.get("supportApOffsetCm")
    )
    if lateral is not None and ap is not None:
        return (lateral, ap)
    if lateral is not None:
        return (lateral, 0.0)
    return None


def _extract_ankle_variance(shot: Mapping[str, Any]) -> Optional[float]:
    number = _safe_float(shot.get("ankle_rigidity"))
    if number is not None:
        return number
    number = _safe_float(shot.get("ankle_rigidity_variance"))
    if number is not None:
        return number
    detail = shot.get("scoreDetail") or shot.get("score_detail") or {}
    if isinstance(detail, Mapping):
        indicators = detail.get("indicators") or {}
        if isinstance(indicators, Mapping):
            entry = indicators.get("ankle_rigidity")
            if isinstance(entry, Mapping):
                number = _safe_float(entry.get("variance"))
                if number is None:
                    number = _safe_float(entry.get("value"))
                return number
    return None


def _extract_ankle_status_code(shot: Mapping[str, Any]) -> int:
    for key in (
        "ankle_lock_status",
        "ankleLockStatus",
        "ankle_status",
        "ankleStatus",
    ):
        code = encode_ankle_lock_status(shot.get(key))
        if code:
            return code
    detail = shot.get("scoreDetail") or shot.get("score_detail") or {}
    if isinstance(detail, Mapping):
        indicators = detail.get("indicators") or {}
        if isinstance(indicators, Mapping):
            entry = indicators.get("ankle_rigidity")
            if isinstance(entry, Mapping):
                code = encode_ankle_lock_status(entry.get("status"))
                if code:
                    return code
    return ankle_status_from_variance(_extract_ankle_variance(shot))


def _extract_fatigue_flag(shot: Mapping[str, Any]) -> int:
    for key in (
        "fatigue_alert_flag",
        "fatigueAlertFlag",
        "is_fatigue",
        "isFatigue",
        "fatigue_drop_flag",
    ):
        if key in shot and shot.get(key) is not None:
            return encode_fatigue_alert(shot.get(key))
    warning = shot.get("fatigue_warning") or shot.get("fatigueWarning")
    if isinstance(warning, Mapping):
        return encode_fatigue_alert(
            warning.get("is_fatigue") or warning.get("isFatigue")
        )
    return 0


def _normalize_cluster_label(cluster_id: Any) -> str:
    text = str(cluster_id or "").strip()
    if not text:
        return ""
    if text.isdigit():
        return f"Class_{text}"
    lower = text.lower().replace(" ", "_")
    if lower.startswith("class"):
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            return f"Class_{digits}"
    return text


def _cluster_code_int(cluster_id: Any) -> int:
    """集群标签 → 整数编码 1–6；未知 → 0。"""
    normalized = _normalize_cluster_label(cluster_id)
    if normalized in CLUSTER_LEVELS:
        try:
            return int(normalized.split("_", 1)[1])
        except (IndexError, ValueError):
            return 0
    return 0


class AcademicDataExporter:
    """V3.1 科研数据自动化清洗与导出：异构 JSON → 全数字化 SPSS 宽表。

    主键：``anonymous_id``（一行一名学生）。
    量纲：各时点 (T0–T4) 均值展平为 ``T{{k}}_{{Metric}}``。
    衍生中介：``Heatmap_Dispersion_Index`` / ``Ankle_Rigidity_Score``。
    状态编码：组别 1/2/3、疲劳 0/1、锁踝 3/2/1。
    固定效应：``Class_Dummy_1``…``Class_Dummy_5``（Class_6 参照）。
    """

    TIMEPOINTS = STUDY_TIMEPOINTS
    METRIC_FIELD_MAP = WIDE_METRIC_FIELD_MAP
    CLUSTER_LEVELS = CLUSTER_LEVELS
    CLUSTER_REFERENCE = CLUSTER_REFERENCE
    CLUSTER_DUMMY_COLUMNS = CLUSTER_DUMMY_COLUMNS
    EXPORT_FILENAME = RESEARCH_MATRIX_V3_FILENAME

    def __init__(
        self,
        shot_logs: Optional[Iterable[Any]] = None,
        student_profiles: Optional[Iterable[Any]] = None,
        timepoint_sessions: Optional[Iterable[Any]] = None,
    ) -> None:
        self.shot_logs: list[dict[str, Any]] = [
            _as_mapping(item) for item in (shot_logs or [])
        ]
        self.student_profiles: list[dict[str, Any]] = [
            _as_mapping(item) for item in (student_profiles or [])
        ]
        self.timepoint_sessions: list[dict[str, Any]] = [
            _as_mapping(item) for item in (timepoint_sessions or [])
        ]

    # ------------------------------------------------------------------
    # 工厂
    # ------------------------------------------------------------------

    @classmethod
    def from_session(cls, session: Any) -> "AcademicDataExporter":
        """从 Cluster-RCT ORM Session 拉取全量射门 / 档案 / 时点课。"""
        from sqlalchemy import select

        from models.shot_attempt_log import ShotAttemptLog
        from models.student_profile import StudentProfile
        from models.timepoint_session import TimepointSession

        shots = list(session.scalars(select(ShotAttemptLog)).all())
        profiles = list(session.scalars(select(StudentProfile)).all())
        sessions = list(session.scalars(select(TimepointSession)).all())
        return cls(
            shot_logs=shots,
            student_profiles=profiles,
            timepoint_sessions=sessions,
        )

    @classmethod
    def from_db(cls) -> "AcademicDataExporter":
        """打开本地 ``cluster_rct.db``；若射门表为空则桥接 global_training_db.json。"""
        try:
            from db import init_db, session_scope

            init_db()
            with session_scope() as session:
                exporter = cls.from_session(session)
                if exporter.shot_logs:
                    return cls(
                        shot_logs=list(exporter.shot_logs),
                        student_profiles=list(exporter.student_profiles),
                        timepoint_sessions=list(exporter.timepoint_sessions),
                    )
        except Exception as exc:  # noqa: BLE001
            _safe_print(f"[academic_exporter] from_db ORM 装载失败，回退 JSON：{exc}")

        return cls.from_global_json()

    @classmethod
    def from_global_json(
        cls, json_path: Optional[str] = None
    ) -> "AcademicDataExporter":
        """从 global_training_db.json（或显式路径）桥接异构归档为宽表输入。"""
        path = json_path or os.path.join(SCRIPT_DIR, "global_training_db.json")
        records: list[dict] = []
        if os.path.exists(path):
            import json

            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if isinstance(raw, list):
                records = [r for r in raw if isinstance(r, dict)]
        return cls.from_global_records(records)

    @classmethod
    def from_global_records(cls, records: Sequence[Mapping[str, Any]]) -> "AcademicDataExporter":
        """把教练端归档记录碾平为 shot_logs + student_profiles。"""
        # 按学生收集测试日，推断 T0–T4
        date_buckets: dict[str, list[str]] = {}
        for record in records:
            anon = str(
                record.get("anonymous_id")
                or record.get("studentId")
                or record.get("student_id")
                or ""
            ).strip()
            if not anon:
                continue
            test_date = str(
                record.get("testDate")
                or record.get("test_date")
                or _extract_test_date(record.get("timestamp"))
            )
            date_buckets.setdefault(anon, []).append(test_date)

        tp_by_anon_date: dict[tuple[str, str], str] = {}
        for anon, dates in date_buckets.items():
            ordered = sorted(set(dates))
            for index, day in enumerate(ordered):
                tp_by_anon_date[(anon, day)] = STUDY_TIMEPOINTS[
                    min(index, len(STUDY_TIMEPOINTS) - 1)
                ]

        # 班级字符串 → Class_1…Class_6（稳定哈希到 1–6，便于哑变量）
        class_labels = sorted(
            {
                str(r.get("classGroup") or r.get("cluster_id") or "").strip()
                for r in records
                if str(r.get("classGroup") or r.get("cluster_id") or "").strip()
            }
        )
        class_to_cluster: dict[str, str] = {}
        for index, label in enumerate(class_labels):
            class_to_cluster[label] = f"Class_{(index % 6) + 1}"

        profiles: dict[str, dict[str, Any]] = {}
        shots: list[dict[str, Any]] = []

        for record in records:
            anon = str(
                record.get("anonymous_id")
                or record.get("studentId")
                or record.get("student_id")
                or ""
            ).strip()
            if not anon:
                continue

            test_date = str(
                record.get("testDate")
                or record.get("test_date")
                or _extract_test_date(record.get("timestamp"))
            )
            timepoint = _coerce_timepoint_label(record.get("timepoint"))
            if timepoint is None:
                timepoint = tp_by_anon_date.get((anon, test_date), "T0")

            class_label = str(
                record.get("cluster_id")
                or record.get("classGroup")
                or ""
            ).strip()
            cluster_id = (
                _normalize_cluster_label(class_label)
                if class_label.lower().startswith("class") or class_label.isdigit()
                else class_to_cluster.get(class_label, "Class_1")
            )

            group_raw = (
                record.get("experimental_group")
                or record.get("groupTypeCode")
                or record.get("type")
            )
            group_code = encode_experimental_group(group_raw)

            if anon not in profiles:
                profiles[anon] = {
                    "anonymous_id": anon,
                    "cluster_id": cluster_id,
                    "experimental_group": group_code,
                }

            ankle_var = _extract_ankle_variance(record)
            knee = _safe_float(
                record.get("impact_knee_angle")
                or record.get("kneeFlexionAngle")
                or record.get("knee_flexion_angle")
            )
            distance = _safe_float(
                record.get("distance_cm")
                or record.get("supportFootDistance")
                or record.get("support_lateral_dist_cm")
            )
            score = _safe_float(
                record.get("total_score")
                or record.get("score")
                or record.get("composite_score")
            )
            xy = _extract_support_xy(record)

            shot: dict[str, Any] = {
                "anonymous_id": anon,
                "cluster_id": cluster_id,
                "experimental_group": group_code,
                "timepoint": timepoint,
                "session_date": test_date,
                "impact_knee_angle": knee,
                "ankle_rigidity": ankle_var,
                "distance_cm": distance,
                "total_score": score,
                "toe_angle": _safe_float(record.get("toe_angle")),
                "max_folding_angle": _safe_float(record.get("max_folding_angle")),
                "whipping_velocity": _safe_float(record.get("whipping_velocity")),
                "support_knee_angle": _safe_float(
                    record.get("support_knee_angle")
                    or record.get("support_knee_angle_resolved")
                ),
                "hip_torsion_angle": _safe_float(record.get("hip_torsion_angle")),
                "fatigue_alert_flag": _extract_fatigue_flag(record),
                "ankle_lock_status": _extract_ankle_status_code(record),
                "scoreDetail": record.get("scoreDetail") or record.get("score_detail"),
                "spatial_trajectory": record.get("spatial_trajectory")
                or record.get("spatialTrajectory"),
                "support_lateral_dist_cm": _safe_float(
                    record.get("support_lateral_dist_cm")
                ),
                "support_ap_offset_cm": _safe_float(
                    record.get("support_ap_offset_cm")
                ),
            }
            if xy is not None:
                shot["dx_support"], shot["dy_support"] = xy
            shots.append(shot)

        return cls(
            shot_logs=shots,
            student_profiles=list(profiles.values()),
            timepoint_sessions=[],
        )

    # ------------------------------------------------------------------
    # 核心：宽表汇聚
    # ------------------------------------------------------------------

    def generate_wide_format_matrix(self) -> pd.DataFrame:
        """以 anonymous_id 为唯一主键，生成全数字化 SPSS 宽表 DataFrame。

        步骤：
            1) 解析每条射门所属时点 (T0–T4)；
            2) 全数字编码组别 / 疲劳 / 锁踝状态；
            3) 按 (anonymous_id, timepoint) 聚合均值 + SEM 衍生中介；
            4) 展平为 ``T{{k}}_{{Metric}}`` / ``T{{k}}_Heatmap_Dispersion_Index`` 等；
            5) 表尾 ``Class_Dummy_1``…``Class_Dummy_5``（Class_6 全 0）。
        """
        profile_index = self._build_profile_index()
        tp_session_index = self._build_timepoint_session_index()
        long_rows = self._shots_to_long_metric_rows(profile_index, tp_session_index)

        id_columns = ["anonymous_id", "experimental_group", "cluster_code"]
        metric_columns = self._wide_metric_column_names()
        dummy_columns = list(self.CLUSTER_DUMMY_COLUMNS)
        all_columns = id_columns + metric_columns + dummy_columns

        subject_ids = list(profile_index.keys())
        if not subject_ids:
            subject_ids = sorted(
                {
                    str(row["anonymous_id"])
                    for row in long_rows
                    if row.get("anonymous_id")
                }
            )

        if not subject_ids:
            return pd.DataFrame(columns=all_columns)

        # ---- 按 (anonymous_id, timepoint) 聚合 ----
        aggregates: dict[tuple[str, str], dict[str, Any]] = {}
        if long_rows:
            long_df = pd.DataFrame(long_rows)
            for (anon_id, tp), group in long_df.groupby(
                ["anonymous_id", "timepoint"], sort=False
            ):
                slot: dict[str, Any] = {}
                for metric_name in self.METRIC_FIELD_MAP:
                    if metric_name not in group.columns:
                        continue
                    series = [
                        v
                        for v in (
                            _safe_float(x) for x in group[metric_name].tolist()
                        )
                        if v is not None
                    ]
                    if series:
                        slot[metric_name] = round(sum(series) / len(series), 4)

                # Heatmap_Dispersion_Index
                points: list[tuple[float, float]] = []
                if "dx_support" in group.columns and "dy_support" in group.columns:
                    for dx, dy in zip(
                        group["dx_support"].tolist(),
                        group["dy_support"].tolist(),
                        strict=False,
                    ):
                        fx, fy = _safe_float(dx), _safe_float(dy)
                        if fx is not None and fy is not None:
                            points.append((fx, fy))
                slot["Heatmap_Dispersion_Index"] = heatmap_dispersion_index(points)

                # Ankle_Rigidity_Score：方差归一化后取均值
                ankle_series = [
                    v
                    for v in (
                        _safe_float(x)
                        for x in group.get("Ankle_Rigidity", pd.Series(dtype=float)).tolist()
                    )
                    if v is not None
                ]
                if not ankle_series and "ankle_rigidity_raw" in group.columns:
                    ankle_series = [
                        v
                        for v in (
                            _safe_float(x)
                            for x in group["ankle_rigidity_raw"].tolist()
                        )
                        if v is not None
                    ]
                rigidity_scores = [
                    s
                    for s in (normalize_ankle_rigidity_score(v) for v in ankle_series)
                    if s is not None
                ]
                slot["Ankle_Rigidity_Score"] = (
                    round(sum(rigidity_scores) / len(rigidity_scores), 4)
                    if rigidity_scores
                    else None
                )

                # Fatigue_Alert：时点内任一熔断 → 1
                if "fatigue_alert" in group.columns:
                    slot["Fatigue_Alert"] = int(
                        max(
                            (
                                encode_fatigue_alert(v)
                                for v in group["fatigue_alert"].tolist()
                            ),
                            default=0,
                        )
                    )
                else:
                    slot["Fatigue_Alert"] = 0

                # Ankle_Lock_Status：众数；无则由均值方差反推
                if "ankle_lock_status" in group.columns:
                    codes = [
                        encode_ankle_lock_status(v)
                        for v in group["ankle_lock_status"].tolist()
                        if encode_ankle_lock_status(v)
                    ]
                    if codes:
                        slot["Ankle_Lock_Status"] = max(
                            set(codes), key=codes.count
                        )
                    else:
                        mean_var = slot.get("Ankle_Rigidity")
                        slot["Ankle_Lock_Status"] = ankle_status_from_variance(
                            mean_var if isinstance(mean_var, (int, float)) else None
                        )
                else:
                    mean_var = slot.get("Ankle_Rigidity")
                    slot["Ankle_Lock_Status"] = ankle_status_from_variance(
                        mean_var if isinstance(mean_var, (int, float)) else None
                    )

                aggregates[(str(anon_id), str(tp))] = slot

        # ---- 组装宽表行 ----
        wide_rows: list[dict[str, Any]] = []
        for anon_id in subject_ids:
            profile = profile_index.get(anon_id, {})
            cluster_id = str(
                profile.get("cluster_id")
                or self._infer_cluster_from_shots(anon_id, long_rows)
                or ""
            )
            group_code = encode_experimental_group(
                profile.get("experimental_group")
            )
            if group_code == 0:
                group_code = self._infer_group_from_shots(anon_id, long_rows)

            row: dict[str, Any] = {
                "anonymous_id": anon_id,
                "experimental_group": int(group_code),
                "cluster_code": int(_cluster_code_int(cluster_id)),
            }

            for tp in self.TIMEPOINTS:
                slot = aggregates.get((anon_id, tp), {})
                for metric_name in self.METRIC_FIELD_MAP:
                    row[f"{tp}_{metric_name}"] = slot.get(metric_name)
                row[f"{tp}_Heatmap_Dispersion_Index"] = slot.get(
                    "Heatmap_Dispersion_Index"
                )
                row[f"{tp}_Ankle_Rigidity_Score"] = slot.get("Ankle_Rigidity_Score")
                row[f"{tp}_Fatigue_Alert"] = int(slot.get("Fatigue_Alert", 0) or 0)
                row[f"{tp}_Ankle_Lock_Status"] = int(
                    slot.get("Ankle_Lock_Status", 0) or 0
                )

            row.update(self._cluster_dummy_encoding(cluster_id))
            wide_rows.append(row)

        wide_df = pd.DataFrame(wide_rows, columns=all_columns)
        wide_df.sort_values(
            by=["cluster_code", "anonymous_id"], inplace=True, kind="stable"
        )
        wide_df.reset_index(drop=True, inplace=True)

        # 强制状态 / 哑变量列为可空整数（SPSS 友好，禁止字符串）
        int_cols = ["experimental_group", "cluster_code", *dummy_columns]
        for tp in self.TIMEPOINTS:
            int_cols.append(f"{tp}_Fatigue_Alert")
            int_cols.append(f"{tp}_Ankle_Lock_Status")
        for col in int_cols:
            if col in wide_df.columns:
                wide_df[col] = (
                    pd.to_numeric(wide_df[col], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
        return wide_df

    def to_csv_bytes(self, dataframe: Optional[pd.DataFrame] = None) -> bytes:
        """将宽表编码为 utf-8-sig CSV 字节流（带 BOM，Excel/SPSS 友好）。"""
        df = dataframe if dataframe is not None else self.generate_wide_format_matrix()
        buffer = df.to_csv(index=False, encoding="utf-8-sig")
        if isinstance(buffer, bytes):
            return buffer
        return buffer.encode("utf-8-sig")

    def export_spss_matrix_file(
        self,
        output_dir: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> dict[str, Any]:
        """宽表落盘为 ``AI_Football_Research_Matrix_V3.csv``，返回路径与行列统计。"""
        try:
            target_dir = output_dir or EXPORT_DIR
            os.makedirs(target_dir, exist_ok=True)
            df = self.generate_wide_format_matrix()
            out_name = filename or self.EXPORT_FILENAME
            full_path = os.path.join(target_dir, out_name)
            df.to_csv(full_path, index=False, encoding="utf-8-sig")
            _safe_print(
                f"[academic_exporter] V3.1 科研宽表已导出：{full_path}"
                f"（{len(df)} 行 × {len(df.columns)} 列）"
            )
            return {
                "success": True,
                "path": os.path.abspath(full_path),
                "filename": out_name,
                "rowCount": int(len(df)),
                "columnCount": int(len(df.columns)),
                "studentCount": int(df["anonymous_id"].nunique())
                if "anonymous_id" in df.columns and len(df)
                else 0,
            }
        except Exception as exc:  # noqa: BLE001
            _safe_print(f"[academic_exporter] 导出 V3.1 宽表失败：{exc}")
            return {"success": False, "message": f"导出 V3.1 宽表失败：{exc}"}

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _build_profile_index(self) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for profile in self.student_profiles:
            anon = str(profile.get("anonymous_id") or "").strip()
            if not anon:
                continue
            index[anon] = profile
        return index

    def _build_timepoint_session_index(self) -> dict[int, str]:
        """timepoint_session.id → 'T0'…'T4'。"""
        index: dict[int, str] = {}
        for session in self.timepoint_sessions:
            session_id = session.get("id")
            label = _coerce_timepoint_label(session.get("timepoint"))
            if session_id is None or label is None:
                continue
            try:
                index[int(session_id)] = label
            except (TypeError, ValueError):
                continue
        return index

    def _shots_to_long_metric_rows(
        self,
        profile_index: dict[str, dict[str, Any]],
        tp_session_index: dict[int, str],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for shot in self.shot_logs:
            anon = str(shot.get("anonymous_id") or "").strip()
            if not anon:
                continue

            timepoint = _coerce_timepoint_label(shot.get("timepoint"))
            if timepoint is None:
                session_fk = shot.get("timepoint_session_id")
                if session_fk is not None:
                    try:
                        timepoint = tp_session_index.get(int(session_fk))
                    except (TypeError, ValueError):
                        timepoint = None
            if timepoint is None:
                continue

            metric_values: dict[str, Optional[float]] = {}
            has_valid_metric = False
            for metric_name, source_field in self.METRIC_FIELD_MAP.items():
                number = _safe_float(shot.get(source_field))
                if number is None and metric_name == "Knee_Flexion":
                    number = _safe_float(shot.get("kneeFlexionAngle"))
                    if number is None:
                        number = _safe_float(shot.get("knee_flexion_angle"))
                if number is None and metric_name == "Ankle_Rigidity":
                    number = _extract_ankle_variance(shot)
                if number is None and metric_name == "Impact_Score":
                    number = _safe_float(shot.get("score"))
                    if number is None:
                        number = _safe_float(shot.get("composite_score"))
                if number is None and metric_name == "Support_Distance":
                    number = _safe_float(shot.get("supportFootDistance"))
                    if number is None:
                        number = _safe_float(shot.get("support_lateral_dist_cm"))
                metric_values[metric_name] = number
                if number is not None:
                    has_valid_metric = True

            xy = _extract_support_xy(shot)
            fatigue = _extract_fatigue_flag(shot)
            ankle_status = _extract_ankle_status_code(shot)

            # 允许仅有空间坐标 / 疲劳标记的记录进入长表（衍生变量仍可结算）
            if not has_valid_metric and xy is None and not fatigue and not ankle_status:
                continue

            cluster_id = ""
            if anon in profile_index:
                cluster_id = str(profile_index[anon].get("cluster_id") or "")
            if not cluster_id:
                cluster_id = str(shot.get("cluster_id") or "")

            group_code = encode_experimental_group(
                shot.get("experimental_group")
                if shot.get("experimental_group") is not None
                else (profile_index.get(anon, {}) or {}).get("experimental_group")
            )

            row: dict[str, Any] = {
                "anonymous_id": anon,
                "cluster_id": cluster_id,
                "experimental_group": group_code,
                "timepoint": timepoint,
                "fatigue_alert": fatigue,
                "ankle_lock_status": ankle_status,
                **metric_values,
            }
            if xy is not None:
                row["dx_support"], row["dy_support"] = xy
            rows.append(row)
        return rows

    def _wide_metric_column_names(self) -> list[str]:
        columns: list[str] = []
        for tp in self.TIMEPOINTS:
            for metric_name in self.METRIC_FIELD_MAP:
                columns.append(f"{tp}_{metric_name}")
            columns.append(f"{tp}_Heatmap_Dispersion_Index")
            columns.append(f"{tp}_Ankle_Rigidity_Score")
            columns.append(f"{tp}_Fatigue_Alert")
            columns.append(f"{tp}_Ankle_Lock_Status")
        return columns

    def _cluster_dummy_encoding(self, cluster_id: str) -> dict[str, int]:
        """Class_1→Dummy_1 … Class_5→Dummy_5；Class_6（参照）与未知 → 全 0。"""
        dummies = {col: 0 for col in self.CLUSTER_DUMMY_COLUMNS}
        normalized = _normalize_cluster_label(cluster_id)

        if normalized == self.CLUSTER_REFERENCE or normalized not in self.CLUSTER_LEVELS:
            return dummies

        try:
            class_index = int(normalized.split("_", 1)[1])
        except (IndexError, ValueError):
            return dummies
        if 1 <= class_index <= 5:
            dummies[f"Class_Dummy_{class_index}"] = 1
        return dummies

    @staticmethod
    def _infer_cluster_from_shots(
        anonymous_id: str, long_rows: Sequence[Mapping[str, Any]]
    ) -> str:
        for row in long_rows:
            if row.get("anonymous_id") == anonymous_id and row.get("cluster_id"):
                return str(row["cluster_id"])
        return ""

    @staticmethod
    def _infer_group_from_shots(
        anonymous_id: str, long_rows: Sequence[Mapping[str, Any]]
    ) -> int:
        for row in long_rows:
            if row.get("anonymous_id") == anonymous_id:
                code = encode_experimental_group(row.get("experimental_group"))
                if code:
                    return code
        return 0


# --------------------------------------------------------------------------
# 独立运行测试（方便单独调试本模块，不需要启动完整的 FastAPI 服务）
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "global_training_db.json")
    if os.path.exists(db_path):
        with open(db_path, "r", encoding="utf-8") as f:
            demo_records = json.load(f)
    else:
        demo_records = []
    result = export_academic_matrix(demo_records)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    demo_exporter = AcademicDataExporter(
        student_profiles=[
            {
                "anonymous_id": "Sub_001",
                "cluster_id": "Class_1",
                "experimental_group": "GROUP_A",
            },
            {
                "anonymous_id": "Sub_002",
                "cluster_id": "Class_6",
                "experimental_group": "GROUP_C",
            },
        ],
        shot_logs=[
            {
                "anonymous_id": "Sub_001",
                "timepoint": "T1",
                "impact_knee_angle": 140.0,
                "ankle_rigidity": 1.5,
                "total_score": 80.0,
                "dx_support": 12.0,
                "dy_support": 3.0,
                "fatigue_alert_flag": False,
                "ankle_lock_status": "GREEN_OPTIMAL",
            },
            {
                "anonymous_id": "Sub_001",
                "timepoint": "T1",
                "impact_knee_angle": 150.0,
                "ankle_rigidity": 4.0,
                "total_score": 90.0,
                "dx_support": 18.0,
                "dy_support": 5.0,
                "fatigue_alert_flag": True,
                "ankle_lock_status": "YELLOW",
            },
            {
                "anonymous_id": "Sub_002",
                "timepoint": "T0",
                "impact_knee_angle": 155.0,
                "ankle_rigidity": 6.0,
                "total_score": 70.0,
                "dx_support": 20.0,
                "dy_support": -2.0,
                "fatigue_alert_flag": False,
                "ankle_lock_status": "RED",
            },
        ],
    )
    wide = demo_exporter.generate_wide_format_matrix()
    preview_cols = [
        "anonymous_id",
        "experimental_group",
        "T1_Ankle_Rigidity",
        "T1_Ankle_Rigidity_Score",
        "T1_Heatmap_Dispersion_Index",
        "T1_Fatigue_Alert",
        "T1_Ankle_Lock_Status",
        "Class_Dummy_1",
        "Class_Dummy_5",
    ]
    print(wide[[c for c in preview_cols if c in wide.columns]].to_string(index=False))
    export_result = demo_exporter.export_spss_matrix_file()
    print(json.dumps(export_result, ensure_ascii=False, indent=2))
