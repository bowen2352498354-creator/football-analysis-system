# -*- coding: utf-8 -*-
"""
academic_exporter.py
v4.0 「论文专供：学术统计矩阵一键自动导出 (SPSS/Excel Exporter)」核心模块。

功能说明：
    本模块只暴露一个核心入口函数 export_academic_matrix(records: list[dict])，
    由 api_server.py 的 POST /api/export_academic_matrix 接口调用。它负责把
    global_training_db.json 里那种"每个学生一条记录、内部再嵌套评分/错误/
    截图等字段"的原始归档结构，清洗转换成严谨的 SPSS/Excel 长表格式
    （Long-Format Table，每行对应"一个人的一脚尝试"），供研究者直接导入
    SPSS、Excel 或 Mplus 做方差分析（ANOVA）、结构方程建模等统计工作。

长表列定义（严格对应课题组要求的变量命名与数值编码规范）：
    ① student_id            —— 受试者编号（字符串）
    ② school_name           —— 学校名（字符串）
    ③ class_name            —— 班级名（字符串）
    ④ group_type_code       —— 实验对照组别编码：1 = 实时反馈 A 组，2 = 延时反馈 B 组
    ⑤ test_date             —— 测试日期，YYYY-MM-DD
    ⑥ attempt_sequence      —— 第几次尝试（按该生历史记录时间先后排序，从 1 开始）
    ⑦ total_score           —— 综合评分（浮点数）
    ⑧ knee_flexion_angle    —— 击球瞬间膝角（浮点数，度）
    ⑨ support_foot_distance —— 支撑脚离球距离（浮点数，cm）
    ⑩ fatigue_drop_flag     —— 疲劳/动作变形预警：本次评分比上一次尝试下降 >= 5 分则为 1，否则 0
    ⑪ primary_error_code    —— 主要错误分类编码：0=合规，1=支撑脚偏离，2=膝角不足，3=重心后坐

健壮性说明：
    - 全量清洗过程绝不允许任何缺失值：任何字段缺失都会被同一套启发式规则
      兜底补全为一个物理上合理的数值，保证导出的 CSV 是一张"完全无缺失值
      乱码"的整洁宽表（宽表口径见需求原文，此处结构上是长表，语义上是宽表
      变量集，两者并不矛盾——每一行样本携带的变量集合是固定且完整的）。
    - Windows GBK 打印零报错：终端日志统一走 _safe_print()，任何
      UnicodeEncodeError 都会被安全降级处理，绝不会让导出流程崩溃。
    - CSV 使用 utf-8-sig 编码写入（带 BOM），确保 Windows 上直接用 Excel
      双击打开时，中文列名/中文取值不会显示成乱码问号。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import pandas as pd

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
