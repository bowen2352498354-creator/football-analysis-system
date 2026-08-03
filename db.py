# -*- coding: utf-8 -*-
"""本地 SQLite 引擎与会话工厂（边缘计算 / 数据不出域）。

同时承载教练端科研看板所需的班级/实验组对比聚合查询
（``compare_cohorts``），数据源优先 ``global_training_db.json``，
失败时静默降级为空结构，绝不向上抛出未捕获异常。

数据安全：
  - ``shot_attempt_logs.is_deleted`` 软删除；禁止物理 DELETE/DROP；
  - ``start_auto_backup_daemon`` 每 12 小时将 JSON/SQLite 打包至 ``./backups/``，
    保留最近 30 天。
"""

from __future__ import annotations

import json
import math
import os
import statistics
import threading
import zipfile
from collections import Counter, defaultdict
from collections.abc import Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from models import Base

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(SCRIPT_DIR, "cluster_rct.db")
DEFAULT_GLOBAL_DB_PATH = os.path.join(SCRIPT_DIR, "global_training_db.json")
DEFAULT_BACKUP_DIR = os.path.join(SCRIPT_DIR, "backups")

# 户外长期实验数据守护：每 12 小时自动备份，保留最近 30 天
BACKUP_INTERVAL_HOURS = 12
BACKUP_RETENTION_DAYS = 30
BACKUP_FILENAME_PREFIX = "db_backup_"
BACKUP_FILENAME_SUFFIX = ".bak"

_backup_daemon_lock = threading.Lock()
_backup_daemon_started = False
_backup_stop_event = threading.Event()
_backup_thread: Optional[threading.Thread] = None
_backup_last_run_lock = threading.Lock()
_backup_last_run_at: Optional[datetime] = None

# 可通过环境变量覆盖；默认落在项目根目录本地文件，符合全边缘架构
DATABASE_URL = os.environ.get(
    "CLUSTER_RCT_DATABASE_URL",
    f"sqlite:///{DEFAULT_DB_PATH.replace(os.sep, '/')}",
)

# 班级对比：单侧有效成绩条数低于该阈值时视为样本量不足
MIN_COHORT_SAMPLES = 2

# 五维雷达：助跑 / 支撑 / 后摆 / 踝锁 / 鞭打
RADAR_COMPARE_DIMS: tuple[tuple[str, str], ...] = (
    ("approach_rhythm", "助跑"),
    ("support_stability", "支撑"),
    ("backswing_folding", "后摆"),
    ("ankle_rigidity", "踝锁"),
    ("whipping_velocity", "鞭打"),
)

_RADAR_DIM_ALIASES: dict[str, tuple[str, ...]] = {
    "approach_rhythm": ("approach_rhythm", "approach_rhythm_score", "approach_score", "approach"),
    "support_stability": ("support_stability", "support_stability_score", "support_score", "support"),
    "backswing_folding": ("backswing_folding", "backswing_folding_score", "backswing_score", "backswing"),
    "ankle_rigidity": ("ankle_rigidity", "ankle_rigidity_score", "ankle_score", "ankle"),
    "whipping_velocity": ("whipping_velocity", "whipping_velocity_score", "whipping_score", "whipping"),
}

# 中文生物力学标签 → ERR_*（兼容历史归档未写入 error_codes 的记录）
_LABEL_TO_ERROR_CODE: dict[str, str] = {
    "支撑脚位置偏离": "ERR_A2_SUPPORT_WIDE",
    "膝关节过度屈曲": "ERR_KNEE_STIFF",
    "随摆转髋不足": "ERR_FOLLOW_THROUGH",
    "身体重心偏移": "ERR_TORSO_TILT",
}

# 8 大量纲 RED/YELLOW → ERR_*（与前端 ProStudioPanels 对齐）
_INDICATOR_ERROR_CODE: dict[str, str] = {
    "distance_cm": "ERR_A2_SUPPORT_WIDE",
    "toe_angle": "ERR_C2_TOE_POKE",
    "max_folding_angle": "ERR_B1_STRAIGHT_LEG",
    "whipping_velocity": "ERR_FOLLOW_THROUGH",
    "impact_knee_angle": "ERR_KNEE_STIFF",
    "ankle_rigidity": "ERR_C1_LOOSE_ANKLE",
    "support_knee_angle": "ERR_KNEE_STIFF",
    "hip_torsion_angle": "ERR_TORSO_TILT",
}


def _configure_sqlite(engine: Engine) -> None:
    if engine.dialect.name != "sqlite":
        return

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:  # noqa: ARG001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def create_db_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    resolved = url or DATABASE_URL
    connect_args = {"check_same_thread": False} if resolved.startswith("sqlite") else {}
    engine = create_engine(resolved, echo=echo, future=True, connect_args=connect_args)
    _configure_sqlite(engine)
    return engine


engine = create_db_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def _ensure_soft_delete_column(bind: Engine) -> None:
    """存量 SQLite 库补齐 ``shot_attempt_logs.is_deleted``（create_all 不会 ALTER）。"""
    if bind.dialect.name != "sqlite":
        return
    from sqlalchemy import text

    with bind.connect() as conn:
        rows = conn.execute(text("PRAGMA table_info(shot_attempt_logs)")).fetchall()
        if not rows:
            return
        colnames = {row[1] for row in rows}
        if "is_deleted" in colnames:
            return
        conn.execute(
            text(
                "ALTER TABLE shot_attempt_logs "
                "ADD COLUMN is_deleted BOOLEAN NOT NULL DEFAULT 0"
            )
        )
        conn.commit()


def init_db(bind: Engine | None = None) -> None:
    """创建全部科研表（含伦理映射表）。"""
    # 确保模型注册进 metadata
    import models  # noqa: F401

    target = bind or engine
    Base.metadata.create_all(target)
    _ensure_soft_delete_column(target)


@contextmanager
def session_scope(bind: Engine | None = None) -> Generator[Session, None, None]:
    """事务作用域：成功 commit，异常 rollback。"""
    factory = sessionmaker(bind=bind or engine, autoflush=False, autocommit=False, future=True)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _coerce_int_ids(ids: Iterable[Any]) -> list[int]:
    """仅保留可解析为 ORM 主键的整型 ID。"""
    out: list[int] = []
    for raw in ids:
        text = str(raw or "").strip()
        if text.isdigit():
            out.append(int(text))
    return out


def soft_delete_shot_attempts_by_ids(
    ids: Sequence[Any],
    *,
    bind: Engine | None = None,
) -> int:
    """批量软删除 ``shot_attempt_logs``：按 ORM 主键将 ``is_deleted`` 置 True。

    返回实际新标记的行数；非数字 UUID（全局 JSON 主键）会被忽略。
    """
    int_ids = _coerce_int_ids(ids)
    if not int_ids:
        return 0

    from models.shot_attempt_log import ShotAttemptLog

    init_db(bind)
    marked = 0
    with session_scope(bind) as session:
        rows = list(
            session.scalars(
                select(ShotAttemptLog).where(
                    ShotAttemptLog.id.in_(int_ids),
                    ShotAttemptLog.is_deleted.is_(False),
                )
            ).all()
        )
        for row in rows:
            row.is_deleted = True
            marked += 1
    return marked


def soft_delete_shot_attempt_matching(
    *,
    anonymous_id: str,
    session_date: Any,
    total_score: float | None = None,
    bind: Engine | None = None,
) -> int:
    """按被试 + 日期（+ 可选总分）尽力软删除 ORM 行（``is_deleted=True``）。"""
    from datetime import date as date_cls

    from models.shot_attempt_log import ShotAttemptLog

    anon = str(anonymous_id or "").strip()
    if not anon or session_date is None:
        return 0

    if isinstance(session_date, date_cls):
        day = session_date
    else:
        text = str(session_date).strip()[:10]
        try:
            day = date_cls.fromisoformat(text)
        except ValueError:
            return 0

    init_db(bind)
    marked = 0
    with session_scope(bind) as session:
        candidates = list(
            session.scalars(
                select(ShotAttemptLog).where(
                    ShotAttemptLog.anonymous_id == anon,
                    ShotAttemptLog.session_date == day,
                    ShotAttemptLog.is_deleted.is_(False),
                )
            ).all()
        )
        if not candidates:
            return 0
        target = None
        if total_score is not None:
            for row in candidates:
                if row.total_score is not None and abs(float(row.total_score) - float(total_score)) < 0.51:
                    target = row
                    break
        if target is None:
            target = candidates[-1]
        target.is_deleted = True
        marked = 1
    return marked


def hard_delete_shot_attempts_by_ids(
    ids: Sequence[Any],
    *,
    bind: Engine | None = None,
) -> int:
    """【已禁用物理删除】转发为软删除，绝对禁止 ``session.delete`` / DROP。"""
    return soft_delete_shot_attempts_by_ids(ids, bind=bind)


def hard_delete_shot_attempt_matching(
    *,
    anonymous_id: str,
    session_date: Any,
    total_score: float | None = None,
    bind: Engine | None = None,
) -> int:
    """【已禁用物理删除】转发为软删除，绝对禁止 ``session.delete`` / DROP。"""
    return soft_delete_shot_attempt_matching(
        anonymous_id=anonymous_id,
        session_date=session_date,
        total_score=total_score,
        bind=bind,
    )


# --------------------------------------------------------------------------
# 班级 / 实验组对比聚合（教练端科研看板）
# --------------------------------------------------------------------------


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        num = float(value)
        if math.isnan(num) or math.isinf(num):
            return None
        return num
    return None


def _is_soft_deleted(record: dict) -> bool:
    raw = record.get("is_deleted", record.get("isDeleted", False))
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    return str(raw or "").strip().lower() in {"1", "true", "yes", "y"}


def _record_test_date(record: dict) -> str:
    for key in ("testDate", "test_date", "session_date"):
        value = str(record.get(key) or "").strip()
        if len(value) >= 10 and value[4] == "-" and value[7] == "-":
            return value[:10]
    ts = str(record.get("timestamp") or "").strip()
    if len(ts) >= 10 and ts[4] == "-" and ts[7] == "-":
        return ts[:10]
    return ""


def _record_class_group(record: dict) -> str:
    return str(
        record.get("classGroup")
        or record.get("class_group")
        or record.get("cluster_id")
        or ""
    ).strip()


def _record_score(record: dict) -> Optional[float]:
    """综合分优先；缺失时回退五维雷达总分。"""
    direct = _safe_float(record.get("score"))
    if direct is not None:
        return direct
    radar = _extract_radar_dict(record)
    if not isinstance(radar, dict):
        return None
    vals: list[float] = []
    for dim_key, _ in RADAR_COMPARE_DIMS:
        num = _pick_radar_dim(radar, dim_key)
        if num is not None:
            vals.append(num)
    if not vals:
        return None
    return round(sum(vals), 2)


def _extract_radar_dict(record: dict) -> Optional[dict]:
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


def _pick_radar_dim(radar: dict, dim_key: str) -> Optional[float]:
    for alias in _RADAR_DIM_ALIASES.get(dim_key, (dim_key,)):
        num = _safe_float(radar.get(alias))
        if num is not None:
            return num
    return None


def _extract_error_codes(record: dict) -> list[str]:
    """从显式 error_codes / 中文标签 / 指标灯色推导 ERR_* 列表（去重保序）。"""
    codes: list[str] = []
    seen: set[str] = set()

    def _push(raw: Any) -> None:
        text = str(raw or "").strip()
        if not text or text in seen:
            return
        if text.startswith("ERR_"):
            seen.add(text)
            codes.append(text)
            return
        mapped = _LABEL_TO_ERROR_CODE.get(text)
        if mapped and mapped not in seen:
            seen.add(mapped)
            codes.append(mapped)

    for key in ("error_codes", "errorCodes", "biomechanicalErrors", "biomechanical_errors"):
        raw = record.get(key)
        if isinstance(raw, list):
            for item in raw:
                _push(item)

    for key in ("primary_error_code", "primaryErrorCode"):
        raw = record.get(key)
        if isinstance(raw, str):
            _push(raw)

    detail = record.get("scoreDetail") or record.get("score_detail") or {}
    if isinstance(detail, dict):
        for key in ("error_codes", "errorCodes"):
            raw = detail.get(key)
            if isinstance(raw, list):
                for item in raw:
                    _push(item)
        primary = detail.get("primary_error_code") or detail.get("primaryErrorCode")
        if isinstance(primary, str):
            _push(primary)
        diagnosis = detail.get("diagnosis")
        if isinstance(diagnosis, dict):
            for key in ("error_codes", "errorCodes"):
                raw = diagnosis.get(key)
                if isinstance(raw, list):
                    for item in raw:
                        _push(item)
            primary = diagnosis.get("primary_error_code") or diagnosis.get("primaryErrorCode")
            if isinstance(primary, str):
                _push(primary)

        indicators = detail.get("indicators")
        if isinstance(indicators, dict):
            for ind_key, entry in indicators.items():
                if not isinstance(entry, dict):
                    continue
                status = str(entry.get("status") or "").upper()
                if "RED" not in status and "YELLOW" not in status:
                    continue
                mapped = _INDICATOR_ERROR_CODE.get(str(ind_key))
                if mapped and mapped not in seen:
                    seen.add(mapped)
                    codes.append(mapped)

    return codes


def load_global_training_records(path: str | None = None) -> list[dict[str, Any]]:
    """安全读取 ``global_training_db.json``；缺失/损坏时返回空列表。"""
    target = path or DEFAULT_GLOBAL_DB_PATH
    if not os.path.exists(target):
        return []
    try:
        with open(target, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            return []
        return [r for r in data if isinstance(r, dict)]
    except Exception:  # noqa: BLE001
        return []


def empty_compare_cohorts_payload(
    cohort_a: str,
    cohort_b: str,
    *,
    message: str = "暂无足够数据对比",
    sample_a: int = 0,
    sample_b: int = 0,
) -> dict[str, Any]:
    dim_keys = [k for k, _ in RADAR_COMPARE_DIMS]
    dim_labels = [lab for _, lab in RADAR_COMPARE_DIMS]
    return {
        "success": True,
        "sufficient_data": False,
        "message": message,
        "cohort_a": cohort_a,
        "cohort_b": cohort_b,
        "sample_counts": {"a": sample_a, "b": sample_b},
        "trend": {"dates": [], "cohort_a": [], "cohort_b": []},
        "radar": {
            "dimensions": dim_labels,
            "keys": dim_keys,
            "cohort_a": [None] * len(dim_keys),
            "cohort_b": [None] * len(dim_keys),
            "cohort_a_scores": {k: None for k in dim_keys},
            "cohort_b_scores": {k: None for k in dim_keys},
        },
        "error_rates": {"cohort_a": [], "cohort_b": [], "union_codes": []},
    }


def _filter_cohort_records(records: Sequence[dict], cohort: str) -> list[dict]:
    name = (cohort or "").strip()
    if not name:
        return []
    out: list[dict] = []
    for record in records:
        if _is_soft_deleted(record):
            continue
        if _record_class_group(record) != name:
            continue
        out.append(record)
    return out


def _daily_trend(records: Sequence[dict]) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for record in records:
        day = _record_test_date(record)
        if not day:
            continue
        score = _record_score(record)
        if score is None:
            continue
        buckets[day].append(score)

    points: list[dict[str, Any]] = []
    for day in sorted(buckets.keys()):
        vals = buckets[day]
        avg = round(sum(vals) / len(vals), 2)
        if len(vals) >= 2:
            try:
                variance = round(statistics.pvariance(vals), 4)
            except statistics.StatisticsError:
                variance = 0.0
        else:
            variance = 0.0
        points.append(
            {
                "date": day,
                "average_score": avg,
                "score_variance": variance,
                "n": len(vals),
            }
        )
    return points


def _radar_means(records: Sequence[dict]) -> dict[str, Optional[float]]:
    buckets: dict[str, list[float]] = {k: [] for k, _ in RADAR_COMPARE_DIMS}
    for record in records:
        radar = _extract_radar_dict(record)
        if not isinstance(radar, dict):
            continue
        for dim_key, _ in RADAR_COMPARE_DIMS:
            num = _pick_radar_dim(radar, dim_key)
            if num is not None:
                buckets[dim_key].append(num)
    return {
        key: (round(sum(vals) / len(vals), 2) if vals else None)
        for key, vals in buckets.items()
    }


def _error_rate_rows(records: Sequence[dict]) -> list[dict[str, Any]]:
    total = len(records)
    if total <= 0:
        return []
    counter: Counter[str] = Counter()
    for record in records:
        # 同一条记录同一错误码只计一次
        for code in _extract_error_codes(record):
            counter[code] += 1
    rows = [
        {
            "code": code,
            "count": count,
            "rate": round(count / total, 4),
            "percentage": round(100.0 * count / total, 2),
        }
        for code, count in counter.most_common()
    ]
    return rows


def compare_cohorts(
    cohort_a: str,
    cohort_b: str,
    *,
    records: Sequence[dict] | None = None,
    min_samples: int = MIN_COHORT_SAMPLES,
    global_db_path: str | None = None,
) -> dict[str, Any]:
    """聚合两个班级/实验组的三维对比数据。

    返回结构：
      - trend：按日期的 average_score / score_variance
      - radar：五维（助跑/支撑/后摆/踝锁/鞭打）均值
      - error_rates：高频 ERR_* 发生占比

    任一侧样本为空或低于 ``min_samples`` 时 ``sufficient_data=False``，
    并填入「暂无足够数据对比」占位结构（前端据此渲染占位符）。
    """
    name_a = (cohort_a or "").strip()
    name_b = (cohort_b or "").strip()
    if not name_a or not name_b:
        return empty_compare_cohorts_payload(
            name_a,
            name_b,
            message="请选择两个有效的对比班级",
        )
    if name_a == name_b:
        return empty_compare_cohorts_payload(
            name_a,
            name_b,
            message="对比班级不可相同",
        )

    try:
        source = list(records) if records is not None else load_global_training_records(global_db_path)
        rows_a = _filter_cohort_records(source, name_a)
        rows_b = _filter_cohort_records(source, name_b)
        n_a, n_b = len(rows_a), len(rows_b)

        if n_a < min_samples or n_b < min_samples:
            return empty_compare_cohorts_payload(
                name_a,
                name_b,
                message="暂无足够数据对比",
                sample_a=n_a,
                sample_b=n_b,
            )

        trend_a = _daily_trend(rows_a)
        trend_b = _daily_trend(rows_b)
        # 趋势至少需要一侧有按日均分点；否则仍视为不足
        if not trend_a and not trend_b:
            return empty_compare_cohorts_payload(
                name_a,
                name_b,
                message="暂无足够数据对比",
                sample_a=n_a,
                sample_b=n_b,
            )

        dates = sorted({p["date"] for p in trend_a} | {p["date"] for p in trend_b})
        radar_a = _radar_means(rows_a)
        radar_b = _radar_means(rows_b)
        dim_keys = [k for k, _ in RADAR_COMPARE_DIMS]
        dim_labels = [lab for _, lab in RADAR_COMPARE_DIMS]

        err_a = _error_rate_rows(rows_a)
        err_b = _error_rate_rows(rows_b)
        union_codes = sorted(
            {row["code"] for row in err_a} | {row["code"] for row in err_b}
        )

        return {
            "success": True,
            "sufficient_data": True,
            "message": None,
            "cohort_a": name_a,
            "cohort_b": name_b,
            "sample_counts": {"a": n_a, "b": n_b},
            "trend": {
                "dates": dates,
                "cohort_a": trend_a,
                "cohort_b": trend_b,
            },
            "radar": {
                "dimensions": dim_labels,
                "keys": dim_keys,
                "cohort_a": [radar_a.get(k) for k in dim_keys],
                "cohort_b": [radar_b.get(k) for k in dim_keys],
                "cohort_a_scores": radar_a,
                "cohort_b_scores": radar_b,
            },
            "error_rates": {
                "cohort_a": err_a,
                "cohort_b": err_b,
                "union_codes": union_codes,
            },
        }
    except Exception as exc:  # noqa: BLE001
        return empty_compare_cohorts_payload(
            name_a,
            name_b,
            message=f"暂无足够数据对比（聚合异常：{exc}）",
        )


# --------------------------------------------------------------------------
# B 组课后复盘：最佳 vs 待改进 双关键帧对比
# --------------------------------------------------------------------------

DEFAULT_WEB_SESSION_LOG_PATH = os.path.join(SCRIPT_DIR, "B_group_web_sessions_log.json")

# 复盘看板展示用短标签（对齐前端错误口径，优先口语化）
_MAIN_ERROR_DISPLAY: dict[str, str] = {
    "ERR_A2_SUPPORT_WIDE": "支撑脚过宽",
    "ERR_SUPPORT_TOO_WIDE": "支撑脚过宽",
    "ERR_SUPPORT_LATERAL": "支撑脚过宽",
    "支撑脚位置偏离": "支撑脚过宽",
    "ERR_WARMUP_CLOSE": "支撑脚过近",
    "ERR_SUPPORT_TOO_CLOSE": "支撑脚过近",
    "ERR_A1_SUPPORT_BACK": "支撑脚偏后",
    "ERR_SUPPORT_AP": "支撑脚偏后",
    "ERR_KNEE_STIFF": "膝关节过度屈曲",
    "膝关节过度屈曲": "膝关节过度屈曲",
    "ERR_B1_STRAIGHT_LEG": "后摆直腿",
    "ERR_B2_SHANK_ONLY": "仅小腿弹射",
    "ERR_SWING_FOLD": "后摆折叠不足",
    "ERR_C1_LOOSE_ANKLE": "踝关节松弛",
    "ERR_ANKLE_LOOSE": "踝关节松弛",
    "ERR_C2_TOE_POKE": "脚尖捅球",
    "ERR_FOLLOW_THROUGH": "随摆转髋不足",
    "随摆转髋不足": "随摆转髋不足",
    "ERR_TORSO_TILT": "身体重心偏移",
    "身体重心偏移": "身体重心偏移",
    "ERR_APPROACH_TOO_STRAIGHT": "助跑过直",
    "ERR_APPROACH_TOO_WIDE": "助跑过宽",
    "ERR_APPROACH_ANGLE": "助跑角度偏差",
    "ERR_EARLY_DECELERATION": "过早减速",
}

_INDICATOR_MAIN_ERROR: dict[str, str] = {
    "distance_cm": "支撑脚过宽",
    "toe_angle": "脚尖捅球",
    "max_folding_angle": "后摆折叠不足",
    "whipping_velocity": "随摆转髋不足",
    "impact_knee_angle": "膝关节过度屈曲",
    "ankle_rigidity": "踝关节松弛",
    "support_knee_angle": "膝关节过度屈曲",
    "hip_torsion_angle": "身体重心偏移",
}


def _ms_to_date_str(ms: Any) -> str:
    """毫秒时间戳 → YYYY-MM-DD；非法时返回空串。"""
    try:
        value = float(ms)
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    # 兼容秒级时间戳
    if value < 1e11:
        value *= 1000.0
    try:
        from datetime import datetime

        return datetime.fromtimestamp(value / 1000.0).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return ""


def _normalize_image_url(raw: Any) -> Optional[str]:
    """将触球关键帧整理为可直接渲染的 data URL / 相对路径。"""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("data:image"):
        return text
    # 纯 Base64（无 data URI 前缀）→ 默认 JPEG
    looks_like_path = text.startswith(("/", ".", "http://", "https://", "file:", "uploads/", "\\\\"))
    if len(text) >= 32 and not looks_like_path and "\\" not in text[:8]:
        sample = text[:120].replace("+", "").replace("/", "").replace("=", "").replace("\n", "")
        if sample and sample.isalnum():
            return f"data:image/jpeg;base64,{text}"
    # 相对/绝对路径或 http(s) URL
    return text


def _attempt_total_score(attempt: Mapping[str, Any] | dict) -> Optional[float]:
    """从 Attempt / 归档记录中提取综合总分。"""
    if not isinstance(attempt, Mapping):
        return None
    for key in ("total_score", "totalScore", "score"):
        num = _safe_float(attempt.get(key))
        if num is not None:
            return round(num, 2)
    report = attempt.get("reportData") or attempt.get("report_data") or {}
    if isinstance(report, Mapping):
        for key in ("score", "total_score", "totalScore"):
            num = _safe_float(report.get(key))
            if num is not None:
                return round(num, 2)
        detail = report.get("scoreDetail") or report.get("score_detail") or {}
        if isinstance(detail, Mapping):
            num = _safe_float(detail.get("TotalScore") or detail.get("total_score"))
            if num is not None:
                return round(num, 2)
    detail = attempt.get("scoreDetail") or attempt.get("score_detail") or {}
    if isinstance(detail, Mapping):
        num = _safe_float(detail.get("TotalScore") or detail.get("total_score"))
        if num is not None:
            return round(num, 2)
    return None


def _attempt_image_url(attempt: Mapping[str, Any] | dict) -> Optional[str]:
    """提取触球瞬间关键帧（Base64 data URL 或可渲染路径）。"""
    if not isinstance(attempt, Mapping):
        return None
    for key in (
        "impactFrameBase64",
        "impact_frame_base64",
        "impactFrameImage",
        "impact_frame_image",
        "image_url",
        "imageUrl",
        "telestrationImagePath",
        "telestration_image_path",
    ):
        url = _normalize_image_url(attempt.get(key))
        if url:
            return url
    report = attempt.get("reportData") or attempt.get("report_data") or {}
    if isinstance(report, Mapping):
        for key in ("impactFrameImage", "impact_frame_image", "impactFrameBase64"):
            url = _normalize_image_url(report.get(key))
            if url:
                return url
    return None


def _display_main_error(raw: Any) -> Optional[str]:
    text = str(raw or "").strip()
    if not text:
        return None
    if text in _MAIN_ERROR_DISPLAY:
        return _MAIN_ERROR_DISPLAY[text]
    if text.startswith("PASS_"):
        return None
    # 已是中文短标签则直接返回
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return text
    return text


def _attempt_main_error(attempt: Mapping[str, Any] | dict) -> Optional[str]:
    """待改进尝试的主要错误短标签（优先诊断码 / 红灯量纲 / 生物力学分类）。"""
    if not isinstance(attempt, Mapping):
        return None

    for key in ("main_error", "mainError", "primaryError", "primary_error"):
        label = _display_main_error(attempt.get(key))
        if label:
            return label

    # 显式 ERR_* / 中文分类
    as_record = dict(attempt)
    report = attempt.get("reportData") or attempt.get("report_data")
    if isinstance(report, Mapping):
        # 把嵌套报告字段摊平到临时 dict，复用 _extract_error_codes
        for nest_key in ("scoreDetail", "score_detail", "biomechanicalErrors", "error_codes"):
            if nest_key in report and nest_key not in as_record:
                as_record[nest_key] = report[nest_key]
        detail = report.get("scoreDetail") or report.get("score_detail")
        if isinstance(detail, Mapping):
            as_record.setdefault("scoreDetail", detail)

    codes = _extract_error_codes(as_record)
    for code in codes:
        label = _display_main_error(code)
        if label:
            return label

    # 红灯量纲优先
    detail = as_record.get("scoreDetail") or as_record.get("score_detail") or {}
    if isinstance(detail, Mapping):
        indicators = detail.get("indicators")
        if isinstance(indicators, Mapping):
            red_first: list[str] = []
            yellow_first: list[str] = []
            for ind_key, entry in indicators.items():
                if not isinstance(entry, Mapping):
                    continue
                status = str(entry.get("status") or "").upper()
                mapped = _INDICATOR_MAIN_ERROR.get(str(ind_key))
                if not mapped:
                    continue
                if "RED" in status:
                    red_first.append(mapped)
                elif "YELLOW" in status:
                    yellow_first.append(mapped)
            if red_first:
                return red_first[0]
            if yellow_first:
                return yellow_first[0]

    # biomechanicalErrors 中文列表
    for key in ("biomechanicalErrors", "biomechanical_errors"):
        raw = as_record.get(key)
        if isinstance(raw, list):
            for item in raw:
                label = _display_main_error(item)
                if label:
                    return label

    # painPoint 兜底：取第一句短文本
    pain = None
    if isinstance(report, Mapping):
        pain = report.get("painPoint") or report.get("pain_point")
    if pain is None:
        pain = attempt.get("painPoint") or attempt.get("pain_point")
    if isinstance(pain, str) and pain.strip():
        snippet = pain.strip().replace("\n", " ")
        # 截断过长 AI 文案，复盘对比只需短标签
        if len(snippet) > 24:
            snippet = snippet[:24].rstrip() + "…"
        return snippet
    return None


def _attempt_id(attempt: Mapping[str, Any] | dict, fallback_index: int) -> Any:
    if not isinstance(attempt, Mapping):
        return fallback_index
    for key in ("attempt_id", "attemptId", "attemptNumber", "attempt_number", "id"):
        raw = attempt.get(key)
        if raw is None or raw == "":
            continue
        if isinstance(raw, bool):
            continue
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            # 保留整数 attemptNumber；UUID 字符串另走
            if float(raw).is_integer():
                return int(raw)
            return float(raw)
        text = str(raw).strip()
        if text.isdigit():
            return int(text)
        if text:
            return text
    return fallback_index


def build_comparison_frames(
    attempts: Sequence[Mapping[str, Any] | dict] | None,
    *,
    min_count: int = 2,
) -> Optional[dict[str, Any]]:
    """按 total_score 极值筛选 best / improve 双关键帧。

    有效尝试数 ``< min_count`` 时返回 ``None``（调用方应跳过对比逻辑）。
    """
    if not attempts:
        return None

    scored: list[tuple[float, int, Mapping[str, Any]]] = []
    for idx, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping):
            continue
        score = _attempt_total_score(attempt)
        if score is None:
            continue
        scored.append((score, idx, attempt))

    if len(scored) < min_count:
        return None

    # 同分时：best 取较晚尝试；improve 取较早尝试（稳定、可复现）
    best_score, best_idx, best_attempt = max(scored, key=lambda row: (row[0], row[1]))
    improve_score, improve_idx, improve_attempt = min(scored, key=lambda row: (row[0], -row[1]))

    best_payload: dict[str, Any] = {
        "score": best_score,
        "image_url": _attempt_image_url(best_attempt),
        "attempt_id": _attempt_id(best_attempt, best_idx + 1),
    }
    improve_payload: dict[str, Any] = {
        "score": improve_score,
        "image_url": _attempt_image_url(improve_attempt),
        "attempt_id": _attempt_id(improve_attempt, improve_idx + 1),
        "main_error": _attempt_main_error(improve_attempt),
    }
    return {"best": best_payload, "improve": improve_payload}


def load_b_group_web_sessions(path: str | None = None) -> list[dict[str, Any]]:
    """安全读取 B 组 Web 归档池；缺失/损坏时返回空列表。"""
    target = path or DEFAULT_WEB_SESSION_LOG_PATH
    if not os.path.exists(target):
        return []
    try:
        with open(target, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            sessions = data.get("sessions") or []
        elif isinstance(data, list):
            sessions = data
        else:
            return []
        return [s for s in sessions if isinstance(s, dict)]
    except Exception:  # noqa: BLE001
        return []


def _session_date_str(session: Mapping[str, Any]) -> str:
    """从课时实体推断 YYYY-MM-DD。"""
    for key in ("session_date", "sessionDate", "testDate", "test_date"):
        value = str(session.get(key) or "").strip()
        if len(value) >= 10 and value[4] == "-" and value[7] == "-":
            return value[:10]
    day = _ms_to_date_str(session.get("timestamp"))
    if day:
        return day
    attempts = session.get("attempts")
    if isinstance(attempts, list):
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                continue
            day = _ms_to_date_str(attempt.get("timestamp"))
            if day:
                return day
            report = attempt.get("reportData") or attempt.get("report_data") or {}
            if isinstance(report, Mapping):
                generated = str(report.get("generatedAt") or report.get("generated_at") or "").strip()
                if len(generated) >= 10 and generated[4] == "-" and generated[7] == "-":
                    return generated[:10]
    return ""


def _normalize_flat_record_as_attempt(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    """把 global_training_db 扁平记录折成 Attempt 形态，便于统一极值筛选。"""
    score = _record_score(dict(record))
    return {
        "attemptNumber": index,
        "attempt_id": record.get("id") or index,
        "timestamp": record.get("timestamp"),
        "total_score": score,
        "score": score,
        "impactFrameBase64": record.get("impactFrameBase64") or record.get("impact_frame_base64"),
        "telestrationImagePath": record.get("telestrationImagePath"),
        "biomechanicalErrors": record.get("biomechanicalErrors")
        or record.get("biomechanical_errors")
        or [],
        "scoreDetail": record.get("scoreDetail") or record.get("score_detail"),
        "main_error": None,
        "painPoint": None,
        "reportData": {
            "score": score,
            "impactFrameImage": record.get("impactFrameBase64"),
            "scoreDetail": record.get("scoreDetail") or record.get("score_detail"),
            "biomechanicalErrors": record.get("biomechanicalErrors")
            or record.get("biomechanical_errors")
            or [],
            "painPoint": (str(record.get("aiFeedback") or "").split("\n") or [None])[0],
        },
    }


def collect_student_review_attempts(
    student_id: str,
    *,
    session_date: str | None = None,
    session_id: str | None = None,
    sessions: Sequence[Mapping[str, Any]] | None = None,
    global_records: Sequence[Mapping[str, Any]] | None = None,
    web_session_path: str | None = None,
    global_db_path: str | None = None,
    include_global_fallback: bool = True,
) -> list[dict[str, Any]]:
    """收集某学生在指定日期/课时内的全部有效 Attempt（含总分）。"""
    sid = (student_id or "").strip()
    if not sid:
        return []
    day_filter = (session_date or "").strip()[:10] or None
    session_filter = (session_id or "").strip() or None

    source_sessions = (
        list(sessions) if sessions is not None else load_b_group_web_sessions(web_session_path)
    )
    attempts: list[dict[str, Any]] = []

    for session in source_sessions:
        if not isinstance(session, Mapping):
            continue
        session_sid = str(session.get("studentId") or session.get("student_id") or "").strip()
        if session_sid != sid:
            continue
        if session_filter:
            if str(session.get("id") or "").strip() != session_filter:
                continue
        elif day_filter:
            if _session_date_str(session) != day_filter:
                continue
        raw_attempts = session.get("attempts")
        if not isinstance(raw_attempts, list):
            continue
        for item in raw_attempts:
            if isinstance(item, dict):
                attempts.append(item)

    # 课时池完全无数据时，回退到全局归档库（延时反馈 B 组）
    if include_global_fallback and not attempts:
        records = (
            list(global_records)
            if global_records is not None
            else load_global_training_records(global_db_path)
        )
        flat: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, Mapping) or _is_soft_deleted(dict(record)):
                continue
            rec_sid = str(
                record.get("studentId") or record.get("student_id") or record.get("anonymous_id") or ""
            ).strip()
            if rec_sid != sid:
                continue
            rec_type = str(record.get("type") or record.get("group") or "").strip().lower()
            # 无类型时仍接收；有类型则优先 delayed / B
            if rec_type and rec_type not in {
                "delayed",
                "b",
                "group_b",
                "group_b_delayed",
                "延时反馈",
            }:
                continue
            if day_filter and _record_test_date(dict(record)) != day_filter:
                continue
            if _attempt_total_score(record) is None and _record_score(dict(record)) is None:
                continue
            flat.append(dict(record))
        # 按时间排序后编号
        flat.sort(key=lambda r: str(r.get("timestamp") or ""))
        for idx, record in enumerate(flat, start=1):
            attempts.append(_normalize_flat_record_as_attempt(record, idx))

    return attempts


def student_review_summary(
    student_id: str,
    *,
    session_date: str | None = None,
    session_id: str | None = None,
    sessions: Sequence[Mapping[str, Any]] | None = None,
    global_records: Sequence[Mapping[str, Any]] | None = None,
    web_session_path: str | None = None,
    global_db_path: str | None = None,
) -> dict[str, Any]:
    """B 组课后复盘聚合：有效尝试列表 + 极值双关键帧对比。"""
    sid = (student_id or "").strip()
    attempts = collect_student_review_attempts(
        sid,
        session_date=session_date,
        session_id=session_id,
        sessions=sessions,
        global_records=global_records,
        web_session_path=web_session_path,
        global_db_path=global_db_path,
    )

    scored_rows: list[dict[str, Any]] = []
    for idx, attempt in enumerate(attempts):
        score = _attempt_total_score(attempt)
        if score is None:
            continue
        scored_rows.append(
            {
                "attempt_id": _attempt_id(attempt, idx + 1),
                "score": score,
                "has_image": bool(_attempt_image_url(attempt)),
            }
        )

    comparison = build_comparison_frames(attempts)
    return {
        "success": True,
        "student_id": sid,
        "session_date": (session_date or "").strip()[:10] or None,
        "session_id": (session_id or "").strip() or None,
        "attempt_count": len(scored_rows),
        "attempts": scored_rows,
        "comparison_frames": comparison,
        "comparison_available": comparison is not None,
    }


# --------------------------------------------------------------------------
# 自动备份守护（户外长期实验数据绝对安全）
# --------------------------------------------------------------------------


def _backup_timestamp(now: datetime | None = None) -> str:
    anchor = now or datetime.now()
    return anchor.strftime("%Y%m%d_%H%M")


def _iter_backup_files(backup_dir: str) -> list[str]:
    if not os.path.isdir(backup_dir):
        return []
    out: list[str] = []
    for name in os.listdir(backup_dir):
        if not name.startswith(BACKUP_FILENAME_PREFIX):
            continue
        if not name.endswith(BACKUP_FILENAME_SUFFIX):
            continue
        path = os.path.join(backup_dir, name)
        if os.path.isfile(path):
            out.append(path)
    return out


def prune_old_backups(
    backup_dir: str | None = None,
    *,
    retention_days: int = BACKUP_RETENTION_DAYS,
    now: datetime | None = None,
) -> int:
    """删除超过保留天数的备份文件，返回删除数量。"""
    root = backup_dir or DEFAULT_BACKUP_DIR
    if retention_days < 1:
        return 0
    cutoff = (now or datetime.now()) - timedelta(days=retention_days)
    removed = 0
    for path in _iter_backup_files(root):
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
        except OSError:
            continue
        if mtime >= cutoff:
            continue
        try:
            os.remove(path)
            removed += 1
        except OSError:
            continue
    return removed


def latest_backup_mtime(backup_dir: str | None = None) -> Optional[datetime]:
    """返回备份目录中最新 ``.bak`` 的修改时间。"""
    newest: Optional[datetime] = None
    for path in _iter_backup_files(backup_dir or DEFAULT_BACKUP_DIR):
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
        except OSError:
            continue
        if newest is None or mtime > newest:
            newest = mtime
    return newest


def should_run_backup(
    backup_dir: str | None = None,
    *,
    interval_hours: float = BACKUP_INTERVAL_HOURS,
    now: datetime | None = None,
) -> bool:
    """距上次成功备份已超过 ``interval_hours``（或尚无备份）时返回 True。"""
    anchor = now or datetime.now()
    with _backup_last_run_lock:
        last_memory = _backup_last_run_at
    last_disk = latest_backup_mtime(backup_dir)
    last = last_memory
    if last_disk is not None and (last is None or last_disk > last):
        last = last_disk
    if last is None:
        return True
    return (anchor - last) >= timedelta(hours=float(interval_hours))


def create_database_backup(
    *,
    backup_dir: str | None = None,
    global_db_path: str | None = None,
    sqlite_db_path: str | None = None,
    now: datetime | None = None,
    force: bool = False,
) -> Optional[str]:
    """将主 JSON / SQLite 数据源打包复制到 ``./backups/db_backup_YYYYMMDD_HHMM.bak``。

    返回备份文件绝对路径；若无需备份或无可备份源则返回 ``None``。
    """
    global _backup_last_run_at

    root = backup_dir or DEFAULT_BACKUP_DIR
    os.makedirs(root, exist_ok=True)
    if not force and not should_run_backup(root, now=now):
        return None

    sources: list[tuple[str, str]] = []
    json_path = global_db_path or DEFAULT_GLOBAL_DB_PATH
    sqlite_path = sqlite_db_path or DEFAULT_DB_PATH
    if os.path.isfile(json_path):
        sources.append((json_path, os.path.basename(json_path)))
    if os.path.isfile(sqlite_path):
        sources.append((sqlite_path, os.path.basename(sqlite_path)))
    if not sources:
        return None

    stamp = _backup_timestamp(now)
    dest_name = f"{BACKUP_FILENAME_PREFIX}{stamp}{BACKUP_FILENAME_SUFFIX}"
    dest_path = os.path.join(root, dest_name)
    # 同一分钟内重复触发时追加序号，避免覆盖
    seq = 1
    while os.path.exists(dest_path):
        dest_name = f"{BACKUP_FILENAME_PREFIX}{stamp}_{seq:02d}{BACKUP_FILENAME_SUFFIX}"
        dest_path = os.path.join(root, dest_name)
        seq += 1

    tmp_path = dest_path + ".tmp"
    try:
        with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for src, arcname in sources:
                zf.write(src, arcname=arcname)
        os.replace(tmp_path, dest_path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise

    with _backup_last_run_lock:
        _backup_last_run_at = now or datetime.now()
    prune_old_backups(root, now=now)
    return dest_path


def start_auto_backup_daemon(
    *,
    backup_dir: str | None = None,
    interval_hours: float = BACKUP_INTERVAL_HOURS,
    check_every_seconds: float = 3600.0,
) -> bool:
    """启动后台备份守护线程（幂等）。

    - 进程启动时立即检查：若距上次备份 ≥ 12h（含首次），立刻备份；
    - 之后每小时巡检一次，满足间隔则再备份并清理 30 天外旧包。
    """
    global _backup_daemon_started, _backup_thread

    with _backup_daemon_lock:
        if _backup_daemon_started and _backup_thread is not None and _backup_thread.is_alive():
            return False

        root = backup_dir or DEFAULT_BACKUP_DIR
        os.makedirs(root, exist_ok=True)
        _backup_stop_event.clear()

        def _loop() -> None:
            # 启动即守护：每天首次启动 / 超过 12h 立刻落盘
            try:
                path = create_database_backup(backup_dir=root, force=False)
                if path:
                    print(f"[db-backup] 已创建自动备份：{path}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[db-backup] 启动备份失败：{exc}", flush=True)

            wait_s = max(30.0, float(check_every_seconds))
            while not _backup_stop_event.wait(timeout=wait_s):
                try:
                    if not should_run_backup(root, interval_hours=interval_hours):
                        continue
                    path = create_database_backup(backup_dir=root, force=True)
                    if path:
                        print(f"[db-backup] 已创建自动备份：{path}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[db-backup] 周期备份失败：{exc}", flush=True)

        _backup_thread = threading.Thread(
            target=_loop,
            name="db-auto-backup-daemon",
            daemon=True,
        )
        _backup_thread.start()
        _backup_daemon_started = True
        return True


def stop_auto_backup_daemon(timeout: float = 2.0) -> None:
    """停止备份守护线程（主要用于单测）。"""
    global _backup_daemon_started, _backup_thread
    _backup_stop_event.set()
    thread = _backup_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout)
    with _backup_daemon_lock:
        _backup_daemon_started = False
        _backup_thread = None
