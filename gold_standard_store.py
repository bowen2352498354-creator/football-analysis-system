# -*- coding: utf-8 -*-
"""Versioned gold-standard imports and criterion-validity reports.

The store keeps uploaded source files immutable, normalizes annotations into a
canonical schema, and aligns them with archived system attempts without ever
rewriting either source.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from empirical_thresholds import classify_impact_knee_angle
from measurement_validation import validate_measurement_pairs
from research_integrity import threshold_version


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STORE_ROOT = SCRIPT_DIR / "gold_standard_data"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".json"}

CANONICAL_FIELDS = (
    "sampleId",
    "attemptId",
    "videoHash",
    "sourceType",
    "annotatorId",
    "metric",
    "referenceValue",
    "unit",
    "referenceFrame",
    "fps",
    "swingLeg",
    "cameraView",
    "annotationVersion",
    "systemValue",
    "systemFrame",
)

_ALIASES = {
    "sampleid": "sampleId",
    "sample": "sampleId",
    "sample_id": "sampleId",
    "attemptid": "attemptId",
    "attempt_id": "attemptId",
    "recordid": "attemptId",
    "videohash": "videoHash",
    "video_hash": "videoHash",
    "sha256": "videoHash",
    "sourcetype": "sourceType",
    "source_type": "sourceType",
    "source": "sourceType",
    "annotatorid": "annotatorId",
    "annotator_id": "annotatorId",
    "annotator": "annotatorId",
    "metrickey": "metric",
    "metric_key": "metric",
    "metric": "metric",
    "referencevalue": "referenceValue",
    "reference_value": "referenceValue",
    "goldvalue": "referenceValue",
    "value": "referenceValue",
    "unit": "unit",
    "referenceframe": "referenceFrame",
    "reference_frame": "referenceFrame",
    "impactframe": "referenceFrame",
    "impact_frame": "referenceFrame",
    "fps": "fps",
    "swinglege": "swingLeg",
    "swingleg": "swingLeg",
    "swing_leg": "swingLeg",
    "cameraview": "cameraView",
    "camera_view": "cameraView",
    "view": "cameraView",
    "annotationversion": "annotationVersion",
    "annotation_version": "annotationVersion",
    "systemvalue": "systemValue",
    "system_value": "systemValue",
    "systemframe": "systemFrame",
    "system_frame": "systemFrame",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_filename(filename: str) -> str:
    name = Path(filename or "annotations.csv").name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name) or "annotations.csv"


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalize_header(value: Any) -> str:
    text = str(value or "").strip().lower().replace("\ufeff", "")
    compact = re.sub(r"[\s\-]+", "_", text)
    return _ALIASES.get(compact, _ALIASES.get(compact.replace("_", ""), ""))


def _parse_rows(filename: str, content: bytes) -> list[dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("仅支持 CSV、XLSX 或 JSON 标注文件")
    if not content:
        raise ValueError("上传文件为空")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("上传文件超过 10 MB 限制")

    if suffix == ".csv":
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("gb18030")
        sample = text[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        return [dict(row) for row in csv.DictReader(io.StringIO(text), dialect=dialect)]

    if suffix == ".json":
        payload = json.loads(content.decode("utf-8-sig"))
        rows = payload.get("records") if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("JSON 必须是对象数组，或包含 records 对象数组")
        return [dict(row) for row in rows]

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - requirements includes pandas
        raise RuntimeError("读取 XLSX 需要 pandas 与 openpyxl") from exc
    frame = pd.read_excel(io.BytesIO(content), dtype=object)
    frame = frame.where(frame.notna(), None)
    return frame.to_dict(orient="records")


def _canonicalize(raw: Mapping[str, Any], row_number: int) -> tuple[dict[str, Any], list[str], list[str]]:
    row: dict[str, Any] = {field: None for field in CANONICAL_FIELDS}
    for key, value in raw.items():
        canonical = _normalize_header(key)
        if canonical:
            row[canonical] = value

    for key in (
        "sampleId",
        "attemptId",
        "videoHash",
        "sourceType",
        "annotatorId",
        "metric",
        "unit",
        "swingLeg",
        "cameraView",
        "annotationVersion",
    ):
        row[key] = str(row.get(key) or "").strip()

    row["sourceType"] = row["sourceType"].lower()
    source_aliases = {
        "kinovea": "kinovea",
        "manual": "manual",
        "人工": "manual",
        "mocap": "mocap",
        "motioncapture": "mocap",
        "motion_capture": "mocap",
    }
    row["sourceType"] = source_aliases.get(row["sourceType"].replace(" ", ""), row["sourceType"])
    row["metric"] = row["metric"].lower()
    row["videoHash"] = row["videoHash"].lower()
    row["swingLeg"] = row["swingLeg"].lower()
    row["cameraView"] = row["cameraView"].lower()
    row["cameraView"] = {"side": "lateral", "侧面": "lateral", "frontal": "front", "正面": "front"}.get(
        row["cameraView"], row["cameraView"]
    )

    row["referenceValue"] = _finite(row.get("referenceValue"))
    row["referenceFrame"] = _finite(row.get("referenceFrame"))
    row["fps"] = _finite(row.get("fps"))
    row["systemValue"] = _finite(row.get("systemValue"))
    row["systemFrame"] = _finite(row.get("systemFrame"))
    row["rowNumber"] = row_number

    errors: list[str] = []
    warnings: list[str] = []
    required_text = (
        "sampleId",
        "attemptId",
        "videoHash",
        "sourceType",
        "annotatorId",
        "metric",
        "unit",
        "swingLeg",
        "cameraView",
        "annotationVersion",
    )
    for key in required_text:
        if not row[key]:
            errors.append(f"{key} 不能为空")
    if row["referenceValue"] is None:
        errors.append("referenceValue 必须是有限数值")
    if row["referenceFrame"] is None or row["referenceFrame"] < 0:
        errors.append("referenceFrame 必须是非负数")
    if row["fps"] is None or not 0 < row["fps"] <= 1000:
        errors.append("fps 必须在 (0, 1000] 范围")
    if row["videoHash"] and not re.fullmatch(r"[0-9a-f]{64}", row["videoHash"]):
        errors.append("videoHash 必须是 64 位 SHA-256")
    if row["sourceType"] and row["sourceType"] not in {"kinovea", "manual", "mocap"}:
        errors.append("sourceType 仅允许 kinovea、manual、mocap")
    if row["swingLeg"] and row["swingLeg"] not in {"left", "right"}:
        errors.append("swingLeg 仅允许 left 或 right")
    if row["metric"] == "impact_knee_angle":
        if row["referenceValue"] is not None and not 0 <= row["referenceValue"] <= 180:
            errors.append("impact_knee_angle 必须在 0–180° 范围")
        if row["unit"].lower() not in {"deg", "degree", "degrees", "°"}:
            errors.append("impact_knee_angle 的 unit 必须为 deg")
        if row["cameraView"] == "front":
            errors.append("膝角金标准不接受正面机位，必须使用侧面机位")
        elif row["cameraView"] != "lateral":
            warnings.append("非标准侧面机位，效度报告中应单独说明")
    return row, errors, warnings


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def preview_import(filename: str, content: bytes, *, store_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(store_root or DEFAULT_STORE_ROOT)
    safe_name = _safe_filename(filename)
    raw_rows = _parse_rows(safe_name, content)
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    duplicate_count = 0

    for index, raw in enumerate(raw_rows, start=2):
        row, errors, row_warnings = _canonicalize(raw, index)
        dedupe_key = (row["sampleId"], row["attemptId"], row["annotatorId"], row["metric"])
        if all(dedupe_key) and dedupe_key in seen:
            errors.append("同一样本、attempt、标注员和指标重复")
            duplicate_count += 1
        if all(dedupe_key):
            seen.add(dedupe_key)
        if errors:
            invalid.append({"rowNumber": index, "errors": errors, "row": row})
        else:
            valid.append(row)
        if row_warnings:
            warnings.append({"rowNumber": index, "warnings": row_warnings})

    token = uuid.uuid4().hex
    digest = hashlib.sha256(content).hexdigest()
    package = {
        "previewToken": token,
        "createdAt": _now_iso(),
        "sourceFilename": safe_name,
        "sourceSha256": digest,
        "sourceBase64": base64.b64encode(content).decode("ascii"),
        "validRows": valid,
        "invalidRows": invalid,
        "warnings": warnings,
        "duplicateCount": duplicate_count,
        "totalRows": len(raw_rows),
    }
    _write_json(root / ".previews" / f"{token}.json", package)
    return {
        "success": bool(valid),
        "canCommit": bool(valid),
        "previewToken": token,
        "sourceFilename": safe_name,
        "sourceSha256": digest,
        "totalRows": len(raw_rows),
        "validRowCount": len(valid),
        "invalidRowCount": len(invalid),
        "duplicateCount": duplicate_count,
        "invalidRows": invalid[:100],
        "warnings": warnings[:100],
        "canonicalFields": list(CANONICAL_FIELDS),
    }


def commit_preview(preview_token: str, *, store_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(store_root or DEFAULT_STORE_ROOT)
    token = re.sub(r"[^a-f0-9]", "", str(preview_token or "").lower())
    preview_path = root / ".previews" / f"{token}.json"
    if len(token) != 32 or not preview_path.is_file():
        raise LookupError("预检令牌不存在或已提交")
    package = json.loads(preview_path.read_text(encoding="utf-8"))
    valid_rows = package.get("validRows") or []
    if not valid_rows:
        raise ValueError("没有可提交的有效标注行")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dataset_id = f"GS_{stamp}_{package['sourceSha256'][:8]}"
    manifest_path = root / "manifests" / f"{dataset_id}.json"
    if manifest_path.exists():
        raise FileExistsError("同名金标准数据集已存在，拒绝覆盖")

    source_name = package["sourceFilename"]
    source_bytes = base64.b64decode(package["sourceBase64"])
    source_path = root / "imports" / dataset_id / source_name
    source_path.parent.mkdir(parents=True, exist_ok=False)
    source_path.write_bytes(source_bytes)

    normalized_json_path = root / "normalized" / f"{dataset_id}.json"
    normalized_csv_path = root / "normalized" / f"{dataset_id}.csv"
    _write_json(normalized_json_path, valid_rows)
    normalized_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with normalized_csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*CANONICAL_FIELDS, "rowNumber"])
        writer.writeheader()
        writer.writerows(valid_rows)

    manifest = {
        "datasetId": dataset_id,
        "createdAt": _now_iso(),
        "sourceFilename": source_name,
        "sourceSha256": package["sourceSha256"],
        "sourcePath": str(source_path),
        "normalizedJsonPath": str(normalized_json_path),
        "normalizedCsvPath": str(normalized_csv_path),
        "totalRows": package.get("totalRows", len(valid_rows)),
        "validRowCount": len(valid_rows),
        "excludedRowCount": len(package.get("invalidRows") or []),
        "duplicateCount": package.get("duplicateCount", 0),
        "thresholdVersion": threshold_version(),
        "schemaVersion": 1,
        "immutable": True,
    }
    _write_json(manifest_path, manifest)
    preview_path.unlink(missing_ok=True)
    return {"success": True, "dataset": manifest}


def list_datasets(*, store_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = Path(store_root or DEFAULT_STORE_ROOT)
    manifests = root / "manifests"
    if not manifests.is_dir():
        return []
    rows = []
    for path in manifests.glob("GS_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                rows.append(payload)
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(rows, key=lambda item: str(item.get("createdAt") or ""), reverse=True)


def _record_id(record: Mapping[str, Any]) -> str:
    for key in ("id", "attemptId", "attempt_id", "sessionId", "session_id"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return ""


def _validated_dataset_id(dataset_id: str) -> str:
    value = str(dataset_id or "").strip()
    if not re.fullmatch(r"GS_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}", value):
        raise LookupError("金标准数据集编号无效")
    return value


def _extract_system_measurement(record: Mapping[str, Any], metric: str) -> tuple[float | None, float | None, str]:
    detail = record.get("scoreDetail") or record.get("score_detail") or {}
    detail = detail if isinstance(detail, Mapping) else {}
    indicators = detail.get("indicators") if isinstance(detail.get("indicators"), Mapping) else {}
    indicator = indicators.get(metric) if isinstance(indicators, Mapping) else None
    indicator = indicator if isinstance(indicator, Mapping) else {}
    value = _finite(indicator.get("value", indicator.get("scoring_value")))
    frame = _finite(
        indicator.get("extreme_frame_index", detail.get("t_impact", record.get("t_impact", record.get("tImpact"))))
    )
    level = str(indicator.get("status") or "").upper()
    return value, frame, level


def _inter_rater_summary(groups: Mapping[tuple[str, str, str], list[dict[str, Any]]]) -> dict[str, Any]:
    differences: list[float] = []
    multi_annotator_samples = 0
    for rows in groups.values():
        by_annotator = {str(row["annotatorId"]): float(row["referenceValue"]) for row in rows}
        values = list(by_annotator.values())
        if len(values) < 2:
            continue
        multi_annotator_samples += 1
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                differences.append(abs(values[left] - values[right]))
    return {
        "multiAnnotatorSampleCount": multi_annotator_samples,
        "pairwiseComparisonCount": len(differences),
        "meanAbsoluteDifference": round(statistics.fmean(differences), 4) if differences else None,
        "maxAbsoluteDifference": round(max(differences), 4) if differences else None,
        "adjudication": "mean_of_unique_annotators",
    }


def generate_validation_report(
    dataset_id: str,
    system_records: Sequence[Mapping[str, Any]],
    *,
    store_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(store_root or DEFAULT_STORE_ROOT)
    dataset_id = _validated_dataset_id(dataset_id)
    manifest_path = root / "manifests" / f"{dataset_id}.json"
    normalized_path = root / "normalized" / f"{dataset_id}.json"
    if not manifest_path.is_file() or not normalized_path.is_file():
        raise LookupError("金标准数据集不存在")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    annotations = json.loads(normalized_path.read_text(encoding="utf-8"))
    records_by_id = {_record_id(record): record for record in system_records if _record_id(record)}

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        grouped[(str(row["sampleId"]), str(row["attemptId"]), str(row["metric"]))].append(row)

    pairs: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    hash_verified_count = 0
    hash_unavailable_count = 0
    hash_mismatch_count = 0
    for (_sample_id, attempt_id, metric), rows in grouped.items():
        reference_values = [float(row["referenceValue"]) for row in rows]
        reference_frames = [float(row["referenceFrame"]) for row in rows]
        direct_values = [float(row["systemValue"]) for row in rows if row.get("systemValue") is not None]
        direct_frames = [float(row["systemFrame"]) for row in rows if row.get("systemFrame") is not None]
        record = records_by_id.get(attempt_id)
        system_value: float | None = statistics.fmean(direct_values) if direct_values else None
        system_frame: float | None = statistics.fmean(direct_frames) if direct_frames else None
        system_level = ""
        if record is not None:
            archive_hash = str(record.get("videoHash") or record.get("video_hash") or "").lower()
            gold_hash = str(rows[0].get("videoHash") or "").lower()
            if archive_hash and archive_hash != gold_hash:
                exclusions.append({"attemptId": attempt_id, "metric": metric, "reason": "VIDEO_HASH_MISMATCH"})
                hash_mismatch_count += 1
                continue
            if archive_hash and archive_hash == gold_hash:
                hash_verified_count += 1
            else:
                hash_unavailable_count += 1
            measured, measured_frame, measured_level = _extract_system_measurement(record, metric)
            system_value = system_value if system_value is not None else measured
            system_frame = system_frame if system_frame is not None else measured_frame
            system_level = measured_level
        if system_value is None:
            exclusions.append({"attemptId": attempt_id, "metric": metric, "reason": "SYSTEM_VALUE_NOT_FOUND"})
            continue
        if record is None:
            hash_unavailable_count += 1

        reference_value = statistics.fmean(reference_values)
        reference_frame = statistics.fmean(reference_frames)
        if metric == "impact_knee_angle":
            system_level = system_level or classify_impact_knee_angle(system_value)
            reference_level = classify_impact_knee_angle(reference_value)
        else:
            reference_level = ""
        pairs.append(
            {
                "metric": metric,
                "systemValue": system_value,
                "referenceValue": reference_value,
                "systemLevel": system_level,
                "referenceLevel": reference_level,
                "systemFrame": system_frame,
                "referenceFrame": reference_frame,
                "attemptId": attempt_id,
                "annotatorCount": len({row["annotatorId"] for row in rows}),
            }
        )

    statistics_payload = validate_measurement_pairs(pairs)
    report = {
        **statistics_payload,
        "datasetId": dataset_id,
        "generatedAt": _now_iso(),
        "thresholdVersion": threshold_version(),
        "sourceSha256": manifest.get("sourceSha256"),
        "annotationRowCount": len(annotations),
        "alignedSampleCount": len(pairs),
        "excludedSampleCount": len(exclusions),
        "exclusions": exclusions,
        "interRater": _inter_rater_summary(grouped),
        "hashVerification": {
            "verifiedCount": hash_verified_count,
            "unavailableCount": hash_unavailable_count,
            "mismatchCount": hash_mismatch_count,
            "policy": "mismatch_excluded; unavailable_retained_and_flagged",
        },
        "pairs": pairs,
    }
    reports_dir = root / "validation_reports"
    report_name = f"{dataset_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    _write_json(reports_dir / report_name, report)
    return report


def latest_validation_report(dataset_id: str, *, store_root: str | Path | None = None) -> dict[str, Any] | None:
    root = Path(store_root or DEFAULT_STORE_ROOT)
    dataset_id = _validated_dataset_id(dataset_id)
    reports = sorted((root / "validation_reports").glob(f"{dataset_id}_*.json"), reverse=True)
    if not reports:
        return None
    return json.loads(reports[0].read_text(encoding="utf-8"))
