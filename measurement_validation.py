# -*- coding: utf-8 -*-
"""Deterministic criterion-validity statistics for annotated reference data."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _agreement_icc(system: Sequence[float], reference: Sequence[float]) -> float | None:
    """ICC(A,1), absolute agreement, two-way random effects, single measure."""
    n = len(system)
    if n < 2 or n != len(reference):
        return None
    rows = [[system[i], reference[i]] for i in range(n)]
    row_means = [statistics.fmean(row) for row in rows]
    col_means = [statistics.fmean(system), statistics.fmean(reference)]
    grand = statistics.fmean(system + reference)
    ms_rows = 2.0 * sum((value - grand) ** 2 for value in row_means) / (n - 1)
    ms_cols = n * sum((value - grand) ** 2 for value in col_means)
    residual = sum(
        (rows[i][j] - row_means[i] - col_means[j] + grand) ** 2
        for i in range(n)
        for j in range(2)
    )
    ms_error = residual / (n - 1)
    denominator = ms_rows + ms_error + 2.0 * (ms_cols - ms_error) / n
    if abs(denominator) < 1e-12:
        return None
    return (ms_rows - ms_error) / denominator


def validate_measurement_pairs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate paired system/reference values grouped by metric name.

    Expected row keys: metric, systemValue, referenceValue. Optional keys:
    systemLevel/referenceLevel and systemFrame/referenceFrame.
    """
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        metric = str(row.get("metric") or "").strip()
        if metric:
            groups[metric].append(row)

    metrics: dict[str, Any] = {}
    for metric, items in groups.items():
        pairs = [
            (_finite(item.get("systemValue")), _finite(item.get("referenceValue")))
            for item in items
        ]
        valid = [(a, b) for a, b in pairs if a is not None and b is not None]
        system = [float(a) for a, _ in valid]
        reference = [float(b) for _, b in valid]
        differences = [a - b for a, b in zip(system, reference)]
        if not valid:
            continue
        bias = statistics.fmean(differences)
        diff_sd = statistics.stdev(differences) if len(differences) >= 2 else 0.0

        levels = [
            (
                str(item.get("systemLevel") or "").upper(),
                str(item.get("referenceLevel") or "").upper(),
            )
            for item in items
            if item.get("systemLevel") and item.get("referenceLevel")
        ]
        confusion = Counter(f"{actual}->{predicted}" for predicted, actual in levels)
        level_accuracy = (
            sum(predicted == actual for predicted, actual in levels) / len(levels)
            if levels
            else None
        )
        frame_errors = []
        for item in items:
            system_frame = _finite(item.get("systemFrame"))
            reference_frame = _finite(item.get("referenceFrame"))
            if system_frame is not None and reference_frame is not None:
                frame_errors.append(abs(system_frame - reference_frame))

        metrics[metric] = {
            "n": len(valid),
            "mae": round(statistics.fmean(abs(value) for value in differences), 4),
            "rmse": round(math.sqrt(statistics.fmean(value * value for value in differences)), 4),
            "bias": round(bias, 4),
            "blandAltmanLower": round(bias - 1.96 * diff_sd, 4),
            "blandAltmanUpper": round(bias + 1.96 * diff_sd, 4),
            "iccAbsoluteAgreement": (
                round(value, 4)
                if (value := _agreement_icc(system, reference)) is not None
                else None
            ),
            "levelAccuracy": round(level_accuracy, 4) if level_accuracy is not None else None,
            "levelConfusion": dict(confusion),
            "impactFrameMae": (
                round(statistics.fmean(frame_errors), 4) if frame_errors else None
            ),
            "impactWithin1FrameRate": (
                round(sum(value <= 1 for value in frame_errors) / len(frame_errors), 4)
                if frame_errors
                else None
            ),
        }

    return {
        "success": bool(metrics),
        "pairCount": sum(item["n"] for item in metrics.values()),
        "metrics": metrics,
        "claimPolicy": "Accuracy claims must use these observed statistics; no fixed ±2° guarantee is assumed.",
    }

