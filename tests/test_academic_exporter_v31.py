# -*- coding: utf-8 -*-
"""V3.1 AcademicDataExporter：全数字编码 / 衍生中介 / 宽表展平单测。"""

from __future__ import annotations

import math

import academic_exporter as ae


def test_digital_encoding_group_fatigue_ankle():
    assert ae.encode_experimental_group("GROUP_A") == 1
    assert ae.encode_experimental_group("GROUP_B_DELAYED") == 2
    assert ae.encode_experimental_group("GROUP_C") == 3
    assert ae.encode_fatigue_alert(True) == 1
    assert ae.encode_fatigue_alert(False) == 0
    assert ae.encode_ankle_lock_status("GREEN_OPTIMAL") == 3
    assert ae.encode_ankle_lock_status("YELLOW") == 2
    assert ae.encode_ankle_lock_status("RED") == 1


def test_heatmap_dispersion_index_standard_distance():
    points = [(0.0, 0.0), (6.0, 0.0), (0.0, 8.0)]
    # mean=(2, 8/3); SD = sqrt(mean of squared distances)
    idx = ae.heatmap_dispersion_index(points)
    assert idx is not None
    assert idx > 0
    assert math.isclose(ae.heatmap_dispersion_index([(1.0, 2.0)]), 0.0)
    assert ae.heatmap_dispersion_index([]) is None


def test_ankle_rigidity_score_normalization():
    assert ae.normalize_ankle_rigidity_score(0.0) == 1.0
    assert ae.normalize_ankle_rigidity_score(10.0) == 0.0
    assert ae.normalize_ankle_rigidity_score(5.0) == 0.5


def test_wide_format_fully_numeric_and_dummies():
    exporter = ae.AcademicDataExporter(
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
    wide = exporter.generate_wide_format_matrix()

    assert list(wide["anonymous_id"]) == ["Sub_001", "Sub_002"]
    assert wide["experimental_group"].tolist() == [1, 3]
    assert "T1_Ankle_Rigidity" in wide.columns
    assert "T1_Heatmap_Dispersion_Index" in wide.columns
    assert "T1_Ankle_Rigidity_Score" in wide.columns
    assert "Class_Dummy_1" in wide.columns
    assert "Class_Dummy_5" in wide.columns

    row1 = wide.loc[wide["anonymous_id"] == "Sub_001"].iloc[0]
    assert row1["experimental_group"] == 1
    assert row1["T1_Fatigue_Alert"] == 1  # any True in window
    assert row1["Class_Dummy_1"] == 1
    assert row1["Class_Dummy_5"] == 0
    assert math.isclose(float(row1["T1_Ankle_Rigidity"]), 2.75, rel_tol=1e-6)
    assert row1["T1_Heatmap_Dispersion_Index"] is not None
    assert 0.0 <= float(row1["T1_Ankle_Rigidity_Score"]) <= 1.0

    row2 = wide.loc[wide["anonymous_id"] == "Sub_002"].iloc[0]
    assert row2["T0_Ankle_Lock_Status"] == 1  # RED
    assert row2["Class_Dummy_1"] == 0  # Class_6 reference → all zeros

    # 状态列不得出现字符串
    for col in ("experimental_group", "T1_Fatigue_Alert", "T0_Ankle_Lock_Status", "Class_Dummy_1"):
        assert wide[col].dtype.kind in ("i", "u", "f", "b")

    csv_bytes = exporter.to_csv_bytes(wide)
    assert csv_bytes.startswith(b"\xef\xbb\xbf") or b"anonymous_id" in csv_bytes
    assert b"GROUP_A" not in csv_bytes
    assert b"GREEN_OPTIMAL" not in csv_bytes
    assert b"Class_Dummy_1" in csv_bytes
