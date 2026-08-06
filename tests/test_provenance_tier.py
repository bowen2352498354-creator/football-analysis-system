# -*- coding: utf-8 -*-
"""全链路 provenance_tier（MEASURED/CALIBRATED/ESTIMATED）契约。"""

from __future__ import annotations

from deterministic_scorer import calculate_biomechanical_score
from indicator_builder import (
    PROVENANCE_TIER_CALIBRATED,
    PROVENANCE_TIER_ESTIMATED,
    PROVENANCE_TIER_MEASURED,
    apply_provenance_tiers,
    resolve_provenance_tier,
)


def test_resolve_provenance_tier_defaults():
    assert resolve_provenance_tier("whipping_velocity", {}) == PROVENANCE_TIER_MEASURED
    assert resolve_provenance_tier("trunk_lean_angle", {}) == PROVENANCE_TIER_MEASURED
    assert resolve_provenance_tier("impact_knee_angle", {}) == PROVENANCE_TIER_MEASURED
    assert resolve_provenance_tier("distance_cm", {}) == PROVENANCE_TIER_CALIBRATED
    assert resolve_provenance_tier("support_ratio", {}) == PROVENANCE_TIER_CALIBRATED
    assert resolve_provenance_tier("hip_torsion_angle", {}) == PROVENANCE_TIER_ESTIMATED


def test_method_hints_override_key_default():
    assert (
        resolve_provenance_tier(
            "impact_knee_angle", {"method": "raw_3d_world"}
        )
        == PROVENANCE_TIER_ESTIMATED
    )
    assert (
        resolve_provenance_tier("toe_angle", {"method": "shoulder_width_ratio"})
        == PROVENANCE_TIER_CALIBRATED
    )


def test_scorer_stamps_provenance_tier_on_all_indicators():
    impact = {
        "t_impact": 0,
        "toe_angle": 5.0,
        "impact_knee_angle": 150.0,
        "support_knee_angle": 155.0,
        "hip_torsion_angle": 25.0,
        "support_ratio": 0.55,
        "ankle_angles_window": [140.0, 140.1, 140.0],
        "trunk_lean_angle": 8.0,
    }
    trajectory = {"max_folding_angle": 70.0, "whipping_velocity": 500.0}
    _score, detail = calculate_biomechanical_score(impact, trajectory)
    indicators = detail["indicators"]
    assert indicators["whipping_velocity"]["provenance_tier"] == PROVENANCE_TIER_MEASURED
    assert indicators["impact_knee_angle"]["provenance_tier"] == PROVENANCE_TIER_MEASURED
    assert indicators["trunk_lean_angle"]["provenance_tier"] == PROVENANCE_TIER_MEASURED
    assert indicators["distance_cm"]["provenance_tier"] == PROVENANCE_TIER_CALIBRATED
    assert indicators["hip_torsion_angle"]["provenance_tier"] == PROVENANCE_TIER_ESTIMATED
    assert indicators["hip_torsion_angle"]["provenance"] == "estimated"
    # 每个指标都必须有血统标签
    for key, entry in indicators.items():
        assert isinstance(entry, dict)
        assert entry.get("provenance_tier") in {
            PROVENANCE_TIER_MEASURED,
            PROVENANCE_TIER_CALIBRATED,
            PROVENANCE_TIER_ESTIMATED,
        }, f"{key} missing provenance_tier"


def test_apply_provenance_tiers_is_idempotent():
    indicators = {
        "whipping_velocity": {"value": 400.0, "provenance": "measured"},
        "hip_torsion_angle": {"value": 20.0},
    }
    once = apply_provenance_tiers(indicators)
    twice = apply_provenance_tiers(once)
    assert once["whipping_velocity"]["provenance_tier"] == PROVENANCE_TIER_MEASURED
    assert twice["hip_torsion_angle"]["provenance_tier"] == PROVENANCE_TIER_ESTIMATED
