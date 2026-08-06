# -*- coding: utf-8 -*-
"""ClinicalBrief → AIGC 数据联动：Brief 组装、载荷注入、三字段解析。"""

from __future__ import annotations

import json

import llm_agent as la


def _ratio_distance_diagnosis() -> dict:
    return {
        "score_detail": {
            "indicators": {
                "distance_cm": {
                    "value": 0.72,
                    "scoring_value": 0.72,
                    "unit": "ratio",
                    "status": "YELLOW_APPROACHING",
                    "penalty": 4.0,
                    "provenance": "calibrated",
                    "method": "shoulder_width_ratio",
                    "support_ratio": 0.72,
                    "distance_cm_estimate": 21.6,
                    "green_band": [0.4, 0.7],
                },
                "ankle_rigidity": {
                    "value": 0.8,
                    "scoring_value": 0.8,
                    "unit": "variance",
                    "status": "GREEN_OPTIMAL",
                    "penalty": 0.0,
                    "provenance": "measured",
                    "method": "impact_window",
                    "green_band": [0.0, 2.0],
                },
                "max_folding_angle": {
                    "value": None,
                    "scoring_value": 80.0,
                    "unit": "deg",
                    "status": "GREEN_OPTIMAL",
                    "penalty": 0.0,
                    "provenance": "default",
                    "method": "neutral_band_center",
                },
            },
            "deductions": [
                {
                    "metric": "distance_cm",
                    "penalty": 4.0,
                    "reason": "支撑脚横距偏宽（约0.72×肩宽）",
                }
            ],
        }
    }


def test_build_clinical_brief_prefers_shoulder_ratio():
    brief = la.build_clinical_brief(_ratio_distance_diagnosis())
    primary = brief["primary"]
    assert primary["metric"] == "distance_cm"
    assert primary["quoteable"] is True
    assert primary["display_unit"] == "×肩宽"
    assert abs(float(primary["display_value"]) - 0.72) < 1e-6
    assert primary["estimate_cm"] is not None
    assert "支撑脚横距比例" in primary["coach_fact"]
    assert any(s["metric"] == "ankle_rigidity" for s in brief["strengths"])
    assert any(b["metric"] == "max_folding_angle" for b in brief["blocked"])
    assert brief["deduction_echo"]


def test_aigc_safe_payload_includes_clinical_brief():
    payload = la.build_aigc_safe_payload(_ratio_distance_diagnosis())
    assert "clinical_brief" in payload
    assert payload["clinical_brief"]["primary"]["metric"] == "distance_cm"
    dist = payload["indicators"]["distance_cm"]
    assert dist.get("support_ratio") == 0.72
    assert dist.get("unit") == "×肩宽"
    msg = la.build_aigc_user_message(_ratio_distance_diagnosis())
    assert "ClinicalBrief" in msg
    assert "overview" in msg
    assert "biomechanical_analysis" in msg
    assert "主要扣分病灶" in msg


def test_parse_optimal_accepts_clinical_echo_and_marks_llm():
    raw = json.dumps(
        {
            "clinical_echo": "支撑脚离球大约零点七倍肩膀宽，比理想站位远一点",
            "correction_metaphor": "你刚才支撑脚离球有点远，就像跨大栏一样！下次试试脚踩近一点。",
            "praise_encouragement": "脚踝绷得像小铁板，特别棒！",
        },
        ensure_ascii=False,
    )
    dual = la._parse_optimal_dual_feedback(raw, _ratio_distance_diagnosis())
    assert dual["aigc_source"] == "llm"
    assert "肩膀" in dual["clinical_echo"] or "肩" in dual["clinical_echo"]
    assert dual["correction_metaphor"]
    assert dual["praise_encouragement"]
    assert dual["clinical_brief"]["primary"]["metric"] == "distance_cm"


def test_fallback_report_exposes_source_and_echo(monkeypatch):
    monkeypatch.setattr(la, "client", None)
    report = la.generate_session_report(
        {"green": 0, "yellow": 1, "red": 0},
        "S01",
        deterministic_score=72.0,
        diagnosis_json=_ratio_distance_diagnosis(),
    )
    assert report["aigc_source"] == "fallback"
    assert report["clinical_echo"]
    assert report["clinical_brief"]["primary"]["metric"] == "distance_cm"
    assert "72" in report["clinical_echo"] or "72" in (report.get("overview") or "")
    joined = " ".join(
        str(report.get(k) or "")
        for k in (
            "clinical_echo",
            "overview",
            "biomechanical_analysis",
            "painPoint",
        )
    )
    assert "肩宽" in joined or "支撑" in joined


def test_dual_to_report_fields_aliases():
    dual = {
        "overview": "依据测试",
        "biomechanical_analysis": "支撑脚比例偏大导致远端代偿。",
        "magic_metaphor": "纠错句测试像弹簧一样！下次试试弯一弯。",
        "action_plan": "把支撑脚踩近一点。",
        "aigc_source": "llm",
    }
    fields = la._dual_to_report_fields(dual, _ratio_distance_diagnosis())
    assert fields["clinicalEcho"] == "依据测试"
    assert fields["overview"] == "依据测试"
    assert fields["biomechanical_analysis"]
    assert fields["magic_metaphor"]
    assert fields["action_plan"]
    assert fields["aigcSource"] == "llm"
    assert fields["clinicalBrief"]["primary"]["metric"] == "distance_cm"
