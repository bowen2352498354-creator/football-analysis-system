from __future__ import annotations

import llm_agent


def test_fallback_individual_summary_has_full_report_sections():
    result = llm_agent._build_fallback_individual_summary(
        [72.0, 80.0],
        {"支撑脚位置偏离": 2},
        {
            "scoreSummary": {"first": 72.0, "latest": 80.0, "change": 8.0},
            "errorRates": {"支撑脚位置偏离": 1.0},
        },
    )

    assert set(result) == {
        "overallAssessment",
        "progressAnalysis",
        "strengths",
        "weaknesses",
        "prescription",
        "dosage",
    }
    assert "72.0" in result["progressAnalysis"]
    assert "支撑脚" in result["prescription"]
