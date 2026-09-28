from __future__ import annotations

import zipfile
import tempfile
import unittest
from pathlib import Path

from docx import Document

from class_print_reporter import create_class_print_report


def _report(student_id: str) -> dict:
    return {
        "studentId": student_id,
        "requestedPeriod": {"start": "2026-08-01", "end": "2026-09-09"},
        "period": {"start": "2026-08-05", "end": "2026-09-08"},
        "formalAttemptCount": 4,
        "scoreSummary": {"mean": 82.4, "latest": 86.0, "change": 7.5},
        "fiveDimensionScores": {
            "approach_rhythm": 84,
            "support_stability": 75,
            "backswing_folding": 79,
            "ankle_rigidity": 68,
            "whipping_velocity": 88,
        },
        "topErrors": [{"label": "支撑脚位置偏离", "count": 2, "rate": 0.5}],
        "overallAssessment": "纳入4次A级有效尝试，动作质量总体稳定。",
        "progressAnalysis": "首末次成绩提升7.5分，后程表现更稳定。",
        "strengths": "鞭打速度与助跑节奏是当前主要优势。",
        "weaknesses": "支撑脚落点仍有横向漂移，需要提高重复性。",
        "prescription": "完成无球支撑脚定点，再衔接半程助跑击球。",
        "dosage": "每组5次，共3组，组间休息30秒。",
        "selfCheckTask": {
            "title": "一拳落点图式",
            "cue": "支撑脚落在球侧一拳处，脚尖朝向目标。",
            "dosage": "3组×5次",
        },
    }


class ClassPrintReporterTests(unittest.TestCase):
    def test_class_print_report_uses_a4_and_exact_two_up_pagination(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "class-two-up.docx"
            create_class_print_report(
                [_report(f"S{i:03d}") for i in range(1, 6)],
                school="示范学校",
                class_group="四年级1班",
                period_text="2026-08-01 至 2026-09-09",
                generated_at="2026-09-09 10:30:00",
                students_per_page=2,
                output_path=output,
            )

            self.assertTrue(output.is_file() and output.stat().st_size > 20_000)
            doc = Document(str(output))
            self.assertEqual(round(doc.sections[0].page_width.cm, 1), 21.0)
            self.assertEqual(round(doc.sections[0].page_height.cm, 1), 29.7)

            with zipfile.ZipFile(output) as archive:
                xml = archive.read("word/document.xml").decode("utf-8")
            self.assertIn("S001", xml)
            self.assertIn("S005", xml)
            self.assertIn("图式自查任务", xml)
            self.assertEqual(xml.count('w:type="page"'), 2)
            self.assertGreaterEqual(xml.count("w:cantSplit"), 5)

    def test_class_print_report_rejects_unsupported_density(self):
        with tempfile.TemporaryDirectory() as temp_dir, self.assertRaisesRegex(ValueError, "2 or 3"):
            create_class_print_report(
                [_report("S001")],
                school="示范学校",
                class_group="四年级1班",
                period_text="全部有效记录",
                generated_at="2026-09-09 10:30:00",
                students_per_page=4,
                output_path=Path(temp_dir) / "invalid.docx",
            )


if __name__ == "__main__":
    unittest.main()
