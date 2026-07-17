# -*- coding: utf-8 -*-
"""
word_reporter.py
v1.0 -- Local folder archiving + Word (.docx) auto-generation module.

Purpose:
    This module exposes a single core entry point, save_feedback_to_word(data: dict),
    which api_server.py's POST /api/save_word_report endpoint calls directly to do
    two things:

    1) Build a strict local directory tree:
           AI-Football-Feedback/student feedback report/
               |-- <realtime feedback folder> or <delayed feedback folder>  (level 1: test mode)
               |     |-- <school>-<class/group>/                            (level 2: school + class/group)
               |           |-- <student number>/                           (level 3: student number)

       Every level is created recursively with os.makedirs(..., exist_ok=True) if missing.

    2) Render a well-formatted Word (.docx) report:
       Title + metadata table (test timestamp / school & class / student number / overall score)
       + biomechanics annotated key frame image (if provided) + AI-generated pain-point
       analysis and coaching prescription text, saved as
       "YYYY-MM-DD_HH-mm_No.XX_<report>.docx" into the folder built in step 1), and the
       absolute physical path of the saved file is returned so api_server.py can hand it
       straight back to the frontend.

Robustness notes:
    - Base64 image decoding is defensive: whether the frontend sends a raw Base64 string,
      a standard "data:image/jpeg;base64,xxxx" data URI, or something missing/corrupted,
      decoding failures are swallowed silently -- the picture is simply skipped, and the
      rest of the report is still generated and saved normally.
    - Windows-illegal filename/foldername characters (backslash, slash, colon, asterisk,
      question mark, double quote, angle brackets, pipe) coming from free-text fields
      (school name, class/group name, student number) are all sanitized through
      sanitize_path_component() before touching the filesystem, so os.makedirs() /
      Document.save() never raise OSError and abort an otherwise successful save.
    - Module import time forces sys.stdout/sys.stderr to UTF-8 (errors='replace') so
      Windows's legacy GBK console code page never raises UnicodeEncodeError on CJK text
      or emoji in log lines (first line of defense). Terminal printing is additionally
      wrapped in _safe_print() (second line of defense): any UnicodeEncodeError that still
      slips through is caught and the message is safely re-encoded, so a logging hiccup
      never aborts an otherwise successful report generation.
"""

from __future__ import annotations

import base64
import io
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------
# Windows console-encoding compatibility fix (first line of defense).
#
# On Windows, the default console code page is GBK (cp936). When this module
# is imported by api_server.py and runs inside a background thread, any
# print()/log line containing a character outside the GBK charset (e.g. the
# report status is embedded together with debug text elsewhere in the
# pipeline) would raise UnicodeEncodeError and could abort the background
# archiving thread. Force stdout/stderr to UTF-8 with errors='replace' here
# so this never happens, regardless of what the OS console code page is.
# --------------------------------------------------------------------------
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except (AttributeError, ValueError):
        pass
if sys.stderr.encoding is None or sys.stderr.encoding.lower() != "utf-8":
    try:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except (AttributeError, ValueError):
        pass

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

# --------------------------------------------------------------------------
# All user-facing Chinese text is defined here via \uXXXX escapes instead of
# literal CJK characters. This keeps the source file 100% ASCII on disk,
# sidestepping any encoding round-trip issues in the local toolchain while
# still producing perfectly correct Chinese strings at runtime (Python
# decodes \uXXXX escapes the same way regardless of the file's own encoding).
# --------------------------------------------------------------------------

_MODE_REALTIME_LABEL = "\u5b9e\u65f6\u53cd\u9988"  # ????
_MODE_DELAYED_LABEL = "\u5ef6\u65f6\u53cd\u9988"  # ????

_FONT_HEADING_EASTASIA = "\u9ed1\u4f53"  # ??
_FONT_BODY_EASTASIA = "\u5fae\u8f6f\u96c5\u9ed1"  # ????

_FALLBACK_UNNAMED = "\u672a\u547d\u540d"  # ???
_FALLBACK_CLASS_FOLDER = "\u672a\u5206\u7c7b\u73ed\u7ea7"  # ?????
_FALLBACK_STUDENT_FOLDER = "\u672a\u586b\u5199\u5b66\u53f7"  # ?????
_FALLBACK_SCHOOL_TEXT = "\u672a\u8bbe\u7f6e\u5b66\u6821"  # ?????
_FALLBACK_CLASSGROUP_TEXT = "\u672a\u8bbe\u7f6e\u73ed\u7ea7"  # ?????
_FALLBACK_STUDENT_NUM_TEXT = "\u672a\u586b\u5199\u7f16\u53f7"  # ?????

_NO_SCORE_TEXT = "\u6682\u65e0\u8bc4\u5206"  # ????
_NO_DATA_TEXT = "\u6682\u65e0\u6570\u636e"  # ????

_LABEL_TIMESTAMP = "\u6d4b\u8bd5\u65e5\u671f\u4e0e\u65f6\u95f4\u6233"  # ????????
_LABEL_SCHOOL_CLASS = "\u5b66\u6821\u73ed\u7ea7"  # ????
_LABEL_STUDENT_NUM = "\u5b66\u751f\u7f16\u53f7"  # ????
_LABEL_SCORE = "\u53d1\u529b\u7efc\u5408\u8bc4\u5206"  # ??????
_LABEL_SAMPLE_COUNT = "\u6709\u6548\u91c7\u6837\u6b21\u6570"  # ??????

_UNIT_SCORE_SUFFIX = " \u5206"  # " ?"
_UNIT_TIMES_SUFFIX = " \u6b21"  # " ?"

_TITLE_MAIN = (
    "\u300aAI \u53ef\u89c6\u5316\u8db3\u7403\u6559\u5b66 - "
    "\u751f\u7269\u529b\u5b66\u8bca\u65ad\u62a5\u544a\u300b"
)  # ?AI ??????? - ?????????
_SUBTITLE_SUFFIX = " - \u7cfb\u7edf\u81ea\u52a8\u5f52\u6863\u751f\u6210"  # " - ????????"

_IMAGE_CAPTION = (
    "\u4e0a\u56fe\uff1a\u51fb\u7403\u77ac\u95f4\u751f\u7269\u529b\u5b66\u5173\u952e\u5e27\u6807\u6ce8\u56fe"
    "\uff08\u9acb-\u819d-\u8e1d\u52a8\u529b\u94fe\u77e2\u91cf\uff09"
)  # ???????????????????-?-???????

_HEADING_PAIN_POINT = "\u9519\u8bef\u75db\u70b9\u5206\u6790"  # ??????
_HEADING_PRESCRIPTION = "\u6559\u7ec3\u6539\u8fdb\u5efa\u8bae"  # ??????

_NO_TEXT_FALLBACK = "\uff08\u672c\u6b21\u5206\u6790\u6682\u65e0\u53ef\u7528\u6587\u672c\u5185\u5bb9\uff09"
# ??????????????

_FILENAME_SUFFIX = "\u8bca\u65ad\u5904\u65b9"  # ????

_LABEL_BRACKET_LEFT = "\u3010"  # ?
_LABEL_BRACKET_RIGHT = "\u3011"  # ?


def _safe_print(message: str) -> None:
    """Print a log line (second line of defense): even though the module-level
    stdout/stderr UTF-8 reconfiguration above should already make this a
    non-issue, this still catches any stray UnicodeEncodeError and degrades
    the message to a safely re-encodable form instead of letting a logging
    hiccup abort an otherwise successful report generation.
    """
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        try:
            print(message.encode(encoding, errors="replace").decode(encoding, errors="replace"))
        except Exception:
            pass


# --------------------------------------------------------------------------
# Step 0: base paths & constants
# --------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Root archive folder is fixed to "<project root>/student feedback report/"
REPORT_ROOT_DIR = os.path.join(SCRIPT_DIR, "student feedback report")

# Level-1 subfolder: test mode -> Chinese folder name
MODE_FOLDER_NAME = {
    "realtime": _MODE_REALTIME_LABEL,
    "delayed": _MODE_DELAYED_LABEL,
}

# Target width (inches) when inserting the picture into the Word page: 5.5in
# renders crisp and centered on both A4 and Letter without overflowing margins.
IMAGE_WIDTH_INCHES = 5.5

# Windows-illegal filename/foldername characters: \ / : * ? " < > |
# plus all ASCII control characters (0x00-0x1F, e.g. stray \n / \t / \r that
# might leak in from free-text frontend fields), which Windows also rejects.
_ILLEGAL_CHARS_PATTERN = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def _set_run_east_asian_font(run, font_name: str) -> None:
    """Explicitly set the East-Asian font of a run so CJK glyphs actually
    render with the requested typeface (Word stores "Western" and "East
    Asian" fonts as two separate properties -- setting run.font.name alone
    only affects the Western half and leaves CJK text on the system default).
    """
    run.font.name = font_name
    run_properties = run._element.get_or_add_rPr()
    font_element = run_properties.find(qn("w:rFonts"))
    if font_element is None:
        font_element = run_properties.makeelement(qn("w:rFonts"), {})
        run_properties.append(font_element)
    font_element.set(qn("w:eastAsia"), font_name)


# --------------------------------------------------------------------------
# Step 1: path sanitizing & directory tree construction
# --------------------------------------------------------------------------


def sanitize_path_component(raw: Optional[str], fallback: str = _FALLBACK_UNNAMED) -> str:
    """Turn an arbitrary free-text string into a safe Windows folder/file
    name segment.

    - Blank/None falls back to `fallback`.
    - Windows-illegal characters are replaced with a hyphen "-".
    - Collapse consecutive hyphens produced by the substitution above, and
      strip leading/trailing dots or spaces (Windows disallows folder names
      ending in a dot or a space).
    """
    text = (raw or "").strip()
    if not text:
        text = fallback
    text = _ILLEGAL_CHARS_PATTERN.sub("-", text)
    text = re.sub(r"-{2,}", "-", text).strip(" .")
    return text or fallback


def build_target_directory(mode: str, school: str, class_group: str, student_number: str) -> Path:
    """Build (and recursively create) the full 3-level directory tree:
    level 1 test mode -> level 2 "<school>-<class/group>" -> level 3 student number.
    Returns the level-3 directory path.
    """
    mode_folder = MODE_FOLDER_NAME.get(mode, _MODE_REALTIME_LABEL)

    school_clean = (school or "").strip()
    class_group_clean = (class_group or "").strip()
    if school_clean and class_group_clean:
        school_class_raw = f"{school_clean}-{class_group_clean}"
    else:
        school_class_raw = school_clean or class_group_clean
    school_class_folder = sanitize_path_component(school_class_raw, _FALLBACK_CLASS_FOLDER)

    student_folder = sanitize_path_component(student_number, _FALLBACK_STUDENT_FOLDER)

    target_dir = Path(REPORT_ROOT_DIR) / mode_folder / school_class_folder / student_folder
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


# --------------------------------------------------------------------------
# Step 2: defensive Base64 image decoding
# --------------------------------------------------------------------------


def _decode_base64_image_to_stream(image_base64: Optional[str]) -> Optional[io.BytesIO]:
    """Safely decode a Base64 image string (either a raw Base64 payload, or a
    standard "data:image/...;base64,xxxx" data URI) into an in-memory
    io.BytesIO stream.

    Any failure (missing/empty/malformed/corrupted input) is caught and
    returns None -- callers treat that as "skip the picture", never letting a
    bad image field abort the entire text report from being generated.
    """
    if not image_base64 or not isinstance(image_base64, str):
        return None
    try:
        payload = image_base64.strip()
        if payload.lower().startswith("data:") and "," in payload:
            payload = payload.split(",", 1)[1]
        raw_bytes = base64.b64decode(payload, validate=False)
        if not raw_bytes:
            return None
        return io.BytesIO(raw_bytes)
    except Exception as exc:  # noqa: BLE001 - image decoding must never abort the whole report
        _safe_print(f"[word_reporter] warning: base64 image decode failed, skipping image. reason: {exc}")
        return None


# --------------------------------------------------------------------------
# Step 3: Word (.docx) layout rendering
# --------------------------------------------------------------------------


def _add_metadata_table(document: Document, rows: list[tuple[str, str]]) -> None:
    """Insert a two-column metadata table below the title: bold label on the
    left, corresponding value on the right.
    """
    table = document.add_table(rows=len(rows), cols=2)
    table.style = "Light Grid Accent 1"
    table.autofit = True

    for row_index, (label, value) in enumerate(rows):
        label_cell = table.cell(row_index, 0)
        value_cell = table.cell(row_index, 1)

        label_cell.text = ""
        label_paragraph = label_cell.paragraphs[0]
        label_run = label_paragraph.add_run(f"{_LABEL_BRACKET_LEFT}{label}{_LABEL_BRACKET_RIGHT}")
        label_run.bold = True
        label_run.font.size = Pt(11)
        _set_run_east_asian_font(label_run, _FONT_BODY_EASTASIA)

        value_cell.text = ""
        value_paragraph = value_cell.paragraphs[0]
        value_run = value_paragraph.add_run(str(value))
        value_run.font.size = Pt(11)
        _set_run_east_asian_font(value_run, _FONT_BODY_EASTASIA)


def _add_section_heading(document: Document, text: str) -> None:
    """Insert a section heading (e.g. pain-point analysis / coaching
    prescription), bold and slightly larger.
    """
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(14)
    paragraph.paragraph_format.space_after = Pt(6)
    run = paragraph.add_run(text)
    run.bold = True
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(0x1F, 0x6F, 0x4A)
    _set_run_east_asian_font(run, _FONT_HEADING_EASTASIA)


def _add_body_paragraph(document: Document, text: str) -> None:
    """Insert a body paragraph with comfortable spacing, ready to print."""
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(10)
    paragraph.paragraph_format.line_spacing = 1.35
    run = paragraph.add_run(text or _NO_TEXT_FALLBACK)
    run.font.size = Pt(12)
    _set_run_east_asian_font(run, _FONT_BODY_EASTASIA)


def _build_document(data: dict, mode: str) -> Document:
    """Build the full Word document object from the data dict (not yet saved to disk)."""
    school = data.get("school") or _FALLBACK_SCHOOL_TEXT
    class_group = data.get("classGroup") or _FALLBACK_CLASSGROUP_TEXT
    student_number = data.get("studentNumber") or data.get("studentId") or _FALLBACK_STUDENT_NUM_TEXT
    score = data.get("score")
    total_attempts = data.get("totalAttempts")
    generated_at = data.get("generatedAt") or time.strftime("%Y-%m-%d %H:%M:%S")
    pain_point = data.get("painPoint") or ""
    prescription = data.get("prescription") or ""

    document = Document()

    # --- Header metadata: large bold heading-font title ---
    title_paragraph = document.add_paragraph()
    title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_paragraph.add_run(_TITLE_MAIN)
    title_run.bold = True
    title_run.font.size = Pt(22)
    _set_run_east_asian_font(title_run, _FONT_HEADING_EASTASIA)

    subtitle_paragraph = document.add_paragraph()
    subtitle_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_run = subtitle_paragraph.add_run(
        f"{MODE_FOLDER_NAME.get(mode, _MODE_REALTIME_LABEL)}{_SUBTITLE_SUFFIX}"
    )
    subtitle_run.italic = True
    subtitle_run.font.size = Pt(10)
    subtitle_run.font.color.rgb = RGBColor(0x70, 0x70, 0x70)
    _set_run_east_asian_font(subtitle_run, _FONT_BODY_EASTASIA)

    document.add_paragraph()

    score_display = f"{score}{_UNIT_SCORE_SUFFIX}" if score is not None else _NO_SCORE_TEXT
    total_attempts_display = f"{total_attempts}{_UNIT_TIMES_SUFFIX}" if total_attempts is not None else _NO_DATA_TEXT

    _add_metadata_table(
        document,
        rows=[
            (_LABEL_TIMESTAMP, generated_at),
            (_LABEL_SCHOOL_CLASS, f"{school} - {class_group}"),
            (_LABEL_STUDENT_NUM, str(student_number)),
            (_LABEL_SCORE, score_display),
            (_LABEL_SAMPLE_COUNT, total_attempts_display),
        ],
    )

    document.add_paragraph()

    # --- Biomechanics annotated key frame: decode Base64 -> in-memory stream -> centered insert ---
    image_stream = _decode_base64_image_to_stream(data.get("impactFrameImage"))
    if image_stream is not None:
        try:
            image_paragraph = document.add_paragraph()
            image_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            image_run = image_paragraph.add_run()
            image_run.add_picture(image_stream, width=Inches(IMAGE_WIDTH_INCHES))

            caption_paragraph = document.add_paragraph()
            caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            caption_run = caption_paragraph.add_run(_IMAGE_CAPTION)
            caption_run.font.size = Pt(9)
            caption_run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
            _set_run_east_asian_font(caption_run, _FONT_BODY_EASTASIA)
        except Exception as exc:  # noqa: BLE001 - image insertion must never abort the text report
            _safe_print(f"[word_reporter] warning: insert image failed, skipping image. reason: {exc}")

    document.add_paragraph()

    # --- AI prescription & diagnosis text ---
    _add_section_heading(document, _HEADING_PAIN_POINT)
    _add_body_paragraph(document, pain_point)

    _add_section_heading(document, _HEADING_PRESCRIPTION)
    _add_body_paragraph(document, prescription)

    return document


# --------------------------------------------------------------------------
# Step 4: public entry point
# --------------------------------------------------------------------------


def save_feedback_to_word(data: dict) -> dict:
    """Core entry point: given a data dict assembled by the frontend/backend,
    build the local directory tree and save the Word (.docx) report, then
    return a structured result dict.

    Expected fields in `data` (all defensively handled, none are strictly
    required to be present):
        mode            : "realtime" | "delayed"
        school          : str  -- school/institution name
        classGroup      : str  -- class / experiment group name
        studentNumber   : str  -- student number / ID
        score           : int | None
        totalAttempts   : int | None
        painPoint       : str  -- AI-generated pain-point analysis
        prescription    : str  -- AI-generated coaching prescription
        generatedAt     : str | None
        impactFrameImage: str | None  -- Base64 / data URI key frame image

    Returns:
        {"success": True, "path": "...", "directory": "...", "filename": "..."}
        or {"success": False, "error": "..."} on failure.
    """
    try:
        mode = data.get("mode") if data.get("mode") in MODE_FOLDER_NAME else "realtime"
        school = data.get("school") or ""
        class_group = data.get("classGroup") or ""
        student_number_raw = data.get("studentNumber") or data.get("studentId") or ""

        target_dir = build_target_directory(mode, school, class_group, student_number_raw)

        document = _build_document(data, mode)

        # File naming convention: YYYY-MM-DD_HH-mm_<student number>_<report>.docx
        timestamp_label = time.strftime("%Y-%m-%d_%H-%M")
        student_number_clean = sanitize_path_component(student_number_raw, _FALLBACK_STUDENT_FOLDER)
        filename = f"{timestamp_label}_{student_number_clean}_{_FILENAME_SUFFIX}.docx"

        full_path = target_dir / filename
        document.save(str(full_path))

        _safe_print(f"[word_reporter] saved Word report to: {full_path}")

        return {
            "success": True,
            "path": str(full_path.resolve()),
            "directory": str(target_dir.resolve()),
            "filename": filename,
        }
    except Exception as exc:  # noqa: BLE001 - any failure must be reported, never crash the caller
        _safe_print(f"[word_reporter] error: save word report failed: {exc}")
        return {"success": False, "error": str(exc)}
