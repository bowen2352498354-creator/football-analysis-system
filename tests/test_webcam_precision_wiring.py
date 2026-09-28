"""Dependency-light wiring checks for the web dual-channel analysis path."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_web_pipeline_buffers_only_after_anonymization():
    source = (ROOT / "shot_analysis_service.py").read_text(encoding="utf-8")
    anonymize = source.index("frame = pt.apply_facial_anonymization(frame, landmarks)")
    buffer_push = source.index("self._auto_capture.push_frame(frame, frame_index)", anonymize)
    skeleton = source.index("pt.draw_pose_landmarks(frame, results.pose_landmarks)", buffer_push)
    assert anonymize < buffer_push < skeleton


def test_report_prepares_precision_pipeline_before_reading_records():
    source = (ROOT / "api_server.py").read_text(encoding="utf-8")
    prepare = source.index("precision_analysis_used = session.prepare_precision_analysis()")
    records = source.index("records = session.get_records_snapshot()", prepare)
    scoring = source.index("session.build_scoring_payloads()", records)
    assert prepare < records < scoring


def test_webcam_session_end_forces_replay_when_motion_signal_is_missing():
    source = (ROOT / "shot_analysis_service.py").read_text(encoding="utf-8")
    finalize = source.index("def _finalize_auto_capture")
    fallback = source.index("engine.save_buffered_window(peak_index)", finalize)
    writer_wait = source.index("engine.wait_for_pending_save", fallback)
    assert finalize < fallback < writer_wait


def test_webcam_capture_requests_browser_compatible_webm():
    source = (ROOT / "shot_analysis_service.py").read_text(encoding="utf-8")
    webcam_engine = source.index("self._auto_capture = AutoShotCaptureEngine(")
    container = source.index('video_container="webm"', webcam_engine)
    assert webcam_engine < container


def test_webcam_report_exposes_replay_to_native_video():
    backend = (ROOT / "api_server.py").read_text(encoding="utf-8")
    frontend = (
        ROOT / "AI-Football-Web" / "src" / "components" / "RealtimeWorkspace.tsx"
    ).read_text(encoding="utf-8")
    assert '"replayVideoPath"' in backend
    assert "replayVideoUrl" in frontend
    assert "videoSourceMode === 'file' ? localVideoObjectUrl : replayVideoUrl" in frontend
    assert "startFromBeginning={Boolean(replayVideoUrl)}" in frontend


def test_attempt_replay_cannot_be_covered_by_stale_live_layer():
    source = (
        ROOT
        / "AI-Football-Web"
        / "src"
        / "components"
        / "SynchronizedVideoWorkspace.tsx"
    ).read_text(encoding="utf-8")
    assert "const replayOwnsStage = Boolean(videoSrc) && startFromBeginning" in source
    assert "!replayOwnsStage && (preferLiveOverlay || !videoSrc)" in source
    assert "key={videoSrc}" in source
