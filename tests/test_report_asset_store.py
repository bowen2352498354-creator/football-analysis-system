from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

import report_asset_store


def _image_data_uri(
    *,
    image_format: str = "JPEG",
    declared_mime: str | None = None,
    size: tuple[int, int] = (32, 24),
) -> str:
    stream = io.BytesIO()
    Image.new("RGB", size, color=(12, 120, 210)).save(stream, format=image_format)
    mime = declared_mime or {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }[image_format]
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def test_save_impact_frame_writes_validated_record_scoped_asset(tmp_path: Path):
    metadata = report_asset_store.save_impact_frame(
        "record-001",
        _image_data_uri(size=(40, 30)),
        root=tmp_path,
    )

    assert metadata is not None
    assert metadata["path"] == "records/record-001/impact.jpg"
    assert metadata["mimeType"] == "image/jpeg"
    assert metadata["width"] == 40
    assert metadata["height"] == 30
    assert len(metadata["sha256"]) == 64

    stored = report_asset_store.resolve_asset_path(metadata, root=tmp_path)
    assert stored is not None and stored.is_file()
    assert stored.stat().st_size == metadata["sizeBytes"]


@pytest.mark.parametrize(
    "record_id,payload",
    [
        ("../escape", _image_data_uri()),
        ("record-002", "data:image/jpeg;base64,not-base64"),
        ("record-003", _image_data_uri(image_format="PNG", declared_mime="image/jpeg")),
    ],
)
def test_save_impact_frame_rejects_unsafe_or_invalid_input(
    tmp_path: Path,
    record_id: str,
    payload: str,
):
    with pytest.raises(report_asset_store.AssetStorageError):
        report_asset_store.save_impact_frame(record_id, payload, root=tmp_path)


def test_resolve_asset_path_rejects_path_traversal(tmp_path: Path):
    with pytest.raises(report_asset_store.AssetStorageError):
        report_asset_store.resolve_asset_path({"path": "../secret.jpg"}, root=tmp_path)


def test_save_word_report_archives_file_metadata_without_inline_base64(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import api_server

    db_path = tmp_path / "global_training_db.json"
    asset_root = tmp_path / "report-assets"
    monkeypatch.setattr(api_server, "GLOBAL_DB_PATH", str(db_path))
    monkeypatch.setattr(report_asset_store, "REPORT_ASSET_ROOT", asset_root)
    monkeypatch.setattr(
        api_server.word_reporter,
        "save_feedback_to_word",
        lambda _data: {
            "success": True,
            "path": str(tmp_path / "report.docx"),
            "directory": str(tmp_path),
            "filename": "report.docx",
        },
    )

    result = api_server.save_word_report(
        api_server.SaveWordReportRequest(
            mode="realtime",
            school="学校一",
            classGroup="四年级1班",
            studentNumber="S001",
            score=88.5,
            totalAttempts=1,
            impactFrameImage=_image_data_uri(size=(48, 36)),
            qualityGate={"grade": "A", "score": 95},
        )
    )

    record = result["record"]
    assert record["impactFrameStorage"] == "file"
    assert record["impactFrameUrl"].endswith("/impact-frame")
    assert record["impactFrameAsset"]["width"] == 48
    assert "impactFrameBase64" not in record

    archived = json.loads(db_path.read_text(encoding="utf-8"))
    assert archived[0]["impactFrameAsset"] == record["impactFrameAsset"]
    assert "impactFrameBase64" not in archived[0]
    stored = report_asset_store.resolve_asset_path(
        record["impactFrameAsset"],
        root=asset_root,
    )
    assert stored is not None and stored.is_file()


def test_impact_frame_endpoint_supports_file_and_legacy_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import api_server
    import db as db_mod
    from fastapi.testclient import TestClient

    asset_root = tmp_path / "report-assets"
    file_payload = _image_data_uri(image_format="PNG")
    file_meta = report_asset_store.save_impact_frame(
        "file-record",
        file_payload,
        root=asset_root,
    )
    records = [
        {
            "id": "file-record",
            "impactFrameAsset": file_meta,
            "is_deleted": False,
        },
        {
            "id": "legacy-record",
            "impactFrameBase64": _image_data_uri(),
            "is_deleted": False,
        },
        {
            "id": "deleted-record",
            "impactFrameBase64": _image_data_uri(),
            "is_deleted": True,
        },
    ]
    db_path = tmp_path / "global_training_db.json"
    db_path.write_text(json.dumps(records), encoding="utf-8")

    monkeypatch.setattr(api_server, "GLOBAL_DB_PATH", str(db_path))
    monkeypatch.setattr(report_asset_store, "REPORT_ASSET_ROOT", asset_root)
    # The endpoint test must not initialize or back up the real project DB.
    monkeypatch.setattr(db_mod, "start_auto_backup_daemon", lambda **_kwargs: False)
    monkeypatch.setattr(db_mod, "init_db", lambda bind=None: None)
    db_mod.stop_auto_backup_daemon()

    with TestClient(api_server.app) as client:
        light_records = client.get("/api/get_all_records").json()["records"]
        legacy_light = next(r for r in light_records if r["id"] == "legacy-record")
        assert "impactFrameBase64" not in legacy_light
        assert legacy_light["impactFrameStorage"] == "legacy_inline"
        assert legacy_light["impactFrameUrl"].endswith("/impact-frame")

        full_records = client.get(
            "/api/get_all_records",
            params={"include_inline_assets": True},
        ).json()["records"]
        legacy_full = next(r for r in full_records if r["id"] == "legacy-record")
        assert legacy_full["impactFrameBase64"].startswith("data:image/jpeg;base64,")

        file_response = client.get("/api/coach/records/file-record/impact-frame")
        assert file_response.status_code == 200
        assert file_response.headers["content-type"].startswith("image/png")

        legacy_response = client.get("/api/coach/records/legacy-record/impact-frame")
        assert legacy_response.status_code == 200
        assert legacy_response.headers["content-type"].startswith("image/jpeg")

        deleted_response = client.get("/api/coach/records/deleted-record/impact-frame")
        assert deleted_response.status_code == 404


def test_attempt_detail_endpoint_returns_structured_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import api_server
    import db as db_mod
    from fastapi.testclient import TestClient

    record = {
        "id": "detail-record",
        "studentId": "S008",
        "timestamp": "2026-09-09 09:20:00",
        "school": "学校一",
        "classGroup": "四年级1班",
        "type": "realtime",
        "score": 86.5,
        "qualityGrade": "A",
        "reportStatus": "formal",
        "overview": "整体动作节奏稳定。",
        "biomechanical_analysis": "支撑脚落点略宽。",
        "magic_metaphor": "像钉子一样把支撑脚钉稳。",
        "action_plan": "完成支撑脚目标区练习。",
        "biomechanicalErrors": ["支撑脚位置偏离"],
        "scoreDetail": {
            "radar_scores": {
                "approach_rhythm": 18,
                "support_stability": 11,
                "backswing_folding": 16,
                "ankle_rigidity": 17,
                "whipping_velocity": 15,
            },
            "indicators": {
                "distance_cm": {
                    "value": 31.2,
                    "status": "YELLOW_APPROACHING",
                    "provenance": "measured",
                }
            },
        },
        "is_deleted": False,
    }
    db_path = tmp_path / "global_training_db.json"
    db_path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(api_server, "GLOBAL_DB_PATH", str(db_path))
    monkeypatch.setattr(db_mod, "start_auto_backup_daemon", lambda **_kwargs: False)
    monkeypatch.setattr(db_mod, "init_db", lambda bind=None: None)
    monkeypatch.setattr(
        api_server.self_check_service,
        "ensure_self_check_task",
        lambda _record: {
            **api_server.self_check_service.build_task_preview(_record),
            "taskId": "task-1",
        },
    )
    db_mod.stop_auto_backup_daemon()

    with TestClient(api_server.app) as client:
        response = client.get("/api/coach/records/detail-record/detail")

    assert response.status_code == 200
    detail = response.json()["record"]
    assert detail["studentId"] == "S008"
    assert detail["fiveDimensionScores"]["support_stability"] == 11
    assert detail["metrics"][0]["value"] == 31.2
    assert detail["aigcPrescription"]["actionPlan"] == "完成支撑脚目标区练习。"
    assert detail["selfCheckTask"]["dimensionKey"] == "support_stability"
