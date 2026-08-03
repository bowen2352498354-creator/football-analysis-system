# -*- coding: utf-8 -*-
"""软删除保护 + 自动备份守护单测。"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select

from db import (
    create_database_backup,
    hard_delete_shot_attempts_by_ids,
    init_db,
    latest_backup_mtime,
    prune_old_backups,
    session_scope,
    should_run_backup,
    soft_delete_shot_attempt_matching,
    soft_delete_shot_attempts_by_ids,
    stop_auto_backup_daemon,
)
from models.enums import ExperimentalGroup
from models.shot_attempt_log import ShotAttemptLog
from models.student_profile import StudentProfile


@pytest.fixture()
def tmp_json_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "global_training_db.json"
    records = [
        {
            "id": "alive-1",
            "studentId": "S001",
            "school": "学校一",
            "classGroup": "五年级一班",
            "type": "realtime",
            "score": 80,
            "timestamp": "2026-08-03 08:00:00",
            "testDate": "2026-08-03",
            "is_deleted": False,
        },
        {
            "id": "ghost-1",
            "studentId": "S002",
            "school": "学校一",
            "classGroup": "五年级一班",
            "type": "delayed",
            "score": 70,
            "timestamp": "2026-08-03 08:10:00",
            "testDate": "2026-08-03",
            "is_deleted": True,
        },
    ]
    db_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    import api_server
    import db as db_mod

    monkeypatch.setattr(api_server, "GLOBAL_DB_PATH", str(db_path))
    # 避免 TestClient lifespan 写真实 backups/ 或启动守护线程
    monkeypatch.setattr(db_mod, "start_auto_backup_daemon", lambda **_kwargs: False)
    monkeypatch.setattr(db_mod, "init_db", lambda bind=None: None)
    stop_auto_backup_daemon()
    yield db_path, api_server


def test_get_all_records_hides_soft_deleted_but_keeps_on_disk(tmp_json_db):
    db_path, api_server = tmp_json_db
    with TestClient(api_server.app) as client:
        resp = client.get("/api/get_all_records")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        ids = {r["id"] for r in body["records"]}
        assert ids == {"alive-1"}

        # 硬盘上软删记录仍在
        on_disk = json.loads(db_path.read_text(encoding="utf-8"))
        assert {r["id"] for r in on_disk} == {"alive-1", "ghost-1"}
        assert any(r["id"] == "ghost-1" and r.get("is_deleted") is True for r in on_disk)

        audit = client.get("/api/get_all_records", params={"include_deleted": True})
        assert {r["id"] for r in audit.json()["records"]} == {"alive-1", "ghost-1"}


def test_coach_delete_is_soft_not_physical(tmp_json_db):
    db_path, api_server = tmp_json_db
    with TestClient(api_server.app) as client:
        resp = client.post("/api/coach/delete_record", json={"id": "alive-1"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body.get("deleted") is True

        on_disk = json.loads(db_path.read_text(encoding="utf-8"))
        by_id = {r["id"]: r for r in on_disk}
        assert "alive-1" in by_id
        assert by_id["alive-1"].get("is_deleted") is True

        listed = client.get("/api/get_all_records").json()["records"]
        assert listed == []


def test_batch_delete_soft_marks_and_hides(tmp_json_db):
    db_path, api_server = tmp_json_db
    records = json.loads(db_path.read_text(encoding="utf-8"))
    records.append(
        {
            "id": "alive-2",
            "studentId": "S003",
            "school": "学校一",
            "classGroup": "五年级一班",
            "type": "realtime",
            "score": 88,
            "timestamp": "2026-08-03 09:00:00",
            "testDate": "2026-08-03",
            "is_deleted": False,
        }
    )
    db_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    with TestClient(api_server.app) as client:
        resp = client.post("/api/records/batch", json={"ids": ["alive-1", "alive-2"]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert set(body["deletedIds"]) == {"alive-1", "alive-2"}

    on_disk = json.loads(db_path.read_text(encoding="utf-8"))
    assert len(on_disk) == 3
    assert all(
        r.get("is_deleted") is True for r in on_disk if r["id"] in {"alive-1", "alive-2"}
    )


def test_orm_hard_delete_api_forwards_to_soft(tmp_path: Path):
    db_file = tmp_path / "cluster.db"
    engine = create_engine(f"sqlite:///{db_file.as_posix()}", future=True)
    init_db(engine)

    with session_scope(engine) as session:
        session.add(
            StudentProfile(
                anonymous_id="anon-1",
                cluster_id="C1",
                experimental_group=ExperimentalGroup.GROUP_A_REALTIME,
            )
        )
        session.add(
            ShotAttemptLog(
                anonymous_id="anon-1",
                session_date=date(2026, 8, 3),
                impact_frame_index=12,
                total_score=77.0,
                is_deleted=False,
            )
        )

    marked = hard_delete_shot_attempts_by_ids([1], bind=engine)
    assert marked == 1

    with session_scope(engine) as session:
        rows = list(session.scalars(select(ShotAttemptLog)).all())
        assert len(rows) == 1
        assert rows[0].is_deleted is True

    with session_scope(engine) as session:
        session.add(
            ShotAttemptLog(
                anonymous_id="anon-1",
                session_date=date(2026, 8, 3),
                impact_frame_index=99,
                total_score=66.0,
                is_deleted=False,
            )
        )
    n = soft_delete_shot_attempt_matching(
        anonymous_id="anon-1",
        session_date="2026-08-03",
        total_score=66.0,
        bind=engine,
    )
    assert n == 1
    # 已软删的行不会被二次计数
    assert soft_delete_shot_attempts_by_ids([2], bind=engine) == 0


def test_backup_creates_bak_and_prunes(tmp_path: Path):
    json_path = tmp_path / "global_training_db.json"
    sqlite_path = tmp_path / "cluster_rct.db"
    backup_dir = tmp_path / "backups"
    json_path.write_text("[]", encoding="utf-8")
    sqlite_path.write_bytes(b"sqlite-bytes")

    path = create_database_backup(
        backup_dir=str(backup_dir),
        global_db_path=str(json_path),
        sqlite_db_path=str(sqlite_path),
        force=True,
        now=datetime(2026, 8, 3, 8, 44),
    )
    assert path is not None
    assert path.endswith(".bak")
    assert os.path.basename(path).startswith("db_backup_20260803_0844")
    assert os.path.isfile(path)
    assert (
        should_run_backup(str(backup_dir), interval_hours=12, now=datetime(2026, 8, 3, 8, 50))
        is False
    )
    assert (
        should_run_backup(str(backup_dir), interval_hours=12, now=datetime(2026, 8, 4, 8, 50))
        is True
    )

    old = backup_dir / "db_backup_20260101_0000.bak"
    old.write_bytes(b"old")
    old_mtime = time.mktime(datetime(2026, 1, 1, 0, 0).timetuple())
    os.utime(old, (old_mtime, old_mtime))
    removed = prune_old_backups(
        str(backup_dir),
        retention_days=30,
        now=datetime(2026, 8, 3, 8, 44),
    )
    assert removed >= 1
    assert not old.exists()
    assert latest_backup_mtime(str(backup_dir)) is not None
