# -*- coding: utf-8 -*-
"""Cluster-RCT 模型层冒烟测试（内存 SQLite）。"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from models import (
    Base,
    ExperimentalGroup,
    ShotAttemptLog,
    StudentProfile,
    StudyTimepoint,
    TimepointSession,
)
from models.student_profile import EthicsIdentityMapping


@pytest.fixture()
def db_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)
    session = factory()
    try:
        yield session
        session.commit()
    finally:
        session.close()
        engine.dispose()


def test_student_profile_has_no_pii_columns():
    colnames = {c.name for c in StudentProfile.__table__.columns}
    assert "anonymous_id" in colnames
    assert "cluster_id" in colnames
    assert "experimental_group" in colnames
    assert "real_full_name" not in colnames
    assert "student_number" not in colnames
    assert "name" not in colnames


def test_immutable_randomization_fields(db_session: Session):
    profile = StudentProfile(
        anonymous_id="Sub_001",
        cluster_id="Class_1",
        experimental_group=ExperimentalGroup.GROUP_A_REALTIME,
    )
    db_session.add(profile)
    db_session.commit()

    profile.cluster_id = "Class_2"
    with pytest.raises(ValueError, match="不可变"):
        db_session.commit()
    db_session.rollback()


def test_shot_attempt_flat_metrics_and_unique_impact(db_session: Session):
    db_session.add(
        StudentProfile(
            anonymous_id="Sub_001",
            cluster_id="Class_1",
            experimental_group=ExperimentalGroup.GROUP_B_DELAYED,
        )
    )
    db_session.commit()

    log = ShotAttemptLog.from_score_detail(
        anonymous_id="Sub_001",
        session_date=date(2026, 3, 10),
        impact_frame_index=42,
        score_detail={
            "TotalScore": 86.5,
            "indicators": {
                "distance_cm": {"value": 17.2},
                "toe_angle": {"value": 8.0},
                "max_folding_angle": {"value": 78.0},
                "whipping_velocity": {"value": 620.0},
                "impact_knee_angle": {"value": 148.0},
                "ankle_rigidity": {"value": 2.1},
                "support_knee_angle": {"value": 155.0},
                "hip_torsion_angle": {"value": 28.0},
            },
        },
    )
    db_session.add(log)
    db_session.commit()

    assert log.impact_knee_angle == 148.0
    assert log.ankle_rigidity == 2.1
    assert "indicators" not in log.__dict__
    metrics = log.as_metric_dict()
    assert set(metrics) == {
        "distance_cm",
        "toe_angle",
        "max_folding_angle",
        "whipping_velocity",
        "impact_knee_angle",
        "ankle_rigidity",
        "support_knee_angle",
        "hip_torsion_angle",
    }


def test_timepoint_dose_compliance(db_session: Session):
    for i, aid in enumerate(("Sub_001", "Sub_002", "Sub_003"), start=1):
        db_session.add(
            StudentProfile(
                anonymous_id=aid,
                cluster_id="Class_1",
                experimental_group=ExperimentalGroup.GROUP_A_REALTIME,
            )
        )
    tp = TimepointSession(
        cluster_id="Class_1",
        timepoint=StudyTimepoint.T1,
        session_date=date(2026, 4, 1),
        experimental_group=ExperimentalGroup.GROUP_A_REALTIME,
        planned_min_shots=15,
    )
    db_session.add(tp)
    db_session.flush()

    # Sub_001 / Sub_002 达标；Sub_003 仅 5 次
    for aid, n in (("Sub_001", 15), ("Sub_002", 20), ("Sub_003", 5)):
        for frame in range(n):
            db_session.add(
                ShotAttemptLog(
                    anonymous_id=aid,
                    session_date=date(2026, 4, 1),
                    impact_frame_index=frame,
                    timepoint_session_id=tp.id,
                    impact_knee_angle=150.0,
                )
            )
    db_session.commit()

    report = tp.check_cluster_dose_compliance(db_session, min_shots=15)
    assert report.compliant_count == 2
    assert report.all_meet_threshold is False
    assert report.noncompliant_anonymous_ids == ("Sub_003",)

    report2 = TimepointSession.verify_dose_for_cluster_timepoint(
        db_session,
        cluster_id="Class_1",
        timepoint="T1",
        session_date=date(2026, 4, 1),
        min_shots=15,
    )
    assert report2.compliant_count == 2


def test_ethics_mapping_is_separate_table(db_session: Session):
    db_session.add(
        StudentProfile(
            anonymous_id="Sub_010",
            cluster_id="Class_2",
            experimental_group=ExperimentalGroup.GROUP_C_CONTROL,
        )
    )
    db_session.add(
        EthicsIdentityMapping(
            anonymous_id="Sub_010",
            real_full_name="测试同学",
            real_student_number="2026001",
            consent_version="v1.0",
        )
    )
    db_session.commit()

    profile = db_session.scalar(
        select(StudentProfile).where(StudentProfile.anonymous_id == "Sub_010")
    )
    assert profile is not None
    assert not hasattr(profile, "real_full_name") or not getattr(
        profile, "real_full_name", None
    )
    assert profile.stores_identifying_pii is False
