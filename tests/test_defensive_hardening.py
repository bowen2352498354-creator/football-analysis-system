# -*- coding: utf-8 -*-
"""防御性重构单测：丢帧/None 关节点、FSM Timeout Guard、LLM Fallback、Pydantic 422。"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from biomech_primitives import (
    LandmarkEMASmoother,
    calculate_3d_joint_angle,
    calculate_3d_joint_angle_or_none,
    gap_fill_scalar_series,
    is_valid_joint_point,
)
from deterministic_scorer import (
    FSM_APPROACH_TIMEOUT_FRAMES,
    fsm_should_timeout_reset,
    _roi_max_folding_angle,
)
from pose_tracker import (
    compute_knee_diagnosis_for_side,
    judge_knee_status,
    reset_knee_diagnosis_caches,
)
from workers.auto_shot_capture import AutoShotCaptureEngine, ShotFsmState


# ---------------------------------------------------------------------------
# Task 1：姿态关节点容错
# ---------------------------------------------------------------------------


class _FakeLM:
    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def test_calculate_3d_joint_angle_handles_none_and_acos_clip():
    """None / 非法点不抛；cos 钳制后 acos 安全。"""
    assert calculate_3d_joint_angle(None, (0, 0, 0), (1, 0, 0)) == 0.0
    assert calculate_3d_joint_angle_or_none(None, (0, 0, 0), (1, 0, 0)) is None
    # 共线近似 → 数值稳定
    ang = calculate_3d_joint_angle((1, 0, 0), (0, 0, 0), (1, 1e-12, 0))
    assert np.isfinite(ang)
    assert is_valid_joint_point((1.0, 2.0, 3.0), 0.9) is True
    assert is_valid_joint_point((1.0, 2.0, 3.0), 0.2) is False
    assert is_valid_joint_point(None, 1.0) is False


def test_landmark_ema_returns_cache_on_dropout():
    sm = LandmarkEMASmoother(alpha=0.5, jump_max_px=100.0)
    a = sm.update("ankle", (10.0, 20.0, 0.0), 0.9)
    assert a is not None
    dropped = sm.update("ankle", None, 0.0)
    assert dropped is not None
    np.testing.assert_allclose(dropped[:2], a[:2], atol=1e-6)
    # 突发跳变被拒
    jumped = sm.update("ankle", (10.0 + 500.0, 20.0, 0.0), 0.99)
    assert jumped is not None
    assert abs(float(jumped[0]) - float(a[0])) < 50.0


def test_pose_tracker_survives_all_none_landmarks():
    """全 None / 低置信度关节点：不崩溃，回退中性或缓存。"""
    reset_knee_diagnosis_caches()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    # 33 个全 None 列表 → Index 可访问但 visibility 低
    lms = [_FakeLM(0.5, 0.5, 0.0, visibility=0.0) for _ in range(33)]
    out = compute_knee_diagnosis_for_side(frame, lms, side="right")
    assert len(out) == 6
    angle, status, *_ = out
    assert np.isfinite(float(angle))
    assert status in ("Green", "Yellow", "Red")

    # 完全无 landmarks
    out2 = compute_knee_diagnosis_for_side(frame, None, side="right")
    assert len(out2) == 6

    # 空列表越界
    out3 = compute_knee_diagnosis_for_side(frame, [], side="left")
    assert len(out3) == 6


def test_pose_tracker_10pct_dropout_stream_stable():
    """模拟约 10% 关节点丢失的连续帧流，系统持续给出有限膝角。"""
    reset_knee_diagnosis_caches()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)

    for i in range(40):
        lms = []
        for j in range(33):
            if rng.random() < 0.10:
                lms.append(None)
            else:
                # 右髋/膝/踝附近合理归一化坐标
                base = {
                    24: (0.55, 0.45),
                    26: (0.58, 0.62),
                    28: (0.60, 0.78),
                    23: (0.45, 0.45),
                    25: (0.42, 0.62),
                    27: (0.40, 0.78),
                }.get(j, (0.5, 0.5))
                jitter = rng.normal(0, 0.005, size=2)
                lms.append(
                    _FakeLM(
                        base[0] + float(jitter[0]),
                        base[1] + float(jitter[1]),
                        0.0,
                        visibility=0.85,
                    )
                )
        angle, status, *_rest = compute_knee_diagnosis_for_side(frame, lms, side="right")
        assert np.isfinite(float(angle))
        assert status in ("Green", "Yellow", "Red")


def test_judge_knee_status_threshold_unchanged():
    """三色判定阈值 140–160 零破坏。"""
    assert judge_knee_status(150)[0] == "Green"
    assert judge_knee_status(135)[0] == "Yellow"
    assert judge_knee_status(120)[0] == "Red"
    assert judge_knee_status(None)[0] == "Red"


# ---------------------------------------------------------------------------
# Task 2：FSM Timeout Guard + 补帧
# ---------------------------------------------------------------------------


def test_fsm_approach_timeout_resets_to_idle():
    engine = AutoShotCaptureEngine(rolling_maxlen=80)
    engine.approach_timeout_frames = 10
    assert engine.notify_approach(omega=120.0) is True
    assert engine.state == ShotFsmState.APPROACH

    # 推入超过 timeout 的帧且未锁定触球 → 强制 IDLE
    for i in range(0, 12):
        engine.push_frame(np.zeros((16, 16, 3), dtype=np.uint8), i)

    assert engine.state == ShotFsmState.IDLE


def test_fsm_should_timeout_reset_helper():
    assert fsm_should_timeout_reset("APPROACH", FSM_APPROACH_TIMEOUT_FRAMES) is True
    assert fsm_should_timeout_reset("APPROACH", FSM_APPROACH_TIMEOUT_FRAMES - 1) is False
    assert fsm_should_timeout_reset("IDLE", 999) is False
    assert fsm_should_timeout_reset("准备击球", 60) is True


def test_gap_fill_scalar_series_fills_single_dropout():
    filled = gap_fill_scalar_series([10.0, None, 30.0], max_gap=2)
    assert filled[1] == pytest.approx(20.0)
    # 空洞过长不填
    long_gap = gap_fill_scalar_series([1.0, None, None, None, 5.0], max_gap=2)
    assert long_gap[2] is None


def test_roi_folding_survives_missing_joint_frames():
    """ROI 折叠角：中间帧缺踝仍可经补帧得到极值。"""
    frames = []
    for i in range(10):
        # 关节点放在 Y-Z 矢状面（膝角算法已废弃纯 XY/3D arccos）
        rec = {
            "right_hip": [0.0, 1.0, 0.0],
            "right_knee": [0.0, 0.0, 0.0],
            "right_ankle": [0.0, 0.0, 0.5],
            "visibility": {
                "right_hip": 1.0,
                "right_knee": 1.0,
                "right_ankle": 1.0,
            },
        }
        if i == 4:
            rec["right_ankle"] = None
            rec["visibility"]["right_ankle"] = 0.0
        frames.append(rec)
    fold, idx, ok = _roi_max_folding_angle(frames, t_impact=8, roi_start=0, roi_end=9)
    assert ok is True
    assert fold is not None
    assert np.isfinite(fold)


# ---------------------------------------------------------------------------
# Task 4：LLM Fallback（断网）
# ---------------------------------------------------------------------------


def test_llm_agent_fallback_when_client_none(monkeypatch):
    import llm_agent as la

    monkeypatch.setattr(la, "client", None)
    diagnosis = {
        "score_detail": {
            "indicators": {
                "distance_cm": {
                    "value": 28.0,
                    "provenance": "measured",
                    "status": "RED_DEVIATED",
                    "unit": "cm",
                }
            }
        }
    }
    dual = la.generate_optimal_dual_feedback(diagnosis)
    assert dual["correction_metaphor"] and dual["praise_encouragement"]
    assert dual["correction_metaphor"].startswith("你刚才")
    assert "就像" in dual["correction_metaphor"] and "下次试试" in dual["correction_metaphor"]
    assert "跨栏" in dual["correction_metaphor"]
    text = la.generate_feedback(diagnosis)
    assert isinstance(text, str) and len(text) > 0
    assert "【魔法指令】" in text and "【闪光点发现】" in text


def test_chat_backoff_eventually_falls_back(monkeypatch):
    import llm_agent as la

    calls = {"n": 0}

    class _Boom:
        def create(self, *args, **kwargs):
            calls["n"] += 1
            raise TimeoutError("simulated timeout")

    class _Client:
        chat = type("C", (), {"completions": _Boom()})()

    monkeypatch.setattr(la, "client", _Client())
    monkeypatch.setattr(la, "LLM_MAX_RETRIES", 3)
    monkeypatch.setattr(la, "LLM_BACKOFF_BASE_SEC", 0.01)
    monkeypatch.setattr(la.time, "sleep", lambda *_a, **_k: None)

    text = la.generate_feedback(
        {
            "score_detail": {
                "indicators": {
                    "impact_knee_angle": {
                        "value": 120.0,
                        "status": "RED_DEVIATED",
                        "provenance": "measured",
                    }
                }
            }
        }
    )
    assert calls["n"] == 3
    assert isinstance(text, str) and len(text) > 0


def test_optimal_json_parse_falls_back_on_garbage():
    import llm_agent as la

    dual = la._parse_optimal_dual_feedback("这不是JSON，模型抽风了", None)
    assert dual == la._STATIC_OPTIMAL_FALLBACK


def test_optimal_json_parse_accepts_valid_payload():
    import llm_agent as la

    raw = '{"correction_metaphor": "别用僵尸腿踢球！", "praise_encouragement": "踢得很用力！"}'
    dual = la._parse_optimal_dual_feedback(raw, None)
    assert dual["correction_metaphor"] == "别用僵尸腿踢球！"
    assert dual["praise_encouragement"] == "踢得很用力！"


# ---------------------------------------------------------------------------
# Task 4：api_server Pydantic 422
# ---------------------------------------------------------------------------


def test_generate_report_request_rejects_blank_session_id():
    from api_server import GenerateReportRequest

    with pytest.raises(ValidationError):
        GenerateReportRequest(session_id="   ", student_number="s1")
    with pytest.raises(ValidationError):
        GenerateReportRequest(session_id=123)  # type: ignore[arg-type]
    ok = GenerateReportRequest(session_id="abc-1", student_number="b101")
    assert ok.session_id == "abc-1"


def test_save_session_request_rejects_non_dict_items():
    from api_server import SaveSessionRequest

    with pytest.raises(ValidationError):
        SaveSessionRequest(sessions=["not-a-dict"])  # type: ignore[list-item]
    ok = SaveSessionRequest(sessions=[{"studentNumber": "b101"}])
    assert len(ok.sessions) == 1


def test_aggregate_report_request_accepts_float_scores_and_coerces_attempt_number():
    """聚合诊断：嵌套叶子全部 float；attemptNumber/score/hitStats/radar 均不得因小数 422。"""
    import numpy as np

    from api_server import (
        AggregateAttemptSummary,
        GenerateAggregateReportRequest,
        GenerateClassPrescriptionRequest,
        GenerateIndividualSummaryRequest,
    )

    item = AggregateAttemptSummary(
        attemptNumber=2.0,
        score=72.35,
        hitStats={"green": 5.0, "yellow": 1.5, "red": 0},
        radar_scores={"ankle_rigidity": 18.200000000000003, "whipping_velocity": 10},
    )
    assert item.attemptNumber == pytest.approx(2.0)
    assert item.score == pytest.approx(72.35)
    assert item.hitStats is not None
    assert item.hitStats.yellow == pytest.approx(1.5)
    assert item.radar_scores is not None
    assert item.radar_scores["ankle_rigidity"] == pytest.approx(18.2)

    payload = GenerateAggregateReportRequest(
        student_number="b101",
        attempts=[
            {
                "attemptNumber": np.float64(1.0),
                "score": 68.2,
                "hitStats": {"green": 3.2, "yellow": 2.1, "red": 1.0},
                "avgKneeAngle": 142.333,
            },
            {"attemptNumber": 2, "score": 74.88, "hitStats": None},
        ],
    )
    assert len(payload.attempts) == 2
    assert payload.attempts[0].attemptNumber == pytest.approx(1.0)
    assert payload.attempts[0].score == pytest.approx(68.2)
    assert payload.attempts[0].avgKneeAngle == pytest.approx(142.333)
    assert payload.attempts[1].score == pytest.approx(74.88)

    class_req = GenerateClassPrescriptionRequest(
        errorStats={"膝关节过度屈曲": 33.3, "支撑脚位置偏离": 12.0},
        totalRecords=10.0,
        avgScore=72.5,
    )
    assert class_req.totalRecords == pytest.approx(10.0)
    assert class_req.errorStats["膝关节过度屈曲"] == pytest.approx(33.3)

    individual = GenerateIndividualSummaryRequest(
        studentId="b101",
        scoreHistory=[60.5, 70.25, 80.0],
        errorCounter={"随摆转髋不足": 2.0, "身体重心偏移": 1.5},
    )
    assert individual.errorCounter["身体重心偏移"] == pytest.approx(1.5)
    assert individual.scoreHistory[1] == pytest.approx(70.25)
