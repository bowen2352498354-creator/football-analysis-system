# -*- coding: utf-8 -*-
"""
llm_agent.py
少儿足球生物力学教练 · 四维深度诊断报告（大模型代理模块）

【权限铁律】
    评分由 DeterministicScorer 纯数学独占；大模型绝不改写分数。
    本模块把 score_detail 全量测量上下文（TotalScore / radar_scores / deductions）
    注入 Prompt，转译为千人千面的四维诊断 JSON。

功能说明：
    1. 使用官方 openai Python 库调用 DeepSeek（OpenAI 兼容协议）；
    2. generate_feedback / generate_session_report → 四维深度报告 JSON；
    3. System Prompt 强制：引用真实测量数据、动力链病理分析、具身隐喻、训练指令；
    4. 强语境注入：总分 + 雷达 + 主/次扣分病灶动态锁定；
    5. response_format=json_object + 严谨 json.loads；解析失败走动态数据兜底；
    6. temperature≈0.65 提升多样性；网络失败仍拼接 TotalScore 与最严重扣分项。
"""

import json
import os
import re
import time
from typing import Any, Optional

from dotenv import load_dotenv

from error_diagnoser import (
    is_aigc_measurable_provenance,
)

# --------------------------------------------------------------------------
# 第一步：配置 DeepSeek 的 API Key 与接口地址
# --------------------------------------------------------------------------

# 【安全机制】API Key 绝对不能以明文形式写在代码里。
# 这里通过 python-dotenv 从项目根目录下的 .env 文件（该文件已被 .gitignore
# 忽略，不会被提交到 Git/GitHub）加载环境变量，再用 os.getenv 读取，
# 从根源上避免 Key 泄露到公开仓库。
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# DeepSeek 官方接口地址完全兼容 OpenAI 的 SDK 调用方式，
# 只需要把 base_url 从默认的 OpenAI 官方地址换成 DeepSeek 的地址即可
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# DeepSeek 提供的对话模型名称（对应"深度思考"关闭状态下的标准对话模型）
DEEPSEEK_MODEL_NAME = "deepseek-chat"

# 【网络韧性】四维报告 Prompt 更长，超时放宽避免频繁落入静态模板
LLM_TIMEOUT_SEC = 25.0
LLM_MAX_RETRIES = 3
LLM_BACKOFF_BASE_SEC = 0.8

# 无 Key / 初始化失败时 client=None，所有 generate_* 自动降级到规则引擎文案
client = None
if not DEEPSEEK_API_KEY:
    print(
        "【llm_agent】警告：未检测到 DEEPSEEK_API_KEY，"
        "AIGC 将全程使用静态模板 Fallback（系统不崩溃）。"
    )
else:
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=LLM_TIMEOUT_SEC,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"【llm_agent】OpenAI 客户端初始化失败，启用 Fallback：{exc}")
        client = None


def _chat_completions_with_backoff(
    *,
    messages: list,
    temperature: float,
    response_format: Optional[dict] = None,
) -> Any:
    """带 5s 超时与指数退避重试的 chat.completions 调用。

    全部重试失败或 client 不可用时抛出最后一次异常，由调用方切换 Fallback。
    """
    if client is None:
        raise RuntimeError("LLM client unavailable (missing API key or init failure)")

    last_exc: Optional[BaseException] = None
    for attempt in range(int(LLM_MAX_RETRIES)):
        try:
            kwargs: dict[str, Any] = {
                "model": DEEPSEEK_MODEL_NAME,
                "messages": messages,
                "temperature": temperature,
                "timeout": LLM_TIMEOUT_SEC,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= int(LLM_MAX_RETRIES) - 1:
                break
            sleep_s = float(LLM_BACKOFF_BASE_SEC) * (2 ** attempt)
            print(
                f"【llm_agent】API 调用失败（第 {attempt + 1}/{LLM_MAX_RETRIES} 次），"
                f"{sleep_s:.1f}s 后指数退避重试：{exc}"
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc

# --------------------------------------------------------------------------
# 第二步：System Prompt —— 四维深度诊断报告（不可违抗）
# --------------------------------------------------------------------------

# 略提高温度以打破复读机；结构仍由 JSON schema + 解析校验锁定
LLM_TEMPERATURE = 0.65

# 四维字段字数软上限（中文字符）
# 教练/研究者面向字段（overview / biomechanical_analysis）需要科研级论述深度，
# 上限放宽；儿童面向字段（magic_metaphor / action_plan）保持短句易记。
_OVERVIEW_MAX_CHARS = 160
_BIOMECH_MAX_CHARS = 420
_MAGIC_MAX_CHARS = 50
_ACTION_MAX_CHARS = 30

# 兼容旧常量名（部分测试 / 内部调用仍引用）
_CLINICAL_ECHO_MAX_CHARS = _OVERVIEW_MAX_CHARS
_CORRECTION_TARGET_CHARS = _MAGIC_MAX_CHARS
_CORRECTION_MAX_CHARS = 55
_PRAISE_MAX_CHARS = _ACTION_MAX_CHARS

_RADAR_LABEL_ZH = {
    "support_stability": "支撑稳定",
    "backswing_folding": "后摆折叠",
    "ankle_rigidity": "脚踝刚性",
    "whipping_velocity": "鞭打速度",
    "approach_rhythm": "助跑节奏",
}

SYSTEM_PROMPT = (
    "你是一名专业的少儿足球生物力学教练。请根据下面传入的真实测量数据，"
    "撰写一份详尽、千人千面的诊断报告。绝不能每次说一样的话。"
    "必须返回如下 JSON 结构：\n"
    "{\n"
    '  "overview": "【综合评价】(结合总分和雷达图优势，用肯定语气开场，'
    f'描述本次动作的整体观感，{_OVERVIEW_MAX_CHARS}字)",\n'
    '  "biomechanical_analysis": "【动力链病理分析】(这是报告的核心！'
    "必须引用传入的具体数据，例如‘你的支撑脚距离比例达到了 1.3’，"
    "并运用生物力学逻辑解释这个原发性错误是如何导致后续膝盖或脚踝错误的，"
    f'{_BIOMECH_MAX_CHARS}字)",\n'
    '  "magic_metaphor": "【具身隐喻处方】(针对最核心的病灶，生成通俗、'
    "有画面感且带有一点幽默的纠错比喻，避免与之前的提示重复，"
    f'{_MAGIC_MAX_CHARS}字)",\n'
    '  "action_plan": "【下一步训练指令】(给出明确的身体部位控制指令，'
    f'{_ACTION_MAX_CHARS}字)"\n'
    "}\n"
    "【铁律】绝不计算或修改分数；不得编造未提供的测量值；"
    "四个字段都必须填写且互不重复；只返回合法 JSON，禁止 Markdown 围栏；"
    "必须使用简体中文。"
)

# 会话报告与单次反馈共用同一不可违抗命令（动态注入时由 build_system_prompt 覆盖）
REPORT_SYSTEM_PROMPT = SYSTEM_PROMPT

# 解析失败时的硬性兜底表扬（旧测试兼容；动态兜底会覆盖）
_HARD_FALLBACK_PRAISE = "动作框架很棒，发力非常坚决，继续保持这种自信！"

# 无诊断上下文时的最后兜底（仅当完全无 score_detail 时使用）
_STATIC_OPTIMAL_FALLBACK = {
    "overview": "本次动作已完成量化诊断，整体框架积极可圈可点。",
    "biomechanical_analysis": "本次缺少完整扣分明细，建议结合支撑脚站位与摆动腿折叠优先排查动力链近端偏差。",
    "magic_metaphor": (
        "你刚才踢球时膝盖太直了，就像一根冻住的冰棍一样！下次试试像弹簧一样把小腿弯一弯再弹出去。"
    ),
    "action_plan": "慢动作练支撑脚近球落位。",
    "correction_metaphor": (
        "你刚才踢球时膝盖太直了，就像一根冻住的冰棍一样！下次试试像弹簧一样把小腿弯一弯再弹出去。"
    ),
    "praise_encouragement": "这次尝试非常勇敢，继续保持！",
}

# 红色缺陷优先级（供兜底规则选用对应话术）
_RED_DEFECT_PRIORITY = (
    "ankle_rigidity",
    "distance_cm",
    "toe_angle",
    "max_folding_angle",
    "impact_knee_angle",
    "support_knee_angle",
    "hip_torsion_angle",
    "whipping_velocity",
)

_INDICATOR_LABEL_ZH = {
    "ankle_rigidity": "脚踝锁紧",
    "distance_cm": "支撑脚站位",
    "toe_angle": "支撑脚尖方向",
    "max_folding_angle": "摆动腿折叠",
    "impact_knee_angle": "触球膝盖",
    "support_knee_angle": "支撑膝盖",
    "hip_torsion_angle": "转髋",
    "trunk_lean_angle": "躯干倾角",
    "whipping_velocity": "摆腿速度",
}

# 首要错误对外描述标签（偏「失误事实」口吻，供强语境注入 / 硬兜底）
_PRIMARY_ERROR_LABEL_ZH = {
    "ankle_rigidity": "脚踝松弛",
    "distance_cm": "支撑脚偏移",
    "toe_angle": "支撑脚尖偏移",
    "max_folding_angle": "摆动腿折叠不足",
    "impact_knee_angle": "触球膝盖过直",
    "support_knee_angle": "支撑膝盖过直",
    "hip_torsion_angle": "躯干后仰/转髋不足",
    "trunk_lean_angle": "躯干后仰/过度折腰",
    "whipping_velocity": "随前动作不足",
}

_PRIMARY_ERROR_QUALIFIER = {
    "ankle_rigidity": "锁踝不够",
    "distance_cm": "距离过远",
    "toe_angle": "指向偏离",
    "max_folding_angle": "折叠不够",
    "impact_knee_angle": "伸展失控",
    "support_knee_angle": "支撑不稳",
    "hip_torsion_angle": "重心后倒",
    "trunk_lean_angle": "重心后仰",
    "whipping_velocity": "鞭打不足",
}

_INDICATOR_UNIT = {
    "ankle_rigidity": "deg",  # 【V3.9】最大形变落差角 deflection_deg
    "distance_cm": "cm",
    "toe_angle": "°",
    "max_folding_angle": "°",
    "impact_knee_angle": "°",
    "support_knee_angle": "°",
    "hip_torsion_angle": "°",
    "trunk_lean_angle": "°",
    "whipping_velocity": "°/s",
}

# 缺陷键 → OPTIMAL 双段兜底（纠错三段式 + 表扬）；模型失败时按主缺陷选用
_OPTIMAL_FALLBACK_BY_DEFECT = {
    "ankle_rigidity": {
        "correction_metaphor": (
            "你刚才触球时脚腕软绵绵的，就像煮熟的面条一样！"
            "下次试试把脚面绷紧，变成一块硬硬的铁板。"
        ),
        "praise_encouragement": "你敢用力出脚，超勇敢！",
    },
    "distance_cm": {
        "correction_metaphor": (
            "你刚才踢球时支撑脚离球太远了，就像跨栏一样！"
            "下次试试把脚踩在离球一拳近的地方。"
        ),
        "praise_encouragement": "这次跑过来很有劲，真棒！",
    },
    "toe_angle": {
        "correction_metaphor": (
            "你刚才支撑脚尖指歪了，就像跨栏跑偏一样！"
            "下次试试让脚尖对准球门再踢球。"
        ),
        "praise_encouragement": "你眼睛盯着球，很专注！",
    },
    "max_folding_angle": {
        "correction_metaphor": (
            "你刚才踢球时膝盖太直了，就像一根冻住的冰棍一样！"
            "下次试试像弹簧一样把小腿弯一弯再弹出去。"
        ),
        "praise_encouragement": "这一脚踢得很用力，好样的！",
    },
    "impact_knee_angle": {
        "correction_metaphor": (
            "你刚才踢球时膝盖太直了，就像一根冻住的冰棍一样！"
            "下次试试像弹簧一样把小腿弯一弯再弹出去。"
        ),
        "praise_encouragement": "你出脚很果断，继续加油！",
    },
    "support_knee_angle": {
        "correction_metaphor": (
            "你刚才支撑腿绷得太直了，就像一根冻住的冰棍一样！"
            "下次试试膝盖微微弯一点再站稳。"
        ),
        "praise_encouragement": "你站得很认真，真不错！",
    },
    "hip_torsion_angle": {
        "correction_metaphor": (
            "你刚才踢球时身子往后倒了，就像坐在太师椅上一样！"
            "下次试试把胸口像大虾一样往前压一压。"
        ),
        "praise_encouragement": "你全身都在动，很有精神！",
    },
    "whipping_velocity": {
        "correction_metaphor": (
            "你刚才踢完球脚马上就停住了，就像踩了急刹车一样！"
            "下次试试踢完后，让脚跟着球再往前送一送。"
        ),
        "praise_encouragement": "你越踢越敢试，太棒了！",
    },
}

# 兼容旧名：单句纠错隐喻（供少数内部调用）
_METAPHOR_FALLBACK_BY_DEFECT = {
    key: val["correction_metaphor"] for key, val in _OPTIMAL_FALLBACK_BY_DEFECT.items()
}

# 旧三节制兜底仍保留给实测值复述测试（不对孩子展示学术词；AIGC 主路径已弃用）
_FALLBACK_CAUSE_BY_DEFECT = {
    key: val["correction_metaphor"] for key, val in _OPTIMAL_FALLBACK_BY_DEFECT.items()
}
_FALLBACK_ACTION_BY_DEFECT = {
    key: val["praise_encouragement"] for key, val in _OPTIMAL_FALLBACK_BY_DEFECT.items()
}


def _cause_fallback_for_defect(defect: str) -> str:
    """兼容旧接口：返回该缺陷的纠错隐喻。"""
    dual = _OPTIMAL_FALLBACK_BY_DEFECT.get(
        defect, _OPTIMAL_FALLBACK_BY_DEFECT["max_folding_angle"]
    )
    return dual["correction_metaphor"]


def _action_fallback_for_defect(defect: str) -> str:
    """兼容旧接口：返回该缺陷的表扬话术。"""
    dual = _OPTIMAL_FALLBACK_BY_DEFECT.get(
        defect, _OPTIMAL_FALLBACK_BY_DEFECT["max_folding_angle"]
    )
    return dual["praise_encouragement"]


def _status_to_band(status: Optional[str]) -> str:
    s = str(status or "")
    if s == "RED_DEVIATED":
        return "red"
    if s == "YELLOW_APPROACHING":
        return "yellow"
    if s == "GREEN_OPTIMAL":
        return "green"
    return "unknown"


def _band_text_for_item(key: str, item: dict, band: str) -> str:
    """把 green_band + 状态转成孩子可听的相对标准描述。"""
    if band == "green":
        return "落在理想区间"
    label = _PRIMARY_ERROR_LABEL_ZH.get(key) or _INDICATOR_LABEL_ZH.get(key, key)
    green = item.get("green_band")
    unit = str(item.get("unit") or _INDICATOR_UNIT.get(key, ""))
    band_hint = "略偏离理想" if band == "yellow" else "明显偏离理想"
    if isinstance(green, (list, tuple)) and len(green) >= 2 and green[0] is not None:
        try:
            gl, gh = float(green[0]), float(green[1]) if green[1] is not None else None
            if unit in {"ratio", "×肩宽"}:
                if gh is not None:
                    return f"{band_hint}（理想约 {gl:.2f}–{gh:.2f}×肩宽）"
                return f"{band_hint}（理想约 ≥{gl:.2f}×肩宽）"
            if unit in {"cm"}:
                if gh is not None:
                    return f"{band_hint}（理想约 {gl:.0f}–{gh:.0f}cm）"
            if unit in {"deg", "°"}:
                if gh is not None:
                    return f"{band_hint}（理想约 {gl:.0f}–{gh:.0f}°）"
        except (TypeError, ValueError):
            pass
    return f"{label}{band_hint}"


def _distance_display_fields(item: dict) -> tuple[Optional[float], str, Optional[float], Optional[float]]:
    """支撑距展示：优先肩宽比，其次 PCR cm。返回 (display_value, unit, estimate_cm, ratio)。"""
    unit = str(item.get("unit") or "cm").strip().lower()
    ratio = item.get("support_ratio")
    estimate = item.get("distance_cm_estimate")
    value = item.get("value")
    if value is None:
        value = item.get("scoring_value")
    try:
        if ratio is not None:
            ratio_f = float(ratio)
        elif unit in {"ratio", "×肩宽"} and value is not None:
            ratio_f = float(value)
        else:
            ratio_f = None
    except (TypeError, ValueError):
        ratio_f = None
    try:
        estimate_f = float(estimate) if estimate is not None else None
    except (TypeError, ValueError):
        estimate_f = None
    if ratio_f is not None and np_isfinite_safe(ratio_f):
        if estimate_f is None:
            try:
                from biomech_primitives import AVERAGE_CHILD_SHOULDER_WIDTH_CM

                estimate_f = float(ratio_f * AVERAGE_CHILD_SHOULDER_WIDTH_CM)
            except Exception:  # noqa: BLE001
                estimate_f = None
        return round(ratio_f, 4), "×肩宽", (
            round(estimate_f, 1) if estimate_f is not None else None
        ), round(ratio_f, 4)
    try:
        cm = float(value) if value is not None else None
    except (TypeError, ValueError):
        cm = None
    if cm is not None and np_isfinite_safe(cm):
        return round(cm, 2), "cm", round(cm, 1), None
    return None, unit or "cm", None, None


def np_isfinite_safe(v: float) -> bool:
    try:
        return bool(__import__("math").isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def _coach_fact_for_indicator(key: str, item: dict, *, quoteable: bool) -> str:
    """确定性教练事实句（供 Brief / clinical_echo 兜底）。"""
    label = _PRIMARY_ERROR_LABEL_ZH.get(key) or _INDICATOR_LABEL_ZH.get(key, key)
    qualifier = _qualify_primary_error(key, item) if item else "需要调整"
    if key == "distance_cm":
        disp, unit, estimate_cm, _ratio = _distance_display_fields(item)
        if quoteable and disp is not None and unit == "×肩宽":
            return f"支撑脚横距比例 {disp:.2f}，{qualifier}"
        if quoteable and disp is not None and unit == "cm":
            return f"支撑脚横距比例 {disp:.2f}，{qualifier}"
        return f"{label}，{qualifier}"
    if not quoteable:
        return f"{label}，{qualifier}（无可靠实测数字）"
    value_part = _format_indicator_value_unit(key, item)
    if value_part:
        return f"{label} {value_part}，{qualifier}"
    return f"{label}，{qualifier}"


def _brief_entry_for_key(key: str, item: dict) -> dict:
    """把单指标打成 ClinicalBrief 条目。"""
    provenance = str(item.get("provenance") or "").strip().lower() or "unknown"
    quoteable = is_aigc_measurable_provenance(provenance)
    status = item.get("status")
    band = _status_to_band(status)
    display_value: Optional[float] = None
    display_unit = str(item.get("unit") or _INDICATOR_UNIT.get(key, ""))
    estimate_cm: Optional[float] = None
    support_ratio: Optional[float] = None

    if key == "distance_cm":
        display_value, display_unit, estimate_cm, support_ratio = _distance_display_fields(item)
        if not quoteable:
            display_value = None
            estimate_cm = None
    elif quoteable:
        raw = item.get("value")
        if raw is None and key == "ankle_rigidity":
            raw = item.get("variance")
        try:
            display_value = round(float(raw), 4) if raw is not None else None
        except (TypeError, ValueError):
            display_value = None
        if display_unit in {"deg", "degree"}:
            display_unit = "°"
        elif display_unit in {"deg/s"}:
            display_unit = "°/s"
        elif display_unit == "variance":
            display_unit = "σ²"

    entry = {
        "metric": key,
        "label_zh": _INDICATOR_LABEL_ZH.get(key, key),
        "status": status,
        "band": band,
        "band_text": _band_text_for_item(key, item, band),
        "display_value": display_value,
        "display_unit": display_unit,
        "estimate_cm": estimate_cm,
        "support_ratio": support_ratio,
        # 故意不透出 penalty / TotalScore：AIGC 只读事实与 band，不算分
        "quoteable": bool(quoteable),
        "provenance": provenance,
        "method": item.get("method"),
        "coach_fact": _coach_fact_for_indicator(key, item, quoteable=quoteable),
    }
    return entry


def build_clinical_brief(diagnosis: Optional[dict] = None) -> dict:
    """由 score_detail 确定性组装 ClinicalBrief（LLM 只读、不算分）。"""
    diagnosis = diagnosis or {}
    detail = diagnosis.get("score_detail") or {}
    indicators = _diagnosis_indicators(diagnosis)
    deductions = detail.get("deductions") or diagnosis.get("deductions") or []
    if not isinstance(deductions, list):
        deductions = []

    primary_key = _pick_max_penalty_defect(diagnosis)
    if not primary_key:
        primary_key = _pick_primary_defect_key(diagnosis)

    primary_item = indicators.get(primary_key) if isinstance(indicators.get(primary_key), dict) else {}
    primary = _brief_entry_for_key(primary_key, primary_item) if primary_item else {
        "metric": primary_key or "max_folding_angle",
        "label_zh": _INDICATOR_LABEL_ZH.get(primary_key or "max_folding_angle", "动作偏差"),
        "status": None,
        "band": "unknown",
        "band_text": "需要调整",
        "display_value": None,
        "display_unit": "",
        "estimate_cm": None,
        "support_ratio": None,
        "quoteable": False,
        "provenance": "missing",
        "method": None,
        "coach_fact": build_primary_error_description(diagnosis),
    }

    # secondary：次高扣分且非 green
    ranked: list[tuple[float, int, str]] = []
    priority_rank = {k: i for i, k in enumerate(_RED_DEFECT_PRIORITY)}
    for key, item in indicators.items():
        if key == primary.get("metric") or not isinstance(item, dict):
            continue
        pen = _indicator_penalty(item)
        if pen <= 0:
            continue
        if _status_to_band(item.get("status")) == "green":
            continue
        ranked.append((-pen, priority_rank.get(key, 99), key))
    ranked.sort()
    secondary: list[dict] = []
    if ranked:
        sk = ranked[0][2]
        secondary.append(_brief_entry_for_key(sk, indicators[sk]))

    strengths: list[dict] = []
    for key in _RED_DEFECT_PRIORITY:
        item = indicators.get(key)
        if not isinstance(item, dict):
            continue
        if _status_to_band(item.get("status")) != "green":
            continue
        if not is_aigc_measurable_provenance(item.get("provenance")):
            continue
        if key == primary.get("metric"):
            continue
        entry = _brief_entry_for_key(key, item)
        strengths.append(
            {
                "metric": entry["metric"],
                "label_zh": entry["label_zh"],
                "coach_fact": f"{entry['label_zh']}表现不错",
                "quoteable": True,
            }
        )
        if len(strengths) >= 2:
            break

    blocked: list[dict] = []
    for key in _FOCUS_INDICATOR_KEYS:
        item = indicators.get(key)
        if not isinstance(item, dict):
            blocked.append({"metric": key, "reason": "未提供该指标"})
            continue
        if not is_aigc_measurable_provenance(item.get("provenance")):
            blocked.append({"metric": key, "reason": "未提供实测值，禁止口述角度/距离数字"})

    deduction_echo: list[str] = []
    for row in deductions[:4]:
        if isinstance(row, dict) and row.get("reason"):
            deduction_echo.append(str(row["reason"]))
        elif isinstance(row, str) and row.strip():
            deduction_echo.append(row.strip())

    return {
        "primary": primary,
        "secondary": secondary,
        "strengths": strengths,
        "blocked": blocked,
        "deduction_echo": deduction_echo,
    }


def _clinical_echo_from_brief(brief: Optional[dict]) -> str:
    """无模型输出时，用 Brief.primary.coach_fact 生成 clinical_echo。"""
    brief = brief or {}
    primary = brief.get("primary") or {}
    fact = str(primary.get("coach_fact") or "").strip()
    if not fact:
        fact = "本次动作有一点小偏差，需要微调"
    return _clamp_optimal_phrase(fact, limit=_CLINICAL_ECHO_MAX_CHARS) or fact[:_CLINICAL_ECHO_MAX_CHARS]


def _normalize_diagnosis_json(diagnosis_json) -> dict:
    """把 error_diagnoser 输出（dict / JSON 字符串）规范为 dict。"""
    if diagnosis_json is None:
        return {}
    if isinstance(diagnosis_json, dict):
        return diagnosis_json
    if isinstance(diagnosis_json, str):
        try:
            parsed = json.loads(diagnosis_json)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {"raw_text": diagnosis_json}
    return {}


def _diagnosis_indicators(diagnosis: Optional[dict]) -> dict:
    """统一取出 score_detail.indicators / indicators。"""
    diagnosis = diagnosis or {}
    detail = diagnosis.get("score_detail") or {}
    indicators = detail.get("indicators") or diagnosis.get("indicators") or {}
    return indicators if isinstance(indicators, dict) else {}


def _indicator_penalty(item: dict) -> float:
    """读取单指标扣分；非法值视为 0。"""
    try:
        return float(item.get("penalty") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _extract_total_score(diagnosis: Optional[dict] = None) -> Optional[float]:
    """从 score_detail / 顶层读取 TotalScore。"""
    diagnosis = diagnosis or {}
    detail = diagnosis.get("score_detail") or {}
    for src in (detail, diagnosis):
        if not isinstance(src, dict):
            continue
        for key in ("TotalScore", "total_score", "totalScore", "score"):
            if src.get(key) is None:
                continue
            try:
                return round(float(src[key]), 2)
            except (TypeError, ValueError):
                continue
    return None


def _extract_radar_scores(diagnosis: Optional[dict] = None) -> dict:
    """读取五维雷达分；非法结构返回空 dict。"""
    diagnosis = diagnosis or {}
    detail = diagnosis.get("score_detail") or {}
    radar = detail.get("radar_scores") or diagnosis.get("radar_scores") or {}
    if not isinstance(radar, dict):
        return {}
    out: dict = {}
    for key, val in radar.items():
        try:
            out[str(key)] = round(float(val), 1)
        except (TypeError, ValueError):
            continue
    return out


def _format_green_band(item: dict) -> str:
    """把 green_band 格式化为可读阈值片段。"""
    green = item.get("green_band") if isinstance(item, dict) else None
    if not isinstance(green, (list, tuple)) or len(green) < 2:
        return ""
    unit = str(item.get("unit") or "")
    try:
        gl = float(green[0]) if green[0] is not None else None
        gh = float(green[1]) if green[1] is not None else None
    except (TypeError, ValueError):
        return ""
    if gl is None and gh is None:
        return ""
    if unit in {"ratio", "×肩宽"}:
        if gl is not None and gh is not None:
            return f"理想阈值 {gl:.2f}–{gh:.2f}×肩宽"
        if gl is not None:
            return f"理想阈值 ≥{gl:.2f}×肩宽"
    if unit in {"cm"}:
        if gl is not None and gh is not None:
            return f"理想阈值 {gl:.0f}–{gh:.0f}cm"
    if unit in {"deg", "°", "degree"}:
        if gl is not None and gh is not None:
            return f"理想阈值 {gl:.0f}–{gh:.0f}°"
    if gl is not None and gh is not None:
        return f"理想阈值 {gl}–{gh}"
    return ""


def _rank_deduction_rows(diagnosis: Optional[dict] = None) -> list[dict]:
    """按扣分从高到低整理 deductions，并补齐测得值与阈值。"""
    diagnosis = diagnosis or {}
    detail = diagnosis.get("score_detail") or {}
    raw = detail.get("deductions") or diagnosis.get("deductions") or []
    if not isinstance(raw, list):
        raw = []
    indicators = _diagnosis_indicators(diagnosis)
    rows: list[dict] = []
    for row in raw:
        if isinstance(row, str) and row.strip():
            rows.append(
                {
                    "metric": None,
                    "penalty": 0.0,
                    "reason": row.strip(),
                    "measured_value": None,
                    "threshold": None,
                }
            )
            continue
        if not isinstance(row, dict):
            continue
        metric = row.get("metric") or row.get("key")
        pen = 0.0
        try:
            pen = float(row.get("penalty") or 0.0)
        except (TypeError, ValueError):
            pen = 0.0
        reason = str(row.get("reason") or "").strip()
        item = indicators.get(metric) if isinstance(metric, str) else None
        measured = None
        threshold = None
        if isinstance(item, dict):
            measured = _format_indicator_value_unit(str(metric), item) or None
            threshold = _format_green_band(item) or None
            if not reason:
                reason = _coach_fact_for_indicator(
                    str(metric),
                    item,
                    quoteable=is_aigc_measurable_provenance(item.get("provenance")),
                )
        rows.append(
            {
                "metric": metric,
                "penalty": pen,
                "reason": reason or "未命名扣分项",
                "measured_value": measured,
                "threshold": threshold,
            }
        )
    # 若 deductions 为空，从 indicators.penalty 构造
    if not rows:
        for key, item in indicators.items():
            if not isinstance(item, dict):
                continue
            pen = _indicator_penalty(item)
            if pen <= 0:
                continue
            rows.append(
                {
                    "metric": key,
                    "penalty": pen,
                    "reason": _coach_fact_for_indicator(
                        key,
                        item,
                        quoteable=is_aigc_measurable_provenance(item.get("provenance")),
                    ),
                    "measured_value": _format_indicator_value_unit(key, item) or None,
                    "threshold": _format_green_band(item) or None,
                }
            )
    rows.sort(key=lambda r: (-float(r.get("penalty") or 0.0), str(r.get("metric") or "")))
    return rows


def build_measurement_context(diagnosis: Optional[dict] = None) -> dict:
    """组装注入大模型的全量测量上下文（总分 / 雷达 / 主次扣分）。"""
    diagnosis = diagnosis or {}
    total = _extract_total_score(diagnosis)
    radar = _extract_radar_scores(diagnosis)
    rows = _rank_deduction_rows(diagnosis)
    primary = rows[0] if rows else None
    secondary = rows[1:] if len(rows) > 1 else []
    radar_readable = []
    for key, val in radar.items():
        label = _RADAR_LABEL_ZH.get(key, key)
        radar_readable.append(f"{label}({key})={val}")
    primary_text = "无显著扣分"
    if primary:
        bits = [str(primary.get("reason") or "主要偏差")]
        if primary.get("measured_value"):
            bits.append(f"测得 {primary['measured_value']}")
        if primary.get("threshold"):
            bits.append(str(primary["threshold"]))
        if primary.get("penalty"):
            bits.append(f"扣 {primary['penalty']} 分")
        primary_text = "；".join(bits)
    secondary_text = [
        (
            f"{s.get('reason')}"
            + (f"（测得 {s['measured_value']}）" if s.get("measured_value") else "")
            + (f"，{s['threshold']}" if s.get("threshold") else "")
            + (f"，扣 {s['penalty']} 分" if s.get("penalty") else "")
        )
        for s in secondary
    ]
    readable_lines = [
        f"总分: {total if total is not None else '暂缺'}",
        f"雷达图得分: {', '.join(radar_readable) if radar_readable else '暂缺'}",
        f"主要扣分病灶: {primary_text}",
        "次要扣分项: " + ("；".join(secondary_text) if secondary_text else "无"),
    ]
    return {
        "TotalScore": total,
        "radar_scores": radar,
        "radar_scores_readable": radar_readable,
        "primary_deduction": primary,
        "secondary_deductions": secondary,
        "deductions": rows,
        "readable_text": "\n".join(readable_lines),
    }


def _pick_max_penalty_defect(diagnosis: dict) -> str | None:
    """从扣分项中提取扣分最多的指标键；同分按 _RED_DEFECT_PRIORITY 决胜。"""
    indicators = _diagnosis_indicators(diagnosis)
    if not indicators:
        return None

    best_key: str | None = None
    best_pen = -1.0
    best_rank = 10**9
    priority_rank = {key: idx for idx, key in enumerate(_RED_DEFECT_PRIORITY)}

    for key, item in indicators.items():
        if not isinstance(item, dict):
            continue
        pen = _indicator_penalty(item)
        if pen <= 0:
            continue
        rank = priority_rank.get(key, 10**9)
        if pen > best_pen or (pen == best_pen and rank < best_rank):
            best_pen = pen
            best_rank = rank
            best_key = key
    return best_key


def _format_indicator_value_unit(key: str, item: dict) -> str:
    """把指标实测值格式化为简短可读片段（如 28.5cm / 0.72×肩宽 / 42.0°）。"""
    if key == "distance_cm":
        disp, unit, _est, _ratio = _distance_display_fields(item)
        if disp is None:
            return ""
        if unit == "×肩宽":
            return f"支撑脚横距比例 {disp:.2f}"
        if unit == "cm":
            return f"{disp:.1f}cm"
    value = item.get("value")
    if value is None and key == "ankle_rigidity":
        value = item.get("variance")
    if value is None:
        return ""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)

    unit = str(item.get("unit") or _INDICATOR_UNIT.get(key, ""))
    if unit == "variance":
        return f"σ² {round(num, 2)}"
    if unit in {"deg", "°"}:
        return f"{round(num, 1)}°"
    if unit in {"deg/s", "°/s"}:
        return f"{round(num, 1)}°/s"
    if unit == "cm":
        return f"{round(num, 1)}cm"
    if unit in {"ratio", "×肩宽"}:
        return f"{round(num, 2)}×肩宽"
    return f"{round(num, 2)}{unit}" if unit else f"{round(num, 2)}"


def _qualify_primary_error(key: str, item: dict) -> str:
    """根据 green_band / status 生成失误定性短语。"""
    green_band = item.get("green_band")
    value = item.get("value")
    if value is None and key == "ankle_rigidity":
        value = item.get("variance")

    if (
        key == "distance_cm"
        and value is not None
        and isinstance(green_band, (list, tuple))
        and len(green_band) >= 2
    ):
        try:
            v = float(value)
            gl, gh = green_band[0], green_band[1]
            if gh is not None and v > float(gh):
                return "距离过远"
            if gl is not None and v < float(gl):
                return "距离过近"
        except (TypeError, ValueError):
            pass

    status = str(item.get("status") or "")
    if status == "YELLOW_APPROACHING":
        base = _PRIMARY_ERROR_QUALIFIER.get(key, "接近警戒")
        return f"{base}（接近警戒）"
    return _PRIMARY_ERROR_QUALIFIER.get(key, "偏离过大")


def build_primary_error_description(diagnosis: Optional[dict] = None) -> str:
    """从 error_diagnoser 扣分项中提取「首要错误」自然语言描述。

    优先取 penalty 最大项；无有效扣分时回落红/黄缺陷优先级。
    例：支撑脚偏移 28.5cm，距离过远
    """
    diagnosis = diagnosis or {}
    key = _pick_max_penalty_defect(diagnosis)
    if not key:
        key = _pick_primary_defect_key(diagnosis)

    indicators = _diagnosis_indicators(diagnosis)
    item = indicators.get(key) if isinstance(indicators.get(key), dict) else {}
    label = _PRIMARY_ERROR_LABEL_ZH.get(key) or _INDICATOR_LABEL_ZH.get(key, key)
    value_part = _format_indicator_value_unit(key, item) if item else ""
    qualifier = _qualify_primary_error(key, item) if item else "需要调整"

    if value_part:
        return f"{label} {value_part}，{qualifier}"
    if item:
        return f"{label}，{qualifier}"
    return "主要动作偏差，需要调整"


def build_system_prompt(
    primary_error_description: str,
    measurement_context: Optional[dict] = None,
) -> str:
    """强语境注入：在 System Prompt 头部锁定总分、雷达优势与主病灶。"""
    desc = (primary_error_description or "").strip() or "主要动作偏差，需要调整"
    ctx = measurement_context or {}
    total = ctx.get("TotalScore")
    total_txt = f"{total}" if total is not None else "暂缺"
    radar_bits = ctx.get("radar_scores_readable") or []
    radar_txt = "、".join(radar_bits[:3]) if radar_bits else "暂缺"
    primary = ctx.get("primary_deduction") or {}
    primary_txt = str(primary.get("reason") or desc)
    if primary.get("measured_value"):
        primary_txt = f"{primary_txt}（测得 {primary['measured_value']}）"
    if primary.get("threshold"):
        primary_txt = f"{primary_txt}，{primary['threshold']}"
    core = (
        f"【核心指令】本次总分 {total_txt}；雷达亮点/短板：{radar_txt}；"
        f"最大失误病灶：{primary_txt}。"
        "你必须引用上述真实数据撰写四维报告，绝不能每次说一样的话！\n"
    )
    return core + SYSTEM_PROMPT


def _clamp_report_phrase(text: str, limit: int) -> str:
    """压成单段报告话术：去围栏、去章节标签，保留空格，按字数硬裁。"""
    cleaned = _strip_code_fences(text)
    cleaned = re.sub(r"^\s+", "", cleaned)
    cleaned = re.sub(r"[\t\r\n]+", " ", cleaned)
    cleaned = re.sub(r" +", " ", cleaned).strip()
    for prefix in (
        "【综合评价】",
        "【动力链病理分析】",
        "【具身隐喻处方】",
        "【下一步训练指令】",
        "【魔法指令】",
        "【闪光点发现】",
        "【依据】",
    ):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    if len(cleaned) > limit:
        cleaned = cleaned[:limit]
    return cleaned


def _depth_fallback_report(diagnosis: Optional[dict] = None) -> dict:
    """动态四维兜底：必须拼接 TotalScore 与最严重扣分项，禁止纯静态串。"""
    diagnosis = diagnosis or {}
    ctx = build_measurement_context(diagnosis)
    total = ctx.get("TotalScore")
    total_txt = f"{float(total):.1f}" if isinstance(total, (int, float)) else "暂缺"
    primary = ctx.get("primary_deduction") or {}
    reason = str(primary.get("reason") or "").strip()
    if not reason:
        deductions = ctx.get("deductions") or []
        if deductions and isinstance(deductions[0], dict):
            reason = str(deductions[0].get("reason") or "").strip()
    if not reason:
        reason = build_primary_error_description(diagnosis)
    measured = primary.get("measured_value")
    threshold = primary.get("threshold")
    measured_bit = f"，测得 {measured}" if measured else ""
    threshold_bit = f"（{threshold}）" if threshold else ""

    # 雷达优势：取最高 1~2 维做肯定开场
    radar = ctx.get("radar_scores") or {}
    top_radar = sorted(radar.items(), key=lambda kv: (-float(kv[1]), kv[0]))[:2]
    if top_radar:
        strength = "、".join(
            f"{_RADAR_LABEL_ZH.get(k, k)}{v}分" for k, v in top_radar
        )
    else:
        strength = "动作框架积极"

    defect_key = _pick_primary_defect_key(diagnosis) if diagnosis else "max_folding_angle"
    # 次要病灶：写入病理分析，避免不同学生只剩同一句脚踝模板
    secondary = ctx.get("secondary_deductions") or []
    secondary_bits: list[str] = []
    for item in secondary[:2]:
        if not isinstance(item, dict):
            continue
        s_reason = str(item.get("reason") or "").strip()
        s_measured = item.get("measured_value")
        if s_reason and s_measured:
            secondary_bits.append(f"{s_reason}（{s_measured}）")
        elif s_reason:
            secondary_bits.append(s_reason)
    secondary_text = "；同时伴有" + "、".join(secondary_bits) if secondary_bits else ""

    # 具身隐喻必须带实测值，禁止复读纯静态模板
    measured_for_magic = measured or ""
    magic_templates = {
        "ankle_rigidity": (
            f"你刚才触球时脚腕软绵绵的（测得 {measured_for_magic or '锁踝不足'}），"
            "就像煮熟的面条一样！下次试试把脚面绷紧成硬硬的铁板。"
        ),
        "distance_cm": (
            f"你刚才支撑脚站位偏了（测得 {measured_for_magic or '站位偏离'}），"
            "就像跨栏一样！下次试试把脚踩到离球大约一拳远。"
        ),
        "max_folding_angle": (
            f"你刚才摆动腿折叠不够（测得 {measured_for_magic or '折叠不足'}），"
            "就像一根冻住的冰棍！下次试试像弹簧一样弯一弯再弹出去。"
        ),
        "impact_knee_angle": (
            f"你刚才触球膝盖过直（测得 {measured_for_magic or '膝角失控'}），"
            "就像门板一样绷着！下次试试膝盖微屈再把球推出去。"
        ),
        "support_knee_angle": (
            f"你刚才支撑膝太直（测得 {measured_for_magic or '支撑不稳'}），"
            "就像插在地里的木棍！下次试试支撑膝微微弯一点站稳。"
        ),
        "hip_torsion_angle": (
            f"你刚才转髋不足（测得 {measured_for_magic or '转髋偏小'}），"
            "就像坐在太师椅上！下次试试胸口像大虾一样往前压。"
        ),
        "whipping_velocity": (
            f"你刚才随前不够（测得 {measured_for_magic or '鞭打偏弱'}），"
            "就像踩了急刹车！下次试试踢完再把脚往前送一送。"
        ),
        "toe_angle": (
            f"你刚才支撑脚尖指偏了（测得 {measured_for_magic or '脚尖偏离'}），"
            "就像跑偏的跨栏！下次试试脚尖对准球门再出脚。"
        ),
    }
    magic = magic_templates.get(defect_key) or (
        f"教练提醒：{reason}{measured_bit}，下次注意调整！"
    )

    reason_hint = reason[:18] if reason else "动作细节"
    overview = (
        f"本次综合评分 {total_txt} 分，{strength}表现突出；"
        f"需关注{reason_hint}。"
    )
    biomechanical = (
        f"核心病灶是「{reason}」{measured_bit}{threshold_bit}{secondary_text}。"
        "该原发性偏差会沿动力链向远端传导，影响膝盖伸展控制或脚踝刚度，需优先纠正。"
    )
    action = (
        f"针对{reason_hint}做 5 次慢动作定点控制"
        + (f"（目标靠近 {threshold}）" if threshold else "。")
    )
    report = {
        "overview": _clamp_report_phrase(overview, _OVERVIEW_MAX_CHARS + 30) or overview,
        "biomechanical_analysis": _clamp_report_phrase(biomechanical, _BIOMECH_MAX_CHARS + 60)
        or biomechanical,
        "magic_metaphor": _clamp_report_phrase(magic, 80) or magic,
        "action_plan": _clamp_report_phrase(action, _ACTION_MAX_CHARS + 25) or action,
        "aigc_source": "fallback",
    }
    return _attach_legacy_aliases(report)


def _attach_legacy_aliases(report: dict) -> dict:
    """四维字段 ↔ 旧 clinical_echo / correction_metaphor / praise 别名。"""
    out = dict(report or {})
    overview = out.get("overview") or out.get("clinical_echo") or ""
    biomech = out.get("biomechanical_analysis") or ""
    magic = out.get("magic_metaphor") or out.get("correction_metaphor") or ""
    action = out.get("action_plan") or out.get("praise_encouragement") or ""
    out["overview"] = overview
    out["biomechanical_analysis"] = biomech
    out["magic_metaphor"] = magic
    out["action_plan"] = action
    out["clinical_echo"] = overview
    out["correction_metaphor"] = magic
    out["praise_encouragement"] = action
    return out


def _hard_fallback_dual(primary_error_description: str, diagnosis: Optional[dict] = None) -> dict:
    """解析失败 / 重复废话时的硬性兜底：动态拼接总分与最严重扣分。"""
    diagnosis = diagnosis or {}
    if not (diagnosis.get("score_detail") or diagnosis.get("indicators") or diagnosis.get("deductions")):
        # 仅有描述字符串时，最小动态拼装
        desc = (primary_error_description or "").strip() or "主要动作偏差，需要调整"
        report = {
            "overview": f"本次动作完成量化诊断，需优先关注：{desc[:18]}。",
            "biomechanical_analysis": (
                f"最严重扣分项为「{desc}」。该原发性偏差会影响后续动力链传导，需优先纠正。"
            ),
            "magic_metaphor": f"教练提醒：{desc}，下次注意调整！",
            "action_plan": "针对核心病灶做分解慢练。",
            "aigc_source": "fallback",
        }
        return _attach_legacy_aliases(report)
    # 把描述写回，确保无 deductions 时仍有病灶文本
    if primary_error_description and not _rank_deduction_rows(diagnosis):
        diagnosis = {
            **diagnosis,
            "score_detail": {
                **(diagnosis.get("score_detail") or {}),
                "deductions": [
                    {"reason": primary_error_description, "penalty": 1.0}
                ],
            },
        }
    return _depth_fallback_report(diagnosis)


def _pick_primary_red_defect(diagnosis: dict) -> str | None:
    """优先取扣分最多项；否则按固定优先级挑选唯一红色缺陷键名。"""
    max_pen_key = _pick_max_penalty_defect(diagnosis)
    if max_pen_key:
        return max_pen_key

    indicators = _diagnosis_indicators(diagnosis)
    if indicators:
        for key in _RED_DEFECT_PRIORITY:
            item = indicators.get(key) or {}
            if isinstance(item, dict) and item.get("status") == "RED_DEVIATED":
                return key
    # 兼容旧错误码
    code = diagnosis.get("primary_error_code") or ""
    code_map = {
        "ERR_C1_LOOSE_ANKLE": "ankle_rigidity",
        "ERR_ANKLE_LOOSE": "ankle_rigidity",
        "ERR_A2_SUPPORT_WIDE": "distance_cm",
        "ERR_SUPPORT_TOO_CLOSE": "distance_cm",
        "ERR_WARMUP_CLOSE": "distance_cm",
        "ERR_A1_SUPPORT_BACK": "toe_angle",
        "ERR_B1_STRAIGHT_LEG": "max_folding_angle",
        "ERR_B2_SHANK_ONLY": "max_folding_angle",
        "ERR_KNEE_STIFF": "support_knee_angle",
        "ERR_D1_TRUNK_LEAN": "trunk_lean_angle",
        "ERR_TORSO_TILT": "trunk_lean_angle",
    }
    return code_map.get(code)


# 三个「焦点指标」由 pack_focus_indicator 构造，AIGC 抗幻觉合同最严格：
# 非 measured/calibrated 时必须完全省略 value 键，并标注 note="未提供实测值"。
_FOCUS_INDICATOR_KEYS: frozenset[str] = frozenset(
    {"distance_cm", "max_folding_angle", "ankle_rigidity"}
)


def _extract_indicator_payload(diagnosis: dict) -> dict:
    """提取指标 status + value（供隐喻转译），剥离评分/扣分字段。

    【AIGC 抗幻觉合同】
    · 焦点指标（_FOCUS_INDICATOR_KEYS）非实测 provenance → 完全省略 "value" 键，
      添加 note="未提供实测值"，measured=False。
    · metrics 回填（backfill）仅在 provenance 为实测/校准时允许，
      以免把 default/scoring 数值悄悄注入 AIGC。
    """
    detail = diagnosis.get("score_detail") or {}
    indicators = detail.get("indicators") or diagnosis.get("indicators") or {}
    metrics = diagnosis.get("metrics") or {}
    out: dict = {}
    if not isinstance(indicators, dict):
        return out
    for key, item in indicators.items():
        if not isinstance(item, dict):
            continue
        provenance = str(item.get("provenance") or "").strip().lower() or "unknown"
        measurable = is_aigc_measurable_provenance(provenance)

        value = item.get("value")
        if value is None and key == "ankle_rigidity":
            value = item.get("variance")
        # metrics 回填：仅在实测/校准 provenance 下允许，防止 default 数值流入 AIGC
        if measurable:
            if value is None and key in metrics:
                value = metrics.get(key)
            if value is None and key == "distance_cm":
                value = metrics.get("support_lateral_dist_cm")

        entry = {
            "label_zh": _INDICATOR_LABEL_ZH.get(key, key),
            "status": item.get("status"),
            "unit": _INDICATOR_UNIT.get(key, ""),
            "provenance": provenance,
        }

        if key in _FOCUS_INDICATOR_KEYS and not measurable:
            # 焦点指标 + 非实测：完全省略 value，添加 note 标记
            entry["measured"] = False
            entry["note"] = "未提供实测值"
        elif value is not None:
            try:
                entry["value"] = round(float(value), 2)
                entry["measured"] = measurable
            except (TypeError, ValueError):
                entry["value"] = value
                entry["measured"] = measurable
        else:
            entry["measured"] = False

        # stiffness_status 仅在实测 ankle_rigidity 且字段非空时透出
        if key == "ankle_rigidity" and measurable and item.get("stiffness_status"):
            entry["stiffness_status"] = item.get("stiffness_status")
        if item.get("method"):
            entry["method"] = item.get("method")
        # 肩宽比支撑距：透出 ratio / 估计 cm，供 Brief 与模型对齐
        if key == "distance_cm" and measurable:
            if item.get("support_ratio") is not None:
                try:
                    entry["support_ratio"] = round(float(item["support_ratio"]), 4)
                except (TypeError, ValueError):
                    pass
            if item.get("distance_cm_estimate") is not None:
                try:
                    entry["distance_cm_estimate"] = round(
                        float(item["distance_cm_estimate"]), 2
                    )
                except (TypeError, ValueError):
                    pass
            if str(item.get("unit") or "") in {"ratio", "×肩宽"}:
                entry["unit"] = "×肩宽"
        out[key] = entry
    return out


def _strip_code_fences(text: str) -> str:
    """剔除模型偶发包裹的 Markdown 代码围栏。"""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json|markdown|md)?\s*", "", cleaned, flags=re.IGNORECASE
        )
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _clamp_optimal_phrase(text: str, limit: int = _PRAISE_MAX_CHARS) -> str:
    """压成单句童趣话术：去围栏、去标签前缀，按字段硬裁字数。"""
    cleaned = _strip_code_fences(text)
    cleaned = re.sub(r"^[\s\-•*·、。]+", "", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned.replace("\n", ""))
    for prefix in ("【魔法指令】", "【闪光点发现】"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    if len(cleaned) > limit:
        cleaned = cleaned[:limit]
    return cleaned


def _pick_primary_defect_key(diagnosis: dict) -> str:
    """扣分最多优先；否则红优先、黄次之；都没有则回落到折叠角。"""
    defect = _pick_max_penalty_defect(diagnosis) or _pick_primary_red_defect(diagnosis)
    if not defect:
        indicators = _diagnosis_indicators(diagnosis)
        if indicators:
            for key in _RED_DEFECT_PRIORITY:
                item = indicators.get(key) or {}
                if isinstance(item, dict) and item.get("status") == "YELLOW_APPROACHING":
                    defect = key
                    break
    return defect or "max_folding_angle"


def _optimal_fallback_dual(diagnosis: Optional[dict] = None) -> dict:
    """按测量上下文生成四维动态兜底；无诊断时走静态底稿。"""
    brief = build_clinical_brief(diagnosis)
    if not diagnosis:
        base = _attach_legacy_aliases(dict(_STATIC_OPTIMAL_FALLBACK))
        base["clinical_echo"] = _clinical_echo_from_brief(brief) or base["overview"]
        base["overview"] = base["clinical_echo"]
        base["aigc_source"] = "fallback"
        base["clinical_brief"] = brief
        return base
    report = _depth_fallback_report(diagnosis)
    report["clinical_brief"] = brief
    return report


def _extract_json_object(text: str) -> Optional[dict]:
    """从模型原文严谨 json.loads 提取 JSON 对象；失败返回 None。"""
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    # 次级兜底：截取首个大括号对象后再 json.loads（仍拒绝非 JSON）
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _is_duplicate_or_filler(correction: str, praise: str) -> bool:
    """阻断纠错句与表扬句重复/同质废话。"""
    c = re.sub(r"\s+", "", (correction or "").strip())
    p = re.sub(r"\s+", "", (praise or "").strip())
    if not c or not p:
        return True
    if c == p:
        return True
    # 一方被另一方完全包含，也视为重复废话
    if c in p or p in c:
        return True
    return False


def _parse_optimal_dual_feedback(
    raw_text: str, diagnosis: Optional[dict] = None
) -> dict:
    """严谨解析四维诊断 JSON；兼容旧三字段；失败时走动态数据兜底。"""
    primary_error = build_primary_error_description(diagnosis)
    brief = build_clinical_brief(diagnosis)
    try:
        parsed = _extract_json_object(raw_text)
        if not parsed:
            raise ValueError("LLM 未返回可解析 JSON")

        overview = _clamp_report_phrase(
            str(parsed.get("overview") or parsed.get("clinical_echo") or ""),
            limit=_OVERVIEW_MAX_CHARS,
        )
        biomech = _clamp_report_phrase(
            str(parsed.get("biomechanical_analysis") or ""),
            limit=_BIOMECH_MAX_CHARS,
        )
        magic = _clamp_report_phrase(
            str(
                parsed.get("magic_metaphor")
                or parsed.get("correction_metaphor")
                or ""
            ),
            limit=_CORRECTION_MAX_CHARS,
        )
        action = _clamp_report_phrase(
            str(
                parsed.get("action_plan")
                or parsed.get("praise_encouragement")
                or ""
            ),
            limit=_ACTION_MAX_CHARS + 15,
        )

        # 旧三字段兼容：缺 overview/biomech 时用 clinical_echo 补
        if not overview:
            overview = _clinical_echo_from_brief(brief)
        if not biomech and overview:
            # 旧模型未给病理分析时，用测量上下文补一段动态说明
            ctx = build_measurement_context(diagnosis)
            primary = ctx.get("primary_deduction") or {}
            reason = str(primary.get("reason") or primary_error)
            measured = primary.get("measured_value")
            measured_bit = f"，测得 {measured}" if measured else ""
            biomech = (
                f"核心病灶是「{reason}」{measured_bit}。"
                "该偏差会沿动力链影响后续环节稳定性。"
            )
            biomech = _clamp_report_phrase(biomech, _BIOMECH_MAX_CHARS)

        if not magic or not action:
            raise ValueError("四维报告关键字段为空")

        # 字段互不重复校验（overview/magic/action 两两去重）
        raw_magic = str(
            parsed.get("magic_metaphor") or parsed.get("correction_metaphor") or ""
        ).strip()
        raw_action = str(
            parsed.get("action_plan") or parsed.get("praise_encouragement") or ""
        ).strip()
        if raw_magic == raw_action or _is_duplicate_or_filler(magic, action):
            raise ValueError("AI生成了重复内容")
        if overview and magic and (
            overview == magic or _is_duplicate_or_filler(overview, magic)
        ):
            raise ValueError("AI生成了重复内容")

        report = _attach_legacy_aliases(
            {
                "overview": overview,
                "biomechanical_analysis": biomech,
                "magic_metaphor": magic,
                "action_plan": action,
                "aigc_source": "llm",
                "clinical_brief": brief,
            }
        )
        return report
    except Exception as exc:  # noqa: BLE001
        print(f"【llm_agent】四维 JSON 解析/校验失败，启用动态数据兜底：{exc}")
        hard = _hard_fallback_dual(primary_error, diagnosis)
        hard["clinical_brief"] = brief
        return hard


def _format_optimal_dual_text(dual: dict) -> str:
    """把四维字典拼成字幕/日志可读字符串（兼容旧 str 调用方）。"""
    overview = (dual or {}).get("overview") or (dual or {}).get("clinical_echo") or ""
    biomech = (dual or {}).get("biomechanical_analysis") or ""
    magic = (dual or {}).get("magic_metaphor") or (dual or {}).get(
        "correction_metaphor"
    ) or _STATIC_OPTIMAL_FALLBACK["magic_metaphor"]
    action = (dual or {}).get("action_plan") or (dual or {}).get(
        "praise_encouragement"
    ) or _STATIC_OPTIMAL_FALLBACK["action_plan"]
    parts = []
    if overview:
        parts.append(f"【综合评价】{overview}")
    if biomech:
        parts.append(f"【动力链病理分析】{biomech}")
    parts.append(f"【具身隐喻处方】{magic}")
    parts.append(f"【下一步训练指令】{action}")
    # 兼容旧字幕关键字
    parts.append(f"【魔法指令】{magic}")
    parts.append(f"【闪光点发现】{action}")
    return " ".join(parts)


def _dual_to_report_fields(dual: dict, diagnosis: Optional[dict] = None) -> dict:
    """映射到前端字段：四维主字段 + 旧 painPoint / prescription 别名。"""
    brief = (dual or {}).get("clinical_brief") or build_clinical_brief(diagnosis)
    dual = _attach_legacy_aliases(dual or {})
    overview = dual.get("overview") or _clinical_echo_from_brief(brief)
    biomech = dual.get("biomechanical_analysis") or ""
    magic = dual.get("magic_metaphor") or _STATIC_OPTIMAL_FALLBACK["magic_metaphor"]
    action = dual.get("action_plan") or _STATIC_OPTIMAL_FALLBACK["action_plan"]
    source = dual.get("aigc_source") or "fallback"
    return {
        # 四维主字段
        "overview": overview,
        "biomechanical_analysis": biomech,
        "magic_metaphor": magic,
        "action_plan": action,
        # 前端既有映射：痛点=病理分析（空则回落隐喻）；处方=训练指令
        "painPoint": biomech or magic,
        "prescription": action,
        "correction_metaphor": magic,
        "praise_encouragement": action,
        "clinical_echo": overview,
        "clinicalEcho": overview,
        "aigc_source": source,
        "aigcSource": source,
        "clinical_brief": brief,
        "clinicalBrief": brief,
    }


def _split_clinical_markdown(text: str) -> tuple[str, str]:
    """兼容旧拆分；优先解析 OPTIMAL JSON，否则整段当作纠错话术。"""
    parsed = _extract_json_object(text)
    if parsed:
        correction = _clamp_optimal_phrase(
            str(parsed.get("correction_metaphor") or ""),
            limit=_CORRECTION_MAX_CHARS,
        )
        praise = _clamp_optimal_phrase(
            str(parsed.get("praise_encouragement") or ""),
            limit=_PRAISE_MAX_CHARS,
        )
        if correction and praise:
            return correction, praise
    cleaned = _strip_code_fences(text)
    match = re.search(
        r"【三[、．.\s]*(?:临床纠正药方|闪光点发现)】", cleaned
    )
    if match:
        return cleaned[: match.start()].strip(), cleaned[match.start() :].strip()
    phrase = _clamp_optimal_phrase(
        cleaned, limit=_CORRECTION_MAX_CHARS
    ) or _STATIC_OPTIMAL_FALLBACK["correction_metaphor"]
    return phrase, phrase


def _clamp_metaphor_phrase(text: str) -> str:
    """兼容旧接口：压成单句纠错隐喻（≤ correction 硬上限）。"""
    cleaned = _clamp_optimal_phrase(text, limit=_CORRECTION_MAX_CHARS)
    return cleaned or _METAPHOR_FALLBACK_BY_DEFECT["max_folding_angle"]


def _pick_metaphor_fallback(diagnosis: dict) -> str:
    """按主缺陷挑选纠错隐喻兜底。"""
    return _optimal_fallback_dual(diagnosis)["correction_metaphor"]


def _build_clinical_fallback_markdown(diagnosis: dict) -> str:
    """实测值复述清单 + OPTIMAL 双段兜底（供 provenance E2E 测试与调试复盘）。

    【一、 客观实测面诊】—— 仅引用 measured/calibrated 数值；
                          焦点指标非实测则写"未提供实测值"，绝不引用 scoring_value。
    【二、 魔法指令】   —— correction_metaphor
    【三、 闪光点发现】 —— praise_encouragement
    """
    diagnosis = diagnosis or {}
    detail = diagnosis.get("score_detail") or {}
    indicators = detail.get("indicators") or diagnosis.get("indicators") or {}

    # ── 第一节：实测数据清单 ──────────────────────────────────────────────────
    lines_sec1: list[str] = []
    if isinstance(indicators, dict):
        for key in (
            "distance_cm",
            "max_folding_angle",
            "ankle_rigidity",
            "impact_knee_angle",
            "support_knee_angle",
            "hip_torsion_angle",
            "whipping_velocity",
            "toe_angle",
        ):
            item = indicators.get(key)
            if not isinstance(item, dict):
                continue
            label = _INDICATOR_LABEL_ZH.get(key, key)
            provenance = str(item.get("provenance") or "").strip().lower() or "unknown"
            measurable = is_aigc_measurable_provenance(provenance)
            unit = _INDICATOR_UNIT.get(key, "")

            if measurable:
                raw = item.get("value")
                if raw is None and key == "ankle_rigidity":
                    raw = item.get("variance")
                if raw is not None:
                    try:
                        val = round(float(raw), 2)
                        display = (
                            f"σ² {val}"
                            if unit == "variance"
                            else (f"{val}°" if unit in {"deg", "°"} else f"{val}{unit}")
                        )
                    except (TypeError, ValueError):
                        display = str(raw)
                    lines_sec1.append(f"  · {label}：{display}")
                else:
                    lines_sec1.append(f"  · {label}：未提供实测值")
            elif key in _FOCUS_INDICATOR_KEYS:
                # 焦点指标非实测：明确标注，绝不引用 scoring_value
                lines_sec1.append(f"  · {label}：未提供实测值")

    sec1_body = "\n".join(lines_sec1) if lines_sec1 else "  · 本次未采集到实测数据"
    dual = _optimal_fallback_dual(diagnosis)

    return (
        f"【一、 客观实测面诊】\n{sec1_body}\n\n"
        f"【二、 魔法指令】\n{dual['correction_metaphor']}\n\n"
        f"【三、 闪光点发现】\n{dual['praise_encouragement']}"
    )


def _safe_round_float(value: Any, ndigits: int = 2) -> Optional[float]:
    """安全转 float 并 round；非法返回 None。"""
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not np_isfinite_safe(num):
        return None
    return round(num, ndigits)


def _indicator_chain_value(
    key: str,
    item: dict,
    *,
    metrics: Optional[dict] = None,
    metric_keys: tuple[str, ...] = (),
) -> tuple[Optional[float], str, bool]:
    """读取动力链指标实测值。

    焦点指标（distance / fold / ankle）非 measured/calibrated 时禁止回填，
    返回 (None, provenance, measured=False)。
    非焦点指标允许从 metrics 补齐（若 indicator 缺值）。
    """
    item = item if isinstance(item, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    provenance = str(item.get("provenance") or "").strip().lower() or "unknown"
    measurable = is_aigc_measurable_provenance(provenance)
    focus = key in _FOCUS_INDICATOR_KEYS

    value = item.get("value")
    if value is None and key == "ankle_rigidity":
        value = item.get("variance")
    if value is None and key == "distance_cm":
        value = item.get("support_ratio")

    if measurable:
        if value is None:
            for mk in metric_keys:
                if metrics.get(mk) is not None:
                    value = metrics.get(mk)
                    break
        num = _safe_round_float(value, 4 if key == "distance_cm" else 2)
        return num, provenance, True

    if focus:
        return None, provenance, False

    # 非焦点：indicator 无 provenance 时仍可从 metrics 透出（供时序推演）
    if value is None:
        for mk in metric_keys:
            if metrics.get(mk) is not None:
                value = metrics.get(mk)
                provenance = "metrics"
                break
    num = _safe_round_float(value, 2)
    return num, provenance, num is not None


def _format_standard_band(
    low: Optional[float],
    high: Optional[float],
    *,
    unit: str = "",
    digits: int = 1,
) -> str:
    """把标准区间格式化为『标准 a–b』片段。"""
    if low is None and high is None:
        return ""
    try:
        if low is not None and high is not None:
            if unit in {"ratio", "×肩宽"}:
                return f"标准 {float(low):.1f}-{float(high):.1f}"
            if unit in {"deg", "°"}:
                return f"标准 {float(low):.0f}-{float(high):.0f}度"
            return f"标准 {round(float(low), digits)}-{round(float(high), digits)}{unit}"
        if low is not None:
            if unit in {"deg", "°"}:
                return f"标准 ≥{float(low):.0f}度"
            return f"标准 ≥{round(float(low), digits)}{unit}"
        if high is not None:
            if unit in {"deg", "°"}:
                return f"标准 ≤{float(high):.0f}度"
            return f"标准 ≤{round(float(high), digits)}{unit}"
    except (TypeError, ValueError):
        return ""
    return ""


def _green_band_pair(item: dict) -> tuple[Optional[float], Optional[float]]:
    green = item.get("green_band") if isinstance(item, dict) else None
    if not isinstance(green, (list, tuple)) or len(green) < 2:
        return None, None
    try:
        gl = float(green[0]) if green[0] is not None else None
    except (TypeError, ValueError):
        gl = None
    try:
        gh = float(green[1]) if green[1] is not None else None
    except (TypeError, ValueError):
        gh = None
    return gl, gh


def _radar_extreme_dims(radar: dict) -> dict:
    """挑出雷达图得分极高 / 极低维度（各最多 2 项）。"""
    scored: list[tuple[str, float]] = []
    for key, val in (radar or {}).items():
        try:
            scored.append((str(key), float(val)))
        except (TypeError, ValueError):
            continue
    if not scored:
        return {"high": [], "low": [], "readable_high": "暂缺", "readable_low": "暂缺"}
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    highs = scored[:2]
    lows = sorted(scored, key=lambda kv: (kv[1], kv[0]))[:2]

    def _fmt(rows: list[tuple[str, float]]) -> str:
        return "、".join(
            f"{_RADAR_LABEL_ZH.get(k, k)}={v:g}" for k, v in rows
        )

    return {
        "high": [{"key": k, "label_zh": _RADAR_LABEL_ZH.get(k, k), "score": v} for k, v in highs],
        "low": [{"key": k, "label_zh": _RADAR_LABEL_ZH.get(k, k), "score": v} for k, v in lows],
        "readable_high": _fmt(highs),
        "readable_low": _fmt(lows),
    }


def _detect_early_deceleration(diagnosis: dict, metrics: dict) -> Optional[bool]:
    """读取 EARLY_DECELERATION 判定；无数据返回 None。"""
    if "early_deceleration" in metrics:
        return bool(metrics.get("early_deceleration"))
    code = str(diagnosis.get("primary_error_code") or "")
    if code == "ERR_EARLY_DECELERATION":
        return True
    codes = diagnosis.get("error_codes") or []
    if isinstance(codes, list) and "ERR_EARLY_DECELERATION" in codes:
        return True
    return None


def build_kinematic_chain_data(diagnosis: Optional[dict] = None) -> dict:
    """组装动力链时序全景数据：支撑 → 蓄力 → 触球 → 动量。

    仅拼入已有实测/可溯源字段；缺测项标记 available=False，禁止编造。
    """
    diagnosis = diagnosis if isinstance(diagnosis, dict) else {}
    indicators = _diagnosis_indicators(diagnosis)
    metrics = diagnosis.get("metrics") if isinstance(diagnosis.get("metrics"), dict) else {}

    dist_item = indicators.get("distance_cm") if isinstance(indicators.get("distance_cm"), dict) else {}
    fold_item = (
        indicators.get("max_folding_angle")
        if isinstance(indicators.get("max_folding_angle"), dict)
        else {}
    )
    ankle_item = (
        indicators.get("ankle_rigidity")
        if isinstance(indicators.get("ankle_rigidity"), dict)
        else {}
    )
    knee_item = (
        indicators.get("impact_knee_angle")
        if isinstance(indicators.get("impact_knee_angle"), dict)
        else {}
    )
    trunk_item = (
        indicators.get("trunk_lean_angle")
        if isinstance(indicators.get("trunk_lean_angle"), dict)
        else {}
    )
    whip_item = (
        indicators.get("whipping_velocity")
        if isinstance(indicators.get("whipping_velocity"), dict)
        else {}
    )

    # 【近端与支撑】支撑脚横距比例 + 躯干前倾
    support_ratio = None
    dist_prov = str(dist_item.get("provenance") or "").strip().lower() or "unknown"
    dist_measurable = is_aigc_measurable_provenance(dist_prov)
    if dist_measurable:
        disp, unit, _est, ratio = _distance_display_fields(dist_item)
        if ratio is not None:
            support_ratio = float(ratio)
        elif unit == "×肩宽" and disp is not None:
            support_ratio = float(disp)
        elif metrics.get("support_ratio") is not None:
            support_ratio = _safe_round_float(metrics.get("support_ratio"), 4)
    ratio_gl, ratio_gh = _green_band_pair(dist_item)
    if ratio_gl is None:
        ratio_gl, ratio_gh = 0.4, 0.7

    trunk_val, trunk_prov, trunk_ok = _indicator_chain_value(
        "trunk_lean_angle",
        trunk_item,
        metrics=metrics,
        metric_keys=("trunk_lean_angle", "trunk_lean_t0_deg"),
    )
    trunk_gl, trunk_gh = _green_band_pair(trunk_item)

    proximal = {
        "phase": "近端与支撑",
        "support_foot_ratio": {
            "value": support_ratio,
            "standard": "0.4-0.7",
            "standard_low": ratio_gl,
            "standard_high": ratio_gh,
            "unit": "×肩宽",
            "available": support_ratio is not None,
            "provenance": dist_prov if support_ratio is not None else "missing",
        },
        "trunk_lean_angle": {
            "value": trunk_val,
            "label": "TRUNK_LEAN",
            "unit": "°",
            "standard_low": trunk_gl,
            "standard_high": trunk_gh,
            "available": bool(trunk_ok and trunk_val is not None),
            "provenance": trunk_prov,
            "status": trunk_item.get("status") or metrics.get("trunk_lean_status"),
        },
    }

    # 【蓄力与折叠】后摆最大折叠角（标准 70-110°；有 green_band 则优先）
    fold_val, fold_prov, fold_ok = _indicator_chain_value(
        "max_folding_angle",
        fold_item,
        metrics=metrics,
        metric_keys=("max_folding_angle", "swing_fold_angle"),
    )
    fold_gl, fold_gh = _green_band_pair(fold_item)
    if fold_gl is None:
        fold_gl = 70.0
    if fold_gh is None:
        fold_gh = 110.0
    loading = {
        "phase": "蓄力与折叠",
        "max_folding_angle": {
            "value": fold_val,
            "standard": "70-110度",
            "standard_low": fold_gl,
            "standard_high": fold_gh,
            "unit": "°",
            "available": bool(fold_ok and fold_val is not None),
            "provenance": fold_prov,
        },
    }

    # 【释放与触球】触球膝角 + 脚踝形变落差
    knee_val, knee_prov, knee_ok = _indicator_chain_value(
        "impact_knee_angle",
        knee_item,
        metrics=metrics,
        metric_keys=("impact_knee_angle",),
    )
    knee_gl, knee_gh = _green_band_pair(knee_item)
    ankle_val, ankle_prov, ankle_ok = _indicator_chain_value(
        "ankle_rigidity",
        ankle_item,
        metrics=metrics,
        metric_keys=("ankle_deflection_deg", "ankle_dorsiflex_drop_deg", "ankle_variance"),
    )
    ankle_gl, ankle_gh = _green_band_pair(ankle_item)
    if ankle_gh is None:
        ankle_gh = 10.0
    release = {
        "phase": "释放与触球",
        "impact_knee_angle": {
            "value": knee_val,
            "unit": "°",
            "standard_low": knee_gl,
            "standard_high": knee_gh,
            "available": bool(knee_ok and knee_val is not None),
            "provenance": knee_prov,
        },
        "ankle_deflection": {
            "value": ankle_val,
            "label": "Ankle Deflection",
            "unit": "°",
            "standard_high": ankle_gh,
            "available": bool(ankle_ok and ankle_val is not None),
            "provenance": ankle_prov,
            "stiffness_status": ankle_item.get("stiffness_status"),
        },
    }

    # 【动量表现】挥腿角速度 + 提前减速
    whip_val, whip_prov, whip_ok = _indicator_chain_value(
        "whipping_velocity",
        whip_item,
        metrics=metrics,
        metric_keys=("whipping_speed_peak", "whipping_velocity"),
    )
    whip_gl, _whip_gh = _green_band_pair(whip_item)
    early = _detect_early_deceleration(diagnosis, metrics)
    momentum = {
        "phase": "动量表现",
        "whipping_velocity": {
            "value": whip_val,
            "label": "Whipping Velocity",
            "unit": "°/s",
            "standard_low": whip_gl,
            "available": bool(whip_ok and whip_val is not None),
            "provenance": whip_prov,
        },
        "early_deceleration": {
            "present": early,
            "label": "EARLY_DECELERATION",
            "available": early is not None,
            "code": "ERR_EARLY_DECELERATION" if early else None,
        },
    }

    return {
        "proximal_support": proximal,
        "loading_fold": loading,
        "release_impact": release,
        "momentum": momentum,
    }


def _format_kinematic_chain_text(chain: dict) -> str:
    """把动力链全景 dict 格式化为时序可读文本（支撑→触球）。"""
    chain = chain or {}
    proximal = chain.get("proximal_support") or {}
    loading = chain.get("loading_fold") or {}
    release = chain.get("release_impact") or {}
    momentum = chain.get("momentum") or {}

    lines: list[str] = []

    # 近端与支撑
    ratio = (proximal.get("support_foot_ratio") or {})
    trunk = (proximal.get("trunk_lean_angle") or {})
    prox_bits: list[str] = []
    if ratio.get("available") and ratio.get("value") is not None:
        std = _format_standard_band(
            ratio.get("standard_low"),
            ratio.get("standard_high"),
            unit="ratio",
        ) or "标准 0.4-0.7"
        prox_bits.append(f"支撑脚横距比例 {float(ratio['value']):.2f}（{std}）")
    else:
        prox_bits.append("支撑脚横距比例：未提供实测值")
    if trunk.get("available") and trunk.get("value") is not None:
        tstd = _format_standard_band(
            trunk.get("standard_low"),
            trunk.get("standard_high"),
            unit="°",
        )
        tbit = f"躯干前倾角(TRUNK_LEAN) {float(trunk['value']):.1f}°"
        if tstd:
            tbit += f"（{tstd}）"
        prox_bits.append(tbit)
    elif trunk.get("value") is None:
        prox_bits.append("躯干前倾角(TRUNK_LEAN)：未测")
    lines.append("【近端与支撑】" + "；".join(prox_bits))

    # 蓄力与折叠
    fold = (loading.get("max_folding_angle") or {})
    if fold.get("available") and fold.get("value") is not None:
        fstd = _format_standard_band(
            fold.get("standard_low"),
            fold.get("standard_high"),
            unit="°",
        ) or "标准 70-110度"
        lines.append(
            f"【蓄力与折叠】后摆最大折叠角 {float(fold['value']):.1f}°（实测值 vs {fstd}）"
        )
    else:
        lines.append("【蓄力与折叠】后摆最大折叠角：未提供实测值（标准 70-110度）")

    # 释放与触球
    knee = (release.get("impact_knee_angle") or {})
    ankle = (release.get("ankle_deflection") or {})
    rel_bits: list[str] = []
    if knee.get("available") and knee.get("value") is not None:
        kstd = _format_standard_band(
            knee.get("standard_low"),
            knee.get("standard_high"),
            unit="°",
        )
        kbit = f"触球膝角 {float(knee['value']):.1f}°"
        if kstd:
            kbit += f"（{kstd}）"
        rel_bits.append(kbit)
    else:
        rel_bits.append("触球膝角：未测")
    if ankle.get("available") and ankle.get("value") is not None:
        astd = _format_standard_band(None, ankle.get("standard_high"), unit="°")
        abit = f"脚踝形变落差(Ankle Deflection) {float(ankle['value']):.1f}°"
        if astd:
            abit += f"（{astd}）"
        if ankle.get("stiffness_status"):
            abit += f"，刚度={ankle['stiffness_status']}"
        rel_bits.append(abit)
    else:
        rel_bits.append("脚踝形变落差(Ankle Deflection)：未提供实测值")
    lines.append("【释放与触球】" + "；".join(rel_bits))

    # 动量表现
    whip = (momentum.get("whipping_velocity") or {})
    early = (momentum.get("early_deceleration") or {})
    mom_bits: list[str] = []
    if whip.get("available") and whip.get("value") is not None:
        wstd = _format_standard_band(whip.get("standard_low"), None, unit="°/s", digits=0)
        wbit = f"挥腿角速度(Whipping Velocity) {float(whip['value']):.1f}°/s"
        if wstd:
            wbit += f"（{wstd}）"
        mom_bits.append(wbit)
    else:
        mom_bits.append("挥腿角速度(Whipping Velocity)：未测")
    if early.get("available"):
        flag = "是" if early.get("present") else "否"
        mom_bits.append(f"是否存在提前减速(EARLY_DECELERATION)：{flag}")
    else:
        mom_bits.append("是否存在提前减速(EARLY_DECELERATION)：未判定")
    lines.append("【动量表现】" + "；".join(mom_bits))

    return "\n".join(lines)


def build_panorama_checkup_text(diagnosis: Optional[dict] = None) -> str:
    """组装「全景体检表」可读文本：基础信息 + 动力链时序 + 原发病灶。"""
    diagnosis = diagnosis if isinstance(diagnosis, dict) else {}
    measurement = build_measurement_context(diagnosis)
    brief = build_clinical_brief(diagnosis)
    chain = build_kinematic_chain_data(diagnosis)
    chain_text = _format_kinematic_chain_text(chain)

    total = measurement.get("TotalScore")
    total_txt = f"{total}" if total is not None else "暂缺"
    extremes = _radar_extreme_dims(measurement.get("radar_scores") or {})

    primary_deduction = measurement.get("primary_deduction") or {}
    primary_fact = ((brief.get("primary") or {}).get("coach_fact") or "").strip()
    primary_desc = build_primary_error_description(diagnosis)
    primary_reason = str(primary_deduction.get("reason") or primary_fact or primary_desc)
    primary_metric = primary_deduction.get("metric") or (brief.get("primary") or {}).get("metric")
    primary_penalty = primary_deduction.get("penalty")
    primary_measured = primary_deduction.get("measured_value")
    primary_threshold = primary_deduction.get("threshold")
    primary_code = diagnosis.get("primary_error_code")

    primary_bits = [f"最大扣分项={primary_reason}"]
    if primary_metric:
        primary_bits.append(f"metric={primary_metric}")
    if primary_code:
        primary_bits.append(f"error_code={primary_code}")
    if primary_measured:
        primary_bits.append(f"测得 {primary_measured}")
    if primary_threshold:
        primary_bits.append(str(primary_threshold))
    if primary_penalty:
        primary_bits.append(f"扣 {primary_penalty} 分")

    strengths = brief.get("strengths") or []
    strength_hint = (
        "、".join(
            str(s.get("coach_fact") or s.get("label_zh") or "") for s in strengths[:2]
        )
        or "结合雷达优势给予肯定"
    )

    sections = [
        "【全景体检表 · 动力链因果推演上下文】",
        "",
        "一、基础信息",
        f"总分 TotalScore={total_txt}",
        f"雷达图极高维度：{extremes.get('readable_high') or '暂缺'}",
        f"雷达图极低维度：{extremes.get('readable_low') or '暂缺'}",
        "",
        "二、动力链时序数据 (Kinematic Chain Data)",
        "（按支撑→蓄力→触球→动量排列，供因果推演；缺测项已标明，禁止编造）",
        chain_text,
        "",
        "三、系统判定原发病灶 (Primary Error)",
        "【警告】以下为系统确定性最大扣分项，禁止自行改判病因：",
        "；".join(primary_bits),
        f"【ClinicalBrief 首要事实】{primary_fact or primary_desc}",
        "",
        "四、可夸优点/雷达优势",
        strength_hint,
        "",
        "五、全量测量明细",
        measurement.get("readable_text") or "暂缺",
    ]
    return "\n".join(sections)


def build_aigc_safe_payload(diagnosis: dict) -> dict:
    """构造交给大模型的动态载荷：全景体检表 + ClinicalBrief + 指标。"""
    diagnosis = diagnosis if isinstance(diagnosis, dict) else {}
    primary_error_description = build_primary_error_description(diagnosis)
    brief = build_clinical_brief(diagnosis)
    measurement = build_measurement_context(diagnosis)
    chain = build_kinematic_chain_data(diagnosis)
    extremes = _radar_extreme_dims(measurement.get("radar_scores") or {})
    return {
        "primary_error_code": diagnosis.get("primary_error_code"),
        "primary_error_description": primary_error_description,
        "t_impact": diagnosis.get("t_impact", diagnosis.get("t0_index")),
        "red_defect_priority_hint": _pick_primary_red_defect(diagnosis),
        "TotalScore": measurement.get("TotalScore"),
        "radar_scores": measurement.get("radar_scores"),
        "radar_extremes": extremes,
        "primary_deduction": measurement.get("primary_deduction"),
        "secondary_deductions": measurement.get("secondary_deductions"),
        "deductions": measurement.get("deductions"),
        "measurement_context_text": measurement.get("readable_text"),
        "kinematic_chain": chain,
        "clinical_brief": brief,
        "indicators": _extract_indicator_payload(diagnosis),
    }


def build_aigc_user_message(diagnosis: dict) -> str:
    """组装全景体检表 User Message（与 generate_feedback 同源，供 E2E 断言）。

    结构：基础信息 → 动力链时序（支撑/蓄力/触球/动量）→ 系统原发病灶 → JSON 载荷。
    """
    safe_payload = build_aigc_safe_payload(diagnosis)
    panorama = build_panorama_checkup_text(diagnosis)
    return (
        f"{panorama}\n\n"
        "请基于上方「全景体检表」做动力链因果推演：从近端支撑偏差如何传导到"
        "蓄力折叠、触球释放与动量表现；必须锚定系统判定的 Primary Error，"
        "不得另起炉灶猜测病因。\n"
        "请严格按系统命令只返回 JSON，字段仅允许 "
        "overview、biomechanical_analysis、magic_metaphor、action_plan；"
        "必须引用上方真实测量数据（总分/雷达/扣分测得值与阈值/动力链时序）；"
        "绝不能每次说一样的话；四个字段必须互不重复：\n"
        f"{json.dumps(safe_payload, ensure_ascii=False)}"
    )


def generate_optimal_dual_feedback(diagnosis_json, status=None) -> dict:
    """调用 DeepSeek：把缺陷 JSON 转译为 OPTIMAL 双段字典。

    返回：
        {"correction_metaphor": str, "praise_encouragement": str}
        任何异常 / 非 JSON / 重复废话均返回硬性或缺陷级兜底，绝不抛出。
    """
    # 旧签名兼容：generate_feedback(angle, status)
    if isinstance(diagnosis_json, (int, float)) and status is not None:
        diagnosis = {
            "legacy_angle": float(diagnosis_json),
            "legacy_status": str(status),
            "score_detail": {
                "indicators": {
                    "impact_knee_angle": {
                        "value": float(diagnosis_json),
                        "status": (
                            "RED_DEVIATED"
                            if str(status).lower() == "red"
                            else "YELLOW_APPROACHING"
                            if str(status).lower() == "yellow"
                            else "GREEN_OPTIMAL"
                        ),
                    }
                }
            },
            "llm_participated": False,
        }
    else:
        diagnosis = _normalize_diagnosis_json(diagnosis_json)

    primary_error_description = build_primary_error_description(diagnosis)
    measurement_context = build_measurement_context(diagnosis)
    system_prompt = build_system_prompt(
        primary_error_description, measurement_context=measurement_context
    )
    user_message = build_aigc_user_message(diagnosis)

    try:
        response = _chat_completions_with_backoff(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=LLM_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        raw_text = response.choices[0].message.content or ""
        return _parse_optimal_dual_feedback(raw_text, diagnosis)

    except Exception as exc:  # noqa: BLE001 - 网络/接口异常时需要兜底，不能让程序崩溃
        print(f"【llm_agent】调用 DeepSeek 接口失败，使用动态数据 Fallback。错误信息：{exc}")
        dual = _optimal_fallback_dual(diagnosis)
        dual["clinical_brief"] = build_clinical_brief(diagnosis)
        return dual


# --------------------------------------------------------------------------
# 第三步：核心对外函数 generate_feedback
# --------------------------------------------------------------------------


def generate_feedback(diagnosis_json, status=None):
    """调用 DeepSeek：把缺陷 JSON 翻译为 OPTIMAL 双段童趣话术。

    【权限解耦】
        - 本函数绝不计算、修改或返回任何数值评分。
        - 唯一合法输入是 error_diagnoser 输出的诊断 JSON（dict 或 JSON 字符串）。
        - 兼容旧调用 generate_feedback(angle: float, status: str)。

    返回：
        str，【魔法指令】+【闪光点发现】可读拼接（兼容字幕/日志调用方）。
    """
    dual = generate_optimal_dual_feedback(diagnosis_json, status)
    return _format_optimal_dual_text(dual)


# --------------------------------------------------------------------------
# 【v1.1 新增：前后端全栈联调】第三步半：整堂课/整次训练的综合诊断报告生成
# --------------------------------------------------------------------------


def _deterministic_session_score_from_hits(hit_stats: dict) -> float:
    """会话级纯数学评分兜底（无逐帧数据时）：由红黄绿命中次数线性推导，LLM 零参与。"""
    green = int(hit_stats.get("green", 0) or 0)
    yellow = int(hit_stats.get("yellow", 0) or 0)
    red = int(hit_stats.get("red", 0) or 0)
    total = max(1, green + yellow + red)
    # 100 起：黄各扣 4，红各扣 12，再按绿色占比微调，保留两位小数
    score = 100.00 - yellow * 4.0 - red * 12.0
    score = score * (0.55 + 0.45 * (green / total))
    return round(max(0.0, min(100.0, float(score))), 2)


def _build_fallback_report(hit_stats, total_attempts, deterministic_score=None, diagnosis=None):
    """当 DeepSeek 接口调用失败或返回内容解析失败时的规则化兜底报告。

    score 永远来自确定性数学；文案为四维动态字典 + ClinicalBrief。
    """
    if deterministic_score is not None:
        score = round(float(deterministic_score), 2)
    else:
        score = _deterministic_session_score_from_hits(hit_stats)

    diagnosis = dict(diagnosis or {})
    # 将会话确定性总分写入 score_detail，供动态兜底引用
    detail = dict(diagnosis.get("score_detail") or {})
    if detail.get("TotalScore") is None:
        detail["TotalScore"] = score
    diagnosis["score_detail"] = detail

    if detail.get("indicators") or detail.get("deductions") or diagnosis.get("indicators"):
        dual = _optimal_fallback_dual(diagnosis)
    elif int(hit_stats.get("red", 0) or 0) > 0:
        dual = _depth_fallback_report(
            {
                "score_detail": {
                    "TotalScore": score,
                    "deductions": [{"reason": "支撑脚站位偏差", "penalty": 1.0}],
                }
            }
        )
    elif int(hit_stats.get("yellow", 0) or 0) > 0:
        dual = _depth_fallback_report(
            {
                "score_detail": {
                    "TotalScore": score,
                    "deductions": [{"reason": "摆动腿折叠不足", "penalty": 1.0}],
                }
            }
        )
    else:
        dual = _attach_legacy_aliases(
            {**_STATIC_OPTIMAL_FALLBACK, "overview": f"本次综合评分 {score:.1f} 分，整体观感积极可圈可点。", "aigc_source": "fallback"}
        )
    _ = total_attempts  # 保留参数语义，供调用方传入会话规模
    return {"score": score, **_dual_to_report_fields(dual, diagnosis)}


def generate_session_report(
    hit_stats,
    student_number,
    sample_angles=None,
    deterministic_score=None,
    diagnosis_json=None,
):
    """把「一整次训练」的 ClinicalBrief / 红黄绿统计转译为 OPTIMAL 话术；评分绝不经 LLM。

    返回：
        dict，含 score、correction_metaphor、praise_encouragement、clinical_echo、
        aigc_source、clinical_brief 等。
    """
    hit_stats = hit_stats or {}
    total_attempts = sum(hit_stats.get(k, 0) for k in ("green", "yellow", "red"))

    if deterministic_score is not None:
        score = round(float(deterministic_score), 2)
    else:
        score = _deterministic_session_score_from_hits(hit_stats)

    if total_attempts == 0 and not diagnosis_json:
        empty_dual = _attach_legacy_aliases(
            {
                "overview": "本次综合评分 0 分，还没有踢出有效一脚，期待你的第一次尝试。",
                "biomechanical_analysis": "暂无扣分明细；先完成一次有效触球，才能诊断动力链病灶。",
                "magic_metaphor": (
                    "你刚才还没踢出去，就像弓弦没拉开一样！"
                    "下次试试先弯弯小腿再弹出去。"
                ),
                "action_plan": "先完成一次完整助跑触球。",
                "aigc_source": "fallback",
            }
        )
        return {"score": 0.0, **_dual_to_report_fields(empty_dual)}

    diagnosis = _normalize_diagnosis_json(diagnosis_json)
    # 将会话确定性总分写回 score_detail，确保 Prompt / 兜底 / 载荷同源
    detail_for_llm = dict(diagnosis.get("score_detail") or {})
    if detail_for_llm.get("TotalScore") is None:
        detail_for_llm["TotalScore"] = score
    diagnosis = {**diagnosis, "score_detail": detail_for_llm}

    primary_error_description = build_primary_error_description(diagnosis)
    measurement_context = build_measurement_context(diagnosis)
    system_prompt = build_system_prompt(
        primary_error_description, measurement_context=measurement_context
    )
    brief = build_clinical_brief(diagnosis)
    base_payload = build_aigc_safe_payload(diagnosis)
    safe_payload = {
        "student_number": student_number or "未填写",
        "hit_stats_labels": {
            "green": hit_stats.get("green", 0),
            "yellow": hit_stats.get("yellow", 0),
            "red": hit_stats.get("red", 0),
        },
        **base_payload,
        "TotalScore": measurement_context.get("TotalScore") or score,
        "measurement_context_text": measurement_context.get("readable_text"),
    }
    if sample_angles:
        try:
            safe_payload["sample_knee_angle_mean_deg"] = round(
                sum(float(a) for a in sample_angles) / len(sample_angles), 1
            )
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    primary_fact = (brief.get("primary") or {}).get("coach_fact") or primary_error_description
    strengths = brief.get("strengths") or []
    strength_hint = (
        "、".join(
            str(s.get("coach_fact") or s.get("label_zh") or "") for s in strengths[:2]
        )
        or "结合雷达优势给予肯定"
    )
    measurement_text = measurement_context.get("readable_text") or ""
    user_message = (
        f"【ClinicalBrief 首要事实】{primary_fact}。\n"
        f"【全量测量数据】\n{measurement_text}\n"
        f"【可夸优点/雷达优势】{strength_hint}\n"
        "请严格按系统命令只返回 JSON，字段仅允许 "
        "overview、biomechanical_analysis、magic_metaphor、action_plan；"
        "必须引用上方真实测量数据；绝不能每次说一样的话；"
        "四个字段必须互不重复：\n"
        f"{json.dumps(safe_payload, ensure_ascii=False)}"
    )

    try:
        response = _chat_completions_with_backoff(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=LLM_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        raw_text = response.choices[0].message.content or ""
        dual = _parse_optimal_dual_feedback(raw_text, diagnosis)
        dual = _ensure_report_cites_measurements(dual, diagnosis)

        # 【铁律】即便模型幻觉输出了 score，也一律丢弃，强制使用确定性分数
        return {"score": score, **_dual_to_report_fields(dual, diagnosis)}

    except Exception as exc:  # noqa: BLE001 - 网络异常/解析失败等都需要兜底
        print(f"【llm_agent】调用 DeepSeek 生成综合报告失败，使用动态数据 Fallback。错误信息：{exc}")
        return _build_fallback_report(
            hit_stats, total_attempts, deterministic_score=score, diagnosis=diagnosis
        )


def _ensure_report_cites_measurements(
    report: Optional[dict], diagnosis: Optional[dict]
) -> dict:
    """若模型文案未引用任何实测数字，用动态兜底覆盖，杜绝千篇一律空话。"""
    report = _attach_legacy_aliases(report or {})
    ctx = build_measurement_context(diagnosis)
    primary = ctx.get("primary_deduction") or {}
    measured = str(primary.get("measured_value") or "").strip()
    total = ctx.get("TotalScore")
    total_txt = f"{total}" if total is not None else ""

    def _has_digit(text: str) -> bool:
        return bool(re.search(r"\d", text or ""))

    biomech = report.get("biomechanical_analysis") or ""
    overview = report.get("overview") or ""
    magic = report.get("magic_metaphor") or ""
    cites_ok = (
        _has_digit(biomech)
        and _has_digit(overview)
        and (not measured or measured.split()[0] in biomech or measured in biomech or measured in magic)
    )
    if cites_ok and (not total_txt or total_txt in overview or total_txt in biomech):
        return report

    # 模型空话 / 未引用数据 → 用测量上下文重写，保留 LLM 隐喻若其已含数字
    fallback = _depth_fallback_report(diagnosis)
    merged = {
        "overview": fallback.get("overview") or overview,
        "biomechanical_analysis": fallback.get("biomechanical_analysis") or biomech,
        "magic_metaphor": magic if _has_digit(magic) else fallback.get("magic_metaphor"),
        "action_plan": report.get("action_plan")
        if _has_digit(report.get("action_plan") or "")
        else fallback.get("action_plan"),
        "aigc_source": report.get("aigc_source") or "fallback",
        "clinical_brief": report.get("clinical_brief"),
    }
    if report.get("aigc_source") == "llm":
        merged["aigc_source"] = "llm_enriched"
    return _attach_legacy_aliases(merged)


# --------------------------------------------------------------------------
# 【v2.0 新增：延时反馈系统跨课时聚合复盘】第三步再半：
# 同一位学生连续 2~3 次尝试之间的「跨次趋势诊断」生成
# --------------------------------------------------------------------------

# 聚合诊断：同一学生历史 ≥2 次有效尝试的跨次/跨课时趋势（科研口径，禁止比喻）
AGGREGATE_SYSTEM_PROMPT = """你是严谨的高校运动生物力学专家。绝对禁止使用任何修辞手法、比喻\
（如木棍、弹簧、弹簧门、滑滑梯、扫把、弓弦、大树、铁板等）、拟人或情绪化词汇。\
违背此规则将被判定为严重错误！

基于同一受试者历史至少 2 次踢球测试的评分/命中统计变化（可跨课时、不限同一天），\
撰写跨次趋势诊断，供教师课后快速判读运动表现稳定性。

【铁律】
- 禁止寒暄、过渡句、鼓励语；只陈述趋势机制与可执行纠正。
- 可用生物力学术语（矢状面、动量、折叠鞭打、重心侧倾等）。
- trendDescription 只聚焦一个最显著的跨次变化机制。
- 禁止输出任何比喻或拟人。

【输出格式】
- 只返回合法 JSON 对象，禁止 Markdown 代码围栏与多余文字。
- 字段仅允许：
    1. "trendDescription"：字符串，40-100字，跨次趋势的力学机制陈述；
    2. "prescription"：字符串，40-100字，下节课厘米级/角度级纠正指令。
- 必须使用简体中文。
"""


def _build_fallback_aggregate_report(attempts_summary):
    """DeepSeek 接口调用失败或解析失败时的规则化兜底聚合诊断。"""
    scores = [a.get("score") for a in attempts_summary if isinstance(a.get("score"), (int, float))]

    if len(scores) < 2:
        return {
            "trendDescription": "该生历史有效测试评分不足 2 次，无法建立跨课时生物力学趋势推断。",
            "prescription": "累计完成至少 2 次踢球测试后再出具聚合诊断（可不在同一天）。",
        }

    first_score, last_score = scores[0], scores[-1]
    if last_score > first_score + 3:
        trend = "从首次到末次尝试，发力稳定性评分上升，提示运动链控制在重复练习中趋于收敛。"
        prescription = "下节课维持当前助跑-支撑-鞭打时序，并将支撑脚横距约束在球心侧方 15-20 厘米。"
    elif last_score < first_score - 3:
        trend = "多次尝试后段发力稳定性评分下降，符合疲劳累积导致远端环节控制精度下降的表现。"
        prescription = "下节课将练习分段并插入被动恢复，每段开始前复核支撑脚横距与后摆折叠角。"
    else:
        trend = "各次尝试发力稳定性评分波动较小，运动表现整体处于可重复区间。"
        prescription = "维持现有技术结构，可在支撑稳定的前提下小幅提高助跑速度以检验鞭打峰值。"

    return {"trendDescription": trend, "prescription": prescription}


def _classify_aggregate_llm_error(exc: BaseException) -> str:
    """把 DeepSeek / 网络异常归类为前端可展示的中文错误文案。"""
    name = type(exc).__name__
    text = str(exc) or ""
    lowered = f"{name} {text}".lower()
    if (
        "timeout" in lowered
        or "timed out" in lowered
        or name in {"APITimeoutError", "TimeoutError", "ReadTimeout", "ConnectTimeout"}
    ):
        return "大模型生成超时，请重试"
    if (
        "connection" in lowered
        or "connect" in lowered
        or name in {"APIConnectionError", "ConnectError"}
    ):
        return "无法连接大模型服务，请检查网络后重试"
    if "rate limit" in lowered or "429" in lowered:
        return "大模型请求过于频繁，请稍后再试"
    if "auth" in lowered or "401" in lowered or "403" in lowered:
        return "大模型鉴权失败，请检查 API Key 配置"
    return f"大模型生成失败：{text[:120] if text else name}，请重试"


def generate_aggregate_diagnosis(student_number, attempts_summary):
    """调用 DeepSeek 大模型，把同一位学生「历史有效尝试（跨课时亦可）」的评分/
    三级命中统计变化趋势，转译成结构化的跨次聚合诊断（trendDescription /
    prescription 两个字段）。

    参数：
        student_number：str，学生学号，仅用于让大模型的表达更有针对性。
        attempts_summary：list[dict]，每一项形如
            {"attemptNumber": 1, "score": 82, "hitStats": {"green": 5, "yellow": 2, "red": 0}}，
            按尝试发生的先后顺序排列。阈值已放宽：只要有效评分样本 ≥ 2
            （无论是否同一天）即可生成趋势聚合。

    返回：
        dict，包含 "trendDescription"（str）、"prescription"（str）两个字段；
        若大模型调用失败，额外附带 "error"（str）供前端展示具体原因，
        同时仍返回规则化兜底正文，避免白屏。
    """
    attempts_summary = attempts_summary or []

    valid_scores = [
        item.get("score")
        for item in attempts_summary
        if isinstance(item.get("score"), (int, float))
    ]

    if len(valid_scores) < 2:
        return {
            "trendDescription": "该生历史有效测试评分不足 2 次，无法建立跨课时生物力学趋势推断。",
            "prescription": "累计完成至少 2 次有效踢球测试后再出具聚合诊断（可不在同一天）。",
            "error": "历史有效数据不足 2 次，暂无法生成聚合诊断",
        }

    lines = []
    for item in attempts_summary:
        attempt_no = item.get("attemptNumber")
        score = item.get("score")
        hit = item.get("hitStats") or {}
        lines.append(
            f"第{attempt_no}次尝试：发力稳定性评分 {score if score is not None else '未知'} 分，"
            f"三级判定 Green {hit.get('green', 0)} / Yellow {hit.get('yellow', 0)} / Red {hit.get('red', 0)}。"
        )
    attempts_text = "\n".join(lines)

    user_message = (
        f"学生学号：{student_number or '未填写'}。"
        f"该学生历史一共完成了 {len(attempts_summary)} 次踢球测试（可跨课时），统计如下：\n"
        f"{attempts_text}\n"
        f"禁止比喻与寒暄。严格按系统提示词返回 JSON（trendDescription / prescription）。"
    )

    try:
        response = _chat_completions_with_backoff(
            messages=[
                {"role": "system", "content": AGGREGATE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=LLM_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        raw_text = response.choices[0].message.content.strip()
        parsed = json.loads(raw_text)

        trend_description = str(parsed["trendDescription"]).strip()
        prescription = str(parsed["prescription"]).strip()

        if not trend_description or not prescription:
            raise ValueError("DeepSeek 返回的聚合诊断字段为空")

        return {"trendDescription": trend_description, "prescription": prescription}

    except Exception as exc:  # noqa: BLE001 - 网络异常/JSON 解析失败等都需要兜底
        error_message = _classify_aggregate_llm_error(exc)
        print(
            f"【llm_agent】调用 DeepSeek 生成聚合诊断失败，使用静态模板 Fallback。"
            f"错误信息：{exc} → 前端提示：{error_message}"
        )
        fallback = _build_fallback_aggregate_report(attempts_summary)
        fallback["error"] = error_message
        return fallback


# --------------------------------------------------------------------------
# 【v3.0 新增：教练端科研指挥中心】第三步再再半：
# 全班集体宏观诊断 —— 基于全班高频生物力学错误分布生成教学处方
# --------------------------------------------------------------------------

# 集体诊断：面向授课教师，科研口径（已禁止儿童化比喻）
CLASS_PRESCRIPTION_SYSTEM_PROMPT = """你是严谨的高校运动生物力学专家与青少年足球教研顾问。\
绝对禁止使用任何修辞手法、比喻（如木棍、弹簧、滑滑梯、扫把、弓弦、大树、铁板等）、\
拟人或情绪化词汇。违背此规则将被判定为严重错误！

基于班级历史测试中各项生物力学错误出现率，撰写集体教学诊断简报：

1. "diagnosis"：指出最突出的 1-2 个集体性技术短板及其力学成因（可用专业术语）。
2. "prescription"：给出 3 条可在 45 分钟体育课内执行的纠正重点（含距离/角度/组次数）。

【输出格式】
- 只返回合法 JSON，禁止 Markdown 代码围栏与寒暄。
- 字段仅允许："diagnosis"（80-150字）、"prescription"（100-200字，可用①②③分点）。
- 不得编造用户未提供的数值；必须使用简体中文。
"""


def _build_fallback_class_prescription(error_stats: dict, total_records: int) -> dict:
    """DeepSeek 接口调用失败或解析失败时的规则化兜底集体诊断简报。"""
    if not error_stats or total_records <= 0:
        return {
            "diagnosis": "当前班级暂无足够的历史测试记录，无法生成集体诊断简报，请先完成至少一轮全班测试。",
            "prescription": "①先安排一次全员基础摸底测试；②确保测试数据成功归档进全局训练数据库；③归档完成后重新点击生成即可。",
        }

    top_error = max(error_stats.items(), key=lambda item: item[1])
    label, rate = top_error
    return {
        "diagnosis": f"集体诊断提示：当前班级由于核心力量与动作控制能力仍在发展中，在连续练习中呈现出「{label}」的高频问题（发生率约 {rate:.0f}%），这通常与儿童此阶段核心稳定肌群和神经-肌肉协调能力尚未完全成熟有关。",
        "prescription": f"建议下一步教学重点实施：①增加静态平衡与核心稳定性专项练习（如单腿站立传接球，每组30秒 x 3组）；②针对「{label}」设计低强度分解动作教学，配合镜面示范与慢动作回放；③课后布置低负荷家庭巩固练习，逐步固化正确动作模式。",
    }


def generate_class_prescription(school, class_group, error_stats, total_records, avg_score=None):
    """调用 DeepSeek 大模型，基于全班历史记录中各项生物力学错误分类的出现
    频率统计，生成一份面向授课教师的「集体教学诊断简报 + 处方」。

    参数：
        school：str，学校/机构名称。
        class_group：str，班级/实验组别名称。
        error_stats：dict，形如 {"支撑脚位置偏离": 48.0, "膝关节过度屈曲": 22.5}，
                      键为错误分类标签，值为该分类在全班记录中的出现百分比。
        total_records：int，参与统计的历史记录总条数。
        avg_score：float | None，全班平均发力综合评分，仅作为参考上下文。

    返回：
        dict，包含 "diagnosis"（str）、"prescription"（str）两个字段。
        任何异常情况下都会返回结构完整的兜底数据，绝不抛出异常。
    """
    error_stats = error_stats or {}

    if total_records <= 0 or not error_stats:
        return _build_fallback_class_prescription(error_stats, total_records)

    stats_lines = "\n".join(
        f"- {label}：出现率约 {rate:.1f}%" for label, rate in sorted(error_stats.items(), key=lambda kv: -kv[1])
    )
    score_hint = f"全班平均发力综合评分约为 {avg_score:.0f} 分。" if isinstance(avg_score, (int, float)) else ""

    user_message = (
        f"学校/机构：{school or '未设置'}。班级/实验组别：{class_group or '未设置'}。"
        f"该班级共有 {total_records} 条历史测试记录，各项生物力学错误分类的出现率统计如下：\n"
        f"{stats_lines}\n{score_hint}"
        f"请严格按照系统提示词规定的 JSON 格式，生成这个班级的集体教学诊断简报。"
    )

    try:
        response = _chat_completions_with_backoff(
            messages=[
                {"role": "system", "content": CLASS_PRESCRIPTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=LLM_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        raw_text = response.choices[0].message.content.strip()
        parsed = json.loads(raw_text)

        diagnosis = str(parsed["diagnosis"]).strip()
        prescription = str(parsed["prescription"]).strip()
        if not diagnosis or not prescription:
            raise ValueError("DeepSeek 返回的集体诊断字段为空")

        return {"diagnosis": diagnosis, "prescription": prescription}

    except Exception as exc:  # noqa: BLE001 - 网络异常/JSON 解析失败等都需要兜底
        print(f"【llm_agent】调用 DeepSeek 生成集体教学诊断简报失败，使用静态模板 Fallback。错误信息：{exc}")
        return _build_fallback_class_prescription(error_stats, total_records)


# --------------------------------------------------------------------------
# 【v3.0 新增：教练端科研指挥中心】第三步再再再半：
# 个体纵向进化画像 —— 基于某学生全周期历史记录生成优缺点总结
# --------------------------------------------------------------------------

INDIVIDUAL_SUMMARY_SYSTEM_PROMPT = """你是严谨的高校运动生物力学专家。绝对禁止使用任何修辞手法、比喻\
（如木棍、弹簧、弹簧门、滑滑梯、扫把、弓弦、大树、铁板等）、拟人或情绪化词汇。\
违背此规则将被判定为严重错误！

基于某受试者全周期历史评分与错误分类统计，撰写个体纵向运动表现摘要。

【铁律】
- 禁止寒暄、鼓励语、儿童化口吻；只用客观力学陈述。
- 每个字段只聚焦最突出的一点。

【输出格式】
- 只返回合法 JSON，禁止 Markdown 代码围栏。
- 字段仅允许：
    1. "strengths"：40-80字，最稳定的生物力学优势（无比喻）；
    2. "weaknesses"：40-80字，最需纠正的习惯性偏差及厘米级/角度级方向（无比喻、无情绪词）。
- 必须使用简体中文。
"""


def _build_fallback_individual_summary(scores, error_counter):
    """DeepSeek 接口调用失败或解析失败时的规则化兜底个体总结。"""
    if not scores:
        return {
            "strengths": "历史测试样本不足，尚无法判定稳定的生物力学优势指标。",
            "weaknesses": "数据不足，暂不能给出针对性纠正区间；需补充完整踢球采样。",
        }

    avg_score = sum(scores) / len(scores)
    strengths = (
        "全周期发力稳定性评分维持在较高区间，运动链时序重复性较好。"
        if avg_score >= 70
        else "已形成基本的后摆-伸展击球时序，近端到远端的动量传递仍有提升空间。"
    )

    if error_counter:
        top_label = max(error_counter.items(), key=lambda item: item[1])[0]
        weakness_map = {
            "支撑脚位置偏离": "高频偏差为支撑脚横距失控；落地应约束在球心侧方 15-20 厘米。",
            "膝关节过度屈曲": "触球膝角偏屈；触球瞬间将摆动腿膝角回调至可控伸展区间。",
            "随摆转髋不足": "髋扭转不足导致角动量传递受限；击球过程完成与助跑方向一致的骨盆旋转。",
            "身体重心偏移": "重心投影偏离支撑基面；击球前先稳定支撑腿刚度再启动摆动腿。",
        }
        weaknesses = weakness_map.get(
            top_label, f"高频问题集中于「{top_label}」，需在分解练习中做厘米级/角度级定点纠正。"
        )
    else:
        weaknesses = "未形成显著集中的错误分类；维持现有技术结构并定期复查关键角与支撑横距。"

    return {"strengths": strengths, "weaknesses": weaknesses}


def generate_individual_summary(student_id, score_history, error_counter):
    """调用 DeepSeek 大模型，基于某学生全周期历史评分序列与生物力学错误分类
    出现次数统计，生成结构化的「个体优缺点总结」。

    参数：
        student_id：str，学生编号/学号。
        score_history：list[int|float]，该生全部历史记录的评分序列（按时间先后排列）。
        error_counter：dict，形如 {"支撑脚位置偏离": 3, "膝关节过度屈曲": 1}，
                       该生历史记录中各项错误分类出现的次数统计。

    返回：
        dict，包含 "strengths"（str）、"weaknesses"（str）两个字段。
        任何异常情况下都会返回结构完整的兜底数据，绝不抛出异常。
    """
    score_history = score_history or []
    error_counter = error_counter or {}

    if not score_history:
        return _build_fallback_individual_summary(score_history, error_counter)

    scores_text = " -> ".join(str(s) for s in score_history)
    error_text = (
        "、".join(f"{label}（出现{count}次）" for label, count in sorted(error_counter.items(), key=lambda kv: -kv[1]))
        or "暂无明显集中的错误分类"
    )

    user_message = (
        f"学生编号：{student_id or '未填写'}。"
        f"该生全周期历史评分序列（从第一次到最近一次）：{scores_text}。"
        f"该生历史错误分类出现次数：{error_text}。"
        f"禁止比喻与寒暄。严格按系统提示词返回 JSON（strengths / weaknesses）。"
    )

    try:
        response = _chat_completions_with_backoff(
            messages=[
                {"role": "system", "content": INDIVIDUAL_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=LLM_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        raw_text = response.choices[0].message.content.strip()
        parsed = json.loads(raw_text)

        strengths = str(parsed["strengths"]).strip()
        weaknesses = str(parsed["weaknesses"]).strip()
        if not strengths or not weaknesses:
            raise ValueError("DeepSeek 返回的个体总结字段为空")

        return {"strengths": strengths, "weaknesses": weaknesses}

    except Exception as exc:  # noqa: BLE001 - 网络异常/JSON 解析失败等都需要兜底
        print(f"【llm_agent】调用 DeepSeek 生成个体纵向进化总结失败，使用静态模板 Fallback。错误信息：{exc}")
        return _build_fallback_individual_summary(score_history, error_counter)


# --------------------------------------------------------------------------
# 第四步：独立运行测试（方便单独调试本模块，不需要打开摄像头）
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # 直接运行本文件时，用一组模拟诊断 JSON 测试 OPTIMAL 双段式转译（不含评分）
    test_diagnosis = {
        "primary_error_code": "ERR_A2_SUPPORT_WIDE",
        "t_impact": 60,
        "score_detail": {
            "indicators": {
                "ankle_rigidity": {"value": 0.12, "status": "GREEN_OPTIMAL"},
                "distance_cm": {"value": 28.5, "status": "RED_DEVIATED"},
                "max_folding_angle": {"value": 42.0, "status": "YELLOW_APPROACHING"},
                "impact_knee_angle": {"value": 151.0, "status": "YELLOW_APPROACHING"},
            }
        },
    }
    print(f"测试输入诊断 JSON（含实测值、无评分数值）：{json.dumps(test_diagnosis, ensure_ascii=False)}")
    print("正在调用 DeepSeek 大模型，请稍候……")
    dual = generate_optimal_dual_feedback(test_diagnosis)
    print("大模型返回的 OPTIMAL 双段 JSON：")
    print(json.dumps(dual, ensure_ascii=False, indent=2))
    print("可读拼接：")
    print(generate_feedback(test_diagnosis))
