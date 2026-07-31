# -*- coding: utf-8 -*-
"""
llm_agent.py
小学五年级足球教练隐喻转译引擎（大模型代理模块）——职责严格解耦

【权限铁律】
    大语言模型完全不接触数值评分计算。评分由 error_diagnoser.DeterministicScorer
    纯数学独占；本模块只允许接收其输出的缺陷 JSON，转译为鼓励孩子的一句隐喻。

功能说明：
    1. 使用官方 openai Python 库调用 DeepSeek（OpenAI 兼容协议）；
    2. 核心函数 generate_feedback(diagnosis_json)：输入诊断 JSON，返回一句隐喻短语；
    3. System Prompt 强制：小学五年级足球教练口径；绝对禁止专业术语；
    4. temperature 强制锁定 LLM_TEMPERATURE=0.2。
"""

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

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

if not DEEPSEEK_API_KEY:
    raise ValueError(
        "未检测到 DEEPSEEK_API_KEY 环境变量，请在项目根目录的 .env 文件中配置，"
        "例如：DEEPSEEK_API_KEY=你的真实密钥"
    )

# DeepSeek 官方接口地址完全兼容 OpenAI 的 SDK 调用方式，
# 只需要把 base_url 从默认的 OpenAI 官方地址换成 DeepSeek 的地址即可
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# DeepSeek 提供的对话模型名称（对应"深度思考"关闭状态下的标准对话模型）
DEEPSEEK_MODEL_NAME = "deepseek-chat"

# 创建一个全局唯一的客户端实例，避免每次调用函数都重新创建连接，提升效率
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# --------------------------------------------------------------------------
# 第二步：System Prompt —— 小学五年级足球教练（一句隐喻，不可违抗）
# --------------------------------------------------------------------------

# 【权限铁律】大模型完全不接触数值评分计算。它只能把缺陷 JSON 翻译成鼓励孩子的隐喻。
# temperature 强制锁定 0.2。
LLM_TEMPERATURE = 0.2

SYSTEM_PROMPT = (
    "你是一名小学五年级足球教练。你的唯一任务是将传入的缺陷JSON，翻译为鼓励孩子的【一句】童话或生活隐喻。\n"
    "【绝对禁令】：严禁使用'矢状面'、'生物力学'、'实测值'、'方差'等任何专业术语。\n"
    "【强制话术库参考】：\n"
    " - 支撑脚问题：'像大树的根一样深深扎进泥土里'\n"
    " - 折叠角/膝盖：'像把弓弦拉到最满，然后嗖的一下射出'\n"
    " - 脚踝未锁紧：'把脚面变成一块坚硬的铁板'\n"
    "直接输出隐喻短语，字数限制在 60 字以内，禁止任何列表或段落格式。"
)

# 会话报告与单次反馈共用同一不可违抗命令
REPORT_SYSTEM_PROMPT = SYSTEM_PROMPT

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
    "whipping_velocity": "摆腿速度",
}

_INDICATOR_UNIT = {
    "ankle_rigidity": "variance",  # 无量纲角速度方差；前端渲染为 σ²
    "distance_cm": "cm",
    "toe_angle": "°",
    "max_folding_angle": "°",
    "impact_knee_angle": "°",
    "support_knee_angle": "°",
    "hip_torsion_angle": "°",
    "whipping_velocity": "°/s",
}

# 缺陷键 → 强制话术库隐喻（模型失败时的规则化兜底）
_METAPHOR_FALLBACK_BY_DEFECT = {
    "ankle_rigidity": "把脚面变成一块坚硬的铁板，踢球时稳稳顶住小球。",
    "distance_cm": "像大树的根一样深深扎进泥土里，站稳了再踢球。",
    "toe_angle": "像大树的根一样深深扎进泥土里，脚尖对准前方目标。",
    "max_folding_angle": "像把弓弦拉到最满，然后嗖的一下射出。",
    "impact_knee_angle": "像把弓弦拉到最满，然后嗖的一下射出。",
    "support_knee_angle": "像大树的根一样深深扎进泥土里，膝盖微微弯着更稳。",
    "hip_torsion_angle": "像把弓弦拉到最满，然后嗖的一下射出。",
    "whipping_velocity": "像把弓弦拉到最满，然后嗖的一下射出。",
}

# ── Fallback 分级：第二节「为什么会这样」，第三节「下次怎么做」 ────────────────
# 【设计约束】LLM 不可用时，孩子仍必须拿到两件不同的信息：
#   · 第二节 = 原因说明（用同一套隐喻解释这一脚为什么没踢好，只讲现象与成因）；
#   · 第三节 = 行动指令（下一脚具体做什么，必须是孩子能立刻执行的一个动作）。
# 两者严禁复用同一句话，否则等于只给了一半反馈。
# 话术同样遵守 SYSTEM_PROMPT 铁律：不出现任何专业术语、不出现任何数值。
_FALLBACK_CAUSE_BY_DEFECT = {
    "ankle_rigidity": "你的脚面这一下软软的，像一块没绷紧的海绵，力气都被它自己吃掉了，没能传给小球。",
    "distance_cm": "你站的位置离小球有点不合适，身子还在晃，就像小树的根没扎稳，风一吹就摇。",
    "toe_angle": "你支撑那只脚的脚尖歪向了别处，身子被它带着跑偏，球自然也跟着往旁边跑。",
    "max_folding_angle": "你的腿是直着甩出去的，就像弓弦只拉开一点点就放手，攒不住劲儿。",
    "impact_knee_angle": "碰到球的那一瞬间膝盖伸得太开了，弓弦提前松掉，最后一下没能弹出来。",
    "support_knee_angle": "你支撑的那条腿绷得太直，像一根硬木棍撑在地上，站不住也缓不住劲。",
    "hip_torsion_angle": "你只用了腿在踢，腰和屁股没转过来，就像拧发条只拧了一半就放手。",
    "whipping_velocity": "你的小腿甩过来的时候慢慢的，像慢慢推门而不是弹簧弹开，速度没攒够。",
}

_FALLBACK_ACTION_BY_DEFECT = {
    "ankle_rigidity": "下一脚记住：踢之前先把脚面绷紧，把它想成一块坚硬的铁板，用铁板稳稳顶一下小球。",
    "distance_cm": "下一脚记住：先站到球的旁边一小步的地方，双脚像大树的根扎进泥土里，站稳了再踢。",
    "toe_angle": "下一脚记住：踢之前低头看一眼支撑脚的脚尖，让它直直指向你想让球去的地方，再出脚。",
    "max_folding_angle": "下一脚记住：出脚前先把小腿往屁股方向折一下，像把弓弦拉到最满，再嗖的一下射出去。",
    "impact_knee_angle": "下一脚记住：碰到球的那一刻膝盖再多留一点弯，等脚背贴上球，才把腿甩直。",
    "support_knee_angle": "下一脚记住：支撑腿的膝盖微微弯一点点，像准备起跳那样，弯着站反而更稳更有劲。",
    "hip_torsion_angle": "下一脚记住：先把肚子和肩膀一起转向球门，让腰带着腿一块儿甩出去，别只用腿。",
    "whipping_velocity": "下一脚记住：把小腿当成弹簧，折起来之后猛地弹出去，越快越好，试着比刚才快一点。",
}


def _cause_fallback_for_defect(defect: str) -> str:
    """第二节话术（原因说明）；未知缺陷键回落到折叠角。"""
    return _FALLBACK_CAUSE_BY_DEFECT.get(
        defect, _FALLBACK_CAUSE_BY_DEFECT["max_folding_angle"]
    )


def _action_fallback_for_defect(defect: str) -> str:
    """第三节话术（行动指令）；未知缺陷键回落到折叠角。"""
    return _FALLBACK_ACTION_BY_DEFECT.get(
        defect, _FALLBACK_ACTION_BY_DEFECT["max_folding_angle"]
    )


_RED_DEFECT_FALLBACK_LINES = {
    key: (
        f"【一、 客观实测面诊】\n{phrase}\n\n"
        f"【二、 生物力学致错根因】\n{_cause_fallback_for_defect(key)}\n\n"
        f"【三、 临床纠正药方】\n{_action_fallback_for_defect(key)}"
    )
    for key, phrase in _METAPHOR_FALLBACK_BY_DEFECT.items()
}


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


def _pick_primary_red_defect(diagnosis: dict) -> str | None:
    """按固定优先级从 JSON 中挑选唯一一个红色缺陷键名。"""
    detail = diagnosis.get("score_detail") or {}
    indicators = detail.get("indicators") or diagnosis.get("indicators") or {}
    if isinstance(indicators, dict):
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
        out[key] = entry
    return out


def _strip_code_fences(text: str) -> str:
    """剔除模型偶发包裹的 Markdown 代码围栏。"""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:markdown|md)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _split_clinical_markdown(text: str) -> tuple[str, str]:
    """兼容旧三节制拆分；新口径下一句隐喻同时写入 painPoint / prescription。"""
    cleaned = _strip_code_fences(text)
    match = re.search(r"【三[、．.\s]*临床纠正药方】", cleaned)
    if match:
        pain_point = cleaned[: match.start()].strip()
        prescription = cleaned[match.start() :].strip()
        return pain_point, prescription
    metaphor = _clamp_metaphor_phrase(cleaned)
    return metaphor, metaphor


def _clamp_metaphor_phrase(text: str) -> str:
    """压成单句隐喻：去围栏、去列表标记，硬裁 60 字。"""
    cleaned = _strip_code_fences(text)
    cleaned = re.sub(r"^[\s\-•*·、。]+", "", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned.replace("\n", ""))
    if len(cleaned) > 60:
        cleaned = cleaned[:60]
    return cleaned or _METAPHOR_FALLBACK_BY_DEFECT["max_folding_angle"]


def _pick_metaphor_fallback(diagnosis: dict) -> str:
    """按红色缺陷优先级挑选话术库隐喻；无红则看黄，再兜底折叠角话术。"""
    defect = _pick_primary_red_defect(diagnosis)
    if not defect:
        detail = (diagnosis or {}).get("score_detail") or {}
        indicators = detail.get("indicators") or (diagnosis or {}).get("indicators") or {}
        if isinstance(indicators, dict):
            for key in _RED_DEFECT_PRIORITY:
                item = indicators.get(key) or {}
                if isinstance(item, dict) and item.get("status") == "YELLOW_APPROACHING":
                    defect = key
                    break
    if not defect:
        defect = "max_folding_angle"
    return _METAPHOR_FALLBACK_BY_DEFECT.get(
        defect, _METAPHOR_FALLBACK_BY_DEFECT["max_folding_angle"]
    )


def _build_clinical_fallback_markdown(diagnosis: dict) -> str:
    """无模型可用时生成结构化三段式报告。

    【一、 客观实测面诊】—— 仅引用 measured/calibrated 数值；
                          焦点指标非实测则写"未提供实测值"，绝不引用 scoring_value。
    【二、 生物力学致错根因】—— 原因说明：用隐喻解释这一脚「为什么」没踢好。
    【三、 临床纠正药方】   —— 行动指令：告诉孩子「下次怎么做」的单个可执行动作。
                          二、三节话术分别取自 _FALLBACK_CAUSE_BY_DEFECT /
                          _FALLBACK_ACTION_BY_DEFECT，严禁复用同一句。
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
                        # ankle_rigidity 是无量纲方差，与前端 MetricCardList 保持一致的 σ² 符号
                        display = f"σ² {val}" if unit == "variance" else f"{val}{unit}"
                    except (TypeError, ValueError):
                        display = str(raw)
                    lines_sec1.append(f"  · {label}：{display}")
                else:
                    lines_sec1.append(f"  · {label}：未提供实测值")
            elif key in _FOCUS_INDICATOR_KEYS:
                # 焦点指标非实测：明确标注，绝不引用 scoring_value
                lines_sec1.append(f"  · {label}：未提供实测值")

    sec1_body = "\n".join(lines_sec1) if lines_sec1 else "  · 本次未采集到实测数据"

    # ── 第二节（原因说明）、第三节（行动指令）：按主要缺陷分别选话术 ──────────
    # 【设计约束】二节 ≠ 三节：前者解释「为什么会这样」，后者告诉孩子「下次怎么做」。
    defect = _pick_primary_red_defect(diagnosis)
    if not defect:
        if isinstance(indicators, dict):
            for key in _RED_DEFECT_PRIORITY:
                item = indicators.get(key) or {}
                if isinstance(item, dict) and item.get("status") == "YELLOW_APPROACHING":
                    defect = key
                    break
    if not defect:
        defect = "max_folding_angle"

    sec2_cause = _cause_fallback_for_defect(defect)
    sec3_action = _action_fallback_for_defect(defect)

    return (
        f"【一、 客观实测面诊】\n{sec1_body}\n\n"
        f"【二、 生物力学致错根因】\n{sec2_cause}\n\n"
        f"【三、 临床纠正药方】\n{sec3_action}"
    )


def build_aigc_safe_payload(diagnosis: dict) -> dict:
    """构造交给大模型的安全载荷（缺陷指标 + 状态，剥离评分计算字段）。"""
    diagnosis = diagnosis if isinstance(diagnosis, dict) else {}
    return {
        "primary_error_code": diagnosis.get("primary_error_code"),
        "t_impact": diagnosis.get("t_impact", diagnosis.get("t0_index")),
        "red_defect_priority_hint": _pick_primary_red_defect(diagnosis),
        "indicators": _extract_indicator_payload(diagnosis),
    }


def build_aigc_user_message(diagnosis: dict) -> str:
    """与 generate_feedback 同源的 user message（供 E2E 断言，不发起网络请求）。"""
    safe_payload = build_aigc_safe_payload(diagnosis)
    return (
        "下面是缺陷 JSON。请按系统命令输出【一句】鼓励孩子的童话或生活隐喻，"
        "禁止专业术语，禁止列表或段落，60 字以内：\n"
        f"{json.dumps(safe_payload, ensure_ascii=False)}"
    )


# --------------------------------------------------------------------------
# 第三步：核心对外函数 generate_feedback
# --------------------------------------------------------------------------


def generate_feedback(diagnosis_json, status=None):
    """调用 DeepSeek：把缺陷 JSON 翻译为鼓励孩子的一句隐喻。

    【权限解耦】
        - 本函数绝不计算、修改或返回任何数值评分。
        - 唯一合法输入是 error_diagnoser 输出的诊断 JSON（dict 或 JSON 字符串）。
        - 兼容旧调用 generate_feedback(angle: float, status: str)。

    返回：
        str，一句隐喻短语（≤60 字）。
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

    user_message = build_aigc_user_message(diagnosis)

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=LLM_TEMPERATURE,
        )
        return _clamp_metaphor_phrase(response.choices[0].message.content or "")

    except Exception as exc:  # noqa: BLE001 - 网络/接口异常时需要兜底，不能让程序崩溃
        print(f"【llm_agent】调用 DeepSeek 接口失败，使用兜底提示语。错误信息：{exc}")
        return _build_clinical_fallback_markdown(diagnosis)


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

    score 永远来自确定性数学；文案为一句鼓励隐喻。
    """
    if deterministic_score is not None:
        score = round(float(deterministic_score), 2)
    else:
        score = _deterministic_session_score_from_hits(hit_stats)

    metaphor = _build_clinical_fallback_markdown(diagnosis or {})
    if not (diagnosis or {}).get("score_detail") and not (diagnosis or {}).get("indicators"):
        if int(hit_stats.get("red", 0) or 0) > 0:
            metaphor = _METAPHOR_FALLBACK_BY_DEFECT["distance_cm"]
        elif int(hit_stats.get("yellow", 0) or 0) > 0:
            metaphor = _METAPHOR_FALLBACK_BY_DEFECT["max_folding_angle"]
        else:
            metaphor = "像大树的根一样深深扎进泥土里，再像弓弦拉满嗖的一下射出。"
        _ = total_attempts  # 保留参数语义，供调用方传入会话规模
    return {"score": score, "painPoint": metaphor, "prescription": metaphor}


def generate_session_report(
    hit_stats,
    student_number,
    sample_angles=None,
    deterministic_score=None,
    diagnosis_json=None,
):
    """把「一整次训练」的缺陷 JSON / 红黄绿统计转译为一句鼓励隐喻；评分绝不经 LLM。

    参数：
        hit_stats：dict，形如 {"green": 12, "yellow": 3, "red": 2}
        student_number：str
        sample_angles：可选 list[float]（仅作内部波动参考，禁止模型输出评分）
        deterministic_score：可选 float，来自 DeterministicScorer / error_diagnoser；
            若提供则原样写入返回的 score 字段。
        diagnosis_json：可选，error_diagnoser 输出的 JSON 诊断报告。

    返回：
        dict，含 "score"（确定性数学）、"painPoint" / "prescription"（同一句隐喻）。
    """
    hit_stats = hit_stats or {}
    total_attempts = sum(hit_stats.get(k, 0) for k in ("green", "yellow", "red"))

    if deterministic_score is not None:
        score = round(float(deterministic_score), 2)
    else:
        score = _deterministic_session_score_from_hits(hit_stats)

    if total_attempts == 0 and not diagnosis_json:
        empty_metaphor = "像大树的根一样深深扎进泥土里，再试一次你会更棒。"
        return {
            "score": 0.0,
            "painPoint": empty_metaphor,
            "prescription": empty_metaphor,
        }

    diagnosis = _normalize_diagnosis_json(diagnosis_json)
    safe_payload = {
        "student_number": student_number or "未填写",
        "hit_stats_labels": {
            "green": hit_stats.get("green", 0),
            "yellow": hit_stats.get("yellow", 0),
            "red": hit_stats.get("red", 0),
        },
        "red_defect_priority_hint": _pick_primary_red_defect(diagnosis),
        "indicators": _extract_indicator_payload(diagnosis),
    }
    if sample_angles:
        try:
            safe_payload["sample_knee_angle_mean_deg"] = round(
                sum(float(a) for a in sample_angles) / len(sample_angles), 1
            )
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    user_message = (
        "下面是缺陷 JSON。请按系统命令输出【一句】鼓励孩子的童话或生活隐喻，"
        "禁止专业术语，禁止列表或段落，60 字以内：\n"
        f"{json.dumps(safe_payload, ensure_ascii=False)}"
    )

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
            messages=[
                {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=LLM_TEMPERATURE,
        )
        metaphor = _clamp_metaphor_phrase(response.choices[0].message.content or "")
        if not metaphor:
            raise ValueError("DeepSeek 返回的隐喻短语为空")

        # 【铁律】即便模型幻觉输出了 score，也一律丢弃，强制使用确定性分数
        return {"score": score, "painPoint": metaphor, "prescription": metaphor}

    except Exception as exc:  # noqa: BLE001 - 网络异常/解析失败等都需要兜底
        print(f"【llm_agent】调用 DeepSeek 生成综合报告失败，使用规则化兜底报告。错误信息：{exc}")
        return _build_fallback_report(
            hit_stats, total_attempts, deterministic_score=score, diagnosis=diagnosis
        )


# --------------------------------------------------------------------------
# 【v2.0 新增：延时反馈系统跨课时聚合复盘】第三步再半：
# 同一位学生连续 2~3 次尝试之间的「跨次趋势诊断」生成
# --------------------------------------------------------------------------

# 聚合诊断：同一学生连续 2~3 次尝试的跨次趋势（科研口径，禁止比喻）
AGGREGATE_SYSTEM_PROMPT = """你是严谨的高校运动生物力学专家。绝对禁止使用任何修辞手法、比喻\
（如木棍、弹簧、弹簧门、滑滑梯、扫把、弓弦、大树、铁板等）、拟人或情绪化词汇。\
违背此规则将被判定为严重错误！

基于同一受试者本节课连续 2~3 次踢球测试的评分/命中统计变化，撰写跨次趋势诊断，\
供教师课前快速判读运动表现稳定性。

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

    if not scores:
        return {
            "trendDescription": "本节课有效测试评分样本不足，无法建立跨次生物力学趋势推断。",
            "prescription": "下节课至少完整完成 2 次踢球测试，以支撑跨次趋势诊断。",
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


def generate_aggregate_diagnosis(student_number, attempts_summary):
    """调用 DeepSeek 大模型，把同一位学生「本节课连续 2~3 次尝试」的评分/
    三级命中统计变化趋势，转译成结构化的跨次聚合诊断（trendDescription /
    prescription 两个字段）。

    参数：
        student_number：str，学生学号，仅用于让大模型的表达更有针对性。
        attempts_summary：list[dict]，每一项形如
            {"attemptNumber": 1, "score": 82, "hitStats": {"green": 5, "yellow": 2, "red": 0}}，
            按尝试发生的先后顺序排列。

    返回：
        dict，包含 "trendDescription"（str）、"prescription"（str）两个字段。
        任何异常情况下都会返回结构完整的兜底数据，绝不抛出异常。
    """
    attempts_summary = attempts_summary or []

    if not attempts_summary:
        return {
            "trendDescription": "本节课未采集到该生尝试数据，无法建立跨次生物力学趋势推断。",
            "prescription": "先完成至少一次踢球测试后再出具聚合诊断。",
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
        f"该学生本节课一共完成了 {len(attempts_summary)} 次踢球测试，统计如下：\n"
        f"{attempts_text}\n"
        f"禁止比喻与寒暄。严格按系统提示词返回 JSON（trendDescription / prescription）。"
    )

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
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
        print(f"【llm_agent】调用 DeepSeek 生成聚合诊断失败，使用规则化兜底报告。错误信息：{exc}")
        return _build_fallback_aggregate_report(attempts_summary)


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
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
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
        print(f"【llm_agent】调用 DeepSeek 生成集体教学诊断简报失败，使用规则化兜底报告。错误信息：{exc}")
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
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
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
        print(f"【llm_agent】调用 DeepSeek 生成个体纵向进化总结失败，使用规则化兜底报告。错误信息：{exc}")
        return _build_fallback_individual_summary(score_history, error_counter)


# --------------------------------------------------------------------------
# 第四步：独立运行测试（方便单独调试本模块，不需要打开摄像头）
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # 直接运行本文件时，用一组模拟诊断 JSON 测试科研级三节制转译（不含评分）
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
    result = generate_feedback(test_diagnosis)
    print("大模型返回的三节制科研诊断 Markdown：")
    print(result)
