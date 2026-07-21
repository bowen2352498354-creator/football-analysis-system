# -*- coding: utf-8 -*-
"""
llm_agent.py
v3.0 科研级生物力学诊断转译引擎（大模型代理模块）——职责严格解耦

【权限铁律】
    大语言模型完全不接触数值评分计算。评分由 error_diagnoser.DeterministicScorer
    纯数学独占；本模块只允许接收其输出的 JSON 诊断报告，转译为严谨的
    三节制科研诊断 Markdown（客观实测 / 致错根因 / 临床纠正）。

功能说明：
    1. 使用官方 openai Python 库调用 DeepSeek（OpenAI 兼容协议）；
    2. 核心函数 generate_feedback(diagnosis_json)：输入诊断 JSON，返回三节制科研诊断 Markdown；
    3. System Prompt 强制：高校运动生物力学专家口径；绝对禁止比喻、拟人、情绪化修辞；
    4. temperature 强制锁定 LLM_TEMPERATURE=0.1，彻底压制发散性自回归输出。
"""

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

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
# 第二步：System Prompt —— V3.0 科研级生物力学诊断（严禁修辞发散）
# --------------------------------------------------------------------------

# 【权限铁律】大模型完全不接触数值评分计算。它只能接收 error_diagnoser.py
# 输出的 JSON 诊断报告，强制复述实测数值并给出力学致错根因与厘米级纠正指令。
# temperature 强制锁定 0.1，彻底封杀发散性自回归输出。
LLM_TEMPERATURE = 0.1

# 三节制科研诊断输出的唯一合法骨架（SYSTEM / REPORT 共用）
_CLINICAL_MARKDOWN_CONTRACT = """\
【最高级别身份与禁令——违背即判定为严重错误】
你是严谨的高校运动生物力学专家，正在撰写科研级诊断意见书。
绝对禁止使用任何修辞手法、比喻（如木棍、弹簧、弹簧门、滑滑梯、扫把、弓弦、大树、铁板、瓶盖、陀螺、冻住的小木棍等）、拟人或情绪化词汇。
绝对禁止寒暄、过渡句、鼓励语、口语化表达。
禁止输出评分、扣分、评级或对评分结果的任何评论。
禁止编造 JSON 中不存在的数值；JSON 缺测值时写“该指标未提供实测值”，不得虚构。
必须使用简体中文、专业生物力学术语（矢状面、额状面、动量、转动惯量、重心投影、折叠鞭打等）。

【输出格式——必须且只能输出以下纯 Markdown 文本，首行即【一、】，末行结束于第三节正文】
【一、 客观实测面诊】
（必须强制提取并复述传入 JSON 中的支撑脚横距、膝关节角度、折叠角、踝关节刚度等具体数值，并给出偏差定性。示例口径：“实测支撑脚距球心横距 28.5cm，严重偏宽。”禁止空泛描述。）

【二、 生物力学致错根因】
（纯力学原理解释，禁止比喻。示例口径：“支撑脚偏远导致身体重心过度侧倾，摆动腿无法在矢状面完成折叠鞭打，动量流失。”）

【三、 临床纠正药方】
（给出明确的厘米级或空间定位纠正指令。示例口径：“支撑脚落地需卡在球心侧方 15-20 厘米处。”禁止模糊建议。）

【硬性否决条款】
- 若输出中出现任何比喻/拟人/情绪词/寒暄/过渡句 → 整份报告作废级错误。
- 若缺少上述三个标题中的任何一个 → 整份报告作废级错误。
- 若第一节未复述至少一项具体数值（含单位 cm 或 °）→ 整份报告作废级错误。
- 除上述三节 Markdown 外，不得输出任何其他字符（含前言、后记、代码围栏）。
"""

SYSTEM_PROMPT = (
    "你是严谨的高校运动生物力学专家。绝对禁止使用任何修辞手法、比喻"
    "（如木棍、弹簧、弹簧门、滑滑梯、扫把、弓弦、大树、铁板等）、拟人或情绪化词汇。"
    "违背此规则将被判定为严重错误！\n\n"
    "你只能依据传入的诊断 JSON 撰写报告；优先级聚焦："
    "脚踝锁紧 > 支撑脚 > 摆动腿折叠 > 膝夹角。\n\n"
    + _CLINICAL_MARKDOWN_CONTRACT
)

# 红色缺陷优先级（与 SYSTEM_PROMPT 一致，供兜底规则使用）
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
    "ankle_rigidity": "踝关节跖屈刚度方差",
    "distance_cm": "支撑脚距球心横距",
    "toe_angle": "支撑脚尖朝向角",
    "max_folding_angle": "摆动腿最大折叠角",
    "impact_knee_angle": "触球瞬间膝关节角",
    "support_knee_angle": "支撑腿膝关节角",
    "hip_torsion_angle": "髋部相对扭转角",
    "whipping_velocity": "鞭打峰值角速度",
}

_INDICATOR_UNIT = {
    "ankle_rigidity": "",
    "distance_cm": "cm",
    "toe_angle": "°",
    "max_folding_angle": "°",
    "impact_knee_angle": "°",
    "support_knee_angle": "°",
    "hip_torsion_angle": "°",
    "whipping_velocity": "°/s",
}

_RED_DEFECT_FALLBACK_LINES = {
    "ankle_rigidity": (
        "【一、 客观实测面诊】\n"
        "触球窗口踝关节角度方差偏大，跖屈锁定不足。\n\n"
        "【二、 生物力学致错根因】\n"
        "踝关节刚性不足导致触球瞬间足段形变吸收动能，动量向球体传递效率下降。\n\n"
        "【三、 临床纠正药方】\n"
        "触球前主动跖屈并维持踝关节锁定至随摆结束；触球面保持足背中段稳定接触。"
    ),
    "distance_cm": (
        "【一、 客观实测面诊】\n"
        "支撑脚距球心横距偏离目标区间（目标约 15-20cm）。\n\n"
        "【二、 生物力学致错根因】\n"
        "支撑脚横距过大导致躯干额状面侧倾，摆动腿难以在矢状面完成折叠鞭打，动量横向流失。\n\n"
        "【三、 临床纠正药方】\n"
        "支撑脚落地需卡在球心侧方 15-20 厘米处，脚尖指向踢球方向。"
    ),
    "toe_angle": (
        "【一、 客观实测面诊】\n"
        "支撑脚尖朝向角偏离踢球前进方向。\n\n"
        "【二、 生物力学致错根因】\n"
        "支撑脚外展/内收改变骨盆定向，髋-膝运动链偏离矢状面，击球矢量偏移。\n\n"
        "【三、 临床纠正药方】\n"
        "支撑脚尖指向目标方向，允许偏差不超过约 10°。"
    ),
    "max_folding_angle": (
        "【一、 客观实测面诊】\n"
        "摆动腿后摆最大折叠角不足。\n\n"
        "【二、 生物力学致错根因】\n"
        "膝屈曲不足使小腿转动惯量偏大，角速度峰值受限，鞭打链条中断。\n\n"
        "【三、 临床纠正药方】\n"
        "后摆极端位将摆动腿膝屈曲增大至充分折叠，再在矢状面加速伸展击球。"
    ),
    "impact_knee_angle": (
        "【一、 客观实测面诊】\n"
        "触球瞬间摆动腿膝关节角偏离最优区间。\n\n"
        "【二、 生物力学致错根因】\n"
        "触球膝角异常改变末端环节有效质量与接触几何，冲量方向失控。\n\n"
        "【三、 临床纠正药方】\n"
        "触球瞬间保持摆动腿膝关节适度屈曲，避免过伸直腿鞭打或过度蹲踞。"
    ),
    "support_knee_angle": (
        "【一、 客观实测面诊】\n"
        "支撑腿膝关节角偏离稳定支撑所需区间。\n\n"
        "【二、 生物力学致错根因】\n"
        "支撑膝过伸或过屈削弱下肢刚度，重心垂直投影不稳，影响摆动腿平面约束。\n\n"
        "【三、 临床纠正药方】\n"
        "支撑腿落地后保持微屈缓冲，膝关节角稳定在可控屈曲区间至击球完成。"
    ),
    "hip_torsion_angle": (
        "【一、 客观实测面诊】\n"
        "髋部相对扭转角不足或过度。\n\n"
        "【二、 生物力学致错根因】\n"
        "骨盆-髋扭转不足使躯干角动量无法有效向摆动腿传递。\n\n"
        "【三、 临床纠正药方】\n"
        "击球过程完成骨盆向击球侧有控制的旋转，幅度与助跑方向一致。"
    ),
    "whipping_velocity": (
        "【一、 客观实测面诊】\n"
        "摆动腿鞭打峰值角速度偏低。\n\n"
        "【二、 生物力学致错根因】\n"
        "近端环节制动与远端加速时序失调，角速度峰值无法在触球前形成。\n\n"
        "【三、 临床纠正药方】\n"
        "后摆折叠后，髋先行加速、膝继发伸展，确保触球前达到角速度峰值。"
    ),
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


def _extract_indicator_payload(diagnosis: dict) -> dict:
    """提取指标 status + 实测 value（供模型复述），剥离评分/扣分字段。"""
    detail = diagnosis.get("score_detail") or {}
    indicators = detail.get("indicators") or diagnosis.get("indicators") or {}
    metrics = diagnosis.get("metrics") or {}
    out: dict = {}
    if not isinstance(indicators, dict):
        return out
    for key, item in indicators.items():
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if value is None and key == "ankle_rigidity":
            value = item.get("variance")
        if value is None and key in metrics:
            value = metrics.get(key)
        if value is None and key == "distance_cm":
            value = metrics.get("support_lateral_dist_cm")
        entry = {
            "label_zh": _INDICATOR_LABEL_ZH.get(key, key),
            "status": item.get("status"),
            "unit": _INDICATOR_UNIT.get(key, ""),
        }
        if value is not None:
            try:
                entry["value"] = round(float(value), 2)
            except (TypeError, ValueError):
                entry["value"] = value
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
    """将三节制 Markdown 拆成 (一+二 → painPoint, 三 → prescription)。"""
    cleaned = _strip_code_fences(text)
    match = re.search(r"【三[、．.\s]*临床纠正药方】", cleaned)
    if match:
        pain_point = cleaned[: match.start()].strip()
        prescription = cleaned[match.start() :].strip()
        return pain_point, prescription
    return cleaned, cleaned


def _build_clinical_fallback_markdown(diagnosis: dict) -> str:
    """无模型可用时，用规则拼出合规的三节制科研诊断 Markdown。"""
    defect = _pick_primary_red_defect(diagnosis) or "max_folding_angle"
    template = _RED_DEFECT_FALLBACK_LINES.get(
        defect, _RED_DEFECT_FALLBACK_LINES["max_folding_angle"]
    )
    indicators = _extract_indicator_payload(diagnosis)
    measured_lines = []
    for key, entry in indicators.items():
        if entry.get("value") is None:
            continue
        unit = entry.get("unit") or ""
        status = entry.get("status") or ""
        measured_lines.append(
            f"实测{entry.get('label_zh', key)} {entry['value']}{unit}（状态 {status}）"
        )
    if measured_lines:
        section_one = "；".join(measured_lines[:4]) + "。"
        template = re.sub(
            r"【一、 客观实测面诊】\n.*?\n\n【二、",
            f"【一、 客观实测面诊】\n{section_one}\n\n【二、",
            template,
            count=1,
            flags=re.DOTALL,
        )
    return template


# --------------------------------------------------------------------------
# 第三步：核心对外函数 generate_feedback
# --------------------------------------------------------------------------


def generate_feedback(diagnosis_json, status=None):
    """调用 DeepSeek：把 error_diagnoser 的 JSON 转译为三节制科研诊断 Markdown。

    【V3.0 权限解耦】
        - 本函数绝不计算、修改或返回任何数值评分。
        - 唯一合法输入是 error_diagnoser 输出的诊断 JSON（dict 或 JSON 字符串）。
        - 兼容旧调用 generate_feedback(angle: float, status: str)：会包装成最小 JSON，
          但不允许模型据此打分。

    返回：
        str，纯 Markdown 三节制科研诊断文本（客观实测 / 致错根因 / 临床纠正）。
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

    # 传入实测值供第一节复述；剥离 TotalScore / penalty，防止模型接触评分计算
    safe_payload = {
        "primary_error_code": diagnosis.get("primary_error_code"),
        "t_impact": diagnosis.get("t_impact", diagnosis.get("t0_index")),
        "red_defect_priority_hint": _pick_primary_red_defect(diagnosis),
        "indicators": _extract_indicator_payload(diagnosis),
    }

    user_message = (
        "下面是确定性诊断引擎输出的 JSON。你必须复述其中的实测数值，"
        "用纯力学语言解释致错根因，并给出厘米级/角度级纠正指令。"
        "禁止打分、禁止比喻、禁止寒暄。只输出规定的三节 Markdown：\n"
        f"{json.dumps(safe_payload, ensure_ascii=False)}"
    )

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=LLM_TEMPERATURE,
        )
        return _strip_code_fences(response.choices[0].message.content or "")

    except Exception as exc:  # noqa: BLE001 - 网络/接口异常时需要兜底，不能让程序崩溃
        print(f"【llm_agent】调用 DeepSeek 接口失败，使用兜底提示语。错误信息：{exc}")
        return _build_clinical_fallback_markdown(diagnosis)


# --------------------------------------------------------------------------
# 【v1.1 新增：前后端全栈联调】第三步半：整堂课/整次训练的综合诊断报告生成
# --------------------------------------------------------------------------

# 综合报告：与 SYSTEM_PROMPT 同一套科研禁令 + 三节制 Markdown（兼容层再拆 painPoint/prescription）
REPORT_SYSTEM_PROMPT = (
    "你是严谨的高校运动生物力学专家。绝对禁止使用任何修辞手法、比喻"
    "（如木棍、弹簧、弹簧门、滑滑梯、扫把、弓弦、大树、铁板、冻住的小木棍等）、"
    "拟人或情绪化词汇。违背此规则将被判定为严重错误！\n\n"
    "你只能依据传入的诊断 JSON 撰写整次训练的科研诊断意见书；"
    "优先级聚焦：脚踝锁紧 > 支撑脚 > 摆动腿折叠 > 膝夹角。\n\n"
    + _CLINICAL_MARKDOWN_CONTRACT
)


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

    【V3.0】score 永远来自确定性数学；文案为无比喻的三节制科研 Markdown。
    """
    if deterministic_score is not None:
        score = round(float(deterministic_score), 2)
    else:
        score = _deterministic_session_score_from_hits(hit_stats)

    markdown = _build_clinical_fallback_markdown(diagnosis or {})
    # 若无指标数据，在第一节补充会话红黄命中的客观陈述（仍禁止比喻）
    if not (diagnosis or {}).get("score_detail") and not (diagnosis or {}).get("indicators"):
        red = int(hit_stats.get("red", 0) or 0)
        yellow = int(hit_stats.get("yellow", 0) or 0)
        session_note = (
            f"本次有效触球 {total_attempts} 次，其中 RED_DEVIATED {red} 次，"
            f"YELLOW_APPROACHING {yellow} 次。"
        )
        markdown = re.sub(
            r"【一、 客观实测面诊】\n",
            f"【一、 客观实测面诊】\n{session_note}",
            markdown,
            count=1,
        )
    pain_point, prescription = _split_clinical_markdown(markdown)
    return {"score": score, "painPoint": pain_point, "prescription": prescription}


def generate_session_report(
    hit_stats,
    student_number,
    sample_angles=None,
    deterministic_score=None,
    diagnosis_json=None,
):
    """把「一整次训练」的诊断 JSON / 红黄绿统计转译为科研诊断；评分绝不经 LLM。

    参数：
        hit_stats：dict，形如 {"green": 12, "yellow": 3, "red": 2}
        student_number：str
        sample_angles：可选 list[float]（仅作内部波动参考，禁止模型输出评分）
        deterministic_score：可选 float，来自 DeterministicScorer / error_diagnoser；
            若提供则原样写入返回的 score 字段。
        diagnosis_json：可选，error_diagnoser 输出的 JSON 诊断报告。

    返回：
        dict，含 "score"（确定性数学）、"painPoint"（【一】+【二】）、
        "prescription"（【三】）。LLM 原文为三节 Markdown，此处拆分以兼容既有 API。
    """
    hit_stats = hit_stats or {}
    total_attempts = sum(hit_stats.get(k, 0) for k in ("green", "yellow", "red"))

    if deterministic_score is not None:
        score = round(float(deterministic_score), 2)
    else:
        score = _deterministic_session_score_from_hits(hit_stats)

    if total_attempts == 0 and not diagnosis_json:
        return {
            "score": 0.0,
            "painPoint": (
                "【一、 客观实测面诊】\n"
                "本次训练未采集到有效触球数据，关键生物力学指标均未提供实测值。\n\n"
                "【二、 生物力学致错根因】\n"
                "数据不足，无法建立运动链致错因果推断。"
            ),
            "prescription": (
                "【三、 临床纠正药方】\n"
                "需重新完成至少一次完整踢球采样后再出具厘米级纠正指令。"
            ),
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
        "下面是确定性诊断引擎输出的 JSON。必须复述 indicators 中的实测数值，"
        "禁止打分，禁止比喻，禁止寒暄。只输出规定的三节 Markdown：\n"
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
        raw_text = _strip_code_fences(response.choices[0].message.content or "")
        if "【一、" not in raw_text or "【三、" not in raw_text:
            raise ValueError("DeepSeek 返回内容未包含强制三节标题")

        pain_point, prescription = _split_clinical_markdown(raw_text)
        if not pain_point or not prescription:
            raise ValueError("DeepSeek 返回的报告字段为空")

        # 【铁律】即便模型幻觉输出了 score，也一律丢弃，强制使用确定性分数
        return {"score": score, "painPoint": pain_point, "prescription": prescription}

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
