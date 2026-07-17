# -*- coding: utf-8 -*-
"""
llm_agent.py
v0.3 AIGC 积极意图认知转译引擎（大模型代理模块）

功能说明：
    1. 使用官方 openai 这个 Python 库来发起网络请求（DeepSeek 官方接口
       完全兼容 OpenAI 的调用格式，只需要把 base_url 换成 DeepSeek 的
       服务器地址，再把 model 换成 "deepseek-chat" 即可，不需要额外安装
       DeepSeek 专用的 SDK）；
    2. 对外只暴露一个核心函数 generate_feedback(angle, status)：
       输入"当前膝关节角度"与"三级容错状态"，返回一句适合读给小学生听的、
       充满正能量、且完全不带生硬力学术语的中文指导语；
    3. System Prompt（系统提示词）严格照抄 project_plan.md 文档中
       "AIGC 积极意图认知转译引擎规范"一节规定的三大原则：
           - 积极意图原则：禁止评判性/惩罚性字眼，只能用建议性表述；
           - 具身隐喻原则：禁止抽象力学术语，必须换成孩子熟悉的身体比喻；
           - 单一焦点原则：一次反馈只讲一个最重要的点，避免信息过载；
    4. 调用参数严格设置 temperature=0.3，把大模型的"随机发挥"程度死死
       压低，保证同样的输入基本会得到风格、尺度都很稳定的输出，
       这对教学场景的安全性和一致性非常重要。

【重要说明】
    本模块只负责"和大模型对话，拿到一句话文本"，完全不涉及摄像头、
    OpenCV 画面绘制、多线程调度这些逻辑——那些是 pose_tracker.py 的职责。
    这样职责分离，方便后续 v0.4/v1.0 阶段单独替换或复用这个 AIGC 模块。
"""

import json
import os

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
# 第二步：System Prompt —— 严格照抄规划文档中的三大转译原则
# --------------------------------------------------------------------------

# 这段系统提示词就是整个 AIGC 模块的"灵魂"：它规定了大模型必须扮演什么角色、
# 必须遵守什么规则、绝对不能说什么话。写得越明确、越有具体范例，
# 大模型输出的内容就越稳定、越符合小学生的认知与情感需求。
SYSTEM_PROMPT = """你是一名经验丰富、充满爱心的小学足球启蒙教练，正在给10-11岁的小学五年级学生做\
动作反馈。你的反馈会被系统直接读出来/展示给孩子看，所以你必须严格遵守下面三条铁律，\
一条都不能违反：

【铁律一：积极意图原则】
- 绝对禁止出现任何评判性、惩罚性、否定性的字眼，比如"你做错了""动作错误""不对""失败""不好"。
- 必须使用信息性、建议性的表达方式，比如"下次可以试着……""再多一点点……就更棒了""我们再感受一下……"。
- 语气要始终温暖、鼓励，让孩子觉得自己是"正在变得更棒"，而不是"被批评"。

【铁律二：具身隐喻原则】
- 绝对禁止使用任何抽象的生物力学/物理学专业术语，比如"角动量""屈曲度""关节角度""力矩""向量""夹角""生物力学"。
- 必须把动作要点转译成孩子熟悉的、生动形象的身体感觉或生活化比喻，比如"像大树的根稳稳扎在地上"\
"膝盖像被轻轻按了一下、弹一下""腿像拉满的弓弦，蓄满了力量""像小猫轻轻落地一样"。

【铁律三：单一焦点原则】
- 每一次反馈只能聚焦在当前输入里最需要关注的这一个点上，绝对不要同时讲很多个知识点，
- 避免孩子的注意力和记忆负担过重，一句话讲清楚一件事就好。

【输出格式要求】
- 只输出一句话（最多可以分成两小句，用逗号或句号连接），不要输出多段文字。
- 不要输出任何解释、前缀、标签（比如不要写"反馈："这种字样），直接输出这句要说给孩子听的话本身。
- 语言必须是简体中文，语气亲切自然，像面对面在跟一个10岁孩子说话。
"""

# --------------------------------------------------------------------------
# 第三步：核心对外函数 generate_feedback
# --------------------------------------------------------------------------


def generate_feedback(angle, status):
    """调用 DeepSeek 大模型，把"膝关节角度 + 三级状态"转译成一句温暖的中文指导语。

    参数：
        angle：float 类型，当前检测到的右膝关节屈曲角度（单位：度），
               例如 118.3、175.6 等。
        status：str 类型，三级容错状态之一："Green" / "Yellow" / "Red"，
                本函数主要在 status 为 "Red"（显著偏离）时被上层代码触发调用，
                但函数本身对三种状态都可以正常处理。

    返回：
        str 类型，一句适合直接展示/朗读给小学生听的中文正向反馈语句。
        如果调用大模型接口失败（例如网络异常、Key 无效等），
        会返回一句兜底的默认鼓励语，保证上层调用方不会因为网络问题而崩溃。
    """
    # 把冷冰冰的数字和英文状态，包装成一句自然语言描述，作为"用户输入"喂给大模型，
    # 方便大模型理解当前具体发生了什么，但注意：这里描述里依然会出现"角度"这样的词，
    # 那是喂给大模型参考的原始数据，大模型自己回复时绝对不能照抄这些术语。
    user_message = (
        f"这位小学生刚刚完成了一次踢球动作，系统检测到他摆动腿触球瞬间的膝关节屈曲角度是 "
        f"{angle:.1f} 度，当前的三级容错诊断状态是「{status}」（Green 表示达标、"
        f"Yellow 表示接近达标、Red 表示明显偏离达标区间）。"
        f"请你根据这个情况，给这位孩子说一句符合三条铁律的正向引导语。"
    )

    try:
        # 发起一次标准的 Chat Completions 调用：
        #   model：使用 DeepSeek 的对话模型
        #   messages：分别传入 system（规则设定）与 user（本次具体情况）两条消息
        #   temperature=0.3：严格限制随机性，确保同样的输入基本会得到风格稳定、
        #                     教学法安全可靠的输出，避免大模型"天马行空"或"幻觉"
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )

        # 从返回结果中取出大模型生成的文本内容，并去除首尾多余的空白字符
        feedback_text = response.choices[0].message.content.strip()
        return feedback_text

    except Exception as exc:  # noqa: BLE001 - 网络/接口异常时需要兜底，不能让程序崩溃
        print(f"【llm_agent】调用 DeepSeek 接口失败，使用兜底提示语。错误信息：{exc}")
        # 兜底默认语句：即使大模型暂时调用失败，也依然保持"积极意图 + 具身隐喻"的风格，
        # 让孩子在网络异常时也不会看到任何生硬的报错信息
        return "再试一次吧，感觉一下腿像拉满的弓弦，稳稳地把力量送出去！"


# --------------------------------------------------------------------------
# 【v1.1 新增：前后端全栈联调】第三步半：整堂课/整次训练的综合诊断报告生成
# --------------------------------------------------------------------------

# 综合报告场景下的系统提示词：与 generate_feedback() 用的单帧提示词共享
# 同样的三大铁律，但视角从"某一帧的即时口头反馈"升级为"整次训练结束后，
# 面向教师/学生的结构化科研诊断报告"，因此额外要求大模型必须只输出
# 严格的 JSON（而不是一句话），方便后端直接解析后传给前端渲染。
REPORT_SYSTEM_PROMPT = """你是一名同时具备儿童运动心理学与运动生物力学背景的小学足球科研教练，\
现在需要根据系统后台真实采集到的「一整次训练/分析」中，摆动腿触球瞬间膝关节屈曲角度的\
红(Red)/黄(Yellow)/绿(Green)三级容错判定统计结果，生成一份面向该小学生（10-11岁）\
本人与授课教师共同查看的「本次综合练习诊断报告」。

你必须严格遵守以下三条铁律，一条都不能违反（这是本报告唯一合法的表达框架）：

【铁律一：积极意图原则】
- 绝对禁止出现任何评判性、惩罚性、否定性的字眼，比如"你做错了""动作错误""不对""失败""不好"。
- 必须使用信息性、建议性的表达方式，例如"下次可以试着……""再多一点点……就更棒了"。

【铁律二：具身隐喻原则】
- 绝对禁止使用任何抽象的生物力学/物理学专业术语，比如"角动量""屈曲度""关节角度""力矩""向量""夹角"。
- 必须把动作要点转译成孩子熟悉的、生动形象的身体感觉或生活化比喻。

【铁律三：单一焦点原则】
- painPoint 字段只聚焦当前统计数据中最值得关注的这一个点，不要同时罗列很多个知识点。

【输出格式要求（极其重要，必须严格遵守）】
- 只允许返回一个合法的 JSON 对象，绝对不能包含任何 Markdown 代码块标记（```）、\
解释性文字、前后缀说明。
- JSON 对象必须且只能包含以下三个字段：
    1. "score"：整数，0-100，代表本次训练的发力稳定性综合评分；
    2. "painPoint"：字符串，本次训练最主要的具身隐喻化痛点描述（遵守单一焦点原则）；
    3. "prescription"：字符串，给孩子/教练的下一步具身隐喻化训练处方建议。
- 两个字符串字段均为简体中文，语气温暖亲切，长度控制在 40-90 字之间。
"""


def _build_fallback_report(hit_stats, total_attempts):
    """当 DeepSeek 接口调用失败或返回内容解析失败时的规则化兜底报告，
    保证前端接口任何情况下都能拿到一份结构完整、语气安全的报告，不会因为
    网络异常而让联调流程中断。
    """
    safe_total = max(1, total_attempts)
    green = hit_stats.get("green", 0)
    yellow = hit_stats.get("yellow", 0)
    red = hit_stats.get("red", 0)

    score = max(35, min(98, round((green / safe_total) * 100 - red * 3)))

    if red >= yellow and red > 0:
        pain_point = f"本次练习中，有 {red} 次触球时腿部像还没完全拉满的弓弦，力量释放得有点着急。"
        prescription = "下次触球前，试着先感受一下摆动腿像弓弦一样慢慢蓄力，再稳稳地把力量送出去。"
    elif yellow > 0:
        pain_point = f"本次练习整体比较稳，只是有 {yellow} 次腿部蓄力的感觉差了那么一点点。"
        prescription = "可以在踢球前多给自己一点点准备时间，感受一下腿像小猫落地一样轻巧又扎实。"
    else:
        pain_point = "本次练习动作整体非常稳定，几乎每一次都保持住了很好的发力感觉。"
        prescription = "继续保持这种感觉，下次可以试着让摆动腿再多一点点加速冲力，会更棒！"

    return {"score": score, "painPoint": pain_point, "prescription": prescription}


def generate_session_report(hit_stats, student_number, sample_angles=None):
    """调用 DeepSeek 大模型，把「一整次训练」的红/黄/绿统计数据转译成结构化的
    综合诊断报告（score / painPoint / prescription 三个字段）。

    参数：
        hit_stats：dict，形如 {"green": 12, "yellow": 3, "red": 2}，
                   三级容错状态各自的命中次数统计（由 api_server.py 汇总后传入）。
        student_number：str，学生学号，仅用于让大模型的表达更有针对性（不会被
                         写进禁止事项，纯粹是上下文信息）。
        sample_angles：可选，list[float]，本次训练部分/全部有效采样角度，
                        用于让大模型对波动情况有更具体的感知（非必需字段）。

    返回：
        dict，包含 "score"（int）、"painPoint"（str）、"prescription"（str）
        三个字段。任何异常情况下都会返回结构完整的兜底数据，绝不抛出异常。
    """
    hit_stats = hit_stats or {}
    total_attempts = sum(hit_stats.get(k, 0) for k in ("green", "yellow", "red"))

    if total_attempts == 0:
        # 没有任何有效采样数据（例如分析刚开始就立刻结束），直接返回中性兜底报告
        return {
            "score": 0,
            "painPoint": "本次训练还没有采集到足够的有效触球数据，暂时无法给出具体的身体感觉建议。",
            "prescription": "别担心，下次多试几次触球动作，系统会帮你记录得更完整哦！",
        }

    angle_hint = ""
    if sample_angles:
        rounded = [round(float(a), 1) for a in sample_angles[:20]]
        angle_hint = f"部分采样到的原始角度数值（仅供你内部参考波动趋势，禁止在输出中提及）：{rounded}。"

    user_message = (
        f"学生学号：{student_number or '未填写'}。"
        f"本次训练系统一共采集到 {total_attempts} 次有效触球瞬间判定数据，"
        f"其中 Green（达标）{hit_stats.get('green', 0)} 次、"
        f"Yellow（接近达标）{hit_stats.get('yellow', 0)} 次、"
        f"Red（明显偏离）{hit_stats.get('red', 0)} 次。"
        f"{angle_hint}"
        f"请严格按照系统提示词规定的 JSON 格式，生成这次训练的综合诊断报告。"
    )

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
            messages=[
                {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw_text = response.choices[0].message.content.strip()
        parsed = json.loads(raw_text)

        score = int(parsed["score"])
        pain_point = str(parsed["painPoint"]).strip()
        prescription = str(parsed["prescription"]).strip()

        if not pain_point or not prescription:
            raise ValueError("DeepSeek 返回的报告字段为空")

        return {"score": max(0, min(100, score)), "painPoint": pain_point, "prescription": prescription}

    except Exception as exc:  # noqa: BLE001 - 网络异常/JSON 解析失败等都需要兜底
        print(f"【llm_agent】调用 DeepSeek 生成综合报告失败，使用规则化兜底报告。错误信息：{exc}")
        return _build_fallback_report(hit_stats, total_attempts)


# --------------------------------------------------------------------------
# 【v2.0 新增：延时反馈系统跨课时聚合复盘】第三步再半：
# 同一位学生连续 2~3 次尝试之间的「跨次趋势诊断」生成
# --------------------------------------------------------------------------

# 聚合诊断场景的系统提示词：视角从"单次训练的整体统计"升级为"同一位学生
# 连续 2~3 次尝试之间，评分/表现是如何变化的"，专门服务于「下节课前 5 分钟
# 结构化反思」环节，帮助教师快速判断该生是"越练越稳"还是"体力下降变形"。
AGGREGATE_SYSTEM_PROMPT = """你是一名同时具备儿童运动心理学与运动生物力学背景的小学足球科研教练，\
现在需要基于同一位小学生（10-11岁）在本节课内连续完成的 2~3 次踢球测试之间的评分变化趋势，\
撰写一份「跨次尝试聚合诊断建议」，帮助授课教师在下节课前 5 分钟内快速了解这位学生的进步/退步趋势，\
并给出针对下节课的具身隐喻化教学处方。

你必须严格遵守以下三条铁律，一条都不能违反：

【铁律一：积极意图原则】
- 绝对禁止出现任何评判性、惩罚性、否定性的字眼，比如"退步了""变差了""失败"。
- 必须使用信息性、建议性的表达方式，例如"后面几次能感觉到……""下次可以试着……"。

【铁律二：具身隐喻原则】
- 绝对禁止使用任何抽象的生物力学/物理学专业术语（角度、角速度、力矩、评分等数字本身）。
- 必须把变化趋势转译成孩子熟悉的身体感觉或生活化比喻。

【铁律三：单一焦点原则】
- trendDescription 只聚焦本次多趟测试中最值得关注的一个变化趋势（例如"后段体力下降造成的动作变形"\
或"越踢越稳、迅速自我纠错"），不要同时罗列多个知识点。

【输出格式要求（极其重要）】
- 只允许返回一个合法的 JSON 对象，绝对不能包含任何 Markdown 代码块标记（```）、解释性文字。
- JSON 对象必须且只能包含以下两个字段：
    1. "trendDescription"：字符串，40-100字，本次多趟测试之间的趋势总结（具身隐喻化）；
    2. "prescription"：字符串，40-100字，给教师下节课前的针对性处方建议。
- 两个字段均为简体中文，语气温暖亲切。
"""


def _build_fallback_aggregate_report(attempts_summary):
    """DeepSeek 接口调用失败或解析失败时的规则化兜底聚合诊断，保证接口
    任何情况下都能返回结构完整、语气安全的结果，不会因网络异常中断复盘流程。
    """
    scores = [a.get("score") for a in attempts_summary if isinstance(a.get("score"), (int, float))]

    if not scores:
        return {
            "trendDescription": "本节课暂未采集到足够的有效测试评分，无法总结多趟之间的变化趋势。",
            "prescription": "建议下节课至少完整完成 2 次踢球测试，方便系统生成聚合诊断建议。",
        }

    first_score, last_score = scores[0], scores[-1]
    if last_score > first_score + 3:
        trend = "从第一次到最后一次尝试，孩子的发力感觉越来越稳，就像慢慢找到了那种熟悉的节奏。"
        prescription = "下节课可以继续保持这个节奏，试着让摆动腿再多一点点自信的加速冲力。"
    elif last_score < first_score - 3:
        trend = "多次尝试到后段，稳定的感觉有一点点下降，可能是身体累积了一些疲惫，像弓弦稍微松了一点。"
        prescription = "下节课建议把练习拆成几个小组，中间穿插一点休息，帮孩子把每一次的蓄力感觉都找回来。"
    else:
        trend = "几次尝试的发力感觉整体保持得比较一致，没有出现明显的忽好忽坏。"
        prescription = "继续保持当前节奏即可，可以尝试给孩子加一点点新的挑战，比如稍微加快助跑速度。"

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
            "trendDescription": "本节课暂未采集到该生的任何尝试数据，无法总结跨次变化趋势。",
            "prescription": "请先完成至少一次踢球测试，再来查看聚合诊断报告。",
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
        f"该学生本节课一共完成了 {len(attempts_summary)} 次踢球测试，"
        f"具体每次测试的统计如下（内部参考数据，禁止在输出中提及任何具体数字）：\n"
        f"{attempts_text}\n"
        f"请严格按照系统提示词规定的 JSON 格式，生成这几次测试之间的跨次趋势诊断建议。"
    )

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
            messages=[
                {"role": "system", "content": AGGREGATE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
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

# 集体诊断场景的系统提示词：这次面向的读者是「授课教师/教练」而不是孩子本人，
# 因此不再需要"具身隐喻+积极意图"这两条面向学生的铁律（教师需要看懂真实的
# 生物力学术语才能制定教学计划），但依然要求语气专业、建议具体可执行，
# 输出严格的结构化 JSON，并且严禁生编造未提供的数据。
CLASS_PRESCRIPTION_SYSTEM_PROMPT = """你是一名拥有运动生物力学与青少年体育教学法双重背景的资深足球教研顾问，\
正在为一线体育教师撰写「班级集体教学诊断简报」。你会收到该班级全部历史测试记录中，各项\
生物力学错误分类的出现频率统计（百分比），请据此完成两件事：

1. "diagnosis"：用专业但易懂的语言，指出该班级当前最突出的1-2个集体性技术短板\
（可以直接使用生物力学术语，如"核心力量不足""支撑腿动态稳定性欠佳"等），并推测可能的\
成因（例如儿童此阶段核心稳定肌群发育尚不完全）。
2. "prescription"：给出3条左右具体、可在常规体育课45分钟内直接执行的下一步教学重点\
（例如具体的专项练习名称、每组次数/组数建议、动作要领提示）。

【输出格式要求（极其重要，必须严格遵守）】
- 只允许返回一个合法的 JSON 对象，绝对不能包含任何 Markdown 代码块标记（```）、解释性文字。
- JSON 对象必须且只能包含以下两个字段："diagnosis"（字符串，80-150字）、"prescription"\
（字符串，100-200字，可用"①②③"分点罗列）。
- 只能基于用户提供的统计数据进行专业推断，不允许编造用户未提及的具体数值。
- 必须使用简体中文。
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
            temperature=0.3,
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

INDIVIDUAL_SUMMARY_SYSTEM_PROMPT = """你是一名同时具备儿童运动心理学与运动生物力学背景的小学足球科研教练，\
现在需要基于某一位小学生（10-11岁）全周期（从第一次测试到最近一次测试）的历史评分与\
生物力学错误分类统计，为他/她撰写一份「个体纵向进化画像」，供教师与学生本人共同查看。

你必须严格遵守以下三条铁律（面向学生本人，语气必须温暖正向）：
【铁律一：积极意图原则】绝对禁止评判性/惩罚性字眼，只能用建议性表述。
【铁律二：具身隐喻原则】绝对禁止抽象力学术语，必须转译为孩子熟悉的身体隐喻。
【铁律三：单一焦点原则】每个字段只聚焦最突出的一点，不要罗列多个知识点。

【输出格式要求（极其重要）】
- 只允许返回一个合法的 JSON 对象，不能包含任何 Markdown 代码块标记、解释性文字。
- JSON 对象必须且只能包含以下两个字段：
    1. "strengths"：字符串，40-80字，该生最稳定的发力优势（具身隐喻化）；
    2. "weaknesses"：字符串，40-80字，该生最需要克服的习惯性盲区（具身隐喻化，且必须\
用积极建议性语气表达，不能出现否定性字眼）。
- 必须使用简体中文。
"""


def _build_fallback_individual_summary(scores, error_counter):
    """DeepSeek 接口调用失败或解析失败时的规则化兜底个体总结。"""
    if not scores:
        return {
            "strengths": "这位同学暂时还没有足够的历史测试数据，无法总结稳定的发力优势，多来试几次吧！",
            "weaknesses": "暂无足够数据总结需要关注的地方，继续加油完成更多测试即可。",
        }

    avg_score = sum(scores) / len(scores)
    strengths = (
        "整体发力节奏保持得很稳，像小马达一样持续输出稳定的力量。"
        if avg_score >= 70
        else "已经能感觉到腿部像弓弦一样开始蓄力，基础发力意识正在逐步建立。"
    )

    if error_counter:
        top_label = max(error_counter.items(), key=lambda item: item[1])[0]
        weakness_map = {
            "支撑脚位置偏离": "下次可以多注意一下支撑脚，试着像大树的根一样稳稳扎在地上再出脚。",
            "膝关节过度屈曲": "触球那一下可以再多给自己一点点缓冲时间，想象膝盖是被轻轻按了一下再弹出去。",
            "随摆转髋不足": "可以再多感受一下身体像陀螺一样转动的感觉，让摆动腿跟着身体一起转起来。",
            "身体重心偏移": "踢球前试着先找一下自己站得最稳的那个感觉，像小树苗一样把重心稳稳落住。",
        }
        weaknesses = weakness_map.get(top_label, "继续保持练习节奏，下次可以再多感受一下整体发力的连贯性。")
    else:
        weaknesses = "目前还没有发现特别明显的习惯性盲区，继续保持当前的练习节奏就很好！"

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
        f"该生历史记录中各项生物力学错误分类出现次数统计（内部参考数据，禁止在输出中提及具体数字）：{error_text}。"
        f"请严格按照系统提示词规定的 JSON 格式，生成这位学生的个体纵向进化画像总结。"
    )

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
            messages=[
                {"role": "system", "content": INDIVIDUAL_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
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
    # 直接运行本文件时，用一组示例数据快速测试大模型是否能正常返回结果
    test_angle = 118.6
    test_status = "Red"
    print(f"测试输入：angle={test_angle}, status={test_status}")
    print("正在调用 DeepSeek 大模型，请稍候……")
    result = generate_feedback(test_angle, test_status)
    print("大模型返回的中文指导语：")
    print(result)
