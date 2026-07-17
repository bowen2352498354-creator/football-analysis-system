# -*- coding: utf-8 -*-
"""
report_generator.py
v1.0 完整科研量产版 —— B组离线报告生成器（教练看板数据可视化）

功能说明：
    本脚本是一个【完全独立】的 Python 脚本，不依赖 pose_tracker.py 里的
    PyQt5 界面代码，可以在训练课结束后单独运行，用来把 B 组"延时反馈"
    实验在后台静默采集、保存在本地 B_group_data_log.json 里的历史数据，
    自动渲染成一张适合教练在"次课前 5 分钟结构化反思"环节使用的
    综合分析图表（B_group_training_report.png）。

    读取的数据文件由 pose_tracker.py 的 VideoWorker._flush_b_group_data_to_disk()
    方法负责生成，每一条记录形如：
        {"timestamp": 1731234567.123, "knee_angle": 142.3, "status": "Green"}

图表布局（上下两个子图）：
    子图一（上）：动作离散度与合规占比——统计右膝屈曲角度被判定为
                  Green（达标）/ Yellow（接近）/ Red（错误）的次数与
                  绝对百分比，用柱状图直观呈现，柱子上方标注具体数值。
    子图二（下）：纵向生长与稳定性——以数据采样帧序号为 X 轴、
                  实测右膝角度为 Y 轴，绘制角度波动折线图，并用
                  ax.axhspan 在 Y 轴 140°~160° 区间画出浅绿色半透明
                  阴影，醒目标出"儿童动作发展合规标准带"，让教练和
                  学生能一眼看清动作脱离合规区间的具体时机。

运行方式：
    1. 先确保已安装可视化依赖：
           pip install matplotlib
       （numpy 在 pose_tracker.py 的依赖里通常已经装好，若没有请一并安装：
           pip install numpy）
    2. 确保本脚本与 B_group_data_log.json 放在同一目录下
       （两者都以 SCRIPT_DIR 为基准路径，与 pose_tracker.py 保持一致）。
    3. 直接运行：
           python report_generator.py
       运行成功后会在同目录下生成 B_group_training_report.png。
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# 第〇步：路径常量（与 pose_tracker.py 保持一致，方便两个脚本配套使用）
# --------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# B组后台静默采集的原始数据文件（由 pose_tracker.py 生成）
B_GROUP_LOG_PATH = os.path.join(SCRIPT_DIR, "B_group_data_log.json")

# 本脚本最终生成的高清综合分析图表
REPORT_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "B_group_training_report.png")

# 图表分辨率：教练看板/打印材料要求达到 300 DPI 的印刷级清晰度
REPORT_DPI = 300

# 三级判定状态对应的显示颜色（与 pose_tracker.py 里的红黄绿容错框颜色语义一致，
# 这里用 matplotlib 认识的颜色名称/十六进制色值，而不是 pose_tracker.py 里的 BGR 元组）
STATUS_COLORS = {
    "Green": "#2e7d32",
    "Yellow": "#f9a825",
    "Red": "#c62828",
}
STATUS_ORDER = ["Green", "Yellow", "Red"]
STATUS_LABELS_CN = {
    "Green": "达标 (Green)",
    "Yellow": "接近 (Yellow)",
    "Red": "错误 (Red)",
}

# 儿童动作发展合规标准带（与 project_plan.md《核心生物力学诊断参数》一致）
COMPLIANT_BAND_LOW = 140
COMPLIANT_BAND_HIGH = 160


# --------------------------------------------------------------------------
# 第一步：中文字体配置（解决 matplotlib 默认字体不支持中文而显示乱码方块的问题）
# --------------------------------------------------------------------------

def configure_chinese_font():
    """依次尝试常见的中文字体名称，配置给 matplotlib 全局使用。

    Windows 系统通常自带"Microsoft YaHei"（微软雅黑）/"SimHei"（黑体），
    这里把它们和几个常见的跨平台候选字体一起列进候选列表，matplotlib 的
    font.sans-serif 支持传入一个"优先级列表"，会自动使用列表中第一个
    在当前系统里能找到的字体，找不到再往后尝试，兼容性最好。
    """
    candidate_fonts = [
        "Microsoft YaHei",  # 微软雅黑（Windows 系统首选，字形清晰美观）
        "SimHei",           # 黑体（Windows 系统备选）
        "PingFang SC",      # macOS 系统常见中文字体
        "Noto Sans CJK SC", # Linux 系统常见中文字体
        "Arial Unicode MS",
    ]
    matplotlib.rcParams["font.sans-serif"] = candidate_fonts
    # 解决保存图片时坐标轴负号（如 -10）被显示成方块乱码的经典问题
    matplotlib.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------------------------------
# 第二步：读取并校验 B 组历史数据
# --------------------------------------------------------------------------

def load_b_group_records():
    """读取 B_group_data_log.json，返回一个记录列表（每条是一个 dict）。

    如果文件不存在或内容为空/格式不对，直接抛出异常并给出清晰的中文提示，
    避免生成一张空白或报错的图表误导教练。
    """
    if not os.path.exists(B_GROUP_LOG_PATH):
        raise FileNotFoundError(
            f"未找到 B 组数据文件：{B_GROUP_LOG_PATH}\n"
            f"请先用 pose_tracker.py 以「B组—延时反馈」模式完整跑完一次训练，"
            f"确保后台已经静默采集并保存过数据。"
        )

    with open(B_GROUP_LOG_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or len(records) == 0:
        raise ValueError(
            f"B 组数据文件内容为空或格式异常：{B_GROUP_LOG_PATH}\n"
            f"请检查该文件是否被意外清空或损坏。"
        )

    return records


# --------------------------------------------------------------------------
# 第三步：子图一——动作离散度与合规占比（柱状图）
# --------------------------------------------------------------------------

def draw_status_distribution_subplot(ax, records):
    """在给定的 Axes 上画出 Green/Yellow/Red 三种判定状态的次数柱状图，
    每根柱子上方额外标注"次数 + 绝对百分比"，方便教练快速掌握整体合规情况。
    """
    total_count = len(records)

    # 统计三种状态各自出现的次数（哪怕某一种状态本次训练一次都没出现，
    # 也要在图表里显示成 0，而不是直接从图上消失，保持柱状图结构稳定）
    status_counts = {status: 0 for status in STATUS_ORDER}
    for record in records:
        status = record.get("status")
        if status in status_counts:
            status_counts[status] += 1

    counts = [status_counts[status] for status in STATUS_ORDER]
    percentages = [count / total_count * 100 for count in counts]
    bar_colors = [STATUS_COLORS[status] for status in STATUS_ORDER]
    bar_labels = [STATUS_LABELS_CN[status] for status in STATUS_ORDER]

    bars = ax.bar(bar_labels, counts, color=bar_colors, width=0.5, edgecolor="white")

    # 在每根柱子正上方标注"次数（百分比）"，让教练不用换算就能直接读数
    for bar, count, percentage in zip(bars, counts, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total_count * 0.02,
            f"{count} 次\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title(
        f"子图一：右膝屈曲角度判定分布与合规占比（本次训练共 {total_count} 个有效采样点）",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("出现次数")
    ax.set_ylim(0, max(counts) * 1.25 if max(counts) > 0 else 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


# --------------------------------------------------------------------------
# 第四步：子图二——纵向生长与稳定性（时间序列折线图 + 合规标准带）
# --------------------------------------------------------------------------

def draw_angle_timeseries_subplot(ax, records):
    """在给定的 Axes 上画出右膝关节角度随采样帧序号变化的折线图，
    并用浅绿色半透明阴影标出 140°~160° 的"儿童动作发展合规标准带"。
    """
    # X 轴用"数据采样帧序号"（即记录在列表里的顺序，从 1 开始编号），
    # 比直接用 Unix 时间戳更直观，教练看图表时不需要换算成分钟秒数。
    frame_indices = np.arange(1, len(records) + 1)
    knee_angles = [record.get("knee_angle", 0.0) for record in records]

    # 核心设计要求：用 ax.axhspan 在 Y 轴 140°~160° 区间画出醒目的
    # 浅绿色半透明阴影，代表"儿童动作发展合规标准带"，必须画在折线的下面
    # （zorder 更低），这样折线穿过标准带时依然清晰可见。
    ax.axhspan(
        COMPLIANT_BAND_LOW,
        COMPLIANT_BAND_HIGH,
        facecolor="#90ee90",   # 浅绿色
        alpha=0.35,             # 半透明
        zorder=0,
        label=f"儿童动作发展合规标准带（{COMPLIANT_BAND_LOW}°~{COMPLIANT_BAND_HIGH}°）",
    )

    ax.plot(
        frame_indices,
        knee_angles,
        color="#1565c0",
        linewidth=1.6,
        marker="o",
        markersize=3,
        zorder=2,
        label="实测右膝关节角度",
    )

    # 额外用红/黄/绿三色小圆点把每个采样点按判定状态再着色一次，
    # 让教练除了看折线走势，也能一眼定位到具体哪几个点是 Red/Yellow。
    for status in STATUS_ORDER:
        status_frame_indices = [
            idx for idx, record in zip(frame_indices, records) if record.get("status") == status
        ]
        status_angles = [
            record.get("knee_angle", 0.0) for record in records if record.get("status") == status
        ]
        if status_frame_indices:
            ax.scatter(
                status_frame_indices,
                status_angles,
                color=STATUS_COLORS[status],
                s=18,
                zorder=3,
                label=f"判定为 {STATUS_LABELS_CN[status]} 的采样点",
            )

    ax.set_title("子图二：右膝关节角度纵向波动趋势与合规标准带对照", fontsize=13, fontweight="bold")
    ax.set_xlabel("数据采样帧序号")
    ax.set_ylabel("右膝关节角度（°）")
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)


# --------------------------------------------------------------------------
# 第五步：主流程——读取数据、绘图、保存 300 DPI 高清图表
# --------------------------------------------------------------------------

def generate_report():
    """完整的报告生成主流程：配置中文字体 -> 读取数据 -> 绘制上下两个子图 ->
    保存为 300 DPI 高清 PNG 文件。
    """
    configure_chinese_font()

    print(f"正在读取 B 组历史数据：{B_GROUP_LOG_PATH}")
    records = load_b_group_records()
    print(f"读取成功，共 {len(records)} 条有效采样记录，开始生成图表……")

    # 上下两个子图布局：figsize 按 A4 纵向比例适当放大，保证 300 DPI 导出后
    # 依然有充足的像素细节，不会因为放大打印而糊掉。
    fig, (ax_top, ax_bottom) = plt.subplots(
        nrows=2, ncols=1, figsize=(11, 12), gridspec_kw={"height_ratios": [1, 1.3]}
    )

    fig.suptitle(
        "B组（延时反馈）训练综合分析报告 —— 教练看板", fontsize=16, fontweight="bold"
    )

    draw_status_distribution_subplot(ax_top, records)
    draw_angle_timeseries_subplot(ax_bottom, records)

    fig.tight_layout(rect=(0, 0, 1, 0.96))  # 给顶部大标题留出空间，避免与子图标题重叠

    fig.savefig(REPORT_OUTPUT_PATH, dpi=REPORT_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"图表生成完成！已保存至：{REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    generate_report()
