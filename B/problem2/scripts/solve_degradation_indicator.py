from __future__ import annotations

"""第二问第 1 小问入口脚本。

该脚本不重新优化调度，而是读取第一问 S0/S1/S2/S3 的结果，
按照“等效吞吐量 × 单位寿命成本”的方法计算寿命损耗指标，
从而回答题目要求的“提出寿命损耗指标或成本折算方式”。
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

from degradation_model import (
    evaluate_scheme,
    load_ev_degradation_parameter_summary,
    load_stationary_battery_degradation_cost,
    load_problem1_scheme,
    result_to_dict,
)

# 用第一问已经生成的四个方案做试算，说明该寿命损耗指标如何落地计算。
SCHEMES = [
    "S0_no_storage",
    "S1_rule_storage",
    "S2_partial_coordination",
    "S3_full_coordination",
]

CHINESE_FONT = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc").get_name()

mpl.rcParams.update({
    "font.family": CHINESE_FONT,
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
})

SCHEME_LABELS = {
    "S0_no_storage": "S0\n朴素方案",
    "S1_rule_storage": "S1\n规则储能",
    "S2_partial_coordination": "S2\n部分协同",
    "S3_full_coordination": "S3\n完整协同",
}

COLOR_BATTERY = "#2563EB"
COLOR_EV = "#F97316"


def _style_stacked_bar(ax, labels: list[str], ylabel: str, title: str) -> None:
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=14)
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", frameon=True, ncol=2)


def _annotate_totals(ax, totals: pd.Series) -> None:
    ymax = float(totals.max())
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1)
    for i, value in enumerate(totals):
        ax.text(i, value + ymax * 0.025, f"{value:.1f}", ha="center", va="bottom", fontsize=9, color="#374151")


def plot_degradation_metrics(metrics: pd.DataFrame, out_dir: Path) -> None:
    """生成第 1 小问的两张核心图：吞吐量指标图和寿命成本折算图。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [SCHEME_LABELS[s] for s in metrics["scheme"]]
    x = range(len(metrics))

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.bar(x, metrics["battery_throughput_kwh"], label="固定储能", color=COLOR_BATTERY, alpha=0.88, width=0.58)
    ax.bar(x, metrics["ev_throughput_kwh"], bottom=metrics["battery_throughput_kwh"], label="电动车车队", color=COLOR_EV, alpha=0.88, width=0.58)
    _style_stacked_bar(ax, labels, "等效吞吐量 / kWh", "不同方案下固定储能与电动车电池等效吞吐量")
    _annotate_totals(ax, metrics["battery_throughput_kwh"] + metrics["ev_throughput_kwh"])
    fig.text(0.5, 0.015, "注：等效吞吐量 = 充电电量 + 放电电量，用于衡量电池使用强度。", ha="center", fontsize=10, color="#4B5563")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.18)
    fig.savefig(out_dir / "p2_q1_throughput_indicator.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.bar(x, metrics["battery_degradation_cost_cny"], label="固定储能", color=COLOR_BATTERY, alpha=0.88, width=0.58)
    ax.bar(x, metrics["ev_degradation_cost_cny"], bottom=metrics["battery_degradation_cost_cny"], label="电动车车队", color=COLOR_EV, alpha=0.88, width=0.58)
    _style_stacked_bar(ax, labels, "寿命损耗成本 / 元", "不同方案下电池寿命损耗成本折算")
    _annotate_totals(ax, metrics["total_degradation_cost_cny"])
    fig.text(0.5, 0.015, "注：寿命损耗成本 = 单位吞吐寿命成本 × 等效吞吐量。", ha="center", fontsize=10, color="#4B5563")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.18)
    fig.savefig(out_dir / "p2_q1_degradation_cost.png")
    plt.close(fig)


def write_summary(metrics: pd.DataFrame, ev_param_summary: pd.DataFrame, out_path: Path, battery_unit_cost: float) -> None:
    """输出文字摘要，方便后续直接查看第 1 小问的公式、参数和试算结果。"""

    lines = [
        "B题第二问第1小问：电池寿命损耗指标与成本折算",
        "=" * 60,
        "",
        "1. 指标选择",
        "采用线性等效吞吐量指标衡量电池寿命损耗：",
        "",
        "E_throughput = Σ(P_charge + P_discharge) × Δt",
        "C_degradation = c_deg × E_throughput",
        "",
        "其中 Δt = 0.25 h。该指标保持线性，便于后续直接加入线性规划目标函数。",
        "",
        "2. 参数设定",
        f"固定储能单位吞吐寿命成本取自 asset_parameters.csv：{battery_unit_cost:.4f} 元/kWh。",
        "EV 单位吞吐寿命成本使用 ev_sessions.csv 中的 degradation_cost_cny_per_kwh_throughput 字段。",
        "",
        "EV 寿命成本参数按车型统计：",
        ev_param_summary.to_string(index=False),
        "",
        "3. 基于第一问四方案的寿命损耗指标试算",
        metrics.to_string(index=False),
        "",
        "4. 结论",
        "该小问已经给出固定储能与 EV 电池寿命损耗的统一度量方式。",
        "后续第二问第2小问可将 battery_degradation_cost_cny 和 ev_degradation_cost_cny 加入目标函数，形成考虑寿命损耗的 S4_degradation_aware 调度方案。",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem1-results-dir", type=Path, default=Path(__file__).resolve().parents[2] / "problem1" / "results")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "B_data")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--battery-unit-cost", type=float, default=None)
    args = parser.parse_args()

    battery_unit_cost = args.battery_unit_cost
    if battery_unit_cost is None:
        battery_unit_cost = load_stationary_battery_degradation_cost(args.data_dir)

    # 逐个读取第一问结果，并把调度功率换算成寿命损耗指标。
    rows = []
    for scheme in SCHEMES:
        schedule, ev_result = load_problem1_scheme(args.problem1_results_dir, scheme)
        rows.append(result_to_dict(evaluate_scheme(scheme, schedule, ev_result, args.data_dir, battery_unit_cost)))
    metrics = pd.DataFrame(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 输出三类结果：指标表、EV 参数统计表、图片和文字摘要。
    ev_param_summary = load_ev_degradation_parameter_summary(args.data_dir)
    metrics.to_csv(args.out_dir / "p2_q1_degradation_metrics.csv", index=False)
    ev_param_summary.to_csv(args.out_dir / "p2_q1_ev_degradation_parameters.csv", index=False)
    plot_degradation_metrics(metrics, fig_dir)
    write_summary(metrics, ev_param_summary, args.out_dir / "p2_q1_summary.txt", battery_unit_cost)

    print((args.out_dir / "p2_q1_summary.txt").read_text(encoding="utf-8"))
    print(f"\nFigures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
