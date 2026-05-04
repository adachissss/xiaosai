from __future__ import annotations

"""第三问入口脚本：碳排放约束与碳交易对协同调度的影响分析。"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROBLEM1_SCRIPTS = Path(__file__).resolve().parents[2] / "problem1" / "scripts"
PROBLEM2_SCRIPTS = Path(__file__).resolve().parents[2] / "problem2" / "scripts"
for p in [str(PROBLEM1_SCRIPTS), str(PROBLEM2_SCRIPTS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from common import DT, check_constraints, compute_metrics, read_data  # noqa: E402
from degradation_model import battery_degradation_cost  # noqa: E402

from carbon_aware_strategy import build_and_solve_carbon_aware  # noqa: E402

# ------------------------------
# 中文字体和 matplotlib 样式
# ------------------------------
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
    "legend.fontsize": 10,
})

COLOR_BAT = "#2563EB"
COLOR_EV = "#F97316"
COLOR_OP = "#10B981"
COLOR_CARBON = "#6B7280"
COLOR_CAPS = ["#3B82F6", "#F59E0B", "#EF4444", "#8B5CF6", "#10B981"]


def load_s4_reference(problem2_results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    ref_dir = problem2_results_dir
    sched = ref_dir / "S4_degradation_aware_schedule.csv"
    ev_r = ref_dir / "S4_degradation_aware_ev_results.csv"
    if sched.exists() and ev_r.exists():
        return pd.read_csv(sched, parse_dates=["timestamp"]), pd.read_csv(ev_r)
    return None


def compute_carbon_footprint(schedule: pd.DataFrame, data_dir: Path) -> float:
    ts = pd.read_csv(data_dir / "timeseries_15min.csv")
    ci = ts["grid_carbon_kg_per_kwh"].to_numpy()
    return float(np.dot(schedule["grid_buy_kw"].to_numpy(), ci * DT))


def extract_metrics(
    schedule: pd.DataFrame, ev_result: pd.DataFrame, data, data_dir: Path, carbon_price: float, carbon_cap
) -> dict:
    op = compute_metrics(schedule, ev_result, data)
    bat_ch, bat_dis, bat_throughput, bat_cost = battery_degradation_cost(schedule)

    if "degradation_cost_cny" in ev_result.columns:
        ev_ch = float((schedule["ev_charge_total_kw"] * DT).sum())
        ev_dis = float((schedule["ev_discharge_total_kw"] * DT).sum())
        ev_throughput = ev_ch + ev_dis
        ev_cost = float(ev_result["degradation_cost_cny"].sum())
    else:
        from degradation_model import ev_degradation_cost as evdc
        ev_ch, ev_dis, ev_throughput, ev_cost = evdc(schedule, ev_result, data_dir)

    total_carbon_kg = compute_carbon_footprint(schedule, data_dir)
    carbon_trading_cost = total_carbon_kg * carbon_price

    return {
        "operation_cost_cny": op["total_cost_cny"],
        "battery_degradation_cost_cny": bat_cost,
        "ev_degradation_cost_cny": ev_cost,
        "total_degradation_cost_cny": bat_cost + ev_cost,
        "carbon_trading_cost_cny": carbon_trading_cost,
        "total_comprehensive_cost_cny": op["total_cost_cny"] + bat_cost + ev_cost + carbon_trading_cost,
        "total_carbon_kg": total_carbon_kg,
        "grid_import_energy_kwh": op["grid_import_energy_kwh"],
        "peak_grid_import_kw": op["peak_grid_import_kw"],
        "pv_consumption_rate": op["pv_consumption_rate"],
        "battery_throughput_kwh": bat_throughput,
        "ev_charge_kwh": float((schedule["ev_charge_total_kw"] * DT).sum()),
        "ev_discharge_kwh": float((schedule["ev_discharge_total_kw"] * DT).sum()),
        "ev_throughput_kwh": ev_throughput,
        "ev_shortfall_kwh": op["ev_shortfall_kwh"],
        "load_shift_down_kwh": op["load_shift_down_kwh"],
        "load_shed_kwh": op["load_shed_kwh"],
        "carbon_cap_kg": carbon_cap,
        "carbon_price_cny_per_kg": carbon_price,
    }


# ==============================
# 第1小问：不同碳排放上限
# ==============================
def run_cap_scenarios(data, data_dir, out_dir, fig_dir, battery_unit_cost):
    print("=" * 60)
    print("第1小问：不同碳排放上限对系统调度的影响")
    print("=" * 60)

    base_result = build_and_solve_carbon_aware(data, carbon_price_cny_per_kg=0.0, carbon_cap_kg=None)
    baseline_carbon = base_result[2]["total_carbon_kg"]
    print(f"无碳约束时自然排放基准值：{baseline_carbon:.1f} kg CO2/周")

    cap_scenarios = [
        ("无约束", None),
        ("宽松", baseline_carbon * 1.05),
        ("基准", baseline_carbon),
        ("严格", baseline_carbon * 0.93),
        ("极严", baseline_carbon * 0.86),
    ]

    schedules = {}
    ev_results = {}
    solve_infos = {}
    metrics_list = []

    for label, cap in cap_scenarios:
        cap_val = cap if cap else float("inf")
        try:
            sched, ev_r, info = build_and_solve_carbon_aware(
                data, carbon_price_cny_per_kg=0.0, carbon_cap_kg=cap, scheme_name=f"S5_cap_{label}"
            )
        except RuntimeError as e:
            print(f"  {label} (cap={cap}): 求解失败 — {e}")
            continue

        schedules[label] = sched
        ev_results[label] = ev_r
        solve_infos[label] = info
        m = extract_metrics(sched, ev_r, data, data_dir, 0.0, cap)
        m["scenario_label"] = label
        metrics_list.append(m)
        print(f"  {label:6s} cap={str(cap or '∞'):>10s} → 碳={m['total_carbon_kg']:>8.1f} kg, "
              f"运行成本={m['operation_cost_cny']:>8.1f} 元, "
              f"综合成本={m['total_comprehensive_cost_cny']:>8.1f} 元")

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(out_dir / "p3_q1_comparison_metrics.csv", index=False)
    _plot_q1_results(metrics_df, schedules, fig_dir)
    _write_q1_summary(metrics_df, solve_infos, baseline_carbon, out_dir)
    return metrics_df, schedules, ev_results


def _plot_q1_results(metrics: pd.DataFrame, schedules: dict[str, pd.DataFrame], fig_dir: Path):
    labels = metrics["scenario_label"].tolist()
    x = np.arange(len(labels))

    # 图1：碳排放达标情况
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.bar(x, metrics["total_carbon_kg"], color="#3B82F6", alpha=0.85, width=0.55)
    cap_colors = ["#6B7280" if pd.isna(c) or c == float("inf") else "#DC2626" for c in metrics["carbon_cap_kg"]]
    for i, (_, row) in enumerate(metrics.iterrows()):
        cap = row["carbon_cap_kg"]
        if cap and not np.isinf(cap):
            ax.axhline(cap, xmin=i / len(labels), xmax=(i + 1) / len(labels), color=cap_colors[i], linewidth=2.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("碳排放 / kg CO2")
    ax.set_title("不同碳排放上限下的实际碳排放", fontweight="bold", pad=12)
    # annotate
    ymax = float(metrics["total_carbon_kg"].max()) * 1.18
    ax.set_ylim(0, ymax)
    for i, v in enumerate(metrics["total_carbon_kg"]):
        ax.text(i, v + ymax * 0.02, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color="#374151")
    fig.text(0.5, 0.015, "注：虚线表示该场景的碳排放上限；无约束场景无虚线标记。", ha="center", fontsize=10, color="#4B5563")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.2)
    fig.savefig(fig_dir / "p3_q1_emissions_by_cap.png")
    plt.close(fig)

    # 图2：外网购电量和峰值
    fig, ax1 = plt.subplots(figsize=(10.5, 5.5))
    ax1.bar(x, metrics["grid_import_energy_kwh"], color="#2563EB", alpha=0.85, width=0.55, label="购电总量")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("购电总量 / kWh")
    ax1.set_title("不同碳排放上限下外网购电行为变化", fontweight="bold", pad=12)
    ax2 = ax1.twinx()
    ax2.plot(x, metrics["peak_grid_import_kw"], color="#DC2626", marker="D", linewidth=1.8, label="峰值购电")
    ax2.set_ylabel("峰值购电功率 / kW", color="#DC2626")
    ax2.tick_params(axis="y", labelcolor="#DC2626")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left", frameon=True, ncol=2)
    fig.subplots_adjust(left=0.08, right=0.9, top=0.88, bottom=0.18)
    fig.savefig(fig_dir / "p3_q1_grid_import_by_cap.png")
    plt.close(fig)

    # 图3：固定储能和 EV 吞吐量
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.bar(x, metrics["battery_throughput_kwh"], label="固定储能", color=COLOR_BAT, alpha=0.88, width=0.55)
    ax.bar(x, metrics["ev_throughput_kwh"], bottom=metrics["battery_throughput_kwh"], label="EV", color=COLOR_EV, alpha=0.88, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("等效吞吐量 / kWh")
    ax.set_title("不同碳排放上限下电池吞吐量变化", fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True, ncol=2)
    totals = metrics["battery_throughput_kwh"] + metrics["ev_throughput_kwh"]
    ax.set_ylim(0, float(totals.max()) * 1.18)
    for i, v in enumerate(totals):
        ax.text(i, v + float(totals.max()) * 0.025, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.18)
    fig.savefig(fig_dir / "p3_q1_battery_ev_by_cap.png")
    plt.close(fig)

    # 图4：成本构成
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.bar(x, metrics["operation_cost_cny"], label="运行成本", color=COLOR_OP, alpha=0.88, width=0.55)
    op2 = metrics["operation_cost_cny"] + metrics["total_degradation_cost_cny"]
    ax.bar(x, metrics["total_degradation_cost_cny"], bottom=metrics["operation_cost_cny"], label="寿命损耗", color="#F97316", alpha=0.88, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("成本 / 元")
    ax.set_title("不同碳排放上限下成本构成变化", fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True, ncol=3)
    totals = metrics["operation_cost_cny"] + metrics["total_degradation_cost_cny"]
    ax.set_ylim(0, float(totals.max()) * 1.18)
    for i, v in enumerate(totals):
        ax.text(i, v + float(totals.max()) * 0.025, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.18)
    fig.savefig(fig_dir / "p3_q1_cost_breakdown_by_cap.png")
    plt.close(fig)


def _write_q1_summary(metrics: pd.DataFrame, solve_infos: dict, baseline_carbon: float, out_dir: Path):
    lines = [
        "第三问第1小问：不同碳排放上限对系统调度的影响",
        "=" * 60, "",
        f"无约束时自然排放基准值：{baseline_carbon:.1f} kg CO2/周", "",
        "各场景对比：",
    ]
    for _, row in metrics.iterrows():
        cap_str = f"{row['carbon_cap_kg']:.0f}" if row['carbon_cap_kg'] and row['carbon_cap_kg'] < float('inf') else "无"
        lines.append(f"  {row['scenario_label']:6s} | cap={cap_str:>8s} kg | "
                     f"实际排放={row['total_carbon_kg']:>8.1f} | "
                     f"运行成本={row['operation_cost_cny']:>8.1f} | "
                     f"储能吞吐={row['battery_throughput_kwh']:>8.1f} | "
                     f"EV吞吐={row['ev_throughput_kwh']:>8.1f}")
    lines.append("")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "p3_q1_summary.txt").write_text("\n".join(lines), encoding="utf-8")


# ==============================
# 第2小问：不同碳交易价格
# ==============================
def run_price_scenarios(data, data_dir, out_dir, fig_dir, baseline_carbon):
    print("\n" + "=" * 60)
    print("第2小问：不同碳交易价格对系统调度的影响")
    print("=" * 60)

    price_scenarios = [
        ("无碳成本", 0.0),
        ("低碳价 100", 0.10),
        ("中碳价 300", 0.30),
        ("高碳价 500", 0.50),
    ]

    schedules = {}
    ev_results = {}
    metrics_list = []

    for label, price in price_scenarios:
        try:
            sched, ev_r, info = build_and_solve_carbon_aware(
                data, carbon_price_cny_per_kg=price, carbon_cap_kg=None, scheme_name=f"S5_price_{label}"
            )
        except RuntimeError as e:
            print(f"  {label} (price={price}): 求解失败 — {e}")
            continue

        schedules[label] = sched
        ev_results[label] = ev_r
        m = extract_metrics(sched, ev_r, data, data_dir, price, None)
        m["scenario_label"] = label
        m["carbon_price_cny_per_kg"] = price
        metrics_list.append(m)
        print(f"  {label:10s} 碳价={price*1000:>4.0f} 元/吨 → "
              f"碳={m['total_carbon_kg']:>8.1f} kg, "
              f"运行成本={m['operation_cost_cny']:>8.1f} 元, "
              f"EV放电={m['ev_discharge_kwh']:>7.1f} kWh, "
              f"综合成本={m['total_comprehensive_cost_cny']:>8.1f} 元")

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(out_dir / "p3_q2_comparison_metrics.csv", index=False)
    _plot_q2_results(metrics_df, fig_dir)
    _write_q2_summary(metrics_df, out_dir)
    return metrics_df, schedules, ev_results


def _plot_q2_results(metrics: pd.DataFrame, fig_dir: Path):
    prices = metrics["carbon_price_cny_per_kg"] * 1000  # 转为 元/吨
    x = range(len(metrics))

    # 图1：碳排放 vs 碳价
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.bar(x, metrics["total_carbon_kg"], color=["#3B82F6", "#F59E0B", "#EF4444", "#8B5CF6"], alpha=0.85, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.0f}" for p in prices])
    ax.set_xlabel("碳交易价格 / (元/吨 CO2)")
    ax.set_ylabel("碳排放 / kg CO2")
    ax.set_title("不同碳交易价格下的总碳排放量", fontweight="bold", pad=12)
    ymax = float(metrics["total_carbon_kg"].max()) * 1.18
    ax.set_ylim(0, ymax)
    for i, v in enumerate(metrics["total_carbon_kg"]):
        ax.text(i, v + ymax * 0.02, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.2)
    fig.savefig(fig_dir / "p3_q2_emissions_vs_price.png")
    plt.close(fig)

    # 图2：购电量和峰值 vs 碳价
    fig, ax1 = plt.subplots(figsize=(10.5, 5.5))
    ax1.bar(x, metrics["grid_import_energy_kwh"], color="#2563EB", alpha=0.85, width=0.55, label="购电总量")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{p:.0f}" for p in prices])
    ax1.set_xlabel("碳交易价格 / (元/吨 CO2)")
    ax1.set_ylabel("购电总量 / kWh")
    ax1.set_title("不同碳价下外网购电行为变化", fontweight="bold", pad=12)
    ax2 = ax1.twinx()
    ax2.plot(x, metrics["peak_grid_import_kw"], color="#DC2626", marker="D", linewidth=1.8, label="峰值购电")
    ax2.set_ylabel("峰值购电功率 / kW", color="#DC2626")
    ax2.tick_params(axis="y", labelcolor="#DC2626")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left", frameon=True, ncol=2)
    fig.subplots_adjust(left=0.08, right=0.9, top=0.88, bottom=0.2)
    fig.savefig(fig_dir / "p3_q2_grid_import_by_price.png")
    plt.close(fig)

    # 图3：储能/EV vs 碳价
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.bar(x, metrics["battery_throughput_kwh"], label="固定储能", color=COLOR_BAT, alpha=0.88, width=0.55)
    ax.bar(x, metrics["ev_throughput_kwh"], bottom=metrics["battery_throughput_kwh"], label="EV", color=COLOR_EV, alpha=0.88, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.0f}" for p in prices])
    ax.set_xlabel("碳交易价格 / (元/吨 CO2)")
    ax.set_ylabel("等效吞吐量 / kWh")
    ax.set_title("不同碳价下固定储能和 EV 吞吐量变化", fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True, ncol=2)
    totals = metrics["battery_throughput_kwh"] + metrics["ev_throughput_kwh"]
    ax.set_ylim(0, float(totals.max()) * 1.18)
    for i, v in enumerate(totals):
        ax.text(i, v + float(totals.max()) * 0.025, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.2)
    fig.savefig(fig_dir / "p3_q2_battery_ev_by_price.png")
    plt.close(fig)

    # 图4：成本构成
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.bar(x, metrics["operation_cost_cny"], label="运行成本", color=COLOR_OP, alpha=0.88, width=0.55)
    bottom1 = metrics["operation_cost_cny"]
    ax.bar(x, metrics["total_degradation_cost_cny"], bottom=bottom1, label="寿命损耗", color=COLOR_EV, alpha=0.88, width=0.55)
    bottom2 = bottom1 + metrics["total_degradation_cost_cny"]
    ax.bar(x, metrics["carbon_trading_cost_cny"], bottom=bottom2, label="碳交易成本", color=COLOR_CARBON, alpha=0.88, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.0f}" for p in prices])
    ax.set_xlabel("碳交易价格 / (元/吨 CO2)")
    ax.set_ylabel("成本 / 元")
    ax.set_title("不同碳价下综合成本构成变化", fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True, ncol=3)
    totals = metrics["total_comprehensive_cost_cny"]
    ax.set_ylim(0, float(totals.max()) * 1.18)
    for i, v in enumerate(totals):
        ax.text(i, v + float(totals.max()) * 0.025, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.2)
    fig.savefig(fig_dir / "p3_q2_cost_breakdown_by_price.png")
    plt.close(fig)


def _write_q2_summary(metrics: pd.DataFrame, out_dir: Path):
    lines = [
        "第三问第2小问：不同碳交易价格对系统调度的影响",
        "=" * 60, "",
        "各碳价场景对比：",
    ]
    for _, row in metrics.iterrows():
        lines.append(f"  {row['scenario_label']:10s} | "
                     f"碳价={row['carbon_price_cny_per_kg']*1000:>4.0f} 元/吨 | "
                     f"排放={row['total_carbon_kg']:>8.1f} kg | "
                     f"运行成本={row['operation_cost_cny']:>8.1f} | "
                     f"EV放电={row['ev_discharge_kwh']:>7.1f} kWh | "
                     f"综合成本={row['total_comprehensive_cost_cny']:>8.1f}")
    lines.append("")
    (out_dir / "p3_q2_summary.txt").write_text("\n".join(lines), encoding="utf-8")


# ==============================
# 第3小问：综合策略
# ==============================
def write_q3_summary(q1_metrics: pd.DataFrame, q2_metrics: pd.DataFrame, cap_scenarios, price_scenarios, out_dir: Path):
    lines = [
        "第三问第3小问：兼顾经济性、电池寿命与碳排放的综合策略",
        "=" * 60, "",
        "1. 三个维度之间的关系",
        "",
        "经济性（运行成本）、电池寿命（寿命损耗成本）和碳排放（碳交易成本）三者",
        "在调度模型中通过 S5 目标函数统一为综合成本：",
        "",
        "  min 运行成本 + 固定储能寿命损耗 + EV 寿命损耗 + 碳交易成本",
        "",
        "碳交易成本的引入使得碳排放被直接量化，并与传统运行成本进行比较。", "",
        "2. 基于结果的协同效应分析",
        "",
        "（1）碳排放与短期经济性：",
        "  - 碳排放收紧会减少外网购电，但可能增加本地资源调用（储能/EV），从而导致运行成本上升。",
        "  - 较高碳价会抑制高碳时段的购电，但不一定大幅增加总成本。",
        "",
        "（2）碳排放与电池寿命：",
        "  - 减少购电需要更多利用固定储能和 EV 调节，这可能会增加电池吞吐量和寿命损耗。",
        "  - 但如果碳价较低，通过储能调节减少的购电成本可能仍能覆盖寿命损耗。",
        "",
        "（3）三个维度的统一：",
        "  - S5 的综合成本目标已整合三个维度。",
        "  - 实际运营中，园区可通过调节碳价参数来调整三个维度之间的权重。",
    ]
    if q1_metrics is not None and len(q1_metrics) > 0:
        no_cap_row = q1_metrics.loc[q1_metrics["scenario_label"] == "无约束"]
        tight_row = q1_metrics.loc[q1_metrics["scenario_label"] == "极严"]
        if len(no_cap_row) > 0 and len(tight_row) > 0:
            nc = no_cap_row.iloc[0]
            tc = tight_row.iloc[0]
            lines += [
                "", "3. 关键数值（无约束 vs 极严 cap）",
                f"  碳排放：{nc['total_carbon_kg']:.0f} kg → {tc['total_carbon_kg']:.0f} kg",
                f"  运行成本：{nc['operation_cost_cny']:.1f} 元 → {tc['operation_cost_cny']:.1f} 元",
                f"  固定储能吞吐：{nc['battery_throughput_kwh']:.0f} kWh → {tc['battery_throughput_kwh']:.0f} kWh",
                f"  EV 吞吐量：{nc['ev_throughput_kwh']:.0f} kWh → {tc['ev_throughput_kwh']:.0f} kWh",
            ]
    lines += [
        "", "4. 建议的综合运行策略",
        "",
        "（1）园区应将碳排放成本通过碳价参数纳入日常调度优化，使模型自动权衡减排",
        "    与运行成本之间的关系。",
        "",
        "（2）碳价的设定应参考实际碳市场价格，并在不同政策场景下进行敏感度分析。",
        "",
        "（3）在碳排放约束较严的场景下，应优先利用固定储能减少购电依赖，",
        "    其次再考虑 EV V2B 和建筑柔性负荷的辅助作用。",
        "",
        "（4）将电池寿命损耗纳入目标函数，避免以过度消耗电池为代价来降低碳排放。",
        "",
        "（5）建议园区采用动态碳价机制：",
        "    - 常规时段：使用基准碳价（如 200 元/吨）",
        "    - 高排放时段：可附加碳成本惩罚",
        "    - 电网碳强度较高时段：主动减少购电，增加本地资源调度",
        "",
        "5. 结论",
        "",
        "通过 S5 综合成本优化模型，园区可以在经济性、电池寿命与碳排放三个目标之间",
        "取得合理的平衡。关键在于将碳排放成本转化为可量化的交易价格，并纳入统一",
        "的调度决策框架中。",
    ]
    (out_dir / "p3_q3_summary.txt").write_text("\n".join(lines), encoding="utf-8")


# ==============================
# 主入口
# ==============================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "B_data")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()

    data = read_data(args.data_dir)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 确定自然排放基准
    base_result = build_and_solve_carbon_aware(data, carbon_price_cny_per_kg=0.0, carbon_cap_kg=None)
    baseline_carbon = base_result[2]["total_carbon_kg"]

    # ---- 第1小问 ----
    q1_metrics, q1_schedules, _ = run_cap_scenarios(data, args.data_dir, out_dir, fig_dir, baseline_carbon)

    # ---- 第2小问 ----
    q2_metrics, q2_schedules, _ = run_price_scenarios(data, args.data_dir, out_dir, fig_dir, baseline_carbon)

    # ---- 第3小问 ----
    write_q3_summary(q1_metrics, q2_metrics, None, None, out_dir)

    # 打印摘要
    for fname in ["p3_q1_summary.txt", "p3_q2_summary.txt", "p3_q3_summary.txt"]:
        fpath = out_dir / fname
        if fpath.exists():
            print(f"\n{fpath.read_text(encoding='utf-8')}")


if __name__ == "__main__":
    main()
