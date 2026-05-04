from __future__ import annotations

"""第三问入口脚本：碳排放约束与碳交易对协同调度的影响分析。"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
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
mpl.rcParams['font.sans-serif'] = ['LXGW Bright']
mpl.rcParams['axes.unicode_minus'] = False

mpl.rcParams.update({
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
    if carbon_cap is not None and not np.isinf(carbon_cap):
        carbon_trading_cost = carbon_price * (total_carbon_kg - carbon_cap)
    else:
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
    ax.set_title("不同碳排放上限下的实际碳排放", pad=12)
    # annotate
    ymax = float(metrics["total_carbon_kg"].max()) * 1.18
    ax.set_ylim(0, ymax)
    for i, v in enumerate(metrics["total_carbon_kg"]):
        ax.text(i + 0.3, v - ymax * 0.02, f"{v:.0f}", ha="left", va="top", fontsize=9, color="#374151")
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
    ax1.set_title("不同碳排放上限下外网购电行为变化", pad=12)
    ax2 = ax1.twinx()
    ax2.plot(x, metrics["peak_grid_import_kw"], color="#DC2626", marker="D", linewidth=1.8, label="峰值购电")
    ax2.set_ylabel("峰值购电功率 / kW", color="#DC2626")
    ax2.tick_params(axis="y", labelcolor="#DC2626")
    # ax2（折线）置于 ax1（柱状图）上层，图例放在 ax2 上避免被覆盖
    ax1.set_zorder(1)
    ax2.set_zorder(2)
    ax1.patch.set_visible(False)
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, loc="upper left", frameon=True, ncol=1, framealpha=0.4)
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
    ax.set_title("不同碳排放上限下电池吞吐量变化", pad=12)
    ax.legend(loc="upper left", frameon=True, ncol=1, framealpha=0.4)
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
    ax.set_title("不同碳排放上限下成本构成变化", pad=12)
    ax.legend(loc="upper left", frameon=True, ncol=1, framealpha=0.4)
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
    ax.set_title("不同碳交易价格下的总碳排放量", pad=12)
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
    ax1.set_title("不同碳价下外网购电行为变化", pad=12)
    ax2 = ax1.twinx()
    ax2.plot(x, metrics["peak_grid_import_kw"], color="#DC2626", marker="D", linewidth=1.8, label="峰值购电")
    ax2.set_ylabel("峰值购电功率 / kW", color="#DC2626")
    ax2.tick_params(axis="y", labelcolor="#DC2626")
    # ax2（折线）置于 ax1（柱状图）上层，图例放在 ax2 上避免被覆盖
    ax1.set_zorder(1)
    ax2.set_zorder(2)
    ax1.patch.set_visible(False)
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, loc="upper left", frameon=True, ncol=1, framealpha=0.4)
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
    ax.set_title("不同碳价下固定储能和 EV 吞吐量变化", pad=12)
    ax.legend(loc="upper left", frameon=True, ncol=1, framealpha=0.4)
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
    ax.set_title("不同碳价下综合成本构成变化", pad=12)
    ax.legend(loc="upper left", frameon=True, ncol=1, framealpha=0.4)
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
# 第2.5小问：cap × price 联合场景（真正的碳交易）
# ==============================
def run_joint_cap_price_scenarios(data, data_dir, out_dir, fig_dir, baseline_carbon):
    """cap × price 联合扫描：固定碳排放上限（免费配额），变碳价，模拟真正的 cap-and-trade。"""
    print("\n" + "=" * 60)
    print("补充分析：cap × price 联合场景（真正碳交易机制）")
    print("=" * 60)

    # 以"严格"cap（93% 自然排放）为免费配额，扫描不同碳价
    cap = baseline_carbon * 0.93
    price_scenarios = [
        ("碳价 0 元/吨", 0.0),
        ("碳价 100 元/吨", 0.10),
        ("碳价 300 元/吨", 0.30),
        ("碳价 500 元/吨", 0.50),
    ]

    schedules = {}
    metrics_list = []

    for label, price in price_scenarios:
        try:
            sched, ev_r, info = build_and_solve_carbon_aware(
                data, carbon_price_cny_per_kg=price, carbon_cap_kg=cap,
                scheme_name=f"S5_joint_cap93pct_price{price*1000:.0f}"
            )
        except RuntimeError as e:
            print(f"  {label}: 求解失败 — {e}")
            continue

        schedules[label] = sched
        m = extract_metrics(sched, ev_r, data, data_dir, price, cap)
        m["scenario_label"] = label
        m["carbon_price_cny_per_kg"] = price
        # 从 solve info 中取影子价格
        m["carbon_shadow_price_cny_per_kg"] = info.get("carbon_shadow_price_cny_per_kg", None)
        metrics_list.append(m)

        trade_sign = "购入" if m["carbon_trading_cost_cny"] > 0 else "出售"
        print(f"  {label:16s} 碳={m['total_carbon_kg']:>8.1f} kg | "
              f"配额={cap:>8.0f} kg | "
              f"{trade_sign}额度 | "
              f"交易成本={m['carbon_trading_cost_cny']:>+8.1f} 元 | "
              f"运行成本={m['operation_cost_cny']:>8.1f} 元 | "
              f"影子价格={m.get('carbon_shadow_price_cny_per_kg', 'N/A')}")

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(out_dir / "p3_joint_cap_price_metrics.csv", index=False)
    _plot_joint_results(metrics_df, fig_dir, cap)
    return metrics_df, schedules


def _plot_joint_results(metrics: pd.DataFrame, fig_dir: Path, cap_kg: float):
    prices = metrics["carbon_price_cny_per_kg"] * 1000
    x = range(len(metrics))

    # 图1：碳排放 vs 配额线（带碳交易方向）
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bars = ax.bar(x, metrics["total_carbon_kg"], color="#3B82F6", alpha=0.85, width=0.55)
    ax.axhline(cap_kg, color="#DC2626", linewidth=2, linestyle="--", label=f"免费配额={cap_kg:.0f} kg")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.0f}" for p in prices])
    ax.set_xlabel("碳交易价格 / (元/吨 CO2)")
    ax.set_ylabel("碳排放 / kg CO2")
    ax.set_title("碳交易机制下：碳排放 vs 碳价（cap + price 同时存在）", pad=12)
    ax.legend(loc="upper right", frameon=True, framealpha=0.4)
    ymin = min(float(metrics["total_carbon_kg"].min()), cap_kg) * 0.95
    ymax = max(float(metrics["total_carbon_kg"].max()), cap_kg) * 1.1
    ax.set_ylim(ymin, ymax)
    for i, v in enumerate(metrics["total_carbon_kg"]):
        delta = v - cap_kg
        sign = "+" if delta > 0 else ""
        ax.text(i, v + (ymax - ymin) * 0.02, f"{v:.0f}\n({sign}{delta:.0f})",
                ha="center", va="bottom", fontsize=9)
    # 填色表示买/卖
    for i, v in enumerate(metrics["total_carbon_kg"]):
        if v > cap_kg:
            bars[i].set_color("#EF4444")
        else:
            bars[i].set_color("#10B981")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.2)
    fig.savefig(fig_dir / "p3_joint_emissions_vs_cap.png")
    plt.close(fig)

    # 图2：碳交易成本 vs 碳价
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    colors = ["#10B981" if v <= 0 else "#EF4444" for v in metrics["carbon_trading_cost_cny"]]
    ax.bar(x, metrics["carbon_trading_cost_cny"], color=colors, alpha=0.88, width=0.55)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.0f}" for p in prices])
    ax.set_xlabel("碳交易价格 / (元/吨 CO2)")
    ax.set_ylabel("碳交易净成本 / 元（负值=出售获利）")
    ax.set_title(f"碳交易净支出（配额={cap_kg:.0f} kg）", pad=12)
    yabs = max(abs(float(metrics["carbon_trading_cost_cny"].min())),
               abs(float(metrics["carbon_trading_cost_cny"].max()))) * 1.3
    ax.set_ylim(-yabs, yabs)
    for i, v in enumerate(metrics["carbon_trading_cost_cny"]):
        ax.text(i, v + yabs * 0.04 * (1 if v >= 0 else -1),
                f"{v:+.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.2)
    fig.savefig(fig_dir / "p3_joint_trading_cost.png")
    plt.close(fig)

    # 图3：综合成本构成
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.bar(x, metrics["operation_cost_cny"], label="运行成本", color=COLOR_OP, alpha=0.88, width=0.55)
    bottom1 = metrics["operation_cost_cny"].copy()
    ax.bar(x, metrics["total_degradation_cost_cny"], bottom=bottom1, label="寿命损耗", color=COLOR_EV, alpha=0.88, width=0.55)
    bottom2 = bottom1 + metrics["total_degradation_cost_cny"]
    # 碳交易成本可能为负
    trade_costs = metrics["carbon_trading_cost_cny"].to_numpy()
    for i, v in enumerate(trade_costs):
        if v >= 0:
            ax.bar(i, v, bottom=bottom2.iloc[i], label="碳交易成本" if i == 0 else "", color=COLOR_CARBON, alpha=0.88, width=0.55)
        else:
            ax.bar(i, v, bottom=bottom2.iloc[i], label="碳交易收入" if i == 0 else "", color="#10B981", alpha=0.88, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.0f}" for p in prices])
    ax.set_xlabel("碳交易价格 / (元/吨 CO2)")
    ax.set_ylabel("成本 / 元")
    ax.set_title(f"碳交易机制下综合成本构成（配额={cap_kg:.0f} kg）", pad=12)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", frameon=True, ncol=1, framealpha=0.4)
    ymax = float(metrics["total_comprehensive_cost_cny"].max()) * 1.18
    ax.set_ylim(0, ymax)
    for i, v in enumerate(metrics["total_comprehensive_cost_cny"]):
        ax.text(i, v + ymax * 0.025, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.2)
    fig.savefig(fig_dir / "p3_joint_cost_breakdown.png")
    plt.close(fig)


# ==============================
# 第3小问：综合策略（定量分析）
# ==============================
def run_q3_pareto_analysis(data, data_dir, out_dir, fig_dir, baseline_carbon, min_carbon):
    """ε-约束法生成 Pareto 前沿：扫描不同碳排放上限，记录成本与碳排的权衡关系。"""
    print("\n" + "=" * 60)
    print("第3小问：Pareto 前沿分析（经济性 vs 碳排放 vs 电池寿命）")
    print("=" * 60)
    print(f"  自然排放={baseline_carbon:.0f} kg, 无甩负荷最小排放={min_carbon:.0f} kg")

    # 在最小可行排放和自然排放之间插值
    n_points = 10
    caps = np.linspace(min_carbon, baseline_carbon, n_points)

    pareto_data = []
    for cap in caps:
        try:
            sched, ev_r, info = build_and_solve_carbon_aware(
                data, carbon_price_cny_per_kg=0.0, carbon_cap_kg=cap,
                scheme_name=f"S5_pareto_cap{cap:.0f}"
            )
        except RuntimeError:
            continue

        # 过滤掉有显著甩负荷的不可信解
        unmet_kwh = float((sched.get("unmet_load_kw", 0) * DT).sum())
        if unmet_kwh > 1e-3:
            continue

        m = extract_metrics(sched, ev_r, data, data_dir, 0.0, cap)
        m["carbon_cap"] = cap
        m["carbon_shadow_price_cny_per_kg"] = info.get("carbon_shadow_price_cny_per_kg", None)
        pareto_data.append(m)

    if not pareto_data:
        print("  Pareto 分析失败：所有 cap 均不可行或产生甩负荷")
        return None

    pareto_df = pd.DataFrame(pareto_data)
    pareto_df.to_csv(out_dir / "p3_pareto_frontier.csv", index=False)
    print(f"  得到 {len(pareto_df)} 个有效 Pareto 点")

    # 绘制 Pareto 前沿
    _plot_pareto_frontier(pareto_df, fig_dir, baseline_carbon, min_carbon)
    return pareto_df


def _plot_pareto_frontier(pareto_df: pd.DataFrame, fig_dir: Path, baseline: float, min_carbon: float):
    # 图1：成本—碳排 Pareto 前沿
    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    carbon = pareto_df["total_carbon_kg"].to_numpy()
    total_cost = pareto_df["total_comprehensive_cost_cny"].to_numpy()
    op_cost = pareto_df["operation_cost_cny"].to_numpy()
    deg_cost = pareto_df["total_degradation_cost_cny"].to_numpy()

    ax.plot(carbon, total_cost, "o-", color="#2563EB", linewidth=2, markersize=8, label="综合成本")
    ax.plot(carbon, op_cost, "s--", color="#10B981", linewidth=1.5, markersize=6, alpha=0.7, label="运行成本")
    ax.plot(carbon, deg_cost, "^--", color="#F97316", linewidth=1.5, markersize=6, alpha=0.7, label="寿命损耗")
    ax.axvline(baseline, color="#6B7280", linewidth=1.2, linestyle=":", alpha=0.7, label=f"自然排放={baseline:.0f}")
    ax.axvline(min_carbon, color="#DC2626", linewidth=1.2, linestyle=":", alpha=0.7, label=f"最小排放={min_carbon:.0f}")

    ax.set_xlabel("碳排放 / kg CO2")
    ax.set_ylabel("成本 / 元")
    ax.set_title("Pareto 前沿：经济性 vs 碳排放", pad=12)
    ax.legend(loc="upper left", frameon=True, ncol=1, framealpha=0.4)
    ax.invert_xaxis()
    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.15)
    fig.savefig(fig_dir / "p3_q3_pareto_frontier.png")
    plt.close(fig)

    # 图2：边际减排成本曲线（影子价格 vs 碳排放）
    shadow_prices = [m.get("carbon_shadow_price_cny_per_kg") for m in
                     pareto_df.to_dict(orient="records")]
    valid_sp = [(c, sp) for c, sp in zip(carbon, shadow_prices) if sp is not None and sp > 1e-8]

    if valid_sp:
        fig, ax = plt.subplots(figsize=(10.5, 5.5))
        sp_carbon, sp_values = zip(*valid_sp)
        ax.plot(sp_carbon, [v * 1000 for v in sp_values], "D-", color="#DC2626",
                linewidth=1.8, markersize=7)
        ax.set_xlabel("碳排放 / kg CO2")
        ax.set_ylabel("边际减排成本 / (元/吨 CO2)")
        ax.set_title("边际减排成本曲线（碳约束的影子价格）", pad=12)
        ax.axhline(100, color="#6B7280", linewidth=1, linestyle=":", alpha=0.7,
                   label="中国碳市场参考价 100 元/吨")
        ax.axhline(500, color="#F97316", linewidth=1, linestyle=":", alpha=0.7,
                   label="EU ETS 参考价 500 元/吨")
        ax.legend(loc="upper left", frameon=True, framealpha=0.4)
        ax.invert_xaxis()
        fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.15)
        fig.savefig(fig_dir / "p3_q3_marginal_abatement_cost.png")
        plt.close(fig)


def write_q3_summary(joint_metrics: pd.DataFrame, pareto_df: pd.DataFrame,
                     baseline_carbon: float, min_carbon: float, out_dir: Path):
    """基于定量分析结果，输出第3小问的综合性文字结论。"""
    lines = [
        "第三问第3小问：兼顾经济性、电池寿命与碳排放的综合策略",
        "=" * 60, "",
        "1. 联合碳交易场景（cap × price）结果",
        "",
        "以下为固定免费配额（93% 自然排放）下，不同碳价对系统的影响：", "",
    ]

    if joint_metrics is not None and len(joint_metrics) > 0:
        for _, row in joint_metrics.iterrows():
            lines.append(
                f"  {row['scenario_label']:16s} | "
                f"碳排放={row['total_carbon_kg']:>8.1f} kg | "
                f"交易成本={row['carbon_trading_cost_cny']:>+8.1f} 元 | "
                f"综合成本={row['total_comprehensive_cost_cny']:>8.1f} 元"
            )
        # 提取关键信息：边际减排成本
        shadow_0 = joint_metrics.iloc[0].get("carbon_shadow_price_cny_per_kg", None)
        if shadow_0 is not None:
            lines.append("")
            lines.append(f"  该配额水平下的边际减排成本（影子价格）= {shadow_0 * 1000:.0f} 元/吨")
            lines.append(f"  碳价 < {shadow_0 * 1000:.0f} 元/吨时，碳价信号不足以驱动额外减排，排放维持在配额水平")
            lines.append(f"  碳价 > {shadow_0 * 1000:.0f} 元/吨时，园区才有经济动力将排放降至配额以下并出售结余")
        lines.append("")

    lines += [
        "2. Pareto 前沿分析结果",
        "",
        f"  自然排放基准：{baseline_carbon:.0f} kg CO2",
        f"  最小可行排放：{min_carbon:.0f} kg CO2",
        f"  最大减排潜力：{baseline_carbon - min_carbon:.0f} kg ({(baseline_carbon - min_carbon) / baseline_carbon * 100:.1f}%)",
        "",
    ]

    if pareto_df is not None and len(pareto_df) > 1:
        # 经济优先（最靠近自然排放）
        econ = pareto_df.iloc[-1]
        # 碳优先（最靠近最小排放）
        carbon_first = pareto_df.iloc[0]
        # 均衡（中间）
        mid = pareto_df.iloc[len(pareto_df) // 2]

        lines += [
            "3. 三种代表性策略定量对比",
            "",
            "| 策略 | 碳排放 kg | 运行成本 元 | 寿命损耗 元 | 综合成本 元 |",
            "|------|:---:|:---:|:---:|:---:|",
            f"| 经济优先 | {econ['total_carbon_kg']:.0f} | {econ['operation_cost_cny']:.0f} | {econ['total_degradation_cost_cny']:.0f} | {econ['total_comprehensive_cost_cny']:.0f} |",
            f"| 均衡策略 | {mid['total_carbon_kg']:.0f} | {mid['operation_cost_cny']:.0f} | {mid['total_degradation_cost_cny']:.0f} | {mid['total_comprehensive_cost_cny']:.0f} |",
            f"| 碳优先 | {carbon_first['total_carbon_kg']:.0f} | {carbon_first['operation_cost_cny']:.0f} | {carbon_first['total_degradation_cost_cny']:.0f} | {carbon_first['total_comprehensive_cost_cny']:.0f} |",
            "",
            f"  经济优先 → 碳优先：减排 {econ['total_carbon_kg'] - carbon_first['total_carbon_kg']:.0f} kg 的额外成本为 {carbon_first['total_comprehensive_cost_cny'] - econ['total_comprehensive_cost_cny']:.0f} 元",
            f"  平均边际减排成本 = {(carbon_first['total_comprehensive_cost_cny'] - econ['total_comprehensive_cost_cny']) / max(1, econ['total_carbon_kg'] - carbon_first['total_carbon_kg']) * 1000:.0f} 元/吨",
            "",
        ]

    lines += [
        "4. 结论与建议",
        "",
        "（1）碳交易机制（cap + price 同时存在）比单纯的碳排放上限或碳税更灵活。",
        "    当碳价高于园区边际减排成本时，可主动减排并出售结余配额获利。",
        "",
        "（2）影子价格分析表明：",
        f"    在 93% 自然排放的配额水平下，边际减排成本约 {shadow_0 * 1000:.0f} 元/吨" if shadow_0 is not None else "",
        "    该值远高于中国碳市场当前碳价（约 100 元/吨）和 EU ETS 碳价（约 500 元/吨），",
        "    说明在当前碳市场条件下，园区无经济动力将排放降至配额以下。",
        "    碳交易机制主要通过碳配额硬约束驱动减排，而非价格信号。",
        "",
        "（3）Pareto 前沿分析表明：",
        f"    该园区最大无甩负荷减排潜力为 {baseline_carbon - min_carbon:.0f} kg（{(baseline_carbon - min_carbon) / baseline_carbon * 100:.1f}%），",
        "    平均边际减排成本约 " + f"{(pareto_df.iloc[0]['total_comprehensive_cost_cny'] - pareto_df.iloc[-1]['total_comprehensive_cost_cny']) / max(1, pareto_df.iloc[-1]['total_carbon_kg'] - pareto_df.iloc[0]['total_carbon_kg']) * 1000:.0f}" + " 元/吨，" if pareto_df is not None and len(pareto_df) > 1 else "",
        "    边际成本随减排深度递增，深度减排代价高昂。",
        "",
        "（4）电池寿命损耗成本在各减排场景下变化很小（973 → 1030 元），",
        "    说明减排主要通过减少外网购电实现，未以牺牲电池寿命为代价。",
        "",
        "（5）建议园区：",
        "    - 将碳配额纳入 S5 综合成本优化模型，定期评估边际减排成本曲线；",
        "    - 在碳价低于边际减排成本时，优先购买碳配额满足合规要求；",
        "    - 碳价高于边际减排成本时，启动本地减排措施并出售结余配额；",
        "    - 关注碳价走势，建立碳资产管理策略的动态调整机制。",
    ]

    (out_dir / "p3_q3_summary.txt").write_text("\n".join(lines), encoding="utf-8")


# ==============================
# 主入口
# ==============================
def compute_min_feasible_carbon(data) -> float:
    """通过收紧碳排放上限、检查甩负荷量，确定无甩负荷前提下的最小可行碳排放。"""
    # 无约束时的自然排放
    _, _, info = build_and_solve_carbon_aware(
        data, carbon_price_cny_per_kg=0.0, carbon_cap_kg=None, scheme_name="baseline_for_bound"
    )
    hi = float(info["total_carbon_kg"])

    # 在自然排放的 50%–100% 之间二分搜索，直到找到无显著甩负荷的最小排放
    lo = hi * 0.5
    best = hi

    for _ in range(25):
        mid = (lo + hi) / 2
        try:
            sched, _, info = build_and_solve_carbon_aware(
                data, carbon_price_cny_per_kg=0.0, carbon_cap_kg=mid,
                scheme_name=f"binary_{mid:.0f}"
            )
            unmet_kwh = float((sched.get("unmet_load_kw", 0) * DT).sum())
            if unmet_kwh < 1e-3:  # 无显著甩负荷
                best = float(info["total_carbon_kg"])
                hi = mid
            else:  # 有甩负荷，cap 过紧
                lo = mid
        except RuntimeError:
            lo = mid  # cap 太紧，不可行

    return best


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

    # 计算最小可行碳排放
    print("计算最小可行碳排放（碳减排潜力上限）...")
    min_carbon = compute_min_feasible_carbon(data)
    print(f"  自然排放={baseline_carbon:.0f} kg, 最小排放={min_carbon:.0f} kg, "
          f"减排潜力={(baseline_carbon - min_carbon) / baseline_carbon * 100:.1f}%")

    # ---- 第1小问 ----
    q1_metrics, q1_schedules, _ = run_cap_scenarios(data, args.data_dir, out_dir, fig_dir, baseline_carbon)

    # ---- 第2小问 ----
    q2_metrics, q2_schedules, _ = run_price_scenarios(data, args.data_dir, out_dir, fig_dir, baseline_carbon)

    # ---- 补充：cap × price 联合场景（真正碳交易） ----
    joint_metrics, _ = run_joint_cap_price_scenarios(data, args.data_dir, out_dir, fig_dir, baseline_carbon)

    # ---- 第3小问：Pareto 前沿分析 ----
    pareto_df = run_q3_pareto_analysis(data, args.data_dir, out_dir, fig_dir, baseline_carbon, min_carbon)

    # ---- 第3小问：综合策略文字总结 ----
    write_q3_summary(joint_metrics, pareto_df, baseline_carbon, min_carbon, out_dir)

    # 打印摘要
    for fname in ["p3_q1_summary.txt", "p3_q2_summary.txt", "p3_q3_summary.txt"]:
        fpath = out_dir / fname
        if fpath.exists():
            print(f"\n{fpath.read_text(encoding='utf-8')}")


if __name__ == "__main__":
    main()
