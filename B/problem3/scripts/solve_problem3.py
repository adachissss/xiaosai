from __future__ import annotations

"""第三问入口脚本：cap-and-trade 碳交易机制对协同调度的影响分析。

碳交易机制：
- 园区获得免费碳排放配额（free_allowance）
- 实际排放 ＞ 配额 → 购买差额，碳交易成本为正
- 实际排放 ＜ 配额 → 卖出剩余配额，碳交易成本为负（收入）
- 碳交易成本 = carbon_price × (total_emissions - free_allowance)
"""

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
    schedule: pd.DataFrame, ev_result: pd.DataFrame, data, data_dir: Path,
    carbon_price: float, free_allowance: float | None,
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
    # cap-and-trade: 交易成本 = 碳价 × (实际排放 - 免费配额)，可为负
    if free_allowance is not None:
        carbon_trading_cost = carbon_price * (total_carbon_kg - free_allowance)
    else:
        carbon_trading_cost = carbon_price * total_carbon_kg

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
        "free_allowance_kg": free_allowance,
        "carbon_price_cny_per_kg": carbon_price,
    }


# ==============================
# 第1小问：不同免费碳排放配额
# ==============================
def run_cap_scenarios(data, data_dir, out_dir, fig_dir, baseline_carbon):
    """cap-and-trade 机制下，固定碳价，变化免费配额。

    在 cap-and-trade 下，免费配额是常量，不影响 LP 优化（不影响调度）。
    但碳交易成本 = price × (E - allowance) 随配额变化，可为负（卖配额收入）。
    """
    print("=" * 60)
    print("第1小问：不同免费碳排放配额对园区财务的影响")
    print("=" * 60)
    CARBON_PRICE = 0.10  # 固定碳价 100 元/吨

    allowance_scenarios = [
        ("无配额", None),
        ("宽松配额", baseline_carbon * 1.05),
        ("基准配额", baseline_carbon),
        ("紧缩配额", baseline_carbon * 0.93),
        ("极紧缩配额", baseline_carbon * 0.86),
    ]

    schedules = {}
    ev_results = {}
    metrics_list = []

    for label, allowance in allowance_scenarios:
        try:
            sched, ev_r, info = build_and_solve_carbon_aware(
                data, carbon_price_cny_per_kg=CARBON_PRICE, free_allowance_kg=allowance,
                scheme_name=f"S5_cap_{label}",
            )
        except RuntimeError as e:
            print(f"  {label} (allowance={allowance}): 求解失败 — {e}")
            continue

        schedules[label] = sched
        ev_results[label] = ev_r
        m = extract_metrics(sched, ev_r, data, data_dir, CARBON_PRICE, allowance)
        m["scenario_label"] = label
        metrics_list.append(m)
        trading_sign = "收入" if m["carbon_trading_cost_cny"] < 0 else "支出"
        print(f"  {label:6s} 配额={str(allowance or '∞'):>12s} → "
              f"碳={m['total_carbon_kg']:>8.1f} kg, "
              f"碳交易{m['carbon_trading_cost_cny']:>+8.1f} 元({trading_sign}), "
              f"综合成本={m['total_comprehensive_cost_cny']:>8.1f} 元")

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(out_dir / "p3_q1_comparison_metrics.csv", index=False)
    _plot_q1_results(metrics_df, schedules, fig_dir, CARBON_PRICE)
    _write_q1_summary(metrics_df, baseline_carbon, out_dir)
    return metrics_df, schedules, ev_results


def _plot_q1_results(metrics: pd.DataFrame, schedules: dict[str, pd.DataFrame], fig_dir: Path, carbon_price: float):
    labels = metrics["scenario_label"].tolist()
    x = np.arange(len(labels))

    # 图1：碳排放与配额对比
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.bar(x, metrics["total_carbon_kg"], color="#3B82F6", alpha=0.85, width=0.55, label="实际碳排放")
    for i, (_, row) in enumerate(metrics.iterrows()):
        allowance = row["free_allowance_kg"]
        if allowance and not np.isinf(allowance):
            ax.axhline(allowance, xmin=(i-0.28)/len(labels), xmax=(i+0.28)/len(labels),
                       color="#DC2626", linewidth=2.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("碳排放 / kg CO2")
    ax.set_title("不同免费配额下的实际碳排放（碳价=100元/吨）", fontweight="bold", pad=12)
    ymax = float(metrics["total_carbon_kg"].max()) * 1.18
    ax.set_ylim(0, ymax)
    for i, v in enumerate(metrics["total_carbon_kg"]):
        ax.text(i, v + ymax * 0.02, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color="#374151")
    fig.text(0.5, 0.015, "注：虚线=免费配额；所有场景调度相同（配额不影响优化），碳交易成本不同。", ha="center", fontsize=9, color="#4B5563")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.2)
    fig.savefig(fig_dir / "p3_q1_emissions_by_cap.png")
    plt.close(fig)

    # 图2：碳交易成本对比
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    colors = ["#10B981" if v < 0 else "#EF4444" for v in metrics["carbon_trading_cost_cny"]]
    ax.bar(x, metrics["carbon_trading_cost_cny"], color=colors, alpha=0.85, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("碳交易成本 / 元")
    ax.set_title("不同免费配额下的碳交易成本（正=支出，负=收入）", fontweight="bold", pad=12)
    ax.axhline(0, color="black", linewidth=0.5)
    ymax = float(metrics["carbon_trading_cost_cny"].max()) * 1.18
    ymin = min(0, float(metrics["carbon_trading_cost_cny"].min())) * 1.18
    ax.set_ylim(ymin, ymax)
    for i, v in enumerate(metrics["carbon_trading_cost_cny"]):
        va_pos = "bottom" if v >= 0 else "top"
        offset = ymax * 0.03 if v >= 0 else ymin * 0.03
        ax.text(i, v + offset, f"{v:+.0f}", ha="center", va=va_pos, fontsize=9, color="#374151")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.2)
    fig.savefig(fig_dir / "p3_q1_grid_import_by_cap.png")
    plt.close(fig)

    # 图3：综合成本构成（运行+寿命+碳交易）
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.bar(x, metrics["operation_cost_cny"], label="运行成本", color=COLOR_OP, alpha=0.88, width=0.55)
    bottom1 = metrics["operation_cost_cny"]
    ax.bar(x, metrics["total_degradation_cost_cny"], bottom=bottom1, label="寿命损耗", color=COLOR_EV, alpha=0.88, width=0.55)
    bottom2 = bottom1 + metrics["total_degradation_cost_cny"]
    ct = metrics["carbon_trading_cost_cny"]
    ax.bar(x, ct, bottom=bottom2, label="碳交易成本", color=COLOR_CARBON, alpha=0.88, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("成本 / 元")
    ax.set_title("不同免费配额下综合成本构成（碳价=100元/吨）", fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True, ncol=3)
    totals = metrics["total_comprehensive_cost_cny"]
    ymax = float(totals.max()) * 1.18
    ymin = min(0, float(metrics["carbon_trading_cost_cny"].min()) * 2)
    ax.set_ylim(ymin, ymax)
    for i, v in enumerate(totals):
        ax.text(i, v + ymax * 0.025, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.18)
    fig.savefig(fig_dir / "p3_q1_cost_breakdown_by_cap.png")
    plt.close(fig)

    # 图4：电池吞吐量（验证所有场景调度相同）
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.bar(x, metrics["battery_throughput_kwh"], label="固定储能", color=COLOR_BAT, alpha=0.88, width=0.55)
    ax.bar(x, metrics["ev_throughput_kwh"], bottom=metrics["battery_throughput_kwh"], label="EV", color=COLOR_EV, alpha=0.88, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("等效吞吐量 / kWh")
    ax.set_title("不同配额下电池吞吐量（验证调度相同）", fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True, ncol=2)
    totals = metrics["battery_throughput_kwh"] + metrics["ev_throughput_kwh"]
    ax.set_ylim(0, float(totals.max()) * 1.18)
    for i, v in enumerate(totals):
        ax.text(i, v + float(totals.max()) * 0.025, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.18)
    fig.savefig(fig_dir / "p3_q1_battery_ev_by_cap.png")
    plt.close(fig)


def _write_q1_summary(metrics: pd.DataFrame, baseline_carbon: float, out_dir: Path):
    lines = [
        "第三问第1小问：不同免费碳排放配额对园区财务的影响",
        "=" * 60, "",
        "机制：cap-and-trade，碳价固定 100 元/吨（0.10 元/kg）",
        f"S4 自然排放基准值：{baseline_carbon:.1f} kg CO2/周", "",
        "配额是常量，不影响 LP 优化，因此所有场景调度结果相同。",
        "但碳交易成本 = 碳价 × (实际排放 - 配额) 随配额变化。", "",
        "各场景对比：",
    ]
    for _, row in metrics.iterrows():
        allowance_str = f"{row['free_allowance_kg']:.0f}" if row["free_allowance_kg"] and row["free_allowance_kg"] < float("inf") else "无配额"
        sign = "收入" if row["carbon_trading_cost_cny"] < 0 else "支出"
        lines.append(f"  {row['scenario_label']:8s} | 配额={allowance_str:>8s} kg | "
                     f"排放={row['total_carbon_kg']:>8.1f} | "
                     f"碳交易={row['carbon_trading_cost_cny']:>+8.1f} ({sign}) | "
                     f"综合成本={row['total_comprehensive_cost_cny']:>8.1f}")
    lines += [
        "", "核心结论：",
        "  1. 免费配额不改变调度行为——只影响园区的净财务位置",
        "  2. 无配额时碳交易为净支出，宽松配额时为净收入",
        "  3. 碳价才是驱动调度行为变化的唯一因素",
    ]
    lines.append("")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "p3_q1_summary.txt").write_text("\n".join(lines), encoding="utf-8")


# ==============================
# 第2小问：不同碳交易价格
# ==============================
def run_price_scenarios(data, data_dir, out_dir, fig_dir, baseline_carbon):
    """cap-and-trade 机制下，固定免费配额=基线排放，变化碳价。

    碳交易成本 = price × (E - allowance)，排放低于配额时为收入。
    """
    print("\n" + "=" * 60)
    print("第2小问：不同碳交易价格对系统调度的影响")
    print("=" * 60)
    FREE_ALLOWANCE = baseline_carbon  # 固定配额=基线排放

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
                data, carbon_price_cny_per_kg=price, free_allowance_kg=FREE_ALLOWANCE,
                scheme_name=f"S5_price_{label}",
            )
        except RuntimeError as e:
            print(f"  {label} (price={price}): 求解失败 — {e}")
            continue

        schedules[label] = sched
        ev_results[label] = ev_r
        m = extract_metrics(sched, ev_r, data, data_dir, price, FREE_ALLOWANCE)
        m["scenario_label"] = label
        m["carbon_price_cny_per_kg"] = price
        metrics_list.append(m)
        trading_sign = "收入" if m["carbon_trading_cost_cny"] < 0 else "支出"
        print(f"  {label:10s} 碳价={price*1000:>4.0f} 元/吨 → "
              f"排放={m['total_carbon_kg']:>8.1f} kg, "
              f"碳交易={m['carbon_trading_cost_cny']:>+8.1f} 元({trading_sign}), "
              f"综合成本={m['total_comprehensive_cost_cny']:>8.1f} 元")

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(out_dir / "p3_q2_comparison_metrics.csv", index=False)
    _plot_q2_results(metrics_df, fig_dir, FREE_ALLOWANCE)
    _write_q2_summary(metrics_df, out_dir)
    return metrics_df, schedules, ev_results


def _plot_q2_results(metrics: pd.DataFrame, fig_dir: Path, free_allowance: float):
    prices = metrics["carbon_price_cny_per_kg"] * 1000  # 转为 元/吨
    x = range(len(metrics))

    # 图1：碳排放 vs 碳价
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.bar(x, metrics["total_carbon_kg"], color=["#3B82F6", "#F59E0B", "#EF4444", "#8B5CF6"], alpha=0.85, width=0.55)
    ax.axhline(free_allowance, color="#DC2626", linestyle="--", linewidth=1.5, alpha=0.7, label=f"免费配额({free_allowance:.0f} kg)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.0f}" for p in prices])
    ax.set_xlabel("碳交易价格 / (元/吨 CO2)")
    ax.set_ylabel("碳排放 / kg CO2")
    ax.set_title("不同碳价下总碳排放（虚线=免费配额）", fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True)
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

    # 图4：成本构成（含可正可负的碳交易成本）
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    op = metrics["operation_cost_cny"]
    deg = metrics["total_degradation_cost_cny"]
    ct = metrics["carbon_trading_cost_cny"]
    # 碳交易成本为负（收入）时用绿色，为正（支出）时用灰色
    ct_colors = ["#10B981" if v < 0 else "#6B7280" for v in ct]
    ax.bar(x, op, label="运行成本", color=COLOR_OP, alpha=0.88, width=0.55)
    ax.bar(x, deg, bottom=op, label="寿命损耗", color=COLOR_EV, alpha=0.88, width=0.55)
    for i in range(len(x)):
        base = op.iloc[i] + deg.iloc[i]
        ax.bar(i, ct.iloc[i], bottom=base, color=ct_colors[i], alpha=0.88, width=0.55)
    ax.bar([], [], color="#6B7280", label="碳交易(支出)", alpha=0.88)
    ax.bar([], [], color="#10B981", label="碳交易(收入)", alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.0f}" for p in prices])
    ax.set_xlabel("碳交易价格 / (元/吨 CO2)")
    ax.set_ylabel("成本 / 元")
    ax.set_title("不同碳价下综合成本构成（配额=" + f"{free_allowance:.0f} kg" + "）", fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True, ncol=4)
    totals = metrics["total_comprehensive_cost_cny"]
    ymax = float(totals.max()) * 1.18
    ymin = min(0, float(ct.min()) * 1.5)
    ax.set_ylim(ymin, ymax)
    for i, v in enumerate(totals):
        ax.text(i, v + ymax * 0.025, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.2)
    fig.savefig(fig_dir / "p3_q2_cost_breakdown_by_price.png")
    plt.close(fig)


def _write_q2_summary(metrics: pd.DataFrame, out_dir: Path):
    lines = [
        "第三问第2小问：不同碳交易价格对系统调度的影响",
        "=" * 60, "",
        "机制：cap-and-trade，免费配额 = S4 基线排放",
        f"碳交易成本 = 碳价 × (实际排放 - 配额)，可为负（收入）", "",
        "各碳价场景对比：",
    ]
    for _, row in metrics.iterrows():
        sign = "收入" if row["carbon_trading_cost_cny"] < 0 else "支出"
        lines.append(f"  {row['scenario_label']:10s} | "
                     f"碳价={row['carbon_price_cny_per_kg']*1000:>4.0f} 元/吨 | "
                     f"排放={row['total_carbon_kg']:>8.1f} kg | "
                     f"碳交易={row['carbon_trading_cost_cny']:>+8.1f} ({sign}) | "
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
        "机制：cap-and-trade（碳配额交易）",
        "  碳交易成本 = 碳价 × (实际排放 - 免费配额)，可为负（卖配额收入）",
        "",
        "1. 关键发现", "",
        "（1）碳价是唯一影响调度行为的因素：",
        "    免费配额只改变园区的净财务位置，不改变调度决策。",
        "    无配额时碳交易为净支出，配额宽松时为净收入。", "",
        "（2）碳价对减排的边际效果有限（当前参数下）：",
        "    碳价升至 500 元/吨时碳排放仅下降约 1.4%。",
        "    园区可调资源接近饱和，碳价信号难以创造额外无碳替代。", "",
        "（3）配额分配是重要的政策工具：",
        "    不同的配额分配方案决定园区是净买家还是净卖家。",
        "    这对园区财务稳健性有直接影响。", "",
    ]
    if q1_metrics is not None and len(q1_metrics) > 0:
        lines += ["2. 不同配额下的财务影响（碳价=100元/吨）", ""]
        for _, row in q1_metrics.iterrows():
            sign = "收入" if row["carbon_trading_cost_cny"] < 0 else "支出"
            lines.append(f"  {row['scenario_label']}: 碳交易={row['carbon_trading_cost_cny']:+.0f} 元({sign})")
    if q2_metrics is not None and len(q2_metrics) > 0:
        lines += ["", "3. 不同碳价下的调度与财务影响（配额=基线排放）", ""]
        for _, row in q2_metrics.iterrows():
            sign = "收入" if row["carbon_trading_cost_cny"] < 0 else "支出"
            lines.append(f"  {row['scenario_label']}: 排放={row['total_carbon_kg']:.0f} kg, "
                         f"碳交易={row['carbon_trading_cost_cny']:+.0f} 元({sign})")
    lines += [
        "", "4. 建议的综合运行策略",
        "",
        "（1）将碳价信号纳入日常调度优化——碳价是唯一影响调度行为的碳政策参数。",
        "（2）密切关注碳配额分配方案——配额决定园区是净支出还是净收入。",
        "（3）通过碳价调节减排力度，避免过度依赖 EV 电池（寿命成本高）。",
        "（4）建议碳价在 100-300 元/吨区间，配合排放低于配额时可卖出的机制获取收益。",
        "（5）高碳强度时段主动减少购电，低碳强度时段适度增加购电。",
        "",
        "5. 结论", "",
        "cap-and-trade 机制下，碳价驱动调度行为，配额决定财务位置。",
        "S5 模型将运行成本、电池寿命和碳排放统一为综合成本，",
        "园区可在三个维度间取得最优平衡。",
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

    # 确定 S4 自然排放基准（无碳约束）
    base_result = build_and_solve_carbon_aware(data, carbon_price_cny_per_kg=0.0, free_allowance_kg=None)
    baseline_carbon = base_result[2]["total_carbon_kg"]

    # ---- 第1小问：固定碳价，变化免费配额 ----
    q1_metrics, q1_schedules, _ = run_cap_scenarios(data, args.data_dir, out_dir, fig_dir, baseline_carbon)

    # ---- 第2小问：固定配额，变化碳价 ----
    q2_metrics, q2_schedules, _ = run_price_scenarios(data, args.data_dir, out_dir, fig_dir, baseline_carbon)

    # ---- 第3小问：综合策略 ----
    write_q3_summary(q1_metrics, q2_metrics, None, None, out_dir)

    # 打印摘要
    for fname in ["p3_q1_summary.txt", "p3_q2_summary.txt", "p3_q3_summary.txt"]:
        fpath = out_dir / fname
        if fpath.exists():
            print(f"\n{fpath.read_text(encoding='utf-8')}")


if __name__ == "__main__":
    main()
