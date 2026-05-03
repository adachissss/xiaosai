from __future__ import annotations

"""第二问第 2 小问入口脚本：求解考虑寿命损耗的 S4 并与 S3 对比。"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROBLEM1_SCRIPTS = Path(__file__).resolve().parents[2] / "problem1" / "scripts"
if str(PROBLEM1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PROBLEM1_SCRIPTS))

from common import DT, check_constraints, compute_metrics, read_data  # noqa: E402
from degradation_aware_strategy import build_and_solve_degradation_aware  # noqa: E402
from degradation_model import (  # noqa: E402
    battery_degradation_cost,
    ev_degradation_cost,
    load_stationary_battery_degradation_cost,
)

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

SCHEME_LABELS = {
    "S3_full_coordination": "S3 未考虑寿命损耗",
    "S4_degradation_aware": "S4 考虑寿命损耗",
}
COLOR_S3 = "#2563EB"
COLOR_S4 = "#DC2626"
COLOR_BATTERY = "#2563EB"
COLOR_EV = "#F97316"


def _load_s3_reference(problem1_results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = pd.read_csv(problem1_results_dir / "S3_full_coordination_schedule.csv", parse_dates=["timestamp"])
    ev_result = pd.read_csv(problem1_results_dir / "S3_full_coordination_ev_results.csv")
    return schedule, ev_result


def _degradation_metrics(schedule: pd.DataFrame, ev_result: pd.DataFrame, data_dir: Path, battery_unit_cost: float) -> dict[str, float]:
    bat_ch, bat_dis, bat_throughput, bat_cost = battery_degradation_cost(schedule, battery_unit_cost)
    if "degradation_cost_cny" in ev_result.columns:
        ev_ch = float((schedule["ev_charge_total_kw"] * DT).sum())
        ev_dis = float((schedule["ev_discharge_total_kw"] * DT).sum())
        ev_throughput = ev_ch + ev_dis
        ev_cost = float(ev_result["degradation_cost_cny"].sum())
    else:
        ev_ch, ev_dis, ev_throughput, ev_cost = ev_degradation_cost(schedule, ev_result, data_dir)
    return {
        "battery_charge_kwh": bat_ch,
        "battery_discharge_kwh": bat_dis,
        "battery_throughput_kwh": bat_throughput,
        "battery_degradation_cost_cny": bat_cost,
        "ev_charge_kwh": ev_ch,
        "ev_discharge_kwh": ev_dis,
        "ev_throughput_kwh": ev_throughput,
        "ev_degradation_cost_cny": ev_cost,
        "total_degradation_cost_cny": bat_cost + ev_cost,
    }


def build_comparison_metrics(
    schedules: dict[str, pd.DataFrame],
    ev_results: dict[str, pd.DataFrame],
    data,
    data_dir: Path,
    battery_unit_cost: float,
) -> pd.DataFrame:
    rows = []
    base_row = None
    for scheme in ["S3_full_coordination", "S4_degradation_aware"]:
        operation = compute_metrics(schedules[scheme], ev_results[scheme], data)
        degradation = _degradation_metrics(schedules[scheme], ev_results[scheme], data_dir, battery_unit_cost)
        row = {
            "scheme": scheme,
            "label": SCHEME_LABELS[scheme],
            "operation_cost_cny": operation["total_cost_cny"],
            "battery_degradation_cost_cny": degradation["battery_degradation_cost_cny"],
            "ev_degradation_cost_cny": degradation["ev_degradation_cost_cny"],
            "total_degradation_cost_cny": degradation["total_degradation_cost_cny"],
            "total_comprehensive_cost_cny": operation["total_cost_cny"] + degradation["total_degradation_cost_cny"],
            "grid_import_energy_kwh": operation["grid_import_energy_kwh"],
            "grid_export_energy_kwh": operation["grid_export_energy_kwh"],
            "peak_grid_import_kw": operation["peak_grid_import_kw"],
            "pv_consumption_rate": operation["pv_consumption_rate"],
            "battery_throughput_kwh": degradation["battery_throughput_kwh"],
            "ev_charge_kwh": degradation["ev_charge_kwh"],
            "ev_discharge_kwh": degradation["ev_discharge_kwh"],
            "ev_throughput_kwh": degradation["ev_throughput_kwh"],
            "ev_shortfall_kwh": operation["ev_shortfall_kwh"],
            "load_shift_down_kwh": operation["load_shift_down_kwh"],
            "load_shed_kwh": operation["load_shed_kwh"],
            "unmet_load_kwh": operation["unmet_load_kwh"],
        }
        if base_row is None:
            base_row = row.copy()
            for key in list(row):
                if key.endswith("_cny") or key.endswith("_kwh") or key.endswith("_kw") or key.endswith("_rate"):
                    row[f"delta_vs_S3_{key}"] = 0.0
        else:
            for key, base_value in base_row.items():
                if isinstance(base_value, (int, float, np.floating)):
                    row[f"delta_vs_S3_{key}"] = row[key] - float(base_value)
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_timeseries_comparison(s3: pd.DataFrame, s4: pd.DataFrame, fig_dir: Path) -> None:
    t = pd.to_datetime(s3["timestamp"])

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.plot(t, s3["grid_buy_kw"], label="S3 未考虑寿命", color=COLOR_S3, linewidth=1.2)
    ax.plot(t, s4["grid_buy_kw"], label="S4 考虑寿命", color=COLOR_S4, linewidth=1.2)
    ax.set_title("S3 与 S4 外网购电功率对比", fontweight="bold", pad=12)
    ax.set_ylabel("购电功率 / kW")
    ax.legend(loc="upper left", frameon=True)
    fig.text(0.5, 0.015, "注：若 S4 减少电池频繁调节，部分时段可能更多依赖外网购电。", ha="center", fontsize=10, color="#4B5563")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.2)
    fig.savefig(fig_dir / "p2_q2_grid_import_s3_vs_s4.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.plot(t, s3["battery_discharge_kw"] - s3["battery_charge_kw"], label="S3 储能净放电", color=COLOR_S3, linewidth=1.2)
    ax.plot(t, s4["battery_discharge_kw"] - s4["battery_charge_kw"], label="S4 储能净放电", color=COLOR_S4, linewidth=1.2)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title("S3 与 S4 固定储能充放电功率对比", fontweight="bold", pad=12)
    ax.set_ylabel("净放电功率 / kW")
    ax.legend(loc="upper left", frameon=True)
    fig.text(0.5, 0.015, "注：正值表示放电供能，负值表示充电吸收。", ha="center", fontsize=10, color="#4B5563")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.2)
    fig.savefig(fig_dir / "p2_q2_battery_power_s3_vs_s4.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.plot(t, s3["battery_energy_kwh"], label="S3 储能电量", color=COLOR_S3, linewidth=1.2)
    ax.plot(t, s4["battery_energy_kwh"], label="S4 储能电量", color=COLOR_S4, linewidth=1.2)
    ax.set_title("S3 与 S4 固定储能 SOC 变化对比", fontweight="bold", pad=12)
    ax.set_ylabel("储能电量 / kWh")
    ax.legend(loc="upper left", frameon=True)
    fig.text(0.5, 0.015, "注：考虑寿命损耗后，SOC 轨迹通常会减少无必要的深度往返波动。", ha="center", fontsize=10, color="#4B5563")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.2)
    fig.savefig(fig_dir / "p2_q2_battery_soc_s3_vs_s4.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.plot(t, s3["ev_discharge_total_kw"], label="S3 EV 放电", color=COLOR_S3, linewidth=1.2)
    ax.plot(t, s4["ev_discharge_total_kw"], label="S4 EV 放电", color=COLOR_S4, linewidth=1.2)
    ax.set_title("S3 与 S4 电动车 V2B 放电功率对比", fontweight="bold", pad=12)
    ax.set_ylabel("EV 放电功率 / kW")
    ax.legend(loc="upper left", frameon=True)
    fig.text(0.5, 0.015, "注：S4 对 EV 电池吞吐计入寿命成本，因此 V2B 调用会更谨慎。", ha="center", fontsize=10, color="#4B5563")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.2)
    fig.savefig(fig_dir / "p2_q2_ev_v2b_s3_vs_s4.png")
    plt.close(fig)


def _plot_bar_comparisons(metrics: pd.DataFrame, fig_dir: Path) -> None:
    labels = ["S3\n未考虑寿命", "S4\n考虑寿命"]
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.bar(x, metrics["battery_throughput_kwh"], label="固定储能", color=COLOR_BATTERY, alpha=0.88, width=0.56)
    ax.bar(x, metrics["ev_throughput_kwh"], bottom=metrics["battery_throughput_kwh"], label="电动车车队", color=COLOR_EV, alpha=0.88, width=0.56)
    totals = metrics["battery_throughput_kwh"] + metrics["ev_throughput_kwh"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("等效吞吐量 / kWh")
    ax.set_title("引入寿命损耗前后电池等效吞吐量对比", fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True, ncol=2)
    ax.set_ylim(0, float(totals.max()) * 1.18)
    for i, value in enumerate(totals):
        ax.text(i, value + float(totals.max()) * 0.025, f"{value:.1f}", ha="center", va="bottom", fontsize=9, color="#374151")
    fig.text(0.5, 0.015, "注：吞吐量越低，表示电池循环使用强度越低。", ha="center", fontsize=10, color="#4B5563")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.18)
    fig.savefig(fig_dir / "p2_q2_throughput_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.bar(x, metrics["operation_cost_cny"], label="运行成本", color="#10B981", alpha=0.88, width=0.56)
    ax.bar(x, metrics["battery_degradation_cost_cny"], bottom=metrics["operation_cost_cny"], label="固定储能寿命损耗", color=COLOR_BATTERY, alpha=0.88, width=0.56)
    bottom = metrics["operation_cost_cny"] + metrics["battery_degradation_cost_cny"]
    ax.bar(x, metrics["ev_degradation_cost_cny"], bottom=bottom, label="EV 寿命损耗", color=COLOR_EV, alpha=0.88, width=0.56)
    totals = metrics["total_comprehensive_cost_cny"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("成本 / 元")
    ax.set_title("引入寿命损耗前后综合成本构成对比", fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True, ncol=3)
    ax.set_ylim(0, float(totals.max()) * 1.18)
    for i, value in enumerate(totals):
        ax.text(i, value + float(totals.max()) * 0.025, f"{value:.1f}", ha="center", va="bottom", fontsize=9, color="#374151")
    fig.text(0.5, 0.015, "注：综合成本 = 运行成本 + 固定储能寿命损耗成本 + EV 寿命损耗成本。", ha="center", fontsize=10, color="#4B5563")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.18)
    fig.savefig(fig_dir / "p2_q2_cost_breakdown.png")
    plt.close(fig)


def plot_problem2_results(schedules: dict[str, pd.DataFrame], metrics: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_timeseries_comparison(schedules["S3_full_coordination"], schedules["S4_degradation_aware"], fig_dir)
    _plot_bar_comparisons(metrics, fig_dir)


def write_summary(metrics: pd.DataFrame, solve_info: dict, issues: dict[str, list[str]], out_path: Path) -> None:
    s3 = metrics.loc[metrics["scheme"] == "S3_full_coordination"].iloc[0]
    s4 = metrics.loc[metrics["scheme"] == "S4_degradation_aware"].iloc[0]
    lines = [
        "B题第二问第2小问：考虑电池寿命损耗的协同调度",
        "=" * 64,
        "",
        "1. 模型思路",
        "在第一问 S3 完整协同模型基础上，保持约束不变，在目标函数中加入固定储能和 EV 电池寿命损耗成本。",
        "目标函数为：min 运行成本 + C_bat_deg + C_ev_deg。",
        "其中固定储能和 EV 均采用第1小问确定的等效吞吐量寿命损耗模型。",
        "",
        "2. S4 求解状态",
        f"求解状态：{solve_info.get('status')}",
        f"目标值：{solve_info.get('objective'):.6f}",
        f"固定储能单位吞吐寿命成本：{solve_info.get('battery_unit_degradation_cost_cny_per_kwh'):.4f} 元/kWh",
        "",
        "3. S3 与 S4 核心指标对比",
        metrics.to_string(index=False),
        "",
        "4. 主要变化",
        f"固定储能吞吐量变化：{s4['battery_throughput_kwh'] - s3['battery_throughput_kwh']:.3f} kWh。",
        f"EV 吞吐量变化：{s4['ev_throughput_kwh'] - s3['ev_throughput_kwh']:.3f} kWh。",
        f"运行成本变化：{s4['operation_cost_cny'] - s3['operation_cost_cny']:.3f} 元。",
        f"寿命损耗成本变化：{s4['total_degradation_cost_cny'] - s3['total_degradation_cost_cny']:.3f} 元。",
        f"综合成本变化：{s4['total_comprehensive_cost_cny'] - s3['total_comprehensive_cost_cny']:.3f} 元。",
        "",
        "5. 约束检查提示",
    ]
    for scheme, scheme_issues in issues.items():
        lines.append(f"{scheme}：")
        if scheme_issues:
            lines.extend(f"  - {x}" for x in scheme_issues)
        else:
            lines.append("  - 未发现关键约束问题。")
    lines.extend([
        "",
        "6. 小问结论",
        "引入寿命损耗后，模型只有在运行收益能够覆盖电池寿命成本时才会调用固定储能或 EV V2B。",
        "因此 S4 能体现短期运行经济性与长期电池健康之间的权衡。",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "B_data")
    parser.add_argument("--problem1-results-dir", type=Path, default=Path(__file__).resolve().parents[2] / "problem1" / "results")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()

    data = read_data(args.data_dir)
    battery_unit_cost = load_stationary_battery_degradation_cost(args.data_dir)

    s3_schedule, s3_ev_result = _load_s3_reference(args.problem1_results_dir)
    s4_schedule, s4_ev_result, solve_info = build_and_solve_degradation_aware(data)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    s3_schedule.to_csv(args.out_dir / "S3_reference_schedule.csv", index=False)
    s3_ev_result.to_csv(args.out_dir / "S3_reference_ev_results.csv", index=False)
    s4_schedule.to_csv(args.out_dir / "S4_degradation_aware_schedule.csv", index=False)
    s4_ev_result.to_csv(args.out_dir / "S4_degradation_aware_ev_results.csv", index=False)

    schedules = {"S3_full_coordination": s3_schedule, "S4_degradation_aware": s4_schedule}
    ev_results = {"S3_full_coordination": s3_ev_result, "S4_degradation_aware": s4_ev_result}
    metrics = build_comparison_metrics(schedules, ev_results, data, args.data_dir, battery_unit_cost)
    metrics.to_csv(args.out_dir / "p2_q2_comparison_metrics.csv", index=False)

    plot_problem2_results(schedules, metrics, fig_dir)
    issues = {scheme: check_constraints(schedules[scheme], ev_results[scheme], data) for scheme in schedules}
    write_summary(metrics, solve_info, issues, args.out_dir / "p2_q2_summary.txt")

    print((args.out_dir / "p2_q2_summary.txt").read_text(encoding="utf-8"))
    print(f"\nFigures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
