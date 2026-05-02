#!/usr/bin/env python3
"""B题第一问总入口脚本。

现在统一运行四组方案，便于论文做多对照组分析：
- S0_no_storage：朴素方案，储能不运行，EV 即插即充，建筑不调节；
- S1_rule_storage：规则储能方案，储能按固定规则运行；
- S2_partial_coordination：部分协同，储能优化 + EV 智能充电，但 EV 不放电、建筑不调节；
- S3_full_coordination：完整协同，储能 + EV V2B + 建筑柔性负荷联合优化。

运行：
    conda run -n xiaosai-b python scripts/solve_problem1.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baseline_strategy import (
    simulate_baseline_no_storage,
    simulate_baseline_rule_storage,
    solve_partial_coordination,
)
from common import check_constraints, compute_metrics, read_data
from coordinated_strategy import build_and_solve_coordinated


SCHEME_LABELS = {
    "S0_no_storage": "S0 朴素方案",
    "S1_rule_storage": "S1 规则储能方案",
    "S2_partial_coordination": "S2 部分协同方案",
    "S3_full_coordination": "S3 完整协同方案",
}


def plot_outputs(out_dir: Path, schedules: dict[str, pd.DataFrame]) -> None:
    """输出论文可用的多方案对比图，统一保存到 results/figures。"""

    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    t = pd.to_datetime(next(iter(schedules.values()))["timestamp"])

    plt.figure(figsize=(14, 5))
    for name, schedule in schedules.items():
        plt.plot(t, schedule["grid_buy_kw"], label=name, linewidth=1)
    plt.ylabel("kW")
    plt.title("Grid Import Power Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "grid_import_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(14, 5))
    for name, schedule in schedules.items():
        plt.plot(t, schedule["battery_energy_kwh"], label=name, linewidth=1)
    plt.ylabel("kWh")
    plt.title("Stationary Battery Energy Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "battery_energy_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(14, 5))
    for name, schedule in schedules.items():
        plt.plot(t, schedule["ev_net_kw"], label=name, linewidth=1)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("kW")
    plt.title("Aggregated EV Net Power Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "ev_net_power_comparison.png", dpi=180)
    plt.close()

    # 光伏图只展示完整协同方案，同时保留 PV available / used / curtailed 三条曲线。
    coord = schedules["S3_full_coordination"]
    plt.figure(figsize=(14, 5))
    plt.plot(t, coord["pv_available_kw"], label="PV available", linewidth=1)
    plt.plot(t, coord["pv_used_kw"], label="PV used", linewidth=1)
    plt.plot(t, coord["pv_curtail_kw"], label="PV curtailed", linewidth=1)
    plt.ylabel("kW")
    plt.title("PV Utilization in Full Coordination Scheme")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "pv_utilization.png", dpi=180)
    plt.close()


def run_all_schemes(data) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, dict]]:
    """运行四组方案，并返回调度表、EV 结果表和求解信息。"""

    schedules = {}
    ev_results = {}
    solve_infos = {}

    # S0：最朴素对照组，不用固定储能，不做需求响应。
    schedules["S0_no_storage"], ev_results["S0_no_storage"] = simulate_baseline_no_storage(data)
    solve_infos["S0_no_storage"] = {"status": "rule simulation", "objective": float("nan")}

    # S1：当前已有的规则储能 baseline。
    schedules["S1_rule_storage"], ev_results["S1_rule_storage"] = simulate_baseline_rule_storage(data)
    solve_infos["S1_rule_storage"] = {"status": "rule simulation", "objective": float("nan")}

    # S2：部分协同，只优化储能和 EV 充电时机。
    schedules["S2_partial_coordination"], ev_results["S2_partial_coordination"], solve_infos["S2_partial_coordination"] = solve_partial_coordination(data)

    # S3：完整协同，允许 EV V2B 和建筑柔性负荷。
    schedules["S3_full_coordination"], ev_results["S3_full_coordination"], solve_infos["S3_full_coordination"] = build_and_solve_coordinated(data)

    return schedules, ev_results, solve_infos


def write_outputs(
    out_dir: Path,
    schedules: dict[str, pd.DataFrame],
    ev_results: dict[str, pd.DataFrame],
    metrics: dict[str, dict],
    solve_infos: dict[str, dict],
    data,
) -> None:
    """保存每组方案的调度表、EV 结果、总指标表、图片和摘要文本。"""

    out_dir.mkdir(parents=True, exist_ok=True)

    for name, schedule in schedules.items():
        schedule.to_csv(out_dir / f"{name}_schedule.csv", index=False)
        ev_results[name].to_csv(out_dir / f"{name}_ev_results.csv", index=False)

    plot_outputs(out_dir, schedules)

    metric_rows = []
    base = metrics["S0_no_storage"]
    for name, values in metrics.items():
        row = {"scheme": name, "label": SCHEME_LABELS[name], **values}
        for key, base_value in base.items():
            value = values[key]
            if isinstance(base_value, (int, float)) and abs(base_value) > 1e-12:
                row[f"improvement_vs_S0_{key}"] = (base_value - value) / base_value
        metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(out_dir / "comparison_metrics.csv", index=False)

    lines = ["B题第一问多方案运行结果摘要", "=" * 48, ""]
    for name in schedules:
        info = solve_infos[name]
        issues = check_constraints(schedules[name], ev_results[name], data)
        lines.append(f"{name} - {SCHEME_LABELS[name]}")
        lines.append(f"求解/仿真状态：{info.get('status')}")
        if pd.notna(info.get("objective", float("nan"))):
            lines.append(f"目标值：{info.get('objective'):.3f}")
        for key in [
            "total_cost_cny",
            "grid_import_energy_kwh",
            "peak_grid_import_kw",
            "pv_consumption_rate",
            "ev_satisfaction_rate",
            "ev_shortfall_kwh",
            "load_shift_down_kwh",
            "load_shed_kwh",
        ]:
            lines.append(f"- {key}: {metrics[name][key]:.6g}")
        if issues:
            lines.append("约束检查提示：")
            lines.extend(f"  - {x}" for x in issues)
        lines.append("")

    (out_dir / "problem1_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "B_data")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()

    data = read_data(args.data_dir)
    schedules, ev_results, solve_infos = run_all_schemes(data)
    metrics = {name: compute_metrics(schedules[name], ev_results[name], data) for name in schedules}
    write_outputs(args.out_dir, schedules, ev_results, metrics, solve_infos, data)
    print((args.out_dir / "problem1_summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
