#!/usr/bin/env python3
"""B题第一问总入口脚本。

这个文件只负责组织流程，不再混放策略细节：
- baseline_strategy.py：非协同运行方案；
- coordinated_strategy.py：协同优化方案；
- common.py：数据读取、公共参数、指标计算、约束检查。

运行：
    conda run -n xiaosai-b python scripts/solve_problem1.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baseline_strategy import simulate_baseline
from common import check_constraints, compute_metrics, read_data
from coordinated_strategy import build_and_solve_coordinated


def plot_outputs(out_dir: Path, baseline: pd.DataFrame, coord: pd.DataFrame) -> None:
    """输出论文可用的几张基础调度图。"""

    import matplotlib.pyplot as plt

    t = pd.to_datetime(coord["timestamp"])

    plt.figure(figsize=(14, 5))
    plt.plot(t, baseline["grid_buy_kw"], label="Baseline grid import", linewidth=1)
    plt.plot(t, coord["grid_buy_kw"], label="Coordinated grid import", linewidth=1)
    plt.ylabel("kW")
    plt.title("Grid Import Power Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "grid_import_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(t, coord["battery_energy_kwh"], label="Stationary battery energy", linewidth=1)
    plt.ylabel("kWh")
    plt.title("Stationary Battery SOC Trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "battery_energy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(t, coord["ev_net_kw"], label="EV net power (+ charge, - discharge)", linewidth=1)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("kW")
    plt.title("Aggregated EV Net Power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "ev_net_power.png", dpi=180)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(t, coord["pv_available_kw"], label="PV available", linewidth=1)
    plt.plot(t, coord["pv_used_kw"], label="PV used", linewidth=1)
    plt.plot(t, coord["pv_curtail_kw"], label="PV curtailed", linewidth=1)
    plt.ylabel("kW")
    plt.title("PV Utilization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pv_utilization.png", dpi=180)
    plt.close()


def write_outputs(
    out_dir: Path,
    baseline: pd.DataFrame,
    coord: pd.DataFrame,
    baseline_ev: pd.DataFrame,
    coord_ev: pd.DataFrame,
    baseline_metrics: dict,
    coord_metrics: dict,
    solve_info: dict,
    data,
) -> None:
    """保存调度表、指标表、图片和摘要文本。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(out_dir / "baseline_schedule.csv", index=False)
    coord.to_csv(out_dir / "coordinated_schedule.csv", index=False)
    baseline_ev.to_csv(out_dir / "baseline_ev_results.csv", index=False)
    coord_ev.to_csv(out_dir / "coordinated_ev_results.csv", index=False)
    plot_outputs(out_dir, baseline, coord)

    metrics = pd.DataFrame([
        {"scheme": "baseline", **baseline_metrics},
        {"scheme": "coordinated", **coord_metrics},
    ])
    for key in baseline_metrics:
        b = baseline_metrics[key]
        c = coord_metrics[key]
        if isinstance(b, (int, float)) and abs(b) > 1e-12:
            metrics.loc[metrics["scheme"] == "coordinated", f"improvement_vs_baseline_{key}"] = (b - c) / b
    metrics.to_csv(out_dir / "comparison_metrics.csv", index=False)

    issues = check_constraints(coord, coord_ev, data)
    lines = [
        "B题第一问运行结果摘要",
        "=" * 40,
        f"求解状态：{solve_info.get('status')}",
        f"协同优化目标值：{solve_info.get('objective'):.3f}",
        "",
        "核心指标对比：",
    ]
    for key in [
        "total_cost_cny",
        "grid_import_energy_kwh",
        "grid_export_energy_kwh",
        "peak_grid_import_kw",
        "pv_consumption_rate",
        "pv_curtailment_rate",
        "ev_satisfaction_rate",
        "load_shift_down_kwh",
        "load_shift_up_kwh",
        "load_shed_kwh",
    ]:
        lines.append(f"- {key}: baseline={baseline_metrics[key]:.6g}, coordinated={coord_metrics[key]:.6g}")
    lines.append("")
    if issues:
        lines.append("约束检查问题：")
        lines.extend(f"- {x}" for x in issues)
    else:
        lines.append("约束检查：协同方案未发现关键约束违反。")
    (out_dir / "problem1_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "B_data")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()

    data = read_data(args.data_dir)

    # 非协同方案：规则仿真。
    baseline, baseline_ev = simulate_baseline(data)

    # 协同方案：线性规划优化。
    coord, coord_ev, solve_info = build_and_solve_coordinated(data)

    baseline_metrics = compute_metrics(baseline, baseline_ev, data)
    coord_metrics = compute_metrics(coord, coord_ev, data)
    write_outputs(args.out_dir, baseline, coord, baseline_ev, coord_ev, baseline_metrics, coord_metrics, solve_info, data)
    print((args.out_dir / "problem1_summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
