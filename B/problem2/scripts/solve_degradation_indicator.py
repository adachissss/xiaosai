from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from degradation_model import (
    DEFAULT_STATIONARY_BATTERY_DEG_COST,
    evaluate_scheme,
    load_ev_degradation_parameter_summary,
    load_problem1_scheme,
    result_to_dict,
)

SCHEMES = [
    "S0_no_storage",
    "S1_rule_storage",
    "S2_partial_coordination",
    "S3_full_coordination",
]

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})


def plot_degradation_metrics(metrics: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x = range(len(metrics))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x, metrics["battery_throughput_kwh"], label="Stationary Battery", alpha=0.85)
    ax.bar(x, metrics["ev_throughput_kwh"], bottom=metrics["battery_throughput_kwh"], label="EV Fleet", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics["scheme"], rotation=20, ha="right")
    ax.set_ylabel("Throughput [kWh]")
    ax.set_title("Battery Throughput Degradation Indicator")
    ax.legend()
    fig.savefig(out_dir / "p2_q1_throughput_indicator.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x, metrics["battery_degradation_cost_cny"], label="Stationary Battery", alpha=0.85)
    ax.bar(x, metrics["ev_degradation_cost_cny"], bottom=metrics["battery_degradation_cost_cny"], label="EV Fleet", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics["scheme"], rotation=20, ha="right")
    ax.set_ylabel("Degradation Cost [CNY]")
    ax.set_title("Equivalent Battery Degradation Cost")
    ax.legend()
    fig.savefig(out_dir / "p2_q1_degradation_cost.png")
    plt.close(fig)


def write_summary(metrics: pd.DataFrame, ev_param_summary: pd.DataFrame, out_path: Path, battery_unit_cost: float) -> None:
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
        f"固定储能单位吞吐寿命成本暂取：{battery_unit_cost:.4f} 元/kWh。",
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
    parser.add_argument("--battery-unit-cost", type=float, default=DEFAULT_STATIONARY_BATTERY_DEG_COST)
    args = parser.parse_args()

    rows = []
    for scheme in SCHEMES:
        schedule, ev_result = load_problem1_scheme(args.problem1_results_dir, scheme)
        rows.append(result_to_dict(evaluate_scheme(scheme, schedule, ev_result, args.data_dir, args.battery_unit_cost)))
    metrics = pd.DataFrame(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ev_param_summary = load_ev_degradation_parameter_summary(args.data_dir)
    metrics.to_csv(args.out_dir / "p2_q1_degradation_metrics.csv", index=False)
    ev_param_summary.to_csv(args.out_dir / "p2_q1_ev_degradation_parameters.csv", index=False)
    plot_degradation_metrics(metrics, fig_dir)
    write_summary(metrics, ev_param_summary, args.out_dir / "p2_q1_summary.txt", args.battery_unit_cost)

    print((args.out_dir / "p2_q1_summary.txt").read_text(encoding="utf-8"))
    print(f"\nFigures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
