from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "legend.framealpha": 0.88,
    "legend.fontsize": 8.5,
})

DT = 0.25
SCHEMES = {
    "S0_no_storage": "S0 No Storage",
    "S1_rule_storage": "S1 Rule Storage",
    "S2_partial_coordination": "S2 Partial Coordination",
    "S3_full_coordination": "S3 Full Coordination",
}
COLORS = {
    "S0_no_storage": "#64748B",
    "S1_rule_storage": "#2563EB",
    "S2_partial_coordination": "#F97316",
    "S3_full_coordination": "#DC2626",
    "pv": "#F5A623",
    "pv_curtail": "#D4A039",
    "bat_ch": "#3B82F6",
    "bat_dis": "#93C5FD",
    "bat_soc": "#7C3AED",
    "ev_ch": "#F59E0B",
    "ev_dis": "#6366F1",
    "grid_buy": "#EF4444",
    "grid_sell": "#F97316",
    "load": "#111827",
    "limit": "#7F1D1D",
    "shed": "#EC4899",
    "bld_office": "#4F46E5",
    "bld_wet": "#7C3AED",
    "bld_teach": "#DB2777",
}


def _read_schedule(results_dir: Path, scheme: str) -> pd.DataFrame:
    return pd.read_csv(results_dir / f"{scheme}_schedule.csv", parse_dates=["timestamp"])


def load_schedules(results_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [s for s in SCHEMES if not (results_dir / f"{s}_schedule.csv").exists()]
    if missing:
        names = ", ".join(f"{s}_schedule.csv" for s in missing)
        raise FileNotFoundError(f"Missing schedule files in {results_dir}: {names}")
    return {scheme: _read_schedule(results_dir, scheme) for scheme in SCHEMES}


def load_metrics(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "comparison_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _day_lines(ax, t) -> None:
    dates = pd.DatetimeIndex(t).normalize().unique()
    for d in dates[1:]:
        ax.axvline(d, color="#D1D5DB", linewidth=0.8, zorder=0)


def _fmt_x(ax) -> None:
    ax.xaxis.set_major_locator(mpl.dates.DayLocator())
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%m/%d"))
    ax.xaxis.set_minor_locator(mpl.dates.HourLocator(byhour=[6, 12, 18]))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)


def _legend(ax, loc="upper right", ncol=1) -> None:
    ax.legend(loc=loc, ncol=ncol, fontsize=8.5, framealpha=0.88)


def fig1_grid_import_comparison(schedules: dict[str, pd.DataFrame], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title("Grid Import Power Comparison Across Four Schemes", fontsize=12, fontweight="bold")
    for scheme, label in SCHEMES.items():
        df = schedules[scheme]
        ax.plot(df["timestamp"], df["grid_buy_kw"], label=label, color=COLORS[scheme], linewidth=1.2)
    t = schedules["S3_full_coordination"]["timestamp"]
    _day_lines(ax, t)
    _fmt_x(ax)
    ax.set_ylabel("Power [kW]")
    _legend(ax, ncol=2)
    fig.savefig(out / "fig1_grid_import_comparison.png")
    plt.close(fig)
    print("[v] fig1_grid_import_comparison.png")


def fig2_battery_soc_comparison(schedules: dict[str, pd.DataFrame], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title("Stationary Battery Energy Comparison", fontsize=12, fontweight="bold")
    for scheme, label in SCHEMES.items():
        df = schedules[scheme]
        ax.plot(df["timestamp"], df["battery_energy_kwh"], label=label, color=COLORS[scheme], linewidth=1.2)
    ax.axhline(1080, color="#A78BFA", linewidth=0.9, linestyle=":", label="Max Energy")
    ax.axhline(120, color="#A78BFA", linewidth=0.9, linestyle=":", label="Min Energy")
    t = schedules["S3_full_coordination"]["timestamp"]
    _day_lines(ax, t)
    _fmt_x(ax)
    ax.set_ylabel("Energy [kWh]")
    _legend(ax, ncol=2)
    fig.savefig(out / "fig2_battery_soc_comparison.png")
    plt.close(fig)
    print("[v] fig2_battery_soc_comparison.png")


def fig3_ev_net_comparison(schedules: dict[str, pd.DataFrame], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title("Aggregated EV Net Power Comparison (+ Charge, - V2B)", fontsize=12, fontweight="bold")
    for scheme, label in SCHEMES.items():
        df = schedules[scheme]
        ax.plot(df["timestamp"], df["ev_net_kw"], label=label, color=COLORS[scheme], linewidth=1.2)
    ax.axhline(0, color="#9CA3AF", linewidth=0.8)
    t = schedules["S3_full_coordination"]["timestamp"]
    _day_lines(ax, t)
    _fmt_x(ax)
    ax.set_ylabel("Power [kW]")
    _legend(ax, ncol=2)
    fig.savefig(out / "fig3_ev_net_comparison.png")
    plt.close(fig)
    print("[v] fig3_ev_net_comparison.png")


def fig4_s3_supply_stack(s3: pd.DataFrame, out: Path) -> None:
    t = s3["timestamp"].to_numpy()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title("S3 Full Coordination — Supply-Side Power Stack", fontsize=12, fontweight="bold")
    ax.stackplot(
        t,
        s3["pv_used_kw"],
        s3["battery_discharge_kw"],
        s3["ev_discharge_total_kw"],
        s3["grid_buy_kw"],
        labels=["PV Used", "Battery Discharge", "EV Discharge", "Grid Import"],
        colors=[COLORS["pv"], COLORS["bat_dis"], COLORS["ev_dis"], COLORS["grid_buy"]],
        alpha=0.78,
    )
    ax.plot(t, s3["adjusted_load_kw"], color=COLORS["load"], linewidth=1.3, linestyle="--", label="Adjusted Load")
    _day_lines(ax, t)
    _fmt_x(ax)
    ax.set_ylabel("Power [kW]")
    _legend(ax)
    fig.savefig(out / "fig4_s3_supply_stack.png")
    plt.close(fig)
    print("[v] fig4_s3_supply_stack.png")


def fig5_pv_utilization(s3: pd.DataFrame, out: Path) -> None:
    t = s3["timestamp"].to_numpy()
    avail = s3["pv_available_kw"].to_numpy()
    used = s3["pv_used_kw"].to_numpy()
    curtail = s3["pv_curtail_kw"].to_numpy()
    sell = s3["grid_sell_kw"].to_numpy()

    stacks = [used]
    labels = ["PV Used"]
    colors = [COLORS["pv"]]
    if sell.max() > 0.1:
        stacks.append(sell)
        labels.append("Grid Export")
        colors.append(COLORS["grid_sell"])
    if curtail.max() > 0.1:
        stacks.append(curtail)
        labels.append("PV Curtailed")
        colors.append(COLORS["pv_curtail"])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title("S3 Full Coordination — PV Utilization", fontsize=12, fontweight="bold")
    ax.fill_between(t, avail, alpha=0.12, color=COLORS["pv"])
    ax.plot(t, avail, color=COLORS["pv"], linewidth=1.1, linestyle="--", label="PV Available")
    ax.stackplot(t, *stacks, labels=labels, colors=colors, alpha=0.85)
    total_avail = float((avail * DT).sum())
    total_used = float((used * DT).sum())
    total_curtail = float((curtail * DT).sum())
    text = f"Consumption Rate: {total_used / total_avail * 100:.1f}%\nCurtailment Rate: {total_curtail / total_avail * 100:.1f}%"
    ax.text(0.01, 0.97, text, transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC", alpha=0.92))
    _day_lines(ax, t)
    _fmt_x(ax)
    ax.set_ylabel("Power [kW]")
    _legend(ax)
    fig.savefig(out / "fig5_pv_utilization.png")
    plt.close(fig)
    print("[v] fig5_pv_utilization.png")


def fig6_flexible_load(s3: pd.DataFrame, out: Path) -> None:
    t = s3["timestamp"].to_numpy()
    buildings = [
        ("office_building", "Office Building", COLORS["bld_office"]),
        ("wet_lab", "Wet Lab", COLORS["bld_wet"]),
        ("teaching_center", "Teaching Center", COLORS["bld_teach"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharey=False)
    fig.suptitle("S3 Full Coordination — Flexible Building Load Adjustments", fontsize=12, fontweight="bold", y=1.02)
    for ax, (bld, title, color) in zip(axes, buildings):
        sd = s3[f"{bld}_shift_down_kw"].to_numpy()
        su = s3[f"{bld}_shift_up_kw"].to_numpy()
        sh = s3[f"{bld}_shed_kw"].to_numpy()
        if sd.max() + su.max() + sh.max() > 0.01:
            ax.fill_between(t, -sd, 0, alpha=0.70, color=color, label="Shift Out")
            ax.fill_between(t, su, 0, alpha=0.50, color=color, label="Shift In")
            ax.fill_between(t, -(sd + sh), -sd, alpha=0.85, color=COLORS["shed"], label="Load Shed")
            _legend(ax)
        else:
            ax.text(0.5, 0.5, "No adjustment", ha="center", va="center", transform=ax.transAxes, fontsize=9, color="#9CA3AF")
        ax.axhline(0, color="#9CA3AF", linewidth=0.8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("Power [kW]")
        _day_lines(ax, t)
        _fmt_x(ax)
    plt.tight_layout()
    fig.savefig(out / "fig6_flexible_load.png")
    plt.close(fig)
    print("[v] fig6_flexible_load.png")


def fig7_cost_comparison(metrics: pd.DataFrame, out: Path) -> None:
    if metrics.empty:
        print("[!] fig7: comparison_metrics.csv not found, skipping")
        return
    metric_cols = [
        ("total_cost_cny", "Total Cost"),
        ("grid_buy_cost_cny", "Grid Buy Cost"),
        ("peak_import_penalty_cny", "Peak Penalty"),
        ("shed_penalty_cny", "Shed Penalty"),
        ("shift_penalty_cny", "Shift Penalty"),
    ]
    metrics = metrics.set_index("scheme")
    x = np.arange(len(SCHEMES))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Cost Component Comparison Across Four Schemes", fontsize=12, fontweight="bold")
    for j, (col, label) in enumerate(metric_cols):
        if col not in metrics.columns:
            continue
        values = [float(metrics.loc[s, col]) for s in SCHEMES]
        ax.bar(x + (j - 2) * width, values, width, label=label, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([label.split(" ", 1)[0] for label in SCHEMES.values()])
    ax.set_ylabel("CNY")
    _legend(ax, ncol=2)
    fig.savefig(out / "fig7_cost_comparison.png")
    plt.close(fig)
    print("[v] fig7_cost_comparison.png")


def fig8_daily_import_comparison(schedules: dict[str, pd.DataFrame], out: Path) -> None:
    daily = {}
    for scheme, df in schedules.items():
        temp = df.copy()
        temp["date"] = temp["timestamp"].dt.date
        daily[scheme] = temp.groupby("date")["grid_buy_kw"].apply(lambda x: (x * DT).sum())
    dates = [str(d)[5:] for d in next(iter(daily.values())).index]
    x = np.arange(len(dates))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Daily Grid Import Energy Comparison", fontsize=12, fontweight="bold")
    for i, (scheme, label) in enumerate(SCHEMES.items()):
        ax.bar(x + (i - 1.5) * width, daily[scheme].values, width, label=label, color=COLORS[scheme], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=30, ha="right")
    ax.set_ylabel("Energy [kWh]")
    _legend(ax, ncol=2)
    fig.savefig(out / "fig8_daily_import_comparison.png")
    plt.close(fig)
    print("[v] fig8_daily_import_comparison.png")


def fig9_summary_metrics(metrics: pd.DataFrame, out: Path) -> None:
    if metrics.empty:
        print("[!] fig9: comparison_metrics.csv not found, skipping")
        return
    metrics = metrics.set_index("scheme")
    cols = [
        ("peak_grid_import_kw", "Peak Import [kW]"),
        ("ev_shortfall_kwh", "EV Shortfall [kWh]"),
        ("load_shift_down_kwh", "Load Shift Out [kWh]"),
        ("load_shed_kwh", "Load Shed [kWh]"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Key Metric Comparison Across Four Schemes", fontsize=12, fontweight="bold", y=1.02)
    x = np.arange(len(SCHEMES))
    labels = [label.split(" ", 1)[0] for label in SCHEMES.values()]
    for ax, (col, title) in zip(axes.ravel(), cols):
        values = [float(metrics.loc[s, col]) for s in SCHEMES]
        ax.bar(x, values, color=[COLORS[s] for s in SCHEMES], alpha=0.85)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    plt.tight_layout()
    fig.savefig(out / "fig9_summary_metrics.png")
    plt.close(fig)
    print("[v] fig9_summary_metrics.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent.parent / "results")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir or (results_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading results from: {results_dir}")
    schedules = load_schedules(results_dir)
    metrics = load_metrics(results_dir)
    s3 = schedules["S3_full_coordination"]

    fig1_grid_import_comparison(schedules, out_dir)
    fig2_battery_soc_comparison(schedules, out_dir)
    fig3_ev_net_comparison(schedules, out_dir)
    fig4_s3_supply_stack(s3, out_dir)
    fig5_pv_utilization(s3, out_dir)
    fig6_flexible_load(s3, out_dir)
    fig7_cost_comparison(metrics, out_dir)
    fig8_daily_import_comparison(schedules, out_dir)
    fig9_summary_metrics(metrics, out_dir)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
