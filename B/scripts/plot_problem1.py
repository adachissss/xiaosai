from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

COLORS = {
    "pv":         "#F5A623",
    "pv_curtail": "#D4A039",
    "bat_ch":     "#3B82F6",
    "bat_dis":    "#93C5FD",
    "bat_soc":    "#7C3AED",
    "ev_ch":      "#F59E0B",
    "ev_dis":     "#6366F1",
    "grid_buy":   "#EF4444",
    "grid_sell":  "#F97316",
    "load":       "#6B7280",
    "shift_down": "#4F46E5",
    "shift_up":   "#C4B5FD",
    "shed":       "#EC4899",
    "baseline":   "#64748B",
    "coord":      "#DC2626",
    "limit":      "#7F1D1D",
    "bld_office": "#4F46E5",
    "bld_wet":    "#7C3AED",
    "bld_teach":  "#DB2777",
    "ch_power":   "#F97316",
    "dis_power":  "#06B6D4",
}
A = 0.78


def _read(d: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(d / name, parse_dates=["timestamp"])


def _day_lines(ax, t):
    dates = pd.DatetimeIndex(t).normalize().unique()
    for d in dates[1:]:
        ax.axvline(d, color="#D1D5DB", linewidth=0.8, zorder=0)


def _fmt_x(ax):
    ax.xaxis.set_major_locator(mpl.dates.DayLocator())
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%m/%d"))
    ax.xaxis.set_minor_locator(mpl.dates.HourLocator(byhour=[6, 12, 18]))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)


def _legend(ax, loc="upper right", ncol=1):
    ax.legend(loc=loc, ncol=ncol, fontsize=8.5, framealpha=0.88)


# ── Fig 1: 供给侧功率平衡堆叠 ──────────────────────────────────
def fig1_supply_stack(coord: pd.DataFrame, out: Path) -> None:
    t     = coord["timestamp"].to_numpy()
    pv    = coord["pv_used_kw"].to_numpy()
    bat_d = coord["battery_discharge_kw"].to_numpy()
    ev_d  = coord["ev_discharge_total_kw"].to_numpy()
    grid  = coord["grid_buy_kw"].to_numpy()
    load  = coord["adjusted_load_kw"].to_numpy()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title("Coordinated Operation — Supply-Side Power Stack", fontsize=12, fontweight="bold")
    ax.stackplot(t, pv, bat_d, ev_d, grid,
                 labels=["PV Utilized", "Battery Discharge", "EV Discharge (V2B)", "Grid Import"],
                 colors=[COLORS["pv"], COLORS["bat_dis"], COLORS["ev_dis"], COLORS["grid_buy"]],
                 alpha=A)
    ax.plot(t, load, color="black", linewidth=1.4, linestyle="--", label="Adjusted Load", zorder=6)
    ax.set_ylabel("Power [kW]", fontsize=10)
    _day_lines(ax, t)
    _fmt_x(ax)
    _legend(ax, loc="upper right", ncol=1)
    fig.savefig(out / "fig1_supply_stack.png")
    plt.close(fig)
    print("[v] fig1_supply_stack.png")


# ── Fig 2: 需求侧消纳堆叠 ──────────────────────────────────────
def fig2_sink_stack(coord: pd.DataFrame, out: Path) -> None:
    t     = coord["timestamp"].to_numpy()
    bat_c = coord["battery_charge_kw"].to_numpy()
    ev_c  = coord["ev_charge_total_kw"].to_numpy()
    sell  = coord["grid_sell_kw"].to_numpy()
    curtl = coord["pv_curtail_kw"].to_numpy()

    stacks = [bat_c, ev_c]
    labels = ["Battery Charge", "EV Charge"]
    colors = [COLORS["bat_ch"], COLORS["ev_ch"]]
    if sell.max() > 0.1:
        stacks.append(sell);  labels.append("Grid Export");  colors.append(COLORS["grid_sell"])
    if curtl.max() > 0.1:
        stacks.append(curtl); labels.append("PV Curtailed"); colors.append(COLORS["pv_curtail"])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title("Coordinated Operation — Demand-Side / Flexible Sink Stack", fontsize=12, fontweight="bold")
    ax.stackplot(t, *stacks, labels=labels, colors=colors, alpha=A)
    ax.set_ylabel("Power [kW]", fontsize=10)
    _day_lines(ax, t)
    _fmt_x(ax)
    _legend(ax, loc="upper right", ncol=1)
    fig.savefig(out / "fig2_sink_stack.png")
    plt.close(fig)
    print("[v] fig2_sink_stack.png")


# ── Fig 3: 储能 SOC (最重要，单图) ────────────────────────────
def fig3_battery_soc(coord: pd.DataFrame, out: Path) -> None:
    t   = coord["timestamp"].to_numpy()
    soc = coord["battery_energy_kwh"].to_numpy()
    ch  = coord["battery_charge_kw"].to_numpy()
    dis = coord["battery_discharge_kw"].to_numpy()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title("Stationary Battery — SOC & Charge/Discharge Power", fontsize=12, fontweight="bold")

    ax.fill_between(t,  ch,  0, where=(ch  > 0), color=COLORS["ch_power"],  alpha=0.42, label="Charge Power [kW]")
    ax.fill_between(t, -dis, 0, where=(dis > 0), color=COLORS["dis_power"], alpha=0.42, label="Discharge Power [kW]")
    ax.axhline( 300, color="#E5E7EB", linewidth=0.7, linestyle=":")
    ax.axhline(-300, color="#E5E7EB", linewidth=0.7, linestyle=":")
    ax.axhline(0, color="#9CA3AF", linewidth=0.8)
    ax.set_ylabel("Charge / Discharge Power [kW]", fontsize=10, color="#374151")
    ax.tick_params(axis="y", labelcolor="#374151")

    ax2 = ax.twinx()
    ax2.plot(t, soc, color=COLORS["bat_soc"], linewidth=2.2, label="Battery SOC [kWh]", zorder=7)
    ax2.axhline(1080, color="#A78BFA", linewidth=0.9, linestyle=":", label="Max SOC (1080 kWh)")
    ax2.axhline(120,  color="#A78BFA", linewidth=0.9, linestyle=":", label="Min SOC (120 kWh)")
    ax2.set_ylabel("State of Charge [kWh]", fontsize=10, color=COLORS["bat_soc"])
    ax2.tick_params(axis="y", labelcolor=COLORS["bat_soc"])
    ax2.spines["top"].set_visible(False)

    _day_lines(ax, t)
    _fmt_x(ax)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", ncol=1, fontsize=8.5, framealpha=0.88)
    fig.savefig(out / "fig3_battery_soc.png")
    plt.close(fig)
    print("[v] fig3_battery_soc.png")


# ── Fig 4: EV 聚合净功率 ───────────────────────────────────────
def fig4_ev_aggregate(coord: pd.DataFrame, out: Path) -> None:
    t   = coord["timestamp"].to_numpy()
    ch  = coord["ev_charge_total_kw"].to_numpy()
    dis = coord["ev_discharge_total_kw"].to_numpy()
    net = coord["ev_net_kw"].to_numpy()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title("Aggregated EV Power  (+ Charging,  - V2B Discharging)", fontsize=12, fontweight="bold")
    ax.fill_between(t,  ch,  0, where=(ch  > 0), color=COLORS["ev_ch"],  alpha=0.65, label="EV Charge [kW]")
    ax.fill_between(t, -dis, 0, where=(dis > 0), color=COLORS["ev_dis"], alpha=0.65, label="EV Discharge V2B [kW]")
    ax.plot(t, net, color="#111827", linewidth=1.3, label="Net EV Power [kW]", zorder=5)
    ax.axhline(0, color="#9CA3AF", linewidth=0.8)
    ax.set_ylabel("Power [kW]", fontsize=10)
    _day_lines(ax, t)
    _fmt_x(ax)
    _legend(ax, loc="upper right", ncol=1)
    fig.savefig(out / "fig4_ev_aggregate.png")
    plt.close(fig)
    print("[v] fig4_ev_aggregate.png")


# ── Fig 5: 电网购电对比 (最重要，单图) ────────────────────────
def fig5_grid_comparison(baseline: pd.DataFrame, coord: pd.DataFrame, out: Path) -> None:
    t      = coord["timestamp"].to_numpy()
    bl_buy = baseline["grid_buy_kw"].to_numpy()
    co_buy = coord["grid_buy_kw"].to_numpy()
    limit  = coord["grid_import_limit_kw"].to_numpy() if "grid_import_limit_kw" in coord.columns else np.full(len(t), 900.0)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title("Grid Import Power — Baseline vs Coordinated", fontsize=12, fontweight="bold")
    ax.fill_between(t, bl_buy, alpha=0.18, color=COLORS["baseline"])
    ax.plot(t, bl_buy, color=COLORS["baseline"], linewidth=1.2, label="Baseline")
    ax.plot(t, co_buy, color=COLORS["coord"],    linewidth=1.6, label="Coordinated", zorder=5)
    ax.step(t, limit,  color=COLORS["limit"],    linewidth=1.0, linestyle="--", where="post", label="Import Limit")

    ymax = max(float(bl_buy.max()), float(co_buy.max())) * 1.10
    special = [
        ("2025-07-17 13:00", "2025-07-17 16:00", "650 kW limit (07-17)"),
        ("2025-07-18 17:00", "2025-07-18 19:00", "700 kW limit (07-18)"),
    ]
    for s, e, lbl in special:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.10, color="#DC2626", zorder=0)
        mid = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2
        ax.text(mid, ymax * 0.94, lbl, fontsize=7.5, color="#7F1D1D", ha="center", va="top")

    ax.set_ylabel("Power [kW]", fontsize=10)
    ax.set_ylim(0, ymax)
    _day_lines(ax, t)
    _fmt_x(ax)
    _legend(ax, loc="upper right", ncol=1)
    fig.savefig(out / "fig5_grid_comparison.png")
    plt.close(fig)
    print("[v] fig5_grid_comparison.png")


# ── Fig 6: 光伏消纳 ─────────────────────────────────────────────
def fig6_pv_utilization(coord: pd.DataFrame, out: Path) -> None:
    t     = coord["timestamp"].to_numpy()
    avail = coord["pv_available_kw"].to_numpy()
    used  = coord["pv_used_kw"].to_numpy()
    curtl = coord["pv_curtail_kw"].to_numpy()
    sell  = coord["grid_sell_kw"].to_numpy()

    stacks = [used]
    labels = ["PV Used (local)"]
    colors = [COLORS["pv"]]
    if sell.max() > 0.1:
        stacks.append(sell);  labels.append("PV -> Grid Export"); colors.append(COLORS["grid_sell"])
    if curtl.max() > 0.1:
        stacks.append(curtl); labels.append("PV Curtailed");      colors.append(COLORS["pv_curtail"])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title("PV Utilization — Local Use / Grid Export / Curtailed", fontsize=12, fontweight="bold")
    ax.fill_between(t, avail, alpha=0.12, color=COLORS["pv"])
    ax.plot(t, avail, color=COLORS["pv"], linewidth=1.1, linestyle="--", label="PV Available")
    ax.stackplot(t, *stacks, labels=labels, colors=colors, alpha=0.85)

    total_avail = float((avail * 0.25).sum())
    total_used  = float((used  * 0.25).sum())
    total_curtl = float((curtl * 0.25).sum())
    rate_used   = total_used  / total_avail * 100 if total_avail else 0.0
    rate_curtl  = total_curtl / total_avail * 100 if total_avail else 0.0
    stats = f"Consumption Rate: {rate_used:.1f}%\nCurtailment Rate: {rate_curtl:.1f}%"
    ax.text(0.01, 0.97, stats, transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC", alpha=0.92))
    ax.set_ylabel("Power [kW]", fontsize=10)
    _day_lines(ax, t)
    _fmt_x(ax)
    _legend(ax, loc="upper right", ncol=1)
    fig.savefig(out / "fig6_pv_utilization.png")
    plt.close(fig)
    print("[v] fig6_pv_utilization.png")


# ── Fig 7: 建筑柔性负荷 (3栋横排) ─────────────────────────────
def fig7_flexible_load(coord: pd.DataFrame, out: Path) -> None:
    t = coord["timestamp"].to_numpy()
    buildings = [
        ("office_building", "Office Building", COLORS["bld_office"]),
        ("wet_lab",         "Wet Lab",          COLORS["bld_wet"]),
        ("teaching_center", "Teaching Center",  COLORS["bld_teach"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharey=False)
    fig.suptitle("Flexible Building Load Adjustments", fontsize=12, fontweight="bold", y=1.02)

    for ax, (bld, title, color) in zip(axes, buildings):
        col_sd = f"{bld}_shift_down_kw"
        col_su = f"{bld}_shift_up_kw"
        col_sh = f"{bld}_shed_kw"

        if col_sd in coord.columns:
            sd = coord[col_sd].to_numpy()
            su = coord[col_su].to_numpy()
            sh = coord[col_sh].to_numpy()
        else:
            sd = coord["load_shift_down_kw"].to_numpy() / 3
            su = coord["load_shift_up_kw"].to_numpy()   / 3
            sh = coord["load_shed_kw"].to_numpy()       / 3

        any_data = sd.max() + su.max() + sh.max() > 0.01
        if any_data:
            ax.fill_between(t, -sd,        0,  alpha=0.70, color=color,          label="Shift Out")
            ax.fill_between(t,  su,        0,  alpha=0.50, color=color,          label="Shift In")
            ax.fill_between(t, -(sd + sh), -sd, alpha=0.85, color=COLORS["shed"], label="Load Shed")
            _legend(ax, loc="upper right", ncol=1)
        else:
            ax.text(0.5, 0.5, "No adjustment\n(not dispatched)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#9CA3AF")

        ax.axhline(0, color="#9CA3AF", linewidth=0.8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("Power [kW]", fontsize=9)
        _day_lines(ax, t)
        _fmt_x(ax)

    plt.tight_layout()
    fig.savefig(out / "fig7_flexible_load.png")
    plt.close(fig)
    print("[v] fig7_flexible_load.png")


# ── Fig 8: 成本构成对比 ─────────────────────────────────────────
def fig8_cost_breakdown(bl_m: dict, co_m: dict, out: Path) -> None:
    all_keys = [
        ("grid_buy_cost_cny",       "Grid Import Cost"),
        ("grid_sell_revenue_cny",   "Grid Export Revenue (saving)"),
        ("shed_penalty_cny",        "Load Shed Penalty"),
        ("shift_penalty_cny",       "Load Shift Penalty"),
        ("curtail_penalty_cny",     "PV Curtail Penalty"),
        ("peak_import_penalty_cny", "Peak Import Penalty"),
    ]

    labels, bl_vals, co_vals = [], [], []
    for k, lbl in all_keys:
        b = abs(float(bl_m.get(k, 0.0)))
        c = abs(float(co_m.get(k, 0.0)))
        if max(b, c) < 1.0:
            continue
        labels.append(lbl)
        bl_vals.append(b)
        co_vals.append(c)

    if not labels:
        print("[!] fig8: no significant cost items, skipping")
        return

    bl_arr = np.array(bl_vals)
    co_arr = np.array(co_vals)
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 1.0 + 1.5)))
    ax.set_title("Cost Breakdown — Baseline vs Coordinated", fontsize=12, fontweight="bold")
    ax.barh(x + width / 2, bl_arr, width, label="Baseline",    color=COLORS["baseline"], alpha=0.85)
    ax.barh(x - width / 2, co_arr, width, label="Coordinated", color=COLORS["coord"],    alpha=0.80)

    for i, (b, c) in enumerate(zip(bl_arr, co_arr)):
        if b > 1e-3:
            delta = (b - c) / b * 100
            marker = "v" if delta > 0 else "^"
            ax.text(max(b, c) * 1.01, i, f"  {marker}{abs(delta):.1f}%",
                    va="center", fontsize=8, color="#374151")

    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Cost [CNY]", fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()
    _legend(ax, loc="lower right", ncol=1)
    plt.tight_layout()
    fig.savefig(out / "fig8_cost_breakdown.png")
    plt.close(fig)
    print("[v] fig8_cost_breakdown.png")


# ── Fig 9: 每日购电量对比 ──────────────────────────────────────
def fig9_daily_import(baseline: pd.DataFrame, coord: pd.DataFrame, out: Path) -> None:
    DT = 0.25
    coord    = coord.copy()
    baseline = baseline.copy()
    coord["date"]    = coord["timestamp"].dt.date
    baseline["date"] = baseline["timestamp"].dt.date

    mc = coord.groupby("date")["grid_buy_kw"].apply(lambda x: (x * DT).sum())
    mb = baseline.groupby("date")["grid_buy_kw"].apply(lambda x: (x * DT).sum())

    dates = [str(d) for d in mc.index]
    x = np.arange(len(dates))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_title("Daily Grid Import Energy — Baseline vs Coordinated", fontsize=12, fontweight="bold")
    ax.bar(x - width / 2, mb.values, width, label="Baseline",    color=COLORS["baseline"], alpha=0.85)
    ax.bar(x + width / 2, mc.values, width, label="Coordinated", color=COLORS["coord"],    alpha=0.80)
    ax.set_xticks(x)
    ax.set_xticklabels([d[5:] for d in dates], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Energy [kWh]", fontsize=10)
    _legend(ax, loc="upper right", ncol=1)
    plt.tight_layout()
    fig.savefig(out / "fig9_daily_import.png")
    plt.close(fig)
    print("[v] fig9_daily_import.png")


# ── 主入口 ──────────────────────────────────────────────────────
def load_metrics(results_dir: Path) -> tuple[dict, dict]:
    p = results_dir / "comparison_metrics.csv"
    if not p.exists():
        return {}, {}
    df = pd.read_csv(p)
    bl = df[df["scheme"] == "baseline"].iloc[0].to_dict()    if "baseline"    in df["scheme"].values else {}
    co = df[df["scheme"] == "coordinated"].iloc[0].to_dict() if "coordinated" in df["scheme"].values else {}
    return bl, co


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "results")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    out_dir: Path     = args.out_dir or (results_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading results from: {results_dir}")
    coord    = _read(results_dir, "coordinated_schedule.csv")
    baseline = _read(results_dir, "baseline_schedule.csv")
    bl_m, co_m = load_metrics(results_dir)

    fig1_supply_stack(coord, out_dir)
    fig2_sink_stack(coord, out_dir)
    fig3_battery_soc(coord, out_dir)
    fig4_ev_aggregate(coord, out_dir)
    fig5_grid_comparison(baseline, coord, out_dir)
    fig6_pv_utilization(coord, out_dir)
    fig7_flexible_load(coord, out_dir)
    if bl_m and co_m:
        fig8_cost_breakdown(bl_m, co_m, out_dir)
    fig9_daily_import(baseline, coord, out_dir)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()