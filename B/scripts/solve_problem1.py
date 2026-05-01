#!/usr/bin/env python3
"""
B题第一问：园区源-荷-储-车协同运行方案

输出：
- results/baseline_schedule.csv
- results/coordinated_schedule.csv
- results/comparison_metrics.csv
- results/problem1_summary.txt
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix
except ImportError as exc:
    raise SystemExit(
        "缺少 scipy。请在 B 目录下运行：\n"
        "  conda env create -f environment.yml\n"
        "  conda activate xiaosai-b\n"
        "  python scripts/solve_problem1.py\n"
    ) from exc


DT = 0.25
BAT_EFF_CH = 0.95
BAT_EFF_DIS = 0.95
EV_EFF_CH = 0.95
EV_EFF_DIS = 0.95
SHIFT_PENALTY = 0.05
CURTAIL_PENALTY = 0.01
HIGH_PRICE_THRESHOLD = 0.90
BIG_PENALTY = 1_000.0
PEAK_IMPORT_PENALTY = 8.0


@dataclass
class ProblemData:
    ts: pd.DataFrame
    ev: pd.DataFrame
    asset: dict
    flex: pd.DataFrame
    time_index: pd.DatetimeIndex


def read_data(data_dir: Path) -> ProblemData:
    ts = pd.read_csv(data_dir / "timeseries_15min.csv")
    ev = pd.read_csv(data_dir / "ev_sessions.csv")
    asset_df = pd.read_csv(data_dir / "asset_parameters.csv")
    flex = pd.read_csv(data_dir / "flexible_load_parameters.csv")

    ts["timestamp"] = pd.to_datetime(ts["timestamp"])
    ev["arrival_time"] = pd.to_datetime(ev["arrival_time"])
    ev["departure_time"] = pd.to_datetime(ev["departure_time"])

    if ev["session_id"].nunique() != len(ev):
        raise ValueError("ev_sessions.csv 中 session_id 不唯一，不能直接作为 EV 调度对象编号。")

    asset = dict(zip(asset_df["parameter"], asset_df["value"]))
    time_index = pd.DatetimeIndex(ts["timestamp"])
    return ProblemData(ts=ts, ev=ev, asset=asset, flex=flex, time_index=time_index)


def time_to_idx_map(time_index: pd.DatetimeIndex) -> dict[pd.Timestamp, int]:
    return {t: i for i, t in enumerate(time_index)}


def add_ev_indices(data: ProblemData) -> pd.DataFrame:
    ev = data.ev.copy()
    mapping = time_to_idx_map(data.time_index)
    ev["arrival_idx"] = ev["arrival_time"].map(mapping)
    ev["departure_idx"] = ev["departure_time"].map(mapping)
    if ev[["arrival_idx", "departure_idx"]].isna().any().any():
        raise ValueError("部分 EV 到离站时间无法映射到 15 分钟时间索引。")
    ev["arrival_idx"] = ev["arrival_idx"].astype(int)
    ev["departure_idx"] = ev["departure_idx"].astype(int)
    ev["ev_index"] = np.arange(len(ev))
    return ev


def simulate_baseline(data: ProblemData) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = data.ts.reset_index(drop=True)
    ev = add_ev_indices(data)
    T = len(ts)
    N = len(ev)

    bat_min = float(data.asset["stationary_battery_min_energy_kwh"])
    bat_max = float(data.asset["stationary_battery_max_energy_kwh"])
    bat_e = float(data.asset["stationary_battery_initial_energy_kwh"])
    bat_p_ch_max = float(data.asset["stationary_battery_max_charge_power_kw"])
    bat_p_dis_max = float(data.asset["stationary_battery_max_discharge_power_kw"])

    ev_e = ev["initial_energy_kwh"].to_numpy(dtype=float).copy()
    ev_ch = np.zeros((N, T))
    ev_dis = np.zeros((N, T))

    for n, row in ev.iterrows():
        need = row.required_energy_at_departure_kwh - ev_e[n]
        if need <= 0:
            continue
        for t in range(row.arrival_idx, row.departure_idx + 1):
            if need <= 1e-9:
                break
            room = row.battery_capacity_kwh - ev_e[n]
            p = min(row.max_charge_power_kw, need / (EV_EFF_CH * DT), room / (EV_EFF_CH * DT))
            if p < 0:
                p = 0
            ev_ch[n, t] = p
            gain = EV_EFF_CH * p * DT
            ev_e[n] += gain
            need -= gain

    bat_ch = np.zeros(T)
    bat_dis = np.zeros(T)
    bat_energy = np.zeros(T)
    grid_buy = np.zeros(T)
    grid_sell = np.zeros(T)
    pv_used = np.zeros(T)
    pv_curtail = np.zeros(T)

    for t in range(T):
        load = ts.loc[t, "total_native_load_kw"]
        ev_load = ev_ch[:, t].sum()
        demand = load + ev_load
        pv = ts.loc[t, "pv_available_kw"]
        buy_price = ts.loc[t, "grid_buy_price_cny_per_kwh"]

        use_pv = min(pv, demand)
        pv_left = pv - use_pv
        demand_left = demand - use_pv

        if pv_left > 0:
            max_ch_by_energy = max(0.0, (bat_max - bat_e) / (BAT_EFF_CH * DT))
            p_ch = min(pv_left, bat_p_ch_max, max_ch_by_energy)
            bat_ch[t] = p_ch
            bat_e += BAT_EFF_CH * p_ch * DT
            pv_left -= p_ch

        if demand_left > 0 and buy_price >= HIGH_PRICE_THRESHOLD:
            max_dis_by_energy = max(0.0, (bat_e - bat_min) * BAT_EFF_DIS / DT)
            p_dis = min(demand_left, bat_p_dis_max, max_dis_by_energy)
            bat_dis[t] = p_dis
            bat_e -= p_dis * DT / BAT_EFF_DIS
            demand_left -= p_dis

        pv_used[t] = use_pv + bat_ch[t]
        grid_buy[t] = min(demand_left, ts.loc[t, "grid_import_limit_kw"])
        unmet = max(0.0, demand_left - grid_buy[t])

        if pv_left > 0:
            grid_sell[t] = min(pv_left, ts.loc[t, "grid_export_limit_kw"])
            pv_left -= grid_sell[t]
        pv_curtail[t] = max(0.0, pv_left)
        bat_energy[t] = bat_e

        if unmet > 1e-6:
            grid_buy[t] += unmet

    schedule = pd.DataFrame({
        "timestamp": ts["timestamp"],
        "native_load_kw": ts["total_native_load_kw"],
        "adjusted_load_kw": ts["total_native_load_kw"],
        "grid_buy_kw": grid_buy,
        "grid_sell_kw": grid_sell,
        "pv_available_kw": ts["pv_available_kw"],
        "pv_used_kw": pv_used,
        "pv_curtail_kw": pv_curtail,
        "battery_charge_kw": bat_ch,
        "battery_discharge_kw": bat_dis,
        "battery_energy_kwh": bat_energy,
        "ev_charge_total_kw": ev_ch.sum(axis=0),
        "ev_discharge_total_kw": ev_dis.sum(axis=0),
        "ev_net_kw": ev_ch.sum(axis=0) - ev_dis.sum(axis=0),
        "load_shift_down_kw": 0.0,
        "load_shift_up_kw": 0.0,
        "load_shed_kw": 0.0,
        "unmet_load_kw": 0.0,
    })

    ev_result = ev[["session_id", "arrival_time", "departure_time", "initial_energy_kwh", "required_energy_at_departure_kwh"]].copy()
    ev_result["final_energy_kwh"] = ev_e
    ev_result["satisfied"] = ev_result["final_energy_kwh"] + 1e-6 >= ev_result["required_energy_at_departure_kwh"]
    return schedule, ev_result


class VarIndex:
    def __init__(self):
        self.slices = {}
        self.size = 0

    def add(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        count = int(np.prod(shape))
        arr = np.arange(self.size, self.size + count).reshape(shape)
        self.slices[name] = arr
        self.size += count
        return arr


def build_and_solve_coordinated(data: ProblemData) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ts = data.ts.reset_index(drop=True)
    ev = add_ev_indices(data)
    flex = data.flex.reset_index(drop=True)
    T = len(ts)
    N = len(ev)
    B = len(flex)

    idx = VarIndex()
    grid_buy = idx.add("grid_buy", (T,))
    grid_sell = idx.add("grid_sell", (T,))
    pv_use = idx.add("pv_use", (T,))
    pv_curtail = idx.add("pv_curtail", (T,))
    bat_ch = idx.add("bat_ch", (T,))
    bat_dis = idx.add("bat_dis", (T,))
    bat_e = idx.add("bat_e", (T,))
    ev_ch = idx.add("ev_ch", (N, T))
    ev_dis = idx.add("ev_dis", (N, T))
    ev_e = idx.add("ev_e", (N, T))
    shift_down = idx.add("shift_down", (B, T))
    shift_up = idx.add("shift_up", (B, T))
    shed = idx.add("shed", (B, T))
    unmet = idx.add("unmet", (T,))
    ev_shortfall = idx.add("ev_shortfall", (N,))
    peak_import = idx.add("peak_import", (1,))

    nvar = idx.size
    c = np.zeros(nvar)
    c[grid_buy] = ts["grid_buy_price_cny_per_kwh"].to_numpy() * DT
    c[grid_sell] = -ts["grid_sell_price_cny_per_kwh"].to_numpy() * DT
    c[pv_curtail] = CURTAIL_PENALTY * DT
    c[unmet] = BIG_PENALTY * DT
    c[ev_shortfall] = BIG_PENALTY
    c[peak_import] = PEAK_IMPORT_PENALTY
    c[shift_down] = SHIFT_PENALTY * DT
    c[shift_up] = SHIFT_PENALTY * DT
    for b, row in flex.iterrows():
        c[shed[b, :]] = float(row.penalty_cny_per_kwh_not_served) * DT

    bounds = [(0.0, None)] * nvar

    for t in range(T):
        bounds[grid_buy[t]] = (0.0, float(ts.loc[t, "grid_import_limit_kw"]))
        bounds[grid_sell[t]] = (0.0, float(ts.loc[t, "grid_export_limit_kw"]))
        bounds[pv_use[t]] = (0.0, float(ts.loc[t, "pv_available_kw"]))
        bounds[pv_curtail[t]] = (0.0, float(ts.loc[t, "pv_available_kw"]))
        bounds[bat_ch[t]] = (0.0, float(data.asset["stationary_battery_max_charge_power_kw"]))
        bounds[bat_dis[t]] = (0.0, float(data.asset["stationary_battery_max_discharge_power_kw"]))
        bounds[bat_e[t]] = (
            float(data.asset["stationary_battery_min_energy_kwh"]),
            float(data.asset["stationary_battery_max_energy_kwh"]),
        )

    for n, row in ev.iterrows():
        for t in range(T):
            in_station = row.arrival_idx <= t <= row.departure_idx
            if in_station:
                bounds[ev_ch[n, t]] = (0.0, float(row.max_charge_power_kw))
                dis_max = float(row.max_discharge_power_kw) if int(row.v2b_allowed) == 1 else 0.0
                bounds[ev_dis[n, t]] = (0.0, dis_max)
                bounds[ev_e[n, t]] = (0.0, float(row.battery_capacity_kwh))
            else:
                bounds[ev_ch[n, t]] = (0.0, 0.0)
                bounds[ev_dis[n, t]] = (0.0, 0.0)
                bounds[ev_e[n, t]] = (0.0, float(row.battery_capacity_kwh))

    for b, row in flex.iterrows():
        for t in range(T):
            native_col = row.load_block + "_kw"
            native = float(ts.loc[t, native_col])
            interruptible = max(0.0, native * (1.0 - float(row.noninterruptible_share)))
            bounds[shift_down[b, t]] = (0.0, min(float(row.max_shiftable_kw), interruptible))
            bounds[shift_up[b, t]] = (0.0, float(row.rebound_factor) * float(row.max_shiftable_kw))
            bounds[shed[b, t]] = (0.0, min(float(row.max_sheddable_kw), interruptible))

    eq_rows: list[dict[int, float]] = []
    b_eq = []
    ub_rows: list[dict[int, float]] = []
    b_ub = []

    def eq(row: dict[int, float], rhs: float):
        eq_rows.append(row)
        b_eq.append(rhs)

    def ub(row: dict[int, float], rhs: float):
        ub_rows.append(row)
        b_ub.append(rhs)

    for t in range(T):
        ub({grid_buy[t]: 1.0, peak_import[0]: -1.0}, 0.0)
        eq({pv_use[t]: 1.0, pv_curtail[t]: 1.0}, float(ts.loc[t, "pv_available_kw"]))

        row = {
            pv_use[t]: 1.0,
            grid_buy[t]: 1.0,
            bat_dis[t]: 1.0,
            unmet[t]: 1.0,
            grid_sell[t]: -1.0,
            bat_ch[t]: -1.0,
        }
        for n in range(N):
            row[ev_dis[n, t]] = row.get(ev_dis[n, t], 0.0) + 1.0
            row[ev_ch[n, t]] = row.get(ev_ch[n, t], 0.0) - 1.0
        for b in range(B):
            row[shift_down[b, t]] = row.get(shift_down[b, t], 0.0) + 1.0
            row[shed[b, t]] = row.get(shed[b, t], 0.0) + 1.0
            row[shift_up[b, t]] = row.get(shift_up[b, t], 0.0) - 1.0
        eq(row, float(ts.loc[t, "total_native_load_kw"]))

        if t == 0:
            eq(
                {bat_e[t]: 1.0, bat_ch[t]: -BAT_EFF_CH * DT, bat_dis[t]: DT / BAT_EFF_DIS},
                float(data.asset["stationary_battery_initial_energy_kwh"]),
            )
        else:
            eq(
                {bat_e[t]: 1.0, bat_e[t - 1]: -1.0, bat_ch[t]: -BAT_EFF_CH * DT, bat_dis[t]: DT / BAT_EFF_DIS},
                0.0,
            )

    for n, row_ev in ev.iterrows():
        arr = row_ev.arrival_idx
        dep = row_ev.departure_idx
        for t in range(T):
            if t < arr:
                eq({ev_e[n, t]: 1.0}, float(row_ev.initial_energy_kwh))
            elif t == arr:
                eq(
                    {ev_e[n, t]: 1.0, ev_ch[n, t]: -EV_EFF_CH * DT, ev_dis[n, t]: DT / EV_EFF_DIS},
                    float(row_ev.initial_energy_kwh),
                )
            else:
                eq(
                    {ev_e[n, t]: 1.0, ev_e[n, t - 1]: -1.0, ev_ch[n, t]: -EV_EFF_CH * DT, ev_dis[n, t]: DT / EV_EFF_DIS},
                    0.0,
                )
        ub({ev_e[n, dep]: -1.0, ev_shortfall[n]: -1.0}, -float(row_ev.required_energy_at_departure_kwh))

    dates = pd.to_datetime(ts["timestamp"]).dt.date.to_numpy()
    for b, row_f in flex.iterrows():
        rebound = float(row_f.rebound_factor)
        for d in sorted(set(dates)):
            row = {}
            for t in np.where(dates == d)[0]:
                row[shift_up[b, t]] = row.get(shift_up[b, t], 0.0) + DT
                row[shift_down[b, t]] = row.get(shift_down[b, t], 0.0) - rebound * DT
            eq(row, 0.0)

    c[bat_ch] += 0.001 * DT
    c[bat_dis] += 0.001 * DT
    c[ev_ch] += 0.001 * DT
    c[ev_dis] += 0.001 * DT

    A_eq = lil_matrix((len(eq_rows), nvar), dtype=float)
    for r, row in enumerate(eq_rows):
        for k, v in row.items():
            A_eq[r, k] = A_eq[r, k] + v
    A_eq = A_eq.tocsr()

    A_ub = None
    if ub_rows:
        A_ub_lil = lil_matrix((len(ub_rows), nvar), dtype=float)
        for r, row in enumerate(ub_rows):
            for k, v in row.items():
                A_ub_lil[r, k] = A_ub_lil[r, k] + v
        A_ub = A_ub_lil.tocsr()

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=A_eq,
        b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(f"协同优化模型求解失败：{result.message}")

    x = result.x
    ev_ch_total = x[ev_ch].sum(axis=0)
    ev_dis_total = x[ev_dis].sum(axis=0)
    shift_down_total = x[shift_down].sum(axis=0)
    shift_up_total = x[shift_up].sum(axis=0)
    shed_total = x[shed].sum(axis=0)

    schedule = pd.DataFrame({
        "timestamp": ts["timestamp"],
        "native_load_kw": ts["total_native_load_kw"],
        "adjusted_load_kw": ts["total_native_load_kw"].to_numpy() - shift_down_total - shed_total + shift_up_total,
        "grid_buy_kw": x[grid_buy],
        "grid_sell_kw": x[grid_sell],
        "pv_available_kw": ts["pv_available_kw"],
        "pv_used_kw": x[pv_use],
        "pv_curtail_kw": x[pv_curtail],
        "battery_charge_kw": x[bat_ch],
        "battery_discharge_kw": x[bat_dis],
        "battery_energy_kwh": x[bat_e],
        "ev_charge_total_kw": ev_ch_total,
        "ev_discharge_total_kw": ev_dis_total,
        "ev_net_kw": ev_ch_total - ev_dis_total,
        "load_shift_down_kw": shift_down_total,
        "load_shift_up_kw": shift_up_total,
        "load_shed_kw": shed_total,
        "unmet_load_kw": x[unmet],
        "peak_import_var_kw": x[peak_import[0]],
    })
    for b, row_f in flex.iterrows():
        name = row_f.load_block
        schedule[f"{name}_shift_down_kw"] = x[shift_down[b]]
        schedule[f"{name}_shift_up_kw"] = x[shift_up[b]]
        schedule[f"{name}_shed_kw"] = x[shed[b]]

    ev_result = ev[["session_id", "arrival_time", "departure_time", "initial_energy_kwh", "required_energy_at_departure_kwh"]].copy()
    final_energy = []
    shortfalls = []
    for n, row_ev in ev.iterrows():
        final_energy.append(x[ev_e[n, row_ev.departure_idx]])
        shortfalls.append(x[ev_shortfall[n]])
    ev_result["final_energy_kwh"] = final_energy
    ev_result["shortfall_kwh"] = shortfalls
    ev_result["satisfied"] = ev_result["final_energy_kwh"] + ev_result["shortfall_kwh"] + 1e-6 >= ev_result["required_energy_at_departure_kwh"]

    return schedule, ev_result, {"objective": result.fun, "status": result.message}


def compute_metrics(schedule: pd.DataFrame, ev_result: pd.DataFrame, data: ProblemData) -> dict[str, float]:
    ts = data.ts.reset_index(drop=True)
    buy_cost = float((schedule["grid_buy_kw"] * ts["grid_buy_price_cny_per_kwh"] * DT).sum())
    sell_revenue = float((schedule["grid_sell_kw"] * ts["grid_sell_price_cny_per_kwh"] * DT).sum())
    shed_penalty = 0.0
    for _, row in data.flex.iterrows():
        col = f"{row.load_block}_shed_kw"
        if col in schedule:
            shed_penalty += float((schedule[col] * float(row.penalty_cny_per_kwh_not_served) * DT).sum())
    shift_penalty = float(((schedule["load_shift_down_kw"] + schedule["load_shift_up_kw"]) * SHIFT_PENALTY * DT).sum())
    curtail_penalty = float((schedule["pv_curtail_kw"] * CURTAIL_PENALTY * DT).sum())
    unmet_penalty = float((schedule.get("unmet_load_kw", 0.0) * BIG_PENALTY * DT).sum())
    peak_penalty = float(schedule["grid_buy_kw"].max() * PEAK_IMPORT_PENALTY)
    # total_cost 不计入 EV 缺口惩罚；缺口单独报告。因为 S102 的需求在物理上不可达，
    # 若计入高惩罚会掩盖运行费用本身的对比。
    pv_available = float((schedule["pv_available_kw"] * DT).sum())
    pv_used = float((schedule["pv_used_kw"] * DT).sum())
    pv_curtail = float((schedule["pv_curtail_kw"] * DT).sum())

    if "shortfall_kwh" in ev_result:
        ev_shortfall_series = ev_result["shortfall_kwh"]
        ev_satisfaction_rate = float((ev_shortfall_series <= 1e-5).mean())
    else:
        ev_shortfall_series = pd.Series(np.maximum(
            0.0,
            ev_result["required_energy_at_departure_kwh"] - ev_result["final_energy_kwh"],
        ))
        ev_satisfaction_rate = float(ev_result["satisfied"].mean())
    ev_shortfall_penalty = float(ev_shortfall_series.sum() * BIG_PENALTY)
    total_cost = buy_cost - sell_revenue + shed_penalty + shift_penalty + curtail_penalty + unmet_penalty + peak_penalty

    return {
        "total_cost_cny": total_cost,
        "grid_buy_cost_cny": buy_cost,
        "grid_sell_revenue_cny": sell_revenue,
        "shed_penalty_cny": shed_penalty,
        "shift_penalty_cny": shift_penalty,
        "curtail_penalty_cny": curtail_penalty,
        "unmet_penalty_cny": unmet_penalty,
        "peak_import_penalty_cny": peak_penalty,
        "unmet_load_kwh": float((schedule.get("unmet_load_kw", 0.0) * DT).sum()),
        "grid_import_energy_kwh": float((schedule["grid_buy_kw"] * DT).sum()),
        "grid_export_energy_kwh": float((schedule["grid_sell_kw"] * DT).sum()),
        "peak_grid_import_kw": float(schedule["grid_buy_kw"].max()),
        "pv_available_kwh": pv_available,
        "pv_used_kwh": pv_used,
        "pv_curtail_kwh": pv_curtail,
        "pv_consumption_rate": pv_used / pv_available if pv_available else np.nan,
        "pv_curtailment_rate": pv_curtail / pv_available if pv_available else np.nan,
        "battery_charge_kwh": float((schedule["battery_charge_kw"] * DT).sum()),
        "battery_discharge_kwh": float((schedule["battery_discharge_kw"] * DT).sum()),
        "ev_charge_kwh": float((schedule["ev_charge_total_kw"] * DT).sum()),
        "ev_discharge_kwh": float((schedule["ev_discharge_total_kw"] * DT).sum()),
        "ev_satisfaction_rate": ev_satisfaction_rate,
        "ev_shortfall_kwh": float(ev_shortfall_series.sum()),
        "ev_shortfall_penalty_cny_report_only": ev_shortfall_penalty,
        "load_shift_down_kwh": float((schedule["load_shift_down_kw"] * DT).sum()),
        "load_shift_up_kwh": float((schedule["load_shift_up_kw"] * DT).sum()),
        "load_shed_kwh": float((schedule["load_shed_kw"] * DT).sum()),
    }


def check_constraints(schedule: pd.DataFrame, ev_result: pd.DataFrame, data: ProblemData) -> list[str]:
    issues = []
    ts = data.ts.reset_index(drop=True)
    if (schedule["grid_buy_kw"] - ts["grid_import_limit_kw"] > 1e-5).any():
        issues.append("存在购电功率超过 grid_import_limit_kw。")
    if (schedule["grid_sell_kw"] - ts["grid_export_limit_kw"] > 1e-5).any():
        issues.append("存在售电功率超过 grid_export_limit_kw。")
    if (schedule["pv_used_kw"] + schedule["pv_curtail_kw"] - schedule["pv_available_kw"]).abs().max() > 1e-4:
        issues.append("光伏利用与弃光之和不等于可用光伏。")
    if schedule.get("unmet_load_kw", pd.Series([0.0])).max() > 1e-5:
        issues.append("存在未满足负荷松弛变量大于 0，说明原硬约束方案不可行或资源不足。")
    if ev_result.get("shortfall_kwh", pd.Series([0.0])).max() > 1e-5:
        issues.append("存在 EV 离站电量缺口松弛变量大于 0，说明至少一个会话在物理充电功率和停留时长下无法达到需求。")
    return issues


def plot_outputs(out_dir: Path, baseline: pd.DataFrame, coord: pd.DataFrame) -> None:
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


def write_outputs(out_dir: Path, baseline: pd.DataFrame, coord: pd.DataFrame,
                  baseline_ev: pd.DataFrame, coord_ev: pd.DataFrame,
                  baseline_metrics: dict, coord_metrics: dict,
                  solve_info: dict, data: ProblemData) -> None:
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
    lines = []
    lines.append("B题第一问运行结果摘要")
    lines.append("=" * 40)
    lines.append(f"求解状态：{solve_info.get('status')}")
    lines.append(f"协同优化目标值：{solve_info.get('objective'):.3f}")
    lines.append("")
    lines.append("核心指标对比：")
    for key in [
        "total_cost_cny", "grid_import_energy_kwh", "grid_export_energy_kwh",
        "peak_grid_import_kw", "pv_consumption_rate", "pv_curtailment_rate",
        "ev_satisfaction_rate", "load_shift_down_kwh", "load_shift_up_kwh", "load_shed_kwh",
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
    baseline, baseline_ev = simulate_baseline(data)
    coord, coord_ev, solve_info = build_and_solve_coordinated(data)
    baseline_metrics = compute_metrics(baseline, baseline_ev, data)
    coord_metrics = compute_metrics(coord, coord_ev, data)
    write_outputs(args.out_dir, baseline, coord, baseline_ev, coord_ev, baseline_metrics, coord_metrics, solve_info, data)
    print((args.out_dir / "problem1_summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
