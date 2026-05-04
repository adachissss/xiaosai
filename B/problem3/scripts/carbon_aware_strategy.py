from __future__ import annotations

"""第三问：cap-and-trade 碳交易机制的协同优化模型（S5）。

在 S4 模型（运行成本 + 电池寿命损耗成本）基础上增加碳交易成本。

碳交易机制（cap-and-trade）：
- 园区获得免费碳排放配额 free_allowance_kg
- 实际碳排放超过配额时，须以 carbon_price 购买差额
- 实际碳排放低于配额时，可以 carbon_price 卖出剩余配额获利
- 碳交易成本 = carbon_price × (total_emissions - free_allowance)，可为负（收入）

在 LP 目标函数中：c[grid_buy[t]] += carbon_price × carbon_intensity[t] × DT
（free_allowance 是常量，不影响优化，只影响事后成本核算）
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

PROBLEM1_SCRIPTS = Path(__file__).resolve().parents[2] / "problem1" / "scripts"
if str(PROBLEM1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PROBLEM1_SCRIPTS))

from common import (  # noqa: E402
    BAT_EFF_CH,
    BAT_EFF_DIS,
    BIG_PENALTY,
    CURTAIL_PENALTY,
    DT,
    EV_EFF_CH,
    EV_EFF_DIS,
    PEAK_IMPORT_PENALTY,
    SHIFT_PENALTY,
    ProblemData,
    add_ev_indices,
)


class VarIndex:
    """把不同类型的决策变量统一映射到 linprog 所需的一维向量。"""

    def __init__(self):
        self.size = 0

    def add(self, shape: tuple[int, ...]) -> np.ndarray:
        count = int(np.prod(shape))
        arr = np.arange(self.size, self.size + count).reshape(shape)
        self.size += count
        return arr


def build_and_solve_carbon_aware(
    data: ProblemData,
    *,
    carbon_price_cny_per_kg: float = 0.0,
    free_allowance_kg: float | None = None,
    scheme_name: str = "S5_carbon_aware",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """求解 S5：cap-and-trade 碳交易机制下的完整协同方案。

    参数：
        carbon_price_cny_per_kg：碳交易价格（元/kg CO₂）。
            100 元/吨 = 0.10 元/kg。
        free_allowance_kg：免费碳排放配额（kg CO₂）。
            为 None 时表示无免费配额（所有排放均需购买）。
            不影响 LP 优化，仅用于事后成本核算。
    """

    ts = data.ts.reset_index(drop=True)
    ev = add_ev_indices(data)
    flex = data.flex.reset_index(drop=True)
    T = len(ts)
    N = len(ev)
    B = len(flex)

    idx = VarIndex()
    grid_buy = idx.add((T,))
    grid_sell = idx.add((T,))
    pv_use = idx.add((T,))
    pv_curtail = idx.add((T,))
    bat_ch = idx.add((T,))
    bat_dis = idx.add((T,))
    bat_e = idx.add((T,))
    ev_ch = idx.add((N, T))
    ev_dis = idx.add((N, T))
    ev_e = idx.add((N, T))
    shift_down = idx.add((B, T))
    shift_up = idx.add((B, T))
    shed = idx.add((B, T))
    unmet = idx.add((T,))
    ev_shortfall = idx.add((N,))
    peak_import = idx.add((1,))

    nvar = idx.size
    c = np.zeros(nvar)

    # --- 运行成本项（与 S4 一致） ---
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

    # --- 固定储能和 EV 寿命损耗成本（与 S4 一致） ---
    battery_deg_cost = float(data.asset["stationary_battery_degradation_cost_cny_per_kwh_throughput"])
    c[bat_ch] += battery_deg_cost * DT
    c[bat_dis] += battery_deg_cost * DT

    ev_cost_col = "degradation_cost_cny_per_kwh_throughput"
    ev_unit_cost = ev[ev_cost_col].astype(float).to_numpy()
    for n, unit_cost in enumerate(ev_unit_cost):
        c[ev_ch[n, :]] += unit_cost * DT
        c[ev_dis[n, :]] += unit_cost * DT

    # --- 碳排放交易成本（加入目标函数） ---
    carbon_intensity = ts["grid_carbon_kg_per_kwh"].to_numpy()
    c[grid_buy] += carbon_price_cny_per_kg * carbon_intensity * DT

    # --- 变量上下界（与 S4 一致） ---
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
                if int(row.v2b_allowed) == 1:
                    dis_max = float(row.max_discharge_power_kw)
                else:
                    dis_max = 0.0
                bounds[ev_dis[n, t]] = (0.0, dis_max)
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

    # --- 约束构建 ---
    eq_rows: list[dict[int, float]] = []
    b_eq = []
    ub_rows: list[dict[int, float]] = []
    b_ub = []

    def eq(row: dict[int, float], rhs: float) -> None:
        eq_rows.append(row)
        b_eq.append(rhs)

    def ub(row: dict[int, float], rhs: float) -> None:
        ub_rows.append(row)
        b_ub.append(rhs)

    # 削峰、光伏平衡、功率平衡、固定储能 SOC
    for t in range(T):
        ub({grid_buy[t]: 1.0, peak_import[0]: -1.0}, 0.0)
        eq({pv_use[t]: 1.0, pv_curtail[t]: 1.0}, float(ts.loc[t, "pv_available_kw"]))

        row = {pv_use[t]: 1.0, grid_buy[t]: 1.0, bat_dis[t]: 1.0, unmet[t]: 1.0, grid_sell[t]: -1.0, bat_ch[t]: -1.0}
        for n in range(N):
            row[ev_dis[n, t]] = row.get(ev_dis[n, t], 0.0) + 1.0
            row[ev_ch[n, t]] = row.get(ev_ch[n, t], 0.0) - 1.0
        for b in range(B):
            row[shift_down[b, t]] = row.get(shift_down[b, t], 0.0) + 1.0
            row[shed[b, t]] = row.get(shed[b, t], 0.0) + 1.0
            row[shift_up[b, t]] = row.get(shift_up[b, t], 0.0) - 1.0
        eq(row, float(ts.loc[t, "total_native_load_kw"]))

        if t == 0:
            eq({bat_e[t]: 1.0, bat_ch[t]: -BAT_EFF_CH * DT, bat_dis[t]: DT / BAT_EFF_DIS}, float(data.asset["stationary_battery_initial_energy_kwh"]))
        else:
            eq({bat_e[t]: 1.0, bat_e[t - 1]: -1.0, bat_ch[t]: -BAT_EFF_CH * DT, bat_dis[t]: DT / BAT_EFF_DIS}, 0.0)

    # EV 电量递推和离站需求
    for n, row_ev in ev.iterrows():
        arr = row_ev.arrival_idx
        dep = row_ev.departure_idx
        for t in range(T):
            if t < arr:
                eq({ev_e[n, t]: 1.0}, float(row_ev.initial_energy_kwh))
            elif t == arr:
                eq({ev_e[n, t]: 1.0, ev_ch[n, t]: -EV_EFF_CH * DT, ev_dis[n, t]: DT / EV_EFF_DIS}, float(row_ev.initial_energy_kwh))
            else:
                eq({ev_e[n, t]: 1.0, ev_e[n, t - 1]: -1.0, ev_ch[n, t]: -EV_EFF_CH * DT, ev_dis[n, t]: DT / EV_EFF_DIS}, 0.0)
        ub({ev_e[n, dep]: -1.0, ev_shortfall[n]: -1.0}, -float(row_ev.required_energy_at_departure_kwh))

    # 建筑可转移负荷：每天内部能量守恒
    dates = pd.to_datetime(ts["timestamp"]).dt.date.to_numpy()
    for b, row_f in flex.iterrows():
        rebound = float(row_f.rebound_factor)
        for d in sorted(set(dates)):
            row = {}
            for t in np.where(dates == d)[0]:
                row[shift_up[b, t]] = row.get(shift_up[b, t], 0.0) + DT
                row[shift_down[b, t]] = row.get(shift_down[b, t], 0.0) - rebound * DT
            eq(row, 0.0)

    # --- 求解 ---
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

    result = linprog(c, A_ub=A_ub, b_ub=np.array(b_ub) if b_ub else None, A_eq=A_eq, b_eq=np.array(b_eq), bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"考虑碳排放的协同优化模型求解失败：{result.message}")

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
    ev_result["final_energy_kwh"] = [x[ev_e[n, row_ev.departure_idx]] for n, row_ev in ev.iterrows()]
    ev_result["shortfall_kwh"] = [x[ev_shortfall[n]] for n in range(N)]
    ev_result["satisfied"] = ev_result["final_energy_kwh"] + ev_result["shortfall_kwh"] + 1e-6 >= ev_result["required_energy_at_departure_kwh"]
    ev_result["charge_throughput_kwh"] = x[ev_ch].sum(axis=1) * DT
    ev_result["discharge_throughput_kwh"] = x[ev_dis].sum(axis=1) * DT
    ev_result["total_throughput_kwh"] = ev_result["charge_throughput_kwh"] + ev_result["discharge_throughput_kwh"]
    ev_result["unit_degradation_cost_cny_per_kwh"] = ev_unit_cost
    ev_result["degradation_cost_cny"] = ev_result["total_throughput_kwh"] * ev_result["unit_degradation_cost_cny_per_kwh"]

    total_carbon_kg = float(np.dot(x[grid_buy], carbon_intensity * DT))
    # cap-and-trade: 碳交易成本 = 碳价 × (实际排放 - 免费配额)，可为负（卖配额收入）
    if free_allowance_kg is not None:
        carbon_cost_cny = carbon_price_cny_per_kg * (total_carbon_kg - free_allowance_kg)
    else:
        carbon_cost_cny = carbon_price_cny_per_kg * total_carbon_kg

    return schedule, ev_result, {
        "objective": result.fun,
        "status": result.message,
        "scheme": scheme_name,
        "carbon_price_cny_per_kg": carbon_price_cny_per_kg,
        "free_allowance_kg": free_allowance_kg,
        "total_carbon_kg": total_carbon_kg,
        "carbon_trading_cost_cny": carbon_cost_cny,
    }
