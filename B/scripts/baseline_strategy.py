from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    BAT_EFF_CH,
    BAT_EFF_DIS,
    DT,
    EV_EFF_CH,
    HIGH_PRICE_THRESHOLD,
    ProblemData,
    add_ev_indices,
)


def simulate_baseline(data: ProblemData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """非协同运行方案。

    这个方案刻意不做全局优化，用一套容易解释的工程规则作为对照组：
    1. 建筑负荷不削减、不转移；
    2. EV 到站后立即按最大功率充电，直到达到离站需求；
    3. EV 不参与 V2B 反向供电；
    4. 固定储能只按简单规则运行：光伏富余时充电，高电价时放电。
    """

    ts = data.ts.reset_index(drop=True)
    ev = add_ev_indices(data)
    T = len(ts)
    N = len(ev)

    # 固定储能参数。
    bat_min = float(data.asset["stationary_battery_min_energy_kwh"])
    bat_max = float(data.asset["stationary_battery_max_energy_kwh"])
    bat_e = float(data.asset["stationary_battery_initial_energy_kwh"])
    bat_p_ch_max = float(data.asset["stationary_battery_max_charge_power_kw"])
    bat_p_dis_max = float(data.asset["stationary_battery_max_discharge_power_kw"])

    # -----------------------------
    # 1) EV 非协同策略：到站即充电。
    # -----------------------------
    ev_e = ev["initial_energy_kwh"].to_numpy(dtype=float).copy()
    ev_ch = np.zeros((N, T))
    ev_dis = np.zeros((N, T))  # baseline 中 EV 不允许反向放电，因此一直为 0。

    for n, row in ev.iterrows():
        need = row.required_energy_at_departure_kwh - ev_e[n]
        if need <= 0:
            continue

        # 从到站时段开始逐时段充电，直到达到离站需求或离站。
        for t in range(row.arrival_idx, row.departure_idx + 1):
            if need <= 1e-9:
                break
            room = row.battery_capacity_kwh - ev_e[n]
            p = min(row.max_charge_power_kw, need / (EV_EFF_CH * DT), room / (EV_EFF_CH * DT))
            p = max(0.0, p)
            ev_ch[n, t] = p
            gain = EV_EFF_CH * p * DT
            ev_e[n] += gain
            need -= gain

    # -----------------------------
    # 2) 固定储能和电网的规则调度。
    # -----------------------------
    bat_ch = np.zeros(T)
    bat_dis = np.zeros(T)
    bat_energy = np.zeros(T)
    grid_buy = np.zeros(T)
    grid_sell = np.zeros(T)
    pv_used = np.zeros(T)
    pv_curtail = np.zeros(T)

    for t in range(T):
        # baseline 中建筑负荷不参与调节，直接使用原生负荷。
        load = ts.loc[t, "total_native_load_kw"]
        ev_load = ev_ch[:, t].sum()
        demand = load + ev_load
        pv = ts.loc[t, "pv_available_kw"]
        buy_price = ts.loc[t, "grid_buy_price_cny_per_kwh"]

        # 光伏优先供给当前建筑和 EV 负荷。
        use_pv = min(pv, demand)
        pv_left = pv - use_pv
        demand_left = demand - use_pv

        # 若光伏还有剩余，优先给固定储能充电。
        if pv_left > 0:
            max_ch_by_energy = max(0.0, (bat_max - bat_e) / (BAT_EFF_CH * DT))
            p_ch = min(pv_left, bat_p_ch_max, max_ch_by_energy)
            bat_ch[t] = p_ch
            bat_e += BAT_EFF_CH * p_ch * DT
            pv_left -= p_ch

        # 若仍有负荷缺口，且当前电价较高，则固定储能放电削减高价购电。
        if demand_left > 0 and buy_price >= HIGH_PRICE_THRESHOLD:
            max_dis_by_energy = max(0.0, (bat_e - bat_min) * BAT_EFF_DIS / DT)
            p_dis = min(demand_left, bat_p_dis_max, max_dis_by_energy)
            bat_dis[t] = p_dis
            bat_e -= p_dis * DT / BAT_EFF_DIS
            demand_left -= p_dis

        # 余下需求从电网购买；若光伏仍有剩余，则售电，超过售电上限则弃光。
        pv_used[t] = use_pv + bat_ch[t]
        grid_buy[t] = min(demand_left, ts.loc[t, "grid_import_limit_kw"])
        unmet = max(0.0, demand_left - grid_buy[t])

        if pv_left > 0:
            grid_sell[t] = min(pv_left, ts.loc[t, "grid_export_limit_kw"])
            pv_left -= grid_sell[t]
        pv_curtail[t] = max(0.0, pv_left)
        bat_energy[t] = bat_e

        # baseline 是规则仿真，不允许因为购电上限导致程序崩溃；若超限，记录为额外购电。
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
