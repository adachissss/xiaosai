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
from coordinated_strategy import solve_lp_strategy


def _immediate_ev_charging(data: ProblemData) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """EV 即插即充规则。

    该函数供 S0/S1 共用：车辆到站后立即以最大功率充电，直到达到离站需求；
    EV 不参与反向供电。返回 EV 充电矩阵、放电矩阵和每辆车离站结果。
    """

    ev = add_ev_indices(data)
    T = len(data.ts)
    N = len(ev)
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
            p = max(0.0, p)
            ev_ch[n, t] = p
            gain = EV_EFF_CH * p * DT
            ev_e[n] += gain
            need -= gain

    ev_result = ev[["session_id", "arrival_time", "departure_time", "initial_energy_kwh", "required_energy_at_departure_kwh"]].copy()
    ev_result["final_energy_kwh"] = ev_e
    ev_result["satisfied"] = ev_result["final_energy_kwh"] + 1e-6 >= ev_result["required_energy_at_departure_kwh"]
    return ev_ch, ev_dis, ev_result


def _simulate_rule_scheme(data: ProblemData, *, use_rule_storage: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S0/S1 的规则仿真骨架。

    use_rule_storage=False：S0 朴素方案，固定储能不运行；
    use_rule_storage=True：S1 规则储能方案，光伏富余时充电、高电价时放电。
    """

    ts = data.ts.reset_index(drop=True)
    T = len(ts)

    bat_min = float(data.asset["stationary_battery_min_energy_kwh"])
    bat_max = float(data.asset["stationary_battery_max_energy_kwh"])
    bat_e = float(data.asset["stationary_battery_initial_energy_kwh"])
    bat_p_ch_max = float(data.asset["stationary_battery_max_charge_power_kw"])
    bat_p_dis_max = float(data.asset["stationary_battery_max_discharge_power_kw"])

    ev_ch, ev_dis, ev_result = _immediate_ev_charging(data)

    bat_ch = np.zeros(T)
    bat_dis = np.zeros(T)
    bat_energy = np.zeros(T)
    grid_buy = np.zeros(T)
    grid_sell = np.zeros(T)
    pv_used = np.zeros(T)
    pv_curtail = np.zeros(T)

    for t in range(T):
        # S0/S1 中建筑负荷均不参与调节。
        load = ts.loc[t, "total_native_load_kw"]
        ev_load = ev_ch[:, t].sum()
        demand = load + ev_load
        pv = ts.loc[t, "pv_available_kw"]
        buy_price = ts.loc[t, "grid_buy_price_cny_per_kwh"]

        # 光伏优先服务当前负荷。
        use_pv = min(pv, demand)
        pv_left = pv - use_pv
        demand_left = demand - use_pv

        if use_rule_storage:
            # S1：光伏富余时给固定储能充电。
            if pv_left > 0:
                max_ch_by_energy = max(0.0, (bat_max - bat_e) / (BAT_EFF_CH * DT))
                p_ch = min(pv_left, bat_p_ch_max, max_ch_by_energy)
                bat_ch[t] = p_ch
                bat_e += BAT_EFF_CH * p_ch * DT
                pv_left -= p_ch

            # S1：高电价时固定储能放电减少购电。
            if demand_left > 0 and buy_price >= HIGH_PRICE_THRESHOLD:
                max_dis_by_energy = max(0.0, (bat_e - bat_min) * BAT_EFF_DIS / DT)
                p_dis = min(demand_left, bat_p_dis_max, max_dis_by_energy)
                bat_dis[t] = p_dis
                bat_e -= p_dis * DT / BAT_EFF_DIS
                demand_left -= p_dis

        # 剩余需求购电，富余光伏售电，售不出去则弃光。
        pv_used[t] = use_pv + bat_ch[t]
        grid_buy[t] = min(demand_left, ts.loc[t, "grid_import_limit_kw"])
        unmet = max(0.0, demand_left - grid_buy[t])

        if pv_left > 0:
            grid_sell[t] = min(pv_left, ts.loc[t, "grid_export_limit_kw"])
            pv_left -= grid_sell[t]
        pv_curtail[t] = max(0.0, pv_left)
        bat_energy[t] = bat_e

        # 对规则方案，若购电上限不足，记录为额外购电，避免仿真中断。
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
    return schedule, ev_result


def simulate_baseline_no_storage(data: ProblemData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S0 朴素方案：储能不运行，EV 即插即充，建筑负荷不调节。"""

    return _simulate_rule_scheme(data, use_rule_storage=False)


def simulate_baseline_rule_storage(data: ProblemData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S1 规则储能方案：当前已有 baseline，储能按固定规则运行。"""

    return _simulate_rule_scheme(data, use_rule_storage=True)


def solve_partial_coordination(data: ProblemData) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """S2 部分协同方案。

    含义：固定储能和 EV 充电时机由线性规划优化；
    限制：EV 不允许 V2B 放电，建筑负荷不允许转移或削减。
    用途：衡量“只做储能优化 + EV 智能充电”的收益。
    """

    return solve_lp_strategy(
        data,
        allow_ev_discharge=False,
        allow_flexible_load=False,
        scheme_name="S2_partial_coordination",
    )


# 保留旧函数名，避免外部调用失效。现在它等价于 S1。
def simulate_baseline(data: ProblemData) -> tuple[pd.DataFrame, pd.DataFrame]:
    return simulate_baseline_rule_storage(data)
