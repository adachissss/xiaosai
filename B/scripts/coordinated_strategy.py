from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from common import (
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


def build_and_solve_coordinated(data: ProblemData) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """协同优化方案。

    这个函数建立车辆级 EV + 固定储能 + 柔性建筑负荷 + 光伏 + 电网的线性规划模型。
    与 baseline 不同，这里允许模型在全周范围内统一决定：
    - 什么时候买电、售电；
    - 固定储能什么时候充放电；
    - EV 什么时候充电，允许 V2B 的车辆什么时候放电；
    - 建筑负荷什么时候转移或削减。
    """

    ts = data.ts.reset_index(drop=True)
    ev = add_ev_indices(data)
    flex = data.flex.reset_index(drop=True)
    T = len(ts)
    N = len(ev)
    B = len(flex)

    # -----------------------------
    # 1) 定义所有决策变量。
    # -----------------------------
    idx = VarIndex()
    grid_buy = idx.add((T,))       # 电网购电功率
    grid_sell = idx.add((T,))      # 电网售电功率
    pv_use = idx.add((T,))         # 实际利用光伏功率
    pv_curtail = idx.add((T,))     # 弃光功率
    bat_ch = idx.add((T,))         # 固定储能充电功率
    bat_dis = idx.add((T,))        # 固定储能放电功率
    bat_e = idx.add((T,))          # 固定储能电量
    ev_ch = idx.add((N, T))        # 每辆 EV 的充电功率
    ev_dis = idx.add((N, T))       # 每辆 EV 的放电功率
    ev_e = idx.add((N, T))         # 每辆 EV 的电量
    shift_down = idx.add((B, T))   # 建筑负荷从该时段移出
    shift_up = idx.add((B, T))     # 建筑负荷移入该时段
    shed = idx.add((B, T))         # 建筑负荷削减
    unmet = idx.add((T,))          # 未满足负荷松弛变量
    ev_shortfall = idx.add((N,))   # EV 离站电量缺口松弛变量
    peak_import = idx.add((1,))    # 一周最大购电功率

    nvar = idx.size
    c = np.zeros(nvar)

    # -----------------------------
    # 2) 目标函数：运行成本 + 舒适度惩罚 + 削峰惩罚。
    # -----------------------------
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

    # -----------------------------
    # 3) 变量上下界。
    # -----------------------------
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

    # 先用字典暂存稀疏约束，最后统一转成 scipy sparse matrix。
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

    # -----------------------------
    # 4) 每个时段的功率平衡、光伏平衡和固定储能状态转移。
    # -----------------------------
    for t in range(T):
        # 削峰约束：所有时段购电功率都不能超过 peak_import。
        ub({grid_buy[t]: 1.0, peak_import[0]: -1.0}, 0.0)

        # 光伏去向：利用 + 弃光 = 可用光伏。
        eq({pv_use[t]: 1.0, pv_curtail[t]: 1.0}, float(ts.loc[t, "pv_available_kw"]))

        # 园区功率平衡：供给侧 = 需求侧。
        row = {pv_use[t]: 1.0, grid_buy[t]: 1.0, bat_dis[t]: 1.0, unmet[t]: 1.0, grid_sell[t]: -1.0, bat_ch[t]: -1.0}
        for n in range(N):
            row[ev_dis[n, t]] = row.get(ev_dis[n, t], 0.0) + 1.0
            row[ev_ch[n, t]] = row.get(ev_ch[n, t], 0.0) - 1.0
        for b in range(B):
            row[shift_down[b, t]] = row.get(shift_down[b, t], 0.0) + 1.0
            row[shed[b, t]] = row.get(shed[b, t], 0.0) + 1.0
            row[shift_up[b, t]] = row.get(shift_up[b, t], 0.0) - 1.0
        eq(row, float(ts.loc[t, "total_native_load_kw"]))

        # 固定储能 SOC 递推。
        if t == 0:
            eq({bat_e[t]: 1.0, bat_ch[t]: -BAT_EFF_CH * DT, bat_dis[t]: DT / BAT_EFF_DIS}, float(data.asset["stationary_battery_initial_energy_kwh"]))
        else:
            eq({bat_e[t]: 1.0, bat_e[t - 1]: -1.0, bat_ch[t]: -BAT_EFF_CH * DT, bat_dis[t]: DT / BAT_EFF_DIS}, 0.0)

    # -----------------------------
    # 5) 每辆 EV 的电量递推和离站需求。
    # -----------------------------
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

        # S102 这类物理不可达会话会通过 ev_shortfall 记录缺口，避免整个模型不可行。
        ub({ev_e[n, dep]: -1.0, ev_shortfall[n]: -1.0}, -float(row_ev.required_energy_at_departure_kwh))

    # -----------------------------
    # 6) 建筑可转移负荷：每天内部能量守恒，并考虑反弹系数。
    # -----------------------------
    dates = pd.to_datetime(ts["timestamp"]).dt.date.to_numpy()
    for b, row_f in flex.iterrows():
        rebound = float(row_f.rebound_factor)
        for d in sorted(set(dates)):
            row = {}
            for t in np.where(dates == d)[0]:
                row[shift_up[b, t]] = row.get(shift_up[b, t], 0.0) + DT
                row[shift_down[b, t]] = row.get(shift_down[b, t], 0.0) - rebound * DT
            eq(row, 0.0)

    # 极小吞吐惩罚，用于减少线性规划退化导致的无意义同时充放电。
    c[bat_ch] += 0.001 * DT
    c[bat_dis] += 0.001 * DT
    c[ev_ch] += 0.001 * DT
    c[ev_dis] += 0.001 * DT

    # -----------------------------
    # 7) 构建稀疏矩阵并调用 HiGHS 求解。
    # -----------------------------
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
        raise RuntimeError(f"协同优化模型求解失败：{result.message}")

    # -----------------------------
    # 8) 把一维求解结果还原为可读的调度表。
    # -----------------------------
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

    return schedule, ev_result, {"objective": result.fun, "status": result.message}
