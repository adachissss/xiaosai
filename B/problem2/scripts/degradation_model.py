from __future__ import annotations

"""第二问第 1 小问：电池寿命损耗指标计算。

本脚本只负责把第一问已有调度结果换算成寿命损耗指标，暂不改变调度方案。
核心思想是用“充放电等效吞吐量”衡量电池使用强度，再乘以单位吞吐成本折算为寿命损耗成本。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# 附件数据是 15 分钟分辨率，因此功率 kW 转电量 kWh 时乘以 0.25 h。
DT = 0.25

# 固定储能寿命成本优先使用 asset_parameters.csv 中的给定值。
# 这里的默认值只作为兜底，主入口脚本会从数据表读取 0.055 元/kWh。
DEFAULT_STATIONARY_BATTERY_DEG_COST = 0.055


@dataclass(frozen=True)
class DegradationResult:
    """单个方案的寿命损耗评价结果。"""
    scheme: str
    battery_charge_kwh: float
    battery_discharge_kwh: float
    battery_throughput_kwh: float
    battery_degradation_cost_cny: float
    ev_charge_kwh: float
    ev_discharge_kwh: float
    ev_throughput_kwh: float
    ev_degradation_cost_cny: float

    @property
    def total_degradation_cost_cny(self) -> float:
        return self.battery_degradation_cost_cny + self.ev_degradation_cost_cny


def battery_degradation_cost(
    schedule: pd.DataFrame,
    unit_cost_cny_per_kwh: float = DEFAULT_STATIONARY_BATTERY_DEG_COST,
) -> tuple[float, float, float, float]:
    """计算固定储能寿命损耗。

    吞吐量按充电电量和放电电量相加计算：
    throughput = Σ(P_battery_charge + P_battery_discharge) × Δt。
    """

    charge_kwh = float((schedule["battery_charge_kw"] * DT).sum())
    discharge_kwh = float((schedule["battery_discharge_kw"] * DT).sum())
    throughput_kwh = charge_kwh + discharge_kwh
    cost_cny = throughput_kwh * unit_cost_cny_per_kwh
    return charge_kwh, discharge_kwh, throughput_kwh, cost_cny


def ev_degradation_cost(schedule: pd.DataFrame, ev_result: pd.DataFrame, data_dir: Path) -> tuple[float, float, float, float]:
    """计算 EV 车队寿命损耗。

    第一问结果文件只保留了 EV 聚合充放电功率，没有保留每辆车逐时段功率。
    因此第 1 小问先用全车队聚合吞吐量计算总损耗，再用会话结果估算一个加权平均 EV 单位寿命成本。
    后续第 2 小问直接在 LP 中保留车辆级变量时，可以精确写成 Σ_i c_i × throughput_i。
    """

    charge_kwh = float((schedule["ev_charge_total_kw"] * DT).sum())
    discharge_kwh = float((schedule["ev_discharge_total_kw"] * DT).sum())
    throughput_kwh = charge_kwh + discharge_kwh

    ev_sessions = pd.read_csv(data_dir / "ev_sessions.csv")
    cost_col = "degradation_cost_cny_per_kwh_throughput"
    if cost_col not in ev_sessions.columns:
        raise KeyError(f"ev_sessions.csv is missing {cost_col}")
    cost_map = ev_sessions.set_index("session_id")[cost_col]

    ev = ev_result.copy()
    # S102 这类物理不可达会话会有 shortfall。这里把 shortfall 加回去，表示车辆真实需求缺口，
    # 避免因为物理不可达导致估算的平均 EV 寿命成本偏低。
    ev["net_energy_gain_kwh"] = (ev["final_energy_kwh"] + ev.get("shortfall_kwh", 0.0)) - ev["initial_energy_kwh"]
    ev["estimated_throughput_kwh"] = np.maximum(0.0, ev["net_energy_gain_kwh"])
    ev["unit_degradation_cost_cny_per_kwh"] = ev["session_id"].map(cost_map)
    if ev["unit_degradation_cost_cny_per_kwh"].isna().any():
        raise ValueError("Some EV sessions cannot be matched to degradation cost parameters.")

    estimated_cost_cny = float((ev["estimated_throughput_kwh"] * ev["unit_degradation_cost_cny_per_kwh"]).sum())
    if ev["estimated_throughput_kwh"].sum() > 1e-9:
        avg_unit_cost = estimated_cost_cny / float(ev["estimated_throughput_kwh"].sum())
    else:
        avg_unit_cost = float(cost_map.mean())
    cost_cny = throughput_kwh * avg_unit_cost
    return charge_kwh, discharge_kwh, throughput_kwh, cost_cny


def evaluate_scheme(
    scheme: str,
    schedule: pd.DataFrame,
    ev_result: pd.DataFrame,
    data_dir: Path,
    battery_unit_cost: float = DEFAULT_STATIONARY_BATTERY_DEG_COST,
) -> DegradationResult:
    """对一个第一问方案计算固定储能、EV 和合计寿命损耗成本。"""

    bat_ch, bat_dis, bat_th, bat_cost = battery_degradation_cost(schedule, battery_unit_cost)
    ev_ch, ev_dis, ev_th, ev_cost = ev_degradation_cost(schedule, ev_result, data_dir)
    return DegradationResult(
        scheme=scheme,
        battery_charge_kwh=bat_ch,
        battery_discharge_kwh=bat_dis,
        battery_throughput_kwh=bat_th,
        battery_degradation_cost_cny=bat_cost,
        ev_charge_kwh=ev_ch,
        ev_discharge_kwh=ev_dis,
        ev_throughput_kwh=ev_th,
        ev_degradation_cost_cny=ev_cost,
    )


def result_to_dict(result: DegradationResult) -> dict[str, float | str]:
    return {
        "scheme": result.scheme,
        "battery_charge_kwh": result.battery_charge_kwh,
        "battery_discharge_kwh": result.battery_discharge_kwh,
        "battery_throughput_kwh": result.battery_throughput_kwh,
        "battery_degradation_cost_cny": result.battery_degradation_cost_cny,
        "ev_charge_kwh": result.ev_charge_kwh,
        "ev_discharge_kwh": result.ev_discharge_kwh,
        "ev_throughput_kwh": result.ev_throughput_kwh,
        "ev_degradation_cost_cny": result.ev_degradation_cost_cny,
        "total_degradation_cost_cny": result.total_degradation_cost_cny,
    }


def load_problem1_scheme(problem1_results_dir: Path, scheme: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = pd.read_csv(problem1_results_dir / f"{scheme}_schedule.csv", parse_dates=["timestamp"])
    ev_result = pd.read_csv(problem1_results_dir / f"{scheme}_ev_results.csv")
    return schedule, ev_result


def load_stationary_battery_degradation_cost(data_dir: Path) -> float:
    """从 asset_parameters.csv 读取固定储能单位吞吐寿命成本。"""

    asset = pd.read_csv(data_dir / "asset_parameters.csv")
    key = "stationary_battery_degradation_cost_cny_per_kwh_throughput"
    matched = asset.loc[asset["parameter"] == key, "value"]
    if matched.empty:
        return DEFAULT_STATIONARY_BATTERY_DEG_COST
    return float(matched.iloc[0])


def load_ev_degradation_parameter_summary(data_dir: Path) -> pd.DataFrame:
    ev = pd.read_csv(data_dir / "ev_sessions.csv")
    cost_col = "degradation_cost_cny_per_kwh_throughput"
    if cost_col not in ev.columns:
        raise KeyError(f"ev_sessions.csv is missing {cost_col}")
    summary = ev.groupby("ev_type")[cost_col].agg(["count", "min", "mean", "max"]).reset_index()
    summary = summary.rename(columns={
        "count": "session_count",
        "min": "min_cost_cny_per_kwh",
        "mean": "mean_cost_cny_per_kwh",
        "max": "max_cost_cny_per_kwh",
    })
    return summary
