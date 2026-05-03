from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DT = 0.25
DEFAULT_STATIONARY_BATTERY_DEG_COST = 0.05


@dataclass(frozen=True)
class DegradationResult:
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
    charge_kwh = float((schedule["battery_charge_kw"] * DT).sum())
    discharge_kwh = float((schedule["battery_discharge_kw"] * DT).sum())
    throughput_kwh = charge_kwh + discharge_kwh
    cost_cny = throughput_kwh * unit_cost_cny_per_kwh
    return charge_kwh, discharge_kwh, throughput_kwh, cost_cny


def ev_degradation_cost(schedule: pd.DataFrame, ev_result: pd.DataFrame, data_dir: Path) -> tuple[float, float, float, float]:
    charge_kwh = float((schedule["ev_charge_total_kw"] * DT).sum())
    discharge_kwh = float((schedule["ev_discharge_total_kw"] * DT).sum())
    throughput_kwh = charge_kwh + discharge_kwh

    ev_sessions = pd.read_csv(data_dir / "ev_sessions.csv")
    cost_col = "degradation_cost_cny_per_kwh_throughput"
    if cost_col not in ev_sessions.columns:
        raise KeyError(f"ev_sessions.csv is missing {cost_col}")
    cost_map = ev_sessions.set_index("session_id")[cost_col]

    ev = ev_result.copy()
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
