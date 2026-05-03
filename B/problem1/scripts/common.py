from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# 统一时间步长：附件数据是 15 分钟分辨率，即 0.25 小时。
DT = 0.25

# 固定储能和 EV 的充放电效率。题目未给 EV 效率，因此沿用 0.95 的常数效率假设。
BAT_EFF_CH = 0.95
BAT_EFF_DIS = 0.95
EV_EFF_CH = 0.95
EV_EFF_DIS = 0.95

# 策略参数：这些不是数据给定值，而是模型偏好权重。
SHIFT_PENALTY = 0.05          # 负荷转移舒适度惩罚，元/kWh
CURTAIL_PENALTY = 0.01        # 弃光惩罚，元/kWh
HIGH_PRICE_THRESHOLD = 0.90   # baseline 中固定储能放电的高电价阈值
BIG_PENALTY = 1_000.0         # 未满足负荷或 EV 缺口的高惩罚
PEAK_IMPORT_PENALTY = 8.0     # 协同方案中峰值购电惩罚，元/kW


@dataclass
class ProblemData:
    """第一问所有输入数据的统一容器。"""

    ts: pd.DataFrame
    ev: pd.DataFrame
    asset: dict
    flex: pd.DataFrame
    time_index: pd.DatetimeIndex


def read_data(data_dir: Path) -> ProblemData:
    """读取附件数据，并做基础时间格式转换与 session_id 唯一性检查。"""

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


def add_ev_indices(data: ProblemData) -> pd.DataFrame:
    """把 EV 到离站时间映射成 0..671 的离散时段索引。"""

    ev = data.ev.copy()
    mapping = {t: i for i, t in enumerate(data.time_index)}
    ev["arrival_idx"] = ev["arrival_time"].map(mapping)
    ev["departure_idx"] = ev["departure_time"].map(mapping)
    if ev[["arrival_idx", "departure_idx"]].isna().any().any():
        raise ValueError("部分 EV 到离站时间无法映射到 15 分钟时间索引。")
    ev["arrival_idx"] = ev["arrival_idx"].astype(int)
    ev["departure_idx"] = ev["departure_idx"].astype(int)
    ev["ev_index"] = np.arange(len(ev))
    return ev


def compute_metrics(schedule: pd.DataFrame, ev_result: pd.DataFrame, data: ProblemData) -> dict[str, float]:
    """计算 baseline 与 coordinated 共用的评价指标。"""

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

    pv_available = float((schedule["pv_available_kw"] * DT).sum())
    pv_used = float((schedule["pv_used_kw"] * DT).sum())
    pv_curtail = float((schedule["pv_curtail_kw"] * DT).sum())

    if "shortfall_kwh" in ev_result:
        ev_shortfall_series = ev_result["shortfall_kwh"]
        ev_satisfaction_rate = float((ev_shortfall_series <= 1e-5).mean())
    else:
        ev_shortfall_series = pd.Series(np.maximum(0.0, ev_result["required_energy_at_departure_kwh"] - ev_result["final_energy_kwh"]))
        ev_satisfaction_rate = float(ev_result["satisfied"].mean())

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
        "ev_shortfall_penalty_cny_report_only": float(ev_shortfall_series.sum() * BIG_PENALTY),
        "load_shift_down_kwh": float((schedule["load_shift_down_kw"] * DT).sum()),
        "load_shift_up_kwh": float((schedule["load_shift_up_kw"] * DT).sum()),
        "load_shed_kwh": float((schedule["load_shed_kw"] * DT).sum()),
    }


def check_constraints(schedule: pd.DataFrame, ev_result: pd.DataFrame, data: ProblemData) -> list[str]:
    """检查结果中最关键的约束是否被违反。"""

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
