 园区微电网—电动车—建筑协同调度


 1. 数据时间范围与分辨率
- 时间范围：2025-07-14 00:00 至 2025-07-20 23:45
- 分辨率：15 分钟
- 典型场景：夏季科研园区（办公楼 + 湿实验楼 + 教学中心）

 2. 文件说明
 (1) timeseries_15min.csv
主时间序列文件，每 15 分钟一行。
字段：
- timestamp：时间戳
- ambient_temp_c：环境温度（°C）
- solar_irradiance_wm2：水平面太阳辐照（W/m²）
- office_building_kw：办公楼原生负荷（kW）
- wet_lab_kw：湿实验楼原生负荷（kW）
- teaching_center_kw：教学中心原生负荷（kW）
- total_native_load_kw：三类建筑原生负荷合计（kW）
- pv_available_kw：该时段可利用 PV 出力上限（kW）
- grid_buy_price_cny_per_kwh：电网购电价（元/kWh）
- grid_sell_price_cny_per_kwh：上网电价（元/kWh）
- grid_carbon_kg_per_kwh：电网边际碳强度（kgCO2/kWh）
- grid_import_limit_kw：允许从大电网购入的最大功率（kW）
- grid_export_limit_kw：允许向大电网反送的最大功率（kW）

 (2) ev_sessions.csv
电动车充放电会话表，每行表示一辆车在园区的一次停放/充电会话。
字段：
- session_id：会话编号
- ev_type：车型（compact / sedan / SUV）
- arrival_time：到站时间
- departure_time：离站时间
- battery_capacity_kwh：电池容量（kWh）
- initial_energy_kwh：到站时电量（kWh）
- required_energy_at_departure_kwh：离站时最低所需电量（kWh）
- max_charge_power_kw：最大充电功率（kW）
- max_discharge_power_kw：最大反向放电功率（kW）；若为 0 则不允许 V2B
- v2b_allowed：是否允许反向放电（1/0）
- degradation_cost_cny_per_kwh_throughput：EV 电池等效吞吐寿命成本（元/kWh）

 (3) asset_parameters.csv
固定资产与站端参数：
- 固定储能容量、功率、效率、初始能量
- PV 装机与逆变器限制
- 充电桩数量与最大可同时接入车辆数
- 时间步长

 (4) flexible_load_parameters.csv
各类建筑负荷的柔性参数：
- noninterruptible_share：不可中断份额
- max_shiftable_kw：可转移功率上限
- max_sheddable_kw：可削减功率上限
- rebound_factor：负荷转移后的反弹系数
- penalty_cny_per_kwh_not_served：未满足或不舒适的惩罚系数

 (5) daily_summary.csv / ev_summary_stats.csv
可用于快速检查数据整体量级与场景合理性。

 (6) scenario_notes.csv
场景背景与特殊情况说明。

 3. 可参考的相似数据源
- HEEW: multi-source campus energy/PV/weather/emissions dataset
- ACN-Data: workplace EV charging sessions
- NREL EULP: calibrated 15-minute building load profiles
- NSRDB: irradiance and meteorological variables for solar applications
