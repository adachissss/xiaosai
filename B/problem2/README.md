# B 题第二问目录说明

本目录用于存放第二问的代码、结果和图片，避免与第一问混淆。

## 当前进度

已完成第二问第 1 小问：提出固定储能与 EV 电池寿命损耗指标或成本折算方式。

已完成第二问第 2 小问：在 S3 完整协同模型基础上加入寿命损耗成本，形成 `S4_degradation_aware` 调度方案，并输出 S3/S4 对比结果。

已完成第二问第 3 小问：基于 S3/S4 对比结果，讨论引入寿命损耗前后系统调度行为的主要变化。

已完成第二问第 4 小问：讨论固定储能与 EV 在削峰、储能和供能中的角色分工。

## 目录结构

```text
problem2/
├── scripts/
│   ├── degradation_model.py              # 寿命损耗指标和成本计算函数
│   ├── degradation_aware_strategy.py     # 第二问第2小问 S4 寿命感知优化模型
│   ├── solve_degradation_indicator.py    # 第二问第1小问入口脚本
│   └── solve_problem2.py                 # 第二问第2小问入口脚本
├── 第二问第1小问解决思路.md
├── 第二问第2小问解决思路.md
├── 第二问第3小问解决思路.md
├── 第二问第4小问解决思路.md
└── results/
    ├── p2_q1_degradation_metrics.csv     # 第一问四方案的寿命损耗指标试算
    ├── p2_q1_ev_degradation_parameters.csv
    ├── p2_q1_summary.txt
    ├── S3_reference_schedule.csv
    ├── S3_reference_ev_results.csv
    ├── S4_degradation_aware_schedule.csv
    ├── S4_degradation_aware_ev_results.csv
    ├── p2_q2_comparison_metrics.csv
    ├── p2_q2_summary.txt
    └── figures/
        ├── p2_q1_throughput_indicator.png
        ├── p2_q1_degradation_cost.png
        ├── p2_q2_grid_import_s3_vs_s4.png
        ├── p2_q2_battery_power_s3_vs_s4.png
        ├── p2_q2_battery_soc_s3_vs_s4.png
        ├── p2_q2_ev_v2b_s3_vs_s4.png
        ├── p2_q2_throughput_comparison.png
        └── p2_q2_cost_breakdown.png
```

## 运行方式

在项目根目录运行第 1 小问：

```bash
conda run -n xiaosai-b python B/problem2/scripts/solve_degradation_indicator.py
```

运行第 2 小问：

```bash
conda run -n xiaosai-b python B/problem2/scripts/solve_problem2.py
```

## 当前采用的寿命损耗模型

采用线性等效吞吐量模型：

```text
E_throughput = Σ(P_charge + P_discharge) × Δt
C_degradation = c_deg × E_throughput
```

固定储能单位吞吐寿命成本来自：

```text
B/B_data/asset_parameters.csv
```

字段：

```text
stationary_battery_degradation_cost_cny_per_kwh_throughput = 0.055 元/kWh
```

EV 单位吞吐寿命成本来自：

```text
B/B_data/ev_sessions.csv
```

字段：

```text
degradation_cost_cny_per_kwh_throughput
```

## 第 2 小问 S4 模型

第 2 小问在第一问 `S3_full_coordination` 的基础上建立 `S4_degradation_aware`。

约束保持 S3 的完整协同约束不变，目标函数加入寿命损耗成本：

```text
min 运行成本 + C_bat_deg + C_ev_deg
```

其中：

```text
C_bat_deg = c_bat × Σ_t(P_bat_ch[t] + P_bat_dis[t]) × Δt
C_ev_deg = Σ_i c_ev[i] × Σ_t(P_ev_ch[i,t] + P_ev_dis[i,t]) × Δt
```

第 2 小问中的 EV 寿命成本使用车辆级 LP 变量精确计算，不再使用第 1 小问的聚合加权近似。
