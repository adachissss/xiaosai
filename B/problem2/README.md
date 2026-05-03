# B 题第二问目录说明

本目录用于存放第二问的代码、结果和图片，避免与第一问混淆。

## 当前进度

已完成第二问第 1 小问：提出固定储能与 EV 电池寿命损耗指标或成本折算方式。

## 目录结构

```text
problem2/
├── scripts/
│   ├── degradation_model.py              # 寿命损耗指标和成本计算函数
│   └── solve_degradation_indicator.py    # 第二问第1小问入口脚本
└── results/
    ├── p2_q1_degradation_metrics.csv     # 第一问四方案的寿命损耗指标试算
    ├── p2_q1_ev_degradation_parameters.csv
    ├── p2_q1_summary.txt
    └── figures/
        ├── p2_q1_throughput_indicator.png
        └── p2_q1_degradation_cost.png
```

## 运行方式

在项目根目录运行：

```bash
conda run -n xiaosai-b python B/problem2/scripts/solve_degradation_indicator.py
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
