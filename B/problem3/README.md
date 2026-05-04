# B 题第三问目录说明

本目录用于存放第三问的代码、结果和图片。

## 当前进度

已完成第三问全部三个小问的碳排放约束与碳交易影响分析。

- 第1小问：不同碳排放上限对调度行为的影响
- 第2小问：不同碳交易价格对购电、储能、EV V2B 的影响
- 第3小问：兼顾经济性、电池寿命与碳排放的综合策略

## 目录结构

```text
problem3/
├── scripts/
│   ├── carbon_aware_strategy.py    # S5 碳排放感知 LP 模型
│   └── solve_problem3.py           # 第三问入口脚本
├── results/
│   ├── p3_q1_comparison_metrics.csv
│   ├── p3_q1_summary.txt
│   ├── p3_q2_comparison_metrics.csv
│   ├── p3_q2_summary.txt
│   ├── p3_q3_summary.txt
│   └── figures/
│       ├── p3_q1_emissions_by_cap.png
│       ├── p3_q1_grid_import_by_cap.png
│       ├── p3_q1_battery_ev_by_cap.png
│       ├── p3_q1_cost_breakdown_by_cap.png
│       ├── p3_q2_emissions_vs_price.png
│       ├── p3_q2_grid_import_by_price.png
│       ├── p3_q2_battery_ev_by_price.png
│       └── p3_q2_cost_breakdown_by_price.png
├── 第三问分析思路.md
├── 第三问第1小问解决思路.md
├── 第三问第2小问解决思路.md
├── 第三问第3小问解决思路.md
└── README.md
```

## 运行方式

在项目根目录运行：

```bash
conda run -n xiaosai-b python B/problem3/scripts/solve_problem3.py
```

## S5 模型说明

S5 在第二问 S4 模型（运行成本 + 电池寿命损耗）基础上增加：

1. **碳排放交易成本**：`carbon_price × grid_buy[t] × carbon_intensity[t] × Δt`
2. **全周总碳排放上限约束（可选）**：`Σ(grid_buy[t] × carbon_intensity[t] × Δt) ≤ cap`

目标函数：

```text
min 运行成本 + 固定储能寿命损耗 + EV 寿命损耗 + 碳交易成本
```

碳强度来源：`timeseries_15min.csv` 中的 `grid_carbon_kg_per_kwh` 字段（0.55-0.74 kg/kWh）。
