# B 题第一问目录说明

本目录存放 B 题第一问的代码、结果和图片，避免与第二问、第三问混淆。

## 目录结构

```text
problem1/
├── scripts/
│   ├── common.py                # 公共参数、数据读取、指标计算、约束检查
│   ├── baseline_strategy.py     # S0/S1 规则方案，以及 S2 部分协同入口
│   ├── coordinated_strategy.py  # 可配置线性规划策略，支持 S2/S3
│   ├── solve_problem1.py        # 第一问总入口
│   └── plot_problem1.py         # 第一问论文图生成脚本
└── results/
    ├── *_schedule.csv           # 各方案调度结果
    ├── *_ev_results.csv         # 各方案 EV 结果
    ├── comparison_metrics.csv   # 四方案指标对比
    ├── problem1_summary.txt     # 运行摘要
    └── figures/                 # 第一问图片和图片说明
```

## 数据来源

第一问仍然读取公共数据目录：

```text
B/B_data/
```

## 运行方式

在 `B/` 目录下运行：

```bash
conda run -n xiaosai-b python problem1/scripts/solve_problem1.py
conda run -n xiaosai-b python problem1/scripts/plot_problem1.py
```

## 四组方案

| 方案 | 含义 |
|---|---|
| S0_no_storage | 储能不运行，EV 即插即充，建筑不调节 |
| S1_rule_storage | 规则储能，EV 即插即充，建筑不调节 |
| S2_partial_coordination | 储能优化 + EV 智能充电，EV 不放电，建筑不调节 |
| S3_full_coordination | 储能 + EV V2B + 建筑柔性负荷完整协同 |
