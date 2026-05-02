# 第一问图片说明

本目录用于统一存放 B 题第一问相关图片。

图片主要由两个脚本生成：

- `B/scripts/solve_problem1.py`：运行 S0/S1/S2/S3 四组方案，并生成基础对比图。
- `B/scripts/plot_problem1.py`：读取已生成的结果 CSV，生成更适合论文展示的分析图。

---

## 1. `grid_import_comparison.png`

- **来源脚本**：`B/scripts/solve_problem1.py`
- **含义**：四组方案的外网购电功率对比。
- **作用**：用于观察 S0、S1、S2、S3 在一周内各时段的购电功率差异，重点体现协同优化是否降低峰值购电、平滑购电曲线。

## 2. `battery_energy_comparison.png`

- **来源脚本**：`B/scripts/solve_problem1.py`
- **含义**：四组方案的固定储能电量变化对比。
- **作用**：展示不同策略下固定储能的充放电行为，说明规则储能和优化调度对储能利用方式的影响。

## 3. `ev_net_power_comparison.png`

- **来源脚本**：`B/scripts/solve_problem1.py`
- **含义**：四组方案的 EV 聚合净功率对比。
- **作用**：正值表示 EV 聚合充电，负值表示 EV 向园区反向放电。该图用于展示 EV 智能充电和 V2B 对系统运行的贡献。

## 4. `pv_utilization.png`

- **来源脚本**：`B/scripts/solve_problem1.py`
- **含义**：S3 完整协同方案下的光伏可用功率、利用功率和弃光功率。
- **作用**：用于说明协同方案对光伏的消纳情况，以及是否存在弃光。

---

## 5. `fig1_grid_import_comparison.png`

- **来源脚本**：`B/scripts/plot_problem1.py`
- **含义**：四组方案外网购电功率对比图。
- **作用**：这是论文中最核心的对比图之一，用于展示协同运行对电网购电曲线和峰值购电的改善效果。

## 6. `fig2_battery_soc_comparison.png`

- **来源脚本**：`B/scripts/plot_problem1.py`
- **含义**：四组方案固定储能 SOC / 电量变化对比。
- **作用**：用于比较不同方案下固定储能是否被有效利用，以及储能运行是否满足上下限约束。

## 7. `fig3_ev_net_comparison.png`

- **来源脚本**：`B/scripts/plot_problem1.py`
- **含义**：四组方案 EV 聚合净功率对比。
- **作用**：用于说明 EV 从“即插即充”到“智能充电 / V2B”的变化，体现 EV 作为柔性资源参与削峰和降低成本的作用。

## 8. `fig4_s3_supply_stack.png`

- **来源脚本**：`B/scripts/plot_problem1.py`
- **含义**：S3 完整协同方案的供给侧功率堆叠图。
- **作用**：展示 S3 中园区负荷由光伏、固定储能、EV 放电和外网购电共同供给的结构，说明多资源协同供能关系。

## 9. `fig5_pv_utilization.png`

- **来源脚本**：`B/scripts/plot_problem1.py`
- **含义**：S3 完整协同方案下光伏利用、售电和弃光情况。
- **作用**：用于展示光伏消纳率和弃光率，说明协同策略是否充分吸收可再生能源。

## 10. `fig6_flexible_load.png`

- **来源脚本**：`B/scripts/plot_problem1.py`
- **含义**：S3 完整协同方案下三类建筑柔性负荷调整情况。
- **作用**：展示办公楼、湿实验楼、教学中心的负荷转移和削减量，说明建筑柔性负荷如何参与系统调度。

## 11. `fig7_cost_comparison.png`

- **来源脚本**：`B/scripts/plot_problem1.py`
- **含义**：四组方案成本构成对比。
- **作用**：用于分析协同方案降低总成本的来源，包括购电费用、峰值购电惩罚、负荷削减惩罚和负荷转移惩罚等。

## 12. `fig8_daily_import_comparison.png`

- **来源脚本**：`B/scripts/plot_problem1.py`
- **含义**：四组方案每日外网购电量对比。
- **作用**：从日尺度比较不同方案的购电量变化，观察协同优化在哪些日期效果更明显。

## 13. `fig9_summary_metrics.png`

- **来源脚本**：`B/scripts/plot_problem1.py`
- **含义**：四组方案关键指标汇总图。
- **作用**：集中展示峰值购电功率、EV 离站电量缺口、建筑负荷转移量和建筑负荷削减量，方便快速总结四组方案的差异。
