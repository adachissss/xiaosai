# 第二问第 1 小问图片说明

本目录当前包含第二问第 1 小问生成的图片。

来源脚本：

```text
B/problem2/scripts/solve_degradation_indicator.py
```

---

## 1. `p2_q1_throughput_indicator.png`

- **含义**：第一问四组方案的电池等效吞吐量对比。
- **指标**：
  ```text
  E_throughput = Σ(P_charge + P_discharge) × Δt
  ```
- **作用**：说明不同方案对固定储能和 EV 电池的使用强度。吞吐量越大，意味着电池经历的充放电能量越多，潜在寿命损耗越大。

## 2. `p2_q1_degradation_cost.png`

- **含义**：第一问四组方案按寿命损耗指标折算后的电池损耗成本对比。
- **指标**：
  ```text
  C_degradation = c_deg × E_throughput
  ```
- **作用**：把固定储能和 EV 的电池损耗统一折算成经济成本，为第二问第 2 小问把寿命损耗加入优化目标函数做准备。
