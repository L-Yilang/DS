# Hyper Selector 调度策略说明

## 1. 一句话定位

`hyper_selector` 是构建在 `time_first_bundle` 骨架之上的 **内嵌型超启发式（hyper-heuristic）**：

- 不再在“策略级”（如 nearest / max_weight / tfb）之间切换；
- 而是用 **Q-learning** 在 TFB 内部，**按时间窗动态切换**两个子层的工作模式：
  - **任务派发子层（dispatch）**：决定派单偏好（紧急优先、拼单优先、能量保守…）。
  - **充电决策子层（charge）**：决定充电再平衡的激进程度。

可以理解为：**TFB 仍然负责“怎么做”，Q-learning 负责“此时此刻该用哪一套参数”。**

---

## 2. 它在系统里怎么运行

策略继承自 `TimeFirstBundleStrategy`，每个 tick 的入口仍是 `build_plans(context)`：

```
build_plans(context)
  └── _ensure_window_policy(context)   # 维护当前 Q-learning 窗口
  └── _apply_active_modes()            # 把当前 (dispatch, charge) 动作映射成 TFB 内部参数
  └── super().build_plans(context)     # 复用 time_first_bundle 的派单/拼单/充电逻辑
```

也就是说：

- 输入仍是 `StrategyContext`（tick、车辆、任务、充电站、配置、最短路 oracle 等）；
- 输出仍是 `Dict[int, VehiclePlan]`（每辆车的关键节点序列 + 动作链）；
- **真正改变行为的是“在窗口开始时选了哪一对动作”**：这套动作会改写 TFB 父类里若干阈值与权重，然后 TFB 像往常一样跑。

---

## 3. Q-learning 设计概览

### 3.1 双 Q 表结构

策略维护两张独立的 Q 表：

- `_q_dispatch: state -> List[float]`，对应 4 个派单动作；
- `_q_charge:   state -> List[float]`，对应 3 个充电动作。

两个子层共享同一个 `state`，但各自独立更新各自的 Q 值。这样可以把“派单偏好”和“充电节奏”这两件相对正交的事拆开学习，状态空间不会爆炸。

### 3.2 时间窗机制

- 每隔 `window_ticks = 20` 个 tick 视为一个窗口；
- 窗口开始时：根据当前 `state` 用 ε-greedy 各选一个动作，并写到 TFB 内部参数里；
- 窗口结束时：根据“窗口内的真实表现”计算奖励，**同时更新两张 Q 表**，然后立即开下一个窗口。

窗口边界由 `_next_switch_tick` 控制，进入新窗口时调用 `_start_new_window(context)`。

### 3.3 状态编码 `_build_state`

状态是离散化后的 6 维元组：

| 维度 | 含义 | 离散化方式 |
|------|------|------------|
| `backlog_bin`  | 待处理任务数 / 车辆数（积压比） | clip 到 0..4 |
| `urgent_bin`   | 紧急任务比例（用 TFB 的紧急判断） | 5 档 0..4 |
| `battery_bin`  | 全队平均电量比 | 5 档 0..4 |
| `pressure_bin` | 充电站平均压力指数 | 4 档 0..3 |
| `idle_bin`     | 空闲车比例 | 5 档 0..4 |
| `phase_bin`    | 时间进度（tick / horizon） | 3 档 0..2 |

设计原则：**只用调度真正在意的“宏观信号”**，避免维度爆炸；同时刻意保留 `phase_bin`，让早/中/晚期可以学到不同节奏。

### 3.4 动作空间

派单子层（4 个动作，作用于本类 `_apply_dispatch_mode`）：

- `dispatch_balanced`：完全使用 TFB 默认参数。
- `dispatch_urgent_first`：抬高 `emergency_margin_ticks`、缩短拼单等待，**模式1 偏置 -0.25**（更倾向单送）。
- `dispatch_bundle_first`：放宽 `max_detour_ratio`、加大拼单等待窗口，**模式2/3 偏置 -0.15 / -0.10**（更倾向拼单）。
- `dispatch_energy_safe`：提高能量代价权重 `w_energy`、收紧绕路阈值，**模式1 偏置 -0.20**。

充电子层（3 个动作，作用于本类 `_apply_charge_mode`）：

- `charge_balanced`：默认。
- `charge_proactive`：更激进补电（`keep_idle_ratio` 降到 0.22、`depot_charge_ratio_no_pressure` 提到 0.95、站点排队权重放大 1.2）。
- `charge_task_first`：更克制补电（保留更多空闲车应对突发任务，站点排队权重缩到 0.8）。

派单与充电相互独立选择，组合后总共 **4 × 3 = 12 套行为画像**。

### 3.5 ε-greedy 选择

`_select_action`：

- 以 `epsilon = 0.12` 概率随机探索；
- 否则取当前 state 下 Q 值最大的动作；
- 多个 Q 值并列最大时随机打破平手（避免实现层面的偏置）。

### 3.6 Q 更新公式

`_q_update` 用的是经典 Q-learning：

```
target = reward + gamma * max(Q[next_state])
Q[state, action] += alpha * (target - Q[state, action])
```

超参数：`alpha = 0.25`，`gamma = 0.85`。两个表用同一份 `reward` 同时更新（窗口奖励对它们没有差别，差别在动作集合上）。

### 3.7 窗口奖励 `_compute_window_reward`

对窗口的“增量统计”做线性组合：

```
reward = reward_completed * Δcompleted
       - reward_overdue   * Δoverdue_penalized
       - reward_distance  * Δtotal_distance
```

默认权重：

- `reward_completed = 1.0`
- `reward_overdue   = 3.0`
- `reward_distance  = 0.02`

含义：**多完成 1 单 +1，多超时 1 单 -3，行驶距离每单位 -0.02**。  
即“多干活赚得最多，但超时是重罚，长距离是温和惩罚”，这与 TFB 的目标导向是一致的。

---

## 4. 与 TFB 的耦合方式

`hyper_selector` 不重写 TFB 的派单主流程，而是通过 **改写实例属性 + 两个轻量 hook** 注入差异：

### 4.1 改写阈值与权重

在窗口开始时调用 `_apply_active_modes`：

1. `_set_balanced_defaults()`：把所有相关参数还原到 TFB 默认；
2. `_apply_dispatch_mode(idx)`：按动作覆盖紧急阈值、拼单等待、能量权重等；
3. `_apply_charge_mode(idx)`：按动作覆盖空闲保留比例、无压力主动充电阈值、站点排队权重。

这样可以保证 **每个窗口都从“干净状态”开始**，不会把上一个窗口的修改累积进来。

### 4.2 两个 hook：模式偏置和站点评分

- `_feasible_modes_for_pair(...)`：在 TFB 计算拼单三模式（mode1/2/3）的代价后，**额外叠加 `_mode_bias`**。这就是 `dispatch_urgent_first / bundle_first / energy_safe` 改变拼单偏好的入口。
- `_select_best_station(...)`：重写最近站选择逻辑，把 `queue_cost` 乘上 `_station_score_queue_scale`。`charge_proactive` 把它放大、`charge_task_first` 把它缩小，从而更看重 / 更不看重站点排队。

### 4.3 学到的不是“新算法”，是“如何在情境间切档”

强调一点：**Q-learning 没有重写匹配逻辑**。它学的是“在当前 6 维状态下，应该把 TFB 调成哪种风格”。所以即使早期 Q 表全 0、纯靠 ε-greedy 探索，行为也已经是合法的 TFB 调度，不会失控。

---

## 5. Trace 日志：可观测性设计

`_append_trace_row` 会把每个窗口的关键信息写到 `outputs/hyper_selector_trace.csv`，列包括：

- `run_tag`：本次运行的随机标记（避免多次运行混在一张表里）；
- `window_index`、`tick_end`：窗口编号与结束 tick；
- `state`：以 `|` 分隔的 6 维状态字符串；
- `dispatch_action`、`charge_action`：本窗口实际执行的动作；
- `reward` 以及 `completed_delta` / `overdue_delta` / `distance_delta`；
- `q_dispatch_before/after`、`q_charge_before/after`：可以直接看到 Q 值是怎么被这一次更新拉动的；
- `vehicle_count`：车辆数（用于不同规模的对比）。

同时控制台还会打两行日志：

- `[hyper_selector] window=... start_tick=... state=... dispatch=... charge=...`
- `[hyper_selector] window=... tick=... reward=... q_dispatch=... q_charge=...`

便于在跑大批次比较实验时，**复盘“哪种状态下学到了哪种行为，以及奖励曲线的走向”**。

---

## 6. 关键参数解释与调参建议

### 6.1 Q-learning 超参数

- `window_ticks`（默认 20）
  - 越大：每个窗口奖励信号更稳定，但适配慢，小规模任务可能整局只学到几步。
  - 越小：响应更快，但奖励噪声大，Q 值抖动明显。
- `epsilon`（默认 0.12）
  - 越大：探索更多，长期可能收益更高，但短期表现波动；
  - 想做“评估跑”时建议直接调到 0（纯利用）。
- `alpha`（默认 0.25）
  - 学习率；越大对最近窗口越敏感，过大容易把偶发好/坏运气学成偏见。
- `gamma`（默认 0.85）
  - 折扣因子；偏大代表更看重“接下来还会持续受益”的动作，对“后期任务密集”的场景更有利。

### 6.2 奖励权重

- `reward_completed / reward_overdue / reward_distance`
  - 想极致追求超时率：再放大 `reward_overdue`；
  - 想压低里程：放大 `reward_distance`；
  - 注意三者的尺度差异：完成数和超时数是“个”，距离是“距离单位”，调比例时建议先看一次实际 trace 中三项的量级再改。

### 6.3 状态离散化

`_build_state` 里那一组 `min(...)` 的 bin 边界是经验值，如果发现某一维度**几乎只取一个值**（比如 `phase_bin` 永远是 0），可以适当放宽 / 收紧档数，让 Q 表的“行”更有区分度，但要注意 **状态数 ≈ 5×5×5×4×5×3 = 7500**，不要再无脑加维度。

### 6.4 动作集合

如果要扩展：

- **加派单动作**：在 `_dispatch_actions` 增加名字，并在 `_apply_dispatch_mode` 给出参数偏移；
- **加充电动作**：同理改 `_charge_actions` 与 `_apply_charge_mode`；
- 注意：动作越多，Q 表收敛越慢，建议每次只加 1~2 个真正“有不同立场”的动作，避免互相重叠。

---

## 7. 与 TFB / 其它策略的关系

- 行为下限：**任何窗口都至少是一个合法 TFB 调度**，不会出现因为 RL 错误产生的非法计划，鲁棒性好。
- 行为上限：受制于 TFB 的设计上限——它仍然是“两单拼单 + 时间优先”，不会突然学出多单链或全局重排。
- 对 `time_first_bundle` 的兼容：所有 TFB 的可调参数都被统一到 `_set_balanced_defaults()` 里集中管理，要给 TFB 增加新参数时，**记得也在这里加一行默认值**，否则窗口切换会把它“漏掉”。
- 与 `RL_charging` 的差异：`RL_charging` 主要是“在车辆级别做 RL 决定个体充电时机”；`hyper_selector` 是“在策略级别做 RL 决定整体策略风格”，两者粒度完全不同，可以相互组合也可以单独使用。

---

## 8. 优点、局限与适用场景

### 8.1 优点

- **稳定**：行为下限就是 TFB，不会因为 RL 训练不充分而崩盘；
- **可解释**：每个动作都对应一组人类能读懂的参数偏移；
- **可观测**：trace CSV + 控制台日志，能直接讲清楚“为什么这个窗口选了 bundle_first”；
- **状态/动作设计正交**：派发与充电分两张表，学起来不打架。

### 8.2 局限

- 是 **离散表格 Q-learning**，状态量化粗、不能自动迁移到“没见过的”宏观情境；
- 单条轨迹的窗口数往往不多（中等规模约几十个窗口），收敛较慢；
- 奖励是“线性加权三项”，没有显式建模公平性、客户体验等指标；
- 探索期 ε-greedy 偶尔会选到明显次优动作，**单次评估时建议用 ε=0 跑一次纯利用结果**。

### 8.3 适用场景

- 课程项目里需要展示“**有学习能力的调度**”但又不想引入大模型 / 神经网络的场合；
- 作为 TFB 的 **自适应包装**，在多种规模/seed 上获得更稳健的整体表现；
- 作为后续“Deep RL 调度”研究的强可解释基线。

---

## 9. 一句话总结

> `hyper_selector` 用一张极小的双 Q 表，给 `time_first_bundle` 装了一个会随情境换档的“自动驾驶模式开关”：  
> **TFB 决定动作怎么落地，Q-learning 决定此刻应该是哪种风格的 TFB。**  
> 它的目标不是“跑出理论最优”，而是“在不同状态下都比单一 TFB 配置更稳一点”。
