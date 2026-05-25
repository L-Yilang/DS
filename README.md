# 新能源物流车队协同调度模拟器

本项目是一个**离散时间步（tick）驱动**的新能源物流车队仿真系统，用于研究：
- 动态任务流下的调度策略表现
- 电量约束、充电站排队与任务截止时间的耦合关系
- 不同策略在同一随机种子下的可复现对比

项目包含两部分：
- `src/`：仿真核心（负责生成环境、推进世界、执行调度、导出结果）
- `replay_ui/`：回放可视化（只读取输出，不影响仿真逻辑）

---

## 1. 快速开始

### 1.1 运行仿真

```bash
python main.py
```

常用参数：

```bash
python main.py --scale small --strategy nearest_task
python main.py --scale medium --strategy hyper_selector
python main.py --experiment-mode long_train --strategy hyper_selector
python main.py --output-dir outputs
python main.py --save-timeline
```

参数说明：
- `--scale`: `small / medium / large / all`
- `--strategy`: `nearest_task / max_weight / time_first_bundle / rl_charging / hyper_selector / all`
- `--experiment-mode`: `standard / long_train`（`long_train` 会按倍率放大各规模 `horizon`）
- `--long-train-multiplier`: long_train 模式下 horizon 放大倍率（默认 `4`）

`hyper_selector` 当前实现为 **TimeFirstBundle 内部 Q-learning 超启发式**：每个窗口会选择 `dispatch` 与 `charge` 子规则，并打印窗口状态、动作与奖励日志。
- `--save-timeline`: 是否额外导出 `*_timeline.json`

### 1.2 打开回放

```bash
python -m http.server 8000
```

浏览器打开：
- `http://localhost:8000/replay_ui/index.html`

在页面中加载：
- `outputs/*_replay.json`

---

## 2. 项目结构

- `main.py`：实验入口，批量运行多规模多策略。
- `src/config.py`：全局参数与规模定义。
- `src/models.py`：任务、车辆、充电站、结果等核心模型。
- `src/graph_utils.py`：路网生成、最短路查询器。
- `src/strategies/`：策略接口与策略实现。
- `src/world.py`：世界管理器，主循环核心。
- `src/scoring.py`：评分函数（任务收益 + 路程减益）。
- `src/exporter.py`：导出 CSV / JSON。
- `replay_ui/`：回放前端页面。

---

## 3. 系统执行总览

每个 tick 固定执行两个操作：

1. **更新世界**（`world_manager_step`）
2. **调度决策**（`schedule_step`）

核心顺序：

```text
for tick in horizon:
    world_manager_step(tick)  # 先推进真实世界
    schedule_step(tick)       # 再基于最新状态做计划
    save_timestep(tick)
```

这样策略总是基于“刚更新完”的世界状态决策。

---

## 4. 核心数据结构

## 4.1 图的数据结构（RoadGraph）

位置：`src/graph_utils.py`

结构：
- `nodes: Dict[int, Node]`
- `adjacency: Dict[int, Dict[int, float]]`

含义：
- 外层 key：节点 `u`
- 内层 key：邻接节点 `v`
- value：边权（距离）

示例：

```python
adjacency = {
    0: {1: 4.2, 3: 5.1},
    1: {0: 4.2, 2: 3.7},
    2: {1: 3.7},
    3: {0: 5.1},
}
```

这是无向图，添加边时会同步写入 `u->v` 与 `v->u`。

---

## 4.2 任务数据结构（Task）

位置：`src/models.py`

字段：
- `task_id`: 任务编号
- `release_time`: 释放时间
- `destination_node`: 目的节点
- `weight`: 任务重量
- `deadline`: 截止时间
- `status`: `pending/assigned/in_progress/completed`
- `assigned_vehicle_id`: 当前承接车辆
- `completion_time`: 完成时间
- `overdue_penalized`: 是否已做超时处罚

示例：

```python
Task(
    task_id=12,
    release_time=35,
    destination_node=9,
    weight=18.5,
    deadline=102,
)
```

---

## 4.3 车辆数据结构（Vehicle，重点）

位置：`src/models.py`

Vehicle 字段可以分为 4 组，便于理解。

### A. 能力
- ID `vehicle_id`
- 电池容量 `battery_capacity`
- 载重量 `load_capacity`
- 速度 `speed`
- 耗电量 `energy_per_distance`

### B. 当前执行状态（每个 tick 变化）
- 状态 `state`：`idle/moving/loading/unloading/waiting_charge/charging`
- 位置 `current_node`
- 电量 `battery`
- 载重 `carried_weight`
- 当前在完成的任务 `assigned_task_id`
- `operation_timer`（装卸剩余时间）
- 总里程 `distance_travelled`

### C. 路径执行缓存（用于“物理移动”）
- `route: Deque[int]`：目标节点序列
- `planned_arrivals: List[(node_id, eta_tick)]` ：预期到达节点的时间
- `planned_actions: Deque[str]`：每个节点的目标行为
- `next_node`: 下一个节点
- `edge_remaining`: 当前边剩余距离
- `planned_task_ids: Deque[int]`：计划的任务ID

它们之间存在对齐关系：
- `planned_arrivals[i]` 对应 `planned_actions[i]`
- 例如到达 `N18@T120` 时执行 `unload`

### Vehicle 对象示例（创建时）

```python
Vehicle(
    vehicle_id=3,
    current_node=0,
    battery_capacity=160,
    load_capacity=24.0,
    speed=1.0,
    energy_per_distance=1.0,
    battery=158.2,
)
```

### Vehicle 运行态示例（timeline 快照中的单车）

```json
{
  "vehicle_id": 3,
  "state": "moving",
  "current_node": 7,
  "battery": 121.6,
  "battery_capacity": 158.2,
  "assigned_task_id": 101,
  "carried_weight": 18.5,
  "distance_travelled": 83.4,
  "route": [11, 14, 18],
  "planned_arrivals": [[11, 124], [14, 146], [18, 155]],
  "planned_actions": ["unload", "keep","charge"],
  "planned_task_ids": [101],
  "next_node": 11,
  "edge_remaining": 2.4
}
```

### Vehicle 计划展开示例（从策略输出到执行）

策略输出（关键点链）：

```python
VehiclePlan(
    vehicle_id=3,
    task_id=[101],
    action=["load", "unload", "charge"],
    planned_path=[0, 18, 5],
)
```

世界管理器：
- `planned_path` 被最短路展开为具体的 `route`
- 同时计算 `planned_arrivals`
- 到达每个关键节点后按 `planned_actions` 触发 `load/unload/charge`

---

## 4.4 充电站数据结构（ChargingStation）

字段：
- `station_id`
- `node_id`
- `piles`: 充电桩数量
- `charge_rate`: 每 tick 充电速度
- `charging_vehicle_ids`: 正在充电的车辆列表
- `queue_vehicle_ids`: 排队车辆队列

方法：
- `pressure_index()`：站点压力指标（正在充电 + 排队）/ 桩数

示例：

```python
ChargingStation(
    station_id=2,
    node_id=18,
    piles=2,
    charge_rate=2.4,
)
```

---

## 5. 路网如何生成

位置：`src/graph_utils.py`

生成流程：

1. 生成抖动网格点 `_generate_grid_nodes`
- 0 号节点固定在几何中心附近。
- 其他点在规则网格基础上做随机扰动，减少棋盘感。

2. 连接主路（低复杂度骨架）
- 只连右、下相邻格点。

3. 添加少量次路
- 低概率加入短对角线。

4. 添加极低概率跳一格短边
- 只允许本地短距离，避免跨区长边。

5. 强化中心仓库可达性
- 给 0 号点加四邻域与两步邻域连接。

### 尺度语义（small / medium / large）

当前实现是：
- **近似固定点间距**
- **地图边界随规模扩张**

所以 `large` 是空间尺度更大，而不是同样空间里更密集。

---

## 6. 世界管理器详解（WorldManager）

位置：`src/world.py`

## 6.1 操作 1：更新世界（`world_manager_step`）

按固定顺序：

1. `_spawn_tasks`
- 按任务率生成新任务。

2. `_apply_timeout_penalties`
- 对超时且未处罚任务施加一次性罚分。

3. `_advance_charging`
- 充电车辆加电。
- 充满车辆离桩转空闲。
- 尝试从排队队列补位。

4. `_advance_vehicle`
- 装/卸货：计时器递减，归零后结算。
- 行驶：推进边剩余距离，扣电，更新里程。
- 到站：按到达计划触发动作。

5. `_promote_waiting_queue`
- 在车辆推进后再次补位，降低“入队后一拍延迟”。

6. 到达动作触发（`_apply_due_arrival_actions`）

触发条件：
- `current_node == planned_arrivals[0].node`
- `tick >= planned_arrivals[0].eta`

更新动作：
- `keep`
- `load`
- `unload`
- `charge`
  
7. 失败机制
典型失败触发：
- 策略输出非法（动作链与目的地链长度不一致）
- 任务冲突（被其他车占用）
- 超载分配
- 电量不足以安全进入下一条边
- 在非充电站执行 `charge`

失败后：
- `simulation_failed=True`
- 记录 `failure_tick / failure_reason`
- 主循环提前结束



## 6.2 操作 2：调度（`schedule_step`）

- 组装 `StrategyContext`（完整世界输入）
- 调用 `strategy.plan(context)`
- 将全车 `VehiclePlan` 安装到车辆执行缓存

---

## 7. 调度接口输入输出示例

位置：`src/strategies/base.py`

## 7.1 输入 `StrategyContext`

```python
context = StrategyContext(
    tick=120,
    depot_node=0,
    config=sim_config,
    graph=road_graph,
    oracle=oracle,
    vehicles={1: v1, 2: v2},
    tasks={101: t101, 102: t102},
    stations={1: s1, 2: s2},
)
```

## 7.2 输出每辆车的 `VehiclePlan`（字典）

```python
VehiclePlan(
    vehicle_id=3,
    task_id=[101],
    action=["load", "unload", "charge"],
    planned_path=[0, 18, 5],
)
```

约束：
- `len(action) == len(planned_path)`
- `planned_path` 是关键目的地序列，不是逐边路径，之后会被world主循环的承接函数拓展。降低调度函数复杂程度。

---

## 8. 评分机制

位置：`src/scoring.py`

- `completion_score`：任务完成收益
- `timeout_penalty`：任务超时罚分
- `distance_penalty`：总里程减益

总体分数可理解为：

```text
总分 = 任务完成收益之和 - 超时罚分之和 - 总里程减益
```

---

## 9. 输出文件与字段

- `outputs/summary.csv`
- `outputs/*_replay.json`
- `outputs/*_timeline.json`（开启 `--save-timeline` 时）

`summary.csv` 重点字段：
- `total_score`
- `completed_tasks / total_tasks`
- `overdue_tasks / timeout_rate`
- `total_distance`
- `simulation_failed / failure_reason / failure_tick`

`*_replay.json` 结构：
- `schema_version`
- `replay_meta`（静态地图）
- `timeline`（动态快照）

---

## 10. 常见扩展方向

1. 新策略
- 在 `src/strategies/` 新增策略类并实现 `select_task`
- 在 `main.py` 的 `build_strategies()` 注册

2. 路网生成
- 修改 `src/graph_utils.py` 中节点生成与连边逻辑

3. 执行语义
- 修改 `src/world.py` 中动作触发、充电逻辑、失败判定

4. 可视化增强
- 修改 `replay_ui/app.jsx` 与 `replay_ui/styles.css`
- 建议保持“UI 不改变仿真语义”的边界

---

## 11. 建议实践

- 做策略对比时固定随机种子（`scale.seed`）。
- 在同一任务规模下比较多个策略，避免跨规模直接比总分。
- 如果做论文/报告，建议同时报告：完成率、超时率、总里程、失败率。
