# Strategies 模块说明

本目录实现“调度策略层”。
核心目标：**策略一次性读取完整世界状态，输出每辆车在当前时间步的计划**，由 `WorldManager` 负责执行。

## 当前架构

- 基类 `SchedulingStrategy`（`base.py`）只负责：
  - 调用子类 `build_plans(context)`
  - 为未返回的车辆补空计划
  - 校验计划合法性（动作链长度、动作类型）
- 子类（如 `nearest_task` / `max_weight`）负责：
  - 任务-车辆匹配
  - 充电决策
  - 任务链与动作链编排（可返回多节点链）

这意味着：
- 匹配逻辑、充电逻辑全部在子类中，可自由演进。
- 基类不再限制“逐车挑任务”或“单任务链”。

## 输入：`StrategyContext`

定义：`base.py`

包含以下信息：
- `tick`：当前时间步
- `depot_node`：仓库节点
- `config`：全局配置
- `graph`：路网图
- `oracle`：最短路查询器
- `vehicles`：车辆全集
- `tasks`：任务全集
- `stations`：充电站全集

## 输出：`VehiclePlan`

定义：`base.py`

按约定仅保留 4 个字段：
- `vehicle_id: int`
- `task_id: List[int]`
- `action: List[str]`
- `planned_path: List[int]`

约束：
- `len(action) == len(planned_path)`
- `action` 取值必须在：`keep/load/unload/charge`

## 链式动作示例

策略可以输出类似链条：

- `planned_path = [A, G, H]`
- `action = ["load", "unload", "charge"]`

这表示：
1. 到 `A` 执行装货
2. 到 `G` 执行卸货
3. 到 `H` 执行充电

`WorldManager` 会把关键点链展开成逐边路线，并在到达时触发动作。

## 与 WorldManager 的衔接

`WorldManager.schedule_step()` 只做两件事：
1. 调用 `strategy.plan(context)` 获取全车计划
2. 承接计划并生成：
   - 逐边行驶路径
   - 预计到达时间步 `planned_arrivals`
   - 对齐动作链 `planned_actions`

之后在 `world_manager_step()` 中，按到达事件执行动作。

## 现有策略

### 1) `NearestTaskStrategy`
- 名称：`nearest_task`
- 任务-车辆匹配目标：优先总路程更短（回仓 + 配送）
- 次目标：截止时间更早
- 再次目标：重量更大

### 2) `MaxWeightStrategy`
- 名称：`max_weight`
- 任务-车辆匹配目标：优先重量更大
- 次目标：截止时间更早
- 再次目标：总路程更短

两者都在子类内部实现：
- 全局贪心匹配
- 低电量补能
- 任务后置充电链（按需追加 `charge`）

## 扩展新策略

1. 新建 `your_strategy.py`
2. 继承 `SchedulingStrategy`
3. 实现 `build_plans(context) -> Dict[int, VehiclePlan]`
4. 在 `__init__.py` 和 `main.py` 注册

最小示例：

```python
class YourStrategy(SchedulingStrategy):
    name = "your_strategy"

    def build_plans(self, context):
        plans = {}
        for vehicle_id, vehicle in context.vehicles.items():
            plans[vehicle_id] = VehiclePlan(vehicle_id=vehicle_id)
        return plans
```
