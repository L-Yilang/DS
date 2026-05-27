# Gurobi 全局求解操作手册

本文档说明 `run_gurobi.py` 怎么跑、参数怎么改、输出文件怎么看。建议所有命令都在项目根目录执行：

```powershell
cd "C:\Users\19653\Desktop\数据结构大作业\DS 5.18"
```

## 1. 推荐运行命令

如果 PowerShell 前面已经显示 `(DS)`，说明环境已经激活，直接运行：

```powershell
python run_gurobi.py --scale small --objective score --rounds 5 --slot-mode balanced --time-limit 600
```

不要再套一层 `conda run -n DS`。`conda run` 默认会捕获输出，容易表现为“终端一直不动”。

如果你没有激活 DS 环境，推荐用 `--no-capture-output`：

```powershell
conda run --no-capture-output -n DS python run_gurobi.py --scale small --objective score --rounds 5 --slot-mode balanced --time-limit 600
```

### 1.1 标准 small，使用已有 distribution，跑 5 轮，总分最大化

```powershell
python run_gurobi.py --scale small --objective score --rounds 5 --slot-mode balanced --time-limit 600
```

含义：

- 从 `outputs/distribution` 自动读取 `small_hyper_selector_rXX_seed..._task_distribution.json`。
- 每轮使用已有任务释放时间、重量、截止期。
- 用全知视角求解。
- 只优化真实业务总分：完成收益 - 超时罚分 - 里程罚分。
- 每轮最多跑 600 秒，到时自动交出当前最好解并进入下一轮。

### 1.2 medium 或 large

```powershell
python run_gurobi.py --scale medium --objective score --rounds 5 --slot-mode balanced --time-limit 600
```

```powershell
python run_gurobi.py --scale large --objective score --rounds 5 --slot-mode balanced --time-limit 600
```

### 1.3 只跑一个指定 distribution 文件

```powershell
python run_gurobi.py --scale small --objective score --rounds 1 --slot-mode balanced --time-limit 600 --distribution-file "outputs\distribution\small_hyper_selector_task_distribution.json"
```

### 1.4 严格求最优，不设时间限制

```powershell
python run_gurobi.py --scale small --objective score --rounds 5 --slot-mode balanced
```

注意：不加 `--time-limit` 就会一直求到 Gurobi 证明最优或你手动停止。small 一轮也可能 30 到 60 分钟。

## 2. 关键参数说明

| 参数 | 推荐值 | 含义 |
|---|---|---|
| `--scale` | `small/medium/large` | 选择问题规模。`all` 会三个规模都跑。 |
| `--objective` | `score` | 优化目标。`score` 是真实业务总分；`multi` 是多目标；`all` 两个都跑。 |
| `--rounds` | `5` | 跑几轮。使用 distribution 时，会按轮次读取已有文件。 |
| `--task-source` | `distribution` | 默认读取 `outputs/distribution` 里已有任务序列。 |
| `--distribution-dir` | `outputs/distribution` | distribution 文件所在目录。 |
| `--distribution-strategy` | `hyper_selector` | 自动匹配文件名时用的策略名。 |
| `--distribution-file` | 空 | 指定单个 distribution 文件。指定后只跑这个文件。 |
| `--slot-mode` | `balanced` | 每辆车可执行任务槽位设置。推荐 balanced，避免 `max_slots=4` 运力封印。 |
| `--slot-buffer` | `8` | balanced 模式额外给每辆车的槽位余量。任务多或结果偏低可调大。 |
| `--max-slots` | 不建议手动设 | 强制每辆车最多执行几单。设太小会严重压低 Gurobi 成绩。 |
| `--time-limit` | `300/600` | 每轮最多求解秒数。到时输出当前最好解，不要手动 Ctrl+C。 |
| `--mip-gap` | 默认空 | 允许相对 gap。正式对比建议不设；调试可设 `0.05`。 |
| `--threads` | 默认空 | Gurobi 线程数。默认让 Gurobi 自己决定。 |
| `--quiet` | 可选 | 不显示完整 Gurobi 日志，但仍打印最终求解状态和结果。 |
| `--output-dir` | `outputs/gurobi` | Gurobi 输出目录。 |

## 3. 目标函数怎么选

### 推荐：`--objective score`

这是和原项目动态策略对齐的真实业务目标：

```text
总分 = 完成收益 - 超时罚分 - 总里程罚分
```

如果你要和 `main.py` 跑出来的 `summary.csv` 比分数，必须用这个。

### 谨慎：`--objective multi`

多目标优先级是：

1. 最大完成数量；
2. 最小超时数量；
3. 最大总分。

它适合做“完成率优先”的附加实验，不适合作为和动态策略比较总分的主结果。因为它可能为了多完成任务牺牲业务总分。

## 4. 输出文件

默认输出到 `outputs/gurobi`：

| 文件 | 内容 |
|---|---|
| `summary.csv` | 所有轮次、所有目标的汇总表。 |
| `summary_original_compatible.csv` | 按原 `main.py` 的 summary 字段整理，方便和动态策略拼表。 |
| `<scale>_<objective>_rounds.csv` | 单个规模+目标的逐轮结果。 |
| `<scale>_<objective>_average.csv` | 单个规模+目标的平均结果。 |
| `report.md` | 可直接放进报告的结果摘要：先平均，后逐轮明细。 |
| `*_solution.json` | Gurobi 最终排班表，包含每辆车执行哪些任务、时间、电量、载重。 |
| `replay/*_replay.json` | 按原项目 replay 结构导出的 Gurobi 静态解快照。 |
| `logs/*.log` | Gurobi 原生日志，可查看 incumbent、bound、gap、节点数。 |

终端每轮求解结束后会立刻打印：

```text
[gurobi-result] scale=small objective=score round=1 seed=...
[gurobi-result] status=TIME_LIMIT runtime=600.00s obj=... bound=... gap=...
[gurobi-result] score=... completed=.../... completion_rate=... overdue=... timeout_rate=... distance=... completed_weight=...
```

常见 `status`：

- `OPTIMAL`：已经证明最优。
- `TIME_LIMIT`：时间到了，但通常已有当前最好可行解。
- `INFEASIBLE`：模型不可行，需要检查参数或槽位。
- `INTERRUPTED`：人为 Ctrl+C 或外部中断，不建议用于正式结果。

## 5. 和原项目动态策略对比

先跑动态策略：

如果已经在 `(DS)` 环境：

```powershell
python main.py --scale small --strategy all
```

再跑 Gurobi：

```powershell
python run_gurobi.py --scale small --objective score --rounds 5 --slot-mode balanced --time-limit 600
```

对比时看：

- 动态策略：`outputs/summary.csv`
- Gurobi：`outputs/gurobi/summary.csv` 或 `outputs/gurobi/report.md`
- 如果要拼到原项目的表格格式里，用 `outputs/gurobi/summary_original_compatible.csv`

注意控制变量：

- 标准实验就都用标准规模，不要一边 long_train 一边 standard。
- Gurobi 默认读取 `outputs/distribution` 中已有任务序列；确保这些 distribution 是你要对比的那批任务。
- 如果你只对比某一个 seed，就用 `--distribution-file` 指定那个文件。

## 6. 参数调试建议

### 6.1 分数明显偏低

先检查命令里有没有手动写：

```powershell
--max-slots 4
```

如果有，删掉它。推荐：

```powershell
--slot-mode balanced
```

如果仍觉得车队运力不够，可以加大：

```powershell
--slot-buffer 12
```

### 6.2 跑太久

不要手动 Ctrl+C。用：

```powershell
--time-limit 300
```

或：

```powershell
--time-limit 600
```

这样 Gurobi 会正常收尾、导出当前最好解、继续下一轮。

### 6.3 想看求解进度

不加 `--quiet`，终端会显示 Gurobi 日志。若你用 `conda run -n DS` 后终端一直不动，换成下面任一方式：

```powershell
python run_gurobi.py --scale small --objective score --rounds 5 --slot-mode balanced --time-limit 600
```

或：

```powershell
conda run --no-capture-output -n DS python run_gurobi.py --scale small --objective score --rounds 5 --slot-mode balanced --time-limit 600
```

也可以看：

```text
outputs/gurobi/logs/<scale>_<objective>_rXX_seedXXXX.log
```

### 6.4 要严格最优

去掉 `--time-limit` 和 `--mip-gap`。这会很慢，但如果最终状态是 `OPTIMAL`，就是已证明最优。

## 7. 当前模型口径

- 任务全知：Gurobi 一开始知道全部任务。
- 任务不完成：不给完成收益；若 deadline 在 horizon 内经过，则扣一次超时罚分。
- 晚完成：保留双重惩罚，即 timeout penalty 加 late completion negative reward。
- 充电：允许在仓库部分充电，用整数 tick 表示充电时长。
- 任务结构：每单都是仓库装货、送到目的地卸货，再回到仓库进入下一单。
- distribution：读取已有 release time、weight、deadline offset；任务目的地用同 seed 复原，因为旧 distribution 文件没有保存 destination。
