# 普通中尺度基准实验

## 实验设置

- 规模：仅运行中尺度，固定测试种子为 `354787187, 1866801784, 736577488, 108402446, 550665828`。
- 规模扰动：中尺度保持默认任务数 162、均匀释放时间和均匀目的地抽样。
- 仿真扰动：仿真参数保持 run_final_outputs.py 的默认设置。
- MAPPO：在该实验设置下单独训练 1 轮，仅训练中尺度；评估 checkpoint 为 `outputs\supplementary_smoke_v2\mappo\standard_baseline\checkpoints\best_medium.pt`。
- 中尺度实际参数：task_count=162, task_time_distribution=uniform, task_time_mean_ratio=0.4, task_time_std_ratio=0.5, energy_per_distance=1.0, traffic=(0.0, 1.0)。

## 聚合结果

| 策略 | 平均得分 | 完成率 | 超时率 | 超时任务 | 总里程 | 失败率 |
|---|---:|---:|---:|---:|---:|---:|
| genetic_hyper | 9734.858 | 0.9975 | 0.0198 | 3.2 | 3218.38 | 0.0000 |
| ALNS | 9552.072 | 0.9938 | 0.0321 | 5.2 | 3153.58 | 0.0000 |
| time_first_bundle | 9511.034 | 0.9975 | 0.0222 | 3.6 | 3174.78 | 0.0000 |
| max_weight | 6435.516 | 0.9593 | 0.1568 | 25.4 | 3589.84 | 0.0000 |
| nearest_task | 6983.924 | 0.9593 | 0.1395 | 22.6 | 3548.22 | 0.0000 |
| MAPPO | 9392.212 | 0.9877 | 0.0395 | 6.4 | 3228.64 | 0.0000 |

## 简要分析

- 该实验作为补充实验的普通情况基准，用于衡量各扰动场景造成的性能变化。
- 综合失败率、平均得分和完成率，本实验中表现最好的策略为 `genetic_hyper`，平均得分 9734.858，完成率 0.9975。
- 风险最高的策略为 `max_weight`，失败率 0.0000，超时率 0.1568。

## 输出文件

- 明细：`standard_baseline/summary.csv`
- 聚合：`standard_baseline/summary_aggregated.csv`
- 回放：`standard_baseline/*_replay.json`
- 任务分布：`standard_baseline/distribution/*_task_distribution.json`