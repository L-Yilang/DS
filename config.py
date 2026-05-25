from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class ScaleConfig:
    """问题规模配置。"""

    name: str  # 规模名称（small / medium / large）
    node_count: int  # 节点数量
    vehicle_count: int  # 车辆数量
    station_count: int  # 充电站数量
    horizon: int  # 仿真时间范围（时间跨度）
    task_count: int  # 整个仿真期间将释放的任务总数
    seed: int  # 随机种子

    #在这里调整是没用的，请在下面每个规模里调整
    task_time_distribution: str   # 任务释放时间分布（uniform / gaussian）
    task_time_mean_ratio: float  # 高斯分布均值在可释放时间窗中的相对位置
    task_time_std_ratio: float   # 高斯分布标准差在可释放时间窗中的相对比例
    task_weight_mean: float  # 任务重量均值
    depot_station_piles: Optional[int] = None  # 中央仓库充电桩数量（None 表示按车辆总数）
    non_depot_station_piles_range: Tuple[int, int] = (1, 3)  # 非仓库充电站桩数范围


@dataclass(frozen=True)
class SimulationConfig:
    """仿真全局参数配置。"""

    depot_node: int = 0  # 仓库（起始节点）编号
    min_edge_distance: int = 3  # 边的最小距离
    max_edge_distance: int = 10  # 边的最大距离
    extra_edge_factor: float = 2.6  # 额外边的生成系数（控制图的稠密程度）
    vehicle_battery_capacity: float = 160.0  # 车辆电池容量
    vehicle_load_capacity_range: Tuple[float, float] = (26.0, 26.0)  # 车辆载重范围
    vehicle_speed: float = 1.0  # 车辆速度
    energy_per_distance: float = 1.0  # 单位距离能耗
    loading_duration: int = 4  # 装载时间
    unloading_duration: int = 4  # 卸载时间
    deadline_range: Tuple[int, int] = (35, 120)  # 任务截止时间范围
    task_weight_range: Tuple[float, float] = (4.0, 26.0)  # 任务重量范围
    timeout_penalty: float = 18.0  # 超时惩罚
    completion_base_reward: float = 80.0  # 完成任务基础奖励
    late_completion_penalty_factor: float = 35.0  # 延迟完成惩罚系数
    distance_penalty_factor: float = 0.06  # 距离惩罚系数
    queue_penalty_factor: float = 4.0  # 排队惩罚系数
    low_battery_ratio: float = 0.22  # 低电量阈值比例（触发充电策略）


def default_scales() -> List[ScaleConfig]:
    """返回蓝图要求的三种规模。"""

    return [
        ScaleConfig(
            name="small",  # 小规模
            node_count=15,  # 节点数量
            vehicle_count=3,  # 车辆数量

            station_count=2,  # 充电站数量

            depot_station_piles= None,  # 中央仓库充电桩数量（None 表示按车辆总数）
            non_depot_station_piles_range = (1, 1),  # 非仓库充电站桩数范围

            horizon=500,  # 仿真时间范围
            task_count=52,  # 任务总数
            task_time_distribution="uniform",  # 任务释放时间分布
            task_time_mean_ratio=0.4,  # 高斯分布均值比例（uniform 下不使用）
            task_time_std_ratio=0.5,  # 高斯分布标准差比例（uniform 下不使用）
            task_weight_mean=13.0,  # 任务重量均值
            seed=20260401,  # 随机种子
        ),

        ScaleConfig(
            name="medium",  # 中等规模
            node_count=40,  # 节点数量
            vehicle_count=8,  # 车辆数量

            station_count=4,  # 充电站数量
            depot_station_piles= None,  # 中央仓库充电桩数量（None 表示按车辆总数） 
            non_depot_station_piles_range = (1, 2),  # 非仓库充电站桩数范围

            horizon=800,  # 仿真时间范围
            task_count=162,  # 任务总数
            task_time_distribution="uniform",  # 任务释放时间分布
            task_time_mean_ratio=0.4,  # 高斯分布均值比例（uniform 下不使用）
            task_time_std_ratio=0.5,  # 高斯分布标准差比例（uniform 下不使用）
            task_weight_mean=14.5,  # 任务重量均值
            seed=20260402,  # 随机种子
        ),
        
        ScaleConfig(
            name="large",  # 大规模
            node_count=80,  # 节点数量
            vehicle_count=15,  # 车辆数量

            station_count=8,  # 充电站数量
            depot_station_piles= None,  # 中央仓库充电桩数量（None 表示按车辆总数） 
            non_depot_station_piles_range = (1, 3),  # 非仓库充电站桩数范围

            horizon=1300,  # 仿真时间范围
            task_count=361,  # 任务总数
            task_time_distribution="uniform",  # 任务释放时间分布（uniform / gaussian）
            task_time_mean_ratio=0.4,  # 高斯分布均值比例（uniform 下不使用）
            task_time_std_ratio=0.5,  # 高斯分布标准差比例（uniform 下不使用）
            task_weight_mean=16.0,  # 任务重量均值
            seed=19700101,  # 随机种子
        ),
    ]
