"""批量测试指定种子下的 ALNS 策略表现。"""
from dataclasses import replace
from src.config import SimulationConfig, default_scales
from src.strategies import EnergyAwareALNSStrategy
from src.world import WorldManager

SEEDS = {
    "small":  [1393869867, 542157503, 1044925344, 92340001, 177964726],
    "medium": [354787187, 1866801784, 736577488, 108402446, 550665828],
    "large":  [2118333588, 362326816, 1167052183, 2012485100, 1287757272],
}

sim_config = SimulationConfig()
scales = {s.name: s for s in default_scales()}

for scale_name, seeds in SEEDS.items():
    print(f"\n{'='*65}")
    print(f"  Scale: {scale_name}")
    print(f"{'='*65}")
    print(f"{'Seed':>12}  {'Score':>10}  {'Complete':>8}  {'Timeout':>7}  {'Distance':>10}  {'Failed':>6}")
    print("-" * 65)
    for seed in seeds:
        scale = replace(scales[scale_name], seed=seed)
        strategy = EnergyAwareALNSStrategy()
        world = WorldManager(scale=scale, config=sim_config, strategy=strategy)
        result = world.run()
        print(
            f"{seed:>12}  {result.total_score:>10.2f}  "
            f"{result.completed_tasks}/{result.total_tasks:>6}  "
            f"{result.timeout_rate:>7.4f}  {result.total_distance:>10.2f}  "
            f"{str(result.simulation_failed):>6}"
        )
