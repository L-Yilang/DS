"""固定 RNG 干净对比：默认4算子 vs 默认5算子(含regret_k 3x权重)。"""
import random
from dataclasses import replace
from typing import Callable, List, Tuple
from src.config import SimulationConfig, default_scales
from src.strategies import EnergyAwareALNSStrategy
from src.world import WorldManager


class ALNS_V4(EnergyAwareALNSStrategy):
    """原有4修复算子。"""
    def _build_repair_ops(self) -> List[Tuple[str, Callable]]:
        return [
            ("greedy", self._repair_greedy),
            ("urgent", self._repair_urgent),
            ("weight", self._repair_weight),
            ("energy_safe", self._repair_energy_safe),
        ]


class ALNS_V5(EnergyAwareALNSStrategy):
    """含 regret_k，初始权重3x。"""
    def _build_repair_ops(self) -> List[Tuple[str, Callable]]:
        return [
            ("greedy", self._repair_greedy),
            ("urgent", self._repair_urgent),
            ("weight", self._repair_weight),
            ("energy_safe", self._repair_energy_safe),
            ("regret_k", self._repair_regret_k),
        ]


SEEDS = {
    "small":  [1393869867, 542157503, 1044925344, 92340001, 177964726],
    "medium": [354787187, 1866801784, 736577488, 108402446, 550665828],
    "large":  [2118333588, 362326816, 1167052183, 2012485100, 1287757272],
}

sim_config = SimulationConfig()
scales = {s.name: s for s in default_scales()}

for scale_name, seeds in SEEDS.items():
    totals = {"v4": 0.0, "v5": 0.0}
    print(f"\n{'='*85}")
    print(f"  Scale: {scale_name}")
    print(f"{'='*85}")
    print(f"{'Seed':>12}  {'V4(4ops)':>12}  {'V5(+regret)':>14}  {'Diff':>10}  {'Winner':>8}")
    print("-" * 85)
    for seed in seeds:
        scale = replace(scales[scale_name], seed=seed)
        rng_state = random.Random(seed)

        # V4: 4 original operators
        s4 = ALNS_V4()
        s4._rng = random.Random(seed)
        # 需要让 V4 也用 regret_k 的初始权重逻辑？不，V4 只有4个算子
        w4 = WorldManager(scale=scale, config=sim_config, strategy=s4)
        r4 = w4.run()

        # V5: 5 operators with regret_k boosted
        s5 = ALNS_V5()
        s5._rng = random.Random(seed)
        w5 = WorldManager(scale=scale, config=sim_config, strategy=s5)
        r5 = w5.run()

        diff = r5.total_score - r4.total_score
        winner = "V5" if diff > 0 else ("V4" if diff < 0 else "tie")
        totals["v4"] += r4.total_score
        totals["v5"] += r5.total_score

        print(
            f"{seed:>12}  {r4.total_score:>12.2f}  {r5.total_score:>14.2f}  "
            f"{diff:>+10.2f}  {winner:>8}"
        )

    print("-" * 85)
    avg_diff = (totals["v5"] - totals["v4"]) / len(seeds)
    print(f"{'AVERAGE':>12}  {totals['v4']/len(seeds):>12.2f}  {totals['v5']/len(seeds):>14.2f}  {avg_diff:>+10.2f}")
