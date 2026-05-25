from __future__ import annotations

from .models import Task


def completion_score(
    task: Task,
    completion_time: int,
    base_reward: float,
    late_penalty_factor: float,
) -> float:
    """任务完成得分：越早完成奖励越高，晚完成会有额外扣分。"""

    allowed = max(1, task.deadline - task.release_time)
    elapsed = max(0, completion_time - task.release_time)

    if completion_time <= task.deadline:
        punctual_ratio = max(0.0, 1.0 - elapsed / allowed)
        return base_reward * (0.4 + 0.6 * punctual_ratio)

    delay = completion_time - task.deadline
    return -late_penalty_factor * (0.5 + delay / allowed)


def distance_penalty(total_distance: float, factor: float) -> float:
    """总路程减益。"""

    return -factor * total_distance
