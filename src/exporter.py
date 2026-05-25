from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping


def write_timeline_json(path: Path, timeline: list[dict]) -> None:
    """导出时间步详细日志（用于后续分析与可视化扩展）。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")


def write_replay_json(path: Path, replay_meta: Mapping[str, object], timeline: list[dict]) -> None:
    """导出回放文件：包含静态地图元数据和动态时间线。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "replay_meta": dict(replay_meta),
        "timeline": timeline,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """导出多次仿真汇总结果。"""

    rows = list(rows)
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_task_distribution_json(path: Path, distribution: Mapping[str, object]) -> None:
    """导出任务分布摘要，供离线分析与独立可视化脚本使用。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(distribution), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
