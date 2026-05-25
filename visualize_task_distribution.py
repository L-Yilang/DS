from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Iterable, List


SVG_WIDTH = 960
SVG_HEIGHT = 320
PADDING_LEFT = 56
PADDING_RIGHT = 20
PADDING_TOP = 20
PADDING_BOTTOM = 36


def _scale_x(index: int, count: int) -> float:
    plot_width = SVG_WIDTH - PADDING_LEFT - PADDING_RIGHT
    if count <= 1:
        return PADDING_LEFT
    return PADDING_LEFT + plot_width * index / (count - 1)


def _scale_y(value: float, max_value: float) -> float:
    plot_height = SVG_HEIGHT - PADDING_TOP - PADDING_BOTTOM
    if max_value <= 0:
        return SVG_HEIGHT - PADDING_BOTTOM
    return PADDING_TOP + plot_height * (1.0 - value / max_value)


def _bar_chart_svg(values: List[int], title: str, x_label: str, y_label: str) -> str:
    max_value = max(values) if values else 0
    count = len(values)
    plot_width = SVG_WIDTH - PADDING_LEFT - PADDING_RIGHT
    bar_width = plot_width / max(1, count)

    bars: List[str] = []
    for index, value in enumerate(values):
        x = PADDING_LEFT + index * bar_width
        y = _scale_y(value, max_value)
        height = max(0.0, SVG_HEIGHT - PADDING_BOTTOM - y)
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(1.0, bar_width - 0.5):.2f}" '
            f'height="{height:.2f}" fill="#2f6fed" opacity="0.85"><title>tick={index}, count={value}</title></rect>'
        )

    y_ticks = _axis_y_ticks(max_value)
    tick_lines = "".join(
        f'<line x1="{PADDING_LEFT}" y1="{_scale_y(tick, max_value):.2f}" x2="{SVG_WIDTH - PADDING_RIGHT}" '
        f'y2="{_scale_y(tick, max_value):.2f}" stroke="#d9dfeb" stroke-width="1"/>'
        f'<text x="{PADDING_LEFT - 8}" y="{_scale_y(tick, max_value) + 4:.2f}" text-anchor="end" '
        f'font-size="11" fill="#445068">{tick}</text>'
        for tick in y_ticks
    )

    x_labels = ""
    if count > 0:
        x_positions = [0, count // 4, count // 2, count * 3 // 4, count - 1]
        x_positions = sorted(set(min(count - 1, pos) for pos in x_positions))
        x_labels = "".join(
            f'<text x="{PADDING_LEFT + (pos + 0.5) * bar_width:.2f}" y="{SVG_HEIGHT - 10}" text-anchor="middle" '
            f'font-size="11" fill="#445068">{pos}</text>'
            for pos in x_positions
        )

    return f"""
    <section class="panel">
      <h2>{escape(title)}</h2>
      <svg viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" role="img" aria-label="{escape(title)}">
        <rect x="0" y="0" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="#ffffff"/>
        {tick_lines}
        <line x1="{PADDING_LEFT}" y1="{SVG_HEIGHT - PADDING_BOTTOM}" x2="{SVG_WIDTH - PADDING_RIGHT}" y2="{SVG_HEIGHT - PADDING_BOTTOM}" stroke="#77839c" stroke-width="1.2"/>
        <line x1="{PADDING_LEFT}" y1="{PADDING_TOP}" x2="{PADDING_LEFT}" y2="{SVG_HEIGHT - PADDING_BOTTOM}" stroke="#77839c" stroke-width="1.2"/>
        {''.join(bars)}
        {x_labels}
        <text x="{SVG_WIDTH / 2:.2f}" y="{SVG_HEIGHT - 2}" text-anchor="middle" font-size="12" fill="#445068">{escape(x_label)}</text>
        <text x="16" y="{SVG_HEIGHT / 2:.2f}" text-anchor="middle" font-size="12" fill="#445068" transform="rotate(-90 16 {SVG_HEIGHT / 2:.2f})">{escape(y_label)}</text>
      </svg>
    </section>
    """


def _line_chart_svg(values: List[int], title: str, x_label: str, y_label: str) -> str:
    cumulative: List[int] = []
    running = 0
    for value in values:
        running += value
        cumulative.append(running)

    max_value = max(cumulative) if cumulative else 0
    points = " ".join(
        f"{_scale_x(index, len(cumulative)):.2f},{_scale_y(value, max_value):.2f}"
        for index, value in enumerate(cumulative)
    )

    y_ticks = _axis_y_ticks(max_value)
    tick_lines = "".join(
        f'<line x1="{PADDING_LEFT}" y1="{_scale_y(tick, max_value):.2f}" x2="{SVG_WIDTH - PADDING_RIGHT}" '
        f'y2="{_scale_y(tick, max_value):.2f}" stroke="#d9dfeb" stroke-width="1"/>'
        f'<text x="{PADDING_LEFT - 8}" y="{_scale_y(tick, max_value) + 4:.2f}" text-anchor="end" '
        f'font-size="11" fill="#445068">{tick}</text>'
        for tick in y_ticks
    )

    count = len(cumulative)
    x_labels = ""
    if count > 0:
        x_positions = [0, count // 4, count // 2, count * 3 // 4, count - 1]
        x_positions = sorted(set(min(count - 1, pos) for pos in x_positions))
        x_labels = "".join(
            f'<text x="{_scale_x(pos, count):.2f}" y="{SVG_HEIGHT - 10}" text-anchor="middle" '
            f'font-size="11" fill="#445068">{pos}</text>'
            for pos in x_positions
        )

    polyline = (
        f'<polyline fill="none" stroke="#e85d04" stroke-width="2.5" points="{points}"/>'
        if points
        else ""
    )

    return f"""
    <section class="panel">
      <h2>{escape(title)}</h2>
      <svg viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" role="img" aria-label="{escape(title)}">
        <rect x="0" y="0" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="#ffffff"/>
        {tick_lines}
        <line x1="{PADDING_LEFT}" y1="{SVG_HEIGHT - PADDING_BOTTOM}" x2="{SVG_WIDTH - PADDING_RIGHT}" y2="{SVG_HEIGHT - PADDING_BOTTOM}" stroke="#77839c" stroke-width="1.2"/>
        <line x1="{PADDING_LEFT}" y1="{PADDING_TOP}" x2="{PADDING_LEFT}" y2="{SVG_HEIGHT - PADDING_BOTTOM}" stroke="#77839c" stroke-width="1.2"/>
        {polyline}
        {x_labels}
        <text x="{SVG_WIDTH / 2:.2f}" y="{SVG_HEIGHT - 2}" text-anchor="middle" font-size="12" fill="#445068">{escape(x_label)}</text>
        <text x="16" y="{SVG_HEIGHT / 2:.2f}" text-anchor="middle" font-size="12" fill="#445068" transform="rotate(-90 16 {SVG_HEIGHT / 2:.2f})">{escape(y_label)}</text>
      </svg>
    </section>
    """


def _histogram_svg(histogram: List[dict], title: str, x_label: str, y_label: str) -> str:
    values = [int(bucket["count"]) for bucket in histogram]
    max_value = max(values) if values else 0
    count = len(histogram)
    plot_width = SVG_WIDTH - PADDING_LEFT - PADDING_RIGHT
    bar_width = plot_width / max(1, count)

    bars: List[str] = []
    for index, bucket in enumerate(histogram):
        value = int(bucket["count"])
        x = PADDING_LEFT + index * bar_width
        y = _scale_y(value, max_value)
        height = max(0.0, SVG_HEIGHT - PADDING_BOTTOM - y)
        label = f"{bucket['start']:.2f}-{bucket['end']:.2f}"
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(1.0, bar_width - 1.0):.2f}" '
            f'height="{height:.2f}" fill="#16a085" opacity="0.85"><title>{label}: {value}</title></rect>'
        )

    y_ticks = _axis_y_ticks(max_value)
    tick_lines = "".join(
        f'<line x1="{PADDING_LEFT}" y1="{_scale_y(tick, max_value):.2f}" x2="{SVG_WIDTH - PADDING_RIGHT}" '
        f'y2="{_scale_y(tick, max_value):.2f}" stroke="#d9dfeb" stroke-width="1"/>'
        f'<text x="{PADDING_LEFT - 8}" y="{_scale_y(tick, max_value) + 4:.2f}" text-anchor="end" '
        f'font-size="11" fill="#445068">{tick}</text>'
        for tick in y_ticks
    )

    x_labels = ""
    if histogram:
        positions = [0, count // 4, count // 2, count * 3 // 4, count - 1]
        positions = sorted(set(min(count - 1, pos) for pos in positions))
        x_labels = "".join(
            f'<text x="{PADDING_LEFT + (pos + 0.5) * bar_width:.2f}" y="{SVG_HEIGHT - 10}" text-anchor="middle" '
            f'font-size="11" fill="#445068">{histogram[pos]["start"]:.1f}</text>'
            for pos in positions
        )

    return f"""
    <section class="panel">
      <h2>{escape(title)}</h2>
      <svg viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" role="img" aria-label="{escape(title)}">
        <rect x="0" y="0" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="#ffffff"/>
        {tick_lines}
        <line x1="{PADDING_LEFT}" y1="{SVG_HEIGHT - PADDING_BOTTOM}" x2="{SVG_WIDTH - PADDING_RIGHT}" y2="{SVG_HEIGHT - PADDING_BOTTOM}" stroke="#77839c" stroke-width="1.2"/>
        <line x1="{PADDING_LEFT}" y1="{PADDING_TOP}" x2="{PADDING_LEFT}" y2="{SVG_HEIGHT - PADDING_BOTTOM}" stroke="#77839c" stroke-width="1.2"/>
        {''.join(bars)}
        {x_labels}
        <text x="{SVG_WIDTH / 2:.2f}" y="{SVG_HEIGHT - 2}" text-anchor="middle" font-size="12" fill="#445068">{escape(x_label)}</text>
        <text x="16" y="{SVG_HEIGHT / 2:.2f}" text-anchor="middle" font-size="12" fill="#445068" transform="rotate(-90 16 {SVG_HEIGHT / 2:.2f})">{escape(y_label)}</text>
      </svg>
    </section>
    """


def _axis_y_ticks(max_value: int) -> List[int]:
    if max_value <= 0:
        return [0]
    tick_count = 5
    return [round(max_value * i / tick_count) for i in range(tick_count + 1)]


def _summary_items(distribution: dict) -> str:
    stats = distribution.get("stats", {})
    time_dist = distribution.get("time_distribution", {})
    weight_dist = distribution.get("weight_distribution", {})
    items = [
        ("scale", distribution.get("scale_name")),
        ("strategy", distribution.get("strategy_name")),
        ("seed", distribution.get("seed")),
        ("task_count", distribution.get("task_count")),
        ("spawn_cutoff_tick", distribution.get("spawn_cutoff_tick")),
        ("time_type", time_dist.get("type")),
        ("time_mean_ratio", time_dist.get("mean_ratio")),
        ("time_std_ratio", time_dist.get("std_ratio")),
        ("weight_target_mean", weight_dist.get("target_mean")),
        ("release_time_mean", stats.get("release_time_mean")),
        ("weight_mean", stats.get("weight_mean")),
        ("deadline_offset_mean", stats.get("deadline_offset_mean")),
    ]
    return "".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in items
    )


def build_html(distribution: dict, source_name: str) -> str:
    release_counts = list(distribution.get("release_time_counts", []))
    histogram = list(distribution.get("weight_histogram", []))

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Task Distribution Viewer</title>
  <style>
    :root {{
      --bg: #f3f6fb;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #667085;
      --border: #d9dfeb;
    }}
    body {{
      margin: 0;
      padding: 24px;
      font-family: "Segoe UI", "PingFang SC", sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #eef4ff 0%, var(--bg) 100%);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    p {{
      margin: 0 0 18px;
      color: var(--muted);
    }}
    .panel {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
      margin-bottom: 18px;
      box-shadow: 0 8px 24px rgba(31, 41, 55, 0.06);
    }}
    .panel h2 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 8px 10px;
      text-align: left;
      font-size: 14px;
    }}
    th {{
      width: 220px;
      color: var(--muted);
      font-weight: 600;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
      background: #fff;
      border-radius: 12px;
    }}
  </style>
</head>
<body>
  <section class="panel">
    <h1>Task Distribution Viewer</h1>
    <p>source: {escape(source_name)}</p>
    <table>
      {_summary_items(distribution)}
    </table>
  </section>
  {_bar_chart_svg(release_counts, "Release Time Histogram", "tick", "task count")}
  {_line_chart_svg(release_counts, "Cumulative Released Tasks", "tick", "cumulative tasks")}
  {_histogram_svg(histogram, "Weight Histogram", "weight bucket start", "task count")}
</body>
</html>
"""


def _iter_distribution_files(path: Path) -> List[Path]:
    """展开待处理的分布 JSON 文件列表。"""

    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*_task_distribution.json"))
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize task distribution JSON as HTML.")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("outputs") / "distribution",
        help="Path to *_task_distribution.json or a directory containing them. Defaults to outputs/distribution.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output HTML path for single-file mode. Defaults to the JSON stem with .html suffix.",
    )
    args = parser.parse_args()

    input_path = args.input
    files = _iter_distribution_files(input_path)
    if not files:
        raise FileNotFoundError(f"未找到任务分布文件: {input_path}")

    if input_path.is_file():
        distribution = json.loads(input_path.read_text(encoding="utf-8"))
        output_path = args.output or input_path.with_suffix(".html")
        html = build_html(distribution, source_name=str(input_path))
        output_path.write_text(html, encoding="utf-8")
        print(f"saved: {output_path}")
        return

    for file_path in files:
        distribution = json.loads(file_path.read_text(encoding="utf-8"))
        output_path = file_path.with_suffix(".html")
        html = build_html(distribution, source_name=str(file_path))
        output_path.write_text(html, encoding="utf-8")
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
