from __future__ import annotations

import heapq
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .config import SimulationConfig
from .models import Node


@dataclass
class RoadGraph:
    """路网图结构（邻接表 + 节点坐标）。"""

    nodes: Dict[int, Node]
    adjacency: Dict[int, Dict[int, float]]

    def neighbors(self, node_id: int) -> Dict[int, float]:
        return self.adjacency[node_id]

    def edge_distance(self, a: int, b: int) -> float:
        return self.adjacency[a][b]

    def edge_count(self) -> int:
        return sum(len(nbrs) for nbrs in self.adjacency.values()) // 2


class ShortestPathOracle:
    """最短路查询器，按源点缓存 Dijkstra 结果。"""

    def __init__(self, graph: RoadGraph):
        self.graph = graph
        self._dist_cache: Dict[int, Dict[int, float]] = {}
        self._prev_cache: Dict[int, Dict[int, int]] = {}

    def shortest_distance(self, source: int, target: int) -> float:
        if source == target:
            return 0.0
        self._ensure_source(source)
        return self._dist_cache[source][target]

    def shortest_path(self, source: int, target: int) -> List[int]:
        if source == target:
            return [source]
        self._ensure_source(source)
        prev = self._prev_cache[source]
        if target not in prev:
            return [source]

        path = [target]
        cursor = target
        while cursor != source:
            cursor = prev[cursor]
            path.append(cursor)
        path.reverse()
        return path

    def _ensure_source(self, source: int) -> None:
        if source in self._dist_cache:
            return

        distances: Dict[int, float] = {node_id: math.inf for node_id in self.graph.nodes}
        previous: Dict[int, int] = {}
        distances[source] = 0.0
        queue: List[Tuple[float, int]] = [(0.0, source)]

        while queue:
            dist, node = heapq.heappop(queue)
            if dist > distances[node]:
                continue
            for neighbor, weight in self.graph.neighbors(node).items():
                candidate = dist + weight
                if candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    previous[neighbor] = node
                    heapq.heappush(queue, (candidate, neighbor))

        self._dist_cache[source] = distances
        self._prev_cache[source] = previous


def generate_road_graph(
    node_count: int,
    config: SimulationConfig,
    rng: random.Random,
) -> RoadGraph:
    """
    简化版城市路网：
    1) 抖动网格点（视觉不规则）
    2) 仅连接局部相邻道路（避免难看的跨区边）
    3) 让 0 号节点位于几何中心附近并保证四通八达
    """

    nodes, cell_to_node, node_to_cell, rows, cols = _generate_grid_nodes(node_count, rng)
    adjacency: Dict[int, Dict[int, float]] = defaultdict(dict)

    # 主路：仅连上下左右，复杂度 O(n)。
    for (r, c), node_id in cell_to_node.items():
        for dr, dc in ((0, 1), (1, 0)):
            nr, nc = r + dr, c + dc
            neighbor_id = cell_to_node.get((nr, nc))
            if neighbor_id is not None:
                _add_edge(adjacency, nodes[node_id], nodes[neighbor_id], config)

    # 次路：低概率对角线（局部短连边），减弱方格感。
    for (r, c), node_id in cell_to_node.items():
        for dr, dc, prob in ((1, 1, 0.08), (1, -1, 0.06)):
            nr, nc = r + dr, c + dc
            neighbor_id = cell_to_node.get((nr, nc))
            if neighbor_id is None or rng.random() >= prob:
                continue
            if _euclidean(nodes[node_id], nodes[neighbor_id]) <= 30.0:
                _add_edge(adjacency, nodes[node_id], nodes[neighbor_id], config)

    # 极低概率跳一格连接，但只允许本地短距离，避免跨区域难看边。
    for (r, c), node_id in cell_to_node.items():
        for dr, dc in ((0, 2), (2, 0)):
            nr, nc = r + dr, c + dc
            neighbor_id = cell_to_node.get((nr, nc))
            if neighbor_id is None or rng.random() >= 0.04:
                continue
            if _euclidean(nodes[node_id], nodes[neighbor_id]) <= 34.0:
                _add_edge(adjacency, nodes[node_id], nodes[neighbor_id], config)

    # 强化中心仓库 0 号点的可达性：连接其四邻域和两步邻域。
    center_r, center_c = node_to_cell[0]
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0), (0, 2), (0, -2), (2, 0), (-2, 0)):
        nr, nc = center_r + dr, center_c + dc
        neighbor_id = cell_to_node.get((nr, nc))
        if neighbor_id is not None:
            _add_edge(adjacency, nodes[0], nodes[neighbor_id], config)

    return RoadGraph(nodes=nodes, adjacency={k: dict(v) for k, v in adjacency.items()})


def _generate_grid_nodes(
    node_count: int,
    rng: random.Random,
) -> Tuple[Dict[int, Node], Dict[Tuple[int, int], int], Dict[int, Tuple[int, int]], int, int]:
    """
    生成抖动网格点，并将 0 号节点固定在中心格点附近。
    该设计避免“后期交换坐标”导致的连边错位问题。
    """

    cols = max(3, math.ceil(math.sqrt(node_count * 1.3)))
    rows = max(3, math.ceil(node_count / cols))

    # 关键修复：
    # 采用“近似恒定网格间距 + 随规模扩展边界”的方式，
    # 避免 large 仅仅是在同样空间里塞更密的点。
    target_step = 14.0
    x_margin = 10.0
    y_margin = 10.0

    # 城市边界随网格维度增大，确保大规模图的空间尺度更大。
    x_span = target_step * max(1, cols - 1)
    y_span = target_step * max(1, rows - 1)

    x_step = x_span / max(1, cols - 1)
    y_step = y_span / max(1, rows - 1)

    center_r = rows // 2
    center_c = cols // 2

    # 只使用前 node_count 个格点，保持拓扑简单且连通。
    used_cells: List[Tuple[int, int]] = []
    for idx in range(node_count):
        used_cells.append((idx // cols, idx % cols))

    # 将中心格点分配给 0 号节点，其余节点顺序填充。
    if (center_r, center_c) not in used_cells:
        # 极端小规模兜底：若中心格点没被覆盖，用离中心最近的已用格点代替。
        center_r, center_c = min(
            used_cells,
            key=lambda rc: abs(rc[0] - center_r) + abs(rc[1] - center_c),
        )

    nodes: Dict[int, Node] = {}
    cell_to_node: Dict[Tuple[int, int], int] = {}
    node_to_cell: Dict[int, Tuple[int, int]] = {}

    next_node_id = 1
    for cell in used_cells:
        if cell == (center_r, center_c):
            node_id = 0
        else:
            node_id = next_node_id
            next_node_id += 1

        r, c = cell
        base_x = x_margin + c * x_step
        base_y = y_margin + r * y_step

        # 抖动适度放大，降低棋盘感；中心仓库抖动更小以保持居中。
        if node_id == 0:
            jitter_x = rng.uniform(-x_step * 0.08, x_step * 0.08)
            jitter_y = rng.uniform(-y_step * 0.08, y_step * 0.08)
        else:
            jitter_x = rng.uniform(-x_step * 0.30, x_step * 0.30)
            jitter_y = rng.uniform(-y_step * 0.30, y_step * 0.30)

        nodes[node_id] = Node(
            node_id=node_id,
            x=round(base_x + jitter_x, 3),
            y=round(base_y + jitter_y, 3),
        )
        cell_to_node[(r, c)] = node_id
        node_to_cell[node_id] = (r, c)

    return nodes, cell_to_node, node_to_cell, rows, cols


def _add_edge(
    adjacency: Dict[int, Dict[int, float]],
    node_a: Node,
    node_b: Node,
    config: SimulationConfig,
) -> None:
    if node_b.node_id in adjacency[node_a.node_id]:
        return

    base = _euclidean(node_a, node_b)
    mapped = config.min_edge_distance + (base / 100.0) * (config.max_edge_distance - config.min_edge_distance)
    distance = max(config.min_edge_distance, min(config.max_edge_distance, round(mapped, 1)))

    adjacency[node_a.node_id][node_b.node_id] = distance
    adjacency[node_b.node_id][node_a.node_id] = distance


def _euclidean(a: Node, b: Node) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

