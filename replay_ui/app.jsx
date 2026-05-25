const { useEffect, useMemo, useRef, useState } = React;

const VEHICLE_COLORS = [
  "#38bdf8", "#a78bfa", "#f472b6", "#fb7185", "#f59e0b",
  "#34d399", "#22d3ee", "#60a5fa", "#c084fc", "#4ade80",
  "#f97316", "#e879f9", "#2dd4bf", "#f43f5e", "#bef264",
  "#fde047", "#14b8a6", "#818cf8", "#fda4af", "#86efac",
];

const STATUS_LABEL = {
  pending: "待分配",
  assigned: "已分配",
  in_progress: "配送中",
  completed: "已完成",
};

const VEHICLE_STATE_LABEL = {
  idle: "空闲",
  moving: "行驶",
  loading: "装货",
  unloading: "卸货",
  waiting_charge: "排队充电",
  charging: "充电中",
};

// 车辆状态配色：用于车辆状态栏的克制型语义着色。
const VEHICLE_STATE_CLASS = {
  idle: "state-idle",
  moving: "state-moving",
  loading: "state-loading",
  unloading: "state-unloading",
  waiting_charge: "state-waiting-charge",
  charging: "state-charging",
};

const ACTION_LABEL = {
  keep: "保持",
  load: "装货",
  unload: "卸货",
  charge: "充电",
};

const STATUS_PRIORITY = {
  pending: 0,
  assigned: 1,
  in_progress: 2,
  completed: 3,
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function edgeKey(a, b) {
  return a < b ? `${a}-${b}` : `${b}-${a}`;
}

function normalizeReplayPayload(raw) {
  if (!raw) return null;

  if (Array.isArray(raw)) {
    return {
      schema_version: 0,
      replay_meta: null,
      timeline: raw,
    };
  }

  if (raw && Array.isArray(raw.timeline)) {
    return {
      schema_version: raw.schema_version ?? 0,
      replay_meta: raw.replay_meta ?? null,
      timeline: raw.timeline,
    };
  }

  return null;
}

function buildMapModel(replayMeta, canvasWidth, canvasHeight) {
  if (!replayMeta || !Array.isArray(replayMeta.nodes) || !Array.isArray(replayMeta.edges)) {
    return null;
  }

  const nodes = replayMeta.nodes.slice().sort((a, b) => a.node_id - b.node_id);
  if (!nodes.length) return null;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const node of nodes) {
    minX = Math.min(minX, node.x);
    minY = Math.min(minY, node.y);
    maxX = Math.max(maxX, node.x);
    maxY = Math.max(maxY, node.y);
  }

  const padding = 58;
  const widthSpan = Math.max(1, maxX - minX);
  const heightSpan = Math.max(1, maxY - minY);
  const drawWidth = Math.max(1, canvasWidth - padding * 2);
  const drawHeight = Math.max(1, canvasHeight - padding * 2);
  const scale = Math.min(drawWidth / widthSpan, drawHeight / heightSpan);

  const nodeCanvasMap = new Map();
  for (const node of nodes) {
    nodeCanvasMap.set(node.node_id, {
      x: padding + (node.x - minX) * scale,
      y: padding + (node.y - minY) * scale,
    });
  }

  const edgeDistanceMap = new Map();
  const edges = [];
  for (const edge of replayMeta.edges) {
    const uPos = nodeCanvasMap.get(edge.u);
    const vPos = nodeCanvasMap.get(edge.v);
    if (!uPos || !vPos) continue;
    edges.push(edge);
    edgeDistanceMap.set(edgeKey(edge.u, edge.v), edge.distance);
  }

  return {
    nodes,
    edges,
    nodeCanvasMap,
    edgeDistanceMap,
    depotNode: replayMeta.depot_node,
  };
}

function getVehicleRawPosition(vehicle, mapModel) {
  const from = mapModel.nodeCanvasMap.get(vehicle.current_node);
  if (!from) return { x: 0, y: 0 };

  if (vehicle.state === "moving" && vehicle.next_node !== null && vehicle.next_node !== undefined) {
    const to = mapModel.nodeCanvasMap.get(vehicle.next_node) || from;
    const distance = mapModel.edgeDistanceMap.get(edgeKey(vehicle.current_node, vehicle.next_node)) || 1;
    const progress = clamp(1 - (vehicle.edge_remaining || 0) / Math.max(1e-6, distance), 0, 1);
    return {
      x: from.x + (to.x - from.x) * progress,
      y: from.y + (to.y - from.y) * progress,
    };
  }

  return { x: from.x, y: from.y };
}

function spreadVehiclesWithCluster(entries) {
  const buckets = new Map();
  for (const entry of entries) {
    const key = `${Math.round(entry.rawX / 17)}_${Math.round(entry.rawY / 17)}`;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(entry);
  }

  const spreadEntries = [];
  const clusters = [];

  for (const [clusterKey, group] of buckets) {
    group.sort((a, b) => a.vehicle.vehicle_id - b.vehicle.vehicle_id);

    const centerX = group.reduce((s, x) => s + x.rawX, 0) / group.length;
    const centerY = group.reduce((s, x) => s + x.rawY, 0) / group.length;

    clusters.push({
      clusterKey,
      size: group.length,
      centerX,
      centerY,
      vehicleIds: group.map((g) => g.vehicle.vehicle_id),
    });

    if (group.length === 1) {
      spreadEntries.push({
        ...group[0],
        drawX: group[0].rawX,
        drawY: group[0].rawY,
        clusterSize: 1,
        clusterKey,
        clusterLeader: true,
      });
      continue;
    }

    const radius = 7 + Math.min(12, group.length * 1.2);
    for (let i = 0; i < group.length; i += 1) {
      const angle = (Math.PI * 2 * i) / group.length;
      spreadEntries.push({
        ...group[i],
        drawX: centerX + Math.cos(angle) * radius,
        drawY: centerY + Math.sin(angle) * radius,
        clusterSize: group.length,
        clusterKey,
        clusterLeader: i === 0,
      });
    }
  }

  return { spreadEntries, clusters };
}

function formatTaskStatus(task) {
  if (task.overdue_penalized && task.status !== "completed") {
    return { text: "超时", className: "overdue" };
  }
  return {
    text: STATUS_LABEL[task.status] || task.status,
    className: task.status || "pending",
  };
}

function drawRoundedTag(ctx, x, y, text, bgColor, textColor = "#e2e8f0") {
  const paddingX = 6;
  const h = 19;
  ctx.save();
  ctx.font = "12px Segoe UI";
  const textWidth = ctx.measureText(text).width;
  const w = textWidth + paddingX * 2;

  const r = 7;
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
  ctx.fillStyle = bgColor;
  ctx.fill();

  ctx.fillStyle = textColor;
  ctx.fillText(text, x + paddingX, y + 13);
  ctx.restore();
}

function buildTaskChain(vehicle) {
  const actions = Array.isArray(vehicle.planned_actions) ? vehicle.planned_actions : [];
  const arrivals = Array.isArray(vehicle.planned_arrivals) ? vehicle.planned_arrivals : [];
  const plannedTaskIds = Array.isArray(vehicle.planned_task_ids) ? vehicle.planned_task_ids.slice() : [];
  const loadedTaskIds = Array.isArray(vehicle.loaded_task_ids) ? vehicle.loaded_task_ids.slice() : [];

  if (!actions.length || !arrivals.length) return [];

  // 任务链推断规则：
  // 1) `load` 消耗 planned_task_ids 中下一个尚未装车的任务；
  // 2) `unload` 按车上任务的先后顺序出队；
  // 3) keep / charge 仅展示节点动作，不强行绑定任务号。
  const unloadQueue = loadedTaskIds.slice();
  const loadedSet = new Set(unloadQueue);
  let loadCursor = 0;

  const chain = [];
  const chainLen = Math.min(actions.length, arrivals.length);
  for (let i = 0; i < chainLen; i += 1) {
    const rawAction = actions[i];
    const [nodeId, eta] = arrivals[i] || [];
    let taskId = null;

    if (rawAction === "load") {
      while (loadCursor < plannedTaskIds.length && loadedSet.has(plannedTaskIds[loadCursor])) {
        loadCursor += 1;
      }
      if (loadCursor < plannedTaskIds.length) {
        taskId = plannedTaskIds[loadCursor];
        unloadQueue.push(taskId);
        loadedSet.add(taskId);
        loadCursor += 1;
      }
    } else if (rawAction === "unload") {
      taskId = unloadQueue.length ? unloadQueue.shift() : (vehicle.assigned_task_id ?? null);
    }

    chain.push({
      action: rawAction,
      nodeId,
      eta,
      taskId,
    });
  }

  return chain;
}

function formatTaskChainPreview(chain) {
  if (!chain.length) return "无计划";
  const preview = [];
  for (let i = 0; i < Math.min(chain.length, 3); i += 1) {
    const step = chain[i];
    const actionText = ACTION_LABEL[step.action] || step.action;
    const taskSuffix = step.taskId !== null && step.taskId !== undefined ? `#${step.taskId}` : "";
    preview.push(`${actionText}${taskSuffix}@N${step.nodeId}(T${step.eta})`);
  }
  if (chain.length > 3) preview.push("...");
  return preview.join(" -> ");
}

function formatTaskList(taskIds) {
  if (!taskIds.length) return "无";
  return taskIds.map((taskId) => `任务${taskId}`).join("，");
}

function ReplayPlayerApp() {
  const canvasRef = useRef(null);

  const [replayPayload, setReplayPayload] = useState(null);
  const [fileName, setFileName] = useState("未加载文件");
  const [errorText, setErrorText] = useState("");

  const [isPlaying, setIsPlaying] = useState(false);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playSpeed, setPlaySpeed] = useState(4);
  const [showRoutes, setShowRoutes] = useState(true);
  const [showTaskHalo, setShowTaskHalo] = useState(true);

  const timeline = replayPayload?.timeline || [];
  const maxFrameIndex = Math.max(0, timeline.length - 1);
  const currentFrame = timeline[frameIndex] || null;
  const replayMeta = replayPayload?.replay_meta || null;

  const [canvasSize, setCanvasSize] = useState({ width: 1200, height: 760 });

  const mapModel = useMemo(
    () => buildMapModel(replayMeta, canvasSize.width, canvasSize.height),
    [replayMeta, canvasSize.width, canvasSize.height],
  );

  const vehicleColorMap = useMemo(() => {
    const map = new Map();
    const firstFrame = timeline[0];
    if (!firstFrame || !Array.isArray(firstFrame.vehicles)) return map;

    firstFrame.vehicles
      .slice()
      .sort((a, b) => a.vehicle_id - b.vehicle_id)
      .forEach((v, index) => {
        map.set(v.vehicle_id, VEHICLE_COLORS[index % VEHICLE_COLORS.length]);
      });
    return map;
  }, [timeline]);

  const vehicleRows = useMemo(() => {
    if (!currentFrame || !Array.isArray(currentFrame.vehicles)) return [];
    return currentFrame.vehicles
      .slice()
      .sort((a, b) => a.vehicle_id - b.vehicle_id)
      .map((vehicle) => {
        const batteryRatio = clamp(vehicle.battery / Math.max(1, vehicle.battery_capacity), 0, 1);
        const taskChain = buildTaskChain(vehicle);

        // 载重条数据：兼容旧回放（可能没有 load_capacity 字段）。
        const carriedWeight = Number(vehicle.carried_weight || 0);
        const parsedCapacity = Number(vehicle.load_capacity);
        const loadCapacity = Number.isFinite(parsedCapacity) && parsedCapacity > 0 ? parsedCapacity : 0;
        const loadRatio = loadCapacity > 0 ? clamp(carriedWeight / loadCapacity, 0, 1) : 0;
        const loadedTaskIds = Array.isArray(vehicle.loaded_task_ids) ? vehicle.loaded_task_ids.slice() : [];
        const plannedTaskIds = Array.isArray(vehicle.planned_task_ids) ? vehicle.planned_task_ids.slice() : [];
        const acceptedTaskIds = [];
        const seenTaskIds = new Set();
        for (const taskId of [vehicle.assigned_task_id, ...loadedTaskIds, ...plannedTaskIds]) {
          if (taskId === null || taskId === undefined || seenTaskIds.has(taskId)) continue;
          seenTaskIds.add(taskId);
          acceptedTaskIds.push(taskId);
        }
        const currentTaskText = vehicle.assigned_task_id ? `任务${vehicle.assigned_task_id}` : "无";
        const acceptedTaskText = formatTaskList(acceptedTaskIds);

        return {
          ...vehicle,
          batteryRatio,
          batteryPct: (batteryRatio * 100).toFixed(1),
          taskChain,
          planPreview: formatTaskChainPreview(taskChain),
          stateClass: VEHICLE_STATE_CLASS[vehicle.state] || "state-idle",
          carriedWeight,
          loadCapacity,
          acceptedTaskIds,
          acceptedTaskText,
          currentTaskText,
          loadedTaskIds,
          loadRatio,
          loadPct: (loadRatio * 100).toFixed(1),
        };
      });
  }, [currentFrame]);

  useEffect(() => {
    if (!isPlaying || timeline.length <= 1) return undefined;

    const intervalMs = Math.max(20, Math.floor(1000 / playSpeed));
    const timer = window.setInterval(() => {
      setFrameIndex((prev) => {
        if (prev >= maxFrameIndex) {
          setIsPlaying(false);
          return maxFrameIndex;
        }
        return prev + 1;
      });
    }, intervalMs);

    return () => window.clearInterval(timer);
  }, [isPlaying, playSpeed, timeline.length, maxFrameIndex]);

  useEffect(() => {
    if (frameIndex > maxFrameIndex) setFrameIndex(maxFrameIndex);
  }, [frameIndex, maxFrameIndex]);

  useEffect(() => {
    const onResize = () => {
      const panel = document.querySelector(".canvas-wrap");
      if (!panel) return;
      const rect = panel.getBoundingClientRect();
      setCanvasSize({
        width: Math.max(680, Math.floor(rect.width)),
        height: Math.max(420, Math.floor(rect.height)),
      });
    };

    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !currentFrame || !mapModel) return;

    const dpr = window.devicePixelRatio || 1;
    const width = Math.floor(canvasSize.width);
    const height = Math.floor(canvasSize.height);

    if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    ctx.fillStyle = "#0b1322";
    ctx.fillRect(0, 0, width, height);

    const gradient = ctx.createRadialGradient(width * 0.2, height * 0.18, 10, width * 0.2, height * 0.18, width * 0.45);
    gradient.addColorStop(0, "rgba(56, 189, 248, 0.12)");
    gradient.addColorStop(1, "rgba(56, 189, 248, 0)");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "rgba(71, 85, 105, 0.75)";
    ctx.lineWidth = 1.3;
    for (const edge of mapModel.edges) {
      const uPos = mapModel.nodeCanvasMap.get(edge.u);
      const vPos = mapModel.nodeCanvasMap.get(edge.v);
      if (!uPos || !vPos) continue;
      ctx.beginPath();
      ctx.moveTo(uPos.x, uPos.y);
      ctx.lineTo(vPos.x, vPos.y);
      ctx.stroke();
    }

    const vehicleEntries = (currentFrame.vehicles || []).map((vehicle) => {
      const pos = getVehicleRawPosition(vehicle, mapModel);
      return { vehicle, rawX: pos.x, rawY: pos.y };
    });

    const { spreadEntries, clusters } = spreadVehiclesWithCluster(vehicleEntries);

    if (showRoutes) {
      for (const item of vehicleEntries) {
        const vehicle = item.vehicle;
        const routeNodes = [];
        if (vehicle.next_node !== null && vehicle.next_node !== undefined) routeNodes.push(vehicle.next_node);
        if (Array.isArray(vehicle.route)) routeNodes.push(...vehicle.route);
        if (!routeNodes.length) continue;

        const color = vehicleColorMap.get(vehicle.vehicle_id) || "#e2e8f0";
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.85;

        ctx.beginPath();
        ctx.moveTo(item.rawX, item.rawY);
        for (const nodeId of routeNodes) {
          const p = mapModel.nodeCanvasMap.get(nodeId);
          if (!p) continue;
          ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    if (showTaskHalo && Array.isArray(currentFrame.tasks)) {
      for (const task of currentFrame.tasks) {
        if (!["pending", "assigned", "in_progress"].includes(task.status)) continue;
        const node = mapModel.nodeCanvasMap.get(task.destination_node);
        if (!node) continue;

        const pulse = 9 + Math.sin((currentFrame.tick + task.task_id) * 0.12) * 2.3;
        let haloColor = "rgba(250, 204, 21, 0.24)";
        if (task.status === "assigned") haloColor = "rgba(125, 211, 252, 0.24)";
        if (task.status === "in_progress") haloColor = "rgba(52, 211, 153, 0.24)";
        if (task.overdue_penalized && task.status !== "completed") {
          haloColor = "rgba(239, 68, 68, 0.32)";
        }

        ctx.beginPath();
        ctx.arc(node.x, node.y, pulse, 0, Math.PI * 2);
        ctx.fillStyle = haloColor;
        ctx.fill();
      }
    }

    for (const node of mapModel.nodes) {
      const p = mapModel.nodeCanvasMap.get(node.node_id);
      if (!p) continue;

      if (node.type === "warehouse") {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 9.4, 0, Math.PI * 2);
        ctx.fillStyle = "#3b82f6";
        ctx.fill();
        ctx.strokeStyle = "#dbeafe";
        ctx.lineWidth = 2;
        ctx.stroke();
      } else if (node.type === "station") {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 7.5, 0, Math.PI * 2);
        ctx.fillStyle = "#10b981";
        ctx.fill();

        ctx.beginPath();
        ctx.arc(p.x, p.y, 10.8, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(16, 185, 129, 0.35)";
        ctx.lineWidth = 1.4;
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3.8, 0, Math.PI * 2);
        ctx.fillStyle = "#64748b";
        ctx.fill();
      }
    }

    for (const station of currentFrame.stations || []) {
      const p = mapModel.nodeCanvasMap.get(station.node_id);
      if (!p) continue;
      const queueLen = (station.queue_vehicle_ids || []).length;
      const chargingLen = (station.charging_vehicle_ids || []).length;
      if (queueLen === 0 && chargingLen === 0) continue;

      const text = `S${station.station_id} 充:${chargingLen}/${station.piles} 队:${queueLen}`;
      drawRoundedTag(ctx, p.x + 10, p.y - 24, text, "rgba(6, 78, 59, 0.86)");
    }

    for (const cluster of clusters) {
      if (cluster.size <= 1) continue;

      // 保留簇轮廓提示，但不再显示“x车重叠”悬浮信息。
      ctx.beginPath();
      ctx.arc(cluster.centerX, cluster.centerY, 12 + Math.min(6, cluster.size), 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(226, 232, 240, 0.35)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    for (const item of spreadEntries) {
      const vehicle = item.vehicle;
      const color = vehicleColorMap.get(vehicle.vehicle_id) || "#f8fafc";
      const batteryRatio = clamp(vehicle.battery / Math.max(1, vehicle.battery_capacity), 0, 1);

      ctx.beginPath();
      ctx.arc(item.drawX, item.drawY, 6.7, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = batteryRatio < 0.2 ? "#ef4444" : "#0f172a";
      ctx.lineWidth = batteryRatio < 0.2 ? 2.3 : 1.1;
      ctx.stroke();

      ctx.fillStyle = "#0f172a";
      ctx.font = "bold 9px Segoe UI";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(vehicle.vehicle_id), item.drawX, item.drawY);

      const barW = 22;
      const barH = 4;
      ctx.fillStyle = "rgba(239, 68, 68, 0.7)";
      ctx.fillRect(item.drawX - barW / 2, item.drawY - 12, barW, barH);
      ctx.fillStyle = batteryRatio < 0.25 ? "#f43f5e" : "#22c55e";
      ctx.fillRect(item.drawX - barW / 2, item.drawY - 12, barW * batteryRatio, barH);

      if ((vehicle.carried_weight || 0) > 0) {
        ctx.font = "10px Segoe UI";
        ctx.fillStyle = "#fef9c3";
        ctx.textAlign = "left";
        ctx.fillText(`L:${Math.round(vehicle.carried_weight)}`, item.drawX + 8, item.drawY + 1);
      }
    }

    drawRoundedTag(
      ctx,
      12,
      10,
      `Tick ${currentFrame.tick} | 得分 ${Math.round(currentFrame.score)}`,
      "rgba(30, 41, 59, 0.84)",
      "#bae6fd",
    );
  }, [currentFrame, mapModel, canvasSize.width, canvasSize.height, showRoutes, showTaskHalo, vehicleColorMap]);

  const taskStats = currentFrame?.task_stats || {
    total: 0,
    pending: 0,
    assigned: 0,
    in_progress: 0,
    completed: 0,
    overdue: 0,
    timeout_rate: 0,
  };

  const sortedTasks = useMemo(() => {
    if (!currentFrame || !Array.isArray(currentFrame.tasks)) return [];

    const tick = currentFrame.tick;

    const bucket = (task) => {
      // 任务仪表盘排序规则：
      // 1) 未超时任务优先展示
      // 2) 超时未完成任务后置
      // 3) 已完成任务最后展示
      if (task.status === "completed") return 2;
      if (task.overdue_penalized) return 1;
      return 0;
    };

    return currentFrame.tasks
      .slice()
      .sort((a, b) => {
        const ba = bucket(a);
        const bb = bucket(b);
        if (ba !== bb) return ba - bb;

        const pa = STATUS_PRIORITY[a.status] ?? 99;
        const pb = STATUS_PRIORITY[b.status] ?? 99;
        if (pa !== pb) return pa - pb;

        const urgentA = (a.deadline ?? tick) - tick;
        const urgentB = (b.deadline ?? tick) - tick;
        if (urgentA !== urgentB) return urgentA - urgentB;

        return a.task_id - b.task_id;
      });
  }, [currentFrame]);

  const stationRows = currentFrame?.stations || [];


  async function handleLoadFile(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;

    try {
      const rawText = await file.text();
      const rawData = JSON.parse(rawText);
      const payload = normalizeReplayPayload(rawData);

      if (!payload) throw new Error("文件格式不正确，请选择 *_replay.json 文件。");
      if (!payload.replay_meta || !payload.replay_meta.nodes || !payload.replay_meta.edges) {
        throw new Error("该文件缺少回放地图信息，请用 main.py 重新导出 *_replay.json。");
      }

      setReplayPayload(payload);
      setFileName(file.name);
      setErrorText("");
      setFrameIndex(0);
      setIsPlaying(false);
    } catch (error) {
      setErrorText(error.message || "读取文件失败，请检查 JSON 内容。");
    }
  }

  async function handleLoadDefault() {
    try {
      const response = await fetch("../outputs/small_nearest_task_replay.json");
      if (!response.ok) throw new Error("默认样例读取失败，请改用本地上传。");
      const payload = normalizeReplayPayload(await response.json());
      if (!payload || !payload.replay_meta) throw new Error("默认样例格式不完整。");

      setReplayPayload(payload);
      setFileName("small_nearest_task_replay.json");
      setErrorText("");
      setFrameIndex(0);
      setIsPlaying(false);
    } catch (error) {
      setErrorText(`加载默认样例失败：${error.message}`);
    }
  }

  function goPrevFrame() {
    setIsPlaying(false);
    setFrameIndex((prev) => Math.max(0, prev - 1));
  }

  function goNextFrame() {
    setIsPlaying(false);
    setFrameIndex((prev) => Math.min(maxFrameIndex, prev + 1));
  }

  const completionRate = taskStats.total > 0 ? (taskStats.completed / taskStats.total) * 100 : 0;

  return (
    <div className="app">
      <aside className="left-panel">
        <div className="header compact">
          <h1>新能源车队回放系统</h1>
          <p>专业回放视图 | 同屏展示地图、任务、车辆、充电状态</p>
        </div>

        <section className="card dense">
          <h3>数据源</h3>
          <div className="control-row">
            <input type="file" accept=".json" onChange={handleLoadFile} />
          </div>
          <div className="control-row">
            <button className="secondary" onClick={handleLoadDefault}>加载默认样例</button>
          </div>
          <div className="helper-text">当前文件：{fileName}</div>
          {errorText ? <div className="error-text">{errorText}</div> : null}
        </section>

        <section className="card dense">
          <h3>核心指标</h3>
          <div className="metrics-grid compact-metrics">
            <div className="metric-box"><div className="metric-label">总收益</div><div className="metric-value score">{Math.round(currentFrame?.score || 0)}</div></div>
            <div className="metric-box"><div className="metric-label">Tick</div><div className="metric-value time">{currentFrame?.tick ?? 0}</div></div>
            <div className="metric-box"><div className="metric-label">已完成</div><div className="metric-value ok">{taskStats.completed}</div></div>
            <div className="metric-box"><div className="metric-label">超时</div><div className="metric-value bad">{taskStats.overdue}</div></div>
            <div className="metric-box"><div className="metric-label">超时率</div><div className="metric-value bad">{((taskStats.timeout_rate || 0) * 100).toFixed(1)}%</div></div>
            <div className="metric-box"><div className="metric-label">完成率</div><div className="metric-value ok">{completionRate.toFixed(1)}%</div></div>
          </div>
        </section>

        <section className="card dense">
          <h3>回放控制</h3>
          <div className="control-row control-3">
            <button onClick={() => setIsPlaying((s) => !s)} disabled={!timeline.length}>{isPlaying ? "暂停" : "播放"}</button>
            <button className="secondary" onClick={goPrevFrame} disabled={!timeline.length}>上一帧</button>
            <button className="secondary" onClick={goNextFrame} disabled={!timeline.length}>下一帧</button>
          </div>

          <div className="control-row">
            <span className="inline-label">速度</span>
            <select value={playSpeed} onChange={(e) => setPlaySpeed(Number(e.target.value))}>
              <option value={1}>1x</option>
              <option value={2}>2x</option>
              <option value={4}>4x</option>
              <option value={8}>8x</option>
              <option value={12}>12x</option>
              <option value={16}>16x</option>
              <option value={32}>32x</option>
            </select>
          </div>

          <div className="control-row">
            <input
              type="range"
              min="0"
              max={maxFrameIndex}
              step="1"
              value={frameIndex}
              disabled={!timeline.length}
              onChange={(e) => {
                setIsPlaying(false);
                setFrameIndex(Number(e.target.value));
              }}
            />
          </div>
          <div className="helper-text">帧：{frameIndex} / {maxFrameIndex}</div>

          <label className="toggle"><input type="checkbox" checked={showRoutes} onChange={(e) => setShowRoutes(e.target.checked)} />显示车辆路线</label>
          <label className="toggle"><input type="checkbox" checked={showTaskHalo} onChange={(e) => setShowTaskHalo(e.target.checked)} />显示任务热点</label>

          <div className="control-row" style={{ marginTop: 6 }}>
            <button className="danger" onClick={() => { setIsPlaying(false); setFrameIndex(0); }} disabled={!timeline.length}>回到起点</button>
          </div>
        </section>

        <section className="card dense">
          <h3>充电站负载</h3>
          {stationRows.length === 0 ? (
            <div className="helper-text">暂无充电站信息</div>
          ) : (
            <div className="station-mini-list">
              {stationRows.map((station) => {
                const queueLen = (station.queue_vehicle_ids || []).length;
                const chargingLen = (station.charging_vehicle_ids || []).length;
                const load = (chargingLen + queueLen) / Math.max(1, station.piles);
                return (
                  <div className="station-row mini" key={`station-${station.station_id}`}>
                    <div className="station-header">
                      <span>S{station.station_id} / N{station.node_id}</span>
                      <span>压 {station.pressure}</span>
                    </div>
                    <div className="helper-text">充 {chargingLen}/{station.piles} | 队 {queueLen} | 速 {station.charge_rate}</div>
                    <div className="bar-bg"><div className="bar-fill" style={{ width: `${Math.min(100, load * 100)}%` }}></div></div>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      </aside>

      <main className="right-panel compact">
        <section className="canvas-wrap">
          <canvas ref={canvasRef} className="map-canvas"></canvas>
          <div className="map-legend">
            <span className="legend-chip"><span className="legend-dot" style={{ background: "#3b82f6" }}></span>中央仓库</span>
            <span className="legend-chip"><span className="legend-dot" style={{ background: "#10b981" }}></span>充电站</span>
            <span className="legend-chip"><span className="legend-dot" style={{ background: "#64748b" }}></span>道路节点</span>
            <span className="legend-chip"><span className="legend-dot" style={{ background: "#facc15" }}></span>任务热点</span>
          </div>
        </section>

        <section className="bottom-grid triple">
          <div className="table-wrap">
            <h3>任务仪表盘（当前帧）</h3>
            <div className="scroll-area">
              <table className="task-table">
                <thead>
                  <tr>
                    <th>任务</th><th>状态</th><th>目的地</th><th>重量</th><th>剩余时限</th><th>车辆</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedTasks.slice(0, 120).map((task) => {
                    const statusView = formatTaskStatus(task);
                    const remain = (task.deadline ?? 0) - (currentFrame?.tick ?? 0);
                    return (
                      <tr key={`task-${task.task_id}`}>
                        <td>#{task.task_id}</td>
                        <td><span className={`badge ${statusView.className}`}>{statusView.text}</span></td>
                        <td>N{task.destination_node}</td>
                        <td>{Number(task.weight).toFixed(1)}</td>
                        <td>{remain}</td>
                        <td>{task.assigned_vehicle_id ?? "-"}</td>
                      </tr>
                    );
                  })}
                  {sortedTasks.length === 0 ? (
                    <tr><td colSpan={6} className="empty-cell">暂无任务数据</td></tr>
                  ) : null}
                </tbody>
              </table>
            </div>
          </div>

          <div className="table-wrap">
            <h3>车辆状态栏（当前帧）</h3>
            <div className="scroll-area vehicle-scroll">
              {vehicleRows.map((vehicle) => {
                const color = vehicleColorMap.get(vehicle.vehicle_id) || "#e2e8f0";
                return (
                  <div className={`vehicle-row ${vehicle.stateClass}`} key={`vehicle-${vehicle.vehicle_id}`}>
                    <div className="vehicle-row-top">
                      <div className="vehicle-id-wrap">
                        <span className="vehicle-dot" style={{ background: color }}></span>
                        <strong>车辆#{vehicle.vehicle_id}</strong>
                      </div>
                      <span className={`vehicle-state-pill ${vehicle.stateClass}`}>{VEHICLE_STATE_LABEL[vehicle.state] || vehicle.state}</span>
                    </div>

                    <div className="vehicle-battery-line">
                      <span>电量 {vehicle.batteryPct}%</span>
                      <span>{vehicle.battery.toFixed(1)} / {vehicle.battery_capacity.toFixed(1)}</span>
                    </div>
                    <div className="battery-bar-bg">
                      <div className={`battery-bar-fill ${vehicle.batteryRatio < 0.2 ? "low" : ""}`} style={{ width: `${vehicle.batteryPct}%` }}></div>
                    </div>

                    <div className="vehicle-meta">
                      <span>位置 N{vehicle.current_node} | 已行驶 {vehicle.distance_travelled.toFixed(1)}</span>
                      <span className="inline-load-wrap">
                        载重 {vehicle.carriedWeight.toFixed(1)}{vehicle.loadCapacity > 0 ? ` / ${vehicle.loadCapacity.toFixed(1)}` : ""}
                        {vehicle.loadCapacity > 0 ? (
                          <span className="mini-load-bar" aria-label={`载重 ${vehicle.loadPct}%`}>
                            <span className="mini-load-fill" style={{ width: `${vehicle.loadPct}%` }}></span>
                          </span>
                        ) : null}
                      </span>
                    </div>
                    <div className="vehicle-plan vehicle-task-line">
                      <span className="vehicle-task-current">当前：{vehicle.currentTaskText}</span>
                      <span className="vehicle-task-all">已接：{vehicle.acceptedTaskText}</span>
                    </div>
                    <div className="vehicle-plan">计划摘要：{vehicle.planPreview}</div>
                    <div className="vehicle-plan-chain">
                      {vehicle.taskChain.length ? (
                        vehicle.taskChain.slice(0, 6).map((step, idx) => {
                          const actionText = ACTION_LABEL[step.action] || step.action;
                          const taskText = step.taskId !== null && step.taskId !== undefined ? `任务#${step.taskId}` : "无任务";
                          return (
                            <span key={`v${vehicle.vehicle_id}-step-${idx}`} className={`plan-chip action-${step.action}`}>
                              N{step.nodeId} {actionText} | {taskText} | T{step.eta}
                            </span>
                          );
                        })
                      ) : (
                        <span className="plan-chip empty">暂无计划链</span>
                      )}
                    </div>
                  </div>
                );
              })}
              {!vehicleRows.length ? <div className="empty-cell">暂无车辆数据</div> : null}
            </div>
          </div>

          <div className="events-wrap">
            <h3>事件日志（当前帧）</h3>
            <div className="scroll-area">
              <ul className="events-list">
                {(currentFrame?.events || []).map((evt, index) => (
                  <li key={`evt-${index}`}>{evt}</li>
                ))}
                {(currentFrame?.events || []).length === 0 ? (
                  <li className="empty-li">该时间步无事件</li>
                ) : null}
              </ul>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<ReplayPlayerApp />);



















