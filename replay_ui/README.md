# 回放 UI 使用说明

该目录是独立的前端回放层，不会修改仿真调度逻辑。

## 1. 先生成回放数据

在项目根目录运行：

```bash
python main.py --scale small --strategy nearest_task
```

会在 `outputs/` 下生成：

- `*_timeline.json`：原始时间线
- `*_replay.json`：回放专用文件（含地图元数据 + 时间线）

## 2. 打开回放 UI

启动本地静态服务后访问：

```bash
python -m http.server 8000
```

然后打开：`http://localhost:8000/replay_ui/index.html`


## 3. 支持的可视化增强

- 车辆路线按车辆颜色显示
- 车辆重叠位置自动分离显示
- 充电站排队与充电负载面板
- 任务仪表盘（状态、目的节点、重量、剩余时限、分配车辆）
- 事件日志逐帧查看
- 播放/暂停/逐帧/速度调节/拖动进度条

## 4. 注意

