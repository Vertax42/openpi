# RTC 实现详细分析：eval_with_real_robot.py

## 架构概览

该脚本使用**双线程架构**实现实时动作生成和执行：

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Thread                               │
│  - 初始化策略和机器人                                         │
│  - 创建 ActionQueue                                          │
│  - 启动两个工作线程                                           │
│  - 监控运行时间                                               │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
    ┌───────▼────────┐            ┌────────▼────────┐
    │ get_actions    │            │  actor_control  │
    │   Thread       │            │     Thread      │
    │                │            │                 │
    │ (Server-like)  │            │  (Client-like)  │
    └───────┬────────┘            └────────┬───────┘
            │                               │
            └───────────┬───────────────────┘
                        │
                ┌───────▼────────┐
                │  ActionQueue   │
                │  (Thread-safe) │
                └────────────────┘
```

## 核心组件

### 1. ActionQueue - 线程安全队列

**职责**：
- 存储**原始动作**（`original_queue`）：用于 RTC 计算 `prev_chunk_left_over`
- 存储**后处理动作**（`queue`）：用于机器人执行
- 跟踪当前消费索引（`last_index`）

**关键方法**：
- `get()`: 获取下一个动作（增加 `last_index`）
- `get_left_over()`: 获取未执行的原始动作（用于 RTC）
- `merge()`: 合并新动作到队列（RTC 模式会替换队列）

### 2. get_actions 线程（动作生成器）

**职责**：类似于 server，负责生成新的动作块

**工作流程**：

```python
while not shutdown:
    if action_queue.qsize() <= threshold:
        # 1. 记录推理开始时的状态
        action_index_before_inference = action_queue.get_action_index()
        prev_actions = action_queue.get_left_over()  # 获取未执行的动作
        
        # 2. 计算推理延迟
        inference_latency = latency_tracker.max()  # 历史最大延迟
        inference_delay = ceil(inference_latency / time_per_chunk)
        
        # 3. 获取当前观察
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        
        # 4. 调用策略生成动作（传入 RTC 参数）
        actions = policy.predict_action_chunk(
            obs_processed,
            inference_delay=inference_delay,      # 推理延迟步数
            prev_chunk_left_over=prev_actions,    # 上一个块未执行的动作
        )
        
        # 5. 保存原始动作（用于下次 RTC 计算）
        original_actions = actions.squeeze(0).clone()
        
        # 6. 后处理动作
        postprocessed_actions = postprocessor(actions)
        
        # 7. 测量实际推理时间
        new_latency = time.perf_counter() - current_time
        new_delay = ceil(new_latency / time_per_chunk)
        latency_tracker.add(new_latency)
        
        # 8. 合并到队列
        action_queue.merge(
            original_actions,           # 原始动作（用于 RTC）
            postprocessed_actions,      # 后处理动作（用于执行）
            new_delay,                  # 实际延迟
            action_index_before_inference,
        )
```

**关键点**：

1. **触发条件**：
   ```python
   get_actions_threshold = cfg.action_queue_size_to_get_new_actions
   if not cfg.rtc.enabled:
       get_actions_threshold = 0  # 非 RTC 模式：队列为空就生成
   
   if action_queue.qsize() <= get_actions_threshold:
       # 生成新动作
   ```

2. **inference_delay 计算**：
   - 使用 `LatencyTracker.max()` 获取历史最大延迟
   - 转换为时间步：`inference_delay = ceil(latency / time_per_chunk)`
   - 这个值告诉 RTC：在推理期间，机器人已经执行了多少步

3. **prev_chunk_left_over 获取**：
   ```python
   prev_actions = action_queue.get_left_over()
   # 返回 original_queue[last_index:]
   # 即：当前队列中还未执行完的原始动作
   ```

### 3. actor_control 线程（动作执行器）

**职责**：类似于 client，负责执行动作

**工作流程**：

```python
while not shutdown:
    start_time = time.perf_counter()
    
    # 1. 从队列获取动作
    action = action_queue.get()  # 自动增加 last_index
    
    if action is not None:
        # 2. 转换为机器人格式
        action_dict = {
            key: action[i].item()
            for i, key in enumerate(robot.action_features())
        }
        
        # 3. 后处理
        action_processed = robot_action_processor((action_dict, None))
        
        # 4. 发送到机器人
        robot.send_action(action_processed)
    
    # 5. 控制执行频率（fps）
    dt_s = time.perf_counter() - start_time
    time.sleep(max(0, (action_interval - dt_s) - 0.001))
```

**关键点**：
- 以固定频率（`cfg.fps`）执行动作
- 每次 `get()` 会自动增加 `last_index`，标记已消费的动作

## RTC 关键机制

### 1. 动作队列替换策略（RTC 模式）

当 RTC 启用时，`ActionQueue.merge()` 会**替换**整个队列：

```python
def _replace_actions_queue(self, original_actions, processed_actions, real_delay):
    # 丢弃前 real_delay 个动作（推理期间已执行）
    self.original_queue = original_actions[real_delay:].clone()
    self.queue = processed_actions[real_delay:].clone()
    self.last_index = 0  # 重置索引
```

**为什么这样做？**
- 推理需要时间（`real_delay` 步）
- 在这段时间内，机器人已经执行了 `real_delay` 个动作
- 新生成的动作块应该从 `real_delay` 之后开始使用

### 2. prev_chunk_left_over 的传递链

```
actor_control 执行动作
    ↓
ActionQueue.last_index 增加
    ↓
get_actions 调用 get_left_over()
    ↓
返回 original_queue[last_index:]  # 未执行的动作
    ↓
传递给 policy.predict_action_chunk(prev_chunk_left_over=...)
    ↓
RTC 使用这些动作计算 guidance correction
```

**示例**：
- 队列有 30 个动作，已执行 10 个（`last_index=10`）
- `get_left_over()` 返回后 20 个动作
- 这 20 个动作作为 `prev_chunk_left_over` 传入下一次推理
- RTC 使用它们来平滑新旧动作块之间的过渡

### 3. inference_delay 的双重作用

1. **传递给 RTC**：
   ```python
   actions = policy.predict_action_chunk(
       ...,
       inference_delay=inference_delay,  # 告诉 RTC 推理延迟
   )
   ```
   - RTC 使用它来计算 prefix weights
   - 权重决定哪些时间步需要 guidance

2. **用于队列替换**：
   ```python
   action_queue.merge(..., real_delay=new_delay)
   ```
   - `real_delay` 是实际测量的延迟
   - 用于丢弃已执行的动作

### 4. 时间同步机制

**LatencyTracker**：
- 跟踪历史推理延迟
- 使用 `max()` 获取最坏情况延迟
- 用于预测下一次推理的延迟

**时间步转换**：
```python
time_per_chunk = 1.0 / cfg.fps  # 每个动作的时间
inference_delay = ceil(inference_latency / time_per_chunk)  # 转换为步数
```

**验证机制**：
```python
def _check_delays(self, real_delay, action_index_before_inference):
    indexes_diff = self.last_index - action_index_before_inference
    if indexes_diff != real_delay:
        logger.warning("Indexes diff != real delay")
```
- 验证：实际执行的动作数 == 计算的延迟步数

## 配置参数

### action_queue_size_to_get_new_actions

**作用**：控制何时触发新动作生成

```python
if action_queue.qsize() <= get_actions_threshold:
    # 生成新动作
```

**设置要求**：
```python
if (cfg.action_queue_size_to_get_new_actions 
    < cfg.rtc.execution_horizon + new_delay):
    logger.warning("Too small!")
```

**原因**：
- 需要保证队列中有足够的动作缓冲
- 必须大于 `execution_horizon + inference_delay`
- 否则可能在动作生成完成前队列就空了

## 与 openpi_client 的对比

| 特性 | eval_with_real_robot.py | openpi_client |
|------|------------------------|---------------|
| 架构 | 单机双线程 | 网络 client-server |
| 通信 | ActionQueue (内存) | WebSocket |
| 同步 | 线程锁 | 网络协议 |
| 延迟测量 | 本地时间戳 | 网络往返时间 |
| 队列管理 | ActionQueue.merge() | ActionChunkBroker |

## 关键代码路径

### RTC 参数传递链

```
get_actions 线程
    ↓
action_queue.get_left_over()  # 获取未执行动作
    ↓
policy.predict_action_chunk(
    inference_delay=inference_delay,
    prev_chunk_left_over=prev_actions,
)
    ↓
PI05Pytorch.sample_actions(
    inference_delay=...,
    prev_chunk_left_over=...,
)
    ↓
RTCProcessor.denoise_step(
    inference_delay=...,
    prev_chunk_left_over=...,
)
    ↓
计算 guidance correction
```

### 动作执行链

```
actor_control 线程
    ↓
action_queue.get()  # 消费动作，last_index++
    ↓
robot.send_action(action_processed)
    ↓
（下次推理时）
    ↓
action_queue.get_left_over()  # 返回剩余动作
```

## 总结

这个实现的核心思想：

1. **双缓冲机制**：原始动作用于 RTC，后处理动作用于执行
2. **队列替换**：RTC 模式下，新动作块替换旧队列（考虑延迟）
3. **延迟补偿**：通过 `inference_delay` 和 `real_delay` 补偿推理时间
4. **平滑过渡**：使用 `prev_chunk_left_over` 实现动作块之间的平滑连接
5. **线程安全**：使用锁保护共享的 ActionQueue

这种设计确保了：
- 实时性：动作生成和执行并行进行
- 连续性：RTC 保证动作块之间的平滑过渡
- 准确性：延迟测量和补偿机制保证时间同步

