# OpenPI Technical Guide

This document provides comprehensive technical documentation for OpenPI, covering normalization statistics, remote inference, and implementation details of key components.

---

## Table of Contents

1. [Normalization Statistics](#normalization-statistics)
2. [Remote Inference](#remote-inference)
3. [ActionQueue Implementation Analysis](#actionqueue-implementation-analysis)

---

# Normalization Statistics

Following common practice, our models normalize the proprioceptive state inputs and action targets during policy training and inference. The statistics used for normalization are computed over the training data and stored alongside the model checkpoint.

## Reloading normalization statistics

When you fine-tune one of our models on a new dataset, you need to decide whether to (A) reuse existing normalization statistics or (B) compute new statistics over your new training data. Which option is better for you depends on the similarity of your robot and task to the robot and task distribution in the pre-training dataset. Below, we list all the available pre-training normalization statistics for each model.

**If your target robot matches one of these pre-training statistics, consider reloading the same normalization statistics.** By reloading the normalization statistics, the actions in your dataset will be more "familiar" to the model, which can lead to better performance. You can reload the normalization statistics by adding an `AssetsConfig` to your training config that points to the corresponding checkpoint directory and normalization statistics ID, like below for the `Trossen` (aka ALOHA) robot statistics of the `pi0_base` checkpoint:

```python
TrainConfig(
    ...
    data=LeRobotAlohaDataConfig(
        ...
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen",
        ),
    ),
)
```

For an example of a full training config that reloads normalization statistics, see the `pi0_aloha_pen_uncap` config in the [training config file](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py).

**Note:** To successfully reload normalization statistics, it's important that your robot + dataset are following the action space definitions used in pre-training. We provide a detailed description of our action space definitions below.

**Note #2:** Whether reloading normalization statistics is beneficial depends on the similarity of your robot and task to the robot and task distribution in the pre-training dataset. We recommend to always try both, reloading and training with a fresh set of statistics computed on your new dataset (see [main README](../README.md) for instructions on how to compute new statistics), and pick the one that works better for your task.


## Provided Pre-training Normalization Statistics

Below is a list of all the pre-training normalization statistics we provide. We provide them for both, the `pi0_base` and `pi0_fast_base` models. For `pi0_base`, set the `assets_dir` to `gs://openpi-assets/checkpoints/pi0_base/assets` and for `pi0_fast_base`, set the `assets_dir` to `gs://openpi-assets/checkpoints/pi0_fast_base/assets`.

| Robot | Description | Asset ID |
|-------|-------------|----------|
| ALOHA | 6-DoF dual arm robot with parallel grippers | trossen |
| Mobile ALOHA | Mobile version of ALOHA mounted on a Slate base | trossen_mobile |
| Franka Emika (DROID) | 7-DoF arm with parallel gripper based on the DROID setup | droid |
| Franka Emika (non-DROID) | Franka FR3 arm with Robotiq 2F-85 gripper | franka |
| UR5e | 6-DoF UR5e arm with Robotiq 2F-85 gripper | ur5e |
| UR5e bi-manual | Bi-manual UR5e setup with Robotiq 2F-85 grippers | ur5e_dual |
| ARX | Bi-manual ARX-5 robot arm setup with parallel gripper | arx |
| ARX mobile | Mobile version of bi-manual ARX-5 robot arm setup mounted on a Slate base | arx_mobile |
| Fibocom mobile | Fibocom mobile robot with 2x ARX-5 arms | fibocom_mobile |


## Pi0 Model Action Space Definitions

Out of the box, both the `pi0_base` and `pi0_fast_base` use the following action space definitions (left and right are defined looking from behind the robot towards the workspace):
```
    "dim_0:dim_5": "left arm joint angles",
    "dim_6": "left arm gripper position",
    "dim_7:dim_12": "right arm joint angles (for bi-manual only)",
    "dim_13": "right arm gripper position (for bi-manual only)",

    # For mobile robots:
    "dim_14:dim_15": "x-y base velocity (for mobile robots only)",
```

The proprioceptive state uses the same definitions as the action space, except for the base x-y position (the last two dimensions) for mobile robots, which we don't include in the proprioceptive state.

For 7-DoF robots (e.g. Franka), we use the first 7 dimensions of the action space for the joint actions, and the 8th dimension for the gripper action.

General info for Pi robots:
- Joint angles are expressed in radians, with position zero corresponding to the zero position reported by each robot's interface library, except for ALOHA, where the standard ALOHA code uses a slightly different convention (see the [ALOHA example code](../examples/aloha_real/README.md) for details).
- Gripper positions are in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
- Control frequencies are either 20 Hz for UR5e and Franka, and 50 Hz for ARX and Trossen (ALOHA) arms.

For DROID, we use the original DROID action configuration, with joint velocity actions in the first 7 dimensions and gripper actions in the 8th dimension + a control frequency of 15 Hz.

---

# Remote Inference

We provide utilities for running openpi models remotely. This is useful for running inference on more powerful GPUs off-robot, and also helps keep the robot and policy environments separate (and e.g. avoid dependency hell with robot software).

## Starting a remote policy server

To start a remote policy server, you can simply run the following command:

```bash
uv run scripts/serve_policy.py --env=[DROID | ALOHA | LIBERO]
```

The `env` argument specifies which $\pi_0$ checkpoint should be loaded. Under the hood, this script will execute a command like the following, which you can use to start a policy server, e.g. for checkpoints you trained yourself (here an example for the DROID environment):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid
```

This will start a policy server that will serve the policy specified by the `config` and `dir` arguments. The policy will be served on the specified port (default: 8000).

## Querying the remote policy server from your robot code

We provide a client utility with minimal dependencies that you can easily embed into any robot codebase.

First, install the `openpi-client` package in your robot environment:

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

Then, you can use the client to query the remote policy server from your robot code. Here's an example of how to do this:

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

for step in range(num_steps):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]

    # Execute the actions in the environment.
    ...

```

Here, the `host` and `port` arguments specify the IP address and port of the remote policy server. You can also specify these as command-line arguments to your robot code, or hard-code them in your robot codebase. The `observation` is a dictionary of observations and the prompt, following the specification of the policy inputs for the policy you are serving. We have concrete examples of how to construct this dictionary for different environments in the [simple client example](examples/simple_client/main.py).

---

# ActionQueue Implementation Analysis

This section provides detailed technical analysis of the ActionQueue implementation and RTC (Receding-horizon Temporal Consistency) mechanism.

## 核心设计原则

**关键点**：ActionQueue 维护两个队列，长度相同但数值不同（归一化 vs 反归一化）：
- `original_queue`: 归一化的动作，用于 RTC 计算
- `queue`: 后处理的动作，用于机器人执行
- 它们共享同一个 `last_index`（消费索引）
- 时间步一一对应，只是数值空间不同

## ActionQueue 方法分类

### 1. 只操作 `queue` 的方法（执行相关）

#### `get()` - 获取下一个动作
```python
def get(self) -> Tensor | None:
    with self.lock:
        if self.queue is None or self.last_index >= len(self.queue):
            return None
        action = self.queue[self.last_index]  # 只从 queue 读取
        self.last_index += 1                  # 共享索引增加
        return action.clone()
```

**特点**：
- ✅ 只读取 `queue`（后处理动作）
- ✅ 增加共享的 `last_index`
- ❌ 不操作 `original_queue`

**用途**：`actor_control` 线程调用，获取动作发送给机器人

---

#### `qsize()` - 获取队列大小
```python
def qsize(self) -> int:
    if self.queue is None:
        return 0
    length = len(self.queue)
    return length - self.last_index
```

**特点**：
- ✅ 只检查 `queue`
- ❌ 不检查 `original_queue`
- 由于两个队列长度相同，结果等价

**用途**：判断还有多少动作可以执行

---

#### `empty()` - 检查是否为空
```python
def empty(self) -> bool:
    if self.queue is None:
        return True
    length = len(self.queue)
    return length - self.last_index <= 0
```

**特点**：
- ✅ 只检查 `queue`
- ❌ 不检查 `original_queue`

**用途**：判断执行队列是否为空

---

### 2. 只操作 `original_queue` 的方法（RTC 相关）

#### `get_left_over()` - 获取未执行的原始动作
```python
def get_left_over(self) -> Tensor | None:
    with self.lock:
        if self.original_queue is None:
            return None
        return self.original_queue[self.last_index :]  # 只从 original_queue 读取
```

**特点**：
- ✅ 只读取 `original_queue`（归一化动作）
- ✅ 使用共享的 `last_index` 作为起始位置
- ❌ 不操作 `queue`
- ❌ 不修改 `last_index`

**用途**：`get_actions` 线程调用，获取未执行的原始动作用于 RTC

**关键**：返回的是 `original_queue[last_index:]`，即从当前消费位置到队列末尾的所有动作

---

### 3. 同时操作两个队列的方法（同步更新）

#### `merge()` - 合并新动作到队列
```python
def merge(self, original_actions, processed_actions, real_delay, ...):
    with self.lock:
        if self.cfg.enabled:
            self._replace_actions_queue(original_actions, processed_actions, real_delay)
        else:
            self._append_actions_queue(original_actions, processed_actions)
```

**特点**：
- ✅ 同时接收两个队列的输入
- ✅ 根据 RTC 模式选择不同的处理方式
- ✅ 两个队列同步更新

---

#### `_replace_actions_queue()` - RTC 模式：替换队列
```python
def _replace_actions_queue(self, original_actions, processed_actions, real_delay):
    # 两个队列都丢弃前 real_delay 个动作
    self.original_queue = original_actions[real_delay:].clone()
    self.queue = processed_actions[real_delay:].clone()

    self.last_index = 0  # 重置索引
```

**特点**：
- ✅ **两个队列处理方式完全相同**
- ✅ 都丢弃前 `real_delay` 个动作
- ✅ 都重置 `last_index = 0`
- ✅ 都使用 `.clone()` 防止外部修改

**为什么丢弃前 real_delay 个动作？**
- 推理需要时间（`real_delay` 步）
- 在这段时间内，机器人已经执行了 `real_delay` 个动作
- 新生成的动作块应该从 `real_delay` 之后开始使用

**示例**：
```python
# 假设 real_delay = 3
original_actions = [a0, a1, a2, a3, a4, a5, ...]  # 10个动作
processed_actions = [p0, p1, p2, p3, p4, p5, ...]  # 10个动作

# 丢弃前3个，保留后7个
self.original_queue = [a3, a4, a5, ...]  # 7个动作
self.queue = [p3, p4, p5, ...]           # 7个动作
self.last_index = 0
```

---

#### `_append_actions_queue()` - 非 RTC 模式：追加队列
```python
def _append_actions_queue(self, original_actions, processed_actions):
    if self.queue is None:
        # 首次初始化
        self.original_queue = original_actions.clone()
        self.queue = processed_actions.clone()
        return

    # 追加新动作
    self.original_queue = torch.cat([self.original_queue, original_actions.clone()])
    self.original_queue = self.original_queue[self.last_index :]  # 移除已消费的

    self.queue = torch.cat([self.queue, processed_actions.clone()])
    self.queue = self.queue[self.last_index :]  # 移除已消费的

    self.last_index = 0
```

**特点**：
- ✅ **两个队列处理方式完全相同**
- ✅ 都先追加新动作
- ✅ 都移除已消费的动作（`[last_index:]`）
- ✅ 都重置 `last_index = 0`

**示例**：
```python
# 假设当前状态
self.original_queue = [a0, a1, a2, a3, a4]  # 5个动作
self.queue = [p0, p1, p2, p3, p4]           # 5个动作
self.last_index = 2  # 已消费前2个

# 新动作
new_original = [a5, a6, a7]  # 3个动作
new_processed = [p5, p6, p7]  # 3个动作

# 追加后
self.original_queue = [a0, a1, a2, a3, a4, a5, a6, a7]  # 8个
self.queue = [p0, p1, p2, p3, p4, p5, p6, p7]            # 8个

# 移除已消费的（last_index=2）
self.original_queue = [a2, a3, a4, a5, a6, a7]  # 6个
self.queue = [p2, p3, p4, p5, p6, p7]           # 6个
self.last_index = 0
```

---

### 4. 共享状态的方法

#### `get_action_index()` - 获取当前消费索引
```python
def get_action_index(self) -> int:
    return self.last_index
```

**特点**：
- ✅ 返回共享的 `last_index`
- ✅ 两个队列使用同一个索引

**用途**：记录推理开始时的消费位置，用于验证延迟

---

## 关键设计模式

### 1. 同步更新原则

两个队列**总是同步更新**：
- 同时初始化
- 同时替换（RTC 模式）
- 同时追加（非 RTC 模式）
- 共享同一个 `last_index`

### 2. 分离读取原则

读取操作**分离**：
- `get()` → 只读 `queue`（执行用）
- `get_left_over()` → 只读 `original_queue`（RTC 用）

### 3. 索引共享原则

`last_index` 是**共享的**：
- 两个队列长度相同
- 时间步一一对应
- 消费进度同步

## ActionQueue 方法总结表

| 方法 | original_queue | queue | last_index | 用途 |
|------|----------------|-------|------------|------|
| `get()` | ❌ | ✅ 读取 | ✅ 增加 | 执行动作 |
| `qsize()` | ❌ | ✅ 检查 | ✅ 使用 | 检查队列大小 |
| `empty()` | ❌ | ✅ 检查 | ✅ 使用 | 检查是否为空 |
| `get_left_over()` | ✅ 读取 | ❌ | ✅ 使用 | RTC 计算 |
| `get_action_index()` | ❌ | ❌ | ✅ 返回 | 获取索引 |
| `merge()` | ✅ 更新 | ✅ 更新 | ✅ 重置 | 合并新动作 |
| `_replace_actions_queue()` | ✅ 替换 | ✅ 替换 | ✅ 重置 | RTC 模式替换 |
| `_append_actions_queue()` | ✅ 追加 | ✅ 追加 | ✅ 重置 | 非 RTC 模式追加 |

---

## RTC 实现详细分析：eval_with_real_robot.py

### 架构概览

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

### 核心组件

#### 1. ActionQueue - 线程安全队列

**职责**：
- 存储**原始动作**（`original_queue`）：用于 RTC 计算 `prev_chunk_left_over`
- 存储**后处理动作**（`queue`）：用于机器人执行
- 跟踪当前消费索引（`last_index`）

**关键方法**：
- `get()`: 获取下一个动作（增加 `last_index`）
- `get_left_over()`: 获取未执行的原始动作（用于 RTC）
- `merge()`: 合并新动作到队列（RTC 模式会替换队列）

#### 2. get_actions 线程（动作生成器）

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

#### 3. actor_control 线程（动作执行器）

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

### RTC 关键机制

#### 1. 动作队列替换策略（RTC 模式）

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

#### 2. prev_chunk_left_over 的传递链

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

#### 3. inference_delay 的双重作用

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

#### 4. 时间同步机制

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

### 配置参数

#### action_queue_size_to_get_new_actions

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

### 与 openpi_client 的对比

| 特性 | eval_with_real_robot.py | openpi_client |
|------|------------------------|---------------|
| 架构 | 单机双线程 | 网络 client-server |
| 通信 | ActionQueue (内存) | WebSocket |
| 同步 | 线程锁 | 网络协议 |
| 延迟测量 | 本地时间戳 | 网络往返时间 |
| 队列管理 | ActionQueue.merge() | ActionChunkBroker |

### 关键代码路径

#### RTC 参数传递链

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

#### 动作执行链

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

---

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
