# ActionQueue 方法对 original_queue 和 queue 的处理差异分析

## 核心设计原则

**关键点**：两个队列使用**共享的 `last_index`**，表示消费进度始终同步。

## 方法分类

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
    length = len(self.queue)  # 只检查 queue
    return length - self.last_index
```

**特点**：
- ✅ 只检查 `queue`
- ❌ 不检查 `original_queue`

**用途**：判断还有多少动作可以执行

---

#### `empty()` - 检查是否为空
```python
def empty(self) -> bool:
    if self.queue is None:
        return True
    length = len(self.queue)  # 只检查 queue
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
- ✅ 只读取 `original_queue`（原始动作）
- ✅ 使用共享的 `last_index` 作为起始位置
- ❌ 不操作 `queue`
- ❌ 不修改 `last_index`

**用途**：`get_actions` 线程调用，获取未执行的原始动作用于 RTC

**关键理解**：
- 返回 `original_queue[last_index:]`，即从当前消费位置到队列末尾的所有动作
- 这些是**还未执行**的原始动作，将作为 `prev_chunk_left_over` 传递给下一次推理

---

### 3. 同时操作两个队列的方法（同步更新）

#### `merge()` - 合并新动作
```python
def merge(self, original_actions, processed_actions, real_delay, ...):
    with self.lock:
        if self.cfg.enabled:
            self._replace_actions_queue(original_actions, processed_actions, real_delay)
        else:
            self._append_actions_queue(original_actions, processed_actions)
```

**特点**：
- ✅ 同时接收两个队列的数据
- ✅ 根据 RTC 模式选择不同的更新策略
- ✅ 两个队列**总是同步更新**

---

#### `_replace_actions_queue()` - RTC 模式：替换队列
```python
def _replace_actions_queue(self, original_actions, processed_actions, real_delay):
    # 同时替换两个队列
    self.original_queue = original_actions[real_delay:].clone()
    self.queue = processed_actions[real_delay:].clone()
    
    self.last_index = 0  # 重置共享索引
```

**特点**：
- ✅ **同时替换**两个队列
- ✅ 都丢弃前 `real_delay` 个动作（推理期间已执行）
- ✅ 重置 `last_index = 0`

**处理方式**：
- `original_queue` ← `original_actions[real_delay:]`
- `queue` ← `processed_actions[real_delay:]`
- 两者**完全同步**，只是数据来源不同（归一化 vs 反归一化）

---

#### `_append_actions_queue()` - 非 RTC 模式：追加队列
```python
def _append_actions_queue(self, original_actions, processed_actions):
    if self.queue is None:
        # 初始化：同时设置两个队列
        self.original_queue = original_actions.clone()
        self.queue = processed_actions.clone()
        return
    
    # 追加：同时操作两个队列
    self.original_queue = torch.cat([self.original_queue, original_actions.clone()])
    self.original_queue = self.original_queue[self.last_index :]  # 移除已消费的
    
    self.queue = torch.cat([self.queue, processed_actions.clone()])
    self.queue = self.queue[self.last_index :]  # 移除已消费的
    
    self.last_index = 0  # 重置索引
```

**特点**：
- ✅ **同时追加**两个队列
- ✅ 都移除已消费的动作（`[self.last_index:]`）
- ✅ 重置 `last_index = 0`

**处理方式**：
1. 将新动作追加到两个队列
2. 从 `last_index` 开始切片，移除已消费的部分
3. 两个队列**完全同步**操作

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

### 1. **共享索引模式**

```
last_index (共享)
    ↓
    ├─→ queue[last_index]          (执行时读取)
    └─→ original_queue[last_index:] (RTC 时读取)
```

**优势**：
- 保证两个队列的消费进度始终一致
- 简化同步逻辑
- 避免索引不一致的问题

### 2. **同步更新模式**

所有写入操作都**同时更新**两个队列：

```python
# RTC 模式
original_queue = new_original[real_delay:]
queue = new_processed[real_delay:]

# 非 RTC 模式
original_queue = cat([old_original, new_original])[last_index:]
queue = cat([old_queue, new_processed])[last_index:]
```

**保证**：
- 两个队列的长度始终相同
- 两个队列的消费进度始终同步
- 只是数据内容不同（归一化 vs 反归一化）

### 3. **分离读取模式**

读取操作根据用途分离：

```
执行用途 → 读取 queue (后处理动作)
RTC 用途 → 读取 original_queue (原始动作)
```

## 处理差异总结表

| 方法 | queue | original_queue | last_index | 用途 |
|------|-------|----------------|------------|------|
| `get()` | ✅ 读取 | ❌ | ✅ 增加 | 执行动作 |
| `qsize()` | ✅ 检查 | ❌ | ✅ 使用 | 检查队列大小 |
| `empty()` | ✅ 检查 | ❌ | ✅ 使用 | 检查是否为空 |
| `get_left_over()` | ❌ | ✅ 读取 | ✅ 使用 | RTC 计算 |
| `get_action_index()` | ❌ | ❌ | ✅ 返回 | 获取索引 |
| `merge()` | ✅ 更新 | ✅ 更新 | ✅ 重置 | 合并新动作 |
| `_replace_actions_queue()` | ✅ 替换 | ✅ 替换 | ✅ 重置 | RTC 模式替换 |
| `_append_actions_queue()` | ✅ 追加 | ✅ 追加 | ✅ 重置 | 非 RTC 模式追加 |

## 关键洞察

1. **写入时同步**：所有写入操作都同时更新两个队列，保证一致性

2. **读取时分离**：
   - 执行线程只读 `queue`
   - RTC 线程只读 `original_queue`

3. **共享索引**：`last_index` 是唯一的状态变量，两个队列共享

4. **数据对应**：两个队列在相同索引位置的元素对应同一个时间步，只是数值不同（归一化 vs 反归一化）

## 示例流程

```
初始状态：
  original_queue = [a0, a1, a2, a3]  (归一化)
  queue = [A0, A1, A2, A3]            (反归一化)
  last_index = 0

执行 2 个动作后：
  original_queue = [a0, a1, a2, a3]
  queue = [A0, A1, A2, A3]
  last_index = 2

get_left_over() 返回：
  original_queue[2:] = [a2, a3]  ← 用于 RTC

get() 返回：
  queue[2] = A2  ← 用于执行

merge() 新动作后：
  original_queue = [new_a0, new_a1, ...]  (新归一化动作)
  queue = [new_A0, new_A1, ...]          (新反归一化动作)
  last_index = 0  (重置)
```

## 总结

**核心设计**：
- 两个队列**结构完全对称**，只是数据内容不同
- 使用**共享索引**保证同步
- **写入同步，读取分离**

这种设计确保了：
1. RTC 计算使用正确的归一化动作
2. 机器人执行使用正确的物理值
3. 两个队列的状态始终保持一致

