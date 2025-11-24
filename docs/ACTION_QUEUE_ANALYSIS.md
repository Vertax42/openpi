# ActionQueue 对 original_queue 和 queue 的处理差异分析

## 核心设计原则

**关键点**：两个队列**长度相同**，但**数值不同**（归一化 vs 反归一化）
- 它们共享同一个 `last_index`（消费索引）
- 时间步一一对应，只是数值空间不同

## 方法分类分析

### 1. **只操作 `queue` 的方法**

#### `get()` - 获取下一个动作（用于执行）
```python
def get(self) -> Tensor | None:
    with self.lock:
        if self.queue is None or self.last_index >= len(self.queue):
            return None
        action = self.queue[self.last_index]  # 只从 queue 读取
        self.last_index += 1
        return action.clone()
```

**特点**：
- ✅ 只读取 `queue`（后处理动作）
- ❌ 不读取 `original_queue`
- ✅ 增加 `last_index`（影响两个队列的剩余部分）

**用途**：`actor_control` 线程调用，获取动作发送给机器人

---

#### `qsize()` - 获取队列剩余大小
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

**用途**：判断是否还有动作可执行

---

#### `empty()` - 检查队列是否为空
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
- 由于两个队列长度相同，结果等价

**用途**：判断队列是否为空

---

### 2. **只操作 `original_queue` 的方法**

#### `get_left_over()` - 获取未执行的原始动作（用于 RTC）
```python
def get_left_over(self) -> Tensor | None:
    with self.lock:
        if self.original_queue is None:
            return None
        return self.original_queue[self.last_index :]  # 只从 original_queue 读取
```

**特点**：
- ✅ 只读取 `original_queue`（归一化动作）
- ❌ 不读取 `queue`
- ✅ 使用共享的 `last_index` 切片

**用途**：`get_actions` 线程调用，获取未执行的动作作为 `prev_chunk_left_over`

**关键**：返回的是 `original_queue[last_index:]`，即从当前消费位置到队列末尾的所有动作

---

### 3. **同时操作两个队列的方法**

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

### 4. **共享状态的方法**

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

### 1. **同步更新原则**

两个队列**总是同步更新**：
- 同时初始化
- 同时替换（RTC 模式）
- 同时追加（非 RTC 模式）
- 共享同一个 `last_index`

### 2. **分离读取原则**

读取操作**分离**：
- `get()` → 只读 `queue`（执行用）
- `get_left_over()` → 只读 `original_queue`（RTC 用）

### 3. **索引共享原则**

`last_index` 是**共享的**：
- 两个队列长度相同
- 时间步一一对应
- 消费进度同步

## 数据流示例

### RTC 模式下的完整流程

```
1. 初始化
   original_queue = None
   queue = None
   last_index = 0

2. 第一次 merge（real_delay=0）
   original_queue = [a0, a1, ..., a9]  # 10个动作
   queue = [p0, p1, ..., p9]            # 10个动作
   last_index = 0

3. actor_control 执行 3 个动作
   get() × 3 → last_index = 3
   original_queue = [a0, a1, a2, a3, ..., a9]  # 未变
   queue = [p0, p1, p2, p3, ..., p9]          # 未变

4. get_actions 获取剩余动作
   get_left_over() → [a3, a4, ..., a9]  # 7个动作

5. 生成新动作（推理延迟 real_delay=3）
   new_original = [a0', a1', ..., a9']   # 10个新动作
   new_processed = [p0', p1', ..., p9'] # 10个新动作

6. 第二次 merge（real_delay=3）
   # 丢弃前3个（推理期间已执行）
   original_queue = [a3', a4', ..., a9']  # 7个动作
   queue = [p3', p4', ..., p9']           # 7个动作
   last_index = 0  # 重置
```

## 总结

| 方法 | original_queue | queue | last_index |
|------|---------------|-------|------------|
| `get()` | ❌ | ✅ 读取 | ✅ 增加 |
| `qsize()` | ❌ | ✅ 检查 | ✅ 使用 |
| `empty()` | ❌ | ✅ 检查 | ✅ 使用 |
| `get_left_over()` | ✅ 读取 | ❌ | ✅ 使用 |
| `get_action_index()` | ❌ | ❌ | ✅ 返回 |
| `merge()` | ✅ 更新 | ✅ 更新 | ✅ 重置 |
| `_replace_actions_queue()` | ✅ 替换 | ✅ 替换 | ✅ 重置 |
| `_append_actions_queue()` | ✅ 追加 | ✅ 追加 | ✅ 重置 |

**核心发现**：
1. **写入操作**：两个队列总是同步更新，处理方式完全相同
2. **读取操作**：分离读取，`queue` 用于执行，`original_queue` 用于 RTC
3. **索引管理**：共享 `last_index`，保证两个队列的消费进度同步
