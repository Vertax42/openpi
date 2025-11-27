# Aloha Policy 训练时的 `_output_transform` 详细分析

## 概述

在训练 aloha_policy 时，`_output_transform` 是一个由多个转换函数组成的转换链，用于将模型输出的原始 actions 转换为最终可用的 actions。这个转换链在 `Policy.infer()` 方法中被应用（见 `src/openpi/policies/policy.py:117`）。

## 转换链的构建

`_output_transform` 的构建发生在 `create_trained_policy()` 函数中（`src/openpi/policies/policy_config.py:90-96`），转换链的顺序如下：

```python
output_transforms=[
    *data_config.model_transforms.outputs,      # 1. 模型相关的输出转换
    transforms.Unnormalize(...),                 # 2. 反归一化
    *data_config.data_transforms.outputs,       # 3. 数据相关的输出转换
    *repack_transforms.outputs,                  # 4. 重新打包转换
]
```

## 详细分析每个转换步骤

### 1. 模型相关的输出转换 (`model_transforms.outputs`)

对于 Aloha policy，`ModelTransformFactory` 根据模型类型创建不同的转换：

- **PI0 和 PI05 模型**：`model_transforms.outputs` 是**空的**（只有输入转换，没有输出转换）
- **PI0_FAST 模型**：包含 `ExtractFASTActions` 转换，用于从 token 序列中提取 actions

**代码位置**：`src/openpi/training/config.py:120-176`

### 2. 反归一化 (`Unnormalize`)

将模型输出的归一化 actions 转换回原始数值范围。

**实现**：`src/openpi/transforms.py:148-181`

**功能**：
- 如果 `use_quantiles=False`：使用 z-score 反归一化
  ```python
  x * (std + 1e-6) + mean
  ```
- 如果 `use_quantiles=True`：使用分位数反归一化
  ```python
  (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
  ```

**使用的统计信息**：从 checkpoint 目录加载的 `norm_stats`，包含 `mean`、`std`（或 `q01`、`q99`）

### 3. 数据相关的输出转换 (`data_transforms.outputs`)

这部分是 Aloha policy 特有的转换，在 `LeRobotAlohaDataConfig.create()` 中定义。

**代码位置**：`src/openpi/training/config.py:293-302`

#### 3.1 基础转换：`AlohaOutputs`

**实现**：`src/openpi/policies/aloha_policy.py:101-112`

**功能**：
1. **截取前 14 维**：只保留 actions 的前 14 个维度
   ```python
   actions = np.asarray(data["actions"][:, :14])
   ```

2. **空间转换**（如果 `adapt_to_pi=True`）：
   - 调用 `_encode_actions(actions, adapt_to_pi=True)` 进行转换
   - 转换包括：
     - **关节角度翻转**：使用 `_joint_flip_mask()` 对某些关节维度进行符号翻转
       ```python
       mask = [1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]
       actions = mask * actions
       ```
     - **夹爪角度转换**：将夹爪从 pi 内部运行时使用的角度空间转换回 Aloha 使用的空间
       ```python
       actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
       ```

**为什么需要 `adapt_to_pi`？**
- Aloha 和 pi 内部运行时使用不同的关节角度和夹爪空间
- 如果使用标准的 Aloha 数据训练，需要设置 `adapt_to_pi=True` 来适配 pi 预训练的基础模型

#### 3.2 Delta Actions 转换：`AbsoluteActions`（可选）

**条件**：如果 `use_delta_joint_actions=True`（默认值）

**实现**：`src/openpi/transforms.py:225-244`

**功能**：将 delta actions（相对于当前状态的增量）转换回绝对 actions

**Delta Action Mask**：
```python
delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
# 结果: (True, True, True, True, True, True,  # 前6维（左臂关节）
#        False,                                  # 第7维（左臂夹爪）
#        True, True, True, True, True, True,    # 8-13维（右臂关节）
#        False)                                  # 第14维（右臂夹爪）
```

**转换逻辑**：
```python
actions[..., :dims] += np.expand_dims(
    np.where(mask, state[..., :dims], 0), 
    axis=-2
)
```

**含义**：
- 对于 mask 为 `True` 的维度（关节角度），将 delta actions 加上当前状态，得到绝对 actions
- 对于 mask 为 `False` 的维度（夹爪），保持原样（已经是绝对值）

**为什么需要这个转换？**
- 训练时，模型输入的是 delta actions（相对于当前状态的增量），这样模型更容易学习相对运动
- 推理时，需要输出绝对 actions，所以需要将 delta actions 转换回绝对 actions

### 4. 重新打包转换 (`repack_transforms.outputs`)

通常为空，用于在需要时重新组织输出数据的结构。

## 完整的转换流程示例

假设模型输出归一化后的 actions 形状为 `[action_horizon, action_dim]`，且 `use_delta_joint_actions=True`，`adapt_to_pi=True`：

1. **模型输出**：归一化的 delta actions（可能超过 14 维）
2. **Unnormalize**：反归一化，得到原始数值范围的 delta actions
3. **AlohaOutputs**：
   - 截取前 14 维：`actions[:, :14]`
   - 如果 `adapt_to_pi=True`：
     - 应用关节翻转 mask
     - 转换夹爪角度空间（从 pi 空间到 Aloha 空间）
4. **AbsoluteActions**：
   - 将前 6 维（左臂关节）的 delta actions 加上当前状态
   - 将第 8-13 维（右臂关节）的 delta actions 加上当前状态
   - 第 7 维（左臂夹爪）和第 14 维（右臂夹爪）保持不变
5. **最终输出**：绝对 actions，形状为 `[action_horizon, 14]`

## 关键参数说明

### `adapt_to_pi` (默认: `True`)
- **作用**：在 Aloha 空间和 pi 内部运行时空间之间转换
- **何时使用**：使用标准 Aloha 数据训练时应该设置为 `True`
- **影响**：
  - 关节角度：某些关节维度会翻转符号
  - 夹爪：从角度空间转换到 Aloha 的线性空间

### `use_delta_joint_actions` (默认: `True`)
- **作用**：训练时使用 delta actions（增量），推理时转换回绝对 actions
- **影响**：
  - 训练时：输入转换使用 `DeltaActions`，将绝对 actions 转换为 delta actions
  - 推理时：输出转换使用 `AbsoluteActions`，将 delta actions 转换回绝对 actions

## 相关代码文件

- `src/openpi/policies/policy.py`：Policy 类的实现，`_output_transform` 的使用
- `src/openpi/policies/policy_config.py`：`create_trained_policy()` 函数，构建转换链
- `src/openpi/policies/aloha_policy.py`：`AlohaOutputs` 转换的实现
- `src/openpi/training/config.py`：`LeRobotAlohaDataConfig` 配置类
- `src/openpi/transforms.py`：各种转换函数的实现（`Unnormalize`、`AbsoluteActions` 等）

## 注意事项

1. **转换顺序很重要**：转换链必须按照正确的顺序执行，因为每个转换都依赖于前一个转换的输出格式
2. **状态依赖**：`AbsoluteActions` 转换需要当前状态（`state`），所以 `outputs` 字典中必须包含 `state` 字段
3. **维度匹配**：确保 `norm_stats` 中的维度与实际的 actions 维度匹配
4. **空间转换**：`adapt_to_pi` 参数的选择取决于训练数据的格式和基础模型的训练方式

