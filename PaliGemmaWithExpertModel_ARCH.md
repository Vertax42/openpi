# PaliGemmaWithExpertModel 架构详解

> 本文档记录 OpenPI PyTorch 实现中 `PaliGemmaWithExpertModel` 的完整架构细节，
> 包括多模态数据流、逐层联合注意力机制、混合精度策略、以及 PI0/PI05 的差异。

## 文件索引

| 文件 | 路径 | 作用 |
|------|------|------|
| `gemma_pytorch.py` | `src/openpi/models_pytorch/gemma_pytorch.py` | PaliGemmaWithExpertModel 定义 |
| `pi0_pytorch.py` | `src/openpi/models_pytorch/pi0_pytorch.py` | PI0Pytorch 顶层模型（调用方） |
| `preprocessing_pytorch.py` | `src/openpi/models_pytorch/preprocessing_pytorch.py` | 观测数据预处理 |
| `modeling_gemma.py` | `transformers_replace/models/gemma/modeling_gemma.py` | 修改版 Gemma（adaRMSNorm） |
| `modeling_paligemma.py` | `transformers_replace/models/paligemma/modeling_paligemma.py` | 修改版 PaliGemma |
| `modeling_siglip.py` | `transformers_replace/models/siglip/modeling_siglip.py` | 修改版 SigLIP |

---

## 1. 模型总体结构

```
PI0Pytorch (pi0_pytorch.py:88)
│
├── paligemma_with_expert (PaliGemmaWithExpertModel)  ← gemma_pytorch.py:12
│   │
│   ├── paligemma (PaliGemmaForConditionalGeneration)  ← gemma_pytorch.py:57
│   │   ├── vision_tower (SiglipVisionModel)
│   │   │   └── vision_model
│   │   │       ├── embeddings
│   │   │       │   ├── patch_embedding (Conv2d)    ← 14x14 patch, stride=14
│   │   │       │   └── position_embedding (Embedding)
│   │   │       └── encoder (27 层 SiglipEncoderLayer)
│   │   ├── multi_modal_projector (Linear → 2048)
│   │   └── language_model (GemmaModel, "gemma_2b", 18 层)
│   │       ├── embed_tokens (Embedding, vocab=257152)
│   │       ├── layers[0..17] (GemmaDecoderLayer)
│   │       │   ├── input_layernorm (GemmaRMSNorm)
│   │       │   ├── self_attn (GemmaAttention)
│   │       │   │   ├── q_proj, k_proj, v_proj, o_proj
│   │       │   │   └── head_dim, num_heads, num_kv_heads
│   │       │   ├── post_attention_layernorm (GemmaRMSNorm)
│   │       │   └── mlp (GemmaMLP: gate_proj, up_proj, down_proj)
│   │       ├── norm (GemmaRMSNorm, final)
│   │       └── rotary_emb (GemmaRotaryEmbedding)
│   │
│   └── gemma_expert (GemmaForCausalLM, "gemma_300m", 18 层)
│       └── model (GemmaModel)
│           ├── embed_tokens = None  ← 被删除 (gemma_pytorch.py:59)
│           ├── layers[0..17] (GemmaDecoderLayer, 可选 adaRMSNorm)
│           │   ├── input_layernorm (GemmaRMSNorm, cond_dim=1024 if PI05)
│           │   ├── self_attn (GemmaAttention)
│           │   ├── post_attention_layernorm (GemmaRMSNorm, cond_dim=1024 if PI05)
│           │   └── mlp (GemmaMLP)
│           ├── norm (GemmaRMSNorm, final)
│           └── rotary_emb
│
├── action_in_proj (Linear, 32 → expert_width)    ← pi0_pytorch.py:104
├── action_out_proj (Linear, expert_width → 32)    ← pi0_pytorch.py:105
│
├── [PI0 模式]
│   ├── state_proj (Linear, 32 → expert_width)           ← pi0_pytorch.py:115
│   ├── action_time_mlp_in (Linear, 2*expert_width → expert_width)  ← pi0_pytorch.py:116-117
│   └── action_time_mlp_out (Linear, expert_width → expert_width)   ← pi0_pytorch.py:119-120
│
└── [PI05 模式]
    ├── time_mlp_in (Linear, expert_width → expert_width)   ← pi0_pytorch.py:108-109
    └── time_mlp_out (Linear, expert_width → expert_width)  ← pi0_pytorch.py:111-112
```

### 1.1 两个 Gemma 模型的参数对比

| 属性 | paligemma.language_model (Gemma 2B) | gemma_expert (Gemma 300M) |
|------|------|------|
| hidden_size (width) | 2048 | 1024 |
| num_heads | 8 | 8 |
| head_dim | 256 | 128 |
| num_kv_heads | 1 | 1 |
| num_hidden_layers (depth) | 18 | 18 |
| mlp_dim | 16384 | 4096 |
| vocab_size | 257152 | 257152 (未使用) |
| embed_tokens | 有 | None (已删除) |
| adaRMSNorm (PI05) | 不启用 | 启用 |

> 配置来源：`_gemma.get_config("gemma_2b")` 和 `_gemma.get_config("gemma_300m")`，
> 在 `pi0_pytorch.py:94-95` 获取。

---

## 2. PaliGemmaWithExpertModel 初始化详解

### 2.1 VLM (PaliGemma) 配置构建

**代码位置**: `gemma_pytorch.py:24-41`

```python
vlm_config_hf = CONFIG_MAPPING["paligemma"]()
```

逐字段设置 HuggingFace PaliGemmaConfig：

| 配置字段 | 值 | 来源/说明 |
|---------|-----|---------|
| `_vocab_size` | 257152 | Gemma tokenizer 词表大小 (line 25) |
| `image_token_index` | 257152 | 图像占位 token ID (line 26) |
| `text_config.hidden_size` | `vlm_config.width` = 2048 | Gemma 2B hidden dim (line 27) |
| `text_config.intermediate_size` | `vlm_config.mlp_dim` = 16384 | FFN 中间层 (line 28) |
| `text_config.num_attention_heads` | `vlm_config.num_heads` = 8 | 注意力头数 (line 29) |
| `text_config.head_dim` | `vlm_config.head_dim` = 256 | 每头维度 (line 30) |
| `text_config.num_hidden_layers` | `vlm_config.depth` = 18 | transformer 层数 (line 31) |
| `text_config.num_key_value_heads` | `vlm_config.num_kv_heads` = 1 | GQA: 1 个 KV 头 (line 32) |
| `text_config.hidden_activation` | `"gelu_pytorch_tanh"` | MLP 激活函数 (line 33) |
| `text_config.use_adarms` | `use_adarms[0]` = False | VLM 不使用 adaRMS (line 36) |
| `vision_config.intermediate_size` | 4304 | SigLIP 中间层 (line 38) |
| `vision_config.projection_dim` | 2048 | SigLIP 输出投影维度 (line 39) |

### 2.2 Action Expert 配置构建

**代码位置**: `gemma_pytorch.py:43-55`

```python
action_expert_config_hf = CONFIG_MAPPING["gemma"](...)
```

| 配置字段 | 值 | 说明 |
|---------|-----|------|
| `hidden_size` | `action_expert_config.width` = 1024 | Expert hidden dim |
| `intermediate_size` | `action_expert_config.mlp_dim` = 4096 | FFN 中间层 |
| `num_attention_heads` | 8 | 注意力头数 |
| `head_dim` | 128 | 每头维度 |
| `num_hidden_layers` | 18 | 与 VLM 层数相同（联合注意力的前提） |
| `use_adarms` | `use_adarms[1]` | PI0=False, PI05=True (line 53) |
| `adarms_cond_dim` | 1024 (PI05) / None (PI0) | 条件向量维度 (line 54) |

### 2.3 模型实例化

**代码位置**: `gemma_pytorch.py:57-59`

```python
self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)  # line 57
self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)      # line 58
self.gemma_expert.model.embed_tokens = None                                # line 59
```

- `PaliGemmaForConditionalGeneration` 包含完整的 SigLIP 视觉编码器 + Gemma 语言模型。
- `GemmaForCausalLM` 包含一个 GemmaModel，但其 `embed_tokens` 被置为 None，
  因为 Action Expert 接收的是连续 embedding（来自投影层），不需要离散 token embedding。

### 2.4 混合精度初始化

**代码位置**: `gemma_pytorch.py:61-83`

```python
self.to_bfloat16_for_selected_params(precision)  # line 61
```

执行逻辑：
1. 先将整个模型转为 bfloat16 (line 65)
2. 再将以下参数强制恢复为 float32 (lines 72-83):

| 保持 float32 的参数 | 原因 |
|-------------------|------|
| `vision_tower.vision_model.embeddings.patch_embedding.weight/bias` | 卷积初始层，精度敏感 |
| `vision_tower.vision_model.embeddings.position_embedding.weight` | 位置编码，精度敏感 |
| `input_layernorm` (所有层) | RMSNorm 的方差计算需要 float32 |
| `post_attention_layernorm` (所有层) | 同上 |
| `model.norm` (final norm) | 同上 |

> 注意：adaRMSNorm 中的 `dense` 层（用于生成 scale/shift/gate）也包含在 layernorm 的匹配规则中，
> 因为它属于 `input_layernorm` 或 `post_attention_layernorm` 的子模块。

---

## 3. 嵌入接口 (Embedding API)

### 3.1 embed_image

**代码位置**: `gemma_pytorch.py:85-86`

```python
def embed_image(self, image: torch.Tensor):
    return self.paligemma.model.get_image_features(image)
```

调用链：
```
image [B, H, W, C]
  → SiglipVisionModel.forward()
    → patch_embedding (Conv2d 14x14, stride=14)
      → [B, 256, vision_hidden]  (224/14 = 16, 16x16 = 256 patches)
    → + position_embedding
    → 27 层 SiglipEncoderLayer
    → last_hidden_state [B, 256, vision_hidden]
  → multi_modal_projector (Linear)
    → image_features [B, 256, 2048]
```

**注意**：原版 transformers 的 `get_image_features` 会对输出除以 `sqrt(hidden_size)`。
替换版 (`modeling_paligemma.py`) 删掉了这行，缩放逻辑在调用方统一处理。

**调用位置**: `pi0_pytorch.py:217-218` (`embed_prefix` 方法内)

### 3.2 embed_language_tokens

**代码位置**: `gemma_pytorch.py:88-89`

```python
def embed_language_tokens(self, tokens: torch.Tensor):
    return self.paligemma.language_model.embed_tokens(tokens)
```

调用链：
```
lang_tokens [B, L]  (token ID序列, L = max_token_len)
  → Embedding(257152, 2048)
    → lang_emb [B, L, 2048]
```

**调用位置**: `pi0_pytorch.py:232`，之后会乘以 `sqrt(2048)` 做缩放 (line 234)：
```python
lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
lang_emb_dim = lang_emb.shape[-1]
return lang_emb * math.sqrt(lang_emb_dim)
```

> 原版 Gemma 在模型内部对 embedding 乘以 `sqrt(hidden_size)`，
> 替换版注释掉了这行 (`modeling_gemma.py` 的 `#hidden_states = hidden_states * normalizer`)，
> 改由 `embed_prefix` 在外部手动处理。

---

## 4. Forward 三种模式

`PaliGemmaWithExpertModel.forward()` 根据 `inputs_embeds` 参数的 None 模式，
走三条完全不同的执行路径。

**代码位置**: `gemma_pytorch.py:91-281`

### 4.1 模式一：Prefix-Only（推理第一步，建 KV Cache）

**条件**: `inputs_embeds = [prefix_embs, None]` (line 102)

**代码位置**: `gemma_pytorch.py:102-113`

```python
if inputs_embeds[1] is None:
    prefix_output = self.paligemma.language_model.forward(
        inputs_embeds=inputs_embeds[0],        # prefix_embs [B, Sp, 2048]
        attention_mask=attention_mask,          # [B, 1, Sp, Sp]
        position_ids=position_ids,             # [B, Sp]
        past_key_values=None,
        use_cache=use_cache,                   # True → 生成 KV cache
        adarms_cond=adarms_cond[0],            # None (VLM不用adaRMS)
    )
    prefix_past_key_values = prefix_output.past_key_values  # 18层的KV缓存
    prefix_output = prefix_output.last_hidden_state         # [B, Sp, 2048]
    suffix_output = None
```

**调用位置**: `pi0_pytorch.py:438-444` (`sample_actions` 方法)

**数据流**:
```
prefix_embs [B, 768+L, 2048]
  → Gemma 2B (18层, use_cache=True)
  → prefix_output [B, 768+L, 2048]  (不使用)
  → past_key_values (18层, 每层 [K, V])  ← 关键输出
```

### 4.2 模式二：Suffix-Only（推理 Denoise Step，读 KV Cache）

**条件**: `inputs_embeds = [None, suffix_embs]` (line 114)

**代码位置**: `gemma_pytorch.py:114-125`

```python
elif inputs_embeds[0] is None:
    suffix_output = self.gemma_expert.model.forward(
        inputs_embeds=inputs_embeds[1],        # suffix_embs [B, Ss, 1024]
        attention_mask=attention_mask,          # [B, 1, Ss, Sp+Ss]
        position_ids=position_ids,             # [B, Ss], 从 Sp 开始
        past_key_values=past_key_values,       # 来自模式一的 KV cache
        use_cache=use_cache,                   # False → 不更新 cache
        adarms_cond=adarms_cond[1],            # time_emb (PI05) / None (PI0)
    )
    suffix_output = suffix_output.last_hidden_state  # [B, Ss, 1024]
    prefix_output = None
    prefix_past_key_values = None
```

**调用位置**: `pi0_pytorch.py:500-507` (`denoise_step` 方法)

**数据流**:
```
suffix_embs [B, 51 或 50, 1024]
  → Gemma 300M (18层)
    每层 attention 时:
      Q 来自 suffix_embs
      K, V = cat([cached_prefix_K, suffix_K], dim=seq)
              cat([cached_prefix_V, suffix_V], dim=seq)
    → suffix 可以 attend 到 prefix 的所有信息
  → suffix_output [B, 51 或 50, 1024]
```

**KV cache 的使用方式** (修改版 `modeling_gemma.py` GemmaAttention):
```python
# use_cache=False 且 past_key_value 不为 None 时:
key_states = torch.cat([past_key_value[self.layer_idx][0], key_states], dim=2)
value_states = torch.cat([past_key_value[self.layer_idx][1], value_states], dim=2)
```
直接拼接 cached K/V，不更新 cache。这样 10 次 denoise step 共用同一份 prefix cache。

### 4.3 模式三：联合 Forward（训练模式）

**条件**: `inputs_embeds = [prefix_embs, suffix_embs]` (两者都不为 None, line 126)

**代码位置**: `gemma_pytorch.py:126-279`

这是最复杂的模式。不使用 HuggingFace 的标准 forward，而是**手动逐层展开**，
让两个不同宽度的模型共享同一次注意力计算。

---

## 5. 联合 Forward 逐层详解 (compute_layer_complete)

**代码位置**: `gemma_pytorch.py:158-238`

### 5.1 输入

```python
def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
```

- `inputs_embeds`: 列表 `[prefix_hidden [B, Sp, 2048], suffix_hidden [B, Ss, 1024]]`
- `attention_mask`: `[B, 1, Sp+Ss, Sp+Ss]` 4D mask
- `position_ids`: `[B, Sp+Ss]`
- `adarms_cond`: `[None, time_emb]` (PI05) 或 `[None, None]` (PI0)

### 5.2 第一步：Input LayerNorm + QKV 投影 (lines 161-178)

对两个模型**分别**做 LayerNorm 和 QKV 投影：

```python
for i, hidden_states in enumerate(inputs_embeds):          # i=0: VLM, i=1: Expert
    layer = models[i].layers[layer_idx]                     # 对应模型的第 layer_idx 层
    hidden_states, gate = layer.input_layernorm(
        hidden_states, cond=adarms_cond[i]                  # Expert 可能有 adaRMS cond
    )
    gates.append(gate)                                      # gate 用于后续 gated residual

    # QKV 投影 (每个模型用自己的投影矩阵)
    query_state = layer.self_attn.q_proj(hidden_states)     # [B, S_i, num_heads * head_dim]
                  .view(hidden_shape).transpose(1, 2)       # [B, H, S_i, head_dim]
    key_state   = layer.self_attn.k_proj(hidden_states)     # 同上
    value_state = layer.self_attn.v_proj(hidden_states)     # 同上
```

**张量维度** (以 PI0 为例, Sp=768+L, Ss=51):

| 模型 | Q/K/V shape | head_dim | num_heads |
|------|------------|----------|-----------|
| VLM (i=0) | `[B, 8, Sp, 256]` | 256 | 8 |
| Expert (i=1) | `[B, 8, Ss, 128]` | 128 | 8 |

> **注意**: 两个模型的 `head_dim` 不同 (256 vs 128)，但 `num_heads` 相同 (都是 8)。
> 这是联合注意力能工作的前提。但下面的拼接实际上是在 **seq 维度** 上拼接，
> 而 head_dim 维度需要 padding 到相同大小（由 HuggingFace 内部处理）。
>
> 实际上观察代码，这里 Q/K/V 是按 seq 维度 (dim=2) 拼接的。两个模型 head_dim 不同时，
> 代码使用 `att_output.reshape(batch_size, -1, 1 * 8 * head_dim)` (line 211)
> 来确保 reshape 正确。

### 5.3 第二步：拼接 Q/K/V，统一注意力 (lines 180-208)

```python
# 在 sequence 维度拼接两个模型的 Q/K/V
query_states = torch.cat(query_states, dim=2)   # [B, H, Sp+Ss, head_dim]
key_states   = torch.cat(key_states, dim=2)     # [B, H, Sp+Ss, head_dim]
value_states = torch.cat(value_states, dim=2)   # [B, H, Sp+Ss, head_dim]

# 使用 VLM 的 rotary_emb 计算 RoPE (line 192)
cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
    query_states, key_states, cos, sin, unsqueeze_dim=1
)

# 统一注意力计算 (line 201-208)
scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling
att_output, _ = modeling_gemma.eager_attention_forward(
    self.paligemma.language_model.layers[layer_idx].self_attn,  # 使用 VLM 的 attn module
    query_states,       # [B, H, Sp+Ss, head_dim]
    key_states,         # [B, H, Sp+Ss, head_dim]
    value_states,       # [B, H, Sp+Ss, head_dim]
    attention_mask,     # [B, 1, Sp+Ss, Sp+Ss]
    scaling,
)
```

**关键点**:
- 两个模型的 Q/K/V 在 **seq 维度** (dim=2) 拼接后做一次统一的注意力
- RoPE 使用 VLM 的 `rotary_emb`，对整个拼接序列统一计算
- `attention_mask` 控制可见性（prefix 不看 suffix，suffix 可看 prefix）
- `scaling` 使用 VLM 的 `self_attn.scaling` = `1/sqrt(head_dim)`

### 5.4 第三步：分割输出，各自做 FFN (lines 213-236)

```python
# Reshape attention output
att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)  # [B, Sp+Ss, 8*head_dim]

# 按 sequence 位置分割回两个模型
start_pos = 0
for i, hidden_states in enumerate(inputs_embeds):
    layer = models[i].layers[layer_idx]
    end_pos = start_pos + hidden_states.shape[1]

    # 用各自模型的 o_proj (line 222)
    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

    # 第一个 gated residual (line 225)
    out_emb = _gated_residual(hidden_states, out_emb, gates[i])

    after_first_residual = out_emb.clone()  # 保存用于第二个 residual (line 226)

    # Post-attention LayerNorm (line 227)
    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])

    # MLP (line 232)
    out_emb = layer.mlp(out_emb)

    # 第二个 gated residual (line 234)
    out_emb = _gated_residual(after_first_residual, out_emb, gate)

    outputs_embeds.append(out_emb)
    start_pos = end_pos
```

**每层内的详细数据流** (以第 k 层为例):

```
prefix_hidden [B, Sp, 2048]              suffix_hidden [B, Ss, 1024]
      │                                         │
      ▼                                         ▼
 input_layernorm (RMSNorm)               input_layernorm (adaRMSNorm if PI05)
  → normed_prefix, gate_0=None            → normed_suffix, gate_1
      │                                         │
      ▼                                         ▼
  q_proj → Q_p [B,8,Sp,256]             q_proj → Q_s [B,8,Ss,128]
  k_proj → K_p [B,8,Sp,256]             k_proj → K_s [B,8,Ss,128]
  v_proj → V_p [B,8,Sp,256]             v_proj → V_s [B,8,Ss,128]
      │                                         │
      └──── cat(dim=2) → Q,K,V [B,8,Sp+Ss,dim] ┘
                         │
                    apply_rotary_pos_emb
                         │
                  eager_attention_forward
                    (with 2D attention mask)
                         │
                  att_out [B, Sp+Ss, H*head_dim]
                         │
              ┌──── split ────┐
              │               │
      att_p [B,Sp,...]   att_s [B,Ss,...]
              │               │
        o_proj_vlm        o_proj_expert
              │               │
    gated_residual_0     gated_residual_0
     (gate=None, 即 +)    (gate=gate_1)
              │               │
   post_attn_layernorm   post_attn_layernorm(cond=time_emb)
              │               │
          mlp_vlm          mlp_expert
              │               │
    gated_residual_1     gated_residual_1
              │               │
      new_prefix_hidden   new_suffix_hidden
```

### 5.5 第四步：Final Norm (lines 260-275)

```python
def compute_final_norms(inputs_embeds, adarms_cond):
    outputs_embeds = []
    for i, hidden_states in enumerate(inputs_embeds):
        out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
        outputs_embeds.append(out_emb)
    return outputs_embeds
```

- VLM: `paligemma.language_model.norm(prefix_hidden, cond=None)` → 标准 RMSNorm
- Expert: `gemma_expert.model.norm(suffix_hidden, cond=time_emb)` → adaRMSNorm (PI05)

### 5.6 返回值 (lines 277-281)

```python
prefix_output = outputs_embeds[0]       # [B, Sp, 2048], 训练时不使用
suffix_output = outputs_embeds[1]       # [B, Ss, 1024], 送入 action_out_proj
prefix_past_key_values = None           # 训练模式不生成 cache
return [prefix_output, suffix_output], prefix_past_key_values
```

---

## 6. Attention Mask 机制

### 6.1 1D mask 构建

**Prefix** (`pi0_pytorch.py:210-252`):
```python
# 图像 token: att_mask = 0 (全部互相可见)
att_masks += [0] * num_img_embs    # × 3张图, 每张256个token → 768个0

# 语言 token: att_mask = 0 (与图像互相可见)
att_masks += [0] * num_lang_embs   # L个0
```

**Suffix** (`pi0_pytorch.py:255-337`):
```python
# PI0: state token att_mask = 1 (prefix 不看 state/action)
att_masks += [1]

# 第一个 action token: att_mask = 1 (新的 attention group 边界)
# 后续 action tokens: att_mask = 0 (同 group 内互相可见)
att_masks += [1] + ([0] * (action_horizon - 1))    # [1, 0, 0, ..., 0] × 50
```

### 6.2 2D mask 生成

**代码位置**: `pi0_pytorch.py:56-85`

```python
cumsum = torch.cumsum(att_masks, dim=1)
att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
return att_2d_masks & pad_2d_masks
```

`cumsum` 的 trick：
- att_mask=0 的 token 累加值不变 → 同组内互相可见
- att_mask=1 的 token 累加值 +1 → 新的 attention group
- `cumsum[q] <= cumsum[k]` 表示 query 可以 attend 到 key

### 6.3 可见性矩阵示意 (PI0, 3图 + L个lang + 1 state + 50 action)

```
att_masks = [0]*768 + [0]*L + [1] + [1] + [0]*49
cumsum    = [0]*768 + [0]*L + [1] + [2] + [2]*49

                   img(768)  lang(L)  state(1)  act_0(1)  act_1..49
  img(768)      │    ✓        ✓         ✗          ✗         ✗
  lang(L)       │    ✓        ✓         ✗          ✗         ✗
  state(1)      │    ✓        ✓         ✓          ✗         ✗
  act_0(1)      │    ✓        ✓         ✓          ✓         ✗
  act_1..49     │    ✓        ✓         ✓          ✓         ✓

图像和语言 token 互相完全可见 (bidirectional)
state 可以看图像和语言，但图像和语言看不到 state
action tokens 可以看所有 prefix + state，action 之间是 causal
```

### 6.4 4D mask 转换

**代码位置**: `pi0_pytorch.py:172-175`

```python
att_2d_masks_4d = att_2d_masks[:, None, :, :]                          # [B, 1, S, S]
return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)               # True→0, False→-inf
```

HuggingFace 的 attention 实现需要 additive mask：
- 可见位置 → `0.0` (不影响 attention score)
- 不可见位置 → `-2.38e38` (近似 -inf，softmax 后为 0)

---

## 7. PI0 vs PI05 的差异

### 7.1 Suffix Embedding 差异

**PI0 模式** (`pi0_pytorch.py:261-308`):

```
state [B,32] → state_proj → [B,1,1024]         ← 独立的 state token
action [B,50,32] → action_in_proj → [B,50,1024]
timestep [B] → sincos_emb → [B,1024] → expand → [B,50,1024]
cat([action, time], dim=-1) → [B,50,2048]
→ action_time_mlp_in → silu → action_time_mlp_out → [B,50,1024]
suffix_embs = cat([state_token, action_time_emb]) = [B, 51, 1024]
adarms_cond = None
```

**PI05 模式** (`pi0_pytorch.py:309-319`):

```
(无 state token, state 编码进 language prompt)
action [B,50,32] → action_in_proj → [B,50,1024]
timestep [B] → sincos_emb → [B,1024]
→ time_mlp_in → silu → time_mlp_out → silu → [B,1024]
suffix_embs = action_emb = [B, 50, 1024]
adarms_cond = time_emb [B, 1024]  ← 传入每层 adaRMSNorm
```

### 7.2 adaRMSNorm 详解

**代码位置**: 修改版 `modeling_gemma.py` `GemmaRMSNorm`

标准 RMSNorm (PI0 的 VLM 和 Expert):
```python
normed = x * rsqrt(mean(x²) + eps)
output = normed * (1 + weight)        # weight 是可学习参数
return output, None                    # gate = None
```

Adaptive RMSNorm (PI05 的 Expert):
```python
normed = x * rsqrt(mean(x²) + eps)
modulation = dense(cond)               # cond = time_emb [B, 1024]
                                       # dense: Linear(1024 → 3072), bias=True
                                       # 初始化为零，训练初期等价于标准RMSNorm
scale, shift, gate = chunk(modulation, 3)   # 各 [B, 1024]
output = normed * (1 + scale) + shift  # 自适应归一化
return output, gate                    # gate 用于 gated residual
```

### 7.3 Gated Residual

**代码位置**: 修改版 `modeling_gemma.py` `_gated_residual`

```python
def _gated_residual(x, y, gate):
    if gate is None:   # PI0 或 VLM: 标准残差
        return x + y
    return x + y * gate  # PI05 Expert: 门控残差
```

- `gate` 由 adaRMSNorm 根据 timestep 条件生成
- 训练初期 `dense` 初始化为零 → `gate=0` → 残差退化为 `x`（identity）
- 随着训练，模型学习在不同 timestep 下如何混合残差信息

---

## 8. 训练完整数据流

**入口**: `PI0Pytorch.forward()` (`pi0_pytorch.py:340-412`)

```
Step 0: 预处理
──────────────
observation
  → _preprocess_observation (pi0_pytorch.py:177-188)
    → images: list of 3 × [B, H, W, C]
    → img_masks: list of 3 × [B]
    → lang_tokens: [B, L] (int)
    → lang_masks: [B, L] (bool)
    → state: [B, 32]

Step 1: Flow Matching 采样 (pi0_pytorch.py:346-354)
────────────────────────
noise = N(0, 1) [B, 50, 32]
time ~ Beta(1.5, 1.0) * 0.999 + 0.001   ← 范围 [0.001, 1.0]
x_t = time * noise + (1 - time) * actions  ← 在 noise 和 actions 间线性插值
u_t = noise - actions                       ← 目标速度场

Step 2: Embedding (pi0_pytorch.py:356-361)
────────────────
embed_prefix(images, img_masks, lang_tokens, lang_masks)
  → prefix_embs [B, 768+L, 2048]
  → prefix_pad_masks [B, 768+L]
  → prefix_att_masks [B, 768+L]

embed_suffix(state, x_t, time)
  → suffix_embs [B, 51/50, 1024]   (PI0: 51, PI05: 50)
  → suffix_pad_masks [B, 51/50]
  → suffix_att_masks [B, 51/50]
  → adarms_cond [B, 1024] (PI05) / None (PI0)

Step 3: 构造全局 mask (pi0_pytorch.py:371-378)
─────────────────────
pad_masks = cat([prefix_pad, suffix_pad])          # [B, Sp+Ss]
att_masks = cat([prefix_att, suffix_att])          # [B, Sp+Ss]
att_2d_masks = make_att_2d_masks(pad, att)         # [B, Sp+Ss, Sp+Ss]
position_ids = cumsum(pad_masks) - 1               # [B, Sp+Ss]
att_2d_masks_4d = where(att_2d, 0.0, -2.38e38)    # [B, 1, Sp+Ss, Sp+Ss]

Step 4: 联合 Forward (pi0_pytorch.py:384-401)
────────────────────
paligemma_with_expert.forward(
    attention_mask = att_2d_masks_4d,
    position_ids = position_ids,
    past_key_values = None,
    inputs_embeds = [prefix_embs, suffix_embs],   ← 触发联合模式
    use_cache = False,
    adarms_cond = [None, adarms_cond],
)
  → 18层 compute_layer_complete (见第5节)
  → [prefix_output, suffix_output]

Step 5: 输出投影 + Loss (pi0_pytorch.py:403-412)
──────────────────────
suffix_out = suffix_output[:, -50:]         # 取最后 action_horizon 个 token
suffix_out = suffix_out.to(float32)
v_t = action_out_proj(suffix_out)           # [B, 50, 1024] → [B, 50, 32]
loss = MSE(u_t, v_t, reduction="none")      # [B, 50, 32], 未 reduce
```

---

## 9. 推理完整数据流

**入口**: `PI0Pytorch.sample_actions()` (`pi0_pytorch.py:414-464`)

```
Step 1: Prefix 编码 + KV Cache (一次性)
──────────────────────────────────────
embed_prefix(images, img_masks, lang_tokens, lang_masks)
  → prefix_embs [B, 768+L, 2048]

paligemma_with_expert.forward(
    inputs_embeds = [prefix_embs, None],     ← 触发 prefix-only 模式
    use_cache = True,
)
  → past_key_values: 18层 × (K, V)

Step 2: 迭代 Denoise (10 步, 从 t=1.0 到 t=0.0)
──────────────────────────────────────────────
x_t = noise [B, 50, 32]  (初始为纯噪声)
dt = -0.1

for time in [1.0, 0.9, 0.8, ..., 0.1]:
    v_t = denoise_step(state, prefix_pad_masks, past_key_values, x_t, time)
    x_t = x_t + dt * v_t    # Euler 积分

return x_t                   # 最终的去噪动作 [B, 50, 32]

Step 2.1: denoise_step 详解 (pi0_pytorch.py:466-512)
─────────────────────────────────────────────────────
embed_suffix(state, x_t, timestep)
  → suffix_embs [B, 51/50, 1024]
  → adarms_cond

# 构造 attention mask: suffix 对 prefix cache 可见
prefix_pad_2d = prefix_pad_masks[:, None, :].expand(B, Ss, Sp)
suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
full_att = cat([prefix_pad_2d, suffix_att_2d], dim=2)   # [B, Ss, Sp+Ss]

# position_ids 从 prefix 末尾继续
position_ids = sum(prefix_pad) + cumsum(suffix_pad) - 1

paligemma_with_expert.forward(
    inputs_embeds = [None, suffix_embs],        ← 触发 suffix-only 模式
    past_key_values = past_key_values,          ← 读取 prefix 的 KV cache
    use_cache = False,
    adarms_cond = [None, adarms_cond],
)
  → suffix_out [B, 50, 1024]

v_t = action_out_proj(suffix_out)               # [B, 50, 32]
```

---

## 10. Gradient Checkpointing

### 10.1 启用方式

**代码位置**: `pi0_pytorch.py:138-147`

```python
def gradient_checkpointing_enable(self):
    self.gradient_checkpointing_enabled = True
    self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
    self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
    self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
```

三个组件分别启用：VLM 语言模型、SigLIP 视觉塔、Action Expert。

### 10.2 在联合 forward 中的应用

**代码位置**: `gemma_pytorch.py:240-252`

```python
for layer_idx in range(num_layers):
    if use_gradient_checkpointing:
        inputs_embeds = torch.utils.checkpoint.checkpoint(
            compute_layer_complete,
            layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond,
            use_reentrant=False,
            preserve_rng_state=False,
        )
```

- `use_reentrant=False`: 使用新版 checkpoint 实现，更安全
- `preserve_rng_state=False`: 不保存 RNG 状态，节省内存
- 每个 transformer 层作为一个 checkpoint 单元

### 10.3 在 PI0Pytorch 中的应用

**代码位置**: `pi0_pytorch.py:164-170`

```python
def _apply_checkpoint(self, func, *args, **kwargs):
    if self.gradient_checkpointing_enabled and self.training:
        return torch.utils.checkpoint.checkpoint(
            func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
        )
    return func(*args, **kwargs)
```

用于包装以下操作：
- `embed_image` (line 220)
- `lang_embed_func` (line 236)
- `state_proj_func` (line 269)
- `action_proj_func` (line 295)
- `mlp_func` / `time_mlp_func` (line 307/317)
- `forward_func` (联合 forward, line 394)
- `action_out_proj_func` (line 410)

---

## 11. transformers 补丁总结

OpenPI 需要对 HuggingFace transformers 4.53.2 打 5 个补丁：

| 文件 | 改动要点 |
|------|---------|
| `gemma/configuration_gemma.py` | 新增 `use_adarms`, `adarms_cond_dim` 配置项 |
| `gemma/modeling_gemma.py` | RMSNorm → adaRMSNorm, gated_residual, KV cache 双模式, adarms_cond 透传, 注释掉 normalizer 缩放 |
| `paligemma/modeling_paligemma.py` | 删除 image_features 的 `/ sqrt(hidden_size)` 缩放 |
| `siglip/modeling_siglip.py` | embedding 后自动转 bfloat16 |
| `siglip/check.py` | 新增版本校验函数 |

**安装方式**:
```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

---

## 12. 缩放因子处理

原版 transformers 中有两处配对的缩放：
1. `get_image_features`: `image_features /= sqrt(hidden_size)` (PaliGemma)
2. `GemmaModel.forward`: `hidden_states *= sqrt(hidden_size)` (normalizer)

OpenPI 的处理方式：
- 补丁删除了 (1) (`modeling_paligemma.py`)
- 补丁注释掉了 (2) (`modeling_gemma.py`)
- 在 `embed_prefix` 中手动对语言 embedding 做 `* sqrt(dim)` (`pi0_pytorch.py:234`)

这样图像 embedding 不做任何缩放，语言 embedding 在外部手动缩放，
保证了两者在拼接时的数值量级匹配。
