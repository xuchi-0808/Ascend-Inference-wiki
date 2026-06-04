# DeepSeek V3.2 O 矩阵切分方案综合分析

> 分析范围：vllm-ascend `AscendSFAImpl`（SFA V1）及 MindIE-LLM DP 场景下的 O 矩阵切分方案
>
> 分析日期：2026-04-29

---

## 1. 背景

DeepSeek V3.2 采用 MLA（Multi-head Latent Attention）架构，其 O 投影矩阵的并行策略在不同推理框架和部署场景下存在显著差异。本文档系统性地分析了两个框架中 O 矩阵相关的方案设计，为后续决策提供依据。

### 1.1 DeepSeek V3.2 MLA 关键参数

| 参数 | 符号 | 值 | 说明 |
|------|------|-----|------|
| `hidden_size` | H | 7168 | 模型隐层维度 |
| `num_attention_heads` | N_total | 128 | 全局 attention head 数 |
| `kv_lora_rank` | L | 512 | KV 压缩隐层维度 |
| `qk_nope_head_dim` | P | 128 | Q/K 不走 RoPE 的维度 |
| `qk_rope_head_dim` | R | 64 | Q/K 走 RoPE 的维度 |
| `v_head_dim` | V | 128 | V head 维度 |
| `q_lora_rank` | Lq | 1536 | Q 压缩隐层维度 |

### 1.2 O 矩阵计算流程

MLA 中 O 矩阵的计算分三个阶段：

```
1. Sparse Flash Attention 输出
   attn_output: (T, N, L)           -- latent 空间的 attention 加权和

2. V Up Projection（W_UV）
   bmm × (N, L, V) → (T, N, V)
   flatten → (T, N*V)

3. O Projection
   (T, N*V) × o_proj → (T, H)      -- 映射回 hidden_size
```

- T：token 数，N：head 数，L：kv_lora_rank=512，V：v_head_dim=128，H：hidden_size=7168

---

## 2. vllm-ascend SFA V1 的 O 矩阵方案分析

> **参考代码**：`vllm-ascend/vllm_ascend/attention/sfa_v1.py`
>
> **分析文档**：`vllm-ascend/SFA_V1_O_Matrix_Analysis.md`

### 2.1 核心矛盾：DSA-CP 引入的 Shape 不匹配

SFA V1 引入 DSA-CP（Data Split Attention - Context Parallelism）后，V Up Projection 的输出来到了一个中间态：

```
DSA-CP 输出: (T_local, N_total * V) = (T_local, 16384)  ← 少 token × 全 head (CP 态)
o_proj 期望: (T_total, N_per_rank * V) = (T_total, 4096) ← 全 token × 分 head (TP 态)
```

所有四种 O 矩阵策略，本质上都是在解决 **CP 态到 TP 态的转换**。

### 2.2 四种场景的 O 矩阵策略

| 场景 | 模式 | O 矩阵策略 | 通信原语 | 适用条件 |
|------|------|-----------|---------|---------|
| PD 分离 P 节点 | Prefill-only | **Layer Sharding**（广播权重） | broadcast | T 大，计算掩盖通信 |
| PD 分离 D 节点 | Decode-only | **不开 DSA-CP**（标准 TP） | all-reduce | T 小，源头消解问题 |
| PD 混部 Prefill | Prefill + Decode 混部 | **全量权重 All-Gather** | all-gather (weight) | T 大，双流并行掩盖 |
| PD 混部 Decode | Prefill + Decode 混部 | **Activation All-to-All** | all-to-all + all-reduce | T 极小，通信量压倒性优势 |

**选择逻辑的核心变量：单卡 token 数 T_local 的大小**

- T 大（Prefill：数百~数千）→ compute-bound，通信可掩盖 → 优先选 weight-gather / broadcast
- T 小（Decode：1~数十）→ memory-bound，通信不可掩盖 → 优先选 activation 通信

### 2.3 关键设计：PD 混部 Decode 的 All-to-All

Decode 阶段 T_local 极小（1~8），若直接套用 Prefill 的权重 all-gather 策略：

```
权重 all-gather: 117M elms ≈ 234MB（无论 T 大小，权重固定）
activation all-to-all: T_local × tp × N_per_rank × V ≈ 1 × 4 × 4096 × 2 ≈ 66KB
```

**234MB vs 66KB，差了约 3500 倍。** All-to-All 做的是 CP→TP 重分布：

```
all-to-all 前 (CP 态): (T_local, N_total * V) = (T_local, 16384)  → 本地 token × 全量 head
all-to-all 后 (TP 态): (T_total, N_per_rank * V) = (T_total, 4096) → 全量 token × 本卡 head
```

> **为什么 PD 混部 decode 不关 DSA-CP？** 因为 DSA-CP 在模型 init 时固化，运行时无法在 prefill 和 decode 之间动态切换。

### 2.4 通信量量化对比

| 策略 | 单层通信量 | 能否被计算掩盖 | 额外显存 |
|------|-----------|---------------|---------|
| DSA-CP 不开（D 节点） | 0 | — | 0 |
| Layer Sharding（P 节点） | ~29M elms ≈ 58MB | 可以（prefetch） | 省 3/4 o_proj |
| 全量权切 All-Gather（混部 Prefill） | ~117M elms ≈ 234MB | 可以（双流并行） | 234MB |
| Activation All-to-All（混部 Decode） | ~T_local × 32K elms ≈ 66KB | 不需要（量极小） | 0 |

### 2.5 设计哲学总结

```
                    ┌─ Prefill-only (T大) ──→ Layer Sharding
                    │   (通信可掩盖 + 显存节省优先)
PD 分离 ────────────┤
                    │
                    └─ Decode-only (T小) ──→ 不开 DSA-CP
                        (源头消解, 成本最低)

                    ┌─ Prefill (T大) ──→ 全量权重 All-Gather
                    │   (通信可掩盖 + 显存允许)
PD 混部 (DSA-CP=ON) ┤
                    │
                    └─ Decode (T小) ──→ Activation All-to-All
                        (通信量 1/3500, 不可掩盖就做最少)
```

---

## 3. MindIE-LLM DP 场景的方案验证

### 3.1 方案背景

在纯 DP（Data Parallelism）场景下推理 DeepSeek V3.2 时，有方案提出对 O 矩阵进行列切分以节省每卡权重显存。

当前实现（`mindie_llm/runtime/models/deepseek_v3/deepseek_v3.py:238-246`）：

```python
self.o_proj = RowParallelLinear(
    self.num_heads * self.qk_nope_head_dim,  # 16384
    config.hidden_size,                       # 7168
    bias=False, reduce_results=True,
)
```

纯 DP 下每卡持有完整 O 矩阵 `[16384, 7168]` ≈ 117M 参数，独立计算，无通信。

### 3.2 方案描述

**前提条件**：MLA 的 V 投影后各 DP 卡 token 长度对齐（local batch size 相同）。

**原始思路**：

1. 将 V 投影后的激活值视为**行切**结果（各卡持有不同 token 的完整激活值）
2. O 矩阵改为**列切**方式加载，每卡只存 `[D, H/n]`
3. 激活值 × 列切 O 矩阵 → 局部 matmul
4. 补一个 ReduceScatter 得到正确结果

**直觉来源**：类比 TP MLP 中 W1 列切 → 激活值 → W3 列切 → AllReduce 的模式，认为 DP 组的输出可以视为"天然的 TP 行切"。

### 3.3 验证分析：为什么不可行

#### 核心区别：分散 ≠ 分片

| 维度 | TP 场景 | DP 场景 |
|------|---------|---------|
| 激活值 | 卡间**互补**（各持特征维的 1/n） | 卡间**独立**（各持不同 batch 的完整特征） |
| 数学关系 | 真分片，Reduce 可重建 | 无分片关系，无法通过通信重建 |
| AllReduce 语义 | 求和跨卡互补部分 → 完整结果 | 不可用 |

#### 数学推导

设 n 张 DP 卡，每卡激活值 `A_i [B, D]`（D=16384），列切 O `W_i [D, H/n]`。

每卡计算 `A_i @ W_i → [B, H/n]` 得到的是**自己的 token × 自己的列分片**这一子块。但每卡实际需要 `[B, H]` = `concat(A_i @ W_0, A_i @ W_1, ..., A_i @ W_{n-1})`。

以 2 卡为例：

| 卡 | 计算 | 得到 |
|---|---|---|
| 卡 0 | A_0 @ W_0 | token 0..B-1 的 列 0..H/n-1 |
| 卡 1 | A_1 @ W_1 | token B..2B-1 的 列 H/n..H-1 |

卡 0 需要的是 `A_0 @ W_full = [A_0 @ W_0, A_0 @ W_1]`，即它自己的 token × 全部列。但 `A_0 @ W_1` 没有任何卡计算过——卡 1 算的是 `A_1 @ W_1`（不同 token）。

**ReduceScatter 不能解决列缺失的问题，因为缺失的列数据从未被任何卡产生。**

### 3.4 关键的直觉误区

方案提出者把 DP 场景类比为 MLP TP 方案中的切分模式，但两者有本质区别：

| 维度 | vllm SFA V1 场景 | MindIE DP 场景 |
|------|------------------|---------------|
| 激活值的卡间关系 | CP → TP 的状态转换（同一个激活值的分布形态变化） | 各卡独立自治（不同 batch 的完整激活值） |
| 变换前 | (T_local, 16384) 同批 token 的完整特征 | (B_0, 16384) 和 (B_1, 16384) 无关联 |
| 变换后 | (T_total, 4096) 分布重组后恰好匹配 TP 输入 | 无法通过集体通信重建 |
| 可行方案 | All-to-All 重分布 / AllGather 权重 | 只能 ZeRO 风格 AllGather 权重 |

**说人话**：vllm 那边能做成，是因为 DSA-CP 这个"上下文并行"层本来就在做数据重分布，激活值在 CP→TP 转换中**从不同视角看同一个东西**。DP 这边每卡的数据是**完全分家**的，不存在需要拼合的数学关系。

---

## 4. vllm 经验对 MindIE DP 场景的借鉴意义

### 4.1 可以借鉴的设计思想

| 思想 | vllm 实现 | 是否适用于 MindIE DP |
|------|----------|-------------------|
| **Layer Sharding**：层权重按 rank 轮流存储，计算时 broadcast | P 节点 Prefill | 可借鉴。需通信系统支持 async broadcast。但 DP 下每卡要算完整的模型层，广播收益不大 |
| **全量权重切换**：Prefill 权重 AllGather + Decode 切回 TP 权重 | 混部 Prefill | DP 场景没有切换需求（一直用同一套权重），参考价值有限 |
| **Activation All-to-All**：用 activation 通信避免 weight 通信 | 混部 Decode | **关键启发**：当 batch 小时应避免 weight 通信，激活值通信量更可控 |
| **不切**：源头消解问题 | D 节点不开 DSA-CP | 当前 DP 方案就是"不切"-- 这是最正确的选择 |

### 4.2 核心启发

vllm 经验给我们的最大启发是：**在一层之内做 weight 切分需要额外的通信，收益是否值得取决于 T_local 的大小**。

对于 MindIE 的 DP 场景：
- 如果确实是纯 DP（tp=1，每卡独立处理不同请求），O 矩阵列的切分**数学上不成立**，根本不能做
- 如果是 DP + TP 混合场景，O 矩阵走 RowParallelLinear（当前已实现）是正确的
- 如果未来要在 DP 场景下优化 O 矩阵显存，ZeRO 风格的 AllGather 权重是唯一可行的理论路径

---

## 5. 结论

### 5.1 方案验证结论

**MindIE-LLM DP 场景下 O 矩阵列切方案：不做。**

根本原因：数学上不成立。DP 各卡的激活值是"独立自治"的，而非"数学分片"的，无法通过集体通信原语重建正确结果。

### 5.2 vllm SFA V1 方案评价

vllm-ascend 的 SFA V1 O 矩阵策略设计完整，覆盖了 PD 分离/混部的四种场景。核心思路围绕一个变量（T_local 的大小）展开，选择权重通信或激活值通信，逻辑自洽。

### 5.3 后续可能的优化方向

| 方向 | 说明 | 优先级 |
|------|------|--------|
| 维持当前纯 DP 方案 | 不做 O 矩阵切分，保持完整权重加载 | —（当前方案） |
| ZeRO 风格权重 AllGather | 各 DP 卡分片存权重，计算前 AllGather 重建，适合多层级联场景 | 低（O 矩阵仅 117M 参数/卡，节省有限） |
| vllm Layer Sharding 思路 | 在有 TP 的场景下可参考，DP 场景不适用 | 低 |

---

## 附录：关键文件索引

### MindIE-LLM

| 文件 | 说明 |
|------|------|
| `mindie_llm/runtime/models/deepseek_v3/deepseek_v3.py` | DeepSeekV3 模型定义，包含 o_proj 为 RowParallelLinear |
| `mindie_llm/runtime/layers/linear/linear.py` | RowParallelLinear / ColumnParallelLinear 实现 |
| `mindie_llm/runtime/layers/linear/linear_op.py` | SequenceRowParallelOp（reduce_scatter） |
| `mindie_llm/runtime/utils/distributed/parallel_info_manager.py` | 并行组管理（含 ATTN_O_PROJ_TP） |

### vllm-ascend

| 文件 | 说明 |
|------|------|
| `vllm_ascend/attention/sfa_v1.py` | SFA V1 核心实现，含 O 矩阵切换逻辑 |
| `vllm_ascend/ops/linear_op.py` | 所有并行线性 Op（ShardedCP/OProj/Flashcomm2/Sequence） |
| `vllm_ascend/ops/layer_shard_linear.py` | Layer Sharding 机制 |
| `vllm_ascend/utils.py` | enable_dsa_cp/layer_shard/o_proj_tp 判定函数 |
| `vllm_ascend/distributed/parallel_state.py` | 通信组管理（otp_group / flashcomm2 组） |
| `vllm_ascend/attention/context_parallel/sfa_cp.py` | 超长序列 CP+TP 实现 |
| `vllm_ascend/ops/mla.py` | AscendMultiHeadLatentAttention 包装器 |
