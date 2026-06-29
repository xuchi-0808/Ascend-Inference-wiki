# lm-head TP 源码追踪与原理分析

> **文档日期**：2026-06-29
> **代码版本**：vllm-ascend
> - 改造前基线：`fe1b1d77`（2026-06-23，上游 main）
> - 本文档追踪行号基于改造后 commit：`eecb0d32`（2026-06-29，PR [#11086](https://github.com/vllm-project/vllm-ascend/pull/11086)）
> - 行号可能随上游演进漂移，追踪时以函数名/关键代码片段为准

## 背景

lm-head TP（`lmhead_tensor_parallel_size`）是 vllm-ascend 的 fine-grained TP 特性之一，专门用于 **pure DP（`tensor_parallel_size=1`）+ MoE 模型** 场景。它把 lm_head（最后的 hidden→vocab 输出层）的权重沿 vocab 维切分到多个 rank 上，节省显存与访存，代价是多一次 `all_gather(hidden)` + 一次 `all_to_all(logits)`。

本仓库 PR [#11086](https://github.com/vllm-project/vllm-ascend/pull/11086) 把原来通过 monkey-patch 实现的 `GroupCoordinator.all_to_all` 移除，改用 `torch.distributed.all_to_all_single` 直连实现。本文档记录该 PR 之后的完整代码路径，便于后续维护与学习。

## 配置与启用

### 1. 配置定义 —— `vllm_ascend/ascend_config.py`

```python
# L442
self.lmhead_tensor_parallel_size = finegrained_tp_config.get("lmhead_tensor_parallel_size", 0)
```

通过 `--additional-config '{"finegrained_tp_config":{"lmhead_tensor_parallel_size":K}}'` 传入。

### 2. 启用判断 —— `vllm_ascend/utils.py`

```python
# L819
def lmhead_tp_enable() -> bool:
    return get_ascend_config().finegrained_tp_config.lmhead_tensor_parallel_size > 0
```

`> 0` 即启用。消费者侧到处调这个函数判断是否走 lm-head TP 路径。

### 3. 配置校验 —— `vllm_ascend/ascend_config.py`

```python
# L487-490
if module_tp_size > 0 and not vllm_config.model_config.is_moe:
    raise AssertionError("The lmhead parallel feature can be enabled only for MOE models.")
if module_tp_size > 0 and data_parallel_size % module_tp_size != 0:
    raise AssertionError("lmhead_tensor_parallel_size must divide by data_parallel_size.")
```

两个硬约束：

- **必须是 MoE 模型**（dense 模型会被拦下）
- **`lmhead_tp_size` 必须整除 `data_parallel_size`**（注意校验的是 DP 维度，不是 TP 维度）

> ⚠️ **当前缺失的校验**：代码没有校验 `tensor_parallel_size == 1`。lm-head TP 在 TP>1 场景下会有语义问题（见下文「TP>1 场景的语义断裂」一节），但配置层并未拦截。

## 通信组创建

### 4. `_create_or_get_group` —— `vllm_ascend/distributed/parallel_state.py`

这是理解 lm-head TP 与 DP/TP 关系的**核心函数**。

```python
# L117-133
def _create_or_get_group(group_size: int, group_name: str) -> GroupCoordinator:
    if group_size not in _group_cache:
        rank_grid = torch.arange(world_size).reshape(global_pp_size, global_dp_size, global_tp_size)
        #                         ↑ 三维网格 [PP, DP, TP]
        num_chunks = global_dp_size // group_size
        group_ranks = []
        for pp_idx in range(global_pp_size):
            stage_ranks = rank_grid[pp_idx]          # (dp, tp)
            for chunk in range(num_chunks):
                for tp_idx in range(global_tp_size):
                    # 切的是 DP 维的行，TP 维固定一个列
                    group = stage_ranks[chunk * group_size : (chunk + 1) * group_size, tp_idx].tolist()
                    group_ranks.append(group)
        pg = init_model_parallel_group(group_ranks, ...)
        _group_cache[group_size] = pg
    return _group_cache[group_size]
```

**关键洞察**：

- lm-head TP 组是**从 DP 维度切出来的**（`chunk * group_size` 切的是 `stage_ranks` 的行 = DP 维）
- 每个 TP 列（`tp_idx`）独立成组，组内只含同一个 `tp_idx` 的 rank
- `num_chunks = dp_size // group_size`，决定每个 TP 列内能切出多少个 lm-head TP 组

### 5. 初始化与获取

```python
# L144-145：启动时创建
if lmhead_tp_size > 0:
    _LMTP = _create_or_get_group(lmhead_tp_size, "lmheadtp")

# L248-250：运行时获取
def get_lmhead_tp_group() -> GroupCoordinator:
    assert _LMTP is not None, "lm head tensor parallel group is not initialized"
    return _LMTP
```

## 权重切分

### 6. `AscendParallelLMHead` —— `vllm_ascend/ops/vocab_parallel_embedding.py`

lm-head TP 启用时，`self.tp_size` 被**重定义**为 lm-head TP 组的 world_size：

```python
# L64-72：选择 comm_group
if lmhead_tp_enable() and "head" in prefix:
    self.comm_group = get_lmhead_tp_group()    # ← lm-head 用 lm-head TP 组
elif embedding_tp_enable() and "embed_tokens" in prefix:
    self.comm_group = get_embed_tp_group()
else:
    self.comm_group = get_tp_group()

self.tp_size = self.comm_group.world_size      # ← 注意：这里被重定义了！
self.tp_rank = self.comm_group.rank_in_group

# L118：vocab 维按 self.tp_size 切分
self.num_embeddings_per_partition = divide(self.num_embeddings_padded, self.tp_size)
```

**这是最关键的语义点**：`self.tp_size` 不是 vllm 原生的 `tensor_parallel_size`，而是 **lm-head TP group 的 world_size**（= `lmhead_tensor_parallel_size`）。vocab 权重据此切分，每卡持 `V / lmhead_tp_size` 列。

## 前向计算路径

### 7. logits 处理路由 —— `vllm_ascend/ops/vocab_parallel_embedding.py`

```python
# L281-290
def _get_logits(self, hidden_states, lm_head, embedding_bias):
    if lmhead_tp_enable():
        return self._get_logits_lmheadtp(...)    # ← 启用走 lm-head TP 路径
    else:
        return self._get_logits_normal(...)
```

### 8. `_get_logits_lmheadtp` —— 核心前向

```python
# L292-311
def _get_logits_lmheadtp(self, hidden_states, lm_head, embedding_bias):
    # Step 1: 在 lm-head TP 组内 all_gather hidden states
    gathered_hidden_states = get_lmhead_tp_group().all_gather(hidden_states, dim=0)
    # → [N_total, H]，每卡拿到全部 token 的 hidden

    # Step 2: 本地 matmul，每卡用自己的 V/P 列权重
    logits = lm_head.quant_method.apply(lm_head, gathered_hidden_states, bias=embedding_bias)
    # → [N_total, V/P]，全 token × 部分 vocab

    # Step 3: all_to_all 重分布成 [N/P, V]
    if not get_ascend_config().enable_reduce_sample:
        logits = lmhead_all_to_all(logits, get_lmhead_tp_group())

    # Step 4: 去掉 vocab padding
    ...
```

**完整数据流**：

```text
hidden [N, H]
   │ all_gather(dim=0) on lm-head TP group
   ▼
[N_total, H]                      # 全 token
   │ × lm_head_weight[V/P, H]ᵀ    # 每卡持 V/P 列
   ▼
[N_total, V/P]                    # 全 token × 部分 vocab
   │ lmhead_all_to_all            # dim0 切(dim0 切 token) + dim-1 拼(拼 vocab)
   ▼
[N/P, V]                          # 部分 token × 全 vocab
```

### 9. `lmhead_all_to_all` —— 重分布实现

PR [#11086](https://github.com/vllm-project/vllm-ascend/pull/11086) 的核心改动，位于 `vocab_parallel_embedding.py:243-272`：

```python
def lmhead_all_to_all(logits, comm_group):
    world_size = comm_group.world_size
    if world_size == 1:
        return logits
    vocab_per_partition = logits.shape[-1]
    # [N, V/P] → view(P, N/P, V/P)：让 dim0 携带 per-rank token shard
    input_ = logits.contiguous().view(world_size, -1, vocab_per_partition)
    output = torch.empty_like(input_)
    dist.all_to_all_single(output, input_, group=comm_group.device_group)
    # [P, N/P, V/P] → permute(1,0,2) → [N/P, P, V/P] → view(N/P, V)
    return output.permute(1, 0, 2).contiguous().view(-1, world_size * vocab_per_partition)
```

**为什么不能直接用 `all_to_all_single`？** `all_to_all_single` 默认只在 dim 0 上又切又拼，无法表达「dim0 切 token + dim-1 拼 vocab」的非对称变换。`view(P, N/P, V/P)` 把 dim0 重塑成 `[P, N/P]`，让 single 能在 P 维正确路由；`permute(1,0,2)` 再把各 rank 的 vocab 分片按 token 交错拼接。详见 PR 描述与 review 讨论。

## spec_decode 路径（MTP proposer）

### 10. `_run_merged_draft` —— `vllm_ascend/spec_decode/llm_base_proposer.py`

MTP（Multi-Token Prediction）draft model 在启用 lm-head TP 时也会走这条路径，共两个调用点：

```python
# L1054：进入 lm-head TP 块
if lmhead_tp_enable():
    ...
    # L1077（mrope 分支）& L1233（普通分支）
    logits = self.model.compute_logits(sample_hidden_states)
    if lmhead_tp_enable():
        logits = lmhead_all_to_all(logits, get_lmhead_tp_group())
```

两个分支逻辑相同，仅缩进不同（mrope 分支 20 空格，普通分支 24 空格）。

## TP>1 场景的语义断裂

lm-head TP 设计为 **pure DP 场景**（代码注释 `vocab_parallel_embedding.py:278` "pure dp scenario"）。当 `tensor_parallel_size > 1` 时会有语义问题：

**核心矛盾**：lm-head TP 组里的 rank 来自**不同的 DP 副本**（不同 batch）。在 pure DP 下这是对的——每个 rank 持不同 batch 的 token。但 TP>1 时，同一个 TP 组的 rank（如 rank0 和 rank1）持有**同一个 batch** 的 hidden（经 TP all_reduce 对齐），而它们的 lm-head TP 组却不同：

```text
DP4 + TP2 + lmhead_tp4，8 卡：
  rank_grid (dp=4, tp=2):
          tp0   tp1
    dp0  [ 0,    1 ]   ← rank0/1 同一 TP 组，同一 batch
    dp1  [ 2,    3 ]
    dp2  [ 4,    5 ]
    dp3  [ 6,    7 ]

  lmhead_tp4 分组：
    {0,2,4,6}  ← 全是 tp0 列
    {1,3,5,7}  ← 全是 tp1 列
```

`_get_logits_lmheadtp` 的 `all_gather(hidden)` 在 lm-head TP 组 `{0,2,4,6}` 内做，拼起来的是 4 个不同 batch 的 hidden——而 rank1 不在这个组里。这导致 rank0 和 rank1（同一 TP 组）算出的 logits 对应的 batch 集合不一致，后续 TP 采样对不上。

**结论**：当前代码不拦 TP>1，但实际跑会有正确性问题。这是一个潜在的配置校验改进点。

## 追踪建议

**第一遍（纵向，顺数据流）**：配置（§1-3）→ 组创建（§4-5）→ 权重切分（§6）→ 前向（§7-9）。

**第二遍（重点突破）**：聚焦三个关键决策点：

1. `parallel_state.py:117-129` —— 分组怎么从 DP 维切（理解 lm-head TP 与 DP/TP 的关系）
2. `vocab_parallel_embedding.py:72` —— `self.tp_size` 被重定义（理解权重切分逻辑）
3. `vocab_parallel_embedding.py:299` —— `all_gather` 在哪个组做（理解 TP>1 时为什么语义断裂）

**第三遍（对比学习）**：对比 `_get_logits_lmheadtp`（L292）与 `_get_logits_normal`（L313），看 lm-head TP 路径多了哪些通信、为什么。
