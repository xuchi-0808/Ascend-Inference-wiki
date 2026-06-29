# lm-head TP 源码追踪与原理分析

> **文档日期**：2026-06-29
> **代码版本**：vllm-ascend
> - 改造前基线：`fe1b1d77`（2026-06-23，上游 main）
> - 本文档追踪行号基于改造后 commit：`eecb0d32`（2026-06-29，PR [#11086](https://github.com/vllm-project/vllm-ascend/pull/11086)）
> - 行号可能随上游演进漂移，追踪时以函数名/关键代码片段为准

## 背景

lm-head TP（`lmhead_tensor_parallel_size`）是 vllm-ascend 的 fine-grained TP 特性之一，用于 **MoE 模型** 场景。它把 lm_head（最后的 hidden→vocab 输出层）的权重沿 vocab 维切分到多个 rank 上，节省显存与访存，代价是多一次 `all_gather(hidden)` + 一次 `all_to_all(logits)`。配置校验要求 MoE 模型且 `lmhead_tp_size` 整除 `dp_size`；代码注释写着 "pure dp scenario"，但实测表明它与 TP 正交、可与 `DP+TP` 叠加（详见下文「正交性」一节）。

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

> 注：代码注释提到 "pure dp scenario"，但**未强制要求 `tensor_parallel_size == 1`**。实测证明 lm-head TP 与 TP 正交、可叠加（见下文「正交性」一节），所以这里不拦 TP>1 是合理的。

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

## lm-head TP 与 TP 的正交性（DP+TP+lmhead_tp 叠加）

代码注释（`vocab_parallel_embedding.py:278`）写着 "pure dp scenario"，这容易让人误以为 lm-head TP 只能在 `tensor_parallel_size=1` 下使用。**实际并非如此**——lm-head TP 与 TP 是正交的，可以叠加。这一节解释为什么正交，并给出实测验证。

### 为什么正交：两步通信的净效果

容易让人误判的点在于 `_get_logits_lmheadtp` 的第一步 `all_gather(hidden)`：它把 lm-head TP 组内各 rank 的 hidden 拼起来，看起来"把不同 batch 的 token 混在一起了"。但**必须结合紧随其后的 `all_to_all(logits)` 一起看**，净效果是：

```text
rank0 持 [N, V/P]  (batch A × 自己的 vocab 切片)
   │ all_gather(hidden) on lm-head TP group
   ▼ [P·N, V/P]   ← P 个 batch (A,B,...) × vocab 切片（中间过渡态）
   │ lmhead_all_to_all: dim0 切 P 份（按 batch 切回） + dim-1 拼 P 份（拼满 vocab）
   ▼ [N, V]        ← batch A × 全 vocab  ← 回到自己负责的 batch
```

`all_gather` 把 token 维放大（收 P 个 batch），`all_to_all` 又把 token 维缩小（每 rank 留 1 个 batch）+ vocab 维放大（拼满）。**净效果就是 `[N, V/P] → [N, V]`，每个 rank 最终拿到的还是自己那个 batch 的完整 logits。** 整条通信在 lm-head TP 组内闭环，与 TP 组互不干扰。

> ⚠️ **教训**：分析张量通信语义时不能只看链路的一段。本文档的早期版本只看了 `all_gather` 就断言"TP>1 会语义断裂"，是错的——必须看 `all_gather + all_to_all` 的完整闭环。

### 实测验证（2026-06-29）

在 S3（A3，16 chip）上用 Qwen3-30B-A3B（MoE，bf16，`enforce_eager`）做了精度对比：

| 组 | 配置 | 卡数 |
|---|---|---|
| 对照 | DP4 + TP2（不开 lmhead_tp） | 8 |
| 实验 | DP4 + TP2 + lmhead_tp2 | 8 |

同一批 5 个 prompt，`temperature=0`（确定性输出），对比两组输出文本：

```text
[0] ✓ The capital of China is
[1] ✓ Write a Python function to compute fibonacci:
[2] ✓ Explain what tensor parallelism is in one sentence.
[3] ✓ Translate to English: 今天天气真好
[4] ✓ 1+1=
VERDICT: ALL MATCH — lmhead_tp + TP 精度无损，正交性成立 ✓
```

**5/5 逐字符完全一致**。证明 `DP + TP + lmhead_tp` 叠加后精度无损，lm-head TP 与 TP 正交。

> 实验踩坑：A3 机器多卡 graph capture 会报 `no notify resource`（错误码 207009，notify 资源耗尽）。精度验证场景用 `--enforce-eager` 绕过，且 eager 启动反而更快（~110s vs graph 模式 ~10min）。

## 追踪建议

**第一遍（纵向，顺数据流）**：配置（§1-3）→ 组创建（§4-5）→ 权重切分（§6）→ 前向（§7-9）。

**第二遍（重点突破）**：聚焦三个关键决策点：

1. `parallel_state.py:117-129` —— 分组怎么从 DP 维切（理解 lm-head TP 与 DP/TP 的关系）
2. `vocab_parallel_embedding.py:72` —— `self.tp_size` 被重定义（理解权重切分逻辑）
3. `vocab_parallel_embedding.py:299-303` —— `all_gather(hidden)` + `all_to_all(logits)` 的完整闭环（理解为什么与 TP 正交，务必看两步合起来的净效果）

**第三遍（对比学习）**：对比 `_get_logits_lmheadtp`（L292）与 `_get_logits_normal`（L313），看 lm-head TP 路径多了哪些通信、为什么。
