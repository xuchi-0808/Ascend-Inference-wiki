# Attention O-Proj Module-Level TP 方案分析

## 背景

DeepSeek V4 的 MLA（Multi-head Latent Attention）中，output projection 由两个矩阵组成：

```
wo_a: [n_groups * o_lora_rank, n_heads * head_dim // n_groups]   # ColumnParallel
wo_b: [dim, n_groups * o_lora_rank]                                 # RowParallel
```

标准的 TP 直接将 wo_a/wo_b 当作普通 Linear 层做 column/row split，所有 rank 计算完整的 group 集合。而 **module-level TP** 的思路是：沿着 `n_groups` 维度将 wo_a 和 wo_b 的 weight 再切分，每个 rank 只计算自己负责的那部分 group，最后通过 all-gather / reduce-scatter 组合结果。

## 方案设计

### 关键改动

1. **`parallel_info_manager.py`**: 新增 `attn_o_proj_tp` 配置项，与 `tp` 解耦，允许独立控制 o_proj 的并行度

2. **`PerGroup` Linear 类**:
   - `ColumnParallelLinearPerGroup`：weight shape `[n_groups * o_lora_rank, hidden]` → 先 view 成 `[n_groups, o_lora_rank, hidden]`，在 `o_lora_rank` 维度上按 TP size 切分
   - `RowParallelLinearPerGroup`：weight shape `[dim, n_groups * o_lora_rank]` → 先 view 成 `[dim, n_groups, o_lora_rank]`，在 `o_lora_rank` 维度上按 TP size 切分

3. **通信插入**: 在 attention 计算路径上插入了两处通信操作
   - **Prefill & Decode 的 wo_a 之前**: `maybe_pad_and_maybe_all_gather` — 将各 rank 的 group 输出 all-gather 合并，还原完整的 group 维度
   - **SequenceRowParallelOp 的 reduce_scatter**: `maybe_reduce_scatter_and_unpad` — 将 wo_b 输出做 reduce-scatter，再 unpadding 回各 dp rank 的 token 数

4. **SequenceColumnParallelOp/SequenceRowParallelOp 路由**: 通过 prefix 匹配 `wo_a`/`wo_b` 来决定走 PerGroup 路径还是普通路径

### 数据流

```
# 标准 TP:
attn_out → [wo_a: matmul] → [wo_b: matmul + allreduce] → output

# Module-level TP (attn_o_proj_tp=2):
attn_out → [per-group split] → wo_a_matmul → [all_gather] → wo_b_matmul → [reduce_scatter] → output
```

## 收益分析

### 1. 显存节省

每个 rank 只加载 `1 / attn_o_proj_tp` 的 wo_a/wo_b weight，显存占用线性减少。

- 以 DeepSeek V4 为例，`n_groups=4, o_lora_rank=512, dim=7168`：
  - wo_a 参数量：`n_groups * o_lora_rank * (n_heads * head_dim // n_groups)`
  - wo_b 参数量：`dim * n_groups * o_lora_rank`
  - `attn_o_proj_tp=2` 时，每卡 weight 减半

### 2. 计算量分摊

matmul 的计算量也按 `1 / attn_o_proj_tp` 分摊。对于 o_proj 占比较高的大 batch 场景，能降低单卡计算延迟。

### 3. 灵活性

`attn_o_proj_tp` 与 `tp` 解耦，可以独立调节。例如 `tp=8, attn_o_proj_tp=2`，在保留 attention 整体并行度的同时，避免 o_proj 切太碎导致通信占比过高。

## 损失分析

### 1. 通信开销（核心瓶颈）

这是本方案**最终没有上库**的根本原因。

每层 attention 额外引入了 **2 次跨 rank 通信**：

| 位置 | 操作 | 通信量 |
|------|------|--------|
| wo_a 前 | all-gather | `n_local_groups * o_lora_rank * num_tokens` |
| wo_b 后 | reduce-scatter | `dim * num_tokens` |

对于 prefill 阶段，`num_tokens` 可能很大（数千到数万），通信量会非常可观。以 `num_tokens=8192, dim=7168, o_lora_rank=512, n_local_groups=4` 为例：

- all-gather 通信量（per rank）：约 `4 * 512 * 8192 * 2 bytes = 32 MB`
- reduce-scatter 通信量（per rank）：约 `7168 * 8192 * 2 bytes = 112 MB`

在 8 卡环境下，这两次 collectives 的延迟（尤其是跨节点时）很容易抵消甚至超过计算节省的时间。

### 2. Padding 开销

变长序列场景下，不同 dp rank 的 token 数不一致，需要在 all-gather 前做 padding、reduce-scatter 后做 unpadding。这引入了额外的显存拷贝和计算。

### 3. 代码复杂度

- 新增了两个 `PerGroup` Linear 子类和对应的 weight loader
- `linear_op.py` 中需要通过 prefix 匹配做条件路由，可维护性下降
- 与现有的 `SequenceColumnParallelOp`/`SequenceRowParallelOp` 逻辑交织，增加了排查问题的难度

### 4. 收益的上限低

对于 decode 阶段（batch size 通常较小，num_tokens 为 1 或很小），matmul 的计算量本身就很低，通信开销相对占比更大，几乎纯负面收益。

## 结论

| 维度 | 评估 |
|------|------|
| 显存 | ✅ 正向，weight 按比例减少 |
| 计算延迟（大 batch prefill） | ✅ 正向，matmul 量降低 |
| 计算延迟（小 batch decode） | ❌ 负面，通信开销 > 计算节省 |
| 扩展性 | ⚠️ 中等，受益于 attn_o_proj_tp 独立调节 |
| 工程复杂度 | ❌ 显著增加 |

**最终决策**：不在当前版本上库。主要原因是通信开销过大（尤其是跨节点的 all-gather/reduce-scatter），在大部分场景下收益无法覆盖成本。如果后续硬件拓扑改善（如节点内带宽大幅提升、NVLink 域扩展），或 MLA 中 o_proj 的计算占比随着模型规模增长而显著提高，可以重新评估此方案。
