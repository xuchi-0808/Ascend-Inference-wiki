# A00301 经验沉淀：Qwen3-235B-A22B-W8A8 精度问题定位全程复盘

> 项目编号：A00301
> 模型：Qwen3-235B-A22B-W8A8（A3，DP16 TP1，PD 分离）
> 时间跨度：2026-07-11 ~ 2026-07-20
> 最终修复：vllm-ascend PR #12371（backport of #12359）
> 参与人员：徐驰、王啸腾、陈波、工具组同事

---

## 一、问题现象

GSM8K 评测中部分请求出现**连续重复输出同一 token**（解码坍缩），表现为模型陷入循环输出。

---

## 二、定位过程时间线

### 阶段 1：PD 分离实验（07-11）——关键线索出现

| 实验                    | 结果         |
| ----------------------- | ------------ |
| PD 分离完整链路 badcase | **复现异常** |
| PD 分离 P 节点单独跑    | 不复现       |
| PD 分离 D 节点单独跑    | 不复现       |
| PD 分离 D 节点 eager    | **复现异常** |

> P 节点和 D 节点单独跑都正常，只有走完整 PD 链路才出问题。
> **这个信号直指 P→D 的 KV 传输环节，但当时没有被充分追问。**

### 阶段 2：fuse_norm_quant 方向（07-12）——误入歧途

通过开关四个 `ascend_compilation_config` 参数定位：

| 参数单独设为 true    | 结果                |
| -------------------- | ------------------- |
| `enable_npugraph_ex` | 正常                |
| `fuse_norm_quant`    | **异常** ← 锁定嫌疑 |
| `fuse_qknorm_rope`   | 正常                |
| `fuse_muls_add`      | 正常                |

随后发现 `div_mode` 差异（分离路径 `div_mode=False` 用乘法 `x * reciprocal`，融合路径 `div_mode=True` 用除法 `x / scale`），这种做法在底层可以认为是不够充分等价的，会引入误差，但其实这种问题应该不会导致那么严重的重复问题，花了一天左右的时间去看 badcase 为啥解决了，其实是做无用功。

**关键反转**：关闭 `fuse_norm_quant` 后，badcase 只是**"转移"**了（换个 prompt 仍然重复），并没有消除。

### 阶段 3：系统性排除（07-13 ~ 07-14）——收窄范围

| 实验                      | 结论                                                                                                           |
| ------------------------- | -------------------------------------------------------------------------------------------------------------- |
| BF16 全量数据集           | 仍有大量重复 → **排除 W8A8 量化**                                                                              |
| GPU H20 BF16 对比         | GPU 也有 ~10 条 "wait wait" 重复 → 模型自身问题，但是可以表明 wait wait 重复不算做 badcase，给问题定位划清界限 |
| v0.18.0                   | 已有此问题 → 问题是 wait wait 重复 → 可以认为非问题                                                            |
| cudagraph_capture_sizes=1 | 仍有重复 → **排除 padding**                                                                                    |
| enforce_eager             | 仍有重复 → **排除图模式以及 fusion pass 相关的所有特性**                                                       |

提炼出三个独立问题层级：

- **问题 C（模型层）**：GPU/NPU 都有的 "wait wait wait..."，模型自身特性，不修
- **问题 B（Ascend 精度层）**：NPU 独有的额外重复 → 真正要修的目标
- **问题 A（图模式特有）**：fuse_norm_quant 的 div_mode 差异 → 次要

### 阶段 4：KV cache 打点——Heisenbug 出现（07-15 ~ 07-17）

07-15 与王啸腾对方案，计划在 model runner 的 forward 前后分别加 D 节点和 P 节点的 hidden_states 打印。但由于 TP 不对等，改到 D 节点的 **Mooncake KVConnector** 里——在拉取 KV cache 之后加打印。

**意外发现**：加上打印后，精度问题**消失了**！这是一个经典的 Heisenbug——观察行为本身改变了被观察系统的行为。

07-17 继续排查具体是哪里的打印导致问题消失，最终定位到：**D 节点 Mooncake Connector 内部的打印**就能让问题消失。

> 为什么打印能修复 bug？后来根因清楚了：打印操作（如 `.shape`、CPU 同步等）无意中引入了**同步屏障**，改变了多线程 pull KV 的时序，碰巧让 reformat 延迟到了正确的时机。加打印 = 无意间加了同步 = 竞态被掩盖。

### 阶段 5：转交定位 + 代码分析（07-17 ~ 07-20）——根因落定

问题单转给王啸腾继续定位。期间拉工具的同事尝试各种方法 dump tensor，但工具有各种 bug，均告失败。

**最终突破口**：陈波和王啸腾**放弃 tensor dump 路线**，改为直接分析 PD 分离的代码逻辑，发现根因：

- **主要原因**：TP 不对等场景下，需要在 pull 完所有节点之后做一次 rerank（reformat）。当前代码实现 rerank 的时机不正确，导致 KV 发生乱序污染。
- **具体原因**：代码判断逻辑是"如果 `rank_id` 是最后一个，就做重排"。但此前做过**多线程 pull KV 的优化**后，无法保证最后一个 `rank_id` 一定是最后完成 pull KV 的那个线程 —— 竞态条件！
- **修复方法**：给多线程 pull KV 加了计数器，当计数器归零（所有 pull 都完成）后才触发重排。

对应 PR #12371（v0.23.0 backport）/ #12359（v0.25.1 原版）。

---

## 三、最终解法：PR #12371

### Bug 位置

`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`

### Bug 机制

TP 不对等场景下，一个 request 的 KV cache 需要从多个 TP rank 分别 pull。所有 rank 的数据到齐后，需要做一次 **rerank / reformat**（重排 KV block 的布局，让 attention 能正确读取）。

**原始代码的判断逻辑**：

```python
# _transfer_kv_cache_all_groups 内部
for reformat_group, is_group_transfer_end in attention_group_reformat_block_ids:
    if is_group_transfer_end:  # ← 判断"当前 rank 是不是最后一个"
        ready_attention_group_reformat_block_ids.append(reformat_group)

# 立即执行 reformat
self.reformat_kv_cache(ready_attention_group_reformat_block_ids)
```

**问题**：此前做了多线程 pull KV 的优化，每个 pull task 在独立线程中执行。`is_group_transfer_end` 判断的是"当前 `rank_id` 是否是最后一个 rank"，但**多线程下最后一个 rank 不一定是最后完成的**：

```text
Thread 1: pull rank 0  ──────────── done ✓
Thread 2: pull rank 1  ────── done ✓ (is_group_transfer_end=True)
                          ↑ 此时 rank 0 的数据还没到，但 reformat 已经执行了！
Thread 1: pull rank 0  ──────── done ✓ (数据到达，但 reformat 早做过了)
```

结果：reformat 在数据不完整时执行 → KV block 布局错乱 → attention 读到错位的 KV → 精度异常 → token 坍缩。

**打印为什么能"修复" bug**：打印操作引入了 CPU/NPU 同步，无意中延缓了触发 reformat 的线程，使 reformat 延迟到所有 pull 都完成后才执行——恰好是正确的时机。

### 修复方式

给多线程 pull KV 加了计数器（`_mark_request_task_done`），将 reformat 从"每个 pull task 后立即执行"改为"所有 pull task 完成后再执行"：

```python
# NEW: 先 stash，不立即 reformat
self._stash_pending_reformat(request_id, shard_idx, reformat_block_ids)

# 在 _handle_request 的 finally 块中，计数器归零（全部完成）后才统一执行
all_tasks_done = self._mark_request_task_done(request_id, all_task_done)
if all_tasks_done:
    self._reformat_pending_kv_caches(request_id)
```

新增了 `pending_reformat` 字典（按 request_id → shard_idx 索引）和配套的 `_stash_pending_reformat` / `_reformat_pending_kv_caches` 方法，确保 reformat 在所有 pull 完成后按 shard 顺序执行。

---

## 四、经验教训

### 教训 1：关键线索出现了但没有被充分追问

07-11 的 PD 分离实验已经证明"只有完整 PD 链路才出问题，P/D 单独跑都正常"。这是根因在 KV 传输环节的强信号。

**为什么会错过**：对 PD 分离架构的 KV 传输机制不够熟悉，没有意识到 P→D 之间还有 reformat 这一步。看到 `fuse_norm_quant` 的 `div_mode` 差异后，觉得找到了一个"技术上讲得通"的解释，就被吸引过去了。

**改进**：遇到"单独跑正常、组合跑异常"的现象时，应该优先排查**组合链路中独有的环节**（KV transfer、all-reduce、跨节点通信），而不是先看模型计算逻辑。

### 教训 2：相关性 ≠ 因果性

关闭 `fuse_norm_quant` 后某个特定 badcase 变好了，但全量数据集上仍有重复。

**正确解读**：该参数只是在**扰动精度分布**，碰巧让当前测试的 badcase 不再触发。badcase "转移"而非"消失" = 不是根因。

**改进**：关闭参数后如果 badcase 换了一批，应该立即意识到"这不是根因，只是改变了触发概率"。只有"关闭后全量数据集零重复"才能确认根因。

### 教训 3：单条 badcase 验证不可靠，拿它怀疑特性只会增加定位噪音

精度问题在整个数据集上的分布是不均匀的——有些 case 容易触发，有些不容易。单条 badcase 的复现/消失，可能只是精度分布的**随机扰动**，不代表任何因果关系。

**本案中的反面教材**：关闭 `fuse_norm_quant` 后，拿之前那条 badcase 去测，发现不重复了 → "确认是 fuse_norm_quant 的问题" → 花了大量时间走读融合链路代码、分析 `div_mode` ULP 误差。但实际上关掉它跑全量数据集，badcase 只是**换了一批**，总量没变。

**为什么单条 badcase 不可靠**：

- 精度是**统计现象**，不是确定性开关。改变任何一个参数（甚至不改变参数，只换一个 prompt），都可能让某条 badcase 碰巧不再触发
- 单条 badcase 的"好了"或"坏了"就像是**掷一次骰子**——信息量极低
- 唯一可靠的验证手段是**全量数据集对比**：看 badcase 总数、类型分布、score 变化

**改进**：

- 精度定位的**任何结论**都必须基于全量数据集，不能基于单条 badcase
- "关了某个特性，之前那条 badcase 不复现了" → 这**什么都不能说明**，不要以此为由去怀疑这个特性
- 正确流程：发现可疑特性 → 跑全量数据集对比 → badcase 总数显著下降才算有效 → 再深入分析
- 如果全量跑一遍成本太高（大模型全量数据集可能要几小时），至少跑一个**足够大的随机采样子集**（如 200+ 条），而不是只测 1~5 条 fixed badcase

### 教训 4：Heisenbug 是竞态条件的强信号

07-15 在 Mooncake Connector 里加打印后精度问题消失。这种"加个无害的打印就修好了 bug"是**多线程竞态条件**的典型症状——打印引入的同步屏障改变了线程调度的时序，碰巧掩盖了竞态。

**改进**：

- 遇到"加打印 / 加日志 / 调整优化级别后 bug 消失/出现"时，**第一反应应该是竞态条件**
- 不要试图通过"加打印"来"修复"问题——这只是掩盖了 bug，竞态在别的条件下还会复现
- 正确做法是分析代码的并发模型，找到共享状态和时序假设

### 教训 5：tensor dump 工具不可靠时，回归代码分析

定位期间，工具同事尝试各种方法 dump tensor 均失败（工具有各种 bug）。最终是陈波和王啸腾通过**直接分析 PD 分离的代码逻辑**发现根因的。

**改进**：

- tensor dump 是辅助手段，不是唯一手段。工具不可用时不要卡住
- 精度问题的根因最终都能在代码逻辑中找到解释。**代码分析 + 实验验证**的组合比纯靠工具 dump 更可靠
- 当一个方向（工具 dump）反复失败时，及时切换到另一个方向（代码静态分析），不要死磕

### 教训 6：多线程优化是精度 bug 的常见引入点

这个 bug 的直接原因是一次**多线程 pull KV 的优化**。优化前，单线程串行 pull 时，"最后一个 rank_id"天然是最后完成的，rerank 时机正确。多线程优化后，完成顺序不再确定，但 rerank 的触发逻辑没有同步更新。

**改进**：

- 做多线程/并发优化时，必须检查**所有依赖执行顺序的逻辑**
- "最后一个 rank = 最后完成"这种隐式假设，在串行代码里成立，在并发代码里是一个定时炸弹
- 代码 review 时，关注并发优化 PR 中是否有同步屏障的遗漏

### 教训 7：实验设计要锁定"唯一差异维度"

关键实验应该是：**同样的模型、同样的 prompt，唯一变量是"走不走 PD 分离的 KV 传输"**。如果走 PD 分离就出问题，不走就正常，那根因必然在 KV 传输路径上。

### 教训 8：分布式特有的 bug 要用分布式的方式定位

单机上的图模式/eager 实验无法暴露 KV 传输 bug，因为单机根本没有 KV 传输环节。

对分布式系统的精度问题，要考虑**哪些计算环节是分布式特有的**（KV transfer、all-reduce、MoE all2all 等），不能只看模型计算逻辑。

---

## 五、可复用的排查流程模板

```text
精度问题（重复 token / 输出乱码）

Step 1: 确定问题坐标系
  ├─ 单机 vs 分布式？        → 缩小到分布式独有环节
  ├─ 图模式 vs eager？       → 缩小到编译融合 pass
  ├─ BF16 vs W8A8？          → 缩小到量化路径
  └─ 哪个版本引入的？         → 回归 vs 长期暗伤

Step 2: 做最小差异实验
  ├─ 同模型同 prompt，只切一个变量
  └─ badcase "转移" ≠ 根因，badcase "消失" 才是

Step 3: 识别 Heisenbug 信号
  ├─ 加打印/日志后 bug 消失？  → 强烈指向竞态条件
  ├─ 调整优化级别后 bug 变化？ → 排查并发优化的时序假设
  └─ 不要用"加打印"当修复手段

Step 4: 按"分布式特有环节"优先排查
  ├─ KV transfer（PD 分离 / Mooncake connector）
  ├─ 跨节点通信（all-reduce / all2all）
  ├─ 多线程并发的共享状态和时序假设
  └─ 最后才是模型计算逻辑（norm / attention / quant）

Step 5: tensor dump 失败时回归代码分析
  ├─ 工具不可靠不要死磕
  └─ 代码静态分析 + 实验验证的组合更可靠

Step 6: 验证修复
  └─ 全量数据集零新增 badcase = 确认修复
```

---

## 六、相关文件索引

| 文件                           | 内容                                    |
| ------------------------------ | --------------------------------------- |
| `问题定位记录.md`              | fuse_norm_quant 方向的完整代码走读      |
| `个人维护的问题定位记录.md`    | 服务器实验原始记录（含 badcase 和命令） |
| `fuse_norm_quant_debug.md`     | div_mode 差异的详细分析                 |
| `findings.md`                  | 三层问题分离（C/B/A）的推演过程         |
| `progress.md` / `task_plan.md` | 阶段性进展和计划                        |
| PR #12371                      | 最终修复（Mooncake KV reformat 时序）   |
| PR #12359                      | 原版修复（v0.25.1）                     |
