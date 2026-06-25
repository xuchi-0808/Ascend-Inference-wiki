# A00282 — Qwen3-235B-A22B 多并发精度回归复盘

> 定位时间：2026-05-10 ~ 2026-06-25
> 模型：Qwen3-235B-A22B-A2（MoE, 235B 总参 / 22B 激活, 128 routed experts, 0 shared）
> 类型：推理精度回归 / MoE graph capture 静态化
> 关键词：graph capture / property 静态化 / MC2 / AllGather / custom op

---

## 1. 问题概述

### 问题现象

Nightly-A2 流水线 GSM8K accuracy 从 96.875%（5/9）骤降至 0%（5/10），持续归零。

详细症状：

| 检测维度 | 结果 | 说明 |
|---------|------|------|
| curl 单请求 | 正常 | 单并发 short prompt 正确 |
| ais_bench 32 并发 | 全乱码 | 多并发 prefill 时输出完全错乱 |
| perf(throughput) | 通过 | 吞吐量指标正常，无性能退化 |
| NCCL/NaN/crash | 无 | 无运行时异常或崩溃 |

核心特征：**单并发正常，多并发乱码**。root cause 必须能解释为什么并发数影响精度。

### 版本基线

| 场景 | vllm | vllm-ascend | 精度 |
|------|------|-------------|------|
| 5/9 成功 | d886c26d4 | 68ff5263 | 96.875% |
| 5/10 失败 | 4d51588e2 | ca4065f2 | 0.0% |
| **修复** | 4d51588e2 | **A00282_E (a186705c)** | **恢复** |

### 环境

| 项目 | 配置 |
|------|------|
| 镜像 | 2026-04-28（CANN 8.5.1 + 驱动 25.5.0 C23） |
| Node 0 | 80.48.37.227（Leader） |
| Node 1 | 80.48.37.228（Worker / Headless） |
| 权重 | /mnt/share/weights/Qwen3-235B-A22B-W8A8 |
| 网卡 | enp189s0f0（RoCE 跨机互联） |
| 分布式 | 2 节点 x 8 NPU，TP=8, DP=2, EP=16 |

---

## 2. 根因分析（5 步详解）

### Step 1：MoE 架构重构

vllm-ascend `7fd2cede`（PR #8899）适配了 vllm upstream 的 MoE refactor（#35782 / #35949 / #40560），把 5/9 的 `SharedFusedMoE` 架构统一成了 `FusedMoE + MoERunner`。

关键变化是通过 `AscendMoERunner` 继承 upstream 的 `MoERunner`，用 override 模式替换硬件相关代码。

### Step 2：reduce 决策改用 property

新架构用 `_fused_output_is_reduced` property 来判断——当前 fused output 是否已经做了 TP all-reduce：

```python
# vllm-ascend AscendMoERunner
@property
def _fused_output_is_reduced(self):
    moe_comm_type = _EXTRA_CTX.moe_comm_type
    return moe_comm_type in {ALLTOALL, MC2, FUSED_MC2}
```

语义：MC2 / AllToAll / FusedMC2 这些通信方式的 combine 算子内部已经做过跨 NPU 归约，不需要额外 all-reduce；AllGather 只做 all-gather（不 reduce），需要额外 all-reduce。

### Step 3：property 在 graph capture 下被静态化（关键 bug）

vllm 的 graph capture（aclgraph / cudagraph）会把 GPU 计算图录制成 static graph。录制时 **Python 表达式会被求值并固化**。

Capture 时调用链：

```text
forward()
  → MoERunner.forward()          ← 行 628
    → if not self._fused_output_is_reduced:
        tensor_model_parallel_all_reduce(states)  ← 这个 if 被静态化
    → return states[..., :trunc_size]
```

Graph 录制时执行了 `_fused_output_is_reduced` property，得到一个布尔常量，编译进 static graph。之后 replay 时 property 不会被重新读取——graph 里已经是固定值。

### Step 4：select_moe_comm_method 的 512 阈值

```python
# ascend_forward_context.py
def select_moe_comm_method(num_tokens):
    if num_tokens <= mc2_tokens_capacity:  # 默认 512
        return CommunicateType.FUSED_MC2
    return CommunicateType.ALLGATHER
```

这个决策是基于 **运行时 token 数** 的动态选择，不同请求可能走不同路径。

### Step 5：capture vs replay 的路径不一致

| 阶段 | 场景 | token 数 | 通信方式 | property 值 | all-reduce |
|------|------|----------|----------|-------------|------------|
| Capture | 单请求，prompt 小 | ≤ 512 | MC2 | True（已 reduce） | 跳过（正确） |
| Replay | 32 并发 prefill | > 512 | AllGather | **True（静态化）** | **跳过（错误！）** |

结果：AllGather 路径漏 final all-reduce → hidden states 跨 NPU 未归约 → 输出乱码。

这也解释了 **为什么单并发正常**：单并发 prompt 小走 MC2，本来就应该跳过 all-reduce，只是运气好没暴露 bug。

### 为什么 5/9 没问题

5/9 的 `SharedFusedMoE` **不走 property 判断**，而是模型层直接调 custom op：

```python
# 5/9 SharedFusedMoE
states = torch.ops.vllm.maybe_all_reduce_tensor_model_parallel(states)
```

Custom op 被 graph 当 opaque call，不做内部静态化。5/10 的 MoE 重构改了这条路径——而重构时没有人意识到 property 在 graph capture 下会被静态化。

---

## 3. 核心概念

### 3.1 Graph capture（aclgraph / cudagraph）

推理框架每次 forward 要经历 Python → PyTorch dispatch → CANN/NPU 算子的链路，dispatch 开销对 decode 这种固定形状的阶段是重复浪费。

Graph capture 做的事：第一次正常跑，但把每一步的算子调用、tensor 形状/地址等信息录下来，编译成 static graph。之后 replay 时直接提交 graph 给硬件执行，跳过 Python 和 dispatch。

**关键**：Python 表达式在 capture 时求值并固化，replay 时不重新求值。

### 3.2 MC2 vs AllGather 的 reduce 差异

| 通信方式 | dispatch 是否 reduce | combine 是否 reduce | 需额外 all-reduce |
|---|:-:|:-:|:-:|
| MC2 / FusedMC2 | 内部处理 | 内部处理 | 不需要 |
| AllToAll | 内部处理 | 内部处理 | 不需要 |
| AllGather | all-gather（不 reduce） | 纯本地 | **需要** |

### 3.3 Custom op 绕过静态化

`torch.ops.vllm.maybe_all_reduce_tensor_model_parallel` 是一个 registered PyTorch custom op。Graph capture 引擎把它当作 opaque call——只录"这里有个 op 要调"，不分析 op 内部逻辑。

Replay 时每次真的调用 op，op 内部 runtime 根据当时的 `moe_comm_type` 判断是否做 all-reduce：

```text
AllGather → 做 all-reduce
MC2 / AllToAll / FusedMC2 → 返回原值（已 reduce）
```

**对比**：

| 方式 | capture 阶段 | replay 阶段 |
|------|-------------|-------------|
| Property | 求值并固化 | 走固定分支 |
| Custom op | 录为 opaque call | runtime 判断 |

---

## 4. 修复方案

### 4.1 核心修改

在 `AscendMoERunner` 中 override `_maybe_reduce_final_output`（8 行新增）：

```python
# vllm_ascend/ops/fused_moe/fused_moe.py
def _maybe_reduce_final_output(
    self,
    states: torch.Tensor,
    trunc_size: int,
) -> torch.Tensor:
    states = torch.ops.vllm.maybe_all_reduce_tensor_model_parallel(states)
    return states[..., :trunc_size]
```

位置：`_maybe_reduce_shared_expert_output` 之后、`forward_impl` 之前。

### 4.2 修复产物

| 项目 | 内容 |
|------|------|
| 分支 | A00282_E（vllm-ascend） |
| commit | a186705c |
| 基于 | ca4065f2（5/10 nightly） |
| 配套 vllm | 4d51588e2 |
| 改动量 | 8 行新增 |
| upstream 来源 | 11803e30（#10557） |

### 4.3 验证

| 场景 | vllm | vllm-ascend | 精度 |
|------|------|-------------|------|
| 5/9 成功 | d886c26d4 | 68ff5263 | 96.875% |
| 5/10 失败 | 4d51588e2 | ca4065f2 | 0.0% |
| A00282_E | 4d51588e2 | a186705c | 恢复 |

辅助验证：`--enforce-eager`（禁 graph capture）也能恢复精度，确认根因在 graph capture。

### 4.4 Upstream 状态

`11803e30`（#10557）已于 2026-06-17 被 vllm-ascend upstream main 合入，当前最新 main 已包含相同修复。

---

## 5. 排查过程

### 5.1 回归坐标系确认（5 个实验）

| 实验 | 组合 | 结果 | 含义 |
|------|------|------|------|
| 1 | vllm-asc 68ff5263 + vllm d886c26d4（5/9 基线） | 正常 | 基线确认 |
| 2 | ca4065f2 + 4d51588e2（5/10） | 乱码 | 复现 |
| 3 | 68ff5263 + 4d51588e2（只升 vllm） | 正常 | **回归在 vllm-ascend 侧** |
| 4 | 7fd2cede + 4d51588e2（升 vllm-ascend） | 乱码 | 回归入口 = 7fd2cede |
| 5 | --enforce-eager | 恢复 | 问题跟 graph capture 相关 |

结论：回归在 vllm-ascend `68ff5263 → 7fd2cede` 内，且跟 graph capture 有关。

### 5.2 代码级别分析

逐行对比 5/9 和 5/10 的 Qwen3 MoE forward 路径，确认 MC2 核心算子一致、reduce 逻辑等效。结论上限是"可能没问题"——Python 代码等效 ≠ 运行时行为一致。

版本 bisect 被兼容性约束阻断：3 个关键 PR（#35949 / #35782 / #40560）都在两个兼容范围的空隙里，无法干净切换。

### 5.3 半适配版本 bisect（弯路）

为验证 vllm upstream 的中间 commit 是否为根因，写了 A00282_A 分支（基于 68ff5263 适配 726efe177），做了 6 次修改：

1. reduce_results 属性 fallback
2. _routed_input_transform 属性 fallback
3. 构造参数改关键字
4. 方法改名（`forward_impl` 改为 `_forward_impl`）
5. 返回类型修正
6. property 修正

方向最终被证明错误——根因不在 vllm upstream，而在 vllm-ascend 自己的 #8899。

### 5.4 专家定位

主管求助专家后，专家给出最高置信根因：

1. _fused_output_is_reduced property 在 graph capture 下被静态化
2. AllGather 路径因这个静态化漏 all-reduce
3. 推荐 backport #10557

### 5.5 已排除方向

| 方向 | 排除依据 |
|------|---------|
| vllm upstream PR（#35949 / #35782 / #40560） | 同版本 vllm 配旧 vllm-ascend 正常 |
| mc2_mask stale | 不在 capture scope |
| prefix-caching | 实测关了不恢复 |
| 量化（BF16 vs W8A8） | 都挂 |
| decode cudagraph mode | FULL_DECODE_ONLY 不恢复 |
| HCCL_BUFFSIZE | 不影响正确性，实测 200 不恢复 |
| reduce 双归约 | 5/9 和 5/10 对 Qwen3（0 shared）等效 |

---

## 6. 复盘：流程问题

### 6.1 回归坐标系确认后，没有直接看 diff

5 个实验已锁定"回归在 vllm-ascend `68ff5263 → 7fd2cede`"，但没有直接看这个区间内的 diff，而是跳到了 vllm upstream 的 199 个 commits。理由是"vllm-ascend 只是适配上游"——但第 3 个实验已经证明同版 vllm 配旧 vllm-ascend 是正常的。

应该做的是：**直接看 `68ff5263 → 7fd2cede` 的代码 diff。** 里面就包含了 `_fused_output_is_reduced` 这个新加的 property。

### 6.2 实验优先级反了

`--enforce-eager`（2 分钟就能验证）在很后面才做，半适配 bisect（2 天开发）早早开始了。

对 graph 相关的问题，`--enforce-eager` 是最高性价比的验证手段。它应该排在最前面。

### 6.3 代码分析不能替代运行时验证

"Python 代码等效"只说了"如果正常运行时确实等效"——但 graph capture 引入了 Python 层面看不到的行为差异。

**对图模式的专项认知**：Python `@property` 在 graph capture 阶段会被求值并静态化。这是排查 graph 相关问题的第一怀疑对象。

### 6.4 没有主动设"方向对不对"的检查点

6 次修复后才停下来。每次修完应该问自己：**"这个实验告诉我什么？它支持/否定了什么假设？"**

如果在第 2 次失败时就问这个问题，早就意识到方向错了。

---

## 7. 关键经验（可复用）

1. **回归在哪里，就看哪里的 diff**。不要跳到外圈。
2. **先做 2 分钟的验证，再做 2 天的实验**。`--enforce-eager` / `graph_mode=none` 这类低成本实验优先。
3. **代码等效 ≠ 行为等效**。对图模式要特别敏感：property 在 graph capture 下会被静态化。
4. **每个实验前写假设 + 预期结果**。回答不了"这个实验告诉我什么"就别做。
5. **2 天没进展就求助**。方向错了做再多也没用。
6. **每完成一个实验，更新"当前地图"**：已确认的事实 / 已排除的方向 / 下一步怀疑点。

---

## 8. 后续待办

- [ ] perf(throughput) benchmark 确认性能无回归
- [ ] nightly 其他模型（DeepSeek-R1 / GLM-5.1 / Kimi-K2.5）确认无副作用
- [ ] 推 a186705c 到 upstream — 已完成（main 已合入 #10557）

---

## 9. 关键 PR 速查

| PR | commit | 日期 | 内容 |
|----|--------|------|------|
| vllm #35949 | 726efe177 | 4/20 | reduce 移入 MoERunnerBase |
| vllm #35782 | 5e584ce9e | 4/21 | 移除 SharedFusedMoE |
| vllm #40560 | 809d83c2d | 4/22 | 合并 MoERunnerBase + DefaultMoERunner |
| vllm #40860 | 4d51588e2 | 4/26 | DeepSeek V4 Rebased（引入 input_ids）|
| **vllm-ascend #8899** | **7fd2cede** | **5/8** | **Main2Main 升级（根因入口）** |
| vllm-ascend #10557 | 11803e30 | 6/17 | **上游正式修复** |
| A00282_E | a186705c | 6/25 | **backport 验证** |

---

## 10. 参考与归档

本文档为 A00282 的正式复盘总结（自包含）。定位过程中产生的中文草稿文档已归档到 wiki `draft/` 目录，内容包括：

- `draft/A00282_根因分析报告.md` — 详细技术分析（含 5 个核心概念详解）
- `draft/A00282_复现步骤.md` — 绿区操作手册（serve 命令、切版本验证）
- `draft/A00282_Nightly环境复现指南.md` — Docker 复现方法
- `draft/A00282_vllm-ascend_commits_68ff5263_ca4065f2.md` — commit 对比分析

workspace 根下补充材料：

- `task_plan.md` — 实验组合 + 根因链
- `findings.md` — 分析阶段原始记录
- `progress.md` — 完整实验日志
