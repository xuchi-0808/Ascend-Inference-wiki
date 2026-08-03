---
date: 2026-07-24
categories:
  - 复盘
tags:
  - DeepSeek
  - DP
  - MTP
  - 挂起
---

# A00296 经验沉淀：DSV3.1 DP32 MTP lmheadtp AllGather 崩溃定位全程复盘

> 项目编号：A00296
> 模型：DeepSeek-V3.1-Terminus-w8a8-mtp-QuaRot（A3，PD 分离，DP32）
> 时间跨度：2026-07-03 ~ 2026-07-08
> 最终处理：revert commit `9099b7f6`（cherry-pick PR #11562）
> 根因状态：**触发条件已定位，底层机制未知，仍在定位中**

---

## 一、问题现象

### 环境

- **框架**：vllm-ascend，昇腾 NPU（A3，单卡双芯）
- **部署**：PD 分离，1 个 Prefill 节点 + 2 个 Decode 节点

| 节点                   | 机器 | 配置     | 关键开关                        |
| ---------------------- | ---- | -------- | ------------------------------- |
| P0 (Prefill)           | S8   | TP8 DP2  | enforce_eager（Prefill 不入图） |
| D1 (Decode rank 0-15)  | S6   | TP1 DP16 | cudagraph_mode=FULL_DECODE_ONLY |
| D0 (Decode rank 16-31) | S5   | TP1 DP16 | cudagraph_mode=FULL_DECODE_ONLY |

- D 节点合计：DP32，lmheadtp=8（4 组 × 8 rank）
- **模型**：DeepSeek-V3.1-Terminus-w8a8-mtp-QuaRot
- **关键开关**：
  - MTP **入图**（num_speculative_tokens=3，method=deepseek_mtp）
  - lmheadtp **开启**（lmhead_tensor_parallel_size=8）
  - cudagraph_mode=**FULL_DECODE_ONLY**
  - cudagraph_capture_sizes=[3, 48]
  - AIV **开启**（`HCCL_OP_EXPANSION_MODE="AIV"`，AIV 是 crash 的必要条件之一）

### 崩溃表现

- D 节点收到推理请求后，在 ACL graph replay 阶段崩溃
- 故障 kernel：`aiv_all_gather_bfloat16_t`（HCCL AIV AllGather）
- rankSize=8（与 lmheadtp 通信组一致）
- size=688128 bytes = 48 tokens × 7168 hidden × 2 bytes（bf16）
- error: 507011 AIVEC timeout
- HCCL 警告：`multi groups include aiv alg, may cause execution stuck. has aiv group count[2]`
- 所有 32 个 D rank 最终因 shm_broadcast 超时（60s）连带退出

### plog 通信组分析（D0, rank 16-31）

| group_name | rankSize | 身份                               |
| ---------- | -------- | ---------------------------------- |
| 261        | 32       | DP 全局组（跨 S5+S6）              |
| 263        | 32       | 另一个 DP 全局组                   |
| 269        | 8        | lmheadtp 组（D0 本地，rank 16-23） |

崩溃的 all_gather 来自 rankSize=8 的 lmheadtp 组（group 269）。

---

## 二、定位过程时间线

### 阶段 1：排除法 + 代码分析（07-03）——锁定三要素，找到规避方案

#### 1.1 环境层面排除法

通过开关三个关键参数，快速锁定崩溃的必要条件：

|   #   | lmheadtp |  MTP   |     cudagraph     | 结果                   |
| :---: | :------: | :----: | :---------------: | ---------------------- |
|   1   | 8（开）  | ✅ 入图 | FULL_DECODE_ONLY  | ❌ **崩溃**（原始问题） |
|   2   | 1（关）  | ✅ 入图 | FULL_DECODE_ONLY  | ✅ 成功                 |
|   3   | 8（开）  | ❌ 关闭 | FULL_DECODE_ONLY  | ✅ 成功                 |
|   4   | 8（开）  | ✅ 入图 | **enforce_eager** | ✅ 成功                 |

**排除法结论**：崩溃需要 **lmheadtp=8 + MTP 入图 + cudagraph** 三个条件**同时满足**，关掉任何一个都不崩。

#### 1.2 代码层面分析——定位到崩溃点，但未触及根因

通过代码走读，定位到崩溃发生的代码路径（draft model 的 lmheadtp all_gather）：

```text
draft model 推理:
  llm_base_proposer._run_merged_draft()
    → self.model.compute_logits(sample_hidden_states)
      → AscendLogitsProcessor._get_logits_lmheadtp()
        → get_lmhead_tp_group().all_gather(hidden_states, dim=0)  ← 崩溃点
```

同时注意到主模型和 draft model 的 graph 覆盖范围存在差异：

- **主模型**：`compute_logits()` 在 graph 区域**外** → **eager**
- **draft model**：**整个 `_run_merged_draft()` 被 ACLGraphWrapper 捕获**，all_gather 在 graph 区域**内**

> 这个差异当时被当作关键线索——认为 all_gather 在 graph 内导致 replay 时地址不稳定。但静态 buffer patch 尝试后未能修复，该怀疑点被暂且排除。**代码分析只定位了崩溃发生在哪里，并没有解释为什么崩溃。根因至今尚未完全厘清。**

#### 1.3 阶段 1 结论——找到两个临时规避方案

配置层面的排除法进一步找到了两个有效临时方案：

|   #   |     cudagraph_mode     | npugraph_ex |  drafter  | 结果                                 |
| :---: | :--------------------: | :---------: | :-------: | ------------------------------------ |
|   2   |    FULL_DECODE_ONLY    |    ✅ 开     | **eager** | ✅ 成功（`A00296_MTP_EAGER=1`）       |
|   3   | **FULL_AND_PIECEWISE** |    ❌ 关     |   graph   | ✅ 成功（删掉 cudagraph_mode 走默认） |

> 但这只是**规避**，不是根因修复。

---

### 阶段 2：第一次误判——AIV 多通信域冲突（07-05）——后被撤销

#### 当时的推理

HCCL 报警 `multi groups include aiv alg, has aiv group count[2]` 引向了一个看似完美的解释：

- 查阅 CANN 文档 `HCCL_OP_EXPANSION_MODE.md`，明确写着：**AIV 配置项不支持多通信域并行的场景**，否则可能导致不可预期行为
- 通过 plog 的 UDI（User-Defined Identifier）标签，锁定了两个 AIV 域的身份：

| HCCL group_name  | rankSize | UDI 标签     | 身份                     |
| ---------------- | -------- | ------------ | ------------------------ |
| `group_name_261` | 32       | udi=ep       | EP 组（Expert Parallel） |
| `group_name_271` | 8        | udi=lmheadtp | lmheadtp 组              |

两个域在同一 devicePhyId 上共存，且都因全局 `HCCL_OP_EXPANSION_MODE="AIV"` 启用了 AIV 算法。当时认为这就是根因——多通信域 AIV 冲突。

#### 为什么这个结论"看起来很对"

- 有 **HCCL 官方文档**明确限制多域并行
- 有 **plog UDI 标签**实证两个 AIV 域共存
- 有"关 AIV 不崩、开 AIV 崩"的实验现象

#### 为什么后来撤销了

**"关 AIV 不崩、开 AIV 崩"这个实验现象是真实的**（同一版本上验证），但**对现象的解释（AIV 多通信域冲突）是错的。**

阶段 4 的 bisect + revert 完成后，发现回退 `9099b7f6` 后即使**开着 AIV** 也不崩：

- AIV 开/关只是改变了问题的触发条件组合，不是根因本身
- 真正的触发条件是 `9099b7f6` 引入的额外 dummy run
- plog 显示的"AIV 多通信域共存"是**现象**，但被错误地解读为**原因**

> ⚠️ **"AIV 多通信域冲突"这个解释已被撤销**。两个 AIV 域共存是事实，但冲突不是崩溃的根因——回退 `9099b7f6` 后开着 AIV 也不崩可证。

---

### 阶段 3：Bisect 锁定回归点（07-06/07）——客观仲裁

代码层面的主观推理（地址 desync、AIV 冲突）反复失败后，转向 **commit 二分（bisect）** ——让数据说话，而不是继续猜。

#### Bisect 实测记录

|   #   | Commit     |   PR   | 说明                               |      结果      |
| :---: | ---------- | :----: | ---------------------------------- | :------------: |
|   1   | `ff4807ea` | #10141 | CI weekly test fix                 |       ✅        |
|   2   | `a87f2b75` | #10139 | 文档翻译                           |     ✅ 跳过     |
|   3   | `2b4a9daa` | #9835  | **main2main 升级 vLLM v0.21.0**    | ✅ **真左边界** |
|   4   | `971d50b3` | #9476  | DSA W8A8 fix                       |    ❌ 右边界    |
|   5   | `be000c00` | #10241 | netloader fix                      |       ❌        |
|   6   | `895fd0c6` | #10239 | CI slash command                   |       ❌        |
|   7   | `5f2ef5a0` | #9962  | Remove legacy capture-size pruning |       ✅        |
|   8   | `caf58a20` | #10087 | AscendKVBlockZeroer                |       ❌        |
|   9   | `fc0b9e35` | #10181 | 310P compressed mask               |       ❌        |
|  10   | `87ae55b9` | #10228 | Fix prefix-mamba-cache             | ❌ **首次出现** |

**首次出现崩溃的 commit**：`87ae55b9`。

#### bisect 确认回归点

bisect 定位到首个崩溃的 commit 为 `87ae55b9`，此前 `5f2ef5a0`（✅）正常。在正常与崩溃之间的区间内，`9099b7f6` 涉及显存预估流程的改动。经后续 revert 验证，确认 `9099b7f6` 就是导致问题的回归点。

---

### 阶段 4：Revert 验证 + 根因修正（07-07/08）——闭环

#### 验证：cherry-pick revert

cherry-pick PR #11562（revert `9099b7f6`）到 `eef4703c` 基线上，在 `xc_vllm_A00296_confirm` 三台容器上验证：

- ✅ MTP 入图 + lmheadtp=8 + cudagraph FULL_DECODE_ONLY **全量通过**
- ✅ AIV **保持开启**也不崩

**确认事实**：`9099b7f6` 是触发条件，回退后崩溃消失。

#### 根因修正：撤销 AIV 结论

由于回退 `9099b7f6` 后即使开着 AIV 也不崩，阶段 2 的"AIV 多通信域冲突"解释**正式撤销**——"关 AIV 不崩"的现象是真实的，但把 plog 多域共存解读为根因是错的：

| 能确认的事实                     | 证据                                                                                                                   |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `9099b7f6`（PR #9865）→ 触发崩溃 | bisect 确认回归起点在此区间                                                                                            |
| 回退此 commit → 崩溃消失         | cherry-pick PR #11562 后全量通过                                                                                       |
| **AIV 的作用**                   | 关 AIV 可避免崩溃（同一版本验证，AIV 是必要条件），但关 AIV 不崩 ≠ 多域冲突是根因——revert `9099b7f6` 后开着 AIV 也不崩 |

#### 问题 PR 干了什么

PR #9865 在 `determine_available_memory()` 中加了**额外一次 dummy run**（走 `profile_cudagraph_memory()`），目的是在 KV cache 分配前更准确预估 ACL graph 池的显存占用。但这次额外 run 触发了后续 MTP + lmhead TP + cudagraph 场景下的 AllGather hang。

#### 机制未知，策略性 revert

- **额外 dummy run 导致 AllGather hang 的具体机制未知**，涉及 ACL graph + HCCL 交互，定位成本高
- revert 后对性能**无实际影响**（显存预估只是优化精度，不预估也不会出错，原代码已有兜底）
- **策略**：直接 revert 规避，根因另提 issue 记录，后续有精力再定位

---

## 三、最终处理

### 方案

revert commit `9099b7f6`（PR #9865），通过 cherry-pick PR #11562 落地。

### 影响范围

- **bug 引入**：`9099b7f6`（PR #9865，[Feature] Estimate ACL graph memory before KV cache allocation）
- **bug 触发条件**：DP32 + lmheadtp=8 + MTP 入图 + cudagraph FULL_DECODE_ONLY + AIV 开启 五者同时满足
- **修复**：revert，merge 为 `a1e4b056`
- **性能影响**：无（显存预估是优化项，回退不损失功能）

### 根因状态

⚠️ **触发条件已定位（`9099b7f6` 的额外 dummy run），但底层机制（额外 dummy run 为何导致 AllGather hang）未知，仍在定位中。**

---

## 四、经验教训

### 教训 1：主观分析有上限，客观手段来兜底

代码分析能定位 crash site（draft all_gather 在 graph 内），但解释不了为什么 revert 一个显存预估的 dummy run 能把问题修好。代码等效 ≠ 行为等效，graph capture/replay 的运行时交互在代码层面是不可见的。

当主观推理反复失败（地址 desync 试了没修好、AIV 多域冲突看似有理但实际是错的），bisect 这种客观手段能跳出思维定势——它不关心"为什么"，只回答"在哪里"，直接锁定了位于显存预估模块的 `9099b7f6`——人工分析时根本不会去看的地方。

**改进**：当代码分析的矛盾（"理论上没问题但实际出问题"）连续出现时，意识到代码分析到上限了，立即转向 bisect。

### 教训 2：现象正确 ≠ 根因正确，日志不是证据

"关 AIV 就不崩"这个实验结论没错（同一版本验证），但把 plog 显示的 AIV 多域共存解读为"多通信域冲突导致崩溃"是错的。回退 `9099b7f6` 后开着 AIV 也不崩，说明 AIV 只是触发条件组合中的一环，不是根因。

plog 的 UDI 标签和 HCCL 的 `has aiv group count[2]` 警告都是真实的"现象"，但它们只证明两个 AIV 域存在，不证明"冲突导致了崩溃"。

**改进**：

- 实验结论正确（"开关 X 有效"）≠ 根因解释正确（"为什么有效"）
- 日志/警告是**线索**不是**证据**——告诉你"发生了什么"，不告诉你"为什么发生"
- 根因定位有两个层次：先找**触发条件**（什么组合会崩），再找**机制**（为什么崩）。混淆两者是常见误判来源

### 教训 3：优化 PR 关注副作用，机制不清晰时止损比深究合理

`9099b7f6` 只是加了一次 dummy run 来更准确预估显存，和崩溃场景（AllGather 通信）在代码上毫无交集，但意外触发了 ACL graph + HCCL 的交互 bug。优化类 PR 的副作用常出现在和优化目标无关的模块，review 时容易被忽略。

额外 dummy run 导致 hang 的具体机制至今未知，但 revert 后性能无影响（显存估算是优化项，原代码有兜底），因此选择策略性 revert，不深入定位。

**改进**：

- 涉及 graph capture、显存 profile、dummy run 的优化 PR，要特别关注对通信算子的副作用
- 不是所有 bug 都需要搞清机制才能闭环——revert 成本低、定位成本高时，止损是合理决策

---
