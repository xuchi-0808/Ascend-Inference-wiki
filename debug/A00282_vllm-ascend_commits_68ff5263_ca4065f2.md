# A00282 — vllm-ascend `68ff5263..ca4065f2` (5/9→5/10) commit 逐笔分析

> 回归窗口：vllm-ascend `68ff5263`(5/9 成功) → `ca4065f2`(5/10 失败)
> 共 14 个 commit，按日期排列。本文分析每笔 commit 是否侵入 Qwen3-235B-A22B 多 DP A2 推理路径。

## 分类总览

| 分类 | 数量 | 是否可能影响 Qwen3 A2 推理 |
|------|:---:|--------------------------|
| Main2Main 适配 | 1 | **✅ 是（唯一）** |
| P/D disagg 修复 | 1 | ❌ Qwen3 A2 非 disagg |
| SpecDecode / Eagle 相关 | 4 | ❌ Qwen3 A2 无 spec_decode |
| CI / 文档 / 其他 | 8 | ❌ 不触碰推理 |

## 逐笔分析

### 1. `93160405` — `(BugFix) fix dcp async bug (#8749)`

**变更文件（5 files, +167/-12）**：

- `tests/e2e/nightly/single_node/2-cards/spec_decode/test_spec_decode.py`
- `vllm_ascend/attention/context_parallel/attention_cp.py`
- `vllm_ascend/attention/utils.py`
- `vllm_ascend/spec_decode/eagle_proposer.py`

**影响分析**：修复 context_parallel + eagle spec_decode 的 dcp async bug。Qwen3-235B-A22B A2 配置不使用 context_parallel 也不使用 eagle spec_decode。

**结论**：❌ 不影响。

---

### 2. `7dd1e282` — (Doc) Update translations and documentation links (#8942)

**变更文件（34 files, +110/-152）**：文档翻译和链接更新。

**结论**：❌ 纯文档，不影响。

---

### 3. `0b8aa91d` — (CI) fix nightly MiniMax-M2.5-w8a8 (#8957)

**变更文件（6 files, +10/-10）**：MiniMax-M2.5 nightly 测试配置修复。

**结论**：❌ 仅 MiniMax 测试，不影响 Qwen3。

---

### 4. `5dd158c5` — (Test) Refactor Eagle Proposer metadata updates (#8868)

**变更文件（1 file, +106/-0）**：`tests/ut/spec_decode/test_eagle_proposer.py`

**结论**：❌ Eagle spec_decode UT，Qwen3 A2 无 eagle。

---

### 5. `55c838bd` — (CI) Increase max-parallel for nightly test (#8946)

**变更文件（1 file, +2/-2）**：CI 配置 `max-parallel` 参数调整。

**结论**：❌ CI 配置，不影响推理。

---

### 6. `7fd2cede` — (Misc) Upgrade vLLM to 0427 (#8899) ⭐

**变更文件（26 files, +366/-251）**：Main2Main 升级，适配 vllm upstream `d886c26d4` → `4d51588e2`（199 commits，包含 PR #35782/#35949/#40560/#40671）。

**关键推理文件变更**：

| 文件 | 变更行数 | 说明 |
|------|:------:|------|
| `vllm_ascend/ops/fused_moe/fused_moe.py` | +132/-153 | MoE 核心：类合并、方法改名、`_fused_output_is_reduced` 新增 |
| `vllm_ascend/_310p/fused_moe/fused_moe.py` | +75/-0 | 310P 版本同步 |
| `vllm_ascend/lora/utils.py` | +32/-0 | LoRA MoE adapter 适配 |
| `vllm_ascend/__init__.py` | +25/-0 | 版本号更新 |
| `vllm_ascend/utils.py` | +6/-0 | suspend/disable DP sync 新接口 |
| `vllm_ascend/patch/platform/patch_kv_cache_utils.py` | +52/-0 | kv_cache layerwise pooling |
| `vllm_ascend/worker/model_runner_v1.py` | +1/-1 | 1 行适配 |
| 其余（spec_decode、worker、attn） | 少量 | 适配性改动 |

**fused_moe.py 的具体变化（逐行对比 `68ff5263` 和 `ca4065f2`）**：

1. **`AscendMoERunner` 父类变化**：`DefaultMoERunner` → `MoERunner`
   - 原因：vllm upstream #40560 合并了 `MoERunnerBase` + `DefaultMoERunner` → 统一 `MoERunner`
   - 影响：继承的 `forward()` 实现来自新 `MoERunner`（但 MC2 核心路径不变）

2. **`forward_dispatch` → `_forward_impl`（方法改名）**：
   - 原因：新 `MoERunner.forward()` 通过 `_forward_entry`（moe_forward op）→ `layer.runner._forward_impl()` 调用
   - 新增 `input_ids` 参数（但 ascend 未使用）
   - 内部逻辑不变：`with self._sequence_parallel_context(): return self.forward_impl(...)`

3. **`AscendSharedFusedMoE` 删除**：
   - 原因：vllm upstream #35782 移除 `SharedFusedMoE` 类
   - 影响：有-shared 模型的逻辑合并进 `AscendFusedMoE.shared_forward_impl`

4. **`_fused_output_is_reduced` 新增**：
   - 对 MC2/FUSED_MC2 返回 `True`，对 ALLGATHER 返回 `False`
   - 原因：vllm upstream #35949 拆分 shared/fused reduce
   - 影响：MC2 finalize 已包含 reduce → 让 upstream MoERunner 跳过 `_maybe_reduce_final_output`
   - **已用 E2 排除**（恢复 TP all-reduce 不恢复精度）

5. **`self.reduce_results` 删除**：
   - 旧版 `finalize` 传 `reduce_results=self.reduce_results`，新版不传
   - MC2 finalize 内部 fixed `reduce_results` 参数

**结论**：✅ **唯一涉推理的 commit。** MC2 核心逻辑（`layer.forward_impl` 的 prepare/apply/finalize）**零变化**。变化全是适配上游 vllm MoE runner 重构的：

- 方法改名：`forward_dispatch` → `_forward_impl`
- 类合并：`AscendSharedFusedMoE` 删除
- reduce 逻辑位置变化：vllm-ascend → vllm upstream

**回归机制**：vllm upstream 的 MoERunner.forward() 重构改变了 forward scope → 某个新增操作（或调用顺序变化）与 cudagraph PIECEWISE capture 交互，导致 0-shared 模型（Qwen3）的 MC2 输出异常。

---

### 7. `4b3a2af7` — (P/D) Fix for transmit kv cache failure (#8959)

**变更文件（4 files, +201/-34）**：Mooncake KV transfer 修复。

**影响分析**：P/D disaggregation（mooncake connector / layerwise connector）的 KV cache 传输修复。Qwen3-235B-A22B A2 不执行 P/D disagg。

**结论**：❌ 不影响。

---

### 8. `67647403` — (BugFix) Fix CpuGpuBuffer patch target for Eagle proposer test (#8999)

**变更文件（1 file, +1/-1）**：Eagle proposer 测试的 CpuGpuBuffer patch 目标修复。Qwen3 A2 无 eagle。

**结论**：❌ 不影响。

---

### 9. `d3185d29` — (Doc) Translated Doc files 2026-05-09 (#9001)

**变更文件（6 files, +373/-58）**：文档中文翻译。

**结论**：❌ 纯文档，不影响。

---

### 10. `0b240395` — (CI) Bugfix: Fix the previously un-updated main2main commit (#9010)

**变更文件（10 files, +10/-10）**：CI 工作流 main2main commit 引用修复。

**结论**：❌ CI 配置，不影响。

---

### 11. `a53f8346` — (Test) Add tests for set_inputs_first_pass (#8781)

**变更文件（2 files, +704/-1）**：SpecDecode 测试代码。Qwen3 A2 无 spec_decode。

**结论**：❌ 不影响。

---

### 12. `07f6fec2` — (BugFix) Revert catlass change (#9014)

**变更文件（1 file, +1/-1）**：回退 catlass submodule 引用。

**结论**：❌ 不影响。

---

### 13. `bd25f2e8` — (Community) Nominate new maintainer (#8996)

**变更文件（1 file, +1/-0）**：CODEOWNERS 文件，maintainer 提名。

**结论**：❌ 不触碰推理。

---

### 14. `ca4065f2` — (Test) Add prepare_inputs UTs (#8828)

**变更文件（1 file, +724/-5）**：SpecDecode 测试代码。Qwen3 A2 无 spec_decode。

**结论**：❌ 不影响。

---

## 结论

| 结论 | 说明 |
|------|------|
| **回归入口** | `7fd2cede`（Main2Main 升级 vLLM→0427）是唯一涉及推理路径的 commit |
| **fused_moe.py 变化** | 285 行改动，但 MC2 核心逻辑（prepare/apply/finalize）**零变化** |
| **变化性质** | 全部是适配 vllm upstream MoE runner 重构的改动（方法改名、类合并、reduce 逻辑位置移动） |
| **上游推测定** | vllm upstream PR #35782/#35949/#40560 改变了 MoERunner.forward() 的 scope → 与 cudagraph PIECEWISE capture 交互 → 仅 Qwen3（0-shared + 无 speculative）受影响 |
| **双仓关联** | 分析 vllm-ascend 的 commit 不足以定位根因——**根因在 vllm upstream 的 199 commits（`d886c26d4..4d51588e2`）中** |
