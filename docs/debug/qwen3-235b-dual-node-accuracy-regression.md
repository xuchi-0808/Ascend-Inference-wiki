# A00275: Qwen3-235B-A22B 双机 nightly GSM8K accuracy 不达标

## 问题现象

Qwen3-235B-A22B（MoE, 235B 总参/22B 激活）在双机 nightly CI 中 GSM8K accuracy 不达标。

| 流水线 | 日期 | GSM8K | 结果 |
|:---|:---:|:---:|:---:|
| main | 5/20 | 92.80% | ✅ 通过（基线 95%，阈值 ±3%） |
| main | 5/22 | 90.52% | ❌ 首次失败 |
| releases-v0.20.2rc | 6/1 | 88.78% | ❌ 持续失败 |

### 测试配置

- **双机** 2 nodes × 16 NPU = 32 NPU，A3 SOC
- **DP=4, TP=8, EP 开启**
- **模型**: Qwen/Qwen3-235B-A22B (bf16)
- **数据集**: gsm8k 0-shot CoT, 2800 prompts
- **Benchmark**: ais_bench

---

## 关键时间线

| 时间 | 事件 |
|:---|:---|
| 19:18:32 | vllm serve 启动，初始化 DP Coordinator |
| 19:19:02 | 第一个 POST 请求进来（perf benchmark） |
| **19:19:06** | **16 ranks 全部打出 `ProcessGroupHCCL.cpp:5442` allgather fake wait 警告** |
| 20:12:12 | GSM8K accuracy 计算完成 (88.78%) |
| 20:12:17.194 | Benchmark 判定失败 |
| 20:12:17.444 | `DP_Coordinator (PID: 202) died with exit code None`（shutdown 副产物） |

---

## 日志对比分析

### 两次关键 CI 日志对比

| 检查项 | 5/20 ✅ 成功 | 5/22 ❌ 失败 |
|:---|:---:|:---:|
| GSM8K | 92.80% ✅ | 90.52% ❌ |
| HCCL allgather fake wait (16 ranks) | ✅ **有** | ✅ **有** |
| DP_Coordinator exit code None | ❌ **无** | ✅ **有** |
| 容器镜像 | `nightly-ci-main-a3` | `nightly-ci-main-a3` |
| CANN / npu-smi | 9.0.0 / 25.5.2 | 9.0.0 / 25.5.2 |

### 关键发现

**HCCL allgather fake wait 在成功的流水线中也存在。** 这说明它造成了约 2% 的系统性精度损失（理论基线 ~95% → 实测 92.80%），但这对阈值 ±3% 来说是可接受的。

**5/22 起的额外损失（92.80% → 90.52% → 88.78%）由另一个因素导致。** 这个因素同时带来了 `DP_Coordinator` 的 exit code None 报错。

```text
95%  (理论基线)
  │ - HCCL fake wait (~2%)
  │
92.80%  (5/20 成功，阈值内 ✅)
  │ - ??? 未知因素 (~2-4%)
  │
90.52% ~ 88.78%  (5/22~6/1 失败 ❌)
  │
  └─ DP_Coordinator error 也同时出现
```

---

## HCCL allgather fake wait 分析

### 触发路径

```text
19:19:02 第一个推理请求
  ↓
Expert Parallel 把 token 分发到不同 EP 组
  ↓
Worker_DP0_TP0_EP0 和 Worker_DP1_TP0_EP8 手里的 token 数量不一致
  ↓
DCP attention 做 all_gather(query, dim=1) 跨组收集 query
  ↓
各 rank 的 query tensor 的 batch 维度不一致
  ↓
⚠️ HCCL defect: "different tensor shape" + "fake wait"
  ↓
Python 以为 all_gather 完成，但数据未完全同步
  ↓
下游算子拿到脏数据 → 精度系统性下降
  ↓
GSM8K accuracy = 88.78% (baseline 95%)
```

### 关键代码位置

| 文件 | 行 | 操作 |
|:---|:---:|:---|
| `vllm_ascend/attention/context_parallel/attention_cp.py` | 559 | `get_dcp_group().all_gather(query.contiguous(), 1)` — decode 阶段 all_gather |
| `vllm_ascend/attention/context_parallel/attention_cp.py` | 683 | `get_dcp_group().all_gather(prefill_query, 1)` — prefill 阶段 all_gather |
| `vllm/vllm/distributed/parallel_state.py` | 523 | `GroupCoordinator.all_gather()` — 标准 all_gather，不处理变长 tensor |
| `vllm/vllm/distributed/parallel_state.py` | 544 | `GroupCoordinator.all_gatherv()` — 支持变长 tensor 的版本（但未使用） |

---

## 代码变更分析（5/20 → 5/22 的 38 个 commit）

两个流水线之间共有 **38 个 commit**，均在 `vllm-ascend main` 分支上（`eb7e9b0f` → `68a4db55`）。

### 最高嫌疑 commit

| Commit | 说明 | 改动范围 | 嫌疑度 |
|:---|:---|:---|:---:|
| `958daf83` | env vars 迁移到 AscendConfig | `moe_comm_method.py`(694行)、`ascend_forward_context.py`、`ascend_config.py` 等 | ⭐⭐⭐⭐⭐ |
| `7bce23cc` | DeepSeekV4 支持 | 2487 行，含 `moe_comm_method.py`、`prepare_finalize.py` 等 | ⭐⭐⭐ |
| `d7cc6652` | NPUIR 升级 UB overflow 修复 | 1 行，但描述了 NPU 环境变更背景 | ⭐⭐ |
| `de00758e` | DFlash FULL_DESCODE_ONLY 精度修复 | `attention_v1.py` | ⭐⭐ |

### Git bisect 计划

```text
eb7e9b0f  ← ✅ 成功 (5/20, GSM8K 92.80%)
    │  38 commits
68a4db55  ← ❌ 首次失败 (5/22, GSM8K 90.52%)
```

最多 log₂(38) ≈ 6 次运行，二分法定位到具体 commit。

---

## 其他线索

| 线索 | 影响 |
|:---|:---|
| Sampling 参数被 `generation_config.json` 覆盖 (temperature=0.6) | 🟡 基线同参数，影响低 |
| Torch compile 缓存失效（16 ranks 全部重新编译） | 🟡 环境差异，影响低 |
| HCCL_OP_EXPANSION_MODE=AIV | 🟡 环境配置，可调整 |
| DP_Coordinator exit code None | ⚪ shutdown 假阳性，P2 可忽略 |

---

## 参考源文件

- `vllm-ascend/tests/e2e/nightly/multi_node/internal_dp/config/Qwen3-235B-A22B.yaml` — 测试配置
- `vllm-ascend/tests/e2e/nightly/multi_node/scripts/test_multi_node.py` — 测试入口
- `vllm-ascend/vllm_ascend/ascend_forward_context.py` — `select_moe_comm_method` MoE 通信方式选择
- `vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py` — MoE 通信实现
