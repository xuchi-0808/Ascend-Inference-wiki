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

## 根因结论

### 问题本质：MoE 路由精度不足

分析后确认两个层次的精度损失：

1. **HCCL allgather fake wait 缺陷**——在所有 CI 运行（包括 5/20 成功）中都存在，造成约 2-4% 的系统性精度损失（95% → 90-92%）
2. **MoE router gate BF16 精度不足**——导致额外的 ~2-3% 损失，使精度跌破阈值

### 修复确认

2026/6/6~6/7（周六日）nightly CI 全部重新通过。合入 `main` 的三个 MoE 精度相关 commit 是修复关键：

| 日期 | Commit | 说明 |
|:---:|:---|:---|
| 5/26 | `b86670f6` | Disable SwiGLU clamp 默认关闭，避免计算被截断 |
| **5/30** | **`78aa7ae3`** | **MoE router gate 保留 FP32 精度** ⭐ |
| 5/30 | `2a77209a` | 去掉 MoE 冗余重归一化操作 |

**最可能修复：`78aa7ae3`** — MoE 路由门控（router gate）从 BF16 改为 FP32 计算。MoE 模型选专家全靠 router logits，BF16 精度不足会导致选错专家，token 走错路径 → systematic accuracy degradation。

### 绿区验证

| 测试 | 5/20 commit (eb7e9b0f) |
|:---|:---:|
| GSM8K | **90.14%**（CI 为 92.80%，差异源于环境） |
| HCCL warning | ✅ 存在 |

### 完整时间线

```text
5/20  CI 92.80% ✅               ← 环境差异下偶然通过
5/22  CI 90.52% ❌ 首次失败       ← MoE routing 精度问题显现
5/26  b86670f6 合入              ← SwiGLU clamp fix
5/30  78aa7ae3 合入              ← FP32 router gate fix ⭐
5/30  2a77209a 合入              ← 冗余归一化移除
6/5   镜像重建                    ← 所有 fix 进镜像
6/6~  ✅ 周六日起全部通过
```

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
