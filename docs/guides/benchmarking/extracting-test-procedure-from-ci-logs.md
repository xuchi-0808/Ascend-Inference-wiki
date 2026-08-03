---
date: 2026-06-04
tags:
  - CI
  - 测试
  - nightly
  - AISBench
---

# 从 nightly CI 流水线日志提取测试拉起方式

## 背景

vllm-ascend 的 nightly 测试在 K8s 环境下通过 LWS 自动编排。当需要在裸机环境手动复现时，**不需要问别人**——所有信息都在 CI 日志里。

本文档教你从一行日志都不跑的情况下，从 GitHub Actions 的 CI 日志中提取出完整的手动拉起命令。

---

## 第一步：找到日志入口

1. 打开 nightly CI 的 GitHub Actions 页面：<https://github.com/vllm-project/vllm-ascend/actions>
2. 找到对应日期的 workflow（如 `Nightly-A3`）
3. 点击进入，找到 `double-node` 或 `multi-node` 相关的 job
4. 点击 **Job summary** 或展开原始日志

---

## 第二步：提取启动命令

在日志中搜索 **`Starting server with command`**：

```log
[2026-05-31 19:01:12] [INFO] Starting server with command: vllm serve Qwen/Qwen3-235B-A22B ...
```

这行日志是测试框架打印出来的完整 `vllm serve` 命令。**直接复制就是你要的。**

### Node 0 的日志

搜索 `Starting server with command`，找到不含 `--headless` 的那行。示例：

```log
vllm serve Qwen/Qwen3-235B-A22B
  --host 0.0.0.0
  --port 8080
  --data-parallel-size 4
  --data-parallel-size-local 2
  --data-parallel-address 10.0.0.135
  --data-parallel-rpc-port 13389
  --tensor-parallel-size 8
  --seed 1024
  --enable-expert-parallel
  ...
```

### Node 1 的日志

找到含 `--headless` 的那行。示例：

```log
vllm serve Qwen/Qwen3-235B-A22B
  --headless
  --data-parallel-size 4
  --data-parallel-size-local 2
  --data-parallel-start-rank 2
  --data-parallel-address 10.0.0.135
  ...
```

---

## 第三步：提取环境变量

### 3a. 从 yaml 配置中提取

在日志中搜索 **`Loading config yaml`**，找到配置文件路径。然后在代码仓库中找该 yaml 文件的 `env_common:` 段。

如果日志中没有输出 yaml 内容，可以搜索 CI 的 Inputs 段（一般在日志开头 30 行内）：

```log
##[group] Inputs
  config_file_path: Qwen3-235B-A22B.yaml
  size: 2
  ...
```

根据 `config_file_path` 的值，在代码仓库中找到对应 yaml，读取 `env_common:` 下的变量。

### 3b. 从运行时日志中提取 vllm serve 的 env

搜索 **`Node 0 envs:`** 或 **`Node 1 envs:`**。示例：

```log
[INFO] Node 0 envs: {
  'HCCL_IF_IP': '90.90.97.36',
  'HCCL_SOCKET_IFNAME': 'enp194s0f0',
  'MASTER_IP': '90.90.97.36',
  ...
}
```

这行日志是 `DistEnvBuilder.build()` 的最终输出——**包含了所有 16 个环境变量的实际值**。直接照抄。

---

## 第四步：提取 benchmark 配置

搜索 **`Benchmark config`** 或 yaml 中的 `benchmarks:` 段：

```yaml
benchmarks:
  acc:
    case_type: accuracy
    dataset_path: vllm-ascend/gsm8k
    request_conf: vllm_api_general_chat
    dataset_conf: gsm8k/gsm8k_gen_0_shot_cot_chat_prompt
    baseline: 95
    threshold: 3
```

在日志中也可以搜索 **`aisbench case failed`** 来找到 benchmark 参数，失败时日志会打出完整的 benchmark config：

```log
[ERROR] The following aisbench case failed: {
  'case_type': 'accuracy',
  'dataset_path': 'vllm-ascend/gsm8k',
  'baseline': 95,
  ...
}
```

---

## 第五步：提取 commit 信息

日志开头有 CI 使用的代码版本：

```log
Uses: vllm-project/vllm-ascend/.github/workflows/...@refs/heads/main (68a4db55)
  vllm_ascend_ref: 68a4db5554475d8e413f13d84016b86f5d2c18b7
  image: swr.cn-southwest-2...vllm-ascend:nightly-ci-main-a3
```

| 信息 | 位置 | 用途 |
|------|------|------|
| `vllm_ascend_ref` | 日志前 30 行 | vllm-ascend 代码版本 |
| `image` | 日志前 30 行 | 容器镜像 |
| `soc_version` | 日志前 30 行 | 芯片型号（a2 / a3） |
| `size` | 日志前 30 行 | 节点数 |

---

## 完整示例：从日志到命令

### CI 日志中的原始信息

```log
# 第 20 行
vllm_ascend_ref: 68a4db5554475d8e413f13d84016b86f5d2c18b7

# 第 570 行
[INFO] Loading config yaml: tests/e2e/nightly/multi_node/config/Qwen3-235B-A22B.yaml

# 第 600 行
[INFO] Starting server with command: vllm serve Qwen/Qwen3-235B-A22B
  --host 0.0.0.0 --port 8080
  --data-parallel-size 4 --data-parallel-size-local 2
  --tensor-parallel-size 8 --seed 1024
  --enable-expert-parallel
  ...

# 第 606 行
[INFO] Node 0 envs: { 'HCCL_IF_IP': '90.90.97.36', 'MASTER_IP': '90.90.97.36', ... }
```

### 提取后的手动拉起命令

```bash
# 1. 切代码
cd vllm-ascend && git checkout 68a4db55 && pip install -e .

# 2. export 环境变量（从 Node 0 envs 日志抄）
export HCCL_IF_IP="90.90.97.36"
export MASTER_IP="90.90.97.36"
export HCCL_SOCKET_IFNAME="enp194s0f0"
# ... 其余变量同理

# 3. 跑 vllm serve（从 Starting server with command 日志抄）
vllm serve Qwen/Qwen3-235B-A22B --host 0.0.0.0 --port 8080 \
  --data-parallel-size 4 --tensor-parallel-size 8 --seed 1024 ...
```

---

## 附录：关键搜索词速查

| 想找什么 | 搜什么 |
|---------|--------|
| vllm serve 完整命令 | `Starting server with command` |
| 所有环境变量的最终值 | `Node 0 envs:` / `Node 1 envs:` |
| yaml 配置文件路径 | `Loading config yaml` |
| benchmark 参数 | `benchmark cases` / `aisbench case` |
| CI 用的代码版本 | `vllm_ascend_ref` |
| 容器镜像 | `image:` |
| 芯片型号 | `soc_version` |
| HCCL 相关警告 | `ProcessGroupHCCL` |
| benchmark 最终精度 | `accuracy` |
| 测试是否通过 | `FAILED` / `Passed` / `test_multi_node` |
